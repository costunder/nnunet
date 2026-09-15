"""Publish a strict train-only curriculum sidecar for a newly verified CP bank.

This command never retroactively certifies old models or edits a bank entry.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import uuid
from pathlib import Path

import numpy as np

from custom_trainers.onlinecp_curriculum_contract import (
    FORMAT, file_sha256, read_curriculum_contract_payload, value_sha256,
    verify_curriculum_contract_payload,
)
from custom_trainers.onlinecp_curriculum_policy import (
    validate_curriculum_config, curriculum_config_sha256, eligible_candidate_indices,
)
from hiercp.contracts import require_current_checkpoint, validate_nested_cohorts
from tools import online_cp_benchmark as online


def _sync_directory(path):
    # Windows has no portable directory fsync. File fsync and atomic hard-link
    # publication still apply; the receipt must not claim power-loss durability.
    if os.name == "nt":
        return "file_fsync_only; directory_fsync_unavailable_on_windows"
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return "file_and_directory_fsync; filesystem_durability_not_independently_tested"


def _write_exclusive_json(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return _sync_directory(path.parent)


def _publish_verified_contract(output, contract, verify):
    """Preserve attempts and publish exactly one fully validated immutable file.

    Concurrent writers use separate UUID generations. The hard-link is an atomic
    no-clobber install on the same filesystem, not a replacement of user output.
    No lock is reclaimed and no old attempt is edited/deleted, even after a crash.
    """
    output = Path(output)
    expected = value_sha256(contract)

    def reuse():
        existing = read_curriculum_contract_payload(output)
        if value_sha256(existing) != expected:
            raise ValueError(f"Existing contract has different inputs; all artifacts preserved: {output}")
        verified = verify(existing)
        # Guard the final content across the potentially long live-data audit.
        if value_sha256(read_curriculum_contract_payload(output)) != expected:
            raise ValueError("Published contract changed during verification; all files preserved")
        return verified

    if output.exists() or output.is_symlink():
        reuse()
        print(f"[VERIFIED REUSE] unchanged contract: {output}")
        return output
    history = output.parent / "contract_publication_history"
    if history.is_symlink():
        raise ValueError("Contract publication history must not be a symlink")
    history.mkdir(exist_ok=True)
    attempt_id = uuid.uuid4().hex
    attempt = history / attempt_id
    attempt.mkdir(exist_ok=False)
    _sync_directory(history)
    _write_exclusive_json(attempt / "attempt.json", {
        "format": "onlinecp_contract_publication_attempt_v1", "attempt_id": attempt_id,
        "host": socket.gethostname(), "pid": os.getpid(), "final_path": str(output.resolve()),
        "contract_sha256": expected, "input_files": contract["files"],
        "entry_sha256": contract["entry_sha256"]})
    staged = attempt / "contract.json"
    durability = _write_exclusive_json(staged, contract)
    staged_payload = read_curriculum_contract_payload(staged)
    if value_sha256(staged_payload) != expected:
        raise ValueError("Staged contract differs from its attempt; all files preserved")
    verify(staged_payload)
    _write_exclusive_json(attempt / "validated.json", {
        "format": "onlinecp_contract_validated_v1", "contract_sha256": expected,
        "staged_file_sha256": file_sha256(staged), "durability": durability})
    # Verify dependencies once more immediately before install. A later mutation
    # still fails the final/runtime verifier; it is never silently certified.
    verify(staged_payload)
    try:
        os.link(staged, output)
        outcome = "published"
    except FileExistsError:
        outcome = "concurrent_verified_reuse"
    _sync_directory(output.parent)
    reuse()
    _write_exclusive_json(attempt / "publication.json", {
        "format": "onlinecp_contract_publication_v1", "attempt_id": attempt_id,
        "outcome": outcome, "contract_sha256": expected,
        "final_file_sha256": file_sha256(output), "durability": durability})
    return output


def publish(layout, outer_fold, dataset_id, curriculum_path):
    import torch
    from hiercp.tensor import load_checkpoint

    raw_config = online.load_json(curriculum_path)
    feedback = raw_config.get("format") == "onlinecp_segmentation_feedback_v1"
    if feedback:
        from custom_trainers.onlinecp_feedback_policy import (
            validate_feedback_config, feedback_config_sha256, quality_eligible_indices,
        )
        config = validate_feedback_config(raw_config)
        config_hash = feedback_config_sha256(config)
        contract_filename = "feedback_contract.json"
    else:
        config = validate_curriculum_config(raw_config)
        config_hash = curriculum_config_sha256(config)
        contract_filename = "curriculum_contract.json"
    train_cfg = online.load_json(layout.train_config)
    nn_cfg = online.load_json(layout.nnunet_config)
    bank = layout.bank(outer_fold)
    output = bank / contract_filename
    bank_index = online.load_json(bank / "index.json")
    if bank_index.get("paste_contract") is not None and not feedback:
        raise ValueError(
            "Raw-target banks require the Full/Basic segmentation-feedback policy; "
            "a legacy rank-only curriculum contract was not written")
    online._verified_bank_identity(layout, outer_fold, train_cfg, nn_cfg, dataset_id)
    split = online.outer_split(layout, outer_fold)
    gnn = layout.gnn(outer_fold)
    inner = online._verified_gnn_split(layout, outer_fold)
    checkpoint = load_checkpoint(gnn / "model.pt", torch.device("cpu"))
    require_current_checkpoint(checkpoint)
    validate_nested_cohorts(split["train"], split["val"], inner["train"], inner["val"],
                            checkpoint["prototype_training_cases"])
    index = online.load_json(bank / "index.json")
    if index["candidate_count"] != config["candidate_count"] or index["cp_probability"] != config["cp_probability"]:
        raise ValueError("Curriculum and bank candidate/event contracts differ")
    entries = {}
    for case_id, names in index["entries_by_case"].items():
        if case_id not in split["train"]:
            raise ValueError(f"Held-out patient in CP bank: {case_id}")
        for name in names:
            path = (bank / name).resolve()
            if not path.is_relative_to(bank.resolve()):
                raise ValueError(f"Bank path escapes root: {name}")
            with np.load(path, allow_pickle=False) as payload:
                scores = np.asarray(payload["scores"])
                if feedback:
                    quality_eligible_indices(scores, config)
                else:
                    for stage in config["stages"]:
                        eligible_candidate_indices(scores, config, stage["start_epoch"])
            entries[name] = file_sha256(path)
    pre = online.preprocessed_dataset_dir(layout, dataset_id, outer_fold)
    paths = {"index": bank / "index.json", "config": bank / "config.json",
             "manifest": bank / "manifest.csv", "complete": bank / "complete.json",
             "gnn_checkpoint": gnn / "model.pt", "gnn_split": gnn / "split.json",
             "prototype": gnn / "prototype.pt", "graph_complete": gnn / "graphs" / "complete.json",
             "outer_splits": layout.outer_splits, "preprocessed_split": pre / "splits_final.json",
             "preprocess_marker": pre / online.PREPROCESS_MARKER_NAME,
             "plans": pre / f"{nn_cfg['dataset']['plans']}.json", "dataset": pre / "dataset.json",
             "train_config": layout.train_config, "nnunet_config": layout.nnunet_config}
    marker = online.load_json(paths["preprocess_marker"])
    if marker.get("input_contract", {}).get("planning_cohort") != "outer_train_only_v1":
        raise ValueError("Rebuild fingerprint/plans on training patients in a NEW experiment workspace")
    contract = {"format": FORMAT, "architecture_version": checkpoint["architecture_version"],
                "geometry_contract": checkpoint["geometry_contract"], "outer_fold": outer_fold,
                "nnunet_fold": 0, "dataset_name": index["dataset_name"],
                "candidate_count": config["candidate_count"],
                "curriculum_sha256": config_hash,
                "train_case_ids": split["train"], "validation_case_ids": split["val"],
                "gnn_train_case_ids": inner["train"], "gnn_validation_case_ids": inner["val"],
                "prototype_training_case_ids": checkpoint["prototype_training_cases"],
                "entry_sha256": entries,
                "files": {name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                          for name, path in paths.items()}}
    def verify(payload):
        return verify_curriculum_contract_payload(
            payload, bank / "index.json", curriculum_sha256=contract["curriculum_sha256"],
            expected_candidate_count=config["candidate_count"],
            dataset_name=index["dataset_name"], nnunet_fold=0,
            preprocessed_root=layout.preprocessed)
    _publish_verified_contract(output, contract, verify)
    print(f"[OK] verified train-only curriculum contract: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root")
    parser.add_argument("--medical-root")
    parser.add_argument("--train-config")
    parser.add_argument("--paired-root", required=True)
    parser.add_argument("--online-root", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--curriculum-config", required=True)
    args = parser.parse_args()
    publish(online.make_layout(args), args.outer_fold, args.dataset_id, args.curriculum_config)


if __name__ == "__main__":
    main()
