"""Publish a strict train-only curriculum sidecar for a newly verified CP bank.

This command never retroactively certifies old models or edits a bank entry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from custom_trainers.onlinecp_curriculum_contract import (
    FORMAT, file_sha256, verify_curriculum_bank_contract,
)
from custom_trainers.onlinecp_curriculum_policy import (
    validate_curriculum_config, curriculum_config_sha256, eligible_candidate_indices,
)
from hiercp.contracts import require_current_checkpoint, validate_nested_cohorts
from tools import online_cp_benchmark as online


def publish(layout, outer_fold, dataset_id, curriculum_path):
    import os
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
    if output.exists():
        raise FileExistsError(f"Contract already exists; no artifacts changed: {output}")
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
    # Exclusive creation: no --overwrite and no fabricated replacement contract.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, allow_nan=False)
        handle.write("\n")
    previous = os.environ.get("nnUNet_preprocessed")
    try:
        os.environ["nnUNet_preprocessed"] = str(layout.preprocessed)
        verify_curriculum_bank_contract(
            bank / "index.json", curriculum_sha256=contract["curriculum_sha256"],
            expected_candidate_count=config["candidate_count"],
            dataset_name=index["dataset_name"], nnunet_fold=0,
            contract_filename=contract_filename)
    finally:
        if previous is None:
            os.environ.pop("nnUNet_preprocessed", None)
        else:
            os.environ["nnUNet_preprocessed"] = previous
    print(f"[OK] verified train-only curriculum contract: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root")
    parser.add_argument("--medical-root")
    parser.add_argument("--paired-root", required=True)
    parser.add_argument("--online-root", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--curriculum-config", required=True)
    args = parser.parse_args()
    publish(online.make_layout(args), args.outer_fold, args.dataset_id, args.curriculum_config)


if __name__ == "__main__":
    main()
