"""Opt-in, provenance-preserving reuse of a completed Basic control.

This never resumes or rewrites an old checkpoint. Source admission runs before
GNN training, and complete ordered bank equivalence is required before new Full
training. Failure is explicit; it never launches a replacement Basic job.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid


SOURCE_FORMAT = "feedback_basic_source_admission_v1"
REUSE_FORMAT = "feedback_basic_reuse_v1"


def _launch():
    from tools import run_feedback_experiment
    return run_feedback_experiment


def _paths(plan):
    root = Path(plan["run_root"]).resolve()
    source = plan.get("basic_source_root")
    if source is None or plan.get("recovery_source_root") or plan.get("upgrade_source_root"):
        raise ValueError("Basic reuse requires an explicitly configured NEW fresh experiment")
    source = Path(source).resolve()
    work = Path(plan["project_root"]).resolve() / "work"
    if (root == work or source == work or not root.is_relative_to(work)
            or not source.is_relative_to(work) or root == source
            or root.is_relative_to(source) or source.is_relative_to(root)):
        raise ValueError("Basic reuse must preserve a distinct, non-overlapping original experiment")
    return root, source, root / "basic_reuse/source.json", root / "basic_reuse/receipt.json"


def inspect_source(plan, *, target_native_package=None):
    from tools.basic_reuse_provenance import inspect_basic_source
    _, source, _, _ = _paths(plan)
    proof = inspect_basic_source(source, plan, require_complete=True,
                                 target_native_package=target_native_package)
    required = {"format", "source_root", "checkpoint", "bank_identity", "runtime_inventory",
                "training_journal_sha256", "bank_binding", "training_contract",
                "native_equivalence", "source_files"}
    if (not isinstance(proof, dict) or set(proof) != required
            or proof["format"] != "hiercp_basic_source_provenance_v1"
            or proof["source_root"] != str(source)):
        raise ValueError("Incomplete original Basic provenance; no reuse was authorized")
    return proof


def _seal(value):
    return {**value, "receipt_sha256": _launch()._json_sha256(value)}


def _read_receipt(root, path):
    launch = _launch()
    launch._bound_files(root, [path])
    value = launch._read_json(path)
    if not isinstance(value, dict) or "receipt_sha256" not in value:
        raise ValueError(f"Malformed Basic reuse receipt: {path}")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value["receipt_sha256"] != launch._json_sha256(unsigned):
        raise ValueError(f"Basic reuse receipt checksum changed: {path}")
    return unsigned


def _publish(root, path, value):
    """Verify first, then publish without clobbering another attempt's result."""
    launch = _launch()
    if path.exists() or path.is_symlink():
        if _read_receipt(root, path) != value:
            raise ValueError(f"Different existing Basic reuse receipt preserved: {path}")
        return
    if (not path.resolve().is_relative_to(root)
            or any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(root))):
        raise ValueError("Unsafe Basic reuse publication path")
    staging_dir = root / "basic_reuse/publication_attempts"
    if not staging_dir.resolve().is_relative_to(root) or staging_dir.is_symlink():
        raise ValueError("Unsafe Basic reuse publication staging")
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = staging_dir / (uuid.uuid4().hex + ".json")
    with staged.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_seal(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if _read_receipt(root, staged) != value:
        raise ValueError("Staged Basic reuse receipt did not verify; attempt preserved")
    try:
        os.link(staged, path)
    except FileExistsError:
        if _read_receipt(root, path) != value:
            raise ValueError("Concurrent different Basic reuse publication preserved")
    if _read_receipt(root, path) != value:
        raise ValueError("Published Basic reuse receipt changed")


def admit_source(plan):
    launch = _launch()
    root, _, path, _ = _paths(plan)
    proof = inspect_source(plan)
    value = {"format": SOURCE_FORMAT, "plan_sha256": launch._json_sha256(plan),
             "source": proof}
    _publish(root, path, value)
    return value


def verify_source(plan):
    launch = _launch()
    root, _, path, _ = _paths(plan)
    value = _read_receipt(root, path)
    if (set(value) != {"format", "plan_sha256", "source"}
            or value["format"] != SOURCE_FORMAT
            or value["plan_sha256"] != launch._json_sha256(plan)
            or value["source"] != inspect_source(plan)):
        raise ValueError("Originally admitted Basic source or requested training conditions changed")
    return value["source"]


def _target_bank(plan):
    from custom_trainers.onlinecp_curriculum_contract import verify_curriculum_bank_contract
    from custom_trainers.onlinecp_feedback_policy import validate_feedback_config, feedback_config_sha256
    from tools.online_cp_benchmark import _require_source_mapping_policy
    launch = _launch()
    root, _, _, _ = _paths(plan)
    bank = root / f"online/folds/fold_{plan['outer_fold']}/bank/index.json"
    launch._bound_files(root, [bank, bank.parent / "config.json", bank.parent / "feedback_contract.json"])
    index = launch._read_json(bank)
    expected_dataset = f"Dataset{plan['dataset_id']:03d}_LiverOnlineCP_OF{plan['outer_fold']}"
    if index.get("dataset_name") != expected_dataset or index.get("dataset_id") != plan["dataset_id"]:
        raise ValueError("NEW bank dataset differs from requested Basic comparison")
    config = validate_feedback_config(launch._read_json(plan["project_root"] / "config/online_cp_feedback.json"))
    identity = verify_curriculum_bank_contract(
        bank, curriculum_sha256=feedback_config_sha256(config), expected_candidate_count=128,
        dataset_name=expected_dataset, nnunet_fold=0, contract_filename="feedback_contract.json",
        preprocessed_root=plan["env_updates"]["nnUNet_preprocessed"])
    _require_source_mapping_policy(index, launch._read_json(bank.parent / "config.json"))
    binding = {"format": "basic_bank_integrity_binding_v1",
               "index_sha256": launch._file_sha256(bank),
               "config_sha256": launch._file_sha256(bank.parent / "config.json"),
               "feedback_contract_sha256": launch._file_sha256(bank.parent / "feedback_contract.json"),
               "original_identity_sha256": launch._json_sha256(identity)}
    return bank, identity, binding


def _verify_native_training_inputs(plan, source, target_identity):
    """CP equality alone cannot certify the non-CP segmentation training data."""
    from tools import online_cp_benchmark as online
    launch = _launch()
    root, _, _, _ = _paths(plan)
    launch._verify_online_artifacts(plan, "plan")
    original = source["native_equivalence"].get("source_preprocessing")
    required = {"marker", "raw_input_contract", "native_outputs", "raw_dataset_contract"}
    if not isinstance(original, dict) or not required.issubset(original):
        raise ValueError("Original complete native preprocessing proof is required, not only CP bank equality")
    marker_record = target_identity["files"]["preprocess_marker"]
    marker_path = Path(marker_record["path"])
    launch._bound_files(root, [marker_path])
    if launch._file_sha256(marker_path) != marker_record["sha256"]:
        raise ValueError("NEW native preprocessing marker changed")
    marker = launch._read_json(marker_path)
    if (not isinstance(marker.get("outputs"), dict)
            or marker["outputs"] != original["native_outputs"]):
        raise ValueError("Basic reuse requires identical validated native data/seg/properties/plans/split outputs")
    raw_path = (Path(plan["env_updates"]["nnUNet_raw"]) / target_identity["dataset_name"]
                / online.RAW_MARKER_NAME)
    launch._bound_files(root, [raw_path])
    raw = launch._read_json(raw_path)
    native_input = marker["input_contract"]
    if (launch._file_sha256(raw_path) != native_input["raw_marker_sha256"]
            or online.value_sha256(raw) != native_input["raw_contract_sha256"]):
        raise ValueError("NEW native training data no longer matches its certified raw cohort")
    original_raw = original["raw_dataset_contract"]
    # Native output hashes cover all actual training data/segmentation/properties
    # and the complete plans/dataset/split. These additional fields bind the
    # original full CT/label cohort, not just liver-cropped CP bank patches.
    fields = ("dataset_name", "train_ids", "val_ids", "source_cases")
    for key in fields:
        if key not in raw or key not in original_raw or raw[key] != original_raw[key]:
            raise ValueError(f"Original and NEW native raw training provenance differs: {key}")
    return {"format": "basic_native_training_inputs_equivalence_v1",
            "source_marker": original["marker"], "target_marker": marker_record,
            "source_raw_contract_sha256": online.value_sha256(original_raw),
            "target_raw_contract_sha256": online.value_sha256(raw),
            "raw_cohort_fields": list(fields),
            "raw_cohort_sha256": launch._json_sha256({key: raw[key] for key in fields}),
            "native_outputs_sha256": launch._json_sha256(marker["outputs"])}


def _certification(plan):
    from tools.basic_bank_equivalence import compare_basic_banks
    launch = _launch()
    root, source_root, source_path, _ = _paths(plan)
    source = verify_source(plan)
    bank, identity, binding = _target_bank(plan)
    native_inputs = _verify_native_training_inputs(plan, source, identity)
    source_bank = source_root / f"online/folds/fold_{plan['outer_fold']}/bank/index.json"
    comparison = compare_basic_banks(source_bank, bank,
        source_proof=source["bank_binding"], target_proof=binding)
    if not isinstance(comparison, dict) or comparison.get("equal") is not True:
        raise ValueError("Basic bank input equivalence was not established; Full training not authorized")
    # The bank comparison is only one sub-proof: inspect_source separately
    # verifies completed native training, requested scientific settings, and
    # supported historical/current runtime equivalence. Never approve on bank
    # equality alone or substitute the new bank identity in the old checkpoint.
    return {"format": REUSE_FORMAT, "plan_sha256": launch._json_sha256(plan),
            "source_admission_sha256": launch._file_sha256(source_path),
            "source": source, "target_bank_identity": identity,
            "target_runtime_inventory": launch._runtime_inventory(plan["package_destination"]),
            "native_training_inputs": native_inputs,
            "bank_equivalence": comparison,
            "checkpoint_rewritten": False, "basic_training_launched": False}


def certify(plan):
    root, _, _, path = _paths(plan)
    value = _certification(plan)
    _publish(root, path, value)
    return value


def verify_reuse(plan):
    root, _, _, path = _paths(plan)
    value = _read_receipt(root, path)
    if value != _certification(plan):
        raise ValueError("Basic reuse proof, original source, or target Basic dependencies changed")
    return value


def basic_origin(plan):
    launch = _launch()
    root, _, _, path = _paths(plan)
    source = verify_reuse(plan)["source"]
    return {"format": "hiercp_verified_basic_origin_v1",
            "experiment_root": source["source_root"], "checkpoint": source["checkpoint"],
            "bank_identity": source["bank_identity"], "runtime_inventory": source["runtime_inventory"],
            "training_journal_sha256": source["training_journal_sha256"],
            "reuse_receipt": {"path": str(path), "sha256": launch._file_sha256(path)}}


def _worker_plan(root, stage):
    launch = _launch()
    root = Path(root).absolute()
    if root.is_symlink() or root != root.resolve():
        raise ValueError("Basic reuse worker requires the canonical experiment root")
    launch._bound_files(root, [root / "launch_plan.json", root / "execution_journal.json"])
    saved = launch._read_json(root / "launch_plan.json")
    journal = launch._read_json(root / "execution_journal.json")
    unsigned = {key: value for key, value in journal.items() if key != "journal_sha256"}
    if (journal.get("format") != "feedback_fresh_execution_v1"
            or journal.get("journal_sha256") != launch._json_sha256(unsigned)
            or journal.get("plan_sha256") != launch._json_sha256(saved)
            or not journal.get("stages") or journal["stages"][-1].get("name") != stage
            or journal["stages"][-1].get("status") != "running"
            or journal["stages"][-1].get("execution_backend") != "native_receipted"):
        raise ValueError("Basic reuse must run inside its recorded native fresh stage")
    plan = dict(saved)
    for key in ("project_root", "medical_root", "run_root", "train_config", "package_destination",
                "recovery_source_root", "preprocessing_source_root", "basic_source_root"):
        if plan.get(key) is not None:
            plan[key] = Path(plan[key])
    if plan["run_root"] != root:
        raise ValueError("Basic reuse worker root differs from its recorded plan")
    _paths(plan)
    # A running JSON row alone is not permission to publish. Bind this process
    # to the launcher-authorized worker, including Windows venv redirectors.
    import socket
    import psutil
    from tools import feedback_stage_execution as execution
    row = journal["stages"][-1]
    attempt_id = row.get("attempt_id")
    if (not isinstance(attempt_id, str) or len(attempt_id) != 32
            or any(c not in "0123456789abcdef" for c in attempt_id)):
        raise ValueError("Basic reuse worker lacks its exact native attempt")
    attempt = root / "execution_attempts" / attempt_id
    launch._bound_files(root, [attempt / filename for filename in
                             ("attempt.json", "child_started.json", "permit.json")])
    spec = launch._read_json(attempt / "attempt.json")
    started = launch._read_json(attempt / "child_started.json")
    permit = launch._read_json(attempt / "permit.json")
    if (spec.get("format") != "feedback_child_attempt_v1" or spec.get("attempt_id") != attempt_id
            or spec.get("argv") != row.get("argv") or spec.get("cwd") != str(plan["project_root"])
            or spec.get("parent") != row.get("owner")
            or not isinstance(started, dict) or set(started) != {"host", "pid", "process_started"}
            or started["host"] != socket.gethostname() or type(started["pid"]) is not int
            or permit != {"attempt_sha256": execution._sha(spec), "child_pid": started["pid"]}):
        raise ValueError("Basic reuse worker authorization changed")
    process = psutil.Process(os.getpid())
    while process is not None and process.pid != started["pid"]:
        process = process.parent()
    if process is None or process.create_time() != started["process_started"]:
        raise ValueError("Basic reuse publisher is not the authorized native worker's descendant")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("source", "certify"))
    parser.add_argument("--experiment-root", required=True)
    args = parser.parse_args()
    stage = "basic_source_preflight" if args.stage == "source" else "basic_reuse"
    plan = _worker_plan(args.experiment_root, stage)
    value = admit_source(plan) if args.stage == "source" else certify(plan)
    print(json.dumps({"format": value["format"], "source": str(plan["basic_source_root"]),
                      "stage": stage, "basic_training_launched": False,
                      "checkpoint_rewritten": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
