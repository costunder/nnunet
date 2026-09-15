"""Read-only admission of an ORIGINAL, completed Basic feedback control.

This is a recorded algorithm/training-contract proof, not a whole-environment
or bitwise replay claim. It does not compare two banks, publish a receipt, load
old Python modules, initialize a trainer, or modify an original experiment.
"""
from __future__ import annotations

from contextlib import nullcontext
import ast
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Mapping

FORMAT = "hiercp_basic_source_provenance_v1"
TRAINER = "nnUNetTrainer_250epochs_OnlineBasicCPFeedbackControl"
RESUME = "onlinecp_segmentation_feedback_resume_v1"
PREFIX = "training/nnUNetTrainer/"
HISTORICAL_HIERCP_SHA = "cbac7b9c9308f74534571630de74c6a46ed452c0b4d6cc1f33b93b8cbf30315e"

# Audited exact d904eeb -> source-content-v4 transition. The three changed
# modules add restore failure guards / pure prevalidation, Full-only region
# reuse, and strict JSON/publication verification with the new GNN architecture.
# Basic draw/paste/augment/loss/optimizer and native inference are unchanged.
# The v4 -> population-metric-v5 private-contract change is only its audited
# ARCHITECTURE literal; the exact module digest below binds that narrow change.
# This is deliberately not a filename-only or AST-size compatibility allowance.
CURRENT_CP = {
    "onlinecp_curriculum_policy.py": "bde202bc0060b47350724a219dfa573d946e58410d46453698bd07bb87ab6b1b",
    "onlinecp_curriculum_contract.py": "4bcb725c896acde1d828db978d35b7d3e8219122eedb6aa9b2fc3dcf1a040493",
    "nnUNetTrainer_OnlineCPCurriculum.py": "37936501a55fd2443e414db5083c564e556d7dee96bbf9f1e7d6d38dc98b3f77",
    "nnUNetTrainer_OnlinePairedCP.py": "ec4b3934a0e05a2810dbab69a8a0f60e06754d4853aea465a3ee735b56008ed1",
    "nnUNetTrainer_OnlinePairedCPArgmaxV3.py": "055c6c154820c8994bb9d59273ca8515c3db4ff6a4ed4064923fefe2b3f46320",
    "onlinecp_feedback_policy.py": "566357115d84fcdfc46b03706bdaa86a00c9ff9d3510cd8490268e8bc219a37c",
    "onlinecp_feedback_metrics.py": "fe0d94b5d98fe376c41b8bf6eb60d8c922b9b1d6f629fe88d5fa1f1ad108982c",
    "nnUNetTrainer_OnlineCPFeedback.py": "2ed7f15acf0aad9f9bcbc161b31b3b69b234792bb34a471c15165e958a173c8c",
    "onlinecp_raw_resampling.py": "49d1c06eed2d5f9ec535ab587e2789f7af7ef8c9d096c0706cc63a9292c318a2",
    "onlinecp_raw_bank.py": "0cf1249958de0bebece45501fc689844fb3e8a1ec1bd47f44d62a0cc790a7a6b",
}
HISTORICAL_CP = {**CURRENT_CP,
    "onlinecp_curriculum_contract.py": "feb7afacb037811212df06516d3b80b701e492b55e0d06fc5fe4d6cfe2768346",
    "nnUNetTrainer_OnlineCPCurriculum.py": "e2df88b73261ae538c2f5210402150c676485df367c20f88f085dd9093b64c51",
    "nnUNetTrainer_OnlineCPFeedback.py": "47ef130863c97207e5cce8fdfd2e0c2da8446ac66ce1df8728d454b457ce18b9",
}


def _launch():
    from tools import run_feedback_experiment
    return run_feedback_experiment


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _file(path):
    path = Path(path)
    if (not path.is_absolute() or not path.is_file()
            or any(item.is_symlink() for item in (path, *path.parents))):
        raise ValueError(f"Missing or unsafe Basic source file: {path}")
    before = path.stat()
    digest = _launch()._file_sha256(path)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"Basic source file changed while hashing: {path}")
    return {"path": str(path), "sha256": digest}


def _read(path):
    proof = _file(path)
    value = _launch()._read_json(path)
    if _file(path) != proof:
        raise ValueError(f"Basic source JSON changed while reading: {path}")
    return value


def _file_record(record, label):
    """Verify consumed path/SHA fields without discarding signed metadata."""
    if (not isinstance(record, dict) or not {"path", "sha256"}.issubset(record)
            or not isinstance(record["path"], str) or not _digest(record["sha256"])):
        raise ValueError(f"Malformed original Basic provenance file: {label}")
    actual = _file(record["path"])
    if actual["sha256"] != record["sha256"]:
        raise ValueError(f"Original Basic provenance file changed: {label}")
    return actual


def _mapping(value, label):
    if (not isinstance(value, dict) or not value
            or any(not isinstance(k, str) or not k or not _digest(v) for k, v in value.items())):
        raise ValueError(f"Missing or malformed {label} SHA256 inventory")
    return value


def _current_hiercp_sha():
    import hiercp
    root = Path(hiercp.__file__).resolve().parent
    return _sha({path.name: _file(path)["sha256"] for path in sorted(root.glob("*.py"))})


def _native_iterations(package):
    path = Path(package) / "training/nnUNetTrainer/nnUNetTrainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "nnUNetTrainer"]
    if len(classes) != 1:
        raise ValueError("Unknown native trainer definition; iteration defaults need review")
    initializers = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == "__init__"]
    if len(initializers) != 1:
        raise ValueError("Unknown native trainer initializer")
    expected = {"num_iterations_per_epoch": [], "num_val_iterations_per_epoch": []}
    for node in ast.walk(initializers[0]):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id == "self" and target.attr in expected):
                    if (node not in initializers[0].body or not isinstance(node.value, ast.Constant)
                            or type(node.value.value) is not int or node.value.value <= 0):
                        raise ValueError("Computed/conditional native iteration setting needs explicit review")
                    expected[target.attr].append(node.value.value)
    if any(len(values) != 1 for values in expected.values()):
        raise ValueError("Missing or ambiguous native iteration defaults")
    return {key: values[0] for key, values in expected.items()}


def _history_receipt(source, old, journal, native):
    """Recheck original history artifacts, never rebuild with current GNN settings."""
    kind, identity = journal["format"], journal["source_identity"]
    if not isinstance(identity, dict):
        raise ValueError("Missing original source identity")
    if kind == "feedback_fresh_execution_v1":
        if (set(identity) != {"native_package", "native_files"}
                or identity["native_files"] != native):
            raise ValueError("Original native copy differs from its source inventory")
        return {"format": kind, "source_identity": identity}
    if kind == "feedback_preparation_execution_v1":
        receipt_path, receipt = source / "recovery/complete.json", journal["preparation_receipt"]
        if (not isinstance(receipt, dict) or receipt.get("format") != "hiercp_preparation_recovery_complete_v1"
                or receipt.get("train_config") != old["train_config"] or receipt.get("source_identity") != identity
                or receipt.get("original_results_preserved") is not True or receipt.get("training_performed") is not False
                or _read(receipt_path) != receipt):
            raise ValueError("Original preparation receipt is missing or changed")
        files = _mapping(receipt.get("files"), "original preparation")
        expected = {Path(name).as_posix(): digest for name, digest in files.items()}
        if _launch()._bound_files(source, files) != expected:
            raise ValueError("Original preparation receipt artifacts changed")
        identity_root = Path(identity.get("source_root", ""))
        source_files = _mapping(identity.get("files"), "original preparation source")
        if not identity_root.is_absolute() or _launch()._bound_files(identity_root, source_files) != source_files:
            raise ValueError("Original preparation source identity artifacts changed")
    else:
        receipt_path, receipt = source / "upgrade/views.json", journal["view_receipt"]
        if (not isinstance(receipt, dict) or receipt.get("format") != "feedback_bank_upgrade_views_v1"
                or receipt.get("preprocessing_recomputed") is not False or receipt.get("quality_training_performed") is not False
                or not isinstance(receipt.get("files"), list) or not receipt["files"] or _read(receipt_path) != receipt):
            raise ValueError("Original bank-upgrade view receipt is missing or changed")
        seen = set()
        for row in receipt["files"]:
            if not isinstance(row, dict) or set(row) != {"source", "target", "mode", "bytes", "sha256"}:
                raise ValueError("Malformed historical view receipt row")
            original, target = Path(row["source"]), Path(row["target"])
            if (not target.is_absolute() or not target.is_relative_to(source) or ".." in target.parts
                    or str(target) in seen or row["mode"] not in {"copy", "hardlink", "symlink"}
                    or not _digest(row["sha256"]) or type(row["bytes"]) is not int or row["bytes"] < 0
                    or any(parent.is_symlink() for parent in target.parents)):
                raise ValueError("Unsafe original data-view row")
            seen.add(str(target))
            # Raw materialization may intentionally be a leaf symlink. Every
            # original/target byte and mode is checked; no symlink is created.
            if (not original.is_file() or not target.is_file()
                    or _launch()._file_sha256(original) != row["sha256"]
                    or _launch()._file_sha256(target) != row["sha256"]
                    or target.stat().st_size != row["bytes"]
                    or (row["mode"] == "symlink" and (not target.is_symlink() or target.resolve() != original.resolve()))
                    or (row["mode"] == "hardlink" and (target.is_symlink() or not target.samefile(original)))
                    or (row["mode"] == "copy" and (target.is_symlink() or target.samefile(original)))):
                raise ValueError("Original data-view bytes or materialization changed")
        identity_root = Path(identity.get("source_root", ""))
        files = _mapping(identity.get("files"), "bank-upgrade source")
        if not identity_root.is_absolute() or _launch()._bound_files(identity_root, files) != files:
            raise ValueError("Original bank-upgrade source provenance changed")
    return {"format": kind, "receipt": _file(receipt_path), "source_identity": identity}


def _source_preprocessing(old, identity):
    dataset = identity["dataset_name"]
    marker_path = Path(old["env_updates"]["nnUNet_preprocessed"]) / dataset / "online_cp_preprocess_complete.json"
    raw_path = Path(old["env_updates"]["nnUNet_raw"]) / dataset / "online_cp_dataset.json"
    marker, raw = _read(marker_path), _read(raw_path)
    inputs, outputs = marker.get("input_contract"), marker.get("outputs")
    raw_file = _file(raw_path)
    if (not isinstance(inputs, dict) or not isinstance(outputs, dict)
            or inputs.get("raw_marker_sha256") != raw_file["sha256"] or inputs.get("raw_contract_sha256") != _sha(raw)
            or raw.get("dataset_name") != dataset or raw.get("train_ids") != identity["train_case_ids"]
            or raw.get("val_ids") != identity["validation_case_ids"]):
        raise ValueError("Original raw/native preprocessing identity is incomplete or changed")
    return {"marker": _file(marker_path), "raw_marker": raw_file,
            "raw_input_contract": inputs, "native_outputs": outputs,
            "raw_dataset_contract": raw, "raw_contract_sha256": _sha(raw)}


def _runtime_equivalence(source, expected, target):
    from tools.feedback_bank_upgrade import _historical_runtime
    from tools.feedback_fresh_execution import native_inventory
    from custom_trainers.install_onlinecp_custom_trainers import MODULES, audit_sources
    from custom_trainers.onlinecp_curriculum_contract import ARCHITECTURE
    from hiercp.contracts import ARCHITECTURE_VERSION
    audit_sources()
    if MODULES != CURRENT_CP:
        raise ValueError("Current Basic private runtime needs a new explicit compatibility review")
    if ARCHITECTURE != ARCHITECTURE_VERSION:
        raise ValueError("Current Basic private runtime and graph architecture contracts differ")
    actual = _historical_runtime(source, expected)
    cp = {name: actual.get(PREFIX + name) for name in CURRENT_CP}
    if cp == CURRENT_CP:
        mode, architecture = "identical_private_runtime_v1", ARCHITECTURE_VERSION
    elif cp == HISTORICAL_CP:
        mode, architecture = "d904eeb_basic_training_equivalence_v1", "hiercp_conditioned_readout_v3"
    else:
        raise ValueError("Unknown historical Basic private runtime; no arbitrary compatibility whitelist")
    native = native_inventory(source)
    if native != native_inventory(target):
        raise ValueError("Original and target native nnU-Net implementation files differ")
    return actual, {"format": mode, "source_cp_modules": cp, "target_cp_modules": dict(CURRENT_CP),
                    "native_files": native, "source_architecture": architecture,
                    "bitwise_replay_claimed": False,
                    "unrecorded_external_versions": ["scipy", "batchgenerators", "batchgeneratorsv2",
                                                       "dynamic-network-architectures", "CUDA driver"]}


def _owner(value):
    return (isinstance(value, dict) and set(value) == {"host", "pid", "process_started"}
            and isinstance(value["host"], str) and bool(value["host"])
            and type(value["pid"]) is int and value["pid"] > 0
            and type(value["process_started"]) in (int, float)
            and math.isfinite(value["process_started"]) and value["process_started"] > 0)


def _native_attempt(old, row):
    """Bind the actual command and all four original producer receipts."""
    from tools.feedback_stage_execution import _attempt_path, verified_child_result
    folder = _attempt_path(old, row)
    paths = [folder / name for name in ("attempt.json", "child_started.json", "permit.json", "child_complete.json")]
    before = {str(path): _file(path)["sha256"] for path in paths}
    spec = _read(folder / "attempt.json")
    if (not isinstance(spec, dict)
            or set(spec) != {"format", "attempt_id", "argv", "cwd", "parent"}
            or spec["format"] != "feedback_child_attempt_v1"
            or spec["attempt_id"] != row["attempt_id"] or spec["argv"] != row["argv"]
            or spec["cwd"] != str(old["project_root"])
            or not _owner(row.get("owner")) or spec["parent"] != row["owner"]):
        raise ValueError("Original native attempt identity/cwd/owner/argv differs from its stage")
    terminal = verified_child_result(old, row)
    if not _owner(terminal["child"]):
        raise ValueError("Original native attempt has a malformed child identity")
    if row["status"] == "completed" and terminal["returncode"] != 0:
        raise ValueError("Completed Basic source stage has a failed native child receipt")
    if before != {str(path): _file(path)["sha256"] for path in paths}:
        raise ValueError("Original native attempt receipts changed during inspection")
    return before


def _history(source, old, journal):
    """Validate known journal shapes without revalidating unused current GNN config."""
    common = {"format", "plan_sha256", "source_identity", "runtime_inventory", "stages", "complete", "journal_sha256"}
    variants = {
        "feedback_preparation_execution_v1": {"preparation_receipt", "training_started"},
        "feedback_bank_upgrade_execution_v1": {"view_receipt"},
        "feedback_fresh_execution_v1": {"input_files", "training_started"},
    }
    kind = journal.get("format") if isinstance(journal, dict) else None
    if kind not in variants or set(journal) != common | variants[kind]:
        raise ValueError("Unsupported Basic source journal format; no legacy metadata is invented")
    payload = {k: v for k, v in journal.items() if k != "journal_sha256"}
    if (journal["journal_sha256"] != _launch()._json_sha256(payload)
            or journal["plan_sha256"] != _launch()._json_sha256(old)
            or type(journal["complete"]) is not bool
            or ("training_started" in journal and journal["training_started"] is not True)):
        raise ValueError("Basic source journal/plan checksum or training status is invalid")
    commands = old.get("commands")
    if (not isinstance(commands, list) or not commands
            or any(not isinstance(row, dict) or set(row) != {"name", "argv"}
                   or not isinstance(row["name"], str) or not row["name"]
                   or not isinstance(row["argv"], list) or not row["argv"]
                   or any(not isinstance(value, str) or not value for value in row["argv"])
                   for row in commands)):
        raise ValueError("Malformed original Basic launch commands")
    names = [row["name"] for row in commands]
    if (len(names) != len(set(names)) or names.count("train_basic") != 1 or old.get("basic_source_root")
            or not {"bank", "feedback_contract", "check_basic", "train_basic"}.issubset(names)):
        raise ValueError("Source must contain its own original Basic training, not a reused control")
    setup = (["copy_private_runtime", "install_private_trainers", "environment", "recover_preparation"]
             if kind == "feedback_preparation_execution_v1" else
             ["copy_private_runtime", "install_private_trainers", "environment", "prepare_views"]
             if kind == "feedback_bank_upgrade_execution_v1" else ["copy_private_runtime"])
    sequence = setup + [name for name in names if name not in setup]
    rows, position, basic = journal["stages"], 0, None
    fresh = kind == "feedback_fresh_execution_v1"
    if fresh:
        _mapping(journal.get("input_files"), "original fresh input")
        identity = journal["source_identity"]
        if (not isinstance(identity, dict) or set(identity) != {"native_package", "native_files"}
                or not isinstance(identity["native_package"], str)
                or not Path(identity["native_package"]).is_absolute()):
            raise ValueError("Malformed original fresh native source identity")
        _mapping(identity["native_files"], "original fresh native source")
    attempts, receipt_files, seen_attempt_ids = {}, {}, set()
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing original Basic stage history")
    for row in rows:
        if (not isinstance(row, dict) or row.get("status") not in {"completed", "failed"}
                or position >= len(sequence) or row.get("name") != sequence[position]):
            raise ValueError("Source has active, ambiguous, duplicated, or out-of-order stages")
        backend = row.get("execution_backend")
        if backend not in {None, "native_receipted"}:
            raise ValueError("DEBUG/unknown execution evidence cannot authorize Basic reuse")
        if fresh and (row.get("input_files") != journal["input_files"]
                      or (row["name"] != "copy_private_runtime" and backend != "native_receipted")):
            raise ValueError("Original fresh stage input/native execution evidence differs")
        if backend is None and row["name"] != "copy_private_runtime" and any(
                key in row for key in ("attempt_id", "owner")):
            raise ValueError("Original native stage cannot discard its execution backend")
        if backend == "native_receipted" or (fresh and row["name"] == "copy_private_runtime"):
            token = row.get("attempt_id")
            if (not isinstance(token, str) or len(token) != 32
                    or any(value not in "0123456789abcdef" for value in token)
                    or token in seen_attempt_ids or not _owner(row.get("owner"))):
                raise ValueError("Missing, malformed or reused original native attempt identity")
            seen_attempt_ids.add(token)
        if backend == "native_receipted" or (row["name"] == "train_basic" and "argv" in row):
            planned = next(command["argv"] for command in commands if command["name"] == row["name"])
            allowed = [planned]
            if attempts.get(row["name"], 0) and row["name"] in {"train_basic", "train_full"}:
                allowed.append([*planned, "--resume"])
            if row.get("argv") not in allowed:
                raise ValueError(f"Original {row['name']} actual stage argv differs from its launch plan")
        if backend == "native_receipted":
            receipt_files.update(_native_attempt(old, row))
        evidence = row.get("completion_evidence")
        if row["status"] == "completed":
            if fresh and row["name"] == "copy_private_runtime":
                expected = {"format": "feedback_native_package_copy_v1", "native_files": journal["source_identity"]["native_files"]}
                if backend is not None or evidence != expected:
                    raise ValueError("Original fresh native copy proof differs from its source inventory")
            elif row["name"] == "environment" and (fresh or evidence is not None):
                _mapping(row.get("input_files"), "original environment input")
                if evidence != {"format": "feedback_environment_preflight_v1", "input_files": row["input_files"]}:
                    raise ValueError("Original environment preflight input proof differs")
            elif (fresh or row["name"] not in setup) and (not isinstance(evidence, dict)
                    or set(evidence) != {"format", "files"} or evidence.get("format") != "feedback_stage_completion_v1"):
                raise ValueError("Original completed stage has no native artifact evidence")
            if isinstance(evidence, dict) and "files" in evidence:
                files = _mapping(evidence["files"], "source stage")
                if _launch()._bound_files(source, files) != files:
                    raise ValueError(f"Original completed stage files changed: {row['name']}")
            if row["name"] == "train_basic":
                if not isinstance(evidence, dict) or set(evidence) != {"format", "files"} or evidence["format"] != "feedback_stage_completion_v1":
                    raise ValueError("Completed Basic needs its native checkpoint completion proof")
                _mapping(row.get("input_files"), "Basic training input")
                basic = row
            position += 1
        attempts[row["name"]] = attempts.get(row["name"], 0) + 1
    if basic is None or (journal["complete"] and position != len(sequence)):
        raise ValueError("Source Basic training is not verifiably complete")
    return basic, next(row["argv"] for row in commands if row["name"] == "train_basic"), receipt_files


def _verify_bank(bank, old, policy, architecture):
    """Same native bank guards, with an explicit recognized historical architecture.

    No contract field is rewritten to v4. The original native identity is
    returned byte-semantically unchanged and later compared to the checkpoint.
    """
    from custom_trainers import onlinecp_curriculum_contract as contract
    from custom_trainers.onlinecp_feedback_policy import feedback_config_sha256
    index_path = Path(bank)
    value = contract.read_curriculum_contract_payload(index_path.parent / "feedback_contract.json")
    dataset = f"Dataset{old['dataset_id']:03d}_LiverOnlineCP_OF{old['outer_fold']}"
    expected = {"format": contract.FORMAT, "architecture_version": architecture,
                "geometry_contract": contract.GEOMETRY, "dataset_name": dataset,
                "nnunet_fold": 0, "curriculum_sha256": feedback_config_sha256(policy), "candidate_count": 128}
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("Original Basic bank architecture/dataset/policy contract differs")
    required = {"index", "config", "manifest", "complete", "gnn_checkpoint", "gnn_split", "prototype",
                "graph_complete", "outer_splits", "preprocessed_split", "preprocess_marker", "plans",
                "dataset", "train_config", "nnunet_config"}
    files = value.get("files", {})
    if set(files) != required:
        raise ValueError("Original Basic bank lacks complete provenance files")
    for name, record in files.items():
        _file_record(record, name)
    if Path(files["index"]["path"]) != index_path:
        raise ValueError("Original contract identifies another bank")
    train = contract._ids(value.get("train_case_ids"), "training")
    val = contract._ids(value.get("validation_case_ids"), "validation")
    gtrain = contract._ids(value.get("gnn_train_case_ids"), "GNN training")
    gval = contract._ids(value.get("gnn_validation_case_ids"), "GNN validation")
    proto = contract._ids(value.get("prototype_training_case_ids"), "prototype fitting")
    if (set(train) & set(val) or set(gtrain) & set(gval) or set(gtrain) | set(gval) != set(train)
            or set(proto) != set(gtrain)):
        raise ValueError("Original Basic bank violates patient split/prototype boundaries")
    split = _read(files["gnn_split"]["path"])
    if split.get("train") != gtrain or split.get("val") != gval or split.get("outer_validation_excluded") != val:
        raise ValueError("Original GNN and Basic cohorts disagree")
    live = Path(old["env_updates"]["nnUNet_preprocessed"]) / dataset
    if Path(files["preprocessed_split"]["path"]) != live / "splits_final.json":
        raise ValueError("Original Basic preprocessing split path differs")
    splits = _read(live / "splits_final.json")
    if not isinstance(splits, list) or not splits or splits[0] != {"train": train, "val": val}:
        raise ValueError("Original Basic native split changed")
    marker = _read(files["preprocess_marker"]["path"])
    if marker.get("input_contract", {}).get("planning_cohort") != "outer_train_only_v1":
        raise ValueError("Original preprocessing was not fitted to outer training only")
    configuration = contract._verify_live_preprocessing(value, live, train, val, marker)
    index = _read(index_path)
    if (index.get("dataset_name") != dataset or index.get("candidate_count") != 128
            or index.get("paste_contract") != "onlinecp_raw_target_paste_v1"):
        raise ValueError("Original Basic requires the complete 128-candidate raw-target bank")
    entries = index.get("entries_by_case", {})
    if not entries or not set(entries).issubset(train) or any(not isinstance(v, list) for v in entries.values()):
        raise ValueError("Original Basic entries/cohorts are invalid")
    names = [name for group in entries.values() for name in group]
    if len(names) != len(set(names)) or set(names) != set(value.get("entry_sha256", {})):
        raise ValueError("Original Basic entry inventory differs")
    for name in names:
        path = index_path.parent / name
        if ".." in Path(name).parts or not path.is_relative_to(index_path.parent) or _file(path)["sha256"] != value["entry_sha256"][name]:
            raise ValueError("Original Basic entry escaped its bank or changed")
    return {**value, "contract_sha256": contract.value_sha256(value), "verified_configuration": configuration}, index


def _checkpoint(path, identity, index, policy, old, native, cp):
    import numpy as np
    import torch
    from custom_trainers.onlinecp_feedback_policy import FeedbackState, feedback_config_sha256, stage_for_epoch
    from hiercp.feedback import tensor_state_sha256
    before = _file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if _file(path) != before:
        raise ValueError("Original Basic checkpoint changed during inspection")
    needed = {"network_weights", "optimizer_state", "grad_scaler_state", "logging", "_best_ema", "current_epoch",
              "init_args", "trainer_name", "inference_allowed_mirroring_axes", "onlinecp_curriculum_resume"}
    if not isinstance(checkpoint, dict) or not needed.issubset(checkpoint) or checkpoint["trainer_name"] != TRAINER:
        raise ValueError("Missing complete original Basic native checkpoint")
    state = checkpoint["onlinecp_curriculum_resume"]
    if (not isinstance(state, dict) or state.get("format") != RESUME or checkpoint["current_epoch"] != 250
            or type(checkpoint["current_epoch"]) is not int or state.get("next_epoch") != 250
            or state.get("bank_identity") != identity or state.get("config") != policy
            or state.get("config_sha256") != feedback_config_sha256(policy)):
        raise ValueError("Original Basic epoch/policy/bank state is not the complete requested control")
    runtime = state.get("runtime_identity", {})
    keys = {"trainer", "online_seed", "source_identity", "plans_sha256", "dataset_json_sha256", "configuration",
            "torch_version", "numpy_version", "num_epochs", "gradient_accumulation_steps", "physical_batch_size",
            "train_iterations_per_epoch", "validation_iterations_per_epoch", "augmentation_workers", "device_type",
            "cuda_device_count", "feedback_gnn_config", "feedback_measurement"}
    if not isinstance(runtime, dict) or set(runtime) != keys:
        raise ValueError("Original Basic checkpoint lacks its exact recorded runtime settings")
    if (runtime["trainer"] != TRAINER or runtime["online_seed"] != old["seed"] or type(runtime["online_seed"]) is not int
            or runtime["num_epochs"] != 250 or runtime["gradient_accumulation_steps"] != 1
            or runtime["device_type"] != "cuda" or runtime["cuda_device_count"] != 1
            or runtime["feedback_gnn_config"] is not None
            or runtime["feedback_measurement"] != policy["difficulty"]["measurement_definition"]):
        raise ValueError("Original Basic seed/epoch/device/GNN/measurement contract differs")
    if any(type(runtime[key]) is not int for key in ("num_epochs", "gradient_accumulation_steps", "cuda_device_count")):
        raise ValueError("Original Basic runtime integer fields cannot be boolean/coerced values")
    for key in ("physical_batch_size", "train_iterations_per_epoch", "validation_iterations_per_epoch", "augmentation_workers"):
        if type(runtime[key]) is not int or runtime[key] <= 0:
            raise ValueError(f"Original Basic actual setting is missing: {key}")
    if runtime["torch_version"] != str(torch.__version__) or runtime["numpy_version"] != str(np.__version__):
        raise ValueError("Recorded original torch/numpy versions differ from the target inference environment")
    plans, dataset = _read(identity["files"]["plans"]["path"]), _read(identity["files"]["dataset"]["path"])
    init = checkpoint["init_args"]
    if (not isinstance(init, dict) or init.get("plans") != plans or init.get("dataset_json") != dataset
            or init.get("fold") != 0 or init.get("configuration") != identity["verified_configuration"]
            or runtime["plans_sha256"] != _sha(plans) or runtime["dataset_json_sha256"] != _sha(dataset)
            or runtime["configuration"] != identity["verified_configuration"]):
        raise ValueError("Original Basic native plans/configuration/checkpoint initialization differ")
    configs, name, seen, merged = plans.get("configurations", {}), runtime["configuration"], set(), {}
    while name is not None:
        if name in seen or not isinstance(configs.get(name), dict):
            raise ValueError("Invalid original native plan configuration inheritance")
        seen.add(name)
        row = configs[name]
        merged = {**row, **merged}
        name = row.get("inherits_from")
    if merged.get("batch_size") != runtime["physical_batch_size"]:
        raise ValueError("Original Basic physical batch differs from its verified native plans")
    code = runtime["source_identity"]
    required_code = {"curriculum_trainer": "nnUNetTrainer_OnlineCPCurriculum.py", "curriculum_policy": "onlinecp_curriculum_policy.py",
                     "bank_verifier": "onlinecp_curriculum_contract.py", "legacy_trainer": "nnUNetTrainer_OnlinePairedCP.py",
                     "feedback_trainer": "nnUNetTrainer_OnlineCPFeedback.py", "feedback_policy": "onlinecp_feedback_policy.py",
                     "feedback_metrics": "onlinecp_feedback_metrics.py", "raw_bank": "onlinecp_raw_bank.py",
                     "raw_resampling": "onlinecp_raw_resampling.py"}
    if (not isinstance(code, dict) or set(code) != set(required_code) | {"base_trainer", "hiercp_sources"}
            or any(code[key] != cp[module] for key, module in required_code.items())
            or code["base_trainer"] != native["training/nnUNetTrainer/nnUNetTrainer.py"]
            or code["hiercp_sources"] != (HISTORICAL_HIERCP_SHA if cp == HISTORICAL_CP else _current_hiercp_sha())):
        raise ValueError("Original checkpoint was not produced with its recorded private/native implementations")
    iterations = _native_iterations(old["package_destination"])
    if (runtime["train_iterations_per_epoch"] != iterations["num_iterations_per_epoch"]
            or runtime["validation_iterations_per_epoch"] != iterations["num_val_iterations_per_epoch"]):
        raise ValueError("Original Basic actual iterations differ from its unchanged native implementation")
    last = state.get("last_epoch", {})
    if (set(last) != {"epoch", "stage", "applied", "samples", "event_digest", "choice_digest"}
            or any(type(last[k]) is not int for k in ("epoch", "stage", "applied", "samples"))
            or last["epoch"] != 249 or last["stage"] != stage_for_epoch(policy, 249)[0]
            or last["samples"] != runtime["physical_batch_size"] * runtime["train_iterations_per_epoch"]
            or not 0 <= last["applied"] <= last["samples"]
            or any(not isinstance(last[k], str) or len(last[k]) != 16 or any(c not in "0123456789abcdef" for c in last[k])
                   for k in ("event_digest", "choice_digest"))):
        raise ValueError("Original Basic final epoch lacks complete event/choice/sample evidence")
    extension = state.get("extension", {})
    extension_keys = {"format", "table", "gnn", "predictions", "prediction_provenance", "prediction_bundle_sha256",
                      "last_epoch", "optimizer_steps", "last_observations"}
    if (not isinstance(extension, dict) or set(extension) != extension_keys or extension["format"] != RESUME
            or any(extension[key] is not None for key in ("gnn", "predictions", "prediction_provenance"))
            or extension["prediction_bundle_sha256"] != _sha({"predictions": None, "provenance": None})):
        raise ValueError("Original completed Basic has Full-GNN or partial next-epoch state")
    entries = {name: case for case, names in index["entries_by_case"].items() for name in names}
    table = FeedbackState(policy, entries, candidate_count=128, identity=identity)
    table.load_state_dict(extension["table"])
    summary, records = extension["last_epoch"], extension["last_observations"]
    if (not isinstance(records, list) or not isinstance(summary, dict) or summary.get("epoch") != 249
            or summary.get("observations") != len(records) or summary.get("observation_sha256") != _sha(records)
            or summary.get("gnn") is not None or type(extension["optimizer_steps"]) is not int
            or not 0 < extension["optimizer_steps"] <= 250 * runtime["train_iterations_per_epoch"]):
        raise ValueError("Original Basic observations/optimizer progress are incomplete")
    weights = checkpoint["network_weights"]
    if (not isinstance(weights, dict) or not weights
            or any(not torch.is_tensor(v) or not bool(torch.isfinite(v).all()) for v in weights.values())):
        raise ValueError("Original Basic network weights are invalid")
    progress = {"completed_epoch": 249, "optimizer_steps": extension["optimizer_steps"],
                "network_sha256": tensor_state_sha256(weights)}
    if summary.get("nnunet_progress") != progress:
        raise ValueError("Original Basic epoch was not produced with these exact checkpoint weights")
    if (not isinstance(checkpoint["optimizer_state"], dict)
            or not {"state", "param_groups"}.issubset(checkpoint["optimizer_state"])
            or not isinstance(state.get("lr_scheduler_state"), dict)
            or not isinstance(checkpoint["grad_scaler_state"], dict)
            or any(key not in state for key in ("python_rng", "numpy_rng", "cpu_rng", "cuda_rng"))):
        raise ValueError("Original Basic optimizer/scaler/scheduler/RNG evidence is incomplete")
    random.Random().setstate(state["python_rng"])
    np.random.RandomState().set_state(state["numpy_rng"])
    cpu, cuda = state["cpu_rng"], state["cuda_rng"]
    if (not torch.is_tensor(cpu) or cpu.dtype != torch.uint8 or cpu.ndim != 1
            or not isinstance(cuda, list) or len(cuda) != runtime["cuda_device_count"]
            or any(not torch.is_tensor(v) or v.dtype != torch.uint8 or v.ndim != 1 or not v.numel() for v in cuda)):
        raise ValueError("Original Basic CPU/CUDA RNG evidence is malformed")
    torch.Generator(device="cpu").set_state(cpu.cpu())
    scaler = checkpoint["grad_scaler_state"]
    if (set(scaler) != {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
            or any(type(scaler[k]) not in (int, float) or not math.isfinite(scaler[k])
                   for k in ("scale", "growth_factor", "backoff_factor"))
            or scaler["scale"] <= 0 or scaler["growth_factor"] <= 1 or not 0 < scaler["backoff_factor"] < 1
            or type(scaler["growth_interval"]) is not int or scaler["growth_interval"] <= 0
            or type(scaler["_growth_tracker"]) is not int or scaler["_growth_tracker"] < 0):
        raise ValueError("Original Basic AMP scaler evidence is malformed")
    return runtime, progress, {"grad_scaler_state_present": True, "native_single_device_guard": True,
                               "accumulation_steps": 1, "native_iteration_defaults": iterations,
                               "whole_environment_equivalence_claimed": False}


def inspect_basic_source(source_root: Path, target_plan: Mapping, *, require_complete=True,
                         target_native_package: Path | None = None) -> dict:
    """Return immutable original evidence; never publish/repair/reconcile a source."""
    if require_complete is not True:
        raise ValueError("Partial Basic checkpoints cannot replace a completed control")
    source = Path(source_root).absolute()
    target = Path(target_plan["run_root"]).absolute()
    if (source == target or source.is_relative_to(target) or target.is_relative_to(source)
            or not source.is_dir() or any(p.is_symlink() for p in (source, *source.parents))):
        raise ValueError("Basic source and new experiment must be distinct preserved directories")
    persistent = source / "feedback_execution.lock"
    if any(path != persistent for path in source.glob("*execution.lock")):
        raise ValueError("Original legacy execution lock exists; inspect/stop its owning job first")
    from tools.feedback_stage_execution import run_lock
    context = run_lock(source, create=False) if persistent.exists() or persistent.is_symlink() else nullcontext()
    with context:
        old = _read(source / "launch_plan.json")
        journal_file = _file(source / "execution_journal.json")
        journal = _read(source / "execution_journal.json")
        if Path(old.get("run_root", "")) != source or Path(old.get("package_destination", "")) != source / "runtime/nnunetv2":
            raise ValueError("Original Basic launch root/private runtime identity differs")
        environment = old.get("env_updates")
        if not isinstance(environment, dict):
            raise ValueError("Original Basic native data/result roots are missing")
        for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            expected_root = source / "online/nnunetv2" / name
            value = environment.get(name)
            if (not isinstance(value, str) or Path(value) != expected_root
                    or any(path.is_symlink() for path in (expected_root, *expected_root.parents))):
                raise ValueError(f"Original Basic {name} root escapes its experiment or changed")
        for key in ("medical_root", "outer_fold", "dataset_id", "seed"):
            expected = str(target_plan[key]) if key == "medical_root" else target_plan[key]
            if old.get(key) != expected:
                raise ValueError(f"Original Basic requested identity differs: {key}")
        if any(type(old.get(key)) is not int for key in ("outer_fold", "dataset_id", "seed")):
            raise ValueError("Original Basic fold/dataset/seed must be explicit integers")
        basic, argv, receipt_files = _history(source, old, journal)
        requested_package = Path(target_native_package) if target_native_package is not None else Path(target_plan["package_destination"])
        runtime_inventory, equivalence = _runtime_equivalence(Path(old["package_destination"]), journal["runtime_inventory"], requested_package)
        equivalence["original_history"] = _history_receipt(source, old, journal, equivalence["native_files"])
        equivalence["original_history"]["stage_receipts"] = receipt_files
        project = Path(old["project_root"])
        policy_path, nn_path = project / "config/online_cp_feedback.json", project / "config/nnunet.json"
        from custom_trainers.onlinecp_feedback_policy import validate_feedback_config
        policy = validate_feedback_config(_read(policy_path))
        target_project = Path(target_plan["project_root"])
        nnconfig = _read(nn_path)
        if (policy != validate_feedback_config(_read(target_project / "config/online_cp_feedback.json"))
                or nnconfig != _read(target_project / "config/nnunet.json")):
            raise ValueError("Original Basic policy/native experiment configuration differs from requested training")
        for key, path in (("config/nnunet.json", nn_path), ("config/online_cp_feedback.json", policy_path)):
            if basic["input_files"].get(key) != _file(path)["sha256"]:
                raise ValueError(f"Original Basic recorded configuration changed or is missing: {key}")
        bank = source / f"online/folds/fold_{old['outer_fold']}/bank/index.json"
        expected_argv = [str(old["python_executable"]), "-m", "tools.train_online_feedback", "--bank", str(bank),
                         "--feedback-config", str(policy_path), "--configuration", nnconfig["dataset"]["configuration"],
                         "--device", "cuda", "--seed", str(old["seed"]), "--arm", "basic"]
        if argv != expected_argv:
            raise ValueError("Original Basic training command has unreviewed arguments or ordering")
        identity, index = _verify_bank(bank, old, policy, equivalence["source_architecture"])
        equivalence["source_preprocessing"] = _source_preprocessing(old, identity)
        if _file_record(identity["files"]["nnunet_config"], "nnunet_config") != _file(nn_path):
            raise ValueError("Original bank and Basic native configuration evidence differ")
        configuration = nnconfig["dataset"]["configuration"]
        if identity["verified_configuration"] != configuration:
            raise ValueError("Original Basic requested native configuration differs from preprocessing")
        checkpoint = Path(old["env_updates"]["nnUNet_results"]) / identity["dataset_name"] / (
            f"{TRAINER}__{Path(identity['files']['plans']['path']).stem}__{configuration}") / "fold_0/checkpoint_final.pth"
        checkpoint_file = _file(checkpoint)
        if basic["completion_evidence"]["files"] != {checkpoint.relative_to(source).as_posix(): checkpoint_file["sha256"]}:
            raise ValueError("Original Basic checkpoint differs from the completed stage proof")
        recorded, progress, precision = _checkpoint(checkpoint, identity, index, policy, old,
            equivalence["native_files"], equivalence["source_cp_modules"])
        if recorded["augmentation_workers"] != nnconfig["training"]["nnunet_n_proc_DA"]:
            raise ValueError("Original Basic actual augmentation workers differ from its requested native configuration")
        files = {str(path): _file(path)["sha256"] for path in (
            source / "launch_plan.json", source / "execution_journal.json", policy_path, nn_path,
            bank, bank.parent / "config.json", bank.parent / "feedback_contract.json", checkpoint)}
        if _file(source / "execution_journal.json") != journal_file:
            raise ValueError("Original Basic journal changed during inspection")
        if receipt_files != {path: _file(path)["sha256"] for path in receipt_files}:
            raise ValueError("Original native receipts changed during Basic inspection")
        original_sha = _sha(identity)
        return {"format": FORMAT, "source_root": str(source), "checkpoint": checkpoint_file,
                "bank_identity": identity, "runtime_inventory": runtime_inventory,
                "training_journal_sha256": journal_file["sha256"],
                "bank_binding": {"format": "basic_bank_integrity_binding_v1",
                    "index_sha256": files[str(bank)], "config_sha256": files[str(bank.parent / 'config.json')],
                    "feedback_contract_sha256": files[str(bank.parent / 'feedback_contract.json')],
                    "original_identity_sha256": original_sha},
                "training_contract": {"arm": "basic", "trainer": TRAINER, "outer_fold": old["outer_fold"],
                    "dataset_id": old["dataset_id"], "seed": old["seed"], "epochs": 250,
                    "policy": policy, "nnunet_config": nnconfig, "recorded_runtime": recorded,
                    "completed_progress": progress, "precision_evidence": precision,
                    "historical_input_files": basic["input_files"]},
                "native_equivalence": equivalence, "source_files": files}
