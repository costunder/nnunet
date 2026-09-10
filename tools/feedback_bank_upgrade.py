"""Verified NEW-bank derivation; never rewrites a source experiment or its journal.

The completed quality-GNN stays at its original absolute paths. Raw/preprocessed
data get a new directory view so native NumPy unpacking cannot write into the
source. Only native read-only payload types are hardlinked; metadata is copied.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import uuid
import zipfile

import numpy as np

FORMAT = "feedback_bank_upgrade_execution_v1"
SETUP = ("copy_private_runtime", "install_private_trainers", "environment", "prepare_views")

# These settings are NOT applied to the completed historical optimizer/model.
# The source config is copied byte-for-byte; only current resource defaults may
# differ. Geometry, sampling, model, loss, precision and epochs remain strict.
HISTORICAL_RESOURCE_KEYS = frozenset({
    "runtime.prepare_workers",
    "training.batch_size", "training.batch_size_candidates",
    "training.batch_calibration_repeats", "training.batch_calibration_max_vram_fraction",
    "training.num_workers", "training.num_worker_candidates", "training.loader_calibration_batches",
    "training.gradient_accumulation_steps", "training.target_effective_batch_size",
    "training.pin_memory", "training.prefetch_factor", "training.persistent_workers", "training.cuda_prefetch",
    "generation.scoring_batch_size", "generation.scoring_batch_size_candidates",
    "generation.scoring_batch_calibration_repeats", "generation.scoring_batch_max_vram_fraction",
    "generation.local_candidate_chunk_size", "generation.pin_memory",
    "generation.cpu_prefetch_workers", "generation.cpu_prefetch_cases",
    "generation.save_queue_depth", "generation.case_batch_size",
})


def source_config_differences(original, current):
    """Explicit resource-only provenance, never a rewrite of source settings."""
    changes = []

    def compare(lhs, rhs, path):
        if isinstance(lhs, dict) and isinstance(rhs, dict):
            if set(lhs) != set(rhs):
                keys = sorted(set(lhs).symmetric_difference(rhs))
                key = keys[0]
                raise ValueError(f"Unsupported config field {path + '.' if path else ''}{key}: "
                                 f"source={lhs.get(key, '<missing>')!r}, current={rhs.get(key, '<missing>')!r}")
            for key in sorted(lhs):
                compare(lhs[key], rhs[key], f"{path}.{key}" if path else key)
            return
        if type(lhs) is type(rhs) and lhs == rhs:
            return
        kind = "historical_resource_setting"
        if path == "graph.adaptive_roi_max_voxels":
            if type(lhs) is not int or type(rhs) is not int or lhs < rhs:
                raise ValueError(f"Source graph allocation ceiling differs: {path}: source={lhs!r}, current={rhs!r}")
            kind = "previously_verified_full_graph_allocation_ceiling"
        elif path not in HISTORICAL_RESOURCE_KEYS:
            raise ValueError(f"Source quality/CP configuration differs: {path}: source={lhs!r}, current={rhs!r}")
        changes.append({"path": path, "source": lhs, "current": rhs, "classification": kind,
                        "applied_to_historical_training": False})

    compare(original, current, "")
    return changes


def _launcher():
    from tools import run_feedback_experiment
    return run_feedback_experiment


def _paths(plan):
    converted = dict(plan)
    for key in ("project_root", "medical_root", "run_root", "train_config", "package_destination"):
        converted[key] = Path(converted[key])
    return converted


def build_upgrade_plan(project_root, medical_root, *, outer_fold, dataset_id, seed,
                       python_executable, run_root, experiment_name, train_config,
                       upgrade_bank_from):
    launch = _launcher()
    project = Path(project_root).resolve()
    work = project / "work"
    source = Path(upgrade_bank_from)
    source = (source if source.is_absolute() else project / source).resolve()
    if source == work or not source.is_relative_to(work):
        raise ValueError("Bank-upgrade source must be an experiment within this checkout/work")
    # The ordinary builder remains byte-for-byte compatible for old launch plans.
    plan = launch.build_plan(project, medical_root, outer_fold=outer_fold, dataset_id=dataset_id,
        seed=seed, python_executable=python_executable, run_root=run_root,
        experiment_name=experiment_name if experiment_name is not None or run_root is not None else "feedback_rawcp")
    root = plan["run_root"]
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Bank-upgrade output must not overlap its preserved source")
    config = Path(train_config) if train_config is not None else root / "upgrade/train_config.json"
    config = (config if config.is_absolute() else project / config).resolve()
    if not config.is_relative_to(root):
        raise ValueError("Bank-upgrade train_config must be a new snapshot inside its new root")
    plan = launch.build_plan(project, medical_root, outer_fold=outer_fold, dataset_id=dataset_id,
        seed=seed, python_executable=python_executable, run_root=root, train_config=config)
    keep = {"install_private_trainers", "environment", "bank", "feedback_contract",
            "check_full", "check_basic", "train_full", "train_basic"}
    plan["commands"] = [item for item in plan["commands"] if item["name"] in keep]
    pair = source / "paired"
    for command in plan["commands"]:
        argv = command["argv"]
        if "--paired-root" in argv:
            argv[argv.index("--paired-root") + 1] = pair.relative_to(work).as_posix()
    plan.update(upgrade_source_root=source, reuse_paired_root=pair,
                upgrade_format="feedback_bank_upgrade_plan_v1",
                scope="NEW bank/private runtime/Full+Basic results; reuse verified completed quality GNN and shared preprocessing. No old segmentation checkpoint resume, no GNN/preprocessing recomputation, no evaluation.")
    return plan


def _historical_runtime(root, expected):
    launch = _launcher()
    root = Path(root)
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Source has no saved private-runtime inventory")
    files = [path for path in root.rglob("*") if "__pycache__" not in path.relative_to(root).parts
             and (path.is_file() or path.is_symlink())]
    actual = launch._bound_files(root, files)
    if actual != expected or "__init__.py" not in actual or "training/nnUNetTrainer/nnUNetTrainer.py" not in actual:
        raise ValueError("Historical source runtime differs from its saved bytes")
    # Intentionally NOT current MODULES: updated trainers belong only to new runtime.
    return actual


def _source_plan(plan):
    launch = _launcher()
    source = Path(plan["upgrade_source_root"])
    launch._bound_files(source, [source / "launch_plan.json", source / "execution_journal.json"])
    old = _paths(launch._read_json(source / "launch_plan.json"))
    for key, expected in (("project_root", plan["project_root"]), ("medical_root", plan["medical_root"]),
                          ("run_root", source), ("package_destination", source / "runtime/nnunetv2")):
        if old[key].resolve() != Path(expected).resolve():
            raise ValueError(f"Source experiment identity differs: {key}")
    for key in ("outer_fold", "dataset_id", "seed", "python_executable"):
        if old.get(key) != plan[key]:
            raise ValueError(f"Source experiment identity differs: {key}")
    if not old["train_config"].is_relative_to(source):
        raise ValueError("Upgrade requires the source recovery's immutable train-config artifact")
    if old.get("upgrade_source_root"):
        raise ValueError("Chained bank upgrades need a separately reviewed lineage; no implicit adoption")
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        expected = source / "online/nnunetv2" / name
        if Path(old["env_updates"][name]).resolve() != expected.resolve():
            raise ValueError(f"Source native dataset/results path differs: {name}")
    return old


def _check_source_history(old, journal):
    launch = _launcher()
    source = old["run_root"]
    keys = {"format", "plan_sha256", "source_identity", "runtime_inventory", "preparation_receipt",
            "stages", "training_started", "complete", "journal_sha256"}
    if not isinstance(journal, dict) or set(journal) != keys:
        raise ValueError("Source execution journal has an unsupported schema")
    payload = {key: value for key, value in journal.items() if key != "journal_sha256"}
    if (journal["format"] != "feedback_preparation_execution_v1"
            or journal["journal_sha256"] != launch._json_sha256(payload)
            or journal["plan_sha256"] != launch._json_sha256(old)
            or type(journal["training_started"]) is not bool or journal["complete"] is not False):
        raise ValueError("Source journal/launch checksum or status is invalid")
    if (source / "recovery_execution.lock").exists() or (source / "bank_upgrade_execution.lock").exists():
        raise ValueError("Source execution lock exists; stop/inspect its exact owning job first")
    _historical_runtime(old["package_destination"], journal["runtime_inventory"])
    preparation = {"copy_private_runtime", "install_private_trainers", "environment", "recover_preparation"}
    sequence = [row["name"] for row in old["commands"] if row["name"] not in {"install_private_trainers", "environment"}]
    if len(sequence) != len(set(sequence)):
        raise ValueError("Source launch has duplicated stages")
    completed, entered = set(), False
    legacy_failures, bound_history_started = [], False
    inputs = launch._resume_inputs(old)
    if not isinstance(journal["stages"], list):
        raise ValueError("Source stage history is malformed")
    for stage_index, row in enumerate(journal["stages"], start=1):
        if not isinstance(row, dict) or row.get("status") not in {"completed", "failed"}:
            raise ValueError("Source has a running or ambiguous stage; no upgrade is launched")
        name = row.get("name")
        if name in {"train_full", "train_basic"}:
            raise ValueError("Source segmentation training was entered; new training is not checkpoint continuation")
        if name in preparation:
            if entered:
                raise ValueError("Source preparation was re-entered after training")
        else:
            entered = True
            position = len(completed - preparation)
            if not preparation.issubset(completed) or position >= len(sequence) or name != sequence[position]:
                raise ValueError("Source stage order is invalid")
            location = f"source stage #{stage_index} {name} ({row['status']})"
            if "input_files" not in row:
                # Before 301482c, the recovery writer recorded only these three
                # fields on failure. No output from that failed attempt is
                # adopted: a later hash-bound completion of this SAME stage is
                # mandatory, including all existing native output checks below.
                if (bound_history_started or row["status"] != "failed"
                        or set(row) != {"name", "status", "error"}
                        or not isinstance(row["error"], str) or not row["error"].strip()):
                    raise ValueError(f"Missing input_files at {location}; not a supported legacy failed "
                                     "attempt. This is missing provenance, not evidence of changed configuration.")
                legacy_failures.append({"stage_index": stage_index, "name": name})
            else:
                bound_history_started = True
                recorded = row["input_files"]
                if (not isinstance(recorded, dict) or not recorded
                        or any(not isinstance(key, str) or not isinstance(value, str)
                               or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
                               for key, value in recorded.items())):
                    raise ValueError(f"Malformed input_files at {location}; expected path-to-SHA256 mapping")
                if recorded != inputs:
                    differences = [{"path": key, "recorded_sha256": recorded.get(key),
                                    "current_sha256": inputs.get(key)}
                                   for key in sorted(set(recorded) | set(inputs))
                                   if recorded.get(key) != inputs.get(key)]
                    raise ValueError(f"Source experiment configurations changed at {location}: "
                                     + json.dumps(differences, sort_keys=True)
                                     + ". Source journal and files were not modified.")
            if row["status"] == "completed":
                evidence = row.get("completion_evidence")
                if not isinstance(evidence, dict) or evidence.get("format") != "feedback_stage_completion_v1":
                    raise ValueError("Source completed stage lacks durable evidence")
                if launch._bound_files(source, evidence.get("files", {})) != evidence.get("files"):
                    raise ValueError("Source completed-stage evidence changed")
                if name in {"gnn-train", "plan"} and evidence != launch._stage_evidence(old, name):
                    raise ValueError(f"Source stage fails native verification: {name}")
        if row["status"] == "completed":
            completed.add(name)
    if any(row["name"] not in completed for row in legacy_failures):
        raise ValueError("Legacy failed source stage has no later hash-bound verified completion: "
                         + json.dumps(legacy_failures, sort_keys=True))
    if not preparation.union({"gnn-train", "plan"}).issubset(completed) or not journal["training_started"]:
        raise ValueError("Upgrade requires completed quality-GNN/causality and native preprocessing")
    results = Path(old["env_updates"]["nnUNet_results"])
    if results.is_symlink() or (results.exists() and any(path.is_file() or path.is_symlink() for path in results.rglob("*"))):
        raise ValueError("Source segmentation output exists; refusing to relabel a training restart as upgrade")
    launch._verify_preparation_receipt(old, journal)
    identity = journal["source_identity"]
    if (not isinstance(identity, dict) or not isinstance(identity.get("source_root"), str)
            or not isinstance(identity.get("files"), dict) or not identity["files"]):
        raise ValueError("Source preparation identity is malformed")
    identity_root = Path(identity["source_root"])
    work = old["project_root"] / "work"
    if not identity_root.is_absolute() or identity_root == work or not identity_root.is_relative_to(work):
        raise ValueError("Source preparation identity escapes checkout/work")
    if launch._bound_files(identity_root, identity["files"]) != identity["files"]:
        raise ValueError("Source preparation identity files changed")
    from tools.feedback_preparation_recovery import verify_identity
    verify_identity(identity)
    return legacy_failures


def _layout(plan, *, source=False):
    from tools import online_cp_benchmark as online
    launch = _launcher()
    active = _source_plan(plan) if source else plan
    root, project = active["run_root"], active["project_root"]
    paired = root / "paired" if source else Path(plan["reuse_paired_root"])
    return online.make_layout(argparse.Namespace(project_root=str(project), medical_root=str(active["medical_root"]),
        paired_root=paired.relative_to(project / "work").as_posix(),
        online_root=(root / "online").relative_to(project / "work").as_posix(),
        train_config=str(active["train_config"])))


def validate_source(plan):
    """Read-only native validation; old bank rows are evidence, never adopted."""
    launch = _launcher()
    from tools import online_cp_benchmark as online
    old = _source_plan(plan)
    source = old["run_root"]
    journal = launch._read_json(source / "execution_journal.json")
    legacy_failures = _check_source_history(old, journal)
    original = launch._read_json(old["train_config"])
    current = launch._read_json(plan["project_root"] / "config/train.json")
    differences = source_config_differences(original, current)
    layout = _layout(plan, source=True)
    causality = online._verified_gnn_causality(layout, plan["outer_fold"])
    marker, _, _ = online._verified_preprocess_contract(layout, plan["outer_fold"], original,
        launch._read_json(plan["project_root"] / "config/nnunet.json"), plan["dataset_id"])
    required = [source / "launch_plan.json", source / "execution_journal.json", old["train_config"],
                source / "recovery/complete.json"]
    for row in journal["stages"]:
        if row.get("name") in {"gnn-train", "plan"} and row["status"] == "completed":
            required.extend(source / name for name in row["completion_evidence"]["files"])
    if legacy_failures:
        print("[SOURCE LEGACY HISTORY] Preserved pre-301482c failed attempts without input hashes; "
              "later same-stage completions, input hashes, native outputs and preparation receipt verified. "
              "No failed-attempt outputs adopted or source files rewritten: "
              + json.dumps(legacy_failures, sort_keys=True), flush=True)
    return {"format": "feedback_bank_upgrade_source_v1", "source_root": str(source),
            "files": launch._bound_files(source, required),
            "runtime_inventory": journal["runtime_inventory"],
            "source_config_differences": differences,
            "causality_sha256": launch._json_sha256(causality),
            "preprocess_sha256": launch._json_sha256(marker)}


def _dataset_paths(plan, old):
    name = f"Dataset{plan['dataset_id']:03d}_LiverOnlineCP_OF{plan['outer_fold']}"
    return [(kind, Path(old["env_updates"][kind]) / name, Path(plan["env_updates"][kind]) / name)
            for kind in ("nnUNet_raw", "nnUNet_preprocessed")]


def _preprocessed_data_shape(path):
    """Read array metadata only, never materialize the native CT volume."""
    path = Path(path)
    if path.suffix == ".npz":
        with zipfile.ZipFile(path) as archive, archive.open("data.npy") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"Unsupported native NumPy header version {version}: {path}")
            if dtype.hasobject:
                raise ValueError(f"Object-valued native preprocessing data is unsupported: {path}")
    elif path.suffix == ".b2nd":
        import blosc2
        shape = blosc2.open(str(path), mode="r").shape
    else:
        raise ValueError(f"Unsupported native data storage: {path}")
    if len(shape) != 4 or int(shape[0]) != 1 or any(int(value) <= 0 for value in shape):
        raise ValueError(f"Expected full one-channel native CT array, found {shape}: {path}")
    return [int(value) for value in shape]


def _runtime_copy_upper_bytes(old):
    source = Path(old["package_destination"])
    native = sum(path.stat().st_size for path in source.rglob("*")
                 if path.is_file() and "__pycache__" not in path.relative_to(source).parts)
    # The actual copy excludes old custom modules. Counting them plus all current
    # custom Python sources is a measured upper bound for the new private package.
    custom = Path(_launcher().__file__).resolve().parents[1] / "custom_trainers"
    modules = list(custom.rglob("*.py"))
    if not modules:
        raise ValueError("Current trainer source inventory is missing")
    return native + sum(path.stat().st_size for path in modules)


def _baseline_data_files(preprocessed):
    from tools.online_cp_benchmark import PREPROCESS_MARKER_NAME
    marker = _launcher()._read_json(preprocessed / PREPROCESS_MARKER_NAME)
    outputs = marker.get("outputs", {})
    training = marker.get("input_contract", {}).get("train_ids")
    cohort = outputs.get("cases")
    if (not isinstance(training, list) or not training or not all(isinstance(x, str) and x for x in training)
            or len(set(training)) != len(training) or not isinstance(cohort, list)):
        raise ValueError("Native preprocessing marker lacks the full outer-train storage cohort")
    ids = [row.get("case_id") for row in cohort if isinstance(row, dict)]
    if (len(ids) != len(cohort) or not all(isinstance(x, str) and x for x in ids)
            or len(set(ids)) != len(ids) or not set(training).issubset(ids)):
        raise ValueError("Native preprocessing cohort is incomplete or duplicated")
    identifier = outputs.get("data_identifier")
    if not isinstance(identifier, str) or not identifier or Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("Native preprocessing data identifier is unsafe")
    storage = outputs.get("storage_format")
    if storage not in {"npz", "blosc2"}:
        raise ValueError(f"Unsupported native preprocessing storage: {storage!r}")
    suffix = ".npz" if storage == "npz" else ".b2nd"
    for case_id in training:
        if Path(case_id).name != case_id or case_id in {".", ".."}:
            raise ValueError("Unsafe case ID in native preprocessing cohort")
    return {preprocessed / identifier / (case_id + suffix): case_id for case_id in training}


def view_plan(plan, *, check_storage=True):
    """Inspect actual storage and native unpack costs before creating any output."""
    launch = _launcher()
    old = _source_plan(plan)
    rows, unpack_bytes, baseline_inventory = [], 0, []
    for kind, source, target in _dataset_paths(plan, old):
        raw_mode = launch._read_json(source / "online_cp_dataset.json")["materialization"] if kind == "nnUNet_raw" else None
        baseline_files = _baseline_data_files(source) if kind == "nnUNet_preprocessed" else {}
        measured = set()
        for path in sorted(source.rglob("*")):
            if path.is_dir() and not path.is_symlink():
                continue
            if kind == "nnUNet_preprocessed" and path.suffix == ".npy":
                # Unpacked caches are not bound by native completion. Regenerate from verified NPZ in new root.
                continue
            if not path.is_file() or (path.is_symlink() and kind != "nnUNet_raw"):
                raise ValueError(f"Unsupported preprocessing artifact/link: {path}")
            relative = path.relative_to(source)
            raw_payload = kind == "nnUNet_raw" and relative.parts[0] in {"imagesTr", "labelsTr"}
            mode = raw_mode if raw_payload else "copy"
            if kind == "nnUNet_preprocessed" and (path.suffix in {".npz", ".b2nd"} or path.name.endswith(".nii.gz")):
                mode = "hardlink"
            if mode not in {"copy", "hardlink", "symlink"}:
                raise ValueError("Unknown native raw materialization mode")
            if kind == "nnUNet_preprocessed" and path.suffix == ".npz":
                with zipfile.ZipFile(path) as archive:
                    entries = {entry.filename: entry for entry in archive.infolist()}
                    if not {"data.npy", "seg.npy"}.issubset(entries):
                        raise ValueError(f"Native NPZ lacks data/seg storage: {path}")
                    unpack_bytes += sum(entries[name].file_size for name in ("data.npy", "seg.npy"))
            if path in baseline_files:
                shape = _preprocessed_data_shape(path)
                baseline_inventory.append({"case_id": baseline_files[path], "source": str(path), "shape": shape,
                                           "payload_bytes": math.prod(shape[1:]) * 10})
                measured.add(path)
            rows.append({"source": str(path), "target": str(target / relative), "mode": mode,
                         "bytes": path.stat().st_size, "sha256": launch._file_sha256(path)})
        if measured != set(baseline_files):
            missing = sorted(str(path) for path in set(baseline_files) - measured)
            raise ValueError(f"Full outer-train storage inventory is missing files: {missing}")
    rows.append({"source": str(old["train_config"]), "target": str(plan["train_config"]),
                 "mode": "copy", "bytes": old["train_config"].stat().st_size,
                 "sha256": launch._file_sha256(old["train_config"])})
    ancestor = plan["run_root"].parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    usage = shutil.disk_usage(ancestor)
    device = ancestor.stat().st_dev
    for row in rows:
        if row["mode"] == "hardlink" and Path(row["source"]).stat().st_dev != device:
            raise OSError("Verified payload hardlink crosses filesystems; no silent full-data copy fallback")
    if not baseline_inventory:
        raise ValueError("No full native CT cases were inventoried for new raw-target bank storage")
    baseline_bytes = sum(row["payload_bytes"] for row in baseline_inventory)
    runtime_bytes = _runtime_copy_upper_bytes(old)
    required = (int(plan["minimum_free_bytes"]) + unpack_bytes + baseline_bytes + runtime_bytes
                + sum(row["bytes"] for row in rows if row["mode"] == "copy"))
    resources = {"format": "feedback_bank_upgrade_storage_v1", "storage_path": str(ancestor),
                 "total_bytes": usage.total, "free_bytes": usage.free, "required_free_bytes": required,
                 "native_numpy_unpack_bytes": unpack_bytes, "minimum_reserve_bytes": int(plan["minimum_free_bytes"]),
                 "private_runtime_copy_upper_bytes": runtime_bytes,
                 "bank_raw_baseline_payload_bytes": baseline_bytes,
                 "bank_raw_baseline_bound_cohort": "all_outer_train_cases_from_verified_preprocess_contract",
                 "bank_raw_baseline_case_inventory": baseline_inventory,
                 "bank_raw_baseline_dtypes": {"ct": "float64", "segmentation": "int16"},
                 "bank_raw_baseline_excludes": "container overhead and temporary resampling buffers; not a peak-RAM estimate"}
    if check_storage and usage.free < required:
        raise OSError(f"Insufficient bank-upgrade storage: {json.dumps(resources, sort_keys=True)}")
    return rows, resources


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, default=str, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def _owned_view_target(plan, target):
    """A leaf raw symlink is allowed; no directory may redirect native writes."""
    root, target = Path(plan["run_root"]), Path(target)
    if not target.is_absolute() or not target.is_relative_to(root) or ".." in target.parts:
        raise ValueError(f"Escaping bank-upgrade data-view target: {target}")
    current = root
    for part in ("", *target.relative_to(root).parts[:-1]):
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Data-view directory symlink could redirect native writes: {current}")
    if not target.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Escaping bank-upgrade data-view parent: {target}")


def prepare_views(plan, rows, resources):
    launch = _launcher()
    for row in rows:
        source, target = Path(row["source"]), Path(row["target"])
        _owned_view_target(plan, target)
        if not target.is_relative_to(plan["run_root"]) or target.exists() or target.is_symlink():
            raise ValueError(f"Existing/escaping bank-upgrade artifact preserved: {target}")
        if launch._file_sha256(source) != row["sha256"]:
            raise ValueError(f"Source changed before materialization: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if row["mode"] == "hardlink":
            os.link(source.resolve(), target)
        elif row["mode"] == "symlink":
            target.symlink_to(source.resolve())
        else:
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
        if launch._file_sha256(target) != row["sha256"]:
            raise ValueError(f"Copied bank-upgrade artifact differs: {target}")
    raw = Path(plan["env_updates"]["nnUNet_raw"]) / f"Dataset{plan['dataset_id']:03d}_LiverOnlineCP_OF{plan['outer_fold']}"
    (raw / "imagesTs").mkdir(exist_ok=True)
    launch._verify_online_artifacts(plan, "plan")
    receipt = {"format": "feedback_bank_upgrade_views_v1", "files": rows, "storage": resources,
               "preprocessing_recomputed": False, "quality_training_performed": False,
               "source_config_differences": source_config_differences(
                   launch._read_json(plan["train_config"]),
                   launch._read_json(plan["project_root"] / "config/train.json"))}
    _write_new(plan["run_root"] / "upgrade/views.json", receipt)
    return receipt


def _verify_views(plan, receipt):
    launch = _launcher()
    if (not isinstance(receipt, dict) or receipt.get("format") != "feedback_bank_upgrade_views_v1"
            or receipt.get("preprocessing_recomputed") is not False or receipt.get("quality_training_performed") is not False
            or not isinstance(receipt.get("files"), list) or not receipt["files"]
            or launch._read_json(plan["run_root"] / "upgrade/views.json") != receipt):
        raise ValueError("Bank-upgrade data view lacks an unchanged receipt")
    expected_rows, _ = view_plan(plan, check_storage=False)
    differences = source_config_differences(launch._read_json(plan["train_config"]),
        launch._read_json(plan["project_root"] / "config/train.json"))
    if receipt["files"] != expected_rows or receipt.get("source_config_differences") != differences:
        raise ValueError("Bank-upgrade data view no longer matches its verified source manifest")
    seen = set()
    for row in receipt["files"]:
        source, target = Path(row["source"]), Path(row["target"])
        _owned_view_target(plan, target)
        if not target.is_relative_to(plan["run_root"]) or str(target) in seen:
            raise ValueError("Duplicate or escaping data-view receipt target")
        seen.add(str(target))
        if (launch._file_sha256(source) != row["sha256"] or launch._file_sha256(target) != row["sha256"]
                or (row["mode"] == "symlink" and (not target.is_symlink() or target.resolve() != source.resolve()))
                or (row["mode"] == "hardlink" and (target.is_symlink() or not target.samefile(source)))
                or (row["mode"] == "copy" and (target.is_symlink() or target.samefile(source)))):
            raise ValueError(f"Data-view contents/materialization changed: {target}")
    launch._verify_online_artifacts(plan, "plan")


def load_upgrade_journal(plan, identity):
    launch = _launcher()
    root = plan["run_root"]
    launch._bound_files(root, [root / "launch_plan.json", root / "execution_journal.json", root / "upgrade/views.json"])
    journal = launch._read_json(root / "execution_journal.json")
    expected = {"format", "plan_sha256", "source_identity", "runtime_inventory", "view_receipt", "stages", "complete", "journal_sha256"}
    if not isinstance(journal, dict) or set(journal) != expected:
        raise ValueError("Malformed bank-upgrade journal")
    checksum = journal.pop("journal_sha256")
    if (journal["format"] != FORMAT or checksum != launch._json_sha256(journal)
            or journal["plan_sha256"] != launch._json_sha256(plan)
            or launch._json_sha256(launch._read_json(root / "launch_plan.json")) != launch._json_sha256(plan)
            or journal["source_identity"] != identity or type(journal["complete"]) is not bool):
        raise ValueError("Bank-upgrade plan/source/journal identity changed")
    if launch._runtime_inventory(plan["package_destination"]) != journal["runtime_inventory"]:
        raise ValueError("Bank-upgrade private runtime changed")
    _verify_views(plan, journal["view_receipt"])
    sequence = [*SETUP, *[item["name"] for item in plan["commands"] if item["name"] not in SETUP]]
    complete = []
    if not isinstance(journal["stages"], list):
        raise ValueError("Malformed bank-upgrade stage history")
    for row in journal["stages"]:
        if (not isinstance(row, dict) or row.get("status") not in {"completed", "failed"}
                or len(complete) >= len(sequence) or row.get("name") != sequence[len(complete)]):
            raise ValueError("Bank-upgrade contains a running/ambiguous or out-of-order attempt")
        name = row["name"]
        if name not in SETUP:
            if row.get("input_files") != launch._resume_inputs(plan):
                raise ValueError("Bank-upgrade configuration changed")
            if row["status"] == "completed" and row.get("completion_evidence") != launch._stage_evidence(plan, name):
                raise ValueError(f"Bank-upgrade completed-stage evidence changed: {name}")
        if row["status"] == "completed":
            complete.append(name)
    if not set(SETUP).issubset(complete):
        raise ValueError("Bank-upgrade setup is incomplete; do not restart or overwrite this partial root")
    if journal["complete"] and complete != sequence:
        raise ValueError("Bank-upgrade completion flag disagrees with its stage evidence")
    return journal


def dry_run_upgrade(plan, *, resume=False):
    root = plan["run_root"]
    if root.is_symlink() or (root.exists() and not resume):
        raise FileExistsError(f"Existing bank-upgrade output preserved: {root}")
    if (root / "bank_upgrade_execution.lock").exists():
        raise FileExistsError("Bank-upgrade execution lock exists; no process was stopped")
    identity = validate_source(plan)
    journal = load_upgrade_journal(plan, identity) if resume else None
    if not resume:
        _, resources = view_plan(plan)
        print(f"[UPGRADE STORAGE] {json.dumps(resources, sort_keys=True)}", flush=True)
    print(f"[DRY RUN ONLY] {'Continue' if resume else 'Create'} bank upgrade {plan['run_root']}; source remains {plan['upgrade_source_root']}; no writes/children/GPU checks")
    for command in plan["commands"]:
        rows = [row for row in journal["stages"] if row["name"] == command["name"]] if journal else []
        if any(row["status"] == "completed" for row in rows):
            print(f"[VERIFIED SKIP {command['name']}]")
        else:
            argv = _launcher()._resume_command(plan, command, previously_attempted=bool(rows)) if resume else command["argv"]
            print(f"[{command['name']}] {shlex.join(argv)}")


def execute_upgrade(plan, *, runner=None, package_root=None, resume=False):
    launch = _launcher()
    runner = subprocess.run if runner is None else runner
    root = plan["run_root"]
    if root.is_symlink() or (root.exists() and not resume):
        raise FileExistsError(f"Existing bank-upgrade output preserved: {root}")
    if (root / "bank_upgrade_execution.lock").exists():
        raise FileExistsError("Bank-upgrade execution lock exists; no process was stopped")
    identity = validate_source(plan)
    old = _source_plan(plan)
    source_runtime = old["package_destination"]
    if package_root is not None and Path(package_root).resolve() != source_runtime.resolve():
        raise ValueError("Bank upgrade must copy the verified historical native runtime")
    journal = load_upgrade_journal(plan, identity) if resume else None
    rows, resources = (None, None) if resume else view_plan(plan)
    print("[HISTORICAL QUALITY CONFIG] Source bytes and completed optimizer trajectory are preserved; "
          "no current batch/worker setting is applied to that training. "
          + json.dumps(identity.get("source_config_differences", []), sort_keys=True), flush=True)
    if resources is not None:
        print("[BANK UPGRADE STORAGE] " + json.dumps(resources, sort_keys=True), flush=True)
    if resume and shutil.disk_usage(root).free < plan["minimum_free_bytes"]:
        raise OSError("Insufficient free space for the configured bank-upgrade reserve")
    launch.audit_sources()
    env = {**os.environ, **plan["env_updates"], "PYTHONDONTWRITEBYTECODE": "1"}
    env["PYTHONPATH"] = os.pathsep.join([str(root / "runtime"), str(plan["project_root"]), env.get("PYTHONPATH", "")])
    gpu = "import torch\nn = torch.cuda.device_count()\nif n != 1:\n    raise RuntimeError(f'Expected one allocated visible GPU, found {n}; GPU visibility was not changed')\n"
    runner([plan["python_executable"], "-c", gpu], cwd=plan["project_root"], env=env, check=True)
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
        _write_new(root / "launch_plan.json", plan)
    lock = root / "bank_upgrade_execution.lock"
    token = {"pid": os.getpid(), "token": uuid.uuid4().hex, "run_root": str(root)}
    _write_new(lock, token)
    try:
        if resume:
            journal = load_upgrade_journal(plan, identity)
            original = (root / "execution_journal.json").read_bytes()
            archive = root / "upgrade/journal_history" / (launch._file_sha256(root / "execution_journal.json") + ".json")
            _owned_view_target(plan, archive)
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                if archive.is_symlink() or archive.read_bytes() != original:
                    raise ValueError("Archived bank-upgrade journal differs")
            else:
                with archive.open("xb") as handle:
                    handle.write(original)
                    handle.flush()
                    os.fsync(handle.fileno())
        else:
            journal = dict(format=FORMAT, plan_sha256=launch._json_sha256(plan), source_identity=identity,
                           runtime_inventory=None, view_receipt=None, stages=[], complete=False)
            launch._save_journal(root, journal)
        commands = {item["name"]: item for item in plan["commands"]}

        def stage(name, action):
            row = {"name": name, "status": "running", "attempt_id": uuid.uuid4().hex}
            if name not in SETUP:
                row["input_files"] = launch._resume_inputs(plan)
            journal["stages"].append(row)
            launch._save_journal(root, journal)
            try:
                action()
                if name not in SETUP:
                    row["completion_evidence"] = launch._stage_evidence(plan, name)
            except (Exception, KeyboardInterrupt) as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                launch._save_journal(root, journal)
                raise
            row["status"] = "completed"
            launch._save_journal(root, journal)

        def child(name, previous=False):
            argv = launch._resume_command(plan, commands[name], previously_attempted=previous)
            print(f"[BANK UPGRADE {name}] {shlex.join(argv)}", flush=True)
            runner(argv, cwd=plan["project_root"], env=env, check=True)

        if not resume:
            stage("copy_private_runtime", lambda: launch.copy_nnunet_package(source_runtime, plan["package_destination"]))
            def install():
                child("install_private_trainers")
                journal["runtime_inventory"] = launch._runtime_inventory(plan["package_destination"])
            stage("install_private_trainers", install)
            stage("environment", lambda: child("environment"))
            def views():
                journal["view_receipt"] = prepare_views(plan, rows, resources)
            stage("prepare_views", views)
        for name in commands:
            if name in SETUP:
                continue
            previous = [row for row in journal["stages"] if row["name"] == name]
            if any(row["status"] == "completed" for row in previous):
                print(f"[VERIFIED SKIP {name}] completed artifacts unchanged", flush=True)
                continue
            stage(name, lambda name=name, previous=previous: child(name, bool(previous)))
        journal["complete"] = True
        launch._save_journal(root, journal)
        print(f"[BANK UPGRADE COMPLETED] {plan['env_updates']['nnUNet_results']}; no downstream evaluation was run", flush=True)
    finally:
        if launch._read_json(lock) != token:
            raise RuntimeError("Bank-upgrade lock ownership changed; it was not removed")
        lock.unlink()
