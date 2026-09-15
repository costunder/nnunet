"""Verified nnU-Net preprocessing derivative for a fresh, independent GNN run.

This does not adopt old learned models, graph caches, optimizers or CP banks.
Only manifest-bound native preprocessing payloads and original raw data are
shared. Metadata and the current completion marker live in the NEW run root.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import uuid
import zipfile

FORMAT = "feedback_preprocessing_reuse_v1"
PROJECTION = "nnunet_preprocessing_dependencies_v1"
_TOP = frozenset("method seed labels ct_clip graph cache training model generation runtime".split())
_FIELDS = {
    "labels": frozenset("liver tumor".split()),
    "cache": frozenset("source_selection source_pad samples_per_case total_candidates candidate_pool_size easy_fraction inter_fraction intra_fraction max_draws min_liver_coverage occupied_clearance_vox min_center_separation_mm min_center_separation_vox no_placement_policy".split()),
    "training": frozenset("val_fraction epochs batch_size batch_size_candidates batch_calibration_repeats batch_calibration_max_vram_fraction num_workers num_worker_candidates loader_calibration_batches lr weight_decay amp grad_clip easy_epochs inter_epochs intra_epochs model_mine_start_epoch semi_hard_low_percentile semi_hard_high_percentile cross_entropy_weight pairwise_weight ordinal_weight mined_weight consistency_weight fixed_validation_epoch checkpoint_metric_precision gradient_accumulation_steps target_effective_batch_size fused_optimizer pin_memory prefetch_factor persistent_workers cuda_prefetch".split()),
    "model": frozenset("hidden_dim heads local_layers patient_layers prototype_layers dropout dense_base_channels dense_feature_dim dense_batch_size channels_last_3d checkpoint_local_blocks checkpoint_dense_encoder ablation_mode architecture_version upper_feature_policy".split()),
    "generation": frozenset("source_selection source_pad num_copies num_candidates scoring_batch_size scoring_batch_size_candidates scoring_batch_calibration_repeats scoring_batch_max_vram_fraction max_draws min_liver_coverage occupied_clearance_vox min_center_separation_mm min_center_separation_vox no_placement_policy top_k temperature intensity_scale_range intensity_shift_range blend_border amp local_candidate_chunk_size pin_memory cpu_prefetch_workers cpu_prefetch_cases save_queue_depth case_batch_size".split()),
    "runtime": frozenset("prepare_workers deterministic allow_tf32 cudnn_benchmark".split()),
}


def _modules():
    from tools import run_feedback_experiment as launch, online_cp_benchmark as online
    from tools import feedback_fresh_execution as fresh
    return launch, online, fresh


def _paths(plan):
    output = dict(plan)
    for key in ("project_root", "medical_root", "run_root", "train_config", "package_destination",
                "preprocessing_source_root", "reuse_paired_root"):
        if key in output:
            output[key] = Path(output[key]).resolve()
    return output


def _layout(plan):
    _, online, _ = _modules()
    work = plan["project_root"] / "work"
    return online.make_layout(argparse.Namespace(
        project_root=str(plan["project_root"]), medical_root=str(plan["medical_root"]),
        paired_root=Path(plan.get("reuse_paired_root", plan["run_root"] / "paired")).relative_to(work).as_posix(),
        online_root=(plan["run_root"] / "online").relative_to(work).as_posix(),
        train_config=str(plan["train_config"])))


def configuration_differences(original, current):
    """Explicit non-preprocessing sections, not an arbitrary SHA exception.

    The native raw builder reads train_cfg.labels only. The native preprocessing
    inputs separately bind the complete nnunet.json, split and original cohort.
    Known GNN/CP-only differences are recorded in full, never applied to old
    training. Unknown sections/keys require a reviewed projection revision.
    """
    from hiercp.schema import GraphBuildConfig
    fields = {**_FIELDS, "graph": frozenset(GraphBuildConfig.__dataclass_fields__)}
    def plain(value):
        return (all(plain(item) for item in value) if isinstance(value, list)
                else value is None or type(value) in (str, bool, int, float))

    for name, config in (("source", original), ("current", current)):
        if not isinstance(config, dict) or set(config) - _TOP:
            raise ValueError(f"Unknown {name} preprocessing-projection config sections")
        if not isinstance(config.get("labels"), dict) or set(config["labels"]) != _FIELDS["labels"]:
            raise ValueError(f"Incomplete {name} label schema")
        if (any(type(value) is not int or value <= 0 for value in config["labels"].values())
                or len(set(config["labels"].values())) != 2):
            raise ValueError(f"Invalid {name} native label identifiers")
        if "method" in config and not isinstance(config["method"], str):
            raise ValueError(f"Unknown {name} method schema")
        if "seed" in config and type(config["seed"]) is not int:
            raise ValueError(f"Unknown {name} seed schema")
        if "ct_clip" in config and (not isinstance(config["ct_clip"], list)
                or len(config["ct_clip"]) != 2 or any(type(value) not in (int, float) for value in config["ct_clip"])):
            raise ValueError(f"Unknown {name} CT clipping schema")
        for section, allowed in fields.items():
            if section not in config:
                continue
            value = config[section]
            if not isinstance(value, dict) or set(value) - allowed:
                raise ValueError(f"Unknown {name} preprocessing-projection key in {section}")
            if any(not plain(item) for item in value.values()):
                raise ValueError(f"Unknown nested configuration schema in {section}")
    if original["labels"] != current["labels"]:
        raise ValueError("Native preprocessing label semantics changed")
    differences = []
    for section in sorted(set(original) | set(current)):
        left, right = original.get(section), current.get(section)
        if type(left) is type(right) and left == right:
            continue
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                if key not in left or key not in right or type(left[key]) is not type(right[key]) or left[key] != right[key]:
                    differences.append({"path": f"{section}.{key}", "source_present": key in left,
                                        "current_present": key in right, "source": left.get(key), "current": right.get(key)})
        else:
            differences.append({"path": section, "source_present": section in original,
                                "current_present": section in current, "source": left, "current": right})
    return differences


def _source_input_differences(old, recorded):
    """Audit all four inputs; only named CP-only SHA changes are projected.

    This is never a checkpoint/training-resume compatibility exception.
    The source train config and complete nnUNet config remain unchanged.
    """
    launch, _, _ = _modules()
    current = launch._resume_inputs(old)
    train_key = old["train_config"].relative_to(old["project_root"]).as_posix()
    permitted = {"config/online_cp_feedback.json", "config/online_cp_feedback_gnn.json"}
    expected = {train_key, "config/nnunet.json", *permitted}
    if (not isinstance(recorded, dict) or set(recorded) != expected or set(current) != expected
            or any(not isinstance(value, str) or len(value) != 64
                   or any(character not in "0123456789abcdef" for character in value)
                   for value in recorded.values())):
        raise ValueError("Unknown or malformed source preprocessing input registry")
    changes = []
    for key in sorted(expected):
        if recorded[key] == current[key]:
            continue
        if key not in permitted:
            raise ValueError(f"Source native preprocessing input changed: {key}")
        changes.append({"path": key, "recorded_sha256": recorded[key], "current_sha256": current[key],
                        "classification": "cp_only_not_a_native_preprocessing_dependency"})
    return changes


def _source(plan, native_package=None):
    launch, online, fresh = _modules()
    plan = _paths(plan)
    source = plan.get("preprocessing_source_root")
    work, target = plan["project_root"] / "work", plan["run_root"]
    if target == work or not target.is_relative_to(work):
        raise ValueError("The fresh preprocessing target must be an experiment inside project/work")
    if source is None or source == work or not source.is_relative_to(work):
        raise ValueError("Preprocessing source must be an explicit experiment inside project/work")
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Preprocessing derivative must not overlap its preserved source")
    launch._bound_files(source, [source / "launch_plan.json", source / "execution_journal.json"])
    saved_old = launch._read_json(source / "launch_plan.json")
    old = _paths(saved_old)
    for key, expected in (("project_root", plan["project_root"]), ("medical_root", plan["medical_root"]),
                          ("run_root", source), ("package_destination", source / "runtime/nnunetv2")):
        if old.get(key) != expected:
            raise ValueError(f"Source preprocessing experiment identity differs: {key}")
    for key in ("outer_fold", "dataset_id", "seed"):
        if old.get(key) != plan.get(key):
            raise ValueError(f"Source planned preprocessing identity differs: {key}")
    if old.get("upgrade_source_root") or old.get("preprocessing_source_root"):
        raise ValueError("Chained preprocessing derivations need an explicitly reviewed lineage")
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if Path(old["env_updates"][name]).resolve() != source / "online/nnunetv2" / name:
            raise ValueError(f"Source native output location differs: {name}")
    journal = launch._read_json(source / "execution_journal.json")
    checksum = journal.get("journal_sha256")
    body = {key: value for key, value in journal.items() if key != "journal_sha256"}
    if (journal.get("format") not in {"feedback_preparation_execution_v1", fresh.FORMAT}
            or checksum != launch._json_sha256(body)
            or journal.get("plan_sha256") != launch._json_sha256(saved_old)
            or not isinstance(journal.get("stages"), list)):
        raise ValueError("Source launch/journal is not checksum-bound")
    if any(not isinstance(row, dict) for row in journal["stages"]):
        raise ValueError("Malformed source stage journal")
    if any(row.get("status") == "running" for row in journal["stages"]):
        raise ValueError("Source execution is active or locked; preserve it and inspect its owner before reuse")
    persistent_lock = source / "feedback_execution.lock"
    if any(path != persistent_lock for path in source.glob("*execution.lock")):
        raise ValueError("A legacy source execution lock requires owner inspection before reuse")
    if persistent_lock.exists() or persistent_lock.is_symlink():
        # Probe the common OS lock without creating or rewriting OLD bytes.
        from tools.feedback_stage_execution import run_lock
        with run_lock(source, create=False):
            pass
    completed = [row for row in journal["stages"] if row.get("name") == "plan" and row.get("status") == "completed"]
    if len(completed) != 1 or not any(command.get("name") == "plan" for command in old["commands"]):
        raise ValueError("Source requires exactly one durably completed native plan stage")
    row = completed[0]
    registry_changes = _source_input_differences(old, row.get("input_files"))
    old_layout, new_layout = _layout(old), _layout(plan)
    raw = online.raw_dataset_dir(old_layout, plan["dataset_id"], plan["outer_fold"])
    pre = online.preprocessed_dataset_dir(old_layout, plan["dataset_id"], plan["outer_fold"])
    evidence = {"format": "feedback_stage_completion_v1", "files": launch._bound_files(
        source, launch._native_plan_evidence_files(old))}
    if row.get("execution_backend") == "injected_debug_runner":
        raise ValueError("Source completed preprocessing stage proof/configuration changed: DEBUG runner evidence cannot authorize reuse")
    recorded = row.get("completion_evidence")
    if recorded != evidence:
        recorded_files = recorded.get("files") if isinstance(recorded, dict) else None
        details = "malformed completion evidence"
        if isinstance(recorded_files, dict):
            expected_files = evidence["files"]
            details = (f"missing={sorted(set(expected_files) - set(recorded_files))}; "
                       f"unexpected={sorted(set(recorded_files) - set(expected_files))}; "
                       f"changed={sorted(key for key in set(recorded_files) & set(expected_files) if recorded_files[key] != expected_files[key])}")
        raise ValueError("Source completed preprocessing stage proof/configuration changed: "
                         "expected the native producer's exact four-file plan proof; " + details +
                         ". Original journal and files were not modified.")
    original, current = launch._read_json(old["train_config"]), launch._read_json(plan["train_config"])
    differences = configuration_differences(original, current)
    nn_cfg = launch._read_json(new_layout.nnunet_config)
    if launch._file_sha256(old_layout.nnunet_config) != launch._file_sha256(new_layout.nnunet_config):
        raise ValueError("Complete nnUNet preprocessing configuration differs")
    marker, split, cases = online._verified_preprocess_contract(
        old_layout, plan["outer_fold"], original, nn_cfg, plan["dataset_id"])
    # The native verifier hashes the current complete original cohort. Labels
    # are equal by the projection; another full raw SHA scan is redundant.
    # The actual NEW split is additionally checked at the worker boundary.
    native = fresh.native_inventory(old["package_destination"])
    saved_native = journal.get("runtime_inventory")
    if not isinstance(saved_native, dict) or any(saved_native.get(key) != value for key, value in native.items()):
        raise ValueError("Historical native implementation differs from its journal")
    native_status = "deferred_until_new_runtime_exists"
    if native_package is not None:
        if fresh.native_inventory(native_package) != native:
            raise ValueError("Native nnUNet implementation changed; preprocessing reuse is not authorized")
        native_status = "verified_equal_excluding_designated_cp_helpers"
    identity = {"source_root": str(source), "launch_plan_sha256": launch._json_sha256(saved_old),
                "journal_sha256": checksum, "plan_completion_sha256": launch._json_sha256(row),
                "train_config_sha256": launch._file_sha256(old["train_config"]),
                "raw_marker_sha256": launch._file_sha256(raw / online.RAW_MARKER_NAME),
                "preprocess_marker_sha256": launch._json_sha256(marker),
                "native_inventory_sha256": launch._json_sha256(native),
                "non_preprocessing_input_registry_differences": registry_changes}
    return plan, old, old_layout, new_layout, original, current, nn_cfg, marker, split, cases, identity, differences, native_status


def _file_rows(context):
    launch, online, _ = _modules()
    plan, old, old_layout, new_layout, _, _, nn_cfg, marker, _, cases, *_ = context
    fold, dataset = plan["outer_fold"], plan["dataset_id"]
    raw, target_raw = online.raw_dataset_dir(old_layout, dataset, fold), online.raw_dataset_dir(new_layout, dataset, fold)
    pre, target_pre = online.preprocessed_dataset_dir(old_layout, dataset, fold), online.preprocessed_dataset_dir(new_layout, dataset, fold)
    raw_contract = launch._read_json(raw / online.RAW_MARKER_NAME)
    if raw_contract["materialization"] not in {"copy", "hardlink", "symlink"}:
        raise ValueError("Unknown original raw materialization mode")
    outputs, rows = marker["outputs"], []
    identifier = outputs["data_identifier"]
    if not isinstance(identifier, str) or Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("Unsafe native preprocessing data identifier")

    def add(source, target, mode, expected=None, *, native_bound=True, allow_source_symlink=False):
        if not source.is_file() or (source.is_symlink() and not source.is_relative_to(raw) and not allow_source_symlink):
            raise ValueError(f"Missing/unverified native artifact: {source}")
        # Native input/output validation already hashed every bound byte.
        # GT is a v1-marker gap and is independently hashed below.
        actual = expected if expected is not None and native_bound else launch._file_sha256(source)
        if expected is not None and actual != expected:
            raise ValueError(f"Native manifest content changed: {source}")
        rows.append({"source": str(source), "target": str(target), "mode": mode,
                     "bytes": source.stat().st_size, "sha256": actual})

    records = {row["case_id"]: row for row in raw_contract["source_cases"]}
    if len(records) != len(cases) or len(records) != len(raw_contract["source_cases"]):
        raise ValueError("Duplicate or incomplete native raw case registry")
    for case in cases:
        name = case.case_id
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe native case identifier")
        for folder, suffix, key in (("imagesTr", "_0000.nii.gz", "image_sha256"),
                                    ("labelsTr", ".nii.gz", "label_sha256")):
            native_file = raw / folder / (name + suffix)
            original_file = case.image if key == "image_sha256" else case.label
            mode = raw_contract["materialization"]
            if ((mode == "symlink" and (not native_file.is_symlink() or native_file.resolve() != original_file.resolve()))
                    or (mode == "hardlink" and (native_file.is_symlink() or not native_file.samefile(original_file)))
                    or (mode == "copy" and (native_file.is_symlink() or native_file.samefile(original_file)))):
                raise ValueError("Source raw materialization no longer matches its native contract")
            add(raw / folder / (name + suffix), target_raw / folder / (name + suffix),
                raw_contract["materialization"], records[name][key])
    for name in ("dataset.json", online.RAW_MARKER_NAME):
        add(raw / name, target_raw / name, "copy")
    for name, key in ((f"{nn_cfg['dataset']['plans']}.json", "plans_sha256"),
                      ("dataset_fingerprint.json", "dataset_fingerprint_sha256"),
                      ("dataset.json", "dataset_json_sha256"), ("splits_final.json", "splits_final_sha256")):
        add(pre / name, target_pre / name, "copy", outputs[key])
    for record in outputs["cases"]:
        name = record["case_id"]
        suffix = ".npz" if outputs["storage_format"] == "npz" else ".b2nd"
        add(pre / identifier / (name + suffix), target_pre / identifier / (name + suffix), "hardlink", record["data_sha256"])
        if outputs["storage_format"] == "blosc2":
            add(pre / identifier / (name + "_seg.b2nd"), target_pre / identifier / (name + "_seg.b2nd"), "hardlink", record["segmentation_sha256"])
        add(pre / identifier / (name + ".pkl"), target_pre / identifier / (name + ".pkl"), "copy", record["properties_sha256"])
    # Native validation reads these labels. The v1 preprocess marker omits
    # them, so bind their complete cohort and bytes additionally to raw GT.
    gt = pre / "gt_segmentations"
    if not gt.is_dir() or gt.is_symlink() or {path.name for path in gt.iterdir()} != {case.case_id + ".nii.gz" for case in cases}:
        raise ValueError("Native validation ground truth is missing or not the complete original cohort")
    for case in cases:
        original_gt = gt / (case.case_id + ".nii.gz")
        if original_gt.is_symlink() and original_gt.resolve() != case.label.resolve():
            raise ValueError("Native ground-truth symlink does not identify its original case label")
        add(original_gt, target_pre / "gt_segmentations" / (case.case_id + ".nii.gz"),
            "hardlink", records[case.case_id]["label_sha256"], native_bound=False, allow_source_symlink=True)
    if len({row["target"] for row in rows}) != len(rows):
        raise ValueError("Duplicate native preprocessing derivative targets")
    return sorted(rows, key=lambda row: row["target"])


def _storage(plan, rows, *, enforce_available=True):
    ancestor = plan["run_root"]
    while not ancestor.exists():
        ancestor = ancestor.parent
    device, usage = ancestor.stat().st_dev, shutil.disk_usage(ancestor)
    unpack = 0
    for row in rows:
        source = Path(row["source"])
        if row["mode"] == "hardlink" and source.stat().st_dev != device:
            raise OSError("Native payload hardlink crosses filesystems; no silent full-data copy fallback")
        if source.suffix == ".npz":
            with zipfile.ZipFile(source) as archive:
                unpack += sum(archive.getinfo(name).file_size for name in ("data.npy", "seg.npy"))
    copies = sum(row["bytes"] for row in rows if row["mode"] == "copy" and not Path(row["target"]).exists())
    # Preserve the launcher's original native-preprocessing reserve, computed
    # from nnunet.runtime.minimum_free_gb_before_preprocess. No smaller fallback.
    reserve = plan.get("minimum_free_bytes")
    if type(reserve) is not int or reserve < 0:
        raise ValueError("An explicit native-preprocessing minimum_free_bytes plan value is required")
    required = reserve + copies + unpack
    if enforce_available and usage.free < required:
        raise OSError(f"Preprocessing derivative needs {required} free bytes; available={usage.free}")
    return {"storage_path": str(ancestor), "free_bytes": usage.free, "required_free_bytes": required,
            "minimum_reserve_bytes": reserve, "metadata_and_raw_copy_bytes": copies,
            "native_numpy_unpack_bytes_all_cases": unpack, "physical_payloads_shared": True}


def _verify_storage_receipt(plan, rows, recorded):
    expected = _storage(plan, rows, enforce_available=False)
    numbers = {"free_bytes", "required_free_bytes", "metadata_and_raw_copy_bytes", "materialization_workers"}
    fixed = {"storage_path", "minimum_reserve_bytes", "native_numpy_unpack_bytes_all_cases", "physical_payloads_shared"}
    if (not isinstance(recorded, dict) or set(recorded) != set(expected) | {"materialization_workers"}
            or any(recorded.get(key) != expected[key] for key in fixed)
            or any(type(recorded.get(key)) is not int or recorded[key] < 0 for key in numbers)
            or not 1 <= recorded["materialization_workers"] <= 32
            or recorded["metadata_and_raw_copy_bytes"] > sum(row["bytes"] for row in rows if row["mode"] == "copy")
            or recorded["required_free_bytes"] != (recorded["minimum_reserve_bytes"]
                + recorded["metadata_and_raw_copy_bytes"] + recorded["native_numpy_unpack_bytes_all_cases"])
            or recorded["free_bytes"] < recorded["required_free_bytes"]):
        raise ValueError("Storage reuse receipt arithmetic or native reserve differs")


def _attempt(plan, context, rows, *, publish=False, verify_existing=True):
    """Own a complete immutable derivative manifest before creating any file."""
    launch, online, _ = _modules()
    path = plan["run_root"] / "preprocessing_reuse/attempt.json"
    _safe_target(plan, path)
    expected = {"format": FORMAT + "_attempt", "dependency_projection": PROJECTION,
                "current_plan_sha256": launch._json_sha256(plan),
                "source_identity": context[10], "files": rows}
    if path.exists() or path.is_symlink():
        if path.is_symlink() or launch._read_json(path) != expected:
            raise ValueError("Existing preprocessing attempt ownership differs; preserved without overwrite")
        if verify_existing:
            for row in rows:
                target = Path(row["target"])
                if target.exists() or target.is_symlink():
                    _verify_row(plan, row)
        return expected
    # Even byte-identical files are foreign before this run owns its manifest.
    for directory in (online.raw_dataset_dir(context[3], plan["dataset_id"], plan["outer_fold"]),
                      online.preprocessed_dataset_dir(context[3], plan["dataset_id"], plan["outer_fold"])):
        _safe_target(plan, directory / "ownership_probe")
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise ValueError("Unowned preprocessing output exists; preserved without adoption")
        if directory.exists() and any(item.is_file() or item.is_symlink() for item in directory.rglob("*")):
            raise ValueError("Unowned preprocessing files exist; preserved without adoption")
    if publish:
        _publish_json(plan, path, expected)
    return expected


def validate_reuse(plan, native_package=None):
    """Read-only validation, including before NEW split/root/GPU allocation."""
    context = _source(plan, native_package)
    rows = _file_rows(context)
    _attempt(context[0], context, rows)
    return {"format": FORMAT, "dependency_projection": PROJECTION, "source_identity": context[10],
            "source_config_differences": context[11], "native_implementation_check": context[12],
            "actual_new_split_check": "required_at_plan_worker", "files": rows,
            "storage": _storage(context[0], rows), "source_outputs": context[7]["outputs"],
            "learned_artifacts_reused": False, "preprocessing_recomputed": False}


def _safe_target(plan, target):
    root = plan["run_root"]
    if not target.is_absolute() or not target.is_relative_to(root) or ".." in target.parts:
        raise ValueError(f"Escaping preprocessing derivative target: {target}")
    for directory in (target.parent, *target.parent.parents):
        if directory.is_symlink():
            raise ValueError(f"Preprocessing derivative directory is a symlink: {directory}")
        if directory == root:
            break


def _verify_row(plan, row, *, content=True):
    launch, _, _ = _modules()
    source, target = Path(row["source"]), Path(row["target"])
    _safe_target(plan, target)
    if (not source.is_file() or not target.is_file()
            or source.stat().st_size != row["bytes"] or target.stat().st_size != row["bytes"]):
        raise ValueError(f"Preprocessing derivative content changed: {target}")
    mode = row["mode"]
    if mode not in {"copy", "hardlink", "symlink"}:
        raise ValueError("Unknown native materialization mode")
    if ((mode == "symlink" and (not target.is_symlink() or target.resolve() != source.resolve()))
            or (mode == "hardlink" and (target.is_symlink() or not target.samefile(source)))
            or (mode == "copy" and (target.is_symlink() or target.samefile(source)))):
        raise ValueError(f"Preprocessing derivative materialization changed: {target}")
    if content and (launch._file_sha256(source) != row["sha256"]
                    or (not source.samefile(target) and launch._file_sha256(target) != row["sha256"])):
        raise ValueError(f"Preprocessing derivative content changed: {target}")


def _materialize(plan, row):
    launch, _, _ = _modules()
    source, target = Path(row["source"]), Path(row["target"])
    _safe_target(plan, target)
    if target.exists() or target.is_symlink():
        _verify_row(plan, row)
        return  # Exact partial-attempt recovery, never overwrite an existing file.
    # Source native SHA verification precedes this call, and complete source
    # plus NEW native hashes are revalidated before the success publication.
    target.parent.mkdir(parents=True, exist_ok=True)
    if row["mode"] == "hardlink":
        os.link(source.resolve(), target)
    elif row["mode"] == "symlink":
        target.symlink_to(source.resolve())
    elif row["mode"] == "copy":
        staging = plan["run_root"] / "preprocessing_reuse/copy_staging"
        temporary = staging / (uuid.uuid4().hex + ".tmp")
        _safe_target(plan, temporary)
        staging.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            with source.open("rb") as incoming, temporary.open("xb") as outgoing:
                created = True
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            os.link(temporary, target)  # Atomic, exclusive, and independent of OLD bytes.
        finally:
            if created and temporary.exists():
                temporary.unlink()  # Only this invocation's newly-created staging file.
    else:
        raise ValueError("Unknown native materialization mode")
    _verify_row(plan, row, content=False)


def _publish_json(plan, path, value):
    launch, _, _ = _modules()
    _safe_target(plan, path)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or launch._read_json(path) != value:
            raise ValueError(f"Existing mismatched publication preserved: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    created = False
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            created = True
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic and exclusive: never replace a user artifact.
    finally:
        if created and temporary.exists():
            temporary.unlink()  # Only this invocation's newly-created file.


def _current_marker(context):
    launch, online, _ = _modules()
    plan, _, _, layout, _, current, nn_cfg, original, _, _, *_ = context
    actual, split, cases = online._preprocess_input_contract(layout, plan["outer_fold"], current, nn_cfg, plan["dataset_id"])
    expected = {**original["input_contract"], "train_config_sha256": launch._file_sha256(plan["train_config"])}
    if actual != expected:
        raise ValueError("Current native preprocessing inputs differ beyond the recorded train-config dependency projection")
    outputs = online._preprocess_output_record(layout, plan["outer_fold"], nn_cfg, plan["dataset_id"], cases, split)
    if outputs != original["outputs"]:
        raise ValueError("Old and NEW complete preprocessing output manifests differ")
    return online._preprocess_marker_payload(actual, outputs)


def verify_reuse(plan):
    """Revalidate both source provenance and complete NEW native derivatives."""
    launch, online, _ = _modules()
    plan = _paths(plan)
    context = _source(plan, plan["package_destination"])
    path = plan["run_root"] / "preprocessing_reuse/receipt.json"
    receipt = launch._read_json(path)
    rows, marker = _file_rows(context), _current_marker(context)
    attempt_path = plan["run_root"] / "preprocessing_reuse/attempt.json"
    if not attempt_path.is_file():
        raise ValueError("Preprocessing receipt lacks an owned attempt manifest")
    _attempt(plan, context, rows, verify_existing=False)
    wanted = {"format": FORMAT, "dependency_projection": PROJECTION,
              "source_identity": context[10], "source_config_differences": context[11],
              "current_plan_sha256": launch._json_sha256(plan),
              "native_implementation_check": context[12], "files": rows,
              "source_output_manifest_sha256": launch._json_sha256(context[7]["outputs"]),
              "current_output_manifest_sha256": launch._json_sha256(marker["outputs"]),
              "current_preprocess_marker_sha256": launch._json_sha256(marker),
              "learned_artifacts_reused": False, "preprocessing_recomputed": False}
    if not isinstance(receipt, dict) or set(receipt) != set(wanted) | {"storage"} or any(receipt.get(key) != value for key, value in wanted.items()):
        raise ValueError("Preprocessing reuse receipt no longer matches its complete source/current contracts")
    _verify_storage_receipt(plan, rows, receipt["storage"])
    for row in rows:
        # _current_marker just rehashed all native inputs/outputs. GT is the
        # explicitly bound marker gap; materialization identity is always read.
        _verify_row(plan, row, content="gt_segmentations" in Path(row["target"]).parts)
    marker_path = online.preprocessed_dataset_dir(context[3], plan["dataset_id"], plan["outer_fold"]) / online.PREPROCESS_MARKER_NAME
    if launch._read_json(marker_path) != marker:
        raise ValueError("Current native preprocessing publication differs from reuse proof")
    return receipt


def execute_reuse(plan):
    launch, online, fresh = _modules()
    plan = _paths(plan)
    root = plan["run_root"]
    launch._bound_files(root, [root / "launch_plan.json", root / "execution_journal.json"])
    journal = launch._read_json(root / "execution_journal.json")
    body = {key: value for key, value in journal.items() if key != "journal_sha256"}
    if (journal.get("format") != fresh.FORMAT
            or not isinstance(journal.get("stages"), list)
            or any(not isinstance(row, dict) for row in journal["stages"])
            or journal.get("journal_sha256") != launch._json_sha256(body)
            or journal.get("plan_sha256") != launch._json_sha256(plan)
            or launch._json_sha256(launch._read_json(root / "launch_plan.json")) != launch._json_sha256(plan)
            or journal.get("input_files") != launch._resume_inputs(plan)):
        raise ValueError("NEW fresh preprocessing worker lacks an unchanged launch/journal/input binding")
    split_rows = [row for row in journal["stages"] if row.get("name") == "split" and row.get("status") == "completed"]
    if len(split_rows) != 1 or split_rows[0].get("completion_evidence") != launch._stage_evidence(plan, "split"):
        raise ValueError("NEW split stage is not durably complete")
    context = _source(plan, plan["package_destination"])
    if launch._file_sha256(context[2].outer_splits) != launch._file_sha256(context[3].outer_splits):
        raise ValueError("NEW actual outer split file differs from verified source preprocessing")
    receipt_path = root / "preprocessing_reuse/receipt.json"
    if receipt_path.exists():
        return verify_reuse(plan)
    rows = _file_rows(context)
    _attempt(plan, context, rows, publish=True)
    storage = _storage(plan, rows)
    # Independent native files share one measured-size, CPU-bounded I/O pool;
    # there is no graph/candidate/sample reduction or full-volume RAM copy.
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    workers = max(1, min(32, cores))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda row: _materialize(plan, row), rows))
    raw = online.raw_dataset_dir(context[3], plan["dataset_id"], plan["outer_fold"])
    (raw / "imagesTs").mkdir(exist_ok=True)
    marker = _current_marker(context)
    marker_path = online.preprocessed_dataset_dir(context[3], plan["dataset_id"], plan["outer_fold"]) / online.PREPROCESS_MARKER_NAME
    # Revalidate original source after I/O before publishing any successful receipt.
    if _source(plan, plan["package_destination"])[10] != context[10]:
        raise ValueError("Source preprocessing identity changed during derivative materialization")
    for row in rows:
        _verify_row(plan, row, content="gt_segmentations" in Path(row["target"]).parts)
    _publish_json(plan, marker_path, marker)
    receipt = {"format": FORMAT, "dependency_projection": PROJECTION,
               "source_identity": context[10], "source_config_differences": context[11],
               "current_plan_sha256": launch._json_sha256(plan),
               "native_implementation_check": context[12], "files": rows,
               "source_output_manifest_sha256": launch._json_sha256(context[7]["outputs"]),
               "current_output_manifest_sha256": launch._json_sha256(marker["outputs"]),
               "current_preprocess_marker_sha256": launch._json_sha256(marker),
               "learned_artifacts_reused": False, "preprocessing_recomputed": False,
               "storage": {**storage, "materialization_workers": workers}}
    _publish_json(plan, receipt_path, receipt)
    # All native bytes and additional GT/link identities were just checked.
    # The stage consumer independently calls verify_reuse; do not hash the
    # complete cohort redundantly a third time inside this worker.
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args(argv)
    launch, _, _ = _modules()
    root = Path(args.experiment_root).resolve()
    plan = _paths(launch._read_json(root / "launch_plan.json"))
    if plan["run_root"] != root or plan.get("preprocessing_source_root") != Path(args.source_root).resolve():
        raise ValueError("Worker arguments differ from the bound fresh launch")
    receipt = execute_reuse(plan)
    print(json.dumps({"format": receipt["format"], "receipt": str(root / "preprocessing_reuse/receipt.json"),
                      "preprocessing_recomputed": False, "learned_artifacts_reused": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
