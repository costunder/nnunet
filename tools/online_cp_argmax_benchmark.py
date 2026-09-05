"""Leakage-safe online Basic-CP versus exact-argmax HierCP benchmark.

The benchmark uses one original-only nnU-Net dataset and the same local patient
fold for both methods. For every eligible source tumor, a fold-specific HierCP
model scores multiple independent pools of hard-valid candidate positions.
During nnU-Net training Copy-Paste is applied online:

- Basic-CP selects one candidate uniformly from the selected shared pool.
- HierCP selects the exact GNN argmax from that same selected pool.

The source schedule, proposal-pool schedule, CP-event schedule, appearance
jitter, standard nnU-Net augmentation RNG, plans, network initialization,
250-epoch LR horizon and original-only validation set are identical. There is
no top-k random sampling in the HierCP arm.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.stats import binom, wilcoxon

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

VERSION = "online_basic_hiercp_argmax_v3"
BANK_FORMAT = "hiercp_online_bank_argmax_v3"
DEFAULT_DATASET_ID = 740
DEFAULT_PAIRED_ROOT = "paired_basic_vs_hiercp"
DEFAULT_ONLINE_ROOT = "online_basic_vs_hiercp"
TRAINER_BASIC = "nnUNetTrainer_250epochs_OnlineBasicCPSharedPoolsV3"
TRAINER_HIER = "nnUNetTrainer_250epochs_OnlineHierCPArgmaxV3"
RAW_MARKER_NAME = "online_cp_dataset.json"
PREPROCESS_MARKER_NAME = "online_cp_preprocess_complete.json"
TRAIN_CONTRACT_NAME = "training_argmax_v3_contract.json"
TRAIN_COMPLETE_NAME = "training_argmax_v3_complete.json"
EVALUATION_COMPLETE_NAME = "complete.json"
GNN_SPLIT_FORMAT = "hiercp_case_split_v1"
OUTER_SPLIT_FORMAT = "paired_cp_outer_split_v1"


class OnlineBenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class Case:
    case_id: str
    image: Path
    label: Path


@dataclass(frozen=True)
class Layout:
    project: Path
    medical: Path
    data: Path
    paired: Path
    online: Path
    source_work: Path
    train_config: Path
    nnunet_config: Path
    outer_splits: Path
    nnroot: Path
    raw: Path
    preprocessed: Path
    results: Path
    logs: Path

    def paired_fold(self, outer_fold: int) -> Path:
        return self.paired / "folds" / f"fold_{outer_fold}"

    def online_fold(self, outer_fold: int) -> Path:
        return self.online / "folds" / f"fold_{outer_fold}"

    def gnn(self, outer_fold: int) -> Path:
        return self.paired_fold(outer_fold) / "gnn"

    def bank(self, outer_fold: int) -> Path:
        return self.online_fold(outer_fold) / "bank_argmax_v3"

    def evaluation(self, outer_fold: int) -> Path:
        return self.online_fold(outer_fold) / "evaluation_argmax_v3"


def natural_key(text: str) -> list[object]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OnlineBenchmarkError(f"Missing JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OnlineBenchmarkError(f"Invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OnlineBenchmarkError(f"JSON root must be an object: {path}")
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_seed(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OnlineBenchmarkError(f"Missing JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OnlineBenchmarkError(f"Invalid JSON {path}: {exc}") from exc


def equivalent_diameter(volume_mm3: float) -> float:
    if volume_mm3 <= 0:
        return 0.0
    return float(2.0 * (3.0 * float(volume_mm3) / (4.0 * math.pi)) ** (1.0 / 3.0))


def discover_cases(root: Path) -> list[Case]:
    image_dir = root / "image"
    label_dir = root / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise OnlineBenchmarkError(f"Expected image/ and labels/ under {root}")
    cases: list[Case] = []
    for image in sorted(image_dir.glob("*_0000.nii.gz"), key=lambda value: natural_key(value.name)):
        case_id = image.name[: -len("_0000.nii.gz")]
        label = label_dir / f"{case_id}.nii.gz"
        if not label.is_file():
            raise OnlineBenchmarkError(f"Missing label for {case_id}: {label}")
        cases.append(Case(case_id, image.resolve(), label.resolve()))
    if not cases:
        raise OnlineBenchmarkError(f"No image/label pairs found under {root}")
    image_ids = {case.case_id for case in cases}
    label_ids = {
        path.name[: -len(".nii.gz")]
        for path in label_dir.glob("*.nii.gz")
        if path.is_file()
    }
    if image_ids != label_ids:
        raise OnlineBenchmarkError(
            "Original image/label cohorts differ: "
            f"images_without_labels={sorted(image_ids - label_ids, key=natural_key)} "
            f"labels_without_images={sorted(label_ids - image_ids, key=natural_key)}"
        )
    return cases


def load_3d(path: Path, dtype: Any) -> tuple[Any, np.ndarray]:
    import nibabel as nib

    nii = nib.load(str(path))
    view = nii
    if len(view.shape) == 4:
        if view.shape[-1] <= 4:
            view = view.slicer[..., 0]
        elif view.shape[0] <= 4:
            view = view.slicer[0, ...]
        else:
            raise OnlineBenchmarkError(f"Cannot infer channel axis: {path} shape={view.shape}")
    if len(view.shape) != 3:
        raise OnlineBenchmarkError(f"Expected 3D NIfTI: {path} shape={view.shape}")
    return view, np.asarray(view.dataobj, dtype=dtype, order="C")


def _assert_same_nifti_geometry(
    reference_nii: Any,
    reference: np.ndarray,
    prediction_nii: Any,
    prediction: np.ndarray,
    prediction_path: Path,
) -> None:
    reference_shape = tuple(int(value) for value in reference.shape)
    prediction_shape = tuple(int(value) for value in prediction.shape)
    if prediction_shape != reference_shape:
        raise OnlineBenchmarkError(
            f"Prediction voxel shape differs from reference for {prediction_path}: "
            f"prediction={prediction_shape} reference={reference_shape}"
        )
    reference_affine = np.asarray(reference_nii.affine, dtype=np.float64)
    prediction_affine = np.asarray(prediction_nii.affine, dtype=np.float64)
    if (
        reference_affine.shape != (4, 4)
        or prediction_affine.shape != (4, 4)
        or not bool(np.all(np.isfinite(reference_affine)))
        or not bool(np.all(np.isfinite(prediction_affine)))
        or not bool(
            np.allclose(
                prediction_affine,
                reference_affine,
                rtol=0.0,
                atol=1e-5,
            )
        )
    ):
        raise OnlineBenchmarkError(
            f"Prediction affine differs from reference for {prediction_path}"
        )
    reference_zooms = np.asarray(reference_nii.header.get_zooms()[:3], dtype=np.float64)
    prediction_zooms = np.asarray(prediction_nii.header.get_zooms()[:3], dtype=np.float64)
    if (
        reference_zooms.shape != (3,)
        or prediction_zooms.shape != (3,)
        or not bool(np.all(np.isfinite(reference_zooms)))
        or not bool(np.all(np.isfinite(prediction_zooms)))
        or bool(np.any(reference_zooms <= 0.0))
        or bool(np.any(prediction_zooms <= 0.0))
        or not bool(
            np.allclose(
                prediction_zooms,
                reference_zooms,
                rtol=0.0,
                atol=1e-6,
            )
        )
    ):
        raise OnlineBenchmarkError(
            f"Prediction voxel spacing differs from reference for {prediction_path}"
        )


def _assert_label_domain(
    array: np.ndarray, allowed_labels: set[int], context: str
) -> None:
    observed = {int(value) for value in np.unique(array).tolist()}
    unexpected = sorted(observed - allowed_labels)
    if unexpected:
        raise OnlineBenchmarkError(
            f"Unexpected segmentation labels for {context}: {unexpected}; "
            f"allowed={sorted(allowed_labels)}"
        )


def require_command(command: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise OnlineBenchmarkError(f"Required command not found: {command}")
    return resolved


def run_command(
    values: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    log: Path | None = None,
    dry_run: bool = False,
) -> None:
    command = [str(value) for value in values]
    print("\n$ " + " ".join(command), flush=True)
    if dry_run:
        return
    handle = None
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=None if env is None else dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        assert process.stdout is not None
        terminal = getattr(sys.stdout, "buffer", None)
        while True:
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                break
            if terminal is not None:
                terminal.write(chunk)
                terminal.flush()
            else:
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            if handle is not None:
                handle.write(chunk)
                handle.flush()
        return_code = process.wait()
    finally:
        if handle is not None:
            handle.close()
    if return_code != 0:
        raise OnlineBenchmarkError(f"Command failed ({return_code}): {' '.join(command)}")


def materialize(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "symlink":
        target.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source.resolve(), target)
    elif mode == "copy":
        shutil.copy2(source.resolve(), target)
    else:
        raise OnlineBenchmarkError(f"Unsupported materialization mode: {mode}")


def outer_split(layout: Layout, outer_fold: int) -> dict[str, Any]:
    payload = load_json(layout.outer_splits)
    if payload.get("format") != OUTER_SPLIT_FORMAT:
        raise OnlineBenchmarkError(
            f"Unsupported outer split format in {layout.outer_splits}: "
            f"{payload.get('format')!r}"
        )
    splits = payload.get("splits")
    if not isinstance(splits, list) or not 0 <= int(outer_fold) < len(splits):
        raise OnlineBenchmarkError(f"Outer fold {outer_fold} is unavailable in {layout.outer_splits}")
    split = splits[int(outer_fold)]
    if not isinstance(split, dict) or not isinstance(split.get("train"), list) or not isinstance(split.get("val"), list):
        raise OnlineBenchmarkError(f"Malformed outer split {outer_fold}")
    stored_fold = split.get("fold")
    if (
        not isinstance(stored_fold, int)
        or isinstance(stored_fold, bool)
        or stored_fold != int(outer_fold)
    ):
        raise OnlineBenchmarkError(
            f"Outer split fold label mismatch: requested={outer_fold} stored={stored_fold!r}"
        )
    train = [str(value) for value in split["train"]]
    val = [str(value) for value in split["val"]]
    if not train or not val:
        raise OnlineBenchmarkError("Outer train and validation cohorts must both be non-empty")
    if len(train) != len(set(train)) or len(val) != len(set(val)):
        raise OnlineBenchmarkError("Outer split contains duplicate patient IDs")
    if any(not case_id.strip() for case_id in (*train, *val)):
        raise OnlineBenchmarkError("Outer split contains an empty patient ID")
    if set(train) & set(val):
        raise OnlineBenchmarkError("Outer train/validation patient leakage detected")
    return {**split, "train": train, "val": val}


def _verified_gnn_split(layout: Layout, outer_fold: int) -> dict[str, Any]:
    outer = outer_split(layout, outer_fold)
    path = layout.gnn(outer_fold) / "split.json"
    payload = load_json(path)
    if payload.get("format") != GNN_SPLIT_FORMAT:
        raise OnlineBenchmarkError(
            f"Unsupported fold-specific GNN split format: {path}"
        )
    stored_fold = payload.get("outer_fold")
    if (
        not isinstance(stored_fold, int)
        or isinstance(stored_fold, bool)
        or stored_fold != int(outer_fold)
    ):
        raise OnlineBenchmarkError(
            f"Fold-specific GNN split outer_fold mismatch: {path}"
        )

    def cohort(name: str) -> list[str]:
        values = payload.get(name)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) for value in values)
        ):
            raise OnlineBenchmarkError(
                f"Fold-specific GNN {name} cohort must be a non-empty list[str]: {path}"
            )
        result = list(values)
        if any(not value.strip() for value in result):
            raise OnlineBenchmarkError(
                f"Fold-specific GNN {name} cohort contains an empty ID: {path}"
            )
        if len(result) != len(set(result)):
            raise OnlineBenchmarkError(
                f"Fold-specific GNN {name} cohort contains duplicates: {path}"
            )
        return result

    train = cohort("train")
    val = cohort("val")
    if set(train) & set(val):
        raise OnlineBenchmarkError(f"Fold-specific GNN train/val leakage: {path}")
    if set(train) | set(val) != set(outer["train"]):
        raise OnlineBenchmarkError(
            "Fold-specific GNN inner cohorts do not exactly partition the current "
            f"outer training cohort: {path}"
        )
    excluded = payload.get("outer_validation_excluded")
    if (
        not isinstance(excluded, list)
        or not all(isinstance(value, str) for value in excluded)
        or excluded != list(outer["val"])
    ):
        raise OnlineBenchmarkError(
            "Fold-specific GNN split does not record the exact current outer "
            f"validation exclusion: {path}"
        )
    return {**payload, "train": train, "val": val}


def _verify_gnn_training_cohort(
    split: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    prototype_training_cases: Sequence[str],
) -> None:
    expected = list(split["train"])
    checkpoint_cases = checkpoint.get("prototype_training_cases")
    if (
        not isinstance(checkpoint_cases, list)
        or not all(isinstance(value, str) for value in checkpoint_cases)
        or checkpoint_cases != expected
    ):
        raise OnlineBenchmarkError(
            "GNN checkpoint prototype-training cohort does not exactly match the "
            "verified inner training cohort"
        )
    prototype_cases = list(prototype_training_cases)
    if prototype_cases != expected:
        raise OnlineBenchmarkError(
            "Prototype bank training cohort does not exactly match the verified "
            "inner training cohort"
        )


def _verified_gnn_causality(layout: Layout, outer_fold: int) -> dict[str, Any]:
    try:
        from hiercp.cache import validate_cache_publication
        from hiercp.model import HierarchicalPyGPlacementModel
        from hiercp.tensor import torch_load_compat
        from tools.causality import (
            REPORT_FORMAT,
            _artifact_contract,
            _cache_entries_for_split,
            _input_contract,
            _strict_failures,
            _training_measurement_plan,
            _validate_checkpoint_contract,
            _validate_preflight_record,
            _validate_reusable_report,
            _value_sha256,
        )
    except ImportError as exc:
        raise OnlineBenchmarkError(
            f"Cannot load the strict causality verifier: {exc}"
        ) from exc

    gnn_root = layout.gnn(outer_fold)
    graph_root = gnn_root / "graphs"
    checkpoint_path = (gnn_root / "model.pt").resolve()
    prototype_path = (gnn_root / "prototype.pt").resolve()
    report_path = gnn_root / "causality.json"
    preflight_path = report_path.with_name(report_path.name + ".preflight.json")
    gnn_split = _verified_gnn_split(layout, outer_fold)
    try:
        graph_index = validate_cache_publication(graph_root)
        graph_config = load_json(graph_root / "config.json")
        expected_selected = [*gnn_split["train"], *gnn_split["val"]]
        if (
            graph_config.get("run_mode") != "benchmark"
            or graph_config.get("subset_active") is not False
            or graph_config.get("train_case_ids") != gnn_split["train"]
            or graph_config.get("val_case_ids") != gnn_split["val"]
            or graph_config.get("selected_case_ids") != expected_selected
        ):
            raise ValueError(
                "Graph-cache split/subset contract differs from the verified inner split"
            )
        checkpoint = torch_load_compat(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("model_kwargs"), dict
        ):
            raise ValueError("Fold-specific checkpoint has no model contract")
        model = HierarchicalPyGPlacementModel(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint.get("state_dict"))
        signature = _validate_checkpoint_contract(
            checkpoint,
            checkpoint_path=checkpoint_path,
            prototype_path=prototype_path,
            cache_dir=graph_root.resolve(),
            cache_config=graph_config,
            cache_index=graph_index,
            model=model,
            run_mode="benchmark",
        )
        selected, selected_entries = _cache_entries_for_split(
            graph_root.resolve(), graph_index, "val"
        )
        expected_seed = int(load_json(layout.train_config).get("seed", 42)) + int(
            outer_fold
        )
        artifact_contract = _artifact_contract(
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            prototype_path=prototype_path,
            signature=signature,
            cache_dir=graph_root.resolve(),
            cache_config=graph_config,
            selected_entries=selected_entries,
            run_mode="benchmark",
            split="val",
            max_batches=0,
            seed=expected_seed,
            permutation_tolerance=1.0e-4,
            response_threshold=1.0e-4,
            strict=True,
        )
        report = load_json(report_path)
        preflight = load_json(preflight_path)
        identity = preflight.get("identity")
        resources = preflight.get("resource_fingerprint")
        if not isinstance(identity, dict) or not isinstance(resources, dict):
            raise ValueError("Causality preflight has no identity/resource contract")
        measurement_plan = _training_measurement_plan(
            checkpoint,
            checkpoint_path=checkpoint_path,
            cache_dir=graph_root.resolve(),
            run_mode="benchmark",
        )
        repeats = identity.get("repeats")
        if (
            identity.get("format") != "hiercp_causality_preflight_identity_v1"
            or identity.get("artifact_contract_sha256")
            != _value_sha256(artifact_contract)
            or identity.get("measurement_plan") != measurement_plan
            or not isinstance(repeats, int)
            or isinstance(repeats, bool)
            or repeats < 3
        ):
            raise ValueError("Causality preflight identity is stale or incomplete")
        batch_size, num_workers = _validate_preflight_record(
            preflight, identity=identity, resource_fingerprint=resources
        )
        current_input = _input_contract(
            artifact_contract,
            preflight_path=preflight_path.resolve(),
            preflight=preflight,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if (
            report.get("format") != REPORT_FORMAT
            or report.get("status") != "complete"
            or report.get("strict_pass") is not True
            or report.get("input_contract") != current_input
            or report.get("input_contract_sha256") != _value_sha256(current_input)
            or report.get("checkpoint") != str(checkpoint_path)
            or report.get("split") != "val"
            or report.get("cache_files") != len(selected)
            or report.get("thresholds")
            != {
                "permutation_tolerance": 1.0e-4,
                "response_threshold": 1.0e-4,
            }
        ):
            raise ValueError("Causality v3 report is stale or not a strict full-val audit")
        _validate_reusable_report(
            report,
            input_contract=current_input,
            expected_samples=len(selected),
            expected_batches=math.ceil(len(selected) / batch_size),
        )
        failures = _strict_failures(report["verdict"])
        if failures:
            raise ValueError(f"Causality strict verdict failed: {failures}")
        resource_report = report.get("resource_preflight")
        if (
            not isinstance(resource_report, dict)
            or resource_report.get("path") != str(preflight_path.resolve())
            or resource_report.get("artifact_sha256") != file_sha256(preflight_path)
            or resource_report.get("contract_sha256") != _value_sha256(preflight)
            or resource_report.get("physical_batch_size") != batch_size
            or resource_report.get("num_workers") != num_workers
            or resource_report.get("batch_trials") != preflight.get("batch_trials")
            or resource_report.get("worker_trials") != preflight.get("worker_trials")
        ):
            raise ValueError("Causality resource/preflight report is inconsistent")
        if validate_cache_publication(graph_root) != graph_index:
            raise ValueError("Graph-cache publication changed during causality verification")
        if _artifact_contract(
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            prototype_path=prototype_path,
            signature=signature,
            cache_dir=graph_root.resolve(),
            cache_config=load_json(graph_root / "config.json"),
            selected_entries=_cache_entries_for_split(
                graph_root.resolve(), validate_cache_publication(graph_root), "val"
            )[1],
            run_mode="benchmark",
            split="val",
            max_batches=0,
            seed=expected_seed,
            permutation_tolerance=1.0e-4,
            response_threshold=1.0e-4,
            strict=True,
        ) != artifact_contract:
            raise ValueError("GNN artifacts changed during causality verification")
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise OnlineBenchmarkError(
            f"Fold-specific GNN causality v3 contract is invalid: {exc}"
        ) from exc
    return report


def dataset_name(dataset_id: int, outer_fold: int) -> str:
    return f"Dataset{int(dataset_id):03d}_LiverOnlineCP_OF{int(outer_fold)}"


def raw_dataset_dir(layout: Layout, dataset_id: int, outer_fold: int) -> Path:
    return layout.raw / dataset_name(dataset_id, outer_fold)


def preprocessed_dataset_dir(layout: Layout, dataset_id: int, outer_fold: int) -> Path:
    return layout.preprocessed / dataset_name(dataset_id, outer_fold)


def _case_cohort(
    layout: Layout, outer_fold: int
) -> tuple[dict[str, Any], list[Case]]:
    split = outer_split(layout, outer_fold)
    cases = discover_cases(layout.data)
    actual = {case.case_id for case in cases}
    expected = set(split["train"]) | set(split["val"])
    if actual != expected:
        raise OnlineBenchmarkError(
            "Outer split does not cover exactly the original dataset: "
            f"missing={sorted(expected - actual, key=natural_key)} "
            f"unexpected={sorted(actual - expected, key=natural_key)}"
        )
    return split, cases


def _dataset_json_payload(
    train_cfg: Mapping[str, Any], cases: Sequence[Case]
) -> dict[str, Any]:
    return {
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0,
            "liver": int(train_cfg["labels"]["liver"]),
            "tumor": int(train_cfg["labels"]["tumor"]),
        },
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
    }


def _raw_contract(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    dataset_id: int,
    materialization: str,
) -> tuple[dict[str, Any], list[Case], dict[str, Any], dict[str, Any]]:
    if materialization not in {"symlink", "hardlink", "copy"}:
        raise OnlineBenchmarkError(
            f"Unsupported materialization mode in raw contract: {materialization!r}"
        )
    split, cases = _case_cohort(layout, outer_fold)
    dataset_payload = _dataset_json_payload(train_cfg, cases)
    source_cases = [
        {
            "case_id": case.case_id,
            "image_sha256": file_sha256(case.image),
            "label_sha256": file_sha256(case.label),
        }
        for case in cases
    ]
    contract = {
        "format": "hiercp_online_raw_contract_v1",
        "version": VERSION,
        "dataset_id": int(dataset_id),
        "dataset_name": dataset_name(dataset_id, outer_fold),
        "outer_fold": int(outer_fold),
        "materialization": materialization,
        "outer_splits_sha256": file_sha256(layout.outer_splits),
        "dataset_json_sha256": value_sha256(dataset_payload),
        "original_cases": len(cases),
        "train_ids": list(split["train"]),
        "val_ids": list(split["val"]),
        "source_cases": source_cases,
    }
    return split, cases, dataset_payload, contract


def _audit_raw_dataset(
    target: Path,
    cases: Sequence[Case],
    dataset_payload: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    marker = target / RAW_MARKER_NAME
    if not target.is_dir() or target.is_symlink():
        raise OnlineBenchmarkError(f"Raw dataset is not a real directory: {target}")
    expected_root = {
        "imagesTr",
        "labelsTr",
        "imagesTs",
        "dataset.json",
        RAW_MARKER_NAME,
    }
    actual_root = {path.name for path in target.iterdir()}
    if actual_root != expected_root:
        raise OnlineBenchmarkError(
            "Raw dataset root contains a partial or unexpected file set: "
            f"missing={sorted(expected_root - actual_root)} "
            f"unexpected={sorted(actual_root - expected_root)}"
        )
    if load_json(marker) != dict(contract):
        raise OnlineBenchmarkError(
            f"Raw dataset provenance marker is stale or incompatible: {marker}"
        )
    if load_json(target / "dataset.json") != dict(dataset_payload):
        raise OnlineBenchmarkError(f"Raw dataset.json is stale or incompatible: {target}")
    images = target / "imagesTr"
    labels = target / "labelsTr"
    tests = target / "imagesTs"
    if not images.is_dir() or not labels.is_dir() or not tests.is_dir():
        raise OnlineBenchmarkError(f"Raw dataset directories are incomplete: {target}")
    expected_images = {f"{case.case_id}_0000.nii.gz" for case in cases}
    expected_labels = {f"{case.case_id}.nii.gz" for case in cases}
    actual_images = {path.name for path in images.iterdir()}
    actual_labels = {path.name for path in labels.iterdir()}
    actual_tests = {path.name for path in tests.iterdir()}
    if actual_images != expected_images or actual_labels != expected_labels or actual_tests:
        raise OnlineBenchmarkError(
            "Raw materialized cohort is not exact: "
            f"images_missing={sorted(expected_images - actual_images)} "
            f"images_extra={sorted(actual_images - expected_images)} "
            f"labels_missing={sorted(expected_labels - actual_labels)} "
            f"labels_extra={sorted(actual_labels - expected_labels)} "
            f"imagesTs={sorted(actual_tests)}"
        )
    records = {
        str(row["case_id"]): row
        for row in contract.get("source_cases", [])
        if isinstance(row, dict) and "case_id" in row
    }
    if set(records) != {case.case_id for case in cases}:
        raise OnlineBenchmarkError(f"Raw source hash inventory is malformed: {marker}")
    for case in cases:
        record = records[case.case_id]
        image_path = images / f"{case.case_id}_0000.nii.gz"
        label_path = labels / f"{case.case_id}.nii.gz"
        if file_sha256(image_path) != record.get("image_sha256"):
            raise OnlineBenchmarkError(f"Raw image content hash mismatch: {image_path}")
        if file_sha256(label_path) != record.get("label_sha256"):
            raise OnlineBenchmarkError(f"Raw label content hash mismatch: {label_path}")


def _verified_raw_contract(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    dataset_id: int,
) -> tuple[dict[str, Any], dict[str, Any], list[Case]]:
    target = raw_dataset_dir(layout, dataset_id, outer_fold)
    marker_path = target / RAW_MARKER_NAME
    marker = load_json(marker_path)
    materialization = str(marker.get("materialization", ""))
    split, cases, dataset_payload, wanted = _raw_contract(
        layout, outer_fold, train_cfg, dataset_id, materialization
    )
    _audit_raw_dataset(target, cases, dataset_payload, wanted)
    return wanted, split, cases


def _preprocess_input_contract(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
    dataset_id: int,
) -> tuple[dict[str, Any], dict[str, Any], list[Case]]:
    raw_contract, split, cases = _verified_raw_contract(
        layout, outer_fold, train_cfg, dataset_id
    )
    value = {
        "format": "hiercp_online_preprocess_input_v1",
        "version": VERSION,
        "dataset_id": int(dataset_id),
        "dataset_name": dataset_name(dataset_id, outer_fold),
        "outer_fold": int(outer_fold),
        "raw_marker_sha256": file_sha256(
            raw_dataset_dir(layout, dataset_id, outer_fold) / RAW_MARKER_NAME
        ),
        "raw_contract_sha256": value_sha256(raw_contract),
        "outer_splits_sha256": file_sha256(layout.outer_splits),
        "train_config_sha256": file_sha256(layout.train_config),
        "nnunet_config_sha256": file_sha256(layout.nnunet_config),
        "planner": str(nn_cfg["dataset"]["planner"]),
        "plans": str(nn_cfg["dataset"]["plans"]),
        "configuration": str(nn_cfg["dataset"]["configuration"]),
        "case_ids": [case.case_id for case in cases],
        "train_ids": list(split["train"]),
        "val_ids": list(split["val"]),
    }
    return value, split, cases


def _preprocess_output_record(
    layout: Layout,
    outer_fold: int,
    nn_cfg: Mapping[str, Any],
    dataset_id: int,
    cases: Sequence[Case],
    split: Mapping[str, Any],
) -> dict[str, Any]:
    preprocessed = preprocessed_dataset_dir(layout, dataset_id, outer_fold)
    plans_name = str(nn_cfg["dataset"]["plans"])
    configuration = str(nn_cfg["dataset"]["configuration"])
    plans_path = preprocessed / f"{plans_name}.json"
    plans = load_json(plans_path)
    try:
        identifier = str(plans["configurations"][configuration]["data_identifier"])
    except (KeyError, TypeError) as exc:
        raise OnlineBenchmarkError(
            f"Plans do not define configuration {configuration!r}: {plans_path}"
        ) from exc
    data_root = preprocessed / identifier
    if not data_root.is_dir():
        raise OnlineBenchmarkError(f"Online preprocessing output missing: {data_root}")
    expected_ids = {case.case_id for case in cases}
    npz = {path.name[: -len(".npz")]: path for path in data_root.glob("*.npz")}
    properties = {path.name[: -len(".pkl")]: path for path in data_root.glob("*.pkl")}
    if set(npz) != expected_ids or set(properties) != expected_ids:
        raise OnlineBenchmarkError(
            "Preprocessed case cohort is not exact: "
            f"npz_missing={sorted(expected_ids - set(npz), key=natural_key)} "
            f"npz_extra={sorted(set(npz) - expected_ids, key=natural_key)} "
            f"pkl_missing={sorted(expected_ids - set(properties), key=natural_key)} "
            f"pkl_extra={sorted(set(properties) - expected_ids, key=natural_key)}"
        )
    desired_split = [{"train": list(split["train"]), "val": list(split["val"])}]
    split_path = preprocessed / "splits_final.json"
    if load_json_value(split_path) != desired_split:
        raise OnlineBenchmarkError(f"Preprocessed split contract mismatch: {split_path}")
    fingerprint_path = preprocessed / "dataset_fingerprint.json"
    dataset_path = preprocessed / "dataset.json"
    if not fingerprint_path.is_file() or not dataset_path.is_file():
        raise OnlineBenchmarkError(
            f"Preprocessed metadata is incomplete under {preprocessed}"
        )
    return {
        "data_identifier": identifier,
        "plans_sha256": file_sha256(plans_path),
        "dataset_fingerprint_sha256": file_sha256(fingerprint_path),
        "dataset_json_sha256": file_sha256(dataset_path),
        "splits_final_sha256": file_sha256(split_path),
        "cases": [
            {
                "case_id": case.case_id,
                "data_sha256": file_sha256(npz[case.case_id]),
                "properties_sha256": file_sha256(properties[case.case_id]),
            }
            for case in cases
        ],
    }


def _preprocess_marker_payload(
    input_contract: Mapping[str, Any], output_record: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "format": "hiercp_online_preprocess_complete_v1",
        "version": VERSION,
        "input_contract": dict(input_contract),
        "input_contract_sha256": value_sha256(input_contract),
        "outputs": dict(output_record),
    }


def _verified_preprocess_contract(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
    dataset_id: int,
) -> tuple[dict[str, Any], dict[str, Any], list[Case]]:
    input_contract, split, cases = _preprocess_input_contract(
        layout, outer_fold, train_cfg, nn_cfg, dataset_id
    )
    marker_path = (
        preprocessed_dataset_dir(layout, dataset_id, outer_fold)
        / PREPROCESS_MARKER_NAME
    )
    output_record = _preprocess_output_record(
        layout, outer_fold, nn_cfg, dataset_id, cases, split
    )
    wanted = _preprocess_marker_payload(input_contract, output_record)
    if load_json(marker_path) != wanted:
        raise OnlineBenchmarkError(
            f"Preprocessing completion marker is stale or incompatible: {marker_path}"
        )
    return wanted, split, cases


def nn_env(layout: Layout, nn_cfg: Mapping[str, Any], bank: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["nnUNet_raw"] = str(layout.raw)
    env["nnUNet_preprocessed"] = str(layout.preprocessed)
    env["nnUNet_results"] = str(layout.results)
    env["nnUNet_n_proc_DA"] = str(int(nn_cfg["training"].get("nnunet_n_proc_DA", 8)))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("PYTHONHASHSEED", "0")
    if bank is not None:
        env["ONLINE_CP_BANK"] = str(bank.resolve())
    return env


def bind_nnunet_device(
    requested: str, environment: Mapping[str, str]
) -> tuple[str, dict[str, str]]:
    """Bind an indexed CUDA request before passing nnU-Net's generic `cuda`."""

    value = str(requested).strip().lower()
    bound = dict(environment)
    match = re.fullmatch(r"cuda:(\d+)", value)
    if match is not None:
        logical_index = int(match.group(1))
        visible = bound.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            selected = str(logical_index)
        else:
            devices = [item.strip() for item in visible.split(",") if item.strip()]
            if not devices or devices == ["-1"] or logical_index >= len(devices):
                raise OnlineBenchmarkError(
                    f"Requested {requested!r}, but CUDA_VISIBLE_DEVICES={visible!r} "
                    "does not expose that logical index"
                )
            selected = devices[logical_index]
        bound["CUDA_VISIBLE_DEVICES"] = selected
        return "cuda", bound
    if value in {"cuda", "cpu", "mps"}:
        return value, bound
    raise OnlineBenchmarkError(
        f"Unsupported nnU-Net device {requested!r}; use cuda, cuda:N, cpu, or mps"
    )


def ensure_outer_split(layout: Layout, outer_fold: int, dry_run: bool) -> bool:
    if layout.outer_splits.is_file():
        outer_split(layout, outer_fold)
        return True
    pairedcp = layout.medical / "pairedcp"
    if not pairedcp.is_file():
        raise OnlineBenchmarkError(f"pairedcp wrapper is missing: {pairedcp}")
    run_command([pairedcp, "split"], cwd=layout.medical, dry_run=dry_run)
    if dry_run:
        print(
            "[Dry-run] dependent stages cannot verify the outer split until the "
            "scheduled split command has materialized it"
        )
        return False
    if not layout.outer_splits.is_file():
        raise OnlineBenchmarkError(
            f"Outer split command completed without publishing {layout.outer_splits}"
        )
    outer_split(layout, outer_fold)
    return True


def ensure_support_assets(
    layout: Layout, outer_fold: int, device: str, dry_run: bool
) -> bool:
    if not ensure_outer_split(layout, outer_fold, dry_run):
        print("[Dry-run] fold-specific GNN preparation depends on the scheduled outer split")
        return False
    gnn = layout.gnn(outer_fold)
    required = [
        gnn / "split.json",
        gnn / "prototype.pt",
        gnn / "model.pt",
        gnn / "model.last.pt",
        gnn / "causality.json",
        gnn / "causality.json.preflight.json",
        gnn / "graphs" / "complete.json",
    ]
    if all(path.is_file() for path in required):
        _verified_gnn_causality(layout, outer_fold)
        print(f"[Reuse] fold-specific GNN assets: {gnn}")
        print(
            "[OK] fold-specific causality: context=true shortcut-safe=true "
            "position-blocked=true clearance-blocked=true"
        )
        return True
    pairedcp = layout.medical / "pairedcp"
    if not pairedcp.is_file():
        raise OnlineBenchmarkError(f"pairedcp wrapper is missing: {pairedcp}")
    run_command(
        [pairedcp, "gnn-prepare", "--outer-fold", str(outer_fold)],
        cwd=layout.medical,
        dry_run=dry_run,
    )
    run_command(
        [pairedcp, "gnn-train", "--outer-fold", str(outer_fold), "--device", device],
        cwd=layout.medical,
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        dry_run=dry_run,
    )
    if dry_run:
        print(
            "[Dry-run] bank construction cannot verify GNN assets until the scheduled "
            "prepare/train commands have published them"
        )
        return False
    if not dry_run:
        if not all(path.is_file() for path in required):
            missing = [str(path) for path in required if not path.is_file()]
            raise OnlineBenchmarkError(
                f"Fold-specific GNN assets remain incomplete: {missing}"
            )
        _verified_gnn_causality(layout, outer_fold)
    return True


def build_original_dataset(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    dataset_id: int,
    materialization: str,
    overwrite: bool,
) -> None:
    _, cases, dataset_payload, expected = _raw_contract(
        layout, outer_fold, train_cfg, dataset_id, materialization
    )
    target = raw_dataset_dir(layout, dataset_id, outer_fold)
    if target.is_dir() and not overwrite:
        _audit_raw_dataset(target, cases, dataset_payload, expected)
        print(f"[Reuse] verified original-only raw dataset: {target}")
        return
    if (target.exists() or target.is_symlink()) and not overwrite:
        raise OnlineBenchmarkError(
            f"Existing raw dataset is not a verified reusable directory: {target}; "
            "use --overwrite for this exact dataset target"
        )
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp.", dir=str(target.parent))
    )
    (temporary / "imagesTr").mkdir()
    (temporary / "labelsTr").mkdir()
    (temporary / "imagesTs").mkdir()
    for case in cases:
        materialize(
            case.image,
            temporary / "imagesTr" / f"{case.case_id}_0000.nii.gz",
            materialization,
        )
        materialize(
            case.label,
            temporary / "labelsTr" / f"{case.case_id}.nii.gz",
            materialization,
        )
    atomic_json(temporary / "dataset.json", dataset_payload)
    atomic_json(temporary / RAW_MARKER_NAME, expected)
    _audit_raw_dataset(temporary, cases, dataset_payload, expected)
    temporary.replace(target)
    print(f"[OK] verified original-only raw dataset: {target} cases={len(cases)}")


def plan_and_preprocess(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
    dataset_id: int,
    dry_run: bool,
    overwrite: bool,
) -> None:
    raw = raw_dataset_dir(layout, dataset_id, outer_fold)
    name = dataset_name(dataset_id, outer_fold)
    preprocessed = preprocessed_dataset_dir(layout, dataset_id, outer_fold)
    plans_name = str(nn_cfg["dataset"]["plans"])
    planner = str(nn_cfg["dataset"]["planner"])
    configuration = str(nn_cfg["dataset"]["configuration"])
    if dry_run and (not raw.is_dir() or not layout.outer_splits.is_file()):
        print(
            "[Dry-run] preprocessing contract depends on the raw dataset and outer "
            f"split scheduled earlier: raw={raw} split={layout.outer_splits}"
        )
        input_contract = None
        split = None
        cases = None
    else:
        input_contract, split, cases = _preprocess_input_contract(
            layout, outer_fold, train_cfg, nn_cfg, dataset_id
        )
    if preprocessed.exists() or preprocessed.is_symlink():
        if input_contract is not None:
            try:
                _verified_preprocess_contract(
                    layout, outer_fold, train_cfg, nn_cfg, dataset_id
                )
            except OnlineBenchmarkError as exc:
                if dry_run:
                    print(f"[Dry-run] preprocessing is stale/incomplete: {exc}")
                    if not overwrite:
                        print("[Dry-run] execution would require --overwrite")
                        return
                elif not overwrite:
                    raise OnlineBenchmarkError(
                        f"Existing preprocessing is stale/incomplete: {exc}; "
                        "use --overwrite to rebuild this exact dataset"
                    ) from exc
            else:
                if not overwrite:
                    print(f"[Reuse] verified original-only preprocessing: {name}")
                    return
                print(f"[Overwrite] rebuilding verified preprocessing: {preprocessed}")
        elif not overwrite:
            print(
                "[Dry-run] existing preprocessing cannot be verified until the raw "
                "dataset and split are available; execution would require --overwrite"
            )
            return
        if not dry_run:
            if preprocessed.is_dir() and not preprocessed.is_symlink():
                shutil.rmtree(preprocessed)
            else:
                preprocessed.unlink()
    env = nn_env(layout, nn_cfg)
    fingerprint = [require_command("nnUNetv2_extract_fingerprint"), "-d", str(dataset_id)]
    if bool(nn_cfg["preprocess"].get("verify_dataset_integrity", True)):
        fingerprint.append("--verify_dataset_integrity")
    run_command(
        fingerprint,
        cwd=layout.project,
        env=env,
        log=layout.logs / f"fingerprint_online_of{outer_fold}.log",
        dry_run=dry_run,
    )
    run_command(
        [
            require_command("nnUNetv2_plan_experiment"),
            "-d",
            str(dataset_id),
            "-pl",
            planner,
        ],
        cwd=layout.project,
        env=env,
        log=layout.logs / f"plan_online_of{outer_fold}.log",
        dry_run=dry_run,
    )
    run_command(
        [
            require_command("nnUNetv2_preprocess"),
            "-d",
            str(dataset_id),
            "-plans_name",
            plans_name,
            "-c",
            configuration,
            "-np",
            str(int(nn_cfg["preprocess"].get("processes", 4))),
            *(
                ["--no_pbar"]
                if bool(nn_cfg["preprocess"].get("no_progress_bar", True))
                else []
            ),
        ],
        cwd=layout.project,
        env=env,
        log=layout.logs / f"preprocess_online_of{outer_fold}.log",
        dry_run=dry_run,
    )
    if dry_run:
        return
    assert input_contract is not None and split is not None and cases is not None
    desired_split = [{"train": list(split["train"]), "val": list(split["val"])}]
    split_path = preprocessed / "splits_final.json"
    if split_path.exists() and load_json_value(split_path) != desired_split:
        raise OnlineBenchmarkError(
            f"nnU-Net produced a conflicting split file: {split_path}"
        )
    if not split_path.exists():
        atomic_json(split_path, desired_split)
    output_record = _preprocess_output_record(
        layout, outer_fold, nn_cfg, dataset_id, cases, split
    )
    marker = _preprocess_marker_payload(input_contract, output_record)
    atomic_json(preprocessed / PREPROCESS_MARKER_NAME, marker)
    _verified_preprocess_contract(layout, outer_fold, train_cfg, nn_cfg, dataset_id)
    print(f"[OK] verified original-only planning/preprocessing: {name}")


def _source_from_component(case: Any, components: np.ndarray, component_id: int, pad: int) -> Any:
    from hiercp.common import SourceTumor, bbox_of_mask

    full_mask = components == int(component_id)
    if not np.any(full_mask):
        raise OnlineBenchmarkError(f"Empty source component {component_id}")
    patch_slices = bbox_of_mask(full_mask, pad=pad)
    patch_mask = full_mask[patch_slices].astype(bool, copy=True)
    patch_image = case.image[patch_slices].astype(np.float32, copy=True)
    starts = np.asarray([value.start for value in patch_slices], dtype=np.int64)
    anchor = starts + np.asarray(patch_mask.shape, dtype=np.int64) // 2
    centroid = ndi.center_of_mass(full_mask)
    return SourceTumor(
        component_id=int(component_id),
        full_mask=full_mask,
        patch_mask=patch_mask,
        patch_image=patch_image,
        patch_slices=patch_slices,
        anchor_center=tuple(int(value) for value in anchor),
        centroid=tuple(float(value) for value in centroid),
        voxel_count=int(full_mask.sum()),
    )


def _map_raw_point(
    point: Sequence[float],
    transpose_forward: Sequence[int],
    properties: Mapping[str, Any],
    pre_shape: Sequence[int],
) -> np.ndarray:
    # HierCP uses nibabel array order (x, y, z). nnU-Net's NIfTI readers
    # (SimpleITKIO and NibabelIO) both expose arrays in (z, y, x) order
    # before plans.transpose_forward is applied. Reverse first, then apply
    # the plan permutation. This avoids silently mapping GNN candidates to
    # the wrong anatomical location in preprocessed space.
    original_zyx = np.asarray(point, dtype=np.float64)[::-1]
    transposed = original_zyx[np.asarray(transpose_forward, dtype=np.int64)]
    bbox = np.asarray(properties["bbox_used_for_cropping"], dtype=np.float64)
    cropped = transposed - bbox[:, 0]
    crop_shape = np.asarray(properties["shape_after_cropping_and_before_resampling"], dtype=np.float64)
    target_shape = np.asarray(pre_shape, dtype=np.float64)
    mapped = (cropped + 0.5) * target_shape / np.maximum(crop_shape, 1.0) - 0.5
    return np.rint(mapped).astype(np.int64)


def _map_raw_bbox(
    patch_slices: Sequence[slice],
    transpose_forward: Sequence[int],
    properties: Mapping[str, Any],
    pre_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    # Raw HierCP slices are nibabel (x, y, z); nnU-Net raw arrays are
    # (z, y, x), then plans.transpose_forward is applied.
    starts_zyx = np.asarray([value.start for value in patch_slices], dtype=np.float64)[::-1]
    stops_zyx = np.asarray([value.stop for value in patch_slices], dtype=np.float64)[::-1]
    order = np.asarray(transpose_forward, dtype=np.int64)
    starts = starts_zyx[order]
    stops = stops_zyx[order]
    bbox = np.asarray(properties["bbox_used_for_cropping"], dtype=np.float64)
    crop_shape = np.asarray(properties["shape_after_cropping_and_before_resampling"], dtype=np.float64)
    target_shape = np.asarray(pre_shape, dtype=np.float64)
    scale = target_shape / np.maximum(crop_shape, 1.0)
    lower = np.floor((starts - bbox[:, 0]) * scale).astype(np.int64)
    upper = np.ceil((stops - bbox[:, 0]) * scale).astype(np.int64)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, target_shape.astype(np.int64))
    return lower, upper


def _component_bbox(mask: np.ndarray, pad: int, shape: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise OnlineBenchmarkError("Cannot build a patch from an empty preprocessed source")
    lower = np.maximum(coordinates.min(axis=0) - int(pad), 0)
    upper = np.minimum(coordinates.max(axis=0) + int(pad) + 1, np.asarray(shape, dtype=np.int64))
    return lower.astype(np.int64), upper.astype(np.int64)


def _preprocessed_source(
    data: np.ndarray,
    seg: np.ndarray,
    properties: Mapping[str, Any],
    plans: Mapping[str, Any],
    source: Any,
    tumor_label: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre_shape = data.shape[1:]
    transpose_forward = plans["transpose_forward"]
    mapped_centroid = _map_raw_point(source.centroid, transpose_forward, properties, pre_shape)
    tumor = seg[0] == int(tumor_label)
    components, count = ndi.label(tumor, structure=ndi.generate_binary_structure(3, 1))
    if count < 1:
        raise OnlineBenchmarkError("Preprocessed case contains no tumor component")
    chosen = 0
    if np.all((mapped_centroid >= 0) & (mapped_centroid < np.asarray(pre_shape))):
        chosen = int(components[tuple(mapped_centroid)])
    if chosen == 0:
        centroids = []
        for component_id in range(1, count + 1):
            centroid = np.asarray(ndi.center_of_mass(components == component_id), dtype=np.float64)
            centroids.append((float(np.linalg.norm(centroid - mapped_centroid)), component_id))
        chosen = min(centroids)[1]
    source_component = components == int(chosen)
    mapped_lower, mapped_upper = _map_raw_bbox(
        source.patch_slices, transpose_forward, properties, pre_shape
    )
    component_lower, component_upper = _component_bbox(source_component, pad=1, shape=pre_shape)
    lower = np.minimum(mapped_lower, component_lower)
    upper = np.maximum(mapped_upper, component_upper)
    if np.any(upper <= lower):
        raise OnlineBenchmarkError("Mapped source patch is empty")
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    source_data = data[(slice(None), *slices)].astype(np.float16, copy=True)
    source_mask = source_component[slices].astype(np.uint8, copy=True)
    mapped_anchor = _map_raw_point(source.anchor_center, transpose_forward, properties, pre_shape)
    anchor_offset = mapped_anchor - lower
    if np.any(anchor_offset < 0) or np.any(anchor_offset >= np.asarray(source_mask.shape)):
        # Mapping can be off by one because source.anchor_center is a voxel
        # center while the patch bounds are voxel edges. Clamp only this
        # discretization offset; the complete component remains in source_mask.
        anchor_offset = np.clip(anchor_offset, 0, np.asarray(source_mask.shape) - 1)
    if not np.any(source_mask):
        raise OnlineBenchmarkError("Mapped source tumor mask is empty")
    return source_data, source_mask, anchor_offset.astype(np.int16)


def _target_slices(
    center: Sequence[int],
    patch_shape: Sequence[int],
    anchor_offset: Sequence[int],
    volume_shape: Sequence[int],
) -> tuple[slice, ...] | None:
    starts = np.asarray(center, dtype=np.int64) - np.asarray(anchor_offset, dtype=np.int64)
    stops = starts + np.asarray(patch_shape, dtype=np.int64)
    if np.any(starts < 0) or np.any(stops > np.asarray(volume_shape, dtype=np.int64)):
        return None
    return tuple(slice(int(a), int(b)) for a, b in zip(starts, stops))


def _preprocessed_candidate_valid(
    center: Sequence[int],
    source_mask: np.ndarray,
    anchor_offset: Sequence[int],
    seg: np.ndarray,
    liver_label: int,
    tumor_label: int,
    minimum_coverage: float,
) -> bool:
    slices = _target_slices(center, source_mask.shape, anchor_offset, seg.shape[1:])
    if slices is None:
        return False
    target = seg[(0, *slices)]
    occupied = target == int(tumor_label)
    if bool(np.any(source_mask.astype(bool) & occupied)):
        return False
    liver = target == int(liver_label)
    coverage = float(np.count_nonzero(source_mask.astype(bool) & liver) / max(1, int(source_mask.sum())))
    return coverage >= float(minimum_coverage)


def candidate_pool_hash(raw_centers: np.ndarray, pre_centers: np.ndarray, scores: np.ndarray) -> str:
    payload = {
        "raw": np.asarray(raw_centers, dtype=np.int64).tolist(),
        "pre": np.asarray(pre_centers, dtype=np.int64).tolist(),
        "scores": np.round(np.asarray(scores, dtype=np.float64), 8).tolist(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_online_bank(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
    dataset_id: int,
    device_name: str,
    candidate_count: int,
    draw_count: int,
    attempts: int,
    cp_probability: float,
    pools_per_source: int,
    overwrite: bool,
) -> None:
    import torch

    from hiercp.cache import CACHE_FORMAT
    from hiercp.common import CasePaths, build_candidate_pool, load_case
    from hiercp.curriculum import build_generation_specs
    from tools.online_scoring import PendingBankScorer, SCORING_FORMAT, validate_scoring_report
    from hiercp.hierarchy import build_patient_graph, build_prototype_graph
    from hiercp.local import build_local_graph, prepare_local_source
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.prototype import PrototypeBank
    from hiercp.region import REGION_CACHE_SEED_SALT, load_or_build_patient_regions
    from hiercp.schema import graph_config_from_dict
    from hiercp.spatial import AdaptiveRoiBudgetError, CanonicalGraphUnavailable
    from hiercp.tensor import configure_runtime, load_checkpoint, resolve_device
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    def is_unrepresentable_geometry(exc: BaseException) -> bool:
        if isinstance(exc, AdaptiveRoiBudgetError):
            return False
        if isinstance(exc, CanonicalGraphUnavailable):
            return True
        message = str(exc).lower()
        known = (
            "context coordinates are invalid",
            "coordinates are invalid",
            "zero context nodes",
            "produced zero",
            "unrepresentable local geometry",
            "canonical node",
        )
        return isinstance(exc, ValueError) and any(token in message for token in known)

    def scoring_sample(
        raw_case: Any,
        source: Any,
        specs: Sequence[Any],
        locals_built: Sequence[Any],
        regions: Any,
    ) -> dict[str, Any]:
        """Assemble inference tensors from already-built local graphs.

        The v2 bank first built every local graph to validate geometry and then
        built the same graph again inside build_inference_sample. Reusing the
        validated local graphs removes that duplicate work, which is critical
        when several proposal pools are stored per source.
        """
        if not specs or len(specs) != len(locals_built):
            raise OnlineBenchmarkError("Cannot score an empty or misaligned proposal pool")
        patient_graph = build_patient_graph(
            raw_case,
            source,
            specs,
            regions,
            tumor_label=tumor_label,
            config=graph_config,
            ct_clip=ct_clip,
        )
        prototype_graph = build_prototype_graph(
            specs,
            patient_graph,
            regions,
            population_bank,
            config=graph_config,
        )
        return {
            "format": CACHE_FORMAT,
            "prototype_fingerprint": population_bank.fingerprint(),
            "case_id": raw_case.paths.case_id,
            "sample_index": -1,
            "split": "inference",
            "source_component": int(source.component_id),
            "source_patch": torch.from_numpy(
                prepared_source.source_patch.astype(np.float16)
            ),
            "target_patches": torch.from_numpy(
                np.stack([item.target_patch for item in locals_built]).astype(np.float16)
            ),
            "source_local": locals_built[0].source_local,
            "target_locals": [item.target_local for item in locals_built],
            "patient_graph": patient_graph,
            "prototype_graph": prototype_graph,
            "difficulties": torch.ones(len(specs), dtype=torch.long),
            "corruptions": torch.zeros(len(specs), dtype=torch.long),
            "candidate_centers": torch.tensor(
                [spec.center for spec in specs], dtype=torch.long
            ),
            "candidate_regions": torch.tensor(
                [spec.region_id for spec in specs], dtype=torch.long
            ),
            "candidate_prototypes": torch.tensor(
                [spec.prototype_id for spec in specs], dtype=torch.long
            ),
            "graph_config": graph_config.to_dict(),
            "ct_clip": tuple(float(value) for value in ct_clip),
        }

    if candidate_count < 2:
        raise OnlineBenchmarkError("candidate_count must be at least 2")
    if draw_count < candidate_count:
        raise OnlineBenchmarkError("draw_count must be >= candidate_count")
    if attempts < 1:
        raise OnlineBenchmarkError("attempts must be positive")
    if pools_per_source < 1:
        raise OnlineBenchmarkError("pools_per_source must be positive")
    if not 0.0 < cp_probability <= 1.0:
        raise OnlineBenchmarkError("cp_probability must be in (0,1]")

    gnn_root = layout.gnn(outer_fold)
    checkpoint_path = gnn_root / "model.pt"
    prototype_path = gnn_root / "prototype.pt"
    gnn_split_path = gnn_root / "split.json"
    causality_path = gnn_root / "causality.json"
    causality_preflight_path = gnn_root / "causality.json.preflight.json"
    graph_complete_path = gnn_root / "graphs" / "complete.json"
    # Region construction is fold-seeded in the current benchmark, so use the
    # fold-local cache produced by pairedcp gnn-prepare.
    region_cache = gnn_root / "regions"
    required_gnn = (
        checkpoint_path,
        prototype_path,
        gnn_split_path,
        causality_path,
        causality_preflight_path,
        graph_complete_path,
    )
    if not all(path.is_file() for path in required_gnn):
        raise OnlineBenchmarkError(f"Fold-specific GNN is incomplete: {gnn_root}")
    gnn_split = _verified_gnn_split(layout, outer_fold)
    _verified_gnn_causality(layout, outer_fold)
    split = outer_split(layout, outer_fold)
    train_ids = list(split["train"])
    preprocess_contract, verified_split, verified_cases = _verified_preprocess_contract(
        layout, outer_fold, train_cfg, nn_cfg, dataset_id
    )
    if verified_split != split:
        raise OnlineBenchmarkError("Verified preprocessing split changed during bank setup")
    case_map = {case.case_id: case for case in verified_cases}

    pre_root = preprocessed_dataset_dir(layout, dataset_id, outer_fold)
    plans_name = str(nn_cfg["dataset"]["plans"])
    configuration = str(nn_cfg["dataset"]["configuration"])
    plans_path = pre_root / f"{plans_name}.json"
    plans = load_json(plans_path)
    configuration_data = plans["configurations"][configuration]
    data_identifier = str(configuration_data["data_identifier"])
    network_patch_size = np.asarray(configuration_data["patch_size"], dtype=np.int64)
    pre_data_root = pre_root / data_identifier
    if not pre_data_root.is_dir():
        raise OnlineBenchmarkError(f"Preprocessed original data missing: {pre_data_root}")
    dataset_class = infer_dataset_class(str(pre_data_root))
    pre_dataset = dataset_class(str(pre_data_root), train_ids)

    normalization = plans.get("foreground_intensity_properties_per_channel", {}).get("0", {})
    ct_mean = float(normalization.get("mean", 0.0))
    ct_std = float(normalization.get("std", 1.0))
    if not np.isfinite(ct_std) or ct_std <= 0:
        raise OnlineBenchmarkError(f"Invalid CT normalization std in plans: {ct_std}")

    generation = train_cfg["generation"]
    blend_border = int(generation.get("blend_border", 0))
    if blend_border != 0:
        raise OnlineBenchmarkError(
            "OnlineCP argmax v3 requires generation.blend_border=0; "
            f"configured={blend_border}"
        )
    labels = train_cfg["labels"]
    liver_label = int(labels["liver"])
    tumor_label = int(labels["tumor"])
    min_diameter = float(nn_cfg["small_tumor"]["augmentation_min_equivalent_diameter_mm"])
    max_diameter = float(nn_cfg["small_tumor"]["augmentation_max_equivalent_diameter_mm"])
    global_seed = int(train_cfg.get("seed", 42)) + int(outer_fold)
    runtime = train_cfg.get("runtime", {})
    configure_runtime(
        deterministic=bool(runtime.get("deterministic", True)),
        allow_tf32=bool(runtime.get("allow_tf32", False)),
        cudnn_benchmark=False,
    )
    device = resolve_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = HierarchicalPyGPlacementModel(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    graph_config = graph_config_from_dict(checkpoint["graph_config"])
    ct_clip = tuple(float(value) for value in checkpoint["ct_clip"])
    population_bank = PrototypeBank.load(prototype_path)
    if population_bank.fingerprint() != checkpoint.get("prototype_fingerprint"):
        raise OnlineBenchmarkError("Fold-specific prototype/checkpoint mismatch")
    _verify_gnn_training_cohort(
        gnn_split, checkpoint, population_bank.training_case_ids
    )
    use_amp = bool(generation.get("amp", True) and device.type == "cuda")
    chunk_size = max(1, int(generation.get("local_candidate_chunk_size", 8)))

    bank_root = layout.bank(outer_fold)
    entries_root = bank_root / "entries"
    index_path = bank_root / "index.json"
    config_path = bank_root / "config.json"
    manifest_path = bank_root / "manifest.csv"
    complete_path = bank_root / "complete.json"
    metadata_contract = {
        "format": BANK_FORMAT,
        "scoring_execution_format": SCORING_FORMAT,
        "version": VERSION,
        "outer_fold": int(outer_fold),
        "dataset_id": int(dataset_id),
        "dataset_name": dataset_name(dataset_id, outer_fold),
        "train_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "outer_splits_sha256": file_sha256(layout.outer_splits),
        "raw_marker_sha256": file_sha256(
            raw_dataset_dir(layout, dataset_id, outer_fold) / RAW_MARKER_NAME
        ),
        "preprocess_marker_sha256": file_sha256(
            preprocessed_dataset_dir(layout, dataset_id, outer_fold)
            / PREPROCESS_MARKER_NAME
        ),
        "preprocess_contract_sha256": value_sha256(preprocess_contract),
        "train_config_sha256": file_sha256(layout.train_config),
        "nnunet_config_sha256": file_sha256(layout.nnunet_config),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "prototype_sha256": file_sha256(prototype_path),
        "gnn_split_sha256": file_sha256(gnn_split_path),
        "gnn_training_cases_sha256": value_sha256(gnn_split["train"]),
        "gnn_causality_sha256": file_sha256(causality_path),
        "gnn_causality_preflight_sha256": file_sha256(causality_preflight_path),
        "gnn_graph_complete_sha256": file_sha256(graph_complete_path),
        "candidate_count": int(candidate_count),
        "pools_per_source": int(pools_per_source),
        "draw_count": int(draw_count),
        "attempts_per_pool": int(attempts),
        "cp_probability": float(cp_probability),
        "basic_selection": "uniform-within-selected-shared-pool",
        "hier_selection": "exact-gnn-argmax-within-selected-shared-pool",
        "pool_schedule": "shared-uniform-pool-index",
        "tumor_label": tumor_label,
        "liver_label": liver_label,
        "minimum_diameter_mm": min_diameter,
        "maximum_diameter_mm": max_diameter,
        "minimum_liver_coverage": float(generation["min_liver_coverage"]),
        "intensity_scale_range": [float(value) for value in generation["intensity_scale_range"]],
        "intensity_shift_range_hu": [float(value) for value in generation["intensity_shift_range"]],
        "blend_border": blend_border,
        "normalization": {"mean": ct_mean, "std": ct_std},
        "network_patch_size": [int(value) for value in network_patch_size],
        "entry_storage": "npz_uncompressed_float32_multi_pool",
        "local_graph_reuse_for_scoring": True,
    }
    if index_path.is_file() and complete_path.is_file() and not overwrite:
        current = load_json(index_path)
        if all(current.get(key) == value for key, value in metadata_contract.items()):
            _verified_bank_identity(
                layout, outer_fold, train_cfg, nn_cfg, dataset_id
            )
            print(f"[Reuse] exact-argmax online CP bank: {index_path}")
            print(
                f"[OK] bank audit: cases={current.get('eligible_cases')} "
                f"sources={current.get('source_entries')} pools/source={pools_per_source} "
                f"candidates/pool={candidate_count}"
            )
            return
        raise OnlineBenchmarkError(
            f"Existing exact-argmax bank has a different contract: {index_path}; use --overwrite"
        )
    if overwrite and bank_root.exists():
        shutil.rmtree(bank_root)
    entries_root.mkdir(parents=True, exist_ok=True)
    if config_path.is_file():
        current_config = load_json(config_path)
        if current_config != metadata_contract:
            raise OnlineBenchmarkError(
                f"Partial exact-argmax bank has a different contract: {config_path}; use --overwrite"
            )
    else:
        atomic_json(config_path, metadata_contract)

    rows: list[dict[str, Any]] = [dict(row) for row in manifest_rows(manifest_path)]
    existing_by_source: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        try:
            key = (str(row.get("case_id", "")), int(row.get("source_component", -1)))
        except (TypeError, ValueError):
            continue
        existing_by_source[key] = row
    entries_by_case: dict[str, list[str]] = {}

    def commit_row(row: Mapping[str, Any]) -> None:
        key = (str(row.get("case_id", "")), int(row.get("source_component", -1) or -1))
        replaced = False
        for index, current in enumerate(rows):
            try:
                current_key = (
                    str(current.get("case_id", "")),
                    int(current.get("source_component", -1) or -1),
                )
            except (TypeError, ValueError):
                continue
            if current_key == key:
                rows[index] = dict(row)
                replaced = True
                break
        if not replaced:
            rows.append(dict(row))
        existing_by_source[key] = dict(row)
        atomic_csv(
            manifest_path,
            sorted(
                rows,
                key=lambda item: (
                    natural_key(str(item.get("case_id", ""))),
                    int(item.get("source_component", -1) or -1),
                ),
            ),
            (
                "case_id",
                "source_component",
                "diameter_mm",
                "status",
                "reason",
                "pool_count",
                "candidate_count",
                "rejected_geometry",
                "rejected_preprocessed",
                "entry",
                "entry_sha256",
                "candidate_pool_sha256",
                "argmax_unique_centers",
                "score_min",
                "score_max",
                "score_std",
            ),
        )

    scorer = PendingBankScorer(model, device, generation, candidate_count)
    source_inventory: dict[str, list[int]] = {}
    for case_index, case_id in enumerate(train_ids, start=1):
        source_case = case_map[case_id]
        print(f"[BankV3] case {case_index}/{len(train_ids)} {case_id}", flush=True)
        raw_case = load_case(
            CasePaths(case_id=case_id, image_path=source_case.image, label_path=source_case.label)
        )
        pre_data, pre_seg, _, properties = pre_dataset.load_case(case_id)
        if pre_seg is None:
            raise OnlineBenchmarkError(f"Preprocessed segmentation missing for {case_id}")
        tumor = raw_case.label == tumor_label
        components, component_count = ndi.label(
            tumor, structure=ndi.generate_binary_structure(3, 1)
        )
        sizes = np.bincount(components.ravel(), minlength=component_count + 1)
        voxel_volume = float(np.prod(raw_case.spacing))
        eligible: list[tuple[int, float]] = []
        for component_id in range(1, component_count + 1):
            diameter = equivalent_diameter(float(sizes[component_id]) * voxel_volume)
            if min_diameter < diameter <= max_diameter:
                eligible.append((component_id, diameter))
        source_inventory[case_id] = [int(component_id) for component_id, _ in eligible]
        if not eligible:
            commit_row({
                "case_id": case_id,
                "source_component": -1,
                "status": "no_eligible_source",
                "reason": f"no source in ({min_diameter}, {max_diameter}] mm",
                "diameter_mm": "",
                "pool_count": 0,
                "candidate_count": 0,
                "rejected_geometry": 0,
                "rejected_preprocessed": 0,
                "entry": "",
            })
            print(f"[Skip] {case_id}: no eligible <= {max_diameter:g} mm source")
            continue

        region_seed = stable_seed(global_seed, case_id, REGION_CACHE_SEED_SALT)
        regions = load_or_build_patient_regions(
            raw_case,
            cache_dir=region_cache,
            liver_label=liver_label,
            tumor_label=tumor_label,
            config=graph_config,
            seed=region_seed,
            ct_clip=ct_clip,
            overwrite=False,
            mmap=True,
        )
        case_entry_names: list[str] = []
        for component_id, diameter in eligible:
            row: dict[str, Any] = {
                "case_id": case_id,
                "source_component": component_id,
                "diameter_mm": f"{diameter:.6f}",
                "status": "error",
                "reason": "",
                "pool_count": 0,
                "candidate_count": 0,
                "rejected_geometry": 0,
                "rejected_preprocessed": 0,
                "entry": "",
            }
            existing_row = existing_by_source.get((case_id, int(component_id)))
            if existing_row is not None and not overwrite:
                status = str(existing_row.get("status", ""))
                relative = str(existing_row.get("entry", ""))
                if status == "ok":
                    if (
                        int(existing_row.get("candidate_count", 0)) != candidate_count
                        or int(existing_row.get("pool_count", 0)) != pools_per_source
                    ):
                        raise OnlineBenchmarkError(
                            f"Cached entry dimensions changed for {case_id}/{component_id}; "
                            "use --overwrite"
                        )
                    _audit_bank_entries(
                        bank_root,
                        {case_id: [relative]},
                        candidate_count,
                        pools_per_source,
                        {relative: existing_row},
                    )
                    case_entry_names.append(relative)
                    print(
                        f"[Reuse] {case_id} component={component_id} "
                        f"pools={pools_per_source} candidates/pool={candidate_count}"
                    )
                    continue
                if status in {
                    "insufficient_candidates",
                    "unrepresentable_source",
                    "source_patch_too_large",
                }:
                    print(f"[Reuse] {case_id} component={component_id} status={status}")
                    continue
                raise OnlineBenchmarkError(
                    "Partial exact-argmax bank row is not safely reusable for "
                    f"{case_id}/{component_id}: status={status!r}; use --overwrite"
                )
            try:
                source = _source_from_component(
                    raw_case, components, component_id, int(generation["source_pad"])
                )
                source_data, source_mask, anchor_offset = _preprocessed_source(
                    pre_data, pre_seg, properties, plans, source, tumor_label
                )
                if np.any(np.asarray(source_mask.shape, dtype=np.int64) > network_patch_size):
                    row.update(
                        status="source_patch_too_large",
                        reason=(
                            f"preprocessed source patch={source_mask.shape} exceeds "
                            f"network patch={tuple(int(v) for v in network_patch_size)}"
                        ),
                    )
                    commit_row(row)
                    print(
                        f"[Skip] {case_id} component={component_id}: "
                        f"source patch {source_mask.shape} exceeds network patch"
                    )
                    continue
                prepared_source = prepare_local_source(
                    raw_case,
                    source,
                    full_organ_mask=regions.full_organ_mask,
                    organ_depth=regions.organ_depth,
                    config=graph_config,
                    rng=np.random.default_rng(
                        stable_seed(global_seed, case_id, component_id, "source")
                    ),
                    ct_clip=ct_clip,
                )

                pools_pre: list[np.ndarray] = []
                pools_raw: list[np.ndarray] = []
                pools_scores: list[np.ndarray] = []
                rejected_geometry_total = 0
                rejected_preprocessed_total = 0
                failed_reason = ""

                for pool_index in range(pools_per_source):
                    selected_candidates: list[Any] = []
                    selected_specs: list[Any] = []
                    selected_locals: list[Any] = []
                    selected_pre_centers: list[np.ndarray] = []
                    seen: set[tuple[int, int, int]] = set()
                    seen_preprocessed: set[tuple[int, int, int]] = set()
                    for attempt in range(attempts):
                        pool_rng = np.random.default_rng(
                            stable_seed(
                                global_seed,
                                case_id,
                                component_id,
                                "argmax_pool",
                                pool_index,
                                attempt,
                            )
                        )
                        proposal, _ = build_candidate_pool(
                            raw_case,
                            source,
                            placement_mask=raw_case.label == liver_label,
                            full_organ_mask=regions.full_organ_mask,
                            occupied_mask=tumor,
                            organ_distance=regions.organ_depth,
                            rng=pool_rng,
                            num_candidates=draw_count,
                            max_draws=max(int(generation["max_draws"]), draw_count * 500),
                            min_liver_coverage=float(generation["min_liver_coverage"]),
                            occupied_clearance_vox=int(generation["occupied_clearance_vox"]),
                            min_center_separation_mm=float(generation["min_center_separation_mm"]),
                        )
                        specs = build_generation_specs(
                            proposal, regions, population_bank, config=graph_config
                        )
                        for candidate, spec in zip(proposal, specs):
                            raw_center = tuple(int(value) for value in candidate.center)
                            if raw_center in seen:
                                continue
                            seen.add(raw_center)
                            mapped = _map_raw_point(
                                raw_center,
                                plans["transpose_forward"],
                                properties,
                                pre_data.shape[1:],
                            )
                            mapped_key = tuple(int(value) for value in mapped)
                            if mapped_key in seen_preprocessed:
                                rejected_preprocessed_total += 1
                                continue
                            if not _preprocessed_candidate_valid(
                                mapped,
                                source_mask,
                                anchor_offset,
                                pre_seg,
                                liver_label,
                                tumor_label,
                                float(generation["min_liver_coverage"]),
                            ):
                                rejected_preprocessed_total += 1
                                continue
                            try:
                                built = build_local_graph(
                                    raw_case,
                                    source,
                                    spec,
                                    full_organ_mask=regions.full_organ_mask,
                                    organ_depth=regions.organ_depth,
                                    config=graph_config,
                                    rng=np.random.default_rng(
                                        stable_seed(
                                            global_seed,
                                            case_id,
                                            component_id,
                                            pool_index,
                                            raw_center,
                                            "local",
                                        )
                                    ),
                                    ct_clip=ct_clip,
                                    prepared_source=prepared_source,
                                )
                            except Exception as exc:
                                if is_unrepresentable_geometry(exc):
                                    rejected_geometry_total += 1
                                    continue
                                raise
                            seen_preprocessed.add(mapped_key)
                            selected_candidates.append(candidate)
                            selected_specs.append(spec)
                            selected_locals.append(built)
                            selected_pre_centers.append(mapped.astype(np.int32))
                            if len(selected_candidates) >= candidate_count:
                                break
                        if len(selected_candidates) >= candidate_count:
                            break
                    if len(selected_candidates) != candidate_count:
                        failed_reason = (
                            f"pool={pool_index} jointly valid={len(selected_candidates)} "
                            f"expected={candidate_count}"
                        )
                        break

                    sample = scoring_sample(
                        raw_case,
                        source,
                        selected_specs,
                        selected_locals,
                        regions,
                    )
                    pools_raw.append(
                        np.asarray(
                            [candidate.center for candidate in selected_candidates],
                            dtype=np.int32,
                        )
                    )
                    pools_pre.append(np.stack(selected_pre_centers).astype(np.int32))
                    def collect_scores(values: list[np.ndarray], *, pool_index=pool_index, target=pools_scores) -> None:
                        if len(values) != 1 or len(target) != pool_index:
                            raise OnlineBenchmarkError("Argmax scoring lost the ordered pool identity")
                        target.append(values[0])

                    scorer.submit([sample], collect_scores)
                    scorer.flush_ready()

                scorer.flush()

                if failed_reason:
                    row.update(
                        status="insufficient_candidates",
                        reason=failed_reason,
                        pool_count=len(pools_scores),
                        candidate_count=(candidate_count if pools_scores else 0),
                        rejected_geometry=rejected_geometry_total,
                        rejected_preprocessed=rejected_preprocessed_total,
                    )
                    commit_row(row)
                    print(
                        f"[Skip] {case_id} component={component_id}: {failed_reason}"
                    )
                    continue

                raw_centers = np.stack(pools_raw).astype(np.int32)
                pre_centers = np.stack(pools_pre).astype(np.int32)

                def publish_scores(
                    pools_scores: list[np.ndarray], *, case_id=case_id, component_id=component_id,
                    diameter=diameter, row=row, source_data=source_data, source_mask=source_mask,
                    anchor_offset=anchor_offset, raw_centers=raw_centers, pre_centers=pre_centers,
                    rejected_geometry_total=rejected_geometry_total,
                    rejected_preprocessed_total=rejected_preprocessed_total,
                ) -> None:
                    if len(pools_scores) != pools_per_source:
                        raise OnlineBenchmarkError("Argmax source received an incorrect ordered pool group")
                    scores_all = np.stack(pools_scores).astype(np.float32)
                    argmax_indices = np.argmax(scores_all, axis=1).astype(np.int16)
                    argmax_centers = pre_centers[np.arange(pools_per_source), argmax_indices.astype(np.int64)]
                    argmax_unique = int(np.unique(argmax_centers, axis=0).shape[0])
                    entry_path = entries_root / f"{case_id}__component_{component_id:03d}.npz"
                    if entry_path.exists() or entry_path.is_symlink():
                        raise OnlineBenchmarkError(f"Untracked exact-argmax bank entry exists: {entry_path}; use --overwrite")
                    temporary = entry_path.with_suffix(entry_path.suffix + ".tmp")
                    if temporary.exists() or temporary.is_symlink():
                        raise OnlineBenchmarkError(f"Stale temporary exact-argmax entry exists: {temporary}; use --overwrite")
                    with temporary.open("xb") as handle:
                        np.savez(handle, source_data=source_data.astype(np.float32),
                                 source_mask=source_mask.astype(np.uint8), anchor_offset=anchor_offset.astype(np.int16),
                                 candidate_centers=pre_centers, candidate_raw_centers=raw_centers, scores=scores_all,
                                 argmax_indices=argmax_indices, source_component=np.asarray([component_id], dtype=np.int16),
                                 source_diameter_mm=np.asarray([diameter], dtype=np.float32))
                        handle.flush()
                        os.fsync(handle.fileno())
                    temporary.replace(entry_path)
                    relative = str(entry_path.relative_to(bank_root))
                    row.update(status="ok", reason="", pool_count=pools_per_source, candidate_count=candidate_count,
                               rejected_geometry=rejected_geometry_total, rejected_preprocessed=rejected_preprocessed_total,
                               entry=relative, entry_sha256=file_sha256(entry_path),
                               candidate_pool_sha256=candidate_pool_hash(raw_centers, pre_centers, scores_all),
                               argmax_unique_centers=argmax_unique, score_min=f"{float(scores_all.min()):.8f}",
                               score_max=f"{float(scores_all.max()):.8f}", score_std=f"{float(scores_all.std()):.8f}")
                    commit_row(row)
                    entries_by_case.setdefault(case_id, []).append(relative)
                    print(f"[OK] {case_id} component={component_id} pools={pools_per_source} "
                          f"candidates/pool={candidate_count} argmax_unique={argmax_unique}")

                publish_scores(pools_scores)
            except AdaptiveRoiBudgetError:
                raise
            except Exception as exc:
                if is_unrepresentable_geometry(exc):
                    row.update(status="unrepresentable_source", reason=str(exc))
                    commit_row(row)
                    print(
                        f"[Skip] {case_id} component={component_id}: "
                        f"source graph unavailable ({exc})"
                    )
                else:
                    row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
                    commit_row(row)
                    print(f"[Error] {case_id} component={component_id}: {exc}")
            scorer.flush_ready()
        if case_entry_names:
            entries_by_case.setdefault(case_id, []).extend(case_entry_names)
    scorer.flush()

    error_rows = [row for row in rows if row.get("status") == "error"]
    incomplete_rows = [
        row
        for row in rows
        if row.get("status") not in {"ok", "no_eligible_source", "error"}
    ]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if error_rows:
        preview = ", ".join(
            f"{row['case_id']}/c{row['source_component']}: {row['reason']}"
            for row in error_rows[:5]
        )
        raise OnlineBenchmarkError(
            f"Exact-argmax bank has {len(error_rows)} error entries: "
            f"{preview}; see {manifest_path}"
        )
    if incomplete_rows:
        preview = ", ".join(
            f"{row['case_id']}/c{row['source_component']}: "
            f"{row.get('status')} ({row.get('reason', '')})"
            for row in incomplete_rows[:5]
        )
        raise OnlineBenchmarkError(
            "Exact-argmax bank cannot publish complete while eligible sources were skipped: "
            f"count={len(incomplete_rows)} {preview}; see {manifest_path}"
        )
    if not ok_rows or not entries_by_case:
        raise OnlineBenchmarkError(
            "Exact-argmax bank produced no usable small-tumor source entries"
        )
    scoring_execution = scorer.report()
    validate_scoring_report(scoring_execution, generation)
    index = {
        **metadata_contract,
        "scoring_execution": scoring_execution,
        "entries_by_case": {
            case_id: entries_by_case[case_id]
            for case_id in sorted(entries_by_case, key=natural_key)
        },
        "eligible_sources_by_case": {
            case_id: source_inventory[case_id]
            for case_id in sorted(source_inventory, key=natural_key)
        },
        "no_eligible_cases": sorted(
            [case_id for case_id, values in source_inventory.items() if not values],
            key=natural_key,
        ),
        "eligible_cases": sum(bool(values) for values in source_inventory.values()),
        "source_entries": sum(len(values) for values in source_inventory.values()),
        "total_pools": sum(len(values) for values in source_inventory.values())
        * pools_per_source,
        "total_candidates": sum(len(values) for values in source_inventory.values())
        * pools_per_source
        * candidate_count,
        "manifest": str(manifest_path.resolve()),
    }
    _audit_completed_bank(
        bank_root,
        index,
        rows,
        source_inventory,
        candidate_count,
        pools_per_source,
    )
    atomic_json(index_path, index)
    atomic_json(
        complete_path,
        {
            "format": BANK_FORMAT,
            "index_sha256": file_sha256(index_path),
            "manifest_sha256": file_sha256(manifest_path),
            "config_sha256": file_sha256(config_path),
            "eligible_inventory_sha256": value_sha256(source_inventory),
            "eligible_cases": index["eligible_cases"],
            "source_entries": index["source_entries"],
            "pools_per_source": pools_per_source,
            "candidate_count": candidate_count,
        },
    )
    _verified_bank_identity(layout, outer_fold, train_cfg, nn_cfg, dataset_id)
    print(
        f"[OK] verified exact-argmax online CP bank complete: "
        f"cases={index['eligible_cases']} sources={index['source_entries']} "
        f"pools/source={pools_per_source} "
        f"candidates/pool={candidate_count} index={index_path}"
    )


def _audit_bank_entries(
    bank_root: Path,
    entries_by_case: Mapping[str, Sequence[str]],
    candidate_count: int,
    pools_per_source: int,
    manifest_entries: Mapping[str, Mapping[str, Any]],
) -> None:
    seen_files: set[str] = set()
    resolved_root = bank_root.resolve()
    expected_centers = (int(pools_per_source), int(candidate_count), 3)
    expected_scores = (int(pools_per_source), int(candidate_count))
    for case_id, relative_paths in entries_by_case.items():
        if not relative_paths:
            raise OnlineBenchmarkError(f"Empty bank entry list for {case_id}")
        for relative in relative_paths:
            if not relative:
                raise OnlineBenchmarkError(f"Empty bank entry path for {case_id}")
            if Path(relative).is_absolute():
                raise OnlineBenchmarkError(f"Bank entry path must be relative: {relative}")
            if relative in seen_files:
                raise OnlineBenchmarkError(f"Duplicate bank entry path: {relative}")
            seen_files.add(relative)
            path = bank_root / relative
            if path.is_symlink():
                raise OnlineBenchmarkError(f"Bank entries must not be symlinks: {path}")
            try:
                resolved_path = path.resolve(strict=True)
                resolved_path.relative_to(resolved_root)
            except (FileNotFoundError, ValueError) as exc:
                raise OnlineBenchmarkError(
                    f"Bank entry escapes its bank root or is missing: {path}"
                ) from exc
            if not resolved_path.is_file():
                raise OnlineBenchmarkError(f"Missing bank entry for {case_id}: {path}")
            row = manifest_entries.get(relative)
            if row is None:
                raise OnlineBenchmarkError(f"Bank entry is absent from manifest: {relative}")
            if row.get("entry_sha256") != file_sha256(resolved_path):
                raise OnlineBenchmarkError(f"Bank entry content hash mismatch: {path}")
            with np.load(resolved_path, allow_pickle=False) as payload:
                required = {
                    "source_data",
                    "source_mask",
                    "anchor_offset",
                    "candidate_centers",
                    "candidate_raw_centers",
                    "scores",
                    "argmax_indices",
                }
                missing = sorted(required - set(payload.files))
                if missing:
                    raise OnlineBenchmarkError(f"Bank entry {path} is missing {missing}")
                source_data = np.asarray(payload["source_data"])
                source_mask = np.asarray(payload["source_mask"])
                anchor = np.asarray(payload["anchor_offset"])
                pre_centers = np.asarray(payload["candidate_centers"])
                raw_centers = np.asarray(payload["candidate_raw_centers"])
                scores = np.asarray(payload["scores"])
                argmax_indices = np.asarray(payload["argmax_indices"])
                if source_data.ndim != 4 or source_mask.ndim != 3:
                    raise OnlineBenchmarkError(
                        f"Invalid source tensor rank in {path}: "
                        f"{source_data.shape}/{source_mask.shape}"
                    )
                if tuple(source_data.shape[1:]) != tuple(source_mask.shape):
                    raise OnlineBenchmarkError(f"Source data/mask mismatch in {path}")
                if not np.any(source_mask):
                    raise OnlineBenchmarkError(f"Empty source mask in {path}")
                if anchor.shape != (3,) or np.any(anchor < 0) or np.any(
                    anchor >= np.asarray(source_mask.shape)
                ):
                    raise OnlineBenchmarkError(f"Invalid anchor in {path}: {anchor}")
                if pre_centers.shape != expected_centers or raw_centers.shape != expected_centers:
                    raise OnlineBenchmarkError(
                        f"Candidate tensor mismatch in {path}: "
                        f"pre={pre_centers.shape} raw={raw_centers.shape} "
                        f"expected={expected_centers}"
                    )
                if scores.shape != expected_scores or not np.all(np.isfinite(scores)):
                    raise OnlineBenchmarkError(
                        f"Score tensor mismatch in {path}: {scores.shape}, "
                        f"expected={expected_scores}"
                    )
                expected_argmax = np.argmax(scores, axis=1).astype(np.int64)
                if argmax_indices.shape != (int(pools_per_source),) or not np.array_equal(
                    argmax_indices.astype(np.int64), expected_argmax
                ):
                    raise OnlineBenchmarkError(f"Argmax audit failed in {path}")
                for pool_index in range(int(pools_per_source)):
                    if np.unique(pre_centers[pool_index], axis=0).shape[0] != int(candidate_count):
                        raise OnlineBenchmarkError(
                            f"Duplicate preprocessed candidate in {path}, pool={pool_index}"
                        )
                    if np.unique(raw_centers[pool_index], axis=0).shape[0] != int(candidate_count):
                        raise OnlineBenchmarkError(
                            f"Duplicate raw candidate in {path}, pool={pool_index}"
                        )
                expected_pool_hash = candidate_pool_hash(raw_centers, pre_centers, scores)
                if row.get("candidate_pool_sha256") != expected_pool_hash:
                    raise OnlineBenchmarkError(
                        f"Candidate pool hash mismatch in bank manifest: {path}"
                    )


def _manifest_key(row: Mapping[str, Any]) -> tuple[str, int]:
    case_id = str(row.get("case_id", "")).strip()
    try:
        component = int(row.get("source_component", ""))
    except (TypeError, ValueError) as exc:
        raise OnlineBenchmarkError(f"Malformed bank manifest row: {dict(row)}") from exc
    if not case_id or component < -1:
        raise OnlineBenchmarkError(f"Malformed bank manifest row: {dict(row)}")
    return case_id, component


def _eligible_source_inventory(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
) -> dict[str, list[int]]:
    import nibabel as nib

    split, cases = _case_cohort(layout, outer_fold)
    case_map = {case.case_id: case for case in cases}
    tumor_label = int(train_cfg["labels"]["tumor"])
    min_diameter = float(
        nn_cfg["small_tumor"]["augmentation_min_equivalent_diameter_mm"]
    )
    max_diameter = float(
        nn_cfg["small_tumor"]["augmentation_max_equivalent_diameter_mm"]
    )
    inventory: dict[str, list[int]] = {}
    structure = ndi.generate_binary_structure(3, 1)
    for case_id in split["train"]:
        _, label = load_3d(case_map[case_id].label, np.int16)
        components, count = ndi.label(label == tumor_label, structure=structure)
        sizes = np.bincount(components.ravel(), minlength=count + 1)
        image_header = nib.load(str(case_map[case_id].image)).header
        voxel_volume = float(np.prod(image_header.get_zooms()[:3]))
        inventory[case_id] = [
            component
            for component in range(1, count + 1)
            if min_diameter
            < equivalent_diameter(float(sizes[component]) * voxel_volume)
            <= max_diameter
        ]
    return inventory


def _audit_completed_bank(
    bank_root: Path,
    index: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    source_inventory: Mapping[str, Sequence[int]],
    candidate_count: int,
    pools_per_source: int,
) -> None:
    expected_keys = {
        (case_id, int(component))
        for case_id, components in source_inventory.items()
        for component in (components if components else [-1])
    }
    keys = [_manifest_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise OnlineBenchmarkError("Bank manifest contains duplicate source rows")
    actual_keys = set(keys)
    if actual_keys != expected_keys:
        raise OnlineBenchmarkError(
            "Bank manifest does not cover the exact current eligible-source inventory: "
            f"missing={sorted(expected_keys - actual_keys)} "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    grouped: dict[str, list[str]] = {}
    manifest_entries: dict[str, Mapping[str, Any]] = {}
    for key, row in zip(keys, rows):
        case_id, component = key
        status = str(row.get("status", ""))
        relative = str(row.get("entry", "")).strip()
        try:
            stored_count = int(row.get("candidate_count", 0) or 0)
            stored_pools = int(row.get("pool_count", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise OnlineBenchmarkError(f"Malformed candidate/pool count for {key}") from exc
        if component == -1:
            if (
                status != "no_eligible_source"
                or relative
                or stored_count != 0
                or stored_pools != 0
            ):
                raise OnlineBenchmarkError(
                    f"Invalid no-eligible-source sentinel for {case_id}: {dict(row)}"
                )
            continue
        if (
            status != "ok"
            or not relative
            or stored_count != int(candidate_count)
            or stored_pools != int(pools_per_source)
        ):
            raise OnlineBenchmarkError(
                f"Eligible source is incomplete for {case_id}/{component}: {dict(row)}"
            )
        if relative in manifest_entries:
            raise OnlineBenchmarkError(f"Duplicate manifest entry path: {relative}")
        manifest_entries[relative] = row
        grouped.setdefault(case_id, []).append(relative)
    grouped = {
        case_id: sorted(paths)
        for case_id, paths in sorted(grouped.items(), key=lambda item: natural_key(item[0]))
    }
    raw_index_entries = index.get("entries_by_case")
    if not isinstance(raw_index_entries, dict):
        raise OnlineBenchmarkError("Bank index entries_by_case must be an object")
    normalized_index_entries: dict[str, list[str]] = {}
    for case_id, paths in raw_index_entries.items():
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise OnlineBenchmarkError(f"Malformed bank index paths for {case_id}")
        normalized_index_entries[str(case_id)] = sorted(paths)
    if normalized_index_entries != grouped:
        raise OnlineBenchmarkError(
            "Bank index/manifest source entries differ: "
            f"index={normalized_index_entries} manifest={grouped}"
        )
    entries_root = bank_root / "entries"
    if not entries_root.is_dir() or entries_root.is_symlink():
        raise OnlineBenchmarkError(f"Bank entries directory is invalid: {entries_root}")
    actual_entry_parts: set[tuple[str, ...]] = set()
    for path in entries_root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            raise OnlineBenchmarkError(
                f"Bank entries contain a non-regular or nested directory artifact: {path}"
            )
        actual_entry_parts.add(path.relative_to(bank_root).parts)
    expected_entry_parts = {Path(relative).parts for relative in manifest_entries}
    if actual_entry_parts != expected_entry_parts:
        raise OnlineBenchmarkError(
            "Bank entries directory does not match the exact manifest file set: "
            f"missing={sorted(expected_entry_parts - actual_entry_parts)} "
            f"unexpected={sorted(actual_entry_parts - expected_entry_parts)}"
        )
    expected_inventory = {
        case_id: [int(value) for value in values]
        for case_id, values in source_inventory.items()
    }
    if index.get("eligible_sources_by_case") != expected_inventory:
        raise OnlineBenchmarkError("Bank eligible-source inventory is stale")
    no_eligible = sorted(
        [case_id for case_id, values in expected_inventory.items() if not values],
        key=natural_key,
    )
    if index.get("no_eligible_cases") != no_eligible:
        raise OnlineBenchmarkError("Bank no-eligible case inventory is stale")
    expected_cases = sum(bool(values) for values in expected_inventory.values())
    expected_sources = sum(len(values) for values in expected_inventory.values())
    if (
        index.get("eligible_cases") != expected_cases
        or index.get("source_entries") != expected_sources
        or index.get("total_pools") != expected_sources * int(pools_per_source)
        or index.get("total_candidates")
        != expected_sources * int(pools_per_source) * int(candidate_count)
    ):
        raise OnlineBenchmarkError("Bank aggregate source/pool counts are inconsistent")
    _audit_bank_entries(
        bank_root,
        grouped,
        int(candidate_count),
        int(pools_per_source),
        manifest_entries,
    )


def _verified_bank_identity(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
    dataset_id: int,
) -> dict[str, Any]:
    from tools.online_scoring import SCORING_FORMAT, validate_scoring_report

    preprocess_contract, split, _ = _verified_preprocess_contract(
        layout, outer_fold, train_cfg, nn_cfg, dataset_id
    )
    gnn_split = _verified_gnn_split(layout, outer_fold)
    _verified_gnn_causality(layout, outer_fold)
    bank_root = layout.bank(outer_fold)
    index_path = bank_root / "index.json"
    config_path = bank_root / "config.json"
    manifest_path = bank_root / "manifest.csv"
    complete_path = bank_root / "complete.json"
    index = load_json(index_path)
    config = load_json(config_path)
    complete = load_json(complete_path)
    try:
        validate_scoring_report(index.get("scoring_execution"), train_cfg["generation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise OnlineBenchmarkError(f"Bank scoring measurement is missing or stale: {exc}") from exc
    if not manifest_path.is_file():
        raise OnlineBenchmarkError(f"Exact-argmax bank manifest is missing: {manifest_path}")
    gnn_root = layout.gnn(outer_fold)
    current_links = {
        "format": BANK_FORMAT,
        "scoring_execution_format": SCORING_FORMAT,
        "version": VERSION,
        "outer_fold": int(outer_fold),
        "dataset_id": int(dataset_id),
        "dataset_name": dataset_name(dataset_id, outer_fold),
        "train_ids_sha256": hashlib.sha256(
            "\n".join(split["train"]).encode()
        ).hexdigest(),
        "outer_splits_sha256": file_sha256(layout.outer_splits),
        "raw_marker_sha256": file_sha256(
            raw_dataset_dir(layout, dataset_id, outer_fold) / RAW_MARKER_NAME
        ),
        "preprocess_marker_sha256": file_sha256(
            preprocessed_dataset_dir(layout, dataset_id, outer_fold)
            / PREPROCESS_MARKER_NAME
        ),
        "preprocess_contract_sha256": value_sha256(preprocess_contract),
        "train_config_sha256": file_sha256(layout.train_config),
        "nnunet_config_sha256": file_sha256(layout.nnunet_config),
        "checkpoint_sha256": file_sha256(gnn_root / "model.pt"),
        "prototype_sha256": file_sha256(gnn_root / "prototype.pt"),
        "gnn_split_sha256": file_sha256(gnn_root / "split.json"),
        "gnn_training_cases_sha256": value_sha256(gnn_split["train"]),
        "gnn_causality_sha256": file_sha256(gnn_root / "causality.json"),
        "gnn_causality_preflight_sha256": file_sha256(
            gnn_root / "causality.json.preflight.json"
        ),
        "gnn_graph_complete_sha256": file_sha256(
            gnn_root / "graphs" / "complete.json"
        ),
    }
    mismatches = [
        key for key, expected in current_links.items() if index.get(key) != expected
    ]
    if mismatches:
        raise OnlineBenchmarkError(
            f"Exact-argmax bank is stale for current inputs; mismatched={mismatches}"
        )
    if any(index.get(key) != value for key, value in config.items()):
        raise OnlineBenchmarkError("Exact-argmax bank config/index contract mismatch")
    if complete.get("format") != BANK_FORMAT:
        raise OnlineBenchmarkError(
            f"Exact-argmax bank completion format mismatch: {complete_path}"
        )
    if complete.get("index_sha256") != file_sha256(index_path):
        raise OnlineBenchmarkError(
            f"Exact-argmax bank index changed after completion: {index_path}"
        )
    if complete.get("manifest_sha256") != file_sha256(manifest_path):
        raise OnlineBenchmarkError(
            f"Exact-argmax bank manifest changed after completion: {manifest_path}"
        )
    if complete.get("config_sha256") != file_sha256(config_path):
        raise OnlineBenchmarkError(
            f"Exact-argmax bank config changed after completion: {config_path}"
        )
    try:
        candidate_count = int(index["candidate_count"])
        pools_per_source = int(index["pools_per_source"])
    except (KeyError, TypeError, ValueError) as exc:
        raise OnlineBenchmarkError(
            "Exact-argmax bank candidate/pool counts are invalid"
        ) from exc
    if candidate_count < 2 or pools_per_source < 1:
        raise OnlineBenchmarkError(
            "Exact-argmax bank candidate/pool counts are outside their contracts"
        )
    source_inventory = _eligible_source_inventory(
        layout, outer_fold, train_cfg, nn_cfg
    )
    if complete.get("eligible_inventory_sha256") != value_sha256(source_inventory):
        raise OnlineBenchmarkError(
            "Exact-argmax bank eligible-source completion hash is stale"
        )
    rows = manifest_rows(manifest_path)
    _audit_completed_bank(
        bank_root,
        index,
        rows,
        source_inventory,
        candidate_count,
        pools_per_source,
    )
    if (
        complete.get("eligible_cases") != index.get("eligible_cases")
        or complete.get("source_entries") != index.get("source_entries")
        or complete.get("pools_per_source") != pools_per_source
        or complete.get("candidate_count") != candidate_count
    ):
        raise OnlineBenchmarkError(
            "Exact-argmax bank completion counts are inconsistent"
        )
    return {
        "bank_format": BANK_FORMAT,
        "index_sha256": file_sha256(index_path),
        "manifest_sha256": file_sha256(manifest_path),
        "config_sha256": file_sha256(config_path),
        "complete_sha256": file_sha256(complete_path),
        "candidate_count": candidate_count,
        "pools_per_source": pools_per_source,
        "source_entries": index["source_entries"],
        "eligible_cases": index["eligible_cases"],
    }


def model_dir(
    layout: Layout,
    dataset_id: int,
    outer_fold: int,
    nn_cfg: Mapping[str, Any],
    trainer: str,
) -> Path:
    name = dataset_name(dataset_id, outer_fold)
    plans = str(nn_cfg["dataset"]["plans"])
    configuration = str(nn_cfg["dataset"]["configuration"])
    return layout.results / name / f"{trainer}__{plans}__{configuration}"


def validation_complete(folder: Path, val_ids: Sequence[str]) -> bool:
    validation = folder / "validation"
    if not validation.is_dir() or not (validation / "summary.json").is_file():
        return False
    actual = {
        path.name[: -len(".nii.gz")]
        for path in validation.glob("*.nii.gz")
        if path.is_file()
    }
    return actual == set(val_ids)


def _online_schedule_records(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise OnlineBenchmarkError(f"Online training log is missing: {path}")
    pattern = re.compile(
        r"\[OnlineCP\] epoch=(?P<epoch>\d+) applied="
        r"(?P<applied>\d+)/(?P<samples>\d+) rate=[0-9.]+ "
        r"schedule=(?P<schedule>[0-9a-f]{16})"
    )
    records: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        epoch = int(match.group("epoch"))
        record = {
            "applied": int(match.group("applied")),
            "samples": int(match.group("samples")),
            "schedule": match.group("schedule"),
        }
        if record["samples"] <= 0:
            raise OnlineBenchmarkError(
                f"Online schedule epoch {epoch} has no physical samples: {path}"
            )
        if not 0 <= record["applied"] <= record["samples"]:
            raise OnlineBenchmarkError(
                f"Online schedule epoch {epoch} has invalid CP event count "
                f"{record['applied']}/{record['samples']}: {path}"
            )
        if epoch in records:
            raise OnlineBenchmarkError(
                f"Duplicate online schedule record at epoch {epoch}: {path}"
            )
        records[epoch] = record
    return records


def _online_schedule_payload(layout: Layout, outer_fold: int) -> dict[str, Any]:
    basic_log = layout.logs / f"train_argmax_basic_of{outer_fold}.log"
    hier_log = layout.logs / f"train_argmax_hier_of{outer_fold}.log"
    basic = _online_schedule_records(basic_log)
    hier = _online_schedule_records(hier_log)
    expected_epochs = set(range(250))
    if set(basic) != expected_epochs or set(hier) != expected_epochs:
        raise OnlineBenchmarkError(
            "Online schedule audit requires exactly epochs 0..249 in both arms: "
            f"basic={len(basic)} hier={len(hier)}"
        )
    mismatches = [
        epoch for epoch in range(250)
        if basic[epoch] != hier[epoch]
    ]
    if mismatches:
        preview = ", ".join(str(value) for value in mismatches[:10])
        raise OnlineBenchmarkError(
            "Basic/Hier online data schedules diverged at epochs "
            f"{preview}; candidate policy must be the only intervention"
        )
    total_samples = int(sum(value["samples"] for value in basic.values()))
    total_cp_events = int(sum(value["applied"] for value in basic.values()))
    if total_samples <= 0 or total_cp_events <= 0:
        raise OnlineBenchmarkError(
            "Online schedule audit requires positive physical samples and CP events"
        )
    return {
        "format": "hiercp_online_argmax_schedule_audit_v3",
        "version": VERSION,
        "outer_fold": int(outer_fold),
        "epochs": 250,
        "matched": True,
        "total_samples": total_samples,
        "total_cp_events": total_cp_events,
        "epoch_records": {str(epoch): basic[epoch] for epoch in range(250)},
    }


def _audit_online_schedules(
    layout: Layout, outer_fold: int, *, overwrite: bool
) -> dict[str, Any]:
    payload = _online_schedule_payload(layout, outer_fold)
    output = layout.online_fold(outer_fold) / "schedule_audit_argmax_v3.json"
    if output.exists():
        if load_json(output) == payload:
            print(f"[Reuse] verified exact-argmax schedule audit: {output}")
            return payload
        if not overwrite:
            raise OnlineBenchmarkError(
                f"Existing exact-argmax schedule audit is stale: {output}; use --overwrite"
            )
    atomic_json(output, payload)
    print(
        f"[OK] online schedule audit: 250/250 epochs identical; "
        f"events={payload['total_cp_events']}/{payload['total_samples']} "
        f"report={output}"
    )
    return payload


def _training_input_contract(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
    dataset_id: int,
    bank_identity: Mapping[str, Any],
) -> dict[str, Any]:
    from tools.online_trainer_contract import trainer_source_identity

    preprocess_contract, split, _ = _verified_preprocess_contract(
        layout, outer_fold, train_cfg, nn_cfg, dataset_id
    )
    return {
        "format": "hiercp_online_argmax_training_input_v3",
        "version": VERSION,
        "outer_fold": int(outer_fold),
        "dataset_id": int(dataset_id),
        "dataset_name": dataset_name(dataset_id, outer_fold),
        "outer_splits_sha256": file_sha256(layout.outer_splits),
        "train_ids": list(split["train"]),
        "val_ids": list(split["val"]),
        "preprocess_marker_sha256": file_sha256(
            preprocessed_dataset_dir(layout, dataset_id, outer_fold)
            / PREPROCESS_MARKER_NAME
        ),
        "preprocess_contract_sha256": value_sha256(preprocess_contract),
        "train_config_sha256": file_sha256(layout.train_config),
        "nnunet_config_sha256": file_sha256(layout.nnunet_config),
        "bank": dict(bank_identity),
        "trainers": {"basic": TRAINER_BASIC, "hier": TRAINER_HIER},
        "trainer_sources": trainer_source_identity(
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCPArgmaxV3",
            (TRAINER_BASIC, TRAINER_HIER),
        ),
        "plans": str(nn_cfg["dataset"]["plans"]),
        "configuration": str(nn_cfg["dataset"]["configuration"]),
        "expected_epochs": 250,
    }


def _training_output_record(
    layout: Layout,
    outer_fold: int,
    nn_cfg: Mapping[str, Any],
    dataset_id: int,
    val_ids: Sequence[str],
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for method, trainer in (("basic", TRAINER_BASIC), ("hier", TRAINER_HIER)):
        fold = model_dir(layout, dataset_id, outer_fold, nn_cfg, trainer) / "fold_0"
        checkpoint = fold / "checkpoint_final.pth"
        validation = fold / "validation"
        if not checkpoint.is_file():
            raise OnlineBenchmarkError(f"Final checkpoint missing: {checkpoint}")
        if not validation_complete(fold, val_ids):
            raise OnlineBenchmarkError(
                "Exact-argmax online "
                f"{method} validation cohort is incomplete or contains extras: {validation}"
            )
        outputs[method] = {
            "trainer": trainer,
            "checkpoint_final_sha256": file_sha256(checkpoint),
            "validation_summary_sha256": file_sha256(validation / "summary.json"),
            "predictions": {
                case_id: file_sha256(validation / f"{case_id}.nii.gz")
                for case_id in val_ids
            },
        }
    schedule = _online_schedule_payload(layout, outer_fold)
    schedule_path = layout.online_fold(outer_fold) / "schedule_audit_argmax_v3.json"
    if load_json(schedule_path) != schedule:
        raise OnlineBenchmarkError(
            f"Exact-argmax schedule audit marker is stale: {schedule_path}"
        )
    outputs["schedule"] = {
        "audit_sha256": file_sha256(schedule_path),
        "basic_log_sha256": file_sha256(
            layout.logs / f"train_argmax_basic_of{outer_fold}.log"
        ),
        "hier_log_sha256": file_sha256(
            layout.logs / f"train_argmax_hier_of{outer_fold}.log"
        ),
    }
    return outputs


def _training_complete_payload(
    input_contract: Mapping[str, Any], outputs: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "format": "hiercp_online_argmax_training_complete_v3",
        "version": VERSION,
        "input_contract": dict(input_contract),
        "input_contract_sha256": value_sha256(input_contract),
        "outputs": dict(outputs),
    }


def _verified_training_completion(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
    dataset_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bank_identity = _verified_bank_identity(
        layout, outer_fold, train_cfg, nn_cfg, dataset_id
    )
    input_contract = _training_input_contract(
        layout, outer_fold, train_cfg, nn_cfg, dataset_id, bank_identity
    )
    contract_path = layout.online_fold(outer_fold) / TRAIN_CONTRACT_NAME
    if load_json(contract_path) != input_contract:
        raise OnlineBenchmarkError(
            f"Exact-argmax training input contract is stale: {contract_path}"
        )
    outputs = _training_output_record(
        layout,
        outer_fold,
        nn_cfg,
        dataset_id,
        input_contract["val_ids"],
    )
    wanted = _training_complete_payload(input_contract, outputs)
    complete_path = layout.online_fold(outer_fold) / TRAIN_COMPLETE_NAME
    if load_json(complete_path) != wanted:
        raise OnlineBenchmarkError(
            f"Exact-argmax training completion marker is stale: {complete_path}"
        )
    return wanted, input_contract


def _remove_exact_artifact(path: Path) -> None:
    if not (path.exists() or path.is_symlink()):
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def train_online_pair(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
    dataset_id: int,
    device: str,
    dry_run: bool,
    overwrite: bool,
) -> None:
    bank = layout.bank(outer_fold) / "index.json"
    if dry_run and not layout.outer_splits.is_file():
        print(
            "[Dry-run] exact-argmax training commands depend on the outer split "
            "that would be generated first"
        )
        return
    split = outer_split(layout, outer_fold)
    if dry_run:
        try:
            bank_identity = _verified_bank_identity(
                layout, outer_fold, train_cfg, nn_cfg, dataset_id
            )
        except (OnlineBenchmarkError, OSError) as exc:
            bank_identity = None
            print(
                "[Dry-run] exact-argmax training requires a verified completed bank: "
                f"{exc}"
            )
    else:
        bank_identity = _verified_bank_identity(
            layout, outer_fold, train_cfg, nn_cfg, dataset_id
        )
    input_contract = (
        _training_input_contract(
            layout, outer_fold, train_cfg, nn_cfg, dataset_id, bank_identity
        )
        if bank_identity is not None
        else None
    )
    configuration = str(nn_cfg["dataset"]["configuration"])
    plans = str(nn_cfg["dataset"]["plans"])
    fold_root = layout.online_fold(outer_fold)
    contract_path = fold_root / TRAIN_CONTRACT_NAME
    complete_path = fold_root / TRAIN_COMPLETE_NAME
    schedule_path = fold_root / "schedule_audit_argmax_v3.json"
    result_roots = [
        model_dir(layout, dataset_id, outer_fold, nn_cfg, trainer)
        for trainer in (TRAINER_BASIC, TRAINER_HIER)
    ]
    log_paths = [
        layout.logs / f"train_argmax_{method}_of{outer_fold}.log"
        for method in ("basic", "hier")
    ]
    if not dry_run:
        assert input_contract is not None
        if complete_path.exists() and not overwrite:
            try:
                _verified_training_completion(
                    layout, outer_fold, train_cfg, nn_cfg, dataset_id
                )
            except (OnlineBenchmarkError, OSError) as exc:
                raise OnlineBenchmarkError(
                    f"Existing exact-argmax training completion is stale: {exc}; "
                    "use --overwrite to rebuild the two exact trainer outputs"
                ) from exc
            print(
                "[Reuse] verified exact-argmax Basic-CP/HierCP training and validation"
            )
            return
        artifacts = [*result_roots, *log_paths, schedule_path, complete_path]
        if contract_path.exists():
            if load_json(contract_path) != input_contract and not overwrite:
                raise OnlineBenchmarkError(
                    f"Existing exact-argmax training contract differs: {contract_path}; "
                    "use --overwrite"
                )
        elif any(path.exists() or path.is_symlink() for path in artifacts) and not overwrite:
            raise OnlineBenchmarkError(
                "Exact-argmax training artifacts exist without an exact input contract; "
                "use --overwrite"
            )
        if overwrite:
            for path in [*artifacts, contract_path]:
                _remove_exact_artifact(path)
        if not contract_path.exists():
            atomic_json(contract_path, input_contract)
    for method, trainer in (("basic", TRAINER_BASIC), ("hier", TRAINER_HIER)):
        result = model_dir(layout, dataset_id, outer_fold, nn_cfg, trainer) / "fold_0"
        final = result / "checkpoint_final.pth"
        if not dry_run and final.is_file() and validation_complete(result, split["val"]):
            print(
                f"[Resume] verified-contract exact-argmax {method} arm already has "
                "final validation"
            )
            continue
        env = nn_env(layout, nn_cfg, bank=bank)
        normalized_device, env = bind_nnunet_device(device, env)
        print(
            "[DeviceBinding] "
            f"requested={device} nnunet_device={normalized_device} "
            f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
        command = [
            require_command("nnUNetv2_train"),
            str(dataset_id),
            configuration,
            "0",
            "-tr",
            trainer,
            "-p",
            plans,
            "-device",
            normalized_device,
        ]
        if final.is_file():
            command.append("--val")
        elif (result / "checkpoint_latest.pth").is_file() or (result / "checkpoint_best.pth").is_file():
            command.append("--c")
        env["ONLINE_CP_SEED"] = str(int(train_cfg.get("seed", 42)) + outer_fold)
        run_command(
            command,
            cwd=layout.project,
            env=env,
            log=layout.logs / f"train_argmax_{method}_of{outer_fold}.log",
            dry_run=dry_run,
        )
        if dry_run:
            continue
        if not final.is_file():
            raise OnlineBenchmarkError(f"Final checkpoint missing: {final}")
        if not validation_complete(result, split["val"]):
            run_command(
                [
                    require_command("nnUNetv2_train"),
                    str(dataset_id),
                    configuration,
                    "0",
                    "-tr",
                    trainer,
                    "-p",
                    plans,
                    "--val",
                    "-device",
                    normalized_device,
                ],
                cwd=layout.project,
                env=env,
                log=layout.logs / f"train_argmax_{method}_of{outer_fold}.log",
                dry_run=False,
            )
        if not validation_complete(result, split["val"]):
            raise OnlineBenchmarkError(f"Online {method} validation incomplete: {result}")
    if not dry_run:
        assert input_contract is not None
        _audit_online_schedules(layout, outer_fold, overwrite=False)
        outputs = _training_output_record(
            layout, outer_fold, nn_cfg, dataset_id, split["val"]
        )
        atomic_json(
            complete_path, _training_complete_payload(input_contract, outputs)
        )
        _verified_training_completion(
            layout, outer_fold, train_cfg, nn_cfg, dataset_id
        )
        print(f"[OK] verified exact-argmax training completion: {complete_path}")


def dice(first: np.ndarray, second: np.ndarray) -> float:
    denominator = int(first.sum() + second.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(first, second).sum() / denominator)


def lesion_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    spacing: Sequence[float],
    bins: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    structure = np.ones((3, 3, 3), np.uint8)
    gt_components, num_gt = ndi.label(ground_truth, structure=structure)
    pred_components, num_pred = ndi.label(prediction, structure=structure)
    gt_sizes = np.bincount(gt_components.ravel(), minlength=num_gt + 1)
    pred_sizes = np.bincount(pred_components.ravel(), minlength=num_pred + 1)
    intersections = np.zeros((num_gt, num_pred), dtype=np.int64)
    overlap = (gt_components > 0) & (pred_components > 0)
    if overlap.any():
        pairs, counts = np.unique(
            np.stack([gt_components[overlap] - 1, pred_components[overlap] - 1], axis=1),
            axis=0,
            return_counts=True,
        )
        intersections[pairs[:, 0], pairs[:, 1]] = counts
    matches: dict[int, int] = {}
    if num_gt and num_pred:
        rows, columns = linear_sum_assignment(-intersections)
        matches = {
            int(row): int(column)
            for row, column in zip(rows, columns)
            if intersections[row, column] > 0
        }
    voxel_volume = float(np.prod(spacing))
    sorted_bins = sorted(float(value) for value in bins)

    def bin_name(diameter: float) -> str:
        if diameter <= sorted_bins[0]:
            return f"le_{sorted_bins[0]:g}mm"
        if diameter <= sorted_bins[1]:
            return f"gt_{sorted_bins[0]:g}_le_{sorted_bins[1]:g}mm"
        return f"gt_{sorted_bins[1]:g}mm"

    output: list[dict[str, Any]] = []
    for gt_index in range(num_gt):
        gt_voxels = int(gt_sizes[gt_index + 1])
        diameter = equivalent_diameter(gt_voxels * voxel_volume)
        pred_index = matches.get(gt_index)
        detected = pred_index is not None
        pred_voxels = int(pred_sizes[pred_index + 1]) if detected else 0
        intersection = int(intersections[gt_index, pred_index]) if detected else 0
        output.append(
            {
                "diameter_mm": diameter,
                "size_bin": bin_name(diameter),
                "detected": int(detected),
                "lesion_dice": float(2 * intersection / (gt_voxels + pred_voxels)) if detected else 0.0,
            }
        )
    matched_prediction = len(set(matches.values()))
    return output, {
        "gt": int(num_gt),
        "pred": int(num_pred),
        "matched_pred": int(matched_prediction),
        "fp": int(num_pred - matched_prediction),
    }


def bootstrap_mean_difference(
    differences: Sequence[float], seed: int, iterations: int = 10000
) -> dict[str, float | None]:
    values = np.asarray(differences, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(iterations, values.size))
    samples = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def paired_wilcoxon(differences: Sequence[float]) -> float | None:
    values = np.asarray(differences, dtype=np.float64)
    if values.size == 0 or np.allclose(values, 0.0):
        return None
    try:
        return float(wilcoxon(values, zero_method="pratt", alternative="two-sided").pvalue)
    except ValueError:
        return None


def exact_mcnemar(basic_detected: Sequence[int], hier_detected: Sequence[int]) -> dict[str, Any]:
    basic = np.asarray(basic_detected, dtype=np.int64)
    hier = np.asarray(hier_detected, dtype=np.int64)
    basic_only = int(np.sum((basic == 1) & (hier == 0)))
    hier_only = int(np.sum((basic == 0) & (hier == 1)))
    discordant = basic_only + hier_only
    p_value = 1.0 if discordant == 0 else float(
        min(1.0, 2.0 * binom.cdf(min(basic_only, hier_only), discordant, 0.5))
    )
    return {
        "basic_only": basic_only,
        "hier_only": hier_only,
        "discordant": discordant,
        "exact_p": p_value,
    }


def _evaluation_input_contract(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
    dataset_id: int,
) -> dict[str, Any]:
    training_complete, training_input = _verified_training_completion(
        layout, outer_fold, train_cfg, nn_cfg, dataset_id
    )
    return {
        "format": "hiercp_online_argmax_evaluation_input_v3",
        "version": VERSION,
        "outer_fold": int(outer_fold),
        "dataset_id": int(dataset_id),
        "validation_ids": list(training_input["val_ids"]),
        "training_complete_sha256": file_sha256(
            layout.online_fold(outer_fold) / TRAIN_COMPLETE_NAME
        ),
        "training_contract_sha256": value_sha256(training_complete),
        "train_config_sha256": file_sha256(layout.train_config),
        "nnunet_config_sha256": file_sha256(layout.nnunet_config),
        "tumor_label": int(train_cfg["labels"]["tumor"]),
        "evaluation_bins_mm": [
            float(value)
            for value in nn_cfg["small_tumor"].get(
                "evaluation_bins_mm", [10.0, 20.0]
            )
        ],
    }


def _evaluation_output_record(
    output: Path, input_contract: Mapping[str, Any]
) -> dict[str, str]:
    expected_files = {
        "case_metrics.csv",
        "lesion_metrics.csv",
        "summary.json",
        "comparison.md",
    }
    allowed = set(expected_files)
    if (output / EVALUATION_COMPLETE_NAME).is_file():
        allowed.add(EVALUATION_COMPLETE_NAME)
    if not output.is_dir() or {path.name for path in output.iterdir()} != allowed:
        raise OnlineBenchmarkError(
            f"Exact-argmax evaluation has a partial or unexpected file set: {output}"
        )
    case_rows = manifest_rows(output / "case_metrics.csv")
    case_ids = [str(row.get("case_id", "")) for row in case_rows]
    expected_ids = list(input_contract["validation_ids"])
    if len(case_ids) != len(set(case_ids)) or set(case_ids) != set(expected_ids):
        raise OnlineBenchmarkError(
            "Exact-argmax evaluation case CSV does not cover the exact "
            f"validation cohort: {output}"
        )
    summary = load_json(output / "summary.json")
    if (
        summary.get("validation_ids") != expected_ids
        or summary.get("validation_cases") != len(expected_ids)
    ):
        raise OnlineBenchmarkError(
            f"Exact-argmax evaluation summary cohort is stale: {output}"
        )
    return {
        name: file_sha256(output / name)
        for name in sorted(expected_files)
    }


def _evaluation_complete_payload(
    input_contract: Mapping[str, Any], output_record: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "format": "hiercp_online_argmax_evaluation_complete_v3",
        "version": VERSION,
        "input_contract": dict(input_contract),
        "input_contract_sha256": value_sha256(input_contract),
        "outputs": dict(output_record),
    }


def _verified_evaluation_completion(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
    dataset_id: int,
) -> dict[str, Any]:
    input_contract = _evaluation_input_contract(
        layout, outer_fold, train_cfg, nn_cfg, dataset_id
    )
    output = layout.evaluation(outer_fold)
    output_record = _evaluation_output_record(output, input_contract)
    wanted = _evaluation_complete_payload(input_contract, output_record)
    if load_json(output / EVALUATION_COMPLETE_NAME) != wanted:
        raise OnlineBenchmarkError(
            f"Exact-argmax evaluation completion marker is stale: {output}"
        )
    return wanted


def evaluate_online_pair(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
    dataset_id: int,
    overwrite: bool,
) -> None:
    input_contract = _evaluation_input_contract(
        layout, outer_fold, train_cfg, nn_cfg, dataset_id
    )
    final_output = layout.evaluation(outer_fold)
    if final_output.exists() or final_output.is_symlink():
        if not overwrite:
            try:
                _verified_evaluation_completion(
                    layout, outer_fold, train_cfg, nn_cfg, dataset_id
                )
            except (OnlineBenchmarkError, OSError) as exc:
                raise OnlineBenchmarkError(
                    f"Existing exact-argmax evaluation is stale/incomplete: {exc}; "
                    "use --overwrite"
                ) from exc
            print(f"[Reuse] verified exact-argmax paired evaluation: {final_output}")
            return
        print(f"[Overwrite] rebuilding exact-argmax paired evaluation: {final_output}")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output.name}.tmp.", dir=str(final_output.parent)
        )
    )
    split = outer_split(layout, outer_fold)
    val_ids = list(split["val"])
    if val_ids != input_contract["validation_ids"]:
        raise OnlineBenchmarkError("Evaluation split changed after contract verification")
    basic_validation = model_dir(layout, dataset_id, outer_fold, nn_cfg, TRAINER_BASIC) / "fold_0" / "validation"
    hier_validation = model_dir(layout, dataset_id, outer_fold, nn_cfg, TRAINER_HIER) / "fold_0" / "validation"
    liver_label = int(train_cfg["labels"]["liver"])
    tumor_label = int(train_cfg["labels"]["tumor"])
    allowed_labels = {0, liver_label, tumor_label}
    bins = nn_cfg["small_tumor"].get("evaluation_bins_mm", [10.0, 20.0])
    _, original_cases = _case_cohort(layout, outer_fold)
    original_map = {case.case_id: case for case in original_cases}
    case_rows: list[dict[str, Any]] = []
    lesion_rows: list[dict[str, Any]] = []
    for case_id in val_ids:
        reference_nii, reference = load_3d(original_map[case_id].label, np.int16)
        basic_path = basic_validation / f"{case_id}.nii.gz"
        hier_path = hier_validation / f"{case_id}.nii.gz"
        if not basic_path.is_file() or not hier_path.is_file():
            raise OnlineBenchmarkError(f"Missing online paired validation prediction for {case_id}")
        basic_nii, basic = load_3d(basic_path, np.int16)
        hier_nii, hier = load_3d(hier_path, np.int16)
        _assert_same_nifti_geometry(
            reference_nii, reference, basic_nii, basic, basic_path
        )
        _assert_same_nifti_geometry(
            reference_nii, reference, hier_nii, hier, hier_path
        )
        _assert_label_domain(reference, allowed_labels, f"reference {case_id}")
        _assert_label_domain(basic, allowed_labels, f"Basic-CP prediction {case_id}")
        _assert_label_domain(hier, allowed_labels, f"HierCP prediction {case_id}")
        ground_truth = reference == tumor_label
        basic_mask = basic == tumor_label
        hier_mask = hier == tumor_label
        basic_lesions, basic_counts = lesion_metrics(
            ground_truth, basic_mask, reference_nii.header.get_zooms()[:3], bins
        )
        hier_lesions, hier_counts = lesion_metrics(
            ground_truth, hier_mask, reference_nii.header.get_zooms()[:3], bins
        )
        if len(basic_lesions) != len(hier_lesions):
            raise OnlineBenchmarkError(f"Ground-truth lesion alignment failed for {case_id}")
        basic_dice = dice(ground_truth, basic_mask)
        hier_dice = dice(ground_truth, hier_mask)
        case_rows.append(
            {
                "case_id": case_id,
                "outer_fold": outer_fold,
                "basic_dice": basic_dice,
                "hier_dice": hier_dice,
                "dice_difference": hier_dice - basic_dice,
                "gt_lesions": basic_counts["gt"],
                "basic_predicted": basic_counts["pred"],
                "hier_predicted": hier_counts["pred"],
                "basic_fp": basic_counts["fp"],
                "hier_fp": hier_counts["fp"],
                "fp_difference": hier_counts["fp"] - basic_counts["fp"],
            }
        )
        for component, (basic_row, hier_row) in enumerate(zip(basic_lesions, hier_lesions), start=1):
            lesion_rows.append(
                {
                    "case_id": case_id,
                    "outer_fold": outer_fold,
                    "component": component,
                    "diameter_mm": basic_row["diameter_mm"],
                    "size_bin": basic_row["size_bin"],
                    "basic_detected": basic_row["detected"],
                    "hier_detected": hier_row["detected"],
                    "basic_lesion_dice": basic_row["lesion_dice"],
                    "hier_lesion_dice": hier_row["lesion_dice"],
                    "lesion_dice_difference": hier_row["lesion_dice"] - basic_row["lesion_dice"],
                }
            )
    if not case_rows:
        raise OnlineBenchmarkError("No validation cases evaluated")
    if not lesion_rows:
        raise OnlineBenchmarkError(
            "No ground-truth tumor lesions were available; lesion comparison is undefined"
        )
    atomic_csv(output / "case_metrics.csv", case_rows, tuple(case_rows[0]))
    atomic_csv(output / "lesion_metrics.csv", lesion_rows, tuple(lesion_rows[0]) if lesion_rows else ("case_id",))

    def method_summary(method: str) -> dict[str, Any]:
        detected = sum(int(row[f"{method}_detected"]) for row in lesion_rows)
        false_positive = sum(int(row[f"{method}_fp"]) for row in case_rows)
        by_size: dict[str, Any] = {}
        for size_bin in sorted({row["size_bin"] for row in lesion_rows}):
            selected = [row for row in lesion_rows if row["size_bin"] == size_bin]
            size_detected = sum(int(row[f"{method}_detected"]) for row in selected)
            by_size[size_bin] = {
                "gt": len(selected),
                "detected": size_detected,
                "recall": size_detected / len(selected) if selected else None,
            }
        return {
            "mean_case_tumor_dice": float(np.mean([float(row[f"{method}_dice"]) for row in case_rows])),
            "gt_lesions": len(lesion_rows),
            "detected": detected,
            "lesion_recall": detected / len(lesion_rows) if lesion_rows else None,
            "false_positive_lesions": false_positive,
            "false_positive_per_case": false_positive / len(case_rows),
            "by_size": by_size,
        }

    basic_summary = method_summary("basic")
    hier_summary = method_summary("hier")
    dice_differences = [float(row["dice_difference"]) for row in case_rows]
    fp_differences = [float(row["fp_difference"]) for row in case_rows]
    lesion_dice_differences = [float(row["lesion_dice_difference"]) for row in lesion_rows]
    mcnemar = exact_mcnemar(
        [int(row["basic_detected"]) for row in lesion_rows],
        [int(row["hier_detected"]) for row in lesion_rows],
    )
    statistics = {
        "case_tumor_dice": {
            **bootstrap_mean_difference(dice_differences, 21000 + outer_fold),
            "wilcoxon_p": paired_wilcoxon(dice_differences),
        },
        "false_positive_per_case": {
            **bootstrap_mean_difference(fp_differences, 22000 + outer_fold),
            "wilcoxon_p": paired_wilcoxon(fp_differences),
        },
        "lesion_dice": {
            **bootstrap_mean_difference(lesion_dice_differences, 23000 + outer_fold),
            "wilcoxon_p": paired_wilcoxon(lesion_dice_differences),
        },
        "lesion_detection_mcnemar": mcnemar,
    }
    summary = {
        "version": VERSION,
        "outer_fold": outer_fold,
        "validation_cases": len(case_rows),
        "validation_ids": val_ids,
        "input_contract_sha256": value_sha256(input_contract),
        "basic_cp": basic_summary,
        "hiercp": hier_summary,
        "difference_hier_minus_basic": {
            "mean_case_tumor_dice": hier_summary["mean_case_tumor_dice"] - basic_summary["mean_case_tumor_dice"],
            "lesion_recall": hier_summary["lesion_recall"] - basic_summary["lesion_recall"],
            "false_positive_per_case": hier_summary["false_positive_per_case"] - basic_summary["false_positive_per_case"],
        },
        "paired_statistics": statistics,
    }
    atomic_json(output / "summary.json", summary)
    lines = [
        f"# Online Basic-CP vs Exact-Argmax HierCP — Outer Fold {outer_fold}",
        "",
        f"- Validation patients: {len(case_rows)}",
        "- Dataset/fold/plans/network/250-epoch trainer schedule: identical",
        "- Online source/event/pool/appearance schedule and proposal bank: identical",
        "- Only candidate selection differs: uniform-within-pool vs exact GNN argmax",
        "- Validation data: original CT only; online CP disabled",
        "",
        "| Metric | Basic-CP | HierCP | Hier−Basic |",
        "|---|---:|---:|---:|",
        f"| Mean tumor Dice | {basic_summary['mean_case_tumor_dice']:.4f} | {hier_summary['mean_case_tumor_dice']:.4f} | {summary['difference_hier_minus_basic']['mean_case_tumor_dice']:+.4f} |",
        f"| Lesion recall | {basic_summary['lesion_recall']:.4f} | {hier_summary['lesion_recall']:.4f} | {summary['difference_hier_minus_basic']['lesion_recall']:+.4f} |",
        f"| FP lesions/case | {basic_summary['false_positive_per_case']:.3f} | {hier_summary['false_positive_per_case']:.3f} | {summary['difference_hier_minus_basic']['false_positive_per_case']:+.3f} |",
        "",
        "## Recall by lesion size",
        "",
        "| Size bin | GT | Basic-CP | HierCP |",
        "|---|---:|---:|---:|",
    ]
    for size_bin in sorted(set(basic_summary["by_size"]) | set(hier_summary["by_size"])):
        if size_bin not in basic_summary["by_size"] or size_bin not in hier_summary["by_size"]:
            raise OnlineBenchmarkError(f"Paired size-bin summary mismatch: {size_bin}")
        basic_value = basic_summary["by_size"][size_bin]
        hier_value = hier_summary["by_size"][size_bin]
        gt = basic_value["gt"]
        lines.append(
            f"| {size_bin} | {gt} | {basic_value['recall']:.4f} | "
            f"{hier_value['recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired tests",
            "",
            f"- Tumor Dice Wilcoxon p: {statistics['case_tumor_dice']['wilcoxon_p']}",
            f"- Lesion detection exact McNemar p: {mcnemar['exact_p']:.6g} "
            f"(Basic-only={mcnemar['basic_only']}, Hier-only={mcnemar['hier_only']})",
        ]
    )
    atomic_text(output / "comparison.md", "\n".join(lines) + "\n")
    output_record = _evaluation_output_record(output, input_contract)
    atomic_json(
        output / EVALUATION_COMPLETE_NAME,
        _evaluation_complete_payload(input_contract, output_record),
    )
    backup: Path | None = None
    if final_output.exists() or final_output.is_symlink():
        backup = final_output.with_name(
            f".{final_output.name}.backup.{os.getpid()}"
        )
        if backup.exists() or backup.is_symlink():
            raise OnlineBenchmarkError(
                f"Refusing to replace evaluation while backup target exists: {backup}"
            )
        final_output.replace(backup)
    try:
        output.replace(final_output)
    except Exception:
        if backup is not None and not final_output.exists():
            backup.replace(final_output)
        raise
    if backup is not None:
        _remove_exact_artifact(backup)
    _verified_evaluation_completion(
        layout, outer_fold, train_cfg, nn_cfg, dataset_id
    )
    print(
        f"[OK] verified exact-argmax paired evaluation: "
        f"{final_output / 'comparison.md'}"
    )


def status(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
    dataset_id: int,
) -> None:
    name = dataset_name(dataset_id, outer_fold)
    print("Online Basic-CP vs exact-argmax HierCP status")
    print(f"  dataset:          {name}")
    print(f"  outer fold:       {outer_fold}")

    def report(label: str, audit: Any, detail: Any | None = None) -> None:
        try:
            result = audit()
        except (OnlineBenchmarkError, OSError) as exc:
            print(f"  {label:<17} incomplete/stale: {exc}")
            return
        suffix = f" {detail(result)}" if detail is not None else ""
        print(f"  {label:<17} verified complete{suffix}")

    report(
        "raw original:",
        lambda: _verified_raw_contract(
            layout, outer_fold, train_cfg, dataset_id
        ),
    )
    report(
        "preprocessed:",
        lambda: _verified_preprocess_contract(
            layout, outer_fold, train_cfg, nn_cfg, dataset_id
        ),
    )
    report(
        "argmax bank:",
        lambda: _verified_bank_identity(
            layout, outer_fold, train_cfg, nn_cfg, dataset_id
        ),
        lambda value: (
            f"cases={value['eligible_cases']} sources={value['source_entries']} "
            f"pools/source={value['pools_per_source']} "
            f"candidates/pool={value['candidate_count']}"
        ),
    )
    report(
        "training pair:",
        lambda: _verified_training_completion(
            layout, outer_fold, train_cfg, nn_cfg, dataset_id
        ),
    )
    report(
        "evaluation:",
        lambda: _verified_evaluation_completion(
            layout, outer_fold, train_cfg, nn_cfg, dataset_id
        ),
    )


def check_environment(layout: Layout, outer_fold: int) -> None:
    for command in (
        "nnUNetv2_extract_fingerprint",
        "nnUNetv2_plan_experiment",
        "nnUNetv2_preprocess",
        "nnUNetv2_train",
    ):
        require_command(command)
    if not layout.data.is_dir():
        raise OnlineBenchmarkError(f"Original data missing: {layout.data}")
    if not layout.train_config.is_file() or not layout.nnunet_config.is_file():
        raise OnlineBenchmarkError("HierCP train/nnU-Net config is missing")
    free_gb = shutil.disk_usage(layout.medical).free / (1024**3)
    if free_gb < 80.0:
        raise OnlineBenchmarkError(f"Insufficient free disk: {free_gb:.1f} GiB; require >=80 GiB")
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCPArgmaxV3 import (
            nnUNetTrainer_250epochs_OnlineBasicCPSharedPoolsV3,
            nnUNetTrainer_250epochs_OnlineHierCPArgmaxV3,
            _smoke_paste,
            _smoke_policy,
        )
    except Exception as exc:
        raise OnlineBenchmarkError(f"Online nnU-Net trainer is not installed correctly: {exc}") from exc
    smoke = _smoke_paste()
    policy_smoke = _smoke_policy()
    print(
        f"[OK] exact-argmax online trainers: "
        f"{nnUNetTrainer_250epochs_OnlineBasicCPSharedPoolsV3.__name__}, "
        f"{nnUNetTrainer_250epochs_OnlineHierCPArgmaxV3.__name__}; "
        f"paste_smoke={smoke} policy_smoke={policy_smoke}"
    )
    print(f"[OK] free disk={free_gb:.1f} GiB")
    if layout.outer_splits.is_file():
        split = outer_split(layout, outer_fold)
        print(f"[OK] outer split {outer_fold}: train={len(split['train'])} val={len(split['val'])}")
    else:
        print("[Info] outer split missing; `onlinecp_argmax all` will call `pairedcp split`")


def locate_project(requested: str | None) -> Path:
    if requested:
        return Path(requested).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "hiercp").is_dir():
        return candidate
    raise OnlineBenchmarkError("Cannot locate HierCP project")


def make_layout(args: argparse.Namespace) -> Layout:
    project = locate_project(args.project_root)
    medical = Path(args.medical_root).expanduser().resolve() if args.medical_root else project.parent
    paired = project / "work" / str(args.paired_root)
    online = project / "work" / str(args.online_root)
    nnroot = online / "nnunetv2"
    return Layout(
        project=project,
        medical=medical,
        data=medical / "Data",
        paired=paired,
        online=online,
        source_work=project / "work" / "full",
        train_config=project / "config" / "train.json",
        nnunet_config=project / "config" / "nnunet.json",
        outer_splits=paired / "outer_splits.json",
        nnroot=nnroot,
        raw=nnroot / "nnUNet_raw",
        preprocessed=nnroot / "nnUNet_preprocessed",
        results=nnroot / "nnUNet_results",
        logs=online / "logs",
    )


def report_overwrite_scope(
    layout: Layout, args: argparse.Namespace, nn_cfg: Mapping[str, Any]
) -> None:
    targets: list[Path] = []
    if args.command in {"dataset", "plan", "all"}:
        targets.append(raw_dataset_dir(layout, args.dataset_id, args.outer_fold))
    if args.command in {"plan", "all"}:
        targets.append(
            preprocessed_dataset_dir(layout, args.dataset_id, args.outer_fold)
        )
    if args.command in {"bank", "all"}:
        targets.append(layout.bank(args.outer_fold))
    if args.command in {"train", "all"}:
        targets.extend(
            model_dir(layout, args.dataset_id, args.outer_fold, nn_cfg, trainer)
            for trainer in (TRAINER_BASIC, TRAINER_HIER)
        )
        fold_root = layout.online_fold(args.outer_fold)
        targets.extend(
            [
                layout.logs / f"train_argmax_basic_of{args.outer_fold}.log",
                layout.logs / f"train_argmax_hier_of{args.outer_fold}.log",
                fold_root / "schedule_audit_argmax_v3.json",
                fold_root / TRAIN_CONTRACT_NAME,
                fold_root / TRAIN_COMPLETE_NAME,
            ]
        )
    if args.command in {"evaluate", "all"}:
        targets.append(layout.evaluation(args.outer_fold))
    unique = list(dict.fromkeys(path.resolve(strict=False) for path in targets))
    print("[OverwriteScope] --overwrite is limited to these benchmark artifacts:")
    for target in unique:
        print(f"  - {target}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "command",
        choices=("check", "dataset", "plan", "bank", "train", "evaluate", "status", "all"),
    )
    value.add_argument("--project-root")
    value.add_argument("--medical-root")
    value.add_argument("--paired-root", default=DEFAULT_PAIRED_ROOT)
    value.add_argument("--online-root", default=DEFAULT_ONLINE_ROOT)
    value.add_argument("--outer-fold", type=int, default=0)
    value.add_argument("--dataset-id", type=int, default=DEFAULT_DATASET_ID)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--materialization", choices=("symlink", "hardlink", "copy"), default="symlink")
    value.add_argument("--candidate-count", type=int, default=128)
    value.add_argument("--draw-count", type=int, default=256)
    value.add_argument("--candidate-attempts", type=int, default=4)
    value.add_argument("--cp-probability", type=float, default=0.5)
    value.add_argument(
        "--pools-per-source",
        type=int,
        default=4,
        help="Independent shared proposal pools stored for each source tumor",
    )
    value.add_argument("--overwrite", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    layout = make_layout(args)
    train_cfg = load_json(layout.train_config)
    nn_cfg = load_json(layout.nnunet_config)
    if args.command == "check":
        check_environment(layout, args.outer_fold)
        return
    if args.command == "status":
        status(layout, args.outer_fold, train_cfg, nn_cfg, args.dataset_id)
        return
    if args.overwrite:
        report_overwrite_scope(layout, args, nn_cfg)
    outer_ready = True
    if args.command in {"dataset", "plan", "all"}:
        outer_ready = ensure_outer_split(layout, args.outer_fold, args.dry_run)
    if args.command in {"dataset", "all"}:
        if args.dry_run:
            print("[Dry-run] original-only exact-argmax dataset construction")
        else:
            build_original_dataset(
                layout,
                args.outer_fold,
                train_cfg,
                args.dataset_id,
                args.materialization,
                args.overwrite,
            )
    if args.command in {"plan", "all"}:
        if not args.dry_run and args.command == "plan":
            try:
                _verified_raw_contract(
                    layout, args.outer_fold, train_cfg, args.dataset_id
                )
            except (OnlineBenchmarkError, OSError):
                raw_target = raw_dataset_dir(
                    layout, args.dataset_id, args.outer_fold
                )
                if (raw_target.exists() or raw_target.is_symlink()) and not args.overwrite:
                    raise
                build_original_dataset(
                    layout,
                    args.outer_fold,
                    train_cfg,
                    args.dataset_id,
                    args.materialization,
                    args.overwrite,
                )
        plan_and_preprocess(
            layout,
            args.outer_fold,
            train_cfg,
            nn_cfg,
            args.dataset_id,
            args.dry_run,
            args.overwrite,
        )
    if args.command in {"bank", "all"}:
        if args.command == "all" and not outer_ready:
            print(
                "[Dry-run] exact-argmax bank/GNN verification is deferred until the "
                "scheduled outer split exists"
            )
            support_ready = False
        else:
            support_ready = ensure_support_assets(
                layout, args.outer_fold, args.device, args.dry_run
            )
        if args.dry_run:
            if support_ready:
                print("[Dry-run] exact-argmax online bank construction")
            else:
                print(
                    "[Dry-run] exact-argmax online bank construction awaits "
                    "verified dependencies"
                )
        else:
            if not support_ready:
                raise OnlineBenchmarkError(
                    "Exact-argmax bank construction dependencies were not verified"
                )
            if not (preprocessed_dataset_dir(layout, args.dataset_id, args.outer_fold) / f"{nn_cfg['dataset']['plans']}.json").is_file():
                raise OnlineBenchmarkError("Run online planning/preprocessing before bank construction")
            build_online_bank(
                layout,
                args.outer_fold,
                train_cfg,
                nn_cfg,
                args.dataset_id,
                args.device,
                args.candidate_count,
                args.draw_count,
                args.candidate_attempts,
                args.cp_probability,
                args.pools_per_source,
                args.overwrite,
            )
    if args.command in {"train", "all"}:
        train_online_pair(
            layout,
            args.outer_fold,
            train_cfg,
            nn_cfg,
            args.dataset_id,
            args.device,
            args.dry_run,
            args.overwrite,
        )
    if args.command in {"evaluate", "all"}:
        if not args.dry_run:
            evaluate_online_pair(
                layout,
                args.outer_fold,
                train_cfg,
                nn_cfg,
                args.dataset_id,
                args.overwrite,
            )
        else:
            print("[Dry-run] online paired evaluation")
    if args.command == "all" and not args.dry_run:
        status(layout, args.outer_fold, train_cfg, nn_cfg, args.dataset_id)


if __name__ == "__main__":
    try:
        main()
    except OnlineBenchmarkError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
