#!/usr/bin/env python3
"""Quality-aware, patient-clustered re-evaluation for Online Basic-CP vs HierCP.

This script does not train or modify models. It reads the existing original
validation labels and the two nnU-Net validation prediction folders, reproduces
the legacy any-overlap lesion metric, and adds quality-aware sensitivity
criteria with patient-cluster bootstrap confidence intervals and patient-level
paired label-swap permutation tests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.online_eval_provenance import (
    EvaluationProvenanceError,
    build_evaluation_contract,
    contract_comparability,
    prepare_new_output,
    verify_evaluation_contract,
)

try:
    import nibabel as nib
except ModuleNotFoundError:
    nib = None  # self-test can run without NIfTI I/O
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon

VERSION = "online_basic_hiercp_evaluation_v4"
DEFAULT_DATASET_ID = 730
DEFAULT_PAIRED_ROOT = "paired_basic_vs_hiercp"
DEFAULT_ONLINE_ROOT = "online_basic_vs_hiercp"
DEFAULT_BASIC_TRAINER = "nnUNetTrainer_250epochs_OnlineBasicCP"
DEFAULT_HIER_TRAINER = "nnUNetTrainer_250epochs_OnlineHierCP"


class EvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Criterion:
    name: str
    label: str
    metric: str
    threshold: float
    legacy: bool = False


CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        "legacy_any_overlap",
        "Any overlap (>0 voxel; legacy)",
        "intersection",
        0.0,
        legacy=True,
    ),
    Criterion("dice_ge_0p10", "Lesion Dice >= 0.10", "dice", 0.10),
    Criterion("dice_ge_0p25", "Lesion Dice >= 0.25", "dice", 0.25),
    Criterion("dice_ge_0p50", "Lesion Dice >= 0.50", "dice", 0.50),
    Criterion("gtcov_ge_0p10", "GT coverage >= 0.10", "gt_coverage", 0.10),
    Criterion("iou_ge_0p10", "Lesion IoU >= 0.10", "iou", 0.10),
)


@dataclass(frozen=True)
class PairMatrices:
    gt_sizes: np.ndarray
    pred_sizes: np.ndarray
    intersections: np.ndarray
    dice: np.ndarray
    iou: np.ndarray
    gt_coverage: np.ndarray
    pred_coverage: np.ndarray

    @property
    def num_gt(self) -> int:
        return int(self.gt_sizes.size)

    @property
    def num_pred(self) -> int:
        return int(self.pred_sizes.size)


@dataclass(frozen=True)
class Match:
    gt_to_pred: Mapping[int, int]
    pred_to_gt: Mapping[int, int]

    @property
    def tp(self) -> int:
        return len(self.gt_to_pred)


@dataclass(frozen=True)
class Paths:
    project: Path
    medical: Path
    data: Path
    paired: Path
    online: Path
    nnroot: Path
    results: Path
    outer_splits: Path
    nnunet_config: Path


# ---------- generic I/O ----------

def natural_key(text: str) -> list[object]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"Missing JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Invalid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"JSON root must be an object: {path}")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic create, never replace an existing file (including a symlink).
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    # Stream full-cohort rows without making an extra whole-CSV copy in RAM.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_prediction_inventory(folder: Path, case_ids: Sequence[str]) -> None:
    expected = {f"{case_id}.nii.gz" for case_id in case_ids}
    actual = {path.name for path in folder.iterdir()
              if path.name.endswith((".nii", ".nii.gz"))}
    if actual != expected:
        raise EvaluationError(
            f"Prediction cohort mismatch in {folder}: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def outer_split(path: Path, fold: int) -> dict[str, list[str]]:
    payload = load_json(path)
    splits = payload.get("splits")
    if not isinstance(splits, list) or not 0 <= int(fold) < len(splits):
        raise EvaluationError(f"Outer fold {fold} is unavailable in {path}")
    split = splits[int(fold)]
    if not isinstance(split, dict):
        raise EvaluationError(f"Malformed outer fold {fold}")
    train = split.get("train")
    val = split.get("val")
    if not isinstance(train, list) or not isinstance(val, list):
        raise EvaluationError(f"Malformed train/val lists for outer fold {fold}")
    if any(not isinstance(x, str) or not x for x in train + val):
        raise EvaluationError("Outer split patient IDs must be nonempty strings")
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise EvaluationError("Duplicate patient IDs in outer split")
    if set(train) & set(val):
        raise EvaluationError("Outer train/validation patient leakage detected")
    return {"train": [str(x) for x in train], "val": [str(x) for x in val]}


def dataset_name(dataset_id: int, outer_fold: int) -> str:
    return f"Dataset{int(dataset_id):03d}_LiverOnlineCP_OF{int(outer_fold)}"


def model_validation_dir(
    paths: Paths,
    dataset_id: int,
    outer_fold: int,
    nn_cfg: Mapping[str, Any],
    trainer: str,
) -> Path:
    name = dataset_name(dataset_id, outer_fold)
    plans = str(nn_cfg["dataset"]["plans"])
    configuration = str(nn_cfg["dataset"]["configuration"])
    return (
        paths.results
        / name
        / f"{trainer}__{plans}__{configuration}"
        / "fold_0"
        / "validation"
    )


def load_label(path: Path, tumor_label: int) -> tuple[Any, np.ndarray]:
    if nib is None:
        raise EvaluationError("nibabel is required for NIfTI evaluation")
    if not path.is_file():
        raise EvaluationError(f"Missing NIfTI: {path}")
    nii = nib.load(str(path))
    if len(nii.shape) == 4:
        if nii.shape[-1] <= 4:
            nii = nii.slicer[..., 0]
        elif nii.shape[0] <= 4:
            nii = nii.slicer[0, ...]
        else:
            raise EvaluationError(f"Cannot infer channel axis: {path} shape={nii.shape}")
    data = np.asarray(nii.dataobj, dtype=np.int16, order="C")
    if data.ndim != 3:
        raise EvaluationError(f"Expected 3-D label: {path} shape={data.shape}")
    return nii, data == int(tumor_label)


def verify_geometry(reference: Any, candidate: Any, path: Path) -> None:
    if tuple(reference.shape[:3]) != tuple(candidate.shape[:3]):
        raise EvaluationError(
            f"Prediction shape mismatch: {path} {candidate.shape[:3]} != {reference.shape[:3]}"
        )
    if not np.allclose(reference.affine, candidate.affine, rtol=0.0, atol=1e-4):
        raise EvaluationError(f"Prediction affine mismatch: {path}")


# ---------- connected components and matching ----------

def equivalent_diameter(volume_mm3: float) -> float:
    if volume_mm3 <= 0:
        return 0.0
    return float(2.0 * (3.0 * float(volume_mm3) / (4.0 * math.pi)) ** (1.0 / 3.0))


def size_bin(diameter: float, bins: Sequence[float]) -> str:
    values = sorted(float(value) for value in bins)
    if len(values) != 2:
        raise EvaluationError(f"Expected exactly two size bins, got {values}")
    if diameter <= values[0]:
        return f"le_{values[0]:g}mm"
    if diameter <= values[1]:
        return f"gt_{values[0]:g}_le_{values[1]:g}mm"
    return f"gt_{values[1]:g}mm"


def component_pair_matrices(gt_mask: np.ndarray, pred_mask: np.ndarray) -> PairMatrices:
    structure = np.ones((3, 3, 3), dtype=np.uint8)  # preserve legacy 26-connectivity
    gt_components, num_gt = ndi.label(gt_mask, structure=structure)
    pred_components, num_pred = ndi.label(pred_mask, structure=structure)
    gt_sizes_full = np.bincount(gt_components.ravel(), minlength=num_gt + 1)
    pred_sizes_full = np.bincount(pred_components.ravel(), minlength=num_pred + 1)
    gt_sizes = gt_sizes_full[1:].astype(np.int64, copy=False)
    pred_sizes = pred_sizes_full[1:].astype(np.int64, copy=False)
    intersections = np.zeros((num_gt, num_pred), dtype=np.int64)
    overlap = (gt_components > 0) & (pred_components > 0)
    if np.any(overlap):
        pairs, counts = np.unique(
            np.stack(
                [gt_components[overlap] - 1, pred_components[overlap] - 1],
                axis=1,
            ),
            axis=0,
            return_counts=True,
        )
        intersections[pairs[:, 0], pairs[:, 1]] = counts

    if num_gt and num_pred:
        gt = gt_sizes[:, None].astype(np.float64)
        pred = pred_sizes[None, :].astype(np.float64)
        inter = intersections.astype(np.float64)
        dice_matrix = np.divide(
            2.0 * inter,
            gt + pred,
            out=np.zeros_like(inter),
            where=(gt + pred) > 0,
        )
        union = gt + pred - inter
        iou_matrix = np.divide(
            inter,
            union,
            out=np.zeros_like(inter),
            where=union > 0,
        )
        gt_coverage = np.divide(inter, gt, out=np.zeros_like(inter), where=gt > 0)
        pred_coverage = np.divide(inter, pred, out=np.zeros_like(inter), where=pred > 0)
    else:
        shape = (num_gt, num_pred)
        dice_matrix = np.zeros(shape, dtype=np.float64)
        iou_matrix = np.zeros(shape, dtype=np.float64)
        gt_coverage = np.zeros(shape, dtype=np.float64)
        pred_coverage = np.zeros(shape, dtype=np.float64)

    return PairMatrices(
        gt_sizes=gt_sizes,
        pred_sizes=pred_sizes,
        intersections=intersections,
        dice=dice_matrix,
        iou=iou_matrix,
        gt_coverage=gt_coverage,
        pred_coverage=pred_coverage,
    )


def metric_matrix(matrices: PairMatrices, metric: str) -> np.ndarray:
    if metric == "intersection":
        return matrices.intersections.astype(np.float64)
    if metric == "dice":
        return matrices.dice
    if metric == "iou":
        return matrices.iou
    if metric == "gt_coverage":
        return matrices.gt_coverage
    if metric == "pred_coverage":
        return matrices.pred_coverage
    raise EvaluationError(f"Unknown matching metric: {metric}")


def match_components(matrices: PairMatrices, criterion: Criterion) -> Match:
    if matrices.num_gt == 0 or matrices.num_pred == 0:
        return Match({}, {})

    score = metric_matrix(matrices, criterion.metric)
    if criterion.legacy:
        # Exact compatibility with the previous evaluator: Hungarian assignment
        # maximizes raw intersection, then zero-overlap pairs are discarded.
        rows, columns = linear_sum_assignment(-matrices.intersections)
        pairs = [
            (int(row), int(column))
            for row, column in zip(rows, columns)
            if matrices.intersections[row, column] > 0
        ]
    else:
        valid = (matrices.intersections > 0) & (score >= float(criterion.threshold))
        # Prioritize the number of valid one-to-one detections, then pair quality.
        # The validity bonus is greater than any possible sum of normalized scores.
        bonus = float(min(matrices.num_gt, matrices.num_pred) + 1)
        objective = valid.astype(np.float64) * bonus + np.clip(score, 0.0, 1.0)
        rows, columns = linear_sum_assignment(-objective)
        pairs = [
            (int(row), int(column))
            for row, column in zip(rows, columns)
            if valid[row, column]
        ]
    gt_to_pred = {gt: pred for gt, pred in pairs}
    pred_to_gt = {pred: gt for gt, pred in pairs}
    return Match(gt_to_pred, pred_to_gt)


def max_dice_match(matrices: PairMatrices) -> Match:
    if matrices.num_gt == 0 or matrices.num_pred == 0:
        return Match({}, {})
    rows, columns = linear_sum_assignment(-matrices.dice)
    pairs = [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if matrices.intersections[row, column] > 0
    ]
    return Match(
        {gt: pred for gt, pred in pairs},
        {pred: gt for gt, pred in pairs},
    )


def whole_mask_dice(first: np.ndarray, second: np.ndarray) -> float:
    denominator = int(first.sum() + second.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(first, second).sum() / denominator)


def pair_values(matrices: PairMatrices, gt_index: int, pred_index: int) -> dict[str, Any]:
    return {
        "intersection_vox": int(matrices.intersections[gt_index, pred_index]),
        "gt_vox": int(matrices.gt_sizes[gt_index]),
        "pred_vox": int(matrices.pred_sizes[pred_index]),
        "dice": float(matrices.dice[gt_index, pred_index]),
        "iou": float(matrices.iou[gt_index, pred_index]),
        "gt_coverage": float(matrices.gt_coverage[gt_index, pred_index]),
        "pred_coverage": float(matrices.pred_coverage[gt_index, pred_index]),
    }


# ---------- clustered inference ----------

def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def metric_from_counts(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    metric: str,
    case_count: int | np.ndarray,
) -> np.ndarray:
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    if metric == "recall":
        return _safe_ratio(tp, tp + fn)
    if metric == "precision":
        return _safe_ratio(tp, tp + fp)
    if metric == "f1":
        return _safe_ratio(2.0 * tp, 2.0 * tp + fp + fn)
    if metric == "fp_per_case":
        denominator = np.asarray(case_count, dtype=np.float64)
        return _safe_ratio(fp, denominator)
    raise EvaluationError(f"Unsupported count metric: {metric}")


def _finite_summary(values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None, None
    return (
        float(np.mean(finite)),
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    )


def _validated_count_resampling_inputs(
    basic: np.ndarray, hier: np.ndarray, iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    basic = np.asarray(basic, dtype=np.float64)
    hier = np.asarray(hier, dtype=np.float64)
    if basic.shape != hier.shape or basic.ndim != 2 or basic.shape[1] != 3:
        raise EvaluationError("Count arrays must have shape [patients, 3]")
    if not np.all(np.isfinite(basic)) or not np.all(np.isfinite(hier)):
        raise EvaluationError("Count arrays must contain finite values")
    if np.any(basic < 0) or np.any(hier < 0):
        raise EvaluationError("Count arrays must contain nonnegative values")
    if not isinstance(iterations, (int, np.integer)) or iterations <= 0:
        raise EvaluationError("Resampling iterations must be a positive integer")
    return basic, hier


def _count_resampling_diagnostics(iterations: int) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": None,
        "requested_resamples": int(iterations),
        "completed_resamples": 0,
        "valid_resamples": 0,
        "invalid_resamples": 0,
    }


def cluster_bootstrap_count_difference(
    basic: np.ndarray,
    hier: np.ndarray,
    metric: str,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    # Columns are tp, fp, fn per patient.
    basic, hier = _validated_count_resampling_inputs(basic, hier, iterations)
    diagnostics = _count_resampling_diagnostics(iterations)
    result = {
        "difference": None,
        "ci_low": None,
        "ci_high": None,
        "bootstrap_diagnostics": diagnostics,
    }
    n = int(basic.shape[0])
    if n == 0:
        diagnostics["reason"] = "no_patients"
        return result
    observed_basic = metric_from_counts(
        np.array([basic[:, 0].sum()]),
        np.array([basic[:, 1].sum()]),
        np.array([basic[:, 2].sum()]),
        metric,
        np.array([n]),
    )[0]
    observed_hier = metric_from_counts(
        np.array([hier[:, 0].sum()]),
        np.array([hier[:, 1].sum()]),
        np.array([hier[:, 2].sum()]),
        metric,
        np.array([n]),
    )[0]
    observed = float(observed_hier - observed_basic)
    if not math.isfinite(observed):
        diagnostics["reason"] = "observed_metric_undefined"
        return result
    result["difference"] = observed
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(int(iterations), n))
    b = basic[indices].sum(axis=1)
    h = hier[indices].sum(axis=1)
    b_metric = metric_from_counts(b[:, 0], b[:, 1], b[:, 2], metric, n)
    h_metric = metric_from_counts(h[:, 0], h[:, 1], h[:, 2], metric, n)
    differences = h_metric - b_metric
    valid = int(np.count_nonzero(np.isfinite(differences)))
    diagnostics.update(
        completed_resamples=int(iterations),
        valid_resamples=valid,
        invalid_resamples=int(iterations) - valid,
    )
    if valid != int(iterations):
        # Dropping undefined draws would report a conditional bootstrap
        # distribution, not the requested whole-patient resampling distribution.
        diagnostics["reason"] = "undefined_metric_in_resamples"
        return result
    diagnostics["status"] = "available"
    result["ci_low"] = float(np.quantile(differences, 0.025))
    result["ci_high"] = float(np.quantile(differences, 0.975))
    return result


def cluster_permutation_count_difference(
    basic: np.ndarray,
    hier: np.ndarray,
    metric: str,
    seed: int,
    iterations: int,
) -> float | None:
    """Compatibility wrapper; use the inference result for availability diagnostics."""
    return cluster_permutation_count_inference(
        basic, hier, metric, seed, iterations
    )["permutation_p"]


def cluster_permutation_count_inference(
    basic: np.ndarray,
    hier: np.ndarray,
    metric: str,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    basic, hier = _validated_count_resampling_inputs(basic, hier, iterations)
    diagnostics = _count_resampling_diagnostics(iterations)
    result = {"permutation_p": None, "permutation_diagnostics": diagnostics}
    n = int(basic.shape[0])
    if n == 0:
        diagnostics["reason"] = "no_patients"
        return result
    observed_b = metric_from_counts(
        np.array([basic[:, 0].sum()]),
        np.array([basic[:, 1].sum()]),
        np.array([basic[:, 2].sum()]),
        metric,
        np.array([n]),
    )[0]
    observed_h = metric_from_counts(
        np.array([hier[:, 0].sum()]),
        np.array([hier[:, 1].sum()]),
        np.array([hier[:, 2].sum()]),
        metric,
        np.array([n]),
    )[0]
    observed = float(observed_h - observed_b)
    if not math.isfinite(observed):
        diagnostics["reason"] = "observed_metric_undefined"
        return result
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    valid = 0
    chunk = 4096
    while completed < int(iterations):
        current = min(chunk, int(iterations) - completed)
        swap = rng.integers(0, 2, size=(current, n), dtype=np.int8).astype(bool)
        pseudo_basic = np.where(swap[:, :, None], hier[None, :, :], basic[None, :, :]).sum(axis=1)
        pseudo_hier = np.where(swap[:, :, None], basic[None, :, :], hier[None, :, :]).sum(axis=1)
        b_metric = metric_from_counts(
            pseudo_basic[:, 0], pseudo_basic[:, 1], pseudo_basic[:, 2], metric, n
        )
        h_metric = metric_from_counts(
            pseudo_hier[:, 0], pseudo_hier[:, 1], pseudo_hier[:, 2], metric, n
        )
        null = h_metric - b_metric
        finite = np.isfinite(null)
        valid += int(np.count_nonzero(finite))
        extreme += int(np.sum(np.abs(null[finite]) >= abs(observed) - 1e-15))
        completed += current
    diagnostics.update(
        completed_resamples=completed,
        valid_resamples=valid,
        invalid_resamples=completed - valid,
    )
    if valid != completed:
        # Undefined precision (for example, no predicted lesions after a swap)
        # is not a non-extreme draw, nor may it be silently discarded/replaced.
        diagnostics["reason"] = "undefined_metric_in_resamples"
        return result
    diagnostics["status"] = "available"
    result["permutation_p"] = float((extreme + 1) / (completed + 1))
    return result


def paired_cluster_bootstrap_mean(
    basic: np.ndarray,
    hier: np.ndarray,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    basic = np.asarray(basic, dtype=np.float64)
    hier = np.asarray(hier, dtype=np.float64)
    if basic.shape != hier.shape or basic.ndim != 1:
        raise EvaluationError("Paired mean arrays must be aligned 1-D arrays")
    n = int(basic.size)
    if n == 0:
        return {"difference": None, "ci_low": None, "ci_high": None}
    differences = hier - basic
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(int(iterations), n))
    samples = differences[indices].mean(axis=1)
    _, low, high = _finite_summary(samples)
    return {"difference": observed, "ci_low": low, "ci_high": high}


def paired_swap_permutation_mean(
    basic: np.ndarray,
    hier: np.ndarray,
    seed: int,
    iterations: int,
) -> float | None:
    basic = np.asarray(basic, dtype=np.float64)
    hier = np.asarray(hier, dtype=np.float64)
    differences = hier - basic
    if differences.size == 0 or not np.all(np.isfinite(differences)):
        return None
    observed = float(abs(np.mean(differences)))
    if np.allclose(differences, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    chunk = 4096
    while completed < int(iterations):
        current = min(chunk, int(iterations) - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(current, differences.size))
        null = np.mean(signs * differences[None, :], axis=1)
        extreme += int(np.sum(np.abs(null) >= observed - 1e-15))
        completed += current
    return float((extreme + 1) / (int(iterations) + 1))


def paired_wilcoxon(basic: np.ndarray, hier: np.ndarray) -> float | None:
    differences = np.asarray(hier, dtype=np.float64) - np.asarray(basic, dtype=np.float64)
    if differences.size == 0 or np.allclose(differences, 0.0):
        return None
    try:
        return float(wilcoxon(differences, zero_method="pratt", alternative="two-sided").pvalue)
    except ValueError:
        return None


def cluster_bootstrap_lesion_mean(
    basic_sum: np.ndarray,
    hier_sum: np.ndarray,
    counts: np.ndarray,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    basic_sum = np.asarray(basic_sum, dtype=np.float64)
    hier_sum = np.asarray(hier_sum, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if basic_sum.ndim != 1 or basic_sum.shape != hier_sum.shape or basic_sum.shape != counts.shape:
        raise EvaluationError("Lesion quality sums and counts must be aligned 1-D arrays")
    if not all(np.all(np.isfinite(values)) and np.all(values >= 0) for values in (basic_sum, hier_sum, counts)):
        raise EvaluationError("Lesion quality sums and counts must be finite and nonnegative")
    if not isinstance(iterations, (int, np.integer)) or iterations <= 0:
        raise EvaluationError("Resampling iterations must be a positive integer")
    diagnostics = _count_resampling_diagnostics(iterations)
    result = {
        "difference": None,
        "ci_low": None,
        "ci_high": None,
        "bootstrap_diagnostics": diagnostics,
    }
    total = counts.sum()
    if total <= 0:
        diagnostics["reason"] = "no_lesions"
        return result
    observed = float(hier_sum.sum() / total - basic_sum.sum() / total)
    result["difference"] = observed
    n = int(counts.size)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(int(iterations), n))
    count_sample = counts[indices].sum(axis=1)
    b = _safe_ratio(basic_sum[indices].sum(axis=1), count_sample)
    h = _safe_ratio(hier_sum[indices].sum(axis=1), count_sample)
    differences = h - b
    valid = int(np.count_nonzero(np.isfinite(differences)))
    diagnostics.update(
        completed_resamples=int(iterations),
        valid_resamples=valid,
        invalid_resamples=int(iterations) - valid,
    )
    if valid != int(iterations):
        diagnostics["reason"] = "undefined_metric_in_resamples"
        return result
    diagnostics["status"] = "available"
    result["ci_low"] = float(np.quantile(differences, 0.025))
    result["ci_high"] = float(np.quantile(differences, 0.975))
    return result


def cluster_permutation_lesion_mean(
    basic_sum: np.ndarray,
    hier_sum: np.ndarray,
    counts: np.ndarray,
    seed: int,
    iterations: int,
) -> float | None:
    basic_sum = np.asarray(basic_sum, dtype=np.float64)
    hier_sum = np.asarray(hier_sum, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return None
    observed = float(hier_sum.sum() / total - basic_sum.sum() / total)
    rng = np.random.default_rng(seed)
    n = int(counts.size)
    extreme = 0
    completed = 0
    chunk = 4096
    while completed < int(iterations):
        current = min(chunk, int(iterations) - completed)
        swap = rng.integers(0, 2, size=(current, n), dtype=np.int8).astype(bool)
        pseudo_basic = np.where(swap, hier_sum[None, :], basic_sum[None, :]).sum(axis=1)
        pseudo_hier = np.where(swap, basic_sum[None, :], hier_sum[None, :]).sum(axis=1)
        null = pseudo_hier / total - pseudo_basic / total
        extreme += int(np.sum(np.abs(null) >= abs(observed) - 1e-15))
        completed += current
    return float((extreme + 1) / (int(iterations) + 1))


# ---------- evaluation ----------

def _method_summary(case_detection_rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    selected = [row for row in case_detection_rows if row["method"] == method]
    tp = int(sum(int(row["tp"]) for row in selected))
    fp = int(sum(int(row["fp"]) for row in selected))
    fn = int(sum(int(row["fn"]) for row in selected))
    cases = len(selected)
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fp_per_case": fp / cases if cases else None,
    }


def _counts_array(
    case_detection_rows: Sequence[Mapping[str, Any]],
    case_ids: Sequence[str],
    criterion: str,
    method: str,
) -> np.ndarray:
    lookup = {
        str(row["case_id"]): row
        for row in case_detection_rows
        if row["criterion"] == criterion and row["method"] == method
    }
    missing = [case_id for case_id in case_ids if case_id not in lookup]
    if missing:
        raise EvaluationError(
            f"Missing case detection rows for criterion={criterion} method={method}: {missing}"
        )
    return np.asarray(
        [
            [
                float(lookup[case_id]["tp"]),
                float(lookup[case_id]["fp"]),
                float(lookup[case_id]["fn"]),
            ]
            for case_id in case_ids
        ],
        dtype=np.float64,
    )


def _size_counts_array(
    lesion_detection_rows: Sequence[Mapping[str, Any]],
    case_ids: Sequence[str],
    criterion: str,
    method: str,
    selected_bin: str,
) -> np.ndarray:
    by_case: dict[str, list[Mapping[str, Any]]] = {case_id: [] for case_id in case_ids}
    for row in lesion_detection_rows:
        if (
            row["criterion"] == criterion
            and row["method"] == method
            and row["size_bin"] == selected_bin
        ):
            by_case[str(row["case_id"])].append(row)
    output = []
    for case_id in case_ids:
        rows = by_case[case_id]
        tp = sum(int(row["detected"]) for row in rows)
        fn = len(rows) - tp
        output.append([tp, 0, fn])
    return np.asarray(output, dtype=np.float64)


def _regression_check(
    legacy_summary: Mapping[str, Any],
    old_summary_path: Path,
    tolerance: float,
    allow_mismatch: bool,
    *,
    evaluation_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise EvaluationError("Regression tolerance must be finite and nonnegative")
    if not old_summary_path.is_file():
        return {"status": "not_available", "path": str(old_summary_path)}
    reference_sha256 = file_sha256(old_summary_path)
    old = load_json(old_summary_path)
    if file_sha256(old_summary_path) != reference_sha256:
        raise EvaluationError(f"Regression reference changed while reading: {old_summary_path}")
    comparison = contract_comparability(old.get("evaluation_contract"), evaluation_contract)
    if allow_mismatch:
        print("[Audit] --allow-regression-mismatch is deprecated and cannot bypass same-input failures")
    old_metrics = old.get("legacy_metrics")
    if old_metrics is None and "basic_cp" in old and "hiercp" in old:
        old_metrics = {"basic": old["basic_cp"], "hier": old["hiercp"]}
    result = {
        **comparison, "path": str(old_summary_path), "reference_sha256": reference_sha256,
        "tolerance": tolerance, "mismatches": [], "regression_verified": False,
    }
    if old_metrics is None:
        if comparison["status"] == "matched_inputs":
            raise EvaluationError("Same-input regression reference has no legacy metrics")
        result["metric_comparison"] = "unavailable: reference lacks comparable metric fields"
        return result
    checks: list[tuple[str, float, float]] = []
    for method in ("basic", "hier"):
        new_value = legacy_summary[method]
        if not isinstance(old_metrics, Mapping) or not isinstance(old_metrics.get(method), Mapping):
            raise EvaluationError(f"Malformed regression metrics for {method}")
        old_value = old_metrics[method]
        for key in (
            "mean_case_tumor_dice",
            "gt_lesions",
            "detected",
            "false_positive_lesions",
        ):
            if key not in old_value or old_value[key] is None:
                raise EvaluationError(f"Missing regression metric: {method}.{key}")
            new_number, old_number = float(new_value[key]), float(old_value[key])
            if not math.isfinite(new_number) or not math.isfinite(old_number):
                raise EvaluationError(f"Nonfinite regression metric: {method}.{key}")
            checks.append((f"{method}.{key}", new_number, old_number))
    mismatches = [
        {"field": name, "new": new, "old": old_value, "difference": new - old_value}
        for name, new, old_value in checks
        if abs(new - old_value) > tolerance
    ]
    result["mismatches"] = mismatches
    if comparison["status"] != "matched_inputs":
        result["metric_comparison"] = "diagnostic_only: input identity is not verified equal"
        return result
    if mismatches:
        preview = "; ".join(
            f"{item['field']}: new={item['new']} old={item['old']}"
            for item in mismatches
        )
        raise EvaluationError(
            "Same-input regression check failed; no evaluation completion was published. " + preview
        )
    result.update(status="pass", regression_verified=True, metric_comparison="same_verified_inputs")
    return result


def evaluate(args: argparse.Namespace) -> Path:
    project = Path(args.project).expanduser().resolve()
    medical = (
        Path(args.medical_root).expanduser().resolve()
        if args.medical_root
        else project.parent
    )
    online = project / "work" / str(args.online_root)
    paired = project / "work" / str(args.paired_root)
    nnroot = online / "nnunetv2"
    paths = Paths(
        project=project,
        medical=medical,
        data=medical / "Data",
        paired=paired,
        online=online,
        nnroot=nnroot,
        results=nnroot / "nnUNet_results",
        outer_splits=paired / "outer_splits.json",
        nnunet_config=project / "config" / "nnunet.json",
    )
    if not paths.project.is_dir():
        raise EvaluationError(f"Project missing: {paths.project}")
    source_paths = (
        paths.nnunet_config, paths.outer_splits, Path(__file__).resolve(),
        Path(__file__).resolve().with_name("online_eval_provenance.py"),
    )
    source_hashes = {str(path): file_sha256(path) for path in source_paths}
    nn_cfg = load_json(paths.nnunet_config)
    split = outer_split(paths.outer_splits, args.outer_fold)
    val_ids = list(split["val"])
    if not val_ids:
        raise EvaluationError("Outer validation split is empty")
    tumor_label = int(args.tumor_label)
    bins = [float(value) for value in args.size_bins]
    basic_validation = model_validation_dir(
        paths,
        args.dataset_id,
        args.outer_fold,
        nn_cfg,
        args.basic_trainer,
    )
    hier_validation = model_validation_dir(
        paths,
        args.dataset_id,
        args.outer_fold,
        nn_cfg,
        args.hier_trainer,
    )
    for folder in (basic_validation, hier_validation):
        if not folder.is_dir():
            raise EvaluationError(f"Validation prediction folder missing: {folder}")

    output = (
        Path(args.output).expanduser().absolute() if args.output
        else online / "folds" / f"fold_{args.outer_fold}" / "evaluation_v2"
    )
    if output.exists() or output.is_symlink():
        raise EvaluationError(f"Refusing existing evaluation output: {output}. Choose a NEW --output path.")
    for input_folder in (paths.data / "labels", basic_validation, hier_validation):
        if output.resolve().is_relative_to(input_folder.resolve()):
            raise EvaluationError(f"Evaluation output must not be inside input folder: {input_folder}")
    if nib is None:
        raise EvaluationError("nibabel is required for actual NIfTI evaluation")
    if len(bins) != 2 or any(not math.isfinite(x) for x in bins) or not 0 < bins[0] < bins[1]:
        raise EvaluationError("Size bins must be two increasing positive finite diameters")
    for folder in (basic_validation, hier_validation):
        verify_prediction_inventory(folder, val_ids)
    evaluation_contract = build_evaluation_contract(
        cohort=val_ids,
        ground_truth_files={case_id: paths.data / "labels" / f"{case_id}.nii.gz" for case_id in val_ids},
        prediction_files={
            side: {case_id: folder / f"{case_id}.nii.gz" for case_id in val_ids}
            for side, folder in (("basic", basic_validation), ("hier", hier_validation))
        },
        trainers={"basic": args.basic_trainer, "hier": args.hier_trainer},
        evaluation_definition={
            "version": VERSION, "outer_fold": args.outer_fold, "dataset_id": args.dataset_id,
            "plans": nn_cfg["dataset"]["plans"], "configuration": nn_cfg["dataset"]["configuration"],
            "tumor_label": tumor_label, "size_bins_mm": bins, "connectivity": 26,
            "matching": "one-to-one threshold-valid Hungarian; lesion quality maximizes Dice",
            "criteria": [{"name": c.name, "metric": c.metric, "threshold": c.threshold,
                          "legacy": c.legacy} for c in CRITERIA],
            "bootstrap_iterations": args.bootstrap_iterations,
            "permutation_iterations": args.permutation_iterations,
            "cluster_unit": "validation patient", "p_values": "unadjusted",
            "source_sha256": {path.name: source_hashes[str(path)] for path in source_paths},
            "numpy_version": np.__version__, "nibabel_version": nib.__version__,
            "scipy_version": sys.modules["scipy"].__version__, "python_version": sys.version.split()[0],
        },
    )
    prepare_new_output(output)
    atomic_json(output / "evaluation_started.json", {
        "version": VERSION, "complete": False, "evaluation_contract": evaluation_contract,
        "source_files": source_hashes, "training_executed": False,
        "note": "Only completion.json certifies successful publication; partial outputs are not final.",
    })
    print(f"[Contract] full validation cohort={len(val_ids)}; output={output}; no training or overwrite")

    case_rows: list[dict[str, Any]] = []
    case_detection_rows: list[dict[str, Any]] = []
    lesion_detection_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    lesion_quality_case: dict[str, dict[str, float]] = {}

    for index, case_id in enumerate(val_ids, start=1):
        sys.stdout.write(f"\r[Evaluate] {index}/{len(val_ids)} {case_id:<20}")
        sys.stdout.flush()
        reference_path = paths.data / "labels" / f"{case_id}.nii.gz"
        basic_path = basic_validation / f"{case_id}.nii.gz"
        hier_path = hier_validation / f"{case_id}.nii.gz"
        reference_nii, ground_truth = load_label(reference_path, tumor_label)
        basic_nii, basic_mask = load_label(basic_path, tumor_label)
        hier_nii, hier_mask = load_label(hier_path, tumor_label)
        verify_geometry(reference_nii, basic_nii, basic_path)
        verify_geometry(reference_nii, hier_nii, hier_path)
        spacing = tuple(float(value) for value in reference_nii.header.get_zooms()[:3])
        voxel_volume = float(np.prod(spacing))
        basic_matrices = component_pair_matrices(ground_truth, basic_mask)
        hier_matrices = component_pair_matrices(ground_truth, hier_mask)
        if basic_matrices.num_gt != hier_matrices.num_gt:
            raise EvaluationError(f"GT component alignment failed: {case_id}")

        basic_case_dice = whole_mask_dice(ground_truth, basic_mask)
        hier_case_dice = whole_mask_dice(ground_truth, hier_mask)
        case_rows.append(
            {
                "case_id": case_id,
                "outer_fold": args.outer_fold,
                "gt_lesions": basic_matrices.num_gt,
                "basic_predicted_lesions": basic_matrices.num_pred,
                "hier_predicted_lesions": hier_matrices.num_pred,
                "basic_tumor_dice": basic_case_dice,
                "hier_tumor_dice": hier_case_dice,
                "tumor_dice_difference": hier_case_dice - basic_case_dice,
            }
        )

        quality_sums: dict[str, float] = {}
        for method, matrices in (("basic", basic_matrices), ("hier", hier_matrices)):
            quality_match = max_dice_match(matrices)
            total_dice = 0.0
            for gt_index in range(matrices.num_gt):
                pred_index = quality_match.gt_to_pred.get(gt_index)
                total_dice += 0.0 if pred_index is None else float(matrices.dice[gt_index, pred_index])
            quality_sums[method] = total_dice
        lesion_quality_case[case_id] = {
            "count": float(basic_matrices.num_gt),
            "basic_sum": quality_sums["basic"],
            "hier_sum": quality_sums["hier"],
        }

        for criterion in CRITERIA:
            for method, matrices in (("basic", basic_matrices), ("hier", hier_matrices)):
                matched = match_components(matrices, criterion)
                tp = matched.tp
                fp = matrices.num_pred - tp
                fn = matrices.num_gt - tp
                recall = tp / matrices.num_gt if matrices.num_gt else None
                precision = tp / matrices.num_pred if matrices.num_pred else None
                f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
                case_detection_rows.append(
                    {
                        "case_id": case_id,
                        "outer_fold": args.outer_fold,
                        "criterion": criterion.name,
                        "criterion_label": criterion.label,
                        "method": method,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "gt_lesions": matrices.num_gt,
                        "predicted_lesions": matrices.num_pred,
                        "recall": recall,
                        "precision": precision,
                        "f1": f1,
                    }
                )
                for gt_index in range(matrices.num_gt):
                    diameter = equivalent_diameter(
                        float(matrices.gt_sizes[gt_index]) * voxel_volume
                    )
                    pred_index = matched.gt_to_pred.get(gt_index)
                    values = (
                        pair_values(matrices, gt_index, pred_index)
                        if pred_index is not None
                        else {
                            "intersection_vox": 0,
                            "gt_vox": int(matrices.gt_sizes[gt_index]),
                            "pred_vox": 0,
                            "dice": 0.0,
                            "iou": 0.0,
                            "gt_coverage": 0.0,
                            "pred_coverage": 0.0,
                        }
                    )
                    lesion_detection_rows.append(
                        {
                            "case_id": case_id,
                            "outer_fold": args.outer_fold,
                            "gt_component": gt_index + 1,
                            "diameter_mm": diameter,
                            "size_bin": size_bin(diameter, bins),
                            "criterion": criterion.name,
                            "criterion_label": criterion.label,
                            "method": method,
                            "detected": int(pred_index is not None),
                            "pred_component": "" if pred_index is None else pred_index + 1,
                            **values,
                        }
                    )
                for pred_index in range(matrices.num_pred):
                    gt_index = matched.pred_to_gt.get(pred_index)
                    values = (
                        pair_values(matrices, gt_index, pred_index)
                        if gt_index is not None
                        else {
                            "intersection_vox": 0,
                            "gt_vox": 0,
                            "pred_vox": int(matrices.pred_sizes[pred_index]),
                            "dice": 0.0,
                            "iou": 0.0,
                            "gt_coverage": 0.0,
                            "pred_coverage": 0.0,
                        }
                    )
                    prediction_rows.append(
                        {
                            "case_id": case_id,
                            "outer_fold": args.outer_fold,
                            "criterion": criterion.name,
                            "criterion_label": criterion.label,
                            "method": method,
                            "pred_component": pred_index + 1,
                            "matched_gt_component": "" if gt_index is None else gt_index + 1,
                            "false_positive": int(gt_index is None),
                            **values,
                        }
                    )
    print()

    # Long-form files are intentionally auditable and criterion-specific.
    atomic_csv(
        output / "case_metrics.csv",
        case_rows,
        (
            "case_id",
            "outer_fold",
            "gt_lesions",
            "basic_predicted_lesions",
            "hier_predicted_lesions",
            "basic_tumor_dice",
            "hier_tumor_dice",
            "tumor_dice_difference",
        ),
    )
    atomic_csv(
        output / "case_detection_metrics.csv",
        case_detection_rows,
        (
            "case_id",
            "outer_fold",
            "criterion",
            "criterion_label",
            "method",
            "tp",
            "fp",
            "fn",
            "gt_lesions",
            "predicted_lesions",
            "recall",
            "precision",
            "f1",
        ),
    )
    atomic_csv(
        output / "lesion_detection_metrics.csv",
        lesion_detection_rows,
        (
            "case_id",
            "outer_fold",
            "gt_component",
            "diameter_mm",
            "size_bin",
            "criterion",
            "criterion_label",
            "method",
            "detected",
            "pred_component",
            "intersection_vox",
            "gt_vox",
            "pred_vox",
            "dice",
            "iou",
            "gt_coverage",
            "pred_coverage",
        ),
    )
    atomic_csv(
        output / "prediction_detection_metrics.csv",
        prediction_rows,
        (
            "case_id",
            "outer_fold",
            "criterion",
            "criterion_label",
            "method",
            "pred_component",
            "matched_gt_component",
            "false_positive",
            "intersection_vox",
            "gt_vox",
            "pred_vox",
            "dice",
            "iou",
            "gt_coverage",
            "pred_coverage",
        ),
    )

    case_ids = [str(row["case_id"]) for row in case_rows]
    basic_case_dice = np.asarray([float(row["basic_tumor_dice"]) for row in case_rows])
    hier_case_dice = np.asarray([float(row["hier_tumor_dice"]) for row in case_rows])
    case_dice_stats = {
        **paired_cluster_bootstrap_mean(
            basic_case_dice,
            hier_case_dice,
            31000 + args.outer_fold,
            args.bootstrap_iterations,
        ),
        "permutation_p": paired_swap_permutation_mean(
            basic_case_dice,
            hier_case_dice,
            32000 + args.outer_fold,
            args.permutation_iterations,
        ),
        "wilcoxon_p": paired_wilcoxon(basic_case_dice, hier_case_dice),
    }

    quality_counts = np.asarray(
        [lesion_quality_case[case_id]["count"] for case_id in case_ids],
        dtype=np.float64,
    )
    basic_quality_sum = np.asarray(
        [lesion_quality_case[case_id]["basic_sum"] for case_id in case_ids],
        dtype=np.float64,
    )
    hier_quality_sum = np.asarray(
        [lesion_quality_case[case_id]["hier_sum"] for case_id in case_ids],
        dtype=np.float64,
    )
    lesion_dice_stats = {
        **cluster_bootstrap_lesion_mean(
            basic_quality_sum,
            hier_quality_sum,
            quality_counts,
            33000 + args.outer_fold,
            args.bootstrap_iterations,
        ),
        "permutation_p": cluster_permutation_lesion_mean(
            basic_quality_sum,
            hier_quality_sum,
            quality_counts,
            34000 + args.outer_fold,
            args.permutation_iterations,
        ),
        "basic_mean": float(basic_quality_sum.sum() / quality_counts.sum()),
        "hier_mean": float(hier_quality_sum.sum() / quality_counts.sum()),
        "matching": "one-to-one Hungarian maximizing lesion Dice; unmatched GT lesions receive Dice 0",
    }

    criterion_summary_rows: list[dict[str, Any]] = []
    size_summary_rows: list[dict[str, Any]] = []
    statistics_rows: list[dict[str, Any]] = []
    criteria_summary: dict[str, Any] = {}
    all_size_bins = sorted({str(row["size_bin"]) for row in lesion_detection_rows}, key=natural_key)

    for criterion_index, criterion in enumerate(CRITERIA):
        selected = [row for row in case_detection_rows if row["criterion"] == criterion.name]
        basic_summary = _method_summary(selected, "basic")
        hier_summary = _method_summary(selected, "hier")
        basic_counts = _counts_array(
            case_detection_rows, case_ids, criterion.name, "basic"
        )
        hier_counts = _counts_array(
            case_detection_rows, case_ids, criterion.name, "hier"
        )
        metric_statistics: dict[str, Any] = {}
        for metric_index, metric in enumerate(("recall", "precision", "f1", "fp_per_case")):
            seed_base = 40000 + args.outer_fold * 1000 + criterion_index * 100 + metric_index * 10
            stats = {
                **cluster_bootstrap_count_difference(
                    basic_counts,
                    hier_counts,
                    metric,
                    seed_base,
                    args.bootstrap_iterations,
                ),
                **cluster_permutation_count_inference(
                    basic_counts,
                    hier_counts,
                    metric,
                    seed_base + 1,
                    args.permutation_iterations,
                ),
            }
            metric_statistics[metric] = stats
            statistics_rows.append(
                {
                    "scope": "overall",
                    "criterion": criterion.name,
                    "criterion_label": criterion.label,
                    "size_bin": "",
                    "metric": metric,
                    **stats,
                }
            )
        by_size: dict[str, Any] = {}
        for size_index, selected_bin in enumerate(all_size_bins):
            basic_size = _size_counts_array(
                lesion_detection_rows,
                case_ids,
                criterion.name,
                "basic",
                selected_bin,
            )
            hier_size = _size_counts_array(
                lesion_detection_rows,
                case_ids,
                criterion.name,
                "hier",
                selected_bin,
            )
            basic_tp = int(basic_size[:, 0].sum())
            hier_tp = int(hier_size[:, 0].sum())
            gt = int((basic_size[:, 0] + basic_size[:, 2]).sum())
            basic_recall = basic_tp / gt if gt else None
            hier_recall = hier_tp / gt if gt else None
            seed_base = 50000 + args.outer_fold * 1000 + criterion_index * 100 + size_index * 10
            stats = {
                **cluster_bootstrap_count_difference(
                    basic_size,
                    hier_size,
                    "recall",
                    seed_base,
                    args.bootstrap_iterations,
                ),
                **cluster_permutation_count_inference(
                    basic_size,
                    hier_size,
                    "recall",
                    seed_base + 1,
                    args.permutation_iterations,
                ),
            }
            by_size[selected_bin] = {
                "gt": gt,
                "basic_detected": basic_tp,
                "hier_detected": hier_tp,
                "basic_recall": basic_recall,
                "hier_recall": hier_recall,
                "statistics": stats,
            }
            size_summary_rows.append(
                {
                    "criterion": criterion.name,
                    "criterion_label": criterion.label,
                    "size_bin": selected_bin,
                    "gt": gt,
                    "basic_detected": basic_tp,
                    "hier_detected": hier_tp,
                    "basic_recall": basic_recall,
                    "hier_recall": hier_recall,
                    **stats,
                }
            )
            statistics_rows.append(
                {
                    "scope": "size_bin",
                    "criterion": criterion.name,
                    "criterion_label": criterion.label,
                    "size_bin": selected_bin,
                    "metric": "recall",
                    **stats,
                }
            )
        criteria_summary[criterion.name] = {
            "label": criterion.label,
            "definition": {
                "metric": criterion.metric,
                "threshold": criterion.threshold,
                "legacy": criterion.legacy,
            },
            "basic_cp": basic_summary,
            "hiercp": hier_summary,
            "difference_hier_minus_basic": {
                key: (
                    None
                    if basic_summary[key] is None or hier_summary[key] is None
                    else float(hier_summary[key] - basic_summary[key])
                )
                for key in ("recall", "precision", "f1", "fp_per_case")
            },
            "statistics": metric_statistics,
            "by_size": by_size,
        }
        for method, method_summary in (("basic", basic_summary), ("hier", hier_summary)):
            criterion_summary_rows.append(
                {
                    "criterion": criterion.name,
                    "criterion_label": criterion.label,
                    "method": method,
                    **method_summary,
                }
            )

    atomic_csv(
        output / "criterion_summary.csv",
        criterion_summary_rows,
        (
            "criterion",
            "criterion_label",
            "method",
            "tp",
            "fp",
            "fn",
            "recall",
            "precision",
            "f1",
            "fp_per_case",
        ),
    )
    atomic_csv(
        output / "size_recall_summary.csv",
        size_summary_rows,
        (
            "criterion",
            "criterion_label",
            "size_bin",
            "gt",
            "basic_detected",
            "hier_detected",
            "basic_recall",
            "hier_recall",
            "difference",
            "ci_low",
            "ci_high",
            "permutation_p",
        ),
    )
    atomic_csv(
        output / "patient_cluster_statistics.csv",
        statistics_rows,
        (
            "scope",
            "criterion",
            "criterion_label",
            "size_bin",
            "metric",
            "difference",
            "ci_low",
            "ci_high",
            "permutation_p",
        ),
    )

    legacy = criteria_summary["legacy_any_overlap"]
    legacy_for_regression = {
        "basic": {
            "mean_case_tumor_dice": float(np.mean(basic_case_dice)),
            "gt_lesions": legacy["basic_cp"]["tp"] + legacy["basic_cp"]["fn"],
            "detected": legacy["basic_cp"]["tp"],
            "false_positive_lesions": legacy["basic_cp"]["fp"],
        },
        "hier": {
            "mean_case_tumor_dice": float(np.mean(hier_case_dice)),
            "gt_lesions": legacy["hiercp"]["tp"] + legacy["hiercp"]["fn"],
            "detected": legacy["hiercp"]["tp"],
            "false_positive_lesions": legacy["hiercp"]["fp"],
        },
    }
    regression = _regression_check(
        legacy_for_regression,
        (Path(args.regression_reference).expanduser().absolute()
         if args.regression_reference else
         online / "folds" / f"fold_{args.outer_fold}" / "evaluation" / "summary.json"),
        args.regression_tolerance,
        args.allow_regression_mismatch,
        evaluation_contract=evaluation_contract,
    )

    summary = {
        "version": VERSION,
        "evaluation_contract": evaluation_contract,
        "source_files": source_hashes,
        "legacy_metrics": legacy_for_regression,
        "prediction_origin_status": "existing_prediction_bytes_verified; original_training_checkpoint_link_unverified",
        "outer_fold": args.outer_fold,
        "dataset_id": args.dataset_id,
        "dataset": dataset_name(args.dataset_id, args.outer_fold),
        "validation_cases": len(case_rows),
        "tumor_label": tumor_label,
        "size_bins_mm": bins,
        "basic_validation": str(basic_validation),
        "hier_validation": str(hier_validation),
        "inference_note": (
            "No post-hoc primary detection threshold is selected. Legacy any-overlap is "
            "retained for compatibility; all quality-aware criteria are reported as a fixed "
            "sensitivity suite. Statistical resampling uses validation patients as clusters."
        ),
        "legacy_regression": regression,
        "case_tumor_dice": {
            "basic_mean": float(np.mean(basic_case_dice)),
            "hier_mean": float(np.mean(hier_case_dice)),
            "statistics": case_dice_stats,
        },
        "lesion_dice_quality": lesion_dice_stats,
        "criteria": criteria_summary,
        "resampling": {
            "bootstrap_iterations": args.bootstrap_iterations,
            "permutation_iterations": args.permutation_iterations,
            "cluster_unit": "validation patient",
        },
    }
    atomic_json(output / "summary.json", summary)

    def fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "NA"
        return f"{float(value):.{digits}f}"

    lines = [
        f"# Online Basic-CP vs Online HierCP — Quality-aware Re-evaluation (Outer Fold {args.outer_fold})",
        "",
        f"- Validation patients: {len(case_rows)}",
        "- Existing predictions only; no retraining and no prediction modification",
        f"- Legacy any-overlap reference audit: {regression['status']}; regression is verified only when input provenance matches",
        "- Quality-aware criteria are a fixed sensitivity suite; no post-hoc primary threshold is selected",
        "- Confidence intervals and permutation tests resample/swap whole patients, not individual lesions",
        "- Reported p-values are unadjusted; this sensitivity suite does not establish a post-hoc primary endpoint",
        "- Undefined count-metric or lesion-mean bootstrap resamples, or undefined count-metric swap resamples, make the corresponding inference NA; invalid draws are not silently discarded or replaced with zero",
        "- Resampling availability, reasons, and valid/invalid draw counts are recorded with the affected statistics in summary.json",
        "",
        "## Whole-volume tumor Dice",
        "",
        "| Basic-CP | HierCP | Hier−Basic | Patient-bootstrap 95% CI | Patient-swap p | Wilcoxon p |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {fmt(np.mean(basic_case_dice))} | {fmt(np.mean(hier_case_dice))} | "
            f"{fmt(case_dice_stats['difference'])} | "
            f"[{fmt(case_dice_stats['ci_low'])}, {fmt(case_dice_stats['ci_high'])}] | "
            f"{fmt(case_dice_stats['permutation_p'])} | {fmt(case_dice_stats['wilcoxon_p'])} |"
        ),
        "",
        "## Lesion detection sensitivity suite",
        "",
        "| Criterion | Basic recall | Hier recall | Δ recall [95% CI] | p | Basic precision | Hier precision | Basic F1 | Hier F1 | Basic FP/case | Hier FP/case |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for criterion in CRITERIA:
        item = criteria_summary[criterion.name]
        basic = item["basic_cp"]
        hier = item["hiercp"]
        recall_stats = item["statistics"]["recall"]
        lines.append(
            f"| {criterion.label} | {fmt(basic['recall'])} | {fmt(hier['recall'])} | "
            f"{fmt(recall_stats['difference'])} [{fmt(recall_stats['ci_low'])}, {fmt(recall_stats['ci_high'])}] | "
            f"{fmt(recall_stats['permutation_p'])} | {fmt(basic['precision'])} | {fmt(hier['precision'])} | "
            f"{fmt(basic['f1'])} | {fmt(hier['f1'])} | {fmt(basic['fp_per_case'], 3)} | {fmt(hier['fp_per_case'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Size-stratified recall",
            "",
        ]
    )
    for criterion in CRITERIA:
        lines.extend(
            [
                f"### {criterion.label}",
                "",
                "| Size bin | GT | Basic detected | Hier detected | Basic recall | Hier recall | Δ recall [95% CI] | p |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for selected_bin in all_size_bins:
            value = criteria_summary[criterion.name]["by_size"][selected_bin]
            stats = value["statistics"]
            lines.append(
                f"| {selected_bin} | {value['gt']} | {value['basic_detected']} | {value['hier_detected']} | "
                f"{fmt(value['basic_recall'])} | {fmt(value['hier_recall'])} | "
                f"{fmt(stats['difference'])} [{fmt(stats['ci_low'])}, {fmt(stats['ci_high'])}] | "
                f"{fmt(stats['permutation_p'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Lesion overlap quality",
            "",
            "One-to-one matching maximizes lesion Dice; unmatched GT lesions receive Dice 0.",
            "",
            "| Basic mean | Hier mean | Hier−Basic | Patient-cluster bootstrap 95% CI | Patient-swap p |",
            "|---:|---:|---:|---:|---:|",
            (
                f"| {fmt(lesion_dice_stats['basic_mean'])} | {fmt(lesion_dice_stats['hier_mean'])} | "
                f"{fmt(lesion_dice_stats['difference'])} | "
                f"[{fmt(lesion_dice_stats['ci_low'])}, {fmt(lesion_dice_stats['ci_high'])}] | "
                f"{fmt(lesion_dice_stats['permutation_p'])} |"
            ),
            "",
            "## Audit",
            "",
            f"- Legacy regression: {regression['status']}",
            "- A report alone is not a completion certificate; verify completion.json and its hashes.",
            "- Existing prediction bytes are verified; their original training-checkpoint linkage remains unverified.",
            f"- Output: `{output}`",
        ]
    )
    atomic_text(output / "comparison.md", "\n".join(lines) + "\n")

    # Recheck the exact inputs before publishing the only completion certificate.
    verify_evaluation_contract(evaluation_contract)
    for folder in (basic_validation, hier_validation):
        verify_prediction_inventory(folder, val_ids)
    for name, expected_sha256 in source_hashes.items():
        if file_sha256(Path(name)) != expected_sha256:
            raise EvaluationError(f"Evaluation source/config/split changed during execution: {name}")
    if "reference_sha256" in regression:
        if file_sha256(Path(regression["path"])) != regression["reference_sha256"]:
            raise EvaluationError("Regression reference changed during evaluation")
    report_hashes = {path.name: file_sha256(path) for path in output.iterdir() if path.is_file()}
    atomic_json(output / "completion.json", {
        "format": "online_eval_completion_v1", "complete": True,
        "summary_sha256": report_hashes["summary.json"], "outputs": report_hashes,
        "evaluation_contract_sha256": evaluation_contract["contract_sha256"],
        "actual_data_evaluation": True, "training_executed": False,
    })

    print(f"[OK] quality-aware evaluation: {output / 'comparison.md'}")
    print(f"[Audit] legacy regression: {regression['status']}")
    return output


# ---------- self-test ----------

def self_test() -> None:
    gt = np.zeros((12, 12, 12), dtype=bool)
    gt[1:3, 1:3, 1:3] = True
    gt[7:10, 7:10, 7:10] = True
    pred = np.zeros_like(gt)
    pred[1:3, 1:3, 1:3] = True       # perfect first lesion
    pred[7, 7, 7] = True             # one-voxel touch of second lesion
    pred[4:6, 4:6, 4:6] = True       # false positive
    matrices = component_pair_matrices(gt, pred)
    if matrices.num_gt != 2 or matrices.num_pred != 3:
        raise AssertionError((matrices.num_gt, matrices.num_pred))
    legacy = match_components(matrices, CRITERIA[0])
    if legacy.tp != 2:
        raise AssertionError(f"legacy TP={legacy.tp}")
    dice_025 = match_components(matrices, CRITERIA[2])
    if dice_025.tp != 1:
        raise AssertionError(f"dice>=0.25 TP={dice_025.tp}")
    basic = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=float)
    hier = np.asarray([[1, 0, 0], [1, 1, 0]], dtype=float)
    result = cluster_bootstrap_count_difference(basic, hier, "recall", 1, 1000)
    if result["difference"] is None:
        raise AssertionError(result)
    print("[OK] component matching smoke")
    print("[OK] patient-cluster bootstrap smoke")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--project", default="/home/aicompetition06/Medical/HierCP")
    value.add_argument("--medical-root")
    value.add_argument("--paired-root", default=DEFAULT_PAIRED_ROOT)
    value.add_argument("--online-root", default=DEFAULT_ONLINE_ROOT)
    value.add_argument("--outer-fold", type=int, default=0)
    value.add_argument("--dataset-id", type=int, default=DEFAULT_DATASET_ID)
    value.add_argument("--basic-trainer", default=DEFAULT_BASIC_TRAINER)
    value.add_argument("--hier-trainer", default=DEFAULT_HIER_TRAINER)
    value.add_argument("--tumor-label", type=int, default=2)
    value.add_argument("--size-bins", type=float, nargs=2, default=(10.0, 20.0))
    value.add_argument("--bootstrap-iterations", type=int, default=20000)
    value.add_argument("--permutation-iterations", type=int, default=50000)
    value.add_argument("--regression-tolerance", type=float, default=1e-9)
    value.add_argument("--allow-regression-mismatch", action="store_true",
                       help="Deprecated; never bypasses a same-input regression failure")
    value.add_argument("--regression-reference", help="Optional previous evaluation summary with input provenance")
    value.add_argument("--output")
    value.add_argument("--self-test", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if not math.isfinite(args.regression_tolerance) or args.regression_tolerance < 0:
        raise EvaluationError("regression-tolerance must be finite and nonnegative")
    if args.bootstrap_iterations < 1000:
        raise EvaluationError("bootstrap-iterations must be >= 1000")
    if args.permutation_iterations < 1000:
        raise EvaluationError("permutation-iterations must be >= 1000")
    if args.self_test:
        self_test()
        return
    evaluate(args)


if __name__ == "__main__":
    try:
        main()
    except (EvaluationError, EvaluationProvenanceError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print("[Recovery] Preserve existing inputs/results. A new output without completion.json is incomplete; correct the error and use a new output path.", file=sys.stderr)
        raise SystemExit(1)
