#!/usr/bin/env python3
"""Leakage-safe paired Basic-CP versus HierCP benchmark.

The benchmark treats the patient split and the augmentation method as separate
experimental axes. For every outer fold, Basic-CP and HierCP use:

- exactly the same original training and validation patients;
- exactly the same source tumor;
- exactly the same valid candidate pool and hard anatomical constraints;
- exactly the same intensity scale/shift;
- exactly one synthetic case per eligible training patient; and
- the same nnU-Net plans, trainer, configuration and local fold number.

The only intended difference is target selection:

- Basic-CP samples one candidate uniformly from the shared valid pool.
- HierCP selects the highest-scoring candidate with a fold-specific GNN.

A fold-specific HierCP model is trained only from that outer fold's training
patients. The outer validation patients are not used for prototype fitting,
GNN graph preparation, GNN checkpoint selection, synthetic generation, or
nnU-Net training.
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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.stats import binom, wilcoxon

# Make the sibling ``hiercp`` package importable when this file is launched
# directly through the Medical/pairedcp wrapper.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

VERSION = "paired_basic_hiercp_nested_v1"
OUTER_SPLIT_FORMAT = "paired_cp_outer_split_v1"
INNER_SPLIT_FORMAT = "hiercp_case_split_v1"
DEFAULT_DATASET_ID_BASE = 720
METHODS = ("basic", "hier")


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class Case:
    case_id: str
    image: Path
    label: Path


@dataclass(frozen=True)
class Profile:
    case_id: str
    tumor_components: int
    le10: int
    gt10_le20: int
    gt20: int
    tumor_voxels: int
    stratum: str


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: int
    dataset_name: str
    condition: str


@dataclass(frozen=True)
class Layout:
    project: Path
    medical: Path
    data: Path
    benchmark: Path
    source_work: Path
    train_config: Path
    nnunet_config: Path
    outer_splits: Path
    profiles_csv: Path
    nnroot: Path
    raw: Path
    preprocessed: Path
    results: Path
    logs: Path

    def fold(self, outer_fold: int) -> Path:
        return self.benchmark / "folds" / f"fold_{outer_fold}"

    def gnn(self, outer_fold: int) -> Path:
        return self.fold(outer_fold) / "gnn"

    def paired(self, outer_fold: int) -> Path:
        return self.fold(outer_fold) / "paired"

    def evaluation(self, outer_fold: int) -> Path:
        return self.fold(outer_fold) / "evaluation"


def natural_key(text: str) -> list[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkError(f"Missing JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"JSON root must be an object: {path}")
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


def atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
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


def discover_cases(root: Path) -> list[Case]:
    image_dir = root / "image"
    label_dir = root / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise BenchmarkError(f"Expected image/ and labels/ under {root}")
    cases: list[Case] = []
    for image in sorted(image_dir.glob("*_0000.nii.gz"), key=lambda p: natural_key(p.name)):
        case_id = image.name[: -len("_0000.nii.gz")]
        label = label_dir / f"{case_id}.nii.gz"
        if not label.is_file():
            raise BenchmarkError(f"Missing label for {image}: {label}")
        cases.append(Case(case_id, image.resolve(), label.resolve()))
    if not cases:
        raise BenchmarkError(f"No image/label pairs under {root}")
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
            raise BenchmarkError(f"Cannot infer channel axis: {path} shape={view.shape}")
    if len(view.shape) != 3:
        raise BenchmarkError(f"Expected 3D NIfTI: {path} shape={view.shape}")
    return view, np.asarray(view.dataobj, dtype=dtype, order="C")


def equivalent_diameter(volume_mm3: float) -> float:
    if volume_mm3 <= 0:
        return 0.0
    return float(2.0 * (3.0 * volume_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0))


def file_signature(path: Path) -> tuple[str, int, int]:
    stat = path.resolve().stat()
    return str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)


def case_fingerprint(cases: Sequence[Case]) -> str:
    payload = [
        (case.case_id, file_signature(case.image), file_signature(case.label))
        for case in cases
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def profile_case(case: Case, tumor_label: int) -> Profile:
    nii, label = load_3d(case.label, np.int16)
    tumor = label == int(tumor_label)
    components, count = ndi.label(tumor, structure=np.ones((3, 3, 3), np.uint8))
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    voxel_mm3 = float(np.prod(nii.header.get_zooms()[:3]))
    le10 = 0
    mid = 0
    gt20 = 0
    for component_id in range(1, count + 1):
        diameter = equivalent_diameter(float(sizes[component_id]) * voxel_mm3)
        if diameter <= 10.0:
            le10 += 1
        elif diameter <= 20.0:
            mid += 1
        else:
            gt20 += 1
    if count == 0:
        stratum = "no_tumor"
    elif le10 > 0:
        stratum = "has_le10"
    elif mid > 0:
        stratum = "has_10_20"
    else:
        stratum = "gt20_only"
    return Profile(
        case_id=case.case_id,
        tumor_components=int(count),
        le10=int(le10),
        gt10_le20=int(mid),
        gt20=int(gt20),
        tumor_voxels=int(tumor.sum()),
        stratum=stratum,
    )


def write_profiles(path: Path, profiles: Sequence[Profile]) -> None:
    fields = (
        "case_id",
        "tumor_components",
        "le10",
        "gt10_le20",
        "gt20",
        "tumor_voxels",
        "stratum",
    )
    atomic_csv(path, [profile.__dict__ for profile in profiles], fields)


def read_profiles(path: Path) -> list[Profile]:
    if not path.is_file():
        raise BenchmarkError(f"Missing case profiles: {path}")
    rows: list[Profile] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                Profile(
                    case_id=row["case_id"],
                    tumor_components=int(row["tumor_components"]),
                    le10=int(row["le10"]),
                    gt10_le20=int(row["gt10_le20"]),
                    gt20=int(row["gt20"]),
                    tumor_voxels=int(row["tumor_voxels"]),
                    stratum=row["stratum"],
                )
            )
    return rows


def balanced_folds(
    profiles: Sequence[Profile],
    n_splits: int,
    seed: int,
) -> list[list[str]]:
    if n_splits < 2:
        raise BenchmarkError("n_splits must be >= 2")
    if len(profiles) < n_splits:
        raise BenchmarkError("Fewer cases than folds")
    rng = np.random.default_rng(int(seed))
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    stratum_counts: list[dict[str, int]] = [dict() for _ in range(n_splits)]
    grouped: dict[str, list[Profile]] = {}
    for profile in profiles:
        grouped.setdefault(profile.stratum, []).append(profile)

    # Rare strata are allocated first. Within a stratum, lesion-rich cases are
    # assigned first so total lesion burden is also distributed.
    for stratum in sorted(grouped, key=lambda key: (len(grouped[key]), key)):
        group = grouped[stratum]
        order = rng.permutation(len(group)).tolist()
        shuffled = [group[index] for index in order]
        shuffled.sort(
            key=lambda item: (
                item.le10,
                item.gt10_le20,
                item.gt20,
                item.tumor_components,
                item.tumor_voxels,
            ),
            reverse=True,
        )
        for profile in shuffled:
            candidates = list(range(n_splits))
            rng.shuffle(candidates)
            candidates.sort(
                key=lambda fold: (
                    stratum_counts[fold].get(stratum, 0),
                    len(folds[fold]),
                    sum(
                        next(
                            p.tumor_components
                            for p in profiles
                            if p.case_id == case_id
                        )
                        for case_id in folds[fold]
                    ),
                )
            )
            selected = candidates[0]
            folds[selected].append(profile.case_id)
            stratum_counts[selected][stratum] = (
                stratum_counts[selected].get(stratum, 0) + 1
            )

    for fold in folds:
        fold.sort(key=natural_key)
    flattened = [case_id for fold in folds for case_id in fold]
    if len(flattened) != len(set(flattened)) or set(flattened) != {
        profile.case_id for profile in profiles
    }:
        raise BenchmarkError("Balanced fold assignment lost or duplicated cases")
    return folds


def build_outer_splits(layout: Layout, train_cfg: Mapping[str, Any], overwrite: bool) -> None:
    cases = discover_cases(layout.data)
    fingerprint = case_fingerprint(cases)
    outer_seed = int(train_cfg.get("seed", 42)) + 270827
    if layout.outer_splits.is_file() and not overwrite:
        current = load_json(layout.outer_splits)
        if current.get("version") != VERSION or current.get("fingerprint") != fingerprint:
            raise BenchmarkError(
                "Existing outer split does not match current data. Use a new benchmark "
                "directory or --overwrite."
            )
        print(f"[Reuse] outer split: {layout.outer_splits}")
        return

    tumor_label = int(train_cfg["labels"]["tumor"])
    profiles: list[Profile] = []
    for index, case in enumerate(cases, start=1):
        profile = profile_case(case, tumor_label)
        profiles.append(profile)
        print(
            f"[Profile] {index}/{len(cases)} {case.case_id} "
            f"stratum={profile.stratum} lesions={profile.tumor_components}",
            flush=True,
        )
    profiles.sort(key=lambda item: natural_key(item.case_id))
    write_profiles(layout.profiles_csv, profiles)
    val_folds = balanced_folds(profiles, 5, outer_seed)
    all_ids = {profile.case_id for profile in profiles}
    splits = []
    for fold_index, val in enumerate(val_folds):
        val_set = set(val)
        train = sorted(all_ids - val_set, key=natural_key)
        splits.append({"fold": fold_index, "train": train, "val": list(val)})
    payload = {
        "format": OUTER_SPLIT_FORMAT,
        "version": VERSION,
        "fingerprint": fingerprint,
        "seed": outer_seed,
        "splits": splits,
    }
    atomic_json(layout.outer_splits, payload)
    for split in splits:
        print(
            f"[OK] outer fold {split['fold']}: "
            f"train={len(split['train'])} val={len(split['val'])}"
        )


def outer_split(layout: Layout, outer_fold: int) -> dict[str, Any]:
    payload = load_json(layout.outer_splits)
    if payload.get("format") != OUTER_SPLIT_FORMAT:
        raise BenchmarkError(f"Unsupported outer split format: {payload.get('format')}")
    splits = payload.get("splits")
    if not isinstance(splits, list) or outer_fold not in range(len(splits)):
        raise BenchmarkError(f"Outer fold is unavailable: {outer_fold}")
    split = splits[outer_fold]
    return {
        "fold": int(split["fold"]),
        "train": [str(value) for value in split["train"]],
        "val": [str(value) for value in split["val"]],
    }


def ensure_inner_split(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    overwrite: bool,
) -> Path:
    split = outer_split(layout, outer_fold)
    path = layout.gnn(outer_fold) / "split.json"
    if path.is_file() and not overwrite:
        current = load_json(path)
        if (
            current.get("format") != INNER_SPLIT_FORMAT
            or set(current.get("train", [])) | set(current.get("val", []))
            != set(split["train"])
            or set(current.get("train", [])) & set(current.get("val", []))
        ):
            raise BenchmarkError(f"Existing inner split is incompatible: {path}")
        return path

    profile_map = {profile.case_id: profile for profile in read_profiles(layout.profiles_csv)}
    profiles = [profile_map[case_id] for case_id in split["train"]]
    inner_seed = int(train_cfg.get("seed", 42)) + 1000 + int(outer_fold)
    inner_folds = balanced_folds(profiles, 5, inner_seed)
    inner_val = set(inner_folds[outer_fold % 5])
    inner_train = sorted(set(split["train"]) - inner_val, key=natural_key)
    payload = {
        "format": INNER_SPLIT_FORMAT,
        "seed": inner_seed,
        "val_fraction": len(inner_val) / len(split["train"]),
        "train": inner_train,
        "val": sorted(inner_val, key=natural_key),
        "outer_fold": int(outer_fold),
        "outer_validation_excluded": split["val"],
    }
    atomic_json(path, payload)
    print(
        f"[OK] inner split fold={outer_fold}: "
        f"train={len(inner_train)} val={len(inner_val)} "
        f"outer_test_excluded={len(split['val'])}"
    )
    return path


def require_command(command: str) -> str:
    value = shutil.which(command)
    if value is None:
        raise BenchmarkError(f"Command not found in active environment: {command}")
    return value


def quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    log: Path | None = None,
    dry_run: bool = False,
) -> None:
    values = [str(value) for value in command]
    print("\n$ " + " ".join(quote(value) for value in values), flush=True)
    if dry_run:
        return
    handle = None
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("a", encoding="utf-8", buffering=1)
        handle.write(f"\n[{time.strftime('%FT%T')}] $ {' '.join(values)}\n")
    process = subprocess.Popen(
        values,
        cwd=str(cwd),
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        if handle is not None:
            handle.write(line)
    return_code = process.wait()
    if handle is not None:
        handle.close()
    if return_code != 0:
        raise BenchmarkError(
            f"Command failed ({return_code}): {' '.join(values)}"
        )


def gnn_paths(layout: Layout, outer_fold: int) -> dict[str, Path]:
    root = layout.gnn(outer_fold)
    return {
        "root": root,
        "split": root / "split.json",
        "prototype": root / "prototype.pt",
        "regions": root / "regions",
        "graphs": root / "graphs",
        "model": root / "model.pt",
        "last": root / "model.last.pt",
        "causality": root / "causality.json",
        "log": root / "run.log",
    }


def gnn_prepare(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    overwrite: bool,
    dry_run: bool,
) -> None:
    paths = gnn_paths(layout, outer_fold)
    split_path = ensure_inner_split(layout, outer_fold, train_cfg, overwrite)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["regions"].mkdir(parents=True, exist_ok=True)
    python = sys.executable
    common = [
        "--run-mode",
        "benchmark",
        "--config",
        str(layout.train_config),
        "--data-dir",
        str(layout.data),
        "--split-file",
        str(split_path),
        "--region-cache-dir",
        str(paths["regions"]),
        "--seed",
        str(int(train_cfg.get("seed", 42)) + outer_fold),
    ]
    prototype = [
        python,
        "-m",
        "hiercp.pipeline",
        "prepare-prototypes",
        *common,
        "--output",
        str(paths["prototype"]),
    ]
    graphs = [
        python,
        "-m",
        "hiercp.pipeline",
        "prepare",
        *common,
        "--prototype-bank",
        str(paths["prototype"]),
        "--cache-dir",
        str(paths["graphs"]),
    ]
    if overwrite:
        prototype.append("--overwrite")
        graphs.append("--overwrite")
    run_command(
        prototype,
        cwd=layout.project,
        log=paths["log"],
        dry_run=dry_run,
    )
    run_command(graphs, cwd=layout.project, log=paths["log"], dry_run=dry_run)


def gnn_train(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    device: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    paths = gnn_paths(layout, outer_fold)
    if not dry_run:
        if not paths["prototype"].is_file():
            raise BenchmarkError(f"Missing fold-specific prototype: {paths['prototype']}")
        if not list(paths["graphs"].glob("*.pt")):
            raise BenchmarkError(f"Missing fold-specific graph cache: {paths['graphs']}")
    command = [
        sys.executable,
        "-m",
        "hiercp.pipeline",
        "train",
        "--run-mode",
        "benchmark",
        "--config",
        str(layout.train_config),
        "--cache-dir",
        str(paths["graphs"]),
        "--prototype-bank",
        str(paths["prototype"]),
        "--checkpoint",
        str(paths["model"]),
        "--prototype-bank",
        str(paths["prototype"]),
        "--run-mode",
        "benchmark",
        "--device",
        device,
        "--seed",
        str(int(train_cfg.get("seed", 42)) + outer_fold),
        "--epochs",
        str(int(train_cfg["training"].get("epochs", 40))),
    ]
    if overwrite:
        command.append("--overwrite")
    run_command(command, cwd=layout.project, log=paths["log"], dry_run=dry_run)
    audit = [
        sys.executable,
        "-m",
        "tools.causality",
        "--cache-dir",
        str(paths["graphs"]),
        "--checkpoint",
        str(paths["model"]),
        "--device",
        device,
        "--split",
        "val",
        "--seed",
        str(int(train_cfg.get("seed", 42)) + outer_fold),
        "--output",
        str(paths["causality"]),
        "--strict",
    ]
    if overwrite:
        audit.append("--overwrite")
    run_command(audit, cwd=layout.project, log=paths["log"], dry_run=dry_run)


def manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pair_output_paths(root: Path, case_id: str) -> tuple[Path, Path]:
    return root / "image" / f"{case_id}_0000.nii.gz", root / "labels" / f"{case_id}.nii.gz"


def candidate_pool_hash(candidates: Sequence[Any]) -> str:
    payload = [
        {
            "center": [int(value) for value in candidate.center],
            "coverage": round(float(candidate.liver_coverage), 8),
            "border_mm": round(float(candidate.border_distance_mm), 8),
            "occupied_mm": round(float(candidate.occupied_distance_mm), 8),
            "mean": round(float(candidate.context_mean_hu), 8),
            "std": round(float(candidate.context_std_hu), 8),
        }
        for candidate in candidates
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def generate_pairs(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
    device_name: str,
    overwrite: bool,
) -> None:
    # Imports are delayed so `--help`, split creation and status work even when
    # PyTorch Geometric is not initialized yet.
    import torch

    from hiercp.cache import build_inference_sample
    from hiercp.common import (
        CasePaths,
        build_candidate_pool,
        choose_source_tumor,
        load_case,
        paste_source,
        save_case_pair,
        stable_case_seed,
    )
    from hiercp.curriculum import build_generation_specs
    from hiercp.data import collate_samples
    from hiercp.local import build_local_graph, prepare_local_source
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.prototype import PrototypeBank
    from hiercp.region import REGION_CACHE_SEED_SALT, load_or_build_patient_regions
    from hiercp.schema import graph_config_from_dict
    from hiercp.spatial import AdaptiveRoiBudgetError, CanonicalGraphUnavailable
    from hiercp.tensor import configure_runtime, load_checkpoint, resolve_device

    gpaths = gnn_paths(layout, outer_fold)
    if not gpaths["model"].is_file():
        raise BenchmarkError(f"Missing fold-specific GNN checkpoint: {gpaths['model']}")
    if not gpaths["prototype"].is_file():
        raise BenchmarkError(f"Missing fold-specific prototype: {gpaths['prototype']}")

    split = outer_split(layout, outer_fold)
    case_map = {case.case_id: case for case in discover_cases(layout.data)}
    unknown = sorted(set(split["train"]) - set(case_map), key=natural_key)
    if unknown:
        raise BenchmarkError(f"Outer train references missing cases: {unknown}")

    generation = train_cfg["generation"]
    if int(generation.get("num_copies", 1)) != 1:
        raise BenchmarkError("Paired benchmark requires generation.num_copies=1")
    labels = train_cfg["labels"]
    liver_label = int(labels["liver"])
    tumor_label = int(labels["tumor"])
    global_seed = int(train_cfg.get("seed", 42))
    runtime = train_cfg.get("runtime", {})
    configure_runtime(
        deterministic=bool(runtime.get("deterministic", True)),
        allow_tf32=bool(runtime.get("allow_tf32", False)),
        cudnn_benchmark=False,
    )
    device = resolve_device(device_name)
    checkpoint = load_checkpoint(gpaths["model"], device)
    model = HierarchicalPyGPlacementModel(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    graph_config = graph_config_from_dict(checkpoint["graph_config"])
    ct_clip = tuple(float(value) for value in checkpoint["ct_clip"])
    bank = PrototypeBank.load(gpaths["prototype"])
    if bank.fingerprint() != checkpoint.get("prototype_fingerprint"):
        raise BenchmarkError("Fold-specific prototype does not match GNN checkpoint")

    use_amp = bool(generation.get("amp", True) and device.type == "cuda")
    chunk_size = max(1, int(generation.get("local_candidate_chunk_size", 8)))
    min_diameter = float(nn_cfg["small_tumor"]["augmentation_min_equivalent_diameter_mm"])
    max_diameter = float(nn_cfg["small_tumor"]["augmentation_max_equivalent_diameter_mm"])

    pair_root = layout.paired(outer_fold)
    basic_root = pair_root / "basic"
    hier_root = pair_root / "hier"
    manifest = pair_root / "manifest.csv"
    fields = (
        "case_id",
        "status",
        "reason",
        "outer_fold",
        "source_component",
        "source_voxels",
        "source_diameter_mm",
        "candidate_count",
        "candidate_pool_sha256",
        "basic_index",
        "hier_index",
        "basic_center",
        "hier_center",
        "basic_hier_score",
        "hier_score",
        "basic_rank_by_hier",
        "intensity_scale",
        "intensity_shift",
        "basic_image",
        "basic_label",
        "hier_image",
        "hier_label",
    )
    existing = {row["case_id"]: row for row in manifest_rows(manifest)}
    rows: dict[str, dict[str, Any]] = dict(existing)

    for index, case_id in enumerate(split["train"], start=1):
        basic_image, basic_label = pair_output_paths(basic_root, case_id)
        hier_image, hier_label = pair_output_paths(hier_root, case_id)
        old = existing.get(case_id)
        if (
            not overwrite
            and old is not None
            and old.get("status") in {"ok", "no_tumor", "outside_diameter", "no_candidate"}
            and (
                old.get("status") != "ok"
                or all(path.is_file() for path in (basic_image, basic_label, hier_image, hier_label))
            )
        ):
            print(f"[Reuse] paired {index}/{len(split['train'])} {case_id} status={old['status']}")
            continue

        print(f"[Pair] {index}/{len(split['train'])} {case_id}", flush=True)
        row: dict[str, Any] = {
            "case_id": case_id,
            "outer_fold": outer_fold,
            "status": "error",
            "reason": "",
        }
        try:
            source_case = case_map[case_id]
            case = load_case(
                CasePaths(
                    case_id=source_case.case_id,
                    image_path=source_case.image,
                    label_path=source_case.label,
                )
            )
            occupied = case.label == tumor_label
            if not np.any(occupied):
                row.update(status="no_tumor", reason="tumor label absent")
                rows[case_id] = row
                atomic_csv(manifest, list(rows.values()), fields)
                print(f"[Skip] {case_id}: no tumor")
                continue

            # The source and candidate pool RNG is shared by both methods.
            pool_rng = np.random.default_rng(
                stable_case_seed(global_seed, case_id, "paired_cp_source_and_pool_v1")
            )
            source, _, _ = choose_source_tumor(
                case.image,
                case.label,
                tumor_label=tumor_label,
                rng=pool_rng,
                selection=str(generation["source_selection"]),
                pad=int(generation["source_pad"]),
            )
            source_volume = float(source.voxel_count) * float(np.prod(case.spacing))
            source_diameter = equivalent_diameter(source_volume)
            row.update(
                source_component=int(source.component_id),
                source_voxels=int(source.voxel_count),
                source_diameter_mm=f"{source_diameter:.6f}",
            )
            if not (min_diameter < source_diameter <= max_diameter):
                row.update(
                    status="outside_diameter",
                    reason=f"source diameter {source_diameter:.6f} mm outside ({min_diameter}, {max_diameter}]",
                )
                rows[case_id] = row
                atomic_csv(manifest, list(rows.values()), fields)
                print(f"[Skip] {case_id}: diameter={source_diameter:.2f} mm")
                continue

            region_seed = stable_case_seed(
                global_seed + outer_fold, case_id, REGION_CACHE_SEED_SALT
            )
            regions = load_or_build_patient_regions(
                case,
                cache_dir=gpaths["regions"],
                liver_label=liver_label,
                tumor_label=tumor_label,
                config=graph_config,
                seed=region_seed,
                ct_clip=ct_clip,
                overwrite=False,
                mmap=True,
            )
            candidates, _ = build_candidate_pool(
                case,
                source,
                placement_mask=case.label == liver_label,
                full_organ_mask=regions.full_organ_mask,
                occupied_mask=occupied,
                organ_distance=regions.organ_depth,
                rng=pool_rng,
                num_candidates=int(generation["num_candidates"]),
                max_draws=int(generation["max_draws"]),
                min_liver_coverage=float(generation["min_liver_coverage"]),
                occupied_clearance_vox=int(generation["occupied_clearance_vox"]),
                min_center_separation_mm=float(generation["min_center_separation_mm"]),
            )
            if not candidates:
                row.update(status="no_candidate", reason="shared valid candidate pool is empty")
                rows[case_id] = row
                atomic_csv(manifest, list(rows.values()), fields)
                print(f"[Skip] {case_id}: no candidate")
                continue

            inference_seed = stable_case_seed(
                global_seed, case_id, "paired_cp_inference_v1"
            )
            filtered_local_geometry = 0
            try:
                sample, _ = build_inference_sample(
                    case,
                    source,
                    candidates,
                    bank,
                    graph_config=graph_config,
                    liver_label=liver_label,
                    tumor_label=tumor_label,
                    ct_clip=ct_clip,
                    seed=inference_seed,
                    regions=regions,
                )
            except AdaptiveRoiBudgetError:
                raise
            except CanonicalGraphUnavailable as initial_exc:
                # Anatomical validity does not guarantee that every center can
                # form the complete V22 local semantic graph. Remove only those
                # unrepresentable centers, then give Basic-CP and HierCP the same
                # filtered pool. This preserves the paired comparison.
                specs = build_generation_specs(
                    candidates,
                    regions,
                    bank,
                    config=graph_config,
                )
                local_rng = np.random.default_rng(
                    stable_case_seed(
                        inference_seed, case.paths.case_id, "infer_local"
                    )
                )
                try:
                    prepared_source = prepare_local_source(
                        case,
                        source,
                        full_organ_mask=regions.full_organ_mask,
                        organ_depth=regions.organ_depth,
                        config=graph_config,
                        rng=local_rng,
                        ct_clip=ct_clip,
                    )
                except AdaptiveRoiBudgetError:
                    raise
                except CanonicalGraphUnavailable as source_exc:
                    row.update(
                        status="no_candidate",
                        reason=(
                            "shared source local graph is unrepresentable: "
                            f"{source_exc}"
                        ),
                        candidate_count=0,
                    )
                    rows[case_id] = row
                    atomic_csv(manifest, list(rows.values()), fields)
                    print(f"[Skip] {case_id}: source local geometry unavailable")
                    continue

                viable_candidates: list[Any] = []
                rejected_examples: list[str] = []
                for candidate, spec in zip(candidates, specs):
                    try:
                        build_local_graph(
                            case,
                            source,
                            spec,
                            full_organ_mask=regions.full_organ_mask,
                            organ_depth=regions.organ_depth,
                            config=graph_config,
                            rng=local_rng,
                            ct_clip=ct_clip,
                            prepared_source=prepared_source,
                        )
                    except AdaptiveRoiBudgetError:
                        raise
                    except CanonicalGraphUnavailable as candidate_exc:
                        filtered_local_geometry += 1
                        if len(rejected_examples) < 3:
                            center = ",".join(
                                str(int(value)) for value in candidate.center
                            )
                            rejected_examples.append(f"{center}: {candidate_exc}")
                    else:
                        viable_candidates.append(candidate)

                if len(viable_candidates) < 2:
                    reason = (
                        "fewer than two jointly representable candidates remain; "
                        f"initial_error={initial_exc}; "
                        f"rejected={filtered_local_geometry}; "
                        f"examples={' | '.join(rejected_examples)}"
                    )
                    row.update(
                        status="no_candidate",
                        reason=reason,
                        candidate_count=len(viable_candidates),
                    )
                    rows[case_id] = row
                    atomic_csv(manifest, list(rows.values()), fields)
                    print(
                        f"[Skip] {case_id}: jointly representable candidates="
                        f"{len(viable_candidates)}"
                    )
                    continue

                candidates = viable_candidates
                sample, _ = build_inference_sample(
                    case,
                    source,
                    candidates,
                    bank,
                    graph_config=graph_config,
                    liver_label=liver_label,
                    tumor_label=tumor_label,
                    ct_clip=ct_clip,
                    seed=inference_seed,
                    regions=regions,
                )
                print(
                    f"[Filter] {case_id}: removed "
                    f"{filtered_local_geometry} unrepresentable candidates; "
                    f"shared_pool={len(candidates)}"
                )
            batch = collate_samples([sample])
            if device.type == "cuda" and bool(generation.get("pin_memory", True)):
                batch.pin_memory()
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                scores_tensor = model.score_inference_chunked(
                    batch, local_chunk_size=chunk_size
                )[0]
            scores = scores_tensor.float().cpu().numpy()
            if scores.ndim != 1 or scores.size != len(candidates):
                raise BenchmarkError(
                    f"Score/candidate mismatch for {case_id}: {scores.shape} vs {len(candidates)}"
                )

            hier_index = int(np.argmax(scores))
            selection_rng = np.random.default_rng(
                stable_case_seed(global_seed, case_id, "paired_cp_basic_uniform_v1")
            )
            basic_index = int(selection_rng.integers(0, len(candidates)))
            ranked = np.argsort(scores)[::-1]
            basic_rank = int(np.where(ranked == basic_index)[0][0]) + 1

            basic_out_image = case.image.copy()
            basic_out_label = case.label.copy()
            basic_occupied = occupied.copy()
            hier_out_image = case.image.copy()
            hier_out_label = case.label.copy()
            hier_occupied = occupied.copy()

            appearance_seed = stable_case_seed(
                global_seed, case_id, "paired_cp_identical_appearance_v1"
            )
            basic_appearance = np.random.default_rng(appearance_seed)
            hier_appearance = np.random.default_rng(appearance_seed)
            basic_scale, basic_shift = paste_source(
                basic_out_image,
                basic_out_label,
                basic_occupied,
                source,
                candidates[basic_index],
                tumor_label=tumor_label,
                rng=basic_appearance,
                intensity_scale_range=tuple(
                    float(value) for value in generation["intensity_scale_range"]
                ),
                intensity_shift_range=tuple(
                    float(value) for value in generation["intensity_shift_range"]
                ),
                blend_border=int(generation["blend_border"]),
            )
            hier_scale, hier_shift = paste_source(
                hier_out_image,
                hier_out_label,
                hier_occupied,
                source,
                candidates[hier_index],
                tumor_label=tumor_label,
                rng=hier_appearance,
                intensity_scale_range=tuple(
                    float(value) for value in generation["intensity_scale_range"]
                ),
                intensity_shift_range=tuple(
                    float(value) for value in generation["intensity_shift_range"]
                ),
                blend_border=int(generation["blend_border"]),
            )
            if basic_scale != hier_scale or basic_shift != hier_shift:
                raise BenchmarkError("Appearance jitter diverged between paired methods")

            save_case_pair(
                case,
                basic_out_image,
                basic_out_label,
                basic_root,
                overwrite=overwrite,
            )
            save_case_pair(
                case,
                hier_out_image,
                hier_out_label,
                hier_root,
                overwrite=overwrite,
            )
            row.update(
                status="ok",
                reason=(
                    "paired"
                    if filtered_local_geometry == 0
                    else f"paired; filtered_local_geometry={filtered_local_geometry}"
                ),
                candidate_count=len(candidates),
                candidate_pool_sha256=candidate_pool_hash(candidates),
                basic_index=basic_index,
                hier_index=hier_index,
                basic_center=",".join(str(int(value)) for value in candidates[basic_index].center),
                hier_center=",".join(str(int(value)) for value in candidates[hier_index].center),
                basic_hier_score=f"{float(scores[basic_index]):.8f}",
                hier_score=f"{float(scores[hier_index]):.8f}",
                basic_rank_by_hier=basic_rank,
                intensity_scale=f"{basic_scale:.8f}",
                intensity_shift=f"{basic_shift:.8f}",
                basic_image=str(basic_image.resolve()),
                basic_label=str(basic_label.resolve()),
                hier_image=str(hier_image.resolve()),
                hier_label=str(hier_label.resolve()),
            )
            print(
                f"[OK] {case_id} basic={row['basic_center']} "
                f"hier={row['hier_center']} candidates={len(candidates)}"
            )
        except AdaptiveRoiBudgetError:
            raise
        except Exception as exc:
            row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
            print(f"[Error] {case_id}: {exc}")
        rows[case_id] = row
        ordered = sorted(rows.values(), key=lambda item: natural_key(str(item["case_id"])))
        atomic_csv(manifest, ordered, fields)

    final_rows = manifest_rows(manifest)
    errors = [row for row in final_rows if row.get("status") == "error"]
    ok = [row for row in final_rows if row.get("status") == "ok"]
    if errors:
        raise BenchmarkError(
            f"Paired generation has {len(errors)} error rows; inspect {manifest}"
        )
    if not ok:
        raise BenchmarkError("Paired generation produced no eligible small-tumor cases")
    print(
        f"[OK] paired generation fold={outer_fold}: eligible={len(ok)} "
        f"manifest={manifest}"
    )


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
        raise BenchmarkError(f"Unsupported materialization mode: {mode}")


def dataset_specs(base_id: int, outer_fold: int) -> dict[str, DatasetSpec]:
    first = int(base_id) + int(outer_fold) * 3
    if first < 1 or first + 2 > 999:
        raise BenchmarkError("Dataset IDs must stay within 1..999")
    return {
        "plan": DatasetSpec(first, f"Dataset{first:03d}_LiverOriginalPlan_OF{outer_fold}", "plan"),
        "basic": DatasetSpec(first + 1, f"Dataset{first + 1:03d}_LiverBasicCP_OF{outer_fold}", "basic"),
        "hier": DatasetSpec(first + 2, f"Dataset{first + 2:03d}_LiverHierCP_OF{outer_fold}", "hier"),
    }


def raw_dataset_dir(layout: Layout, spec: DatasetSpec) -> Path:
    return layout.raw / spec.dataset_name


def preprocessed_dataset_dir(layout: Layout, spec: DatasetSpec) -> Path:
    return layout.preprocessed / spec.dataset_name


def build_one_raw_dataset(
    layout: Layout,
    spec: DatasetSpec,
    originals: Sequence[Case],
    synthetic_rows: Sequence[Mapping[str, str]],
    outer_fold: int,
    tumor_label: int,
    materialization: str,
    overwrite: bool,
) -> tuple[list[str], list[str]]:
    target = raw_dataset_dir(layout, spec)
    marker = target / "paired_benchmark.json"
    split = outer_split(layout, outer_fold)
    original_ids = [case.case_id for case in originals]
    train_originals = list(split["train"])
    val_originals = list(split["val"])
    if spec.condition == "plan":
        selected_rows: list[Mapping[str, str]] = []
    else:
        selected_rows = [row for row in synthetic_rows if row.get("status") == "ok"]
    synthetic_ids = [
        f"{row['case_id']}__{spec.condition}_of{outer_fold}"
        for row in selected_rows
    ]
    train_ids = train_originals + synthetic_ids
    expected = {
        "version": VERSION,
        "condition": spec.condition,
        "outer_fold": outer_fold,
        "original_cases": len(originals),
        "synthetic_cases": len(selected_rows),
        "train_originals": len(train_originals),
        "validation_originals": len(val_originals),
        "train_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "val_ids_sha256": hashlib.sha256("\n".join(val_originals).encode()).hexdigest(),
    }
    if target.is_dir() and not overwrite:
        if not marker.is_file() or any(load_json(marker).get(key) != value for key, value in expected.items()):
            raise BenchmarkError(f"Existing raw dataset is incompatible: {target}")
        print(f"[Reuse] raw dataset {spec.dataset_name}")
        return train_ids, val_originals

    if target.exists():
        shutil.rmtree(target)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "imagesTr").mkdir(parents=True)
    (temporary / "labelsTr").mkdir(parents=True)
    (temporary / "imagesTs").mkdir(parents=True)
    for case in originals:
        materialize(case.image, temporary / "imagesTr" / f"{case.case_id}_0000.nii.gz", materialization)
        materialize(case.label, temporary / "labelsTr" / f"{case.case_id}.nii.gz", materialization)
    for row, synthetic_id in zip(selected_rows, synthetic_ids):
        image = Path(row[f"{spec.condition}_image"])
        label = Path(row[f"{spec.condition}_label"])
        if not image.is_file() or not label.is_file():
            raise BenchmarkError(f"Missing paired synthetic files for {row['case_id']}")
        materialize(image, temporary / "imagesTr" / f"{synthetic_id}_0000.nii.gz", materialization)
        materialize(label, temporary / "labelsTr" / f"{synthetic_id}.nii.gz", materialization)
    atomic_json(
        temporary / "dataset.json",
        {
            "channel_names": {"0": "CT"},
            "labels": {"background": 0, "liver": 1, "tumor": int(tumor_label)},
            "numTraining": len(originals) + len(selected_rows),
            "file_ending": ".nii.gz",
        },
    )
    atomic_json(temporary / "paired_benchmark.json", expected)
    temporary.replace(target)
    print(
        f"[OK] raw dataset {spec.dataset_name}: originals={len(originals)} "
        f"synthetic={len(selected_rows)}"
    )
    return train_ids, val_originals


def build_nnunet_datasets(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
    base_id: int,
    materialization: str,
    overwrite: bool,
) -> None:
    split = outer_split(layout, outer_fold)
    manifest = layout.paired(outer_fold) / "manifest.csv"
    rows = manifest_rows(manifest)
    ok = [row for row in rows if row.get("status") == "ok"]
    if not ok:
        raise BenchmarkError(f"No paired synthetic rows: {manifest}")
    if any(row["case_id"] not in set(split["train"]) for row in ok):
        raise BenchmarkError("Paired manifest contains outer-validation source cases")
    originals = discover_cases(layout.data)
    specs = dataset_specs(base_id, outer_fold)
    tumor_label = int(train_cfg["labels"]["tumor"])
    plan_train, plan_val = build_one_raw_dataset(
        layout,
        specs["plan"],
        originals,
        [],
        outer_fold,
        tumor_label,
        materialization,
        overwrite,
    )
    basic_train, basic_val = build_one_raw_dataset(
        layout,
        specs["basic"],
        originals,
        ok,
        outer_fold,
        tumor_label,
        materialization,
        overwrite,
    )
    hier_train, hier_val = build_one_raw_dataset(
        layout,
        specs["hier"],
        originals,
        ok,
        outer_fold,
        tumor_label,
        materialization,
        overwrite,
    )
    if basic_val != hier_val or basic_val != plan_val:
        raise BenchmarkError("Basic and Hier validation IDs diverged")
    if len(basic_train) != len(hier_train):
        raise BenchmarkError("Basic and Hier training counts diverged")
    # Synthetic IDs differ only by method; source case sets must be identical.
    basic_sources = {value.split("__basic_of", 1)[0] for value in basic_train if "__basic_of" in value}
    hier_sources = {value.split("__hier_of", 1)[0] for value in hier_train if "__hier_of" in value}
    if basic_sources != hier_sources:
        raise BenchmarkError("Basic and Hier synthetic source sets diverged")
    metadata = {
        "version": VERSION,
        "outer_fold": outer_fold,
        "datasets": {key: spec.__dict__ for key, spec in specs.items()},
        "original_train": split["train"],
        "original_val": split["val"],
        "basic_train": basic_train,
        "hier_train": hier_train,
        "paired_synthetic_sources": sorted(basic_sources, key=natural_key),
        "trainer": nn_cfg["dataset"]["trainer"],
        "configuration": nn_cfg["dataset"]["configuration"],
        "plans": nn_cfg["dataset"]["plans"],
    }
    atomic_json(layout.fold(outer_fold) / "nnunet_pair.json", metadata)
    print(
        f"[OK] paired nnU-Net datasets fold={outer_fold}: "
        f"train_original={len(split['train'])} paired_synthetic={len(basic_sources)} "
        f"val_original={len(split['val'])}"
    )


def nn_env(layout: Layout, nn_cfg: Mapping[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env["nnUNet_raw"] = str(layout.raw)
    env["nnUNet_preprocessed"] = str(layout.preprocessed)
    env["nnUNet_results"] = str(layout.results)
    env["nnUNet_n_proc_DA"] = str(int(nn_cfg["training"].get("nnunet_n_proc_DA", 8)))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    return env


def recursively_replace(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [recursively_replace(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: recursively_replace(item, old, new) for key, item in value.items()}
    return value


def install_single_split(
    layout: Layout,
    spec: DatasetSpec,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
) -> None:
    destination = preprocessed_dataset_dir(layout, spec) / "splits_final.json"
    atomic_json(destination, [{"train": list(train_ids), "val": list(val_ids)}])
    print(f"[OK] split installed {spec.condition}: local_fold=0 path={destination}")


def extract_fingerprint(
    layout: Layout,
    spec: DatasetSpec,
    nn_cfg: Mapping[str, Any],
    dry_run: bool,
) -> None:
    command = [
        require_command("nnUNetv2_extract_fingerprint"),
        "-d",
        str(spec.dataset_id),
    ]
    if bool(nn_cfg["preprocess"].get("verify_dataset_integrity", True)):
        command.append("--verify_dataset_integrity")
    run_command(
        command,
        cwd=layout.project,
        env=nn_env(layout, nn_cfg),
        log=layout.logs / f"fingerprint_{spec.condition}_{spec.dataset_id}.log",
        dry_run=dry_run,
    )


def plan_and_preprocess(
    layout: Layout,
    outer_fold: int,
    nn_cfg: Mapping[str, Any],
    base_id: int,
    dry_run: bool,
) -> None:
    pair_meta = load_json(layout.fold(outer_fold) / "nnunet_pair.json")
    specs = dataset_specs(base_id, outer_fold)
    plans_name = str(nn_cfg["dataset"]["plans"])
    planner = str(nn_cfg["dataset"]["planner"])
    configuration = str(nn_cfg["dataset"]["configuration"])
    processes = int(nn_cfg["preprocess"].get("processes", 4))

    # Plan exactly once from original data only. The resulting CT normalization,
    # spacing, patch size, batch size and architecture are copied to both arms.
    plan_spec = specs["plan"]
    plan_dir = preprocessed_dataset_dir(layout, plan_spec)
    plan_file = plan_dir / f"{plans_name}.json"
    if not plan_file.is_file():
        extract_fingerprint(layout, plan_spec, nn_cfg, dry_run)
        command = [
            require_command("nnUNetv2_plan_experiment"),
            "-d",
            str(plan_spec.dataset_id),
            "-pl",
            planner,
        ]
        run_command(
            command,
            cwd=layout.project,
            env=nn_env(layout, nn_cfg),
            log=layout.logs / f"plan_reference_{outer_fold}.log",
            dry_run=dry_run,
        )
    if dry_run:
        print("[Dry-run] shared plans would be copied to Basic and Hier datasets")
        return
    if not plan_file.is_file():
        raise BenchmarkError(f"Reference plans missing after planning: {plan_file}")
    reference_plans = load_json(plan_file)

    for method in METHODS:
        spec = specs[method]
        target_preprocessed = preprocessed_dataset_dir(layout, spec)
        target_plan_file = target_preprocessed / f"{plans_name}.json"
        config_dir_name = str(
            reference_plans["configurations"][configuration]["data_identifier"]
        )
        ready = target_plan_file.is_file() and (target_preprocessed / config_dir_name).is_dir()
        train_ids = pair_meta[f"{method}_train"]
        val_ids = pair_meta["original_val"]
        if ready:
            install_single_split(layout, spec, train_ids, val_ids)
            print(f"[Reuse] preprocessing ready: {spec.dataset_name}")
            continue

        extract_fingerprint(layout, spec, nn_cfg, False)
        target_preprocessed.mkdir(parents=True, exist_ok=True)
        target_plans = recursively_replace(reference_plans, plan_spec.dataset_name, spec.dataset_name)
        # PlansManager expects the plans identifier to remain stable.
        target_plans["plans_name"] = plans_name
        atomic_json(target_plan_file, target_plans)
        # Some nnU-Net versions create dataset.json during fingerprinting; make
        # the contract explicit for versions that do not.
        shutil.copy2(raw_dataset_dir(layout, spec) / "dataset.json", target_preprocessed / "dataset.json")
        command = [
            require_command("nnUNetv2_preprocess"),
            "-d",
            str(spec.dataset_id),
            "-plans_name",
            plans_name,
            "-c",
            configuration,
            "-np",
            str(processes),
        ]
        if bool(nn_cfg["preprocess"].get("no_progress_bar", True)):
            command.append("--no_pbar")
        run_command(
            command,
            cwd=layout.project,
            env=nn_env(layout, nn_cfg),
            log=layout.logs / f"preprocess_{method}_{outer_fold}.log",
            dry_run=False,
        )
        if not (target_preprocessed / config_dir_name).is_dir():
            raise BenchmarkError(f"Preprocessing output missing: {target_preprocessed / config_dir_name}")
        install_single_split(layout, spec, train_ids, val_ids)

    basic_plan = load_json(preprocessed_dataset_dir(layout, specs["basic"]) / f"{plans_name}.json")
    hier_plan = load_json(preprocessed_dataset_dir(layout, specs["hier"]) / f"{plans_name}.json")
    normalized_basic = recursively_replace(basic_plan, specs["basic"].dataset_name, "DATASET")
    normalized_hier = recursively_replace(hier_plan, specs["hier"].dataset_name, "DATASET")
    if normalized_basic != normalized_hier:
        raise BenchmarkError("Basic and Hier plans are not byte-equivalent after dataset-name normalization")
    print(f"[OK] shared nnU-Net plans and preprocessing installed for outer fold {outer_fold}")


def result_model_dir(
    layout: Layout,
    spec: DatasetSpec,
    nn_cfg: Mapping[str, Any],
    trainer_override: str | None,
) -> Path:
    trainer = trainer_override or str(nn_cfg["dataset"]["trainer"])
    plans = str(nn_cfg["dataset"]["plans"])
    configuration = str(nn_cfg["dataset"]["configuration"])
    return layout.results / spec.dataset_name / f"{trainer}__{plans}__{configuration}"


def validation_complete(fold_dir: Path, val_ids: Sequence[str]) -> bool:
    validation = fold_dir / "validation"
    return (validation / "summary.json").is_file() and all(
        (validation / f"{case_id}.nii.gz").is_file() for case_id in val_ids
    )


def train_nnunet_pair(
    layout: Layout,
    outer_fold: int,
    nn_cfg: Mapping[str, Any],
    base_id: int,
    device: str,
    trainer_override: str | None,
    dry_run: bool,
) -> None:
    specs = dataset_specs(base_id, outer_fold)
    pair_meta = load_json(layout.fold(outer_fold) / "nnunet_pair.json")
    trainer = trainer_override or str(nn_cfg["dataset"]["trainer"])
    configuration = str(nn_cfg["dataset"]["configuration"])
    plans = str(nn_cfg["dataset"]["plans"])
    normalized_device = "cuda" if str(device).startswith("cuda") else str(device)
    for method in METHODS:
        spec = specs[method]
        model_dir = result_model_dir(layout, spec, nn_cfg, trainer_override)
        fold_dir = model_dir / "fold_0"
        final = fold_dir / "checkpoint_final.pth"
        val_ids = list(pair_meta["original_val"])
        if final.is_file() and validation_complete(fold_dir, val_ids):
            print(f"[Reuse] {method} outer_fold={outer_fold} complete")
            continue
        command = [
            require_command("nnUNetv2_train"),
            str(spec.dataset_id),
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
        elif (fold_dir / "checkpoint_latest.pth").is_file() or (fold_dir / "checkpoint_best.pth").is_file():
            command.append("--c")
        run_command(
            command,
            cwd=layout.project,
            env=nn_env(layout, nn_cfg),
            log=layout.logs / f"train_{method}_outer{outer_fold}.log",
            dry_run=dry_run,
        )
        if dry_run:
            continue
        if not final.is_file():
            raise BenchmarkError(f"Final checkpoint missing: {final}")
        if not validation_complete(fold_dir, val_ids):
            validation_command = [
                require_command("nnUNetv2_train"),
                str(spec.dataset_id),
                configuration,
                "0",
                "-tr",
                trainer,
                "-p",
                plans,
                "--val",
                "-device",
                normalized_device,
            ]
            run_command(
                validation_command,
                cwd=layout.project,
                env=nn_env(layout, nn_cfg),
                log=layout.logs / f"train_{method}_outer{outer_fold}.log",
                dry_run=False,
            )
        if not validation_complete(fold_dir, val_ids):
            raise BenchmarkError(f"Validation incomplete: {fold_dir}")


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

    rows: list[dict[str, Any]] = []
    for gt_index in range(num_gt):
        gt_voxels = int(gt_sizes[gt_index + 1])
        diameter = equivalent_diameter(gt_voxels * voxel_volume)
        pred_index = matches.get(gt_index)
        detected = pred_index is not None
        pred_voxels = int(pred_sizes[pred_index + 1]) if detected else 0
        intersection = int(intersections[gt_index, pred_index]) if detected else 0
        rows.append(
            {
                "diameter_mm": diameter,
                "size_bin": bin_name(diameter),
                "detected": int(detected),
                "lesion_dice": float(2 * intersection / (gt_voxels + pred_voxels))
                if detected
                else 0.0,
            }
        )
    matched_prediction = len(set(matches.values()))
    return rows, {
        "gt": int(num_gt),
        "pred": int(num_pred),
        "matched_pred": int(matched_prediction),
        "fp": int(num_pred - matched_prediction),
    }


def bootstrap_mean_difference(
    differences: Sequence[float],
    seed: int,
    iterations: int = 10000,
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
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(
            min(1.0, 2.0 * binom.cdf(min(basic_only, hier_only), discordant, 0.5))
        )
    return {
        "basic_only": basic_only,
        "hier_only": hier_only,
        "discordant": discordant,
        "exact_p": p_value,
    }


def evaluate_pair(
    layout: Layout,
    outer_fold: int,
    train_cfg: Mapping[str, Any],
    nn_cfg: Mapping[str, Any],
    base_id: int,
    trainer_override: str | None,
) -> None:
    specs = dataset_specs(base_id, outer_fold)
    pair_meta = load_json(layout.fold(outer_fold) / "nnunet_pair.json")
    val_ids = list(pair_meta["original_val"])
    basic_model = result_model_dir(layout, specs["basic"], nn_cfg, trainer_override) / "fold_0" / "validation"
    hier_model = result_model_dir(layout, specs["hier"], nn_cfg, trainer_override) / "fold_0" / "validation"
    tumor_label = int(train_cfg["labels"]["tumor"])
    bins = nn_cfg["small_tumor"].get("evaluation_bins_mm", [10.0, 20.0])
    original_map = {case.case_id: case for case in discover_cases(layout.data)}
    case_rows: list[dict[str, Any]] = []
    lesion_rows: list[dict[str, Any]] = []
    for case_id in val_ids:
        if case_id not in original_map:
            raise BenchmarkError(f"Validation case missing: {case_id}")
        reference_nii, reference = load_3d(original_map[case_id].label, np.int16)
        basic_path = basic_model / f"{case_id}.nii.gz"
        hier_path = hier_model / f"{case_id}.nii.gz"
        if not basic_path.is_file() or not hier_path.is_file():
            raise BenchmarkError(f"Missing paired validation prediction for {case_id}")
        _, basic = load_3d(basic_path, np.int16)
        _, hier = load_3d(hier_path, np.int16)
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
            raise BenchmarkError(f"Ground-truth lesion alignment failed for {case_id}")
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
        for component, (basic_row, hier_row) in enumerate(
            zip(basic_lesions, hier_lesions), start=1
        ):
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
                    "lesion_dice_difference": hier_row["lesion_dice"]
                    - basic_row["lesion_dice"],
                }
            )

    if not case_rows:
        raise BenchmarkError("No validation cases evaluated")
    output = layout.evaluation(outer_fold)
    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(output / "case_metrics.csv", case_rows, tuple(case_rows[0]))
    atomic_csv(
        output / "lesion_metrics.csv",
        lesion_rows,
        tuple(lesion_rows[0]) if lesion_rows else ("case_id",),
    )

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
            "mean_case_tumor_dice": float(
                np.mean([float(row[f"{method}_dice"]) for row in case_rows])
            ),
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
    lesion_dice_differences = [
        float(row["lesion_dice_difference"]) for row in lesion_rows
    ]
    mcnemar = exact_mcnemar(
        [int(row["basic_detected"]) for row in lesion_rows],
        [int(row["hier_detected"]) for row in lesion_rows],
    )
    statistics = {
        "case_tumor_dice": {
            **bootstrap_mean_difference(dice_differences, 11000 + outer_fold),
            "wilcoxon_p": paired_wilcoxon(dice_differences),
        },
        "false_positive_per_case": {
            **bootstrap_mean_difference(fp_differences, 12000 + outer_fold),
            "wilcoxon_p": paired_wilcoxon(fp_differences),
        },
        "lesion_dice": {
            **bootstrap_mean_difference(
                lesion_dice_differences, 13000 + outer_fold
            ),
            "wilcoxon_p": paired_wilcoxon(lesion_dice_differences),
        },
        "lesion_detection_mcnemar": mcnemar,
    }
    summary = {
        "version": VERSION,
        "outer_fold": outer_fold,
        "validation_cases": len(case_rows),
        "basic_cp": basic_summary,
        "hiercp": hier_summary,
        "difference_hier_minus_basic": {
            "mean_case_tumor_dice": hier_summary["mean_case_tumor_dice"]
            - basic_summary["mean_case_tumor_dice"],
            "lesion_recall": hier_summary["lesion_recall"]
            - basic_summary["lesion_recall"],
            "false_positive_per_case": hier_summary["false_positive_per_case"]
            - basic_summary["false_positive_per_case"],
        },
        "paired_statistics": statistics,
    }
    atomic_json(output / "summary.json", summary)

    lines = [
        f"# Paired Basic-CP vs HierCP — Outer Fold {outer_fold}",
        "",
        f"- Validation patients: {len(case_rows)}",
        "- Validation data: original CT only",
        "- Basic-CP and HierCP use the same source tumors, candidate pools and intensity jitter.",
        "- Only candidate selection differs.",
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
    for size_bin in sorted(
        set(basic_summary["by_size"]) | set(hier_summary["by_size"])
    ):
        basic_value = basic_summary["by_size"].get(size_bin, {})
        hier_value = hier_summary["by_size"].get(size_bin, {})
        gt = basic_value.get("gt", hier_value.get("gt", 0))
        lines.append(
            f"| {size_bin} | {gt} | {basic_value.get('recall', float('nan')):.4f} | {hier_value.get('recall', float('nan')):.4f} |"
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
    print(f"[OK] paired evaluation: {output / 'summary.json'}")


def aggregate(layout: Layout) -> None:
    summaries = []
    case_rows: list[dict[str, str]] = []
    lesion_rows: list[dict[str, str]] = []
    for fold in range(5):
        summary_path = layout.evaluation(fold) / "summary.json"
        if not summary_path.is_file():
            continue
        summaries.append(load_json(summary_path))
        for filename, output_rows in (
            ("case_metrics.csv", case_rows),
            ("lesion_metrics.csv", lesion_rows),
        ):
            with (layout.evaluation(fold) / filename).open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                output_rows.extend(csv.DictReader(handle))
    if not summaries:
        raise BenchmarkError("No completed fold evaluations to aggregate")
    output = layout.benchmark / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    if case_rows:
        atomic_csv(output / "case_metrics.csv", case_rows, tuple(case_rows[0]))
    if lesion_rows:
        atomic_csv(output / "lesion_metrics.csv", lesion_rows, tuple(lesion_rows[0]))
    dice_differences = [float(row["dice_difference"]) for row in case_rows]
    fp_differences = [float(row["fp_difference"]) for row in case_rows]
    basic_detected = [int(row["basic_detected"]) for row in lesion_rows]
    hier_detected = [int(row["hier_detected"]) for row in lesion_rows]
    basic_recall = sum(basic_detected) / len(basic_detected) if basic_detected else None
    hier_recall = sum(hier_detected) / len(hier_detected) if hier_detected else None
    summary = {
        "version": VERSION,
        "completed_outer_folds": sorted(int(item["outer_fold"]) for item in summaries),
        "validation_cases": len(case_rows),
        "gt_lesions": len(lesion_rows),
        "basic_cp": {
            "mean_case_tumor_dice": float(
                np.mean([float(row["basic_dice"]) for row in case_rows])
            ),
            "lesion_recall": basic_recall,
            "false_positive_per_case": float(
                np.mean([float(row["basic_fp"]) for row in case_rows])
            ),
        },
        "hiercp": {
            "mean_case_tumor_dice": float(
                np.mean([float(row["hier_dice"]) for row in case_rows])
            ),
            "lesion_recall": hier_recall,
            "false_positive_per_case": float(
                np.mean([float(row["hier_fp"]) for row in case_rows])
            ),
        },
        "paired_statistics": {
            "case_tumor_dice": {
                **bootstrap_mean_difference(dice_differences, 21000),
                "wilcoxon_p": paired_wilcoxon(dice_differences),
            },
            "false_positive_per_case": {
                **bootstrap_mean_difference(fp_differences, 22000),
                "wilcoxon_p": paired_wilcoxon(fp_differences),
            },
            "lesion_detection_mcnemar": exact_mcnemar(
                basic_detected, hier_detected
            ),
        },
    }
    summary["difference_hier_minus_basic"] = {
        key: summary["hiercp"][key] - summary["basic_cp"][key]
        for key in summary["basic_cp"]
    }
    atomic_json(output / "summary.json", summary)
    print(
        f"[OK] aggregate evaluation folds={summary['completed_outer_folds']} "
        f"path={output / 'summary.json'}"
    )


def status(
    layout: Layout,
    outer_fold: int,
    nn_cfg: Mapping[str, Any],
    base_id: int,
    trainer_override: str | None,
) -> None:
    print("Paired Basic-CP vs HierCP benchmark")
    print(f"  version:          {VERSION}")
    print(f"  benchmark:        {layout.benchmark}")
    print(f"  outer splits:     {'ready' if layout.outer_splits.is_file() else 'missing'}")
    if not layout.outer_splits.is_file():
        return
    split = outer_split(layout, outer_fold)
    print(f"  outer fold:       {outer_fold}")
    print(f"  original split:   train={len(split['train'])} val={len(split['val'])}")
    gpaths = gnn_paths(layout, outer_fold)
    print(f"  inner split:      {'ready' if gpaths['split'].is_file() else 'missing'}")
    print(f"  GNN prototype:    {'ready' if gpaths['prototype'].is_file() else 'missing'}")
    print(f"  GNN graph cache:  {len(list(gpaths['graphs'].glob('*.pt'))) if gpaths['graphs'].is_dir() else 0}")
    print(f"  GNN checkpoint:   {'ready' if gpaths['model'].is_file() else 'missing'}")
    print(f"  GNN causality:    {'ready' if gpaths['causality'].is_file() else 'missing'}")
    manifest = layout.paired(outer_fold) / "manifest.csv"
    rows = manifest_rows(manifest)
    print(f"  paired generated: {sum(row.get('status') == 'ok' for row in rows)}")
    specs = dataset_specs(base_id, outer_fold)
    for method in ("plan", "basic", "hier"):
        spec = specs[method]
        raw_ready = (raw_dataset_dir(layout, spec) / "paired_benchmark.json").is_file()
        print(f"  raw {method:5s}:       {'ready' if raw_ready else 'missing'} ({spec.dataset_name})")
    for method in METHODS:
        spec = specs[method]
        plan_file = preprocessed_dataset_dir(layout, spec) / f"{nn_cfg['dataset']['plans']}.json"
        model = result_model_dir(layout, spec, nn_cfg, trainer_override) / "fold_0"
        val_ids = split["val"]
        print(f"  preprocessed {method}: {'ready' if plan_file.is_file() else 'missing'}")
        print(
            f"  trained {method:5s}:    "
            f"{'ready' if (model / 'checkpoint_final.pth').is_file() and validation_complete(model, val_ids) else 'missing'}"
        )
    print(
        f"  evaluation:      "
        f"{'ready' if (layout.evaluation(outer_fold) / 'summary.json').is_file() else 'missing'}"
    )


def locate_project(requested: str | None) -> Path:
    candidates: list[Path] = []
    if requested:
        candidates.append(Path(requested).expanduser())
    here = Path(__file__).resolve()
    candidates.extend([here.parents[1], Path.cwd()])
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "hiercp" / "pipeline.py").is_file() and (root / "tools").is_dir():
            return root
    raise BenchmarkError("Cannot locate HierCP project; use --project-root")


def locate_medical(project: Path, requested: str | None) -> Path:
    candidates = []
    if requested:
        candidates.append(Path(requested).expanduser())
    candidates.extend([project.parent, Path.cwd()])
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "Data" / "image").is_dir() and (root / "Data" / "labels").is_dir():
            return root
    raise BenchmarkError("Cannot locate Medical/Data; use --medical-root")


def make_layout(args: argparse.Namespace) -> Layout:
    project = locate_project(args.project_root)
    medical = locate_medical(project, args.medical_root)
    benchmark = (
        Path(args.work).expanduser().resolve()
        if args.work
        else (project / "work" / "paired_basic_vs_hiercp").resolve()
    )
    source_work = (
        Path(args.source_work).expanduser().resolve()
        if args.source_work
        else (project / "work" / "full").resolve()
    )
    train_config = (
        Path(args.train_config).expanduser().resolve()
        if args.train_config
        else (project / "config" / "train.json").resolve()
    )
    nnunet_config = (
        Path(args.nnunet_config).expanduser().resolve()
        if args.nnunet_config
        else (project / "config" / "nnunet.json").resolve()
    )
    nnroot = benchmark / "nnunetv2"
    layout = Layout(
        project=project,
        medical=medical,
        data=(medical / "Data").resolve(),
        benchmark=benchmark,
        source_work=source_work,
        train_config=train_config,
        nnunet_config=nnunet_config,
        outer_splits=benchmark / "outer_splits.json",
        profiles_csv=benchmark / "case_profiles.csv",
        nnroot=nnroot,
        raw=nnroot / "nnUNet_raw",
        preprocessed=nnroot / "nnUNet_preprocessed",
        results=nnroot / "nnUNet_results",
        logs=benchmark / "logs",
    )
    for directory in (
        layout.benchmark,
        layout.raw,
        layout.preprocessed,
        layout.results,
        layout.logs,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "target",
        choices=(
            "check",
            "split",
            "gnn-prepare",
            "gnn-train",
            "generate",
            "dataset",
            "plan",
            "train",
            "evaluate",
            "aggregate",
            "status",
            "all",
        ),
    )
    result.add_argument("--project-root")
    result.add_argument("--medical-root")
    result.add_argument("--work")
    result.add_argument("--source-work")
    result.add_argument("--train-config")
    result.add_argument("--nnunet-config")
    result.add_argument("--outer-fold", type=int, default=0)
    result.add_argument("--dataset-id-base", type=int, default=DEFAULT_DATASET_ID_BASE)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--trainer", default=None)
    result.add_argument(
        "--materialization", choices=("symlink", "hardlink", "copy"), default="symlink"
    )
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def check_environment(layout: Layout) -> None:
    for command in (
        "nnUNetv2_extract_fingerprint",
        "nnUNetv2_plan_experiment",
        "nnUNetv2_preprocess",
        "nnUNetv2_train",
    ):
        require_command(command)
    if not layout.train_config.is_file() or not layout.nnunet_config.is_file():
        raise BenchmarkError("Training or nnU-Net config is missing")
    if not (layout.project / "hiercp" / "pipeline.py").is_file():
        raise BenchmarkError("HierCP source is missing")
    print(
        f"[OK] environment project={layout.project} data={layout.data} "
        f"free={shutil.disk_usage(layout.benchmark).free / 1024**3:.1f} GiB"
    )


def main() -> None:
    args = parser().parse_args()
    try:
        if args.outer_fold not in range(5):
            raise BenchmarkError("--outer-fold must be 0..4")
        layout = make_layout(args)
        train_cfg = load_json(layout.train_config)
        nn_cfg = load_json(layout.nnunet_config)
        if args.target == "check":
            check_environment(layout)
            return
        if args.target in {"split", "all"}:
            build_outer_splits(layout, train_cfg, args.overwrite)
        elif not layout.outer_splits.is_file():
            raise BenchmarkError("Outer split is missing; run target 'split' first")

        if args.target == "status":
            status(layout, args.outer_fold, nn_cfg, args.dataset_id_base, args.trainer)
            return
        if args.target == "aggregate":
            aggregate(layout)
            return
        if args.target in {"gnn-prepare", "all"}:
            gnn_prepare(
                layout,
                args.outer_fold,
                train_cfg,
                args.overwrite,
                args.dry_run,
            )
        if args.target in {"gnn-train", "all"}:
            gnn_train(
                layout,
                args.outer_fold,
                train_cfg,
                args.device,
                args.overwrite,
                args.dry_run,
            )
        if args.target in {"generate", "all"}:
            if args.dry_run:
                print("[Dry-run] paired generation skipped")
            else:
                generate_pairs(
                    layout,
                    args.outer_fold,
                    train_cfg,
                    nn_cfg,
                    args.device,
                    args.overwrite,
                )
        if args.target in {"dataset", "all"}:
            build_nnunet_datasets(
                layout,
                args.outer_fold,
                train_cfg,
                nn_cfg,
                args.dataset_id_base,
                args.materialization,
                args.overwrite,
            )
        if args.target in {"plan", "all"}:
            plan_and_preprocess(
                layout,
                args.outer_fold,
                nn_cfg,
                args.dataset_id_base,
                args.dry_run,
            )
        if args.target in {"train", "all"}:
            train_nnunet_pair(
                layout,
                args.outer_fold,
                nn_cfg,
                args.dataset_id_base,
                args.device,
                args.trainer,
                args.dry_run,
            )
        if args.target in {"evaluate", "all"}:
            if args.dry_run:
                print("[Dry-run] evaluation skipped")
            else:
                evaluate_pair(
                    layout,
                    args.outer_fold,
                    train_cfg,
                    nn_cfg,
                    args.dataset_id_base,
                    args.trainer,
                )
        print("\n[Done]")
        status(layout, args.outer_fold, nn_cfg, args.dataset_id_base, args.trainer)
    except (BenchmarkError, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc


if __name__ == "__main__":
    main()
