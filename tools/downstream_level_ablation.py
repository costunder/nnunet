#!/usr/bin/env python3
"""Leakage-safe downstream ablation for HierCP Level 1 and Level 2.

This tool reuses the completed Fold-0 online Basic-CP and Full-M3 exact-argmax
experiments. It does not retrain Basic-CP or Full M3. It performs only the
missing downstream tests:

1. train Fold-specific GNN ablations on the exact Fold GNN cache/prototype:
   - no_patient    (M3 without Level 1)
   - no_population (M3 without Level 2)
2. rescore the *existing identical OnlineCP candidate bank* with those GNNs;
3. train two new exact-argmax nnU-Net models on Dataset730/fold 0;
4. verify that Basic, Full, w/o-L1 and w/o-L2 consumed identical online event,
   source, intensity and augmentation schedules;
5. evaluate Full M3 against each one-level ablation using the quality-aware,
   patient-cluster evaluator and produce one combined report.

The source tumor, candidate centers, CP event schedule, source schedule,
intensity jitter, nnU-Net dataset/fold/plans/network/initialization and 250-epoch
LR schedule remain fixed. Only the GNN used to score the common candidates is
changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

PROJECT_DEFAULT = Path("/home/aicompetition06/Medical/HierCP")
BASIC_TRAINER = "nnUNetTrainer_250epochs_OnlineBasicCP"
FULL_TRAINER = "nnUNetTrainer_250epochs_OnlineHierCPExactArgmax"
NO_PATIENT_TRAINER = (
    "nnUNetTrainer_250epochs_OnlineHierCPNoPatientExactArgmax"
)
NO_POPULATION_TRAINER = (
    "nnUNetTrainer_250epochs_OnlineHierCPNoPopulationExactArgmax"
)
MODES = ("no_patient", "no_population")
MODE_TRAINERS = {
    "no_patient": NO_PATIENT_TRAINER,
    "no_population": NO_POPULATION_TRAINER,
}
MODE_LABELS = {
    "no_patient": "M3 w/o Level 1 — Patient",
    "no_population": "M3 w/o Level 2 — Population",
}
BANK_FORMAT = "hiercp_online_bank_v2"
DERIVED_BANK_VERSION = "hiercp_online_bank_level_ablation_exact_argmax_v1"
TOOL_VERSION = "hiercp_downstream_level_ablation_v1"


class DownstreamAblationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Layout:
    project: Path
    medical: Path
    data: Path
    full_work: Path
    paired: Path
    online: Path
    gnn: Path
    source_bank: Path
    nnroot: Path
    raw: Path
    preprocessed: Path
    results: Path
    logs: Path
    output: Path
    train_config: Path
    nnunet_config: Path
    outer_splits: Path

@dataclass(frozen=True)
class RuntimeLayout:
    base: Layout
    outer_fold: int
    dataset_id: int

    def bank_for(self, mode: str) -> Path:
        return (
            self.base.online
            / "folds"
            / f"fold_{self.outer_fold}"
            / "bank_level_ablation"
            / mode
        )

    def evaluation_for(self, name: str) -> Path:
        return (
            self.base.online
            / "folds"
            / f"fold_{self.outer_fold}"
            / "downstream_level_ablation"
            / name
        )

    @property
    def report_root(self) -> Path:
        return (
            self.base.online
            / "folds"
            / f"fold_{self.outer_fold}"
            / "downstream_level_ablation"
        )


def _natural_key(text: str) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", text)
    ]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DownstreamAblationError(f"Missing JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DownstreamAblationError(f"Invalid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DownstreamAblationError(f"JSON root must be an object: {path}")
    return payload


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**32)


def _run(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    log: Path | None = None,
    dry_run: bool = False,
) -> None:
    values = [str(value) for value in command]
    print("\n$ " + " ".join(values), flush=True)
    if dry_run:
        return
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        values,
        cwd=str(cwd),
        env=None if env is None else dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
    )
    assert process.stdout is not None
    terminal = getattr(sys.stdout, "buffer", None)
    handle = log.open("ab") if log is not None else None
    try:
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
        code = process.wait()
    finally:
        if handle is not None:
            handle.close()
    if code != 0:
        raise DownstreamAblationError(
            f"Command failed ({code}): {' '.join(values)}"
        )


def _require_command(name: str) -> str:
    result = shutil.which(name)
    if result is None:
        raise DownstreamAblationError(f"Required command is missing: {name}")
    return result


def _dataset_name(dataset_id: int, outer_fold: int) -> str:
    return f"Dataset{int(dataset_id):03d}_LiverOnlineCP_OF{int(outer_fold)}"


def _model_dir(run: RuntimeLayout, trainer: str) -> Path:
    nn_cfg = _load_json(run.base.nnunet_config)
    plans = str(nn_cfg["dataset"]["plans"])
    configuration = str(nn_cfg["dataset"]["configuration"])
    return (
        run.base.results
        / _dataset_name(run.dataset_id, run.outer_fold)
        / f"{trainer}__{plans}__{configuration}"
        / "fold_0"
    )


def _validation_ids(run: RuntimeLayout) -> list[str]:
    payload = _load_json(run.base.outer_splits)
    splits = payload.get("splits")
    if not isinstance(splits, list) or not (0 <= run.outer_fold < len(splits)):
        raise DownstreamAblationError(
            f"Outer fold {run.outer_fold} is unavailable in {run.base.outer_splits}"
        )
    selected = splits[run.outer_fold]
    if not isinstance(selected, dict):
        raise DownstreamAblationError("Malformed outer split entry")
    values = selected.get("val")
    if not isinstance(values, list):
        values = selected.get("test")
    if not isinstance(values, list):
        raise DownstreamAblationError("Outer split has no val/test patient list")
    return [str(value) for value in values]


def _validation_complete(folder: Path, case_ids: Sequence[str]) -> bool:
    validation = folder / "validation"
    return (validation / "summary.json").is_file() and all(
        (validation / f"{case_id}.nii.gz").is_file() for case_id in case_ids
    )


def _checkpoint_complete(path: Path, expected_mode: str) -> bool:
    if not path.is_file():
        return False
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return False
    kwargs = payload.get("model_kwargs")
    if not isinstance(kwargs, dict):
        return False
    mode = str(kwargs.get("ablation_mode", "full"))
    return mode == expected_mode and bool(payload.get("training_complete", False))


def _make_layout(args: argparse.Namespace) -> RuntimeLayout:
    project = Path(args.project).expanduser().resolve()
    if not (project / "hiercp" / "model.py").is_file():
        raise DownstreamAblationError(f"HierCP project not found: {project}")
    medical = project.parent
    online = project / "work" / "online_basic_vs_hiercp"
    paired = project / "work" / "paired_basic_vs_hiercp"
    nnroot = online / "nnunetv2"
    base = Layout(
        project=project,
        medical=medical,
        data=medical / "Data",
        full_work=project / "work" / "full",
        paired=paired,
        online=online,
        gnn=paired / "folds" / f"fold_{args.outer_fold}" / "gnn",
        source_bank=online / "folds" / f"fold_{args.outer_fold}" / "bank",
        nnroot=nnroot,
        raw=nnroot / "nnUNet_raw",
        preprocessed=nnroot / "nnUNet_preprocessed",
        results=nnroot / "nnUNet_results",
        logs=online / "logs",
        output=online / "folds" / f"fold_{args.outer_fold}" / "downstream_level_ablation",
        train_config=project / "config" / "train.json",
        nnunet_config=project / "config" / "nnunet.json",
        outer_splits=paired / "outer_splits.json",
    )
    return RuntimeLayout(base=base, outer_fold=int(args.outer_fold), dataset_id=int(args.dataset_id))


def _check_required_assets(run: RuntimeLayout) -> None:
    required = [
        run.base.train_config,
        run.base.nnunet_config,
        run.base.outer_splits,
        run.base.gnn / "graphs" / "config.json",
        run.base.gnn / "graphs" / "index.json",
        run.base.gnn / "prototype.pt",
        run.base.gnn / "split.json",
        run.base.gnn / "model.pt",
        run.base.source_bank / "index.json",
        _model_dir(run, BASIC_TRAINER) / "checkpoint_final.pth",
        _model_dir(run, FULL_TRAINER) / "checkpoint_final.pth",
        run.base.preprocessed
        / _dataset_name(run.dataset_id, run.outer_fold)
        / f"{_load_json(run.base.nnunet_config)['dataset']['plans']}.json",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise DownstreamAblationError(
            "Required completed Fold assets are missing:\n  "
            + "\n  ".join(str(path) for path in missing)
        )
    val_ids = _validation_ids(run)
    for trainer in (BASIC_TRAINER, FULL_TRAINER):
        result = _model_dir(run, trainer)
        if not _validation_complete(result, val_ids):
            raise DownstreamAblationError(
                f"Existing validation is incomplete for {trainer}: {result}"
            )

    bank = _load_json(run.base.source_bank / "index.json")
    if bank.get("format") != BANK_FORMAT:
        raise DownstreamAblationError(
            f"Unsupported source bank format: {bank.get('format')!r}"
        )
    if int(bank.get("outer_fold", -1)) != run.outer_fold:
        raise DownstreamAblationError("Source bank outer-fold mismatch")
    if int(bank.get("dataset_id", -1)) != run.dataset_id:
        raise DownstreamAblationError("Source bank dataset-id mismatch")
    if str(bank.get("checkpoint_sha256")) != _sha256(run.base.gnn / "model.pt"):
        raise DownstreamAblationError(
            "Source OnlineCP bank was not scored by the current Fold full-M3 checkpoint"
        )
    if str(bank.get("prototype_sha256")) != _sha256(run.base.gnn / "prototype.pt"):
        raise DownstreamAblationError(
            "Source OnlineCP bank prototype does not match the Fold GNN prototype"
        )


def _print_status(run: RuntimeLayout) -> None:
    print("HierCP downstream level-ablation status")
    print(f"  outer fold:       {run.outer_fold}")
    print(f"  dataset:          {_dataset_name(run.dataset_id, run.outer_fold)}")
    print(f"  Fold GNN work:    {run.base.gnn}")
    print(f"  source bank:      {run.base.source_bank}")
    print(f"  Basic result:     {'complete' if (_model_dir(run, BASIC_TRAINER) / 'checkpoint_final.pth').is_file() else 'missing'}")
    print(f"  Full result:      {'complete' if (_model_dir(run, FULL_TRAINER) / 'checkpoint_final.pth').is_file() else 'missing'}")
    for mode in MODES:
        gnn_checkpoint = (
            run.base.gnn / "ablation_independent" / mode / "model.pt"
        )
        bank = run.bank_for(mode) / "index.json"
        trainer = MODE_TRAINERS[mode]
        result = _model_dir(run, trainer)
        print(
            f"  {mode:<14} "
            f"gnn={'complete' if _checkpoint_complete(gnn_checkpoint, mode) else 'missing'} "
            f"bank={'ready' if bank.is_file() else 'missing'} "
            f"nnunet={'complete' if (result / 'checkpoint_final.pth').is_file() else 'missing'}"
        )
    print(
        f"  report:           {'ready' if (run.report_root / 'comparison.md').is_file() else 'missing'}"
    )


def _train_gnn_ablations(run: RuntimeLayout, args: argparse.Namespace) -> None:
    base_seed = int(_load_json(run.base.train_config).get("seed", 42)) + run.outer_fold
    tool = run.base.medical / "hiercp_ablate"
    if not tool.is_file():
        raise DownstreamAblationError(
            f"hiercp_ablate is missing: {tool}. Install the independent ablation patch first."
        )
    _run(
        [
            tool,
            "train",
            "--work",
            run.base.gnn,
            "--modes",
            "no_patient,no_population",
            "--epochs",
            str(int(args.gnn_epochs)),
            "--seed",
            str(base_seed),
            "--device",
            args.device,
        ],
        cwd=run.base.medical,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return
    for mode in MODES:
        checkpoint = run.base.gnn / "ablation_independent" / mode / "model.pt"
        if not _checkpoint_complete(checkpoint, mode):
            raise DownstreamAblationError(
                f"Fold-specific GNN ablation did not complete: {checkpoint}"
            )


def _load_models(run: RuntimeLayout, device: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.tensor import torch_load_compat

    paths = {
        "full": run.base.gnn / "model.pt",
        "no_patient": run.base.gnn / "ablation_independent" / "no_patient" / "model.pt",
        "no_population": run.base.gnn / "ablation_independent" / "no_population" / "model.pt",
    }
    models: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    reference_graph = None
    reference_clip = None
    reference_prototype = None
    for mode, path in paths.items():
        checkpoint = torch_load_compat(path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise DownstreamAblationError(f"Invalid checkpoint: {path}")
        kwargs = dict(checkpoint.get("model_kwargs") or {})
        recorded = str(kwargs.get("ablation_mode", "full"))
        if recorded != mode:
            raise DownstreamAblationError(
                f"Checkpoint mode mismatch: {path}: {recorded!r} != {mode!r}"
            )
        graph = checkpoint.get("graph_config")
        clip = checkpoint.get("ct_clip")
        prototype = checkpoint.get("prototype_fingerprint")
        if mode == "full":
            reference_graph = graph
            reference_clip = clip
            reference_prototype = prototype
        elif graph != reference_graph or clip != reference_clip or prototype != reference_prototype:
            raise DownstreamAblationError(
                f"Ablation checkpoint contract differs from Fold Full M3: {path}"
            )
        model = HierarchicalPyGPlacementModel(**kwargs)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device)
        model.eval()
        models[mode] = model
        checkpoints[mode] = checkpoint
        print(f"[OK] GNN model {mode}: {path}", flush=True)
    return models, checkpoints


def _source_from_component(case: Any, components: np.ndarray, component_id: int, pad: int) -> Any:
    from hiercp.common import SourceTumor, bbox_of_mask
    from scipy import ndimage as ndi

    full_mask = components == int(component_id)
    if not np.any(full_mask):
        raise DownstreamAblationError(
            f"Empty source component {component_id} for {case.paths.case_id}"
        )
    slices = bbox_of_mask(full_mask, pad=int(pad))
    patch_mask = full_mask[slices].astype(bool, copy=True)
    patch_image = case.image[slices].astype(np.float32, copy=True)
    starts = np.asarray([value.start for value in slices], dtype=np.int64)
    anchor = starts + np.asarray(patch_mask.shape, dtype=np.int64) // 2
    centroid = ndi.center_of_mass(full_mask)
    return SourceTumor(
        component_id=int(component_id),
        full_mask=full_mask,
        patch_mask=patch_mask,
        patch_image=patch_image,
        patch_slices=slices,
        anchor_center=tuple(int(value) for value in anchor),
        centroid=tuple(float(value) for value in centroid),
        voxel_count=int(full_mask.sum()),
    )


def _candidate_infos(case: Any, source: Any, centers: np.ndarray, regions: Any, *, liver_label: int, tumor_label: int) -> list[Any]:
    from hiercp.common import (
        CandidateInfo,
        context_ring_mask,
        distance_to_mask_mm,
        slices_for_center,
    )

    liver = case.label == int(liver_label)
    occupied = case.label == int(tumor_label)
    occupied_distance = distance_to_mask_mm(occupied, case.spacing)
    source_ring = context_ring_mask(source.patch_mask, width=3)
    output: list[Any] = []
    for raw in np.asarray(centers, dtype=np.int64):
        center = tuple(int(value) for value in raw)
        slices = slices_for_center(center, source.patch_mask.shape, case.shape)
        if slices is None:
            raise DownstreamAblationError(
                f"Stored candidate no longer fits volume: {case.paths.case_id} {center}"
            )
        roi_liver = liver[slices]
        coverage = float(
            np.sum(source.patch_mask & roi_liver) / max(1, source.voxel_count)
        )
        roi_image = case.image[slices]
        roi_organ = regions.full_organ_mask[slices]
        values = roi_image[source_ring & roi_organ]
        if values.size < 8:
            values = roi_image[roi_organ & ~source.patch_mask]
        if values.size == 0:
            values = roi_image.reshape(-1)
        output.append(
            CandidateInfo(
                center=center,
                slices=slices,
                liver_coverage=coverage,
                border_distance_mm=float(regions.organ_depth[center]),
                occupied_distance_mm=float(occupied_distance[center]),
                context_mean_hu=float(np.mean(values)),
                context_std_hu=float(np.std(values)),
            )
        )
    return output


def _select_region_cache(run: RuntimeLayout) -> Path:
    candidates = [
        run.base.gnn / "regions",
        run.base.full_work / "regions",
    ]
    for path in candidates:
        if path.is_dir() and any(path.iterdir()):
            return path
    raise DownstreamAblationError(
        "No compatible region cache directory is available for Fold rescoring"
    )


def _write_derived_bank(
    run: RuntimeLayout,
    *,
    mode: str,
    source_index: Mapping[str, Any],
    checkpoint_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    root = run.bank_for(mode)
    entries_by_case = source_index.get("entries_by_case")
    if not isinstance(entries_by_case, dict):
        raise DownstreamAblationError("Source bank has no entries_by_case mapping")
    metadata = dict(source_index)
    metadata.update(
        {
            "format": BANK_FORMAT,
            "version": DERIVED_BANK_VERSION,
            "ablation_mode": mode,
            "candidate_policy": "exact_gnn_argmax",
            "source_bank": str((run.base.source_bank / "index.json").resolve()),
            "source_bank_sha256": _sha256(run.base.source_bank / "index.json"),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "prototype_sha256": _sha256(run.base.gnn / "prototype.pt"),
            "entries_by_case": entries_by_case,
            "manifest": str((root / "manifest.csv").resolve()),
        }
    )
    _atomic_json(root / "index.json", metadata)
    fields = (
        "case_id",
        "source_component",
        "entry",
        "candidate_count",
        "source_pool_sha256",
        "full_reproduction_max_abs",
        "full_argmax_match",
        "full_argmax_tie_equivalent",
        "full_original_top_gap",
        "full_reproduced_top_gap",
        "score_min",
        "score_max",
        "score_std",
        "status",
    )
    _atomic_csv(root / "manifest.csv", rows, fields)
    _atomic_json(
        root / "complete.json",
        {
            "format": BANK_FORMAT,
            "version": DERIVED_BANK_VERSION,
            "ablation_mode": mode,
            "index_sha256": _sha256(root / "index.json"),
            "source_entries": len(rows),
            "candidate_count": int(source_index["candidate_count"]),
        },
    )


def _rescore_banks(run: RuntimeLayout, args: argparse.Namespace) -> None:
    if args.dry_run:
        print("[Dry-run] rescore existing Fold candidate bank with no_patient/no_population")
        return
    if str(run.base.project) not in sys.path:
        sys.path.insert(0, str(run.base.project))

    import torch
    from scipy import ndimage as ndi
    from hiercp.cache import build_inference_sample
    from hiercp.common import CasePaths, load_case
    from hiercp.data import collate_samples
    from hiercp.prototype import PrototypeBank
    from hiercp.region import REGION_CACHE_SEED_SALT, load_or_build_patient_regions
    from hiercp.schema import graph_config_from_dict
    from hiercp.tensor import configure_runtime, resolve_device, set_seed

    source_index_path = run.base.source_bank / "index.json"
    source_index = _load_json(source_index_path)
    entries_by_case = source_index.get("entries_by_case")
    if not isinstance(entries_by_case, dict):
        raise DownstreamAblationError("Malformed source bank entries_by_case")
    candidate_count = int(source_index["candidate_count"])
    train_cfg = _load_json(run.base.train_config)
    labels = train_cfg["labels"]
    liver_label = int(labels["liver"])
    tumor_label = int(labels["tumor"])
    source_pad = int(train_cfg["cache"].get("source_pad", 4))
    base_seed = int(train_cfg.get("seed", 42)) + run.outer_fold
    deterministic = bool(train_cfg.get("runtime", {}).get("deterministic", True))
    set_seed(base_seed, deterministic=deterministic)
    configure_runtime(
        deterministic=deterministic,
        allow_tf32=bool(train_cfg.get("runtime", {}).get("allow_tf32", False)),
        cudnn_benchmark=False,
    )
    device = resolve_device(args.device)
    models, checkpoints = _load_models(run, device)
    full_checkpoint = checkpoints["full"]
    graph_config = graph_config_from_dict(dict(full_checkpoint["graph_config"]))
    ct_clip = tuple(float(value) for value in full_checkpoint["ct_clip"])
    prototype = PrototypeBank.load(run.base.gnn / "prototype.pt")
    if prototype.fingerprint() != full_checkpoint.get("prototype_fingerprint"):
        raise DownstreamAblationError("Fold prototype/checkpoint fingerprint mismatch")
    region_cache = _select_region_cache(run)
    print(f"[INFO] region cache: {region_cache}", flush=True)

    case_paths = {
        path.name[: -len("_0000.nii.gz")]: CasePaths(
            case_id=path.name[: -len("_0000.nii.gz")],
            image_path=path.resolve(),
            label_path=(run.base.data / "labels" / f"{path.name[: -len('_0000.nii.gz')]}.nii.gz").resolve(),
        )
        for path in (run.base.data / "image").glob("*_0000.nii.gz")
    }
    rows_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    total_entries = sum(len(values) for values in entries_by_case.values())
    progress = 0
    use_amp = device.type == "cuda"

    for case_id in sorted(entries_by_case, key=_natural_key):
        relative_paths = entries_by_case[case_id]
        if case_id not in case_paths:
            raise DownstreamAblationError(f"Raw case missing: {case_id}")
        case = load_case(case_paths[case_id])
        components, _ = ndi.label(
            case.label == tumor_label,
            structure=ndi.generate_binary_structure(3, 1),
        )
        region_seed = _stable_seed(base_seed, case_id, REGION_CACHE_SEED_SALT)
        regions = load_or_build_patient_regions(
            case,
            cache_dir=region_cache,
            liver_label=liver_label,
            tumor_label=tumor_label,
            config=graph_config,
            seed=region_seed,
            ct_clip=ct_clip,
            overwrite=False,
            mmap=False,
        )
        for relative in relative_paths:
            progress += 1
            source_path = run.base.source_bank / str(relative)
            if not source_path.is_file():
                raise DownstreamAblationError(f"Missing source bank entry: {source_path}")
            target_paths = {
                mode: run.bank_for(mode) / str(relative)
                for mode in MODES
            }
            if all(path.is_file() for path in target_paths.values()) and not args.overwrite_banks:
                print(f"[Reuse] score bank {progress}/{total_entries} {case_id}/{Path(relative).name}", flush=True)
                for mode in MODES:
                    with np.load(target_paths[mode], allow_pickle=False) as payload:
                        scores = np.asarray(payload["scores"], dtype=np.float32)
                        component = int(np.asarray(payload["source_component"]).reshape(-1)[0])
                        raw_centers = np.asarray(payload["candidate_raw_centers"], dtype=np.int32)
                        pre_centers = np.asarray(payload["candidate_centers"], dtype=np.int32)
                    reused_pool_hash = hashlib.sha256(
                        raw_centers.tobytes() + pre_centers.tobytes()
                    ).hexdigest()
                    rows_by_mode[mode].append(
                        {
                            "case_id": case_id,
                            "source_component": component,
                            "entry": str(relative),
                            "candidate_count": candidate_count,
                            "source_pool_sha256": reused_pool_hash,
                            "full_reproduction_max_abs": "reused",
                            "full_argmax_match": "reused",
                            "full_argmax_tie_equivalent": "reused",
                            "full_original_top_gap": "reused",
                            "full_reproduced_top_gap": "reused",
                            "score_min": f"{float(scores.min()):.8f}",
                            "score_max": f"{float(scores.max()):.8f}",
                            "score_std": f"{float(scores.std()):.8f}",
                            "status": "ok",
                        }
                    )
                continue

            print(f"[Rescore] {progress}/{total_entries} {case_id}/{Path(relative).name}", flush=True)
            with np.load(source_path, allow_pickle=False) as payload:
                source_data = np.asarray(payload["source_data"], dtype=np.float32)
                source_mask = np.asarray(payload["source_mask"], dtype=np.uint8)
                anchor_offset = np.asarray(payload["anchor_offset"])
                pre_centers = np.asarray(payload["candidate_centers"], dtype=np.int32)
                raw_centers = np.asarray(payload["candidate_raw_centers"], dtype=np.int32)
                original_scores = np.asarray(payload["scores"], dtype=np.float32)
                source_component = int(np.asarray(payload["source_component"]).reshape(-1)[0])
                source_diameter = np.asarray(payload["source_diameter_mm"], dtype=np.float32)
            if raw_centers.shape != (candidate_count, 3):
                raise DownstreamAblationError(
                    f"Bad raw candidates in {source_path}: {raw_centers.shape}"
                )
            source = _source_from_component(
                case, components, source_component, source_pad
            )
            candidates = _candidate_infos(
                case,
                source,
                raw_centers,
                regions,
                liver_label=liver_label,
                tumor_label=tumor_label,
            )
            sample, _ = build_inference_sample(
                case,
                source,
                candidates,
                prototype,
                graph_config=graph_config,
                liver_label=liver_label,
                tumor_label=tumor_label,
                ct_clip=ct_clip,
                seed=_stable_seed(base_seed, case_id, source_component, "score"),
                regions=regions,
            )
            batch = collate_samples([sample])
            scores_by_mode: dict[str, np.ndarray] = {}
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                for mode in ("full", *MODES):
                    values = models[mode].score_inference_chunked(
                        batch, local_chunk_size=int(args.local_chunk_size)
                    )[0]
                    scores_by_mode[mode] = (
                        values.detach().float().cpu().numpy().astype(np.float32)
                    )
            reproduced = scores_by_mode["full"]
            max_abs = float(np.max(np.abs(reproduced - original_scores)))
            original_argmax = int(np.argmax(original_scores))
            reproduced_argmax = int(np.argmax(reproduced))
            argmax_match = reproduced_argmax == original_argmax
            score_close = bool(
                np.allclose(
                    reproduced,
                    original_scores,
                    atol=float(args.score_tolerance),
                    rtol=float(args.score_rtol),
                )
            )

            # The source bank was scored under CUDA AMP. Relation-wise GAT/scatter
            # reductions can differ by one or two float16 ULPs across processes.
            # An exact argmax requirement therefore rejects numerically tied top
            # candidates even when the complete score vector is reproduced within
            # the configured tolerance. Accept an argmax flip only when the old and
            # new winners are mutually indistinguishable under that same tolerance.
            pair_scale = max(
                1.0,
                abs(float(original_scores[original_argmax])),
                abs(float(original_scores[reproduced_argmax])),
                abs(float(reproduced[original_argmax])),
                abs(float(reproduced[reproduced_argmax])),
            )
            configured_tolerance = (
                float(args.score_tolerance) + float(args.score_rtol) * pair_scale
            )
            # With a uniform max error e, two candidates can exchange order only
            # when their score separation is at most 2e. Cap the acceptance band
            # by the user-configured allclose tolerance as an additional guard.
            numerical_flip_bound = (
                2.0 * max_abs
                + 8.0 * float(np.finfo(np.float32).eps) * pair_scale
            )
            tie_tolerance = min(configured_tolerance, numerical_flip_bound)
            original_winner_advantage = float(
                original_scores[original_argmax] - original_scores[reproduced_argmax]
            )
            reproduced_winner_advantage = float(
                reproduced[reproduced_argmax] - reproduced[original_argmax]
            )
            tie_equivalent = bool(
                (not argmax_match)
                and original_winner_advantage <= tie_tolerance
                and reproduced_winner_advantage <= tie_tolerance
            )

            original_sorted = np.sort(original_scores.astype(np.float64))
            reproduced_sorted = np.sort(reproduced.astype(np.float64))
            original_top_gap = float(original_sorted[-1] - original_sorted[-2])
            reproduced_top_gap = float(reproduced_sorted[-1] - reproduced_sorted[-2])

            if not score_close or (not argmax_match and not tie_equivalent):
                raise DownstreamAblationError(
                    f"Full-M3 score reproduction failed for {case_id}/c{source_component}: "
                    f"max_abs={max_abs:.6g} argmax_match={argmax_match} "
                    f"tie_equivalent={tie_equivalent} old_argmax={original_argmax} "
                    f"new_argmax={reproduced_argmax} old_gap={original_top_gap:.6g} "
                    f"new_gap={reproduced_top_gap:.6g} tie_tol={tie_tolerance:.6g}. "
                    "Refusing to build non-comparable banks."
                )
            if tie_equivalent:
                print(
                    f"[Warn] Full-M3 near-tie accepted for {case_id}/c{source_component}: "
                    f"old_argmax={original_argmax} new_argmax={reproduced_argmax} "
                    f"max_abs={max_abs:.6g} old_gap={original_top_gap:.6g} "
                    f"new_gap={reproduced_top_gap:.6g} tie_tol={tie_tolerance:.6g}",
                    flush=True,
                )

            source_pool_hash = hashlib.sha256(
                raw_centers.tobytes() + pre_centers.tobytes()
            ).hexdigest()
            for mode in MODES:
                destination = target_paths[mode]
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
                scores = scores_by_mode[mode]
                with temporary.open("wb") as handle:
                    np.savez(
                        handle,
                        source_data=source_data,
                        source_mask=source_mask,
                        anchor_offset=anchor_offset,
                        candidate_centers=pre_centers,
                        candidate_raw_centers=raw_centers,
                        scores=scores,
                        source_component=np.asarray([source_component], dtype=np.int16),
                        source_diameter_mm=source_diameter,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
                rows_by_mode[mode].append(
                    {
                        "case_id": case_id,
                        "source_component": source_component,
                        "entry": str(relative),
                        "candidate_count": candidate_count,
                        "source_pool_sha256": source_pool_hash,
                        "full_reproduction_max_abs": f"{max_abs:.8f}",
                        "full_argmax_match": int(argmax_match),
                        "full_argmax_tie_equivalent": int(tie_equivalent),
                        "full_original_top_gap": f"{original_top_gap:.8f}",
                        "full_reproduced_top_gap": f"{reproduced_top_gap:.8f}",
                        "score_min": f"{float(scores.min()):.8f}",
                        "score_max": f"{float(scores.max()):.8f}",
                        "score_std": f"{float(scores.std()):.8f}",
                        "status": "ok",
                    }
                )
            del batch, sample
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for mode in MODES:
        if len(rows_by_mode[mode]) != total_entries:
            raise DownstreamAblationError(
                f"Derived bank {mode} has {len(rows_by_mode[mode])}/{total_entries} entries"
            )
        _write_derived_bank(
            run,
            mode=mode,
            source_index=source_index,
            checkpoint_path=(
                run.base.gnn / "ablation_independent" / mode / "model.pt"
            ),
            rows=rows_by_mode[mode],
        )
        print(f"[OK] derived exact-argmax bank {mode}: {run.bank_for(mode) / 'index.json'}")


def _nn_env(
    run: RuntimeLayout,
    bank: Path,
    requested_device: str,
) -> dict[str, str]:
    config = _load_json(run.base.train_config)
    seed = int(config.get("seed", 42)) + run.outer_fold
    env = os.environ.copy()
    env.update(
        {
            "nnUNet_raw": str(run.base.raw),
            "nnUNet_preprocessed": str(run.base.preprocessed),
            "nnUNet_results": str(run.base.results),
            "nnUNet_n_proc_DA": "8",
            "ONLINE_CP_BANK": str(bank.resolve()),
            "ONLINE_CP_SEED": str(seed),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )

    # nnUNetv2_train accepts ``-device cuda`` rather than a physical CUDA
    # index. Select the requested physical GPU through CUDA_VISIBLE_DEVICES.
    # Inside that child process the selected physical GPU is intentionally
    # remapped to logical ``cuda:0``.
    device_text = str(requested_device).strip().lower()
    if device_text.startswith("cuda:"):
        gpu_text = device_text.split(":", 1)[1]
        if not gpu_text.isdigit():
            raise DownstreamAblationError(
                f"Invalid CUDA device specification: {requested_device!r}"
            )
        env["CUDA_VISIBLE_DEVICES"] = gpu_text
    return env


def _train_nnunet_ablations(run: RuntimeLayout, args: argparse.Namespace) -> None:
    nn_cfg = _load_json(run.base.nnunet_config)
    configuration = str(nn_cfg["dataset"]["configuration"])
    plans = str(nn_cfg["dataset"]["plans"])
    device = "cuda" if str(args.device).startswith("cuda") else str(args.device)
    if str(args.device).startswith("cuda:"):
        physical_gpu = str(args.device).split(":", 1)[1]
        print(
            "[Device] nnU-Net requested="
            f"{args.device} physical_gpu={physical_gpu} "
            f"CUDA_VISIBLE_DEVICES={physical_gpu} child_logical_device=cuda:0"
        )
    val_ids = _validation_ids(run)
    for mode in MODES:
        trainer = MODE_TRAINERS[mode]
        bank = run.bank_for(mode) / "index.json"
        if not bank.is_file() and not args.dry_run:
            raise DownstreamAblationError(f"Derived bank is missing: {bank}")
        result = _model_dir(run, trainer)
        final = result / "checkpoint_final.pth"
        if final.is_file() and _validation_complete(result, val_ids):
            print(f"[Reuse] nnU-Net {MODE_LABELS[mode]} complete")
            continue
        command: list[str | os.PathLike[str]] = [
            _require_command("nnUNetv2_train"),
            str(run.dataset_id),
            configuration,
            "0",
            "-tr",
            trainer,
            "-p",
            plans,
            "-device",
            device,
        ]
        if final.is_file():
            command.append("--val")
        elif (result / "checkpoint_latest.pth").is_file() or (
            result / "checkpoint_best.pth"
        ).is_file():
            command.append("--c")
        log = run.base.logs / f"train_online_hier_{mode}_exact_argmax_of{run.outer_fold}.log"
        _run(
            command,
            cwd=run.base.project,
            env=_nn_env(run, bank, args.device),
            log=log,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            continue
        if not final.is_file():
            raise DownstreamAblationError(f"Final checkpoint missing: {final}")
        if not _validation_complete(result, val_ids):
            _run(
                [
                    _require_command("nnUNetv2_train"),
                    str(run.dataset_id),
                    configuration,
                    "0",
                    "-tr",
                    trainer,
                    "-p",
                    plans,
                    "--val",
                    "-device",
                    device,
                ],
                cwd=run.base.project,
                env=_nn_env(run, bank, args.device),
                log=log,
                dry_run=False,
            )
        if not _validation_complete(result, val_ids):
            raise DownstreamAblationError(
                f"Validation incomplete for {trainer}: {result}"
            )


def _schedule_records_from_result(result: Path) -> dict[int, tuple[int, int, str]]:
    pattern = re.compile(
        r"\[OnlineCP\] epoch=(\d+) applied=(\d+)/(\d+) "
        r"rate=[0-9.]+ schedule=([0-9a-fA-F]{16})"
    )
    records: dict[int, tuple[int, int, str]] = {}
    candidates = sorted(result.glob("training_log_*.txt"))
    for path in candidates:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                epoch, applied, samples, schedule = match.groups()
                records[int(epoch)] = (int(applied), int(samples), schedule.lower())
    return records


def _audit_schedules(run: RuntimeLayout) -> None:
    trainers = {
        "basic": BASIC_TRAINER,
        "full": FULL_TRAINER,
        "no_patient": NO_PATIENT_TRAINER,
        "no_population": NO_POPULATION_TRAINER,
    }
    records = {
        name: _schedule_records_from_result(_model_dir(run, trainer))
        for name, trainer in trainers.items()
    }
    expected = set(range(250))
    incomplete = {
        name: sorted(expected - set(values))
        for name, values in records.items()
        if set(values) != expected
    }
    if incomplete:
        raise DownstreamAblationError(
            f"Online schedule logs are incomplete: {incomplete}"
        )
    reference = records["basic"]
    mismatch: dict[str, list[int]] = {}
    for name, values in records.items():
        if name == "basic":
            continue
        changed = [epoch for epoch in range(250) if values[epoch] != reference[epoch]]
        if changed:
            mismatch[name] = changed
    if mismatch:
        raise DownstreamAblationError(
            f"Online CP schedules differ from Basic: {mismatch}"
        )
    payload = {
        "format": "hiercp_downstream_ablation_schedule_audit_v1",
        "outer_fold": run.outer_fold,
        "dataset_id": run.dataset_id,
        "matched": True,
        "epochs": 250,
        "methods": trainers,
        "total_samples": sum(value[1] for value in reference.values()),
        "total_cp_events": sum(value[0] for value in reference.values()),
    }
    _atomic_json(run.report_root / "schedule_audit.json", payload)
    print(
        f"[OK] online schedule audit: Basic/Full/w-o-L1/w-o-L2 matched 250/250 epochs; "
        f"events={payload['total_cp_events']}/{payload['total_samples']}"
    )


def _run_quality_evaluation(
    run: RuntimeLayout,
    *,
    first_trainer: str,
    second_trainer: str,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    evaluator = run.base.project / "tools" / "online_eval_v2.py"
    if not evaluator.is_file():
        evaluator = run.base.medical / "online_eval_v2.py"
    if not evaluator.is_file():
        raise DownstreamAblationError(
            "online_eval_v2.py is missing from both project/tools and Medical root"
        )
    _run(
        [
            sys.executable,
            evaluator,
            "--project",
            run.base.project,
            "--outer-fold",
            str(run.outer_fold),
            "--dataset-id",
            str(run.dataset_id),
            "--basic-trainer",
            first_trainer,
            "--hier-trainer",
            second_trainer,
            "--bootstrap-iterations",
            str(int(args.bootstrap_iterations)),
            "--permutation-iterations",
            str(int(args.permutation_iterations)),
            "--allow-regression-mismatch",
            "--output",
            output,
        ],
        cwd=run.base.medical,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return {}
    return _load_json(output / "summary.json")


def _method_from_pair(summary: Mapping[str, Any], side: str) -> dict[str, Any]:
    if side not in {"basic", "hier"}:
        raise ValueError(side)
    case = summary["case_tumor_dice"]
    criteria = summary["criteria"]
    method_key = "basic_cp" if side == "basic" else "hiercp"
    size_key = "basic_recall" if side == "basic" else "hier_recall"
    output = {
        "tumor_dice": float(case[f"{side}_mean"]),
        "lesion_dice_mean": float(summary["lesion_dice_quality"][f"{side}_mean"]),
        "criteria": {},
    }
    for name, values in criteria.items():
        item = dict(values[method_key])
        item["le_10mm_recall"] = values["by_size"]["le_10mm"][size_key]
        item["gt_10_le_20mm_recall"] = values["by_size"]["gt_10_le_20mm"][size_key]
        output["criteria"][name] = item
    return output


def _aggregate_evaluation(run: RuntimeLayout, summaries: Mapping[str, Mapping[str, Any]]) -> None:
    basic_vs_full = summaries["basic_vs_full"]
    no_patient_vs_full = summaries["no_patient_vs_full"]
    no_population_vs_full = summaries["no_population_vs_full"]
    methods = {
        "basic": _method_from_pair(basic_vs_full, "basic"),
        "full": _method_from_pair(basic_vs_full, "hier"),
        "no_patient": _method_from_pair(no_patient_vs_full, "basic"),
        "no_population": _method_from_pair(no_population_vs_full, "basic"),
    }
    # The Full prediction must be identical in all pairwise evaluations.
    full_reference = methods["full"]
    for key, summary in (
        ("no_patient_vs_full", no_patient_vs_full),
        ("no_population_vs_full", no_population_vs_full),
    ):
        candidate = _method_from_pair(summary, "hier")
        if abs(candidate["tumor_dice"] - full_reference["tumor_dice"]) > 1e-12:
            raise DownstreamAblationError(f"Full metric mismatch in {key}")

    direct = {
        "level1_patient": {
            "definition": "Full M3 - M3 w/o Level 1",
            "pair_summary": "no_patient_vs_full",
            "case_tumor_dice": no_patient_vs_full["case_tumor_dice"]["statistics"],
            "criteria": {
                name: values["statistics"]
                for name, values in no_patient_vs_full["criteria"].items()
            },
        },
        "level2_population": {
            "definition": "Full M3 - M3 w/o Level 2",
            "pair_summary": "no_population_vs_full",
            "case_tumor_dice": no_population_vs_full["case_tumor_dice"]["statistics"],
            "criteria": {
                name: values["statistics"]
                for name, values in no_population_vs_full["criteria"].items()
            },
        },
    }
    payload = {
        "format": TOOL_VERSION,
        "outer_fold": run.outer_fold,
        "dataset_id": run.dataset_id,
        "methods": methods,
        "direct_downstream_effects": direct,
        "schedule_audit": str((run.report_root / "schedule_audit.json").resolve()),
    }
    _atomic_json(run.report_root / "summary.json", payload)

    criterion = "dice_ge_0p10"
    criterion25 = "dice_ge_0p25"
    lines = [
        "# HierCP Downstream Leave-One-Level-Out Ablation",
        "",
        f"- Outer fold: {run.outer_fold}",
        f"- Dataset: {_dataset_name(run.dataset_id, run.outer_fold)}",
        "- Basic-CP and Full-M3 exact-argmax results are reused; they are not retrained.",
        "- M3 w/o Level 1 and M3 w/o Level 2 use the exact same source entries and candidate centers as Full M3; only the Fold-specific ablation GNN scores differ.",
        "- All four nnU-Net runs use the same online CP event/source/appearance/augmentation schedule.",
        "",
        "## Downstream performance",
        "",
        "| Model | Tumor Dice | Lesion Dice mean | Recall D>=0.10 | Precision | F1 | FP/case | <=10 mm recall D>=0.10 | Recall D>=0.25 | <=10 mm recall D>=0.25 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "basic": "Basic-CP",
        "full": "Full M3 exact argmax",
        "no_patient": "M3 w/o Level 1",
        "no_population": "M3 w/o Level 2",
    }
    for mode in ("basic", "full", "no_patient", "no_population"):
        value = methods[mode]
        c10 = value["criteria"][criterion]
        c25 = value["criteria"][criterion25]
        lines.append(
            f"| {labels[mode]} | {value['tumor_dice']:.4f} | {value['lesion_dice_mean']:.4f} | "
            f"{c10['recall']:.4f} | {c10['precision']:.4f} | {c10['f1']:.4f} | "
            f"{c10['fp_per_case']:.3f} | {c10['le_10mm_recall']:.4f} | "
            f"{c25['recall']:.4f} | {c25['le_10mm_recall']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Direct downstream effect of each retained level",
            "",
            "Positive values mean Full M3 performed better than the model with that level removed. For FP/case, a positive reduction means Full M3 produced fewer false positives.",
            "",
            "| Level | Delta tumor Dice | 95% CI | p | Delta recall D>=0.10 | 95% CI | p | Delta F1 D>=0.10 | Delta FP reduction/case |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    pairs = {
        "Level 1 — Patient": no_patient_vs_full,
        "Level 2 — Population": no_population_vs_full,
    }
    for label, summary in pairs.items():
        dice_stats = summary["case_tumor_dice"]["statistics"]
        stats = summary["criteria"][criterion]["statistics"]
        recall = stats["recall"]
        f1 = stats["f1"]
        fp = stats["fp_per_case"]
        # Evaluator difference is second (Full) - first (ablation).
        fp_reduction = -float(fp["difference"])
        lines.append(
            f"| {label} | {float(dice_stats['difference']):+.4f} | "
            f"[{float(dice_stats['ci_low']):+.4f}, {float(dice_stats['ci_high']):+.4f}] | "
            f"{float(dice_stats['permutation_p']):.4f} | "
            f"{float(recall['difference']):+.4f} | "
            f"[{float(recall['ci_low']):+.4f}, {float(recall['ci_high']):+.4f}] | "
            f"{float(recall['permutation_p']):.4f} | "
            f"{float(f1['difference']):+.4f} | {fp_reduction:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "- Full > w/o Level 1: Patient Graph reranking improves actual downstream segmentation.",
            "- Full ~= or < w/o Level 2: the current Population Prototype branch is unnecessary or needs redesign.",
            "- Full > w/o Level 2: population alignment contributes despite weak positive-ranking gains in the 128-candidate audit.",
            "",
            f"- Pairwise quality-aware outputs: `{run.report_root}`",
        ]
    )
    _atomic_text(run.report_root / "comparison.md", "\n".join(lines) + "\n")
    print(f"[OK] downstream ablation report: {run.report_root / 'comparison.md'}")


def _evaluate(run: RuntimeLayout, args: argparse.Namespace) -> None:
    if not args.dry_run:
        _audit_schedules(run)
    summaries = {
        "basic_vs_full": _run_quality_evaluation(
            run,
            first_trainer=BASIC_TRAINER,
            second_trainer=FULL_TRAINER,
            output=run.evaluation_for("basic_vs_full"),
            args=args,
        ),
        "no_patient_vs_full": _run_quality_evaluation(
            run,
            first_trainer=NO_PATIENT_TRAINER,
            second_trainer=FULL_TRAINER,
            output=run.evaluation_for("no_patient_vs_full"),
            args=args,
        ),
        "no_population_vs_full": _run_quality_evaluation(
            run,
            first_trainer=NO_POPULATION_TRAINER,
            second_trainer=FULL_TRAINER,
            output=run.evaluation_for("no_population_vs_full"),
            args=args,
        ),
    }
    if not args.dry_run:
        _aggregate_evaluation(run, summaries)


def _check_trainer_import() -> None:
    code = f"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCP import (
    {FULL_TRAINER},
    {NO_PATIENT_TRAINER},
    {NO_POPULATION_TRAINER},
)
assert {FULL_TRAINER}.online_policy == 'hier_argmax'
assert {NO_PATIENT_TRAINER}.expected_ablation_mode == 'no_patient'
assert {NO_POPULATION_TRAINER}.expected_ablation_mode == 'no_population'
print('[OK] downstream exact-argmax trainer classes')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        raise DownstreamAblationError("Trainer import smoke failed")


def command_check(run: RuntimeLayout, args: argparse.Namespace) -> None:
    _check_required_assets(run)
    _check_trainer_import()
    for command in ("nnUNetv2_train",):
        _require_command(command)
    evaluator = run.base.project / "tools" / "online_eval_v2.py"
    if not evaluator.is_file() and not (run.base.medical / "online_eval_v2.py").is_file():
        raise DownstreamAblationError("online_eval_v2.py is missing")
    print("[OK] Fold-specific GNN, shared OnlineCP bank, Basic and Full exact-argmax assets")
    _print_status(run)


def command_prepare(run: RuntimeLayout, args: argparse.Namespace) -> None:
    _check_required_assets(run)
    _train_gnn_ablations(run, args)
    _rescore_banks(run, args)


def command_train(run: RuntimeLayout, args: argparse.Namespace) -> None:
    _check_required_assets(run)
    _train_nnunet_ablations(run, args)


def command_all(run: RuntimeLayout, args: argparse.Namespace) -> None:
    command_check(run, args)
    command_prepare(run, args)
    command_train(run, args)
    _evaluate(run, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("check", "status", "prepare", "train", "evaluate", "all"),
    )
    parser.add_argument("--project", default=str(PROJECT_DEFAULT))
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--dataset-id", type=int, default=730)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gnn-epochs", type=int, default=40)
    parser.add_argument("--local-chunk-size", type=int, default=8)
    parser.add_argument("--score-tolerance", type=float, default=0.02)
    parser.add_argument("--score-rtol", type=float, default=0.005)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--permutation-iterations", type=int, default=50000)
    parser.add_argument("--overwrite-banks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run = _make_layout(args)
    try:
        if args.command == "check":
            command_check(run, args)
        elif args.command == "status":
            _print_status(run)
        elif args.command == "prepare":
            command_prepare(run, args)
        elif args.command == "train":
            command_train(run, args)
        elif args.command == "evaluate":
            _evaluate(run, args)
        elif args.command == "all":
            command_all(run, args)
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (DownstreamAblationError, FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
