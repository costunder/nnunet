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
4. audit recorded online event/source/intensity/augmentation schedule counts
   and fingerprints for Basic, Full, w/o-L1 and w/o-L2;
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
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.online_eval_provenance import full_method_identity, verify_evaluation_contract

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
TOOL_VERSION = "hiercp_downstream_level_ablation_v3"
SCHEDULE_AUDIT_FORMAT = "hiercp_downstream_ablation_schedule_audit_v3"
SCHEDULE_DUPLICATE_POLICIES = ("error", "coalesce-identical")
SCHEDULE_COUNT_SEMANTICS = "unique_epoch_schedules_not_optimizer_steps"


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
        return self.report_root / name

    @property
    def report_root(self) -> Path:
        return self.base.output


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


def _publish_report_text(path: Path, content: str) -> None:
    """Publish one report without replacing an existing result, even in a race."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise DownstreamAblationError(f"Refusing to replace existing report: {path}. Use a new --evaluation-output directory.") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _publish_report_json(path: Path, payload: Any) -> None:
    _publish_report_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _reserve_evaluation_output(run: RuntimeLayout, *, duplicate_policy: str = "error") -> None:
    _validate_schedule_duplicate_policy(duplicate_policy)
    try:
        run.report_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise DownstreamAblationError(
            f"Evaluation output already exists: {run.report_root}. Existing results are preserved; "
            "choose a NEW --evaluation-output directory, including after an interrupted evaluation."
        ) from exc
    _publish_report_json(run.report_root / "evaluation_started.json", {
        "format": TOOL_VERSION, "complete": False, "outer_fold": run.outer_fold,
        "dataset_id": run.dataset_id, "purpose": "reevaluation_of_existing_predictions",
        "existing_training_and_predictions_modified": False,
        "schedule_duplicate_policy": duplicate_policy,
    })


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
        output=(Path(args.evaluation_output).expanduser().resolve() if args.evaluation_output else
                online / "folds" / f"fold_{args.outer_fold}" / "downstream_level_ablation"),
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
    print(f"  Basic result:     {'checkpoint_present_unverified' if (_model_dir(run, BASIC_TRAINER) / 'checkpoint_final.pth').is_file() else 'missing'}")
    print(f"  Full result:      {'checkpoint_present_unverified' if (_model_dir(run, FULL_TRAINER) / 'checkpoint_final.pth').is_file() else 'missing'}")
    for mode in MODES:
        gnn_checkpoint = (
            run.base.gnn / "ablation_independent" / mode / "model.pt"
        )
        bank = run.bank_for(mode) / "index.json"
        trainer = MODE_TRAINERS[mode]
        result = _model_dir(run, trainer)
        print(
            f"  {mode:<14} "
            f"gnn={'completion_marker_present_unverified' if _checkpoint_complete(gnn_checkpoint, mode) else 'missing_or_incomplete_marker'} "
            f"bank={'index_present_unverified' if bank.is_file() else 'missing'} "
            f"nnunet={'checkpoint_present_unverified' if (result / 'checkpoint_final.pth').is_file() else 'missing'}"
        )
    print(
        f"  report:           {'report_present_unverified' if (run.report_root / 'comparison.md').is_file() else 'missing'}"
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


def _schedule_log_contract(result: Path) -> list[dict[str, str]]:
    return [{"path": str(path.absolute()), "sha256": _sha256(path)}
            for path in sorted(result.glob("training_log_*.txt"))]


def _verify_schedule_log_contract(result: Path, contract: Sequence[Mapping[str, str]]) -> None:
    if _schedule_log_contract(result) != list(contract):
        raise DownstreamAblationError(f"Training logs changed during schedule audit: {result}. Preserve the logs and evaluate only completed, idle runs.")


def _validate_schedule_duplicate_policy(duplicate_policy: str) -> None:
    if duplicate_policy not in SCHEDULE_DUPLICATE_POLICIES:
        raise DownstreamAblationError(f"Unsupported schedule duplicate policy: {duplicate_policy!r}; choose {SCHEDULE_DUPLICATE_POLICIES}.")


def _schedule_observations_from_result(
    result: Path, *, log_contract: Sequence[Mapping[str, str]] | None = None,
    duplicate_policy: str = "error",
) -> dict[str, Any]:
    """Audit every log occurrence; canonical rows never assert resume continuity."""
    _validate_schedule_duplicate_policy(duplicate_policy)
    pattern = re.compile(
        r"\[OnlineCP\] epoch=(-?\d+) applied=(-?\d+)/(-?\d+) "
        r"rate=[0-9.]+ schedule=([0-9a-fA-F]{16})(?=$|\s)"
    )
    records: dict[int, tuple[int, int, str]] = {}
    locations: dict[int, str] = {}
    occurrences: list[dict[str, Any]] = []
    duplicate_epochs: set[int] = set()
    contract = _schedule_log_contract(result) if log_contract is None else list(log_contract)
    for item in contract:
        path = Path(item["path"])
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = pattern.search(line)
            location = f"{path}:{line_number}"
            if not match:
                if "[OnlineCP] epoch=" in line:
                    raise DownstreamAblationError(f"Malformed OnlineCP epoch record at {location}: {line}")
                continue
            raw_epoch, raw_applied, raw_samples, schedule = match.groups()
            epoch, applied, samples = int(raw_epoch), int(raw_applied), int(raw_samples)
            if epoch not in range(250):
                raise DownstreamAblationError(f"Unexpected OnlineCP epoch {epoch} at {location}; required epochs are 0..249.")
            if samples <= 0 or not 0 <= applied <= samples:
                raise DownstreamAblationError(f"Invalid OnlineCP counts at {location}: applied={applied}, samples={samples}; require samples>0 and 0<=applied<=samples.")
            value = (applied, samples, schedule.lower())
            if epoch in records:
                if duplicate_policy == "error":
                    raise DownstreamAblationError(
                        f"Duplicate OnlineCP epoch {epoch}: first at {locations[epoch]}, again at {location}. "
                        "No duplicate (including an identical resume record) is silently overwritten. "
                        "Only explicitly selecting --schedule-duplicate-policy coalesce-identical permits identical recorded values; "
                        "it does not verify training resume continuity."
                    )
                if records[epoch] != value:
                    raise DownstreamAblationError(
                        f"Conflicting OnlineCP epoch {epoch}: first at {locations[epoch]} with {records[epoch]}, "
                        f"again at {location} with {value}. Conflicting duplicates cannot be coalesced."
                    )
                duplicate_epochs.add(epoch)
            else:
                records[epoch] = value
                locations[epoch] = location
            occurrences.append({
                "path": str(path), "line_number": line_number, "text": line,
                "epoch": epoch, "applied": applied, "samples": samples, "schedule": value[2],
            })
    _verify_schedule_log_contract(result, contract)
    return {"records": records, "audit": {
        "status": "identical_duplicates_coalesced" if duplicate_epochs else "no_duplicates",
        "duplicate_epochs": sorted(duplicate_epochs),
        "duplicate_occurrences": len(occurrences) - len(records),
        "occurrences": occurrences,
        "raw_logged_occurrences": len(occurrences),
        "raw_total_samples": sum(item["samples"] for item in occurrences),
        "raw_total_cp_events": sum(item["applied"] for item in occurrences),
        "unique_epochs": len(records),
        "unique_total_samples": sum(item[1] for item in records.values()),
        "unique_total_cp_events": sum(item[0] for item in records.values()),
    }}


def _schedule_records_from_result(
    result: Path, *, log_contract: Sequence[Mapping[str, str]] | None = None,
    duplicate_policy: str = "error",
) -> dict[int, tuple[int, int, str]]:
    return _schedule_observations_from_result(
        result, log_contract=log_contract, duplicate_policy=duplicate_policy,
    )["records"]


def _audit_schedules(run: RuntimeLayout, *, duplicate_policy: str = "error") -> None:
    _validate_schedule_duplicate_policy(duplicate_policy)
    trainers = {
        "basic": BASIC_TRAINER,
        "full": FULL_TRAINER,
        "no_patient": NO_PATIENT_TRAINER,
        "no_population": NO_POPULATION_TRAINER,
    }
    log_inputs = {}
    records = {}
    duplicate_audit = {}
    for name, trainer in trainers.items():
        result = _model_dir(run, trainer)
        contract = _schedule_log_contract(result)
        observations = _schedule_observations_from_result(result, log_contract=contract, duplicate_policy=duplicate_policy)
        records[name] = observations["records"]
        duplicate_audit[name] = observations["audit"]
        log_inputs[name] = {"result": str(result.absolute()), "files": contract}
    expected = set(range(250))
    incomplete = {
        name: sorted(expected - set(values))
        for name, values in records.items()
        if set(values) != expected
    }
    if incomplete:
        raise DownstreamAblationError(
            f"Online schedule logs must contain exactly epochs 0..249; missing: {incomplete}"
        )
    if any(sum(value[0] for value in values.values()) <= 0 for values in records.values()):
        raise DownstreamAblationError("Online CP schedule has zero total applied events; refusing a CP experiment completion claim.")
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
        "format": SCHEDULE_AUDIT_FORMAT,
        "outer_fold": run.outer_fold,
        "dataset_id": run.dataset_id,
        "matched": True,
        "epochs": 250,
        "epochs_kind": "unique_recorded_epoch_indices",
        "count_semantics": SCHEDULE_COUNT_SEMANTICS,
        "duplicate_policy": duplicate_policy,
        "duplicate_audit": duplicate_audit,
        "training_resume_status": "unverified",
        "methods": trainers,
        "total_samples": sum(value[1] for value in reference.values()),
        "total_cp_events": sum(value[0] for value in reference.values()),
        "log_inputs": log_inputs,
        "epoch_records": {name: [{"epoch": epoch, "applied": item[0], "samples": item[1], "schedule": item[2]}
                                  for epoch, item in sorted(values.items())] for name, values in records.items()},
    }
    _verify_schedule_audit(payload, duplicate_policy=duplicate_policy)
    _publish_report_json(run.report_root / "schedule_audit.json", payload)
    print(
        f"[OK] online schedule audit: Basic/Full/w-o-L1/w-o-L2 matched 250/250 unique epoch indices; "
        f"unique recorded events={payload['total_cp_events']}/{payload['total_samples']}; "
        f"duplicate_policy={duplicate_policy}; training_resume_status=unverified"
    )
    for name, item in duplicate_audit.items():
        print(f"[AUDIT] {name}: logged_records={item['raw_logged_occurrences']}, "
              f"extra_identical_records={item['duplicate_occurrences']}; source lines retained in schedule_audit.json")


def _verify_schedule_audit(payload: Mapping[str, Any], *, duplicate_policy: str = "error") -> None:
    _validate_schedule_duplicate_policy(duplicate_policy)
    if payload.get("duplicate_policy") != duplicate_policy:
        raise DownstreamAblationError("Schedule audit duplicate policy mismatch; use the explicitly intended policy and a new output directory.")
    names = {"basic", "full", "no_patient", "no_population"}
    if (payload.get("format") != SCHEDULE_AUDIT_FORMAT
            or payload.get("matched") is not True or payload.get("epochs") != 250
            or payload.get("epochs_kind") != "unique_recorded_epoch_indices"
            or payload.get("count_semantics") != SCHEDULE_COUNT_SEMANTICS
            or payload.get("training_resume_status") != "unverified"
            or set(payload.get("duplicate_audit", {})) != names
            or set(payload.get("epoch_records", {})) != names
            or set(payload.get("log_inputs", {})) != names):
        raise DownstreamAblationError("Missing or unsupported SHA-bound 250-epoch schedule audit; rerun evaluation in a new directory.")
    reference = None
    for name, item in payload["log_inputs"].items():
        observations = _schedule_observations_from_result(
            Path(item["result"]), log_contract=item["files"], duplicate_policy=duplicate_policy,
        )
        records = observations["records"]
        if payload["duplicate_audit"][name] != observations["audit"]:
            raise DownstreamAblationError(f"Schedule audit duplicate occurrence/count provenance mismatch for {name}.")
        expected_rows = [{"epoch": epoch, "applied": value[0], "samples": value[1], "schedule": value[2]}
                         for epoch, value in sorted(records.items())]
        if (set(records) != set(range(250)) or sum(value[0] for value in records.values()) <= 0
                or payload.get("epoch_records", {}).get(name) != expected_rows):
            raise DownstreamAblationError(f"Schedule audit record/count provenance mismatch for {name}.")
        if reference is None:
            reference = records
        elif records != reference:
            raise DownstreamAblationError(f"Schedule audit recorded nonmatching schedules for {name}.")
    assert reference is not None
    if (payload.get("total_samples") != sum(value[1] for value in reference.values())
            or payload.get("total_cp_events") != sum(value[0] for value in reference.values())):
        raise DownstreamAblationError("Schedule audit total-count mismatch.")


def _verified_pair_summary(output: Path) -> dict[str, Any]:
    summary_path = output / "summary.json"
    digest = _sha256(summary_path)
    summary = _load_json(summary_path)
    completion = _load_json(output / "completion.json")
    contract = summary.get("evaluation_contract")
    if (completion.get("format") != "online_eval_completion_v1" or completion.get("complete") is not True
            or completion.get("summary_sha256") != digest or not isinstance(contract, dict)
            or completion.get("evaluation_contract_sha256") != contract.get("contract_sha256")):
        raise DownstreamAblationError(f"Missing, incomplete or changed pairwise evaluation: {output}. Use a new evaluation output directory.")
    outputs = completion.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("summary.json") != digest:
        raise DownstreamAblationError(f"Missing pairwise output manifest: {output}")
    for relative, expected_digest in outputs.items():
        path = output / relative
        if Path(relative).is_absolute() or not path.resolve().is_relative_to(output.resolve()):
            raise DownstreamAblationError(f"Invalid pairwise output manifest path: {relative}")
        if _sha256(path) != expected_digest:
            raise DownstreamAblationError(f"Pairwise evaluation artifact changed: {path}")
    verify_evaluation_contract(contract)
    if _sha256(summary_path) != digest:
        raise DownstreamAblationError(f"Pairwise summary changed while verifying: {summary_path}")
    return summary


def _run_quality_evaluation(
    run: RuntimeLayout,
    *,
    first_trainer: str,
    second_trainer: str,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    evaluator = Path(__file__).resolve().with_name("online_eval_v2.py")
    if not evaluator.is_file():
        raise DownstreamAblationError(
            f"Matching evaluator is missing from this checkout: {evaluator}; refusing an older external-script fallback."
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
            "--output",
            output,
        ],
        cwd=run.base.medical,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return {}
    return _verified_pair_summary(output)


def _method_from_pair(summary: Mapping[str, Any], side: str) -> dict[str, Any]:
    if side not in {"basic", "hier"}:
        raise ValueError(side)
    case = summary["case_tumor_dice"]
    criteria = summary["criteria"]
    method_key = "basic_cp" if side == "basic" else "hiercp"
    size_key = "basic_recall" if side == "basic" else "hier_recall"
    output = {
        "tumor_dice": case[f"{side}_mean"],
        "lesion_dice_mean": summary["lesion_dice_quality"][f"{side}_mean"],
        "criteria": {},
    }
    for name, values in criteria.items():
        item = dict(values[method_key])
        item["le_10mm_recall"] = values["by_size"].get("le_10mm", {}).get(size_key)
        item["gt_10_le_20mm_recall"] = values["by_size"].get("gt_10_le_20mm", {}).get(size_key)
        output["criteria"][name] = item
    return output


def _validate_pairwise_provenance(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    expected = {"basic_vs_full": BASIC_TRAINER, "no_patient_vs_full": NO_PATIENT_TRAINER,
                "no_population_vs_full": NO_POPULATION_TRAINER}
    if set(summaries) != set(expected):
        raise DownstreamAblationError("Exactly the three prespecified pairwise evaluations are required.")
    full_identity = None
    for name, first_trainer in expected.items():
        contract = summaries[name].get("evaluation_contract")
        if not isinstance(contract, dict):
            raise DownstreamAblationError(f"Missing evaluation provenance in {name}; reevaluate existing predictions in a new directory.")
        verify_evaluation_contract(contract)
        if contract["methods"]["basic"]["trainer"] != first_trainer or contract["methods"]["hier"]["trainer"] != FULL_TRAINER:
            raise DownstreamAblationError(f"Unexpected trainer pair in {name}.")
        candidate = full_method_identity(contract, "hier")
        if full_identity is None:
            full_identity = candidate
        elif candidate != full_identity:
            raise DownstreamAblationError(f"Full prediction/cohort/ground-truth/evaluation-definition identity mismatch in {name}; equal mean Dice is insufficient.")
    assert full_identity is not None
    return full_identity


def _format_number(value: Any, digits: int = 4, *, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    if not np.isfinite(number):
        raise DownstreamAblationError(f"Non-finite report metric: {value!r}")
    return format(number, ("+" if signed else "") + f".{digits}f")


def _aggregate_evaluation(
    run: RuntimeLayout, summaries: Mapping[str, Mapping[str, Any]], *, duplicate_policy: str = "error",
) -> None:
    full_identity = _validate_pairwise_provenance(summaries)
    case_ids = _validation_ids(run)
    if (len(case_ids) != len(set(case_ids)) or len(case_ids) != len(full_identity["cohort"])
            or set(case_ids) != set(full_identity["cohort"])):
        raise DownstreamAblationError("Verified evaluation cohort differs from the requested outer split.")
    for name, summary in summaries.items():
        if (summary.get("outer_fold") != run.outer_fold or summary.get("dataset_id") != run.dataset_id
                or summary["evaluation_contract"]["evaluation_definition"].get("outer_fold") != run.outer_fold
                or summary["evaluation_contract"]["evaluation_definition"].get("dataset_id") != run.dataset_id):
            raise DownstreamAblationError(f"Verified evaluation fold/dataset mismatch in {name}.")
        if _verified_pair_summary(run.evaluation_for(name)) != summary:
            raise DownstreamAblationError(f"Pair summary changed since evaluation: {name}")
    schedule_path = run.report_root / "schedule_audit.json"
    schedule_sha256 = _sha256(schedule_path)
    schedule_audit = _load_json(schedule_path)
    _verify_schedule_audit(schedule_audit, duplicate_policy=duplicate_policy)
    coalesced_occurrences = sum(item["duplicate_occurrences"] for item in schedule_audit["duplicate_audit"].values())
    basic_vs_full = summaries["basic_vs_full"]
    no_patient_vs_full = summaries["no_patient_vs_full"]
    no_population_vs_full = summaries["no_population_vs_full"]
    methods = {
        "basic": _method_from_pair(basic_vs_full, "basic"),
        "full": _method_from_pair(basic_vs_full, "hier"),
        "no_patient": _method_from_pair(no_patient_vs_full, "basic"),
        "no_population": _method_from_pair(no_population_vs_full, "basic"),
    }
    # Input identity was checked above using the exact cohort and all GT/Full
    # prediction hashes. Retain a separate metric consistency check as well.
    full_reference = methods["full"]
    for key, summary in (
        ("no_patient_vs_full", no_patient_vs_full),
        ("no_population_vs_full", no_population_vs_full),
    ):
        candidate = _method_from_pair(summary, "hier")
        if candidate != full_reference:
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
        "schedule_audit_sha256": schedule_sha256,
        "schedule_duplicate_policy": duplicate_policy,
        "schedule_duplicate_summary": {
            name: {key: value for key, value in item.items() if key != "occurrences"}
            for name, item in schedule_audit["duplicate_audit"].items()
        },
        "training_resume_status": "unverified",
        "full_input_identity": full_identity,
        "pair_evaluation_contract_sha256": {name: summary["evaluation_contract"]["contract_sha256"] for name, summary in summaries.items()},
        "interpretation": {"scope": "exploratory_single_outer_fold", "multiplicity_adjusted": False,
                           "primary_endpoint_selected_by_this_report": False,
                           "automatic_architecture_removal_supported": False},
    }

    criterion = "dice_ge_0p10"
    criterion25 = "dice_ge_0p25"
    lines = [
        "# HierCP Downstream Leave-One-Level-Out Ablation",
        "",
        f"- Outer fold: {run.outer_fold}",
        f"- Dataset: {_dataset_name(run.dataset_id, run.outer_fold)}",
        "- Basic-CP and Full-M3 exact-argmax results are reused; they are not retrained.",
        "- The experimental design holds source entries and candidate centers fixed and changes only the ablation GNN scores. This reevaluation does not independently re-audit bank creation or the original training run.",
        "- SHA-bound logs report matching CP counts and schedule fingerprints for all 250 unique epoch indices in all four runs; this checks recorded schedules, not unlogged execution.",
        f"- Schedule duplicate policy: `{duplicate_policy}`; {coalesced_occurrences} extra identical log occurrences coalesced, with all source lines retained. Training resume continuity, optimizer steps and checkpoint lineage remain unverified.",
        "- Exact patient IDs, ground-truth files, Full prediction files and evaluation definitions match across all three pairwise evaluations (SHA-256 verified).",
        "- This is an exploratory single-fold report. Pairwise confidence intervals and p-values are not adjusted for multiple comparisons.",
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
            f"| {labels[mode]} | {_format_number(value['tumor_dice'])} | {_format_number(value['lesion_dice_mean'])} | "
            f"{_format_number(c10['recall'])} | {_format_number(c10['precision'])} | {_format_number(c10['f1'])} | "
            f"{_format_number(c10['fp_per_case'], 3)} | {_format_number(c10['le_10mm_recall'])} | "
            f"{_format_number(c25['recall'])} | {_format_number(c25['le_10mm_recall'])} |"
        )

    lines.extend([
        "", "## Recorded schedule occurrence audit", "",
        "Unique totals count each recorded epoch index once. Logged totals count every occurrence, including repeats. Neither proves the number of executed optimizer steps. Original paths, line numbers, text and file hashes are retained in schedule_audit.json.",
        "",
        "| Model | Unique epochs | Logged records | Extra identical records | CP events (unique / logged) | Samples (unique / logged) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, label in labels.items():
        item = schedule_audit["duplicate_audit"][name]
        lines.append(f"| {label} | {item['unique_epochs']} | {item['raw_logged_occurrences']} | "
                     f"{item['duplicate_occurrences']} | {item['unique_total_cp_events']} / {item['raw_total_cp_events']} | "
                     f"{item['unique_total_samples']} / {item['raw_total_samples']} |")
    lines.extend(
        [
            "",
            "## Exploratory paired differences for each retained level",
            "",
            "Positive differences favor Full M3 on that metric in this fold; they do not establish a generalizable level effect. For FP/case, positive reduction means fewer false positives with Full M3.",
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
        fp_reduction = None if fp["difference"] is None else -float(fp["difference"])
        lines.append(
            f"| {label} | {_format_number(dice_stats['difference'], signed=True)} | "
            f"[{_format_number(dice_stats['ci_low'], signed=True)}, {_format_number(dice_stats['ci_high'], signed=True)}] | "
            f"{_format_number(dice_stats['permutation_p'])} | "
            f"{_format_number(recall['difference'], signed=True)} | "
            f"[{_format_number(recall['ci_low'], signed=True)}, {_format_number(recall['ci_high'], signed=True)}] | "
            f"{_format_number(recall['permutation_p'])} | "
            f"{_format_number(f1['difference'], signed=True)} | {_format_number(fp_reduction, 3, signed=True)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits and next comparison",
            "",
            "- A favorable point estimate alone does not establish benefit. Read the paired confidence interval and p-value for each prespecified endpoint; an interval crossing zero leaves the direction uncertain.",
            "- A non-significant difference does not demonstrate equivalence or that a branch is unnecessary. Mixed Dice/recall/false-positive changes describe a trade-off, not an automatic architecture decision.",
            "- Do not select a primary endpoint or remove/redesign Patient or Population levels from this single-fold table. Preserve the full design and complete the originally planned outer-fold/repeat comparisons.",
            "- N/A denotes an undefined metric or an absent lesion-size group, never a substituted zero. This report evaluates existing predictions and does not certify training/checkpoint provenance that was not recorded.",
            "",
            f"- Pairwise quality-aware outputs: `{run.report_root}`",
        ]
    )
    _validate_pairwise_provenance(summaries)
    for name, summary in summaries.items():
        if _verified_pair_summary(run.evaluation_for(name)) != summary:
            raise DownstreamAblationError(f"Pair summary changed before aggregate publication: {name}")
    _verify_schedule_audit(schedule_audit, duplicate_policy=duplicate_policy)
    if _sha256(schedule_path) != schedule_sha256:
        raise DownstreamAblationError("Schedule audit changed before aggregate publication.")
    _publish_report_text(run.report_root / "comparison.md", "\n".join(lines) + "\n")
    _publish_report_json(run.report_root / "summary.json", payload)
    _publish_report_json(run.report_root / "completion.json", {
        "format": TOOL_VERSION, "complete": True,
        "schedule_duplicate_policy": duplicate_policy,
        "coalesced_schedule_occurrences": coalesced_occurrences,
        "training_resume_status": "unverified",
        "summary_sha256": _sha256(run.report_root / "summary.json"),
        "comparison_sha256": _sha256(run.report_root / "comparison.md"),
        "schedule_audit_sha256": schedule_sha256,
        "pair_summary_sha256": {name: _sha256(run.evaluation_for(name) / "summary.json") for name in summaries},
    })
    print(f"[OK] downstream ablation report: {run.report_root / 'comparison.md'}")


def _evaluate(run: RuntimeLayout, args: argparse.Namespace) -> None:
    if not args.dry_run:
        _reserve_evaluation_output(run, duplicate_policy=args.schedule_duplicate_policy)
        _audit_schedules(run, duplicate_policy=args.schedule_duplicate_policy)
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
        _aggregate_evaluation(run, summaries, duplicate_policy=args.schedule_duplicate_policy)


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
    evaluator = Path(__file__).resolve().with_name("online_eval_v2.py")
    if not evaluator.is_file():
        raise DownstreamAblationError(f"Matching online_eval_v2.py is missing: {evaluator}")
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
    if not args.dry_run and run.report_root.exists():
        raise DownstreamAblationError(f"Evaluation output already exists: {run.report_root}. Choose a NEW --evaluation-output before starting any training work.")
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
    parser.add_argument("--evaluation-output", help="NEW directory for audit, pairwise reevaluation and combined report; existing outputs are never replaced.")
    parser.add_argument(
        "--schedule-duplicate-policy", choices=SCHEDULE_DUPLICATE_POLICIES, default="error",
        help="Default: reject every repeated epoch. coalesce-identical explicitly combines only identical applied/sample/schedule tuples, retains every source line, and does not certify training resume continuity.",
    )
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
