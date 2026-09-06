#!/usr/bin/env python3
"""Prepare, train and generate with the optimized three-level PyG model."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
import csv
from dataclasses import dataclass, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from hiercp.cache import (
    build_inference_sample,
    prepare_hierarchical_cache,
    prepare_prototype_bank,
)
from hiercp.common import (
    CasePaths,
    LoadedCase,
    SourceTumor,
    build_candidate_pool,
    choose_from_top_k,
    choose_source_tumor,
    discover_cases,
    load_case,
    output_paths,
    paste_source,
    save_case_pair,
    stable_case_seed,
    write_manifest,
)
from hiercp.prototype import PrototypeBank
from hiercp.region import (
    REGION_CACHE_SEED_SALT,
    PatientRegionData,
    build_patient_regions,
    load_or_build_patient_regions,
)
from hiercp.schema import (
    GraphBuildConfig,
    UPPER_FEATURE_POLICY,
    graph_config_from_dict,
)
from hiercp.split import load_case_split


CHECKPOINT_METHOD = "hiercp-full"
RUN_MODES = ("production", "ablation", "benchmark", "debug")
UNAVAILABLE = "unavailable"


def _print_report(name: str, payload: dict[str, Any]) -> None:
    print(f"[{name}] {json.dumps(payload, sort_keys=True, default=str)}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "run_mode", "production"))
    if mode not in RUN_MODES:
        raise ValueError(f"Unsupported run mode {mode!r}; expected one of {RUN_MODES}")
    return mode


def _guard_reduction_overrides(
    args: argparse.Namespace,
    *,
    option_names: tuple[str, ...],
) -> str:
    mode = _run_mode(args)
    supplied = [
        name
        for name in option_names
        if getattr(args, name, None) not in (None, [])
    ]
    if mode == "production" and supplied:
        options = ", ".join("--" + name.replace("_", "-") for name in supplied)
        raise ValueError(
            f"Production mode forbids scale/subset overrides ({options}). Use the "
            "complete config and cohort, or explicitly select --run-mode ablation, "
            "benchmark, or debug for a clearly labelled non-production run."
        )
    return mode


def _add_run_mode(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-mode",
        choices=RUN_MODES,
        default="production",
        help="label non-production reductions explicitly; production rejects them",
    )


def _require_full_graph(graph_config: GraphBuildConfig, context: str) -> None:
    if (
        graph_config.graph_schema_version != "full_v22"
        or not graph_config.canonical_full_graph
        or not graph_config.adaptive_source_full_shape
    ):
        raise RuntimeError(
            f"{context} requires graph_schema_version=full_v22, "
            "canonical_full_graph=true, and adaptive_source_full_shape=true"
        )

def _load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config must contain a JSON object")
    return payload


def _add_prepare_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config/train.json")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--region-cache-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    _add_run_mode(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prototype = commands.add_parser("prepare-prototypes")
    _add_prepare_common(prototype)
    prototype.add_argument("--output", required=True)

    prepare = commands.add_parser("prepare")
    _add_prepare_common(prepare)
    prepare.add_argument("--prototype-bank", required=True)
    prepare.add_argument("--cache-dir", required=True)
    prepare.add_argument("--max-cases", type=int, default=None)

    train = commands.add_parser("train")
    train.add_argument("--config", default="config/train.json")
    train.add_argument("--cache-dir", required=True)
    train.add_argument("--prototype-bank", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--num-workers", type=int, default=None)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument(
        "--ablation-mode",
        choices=("full", "no_local", "no_patient", "no_population"),
        default="full",
    )
    train.add_argument("--overwrite", action="store_true")
    _add_run_mode(train)

    generate = commands.add_parser("generate")
    generate.add_argument("--config", default="config/train.json")
    generate.add_argument("--data-dir", required=True)
    generate.add_argument("--region-cache-dir", required=True)
    generate.add_argument("--prototype-bank", required=True)
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--out-dir", required=True)
    generate.add_argument("--device", default="auto")
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--max-cases", type=int, default=None)
    generate.add_argument("--case-id", action="append", dest="case_ids", default=None)
    generate.add_argument("--overwrite", action="store_true")
    _add_run_mode(generate)
    return parser


def run_prepare_prototypes(args: argparse.Namespace) -> None:
    from hiercp.tensor import collect_runtime_resources, process_memory_snapshot

    run_mode = _run_mode(args)
    config = _load_json(args.config)
    split = load_case_split(args.split_file)
    labels = config["labels"]
    graph = graph_config_from_dict(config["graph"])
    seed = int(args.seed if args.seed is not None else config["seed"])
    training_case_ids = tuple(str(value) for value in split["train"])
    workers = config["runtime"]["prepare_workers"]
    if workers != "auto" and (type(workers) is not int or workers < 1):
        raise ValueError("runtime.prepare_workers must be 'auto' or a positive integer")
    output_preexisting = Path(args.output).is_file() and not args.overwrite
    resources_before = collect_runtime_resources(
        storage_path=Path(args.output).parent
    )
    _print_report(
        "PreRun",
        {
            "command": "prepare-prototypes",
            "run_mode": run_mode,
            "model": f"{UNAVAILABLE} (prototype preparation has no trainable model)",
            "dataset": {
                "configured_training_cases": len(training_case_ids),
                "selected_training_cases": len(training_case_ids),
                "usage_ratio": 1.0,
                "subset_active": False,
            },
            "parallelism": {
                "configured_workers": workers,
                "selection_basis": (
                    "measured full-size case waves; automatic CPU/RAM-constrained "
                    "selection when prepare_workers=auto"
                ),
            },
            "device_resources": resources_before,
            "peak_vram": f"{UNAVAILABLE} (no GPU preparation kernels instrumented)",
        },
    )
    started = time.perf_counter()
    cpu_started = time.process_time()
    bank = prepare_prototype_bank(
        data_dir=args.data_dir,
        output_path=args.output,
        region_cache_dir=args.region_cache_dir,
        training_case_ids=training_case_ids,
        graph_config=graph,
        liver_label=int(labels["liver"]),
        tumor_label=int(labels["tumor"]),
        ct_clip=tuple(float(value) for value in config["ct_clip"]),
        seed=seed,
        overwrite=args.overwrite,
        workers=workers,
    )
    elapsed = time.perf_counter() - started
    cpu_elapsed = time.process_time() - cpu_started
    processed_cases = 0 if output_preexisting else len(training_case_ids)
    affinity = resources_before.get("cpu_affinity_cores")
    cpu_capacity = (
        int(affinity)
        if isinstance(affinity, int) and affinity > 0
        else (os.cpu_count() or 1)
    )
    _print_report(
        "PostRun",
        {
            "command": "prepare-prototypes",
            "run_mode": run_mode,
            "status": "reused" if output_preexisting else "complete",
            "output": str(args.output),
            "configured_training_cases": len(training_case_ids),
            "bank_training_cases": len(bank.training_case_ids),
            "cases_processed_this_invocation": processed_cases,
            "elapsed_seconds": elapsed,
            "cases_per_second": (
                processed_cases / elapsed
                if processed_cases and elapsed > 0.0
                else f"{UNAVAILABLE} (existing prototype bank reused)"
            ),
            "process_cpu_seconds": cpu_elapsed,
            "process_cpu_utilization_percent_of_allocated_capacity": (
                cpu_elapsed / elapsed * 100.0 / cpu_capacity
                if elapsed > 0.0
                else UNAVAILABLE
            ),
            "process_memory": process_memory_snapshot(),
            "device_resources_after": collect_runtime_resources(
                storage_path=Path(args.output).parent
            ),
            "peak_vram": f"{UNAVAILABLE} (no GPU preparation kernels instrumented)",
            "configured_workers": workers,
            "worker_selection_measured": workers == "auto" and not output_preexisting,
        },
    )


def run_prepare(args: argparse.Namespace) -> None:
    from hiercp.tensor import collect_runtime_resources, process_memory_snapshot

    run_mode = _guard_reduction_overrides(args, option_names=("max_cases",))
    config = _load_json(args.config)
    split = load_case_split(args.split_file)
    labels = config["labels"]
    cache = config["cache"]
    graph = graph_config_from_dict(config["graph"])
    seed = int(args.seed if args.seed is not None else config["seed"])
    workers = config["runtime"]["prepare_workers"]
    if workers != "auto" and (type(workers) is not int or workers < 1):
        raise ValueError("runtime.prepare_workers must be 'auto' or a positive integer")
    configured_case_count = len(split["train"]) + len(split["val"])
    selected_case_count = (
        configured_case_count
        if args.max_cases is None
        else min(configured_case_count, int(args.max_cases))
    )
    resources_before = collect_runtime_resources(
        storage_path=args.cache_dir
    )
    _print_report(
        "PreRun",
        {
            "command": "prepare",
            "run_mode": run_mode,
            "model": f"{UNAVAILABLE} (cache preparation has no trainable model)",
            "graph_and_input": {
                "graph_configuration": graph.to_dict(),
                "input_resolution": [graph.patch_size] * 3,
            },
            "dataset": {
                "configured_cases": configured_case_count,
                "selected_cases": selected_case_count,
                "usage_ratio": (
                    selected_case_count / configured_case_count
                    if configured_case_count
                    else 0.0
                ),
                "subset_active": args.max_cases is not None,
                "max_cases": (
                    args.max_cases if args.max_cases is not None else UNAVAILABLE
                ),
            },
            "parallelism": {
                "configured_workers": workers,
                "selection_basis": (
                    "measured full-size case waves; automatic CPU/RAM-constrained "
                    "selection when prepare_workers=auto"
                ),
            },
            "device_resources": resources_before,
            "peak_vram": f"{UNAVAILABLE} (no GPU preparation kernels instrumented)",
        },
    )
    started = time.perf_counter()
    cpu_started = time.process_time()
    case_rows = prepare_hierarchical_cache(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        region_cache_dir=args.region_cache_dir,
        bank_path=args.prototype_bank,
        train_case_ids=split["train"],
        val_case_ids=split["val"],
        graph_config=graph,
        liver_label=int(labels["liver"]),
        tumor_label=int(labels["tumor"]),
        source_selection=str(cache["source_selection"]),
        source_pad=int(cache["source_pad"]),
        samples_per_case=int(cache["samples_per_case"]),
        total_candidates=int(cache["total_candidates"]),
        candidate_pool_size=int(cache["candidate_pool_size"]),
        easy_fraction=float(cache["easy_fraction"]),
        inter_fraction=float(cache["inter_fraction"]),
        intra_fraction=float(cache["intra_fraction"]),
        max_draws=int(cache["max_draws"]),
        min_liver_coverage=float(cache["min_liver_coverage"]),
        occupied_clearance_vox=int(cache["occupied_clearance_vox"]),
        min_center_separation_mm=float(cache["min_center_separation_mm"]),
        ct_clip=tuple(float(value) for value in config["ct_clip"]),
        seed=seed,
        max_cases=args.max_cases,
        overwrite=args.overwrite,
        workers=workers,
        run_mode=run_mode,
    )
    elapsed = time.perf_counter() - started
    cpu_elapsed = time.process_time() - cpu_started
    status_counts = Counter(str(row.get("status", "missing")) for row in case_rows)
    processed_case_ids = sorted(
        {
            str(row["case_id"])
            for row in case_rows
            if row.get("case_id") not in (None, "")
        }
    )
    processed_case_count = len(processed_case_ids)
    affinity = resources_before.get("cpu_affinity_cores")
    cpu_capacity = (
        int(affinity)
        if isinstance(affinity, int) and affinity > 0
        else (os.cpu_count() or 1)
    )
    _print_report(
        "PostRun",
        {
            "command": "prepare",
            "run_mode": run_mode,
            "status": "complete",
            "cache_publication_verified": True,
            "subset_active": args.max_cases is not None,
            "cache_dir": str(args.cache_dir),
            "configured_cases": configured_case_count,
            "selected_cases": selected_case_count,
            "cases_processed_this_invocation": processed_case_count,
            "processed_case_ids": processed_case_ids,
            "cases_per_second": (
                processed_case_count / elapsed
                if processed_case_count and elapsed > 0.0
                else f"{UNAVAILABLE} (complete cache reused without case rebuilds)"
            ),
            "case_results": len(case_rows),
            "status_counts": dict(status_counts),
            "elapsed_seconds": elapsed,
            "case_results_per_second": (
                len(case_rows) / elapsed
                if case_rows and elapsed > 0.0
                else f"{UNAVAILABLE} (no case results)"
            ),
            "process_cpu_seconds": cpu_elapsed,
            "process_cpu_utilization_percent_of_allocated_capacity": (
                cpu_elapsed / elapsed * 100.0 / cpu_capacity
                if elapsed > 0.0
                else UNAVAILABLE
            ),
            "process_memory": process_memory_snapshot(),
            "device_resources_after": collect_runtime_resources(
                storage_path=args.cache_dir
            ),
            "peak_vram": f"{UNAVAILABLE} (no GPU preparation kernels instrumented)",
            "configured_workers": workers,
            "worker_selection_measured": workers == "auto" and processed_case_count > 0,
        },
    )


def _validate_cache_metadata(
    cache_dir: str | Path,
    *,
    prototype_fingerprint: str,
    graph_config: dict,
    ct_clip: tuple[float, float],
) -> None:
    """Validate one lightweight metadata file instead of every graph payload."""

    from hiercp.cache import CACHE_FORMAT
    from hiercp.data import load_cache_config, load_cache_index

    metadata = load_cache_config(cache_dir)
    expected = {
        "format": CACHE_FORMAT,
        "prototype_fingerprint": prototype_fingerprint,
        "graph_config": graph_config,
        "ct_clip": [float(value) for value in ct_clip],
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError(
            "Hierarchical cache metadata is incompatible with the current request: "
            + ", ".join(mismatches)
        )
    index = load_cache_index(cache_dir)
    if index.get("prototype_fingerprint") != prototype_fingerprint:
        raise ValueError("Cache index prototype fingerprint does not match the bank")



def _finite_metric(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(
            f"Checkpoint selection metric {name!r} is non-finite ({value!r}); "
            "training is stopped and no sentinel or replacement score is written."
        )
    return value


def _checkpoint_selection_record(metrics: dict[str, float]) -> dict[str, float]:
    """Validate fixed-validation metrics stored in best/last checkpoints."""

    names = ("mrr", "acc", "margin", "ranking", "consistency")
    missing = [name for name in names if name not in metrics]
    if missing:
        raise KeyError(
            "Checkpoint selection metrics are incomplete; missing required finite "
            f"values: {missing}. No sentinel values will be substituted."
        )
    return {name: _finite_metric(metrics[name], name=name) for name in names}


def _checkpoint_selection_key(
    selection: dict[str, float],
    *,
    precision: int = 8,
) -> tuple[float, float, float, float, float]:
    """Lexicographic best-checkpoint key.

    MRR and top-1 remain primary. Exact ties are broken by a larger
    positive-versus-hardest-negative margin, then smaller fixed-validation
    ranking and view-consistency losses. Rounding prevents insignificant
    floating-point noise from replacing a genuinely equivalent checkpoint.
    """

    digits = max(0, int(precision))
    validated = _checkpoint_selection_record(selection)
    values = (
        validated["mrr"],
        validated["acc"],
        validated["margin"],
        -validated["ranking"],
        -validated["consistency"],
    )
    return tuple(round(float(value), digits) for value in values)


def _checkpoint_selection_is_better(
    candidate: dict[str, float],
    best: dict[str, float] | None,
    *,
    precision: int = 8,
) -> bool:
    return best is None or _checkpoint_selection_key(
        candidate, precision=precision
    ) > _checkpoint_selection_key(best, precision=precision)


def _loader_kwargs(
    *,
    workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": int(workers),
        "pin_memory": bool(pin_memory),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        kwargs["persistent_workers"] = bool(persistent_workers)
    return kwargs


def _fused_adamw_is_unavailable(exc: BaseException) -> bool:
    """Recognize only version/device capability failures for ``fused=True``."""

    message = str(exc).lower()
    if "fused" not in message:
        return False
    if isinstance(exc, TypeError):
        return "unexpected keyword argument" in message
    if isinstance(exc, RuntimeError):
        return any(
            marker in message
            for marker in ("not supported", "unsupported", "not implemented")
        )
    return False


def _create_adamw(
    *,
    torch_module,
    parameters: list[Any],
    optimizer_kwargs: dict[str, float],
    request_fused: bool,
    context: str,
) -> tuple[Any, bool]:
    if not parameters:
        raise RuntimeError(f"{context} has no trainable parameters")
    if not request_fused:
        return torch_module.optim.AdamW(parameters, **optimizer_kwargs), False
    try:
        return (
            torch_module.optim.AdamW(
                parameters,
                **optimizer_kwargs,
                fused=True,
            ),
            True,
        )
    except (TypeError, RuntimeError) as exc:
        message = str(exc).lower()
        cuda_oom_type = getattr(
            getattr(torch_module, "cuda", None), "OutOfMemoryError", None
        )
        typed_cuda_oom = (
            isinstance(cuda_oom_type, type) and isinstance(exc, cuda_oom_type)
        )
        textual_cuda_oom = "cuda" in message and "out of memory" in message
        if typed_cuda_oom or textual_cuda_oom:
            raise
        if not _fused_adamw_is_unavailable(exc):
            raise
        _print_report(
            "OptimizerCapabilityFallback",
            {
                "context": context,
                "requested": "AdamW(fused=True)",
                "selected": "AdamW(fused=False)",
                "reason": f"{type(exc).__name__}: {exc}",
            },
        )
        return torch_module.optim.AdamW(parameters, **optimizer_kwargs), False


def _measurement_candidates(
    settings: dict[str, Any],
    key: str,
    *,
    minimum: int,
) -> list[int]:
    raw = settings.get(key)
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError(
            f"{key} must be an explicit list with at least two measurement candidates "
            "when the corresponding setting is 'auto'."
        )
    candidates = sorted({int(value) for value in raw})
    if len(candidates) < 2:
        raise ValueError(
            f"{key} must contain at least two distinct measurement candidates: {raw}"
        )
    if any(value < minimum for value in candidates):
        raise ValueError(f"{key} values must all be >= {minimum}: {candidates}")
    return candidates


def _physical_batch_candidates(training: dict[str, Any], sample_count: int) -> list[int]:
    if training.get("batch_size_candidates") != "powers_of_two_to_cohort":
        return _measurement_candidates(training, "batch_size_candidates", minimum=1)
    if sample_count < 2:
        raise ValueError("Physical-batch calibration needs at least two real training samples")
    candidates = [1]
    while candidates[-1] * 2 < sample_count:
        candidates.append(candidates[-1] * 2)
    candidates.append(sample_count)
    return candidates


def _calibration_resource_fingerprint(
    resource_report: dict[str, Any],
    *,
    device: Any,
) -> dict[str, Any]:
    """Return stable allocation fields; exclude volatile free/used measurements."""

    fields = (
        "platform",
        "cpu_logical_cores",
        "cpu_affinity_cores",
        "ram_total_bytes",
        "cgroup_memory_limit_bytes",
        "cgroup_cpu_limit_cores",
        "cgroup_cpuset",
        "scheduler_allocation",
        "container_hint",
        "cuda_visible_devices_env",
        "nvidia_visible_devices_env",
        "cuda_available",
        "cuda_visible_device_count",
        "gpu_devices",
        "mig_visibility",
    )
    return {
        "selected_device": str(device),
        **{field: resource_report.get(field, UNAVAILABLE) for field in fields},
    }


def _resume_auto_values(
    resume_path: Path,
    *,
    torch_load,
    expected_identity: dict[str, Any],
    expected_resource_fingerprint: dict[str, Any],
) -> tuple[int | None, int | None, dict[str, Any] | None]:
    if not resume_path.is_file():
        return None, None, None
    payload = torch_load(resume_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Resumable training state is not a dictionary: {resume_path}")
    signature = payload.get("training_signature")
    if not isinstance(signature, dict):
        raise ValueError(
            f"Auto-calibration resume state has no training signature: {resume_path}"
        )
    batch = signature.get("batch_size")
    workers = signature.get("num_workers")
    calibration = payload.get("preflight_calibration")
    if not isinstance(calibration, dict):
        raise ValueError(
            f"Auto-calibration resume state has no calibration record: {resume_path}"
        )
    if calibration.get("format") != "hiercp_preflight_calibration_v2":
        raise ValueError(
            "Existing calibration predates allocation/identity validation and cannot "
            "be reused. Use --overwrite to measure again or a separate checkpoint path."
        )
    if calibration.get("identity") != expected_identity:
        raise ValueError(
            "Existing auto-calibration belongs to a different training identity; "
            "refusing silent reuse. Use --overwrite or a separate checkpoint path."
        )
    if calibration.get("resource_fingerprint") != expected_resource_fingerprint:
        raise ValueError(
            "Existing auto-calibration was measured on a different hardware/resource "
            "allocation; refusing silent reuse. Use --overwrite to remeasure explicitly."
        )
    if not isinstance(batch, int) or batch < 1:
        raise ValueError("Auto-calibration resume signature has no valid batch_size")
    if not isinstance(workers, int) or workers < 0:
        raise ValueError("Auto-calibration resume signature has no valid num_workers")
    return (
        int(batch),
        int(workers),
        calibration,
    )


def _measure_batch_candidates(
    *,
    torch_module,
    model,
    dataset_type,
    collate_fn,
    train_files: list[Path],
    candidates: list[int],
    repeats: int,
    max_vram_fraction: float,
    device,
    use_amp: bool,
    seed: int,
    optimizer_kwargs: dict[str, float],
    fused_optimizer: bool,
    trainable_parameters: list[Any],
    curriculum_config,
    consistency_weight: float,
) -> tuple[int, list[dict[str, Any]]]:
    from hiercp.loss import curriculum_ranking_loss
    from hiercp.tensor import (
        capture_rng_state,
        cuda_memory_snapshot,
        is_cuda_out_of_memory,
        restore_rng_state,
    )

    if repeats < 1:
        raise ValueError("training.batch_calibration_repeats must be positive")
    if not 0.0 < max_vram_fraction < 1.0:
        raise ValueError("training.batch_calibration_max_vram_fraction must be in (0,1)")
    if not train_files:
        raise RuntimeError("Batch calibration requires at least one training cache file")
    largest_first = sorted(
        train_files, key=lambda path: path.stat().st_size, reverse=True
    )
    rng_state = capture_rng_state()
    prior_training = bool(model.training)
    trials: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    first_oom: int | None = None
    try:
        model.train(True)
        for candidate in candidates:
            if candidate > len(largest_first):
                trials.append(
                    {
                        "batch_size": candidate,
                        "status": "not_measured",
                        "reason": f"only {len(largest_first)} training samples exist",
                    }
                )
                continue
            if first_oom is not None and candidate > first_oom:
                trials.append(
                    {
                        "batch_size": candidate,
                        "status": "not_measured",
                        "reason": f"larger than CUDA-OOM candidate {first_oom}",
                    }
                )
                continue
            elapsed = 0.0
            processed = 0
            trial_error: BaseException | None = None
            trial_diagnostics: dict[str, Any] | None = None
            for _ in range(repeats):
                restore_rng_state(rng_state)
                batch = None
                output = None
                loss = None
                calibration_optimizer = None
                try:
                    probe = dataset_type(
                        largest_first[:candidate], mmap=True, training=True, seed=seed
                    )
                    batch = collate_fn([probe[index] for index in range(candidate)])
                    if device.type == "cuda":
                        batch.pin_memory()
                    batch.to(device, non_blocking=device.type == "cuda")
                    model.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch_module.cuda.empty_cache()
                        torch_module.cuda.reset_peak_memory_stats(device)
                        torch_module.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch_module.autocast(
                        device_type=device.type, enabled=use_amp
                    ):
                        output = model(batch)
                        ranking, _ = curriculum_ranking_loss(
                            output.scores, batch.difficulty_list(),
                            epoch=max(curriculum_config.model_mine_start_epoch,
                                      curriculum_config.intra_epochs + 1),
                            config=curriculum_config,
                        )
                        loss = ranking + consistency_weight * output.consistency
                    if not bool(torch_module.isfinite(loss)):
                        raise FloatingPointError("Non-finite actual ranking loss during batch calibration")
                    loss.backward()
                    finite = [torch_module.isfinite(p.grad).all() for p in trainable_parameters
                              if p.grad is not None]
                    if not finite or not bool(torch_module.stack(finite).all()):
                        raise FloatingPointError("Non-finite or absent gradients during batch calibration")
                    calibration_optimizer_kwargs = dict(optimizer_kwargs)
                    calibration_optimizer_kwargs["lr"] = 0.0
                    calibration_optimizer, calibration_fused = _create_adamw(
                        torch_module=torch_module,
                        parameters=trainable_parameters,
                        optimizer_kwargs=calibration_optimizer_kwargs,
                        request_fused=fused_optimizer,
                        context="physical-batch preflight calibration",
                    )
                    if calibration_fused != fused_optimizer:
                        raise RuntimeError(
                            "Calibration optimizer capability changed after the training "
                            "optimizer was selected"
                        )
                    calibration_optimizer.step()
                    if device.type == "cuda":
                        torch_module.cuda.synchronize(device)
                    elapsed += time.perf_counter() - started
                    processed += candidate
                except Exception as exc:
                    if not is_cuda_out_of_memory(exc):
                        raise
                    trial_error = exc
                    trial_diagnostics = cuda_memory_snapshot(device)
                    first_oom = candidate
                    break
                finally:
                    model.zero_grad(set_to_none=True)
                    del calibration_optimizer, loss, output, batch
                    if "ranking" in locals():
                        del ranking
                    gc.collect()
                    if device.type == "cuda":
                        torch_module.cuda.empty_cache()
            if trial_error is not None:
                after_cleanup = cuda_memory_snapshot(device)
                trial = {
                    "batch_size": candidate,
                    "status": "cuda_oom",
                    "error": f"{type(trial_error).__name__}: {trial_error}",
                    "at_failure": trial_diagnostics,
                    "after_cleanup": after_cleanup,
                    "automatic_model_graph_data_reduction": False,
                }
                trials.append(trial)
                _print_report("CalibrationCUDAOutOfMemory", trial)
                continue
            peak_bytes: int | str = UNAVAILABLE
            total_bytes: int | str = UNAVAILABLE
            vram_fraction: float | str = UNAVAILABLE
            memory_safe = True
            if device.type == "cuda":
                snapshot = cuda_memory_snapshot(device)
                peak_bytes = int(snapshot["cuda_peak_allocated_bytes"])
                total_value = snapshot.get("cuda_total_bytes")
                if isinstance(total_value, int) and total_value > 0:
                    total_bytes = total_value
                    vram_fraction = peak_bytes / total_value
                    memory_safe = vram_fraction <= max_vram_fraction
                else:
                    memory_safe = False
            throughput = processed / elapsed if elapsed > 0.0 else 0.0
            trial = {
                "batch_size": candidate,
                "status": "accepted" if memory_safe else "rejected_vram_headroom",
                "samples": processed,
                "elapsed_seconds": elapsed,
                "samples_per_second": throughput,
                "peak_vram_bytes": peak_bytes,
                "total_vram_bytes": total_bytes,
                "peak_vram_fraction": vram_fraction,
                "max_vram_fraction": max_vram_fraction,
            }
            trials.append(trial)
            if memory_safe:
                accepted.append(trial)
    finally:
        model.train(prior_training)
        restore_rng_state(rng_state)
        model.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == "cuda":
            torch_module.cuda.empty_cache()
    if not accepted:
        raise RuntimeError(
            "No physical batch candidate passed measured memory/throughput preflight. "
            f"Trials: {trials}"
        )
    selected = max(
        accepted, key=lambda item: (float(item["samples_per_second"]), int(item["batch_size"]))
    )
    return int(selected["batch_size"]), trials


def _measure_worker_candidates(
    *,
    data_loader_type,
    dataset_type,
    collate_fn,
    train_files: list[Path],
    batch_size: int,
    candidates: list[int],
    measurement_batches: int,
    pin_memory: bool,
    prefetch_factor: int,
    seed: int,
) -> tuple[int, list[dict[str, Any]]]:
    if measurement_batches < 1:
        raise ValueError("training.loader_calibration_batches must be positive")
    trials: list[dict[str, Any]] = []
    for workers in candidates:
        dataset = dataset_type(train_files, mmap=True, training=False, seed=seed)
        loader = data_loader_type(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **_loader_kwargs(
                workers=workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=False,
            ),
        )
        started = time.perf_counter()
        samples = 0
        batches = 0
        try:
            for batch in loader:
                samples += int(batch.sample_count)
                batches += 1
                if batches >= measurement_batches:
                    break
        except Exception as exc:
            trials.append(
                {
                    "num_workers": workers,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        finally:
            del loader, dataset
            gc.collect()
        elapsed = time.perf_counter() - started
        trials.append(
            {
                "num_workers": workers,
                "status": "accepted",
                "batches": batches,
                "samples": samples,
                "elapsed_seconds": elapsed,
                "samples_per_second": samples / elapsed if elapsed > 0.0 else 0.0,
                "persistent_workers_during_measurement": False,
            }
        )
    accepted = [trial for trial in trials if trial["status"] == "accepted"]
    if not accepted:
        raise RuntimeError(f"No DataLoader worker candidate completed preflight: {trials}")
    selected = max(
        accepted,
        key=lambda item: (float(item["samples_per_second"]), int(item["num_workers"])),
    )
    return int(selected["num_workers"]), trials


def run_train(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader, RandomSampler

    from hiercp.data import (
        CudaPrefetchLoader,
        HierarchicalCacheDataset,
        collate_samples,
        list_cache_files,
        split_files_from_cache,
        summarize_cache_usage,
        summarize_hierarchical_batch,
    )
    from hiercp.loss import (
        CurriculumConfig,
        curriculum_ranking_loss,
        ranking_metric_sums,
    )
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.tensor import (
        capture_rng_state,
        collect_runtime_resources,
        configure_runtime,
        cuda_memory_snapshot,
        enforce_single_device_execution,
        is_cuda_out_of_memory,
        process_memory_snapshot,
        raise_cuda_out_of_memory,
        resolve_device,
        restore_rng_state,
        save_checkpoint_atomic,
        set_seed,
        torch_load_compat,
        training_state_path,
    )

    run_mode = _guard_reduction_overrides(
        args, option_names=("epochs", "batch_size", "num_workers")
    )
    config = _load_json(args.config)
    training = config["training"]
    runtime = config.get("runtime", {})
    model_config = config["model"]
    ablation_mode = str(getattr(args, "ablation_mode", "full"))
    if ablation_mode != "full" and run_mode != "ablation":
        raise ValueError(
            f"--ablation-mode {ablation_mode} requires --run-mode ablation; "
            "production/benchmark/debug must not be mislabeled."
        )
    graph_config = graph_config_from_dict(config["graph"])
    _require_full_graph(graph_config, "full-graph training config")
    seed = int(args.seed if args.seed is not None else config["seed"])
    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    batch_setting = (
        args.batch_size if args.batch_size is not None else training["batch_size"]
    )
    worker_setting = (
        args.num_workers if args.num_workers is not None else training["num_workers"]
    )
    batch_auto = isinstance(batch_setting, str) and batch_setting.lower() == "auto"
    workers_auto = isinstance(worker_setting, str) and worker_setting.lower() == "auto"
    if isinstance(batch_setting, str) and not batch_auto:
        raise ValueError("training.batch_size must be a positive integer or 'auto'")
    if isinstance(worker_setting, str) and not workers_auto:
        raise ValueError("training.num_workers must be a non-negative integer or 'auto'")
    batch_size = None if batch_auto else int(batch_setting)
    workers = None if workers_auto else int(worker_setting)
    if epochs < 1 or (batch_size is not None and batch_size < 1):
        raise ValueError("epochs and explicit batch_size must be positive")
    if workers is not None and workers < 0:
        raise ValueError(
            "num_workers must be non-negative; use the explicit string 'auto' for measurement"
        )
    accumulation_setting = training.get("gradient_accumulation_steps", 1)
    accumulation_auto = (
        isinstance(accumulation_setting, str)
        and accumulation_setting.lower() == "auto"
    )
    if isinstance(accumulation_setting, str) and not accumulation_auto:
        raise ValueError(
            "training.gradient_accumulation_steps must be a positive integer or 'auto'"
        )
    accumulation_steps = None if accumulation_auto else int(accumulation_setting)
    if accumulation_steps is not None and accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be positive")
    target_setting = training.get("target_effective_batch_size")
    if accumulation_auto and target_setting is None:
        raise ValueError(
            "training.gradient_accumulation_steps='auto' requires the explicit "
            "training.target_effective_batch_size"
        )
    target_effective_batch_size = (
        None if target_setting is None else int(target_setting)
    )
    if target_effective_batch_size is not None and target_effective_batch_size < 1:
        raise ValueError("training.target_effective_batch_size must be positive")

    checkpoint_path = Path(args.checkpoint)
    resume_path = training_state_path(checkpoint_path)
    if args.overwrite:
        for path in (checkpoint_path, resume_path):
            if path.exists():
                path.unlink()

    files = list_cache_files(args.cache_dir)
    bank = PrototypeBank.load(args.prototype_bank)
    expected_fingerprint = bank.fingerprint()
    expected_graph_config = graph_config.to_dict()
    expected_ct_clip = tuple(float(value) for value in config["ct_clip"])
    _validate_cache_metadata(
        args.cache_dir,
        prototype_fingerprint=expected_fingerprint,
        graph_config=expected_graph_config,
        ct_clip=expected_ct_clip,
    )
    train_files, val_files = split_files_from_cache(files)
    cache_usage = summarize_cache_usage(args.cache_dir, selected_files=files)
    materialized_ratio = cache_usage["materialized_sample_ratio"]
    if run_mode == "production" and (
        cache_usage["subset_active"]
        or not isinstance(materialized_ratio, float)
        or not math.isclose(materialized_ratio, 1.0, rel_tol=0.0, abs_tol=0.0)
    ):
        raise RuntimeError(
            "Production training requires the complete configured cache cohort and all "
            f"expected samples; cache_usage={cache_usage}"
        )
    if run_mode == "production" and not val_files:
        raise RuntimeError(
            "Production training requires a non-empty fixed validation split; refusing "
            "to select a best checkpoint from training metrics."
        )

    deterministic = bool(runtime.get("deterministic", True))
    set_seed(seed, deterministic=deterministic)
    configure_runtime(
        deterministic=deterministic,
        allow_tf32=bool(runtime.get("allow_tf32", False)),
        cudnn_benchmark=bool(runtime.get("cudnn_benchmark", not deterministic)),
    )
    device = resolve_device(args.device)
    if run_mode == "production":
        enforce_single_device_execution(device, context="hiercp.pipeline train")
    use_amp = bool(training["amp"] and device.type == "cuda")
    model_kwargs = {
        "hidden_dim": int(model_config["hidden_dim"]),
        "heads": int(model_config["heads"]),
        "local_layers": int(model_config["local_layers"]),
        "patient_layers": int(model_config["patient_layers"]),
        "prototype_layers": int(model_config["prototype_layers"]),
        "dropout": float(model_config["dropout"]),
        "dense_base_channels": int(model_config["dense_base_channels"]),
        "dense_feature_dim": int(model_config["dense_feature_dim"]),
        "dense_batch_size": int(model_config.get("dense_batch_size", 8)),
        "channels_last_3d": bool(model_config.get("channels_last_3d", True)),
        "checkpoint_local_blocks": bool(
            model_config.get("checkpoint_local_blocks", True)
        ),
        "checkpoint_dense_encoder": bool(
            model_config.get("checkpoint_dense_encoder", True)
        ),
    }
    # Keep full-mode model_kwargs byte-compatible with pre-ablation M3
    # checkpoints. Ablation checkpoints explicitly record their mode.
    if ablation_mode != "full":
        model_kwargs["ablation_mode"] = ablation_mode

    model = HierarchicalPyGPlacementModel(**model_kwargs).to(device)
    trainable_parameters = list(model.trainable_parameters())
    trainable_named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("Model exposes no trainable parameters")
    if {id(parameter) for parameter in trainable_parameters} != {
        id(parameter) for _, parameter in trainable_named_parameters
    }:
        raise RuntimeError(
            "model.trainable_parameters() disagrees with named requires_grad parameters"
        )
    optimizer_kwargs = {
        "lr": float(training["lr"]),
        "weight_decay": float(training["weight_decay"]),
    }
    request_fused_optimizer = bool(
        device.type == "cuda" and training.get("fused_optimizer", True)
    )
    optimizer, fused_optimizer = _create_adamw(
        torch_module=torch,
        parameters=trainable_parameters,
        optimizer_kwargs=optimizer_kwargs,
        request_fused=request_fused_optimizer,
        context="training",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    curriculum = CurriculumConfig(
        easy_epochs=int(training["easy_epochs"]),
        inter_epochs=int(training["inter_epochs"]),
        intra_epochs=int(training["intra_epochs"]),
        model_mine_start_epoch=int(training["model_mine_start_epoch"]),
        semi_hard_low_percentile=float(training["semi_hard_low_percentile"]),
        semi_hard_high_percentile=float(training["semi_hard_high_percentile"]),
        cross_entropy_weight=float(training["cross_entropy_weight"]),
        pairwise_weight=float(training["pairwise_weight"]),
        ordinal_weight=float(training["ordinal_weight"]),
        mined_weight=float(training["mined_weight"]),
    )
    curriculum.validate()

    resource_report = collect_runtime_resources(device, storage_path=args.cache_dir)
    resource_fingerprint = _calibration_resource_fingerprint(
        resource_report,
        device=device,
    )
    calibration_identity = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "run_mode": run_mode,
        "seed": seed,
        "prototype_fingerprint": expected_fingerprint,
        "graph_config": expected_graph_config,
        "model_kwargs": model_kwargs,
        "cache_publication": {
            "config_sha256": _sha256_file(Path(args.cache_dir) / "config.json"),
            "index_sha256": _sha256_file(Path(args.cache_dir) / "index.json"),
            "complete_sha256": _sha256_file(Path(args.cache_dir) / "complete.json"),
        },
        "optimizer": {
            **optimizer_kwargs,
            "fused": fused_optimizer,
        },
        "train_cache_files": [path.name for path in train_files],
        "val_cache_files": [path.name for path in val_files],
        "batch_setting": batch_setting,
        "worker_setting": worker_setting,
        "gradient_accumulation_setting": accumulation_setting,
        "target_effective_batch_size": target_effective_batch_size,
        "batch_size_candidates": training.get("batch_size_candidates", UNAVAILABLE),
        "batch_calibration_repeats": training.get(
            "batch_calibration_repeats", UNAVAILABLE
        ),
        "batch_calibration_max_vram_fraction": training.get(
            "batch_calibration_max_vram_fraction", UNAVAILABLE
        ),
        "num_worker_candidates": training.get("num_worker_candidates", UNAVAILABLE),
        "loader_calibration_batches": training.get(
            "loader_calibration_batches", UNAVAILABLE
        ),
        "prefetch_factor": training.get("prefetch_factor", 2),
        "pin_memory": training.get("pin_memory", True),
    }
    preflight_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".preflight.json")
    resumed_batch, resumed_workers, resumed_calibration = (None, None, None)
    calibration_reused = False
    if (batch_auto or workers_auto) and not args.overwrite:
        if preflight_path.exists() and not resume_path.is_file():
            raise FileExistsError(
                f"Preflight record already exists without its resumable training state: "
                f"{preflight_path}. Refusing to overwrite it without --overwrite."
            )
        resumed_batch, resumed_workers, resumed_calibration = _resume_auto_values(
            resume_path,
            torch_load=torch_load_compat,
            expected_identity=calibration_identity,
            expected_resource_fingerprint=resource_fingerprint,
        )
    preflight_calibration: dict[str, Any] = {
        "format": "hiercp_preflight_calibration_v2",
        "identity": calibration_identity,
        "resource_fingerprint": resource_fingerprint,
        "run_mode": run_mode,
        "batch_setting": batch_setting,
        "worker_setting": worker_setting,
        "gradient_accumulation_setting": accumulation_setting,
        "target_effective_batch_size": target_effective_batch_size,
        "batch_trials": [],
        "worker_trials": [],
        "reused_from_resume": False,
    }
    if batch_auto:
        if resumed_batch is not None:
            batch_size = resumed_batch
            if resumed_calibration is None:
                raise RuntimeError("Resume selected a batch size without a calibration record")
            preflight_calibration = resumed_calibration
            calibration_reused = True
        else:
            batch_candidates = _physical_batch_candidates(training, len(train_files))
            if accumulation_auto:
                if target_effective_batch_size is None:
                    raise RuntimeError(
                        "Internal validation lost target_effective_batch_size"
                    )
                configured_batch_candidates = list(batch_candidates)
                batch_candidates = [
                    candidate
                    for candidate in configured_batch_candidates
                    if candidate <= target_effective_batch_size
                    and target_effective_batch_size % candidate == 0
                ]
                rejected_candidates = [
                    candidate
                    for candidate in configured_batch_candidates
                    if candidate not in batch_candidates
                ]
                preflight_calibration["effective_batch_candidate_filter"] = {
                    "target_effective_batch_size": target_effective_batch_size,
                    "eligible_divisors": batch_candidates,
                    "rejected_nondivisors_or_above_target": rejected_candidates,
                }
                if len(batch_candidates) < 2:
                    raise ValueError(
                        "Auto physical-batch calibration requires at least two configured "
                        "batch_size_candidates that are divisors of and no larger than "
                        f"target_effective_batch_size={target_effective_batch_size}; "
                        f"configured={configured_batch_candidates}, eligible={batch_candidates}"
                    )
            required_batch_plan = (
                "batch_calibration_repeats",
                "batch_calibration_max_vram_fraction",
            )
            missing_plan = [key for key in required_batch_plan if key not in training]
            if missing_plan:
                raise ValueError(
                    "training.batch_size='auto' requires an explicit measurement plan; "
                    f"missing {missing_plan}"
                )
            batch_size, trials = _measure_batch_candidates(
                torch_module=torch,
                model=model,
                dataset_type=HierarchicalCacheDataset,
                collate_fn=collate_samples,
                train_files=train_files,
                candidates=batch_candidates,
                repeats=int(training["batch_calibration_repeats"]),
                max_vram_fraction=float(
                    training["batch_calibration_max_vram_fraction"]
                ),
                device=device,
                use_amp=use_amp,
                seed=seed,
                optimizer_kwargs=optimizer_kwargs,
                fused_optimizer=fused_optimizer,
                trainable_parameters=trainable_parameters,
                curriculum_config=curriculum,
                consistency_weight=float(training["consistency_weight"]),
            )
            preflight_calibration["batch_trials"] = trials
            preflight_calibration["selected_batch_size"] = batch_size
    if batch_size is None:
        raise RuntimeError("Physical batch-size resolution produced no value")
    if accumulation_auto:
        if target_effective_batch_size is None:
            raise RuntimeError("Auto accumulation lost its required effective-batch target")
        if batch_size > target_effective_batch_size or (
            target_effective_batch_size % batch_size != 0
        ):
            raise ValueError(
                "Selected/resumed physical batch does not exactly divide the requested "
                f"effective batch: physical={batch_size}, "
                f"target={target_effective_batch_size}"
            )
        accumulation_steps = target_effective_batch_size // batch_size
    if accumulation_steps is None:
        raise RuntimeError("Gradient-accumulation resolution produced no value")
    resolved_effective_batch_size = int(batch_size) * int(accumulation_steps)
    if (
        target_effective_batch_size is not None
        and resolved_effective_batch_size != target_effective_batch_size
    ):
        raise ValueError(
            "physical_batch_size * gradient_accumulation_steps does not preserve "
            f"target_effective_batch_size: {batch_size} * {accumulation_steps} != "
            f"{target_effective_batch_size}"
        )

    pin_memory = device.type == "cuda" and bool(training.get("pin_memory", True))
    prefetch_factor = int(training.get("prefetch_factor", 2))
    if workers_auto:
        if resumed_workers is not None:
            workers = resumed_workers
            if resumed_calibration is None:
                raise RuntimeError("Resume selected workers without a calibration record")
            preflight_calibration = resumed_calibration
            calibration_reused = True
        else:
            worker_candidates = _measurement_candidates(
                training, "num_worker_candidates", minimum=0
            )
            if "loader_calibration_batches" not in training:
                raise ValueError(
                    "training.num_workers='auto' requires loader_calibration_batches"
                )
            workers, trials = _measure_worker_candidates(
                data_loader_type=DataLoader,
                dataset_type=HierarchicalCacheDataset,
                collate_fn=collate_samples,
                train_files=train_files,
                batch_size=batch_size,
                candidates=worker_candidates,
                measurement_batches=int(training["loader_calibration_batches"]),
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                seed=seed,
            )
            preflight_calibration["worker_trials"] = trials
            preflight_calibration["selected_num_workers"] = workers
    if workers is None:
        raise RuntimeError("DataLoader worker calibration produced no value")
    if batch_auto or workers_auto:
        preflight_calibration["selected_batch_size"] = batch_size
        preflight_calibration["selected_num_workers"] = workers
        if preflight_path.exists() and not args.overwrite:
            existing_preflight = _load_json(preflight_path)
            if existing_preflight != preflight_calibration:
                raise ValueError(
                    "Existing preflight JSON differs from the resume-authorized "
                    "calibration. Refusing to overwrite it without --overwrite."
                )
        else:
            _write_json_atomic(preflight_path, preflight_calibration)
        _print_report(
            "PreflightCalibration",
            {
                **preflight_calibration,
                "reused_this_invocation": calibration_reused,
                "record_path": str(preflight_path),
            },
        )

    loader_options = _loader_kwargs(
        workers=workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=bool(training.get("persistent_workers", True)),
    )

    # Use dedicated generators so resume restores the next epoch's shuffle
    # independently of worker startup, validation iteration, or global RNG use.
    train_shuffle_generator = torch.Generator()
    train_shuffle_generator.manual_seed(seed + 2003)
    train_worker_generator = torch.Generator()
    train_worker_generator.manual_seed(seed + 3001)
    val_worker_generator = torch.Generator()
    val_worker_generator.manual_seed(seed + 4001)

    train_dataset = HierarchicalCacheDataset(
        train_files, mmap=True, training=True, seed=seed
    )
    train_sampler = RandomSampler(train_dataset, generator=train_shuffle_generator)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        generator=train_worker_generator,
        collate_fn=collate_samples,
        **loader_options,
    )
    val_loader = (
        DataLoader(
            HierarchicalCacheDataset(
                val_files, mmap=True, training=False, seed=seed
            ),
            batch_size=batch_size,
            shuffle=False,
            generator=val_worker_generator,
            collate_fn=collate_samples,
            **loader_options,
        )
        if val_files
        else None
    )
    cuda_prefetch = bool(
        device.type == "cuda" and training.get("cuda_prefetch", True)
    )
    train_epoch_loader = (
        CudaPrefetchLoader(train_loader, device) if cuda_prefetch else train_loader
    )
    val_epoch_loader = (
        CudaPrefetchLoader(val_loader, device)
        if cuda_prefetch and val_loader is not None
        else val_loader
    )

    consistency_weight = float(training.get("consistency_weight", 0.1))
    if consistency_weight < 0.0:
        raise ValueError("training.consistency_weight must be non-negative")
    fixed_validation_epoch = int(
        training.get(
            "fixed_validation_epoch",
            max(curriculum.model_mine_start_epoch, curriculum.intra_epochs + 1),
        )
    )
    if fixed_validation_epoch < curriculum.model_mine_start_epoch:
        raise ValueError(
            "training.fixed_validation_epoch must activate the complete curriculum "
            f"(>= {curriculum.model_mine_start_epoch})"
        )
    checkpoint_metric_precision = int(
        training.get("checkpoint_metric_precision", 8)
    )
    if checkpoint_metric_precision < 0 or checkpoint_metric_precision > 12:
        raise ValueError("training.checkpoint_metric_precision must be in [0,12]")
    training_signature = {
        "format": "hiercp_training_signature_v1",
        "run_mode": run_mode,
        "target_epochs": int(epochs),
        "seed": int(seed),
        "batch_setting": batch_setting,
        "batch_size": int(batch_size),
        "worker_setting": worker_setting,
        "num_workers": int(workers),
        "gradient_accumulation_setting": accumulation_setting,
        "gradient_accumulation_steps": int(accumulation_steps),
        "target_effective_batch_size": target_effective_batch_size,
        "resolved_effective_batch_size": resolved_effective_batch_size,
        "calibration_resource_fingerprint": resource_fingerprint,
        "consistency_weight": float(consistency_weight),
        "optimizer": {
            "name": "AdamW",
            "lr": float(training["lr"]),
            "weight_decay": float(training["weight_decay"]),
            "fused": bool(fused_optimizer),
        },
        "scheduler": {"name": "CosineAnnealingLR", "t_max": int(epochs)},
        "amp": bool(use_amp),
        "grad_clip": float(training["grad_clip"]),
        "deterministic": bool(deterministic),
        "allow_tf32": bool(runtime.get("allow_tf32", False)),
        "curriculum": {
            "easy_epochs": curriculum.easy_epochs,
            "inter_epochs": curriculum.inter_epochs,
            "intra_epochs": curriculum.intra_epochs,
            "model_mine_start_epoch": curriculum.model_mine_start_epoch,
            "semi_hard_low_percentile": curriculum.semi_hard_low_percentile,
            "semi_hard_high_percentile": curriculum.semi_hard_high_percentile,
            "cross_entropy_weight": curriculum.cross_entropy_weight,
            "pairwise_weight": curriculum.pairwise_weight,
            "ordinal_weight": curriculum.ordinal_weight,
            "mined_weight": curriculum.mined_weight,
        },
        "train_cache_files": [path.name for path in train_files],
        "val_cache_files": [path.name for path in val_files],
    }
    if ablation_mode != "full":
        training_signature["ablation_mode"] = ablation_mode
    validation_policy = {
        "format": "hiercp_fixed_validation_v1",
        "epoch": int(fixed_validation_epoch),
        "checkpoint_order": [
            "mrr",
            "acc",
            "margin",
            "-ranking",
            "-consistency",
        ],
        "metric_precision": int(checkpoint_metric_precision),
    }
    static_checkpoint_metadata = {
        "method": CHECKPOINT_METHOD,
        "architecture_version": model.architecture_version,
        "geometry_contract": expected_graph_config["geometry_contract"],
        "cache_publication": dict(calibration_identity["cache_publication"]),
        "framework": "torch_geometric",
        "upper_feature_policy": UPPER_FEATURE_POLICY,
        "model_kwargs": model_kwargs,
        "graph_config": expected_graph_config,
        "ct_clip": expected_ct_clip,
        "prototype_training_cases": list(bank.training_case_ids),
        "prototype_fingerprint": expected_fingerprint,
    }
    if cache_usage.get("donor_contract_sha256") is not None:
        static_checkpoint_metadata.update(
            donor_contract_sha256=cache_usage["donor_contract_sha256"],
            donor_case_ids=cache_usage["eligible_source_case_ids"],
            source_patient_case_ids=cache_usage["source_patient_case_ids"],
        )

    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable_parameter_count = sum(
        int(parameter.numel()) for parameter in trainable_parameters
    )
    probe_path = max(train_files, key=lambda path: path.stat().st_size)
    probe_dataset = HierarchicalCacheDataset(
        [probe_path], mmap=True, training=False, seed=seed
    )
    probe_batch = collate_samples([probe_dataset[0]])
    graph_statistics = summarize_hierarchical_batch(probe_batch)
    graph_statistics["scope"] = "largest_training_cache_file_by_bytes"
    graph_statistics["cache_file"] = probe_path.name
    del probe_batch, probe_dataset
    gc.collect()
    data_parallel_workers = 1
    pre_run_report = {
        "command": "train",
        "run_mode": run_mode,
        "model": {
            "name": type(model).__name__,
            "configuration": model_kwargs,
            "local_layers": model_kwargs["local_layers"],
            "patient_layers": model_kwargs["patient_layers"],
            "prototype_layers": model_kwargs["prototype_layers"],
            "hidden_dimension": model_kwargs["hidden_dim"],
            "dense_base_channels": model_kwargs["dense_base_channels"],
            "dense_feature_dimension": model_kwargs["dense_feature_dim"],
            "attention_heads": model_kwargs["heads"],
            "total_parameters": parameter_count,
            "trainable_parameters": trainable_parameter_count,
        },
        "graph_and_input": {
            "representative_statistics": graph_statistics,
            "input_resolution": [graph_config.patch_size] * 3,
            "sample_hops": graph_config.sample_hops,
            "sampling_ratio": f"{UNAVAILABLE} (not defined by cached graph contract)",
            "time_window": f"{UNAVAILABLE} (not a temporal model)",
        },
        "dataset": cache_usage,
        "optimization": {
            "physical_batch_size": batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "data_parallel_workers": data_parallel_workers,
            "effective_batch_size": batch_size
            * accumulation_steps
            * data_parallel_workers,
            "epochs": epochs,
            "batches_per_epoch": len(train_loader),
            "optimizer_steps_per_epoch": math.ceil(
                len(train_loader) / accumulation_steps
            ),
            "total_optimizer_steps": epochs
            * math.ceil(len(train_loader) / accumulation_steps),
            "precision": "AMP autocast" if use_amp else "float32",
            "allow_tf32": bool(runtime.get("allow_tf32", False)),
        },
        "loader": {
            "num_workers": workers,
            "persistent_workers": bool(
                workers and training.get("persistent_workers", True)
            ),
            "prefetch_factor": prefetch_factor if workers > 0 else UNAVAILABLE,
            "pin_memory": pin_memory,
            "cuda_prefetch": cuda_prefetch,
            "cache": "file-backed torch.load mmap",
        },
        "flags": {
            "debug": run_mode == "debug",
            "benchmark": run_mode == "benchmark",
            "ablation": run_mode == "ablation",
            "subset": bool(cache_usage["subset_active"]),
            "fast_mode": False,
            "cli_overrides": {
                "epochs": args.epochs is not None,
                "batch_size": args.batch_size is not None,
                "num_workers": args.num_workers is not None,
            },
        },
        "device_resources": resource_report,
        "parallelism": {
            "model_data_parallel": False,
            "single_device_limit": True,
            "visible_cuda_devices": resource_report["cuda_visible_device_count"],
        },
        "unavailable_metrics_before_execution": [
            "measured throughput",
            "step timing",
            "peak training VRAM",
            "process CPU utilization",
        ],
    }
    _print_report("PreRun", pre_run_report)

    expected_gradient_parameter_names = {
        name for name, _ in trainable_named_parameters
    }
    gradient_connected_parameter_names: set[str] = set()

    def gradient_connectivity_record() -> dict[str, Any]:
        connected = sorted(gradient_connected_parameter_names)
        missing = sorted(
            expected_gradient_parameter_names - gradient_connected_parameter_names
        )
        return {
            "format": "hiercp_gradient_connectivity_v1",
            "expected_parameter_count": len(expected_gradient_parameter_names),
            "connected_parameter_count": len(connected),
            "connected_parameters": connected,
            "missing_parameters": missing,
            "verified": not missing,
        }

    def restore_gradient_connectivity(
        payload: dict[str, Any],
        *,
        source: Path,
        require_verified: bool,
    ) -> None:
        record = payload.get("gradient_connectivity")
        if not isinstance(record, dict):
            raise ValueError(
                f"Checkpoint {source} has no gradient-connectivity record; refusing "
                "to treat it as a verified final training artifact."
            )
        connected = record.get("connected_parameters")
        if not isinstance(connected, list) or not all(
            isinstance(name, str) for name in connected
        ):
            raise ValueError(f"Checkpoint {source} has an invalid connectivity record")
        unknown = sorted(set(connected) - expected_gradient_parameter_names)
        if unknown:
            raise ValueError(
                f"Checkpoint {source} connectivity references unknown parameters: {unknown}"
            )
        gradient_connected_parameter_names.update(connected)
        restored = gradient_connectivity_record()
        if require_verified and not restored["verified"]:
            raise RuntimeError(
                f"Completed checkpoint {source} did not verify gradient connectivity: "
                f"missing={restored['missing_parameters']}"
            )

    def validate_static_checkpoint(payload: dict[str, Any], *, source: Path) -> None:
        mismatches = [
            key
            for key, expected in static_checkpoint_metadata.items()
            if payload.get(key) != expected
        ]
        if set(payload.get("prototype_training_cases", ())) != set(bank.training_case_ids):
            mismatches.append("prototype_training_cases")
        if mismatches:
            raise ValueError(
                f"Checkpoint {source} is incompatible with the current cache/config/bank: "
                + ", ".join(sorted(set(mismatches)))
                + ". Use --overwrite to restart training or use a separate workspace."
            )

    resume_payload: dict[str, Any] | None = None
    if resume_path.is_file() and not args.overwrite:
        loaded = torch_load_compat(resume_path, map_location=device)
        if not isinstance(loaded, dict) or loaded.get("format") != "hiercp_training_state_v1":
            raise ValueError(f"Unsupported resumable training state: {resume_path}")
        validate_static_checkpoint(loaded, source=resume_path)
        if loaded.get("training_signature") != training_signature:
            raise ValueError(
                "Resumable training-state settings differ from the current request. "
                "Keep the same config, seed, total --epochs, batch size, cache, and "
                "CUDA optimizer mode; otherwise use --overwrite or another workspace."
            )
        completed_epoch = int(loaded.get("epoch", 0))
        if completed_epoch >= epochs or bool(loaded.get("training_complete", False)):
            restore_gradient_connectivity(
                loaded,
                source=resume_path,
                require_verified=True,
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Training state is complete but best checkpoint is missing: {checkpoint_path}"
                )
            print(
                f"[Skip] Training already complete at epoch {completed_epoch}/{epochs}: "
                f"{resume_path}"
            )
            _print_report(
                "PostRun",
                {
                    "command": "train",
                    "run_mode": run_mode,
                    "status": "already_complete",
                    "completed_epoch": completed_epoch,
                    "target_epochs": epochs,
                    "training_executed_this_invocation": False,
                    "full_training_complete": True,
                    "full_evaluation_complete": False,
                },
            )
            return
        resume_payload = loaded
    elif checkpoint_path.is_file() and not args.overwrite:
        existing = torch_load_compat(checkpoint_path, map_location="cpu")
        if not isinstance(existing, dict):
            raise ValueError(f"Checkpoint payload is invalid: {checkpoint_path}")
        validate_static_checkpoint(existing, source=checkpoint_path)
        if existing.get("training_signature") != training_signature:
            raise ValueError(
                "Existing completed-checkpoint training identity differs from the "
                "current seed, optimization, runtime, curriculum, cache, or run-mode "
                "request. Refusing silent reuse without an exact training signature; "
                "use --overwrite or a separate checkpoint path."
            )
        if bool(existing.get("training_complete", False)) and int(
            existing.get("target_epochs", -1)
        ) == epochs:
            restore_gradient_connectivity(
                existing,
                source=checkpoint_path,
                require_verified=True,
            )
            print(f"[Skip] Compatible completed checkpoint exists: {checkpoint_path}")
            _print_report(
                "PostRun",
                {
                    "command": "train",
                    "run_mode": run_mode,
                    "status": "already_complete",
                    "target_epochs": epochs,
                    "training_executed_this_invocation": False,
                    "full_training_complete": True,
                    "full_evaluation_complete": False,
                },
            )
            return
        raise ValueError(
            "A best-model checkpoint exists, but its resumable last-epoch sidecar is "
            f"missing: {resume_path}. This is a pre-resume/v2 checkpoint and cannot "
            "restore optimizer, scheduler, AMP scaler, or RNG state exactly. To preserve "
            "training semantics, restart only the training stage once with --overwrite; "
            "the prototype and graph caches will still be reused."
        )

    start_epoch = 1
    best_mrr: float | None = None
    best_epoch = 0
    best_selection: dict[str, float] | None = None
    if resume_payload is not None:
        restore_gradient_connectivity(
            resume_payload,
            source=resume_path,
            require_verified=False,
        )
        model.load_state_dict(resume_payload["state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        scaler_state = resume_payload.get("scaler_state_dict")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        train_shuffle_generator.set_state(
            resume_payload["train_shuffle_generator_state"].cpu()
        )
        train_worker_generator.set_state(
            resume_payload["train_worker_generator_state"].cpu()
        )
        val_worker_generator.set_state(
            resume_payload["val_worker_generator_state"].cpu()
        )
        restore_rng_state(resume_payload["rng_state"])
        completed_epoch = int(resume_payload["epoch"])
        start_epoch = completed_epoch + 1
        best_epoch = int(resume_payload.get("best_epoch", 0))
        saved_selection = resume_payload.get("best_selection")
        if not isinstance(saved_selection, dict):
            raise ValueError(
                "Resume checkpoint has no complete finite best_selection record; "
                "sentinel-based MRR-only migration is forbidden."
            )
        best_selection = _checkpoint_selection_record(saved_selection)
        if "best_mrr" not in resume_payload:
            raise KeyError(
                "Resume checkpoint is missing best_mrr; no non-finite sentinel will "
                "be substituted for checkpoint selection."
            )
        best_mrr = _finite_metric(resume_payload["best_mrr"], name="resume best_mrr")
        if not math.isclose(
            best_mrr,
            best_selection["mrr"],
            rel_tol=0.0,
            abs_tol=10.0 ** (-checkpoint_metric_precision),
        ):
            raise ValueError(
                "Resume checkpoint best_mrr disagrees with finite best_selection.mrr"
            )
        print(
            f"[Resume] completed_epoch={completed_epoch}/{epochs} "
            f"next_epoch={start_epoch} best_mrr={best_mrr:.4f} "
            f"best_epoch={best_epoch}"
        )

    def epoch_pass(loader: DataLoader, epoch: int, training_mode: bool) -> dict[str, Any]:
        model.train(training_mode)
        # total, ranking, consistency, ce, pair, ordinal, mined,
        # acc, RR, count, positive score, hardest-negative score, margin
        totals = torch.zeros(13, dtype=torch.float64, device=device)
        total_batches = len(loader)
        if total_batches < 1:
            raise RuntimeError(
                f"{'Training' if training_mode else 'Validation'} loader has no batches"
            )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        pass_started = time.perf_counter()
        cpu_started = time.process_time()
        if training_mode:
            optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            group_start = (batch_index // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, total_batches - group_start)
            with torch.set_grad_enabled(training_mode):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    output = model(batch)
                    ranking_loss, parts = curriculum_ranking_loss(
                        output.scores,
                        batch.difficulty_list(),
                        epoch=epoch,
                        config=curriculum,
                    )
                    loss = ranking_loss + consistency_weight * output.consistency
                if training_mode:
                    scaler.scale(loss / float(group_size)).backward()
                    gradient_connected_parameter_names.update(
                        name
                        for name, parameter in trainable_named_parameters
                        if parameter.grad is not None
                    )
                    should_step = (
                        (batch_index + 1) % accumulation_steps == 0
                        or batch_index + 1 == total_batches
                    )
                    if should_step:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            trainable_parameters, float(training["grad_clip"])
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

            accuracy_sum, reciprocal_rank_sum, sample_count = ranking_metric_sums(
                [score.detach() for score in output.scores]
            )
            batch_count = sample_count.to(torch.float64)
            part_vector = torch.stack(
                [
                    loss.detach(),
                    ranking_loss.detach(),
                    output.consistency.detach(),
                    parts["ce"],
                    parts["pair"],
                    parts["ordinal"],
                    parts["mined"],
                ]
            ).to(torch.float64)
            totals[:7] += part_vector * batch_count
            totals[7] += accuracy_sum.to(torch.float64)
            totals[8] += reciprocal_rank_sum.to(torch.float64)
            totals[9] += batch_count
            positive_scores = torch.stack([score[0] for score in output.scores])
            hardest_negative_scores = torch.stack(
                [score[1:].max() for score in output.scores]
            )
            margins = positive_scores - hardest_negative_scores
            totals[10] += positive_scores.detach().to(torch.float64).sum()
            totals[11] += hardest_negative_scores.detach().to(torch.float64).sum()
            totals[12] += margins.detach().to(torch.float64).sum()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - pass_started
        cpu_elapsed = time.process_time() - cpu_started
        values = totals.detach().cpu().tolist()
        count = float(values[9])
        if count <= 0.0:
            raise RuntimeError(
                f"{'Training' if training_mode else 'Validation'} pass produced no samples"
            )
        names = ("loss", "ranking", "consistency", "ce", "pair", "ordinal", "mined")
        result = {name: float(values[index]) / count for index, name in enumerate(names)}
        result["acc"] = float(values[7]) / count
        result["mrr"] = float(values[8]) / count
        result["positive"] = float(values[10]) / count
        result["hardest_negative"] = float(values[11]) / count
        result["margin"] = float(values[12]) / count
        for metric_name in (
            "loss",
            "ranking",
            "consistency",
            "ce",
            "pair",
            "ordinal",
            "mined",
            "acc",
            "mrr",
            "positive",
            "hardest_negative",
            "margin",
        ):
            result[metric_name] = _finite_metric(
                result[metric_name],
                name=f"{'train' if training_mode else 'validation'} {metric_name}",
            )
        affinity = resource_report.get("cpu_affinity_cores")
        cpu_capacity = int(affinity) if isinstance(affinity, int) and affinity > 0 else (os.cpu_count() or 1)
        result.update(
            {
                "samples": int(count),
                "batches": total_batches,
                "optimizer_steps": (
                    math.ceil(total_batches / accumulation_steps) if training_mode else 0
                ),
                "elapsed_seconds": elapsed,
                "samples_per_second": count / elapsed if elapsed > 0.0 else UNAVAILABLE,
                "process_cpu_seconds": cpu_elapsed,
                "process_cpu_utilization_percent_of_allocated_capacity": (
                    cpu_elapsed / elapsed * 100.0 / cpu_capacity
                    if elapsed > 0.0
                    else UNAVAILABLE
                ),
                "gpu_utilization_percent": f"{UNAVAILABLE} (external profiler not attached)",
                **process_memory_snapshot(),
            }
        )
        if device.type == "cuda":
            result.update(cuda_memory_snapshot(device))
        else:
            result["cuda_peak_allocated_bytes"] = f"{UNAVAILABLE} (CPU run)"
        return result

    print(
        "[Runtime] "
        f"device={device} amp={use_amp} batch_size={batch_size} "
        f"workers={workers} prefetch={training.get('prefetch_factor', 2)} "
        f"persistent_workers={bool(workers and training.get('persistent_workers', True))} "
        f"cuda_prefetch={cuda_prefetch} "
        f"fused_adamw={fused_optimizer} "
        f"gradient_accumulation={accumulation_steps} "
        f"consistency_weight={consistency_weight:.3f} "
        f"fixed_validation_epoch={fixed_validation_epoch} "
        f"checkpoint_order=mrr/acc/margin/loss "
        f"ablation_mode={ablation_mode} "
        f"epoch_range={start_epoch}-{epochs}"
    )

    run_started = time.perf_counter()
    run_cpu_started = time.process_time()
    completed_train_samples = 0
    completed_validation_samples = 0
    observed_peak_vram_bytes = 0
    for epoch in range(start_epoch, epochs + 1):
        train_dataset.set_epoch(epoch)
        try:
            train_metrics = epoch_pass(train_epoch_loader, epoch, True)
        except Exception as exc:
            if is_cuda_out_of_memory(exc):
                raise_cuda_out_of_memory(
                    exc,
                    device=device,
                    context=f"training epoch {epoch}",
                    extra={
                        "physical_batch_size": batch_size,
                        "gradient_accumulation_steps": accumulation_steps,
                        "run_mode": run_mode,
                    },
                )
            raise
        completed_train_samples += int(train_metrics["samples"])
        if val_epoch_loader is not None:
            try:
                with torch.no_grad():
                    val_metrics = epoch_pass(
                        val_epoch_loader, fixed_validation_epoch, False
                    )
            except Exception as exc:
                if is_cuda_out_of_memory(exc):
                    raise_cuda_out_of_memory(
                        exc,
                        device=device,
                        context=f"validation after training epoch {epoch}",
                        extra={"physical_batch_size": batch_size, "run_mode": run_mode},
                    )
                raise
            completed_validation_samples += int(val_metrics["samples"])
            selection = _checkpoint_selection_record(val_metrics)
        else:
            val_metrics = {
                "availability": "unavailable (non-production run has no validation split)"
            }
            selection = _checkpoint_selection_record(train_metrics)
        scheduler.step()
        connectivity = gradient_connectivity_record()
        if epoch == epochs and not connectivity["verified"]:
            raise RuntimeError(
                "Final training epoch completed without gradient connectivity for all "
                f"trainable parameters: {connectivity['missing_parameters']}"
            )
        improved = _checkpoint_selection_is_better(
            selection,
            best_selection,
            precision=checkpoint_metric_precision,
        )
        if improved:
            best_selection = dict(selection)
            best_mrr = float(selection["mrr"])
            best_epoch = epoch
            save_checkpoint_atomic(
                {
                    **static_checkpoint_metadata,
                    "state_dict": model.state_dict(),
                    "prototype_bank": str(Path(args.prototype_bank).resolve()),
                    "epoch": epoch,
                    "target_epochs": epochs,
                    "selection_mrr": float(selection["mrr"]),
                    "selection": dict(selection),
                    "best_selection": dict(best_selection),
                    "best_mrr": float(best_mrr),
                    "best_epoch": int(best_epoch),
                    "validation_policy": validation_policy,
                    "training_signature": training_signature,
                    "preflight_calibration": preflight_calibration,
                    "gradient_connectivity": connectivity,
                    "training_complete": bool(epoch == epochs),
                    "train_files": len(train_files),
                    "val_files": len(val_files),
                    "runtime": {
                        "amp": use_amp,
                        "deterministic": deterministic,
                        "allow_tf32": bool(runtime.get("allow_tf32", False)),
                        "workers": workers,
                        "prefetch_factor": int(training.get("prefetch_factor", 2)),
                        "cuda_prefetch": cuda_prefetch,
                        "fused_adamw": fused_optimizer,
                    },
                },
                checkpoint_path,
            )

        if best_mrr is None or best_selection is None:
            raise RuntimeError(
                "Finite checkpoint selection did not produce a best metric record"
            )

        # This sidecar is the authoritative restart point. It is written after
        # a full train+validation epoch and scheduler step, so an interrupted
        # process repeats at most the currently unfinished epoch.
        save_checkpoint_atomic(
            {
                "format": "hiercp_training_state_v1",
                **static_checkpoint_metadata,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "rng_state": capture_rng_state(),
                "train_shuffle_generator_state": train_shuffle_generator.get_state(),
                "train_worker_generator_state": train_worker_generator.get_state(),
                "val_worker_generator_state": val_worker_generator.get_state(),
                "training_signature": training_signature,
                "preflight_calibration": preflight_calibration,
                "gradient_connectivity": connectivity,
                "epoch": int(epoch),
                "target_epochs": int(epochs),
                "best_mrr": float(best_mrr),
                "best_epoch": int(best_epoch),
                "best_selection": (
                    dict(best_selection) if best_selection is not None else None
                ),
                "validation_policy": validation_policy,
                "training_complete": bool(epoch == epochs),
                "best_checkpoint": str(checkpoint_path.resolve()),
            },
            resume_path,
        )
        for metrics in (train_metrics, val_metrics):
            peak = metrics.get("cuda_peak_allocated_bytes")
            if isinstance(peak, int):
                observed_peak_vram_bytes = max(observed_peak_vram_bytes, peak)
        _print_report(
            "EpochPostRun",
            {
                "epoch": epoch,
                "run_mode": run_mode,
                "train": train_metrics,
                "validation": val_metrics,
                "checkpoint_selection": selection,
                "best_mrr": best_mrr,
                "best_epoch": best_epoch,
                "best_checkpoint_updated": improved,
                "resume_checkpoint": str(resume_path),
            },
        )

    if not checkpoint_path.is_file():
        raise RuntimeError("Training completed without producing a best-model checkpoint")
    if best_mrr is None or best_selection is None:
        raise RuntimeError("Training completed without a finite best checkpoint selection")

    # Mark the best-model artifact as belonging to a completed run. Generation
    # still loads this best checkpoint; future training resumes from the sidecar.
    completed_best = torch_load_compat(checkpoint_path, map_location="cpu")
    completed_best["training_complete"] = True
    completed_best["completed_epoch"] = int(epochs)
    completed_best["target_epochs"] = int(epochs)
    completed_best["best_mrr"] = float(best_mrr)
    completed_best["best_epoch"] = int(best_epoch)
    completed_best["best_selection"] = (
        dict(best_selection) if best_selection is not None else None
    )
    completed_best["validation_policy"] = validation_policy
    completed_best["preflight_calibration"] = preflight_calibration
    final_connectivity = gradient_connectivity_record()
    if not final_connectivity["verified"]:
        raise RuntimeError(
            "Refusing to mark training complete without full trainable-parameter "
            f"gradient connectivity: {final_connectivity['missing_parameters']}"
        )
    completed_best["gradient_connectivity"] = final_connectivity
    save_checkpoint_atomic(completed_best, checkpoint_path)
    print(
        f"[OK] Training complete: best={checkpoint_path} last={resume_path} "
        f"best_mrr={best_mrr:.4f} best_epoch={best_epoch}"
    )
    run_elapsed = time.perf_counter() - run_started
    run_cpu_elapsed = time.process_time() - run_cpu_started
    total_processed = completed_train_samples + completed_validation_samples
    _print_report(
        "PostRun",
        {
            "command": "train",
            "run_mode": run_mode,
            "status": "complete",
            "implementation_complete": True,
            "static_checks_complete": UNAVAILABLE,
            "unit_tests_complete": UNAVAILABLE,
            "smoke_test_complete": False,
            "full_training_complete": True,
            "full_evaluation_complete": False,
            "actual_cached_data_used": True,
            "epochs_executed_this_invocation": epochs - start_epoch + 1,
            "train_samples_processed": completed_train_samples,
            "validation_samples_processed": completed_validation_samples,
            "elapsed_seconds": run_elapsed,
            "samples_per_second": (
                total_processed / run_elapsed if run_elapsed > 0.0 else UNAVAILABLE
            ),
            "process_cpu_seconds": run_cpu_elapsed,
            "observed_peak_vram_bytes": (
                observed_peak_vram_bytes
                if device.type == "cuda"
                else f"{UNAVAILABLE} (CPU run)"
            ),
            "process_memory": process_memory_snapshot(),
            "best_mrr": best_mrr,
            "best_epoch": best_epoch,
            "gradient_connectivity": final_connectivity,
            "remaining_limitations": [
                "single-device execution only",
                "GPU utilization requires an external profiler",
            ],
        },
    )


@dataclass
class _PreparedGenerationCase:
    case_id: str
    status: str
    message: str = ""
    case: LoadedCase | None = None
    source: SourceTumor | None = None
    candidates: list[Any] | None = None
    sample: dict[str, Any] | None = None
    regions: PatientRegionData | None = None
    rng: np.random.Generator | None = None


@dataclass
class _GenerationState:
    case_id: str
    case: LoadedCase
    source: SourceTumor
    rng: np.random.Generator
    occupied: np.ndarray
    out_image: np.ndarray
    out_label: np.ndarray
    candidates: list[Any]
    sample: dict[str, Any]
    centers: list[str]
    selected_scores: list[str]
    coverages: list[str]
    active: bool = True
    error_message: str = ""


def run_generate(args: argparse.Namespace) -> None:
    import torch

    from hiercp.data import collate_samples, summarize_hierarchical_batch
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.tensor import (
        CUDAOutOfMemoryError,
        collect_runtime_resources,
        configure_runtime,
        cuda_memory_snapshot,
        enforce_single_device_execution,
        is_cuda_out_of_memory,
        load_checkpoint,
        process_memory_snapshot,
        raise_cuda_out_of_memory,
        resolve_device,
    )

    run_mode = _guard_reduction_overrides(
        args, option_names=("max_cases", "case_ids")
    )
    config = _load_json(args.config)
    generation = config["generation"]
    runtime = config.get("runtime", {})
    labels = config["labels"]
    liver_label = int(labels["liver"])
    tumor_label = int(labels["tumor"])
    seed = int(args.seed if args.seed is not None else config["seed"])
    deterministic = bool(runtime.get("deterministic", True))
    configure_runtime(
        deterministic=deterministic,
        allow_tf32=bool(runtime.get("allow_tf32", False)),
        cudnn_benchmark=bool(runtime.get("cudnn_benchmark", not deterministic)),
    )
    device = resolve_device(args.device)
    if run_mode == "production":
        enforce_single_device_execution(device, context="hiercp.pipeline generate")
    checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint.get("method") != CHECKPOINT_METHOD or checkpoint.get("framework") != "torch_geometric":
        raise ValueError(f"Not an optimized hierarchical PyG checkpoint: {checkpoint.get('method')}")
    if checkpoint.get("upper_feature_policy") != UPPER_FEATURE_POLICY:
        raise ValueError(
            "Checkpoint predates the shortcut-safe upper feature policy; "
            "restart only the training stage with --overwrite"
        )
    if run_mode == "production" and not bool(checkpoint.get("training_complete", False)):
        raise ValueError(
            "Production generation requires a checkpoint from a completed full training run"
        )
    if run_mode == "production":
        training_signature = checkpoint.get("training_signature")
        if (
            not isinstance(training_signature, dict)
            or training_signature.get("format") != "hiercp_training_signature_v1"
            or training_signature.get("run_mode") != "production"
            or training_signature.get("ablation_mode", "full") != "full"
        ):
            raise ValueError(
                "Production generation requires a checkpoint whose signed training "
                "identity is run_mode='production' with the full model path; ablation, "
                "benchmark, debug, and legacy unsigned checkpoints are not accepted"
            )
    model = HierarchicalPyGPlacementModel(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if run_mode == "production":
        connectivity = checkpoint.get("gradient_connectivity")
        expected_trainable = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        connected_values = (
            connectivity.get("connected_parameters")
            if isinstance(connectivity, dict)
            else None
        )
        connected_values_valid = isinstance(connected_values, list) and all(
            isinstance(name, str) for name in connected_values
        )
        connected = (
            set(connected_values)
            if connected_values_valid
            else set()
        )
        if (
            not isinstance(connectivity, dict)
            or connectivity.get("verified") is not True
            or not connected_values_valid
            or connected != expected_trainable
        ):
            raise ValueError(
                "Production generation requires a completed checkpoint with exact "
                "trainable-parameter gradient-connectivity verification"
            )
    graph_config = graph_config_from_dict(checkpoint["graph_config"])
    _require_full_graph(graph_config, "full-graph checkpoint")
    ct_clip = tuple(float(value) for value in checkpoint["ct_clip"])
    bank = PrototypeBank.load(args.prototype_bank)
    if set(bank.training_case_ids) != set(checkpoint["prototype_training_cases"]):
        raise ValueError("Prototype bank training cases do not match the placement checkpoint")
    if bank.fingerprint() != checkpoint.get("prototype_fingerprint"):
        raise ValueError("Prototype bank content does not match the placement checkpoint")

    use_amp = bool(generation.get("amp", True) and device.type == "cuda")
    pin_memory = bool(generation.get("pin_memory", True) and device.type == "cuda")
    local_candidate_chunk_size = max(
        1, int(generation.get("local_candidate_chunk_size", 8))
    )
    cpu_workers = max(1, int(generation.get("cpu_prefetch_workers", 2)))
    prefetch_cases = max(1, int(generation.get("cpu_prefetch_cases", 2)))
    save_queue_depth = max(1, int(generation.get("save_queue_depth", 2)))
    case_batch_size = int(generation.get("case_batch_size", 2))
    if case_batch_size < 1:
        raise ValueError("generation.case_batch_size must be positive")
    if run_mode == "production" and case_batch_size < 2:
        raise ValueError(
            "Production generation requires generation.case_batch_size >= 2 so "
            "independent cases are not silently processed serially"
        )
    if prefetch_cases < case_batch_size:
        raise ValueError(
            "generation.cpu_prefetch_cases must be at least case_batch_size to keep "
            f"the configured GPU case batch supplied ({prefetch_cases} < {case_batch_size})"
        )
    requested_copies = int(generation["num_copies"])
    if requested_copies < 1:
        raise ValueError("generation.num_copies must be positive")
    resource_report = collect_runtime_resources(device, storage_path=args.out_dir)
    total_parameters = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable_parameters = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    generation_started = time.perf_counter()
    generation_cpu_started = time.process_time()
    score_elapsed_seconds = 0.0
    score_calls = 0
    scored_candidates = 0
    scored_case_instances = 0
    observed_case_batch_sizes: list[int] = []
    graph_sample_reported = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def prepare_first_copy(paths: CasePaths) -> _PreparedGenerationCase:
        rng = np.random.default_rng(
            stable_case_seed(seed, paths.case_id, "hierarchical_pyg_generation")
        )
        try:
            case = load_case(paths)
            occupied = case.label == tumor_label
            if not np.any(occupied):
                return _PreparedGenerationCase(paths.case_id, "no_tumor")
            source, _, _ = choose_source_tumor(
                case.image,
                case.label,
                tumor_label=tumor_label,
                rng=rng,
                selection=str(generation["source_selection"]),
                pad=int(generation["source_pad"]),
            )
            region_seed = stable_case_seed(seed, paths.case_id, REGION_CACHE_SEED_SALT)
            regions = load_or_build_patient_regions(
                case,
                cache_dir=args.region_cache_dir,
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
                rng=rng,
                num_candidates=int(generation["num_candidates"]),
                max_draws=int(generation["max_draws"]),
                min_liver_coverage=float(generation["min_liver_coverage"]),
                occupied_clearance_vox=int(generation["occupied_clearance_vox"]),
                min_center_separation_mm=float(generation["min_center_separation_mm"]),
            )
            if not candidates:
                return _PreparedGenerationCase(
                    paths.case_id,
                    "no_candidate",
                    case=case,
                    source=source,
                    regions=regions,
                    rng=rng,
                )
            sample, _ = build_inference_sample(
                case,
                source,
                candidates,
                bank,
                graph_config=graph_config,
                liver_label=liver_label,
                tumor_label=tumor_label,
                ct_clip=ct_clip,
                seed=stable_case_seed(seed, paths.case_id, "infer_0"),
                regions=regions,
            )
            return _PreparedGenerationCase(
                paths.case_id,
                "ready",
                case=case,
                source=source,
                candidates=list(candidates),
                sample=sample,
                regions=regions,
                rng=rng,
            )
        except Exception as exc:
            if is_cuda_out_of_memory(exc):
                raise_cuda_out_of_memory(
                    exc,
                    device=device,
                    context=f"generation preparation for case {paths.case_id}",
                    extra={"run_mode": run_mode},
                )
            return _PreparedGenerationCase(paths.case_id, "error", message=str(exc))

    def build_followup(
        active_case: LoadedCase,
        source: SourceTumor,
        occupied: np.ndarray,
        rng: np.random.Generator,
        copy_index: int,
    ) -> tuple[list[Any], dict[str, Any]] | None:
        # For additional copies, preserve the original behavior by rebuilding
        # context descriptors from the modified volume. Region geometry remains
        # deterministic; no stale on-disk cache is used for an in-memory case.
        regions = build_patient_regions(
            active_case,
            liver_label=liver_label,
            tumor_label=tumor_label,
            config=graph_config,
            rng=np.random.default_rng(
                stable_case_seed(
                    seed, active_case.paths.case_id, REGION_CACHE_SEED_SALT
                )
            ),
            ct_clip=ct_clip,
        )
        candidates, _ = build_candidate_pool(
            active_case,
            source,
            placement_mask=active_case.label == liver_label,
            full_organ_mask=regions.full_organ_mask,
            occupied_mask=occupied,
            organ_distance=regions.organ_depth,
            rng=rng,
            num_candidates=int(generation["num_candidates"]),
            max_draws=int(generation["max_draws"]),
            min_liver_coverage=float(generation["min_liver_coverage"]),
            occupied_clearance_vox=int(generation["occupied_clearance_vox"]),
            min_center_separation_mm=float(generation["min_center_separation_mm"]),
        )
        if not candidates:
            return None
        sample, _ = build_inference_sample(
            active_case,
            source,
            candidates,
            bank,
            graph_config=graph_config,
            liver_label=liver_label,
            tumor_label=tumor_label,
            ct_clip=ct_clip,
            seed=stable_case_seed(
                seed, active_case.paths.case_id, f"infer_{copy_index}"
            ),
            regions=regions,
        )
        return list(candidates), sample

    def score_batch(
        batch: Any,
        *,
        expected_case_ids: tuple[str, ...],
        expected_counts: tuple[int, ...],
    ) -> list[np.ndarray]:
        nonlocal score_calls, score_elapsed_seconds, scored_candidates
        nonlocal scored_case_instances
        actual_case_ids = tuple(str(value) for value in batch.case_ids)
        actual_counts = tuple(int(value) for value in batch.counts)
        if actual_case_ids != expected_case_ids or actual_counts != expected_counts:
            raise RuntimeError(
                "Generation collation changed ordered case/candidate mapping: "
                f"expected_ids={expected_case_ids}, actual_ids={actual_case_ids}, "
                f"expected_counts={expected_counts}, actual_counts={actual_counts}"
            )
        try:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                score_tensors = model.score_inference_chunked(
                    batch, local_chunk_size=local_candidate_chunk_size
                )
            if len(score_tensors) != len(expected_case_ids):
                raise RuntimeError(
                    "Generation scorer returned the wrong number of ordered cases: "
                    f"expected={len(expected_case_ids)}, actual={len(score_tensors)}"
                )
            values = [tensor.float().cpu().numpy() for tensor in score_tensors]
            cardinality_mismatches = [
                (index, expected_counts[index], int(scores.size))
                for index, scores in enumerate(values)
                if int(scores.size) != expected_counts[index]
            ]
            if cardinality_mismatches:
                raise RuntimeError(
                    "Generation scorer changed per-case candidate mapping: "
                    f"{cardinality_mismatches}"
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            score_elapsed_seconds += time.perf_counter() - started
            score_calls += 1
            scored_candidates += sum(int(scores.size) for scores in values)
            scored_case_instances += len(values)
            observed_case_batch_sizes.append(len(values))
            return values
        except Exception as exc:
            if is_cuda_out_of_memory(exc):
                raise_cuda_out_of_memory(
                    exc,
                    device=device,
                    context="multi-case chunked generation scoring",
                    extra={
                        "candidate_count": sum(int(value) for value in batch.counts),
                        "local_candidate_chunk_size": local_candidate_chunk_size,
                        "independent_case_gpu_batch_size": len(batch.counts),
                        "configured_case_batch_size": case_batch_size,
                        "case_ids": list(expected_case_ids),
                        "run_mode": run_mode,
                    },
                )
            raise

    def save_payload(
        case: LoadedCase,
        out_image: np.ndarray,
        out_label: np.ndarray,
        row: dict[str, object],
    ) -> dict[str, object]:
        save_case_pair(
            case,
            out_image,
            out_label,
            args.out_dir,
            overwrite=args.overwrite,
        )
        image_path, label_path = output_paths(args.out_dir, case.paths.case_id)
        saved_row = dict(row)
        saved_row["output_image_sha256"] = _sha256_file(image_path)
        saved_row["output_label_sha256"] = _sha256_file(label_path)
        return saved_row

    full_paths = discover_cases(args.data_dir)
    all_paths = discover_cases(
        args.data_dir,
        max_cases=args.max_cases,
        case_ids=args.case_ids,
        run_mode=run_mode,
    )
    checkpoint_sha256 = _sha256_file(args.checkpoint)
    config_sha256 = _sha256_file(args.config)
    generation_identity = {
        "format": "hiercp_generation_identity_v1",
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": config_sha256,
        "prototype_fingerprint": bank.fingerprint(),
        "seed": seed,
        "run_mode": run_mode,
        "generation_config": generation,
        "selected_case_ids": [paths.case_id for paths in all_paths],
    }
    generation_identity_sha256 = hashlib.sha256(
        json.dumps(
            generation_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = Path(args.out_dir) / "manifest.csv"
    existing_manifest_rows: dict[str, dict[str, str]] = {}
    if manifest_path.is_file():
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                case_id = str(row.get("case_id", ""))
                if not case_id:
                    raise ValueError(
                        f"Existing generation manifest has a row without case_id: {manifest_path}"
                    )
                if case_id in existing_manifest_rows:
                    raise ValueError(
                        f"Existing generation manifest repeats case_id={case_id!r}: "
                        f"{manifest_path}"
                    )
                existing_manifest_rows[case_id] = dict(row)
    rows: list[dict[str, object]] = []
    pending_saves: deque[tuple[str, Future[dict[str, object]]]] = deque()

    def flush_one_save() -> None:
        case_id, future = pending_saves.popleft()
        try:
            row = future.result()
            rows.append(row)
            print(f"[OK] {row['case_id']} pasted={row['pasted']}")
        except Exception as exc:
            rows.append({"case_id": case_id, "status": "save_error", "message": str(exc)})
            print(f"[Error] save {case_id}: {exc}")

    eligible: list[CasePaths] = []
    for paths in all_paths:
        output_image_path, output_label_path = output_paths(
            args.out_dir, paths.case_id
        )
        image_exists = output_image_path.exists()
        label_exists = output_label_path.exists()
        if image_exists != label_exists and not args.overwrite:
            raise FileExistsError(
                f"Partial generation pair exists for {paths.case_id}: "
                f"image={image_exists}, label={label_exists}. Use --overwrite or a "
                "new output directory; no file will be overwritten implicitly."
            )
        if image_exists and label_exists and not args.overwrite:
            existing = existing_manifest_rows.get(paths.case_id)
            if existing is None:
                raise FileExistsError(
                    f"Existing pair for {paths.case_id} has no manifest provenance. "
                    "Use --overwrite or a new output directory."
                )
            expected_provenance = {
                "generation_identity_sha256": generation_identity_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "config_sha256": config_sha256,
                "prototype_fingerprint": bank.fingerprint(),
                "source_image_sha256": _sha256_file(paths.image_path),
                "source_label_sha256": _sha256_file(paths.label_path),
                "output_image_sha256": _sha256_file(output_image_path),
                "output_label_sha256": _sha256_file(output_label_path),
            }
            mismatches = [
                key
                for key, expected in expected_provenance.items()
                if existing.get(key) != expected
            ]
            try:
                requested_matches = int(existing.get("requested", "-1")) == requested_copies
                pasted_matches = int(existing.get("pasted", "-1")) == requested_copies
            except (TypeError, ValueError):
                requested_matches = False
                pasted_matches = False
            if existing.get("status") not in {"ok", "exists_verified"}:
                mismatches.append("status")
            if not requested_matches:
                mismatches.append("requested")
            if not pasted_matches:
                mismatches.append("pasted")
            if mismatches:
                raise FileExistsError(
                    f"Existing pair for {paths.case_id} failed exact resume "
                    f"verification ({sorted(set(mismatches))}). Use --overwrite or "
                    "a new output directory."
                )
            verified_row: dict[str, object] = dict(existing)
            verified_row["status"] = "exists_verified"
            verified_row["reused_verified"] = True
            rows.append(verified_row)
            print(f"[Skip] Verified existing output: {paths.case_id}")
            continue
        eligible.append(paths)

    selected_case_count = len(all_paths)
    total_case_count = len(full_paths)
    subset_active = args.max_cases is not None or bool(args.case_ids)
    _print_report(
        "PreRun",
        {
            "command": "generate",
            "run_mode": run_mode,
            "model": {
                "name": type(model).__name__,
                "configuration": checkpoint["model_kwargs"],
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
            },
            "graph_and_input": {
                "graph_configuration": graph_config.to_dict(),
                "input_resolution": [graph_config.patch_size] * 3,
                "representative_node_edge_shape_statistics": (
                    f"{UNAVAILABLE} (reported after first prepared real case)"
                ),
                "sampling_ratio": f"{UNAVAILABLE} (not defined by generation contract)",
                "time_window": f"{UNAVAILABLE} (not a temporal model)",
            },
            "dataset": {
                "total_discovered_cases": total_case_count,
                "selected_cases": selected_case_count,
                "eligible_cases_this_invocation": len(eligible),
                "already_existing_cases": selected_case_count - len(eligible),
                "usage_ratio": (
                    selected_case_count / total_case_count if total_case_count else 0.0
                ),
            },
            "inference": {
                "physical_case_batch_size": case_batch_size,
                "effective_case_batch_size": case_batch_size,
                "precision": "AMP autocast" if use_amp else "float32",
                "local_candidate_chunk_size": local_candidate_chunk_size,
                "requested_copies_per_case": requested_copies,
            },
            "parallelism": {
                "cpu_prepare_workers": cpu_workers,
                "cpu_prefetch_cases": prefetch_cases,
                "save_workers": 1,
                "save_queue_depth": save_queue_depth,
                "independent_case_gpu_batching": True,
                "constraint": (
                    "Independent cases at the same copy index are disjoint-union "
                    "batched on GPU. Copy indices remain sequential because each "
                    "case's next graph depends on its preceding paste."
                ),
                "single_device_limit": True,
            },
            "flags": {
                "debug": run_mode == "debug",
                "benchmark": run_mode == "benchmark",
                "ablation": run_mode == "ablation",
                "subset": subset_active,
                "fast_mode": False,
                "max_cases": args.max_cases if args.max_cases is not None else UNAVAILABLE,
                "case_id_filter": args.case_ids if args.case_ids else UNAVAILABLE,
            },
            "device_resources": resource_report,
            "unavailable_metrics_before_execution": [
                "throughput",
                "scoring time",
                "peak generation VRAM",
                "process CPU utilization",
            ],
        },
    )

    def state_from_prepared(prepared: _PreparedGenerationCase) -> _GenerationState:
        missing = [
            name
            for name, value in (
                ("case", prepared.case),
                ("source", prepared.source),
                ("candidates", prepared.candidates),
                ("sample", prepared.sample),
                ("rng", prepared.rng),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(
                f"Ready generation case {prepared.case_id} lacks fields: {missing}"
            )
        case = prepared.case
        source = prepared.source
        candidates = prepared.candidates
        sample = prepared.sample
        rng = prepared.rng
        if not isinstance(sample, dict) or str(sample.get("case_id", "")) != prepared.case_id:
            raise RuntimeError(
                "Prepared generation sample lost case identity: "
                f"expected={prepared.case_id!r}, actual={sample.get('case_id') if isinstance(sample, dict) else type(sample).__name__!r}"
            )
        return _GenerationState(
            case_id=prepared.case_id,
            case=case,
            source=source,
            rng=rng,
            occupied=case.label == tumor_label,
            out_image=case.image.copy(),
            out_label=case.label.copy(),
            candidates=list(candidates),
            sample=sample,
            centers=[],
            selected_scores=[],
            coverages=[],
        )

    with ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="hiercp-prepare") as prepare_pool, ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="hiercp-save"
    ) as save_pool:
        iterator = iter(eligible)
        pending_prepare: deque[Future[_PreparedGenerationCase]] = deque()
        for _ in range(min(prefetch_cases, len(eligible))):
            try:
                pending_prepare.append(prepare_pool.submit(prepare_first_copy, next(iterator)))
            except StopIteration:
                break

        while pending_prepare:
            states: list[_GenerationState] = []
            while pending_prepare and len(states) < case_batch_size:
                prepared_future = pending_prepare.popleft()
                try:
                    next_paths = next(iterator)
                except StopIteration:
                    next_paths = None
                if next_paths is not None:
                    pending_prepare.append(
                        prepare_pool.submit(prepare_first_copy, next_paths)
                    )

                prepared = prepared_future.result()
                if prepared.status != "ready":
                    status = prepared.status
                    rows.append(
                        {
                            "case_id": prepared.case_id,
                            "status": status,
                            "requested": requested_copies,
                            "pasted": 0,
                            **(
                                {"message": prepared.message}
                                if prepared.message
                                else {}
                            ),
                        }
                    )
                    print(
                        f"[Skip] {status}: {prepared.case_id}"
                        if status != "error"
                        else f"[Error] {prepared.case_id}: {prepared.message}"
                    )
                    continue
                states.append(state_from_prepared(prepared))

            if not states:
                continue

            for copy_index in range(requested_copies):
                round_states = [
                    state
                    for state in states
                    if state.active and not state.error_message
                ]
                if not round_states:
                    break

                if copy_index > 0:
                    followup_futures: list[
                        tuple[_GenerationState, Future[tuple[list[Any], dict[str, Any]] | None]]
                    ] = []
                    for state in round_states:
                        active_case = replace(
                            state.case,
                            image=state.out_image,
                            label=state.out_label,
                        )
                        future = prepare_pool.submit(
                            build_followup,
                            active_case,
                            state.source,
                            state.occupied,
                            state.rng,
                            copy_index,
                        )
                        followup_futures.append((state, future))

                    prepared_round: list[_GenerationState] = []
                    for state, future in followup_futures:
                        try:
                            followup = future.result()
                        except CUDAOutOfMemoryError:
                            raise
                        except Exception as exc:
                            state.active = False
                            state.error_message = str(exc)
                            print(f"[Error] {state.case_id}: {exc}")
                            continue
                        if followup is None:
                            state.active = False
                            print(
                                f"[Warn] {state.case_id}: no valid target for copy "
                                f"{copy_index + 1}"
                            )
                            continue
                        candidates, sample = followup
                        if str(sample.get("case_id", "")) != state.case_id:
                            raise RuntimeError(
                                "Follow-up preparation changed case identity: "
                                f"expected={state.case_id!r}, "
                                f"actual={sample.get('case_id')!r}"
                            )
                        state.candidates = candidates
                        state.sample = sample
                        prepared_round.append(state)
                    round_states = prepared_round
                    if not round_states:
                        continue

                expected_case_ids = tuple(state.case_id for state in round_states)
                expected_counts = tuple(
                    len(state.candidates) for state in round_states
                )
                batch = collate_samples([state.sample for state in round_states])
                if pin_memory:
                    batch.pin_memory()
                if not graph_sample_reported:
                    graph_report = summarize_hierarchical_batch(batch)
                    graph_report["scope"] = "first_ordered_generation_case_batch"
                    graph_report["expected_case_ids"] = list(expected_case_ids)
                    _print_report("GenerationGraphSample", graph_report)
                    graph_sample_reported = True

                try:
                    score_groups = score_batch(
                        batch,
                        expected_case_ids=expected_case_ids,
                        expected_counts=expected_counts,
                    )
                except CUDAOutOfMemoryError:
                    raise
                except Exception as exc:
                    for state in round_states:
                        state.active = False
                        state.error_message = str(exc)
                    print(
                        f"[Error] ordered case batch {list(expected_case_ids)}: {exc}"
                    )
                    break

                for state, scores in zip(round_states, score_groups):
                    try:
                        selected_index = choose_from_top_k(
                            scores,
                            rng=state.rng,
                            top_k=int(generation["top_k"]),
                            temperature=float(generation["temperature"]),
                        )
                        selected = state.candidates[selected_index]
                        paste_source(
                            state.out_image,
                            state.out_label,
                            state.occupied,
                            state.source,
                            selected,
                            tumor_label=tumor_label,
                            rng=state.rng,
                            intensity_scale_range=tuple(
                                float(value)
                                for value in generation["intensity_scale_range"]
                            ),
                            intensity_shift_range=tuple(
                                float(value)
                                for value in generation["intensity_shift_range"]
                            ),
                            blend_border=int(generation["blend_border"]),
                        )
                        state.centers.append(
                            ",".join(str(int(value)) for value in selected.center)
                        )
                        state.selected_scores.append(
                            f"{float(scores[selected_index]):.6f}"
                        )
                        state.coverages.append(
                            f"{float(selected.liver_coverage):.6f}"
                        )
                    except CUDAOutOfMemoryError:
                        raise
                    except Exception as exc:
                        state.active = False
                        state.error_message = str(exc)
                        print(f"[Error] {state.case_id}: {exc}")

            for state in states:
                status = (
                    "error"
                    if state.error_message
                    else (
                        "ok"
                        if len(state.centers) == requested_copies
                        else "incomplete"
                    )
                )
                row = {
                    "case_id": state.case_id,
                    "status": status,
                    "method": "hierarchical_pyg_context_region_prototype_optimized",
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "generation_identity_sha256": generation_identity_sha256,
                    "checkpoint_sha256": checkpoint_sha256,
                    "config_sha256": config_sha256,
                    "prototype_fingerprint": bank.fingerprint(),
                    "source_image_sha256": _sha256_file(state.case.paths.image_path),
                    "source_label_sha256": _sha256_file(state.case.paths.label_path),
                    "source_component": state.source.component_id,
                    "requested": requested_copies,
                    "pasted": len(state.centers),
                    "target_centers": ";".join(state.centers),
                    "selection_scores": ";".join(state.selected_scores),
                    "liver_coverages": ";".join(state.coverages),
                    **(
                        {"message": state.error_message}
                        if state.error_message
                        else {}
                    ),
                }
                if status == "error":
                    rows.append(row)
                    print(
                        f"[Error] {state.case_id}: {state.error_message}; "
                        "partial output was not saved"
                    )
                elif status == "incomplete" and run_mode == "production":
                    rows.append(row)
                    print(
                        f"[Error] {state.case_id}: requested={requested_copies} "
                        f"pasted={len(state.centers)}; partial output was not saved"
                    )
                else:
                    pending_saves.append(
                        (
                            state.case_id,
                            save_pool.submit(
                                save_payload,
                                state.case,
                                state.out_image,
                                state.out_label,
                                row,
                            ),
                        )
                    )
                    if len(pending_saves) >= save_queue_depth:
                        flush_one_save()

        while pending_saves:
            flush_one_save()

    discovery_order = {
        paths.case_id: index for index, paths in enumerate(all_paths)
    }
    rows.sort(
        key=lambda row: (
            discovery_order.get(str(row.get("case_id", "")), len(discovery_order)),
            str(row.get("case_id", "")),
        )
    )
    write_manifest(rows, manifest_path)
    statuses = Counter(str(row.get("status", "missing")) for row in rows)
    failed_rows = [
        row
        for row in rows
        if row.get("status") in {"error", "save_error", "no_candidate", "incomplete"}
        or (
            "requested" in row
            and int(row.get("pasted", 0)) < int(row.get("requested", 0))
        )
    ]
    generation_elapsed = time.perf_counter() - generation_started
    generation_cpu_elapsed = time.process_time() - generation_cpu_started
    post_run_report = {
        "command": "generate",
        "run_mode": run_mode,
        "status": "complete" if not failed_rows else "incomplete",
        "selected_cases": selected_case_count,
        "manifest_rows": len(rows),
        "status_counts": dict(statuses),
        "failure_count": len(failed_rows),
        "requested_copies_per_new_case": requested_copies,
        "scoring_calls": score_calls,
        "scored_case_instances": scored_case_instances,
        "scored_candidates": scored_candidates,
        "configured_physical_case_batch_size": case_batch_size,
        "configured_effective_case_batch_size": case_batch_size,
        "observed_case_batch_sizes": sorted(set(observed_case_batch_sizes)),
        "maximum_observed_case_batch_size": (
            max(observed_case_batch_sizes)
            if observed_case_batch_sizes
            else f"{UNAVAILABLE} (no scoring completed)"
        ),
        "scoring_elapsed_seconds": score_elapsed_seconds,
        "candidates_per_second": (
            scored_candidates / score_elapsed_seconds
            if score_elapsed_seconds > 0.0
            else f"{UNAVAILABLE} (no scoring completed)"
        ),
        "elapsed_seconds": generation_elapsed,
        "cases_per_second": (
            selected_case_count / generation_elapsed
            if generation_elapsed > 0.0
            else UNAVAILABLE
        ),
        "process_cpu_seconds": generation_cpu_elapsed,
        "process_memory": process_memory_snapshot(),
        "gpu_memory": (
            cuda_memory_snapshot(device)
            if device.type == "cuda"
            else {"cuda_memory": f"{UNAVAILABLE} (CPU run)"}
        ),
        "gpu_utilization_percent": f"{UNAVAILABLE} (external profiler not attached)",
        "actual_data_used": bool(score_calls),
        "full_generation_complete": not failed_rows,
        "full_training_executed": False,
        "full_evaluation_executed": False,
        "independent_case_gpu_batching": True,
        "remaining_limitations": [
            "single-device execution only",
            "copy indices are sequential within each case due to paste dependency",
            "the final or failure-filtered case batch may be smaller than configured",
        ],
    }
    _print_report("PostRun", post_run_report)
    if run_mode == "production" and failed_rows:
        examples = [
            {
                "case_id": row.get("case_id", "unknown"),
                "status": row.get("status", "missing"),
                "requested": row.get("requested", UNAVAILABLE),
                "pasted": row.get("pasted", UNAVAILABLE),
                "message": row.get("message", ""),
            }
            for row in failed_rows[:10]
        ]
        raise RuntimeError(
            "Production generation was incomplete. Diagnostic manifest was written, "
            "but no successful-completion status is emitted. "
            f"status_counts={dict(statuses)} examples={examples}"
        )


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare-prototypes":
            run_prepare_prototypes(args)
        elif args.command == "prepare":
            run_prepare(args)
        elif args.command == "train":
            run_train(args)
        elif args.command == "generate":
            run_generate(args)
        else:  # pragma: no cover
            raise RuntimeError(args.command)
    except Exception as exc:
        from hiercp.tensor import (
            CUDAOutOfMemoryError,
            is_cuda_out_of_memory,
            raise_cuda_out_of_memory,
        )

        if isinstance(exc, CUDAOutOfMemoryError):
            raise
        if is_cuda_out_of_memory(exc):
            raise_cuda_out_of_memory(
                exc,
                device=None,
                context=f"{args.command} pipeline initialization/execution",
                extra={
                    "run_mode": _run_mode(args),
                    "requested_device": getattr(args, "device", UNAVAILABLE),
                },
            )
        raise


if __name__ == "__main__":
    main()
