#!/usr/bin/env python3
"""Audit whether a trained HierCP model actually uses local context and geometry.

The audit preserves the candidate set and upper hierarchy while applying
controlled interventions to the cached local graphs:

* node_order: graph-isomorphic node permutation (must not change scores);
* upper_position_noise: forbidden source/candidate coordinates are randomized
  (must not change scores under the shortcut-safe policy);
* upper_clearance_noise: forbidden occupied-clearance values are randomized
  (must not change scores under the shortcut-safe policy);
* target_context: wrong target CT/context attached to the original topology;
* source_context: source-context features reassigned to different source nodes;
* edge_attr_zero: physical local edge attributes removed;
* topology_shuffle: local relation endpoints corrupted without rebuilding k-NN;
* view1_only: second sampled view removed.

Results include Top-1, MRR, positive-to-hardest-negative margin, score deltas,
local-embedding cosine similarity, and view-overlap diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch_geometric.data import Batch, HeteroData
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch Geometric is required. Run: python -m tools.install"
    ) from exc

from hiercp.data import (
    HierarchicalBatch,
    HierarchicalCacheDataset,
    collate_samples,
    list_cache_files,
    split_files_from_cache,
)
from hiercp.cache import validate_cache_publication
from hiercp.model import HierarchicalPyGPlacementModel
from hiercp.prototype import PrototypeBank
from hiercp.causality_overlap import local_edge_jaccard as exact_local_edge_jaccard
from hiercp.preparation_runtime import Measurement
from tools.causality_resources import (
    audit_allocation_fingerprint,
    audit_batch_input_bytes,
    audit_host_budget,
    build_audit_inventory,
    create_progress_log,
    validate_audit_host_budget,
)
from hiercp.schema import (
    LOCAL_EDGE_TYPES,
    LOCAL_NODE_TYPES,
    PATIENT_POSITION_EDGE_COLUMNS,
    UPPER_FEATURE_POLICY,
    UPPER_OCCUPIED_DISTANCE_INDEX,
    UPPER_POSITION_COLUMNS,
)
from hiercp.tensor import (
    collect_runtime_resources,
    configure_runtime,
    resolve_device,
    torch_load_compat,
)


CONTEXT_TYPES = (
    "source_context",
    "source_liver_surface",
    "target_context",
    "target_liver_surface",
)
TARGET_CONTEXT_TYPES = ("target_context", "target_liver_surface")
SOURCE_CONTEXT_TYPES = ("source_context", "source_liver_surface")
EMBEDDING_KEYS = (
    "tumor",
    "source_context",
    "target_context",
    "source_relation",
    "target_relation",
    "fused",
)
REPORT_FORMAT = "hiercp_causality_v3_hash_bound"
INPUT_FORMAT = "hiercp_causality_input_v3"
PREFLIGHT_FORMAT = "hiercp_causality_preflight_v2_host_bounded"
PREFLIGHT_IDENTITY_FORMAT = "hiercp_causality_preflight_identity_v2_host_bounded"
TRAINING_PREFLIGHT_FORMAT = "hiercp_preflight_calibration_v2"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required JSON is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.tmp.",
            suffix=".json",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _exact_positive_int(value: Any, *, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{context} must be a positive integer, got {value!r}")
    return int(value)


def _stable_resource_fingerprint(
    resource_report: dict[str, Any], *, device: torch.device
) -> dict[str, Any]:
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
        **{name: resource_report.get(name, "unavailable") for name in fields},
    }


def _candidate_values(trials: Any, *, key: str, minimum: int) -> list[int]:
    if not isinstance(trials, list):
        raise ValueError(f"Training preflight has no {key} trials")
    values: list[int] = []
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError(f"Training preflight {key} trial is not an object")
        value = trial.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < minimum
            or value in values
        ):
            raise ValueError(f"Training preflight has invalid {key}={value!r}")
        values.append(int(value))
    if len(values) < 2:
        raise ValueError(
            f"Causality auto-calibration requires at least two measured {key} candidates"
        )
    return values


def _training_measurement_plan(
    checkpoint: dict[str, Any],
    *,
    checkpoint_path: Path,
    cache_dir: Path,
    run_mode: str,
) -> dict[str, Any]:
    signature = checkpoint.get("training_signature")
    calibration = checkpoint.get("preflight_calibration")
    if not isinstance(signature, dict) or not isinstance(calibration, dict):
        raise ValueError("Checkpoint has no signed training preflight measurement plan")
    if signature.get("batch_setting") != "auto" or signature.get("worker_setting") != "auto":
        raise ValueError(
            "Causality requires checkpoint training with measured auto batch/worker plans"
        )
    if calibration.get("format") != TRAINING_PREFLIGHT_FORMAT:
        raise ValueError("Checkpoint training preflight format is unsupported")
    identity = calibration.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Checkpoint training preflight has no identity")
    expected_identity = {
        "checkpoint_path": str(checkpoint_path),
        "cache_dir": str(cache_dir),
        "run_mode": run_mode,
    }
    mismatches = [
        name for name, expected in expected_identity.items() if identity.get(name) != expected
    ]
    if mismatches:
        raise ValueError(f"Checkpoint training preflight identity mismatch: {mismatches}")
    if calibration.get("resource_fingerprint") != signature.get(
        "calibration_resource_fingerprint"
    ):
        raise ValueError("Checkpoint training preflight resource fingerprint is unsigned")
    maximum_vram_fraction = identity.get("batch_calibration_max_vram_fraction")
    if (
        not isinstance(maximum_vram_fraction, (int, float))
        or isinstance(maximum_vram_fraction, bool)
        or not math.isfinite(float(maximum_vram_fraction))
        or not 0.0 < float(maximum_vram_fraction) < 1.0
    ):
        raise ValueError("Training preflight has no valid VRAM headroom threshold")
    loader_batches = identity.get("loader_calibration_batches")
    if not isinstance(loader_batches, int) or isinstance(loader_batches, bool) or loader_batches < 1:
        raise ValueError("Training preflight has no valid loader measurement length")
    prefetch_factor = identity.get("prefetch_factor", 2)
    if not isinstance(prefetch_factor, int) or isinstance(prefetch_factor, bool) or prefetch_factor < 1:
        raise ValueError("Training preflight has no valid prefetch_factor")
    return {
        "format": "hiercp_causality_measurement_plan_v2_host_bounded",
        "batch_candidates": _candidate_values(
            calibration.get("batch_trials"), key="batch_size", minimum=1
        ),
        "worker_candidates": _candidate_values(
            calibration.get("worker_trials"), key="num_workers", minimum=0
        ),
        "maximum_vram_fraction": float(maximum_vram_fraction),
        "loader_measurement_batches": int(loader_batches),
        "prefetch_factor": int(prefetch_factor),
        "pin_memory": bool(identity.get("pin_memory", True)),
        "training_preflight_sha256": _value_sha256(calibration),
        "training_candidate_statuses": {
            "batch": [{"batch_size": row["batch_size"], "status": row.get("status")}
                      for row in calibration["batch_trials"]],
            "worker": [{"num_workers": row["num_workers"], "status": row.get("status")}
                       for row in calibration["worker_trials"]],
        },
        "candidate_policy": "remeasure_inference_only_after_fresh_host_budget; no cohort-oversize batches",
        "calibration_order": "descending_canonical_input_upper_bound",
    }


def _build_preflight_identity(
    artifact_contract: dict[str, Any],
    measurement_plan: dict[str, Any],
    *,
    device: str | torch.device,
    repeats: int,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    """One identity contract for audit publication, reuse and both bank readers.

    Reconstruct bounds from the complete current cache inventory, not from a
    saved preflight's claims. This retains the exact v2 identity already written
    by the memory-bounded audit; no report/checkpoint migration is necessary.
    Device describes the recorded audit, not the bank verifier's current device.
    """
    repeats = _exact_positive_int(repeats, context="causality preflight repeats")
    if repeats < 3:
        raise ValueError("Causality preflight repeats must be at least 3")
    if not isinstance(device, (str, torch.device)):
        raise ValueError("Causality preflight has no valid recorded device")
    recorded_device = str(torch.device(device))
    if not isinstance(inventory, dict) or not isinstance(inventory.get("selected_files"), list) or not inventory["selected_files"]:
        raise ValueError("Causality preflight requires the complete input inventory")
    selected = [Path(path) for path in inventory["selected_files"]]
    bounds = {
        str(batch): audit_batch_input_bytes(inventory, selected, batch)
        for batch in measurement_plan["batch_candidates"]
    }
    return {
        "format": PREFLIGHT_IDENTITY_FORMAT,
        "artifact_contract_sha256": _value_sha256(artifact_contract),
        "measurement_plan": measurement_plan,
        "device": recorded_device,
        "repeats": repeats,
        "input_inventory_sha256": _value_sha256(inventory),
        "input_inventory_sample_count": len(selected),
        "batch_input_upper_bounds": bounds,
    }


def _run_preflight_workload(
    model: Any,
    cpu_batch: HierarchicalBatch,
    *,
    device: torch.device,
    seed: int,
    progress: Callable[..., None] | None = None,
) -> None:
    if progress is not None:
        progress(event="baseline_start", condition="baseline")
    baseline = _fork_batch_for_audit(cpu_batch)
    try:
        _evaluate(model, baseline, device=device, amp=False)
    finally:
        del baseline
    if progress is not None:
        progress(event="baseline_end", condition="baseline")
    for condition_index, (name, transform) in enumerate(CONDITIONS.items()):
        if progress is not None:
            progress(event="condition_start", condition=name)
        changed = _fork_batch_for_audit(cpu_batch)
        try:
            changed = transform(changed, int(seed) + condition_index * 10_007)
            _evaluate(model, changed, device=device, amp=False)
        finally:
            # model.forward transfers this container to CUDA in place. Do not
            # retain its device tensors during the next full-size intervention.
            del changed
        if progress is not None:
            progress(event="condition_end", condition=name)


def _measure_batch_candidates(
    model: Any,
    selected: list[Path],
    candidates: list[int],
    *,
    repeats: int,
    maximum_vram_fraction: float,
    device: torch.device,
    seed: int,
    inventory: dict[str, Any],
    progress: Callable[..., None],
) -> tuple[int, list[dict[str, Any]]]:
    dataset = HierarchicalCacheDataset(selected, mmap=True, training=False, seed=seed)
    if len(dataset) < 1:
        raise ValueError("Cannot calibrate causality on an empty selected dataset")
    trials: list[dict[str, Any]] = []
    for batch_size in candidates:
        gc.collect()
        progress(event="batch_trial_start", batch_size=batch_size)
        budget = None
        if batch_size > len(selected):
            status = "not_measured_cohort"
        else:
            budget = audit_host_budget(inventory, selected, batch_size, 0, 1, False)
            status = "accepted" if budget["accepted"] else "rejected_host_input_budget"
        progress(event="batch_trial_budget", batch_size=batch_size, status=status, host_budget=budget)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        completed = 0
        measurement = None
        try:
            if status == "accepted":
                with Measurement() as measurement:
                    for repeat in range(repeats):
                        progress(event="batch_repeat_start", batch_size=batch_size, repeat=repeat)
                        samples, batch = [], None
                        try:
                            samples = [
                                dataset[(repeat * batch_size + offset) % len(dataset)]
                                for offset in range(batch_size)
                            ]
                            batch = collate_samples(samples)
                            samples.clear()

                            def workload_progress(**fields):
                                if fields.get("event") in {"baseline_start", "condition_start"}:
                                    _require_host_budget(
                                        inventory, selected, batch_size, 0, 1, False,
                                        progress=progress, phase="resident_transform",
                                    )
                                progress(batch_size=batch_size, repeat=repeat, **fields)

                            _run_preflight_workload(
                                model, batch, device=device,
                                seed=seed + repeat * 100_003, progress=workload_progress,
                            )
                            completed += batch_size
                        finally:
                            samples.clear()
                            batch = None
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            status = "rejected_cuda_oom"
            if device.type == "cuda":
                torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        total = (
            int(torch.cuda.get_device_properties(device).total_memory)
            if device.type == "cuda"
            else None
        )
        peak_fraction = (
            float(peak / total) if peak is not None and total is not None and total > 0 else None
        )
        if status == "accepted" and peak_fraction is not None and peak_fraction > maximum_vram_fraction:
            status = "rejected_vram_headroom"
        trials.append(
            {
                "batch_size": int(batch_size),
                "status": status,
                "host_budget": budget,
                "host_measurement": None if measurement is None else measurement.report,
                "cohort_size": len(selected),
                "repeats": int(repeats),
                "completed_samples": int(completed),
                "elapsed_seconds": float(elapsed),
                "samples_per_second": (
                    float(completed / elapsed) if completed > 0 and elapsed > 0.0 else None
                ),
                "peak_vram_bytes": peak,
                "total_vram_bytes": total,
                "peak_vram_fraction": peak_fraction,
                "vram_headroom_bytes": (
                    int(total - peak) if peak is not None and total is not None else None
                ),
            }
        )
        progress(event="batch_trial_end", **trials[-1])
    accepted = [trial for trial in trials if trial["status"] == "accepted"]
    if not accepted:
        raise RuntimeError(f"No causality physical-batch candidate passed preflight: {trials}")
    winner = max(
        accepted,
        key=lambda trial: (
            float(trial["samples_per_second"]),
            int(trial["batch_size"]),
        ),
    )
    return int(winner["batch_size"]), trials


def _measure_worker_candidates(
    selected: list[Path],
    candidates: list[int],
    *,
    batch_size: int,
    measurement_batches: int,
    pin_memory: bool,
    prefetch_factor: int,
    seed: int,
    inventory: dict[str, Any],
    progress: Callable[..., None],
) -> tuple[int, list[dict[str, Any]]]:
    required_files = max(1, batch_size * measurement_batches)
    repeated = [selected[index % len(selected)] for index in range(required_files)]
    trials: list[dict[str, Any]] = []
    for workers in candidates:
        gc.collect()
        progress(event="worker_trial_start", num_workers=workers, batch_size=batch_size)
        budget = audit_host_budget(
            inventory, selected, batch_size, workers, prefetch_factor, pin_memory,
        )
        progress(event="worker_trial_budget", num_workers=workers, host_budget=budget)
        if not budget["accepted"]:
            trials.append({"num_workers": workers, "status": "rejected_host_input_budget",
                           "host_budget": budget, "host_measurement": None,
                           "measurement_batches": 0, "samples": 0,
                           "elapsed_seconds": 0.0, "samples_per_second": None})
            progress(event="worker_trial_end", **trials[-1])
            continue
        loader_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": workers,
            "collate_fn": collate_samples,
            "pin_memory": pin_memory,
        }
        if workers > 0:
            loader_kwargs.update(
                {"prefetch_factor": prefetch_factor, "persistent_workers": True}
            )
        loader = DataLoader(
            HierarchicalCacheDataset(
                repeated, mmap=True, training=False, seed=seed
            ),
            **loader_kwargs,
        )
        started = time.perf_counter()
        samples = 0
        batches = 0
        iterator = None
        try:
            with Measurement() as measurement:
                iterator = iter(loader)
                for cpu_batch in iterator:
                    try:
                        samples += len(cpu_batch.counts)
                        batches += 1
                        progress(event="worker_batch_end", num_workers=workers, batch_index=batches)
                    finally:
                        del cpu_batch
        finally:
            # Release consumers before destroying the iterator/worker pool.
            # No old batch survives while the next candidate is allocated.
            del iterator, loader
            gc.collect()
        elapsed = time.perf_counter() - started
        if batches != measurement_batches or samples <= 0 or elapsed <= 0.0:
            raise RuntimeError(
                f"DataLoader worker preflight did not complete: workers={workers} "
                f"batches={batches}/{measurement_batches} samples={samples}"
            )
        trials.append(
            {
                "num_workers": int(workers),
                "status": "accepted",
                "host_budget": budget,
                "host_measurement": measurement.report,
                "measurement_batches": int(batches),
                "samples": int(samples),
                "elapsed_seconds": float(elapsed),
                "samples_per_second": float(samples / elapsed),
            }
        )
        progress(event="worker_trial_end", **trials[-1])
    accepted = [row for row in trials if row["status"] == "accepted"]
    if not accepted:
        raise RuntimeError(f"No causality worker candidate fits the current allocation: {trials}")
    winner = max(
        accepted,
        key=lambda trial: (
            float(trial["samples_per_second"]),
            int(trial["num_workers"]),
        ),
    )
    return int(winner["num_workers"]), trials


def _require_host_budget(inventory, selected, batch_size, workers, prefetch_factor,
                         pin_memory, *, progress, phase="full"):
    budget = audit_host_budget(
        inventory, selected, batch_size, workers, prefetch_factor, pin_memory, phase=phase,
    )
    progress(event="host_budget_check", host_budget=budget)
    if not budget["accepted"]:
        raise RuntimeError(
            "Causality host allocation is insufficient before allocation: "
            f"phase={phase}, batch_size={batch_size}, workers={workers}, "
            f"required={budget['estimated_input_bytes']} bytes, "
            f"available={budget['available_allocation_bytes']} bytes (headroom reserved). "
            "No graph/data/model reduction or checkpoint restart was performed. "
            "See the causality progress JSONL allocation evidence."
        )
    return budget


def _validate_preflight_record(
    record: dict[str, Any],
    *,
    identity: dict[str, Any],
    resource_fingerprint: dict[str, Any],
) -> tuple[int, int]:
    if record.get("format") != PREFLIGHT_FORMAT:
        raise ValueError("Causality preflight is legacy or unsupported")
    if record.get("identity") != identity or record.get("identity_sha256") != _value_sha256(identity):
        raise ValueError("Causality preflight identity differs from current inputs")
    if record.get("resource_fingerprint") != resource_fingerprint:
        raise ValueError("Causality preflight hardware/allocation fingerprint changed")
    batch = _exact_positive_int(
        record.get("selected_batch_size"), context="selected causality batch size"
    )
    workers = record.get("selected_num_workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise ValueError("Causality preflight selected_num_workers is invalid")
    batch_trials = record.get("batch_trials")
    worker_trials = record.get("worker_trials")
    expected_batches = identity["measurement_plan"]["batch_candidates"]
    expected_workers = identity["measurement_plan"]["worker_candidates"]
    if not isinstance(batch_trials, list) or any(not isinstance(row, dict) for row in batch_trials) or [row.get("batch_size") for row in batch_trials] != expected_batches:
        raise ValueError("Causality preflight batch trial cohort is not exact")
    if not isinstance(worker_trials, list) or any(not isinstance(row, dict) for row in worker_trials) or [row.get("num_workers") for row in worker_trials] != expected_workers:
        raise ValueError("Causality preflight worker trial cohort is not exact")
    for row in batch_trials:
        _validate_resource_trial(row, kind="batch", identity=identity)
    for row in worker_trials:
        _validate_resource_trial(row, kind="worker", identity=identity, batch_size=batch)
    accepted_batches = [row for row in batch_trials if row.get("status") == "accepted"]
    if not accepted_batches or any(row.get("repeats") != identity["repeats"] for row in batch_trials):
        raise ValueError("Causality preflight batch trials are incomplete")
    winner_batch = max(
        accepted_batches,
        key=lambda row: (float(row["samples_per_second"]), int(row["batch_size"])),
    )
    if int(winner_batch["batch_size"]) != batch:
        raise ValueError("Causality selected batch is not the measured winner")
    accepted_workers = [row for row in worker_trials if row.get("status") == "accepted"]
    if not accepted_workers:
        raise ValueError("Causality preflight worker trials are incomplete")
    winner_workers = max(
        accepted_workers,
        key=lambda row: (float(row["samples_per_second"]), int(row["num_workers"])),
    )
    if int(winner_workers["num_workers"]) != workers:
        raise ValueError("Causality selected workers are not the measured winner")
    return batch, int(workers)


def _validate_resource_trial(row, *, kind, identity, batch_size=None):
    """Reject missing/contradictory allocation evidence, not just bad winners."""
    status = row.get("status")
    allowed = {"accepted", "rejected_host_input_budget"}
    if kind == "batch":
        allowed.update({"not_measured_cohort", "rejected_cuda_oom", "rejected_vram_headroom"})
        batch_size = _exact_positive_int(row.get("batch_size"), context="trial batch size")
        if row.get("repeats") != identity["repeats"]:
            raise ValueError("Causality batch repeat contract changed")
        cohort_size = identity["input_inventory_sample_count"]
        if row.get("cohort_size") != cohort_size:
            raise ValueError("Causality trial cohort size changed")
        if status == "not_measured_cohort":
            if batch_size <= cohort_size or row.get("completed_samples") != 0 or row.get("host_budget") is not None:
                raise ValueError("Causality unmeasured cohort trial is inconsistent")
            return
    if status not in allowed:
        raise ValueError(f"Causality resource trial has unsupported status: {status}")
    budget = row.get("host_budget")
    if not isinstance(budget, dict) or budget.get("phase") != "full":
        raise ValueError("Causality trial lacks full host allocation evidence")
    plan = identity["measurement_plan"]
    workers = 0 if kind == "batch" else row["num_workers"]
    validate_audit_host_budget(
        budget, batch_size=batch_size, workers=workers,
        prefetch_factor=1 if kind == "batch" else plan["prefetch_factor"],
        pin_memory=False if kind == "batch" else bool(
            torch.device(identity["device"]).type == "cuda" and plan["pin_memory"]),
        batch_input_bytes_upper_bound=identity["batch_input_upper_bounds"][str(batch_size)],
    )
    required = budget.get("estimated_input_bytes")
    available = budget.get("available_allocation_bytes")
    for value in (required, available):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("Causality trial host byte estimate is invalid")
    if budget.get("reserved_headroom_fraction") != 0.2:
        raise ValueError("Causality host allocation headroom changed")
    feasible = required <= available * 4 // 5
    if budget.get("accepted") is not feasible or (status == "rejected_host_input_budget") == feasible:
        raise ValueError("Causality trial host rejection contradicts its byte budget")
    if status == "rejected_host_input_budget":
        count = row.get("completed_samples") if kind == "batch" else row.get("samples")
        if count != 0 or row.get("host_measurement") is not None:
            raise ValueError("Host-rejected trial unexpectedly allocated inputs")
        return
    if status == "accepted":
        count = row.get("completed_samples") if kind == "batch" else row.get("samples")
        repetitions = identity["repeats"] if kind == "batch" else identity["measurement_plan"]["loader_measurement_batches"]
        if count != batch_size * repetitions:
            raise ValueError("Accepted causality trial did not finish all measurements")
        if kind == "worker" and row.get("measurement_batches") != repetitions:
            raise ValueError("Accepted causality worker trial is incomplete")
        elapsed, rate = row.get("elapsed_seconds"), row.get("samples_per_second")
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x <= 0
               for x in (elapsed, rate)) or not math.isclose(rate, count / elapsed, rel_tol=1e-9):
            raise ValueError("Causality trial throughput is invalid")
        if not isinstance(row.get("host_measurement"), dict) or row["host_measurement"].get("status") != "complete":
            raise ValueError("Accepted causality trial lacks resource measurements")
        if kind == "batch":
            fraction = row.get("peak_vram_fraction")
            if fraction is not None and (not math.isfinite(fraction) or not 0 <= fraction <= identity["measurement_plan"]["maximum_vram_fraction"]):
                raise ValueError("Accepted causality batch exceeds measured VRAM headroom")


def _resolve_preflight(
    *,
    path: Path,
    identity: dict[str, Any],
    resource_fingerprint: dict[str, Any],
    model: Any,
    selected: list[Path],
    device: torch.device,
    seed: int,
    repeats: int,
    overwrite: bool,
    inventory: dict[str, Any],
    progress: Callable[..., None],
) -> tuple[dict[str, Any], int, int]:
    if path.exists() or path.is_symlink():
        if not path.is_file():
            raise FileExistsError(f"Causality preflight path is not a regular file: {path}")
        if not overwrite:
            try:
                current = _load_json_object(path)
                batch, workers = _validate_preflight_record(
                    current,
                    identity=identity,
                    resource_fingerprint=resource_fingerprint,
                )
            except (OSError, ValueError) as exc:
                raise FileExistsError(
                    f"Existing causality preflight is stale/incompatible: {exc}; "
                    f"pass --overwrite for this exact file: {path}"
                ) from exc
            print(f"[Reuse] verified measured causality preflight: {path}")
            progress(event="preflight_reused", path=str(path), batch_size=batch, num_workers=workers)
            return current, batch, workers
    plan = identity["measurement_plan"]
    batch, batch_trials = _measure_batch_candidates(
        model,
        selected,
        plan["batch_candidates"],
        repeats=repeats,
        maximum_vram_fraction=plan["maximum_vram_fraction"],
        device=device,
        seed=seed,
        inventory=inventory,
        progress=progress,
    )
    workers, worker_trials = _measure_worker_candidates(
        selected,
        plan["worker_candidates"],
        batch_size=batch,
        measurement_batches=plan["loader_measurement_batches"],
        pin_memory=bool(device.type == "cuda" and plan["pin_memory"]),
        prefetch_factor=plan["prefetch_factor"],
        seed=seed,
        inventory=inventory,
        progress=progress,
    )
    record = {
        "format": PREFLIGHT_FORMAT,
        "identity": identity,
        "identity_sha256": _value_sha256(identity),
        "resource_fingerprint": resource_fingerprint,
        "selected_batch_size": batch,
        "selected_num_workers": workers,
        "batch_trials": batch_trials,
        "worker_trials": worker_trials,
    }
    _validate_preflight_record(
        record, identity=identity, resource_fingerprint=resource_fingerprint
    )
    _atomic_json(path, record)
    progress(event="preflight_complete", path=str(path), batch_size=batch, num_workers=workers)
    print(f"[OK] measured causality preflight: {path}")
    return record, batch, workers


def _cache_entries_for_split(
    cache_dir: Path, index: dict[str, Any], split: str
) -> tuple[list[Path], list[dict[str, Any]]]:
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Validated cache index has no entries list")
    wanted = {"train", "val"} if split == "all" else {split}
    selected_entries = sorted(
        (dict(entry) for entry in entries if entry.get("split") in wanted),
        key=lambda entry: str(entry["path"]),
    )
    if not selected_entries:
        raise ValueError(f"Validated cache contains no {split!r} audit entries")
    return [cache_dir / str(entry["path"]) for entry in selected_entries], selected_entries


def _ct_clip_bounds(value: Any, *, context: str) -> tuple[float, float]:
    """Validate the same numeric interval across torch tuples and JSON lists.

    Normalize only the container, not the bounds: no coercion, rounding, or
    tolerance may hide a real preprocessing difference between artifacts.
    """
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{context} ct_clip must contain exactly two numeric bounds: {value!r}")
    for bound in value:
        if (
            isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or (isinstance(bound, float) and not math.isfinite(bound))
        ):
            raise ValueError(f"{context} ct_clip bounds must be finite numbers: {value!r}")
    lower, upper = value
    if lower >= upper:
        raise ValueError(f"{context} ct_clip bounds must be strictly increasing: {value!r}")
    return lower, upper


def _validate_checkpoint_contract(
    checkpoint: dict[str, Any],
    *,
    checkpoint_path: Path,
    prototype_path: Path,
    cache_dir: Path,
    cache_config: dict[str, Any],
    cache_index: dict[str, Any],
    model: Any,
    run_mode: str,
) -> dict[str, Any]:
    if checkpoint.get("method") != "hiercp-full":
        raise ValueError(f"Not a hiercp-full checkpoint: {checkpoint.get('method')}")
    if checkpoint.get("framework") != "torch_geometric":
        raise ValueError("Causality requires a torch_geometric checkpoint")
    if checkpoint.get("training_complete") is not True:
        raise ValueError("Causality requires a completed full training checkpoint")
    target_epochs = _exact_positive_int(
        checkpoint.get("target_epochs"), context="checkpoint target_epochs"
    )
    completed_epoch = _exact_positive_int(
        checkpoint.get("completed_epoch"), context="checkpoint completed_epoch"
    )
    if completed_epoch != target_epochs:
        raise ValueError("Checkpoint completed_epoch does not equal target_epochs")
    signature = checkpoint.get("training_signature")
    if (
        not isinstance(signature, dict)
        or signature.get("format") != "hiercp_training_signature_v1"
        or signature.get("run_mode") != run_mode
        or signature.get("ablation_mode", "full") != "full"
        or signature.get("target_epochs") != target_epochs
    ):
        raise ValueError(
            "Checkpoint does not have an exact completed full-model training signature"
        )
    if cache_config.get("run_mode") != run_mode:
        raise ValueError("Checkpoint and cache run_mode differ from the audit request")
    entries = cache_index.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Validated cache index has no entries")
    train_names = sorted(
        str(entry["path"]) for entry in entries if entry.get("split") == "train"
    )
    val_names = sorted(
        str(entry["path"]) for entry in entries if entry.get("split") == "val"
    )
    if signature.get("train_cache_files") != train_names:
        raise ValueError("Checkpoint training cache cohort differs from publication")
    if signature.get("val_cache_files") != val_names:
        raise ValueError("Checkpoint validation cache cohort differs from publication")
    if checkpoint.get("graph_config") != cache_config.get("graph_config"):
        raise ValueError("Checkpoint graph_config differs from cache publication")
    checkpoint_clip = _ct_clip_bounds(checkpoint.get("ct_clip"), context="Checkpoint")
    cache_clip = _ct_clip_bounds(cache_config.get("ct_clip"), context="Cache publication")
    if checkpoint_clip != cache_clip:
        raise ValueError(
            "Checkpoint ct_clip differs from cache publication: "
            f"checkpoint={checkpoint_clip!r}, cache={cache_clip!r}"
        )
    configured_prototype = Path(
        str(cache_config.get("prototype_bank", ""))
    ).expanduser().resolve()
    if configured_prototype != prototype_path:
        raise ValueError(
            "Requested prototype path differs from the cache publication contract"
        )
    if not prototype_path.is_file():
        raise FileNotFoundError(f"Prototype artifact is missing: {prototype_path}")
    if _file_sha256(prototype_path) != cache_config.get("prototype_artifact_sha256"):
        raise ValueError("Prototype artifact SHA-256 differs from cache publication")
    prototype = PrototypeBank.load(prototype_path)
    train_ids = cache_config.get("train_case_ids")
    if not isinstance(train_ids, list) or not train_ids:
        raise ValueError("Cache publication has no exact training case cohort")
    if list(prototype.training_case_ids) != train_ids:
        raise ValueError("Prototype-bank training cohort differs from cache split")
    if prototype.fingerprint() != cache_config.get("prototype_fingerprint"):
        raise ValueError("Prototype-bank fingerprint differs from cache publication")
    if checkpoint.get("prototype_fingerprint") != cache_config.get("prototype_fingerprint"):
        raise ValueError("Checkpoint prototype fingerprint differs from cache publication")
    if checkpoint.get("prototype_training_cases") != train_ids:
        raise ValueError("Checkpoint prototype training cohort differs from cache split")
    connectivity = checkpoint.get("gradient_connectivity")
    expected_parameters = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not expected_parameters:
        raise ValueError("Checkpoint model exposes no trainable parameters")
    if (
        not isinstance(connectivity, dict)
        or connectivity.get("format") != "hiercp_gradient_connectivity_v1"
        or connectivity.get("verified") is not True
        or connectivity.get("expected_parameter_count") != len(expected_parameters)
        or connectivity.get("connected_parameter_count") != len(expected_parameters)
        or connectivity.get("connected_parameters") != expected_parameters
        or connectivity.get("missing_parameters") != []
    ):
        raise ValueError(
            "Checkpoint gradient connectivity does not exactly match trainable parameters"
        )
    return signature


def _artifact_contract(
    *,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    prototype_path: Path,
    signature: dict[str, Any],
    cache_dir: Path,
    cache_config: dict[str, Any],
    selected_entries: list[dict[str, Any]],
    run_mode: str,
    split: str,
    max_batches: int,
    seed: int,
    permutation_tolerance: float,
    response_threshold: float,
    strict: bool,
) -> dict[str, Any]:
    return {
        "format": "hiercp_causality_artifacts_v3",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _file_sha256(checkpoint_path),
            "method": checkpoint["method"],
            "framework": checkpoint["framework"],
            "completed_epoch": checkpoint["completed_epoch"],
            "target_epochs": checkpoint["target_epochs"],
            "training_signature_sha256": _value_sha256(signature),
            "gradient_connectivity_sha256": _value_sha256(
                checkpoint["gradient_connectivity"]
            ),
        },
        "prototype": {
            "path": str(prototype_path),
            "sha256": _file_sha256(prototype_path),
            "fingerprint": cache_config["prototype_fingerprint"],
            "training_case_ids": list(cache_config["train_case_ids"]),
        },
        "cache": {
            "path": str(cache_dir),
            "config_sha256": _file_sha256(cache_dir / "config.json"),
            "index_sha256": _file_sha256(cache_dir / "index.json"),
            "manifest_sha256": _file_sha256(cache_dir / "manifest.csv"),
            "complete_sha256": _file_sha256(cache_dir / "complete.json"),
            "config_fingerprint": cache_config["config_fingerprint"],
            "run_mode": cache_config["run_mode"],
            "train_case_ids": list(cache_config["train_case_ids"]),
            "val_case_ids": list(cache_config["val_case_ids"]),
            "selected_case_ids": list(cache_config["selected_case_ids"]),
            "selected_entries": selected_entries,
        },
        "audit": {
            "run_mode": run_mode,
            "split": split,
            "max_batches": int(max_batches),
            "subset_active": bool(max_batches > 0),
            "seed": int(seed),
            "strict": bool(strict),
            "conditions": list(CONDITIONS),
            "precision": "float32",
            "permutation_tolerance": float(permutation_tolerance),
            "response_threshold": float(response_threshold),
        },
    }


def _input_contract(
    artifact_contract: dict[str, Any],
    *,
    preflight_path: Path,
    preflight: dict[str, Any],
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    return {
        "format": INPUT_FORMAT,
        "artifacts": artifact_contract,
        "artifacts_sha256": _value_sha256(artifact_contract),
        "execution": {
            "physical_batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "preflight_path": str(preflight_path),
            "preflight_artifact_sha256": _file_sha256(preflight_path),
            "preflight_contract_sha256": _value_sha256(preflight),
            "preflight_resource_fingerprint": preflight["resource_fingerprint"],
        },
    }


def _strict_failures(verdict: dict[str, Any]) -> list[str]:
    checks = (
        ("permutation_invariant", "node-order invariance"),
        ("upper_position_shortcut_blocked", "upper-position shortcut invariance"),
        ("upper_clearance_shortcut_blocked", "upper-clearance shortcut invariance"),
        ("shortcut_safety_supported", "shortcut-safety composite verdict"),
        ("target_context_sensitive", "target-context response"),
        ("context_causality_supported", "context-causality composite verdict"),
    )
    failures = [label for key, label in checks if verdict.get(key) is not True]
    if not (
        verdict.get("spatial_edge_sensitive") is True
        or verdict.get("topology_sensitive") is True
    ):
        failures.append("spatial graph response")
    return failures


def _validate_reusable_report(
    report: dict[str, Any],
    *,
    input_contract: dict[str, Any],
    expected_samples: int,
    expected_batches: int,
) -> None:
    if report.get("format") != REPORT_FORMAT:
        raise ValueError("existing report is legacy or has an unsupported format")
    if report.get("input_contract") != input_contract:
        raise ValueError("existing report input contract differs from current artifacts")
    if report.get("input_contract_sha256") != _value_sha256(input_contract):
        raise ValueError("existing report input-contract hash is invalid")
    if report.get("evaluated_batches") != expected_batches:
        raise ValueError("existing report evaluated-batch count is inconsistent")
    clean = report.get("clean")
    conditions = report.get("conditions")
    if not isinstance(clean, dict) or clean.get("samples") != expected_samples:
        raise ValueError("existing report sample count is inconsistent")
    if not isinstance(conditions, dict) or set(conditions) != set(CONDITIONS):
        raise ValueError("existing report condition cohort is not exact")
    if any(
        not isinstance(conditions[name], dict)
        or conditions[name].get("samples") != expected_samples
        for name in CONDITIONS
    ):
        raise ValueError("existing report condition sample counts are inconsistent")
    verdict = report.get("verdict")
    if not isinstance(verdict, dict):
        raise ValueError("existing report has no verdict")
    failures = _strict_failures(verdict)
    expected_status = "complete" if not failures else "failed"
    if (
        report.get("status") != expected_status
        or report.get("strict_pass") is not (not failures)
    ):
        raise ValueError("existing report status does not match its strict verdict")


def _jaccard(first: set[Any], second: set[Any]) -> float:
    union = first | second
    return 1.0 if not union else float(len(first & second) / len(union))


def _score_metrics(scores: list[Tensor]) -> dict[str, float]:
    if not scores:
        raise ValueError("No score vectors were produced")
    top1 = 0.0
    rr = 0.0
    positive = 0.0
    hardest = 0.0
    margin = 0.0
    for score in scores:
        values = score.float().cpu()
        if values.ndim != 1 or values.numel() < 2:
            raise ValueError(f"Bad score vector: {tuple(values.shape)}")
        best_negative = values[1:].max()
        rank = 1 + int(torch.sum(values[1:] >= values[0]))
        top1 += float(rank == 1)
        rr += 1.0 / float(rank)
        positive += float(values[0])
        hardest += float(best_negative)
        margin += float(values[0] - best_negative)
    count = float(len(scores))
    return {
        "samples": int(len(scores)),
        "top1": top1 / count,
        "mrr": rr / count,
        "positive": positive / count,
        "hardest_negative": hardest / count,
        "margin": margin / count,
    }


def _merge_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    total = sum(int(row["samples"]) for row in rows)
    if total <= 0:
        raise ValueError("Cannot merge empty metric rows")
    result: dict[str, float] = {"samples": int(total)}
    for key in ("top1", "mrr", "positive", "hardest_negative", "margin"):
        result[key] = sum(float(row[key]) * int(row["samples"]) for row in rows) / total
    return result


def _detach_output(output) -> dict[str, Any]:
    return {
        "scores": [score.detach().float().cpu() for score in output.scores],
        "embeddings": {
            key: output.local_embeddings[key].detach().float().cpu()
            for key in EMBEDDING_KEYS
        },
    }


def _evaluate(model, batch: HierarchicalBatch, *, device: torch.device, amp: bool) -> dict[str, Any]:
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
        output = model(batch)
    return _detach_output(output)


def _fork_batch_for_audit(batch: HierarchicalBatch) -> HierarchicalBatch:
    """Isolate mutable containers while sharing read-only input tensors.

    Every intervention replaces its changed tensors out of place. PyG .to()
    does mutate store mappings, so independently copy every graph container as
    well as the outer batch; sharing only tensor storage is intentional.
    """
    result = copy.copy(batch)
    for name in ("local_batch", "local_batch_view2", "patient_batch", "prototype_batch"):
        graph = getattr(batch, name)
        setattr(result, name, None if graph is None else copy.copy(graph))
    return result


def _seeded_perm(count: int, seed: int) -> Tensor:
    if count <= 1:
        return torch.arange(count, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(int(seed) & 0x7FFFFFFF)
    perm = torch.randperm(count, generator=generator)
    if bool(torch.equal(perm, torch.arange(count))):
        perm = torch.roll(perm, shifts=1)
    return perm


def _permute_graph_nodes(graph: HeteroData, seed: int) -> HeteroData:
    graph = copy.copy(graph)
    inverse: dict[str, Tensor] = {}
    for type_index, node_type in enumerate(graph.node_types):
        count = int(graph[node_type].num_nodes)
        perm = _seeded_perm(count, seed + 1009 * (type_index + 1))
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(count, dtype=torch.long)
        inverse[node_type] = inv
        for key, value in list(graph[node_type].items()):
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == count:
                graph[node_type][key] = value[perm]

    for edge_type in graph.edge_types:
        source_type, _, destination_type = edge_type
        edge_index = graph[edge_type].edge_index
        graph[edge_type].edge_index = torch.stack(
            [
                inverse[source_type][edge_index[0]],
                inverse[destination_type][edge_index[1]],
            ],
            dim=0,
        )
    return graph


def _rotate_context_features(
    graph: HeteroData,
    node_types: tuple[str, ...],
    seed: int,
) -> HeteroData:
    graph = copy.copy(graph)
    for type_index, node_type in enumerate(node_types):
        if node_type not in graph.node_types:
            continue
        count = int(graph[node_type].num_nodes)
        perm = _seeded_perm(count, seed + 2017 * (type_index + 1))
        for key in ("x", "grid"):
            value = graph[node_type].get(key)
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == count:
                graph[node_type][key] = value[perm]
    return graph


def _zero_edge_attributes(graph: HeteroData) -> HeteroData:
    graph = copy.copy(graph)
    for edge_type in graph.edge_types:
        edge_attr = graph[edge_type].get("edge_attr")
        if torch.is_tensor(edge_attr):
            graph[edge_type].edge_attr = torch.zeros_like(edge_attr)
    return graph


def _shuffle_topology(graph: HeteroData, seed: int) -> HeteroData:
    graph = copy.copy(graph)
    for relation_index, edge_type in enumerate(graph.edge_types):
        edge_index = graph[edge_type].edge_index
        edge_count = int(edge_index.shape[1])
        if edge_count <= 1:
            continue
        perm = _seeded_perm(edge_count, seed + 3011 * (relation_index + 1))
        corrupted = edge_index.clone()
        # Keep the source incidence and relation size while assigning each
        # message to a different destination. Existing edge_attr is deliberately
        # left attached to its old edge, breaking topology/geometry agreement.
        corrupted[1] = edge_index[1, perm]
        graph[edge_type].edge_index = corrupted
    return graph


def _map_local_batch(
    batch: Batch | None,
    transform: Callable[[HeteroData, int], HeteroData],
) -> Batch | None:
    if batch is None:
        return None
    # PyG separates x/grid/edge_attr as views into the complete batch storage.
    # Deep-copying each such graph would copy that complete backing storage
    # once per candidate. Helpers use independent stores and replace only the
    # affected tensors, preserving untouched views until linear-size collation.
    graphs = batch.to_data_list()
    return Batch.from_data_list(
        [transform(graph, index) for index, graph in enumerate(graphs)]
    )


def _rotate_target_patches(batch: HierarchicalBatch) -> None:
    source = batch.target_patches
    rotated = source.clone()
    start = 0
    for count in batch.counts:
        stop = start + int(count)
        ids = torch.arange(start, stop, dtype=torch.long)
        rotated[start:stop] = source[torch.roll(ids, shifts=1)]
        start = stop
    batch.target_patches = rotated


def _noise_columns(value: Tensor, columns: tuple[int, ...], seed: int) -> Tensor:
    if value.ndim != 2:
        raise ValueError(f"Expected a matrix for causality noise, got {tuple(value.shape)}")
    output = value.clone()
    count = int(value.shape[0]) * len(columns)
    base = torch.arange(count, device=value.device, dtype=torch.float32)
    noise = torch.sin(base + float(int(seed) % 10_007)) * 1.75
    # Empty relations are valid. Explicit dimensions preserve (0, K) without
    # inferring an ambiguous -1 axis; populated relations keep identical noise.
    output[:, list(columns)] = noise.reshape(int(value.shape[0]), len(columns)).to(value.dtype)
    return output


def _condition_upper_position_noise(
    batch: HierarchicalBatch, seed: int
) -> HierarchicalBatch:
    for offset, (graph, node_type) in enumerate(
        (
            (batch.patient_batch, "tumor"),
            (batch.patient_batch, "candidate"),
            (batch.prototype_batch, "candidate"),
        )
    ):
        graph[node_type].raw_x = _noise_columns(
            graph[node_type].raw_x, UPPER_POSITION_COLUMNS, seed + offset * 101
        )
        if "pos" in graph[node_type]:
            graph[node_type].pos = _noise_columns(
                graph[node_type].pos, (0, 1, 2), seed + offset * 103
            )
    for relation_index, edge_type in enumerate(batch.patient_batch.edge_types):
        if "tumor" not in (edge_type[0], edge_type[2]):
            continue
        edge_attr = batch.patient_batch[edge_type].get("edge_attr")
        if torch.is_tensor(edge_attr):
            batch.patient_batch[edge_type].edge_attr = _noise_columns(
                edge_attr,
                PATIENT_POSITION_EDGE_COLUMNS,
                seed + 10_000 + relation_index * 107,
            )
    return batch


def _condition_upper_clearance_noise(
    batch: HierarchicalBatch, seed: int
) -> HierarchicalBatch:
    column = (UPPER_OCCUPIED_DISTANCE_INDEX,)
    for offset, (graph, node_type) in enumerate(
        (
            (batch.patient_batch, "tumor"),
            (batch.patient_batch, "candidate"),
            (batch.prototype_batch, "candidate"),
        )
    ):
        graph[node_type].raw_x = _noise_columns(
            graph[node_type].raw_x, column, seed + offset * 109
        )
    return batch


def _condition_node_order(batch: HierarchicalBatch, seed: int) -> HierarchicalBatch:
    batch.local_batch = _map_local_batch(
        batch.local_batch,
        lambda graph, index: _permute_graph_nodes(graph, seed + index * 17),
    )
    batch.local_batch_view2 = _map_local_batch(
        batch.local_batch_view2,
        lambda graph, index: _permute_graph_nodes(graph, seed + 500_000 + index * 17),
    )
    return batch


def _condition_target_context(batch: HierarchicalBatch, seed: int) -> HierarchicalBatch:
    _rotate_target_patches(batch)
    batch.local_batch = _map_local_batch(
        batch.local_batch,
        lambda graph, index: _rotate_context_features(
            graph, TARGET_CONTEXT_TYPES, seed + index * 19
        ),
    )
    batch.local_batch_view2 = _map_local_batch(
        batch.local_batch_view2,
        lambda graph, index: _rotate_context_features(
            graph, TARGET_CONTEXT_TYPES, seed + 600_000 + index * 19
        ),
    )
    return batch


def _condition_source_context(batch: HierarchicalBatch, seed: int) -> HierarchicalBatch:
    batch.local_batch = _map_local_batch(
        batch.local_batch,
        lambda graph, index: _rotate_context_features(
            graph, SOURCE_CONTEXT_TYPES, seed + index * 23
        ),
    )
    batch.local_batch_view2 = _map_local_batch(
        batch.local_batch_view2,
        lambda graph, index: _rotate_context_features(
            graph, SOURCE_CONTEXT_TYPES, seed + 700_000 + index * 23
        ),
    )
    return batch


def _condition_edge_attr_zero(batch: HierarchicalBatch, seed: int) -> HierarchicalBatch:
    del seed
    batch.local_batch = _map_local_batch(
        batch.local_batch, lambda graph, index: _zero_edge_attributes(graph)
    )
    batch.local_batch_view2 = _map_local_batch(
        batch.local_batch_view2, lambda graph, index: _zero_edge_attributes(graph)
    )
    return batch


def _condition_topology(batch: HierarchicalBatch, seed: int) -> HierarchicalBatch:
    batch.local_batch = _map_local_batch(
        batch.local_batch,
        lambda graph, index: _shuffle_topology(graph, seed + index * 29),
    )
    batch.local_batch_view2 = _map_local_batch(
        batch.local_batch_view2,
        lambda graph, index: _shuffle_topology(
            graph, seed + 800_000 + index * 29
        ),
    )
    return batch


def _condition_view1_only(batch: HierarchicalBatch, seed: int) -> HierarchicalBatch:
    del seed
    batch.local_batch_view2 = None
    return batch


CONDITIONS: dict[str, Callable[[HierarchicalBatch, int], HierarchicalBatch]] = {
    "node_order": _condition_node_order,
    "upper_position_noise": _condition_upper_position_noise,
    "upper_clearance_noise": _condition_upper_clearance_noise,
    "target_context": _condition_target_context,
    "source_context": _condition_source_context,
    "edge_attr_zero": _condition_edge_attr_zero,
    "topology_shuffle": _condition_topology,
    "view1_only": _condition_view1_only,
}


def _view_overlap(batch: HierarchicalBatch, *, work_dir: Path | None = None) -> dict[str, list[float]]:
    result = {
        "source_context_jaccard": [],
        "target_context_jaccard": [],
        "edge_jaccard": [],
        "source_context_fraction": [],
        "target_context_fraction": [],
    }
    if batch.local_batch_view2 is None:
        return result
    first = batch.local_batch.to_data_list()
    second = batch.local_batch_view2.to_data_list()
    if len(first) != len(second):
        raise RuntimeError("Local sampled views contain different graph counts")
    source_index = LOCAL_NODE_TYPES.index("source_context")
    target_index = LOCAL_NODE_TYPES.index("target_context")
    for graph1, graph2 in zip(first, second):
        source1 = set(int(value) for value in graph1["source_context"].full_id.tolist())
        source2 = set(int(value) for value in graph2["source_context"].full_id.tolist())
        target1 = set(int(value) for value in graph1["target_context"].full_id.tolist())
        target2 = set(int(value) for value in graph2["target_context"].full_id.tolist())
        result["source_context_jaccard"].append(_jaccard(source1, source2))
        result["target_context_jaccard"].append(_jaccard(target1, target2))
        result["edge_jaccard"].append(
            exact_local_edge_jaccard(graph1, graph2, temporary_directory=work_dir)
        )
        canonical = graph1.canonical_counts.reshape(-1).to(torch.float64)
        sampled = graph1.sampled_counts.reshape(-1).to(torch.float64)
        result["source_context_fraction"].append(
            float(sampled[source_index] / canonical[source_index].clamp_min(1.0))
        )
        result["target_context_fraction"].append(
            float(sampled[target_index] / canonical[target_index].clamp_min(1.0))
        )
    return result


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _condition_delta(clean: dict[str, Any], changed: dict[str, Any]) -> dict[str, Any]:
    clean_scores = torch.cat(clean["scores"])
    changed_scores = torch.cat(changed["scores"])
    if clean_scores.shape != changed_scores.shape:
        raise RuntimeError("Causality intervention changed candidate score shape")
    clean_metrics = _score_metrics(clean["scores"])
    changed_metrics = _score_metrics(changed["scores"])
    embedding_cosine = {}
    for key in EMBEDDING_KEYS:
        first = clean["embeddings"][key]
        second = changed["embeddings"][key]
        if first.shape != second.shape:
            raise RuntimeError(f"Embedding shape changed for {key}")
        embedding_cosine[key] = float(
            F.cosine_similarity(first, second, dim=-1).mean()
        )
    return {
        "score_abs_sum": float(torch.abs(clean_scores - changed_scores).sum()),
        "score_count": int(clean_scores.numel()),
        "score_max_error": float(torch.abs(clean_scores - changed_scores).max()),
        "sample_count": int(clean_metrics["samples"]),
        "positive_delta_sum": (
            changed_metrics["positive"] - clean_metrics["positive"]
        )
        * int(clean_metrics["samples"]),
        "margin_drop_sum": (
            clean_metrics["margin"] - changed_metrics["margin"]
        )
        * int(clean_metrics["samples"]),
        "embedding_cosine_sum": {
            key: embedding_cosine[key] * int(first.shape[0])
            for key, first in clean["embeddings"].items()
        },
        "embedding_count": {
            key: int(first.shape[0]) for key, first in clean["embeddings"].items()
        },
    }


def _merge_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    score_count = sum(int(row["score_count"]) for row in rows)
    sample_count = sum(int(row["sample_count"]) for row in rows)
    result = {
        "mean_abs_score_delta": sum(float(row["score_abs_sum"]) for row in rows)
        / max(1, score_count),
        "max_abs_score_delta": max(
            (float(row["score_max_error"]) for row in rows), default=0.0
        ),
        "positive_score_delta": sum(
            float(row["positive_delta_sum"]) for row in rows
        )
        / max(1, sample_count),
        "margin_drop": sum(float(row["margin_drop_sum"]) for row in rows)
        / max(1, sample_count),
        "embedding_cosine": {},
    }
    for key in EMBEDDING_KEYS:
        count = sum(int(row["embedding_count"][key]) for row in rows)
        result["embedding_cosine"][key] = sum(
            float(row["embedding_cosine_sum"][key]) for row in rows
        ) / max(1, count)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prototype-bank", required=True)
    parser.add_argument(
        "--run-mode", choices=("production", "benchmark"), required=True
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Assert the physical batch selected by measured causality preflight",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Assert the worker count selected by measured causality preflight",
    )
    parser.add_argument("--preflight-repeats", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--permutation-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--response-threshold", type=float, default=1.0e-4)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _audit_loader(model, loader, *, device, amp, seed, max_batches, work_dir, progress):
    """Evaluate every selected batch with independent, copy-on-write inputs."""
    clean_rows = []
    condition_rows = {name: [] for name in CONDITIONS}
    delta_rows = {name: [] for name in CONDITIONS}
    overlap_rows = {name: [] for name in (
        "source_context_jaccard", "target_context_jaccard", "edge_jaccard",
        "source_context_fraction", "target_context_fraction",
    )}
    # enumerate retains its last result tuple while fetching the next input.
    # Keep the index separately so a multi-GB old batch cannot overlap the fetch.
    batch_index = 0
    for cpu_batch in loader:
        if max_batches and batch_index >= max_batches:
            del cpu_batch
            break
        clean = None
        try:
            progress(event="audit_batch_start", batch_index=batch_index)
            progress(event="audit_overlap_start", batch_index=batch_index)
            overlap = _view_overlap(cpu_batch, work_dir=work_dir)
            for key, values in overlap.items():
                overlap_rows[key].extend(values)
            del overlap
            progress(event="audit_overlap_end", batch_index=batch_index)
            progress(event="baseline_start", batch_index=batch_index, condition="baseline")
            baseline = _fork_batch_for_audit(cpu_batch)
            try:
                clean = _evaluate(model, baseline, device=device, amp=amp)
            finally:
                del baseline
            clean_rows.append(_score_metrics(clean["scores"]))
            progress(event="baseline_end", batch_index=batch_index, condition="baseline")
            for condition_index, (name, transform) in enumerate(CONDITIONS.items()):
                progress(event="condition_start", batch_index=batch_index, condition=name)
                changed_batch = _fork_batch_for_audit(cpu_batch)
                changed = None
                try:
                    changed_batch = transform(
                        changed_batch, int(seed) + batch_index * 100_003 + condition_index * 10_007,
                    )
                    changed = _evaluate(model, changed_batch, device=device, amp=amp)
                    condition_rows[name].append(_score_metrics(changed["scores"]))
                    delta_rows[name].append(_condition_delta(clean, changed))
                finally:
                    del changed, changed_batch
                progress(event="condition_end", batch_index=batch_index, condition=name)
            progress(event="audit_batch_end", batch_index=batch_index)
        finally:
            del clean, cpu_batch
        batch_index += 1
    return clean_rows, condition_rows, delta_rows, overlap_rows


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output).expanduser().resolve()
    with create_progress_log(output) as progress:
        print(f"[CausalityProgressFile] {progress.path}", flush=True)
        progress(event="invocation_start", output=str(output), device=args.device)
        _execute_audit(args, progress)
        progress(event="invocation_complete")


def _execute_audit(args: argparse.Namespace, progress: Any) -> None:
    progress_path = str(progress.path)
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch-size assertion must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("num-workers assertion must be non-negative")
    if args.preflight_repeats < 3:
        raise ValueError("preflight-repeats must be at least 3")
    if args.max_batches < 0:
        raise ValueError("max-batches must be non-negative")
    if args.strict and args.max_batches != 0:
        raise ValueError("--strict requires the complete split; max-batches must be 0")

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    prototype_path = Path(args.prototype_bank).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    preflight_path = output.with_name(output.name + ".preflight.json")
    existing_report: dict[str, Any] | None = None
    if output.exists() or output.is_symlink():
        if not output.is_file():
            raise FileExistsError(f"Causality output is not a regular file: {output}")
        if not args.overwrite:
            existing_report = _load_json_object(output)
            if existing_report.get("format") != REPORT_FORMAT:
                raise FileExistsError(
                    f"Existing causality report is legacy/incompatible; pass --overwrite "
                    f"for this exact file: {output}"
                )
            if not preflight_path.is_file():
                raise FileExistsError(
                    "Existing causality report has no measured preflight sidecar; "
                    f"pass --overwrite for: {output}"
                )
    if args.overwrite:
        print("[OverwriteScope] causality overwrite is limited to:")
        print(f"  - {preflight_path}")
        print(f"  - {output}")
    cache_index = validate_cache_publication(cache_dir)
    cache_config = _load_json_object(cache_dir / "config.json")
    # Loading optimizer/training metadata onto the GPU creates an unnecessary
    # second resident checkpoint. Only model parameters are copied to device.
    payload = torch_load_compat(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Invalid HierCP checkpoint/state: {checkpoint_path}")
    if payload.get("upper_feature_policy") != UPPER_FEATURE_POLICY:
        raise ValueError(
            "Checkpoint predates the shortcut-safe upper feature policy"
        )
    model_kwargs = payload.get("model_kwargs")
    if not isinstance(model_kwargs, dict):
        raise ValueError("Checkpoint has no model_kwargs")

    configure_runtime(
        deterministic=True,
        allow_tf32=False,
        cudnn_benchmark=False,
    )
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    model = HierarchicalPyGPlacementModel(**model_kwargs).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    amp = False  # FP32 audit: AMP rounding must not decide permutation invariance

    signature = _validate_checkpoint_contract(
        payload,
        checkpoint_path=checkpoint_path,
        prototype_path=prototype_path,
        cache_dir=cache_dir,
        cache_config=cache_config,
        cache_index=cache_index,
        model=model,
        run_mode=args.run_mode,
    )
    selected, selected_entries = _cache_entries_for_split(
        cache_dir, cache_index, args.split
    )
    tolerance = float(args.permutation_tolerance)
    response = float(args.response_threshold)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("permutation-tolerance must be finite and non-negative")
    if not math.isfinite(response) or response <= 0.0:
        raise ValueError("response-threshold must be finite and positive")
    artifact_contract = _artifact_contract(
        checkpoint_path=checkpoint_path,
        checkpoint=payload,
        prototype_path=prototype_path,
        signature=signature,
        cache_dir=cache_dir,
        cache_config=cache_config,
        selected_entries=selected_entries,
        run_mode=args.run_mode,
        split=args.split,
        max_batches=args.max_batches,
        seed=args.seed,
        permutation_tolerance=tolerance,
        response_threshold=response,
        strict=args.strict,
    )
    measurement_plan = _training_measurement_plan(
        payload,
        checkpoint_path=checkpoint_path,
        cache_dir=cache_dir,
        run_mode=args.run_mode,
    )
    progress(event="input_inventory_start", selected_samples=len(selected))
    inventory = build_audit_inventory(selected)
    progress(event="input_inventory_complete", inventory=inventory,
             model_kwargs=model_kwargs,
             total_parameters=sum(p.numel() for p in model.parameters()),
             trainable_parameters=sum(p.numel() for p in model.parameters()))
    input_sizes = {row["cache_file"]: row["input_bytes_upper_bound"] for row in inventory["rows"]}
    calibration_selected = sorted(selected, key=lambda path: (-input_sizes[path.name], path.name))
    resources = collect_runtime_resources(device, storage_path=cache_dir)
    resource_fingerprint = _stable_resource_fingerprint(resources, device=device)
    resource_fingerprint["audit_allocation"] = audit_allocation_fingerprint()
    preflight_identity = _build_preflight_identity(
        artifact_contract, measurement_plan, device=device,
        repeats=int(args.preflight_repeats), inventory=inventory,
    )
    preflight, batch_size, num_workers = _resolve_preflight(
        path=preflight_path,
        identity=preflight_identity,
        resource_fingerprint=resource_fingerprint,
        model=model,
        selected=calibration_selected,
        device=device,
        seed=int(args.seed),
        repeats=int(args.preflight_repeats),
        overwrite=bool(args.overwrite),
        inventory=inventory,
        progress=progress,
    )
    if args.batch_size is not None and args.batch_size != batch_size:
        raise ValueError(
            "--batch-size must equal the measured causality value: "
            f"requested={args.batch_size} measured={batch_size}"
        )
    if args.num_workers is not None and args.num_workers != num_workers:
        raise ValueError(
            "--num-workers must equal the measured causality value: "
            f"requested={args.num_workers} measured={num_workers}"
        )
    input_contract = _input_contract(
        artifact_contract,
        preflight_path=preflight_path,
        preflight=preflight,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    selected_limit = (
        len(selected)
        if args.max_batches == 0
        else min(len(selected), int(args.max_batches) * batch_size)
    )
    expected_batches = math.ceil(selected_limit / batch_size)
    if existing_report is not None:
        try:
            _validate_reusable_report(
                existing_report,
                input_contract=input_contract,
                expected_samples=selected_limit,
                expected_batches=expected_batches,
            )
        except (OSError, ValueError) as exc:
            raise FileExistsError(
                f"Existing causality report is stale/incompatible: {exc}; "
                f"pass --overwrite for this exact file: {output}"
            ) from exc
        print(f"[Reuse] verified hash-bound causality report: {output}")
        failures = _strict_failures(existing_report["verdict"])
        if args.strict and failures:
            raise SystemExit("[FAIL] causality gate: " + ", ".join(failures))
        return

    _require_host_budget(
        inventory, selected, batch_size, num_workers, measurement_plan["prefetch_factor"],
        bool(device.type == "cuda" and measurement_plan["pin_memory"]), progress=progress,
    )
    progress(event="audit_loader_start", batch_size=batch_size, num_workers=num_workers,
             prefetch_factor=measurement_plan["prefetch_factor"], selected_samples=len(selected))
    base_progress = progress

    def runtime_progress(**fields):
        if fields.get("event") in {"baseline_start", "condition_start", "audit_overlap_start"}:
            _require_host_budget(
                inventory, selected, batch_size, num_workers, measurement_plan["prefetch_factor"],
                bool(device.type == "cuda" and measurement_plan["pin_memory"]),
                progress=base_progress, phase="resident_transform",
            )
        base_progress(**fields)

    progress = runtime_progress
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_samples,
        "pin_memory": bool(
            device.type == "cuda" and measurement_plan["pin_memory"]
        ),
    }
    if num_workers > 0:
        loader_kwargs.update(
            {
                "prefetch_factor": measurement_plan["prefetch_factor"],
                "persistent_workers": True,
            }
        )
    loader = DataLoader(
        HierarchicalCacheDataset(
            selected,
            mmap=True,
            training=False,
            seed=int(args.seed),
        ),
        **loader_kwargs,
    )

    clean_rows, condition_rows, delta_rows, overlap_rows = _audit_loader(
        model, loader, device=device, amp=amp, seed=int(args.seed),
        max_batches=int(args.max_batches), work_dir=output.parent, progress=progress,
    )

    if not clean_rows:
        raise RuntimeError("No audit batches were evaluated")

    clean_metrics = _merge_metric_rows(clean_rows)
    conditions: dict[str, Any] = {}
    for name in CONDITIONS:
        metrics = _merge_metric_rows(condition_rows[name])
        delta = _merge_deltas(delta_rows[name])
        conditions[name] = {**metrics, **delta}

    overlap_summary = {
        key: {
            "mean": _mean(values),
            "min": min(values) if values else float("nan"),
            "max": max(values) if values else float("nan"),
            "count": len(values),
        }
        for key, values in overlap_rows.items()
    }

    node_invariant = (
        conditions["node_order"]["max_abs_score_delta"] <= tolerance
    )
    upper_position_invariant = (
        conditions["upper_position_noise"]["max_abs_score_delta"] <= tolerance
    )
    upper_clearance_invariant = (
        conditions["upper_clearance_noise"]["max_abs_score_delta"] <= tolerance
    )
    target_context_responds = (
        conditions["target_context"]["mean_abs_score_delta"] >= response
    )
    source_context_responds = (
        conditions["source_context"]["mean_abs_score_delta"] >= response
    )
    edge_attr_responds = (
        conditions["edge_attr_zero"]["mean_abs_score_delta"] >= response
    )
    topology_responds = (
        conditions["topology_shuffle"]["mean_abs_score_delta"] >= response
    )
    context_margin_support = (
        conditions["target_context"]["margin_drop"] > 0.0
        or conditions["source_context"]["margin_drop"] > 0.0
    )
    spatial_margin_support = (
        conditions["edge_attr_zero"]["margin_drop"] > 0.0
        or conditions["topology_shuffle"]["margin_drop"] > 0.0
    )
    verdict = {
        "permutation_invariant": bool(node_invariant),
        "upper_position_shortcut_blocked": bool(upper_position_invariant),
        "upper_clearance_shortcut_blocked": bool(upper_clearance_invariant),
        "shortcut_safety_supported": bool(
            upper_position_invariant and upper_clearance_invariant
        ),
        "target_context_sensitive": bool(target_context_responds),
        "source_context_sensitive": bool(source_context_responds),
        "spatial_edge_sensitive": bool(edge_attr_responds),
        "topology_sensitive": bool(topology_responds),
        "context_corruption_reduces_margin": bool(context_margin_support),
        "spatial_corruption_reduces_margin": bool(spatial_margin_support),
        "context_causality_supported": bool(
            node_invariant
            and target_context_responds
            and (edge_attr_responds or topology_responds)
            and (context_margin_support or spatial_margin_support)
        ),
    }
    strict_failures = _strict_failures(verdict)

    report = {
        "format": REPORT_FORMAT,
        "status": "complete" if not strict_failures else "failed",
        "strict_pass": not strict_failures,
        "input_contract": input_contract,
        "input_contract_sha256": _value_sha256(input_contract),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "checkpoint_best_epoch": int(
            payload.get("best_epoch", payload.get("epoch", 0))
        ),
        "split": args.split,
        "cache_files": len(selected),
        "evaluated_batches": len(clean_rows),
        "device": str(device),
        "resource_preflight": {
            "progress_path": progress_path,
            "input_inventory": inventory,
            "path": str(preflight_path),
            "artifact_sha256": _file_sha256(preflight_path),
            "contract_sha256": _value_sha256(preflight),
            "physical_batch_size": batch_size,
            "num_workers": num_workers,
            "resources": resources,
            "batch_trials": preflight["batch_trials"],
            "worker_trials": preflight["worker_trials"],
        },
        "thresholds": {
            "permutation_tolerance": tolerance,
            "response_threshold": response,
        },
        "clean": clean_metrics,
        "conditions": conditions,
        "view_overlap": overlap_summary,
        "verdict": verdict,
    }
    current_index = validate_cache_publication(cache_dir)
    current_config = _load_json_object(cache_dir / "config.json")
    current_selected, current_entries = _cache_entries_for_split(
        cache_dir, current_index, args.split
    )
    current_artifact_contract = _artifact_contract(
        checkpoint_path=checkpoint_path,
        checkpoint=payload,
        prototype_path=prototype_path,
        signature=signature,
        cache_dir=cache_dir,
        cache_config=current_config,
        selected_entries=current_entries,
        run_mode=args.run_mode,
        split=args.split,
        max_batches=args.max_batches,
        seed=args.seed,
        permutation_tolerance=tolerance,
        response_threshold=response,
        strict=args.strict,
    )
    current_preflight = _load_json_object(preflight_path)
    current_input_contract = _input_contract(
        current_artifact_contract,
        preflight_path=preflight_path,
        preflight=current_preflight,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if current_selected != selected or current_input_contract != input_contract:
        raise RuntimeError(
            "Causality input artifacts changed during the audit; refusing publication"
        )
    _atomic_json(output, report)
    progress(event="report_published", path=str(output), strict_pass=not strict_failures)

    print("[OK] HierCP context-causality audit complete")
    print("checkpoint:", checkpoint_path)
    print("split:", args.split, "samples:", int(clean_metrics["samples"]))
    print(
        "clean:",
        f"top1={clean_metrics['top1']:.4f}",
        f"mrr={clean_metrics['mrr']:.4f}",
        f"margin={clean_metrics['margin']:.4f}",
    )
    for name in CONDITIONS:
        row = conditions[name]
        print(
            f"{name}:",
            f"score_delta={row['mean_abs_score_delta']:.6f}",
            f"margin_drop={row['margin_drop']:.6f}",
            f"mrr={row['mrr']:.4f}",
        )
    print("verdict:", json.dumps(verdict, ensure_ascii=False, sort_keys=True))
    print("report:", output)

    if args.strict and strict_failures:
        raise SystemExit(
            "[FAIL] causality gate: " + ", ".join(strict_failures)
        )


if __name__ == "__main__":
    main()
