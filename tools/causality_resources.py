"""Allocation-aware input accounting and durable progress for causality audits.

The inventory visits every selected cache file once, without materializing its
graphs or modifying it. A rejected budget rejects a resource trial, never a
patient, candidate, node or edge. These are conservative *input* bounds, not an
OOM guarantee: model activations, Python objects, allocator overhead and changes
in other processes still require the runtime measurements and progress log.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any, Sequence
import uuid

from hiercp.preparation_runtime import snapshot
from hiercp.causality_overlap import local_edge_jaccard_workspace_bytes
from hiercp.training_resources import (
    INPUT_ACCOUNTING_VERSION,
    cache_input_inventory,
    host_input_budget,
)


AUDIT_INPUT_ACCOUNTING_VERSION = "hiercp_causality_input_budget_v1"
PROGRESS_FORMAT = "hiercp_causality_progress_v1"
# After copy-on-write transformations, a full batch bounds: (1) changed graph
# fields, (2) rebatching their concatenated tensors, (3) permutation/indexing
# scratch and (4) the prior transformed view while the other view is rebuilt.
# The original batch, active collations and prefetched batches are accounted for
# separately below. This does NOT cover the old per-graph deepcopy of full
# backing storage, or Python tuple/set representations of every graph edge.
TRANSFORM_WORKSPACE_INPUT_COPIES = 4


def _integer(value: Any, *, name: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _selected_paths(selected: Sequence[Path]) -> list[Path]:
    paths = [Path(path).resolve() for path in selected]
    if not paths:
        raise ValueError("Causality input inventory requires the complete nonempty selected cohort")
    if len(set(paths)) != len(paths):
        raise ValueError("Causality selected cohort contains duplicate cache paths")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Causality selected cohort contains ambiguous cache file names")
    return paths


def build_audit_inventory(selected: Sequence[Path]) -> dict[str, Any]:
    """Read all selected canonical tensor shapes once; retain no loaded sample."""
    paths = _selected_paths(selected)
    inventory = cache_input_inventory(paths)
    inventory.update({
        "audit_format": AUDIT_INPUT_ACCOUNTING_VERSION,
        "selected_files": [str(path) for path in paths],
        "scope": "complete_selected_causality_cohort",
        "transform_workspace_input_copies": TRANSFORM_WORKSPACE_INPUT_COPIES,
    })
    _input_sizes(inventory, paths)
    return inventory


def _input_sizes(inventory: dict[str, Any], selected: Sequence[Path]) -> list[int]:
    paths = _selected_paths(selected)
    if inventory.get("format") != INPUT_ACCOUNTING_VERSION or inventory.get("audit_format") != AUDIT_INPUT_ACCOUNTING_VERSION:
        raise ValueError("Unsupported causality input inventory")
    if set(inventory.get("selected_files", [])) != {str(path) for path in paths}:
        raise ValueError("Causality inventory must cover the exact complete selected cohort")
    if inventory.get("sample_count") != len(paths):
        raise ValueError("Causality inventory sample count differs from the selected cohort")
    rows = inventory.get("rows")
    if not isinstance(rows, list) or len(rows) != len(paths):
        raise ValueError("Causality inventory rows do not cover the selected cohort")
    by_name: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("cache_file"), str):
            raise ValueError("Invalid causality inventory row")
        name = row["cache_file"]
        if name in by_name:
            raise ValueError("Duplicate causality inventory row")
        by_name[name] = _integer(row.get("input_bytes_upper_bound"), name="input_bytes_upper_bound", minimum=1)
    if set(by_name) != {path.name for path in paths}:
        raise ValueError("Causality inventory rows differ from the selected cohort")
    return [by_name[path.name] for path in paths]


def audit_batch_input_bytes(inventory: dict[str, Any], selected: Sequence[Path], batch_size: int) -> int:
    """Bound any cohort batch, including cyclic repeated loader probe batches.

    B=q*N+r consumes q complete cohorts plus r files. Summing the r largest
    files bounds every cyclic start and ordering. Unlike slicing unique rows,
    this continues to account for *all B samples* when B exceeds the cohort.
    For a non-repeating final partial batch the bound is conservative.
    """
    batch_size = _integer(batch_size, name="batch_size", minimum=1)
    sizes = _input_sizes(inventory, selected)
    cycles, remainder = divmod(batch_size, len(sizes))
    return cycles * sum(sizes) + sum(sorted(sizes, reverse=True)[:remainder])


def audit_host_budget(
    inventory: dict[str, Any],
    selected: Sequence[Path],
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    *,
    phase: str = "full",
) -> dict[str, Any]:
    """Check fresh host/cgroup headroom, without changing the resource candidate.

    ``full`` includes uncollated+collated inputs per active worker, queued
    prefetches, the consumer, a transient pinned copy and transformation work.
    It is also used for worker trials, so their winner fits the actual audit.
    ``resident_transform`` is for a live consumer batch/loader: only additional
    transformation workspace is charged against remaining allocation memory.
    The already allocated queues/batch must not be charged a second time.
    """
    workers = _integer(workers, name="workers", minimum=0)
    prefetch_factor = _integer(prefetch_factor, name="prefetch_factor", minimum=1)
    if not isinstance(pin_memory, bool):
        raise ValueError("pin_memory must be a boolean")
    if phase not in {"full", "resident_transform"}:
        raise ValueError(f"Unsupported causality budget phase: {phase!r}")
    batch_bytes = audit_batch_input_bytes(inventory, selected, batch_size)
    # This existing function uses preparation_runtime.snapshot(), including the
    # process's v1/v2 cgroup leaf and all applicable ancestor limits/current use.
    baseline = host_input_budget(batch_bytes, workers=workers,
                                 prefetch_factor=prefetch_factor, pin_memory=pin_memory)
    available = _integer(baseline["available_allocation_bytes"], name="available_allocation_bytes", minimum=0)
    loader_copies = int(baseline["input_copies_accounted"]) if phase == "full" else 0
    copies = loader_copies + TRANSFORM_WORKSPACE_INPUT_COPIES
    overlap_workspace = _integer(local_edge_jaccard_workspace_bytes(), name="overlap_workspace_bytes", minimum=1)
    required = batch_bytes * copies + overlap_workspace
    permitted = available * 4 // 5
    return {
        **baseline,
        "format": AUDIT_INPUT_ACCOUNTING_VERSION,
        "phase": phase,
        "batch_size": batch_size,
        "num_workers": workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
        "selected_cohort_samples": len(selected),
        "batch_input_bytes_upper_bound": batch_bytes,
        "loader_input_copies_accounted": loader_copies,
        "transform_workspace_input_copies": TRANSFORM_WORKSPACE_INPUT_COPIES,
        "input_copies_accounted": copies,
        "overlap_workspace_bytes": overlap_workspace,
        "estimated_input_bytes": required,
        "permitted_input_bytes": permitted,
        "accepted": required <= permitted,
        "graph_or_data_reduction": False,
        "limitation": (
            "Canonical input, copy-on-write transform and bounded numeric overlap workspace; "
            "not a bound on model activations, allocator overhead, or other processes' future use"
        ),
    }


def validate_audit_host_budget(
    record: dict[str, Any],
    *,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    batch_input_bytes_upper_bound: int,
    phase: str = "full",
) -> None:
    """Validate saved arithmetic against the signed inventory, without resampling.

    Current availability is deliberately not read here: this verifies historic
    trial evidence. A separate fresh ``audit_host_budget`` is still required
    before a resumed loader or transformation allocates its inputs.
    """
    batch_size = _integer(batch_size, name="batch_size", minimum=1)
    workers = _integer(workers, name="workers", minimum=0)
    prefetch_factor = _integer(prefetch_factor, name="prefetch_factor", minimum=1)
    batch_bytes = _integer(batch_input_bytes_upper_bound, name="batch_input_bytes_upper_bound", minimum=1)
    if not isinstance(record, dict):
        raise ValueError("Saved causality host budget is not an object")
    if not isinstance(pin_memory, bool) or phase not in {"full", "resident_transform"}:
        raise ValueError("Invalid saved causality budget candidate/phase")
    available = _integer(record.get("available_allocation_bytes"), name="available_allocation_bytes", minimum=0)
    loader_copies = (2 * max(1, workers) + workers * prefetch_factor + 1 + int(pin_memory)) if phase == "full" else 0
    copies = loader_copies + TRANSFORM_WORKSPACE_INPUT_COPIES
    overlap = _integer(local_edge_jaccard_workspace_bytes(), name="overlap_workspace_bytes", minimum=1)
    required = batch_bytes * copies + overlap
    permitted = available * 4 // 5
    expected = {
        "format": AUDIT_INPUT_ACCOUNTING_VERSION,
        "phase": phase,
        "batch_size": batch_size,
        "num_workers": workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
        "batch_input_bytes_upper_bound": batch_bytes,
        "loader_input_copies_accounted": loader_copies,
        "transform_workspace_input_copies": TRANSFORM_WORKSPACE_INPUT_COPIES,
        "input_copies_accounted": copies,
        "overlap_workspace_bytes": overlap,
        "estimated_input_bytes": required,
        "permitted_input_bytes": permitted,
        "reserved_headroom_fraction": 0.2,
        "accepted": required <= permitted,
        "graph_or_data_reduction": False,
        "is_input_bound_not_total_memory_guarantee": True,
    }
    for name, value in expected.items():
        saved = record.get(name)
        if type(saved) is not type(value) or saved != value:
            raise ValueError(f"Saved causality host budget differs at {name}: {saved!r} != {value!r}")
    _integer(record.get("selected_cohort_samples"), name="selected_cohort_samples", minimum=1)
    allocation = record.get("allocation")
    if not isinstance(allocation, dict):
        raise ValueError("Saved causality host budget lacks its allocation snapshot")
    measured_available = _integer(allocation.get("available_memory_bytes"), name="allocation.available_memory_bytes", minimum=0)
    if measured_available != available:
        raise ValueError("Saved causality host budget contradicts its allocation snapshot")


def audit_allocation_fingerprint() -> dict[str, Any]:
    """Bind audit reuse to actual cgroup limits, not fluctuating current use."""
    state = snapshot()
    allocation = state["cgroup"]
    if not isinstance(allocation, dict) or not isinstance(allocation.get("values"), dict):
        raise ValueError("Allocation snapshot has no cgroup accounting evidence")
    stable_names = {"memory.max", "memory.limit_in_bytes", "cpu.max", "cpu.cfs_quota_us", "cpu.cfs_period_us"}
    limits = {
        location: {name: value for name, value in values.items() if name in stable_names}
        for location, values in allocation["values"].items()
    }
    return {
        "format": "hiercp_causality_allocation_v1",
        "host_total_memory_bytes": state["host_total_memory_bytes"],
        "cpu_affinity_cores": state["cpu_affinity_cores"],
        "cpu_allocation_cores": state["cpu_allocation_cores"],
        "cgroup_locations_resolved": allocation["locations_resolved"],
        "cgroup_limits": limits,
        "missing_cgroup_files": allocation["missing_files"],
        "limitation": allocation["limitation"],
    }


class AuditProgressLog:
    """One exclusive append-only file per invocation; every event is fsynced."""

    def __init__(self, output_path: Path):
        output_path = Path(output_path)
        self.run_id = uuid.uuid4().hex
        self.path = output_path.with_name(f"{output_path.name}.progress.{self.run_id}.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)
        try:
            self._handle = os.fdopen(descriptor, "a", encoding="utf-8", newline="\n")
        except BaseException:
            os.close(descriptor)
            raise
        self._sequence = 0
        self._lock = threading.Lock()

    def __call__(self, **fields: Any) -> None:
        reserved = {"format", "run_id", "sequence", "timestamp_utc", "pid", "resource_snapshot"}
        if reserved.intersection(fields):
            raise ValueError(f"Progress fields collide with reserved keys: {sorted(reserved.intersection(fields))}")
        with self._lock:
            if self._handle.closed:
                raise ValueError("Causality progress log is closed")
            resources = snapshot()
            record = {
                "format": PROGRESS_FORMAT,
                "run_id": self.run_id,
                "sequence": self._sequence,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
                "resource_snapshot": resources,
                **fields,
            }
            encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True)
            self._handle.write(encoded + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._sequence += 1
            labels = " ".join(f"{key}={fields[key]}" for key in
                              ("stage", "event", "batch_size", "num_workers", "repeat", "condition") if key in fields)
            print(f"[CausalityProgress] {labels} rss={resources['rss_bytes']} "
                  f"available={resources['available_memory_bytes']} pid={record['pid']}", flush=True)

    def close(self) -> None:
        with self._lock:
            self._handle.close()

    def __enter__(self) -> "AuditProgressLog":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.close()
        return False


def create_progress_log(output_path: Path) -> AuditProgressLog:
    return AuditProgressLog(output_path)
