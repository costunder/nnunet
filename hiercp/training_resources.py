"""Read-only full-cohort graph input accounting (not a graph-size limit).

Canonical tensors are memory mapped and only their shapes are read. Counts are
upper bounds for every induced view, including later epochs: selection cannot
add a node or an edge to the canonical topology. No cache bytes are modified.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from hiercp.preparation_runtime import snapshot
from hiercp.tensor import torch_load_compat


INPUT_ACCOUNTING_VERSION = "hiercp_canonical_input_upper_bound_v1"


def _tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    if hasattr(value, "to_dict"):
        return _tensor_bytes(value.to_dict())
    return 0


def _local_bound(local):
    if local.get("format") != "canonical-full-v22":
        raise ValueError("Input accounting requires canonical-full-v22 tables")
    nodes = sum(int(table["x"].shape[0]) for table in local["nodes"].values())
    edges = sum(int(index.shape[1]) for index in local["edges"].values())
    # edge_index int64[2,E], edge_attr float32[E,10]; node table copies plus
    # per-node canonical IDs, batch IDs, masks and collation bookkeeping.
    # Each view combines two local tables: 2*512 bytes also cover graph-level
    # count/audit tensors and PyG pointers, including very small DEBUG graphs.
    size = edges * (2 * 8 + 10 * 4) + _tensor_bytes(local["nodes"]) + nodes * 64 + 512
    return nodes, edges, size


def cache_input_inventory(files):
    rows = []
    for path in files:
        path = Path(path)
        sample = torch_load_compat(path, map_location="cpu", mmap=True)
        source = _local_bound(sample["source_local"])
        targets = [_local_bound(target) for target in sample["target_locals"]]
        if not targets:
            raise ValueError(f"No target graphs in {path}")
        count = len(targets)
        row = {
            "cache_file": path.name,
            "candidate_count": count,
            "canonical_nodes_two_views": 2 * (count * source[0] + sum(x[0] for x in targets)),
            "canonical_edges_two_views": 2 * (count * source[1] + sum(x[1] for x in targets)),
            "input_bytes_upper_bound": 2 * (count * source[2] + sum(x[2] for x in targets))
                + sum(_tensor_bytes(value) for key, value in sample.items()
                      if key not in {"source_local", "target_locals"}),
        }
        rows.append(row)
        del sample
    if len({row["cache_file"] for row in rows}) != len(rows):
        raise ValueError("Input inventory requires distinct cache file names")
    return {
        "format": INPUT_ACCOUNTING_VERSION,
        "scope": "all_supplied_train_and_validation_cache_files",
        "sample_count": len(rows),
        "rows": rows,
        "limitation": "Canonical input bounds, not a bound on model activations or allocator overhead; GPU forward/backward remains measured",
        "cache_modified": False,
        "graph_or_data_reduction": False,
    }


def worst_input_bytes(inventory, files, batch_size):
    names = {Path(path).name for path in files}
    rows = [row for row in inventory["rows"] if row["cache_file"] in names]
    if len(rows) != len(names):
        raise ValueError("Input inventory does not cover the requested cache files")
    return sum(sorted((int(row["input_bytes_upper_bound"]) for row in rows), reverse=True)[:batch_size])


def host_input_budget(batch_bytes, *, workers, prefetch_factor, pin_memory):
    """Account for collation scratch, concurrent workers and prefetched inputs.

    This rejects a resource candidate, never removes a training sample/edge.
    2 input copies cover uncollated+collated tensors per active worker; queued
    batches and the consumer each require another copy. Pinning can temporarily
    duplicate a queued batch. Twenty percent of available allocation remains
    outside this tensor estimate for Python, mmap pages and other processes.
    """
    state = snapshot()
    active = max(1, int(workers))
    queued = int(workers) * int(prefetch_factor)
    copies = 2 * active + queued + 1 + int(bool(pin_memory))
    required = int(batch_bytes) * copies
    available = int(state["available_memory_bytes"])
    return {
        "estimated_input_bytes": required,
        "input_copies_accounted": copies,
        "available_allocation_bytes": available,
        "allocation": state,
        "reserved_headroom_fraction": 0.2,
        "accepted": required <= int(available * 0.8),
        "is_input_bound_not_total_memory_guarantee": True,
    }
