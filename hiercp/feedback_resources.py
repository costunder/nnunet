"""Full-pool feedback input accounting, never a graph/sample size limit."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math

import numpy as np
import torch

from hiercp.preparation_runtime import snapshot
from hiercp.training_resources import host_input_budget

INVENTORY_FORMAT = "hiercp_feedback_materialized_inventory_v1"
EXECUTION_VERSION = "hiercp_feedback_full_pool_resources_v2"


def tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=repr):
            yield from tensors(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)
    elif hasattr(value, "to_dict"):
        yield from tensors(value.to_dict())


def tensor_bytes(value):
    return sum(t.numel() * t.element_size() for t in tensors(value))


def content_digest(value):
    """Deterministic logical content digest, separate from serialization bytes."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            if item.device.type != "cpu":
                raise ValueError("Feedback graph publication requires CPU tensors")
            digest.update(json.dumps(["tensor", str(item.dtype), list(item.shape)]).encode())
            if item.numel():
                flat = item.detach().contiguous().reshape(-1)
                # PyTorch considers empty/singleton zero-stride views contiguous,
                # but dtype reinterpretation requires a final stride of one.
                # Skip empty values; only the degenerate nonempty view needs a
                # tiny explicit contiguous clone, not the full graph/batch.
                if flat.stride(0) != 1:
                    flat = flat.clone(memory_format=torch.contiguous_format)
                data = memoryview(flat.view(torch.uint8).numpy()).cast("B")
                for offset in range(0, len(data), 8 * 1024 * 1024):
                    digest.update(data[offset:offset + 8 * 1024 * 1024])
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError("Object arrays cannot enter feedback graph storage")
            digest.update(json.dumps(["numpy", item.dtype.str, list(item.shape)]).encode())
            if item.nbytes:
                digest.update(memoryview(np.ascontiguousarray(item)).cast("B"))
        elif isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}[".encode())
            for child in item:
                visit(child)
            digest.update(b"]")
        elif hasattr(item, "to_dict"):
            visit(item.to_dict())
        elif isinstance(item, np.generic):
            visit(item.item())
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())
        else:
            raise TypeError(f"Unsupported feedback graph content: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def sample_inventory(sample, *, entry_id, candidate_count):
    first, second = sample["local_graphs"], sample["local_graphs_view2"]
    if (not first or len(first) != candidate_count or len(second) != candidate_count
            or sample["target_patches"].shape[0] != candidate_count
            or sample["patient_graph"]["candidate"].num_nodes != candidate_count
            or sample["prototype_graph"]["candidate"].num_nodes != candidate_count):
        raise ValueError("Feedback inventory lost the exact complete candidate pool")
    stores = {}
    for tensor in tensors(sample):
        if tensor.device.type != "cpu":
            raise ValueError("Feedback input inventory must be built before CUDA allocation")
        storage = tensor.untyped_storage()
        stores[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    nodes = sum(g.num_nodes for g in (*first, *second))
    edges = sum(g.num_edges for g in (*first, *second))
    upper_nodes = sum(sample[name].num_nodes for name in ("patient_graph", "prototype_graph"))
    upper_edges = sum(sample[name].num_edges for name in ("patient_graph", "prototype_graph"))
    logical = tensor_bytes(sample)
    unique = sum(stores.values())
    # Existing tensors plus per-node batch/mask/canonical bookkeeping, graph
    # pointers and collation metadata. This does not bound model activations.
    bound = max(logical, unique) + (nodes + upper_nodes) * 64 + (2 * candidate_count + 2) * 512
    return {"format": INVENTORY_FORMAT, "entry_id": str(entry_id),
            "case_id": str(sample["case_id"]), "candidate_count": int(candidate_count),
            "local_nodes_two_views": int(nodes), "local_edges_two_views": int(edges),
            "upper_nodes": int(upper_nodes), "upper_edges": int(upper_edges),
            "logical_tensor_bytes": int(logical), "unique_storage_bytes": int(unique),
            "input_bytes_upper_bound": int(bound)}


def validate_inventory(rows, entries):
    if not entries or len(entries) != len(set(entries)) or set(rows) != set(entries):
        raise ValueError("Feedback inventory must cover every distinct requested bank entry")
    for entry, row in rows.items():
        if row.get("format") != INVENTORY_FORMAT or row.get("entry_id") != entry:
            raise ValueError("Feedback inventory entry/format differs")
        for name in ("candidate_count", "input_bytes_upper_bound", "logical_tensor_bytes", "unique_storage_bytes"):
            if type(row.get(name)) is not int or row[name] <= 0:
                raise ValueError(f"Invalid feedback inventory field: {entry}/{name}")


def worst_batch_bytes(rows, entries, batch_size):
    if type(batch_size) is not int or batch_size < 1 or not entries:
        raise ValueError("Feedback resource candidate must be a positive complete batch")
    sizes = sorted((rows[name]["input_bytes_upper_bound"] for name in entries), reverse=True)
    quotient, remainder = divmod(batch_size, len(sizes))
    return quotient * sum(sizes) + sum(sizes[:remainder])


def calibration_batches(rows, entries, size):
    """Largest and high/low mixed REAL entries; never invent observations."""
    if not 1 <= size <= len(entries) or len(entries) != len(set(entries)):
        raise ValueError("Invalid feedback calibration cohort/batch size")
    ranked = sorted(entries, key=lambda name: (-rows[name]["input_bytes_upper_bound"], name))
    mixed = []
    left, right = 0, len(ranked) - 1
    while left <= right:
        mixed.append(ranked[left])
        left += 1
        if left <= right:
            mixed.append(ranked[right])
            right -= 1
    result = [ranked[:size]]
    if mixed[:size] != result[0]:
        result.append(mixed[:size])
    return result


def allocation_guard(rows, entries, size, *, workers, prefetch_factor, pin_memory, extra_bytes=0):
    if type(workers) is not int or workers < 0 or type(prefetch_factor) is not int or prefetch_factor < 1:
        raise ValueError("Invalid measured feedback worker/prefetch candidate")
    if type(extra_bytes) is not int or extra_bytes < 0:
        raise ValueError("Invalid feedback resident/workspace allowance")
    record = host_input_budget(worst_batch_bytes(rows, entries, size), workers=workers,
                               prefetch_factor=prefetch_factor, pin_memory=pin_memory)
    required = record["estimated_input_bytes"] + extra_bytes
    return {**record, "format": EXECUTION_VERSION, "extra_workspace_bytes": extra_bytes,
            "required_bytes": required, "accepted": required <= record["available_allocation_bytes"] * 4 // 5,
            "scope": "all_requested_sources_and_complete_candidate_pools"}


def require_allocation(record):
    if not record["accepted"]:
        raise MemoryError("Full feedback workload exceeds allocated host headroom; no source/graph was reduced: "
                          + json.dumps(record, sort_keys=True))


def snapshot_guard(additional_bytes, *, phase):
    state = snapshot()
    required = int(additional_bytes)
    if required < 0:
        raise ValueError("Negative feedback resource allowance")
    record = {"phase": phase, "additional_bytes": required, "allocation": state,
              "accepted": required <= state["available_memory_bytes"] * 4 // 5}
    if not record["accepted"]:
        raise MemoryError("Insufficient allocated feedback workspace: " + json.dumps(record, sort_keys=True))
    return record


def missing_optimizer_bytes(actual, probe):
    """Past relation moments absent from this probe still become GPU-resident."""
    return sum(value.numel() * value.element_size()
               for parameter, state in actual.state.items() for name, value in state.items()
               if name != "step" and torch.is_tensor(value)
               and name not in probe.state.get(parameter, {}))


def optimizer_probe_gap(actual, probe, parameters):
    """Bound known state gaps without inventing observations for rare paths.

    A relation absent from measured observations may first occur in a later
    real batch. Besides its two Adam moments, allow one gradient and two
    parameter-sized step scratch buffers. Activations are still measured;
    this is explicitly not a proof of total CUDA peak for arbitrary graphs.
    """
    missing = [parameter for parameter in parameters if parameter.requires_grad
               and not probe.state.get(parameter)]
    previous = missing_optimizer_bytes(actual, probe)
    future = sum(2 * parameter.numel() * parameter.element_size() for parameter in missing
                 if not actual.state.get(parameter))
    scratch = sum(3 * parameter.numel() * parameter.element_size() for parameter in missing)
    return {"historical_moment_bytes": previous, "potential_new_moment_bytes": future,
            "unprobed_gradient_and_step_workspace_bytes": scratch,
            "additional_bytes": previous + future + scratch,
            "is_optimizer_workspace_bound_not_activation_guarantee": True}
