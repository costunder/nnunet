"""Exact local-edge overlap with bounded numeric workspace.

An edge identity is the signed-int64 triple (schema relation index, full source
ID, full destination ID), not a hash or a packed integer. All edges are read;
duplicates are removed only for the diagnostic's existing set semantics.

Sorted numeric runs spill to an owned temporary directory. Pairwise external
merges and the final intersection use bounded NumPy buffers, never Python
objects per edge or a whole-file memory mapping. Extra numeric RAM is O(C),
temporary disk is O(E), and sorting/merging time is O(E log C log(E/C)). C is an
execution chunk size, not a graph cap. The directory is removed on success or
failure; disk errors propagate and never turn into a fabricated metric.
"""
from __future__ import annotations

import errno
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Iterator

import numpy as np
import torch

from hiercp.schema import LOCAL_EDGE_TYPES


EDGE_OVERLAP_CHUNK_EDGES = 262144  # 6 MiB of exact triple keys per read buffer.
_KEY_DTYPE = np.dtype([("relation", "<i8"), ("source", "<i8"), ("destination", "<i8")])
_INTEGRAL_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _validate_chunk_edges(chunk_edges: int) -> None:
    if isinstance(chunk_edges, bool) or not isinstance(chunk_edges, int) or chunk_edges < 1:
        raise ValueError("chunk_edges must be a positive integer workspace size")


def local_edge_jaccard_workspace_bytes(chunk_edges: int = EDGE_OVERLAP_CHUNK_EDGES) -> int:
    """Conservative additive numeric-buffer allowance, not total process RSS.

    Four retained reader/prefix buffers cost 96*C bytes; union concatenation,
    sorting scratch and unique output cost at most another 144*C. Indices,
    masks, gathers and transfers fit within 96*C. Reserve 512*C plus 1 MiB
    rather than depending on tight buffer lifetime/allocator estimates. OS
    filesystem cache and the caller's existing graph tensors are not included.
    """
    _validate_chunk_edges(chunk_edges)
    return 512 * chunk_edges + 1024 * 1024


def local_edge_jaccard_disk_bytes(graph1: Any, graph2: Any, *,
                                  chunk_edges: int = EDGE_OVERLAP_CHUNK_EDGES) -> int:
    """Conservative peak scratch bytes for every input edge, without reading IDs."""
    _validate_chunk_edges(chunk_edges)
    edges = runs = 0
    for graph in (graph1, graph2):
        for edge_type in LOCAL_EDGE_TYPES:
            if edge_type in graph.edge_types:
                count = int(graph[edge_type].edge_index.shape[1])
                edges += count
                runs += (count + chunk_edges - 1) // chunk_edges
    # A merge temporarily retains its input pair and output. Across both graphs
    # payload is <= 2*24*E; reserve 3*24*E plus generous block/run metadata.
    return 3 * _KEY_DTYPE.itemsize * edges + 16384 * (runs + 64)


def _maximum(stats: dict[str, int], name: str, value: int) -> None:
    stats[name] = max(stats.get(name, 0), int(value))


def _edge_key_chunks(graph: Any, chunk_edges: int,
                     stats: dict[str, int]) -> Iterator[np.ndarray]:
    for relation, edge_type in enumerate(LOCAL_EDGE_TYPES):
        if edge_type not in graph.edge_types:
            continue
        edges = graph[edge_type].edge_index
        source_ids = graph[edge_type[0]].full_id
        destination_ids = graph[edge_type[2]].full_id
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype not in _INTEGRAL_DTYPES:
            raise ValueError(f"Invalid integral edge_index for {edge_type}")
        for ids in (source_ids, destination_ids):
            if ids.ndim != 1 or ids.dtype not in _INTEGRAL_DTYPES:
                raise ValueError(f"Expected one-dimensional signed integral full_id for {edge_type}")
        for start in range(0, edges.shape[1], chunk_edges):
            stop = min(start + chunk_edges, edges.shape[1])
            keys = np.empty(stop - start, dtype=_KEY_DTYPE)
            keys["relation"] = relation
            for row, ids, field in ((0, source_ids, "source"), (1, destination_ids, "destination")):
                indices = edges[row, start:stop].to(device=ids.device, dtype=torch.long)
                # A malformed local index must fail, not wrap around as a Python
                # negative index. Validation and gathers are chunk-vectorized.
                if bool(torch.any(indices < 0)) or bool(torch.any(indices >= ids.shape[0])):
                    raise ValueError(f"Local edge index outside full_id bounds for {edge_type}")
                values = ids[indices].detach().to(device="cpu", dtype=torch.int64).numpy()
                keys[field] = values
                del indices, values
            _maximum(stats, "peak_key_chunk_rows", keys.size)
            stats["input_edges"] = stats.get("input_edges", 0) + int(keys.size)
            yield keys
            del keys


def _read_chunk(stream: Any, chunk_edges: int, stats: dict[str, int]) -> np.ndarray:
    values = np.fromfile(stream, dtype=_KEY_DTYPE, count=chunk_edges)
    _maximum(stats, "peak_read_rows", values.size)
    return values


def _paired_prefixes(first_path: Path, second_path: Path, chunk_edges: int,
                     stats: dict[str, int], *, include_tails: bool
                     ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ordered, disjoint key ranges while retiring at least one buffer."""
    with first_path.open("rb") as first_stream, second_path.open("rb") as second_stream:
        first = _read_chunk(first_stream, chunk_edges, stats)
        second = _read_chunk(second_stream, chunk_edges, stats)
        while first.size and second.size:
            # Only two scalar triples become Python tuples, never an edge list.
            boundary = first[-1] if first[-1].tolist() <= second[-1].tolist() else second[-1]
            first_end = int(np.searchsorted(first, boundary, side="right"))
            second_end = int(np.searchsorted(second, boundary, side="right"))
            _maximum(stats, "peak_paired_input_rows", first.size + second.size)
            yield first[:first_end], second[:second_end]
            first = first[first_end:]
            second = second[second_end:]
            if not first.size:
                first = _read_chunk(first_stream, chunk_edges, stats)
            if not second.size:
                second = _read_chunk(second_stream, chunk_edges, stats)
        if include_tails:
            empty = np.empty(0, dtype=_KEY_DTYPE)
            while first.size:
                yield first, empty
                first = _read_chunk(first_stream, chunk_edges, stats)
            while second.size:
                yield empty, second
                second = _read_chunk(second_stream, chunk_edges, stats)


def _merge_runs(first: Path, second: Path, output: Path, chunk_edges: int,
                stats: dict[str, int]) -> None:
    with output.open("xb") as stream:
        for left, right in _paired_prefixes(first, second, chunk_edges, stats, include_tails=True):
            merged = np.union1d(left, right)
            _maximum(stats, "peak_merge_output_rows", merged.size)
            merged.tofile(stream)
            del merged


def _sorted_unique_run(graph: Any, directory: Path, prefix: str, chunk_edges: int,
                       stats: dict[str, int]) -> Path:
    run_count = 0
    for keys in _edge_key_chunks(graph, chunk_edges, stats):
        unique = np.unique(keys)
        path = directory / f"{prefix}.0.{run_count}.bin"
        with path.open("xb") as stream:
            unique.tofile(stream)
        run_count += 1
        del unique, keys
    stats["initial_runs"] = stats.get("initial_runs", 0) + run_count
    if run_count == 0:
        path = directory / f"{prefix}.empty.bin"
        path.touch(exist_ok=False)
        return path
    level = 0
    while run_count > 1:
        next_count = (run_count + 1) // 2
        for index in range(next_count):
            first = directory / f"{prefix}.{level}.{2 * index}.bin"
            output = directory / f"{prefix}.{level + 1}.{index}.bin"
            if 2 * index + 1 == run_count:
                first.rename(output)
            else:
                second = directory / f"{prefix}.{level}.{2 * index + 1}.bin"
                _merge_runs(first, second, output, chunk_edges, stats)
                # These are exact files created by this call under its own
                # TemporaryDirectory, never caller data or experiment outputs.
                first.unlink()
                second.unlink()
        run_count = next_count
        level += 1
    _maximum(stats, "merge_levels", level)
    return directory / f"{prefix}.{level}.0.bin"


def local_edge_jaccard(graph1: Any, graph2: Any, *,
                       chunk_edges: int = EDGE_OVERLAP_CHUNK_EDGES,
                       temporary_directory: str | Path | None = None,
                       diagnostics: dict[str, int] | None = None) -> float:
    """Return exact edge-set Jaccard, with empty/empty equal to 1.0.

    Optional diagnostics report execution buffer sizes and exact counts, not
    process RSS. The caller can place ephemeral runs on a measured scratch
    filesystem via ``temporary_directory``; its existing contents are untouched.
    """
    _validate_chunk_edges(chunk_edges)
    required_disk = local_edge_jaccard_disk_bytes(graph1, graph2, chunk_edges=chunk_edges)
    stats: dict[str, int] = {"chunk_edges": chunk_edges, "key_bytes": _KEY_DTYPE.itemsize,
                            "workspace_bytes_bound": local_edge_jaccard_workspace_bytes(chunk_edges),
                            "temporary_disk_bytes_bound": required_disk}
    with TemporaryDirectory(prefix="hiercp_edge_overlap_", dir=temporary_directory) as root:
        directory = Path(root)
        free_disk = shutil.disk_usage(directory).free
        if free_disk < required_disk:
            raise OSError(errno.ENOSPC, "Exact local-edge overlap needs up to "
                          f"{required_disk} temporary bytes; only {free_disk} free in {directory.parent}. "
                          "No edges were dropped. Choose scratch storage with sufficient free space.")
        try:
            first = _sorted_unique_run(graph1, directory, "first", chunk_edges, stats)
            second = _sorted_unique_run(graph2, directory, "second", chunk_edges, stats)
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise OSError(errno.ENOSPC, "Exact local-edge overlap exhausted temporary disk "
                              f"in {directory.parent}; other processes may have consumed the preflight "
                              "free space. No metric was substituted; owned temporary runs are cleaned.") from exc
            raise
        first_count = first.stat().st_size // _KEY_DTYPE.itemsize
        second_count = second.stat().st_size // _KEY_DTYPE.itemsize
        intersection = 0
        for left, right in _paired_prefixes(first, second, chunk_edges, stats, include_tails=False):
            if left.size and right.size:
                locations = np.searchsorted(right, left)
                valid = locations < right.size
                intersection += int(np.count_nonzero(right[locations[valid]] == left[valid]))
                del locations, valid
        union = first_count + second_count - intersection
        stats.update(first_unique_edges=first_count, second_unique_edges=second_count,
                     intersection_edges=intersection, union_edges=union)
    if diagnostics is not None:
        diagnostics.update(stats)
    return float(intersection / union) if union else 1.0
