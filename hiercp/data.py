"""High-throughput dataset helpers for cached hierarchical PyG samples."""

from __future__ import annotations

import json
import os
from multiprocessing import Value
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Batch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch Geometric is required. Run: python -m tools.install"
    ) from exc

from hiercp.cache import (
    CACHE_FORMAT,
    CACHE_INDEX_FORMAT,
    validate_cache_publication,
)
from hiercp.sample import materialize_sample_views
from hiercp.tensor import torch_load_compat


@dataclass
class HierarchicalBatch:
    """CPU-collated hierarchy that can be transferred with three graph copies."""

    source_patches: Tensor
    target_patches: Tensor
    local_batch: Batch
    local_batch_view2: Batch | None
    patient_batch: Batch
    prototype_batch: Batch
    difficulties: Tensor
    counts: tuple[int, ...]
    case_ids: tuple[str, ...]

    @property
    def sample_count(self) -> int:
        return len(self.counts)

    def difficulty_list(self) -> tuple[Tensor, ...]:
        return torch.split(self.difficulties, self.counts, dim=0)

    def record_stream(self, stream: torch.cuda.Stream) -> "HierarchicalBatch":
        """Associate transferred tensors with a CUDA stream when prefetching."""

        self.source_patches.record_stream(stream)
        self.target_patches.record_stream(stream)
        self.difficulties.record_stream(stream)

        def record(value: Tensor) -> Tensor:
            if torch.is_tensor(value) and value.is_cuda:
                value.record_stream(stream)
            return value

        self.local_batch.apply(record)
        if self.local_batch_view2 is not None:
            self.local_batch_view2.apply(record)
        self.patient_batch.apply(record)
        self.prototype_batch.apply(record)
        return self

    def pin_memory(self) -> "HierarchicalBatch":
        self.source_patches = self.source_patches.pin_memory()
        self.target_patches = self.target_patches.pin_memory()
        self.difficulties = self.difficulties.pin_memory()
        self.local_batch = self.local_batch.pin_memory()
        if self.local_batch_view2 is not None:
            self.local_batch_view2 = self.local_batch_view2.pin_memory()
        self.patient_batch = self.patient_batch.pin_memory()
        self.prototype_batch = self.prototype_batch.pin_memory()
        return self

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "HierarchicalBatch":
        self.source_patches = self.source_patches.to(
            device, non_blocking=non_blocking
        )
        self.target_patches = self.target_patches.to(
            device, non_blocking=non_blocking
        )
        self.difficulties = self.difficulties.to(
            device, non_blocking=non_blocking
        )
        self.local_batch = self.local_batch.to(
            device, non_blocking=non_blocking
        )
        if self.local_batch_view2 is not None:
            self.local_batch_view2 = self.local_batch_view2.to(
                device, non_blocking=non_blocking
            )
        self.patient_batch = self.patient_batch.to(
            device, non_blocking=non_blocking
        )
        self.prototype_batch = self.prototype_batch.to(
            device, non_blocking=non_blocking
        )
        return self


class CudaPrefetchLoader:
    """Overlap pinned-memory H2D copies with the current optimizer step.

    Only cached input graphs/patches are double-buffered; model activations are
    not duplicated. This adds a small input-memory overhead and is safe for the
    A100 10 GB MIG profile used by this project.
    """

    def __init__(self, loader, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("CudaPrefetchLoader requires a CUDA device")
        self.loader = loader
        self.device = device

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        stream = torch.cuda.Stream(device=self.device)
        next_batch: HierarchicalBatch | None = None

        def preload() -> HierarchicalBatch | None:
            try:
                batch = next(iterator)
            except StopIteration:
                return None
            with torch.cuda.stream(stream):
                batch.to(self.device, non_blocking=True)
            return batch

        next_batch = preload()
        while next_batch is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            current = next_batch
            current.record_stream(torch.cuda.current_stream(self.device))
            next_batch = preload()
            yield current


class HierarchicalCacheDataset(Dataset[dict]):
    def __init__(
        self,
        files: Sequence[str | os.PathLike[str]],
        *,
        mmap: bool = True,
        training: bool = False,
        seed: int = 0,
    ) -> None:
        self.files = [Path(path) for path in files]
        self.mmap = bool(mmap)
        self.training = bool(training)
        self.seed = int(seed)
        # Shared memory keeps deterministic epoch-dependent graph views visible
        # to persistent DataLoader workers.
        self._epoch = Value("q", 0, lock=False)

    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = int(epoch)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        sample = torch_load_compat(
            path,
            map_location="cpu",
            mmap=self.mmap,
        )
        if sample.get("format") != CACHE_FORMAT:
            raise ValueError(
                f"Unsupported cache format in {path}: {sample.get('format')}"
            )
        # Keep dense patches in the on-disk float16 representation.  Converting
        # to float32 here doubles RAM and PCIe traffic without recovering any
        # precision; autocast handles them on the GPU.
        if sample["source_patch"].dtype != torch.float16:
            sample["source_patch"] = sample["source_patch"].to(torch.float16)
        if sample["target_patches"].dtype != torch.float16:
            sample["target_patches"] = sample["target_patches"].to(torch.float16)
        difficulties = sample["difficulties"]
        if difficulties.ndim != 1 or difficulties.numel() < 2:
            raise ValueError(f"Bad difficulty vector in {path}")
        if int(difficulties[0]) != 0:
            raise ValueError(f"Positive candidate must be index zero in {path}")
        return materialize_sample_views(
            sample,
            training=self.training,
            epoch=int(self._epoch.value),
            global_seed=self.seed,
        )



def load_cache_config(cache_dir: str | os.PathLike[str]) -> dict:
    path = Path(cache_dir) / "config.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cache config is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid cache config {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != CACHE_FORMAT:
        raise ValueError(f"Unsupported cache config: {path}")
    return payload

def load_cache_index(cache_dir: str | os.PathLike[str]) -> dict:
    payload = validate_cache_publication(cache_dir)
    if payload.get("format") != CACHE_INDEX_FORMAT:
        raise ValueError(f"Unsupported cache index under {Path(cache_dir)}")
    return payload


def list_cache_files(cache_dir: str | os.PathLike[str]) -> list[Path]:
    root = Path(cache_dir)
    index = load_cache_index(root)
    files = [root / str(entry["path"]) for entry in index["entries"]]
    if not files:
        raise RuntimeError(f"Validated cache index is empty: {root}")
    return files


def split_files_from_cache(files: Sequence[Path]) -> tuple[list[Path], list[Path]]:
    if not files:
        raise RuntimeError("No cache files were supplied")
    roots = {path.parent.resolve() for path in files}
    if len(roots) != 1:
        raise ValueError(
            "Cache files must come from one exact validated publication directory"
        )
    root = next(iter(roots))
    index = load_cache_index(root)
    selected = {path.name for path in files}
    indexed = {str(entry["path"]) for entry in index["entries"]}
    unknown = sorted(selected - indexed)
    if unknown:
        raise ValueError(
            "Selected cache files are absent from the validated index: "
            + ", ".join(unknown[:5])
        )
    train = [
        root / str(entry["path"])
        for entry in index["entries"]
        if str(entry.get("path")) in selected and entry.get("split") == "train"
    ]
    val = [
        root / str(entry["path"])
        for entry in index["entries"]
        if str(entry.get("path")) in selected and entry.get("split") == "val"
    ]
    if not train:
        raise RuntimeError("No training cache files were found in validated index.json")
    return sorted(train), sorted(val)


def summarize_cache_usage(
    cache_dir: str | os.PathLike[str],
    *,
    selected_files: Sequence[str | os.PathLike[str]] | None = None,
) -> dict[str, object]:
    """Describe requested, materialized, and actually selected cache coverage."""

    config = load_cache_config(cache_dir)
    index = load_cache_index(cache_dir)
    entries = list(index["entries"])
    indexed_names = {str(entry["path"]) for entry in entries}
    if selected_files is None:
        selected_names = set(indexed_names)
    else:
        selected_names = {Path(path).name for path in selected_files}
    unknown = sorted(selected_names - indexed_names)
    if unknown:
        raise ValueError(
            "Selected cache files are absent from index.json: " + ", ".join(unknown[:5])
        )

    train_case_ids = {str(value) for value in config.get("train_case_ids", ())}
    val_case_ids = {str(value) for value in config.get("val_case_ids", ())}
    requested_case_ids = train_case_ids | val_case_ids
    configured_selected_ids = {
        str(value) for value in config.get("selected_case_ids", requested_case_ids)
    }
    indexed_case_ids = {str(entry.get("case_id", "")) for entry in entries}
    indexed_case_ids.discard("")
    actually_selected_entries = [
        entry for entry in entries if str(entry["path"]) in selected_names
    ]
    actually_selected_case_ids = {
        str(entry.get("case_id", "")) for entry in actually_selected_entries
    }
    actually_selected_case_ids.discard("")
    samples_per_case = config.get("samples_per_case")
    expected_samples: int | str = "unavailable (samples_per_case absent)"
    materialized_ratio: float | str = "unavailable (expected sample count absent)"
    if isinstance(samples_per_case, int) and samples_per_case > 0:
        expected_samples = len(configured_selected_ids) * samples_per_case
        materialized_ratio = (
            len(entries) / expected_samples if expected_samples > 0 else 0.0
        )
    requested_count = len(requested_case_ids)
    configured_count = len(configured_selected_ids)
    return {
        "requested_case_count": requested_count,
        "configured_selected_case_count": configured_count,
        "indexed_case_count": len(indexed_case_ids),
        "actually_used_case_count": len(actually_selected_case_ids),
        "indexed_sample_count": len(entries),
        "actually_used_sample_count": len(actually_selected_entries),
        "expected_selected_sample_count": expected_samples,
        "materialized_sample_ratio": materialized_ratio,
        "actual_index_usage_ratio": (
            len(actually_selected_entries) / len(entries) if entries else 0.0
        ),
        "configured_case_usage_ratio": (
            configured_count / requested_count if requested_count else 0.0
        ),
        "subset_active": bool(
            requested_case_ids and configured_selected_ids != requested_case_ids
        ),
    }


def _graph_collection_statistics(graph_batch: Any) -> dict[str, object]:
    if graph_batch is None:
        return {"available": False, "reason": "graph view is disabled"}
    graphs = (
        list(graph_batch.to_data_list())
        if hasattr(graph_batch, "to_data_list")
        else [graph_batch]
    )
    node_counts: list[int] = []
    edge_counts: list[int] = []
    node_type_totals: dict[str, int] = {}
    edge_type_totals: dict[str, int] = {}
    for graph in graphs:
        graph_nodes = 0
        for node_type in getattr(graph, "node_types", ()):
            store = graph[node_type]
            count = getattr(store, "num_nodes", None)
            if count is None and hasattr(store, "x"):
                count = int(store.x.shape[0])
            count = int(count or 0)
            graph_nodes += count
            node_type_totals[str(node_type)] = node_type_totals.get(str(node_type), 0) + count
        graph_edges = 0
        for edge_type in getattr(graph, "edge_types", ()):
            store = graph[edge_type]
            count = getattr(store, "num_edges", None)
            if count is None and hasattr(store, "edge_index"):
                count = int(store.edge_index.shape[1])
            count = int(count or 0)
            graph_edges += count
            key = "|".join(str(value) for value in edge_type)
            edge_type_totals[key] = edge_type_totals.get(key, 0) + count
        node_counts.append(graph_nodes)
        edge_counts.append(graph_edges)

    def distribution(values: list[int]) -> dict[str, int | float]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0.0, "total": 0}
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "total": sum(values),
        }

    return {
        "available": True,
        "graph_count": len(graphs),
        "nodes_per_graph": distribution(node_counts),
        "edges_per_graph": distribution(edge_counts),
        "node_type_totals": node_type_totals,
        "edge_type_totals": edge_type_totals,
    }


def summarize_hierarchical_batch(batch: HierarchicalBatch) -> dict[str, object]:
    """Return truthful tensor shapes and per-graph node/edge distributions."""

    counts = [int(value) for value in batch.counts]
    return {
        "scope": "provided_batch",
        "sample_count": batch.sample_count,
        "case_ids": list(batch.case_ids),
        "candidate_counts": {
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "mean": sum(counts) / len(counts) if counts else 0.0,
            "total": sum(counts),
        },
        "input_shapes": {
            "source_patches": list(batch.source_patches.shape),
            "target_patches": list(batch.target_patches.shape),
            "difficulties": list(batch.difficulties.shape),
        },
        "local_graphs": _graph_collection_statistics(batch.local_batch),
        "local_graphs_view2": _graph_collection_statistics(batch.local_batch_view2),
        "patient_graphs": _graph_collection_statistics(batch.patient_batch),
        "prototype_graphs": _graph_collection_statistics(batch.prototype_batch),
    }


def collate_samples(samples: list[dict]) -> HierarchicalBatch:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    for sample in samples:
        if "local_graphs" not in sample:
            materialize_sample_views(
                sample, training=False, epoch=0, global_seed=0
            )
    counts = tuple(int(len(sample["local_graphs"])) for sample in samples)
    if any(count < 2 for count in counts):
        raise ValueError("Each hierarchy needs one positive and at least one negative")
    for sample, count in zip(samples, counts):
        if int(sample["target_patches"].shape[0]) != count:
            raise ValueError("target patch count does not match local graph count")
        if int(sample["difficulties"].numel()) != count:
            raise ValueError("difficulty count does not match local graph count")
        if int(sample["patient_graph"]["candidate"].raw_x.shape[0]) != count:
            raise ValueError("patient candidate count does not match local graph count")
        if int(sample["prototype_graph"]["candidate"].raw_x.shape[0]) != count:
            raise ValueError("prototype candidate count does not match local graph count")

    flat_local = [graph for sample in samples for graph in sample["local_graphs"]]
    flat_local_view2 = [
        graph for sample in samples for graph in sample["local_graphs_view2"]
    ]
    return HierarchicalBatch(
        source_patches=torch.stack(
            [sample["source_patch"] for sample in samples], dim=0
        ),
        target_patches=torch.cat(
            [sample["target_patches"] for sample in samples], dim=0
        ),
        local_batch=Batch.from_data_list(flat_local),
        local_batch_view2=Batch.from_data_list(flat_local_view2),
        patient_batch=Batch.from_data_list(
            [sample["patient_graph"] for sample in samples]
        ),
        prototype_batch=Batch.from_data_list(
            [sample["prototype_graph"] for sample in samples]
        ),
        difficulties=torch.cat(
            [sample["difficulties"] for sample in samples], dim=0
        ),
        counts=counts,
        case_ids=tuple(str(sample.get("case_id", "")) for sample in samples),
    )
