"""Canonical-node sampling and sampled spatial HeteroData construction.

The canonical cache contains every deterministic physical grid cell and every
physical-radius edge.  A training view samples only context nodes, preserves all
tumor and liver-surface anchors, and retains every cached edge whose endpoints
remain.  No k-NN graph is introduced.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor
from torch.nn import functional as F

try:
    from torch_geometric.data import HeteroData
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
    raise ModuleNotFoundError(
        "PyTorch Geometric is required. Run: python -m tools.install"
    ) from exc

from hiercp.schema import (
    CONTEXT_SHELL_COUNT,
    CONTEXT_SHELL_FEATURE_INDEX,
    LOCAL_EDGE_DIM,
    LOCAL_EDGE_TYPES,
    LOCAL_NODE_TYPES,
    GraphBuildConfig,
)


def sample_dense_features_variable(
    feature_map: Tensor,
    grid: Tensor,
    node_batch: Tensor,
) -> Tensor:
    """Sample unequal node counts while tolerating compact float16 cache grids.

    Cached node coordinates are stored as float16 to reduce graph-cache size,
    while dense feature maps are float32 outside AMP and float16 inside CUDA
    AMP. ``grid_sample`` requires matching input/grid dtypes, so the grid is
    converted to the feature-map compute dtype before the padded sampling call.
    """
    if feature_map.ndim != 5:
        raise ValueError(f"feature_map must be [B,C,D,H,W], got {tuple(feature_map.shape)}")
    if not feature_map.is_floating_point():
        raise TypeError(f"feature_map must be floating point, got {feature_map.dtype}")
    if grid.ndim != 2 or int(grid.shape[1]) != 3:
        raise ValueError(f"grid must be [N,3], got {tuple(grid.shape)}")
    if not grid.is_floating_point():
        raise TypeError(f"grid must be floating point, got {grid.dtype}")
    if node_batch.ndim != 1 or int(node_batch.numel()) != int(grid.shape[0]):
        raise ValueError(
            "node_batch must contain one graph index per node: "
            f"batch={tuple(node_batch.shape)}, grid={tuple(grid.shape)}"
        )

    graph_count = int(feature_map.shape[0])
    node_total = int(grid.shape[0])
    if graph_count < 1:
        raise ValueError("feature_map contains no graphs")
    if node_total == 0:
        return feature_map.new_empty((0, int(feature_map.shape[1])))

    device = feature_map.device
    # CUDA grid_sample supports float16. CPU float16 and bfloat16 grid_sample
    # are not consistently available across supported PyTorch releases, so
    # those cases use float32 compute and cast the result back afterwards.
    compute_dtype = feature_map.dtype
    if feature_map.dtype == torch.bfloat16 or (
        device.type == "cpu" and feature_map.dtype == torch.float16
    ):
        compute_dtype = torch.float32

    sample_input = feature_map
    if sample_input.dtype != compute_dtype:
        sample_input = sample_input.to(dtype=compute_dtype)
    sample_grid = grid.to(device=device, dtype=compute_dtype)
    node_batch = node_batch.to(device=device, dtype=torch.long)

    if int(node_batch.min()) < 0 or int(node_batch.max()) >= graph_count:
        raise ValueError(
            f"node_batch index outside [0,{graph_count - 1}]: "
            f"min={int(node_batch.min())}, max={int(node_batch.max())}"
        )
    counts = torch.bincount(node_batch, minlength=graph_count)
    if bool(torch.any(counts == 0)):
        raise ValueError(
            "Every dense feature map must own at least one node of this semantic type: "
            f"counts={counts.tolist()}"
        )
    expected_batch = torch.repeat_interleave(
        torch.arange(graph_count, device=device), counts
    )
    if not bool(torch.equal(node_batch, expected_batch)):
        raise ValueError("PyG node_batch is not graph-major; dense-map association is ambiguous")

    max_nodes = int(counts.max())
    starts = torch.cumsum(counts, dim=0) - counts
    local_index = torch.arange(node_total, device=device) - starts[node_batch]
    node_grid = torch.zeros(
        (graph_count, max_nodes, 1, 1, 3),
        device=device,
        dtype=compute_dtype,
    )
    node_grid[node_batch, local_index, 0, 0] = sample_grid
    sampled = F.grid_sample(
        sample_input,
        node_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    dense = sampled[:, :, :, 0, 0].permute(0, 2, 1)
    output = dense[node_batch, local_index]
    if output.dtype != feature_map.dtype:
        output = output.to(dtype=feature_map.dtype)
    return output


def stable_view_seed(
    global_seed: int,
    case_id: str,
    sample_index: int,
    candidate_index: int,
    epoch: int,
    view_index: int,
) -> int:
    payload = (
        f"{int(global_seed)}|{case_id}|{int(sample_index)}|{int(candidate_index)}|"
        f"{int(epoch)}|{int(view_index)}"
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def _numpy(value: Tensor | np.ndarray, *, dtype: np.dtype | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _balanced_indices(
    positions_mm: np.ndarray,
    features: np.ndarray,
    budget: int,
    rng: np.random.Generator,
    *,
    radial_bins: int,
    azimuth_bins: int,
    elevation_bins: int,
    required: np.ndarray | None = None,
) -> np.ndarray:
    count = int(positions_mm.shape[0])
    if count == 0:
        return np.empty((0,), dtype=np.int64)
    budget = int(budget)
    if budget <= 0 or count <= budget:
        return np.arange(count, dtype=np.int64)

    required_ids = (
        np.unique(np.asarray(required, dtype=np.int64))
        if required is not None and np.asarray(required).size
        else np.empty((0,), dtype=np.int64)
    )
    required_ids = required_ids[(required_ids >= 0) & (required_ids < count)]
    if required_ids.size >= budget:
        # Preserve physical coverage even when interface-neighbour recall alone
        # exceeds the view budget.
        positions_req = positions_mm[required_ids]
        features_req = features[required_ids]
        local = _balanced_indices(
            positions_req,
            features_req,
            budget,
            rng,
            radial_bins=radial_bins,
            azimuth_bins=azimuth_bins,
            elevation_bins=elevation_bins,
        )
        return np.sort(required_ids[local])

    radius = np.linalg.norm(positions_mm, axis=1)
    unit = positions_mm / np.maximum(radius[:, None], 1e-6)
    azimuth = np.mod(np.arctan2(unit[:, 1], unit[:, 2]), 2.0 * np.pi)
    elevation = np.arcsin(np.clip(unit[:, 0], -1.0, 1.0)) + np.pi / 2.0
    radial = np.clip(
        np.rint(features[:, CONTEXT_SHELL_FEATURE_INDEX] * CONTEXT_SHELL_COUNT).astype(
            np.int64
        ),
        0,
        max(0, int(radial_bins) - 1),
    )
    az = np.clip(
        np.floor(azimuth / (2.0 * np.pi) * max(1, int(azimuth_bins))).astype(np.int64),
        0,
        max(1, int(azimuth_bins)) - 1,
    )
    el = np.clip(
        np.floor(elevation / np.pi * max(1, int(elevation_bins))).astype(np.int64),
        0,
        max(1, int(elevation_bins)) - 1,
    )
    sector = (
        radial * max(1, int(azimuth_bins)) * max(1, int(elevation_bins))
        + az * max(1, int(elevation_bins))
        + el
    )

    required_set = set(int(value) for value in required_ids.tolist())
    buckets: dict[int, list[int]] = {}
    for sector_id in np.unique(sector):
        ids = np.flatnonzero(sector == sector_id).astype(np.int64)
        ids = np.asarray([value for value in ids if int(value) not in required_set], dtype=np.int64)
        if ids.size:
            buckets[int(sector_id)] = ids[rng.permutation(ids.size)].tolist()
    sectors = sorted(buckets)
    if sectors:
        shift = int(rng.integers(len(sectors)))
        sectors = sectors[shift:] + sectors[:shift]

    selected = [int(value) for value in required_ids.tolist()]
    offsets = {sector_id: 0 for sector_id in sectors}
    while len(selected) < budget:
        progress = False
        for sector_id in sectors:
            offset = offsets[sector_id]
            bucket = buckets[sector_id]
            if offset >= len(bucket):
                continue
            selected.append(int(bucket[offset]))
            offsets[sector_id] = offset + 1
            progress = True
            if len(selected) >= budget:
                break
        if not progress:
            break
    if len(selected) < budget:
        remaining = np.asarray(
            [index for index in range(count) if index not in set(selected)], dtype=np.int64
        )
        if remaining.size:
            take = min(budget - len(selected), int(remaining.size))
            selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
    return np.sort(np.asarray(selected[:budget], dtype=np.int64))


def _radius_neighbor_ids(
    full_position_mm: np.ndarray,
    query_position_mm: np.ndarray,
    radius_mm: float,
) -> np.ndarray:
    if full_position_mm.shape[0] == 0 or query_position_mm.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    tree = cKDTree(full_position_mm, compact_nodes=True, balanced_tree=True)
    local = tree.query_ball_point(query_position_mm, float(radius_mm))
    values = [int(index) for row in local for index in row]
    return np.unique(np.asarray(values, dtype=np.int64)) if values else np.empty((0,), dtype=np.int64)


def _select_context(
    node: Mapping[str, Tensor],
    anchor_position_mm: np.ndarray,
    config: GraphBuildConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    position = _numpy(node["pos_mm"], dtype=np.float32)
    features = _numpy(node["x"], dtype=np.float32)
    budget = int(config.sample_context_nodes)
    if budget <= 0 or position.shape[0] <= budget:
        return np.arange(position.shape[0], dtype=np.int64)

    required = _radius_neighbor_ids(
        position,
        anchor_position_mm,
        float(config.sample_interface_radius_mm),
    )
    selected = _balanced_indices(
        position,
        features,
        min(budget, max(int(required.size), budget // 2)),
        rng,
        radial_bins=int(config.context_radial_bins),
        azimuth_bins=int(config.context_azimuth_bins),
        elevation_bins=int(config.context_elevation_bins),
        required=required,
    )
    for _ in range(max(0, int(config.sample_hops))):
        expanded = _radius_neighbor_ids(
            position,
            position[selected],
            float(config.sample_hop_radius_mm),
        )
        union = np.union1d(selected, expanded).astype(np.int64, copy=False)
        selected = _balanced_indices(
            position,
            features,
            budget,
            rng,
            radial_bins=int(config.context_radial_bins),
            azimuth_bins=int(config.context_azimuth_bins),
            elevation_bins=int(config.context_elevation_bins),
            required=union,
        )
        if selected.size >= budget:
            break
    return _balanced_indices(
        position,
        features,
        budget,
        rng,
        radial_bins=int(config.context_radial_bins),
        azimuth_bins=int(config.context_azimuth_bins),
        elevation_bins=int(config.context_elevation_bins),
        required=selected,
    )


def _edge_radius(edge_type: tuple[str, str, str], config: GraphBuildConfig) -> float:
    source_type, relation, destination_type = edge_type
    if source_type == destination_type:
        if "context" in source_type:
            return float(config.context_edge_radius_mm)
        if "liver_surface" in source_type:
            return float(config.liver_edge_radius_mm)
        if source_type == "tumor_surface":
            return float(config.surface_edge_radius_mm)
        if source_type == "tumor_interior":
            return float(config.interior_edge_radius_mm)
    if relation in {"interfaces_source", "interfaces_target", "interfaces_tumor"}:
        return float(config.interface_edge_radius_mm)
    if relation in {"near_liver_surface", "anchors_context"}:
        return float(config.liver_edge_radius_mm)
    if relation in {"corresponds_to", "corresponds_to_source"}:
        return float(config.correspondence_radius_mm)
    return float(config.cross_edge_radius_mm)


def _edge_attributes(
    source_features: np.ndarray,
    destination_features: np.ndarray,
    source_positions: np.ndarray,
    destination_positions: np.ndarray,
    edge_index: np.ndarray,
    *,
    source_position_override: np.ndarray | None = None,
    source_normal_override: np.ndarray | None = None,
) -> np.ndarray:
    if edge_index.shape[1] == 0:
        return np.empty((0, LOCAL_EDGE_DIM), dtype=np.float32)
    source_ids, destination_ids = edge_index
    source_position = source_positions if source_position_override is None else source_position_override
    delta = destination_positions[destination_ids] - source_position[source_ids]
    distance = np.linalg.norm(delta, axis=1, keepdims=True)
    source_normals = source_features[:, 9:12] if source_normal_override is None else source_normal_override
    normal_dot = np.sum(
        source_normals[source_ids] * destination_features[destination_ids, 9:12],
        axis=1,
        keepdims=True,
    )
    attributes = np.concatenate(
        [
            delta,
            distance,
            np.abs(destination_features[destination_ids, 0:1] - source_features[source_ids, 0:1]),
            np.abs(destination_features[destination_ids, 7:8] - source_features[source_ids, 7:8]),
            np.abs(destination_features[destination_ids, 8:9] - source_features[source_ids, 8:9]),
            normal_dot,
            source_features[source_ids, 13:14],
            destination_features[destination_ids, 13:14],
        ],
        axis=1,
    ).astype(np.float32)
    if attributes.shape[1] != LOCAL_EDGE_DIM:
        raise RuntimeError(f"Local edge attribute mismatch: {attributes.shape}")
    return attributes


def _subset_node(node: Mapping[str, Tensor], ids: np.ndarray) -> dict[str, Tensor]:
    index = torch.from_numpy(np.asarray(ids, dtype=np.int64))
    output: dict[str, Tensor] = {}
    count = int(next(iter(node.values())).shape[0])
    for key, value in node.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == count:
            output[key] = value[index]
        elif torch.is_tensor(value):
            output[key] = value.clone()
    output["full_id"] = index
    return output


def _induced_edge_index(
    full_edge_index: Tensor | np.ndarray,
    source_ids: np.ndarray,
    destination_ids: np.ndarray,
    *,
    source_count: int,
    destination_count: int,
) -> np.ndarray:
    """Retain every cached full-graph edge whose endpoints are in the view."""
    edge = _numpy(full_edge_index, dtype=np.int64)
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError(f"Canonical edge_index must be [2,E], got {edge.shape}")
    if edge.shape[1] == 0:
        return np.empty((2, 0), dtype=np.int64)
    if (
        int(edge[0].min()) < 0
        or int(edge[0].max()) >= int(source_count)
        or int(edge[1].min()) < 0
        or int(edge[1].max()) >= int(destination_count)
    ):
        raise ValueError("Canonical edge_index references an invalid full node id")
    source_map = np.full(int(source_count), -1, dtype=np.int64)
    destination_map = np.full(int(destination_count), -1, dtype=np.int64)
    source_ids = np.asarray(source_ids, dtype=np.int64)
    destination_ids = np.asarray(destination_ids, dtype=np.int64)
    source_map[source_ids] = np.arange(source_ids.size, dtype=np.int64)
    destination_map[destination_ids] = np.arange(destination_ids.size, dtype=np.int64)
    local_source = source_map[edge[0]]
    local_destination = destination_map[edge[1]]
    keep = (local_source >= 0) & (local_destination >= 0)
    if not np.any(keep):
        return np.empty((2, 0), dtype=np.int64)
    return np.stack([local_source[keep], local_destination[keep]], axis=0)


def build_local_view(
    source_local: Mapping[str, Any],
    target_local: Mapping[str, Any],
    config: GraphBuildConfig,
    *,
    seed: int,
) -> HeteroData:
    """Materialise one sampled radius-graph view from canonical node tables."""
    rng = np.random.default_rng(int(seed))
    source_nodes: Mapping[str, Mapping[str, Tensor]] = source_local["nodes"]
    target_nodes: Mapping[str, Mapping[str, Tensor]] = target_local["nodes"]
    all_nodes: dict[str, Mapping[str, Tensor]] = {**source_nodes, **target_nodes}

    if source_local.get("format") != "canonical-full-v22" or target_local.get("format") != "canonical-full-v22":
        raise ValueError("Sampled views require canonical-full-v22 cache payloads")

    selected: dict[str, np.ndarray] = {
        "tumor_surface": np.arange(
            int(source_nodes["tumor_surface"]["x"].shape[0]), dtype=np.int64
        ),
        "tumor_interior": np.arange(
            int(source_nodes["tumor_interior"]["x"].shape[0]), dtype=np.int64
        ),
        "source_liver_surface": np.arange(
            int(source_nodes["source_liver_surface"]["x"].shape[0]), dtype=np.int64
        ),
        "target_liver_surface": np.arange(
            int(target_nodes["target_liver_surface"]["x"].shape[0]), dtype=np.int64
        ),
    }

    source_surface_mm = _numpy(
        source_nodes["tumor_surface"]["pos_mm"], dtype=np.float32
    )
    transform = _numpy(target_local["transform"], dtype=np.float32)
    target_surface_mm = source_surface_mm @ transform.T
    selected["source_context"] = _select_context(
        source_nodes["source_context"], source_surface_mm, config, rng
    )
    selected["target_context"] = _select_context(
        target_nodes["target_context"], target_surface_mm, config, rng
    )

    graph = HeteroData()
    sampled: dict[str, dict[str, Tensor]] = {}
    for node_type in LOCAL_NODE_TYPES:
        ids = selected[node_type]
        if ids.size == 0:
            raise RuntimeError(f"V22 sampled zero nodes for {node_type}")
        sampled[node_type] = _subset_node(all_nodes[node_type], ids)
        for key, value in sampled[node_type].items():
            graph[node_type][key] = value

    features = {
        node_type: _numpy(sampled[node_type]["x"], dtype=np.float32)
        for node_type in LOCAL_NODE_TYPES
    }
    positions = {
        node_type: _numpy(sampled[node_type]["pos"], dtype=np.float32)
        for node_type in LOCAL_NODE_TYPES
    }
    virtual_surface_position = positions["tumor_surface"] @ transform.T
    normal_transform = np.linalg.inv(transform).T
    virtual_surface_normal = features["tumor_surface"][:, 9:12] @ normal_transform.T
    virtual_surface_normal /= np.maximum(
        np.linalg.norm(virtual_surface_normal, axis=1, keepdims=True), 1e-6
    )

    source_edges = source_local.get("edges")
    target_edges = target_local.get("edges")
    if not isinstance(source_edges, Mapping) or not isinstance(target_edges, Mapping):
        raise ValueError("Canonical full graph has no cached edge topology")
    full_edges = {**source_edges, **target_edges}
    if set(full_edges) != set(LOCAL_EDGE_TYPES):
        missing = [edge_type for edge_type in LOCAL_EDGE_TYPES if edge_type not in full_edges]
        extra = [edge_type for edge_type in full_edges if edge_type not in LOCAL_EDGE_TYPES]
        raise ValueError(f"Canonical edge schema mismatch: missing={missing}, extra={extra}")

    edge_counts: list[int] = []
    canonical_edge_counts: list[int] = []
    for edge_type in LOCAL_EDGE_TYPES:
        source_type, relation, destination_type = edge_type
        canonical_edge = full_edges[edge_type]
        canonical_edge_counts.append(int(canonical_edge.shape[1]))
        edge_index = _induced_edge_index(
            canonical_edge,
            selected[source_type],
            selected[destination_type],
            source_count=int(all_nodes[source_type]["x"].shape[0]),
            destination_count=int(all_nodes[destination_type]["x"].shape[0]),
        )
        edge_count = int(edge_index.shape[1])
        edge_limit = int(config.sample_relation_edge_limit)
        if edge_limit > 0 and edge_count > edge_limit:
            raise RuntimeError(
                "V22 induced relation exceeds sample_relation_edge_limit without truncation: "
                f"edge_type={edge_type}, edges={edge_count}, limit={edge_limit}"
            )
        edge_counts.append(edge_count)
        source_override_position = None
        source_override_normal = None
        if source_type == "tumor_surface" and relation == "interfaces_target":
            source_override_position = virtual_surface_position
            source_override_normal = virtual_surface_normal
        elif source_type == "source_context" and relation == "corresponds_to":
            source_override_position = positions["source_context"] @ transform.T
            source_normal = features["source_context"][:, 9:12] @ normal_transform.T
            source_normal /= np.maximum(
                np.linalg.norm(source_normal, axis=1, keepdims=True), 1e-6
            )
            source_override_normal = source_normal
        graph[edge_type].edge_index = torch.from_numpy(
            edge_index.astype(np.int64, copy=False)
        )
        graph[edge_type].edge_attr = torch.from_numpy(
            _edge_attributes(
                features[source_type],
                features[destination_type],
                positions[source_type],
                positions[destination_type],
                edge_index,
                source_position_override=source_override_position,
                source_normal_override=source_override_normal,
            )
        )

    graph.target_transform = torch.from_numpy(transform.astype(np.float32))[None]
    graph.full_schema = torch.tensor([22], dtype=torch.int16)
    graph.view_seed = torch.tensor([int(seed) & 0x7FFFFFFF], dtype=torch.int64)
    graph.canonical_counts = torch.tensor(
        [[int(all_nodes[node_type]["x"].shape[0]) for node_type in LOCAL_NODE_TYPES]],
        dtype=torch.int64,
    )
    graph.sampled_counts = torch.tensor(
        [[int(selected[node_type].size) for node_type in LOCAL_NODE_TYPES]],
        dtype=torch.int64,
    )
    graph.canonical_edge_counts = torch.tensor(
        [canonical_edge_counts], dtype=torch.int64
    )
    graph.relation_edge_counts = torch.tensor([edge_counts], dtype=torch.int64)
    return graph


def materialize_sample_views(
    sample: dict[str, Any],
    *,
    training: bool,
    epoch: int,
    global_seed: int,
) -> dict[str, Any]:
    """Replace cached canonical tables by two sampled HeteroData views."""
    if "local_graphs" in sample and "local_graphs_view2" in sample:
        return sample
    source_local = sample.get("source_local")
    targets = sample.get("target_locals")
    if not isinstance(source_local, Mapping) or not isinstance(targets, Sequence):
        raise ValueError("V22 cache payload has no canonical source/target node tables")
    from hiercp.schema import graph_config_from_dict

    config = graph_config_from_dict(dict(sample["graph_config"]))
    case_id = str(sample.get("case_id", ""))
    sample_index = int(sample.get("sample_index", -1))
    effective_epoch = int(epoch) if training else 0
    view1: list[HeteroData] = []
    view2: list[HeteroData] = []
    for candidate_index, target_local in enumerate(targets):
        seed1 = stable_view_seed(
            global_seed,
            case_id,
            sample_index,
            candidate_index,
            effective_epoch,
            0,
        )
        seed2 = stable_view_seed(
            global_seed,
            case_id,
            sample_index,
            candidate_index,
            effective_epoch,
            1,
        )
        view1.append(build_local_view(source_local, target_local, config, seed=seed1))
        view2.append(build_local_view(source_local, target_local, config, seed=seed2))
    sample["local_graphs"] = view1
    sample["local_graphs_view2"] = view2
    sample.pop("source_local", None)
    sample.pop("target_locals", None)
    return sample
