"""Strict nested reduction of ONE exact native sampled-v1 HeteroData view.

Unlike the preserved canonical-resampling experiment, this module cannot draw a
node or relay from outside the supplied native sampled view. Edges, attributes,
ordering, coordinates, features and canonical full_ids are indexed unchanged.
It never changes a model, CNN input, GT, loss, mask or receptive-field setting.
"""
from __future__ import annotations

from typing import Any
import copy
import time

import numpy as np
import torch

# A file-loader must register the CURRENT graph_size module under this name
# before activating an archived helper package; the DEBUG runner does so.
from hiercp_v1x.graph_size import (
    GraphSeedProfile, ROLES, _role_seeds, _topology, _with_original_paths, _shells,
)

SamplingProfile = GraphSeedProfile
SAMPLING_CONTRACT = "v1_strict_nested_native_view_role_shell_seeds_relays_v1"


def _array(value, name: str) -> np.ndarray:
    if not torch.is_tensor(value) or value.device.type != "cpu":
        raise ValueError(f"Native sampled preparation tensor must be on CPU: {name}")
    return value.detach().numpy()


def _validate(original, config, seed: int):
    from torch_geometric.data import HeteroData, Batch
    from hiercp.schema import LOCAL_NODE_TYPES, LOCAL_EDGE_TYPES, LOCAL_HANDCRAFTED_DIM, LOCAL_EDGE_DIM
    from hiercp.sample import FULL_INDUCED_EDGE_CONTRACT
    if not isinstance(original, HeteroData) or isinstance(original, Batch):
        raise TypeError("One unbatched exact native HeteroData sampled view is required")
    if original.get("graph_seed_sampling_contract") is not None:
        raise ValueError("Nested reduction must start from an exact native view, not an already-modified sampler view")
    if tuple(LOCAL_NODE_TYPES) != ROLES or len(LOCAL_EDGE_TYPES) != 16:
        raise ValueError("Active original schema must have six roles and sixteen directed relations")
    if set(original.node_types) != set(LOCAL_NODE_TYPES) or set(original.edge_types) != set(LOCAL_EDGE_TYPES):
        raise ValueError("Native sampled role/relation schema incomplete or unexpected")
    config.validate()
    for key, shape in (("canonical_counts", (1, 6)), ("sampled_counts", (1, 6)),
                       ("canonical_edge_counts", (1, 16)), ("relation_edge_counts", (1, 16))):
        value = _array(original.get(key), key)
        if value.shape != shape or value.dtype.kind not in "iu" or (value < 0).any():
            raise ValueError(f"Invalid native original count metadata: {key}")
    if original.get("edge_execution_contract") != FULL_INDUCED_EDGE_CONTRACT:
        raise ValueError("Native original must preserve the full induced-edge contract")
    version = _array(original.get("full_schema"), "full_schema")
    parent_seed = _array(original.get("view_seed"), "view_seed")
    if version.shape != (1,) or int(version[0]) != 22:
        raise ValueError("Native original sampled schema marker mismatch")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("Explicit integer native view seed required")
    if parent_seed.shape != (1,) or parent_seed.dtype.kind not in "iu" or int(parent_seed[0]) != (int(seed) & 0x7FFFFFFF):
        raise ValueError("Nested sampler seed differs from the exact native view seed")
    transform = _array(original.get("target_transform"), "target_transform")
    if transform.shape != (1, 3, 3) or transform.dtype.kind != "f" or not np.isfinite(transform).all():
        raise ValueError("Invalid native target transform")
    if abs(float(np.linalg.det(transform[0]))) < 1e-10:
        raise ValueError("Singular native target transform")
    arrays = {}
    for role_index, role in enumerate(ROLES):
        store = original[role]
        required = ("x", "grid", "pos", "pos_mm", "full_id")
        if any(key not in store for key in required):
            raise ValueError(f"Native original node fields missing: {role}")
        item = {key: _array(store[key], f"{role}.{key}") for key in required}
        count = item["x"].shape[0] if item["x"].ndim else 0
        if count < 1 or item["x"].shape != (count, LOCAL_HANDCRAFTED_DIM):
            raise ValueError(f"Invalid native sampled feature shape: {role}")
        if any(item[key].shape != (count, 3) for key in ("grid", "pos", "pos_mm")):
            raise ValueError(f"Invalid native sampled coordinate shape: {role}")
        if any(item[key].dtype.kind != "f" or not np.isfinite(item[key]).all() for key in ("x", "grid", "pos", "pos_mm")):
            raise ValueError(f"Nonfinite or nonfloating native feature/coordinate: {role}")
        full_id = item["full_id"]
        canonical_count = int(original.canonical_counts[0, role_index])
        if full_id.shape != (count,) or full_id.dtype.kind not in "iu" or len(np.unique(full_id)) != count:
            raise ValueError(f"Invalid/nonunique canonical full_id: {role}")
        if (full_id < 0).any() or (full_id >= canonical_count).any():
            raise ValueError(f"Canonical full_id outside original count: {role}")
        if int(original.sampled_counts[0, role_index]) != count:
            raise ValueError(f"Native sampled node-count metadata mismatch: {role}")
        for key, value in store.items():
            if torch.is_tensor(value):
                _array(value, f"{role}.{key}")
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Nonfinite native node field: {role}.{key}")
        arrays[role] = item
    edge_arrays = {}
    for relation_index, relation in enumerate(LOCAL_EDGE_TYPES):
        store = original[relation]
        if "edge_index" not in store or "edge_attr" not in store:
            raise ValueError(f"Native original relation data missing: {relation}")
        edge = _array(store.edge_index, f"{relation}.edge_index")
        attributes = _array(store.edge_attr, f"{relation}.edge_attr")
        if edge.ndim != 2 or edge.shape[0] != 2 or edge.dtype.kind not in "iu":
            raise ValueError(f"Invalid native sampled edge index: {relation}")
        if attributes.shape != (edge.shape[1], LOCAL_EDGE_DIM) or attributes.dtype.kind != "f" or not np.isfinite(attributes).all():
            raise ValueError(f"Invalid/nonfinite native ten-dimensional edge attributes: {relation}")
        if edge.shape[1] and (edge.min() < 0 or edge[0].max() >= len(arrays[relation[0]]["x"])
                              or edge[1].max() >= len(arrays[relation[2]]["x"])):
            raise ValueError(f"Native sampled edge endpoint outside its role: {relation}")
        if int(original.relation_edge_counts[0, relation_index]) != edge.shape[1]:
            raise ValueError(f"Native sampled edge-count metadata mismatch: {relation}")
        if int(original.canonical_edge_counts[0, relation_index]) < edge.shape[1]:
            raise ValueError(f"Canonical edge count smaller than native sampled count: {relation}")
        for key, value in store.items():
            if torch.is_tensor(value):
                _array(value, f"{relation}.{key}")
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Nonfinite native relation field: {relation}.{key}")
        edge_arrays[relation] = edge
    return arrays, edge_arrays, tuple(LOCAL_EDGE_TYPES)


def _induced_with_parent_indices(edge: np.ndarray, left: np.ndarray, right: np.ndarray,
                                 left_count: int, right_count: int):
    from hiercp.sample import EDGE_MATERIALIZATION_CHUNK_EDGES
    lm = np.full(left_count, -1, dtype=np.int64)
    rm = np.full(right_count, -1, dtype=np.int64)
    lm[left] = np.arange(len(left), dtype=np.int64)
    rm[right] = np.arange(len(right), dtype=np.int64)
    pieces = []
    for start in range(0, edge.shape[1], EDGE_MATERIALIZATION_CHUNK_EDGES):
        stop = min(edge.shape[1], start + EDGE_MATERIALIZATION_CHUNK_EDGES)
        keep = (lm[edge[0, start:stop]] >= 0) & (rm[edge[1, start:stop]] >= 0)
        pieces.append(np.flatnonzero(keep).astype(np.int64) + start)
    parent = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
    induced = np.stack((lm[edge[0, parent]], rm[edge[1, parent]]))
    return induced, parent


def _coverage(points: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    from scipy.spatial import cKDTree
    retained = points[selected].astype(np.float64)
    distances, _ = cKDTree(retained).query(points.astype(np.float64), k=1)
    reverse_distances, _ = cKDTree(points.astype(np.float64)).query(retained, k=1)
    return {"parent_nodes": int(len(points)), "selected_nodes": int(len(selected)),
        "parent_to_nearest_selected_mm": {"mean": float(np.mean(distances)),
            "median": float(np.median(distances)), "p95": float(np.quantile(distances, .95)),
            "max": float(np.max(distances))},
        "selected_to_nearest_parent_mm_max": float(np.max(reverse_distances)),
        "parent_bbox_min_mm": points.min(axis=0).astype(float).tolist(),
        "parent_bbox_max_mm": points.max(axis=0).astype(float).tolist(),
        "selected_bbox_min_mm": retained.min(axis=0).tolist(),
        "selected_bbox_max_mm": retained.max(axis=0).tolist()}


def _verify_exact_subset(original, nested, selected, parent_edge_indices, relations):
    """Verify materialized slices before writing affirmative audit fields."""
    for role in ROLES:
        ids = torch.from_numpy(selected[role])
        if not torch.equal(nested[role].native_parent_id, ids):
            raise RuntimeError(f"Nested native-parent identity mismatch: {role}")
        for key in ("x", "grid", "pos", "pos_mm", "full_id"):
            if not torch.equal(nested[role][key], original[role][key][ids]):
                raise RuntimeError(f"Nested original node-field mismatch: {role}.{key}")
    for relation in relations:
        parent = torch.tensor(parent_edge_indices["|".join(relation)], dtype=torch.long)
        if not torch.equal(nested[relation].native_parent_edge_id, parent):
            raise RuntimeError(f"Nested parent-edge identity mismatch: {relation}")
        if not torch.equal(nested[relation].edge_attr, original[relation].edge_attr[parent]):
            raise RuntimeError(f"Nested edge-attribute/order mismatch: {relation}")
        left = nested[relation[0]].native_parent_id[nested[relation].edge_index[0]]
        right = nested[relation[2]].native_parent_id[nested[relation].edge_index[1]]
        if not torch.equal(torch.stack((left, right)), original[relation].edge_index[:, parent]):
            raise RuntimeError(f"Nested induced parent-edge endpoint/order mismatch: {relation}")


def build_local_view(original, config, *, seed: int, profile: GraphSeedProfile):
    """Return Vsmall⊆Vnative and EXACT Eoriginal restricted to selected endpoints.

    All original edge attributes/order and canonical identities are retained.
    Relay nodes come from the supplied sampled graph only. No automatic budget
    change, skipping, invented connections or quality-readiness claim exists.
    """
    from torch_geometric.data import HeteroData
    from scipy.sparse.csgraph import connected_components
    if not isinstance(profile, GraphSeedProfile):
        raise TypeError("Explicit immutable six-role GraphSeedProfile required")
    began = time.perf_counter()
    arrays, edge_arrays, relations = _validate(original, config, seed)
    rng = np.random.default_rng(int(seed))
    protected = {role: set() for role in ROLES}
    witnesses = {}
    for relation in relations:
        edge = edge_arrays[relation]
        if edge.shape[1]:
            column = int(rng.integers(edge.shape[1]))
            left, right = map(int, edge[:, column])
            protected[relation[0]].add(left)
            protected[relation[2]].add(right)
            witnesses["|".join(relation)] = {"native_parent_edge_index": column,
                                           "native_parent_node_indices": [left, right]}
    seeds = {role: _role_seeds(role, arrays[role], profile.budgets[role], protected[role], rng) for role in ROLES}
    topology, offsets = _topology(arrays, edge_arrays, relations)
    selected, original_components, occupied_components, _, _ = _with_original_paths(topology, offsets, seeds)
    global_selected = np.concatenate([selected[role] + offsets[role] for role in ROLES])
    global_seeds = np.concatenate([seeds[role] + offsets[role] for role in ROLES])
    after_topology = topology[global_selected][:, global_selected]
    after_components = int(connected_components(after_topology, directed=False, return_labels=False))
    if after_components != occupied_components:
        raise RuntimeError("Nested native BFS relay failed selected weak-component connectivity")
    graph = HeteroData()
    for key, value in original._global_store.items():
        graph[key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
    for role in ROLES:
        indices = torch.from_numpy(selected[role])
        count = len(arrays[role]["x"])
        for key, value in original[role].items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == count:
                graph[role][key] = value[indices]
            else:
                graph[role][key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        graph[role].native_parent_id = indices
    edge_parent_indices = {}
    seed_edge_counts = {}
    for relation in relations:
        a, _, b = relation
        edge, parent = _induced_with_parent_indices(edge_arrays[relation], selected[a], selected[b],
                                                    len(arrays[a]["x"]), len(arrays[b]["x"]))
        if edge_arrays[relation].shape[1] and not edge.shape[1]:
            raise RuntimeError(f"Nonempty native relation lost: {relation}")
        indices = torch.from_numpy(parent)
        for key, value in original[relation].items():
            if key == "edge_index":
                graph[relation].edge_index = torch.from_numpy(edge)
            elif torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == edge_arrays[relation].shape[1]:
                graph[relation][key] = value[indices]
            else:
                graph[relation][key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        graph[relation].native_parent_edge_id = indices
        edge_parent_indices["|".join(relation)] = parent.tolist()
        seed_edge_counts["|".join(relation)] = int(_induced_with_parent_indices(
            edge_arrays[relation], seeds[a], seeds[b], len(arrays[a]["x"]), len(arrays[b]["x"]))[0].shape[1])
    _verify_exact_subset(original, graph, selected, edge_parent_indices, relations)
    before_counts = {role: len(arrays[role]["x"]) for role in ROLES}
    after_counts = {role: len(selected[role]) for role in ROLES}
    seed_counts = {role: len(seeds[role]) for role in ROLES}
    relays = {role: after_counts[role] - seed_counts[role] for role in ROLES}
    edge_counts = [int(graph[relation].edge_index.shape[1]) for relation in relations]
    graph.sampled_counts = torch.tensor([[after_counts[role] for role in ROLES]], dtype=torch.long)
    graph.relation_edge_counts = torch.tensor([edge_counts], dtype=torch.long)
    limit = int(original.legacy_sample_relation_edge_limit[0])
    graph.relation_exceeds_legacy_limit = torch.tensor([[limit > 0 and count > limit for count in edge_counts]], dtype=torch.bool)
    graph.sampling_seed_counts = torch.tensor([[seed_counts[role] for role in ROLES]], dtype=torch.long)
    graph.sampling_relay_counts = torch.tensor([[relays[role] for role in ROLES]], dtype=torch.long)
    graph.native_parent_sampled_counts = original.sampled_counts.clone()
    graph.native_parent_relation_edge_counts = original.relation_edge_counts.clone()
    graph.graph_seed_sampling_contract = SAMPLING_CONTRACT
    coverage = {}
    shell_counts = {}
    components_by_role = {}
    for role in ROLES:
        ids = selected[role] + offsets[role]
        parent_ids = np.arange(before_counts[role], dtype=np.int64) + offsets[role]
        components_by_role[role] = {
            "original": int(connected_components(topology[parent_ids][:, parent_ids], directed=False, return_labels=False)),
            "selected": int(connected_components(topology[ids][:, ids], directed=False, return_labels=False))}
        coverage[role] = _coverage(arrays[role]["pos_mm"], selected[role])
        if "context" in role:
            shells = _shells(arrays[role]["x"])
            shell_counts[role] = {}
            for shell in np.unique(shells):
                parent_shell = np.flatnonzero(shells == shell)
                retained_shell = selected[role][shells[selected[role]] == shell]
                retained_within_shell = np.flatnonzero(np.isin(parent_shell, retained_shell))
                shell_ids = retained_shell + offsets[role]
                shell_counts[role][str(int(shell))] = {
                    "original": int(len(parent_shell)), "seeds": int(np.count_nonzero(shells[seeds[role]] == shell)),
                    "final": int(len(retained_shell)),
                    "selected_weak_components": int(connected_components(topology[shell_ids][:, shell_ids], directed=False, return_labels=False)),
                    "coverage": _coverage(arrays[role]["pos_mm"][parent_shell], retained_within_shell)}
    audit = {"contract": SAMPLING_CONTRACT, "profile": {**profile.to_dict(), "contract": SAMPLING_CONTRACT},
        "view_seed": int(seed), "original_graph_config": config.to_dict(),
        "selection_domain": "exact supplied native sampled view only", "strict_nested": True,
        "vertices_subset_verified": True, "induced_edges_verified": True,
        "node_coordinates_features_full_ids_exact": True, "edge_attributes_order_exact": True,
        "nodes_outside_native_view": 0, "edge_rewiring": False, "node_features_aggregated": False,
        "physical_range_unchanged": True,
        "range_scope": "eligible native domain/CNN/receptive-field settings unchanged; realized coverage reported",
        "original_all_inputs_verified": True, "directed_reachability": "NOT_CHECKED",
        "native_before": {"nodes": sum(before_counts.values()), "edges": sum(edge_arrays[r].shape[1] for r in relations),
            "nodes_by_role": before_counts, "edges_by_relation": {"|".join(r): int(edge_arrays[r].shape[1]) for r in relations}},
        "seed_only": {"nodes": sum(seed_counts.values()), "nodes_by_role": seed_counts,
            "edges_by_relation": seed_edge_counts, "weak_components": int(connected_components(
                topology[global_seeds][:, global_seeds], directed=False, return_labels=False))},
        "after": {"nodes": sum(after_counts.values()), "edges": sum(edge_counts), "nodes_by_role": after_counts,
            "edges_by_relation": {"|".join(r): edge_counts[i] for i, r in enumerate(relations)},
            "weak_components": after_components, "weak_components_by_role": components_by_role,
            "isolated_nodes": int(np.count_nonzero(np.diff(after_topology.indptr) == 0))},
        "relay_nodes": sum(relays.values()), "relay_nodes_by_role": relays,
        "actual_nodes_above_configured_seed_total": max(0, sum(after_counts.values()) - profile.seed_total),
        "original_weak_components": original_components, "occupied_original_weak_components": occupied_components,
        "connectivity_scope": "native sampled weak topology only; not directed GAT information preservation",
        "original_relation_witnesses": witnesses, "shell_counts": shell_counts, "coverage_by_role": coverage,
        "selected_native_indices": {role: selected[role].tolist() for role in ROLES},
        "seed_native_indices": {role: seeds[role].tolist() for role in ROLES},
        "selected_canonical_full_ids": {role: graph[role].full_id.tolist() for role in ROLES},
        "induced_parent_edge_indices": edge_parent_indices,
        "preparation_seconds": time.perf_counter() - began,
        "quality_verified": False, "production_ready": False}
    return graph, audit


sample_native_view = build_local_view
