"""Explicit all-role seed sampling on the preserved v1 canonical topology.

This changes graph sampling density only. It never changes CNN inputs, physical
scope, role/edge meaning, model depth, the source mask or any ranking target.
Seed budgets are required configuration, not hidden final-node/edge caps. Paths
in the ORIGINAL topology may add relay nodes and their cost is always reported.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch

ROLES = ("tumor_surface", "tumor_interior", "source_context",
         "source_liver_surface", "target_context", "target_liver_surface")
SOURCE_ROLES = frozenset(ROLES[:4])
TARGET_ROLES = frozenset(ROLES[4:])
SAMPLING_CONTRACT = "v1_original_topology_role_shell_seeds_and_relays_v1"


@dataclass(frozen=True, init=False)
class GraphSeedProfile:
    """Immutable, explicit six-role SEED counts; no automatically chosen values."""

    role_seed_counts: tuple[int, ...]

    def __init__(self, budgets: Mapping[str, int]):
        if not isinstance(budgets, Mapping) or set(budgets) != set(ROLES):
            raise ValueError("Explicit seed budgets must contain exactly the six original v1 roles")
        counts = []
        for role in ROLES:
            count = budgets[role]
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
                raise ValueError(f"Positive integer SEED budget required for {role}: {count!r}")
            counts.append(int(count))
        object.__setattr__(self, "role_seed_counts", tuple(counts))

    @property
    def budgets(self) -> Mapping[str, int]:
        return MappingProxyType(dict(zip(ROLES, self.role_seed_counts)))

    @property
    def seed_total(self) -> int:
        return sum(self.role_seed_counts)

    def to_dict(self) -> dict[str, Any]:
        return {"contract": SAMPLING_CONTRACT, "role_seed_budgets": dict(self.budgets),
                "configured_seed_total": self.seed_total,
                "final_node_cap": None, "final_edge_cap": None,
                "relay_policy": "original weak-component shortest paths; no rewiring"}


# Descriptive public name used by the graph-size DEBUG runner.
SamplingProfile = GraphSeedProfile


def _array(value: torch.Tensor, name: str) -> np.ndarray:
    if not torch.is_tensor(value) or value.device.type != "cpu":
        raise ValueError(f"Canonical preparation table must be a CPU tensor: {name}")
    return value.detach().numpy()


def _validate(source: Mapping[str, Any], target: Mapping[str, Any], config):
    # Original modules resolve only after the caller activates its pinned source.
    from hiercp.schema import LOCAL_EDGE_TYPES, LOCAL_NODE_TYPES, LOCAL_HANDCRAFTED_DIM
    from hiercp.spatial import LEVEL0_GEOMETRY_CONTRACT
    if tuple(LOCAL_NODE_TYPES) != ROLES:
        raise ValueError("Active source is not the preserved six-role v1 schema")
    if config.geometry_contract != LEVEL0_GEOMETRY_CONTRACT:
        raise ValueError("Graph config has an incompatible physical geometry contract")
    config.validate()
    for name, branch, expected in (("source", source, SOURCE_ROLES), ("target", target, TARGET_ROLES)):
        if branch.get("format") != "canonical-full-v22" or branch.get("geometry_contract") != LEVEL0_GEOMETRY_CONTRACT:
            raise ValueError(f"{name} canonical format/geometry mismatch")
        if not isinstance(branch.get("nodes"), Mapping) or set(branch["nodes"]) != expected:
            raise ValueError(f"{name} canonical roles are incomplete or unexpected")
        if not isinstance(branch.get("edges"), Mapping):
            raise ValueError(f"{name} canonical edge table missing")
    nodes = {**source["nodes"], **target["nodes"]}
    arrays = {}
    for role in ROLES:
        row = nodes[role]
        if not isinstance(row, Mapping):
            raise ValueError(f"Canonical node table missing: {role}")
        item = {key: _array(row[key], f"{role}.{key}") for key in ("x", "grid", "pos", "pos_mm") if key in row}
        if set(item) != {"x", "grid", "pos", "pos_mm"}:
            raise ValueError(f"Canonical coordinates/features incomplete: {role}")
        count = item["x"].shape[0] if item["x"].ndim else 0
        if count < 1 or item["x"].shape != (count, LOCAL_HANDCRAFTED_DIM):
            raise ValueError(f"Invalid canonical feature shape: {role}")
        if any(item[key].shape != (count, 3) for key in ("grid", "pos", "pos_mm")):
            raise ValueError(f"Invalid canonical coordinate shape: {role}")
        if any(value.dtype.kind != "f" for value in item.values()):
            raise ValueError(f"Canonical coordinates/features must be floating point: {role}")
        if any(not np.isfinite(value).all() for value in item.values()):
            raise ValueError(f"Nonfinite canonical feature/coordinate: {role}")
        arrays[role] = item
    transform = _array(target.get("transform"), "target.transform")
    if transform.shape != (3, 3) or not np.isfinite(transform).all() or abs(float(np.linalg.det(transform))) < 1e-10:
        raise ValueError("Invalid/nonfinite/singular original target transform")
    if set(source["edges"]) & set(target["edges"]):
        raise ValueError("Duplicate canonical relation ownership")
    edges = {**source["edges"], **target["edges"]}
    if set(edges) != set(LOCAL_EDGE_TYPES):
        raise ValueError("Canonical relation schema must contain all original sixteen relations")
    edge_arrays = {}
    for relation in LOCAL_EDGE_TYPES:
        edge = _array(edges[relation], str(relation))
        if edge.ndim != 2 or edge.shape[0] != 2 or edge.dtype.kind not in "iu":
            raise ValueError(f"Invalid canonical edge index: {relation}")
        if edge.shape[1] and (edge.min() < 0 or edge[0].max() >= len(arrays[relation[0]]["x"])
                              or edge[1].max() >= len(arrays[relation[2]]["x"])):
            raise ValueError(f"Canonical edge endpoint outside its role: {relation}")
        edge_arrays[relation] = edge
    return nodes, arrays, edges, edge_arrays, tuple(LOCAL_EDGE_TYPES)


def _shells(features: np.ndarray) -> np.ndarray:
    from hiercp.schema import CONTEXT_SHELL_COUNT, CONTEXT_SHELL_FEATURE_INDEX
    values = np.rint(features[:, CONTEXT_SHELL_FEATURE_INDEX].astype(np.float64) * CONTEXT_SHELL_COUNT)
    if np.any(values < 0) or np.any(values > CONTEXT_SHELL_COUNT):
        raise ValueError("Original context shell feature outside its declared range")
    return values.astype(np.int64)


def _fps(positions: np.ndarray, count: int, required: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Deterministic-seeded farthest-point selection of ORIGINAL node indices."""
    total = len(positions)
    chosen = np.unique(np.asarray(required, dtype=np.int64))
    if chosen.size > count:
        raise ValueError("SEED budget cannot represent mandatory original relation/shell witnesses")
    if total <= count:
        return np.arange(total, dtype=np.int64)
    if not chosen.size:
        chosen = np.asarray([int(rng.integers(total))], dtype=np.int64)
    nearest = np.full(total, np.inf, dtype=np.float64)
    points = positions.astype(np.float64, copy=False)
    for index in chosen:
        delta = points - points[index]
        np.minimum(nearest, np.einsum("ij,ij->i", delta, delta), out=nearest)
    nearest[chosen] = -1.0
    output = chosen.tolist()
    while len(output) < count:
        index = int(np.argmax(nearest))
        output.append(index)
        delta = points - points[index]
        np.minimum(nearest, np.einsum("ij,ij->i", delta, delta), out=nearest)
        nearest[output] = -1.0
    return np.sort(np.asarray(output, dtype=np.int64))


def _role_seeds(role: str, array: Mapping[str, np.ndarray], budget: int,
                protected: set[int], rng: np.random.Generator) -> np.ndarray:
    count = min(int(budget), len(array["x"]))
    required = np.asarray(sorted(protected), dtype=np.int64)
    if "context" not in role:
        return _fps(array["pos_mm"], count, required, rng)
    shells = _shells(array["x"])
    present = np.unique(shells)
    for shell in present:
        ids = np.flatnonzero(shells == shell)
        if not any(int(index) in protected for index in ids):
            protected.add(int(ids[int(rng.integers(len(ids)))]))
    if len(protected) > count:
        raise ValueError(f"{role} SEED budget cannot preserve occupied shells and original relation witnesses")
    quota = {int(shell): sum(shells[index] == shell for index in protected) for shell in present}
    capacities = {int(shell): int(np.count_nonzero(shells == shell)) for shell in present}
    while sum(quota.values()) < count:
        available = [shell for shell in quota if quota[shell] < capacities[shell]]
        shell = min(available, key=lambda item: (quota[item], item))
        quota[shell] += 1
    selected = []
    for shell in present:
        ids = np.flatnonzero(shells == shell)
        required_local = np.flatnonzero(np.isin(ids, list(protected)))
        local = _fps(array["pos_mm"][ids], quota[int(shell)], required_local, rng)
        selected.extend(ids[local].tolist())
    return np.sort(np.asarray(selected, dtype=np.int64))


def _topology(arrays, edge_arrays, relations):
    from scipy import sparse
    offsets = {}
    total = 0
    for role in ROLES:
        offsets[role] = total
        total += len(arrays[role]["x"])
    dtype = np.int32 if total <= np.iinfo(np.int32).max else np.int64
    count = sum(edge_arrays[relation].shape[1] for relation in relations)
    rows = np.empty(count, dtype=dtype)
    columns = np.empty(count, dtype=dtype)
    start = 0
    for relation in relations:
        edge = edge_arrays[relation]
        stop = start + edge.shape[1]
        rows[start:stop] = edge[0] + offsets[relation[0]]
        columns[start:stop] = edge[1] + offsets[relation[2]]
        start = stop
    original = sparse.coo_matrix((np.ones(count, dtype=bool), (rows, columns)), shape=(total, total)).tocsr()
    original = original.maximum(original.T).tocsr()
    original.sort_indices()
    return original, offsets


def _with_original_paths(topology, offsets, seeds):
    from scipy.sparse.csgraph import breadth_first_order, connected_components
    component_count, labels = connected_components(topology, directed=False)
    selected_global = np.concatenate([seeds[role] + offsets[role] for role in ROLES])
    occupied = np.unique(labels[selected_global])
    keep = np.zeros(topology.shape[0], dtype=bool)
    keep[selected_global] = True
    for component in occupied:
        members = selected_global[labels[selected_global] == component]
        if len(members) < 2:
            continue
        root = int(members.min())
        _, predecessor = breadth_first_order(topology, root, directed=False, return_predecessors=True)
        for seed in members:
            current = int(seed)
            while current != root:
                parent = int(predecessor[current])
                if parent < 0:
                    raise RuntimeError("Original weak-component shortest-path lookup failed")
                keep[parent] = True
                current = parent
    selected = {}
    for index, role in enumerate(ROLES):
        stop = offsets[ROLES[index + 1]] if index + 1 < len(ROLES) else len(keep)
        selected[role] = np.flatnonzero(keep[offsets[role]:stop]).astype(np.int64)
    return selected, int(component_count), int(len(occupied)), labels, selected_global


def _relation_name(relation) -> str:
    return "|".join(relation)


def build_local_view(source_local: Mapping[str, Any], target_local: Mapping[str, Any],
                     config, *, seed: int, profile: GraphSeedProfile):
    """Return original v1 HeteroData plus full density/relay/connectivity audit.

    No failed pair is skipped, repaired by new edges or returned as a zero graph.
    Resource admission belongs to the caller, because relays have no hidden cap.
    """
    import time
    from scipy.sparse.csgraph import connected_components
    from hiercp.sample import _subset_node, _induced_edge_index, build_local_view as original_view
    if not isinstance(profile, GraphSeedProfile):
        raise TypeError("An explicit immutable GraphSeedProfile is required")
    began = time.perf_counter()
    nodes, arrays, edges, edge_arrays, relations = _validate(source_local, target_local, config)
    rng = np.random.default_rng(int(seed))
    protected = {role: set() for role in ROLES}
    witnesses = {}
    for relation in relations:
        edge = edge_arrays[relation]
        if not edge.shape[1]:
            continue
        column = int(rng.integers(edge.shape[1]))
        source_id, target_id = map(int, edge[:, column])
        protected[relation[0]].add(source_id)
        protected[relation[2]].add(target_id)
        witnesses[_relation_name(relation)] = [source_id, target_id]
    seeds = {role: _role_seeds(role, arrays[role], profile.budgets[role], protected[role], rng) for role in ROLES}
    topology, offsets = _topology(arrays, edge_arrays, relations)
    seed_global = np.concatenate([seeds[role] + offsets[role] for role in ROLES])
    seed_topology = topology[seed_global][:, seed_global]
    seed_component_count = int(connected_components(seed_topology, directed=False, return_labels=False))
    selected, original_components, occupied_components, labels, _ = _with_original_paths(topology, offsets, seeds)
    selected_global = np.concatenate([selected[role] + offsets[role] for role in ROLES])
    after_topology = topology[selected_global][:, selected_global]
    after_components = int(connected_components(after_topology, directed=False, return_labels=False))
    if after_components != occupied_components:
        raise RuntimeError("Original shortest-path relays did not preserve selected weak-component connectivity")
    reduced_nodes = {role: _subset_node(nodes[role], selected[role]) for role in ROLES}
    reduced_edges = {}
    seed_only_edge_counts = {}
    for relation in relations:
        a, _, b = relation
        induced = _induced_edge_index(edges[relation], selected[a], selected[b],
                                     source_count=len(arrays[a]["x"]), destination_count=len(arrays[b]["x"]))
        reduced_edges[relation] = torch.from_numpy(induced)
        seed_only_edge_counts[_relation_name(relation)] = int(_induced_edge_index(
            edges[relation], seeds[a], seeds[b], source_count=len(arrays[a]["x"]),
            destination_count=len(arrays[b]["x"])).shape[1])
        if edge_arrays[relation].shape[1] and not induced.shape[1]:
            raise RuntimeError(f"Nonempty original relation lost after seed/relay selection: {relation}")
    # Shallow payload wrappers share original metadata; no canonical edge clone.
    reduced_source = dict(source_local, nodes={role: reduced_nodes[role] for role in SOURCE_ROLES},
                          edges={relation: reduced_edges[relation] for relation in source_local["edges"]})
    reduced_target = dict(target_local, nodes={role: reduced_nodes[role] for role in TARGET_ROLES},
                          edges={relation: reduced_edges[relation] for relation in target_local["edges"]})
    # Internal only: already-complete node selection must not trigger native
    # 384-seed + interface-neighbour + two-hop expansion a second time.
    graph = original_view(reduced_source, reduced_target, replace(config, sample_context_nodes=0), seed=int(seed))
    for role in ROLES:
        graph[role].full_id = torch.from_numpy(selected[role])
    before_counts = {role: len(arrays[role]["x"]) for role in ROLES}
    after_counts = {role: len(selected[role]) for role in ROLES}
    seed_counts = {role: len(seeds[role]) for role in ROLES}
    relays = {role: after_counts[role] - seed_counts[role] for role in ROLES}
    graph.canonical_counts = torch.tensor([[before_counts[role] for role in ROLES]], dtype=torch.long)
    graph.canonical_edge_counts = torch.tensor([[edge_arrays[r].shape[1] for r in relations]], dtype=torch.long)
    graph.sampling_seed_counts = torch.tensor([[seed_counts[role] for role in ROLES]], dtype=torch.long)
    graph.sampling_relay_counts = torch.tensor([[relays[role] for role in ROLES]], dtype=torch.long)
    graph.graph_seed_sampling_contract = SAMPLING_CONTRACT
    shell_counts = {}
    components_by_role = {}
    for role in ROLES:
        ids = selected[role] + offsets[role]
        role_topology = topology[ids][:, ids]
        components_by_role[role] = int(connected_components(role_topology, directed=False, return_labels=False))
        if "context" in role:
            shells = _shells(arrays[role]["x"])
            shell_counts[role] = {str(int(shell)): {"canonical": int(np.count_nonzero(shells == shell)),
                "seeds": int(np.count_nonzero(shells[seeds[role]] == shell)),
                "final": int(np.count_nonzero(shells[selected[role]] == shell)),
                "selected_weak_components": int(connected_components(
                    topology[ids[shells[selected[role]] == shell]][:, ids[shells[selected[role]] == shell]],
                    directed=False, return_labels=False)) if np.count_nonzero(shells[selected[role]] == shell) else 0}
                for shell in np.unique(shells)}
    audit = {"contract": SAMPLING_CONTRACT, "profile": profile.to_dict(), "view_seed": int(seed),
        "original_graph_config": config.to_dict(), "physical_range_unchanged": True,
        "node_features_aggregated": False, "edge_rewiring": False,
        "original_all_inputs_verified": True, "directed_reachability": "NOT_CHECKED",
        "canonical_before": {"nodes": sum(before_counts.values()), "edges": sum(edge_arrays[r].shape[1] for r in relations),
                             "nodes_by_role": before_counts, "edges_by_relation": {_relation_name(r): int(edge_arrays[r].shape[1]) for r in relations}},
        "seed_only": {"nodes": sum(seed_counts.values()), "nodes_by_role": seed_counts,
                      "edges_by_relation": seed_only_edge_counts, "weak_components": seed_component_count},
        "after": {"nodes": sum(after_counts.values()), "edges": sum(graph[r].edge_index.shape[1] for r in relations),
                  "nodes_by_role": after_counts, "edges_by_relation": {_relation_name(r): int(graph[r].edge_index.shape[1]) for r in relations},
                  "weak_components": after_components,
                  "weak_components_by_role": components_by_role,
                  "isolated_nodes": int(np.count_nonzero(np.diff(after_topology.indptr) == 0))},
        "relay_nodes": sum(relays.values()), "relay_nodes_by_role": relays,
        "actual_nodes_above_configured_seed_total": max(0, sum(after_counts.values()) - profile.seed_total),
        "original_weak_components": original_components, "occupied_original_weak_components": occupied_components,
        "connectivity_scope": "weak original-topology connectivity; not directed message reachability",
        "shell_counts": shell_counts, "original_relation_witnesses": witnesses,
        "selected_original_ids": {role: selected[role].tolist() for role in ROLES},
        "seed_original_ids": {role: seeds[role].tolist() for role in ROLES},
        "preparation_seconds": time.perf_counter() - began,
        "quality_verified": False, "production_ready": False}
    return graph, audit
