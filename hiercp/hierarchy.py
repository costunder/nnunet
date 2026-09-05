"""Build Level-1 patient and Level-2 population-prototype PyG graphs."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage as ndi
import torch

try:
    from torch_geometric.data import HeteroData
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch Geometric is required. Run: python -m tools.install"
    ) from exc

from hiercp.common import (
    LoadedCase,
    SourceTumor,
    bbox_of_mask,
    context_stats_for_local_mask,
    distance_to_mask_mm,
    normalized_position,
)
from hiercp.curriculum import CandidateSpec
from hiercp.prototype import PrototypeBank
from hiercp.region import PatientRegionData, upper_geometry_vector
from hiercp.schema import (
    PATIENT_EDGE_DIM,
    PATIENT_EDGE_TYPES,
    PATIENT_NODE_TYPES,
    PROTOTYPE_EDGE_DIM,
    PROTOTYPE_EDGE_TYPES,
    PROTOTYPE_FEATURE_DIM,
    PROTOTYPE_NODE_TYPES,
    REGION_FEATURE_DIM,
    UPPER_RAW_DIM,
    GraphBuildConfig,
)


def _full_bipartite(num_source: int, num_destination: int) -> np.ndarray:
    if num_source <= 0 or num_destination <= 0:
        return np.empty((2, 0), dtype=np.int64)
    source = np.repeat(np.arange(num_source, dtype=np.int64), num_destination)
    destination = np.tile(np.arange(num_destination, dtype=np.int64), num_source)
    return np.stack([source, destination], axis=0)


def _knn(position: np.ndarray, k: int) -> np.ndarray:
    count = int(position.shape[0])
    if count <= 1:
        return np.empty((2, 0), dtype=np.int64)
    effective = min(max(1, int(k)), count - 1)
    distances = np.linalg.norm(position[:, None] - position[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argpartition(distances, kth=effective - 1, axis=1)[:, :effective]
    source = neighbors.reshape(-1).astype(np.int64)
    destination = np.repeat(np.arange(count, dtype=np.int64), effective)
    return np.stack([source, destination], axis=0)


def _principal_axis(mask: np.ndarray, spacing: np.ndarray) -> tuple[np.ndarray, float]:
    coordinates = np.column_stack(np.where(mask)).astype(np.float32) * spacing[None]
    if coordinates.shape[0] < 3:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32), 1.0
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(coordinates.shape[0] - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axis = vectors[:, order[0]].astype(np.float32)
    axis /= max(float(np.linalg.norm(axis)), 1e-6)
    anisotropy = float(
        np.sqrt(max(float(values[order[0]]), 1e-6) / max(float(values[order[-1]]), 1e-6))
    )
    return axis, min(anisotropy, 10.0)


def _surface_normal(depth: np.ndarray, center: Sequence[int], spacing: np.ndarray) -> np.ndarray:
    """Evaluate the depth-field gradient at one voxel with a local stencil.

    This is equivalent to ``np.gradient(..., edge_order=1)`` at the requested
    coordinate, but avoids allocating three full-volume gradient arrays for
    every candidate.
    """

    coordinate = [int(value) for value in center]
    vector = np.empty(3, dtype=np.float32)
    for axis in range(3):
        size = int(depth.shape[axis])
        index = coordinate[axis]
        step = max(float(spacing[axis]), 1e-6)
        lower = coordinate.copy()
        upper = coordinate.copy()
        if index <= 0:
            lower[axis] = 0
            upper[axis] = min(1, size - 1)
            denominator = step
        elif index >= size - 1:
            lower[axis] = max(0, size - 2)
            upper[axis] = size - 1
            denominator = step
        else:
            lower[axis] = index - 1
            upper[axis] = index + 1
            denominator = 2.0 * step
        vector[axis] = (
            float(depth[tuple(upper)]) - float(depth[tuple(lower)])
        ) / denominator
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return vector / norm


def _context_stats(
    case: LoadedCase,
    mask: np.ndarray,
    full_organ: np.ndarray,
) -> tuple[float, float]:
    slices = bbox_of_mask(mask, pad=3)
    return context_stats_for_local_mask(
        case.image[slices],
        full_organ[slices],
        mask[slices],
        ring_width=2,
    )


def _source_raw(
    case: LoadedCase,
    source: SourceTumor,
    regions: PatientRegionData,
    occupied_without_source: np.ndarray,
    *,
    ct_clip: tuple[float, float],
) -> np.ndarray:
    mean, std = _context_stats(case, source.full_mask, regions.full_organ_mask)
    other_distance = distance_to_mask_mm(occupied_without_source, case.spacing)
    axis, anisotropy = _principal_axis(source.full_mask, case.spacing)
    normal = _surface_normal(regions.organ_depth, source.anchor_center, case.spacing)
    depth = float(regions.organ_depth[source.anchor_center])
    return upper_geometry_vector(
        center=source.anchor_center,
        shape=case.shape,
        border_distance_mm=depth,
        occupied_distance_mm=float(other_distance[source.anchor_center]),
        context_mean_hu=mean,
        context_std_hu=std,
        volume_vox=source.voxel_count,
        coverage=1.0,
        local_thickness_mm=max(2.0 * depth, float(np.min(case.spacing))),
        surface_alignment=float(abs(np.dot(axis, normal))),
        scale_mean=1.0,
        anisotropy=anisotropy,
        ct_clip=ct_clip,
    )


def _candidate_raw(
    case: LoadedCase,
    source: SourceTumor,
    spec: CandidateSpec,
    regions: PatientRegionData,
    *,
    source_axis: np.ndarray,
    source_anisotropy: float,
    ct_clip: tuple[float, float],
) -> np.ndarray:
    transformed_axis = spec.rotation_matrix @ (source_axis * spec.scale_array)
    transformed_axis /= max(float(np.linalg.norm(transformed_axis)), 1e-6)
    normal = _surface_normal(regions.organ_depth, spec.center, case.spacing)
    depth = float(spec.border_distance_mm)
    scale = spec.scale_array
    return upper_geometry_vector(
        center=spec.center,
        shape=case.shape,
        border_distance_mm=depth,
        occupied_distance_mm=spec.occupied_distance_mm,
        context_mean_hu=spec.context_mean_hu,
        context_std_hu=spec.context_std_hu,
        volume_vox=int(round(source.voxel_count * float(np.prod(scale)))),
        coverage=spec.liver_coverage,
        local_thickness_mm=max(2.0 * depth, float(np.min(case.spacing))),
        surface_alignment=float(abs(np.dot(transformed_axis, normal))),
        scale_mean=float(scale.mean()),
        anisotropy=max(source_anisotropy, float(scale.max() / max(float(scale.min()), 1e-6))),
        ct_clip=ct_clip,
    )


def _lesions(
    case: LoadedCase,
    source: SourceTumor,
    regions: PatientRegionData,
    *,
    tumor_label: int,
    max_lesions: int | None,
    ct_clip: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occupied = (case.label == int(tumor_label)) & ~source.full_mask
    components, count = ndi.label(occupied, structure=ndi.generate_binary_structure(3, 1))
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    component_ids = sorted(
        range(1, count + 1),
        key=lambda component: int(sizes[component]),
        reverse=True,
    )
    lesion_limit = None if max_lesions is None else int(max_lesions)
    if lesion_limit is not None and lesion_limit > 0 and count > lesion_limit:
        case_id = str(getattr(getattr(case, "paths", None), "case_id", "<unknown>"))
        component_sizes = [int(sizes[component_id]) for component_id in component_ids]
        raise RuntimeError(
            "Patient graph lesion count exceeds graph.max_lesions; no lesions were "
            "dropped. "
            f"case_id={case_id}, detected_lesions_excluding_source={count}, "
            f"configured_max_lesions={lesion_limit}, "
            f"component_sizes_voxels_desc={component_sizes}. "
            "Increase graph.max_lesions after measuring graph memory/throughput, or "
            "use a nonpositive/None limit for unlimited lesions."
        )
    rows: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    region_ids: list[int] = []
    for component_id in component_ids:
        mask = components == component_id
        center_float = ndi.center_of_mass(mask)
        center = tuple(
            int(np.clip(round(value), 0, case.shape[axis] - 1))
            for axis, value in enumerate(center_float)
        )
        mean, std = _context_stats(case, mask, regions.full_organ_mask)
        depth = float(regions.organ_depth[center])
        rows.append(
            upper_geometry_vector(
                center=center,
                shape=case.shape,
                border_distance_mm=depth,
                occupied_distance_mm=0.0,
                context_mean_hu=mean,
                context_std_hu=std,
                volume_vox=int(mask.sum()),
                coverage=1.0,
                local_thickness_mm=max(2.0 * depth, float(np.min(case.spacing))),
                surface_alignment=0.0,
                scale_mean=1.0,
                anisotropy=1.0,
                valid=1.0,
                ct_clip=ct_clip,
            )
        )
        positions.append(normalized_position(center, case.shape).astype(np.float32))
        region_ids.append(regions.region_at(center))
    if not rows:
        return (
            np.empty((0, UPPER_RAW_DIM), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.stack(rows).astype(np.float32),
        np.stack(positions).astype(np.float32),
        np.asarray(region_ids, dtype=np.int64),
    )


def _liver_raw(
    case: LoadedCase,
    regions: PatientRegionData,
    *,
    tumor_label: int,
    ct_clip: tuple[float, float],
) -> np.ndarray:
    mask = regions.full_organ_mask
    center = ndi.center_of_mass(mask)
    values = case.image[mask]
    mean = float(values.mean()) if values.size else 0.0
    std = float(values.std()) if values.size else 0.0
    depth_values = regions.organ_depth[mask]
    tumor_count = int(ndi.label(case.label == int(tumor_label))[1])
    max_depth = float(depth_values.max()) if depth_values.size else 0.0
    return upper_geometry_vector(
        center=center,
        shape=case.shape,
        border_distance_mm=float(depth_values.mean()) if depth_values.size else 0.0,
        occupied_distance_mm=np.inf,
        context_mean_hu=mean,
        context_std_hu=std,
        volume_vox=int(mask.sum()),
        coverage=1.0,
        local_thickness_mm=2.0 * max_depth,
        surface_alignment=np.clip(tumor_count / 10.0, 0.0, 1.0),
        scale_mean=1.0,
        anisotropy=1.0,
        valid=1.0,
        ct_clip=ct_clip,
    )


def _node_meta(raw: np.ndarray, node_type: str) -> np.ndarray:
    if node_type == "region":
        # depth mean, CT mean, valid
        return np.column_stack([raw[:, 3], raw[:, 6], raw[:, 15]]).astype(np.float32)
    return np.column_stack([raw[:, 3], raw[:, 5], raw[:, 13]]).astype(np.float32)


def _patient_edge_attributes(
    graph: HeteroData,
    edge_type: tuple[str, str, str],
    edge_index: np.ndarray,
    relation_scalar: np.ndarray | float = 1.0,
) -> np.ndarray:
    source_type, _, destination_type = edge_type
    if edge_index.shape[1] == 0:
        return np.empty((0, PATIENT_EDGE_DIM), dtype=np.float32)
    source_ids, destination_ids = edge_index
    source_position = graph[source_type].pos.numpy()[source_ids]
    destination_position = graph[destination_type].pos.numpy()[destination_ids]
    delta = destination_position - source_position
    distance = np.linalg.norm(delta, axis=1, keepdims=True)
    source_meta = graph[source_type].meta.numpy()[source_ids]
    destination_meta = graph[destination_type].meta.numpy()[destination_ids]
    scalar = np.broadcast_to(
        np.asarray(relation_scalar, dtype=np.float32),
        (edge_index.shape[1],),
    )[:, None]
    source_region = graph[source_type].region_index.numpy()[source_ids]
    destination_region = graph[destination_type].region_index.numpy()[destination_ids]
    same_region = (
        (source_region >= 0)
        & (destination_region >= 0)
        & (source_region == destination_region)
    ).astype(np.float32)[:, None]
    attributes = np.concatenate(
        [
            delta,
            distance,
            source_meta,
            destination_meta,
            scalar,
            same_region,
        ],
        axis=1,
    ).astype(np.float32)
    if attributes.shape[1] != PATIENT_EDGE_DIM:
        raise RuntimeError(f"Patient edge attribute mismatch: {attributes.shape}")
    return attributes


def _set_patient_relation(
    graph: HeteroData,
    edge_type: tuple[str, str, str],
    edge_index: np.ndarray,
    relation_scalar: np.ndarray | float = 1.0,
) -> None:
    graph[edge_type].edge_index = torch.from_numpy(edge_index.astype(np.int64))
    graph[edge_type].edge_attr = torch.from_numpy(
        _patient_edge_attributes(graph, edge_type, edge_index, relation_scalar)
    )


def build_patient_graph(
    case: LoadedCase,
    source: SourceTumor,
    specs: Sequence[CandidateSpec],
    regions: PatientRegionData,
    *,
    tumor_label: int,
    config: GraphBuildConfig,
    ct_clip: tuple[float, float],
) -> HeteroData:
    """Build one patient-level PyG heterogeneous graph."""

    occupied_without_source = (case.label == int(tumor_label)) & ~source.full_mask
    source_raw = _source_raw(
        case,
        source,
        regions,
        occupied_without_source,
        ct_clip=ct_clip,
    )[None]
    source_axis, source_anisotropy = _principal_axis(source.full_mask, case.spacing)
    candidate_raw = np.stack(
        [
            _candidate_raw(
                case,
                source,
                spec,
                regions,
                source_axis=source_axis,
                source_anisotropy=source_anisotropy,
                ct_clip=ct_clip,
            )
            for spec in specs
        ]
    ).astype(np.float32)
    candidate_position = np.stack(
        [normalized_position(spec.center, case.shape) for spec in specs]
    ).astype(np.float32)
    candidate_regions = np.asarray([int(spec.region_id) for spec in specs], dtype=np.int64)
    lesion_raw, lesion_position, lesion_regions = _lesions(
        case,
        source,
        regions,
        tumor_label=tumor_label,
        max_lesions=config.max_lesions,
        ct_clip=ct_clip,
    )
    liver_raw = _liver_raw(case, regions, tumor_label=tumor_label, ct_clip=ct_clip)[None]

    graph = HeteroData()
    graph["tumor"].raw_x = torch.from_numpy(source_raw.astype(np.float32))
    graph["tumor"].pos = torch.from_numpy(
        normalized_position(source.anchor_center, case.shape)[None].astype(np.float32)
    )
    graph["tumor"].region_index = torch.tensor(
        [regions.region_at(source.anchor_center)], dtype=torch.long
    )
    graph["candidate"].raw_x = torch.from_numpy(candidate_raw)
    graph["candidate"].pos = torch.from_numpy(candidate_position)
    graph["candidate"].region_index = torch.from_numpy(candidate_regions)
    graph["region"].raw_x = torch.from_numpy(regions.region_features.astype(np.float32))
    graph["region"].pos = torch.from_numpy(regions.region_positions.astype(np.float32))
    graph["region"].region_index = torch.arange(regions.num_regions, dtype=torch.long)
    graph["lesion"].raw_x = torch.from_numpy(lesion_raw)
    graph["lesion"].pos = torch.from_numpy(lesion_position)
    graph["lesion"].region_index = torch.from_numpy(lesion_regions)
    graph["liver"].raw_x = torch.from_numpy(liver_raw.astype(np.float32))
    graph["liver"].pos = torch.zeros((1, 3), dtype=torch.float32)
    graph["liver"].region_index = torch.tensor([-1], dtype=torch.long)
    for node_type in PATIENT_NODE_TYPES:
        graph[node_type].meta = torch.from_numpy(
            _node_meta(graph[node_type].raw_x.numpy(), node_type)
        )

    num_candidates = len(specs)
    num_regions = regions.num_regions
    num_lesions = int(lesion_raw.shape[0])
    tumor_to_candidate = _full_bipartite(1, num_candidates)
    candidate_to_region = np.stack(
        [np.arange(num_candidates, dtype=np.int64), candidate_regions], axis=0
    )
    tumor_region = int(graph["tumor"].region_index.item())
    tumor_to_region = np.asarray([[0], [tumor_region]], dtype=np.int64)
    candidate_to_lesion = _full_bipartite(num_candidates, num_lesions)
    lesion_to_region = np.stack(
        [np.arange(num_lesions, dtype=np.int64), lesion_regions], axis=0
    )
    tumor_to_lesion = _full_bipartite(1, num_lesions)
    region_to_liver = _full_bipartite(num_regions, 1)
    relation_indices: dict[tuple[str, str, str], np.ndarray] = {
        ("tumor", "compatible_with", "candidate"): tumor_to_candidate,
        ("candidate", "matched_to", "tumor"): tumor_to_candidate[[1, 0]],
        ("candidate", "spatial_neighbor", "candidate"): _knn(
            candidate_position, config.candidate_k
        ),
        ("candidate", "belongs_to", "region"): candidate_to_region,
        ("region", "contains_candidate", "candidate"): candidate_to_region[[1, 0]],
        ("tumor", "hosted_by", "region"): tumor_to_region,
        ("region", "hosts_tumor", "tumor"): tumor_to_region[[1, 0]],
        ("candidate", "near", "lesion"): candidate_to_lesion,
        ("lesion", "near", "candidate"): candidate_to_lesion[[1, 0]],
        ("lesion", "hosted_by", "region"): lesion_to_region,
        ("region", "hosts_lesion", "lesion"): lesion_to_region[[1, 0]],
        ("tumor", "coexists_with", "lesion"): tumor_to_lesion,
        ("lesion", "coexists_with", "tumor"): tumor_to_lesion[[1, 0]],
        ("region", "adjacent_to", "region"): regions.region_edge_index.astype(np.int64),
        ("region", "inside", "liver"): region_to_liver,
        ("liver", "contains", "region"): region_to_liver[[1, 0]],
    }
    for edge_type in PATIENT_EDGE_TYPES:
        _set_patient_relation(graph, edge_type, relation_indices[edge_type])
    return graph


def _prototype_edge_attributes(
    source_position: np.ndarray,
    destination_position: np.ndarray,
    edge_index: np.ndarray,
    weights: np.ndarray,
    ranks: np.ndarray,
) -> np.ndarray:
    if edge_index.shape[1] == 0:
        return np.empty((0, PROTOTYPE_EDGE_DIM), dtype=np.float32)
    delta = destination_position[edge_index[1]] - source_position[edge_index[0]]
    distance = np.linalg.norm(delta, axis=1, keepdims=True)
    attributes = np.concatenate(
        [delta, distance, weights[:, None], ranks[:, None]],
        axis=1,
    ).astype(np.float32)
    if attributes.shape[1] != PROTOTYPE_EDGE_DIM:
        raise RuntimeError(f"Prototype edge attribute mismatch: {attributes.shape}")
    return attributes


def _set_prototype_relation(
    graph: HeteroData,
    edge_type: tuple[str, str, str],
    edge_index: np.ndarray,
    weights: np.ndarray,
    ranks: np.ndarray,
) -> None:
    source_type, _, destination_type = edge_type
    graph[edge_type].edge_index = torch.from_numpy(edge_index.astype(np.int64))
    graph[edge_type].edge_attr = torch.from_numpy(
        _prototype_edge_attributes(
            graph[source_type].pos.numpy(),
            graph[destination_type].pos.numpy(),
            edge_index,
            weights.astype(np.float32),
            ranks.astype(np.float32),
        )
    )


def build_prototype_graph(
    specs: Sequence[CandidateSpec],
    patient_graph: HeteroData,
    regions: PatientRegionData,
    bank: PrototypeBank,
    *,
    config: GraphBuildConfig,
) -> HeteroData:
    """Build candidate-region-population prototype PyG graph."""

    assignments, assignment_weights = bank.assign(
        regions.region_features,
        top_k=config.prototype_top_m,
        temperature=config.prototype_temperature,
    )
    region_count, top_m = assignments.shape
    region_source = np.repeat(np.arange(region_count, dtype=np.int64), top_m)
    prototype_destination = assignments.reshape(-1).astype(np.int64)
    region_to_prototype = np.stack([region_source, prototype_destination], axis=0)
    flat_weights = assignment_weights.reshape(-1).astype(np.float32)
    ranks = np.tile(np.arange(top_m, dtype=np.float32), region_count)
    ranks /= max(top_m - 1, 1)

    candidate_regions = np.asarray([spec.region_id for spec in specs], dtype=np.int64)
    candidate_to_region = np.stack(
        [np.arange(len(specs), dtype=np.int64), candidate_regions], axis=0
    )

    graph = HeteroData()
    graph["candidate"].raw_x = patient_graph["candidate"].raw_x.clone()
    graph["candidate"].pos = patient_graph["candidate"].pos.clone()
    graph["candidate"].region_index = torch.from_numpy(candidate_regions)
    graph["region"].raw_x = torch.from_numpy(regions.region_features.astype(np.float32))
    graph["region"].pos = torch.from_numpy(regions.region_positions.astype(np.float32))
    graph["prototype"].raw_x = torch.from_numpy(bank.features.astype(np.float32))
    graph["prototype"].pos = torch.from_numpy(bank.features[:, :3].astype(np.float32))
    if graph["region"].raw_x.shape[1] != REGION_FEATURE_DIM:
        raise RuntimeError("Bad region feature dimension in prototype graph")
    if graph["prototype"].raw_x.shape[1] != PROTOTYPE_FEATURE_DIM:
        raise RuntimeError("Bad prototype feature dimension")

    one_candidate = np.ones(candidate_to_region.shape[1], dtype=np.float32)
    zero_candidate = np.zeros(candidate_to_region.shape[1], dtype=np.float32)
    _set_prototype_relation(
        graph,
        ("candidate", "belongs_to", "region"),
        candidate_to_region,
        one_candidate,
        zero_candidate,
    )
    _set_prototype_relation(
        graph,
        ("region", "contains_candidate", "candidate"),
        candidate_to_region[[1, 0]],
        one_candidate,
        zero_candidate,
    )
    _set_prototype_relation(
        graph,
        ("region", "assigned_to", "prototype"),
        region_to_prototype,
        flat_weights,
        ranks,
    )
    _set_prototype_relation(
        graph,
        ("prototype", "represents", "region"),
        region_to_prototype[[1, 0]],
        flat_weights,
        ranks,
    )

    prototype_edges = bank.edge_index.astype(np.int64)
    if prototype_edges.shape[1]:
        delta = bank.features[prototype_edges[1], :3] - bank.features[prototype_edges[0], :3]
        similarity = np.exp(-np.linalg.norm(delta, axis=1)).astype(np.float32)
    else:
        similarity = np.empty((0,), dtype=np.float32)
    _set_prototype_relation(
        graph,
        ("prototype", "similar_to", "prototype"),
        prototype_edges,
        similarity,
        np.zeros_like(similarity),
    )
    for edge_type in PROTOTYPE_EDGE_TYPES:
        if edge_type not in graph.edge_types:
            raise RuntimeError(f"Missing prototype relation: {edge_type}")
    return graph
