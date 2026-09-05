"""Canonical Level-0 tumor/context node construction.

The cache stores complete deterministic node tables and complete physical-radius
edge indices for the source tumor and every erased target context.  It does not
build a fixed-size graph and it never creates k-NN edges.  ``hiercp.sample``
materialises induced HeteroData views inside the training/inference pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi
import torch

try:
    from torch_geometric.data import HeteroData
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
    raise ModuleNotFoundError(
        "PyTorch Geometric is required. Run: python -m tools.install"
    ) from exc

from hiercp.common import (
    LoadedCase,
    SourceTumor,
    ct_normalize,
    erase_mask_with_context,
)
from hiercp.curriculum import CandidateSpec
from hiercp.schema import (
    LOCAL_EDGE_TYPES,
    LOCAL_HANDCRAFTED_DIM,
    EdgeType,
    GraphBuildConfig,
)
from hiercp.spatial import (
    build_patch_payload,
    canonical_coordinate_sets,
    cross_radius_edges,
    exact_source_footprint,
    radius_edges,
    source_node_specifications,
    target_node_specifications,
)


@dataclass
class BuiltLocalGraph:
    """Canonical cache payload for one source/target placement pair."""

    graph: HeteroData | None
    source_patch: np.ndarray
    target_patch: np.ndarray
    source_local: dict[str, Any]
    target_local: dict[str, Any]


@dataclass
class PreparedLocalSource:
    """Candidate-invariant source branch, constructed once per training sample."""

    source_footprint: np.ndarray
    source_patch: np.ndarray
    canonical_nodes: dict[str, dict[str, torch.Tensor]]
    canonical_edges: dict[EdgeType, torch.Tensor]
    canonical_counts: dict[str, int]


@dataclass
class PatchFields:
    model_input: np.ndarray
    ct_norm: np.ndarray
    organ: np.ndarray
    footprint: np.ndarray
    tumor_sdf_mm: np.ndarray
    tumor_sdf_norm: np.ndarray
    liver_depth_mm: np.ndarray
    liver_depth_norm: np.ndarray
    gradient: np.ndarray
    local_mean: np.ndarray
    local_std: np.ndarray
    tumor_normal: np.ndarray
    tumor_curvature: np.ndarray
    liver_normal: np.ndarray
    liver_curvature: np.ndarray
    outside_tumor_mm: np.ndarray


def _require_full_graph(config: GraphBuildConfig) -> None:
    config.validate()
    if (
        config.graph_schema_version != "full_v22"
        or not config.canonical_full_graph
        or not config.adaptive_source_full_shape
    ):
        raise ValueError(
            "local_graph requires graph_schema_version=full_v22, "
            "canonical_full_graph=true and adaptive_source_full_shape=true"
        )


def _transform_footprint(
    footprint: np.ndarray,
    spec: CandidateSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an explicit relation corruption around the footprint centre."""

    forward = spec.rotation_matrix @ np.diag(spec.scale_array)
    if np.allclose(forward, np.eye(3), atol=1e-6):
        return footprint.astype(bool, copy=True), forward.astype(np.float32)
    inverse = np.linalg.inv(forward).astype(np.float64)
    center = (np.asarray(footprint.shape, dtype=np.float64) - 1.0) * 0.5
    offset = center - inverse @ center
    transformed = ndi.affine_transform(
        footprint.astype(np.float32),
        matrix=inverse,
        offset=offset,
        output_shape=footprint.shape,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ) > 0.5
    if not np.any(transformed):
        return footprint.astype(bool, copy=True), np.eye(3, dtype=np.float32)
    return transformed, forward.astype(np.float32)


def _patch_fields(
    case: LoadedCase,
    center: tuple[int, int, int],
    footprint: np.ndarray,
    full_organ_mask: np.ndarray,
    organ_depth: np.ndarray,
    *,
    config: GraphBuildConfig,
    erase_target: bool,
    ct_clip: tuple[float, float],
) -> PatchFields:
    payload = build_patch_payload(
        image=case.image,
        center=center,
        footprint=footprint,
        full_organ=full_organ_mask,
        organ_depth=organ_depth,
        spacing=case.spacing,
        config=config,
        erase_target=erase_target,
        ct_clip=ct_clip,
        ct_normalize_fn=ct_normalize,
        erase_fn=erase_mask_with_context,
    )
    declared = tuple(PatchFields.__dataclass_fields__)
    missing = [name for name in declared if name not in payload]
    if missing:
        raise RuntimeError(
            "PatchFields compatibility failure: "
            f"missing={missing}, available={sorted(payload)}"
        )
    return PatchFields(**{name: payload[name] for name in declared})


def _grid_coordinates(coordinates: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    denominator = np.maximum(np.asarray(shape, dtype=np.float32) - 1.0, 1.0)
    normalized_zyx = coordinates.astype(np.float32) / denominator[None] * 2.0 - 1.0
    return normalized_zyx[:, [2, 1, 0]].astype(np.float32)


def _shell_value(distance_mm: np.ndarray, shells: tuple[float, ...]) -> np.ndarray:
    bins = np.digitize(distance_mm, np.asarray(shells, dtype=np.float32), right=True)
    return bins.astype(np.float32) / max(1.0, float(len(shells)))


def _node_features(
    fields: PatchFields,
    coordinates: np.ndarray,
    *,
    spacing: np.ndarray,
    config: GraphBuildConfig,
    normal_source: str,
    surface_flag: float,
    branch_flag: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = coordinates.astype(np.int64, copy=False)
    index = tuple(coordinates.T)
    center = (np.asarray(fields.ct_norm.shape, dtype=np.float32) - 1.0) * 0.5
    relative_mm = (coordinates.astype(np.float32) - center[None]) * spacing[None]
    position = np.clip(
        relative_mm / max(float(config.context_radius_mm), 1.0),
        -2.0,
        2.0,
    ).astype(np.float32)
    if normal_source == "liver":
        normal = fields.liver_normal[index]
        curvature = fields.liver_curvature[index]
    else:
        normal = fields.tumor_normal[index]
        curvature = fields.tumor_curvature[index]
    features = np.concatenate(
        [
            fields.ct_norm[index][..., None],
            fields.local_mean[index][..., None],
            fields.local_std[index][..., None],
            fields.gradient[index][..., None],
            position,
            fields.tumor_sdf_norm[index][..., None],
            fields.liver_depth_norm[index][..., None],
            normal.astype(np.float32),
            curvature.astype(np.float32)[..., None],
            _shell_value(
                fields.outside_tumor_mm[index], config.context_shells_mm
            )[..., None],
            np.full((coordinates.shape[0], 1), surface_flag, dtype=np.float32),
            np.full((coordinates.shape[0], 1), branch_flag, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    if features.shape[1] != LOCAL_HANDCRAFTED_DIM:
        raise RuntimeError(f"Local node feature mismatch: {features.shape}")
    return features, _grid_coordinates(coordinates, fields.ct_norm.shape), position


def _relative_mm(
    coordinates: np.ndarray,
    shape: tuple[int, int, int],
    spacing: np.ndarray,
) -> np.ndarray:
    center = (np.asarray(shape, dtype=np.float32) - 1.0) * 0.5
    return (
        (coordinates.astype(np.float32) - center[None])
        * np.asarray(spacing, dtype=np.float32)[None]
    ).astype(np.float32)


def _pack_nodes(
    fields: PatchFields,
    specifications: Mapping[str, tuple[np.ndarray, str, float]],
    case: LoadedCase,
    config: GraphBuildConfig,
    *,
    branch_flag: float,
) -> dict[str, dict[str, torch.Tensor]]:
    output: dict[str, dict[str, torch.Tensor]] = {}
    for node_type, (coordinates, normal_source, surface_flag) in specifications.items():
        features, grid, position = _node_features(
            fields,
            coordinates,
            spacing=case.spacing,
            config=config,
            normal_source=str(normal_source),
            surface_flag=float(surface_flag),
            branch_flag=float(branch_flag),
        )
        output[node_type] = {
            "x": torch.from_numpy(features.astype(np.float16)),
            "grid": torch.from_numpy(grid.astype(np.float16)),
            "pos": torch.from_numpy(position.astype(np.float32)),
            "pos_mm": torch.from_numpy(
                _relative_mm(coordinates, fields.ct_norm.shape, case.spacing)
            ),
        }
    return output


def _edge_radius(edge_type: EdgeType, config: GraphBuildConfig) -> float:
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
    if relation == "corresponds_to":
        return float(config.correspondence_radius_mm)
    return float(config.cross_edge_radius_mm)


def _canonical_edges(
    nodes: Mapping[str, Mapping[str, torch.Tensor]],
    edge_types: Sequence[EdgeType],
    config: GraphBuildConfig,
    *,
    transform: np.ndarray | None = None,
) -> dict[EdgeType, torch.Tensor]:
    """Build the complete physical-radius topology for canonical node tables.

    Edge attributes are deterministic functions of node features and are built
    only after an induced training view is selected.  The cached full graph
    therefore stores complete edge indices in compact int32 form.
    """
    positions = {
        node_type: value["pos_mm"].detach().cpu().numpy().astype(np.float32, copy=False)
        for node_type, value in nodes.items()
    }
    result: dict[EdgeType, torch.Tensor] = {}
    for edge_type in edge_types:
        source_type, relation, destination_type = edge_type
        if source_type not in positions or destination_type not in positions:
            continue
        source_position = positions[source_type]
        if transform is not None and (
            (source_type == "tumor_surface" and relation == "interfaces_target")
            or (source_type == "source_context" and relation == "corresponds_to")
        ):
            source_position = source_position @ transform.T
        destination_position = positions[destination_type]
        radius = _edge_radius(edge_type, config)
        if source_type == destination_type:
            edge_index = radius_edges(source_position, radius, include_self=False)
        else:
            edge_index = cross_radius_edges(
                source_position, destination_position, radius
            )
        edge_count = int(edge_index.shape[1])
        limit = int(config.canonical_relation_edge_limit)
        if limit > 0 and edge_count > limit:
            raise RuntimeError(
                "Canonical relation exceeds canonical_relation_edge_limit without "
                f"truncation: edge_type={edge_type}, edges={edge_count}, limit={limit}"
            )
        result[edge_type] = torch.from_numpy(
            edge_index.astype(np.int32, copy=False)
        )
    return result


def prepare_local_source(
    case: LoadedCase,
    source: SourceTumor,
    *,
    full_organ_mask: np.ndarray,
    organ_depth: np.ndarray,
    config: GraphBuildConfig,
    rng: np.random.Generator,
    ct_clip: tuple[float, float],
) -> PreparedLocalSource:
    """Build all canonical source nodes without applying a training budget."""

    del rng  # canonical construction is deterministic
    _require_full_graph(config)
    source_footprint = exact_source_footprint(source)
    fields = _patch_fields(
        case,
        source.anchor_center,
        source_footprint,
        full_organ_mask,
        organ_depth,
        config=config,
        erase_target=False,
        ct_clip=ct_clip,
    )
    if int(fields.footprint.sum()) != int(source_footprint.sum()):
        raise RuntimeError(
            "Source footprint was cropped before canonical construction: "
            f"paste={int(source_footprint.sum())}, graph={int(fields.footprint.sum())}"
        )
    coordinates = canonical_coordinate_sets(fields, config, case.spacing)
    nodes = _pack_nodes(
        fields,
        source_node_specifications(coordinates),
        case,
        config,
        branch_flag=1.0,
    )
    source_types = frozenset(nodes)
    source_edges = _canonical_edges(
        nodes,
        [
            edge_type
            for edge_type in LOCAL_EDGE_TYPES
            if edge_type[0] in source_types and edge_type[2] in source_types
        ],
        config,
    )
    return PreparedLocalSource(
        source_footprint=source_footprint,
        source_patch=fields.model_input.astype(np.float32, copy=False),
        canonical_nodes=nodes,
        canonical_edges=source_edges,
        canonical_counts={key: int(value["x"].shape[0]) for key, value in nodes.items()},
    )


def build_local_graph(
    case: LoadedCase,
    source: SourceTumor,
    spec: CandidateSpec,
    *,
    full_organ_mask: np.ndarray,
    organ_depth: np.ndarray,
    config: GraphBuildConfig,
    rng: np.random.Generator,
    ct_clip: tuple[float, float],
    prepared_source: PreparedLocalSource | None = None,
) -> BuiltLocalGraph:
    """Build complete canonical source/target topology; views are sampled later."""

    _require_full_graph(config)
    prepared = prepared_source or prepare_local_source(
        case,
        source,
        full_organ_mask=full_organ_mask,
        organ_depth=organ_depth,
        config=config,
        rng=rng,
        ct_clip=ct_clip,
    )
    virtual_footprint, transform = _transform_footprint(
        prepared.source_footprint, spec
    )
    target_fields = _patch_fields(
        case,
        spec.center,
        virtual_footprint,
        full_organ_mask,
        organ_depth,
        config=config,
        erase_target=True,
        ct_clip=ct_clip,
    )
    target_coordinates = canonical_coordinate_sets(
        target_fields, config, case.spacing
    )
    target_nodes = _pack_nodes(
        target_fields,
        target_node_specifications(target_coordinates),
        case,
        config,
        branch_flag=-1.0,
    )
    all_nodes: dict[str, Mapping[str, torch.Tensor]] = {
        **prepared.canonical_nodes,
        **target_nodes,
    }
    target_edges = _canonical_edges(
        all_nodes,
        [edge_type for edge_type in LOCAL_EDGE_TYPES if edge_type not in prepared.canonical_edges],
        config,
        transform=transform,
    )
    source_local = {
        "format": "canonical-full-v22",
        "nodes": prepared.canonical_nodes,
        "edges": prepared.canonical_edges,
        "footprint_voxels": int(prepared.source_footprint.sum()),
        "counts": prepared.canonical_counts,
        "edge_counts": {
            edge_type: int(edge_index.shape[1])
            for edge_type, edge_index in prepared.canonical_edges.items()
        },
    }
    target_local = {
        "format": "canonical-full-v22",
        "nodes": target_nodes,
        "edges": target_edges,
        "transform": torch.from_numpy(transform.astype(np.float32)),
        "counts": {key: int(value["x"].shape[0]) for key, value in target_nodes.items()},
        "edge_counts": {
            edge_type: int(edge_index.shape[1])
            for edge_type, edge_index in target_edges.items()
        },
    }
    return BuiltLocalGraph(
        graph=None,
        source_patch=prepared.source_patch,
        target_patch=target_fields.model_input.astype(np.float32, copy=False),
        source_local=source_local,
        target_local=target_local,
    )
