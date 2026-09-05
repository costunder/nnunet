"""Shared schemas for the three-level PyTorch Geometric experiment.

The graph hierarchy is explicit and uses separate PyG ``HeteroData`` objects:

* Level 0: local tumor/source-context/target-context graph;
* Level 1: patient tumor/candidate/liver-region graph;
* Level 2: candidate/region/population-prototype graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

NodeType = str
EdgeType = tuple[str, str, str]

PATCH_CHANNELS: Final[int] = 5
LOCAL_HANDCRAFTED_DIM: Final[int] = 16
LOCAL_EDGE_DIM: Final[int] = 10
CONTEXT_SHELL_COUNT: Final[int] = 3
CONTEXT_SHELL_FEATURE_INDEX: Final[int] = 13
UPPER_RAW_DIM: Final[int] = 14
REGION_FEATURE_DIM: Final[int] = 16
PROTOTYPE_FEATURE_DIM: Final[int] = REGION_FEATURE_DIM + 2
PATIENT_EDGE_DIM: Final[int] = 12
PROTOTYPE_EDGE_DIM: Final[int] = 6

# Runtime feature policy: source/candidate absolute coordinates and occupied
# clearance are forbidden ranking inputs.  They remain in the V22 cache for
# backward-compatible deserialization, but the model masks them before every
# projection.  Patient relations incident to the source tumor likewise mask
# their position delta and distance columns.
UPPER_FEATURE_POLICY: Final[str] = "shortcut_safe_upper_v1"
UPPER_POSITION_COLUMNS: Final[tuple[int, ...]] = (0, 1, 2)
UPPER_OCCUPIED_DISTANCE_INDEX: Final[int] = 4
UPPER_FORBIDDEN_RAW_COLUMNS: Final[tuple[int, ...]] = (0, 1, 2, 4)
PATIENT_POSITION_EDGE_COLUMNS: Final[tuple[int, ...]] = (0, 1, 2, 3)

LOCAL_NODE_TYPES: Final[tuple[NodeType, ...]] = (
    "tumor_surface",
    "tumor_interior",
    "source_context",
    "source_liver_surface",
    "target_context",
    "target_liver_surface",
)

SOURCE_LOCAL_NODE_TYPES: Final[frozenset[NodeType]] = frozenset(
    {"tumor_surface", "tumor_interior", "source_context", "source_liver_surface"}
)
TARGET_LOCAL_NODE_TYPES: Final[frozenset[NodeType]] = frozenset(
    {"target_context", "target_liver_surface"}
)

LOCAL_EDGE_TYPES: Final[tuple[EdgeType, ...]] = (
    ("tumor_surface", "surface_neighbor", "tumor_surface"),
    ("tumor_interior", "interior_neighbor", "tumor_interior"),
    ("source_context", "context_neighbor", "source_context"),
    ("source_liver_surface", "surface_neighbor", "source_liver_surface"),
    ("target_context", "context_neighbor", "target_context"),
    ("target_liver_surface", "surface_neighbor", "target_liver_surface"),
    ("tumor_interior", "supports", "tumor_surface"),
    ("tumor_surface", "supported_by", "tumor_interior"),
    ("tumor_surface", "interfaces_source", "source_context"),
    ("source_context", "interfaces_tumor", "tumor_surface"),
    ("tumor_surface", "interfaces_target", "target_context"),
    ("source_context", "corresponds_to", "target_context"),
    ("source_context", "near_liver_surface", "source_liver_surface"),
    ("source_liver_surface", "anchors_context", "source_context"),
    ("target_context", "near_liver_surface", "target_liver_surface"),
    ("target_liver_surface", "anchors_context", "target_context"),
)

PATIENT_NODE_TYPES: Final[tuple[NodeType, ...]] = (
    "tumor",
    "candidate",
    "region",
    "lesion",
    "liver",
)
PATIENT_EDGE_TYPES: Final[tuple[EdgeType, ...]] = (
    ("tumor", "compatible_with", "candidate"),
    ("candidate", "matched_to", "tumor"),
    ("candidate", "spatial_neighbor", "candidate"),
    ("candidate", "belongs_to", "region"),
    ("region", "contains_candidate", "candidate"),
    ("tumor", "hosted_by", "region"),
    ("region", "hosts_tumor", "tumor"),
    ("candidate", "near", "lesion"),
    ("lesion", "near", "candidate"),
    ("lesion", "hosted_by", "region"),
    ("region", "hosts_lesion", "lesion"),
    ("tumor", "coexists_with", "lesion"),
    ("lesion", "coexists_with", "tumor"),
    ("region", "adjacent_to", "region"),
    ("region", "inside", "liver"),
    ("liver", "contains", "region"),
)

PROTOTYPE_NODE_TYPES: Final[tuple[NodeType, ...]] = (
    "candidate",
    "region",
    "prototype",
)
PROTOTYPE_EDGE_TYPES: Final[tuple[EdgeType, ...]] = (
    ("candidate", "belongs_to", "region"),
    ("region", "contains_candidate", "candidate"),
    ("region", "assigned_to", "prototype"),
    ("prototype", "represents", "region"),
    ("prototype", "similar_to", "prototype"),
)

DIFFICULTY_POSITIVE: Final[int] = 0
DIFFICULTY_EASY: Final[int] = 1
DIFFICULTY_INTER_REGION: Final[int] = 2
DIFFICULTY_INTRA_CORRUPTED: Final[int] = 3

CORRUPTION_NONE: Final[int] = 0
CORRUPTION_THICKNESS_SCALE: Final[int] = 1
CORRUPTION_ANISOTROPIC_SCALE: Final[int] = 2
CORRUPTION_ORIENTATION: Final[int] = 3
NUM_CORRUPTIONS: Final[int] = 4


@dataclass(frozen=True)
class GraphBuildConfig:
    """Active canonical geometry, sampled-view and hierarchy settings."""

    patch_size: int = 48
    context_radius_mm: float = 28.0
    context_shells_mm: tuple[float, ...] = (4.0, 12.0, 28.0)
    boundary_depth_mm: float = 3.0

    candidate_k: int = 8
    num_regions: int = 24
    region_sample_voxels: int = 20_000
    region_lloyd_iters: int = 12
    region_k: int = 4
    num_prototypes: int = 16
    prototype_top_m: int = 2
    prototype_k: int = 4
    prototype_lloyd_iters: int = 30
    prototype_temperature: float = 0.5
    # ``None`` (or a non-positive legacy value) means unlimited.  A positive
    # value is a fail-fast resource guard; hierarchy construction never slices
    # lesions to satisfy it.
    max_lesions: int | None = None

    graph_schema_version: str = "full_v22"
    adaptive_source_full_shape: bool = True
    adaptive_roi_margin_mm: float = 30.0
    adaptive_roi_max_radius_mm: float = 64.0
    adaptive_roi_max_voxels: int = 8_000_000
    context_inner_radius_mm: float = 2.0
    context_outer_radius_mm: float = 28.0
    context_liver_surface_separation_mm: float = 1.0
    context_radial_bins: int = 4
    context_azimuth_bins: int = 8
    context_elevation_bins: int = 4
    liver_anchor_search_mm: float = 64.0
    kdtree_leafsize: int = 32

    canonical_full_graph: bool = True
    canonical_surface_spacing_mm: float = 2.0
    canonical_interior_spacing_mm: float = 3.0
    canonical_context_spacing_mm: float = 2.5
    canonical_liver_spacing_mm: float = 4.0
    canonical_node_limit: int = 250_000

    sample_context_nodes: int = 384
    sample_hops: int = 2
    sample_interface_radius_mm: float = 8.0
    sample_hop_radius_mm: float = 6.0

    surface_edge_radius_mm: float = 4.5
    interior_edge_radius_mm: float = 5.5
    context_edge_radius_mm: float = 6.0
    interface_edge_radius_mm: float = 8.0
    cross_edge_radius_mm: float = 7.0
    liver_edge_radius_mm: float = 10.0
    correspondence_radius_mm: float = 5.0
    canonical_relation_edge_limit: int = 5_000_000
    sample_relation_edge_limit: int = 500_000

    def validate(self) -> None:
        if self.graph_schema_version != "full_v22":
            raise ValueError(
                "Only graph_schema_version=full_v22 is supported by the active project"
            )
        if self.patch_size < 16 or self.patch_size % 4 != 0:
            raise ValueError("patch_size must be >=16 and divisible by four")
        positive = {
            "context_radius_mm": self.context_radius_mm,
            "boundary_depth_mm": self.boundary_depth_mm,
            "candidate_k": self.candidate_k,
            "num_regions": self.num_regions,
            "region_sample_voxels": self.region_sample_voxels,
            "region_lloyd_iters": self.region_lloyd_iters,
            "region_k": self.region_k,
            "num_prototypes": self.num_prototypes,
            "prototype_top_m": self.prototype_top_m,
            "prototype_k": self.prototype_k,
            "prototype_lloyd_iters": self.prototype_lloyd_iters,
            "prototype_temperature": self.prototype_temperature,
            "adaptive_roi_margin_mm": self.adaptive_roi_margin_mm,
            "adaptive_roi_max_radius_mm": self.adaptive_roi_max_radius_mm,
            "adaptive_roi_max_voxels": self.adaptive_roi_max_voxels,
            "context_inner_radius_mm": self.context_inner_radius_mm,
            "context_outer_radius_mm": self.context_outer_radius_mm,
            "context_radial_bins": self.context_radial_bins,
            "context_azimuth_bins": self.context_azimuth_bins,
            "context_elevation_bins": self.context_elevation_bins,
            "liver_anchor_search_mm": self.liver_anchor_search_mm,
            "kdtree_leafsize": self.kdtree_leafsize,
            "canonical_surface_spacing_mm": self.canonical_surface_spacing_mm,
            "canonical_interior_spacing_mm": self.canonical_interior_spacing_mm,
            "canonical_context_spacing_mm": self.canonical_context_spacing_mm,
            "canonical_liver_spacing_mm": self.canonical_liver_spacing_mm,
            "canonical_node_limit": self.canonical_node_limit,
            "sample_context_nodes": self.sample_context_nodes,
            "sample_interface_radius_mm": self.sample_interface_radius_mm,
            "sample_hop_radius_mm": self.sample_hop_radius_mm,
            "surface_edge_radius_mm": self.surface_edge_radius_mm,
            "interior_edge_radius_mm": self.interior_edge_radius_mm,
            "context_edge_radius_mm": self.context_edge_radius_mm,
            "interface_edge_radius_mm": self.interface_edge_radius_mm,
            "cross_edge_radius_mm": self.cross_edge_radius_mm,
            "liver_edge_radius_mm": self.liver_edge_radius_mm,
            "correspondence_radius_mm": self.correspondence_radius_mm,
            "canonical_relation_edge_limit": self.canonical_relation_edge_limit,
            "sample_relation_edge_limit": self.sample_relation_edge_limit,
        }
        for name, value in positive.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        exact_integers = {
            "patch_size": self.patch_size,
            "candidate_k": self.candidate_k,
            "num_regions": self.num_regions,
            "region_sample_voxels": self.region_sample_voxels,
            "region_lloyd_iters": self.region_lloyd_iters,
            "region_k": self.region_k,
            "num_prototypes": self.num_prototypes,
            "prototype_top_m": self.prototype_top_m,
            "prototype_k": self.prototype_k,
            "prototype_lloyd_iters": self.prototype_lloyd_iters,
            "adaptive_roi_max_voxels": self.adaptive_roi_max_voxels,
            "context_radial_bins": self.context_radial_bins,
            "context_azimuth_bins": self.context_azimuth_bins,
            "context_elevation_bins": self.context_elevation_bins,
            "kdtree_leafsize": self.kdtree_leafsize,
            "canonical_node_limit": self.canonical_node_limit,
            "sample_context_nodes": self.sample_context_nodes,
            "canonical_relation_edge_limit": self.canonical_relation_edge_limit,
            "sample_relation_edge_limit": self.sample_relation_edge_limit,
        }
        for name, value in exact_integers.items():
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must be an exact integer, got {value!r}")
        if self.max_lesions is not None:
            try:
                lesion_limit = int(self.max_lesions)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "max_lesions must be null or an integer fail-fast guard"
                ) from exc
            if float(self.max_lesions) != float(lesion_limit):
                raise ValueError("max_lesions must be an integer when supplied")
        if isinstance(self.sample_hops, bool) or int(self.sample_hops) != self.sample_hops:
            raise ValueError(
                f"sample_hops must be an exact integer, got {self.sample_hops!r}"
            )
        if int(self.sample_hops) < 0:
            raise ValueError("sample_hops must be non-negative")
        if self.num_regions > self.region_sample_voxels:
            raise ValueError(
                "num_regions cannot exceed region_sample_voxels; configured region "
                "clusters must not be silently reduced"
            )
        if self.region_k >= self.num_regions:
            raise ValueError("region_k must be smaller than num_regions")
        if self.prototype_top_m > self.num_prototypes:
            raise ValueError("prototype_top_m cannot exceed num_prototypes")
        if self.prototype_k >= self.num_prototypes:
            raise ValueError("prototype_k must be smaller than num_prototypes")
        shells = tuple(float(value) for value in self.context_shells_mm)
        if len(shells) != CONTEXT_SHELL_COUNT:
            raise ValueError(
                f"context_shells_mm must contain exactly {CONTEXT_SHELL_COUNT} radii"
            )
        if any(value <= 0 for value in shells) or tuple(sorted(shells)) != shells:
            raise ValueError("context_shells_mm must be strictly increasing and positive")
        if shells[-1] > float(self.context_radius_mm) + 1e-6:
            raise ValueError("largest context shell exceeds context_radius_mm")
        if not self.canonical_full_graph or not self.adaptive_source_full_shape:
            raise ValueError(
                "full_v22 requires canonical_full_graph=true and "
                "adaptive_source_full_shape=true"
            )
        if self.context_inner_radius_mm >= self.context_outer_radius_mm:
            raise ValueError(
                "context_inner_radius_mm must be smaller than context_outer_radius_mm"
            )
        if self.context_outer_radius_mm > self.adaptive_roi_max_radius_mm:
            raise ValueError("context_outer_radius_mm exceeds adaptive_roi_max_radius_mm")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["context_shells_mm"] = list(self.context_shells_mm)
        return payload


_ALIASES: Final[dict[str, str]] = {
    "local_patch_size": "patch_size",
    "liver_surface_depth_mm": "boundary_depth_mm",
    "region_kmeans_iters": "region_lloyd_iters",
    "prototype_assignment_k": "prototype_top_m",
}


def graph_config_from_dict(payload: dict[str, object]) -> GraphBuildConfig:
    """Parse the active full-graph configuration and a few stable aliases."""

    normalized = dict(payload)
    for old, new in _ALIASES.items():
        if old in normalized and new not in normalized:
            normalized[new] = normalized[old]
    if "context_shells_mm" in normalized:
        normalized["context_shells_mm"] = tuple(
            float(value) for value in normalized["context_shells_mm"]
        )
    allowed = GraphBuildConfig.__dataclass_fields__.keys()
    config = GraphBuildConfig(
        **{key: normalized[key] for key in allowed if key in normalized}
    )
    config.validate()
    return config
