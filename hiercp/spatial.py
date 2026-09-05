"""Canonical full-node geometry for sampled spatial HierCP.

The existing dense encoder still receives a fixed ``patch_size³`` tensor, but
all handcrafted geometry and semantic nodes are computed in an adaptive native
ROI that contains the complete paste mask. Coordinates follow the project's
Z-Y-X voxel convention; physical coordinates are ``coordinate * spacing``.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


class CanonicalGraphUnavailable(ValueError):
    """A sample cannot form the required non-cropped canonical local graph."""


class AdaptiveRoiBudgetError(CanonicalGraphUnavailable):
    """The requested full-context ROI exceeds a configured resource ceiling."""


class EmptyCanonicalNodeError(CanonicalGraphUnavailable):
    """A required semantic node set is empty for this placement."""

    def __init__(self, node_type: str, shape: tuple[int, ...]) -> None:
        self.node_type = str(node_type)
        self.coordinate_shape = tuple(int(v) for v in shape)
        super().__init__(
            f"V22 {self.node_type} coordinates are invalid: {self.coordinate_shape}"
        )


def _cfg(config: object, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _field(fields: object, *names: str) -> Any:
    """Read a PatchFields attribute across the legacy and latest schemas."""
    for name in names:
        if hasattr(fields, name):
            return getattr(fields, name)
    available = sorted(vars(fields)) if hasattr(fields, "__dict__") else []
    raise AttributeError(
        f"PatchFields does not expose any of {names}; available={available}"
    )



def exact_source_footprint(source: object) -> np.ndarray:
    """Return the exact mask transformed by Copy-Paste, with a zero-loss check."""
    if not hasattr(source, "patch_mask"):
        raise AttributeError("SourceTumor.patch_mask is required")
    footprint = np.asarray(getattr(source, "patch_mask"), dtype=bool)
    if footprint.ndim != 3 or not np.any(footprint):
        raise ValueError(f"Invalid SourceTumor.patch_mask: {footprint.shape}")
    actual = int(np.count_nonzero(footprint))
    expected: int | None = None
    if hasattr(source, "voxel_count"):
        expected = int(getattr(source, "voxel_count"))
    elif hasattr(source, "full_mask"):
        expected = int(np.count_nonzero(np.asarray(getattr(source, "full_mask"))))
    if expected is not None and actual != expected:
        raise ValueError(
            "Exact paste-mask invariant failed: "
            f"patch_mask={actual}, expected={expected}, voxel_loss={expected - actual}"
        )
    return footprint.copy()


def center_crop_or_pad(
    array: np.ndarray,
    shape: Sequence[int],
    pad_value: float | int | bool = 0,
) -> np.ndarray:
    target = tuple(int(value) for value in shape)
    if len(target) != array.ndim:
        raise ValueError(f"Rank mismatch: {array.shape} -> {target}")
    output = np.full(target, pad_value, dtype=array.dtype)
    source_slices: list[slice] = []
    destination_slices: list[slice] = []
    for source_size, target_size in zip(array.shape, target):
        length = min(int(source_size), int(target_size))
        source_start = max(0, (int(source_size) - length) // 2)
        destination_start = max(0, (int(target_size) - length) // 2)
        source_slices.append(slice(source_start, source_start + length))
        destination_slices.append(slice(destination_start, destination_start + length))
    output[tuple(destination_slices)] = array[tuple(source_slices)]
    return output


def extract_centered(
    array: np.ndarray,
    center: Sequence[int],
    shape: Sequence[int],
    *,
    pad_value: float | int | bool,
) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected 3-D array, got {array.shape}")
    target_shape = np.asarray(shape, dtype=np.int64)
    center_array = np.asarray(center, dtype=np.int64)
    if target_shape.shape != (3,) or center_array.shape != (3,):
        raise ValueError("center and shape must contain exactly three values")
    start = center_array - target_shape // 2
    stop = start + target_shape
    output = np.full(tuple(int(value) for value in target_shape), pad_value, dtype=array.dtype)
    source_start = np.maximum(start, 0)
    source_stop = np.minimum(stop, np.asarray(array.shape, dtype=np.int64))
    if np.any(source_stop <= source_start):
        return output
    destination_start = source_start - start
    destination_stop = destination_start + (source_stop - source_start)
    source_slices = tuple(slice(int(a), int(b)) for a, b in zip(source_start, source_stop))
    destination_slices = tuple(
        slice(int(a), int(b)) for a, b in zip(destination_start, destination_stop)
    )
    output[destination_slices] = array[source_slices]
    return output


def _odd(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.int64, copy=True)
    result += (result + 1) % 2
    return result


def adaptive_native_shape(
    footprint_shape: Sequence[int],
    spacing: Sequence[float],
    config: object,
    *,
    center_liver_depth_mm: float,
) -> tuple[int, int, int]:
    """Compute a native ROI without silently reducing its requested context."""
    footprint = np.asarray(footprint_shape, dtype=np.int64)
    spacing_array = np.maximum(np.asarray(spacing, dtype=np.float64), 1e-6)
    if footprint.shape != (3,) or np.any(footprint < 1):
        raise ValueError(f"Invalid footprint shape: {tuple(footprint)}")
    context_outer = float(
        _cfg(config, "context_outer_radius_mm", _cfg(config, "context_radius_mm", 24.0))
    )
    base_margin = max(float(_cfg(config, "adaptive_roi_margin_mm", 30.0)), context_outer + 2.0)
    anchor_search = float(_cfg(config, "liver_anchor_search_mm", 64.0))
    radius_cap = float(_cfg(config, "adaptive_roi_max_radius_mm", 64.0))
    requested_margin = max(base_margin, min(float(center_liver_depth_mm) + 4.0, anchor_search))
    if radius_cap <= 0.0:
        raise ValueError("adaptive_roi_max_radius_mm must be positive")
    if requested_margin > radius_cap:
        raise AdaptiveRoiBudgetError(
            "Adaptive ROI context exceeds the configured physical-radius ceiling; "
            "no context reduction was applied. "
            f"footprint_shape={tuple(int(value) for value in footprint)}, "
            f"spacing_mm={tuple(float(value) for value in spacing_array)}, "
            f"requested_margin_mm={requested_margin:.6g}, "
            f"configured_radius_limit_mm={radius_cap:.6g}. "
            "Increase graph.adaptive_roi_max_radius_mm after measuring RAM and graph "
            "cost, or explicitly revise the requested context with user approval."
        )
    pad = np.ceil(requested_margin / spacing_array).astype(np.int64)
    native_shape = _odd(footprint + 2 * pad)

    # This is a fail-fast resource guard, never a context-reduction mechanism.
    # The requested shape is returned unchanged when it fits; otherwise callers
    # receive actionable diagnostics before allocating or resampling the ROI.
    max_voxels = int(_cfg(config, "adaptive_roi_max_voxels", 8_000_000))
    if max_voxels <= 0:
        raise ValueError("adaptive_roi_max_voxels must be positive")
    native_voxels = int(np.prod(native_shape, dtype=np.int64))
    if native_voxels > max_voxels:
        requested_shape = tuple(int(value) for value in native_shape)
        raise AdaptiveRoiBudgetError(
            "Adaptive ROI request exceeds graph.adaptive_roi_max_voxels; no context "
            "reduction was applied. "
            f"footprint_shape={tuple(int(value) for value in footprint)}, "
            f"spacing_mm={tuple(float(value) for value in spacing_array)}, "
            f"requested_margin_mm={requested_margin:.6g}, "
            f"requested_shape={requested_shape}, requested_voxels={native_voxels}, "
            f"effective_shape={requested_shape}, effective_voxels={native_voxels}, "
            f"voxel_budget={max_voxels}. "
            "Increase graph.adaptive_roi_max_voxels after measuring RAM and processing "
            "time, or explicitly revise the requested context with user approval."
        )
    return tuple(int(value) for value in native_shape)


def _resample_exact(array: np.ndarray, shape: Sequence[int], order: int) -> np.ndarray:
    target = np.asarray(shape, dtype=np.int64)
    source = np.asarray(array.shape, dtype=np.int64)
    if np.array_equal(target, source):
        return array.copy()
    zoom = target.astype(np.float64) / np.maximum(source.astype(np.float64), 1.0)
    result = ndi.zoom(
        array,
        zoom=tuple(float(value) for value in zoom),
        order=int(order),
        mode="nearest",
        prefilter=bool(order > 1),
    )
    if tuple(result.shape) != tuple(int(value) for value in target):
        result = center_crop_or_pad(result, target, pad_value=0)
    return result


def signed_distance(mask: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    sampling = tuple(float(value) for value in spacing)
    outside = ndi.distance_transform_edt(~binary, sampling=sampling).astype(np.float32)
    inside = ndi.distance_transform_edt(binary, sampling=sampling).astype(np.float32)
    outside[binary] = -inside[binary]
    return outside


def normal_and_curvature(
    signed_distance_mm: np.ndarray,
    spacing: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    spacing_tuple = tuple(float(value) for value in spacing)
    gradients = np.gradient(
        signed_distance_mm.astype(np.float32),
        *spacing_tuple,
        edge_order=1,
    )
    vector = np.stack(gradients, axis=-1).astype(np.float32)
    magnitude = np.linalg.norm(vector, axis=-1, keepdims=True)
    normal = vector / np.maximum(magnitude, 1e-6)
    curvature = np.zeros_like(signed_distance_mm, dtype=np.float32)
    for axis in range(3):
        curvature += np.gradient(
            normal[..., axis],
            spacing_tuple[axis],
            axis=axis,
            edge_order=1,
        )
    return normal.astype(np.float32), np.clip(curvature / 3.0, -2.0, 2.0).astype(np.float32)


def _erase_fallback(raw_ct: np.ndarray, footprint: np.ndarray, organ: np.ndarray) -> np.ndarray:
    output = raw_ct.astype(np.float32, copy=True)
    ring = ndi.binary_dilation(footprint, iterations=3) & organ & ~footprint
    values = raw_ct[ring]
    if values.size == 0:
        values = raw_ct[organ & ~footprint]
    fill = float(np.median(values)) if values.size else float(np.median(raw_ct))
    output[footprint] = fill
    return output


def build_patch_payload(
    *,
    image: np.ndarray,
    center: Sequence[int],
    footprint: np.ndarray,
    full_organ: np.ndarray,
    organ_depth: np.ndarray,
    spacing: Sequence[float],
    config: object,
    erase_target: bool,
    ct_clip: tuple[float, float],
    ct_normalize_fn: Callable[[np.ndarray, tuple[float, float]], np.ndarray] | None = None,
    erase_fn: Callable[..., np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Build native handcrafted fields and a fixed dense-CNN view."""
    footprint = np.asarray(footprint, dtype=bool)
    if footprint.ndim != 3 or not np.any(footprint):
        raise ValueError(f"footprint must be a non-empty 3-D mask: {footprint.shape}")
    center_tuple = tuple(int(value) for value in center)
    center_in_bounds = all(
        0 <= center_tuple[axis] < organ_depth.shape[axis] for axis in range(3)
    )
    center_depth = float(organ_depth[center_tuple]) if center_in_bounds else 0.0
    native_shape = adaptive_native_shape(
        footprint.shape,
        spacing,
        config,
        center_liver_depth_mm=center_depth,
    )
    raw_ct = extract_centered(
        np.asarray(image),
        center,
        native_shape,
        pad_value=float(ct_clip[0]),
    ).astype(np.float32)
    organ = extract_centered(
        np.asarray(full_organ, dtype=np.uint8),
        center,
        native_shape,
        pad_value=0,
    ).astype(bool)
    depth = extract_centered(
        np.asarray(organ_depth, dtype=np.float32),
        center,
        native_shape,
        pad_value=0.0,
    ).astype(np.float32)
    native_footprint = center_crop_or_pad(
        footprint.astype(np.uint8),
        native_shape,
        pad_value=0,
    ).astype(bool)
    before = int(np.count_nonzero(footprint))
    after = int(np.count_nonzero(native_footprint))
    if before != after:
        raise RuntimeError(f"adaptive ROI cropped tumor: before={before}, after={after}")

    if erase_target:
        if erase_fn is None:
            visible_ct = _erase_fallback(raw_ct, native_footprint, organ)
        else:
            try:
                visible_ct = erase_fn(raw_ct, native_footprint, organ, ring_width=3)
            except TypeError:
                visible_ct = erase_fn(raw_ct, native_footprint, organ)
        visible_ct = np.asarray(visible_ct, dtype=np.float32)
    else:
        visible_ct = raw_ct

    if ct_normalize_fn is None:
        low, high = float(ct_clip[0]), float(ct_clip[1])
        clipped = np.clip(visible_ct, low, high)
        ct_norm = ((clipped - low) / max(high - low, 1e-6) * 2.0 - 1.0).astype(np.float32)
    else:
        ct_norm = np.asarray(ct_normalize_fn(visible_ct, ct_clip), dtype=np.float32)

    spacing_array = np.asarray(spacing, dtype=np.float32)
    context_radius = float(
        _cfg(config, "context_outer_radius_mm", _cfg(config, "context_radius_mm", 24.0))
    )
    tumor_sdf_mm = signed_distance(native_footprint, spacing_array)
    tumor_sdf_norm = np.clip(
        tumor_sdf_mm / max(context_radius, 1.0),
        -2.0,
        2.0,
    ).astype(np.float32)
    liver_depth_norm = np.clip(depth / 80.0, 0.0, 2.0).astype(np.float32)
    gradient_parts = np.gradient(ct_norm, edge_order=1)
    gradient = np.sqrt(sum(part.astype(np.float32) ** 2 for part in gradient_parts))
    gradient = np.clip(gradient, 0.0, 2.0).astype(np.float32)
    local_mean = ndi.uniform_filter(ct_norm, size=3, mode="nearest").astype(np.float32)
    local_square = ndi.uniform_filter(ct_norm * ct_norm, size=3, mode="nearest")
    local_std = np.sqrt(
        np.maximum(local_square - local_mean * local_mean, 0.0)
    ).astype(np.float32)
    tumor_normal, tumor_curvature = normal_and_curvature(tumor_sdf_mm, spacing_array)
    liver_sdf_mm = signed_distance(organ, spacing_array)
    liver_normal, liver_curvature = normal_and_curvature(liver_sdf_mm, spacing_array)
    outside_tumor_mm = ndi.distance_transform_edt(
        ~native_footprint,
        sampling=tuple(float(value) for value in spacing_array),
    ).astype(np.float32)

    fixed_shape = (int(_cfg(config, "patch_size", 48)),) * 3
    model_input = np.stack(
        [
            _resample_exact(ct_norm, fixed_shape, order=1).astype(np.float32),
            (
                _resample_exact(native_footprint.astype(np.float32), fixed_shape, order=0)
                > 0.5
            ).astype(np.float32),
            (
                _resample_exact(organ.astype(np.float32), fixed_shape, order=0) > 0.5
            ).astype(np.float32),
            _resample_exact(tumor_sdf_norm, fixed_shape, order=1).astype(np.float32),
            _resample_exact(liver_depth_norm, fixed_shape, order=1).astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)

    inside_tumor_mm = np.maximum(-tumor_sdf_mm, 0.0).astype(np.float32)
    tumor_outer = np.maximum(tumor_sdf_mm, 0.0).astype(np.float32)
    depth_float = depth.astype(np.float32, copy=False)
    # Expose both known PatchFields schemas. The local integration layer selects
    # only fields declared by the mounted project's PatchFields dataclass.
    return {
        "model_input": model_input,
        # Latest schema (2026-08 source snapshot).
        "intensity": ct_norm,
        "footprint": native_footprint,
        "organ_mask": organ,
        "organ_depth": depth_float,
        "tumor_signed": tumor_sdf_mm,
        "tumor_outer": tumor_outer,
        "outside_tumor_mm": outside_tumor_mm,
        "inside_tumor_mm": inside_tumor_mm,
        "organ_boundary_mm": depth_float,
        "tumor_normals": tumor_normal,
        "organ_normals": liver_normal,
        "curvature": tumor_curvature,
        # Legacy V16/V17 schema.
        "ct_norm": ct_norm,
        "organ": organ,
        "tumor_sdf_mm": tumor_sdf_mm,
        "tumor_sdf_norm": tumor_sdf_norm,
        "liver_depth_mm": depth_float,
        "liver_depth_norm": liver_depth_norm,
        "gradient": gradient,
        "local_mean": local_mean,
        "local_std": local_std,
        "tumor_normal": tumor_normal,
        "tumor_curvature": tumor_curvature,
        "liver_normal": liver_normal,
        "liver_curvature": liver_curvature,
    }



def surface_mask(mask: np.ndarray) -> np.ndarray:
    """Return the complete six-connected boundary of a binary object."""
    binary = np.asarray(mask, dtype=bool)
    eroded = ndi.binary_erosion(
        binary,
        structure=ndi.generate_binary_structure(3, 1),
        border_value=0,
    )
    return binary & ~eroded


def voxel_grid_coordinates(
    coordinates: np.ndarray,
    spacing: Sequence[float],
    cell_mm: float,
) -> np.ndarray:
    """Deterministically retain one real voxel for every occupied physical cell.

    This is graph discretisation, not stochastic training sampling: every
    occupied cell is represented in the canonical node table.  Training views
    are sampled later from this complete table.
    """
    coords = np.asarray(coordinates, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coordinates must be [N,3], got {coords.shape}")
    if coords.shape[0] <= 1:
        return coords.copy()
    cell = float(cell_mm)
    if cell <= 0.0:
        return _unique_rows(coords)
    physical = coords.astype(np.float64) * np.asarray(spacing, dtype=np.float64)[None]
    cell_id = np.floor(physical / cell).astype(np.int64)
    # np.argwhere is lexicographic; keeping the first voxel in each sorted cell
    # is deterministic across processes and does not depend on an RNG stream.
    _, first = np.unique(cell_id, axis=0, return_index=True)
    return coords[np.sort(first)].astype(np.int64, copy=False)


def _unique_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.shape[0] <= 1:
        return array.copy()
    _, first = np.unique(array, axis=0, return_index=True)
    return array[np.sort(first)]


def is_full_graph(config: object) -> bool:
    return (
        str(_cfg(config, "graph_schema_version", "")) == "full_v22"
        and bool(_cfg(config, "canonical_full_graph", False))
        and bool(_cfg(config, "adaptive_source_full_shape", False))
    )


def canonical_coordinate_sets(
    fields: object,
    config: object,
    spacing: Sequence[float],
) -> dict[str, np.ndarray]:
    """Build the complete deterministic V22 canonical node tables.

    No training-view budget is applied here.  Every occupied physical grid cell
    that satisfies a semantic mask enters the canonical table.  Stochastic
    graph sampling happens only after this stage in :mod:`hiercp.sample`.
    """
    footprint = np.asarray(_field(fields, "footprint"), dtype=bool)
    organ = np.asarray(_field(fields, "organ_mask", "organ"), dtype=bool)
    outside = np.asarray(
        _field(fields, "outside_tumor_mm", "tumor_outer"), dtype=np.float32
    )
    liver_depth = np.asarray(
        _field(fields, "organ_depth", "organ_boundary_mm", "liver_depth_mm"),
        dtype=np.float32,
    )
    if not (footprint.shape == organ.shape == outside.shape == liver_depth.shape):
        raise ValueError("V22 PatchFields arrays have inconsistent shapes")
    spacing_array = np.asarray(spacing, dtype=np.float32)

    surface_all = np.argwhere(surface_mask(footprint))
    interior_mask = ndi.binary_erosion(
        footprint,
        structure=ndi.generate_binary_structure(3, 1),
        border_value=0,
    )
    if not np.any(interior_mask):
        interior_mask = footprint
    interior_all = np.argwhere(interior_mask)

    inner = float(_cfg(config, "context_inner_radius_mm", 2.0))
    outer = float(
        _cfg(config, "context_outer_radius_mm", _cfg(config, "context_radius_mm", 28.0))
    )
    boundary_depth = float(_cfg(config, "boundary_depth_mm", 3.0))
    separation = float(_cfg(config, "context_liver_surface_separation_mm", 1.0))
    context_mask = (
        organ
        & ~footprint
        & (outside >= inner)
        & (outside <= outer)
        & (liver_depth > boundary_depth + separation)
    )
    context_all = np.argwhere(context_mask)

    liver_mask = organ & ~footprint & (liver_depth <= boundary_depth)
    if not np.any(liver_mask):
        valid = organ & ~footprint
        if np.any(valid):
            minimum = float(np.min(liver_depth[valid]))
            liver_mask = valid & (liver_depth <= minimum + max(1.0, boundary_depth))
    liver_all = np.argwhere(liver_mask)

    coordinates = {
        "surface": voxel_grid_coordinates(
            surface_all,
            spacing_array,
            float(_cfg(config, "canonical_surface_spacing_mm", 2.0)),
        ),
        "interior": voxel_grid_coordinates(
            interior_all,
            spacing_array,
            float(_cfg(config, "canonical_interior_spacing_mm", 3.0)),
        ),
        "context": voxel_grid_coordinates(
            context_all,
            spacing_array,
            float(_cfg(config, "canonical_context_spacing_mm", 2.5)),
        ),
        "liver_surface": voxel_grid_coordinates(
            liver_all,
            spacing_array,
            float(_cfg(config, "canonical_liver_spacing_mm", 4.0)),
        ),
    }
    validate_canonical_coordinates(fields, coordinates, config)
    total = sum(int(value.shape[0]) for value in coordinates.values())
    limit = int(_cfg(config, "canonical_node_limit", 250_000))
    if limit > 0 and total > limit:
        counts = {key: int(value.shape[0]) for key, value in coordinates.items()}
        raise RuntimeError(
            "V22 canonical graph exceeds canonical_node_limit without truncation: "
            f"counts={counts}, total={total}, limit={limit}. Increase physical "
            "cell spacing or the explicit hard limit; the builder will not silently sample."
        )
    return coordinates


def validate_canonical_coordinates(
    fields: object,
    coordinates: Mapping[str, np.ndarray],
    config: object,
) -> None:
    footprint = np.asarray(_field(fields, "footprint"), dtype=bool)
    organ = np.asarray(_field(fields, "organ_mask", "organ"), dtype=bool)
    outside = np.asarray(
        _field(fields, "outside_tumor_mm", "tumor_outer"), dtype=np.float32
    )
    liver_depth = np.asarray(
        _field(fields, "organ_depth", "organ_boundary_mm", "liver_depth_mm"),
        dtype=np.float32,
    )
    shape = np.asarray(footprint.shape, dtype=np.int64)
    for name in ("surface", "interior", "context", "liver_surface"):
        values = np.asarray(coordinates[name], dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"V22 {name} coordinates are invalid: {values.shape}")
        if values.shape[0] == 0:
            raise EmptyCanonicalNodeError(name, tuple(values.shape))
        if np.any(values < 0) or np.any(values >= shape[None]):
            raise ValueError(f"V22 {name} contains out-of-bounds coordinates")
        if np.unique(values, axis=0).shape[0] != values.shape[0]:
            raise ValueError(f"V22 {name} contains duplicate nodes")

    surface = np.asarray(coordinates["surface"], dtype=np.int64)
    context = np.asarray(coordinates["context"], dtype=np.int64)
    liver = np.asarray(coordinates["liver_surface"], dtype=np.int64)
    if not bool(np.all(surface_mask(footprint)[tuple(surface.T)])):
        raise ValueError("V22 tumor-surface node is not on the tumor boundary")
    if bool(np.any(footprint[tuple(context.T)])) or not bool(
        np.all(organ[tuple(context.T)])
    ):
        raise ValueError("V22 context nodes are not pure liver parenchyma")
    inner = float(_cfg(config, "context_inner_radius_mm", 2.0))
    outer = float(
        _cfg(config, "context_outer_radius_mm", _cfg(config, "context_radius_mm", 28.0))
    )
    distance = outside[tuple(context.T)]
    if bool(np.any(distance < inner - 1e-4)) or bool(
        np.any(distance > outer + 1e-4)
    ):
        raise ValueError("V22 context nodes are outside the configured annulus")
    boundary = float(_cfg(config, "boundary_depth_mm", 3.0))
    separation = float(_cfg(config, "context_liver_surface_separation_mm", 1.0))
    if bool(np.any(liver_depth[tuple(context.T)] <= boundary + separation - 1e-4)):
        raise ValueError("V22 context overlaps the liver-surface band")
    context_rows = {tuple(int(v) for v in row) for row in context}
    if any(tuple(int(v) for v in row) in context_rows for row in liver):
        raise ValueError("V22 context and liver-surface nodes overlap")


def radius_edges(
    position_mm: np.ndarray,
    radius_mm: float,
    *,
    include_self: bool = False,
) -> np.ndarray:
    """Build directed sparse radius edges with a physical-space KD-tree."""
    points = np.asarray(position_mm, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"position_mm must be [N,3], got {points.shape}")
    count = int(points.shape[0])
    if count == 0:
        return np.empty((2, 0), dtype=np.int64)
    radius = float(radius_mm)
    if radius <= 0.0:
        raise ValueError("radius_mm must be positive")
    if count == 1:
        if include_self:
            return np.asarray([[0], [0]], dtype=np.int64)
        return np.empty((2, 0), dtype=np.int64)
    tree = cKDTree(points, compact_nodes=True, balanced_tree=True)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if pairs.size == 0:
        edge = np.empty((2, 0), dtype=np.int64)
    else:
        first = pairs[:, 0].astype(np.int64, copy=False)
        second = pairs[:, 1].astype(np.int64, copy=False)
        edge = np.stack(
            [np.concatenate([first, second]), np.concatenate([second, first])],
            axis=0,
        )
    if include_self:
        ids = np.arange(count, dtype=np.int64)
        self_edge = np.stack([ids, ids], axis=0)
        edge = np.concatenate([edge, self_edge], axis=1)
    return edge


def cross_radius_edges(
    source_position_mm: np.ndarray,
    destination_position_mm: np.ndarray,
    radius_mm: float,
) -> np.ndarray:
    """Build all source-to-destination edges inside a physical radius."""
    source = np.asarray(source_position_mm, dtype=np.float32)
    destination = np.asarray(destination_position_mm, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"source_position_mm must be [N,3], got {source.shape}")
    if destination.ndim != 2 or destination.shape[1] != 3:
        raise ValueError(
            f"destination_position_mm must be [N,3], got {destination.shape}"
        )
    if source.shape[0] == 0 or destination.shape[0] == 0:
        return np.empty((2, 0), dtype=np.int64)
    radius = float(radius_mm)
    if radius <= 0.0:
        raise ValueError("radius_mm must be positive")
    source_tree = cKDTree(source, compact_nodes=True, balanced_tree=True)
    destination_tree = cKDTree(destination, compact_nodes=True, balanced_tree=True)
    neighbors = destination_tree.query_ball_tree(source_tree, radius)
    source_ids: list[int] = []
    destination_ids: list[int] = []
    for destination_id, local in enumerate(neighbors):
        if not local:
            continue
        source_ids.extend(int(value) for value in local)
        destination_ids.extend([int(destination_id)] * len(local))
    if not source_ids:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray([source_ids, destination_ids], dtype=np.int64)


def source_node_specifications(
    coordinates: Mapping[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, str, float]]:
    return {
        "tumor_surface": (np.asarray(coordinates["surface"]), "tumor", 1.0),
        "tumor_interior": (np.asarray(coordinates["interior"]), "tumor", 0.0),
        "source_context": (np.asarray(coordinates["context"]), "tumor", 0.0),
        "source_liver_surface": (
            np.asarray(coordinates["liver_surface"]),
            "liver",
            1.0,
        ),
    }


def target_node_specifications(
    coordinates: Mapping[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, str, float]]:
    return {
        "target_context": (np.asarray(coordinates["context"]), "tumor", 0.0),
        "target_liver_surface": (
            np.asarray(coordinates["liver_surface"]),
            "liver",
            1.0,
        ),
    }
