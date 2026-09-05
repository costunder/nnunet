"""HierCP V21 adaptive full-shape Level-0 graph primitives.

The existing dense encoder still receives a fixed ``patch_size³`` tensor, but
all handcrafted geometry and semantic nodes are computed in an adaptive native
ROI that contains the complete paste mask. Coordinates follow the project's
Z-Y-X voxel convention; physical coordinates are ``coordinate * spacing``.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil, pi
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

V21_SCHEMA_VERSION = "adaptive_balanced_v21"
V21_MARKER = "HIERCP_ADAPTIVE_BALANCED_V21"


@dataclass(frozen=True)
class CoordinateDiagnostics:
    surface_nodes: int
    interior_nodes: int
    context_nodes: int
    liver_surface_nodes: int
    context_surface_ratio: float
    active_context_sectors: int
    max_context_sector_fraction: float
    context_liver_surface_overlap: int
    exact_liver_surface_fraction: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "surface_nodes": int(self.surface_nodes),
            "interior_nodes": int(self.interior_nodes),
            "context_nodes": int(self.context_nodes),
            "liver_surface_nodes": int(self.liver_surface_nodes),
            "context_surface_ratio": float(self.context_surface_ratio),
            "active_context_sectors": int(self.active_context_sectors),
            "max_context_sector_fraction": float(self.max_context_sector_fraction),
            "context_liver_surface_overlap": int(self.context_liver_surface_overlap),
            "exact_liver_surface_fraction": float(self.exact_liver_surface_fraction),
        }


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


def is_v21_graph(config: object) -> bool:
    return (
        str(_cfg(config, "graph_schema_version", "")) == V21_SCHEMA_VERSION
        and bool(_cfg(config, "adaptive_nodes", False))
        and bool(_cfg(config, "adaptive_source_full_shape", False))
    )


def exact_source_footprint(source: object) -> np.ndarray:
    """Return the exact mask transformed by Copy-Paste, with a zero-loss check."""
    if not hasattr(source, "patch_mask"):
        raise AttributeError("V21 requires SourceTumor.patch_mask")
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
    """Compute a native ROI that cannot crop the footprint."""
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
    if requested_margin > radius_cap:
        raise ValueError(
            "Requested adaptive ROI context exceeds adaptive_roi_max_radius_mm; "
            "automatic context reduction is disabled. "
            f"requested_margin_mm={requested_margin:.6g}, "
            f"radius_guard_mm={radius_cap:.6g}, footprint_shape={tuple(footprint)}, "
            f"spacing_mm={tuple(float(value) for value in spacing_array)}. "
            "Increase the guard only after measuring VRAM/RAM and documenting the "
            "effect, or use hardware with sufficient capacity."
        )
    pad = np.ceil(requested_margin / spacing_array).astype(np.int64)
    native_shape = _odd(footprint + 2 * pad)

    max_voxels = int(_cfg(config, "adaptive_roi_max_voxels", 8_000_000))
    if max_voxels <= 0:
        raise ValueError("adaptive_roi_max_voxels must be positive")
    native_voxels = int(np.prod(native_shape, dtype=np.int64))
    if native_voxels > max_voxels:
        raise ValueError(
            "Requested adaptive ROI exceeds adaptive_roi_max_voxels; automatic "
            "margin/shape reduction is disabled. "
            f"footprint_shape={tuple(footprint)}, "
            f"spacing_mm={tuple(float(value) for value in spacing_array)}, "
            f"requested_margin_mm={requested_margin:.6g}, "
            f"requested_shape={tuple(int(value) for value in native_shape)}, "
            f"requested_voxels={native_voxels}, voxel_guard={max_voxels}. "
            "Measure memory and choose a larger guard/hardware; do not crop the "
            "research input implicitly."
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
        raise ValueError(f"V21 footprint must be a non-empty 3-D mask: {footprint.shape}")
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
        raise RuntimeError(f"V21 adaptive ROI cropped tumor: before={before}, after={after}")

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
    eroded = ndi.binary_erosion(
        mask,
        structure=ndi.generate_binary_structure(3, 1),
        border_value=0,
    )
    return np.asarray(mask, dtype=bool) & ~eroded


def physical_surface_area_mm2(mask: np.ndarray, spacing: Sequence[float]) -> float:
    """Estimate surface area by counting exposed six-connected voxel faces."""
    binary = np.asarray(mask, dtype=bool)
    spacing_array = np.asarray(spacing, dtype=np.float64)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    area = 0.0
    for axis in range(3):
        before = [slice(1, -1)] * 3
        after = [slice(1, -1)] * 3
        before[axis] = slice(0, -2)
        after[axis] = slice(2, None)
        face_area = float(np.prod(np.delete(spacing_array, axis)))
        area += float(np.count_nonzero(binary & ~padded[tuple(before)])) * face_area
        area += float(np.count_nonzero(binary & ~padded[tuple(after)])) * face_area
    return area


def _adaptive_count(
    value: float,
    per_node: float,
    minimum: int,
    maximum: int,
    available: int,
) -> int:
    if available <= 0:
        return 0
    proposed = int(ceil(float(value) / max(float(per_node), 1e-6)))
    return min(int(available), max(int(minimum), min(int(maximum), proposed)))


def _fps_indices(
    points_mm: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    pool_limit: int,
) -> np.ndarray:
    points = np.asarray(points_mm, dtype=np.float32)
    available = int(points.shape[0])
    count = min(max(0, int(count)), available)
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    if count == available:
        return np.arange(available, dtype=np.int64)
    if available > int(pool_limit):
        pool_ids = np.sort(rng.choice(available, size=int(pool_limit), replace=False))
        pool = points[pool_ids]
    else:
        pool_ids = np.arange(available, dtype=np.int64)
        pool = points
    count = min(count, int(pool.shape[0]))
    selected = np.empty(count, dtype=np.int64)
    centroid = pool.mean(axis=0, keepdims=True)
    selected[0] = int(np.argmax(np.sum((pool - centroid) ** 2, axis=1)))
    minimum_distance = np.sum((pool - pool[selected[0]]) ** 2, axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(minimum_distance))
        distance = np.sum((pool - pool[selected[index]]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    return pool_ids[selected]


def fps_coordinates(
    coordinates: np.ndarray,
    count: int,
    spacing: Sequence[float],
    rng: np.random.Generator,
    *,
    pool_limit: int,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected coordinates [N,3], got {coords.shape}")
    if coords.shape[0] == 0 or count <= 0:
        return np.empty((0, 3), dtype=np.int64)
    points_mm = coords.astype(np.float32) * np.asarray(spacing, dtype=np.float32)[None]
    return coords[_fps_indices(points_mm, count, rng, pool_limit=pool_limit)]


def _unique_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.shape[0] <= 1:
        return array
    _, first = np.unique(array, axis=0, return_index=True)
    return array[np.sort(first)]


def _sector_ids(
    coordinates: np.ndarray,
    center_mm: np.ndarray,
    spacing: np.ndarray,
    outside_distance_mm: np.ndarray,
    config: object,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.int64)
    vector = coords.astype(np.float32) * spacing[None] - center_mm[None]
    radius = np.linalg.norm(vector, axis=1)
    unit = vector / np.maximum(radius[:, None], 1e-6)
    azimuth = np.mod(np.arctan2(unit[:, 1], unit[:, 2]), 2.0 * pi)
    elevation = np.arcsin(np.clip(unit[:, 0], -1.0, 1.0)) + pi / 2.0

    radial_bins = max(1, int(_cfg(config, "context_radial_bins", 4)))
    azimuth_bins = max(1, int(_cfg(config, "context_azimuth_bins", 8)))
    elevation_bins = max(1, int(_cfg(config, "context_elevation_bins", 4)))
    inner = float(_cfg(config, "context_inner_radius_mm", 2.0))
    outer = float(
        _cfg(config, "context_outer_radius_mm", _cfg(config, "context_radius_mm", 24.0))
    )
    distance = outside_distance_mm[tuple(coords.T)]
    radial = np.floor(
        (np.clip(distance, inner, outer) - inner)
        / max(outer - inner, 1e-6)
        * radial_bins
    ).astype(np.int64)
    radial = np.clip(radial, 0, radial_bins - 1)
    az = np.clip(
        np.floor(azimuth / (2.0 * pi) * azimuth_bins).astype(np.int64),
        0,
        azimuth_bins - 1,
    )
    el = np.clip(
        np.floor(elevation / pi * elevation_bins).astype(np.int64),
        0,
        elevation_bins - 1,
    )
    return radial * azimuth_bins * elevation_bins + az * elevation_bins + el


def _balanced_sector_select(
    coordinates: np.ndarray,
    target_count: int,
    sector_ids: np.ndarray,
    rng: np.random.Generator,
    config: object,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.int64)
    target_count = min(max(int(target_count), 0), int(coords.shape[0]))
    if target_count <= 0:
        return np.empty((0, 3), dtype=np.int64)
    buckets: dict[int, np.ndarray] = {}
    for sector in np.unique(sector_ids):
        local = coords[sector_ids == sector]
        buckets[int(sector)] = local[rng.permutation(len(local))]
    sectors = sorted(buckets)
    if sectors:
        shift = int(rng.integers(len(sectors)))
        sectors = sectors[shift:] + sectors[:shift]
    active = max(1, len(sectors))
    configured_fraction = float(_cfg(config, "context_sector_max_fraction", 0.15))
    cap = max(
        1,
        int(ceil(target_count * configured_fraction)),
        int(ceil(target_count / active)),
    )
    offsets: dict[int, int] = defaultdict(int)
    used: dict[int, int] = defaultdict(int)
    selected: list[np.ndarray] = []
    progress = True
    while len(selected) < target_count and progress:
        progress = False
        for sector in sectors:
            if len(selected) >= target_count:
                break
            offset = offsets[sector]
            bucket = buckets[sector]
            if offset >= len(bucket) or used[sector] >= cap:
                continue
            selected.append(bucket[offset])
            offsets[sector] += 1
            used[sector] += 1
            progress = True
    if not selected:
        return np.empty((0, 3), dtype=np.int64)
    return np.stack(selected, axis=0).astype(np.int64)


def _outward_walk(
    surface_coordinates: np.ndarray,
    tumor_normal: np.ndarray,
    valid_context_coordinates: np.ndarray,
    spacing: Sequence[float],
    outside_distance_mm: np.ndarray,
    config: object,
) -> np.ndarray:
    if surface_coordinates.shape[0] == 0 or valid_context_coordinates.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64)
    spacing_array = np.asarray(spacing, dtype=np.float32)
    context_points_mm = valid_context_coordinates.astype(np.float32) * spacing_array[None]
    tree = cKDTree(
        context_points_mm,
        leafsize=max(4, int(_cfg(config, "kdtree_leafsize", 32))),
        compact_nodes=True,
        balanced_tree=True,
    )
    step_mm = float(_cfg(config, "outward_step_mm", 2.0))
    max_steps = max(1, int(_cfg(config, "outward_max_steps", 14)))
    snap_radius = float(_cfg(config, "outward_snap_radius_mm", 2.75))
    minimum_cosine = float(_cfg(config, "outward_min_cosine", 0.10))
    inner = float(_cfg(config, "context_inner_radius_mm", 2.0))
    outer = float(
        _cfg(config, "context_outer_radius_mm", _cfg(config, "context_radius_mm", 24.0))
    )

    candidates: list[np.ndarray] = []
    for coordinate in surface_coordinates:
        normal = np.asarray(tumor_normal[tuple(coordinate)], dtype=np.float32)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm < 1e-5:
            continue
        normal /= norm
        origin_mm = coordinate.astype(np.float32) * spacing_array
        for step in range(1, max_steps + 1):
            query_mm = origin_mm + normal * (step_mm * step)
            distance, neighbor = tree.query(
                query_mm,
                k=1,
                distance_upper_bound=snap_radius,
            )
            if not np.isfinite(distance) or int(neighbor) >= len(valid_context_coordinates):
                continue
            candidate = valid_context_coordinates[int(neighbor)]
            radial_distance = float(outside_distance_mm[tuple(candidate)])
            if radial_distance < inner or radial_distance > outer:
                continue
            displacement = candidate.astype(np.float32) * spacing_array - origin_mm
            displacement_norm = float(np.linalg.norm(displacement))
            if displacement_norm < 1e-6:
                continue
            cosine = float(np.dot(displacement / displacement_norm, normal))
            if cosine < minimum_cosine:
                continue
            candidates.append(candidate)
    if not candidates:
        return np.empty((0, 3), dtype=np.int64)
    return _unique_rows(np.stack(candidates, axis=0))


def adaptive_coordinate_sets(
    fields: object,
    config: object,
    rng: np.random.Generator,
    spacing: Sequence[float],
) -> dict[str, np.ndarray]:
    """Create V21 variable-size semantic node coordinates."""
    footprint = np.asarray(_field(fields, "footprint"), dtype=bool)
    organ = np.asarray(_field(fields, "organ_mask", "organ"), dtype=bool)
    outside = np.asarray(_field(fields, "outside_tumor_mm", "tumor_outer"), dtype=np.float32)
    liver_depth = np.asarray(_field(fields, "organ_depth", "organ_boundary_mm", "liver_depth_mm"), dtype=np.float32)
    tumor_normal = np.asarray(_field(fields, "tumor_normals", "tumor_normal"), dtype=np.float32)
    if not (footprint.shape == organ.shape == outside.shape == liver_depth.shape):
        raise ValueError("V21 PatchFields arrays have inconsistent shapes")
    spacing_array = np.asarray(spacing, dtype=np.float32)
    pool_limit = max(512, int(_cfg(config, "context_candidate_pool_max", 50_000)))

    all_surface = np.argwhere(surface_mask(footprint))
    surface_count = _adaptive_count(
        physical_surface_area_mm2(footprint, spacing_array),
        float(_cfg(config, "surface_area_per_node_mm2", 28.0)),
        int(_cfg(config, "surface_nodes_min", 48)),
        int(_cfg(config, "surface_nodes_max", 160)),
        int(all_surface.shape[0]),
    )
    surface = fps_coordinates(
        all_surface,
        surface_count,
        spacing_array,
        rng,
        pool_limit=pool_limit,
    )

    interior_mask = ndi.binary_erosion(
        footprint,
        structure=ndi.generate_binary_structure(3, 1),
        border_value=0,
    )
    if not np.any(interior_mask):
        interior_mask = footprint
    all_interior = np.argwhere(interior_mask)
    physical_volume_mm3 = float(np.count_nonzero(footprint)) * float(np.prod(spacing_array))
    interior_count = _adaptive_count(
        physical_volume_mm3,
        float(_cfg(config, "interior_volume_per_node_mm3", 500.0)),
        int(_cfg(config, "interior_nodes_min", 24)),
        int(_cfg(config, "interior_nodes_max", 96)),
        int(all_interior.shape[0]),
    )
    interior = fps_coordinates(
        all_interior,
        interior_count,
        spacing_array,
        rng,
        pool_limit=pool_limit,
    )

    inner = float(_cfg(config, "context_inner_radius_mm", 2.0))
    outer = float(
        _cfg(config, "context_outer_radius_mm", _cfg(config, "context_radius_mm", 24.0))
    )
    boundary_depth = float(_cfg(config, "boundary_depth_mm", 3.0))
    surface_separation = float(
        _cfg(config, "context_liver_surface_separation_mm", 1.0)
    )
    parenchyma_mask = (
        organ
        & ~footprint
        & (outside >= inner)
        & (outside <= outer)
        & (liver_depth > boundary_depth + surface_separation)
    )
    all_context = np.argwhere(parenchyma_mask)
    requested_context = int(
        round(float(_cfg(config, "context_ratio", 2.0)) * max(1, len(surface)))
    )
    requested_context = max(
        int(_cfg(config, "context_nodes_min", 96)),
        min(int(_cfg(config, "context_nodes_max", 384)), requested_context),
    )
    requested_context = min(requested_context, int(all_context.shape[0]))

    walked = _outward_walk(
        surface,
        tumor_normal,
        all_context,
        spacing_array,
        outside,
        config,
    )
    if walked.shape[0] >= requested_context:
        candidate_pool = walked
    else:
        if all_context.shape[0] > pool_limit:
            recall_ids = rng.choice(all_context.shape[0], size=pool_limit, replace=False)
            recall = all_context[np.sort(recall_ids)]
        else:
            recall = all_context
        candidate_pool = (
            _unique_rows(np.concatenate([walked, recall], axis=0))
            if walked.size
            else recall
        )
    if candidate_pool.shape[0] and requested_context > 0:
        center_mm = np.argwhere(footprint).mean(axis=0).astype(np.float32) * spacing_array
        sectors = _sector_ids(
            candidate_pool,
            center_mm,
            spacing_array,
            outside,
            config,
        )
        context = _balanced_sector_select(
            candidate_pool,
            requested_context,
            sectors,
            rng,
            config,
        )
    else:
        context = np.empty((0, 3), dtype=np.int64)

    exact_liver_surface = organ & ~footprint & (liver_depth <= boundary_depth)
    liver_candidates = np.argwhere(exact_liver_surface)
    if liver_candidates.shape[0] == 0:
        valid_organ = organ & ~footprint
        if np.any(valid_organ):
            minimum_depth = float(np.min(liver_depth[valid_organ]))
            liver_candidates = np.argwhere(
                valid_organ
                & (liver_depth <= minimum_depth + max(1.0, boundary_depth))
            )
    requested_liver = int(
        round(float(_cfg(config, "liver_anchor_ratio", 0.5)) * max(1, len(surface)))
    )
    requested_liver = max(
        int(_cfg(config, "liver_anchor_nodes_min", 24)),
        min(int(_cfg(config, "liver_anchor_nodes_max", 96)), requested_liver),
    )
    requested_liver = min(requested_liver, int(liver_candidates.shape[0]))
    liver_surface = fps_coordinates(
        liver_candidates,
        requested_liver,
        spacing_array,
        rng,
        pool_limit=pool_limit,
    )

    coordinates = {
        "surface": surface.astype(np.int64, copy=False),
        "interior": interior.astype(np.int64, copy=False),
        "context": context.astype(np.int64, copy=False),
        "liver_surface": liver_surface.astype(np.int64, copy=False),
    }
    validate_coordinate_sets(fields, coordinates, config, spacing_array)
    return coordinates


def _row_overlap(first: np.ndarray, second: np.ndarray) -> int:
    if first.size == 0 or second.size == 0:
        return 0
    lookup = {tuple(int(value) for value in row) for row in np.asarray(first)}
    return sum(tuple(int(value) for value in row) in lookup for row in np.asarray(second))


def coordinate_diagnostics(
    fields: object,
    coordinates: Mapping[str, np.ndarray],
    config: object,
    spacing: Sequence[float],
) -> CoordinateDiagnostics:
    surface = np.asarray(coordinates["surface"], dtype=np.int64)
    interior = np.asarray(coordinates["interior"], dtype=np.int64)
    context = np.asarray(coordinates["context"], dtype=np.int64)
    liver_surface = np.asarray(coordinates["liver_surface"], dtype=np.int64)
    footprint = np.asarray(_field(fields, "footprint"), dtype=bool)
    outside = np.asarray(_field(fields, "outside_tumor_mm", "tumor_outer"), dtype=np.float32)
    liver_depth = np.asarray(_field(fields, "organ_depth", "organ_boundary_mm", "liver_depth_mm"), dtype=np.float32)
    spacing_array = np.asarray(spacing, dtype=np.float32)
    if context.shape[0]:
        center_mm = np.argwhere(footprint).mean(axis=0).astype(np.float32) * spacing_array
        sectors = _sector_ids(context, center_mm, spacing_array, outside, config)
        _, counts = np.unique(sectors, return_counts=True)
        active_sectors = int(counts.shape[0])
        maximum_fraction = float(counts.max() / max(1, counts.sum()))
    else:
        active_sectors = 0
        maximum_fraction = 0.0
    boundary_depth = float(_cfg(config, "boundary_depth_mm", 3.0))
    exact_fraction = (
        float(np.mean(liver_depth[tuple(liver_surface.T)] <= boundary_depth))
        if liver_surface.shape[0]
        else 0.0
    )
    return CoordinateDiagnostics(
        surface_nodes=int(surface.shape[0]),
        interior_nodes=int(interior.shape[0]),
        context_nodes=int(context.shape[0]),
        liver_surface_nodes=int(liver_surface.shape[0]),
        context_surface_ratio=float(context.shape[0] / max(1, surface.shape[0])),
        active_context_sectors=active_sectors,
        max_context_sector_fraction=maximum_fraction,
        context_liver_surface_overlap=int(_row_overlap(context, liver_surface)),
        exact_liver_surface_fraction=exact_fraction,
    )


def validate_coordinate_sets(
    fields: object,
    coordinates: Mapping[str, np.ndarray],
    config: object,
    spacing: Sequence[float],
) -> None:
    footprint = np.asarray(_field(fields, "footprint"), dtype=bool)
    organ = np.asarray(_field(fields, "organ_mask", "organ"), dtype=bool)
    outside = np.asarray(_field(fields, "outside_tumor_mm", "tumor_outer"), dtype=np.float32)
    liver_depth = np.asarray(_field(fields, "organ_depth", "organ_boundary_mm", "liver_depth_mm"), dtype=np.float32)
    shape = np.asarray(footprint.shape, dtype=np.int64)
    for name in ("surface", "interior", "context", "liver_surface"):
        values = np.asarray(coordinates[name], dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"{name} coordinates must have shape [N,3], got {values.shape}")
        if values.shape[0] == 0:
            raise ValueError(f"V21 produced zero {name} nodes")
        if np.any(values < 0) or np.any(values >= shape[None]):
            raise ValueError(f"{name} contains out-of-bounds coordinates")
        if np.unique(values, axis=0).shape[0] != values.shape[0]:
            raise ValueError(f"{name} contains duplicate nodes")

    surface = np.asarray(coordinates["surface"], dtype=np.int64)
    context = np.asarray(coordinates["context"], dtype=np.int64)
    liver_surface = np.asarray(coordinates["liver_surface"], dtype=np.int64)
    if not bool(np.all(surface_mask(footprint)[tuple(surface.T)])):
        raise ValueError("V21 tumor_surface node is not on the complete tumor boundary")
    if bool(np.any(footprint[tuple(context.T)])) or not bool(np.all(organ[tuple(context.T)])):
        raise ValueError("V21 context nodes are not pure liver parenchyma")
    inner = float(_cfg(config, "context_inner_radius_mm", 2.0))
    outer = float(
        _cfg(config, "context_outer_radius_mm", _cfg(config, "context_radius_mm", 24.0))
    )
    context_distance = outside[tuple(context.T)]
    if bool(np.any(context_distance < inner - 1e-4)) or bool(
        np.any(context_distance > outer + 1e-4)
    ):
        raise ValueError("V21 context distance is outside the configured annulus")
    boundary_depth = float(_cfg(config, "boundary_depth_mm", 3.0))
    separation = float(_cfg(config, "context_liver_surface_separation_mm", 1.0))
    if bool(
        np.any(liver_depth[tuple(context.T)] <= boundary_depth + separation - 1e-4)
    ):
        raise ValueError("V21 parenchymal context overlaps the liver-surface band")
    if _row_overlap(context, liver_surface):
        raise ValueError("V21 context and liver-surface anchors overlap")

    diagnostics = coordinate_diagnostics(fields, coordinates, config, spacing)
    minimum_ratio = float(_cfg(config, "context_ratio_min_audit", 1.25))
    if diagnostics.context_surface_ratio + 1e-6 < minimum_ratio:
        raise ValueError(
            "V21 context/surface ratio is too small: "
            f"ratio={diagnostics.context_surface_ratio:.3f}, minimum={minimum_ratio:.3f}"
        )
    allowed_fraction = max(
        float(_cfg(config, "context_sector_max_fraction", 0.15)) + 0.05,
        1.0 / max(1, diagnostics.active_context_sectors) + 0.05,
    )
    if diagnostics.max_context_sector_fraction > allowed_fraction + 1e-6:
        raise ValueError(
            "V21 context is concentrated in one sector: "
            f"max_fraction={diagnostics.max_context_sector_fraction:.3f}, "
            f"allowed={allowed_fraction:.3f}"
        )


def kdtree_knn_edges(position: np.ndarray, k: int, *, leafsize: int = 32) -> np.ndarray:
    """Directed nearest-neighbour-to-query edges using a sparse KD-tree."""
    points = np.asarray(position, dtype=np.float32)
    node_count = int(points.shape[0])
    if node_count <= 1:
        return np.empty((2, 0), dtype=np.int64)
    effective_k = min(max(1, int(k)), node_count - 1)
    tree = cKDTree(
        points,
        leafsize=max(4, int(leafsize)),
        compact_nodes=True,
        balanced_tree=True,
    )
    _, neighbors = tree.query(points, k=effective_k + 1)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    selected = np.empty((node_count, effective_k), dtype=np.int64)
    for row in range(node_count):
        candidates = [int(value) for value in neighbors[row] if int(value) != row]
        if len(candidates) < effective_k:
            _, fallback = tree.query(points[row], k=node_count)
            for value in np.atleast_1d(fallback):
                value_int = int(value)
                if value_int != row and value_int not in candidates:
                    candidates.append(value_int)
        selected[row] = np.asarray(candidates[:effective_k], dtype=np.int64)
    source = selected.reshape(-1)
    destination = np.repeat(np.arange(node_count, dtype=np.int64), effective_k)
    return np.stack([source, destination], axis=0)


def kdtree_cross_edges(
    source_position: np.ndarray,
    destination_position: np.ndarray,
    k: int,
    *,
    leafsize: int = 32,
) -> np.ndarray:
    source_points = np.asarray(source_position, dtype=np.float32)
    destination_points = np.asarray(destination_position, dtype=np.float32)
    if source_points.shape[0] == 0 or destination_points.shape[0] == 0:
        return np.empty((2, 0), dtype=np.int64)
    effective_k = min(max(1, int(k)), int(source_points.shape[0]))
    tree = cKDTree(
        source_points,
        leafsize=max(4, int(leafsize)),
        compact_nodes=True,
        balanced_tree=True,
    )
    _, neighbors = tree.query(destination_points, k=effective_k)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    source = neighbors.reshape(-1)
    destination = np.repeat(
        np.arange(destination_points.shape[0], dtype=np.int64),
        effective_k,
    )
    return np.stack([source, destination], axis=0)


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


def _synthetic_fields(radius: float, shape: tuple[int, int, int]) -> SimpleNamespace:
    grid = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, dtype=np.float32) - 1.0)[:, None, None, None] * 0.5
    distance = np.sqrt(np.sum((grid - center) ** 2, axis=0))
    footprint = distance <= float(radius)
    organ = distance <= min(shape) * 0.46
    sdf = signed_distance(footprint, (1.0, 1.0, 1.0))
    normal, curvature = normal_and_curvature(sdf, (1.0, 1.0, 1.0))
    liver_depth = ndi.distance_transform_edt(organ).astype(np.float32)
    outside = ndi.distance_transform_edt(~footprint).astype(np.float32)
    zeros = np.zeros(shape, dtype=np.float32)
    return SimpleNamespace(
        footprint=footprint,
        organ=organ,
        outside_tumor_mm=outside,
        liver_depth_mm=liver_depth,
        tumor_normal=normal,
        ct_norm=zeros,
        local_mean=zeros,
        local_std=zeros,
        gradient=zeros,
        tumor_sdf_mm=sdf,
        tumor_sdf_norm=sdf,
        liver_depth_norm=liver_depth,
        tumor_curvature=curvature,
        liver_normal=normal,
        liver_curvature=curvature,
    )


def smoke_config() -> SimpleNamespace:
    return SimpleNamespace(
        graph_schema_version=V21_SCHEMA_VERSION,
        adaptive_nodes=True,
        adaptive_source_full_shape=True,
        surface_area_per_node_mm2=28.0,
        surface_nodes_min=24,
        surface_nodes_max=128,
        interior_volume_per_node_mm3=300.0,
        interior_nodes_min=12,
        interior_nodes_max=64,
        context_ratio=2.0,
        context_nodes_min=48,
        context_nodes_max=256,
        context_ratio_min_audit=1.25,
        context_inner_radius_mm=2.0,
        context_outer_radius_mm=20.0,
        context_liver_surface_separation_mm=1.0,
        context_radial_bins=4,
        context_azimuth_bins=8,
        context_elevation_bins=4,
        context_sector_max_fraction=0.20,
        outward_step_mm=2.0,
        outward_max_steps=10,
        outward_snap_radius_mm=2.5,
        outward_min_cosine=0.0,
        liver_anchor_ratio=0.5,
        liver_anchor_nodes_min=12,
        liver_anchor_nodes_max=64,
        boundary_depth_mm=3.0,
        context_candidate_pool_max=20_000,
        kdtree_leafsize=16,
    )


def run_geometry_smoke(config: object | None = None) -> dict[str, Any]:
    config = config or smoke_config()
    small_fields = _synthetic_fields(6.0, (65, 65, 65))
    large_fields = _synthetic_fields(13.0, (81, 81, 81))
    small = adaptive_coordinate_sets(
        small_fields,
        config,
        np.random.default_rng(11),
        (1.0, 1.0, 1.0),
    )
    large = adaptive_coordinate_sets(
        large_fields,
        config,
        np.random.default_rng(12),
        (1.0, 1.0, 1.0),
    )
    small_diagnostics = coordinate_diagnostics(
        small_fields,
        small,
        config,
        (1.0, 1.0, 1.0),
    )
    large_diagnostics = coordinate_diagnostics(
        large_fields,
        large,
        config,
        (1.0, 1.0, 1.0),
    )
    if len(small["surface"]) == len(large["surface"]):
        raise AssertionError("Adaptive surface counts did not vary across tumor sizes")
    if len(small["context"]) == len(large["context"]):
        raise AssertionError("Adaptive context counts did not vary across tumor sizes")
    edges = kdtree_knn_edges(large["context"].astype(np.float32), 6)
    if edges.shape[0] != 2 or edges.shape[1] < len(large["context"]):
        raise AssertionError(f"Invalid KD-tree edge matrix: {edges.shape}")
    return {
        "schema": V21_SCHEMA_VERSION,
        "small": small_diagnostics.to_dict(),
        "large": large_diagnostics.to_dict(),
        "large_context_edges": int(edges.shape[1]),
    }
