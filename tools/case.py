#!/usr/bin/env python3
"""Audit fixed patch crop loss and V21 graph geometry on a NIfTI label."""
from __future__ import annotations

import argparse
import gzip
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from hiercp.geometry import (
    V21_SCHEMA_VERSION,
    adaptive_coordinate_sets,
    adaptive_native_shape,
    center_crop_or_pad,
    coordinate_diagnostics,
    extract_centered,
    normal_and_curvature,
    signed_distance,
)

_NIFTI_DTYPES = {
    2: np.dtype("u1"),
    4: np.dtype("i2"),
    8: np.dtype("i4"),
    16: np.dtype("f4"),
    64: np.dtype("f8"),
    256: np.dtype("i1"),
    512: np.dtype("u2"),
    768: np.dtype("u4"),
}


def load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    try:
        import nibabel as nib  # type: ignore
    except ImportError:
        nib = None
    if nib is not None:
        image = nib.load(str(path))
        data = np.asanyarray(image.dataobj)
        if data.ndim == 4:
            data = data[..., 0]
        return np.asarray(data), tuple(float(v) for v in image.header.get_zooms()[:3])

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:  # type: ignore[arg-type]
        blob = stream.read()
    if len(blob) < 352:
        raise ValueError(f"Not a NIfTI-1 file: {path}")
    little = struct.unpack("<i", blob[:4])[0] == 348
    endian = "<" if little else ">"
    if struct.unpack(f"{endian}i", blob[:4])[0] != 348:
        raise ValueError("Only NIfTI-1 single-file images are supported")
    dim = struct.unpack(f"{endian}8h", blob[40:56])
    ndim = int(dim[0])
    shape = tuple(int(v) for v in dim[1 : ndim + 1])
    datatype = int(struct.unpack(f"{endian}h", blob[70:72])[0])
    pixdim = struct.unpack(f"{endian}8f", blob[76:108])
    vox_offset = int(round(struct.unpack(f"{endian}f", blob[108:112])[0]))
    slope = float(struct.unpack(f"{endian}f", blob[112:116])[0])
    intercept = float(struct.unpack(f"{endian}f", blob[116:120])[0])
    if datatype not in _NIFTI_DTYPES:
        raise ValueError(f"Unsupported NIfTI datatype code: {datatype}")
    dtype = _NIFTI_DTYPES[datatype].newbyteorder(endian)
    count = int(np.prod(shape, dtype=np.int64))
    data = np.frombuffer(blob, dtype=dtype, count=count, offset=vox_offset).reshape(shape, order="F")
    if ndim == 4:
        data = data[..., 0]
    if slope not in (0.0, 1.0) or intercept != 0.0:
        data = data.astype(np.float32) * (1.0 if slope == 0.0 else slope) + intercept
    return np.asarray(data), tuple(float(v) for v in pixdim[1:4])


def bounding_box(mask: np.ndarray, pad: int = 0) -> tuple[slice, slice, slice]:
    coordinates = np.argwhere(mask)
    if coordinates.shape[0] == 0:
        raise ValueError("Cannot compute bounding box of an empty mask")
    lower = np.maximum(coordinates.min(axis=0) - int(pad), 0)
    upper = np.minimum(coordinates.max(axis=0) + 1 + int(pad), np.asarray(mask.shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))  # type: ignore[return-value]


def audit_config() -> SimpleNamespace:
    return SimpleNamespace(
        graph_schema_version=V21_SCHEMA_VERSION,
        adaptive_nodes=True,
        adaptive_source_full_shape=True,
        patch_size=48,
        adaptive_roi_margin_mm=30.0,
        adaptive_roi_max_radius_mm=64.0,
        adaptive_roi_max_voxels=8_000_000,
        surface_area_per_node_mm2=28.0,
        surface_nodes_min=48,
        surface_nodes_max=160,
        interior_volume_per_node_mm3=500.0,
        interior_nodes_min=24,
        interior_nodes_max=96,
        context_ratio=2.0,
        context_ratio_min_audit=1.25,
        context_nodes_min=96,
        context_nodes_max=384,
        context_inner_radius_mm=2.0,
        context_outer_radius_mm=28.0,
        context_liver_surface_separation_mm=1.0,
        context_radial_bins=4,
        context_azimuth_bins=8,
        context_elevation_bins=4,
        context_sector_max_fraction=0.15,
        outward_step_mm=2.0,
        outward_max_steps=14,
        outward_snap_radius_mm=2.75,
        outward_min_cosine=0.1,
        liver_anchor_ratio=0.5,
        liver_anchor_nodes_min=24,
        liver_anchor_nodes_max=96,
        liver_anchor_search_mm=64.0,
        boundary_depth_mm=3.0,
        context_candidate_pool_max=50_000,
        kdtree_leafsize=32,
    )


def audit_component(
    component: np.ndarray,
    organ: np.ndarray,
    organ_bbox: tuple[slice, slice, slice],
    organ_depth_bbox: np.ndarray,
    spacing: tuple[float, float, float],
    component_id: int,
    seed: int,
) -> dict[str, Any]:
    component_bbox = bounding_box(component, pad=4)
    paste_patch = component[component_bbox]
    starts = np.asarray([item.start for item in component_bbox], dtype=np.int64)
    center = starts + np.asarray(paste_patch.shape, dtype=np.int64) // 2
    source_voxels = int(np.count_nonzero(component))
    fixed_patch = extract_centered(
        component.astype(np.uint8),
        center,
        (48, 48, 48),
        pad_value=0,
    ).astype(bool)
    fixed_voxels = int(np.count_nonzero(fixed_patch))

    organ_start = np.asarray([item.start for item in organ_bbox], dtype=np.int64)
    center_in_organ_bbox = center - organ_start
    center_depth = float(organ_depth_bbox[tuple(center_in_organ_bbox)])
    config = audit_config()
    native_shape = adaptive_native_shape(
        paste_patch.shape,
        spacing,
        config,
        center_liver_depth_mm=center_depth,
    )
    local_organ = extract_centered(
        organ[organ_bbox].astype(np.uint8),
        center_in_organ_bbox,
        native_shape,
        pad_value=0,
    ).astype(bool)
    local_depth = extract_centered(
        organ_depth_bbox.astype(np.float32),
        center_in_organ_bbox,
        native_shape,
        pad_value=0.0,
    )
    footprint = center_crop_or_pad(
        paste_patch.astype(np.uint8),
        native_shape,
        pad_value=0,
    ).astype(bool)
    signed = signed_distance(footprint, spacing)
    tumor_normal, curvature = normal_and_curvature(signed, spacing)
    outside = ndi.distance_transform_edt(~footprint, sampling=spacing).astype(np.float32)
    zeros = np.zeros(native_shape, dtype=np.float32)
    fields = SimpleNamespace(
        footprint=footprint,
        organ=local_organ,
        outside_tumor_mm=outside,
        liver_depth_mm=local_depth,
        tumor_normal=tumor_normal,
        ct_norm=zeros,
        local_mean=zeros,
        local_std=zeros,
        gradient=zeros,
        tumor_sdf_mm=signed,
        tumor_sdf_norm=signed,
        liver_depth_norm=local_depth,
        tumor_curvature=curvature,
        liver_normal=tumor_normal,
        liver_curvature=curvature,
    )
    coordinates = adaptive_coordinate_sets(
        fields,
        config,
        np.random.default_rng(seed),
        spacing,
    )
    diagnostics = coordinate_diagnostics(fields, coordinates, config, spacing)
    nonzero = np.argwhere(component)
    extent = nonzero.max(axis=0) - nonzero.min(axis=0) + 1
    return {
        "component_id": int(component_id),
        "source_voxels": source_voxels,
        "physical_volume_mm3": float(source_voxels * np.prod(spacing)),
        "tumor_bbox_shape": [int(value) for value in extent],
        "paste_patch_shape_with_pad4": [int(value) for value in paste_patch.shape],
        "fixed_patch_size": 48,
        "fixed_retained_voxels": fixed_voxels,
        "fixed_crop_voxel_loss": int(source_voxels - fixed_voxels),
        "fixed_crop_retained_fraction": float(fixed_voxels / source_voxels),
        "v21_native_roi_shape": [int(value) for value in native_shape],
        "v21_graph_footprint_voxels": int(np.count_nonzero(footprint)),
        "v21_crop_voxel_loss": int(source_voxels - int(np.count_nonzero(footprint))),
        "v21_nodes": diagnostics.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", type=Path)
    parser.add_argument("--tumor-label", type=int, default=2)
    parser.add_argument("--liver-label", type=int, default=1)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    label, spacing = load_nifti(args.label)
    tumor = label == args.tumor_label
    organ = (label == args.liver_label) | tumor
    component_map, component_count = ndi.label(
        tumor,
        structure=ndi.generate_binary_structure(3, 1),
    )
    sizes = np.bincount(component_map.ravel())[1:]
    order = np.argsort(sizes)[::-1][: args.components] + 1

    organ_bbox = bounding_box(organ, pad=2)
    organ_crop = organ[organ_bbox]
    organ_depth = ndi.distance_transform_edt(organ_crop, sampling=spacing).astype(np.float32)
    reports = [
        audit_component(
            component_map == int(component_id),
            organ,
            organ_bbox,
            organ_depth,
            spacing,
            int(component_id),
            2100 + rank,
        )
        for rank, component_id in enumerate(order)
    ]
    result = {
        "schema": V21_SCHEMA_VERSION,
        "label_path": str(args.label),
        "volume_shape": [int(value) for value in label.shape],
        "spacing_mm": [float(value) for value in spacing],
        "tumor_component_count": int(component_count),
        "audited_components": reports,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
