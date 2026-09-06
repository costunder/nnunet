"""Header-only, extent-aware donor grid validation; never resample input data.

Numerical equivalence is defined by displacement over the entire voxel-cell
extent: at most 0.1 micrometre and 0.0001 voxel in both grids. Float storage
step counts are diagnostic only: their physical size changes with coordinate
magnitude. This is not registration or proof of anatomical array alignment.
"""
from __future__ import annotations

import itertools

import numpy as np


GEOMETRY_POLICY_VERSION = "donor_grid_physical_extent_v2"
LEGACY_AFFINE_ATOL = 1e-5
MAX_CORNER_MM = 1e-4
MAX_CORNER_VOXELS = 1e-4


def _float32_ulp_distance(first, second):
    """Return stored float32 step distances, or None for non-float32 values."""
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        a, b = first.astype(np.float32), second.astype(np.float32)
    if (not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
            or not np.array_equal(a.astype(np.float64), first)
            or not np.array_equal(b.astype(np.float64), second)):
        return None
    # Collapse signed zero before mapping IEEE-754 encodings into numeric order.
    def ordered(values):
        values = np.where(values == 0, np.float32(0), values).astype(np.float32)
        bits = values.view(np.uint32).astype(np.int64)
        return np.where(bits & 0x80000000, 0x100000000 - bits, bits + 0x80000000)
    return np.abs(ordered(a) - ordered(b))


def _forms(header):
    result = {"qform_code": int(header["qform_code"]), "sform_code": int(header["sform_code"])}
    result["selected_form"] = ("sform" if result["sform_code"] else
                               "qform" if result["qform_code"] else "base")
    for name in ("qform", "sform"):
        if not result[name + "_code"]:
            result[name + "_affine"] = None
            result[name + "_status"] = "unset"
            continue
        try:
            affine = np.asarray(getattr(header, "get_" + name)(), dtype=np.float64)
        except (ValueError, np.linalg.LinAlgError) as exc:
            result[name + "_affine"] = None
            result[name + "_status"] = f"invalid: {type(exc).__name__}: {exc}"
        else:
            if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
                result[name + "_affine"] = None
                result[name + "_status"] = "invalid: nonfinite or malformed alternate transform"
            else:
                result[name + "_affine"] = affine.tolist()
                result[name + "_status"] = "available"
    return result


def validate_donor_geometry(image_nii, label_nii, *, case_id):
    """Validate selected grids without loading, changing or canonicalizing data.

    All eight outer voxel-cell corners are tested. For affine maps, the norm
    of their displacement is convex, so its maximum over the box is attained
    at a corner. Both inverse grid bases are checked to avoid one-sided scale
    tolerance. Unknown spatial units retain only the legacy strict path.
    """
    def fail(reason):
        raise ValueError(f"Donor image/label affine mismatch: {case_id}: {reason}")

    def spatial_shape(nii):
        shape = tuple(nii.shape)
        if len(shape) == 4 and shape[3] == 1:
            shape = shape[:3]
        if len(shape) != 3 or any(int(size) != size or size <= 0 for size in shape):
            fail(f"invalid spatial dimensions {tuple(nii.shape)}")
        return tuple(int(size) for size in shape)

    image_shape, label_shape = spatial_shape(image_nii), spatial_shape(label_nii)
    if image_shape != label_shape:
        fail(f"spatial dimensions differ: {image_shape} versus {label_shape}")
    affines, inverses, units, pixdims = [], [], [], []
    for name, nii in (("image", image_nii), ("label", label_nii)):
        affine = np.asarray(nii.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
            fail(f"{name} selected affine is nonfinite or malformed")
        if not np.array_equal(affine[3], [0, 0, 0, 1]):
            fail(f"{name} selected affine has an invalid homogeneous row")
        try:
            inverse = np.linalg.inv(affine[:3, :3])
        except np.linalg.LinAlgError:
            fail(f"{name} selected affine is singular")
        if not np.all(np.isfinite(inverse)):
            fail(f"{name} selected affine inverse is nonfinite")
        try:
            unit = nii.header.get_xyzt_units()[0]
        except (KeyError, ValueError) as exc:
            fail(f"{name} spatial unit is invalid: {exc}")
        pixdim = np.asarray(nii.header["pixdim"][1:4], dtype=np.float64)
        if pixdim.shape != (3,) or not np.all(np.isfinite(pixdim)) or np.any(pixdim <= 0):
            fail(f"{name} spatial pixdim is nonfinite or nonpositive")
        affines.append(affine)
        inverses.append(inverse)
        units.append(unit)
        pixdims.append(pixdim)
    if units[0] != units[1]:
        fail(f"spatial units differ: {units[0]} versus {units[1]}")
    corners = np.asarray([(*corner, 1.0) for corner in itertools.product(
        *[(-0.5, size - 0.5) for size in image_shape])], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        displacement = corners @ (affines[1] - affines[0])[:3].T
        native_max = float(np.max(np.linalg.norm(displacement, axis=1)))
        voxel_max = [float(np.max(np.linalg.norm(displacement @ inverse.T, axis=1)))
                     for inverse in inverses]
    if not np.all(np.isfinite([native_max, *voxel_max])):
        fail("corner displacement is nonfinite")
    unit_to_mm = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(units[0])
    physical_max = None if unit_to_mm is None else native_max * unit_to_mm
    if physical_max is not None and not np.isfinite(physical_max):
        fail("physical corner displacement is nonfinite")
    bound_details = (f"max_cell_corner_mm={physical_max}, "
                     f"max_image_voxels={voxel_max[0]:.12g}, max_label_voxels={voxel_max[1]:.12g}; "
                     f"limits={MAX_CORNER_MM} mm and {MAX_CORNER_VOXELS} voxel")
    if max(voxel_max) > MAX_CORNER_VOXELS or (physical_max is not None and physical_max > MAX_CORNER_MM):
        fail(bound_details)
    strict = bool(np.allclose(affines[0], affines[1], rtol=0, atol=LEGACY_AFFINE_ATOL))
    forms = [_forms(nii.header) for nii in (image_nii, label_nii)]
    affine_ulps = _float32_ulp_distance(affines[0], affines[1])
    pixdim_ulps = _float32_ulp_distance(pixdims[0], pixdims[1])
    max_affine_ulps = None if affine_ulps is None else int(affine_ulps.max())
    max_pixdim_ulps = None if pixdim_ulps is None else int(pixdim_ulps.max())
    effective_codes = [row.get(row["selected_form"] + "_code", 0) for row in forms]
    # Storage format, form name and ULP counts do not determine grid error.
    # For the additional numerical path, establish comparable coded mm frames
    # as well as both bounds. Unknown units keep the historical strict path.
    if not strict:
        if units != ["mm", "mm"]:
            fail(f"additional numerical equivalence requires matching mm units; {bound_details}")
        if effective_codes[0] <= 0 or effective_codes[0] != effective_codes[1]:
            fail(f"additional numerical equivalence requires compatible coded coordinate frames: {effective_codes}; {bound_details}")
    return {
        "policy_version": GEOMETRY_POLICY_VERSION,
        "accepted_as": "strict" if strict else "extent_equivalent",
        "acceptance_basis": "whole_cell_physical_and_reciprocal_voxel_bounds" if physical_max is not None
                            else "unknown_units_legacy_strict_and_reciprocal_voxel_bounds",
        "float32_ulps_used_for_acceptance": False,
        "effective_coordinate_frame_codes": effective_codes,
        "legacy_strict_pass": strict,
        "spatial_shape": list(image_shape),
        "image_selected_affine": affines[0].tolist(),
        "label_selected_affine": affines[1].tolist(),
        "image_forms": forms[0], "label_forms": forms[1],
        "image_spatial_unit": units[0], "label_spatial_unit": units[1],
        "image_spatial_pixdim": pixdims[0].tolist(), "label_spatial_pixdim": pixdims[1].tolist(),
        "corner_extent": "outer_voxel_cell_corners_minus_half_to_shape_minus_half",
        "max_corner_displacement_native_units": native_max,
        "max_corner_displacement_mm": physical_max,
        "max_corner_displacement_image_voxels": voxel_max[0],
        "max_corner_displacement_label_voxels": voxel_max[1],
        "max_affine_float32_ulps": max_affine_ulps,
        "max_pixdim_float32_ulps": max_pixdim_ulps,
        "thresholds": {"legacy_affine_atol": LEGACY_AFFINE_ATOL, "legacy_affine_rtol": 0,
                       "max_corner_mm": MAX_CORNER_MM, "max_corner_voxels": MAX_CORNER_VOXELS},
        "data_resampled": False,
    }
