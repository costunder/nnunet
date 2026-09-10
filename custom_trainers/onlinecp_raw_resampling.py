"""Candidate-conditioned raw CP followed by the native nnU-Net grid transform.

Only standard single-channel CTNormalization/DefaultPreprocessor and the
default cubic-data, linear-label, nearest-separate-axis contract are supported.
No donor is resampled once and reused as a translated preprocessed mask.

The full-grid cubic interpolation operator is linear before native clipping.
Its one-dimensional basis matrices retain the global spline-prefilter tails.
Only a raw patch delta is contracted into an output crop during training.
All persistent values are plain dictionaries, lists and NumPy arrays; the
``preparation`` subtree is transient and is never required by apply_candidate.
"""
from __future__ import annotations

from itertools import product
from typing import Mapping

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure, map_coordinates
from skimage.transform import resize


RAW_RESAMPLING_FORMAT = "onlinecp_raw_candidate_resampling_v1"
SUPPORT_DEFINITION = "newly_labelled_tumor_after_full_label_resampling_v1"


class RawResamplingError(ValueError):
    pass


def _integer_vector(value, *, name, length=3):
    array = np.asarray(value)
    if (array.shape != (length,) or not np.issubdtype(array.dtype, np.number)
            or not np.all(np.isfinite(array)) or not np.all(array == np.rint(array))):
        raise RawResamplingError(f"{name} must contain exactly {length} integer values")
    return array.astype(np.int64)


def _box(value, shape, *, name):
    array = np.asarray(value)
    if (array.shape != (3, 2) or not np.issubdtype(array.dtype, np.number)
            or not np.all(np.isfinite(array)) or not np.all(array == np.rint(array))):
        raise RawResamplingError(f"{name} must be three [lower, upper] integer pairs")
    array = array.astype(np.int64)
    if np.any(array[:, 0] < 0) or np.any(array[:, 1] > shape) or np.any(array[:, 1] <= array[:, 0]):
        raise RawResamplingError(f"{name} is outside its exact grid: {array.tolist()} versus {list(shape)}")
    return array


def _slices(box):
    return tuple(slice(int(lo), int(hi)) for lo, hi in box)


def _normalize(image, normalization):
    # Match CTNormalization's float32 in-place arithmetic, including the order
    # of clipping, subtraction and division. HU jitter happens before this.
    output = np.asarray(image, dtype=np.float32).copy()
    np.clip(output, normalization["lower"], normalization["upper"], out=output)
    output -= normalization["mean"]
    output /= max(normalization["std"], 1e-8)
    return output


def _axis_operator(length, new_length, *, order, separate=False):
    """Every column is a full-input-grid impulse, never a local ROI resize."""
    length, new_length = int(length), int(new_length)
    if min(length, new_length) <= 0:
        raise RawResamplingError("Interpolation grids must be nonempty")
    if separate:
        coordinates = (np.arange(new_length, dtype=np.float64) + .5) * (length / new_length) - .5
        selected = map_coordinates(np.arange(length, dtype=np.float64), coordinates[None],
                                   order=0, mode="nearest").astype(np.int64)
        output = np.zeros((new_length, length), dtype=np.float64)
        output[np.arange(new_length), selected] = 1.
        return output
    if length == new_length:
        return np.eye(length, dtype=np.float64)
    # Columns are independent full-grid impulses. The second axis retains its
    # integer grid, allowing native resize to build every column in one call.
    # DEBUG tests compare order-1 exactly and cubic to float64 precision against
    # the per-column native oracle, including the global spline tails.
    return resize(np.eye(length, dtype=np.float64), (new_length, length),
                  order=order, mode="edge", clip=False, anti_aliasing=False,
                  preserve_range=True)


def _patch_operator(operator, rows, start, length):
    """Full raw-patch columns, including inactive padding outside native crop."""
    lower, upper = max(0, int(start)), min(operator.shape[1], int(start + length))
    if upper <= lower:
        raise RawResamplingError("Raw patch has no intersection with the verified native grid")
    if lower == start and upper == start + length:
        return operator[int(rows[0]):int(rows[1]), lower:upper]
    result = np.zeros((int(rows[1] - rows[0]), int(length)), dtype=np.float64)
    result[:, lower - start:upper - start] = operator[int(rows[0]):int(rows[1]), lower:upper]
    return result


def _contract(value, matrices):
    """Separable dense contraction over all requested rows and patch columns."""
    output = np.asarray(value, dtype=np.float64)
    # This changes only the order of algebraically independent tensor axes.
    # It neither samples axes nor caps nodes/candidates/voxels.
    order = sorted(range(3), key=lambda axis: matrices[axis].shape[0] / max(1, output.shape[axis]))
    for axis in order:
        output = np.moveaxis(np.tensordot(matrices[axis], output, axes=(1, axis)), 0, axis)
    return output


def _unclipped_baseline(normalized, new_shape, separate_axis, data_operators):
    data = normalized[0].astype(np.float64)
    if tuple(data.shape) == tuple(new_shape):
        return data[None].copy()
    if separate_axis is None:
        return resize(data, tuple(new_shape), order=3, mode="edge", clip=False,
                      anti_aliasing=False, preserve_range=True)[None]
    # Native separate-z performs independent 2-D cubic interpolations and clips
    # each input plane BEFORE nearest-neighbor selection along its coarse axis.
    # Store selected *unclipped* planes; apply_candidate supplies changed bounds.
    shape_2d = tuple(int(new_shape[axis]) for axis in range(3) if axis != separate_axis)
    selected = np.argmax(data_operators[separate_axis], axis=1)
    output = np.empty(tuple(new_shape), dtype=np.float64)
    # Only planes selected by exact nearest interpolation contribute. Compute
    # each such plane once even when upsampling repeats it several times.
    for plane in np.unique(selected):
        resized = resize(np.take(data, int(plane), axis=separate_axis), shape_2d,
                         order=3, mode="edge", clip=False, anti_aliasing=False, preserve_range=True)
        for destination in np.flatnonzero(selected == plane):
            index = [slice(None)] * 3
            index[separate_axis] = int(destination)
            output[tuple(index)] = resized
    return output[None]


def _stats(image, axis):
    if axis is None:
        minimum, maximum = float(image.min()), float(image.max())
        return (np.array([minimum]), np.array([maximum]),
                np.array([np.count_nonzero(image == minimum)], dtype=np.int64),
                np.array([np.count_nonzero(image == maximum)], dtype=np.int64))
    dimensions = tuple(index for index in range(3) if index != axis)
    minimum, maximum = image.min(axis=dimensions), image.max(axis=dimensions)
    view_shape = [1, 1, 1]
    view_shape[axis] = image.shape[axis]
    return (minimum.astype(np.float64), maximum.astype(np.float64),
            (image == minimum.reshape(view_shape)).sum(axis=dimensions),
            (image == maximum.reshape(view_shape)).sum(axis=dimensions))


def prepare_case(raw_image_xyz, raw_label_xyz, properties, plans, *, configuration_name,
                 raw_spacing_xyz, raw_spatial_unit="mm"):
    """Prepare one case once; coordinates supplied to other APIs remain raw XYZ."""
    from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
    from nnunetv2.preprocessing.resampling.default_resampling import (
        compute_new_shape, determine_do_sep_z_and_axis, resample_data_or_seg_to_shape,
    )
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    raw_ct, raw_seg = np.asarray(raw_image_xyz), np.asarray(raw_label_xyz)
    if (raw_ct.ndim != 3 or raw_seg.shape != raw_ct.shape or not np.all(np.isfinite(raw_ct))
            or not np.all(np.isin(raw_seg, (0, 1, 2)))):
        raise RawResamplingError("Expected finite single-channel raw XYZ CT and matching labels 0/1/2")
    if raw_spatial_unit != "mm" or plans.get("image_reader_writer") not in {"SimpleITKIO", "NibabelIO"}:
        raise RawResamplingError("Only verified millimetre SimpleITKIO/NibabelIO geometry is supported")
    transpose = _integer_vector(plans.get("transpose_forward"), name="transpose_forward")
    if sorted(transpose.tolist()) != [0, 1, 2]:
        raise RawResamplingError("transpose_forward must be a permutation")
    spacing = np.asarray(raw_spacing_xyz, dtype=np.float64)
    reader_spacing = np.asarray(properties.get("spacing"), dtype=np.float64)
    if (spacing.shape != (3,) or reader_spacing.shape != (3,) or np.any(spacing <= 0)
            or not np.all(np.isfinite(spacing)) or not np.all(np.isfinite(reader_spacing))
            or not np.allclose(reader_spacing, spacing[::-1], rtol=0, atol=1e-5)):
        raise RawResamplingError("Raw spacing and the verified reader spacing disagree")
    manager = PlansManager(dict(plans)).get_configuration(configuration_name)
    configuration = manager.configuration
    if (configuration.get("preprocessor_name") != "DefaultPreprocessor"
            or configuration.get("normalization_schemes") != ["CTNormalization"]):
        raise RawResamplingError("Raw CP requires the standard one-channel CT DefaultPreprocessor")
    common_options = {"is_seg", "order", "order_z", "force_separate_z", "separate_z_anisotropy_threshold"}
    kwargs = {}
    for name, order, is_seg in (("data", 3, False), ("seg", 1, True)):
        supplied = dict(configuration.get(f"resampling_fn_{name}_kwargs", {}))
        if (configuration.get(f"resampling_fn_{name}") != "resample_data_or_seg_to_shape"
                or set(supplied) - common_options or supplied.get("is_seg") is not is_seg
                or supplied.get("order", 3) != order or supplied.get("order_z", 0) != 0):
            raise RawResamplingError(f"Unsupported {name} resampling contract; no kernel fallback")
        kwargs[name] = supplied
    original_spacing = reader_spacing[transpose]
    target_spacing = np.asarray(manager.spacing, dtype=np.float64)
    if target_spacing.shape != (3,) or np.any(target_spacing <= 0) or not np.all(np.isfinite(target_spacing)):
        raise RawResamplingError("Only native 3-D target spacing is supported")
    axes = []
    for name in ("data", "seg"):
        options = kwargs[name]
        separate, axis = determine_do_sep_z_and_axis(
            options.get("force_separate_z", False), original_spacing, target_spacing,
            options.get("separate_z_anisotropy_threshold", 3))
        axes.append(int(axis) if separate else None)
    if axes[0] != axes[1]:
        raise RawResamplingError("Data and label separate-axis contracts differ")
    axis = axes[0]
    reader_ct = np.asarray(raw_ct, dtype=np.float32).transpose(2, 1, 0).transpose(transpose)[None]
    reader_seg = raw_seg.transpose(2, 1, 0).transpose(transpose)[None].astype(np.int16, copy=True)
    if tuple(properties.get("shape_before_cropping", ())) != reader_ct.shape[1:]:
        raise RawResamplingError("Raw reader/transposed shape differs from preprocessing metadata")
    # Native no-seg cropping exposes its exact filled-nonzero mask (0 inside,
    # -1 outside), so this is computed once, not rescanned for 128 candidates.
    ct, native_nonzero_labels, crop = crop_to_nonzero(reader_ct, None)
    seg = reader_seg[(slice(None), *_slices(crop))].copy()
    seg[(seg == 0) & (native_nonzero_labels < 0)] = -1
    safe_interior = binary_erosion(native_nonzero_labels[0] == 0,
                                  structure=generate_binary_structure(3, 1), border_value=0)
    if (crop != properties.get("bbox_used_for_cropping")
            or tuple(properties.get("shape_after_cropping_and_before_resampling", ())) != ct.shape[1:]):
        raise RawResamplingError("Native nonzero crop differs from the verified preprocessing grid")
    props = plans.get("foreground_intensity_properties_per_channel", {}).get("0", {})
    required = ("mean", "std", "percentile_00_5", "percentile_99_5")
    if not all(key in props and np.isfinite(props[key]) for key in required):
        raise RawResamplingError("Missing finite CTNormalization intensity properties")
    normalization = dict(mean=float(props["mean"]), std=float(props["std"]),
                         lower=float(props["percentile_00_5"]), upper=float(props["percentile_99_5"]))
    if normalization["std"] < 0 or normalization["lower"] > normalization["upper"]:
        raise RawResamplingError("Invalid CT normalization bounds or standard deviation")
    normalized = _normalize(ct, normalization)
    new_shape = compute_new_shape(ct.shape[1:], original_spacing, target_spacing)
    data_operators = [_axis_operator(length, output, order=3, separate=index == axis)
                      for index, (length, output) in enumerate(zip(ct.shape[1:], new_shape))]
    seg_operators = [_axis_operator(length, output, order=1, separate=index == axis)
                     for index, (length, output) in enumerate(zip(ct.shape[1:], new_shape))]
    minimum, maximum, min_count, max_count = _stats(normalized[0], axis)
    baseline = _unclipped_baseline(normalized, new_shape, axis, data_operators)
    baseline_seg = resample_data_or_seg_to_shape(seg, new_shape, original_spacing, target_spacing, **kwargs["seg"])
    return {
        "metadata": {"format": RAW_RESAMPLING_FORMAT, "preprocessed_shape": new_shape.tolist(),
                     "cropped_shape": list(ct.shape[1:]), "raw_shape_xyz": list(raw_ct.shape),
                     "transpose_forward": transpose.tolist(), "crop_bbox": crop,
                     "configuration_name": str(configuration_name), "normalization": normalization,
                     "original_spacing": original_spacing.tolist(), "target_spacing": target_spacing.tolist(),
                     "resampling_kwargs": kwargs, "separate_z_axis": axis,
                     "support_definition": SUPPORT_DEFINITION,
                     "crop_stability_proof": "paste_mask_inside_native_filled_nonzero_erosion1_v1",
                     "baseline_dtype": "float64_unclipped", "global_spline_tails_retained": True},
        "baseline_unclipped": baseline, "baseline_seg": np.asarray(baseline_seg, dtype=np.int16),
        "data_operators": data_operators, "seg_operators": seg_operators,
        "clip_min": minimum, "clip_max": maximum,
        "preparation": {"raw_ct": np.asarray(ct, dtype=np.float32), "raw_seg": seg,
                        "raw_uncropped_ct": reader_ct, "safe_crop_interior": safe_interior,
                        "normalized_ct": normalized, "minimum_count": min_count, "maximum_count": max_count},
    }


def _clip_rows(case, crop, minimum, maximum):
    axis = case["metadata"]["separate_z_axis"]
    if axis is None:
        return float(minimum[0]), float(maximum[0])
    mapping = np.argmax(case["data_operators"][axis][crop[axis, 0]:crop[axis, 1]], axis=1)
    shape = [1, 1, 1]
    shape[axis] = mapping.size
    return minimum[mapping].reshape(shape), maximum[mapping].reshape(shape)


def baseline_output(case):
    """Native clipped baseline, useful for byte-contract validation by the bank."""
    shape = case["metadata"]["preprocessed_shape"]
    crop = np.array([[0, length] for length in shape], dtype=np.int64)
    low, high = _clip_rows(case, crop, case["clip_min"], case["clip_max"])
    return np.clip(case["baseline_unclipped"], low, high).astype(np.float32)


def _outside_stats(volume, origin, mask, minimum, maximum, min_count, max_count):
    slices = tuple(slice(int(start), int(start + length)) for start, length in zip(origin, mask.shape))
    removed = volume[slices][mask]
    remaining = int(volume.size - removed.size)
    if remaining == 0:
        return 0., 0., 0, False  # explicitly empty complement, never an output fallback
    keep_min = int(min_count) > int(np.count_nonzero(removed == minimum))
    keep_max = int(max_count) > int(np.count_nonzero(removed == maximum))
    if keep_min and keep_max:
        return float(minimum), float(maximum), remaining, False
    # Rare case: CP replaces every occurrence of a global/plane extreme. Scan
    # the exact complement once at bank preparation, NOT each training event.
    pieces = [volume[slices][~mask]]
    for axis in range(volume.ndim):
        prefix = [slice(int(origin[index]), int(origin[index] + mask.shape[index]))
                  if index < axis else slice(None) for index in range(volume.ndim)]
        for start, stop in ((0, int(origin[axis])), (int(origin[axis] + mask.shape[axis]), volume.shape[axis])):
            if stop > start:
                index = list(prefix)
                index[axis] = slice(start, stop)
                pieces.append(volume[tuple(index)])
    low = min(float(piece.min()) for piece in pieces if piece.size)
    high = max(float(piece.max()) for piece in pieces if piece.size)
    return low, high, remaining, True


def _linear_label_grid(labels, matrices, values):
    """Native multilinear stencil ordering, with the complete label mixture.

    Basis rows have at most two nonzero entries (nearest: one). Every source
    label in that stencil participates, including other tumors. No source-only
    mask threshold is used as a substitute for label resampling.
    """
    indices, weights = [], []
    for matrix in matrices:
        if np.any(np.count_nonzero(matrix, axis=1) > 2):
            raise RawResamplingError("Label operator is not the supported linear/nearest kernel")
        first = np.argmax(matrix != 0, axis=1)
        second = np.maximum(first, matrix.shape[1] - 1 - np.argmax((matrix != 0)[:, ::-1], axis=1))
        indices.append((first, second))
        first_weights = matrix[np.arange(matrix.shape[0]), first]
        second_weights = matrix[np.arange(matrix.shape[0]), second].copy()
        second_weights[second == first] = 0.
        weights.append((first_weights, second_weights))
    shape = tuple(matrix.shape[0] for matrix in matrices)
    result = np.zeros(shape, dtype=np.int16)
    for label in values:
        probability = np.zeros(shape, dtype=np.float64)
        for corner in product((0, 1), repeat=3):
            coordinates = np.ix_(*(indices[axis][side] for axis, side in enumerate(corner)))
            contribution = (labels[coordinates] == label).astype(np.float64)
            for axis, side in enumerate(corner):
                view = [1, 1, 1]
                view[axis] = shape[axis]
                contribution *= weights[axis][side].reshape(view)
            probability += contribution
        result[probability >= .5] = label
    return result


def prepare_candidate(case, source_ct_xyz, source_mask_xyz, source_anchor_xyz, target_center_xyz):
    """Store the actual target-phase CT delta inputs and complete label result."""
    metadata = case["metadata"]
    if metadata.get("format") != RAW_RESAMPLING_FORMAT or "preparation" not in case:
        raise RawResamplingError("Candidate preparation requires the verified case preparation subtree")
    source_ct, source_mask = np.asarray(source_ct_xyz), np.asarray(source_mask_xyz)
    if (source_ct.ndim != 3 or source_ct.shape != source_mask.shape or not np.all(np.isfinite(source_ct))
            or not np.all((source_mask == 0) | (source_mask == 1)) or not np.any(source_mask)):
        raise RawResamplingError("Expected a nonempty raw source mask and its actual finite CT patch")
    anchor = _integer_vector(source_anchor_xyz, name="source_anchor_xyz")
    target = _integer_vector(target_center_xyz, name="target_center_xyz")
    if np.any(anchor < 0) or np.any(anchor >= source_mask.shape):
        raise RawResamplingError("Source anchor is outside the raw source patch")
    transpose = metadata["transpose_forward"]
    anchor = anchor[::-1][transpose]
    mapped_target = target[::-1][transpose] - np.asarray(metadata["crop_bbox"])[:, 0]
    origin = mapped_target - anchor
    ct = source_ct.transpose(2, 1, 0).transpose(transpose).astype(np.float32, copy=True)[None]
    mask = source_mask.transpose(2, 1, 0).transpose(transpose).astype(bool, copy=True)
    shape = np.asarray(metadata["cropped_shape"])
    raw_origin = origin + np.asarray(metadata["crop_bbox"])[:, 0]
    raw_shape = np.asarray(metadata["raw_shape_xyz"])[::-1][transpose]
    if np.any(raw_origin < 0) or np.any(raw_origin + mask.shape > raw_shape):
        raise RawResamplingError("Raw target patch escapes the actual source-volume grid")
    coordinates = np.argwhere(mask)
    placed = coordinates + origin
    if (np.any(placed < 0) or np.any(placed >= shape)
            or not np.all(case["preparation"]["safe_crop_interior"][tuple(placed.T)])):
        raise RawResamplingError("Fixed-grid representation is unproved: actual pasted mask reaches the native "
                                 "filled-nonzero boundary/outside. This does not prove raw CP impossible; "
                                 "no candidate is discarded or substituted")
    raw_input_box = np.stack((raw_origin, raw_origin + mask.shape), axis=1)
    recipient = case["preparation"]["raw_uncropped_ct"][(slice(None), *_slices(raw_input_box))].copy()
    recipient_labels = case["preparation"]["raw_seg"][0][tuple(placed.T)]
    # Candidate geometry applies the original minimum liver coverage (0.85).
    # The remaining background is deliberately pasted too, exactly as in the
    # raw baseline. Requiring 100% liver here would silently tighten CP rules.
    if np.any(recipient_labels == 2):
        raise RawResamplingError("Raw CP support overlaps an existing recipient tumor")
    positive_lower = origin + coordinates.min(axis=0)
    positive_upper = origin + coordinates.max(axis=0) + 1
    output_rows = [np.flatnonzero(np.any(operator[:, lo:hi] != 0, axis=1))
                   for operator, lo, hi in zip(case["seg_operators"], positive_lower, positive_upper)]
    zero_transport = any(rows.size == 0 for rows in output_rows)
    if zero_transport:
        # Exact nearest/linear sampling may transport no support. The anchor
        # observation is still a real baseline patch, not a fabricated CP mask.
        center = np.rint((mapped_target + .5) * np.asarray(metadata["preprocessed_shape"]) / shape - .5).astype(int)
        center = np.clip(center, 0, np.asarray(metadata["preprocessed_shape"]) - 1)
        output_box = np.stack((center, center + 1), axis=1)
        seg_patch = case["baseline_seg"][(slice(None), *_slices(output_box))].copy()
    else:
        output_box = np.array([[int(rows.min()), int(rows.max()) + 1] for rows in output_rows])
        row_operators = [operator[lo:hi] for operator, (lo, hi) in zip(case["seg_operators"], output_box)]
        needed = [np.flatnonzero(np.any(operator != 0, axis=0)) for operator in row_operators]
        labels = case["preparation"]["raw_seg"][0][np.ix_(*needed)].copy()
        # The sparse label stencils include ALL contributing recipient labels,
        # not just the shifted source. Fill CP positions that are in this grid.
        local_coordinates = [np.searchsorted(needed[axis], origin[axis] + coordinates[:, axis]) for axis in range(3)]
        valid = np.ones(coordinates.shape[0], dtype=bool)
        for axis in range(3):
            ids = local_coordinates[axis]
            in_range = ids < needed[axis].size
            valid &= in_range
            valid[in_range] &= needed[axis][ids[in_range]] == origin[axis] + coordinates[in_range, axis]
        labels[tuple(ids[valid] for ids in local_coordinates)] = 2
        matrices = [operator[:, ids] for operator, ids in zip(row_operators, needed)]
        seg_patch = _linear_label_grid(labels, matrices, (-1, 0, 1, 2))[None]
    baseline_labels = case["baseline_seg"][(slice(None), *_slices(output_box))]
    support = (seg_patch[0] == 2) & (baseline_labels[0] != 2)
    normalized = case["preparation"]["normalized_ct"][0]
    axis = metadata["separate_z_axis"]
    stats_origin = np.maximum(origin, 0)
    stats_upper = np.minimum(origin + mask.shape, shape)
    stats_patch_box = np.stack((stats_origin - origin, stats_upper - origin), axis=1)
    stats_mask = mask[_slices(stats_patch_box)]
    outside_min, outside_max, outside_count, scanned = [], [], [], []
    if axis is None:
        stats = _outside_stats(normalized, stats_origin, stats_mask, case["clip_min"][0], case["clip_max"][0],
                               case["preparation"]["minimum_count"][0], case["preparation"]["maximum_count"][0])
        outside_min.append(stats[0]); outside_max.append(stats[1]); outside_count.append(stats[2]); scanned.append(stats[3])
    else:
        for local in range(mask.shape[axis]):
            plane = int(origin[axis]) + local
            if plane < 0 or plane >= shape[axis]:
                # Inactive pad only: the actual mask was proved inside above.
                outside_min.append(0.); outside_max.append(0.)
                outside_count.append(0); scanned.append(False)
                continue
            stats = _outside_stats(np.take(normalized, plane, axis=axis), np.delete(stats_origin, axis),
                                   np.take(stats_mask, plane - stats_origin[axis], axis=axis),
                                   case["clip_min"][plane], case["clip_max"][plane],
                                   case["preparation"]["minimum_count"][plane], case["preparation"]["maximum_count"][plane])
            outside_min.append(stats[0]); outside_max.append(stats[1]); outside_count.append(stats[2]); scanned.append(stats[3])
    return {"format": RAW_RESAMPLING_FORMAT, "source_ct": ct, "source_mask": mask,
            "recipient_ct": recipient, "target_input_origin": origin.astype(np.int64),
            "target_center_xyz": target.astype(np.int64), "output_bbox": output_box.tolist(),
            "seg_patch": seg_patch.astype(np.int16), "pasted_support": support,
            "outside_min": np.asarray(outside_min, dtype=np.float64),
            "outside_max": np.asarray(outside_max, dtype=np.float64),
            "outside_count": np.asarray(outside_count, dtype=np.int64),
            "audit": {"support_definition": SUPPORT_DEFINITION, "newly_labelled_voxels": int(support.sum()),
                      "raw_pasted_voxels": int(mask.sum()), "zero_label_transport": bool(zero_transport),
                      "raw_liver_coverage": float(np.mean(recipient_labels == 1)),
                      "fixed_crop_and_nonzero_mask_preserved": True,
                      "extrema_complement_scans": int(sum(scanned)), "source_position_mask_reused": False}}


def apply_candidate(case, candidate, crop_bbox, *, scale, shift_hu):
    """Return native-preprocessed raw CP in an INSIDE-case output crop only."""
    metadata = case["metadata"]
    if metadata.get("format") != RAW_RESAMPLING_FORMAT or candidate.get("format") != RAW_RESAMPLING_FORMAT:
        raise RawResamplingError("Incompatible raw resampling payload")
    if not np.isfinite(scale) or not np.isfinite(shift_hu):
        raise RawResamplingError("HU appearance parameters must be finite")
    crop = _box(crop_bbox, np.asarray(metadata["preprocessed_shape"]), name="output crop")
    mask = candidate["source_mask"]
    source = np.asarray(candidate["source_ct"], dtype=np.float32)
    jittered = source * float(scale) + float(shift_hu)
    if not np.all(np.isfinite(jittered)):
        raise RawResamplingError("Raw HU jitter overflowed; no intensity fallback")
    replacement = _normalize(jittered, metadata["normalization"])[0]
    previous = _normalize(candidate["recipient_ct"], metadata["normalization"])[0]
    delta = replacement.astype(np.float64) - previous.astype(np.float64)
    delta[~mask] = 0.
    origin = candidate["target_input_origin"]
    matrices = [_patch_operator(operator, rows, int(start), int(length)) for operator, rows, start, length
                in zip(case["data_operators"], crop, origin, mask.shape)]
    transported = _contract(delta, matrices)
    output = case["baseline_unclipped"][(slice(None), *_slices(crop))].copy()
    output[0] += transported
    minimum, maximum = np.asarray(case["clip_min"]).copy(), np.asarray(case["clip_max"]).copy()
    axis = metadata["separate_z_axis"]
    positions = (0,) if axis is None else range(mask.shape[axis])
    for local in positions:
        plane_mask = mask if axis is None else np.take(mask, local, axis=axis)
        if not np.any(plane_mask):
            continue
        plane_values = replacement if axis is None else np.take(replacement, local, axis=axis)
        new_low, new_high = float(plane_values[plane_mask].min()), float(plane_values[plane_mask].max())
        if candidate["outside_count"][local] > 0:
            new_low = min(new_low, float(candidate["outside_min"][local]))
            new_high = max(new_high, float(candidate["outside_max"][local]))
        destination = 0 if axis is None else int(origin[axis]) + local
        minimum[destination], maximum[destination] = new_low, new_high
    low, high = _clip_rows(case, crop, minimum, maximum)
    np.clip(output, low, high, out=output)
    segmentation = case["baseline_seg"][(slice(None), *_slices(crop))].copy()
    support = np.zeros(tuple(crop[:, 1] - crop[:, 0]), dtype=bool)
    candidate_box = np.asarray(candidate["output_bbox"])
    intersection = np.stack((np.maximum(crop[:, 0], candidate_box[:, 0]),
                             np.minimum(crop[:, 1], candidate_box[:, 1])), axis=1)
    if np.all(intersection[:, 1] > intersection[:, 0]):
        src = _slices(intersection - candidate_box[:, 0, None])
        dst = _slices(intersection - crop[:, 0, None])
        segmentation[(0, *dst)] = candidate["seg_patch"][(0, *src)]
        support[dst] = candidate["pasted_support"][src]
    return {"data": output.astype(np.float32), "seg": segmentation.astype(np.int16),
            "pasted_support": support, "audit": {**candidate["audit"],
                "crop_newly_labelled_voxels": int(support.sum()), "full_volume_resampling_at_runtime": False,
                "global_spline_tails_retained": True}}


def estimate_runtime_bytes(raw_shape, preprocessed_shape, source_patch_shape, output_crop_shape):
    """Array accounting only; raw_shape is the cropped, transposed input grid.

    This excludes the caller's uncropped arrays and native preparation spline
    buffers. It is not a measured RSS or total-process memory guarantee.
    """
    raw, pre = np.asarray(raw_shape, dtype=np.int64), np.asarray(preprocessed_shape, dtype=np.int64)
    patch, crop = np.asarray(source_patch_shape, dtype=np.int64), np.asarray(output_crop_shape, dtype=np.int64)
    return {"case_persistent_array_bytes": int(10 * np.prod(pre) + 16 * np.sum(raw * pre)),
            "case_preparation_raw_and_normalized_bytes": int(11 * np.prod(raw)),
            "candidate_raw_array_bytes": int(9 * np.prod(patch)),
            "candidate_label_and_support_bytes_upper_bound": int(3 * np.prod(pre)),
            "event_output_and_delta_bytes": int(19 * np.prod(crop) + 28 * np.prod(patch)),
            "event_array_bytes_conservative": int(40 * np.prod(crop) + 56 * np.prod(patch)
                                                    + 32 * max(np.prod(crop), np.prod(patch))),
            "no_candidate_or_resolution_reduction": True,
            "limitation": "Excludes native preparation spline buffers, BLAS workspace, Python and concurrent workers"}
