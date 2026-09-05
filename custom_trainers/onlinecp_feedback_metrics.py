"""Actual, pre-update nnU-Net feedback on surviving pasted lesions.

The normal nnU-Net image/label transform is authoritative and is never changed.
An auxiliary pasted-support channel shares its spatial transform and every deep
supervision downsampling. Its intersection with the transformed tumor label is
the attributed lesion; excluded voxels are reported, not silently relabelled.
This is not a claim that interpolation preserves original voxel identity.

Metrics are batched, detached observations, not a replacement segmentation loss.
No-event, erased, boundary-contact and empty-neighborhood observations are NaN
with an explicit status; callers must exclude and count them, never learn zero.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


METRIC_DEFINITION_ID = "onlinecp_surviving_lesion_feedback_v1"
ATTRIBUTION_DEFINITION_ID = "onlinecp_shared_transform_surviving_support_v1"
STATUS_AVAILABLE = 0
STATUS_NO_EVENT = 1
STATUS_EMPTY_AFTER_AUGMENTATION = 2
STATUS_BOUNDARY_CONTACT = 3
STATUS_EMPTY_ADJACENT = 4
STATUS_REPORTED_TRUNCATED = 5
STATUS_NAMES = {
    STATUS_AVAILABLE: "available",
    STATUS_NO_EVENT: "no_cp_event",
    STATUS_EMPTY_AFTER_AUGMENTATION: "empty_after_augmentation",
    STATUS_BOUNDARY_CONTACT: "boundary_contact_truncation_unverified",
    STATUS_EMPTY_ADJACENT: "no_valid_adjacent_nontumor",
    STATUS_REPORTED_TRUNCATED: "reported_truncation_risk",
}
METRIC_DEFINITION = {
    "id": METRIC_DEFINITION_ID,
    "labels": {"background": 0, "liver": 1, "tumor": 2},
    "timing": "actual_training_forward_pre_optimizer_update",
    "foreground_ce": "1-exp(-mean_surviving_tumor_negative_log_probability)",
    "foreground_ce_raw": "mean_surviving_tumor_negative_log_probability",
    "foreground_error": "mean_surviving_tumor_one_minus_tumor_probability",
    "boundary_error": "mean_inner_surviving_mask_boundary_one_minus_tumor_probability",
    "adjacent_fp": "mean_valid_adjacent_nontumor_tumor_probability",
    "neighborhood": "3D Chebyshev voxel neighborhoods; explicit width arguments",
    "normalization": "each component divided by its own nonempty mask voxel count",
    "unavailable": "NaN with explicit status, never a zero training observation",
    "attribution": ATTRIBUTION_DEFINITION_ID,
}


class FeedbackMetricError(ValueError):
    """An unsupported or internally inconsistent actual-feedback input."""


def _tensor_shape(value: torch.Tensor, shape: tuple[int, ...], name: str,
                  device: torch.device) -> None:
    if not torch.is_tensor(value) or tuple(value.shape) != shape or value.device != device:
        raise FeedbackMetricError(f"{name} must have shape {shape} on device {device}")


def _is_binary(value: torch.Tensor) -> torch.Tensor:
    return ((value == 0) | (value == 1)).all()


def _reject_invalid(checks: dict[str, torch.Tensor]) -> None:
    # One necessary batched validity synchronization, not a per-sample GPU loop.
    failed = torch.stack(tuple(checks.values()))
    if bool(failed.any()):
        names = [name for name, bad in zip(checks, failed.detach().cpu().tolist()) if bad]
        raise FeedbackMetricError("Invalid feedback input: " + "; ".join(names))


def _border_contact(mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # Padding is explicitly invalid; a complete 3x3x3 valid neighborhood is
    # needed to certify the inner boundary. Contact is conservative exclusion,
    # not proof that original lesion voxels were actually lost.
    interior = _erode(valid, 1)
    return (mask & ~interior).flatten(1).any(1)


def _erode(mask: torch.Tensor, width: int) -> torch.Tensor:
    # Explicit invalid exterior also handles a spatial dimension of length one;
    # avg_pool3d rejects inputs smaller than its kernel even with padding.
    complement = F.pad((~mask).float(), (width,) * 6, value=1)
    return ~F.max_pool3d(complement, 2 * width + 1, stride=1).bool()


@torch.no_grad()
def compute_feedback_metrics(
    logits: torch.Tensor, labels: torch.Tensor, pasted_mask: torch.Tensor, *,
    valid_mask: torch.Tensor, event_applied: torch.Tensor,
    mask_truncated: torch.Tensor | None = None,
    boundary_width: int = 1, adjacent_width: int = 1,
) -> dict[str, torch.Tensor]:
    """Return one observation per batch item, on the logits' device.

    Required shapes are B,3,D,H,W for logits; B,1,D,H,W for labels/masks;
    and B for event flags. This contract supports one pasted event per item,
    class labels 0/1/2, and real 3-D tensors only. ``pasted_mask`` is the wrapper's
    surviving attribution, not all tumor voxels or a whole-patch loss mask.
    Widths are explicit voxel radii and do not change the training graph/data.
    Global input/numerical-integrity checks surround vectorized GPU operations.
    """
    if (not torch.is_tensor(logits) or logits.ndim != 5 or logits.shape[1] != 3
            or logits.shape[0] < 1 or min(logits.shape[2:]) < 1
            or not logits.is_floating_point()):
        raise FeedbackMetricError("Only nonempty B,3,D,H,W class logits are supported; no 2D, regions or cascade")
    if (type(boundary_width) is not int or boundary_width < 1
            or type(adjacent_width) is not int or adjacent_width < 1):
        raise FeedbackMetricError("Boundary and adjacent voxel widths must be positive integers")
    batch_shape = (logits.shape[0],)
    mask_shape = (logits.shape[0], 1, *logits.shape[2:])
    for name, tensor in (("labels", labels), ("pasted_mask", pasted_mask),
                         ("valid_mask", valid_mask)):
        _tensor_shape(tensor, mask_shape, name, logits.device)
    _tensor_shape(event_applied, batch_shape, "event_applied", logits.device)
    if mask_truncated is None:
        mask_truncated = torch.zeros(batch_shape, dtype=torch.bool, device=logits.device)
    _tensor_shape(mask_truncated, batch_shape, "mask_truncated", logits.device)
    values = logits.detach().float()
    mask, valid, applied = pasted_mask.bool(), valid_mask.bool(), event_applied.bool()
    _reject_invalid({
        "nonfinite logits": ~torch.isfinite(values).all(),
        "nonbinary pasted mask": ~_is_binary(pasted_mask),
        "nonbinary validity mask": ~_is_binary(valid_mask),
        "nonbinary event flag": ~_is_binary(event_applied),
        "nonbinary truncation flag": ~_is_binary(mask_truncated),
        "unsupported/nonfinite valid labels": (valid & ~((labels == 0) | (labels == 1) | (labels == 2))).any(),
        "pasted attribution outside valid tumor target": (mask & (~valid | (labels != 2))).any(),
        "pasted mask without an applied event": (mask.flatten(1).any(1) & ~applied).any(),
        "truncation flag without an applied event": (mask_truncated.bool() & ~applied).any(),
    })

    log_probability = F.log_softmax(values, dim=1)[:, 2:3]
    _reject_invalid({"nonfinite tumor log probability": ~torch.isfinite(log_probability).all()})
    tumor_probability = log_probability.exp()
    eroded = _erode(mask, boundary_width)
    boundary = mask & ~eroded
    dilated = F.max_pool3d(mask.float(), 2 * adjacent_width + 1, stride=1,
                           padding=adjacent_width).bool()
    adjacent = dilated & ~mask & valid & (labels != 2)
    foreground_count = mask.flatten(1).sum(1)
    boundary_count = boundary.flatten(1).sum(1)
    adjacent_count = adjacent.flatten(1).sum(1)
    touches_border = _border_contact(mask, valid)

    status = torch.full(batch_shape, STATUS_AVAILABLE, dtype=torch.int64, device=logits.device)
    status = torch.where(adjacent_count == 0, STATUS_EMPTY_ADJACENT, status)
    status = torch.where(touches_border, STATUS_BOUNDARY_CONTACT, status)
    status = torch.where(mask_truncated.bool(), STATUS_REPORTED_TRUNCATED, status)
    status = torch.where(foreground_count == 0, STATUS_EMPTY_AFTER_AUGMENTATION, status)
    status = torch.where(~applied, STATUS_NO_EVENT, status)
    available = status == STATUS_AVAILABLE

    def masked_mean(tensor: torch.Tensor, region: torch.Tensor,
                    count: torch.Tensor) -> torch.Tensor:
        # Zero denominators remain NaN, and every excluded event is explicitly
        # unavailable. No empty-mask score is replaced by zero or epsilon.
        denominator = torch.where(count > 0, count.float(), float("nan"))
        normalized = torch.where(region, tensor, 0) / denominator[:, None, None, None, None]
        measured = normalized.flatten(1).sum(1)
        return torch.where(available, measured, float("nan"))

    raw_ce = masked_mean(-log_probability, mask, foreground_count)
    return {
        "foreground_ce": -torch.expm1(-raw_ce),
        "foreground_ce_raw": raw_ce,
        # These are analytically means of probabilities in [0,1]. Restore that
        # range after floating-point reduction rounding, not after an invalid
        # input or failed metric (NaN unavailable observations remain NaN).
        "foreground_error": masked_mean(1 - tumor_probability, mask, foreground_count).clamp(0, 1),
        "boundary_error": masked_mean(1 - tumor_probability, boundary, boundary_count).clamp(0, 1),
        "adjacent_fp": masked_mean(tumor_probability, adjacent, adjacent_count).clamp(0, 1),
        "available": available, "status": status,
        "foreground_voxels": foreground_count,
        "boundary_voxels": boundary_count, "adjacent_voxels": adjacent_count,
        "touches_boundary": touches_border,
    }


def _validate_transform_tree(transform: Any) -> None:
    supported = {
        "ComposeTransforms", "RandomTransform", "SpatialTransform", "MirrorTransform",
        "GaussianNoiseTransform", "GaussianBlurTransform", "MultiplicativeBrightnessTransform",
        "ContrastTransform", "SimulateLowResolutionTransform", "GammaTransform",
        "MaskImageTransform", "RemoveLabelTransform", "DownsampleSegForDSTransform",
    }
    pending = [transform]
    while pending:
        current = pending.pop()
        name = type(current).__name__
        if (name not in supported
                or not type(current).__module__.startswith("batchgeneratorsv2.transforms.")):
            raise FeedbackMetricError(f"Unsupported feedback augmentation transform: {type(current).__module__}.{name}")
        if name == "SpatialTransform":
            if (len(current.patch_size) != 3 or current.padding_mode_image != "zeros"
                    or current.border_mode_seg != "zeros"
                    or current.mode_image != "bilinear"
                    or current.mode_seg not in ("bilinear", "nearest")):
                raise FeedbackMetricError("Feedback requires 3-D shared-grid bilinear image/zero-padding spatial transforms")
        if name == "RemoveLabelTransform" and (current.label_value != -1 or current.set_to != 0):
            raise FeedbackMetricError("Only nnU-Net padding label removal (-1 to 0) is supported")
        pending.extend(getattr(current, "transforms", ()))
        nested = getattr(current, "transform", None)
        if nested is not None:
            pending.append(nested)


@torch.no_grad()
def transform_with_feedback(
    base_transform: Any, image: torch.Tensor, segmentation: torch.Tensor,
    pasted_mask: torch.Tensor, *, is_cascaded: bool = False, regions: Any = None,
    do_dummy_2d_data_aug: bool = False, ignore_label: int | None = None,
) -> dict[str, Any]:
    """One CPU loader sample; preserve the base transform's RNG and class target.

    Returns clean ``segmentation`` (tensor or its original DS list), fullres
    ``pasted_mask``/``valid_mask``, corresponding ``*_ds`` lists, counts and a
    conservative ``mask_truncated`` boundary-contact flag. The normal loader
    must collate these fields with the same sample ordering as its CP event ID.

    Validity is a continuous regression target sharing the SAME spatial grid
    and mirror draw. A binary all-one segmentation channel is unsafe: nnU-Net's
    per-channel one-hot argmax can otherwise fill out-of-grid regions with one.
    Only full valid interpolation support (within float32 tolerance 1e-6) is
    accepted. Validity DS uses nnU-Net's own nearest-exact convention.
    """
    if is_cascaded or regions is not None or do_dummy_2d_data_aug or ignore_label is not None:
        raise FeedbackMetricError("Feedback supports non-cascade, non-region, 3-D class training without an ignore label only")
    if (not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] < 1
            or min(image.shape[1:]) < 1 or image.device.type != "cpu"):
        raise FeedbackMetricError("Feedback augmentation expects a CPU C,D,H,W loader sample")
    shape = (1, *image.shape[1:])
    _tensor_shape(segmentation, shape, "segmentation", image.device)
    _tensor_shape(pasted_mask, shape, "pasted_mask", image.device)
    if segmentation.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise FeedbackMetricError("Class segmentation must be a signed integer tensor")
    _reject_invalid({
        "nonfinite image": ~torch.isfinite(image).all(),
        "unsupported input labels": ~((segmentation == -1) | (segmentation == 0) | (segmentation == 1) | (segmentation == 2)).all(),
        "nonbinary input pasted mask": ~_is_binary(pasted_mask),
        "input pasted mask disagrees with tumor target": (pasted_mask.bool() & (segmentation != 2)).any(),
    })
    _validate_transform_tree(base_transform)
    transformed = base_transform(
        image=image.clone(),
        segmentation=torch.cat((segmentation, pasted_mask.to(segmentation.dtype)), dim=0),
        regression_target=(segmentation != -1).float(),
    )
    image_out = transformed.get("image")
    targets = transformed.get("segmentation")
    validity = transformed.get("regression_target")
    if (not torch.is_tensor(image_out) or image_out.ndim != 4
            or not torch.is_tensor(validity) or tuple(validity.shape) != (1, *image_out.shape[1:])):
        raise FeedbackMetricError("Transform did not retain the shared full-resolution validity target")
    _reject_invalid({
        "nonfinite augmented image": ~torch.isfinite(image_out).all(),
        "invalid transformed validity support": (~torch.isfinite(validity) | (validity < -1e-6) | (validity > 1 + 1e-6)).any(),
    })
    target_list = targets if isinstance(targets, list) else [targets]
    if not target_list:
        raise FeedbackMetricError("Transform returned no deep-supervision target")
    if not torch.is_tensor(target_list[0]) or tuple(target_list[0].shape) != (2, *image_out.shape[1:]):
        raise FeedbackMetricError("First DS target must be full resolution with the class and attribution channels")
    labels_out, masks_out, valid_out, raw_counts, removed_labels, removed_padding = [], [], [], [], [], []
    full_valid = validity >= 1 - 1e-6
    # This loop is only over the architecture's DS scales, not batch samples or
    # GPU forward passes. The loader already transforms one CPU sample at a time.
    for target in target_list:
        if (not torch.is_tensor(target) or target.ndim != 4 or target.shape[0] != 2
                or min(target.shape[1:]) < 1):
            raise FeedbackMetricError("Every DS target must retain exactly two channels before extraction")
        label, raw_support = target[:1], target[1:2]
        _reject_invalid({
            "unsupported transformed labels": ~((label == 0) | (label == 1) | (label == 2)).all(),
            "nonbinary transformed pasted support": ~_is_binary(raw_support),
        })
        valid = (full_valid if tuple(target.shape[1:]) == tuple(full_valid.shape[1:]) else
                 F.interpolate(full_valid[None].float(), size=target.shape[1:], mode="nearest-exact")[0].bool())
        support = raw_support.bool()
        surviving = support & valid & (label == 2)
        labels_out.append(label)
        masks_out.append(surviving)
        valid_out.append(valid)
        raw_counts.append(support.sum())
        removed_labels.append((support & valid & (label != 2)).sum())
        removed_padding.append((support & ~valid).sum())
    return {
        "image": image_out,
        "segmentation": labels_out if isinstance(targets, list) else labels_out[0],
        "pasted_mask": masks_out[0], "valid_mask": valid_out[0],
        "pasted_mask_ds": masks_out, "valid_mask_ds": valid_out,
        "input_pasted_voxels": pasted_mask.bool().sum(),
        "raw_support_count": raw_counts[0],
        "removed_by_label_resampling": removed_labels[0],
        "removed_by_padding": removed_padding[0],
        "raw_support_count_ds": torch.stack(raw_counts),
        "removed_by_label_resampling_ds": torch.stack(removed_labels),
        "removed_by_padding_ds": torch.stack(removed_padding),
        "mask_truncated": _border_contact(masks_out[0][None], valid_out[0][None])[0],
    }
