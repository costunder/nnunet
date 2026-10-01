"""Read-only decomposition of the approved native LocalCNN L0 forward.

The production output is unchanged. All returned stage tensors are detached,
candidate-level vectors; no 3-D CNN feature maps are retained by the result.
``anchor_radius_mm`` is required and is a diagnostic readout, not a change to
the production ROI, encoder receptive field, masks, or physical batch size.

The CNN uses odd-kernel padded convolutions. Feature-cell centers therefore
occur at native crop indices 0, stride, 2 * stride for strides 1, 2, 4. The
anchor sphere below averages *feature-cell centers* within its physical radius
and the actual downsampled organ mask. Features still have the production CNN
receptive field; this is not a second CNN confined to that small sphere.
"""
import math

import torch

from l0_local_cnn.model import LocalBatch
from l0_regions.resident import check_verified


def _geometry(batch, recipient_centers_native, donor_centers_native):
    """Keep source/native coordinate metadata explicit; never infer an anchor."""
    if len(batch.audit) != len(batch.images):
        raise ValueError('One native geometry audit per unique crop required')
    device = batch.images.device
    origins = torch.as_tensor([a['origin'] for a in batch.audit], device=device)
    shapes = torch.as_tensor([a['shape'] for a in batch.audit], device=device)
    spacing = torch.as_tensor([a['spacing'] for a in batch.audit], device=device, dtype=torch.float32)
    padded_shape = batch.images.shape[2:]
    padded = torch.as_tensor(padded_shape, device=device)
    if origins.shape != shapes.shape or origins.shape != spacing.shape or origins.shape != (len(batch.images), 3):
        raise ValueError('Audit origin/shape/spacing must have three native axes')
    if not bool(torch.isfinite(origins).all() & (origins == origins.round()).all() & (origins >= 0).all()):
        raise ValueError('Finite nonnegative integer crop origin required')
    if not bool(torch.isfinite(shapes).all() & (shapes == shapes.round()).all() & (shapes > 0).all() & (shapes <= padded).all()):
        raise ValueError('Finite positive native shape inside padded tensor required')
    if not bool(torch.isfinite(spacing).all() & (spacing > 0).all()):
        raise ValueError('Finite positive native physical spacing required')
    # CropStore pads only at the upper faces. Padding must never contribute to
    # either the production organ mean or the diagnostic local feature readout.
    inside_shape = (
        (torch.arange(padded_shape[0], device=device)[None, :, None, None] < shapes[:, 0, None, None, None])
        & (torch.arange(padded_shape[1], device=device)[None, None, :, None] < shapes[:, 1, None, None, None])
        & (torch.arange(padded_shape[2], device=device)[None, None, None, :] < shapes[:, 2, None, None, None])
    )
    if bool((batch.organ[:, 0] & ~inside_shape).any()):
        raise ValueError('Organ mask includes padding outside native crop shape')
    positions = {}
    for role, values, ids in (
        ('recipient', recipient_centers_native, batch.recipient),
        ('donor', donor_centers_native, batch.donor),
    ):
        centers = torch.as_tensor(values, device=device, dtype=torch.float32)
        if centers.shape != (len(batch), 3) or not bool(torch.isfinite(centers).all() & (centers == centers.round()).all()):
            raise ValueError(f'{role} requires every pair\'s finite integer native anchor')
        local = centers - origins[ids]
        if not bool(((local >= 0) & (local < shapes[ids])).all()):
            raise ValueError(f'{role} anchor lies outside its bound native crop')
        # Concave tumor bbox anchors may be background. Do not move the anchor
        # or impose a new organ-membership condition on production observations.
        index = local.long()
        in_organ = batch.organ[ids, 0, index[:, 0], index[:, 1], index[:, 2]]
        positions[role] = (local, spacing[ids], ids, centers, in_organ)
    return positions


def _anchor_sphere_mean(features, mask, positions, spacing, ids, stride, radius_mm):
    """Vectorized B x local-cell gather; never duplicate full maps per pair."""
    shape = torch.as_tensor(features.shape[2:], device=features.device)
    step = spacing * stride
    p = positions / stride
    nearest = torch.floor(p).long()
    # A full physical sphere is considered. There is no node/voxel cap or
    # silently reduced radius. Parent admission retains the explicit GPU budget.
    reach = torch.ceil(radius_mm / step.min(0).values).long() + 1
    axes = [torch.arange(-n, n + 1, device=features.device) for n in reach.unbind()]
    offsets = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, 3)
    cells = nearest[:, None] + offsets[None]
    valid = ((cells >= 0) & (cells < shape)).all(-1)
    distance2 = ((cells.to(p.dtype) - p[:, None]) * step[:, None]).square().sum(-1)
    valid &= distance2 <= radius_mm ** 2
    safe = cells.maximum(torch.zeros_like(cells)).minimum(shape - 1)
    flat = (safe[..., 0] * shape[1] + safe[..., 1]) * shape[2] + safe[..., 2]
    organ = mask[:, 0].flatten(1)[ids[:, None], flat]
    admitted = valid & organ
    count = admitted.sum(1)
    covered = count > 0
    report = dict(
        status='MEASURED' if bool(covered.all()) else 'NOT_RUN',
        reason=None if bool(covered.all()) else 'At least one candidate has no organ-supported feature-cell center in the explicit anchor sphere; no candidate was skipped',
        feature_stride_native=stride,
        physical_radius_mm=float(radius_mm),
        candidate_organ_cell_counts=count.cpu().tolist(),
        candidate_covered=covered.cpu().tolist(),
        all_candidates_retained=True,
    )
    if not bool(covered.all()):
        # Do not turn an unavailable diagnostic into a zero-vector observation
        # or report variance over a smaller set of candidates as the full case.
        return None, report
    values = features.flatten(2).transpose(1, 2)[ids[:, None], flat]
    pooled = torch.where(admitted[..., None], values, 0).sum(1) / count[:, None]
    if not bool(torch.isfinite(pooled).all()):
        raise FloatingPointError('Nonfinite anchor feature readout')
    return pooled.detach(), report


@torch.no_grad()
def trace_local_cnn(local, batch, *, recipient_centers_native, donor_centers_native, anchor_radius_mm):
    """Return ``(production_output, stages, audit)`` for a same-donor batch.

    Native anchors must be supplied from each observation's ``row['center']``
    and ``CropStore.donor_bounds(row)[0]``. ``LocalBatch.audit`` does not record
    anchors, so guessing the padded tensor midpoint would be incorrect.

    Stage values are detached ``[pairs, channels]`` tensors. An anchor stage is
    ``None`` when *any* candidate has no supported cell; its audit is NOT_RUN.
    A caller concatenating physical batches must propagate this status to the
    whole-case stage and must not silently compare only available candidates.
    Normalized spread measures directional differences, not feature semantics.
    """
    if not isinstance(batch, LocalBatch):
        raise TypeError('Approved native local CT batch required')
    check_verified(batch)
    if any(module.training for module in local.modules()):
        raise ValueError('L0 read-only deterministic trace requires caller-selected eval mode')
    if not isinstance(anchor_radius_mm, (int, float)) or not math.isfinite(anchor_radius_mm) or anchor_radius_mm <= 0:
        raise ValueError('Explicit finite positive diagnostic anchor radius in mm required')
    if not bool((batch.donor == batch.donor[0]).all()):
        raise ValueError('Same-donor candidate comparison required; donor crop differs within batch')
    if local.resource_budget:
        local.resource_budget.check()
    positions = _geometry(batch, recipient_centers_native, donor_centers_native)
    maps = local.cnn(batch.images, batch.organ)
    stages = {}
    audit = dict(
        diagnostic_only=True,
        anchor_radius_mm=float(anchor_radius_mm),
        physical_pairs=len(batch),
        unique_cnn_crops=len(batch.images),
        donor_crop_fixed=True,
        feature_strides_native=[1, 2, 4],
        anchor_readouts={},
        map_spatial_variance={},
        geometry=batch.audit,
        anchor_note='The sphere uses native feature-cell centers and actual organ masks; feature receptive fields remain those of the production CNN',
        variance_note='Vector spread is a representation statistic, not evidence of tumor-specific semantic information',
        unavailability_policy='If any anchor ROI is unsupported, that stage is NOT_RUN for the full candidate set; no candidate skip or zero-vector fallback',
    )
    pooled = []
    for level, ((x, m), stride) in enumerate(zip(maps, (1, 2, 4)), 1):
        count = m.sum((2, 3, 4))
        if not bool((count > 0).all()):
            raise ValueError('Organ vanished at a CNN scale; no silent zero representation')
        mean = torch.where(m, x, 0).sum((2, 3, 4)) / count
        pooled.append(mean)
        # Within-map spatial variation has different meaning from between-
        # candidate variation. Retain one scalar per crop, not voxel tensors.
        # OrganPyramid already zeros every nonorgan feature cell. Batched dot
        # products compute second moments without allocating an x-sized square
        # tensor on top of the production feature maps.
        flat = x.flatten(2)
        second = torch.einsum('vcs,vcs->vc', flat, flat) / count
        spatial = (second - mean.square()).clamp_min(0).sum(1)
        audit['map_spatial_variance'][f'scale{level}'] = dict(
            definition='sum of channel-wise organ-masked spatial variances within each crop; not candidate variance',
            by_unique_crop=spatial.cpu().tolist(),
            organ_cell_counts=count[:, 0].cpu().tolist(),
        )
        for role, (pos, spacing, ids, centers, in_organ) in positions.items():
            stages[f'scale{level}_global_mean_{role}'] = mean[ids].detach()
            key = f'scale{level}_anchor_roi_{role}'
            value, report = _anchor_sphere_mean(x, m, pos, spacing, ids, stride, anchor_radius_mm)
            stages[key] = value
            report['native_anchors'] = centers.cpu().tolist()
            report['anchors_in_native_organ'] = in_organ.cpu().tolist()
            audit['anchor_readouts'][key] = report
    concat = torch.cat(pooled, 1)
    features = local.project(concat)
    d = features[batch.donor]
    r = features[batch.recipient]
    fusion_input = torch.cat((d, r, r - d, r * d), 1)
    output = local.fuse(fusion_input)
    if output.shape != (len(batch), 128) or not bool(torch.isfinite(output).all()):
        raise FloatingPointError('Invalid local CNN output')
    for role, ids in (('donor', batch.donor), ('recipient', batch.recipient)):
        stages[f'global_mean_concat_{role}'] = concat[ids].detach()
        stages[f'project_{role}'] = features[ids].detach()
    stages['paired_recipient_minus_donor'] = (r - d).detach()
    stages['paired_recipient_times_donor'] = (r * d).detach()
    stages['fusion_input'] = fusion_input.detach()
    stages['fusion_output'] = output.detach()
    if local.resource_budget:
        local.resource_budget.check()
    return output.detach(), stages, audit
