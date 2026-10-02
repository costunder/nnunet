"""GPU-only all-pool representation distances for DEBUG budget comparisons.

These are geometric/CNN representation errors, not cancer relevance, retained
information percentages, or ranking accuracy. Query is abstract and excluded.
Every eligible point and every selected context node participates. Workspace
chunking changes execution storage only, never the pool or graph definition.
"""
import torch
from torch.nn import functional as F


def distance_stats(values, mask=None):
    if values.device.type != 'cuda':
        raise ValueError('CUDA distance statistics required')
    if mask is None:
        if values.numel() == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError('Nonempty finite metric required')
        return torch.stack((values.mean(), torch.quantile(values.flatten(), .95), values.max()))
    if values.ndim != 2 or mask.shape != values.shape or not bool(mask.any(-1).all()):
        raise ValueError('Every measured scene needs its complete nonempty pool')
    if not bool(torch.isfinite(values[mask]).all()):
        raise FloatingPointError('Nonfinite representation distance')
    mean = torch.where(mask, values, 0).sum(-1) / mask.sum(-1)
    percentile = torch.nanquantile(values.masked_fill(~mask, float('nan')), .95, dim=-1)
    maximum = values.masked_fill(~mask, -float('inf')).max(-1).values
    return torch.stack((mean, percentile, maximum), -1)


@torch.no_grad()
def all_pool_distances(graph, *, workspace_bytes):
    """Return G x full-pool errors and vectorized scene/band summaries.

    Feature and space minima are independent. The joint minimum uses ONE SAME
    selected node for both terms. All three budgets use original full-band
    dispersions already computed by the selector, never selected-set variance.
    """
    if type(workspace_bytes) is not int or workspace_bytes <= 0:
        raise ValueError('Explicit positive coverage workspace required')
    pool = graph['coverage_pool']
    f, xyz, eligible, band = (pool[k] for k in ('features', 'relative_mm', 'eligible', 'band'))
    if f.device.type != 'cuda' or any(t.device != f.device for t in (xyz, eligible, band)):
        raise ValueError('CUDA complete feature/geometry pool required; no CPU fallback')
    if f.ndim != 3 or f.shape[:2] != eligible.shape or xyz.shape != (*eligible.shape, 3):
        raise ValueError('Malformed complete pool')
    norms = f.norm(dim=-1)
    if (not bool(torch.isfinite(f[eligible]).all()) or not bool(torch.isfinite(norms[eligible]).all())
            or not bool((norms[eligible] >= 1e-12).all())):
        raise FloatingPointError('Cosine metric undefined for nonfinite/zero-norm eligible CNN feature; no point exclusion')
    if not bool(torch.isfinite(xyz[eligible]).all()):
        raise FloatingPointError('Nonfinite eligible physical geometry')
    normalized = F.normalize(f, dim=-1)
    g, width = eligible.shape
    bands = []
    for role in (1, 2, 3):
        member = eligible & (band == role)
        if not bool(member.any(-1).all()):
            raise ValueError('Empty complete band')
        # Retained paths have scene-specific shell positions and padding.
        # Fixed quota graphs are the special case where every row agrees.
        slots = (graph['role'] == role).any(0).nonzero().flatten()
        if not len(slots):
            raise ValueError('Missing selected context role')
        selected_mask = graph['node_mask'][:, slots] & (graph['role'][:, slots] == role)
        if not bool(selected_mask.any(-1).all()):
            raise ValueError('Empty selected context band')
        selected_features = graph['features'][:, slots]
        selected_norms = selected_features.norm(dim=-1)
        if (not bool(torch.isfinite(selected_norms[selected_mask]).all())
                or not bool((selected_norms[selected_mask] >= 1e-12).all())):
            raise FloatingPointError('Selected CNN cosine norm is zero')
        selected = F.normalize(selected_features, dim=-1)
        selected_xyz = graph['relative_xyz_mm'][:, slots]
        df = graph['statistics']['band_feature_variance'][:, role - 1]
        ds = graph['statistics']['band_spatial_variance_mm2'][:, role - 1]
        if not bool(torch.isfinite(df).all() & torch.isfinite(ds).all()):
            raise FloatingPointError('Nonfinite full-pool metric normalization')
        all_retained = member.sum(-1) == selected_mask.sum(-1)
        if not bool(((df > 0) & (ds > 0) | all_retained).all()):
            raise FloatingPointError('Nonpositive dispersion for an incompletely represented band')
        # A constant retain-all component has zero distance. This inactive
        # denominator does not invent a feature, remove a point or alter nodes.
        df = torch.where(df > 0, df, 1)
        ds = torch.where(ds > 0, ds, 1)
        # Conservative bound for simultaneous distance/normalization matrices.
        per_point_bytes = g * len(slots) * f.element_size() * 12
        chunk = min(width, workspace_bytes // per_point_bytes)
        if chunk < 1:
            raise MemoryError('Coverage workspace cannot hold one complete point/selected-node row')
        feature_parts, spatial_parts, joint_parts = [], [], []
        for start in range(0, width, chunk):
            stop = min(width, start + chunk)
            current = normalized[:, start:stop]
            dot = torch.bmm(current, selected.transpose(1, 2))
            deficit = (1 - dot).clamp_min(0)
            feature_d2 = (current.square().sum(-1)[..., None] + selected.square().sum(-1)[:, None] - 2 * dot).clamp_min(0)
            pos = xyz[:, start:stop]
            spatial_d2 = (pos.square().sum(-1)[..., None] + selected_xyz.square().sum(-1)[:, None]
                - 2 * torch.bmm(pos, selected_xyz.transpose(1, 2))).clamp_min(0)
            legal = selected_mask[:, None]
            feature_parts.append(deficit.masked_fill(~legal, float('inf')).min(-1).values)
            spatial_parts.append(spatial_d2.masked_fill(~legal, float('inf')).min(-1).values.sqrt())
            joint = (spatial_d2 / ds[:, None, None] + feature_d2 / df[:, None, None])
            joint_parts.append(joint.masked_fill(~legal, float('inf')).min(-1).values)
        values = dict(feature_deficit=torch.cat(feature_parts, -1),
            spatial_mm=torch.cat(spatial_parts, -1), joint=torch.cat(joint_parts, -1))
        summaries = {key: distance_stats(value, member) for key, value in values.items()}
        bands.append(dict(role=role, mask=member, values=values, summaries=summaries,
            full_counts=member.sum(-1), selected_counts=selected_mask.sum(-1),
            workspace_chunk_points=chunk))
    return bands
