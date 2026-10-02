"""Batched, organ-only sparse feature graph for a separate DEBUG L0 comparison.

The CNN is copied from the named reference. New graph parameters are seeded,
untrained parameters, not a trained recommendation model. Hard feature/space
selection is nondifferentiable; selected features and all GNN layers remain
differentiable. Graph edges describe message passing, never a sampled CT path.
"""
import copy
import math
from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.nn import functional as F

from l0_local_cnn.model import LocalBatch
from l0_regions.resident import check_verified
from tools.local_cnn_l0_probe import _geometry


class SparseFeatureCoverageError(ValueError):
    """An explicit role/query cannot be represented without a fallback."""


@dataclass(frozen=True)
class SparseFeatureProfile:
    context_nodes_per_band: int
    query_radius_mm: float
    near_radius_mm: float
    mid_radius_mm: float
    spatial_neighbors: int = 3
    feature_neighbors: int = 1
    spatial_metric_weight: float = 1.0
    feature_metric_weight: float = 1.0
    hidden_dim: int = 128
    sage_layers: int = 3
    initialization_seed: int = 42
    debug: bool = True

    def validate(self):
        if not self.debug:
            raise ValueError('This new sparse feature graph has a DEBUG-only profile')
        if type(self.context_nodes_per_band) is not int or self.context_nodes_per_band not in (16, 32, 64):
            raise ValueError('Explicit DEBUG context quota must be 16/32/64 per band (48/96/192 context nodes)')
        if (self.spatial_neighbors, self.feature_neighbors,
                self.hidden_dim, self.sage_layers) != (3, 1, 128, 3):
            raise ValueError('Reviewed 3+1 neighbors / 3x128 DEBUG architecture required')
        values = (self.query_radius_mm, self.near_radius_mm, self.mid_radius_mm,
                  self.spatial_metric_weight, self.feature_metric_weight)
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Explicit finite positive radii and selection metric weights required')
        if self.near_radius_mm >= self.mid_radius_mm:
            raise ValueError('Near radius must be smaller than mid radius')


def _masked_corner_features(maps, positions, crop_ids):
    """All eight corners at strides 1/2/4, with organ weight renormalization.

    Unsupported points have an explicit support mask. They never become graph
    nodes or zero-vector query fallbacks. This is eight vectorized gathers per
    scale, not a per-scene or per-node forward.
    """
    pieces, support = [], []
    for (features, mask), stride in zip(maps, (1, 2, 4)):
        p = positions / stride
        lo = p.floor().long()
        frac = p - lo
        shape = torch.as_tensor(features.shape[2:], device=p.device)
        flat_features = features.flatten(2).transpose(1, 2)
        flat_mask = mask[:, 0].flatten(1)
        total = features.new_zeros((len(p), features.shape[1]))
        mass = features.new_zeros(len(p))
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    bit = p.new_tensor((a, b, c), dtype=torch.long)
                    corner = lo + bit
                    inside = ((corner >= 0) & (corner < shape)).all(-1)
                    safe = corner.maximum(torch.zeros_like(corner)).minimum(shape - 1)
                    flat = (safe[:, 0] * shape[1] + safe[:, 1]) * shape[2] + safe[:, 2]
                    valid = inside & flat_mask[crop_ids, flat]
                    weight = torch.where(bit.bool(), frac, 1 - frac).prod(-1)
                    weight = torch.where(valid, weight, 0)
                    total = total + torch.where(valid[:, None], flat_features[crop_ids, flat], 0) * weight[:, None]
                    mass = mass + weight
        covered = mass > 0
        pieces.append(total / mass.clamp_min(torch.finfo(total.dtype).tiny)[:, None])
        support.append(covered)
    return torch.cat(pieces, -1), torch.stack(support, -1)


class _MeanSAGE(nn.Module):
    """Neighbor transform only: root information enters one external residual."""
    root_weight = False

    def __init__(self, hidden):
        super().__init__()
        self.neighbor = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, adjacency, node_mask):
        degree = adjacency.sum(-1, keepdim=True).clamp_min(1)
        mean = torch.bmm(adjacency.to(x.dtype), x) / degree
        return torch.where(node_mask[..., None], F.silu(self.norm(x + self.neighbor(mean))), 0)


@torch.no_grad()
def _select_context(pool_mm, pool_features, eligible, band, distance, profile):
    """Batched hard k-center with measured, per-band distance normalization.

    2 * E[||v - E[v]||^2] equals the mean squared distance between two
    independently drawn band members, including equal-point draws. Centered
    arithmetic avoids subtracting nearly equal second moments for CNN vectors.
    These statistics scale the selection metric; they are not CT descriptors.
    A constant/nonfinite band needing selection is rejected, never changed to
    geometry-only FPS. When all members fit the explicit quota, every member is
    retained; no metric selection is needed for that scene/band.
    """
    normalized = F.normalize(pool_features.detach(), dim=-1)
    feature_variance, spatial_variance, normalization_required = [], [], []
    chosen_ids, chosen_mask = [], []
    rows = torch.arange(len(eligible), device=eligible.device)
    for role in (1, 2, 3):
        member = eligible & (band == role)
        count = member.sum(-1)
        if not bool((count > 0).all()):
            raise SparseFeatureCoverageError('Empty near/mid/wide organ-supported band; no role fallback')
        # Translate by a real member before averaging. Identical float32 CNN
        # vectors then produce exactly zero dispersion instead of a spurious
        # positive value caused by rounding a repeated-vector sum/division.
        first = member.long().argmax(-1)
        feature_offset = normalized - normalized[rows, first, None]
        spatial_offset = pool_mm - pool_mm[rows, first, None]
        feature_mean = torch.where(member[..., None], feature_offset, 0).sum(1) / count[:, None]
        spatial_mean = torch.where(member[..., None], spatial_offset, 0).sum(1) / count[:, None]
        feature_dispersion = 2 * torch.where(member,
            (feature_offset - feature_mean[:, None]).square().sum(-1), 0).sum(-1) / count
        spatial_dispersion = 2 * torch.where(member,
            (spatial_offset - spatial_mean[:, None]).square().sum(-1), 0).sum(-1) / count
        needs_selection = count > profile.context_nodes_per_band
        finite = torch.isfinite(feature_dispersion) & torch.isfinite(spatial_dispersion)
        valid = finite & (~needs_selection | ((feature_dispersion > 0) & (spatial_dispersion > 0)))
        if not bool(valid.all()):
            raise FloatingPointError('Zero or nonfinite per-band CNN/physical selection dispersion; no geometry-only fallback')
        feature_variance.append(feature_dispersion)
        spatial_variance.append(spatial_dispersion)
        normalization_required.append(needs_selection)
        # The value 1 for a retain-all band is only an inactive tensor
        # denominator. It never changes a selected set or supplies a feature.
        feature_denominator = torch.where(needs_selection, feature_dispersion, 1)
        spatial_denominator = torch.where(needs_selection, spatial_dispersion, 1)
        available = member.clone()
        nearest = distance.masked_fill(~available, float('inf')).argmin(-1)
        minimum = distance.new_full(distance.shape, float('inf'))
        for step in range(profile.context_nodes_per_band):
            exists = available.any(-1)
            if step == 0:
                selected = nearest
            else:
                fps_choice = minimum.masked_fill(~available, -float('inf')).argmax(-1)
                retain_all_choice = available.long().argmax(-1)
                selected = torch.where(needs_selection, fps_choice, retain_all_choice)
            selected = torch.where(exists, selected, 0)
            chosen_ids.append(selected)
            chosen_mask.append(exists)
            spatial_delta = (pool_mm - pool_mm[rows, selected, None]).square().sum(-1) / spatial_denominator[:, None]
            feature_delta = (normalized - normalized[rows, selected, None]).square().sum(-1) / feature_denominator[:, None]
            metric = (profile.spatial_metric_weight * spatial_delta
                      + profile.feature_metric_weight * feature_delta)
            minimum = torch.minimum(minimum, metric)
            available[rows[exists], selected[exists]] = False
    statistics = dict(
        band_feature_variance=torch.stack(feature_variance, -1),
        band_spatial_variance_mm2=torch.stack(spatial_variance, -1),
        band_normalization_required=torch.stack(normalization_required, -1),
        selection_metric='w_space*||delta_mm||^2/band_spatial_dispersion + w_feature*||delta_normalized_CNN||^2/band_feature_dispersion',
        dispersion_definition='2 * mean(||v - mean(v)||^2) over all eligible members of each scene/band; selection normalization only, not input features',
        zero_variance_policy='If count exceeds quota: explicit FloatingPointError, no geometry-only fallback. Otherwise retain every available member without FPS selection.')
    return torch.stack(chosen_ids, -1), torch.stack(chosen_mask, -1), statistics


def _connected(adjacency, node_mask):
    reached = torch.zeros_like(node_mask)
    reached[:, 0] = True
    for _ in range(adjacency.shape[1]):
        reached |= torch.bmm(adjacency.float(), reached.float()[..., None]).squeeze(-1) > 0
    return (reached | ~node_mask).all(-1)


def _build_edges(xyz_mm, features, node_mask, spatial_k, feature_k):
    """Symmetric local neighbors plus an explicit MST only if disconnected."""
    g, n = node_mask.shape
    valid = node_mask[:, :, None] & node_mask[:, None, :]
    eye = torch.eye(n, device=node_mask.device, dtype=torch.bool)[None]
    spatial = torch.cdist(xyz_mm, xyz_mm).square()
    normalized = F.normalize(features.detach(), dim=-1)
    feature = (2 - 2 * torch.bmm(normalized, normalized.transpose(1, 2))).clamp_min(0)
    legal = valid & ~eye
    kind = torch.zeros((g, n, n), device=node_mask.device, dtype=torch.int16)
    for distance, k, flag in ((spatial, spatial_k, 1), (feature, feature_k, 2)):
        ids = distance.masked_fill(~legal, float('inf')).argsort(dim=-1, stable=True)[..., :k]
        chosen = torch.zeros_like(legal).scatter(-1, ids, True) & legal
        chosen |= chosen.transpose(1, 2).clone()
        kind |= chosen.to(kind.dtype) * flag
    adjacency = kind != 0
    before = _connected(adjacency, node_mask)
    active = ~before
    visited = torch.zeros_like(node_mask)
    visited[:, 0] = True
    rows = torch.arange(g, device=node_mask.device)
    mst_added = torch.zeros(g, device=node_mask.device, dtype=torch.long)
    if bool(active.any()):
        for _ in range(n - 1):
            candidates = legal & visited[:, :, None] & ~visited[:, None, :]
            best = spatial.masked_fill(~candidates, float('inf')).flatten(1).argmin(-1)
            u, v = best // n, best % n
            exists = candidates.flatten(1).any(-1)
            add = active & exists
            new_edge = add & (kind[rows, u, v] == 0)
            kind[rows[add], u[add], v[add]] |= 4
            kind[rows[add], v[add], u[add]] |= 4
            mst_added += new_edge
            visited[rows[exists], v[exists]] = True
    adjacency = kind != 0
    if not bool(_connected(adjacency, node_mask).all()):
        raise SparseFeatureCoverageError('Graph remains disconnected; no scene or node was skipped')
    return adjacency, kind, before, mst_added


class SparseFeatureL0(nn.Module):
    def __init__(self, reference_local, profile, budget=None):
        super().__init__()
        if not isinstance(profile, SparseFeatureProfile):
            raise TypeError('Explicit SparseFeatureProfile required')
        profile.validate()
        if reference_local.config['channels'] != [12, 24, 32] or reference_local.config['convolutions'] != [2, 3, 3]:
            raise ValueError('Reference CNN must retain [12,24,32] and eight convolutions')
        self.profile = profile
        self.margin_mm = float(reference_local.config['margin_mm'])
        self.resource_budget = budget
        self.cnn = copy.deepcopy(reference_local.cnn)
        self.fuse = copy.deepcopy(reference_local.fuse)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(profile.initialization_seed)
            self.node_project = nn.Linear(68 + 3, 128)
            self.role_embedding = nn.Embedding(4, 128)
            self.blocks = nn.ModuleList([_MeanSAGE(128) for _ in range(3)])
            self.scene_project = nn.Sequential(nn.Linear(4 * 128, 128), nn.LayerNorm(128), nn.SiLU())
        self.weight_sources = dict(CNN='exact deep copy of caller reference_local.cnn',
            paired_fusion='exact deep copy of caller reference_local.fuse',
            graph='fresh seed42 initialization; untrained DEBUG comparison')
        self.to(next(reference_local.parameters()).device)

    def _budget(self):
        if self.resource_budget:
            self.resource_budget.check()

    def forward(self, batch, *, recipient_centers_native, donor_centers_native,
                return_graph=False, return_pool=False):
        if return_pool and not return_graph:
            raise ValueError('return_pool requires return_graph; no implicit large diagnostic output')
        if not isinstance(batch, LocalBatch):
            raise TypeError('Native LocalBatch required')
        check_verified(batch)
        if batch.images.device.type != 'cuda':
            raise ValueError('Sparse feature numerical path requires CUDA; no CPU fallback')
        self._budget()
        positions = _geometry(batch, recipient_centers_native, donor_centers_native)
        donor, recipient = positions['donor'], positions['recipient']
        ids = torch.cat((donor[2], recipient[2]))
        anchors_local = torch.cat((donor[0], recipient[0]))
        anchors_native = torch.cat((donor[3], recipient[3]))
        spacing = torch.cat((donor[1], recipient[1]))
        maps = self.cnn(batch.images, batch.organ)
        points = batch.organ[:, 0].nonzero()
        crop_ids, local_points = points[:, 0], points[:, 1:].float()
        with torch.no_grad():
            pool_features, scale_support = _masked_corner_features(maps, local_points, crop_ids)
        if not bool(torch.isfinite(pool_features[scale_support.all(-1)]).all()):
            raise FloatingPointError('Nonfinite supported fine-pool CNN features')
        crop_counts = torch.bincount(crop_ids, minlength=len(batch.images))
        offsets = crop_counts.cumsum(0) - crop_counts
        width = int(crop_counts.max())
        slots = torch.arange(width, device=ids.device)
        pool_index = (offsets[ids, None] + slots).clamp_max(len(points) - 1)
        fine_mask = slots[None] < crop_counts[ids, None]
        supported = scale_support[pool_index]
        eligible = fine_mask & supported.all(-1)
        pool_xyz = local_points[pool_index]
        pool_mm = (pool_xyz - anchors_local[:, None]) * spacing[:, None]
        distance = torch.linalg.vector_norm(pool_mm, dim=-1)
        pool = pool_features[pool_index]
        band = torch.where(distance <= self.profile.near_radius_mm, 1,
            torch.where(distance <= self.profile.mid_radius_mm, 2, 3))
        counts = torch.stack([(eligible & (band == role)).sum(-1) for role in (1, 2, 3)], -1)
        if not bool((counts > 0).all()):
            raise SparseFeatureCoverageError('Empty near/mid/wide organ-supported band; no node drop, radius change or zero readout')
        query_mask = eligible & (distance <= self.profile.query_radius_mm)
        query_counts = query_mask.sum(-1)
        if not bool((query_counts > 0).all()):
            raise SparseFeatureCoverageError('Original anchor sphere has no all-scale organ support; anchor and radius unchanged')
        self._budget()
        selection, context_mask, metric_statistics = _select_context(pool_mm, pool,
            eligible, band, distance, self.profile)
        selected_xyz = pool_xyz.gather(1, selection[..., None].expand(-1, -1, 3))
        selected_features, selected_support = _masked_corner_features(maps,
            selected_xyz.reshape(-1, 3), ids[:, None].expand_as(selection).reshape(-1))
        selected_features = selected_features.reshape(len(ids), -1, 68)
        if not bool((selected_support.all(-1).reshape_as(context_mask) | ~context_mask).all()):
            raise SparseFeatureCoverageError('Selected graph node lost CNN scale support')
        query_scene, query_slot = query_mask.nonzero(as_tuple=True)
        query_values, query_support = _masked_corner_features(maps,
            pool_xyz[query_scene, query_slot], ids[query_scene])
        if not bool(query_support.all()):
            raise SparseFeatureCoverageError('Query sphere has an unsupported CNN scale')
        query_features = query_values.new_zeros((len(ids), 68)).index_add(0, query_scene, query_values)
        query_features = query_features / query_counts[:, None]
        features = torch.cat((query_features[:, None], selected_features), 1)
        mask = torch.cat((torch.ones((len(ids), 1), device=ids.device, dtype=torch.bool), context_mask), 1)
        role = torch.cat((torch.zeros(1, device=ids.device, dtype=torch.long),
            torch.arange(1, 4, device=ids.device).repeat_interleave(self.profile.context_nodes_per_band)))
        role = role[None].expand(len(ids), -1)
        xyz_local = torch.cat((anchors_local[:, None], selected_xyz), 1)
        relative_mm = (xyz_local - anchors_local[:, None]) * spacing[:, None]
        features = torch.where(mask[..., None], features, 0)
        adjacency, edge_kind, connected_before, mst_added = _build_edges(relative_mm,
            features, mask, self.profile.spatial_neighbors, self.profile.feature_neighbors)
        self._budget()
        h = self.node_project(torch.cat((features, relative_mm / self.margin_mm), -1)) + self.role_embedding(role)
        h = torch.where(mask[..., None], h, 0)
        for block in self.blocks:
            h = block(h, adjacency, mask)
        summaries = [h[:, 0]]
        for value in (1, 2, 3):
            member = mask & (role == value)
            summaries.append(torch.where(member[..., None], h, 0).sum(1) / member.sum(1)[:, None])
        scene = self.scene_project(torch.cat(summaries, -1))
        d, r = scene[:len(batch)], scene[len(batch):]
        output = self.fuse(torch.cat((d, r, r - d, r * d), -1))
        if output.shape != (len(batch), 128) or not bool(torch.isfinite(output).all()):
            raise FloatingPointError('Invalid sparse feature L0 output')
        self._budget()
        if not return_graph:
            return output
        origins = torch.as_tensor([a['origin'] for a in batch.audit], device=ids.device)
        xyz_native = xyz_local + origins[ids, None]
        radii = torch.stack([distance.masked_fill(~(eligible & (band == value)), -float('inf')).max(-1).values
                             for value in (1, 2, 3)], -1)
        graph = dict(features=features, xyz_native=xyz_native,
            xyz_mm=xyz_native * spacing[:, None], relative_xyz_mm=relative_mm,
            role=role, node_mask=mask, adjacency=adjacency, edge_kind=edge_kind,
            scene_pair=torch.arange(len(batch), device=ids.device).repeat(2),
            scene_side=torch.arange(2, device=ids.device).repeat_interleave(len(batch)),
            crop_index=ids, anchors_native=anchors_native,
            query_is_abstract=True, query_is_ct_sample=False,
            edge_note='Message-passing relation, not CT samples along a straight line or a vessel path',
            profile=asdict(self.profile), weight_sources=self.weight_sources,
            statistics=dict(organ_fine_pool_counts=crop_counts[ids],
                eligible_all_scale_pool_counts=eligible.sum(-1),
                unsupported_scale_counts=(fine_mask[..., None] & ~supported).sum(1),
                unsupported_any_scale_counts=(fine_mask & ~supported.all(-1)).sum(-1),
                band_pool_counts=counts, band_selected_counts=counts.clamp_max(self.profile.context_nodes_per_band),
                band_max_radius_mm=radii, query_pool_counts=query_counts,
                connected_before_mst=connected_before, mst_added_edges=mst_added,
                mst_marked_tree_edges=((edge_kind & 4) != 0).sum((1, 2)) // 2,
                unique_cnn_crops=len(batch.images), physical_pairs=len(batch),
                hard_selection_differentiable=False, fine_pool_stride=1))
        graph['statistics'].update(metric_statistics)
        if return_pool:
            # Existing tensors only: no second CNN/feature-pool calculation.
            # Callers must use these on-device for coverage, not serialize the
            # fine pool as a report or retain an autograd training graph.
            graph['coverage_pool'] = dict(features=pool.detach(),
                relative_mm=pool_mm.detach(), eligible=eligible, band=band)
        return output, graph
