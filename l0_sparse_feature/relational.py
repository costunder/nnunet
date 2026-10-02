"""Separate DEBUG L0 with sparse, directed v1-style contextual relations.

This comparison borrows the v1 context/interface/correspondence radii, not its
anatomical node meanings or full GAT architecture. A query is the organ-supported
CNN mean inside a 3 mm sphere at the *original* anchor. It is neither a sampled
tumor surface/interior node nor a CT sample at the anchor itself. Native crops,
organ masks, the eight-convolution [12, 24, 32] CNN, and paired fusion are copied
unchanged from the caller's reference. New graph weights are untrained.

Nodes are selected by the existing space+CNN per-shell hard FPS. The graph has
eight directed relations, batched without any cross-pair edges. Each receiving
node first applies its relation's physical-radius gate, then selects at most k
nearest eligible sources. A singleton query can therefore reach many context
targets inside 8 mm: k is an incoming-source ceiling, not an outgoing-edge cap.
No feature edges, MST, hop expansion, or out-of-radius repair are added. Missing
connections remain present in the diagnostic counts; samples are never dropped.
"""

import copy
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from l0_local_cnn.model import LocalBatch
from l0_regions.resident import check_verified
from l0_sparse_feature.model import (
    SparseFeatureCoverageError,
    _masked_corner_features,
    _select_context,
)
from tools.local_cnn_l0_probe import _geometry


RELATION_NAMES = (
    'source_context_neighbor',
    'target_context_neighbor',
    'source_query_to_context',
    'source_context_to_query',
    'target_query_to_context',
    'target_context_to_query',
    'source_context_to_target_context',
    'source_query_to_target_context',
)
NODE_ROLE_NAMES = (
    'donor_query', 'donor_near', 'donor_mid', 'donor_wide',
    'recipient_query', 'recipient_near', 'recipient_mid', 'recipient_wide',
)


@dataclass(frozen=True)
class V1RelationalSparseProfile:
    """Explicit approved DEBUG candidates, not a production default profile.

    The required quota and sphere/shell radii must be supplied by the caller.
    Fixed relation radii and nearest-source counts are named fields so reports
    expose the complete graph contract. They cannot be silently changed.
    """

    context_nodes_per_band: int
    query_radius_mm: float
    near_radius_mm: float
    mid_radius_mm: float
    context_neighbor_radius_mm: float = 6.0
    query_context_radius_mm: float = 8.0
    cross_context_radius_mm: float = 5.0
    cross_query_radius_mm: float = 8.0
    context_neighbors: int = 3
    query_context_neighbors: int = 3
    cross_context_neighbors: int = 1
    cross_query_neighbors: int = 3
    spatial_metric_weight: float = 1.0
    feature_metric_weight: float = 1.0
    hidden_dim: int = 128
    sage_layers: int = 3
    initialization_seed: int = 42
    debug: bool = True

    def validate(self):
        if self.debug is not True:
            raise ValueError('The relational sparse L0 profile is DEBUG-only')
        if (type(self.context_nodes_per_band) is not int
                or self.context_nodes_per_band not in (16, 32, 64)):
            raise ValueError('Explicit DEBUG quota must be 16/32/64 per band, or 48/96/192 context nodes per branch')
        radii = (self.query_radius_mm, self.near_radius_mm, self.mid_radius_mm,
                 self.context_neighbor_radius_mm, self.query_context_radius_mm,
                 self.cross_context_radius_mm, self.cross_query_radius_mm)
        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in radii)
                or radii != (3.0, 5.0, 10.0, 6.0, 8.0, 5.0, 8.0)):
            raise ValueError('Reviewed DEBUG query/near/mid radii 3/5/10 mm and relation radii 6/8/5/8 mm are required')
        counts = (self.context_neighbors, self.query_context_neighbors,
                  self.cross_context_neighbors, self.cross_query_neighbors)
        if any(type(value) is not int for value in counts) or counts != (3, 3, 1, 3):
            raise ValueError('Reviewed incoming nearest-source counts 3/3/1/3 are required')
        if (type(self.hidden_dim) is not int or type(self.sage_layers) is not int
                or (self.hidden_dim, self.sage_layers) != (128, 3)):
            raise ValueError('The separate DEBUG architecture retains three 128D SAGE layers')
        if type(self.initialization_seed) is not int or self.initialization_seed != 42:
            raise ValueError('The reviewed DEBUG comparison retains initialization seed42')
        weights = (self.spatial_metric_weight, self.feature_metric_weight)
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in weights):
            raise ValueError('Finite positive space/CNN selection metric weights are required')

    @property
    def relation_radii_mm(self):
        return (self.context_neighbor_radius_mm,) * 2 + (self.query_context_radius_mm,) * 4 + (
            self.cross_context_radius_mm, self.cross_query_radius_mm)

    @property
    def relation_neighbors(self):
        return (self.context_neighbors,) * 2 + (self.query_context_neighbors,) * 4 + (
            self.cross_context_neighbors, self.cross_query_neighbors)


@torch.no_grad()
def _weak_components(edge_index, node_mask):
    """Batched weak components of the actual directed graph, without repair.

    Integer minimum-label propagation treats each directed edge as undirected
    solely for this diagnostic. Every valid node, including an isolate, gets a
    component label. This routine is requested only for graph diagnostics.
    """
    b, n = node_mask.shape
    local = torch.arange(n, device=node_mask.device)[None].expand(b, -1)
    labels = torch.where(node_mask, local, n).reshape(-1)
    pair, _, target, source = edge_index
    source_flat, target_flat = pair * n + source, pair * n + target
    for _ in range(n):
        updated = labels.clone()
        updated.scatter_reduce_(0, target_flat, labels[source_flat], reduce='amin', include_self=True)
        updated.scatter_reduce_(0, source_flat, labels[target_flat], reduce='amin', include_self=True)
        if torch.equal(updated, labels):
            break
        labels = updated
    labels = labels.reshape(b, n)
    representatives = node_mask & (labels == local)
    count = representatives.sum(-1)
    return dict(component_labels=torch.where(node_mask, labels, -1),
                components=count, weak_connected=count == 1)


@torch.no_grad()
def _build_pair_edges(relative_xyz_mm, node_mask, profile, *, diagnostics=True):
    """Return exact typed edges and a relation-preserving sparse mean operator.

    Inputs are [B, 2, S, 3] and [B, 2, S], with query at branch slot zero.
    All physical coordinates are relative to the original branch anchor, so
    correspondence compares donor/recipient positions after anchor alignment.
    Edge index is [4, E] ordered (pair, relation, TARGET row, SOURCE column).
    Mean adjacency is a single sparse COO [B*R*N, B*N], N=2*S. Its row ordering
    preserves [B,R,N] after SpMM, hence never merges relation-specific means.
    The helper permits small explicit UNIT fixtures; the model packs the exact
    profile quota. No receiving node acquires a nearest source outside radius.
    """
    if not isinstance(profile, V1RelationalSparseProfile):
        raise TypeError('Explicit V1RelationalSparseProfile required')
    profile.validate()
    if (relative_xyz_mm.ndim != 4 or relative_xyz_mm.shape[1] != 2
            or relative_xyz_mm.shape[-1] != 3 or relative_xyz_mm.shape[2] < 2
            or node_mask.shape != relative_xyz_mm.shape[:-1]
            or node_mask.dtype != torch.bool or node_mask.device != relative_xyz_mm.device
            or not relative_xyz_mm.is_floating_point() or not len(node_mask)):
        raise ValueError('Physical pair coordinates [B,2,S,3] and boolean [B,2,S] mask required')
    if not bool(node_mask[:, :, 0].all()):
        raise SparseFeatureCoverageError('Every original donor/recipient query must remain represented')
    if not bool(torch.isfinite(relative_xyz_mm[node_mask]).all()):
        raise FloatingPointError('Nonfinite valid physical node coordinates')
    b, _, s = node_mask.shape
    n, r = 2 * s, len(RELATION_NAMES)
    alive = node_mask.reshape(b, n)
    coordinates = torch.where(node_mask[..., None], relative_xyz_mm, 0).reshape(b, n, 3)
    # Disable the cdist matrix-multiplication shortcut: at coincident anchors,
    # cancellation in ||a||²+||b||²-2a.b must not change a strict radius gate.
    distances = torch.cdist(coordinates, coordinates, compute_mode='donot_use_mm_for_euclid_dist')
    index = torch.arange(n, device=alive.device)
    donor_query = alive & (index == 0)
    recipient_query = alive & (index == s)
    donor_context = alive & ((index > 0) & (index < s))
    recipient_context = alive & (index > s)
    source_masks = (donor_context, recipient_context, donor_query, donor_context,
                    recipient_query, recipient_context, donor_context, donor_query)
    target_masks = (donor_context, recipient_context, donor_context, donor_query,
                    recipient_context, recipient_query, recipient_context, recipient_context)
    relation_source_mask = torch.stack(source_masks, 1)
    relation_target_mask = torch.stack(target_masks, 1)
    edge_parts, candidates = [], []
    eye = torch.eye(n, device=alive.device, dtype=torch.bool)[None]
    for relation, (sources, targets, radius, k) in enumerate(zip(
            source_masks, target_masks, profile.relation_radii_mm, profile.relation_neighbors)):
        legal = targets[:, :, None] & sources[:, None, :] & (distances <= radius) & ~eye
        candidates.append(legal.sum(-1))
        nearest = distances.masked_fill(~legal, float('inf')).argsort(dim=-1, stable=True)[..., :min(k, n)]
        selected = legal.gather(-1, nearest)
        pair, target, slot = selected.nonzero(as_tuple=True)
        source = nearest[pair, target, slot]
        edge_parts.append(torch.stack((pair, torch.full_like(pair, relation), target, source)))
    edges = torch.cat(edge_parts, -1)
    # Canonical pair/relation/target/source ordering also makes the diagnostic
    # COO exactly match nonzero coordinates of the typed adjacency.
    edge_key = (((edges[0] * r + edges[1]) * n + edges[2]) * n + edges[3])
    edges = edges[:, edge_key.argsort(stable=True)]
    pair, relation, target, source = edges
    row = (pair * r + relation) * n + target
    column = pair * n + source
    source_row = (pair * r + relation) * n + source
    in_degree = torch.bincount(row, minlength=b * r * n).reshape(b, r, n)
    out_degree = torch.bincount(source_row, minlength=b * r * n).reshape(b, r, n)
    values = in_degree.reshape(-1)[row].to(torch.float32).reciprocal()
    mean = torch.sparse_coo_tensor(torch.stack((row, column)), values,
                                  (b * r * n, b * n), device=alive.device).coalesce()
    edge_distance = distances[pair, target, source]
    unmatched_target = relation_target_mask & (in_degree == 0)
    unmatched_source = relation_source_mask & (out_degree == 0)
    result = dict(edge_index=edges, mean_adjacency=mean,
        in_degree=in_degree, out_degree=out_degree,
        relation_source_mask=relation_source_mask, relation_target_mask=relation_target_mask,
        candidate_counts=torch.stack(candidates, 1),
        unmatched_target_mask=unmatched_target, unmatched_source_mask=unmatched_source,
        edge_distance_mm=edge_distance, node_mask=alive,
        relation_names=RELATION_NAMES, relation_radii_mm=profile.relation_radii_mm,
        relation_neighbors=profile.relation_neighbors)
    if diagnostics:
        result.update(_weak_components(edges, alive))
    return result


def _dense_pair_adjacency(edge_data):
    """Materialize the diagnostic [B,R,N,N], never the message-passing path."""
    b, r, n = edge_data['in_degree'].shape
    adjacency = torch.zeros((b, r, n, n), device=edge_data['node_mask'].device, dtype=torch.bool)
    pair, relation, target, source = edge_data['edge_index']
    adjacency[pair, relation, target, source] = True
    return adjacency


class _RelationalMeanSAGE(nn.Module):
    """Separate neighbor means/transforms; root enters once through residual."""

    root_weight = False

    def __init__(self, hidden=128, relations=8):
        super().__init__()
        self.hidden_dim, self.relations = hidden, relations
        self.neighbor_weight = nn.Parameter(torch.empty(relations, hidden, hidden))
        for weight in self.neighbor_weight:
            nn.init.xavier_uniform_(weight)
        self.norm = nn.LayerNorm(hidden)

    def relation_means(self, x, mean_adjacency):
        if x.ndim != 3 or x.shape[-1] != self.hidden_dim or mean_adjacency.layout != torch.sparse_coo:
            raise ValueError('Batched hidden states and the typed sparse COO mean operator required')
        b, n, d = x.shape
        if mean_adjacency.shape != (b * self.relations * n, b * n):
            raise ValueError('Typed sparse mean shape differs from the physical pair batch')
        means = torch.sparse.mm(mean_adjacency.to(dtype=x.dtype), x.reshape(b * n, d))
        return means.reshape(b, self.relations, n, d)

    def forward(self, x, edge_data, node_mask):
        means = self.relation_means(x, edge_data['mean_adjacency'])
        messages = torch.einsum('brni,rio->bno', means, self.neighbor_weight)
        return torch.where(node_mask[..., None], F.silu(self.norm(x + messages)), 0)


class _RoleShellAttentionPool(nn.Module):
    """Eight role-specific learned value pools with six context scorers.

    Each branch query contains one abstract node. Its attention is exactly one;
    learning a query score would create a parameter with identically zero
    gradient. Its own learned value transform is active instead. Near/mid/wide
    pools have distinct donor/recipient scorers and value transforms.
    """

    def __init__(self, hidden=128):
        super().__init__()
        self.value_weight = nn.Parameter(torch.empty(8, hidden, hidden))
        self.value_bias = nn.Parameter(torch.zeros(8, hidden))
        self.score_weight = nn.Parameter(torch.empty(6, hidden))
        for weight in self.value_weight:
            nn.init.xavier_uniform_(weight)
        nn.init.normal_(self.score_weight, std=hidden ** -0.5)
        self.register_buffer('context_roles', torch.tensor((1, 2, 3, 5, 6, 7)), persistent=False)

    def forward(self, h, node_mask, role):
        if h.shape[:2] != node_mask.shape or role.shape != node_mask.shape:
            raise ValueError('Batched pair states, mask and explicit global roles required')
        all_roles = torch.arange(8, device=h.device)
        members = node_mask[:, None] & (role[:, None] == all_roles[None, :, None])
        if not bool(members.any(-1).all()):
            raise SparseFeatureCoverageError('Role/shell attention cannot replace an empty group with zero')
        context_scores = torch.einsum('bnd,rd->brn', h, self.score_weight)
        scores = h.new_zeros((len(h), 8, h.shape[1])).index_copy(1, self.context_roles, context_scores)
        weights = scores.masked_fill(~members, -float('inf')).softmax(-1)
        pooled_h = torch.bmm(weights, h)
        values = torch.einsum('bri,rio->bro', pooled_h, self.value_weight) + self.value_bias
        return values, weights


class V1RelationalSparseL0(nn.Module):
    """Paired organ-only DEBUG graph; production L0/L1/L2 are untouched."""

    def __init__(self, reference_local, profile, budget=None):
        super().__init__()
        if not isinstance(profile, V1RelationalSparseProfile):
            raise TypeError('Explicit V1RelationalSparseProfile required')
        profile.validate()
        if (reference_local.config['channels'] != [12, 24, 32]
                or reference_local.config['convolutions'] != [2, 3, 3]
                or reference_local.config['hidden_dim'] != 128):
            raise ValueError('Reference CNN must retain [12,24,32], eight convolutions and 128D fusion')
        self.profile = profile
        self.margin_mm = float(reference_local.config['margin_mm'])
        if not math.isfinite(self.margin_mm) or self.margin_mm <= 0:
            raise ValueError('The unchanged reference crop margin must be finite and positive')
        self.resource_budget = budget
        self.cnn = copy.deepcopy(reference_local.cnn)
        self.fuse = copy.deepcopy(reference_local.fuse)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(profile.initialization_seed)
            self.node_project = nn.Linear(68 + 3, 128)
            self.role_embedding = nn.Embedding(8, 128)
            self.blocks = nn.ModuleList([_RelationalMeanSAGE(128, 8) for _ in range(3)])
            self.attention_pool = _RoleShellAttentionPool(128)
            self.scene_project = nn.ModuleList([
                nn.Sequential(nn.Linear(4 * 128, 128), nn.LayerNorm(128), nn.SiLU())
                for _ in range(2)])
        self.weight_sources = dict(CNN='exact deep copy of caller reference_local.cnn',
            paired_fusion='exact deep copy of caller reference_local.fuse',
            graph=f'fresh seed{profile.initialization_seed} initialization; untrained DEBUG v1-style contextual comparison')
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
            raise ValueError('Relational sparse numerical path requires CUDA; no CPU fallback')
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
        counts = torch.stack([(eligible & (band == value)).sum(-1) for value in (1, 2, 3)], -1)
        if not bool((counts > 0).all()):
            raise SparseFeatureCoverageError('Empty near/mid/wide organ-supported band; no node/sample drop or radius fallback')
        query_mask = eligible & (distance <= self.profile.query_radius_mm)
        query_counts = query_mask.sum(-1)
        if not bool((query_counts > 0).all()):
            raise SparseFeatureCoverageError('Original anchor sphere lacks all-scale organ support; anchor/radius unchanged')
        self._budget()
        selection, context_mask, metric_statistics = _select_context(
            pool_mm, pool, eligible, band, distance, self.profile)
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
            raise SparseFeatureCoverageError('Query sphere has unsupported CNN scale')
        query_features = query_values.new_zeros((len(ids), 68)).index_add(0, query_scene, query_values)
        query_features = query_features / query_counts[:, None]
        features = torch.cat((query_features[:, None], selected_features), 1)
        mask = torch.cat((torch.ones((len(ids), 1), device=ids.device, dtype=torch.bool), context_mask), 1)
        shell = torch.cat((torch.zeros(1, device=ids.device, dtype=torch.long),
            torch.arange(1, 4, device=ids.device).repeat_interleave(self.profile.context_nodes_per_band)))
        role = shell[None].expand(len(ids), -1)
        xyz_local = torch.cat((anchors_local[:, None], selected_xyz), 1)
        relative_mm = (xyz_local - anchors_local[:, None]) * spacing[:, None]
        features = torch.where(mask[..., None], features, 0)
        b, s = len(batch), len(shell)
        pair_mask = mask.reshape(2, b, s).transpose(0, 1).reshape(b, 2 * s)
        pair_xyz = relative_mm.reshape(2, b, s, 3).transpose(0, 1)
        edges = _build_pair_edges(pair_xyz, pair_mask.reshape(b, 2, s),
                                 self.profile, diagnostics=return_graph)
        pair_role = torch.cat((shell, shell + 4))[None].expand(b, -1)
        global_role = torch.cat((role[:b], role[b:] + 4))
        self._budget()
        scene_h = self.node_project(torch.cat((features, relative_mm / self.margin_mm), -1))
        scene_h = scene_h + self.role_embedding(global_role)
        h = scene_h.reshape(2, b, s, 128).transpose(0, 1).reshape(b, 2 * s, 128)
        h = torch.where(pair_mask[..., None], h, 0)
        for block in self.blocks:
            h = block(h, edges, pair_mask)
        pooled, pooling_weights = self.attention_pool(h, pair_mask, pair_role)
        # Two batched branch projections, never a per-pair GPU forward.
        d = self.scene_project[0](pooled[:, :4].reshape(b, 4 * 128))
        r = self.scene_project[1](pooled[:, 4:].reshape(b, 4 * 128))
        scene = torch.cat((d, r))
        output = self.fuse(torch.cat((d, r, r - d, r * d), -1))
        if output.shape != (b, 128) or not bool(torch.isfinite(output).all()):
            raise FloatingPointError('Invalid relational sparse L0 output')
        self._budget()
        if not return_graph:
            return output
        origins = torch.as_tensor([a['origin'] for a in batch.audit], device=ids.device)
        xyz_native = xyz_local + origins[ids, None]
        radii = torch.stack([distance.masked_fill(~(eligible & (band == value)), -float('inf')).max(-1).values
                             for value in (1, 2, 3)], -1)
        pair_adjacency = _dense_pair_adjacency(edges)
        scene_adjacency = torch.cat((pair_adjacency[:, :, :s, :s].any(1),
                                    pair_adjacency[:, :, s:, s:].any(1)))
        relation_flags = (1 << torch.arange(8, device=ids.device, dtype=torch.int16))[None, :, None, None]
        pair_kind = (pair_adjacency.to(torch.int16) * relation_flags).sum(1).to(torch.int16)
        scene_kind = torch.cat((pair_kind[:, :s, :s], pair_kind[:, s:, s:]))
        in_degree, out_degree = edges['in_degree'], edges['out_degree']
        relation_radii = features.new_tensor(self.profile.relation_radii_mm)
        edge_pair, edge_relation, edge_target, edge_source = edges['edge_index']
        endpoint_gate = (edges['relation_target_mask'][edge_pair, edge_relation, edge_target]
                         & edges['relation_source_mask'][edge_pair, edge_relation, edge_source])
        radius_gate = edges['edge_distance_mm'] <= relation_radii[edge_relation]
        violations = torch.bincount(edge_relation[~(endpoint_gate & radius_gate)], minlength=8)
        if bool((violations != 0).any()):
            raise RuntimeError('Actual relation edges violated the explicit role/radius gate')
        graph = dict(features=features, xyz_native=xyz_native,
            xyz_mm=xyz_native * spacing[:, None], relative_xyz_mm=relative_mm,
            role=role, global_role=global_role, role_names=NODE_ROLE_NAMES,
            node_mask=mask, adjacency=scene_adjacency, edge_kind=scene_kind,
            scene_pair=torch.arange(b, device=ids.device).repeat(2),
            scene_side=torch.arange(2, device=ids.device).repeat_interleave(b),
            crop_index=ids, anchors_native=anchors_native,
            query_is_abstract=True, query_is_ct_sample=False,
            query_note='3 mm organ-supported CNN mean at the original anchor; not an anatomical tumor surface/interior node',
            edge_note='Directed message-passing relation, not CT samples along a straight line or a vessel path',
            profile=asdict(self.profile), weight_sources=self.weight_sources,
            pair_adjacency=pair_adjacency, pair_edge_index=edges['edge_index'],
            edge_data=edges,
            pair_edge_distance_mm=edges['edge_distance_mm'],
            pair_node_role=pair_role, pair_node_mask=pair_mask,
            pair_node_branch=(torch.arange(2, device=ids.device).repeat_interleave(s)[None].expand(b, -1)),
            relation_names=RELATION_NAMES,
            pair_adjacency_orientation='[pair, relation, target-row, source-column]',
            joint_hidden=h, scene_readout=scene, pooled_roles=pooled, pool_weights=pooling_weights,
            relation_edge_audit=dict(
                radii_mm=relation_radii, nearest_sources=tuple(self.profile.relation_neighbors),
                edge_counts=in_degree.sum(-1), max_incoming=in_degree.max(-1).values,
                max_outgoing=out_degree.max(-1).values, role_radius_violations=violations,
                radius_filter='distance <= radius BEFORE nearest-source selection; never an out-of-radius fallback',
                nearest_policy='At most k incoming sources per receiving node; singleton query can reach every in-radius target',
                cross_coordinate_policy='Physical coordinates relative to each unchanged original anchor',
                self_edges=False, reverse_cross_relations=False, feature_edges=False,
                mst=False, hop_expansion=False, out_of_radius_fallback=False,
                cross_pair_edges=False, missing_connection_policy='Report unmatched targets/sources and weak components; retain every node and pair'),
            statistics=dict(organ_fine_pool_counts=crop_counts[ids],
                eligible_all_scale_pool_counts=eligible.sum(-1),
                unsupported_scale_counts=(fine_mask[..., None] & ~supported).sum(1),
                unsupported_any_scale_counts=(fine_mask & ~supported.all(-1)).sum(-1),
                band_pool_counts=counts,
                band_selected_counts=counts.clamp_max(self.profile.context_nodes_per_band),
                band_max_radius_mm=radii, query_pool_counts=query_counts,
                relation_edge_counts=in_degree.sum(-1),
                relation_max_incoming=in_degree.max(-1).values,
                relation_max_outgoing=out_degree.max(-1).values,
                relation_in_degree=in_degree, relation_out_degree=out_degree,
                relation_candidate_counts=edges['candidate_counts'],
                relation_unmatched_target_counts=edges['unmatched_target_mask'].sum(-1),
                relation_unmatched_source_counts=edges['unmatched_source_mask'].sum(-1),
                relation_unmatched_target_mask=edges['unmatched_target_mask'],
                relation_unmatched_source_mask=edges['unmatched_source_mask'],
                pair_components=edges['components'], pair_component_labels=edges['component_labels'],
                pair_weak_connected=edges['weak_connected'],
                pair_isolated_node_counts=(pair_mask & ((in_degree + out_degree).sum(1) == 0)).sum(-1),
                cross_context_matched_target_counts=(in_degree[:, 6] > 0).sum(-1),
                cross_query_matched_target_counts=(in_degree[:, 7] > 0).sum(-1),
                unique_cnn_crops=len(batch.images), physical_pairs=b,
                hard_selection_differentiable=False, fine_pool_stride=1))
        graph['statistics'].update(metric_statistics)
        if return_pool:
            # Existing on-device tensors only; callers must not retain/serialize
            # full fine pools across all cases. These are not training inputs.
            graph['coverage_pool'] = dict(features=pool.detach(),
                relative_mm=pool_mm.detach(), eligible=eligible, band=band)
        return output, graph
