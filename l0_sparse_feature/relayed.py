"""DEBUG comparison: retain actual organ paths between CNN-selected seeds.

The original relational comparator remains available. This path keeps its
CNN, seeds, radii, three 128D typed-SAGE layers and role attention. Only node
retention and mandatory context edges change. A native six-neighbour parent
chain supplies a path witness; it is not a vessel annotation. The dense native
substrate is used for sampling, never for neural message passing.
"""
import torch

from l0_sparse_feature.model import _select_context
from l0_sparse_feature.relational import (
    V1RelationalSparseL0, _build_pair_edges, _weak_components,
)
from l0_sparse_feature.relay_sampling import retain_path_relays, verify_retained_paths


@torch.no_grad()
def build_relay_pair_edges(pair_xyz, pair_mask, profile, sampling, *, diagnostics=True):
    """Union radius-gated nearest3 edges and witnessed bidirectional relays.

Mandatory edges can increase incoming degree beyond three. This is explicit:
    nearest3 is the baseline selection, not a cap on the completed graph.
All mandatory nodes/edges are validated before the sparse mean is rebuilt.
"""
    base = _build_pair_edges(pair_xyz, pair_mask, profile, diagnostics=False)
    b, _, s = pair_mask.shape
    n, r = 2 * s, len(base['relation_names'])
    scene, target_slot, source_slot = sampling['mandatory_edges']
    if bool(((scene < 0) | (scene >= 2*b) | (target_slot < 0) |
             (target_slot >= s-1) | (source_slot < 0) | (source_slot >= s-1)).any()):
        raise IndexError('Mandatory relay indices are outside the actual physical batch')
    side, pair = torch.div(scene, b, rounding_mode='floor'), scene.remainder(b)
    target, source = side*s + target_slot + 1, side*s + source_slot + 1
    mandatory = torch.stack((pair, side, target, source))
    mask = pair_mask.reshape(b, n)
    if not bool((mask[pair, target] & mask[pair, source]).all()):
        raise ValueError('Mandatory relay endpoints cannot be padding')
    xyz = pair_xyz.reshape(b, n, 3)
    lengths = torch.linalg.vector_norm(xyz[pair, target] - xyz[pair, source], dim=-1)
    witness = sampling['edge_path_length_mm']
    if witness.shape != lengths.shape or not bool((torch.isfinite(witness) &
            (witness > 0) & (witness <= profile.context_neighbor_radius_mm)).all()):
        raise ValueError('Mandatory relay lacks a positive bounded parent-path witness')
    verify_retained_paths(sampling)
    native_delta = (sampling['fine_xyz_local'][scene,sampling['selection_index'][scene,target_slot]]
                  -sampling['fine_xyz_local'][scene,sampling['selection_index'][scene,source_slot]])
    physical_delta = native_delta*sampling['fine_spacing_mm'][scene]
    if not bool(torch.isclose(physical_delta,xyz[pair,target]-xyz[pair,source],rtol=1e-6,atol=1e-5).all()):
        raise ValueError('Mandatory native path does not match the actual paired graph coordinates')
    if not bool((torch.isfinite(lengths) & (lengths <= profile.context_neighbor_radius_mm)
                 & (target != source)).all()):
        raise ValueError('Mandatory relay violates the unchanged physical context radius')
    packed = torch.cat((base['edge_index'], mandatory), -1)
    keys = (((packed[0]*r + packed[1])*n + packed[2])*n + packed[3])
    keys = keys.unique(sorted=True)
    source = keys.remainder(n)
    target = torch.div(keys, n, rounding_mode='floor').remainder(n)
    relation = torch.div(keys, n*n, rounding_mode='floor').remainder(r)
    pair = torch.div(keys, r*n*n, rounding_mode='floor')
    edges = torch.stack((pair, relation, target, source))
    rows, columns = (pair*r+relation)*n+target, pair*n+source
    indegree = torch.bincount(rows, minlength=b*r*n).reshape(b,r,n)
    outdegree = torch.bincount((pair*r+relation)*n+source, minlength=b*r*n).reshape(b,r,n)
    adjacency = torch.sparse_coo_tensor(torch.stack((rows,columns)),
        indegree.flatten()[rows].float().reciprocal(), (b*r*n,b*n), device=pair_xyz.device).coalesce()
    base.update(edge_index=edges, mean_adjacency=adjacency, in_degree=indegree, out_degree=outdegree,
        edge_distance_mm=torch.linalg.vector_norm(xyz[pair,target]-xyz[pair,source],dim=-1),
        unmatched_target_mask=base['relation_target_mask'] & (indegree == 0),
        unmatched_source_mask=base['relation_source_mask'] & (outdegree == 0))
    if diagnostics:
        base.update(_weak_components(edges, mask))
    base['mandatory_edge_audit'] = dict(
        nearest_policy='Radius-gated nearest3/3/1/3 PLUS witnessed bidirectional context-path edges; total incoming degree is not capped at three',
        mandatory_path_edges=True, mandatory_directed_edges=mandatory.shape[1],
        distinct_mandatory_directed_edges=torch.unique(
            (((mandatory[0]*r+mandatory[1])*n+mandatory[2])*n+mandatory[3])).numel(),
        added_directed_edges=edges.shape[1]-packed.shape[1]+mandatory.shape[1],
        hop_expansion=False, mst=False, out_of_radius_fallback=False,
        path_substrate='Native organ-supported six-neighbour voxels; each compressed parent-chain has cumulative physical length <=6mm',
        anatomical_path_claim=False, seed_budget_is_final_node_cap=False)
    base['mandatory_edge_audit']['native_parent_path_recomputed_on_GPU'] = True
    return base


class RelayedV1SparseL0(V1RelationalSparseL0):
    """Untrained DEBUG sampler correction; no production readiness marker."""

    def _select_nodes(self, pool_xyz, pool_mm, pool, eligible, band, distance, spacing, *, scene_keys):
        # Identical unique CNN crop + original anchor means identical scene.
        # Donor repeats in physical32 are selected/traversed once, then expanded
        # back into the original disjoint pair batch. No observations are lost.
        keys, inverse = torch.unique(scene_keys, dim=0, sorted=True, return_inverse=True)
        g, u = len(eligible), len(keys)
        first = torch.full((u,),g,device=inverse.device,dtype=torch.long)
        first.scatter_reduce_(0,inverse,torch.arange(g,device=inverse.device),reduce='amin')
        for value in (pool_xyz, pool_mm, eligible, band, distance, spacing):
            if not torch.equal(value, value[first][inverse]):
                raise ValueError('Shared crop/anchor key does not identify identical native scene geometry/support')
        seeds, seed_mask, statistics = _select_context(
            pool_mm[first], pool[first], eligible[first], band[first], distance[first], self.profile)
        sampling = retain_path_relays(pool_xyz[first], eligible[first], band[first], distance[first], spacing[first],
            seeds, seed_mask, query_radius_mm=self.profile.query_radius_mm,
            context_radius_mm=self.profile.context_neighbor_radius_mm)
        # Explicit scene/edge expansion keeps shared work separate from the
        # actual physical batch. Typed edges are later scoped to each pair.
        for name in ('selection_index','selection_mask','roles','seed_mask','fine_parent',
                     'fine_reachable','fine_depth','fine_xyz_local','fine_eligible','fine_spacing_mm'):
            sampling[name] = sampling[name][inverse]
        mandatory = sampling['mandatory_edges'][:, :sampling['mandatory_edges'].shape[1]//2]
        scene, edge = (inverse[:,None] == mandatory[0][None]).nonzero(as_tuple=True)
        forward = torch.stack((scene,mandatory[1,edge],mandatory[2,edge]))
        sampling['mandatory_edges'] = torch.cat((forward,forward[[0,2,1]]),-1)
        for name in ('edge_path_length_mm','edge_path_steps'):
            sampling[name] = sampling[name][edge].repeat(2)
        roots = sampling['query_roots']
        scene, root = (inverse[:,None] == roots[0][None]).nonzero(as_tuple=True)
        sampling['query_roots'] = torch.stack((scene,roots[1,root]))
        audit = sampling['diagnostics']
        for name in ('eligible_pool_count','native_root_count','reached_pool_count','unreachable_pool_count',
                     'original_seed_count','reached_seed_count','unreachable_seed_count',
                     'sampling_connected_per_scene','final_context_count','added_relay_count','max_parent_depth'):
            audit[name] = audit[name][inverse]
        missing_scene, missing_slot = (sampling['seed_mask'] &
            ~sampling['fine_reachable'].gather(1,sampling['selection_index'])).nonzero(as_tuple=True)
        audit['unreachable_seed_indices'] = torch.stack((missing_scene,
            sampling['selection_index'][missing_scene,missing_slot]))
        # One distance for every expanded root, in its actual physical scene.
        audit['query_root_distance_mm'] = distance[
            sampling['query_roots'][0],sampling['selection_index'][
                sampling['query_roots'][0],sampling['query_roots'][1]]]
        statistics = {name:(value[inverse] if isinstance(value,torch.Tensor) else value)
                      for name,value in statistics.items()}
        audit.update(selection_unique_scenes=u, selection_physical_scene_occurrences=g,
                     duplicate_scene_reuse=g-u, exact_scene_geometry_support_verified=True)
        audit.update(seed_counts=audit['original_seed_count'], relay_counts=audit['added_relay_count'],
            final_context_counts=audit['final_context_count'],
            native6_substrate_unreachable_counts=audit['unreachable_seed_count'],
            fine_reachable_counts=audit['reached_pool_count'], bfs_wave_count=audit['BFS_waves'],
            max_compressed_path_length_mm=audit['max_geodesic_edge_length_mm'])
        return (sampling['selection_index'], sampling['selection_mask'], sampling['roles'],
                statistics, sampling)

    def _pair_edges(self, pair_xyz, pair_mask, sampling, *, diagnostics):
        return build_relay_pair_edges(pair_xyz, pair_mask, self.profile, sampling,
                                      diagnostics=diagnostics)
