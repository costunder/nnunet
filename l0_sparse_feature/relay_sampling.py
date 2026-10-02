"""CUDA-only retention of real native-grid paths between sparse context seeds.

This is a separate diagnostic sampling substrate. It preserves every original
space/CNN-selected seed and adds actual organ/all-scale-supported voxels as
relays. Its parent forest uses axis-adjacent native voxels whose physical step
does not exceed the existing context radius. A compressed relay edge has a
recorded supported parent path of cumulative physical length <= that radius.
It is never a direct bridge through background, a vessel edge, or a new kNN
relation. Seeds outside the sphere-rooted substrate are retained and reported.
Native six-neighbor nonreachability is NOT proof that the full physical-radius
graph is disconnected.
"""

import math

import torch


def _decision(condition, message):
    if not bool(condition):
        raise ValueError(message)


@torch.no_grad()
def retain_path_relays(pool_xyz_native, eligible, band, distance_mm, spacing_mm,
                       seed_index, seed_mask, *, query_radius_mm=3.0,
                       context_radius_mm=6.0, measure=False):
    """Keep seeds and compressed native parent paths, batched across scenes.

    Inputs use LOCAL integer native coordinates [G,M,3], boolean eligibility,
    physical distance to the unchanged anchor, actual shell IDs 1/2/3, native
    spacing [G,3], and original pool-slot seed IDs [G,Q]. Returned context slot
    IDs exclude the model's abstract query slot. ``mandatory_edges`` is
    [scene,target_context_slot,source_context_slot], in both directions. It must
    be unioned with the existing context relation after radius filtering rather
    than passed through another nearest-k truncation. All original seeds are
    retained, including the explicitly flagged unreachable ones. No sample is
    admitted to production here and no node ceiling or fallback is introduced.
    """
    if (not isinstance(pool_xyz_native, torch.Tensor)
            or pool_xyz_native.device.type != 'cuda'):
        raise ValueError('Native path retention requires CUDA; no CPU fallback')
    device = pool_xyz_native.device
    tensors = (eligible, band, distance_mm, spacing_mm, seed_index, seed_mask)
    if any(not isinstance(x, torch.Tensor) or x.device != device for x in tensors):
        raise ValueError('All native path inputs must share one CUDA device')
    if (pool_xyz_native.ndim != 3 or pool_xyz_native.shape[-1] != 3
            or not pool_xyz_native.is_floating_point()
            or not pool_xyz_native.shape[0] or not pool_xyz_native.shape[1]):
        raise ValueError('Nonempty floating native pool [G,M,3] required')
    g, m, _ = pool_xyz_native.shape
    if (eligible.shape != (g, m) or eligible.dtype != torch.bool
            or band.shape != (g, m) or band.dtype != torch.long
            or distance_mm.shape != (g, m) or not distance_mm.is_floating_point()
            or spacing_mm.shape != (g, 3) or not spacing_mm.is_floating_point()
            or seed_index.ndim != 2 or seed_index.shape[0] != g
            or seed_index.dtype != torch.long or seed_mask.shape != seed_index.shape
            or seed_mask.dtype != torch.bool or not seed_index.shape[1]):
        raise ValueError('Pool masks, shells, spacing, and seed shapes/dtypes differ')
    for value in (query_radius_mm, context_radius_mm):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Explicit finite positive physical radii required')
    if type(measure) is not bool:
        raise ValueError('Explicit boolean timing switch required')
    _decision(eligible.any(-1).all(), 'Every scene requires supported native voxels')
    xyz = torch.where(eligible[..., None], pool_xyz_native, 0)
    _decision((torch.isfinite(xyz) & (xyz == xyz.round()) & (xyz >= 0)).all(),
              'Eligible pool coordinates must be finite nonnegative native integers')
    _decision((torch.isfinite(spacing_mm) & (spacing_mm > 0)).all(),
              'Native physical spacing must be finite and positive')
    _decision((torch.isfinite(distance_mm[eligible]) & (distance_mm[eligible] >= 0)).all(),
              'Eligible anchor distances must be finite and nonnegative')
    _decision(((band[eligible] >= 1) & (band[eligible] <= 3)).all(),
              'Eligible native voxels require actual near/mid/wide shell IDs')
    _decision(seed_mask.any(-1).all(), 'Each scene requires an original context seed')
    _decision(((seed_index[seed_mask] >= 0) & (seed_index[seed_mask] < m)).all(),
              'Original seed index outside the supplied native pool')
    safe_seed = torch.where(seed_mask, seed_index, 0)
    _decision((eligible.gather(1, safe_seed) | ~seed_mask).all(),
              'An original context seed lacks organ/all-scale support')

    events = []
    if measure:
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
    coordinates = xyz.long()
    # Linear keys index the exact sparse eligible pool. They do not allocate a
    # full dense volume or silently substitute a regular lattice for CT nodes.
    shape = coordinates.amax((0, 1)) + 1
    strides = torch.stack((shape[1] * shape[2], shape[2], shape.new_ones(())))
    maximum_key = torch.iinfo(torch.long).max
    keys = (coordinates * strides).sum(-1).masked_fill(~eligible, maximum_key)
    sorted_keys, order = keys.sort(dim=-1, stable=True)
    duplicate = (sorted_keys[:, 1:] == sorted_keys[:, :-1]) & (sorted_keys[:, 1:] != maximum_key)
    _decision(~duplicate.any(), 'Duplicate eligible native pool coordinate')
    rows = torch.arange(g, device=device)[:, None]
    seeded = torch.zeros((g, m), device=device, dtype=torch.bool)
    seed_scene, seed_slot = seed_mask.nonzero(as_tuple=True)
    seeded[seed_scene, safe_seed[seed_scene, seed_slot]] = True
    _decision((seeded.sum(-1) == seed_mask.sum(-1)).all(), 'Duplicate original context seed')

    neighbor_parts = []
    for axis in range(3):
        for sign in (-1, 1):
            offset = torch.zeros(3, device=device, dtype=torch.long)
            offset[axis] = sign
            neighbor_xyz = coordinates + offset
            legal = (eligible & ((neighbor_xyz >= 0) & (neighbor_xyz < shape)).all(-1)
                     & (spacing_mm[:, axis, None] <= context_radius_mm))
            neighbor_key = (neighbor_xyz * strides).sum(-1).masked_fill(~legal, maximum_key)
            location = torch.searchsorted(sorted_keys, neighbor_key, right=False).clamp_max(m - 1)
            exists = legal & (sorted_keys.gather(1, location) == neighbor_key)
            neighbor_parts.append(torch.where(exists, order.gather(1, location), -1))
    neighbors = torch.stack(neighbor_parts, -1)
    valid_neighbor = neighbors >= 0
    safe_neighbor = neighbors.clamp_min(0)
    roots = eligible & (distance_mm <= query_radius_mm)
    _decision(roots.any(-1).all(), 'Original anchor sphere lacks native all-scale support')
    parent = torch.full((g, m), -1, device=device, dtype=torch.long)
    depth = torch.where(roots, 0, -1)
    frontier = roots
    waves = 0
    # One synchronous wave handles every scene and every node. No per-scene
    # Python traversal, CPU graph search, random fallback, or capped depth.
    while bool(frontier.any()):
        frontier_neighbors = frontier[rows[..., None], safe_neighbor] & valid_neighbor
        candidate_parent = neighbors.masked_fill(~frontier_neighbors, m).min(-1).values
        added = eligible & (depth < 0) & (candidate_parent < m)
        if not bool(added.any()):
            break
        parent = torch.where(added, candidate_parent, parent)
        waves += 1
        depth = torch.where(added, waves, depth)
        frontier = added
    reachable = depth >= 0
    if measure:
        events[1].record()

    # Furthest parent on the ACTUAL axis-supported path within cumulative
    # context-radius length. Euclidean endpoint proximity alone is insufficient.
    node_ids = torch.arange(m, device=device)[None].expand(g, -1)
    jump = node_ids.clone()
    path_length = distance_mm.new_zeros((g, m))
    path_steps = torch.zeros((g, m), device=device, dtype=torch.long)
    active = reachable & (parent >= 0)
    while bool(active.any()):
        next_parent = parent.gather(1, jump)
        valid_parent = next_parent >= 0
        safe_parent = next_parent.clamp_min(0)
        here = coordinates.gather(1, jump[..., None].expand(-1, -1, 3))
        there = coordinates.gather(1, safe_parent[..., None].expand(-1, -1, 3))
        step = torch.linalg.vector_norm((here - there).to(spacing_mm.dtype) * spacing_mm[:, None], dim=-1)
        advance = active & valid_parent & (path_length + step <= context_radius_mm)
        jump = torch.where(advance, next_parent, jump)
        path_length = torch.where(advance, path_length + step, path_length)
        path_steps += advance.long()
        active = advance
    _decision(((jump != node_ids) | ~reachable | roots).all(),
              'A supported parent step cannot fit the unchanged context radius')
    retained = seeded.clone()
    frontier = seeded & reachable
    compression_waves = 0
    while bool(frontier.any()):
        scene, node = frontier.nonzero(as_tuple=True)
        target = jump[scene, node]
        added = torch.zeros_like(retained)
        added[scene, target] = True
        added &= ~retained
        retained |= added
        frontier = added
        compression_waves += 1
    _decision((retained | ~seeded).all(), 'Internal path retention lost an original seed')
    if measure:
        events[2].record()

    counts = retained.sum(-1)
    width = int(counts.max())  # Dynamic padded allocation, never a node ceiling.
    selection = torch.zeros((g, width), device=device, dtype=torch.long)
    selection_mask = torch.zeros((g, width), device=device, dtype=torch.bool)
    inverse = torch.full((g, m), -1, device=device, dtype=torch.long)
    scene, node = retained.nonzero(as_tuple=True)
    slot = retained.cumsum(-1)[scene, node] - 1
    selection[scene, slot] = node
    selection_mask[scene, slot] = True
    inverse[scene, node] = slot
    roles = torch.where(selection_mask, band.gather(1, selection), 0)
    selected_seed = selection_mask & seeded.gather(1, selection)
    has_parent_edge = reachable[scene, node] & (jump[scene, node] != node)
    edge_scene = scene[has_parent_edge]
    target_pool = node[has_parent_edge]
    source_pool = jump[edge_scene, target_pool]
    target_slot = inverse[edge_scene, target_pool]
    source_slot = inverse[edge_scene, source_pool]
    _decision((source_slot >= 0).all(), 'Compressed ancestor absent from retained native nodes')
    forward_edges = torch.stack((edge_scene, target_slot, source_slot))
    backward_edges = torch.stack((edge_scene, source_slot, target_slot))
    mandatory_edges = torch.cat((forward_edges, backward_edges), -1)
    edge_length = path_length[edge_scene, target_pool].repeat(2)
    edge_steps = path_steps[edge_scene, target_pool].repeat(2)
    root_selected = roots[scene, node]
    query_roots = torch.stack((scene[root_selected], slot[root_selected]))
    unreachable_seed = seeded & ~reachable
    diagnostics = dict(
        substrate='native six-axis-neighbor parent forest; each eligible voxel and each physical step verified',
        nonreachability_scope='Not proof that the full 6mm radius graph is disconnected; native-axis substrate only',
        query_radius_mm=float(query_radius_mm), context_radius_mm=float(context_radius_mm),
        eligible_pool_count=eligible.sum(-1), native_root_count=roots.sum(-1),
        reached_pool_count=reachable.sum(-1), unreachable_pool_count=(eligible & ~reachable).sum(-1),
        original_seed_count=seeded.sum(-1), reached_seed_count=(seeded & reachable).sum(-1),
        unreachable_seed_count=unreachable_seed.sum(-1),
        unreachable_seed_indices=unreachable_seed.nonzero().transpose(0, 1),
        sampling_connected_per_scene=~unreachable_seed.any(-1),
        final_context_count=counts, added_relay_count=(retained & ~seeded).sum(-1),
        BFS_waves=waves, path_compression_waves=compression_waves,
        max_parent_depth=depth.max(-1).values,
        max_geodesic_edge_length_mm=edge_length.max() if len(edge_length) else distance_mm.new_zeros(()),
        query_root_distance_mm=distance_mm[scene[root_selected], node[root_selected]],
        edge_path_contract='Actual native parent chain, all-scale organ support at every step; cumulative length <= existing context radius; mandatory bidirectional relation edges are not truncated by k',
        unreachable_policy='Keep every original seed; report diagnostic-only nonreachability; no fake bridge, sample drop, radius change, or production admission',
        production_ready=False)
    if measure:
        events[3].record()
        events[-1].synchronize()
        diagnostics['GPU_seconds'] = dict(native_neighbor_BFS=events[0].elapsed_time(events[1]) / 1000,
            supported_path_compression=events[1].elapsed_time(events[2]) / 1000,
            packing=events[2].elapsed_time(events[3]) / 1000)
    return dict(selection_index=selection, selection_mask=selection_mask, roles=roles,
        seed_mask=selected_seed, mandatory_edges=mandatory_edges,
        edge_path_length_mm=edge_length, edge_path_steps=edge_steps,
        query_roots=query_roots, fine_parent=parent, fine_reachable=reachable,
        fine_depth=depth, fine_xyz_local=pool_xyz_native, fine_eligible=eligible,
        fine_spacing_mm=spacing_mm,
        diagnostics=diagnostics)


@torch.no_grad()
def verify_retained_paths(sampling):
    """Recompute every mandatory path from its parent/voxel/support witness.

    Scalar lengths alone are not accepted as a proof. This batched CUDA walk
    validates every native step and catches forged shortcuts, lost ancestors,
    background traversal, cycles and a falsely shortened length. It never
    generates repairs or removes failed edges.
    """
    selection, mask = sampling['selection_index'], sampling['selection_mask']
    edges = sampling['mandatory_edges']
    xyz, support, parent, depth, spacing = (sampling[name] for name in
        ('fine_xyz_local','fine_eligible','fine_parent','fine_depth','fine_spacing_mm'))
    if (selection.device.type!='cuda' or mask.shape!=selection.shape or edges.ndim!=2
            or edges.shape[0]!=3 or xyz.ndim!=3 or xyz.shape[-1]!=3
            or support.shape!=xyz.shape[:2] or parent.shape!=support.shape
            or depth.shape!=support.shape or spacing.shape!=(len(selection),3)
            or xyz.shape[0]!=len(selection)):
        raise ValueError('Complete aligned CUDA native-path metadata required')
    g, width = selection.shape
    m = support.shape[1]
    scene, target_slot, source_slot = edges
    _decision(((scene>=0)&(scene<g)&(target_slot>=0)&(target_slot<width)
               &(source_slot>=0)&(source_slot<width)).all(), 'Invalid mandatory path scene/slot')
    _decision((mask[scene,target_slot]&mask[scene,source_slot]).all(),
              'Mandatory path endpoint is padding')
    target, source = selection[scene,target_slot], selection[scene,source_slot]
    _decision(((target>=0)&(target<m)&(source>=0)&(source<m)).all(),
              'Mandatory path pool index outside fine substrate')
    _decision((support[scene,target]&support[scene,source]).all(),
              'Mandatory path endpoint lacks organ/all-scale support')
    _decision((torch.isfinite(spacing)&(spacing>0)).all(), 'Invalid mandatory native spacing')
    # Either directed message orientation may correspond to the downward path.
    target_depth, source_depth = depth[scene,target], depth[scene,source]
    _decision(((target_depth>=0)&(source_depth>=0)&(target_depth!=source_depth)).all(),
              'Mandatory endpoints need distinct reachable parent depths')
    descendant = torch.where(target_depth>source_depth,target,source)
    ancestor = torch.where(target_depth>source_depth,source,target)
    current = descendant
    length = spacing.new_zeros((edges.shape[1],))
    steps = torch.zeros_like(current)
    active = current!=ancestor
    while bool(active.any()):
        next_parent = parent[scene,current]
        valid = (next_parent>=0)&(next_parent<m)
        _decision((valid|~active).all(), 'Mandatory edge has no supported ancestor chain')
        safe = next_parent.clamp(0,m-1)
        delta = xyz[scene,current]-xyz[scene,safe]
        _decision(((support[scene,safe]&torch.isfinite(delta).all(-1)&
                    (delta.abs().sum(-1)==1)&(depth[scene,safe]==depth[scene,current]-1))
                   |~active).all(), 'Mandatory chain crosses background, a nonnative step or a cycle')
        increment = torch.linalg.vector_norm(delta*spacing[scene],dim=-1)
        length += torch.where(active,increment,0)
        steps += active.long()
        current = torch.where(active,safe,current)
        _decision(((length<=sampling['diagnostics']['context_radius_mm'])|~active).all(),
                  'Actual mandatory path exceeds the unchanged physical context radius')
        active = current!=ancestor
    claimed = sampling['edge_path_length_mm']
    _decision(claimed.shape==length.shape and torch.isclose(claimed,length,rtol=1e-6,atol=1e-5).all(),
              'Claimed mandatory length differs from its actual native path')
    _decision(torch.equal(sampling['edge_path_steps'],steps),
              'Claimed mandatory step count differs from its native path')
    return length
