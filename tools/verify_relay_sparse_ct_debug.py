"""Actual liver_66 CT: previous relational vs retained-path L0, DEBUG only.

All P5/U128 candidates, original donor/masks, quotas16/32/64, physical32.
Every representation rebuilds the complete saved DEBUG support. One disposable
warmup is restored before three continuous fresh-AdamW updates on the same tile.
No checkpoint, ready flag, production training, HTML, or reduced input is written.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
import torch
from tqdm import tqdm
from l0_local_cnn.data import CropStore, Dataset
from l0_regions.donor_learning import LiveContext, configuration, forward_loss
from l0_regions.training import FORMAT, hash_state, make_model
from l0_regions.training_data import Budget, sha, source_identity
from l0_sparse_feature.coverage import all_pool_distances, distance_stats
import l0_sparse_feature.model as baseline_module
import l0_sparse_feature.relational as relational_module
import l0_sparse_feature.relayed as relayed_module
import l0_sparse_feature.relay_sampling as relay_sampling_module
from l0_sparse_feature.relayed import RelayedV1SparseL0
from l0_sparse_feature.model import SparseFeatureL0, SparseFeatureProfile
from l0_sparse_feature.relational import V1RelationalSparseL0, V1RelationalSparseProfile
from tools.compare_sparse_feature_budgets_debug import assert_nested, assert_same_pool, score_vectors
from tools.diagnose_local_cnn_learning import checkpoint_for
from tools.local_cnn_interaction_runtime import support_binding
from tools.verify_sparse_feature_ct_debug import GeometryBatch, GraphLocalAdapter, anatomy
from hiercp_v222.v1_execution import restore_rng, rng_state

QUOTAS = (16, 32, 64)
VARIANTS = ('baseline', 'relational')
PHYSICAL_BATCH = 32
METRICS = ('feature_deficit', 'spatial_mm', 'joint')
IMPLEMENTATIONS = (
    'l0_sparse_feature/model.py', 'l0_sparse_feature/relational.py',
    'l0_sparse_feature/relayed.py', 'l0_sparse_feature/relay_sampling.py',
    'l0_sparse_feature/coverage.py', 'tools/verify_relay_sparse_ct_debug.py',
    'tools/verify_relational_sparse_ct_debug.py',
    'tools/verify_sparse_feature_ct_debug.py',
    'tools/compare_sparse_feature_budgets_debug.py',
    'tools/local_cnn_interaction_runtime.py', 'tools/local_cnn_l0_probe.py',
)


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu(item) for item in value]
    return value


def state_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: state_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [state_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(state_cpu(item) for item in value)
    return copy.deepcopy(value)


def statistics(values):
    mean, p95, maximum = cpu(distance_stats(values))
    return dict(count=values.numel(), mean=mean, p95=p95, max=maximum)


def synchronized_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def measured_batch(reader, rows, ids):
    """The same verified binding as bound_batch, separating load and transfer."""
    started = time.perf_counter()
    original = reader.batch(rows, ids)
    recipient = torch.tensor([row['center'] for row in rows], dtype=torch.float32).pin_memory()
    donor = torch.tensor(np.stack([reader.donor_bounds(row)[0] for row in rows]),
                         dtype=torch.float32).pin_memory()
    load_seconds = time.perf_counter() - started
    started = synchronized_time()
    batch = GeometryBatch(original.to('cuda', non_blocking=True),
                          recipient.to('cuda', non_blocking=True),
                          donor.to('cuda', non_blocking=True))
    transfer_seconds = synchronized_time() - started
    return batch, dict(load_crop_binding_seconds=load_seconds,
        transfer_geometry_validation_seconds=transfer_seconds,
        input_shape=list(batch.images.shape), unique_crops=len(batch.images),
        physical_pairs=len(batch), pinned_memory=True, non_blocking=True,
        cache=reader.report())


class ForwardPhases:
    """Events around the existing numerical functions; no alternate forward."""
    def __init__(self, encoder, variant):
        self.encoder = encoder
        self.variant = variant
        self.events, self.handles, self.contexts = {}, [], []

    def begin(self, name):
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        self.events.setdefault(name, []).append((start, stop))

    def end(self, name):
        self.events[name][-1][1].record()

    def wrap(self, function, name):
        def invoke(*args, **kwargs):
            self.begin(name)
            result = function(*args, **kwargs)
            self.end(name)
            return result
        return invoke

    def __enter__(self):
        module = relayed_module if isinstance(self.encoder, RelayedV1SparseL0) else relational_module
        for owner, attribute, name in ((module, '_select_context', 'seed_selection'),
                (self.encoder, '_select_nodes', 'selection_total'),
                (self.encoder, '_pair_edges', 'graph_build')):
            context = patch.object(owner, attribute, self.wrap(getattr(owner, attribute), name))
            context.__enter__()
            self.contexts.append(context)
        if isinstance(self.encoder, RelayedV1SparseL0):
            context = patch.object(relayed_module, 'retain_path_relays',
                self.wrap(relayed_module.retain_path_relays, 'relay_retention'))
            context.__enter__()
            self.contexts.append(context)
        self.handles.append(self.encoder.cnn.register_forward_pre_hook(lambda *_: self.begin('CNN')))
        self.handles.append(self.encoder.cnn.register_forward_hook(lambda *_: self.end('CNN')))
        self.handles.append(self.encoder.blocks[0].register_forward_pre_hook(lambda *_: self.begin('message_passing_3_layers')))
        self.handles.append(self.encoder.blocks[-1].register_forward_hook(lambda *_: self.end('message_passing_3_layers')))
        return self

    def __exit__(self, kind, value, trace):
        for handle in self.handles:
            handle.remove()
        for context in reversed(self.contexts):
            context.__exit__(kind, value, trace)

    def seconds(self):
        torch.cuda.synchronize()
        return {name: sum(start.elapsed_time(stop) for start, stop in events) / 1000
                for name, events in self.events.items()}


def module_prefixes(variant):
    result = dict(CNN='local.encoder.cnn', node_projection='local.encoder.node_project',
        role_embedding='local.encoder.role_embedding', message_passing='local.encoder.blocks',
        graph_readout='local.encoder.scene_project', fusion='local.encoder.fuse', L1='l1', L2='l2')
    result['attention_pool'] = 'local.encoder.attention_pool'
    return result


@torch.no_grad()
def refresh_support(net, saved_memory, ds, reader, query_group):
    modes = [(module, module.training) for module in net.modules()]
    net.eval()
    parts, loading, forward_times = [], [], []
    started = synchronized_time()
    for start in range(0, len(ds.rows), PHYSICAL_BATCH):
        ids = list(range(start, min(start + PHYSICAL_BATCH, len(ds.rows))))
        batch, load = measured_batch(reader, [ds.rows[i] for i in ids], ids)
        loading.append(load)
        current = synchronized_time()
        parts.append(net.local(batch).detach())
        forward_times.append(synchronized_time() - current)
        del batch
    memory = copy.deepcopy(saved_memory)
    memory['embeddings'] = torch.cat(parts)
    memory['owners'], memory['classes'] = memory['owners'].cuda(), memory['classes'].cuda()
    support, ids, _ = support_binding(memory, ds.rows, query_group)
    teacher_started = synchronized_time()
    plan = net.fit_support_clusters(*support)
    teacher_seconds = synchronized_time() - teacher_started
    elapsed = synchronized_time() - started
    for module, mode in modes:
        module.training = mode
    return support, plan, dict(records=len(ids), ids=ids, complete_original_DEBUG_records=len(ds.rows),
        complete_memory_reencoded=True, old_mean_memory_used=False, teacher_refitted=True,
        recipient_and_donor_group_exclusion_unchanged=True, query_group=query_group,
        memory_sha256=hash_state(memory['embeddings']), support_sha256=hash_state(support),
        teacher_sha256=hash_state(plan), total_seconds=elapsed,
        memory_CNN_selection_graph_forward_seconds=sum(forward_times),
        teacher_fit_seconds=teacher_seconds, load_transfer=loading,
        scope='Complete saved DEBUG memory; all eligible original support records; not production support')


def _components(node_count, edges):
    """CPU-only export audit, outside the numerical/timed model path."""
    neighbors = [set() for _ in range(node_count)]
    for source, target, _ in edges:
        neighbors[source].add(target)
        neighbors[target].add(source)
    unseen, components = set(range(node_count)), 0
    while unseen:
        components += 1
        reached, todo = set(), [min(unseen)]
        while todo:
            node = todo.pop()
            if node not in reached:
                reached.add(node)
                todo.extend(neighbors[node] - reached)
        unseen -= reached
    return components


def exact_scene(graph, index, batch, raw, case):
    """Exact model coordinates; anatomy contours are deduplicated separately."""
    alive, xyz, native, relative, roles, features = (graph[key][index].detach().cpu()
        for key in ('node_mask', 'xyz_mm', 'xyz_native', 'relative_xyz_mm', 'role', 'features'))
    slots = alive.nonzero().flatten().tolist()
    lookup = {slot: output for output, slot in enumerate(slots)}
    normalized = torch.nn.functional.normalize(features, dim=-1)
    global_role = graph.get('global_role')
    role_names = ('query', 'near', 'mid', 'wide')
    nodes = []
    for slot in slots:
        abstract = int(roles[slot]) == 0
        voxel = native[slot].round().long().numpy()
        if not abstract and (not np.all((voxel >= 0) & (voxel < np.asarray(raw['organ'].shape)))
                or not raw['organ'][tuple(voxel)]):
            raise AssertionError('Selected actual CT context node is outside original organ mask')
        role = role_names[int(roles[slot])]
        nodes.append(dict(xyz=xyz[slot].tolist(), native_voxel=native[slot].tolist(),
            relative_xyz_mm=relative[slot].tolist(), role=role,
            global_role=int(global_role[index, slot]) if global_role is not None else None,
            abstract=abstract, feature_norm=float(features[slot].norm()),
            cosine_to_query=float((normalized[slot] * normalized[0]).sum()),
            distance_mm=float(relative[slot].norm()), feature_channels=68,
            scale='CNN strides 1 / 2 / 4', model_slot=slot))
        nodes[-1]['retention'] = ('query' if abstract else 'seed' if 'sampling' not in graph
            or bool(graph['sampling']['seed_mask'][index,slot-1]) else 'relay')
    adjacency = graph['adjacency'][index].detach().cpu().bool()
    kinds = graph['edge_kind'][index].detach().cpu().long()
    names = {1: 'spatial', 2: 'feature', 3: 'spatial + feature', 4: 'connectivity',
             5: 'spatial + connectivity', 6: 'feature + connectivity', 7: 'spatial + feature + connectivity'}
    if 'pair_edge_index' in graph:
        # Typed relational edges are populated from the joint graph below.
        edges = []
    else:
        edges = [[lookup[source], lookup[target], names[int(kinds[target, source])]]
            for target, source in adjacency.nonzero().tolist()
            if source in lookup and target in lookup]
    audit = batch.audit[int(graph['crop_index'][index])]
    spacing, origin = np.asarray(audit['spacing']), np.asarray(audit['origin'])
    return dict(case=case, nodes=nodes, edges=edges, center=xyz[0].tolist(),
        box=[((origin - .5) * spacing).tolist(),
             ((origin + np.asarray(audit['shape']) - .5) * spacing).tolist()],
        stats=dict(node_count=len(nodes), directed_edges=int(adjacency.sum()),
            role_counts={role: sum(node['role'] == role for node in nodes) for role in role_names},
            components=_components(len(nodes), edges) if edges else None),
        original_native_coordinates=True, coordinate_rounding_applied=False)


def exact_joint(graph, pair, donor, recipient, batch):
    """Directed [source,target,type], retaining every actual typed relation."""
    nodes = [dict(node, side=side, case=scene['case'], local_index=index)
        for side, scene in (('donor', donor), ('recipient', recipient))
        for index, node in enumerate(scene['nodes'])]
    width = graph['node_mask'].shape[1]
    donor_map = {node['model_slot']: index for index, node in enumerate(donor['nodes'])}
    recipient_map = {node['model_slot'] + width: index + len(donor['nodes'])
                     for index, node in enumerate(recipient['nodes'])}
    lookup = {**donor_map, **recipient_map}
    if 'pair_edge_index' in graph:
        packed = graph['pair_edge_index'].detach().cpu()
        selected = (packed[0] == pair).nonzero().flatten()
        edge_ids = packed[:, selected].T.tolist()
        names = graph['relation_names']
        edges = [[lookup[source], lookup[target], names[relation]]
                 for _, relation, target, source in edge_ids]
        distances = graph['pair_edge_distance_mm'][selected.to(graph['pair_edge_distance_mm'].device)].detach().cpu().tolist()
    else:
        edges = copy.deepcopy(donor['edges']) + [
            [source + len(donor['nodes']), target + len(donor['nodes']), name]
            for source, target, name in recipient['edges']]
        distances = [float(np.linalg.norm(np.asarray(nodes[source]['relative_xyz_mm'])
                                         - np.asarray(nodes[target]['relative_xyz_mm'])))
                     for source, target, _ in edges]
    for scene, side in ((donor, 'donor'), (recipient, 'recipient')):
        if 'pair_edge_index' in graph:
            scene['edges'] = [[nodes[source]['local_index'], nodes[target]['local_index'], name]
                for source, target, name in edges
                if nodes[source]['side'] == side and nodes[target]['side'] == side]
            scene['stats']['directed_edges'] = len(scene['edges'])
            scene['stats']['components'] = _components(len(scene['nodes']), scene['edges'])
    cross = sum(nodes[source]['side'] != nodes[target]['side'] for source, target, _ in edges)
    paths = [None] * len(edges)
    if 'sampling' in graph:
        sample = graph['sampling']
        mandatory = sample['mandatory_edges'].detach().cpu()
        selection = sample['selection_index'].detach().cpu()
        parent = sample['fine_parent'].detach().cpu()
        xyz_local = sample['fine_xyz_local'].detach().cpu()
        eligible = sample['fine_eligible'].detach().cpu()
        path_lookup = {}
        half = mandatory.shape[1] // 2
        for column in range(half):
            scene, descendant_slot, ancestor_slot = mandatory[:,column].tolist()
            if scene not in (pair, len(batch)+pair):
                continue
            side = 0 if scene==pair else 1
            descendant = int(selection[scene,descendant_slot])
            ancestor = int(selection[scene,ancestor_slot])
            chain, current = [descendant], descendant
            while current != ancestor:
                current = int(parent[scene,current])
                if current < 0 or len(chain)>parent.shape[1] or not bool(eligible[scene,current]):
                    raise AssertionError('Exported mandatory path lacks actual fine-pool support')
                chain.append(current)
            native_points = xyz_local[scene,chain].numpy()
            if not bool(eligible[scene,chain].all()):
                raise AssertionError('Mandatory path contains an unsupported voxel')
            audit = batch.audit[int(graph['crop_index'][scene])]
            spacing = np.asarray(audit['spacing'])
            steps = np.diff(native_points,axis=0)
            if not (np.abs(steps).sum(1)==1).all():
                raise AssertionError('Mandatory parent witness is not native axis-adjacent')
            physical_length = np.linalg.norm(steps*spacing,axis=1).sum()
            if physical_length > 6.0+1e-5:
                raise AssertionError('Mandatory parent witness exceeds its fixed6mm radius')
            absolute = ((native_points+np.asarray(audit['origin']))*spacing).tolist()
            prefix = side*width
            child, root = lookup[prefix+descendant_slot+1], lookup[prefix+ancestor_slot+1]
            relation = graph['relation_names'][side]
            path_lookup[(root,child,relation)] = list(reversed(absolute))
            path_lookup[(child,root,relation)] = absolute
        paths = [path_lookup.get(tuple(edge)) for edge in edges]
        if sum(path is not None for path in paths) != len(path_lookup):
            raise AssertionError('A mandatory parent edge disappeared from actual graph export')
    return dict(nodes=nodes, edges=edges, edge_distance_mm=distances,
        edge_path_mm=paths, native_parent_paths_verified=True,
        edge_orientation='source,target,relation_name',
        edge_distance_space='Actual original anchor-relative millimetres; display offset never used',
        stats=dict(node_count=len(nodes), directed_typed_edges=len(edges),
            components=_components(len(nodes), edges), cross_edges=cross,
            relation_counts={name: sum(edge[2] == name for edge in edges)
                             for name in sorted({edge[2] for edge in edges})}),
        display_geometry_generated=False, anatomical_path_claim=False)


@torch.no_grad()
def selected_equal(first, second):
    assert_same_pool(first['coverage_pool'], second['coverage_pool'])
    # Every original seed is present with its exact native position, features,
    # role and anchor. Added relays are allowed; no original seed can be lost.
    matches = (first['xyz_native'][:, :, None] == second['xyz_native'][:, None]).all(-1)
    matches &= second['node_mask'][:, None]
    matches[:, 0, :] = False
    matches[:, 0, 0] = True  # Abstract query can coincide with a real seed.
    matches[:, 1:, 0] = False
    if not bool(((matches.sum(-1) == 1) | ~first['node_mask']).all()):
        raise AssertionError('Original seed lost or duplicated in the retained graph')
    index = matches.long().argmax(-1)
    for name in ('features', 'relative_xyz_mm', 'role'):
        shape = first[name].shape
        lookup = index if len(shape) == 2 else index[...,None].expand(-1,-1,shape[-1])
        retained = second[name].gather(1, lookup)
        if not torch.equal(first[name][first['node_mask']], retained[first['node_mask']]):
            raise AssertionError('Original seed representation changed: ' + name)


def measured_update(net, variant, optimizer, batch, support, plan, targets, context, tile, base, initial):
    net.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = synchronized_time()
    with ForwardPhases(net.local.encoder, variant) as phases:
        loss, terms = forward_loss(net, batch, support, plan, targets, None,
                                  context, configuration(), indices=tile)
    forward_seconds = synchronized_time() - started
    current = synchronized_time()
    loss.backward()
    backward_seconds = synchronized_time() - current
    current = synchronized_time()
    named = [(name, parameter) for name, parameter in net.named_parameters() if parameter.requires_grad]
    for name, parameter in named:
        if parameter.grad is None:
            raise AssertionError('Trainable parameter missing actual loss gradient: ' + name)
    if not bool(torch.stack([torch.isfinite(parameter.grad).all() for _, parameter in named]).all()):
        raise FloatingPointError('Nonfinite actual complete-objective gradient')
    prefixes = module_prefixes(variant)
    norms = {}
    zero_counts = {}
    for name, prefix in prefixes.items():
        members = [parameter for key, parameter in named if key.startswith(prefix)]
        if not members:
            raise AssertionError('Required module absent from actual model: ' + prefix)
        norms[name] = torch.stack([parameter.grad.float().square().sum() for parameter in members]).sum().sqrt()
        zero_counts[name] = torch.stack([parameter.grad.eq(0).all() for parameter in members]).sum()
    if not bool(torch.stack(list(norms.values())).gt(0).all()):
        raise AssertionError('A required module received no actual loss gradient: ' + str(cpu(norms)))
    typed_gradients = None
    if True:  # Both branches use the same typed relational modules.
        encoder = net.local.encoder
        typed_gradients = dict(
            message_layers=[dict(zip(relational_module.RELATION_NAMES,
                cpu(block.neighbor_weight.grad.float().square().sum((1,2)).sqrt())))
                for block in encoder.blocks],
            pool_value_by_role=dict(zip(relational_module.NODE_ROLE_NAMES,
                cpu(encoder.attention_pool.value_weight.grad.float().square().sum((1,2)).sqrt()))),
            pool_score_by_context_role=dict(zip(
                ('donor_near','donor_mid','donor_wide','recipient_near','recipient_mid','recipient_wide'),
                cpu(encoder.attention_pool.score_weight.grad.float().square().sum(-1).sqrt()))))
    unclipped_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), base['training']['grad_clip'], error_if_nonfinite=True)
    audit_clip_seconds = synchronized_time() - current
    current = synchronized_time()
    optimizer.step()
    optimizer_seconds = synchronized_time() - current
    elapsed = synchronized_time() - started
    peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    # CPU delta copies and diagnostic summaries occur outside measured update.
    deltas = {name: sum(float((parameter.detach().cpu() - initial[key]).float().square().sum())
                        for key, parameter in named if key.startswith(prefix)) ** .5
              for name, prefix in prefixes.items()}
    if not all(value > 0 for value in deltas.values()):
        raise AssertionError('A connected module was not updated by actual optimizer')
    typed_deltas = None
    if True:
        typed_deltas = dict(message_layers=[dict(zip(relational_module.RELATION_NAMES,
            (block.neighbor_weight.detach().cpu() - initial[f'local.encoder.blocks.{index}.neighbor_weight'])
            .float().square().sum((1,2)).sqrt().tolist()))
            for index,block in enumerate(net.local.encoder.blocks)],
            pool_value_by_role=dict(zip(relational_module.NODE_ROLE_NAMES,
                (net.local.encoder.attention_pool.value_weight.detach().cpu()
                 - initial['local.encoder.attention_pool.value_weight']).float().square().sum((1,2)).sqrt().tolist())),
            pool_score_by_context_role=dict(zip(
                ('donor_near','donor_mid','donor_wide','recipient_near','recipient_mid','recipient_wide'),
                (net.local.encoder.attention_pool.score_weight.detach().cpu()
                 - initial['local.encoder.attention_pool.score_weight']).float().square().sum(-1).sqrt().tolist())))
    result = dict(loss=float(loss.detach()), terms=cpu(terms), gradient_norms=cpu(norms),
        zero_gradient_parameter_tensors=cpu(zero_counts), missing_gradients=[],
        parameter_delta_norms_from_initial=deltas, unclipped_global_gradient_norm=float(unclipped_norm),
        typed_relation_role_gradient_norms=typed_gradients,
        typed_relation_role_parameter_delta_norms_from_initial=typed_deltas,
        forward_seconds=forward_seconds, backward_seconds=backward_seconds,
        gradient_audit_clip_seconds=audit_clip_seconds, optimizer_seconds=optimizer_seconds,
        full_update_seconds=elapsed, forward_phases_seconds=phases.seconds(),
        peak_allocated_bytes=peak, peak_reserved_bytes=reserved,
        steady_allocated_bytes=torch.cuda.memory_allocated(),
        RSS_bytes=psutil.Process().memory_info().rss, pairs_per_second=len(batch) / elapsed)
    del loss, terms
    return result


def hardware(budget, a):
    nvidia = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,utilization.gpu,mig.mode.current',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=False)
    process = psutil.Process()
    return dict(gpu=torch.cuda.get_device_name(), gpu_count=torch.cuda.device_count(),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        total_VRAM=torch.cuda.get_device_properties(0).total_memory,
        free_at_start=torch.cuda.mem_get_info()[0], cuda_limit=budget.cuda_bytes,
        RSS_limit=budget.rss_bytes, resident_limit=int(a.resident_gib * 2**30),
        workers=a.workers, cpu_logical=psutil.cpu_count(), cpu_physical=psutil.cpu_count(logical=False),
        cpu_affinity=process.cpu_affinity(), total_RAM=psutil.virtual_memory().total,
        available_RAM=psutil.virtual_memory().available, process_RSS=process.memory_info().rss,
        nvidia_smi=dict(returncode=nvidia.returncode, output=nvidia.stdout.strip(), error=nvidia.stderr.strip()),
        scheduler={key: os.environ[key] for key in ('SLURM_CPUS_PER_TASK', 'SLURM_JOB_ID', 'PBS_JOBID') if key in os.environ},
        storage=dict(workspace=str(ROOT), run=str(a.run.resolve()),
            free_bytes=psutil.disk_usage(str(ROOT)).free),
        parallelism='One CUDA device, identical physical32 paired batches; parallel raw readers, cached crops; comparative DEBUG only')


def run(a):
    if a.output.exists():
        raise FileExistsError('Existing results are preserved. Select an explicitly new --output directory.')
    if a.case != 'liver_66' or (a.warmup_updates, a.measured_updates) != (1, 3):
        raise ValueError('This explicitly scoped DEBUG requires liver_66, one warmup and three continuous measured updates')
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA is required; no CPU model fallback')
    if a.workers < 2 or not 0 < a.resident_gib < a.rss_gib or a.coverage_workspace_mib <= 0:
        raise ValueError('Explicit parallel readers, RAM headroom, and positive coverage workspace are required')
    # Fail on an unwritable destination before spending time on CT/CUDA work.
    # This new task-owned directory never replaces an earlier result.
    implementation_before = {name: sha(ROOT / name) for name in IMPLEMENTATIONS}
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output / 'started.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(dict(debug=True, production_training_started=False,
            checkpoint_written=False, completed=False, case=a.case,
            implementation_sha256=implementation_before), stream, indent=2)
    torch.set_num_threads(a.workers)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    free, total = torch.cuda.mem_get_info()
    budget = Budget(int(a.cuda_gib * 2**30), int(a.rss_gib * 2**30))
    if not 0 < budget.cuda_bytes < min(free, total):
        raise ValueError('Explicit CUDA budget must leave measured free VRAM headroom')
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes / total)
    resources = hardware(budget, a)
    source_before = source_identity()
    cp = checkpoint_for(a.run.resolve())
    cp_sha, assignment_sha, receipt_sha = sha(cp), sha(a.assignment), sha(a.assignment_receipt)
    receipt = json.loads(a.assignment_receipt.read_text(encoding='utf-8'))
    if (receipt.get('assignment_sha256') != assignment_sha or receipt.get('case') != a.case
            or receipt.get('checkpoint_sha256') != cp_sha):
        raise ValueError('Original full-case CT assignment/checkpoint receipt differs')
    saved = torch.load(cp, map_location='cpu', weights_only=False)
    if saved.get('format') != FORMAT or not saved['identity'].get('debug') or saved['state'].get('step') != 4:
        raise ValueError('Matching optimized local DEBUG snapshot step4 required')
    if saved['identity']['source'] != source_before or hash_state({key: value for key, value in saved.items()
            if key != 'content_sha256'}) != saved['content_sha256']:
        raise ValueError('Original checkpoint source/content integrity mismatch')
    saved_memory_hash = hash_state(saved['state']['memory'])
    ds = Dataset(a.run / 'inventory/index.json', 'inner_train', True)
    if saved['identity']['cache_sha256'] != sha(ds.path):
        raise ValueError('Original native inventory binding mismatch')
    observed = [row for row in ds.rows if row['case_id'] == a.case]
    if not observed:
        raise ValueError('liver_66 is absent from verified saved DEBUG cohort')
    donor = {key: observed[0][key] for key in ('donor_case_id', 'donor_component', 'donor_group')}
    if any({key: row[key] for key in donor} != donor for row in observed) or receipt.get('donor_binding') != donor:
        raise ValueError('Original fixed independent donor binding changed')
    original = [row for row in json.loads(a.assignment.read_text(encoding='utf-8')) if row['case_id'] == a.case]
    positive, unobserved = ([row for row in original if row['target'] == label] for label in (1, 0))
    raw_record = next(row for row in ds.meta['raw_records'] if row['case_id'] == a.case)
    if len(positive) != 5 or len(positive) != len(raw_record['positives']) or len(unobserved) != 128:
        raise ValueError('Every original liver_66 P5 and U128 is required; no subset/candidate substitution')
    rows = [dict(row, **donor, bounds={'edges': index}) for index, row in enumerate(positive + unobserved)]
    context_rows = rows + [row for row in ds.rows if row['case_id'] != a.case]
    context = LiveContext(SimpleNamespace(rows=context_rows), PHYSICAL_BATCH)
    context_hash = hash_state(dict(rows=context_rows, audit=context.audit, uses=dict(context.uses), order=context.order))
    tile = next(ids for ids in context.order if len(ids) == PHYSICAL_BATCH
        and {context_rows[i]['case_id'] for i in ids} == {a.case}
        and 0 < sum(context_rows[i]['target'] for i in ids) < PHYSICAL_BATCH)
    reference = make_model(ds, budget, True, 'retained')
    reference.load_state_dict(saved['model'], strict=True)
    reference_copied_hash = hash_state(reference.state_dict())
    models, reports, initial_hashes = {}, {}, {}
    for quota in QUOTAS:
        for variant in VARIANTS:
            profile_type = V1RelationalSparseProfile
            encoder_type = V1RelationalSparseL0 if variant == 'baseline' else RelayedV1SparseL0
            profile = profile_type(context_nodes_per_band=quota, query_radius_mm=a.query_radius_mm,
                near_radius_mm=a.near_radius_mm, mid_radius_mm=a.mid_radius_mm)
            profile.validate()
            net = copy.deepcopy(reference)
            net.local = GraphLocalAdapter(encoder_type(reference.local, profile, budget=budget)).cuda()
            net.eval()
            key = (quota, variant)
            models[key] = net
            initial_hashes[key] = hash_state(net.state_dict())
            reports[key] = dict(variant=variant, context_nodes=quota * 3, quota=quota,
                profile=vars(profile), graph_audits=[], scenes=[],
                total_parameters=sum(parameter.numel() for parameter in net.parameters()),
                trainable_parameters=sum(parameter.numel() for parameter in net.parameters() if parameter.requires_grad),
                initial_model_sha256=initial_hashes[key])
            for attribute in ('cnn', 'fuse'):
                if hash_state(getattr(net.local.encoder, attribute).state_dict()) != hash_state(getattr(reference.local, attribute).state_dict()):
                    raise AssertionError('Original CNN/fusion weights differ: ' + attribute)
            for attribute in ('l1', 'l2'):
                if hash_state(getattr(net, attribute).state_dict()) != hash_state(getattr(reference, attribute).state_dict()):
                    raise AssertionError('Original L1/L2 weights differ: ' + attribute)
    for variant in VARIANTS:
        if len({initial_hashes[(quota, variant)] for quota in QUOTAS}) != 1:
            raise AssertionError('Node quota changed initial parameters within ' + variant)
    for quota in QUOTAS:
        if initial_hashes[(quota,'baseline')] != initial_hashes[(quota,'relational')]:
            raise AssertionError('Sampler comparison changed any initial model weights')
    del reference
    reader = CropStore(ds.meta, a.workers, int(a.resident_gib * 2**30), budget.rss_bytes)
    print(json.dumps(dict(debug=True, actual_CUDA=True, case=a.case, original_P=5, original_U=128,
        original_donor=donor, saved_step=4, full_case_forward_records=len(rows),
        saved_DEBUG_memory_records=len(ds.rows), complete_loss_context=context.audit,
        physical_batch=PHYSICAL_BATCH, effective_batch=PHYSICAL_BATCH, gradient_accumulation=1,
        model=dict(CNN_channels=[12,24,32], CNN_convolutions=[2,3,3], graph_layers=3,
            hidden_dim=128, L1_layers=len(models[(16,'baseline')].l1), L2_layers=len(models[(16,'baseline')].l2)),
        resources=resources, budgets={f'{quota * 3}:{variant}': reports[(quota,variant)]['profile']
            for quota,variant in models}, debug_only=True, full_training=False), indent=2), flush=True)
    supports, plans = {}, {}
    for key, net in models.items():
        supports[key], plans[key], reports[key]['initial_memory_teacher_refresh'] = refresh_support(
            net, saved['state']['memory'], ds, reader, rows[0]['patient_group'])
    if len({tuple(report['initial_memory_teacher_refresh']['ids']) for report in reports.values()}) != 1:
        raise AssertionError('Representation changed original support group exclusion or record coverage')
    display = set(range(len(positive))) | set(range(len(positive), len(positive) + 3))
    anatomy_cache = {}
    displayed = {key: [] for key in models}
    donor_scenes, vectors = {}, {key: [] for key in models}
    coverage_parts = {key: {name: [] for name in METRICS} for key in models}
    torch.cuda.reset_peak_memory_stats()
    all_case_started = synchronized_time()
    for start in tqdm(range(0, len(rows), PHYSICAL_BATCH), desc='actual paired CT all133 DEBUG', unit='batch'):
        ids = list(range(start, min(start + PHYSICAL_BATCH, len(rows))))
        batch, loading = measured_batch(reader, [rows[index] for index in ids], ids)
        previous, previous_distances, first_pool = None, None, None
        with torch.no_grad():
            for quota in QUOTAS:
                baseline_graph = None
                for variant in VARIANTS:
                    key, net = (quota, variant), models[(quota, variant)]
                    started = synchronized_time()
                    with ForwardPhases(net.local.encoder, variant) as phases:
                        output, graph = net.local.encoder(batch,
                            recipient_centers_native=batch.recipient_centers,
                            donor_centers_native=batch.donor_centers, return_graph=True, return_pool=True)
                    elapsed = synchronized_time() - started
                    if first_pool is None:
                        first_pool = graph['coverage_pool']
                    else:
                        assert_same_pool(first_pool, graph['coverage_pool'])
                    if variant == 'baseline':
                        baseline_graph = graph
                    else:
                        selected_equal(baseline_graph, graph)
                    started = synchronized_time()
                    distances = all_pool_distances(graph, workspace_bytes=int(a.coverage_workspace_mib * 2**20))
                    coverage_seconds = synchronized_time() - started
                    if variant == 'baseline':
                        if previous is not None:
                            assert_nested(previous, graph, previous_distances, distances)
                        previous, previous_distances = graph, distances
                    else:
                        for old, current in zip(previous_distances, distances):
                            for name in METRICS:
                                tolerance = 1e-4 if name == 'joint' else 1e-3 if name == 'spatial_mm' else 2e-6
                                if not bool((current['values'][name][current['mask']] <=
                                             old['values'][name][current['mask']] + tolerance).all()):
                                    raise AssertionError('Retaining every seed made fine-pool coverage worse: ' + name)
                    scene_values = {name: torch.zeros_like(graph['coverage_pool']['eligible'], dtype=torch.float32) for name in METRICS}
                    for entry in distances:
                        for name, value in entry['values'].items():
                            coverage_parts[key][name].append(value[len(batch):][entry['mask'][len(batch):]].detach())
                            scene_values[name] += torch.where(entry['mask'], value, 0)
                    complete_mask = graph['coverage_pool']['eligible']
                    scene_summaries = cpu({name: distance_stats(value, complete_mask) for name, value in scene_values.items()})
                    scene_counts = cpu(graph['node_mask'].sum(-1))
                    side_edges = cpu(graph['adjacency'].sum((1,2)))
                    # Preserve per-record diagnostics; do not dump padded
                    # O(B*R*N) masks/degree arrays for every cohort record.
                    keep = ('organ_fine_pool_counts','eligible_all_scale_pool_counts',
                        'unsupported_scale_counts','unsupported_any_scale_counts','band_pool_counts',
                        'band_selected_counts','query_pool_counts','relation_edge_counts',
                        'relation_max_incoming','relation_max_outgoing','relation_unmatched_target_counts',
                        'relation_unmatched_source_counts','pair_components','pair_weak_connected',
                        'pair_isolated_node_counts','physical_pairs','unique_cnn_crops',
                        'seed_counts','relay_counts','final_context_counts','native6_substrate_unreachable_counts',
                        'mandatory_path_edge_counts','max_compressed_path_length_mm',
                        'bfs_wave_count','fine_reachable_counts')
                    graph_statistics = cpu({name:graph['statistics'][name] for name in keep if name in graph['statistics']})
                    if True:
                        relation_counts = cpu(graph['statistics']['relation_edge_counts'])
                        components = cpu(graph['statistics']['pair_components'])
                        unmatched_target = cpu(graph['statistics']['relation_unmatched_target_counts'])
                        unmatched_source = cpu(graph['statistics']['relation_unmatched_source_counts'])
                    reports[key]['graph_audits'].append(dict(original_indices=ids, statistics=graph_statistics,
                        load_transfer=loading, forward_seconds=elapsed, forward_phases_seconds=phases.seconds(),
                        coverage_seconds=coverage_seconds, actual_scene_node_counts=scene_counts,
                        actual_scene_directed_edge_counts=side_edges,
                        relation_edge_audit=cpu(graph.get('relation_edge_audit'))))
                    for position, index in enumerate(ids):
                        scene_index = len(batch) + position
                        entry = dict(id=rows[index]['id'], kind='P' if rows[index]['target'] else 'U',
                            donor_node_count=scene_counts[position], recipient_node_count=scene_counts[scene_index],
                            pair_node_count=scene_counts[position] + scene_counts[scene_index],
                            donor_directed_edges=side_edges[position], recipient_directed_edges=side_edges[scene_index],
                            full_eligible_count=int(complete_mask[scene_index].sum()),
                            coverage={name: dict(zip(('mean','p95','max'), summary[scene_index])) for name,summary in scene_summaries.items()},
                            bands=[dict(role=band['role'], pool_count=int(band['full_counts'][scene_index]),
                                selected_count=int(band['selected_counts'][scene_index]),
                                metrics={name: dict(zip(('mean','p95','max'), cpu(summary[scene_index])))
                                         for name,summary in band['summaries'].items()}) for band in distances])
                        if True:
                            entry.update(components=components[position], relation_counts=dict(zip(graph['relation_names'], relation_counts[position])),
                                cross_edges=sum(relation_counts[position][6:]),
                                isolated_nodes=int(graph['statistics']['pair_isolated_node_counts'][position]),
                                seed_context_counts=[int((graph['statistics']['seed_counts'] if variant=='relational'
                                    else graph['node_mask'].sum(-1)-1)[j]) for j in (position,scene_index)],
                                relay_counts=[int(graph['statistics']['relay_counts'][j]) if variant=='relational' else 0 for j in (position,scene_index)],
                                native6_substrate_unreachable_counts=[int(graph['statistics']['native6_substrate_unreachable_counts'][j]) if variant=='relational'
                                    else None for j in (position,scene_index)],
                                unmatched_target_counts=dict(zip(graph['relation_names'], unmatched_target[position])),
                                unmatched_source_counts=dict(zip(graph['relation_names'], unmatched_source[position])))
                        else:
                            entry.update(components=2, cross_edges=0, joint_directed_edges=side_edges[position]+side_edges[scene_index],
                                cross_relation_applicable=False)
                        reports[key]['scenes'].append(entry)
                        if index not in display:
                            continue
                        row = rows[index]
                        for case in (row['case_id'], row['donor_case_id']):
                            if case not in anatomy_cache:
                                anatomy_cache[case] = anatomy(reader.raw.cache[case])
                        source = exact_scene(graph, position, batch, reader.raw.cache[row['donor_case_id']], row['donor_case_id'])
                        target = exact_scene(graph, scene_index, batch, reader.raw.cache[row['case_id']], row['case_id'])
                        joint = exact_joint(graph, position, source, target, batch)
                        target['stats']['coverage'] = entry['coverage']
                        displayed[key].append(dict(id=row['id'], kind='P' if row['target'] else 'U', scene=target,
                            donor=source, joint=joint))
                        if key not in donor_scenes:
                            donor_scenes[key] = source
                    vectors[key].append(output.detach())
                    del graph, distances, scene_values
        del batch, previous, previous_distances, first_pool, baseline_graph
    all_case_seconds = synchronized_time() - all_case_started
    coverage_peak = torch.cuda.max_memory_allocated()
    for key, net in models.items():
        reports[key]['recipient_all_points'] = {name: statistics(torch.cat(parts)) for name,parts in coverage_parts[key].items()}
        reports[key]['initial_full_case_score'] = score_vectors(net, vectors[key], supports[key], plans[key], rows)
        if len(reports[key]['scenes']) != 133 or len(displayed[key]) != 8:
            raise AssertionError('Original all133 forward/displayP5U3 contract is incomplete')
    del vectors, coverage_parts
    gc.collect()
    batch, update_loading = measured_batch(reader, [context_rows[index] for index in tile], tile)
    targets = torch.tensor([context_rows[index]['target'] for index in tile], device='cuda', dtype=torch.long)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    fair_rng = rng_state()
    for key, net in models.items():
        quota, variant = key
        initial = state_cpu(net.state_dict())
        restore_rng(fair_rng)
        warm_optimizer = torch.optim.AdamW(net.parameters(), lr=ds.meta['base']['training']['lr'],
            weight_decay=ds.meta['base']['training']['weight_decay'])
        reports[key]['warmup'] = measured_update(net, variant, warm_optimizer, batch,
            supports[key], plans[key], targets, context, tile, ds.meta['base'], initial)
        del warm_optimizer
        net.load_state_dict(initial, strict=True)
        net.zero_grad(set_to_none=True)
        restore_rng(fair_rng)
        if hash_state(net.state_dict()) != initial_hashes[key]:
            raise AssertionError('Disposable warmup did not restore exact initial model/buffers')
        optimizer = torch.optim.AdamW(net.parameters(), lr=ds.meta['base']['training']['lr'],
            weight_decay=ds.meta['base']['training']['weight_decay'])
        expected = {id(parameter) for parameter in net.parameters() if parameter.requires_grad}
        actual = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
        if actual != expected:
            raise AssertionError('Trainable modules are not exactly covered by actual optimizer')
        runs = []
        for step in tqdm(range(1,a.measured_updates+1), desc=f'{variant} context{quota*3} continuous AdamW DEBUG', unit='update'):
            runs.append(dict(step=step, **measured_update(net, variant, optimizer, batch,
                supports[key], plans[key], targets, context, tile, ds.meta['base'], initial)))
        optimizer_steps = sorted({int(state['step']) for state in optimizer.state.values()})
        if optimizer_steps != [3]:
            raise AssertionError('AdamW moments did not persist over all three actual measured updates')
        reports[key]['measured_updates'] = runs
        reports[key]['update_summary'] = {name: dict(mean=float(np.mean([run[name] for run in runs])),
            median=float(np.median([run[name] for run in runs])), min=min(run[name] for run in runs),
            max=max(run[name] for run in runs)) for name in ('full_update_seconds','forward_seconds','backward_seconds',
                'gradient_audit_clip_seconds','optimizer_seconds','peak_allocated_bytes','pairs_per_second')}
        reports[key]['optimizer'] = dict(name='AdamW', moments='Fresh after restored warmup; continuous across measured updates',
            verified_state_steps=optimizer_steps, learning_rate=ds.meta['base']['training']['lr'],
            weight_decay=ds.meta['base']['training']['weight_decay'], grad_clip=ds.meta['base']['training']['grad_clip'],
            saved_optimizer_moments_used=False, all_trainable_parameters_in_optimizer=True)
        reports[key]['post_update_model_sha256'] = hash_state(net.state_dict())
        # Fresh representation and teacher cost after the short prefix, without
        # silently mixing updated queries with the old CNN mean support table.
        refreshed, refreshed_plan, refresh = refresh_support(net, saved['state']['memory'], ds,
            reader, rows[0]['patient_group'])
        reports[key]['post_update_memory_teacher_refresh'] = refresh
        reports[key]['teacher_policy_during_three_updates'] = 'Own initial reencoded detached DEBUG support and teacher fixed within this short prefix'
        with (a.output / f'context{quota*3}_{variant}.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(reports[key], stream, ensure_ascii=False, indent=2, allow_nan=False)
        del refreshed, refreshed_plan, initial, optimizer
        net.zero_grad(set_to_none=True)
    if (sha(cp) != cp_sha or sha(a.assignment) != assignment_sha or sha(a.assignment_receipt) != receipt_sha
            or source_identity() != source_before or hash_state(saved['state']['memory']) != saved_memory_hash
            or hash_state(dict(rows=context_rows,audit=context.audit,uses=dict(context.uses),order=context.order)) != context_hash):
        raise AssertionError('Original checkpoint/assignment/source/saved memory/full loss schedule changed')
    if {name: sha(ROOT / name) for name in IMPLEMENTATIONS} != implementation_before:
        raise AssertionError('Executed comparison implementation changed during this run')
    payload_budgets = []
    for quota in QUOTAS:
        variants = {variant: dict(donor=donor_scenes[(quota,variant)], candidates=displayed[(quota,variant)]) for variant in VARIANTS}
        payload_budgets.append(dict(context_nodes=quota*3, quota=quota, variants=variants,
            metrics={variant: dict(coverage=reports[(quota,variant)]['recipient_all_points'],
                update_seconds=reports[(quota,variant)]['update_summary']['full_update_seconds']['median'],
                peak_GiB=reports[(quota,variant)]['update_summary']['peak_allocated_bytes']['max']/2**30)
                for variant in VARIANTS}))
    payload = dict(case=a.case, donor_case=donor['donor_case_id'], margin_mm=ds.meta['local_cnn']['margin_mm'],
        anatomy=anatomy_cache, budgets=payload_budgets,
        weight_source=dict(label='CNN/fusion/L1/L2: integrity-verified local DEBUG snapshot step4; new graph modules: seed42'),
        scope='Actual forward nodes and directed typed relations; all133 candidates; DEBUG only, no ranking-quality claim',
        coordinate_contract='Native/mm values exported without rounding or display offsets; edges are messages, not anatomical paths')
    final = dict(debug=True, actual_CT=True, actual_CUDA=True, production_ready=False,
        checkpoint_written=False, production_default_changed=False, production_training_started=False,
        full_training=False, full_evaluation=False, actual_data_used=True, case=a.case, original_P=5, original_U=128,
        complete_case_observations=133, actual_case_observations_used=133, case_usage_ratio=1.,
        complete_DEBUG_loss_context=len(context_rows), saved_DEBUG_memory_records=len(ds.rows),
        physical_batch=32, effective_batch=32, gradient_accumulation=1, data_parallel_workers=1,
        warmup_updates_per_branch=1, measured_continuous_updates_per_branch=3,
        original_tile_indices=tile, original_tile_ids=[context_rows[index]['id'] for index in tile],
        complete_original_loss_schedule=context.audit, epoch_completed=False,
        original_assignment_sha256=assignment_sha, original_donor=donor,
        assignment_receipt=dict(path=str(a.assignment_receipt),sha256=receipt_sha),
        weight_source=dict(checkpoint=str(cp), sha256=cp_sha, saved_step=4, copied_reference_sha256=reference_copied_hash,
            saved_source_file_count=sum(len(value) for value in source_before.values()),
            CNN_fusion_L1_L2='Exact copied local DEBUG snapshot weights', graph='Fresh seed42 modules',
            exact_checkpoint_resume=False, server_trained_weights_used=False),
        maintained=dict(CNN_depth_channels=True, graph_layers_3_hidden_128=True, L1_L2_architecture_equations=True,
            loss_and_normalization=True, P_U_definition=True, same_original_donor=True, all_original128U=True,
            Basic_CP=True, original_masks=True, native_resolution=True, full_feature_pool=True),
        budgets={str(quota*3):{variant:reports[(quota,variant)] for variant in VARIANTS} for quota in QUOTAS},
        resources=dict(**resources, coverage_peak_allocated=coverage_peak,
            final_RSS=psutil.Process().memory_info().rss, final_available_RAM=psutil.virtual_memory().available,
            CUDA_peak_reserved=torch.cuda.max_memory_reserved()),
        precision=dict(dtype='FP32', autocast=False, TF32=False, deterministic=True),
        data_loading=dict(workers=a.workers,persistent_workers='Parallel RawStore reader executor',
            prefetch_factor='Not a DataLoader; one cached verified CropStore producer',pin_memory=True,
            non_blocking=True,cache=reader.report(),crop_reduction=False,update_batch_load_transfer=update_loading),
        timing=dict(all_case_forward_coverage_CPU_export_seconds=all_case_seconds,
            measured_update_scope='Prebound physical32: actual CNN/selection/graph/L1/L2/loss/backward/finite/clip/continuousAdamW; diagnostic delta copies and loading excluded',
            memory_teacher_refresh_scope='Full original DEBUG support forward plus unchanged group exclusion and teacher refit, loading/transfer included',
            warmup_excluded_from_three_continuous_updates=True, production_epoch_comparison=False),
        assertions=dict(all133_candidates_forwarded=True,exact_pool_equal_all_budgets_and_variants=True,
            every_original_seed_features_coordinates_roles_retained_same_quota=True,
            previous_sampler_nested_selected_sets=True,seed_quota_is_not_final_node_cap=True,
            all_point_coverage_nonincrease=True,initial_parameters_equal_within_each_variant=True,
            original_CNN_fusion_L1_L2_equal_all_branches=True,support_group_exclusion_ids_equal=True,
            optimizer_continuity_verified=True,warmup_model_buffers_RNG_restored=True,
            original_inputs_saved_memory_sources_schedule_preserved=True),
        implementation_sha256=implementation_before,
        implementation_unchanged_during_run=True,
        visualization=dict(path=str(a.output/'visual_payload.json'), displayed_P=5,displayed_U=3,
            same_donor=True, typed_joint_edges=True, actual_forward_coordinates=True,
            anatomy_deduplicated_by_case=True, HTML_written=False),
        limitations=['This is a local DEBUG step4 reference, not trained-server ranking weights.',
            'Three same-tile updates demonstrate mechanical learning and resource cost, not epoch throughput or CP quality.',
            'Own newly encoded support/teacher is frozen inside the three-update prefix; post-prefix refresh is measured separately.',
            'Coverage distances describe spatial/CNN landmark representation, not tumor relevance or retained-information percentage.',
            'Single CUDA-device physical32 is controlled for this comparison; no multi-GPU production throughput measurement.',
            'Hard node selection indices are nondifferentiable; selected CNN features and actual message/readout modules receive gradients.'])
    with (a.output/'report.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(final,stream,ensure_ascii=False,indent=2,allow_nan=False)
    with (a.output/'visual_payload.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(payload,stream,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    print(json.dumps(dict(report=str(a.output/'report.json'),visual_payload=str(a.output/'visual_payload.json'),
        case=a.case, original_P=5, original_U=128, measured_continuous_updates_per_branch=3,
        full_training=False, full_evaluation=False),indent=2),flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run','assignment','assignment-receipt','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--case',required=True)
    for name in ('query-radius-mm','near-radius-mm','mid-radius-mm','cuda-gib','rss-gib','resident-gib','coverage-workspace-mib'):
        parser.add_argument('--'+name,type=float,required=True)
    parser.add_argument('--workers',type=int,required=True)
    parser.add_argument('--warmup-updates',type=int,required=True)
    parser.add_argument('--measured-updates',type=int,required=True)
    original_rng = rng_state()
    original_runtime = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        run(parser.parse_args())
    finally:
        restore_rng(original_rng)
        torch.set_num_threads(original_runtime[0])
        torch.use_deterministic_algorithms(original_runtime[1])
        torch.backends.cuda.matmul.allow_tf32 = original_runtime[2]
        torch.backends.cudnn.allow_tf32 = original_runtime[3]


if __name__ == '__main__':
    main()
