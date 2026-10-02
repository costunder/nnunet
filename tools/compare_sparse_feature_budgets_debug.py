"""Same native CT, CNN/graph weights and batch32: context48/96/192 DEBUG.

All133 original observations participate; graph geometry and complete supported
feature pool are invariant. No long training, checkpoint or production-ready
flag. Coverage distances are representation diagnostics, not CP accuracy.
"""
import argparse
import base64
import copy
import gc
import gzip
import json
import os
from pathlib import Path
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
from l0_local_cnn.data import Dataset, CropStore
from l0_regions.donor_learning import LiveContext, forward_loss, configuration
from l0_regions.training import FORMAT, hash_state, make_model
from l0_regions.training_data import Budget, sha, source_identity
import l0_sparse_feature.model as graph_module
from l0_sparse_feature.model import SparseFeatureProfile, SparseFeatureL0
from l0_sparse_feature.coverage import all_pool_distances, distance_stats
from tools.diagnose_local_cnn_learning import checkpoint_for
from tools.local_cnn_interaction_runtime import support_binding
from tools.verify_sparse_feature_ct_debug import bound_batch, anatomy, graph_scene, GraphLocalAdapter
from tools.v22_candidate_order import record_key
from tools.v22_rank_objective import ranking_metrics
from hiercp_v222.v1_execution import rng_state, restore_rng

PREFIXES = {'CNN': 'local.encoder.cnn', 'node_projection': 'local.encoder.node_project',
    'SAGE': 'local.encoder.blocks', 'graph_readout': 'local.encoder.scene_project',
    'fusion': 'local.encoder.fuse', 'L1': 'l1', 'L2': 'l2'}


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu(v) for v in value]
    return value


def stats_json(values):
    mean, p95, maximum = distance_stats(values).cpu().tolist()
    return dict(count=values.numel(), mean=mean, p95=p95, max=maximum)


class ForwardPhases:
    """CUDA events for existing numerical functions, with no alternate path."""
    def __init__(self, encoder):
        self.encoder = encoder
        self.events = {}
        self.handles = []
        self.contexts = []

    def begin(self, name):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.events.setdefault(name, []).append((start, end))

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
        for attribute, name in (('_select_context', 'selection'), ('_build_edges', 'edge_build')):
            context = patch.object(graph_module, attribute, self.wrap(getattr(graph_module, attribute), name))
            context.__enter__()
            self.contexts.append(context)
        self.handles.append(self.encoder.cnn.register_forward_pre_hook(lambda *_: self.begin('CNN')))
        self.handles.append(self.encoder.cnn.register_forward_hook(lambda *_: self.end('CNN')))
        self.handles.append(self.encoder.blocks[0].register_forward_pre_hook(lambda *_: self.begin('SAGE3')))
        self.handles.append(self.encoder.blocks[-1].register_forward_hook(lambda *_: self.end('SAGE3')))
        return self

    def __exit__(self, typ, value, trace):
        for handle in self.handles:
            handle.remove()
        for context in reversed(self.contexts):
            context.__exit__(typ, value, trace)

    def seconds(self):
        torch.cuda.synchronize()
        return {name: sum(start.elapsed_time(end) for start, end in pairs) / 1000
            for name, pairs in self.events.items()}


@torch.no_grad()
def assert_same_pool(previous, current):
    for name in ('features', 'relative_mm', 'eligible', 'band'):
        if not torch.equal(previous[name], current[name]):
            raise AssertionError('Node budget changed the original complete pool: ' + name)


@torch.no_grad()
def assert_nested(previous, current, previous_distances, current_distances):
    for role in (1, 2, 3):
        old_mask = previous['node_mask'] & (previous['role'] == role)
        new_mask = current['node_mask'] & (current['role'] == role)
        old_slots = (previous['role'][0] == role).nonzero().flatten()
        new_slots = (current['role'][0] == role).nonzero().flatten()
        old = previous['xyz_native'][:, old_slots]
        new = current['xyz_native'][:, new_slots]
        matches = (old[:, :, None] == new[:, None]).all(-1) & new_mask[:, None, new_slots]
        if not bool((matches.any(-1) | ~old_mask[:, old_slots]).all()):
            raise AssertionError('A larger quota lost a smaller selected set')
        enough = current['statistics']['band_pool_counts'][:, role - 1] > len(new_slots)
        if bool(enough.any()) and not torch.equal(old[enough], new[enough, :len(old_slots)]):
            raise AssertionError('FPS ordered prefix changed for a non-retain-all band')
        before, after = previous_distances[role - 1], current_distances[role - 1]
        if not torch.equal(before['mask'], after['mask']):
            raise AssertionError('Coverage pool membership changed')
        for name in ('feature_deficit', 'spatial_mm', 'joint'):
            # FP32 distance via bmm is algebraically exact but different K may
            # choose a different CUDA kernel. This is fixed numeric tolerance,
            # never an admission profile or a PASS-seeking threshold change.
            tolerance = 1e-4 if name == 'joint' else 1e-3 if name == 'spatial_mm' else 2e-6
            if not bool((after['values'][name][after['mask']] <= before['values'][name][before['mask']] + tolerance).all()):
                raise AssertionError('All-point nearest coverage increased with a nested selected set: ' + name)


def measured_update(net, initial, batch, support, plan, targets, context, tile, base):
    net.load_state_dict(initial, strict=True)
    net.zero_grad(set_to_none=True)
    net.train()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    optimizer = torch.optim.AdamW(net.parameters(), lr=base['training']['lr'],
        weight_decay=base['training']['weight_decay'])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with ForwardPhases(net.local.encoder) as phases:
        loss, terms = forward_loss(net, batch, support, plan, targets, None, context,
            configuration(), indices=tile)
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    backward_started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - backward_started
    check_started = time.perf_counter()
    finite = []
    for name, parameter in net.named_parameters():
        if parameter.grad is None:
            raise AssertionError('Missing trainable gradient: ' + name)
        finite.append(torch.isfinite(parameter.grad).all())
    if not bool(torch.stack(finite).all()):
        raise FloatingPointError('Nonfinite complete objective gradient')
    norms = {name: torch.stack([p.grad.float().square().sum() for key, p in net.named_parameters()
        if key.startswith(prefix)]).sum().sqrt() for name, prefix in PREFIXES.items()}
    if not bool(torch.stack(list(norms.values())).gt(0).all()):
        raise AssertionError('Complete objective did not reach every module')
    torch.nn.utils.clip_grad_norm_(net.parameters(), base['training']['grad_clip'])
    optimizer.step()
    torch.cuda.synchronize()
    checked_optimizer_seconds = time.perf_counter() - check_started
    full_update_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    # Diagnostic copies and state hashes are OUTSIDE the measured update.
    deltas = {name: torch.stack([(p.detach() - initial[key]).float().square().sum()
        for key, p in net.named_parameters() if key.startswith(prefix)]).sum().sqrt()
        for name, prefix in PREFIXES.items()}
    if not bool(torch.stack(list(deltas.values())).gt(0).all()):
        raise AssertionError('Connected optimizer module was not updated')
    result = dict(loss=float(loss.detach()), terms=cpu(terms), gradient_norms=cpu(norms),
        parameter_delta_norms=cpu(deltas), forward_seconds=forward_seconds,
        backward_seconds=backward_seconds, finite_clip_optimizer_seconds=checked_optimizer_seconds,
        full_update_seconds=full_update_seconds, forward_phases_seconds=phases.seconds(),
        peak_allocated_bytes=peak, peak_reserved_bytes=reserved)
    del optimizer, loss, terms
    net.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def score_vectors(net, vectors, support, plan, rows):
    net.eval()
    output = net.predict_embeddings(torch.cat(vectors), net.prepare_support(*support, cluster_plan=plan))
    logits = output['logits'].float()
    score = logits[:, 1] - logits[:, 0]
    truth = torch.tensor([row['target'] for row in rows], device=score.device)
    difference = score[truth == 1, None] - score[None, truth == 0]
    metric, details = ranking_metrics(score.cpu(), truth.cpu(), [row['case_id'] for row in rows],
        candidate_keys=[record_key(row) for row in rows])
    return dict(metrics=metric, details=details, pair_win=float((difference > 0).float().mean()),
        exact_tie=float((difference == 0).float().mean()), score_std=float(score.std(unbiased=False)),
        accuracy_evidence=False, scope='same untrained graph initialization; mechanical full-case scoring only')


def run(a):
    if a.output.exists() or a.visual.exists():
        raise FileExistsError('Existing evidence preserved; select fresh output/visual paths')
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; no CPU fallback')
    if a.workers < 2 or not 0 < a.resident_gib < a.rss_gib or a.measure_repeats != 3:
        raise ValueError('Explicit parallel readers/RAM headroom and three measured clone updates required')
    torch.set_num_threads(a.workers)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    free, total = torch.cuda.mem_get_info()
    budget = Budget(int(a.cuda_gib * 2**30), int(a.rss_gib * 2**30))
    if not 0 < budget.cuda_bytes < min(free, total):
        raise ValueError('Explicit CUDA budget must leave actual free VRAM headroom')
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes / total)
    resources = dict(gpu=torch.cuda.get_device_name(), gpu_count=torch.cuda.device_count(),
        total_VRAM=total, free_at_start=free, cpu_logical=psutil.cpu_count(),
        available_RAM=psutil.virtual_memory().available, cuda_limit=budget.cuda_bytes,
        RSS_limit=budget.rss_bytes, resident_limit=int(a.resident_gib * 2**30), workers=a.workers)
    source_before = source_identity()
    cp = checkpoint_for(a.run.resolve())
    cp_sha, assignment_sha = sha(cp), sha(a.assignment)
    receipt = json.loads(a.assignment_receipt.read_text(encoding='utf-8'))
    if receipt.get('assignment_sha256') != assignment_sha or receipt.get('case') != a.case:
        raise ValueError('Original complete-case assignment receipt mismatch')
    saved = torch.load(cp, map_location='cpu', weights_only=False)
    if saved.get('format') != FORMAT or not saved['identity'].get('debug'):
        raise ValueError('Integrity-verified local DEBUG checkpoint required')
    if saved['identity']['source'] != source_before or hash_state({k: v for k, v in saved.items()
        if k != 'content_sha256'}) != saved['content_sha256']:
        raise ValueError('Existing checkpoint source/content integrity mismatch')
    ds = Dataset(a.run / 'inventory/index.json', 'inner_train', True)
    if saved['identity']['cache_sha256'] != sha(ds.path):
        raise ValueError('Native inventory binding mismatch')
    observed = [row for row in ds.rows if row['case_id'] == a.case]
    if not observed:
        raise ValueError('Explicit original DEBUG cohort case required')
    donor = {key: observed[0][key] for key in ('donor_case_id', 'donor_component', 'donor_group')}
    original = [row for row in json.loads(a.assignment.read_text(encoding='utf-8')) if row['case_id'] == a.case]
    positive = [row for row in original if row['target'] == 1]
    unobserved = [row for row in original if row['target'] == 0]
    raw_record = next(row for row in ds.meta['raw_records'] if row['case_id'] == a.case)
    if len(unobserved) != 128 or len(positive) != len(raw_record['positives']) or not positive:
        raise ValueError('Every original P and all128 U required')
    rows = [dict(row, **donor, bounds={'edges': i}) for i, row in enumerate(positive + unobserved)]
    context_rows = rows + [row for row in ds.rows if row['case_id'] != a.case]
    context = LiveContext(SimpleNamespace(rows=context_rows), 32)
    tile = next(ids for ids in context.order if len(ids) == 32
        and {context_rows[i]['case_id'] for i in ids} == {a.case}
        and 0 < sum(context_rows[i]['target'] for i in ids) < 32)
    reference_net = make_model(ds, budget, True, 'retained')
    reference_net.load_state_dict(saved['model'], strict=True)
    models, supports, plans, reports, state_hashes = {}, {}, {}, {}, {}
    for count in (48, 96, 192):
        net = copy.deepcopy(reference_net)
        profile = SparseFeatureProfile(context_nodes_per_band=count // 3,
            query_radius_mm=a.query_radius_mm, near_radius_mm=a.near_radius_mm, mid_radius_mm=a.mid_radius_mm)
        net.local = GraphLocalAdapter(SparseFeatureL0(reference_net.local, profile, budget=budget)).cuda()
        net.eval()
        models[count] = net
        state_hashes[count] = hash_state(net.state_dict())
        reports[count] = dict(context_nodes=count, profile=vars(profile), graph_audits=[], scenes=[],
            total_parameters=sum(p.numel() for p in net.parameters()),
            trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad))
    if len(set(state_hashes.values())) != 1:
        raise AssertionError('Initial CNN/GNN/L1/L2 weights differ between budgets')
    del reference_net
    reader = CropStore(ds.meta, a.workers, int(a.resident_gib * 2**30), budget.rss_bytes)
    started = time.perf_counter()
    support_parts = {count: [] for count in models}
    with torch.no_grad():
        for start in range(0, len(ds.rows), 32):
            ids = list(range(start, min(start + 32, len(ds.rows))))
            batch = bound_batch(reader, [ds.rows[i] for i in ids], ids)
            for count, net in models.items():
                support_parts[count].append(net.local(batch).detach())
        for count, net in models.items():
            memory = copy.deepcopy(saved['state']['memory'])
            memory['embeddings'] = torch.cat(support_parts[count])
            memory['owners'], memory['classes'] = memory['owners'].cuda(), memory['classes'].cuda()
            support, ids, _ = support_binding(memory, ds.rows, rows[0]['patient_group'])
            supports[count] = support
            plans[count] = net.fit_support_clusters(*support)
            reports[count]['support'] = dict(ids=ids, records=len(ids), reencoded=True,
                teacher_refitted=True, old_mean_memory_used=False, scope='complete eligible saved DEBUG support only')
    support_seconds = time.perf_counter() - started
    del support_parts, memory, batch
    if len({tuple(report['support']['ids']) for report in reports.values()}) != 1:
        raise AssertionError('Node budget changed support ownership/observations')
    display = set(range(len(positive))) | set(range(len(positive), len(positive) + 3))
    anatomy_cache, displayed, donor_scenes = {}, {count: [] for count in models}, {}
    vectors = {count: [] for count in models}
    coverage_parts = {count: {name: [] for name in ('feature_deficit', 'spatial_mm', 'joint')} for count in models}
    donor_coverage = {}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for start in tqdm(range(0, len(rows), 32), desc='same CT 48/96/192 DEBUG', unit='batch'):
            ids = list(range(start, min(start + 32, len(rows))))
            batch = bound_batch(reader, [rows[i] for i in ids], ids)
            baseline_pool, previous_graph, previous_distances = None, None, None
            for count, net in models.items():
                torch.cuda.synchronize()
                begin = time.perf_counter()
                with ForwardPhases(net.local.encoder) as phases:
                    output, graph = net.local.encoder(batch, recipient_centers_native=batch.recipient_centers,
                        donor_centers_native=batch.donor_centers, return_graph=True, return_pool=True)
                torch.cuda.synchronize()
                forward_seconds = time.perf_counter() - begin
                if baseline_pool is None:
                    baseline_pool = graph['coverage_pool']
                else:
                    assert_same_pool(baseline_pool, graph['coverage_pool'])
                begin = time.perf_counter()
                distances = all_pool_distances(graph, workspace_bytes=int(a.coverage_workspace_mib * 2**20))
                torch.cuda.synchronize()
                coverage_seconds = time.perf_counter() - begin
                if previous_graph is not None:
                    assert_nested(previous_graph, graph, previous_distances, distances)
                scene_metrics = {key: torch.zeros_like(graph['coverage_pool']['eligible'], dtype=torch.float32)
                    for key in ('feature_deficit', 'spatial_mm', 'joint')}
                for entry in distances:
                    mask = entry['mask']
                    for name, value in entry['values'].items():
                        coverage_parts[count][name].append(value[len(batch):][mask[len(batch):]])
                        scene_metrics[name] += torch.where(mask, value, 0)
                complete_mask = graph['coverage_pool']['eligible']
                per_scene = {name: distance_stats(value, complete_mask) for name, value in scene_metrics.items()}
                per_scene_cpu = cpu(per_scene)
                edge_counts = graph['adjacency'].sum((1, 2)) // 2
                reports[count]['graph_audits'].append(dict(statistics=cpu(graph['statistics']),
                    actual_node_counts=cpu(graph['node_mask'].sum(-1)), undirected_edge_counts=cpu(edge_counts),
                    original_indices=ids, forward_seconds=forward_seconds,
                    forward_phases_seconds=phases.seconds(), coverage_seconds=coverage_seconds))
                if count not in donor_coverage:
                    donor_coverage[count] = {name: stats_json(value[0][complete_mask[0]])
                        for name, value in scene_metrics.items()}
                for j, index in enumerate(ids):
                    scene_index = len(batch) + j
                    reports[count]['scenes'].append(dict(id=rows[index]['id'], kind='P' if rows[index]['target'] else 'U',
                        node_count=int(graph['node_mask'][scene_index].sum()),
                        undirected_edge_count=int(edge_counts[scene_index]),
                        full_eligible_count=int(complete_mask[scene_index].sum()),
                        coverage={name: dict(zip(('mean', 'p95', 'max'), summary[scene_index]))
                            for name, summary in per_scene_cpu.items()},
                        bands=[dict(role=entry['role'], pool_count=int(entry['full_counts'][scene_index]),
                            selected_count=int(entry['selected_counts'][scene_index]),
                            metrics={name: dict(zip(('mean', 'p95', 'max'), cpu(summary[scene_index])))
                                for name, summary in entry['summaries'].items()}) for entry in distances]))
                    if index not in display:
                        continue
                    row = rows[index]
                    for case in (row['case_id'], row['donor_case_id']):
                        if case not in anatomy_cache:
                            anatomy_cache[case] = anatomy(reader.raw.cache[case])
                    scene = graph_scene(graph, scene_index, batch, reader.raw.cache[row['case_id']],
                        row['case_id'], anatomy_cache[row['case_id']])
                    scene['stats']['coverage'] = reports[count]['scenes'][-1]['coverage']
                    displayed[count].append(dict(id=row['id'], kind='P' if row['target'] else 'U', scene=scene))
                    if count not in donor_scenes:
                        donor_scenes[count] = graph_scene(graph, j, batch, reader.raw.cache[row['donor_case_id']],
                            row['donor_case_id'], anatomy_cache[row['donor_case_id']])
                vectors[count].append(output.detach())
                previous_graph, previous_distances = graph, distances
            del baseline_pool, previous_graph, previous_distances, graph, distances, batch, scene_metrics
    coverage_peak = torch.cuda.max_memory_allocated()
    for count, net in models.items():
        reports[count]['recipient_all_points'] = {name: stats_json(torch.cat(parts))
            for name, parts in coverage_parts[count].items()}
        reports[count]['donor_unique_scene'] = donor_coverage[count]
        reports[count]['initial_full_case_score'] = score_vectors(net, vectors[count], supports[count], plans[count], rows)
    del coverage_parts, vectors
    gc.collect()
    # Same actual batch and fresh initial model/Adam state on EVERY repeat.
    batch = bound_batch(reader, [context_rows[i] for i in tile], tile)
    targets = torch.tensor([context_rows[i]['target'] for i in tile], device='cuda', dtype=torch.long)
    for count, net in models.items():
        initial = {name: parameter.detach().clone() for name, parameter in net.state_dict().items()}
        reports[count]['warmup'] = measured_update(net, initial, batch, supports[count], plans[count],
            targets, context, tile, ds.meta['base'])
        runs = []
        for _ in tqdm(range(a.measure_repeats), desc=f'context {count} cloned updates DEBUG', unit='update'):
            runs.append(measured_update(net, initial, batch, supports[count], plans[count],
                targets, context, tile, ds.meta['base']))
        reports[count]['measured_updates'] = runs
        reports[count]['update_summary'] = {name: dict(mean=float(np.mean([run[name] for run in runs])),
            median=float(np.median([run[name] for run in runs])), min=min(run[name] for run in runs),
            max=max(run[name] for run in runs)) for name in ('full_update_seconds', 'forward_seconds', 'backward_seconds',
                'finite_clip_optimizer_seconds', 'peak_allocated_bytes')}
        net.load_state_dict(initial, strict=True)
        net.eval()
        if hash_state(net.state_dict()) != state_hashes[count]:
            raise AssertionError('Cloned update did not restore its identical initial state')
        del initial
    if sha(cp) != cp_sha or sha(a.assignment) != assignment_sha or source_identity() != source_before:
        raise AssertionError('Original checkpoint, assignment or production files changed')
    budgets = []
    for count, report in reports.items():
        coverage = report['recipient_all_points']
        budgets.append(dict(context_nodes=count, donor=donor_scenes[count], candidates=displayed[count],
            metrics=dict(joint_p95=coverage['joint']['p95'], feature_deficit_p95=coverage['feature_deficit']['p95'],
                spatial_mm_p95=coverage['spatial_mm']['p95'],
                full_update_seconds=report['update_summary']['full_update_seconds']['median'],
                peak_GiB=report['update_summary']['peak_allocated_bytes']['max'] / 2**30)))
    payload = dict(case=a.case, margin_mm=ds.meta['local_cnn']['margin_mm'], budgets=budgets,
        weight_source=dict(label=f'CNN: 로컬 DEBUG saved step {saved["state"].get("step")} · GraphSAGE/readout: 동일 새 초기값 seed42'),
        scope='실제 forward · 간 내부 CT · 133개 후보 거리 진단 · CP 정확도 검증 전')
    packed = base64.b64encode(gzip.compress(json.dumps(payload, separators=(',', ':'), allow_nan=False).encode(), 9)).decode()
    html = (ROOT / 'tools/sparse-feature-budget.template.html').read_text(encoding='utf-8').replace('__DATA__', packed)
    if len(html.encode()) >= 1_000_000:
        raise ValueError('Visualization payload exceeds1MB; do not reduce model/data')
    implementation = ('l0_sparse_feature/model.py', 'l0_sparse_feature/coverage.py',
        'tools/compare_sparse_feature_budgets_debug.py', 'tools/sparse-feature-budget.template.html',
        'tests/test_sparse_feature_graph.py', 'tests/test_sparse_feature_coverage.py')
    final = dict(debug=True, actual_CT=True, actual_CUDA=True, production_ready=False,
        checkpoint_written=False, production_default_changed=False, production_training_started=False,
        full_training=False, full_evaluation=False, case=a.case, original_P=len(positive), original_U=len(unobserved),
        complete_case_observations=len(rows), complete_DEBUG_context=len(context_rows), physical_batch=32,
        effective_batch=32, gradient_accumulation=1, warmup_updates_per_clone=1, measured_fresh_updates_per_clone=3,
        original_tile_indices=tile, original_assignment_sha256=assignment_sha,
        weight_source=dict(checkpoint=str(cp), sha256=cp_sha, saved_step=saved['state'].get('step'),
            CNN='local DEBUG saved weights', GNN='same fresh seed42 weights', initial_model_sha256=state_hashes),
        maintained=dict(CNN_depth_channels=True, GNN_layers128D=True, L1_L2_equations=True, full_objective=True,
            P_U_definition=True, same_donor=True, candidate128=True, Basic_CP80_seed42=True, original_masks=True),
        budgets={str(k): v for k, v in reports.items()}, resources=dict(**resources,
            coverage_peak_allocated=coverage_peak, RSS=psutil.Process().memory_info().rss),
        precision=dict(dtype='FP32', autocast=False, TF32=False, deterministic=True),
        timing=dict(support_reencoding_teacher_seconds=support_seconds,
            measured_update_scope='prebound physical32 input: current CNN/selection/GNN/L1/L2/all loss/backward/finite+clip/Adam; excludes load/transfer/checkpoint save and diagnostic hash/delta copies',
            production_epoch_comparison=False),
        assertions=dict(exact_complete_pool_equal_all_budgets=True, nested_selected_sets=True,
            ordered_FPS_prefix_when_pool_exceeds64=True, all_point_distance_nonincrease=True,
            model_initial_weights_equal=True, support_record_ids_equal=True, original_inputs_preserved=True),
        implementation_sha256={name: sha(ROOT / name) for name in implementation},
        limitations=['CNN/GNN are not the trained server ranking model.',
            'Coverage decrease is expected for nested landmarks, not proof of cancer-relevant features or improved ranking.',
            'One complete133-observation case and saved DEBUG support are not full-dataset accuracy or epoch performance.',
            'The reported pooled percentile is over recipient eligible points; donor repetitions are not mixed into it.',
            'Coverage metric uses all supported points. Coarse unsupported positions remain explicit in raw audits.'],
        visual=dict(path=str(a.visual), displayed_P=len(positive), displayed_U=3, anatomy_payload_published=False,
            logical_UTF8_bytes=len(html.encode())))
    a.output.mkdir(parents=True, exist_ok=False)
    with a.visual.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(html)
    final['visual']['physical_file_bytes'] = a.visual.stat().st_size
    final['visual']['sha256'] = sha(a.visual)
    with (a.output / 'report.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(final, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with (a.output / 'visual_data.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(report=str(a.output / 'report.json'), visual=str(a.visual),
        case=a.case, original_P=len(positive), original_U=len(unobserved),
        comparisons=[dict(context_nodes=b['context_nodes'], **b['metrics']) for b in budgets]),
        ensure_ascii=False, indent=2), flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'assignment', 'assignment-receipt', 'output', 'visual'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', required=True)
    for name in ('query-radius-mm', 'near-radius-mm', 'mid-radius-mm', 'cuda-gib', 'rss-gib', 'resident-gib', 'coverage-workspace-mib'):
        parser.add_argument('--' + name, type=float, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--measure-repeats', type=int, required=True)
    original = rng_state()
    try:
        run(parser.parse_args())
    finally:
        restore_rng(original)


if __name__ == '__main__':
    main()
