"""Continuous real-CT DEBUG learning for explicit 48/96/192 L0 candidates.

This is NOT optimal-node admission or a server/full-dataset training launcher.
All original P and128U of every existing DEBUG case are restored. Full loss,
same-donor policy, physical32 and original held-out split remain unchanged.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
import torch
from tqdm import tqdm
from hiercp_v222.v1_execution import rng_state, restore_rng
from l0_local_cnn.data import Dataset, CropStore
from l0_regions.donor_learning import LiveContext, groups, configuration
from l0_regions.donor_learning import forward_loss
from l0_regions.training import FORMAT, hash_state, make_model
from l0_regions.training_data import Budget, sha
from l0_regions.training_data import source_identity
from l0_sparse_feature.model import SparseFeatureProfile, SparseFeatureL0
from l0_sparse_feature.learning import expand_rows, build_memory, evaluate_current
from l0_sparse_feature.learning import validate_train_memory
from tools.diagnose_local_cnn_learning import checkpoint_for
from tools.local_cnn_interaction_runtime import support_binding
from tools.verify_sparse_feature_ct_debug import GeometryBatch, GraphLocalAdapter
from tools.compare_sparse_feature_budgets_debug import PREFIXES


def new_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def scalar(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar(v) for v in value]
    return value


class AuditedGraphLocalAdapter(GraphLocalAdapter):
    """Record the first real batched graph without a second CNN execution."""
    def __init__(self, encoder):
        super().__init__(encoder)
        self.first_input = None

    def forward(self, batch):
        if self.first_input is not None:
            return super().forward(batch)
        if not isinstance(batch, GeometryBatch):
            raise TypeError('Explicit verified original pair geometry required')
        output, graph = self.encoder(batch,
            recipient_centers_native=batch.recipient_centers,
            donor_centers_native=batch.donor_centers, return_graph=True)
        self.first_input = dict(images_shape=list(batch.images.shape),
            organ_shape=list(batch.organ.shape), physical_pairs=len(batch),
            scene_nodes=scalar(graph['node_mask'].sum(-1)),
            scene_directed_edges=scalar(graph['adjacency'].sum((1, 2))),
            statistics=scalar(graph['statistics']),
            scope='First complete real train-memory batch; not whole-cohort extrema')
        print('ACTUAL DEBUG GRAPH:', json.dumps(dict(
            images_shape=self.first_input['images_shape'], physical_pairs=len(batch),
            node_range=[min(self.first_input['scene_nodes']), max(self.first_input['scene_nodes'])],
            directed_edge_range=[min(self.first_input['scene_directed_edges']),
                                 max(self.first_input['scene_directed_edges'])])), flush=True)
        return output


class BoundLoader:
    """One CPU producer ahead; raw CT decompression itself uses reader workers."""
    def __init__(self, rows, store):
        self.rows, self.store = rows, store

    def get(self, indices):
        rows = [self.rows[i] for i in indices]
        base = self.store.batch(rows, indices)
        recipient = torch.tensor([r['center'] for r in rows], dtype=torch.float32).pin_memory()
        donor = torch.tensor(np.stack([self.store.donor_bounds(r)[0] for r in rows]),
                             dtype=torch.float32).pin_memory()
        return base, recipient, donor

    def batches(self, order):
        with ThreadPoolExecutor(max_workers=1) as pool:
            iterator = iter(order)
            first = next(iterator, None)
            if first is None:
                raise ValueError('Empty complete DEBUG schedule')
            future = pool.submit(self.get, first)
            for indices in iterator:
                current = future.result()
                future = pool.submit(self.get, indices)
                yield current
            yield future.result()


def bind_loaded(loaded):
    base, recipient, donor = loaded
    return GeometryBatch(base.to('cuda'), recipient.to('cuda', non_blocking=True),
                         donor.to('cuda', non_blocking=True))


@torch.no_grad()
def refresh_memory(net, loader, rows, meta, budget):
    modes = [(m, m.training) for m in net.modules()]
    net.eval()
    start = time.perf_counter()
    parts = []
    order = [list(range(i, min(i + 32, len(rows)))) for i in range(0, len(rows), 32)]
    try:
        for indices, loaded in zip(order, loader.batches(order)):
            batch = bind_loaded(loaded)
            if batch.indices.tolist() != indices:
                raise ValueError('Complete train memory order changed')
            parts.append(net.local(batch).detach().float())
            budget.check()
            del batch, loaded
        memory = build_memory(rows, torch.cat(parts))
        validate_train_memory(memory, meta)
        torch.cuda.synchronize()
        return memory, time.perf_counter() - start
    finally:
        for module, mode in modes:
            module.training = mode


def evaluate_preserving(net, optimizer, train_rows, val_rows, store, memory, budget):
    """Evaluation must not change model buffers, Adam moments or training RNG."""
    saved_rng = rng_state()
    model_hash = hash_state(net.state_dict())
    optimizer_hash = hash_state(optimizer.state_dict())
    start = time.perf_counter()
    try:
        result = {
            'train': evaluate_current(net, train_rows, store, memory, budget, batch=32),
            'validation': evaluate_current(net, val_rows, store, memory, budget, batch=32),
        }
    finally:
        restore_rng(saved_rng)
    if hash_state(net.state_dict()) != model_hash:
        raise AssertionError('Evaluation changed current weights/buffers')
    if hash_state(optimizer.state_dict()) != optimizer_hash:
        raise AssertionError('Evaluation changed continuous Adam state')
    result['seconds'] = time.perf_counter() - start
    result['model_optimizer_RNG_preserved'] = True
    return result


def norms(net, initial=None):
    if initial is None:
        terms = {name: torch.stack([p.grad.float().square().sum()
            for key, p in net.named_parameters() if key.startswith(prefix)]).sum().sqrt()
            for name, prefix in PREFIXES.items()}
    else:
        terms = {name: torch.stack([(p.detach() - initial[key]).float().square().sum()
            for key, p in net.named_parameters() if key.startswith(prefix)]).sum().sqrt()
            for name, prefix in PREFIXES.items()}
    if not bool(torch.stack(list(terms.values())).isfinite().all()):
        raise FloatingPointError('Nonfinite connected module norm')
    if not bool(torch.stack(list(terms.values())).gt(0).all()):
        raise AssertionError('Complete objective failed to reach/update a module')
    return scalar(terms)


def optimize(net, optimizer, query, support, plan, context, indices, base, expected_step):
    targets = torch.tensor([context.rows[i]['target'] for i in indices], device='cuda')
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss, terms = forward_loss(net, query, support, plan, targets, None, context,
                              configuration(), indices=indices)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite complete live loss')
    loss.backward()
    parameters = list(net.named_parameters())
    if any(p.grad is None for _, p in parameters):
        raise AssertionError('Trainable parameter missing objective gradient')
    if not bool(torch.stack([torch.isfinite(p.grad).all() for _, p in parameters]).all()):
        raise FloatingPointError('Nonfinite full objective gradient')
    gradients = norms(net)
    grad_before_clip = torch.nn.utils.clip_grad_norm_(net.parameters(), base['training']['grad_clip'])
    optimizer.step()
    # Check continuity: never reset Adam or load the initial model each update.
    steps = torch.stack([state['step'].to('cuda') for state in optimizer.state.values()])
    if len(steps) != len(parameters) or not bool((steps == expected_step).all()):
        raise AssertionError('Adam moments/step did not continue across real updates')
    torch.cuda.synchronize()
    return dict(step=expected_step, loss=float(loss.detach()), terms=scalar(terms),
        gradient_norms=gradients, grad_before_clip=float(grad_before_clip),
        adam_step_min=float(steps.min()), adam_step_max=float(steps.max()),
        compute_update_seconds=time.perf_counter() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated())


def run(a):
    if a.output.exists():
        raise FileExistsError('Prior results preserved; choose a new DEBUG output')
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; no CPU fallback')
    if a.debug_cycles != 3 or a.workers < 2 or not 0 < a.resident_gib < a.rss_gib:
        raise ValueError('Explicit three complete DEBUG cycles and parallel/RAM profile required')
    before_source = source_identity()
    checkpoint = checkpoint_for(a.run.resolve())
    checkpoint_hash, assignment_hash = sha(checkpoint), sha(a.assignment)
    receipt = json.loads(a.assignment_receipt.read_text(encoding='utf-8'))
    if receipt.get('assignment_sha256') != assignment_hash:
        raise ValueError('Original assignment receipt mismatch')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('format') != FORMAT or not saved['identity'].get('debug'):
        raise ValueError('Original verified DEBUG checkpoint required')
    if saved['identity']['source'] != before_source or hash_state({k: v for k, v in saved.items()
        if k != 'content_sha256'}) != saved['content_sha256']:
        raise ValueError('Saved source/model/optimizer content integrity mismatch')
    ds = Dataset(a.run / 'inventory/index.json', 'inner_train', True)
    if saved['identity']['cache_sha256'] != sha(ds.path):
        raise ValueError('Original inventory binding mismatch')
    meta = ds.meta
    original = json.loads(a.assignment.read_text(encoding='utf-8'))
    debug_cases = sorted({r['case_id'] for r in meta['records']})
    train_cases = [c for c in debug_cases if c in meta['split']['inner_train']]
    val_cases = [c for c in debug_cases if c in meta['split']['inner_val']]
    if set(train_cases) & set(val_cases) or set(train_cases + val_cases) != set(debug_cases):
        raise ValueError('Original DEBUG case split coverage mismatch')
    if not train_cases or not val_cases:
        raise ValueError('Actual original train and held-out DEBUG cases required')
    donors = {}
    for row in meta['records']:
        donor = {key: row[key] for key in ('donor_case_id', 'donor_component', 'donor_group')}
        if row['case_id'] in donors and donors[row['case_id']] != donor:
            raise ValueError('Saved same-donor DEBUG contract differs within a case')
        donors[row['case_id']] = donor
    train_rows = expand_rows(meta, original, train_cases, {c: donors[c] for c in train_cases})
    val_rows = expand_rows(meta, original, val_cases, {c: donors[c] for c in val_cases})
    train_ds = SimpleNamespace(rows=train_rows, meta=meta)
    context = LiveContext(train_ds, 32)
    schedules = [list(groups(train_ds, 32, seed=meta['config']['seed'], epoch=e))
                 for e in range(a.debug_cycles)]
    expected_steps = sum(len(v) for v in schedules)
    free, total = torch.cuda.mem_get_info()
    budget = Budget(int(a.cuda_gib * 2**30), int(a.rss_gib * 2**30))
    if not 0 < budget.cuda_bytes < min(free, total):
        raise ValueError('Explicit CUDA budget must leave actual headroom')
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes / total)
    torch.set_num_threads(a.workers)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = make_model(ds, budget, True, 'retained')
    reference.load_state_dict(saved['model'], strict=True)
    torch.set_num_threads(a.workers)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    store = CropStore(meta, a.workers, int(a.resident_gib * 2**30), budget.rss_bytes)
    loader = BoundLoader(train_rows, store)
    report = dict(debug=True, actual_CT=True, actual_CUDA=True, optimal_graph_size_established=False,
        full_training=False, full_evaluation=False, production_ready=False,
        production_checkpoint_written=False, original_production_model_changed=False,
        source_checkpoint=str(checkpoint), source_checkpoint_sha256=checkpoint_hash,
        initial_CNN_saved_DEBUG_step=saved['state']['step'], assignment_sha256=assignment_hash,
        original_inventory_sha256=sha(ds.path), same_donor_overrides=donors,
        train_cases=train_cases, validation_cases=val_cases,
        train_observations=len(train_rows), validation_observations=len(val_rows),
        train_P=sum(r['target'] for r in train_rows), train_U=sum(1-r['target'] for r in train_rows),
        validation_P=sum(r['target'] for r in val_rows), validation_U=sum(1-r['target'] for r in val_rows),
        complete_original_case_P_and_U128=True, original_DEBUG_case_membership_preserved=True,
        original_full_observations=len(original), physical_batch=32, effective_batch=32,
        gradient_accumulation=1, complete_DEBUG_cycles=a.debug_cycles,
        optimization_steps_per_variant=expected_steps, schedule=context.audit,
        exact_schedule_record_ids=[[[train_rows[i]['id'] for i in ids] for ids in order] for order in schedules],
        loss_contract=configuration(), original_optimizer_settings=meta['base']['training'],
        maintained=['Basic CP80/seed42/split/P-U definitions/allP/U128/native CT/original masks',
                    'same donor per case/CNN8[12,24,32]/GNN3x128D/L1/L2/full objective/physical32'],
        resources=dict(gpu=torch.cuda.get_device_name(), gpu_count=torch.cuda.device_count(),
            total_VRAM=total, free_at_start=free, cpu_logical=psutil.cpu_count(),
            available_RAM=psutil.virtual_memory().available, cuda_limit=budget.cuda_bytes,
            RSS_limit=budget.rss_bytes, resident_limit=int(a.resident_gib * 2**30), workers=a.workers),
        precision=dict(dtype='FP32', autocast=False, TF32=False, deterministic=True),
        limitations=['Continuous short DEBUG fitting, not graph-size optimum or CP efficacy.',
            'Existing local CNN saved step4; fresh seeded SAGE/readout and fresh AdamW.',
            'Four existing DEBUG train cases and one held-out case are not full84/21 evaluation.',
            'Candidate comparisons within the same CT are correlated, not independent case samples.',
            'Validation is never used in support/update/teacher fitting; final step predeclared.',
            'Natural last tiles are recorded; physical32 was not silently reduced.'], budgets={})
    a.output.mkdir(parents=True)
    new_json(a.output / 'execution_contract.json', report)
    implementation = {name: sha(ROOT / name) for name in (
        'l0_sparse_feature/model.py', 'l0_sparse_feature/learning.py',
        'tools/compare_sparse_feature_learning_debug.py', 'tests/test_sparse_feature_learning.py')}
    initial_hashes = {}
    for count in (48, 96, 192):
        net = copy.deepcopy(reference)
        profile = SparseFeatureProfile(count // 3, a.query_radius_mm, a.near_radius_mm, a.mid_radius_mm)
        net.local = AuditedGraphLocalAdapter(SparseFeatureL0(reference.local, profile, budget=budget)).cuda()
        net.eval()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        initial = {key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                   for key, value in net.state_dict().items()}
        initial_hashes[count] = hash_state(initial)
        if len(set(initial_hashes.values())) != 1:
            raise AssertionError('Budgets did not begin from identical initial weights')
        optimizer = torch.optim.AdamW(net.parameters(), lr=meta['base']['training']['lr'],
                                     weight_decay=meta['base']['training']['weight_decay'])
        if optimizer.state:
            raise AssertionError('Fresh initial AdamW required')
        result = dict(profile=vars(profile), initial_model_sha256=initial_hashes[count],
            total_parameters=sum(p.numel() for p in net.parameters()),
            trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
            continuous_optimizer=True, actual_updates=[], cycles=[], evaluation=[])
        memory, refresh_seconds = refresh_memory(net, loader, train_rows, meta, budget)
        result['initial_memory_seconds'] = refresh_seconds
        result['actual_initial_input'] = net.local.first_input
        initial_evaluation = evaluate_preserving(net, optimizer, train_rows, val_rows, store, memory, budget)
        result['evaluation'].append(dict(cycle=0, step=0, **initial_evaluation))
        step = 0
        for cycle, order in enumerate(schedules, 1):
            net.train()
            cycle_begin = time.perf_counter()
            last_group, support, plan = None, None, None
            support_audits = []
            iterator = iter(loader.batches(order))
            bar = tqdm(order, desc=f'DEBUG {count} nodes cycle {cycle}/{a.debug_cycles}', unit='update')
            for indices in bar:
                wait_start = time.perf_counter()
                loaded = next(iterator)
                wait = time.perf_counter() - wait_start
                transfer_start = time.perf_counter()
                query = bind_loaded(loaded)
                torch.cuda.synchronize()
                transfer = time.perf_counter() - transfer_start
                if query.indices.tolist() != indices:
                    raise ValueError('Training query tile differs from original complete schedule')
                group = train_rows[indices[0]]['patient_group']
                plan_start = time.perf_counter()
                if group != last_group:
                    support, record_ids, _ = support_binding(memory, train_rows, group)
                    with torch.no_grad():
                        plan = net.fit_support_clusters(*support)
                    support_audits.append(dict(group=group, record_ids=record_ids,
                        fitted_before_step=step + 1, current_L1_at_case_start=True))
                    last_group = group
                torch.cuda.synchronize()
                plan_seconds = time.perf_counter() - plan_start
                step += 1
                update = optimize(net, optimizer, query, support, plan, context, indices,
                                  meta['base'], step)
                update.update(cycle=cycle, record_ids=[train_rows[i]['id'] for i in indices],
                    actual_batch=len(indices), loader_wait_seconds=wait,
                    transfer_seconds=transfer, teacher_preparation_seconds=plan_seconds)
                result['actual_updates'].append(update)
                bar.set_postfix(step=step, loss=round(update['loss'], 4),
                    sec=round(update['compute_update_seconds'], 3),
                    GiB=round(update['peak_allocated_bytes'] / 2**30, 2))
                budget.check()
                del query, loaded
            del iterator, support, plan
            # Production evaluates with refreshed current L0 support, not stale
            # memory. This same memory is reused at the next cycle start.
            memory, refresh_seconds = refresh_memory(net, loader, train_rows, meta, budget)
            evaluation = evaluate_preserving(net, optimizer, train_rows, val_rows, store, memory, budget)
            result['evaluation'].append(dict(cycle=cycle, step=step, **evaluation))
            result['cycles'].append(dict(cycle=cycle, complete_original_tiles=len(order),
                elapsed_including_refresh_evaluation=time.perf_counter() - cycle_begin,
                support_audits=support_audits, refresh_current_memory_seconds=refresh_seconds,
                parameter_delta_from_initial=norms(net, initial)))
            brief = evaluation['validation']['metrics']
            print(f'DEBUG {count} cycle {cycle} | held-out MRR={brief["ranking_mrr"]:.6f} '
                  f'R@5={brief["ranking_recall_at_5"]:.6f} '
                  f'rank_loss={brief["ranking_pairwise_loss"]:.6f}', flush=True)
            # A cycle record is a new immutable result, not a resumable or ready checkpoint.
            new_json(a.output / f'budget{count}_cycle{cycle}.json', result)
        if step != expected_steps:
            raise AssertionError('Complete DEBUG learning schedule did not finish')
        result['final_model_sha256'] = hash_state(net.state_dict())
        result['final_optimizer_sha256'] = hash_state(optimizer.state_dict())
        report['budgets'][str(count)] = result
        del net, optimizer, initial, memory
        torch.cuda.empty_cache()
    report['implementation_sha256'] = implementation
    report['source_preserved'] = source_identity() == before_source
    report['source_checkpoint_preserved'] = sha(checkpoint) == checkpoint_hash
    report['assignment_preserved'] = sha(a.assignment) == assignment_hash
    if not all(report[key] for key in ('source_preserved', 'source_checkpoint_preserved', 'assignment_preserved')):
        raise AssertionError('Original production sources/weights/observations changed')
    if any(sha(ROOT / key) != value for key, value in implementation.items()):
        raise AssertionError('Measured learning implementation changed during DEBUG run')
    report['resources']['RSS'] = psutil.Process().memory_info().rss
    report['crop_reuse'] = store.report()
    report['final_comparison_scope'] = 'Predeclared cycle3, whole original DEBUG train/held-out cases; not optimum admission.'
    new_json(a.output / 'report.json', report)
    print('REPORT:', a.output / 'report.json', flush=True)
    print('Continuous learning DEBUG complete; no production checkpoint/ready or optimal-node declaration.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'assignment', 'assignment-receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--debug-cycles', type=int, required=True)
    for name in ('query-radius-mm', 'near-radius-mm', 'mid-radius-mm', 'cuda-gib', 'rss-gib', 'resident-gib'):
        parser.add_argument('--' + name, type=float, required=True)
    parser.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    previous_rng = rng_state()
    try:
        run(args)
    finally:
        restore_rng(previous_rng)


if __name__ == '__main__':
    main()
