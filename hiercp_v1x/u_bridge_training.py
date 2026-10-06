"""Original-v1 U-bridge training: seven comparisons per source problem.

The model and its two genuine local views are caller-owned. Only the approved
objective and candidate provider change. Production requires CUDA, forty epochs,
measured physical batching, and full 129-candidate joint validation. UNIT helpers
below can be tested with score tensors without constructing a neural model.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F


FORMAT = 'original_v1_u_bridge_training_v1'
ARMS = ('selected', 'native')
EXPECTED_PARAMETERS = 10434532


def cpu_copy(value):
    """Detach AND clone, including CPU tensors: saves never alias live state."""
    if torch.is_tensor(value):
        return value.detach().to('cpu').clone(memory_format=torch.preserve_format)
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def digest(value):
    """Value hash independent of device and tensor storage identity."""
    result = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            result.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            visit(torch.from_numpy(item.copy()))
        elif isinstance(item, dict):
            result.update(b'dict')
            for key in sorted(item, key=str):
                visit(str(key)); visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(type(item).__name__.encode())
            for child in item: visit(child)
        else:
            result.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return result.hexdigest()


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.random.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available(): raise RuntimeError('CUDA RNG cannot resume on CPU')
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(path, payload):
    """Atomically publish an immutable snapshot in the owned experiment only."""
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        torch.save(cpu_copy(payload), stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _append(path, value):
    with Path(path).open('a', encoding='utf8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n'); stream.flush()


def pair_objective(scores, consistency, *, expected_candidates=8):
    """Mean over problems, each with one fixed P and every supplied U once."""
    if not isinstance(scores, (list, tuple)) or not scores:
        raise ValueError('Nonempty original ordered score list required')
    if not torch.is_tensor(consistency) or consistency.numel() != 1:
        raise ValueError('Actual original two-view consistency scalar required')
    terms = []
    for score in scores:
        if score.ndim != 1 or score.numel() != expected_candidates:
            raise ValueError(f'One P plus {expected_candidates - 1} U required per problem')
        terms.append(F.softplus(score[1:].float() - score[0].float()).mean())
    ranking = torch.stack(terms).mean()
    return ranking + .1 * consistency.float(), dict(ranking=ranking, consistency=consistency.float())


def score_row(score, example, candidate_keys, *, expected_candidates):
    """Pessimistic ties preserve original-v1 ranking semantics; no P boost."""
    score = torch.as_tensor(score, dtype=torch.float64).detach().cpu()
    if score.ndim != 1 or score.numel() != expected_candidates or not bool(torch.isfinite(score).all()):
        raise ValueError('Complete finite bound source-problem scores required')
    if len(candidate_keys) != expected_candidates or len(set(map(str, candidate_keys))) != expected_candidates:
        raise ValueError('Complete unique GT-independent candidate identities required')
    if candidate_keys[0] != 'P': raise ValueError('Original source P must occupy fixed output index zero')
    centers = [list(example['positive_center'])]
    for key in candidate_keys[1:]:
        if not isinstance(key, str) or ':' not in key: raise ValueError('Bound source-center key required')
        category, token = key.split(':', 1)
        field = {'S': 'selected_centers', 'U': 'native_centers'}.get(category)
        if field is None or not token.isdecimal(): raise ValueError('Unknown source-center identity')
        index = int(token)
        if index >= len(example[field]): raise ValueError('Source-center key outside frozen candidate inventory')
        centers.append(list(example[field][index]))
    negative = score[1:]
    rank = 1 + int((negative >= score[0]).sum())
    return dict(sample_index=int(example['index']), source_id=example['id'], case_id=example['case_id'],
                source_component=int(example['source_component']),
                positive_center=list(example['positive_center']), candidate_keys=list(candidate_keys),
                candidate_centers=centers, positive_index=0, candidate_difficulties=[0] + [1] * len(negative),
                scores=score.tolist(), first_P_rank=rank, mrr=1. / rank,
                top1=float(rank == 1), pair_loss=float(F.softplus(negative - score[0]).mean()),
                pair_win=float((score[0] > negative).double().mean()),
                pair_tie=float((score[0] == negative).double().mean()),
                margin=float(score[0] - negative.max()), candidate_count=expected_candidates)


def aggregate_rows(rows):
    """Patient macro: average source problems within a patient, then patients."""
    if not rows: raise ValueError('No actual prediction rows to aggregate')
    if len({row['source_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate source problem in evaluation')
    by_case = defaultdict(list)
    for row in rows: by_case[row['case_id']].append(row)
    keys = ('mrr', 'top1', 'pair_loss', 'pair_win', 'pair_tie', 'margin')
    cases = [dict(case_id=case, source_problems=len(values),
                  **{key: sum(v[key] for v in values) / len(values) for key in keys})
             for case, values in sorted(by_case.items())]
    metrics = {key: sum(row[key] for row in cases) / len(cases) for key in keys}
    return dict(metrics=metrics, patients=len(cases), source_problems=len(rows), cases=cases,
                metric_weighting='patient macro of within-patient source-problem means',
                tie_policy='positive rank=1+count(U score >= P score); strict top1 and pair wins',
                scope='observed/source-anchor proxy; not donor-specific CP suitability or clinical quality')


def selection_key(report):
    metrics = report['metrics']
    key = (metrics['mrr'], metrics['top1'], -metrics['pair_loss'])
    if not all(math.isfinite(value) for value in key): raise ValueError('Nonfinite BEST metric')
    return key


def record_comparisons(state, rows, *, optimizer_updated):
    """Coverage is earned only by actual finite optimizer updates."""
    if type(optimizer_updated) is not bool: raise ValueError('Explicit actual-update receipt required')
    if not optimizer_updated: return
    for row in rows:
        key = str(row['sample_index'])
        state['seen_comparisons'][key] = sorted(set(state['seen_comparisons'].get(key, ()))
                                               | set(row['candidate_keys'][1:]))


def permutation(indices, generator):
    if not indices or len(set(indices)) != len(indices): raise ValueError('Complete unique problem indices required')
    # Preserve original V1 RandomSampler order AND its complete RNG consumption.
    sampler = torch.utils.data.RandomSampler(range(len(indices)), generator=generator)
    return [indices[i] for i in sampler]


def validate_cursor(state, indices, physical_batch):
    order = state['order']
    if order is None:
        if state['position'] != 0: raise ValueError('Missing permutation at nonzero cursor')
    elif sorted(order) != sorted(indices) or len(order) != len(indices):
        raise ValueError('Checkpoint permutation changed source-problem coverage')
    position = state['position']
    if type(position) is not int or not 0 <= position <= len(indices): raise ValueError('Invalid source cursor')
    if position != len(indices) and position % physical_batch: raise ValueError('Cursor splits an optimizer batch')


def restore_optimizer_history(optimizer, updates):
    """Restore PyTorch's transient scheduler receipt from verified AdamW steps.

    Optimizer state_dict omits _opt_called. This flag records only whether a
    genuine optimizer step already happened; restoring it changes no tensor,
    learning rate, scheduler counter, or optimization calculation.
    """
    if type(updates) is not int or updates < 0: raise ValueError('Invalid saved optimizer update count')
    parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError('Duplicate parameter in resumed optimizer')
    if any(id(parameter) not in {id(p) for p in parameters} for parameter in optimizer.state):
        raise ValueError('Saved optimizer state contains an unknown parameter')
    steps = []; zero_moments = []
    for parameter in parameters:
        entry = optimizer.state.get(parameter)
        if not entry:
            if updates:
                raise ValueError('Recorded genuine update lacks AdamW state for every parameter')
            continue
        if any(key not in entry for key in ('step', 'exp_avg', 'exp_avg_sq')):
            raise ValueError('Incomplete saved AdamW state')
        step = torch.as_tensor(entry['step']).detach()
        if step.numel() != 1: raise ValueError('Saved AdamW step must be a scalar')
        if (not torch.is_tensor(entry['exp_avg']) or not torch.is_tensor(entry['exp_avg_sq'])
                or entry['exp_avg'].shape != parameter.shape or entry['exp_avg_sq'].shape != parameter.shape):
            raise ValueError('Saved AdamW moment shape changed')
        steps.append(step.reshape(()).to(dtype=torch.float64))
        if not updates:
            # Fused AdamW initializes moments before its inf-aware kernel skips.
            zero_moments.append((entry['exp_avg'] == 0).all() & (entry['exp_avg_sq'] == 0).all())
    if not parameters: raise ValueError('No actual optimizer parameters')
    if steps:
        counts = torch.stack(steps).cpu()
        if not bool(torch.isfinite(counts).all()) or not bool((counts == updates).all()):
            raise ValueError('AdamW parameter steps disagree with recorded genuine update count')
    if zero_moments and not bool(torch.stack(zero_moments).all()):
        raise ValueError('Zero-update AdamW state has nonzero or nonfinite moments')
    optimizer._opt_called = bool(updates)
    return dict(verified_genuine_updates=updates, verified_parameter_tensors=len(parameters),
                verified_initialized_parameter_tensors=len(steps),
                transient_optimizer_called=bool(updates), optimizer_math_changed=False)


def validate_resume_progress(state, train_ids, val_ids, physical_batch, epochs, scheduler):
    """Reject a cursor that would repeat work or advance the epoch schedule."""
    validate_cursor(state, train_ids, physical_batch)
    phase = state['phase']; epoch = state['epoch']
    if phase not in ('initial_validation', 'training', 'validation', 'complete'):
        raise ValueError('Unknown saved training phase')
    if type(epoch) is not int or not 1 <= epoch <= epochs + 1:
        raise ValueError('Saved epoch outside the complete experiment')
    completed = epoch - 1
    if phase == 'complete':
        if epoch != epochs + 1 or state['position'] or state['order'] is not None:
            raise ValueError('Completed checkpoint contains an unfinished source cursor')
    elif epoch > epochs:
        raise ValueError('Unfinished phase cannot exceed the requested epochs')
    if phase == 'initial_validation' and (epoch != 1 or state['position'] or state['updates']):
        raise ValueError('Initial validation cannot contain optimizer work')
    if scheduler.last_epoch != completed or scheduler.T_max != 40:
        raise ValueError('Saved cosine schedule disagrees with completed epochs')
    if [row['epoch'] for row in state['history']] != list(range(1, completed + 1)):
        raise ValueError('Saved epoch history would repeat or skip an epoch')
    per_epoch = math.ceil(len(train_ids) / physical_batch)
    current_updates = math.ceil(state['position'] / physical_batch)
    expected_updates = completed * per_epoch + current_updates
    if state['updates'] != expected_updates:
        raise ValueError('Genuine update count disagrees with exact source cursor')
    if state['attempts'] != state['updates'] + state['overflows']:
        raise ValueError('Saved backward attempts disagree with actual updates and AMP skips')
    if len(state['train_rows']) != state['position']:
        raise ValueError('Saved live train metrics disagree with the updated source cursor')
    if state['order'] is not None:
        if [row['sample_index'] for row in state['train_rows']] != state['order'][:state['position']]:
            raise ValueError('Saved train metrics lost exact source update order')
    if phase == 'validation' and state['position'] != len(train_ids):
        raise ValueError('Epoch validation started before all source problems were updated')
    val_position = state['validation_position']
    if type(val_position) is not int or not 0 <= val_position <= len(val_ids):
        raise ValueError('Invalid saved full-validation cursor')
    if val_position != len(val_ids) and val_position % physical_batch:
        raise ValueError('Saved validation cursor splits a physical batch')
    if [row['sample_index'] for row in state['validation_rows']] != val_ids[:val_position]:
        raise ValueError('Saved full-validation metrics would repeat or skip sources')
    if phase in ('training', 'complete') and val_position:
        raise ValueError('Training/completion contains unfinished validation work')


def _groups(net):
    required = ('local_encoder', 'patient_encoder', 'prototype_encoder', 'score_head')
    if any(not hasattr(net, name) for name in required): raise ValueError('Actual complete original v1 model required')
    local = net.local_encoder
    if not hasattr(local, 'dense_encoder'): raise ValueError('Original CNN path required')
    return dict(CNN=local.dense_encoder, L0=local, L1=net.patient_encoder,
                L2=net.prototype_encoder, scalar=net.score_head)


def gradient_receipt(net, groups):
    parameters = [(name, value) for name, value in net.named_parameters() if value.requires_grad]
    missing = [name for name, value in parameters if value.grad is None]
    gradients = [value.grad for _, value in parameters if value.grad is not None]
    if not gradients: raise RuntimeError('No actual trainable gradients')
    finite = torch.stack([torch.isfinite(value).all() for value in gradients]).all()
    norms = []
    for name, module in groups.items():
        values = [p.grad.detach().double().square().sum() for p in module.parameters()
                  if p.requires_grad and p.grad is not None]
        if not values: raise RuntimeError('No gradient reached ' + name)
        norms.append(torch.stack(values).sum().sqrt())
    packed = torch.stack([finite.double(), *norms]).cpu().tolist()
    return dict(finite=bool(packed[0]), missing=missing,
                trainable_parameter_tensors=len(parameters), gradient_present=len(parameters) - len(missing),
                norms={name: value if math.isfinite(value) else str(value)
                       for name, value in zip(groups, packed[1:])})


def _probe_before(groups):
    probes = {}
    for name, module in groups.items():
        values = [(key, value) for key, value in module.named_parameters() if value.requires_grad]
        if name == 'L0': values = [(key, value) for key, value in values if not key.startswith('dense_encoder.')]
        if not values: raise ValueError('No trainable parameter in ' + name)
        key, value = values[0]
        probes[name] = (key, value, value.detach().flatten()[:32].clone())
    return probes


def _probe_after(probes):
    changes = torch.stack([(value.detach().flatten()[:32] - before).abs().max()
                           for _, value, before in probes.values()]).cpu().tolist()
    return dict(parameters={name: entry[0] for name, entry in probes.items()},
                max_abs_delta=dict(zip(probes, changes)),
                scope='32 deterministic elements of one actual parameter per module; sampled update probe')


def _check_budget(budget):
    if callable(getattr(budget, 'check', None)): budget.check(); return
    import psutil
    if torch.cuda.memory_allocated() > budget.cuda_bytes or psutil.Process().memory_info().rss > budget.rss_bytes:
        raise MemoryError('Explicit CUDA/RSS budget exceeded; no model/data/batch fallback')


def retry_amp_overflow(scaler, optimizer):
    """GradScaler owns its recorded inf test and skips this optimizer step."""
    if not scaler.is_enabled(): raise FloatingPointError('Nonfinite gradients with AMP disabled')
    previous = scaler.get_scale(); scaler.step(optimizer); scaler.update()
    following = scaler.get_scale()
    if not math.isfinite(following) or following <= 0 or following >= previous:
        raise RuntimeError('AMP overflow did not produce a finite positive lower scale')
    return previous, following


def batch_workload(batch):
    """Read CPU graph inventories before transfer; report actual nodes/edges."""
    result = dict(source_patch_shape=list(batch.source_patches.shape),
                  target_patch_shape=list(batch.target_patches.shape), counts=list(batch.counts))
    for name in ('local_batch', 'local_batch_view2', 'patient_batch', 'prototype_batch'):
        graph = getattr(batch, name)
        if graph is None: raise ValueError('Missing required original graph view: ' + name)
        graphs = int(graph.num_graphs)
        node_counts = torch.zeros(graphs, dtype=torch.long)
        edge_counts = torch.zeros(graphs, dtype=torch.long)
        types = {}
        for kind in graph.node_types:
            store = graph[kind]
            counts = store.ptr[1:] - store.ptr[:-1]
            if counts.is_cuda or counts.numel() != graphs: raise ValueError('CPU batched node partitions required')
            node_counts += counts
            types[str(kind)] = int(counts.sum())
        relations = {}
        for relation in graph.edge_types:
            index = graph[relation].edge_index
            if index.is_cuda: raise ValueError('CPU input edge inventory required')
            assignments = graph[relation[0]].batch[index[0]]
            edge_counts += torch.bincount(assignments, minlength=graphs)
            relations[str(relation)] = int(index.shape[1])
        def stats(values):
            return dict(min=int(values.min()), max=int(values.max()),
                        mean=float(values.double().mean()), total=int(values.sum()))
        result[name] = dict(graphs=graphs, nodes=stats(node_counts), edges=stats(edge_counts),
                            node_types=types, edge_types=relations)
    return result


def _examples(provider, partition):
    examples = list(provider.examples(partition))
    if not examples or len({row['index'] for row in examples}) != len(examples):
        raise ValueError('Complete unique original source examples required: ' + partition)
    if len({row['id'] for row in examples}) != len(examples): raise ValueError('Duplicate source identities')
    return examples


def _model_contract(net, debug):
    if getattr(net, 'ablation_mode', None) != 'full' or getattr(net, 'hidden_dim', None) != 128:
        raise ValueError('Unchanged original-v1 full128D model required')
    groups = _groups(net)
    if (len(net.local_encoder.blocks) != 3 or len(net.patient_encoder.blocks) != 2
            or len(net.prototype_encoder.blocks) != 2):
        raise ValueError('Unchanged original3/2/2 graph depth required')
    count = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if count != EXPECTED_PARAMETERS or trainable != EXPECTED_PARAMETERS:
        raise ValueError(f'Original {EXPECTED_PARAMETERS} parameters required, actual {count}/{trainable}')
    if any(getattr(net, name, None) not in (None, False) for name in ('half_b', 'half_a_mode', 'v1x_stage')):
        raise ValueError('Architecture replacement is not an original-v1 U bridge')
    return groups


class CalibrationRejected(RuntimeError):
    def __init__(self, reports):
        super().__init__('No explicit physical batch candidate passed actual CUDA calibration')
        self.reports = reports


def calibrate_batches(net, provider, config, *, arm, candidates, workers, budget, debug=False):
    """Actual-shape clone update probes; original model and RNG remain unchanged.

    Each explicit candidate measures three complete updates, excluding the first
    warm-up. Probe coverage is reported and is not advertised as a worst-cohort
    proof. Every explicit candidate gets a receipt, including OOM rejection;
    the caller can select a shared measured batch across independent arms.
    """
    if arm not in ARMS or candidates != sorted(set(candidates)) or not candidates or min(candidates) < 1:
        raise ValueError('Explicit increasing physical sample candidates required')
    if workers < 2 or not torch.cuda.is_available(): raise RuntimeError('Parallel readers and actual CUDA required')
    _model_contract(net, debug)
    examples = _examples(provider, 'train')
    rng = capture_rng(); original = digest(net.state_dict()); reports = []
    training = config['training']
    try:
        for size in candidates:
            clone = batch = optimizer = scaler = output = loss = groups = terms = None
            ids = [row['index'] for row in examples[:size]]
            if size > len(examples):
                reports.append(dict(physical_batch=size, accepted=False, actual_sample_indices=ids,
                    error='Insufficient distinct source problems; candidate not synthesized or measured'))
                continue
            try:
                clone = copy.deepcopy(net).cuda().train(); groups = _groups(clone)
                batch = provider.batch(ids, arm, 1, True, full=False).pin_memory().to('cuda', non_blocking=True)
                if batch.local_batch_view2 is None or list(batch.counts) != [8] * size:
                    raise ValueError('Calibration requires genuine two-view eight-candidate batches')
                optimizer = torch.optim.AdamW(clone.parameters(), lr=training['lr'],
                    weight_decay=training['weight_decay'], fused=training['fused_optimizer'])
                scaler = torch.amp.GradScaler('cuda', enabled=training['amp'])
                times = []; peaks = []; attempts = overflows = 0
                for trial in range(3):
                    torch.cuda.synchronize(); start = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    while True:
                        optimizer.zero_grad(set_to_none=True); attempts += 1
                        with torch.autocast('cuda', enabled=training['amp']):
                            output = clone(batch); loss, terms = pair_objective(output.scores, output.consistency)
                        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite calibration loss')
                        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                        receipt = gradient_receipt(clone, groups)
                        if receipt['finite']: break
                        if not scaler.is_enabled(): raise FloatingPointError('Nonfinite calibration FP32 gradients')
                        retry_amp_overflow(scaler, optimizer)
                        overflows += 1; del output, loss, terms; output = loss = terms = None
                        _check_budget(budget)
                    if receipt['missing']: raise RuntimeError('Disconnected original calibration gradients: ' + repr(receipt['missing']))
                    torch.nn.utils.clip_grad_norm_(clone.parameters(), training['grad_clip'], error_if_nonfinite=True)
                    scaler.step(optimizer); scaler.update(); torch.cuda.synchronize(); _check_budget(budget)
                    peaks.append(torch.cuda.max_memory_allocated())
                    if trial: times.append(time.perf_counter() - start)
                    del output, loss, terms; output = loss = terms = None
                reports.append(dict(physical_batch=size, accepted=True, actual_sample_indices=ids,
                    candidate_graphs_per_update=size * 8, sampled_graph_views_per_update=size * 16,
                    samples_per_second=size / (sum(times) / len(times)), peak_cuda_bytes=max(peaks),
                    measured_updates=3, timed_updates=2, backward_attempts=attempts, AMP_overflows=overflows,
                    scope='representative ordered actual samples; not entire-cohort worst-case'))
            except (torch.cuda.OutOfMemoryError, MemoryError) as error:
                reports.append(dict(physical_batch=size, accepted=False, actual_sample_indices=ids,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    error=f'{type(error).__name__}: {error}'))
                print(f'{arm} measured physical batch {size} rejected: {type(error).__name__}: {error}', flush=True)
            finally:
                del clone, batch, optimizer, scaler, output, loss, groups, terms
                torch.cuda.empty_cache()
    finally:
        restore_rng(rng)
        if digest(net.state_dict()) != original: raise RuntimeError('Calibration mutated original model')
    accepted = [row for row in reports if row['accepted']]
    if not accepted: raise CalibrationRejected(reports)
    selected = max(accepted, key=lambda row: row['samples_per_second'])['physical_batch']
    return selected, dict(selected_physical_batch=selected, arm=arm, reports=reports,
                         initial_state_sha256=original, original_model_and_RNG_preserved=True)


@contextmanager
def _pause_signal():
    flag = {'requested': False}
    previous = signal.getsignal(signal.SIGINT)
    def request(signum, frame):
        flag['requested'] = True
        print('Pause requested; finishing this batch and saving its exact cursor.', flush=True)
    signal.signal(signal.SIGINT, request)
    try: yield flag
    finally: signal.signal(signal.SIGINT, previous)


def _prefetch(provider, batches, arm, epoch):
    # Provider uses source/epoch-local sampling generators, not global model RNG.
    def obtain(indices):
        return provider.batch(indices, arm, epoch, True, full=False).pin_memory()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='u-bridge-cpu') as pool:
        iterator = iter(batches); ids = next(iterator, None)
        if ids is None: return
        future = pool.submit(obtain, ids)
        for following in iterator:
            batch = future.result(); future = pool.submit(obtain, following)
            yield batch
        yield future.result()


def run_arm(net, provider, config, *, arm, output, physical_batch, workers, epochs, identity, budget, debug=False):
    """Run/resume one independent arm; caller supplies a freshly initialized net."""
    if arm not in ARMS or type(debug) is not bool or type(epochs) is not int or epochs < 1:
        raise ValueError('Explicit arm/DEBUG/epoch contract required')
    if not debug and epochs != 40: raise ValueError('Production U bridge preserves forty epochs')
    if type(physical_batch) is not int or physical_batch < 1 or workers < 2:
        raise ValueError('Measured physical sample batch and parallel workers required')
    if not torch.cuda.is_available(): raise RuntimeError('Actual CUDA required; no CPU training fallback')
    import psutil
    from tqdm import tqdm
    runtime = copy.deepcopy(config['u_bridge_runtime']); training = config['training']
    pause_update = runtime.get('debug_pause_after_updates')
    if pause_update is not None and (not debug or type(pause_update) is not int or pause_update < 1):
        raise ValueError('Explicit pause-after-update diagnostic is allowed only in DEBUG')
    if training['consistency_weight'] != .1 or training['gradient_accumulation_steps'] != 1:
        raise ValueError('Original consistency0.1 and explicit accumulation1 required')
    calibration = runtime['batch_calibration']
    if (calibration['arm'] != arm or calibration['selected_physical_batch'] != physical_batch
            or not calibration['original_model_and_RNG_preserved']
            or not any(row['accepted'] and row['physical_batch'] == physical_batch for row in calibration['reports'])):
        raise ValueError('Physical batch requires this arm\'s actual accepted CUDA measurement')
    chunk = runtime['validation_local_chunk_size']
    if type(chunk) is not int or chunk < 1: raise ValueError('Explicit whole-case inference execution chunk required')
    groups = _model_contract(net, debug)
    examples = _examples(provider, 'train'); validation = _examples(provider, 'val')
    train_ids = [row['index'] for row in examples]; val_ids = [row['index'] for row in validation]
    train_cases = {row['case_id'] for row in examples}; val_cases = {row['case_id'] for row in validation}
    if train_cases & val_cases or set(train_ids) & set(val_ids): raise ValueError('Source train/validation leakage')
    if physical_batch > len(examples): raise ValueError('Physical batch cannot duplicate examples')
    lookup = {row['index']: row for row in examples + validation}
    device = torch.device('cuda', torch.cuda.current_device()); net.to(device)
    initial_hash = digest(net.state_dict())
    if calibration['initial_state_sha256'] != initial_hash: raise ValueError('Calibration belongs to other initial weights')
    expected_initial = identity.get('initial_state_sha256', initial_hash)
    if expected_initial != initial_hash: raise ValueError('Both arms require identical declared fresh seed42 weights')
    if getattr(provider, 'global_rng_free', False) is not True:
        raise ValueError('Provider must certify per-source sampling independent of global model RNG before prefetch')
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=training['lr'], weight_decay=training['weight_decay'], fused=training['fused_optimizer'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    scaler = torch.amp.GradScaler('cuda', enabled=training['amp'])
    generator = torch.Generator().manual_seed(42 + 2003)
    bound_config = copy.deepcopy(config)
    for field in ('resume_checkpoint', 'pause_file', 'debug_pause_after_updates'):
        bound_config['u_bridge_runtime'].pop(field, None)
    binding = dict(format=FORMAT, identity=identity, config=bound_config, arm=arm, debug=debug,
                   epochs=epochs, physical_batch=physical_batch, workers=workers,
                   train_examples=examples, val_examples=validation, initial_state_sha256=initial_hash)
    binding_hash = digest(binding)
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=True)
    ownership = root / 'training_identity.json'
    if ownership.exists():
        if json.loads(ownership.read_text(encoding='utf8'))['identity_sha256'] != binding_hash:
            raise ValueError('Existing training output belongs to another arm/config/source')
        if not runtime.get('resume_checkpoint'): raise FileExistsError('Existing owned output requires explicit resume')
    else:
        if any(root.iterdir()): raise FileExistsError('Existing files preserved; choose a new arm training directory')
        _write_new(ownership, dict(identity_sha256=binding_hash, binding=cpu_copy(binding)))
    state = dict(epoch=1, phase='initial_validation', position=0, order=None, updates=0, attempts=0, overflows=0,
                 initial_validation=None, history=[], best=None, train_rows=[], validation_rows=[], validation_position=0,
                 connected=[], invocation_status='RUNNING', validation_active_seconds=0., epoch_active_seconds=0.,
                 seen_comparisons={str(index): [] for index in train_ids})
    resume_receipt = None
    if runtime.get('resume_checkpoint'):
        saved = torch.load(runtime['resume_checkpoint'], map_location='cpu', weights_only=False)
        if saved.get('format') != FORMAT or saved.get('identity_sha256') != binding_hash:
            raise ValueError('Resume requires exact arm/model/source/data/execution identity')
        checksum = saved.pop('content_sha256', None)
        if checksum != digest(saved): raise ValueError('Checkpoint content identity changed')
        net.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        state = saved['state']; generator.set_state(saved['shuffle_generator']); restore_rng(saved['rng'])
        validate_resume_progress(state, train_ids, val_ids, physical_batch, epochs, scheduler)
        resume_receipt = restore_optimizer_history(optimizer, state['updates'])
        del saved
    invocation_start = dict(phase=state['phase'], updates=state['updates'], attempts=state['attempts'],
                            completed_epochs=len(state['history']))
    completed_baseline = None
    if resume_receipt is not None and state['phase'] == 'complete':
        completed_baseline = dict(model=digest(net.state_dict()), optimizer=digest(optimizer.state_dict()),
                                  scheduler=digest(scheduler.state_dict()), rng=digest(capture_rng()))
    torch.set_num_threads(workers)
    process = psutil.Process(); process.cpu_percent()
    receipt_path = root / 'execution_contract.json'
    if not receipt_path.exists():
        _write_new(receipt_path, dict(format=FORMAT, arm=arm, debug=debug, epochs=epochs,
            parameters=EXPECTED_PARAMETERS, trainable_parameters=EXPECTED_PARAMETERS,
            layers=dict(L0=3, L1=2, L2=2), hidden=128, heads=4, margin_mm=10,
            training_samples=len(examples), validation_samples=len(validation), training_cases=len(train_cases),
            validation_cases=len(val_cases), physical_sample_batch=physical_batch,
            physical_candidate_rows=physical_batch * 8, physical_graph_views=physical_batch * 16,
            accumulation=1, effective_sample_batch=physical_batch, workers=workers, prefetch_batches=1,
            updates_per_epoch=math.ceil(len(examples) / physical_batch), total_planned_updates=epochs * math.ceil(len(examples) / physical_batch),
            train_candidates_per_problem=8, validation_candidates_per_problem=129,
            query_positive='original source anchor only', original_geometry_corruption=False,
            loss='per-problem mean softplus(U-P), mean over problems, plus original two-view consistency0.1',
            precision='CUDA AMP' if training['amp'] else 'CUDA FP32', scheduler='cosine Tmax40',
            gpu=torch.cuda.get_device_name(), gpu_count=torch.cuda.device_count(),
            cpu_logical=psutil.cpu_count(), RAM=psutil.virtual_memory()._asdict(),
            calibration=calibration, full_training=False, quality_verified=False, CP_quality_verified=False))
    def checkpoint(status='RUNNING', best=False):
        began = time.perf_counter(); state['invocation_status'] = status
        payload = dict(format=FORMAT, identity_sha256=binding_hash, model=cpu_copy(net.state_dict()),
            optimizer=cpu_copy(optimizer.state_dict()), scheduler=cpu_copy(scheduler.state_dict()),
            scaler=cpu_copy(scaler.state_dict()), state=cpu_copy(state), rng=cpu_copy(capture_rng()),
            shuffle_generator=generator.get_state().clone())
        payload['content_sha256'] = digest(payload)
        atomic_save(root / 'checkpoint_latest.pt', payload)
        if best: atomic_save(root / 'checkpoint_best.pt', payload)
        return time.perf_counter() - began
    def pause_requested(flag):
        return (flag['requested'] or Path(runtime.get('pause_file', root / 'STOP_AFTER_BATCH')).exists()
                or (pause_update is not None and state['updates'] >= pause_update))
    def evaluate_phase(flag):
        phase = state['phase']; label_epoch = 0 if phase == 'initial_validation' else state['epoch']
        started = time.perf_counter(); net.eval()
        for start in range(state['validation_position'], len(val_ids), physical_batch):
            if pause_requested(flag): return False
            ids = val_ids[start:start + physical_batch]
            before_rng = capture_rng(); loading = time.perf_counter()
            try:
                batch = provider.batch(ids, arm, training['fixed_validation_epoch'], False, full=True)
                if list(batch.counts) != [129] * len(ids) or batch.local_batch_view2 is None:
                    raise ValueError('Full validation requires P+128U and two actual original views')
                if tuple(getattr(batch, 'bridge_indices', ())) != tuple(ids): raise ValueError('Validation source binding changed')
                load_seconds = time.perf_counter() - loading; torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); forward = time.perf_counter()
                with torch.no_grad(), torch.autocast('cuda', enabled=training['amp']):
                    scores = net.score_inference_chunked(batch, local_chunk_size=chunk)
                torch.cuda.synchronize(); forward_seconds = time.perf_counter() - forward
                if len(scores) != len(ids): raise ValueError('Full joint upper lost source problem mapping')
                rows = [score_row(score, lookup[index], provider.candidate_keys(index, arm,
                    training['fixed_validation_epoch'], full=True), expected_candidates=129)
                    for index, score in zip(ids, scores)]
                state['validation_rows'].extend(rows); state['validation_position'] += len(ids)
                _check_budget(budget)
                _append(root / 'validation_timing.jsonl', dict(epoch=label_epoch, source_indices=ids,
                    loader_seconds=load_seconds, full_joint_forward_seconds=forward_seconds,
                    source_problems=len(ids), candidate_rows=129 * len(ids), local_chunk_size=chunk,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(), rss_bytes=psutil.Process().memory_info().rss))
                del scores, batch
            finally: restore_rng(before_rng)
            state['validation_active_seconds'] += time.perf_counter() - loading
            checkpoint_seconds = checkpoint()
            state['validation_active_seconds'] += checkpoint_seconds
        rows = state['validation_rows']
        if {row['sample_index'] for row in rows} != set(val_ids) or len(rows) != len(val_ids):
            raise ValueError('Full129 validation coverage changed')
        report = aggregate_rows(rows)
        report.update(epoch=label_epoch, update=state['updates'], candidate_universe='fixed original P+128 native U',
                      rows=copy.deepcopy(rows), wall_seconds=state['validation_active_seconds'],
                      wall_scope='active full-validation time across resumed invocation segments', debug=debug,
                      full_candidate_evaluation=True, joint_upper_once_per_source_problem=True)
        path = root / f'validation_epoch_{label_epoch:03d}.json'
        if not path.exists(): _write_new(path, report)
        else:
            stored = json.loads(path.read_text(encoding='utf8'))
            if digest(stored['rows']) != digest(report['rows']): raise ValueError('Existing validation rows disagree with resumed result')
        state['validation_rows'] = []; state['validation_position'] = 0; state['validation_active_seconds'] = 0.
        if phase == 'initial_validation':
            state['initial_validation'] = report; state['phase'] = 'training'; checkpoint()
        else:
            train_report = aggregate_rows(state['train_rows'])
            row = dict(epoch=state['epoch'], update=state['updates'], train7_live_before_updates=train_report,
                       validation129=report, optimization_loader_save_seconds=state['epoch_active_seconds'],
                       validation_seconds=report['wall_seconds'],
                       epoch_wall_seconds=state['epoch_active_seconds'] + report['wall_seconds'],
                       wall_scope='active optimization, loading, checkpointing, and full129 validation; excludes pauses')
            improved = state['best'] is None or selection_key(report) > tuple(state['best']['selection_key'])
            if improved: state['best'] = dict(epoch=state['epoch'], update=state['updates'], selection_key=list(selection_key(report)))
            state['history'].append(row); _append(root / 'curve.jsonl', row)
            scheduler.step(); state.update(epoch=state['epoch'] + 1, phase='training', position=0,
                order=None, train_rows=[], epoch_active_seconds=0.)
            if state['epoch'] > epochs: state['phase'] = 'complete'
            checkpoint(best=improved)
        metrics = report['metrics']
        tqdm.write(f'{arm} validation129 epoch{label_epoch} | MRR={metrics["mrr"]:.6f} top1={metrics["top1"]:.6f} '
                   f'pair-win={metrics["pair_win"]:.6f} pair-loss={metrics["pair_loss"]:.6f} margin={metrics["margin"]:.6f}')
        return True
    started = time.perf_counter(); status = 'RUNNING'
    with _pause_signal() as flag:
        checkpoint()
        try:
            while state['phase'] != 'complete':
                _check_budget(budget)
                if pause_requested(flag): status = 'PAUSED'; break
                if state['phase'] in ('initial_validation', 'validation'):
                    if not evaluate_phase(flag): status = 'PAUSED'; break
                    continue
                net.train()
                if state['order'] is None: state['order'] = permutation(train_ids, generator); checkpoint()
                order = state['order']; validate_cursor(state, train_ids, physical_batch)
                remaining = [order[i:i + physical_batch] for i in range(state['position'], len(order), physical_batch)]
                previous = time.perf_counter()
                bar = tqdm(_prefetch(provider, remaining, arm, state['epoch']), total=math.ceil(len(order) / physical_batch),
                           initial=state['position'] // physical_batch, desc=f'{arm} epoch{state["epoch"]}/{epochs}')
                for batch in bar:
                    loading_seconds = time.perf_counter() - previous; step_start = time.perf_counter()
                    ids = order[state['position']:state['position'] + physical_batch]
                    if list(batch.counts) != [8] * len(ids) or batch.local_batch_view2 is None:
                        raise ValueError('Actual original8/two-view training batch required')
                    if tuple(getattr(batch, 'bridge_indices', ())) != tuple(ids): raise ValueError('Batch source binding changed')
                    workload = batch_workload(batch); _check_budget(budget)
                    while True:
                        attempt_start = time.perf_counter()
                        optimizer.zero_grad(set_to_none=True); state['attempts'] += 1
                        torch.cuda.reset_peak_memory_stats(); events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
                        events[0].record(); batch.to(device, non_blocking=True); events[1].record()
                        with torch.autocast('cuda', enabled=training['amp']):
                            result = net(batch)
                            if len(result.scores) != len(ids): raise ValueError('Training forward lost source problem mapping')
                            loss, terms = pair_objective(result.scores, result.consistency)
                        events[2].record()
                        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite actual U-bridge loss')
                        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                        events[3].record(); gradients = gradient_receipt(net, groups)
                        if gradients['finite']: break
                        scale_before, scale_after = retry_amp_overflow(scaler, optimizer)
                        events[4].record(); events[4].synchronize(); state['overflows'] += 1
                        del result, loss, terms
                        _check_budget(budget)
                        checkpoint_seconds = checkpoint()
                        _append(root / 'update_timing.jsonl', dict(status='AMP_OVERFLOW_SKIPPED_RETRY_SAME_INPUT',
                            epoch=state['epoch'], position=state['position'], attempt=state['attempts'],
                            optimizer_updates=state['updates'], AMP_overflows=state['overflows'], sample_indices=ids,
                            gradient=gradients, scale_before=scale_before, scale_after=scale_after,
                            checkpoint_seconds=checkpoint_seconds, retry_seconds=time.perf_counter() - attempt_start))
                        tqdm.write(f'{arm} AMP overflow: same sources {ids}, cursor {state["position"]}, '
                                   f'no optimizer update, scale {scale_before:g}->{scale_after:g}; retrying')
                        if pause_requested(flag): status = 'PAUSED'; break
                    if status == 'PAUSED':
                        state['epoch_active_seconds'] += loading_seconds + time.perf_counter() - step_start
                        del batch; break
                    if gradients['missing']: raise RuntimeError('Disconnected original trainable parameters: ' + repr(gradients['missing']))
                    clipped = torch.nn.utils.clip_grad_norm_(params, training['grad_clip'], error_if_nonfinite=True)
                    probes = _probe_before(groups); scaler.step(optimizer); scaler.update()
                    events[4].record(); events[4].synchronize()
                    changed = _probe_after(probes)
                    rows = [score_row(score, lookup[index], provider.candidate_keys(index, arm, state['epoch'], full=False),
                                      expected_candidates=8) for index, score in zip(ids, result.scores)]
                    if len(rows) != len(ids): raise ValueError('Training forward lost source problem mapping')
                    state['train_rows'].extend(rows); state['position'] += len(ids); state['updates'] += 1
                    record_comparisons(state, rows, optimizer_updated=True)
                    state['connected'] = sorted(set(state['connected']) | {n for n, p in net.named_parameters() if p.requires_grad and p.grad is not None})
                    values = dict(loss=float(loss.detach()), ranking=float(terms['ranking'].detach()),
                                  consistency=float(terms['consistency'].detach()), gradient_before_clip=float(clipped))
                    del result, loss, terms, probes, batch
                    checkpoint_seconds = checkpoint(); elapsed = time.perf_counter() - step_start
                    state['epoch_active_seconds'] = state.get('epoch_active_seconds', 0.) + loading_seconds + elapsed
                    _check_budget(budget)
                    row = dict(status='OPTIMIZER_UPDATED', epoch=state['epoch'], update=state['updates'], attempt=state['attempts'],
                        sample_indices=ids, physical_samples=len(ids), candidate_rows=8 * len(ids), graph_views=16 * len(ids),
                        candidate_keys={str(row['sample_index']): row['candidate_keys'] for row in rows},
                        actual_input=workload, AMP_overflows=state['overflows'],
                        **values, gradient=gradients, sampled_parameter_changes=changed,
                        margin=sum(r['margin'] for r in rows) / len(rows), train7_pair_win=sum(r['pair_win'] for r in rows) / len(rows),
                        loader_wait_seconds=loading_seconds, transfer_seconds=events[0].elapsed_time(events[1]) / 1000,
                        forward_seconds=events[1].elapsed_time(events[2]) / 1000,
                        backward_seconds=events[2].elapsed_time(events[3]) / 1000,
                        check_clip_optimizer_seconds=events[3].elapsed_time(events[4]) / 1000,
                        checkpoint_seconds=checkpoint_seconds, step_seconds=elapsed,
                        samples_per_second=len(ids) / elapsed, candidate_graphs_per_second=8 * len(ids) / elapsed,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        rss_bytes=process.memory_info().rss, cpu_percent=process.cpu_percent(), learning_rate=optimizer.param_groups[0]['lr'])
                    _append(root / 'update_timing.jsonl', row)
                    bar.set_postfix(loss=round(values['loss'], 5), rank=round(values['ranking'], 5),
                        margin=round(row['margin'], 5), grad=round(values['gradient_before_clip'], 3),
                        probe=sum(v > 0 for v in changed['max_abs_delta'].values()), sec=round(elapsed, 2),
                        GiB=round(row['peak_cuda_bytes'] / 2**30, 3))
                    if pause_requested(flag): status = 'PAUSED'; break
                    previous = time.perf_counter()
                if status == 'PAUSED': break
                if state['position'] != len(order): raise ValueError('Incomplete source-problem epoch')
                if {row['sample_index'] for row in state['train_rows']} != set(train_ids): raise ValueError('Training coverage changed')
                state['phase'] = 'validation'; checkpoint()
        except Exception as error:
            _append(root / 'failures.jsonl', dict(epoch=state['epoch'], phase=state['phase'], position=state['position'],
                updates=state['updates'], error=f'{type(error).__name__}: {error}',
                recovery='Latest owned successful/AMP-skip checkpoint preserved; exact resume retries unchanged cursor'))
            raise
        if state['phase'] == 'complete': status = 'DEBUG_COMPLETE' if debug else 'COMPLETE'
        checkpoint(status)
    report = dict(format=FORMAT, arm=arm, status=status, debug=debug, actual_CUDA=True,
        completed_epochs=state['epoch'] - 1, updates=state['updates'], backward_attempts=state['attempts'], AMP_overflows=state['overflows'],
        initial_state_sha256=initial_hash, identity_sha256=binding_hash, best=state['best'],
        model_sha256=digest(net.state_dict()), optimizer_sha256=digest(optimizer.state_dict()),
        connected_parameter_tensors=len(state['connected']), expected_parameter_tensors=len(list(net.named_parameters())),
        full_training=status == 'COMPLETE', full_evaluation=status == 'COMPLETE',
        evaluation_scope='complete configured held-out source problems with joint P+128U every completed epoch',
        final_phase=state['phase'], next_source_position=state['position'], epochs_requested=epochs,
        quality_verified=False, CP_quality_verified=False, nnunet_training=False,
        invocation_seconds=time.perf_counter() - started, checkpoint=str(root / 'checkpoint_latest.pt'))
    report['resume'] = dict(checkpoint=runtime.get('resume_checkpoint'),
        initial_phase=invocation_start['phase'], verified_optimizer_history=resume_receipt,
        invocation_optimizer_updates=state['updates'] - invocation_start['updates'],
        invocation_backward_attempts=state['attempts'] - invocation_start['attempts'],
        invocation_epoch_completions=len(state['history']) - invocation_start['completed_epochs'])
    if completed_baseline is not None:
        current = dict(model=report['model_sha256'], optimizer=report['optimizer_sha256'],
                       scheduler=digest(scheduler.state_dict()), rng=digest(capture_rng()))
        no_repeat = (current == completed_baseline and report['resume']['invocation_optimizer_updates'] == 0
                     and report['resume']['invocation_backward_attempts'] == 0
                     and report['resume']['invocation_epoch_completions'] == 0)
        if not no_repeat: raise RuntimeError('Completed checkpoint resume changed model/optimizer/schedule/RNG or repeated work')
        report['resume']['completed_checkpoint_no_repeat_verified'] = True
    expected_coverage = 7 if arm == 'selected' else 128
    report['trained_comparisons'] = dict(keys_by_source=cpu_copy(state['seen_comparisons']),
        counts_by_source={key: len(values) for key, values in state['seen_comparisons'].items()},
        expected_unique_U_per_source=expected_coverage,
        complete_expected_coverage=all(len(values) == expected_coverage for values in state['seen_comparisons'].values()),
        scope='unique identities participating in finite optimizer updates; AMP-overflow attempts excluded')
    path = root / ('paused.json' if status == 'PAUSED' else 'training_complete.json')
    if not path.exists(): _write_new(path, report)
    return report
