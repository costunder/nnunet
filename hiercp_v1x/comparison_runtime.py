"""Opt-in execution overlay for the byte-sealed comparison training engine.

Input scheduling, progress and measured activation execution differ. The binder supplies
frozen engine helpers and the existing arm-specific objective/checkpoint policy.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import closing as _closing
import copy
import json
import math
from pathlib import Path
import time
import sys

import torch

from .u_bridge_training import (
    ARMS, FORMAT, EXPECTED_PARAMETERS, _model_contract, _examples, digest,
    capture_rng, restore_rng, restore_optimizer_history, validate_resume_progress,
    _write_new, _append, cpu_copy, atomic_save, aggregate_rows, selection_key,
    _check_budget, score_row, _pause_signal, permutation, validate_cursor,
    batch_workload, pair_objective, gradient_receipt, retry_amp_overflow,
    _probe_before, _probe_after, record_comparisons,
)
from .comparison_progress import PhaseProgress, RunningPatientMetrics
from .comparison_gpu_runtime import ComparisonGpuRuntime


class closing(_closing):
    """Drain staged work while retaining a consumer error as the primary error."""
    def __exit__(self, kind, value, traceback):
        try:
            self.thing.close()
        except Exception as staged_error:
            if value is not None:
                raise value.with_traceback(traceback) from staged_error
            raise


def _prefetch_resources():
    import psutil
    return int(psutil.Process().memory_info().rss), int(psutil.virtual_memory().available)


class _StagingCleanupError(RuntimeError):
    """Retain every worker failure on Python 3.10 as well as newer runtimes."""
    def __init__(self, failures):
        self.failures = tuple(failures)
        super().__init__('CPU staging cleanup failures: ' + '; '.join(
            f'{type(error).__name__}: {error}' for error in self.failures))
        self.__cause__ = self.failures[0]


def _prefetch(provider, batches, arm, epoch, *, training=True, full=False,
              pin_memory=True, with_timing=False, receipt_path=None):
    """Ordered CPU staging with measured admission for independent inputs.

    The first batch measures actual memory. Cold batches may share a bounded
    worker pool only when their provider certifies isolated upper helpers and
    supplies a complete CT workspace estimate. Legacy providers stay serial.
    Memory estimates include observed RSS growth and tensor storage, retaining
    the declared uncached workspace reserve. They are admission estimates,
    not worst-case guarantees; the original hard resource guard still applies.
    """
    if getattr(provider, 'global_rng_free', False) is not True:
        raise ValueError('CPU prefetch requires certified global-RNG-independent provider')
    workers = getattr(provider, 'workers', 0)
    budget = getattr(provider, 'budget', None)
    ready = getattr(provider, 'cached_batch_ready', None)
    staging = getattr(provider, 'cpu_staging', None)
    capable = (type(workers) is int and workers >= 4 and callable(ready)
               and callable(staging) and budget is not None
               and 0 < getattr(provider, 'resident_limit', 0) < getattr(budget, 'rss_bytes', 0))
    ceiling = workers - 2 if capable else 1
    declared_workspace = int(budget.rss_bytes - provider.resident_limit) if capable else 0
    cold_estimate = getattr(provider, 'cold_staging_bytes', None)
    cold_capable = (capable and getattr(provider, 'independent_cold_inputs', False) is True
                    and callable(cold_estimate))
    # Cold inputs now reserve their dense arrays explicitly. Keep the same
    # unestimated headroom as PressureBudget's pressure trigger, instead of
    # reserving that entire workspace twice and forcing a full LRU to serial.
    reserve = declared_workspace // 2 if cold_capable else declared_workspace
    plan = dict(format='comparison_cpu_prefetch_v1', arm=arm, view_epoch=epoch,
                training=training, full129=full, declared_workers=workers,
                certified_cached_concurrency=capable, slots=1,
                certified_cold_concurrency=cold_capable,
                cold_workspace_policy='full CT dense-array inventory plus measured batch storage',
                declared_workspace_reserve_bytes=declared_workspace,
                retained_unestimated_headroom_bytes=reserve,
                admission_scope='measured estimate; original hard budget remains authoritative',
                first_batch_peak_rss_bytes=None, first_batch_rss_growth_bytes=None,
                largest_batch_storage_bytes=0, slot_estimate_bytes=0)
    counts = dict(submitted=0, yielded=0, cold_batches=0, warm_batches=0,
                  peak_queued_batches=0, loader_wait_seconds=0.)

    def receipt(event, **values):
        if receipt_path is not None:
            _append(Path(receipt_path), dict(plan, event=event, observed_at=time.time(), **values))

    def obtain(indices):
        if capable:
            _check_budget(budget)
        began = time.perf_counter()
        batch = provider.batch(indices, arm, epoch, training, full=full)
        if pin_memory:
            batch = batch.pin_memory()
        if capable:
            _check_budget(budget)
        return batch, time.perf_counter() - began

    iterator = iter(batches)
    following = next(iterator, None)
    if following is None:
        return
    pending = deque()
    scope = None
    last_mode = None
    failed = False
    with ThreadPoolExecutor(max_workers=ceiling, thread_name_prefix='comparison-cpu') as pool:
        if capable:
            rss_before, _ = _prefetch_resources()
        pending.append((pool.submit(obtain, following), False, 0))
        counts['submitted'] += 1
        counts['cold_batches'] += 1
        counts['peak_queued_batches'] = 1
        try:
            following = next(iterator, None)
            first = True
            while pending:
                waiting = time.perf_counter()
                active, active_warm, _ = pending.popleft()
                if first and capable:
                    peak = rss_before
                    while True:
                        rss, _ = _prefetch_resources()
                        peak = max(peak, rss)
                        try:
                            batch, loader_seconds = active.result(timeout=.05)
                            break
                        except FutureTimeout:
                            # A completed provider is allowed to raise TimeoutError.
                            if active.done():
                                batch, loader_seconds = active.result()
                                break
                    peak = max(peak, _prefetch_resources()[0])
                    plan['first_batch_peak_rss_bytes'] = peak
                    plan['first_batch_rss_growth_bytes'] = max(0, peak - rss_before)
                else:
                    batch, loader_seconds = active.result()
                del active
                loader_wait_seconds = time.perf_counter() - waiting
                counts['loader_wait_seconds'] += loader_wait_seconds
                if capable:
                    from .u_bridge_data import _resident_size
                    size = _resident_size(batch)
                    plan['largest_batch_storage_bytes'] = max(plan['largest_batch_storage_bytes'], size)
                    # Two inventories plus pin/collate scratch, and 25% observed
                    # RSS-growth margin. Aliased tensor storage is counted once.
                    plan['slot_estimate_bytes'] = max(1, 3 * plan['largest_batch_storage_bytes'],
                        math.ceil(plan['first_batch_rss_growth_bytes'] * 1.25))
                    rss, available = _prefetch_resources()
                    usable = min(max(0, budget.rss_bytes - rss), available) - reserve
                    measured_slots = max(1, min(ceiling, usable // plan['slot_estimate_bytes']))
                    # Re-evaluate after cache release or larger inputs. A low
                    # first reading must not lock the whole epoch to one slot.
                    plan['slots'] = measured_slots
                    if first:
                        receipt('first_batch_measured', current_rss_bytes=rss,
                                os_available_bytes=available)
                elif first:
                    receipt('single_batch_compatibility')
                first = False

                # Original providers still serialize cold hierarchies. The
                # isolated provider can overlap different cases and sources.
                while following is not None:
                    warm = bool(capable and ready(following, arm, epoch,
                                                  training=training, full=full))
                    cold_workspace = cold_estimate(following) if cold_capable and not warm else None
                    independent = warm or (cold_capable and cold_workspace is not None)
                    if not independent and pending:
                        break
                    if pending and scope is None:
                        break
                    if capable:
                        _check_budget(budget)
                        rss, available = _prefetch_resources()
                        usable = min(max(0, budget.rss_bytes - rss), available) - reserve
                        admission = max(1, min(plan['slots'], usable // plan['slot_estimate_bytes']))
                    else:
                        admission = 1
                    estimate = plan['slot_estimate_bytes']
                    if cold_workspace is not None:
                        estimate += cold_workspace
                    if len(pending) >= (admission if independent else 1):
                        break
                    if pending and sum(job[2] for job in pending) + estimate > usable:
                        break
                    if not pending and scope is not None:
                        scope.__exit__(None, None, None)
                        scope = None
                    if independent and scope is None:
                        new_scope = staging(plan['slots'])
                        new_scope.__enter__()
                        scope = new_scope
                    mode = (('cached_shared' if warm else 'cold_shared', plan['slots'])
                            if independent else ('cold_serial', 1))
                    if mode != last_mode:
                        receipt('scheduling_mode', mode=mode[0], planned_slots=mode[1],
                                current_admission_slots=admission if independent else 1,
                                next_batch_estimated_bytes=estimate,
                                cold_workspace_measured_or_inventory_known=cold_workspace is not None,
                                candidate_workers=workers, coordinator_slots=mode[1],
                                executor_capacity_total=workers+mode[1],
                                executor_capacity_scope='one shared candidate pool plus batch coordinator slots; '
                                                        'excludes main thread and native library pools')
                        last_mode = mode
                    pending.append((pool.submit(obtain, following), warm, estimate))
                    counts['submitted'] += 1
                    counts['warm_batches' if warm else 'cold_batches'] += 1
                    counts['peak_queued_batches'] = max(counts['peak_queued_batches'], len(pending))
                    following = next(iterator, None)
                    if not independent:
                        break
                counts['yielded'] += 1
                if with_timing:
                    # PhaseProgress and the sealed callers accept elapsed
                    # seconds only. Queue policy belongs in prefetch.jsonl.
                    yield batch, dict(loader_seconds=loader_seconds, loader_wait_seconds=loader_wait_seconds)
                else:
                    yield batch
                del batch
        except BaseException:
            failed = True
            raise
        finally:
            primary = sys.exc_info()[1]
            errors = []
            while pending:
                future, _, _ = pending.popleft()
                try:
                    # Already-running jobs are drained, including their errors.
                    # Jobs that have not started need no materialization on pause.
                    if not capable or not future.cancel():
                        future.result()
                except Exception as error:
                    errors.append(error)
                finally:
                    del future
            if scope is not None:
                try:
                    scope.__exit__(None, None, None)
                except Exception as error:
                    errors.append(error)
            try:
                receipt('finished', status='interrupted' if failed else 'complete', **counts)
            except Exception as error:
                errors.append(error)
            if errors:
                drain_error = errors[0] if len(errors) == 1 else _StagingCleanupError(errors)
                if primary is not None and not isinstance(primary, GeneratorExit):
                    raise primary.with_traceback(primary.__traceback__) from drain_error
                raise drain_error


EXECUTION_HELPERS = dict(_prefetch=_prefetch, PhaseProgress=PhaseProgress,
                         RunningPatientMetrics=RunningPatientMetrics, closing=closing,
                         ComparisonGpuRuntime=ComparisonGpuRuntime)


def run_arm(net, provider, config, *, arm, output, physical_batch, workers, epochs, identity, budget, debug=False):
    """Run/resume one independent arm; caller supplies a freshly initialized net."""
    if arm not in ARMS or type(debug) is not bool or type(epochs) is not int or epochs < 1:
        raise ValueError('Explicit arm/DEBUG/epoch contract required')
    if not debug and epochs != 40: raise ValueError('Production U bridge preserves forty epochs')
    if type(physical_batch) is not int or physical_batch < 1 or workers < 2:
        raise ValueError('Measured physical sample batch and parallel workers required')
    if not torch.cuda.is_available(): raise RuntimeError('Actual CUDA required; no CPU training fallback')
    import psutil
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
    # Runtime attributes never enter the sealed model/optimizer identity. Probe
    # the actual resumed weights and the facade's actual objective, not an
    # imported pairwise loss that would override native_listwise.
    gpu_execution = ComparisonGpuRuntime(net,
        lambda result: pair_objective(result.scores, result.consistency)[0],
        amp=training['amp'], cuda_limit_bytes=budget.cuda_bytes,
        output=root, notify=PhaseProgress.write)
    receipt_path = root / 'execution_contract.json'
    if not receipt_path.exists():
        _write_new(receipt_path, dict(format=FORMAT, arm=arm, debug=debug, epochs=epochs,
            parameters=EXPECTED_PARAMETERS, trainable_parameters=EXPECTED_PARAMETERS,
            layers=dict(L0=3, L1=2, L2=2), hidden=128, heads=4, margin_mm=10,
            training_samples=len(examples), validation_samples=len(validation), training_cases=len(train_cases),
            validation_cases=len(val_cases), physical_sample_batch=physical_batch,
            physical_candidate_rows=physical_batch * 8, physical_graph_views=physical_batch * 16,
            accumulation=1, effective_sample_batch=physical_batch, workers=workers, prefetch_batches=1,
            prefetch_batches_scope='initial and compatibility path; active measured CPU staging policy in prefetch.jsonl',
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
        gpu_execution.restore_original()
        net.eval()
        remaining = [val_ids[i:i + physical_batch]
                     for i in range(state['validation_position'], len(val_ids), physical_batch)]
        running = RunningPatientMetrics(state['validation_rows'])
        with PhaseProgress(root=root, arm=arm, phase='validation129', epoch=label_epoch,
                           epochs=epochs, total=len(val_ids), initial=state['validation_position'],
                           physical_batch=physical_batch, metrics=running.metrics()) as progress:
            # Publish before the first load. Measured cached CPU staging does
            # not change the full129 GPU/chunk transfer strategy.
            with closing(_prefetch(provider, remaining, arm, training['fixed_validation_epoch'],
                                   training=False, full=True, pin_memory=False, with_timing=True,
                                   receipt_path=root / 'prefetch.jsonl')) as staged:
                for ids in remaining:
                    if pause_requested(flag):
                        return False
                    progress.update(stage='load')
                    before_rng = capture_rng(); loading = time.perf_counter()
                    try:
                        batch, input_timing = next(staged)
                        _check_budget(budget)
                        if list(batch.counts) != [129] * len(ids) or batch.local_batch_view2 is None:
                            raise ValueError('Full validation requires P+128U and two actual original views')
                        if tuple(getattr(batch, 'bridge_indices', ())) != tuple(ids): raise ValueError('Validation source binding changed')
                        progress.update(stage='forward', timings=input_timing)
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize(); forward = time.perf_counter()
                        with torch.no_grad(), torch.autocast('cuda', enabled=training['amp']):
                            scores = net.score_inference_chunked(batch, local_chunk_size=chunk)
                        torch.cuda.synchronize(); forward_seconds = time.perf_counter() - forward
                        if len(scores) != len(ids): raise ValueError('Full joint upper lost source problem mapping')
                        rows = [score_row(score, lookup[index], provider.candidate_keys(index, arm,
                            training['fixed_validation_epoch'], full=True), expected_candidates=129)
                            for index, score in zip(ids, scores)]
                        state['validation_rows'].extend(rows); state['validation_position'] += len(ids)
                        running.add(rows)
                        _check_budget(budget)
                        timing = dict(epoch=label_epoch, source_indices=ids,
                            **input_timing, full_joint_forward_seconds=forward_seconds,
                            source_problems=len(ids), candidate_rows=129 * len(ids), local_chunk_size=chunk,
                            peak_cuda_bytes=torch.cuda.max_memory_allocated(), rss_bytes=psutil.Process().memory_info().rss)
                        del scores, batch
                    finally: restore_rng(before_rng)
                    progress.update(stage='save', completed=state['validation_position'], metrics=running.metrics(),
                                    timings=dict(forward_seconds=forward_seconds))
                    state['validation_active_seconds'] += time.perf_counter() - loading
                    checkpoint_seconds = checkpoint()
                    state['validation_active_seconds'] += checkpoint_seconds
                    elapsed = time.perf_counter() - loading
                    timing.update(checkpoint_seconds=checkpoint_seconds, batch_wall_seconds=elapsed,
                                  samples_per_second_including_wait=len(ids) / elapsed)
                    _append(root / 'validation_timing.jsonl', timing)
                    progress.update(timings=dict(checkpoint_seconds=checkpoint_seconds))
            progress.update(stage='save')
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
            progress.update(metrics=report['metrics'])
            progress.finish()
            metrics = report['metrics']
            PhaseProgress.write(f'{arm} validation129 epoch {label_epoch} COMPLETE '
                                f'{len(rows)}/{len(val_ids)} sources | patient macro')
            PhaseProgress.write(f'  MRR={metrics["mrr"]:.6f} top1={metrics["top1"]:.6f} '
                                f'pair-win={metrics["pair_win"]:.6f}')
            PhaseProgress.write(f'  pair-loss={metrics["pair_loss"]:.6f} margin={metrics["margin"]:.6f} '
                                f'validation={report["wall_seconds"]:.1f}s')
            if phase != 'initial_validation':
                best = state['best']
                PhaseProgress.write(f'  epoch total={row["epoch_wall_seconds"]:.1f}s '
                                    f'train/load/save={row["optimization_loader_save_seconds"]:.1f}s '
                                    f'BEST={"updated" if improved else "unchanged"} epoch {best["epoch"]} '
                                    f'MRR={best["selection_key"][0]:.6f}')
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
                running = RunningPatientMetrics(state['train_rows'])
                previous = time.perf_counter()
                with PhaseProgress(root=root, arm=arm, phase='train7', epoch=state['epoch'],
                                   epochs=epochs, total=len(order), initial=state['position'],
                                   physical_batch=physical_batch, metrics=running.metrics()) as bar:
                    with closing(_prefetch(provider, remaining, arm, state['epoch'], with_timing=True,
                                           receipt_path=root / 'prefetch.jsonl')) as staged:
                        for batch, input_timing in staged:
                            loading_seconds = time.perf_counter() - previous; step_start = time.perf_counter()
                            bar.update(stage='forward', timings=input_timing)
                            ids = order[state['position']:state['position'] + physical_batch]
                            if list(batch.counts) != [8] * len(ids) or batch.local_batch_view2 is None:
                                raise ValueError('Actual original8/two-view training batch required')
                            if tuple(getattr(batch, 'bridge_indices', ())) != tuple(ids): raise ValueError('Batch source binding changed')
                            workload = batch_workload(batch); _check_budget(budget)
                            optimizer.zero_grad(set_to_none=True)
                            # This is the same complete batch used by the update.
                            # Calibration does not step Adam, advance the cursor,
                            # or consume the training RNG sequence.
                            transfer_start = time.perf_counter()
                            batch.to(device, non_blocking=True)
                            torch.cuda.synchronize()
                            input_transfer_seconds = time.perf_counter() - transfer_start
                            bar.update(stage='GPU preparation/calibration')
                            calibration_start = time.perf_counter()
                            gpu_execution.ensure(batch, workload)
                            execution_calibration_seconds = time.perf_counter() - calibration_start
                            bar.update(stage='forward', timings=dict(
                                execution_calibration_seconds=execution_calibration_seconds))
                            while True:
                                attempt_start = time.perf_counter()
                                optimizer.zero_grad(set_to_none=True); state['attempts'] += 1
                                torch.cuda.reset_peak_memory_stats(); events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
                                events[0].record(); events[1].record()
                                attempt_rng = capture_rng()
                                result = loss = terms = None
                                try:
                                    with torch.autocast('cuda', enabled=training['amp']):
                                        result = net(batch)
                                        if len(result.scores) != len(ids): raise ValueError('Training forward lost source problem mapping')
                                        loss, terms = pair_objective(result.scores, result.consistency)
                                    events[2].record()
                                    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite actual U-bridge loss')
                                    scaler.scale(loss).backward()
                                except torch.cuda.OutOfMemoryError as error:
                                    # No optimizer/scaler step or unscale has occurred.
                                    # Drop failed autograd references before retrying
                                    # the exact same input/RNG with original settings.
                                    result = loss = terms = None
                                    optimizer.zero_grad(set_to_none=True)
                                    restore_rng(attempt_rng)
                                    if not gpu_execution.reject_optimized_oom(workload, error):
                                        raise
                                    import traceback
                                    traceback.clear_frames(error.__traceback__)
                                    error.__traceback__ = None
                                    import gc
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    state['attempts'] -= 1
                                    continue
                                scaler.unscale_(optimizer)
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
                                PhaseProgress.write(f'{arm} AMP overflow: same sources {ids}, cursor {state["position"]}, '
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
                            running.add(rows)
                            record_comparisons(state, rows, optimizer_updated=True)
                            state['connected'] = sorted(set(state['connected']) | {n for n, p in net.named_parameters() if p.requires_grad and p.grad is not None})
                            values = dict(loss=float(loss.detach()), ranking=float(terms['ranking'].detach()),
                                          consistency=float(terms['consistency'].detach()), gradient_before_clip=float(clipped))
                            del result, loss, terms, probes, batch
                            bar.update(stage='save', completed=state['position'], metrics=running.metrics(), loss=values['loss'])
                            checkpoint_seconds = checkpoint(); elapsed = time.perf_counter() - step_start
                            state['epoch_active_seconds'] = state.get('epoch_active_seconds', 0.) + loading_seconds + elapsed
                            _check_budget(budget)
                            row = dict(status='OPTIMIZER_UPDATED', epoch=state['epoch'], update=state['updates'], attempt=state['attempts'],
                                sample_indices=ids, physical_samples=len(ids), candidate_rows=8 * len(ids), graph_views=16 * len(ids),
                                candidate_keys={str(row['sample_index']): row['candidate_keys'] for row in rows},
                                actual_input=workload, AMP_overflows=state['overflows'],
                                **values, gradient=gradients, sampled_parameter_changes=changed,
                                margin=sum(r['margin'] for r in rows) / len(rows), train7_pair_win=sum(r['pair_win'] for r in rows) / len(rows),
                                loader_seconds=input_timing['loader_seconds'], loader_wait_seconds=loading_seconds,
                                batch_wall_seconds=loading_seconds + elapsed,
                                samples_per_second_including_wait=len(ids) / (loading_seconds + elapsed), transfer_seconds=input_transfer_seconds,
                                execution_calibration_seconds=execution_calibration_seconds,
                                gpu_execution=gpu_execution.current_receipt(),
                                forward_seconds=events[1].elapsed_time(events[2]) / 1000,
                                backward_seconds=events[2].elapsed_time(events[3]) / 1000,
                                check_clip_optimizer_seconds=events[3].elapsed_time(events[4]) / 1000,
                                checkpoint_seconds=checkpoint_seconds, step_seconds=elapsed,
                                samples_per_second=len(ids) / elapsed, candidate_graphs_per_second=8 * len(ids) / elapsed,
                                peak_cuda_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                rss_bytes=process.memory_info().rss, cpu_percent=process.cpu_percent(), learning_rate=optimizer.param_groups[0]['lr'])
                            _append(root / 'update_timing.jsonl', row)
                            bar.update(timings=dict(forward_seconds=row['forward_seconds'], checkpoint_seconds=checkpoint_seconds))
                            if pause_requested(flag): status = 'PAUSED'; break
                            bar.update(stage='load')
                            previous = time.perf_counter()
                    bar.finish('PAUSED' if status == 'PAUSED' else 'COMPLETE')
                if status == 'PAUSED': break
                if state['position'] != len(order): raise ValueError('Incomplete source-problem epoch')
                if {row['sample_index'] for row in state['train_rows']} != set(train_ids): raise ValueError('Training coverage changed')
                state['phase'] = 'validation'; checkpoint()
        except Exception as error:
            _append(root / 'failures.jsonl', dict(epoch=state['epoch'], phase=state['phase'], position=state['position'],
                updates=state['updates'], error=f'{type(error).__name__}: {error}',
                recovery='Latest owned successful/AMP-skip checkpoint preserved; exact resume retries unchanged cursor'))
            raise
        finally:
            gpu_execution.restore_original()
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
