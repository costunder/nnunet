"""Fresh GT-blind v2.4 singleton training with measured cumulative coverage.

The all-P patient loss and physical disjoint-union batching are shared with
the frozen v2.3 implementation. Inputs and model are explicitly v2.4-owned;
v2.3 trained weights and annotation-dependent canonical caches are refused.
"""
from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
import time

import torch

from .v23_training import (V23Scorer, patient_balanced_objective, score_patient,
    aggregate_patients, best_selection_key, parallel_patient_batches,
    publish_epoch_curve, _plans, _positive_indices)
from .u_bridge_training import (_append, _check_budget, _groups, _write_new,
    atomic_save, capture_rng, cpu_copy, digest, gradient_receipt,
    restore_optimizer_history, restore_rng, retry_amp_overflow)
from .v24_targets import new_curriculum, finish_curriculum_epoch, validate_policy

FORMAT = 'v24_GT_blind_all_P_patient_balanced_training_v1'
EVALUATION_PHASES = ('initial_full_validation', 'train_probe', 'stage_validation', 'full_validation')
KNOWN_BUDGET_ERRORS=frozenset((
    'Explicit CUDA/RSS budget exceeded; no model/data/batch fallback',
    'Full active GT-free input exceeds shared RSS; no graph or data reduction',
    'Complete active v24 canonical/view chunk exceeds resident budget',
    'Actual complete v24 workspace exceeds process RSS; no subset or fallback',
    'v2.4 geometry RSS budget exceeded; no graph/data reduction',
    'Historical evaluation CUDA budget exceeded; no input/model reduction',
    'Historical evaluation RSS budget exceeded; no cases skipped'))


def patient_batches(case_ids, physical):
    """All patients once, nominal measured maximum; rebalance singleton tails."""
    batches=parallel_patient_batches(case_ids,physical,1)
    if len(batches)>1 and len(batches[-1])==1:
        joined=batches[-2]+batches[-1]; split=len(joined)//2
        batches[-2:]=[joined[:split],joined[split:]]
    if any(not 2<=len(batch)<=physical for batch in batches):
        raise ValueError('Complete population cannot form parallel patient batches within measured capacity')
    return batches


class V24Scorer(V23Scorer):
    """Preserve real batched arithmetic and additionally require GT blindness."""
    def __init__(self, net, providers, geometry, **kwargs):
        if any(getattr(p, 'recipient_GT_used_in_forward', None) is not False for p in providers.values()):
            raise ValueError('v24 refuses GT-dependent local/canonical input providers before forward')
        super().__init__(net, providers, geometry, **kwargs)

    def forward(self, plans, *, epoch, training):
        if not plans:
            raise ValueError('Complete real v24 patient batch required')
        provider = self.providers[plans[0].partition]
        if getattr(provider, 'recipient_GT_used_in_forward', None) is not False:
            raise ValueError('v24 refuses GT-dependent local/canonical input providers')
        # The v24 geometry cache admits every audit before returning tensors to
        # the original batched scorer. Refuse replacement with a legacy cache.
        if getattr(self.geometry, 'recipient_GT_used_in_forward', None) is not False:
            raise ValueError('v24 requires GT-blind upper input ownership before forward')
        result = super().forward(plans, epoch=epoch, training=training)
        if any(audit.get('recipient_GT_used_in_forward') is not False for audit in result.audits):
            raise ValueError('v24 requires explicit recipient-GT-blind upper inputs')
        result.workload.update(version='v2.4', recipient_GT_used_in_forward=False,
            recipient_annotation_dependent_canonical_cache_used=False)
        return result


def optimizer_groups(net, training):
    factory = getattr(net, 'v24_optimizer_parameter_groups', None)
    groups = (factory(training['lr'], training['weight_decay']) if callable(factory)
        else [dict(params=[p for p in net.parameters() if p.requires_grad], lr=training['lr'],
                   weight_decay=training['weight_decay'])])
    params = [p for group in groups for p in group['params']]
    expected = [p for p in net.parameters() if p.requires_grad]
    if len(params) != len(set(map(id, params))) or set(map(id, params)) != set(map(id, expected)):
        raise ValueError('Every actual trainable parameter must occur exactly once in the optimizer')
    if any(group['lr'] <= 0 or group['weight_decay'] < 0 for group in groups):
        raise ValueError('Explicit positive optimizer LR and nonnegative decay required')
    return groups


def validate_model(net, model_contract, *, debug=False):
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if (model_contract.get('recipient_GT_used_in_forward') is not False
            or model_contract.get('trained_v23_weights_loaded') is not False
            or model_contract.get('parameters') != total
            or model_contract.get('trainable_parameters') != trainable):
        raise ValueError('Actual v24 GT-blind model/initialization/count provenance required')
    if not debug and next(net.parameters()).device.type != 'cuda':
        raise RuntimeError('Production v24 requires actual CUDA without fallback')
    return _groups(net)


def calibrate_training_batches(net, scorer, population, config, *, budget, model_contract, debug=False):
    """Full native allP+128U clone updates; production weights/RNG stay intact."""
    validate_model(net, model_contract, debug=debug)
    runtime, training = config['v24_runtime'], config['training']
    patients, chunks = runtime['physical_patient_batch_candidates'], runtime['physical_candidate_batch_candidates']
    if (not patients or not chunks or patients != sorted(set(patients)) or min(patients) < 4
            or chunks != sorted(set(chunks)) or min(chunks) < 32
            or runtime['calibration_repeats'] < 3):
        raise ValueError('Measured search preserves physical patient4/candidate32 minimum and three repeats')
    cases = list(population.partition_cases('inner_train', ranking_only=True))
    if max(patients) > len(cases): raise ValueError('Calibration cannot duplicate real patients')
    provider=scorer.providers['inner_train']
    def measured_cost(case):
        plan=population.case(case,128)
        return sum(provider.records[record_id]['sampled_two_view_edges'] for record_id in plan.record_ids)
    ordered = sorted(cases, key=measured_cost, reverse=True)
    old_rng, old_hash = capture_rng(), digest(net.state_dict())
    old_model, old_chunk = scorer.net, scorer.physical_candidate_batch
    trials = []
    try:
        for physical in patients:
            plans = _plans(population, ordered[:physical], 128)
            for chunk in chunks:
                restore_rng(old_rng)
                clone = copy.deepcopy(net).train(); scorer.net = clone; scorer.physical_candidate_batch = chunk
                optimizer = torch.optim.AdamW(optimizer_groups(clone, training), fused=training['fused_optimizer'])
                scaler = torch.amp.GradScaler('cuda', enabled=training['amp'])
                measurements, overflow = [], []
                print(f'v24 CALIBRATION patients{physical}/chunk{chunk}/allP+128U', flush=True)
                result = loss = None
                try:
                    torch.cuda.reset_peak_memory_stats()
                    for repeat in range(runtime['calibration_repeats']):
                        torch.cuda.synchronize(); started = time.perf_counter()
                        while True:
                            optimizer.zero_grad(set_to_none=True)
                            result = scorer(plans, epoch=1, training=True)
                            loss, _ = patient_balanced_objective(result.scores,
                                [_positive_indices(plan) for plan in plans], result.consistency)
                            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                            gradient = gradient_receipt(clone, _groups(clone))
                            if gradient['missing']: raise RuntimeError('Disconnected v24 calibration parameters: ' + repr(gradient['missing']))
                            if gradient['finite']: break
                            before, after = retry_amp_overflow(scaler, optimizer)
                            overflow.append(dict(repeat=repeat, scale_before=before, scale_after=after))
                            result = loss = None
                        torch.nn.utils.clip_grad_norm_(clone.parameters(), training['grad_clip'], error_if_nonfinite=True)
                        scaler.step(optimizer); scaler.update(); torch.cuda.synchronize(); _check_budget(budget)
                        measurements.append(dict(seconds=time.perf_counter()-started, gradient=gradient))
                        print(f'v24 CALIBRATION repeat{repeat+1} {measurements[-1]["seconds"]:.3f}s', flush=True)
                    seconds = sum(row['seconds'] for row in measurements)/len(measurements)
                    peak_cuda_bytes = torch.cuda.max_memory_allocated()
                    if peak_cuda_bytes > budget.cuda_bytes:
                        trials.append(dict(physical_patient_batch=physical, physical_candidate_batch=chunk,
                            accepted=False, failure='CUDA_BUDGET', peak_cuda_bytes=peak_cuda_bytes,
                            CUDA_budget_bytes=budget.cuda_bytes, graph_model_population_unchanged=True))
                        continue
                    trials.append(dict(physical_patient_batch=physical, physical_candidate_batch=chunk,
                        accepted=True, seconds=seconds, patients_per_second=physical/seconds,
                        measurements=measurements, gradient=gradient, all_P_and_U128=True,
                        case_ids=list(result.case_ids), workload=result.workload,
                        optimizer_updates_on_clone=len(measurements), AMP_overflows=overflow,
                        peak_cuda_bytes=peak_cuda_bytes))
                except torch.cuda.OutOfMemoryError as error:
                    trials.append(dict(physical_patient_batch=physical, physical_candidate_batch=chunk,
                        accepted=False, failure='CUDA_OOM', error=str(error), graph_model_population_unchanged=True))
                except MemoryError as error:
                    if str(error)not in KNOWN_BUDGET_ERRORS:raise
                    trials.append(dict(physical_patient_batch=physical,physical_candidate_batch=chunk,
                        accepted=False,failure='CUDA_BUDGET' if torch.cuda.memory_allocated()>budget.cuda_bytes else 'RAM_BUDGET',
                        error=str(error),graph_model_population_unchanged=True,
                        allocated_CUDA_bytes=torch.cuda.memory_allocated()))
                finally:
                    result = loss = None; scorer.net = old_model
                    del clone, optimizer, scaler
                    gc.collect()
                    torch.cuda.empty_cache()
        accepted = [row for row in trials if row['accepted']]
        if not accepted: raise MemoryError('No full-model/full128 configuration fits without reducing physical4; no fallback')
        selected = max(accepted, key=lambda row: row['patients_per_second'])
    finally:
        scorer.net, scorer.physical_candidate_batch = old_model, old_chunk
        restore_rng(old_rng)
    if digest(net.state_dict()) != old_hash: raise RuntimeError('Calibration altered fresh production model')
    return dict(format=FORMAT, trials=trials, world_size=1, initial_state_sha256=old_hash,
        selected_physical_patient_batch=selected['physical_patient_batch'],
        selected_physical_candidate_batch=selected['physical_candidate_batch'],
        measured_full_P_U128_backward=True, original_model_and_RNG_preserved=True,
        recipient_GT_used_in_forward=False, model_contract=model_contract, debug=debug)


def _validate_progress(state, train_cases, val_cases, physical, epochs, scheduler):
    phases = ('training', 'epoch_completion', 'complete', *EVALUATION_PHASES)
    if state['phase'] not in phases or not 1 <= state['epoch'] <= epochs+1:
        raise ValueError('Invalid v24 checkpoint phase/epoch')
    if scheduler.last_epoch != len(state['history']) or state['epoch'] != len(state['history'])+1:
        raise ValueError('Checkpoint scheduler/completed epoch history differs')
    if state['train_order'] is not None and (len(state['train_order']) != len(train_cases)
            or set(state['train_order']) != set(train_cases)):
        raise ValueError('Checkpoint train order lost complete patient coverage')
    if state['train_position'] != len(state['train_rows']): raise ValueError('Saved train cursor/rows differ')
    if state['train_order'] is not None and [r['case_id'] for r in state['train_rows']] != state['train_order'][:state['train_position']]:
        raise ValueError('Saved patient order and completed updates differ')
    phase = state['phase']
    cases = train_cases if phase == 'train_probe' else val_cases if phase in EVALUATION_PHASES else []
    if state['evaluation_position'] != len(state['evaluation_rows']) or [r['case_id'] for r in state['evaluation_rows']] != cases[:state['evaluation_position']]:
        raise ValueError('Saved evaluation cursor/complete population differs')
    if state['train_order'] is not None:
        offsets = {0}; cursor = 0
        for batch in patient_batches(state['train_order'], physical):
            cursor += len(batch); offsets.add(cursor)
        if state['train_position'] not in offsets: raise ValueError('Train cursor is not a complete physical batch')
    if cases:
        offsets = {0}; cursor = 0
        for batch in patient_batches(cases, physical): cursor += len(batch); offsets.add(cursor)
        if state['evaluation_position'] not in offsets: raise ValueError('Evaluation cursor is not a complete physical batch')
    if state['updates'] < 0 or state['attempts'] < state['updates']:
        raise ValueError('Invalid saved optimizer counters')


def validate_calibration(calibration, net, scorer, contract, physical, *, repeats, budget, debug=False):
    expected=len([p for p in net.parameters() if p.requires_grad])
    if (calibration['format']!=FORMAT or calibration['initial_state_sha256']!=digest(net.state_dict())
            or calibration['world_size']!=1 or calibration['selected_physical_patient_batch']!=physical
            or calibration['selected_physical_candidate_batch']!=scorer.physical_candidate_batch
            or calibration['model_contract']!=contract or calibration['recipient_GT_used_in_forward']is not False
            or calibration['debug']is not debug or calibration['original_model_and_RNG_preserved']is not True
            or calibration['measured_full_P_U128_backward']is not True):
        raise ValueError('Actual fresh-v24 model-specific full128 calibration required')
    matches=[trial for trial in calibration['trials'] if trial['physical_patient_batch']==physical
        and trial['physical_candidate_batch']==scorer.physical_candidate_batch and trial['accepted']is True]
    if len(matches)!=1:raise ValueError('Exactly one actual selected calibration trial required')
    trial=matches[0]
    if (trial['all_P_and_U128']is not True or len(trial['measurements'])!=repeats
            or trial['optimizer_updates_on_clone']!=repeats or trial['peak_cuda_bytes']>budget.cuda_bytes
            or trial['workload']['patients']!=physical or trial['workload']['upper_invocations']!=1):
        raise ValueError('Selected full128 trial lacks actual repeat/batching/VRAM evidence')
    for row in trial['measurements']:
        gradient=row['gradient']
        if (row['seconds']<=0 or gradient['finite']is not True or gradient['missing']
                or gradient['trainable_parameter_tensors']!=expected or gradient['gradient_present']!=expected):
            raise ValueError('Every actual calibration parameter must have a finite loss gradient')
    return trial


def validate_saved_best(root,state,identity_sha256):
    """A saved metric never substitutes for the actual owned winning weights."""
    if state['best'] is None:return
    path=Path(root)/'checkpoint_best.pt'
    if not path.is_file() or path.is_symlink():
        raise ValueError('Actual owned BEST checkpoint is missing; preserve latest and repair explicitly')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if (saved.get('format')!=FORMAT or saved.get('identity_sha256')!=identity_sha256
            or saved.get('content_sha256')!=digest({k:v for k,v in saved.items() if k!='content_sha256'})
            or saved['state']['best']!=state['best'] or saved['state']['epoch']!=state['best']['epoch']
            or saved['state']['updates']!=state['best']['updates']):
        raise ValueError('Actual owned BEST content/identity/winning state differs')
    full=saved['state']['pending_epoch_completion']['full_validation']
    if full['phase']!='full_validation' or full['active_u']!=128 or list(best_selection_key(full))!=state['best']['selection_key']:
        raise ValueError('Actual BEST was not selected by its full128 validation')


def run_training(net, scorer, population, config, *, output, identity, budget,
                 model_contract, physical_patient_batch, workers=4, debug=False):
    """Full40 GT-blind run/resume with explicit train/stage/full evaluation."""
    import psutil
    runtime, training = config['v24_runtime'], config['training']
    groups = validate_model(net, model_contract, debug=debug)
    policy = runtime['curriculum']; validate_policy(policy)
    epochs = policy['total_epochs']
    if not debug and (epochs != 40 or policy['total_u'] != 128 or physical_patient_batch < 4 or workers < 4):
        raise ValueError('Production requires40epochs/128U/physical4+/explicit CPU4+ without hidden reduction')
    if torch.distributed.is_initialized(): raise ValueError('v24 GPU5/6 experiments are independent singletons')
    if training['consistency_weight'] != .1 or training['gradient_accumulation_steps'] != 1:
        raise ValueError('Original patient consistency0.1/accumulation1 objective contract required')
    if scorer.net is not net or scorer.amp != bool(training['amp']): raise ValueError('Scorer/model precision ownership differs')
    calibration = runtime['batch_calibration']; initial_hash = digest(net.state_dict())
    validate_calibration(calibration,net,scorer,model_contract,physical_patient_batch,
        repeats=runtime['calibration_repeats'],budget=budget,debug=debug)
    train_cases = list(population.partition_cases('inner_train', ranking_only=True))
    val_cases = list(population.partition_cases('inner_val', ranking_only=False))
    if not train_cases or not val_cases or set(train_cases)&set(val_cases): raise ValueError('Complete disjoint actual populations required')
    if not debug and (len(train_cases) != 65 or len(val_cases) != 21
            or sum(len(population.case(case,128).positive_indices) for case in train_cases) != 527
            or sum(len(population.case(case,128).positive_indices) for case in val_cases) != 135):
        raise ValueError('Production all527P/65rankingtrain/21val135P population differs')
    optimizer = torch.optim.AdamW(optimizer_groups(net, training), fused=training['fused_optimizer'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=training['amp'])
    generator = torch.Generator().manual_seed(42+2003)
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=True)
    bound_config = copy.deepcopy(config)
    bound_config['v24_runtime'].pop('resume_checkpoint', None)
    binding = dict(format=FORMAT, identity=identity, config=bound_config, population=population.manifest(),
        model_contract=model_contract, train_cases=train_cases, val_cases=val_cases,
        initial_state_sha256=initial_hash, physical_patient_batch=physical_patient_batch,
        physical_candidate_batch=scorer.physical_candidate_batch, workers=workers, epochs=epochs, debug=debug)
    binding_hash = digest(binding); ownership = root/'training_identity.json'
    resume = runtime.get('resume_checkpoint')
    if ownership.exists():
        if not resume or json.loads(ownership.read_text())['identity_sha256'] != binding_hash:
            raise FileExistsError('Existing v24 results require explicit exact-identity resume; no overwrite')
    else:
        if any(root.iterdir()): raise FileExistsError('Existing results preserved; choose a fresh v24 training namespace')
        _write_new(ownership, dict(identity_sha256=binding_hash,binding=cpu_copy(binding)))
    state = dict(epoch=1, phase='initial_full_validation', status='RUNNING', updates=0, attempts=0,
        overflows=0, history=[], best=None, connected=[], train_order=None, train_position=0,
        train_rows=[], evaluation_position=0, evaluation_rows=[], epoch_start_updates=0,
        curriculum=new_curriculum(policy))
    if resume:
        saved = torch.load(resume,map_location='cpu',weights_only=False)
        checksum = saved.pop('content_sha256',None)
        if saved.get('format') != FORMAT or saved.get('identity_sha256') != binding_hash or digest(saved) != checksum:
            raise ValueError('Only exact owned v24 resume is allowed; leaked v23 weights are refused')
        net.load_state_dict(saved['model'],strict=True); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        if len(saved['rank_rng']) != 1: raise ValueError('Only owned singleton RNG resume allowed')
        state = saved['state']; generator.set_state(saved['shuffle_generator']); restore_rng(saved['rank_rng'][0])
        if state['curriculum']['policy']!=policy:raise ValueError('Owned resume curriculum differs from exact bound policy')
        restore_optimizer_history(optimizer,state['updates']); _validate_progress(state,train_cases,val_cases,physical_patient_batch,epochs,scheduler)
        validate_saved_best(root,state,binding_hash)
        del saved
    pause_file = root/'STOP_AFTER_BATCH'
    def save(status='RUNNING',best=False):
        state['status'] = status
        payload = dict(format=FORMAT,identity_sha256=binding_hash,model=cpu_copy(net.state_dict()),
            optimizer=cpu_copy(optimizer.state_dict()),scheduler=cpu_copy(scheduler.state_dict()),
            scaler=cpu_copy(scaler.state_dict()),state=cpu_copy(state),rank_rng=[cpu_copy(capture_rng())],
            shuffle_generator=generator.get_state().clone())
        payload['content_sha256'] = digest(payload)
        if best: atomic_save(root/'checkpoint_best.pt',payload)
        atomic_save(root/'checkpoint_latest.pt',payload)
    def publish(path,value):
        if path.exists():
            if digest(json.loads(path.read_text())) != digest(value): raise ValueError('Previously published owned report differs')
        else: _write_new(path,value)
    publish(root/'execution_contract.json',dict(binding=binding,model=type(net).__name__,
        parameters=sum(p.numel() for p in net.parameters()),model_contract=model_contract,
        fresh_model_optimizer=True,trained_v23_weights_loaded=False,
        loss='mean_U softplus(U-P), mean_P, mean_patient +0.1 two-view consistency',
        epochs=epochs,physical_patient_batch=physical_patient_batch,effective_patient_batch=physical_patient_batch,
        actual_tail_batches='Last two patient batches rebalanced within measured maximum; no singleton or duplicate',
        world_size=1,workers=workers,train_probe='all65 current-U fixed29 eval/no_grad',
        stage_validation='all21 current-U fixed29',BEST='all21 fixed128U only',
        curriculum=validate_policy(policy),recipient_GT_used_in_forward=False,
        debug=debug,full_training_completed=False))
    save(); began_run = time.perf_counter()
    try:
        while state['phase'] != 'complete':
            _check_budget(budget)
            if pause_file.exists(): save('PAUSED'); break
            phase = state['phase']; active = state['curriculum']['active_u']
            if phase == 'epoch_completion':
                pending = state['pending_epoch_completion']
                curve = dict(epoch=state['epoch'],update=state['updates'],train=aggregate_patients(state['train_rows']),
                    train_probe=pending['train_probe'],stage_validation=pending['stage_validation'],full_validation=pending['full_validation'],
                    curriculum_transition=pending['transition'])
                publish_epoch_curve(root/'curve.jsonl',curve)
                state['history'].append(curve); scheduler.step()
                state.update(epoch=state['epoch']+1,phase='training',curriculum=pending['curriculum'],
                    train_order=None,train_position=0,train_rows=[])
                del state['pending_epoch_completion']; save(); continue
            if phase == 'training':
                if state['epoch'] > epochs:
                    state['phase'] = 'complete'; save('COMPLETE'); continue
                net.train(); scorer.geometry.prepare(active)
                if state['train_order'] is None:
                    state['train_order'] = [train_cases[i] for i in torch.randperm(len(train_cases),generator=generator).tolist()]
                    state['epoch_start_updates'] = state['updates']; save()
                cursor = 0
                for cases in patient_batches(state['train_order'],physical_patient_batch):
                    start = cursor; cursor += len(cases)
                    if start < state['train_position']: continue
                    if pause_file.exists(): break
                    plans = _plans(population,cases,active); began = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
                    while True:
                        optimizer.zero_grad(set_to_none=True); state['attempts'] += 1
                        result = scorer(plans,epoch=state['epoch'],training=True)
                        loss,terms = patient_balanced_objective(result.scores,[_positive_indices(plan) for plan in plans],result.consistency)
                        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite actual v24 all-P loss')
                        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                        gradient = gradient_receipt(net,groups)
                        if gradient['missing']: raise RuntimeError('Disconnected v24 parameters: '+repr(gradient['missing']))
                        if gradient['finite']: break
                        before,after = retry_amp_overflow(scaler,optimizer); state['overflows'] += 1
                        _append(root/'update_timing.jsonl',dict(status='AMP_OVERFLOW_RETRY_SAME_INPUT',epoch=state['epoch'],
                            cursor=start,scale_before=before,scale_after=after)); result = loss = terms = None; save()
                    torch.nn.utils.clip_grad_norm_(net.parameters(),training['grad_clip'],error_if_nonfinite=True)
                    scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
                    rows = [score_patient(score,plan) for score,plan in zip(result.scores,plans)]
                    state['train_rows'].extend(rows); state['train_position'] = cursor; state['updates'] += 1
                    state['connected'] = sorted(set(state['connected'])|{n for n,p in net.named_parameters() if p.requires_grad and p.grad is not None})
                    _append(root/'update_timing.jsonl',dict(status='OPTIMIZER_UPDATED',epoch=state['epoch'],update=state['updates'],
                        active_u=active,case_ids=cases,gradient=gradient,loss=float(loss.detach()),workload=result.workload,
                        seconds=time.perf_counter()-began,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                        RSS_bytes=psutil.Process().memory_info().rss))
                    print(f'v24 epoch{state["epoch"]}/{epochs} U{active} patients{cursor}/{len(train_cases)} loss{float(loss.detach()):.6f}',flush=True)
                    del result,loss,terms; save()
                if pause_file.exists(): continue
                if [r['case_id'] for r in state['train_rows']] != state['train_order']: raise ValueError('Incomplete real training epoch')
                state['phase'] = 'train_probe'; save(); continue
            if phase not in EVALUATION_PHASES: raise ValueError('Unknown v24 checkpoint phase')
            net.eval(); eval_u = 128 if phase in ('initial_full_validation','full_validation') else active
            cases_all = train_cases if phase == 'train_probe' else val_cases
            scorer.geometry.prepare(eval_u); cursor = 0
            for cases in patient_batches(cases_all,physical_patient_batch):
                start = cursor; cursor += len(cases)
                if start < state['evaluation_position']: continue
                if pause_file.exists(): break
                plans = _plans(population,cases,eval_u); rng = capture_rng(); began = time.perf_counter()
                try:
                    with torch.no_grad(): result = scorer(plans,epoch=29,training=False)
                    rows = [score_patient(score,plan) for score,plan in zip(result.scores,plans)]
                    torch.cuda.synchronize(); state['evaluation_rows'].extend(rows); state['evaluation_position'] = cursor
                    _append(root/'validation_timing.jsonl',dict(epoch=state['epoch'],phase=phase,active_u=eval_u,
                        case_ids=cases,seconds=time.perf_counter()-began,workload=result.workload)); del result
                finally: restore_rng(rng)
                save()
            if pause_file.exists(): continue
            if [r['case_id'] for r in state['evaluation_rows']] != cases_all: raise ValueError('Evaluation omitted or duplicated actual patients')
            report = aggregate_patients(state['evaluation_rows']); report.update(epoch=0 if phase=='initial_full_validation' else state['epoch'],
                phase=phase,active_u=eval_u,evaluation_view_epoch=29,all_P_scored=True,updates=state['updates'],
                recipient_GT_used_in_forward=False,debug=debug)
            publish(root/f'{phase}_epoch_{report["epoch"]:03d}.json',report)
            state.update(evaluation_position=0,evaluation_rows=[])
            if phase == 'initial_full_validation': state['initial_validation'] = report; state['phase'] = 'training'; save()
            elif phase == 'train_probe': state['train_probe'] = report; state['phase'] = 'stage_validation'; save()
            elif phase == 'stage_validation': state['stage_validation'] = report; state['phase'] = 'full_validation'; save()
            else:
                probe,stage = state.pop('train_probe'),state.pop('stage_validation')
                curriculum,transition = finish_curriculum_epoch(state['curriculum'],probe,stage,epoch=state['epoch'],
                    actual_train_cases=[r['case_id'] for r in state['train_rows']],expected_train_cases=train_cases,
                    updates_in_epoch=state['updates']-state['epoch_start_updates'])
                key = best_selection_key(report); improved = state['best'] is None or key > tuple(state['best']['selection_key'])
                if improved: state['best'] = dict(epoch=state['epoch'],updates=state['updates'],selection_key=list(key),selected_by='fixed_full128_validation_only')
                state['pending_epoch_completion'] = dict(train_probe=probe,stage_validation=stage,
                    full_validation=report,curriculum=curriculum,transition=transition)
                state['phase'] = 'epoch_completion'; save(best=improved)
            print(f'v24 {phase} epoch{report["epoch"]} U{eval_u} MRR{report["metrics"]["per_P_patient_mrr"]:.6f}',flush=True)
    except Exception as error:
        _append(root/'failures.jsonl',dict(epoch=state['epoch'],phase=state['phase'],updates=state['updates'],
            error=f'{type(error).__name__}: {error}',recovery='Preserve last exact checkpoint; resume same complete physical batch'))
        raise
    completed = state['phase'] == 'complete'; save('DEBUG_COMPLETE' if debug and completed else 'COMPLETE' if completed else 'PAUSED')
    receipt = dict(format=FORMAT,status=state['status'],completed_epochs=len(state['history']),
        optimizer_updates=state['updates'],full_training=completed and not debug,debug=debug,
        actual_CUDA=True,recipient_GT_used_in_forward=False,best=state['best'],active_u=state['curriculum']['active_u'],
        connected_parameter_tensors=len(state['connected']),expected_parameter_tensors=len([p for p in net.parameters() if p.requires_grad]),
        seconds=time.perf_counter()-began_run,checkpoint=str(root/'checkpoint_latest.pt'),quality_verified=False)
    _append(root/'invocations.jsonl',receipt)
    return receipt
