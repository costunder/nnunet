"""CUDA-only C training: native LocalCNN + fixed B upper + original v1 ranking.

The dependency-closed native input block explicitly removes comparison
corruption and six-readout consistency. This module never manufactures those
outputs. Production retains all supplied signed samples and forty epochs;
only a separately declared DEBUG invocation may request a successful-update
bound. Every optimizer update has an atomic exact restart state, including the
detached bank AND the lazily fitted teacher cluster plans.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import dataclasses
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import time
import uuid

FORMAT='crossed_native_C_training_state_v1'
WALL_FORMAT='crossed_active_epoch_wall_v1'


class ActiveEpochWall:
    """Continuous monotonic epoch wall, rebased on resume to exclude downtime.

    Phase durations describe operations and may overlap for asynchronous
    writers. They never determine the elapsed wall or sum into another wall.
    Snapshots contain no monotonic timestamps tied to the previous process.
    """
    def __init__(self,epoch,saved=None,*,clock=None):
        if type(epoch) is not int or not 1<=epoch<=40:raise ValueError('Actual active epoch1..40 required')
        self.epoch=epoch;self._clock=time.perf_counter if clock is None else clock
        self._base=0.;self._phases={};self._running={}
        if saved is not None:
            if (not isinstance(saved,dict) or saved.get('format')!=WALL_FORMAT or saved.get('epoch')!=epoch
                    or not isinstance(saved.get('phase_wall_seconds'),dict)):
                raise ValueError('Another epoch or incomplete elapsed wall cursor')
            values=[saved.get('elapsed_seconds'),*saved['phase_wall_seconds'].values()]
            if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
                raise ValueError('Elapsed wall and phases require finite nonnegative measurements')
            if any(not isinstance(key,str) or not key for key in saved['phase_wall_seconds']):
                raise ValueError('Named measured wall phases required')
            self._base=float(saved['elapsed_seconds']);self._phases=copy.deepcopy(saved['phase_wall_seconds'])
        self._anchor=self._clock()

    def elapsed(self):
        value=self._base+self._clock()-self._anchor
        if not math.isfinite(value) or value<self._base:raise ValueError('Monotonic epoch wall moved backwards')
        return value

    def begin(self,name):
        if not isinstance(name,str) or not name:raise ValueError('Named measured wall phase required')
        token=uuid.uuid4().hex;self._running[token]=(name,self._clock());return token

    def finish(self,token):
        name,began=self._running.pop(token);duration=self._clock()-began
        if not math.isfinite(duration) or duration<0:raise ValueError('Monotonic wall phase moved backwards')
        self._phases[name]=self._phases.get(name,0.)+duration;return duration

    def record_duration(self,name,duration):
        """Attach an already measured iterator wait without changing wall."""
        if (not isinstance(name,str) or not name or type(duration) not in (int,float)
                or not math.isfinite(duration) or duration<0):
            raise ValueError('Named finite nonnegative measured phase duration required')
        self._phases[name]=self._phases.get(name,0.)+duration

    @contextmanager
    def phase(self,name):
        token=self.begin(name)
        try:yield
        finally:self.finish(token)

    def snapshot(self,scope='through_current_completed_operations'):
        now=self._clock();elapsed=self._base+now-self._anchor
        if not math.isfinite(elapsed) or elapsed<self._base:raise ValueError('Monotonic epoch wall moved backwards')
        phases=copy.deepcopy(self._phases)
        for name,began in self._running.values():phases[name]=phases.get(name,0.)+now-began
        return dict(format=WALL_FORMAT,epoch=self.epoch,elapsed_seconds=elapsed,
            phase_wall_seconds=phases,phase_durations_may_overlap=True,scope=scope,
            stopped_process_downtime_included=False)

    def history_fields(self,scope='through_epoch_end_checkpoint_and_timing_journal_close'):
        value=self.snapshot(scope)
        return dict(epoch_wall_seconds=value['elapsed_seconds'],
            phase_wall_seconds=value['phase_wall_seconds'],epoch_wall_timing=value)


def _wall_receipt_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def restore_epoch_wall_timing(saved,output):
    """Recover the bound atomic-write tail; never count restart downtime.

    A crash before a complete timing receipt leaves an explicit pre-write
    lower bound. The receipt's own final write/close is outside its snapshot;
    live epoch history and the next checkpoint include that writer wait.
    """
    cursor=copy.deepcopy(saved['cursor']);reference=saved.get('checkpoint_wall_receipt')
    cursor['wall_timing_resume_scope']='cursor_before_atomic_checkpoint_write'
    if not isinstance(reference,dict):return cursor
    name=reference.get('journal_file')
    if (not isinstance(name,str) or not re.fullmatch(r'c_checkpoint_timings_[0-9a-f]{32}\.jsonl',name)
            or not isinstance(reference.get('checkpoint_id'),str)):
        raise ValueError('Checkpoint wall receipt identity/path changed')
    path=Path(output)/name
    if not path.exists():return cursor
    if path.is_symlink():raise ValueError('Checkpoint wall timing receipt cannot be a symlink')
    for line in path.read_text(encoding='utf8').splitlines(keepends=True):
        if not line.endswith('\n'):break  # Explicit incomplete receipt; use the saved lower bound.
        row=json.loads(line)
        if row.get('checkpoint_id')!=reference['checkpoint_id']:continue
        if row.get('format')!='crossed_C_checkpoint_wall_receipt_v1':
            raise ValueError('Checkpoint wall timing receipt format changed')
        digest=row.get('content_sha256')
        if _wall_receipt_digest({k:v for k,v in row.items() if k!='content_sha256'})!=digest:
            raise ValueError('Checkpoint wall timing receipt bytes changed')
        if (row.get('run_identity_sha256')!=saved['run_identity_sha256']
                or row.get('cursor_epoch')!=cursor['epoch'] or row.get('updates')!=cursor['updates']):
            raise ValueError('Checkpoint wall timing receipt belongs to another scientific cursor')
        wall=row.get('active_epoch_wall')
        if wall is not None:
            expected_epoch=cursor['epoch']-int(bool(row.get('closing_epoch')))
            if wall.get('epoch')!=expected_epoch:
                raise ValueError('Checkpoint wall timing receipt names another active/completed epoch')
            # Admission validates clock fields without retaining a process anchor.
            ActiveEpochWall(wall['epoch'],wall)
            if row.get('closing_epoch'):
                history=[entry for entry in cursor['history'] if entry['epoch']==wall['epoch']]
                if len(history)!=1:raise ValueError('Closing wall receipt has another epoch history')
                history[0].update(epoch_wall_seconds=wall['elapsed_seconds'],
                    phase_wall_seconds=copy.deepcopy(wall['phase_wall_seconds']),epoch_wall_timing=wall)
            else:cursor['active_epoch_wall']=wall
        cursor['wall_timing_resume_scope']='through_atomic_checkpoint_write_before_its_timing_receipt_write'
        return cursor
    return cursor


class TrainingPauseRequested(Exception):
    """Cooperative pause; the latest exact state is preserved, not a failure."""


def own_selection_key(own,precision=8):
    """Preserved rounded MRR/top1/margin/ranking order; absent views omitted."""
    if type(precision) is not int or not 0<=precision<=12:
        raise ValueError('Original checkpoint precision must be in[0,12]')
    values=(own['MRR'],own['top1'],own['mean_margin'],-own['loss'])
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite own checkpoint selection')
    # The fifth position records the approved absent-consistency factor.
    # No consistency tensor/output is manufactured or added to the loss.
    return tuple(round(float(v),precision) for v in values)+(0.,)


def validate_signed_cohort(loader,runtime,split,configured_cases,*,debug):
    """Exact baseline-index materialized rows; declared split remains84/21."""
    cases=tuple(dict.fromkeys(s['case_id'] for s in loader.samples))
    if any(s['split']!=split for s in loader.samples) or not set(cases)<=set(configured_cases):
        raise ValueError('Loader rows disagree with original configured split')
    if debug:return dict(debug=True,actual_cases=list(cases),samples=len(loader),signed_production=False)
    prefix='validation' if split=='val' else split
    expected_cases=runtime.get('actual_signed_'+prefix+'_cases',runtime.get('actual_signed_'+split+'_cases'))
    rows=runtime.get('actual_signed_'+prefix+'_samples',runtime.get('actual_signed_'+split+'_samples'))
    if (not isinstance(expected_cases,(list,tuple)) or len(set(expected_cases))!=len(expected_cases)
            or set(cases)!=set(expected_cases) or not isinstance(rows,list) or not rows
            or any(not isinstance(runtime.get(key),str) or not re.fullmatch(r'[0-9a-f]{64}',runtime[key])
                for key in ('baseline_proof_sha256','cache_index_sha256'))):
        raise ValueError('Production requires exact signed baseline-index case/sample coverage and proof digests')
    expected={}
    for row in rows:
        if (not isinstance(row,dict) or row.get('case_id') not in expected_cases
                or type(row.get('sample_index')) is not int or row['sample_index']<0
                or row.get('path')!=f"{row['case_id']}__{row['sample_index']:03d}.pt"
                or not isinstance(row.get('sha256'),str) or not re.fullmatch(r'[0-9a-f]{64}',row['sha256'])):
            raise ValueError('Malformed signed original cache-index sample provenance')
        key=(row['case_id'],row['sample_index'])
        if key in expected:raise ValueError('Duplicate signed original sample')
        expected[key]=row
    actual={}
    for sample,source in zip(loader.samples,loader.sample_sources):
        key=(sample['case_id'],sample['sample_index'])
        if key in actual or source.get('kind')!='signed_original_cache' or key not in expected:
            raise ValueError('Production loader contains unsigned/extra/duplicate original sample')
        row=expected[key]
        if Path(source.get('path','')).name!=row['path'] or source.get('sha256')!=row['sha256']:
            raise ValueError('Actual source path/bytes differ from signed baseline cache index')
        actual[key]=source
    if set(actual)!=set(expected) or set(expected_cases)!={k[0] for k in expected}:
        raise ValueError('Production loader omitted a signed materialized original sample/case')
    return dict(debug=False,signed_production=True,samples=len(actual),actual_cases=list(cases),
        configured_cases=len(configured_cases),
        configured_but_not_materialized_cases=[case for case in configured_cases if case not in cases],
        baseline_proof_sha256=runtime['baseline_proof_sha256'],cache_index_sha256=runtime['cache_index_sha256'])


def runtime_contract(config,*,debug,debug_updates,train_samples,val_samples):
    runtime=config.get('transition_runtime')
    if not isinstance(runtime,dict): raise ValueError('Explicit measured transition_runtime required')
    keys=('physical_batch_size','support_batch_size','validation_batch_size','workers','device')
    if any(k not in runtime for k in keys): raise ValueError('Measured query/support/validation batches, workers and CUDA device required')
    for key in keys[:3]:
        if type(runtime[key]) is not int or runtime[key]<1: raise ValueError('Explicit measured positive physical sample batches required')
    if type(runtime['workers']) is not int or runtime['workers']<2 or not str(runtime['device']).startswith('cuda'):
        raise ValueError('Parallel workers>=2 and actual CUDA required; no CPU fallback')
    training=config['training']
    if (type(debug) is not bool or config['seed']!=42 or training['epochs']!=40
            or training['gradient_accumulation_steps']!=1 or training['target_effective_batch_size'] is not None
            or config['cache']['total_candidates']!=8 or config['cache']['candidate_pool_size']!=128
            or training['fixed_validation_epoch']!=29 or train_samples<1 or val_samples<1):
        raise ValueError('Original full40/eight/pool128/physical-update/validation29 contract required')
    if debug:
        if type(debug_updates) is not int or debug_updates<1 or runtime.get('debug_curriculum_epoch')!=29:
            raise ValueError('DEBUG needs explicit successful-update bound and curriculum29')
    elif debug_updates is not None:
        raise ValueError('Production cannot contain a DEBUG update bound')
    return dict(debug=debug,successful_DEBUG_updates=debug_updates,epochs=40,scheduler_T_max=40,
        physical_sample_batch=runtime['physical_batch_size'],physical_candidate_batch=runtime['physical_batch_size']*8,
        effective_sample_batch=runtime['physical_batch_size'],gradient_accumulation_steps=1,
        support_sample_batch=runtime['support_batch_size'],validation_sample_batch=runtime['validation_batch_size'],
        workers=runtime['workers'],device=str(runtime['device']),training_samples=train_samples,validation_samples=val_samples,
        input_block_differences=['raw native variable organ-only crop','native LocalCNN fused128',
            'comparison transforms/corruptions not applied','six sampled-view consistency removed'],
        original_ranking_objective_preserved=True,common_B_upper_preserved=True,
        native_support_single_view=True,original_support_two_views=False,checkpoint_every_optimizer_update=True)


def _tree(value,device):
    import torch
    if isinstance(value,torch.Tensor): return value.detach().to(device).clone()
    if isinstance(value,dict): return {k:_tree(v,device) for k,v in value.items()}
    if isinstance(value,list): return [_tree(v,device) for v in value]
    if isinstance(value,tuple): return tuple(_tree(v,device) for v in value)
    return copy.deepcopy(value)


def _digest(value):
    import torch
    h=hashlib.sha256()
    def add(v):
        if isinstance(v,torch.Tensor):
            v=v.detach().cpu().contiguous();h.update(str((str(v.dtype),tuple(v.shape))).encode())
            h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=repr): add(k);add(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(str(len(v)).encode())
            for x in v: add(x)
        else: h.update(json.dumps(v,sort_keys=True,allow_nan=False).encode())
        h.update(b'\0')
    add(value);return h.hexdigest()


def support_runtime_snapshot(net,bank):
    """Plans are scientific state: refitting them after resume is not exact."""
    upper=net.upper
    value=dict(bank=_tree(bank,'cpu'),teacher_plans=_tree(upper._plans,'cpu'),
        episodes=_tree(upper._episodes,'cpu'),seen_generations=sorted(upper._seen_generations))
    value['content_sha256']=_digest(value)
    return value


def restore_support_runtime(net,snapshot,device):
    import torch
    expected=snapshot.get('content_sha256')
    if _digest({k:v for k,v in snapshot.items() if k!='content_sha256'})!=expected:
        raise ValueError('Checkpoint detached support/teacher state digest changed')
    if snapshot['bank'] is None:
        if snapshot['teacher_plans'] or snapshot['episodes']:
            raise ValueError('An uninitialized bank cannot contain teacher or exclusion state')
        return None
    bank=_tree(snapshot['bank'],device)
    net.upper.bind_support(bank)  # All full8/owner/class/single-view admission checks run.
    for key,saved in snapshot['episodes'].items():
        actual=net.upper._episode(key)
        if _digest(actual)!=_digest(saved): raise ValueError('Saved exclusion episode no longer matches admitted support ownership')
    plans=_tree(snapshot['teacher_plans'],device)
    for key,plan in plans.items():
        episode=net.upper._episode(key)
        if (not torch.equal(plan['owners'],episode['owners']) or not torch.equal(plan['classes'],episode['classes'])
                or any(t.requires_grad for t in plan.values() if isinstance(t,torch.Tensor))):
            raise ValueError('Saved detached teacher plan has another query exclusion or class binding')
    net.upper._plans=plans
    net.upper._seen_generations.update(snapshot['seen_generations'])
    net._last_C_generation=bank['generation']
    return bank


def _check_budget(budget):
    if callable(budget): budget()
    elif budget is not None and callable(getattr(budget,'check',None)): budget.check()
    else: raise ValueError('An explicit active resource budget is required')


def _append(path,value):
    with path.open('a',encoding='utf8') as stream: stream.write(json.dumps(value,allow_nan=False)+'\n')


def _gradients(net,groups):
    import torch
    present=[p.grad for p in net.parameters() if p.grad is not None]
    names=[];norms=[]
    for name,module in groups.items():
        values=[p.grad.detach().double().square().sum() for p in module.parameters() if p.requires_grad and p.grad is not None]
        if not values: raise ValueError('Ranking loss did not reach '+name)
        names.append(name);norms.append(torch.stack(values).sum().sqrt())
    if not present: raise ValueError('No actual trainable gradients')
    packed=torch.stack([torch.isfinite(g).all() for g in present]+norms).cpu().tolist()
    raw=dict(zip(names,packed[len(present):]))
    return dict(finite=all(packed[:len(present)]),
        norms={k:v if math.isfinite(v) else str(v) for k,v in raw.items()},
        zero_groups=[k for k,v in raw.items() if v==0],nonfinite_groups=[k for k,v in raw.items() if not math.isfinite(v)])


def _parameter_changes(groups,before):
    import torch
    names=[];values=[]
    for name,module in groups.items():
        params=[p for p in module.parameters() if p.requires_grad]
        delta=torch.stack([(p.detach().double()-before[id(p)].double()).square().sum() for p in params]).sum().sqrt()
        changed=torch.stack([torch.ne(p.detach(),before[id(p)]).any().double() for p in params]).sum()
        names.append(name);values.append(torch.stack([delta,changed]))
    return {name:dict(L2_change=row[0],changed_parameter_tensors=int(row[1]))
        for name,row in zip(names,torch.stack(values).cpu().tolist())}


def build_native_support_bank(net,loader,*,batch_size,device,use_amp,manifest_sha256,
                              generation,epoch,training_case_ids,validation_case_ids,budget,debug):
    """All supplied signed training candidates, genuine SINGLE native view."""
    import torch
    from hiercp.tensor import capture_rng_state,restore_rng_state
    from .transition_model import C_SUPPORT
    if type(batch_size) is not int or batch_size<1: raise ValueError('Measured explicit support sample batch required')
    cases=tuple(dict.fromkeys(s['case_id'] for s in loader.samples))
    if len(cases)<3 or any(s['split']!='train' for s in loader.samples):
        raise ValueError('At least three actual training-only support patients required')
    modes=[(m,m.training) for m in net.modules()];rng=capture_rng_state();values=[]
    try:
        net.eval()
        with torch.no_grad(),torch.autocast('cuda',enabled=use_amp):
            for start in range(0,len(loader),batch_size):
                _check_budget(budget);ids=list(range(start,min(start+batch_size,len(loader))))
                batch=loader.get(ids).to(device);value=net.local(batch).detach().float()
                if value.shape!=(len(ids)*8,128) or not bool(torch.isfinite(value).all()):
                    raise ValueError('Native full training support embedding invalid')
                values.append(value);del batch
        samples=tuple((r.get('path') or f"native_C:{s['case_id']}:{s['sample_index']}")
            for r,s in zip(loader.sample_sources,loader.samples))
        if len(set(samples))!=len(samples): raise ValueError('Original support sample IDs must be unique')
        owner_ids={c:i for i,c in enumerate(cases)}
        return dict(embeddings=torch.cat(values),
            owners=torch.tensor([owner_ids[s['case_id']] for s in loader.samples],device=device,dtype=torch.long).repeat_interleave(8),
            classes=torch.tensor([1,0,0,0,0,0,0,0],device=device,dtype=torch.long).repeat(len(loader)),
            patient_case_ids=cases,sample_ids=tuple(s for s in samples for _ in range(8)),
            candidate_indices=tuple(range(8))*len(loader),expected_samples=dict(zip(samples,loader.case_ids(range(len(loader))))),
            training_case_ids=tuple(training_case_ids),validation_case_ids=tuple(validation_case_ids),
            manifest_sha256=manifest_sha256,generation=generation,epoch=epoch,fixed_view_epoch=0,
            debug=debug,full_signed_training_cache=not debug,support_policy=C_SUPPORT,
            native_single_view=True,original_two_views=False)
    finally:
        for module,mode in modes:module.training=mode
        restore_rng_state(rng)


def calibrate_c(net,train_loader,config,candidates,*,budget,debug=False):
    """Measure complete cloned updates; retain fresh production weights/RNG.

    One successful warmup and two successful timed updates per candidate.
    AMP skipped attempts are recorded explicitly. Every candidate retains the
    complete native architecture, full8 objective and same full support cohort.
    A resource failure is a measured rejected candidate, never a smaller-model
    fallback. Returns the selected physical SAMPLE batch and every report.
    """
    import torch
    from hiercp.tensor import capture_rng_state,restore_rng_state
    device=next(net.parameters()).device
    if (device.type!='cuda' or not torch.cuda.is_available() or getattr(net,'arm',None)!='C'
            or type(debug) is not bool or train_loader.debug!=debug
            or not candidates or any(type(n) is not int or not 1<=n<=len(train_loader) for n in candidates)
            or len(set(candidates))!=len(candidates)):
        raise ValueError('Actual CUDA C model and explicit distinct measured batch candidates required')
    if net.upper._memory is not None:raise ValueError('Calibrate a fresh production model before binding support')
    rng=capture_rng_state();before=_digest(net.state_dict());reports=[]
    runtime=config.get('transition_runtime',{})
    train_cases=tuple(dict.fromkeys(s['case_id'] for s in train_loader.samples))
    configured_train=tuple(runtime.get('training_case_ids',train_cases))
    configured_val=tuple(runtime.get('validation_case_ids',()))
    if not configured_val:raise ValueError('Calibration requires explicit original validation allowlist')
    support_batch=runtime.get('support_batch_size')
    if type(support_batch) is not int or support_batch<1:
        raise ValueError('Explicit measured support sample batch required for calibration')
    identity=_digest(dict(config=config,sample_sources=train_loader.sample_sources,records=train_loader.records))
    try:
        memory=build_native_support_bank(net,train_loader,batch_size=support_batch,device=device,
            use_amp=config['training']['amp'],manifest_sha256=identity,generation='calibration_full_native_single_view',
            epoch=0,training_case_ids=configured_train,validation_case_ids=configured_val,budget=budget,debug=debug)
        for candidate in candidates:
            _check_budget(budget);clone=None;batch=None;optimizer=None;scaler=None
            embedding=scores=loss=terms=params=groups=gradient=None
            trial=dict(physical_samples=candidate,physical_candidate_rows=candidate*8,
                full_training_bank_samples=len(train_loader),full_training_bank_rows=len(memory['embeddings']),
                warmup_successful_updates=1,timed_successful_updates=2,attempts=[],timed_seconds=[],status='RUNNING')
            try:
                active_budget=getattr(net.local,'resource_budget',None)
                net.local.resource_budget=None
                try:clone=copy.deepcopy(net)
                finally:net.local.resource_budget=active_budget
                clone.local.resource_budget=active_budget
                clone.upper.bind_support(memory);clone._last_C_generation=memory['generation'];clone.train()
                params=[p for p in clone.parameters() if p.requires_grad]
                optimizer=torch.optim.AdamW(params,lr=config['training']['lr'],weight_decay=config['training']['weight_decay'],
                    fused=config['training']['fused_optimizer'])
                scaler=torch.amp.GradScaler('cuda',enabled=config['training']['amp'])
                ids=list(range(candidate));began=time.perf_counter();batch=train_loader.get(ids).to(device)
                trial['input_prepare_seconds']=time.perf_counter()-began
                groups=dict(native_CNN=clone.local.cnn,native_readout=clone.local.project,native_fusion=clone.local.fuse,
                    upper_L1=clone.core.l1,upper_L2=clone.core.l2,upper_L2_updates=clone.core.l2_updates)
                successful=0;peak_allocated=0;peak_reserved=0
                while successful<3:
                    _check_budget(budget);optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device);began=time.perf_counter()
                    with torch.autocast('cuda',enabled=config['training']['amp']):
                        embedding=clone.local(batch);scores=clone.score_old(embedding,train_loader.case_ids(ids),train_loader.counts(ids),memory)
                        loss,terms=clone.original_ranking_loss(scores,train_loader.difficulty_list(ids,device=device),epoch=29,config=config)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Calibration original full ranking loss nonfinite')
                    scale=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer)
                    gradient=_gradients(clone,groups)
                    if not gradient['finite']:
                        if not scaler.is_enabled():raise FloatingPointError('Calibration nonfinite gradients with AMP disabled')
                        scaler.step(optimizer);scaler.update()
                        if not scaler.get_scale()<scale:raise ValueError('AMP overflow failed to skip calibration update')
                        trial['attempts'].append(dict(status='AMP_OVERFLOW_SKIPPED',scale_before=scale,scale_after=scaler.get_scale()))
                    else:
                        torch.nn.utils.clip_grad_norm_(params,config['training']['grad_clip'],error_if_nonfinite=True)
                        step_before=optimizer.state.get(clone.core.label_seed,{}).get('step',0)
                        step_before=int(step_before.item()) if isinstance(step_before,torch.Tensor) else int(step_before)
                        scaler.step(optimizer);scaler.update();torch.cuda.synchronize(device)
                        step_after=optimizer.state[clone.core.label_seed]['step'];step_after=int(step_after.item())
                        if step_after!=step_before+1:raise ValueError('Calibration did not actually update clone parameters')
                        successful+=1;elapsed=time.perf_counter()-began
                        trial['attempts'].append(dict(status='OPTIMIZER_UPDATED',successful_update=successful,
                            loss=float(loss),gradient=gradient,seconds=elapsed))
                        if successful>1:trial['timed_seconds'].append(elapsed)
                    peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated(device))
                    peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved(device))
                    embedding=scores=loss=terms=None
                trial.update(status='COMPLETE',peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,
                    sample_queries_per_second=candidate*2/sum(trial['timed_seconds']),
                    candidate_queries_per_second=candidate*8*2/sum(trial['timed_seconds']),
                    cloned_model_actually_updated=True)
            except (torch.OutOfMemoryError,MemoryError) as exc:
                trial.update(status='RESOURCE_REJECTED',error=f'{type(exc).__name__}: {exc}')
            finally:
                del clone,batch,optimizer,scaler,embedding,scores,loss,terms,params,groups,gradient
                torch.cuda.empty_cache()
            reports.append(trial)
        complete=[r for r in reports if r['status']=='COMPLETE']
        if not complete:raise RuntimeError('No explicitly measured full-architecture candidate fits; no model/batch fallback')
        selected=max(complete,key=lambda r:(r['candidate_queries_per_second'],r['physical_samples']))['physical_samples']
        for row in reports:row['selected']=row['physical_samples']==selected
    finally:
        restore_rng_state(rng)
    if _digest(net.state_dict())!=before or net.upper._memory is not None:
        raise ValueError('Calibration altered production neural state or support binding')
    for row in reports:row['production_initial_state_preserved']=True;row['caller_RNG_restored']=True
    return selected,reports


def _source_identity(net):
    from .transition_native_local import _sha
    root=Path(__file__).resolve().parents[1]
    paths=[root/'hiercp_v1x/transition_c_training.py',root/'hiercp_v1x/transition_native_local.py',
        root/'hiercp_v1x/transition_model.py',root/'hiercp_v1x/half_b_model.py',root/'hiercp_v222/model.py',
        root/'hiercp_v222/clustering.py',root/'l0_local_cnn/model.py',root/'l0_exploration/model.py',
        Path(inspect.getfile(net.original_ranking_loss))]
    from hiercp.loss import curriculum_ranking_loss
    paths.append(Path(inspect.getfile(curriculum_ranking_loss)))
    return {str(p.resolve()):_sha(p) for p in paths}


def _train_c_impl(net,train_loader,val_loader,config,*,output,budget,debug=False,debug_updates=None,common_evaluate=None,_pause_flag=None):
    """Full C arm with exact pause/resume and own/common evaluation callbacks.

    Runtime supplies resolved physical/support/validation sample batches and
    CUDA device. Resume is an explicit transition_runtime.resume_checkpoint.
    A pause is requested by writing output/pause_requested.json; safe stopping
    saves the next minibatch cursor. The caller constructs a FRESH same-arm
    model before exact resume. Common callback is called as
    common_evaluate(net, bank=bank, epoch=epoch, update=step, output=output).
    """
    import torch
    from hiercp.tensor import capture_rng_state,restore_rng_state,save_checkpoint_atomic
    from hiercp.loss import ranking_metrics
    from .transition_model import C_SUPPORT
    from .transition_native_local import native_transition_spec,_sha
    from l0_local_cnn.model import LocalCNN
    if (getattr(net,'arm',None)!='C' or not isinstance(net.local,LocalCNN)
            or getattr(net,'debug',None)!=debug or not callable(getattr(net,'score_old',None))):
        raise ValueError('Fresh actual native LocalCNN crossed C model required')
    contract=runtime_contract(config,debug=debug,debug_updates=debug_updates,
        train_samples=len(train_loader),val_samples=len(val_loader))
    runtime=config['transition_runtime'];device=torch.device(runtime['device'])
    if device.type=='cuda' and device.index is None:device=torch.device('cuda',torch.cuda.current_device())
    if device.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; no CPU training fallback')
    if not debug and not callable(common_evaluate):
        raise ValueError('Production requires the full common evaluation callback')
    train_cases=tuple(dict.fromkeys(s['case_id'] for s in train_loader.samples))
    val_cases=tuple(dict.fromkeys(s['case_id'] for s in val_loader.samples))
    if len(train_cases)<3 or set(train_cases)&set(val_cases):
        raise ValueError('Full native support needs>=3 training patients and disjoint held-out patients')
    configured_train=tuple(runtime.get('training_case_ids',train_cases))
    configured_val=tuple(runtime.get('validation_case_ids',val_cases))
    if (not set(train_cases)<=set(configured_train) or not set(val_cases)<=set(configured_val)
            or set(configured_train)&set(configured_val)
            or not debug and (len(configured_train)!=84 or len(configured_val)!=21)):
        raise ValueError('Original disjoint84/21 configuration and complete declared loader support required')
    cohorts={split:validate_signed_cohort(loader,runtime,split,cases,debug=debug)
        for split,loader,cases in (('train',train_loader,configured_train),('val',val_loader,configured_val))}
    if train_loader.debug!=debug or val_loader.debug!=debug:
        raise ValueError('Loader/model DEBUG profiles must agree')
    net.to(device)
    params=[p for p in net.parameters() if p.requires_grad]
    if not params or any(p.device!=device for p in params): raise ValueError('All actual trainable parameters must be on configured CUDA')
    optimizer=torch.optim.AdamW(params,lr=config['training']['lr'],weight_decay=config['training']['weight_decay'],
        fused=config['training']['fused_optimizer'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=40)
    scaler=torch.amp.GradScaler('cuda',enabled=config['training']['amp'])
    shuffle=torch.Generator().manual_seed(42+2003)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    resume=runtime.get('resume_checkpoint');sources=_source_identity(net)
    runtime_math={k:v for k,v in runtime.items() if k not in ('resume_checkpoint','pause_file')}
    provenance=dict(format=FORMAT,arm='C',configuration={**config,'transition_runtime':runtime_math},
        execution=contract,input_spec=native_transition_spec(),source_sha256=sources,
        signed_materialized_cohorts=cohorts,
        train_loader=train_loader.report(),validation_loader=val_loader.report(),
        upper_initial_sha256=net.initial_upper_sha256)
    # Cache hit counters and prepared row annotations are execution diagnostics,
    # not a resume identity. Bind small source/candidate/difficulty metadata.
    for name,loader in (('train_loader',train_loader),('validation_loader',val_loader)):
        for key in ('cache_hits','cache_misses','resident_raw_bytes','resident_crop_bytes'):
            provenance[name].pop(key,None)
        for row in provenance[name]['records']: row.pop('source_full_mask_voxels',None)
    provenance=json.loads(json.dumps(provenance,allow_nan=False))
    identity=_digest(provenance);provenance['identity_sha256']=identity
    contract_path=output/'c_training_contract.json'
    if contract_path.exists():
        if not resume or json.loads(contract_path.read_text(encoding='utf8'))!=provenance:
            raise FileExistsError('Existing C training results require their exact own resume checkpoint')
    else:
        if resume: raise ValueError('Exact resume needs original C training contract')
        with contract_path.open('x',encoding='utf8') as stream: json.dump(provenance,stream,indent=2,allow_nan=False)
    invocation=uuid.uuid4().hex;journal=output/f'c_updates_{invocation}.jsonl'
    timing_journal=output/f'c_checkpoint_timings_{invocation}.jsonl'
    state=dict(epoch=1,position=0,order=None,phase='training',updates=0,attempts=0,
        history=[],initial_evaluation=None,best_own=None,connected=[],active_epoch_wall=None)
    bank=None;started=time.perf_counter();expected={n for n,p in net.named_parameters() if p.requires_grad}
    epoch_clock=None
    groups=dict(native_CNN=net.local.cnn,native_readout=net.local.project,native_fusion=net.local.fuse,
        upper_L1=net.core.l1,upper_L2=net.core.l2,upper_L2_updates=net.core.l2_updates,upper_total=net.upper)
    def checkpoint(status='RUNNING',*,closing_epoch_row=None):
        began=time.perf_counter();token=epoch_clock.begin('checkpoint_save') if epoch_clock else None
        checkpoint_id=uuid.uuid4().hex
        payload=dict(format=FORMAT,arm='C',run_identity_sha256=identity,debug=debug,status=status,
            model_state_dict=net.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),scaler_state_dict=scaler.state_dict(),
            rng_state=capture_rng_state(),shuffle_generator_state=shuffle.get_state(),
            support_runtime=support_runtime_snapshot(net,bank),
            checkpoint_every_optimizer_update=True,source_sha256=sources,
            checkpoint_wall_receipt=dict(checkpoint_id=checkpoint_id,journal_file=timing_journal.name))
        if epoch_clock:
            if closing_epoch_row is None:
                state['active_epoch_wall']=epoch_clock.snapshot('through_cursor_snapshot_before_atomic_checkpoint_write')
            else:
                closing_epoch_row.update(epoch_clock.history_fields('through_cursor_snapshot_before_epoch_end_checkpoint_write'))
                state['active_epoch_wall']=None
        payload['cursor']=copy.deepcopy(state)
        snapshot_elapsed=(payload['cursor']['active_epoch_wall']['elapsed_seconds'] if closing_epoch_row is None
            else closing_epoch_row['epoch_wall_seconds']) if epoch_clock else None
        save_checkpoint_atomic(payload,output/'checkpoint_latest.pt')
        receipt=dict(format='crossed_C_checkpoint_wall_receipt_v1',checkpoint_id=checkpoint_id,
            run_identity_sha256=identity,cursor_epoch=state['epoch'],updates=state['updates'],
            closing_epoch=closing_epoch_row is not None,checkpoint_wall_seconds=time.perf_counter()-began,
            checkpoint_write_tail_seconds=epoch_clock.elapsed()-snapshot_elapsed if epoch_clock else None,
            active_epoch_wall=epoch_clock.snapshot('through_atomic_checkpoint_write_before_its_timing_receipt_write') if epoch_clock else None,
            receipt_self_write_tail_in_this_row=False)
        receipt['content_sha256']=_wall_receipt_digest(receipt)
        _append(timing_journal,receipt)  # Close/writer wait belongs to the live epoch clock.
        if epoch_clock:
            epoch_clock.finish(token)
            if closing_epoch_row is not None:closing_epoch_row.update(epoch_clock.history_fields())
    if resume:
        path=Path(resume).resolve(strict=True)
        if path.parent!=output: raise ValueError('Resume must be the C state in this exact arm directory')
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=False)
        if (saved.get('format')!=FORMAT or saved.get('arm')!='C' or saved.get('debug')!=debug
                or saved.get('run_identity_sha256')!=identity or saved.get('source_sha256')!=sources):
            raise ValueError('Another arm/configuration/source cannot be called exact C resume')
        net.load_state_dict(saved['model_state_dict'],strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict']);scheduler.load_state_dict(saved['scheduler_state_dict'])
        scaler.load_state_dict(saved['scaler_state_dict']);shuffle.set_state(saved['shuffle_generator_state'])
        bank=restore_support_runtime(net,saved['support_runtime'],device);state=restore_epoch_wall_timing(saved,output)
        restore_rng_state(saved['rng_state'])
        if state.get('active_epoch_wall') is not None:
            epoch_clock=ActiveEpochWall(state['epoch'],state['active_epoch_wall'])
    def _refresh(epoch):
        _check_budget(budget);modes=[(m,m.training) for m in net.modules()];rng=capture_rng_state()
        began=time.perf_counter();values=[]
        try:
            net.eval()
            with torch.no_grad(),torch.autocast('cuda',enabled=config['training']['amp']):
                for start in range(0,len(train_loader),runtime['support_batch_size']):
                    if pause_requested():raise TrainingPauseRequested('Pause at native support batch boundary')
                    _check_budget(budget);ids=list(range(start,min(start+runtime['support_batch_size'],len(train_loader))))
                    batch=train_loader.get(ids).to(device)
                    embedding=net.local(batch).detach().float()
                    if embedding.shape!=(8*len(ids),128) or not bool(torch.isfinite(embedding).all()):
                        raise ValueError('Native full training support embedding invalid')
                    values.append(embedding);del batch
            sample_ids=tuple((r.get('path') or f"native_C:{s['case_id']}:{s['sample_index']}")
                for r,s in zip(train_loader.sample_sources,train_loader.samples))
            if len(set(sample_ids))!=len(sample_ids): raise ValueError('Support source sample IDs are not unique')
            owners_by_case={c:i for i,c in enumerate(train_cases)}
            owners=torch.tensor([owners_by_case[s['case_id']] for s in train_loader.samples],device=device).repeat_interleave(8)
            memory=dict(embeddings=torch.cat(values),owners=owners,
                classes=torch.tensor([1,0,0,0,0,0,0,0],device=device,dtype=torch.long).repeat(len(train_loader)),
                patient_case_ids=train_cases,sample_ids=tuple(s for s in sample_ids for _ in range(8)),
                candidate_indices=tuple(range(8))*len(train_loader),expected_samples=dict(zip(sample_ids,train_loader.case_ids(range(len(train_loader))))),
                training_case_ids=configured_train,validation_case_ids=configured_val,
                manifest_sha256=identity,generation=f'{identity}:epoch{epoch}:update{state["updates"]}:refresh{uuid.uuid4().hex}',epoch=epoch,
                fixed_view_epoch=0,debug=debug,full_signed_training_cache=not debug,support_policy=C_SUPPORT,
                native_single_view=True,original_two_views=False)
            net.upper.bind_support(memory);net._last_C_generation=memory['generation']
            _append(output/f'c_support_refreshes_{invocation}.jsonl',dict(epoch=epoch,update=state['updates'],
                generation=memory['generation'],samples=len(train_loader),patients=len(train_cases),
                embeddings=len(memory['embeddings']),native_single_view=True,original_two_views=False,
                seconds=time.perf_counter()-began))
            return memory
        finally:
            for m,mode in modes:m.training=mode
            restore_rng_state(rng)
    def refresh(epoch):
        token=epoch_clock.begin('support_refresh') if epoch_clock else None
        try:return _refresh(epoch)
        finally:
            if epoch_clock:epoch_clock.finish(token)
    def evaluate(epoch):
        own_began=time.perf_counter();own_token=epoch_clock.begin('own_evaluation') if epoch_clock else None
        common_began=None;common_token=None;own_wall=None;result=None
        _check_budget(budget);modes=[(m,m.training) for m in net.modules()];rng=capture_rng_state()
        rows=[];loss_sum=0.;terms_sum={};n=0
        try:
            net.eval()
            with torch.no_grad(),torch.autocast('cuda',enabled=config['training']['amp']):
                for start in range(0,len(val_loader),runtime['validation_batch_size']):
                    if pause_requested():raise TrainingPauseRequested('Pause at own validation batch boundary')
                    _check_budget(budget);ids=list(range(start,min(start+runtime['validation_batch_size'],len(val_loader))))
                    batch=val_loader.get(ids).to(device);embedding=net.local(batch)
                    scores=net.score_old(embedding,val_loader.case_ids(ids),val_loader.counts(ids),bank)
                    loss,terms=net.original_ranking_loss(scores,val_loader.difficulty_list(ids,device=device),epoch=29,config=config)
                    if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite actual own ranking evaluation')
                    score_cpu=torch.stack(scores).detach().float().cpu()
                    if score_cpu.shape!=(len(ids),8) or not bool(torch.isfinite(score_cpu).all()): raise ValueError('Missing/nonfinite own eight scores')
                    rows.extend(score_cpu.unbind());loss_sum+=float(loss)*len(ids);n+=len(ids)
                    for k,v in terms.items():terms_sum[k]=terms_sum.get(k,0.)+float(v)*len(ids)
                    del batch,embedding,scores,loss,terms
            if n!=len(val_loader): raise ValueError('Full own validation coverage missing')
            top1,mrr=ranking_metrics(rows)
            own=dict(samples=n,top1=top1,MRR=mrr,loss=loss_sum/n,
                loss_terms={k:v/n for k,v in terms_sum.items()},scores=[r.tolist() for r in rows],
                mean_margin=sum(float(r[0]-r[1:].max()) for r in rows)/n,loss_epoch=29,
                original_consistency_added=False)
            own_wall=time.perf_counter()-own_began
            if epoch_clock:epoch_clock.finish(own_token);own_token=None
            common_began=time.perf_counter()
            common_token=epoch_clock.begin('common_evaluation' if common_evaluate else 'evaluation_finalize') if epoch_clock else None
            if pause_requested():raise TrainingPauseRequested('Pause before common evaluation')
            common=common_evaluate(net,bank=bank,epoch=epoch,update=state['updates'],output=output) if common_evaluate else None
            if common is not None: json.dumps(common,allow_nan=False)
            result=dict(epoch=epoch,update=state['updates'],own=own,common=common,common_evaluation_invoked=common_evaluate is not None)
            return result
        finally:
            for m,mode in modes:m.training=mode
            restore_rng_state(rng)
            if epoch_clock:
                if own_token is not None:epoch_clock.finish(own_token)
                if common_token is not None:epoch_clock.finish(common_token)
            if result is not None:
                result['own_evaluation_wall_seconds']=own_wall
                result['common_evaluation_wall_seconds']=time.perf_counter()-common_began if common_evaluate else None
    def pause_requested():
        if _pause_flag and _pause_flag['requested']:return True
        path=Path(runtime.get('pause_file',output/'pause_requested.json'))
        if not path.exists(): return False
        item=json.loads(path.read_text(encoding='utf8'))
        if item.get('action')!='pause': raise ValueError('Unknown C training control action')
        return True
    checkpoint()  # Untrained state is a valid restart point before full bank preparation.
    if pause_requested():raise TrainingPauseRequested('Pause before training support preparation')
    if bank is None: bank=refresh(0)
    if state['initial_evaluation'] is None:
        state['initial_evaluation']=evaluate(0);checkpoint()
    status='RUNNING'
    try:
        while state['epoch']<=40:
            _check_budget(budget)
            if pause_requested():status='PAUSED';checkpoint(status);break
            if debug and state['updates']>=debug_updates:status='DEBUG_COMPLETE';checkpoint(status);break
            epoch=state['epoch']
            if epoch_clock is None:epoch_clock=ActiveEpochWall(epoch)
            if state['order'] is None:state['order']=torch.randperm(len(train_loader),generator=shuffle).tolist()
            while state['phase']=='training' and state['position']<len(train_loader):
                _check_budget(budget)
                if pause_requested():status='PAUSED';checkpoint(status);break
                if debug and state['updates']>=debug_updates:status='DEBUG_COMPLETE';checkpoint(status);break
                ids=state['order'][state['position']:state['position']+runtime['physical_batch_size']]
                with epoch_clock.phase('train_loader'):
                    batch=train_loader.get(ids).to(device)
                net.train();optimizer.zero_grad(set_to_none=True)
                began=time.perf_counter();state['attempts']+=1;torch.cuda.reset_peak_memory_stats(device)
                update_token=epoch_clock.begin('optimizer_update')
                with torch.autocast('cuda',enabled=config['training']['amp']):
                    embedding=net.local(batch);scores=net.score_old(embedding,train_loader.case_ids(ids),train_loader.counts(ids),bank)
                    loss,terms=net.original_ranking_loss(scores,train_loader.difficulty_list(ids,device=device),
                        epoch=29 if debug else epoch,config=config)
                if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite actual complete original ranking loss')
                scale=scaler.get_scale();observed=net.core.label_seed
                old_step=optimizer.state.get(observed,{}).get('step',0)
                old_step=int(old_step.item()) if isinstance(old_step,torch.Tensor) else int(old_step)
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                gradient=_gradients(net,groups)
                if not gradient['finite']:
                    if not scaler.is_enabled():raise ValueError('Nonfinite native gradients with AMP disabled')
                    scaler.step(optimizer);scaler.update()
                    after=optimizer.state.get(observed,{}).get('step',0)
                    after=int(after.item()) if isinstance(after,torch.Tensor) else int(after)
                    if after!=old_step or not scaler.get_scale()<scale:raise ValueError('AMP overflow did not skip honestly')
                    epoch_clock.finish(update_token);write_token=epoch_clock.begin('update_journal_write')
                    _append(journal,dict(attempt=state['attempts'],status='AMP_OVERFLOW_SKIPPED',updates=state['updates'],
                        epoch=epoch,position=state['position'],gradient=gradient,scale_before=scale,scale_after=scaler.get_scale()))
                    epoch_clock.finish(write_token)
                    del batch,embedding,scores,loss,terms;checkpoint();continue
                clip=torch.nn.utils.clip_grad_norm_(params,config['training']['grad_clip'],error_if_nonfinite=True)
                before={id(p):p.detach().clone() for p in params}
                scaler.step(optimizer);scaler.update();torch.cuda.synchronize(device)
                actual=optimizer.state[observed]['step'];actual=int(actual.item()) if isinstance(actual,torch.Tensor) else int(actual)
                if actual!=old_step+1:raise ValueError('Finite attempt failed to complete one optimizer update')
                change=_parameter_changes(groups,before);del before
                connected=set(state['connected']);connected.update(n for n,p in net.named_parameters() if p.requires_grad and p.grad is not None)
                state['connected']=sorted(connected);state['updates']+=1;state['position']+=len(ids)
                epoch_clock.finish(update_token);write_token=epoch_clock.begin('update_journal_write')
                _append(journal,dict(attempt=state['attempts'],status='OPTIMIZER_UPDATED',update=state['updates'],epoch=epoch,
                    actual_curriculum_epoch=29 if debug else epoch,sample_indices=ids,physical_samples=len(ids),physical_candidate_rows=8*len(ids),
                    loss=float(loss),loss_terms={k:float(v) for k,v in terms.items()},original_consistency_added=False,
                    gradient=gradient,parameter_updates=change,gradient_before_clip=float(clip),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    elapsed_update_seconds=time.perf_counter()-began))
                epoch_clock.finish(write_token)
                del batch,embedding,scores,loss,terms,clip
                checkpoint()  # Includes the old in-epoch bank and unchanged teacher plan.
            if status in ('PAUSED','DEBUG_COMPLETE'):break
            if state['phase']=='training':
                bank=refresh(epoch);state['phase']='validation';checkpoint()
            row=evaluate(epoch);state['history'].append(row)
            metric=own_selection_key(row['own'],config['training'].get('checkpoint_metric_precision',8))
            if state['best_own'] is None or metric>tuple(state['best_own']['metric']):
                state['best_own']=dict(epoch=epoch,update=state['updates'],metric=list(metric))
                with epoch_clock.phase('best_checkpoint_save'):
                    save_checkpoint_atomic(dict(format='crossed_C_best_own_model_v1',run_identity_sha256=identity,
                        arm='C',debug=debug,state_dict=net.state_dict(),selection=state['best_own']),output/'checkpoint_best_own.pt')
            with epoch_clock.phase('epoch_finalization'):
                scheduler.step();state.update(epoch=epoch+1,position=0,order=None,phase='training')
            checkpoint(closing_epoch_row=row);epoch_clock=None
        if state['epoch']>40:status='COMPLETE';checkpoint(status)
        # A bound DEBUG stop can occur inside an epoch. Evaluate the current
        # model with a fresh bank, then preserve that exact resulting state.
        if status=='DEBUG_COMPLETE':
            bank=refresh(min(state['epoch'],40));state['final_DEBUG_evaluation']=evaluate(min(state['epoch'],40));checkpoint(status)
    except TrainingPauseRequested:
        status='PAUSED';checkpoint(status)
    except Exception:
        # Latest successful/AMP-skip checkpoint remains the exact restart point.
        _append(journal,dict(status='FAILED',epoch=state['epoch'],position=state['position'],updates=state['updates']))
        raise
    active_wall=epoch_clock.snapshot('through_safe_stop_checkpoint_and_timing_journal_close') if epoch_clock else None
    preserved=all(_sha(Path(p))==h for p,h in sources.items())
    missing=sorted(expected-set(state['connected']))
    if not preserved or state['updates'] and missing:raise ValueError('Source changed or trainable parameters disconnected')
    report=dict(format=FORMAT,arm='C',status=status,debug=debug,actual_CUDA=True,actual_raw_CT=True,
        full_training=status=='COMPLETE' and not debug,full_evaluation=False,quality_verified=False,production_ready=False,
        updates=state['updates'],backward_attempts=state['attempts'],completed_epochs=state['epoch']-1,
        input_spec=native_transition_spec(),execution=contract,run_identity_sha256=identity,
        initial_evaluation=state['initial_evaluation'],history=state['history'],best_own=state['best_own'],
        final_DEBUG_evaluation=state.get('final_DEBUG_evaluation'),common_evaluation_implemented=common_evaluate is not None,
        missing_gradients=missing,trainable_parameter_tensors=len(expected),source_preserved=preserved,
        total_wall_seconds=time.perf_counter()-started,checkpoint=str(output/'checkpoint_latest.pt'),
        total_wall_seconds_scope='current_invocation_only',
        completed_epoch_wall_seconds=sum(row['epoch_wall_seconds'] for row in state['history']),
        active_epoch_wall=active_wall,
        epoch_wall_includes=['train_loader','optimizer_update','support_refresh','own_evaluation',
            'common_evaluation','update_journal_write','checkpoint_save_and_writer_close','best_checkpoint_save'],
        epoch_wall_excludes=['stopped_process_downtime','initial_preparation_and_initial_evaluation',
            'separate_clone_calibration','root_final_selected_best_full_common_evaluation'],
        checkpoint_wall_timing_journal=str(timing_journal),
        checkpoint_wall_resume_scope=state.get('wall_timing_resume_scope'),
        last_timing_receipt_own_write_tail_excluded_from_its_saved_snapshot=True,
        exact_resume_includes_detached_teacher_plans=True,paused_at_next_minibatch_cursor=status=='PAUSED')
    if status=='COMPLETE' and state['history']:
        last=state['history'][-1]['common']
        report['full_evaluation']=not debug and isinstance(last,dict) and last.get('full_evaluation') is True
    with (output/f'c_training_report_{invocation}.json').open('x',encoding='utf8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    return report


def train_c(net,train_loader,val_loader,config,*,output,budget,debug=False,debug_updates=None,common_evaluate=None):
    """Scoped SIGINT requests a safe pause; never kills a shell or another process.

    The common evaluator may call net._transition_pause_requested() between
    complete query batches, then raise TrainingPauseRequested. The SIGINT
    handler records intent only; the active optimizer update/checkpoint finishes.
    The prior handler and any existing callback attribute are always restored.
    """
    import signal
    import threading
    flag={'requested':False};previous=signal.getsignal(signal.SIGINT)
    main=threading.current_thread() is threading.main_thread()
    old_callback=getattr(net,'_transition_pause_requested',None)
    def requested():
        if flag['requested']:return True
        path=Path(config.get('transition_runtime',{}).get('pause_file',Path(output)/'pause_requested.json'))
        if not path.exists():return False
        item=json.loads(path.read_text(encoding='utf8'))
        if item.get('action')!='pause':raise ValueError('Unknown C training control action')
        return True
    net._transition_pause_requested=requested
    def request(signum,frame):flag['requested']=True
    if main:signal.signal(signal.SIGINT,request)
    try:
        return _train_c_impl(net,train_loader,val_loader,config,output=output,budget=budget,debug=debug,
            debug_updates=debug_updates,common_evaluate=common_evaluate,_pause_flag=flag)
    except TrainingPauseRequested:
        # Initial bank/validation can pause before the main epoch loop. Its
        # atomically committed state is the restart point; no update is invented.
        import torch
        from hiercp.tensor import save_checkpoint_atomic
        path=Path(output)/'checkpoint_latest.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=False)
        saved['status']='PAUSED';save_checkpoint_atomic(saved,path)
        return dict(format=FORMAT,arm='C',status='PAUSED',debug=debug,actual_CUDA=True,
            full_training=False,full_evaluation=False,quality_verified=False,production_ready=False,
            updates=saved['cursor']['updates'],checkpoint=str(path.resolve()),
            paused_at_next_minibatch_cursor=True,exact_resume_includes_detached_teacher_plans=True)
    finally:
        if main:signal.signal(signal.SIGINT,previous)
        if old_callback is None:delattr(net,'_transition_pause_requested')
        else:net._transition_pause_requested=old_callback
