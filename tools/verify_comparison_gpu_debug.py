"""Actual full-model CUDA DEBUG for comparison execution-only GPU policies.

Reuse completed real-CT sample layouts; no CT/graph preprocessing is permitted.
The original physical2/train8/validation129 contract is unchanged. Measurements
on this selected GPU are not an A6000/server throughput or quality claim.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=ROOT/'work/v19_checked_DEBUG')
    p.add_argument('--completed-run', type=Path,
        default=ROOT/'work/runs/v1.9/sample-cache/final-20261008')
    p.add_argument('--inventory', type=Path, default=ROOT/'work/v22_cumulative_u16_DEBUG_20261006/inventory.json')
    p.add_argument('--bank', type=Path, default=ROOT/'work/v1x_real_CT_DEBUG_20261003_prepare/fixture.pt')
    p.add_argument('--cuda-gib', type=float,
        help='Explicit execution budget; otherwise retain the sealed fixture budget')
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--skip-runtime', action='store_true',
        help='Explicitly omit the separate original-engine two-update/pause/resume DEBUG check')
    return p.parse_args(argv)


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def tree_stats(root):
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(Path(root).rglob('*')) if p.is_file()}


def denied(*args, **kwargs):
    raise AssertionError('Completed real-CT layouts must not rerun raw/source/upper preparation')


def state_receipt(net, optimizer):
    from hiercp_v1x.u_bridge_training import capture_rng, digest
    from hiercp_v1x.comparison_gpu_policy import execution_settings
    return dict(model=digest(net.state_dict()), optimizer=digest(optimizer.state_dict()),
        gradients=digest({name: value.grad for name,value in net.named_parameters()}),
        rng=digest(capture_rng()), modes=[module.training for module in net.modules()],
        settings=execution_settings(net))


def gradient_norms(net):
    import torch
    names, values = [], []
    for name, parameter in net.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise AssertionError('Disconnected full-model gradient: '+name)
            names.append(name)
            values.append(parameter.grad.detach().float().norm())
    norms = torch.stack(values).cpu()
    if len(names) != 1085 or not bool(torch.isfinite(norms).all()):
        raise AssertionError('Expected all1085 finite original trainable gradients')
    return dict(zip(names, map(float,norms)))


def update_trial(net, batch, saved, config, policy, budget, label):
    """One genuine AdamW update from the same immutable complete checkpoint."""
    import torch
    from hiercp_v1x import comparison_gpu_policy as gpu
    from hiercp_v1x import u_bridge_training as engine
    training=config['training']
    net.load_state_dict(saved['model'],strict=True)
    gpu.apply_execution_settings(net,policy)
    net.train()
    optimizer=torch.optim.AdamW(net.parameters(),lr=training['lr'],
        weight_decay=training['weight_decay'],fused=training['fused_optimizer'])
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    scaler=torch.amp.GradScaler('cuda',enabled=training['amp'])
    scaler.load_state_dict(copy.deepcopy(saved['scaler']))
    expected_steps=saved['state']['updates']
    engine.restore_optimizer_history(optimizer,expected_steps)
    optimizer.zero_grad(set_to_none=True)
    engine.restore_rng(saved['rng'])
    before=engine.digest(net.state_dict())
    budget.check()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    events=[torch.cuda.Event(enable_timing=True) for _ in range(5)]
    events[0].record()
    with torch.autocast('cuda',enabled=training['amp']):
        output=net(batch)
        loss,terms=engine.pair_objective(output.scores,output.consistency)
    events[1].record()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite actual full-model update loss')
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    events[2].record();events[2].synchronize()
    receipt=engine.gradient_receipt(net,engine._model_contract(net,True))
    if receipt['gradient_present']!=1085 or receipt['missing'] or not receipt['finite']:
        raise FloatingPointError('Actual scaled backward must reach all1085 finite gradients')
    norms=gradient_norms(net)
    observed=dict(scores=[s.detach().float().cpu().tolist() for s in output.scores],
        loss=float(loss.detach()),consistency=float(output.consistency.detach()),gradient_norms=norms)
    events[3].record()
    clipped=torch.nn.utils.clip_grad_norm_(list(net.parameters()),training['grad_clip'],error_if_nonfinite=True)
    scaler.step(optimizer);scaler.update()
    events[4].record();events[4].synchronize()
    elapsed=time.perf_counter()-start
    steps=engine.restore_optimizer_history(optimizer,expected_steps+1)
    budget.check()
    after=engine.digest(net.state_dict())
    if before==after:
        raise AssertionError('Genuine DEBUG optimizer update did not change neural weights')
    report=dict(label=label,policy=policy,**observed,gradient=receipt,
        gradient_before_clip=float(clipped),optimizer_step_before=expected_steps,
        optimizer_step_after=expected_steps+1,optimizer_history=steps,
        forward_seconds=events[0].elapsed_time(events[1])/1000,
        backward_seconds=events[1].elapsed_time(events[2])/1000,
        clip_optimizer_seconds=events[3].elapsed_time(events[4])/1000,
        wall_seconds_including_untimed_diagnostics=elapsed,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        initial_state_sha256=before,updated_state_sha256=after,
        optimizer_state_sha256=engine.digest(optimizer.state_dict()),
        rng_sha256=engine.digest(engine.capture_rng()),genuine_optimizer_updates=1)
    report['compute_seconds']=sum(report[name] for name in ('forward_seconds','backward_seconds','clip_optimizer_seconds'))
    report['samples_per_second']=len(batch.counts)/report['compute_seconds']
    del output,loss,terms,optimizer,scaler
    net.zero_grad(set_to_none=True)
    gc.collect();torch.cuda.empty_cache()
    print(f'DEBUG GPU update {label} | compute={report["compute_seconds"]:.3f}s '
          f'peak={report["peak_cuda_bytes"]/2**30:.3f}GiB | finite1085/1085',flush=True)
    return report


def numerical_comparison(baselines, winner, tolerance):
    """Publish observed baseline scatter separately from the explicit AMP bound."""
    import torch
    rtol,atol=tolerance['rtol'],tolerance['atol']
    result={}
    for name in ('scores','loss','consistency','gradient_norms'):
        def tensor(row):
            value=row[name]
            return torch.tensor(list(value.values()) if isinstance(value,dict) else value,dtype=torch.float64)
        first,second,actual=tensor(baselines[0]),tensor(baselines[1]),tensor(winner)
        noise=(first-second).abs()
        difference=(first-actual).abs()
        # Per-parameter norm bounds are the norm of the policy's elementwise
        # bound, not a new relative-only test that exaggerates near-zero norms.
        absolute=atol
        if name=='gradient_norms':
            absolute=torch.tensor([atol*math.sqrt(n) for n in winner['gradient_parameter_sizes']],dtype=torch.float64)
        bound=noise+absolute+rtol*first.abs()
        result[name]=dict(baseline_repeat_max_abs=float(noise.max()),winner_max_abs=float(difference.max()),
            within_bound=bool((difference<=bound).all()),bound='observed baseline scatter + atol + rtol*abs(reference)'
                if name!='gradient_norms' else 'baseline scatter + sqrt(parameter_numel)*atol + rtol*reference_norm')
    return dict(checks=result,accepted=all(row['within_bound'] for row in result.values()),
        rtol=rtol,atol=atol,bitwise_equivalence_claim=False,
        scope='Actual AMP-scaled optimizer update; separate from calibrator per-tensor and global gradient-vector checks')


def validation_trial(net,batch,saved,policy,config,chunk,budget,label):
    import torch
    from hiercp_v1x.comparison_gpu_policy import apply_execution_settings
    from hiercp_v1x.u_bridge_training import restore_rng,capture_rng,digest
    net.load_state_dict(saved['model'],strict=True);apply_execution_settings(net,policy);net.eval()
    restore_rng(saved['rng']);rng_before=digest(capture_rng())
    # Chunked inference moves its upper graphs in-place. Keep the cached CPU
    # batch intact, cloning only these small upper objects for this invocation.
    current=copy.copy(batch)
    current.patient_batch=batch.patient_batch.clone()
    current.prototype_batch=batch.prototype_batch.clone()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
    with torch.no_grad(),torch.autocast('cuda',enabled=config['training']['amp']):
        scores=net.score_inference_chunked(current,local_chunk_size=chunk)
    torch.cuda.synchronize();seconds=time.perf_counter()-began
    if len(scores)!=1 or scores[0].numel()!=129 or not bool(torch.isfinite(scores[0]).all()):
        raise AssertionError('Validation must retain all129 ordered finite scores')
    budget.check()
    result=dict(label=label,policy=policy,local_chunk_size=chunk,seconds=seconds,
        scores=scores[0].detach().float().cpu().tolist(),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),RNG_unchanged=digest(capture_rng())==rng_before,
        original_joint_upper_candidates=129)
    if not result['RNG_unchanged']:
        raise AssertionError('Evaluation execution policy consumed global RNG')
    del current,scores
    gc.collect();torch.cuda.empty_cache()
    print(f'DEBUG full129 {label} | {seconds:.3f}s | peak={result["peak_cuda_bytes"]/2**30:.3f}GiB',flush=True)
    return result


def runtime_smoke(net,provider,manifest,initial,calibration,budget,output):
    import torch
    from hiercp_v1x import comparison_training as training
    from hiercp_v1x.comparison_execution import comparison_execution
    from hiercp_v1x.comparison_gpu_policy import apply_execution_settings,FIELDS
    from hiercp_v1x.comparison_gpu_runtime import ComparisonGpuRuntime
    from tools.verify_comparison_cache_debug import read,sha
    helpers=('comparison_execution.py','comparison_runtime.py','comparison_gpu_runtime.py','comparison_gpu_policy.py')
    sources_before={name:sha(ROOT/'hiercp_v1x'/name) for name in helpers}
    root=output/'runtime_native_fixed'
    receipt=copy.deepcopy(calibration['reports']['native_fixed']);receipt['selected_physical_batch']=2
    runtime=dict(batch_calibration=receipt,validation_local_chunk_size=manifest['validation_local_chunk'],
        expected_parameters=10434532,pause_file=str(root/'STOP_AFTER_BATCH'),debug_pause_after_updates=1)
    config=copy.deepcopy(manifest['config']);config['u_bridge_runtime']=runtime
    identity=dict(contract_sha256=manifest['sha256'],initial_neural_sha256=initial['model_sha256'],
        initial_state_sha256=training.digest(initial['model']))
    results=[]
    # One explicit DEBUG exception proves the real runtime's same-batch retry.
    # Arm it only after a genuine measured optimized policy is admitted, so
    # calibration forwards and the ordinary original-policy route are untouched.
    injected=dict(injected=False,retry_verified=False,scope='DEBUG injected CUDA OOM; not a measured hardware OOM')
    pending={}
    original_ensure=ComparisonGpuRuntime.ensure
    original_forward=net.forward
    def ensure_with_debug_oom(controller,batch,workload):
        value=original_ensure(controller,batch,workload)
        if not injected['injected'] and controller.current_settings()!=controller.original:
            pending.update(batch=batch, rng=training.digest(training.capture_rng()),
                model=training.digest(net.state_dict()),original=copy.deepcopy(controller.original),
                optimized=controller.current_settings(),armed=True)
        return value
    def forward_with_debug_oom(batch,*args,**kwargs):
        if pending.get('armed'):
            pending['armed']=False;pending['retry']=True;injected['injected']=True
            injected['optimized_policy']=pending['optimized']
            # A failed stochastic forward may already have advanced both RNGs.
            torch.rand(3);torch.rand(3,device='cuda')
            raise torch.cuda.OutOfMemoryError('DEBUG injected once after calibrated admission; before optimizer update')
        if pending.get('retry'):
            from hiercp_v1x.comparison_gpu_policy import execution_settings
            checks=dict(same_complete_batch=batch is pending['batch'],
                rng_restored=training.digest(training.capture_rng())==pending['rng'],
                model_unchanged=training.digest(net.state_dict())==pending['model'],
                gradients_cleared=all(parameter.grad is None for parameter in net.parameters()),
                original_execution_restored=execution_settings(net)==pending['original'])
            injected['retry_checks']=checks
            if not all(checks.values()):
                raise AssertionError('Runtime OOM retry changed input/RNG/model/gradient state: '+repr(checks))
            injected['retry_verified']=True;pending.clear()
        return original_forward(batch,*args,**kwargs)
    with comparison_execution(),patch.object(ComparisonGpuRuntime,'ensure',ensure_with_debug_oom),\
            patch.object(net,'forward',forward_with_debug_oom):
        for mode in ('pause_after_one','resume_to_complete','completed_resume'):
            net.load_state_dict(initial['model'],strict=True);training.restore_rng(initial['rng'])
            # The integrated controller must observe the real conservative
            # starting attributes and select its own execution policy.
            apply_execution_settings(net,{name:manifest['config']['model'][name] for name in FIELDS})
            if mode!='pause_after_one':
                runtime.pop('debug_pause_after_updates',None)
                runtime['resume_checkpoint']=str(root/'checkpoint_latest.pt')
            result=training.run_arm(net,provider,config,arm='native_fixed',output=root,physical_batch=2,
                workers=4,epochs=2,identity=identity,budget=budget,debug=True)
            results.append(dict(mode=mode,result=result))
            checkpoint=torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
            if mode=='pause_after_one' and checkpoint['state']['updates']!=1:
                raise AssertionError('Runtime DEBUG did not pause after exactly one real update')
            if mode=='pause_after_one' and injected['injected']:
                injected['pause_checkpoint_updates']=checkpoint['state']['updates']
                injected['pause_checkpoint_attempts']=checkpoint['state']['attempts']
                if not injected['retry_verified'] or checkpoint['state']['attempts']!=1:
                    raise AssertionError('Injected failed CUDA attempt was counted or repeated an optimizer update')
            if mode=='resume_to_complete':
                content_before=checkpoint['content_sha256']
                updates_before=(root/'update_timing.jsonl').read_bytes()
            if mode=='completed_resume' and (checkpoint['content_sha256']!=content_before
                    or (root/'update_timing.jsonl').read_bytes()!=updates_before):
                raise AssertionError('Completed runtime resume changed neural history or repeated updates')
            del checkpoint
    rows=[json.loads(line) for line in (root/'update_timing.jsonl').read_text().splitlines()]
    if len(rows)!=2 or any(row['status']!='OPTIMIZER_UPDATED' or row['gradient']['gradient_present']!=1085
            or not row['gradient']['finite'] for row in rows):
        raise AssertionError('Original runtime must complete exactly two full1085-gradient updates')
    ownership=read(root/'training_identity.json')
    bound=ownership['binding']['config']
    if bound['model']!=manifest['config']['model']:
        raise AssertionError('Execution tuning changed checkpoint-bound model configuration')
    if {name:sha(ROOT/'hiercp_v1x'/name) for name in helpers}!=sources_before:
        raise AssertionError('GPU execution helpers changed during the runtime smoke')
    return dict(results=results,optimizer_updates=2,validation129_passes=3,
        pause_then_resume=True,completed_resume_additional_updates=0,
        injected_optimized_oom=injected,
        execution_helper_sha256=sources_before,
        checkpoint_bound_model_config_unchanged=True,checkpoint_sha256=sha(root/'checkpoint_latest.pt'))


def main(argv=None):
    a=parse(argv)
    from tools.current_gpu import select_record
    device_selection=select_record(a.gpu)  # Must precede Torch/CUDA initialization.
    import torch
    from tools.verify_comparison_cache_debug import read,sha,canonical
    from tools.verify_comparison_preparation_debug import fingerprint
    from hiercp_v1x.comparison_experiment import FORMAT,FILES,digest
    from hiercp_v1x.scope_probe_support import activate_original,state_digest
    from hiercp_v1x import bounded_scope
    from hiercp_v1x import u_bridge_training as engine
    from hiercp_v1x import comparison_gpu_policy as gpu
    a.reference=a.reference.resolve(strict=True);a.completed_run=a.completed_run.resolve(strict=True)
    a.output=a.output.resolve()
    if a.output.exists() or any(a.output.is_relative_to(p) or p.is_relative_to(a.output)
            for p in (a.reference,a.completed_run)):
        raise FileExistsError('Use a new DEBUG output disjoint from all completed experiments')
    manifest=read(a.reference/'experiment.json')
    if (manifest['format']!=FORMAT or not manifest['debug'] or manifest['epochs']!=2
            or manifest['workers']!=4 or manifest['explicit_batch_candidates']!=[2]
            or manifest['sha256']!=digest({k:v for k,v in manifest.items() if k!='sha256'})):
        raise ValueError('Requires the sealed real-CT physical2,worker4,two-epoch DEBUG fixture')
    frozen_before={name:sha(ROOT/name) for name in FILES}
    if frozen_before!=manifest['helpers'] or sha(a.inventory)!=manifest['baseline']['inventory_sha256'] \
            or sha(a.bank)!=manifest['baseline']['bank_fixture_sha256']:
        raise ValueError('Frozen helpers, inventory or prototype fixture changed')
    completed=read(a.completed_run/'verification.json')
    if (completed.get('exact_inputs') is not True or completed.get('optimizer_updates')!=2
            or completed.get('parameters')!=10434532):
        raise ValueError('Completed exact-input/full-model CUDA evidence is required')
    if sha(a.completed_run/'report.json')!=completed['CUDA_report']['sha256']:
        raise ValueError('Completed CUDA evidence changed')
    source=Path(manifest['original']['source'])
    if canonical(activate_original(source))!=manifest['original'] \
            or canonical(bounded_scope.install(10,expected_snapshot_root=source))!=manifest['scope']:
        raise ValueError('Original source or complete10mm scope changed')
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.tensor import configure_runtime,collect_runtime_resources
    from hiercp_v1x.comparison_data import ComparisonData
    from hiercp_v1x.comparison_preparation import prepared_provider
    from hiercp_v1x.host_memory import PressureBudget,pressure_aware_provider,_evict_provider
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one selected real CUDA device required')
    torch.set_num_threads(4)
    config=copy.deepcopy(manifest['config'])
    configure_runtime(deterministic=bool(config['runtime'].get('deterministic',True)),
        allow_tf32=bool(config['runtime'].get('allow_tf32',False)),
        cudnn_benchmark=bool(config['runtime'].get('cudnn_benchmark',False)))
    capacity=torch.cuda.get_device_properties(0).total_memory
    cuda_gib=manifest['cuda_gib'] if a.cuda_gib is None else a.cuda_gib
    cuda_bytes=int(cuda_gib*2**30)
    if not 0<cuda_bytes<capacity:
        raise ValueError('Explicit CUDA budget must fit actual selected GPU with driver headroom')
    torch.cuda.set_per_process_memory_fraction(cuda_bytes/capacity)
    budget=PressureBudget(cuda_bytes,int(manifest['rss_gib']*2**30),
        resident_bytes=int(manifest['resident_gib']*2**30))
    initial=torch.load(a.reference/'initial.pt',map_location='cpu',weights_only=False)
    saved_path=a.completed_run/'native_fixed/checkpoint_latest.pt'
    saved=torch.load(saved_path,map_location='cpu',weights_only=False)
    signed=dict(saved);signed.pop('content_sha256',None)
    if (engine.digest(signed)!=saved['content_sha256'] or saved['state']['updates']!=2
            or initial['model_sha256']!=state_digest(initial['model'])
            or initial['contract_sha256']!=manifest['sha256']):
        raise ValueError('Initial/full completed checkpoint identity changed')
    reference_paths=[a.reference/'experiment.json',a.reference/'initial.pt',a.reference/'calibration.json',saved_path]
    reference_before={str(p):sha(p) for p in reference_paths}
    data_root=a.completed_run/'data'
    cache_before=tree_stats(data_root)
    fixture=torch.load(a.bank,map_location='cpu',weights_only=False,mmap=True)
    bank=fixture['prototype_bank'];del fixture
    if bank.fingerprint()!=manifest['prototype_fingerprint']:
        raise ValueError('Prototype bank identity differs')
    a.output.mkdir(parents=True,exist_ok=False)
    resources=collect_runtime_resources('cuda',storage_path=a.output)
    write_new(a.output/'request.json',dict(debug=True,actual_resources=resources,device_selection=device_selection,
        full_model_parameters=10434532,physical_source_batch=2,train_candidates=8,validation_candidates=129,
        actual_train_sources=['liver_5','liver_6'],actual_validation_sources=['liver_31'],workers=4,
        warmup=a.warmup,repeats=a.repeats,cuda_gib=cuda_gib,sealed_fixture_cuda_gib=manifest['cuda_gib'],
        optimizer_updates_during_calibration=0,full_training=False,quality_verified=False))
    provider=prepared_provider(pressure_aware_provider(ComparisonData))(
        manifest['baseline']['source_samples'],read(a.inventory)['raw_records'],config,bank,data_root,
        4,int(manifest['resident_gib']*2**30),budget,regions_dir=Path(manifest['prepared_data_root'])/'regions')
    net=HierarchicalPyGPlacementModel(**config['model']).cuda()
    if sum(p.numel() for p in net.parameters())!=10434532:
        raise AssertionError('Full original model size changed')
    net.load_state_dict(saved['model'],strict=True);net.train()
    original=gpu.execution_settings(net)
    training=config['training']
    optimizer=torch.optim.AdamW(net.parameters(),lr=training['lr'],weight_decay=training['weight_decay'],
        fused=training['fused_optimizer'])
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    engine.restore_optimizer_history(optimizer,2)
    engine.restore_rng(saved['rng'])
    began=time.perf_counter()
    result=dict(debug=True,actual_CT=True,actual_CUDA=True,full_training=False,quality_verified=False,
        harness_sha256=sha(Path(__file__)),gpu_policy_sha256=sha(ROOT/'hiercp_v1x/comparison_gpu_policy.py'))
    try:
        import hiercp.cache as archived_cache
        with ExitStack() as guard:
            for name in ('_case','_source','_upper_context'):
                guard.enter_context(patch.object(provider,name,denied))
            guard.enter_context(patch.object(archived_cache,'build_inference_sample',denied))
            train_ids=[row['index'] for row in provider.examples('train')]
            val_ids=[row['index'] for row in provider.examples('val')]
            if len(train_ids)!=2 or len(val_ids)!=1:
                raise AssertionError('Entire existing real fixture inventory is required')
            load_start=time.perf_counter()
            train=provider.batch(train_ids,'native_fixed',1,True,full=False)
            validation=provider.batch(val_ids,'native_fixed',training['fixed_validation_epoch'],False,full=True)
            result['input_seconds']=time.perf_counter()-load_start
            result['inputs']=dict(train=fingerprint(vars(train)),validation=fingerprint(vars(validation)))
            write_new(a.output/'inputs.json',result.pop('inputs'))
            train=train.pin_memory().to('cuda',non_blocking=True);torch.cuda.synchronize()
            before=state_receipt(net,optimizer)
            def notify(row):
                with (a.output/'calibration_trials.jsonl').open('a',encoding='utf8') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                print('DEBUG GPU calibration | '+str(row),flush=True)
            calibration_began=time.perf_counter()
            try:
                selected,calibration=gpu.calibrate_execution_policy(net,train,
                    lambda output:engine.pair_objective(output.scores,output.consistency)[0],
                    amp=training['amp'],cuda_limit_bytes=cuda_bytes,warmup=a.warmup,repeats=a.repeats,notify=notify)
            finally:
                result['calibration_wall_seconds']=time.perf_counter()-calibration_began
                restored=state_receipt(net,optimizer)
                result['calibration_state_preservation']=dict(before=before,after=restored,exact=before==restored)
                write_new(a.output/'calibration_restoration.json',result['calibration_state_preservation'])
            after=state_receipt(net,optimizer)
            if before!=after:
                raise AssertionError('Calibration modified model/optimizer/gradients/RNG/modes/settings')
            measured_compute=sum(trial['forward_backward_seconds'] for row in calibration['reports']
                for trial in row.get('trials',[]))
            result['calibration_forward_backward_seconds']=measured_compute
            result['calibration_other_wall_seconds']=result['calibration_wall_seconds']-measured_compute
            result['calibration_other_wall_scope']='CPU snapshots, equivalence checks, device transfers, cleanup and orchestration; not GPU compute'
            result['calibration']=calibration
            result['calibration_state_preservation']=dict(before=before,after=after,exact=True)
            write_new(a.output/'calibration.json',calibration)
            # An explicit DEBUG failure injection checks restoration on rejection,
            # using the real full model and one real batch; no fake success result.
            rejected=False
            try:
                gpu.calibrate_execution_policy(net,train,
                    lambda output:engine.pair_objective(output.scores,output.consistency)[0]*float('nan'),
                    amp=training['amp'],cuda_limit_bytes=cuda_bytes,warmup=1,repeats=2)
            except gpu.ExecutionCalibrationRejected as error:
                rejected=bool(error.reports and any('Nonfinite' in row.get('error','') for row in error.reports))
                result['injected_nonfinite_reports']=error.reports
            if not rejected or state_receipt(net,optimizer)!=before:
                raise AssertionError('Nonfinite DEBUG calibration was not rejected with exact state restoration')
            result['injected_nonfinite_rejected_and_state_restored']=True
            del optimizer
            net.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            update_rows=[]
            for label,policy in (('baseline_1',original),('selected_1',selected),
                                 ('baseline_2',original),('selected_2',selected)):
                row=update_trial(net,train,saved,config,policy,budget,label)
                row['gradient_parameter_sizes']=[p.numel() for p in net.parameters() if p.requires_grad]
                update_rows.append(row)
                write_new(a.output/(label+'.json'),row)
            baselines=[row for row in update_rows if row['label'].startswith('baseline')]
            checks=[numerical_comparison(baselines,row,calibration['numerical_equivalence'])
                    for row in update_rows if row['label'].startswith('selected')]
            if not all(check['accepted'] for check in checks):
                write_new(a.output/'numerical_failure.json',checks)
                raise AssertionError('Actual scaled optimizer update differs beyond recorded numerical bounds')
            result['actual_updates']=update_rows
            result['update_numerical_checks']=checks
            result['update_RNG_equal']=len({row['rng_sha256'] for row in update_rows})==1
            if not result['update_RNG_equal']:
                raise AssertionError('Execution settings changed original update RNG consumption')
            validation_rows=[validation_trial(net,validation,saved,policy,config,
                manifest['validation_local_chunk'],budget,label) for label,policy in
                (('baseline_1',original),('selected',selected),('baseline_2',original))]
            scores=[torch.tensor(row['scores']) for row in validation_rows]
            noise=(scores[0]-scores[2]).abs();difference=(scores[0]-scores[1]).abs()
            tol=calibration['numerical_equivalence']
            accepted=bool((difference<=noise+tol['atol']+tol['rtol']*scores[0].abs()).all())
            result['validation']=dict(rows=validation_rows,baseline_repeat_max_abs=float(noise.max()),
                selected_max_abs=float(difference.max()),within_recorded_bounds=accepted,
                bound='observed baseline repeat noise + atol + rtol*abs(reference)',
                dense_policy_selected=True,original_local_graph_chunk_unchanged=True)
            if not accepted:
                raise AssertionError('Full129 selected-policy scores exceed numerical consistency bounds')
            del train,validation
            net.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            result['runtime_smoke']=None if a.skip_runtime else runtime_smoke(net,provider,manifest,initial,
                read(a.reference/'calibration.json'),budget,a.output)
        result.update(selected_policy=selected,original_policy=original,resources=resources,
            full_model_parameters=10434532,gradient_parameter_tensors=1085,physical_source_batch=2,
            train_candidates=8,validation_candidates=129,workers=4,
            original_bound_config_unchanged=config==manifest['config'],
            source_reference_bytes_preserved={str(p):sha(p) for p in reference_paths}==reference_before,
            existing_cache_inventory_size_mtime_preserved=tree_stats(data_root)==cache_before,
            frozen_helpers_preserved={name:sha(ROOT/name) for name in FILES}==frozen_before,
            provider=provider.report(),wall_seconds=time.perf_counter()-began,
            limitation='Local complete three-case DEBUG fixture. Selected GPU is recorded; no server/A6000 '
                'speed, full-cohort worst-case VRAM, forty-epoch quality or bitwise trajectory claim.')
        if not all(result[name] for name in ('original_bound_config_unchanged','source_reference_bytes_preserved',
                'existing_cache_inventory_size_mtime_preserved','frozen_helpers_preserved')):
            raise AssertionError('Reference/config/cache/frozen source preservation check failed')
        if (sha(Path(__file__))!=result['harness_sha256']
                or sha(ROOT/'hiercp_v1x/comparison_gpu_policy.py')!=result['gpu_policy_sha256']):
            raise AssertionError('GPU probe/calibration source changed during its execution')
        write_new(a.output/'report.json',result)
        print(json.dumps(dict(report=str(a.output/'report.json'),selected_policy=selected,
            calibrated_speedup=calibration['measured_speedup'],calibration_optimizer_updates=0,
            actual_update_checks=4,runtime_optimizer_updates=0 if a.skip_runtime else 2,
            full_training=False,quality_verified=False)),flush=True)
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}',wall_seconds=time.perf_counter()-began)
        if hasattr(error,'reports'):
            result['rejected_calibration_reports']=error.reports
        write_new(a.output/'failure.json',result)
        raise
    finally:
        _evict_provider(provider)
        gpu.apply_execution_settings(net,original)
        engine.restore_rng(saved['rng'])


if __name__=='__main__':
    main()
