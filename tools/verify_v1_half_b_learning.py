"""Actual-CUDA DEBUG: preserved original-v1 L0 with the v2.2 prompt upper model.

Original query fixture, full eight/pool128, both training views, epoch29 ranking
and consistency loss remain fixed. One real extra training patient is support
only. Every query excludes its own patient before native upper construction.
This is bounded learning evidence, never production training or quality proof.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import json
import math
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from hiercp_v1x import bounded_scope
from hiercp_v1x.contracts import canonical_hash
from hiercp_v1x.scope_learning_inputs import rebuild_scope, supervision_digest
from hiercp_v1x.scope_probe_support import activate_original, state_digest, _sha
from tools.prepare_v1_half_b_support_debug import FORMAT
from tools.verify_v14_learning_replay import write_new
from tools.verify_v1_half_a_learning import (verify_bounded_reference, compare_curves,
    half_a_state_digest as neural_digest, same_rng, gradient_observation, parameter_changes,
    evaluate, recorded_path)


def validate_support_manifest(extension, original, baseline, raw):
    """Bind support-only real CT to the unchanged query contract and training split."""
    if (extension.get('format') != FORMAT or extension.get('status') != 'COMPLETE'
            or any(extension.get(k) is not True for k in
                ('debug','actual_CT','prototype_bank_unchanged','original_inputs_preserved','source_preserved'))
            or any(extension.get(k) is not False for k in
                ('actual_CUDA','full_training','full_evaluation','quality_verified','production_ready',
                 'checkpoint_written','GPU_context_initialized'))
            or canonical_hash({k:v for k,v in extension.items() if k != 'identity_sha256'})
                != extension.get('identity_sha256')):
        raise ValueError('Completed signed CPU-only actual-CT DEBUG support extension required')
    extra = extension.get('additional_support_only_cases', [])
    query, held = original['train_cases'], original['validation_cases']
    cohort = [*query,*extra]
    if (not extra or len(cohort) < 3 or len(set(cohort)) != len(cohort)
            or set(cohort) & set(held) or any(c not in raw['split']['inner_train'] for c in cohort)
            or set(cohort) & set(raw['split']['inner_val'])
            or extension['support_train_cases'] != cohort
            or extension['query_train_cases'] != query or extension['validation_cases'] != held
            or extension['raw_training_split'] != raw['split']['inner_train']
            or extension['prototype_training_cases'] != original['prototype_training_cases']
            or extension['original_fixture_sha256'] != original['fixture_sha256']
            or extension['original_supervision_sha256'] != baseline['supervision_sha256']
            or extension['configuration'] != original['config']
            or extension['original_configuration_sha256'] != canonical_hash(original['config'])
            or extension['bounded_scope'] != baseline['bounded_scope']
            or extension['candidate_count'] != 8 or extension['candidate_pool_size'] != 128):
        raise ValueError('Support must add real training-only tasks without altering original queries/GT/bank')
    return cohort


def matched_comparisons(curve, baseline):
    result = compare_curves(curve,baseline)
    for row in result:
        row['half_b'] = row.pop('half_a')
        row['support_cohort_differs_from_baseline'] = True
        row['score_scale_differs_from_baseline'] = True
    return result


def groups_for(net):
    local, core = net.local_encoder, net.half_b.core
    return dict(CNN=local.dense_encoder, L0=local.blocks, original_L0_total=local,
        role_attention_pool=local.pool, shell_attention_pool=local.context_shell_pool,
        local_fusion=local.final_fuse, v222_L1=core.l1, v222_L2=core.l2,
        v222_L2_updates=core.l2_updates, v222_upper_total=net.half_b)


def encode_support(net, cpu_batch, *, amp):
    """One disjoint-union local batch, shared dense maps and both fixed views.

    Support is detached eval memory; no query scores/upper/GT edges are run here.
    Keep the query encoder mode/RNG untouched and CPU graph stores cached.
    """
    import torch
    from hiercp.tensor import capture_rng_state, restore_rng_state
    from hiercp_v1x.scope_learning_loop import independent_transfer
    from hiercp_v1x.half_b_support import encode_local_support
    rng = capture_rng_state()
    device_batch = None
    try:
        device_batch = independent_transfer(cpu_batch)
        if device_batch.local_batch_view2 is None:
            raise AssertionError('Fixed support memory requires both original epoch0 local views')
        value = encode_local_support(net,device_batch,device='cuda',use_amp=amp)
        if value.shape != (sum(cpu_batch.counts),128) or not bool(torch.isfinite(value).all()):
            raise AssertionError('Actual original-L0 support fused embeddings are malformed/nonfinite')
        return value
    finally:
        del device_batch
        restore_rng_state(rng)


def execute(a, root, baseline, manifest, protocol, extension, raw, inputs):
    from tools.local_cnn_device import select
    select(a.gpu)
    source = a.source.resolve(strict=True)
    proof = activate_original(source)
    import psutil
    import torch
    from hiercp.data import collate_samples, summarize_hierarchical_batch
    from hiercp.loss import CurriculumConfig, curriculum_ranking_loss
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.sample import materialize_sample_views
    from hiercp.tensor import capture_rng_state, configure_runtime, set_seed
    from hiercp_v1x.half_b_model import install_half_b, half_b_identity, model_contract, SUPPORT_POLICY
    from tqdm import tqdm
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one actual CUDA GPU required; no CPU training fallback')
    total = torch.cuda.get_device_properties(0).total_memory
    limit = int(a.cuda_gib*2**30)
    if not 0 < limit < total:
        raise ValueError('Explicit CUDA budget must leave device headroom')
    torch.cuda.set_per_process_memory_fraction(limit/total)
    torch.set_num_threads(protocol['workers'])
    started, process = time.perf_counter(), psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def monitor():
        while not stop.wait(.1):
            peak[0] = max(peak[0],process.memory_info().rss)
    watcher = threading.Thread(target=monitor,daemon=True); watcher.start()
    def budget():
        peak[0] = max(peak[0],process.memory_info().rss)
        if peak[0] > int(a.rss_gib*2**30):
            raise MemoryError('Explicit RSS budget exceeded; complete original inputs preserved')
        if time.perf_counter()-started > a.wall_seconds:
            raise TimeoutError('Explicit DEBUG wall budget exceeded; no quality PASS')
    try:
        config = manifest['config']
        fixture = torch.load(a.fixture,map_location='cpu',weights_only=False,mmap=True)
        if (fixture['config'] != config or fixture['train_cases'] != protocol['train_cases']
                or fixture['validation_cases'] != protocol['validation_cases']
                or fixture['prototype_bank'].fingerprint() != extension['prototype_bank_fingerprint']):
            raise ValueError('Original query configuration/cohort/bank differs')
        samples = fixture['samples']
        expected_order = [(c,'train') for c in protocol['train_cases']] + [(c,'val') for c in protocol['validation_cases']]
        if [(s['case_id'],s['split']) for s in samples] != expected_order:
            raise ValueError('Original complete query fixture order differs')
        extra_payload = torch.load(a.support_fixture,map_location='cpu',weights_only=False,mmap=True)
        if extra_payload['format'] != FORMAT or extra_payload['config'] != config:
            raise ValueError('Actual support extension format/configuration differs')
        extra = extra_payload['bounded_samples']
        if ([(s['case_id'],s['split']) for s in extra] != [(c,'train') for c in extension['additional_support_only_cases']]
                or supervision_digest(extra) != extension['native_supervision_sha256']
                or supervision_digest(extra_payload['native_samples']) != extension['native_supervision_sha256']):
            raise ValueError('Support-only original candidates/GT/upper input differs')
        for row in [*manifest['source_records'],*extension['source_records']]:
            for kind in ('image','label'):
                if _sha(Path(row[kind])) != row[kind+'_sha256']:
                    raise ValueError('Actual CT/annotation changed')
                inputs[str(Path(row[kind]).resolve())] = row[kind+'_sha256']
        before_supervision = supervision_digest(samples)
        if before_supervision != baseline['supervision_sha256']:
            raise ValueError('Original query GT/candidates changed from bounded baseline')
        receipt = bounded_scope.install(10.0,source)
        if receipt != baseline['bounded_scope']:
            raise ValueError('The exact original bounded10mm adapter differs')
        for sample in extra:
            if sample['graph_config'] != bounded_scope.configure(sample['graph_config'],10.0).to_dict():
                raise ValueError('Additional support canonical graph is not exact bounded10mm')
        bounded, preparation = rebuild_scope(samples,manifest,config,bounded_scope,10.0,protocol['workers'],budget)
        def reset():
            runtime = config['runtime']
            set_seed(config['seed'],deterministic=runtime['deterministic'])
            configure_runtime(deterministic=runtime['deterministic'],allow_tf32=runtime['allow_tf32'],
                cudnn_benchmark=runtime['cudnn_benchmark'])
        reset()
        with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
            views = list(pool.map(lambda s:materialize_sample_views(copy.deepcopy(s),
                training=s['split']=='train',epoch=29,global_seed=42),bounded))
            support_canonical = [s for s in bounded if s['split']=='train'] + extra
            support_views = list(pool.map(lambda s:materialize_sample_views(copy.deepcopy(s),
                training=False,epoch=0,global_seed=42),support_canonical))
        reset()
        net = HierarchicalPyGPlacementModel(**config['model'])
        native_identity = state_digest(net.state_dict())
        if native_identity != protocol['initial_neural_sha256']:
            raise AssertionError('Original seed42 initialization differs')
        old_state = {n:v.detach().clone() for n,v in net.state_dict().items()}
        local_before = {n:v for n,v in old_state.items() if n.startswith('local_encoder.')}
        local_identity, local_object, old_rng = state_digest(local_before), net.local_encoder, capture_rng_state()
        install_half_b(net,debug_support=True)
        after_rng = capture_rng_state()
        if not same_rng(old_rng,after_rng) or net.local_encoder is not local_object:
            raise AssertionError('HALF B installation altered caller RNG or original local object')
        if state_digest({n:v for n,v in net.state_dict().items() if n.startswith('local_encoder.')}) != local_identity:
            raise AssertionError('HALF B installation changed original local initialization bytes')
        initial_identity = neural_digest(net.state_dict())
        with (root/'initialization_audit_DEBUG.pt').open('xb') as stream:
            torch.save(dict(debug=True,production_ready=False,full_training=False,
                purpose='untrained original/HALF B state and RNG audit, no production checkpoint',
                native_initial_state=old_state,half_b_initial_state=net.state_dict(),
                native_rng=old_rng,half_b_rng=after_rng),stream)
        del old_state,old_rng,after_rng,local_before,local_object
        net.cuda()
        groups = groups_for(net)
        training = [s for s in views if s['split']=='train']
        validation = [s for s in views if s['split']=='val']
        support_cpu = collate_samples(support_views).pin_memory()
        batch = collate_samples(training).to('cuda')
        held = collate_samples(validation).to('cuda')
        if batch.local_batch_view2 is None or support_cpu.local_batch_view2 is None:
            raise AssertionError('Original query/support two-view paths required')
        fields = {f.name for f in dataclasses.fields(CurriculumConfig)}
        objective = CurriculumConfig(**{k:v for k,v in config['training'].items() if k in fields}); objective.validate()
        parameters = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters,lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay'],fused=config['training']['fused_optimizer'])
        if {id(p) for g in optimizer.param_groups for p in g['params']} != {id(p) for p in parameters}:
            raise AssertionError('Fresh original optimizer omitted trainable parameters')
        scaler = torch.amp.GradScaler('cuda',enabled=config['training']['amp'])
        result = dict(status='RUNNING',experiment='HALF B: preserved v1 L0 + v2.2 prompt L1/L2/cosine score',
            debug=True,smoke_test=True,actual_CT=True,actual_CUDA=True,full_training=False,
            full_evaluation=False,quality_verified=False,production_ready=False,checkpoint_written=False,
            diagnostic_initial_state_written=True,original_configuration=config,
            configuration_sha256=canonical_hash(config),protocol=protocol,bounded_scope=receipt,
            margin_mm=10.0,half_b=half_b_identity(),half_b_model_contract=model_contract(),
            native_equivalence_claimed=False,original_L0_object_preserved=True,
            original_L0_initial_state_preserved=True,caller_RNG_preserved_during_install=True,
            original_neural_sha256=native_identity,original_L0_initial_sha256=local_identity,
            half_b_initial_neural_sha256=initial_identity,supervision_sha256=before_supervision,
            model_parameters=sum(p.numel() for p in net.parameters()),trainable_parameters=sum(p.numel() for p in parameters),
            group_parameters={n:sum(p.numel() for p in m.parameters()) for n,m in groups.items()},
            score_has_trainable_parameters=False,bridge_has_trainable_parameters=False,
            score_semantics='native patient-mass-weighted cosine-mixture logit1 minus logit0',
            added_alignment_or_query_CE_loss=False,original_complete_ranking_consistency_loss=True,
            input=summarize_hierarchical_batch(batch),support_input=summarize_hierarchical_batch(support_cpu),
            query_train_cases=protocol['train_cases'],held_out_cases=protocol['validation_cases'],
            support_train_cases=extension['support_train_cases'],support_only_cases=extension['additional_support_only_cases'],
            support_policy=SUPPORT_POLICY,support_fixed_view_epoch=0,query_training_view_epoch=29,
            held_out_view_epoch=0,query_patient_always_excluded_from_support=True,
            support_labels='original ranking task: candidate0 class1, other seven class0; no biological absence claim',
            support_full_signed_training_cache=False,support_debug_subset=True,
            support_sample_count=len(support_views),support_candidate_count=len(support_views)*8,
            physical_sample_batch=len(training),physical_candidate_graph_batch=len(training)*8,
            gradient_accumulation_steps=1,data_parallel_workers=1,effective_sample_batch=len(training),
            production_epochs_preserved=40,fixed_optimization_curriculum_epoch=29,candidate_pool=128,candidates_per_sample=8,
            workers=protocol['workers'],cached_query_GPU_batch=True,cached_support_CPU_batch=True,
            GPU=torch.cuda.get_device_name(0),torch_version=torch.__version__,cuda_build=torch.version.cuda,
            device_total_bytes=total,cuda_limit_bytes=limit,RSS_limit_bytes=int(a.rss_gib*2**30),
            precision=dict(amp=config['training']['amp']),
            resources_at_start=dict(cpu_physical=psutil.cpu_count(logical=False),cpu_logical=psutil.cpu_count(),
                ram_total_bytes=psutil.virtual_memory().total,ram_available_bytes=psutil.virtual_memory().available,
                GPU_free_bytes=torch.cuda.mem_get_info()[0]),
            source_proof=proof,preparation=preparation,optimizer_fresh=True,
            initial_support_extension=extension,support_refreshes=[],attempts=[],updates=[],curve=[],
            input_sha256=inputs,current_sources_sha256={n:_sha(ROOT/n) for n in (
                'tools/verify_v1_half_b_learning.py','tools/prepare_v1_half_b_support_debug.py',
                'tools/verify_v1_half_a_learning.py','hiercp_v1x/half_b_model.py','hiercp_v1x/half_b_support.py','hiercp_v1x/bounded_scope.py',
                'hiercp_v1x/scope_learning_inputs.py','hiercp_v1x/scope_probe_support.py')})
        print(json.dumps({k:result[k] for k in ('experiment','debug','model_parameters','group_parameters',
            'physical_sample_batch','support_train_cases','support_only_cases','support_input','precision','GPU')},allow_nan=False),flush=True)
        def append(name,row):
            result[name].append(row)
            with (root/f'{name}.jsonl').open('a',encoding='utf8') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        support_cases = tuple(s['case_id'] for s in support_canonical)
        sample_ids = tuple(f"{manifest['fixture_sha256'] if s['case_id'] in protocol['train_cases'] else extension['support_fixture_sha256']}:{s['case_id']}:{s['sample_index']}"
            for s in support_canonical)
        expected_samples = dict(zip(sample_ids,support_cases))
        owners = torch.arange(len(support_cases),device='cuda',dtype=torch.long).repeat_interleave(8)
        classes = torch.tensor([1,0,0,0,0,0,0,0],device='cuda',dtype=torch.long).repeat(len(support_cases))
        def refresh(step):
            budget(); began = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            memory = dict(embeddings=encode_support(net,support_cpu,amp=config['training']['amp']),
                owners=owners,classes=classes,patient_case_ids=support_cases,
                sample_ids=tuple(s for s in sample_ids for _ in range(8)),candidate_indices=tuple(range(8))*len(support_cases),
                expected_samples=expected_samples,training_case_ids=tuple(raw['split']['inner_train']),
                validation_case_ids=tuple(raw['split']['inner_val']),manifest_sha256=_sha(a.support_manifest),
                generation=f'DEBUG_successful_update_{step}',epoch=0,fixed_view_epoch=0,support_policy=SUPPORT_POLICY,
                debug=True,full_signed_training_cache=False)
            admission = net.half_b.bind_support(memory)
            torch.cuda.synchronize()
            append('support_refreshes',dict(after_successful_update=step,generation=memory['generation'],
                seconds=time.perf_counter()-began,shape=list(memory['embeddings'].shape),
                detached=not memory['embeddings'].requires_grad,original_fixed_two_views=True,
                admission=admission,
                physical_sample_batch=len(support_cases),physical_candidate_graph_batch=len(support_cases)*8,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        def evaluate_step(step):
            budget(); row = dict(step=step,train=evaluate(net,batch,config,objective),
                validation=evaluate(net,held,config,objective)); append('curve',row)
            print(f"HALF B DEBUG step{step} | train loss={row['train']['loss']:.5f} MRR={row['train']['MRR']:.4f} "
                f"top1={row['train']['top1']:.4f} margin={row['train']['mean_margin']:.5f} | "
                f"held loss={row['validation']['loss']:.5f} MRR={row['validation']['MRR']:.4f} "
                f"top1={row['validation']['top1']:.4f} margin={row['validation']['mean_margin']:.5f}",flush=True)
        initial_parameters = {id(p):p.detach().clone() for p in parameters}
        refresh(0); evaluate_step(0); reset()
        training_rng = capture_rng_state()
        connected, executed = set(), dict(CNN_calls=0,CNN_ROIs=0,L0_block_calls=0,v222_L1_calls=0,v222_L2_calls=0)
        def hook(name):
            def observe(module,args,output):
                executed[name] += 1
                if name=='CNN_calls': executed['CNN_ROIs'] += int(args[0].shape[0])
            return observe
        handles = [net.local_encoder.dense_encoder.register_forward_hook(hook('CNN_calls'))]
        for modules,name in ((net.local_encoder.blocks,'L0_block_calls'),(net.half_b.core.l1,'v222_L1_calls'),(net.half_b.core.l2,'v222_L2_calls')):
            handles.extend(m.register_forward_hook(hook(name)) for m in modules)
        observed = net.half_b.core.label_seed
        def counter():
            value = optimizer.state.get(observed,{}).get('step',0)
            return int(value.item()) if isinstance(value,torch.Tensor) else int(value)
        step, attempt = 0,0
        bar = tqdm(total=a.updates,desc='HALF B actual CUDA DEBUG',unit='update')
        try:
            while step < a.updates:
                attempt += 1; budget(); net.train(); optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); began = time.perf_counter()
                counts_before = dict(executed)
                with torch.autocast('cuda',enabled=config['training']['amp']):
                    out = net(batch)
                    base, terms = curriculum_ranking_loss(out.scores,batch.difficulty_list(),epoch=29,config=objective)
                    loss = base+config['training']['consistency_weight']*out.consistency
                if not bool(torch.isfinite(loss)):
                    append('attempts',dict(attempt=attempt,status='NONFINITE_FORWARD',completed_updates=step))
                    raise AssertionError('Nonfinite complete original loss; optimizer not called')
                torch.cuda.synchronize(); forward = time.perf_counter()-began
                back = time.perf_counter(); scale = scaler.get_scale(); old_counter = counter()
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.cuda.synchronize()
                backward = time.perf_counter()-back; gradient = gradient_observation(net,groups)
                if not gradient['finite']:
                    if not scaler.is_enabled():
                        raise AssertionError('Nonfinite gradients with original AMP disabled')
                    scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
                    append('attempts',dict(attempt=attempt,status='AMP_OVERFLOW_SKIPPED',completed_updates=step,
                        gradient=gradient,scale_before=scale,scale_after=scaler.get_scale(),optimizer_step_unchanged=counter()==old_counter))
                    if counter()!=old_counter or not scaler.get_scale()<scale:
                        raise AssertionError('AMP overflow did not honestly skip optimizer/update scale')
                    del out,loss,base,terms
                    continue
                if gradient['nonfinite_groups']: raise AssertionError('Nonfinite diagnostic norm')
                clip = torch.nn.utils.clip_grad_norm_(parameters,config['training']['grad_clip'],error_if_nonfinite=True)
                before = {id(p):p.detach().clone() for p in parameters}; opt = time.perf_counter()
                scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
                optimizer_seconds = time.perf_counter()-opt
                if counter()!=old_counter+1: raise AssertionError('Finite backward failed to update optimizer once')
                changes = parameter_changes(groups,before); del before
                connected.update(n for n,p in net.named_parameters() if p.requires_grad and p.grad is not None)
                step += 1
                append('attempts',dict(attempt=attempt,status='OPTIMIZER_UPDATED',completed_updates=step,
                    optimizer_counter=counter(),scale_before=scale,scale_after=scaler.get_scale()))
                append('updates',dict(step=step,attempt=attempt,loss=float(loss.detach()),
                    loss_terms={k:float(v.detach()) for k,v in terms.items()},consistency=float(out.consistency.detach()),
                    gradient_observation=gradient,gradient_groups=gradient['norms'],actual_parameter_updates=changes,
                    gradient_before_clip=float(clip),forward_seconds=forward,backward_seconds=backward,
                    optimizer_seconds=optimizer_seconds,update_seconds=time.perf_counter()-began,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    RSS_bytes=process.memory_info().rss,sample_scores=len(out.scores),candidate_scores=sum(v.numel() for v in out.scores),
                    neural_execution={k:v-counts_before[k] for k,v in executed.items()},
                    original_bridge_embeddings={k:list(v.shape) for k,v in out.local_embeddings.items()},
                    original_second_view_embeddings={k:list(v.shape) for k,v in out.local_embeddings_view2.items()}))
                del out,loss,base,terms,clip
                bar.update(1); refresh(step)
                if step==2 or step%4==0 or step==a.updates: evaluate_step(step)
        finally:
            bar.close()
            for handle in handles: handle.remove()
        changes = parameter_changes(groups,initial_parameters); del initial_parameters
        expected = {n for n,p in net.named_parameters() if p.requires_grad}
        result.update(status='COMPLETE',completed_optimizer_updates=step,attempted_backward_passes=attempt,
            amp_overflow_skipped_attempts=sum(r['status']=='AMP_OVERFLOW_SKIPPED' for r in result['attempts']),
            finite_complete_updates=True,missing_gradients=sorted(expected-connected),
            connected_trainable_parameter_tensors=len(connected),expected_trainable_parameter_tensors=len(expected),
            final_parameter_changes=changes,every_core_group_updated=all(r['changed_parameter_tensors']>0 for r in changes.values()),
            every_core_group_had_nonzero_gradient=all(any(r['gradient_groups'][name]>0 for r in result['updates']) for name in groups),
            final_neural_sha256=neural_digest(net.state_dict()),initial_train=result['curve'][0]['train'],
            initial_held_out=result['curve'][0]['validation'],final_train=result['curve'][-1]['train'],
            final_held_out=result['curve'][-1]['validation'],matched_update_comparisons=matched_comparisons(result['curve'],baseline['curve']),
            reference_full_debug_curve=baseline['curve'],baseline_total_updates=len(baseline['updates']),
            process_peak_rss_bytes=peak[0],total_wall_seconds=time.perf_counter()-started,
            GPU_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in [*result['updates'],*result['support_refreshes']]),
            GPU_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in [*result['updates'],*result['support_refreshes']]),
            update_compute_samples_per_second=len(training)*step/sum(r['update_seconds'] for r in result['updates']),
            original_inputs_preserved=all(_sha(Path(p))==h for p,h in inputs.items()),
            original_source_preserved=all(_sha(source/p)==h for p,h in proof['verified_files'].items()),
            original_archive_preserved=_sha(ROOT/'versions/v1/pipeline_v1_source.zip')==proof['archive_sha256'],
            experiment_sources_preserved=all(_sha(ROOT/p)==h for p,h in result['current_sources_sha256'].items()),
            upper_dependencies_preserved=all(_sha(ROOT/p)==h for p,h in result['half_b']['source_sha256'].items()))
        with (root/'training_rng_audit_DEBUG.pt').open('xb') as stream:
            torch.save(dict(debug=True,production_ready=False,training_rng_before=training_rng,
                training_rng_after=capture_rng_state(),completed_optimizer_updates=step),stream)
        if result['missing_gradients'] or not all(result[k] for k in ('every_core_group_updated',
                'every_core_group_had_nonzero_gradient','original_inputs_preserved','original_source_preserved',
                'original_archive_preserved','experiment_sources_preserved','upper_dependencies_preserved')):
            result['status']='INCOMPLETE'; write_new(root/'incomplete_report.json',result)
            raise AssertionError('Core connectivity/update or preserved source/input audit failed')
        write_new(root/'report.json',result)
        return result
    finally:
        stop.set(); watcher.join()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--support-manifest',type=Path,required=True)
    p.add_argument('--source',type=Path)
    p.add_argument('--updates',type=int,required=True)
    p.add_argument('--cuda-gib',type=float,required=True)
    p.add_argument('--rss-gib',type=float,required=True)
    p.add_argument('--wall-seconds',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.gpu<0 or a.updates<2 or not all(math.isfinite(v) and v>0 for v in (a.cuda_gib,a.rss_gib,a.wall_seconds)):
        p.error('Explicit successful DEBUG updates>=2 and positive finite resource budgets required')
    baseline = json.loads(a.reference.read_text(encoding='utf8'))
    baseline_contract_path = a.reference.with_name('execution_contract.json')
    baseline_contract = json.loads(baseline_contract_path.read_text(encoding='utf8'))
    native_path = recorded_path(baseline_contract['arguments']['reference'])
    native_contract_path = native_path.with_name('execution_contract.json')
    manifest_path = a.fixture.with_name('fixture_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    protocol = verify_bounded_reference(baseline,baseline_contract,json.loads(native_path.read_text(encoding='utf8')),
        json.loads(native_contract_path.read_text(encoding='utf8')),manifest)
    extension = json.loads(a.support_manifest.read_text(encoding='utf8'))
    support_contract_path = a.support_manifest.with_name('execution_contract.json')
    support_contract = json.loads(support_contract_path.read_text(encoding='utf8'))
    raw_path = recorded_path(support_contract['arguments']['raw_index'])
    raw = json.loads(raw_path.read_text(encoding='utf8'))
    if raw.get('complete') is not True or raw.get('debug') is not False or _sha(raw_path)!=extension['raw_index_sha256']:
        raise ValueError('Verified actual training/validation inventory required')
    validate_support_manifest(extension,manifest,baseline,raw)
    a.support_fixture = a.support_manifest.with_name('support_fixture_DEBUG.pt')
    if (_sha(a.fixture)!=manifest['fixture_sha256'] or _sha(a.support_fixture)!=extension['support_fixture_sha256']
            or _sha(native_path)!=baseline['reference_report_sha256']
            or a.updates not in {r['step'] for r in baseline['curve']}):
        raise ValueError('Exact preserved native/query/support input and matched reference point required')
    for name,identity in baseline['current_sources_sha256'].items():
        if _sha(ROOT/name)!=identity: raise ValueError(f'Frozen baseline helper differs: {name}')
    for path,identity in extension['input_sha256'].items():
        if _sha(Path(path))!=identity: raise ValueError(f'Support preparation input changed: {path}')
    paths = (a.reference,baseline_contract_path,native_path,native_contract_path,a.fixture,manifest_path,
        a.support_manifest,a.support_fixture,support_contract_path,raw_path)
    inputs = {str(path.resolve()):_sha(path) for path in paths}
    if any(inputs.get(path)!=identity for path,identity in baseline_contract['inputs'].items()):
        raise ValueError('Original reference/fixture metadata differs')
    a.source = a.source or a.reference.parent/'source/v1.0'
    root = a.output.resolve()
    if root.exists(): raise FileExistsError('Preserve existing evidence: choose a fresh DEBUG output')
    if root.is_relative_to(a.source.resolve()) or any(root.is_relative_to(path.resolve().parent) for path in paths):
        raise ValueError('New output must be disjoint from all preserved inputs/source')
    root.mkdir(parents=True)
    contract = dict(debug=True,smoke_test=True,full_training=False,full_evaluation=False,
        production_ready=False,quality_verified=False,protocol=protocol,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},input_sha256=inputs,
        intended_change='original v1 L0 + v2.2 L1/L2/cosine scoring; complete original ranking+consistency only',
        explicit_DEBUG_differences=['one real extra train support-only case','selected support cohort3',
            'query fixture2train/1held reused','successful updates fixed explicitly; production40 unchanged'])
    contract['identity_sha256']=canonical_hash(contract); write_new(root/'execution_contract.json',contract)
    try:
        result = execute(a,root,baseline,manifest,protocol,extension,raw,inputs)
    except Exception:
        write_new(root/'failed.json',dict(debug=True,full_training=False,quality_verified=False,
            production_ready=False,error=traceback.format_exc()))
        raise
    print(f"REPORT: {root/'report.json'}",flush=True)
    print(json.dumps(dict(completed_optimizer_updates=result['completed_optimizer_updates'],
        initial_train=result['initial_train'],final_train=result['final_train'],
        initial_held_out=result['initial_held_out'],final_held_out=result['final_held_out'],
        full_training=False,quality_verified=False),allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
