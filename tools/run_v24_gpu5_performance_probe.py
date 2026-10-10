"""GPU5 DEBUG: complete native full128 numerical/performance probe.

Runs the original full CNN+GAT, physical patient batch4 and candidate chunk32,
allP+128U in both views for all four original calibration stress patients.
No optimizer updates; unchanged original checkpoint, AdamW, scaler and RNG
are verified before admitting the execution-only hash/prefetch/input adapters.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import warnings
from types import MethodType

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
CASES=('liver_6','liver_129','liver_123','liver_69')
NATIVE_GRID_OPERATOR='grid_sampler_3d_backward_cuda'


def parse_probe_options(argv):
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--variant',choices=('baseline','optimized'),required=True)
    parser.add_argument('--source-output',type=Path,required=True)
    parser.add_argument('--source-code',type=Path,required=True)
    parser.add_argument('--source-interruption-proof',type=Path)
    parser.add_argument('--probe-output',type=Path,required=True)
    parser.add_argument('--debug-performance-probe',action='store_true',required=True)
    policy=parser.add_mutually_exclusive_group()
    policy.add_argument('--debug-deterministic-numerics',action='store_true',
        help='DEBUG-only strict deterministic kernels; unsupported operators fail without fallback')
    policy.add_argument('--debug-native-grid-sampler-numerics',action='store_true',
        help='Explicit DEBUG native atomic grid backward; requires repeated original baseline admission')
    return parser.parse_known_args(argv)


def configure_debug_environment(enabled):
    """Called before importing Torch or initializing any CUDA context."""
    if type(enabled) is not bool:raise TypeError('Explicit DEBUG numerics flag required')
    if enabled:
        loaded=sys.modules.get('torch')
        if loaded is not None and loaded.cuda.is_initialized():
            raise RuntimeError('DEBUG CUBLAS workspace requires a fresh pre-CUDA process')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'


def numerical_flags(torch):
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory)


def configure_debug_numerics(torch,enabled,native_grid=False):
    """After original build_runtime/configure_runtime; never a production edit."""
    if type(enabled) is not bool or type(native_grid) is not bool:
        raise TypeError('Explicit DEBUG numerics flags required')
    enabled=enabled or native_grid
    before=numerical_flags(torch)
    if enabled:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
            raise ValueError('DEBUG CUBLAS workspace must be configured before CUDA initialization')
        torch.use_deterministic_algorithms(True,warn_only=native_grid)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    after=numerical_flags(torch)
    if any(before[key]!=after[key] for key in
        ('matmul_allow_tf32','cudnn_allow_tf32','float32_matmul_precision')):
        raise ValueError('DEBUG determinism must preserve original arithmetic precision')
    return dict(debug_deterministic_numerics=enabled,original_after_build_flags=before,
        actual_forward_flags=after,unsupported_operator_fallback=False,
        debug_native_grid_sampler_numerics=native_grid,
        admitted_native_nondeterministic_operator=NATIVE_GRID_OPERATOR if native_grid else None,
        production_config_modified=False,
        strict_parity_scope='same explicit DEBUG kernel policy; original production trajectory equivalence is not claimed')


def native_grid_warning_receipt(records):
    """Permit only the explicitly declared PyTorch native grid backward warning."""
    rows=[dict(category=record.category.__name__,message=str(record.message)) for record in records]
    for row in rows:
        if (row['category']!='UserWarning' or not row['message'].startswith(
                NATIVE_GRID_OPERATOR+' does not have a deterministic implementation, but you set ')
                or 'torch.use_deterministic_algorithms(True, warn_only=True)' not in row['message']):
            raise ValueError('Unexpected warning in native grid DEBUG policy: '+row['message'])
    if not rows:
        raise ValueError('Native grid DEBUG policy requires the actual declared operator warning')
    return dict(operator=NATIVE_GRID_OPERATOR,only_declared_operator=True,
                original_native_atomic_kernel=True,warning_count=len(rows),warnings=rows)


@contextmanager
def native_grid_warning_scope(enabled):
    receipt={}
    if not enabled:
        yield receipt
        return
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        yield receipt
        receipt.update(native_grid_warning_receipt(records))


class InputTensorObserver:
    """Read every actual CPU batch/upper tensor; return inputs unmodified."""
    def __init__(self,scorer,content_digest):
        self.scorer=scorer;self.digest=content_digest;self.lock=threading.Lock()
        self.local=[];self.upper=[];self.seconds=0.;self.references=[]
        self.original_geometry=scorer.geometry
        for partition,provider in scorer.providers.items():
            original=provider.get
            def observed(owner,ids,*,epoch=0,_original=original,_partition=partition):
                batch=_original(ids,epoch=epoch);started=time.perf_counter()
                payload=dict(graph=batch.graph.to_dict(),source_patches=batch.source_patches,
                    target_patches=batch.target_patches,source_index=batch.source_index,
                    graph_observation_index=batch.graph_observation_index,indices=batch.indices)
                row=dict(partition=_partition,epoch=epoch,ordered_indices=list(ids),
                    content_sha256=self.digest(payload),source_shape=list(batch.source_patches.shape),
                    target_shape=list(batch.target_patches.shape),
                    both_view_graphs=int(batch.graph.num_graphs),
                    local_nodes=sum(store.num_nodes for store in batch.graph.node_stores),
                    local_edges=sum(store.num_edges for store in batch.graph.edge_stores))
                if batch.indices.tolist()!=list(ids) or row['both_view_graphs']!=2*len(ids):
                    raise ValueError('Observed native batch lost actual ordered indices/two views')
                with self.lock:self.local.append(row);self.seconds+=time.perf_counter()-started
                return batch
            self.references.append((provider,original,'get' in vars(provider)))
            provider.get=MethodType(observed,provider)
        observer=self
        class GeometryObserver:
            def __getattr__(self,name):return getattr(observer.original_geometry,name)
            def __call__(self,plan,provider=None):
                value=observer.original_geometry(plan,provider);started=time.perf_counter()
                graph,prototype,audit=value
                row=dict(case_id=plan.case_id,record_ids=list(plan.record_ids),
                    content_sha256=observer.digest((graph.to_dict(),prototype.to_dict())))
                with observer.lock:
                    observer.upper.append(row);observer.seconds+=time.perf_counter()-started
                return value
        self.geometry=GeometryObserver();scorer.geometry=self.geometry

    def proof(self,plans,*,epoch,physical_candidate_chunk):
        if not plans or type(physical_candidate_chunk)is not int or physical_candidate_chunk<1:
            raise ValueError('Complete plans and explicit native candidate chunk required')
        partition=plans[0].partition;provider=self.scorer.providers[partition]
        if any(plan.partition!=partition for plan in plans):
            raise ValueError('Probe plans must use one unchanged partition')
        lookup={row['id']:i for i,row in enumerate(provider.ds.rows)}
        expected=[lookup[record_id] for plan in plans for record_id in plan.record_ids]
        observed=[index for row in self.local for index in row['ordered_indices']]
        schedule=[expected[start:start+physical_candidate_chunk]
            for start in range(0,len(expected),physical_candidate_chunk)]
        if (observed!=expected or any(row['partition']!=partition or row['epoch']!=epoch
                for row in self.local) or [row['ordered_indices'] for row in self.local]!=schedule):
            raise ValueError('Actual CPU tensor proof does not cover complete ordered native candidates')
        if [(row['case_id'],row['record_ids']) for row in self.upper]!=[
            (plan.case_id,list(plan.record_ids)) for plan in plans]:
            raise ValueError('Actual CPU upper proof does not cover complete patient graphs')
        content=dict(local_chunks=self.local,upper_graphs=self.upper)
        return dict(format='v24_actual_full_CPU_input_tensor_proof_v1',complete=True,
            every_actual_value_hashed=True,hash_memoization=False,ordered_records=len(expected),
            both_sampled_views=True,epoch=epoch,physical_candidate_chunk=physical_candidate_chunk,
            observed_native_chunks=len(self.local),expected_native_chunks=len(schedule),
            observed_upper_graphs=len(self.upper),expected_upper_graphs=len(plans),
            content=content,content_sha256=self.digest(content),
            observation_wall_seconds=self.seconds,
            observation_overhead_included_in_total_step_seconds=True,
            no_raw_CT_or_input_tensor_files_written=True)

    def close(self):
        if self.scorer.geometry is not self.geometry:raise ValueError('Probe geometry observer changed')
        self.scorer.geometry=self.original_geometry
        for provider,original,had_instance in self.references:
            if had_instance:provider.get=original
            else:del provider.get


def _sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def main(argv=None):
    options,remaining=parse_probe_options(argv)
    configure_debug_environment(options.debug_deterministic_numerics or options.debug_native_grid_sampler_numerics)
    from tools.run_v24_all_p import parse,validate_config
    args=parse(remaining)
    if args.mode!='train' or args.gpu!=5:
        raise ValueError('Original GPU5 full training config required for diagnostic')
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    import psutil
    process=psutil.Process()
    if len(process.cpu_affinity())!=4:raise ValueError('Original four logical CPU IDs required')
    import torch
    torch.set_num_threads(1)
    from tools.run_v24_gpu5_performance_continuation import admit
    admission=admit(args,options.source_output,options.source_code,options.source_interruption_proof)
    from hiercp_v1x.v24_training_continuation import inspect_continuation
    source=inspect_continuation(args.output,options.source_code)
    if source['latest']['status'] not in ('PAUSED','RUNNING'):
        raise ValueError('Exact admitted paused or interrupted snapshot required for diagnostic')
    if source['latest']['status']=='RUNNING' and options.source_interruption_proof is None:
        raise ValueError('Interrupted source requires explicit verified interruption proof')
    owner=json.loads((args.output/'training/training_identity.json').read_text())
    saved=torch.load(args.output/'training/checkpoint_latest.pt',map_location='cpu',weights_only=False)
    calibration=json.loads((args.output/'calibration.json').read_text())
    physical=calibration['selected_physical_patient_batch'];chunk=calibration['selected_physical_candidate_batch']
    trial=[row for row in calibration['trials'] if row.get('accepted')
           and row['physical_patient_batch']==physical and row['physical_candidate_batch']==chunk]
    if len(trial)!=1 or tuple(trial[0]['case_ids'])!=CASES or physical!=4 or chunk!=32:
        raise ValueError('Original measured maximum-cost full128 diagnostic batch differs')
    config=validate_config(json.loads(args.config.read_text()),args.gpu,args.stunet_checkpoint)
    from hiercp_v1x import v24_hash_runtime as hashes
    hash_receipt=hashes.install() if options.variant=='optimized' else None
    from hiercp_v1x import v24_factory,v24_memory_runtime as memory,v24_prefetch_runtime as prefetch
    from hiercp_v1x.v24_preparation_runtime import install_runtime
    install_runtime(v24_factory);memory.install_memory_runtime(v24_factory)
    prefetch_receipt=prefetch.install_runtime(memory) if options.variant=='optimized' else None
    input_runtime=None;input_runtime_receipt=None
    if options.variant=='optimized':
        from hiercp_v1x import v24_input_runtime as input_runtime
        input_runtime_receipt=input_runtime.install_runtime(pin_final_outputs=True)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Real singleton CUDA required, no fallback')
    properties=torch.cuda.get_device_properties(0);free,total=torch.cuda.mem_get_info()
    if 'A6000' not in properties.name or free<=40*2**30:
        raise ValueError('Original40GiB free A6000 budget required')
    torch.cuda.set_per_process_memory_fraction(40*2**30/total)
    from hiercp_v1x.historical_evaluation import ResourceBudget
    from hiercp_v1x.u_bridge_training import capture_rng,restore_rng,digest,gradient_receipt,_groups
    from hiercp_v1x.v24_training import optimizer_groups,_plans,_positive_indices,patient_balanced_objective
    budget=ResourceBudget(40*2**30,64*2**30)
    options.probe_output.mkdir(parents=True,exist_ok=False)
    net,scorer,population,training_config,contract,close=v24_factory.build_runtime(args,config,budget)
    observer=None;debug_numerics=None
    try:
        # The original factory has just reset cudnn/precision config. Apply the
        # explicit DEBUG policy now, before the actual diagnostic forward.
        debug_numerics=configure_debug_numerics(torch,options.debug_deterministic_numerics,
                                               options.debug_native_grid_sampler_numerics)
        memory.bind_memory_runtime(scorer)
        if options.variant=='optimized':prefetch.bind(scorer)
        if input_runtime is not None:input_runtime_receipt=input_runtime.bind(scorer)
        net.load_state_dict(saved['model'],strict=True);net.train()
        scorer.physical_candidate_batch=chunk
        optimizer=torch.optim.AdamW(optimizer_groups(net,training_config['training']))
        optimizer.load_state_dict(saved['optimizer'])
        expected_parameters={id(value) for value in net.parameters() if value.requires_grad}
        optimizer_parameters=[id(value) for group in optimizer.param_groups for value in group['params']]
        if (len(expected_parameters)!=981 or set(optimizer_parameters)!=expected_parameters
                or len(optimizer_parameters)!=len(expected_parameters)):
            raise ValueError('Every original981 trainable parameter must occur exactly once in AdamW')
        initial_optimizer_sha256=digest(optimizer.state_dict())
        scaler=torch.amp.GradScaler('cuda',enabled=training_config['training']['amp'])
        scaler.load_state_dict(saved['scaler'])
        initial_scaler_sha256=digest(scaler.state_dict())
        restore_rng(saved['rank_rng'][0])
        before_model=digest(net.state_dict());before_rng=digest(capture_rng())
        if (before_model!=source['latest']['numerical_state_sha256']['model']
                or initial_optimizer_sha256!=source['latest']['numerical_state_sha256']['optimizer']
                or initial_scaler_sha256!=source['latest']['numerical_state_sha256']['scaler']
                or before_rng!=digest(saved['rank_rng'][0])):
            raise ValueError('Actual loaded model/AdamW/scaler/RNG must equal the original checkpoint values')
        before_providers={key:provider.profile() for key,provider in scorer.providers.items()}
        plans=_plans(population,list(CASES),128)
        if len(plans)!=4 or any(plan.active_u_count!=128 for plan in plans):
            raise ValueError('Complete actual four-patient allP+128U plans required')
        observer=InputTensorObserver(scorer,hashes.tensor_digest)
        optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        # Epoch1 matches the original full128 maximum-cost calibration's seeds.
        with native_grid_warning_scope(options.debug_native_grid_sampler_numerics) as native_warnings:
            result=scorer(plans,epoch=1,training=True)
            loss,terms=patient_balanced_objective(result.scores,[_positive_indices(plan) for plan in plans],result.consistency)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite native diagnostic loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        gradient=gradient_receipt(net,_groups(net))
        if gradient['trainable_parameter_tensors']!=981 or gradient['gradient_present']!=981:
            raise ValueError('All981 original GPU5 named trainable gradients are required')
        after_optimizer_sha256=digest(optimizer.state_dict())
        after_scaler_sha256=digest(scaler.state_dict())
        if (initial_optimizer_sha256!=after_optimizer_sha256
                or initial_scaler_sha256!=after_scaler_sha256):
            raise ValueError('DEBUG probe changed persisted AdamW or scaler state without authorization')
        if not gradient['finite'] or gradient['missing']:
            raise FloatingPointError('Native diagnostic gradient nonfinite or disconnected: '+repr(gradient))
        budget.check()
        input_proof=observer.proof(plans,epoch=1,physical_candidate_chunk=chunk)
        actual_flags=numerical_flags(torch)
        if actual_flags!=debug_numerics['actual_forward_flags']:
            raise ValueError('Explicit DEBUG numerical kernel policy changed during native forward')
        tensors=dict(scores=tuple(score.detach().cpu() for score in result.scores),
            consistency=result.consistency.detach().cpu(),loss=loss.detach().cpu(),
            gradients={name:parameter.grad.detach().cpu() for name,parameter in net.named_parameters()
                if parameter.requires_grad})
        tensor_file=options.probe_output/'numerical_tensors.pt'
        with tensor_file.open('xb') as stream:torch.save(tensors,stream)
        proof=dict(debug=True,diagnostic='actual_full128_forward_loss_backward_no_optimizer_update',
            variant=options.variant,source_checkpoint=source['latest'],
            original_training_identity_sha256=owner['identity_sha256'],
            initial_model_sha256=before_model,initial_RNG_sha256=before_rng,
            GPU_arm=5,original_named_trainable_parameter_count=981,
            initial_optimizer_sha256=initial_optimizer_sha256,
            after_optimizer_sha256=after_optimizer_sha256,
            initial_scaler_sha256=initial_scaler_sha256,after_scaler_sha256=after_scaler_sha256,
            optimizer_contains_all_trainable_parameters_exactly_once=True,
            scheduler_and_shuffle_state_unchanged=True,
            GPU5_probe_source_sha256=_sha(__file__),
            native_grid_sampler_warning_receipt=native_warnings,
            output_sha256=digest(tuple(score.detach().cpu() for score in result.scores)),
            loss=float(loss.detach()),gradient_sha256=digest({name:parameter.grad.detach().cpu()
                for name,parameter in net.named_parameters() if parameter.requires_grad}),
            after_forward_model_sha256=digest(net.state_dict()),after_RNG_sha256=digest(capture_rng()),
            case_ids=list(CASES),active_U=128,all_P_included=True,
            physical_patient_batch=physical,physical_candidate_chunk=chunk,gradient_accumulation=1,
            total_ranking_train_patients=65,actual_probe_patients=4,production_patient_subset_used=False,
            production_optimizer_updates=0,production_calibration_repeats=3,diagnostic_batches=1,
            seconds=seconds,workload=result.workload,gradient=gradient,
            peak_CUDA_bytes=torch.cuda.max_memory_allocated(),RSS_bytes=process.memory_info().rss,
            CPU_affinity=process.cpu_affinity(),GPU=properties.name,
            provider_profiles_before=before_providers,
            provider_profiles_after={key:provider.profile() for key,provider in scorer.providers.items()},
            hash_runtime=hash_receipt,prefetch_runtime=prefetch_receipt,
            input_runtime=input_runtime_receipt,
            input_runtime_profile=input_runtime.profile() if input_runtime is not None else None,
            debug_numerics=debug_numerics,input_tensor_proof=input_proof,
            input_tensor_proof_sha256=input_proof['content_sha256'],
            ordered_CPU_input_sha256=input_proof['content_sha256'],
            numerical_tensor_file=dict(path=str(tensor_file),raw_sha256=_sha(tensor_file),
                content_sha256=digest(tensors),size=tensor_file.stat().st_size,
                all_named_trainable_gradients=len(tensors['gradients']),no_model_update=True,
                server_only=True,no_raw_CT=True),
            model_contract=contract,full_training_or_evaluation_completion_claimed=False)
        if inspect_continuation(args.output,options.source_code)!=source:
            raise ValueError('Diagnostic changed the exact preserved continuation snapshot')
        with (options.probe_output/'result.json').open('x') as stream:json.dump(proof,stream,indent=2,allow_nan=False)
        print(json.dumps(dict(variant=options.variant,seconds=seconds,loss=proof['loss'],
            result=str(options.probe_output/'result.json'),peak_CUDA_bytes=proof['peak_CUDA_bytes'],
            exposed_CPU_wait=result.workload.get('CPU_timing',{}).get('cpu_input_exposed_wait_seconds'))),flush=True)
    except Exception as error:
        with (options.probe_output/'failure.json').open('x') as stream:
            json.dump(dict(debug=True,variant=options.variant,error_type=type(error).__name__,
                error=str(error),debug_numerics=debug_numerics,
                unsupported_operator_fallback=False,production_optimizer_updates=0,
                source_checkpoint=source['latest']),stream,indent=2,allow_nan=False)
        raise
    finally:
        try:
            if observer is not None:observer.close()
        finally:close()


if __name__=='__main__':main()
