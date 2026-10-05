"""Execute ONE complete crossed arm, with its own bound restart and full128 audit.

Production uses the completed v1-m10 publication and actual native inventory.
DEBUG uses explicit real-CT source/curriculum fixtures and selected whole cases.
No prior experiment is overwritten, no original A/B is automatically retrained.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def prepare_source(source):
    # Current helpers are explicitly resolved from this worktree; only the
    # frozen hiercp namespace is activated. No snapshot file is changed.
    import tools
    if str(ROOT/'tools') not in tools.__path__:tools.__path__.append(str(ROOT/'tools'))
    from hiercp_v1x.scope_probe_support import activate_original
    proof=activate_original(source)
    from hiercp_v1x import bounded_scope
    scope=bounded_scope.install(10,expected_snapshot_root=source)
    return proof,scope


class NativeRows:
    """Explicit whole-case DEBUG selection; production never selects cases."""
    def __init__(self,index,partition,debug,cases=None):
        from l0_local_cnn.data import Dataset
        native=Dataset(index,partition,debug)
        self.__dict__.update(native.__dict__)
        if cases is not None:
            if not debug or not set(cases)<=set(self.meta['split'][partition]):
                raise ValueError('Case selection is DEBUG only and must preserve its split')
            self.rows=[r for r in self.rows if r['case_id'] in cases]
            raw={r['case_id']:r for r in self.meta['raw_records']}
            for case in cases:
                rows=[r for r in self.rows if r['case_id']==case]
                if sum(r['target']==0 for r in rows)!=128 or sum(r['target']==1 for r in rows)!=len(raw[case]['positives']):
                    raise ValueError('DEBUG selected case must retain all P plus128U')
    def __len__(self):return len(self.rows)


class NativeRowsLoader:
    def __init__(self,ds,workers,resident,rss):
        from l0_local_cnn.data import Loader
        self.ds=ds;self.inner=Loader(ds,workers,resident,rss)
    def get(self,ids,*,epoch=0):return self.inner.get(ids)
    def batches(self,schedule,*,epoch=0):return self.inner.batches(schedule)


def record_model(output,net,arm,base,train_ds,val_ds,*,workers,debug,candidates,own_samples=None,graph_cost=None):
    """Report the constructed model and actual cohort before neural execution."""
    from hiercp_v1x.transition_runtime import append
    from hiercp_v1x.transition_native_local import native_transition_spec
    rows=train_ds.rows+val_ds.rows
    report=dict(stage='constructed_crossed_model',arm=arm,debug=debug,
        model_name=type(net).__name__,local_model_name=type(net.local).__name__,
        total_parameters=sum(p.numel() for p in net.parameters()),
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
        local_architecture=native_transition_spec() if arm=='C' else base['model'],
        upper=dict(hidden_dim=128,heads=4,L1_layers=2,L2_layers=2,output_labels=2),
        input_shape_contract='native-variable donor/recipient crops, N x 1 x D x H x W' if arm=='C' else 'two original views, dense patches N x 5 x 48 x 48 x 48',
        actual_common_train_observations=len(train_ds),actual_common_validation_observations=len(val_ds),
        full_native_observations=14102,common_used_fraction=len(rows)/14102,
        own_curriculum_samples=own_samples,actual_graph_cost_epoch0=graph_cost,
        sampling_ratio='original sampled-view rule, no additional reduction',time_window='not a temporal model',
        physical_batch_candidates=candidates,gradient_accumulation_steps=1,data_parallel_workers=1,
        effective_batch='selected measured physical batch; no accumulation',epochs=1 if debug else 40,
        precision='original AMP and GradScaler' if arm=='C' else 'native FP32',
        worker_count=workers,preparation_and_input_cache='verified resident/mmap with prefetch',
        peak_VRAM_and_throughput='measured in calibration/update receipts, unavailable before execution',
        full_training=False,full_evaluation=False,quality_verified=False)
    append(Path(output)/'model_execution.jsonl',report)
    print(json.dumps(report,allow_nan=False),flush=True)


def inputs(a,source):
    """Read real signed original curriculum inputs; no synthesized GT/centers."""
    import torch
    if a.debug:
        if not a.debug_original_fixture or not a.debug_support_fixture:
            raise ValueError('DEBUG requires explicit actual original/support fixture paths')
        fixture=torch.load(a.debug_original_fixture,map_location='cpu',weights_only=False)
        extra=torch.load(a.debug_support_fixture,map_location='cpu',weights_only=False)
        if extra.get('debug') is not True or fixture['config']!=extra['config']:
            raise ValueError('Real DEBUG fixture original configurations differ')
        train=[s for s in fixture['samples'] if s['split']=='train']+extra['bounded_samples']
        val=[s for s in fixture['samples'] if s['split']=='val']
        config=copy.deepcopy(fixture['config'])
        return train,val,config,dict(debug=True,fixture_sha256=sha(a.debug_original_fixture),
                                     support_fixture_sha256=sha(a.debug_support_fixture))
    from hiercp_v1x.half_a_training import baseline_proof
    native,proof=baseline_proof(a.baseline)
    signature=proof['neural_baseline']['training_signature'];cache=a.baseline/'shared/cache'
    train=[cache/name for name in signature['train_cache_files']]
    val=[cache/name for name in signature['val_cache_files']]
    index=json.loads((cache/'index.json').read_text(encoding='utf8'))
    rows=index['entries'];expected={r['path']:r for r in rows}
    if set(expected)!=set(signature['train_cache_files']+signature['val_cache_files']):
        raise ValueError('All original signed materialized sample files must be retained')
    signed={str((cache/name).resolve()):expected[name]['artifact_sha256']
            for name in signature['train_cache_files']+signature['val_cache_files']}
    actual_train=sorted({expected[p.name]['case_id'] for p in train})
    actual_val=sorted({expected[p.name]['case_id'] for p in val})
    def sample_receipts(paths):
        return [dict(case_id=expected[p.name]['case_id'],sample_index=expected[p.name]['sample_index'],
                     path=p.name,sha256=expected[p.name]['artifact_sha256']) for p in paths]
    from hiercp_v1x.contracts import canonical_hash
    return train,val,copy.deepcopy(native['config']),dict(debug=False,baseline=proof,
                training_case_ids=native['split']['train'],validation_case_ids=native['split']['val'],
                actual_signed_train_cases=actual_train,actual_signed_validation_cases=actual_val,
                original_sample_sha256=signed,
                actual_signed_train_samples=sample_receipts(train),actual_signed_validation_samples=sample_receipts(val),
                baseline_proof_sha256=canonical_hash(proof),cache_index_sha256=sha(cache/'index.json'),
                configured_but_not_materialized_validation_cases=sorted(set(native['split']['val'])-set(actual_val)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['C','D'],required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--native-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,required=True)
    p.add_argument('--physical-batch-candidates',type=int,nargs='+',required=True,
                   help='C: complete eight-candidate samples; D: observation rows, each two real graphs')
    p.add_argument('--prepared-cache',type=Path,
                   help='D only: read-only reuse of a complete byte/assignment/source-bound canonical index')
    p.add_argument('--reuse-preparation',type=Path,
                   help='D only: import verified completed receipts from an interrupted preparation into a new arm output')
    for name in ('cuda-gib','rss-gib','resident-gib'):p.add_argument('--'+name,type=float,required=True)
    p.add_argument('--debug',action='store_true');p.add_argument('--debug-updates',type=int)
    p.add_argument('--debug-original-fixture',type=Path);p.add_argument('--debug-support-fixture',type=Path)
    p.add_argument('--debug-train-cases',nargs='+');p.add_argument('--debug-validation-cases',nargs='+')
    a=p.parse_args()
    if (a.prepared_cache is not None or a.reuse_preparation is not None) and a.arm!='D':
        raise ValueError('Prepared canonical graphs belong only to D original input')
    if a.prepared_cache is not None and a.reuse_preparation is not None:
        raise ValueError('Choose a complete prepared cache or interrupted preparation reuse, not both')
    if (a.workers<2 or not 0<a.resident_gib<a.rss_gib or a.cuda_gib<=0
            or a.physical_batch_candidates!=sorted(set(a.physical_batch_candidates))
            or min(a.physical_batch_candidates)<(1 if a.arm=='C' else 2)):
        raise ValueError('Explicit parallel/resource/physical batch candidates required')
    if not a.debug and any((a.debug_updates,a.debug_original_fixture,a.debug_support_fixture,
                            a.debug_train_cases,a.debug_validation_cases)):
        raise ValueError('Production cannot accept a DEBUG subset/update/fixture setting')
    if a.debug and (not a.debug_updates or not a.debug_train_cases or not a.debug_validation_cases):
        raise ValueError('DEBUG requires explicit successful updates and whole train/validation cases')
    a.output=a.output.resolve();a.baseline=a.baseline.resolve(strict=True);a.native_run=a.native_run.resolve(strict=True)
    for old in (a.baseline,a.native_run):
        if a.output==old or a.output.is_relative_to(old) or old.is_relative_to(a.output):
            raise ValueError('New crossed results must be disjoint from original experiments')
    from tools.local_cnn_device import select
    select(a.gpu)
    import torch,numpy as np,psutil
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one selected actual CUDA device required')
    total=torch.cuda.get_device_properties(0).total_memory
    source=a.baseline/'source/v1.0';proof,scope=prepare_source(source)
    from hiercp.preparation_runtime import snapshot
    allocation=snapshot();free,_=torch.cuda.mem_get_info()
    if a.workers>allocation['cpu_capacity']:
        raise ValueError('Explicit workers exceed the effective affinity/cgroup/scheduler CPU allocation')
    if a.rss_gib*2**30>allocation['rss_bytes']+allocation['available_memory_bytes']:
        raise ValueError('RSS budget exceeds current host/cgroup RAM availability; no silent resource migration')
    if a.cuda_gib*2**30>free:
        raise ValueError('Requested CUDA budget exceeds currently free selected-device VRAM')
    if a.cuda_gib*2**30>=total:
        raise ValueError(f'Explicit CUDA budget {a.cuda_gib}GiB must leave headroom in actual {total/2**30:.3f}GiB device')
    torch.cuda.set_per_process_memory_fraction(a.cuda_gib*2**30/total)
    from hiercp_v1x.transition_runtime import WholeCandidateEvaluation,train_D,append
    from hiercp_v1x.transition_model import CrossedModel,tensor_hash
    from l0_regions.training_data import Budget
    from hiercp_v1x.contracts import canonical_hash
    from tools.run_local_cnn_experiment import exclusive,write_json
    budget=Budget(int(a.cuda_gib*2**30),int(a.rss_gib*2**30))
    inventory_path=a.native_run/'inventory/index.json'
    inventory=json.loads(inventory_path.read_text(encoding='utf8'))
    if inventory.get('debug')!=a.debug or inventory.get('local_cnn',{}).get('margin_mm')!=10:
        raise ValueError('Bound actual native inventory must match DEBUG/full and10mm')
    native_binding=dict(debug=a.debug,inventory_sha256=sha(inventory_path),weights_transferred=False)
    if not a.debug:
        state_path=a.native_run/'experiment.json'
        state=json.loads(state_path.read_text(encoding='utf8'))
        request=state.get('request',{})
        if (state.get('format')!='local_cnn_experiment_v1'
                or request.get('original_inventory_sha256')!=inventory['original_inventory_sha256']
                or request.get('local_cnn')!=inventory['local_cnn']
                or inventory['config']['gnn_epochs']!=40):
            raise ValueError('Actual native experiment recipe is not bound to this full40 ten-mm inventory')
        from hiercp_v1x.transition_reference import admit_native_reference
        native_binding=admit_native_reference(a.native_run,inventory_path=inventory_path)
    from hiercp_v1x.transition_evaluation import validate_cohort
    validate_cohort(inventory,case_ids=a.debug_validation_cases if a.debug else None,debug=a.debug)
    torch.set_num_threads(a.workers);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    from hiercp.tensor import configure_runtime
    configure_runtime(deterministic=inventory['base']['runtime']['deterministic'],
                      allow_tf32=inventory['base']['runtime']['allow_tf32'],
                      cudnn_benchmark=inventory['base']['runtime']['cudnn_benchmark'])
    packages=('hiercp_v1x','hiercp_v22','hiercp_v222','l0_regions','l0_local_cnn','l0_exploration','l0_sage','l0_ezsp')
    tracked=subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'ls-files','--',*packages],
        cwd=ROOT,check=True,capture_output=True,text=True).stdout.splitlines()
    source_paths={ROOT/name for name in tracked if name.endswith('.py')}
    source_paths.update(ROOT.glob('hiercp_v1x/transition_*.py'))
    source_paths.update((Path(__file__),ROOT/'tools/__init__.py',ROOT/'tools/prepare_v17_transition_debug.py',ROOT/'config/v17_crossed_training.json'))
    # Bind the tools imported by these packages, recursively. Unrelated visual
    # renderers are not execution dependencies or implicit training revisions.
    pending=[path for path in source_paths if path.suffix=='.py'];scanned=set()
    while pending:
        path=pending.pop()
        if path in scanned:continue
        scanned.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))):
            names=[]
            if isinstance(node,ast.ImportFrom) and node.module:
                if node.module.startswith('tools.'):names.append(node.module)
                elif node.module=='tools':names.extend('tools.'+alias.name for alias in node.names)
            elif isinstance(node,ast.Import):names.extend(alias.name for alias in node.names if alias.name.startswith('tools.'))
            for name in names:
                dependency=ROOT/'tools'/(name.split('.')[1]+'.py')
                if dependency.is_file() and dependency not in source_paths:
                    source_paths.add(dependency);pending.append(dependency)
    sources={path.relative_to(ROOT).as_posix():sha(path) for path in sorted(source_paths)}
    if not a.debug:
        subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'diff','--exit-code','HEAD','--',
                        *sources.keys()],cwd=ROOT,check=True,capture_output=True,text=True)
    identity=dict(format='v17_crossed_training_identity_v1',arm=a.arm,debug=a.debug,
                  scope=scope['contract_sha256'],native_inventory_sha256=sha(inventory_path),
                  native_experiment_binding=native_binding,
                  source=proof,sources=sources,settings={k:getattr(a,k) for k in
                    ('workers','physical_batch_candidates','cuda_gib','rss_gib','resident_gib',
                     'debug_updates','debug_train_cases','debug_validation_cases')})
    props=torch.cuda.get_device_properties(0)
    identity['hardware']=dict(name=props.name,total_memory=props.total_memory,compute_capability=[props.major,props.minor],
        multiprocessors=props.multi_processor_count,torch=torch.__version__,cuda_build=torch.version.cuda)
    if a.prepared_cache is not None:
        a.prepared_cache=a.prepared_cache.resolve(strict=True)
        identity['reused_prepared_cache']=dict(path=str(a.prepared_cache),sha256=sha(a.prepared_cache),read_only=True)
    if a.reuse_preparation is not None:
        a.reuse_preparation=a.reuse_preparation.resolve(strict=True)
        identity['reused_preparation']=dict(path=str(a.reuse_preparation),
            request_sha256=sha(a.reuse_preparation/'prepare_request.json'),read_only=True)
    train_files,val_files,config,binding=inputs(a,source)
    if not a.debug and (set(binding['training_case_ids'])!=set(inventory['split']['inner_train'])
                        or set(binding['validation_case_ids'])!=set(inventory['split']['inner_val'])):
        raise ValueError('Original and native84/21 patient identities differ; no data split substitution')
    identity['original_curriculum_binding']=binding
    resources=dict(GPU=torch.cuda.get_device_name(),visible_GPUs=1,total_VRAM_bytes=total,
                   free_VRAM_bytes=free,effective_allocation=allocation,
                   CPU_logical=psutil.cpu_count(),available_RAM_bytes=psutil.virtual_memory().available,
                   cuda_limit_bytes=budget.cuda_bytes,rss_limit_bytes=budget.rss_bytes,
                   physical_batch_unit='eight-candidate samples' if a.arm=='C' else 'observation rows;2graphs/row')
    with exclusive(a.output):
        manifest=a.output/'manifest.json'
        if manifest.exists():
            from hiercp_v1x.transition_reference_identity import bind_manifest_reference, compare_reference
            previous=json.loads(manifest.read_text(encoding='utf8'))
            current_reference=identity['native_experiment_binding']
            identity=bind_manifest_reference(previous,identity)
            comparison=compare_reference(previous['native_experiment_binding'],current_reference)
            if not comparison['exact']:
                # The experiment/checkpoint identity stays frozen at creation.
                # A live reference admission is provenance, never trained state
                # to import, a replacement baseline, or a change to D geometry.
                append(a.output/'reference_admissions.jsonl',dict(
                    format='v17_reference_endpoint_observation_v1',comparison=comparison,
                    current_admission=current_reference,experiment_identity_preserved=True,
                    model_or_optimizer_imported=False))
                print('Reference checkpoint advanced; crossed identity and training state preserved.',flush=True)
        else:write_json(manifest,identity)
        append(a.output/'resources.jsonl',resources);print(json.dumps(resources),flush=True)
        if (a.output/'training/training_complete.json').exists():
            complete=json.loads((a.output/'training/training_complete.json').read_text())
            if (complete.get('arm')!=a.arm or complete.get('identity')!=identity or complete.get('debug') is not False
                    or complete.get('full_training') is not True or complete.get('full_evaluation') is not True
                    or complete.get('completed_epochs')!=40
                    or sha(a.output/'training/checkpoint_latest.pt')!=complete.get('checkpoint_sha256')
                    or sha(complete['evaluation_artifact'])!=complete.get('evaluation_sha256')):
                raise ValueError('Existing completion marker is not bound to this complete arm/checkpoint/full21 evaluation')
            print(f'COMPLETE: {a.output}; no retraining',flush=True);return
        resident=int(a.resident_gib*2**30)
        if a.arm=='C':
            from hiercp_v1x.transition_native_local import NativeCurriculumLoader,make_native_local
            from hiercp_v1x.transition_c_training import train_c,calibrate_c
            # These loaders use only real raw CT and the exact source/centers
            # read from all signed original cache samples.
            train_cases={s['case_id'] for s in train_files} if a.debug else set(binding['actual_signed_train_cases'])
            val_cases={s['case_id'] for s in val_files} if a.debug else set(binding['actual_signed_validation_cases'])
            train_raw=[r for r in inventory['raw_records'] if r['case_id'] in train_cases]
            val_raw=[r for r in inventory['raw_records'] if r['case_id'] in val_cases]
            kwargs=dict(workers=a.workers,resident_bytes=resident,rss_bytes=budget.rss_bytes,debug=a.debug,pin_memory=True)
            train_loader=NativeCurriculumLoader(train_files,train_raw,config,sample_sha256=binding.get('original_sample_sha256'),**kwargs)
            val_loader=NativeCurriculumLoader(val_files,val_raw,config,sample_sha256=binding.get('original_sample_sha256'),**kwargs)
            local=make_native_local(checkpointing=True,budget=budget,dropout=.1)
            net=CrossedModel(local,arm='C',scope_contract=scope['contract_sha256'],dropout=.1,debug=a.debug).cuda()
            train_ds=NativeRows(inventory_path,'inner_train',a.debug,a.debug_train_cases)
            val_ds=NativeRows(inventory_path,'inner_val',a.debug,a.debug_validation_cases)
            record_model(a.output,net,'C',config,train_ds,val_ds,workers=a.workers,debug=a.debug,
                candidates=a.physical_batch_candidates,own_samples=dict(train=len(train_loader),validation=len(val_loader)))
            common_train=NativeRowsLoader(train_ds,a.workers,resident,budget.rss_bytes)
            common_val=NativeRowsLoader(val_ds,a.workers,resident,budget.rss_bytes)
            config['transition_runtime']=dict(physical_batch_size=min(a.physical_batch_candidates),
                support_batch_size=min(a.physical_batch_candidates),validation_batch_size=min(a.physical_batch_candidates),
                workers=a.workers,device='cuda:0',debug_curriculum_epoch=29 if a.debug else None,
                training_case_ids=a.debug_train_cases if a.debug else inventory['split']['inner_train'],
                validation_case_ids=a.debug_validation_cases if a.debug else inventory['split']['inner_val'],
                actual_signed_train_cases=a.debug_train_cases if a.debug else binding['actual_signed_train_cases'],
                actual_signed_validation_cases=a.debug_validation_cases if a.debug else binding['actual_signed_validation_cases'])
            if not a.debug:
                for key in ('actual_signed_train_samples','actual_signed_validation_samples','baseline_proof_sha256','cache_index_sha256'):
                    config['transition_runtime'][key]=binding[key]
            checkpoint=a.output/'training/checkpoint_latest.pt'
            if checkpoint.exists():
                contract=json.loads((a.output/'training/c_training_contract.json').read_text())
                runtime=contract['configuration']['transition_runtime'];config['transition_runtime'].update(runtime)
                config['transition_runtime']['resume_checkpoint']=str(checkpoint)
            else:
                selected,reports=calibrate_c(net,train_loader,config,a.physical_batch_candidates,budget=budget,debug=a.debug)
                for key in ('physical_batch_size','support_batch_size','validation_batch_size'):config['transition_runtime'][key]=selected
                append(a.output/'calibration.jsonl',dict(selected=selected,reports=reports))
            evaluator=WholeCandidateEvaluation(inventory,train_ds,common_train,val_ds,common_val,
                batch=config['transition_runtime']['physical_batch_size']*8,budget=budget,debug=a.debug)
            def common(model,**kwargs):
                path=kwargs.get('output')
                return evaluator(model,output=None if path is None else Path(path)/f"common128_epoch_{kwargs['epoch']:03d}_step_{kwargs['update']}.json")
            result=train_c(net,train_loader,val_loader,config,output=a.output/'training',budget=budget,
                debug=a.debug,debug_updates=a.debug_updates,common_evaluate=common)
            if result['status']=='COMPLETE' and not a.debug:
                best=torch.load(a.output/'training/checkpoint_best_own.pt',map_location='cpu',weights_only=False)
                if (best.get('format')!='crossed_C_best_own_model_v1' or best.get('arm')!='C'
                        or best.get('debug') is not False or best.get('run_identity_sha256')!=result.get('run_identity_sha256')
                        or best.get('selection')!=result.get('best_own')):
                    raise ValueError('C selected model is not the bound own-task best from this full40 run')
                net.load_state_dict(best['state_dict'])
                final=evaluator(net,output=a.output/'training/full_candidate_best_own.json')
                completion=dict(arm='C',identity=identity,debug=False,full_training=True,full_evaluation=final['full_evaluation'],
                    completed_epochs=result['completed_epochs'],quality_verified=False,
                    selected_own_epoch=best['selection']['epoch'],metrics=final['metrics'],
                    evaluation_artifact=final['evaluation_artifact'],nnunet_training=False,
                    checkpoint_sha256=sha(a.output/'training/checkpoint_latest.pt'),evaluation_sha256=sha(final['evaluation_artifact']))
                write_json(a.output/'training/training_complete.json',completion)
            result={key:result.get(key) for key in ('status','checkpoint','updates','completed_epochs','full_training','full_evaluation','quality_verified')}
        else:
            from hiercp_v1x.transition_v1_data import NativeObservationDataset,OriginalInputProvider
            from hiercp_v1x.transition_v1_local import PreservedV1LocalEncoder
            kwargs=dict(workers=a.workers,resident_bytes=resident,rss_bytes=budget.rss_bytes)
            train_ds=NativeObservationDataset(inventory_path,'inner_train',a.debug,a.debug_train_cases)
            val_ds=NativeObservationDataset(inventory_path,'inner_val',a.debug,a.debug_validation_cases)
            all_ds=NativeObservationDataset(inventory_path,'outer_train',a.debug,
                         a.debug_train_cases+a.debug_validation_cases if a.debug else None)
            cache=a.prepared_cache if a.prepared_cache is not None else a.output/'canonical_cache/index.json'
            if not cache.exists():
                provider=OriginalInputProvider(all_ds,**kwargs)
                if a.reuse_preparation is not None:
                    from tools.reuse_v17_D_preparation import import_preparation
                    request=provider.preparation_request()
                    import_preparation(a.reuse_preparation,cache.parent,request,
                        expected_rows=all_ds.rows if a.debug else all_ds.meta['records'],
                        expected_identity=identity)
                cache=provider.preflight(cache.parent,minimum_free_bytes=int(inventory['config']['minimum_free_gb']*2**30))
                del provider
            train_loader=OriginalInputProvider(train_ds,cache_index=cache,**kwargs)
            val_loader=OriginalInputProvider(val_ds,cache_index=cache,**kwargs)
            encoder_base=copy.deepcopy(train_ds.base)
            encoder_base['model'].update(checkpoint_dense_encoder=False,checkpoint_local_blocks=False)
            net=CrossedModel(PreservedV1LocalEncoder(encoder_base,scope_contract=scope['contract_sha256'],
                expected_snapshot_root=source),arm='D',scope_contract=scope['contract_sha256'],dropout=.1,debug=a.debug).cuda()
            record_model(a.output,net,'D',encoder_base,train_ds,val_ds,workers=a.workers,debug=a.debug,
                candidates=a.physical_batch_candidates,graph_cost=train_loader.measured_cost(list(range(len(train_ds)))))
            evaluator=WholeCandidateEvaluation(inventory,train_ds,train_loader,val_ds,val_loader,
                batch=min(a.physical_batch_candidates),budget=budget,debug=a.debug)
            checkpoint=a.output/'training/checkpoint_latest.pt'
            result=train_D(net,train_ds,train_loader,evaluator,encoder_base,output=a.output/'training',budget=budget,
                candidates=a.physical_batch_candidates,support_patients=2 if a.debug else 16,identity=identity,
                debug=a.debug,debug_updates=a.debug_updates,resume=checkpoint if checkpoint.exists() else None)
        print(f'ARM {a.arm} RESULT: {result}\nOriginal experiments preserved; no CP/nnU-Net training started.',flush=True)


if __name__=='__main__':main()
