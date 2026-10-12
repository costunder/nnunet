"""v2.5 admission around the unchanged historical Basic native computation.

Private module copies redirect admission, resources and child entry points.
Model, loader, paste, loss, optimizer, clone gradients and BEST inference use
the complete original implementation; old module globals are never changed.
"""
from __future__ import annotations
import ast
import copy
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

from . import v25_cp as cp

ROOT=Path(__file__).resolve().parents[1]
NATIVE_FORMAT='v25_four_arm_Basic642_full250_native_v1'
FILES=('hiercp_v1x/v25_native_cp.py','hiercp_v1x/v25_native_storage.py','hiercp_v1x/v25_cp.py',
    'hiercp_v1x/v25_context.py','hiercp_v1x/v25_downstream.py','hiercp_v1x/v25_direct_resources.py',
    'hiercp_v1x/v25_terminal_descendant_resources.py',
    'hiercp_v1x/v25_parallel_native_storage.py','tools/run_v25_downstream.py',
    'hiercp_v1x/v24_matched_basic_cp_pipeline.py','hiercp_v1x/v24_matched_basic_cp_runtime.py',
    'hiercp_v1x/v24_nnunet_cp.py','hiercp_v1x/v24_matched_native_gradient.py',
    'hiercp_v1x/v24_native_gradient_runtime.py','hiercp_v1x/v24_native_best_eval_runtime.py',
    'hiercp_v1x/v24_native_calibration_runtime.py','hiercp_v1x/v24_readonly_native_storage.py',
    'hiercp_v1x/v24_readonly_static_operator_storage.py','hiercp_v1x/v24_direct_gpu_access.py')
ENV_REQUEST='V25_DOWNSTREAM_REQUEST'
ENV_SHA='V25_DOWNSTREAM_REQUEST_SHA256'
ENV_EXECUTION='V25_DOWNSTREAM_EXECUTION'
ENV_EXECUTION_SHA='V25_DOWNSTREAM_EXECUTION_SHA256'
RESOURCE_FIELDS=('CPU_affinity','allocation_request','allocation_request_sha256')
_MODULES={}
_GUARDS={}


def __getattr__(name):
    if name in ('NATIVE_RUNTIME','NATIVE_PIPELINE'):return private_modules()[name]
    if name=='NATIVE_BUDGET':return native_budget()
    raise AttributeError(name)


def _clone(module,*,overrides,expressions=None,constants=None):
    source=inspect.getsource(module);tree=ast.parse(source);expressions=expressions or {};constants=constants or {}
    patterns={ast.dump(ast.parse(old,mode='eval').body):(old,new) for old,new in expressions.items()}
    counts={old:0 for old in expressions};constant_counts={old:0 for old in constants}
    class Route(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if node.level==1 and node.module=='v24_matched_basic_cp_pipeline':
                if any(alias.name!='early_stage_guard' for alias in node.names):raise ValueError('Unexpected native pipeline import')
                node.module='v25_native_cp'
            elif node.level==1 and node.module=='v24_readonly_native_storage':
                if [alias.name for alias in node.names]==['publish_json']:
                    node.module='v25_native_cp'
                elif all(alias.name=='install_dataset_adapter' for alias in node.names):
                    node.module='v25_native_storage'
            elif node.level==1 and node.module=='v24_nnunet_cp':
                # Only resource imports occur here; mathematical routines stay original.
                if any(alias.name not in ('require_project_budget','snapshot') for alias in node.names):
                    return node
                return [ast.copy_location(ast.ImportFrom(module='v25_native_cp',names=[ast.alias(name=alias.name,asname=alias.asname)],level=1),node) for alias in node.names]
            elif node.level==1 and node.module is None:
                result=[]
                for alias in node.names:
                    if alias.name=='v24_matched_basic_cp_runtime':
                        result.append(ast.ImportFrom(module='v25_native_cp',names=[ast.alias(name='NATIVE_RUNTIME',asname=alias.asname or alias.name)],level=1))
                    elif alias.name=='v24_readonly_static_operator_storage':
                        result.append(ast.ImportFrom(module=None,names=[ast.alias(name='v25_native_storage',asname=alias.asname or alias.name)],level=1))
                    elif alias.name=='v24_nnunet_cp':
                        result.append(ast.ImportFrom(module='v25_native_cp',names=[ast.alias(name='NATIVE_BUDGET',asname=alias.asname or alias.name)],level=1))
                    else:result.append(ast.ImportFrom(module=None,names=[alias],level=1))
                return [ast.copy_location(item,node) for item in result]
            return node
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in constants:
                constant_counts[node.value]+=1;return ast.copy_location(ast.Constant(constants[node.value]),node)
            return node
        def generic_visit(self,node):
            encoded=ast.dump(node)
            if encoded in patterns:
                old,new=patterns[encoded]
                counts[old]+=1;return ast.copy_location(ast.parse(new,mode='eval').body,node)
            return super().generic_visit(node)
    tree=ast.fix_missing_locations(Route().visit(tree))
    expected_counts={old:(2 if old=="static.verify_admission(core.guard(document['original_readonly_admission']))" else 1) for old in expressions}
    expected_constant_counts={old:(2 if old=='tools/run_v24_matched_basic_cp.py' else 1) for old in constants}
    if counts!=expected_counts or constant_counts!=expected_constant_counts:
        raise ValueError('Original native admission/entry expressions changed; explicit review required: '+str((counts,constant_counts)))
    clone=types.ModuleType(module.__name__+'_private_v25');clone.__dict__.update(__package__='hiercp_v1x',__file__=module.__file__)
    sys.modules[clone.__name__]=clone
    exec(compile(tree,str(module.__file__)+':v25_admission','exec'),clone.__dict__)
    clone.__dict__.update(overrides)
    clone.v25_adaptation_proof=dict(original_source_sha256=cp._sha(module.__file__),
        admission_expression_counts=counts,child_entry_counts=constant_counts,
        model_loader_paste_loss_optimizer_expressions_changed=False)
    return clone


def request():
    path=Path(os.environ[ENV_REQUEST]);value=cp._read(path)
    if (cp._sha(path)!=os.environ[ENV_SHA] or value.get('arm')not in ('A','B','C','D')
            or value.get('format')!='v25_full_own_arm_downstream_request_v1'):
        raise ValueError('Exact immutable own v2.5 downstream request required')
    if ENV_EXECUTION in os.environ:
        execution_path=Path(os.environ[ENV_EXECUTION]);execution=cp._read(execution_path)
        if (cp._sha(execution_path)!=os.environ.get(ENV_EXECUTION_SHA)
                or execution.get('format')!='v25_full_own_arm_downstream_execution_v1'
                or execution.get('scientific_request')!=str(path)
                or execution.get('scientific_request_sha256')!=os.environ[ENV_SHA]
                or execution.get('arm')!=value['arm'] or execution.get('gpu')!=value['gpu']
                or set(execution.get('resource_binding',{}))!=set(RESOURCE_FIELDS)):
            raise ValueError('Explicit exact resource-only native execution overlay required')
        value=dict(value,**execution['resource_binding'])
    return value,path


def require_project_budget():
    import psutil
    declared,_=request();actor=psutil.Process()
    if actor.cpu_affinity()!=declared['CPU_affinity']:
        raise ValueError('Native must use exactly its current assigned CPU set')
    processes=[actor]+actor.children(recursive=True)
    rss=0
    for process in processes:
        try:
            if process.status()!=psutil.STATUS_ZOMBIE:rss+=process.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    limit=declared['process_and_children_RSS_bytes']
    if rss>limit:raise MemoryError('Native process tree exceeds its measured own allocation budget')
    return dict(logical_CPU_cores=len(declared['CPU_affinity']),data_augmentation_workers=4,
        process_and_children_RSS_bytes=limit,actual_process_and_children_RSS_bytes=rss,
        actual_affinity=actor.cpu_affinity(),Torch_threads=1)


def snapshot():
    from hiercp.preparation_runtime import snapshot as original_snapshot
    return dict(original_snapshot(),v25_own_resources=require_project_budget())


def native_budget():
    from . import v24_nnunet_cp as original
    module=types.ModuleType(original.__name__+'_v25_resources')
    module.__dict__.update(original.__dict__)
    module.require_project_budget=require_project_budget
    module.resource_contract=lambda:require_project_budget()
    module.snapshot=snapshot
    return module


def _admission(path):
    from . import v25_native_storage as storage
    value=storage.verify_admission(path);declared,_=request();arm=declared['arm'];gpu=declared['gpu']
    # A derived execution view binds the logical arm to this current device.
    # The immutable admission continues to contain all four logical arms.
    result=copy.deepcopy(value);result['arm_roots']={'gpu'+str(gpu):value['arm_roots'][arm]}
    if 'checkpoint_coordination'in result:
        result['checkpoint_coordination']['folds']={'gpu'+str(gpu):value['checkpoint_coordination']['folds'][arm]}
    return result


def _check_disk(document,armroot,*,preparation=True):
    from . import v25_native_storage as storage
    declared,_=request()
    return storage.check_aggregate_disk(declared['storage_admission'],armroot,preparation=preparation)


def _native_profile(native,overlay):
    declared,_=request()
    if (native.get('arm')!=declared['arm'] or overlay['best_provenance']['arm']!=declared['arm']
            or native.get('context_contract')!=overlay['context_contract']
            or native.get('GNN_train_validation_CP_context_identical')is not True):
        raise ValueError('Native must retain own arm and identical newly trained v2.5 CP context')


def _publish(path,value):
    declared,_=request();value=copy.deepcopy(value)
    if value.get('format')==NATIVE_FORMAT:
        value['arm']=declared['arm']
        if 'root'in value:
            value['context_contract']=cp._read(declared['score_overlay'])['context_contract']
            value['GNN_train_validation_CP_context_identical']=True
    return cp._publish(path,value)


def publish_json(path,value):
    """Preserve the first actual configuration and publish explicit resume metadata."""
    path=Path(path)
    if path.name=='actual_nnunet_configuration.json' and path.exists():
        prior=cp._read(path)
        keys=('train_cases','validation_cases','epochs','physical_batch','gradient_accumulation',
            'effective_batch','patch_size','train_steps_per_epoch','validation_steps_per_epoch',
            'CPU_workers','cp_probability','source_policy','candidate_count','source_entries',
            'eligible_cases','validation_checkpoint','parameters','trainable_parameters')
        if any(prior.get(key)!=value.get(key) for key in keys):
            raise ValueError('Exact resumed native scientific configuration changed')
        path=path.with_name('actual_nnunet_configuration_resume_'+str(time.time_ns())+'.json')
        value=dict(value,exact_previous_configuration_sha256=cp._sha(Path(path.parent)/'actual_nnunet_configuration.json'))
    return cp._publish(path,value)


def _mutable(path,value):
    path=Path(path);temporary=path.with_name('.'+path.name+'.'+str(time.time_ns())+'.tmp')
    try:
        with temporary.open('x',encoding='utf8') as stream:
            json.dump(value,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def _child(command,env,log_path):
    """Original subprocess streaming plus exact own child identity for safe resume."""
    import psutil
    declared,path=request();metadata=path.parent;parent=psutil.Process()
    record_path=metadata/'native_child.json';stamp=str(time.time_ns())
    parent_identity=dict(pid=parent.pid,create_time=parent.create_time(),command=parent.cmdline(),UID=os.getuid())
    with Path(log_path).open('x',encoding='utf8') as stream:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,bufsize=1)
        actor=psutil.Process(child.pid)
        record=dict(format='v25_exact_owned_downstream_child_v1',arm=declared['arm'],pid=child.pid,
            create_time=actor.create_time(),command=actor.cmdline(),UID=actor.uids().real,
            parent_actor=parent_identity,state='RUNNING',request_sha256=cp._sha(path),
            log_path=str(Path(log_path).absolute()),updated_at=time.time())
        cp._publish(metadata/('native_child_started_'+stamp+'.json'),record);_mutable(record_path,record)
        for line in child.stdout:
            stream.write(line);stream.flush();print(line,end='',flush=True)
        result=child.wait()
    record.update(state='EXITED',returncode=result,updated_at=time.time())
    cp._publish(metadata/('native_child_exited_'+stamp+'.json'),record);_mutable(record_path,record)
    if result:raise RuntimeError('Own v2.5 stage failed; exact child/log/checkpoints preserved: '+str(log_path))


def private_modules():
    if not _MODULES:
        from . import v24_matched_basic_cp_pipeline as original_pipeline
        from . import v24_matched_basic_cp_runtime as original_runtime
        from . import v25_native_storage as storage
        runtime=_clone(original_runtime,overrides=dict(validate_overlay=cp.validate_overlay),expressions={
            "native['physical_GPU'] != binding['overlay']['pin']['physical_GPU']":
            "native['arm'] != binding['overlay']['best_provenance']['arm']"})
        pipeline=_clone(original_pipeline,overrides=dict(verify_admission=_admission,
            check_aggregate_disk=_check_disk,validate_profiled_overlay=cp.validate_overlay,
            validate_native_scoring_profile=_native_profile,early_stage_guard=early_stage_guard,publish=_publish,
            FORMAT=storage.FORMAT,PROFILE=storage.PROFILE,NATIVE_FORMAT=NATIVE_FORMAT,FILES=FILES,_child=_child),expressions={
            "static.verify_admission(core.guard(document['original_readonly_admission']))":"static.verify_admission(admission_path)",
            "document['original_readonly_admission']['path']":"str(Path(admission_path).absolute())",
            "document['original_readonly_admission']['sha256']":"sha(admission_path)",
            "native.get('physical_GPU') not in (4, 5, 6)":"native.get('physical_GPU') != request()[0]['gpu']",
            "overlay['overlay'].get('training_context_depth_exclusive_mm', 4.)":"1."},constants={
            'tools/run_v24_matched_basic_cp.py':'tools/run_v25_downstream.py',
            'from hiercp_v1x.v24_matched_basic_cp_runtime import run_training_entry; run_training_entry()':
            'from hiercp_v1x.v25_native_cp import run_training_entry; run_training_entry()'})
        pipeline.request=request
        _MODULES.update(NATIVE_RUNTIME=runtime,NATIVE_PIPELINE=pipeline)
    return _MODULES


def early_stage_guard(*,stage_action,gpu=None,chain_root=None):
    """Truthful owned PBS or explicit direct proof guards every native step."""
    import psutil
    from .v25_terminal_descendant_resources import install_runtime
    install_runtime()
    declared,path=request();key=(str(path),os.environ[ENV_SHA],os.environ.get(ENV_EXECUTION_SHA))
    if gpu not in (None,declared['gpu']):raise ValueError('Native physical execution GPU changed')
    if key not in _GUARDS:
        import socket
        proof_path=Path(declared['allocation_request']);proof=cp._read(proof_path);digest=cp._sha(proof_path)
        if digest!=declared['allocation_request_sha256']:
            raise ValueError('Exact admitted native allocation proof changed')
        if proof.get('format')=='v25_explicit_direct_GPU_execution_v1':
            from .v25_direct_resources import DirectAllocationLease
            lease=DirectAllocationLease(proof_path)
            if (lease.request.get('gpu')!=declared['gpu']
                    or lease.request.get('arm')!=declared['arm']
                    or lease.request.get('CPU_affinity')!=declared['CPU_affinity']
                    or lease.request.get('group_root')!=str(Path(declared['output']).parent)):
                raise ValueError('Exact own-arm direct native device/CPU/group proof required')
            def direct_guard():
                if cp._sha(path)!=os.environ[ENV_SHA] or cp._sha(proof_path)!=digest:
                    raise ValueError('Immutable direct native resource request changed')
                request()
                lease.select_cuda()
                return require_project_budget()
            lease.verify(force=True)
            _GUARDS[key]=direct_guard
            _GUARDS[key]()
            return _GUARDS[key]
        if (proof.get('format')!='v25_owned_GPU_allocation_v1' or proof.get('gpu')!=declared['gpu']
                or proof.get('CPU_affinity')!=declared['CPU_affinity'] or proof.get('UID')!=41833
                or proof.get('owner')!='aicompetition06' or proof.get('node')!=socket.gethostname().split('.')[0]):
            raise ValueError('Exact current own PBS/host/physical GPU/CPU allocation required')
        last_check=-1e99
        def guard():
            nonlocal last_check
            if cp._sha(path)!=os.environ[ENV_SHA] or cp._sha(proof_path)!=digest:raise ValueError('Immutable native resource request changed')
            request() # Exact resource execution overlay remains immutable too.
            actor=psutil.Process()
            if actor.uids().real!=41833 or actor.cpu_affinity()!=declared['CPU_affinity']:
                raise RuntimeError('Own native UID/assigned CPU affinity changed')
            if time.time()>=proof['expires_at_unix']-180:raise RuntimeError('Own allocation reaches180-second reserve; preserve exact checkpoint')
            if time.monotonic()-last_check>=30:
                raw=subprocess.check_output(['/opt/pbs/bin/qstat','-f','-F','json',proof['job_id']],text=True,timeout=10)
                jobs=json.loads(raw)['Jobs']
                if list(jobs)!=[proof['job_id']]:raise RuntimeError('Exact current own allocation unavailable')
                job=next(iter(jobs.values()))
                if (job.get('job_state')!='R' or job.get('Job_Owner','').split('@')[0]!=proof['owner']
                        or int(job.get('Resource_List',{}).get('ngpus',0))!=1
                        or any(piece.split('/')[0]!=proof['node'] for piece in job.get('exec_host','').split('+'))):
                    raise RuntimeError('Own running PBS host allocation changed')
                import re
                memory=str(job.get('Resource_List',{}).get('mem','')).lower()
                match=re.fullmatch(r'(\d+(?:\.\d+)?)([kmgt]?b)',memory)
                multiplier={'b':1,'kb':2**10,'mb':2**20,'gb':2**30,'tb':2**40}
                if not match or float(match[1])*multiplier[match[2]]<128*2**30:
                    raise RuntimeError('Native96GiB plus shared-cache32GiB requires actual PBS memory>=128GiB')
                walltime=job.get('Resource_List',{}).get('walltime','')
                try:
                    hours,minutes,seconds=map(int,walltime.split(':'))
                    expiration=float(job['stime'])+hours*3600+minutes*60+seconds
                except (ValueError,KeyError,TypeError) as error:
                    raise RuntimeError('Actual current PBS start/walltime proof required') from error
                if abs(expiration-proof['expires_at_unix'])>1:
                    raise RuntimeError('Current PBS lifetime differs from sealed allocation')
                row=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','-i',str(declared['gpu'])],text=True).strip()
                if row!=proof['GPU_UUID']:raise RuntimeError('Assigned physical GPU UUID changed')
                last_check=time.monotonic()
            os.environ['CUDA_VISIBLE_DEVICES']=proof['GPU_UUID']
            return require_project_budget()
        _GUARDS[key]=guard
    _GUARDS[key]();return _GUARDS[key]


def prepare_native(score_overlay,output,admission_path,*,gpu):
    return private_modules()['NATIVE_PIPELINE'].prepare_native(score_overlay,output,admission_path,gpu=gpu)


def calibrate_native(native_path,*,gpu):return private_modules()['NATIVE_PIPELINE'].calibrate_native(native_path,gpu=gpu)


def calibration_worker(native_path,physical_batch,output):
    return private_modules()['NATIVE_PIPELINE'].calibration_worker(native_path,physical_batch,output)


def run_training_entry():return private_modules()['NATIVE_RUNTIME'].run_training_entry()


def native_completion(native_path):
    from . import v24_native_best_eval_runtime as best
    pipeline=private_modules()['NATIVE_PIPELINE'];native,overlay,admission=pipeline._guard_native(native_path)
    root=Path(native['root']);fold=root/'nnUNet_results'/native['baseline']['dataset_name']/(pipeline.TRAINER+'__'+pipeline.PLANS+'__3d_fullres')/'fold_0'
    receipt=best.validate_best_validation_receipt(fold/best.RECEIPT,fold=fold,
        expected_cases=native['split']['outer_val'],bank_sha256=native['bank_sha256'])
    final=fold/'checkpoint_final.pth'
    import torch
    witness=torch.load(final,map_location='cpu',weights_only=False)
    if witness.get('current_epoch')!=250:raise ValueError('Actual saved full250 native checkpoint required')
    return dict(full_native_training_completed=True,full_native_evaluation_completed=True,completed_epochs=250,
        evaluated_cases=26,checkpoint='checkpoint_best.pth',best_receipt=receipt,
        final_checkpoint=dict(path=str(final),sha256=cp._sha(final)),prediction=str(fold/'validation'),native=native)


def train(native_path,*,gpu):
    """Original native CLI/math; a stopped exact run resumes its own latest."""
    early_stage_guard(stage_action='train',gpu=gpu)
    pipeline=private_modules()['NATIVE_PIPELINE'];native,_,_=pipeline._guard_native(native_path)
    calibration=pipeline.validate_calibration(native_path);root=Path(native['root'])
    fold=root/'nnUNet_results'/native['baseline']['dataset_name']/(pipeline.TRAINER+'__'+pipeline.PLANS+'__3d_fullres')/'fold_0'
    if (fold/'best_validation_complete.json').exists():return native_completion(native_path)
    command=[sys.executable,'-B','-u','-c',
        'from hiercp_v1x.v25_native_cp import run_training_entry; run_training_entry()',
        '730','3d_fullres','0','-tr',pipeline.TRAINER,'-p',pipeline.PLANS,'--val_best']
    resumed=False
    if fold.exists():
        import torch
        final=fold/'checkpoint_final.pth';latest=fold/'checkpoint_latest.pth'
        if final.is_file() and (fold/'checkpoint_best.pth').is_file():
            saved=torch.load(final,map_location='cpu',weights_only=False)
            if saved.get('current_epoch')!=250:raise ValueError('Validation recovery requires actual full250 final witness')
            if latest.exists():raise RuntimeError('Interrupted final publication requires explicit storage repair; results preserved')
            command.append('--val');resumed=True
        elif latest.is_file():
            saved=torch.load(latest,map_location='cpu',weights_only=False)
            if not 0<saved.get('current_epoch',0)<250:raise ValueError('Exact interrupted native full250 latest required')
            command.append('--c');resumed=True
        else:raise RuntimeError('Existing native run has no exact recoverable checkpoint; preserve failed outputs')
    stamp=str(time.time_ns())
    cp._publish(root/('training_started_'+stamp+'.json'),dict(format=NATIVE_FORMAT,command=command,arm=native['arm'],
        epochs=250,physical_batch=2,cp_probability=.5,bank_sha256=native['bank_sha256'],
        calibration_sha256=cp._sha(calibration),evaluation_checkpoint='checkpoint_best.pth',resume=resumed,debug=False))
    env=pipeline.environment(native,native_path)
    env.update({ENV_REQUEST:os.environ[ENV_REQUEST],ENV_SHA:os.environ[ENV_SHA]})
    pipeline._child(command,env,root/('train_'+stamp+'.log'))
    proof=native_completion(native_path)
    cp._publish(root/('training_complete_'+stamp+'.json'),dict(format=NATIVE_FORMAT,arm=native['arm'],epochs=250,
        checkpoint=proof['final_checkpoint'],evaluation_checkpoint='checkpoint_best.pth',
        best_validation_receipt_sha256=cp._sha(fold/'best_validation_complete.json'),bank_sha256=native['bank_sha256']))
    return proof
