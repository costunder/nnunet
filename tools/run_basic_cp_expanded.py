"""Seven full-size one-factor Basic CP experiments; shared immutable inputs."""
from __future__ import annotations
import argparse
import ast
import copy
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_basic_cp_ablation as original
from hiercp_v1x.basic_cp_gpu_admission import (
    GPUAdmissionUnavailable, GPU_WAIT_EXIT_CODE, GPU_WAIT_FORMAT)
read, sha, publish, status = original.read, original.sha, original.publish, original.status
ARMS = {'multi5': (5,128), 'multi7': (7,128), 'multi10': (10,128),
        'candidates512': (1,512), 'candidates1024': (1,1024), 'candidates2048': (1,2048),
        'blend50': (1,128)}
TRAINERS = {arm:'nnUNetTrainer_250epochs_BasicCP'+arm.capitalize() for arm in ARMS}
LEASES = {}


def _lease_class(value):
    if 'RAM_admission_pause_revision'in value:
        from hiercp_v1x.basic_cp_RAM_admission_pause import BasicLease
    elif value.get('group_observation_revision')==5:
        from hiercp_v1x.basic_cp_group_resources_v5 import BasicLease
    elif value.get('group_observation_revision')==4:
        from hiercp_v1x.basic_cp_group_resources_v4 import BasicLease
    elif value.get('shared_resource_guard_version')==3:
        from hiercp_v1x.basic_cp_group_resources_v3 import BasicLease
    elif value.get('shared_resource_guard_version')==2:
        from hiercp_v1x.basic_cp_group_resources_v2 import BasicLease
    elif value.get('shared_resource_registry'):
        from hiercp_v1x.basic_cp_group_resources import BasicLease
    else:
        from hiercp_v1x.basic_cp_seven_resources import BasicLease
    return BasicLease


def _lease_for(value):
    key=(value['root'],value['bank'],value['bank_sha256'])
    if key not in LEASES: LEASES[key]=_lease_class(value)(value)
    return LEASES[key]


def resource_guard(value, *, idle=False):
    return _lease_for(value).verify(force=idle,idle=idle)


def _wait_for_GPU(value):
    from hiercp_v1x.basic_cp_group_resources import RAMAdmissionUnavailable
    lease=_lease_for(value)
    while True:
        try:
            return lease.wait_until_available(
                on_wait=lambda observation:status(value,'waiting_GPU','WAITING',reason=str(observation)))
        except RAMAdmissionUnavailable as failure:
            # Only admission has a queue path. Runtime caps, cgroups, identity,
            # scientific/source contracts and actual training errors still fail.
            status(value,'waiting_RAM','WAITING',reason=str(failure),
                   RAM_admission=copy.deepcopy(failure.observation))
            time.sleep(5)
        except GPUAdmissionUnavailable as failure:
            if value.get('GPU_admission_retry_version') != 1:
                raise
            status(value,'waiting_GPU','WAITING',reason=str(failure.observation),
                   GPU_admission=copy.deepcopy(failure.observation))
            time.sleep(5)


def clone(function, **bindings):
    scope=dict(function.__globals__, **globals(), **bindings)
    result=FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


def prepare(request):
    core=original.storage_core(request)
    document=core._admission(request['readonly_admission'])
    root=Path(request['native_root']);runtime=root/'runtime/nnunetv2'
    shutil.copytree(document['runtime_package'],runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    historical=runtime/'training/nnUNetTrainer/nnUNetTrainer_OnlinePairedCP.py'
    if sha(historical)!=original.HISTORICAL_SHA:raise RuntimeError('Historical trainer changed')
    for name in ('nnUNetTrainer_BasicCPAblations.py','nnUNetTrainer_BasicCPExpanded.py'):
        with(runtime/'training/nnUNetTrainer'/name).open('xb')as f:f.write((ROOT/'custom_trainers'/name).read_bytes())
    if request.get('primary_joint_resampling_policy'):
        name='nnUNetTrainer_BasicCPMultiNJoint.py'
        with(runtime/'training/nnUNetTrainer'/name).open('xb')as f:f.write((ROOT/'custom_trainers'/name).read_bytes())
    baseline=document['baseline'];pre=root/'nnUNet_preprocessed'/baseline['dataset_name']
    data=pre/baseline['data_identifier'];data.mkdir(parents=True)
    for name,witness in document['private_metadata_files'].items():core.copy_private(witness,pre/name)
    gt=pre/'gt_segmentations';gt.mkdir()
    for name,witness in document['ground_truth_files'].items():core.copy_private(witness,gt/name)
    native=dict(request,baseline=baseline,split=document['split'],private_runtime=str(runtime.parent),
        private_pre=str(pre),private_data_folder=str(data),trainer=TRAINERS[request['arm']]+('Joint'if request.get('primary_joint_resampling_policy')else ''),
        historical_trainer_sha256=original.HISTORICAL_SHA,shared_preprocessed_cases=131,
        source_preprocessed_arrays_written=False,full_training_cases=105,full_validation_cases=26)
    publish(Path(request['root'])/'native.json',native)
    return native


def environment(native, *, output=None):
    env=original.environment(native,output=output)
    env.update(BASIC_CP_EXPANDED_ARM=native['arm'],
        BASIC_CP_EXPANDED_PREFLIGHT=str(Path(native['root'])/'preflight.json'),
        BASIC_CP_LEASE_REQUEST=str(Path(native['root'])/'request.json'))
    return env


def initialize_native(native):
    core=original.storage_core(native);sys.path.insert(0,native['private_runtime'])
    adapter=core.install_dataset_adapter(native['readonly_admission'],native['private_data_folder'])
    name='nnUNetTrainer_BasicCPMultiNJoint'if native.get('primary_joint_resampling_policy')else 'nnUNetTrainer_BasicCPExpanded'
    module=importlib.import_module('nnunetv2.training.nnUNetTrainer.'+name)
    return module,adapter


def preflight(native):
    resource_guard(native);module,adapter=initialize_native(native)
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
    dataset=nnUNetDatasetBlosc2(native['private_data_folder'],list(native['split']['outer_train']))
    result=module.preflight_bank(native['bank'],dataset,arm=native['arm'],network_patch_size=(128,128,128))
    result['readonly_adapter']=adapter;publish(Path(native['root'])/'preflight.json',result)
    print('BASIC_CP_PREFLIGHT_COMPLETE '+json.dumps(result.get('summary',{})),flush=True)


def debug_trial(native,batch,output):
    """Same full-model native CUDA debug, with the requested paste count."""
    source=inspect.getsource(original.debug_trial)
    source=source.replace("flags * (3 if native['arm'] == 'multi3' else 1)","flags * native['copies_per_CP_event']")
    if source==inspect.getsource(original.debug_trial):raise RuntimeError('Exact debug paste guard not found')
    scope=dict(original.__dict__,**globals());exec(compile(source,str(Path(__file__).resolve()),'exec'),scope)
    return scope['debug_trial'](native,batch,output)


def mutable(path,value):
    path=Path(path);temporary=path.with_name('.'+path.name+'_'+str(time.time_ns()))
    publish(temporary,value);os.replace(temporary,path)


def fold(native):
    return Path(native['native_root'])/'nnUNet_results'/native['baseline']['dataset_name']/(
        native['trainer']+'__'+native['baseline']['plans_name']+'__3d_fullres')/'fold_0'


def _train_native(native,active):
    resource_guard(native)
    import torch,psutil
    from hiercp_v1x import v24_native_best_eval_runtime as best
    from hiercp_v1x.v24_direct_gpu_access import install_native_step_guards
    module,adapter=initialize_native(native);cls=getattr(module,native['trainer'])
    best_guard=best._install(cls,native['split']['outer_val'],bank_sha256=native['bank_sha256'])
    step_guard=install_native_step_guards(cls,lambda:resource_guard(native))
    old_init,old_end=cls.initialize,cls.on_epoch_end
    def initialize(self):
        old_init(self);active.append(self);tr,val=self.do_split()
        if self.batch_size!=2 or self.num_epochs!=250 or len(tr)!=105 or len(val)!=26 or self.configuration_manager.patch_size!=[128]*3:
            raise RuntimeError('Original full native training configuration differs')
        model=dict(trainer=native['trainer'],architecture=self.configuration_manager.network_arch_init_kwargs,
            model_class=type(self.network).__name__,parameters=sum(p.numel()for p in self.network.parameters()),
            trainable_parameters=sum(p.numel()for p in self.network.parameters()if p.requires_grad),physical_batch=2,
            gradient_accumulation=1,effective_batch=2,epochs=250,train_steps_per_epoch=self.num_iterations_per_epoch,
            validation_steps_per_epoch=self.num_val_iterations_per_epoch,actual_train_cases=tr,actual_validation_cases=val,
            patch_size=[128]*3,workers=4,precision='native CUDA autocast/GradScaler',debug=False,subset=False,
            pretrained_segmentation_used=False,own_BEST_guard=best_guard,resource_step_guard=step_guard,readonly_adapter=adapter)
        path=Path(native['root'])/'actual_model.json'
        if not path.exists():publish(path,model)
        else:
            old=read(path)
            for k in ('trainer','architecture','parameters','physical_batch','epochs','actual_train_cases','actual_validation_cases'):
                if old[k]!=model[k]:raise RuntimeError('Resume model differs: '+k)
    def epoch_end(self):
        result=old_end(self)
        if 0<self.current_epoch<250:
            target=Path(self.output_folder)/'checkpoint_latest.pth'
            temporary=target.with_name('whole_epoch_'+str(time.time_ns())+'.pth')
            from hiercp_v1x.basic_cp_completed_epoch_checkpoint import save_completed_epoch
            save_completed_epoch(self,str(temporary));os.replace(temporary,target)
        status(native,'train','RUNNING',completed_epochs=int(self.current_epoch),
               RSS_bytes=psutil.Process().memory_info().rss,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        return result
    cls.initialize,cls.on_epoch_end=initialize,epoch_end
    goal=fold(native);sys.argv=['BasicCPExpanded','730','3d_fullres','0','-tr',native['trainer'],
                             '-p',native['baseline']['plans_name'],'--val_best']
    if(goal/'checkpoint_final.pth').exists():
        if torch.load(goal/'checkpoint_final.pth',map_location='cpu',weights_only=False)['current_epoch']!=250:
            raise RuntimeError('Final completion witness is not full250')
        sys.argv+=['--val']
    elif(goal/'checkpoint_latest.pth').exists():sys.argv+=['--c']
    from nnunetv2.run.run_training import run_training_entry
    run_training_entry()
    return goal,best


def train(native):
    # Even an initial admission race crosses this finally before exit75 is published.
    active=[]
    try:goal,best=_train_native(native,active)
    finally:
        for instance in active:
            for loader in (getattr(instance,'dataloader_train',None),getattr(instance,'dataloader_val',None)):
                finish=getattr(loader,'_finish',None)
                if finish is not None:finish()
    receipt=best.validate_best_validation_receipt(goal/best.RECEIPT,fold=goal,
        expected_cases=native['split']['outer_val'],bank_sha256=native['bank_sha256'])
    publish(Path(native['root'])/'training_and_BEST26_complete.json',dict(epochs=250,checkpoint='checkpoint_best.pth',
        best_sha256=receipt['checkpoint_sha256'],validation_cases=receipt['validation_cases'],fold=str(goal),
        final_training_witness_sha256=sha(goal/'checkpoint_final.pth'),evaluation_checkpoint_is_final=False))


def _native_label(stage,batch=None):
    return stage+('_B'+str(batch)if stage=='debug'and batch is not None else '')


def _native_wait_receipt(native,label):
    return Path(native['root'])/(label+'_GPU_admission_wait.json')


def _publish_native_GPU_wait(native,stage,batch,failure):
    """Called only after the native stage's own finally has completed cleanup."""
    import psutil
    from hiercp_v1x.basic_cp_ablation_resources import _actor
    value=dict(format=GPU_WAIT_FORMAT,stage=stage,label=_native_label(stage,batch),
        actor_identity=_actor(psutil.Process()),bank_sha256=native['bank_sha256'],
        observation=copy.deepcopy(failure.observation),exit_code=GPU_WAIT_EXIT_CODE,
        native_stage_finally_completed=True,processes_signaled=False,observed_at=time.time())
    path=_native_wait_receipt(native,value['label'])
    if path.exists():
        publish(path.with_name(path.stem+'_prior_'+str(time.time_ns())+'.json'),read(path))
        mutable(path,value)
    else:publish(path,value)
    status(native,'waiting_GPU','WAITING',reason=str(failure.observation),
           GPU_admission=copy.deepcopy(failure.observation),native_actor_wait_receipt=str(path))
    return GPU_WAIT_EXIT_CODE


def _capture_native_tree(native,tracked):
    """Capture actual admitted descendants, including workers before reparenting."""
    import psutil
    from hiercp_v1x.basic_cp_ablation_resources import _exact,_metadata,_kernel_exiting
    lease=_lease_for(native);controller=_exact(native['controller_identity'])
    tree=lease._tree(controller,controller)
    for pid,process in tree.items():
        if pid==controller.pid:continue
        # _tree has already pinned a coherent descendant identity. Reuse that
        # actual snapshot rather than racing another argv read during task exit.
        identity=lease._managed.get(pid)
        if identity is None:
            try:
                metadata=_metadata(process);command=process.cmdline()
                if command or not _kernel_exiting(process):
                    raise RuntimeError('Native capture lacks an exact admitted or kernel-exiting identity')
                identity=dict(metadata,command=command,kernel_exiting_observed=True)
            except psutil.NoSuchProcess:
                continue  # A previously captured identity is retained unchanged.
        metadata={key:identity[key]for key in('pid','create_time','UID')}
        previous=tracked.get(pid)
        if previous is not None and any(metadata[k]!=previous[k]for k in('pid','create_time','UID')):
            raise RuntimeError('Native cleanup PID/birth/UID changed')
        tracked[pid]=copy.deepcopy(identity)


def _wait_native_release(native,child,identity,tracked,failure):
    """No replacement actor until exact children naturally terminate; no signals."""
    import psutil
    from hiercp_v1x.basic_cp_ablation_resources import _metadata
    lease=_lease_for(native);released={}
    while True:
        _capture_native_tree(native,tracked)
        observation=lease.observe_waiting()  # SHA, UID, PBS, CPU, RAM and GPU UUID remain checked.
        child.poll();alive=[]
        for pid,expected in tracked.items():
            if pid in released:continue
            try:
                process=psutil.Process(pid);actual=_metadata(process)
                if any(actual[k]!=expected[k]for k in('pid','create_time','UID')):
                    raise RuntimeError('Native cleanup exact PID/birth/UID was reused or changed')
                state=process.status()
                if state in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD):
                    released[pid]=dict(expected,status=state,naturally_terminal=True)
                else:
                    if process.cmdline()!=expected['command']:
                        # The original descendant guard recognizes only its exact fork/exec transition.
                        if lease._descendant(process)is None:
                            released[pid]=dict(expected,status='naturally_terminal',naturally_terminal=True)
                        else:
                            tracked[pid]=dict(_metadata(process),command=process.cmdline());alive.append(pid)
                    else:alive.append(pid)
            except psutil.NoSuchProcess:
                released[pid]=dict(expected,status='NoSuchProcess',naturally_terminal=True)
        status(native,'waiting_GPU','WAITING',reason=str(failure.observation),
            GPU_admission=copy.deepcopy(failure.observation),native_actor_cleanup=dict(
                actor=identity,returncode=child.returncode,alive_captured_PIDs=alive,
                naturally_released=copy.deepcopy(released),processes_signaled=False),
            GPU_wait_observation=observation)
        if child.returncode is not None and not alive:
            return dict(actor=identity,returncode=child.returncode,
                        naturally_released=copy.deepcopy(released),processes_signaled=False)
        time.sleep(2)


def _validate_native_GPU_wait(native,label,identity):
    receipt=read(_native_wait_receipt(native,label))
    expected=dict(pid=identity['pid'],create_time=identity['birth'],UID=identity['UID'],command=identity['command'])
    if (receipt.get('format')!=GPU_WAIT_FORMAT or receipt.get('label')!=label
            or receipt.get('stage')!=label.split('_B')[0]
            or receipt.get('actor_identity')!=expected or receipt.get('bank_sha256')!=native['bank_sha256']
            or receipt.get('exit_code')!=GPU_WAIT_EXIT_CODE or receipt.get('native_stage_finally_completed')is not True
            or receipt.get('processes_signaled')is not False or not isinstance(receipt.get('observation'),dict)):
        raise RuntimeError('Native GPU queue exit75 lacks an exact actor/stage/bank/finally receipt')
    observation=receipt['observation']
    if (observation.get('GPU_UUID')!=native['gpu_uuid'] or observation.get('physical_GPU')!=native['gpu']
            or type(observation.get('pid'))is not int or observation['pid']<=0
            or observation.get('admitted')is not False or observation.get('processes_signaled')is not False
            or not isinstance(observation.get('fresh_tree_PIDs'),list)
            or observation['pid']in observation['fresh_tree_PIDs']
            or type(observation.get('create_time'))not in(int,float)
            or not math.isfinite(observation['create_time']) or observation['create_time']<=0
            or type(observation.get('UID'))is not int or observation['UID']<0
            or not isinstance(observation.get('command'),list) or not observation['command']
            or any(not isinstance(part,str)or not part for part in observation['command'])
            or observation.get('status')in(None,'zombie','dead','NoSuchProcess')):
        raise RuntimeError('Native GPU queue receipt lacks an exact live outside CUDA identity')
    return GPUAdmissionUnavailable(receipt['observation'])


def actor(native,command,label,env):
    import psutil
    resource_guard(native);root=Path(native['root']);log=root/(label+'.log')
    shared=label=='candidate_bank'and native.get('shared_candidate_geometry_cache')is True
    lease=None
    if shared:
        if 'RAM_admission_pause_revision'in native:
            from hiercp_v1x.basic_cp_RAM_admission_pause import SharedBankLease
        elif native.get('shared_resource_observation_revision')==6:
            from hiercp_v1x.basic_cp_shared_bank_resources_v6 import SharedBankLease
        elif native.get('group_observation_revision')==5:
            from hiercp_v1x.basic_cp_shared_bank_resources_v5 import SharedBankLease
        elif native.get('shared_resource_guard_version')==3:
            from hiercp_v1x.basic_cp_shared_bank_resources_v3 import SharedBankLease
        else:
            from hiercp_v1x.basic_cp_shared_bank_resources import SharedBankLease
        lease=SharedBankLease(native);psutil.Process().cpu_affinity(native['shared_bank_CPU_affinity']);lease.verify(force=True)
    native_retry=native.get('GPU_admission_retry_version')==1 and(label in('preflight','train')or label.startswith('debug_B'))
    tracked={};admission_failure=None;waiting_for_GPU=False
    try:
        with log.open('ab',buffering=0)as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            p=psutil.Process(child.pid);identity=dict(pid=p.pid,birth=p.create_time(),UID=os.getuid(),command=command,log=str(log),stage=label)
            receipt=root/(label+'_actor.json')
            if receipt.exists():publish(root/(label+'_prior_actor_'+str(time.time_ns())+'.json'),read(receipt));mutable(receipt,identity)
            else:publish(receipt,identity)
            if native_retry:
                tracked[p.pid]=dict(pid=p.pid,create_time=identity['birth'],UID=identity['UID'],command=command)
            while child.poll()is None:
                if native_retry:_capture_native_tree(native,tracked)
                try:
                    if shared:lease.verify(force=True)
                    elif waiting_for_GPU:_lease_for(native).verify(force=True,idle=False)
                    else:resource_guard(native)
                except GPUAdmissionUnavailable as failure:
                    if not native_retry:raise
                    admission_failure=failure;waiting_for_GPU=True
                    status(native,'waiting_GPU','WAITING',reason=str(failure.observation),
                        GPU_admission=copy.deepcopy(failure.observation),native_actor_cleanup=dict(
                            actor=identity,alive_captured_PIDs=sorted(tracked),processes_signaled=False))
                else:
                    if waiting_for_GPU:
                        status(native,'CUDA_DEBUG'if label.startswith('debug_B')else label,'RUNNING',
                               native_actor_GPU_readmitted=True,native_actor=identity)
                        waiting_for_GPU=False
                time.sleep(2)
    finally:
        if shared:psutil.Process().cpu_affinity(native['cpu_affinity'])
    if native_retry and(admission_failure is not None or child.returncode==GPU_WAIT_EXIT_CODE):
        if admission_failure is None:
            admission_failure=_validate_native_GPU_wait(native,label,identity)
        cleanup=_wait_native_release(native,child,identity,tracked,admission_failure)
        publish(root/(label+'_natural_cleanup_'+str(time.time_ns())+'.json'),cleanup)
        if child.returncode==GPU_WAIT_EXIT_CODE:
            raise _validate_native_GPU_wait(native,label,identity)
        if child.returncode==0:return
    if child.returncode:raise RuntimeError(f'{label} failed ({child.returncode}); log: {log}')


def child(native,stage,*,batch=None,output=None):
    # Bank preparation can outlast the initial GPU admission. Re-read actual
    # release before each native actor; keep the exact same controller/lease.
    attempt_output=output
    while True:
        _wait_for_GPU(native)
        status(native,'CUDA_DEBUG'if stage=='debug'else stage,'RUNNING')
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--native',str(Path(native['root'])/'native.json'),'--stage',stage]
        if batch is not None:command+=['--batch',str(batch),'--output',str(attempt_output)]
        try:
            result=actor(native,command,_native_label(stage,batch),environment(native,output=str(Path(attempt_output)/'nnUNet_results')if attempt_output else None))
            if stage=='debug'and attempt_output!=output:
                trial_path=Path(attempt_output)/'trial.json';trial=read(trial_path)
                trial['GPU_admission_retry_DEBUG']=dict(actual_completed_attempt_output=str(attempt_output),
                    actual_trial_sha256=sha(trial_path),original_partial_output=str(output),
                    original_partial_files_preserved=True)
                publish(Path(output)/'trial.json',trial)
            return result
        except GPUAdmissionUnavailable as failure:
            if native.get('GPU_admission_retry_version')!=1:raise
            status(native,'waiting_GPU','WAITING',reason=str(failure.observation),
                   GPU_admission=copy.deepcopy(failure.observation),retry_native_stage=stage)
            if stage=='debug'and output is not None:
                # Partial DEBUG witnesses stay intact; a retry gets fresh output files.
                attempt_output=Path(output).with_name(Path(output).name+'_GPU_RETRY_'+str(time.time_ns()))
                attempt_output.mkdir()


def prepare_bank(request):
    lease=_lease_for(request);count=request['candidates_per_source']
    if count==128:return request
    banks=Path(request['shared_banks']);target=banks/str(count)
    if count==2048 and not(target/'complete.json').exists():
        shared=request.get('shared_candidate_geometry_cache')is True
        recovery=request.get('interrupted_candidate_recovery_revision')
        if recovery is not None and(type(recovery)is not int or recovery!=1 or not shared
                or 'hiercp_v1x/basic_cp_large_bank_recovery.py'not in request['code_sha256']):
            raise ValueError('Explicit shared candidate recovery revision1 and exact new module SHA required')
        provider='hiercp_v1x.basic_cp_large_bank_recovery'if recovery==1 else(
            'hiercp_v1x.basic_cp_large_bank_fast'if shared else'hiercp_v1x.basic_cp_large_bank')
        command=[sys.executable,'-B','-u','-m',provider,
                 '--binding',request['candidate_binding'],'--output',str(banks),
                 '--workers',str(request['shared_bank_workers'])if shared else'4','--rss-limit-gib',str(request['shared_bank_RSS_limit_GiB'])if shared else'96']
        if request.get('prepared256_witness'):command+=['--prepared256-witness',request['prepared256_witness']]
        if shared:command+=['--reuse-partial-from',request['previous_partial_banks'],
                            '--reuse-request',request['previous_producer_request'],
                            '--reuse-request-sha256',request['previous_producer_request_sha256']]
        if recovery==1:command+=['--recovery-snapshot',request['interrupted_candidate_recovery_snapshot'],
            '--recovery-snapshot-sha256',request['interrupted_candidate_recovery_snapshot_sha256']]
        status(request,'candidate_bank','RUNNING',candidate_bank_root=str(target),candidate_count=count)
        actor(request,command,'candidate_bank',dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='',
            BASIC_CP_LEASE_REQUEST=str(Path(request['root'])/'request.json'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    while not(target/'complete.json').exists():
        lease.observe_waiting()
        producer=read(Path(request['producer_root'])/'status.json')if(Path(request['producer_root'])/'status.json').exists()else {}
        if producer.get('status')=='FAILED':raise RuntimeError('Shared candidate preparation failed: '+str(producer.get('error')))
        status(request,'candidate_bank','WAITING',candidate_bank_root=str(target),candidate_count=count,
               reason='Shared full642 candidate-prefix completion pending')
        time.sleep(5)
    index=target/'index.json';meta=read(index);done=read(target/'complete.json')
    if meta['source_entries']!=642 or meta['candidate_count']!=count or done.get('complete')is not True or done['index_sha256']!=sha(index):
        raise RuntimeError('Actual complete642 requested candidate bank required')
    from hiercp_v1x.basic_cp_large_bank import verify_prepared_bank
    witness=dict(index=str(index),index_sha256=sha(index),complete_sha256=sha(target/'complete.json'),
                 manifest_sha256=sha(target/'preservation_manifest.json'),preparation_started_sha256=sha(target/'preparation_started.json'))
    verify_prepared_bank(witness,Path(request['candidate_binding']),request['candidate_binding_sha256'],candidate_count=count)
    publish(Path(request['root'])/'prepared_candidate_bank_admission.json',witness)
    return dict(request,bank=str(index),bank_sha256=sha(index),source_bank=request['bank'],source_bank_sha256=request['bank_sha256'])


def run(request):
    _wait_for_GPU(request)
    request=prepare_bank(request);path=Path(request['root'])/'native.json'
    native=read(path)if path.exists()else prepare(request)
    if not(Path(native['root'])/'preflight.json').exists():status(native,'preflight','RUNNING');child(native,'preflight')
    if not(Path(native['root'])/'calibration.json').exists():
        trials=[]
        for batch in (2,4):
            output=Path(native['root'])/('B'+str(batch)+'_DEBUG');output.mkdir()
            status(native,'CUDA_DEBUG','RUNNING');child(native,'debug',batch=batch,output=output);trials.append(read(output/'trial.json'))
        if not trials[0]['accepted']:raise RuntimeError('Original physical B2 actual CUDA debug failed')
        publish(Path(native['root'])/'calibration.json',dict(debug=True,trials=trials,production_physical_batch=2,
            production_effective_batch=2,decision='Keep historical B2 for controlled one-factor comparison',production_optimizer_updates=0))
    status(native,'train','RUNNING');child(native,'train')
    proof=read(Path(native['root'])/'training_and_BEST26_complete.json')
    status(native,'historical_metrics','RUNNING',completed_epochs=250)
    command=[sys.executable,'-B','-u',str(ROOT/'tools/evaluate_v24_historical_metrics.py'),
        '--evaluator-root',native['evaluator_root'],'--prediction',str(Path(proof['fold'])/'validation'),
        '--output',str(Path(native['root'])/'evaluation'),'--project',native['project'],'--checkpoint','BEST',
        '--trainer',native['trainer'],'--model-label','Basic CP '+native['arm'],'--gpu-label','GPU'+str(native['gpu'])]
    actor(native,command,'historical_metrics',environment(native))
    from tools.basic_cp_expanded_result import completed_report,archive_completed
    status(native,'report','RUNNING',completed_epochs=250)
    reports=completed_report(native,proof);archive=archive_completed(native,proof,reports)
    status(native,'complete','COMPLETE',completed_epochs=250,evaluated_cases=26,full_native_training=True,
           full_native_evaluation=True,historical_metrics_complete=True,reports=reports,archive=archive)


def main():
    import psutil
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path);parser.add_argument('--native',type=Path)
    parser.add_argument('--stage',choices=('preflight','debug','train'));parser.add_argument('--batch',type=int);parser.add_argument('--output',type=Path)
    args=parser.parse_args();target=args.request or args.native;deadline=time.monotonic()+120
    while not target.exists()and time.monotonic()<deadline:time.sleep(.1)
    value=read(target);psutil.Process().cpu_affinity(value['cpu_affinity'])
    if value['arm']not in ARMS or(value['copies_per_CP_event'],value['candidates_per_source'])!=ARMS[value['arm']]:
        raise ValueError('Exact requested one-factor arm settings required')
    if value.get('primary_joint_resampling_policy') and (value['arm'] not in ('multi5','multi7','multi10')
            or value['primary_joint_resampling_policy']!='basic_cp_multin_same_source128_primary_rejection_v1'):
        raise ValueError('Only the explicitly approved same-source128 impossible-primary rejection policy is allowed')
    try:
        if args.stage=='preflight':preflight(value)
        elif args.stage=='debug':debug_trial(value,args.batch,args.output)
        elif args.stage=='train':train(value)
        else:run(value)
    except GPUAdmissionUnavailable as error:
        if args.stage is not None and value.get('GPU_admission_retry_version')==1:
            return _publish_native_GPU_wait(value,args.stage,args.batch,error)
        prior=read(Path(value['root'])/'status.json')if(Path(value['root'])/'status.json').exists()else {}
        status(value,args.stage or prior.get('stage','controller'),'FAILED',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
        raise
    except BaseException as error:
        prior=read(Path(value['root'])/'status.json')if(Path(value['root'])/'status.json').exists()else {}
        status(value,args.stage or prior.get('stage','controller'),'FAILED',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    result=main()
    if result is not None:raise SystemExit(result)
