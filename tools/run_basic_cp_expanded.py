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
read, sha, publish, status = original.read, original.sha, original.publish, original.status
ARMS = {'multi5': (5,128), 'multi7': (7,128), 'multi10': (10,128),
        'candidates512': (1,512), 'candidates1024': (1,1024), 'candidates2048': (1,2048),
        'blend50': (1,128)}
TRAINERS = {arm:'nnUNetTrainer_250epochs_BasicCP'+arm.capitalize() for arm in ARMS}
LEASES = {}


def resource_guard(value, *, idle=False):
    if value.get('shared_resource_registry'):
        from hiercp_v1x.basic_cp_group_resources import BasicLease
    else:
        from hiercp_v1x.basic_cp_seven_resources import BasicLease
    key=(value['root'],value['bank'],value['bank_sha256'])
    if key not in LEASES: LEASES[key]=BasicLease(value)
    return LEASES[key].verify(force=idle,idle=idle)


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


def train(native):
    resource_guard(native)
    import torch,psutil
    from hiercp_v1x import v24_native_best_eval_runtime as best
    from hiercp_v1x.v24_direct_gpu_access import install_native_step_guards
    module,adapter=initialize_native(native);cls=getattr(module,native['trainer'])
    best_guard=best._install(cls,native['split']['outer_val'],bank_sha256=native['bank_sha256'])
    step_guard=install_native_step_guards(cls,lambda:resource_guard(native))
    old_init,old_end=cls.initialize,cls.on_epoch_end;active=[]
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
    try:run_training_entry()
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


def actor(native,command,label,env):
    import psutil
    resource_guard(native);root=Path(native['root']);log=root/(label+'.log')
    shared=label=='candidate_bank'and native.get('shared_candidate_geometry_cache')is True
    lease=None
    if shared:
        from hiercp_v1x.basic_cp_shared_bank_resources import SharedBankLease
        lease=SharedBankLease(native);psutil.Process().cpu_affinity(native['shared_bank_CPU_affinity']);lease.verify(force=True)
    try:
        with log.open('ab',buffering=0)as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            p=psutil.Process(child.pid);identity=dict(pid=p.pid,birth=p.create_time(),UID=os.getuid(),command=command,log=str(log),stage=label)
            receipt=root/(label+'_actor.json')
            if receipt.exists():publish(root/(label+'_prior_actor_'+str(time.time_ns())+'.json'),read(receipt));mutable(receipt,identity)
            else:publish(receipt,identity)
            while child.poll()is None:
                lease.verify(force=True)if shared else resource_guard(native)
                time.sleep(2)
    finally:
        if shared:psutil.Process().cpu_affinity(native['cpu_affinity'])
    if child.returncode:raise RuntimeError(f'{label} failed ({child.returncode}); log: {log}')


def child(native,stage,*,batch=None,output=None):
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--native',str(Path(native['root'])/'native.json'),'--stage',stage]
    if batch is not None:command+=['--batch',str(batch),'--output',str(output)]
    actor(native,command,stage+('_B'+str(batch)if batch else ''),environment(native,output=str(Path(output)/'nnUNet_results')if output else None))


def prepare_bank(request):
    if request.get('shared_resource_registry'):
        from hiercp_v1x.basic_cp_group_resources import BasicLease
    else:
        from hiercp_v1x.basic_cp_seven_resources import BasicLease
    lease=BasicLease(request);count=request['candidates_per_source']
    if count==128:return request
    banks=Path(request['shared_banks']);target=banks/str(count)
    if count==2048 and not(target/'complete.json').exists():
        shared=request.get('shared_candidate_geometry_cache')is True
        command=[sys.executable,'-B','-u','-m','hiercp_v1x.basic_cp_large_bank_fast'if shared else'hiercp_v1x.basic_cp_large_bank',
                 '--binding',request['candidate_binding'],'--output',str(banks),
                 '--workers',str(request['shared_bank_workers'])if shared else'4','--rss-limit-gib','288'if shared else'96']
        if request.get('prepared256_witness'):command+=['--prepared256-witness',request['prepared256_witness']]
        if shared:command+=['--reuse-partial-from',request['previous_partial_banks'],
                            '--reuse-request',request['previous_producer_request'],
                            '--reuse-request-sha256',request['previous_producer_request_sha256']]
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
    if request.get('shared_resource_registry'):
        from hiercp_v1x.basic_cp_group_resources import BasicLease
    else:
        from hiercp_v1x.basic_cp_seven_resources import BasicLease
    lease=BasicLease(request)
    lease.wait_until_available(on_wait=lambda observation:status(request,'waiting_GPU','WAITING',reason=str(observation)))
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
    except BaseException as error:
        prior=read(Path(value['root'])/'status.json')if(Path(value['root'])/'status.json').exists()else {}
        status(value,args.stage or prior.get('stage','controller'),'FAILED',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
