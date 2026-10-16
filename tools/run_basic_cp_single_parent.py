"""Unchanged unfinished Basic CP continuations from ONE canonical Git checkout.

The launcher consumes a SHA-bound cohort, starts independently registered
processes inside its existing PBS allocation and preserves native log/UI data.
No code tree, runtime package, source bank or preprocessed dataset is copied.
Only exact checkpoint bytes and small immutable evidence enter new output roots.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hiercp_v1x.basic_cp_single_parent_resources import(ARMS,EPOCHS,GIB,DIRECT_MODE,Lease,digest,exact,execution_arms,
    exact_actor,identity,pin,require,signature,validate_allocation,nvml_devices,_memory_cgroup,observe_holder,cgroup_identity)
import psutil

FORMAT='basic_cp_single_parent_continuation_v1'
HISTORICAL_SHA='a3df56435fedff46fbc42a502596fdade2632d0211ea412630dba1555f32567d'
NATIVE_TRAINER_SHA='7096efb2040135eb60df3c8bf39cdbcc373e299671fcee365b3aff2f4da4dfbd'


def read(path):return json.loads(Path(path).read_bytes())


def seal(path,value):
    with Path(path).open('x',encoding='utf8')as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def publish(path,value):
    path=Path(path);temporary=path.with_name('.'+path.name+'.'+str(os.getpid())+'.'+str(time.time_ns()))
    seal(temporary,value);os.replace(temporary,path)


def arm_status(cohort,arm,stage,state,**values):
    row=cohort['experiments'][arm]
    publish(Path(row['output_root'])/'status.json',dict(format='basic_cp_ablation_full250_status_v1',arm=arm,
        physical_GPU=cohort['physical_parent_index'],logical_slot=row['logical_slot'],stage=stage,status=state,
        time=time.time(),epochs=250,checkpoint='checkpoint_best.pth',**values))


def record_child_failure(cohort,arm,error):
    root=Path(cohort['experiments'][arm]['output_root']);prior=read(root/'status.json')
    arm_status(cohort,arm,'train','FAILED',error=type(error).__name__+': '+str(error),
        failed_stage=prior.get('failed_stage')or prior.get('native_substage')or prior.get('stage'),
        completed_epochs=prior.get('completed_epochs'),traceback=traceback.format_exc())


def copy_exact(reference,target):
    source=pin(reference);target=Path(target)
    if target.exists():
        require(digest(target)==reference['sha256'],'Existing target differs; no result may be overwritten')
        return target
    target.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb')as before,target.open('xb')as after:
        shutil.copyfileobj(before,after,8*2**20);after.flush();os.fsync(after.fileno())
    require(digest(target)==reference['sha256'],'Transferred artifact differs from original exact bytes')
    return target


def bound_completed_epochs(arm,binding):
    """The original saved epoch is a floor; an explicit later whole save may resume."""
    require(arm in EPOCHS and isinstance(binding,dict),'Exact original arm/input binding required')
    epoch=binding.get('completed_epochs',EPOCHS[arm])
    require(type(epoch)is int and EPOCHS[arm]<=epoch<250,
        'Bound completed epoch must be an integer at or after the original save and before250: '+arm)
    return epoch


def validate_bound_resume(arm,row,binding):
    epoch=bound_completed_epochs(arm,binding)
    require(type(row.get('completed_epochs'))is int and row['completed_epochs']==epoch,
        'Cohort completed epoch differs from its pinned input binding: '+arm)
    for name in('effective_bank','preflight','LATEST','BEST','calibration'):
        require(row.get(name)==binding.get(name),'Cohort artifact differs from pinned input binding: '+arm+'/'+name)
    proof=row.get('actual_resume_checkpoint_proof',{})
    require(type(proof.get('current_epoch'))is int and proof['current_epoch']==epoch
        and type(proof.get('logging_epochs'))is int and proof['logging_epochs']==epoch,
        'Actual resume checkpoint proof differs from bound whole completed epoch: '+arm)
    best=row.get('actual_original_BEST_proof',{})
    require(type(best.get('current_epoch'))is int and 0<best['current_epoch']<=epoch
        and best.get('logging_epochs')==best['current_epoch'],
        'Actual own BEST proof lies outside the bound completed prefix: '+arm)
    prefix=row.get('completed_log_prefix_proof',{})
    require(prefix.get('epochs')==epoch and prefix.get('first_epoch')==0 and prefix.get('last_epoch')==epoch-1
        and prefix.get('CP_attempted')==500*epoch
        and row.get('raw_original_log')==binding.get('log_prefix')
        and prefix.get('original_raw_log')==binding.get('log_prefix'),
        'Completed log prefix differs from the pinned raw log/bound checkpoint epoch: '+arm)
    return epoch


def continuation_arms(value):
    """Omitted arms must be explicitly complete, never silently dropped."""
    arms=value.get('active_arms',list(ARMS))
    require(isinstance(arms,(list,tuple))and len(arms)>0
        and list(arms)==[arm for arm in ARMS if arm in arms],
        'Explicit unique canonical-order unfinished experiment arms required')
    retained=[arm for arm in ARMS if arm not in arms]
    require(value.get('retained_complete_arms')==['blend50',*retained]and value.get('debug')is False,
        'Every omitted experiment must be retained complete; DEBUG production is forbidden')
    evidence=value.get('retained_completion_evidence',{})
    require(isinstance(evidence,dict)and set(evidence)==set(retained),
        'Every omitted experiment needs SHA-pinned full completion evidence')
    for arm in retained:
        record=evidence[arm]
        require(set(record)=={'status','training_and_BEST26','BEST','FINAL','reports','archive','request','output_root','logical_slot'},
            'Exact retained completion artifact set required: '+arm)
        status=read(pin(record['status']));proof=read(pin(record['training_and_BEST26']))
        require(status.get('arm')==arm and status.get('status')=='COMPLETE'and status.get('stage')=='complete'
            and status.get('completed_epochs')==250 and status.get('evaluated_cases')==26
            and all(status.get(key)is True for key in('full_native_training','full_native_evaluation','historical_metrics_complete')),
            'Omitted experiment has not completed full250 training, BEST26 and statistics: '+arm)
        require(proof.get('epochs')==250 and proof.get('checkpoint')=='checkpoint_best.pth'
            and proof.get('evaluation_checkpoint_is_final')is False
            and len(set(proof.get('validation_cases',[])))==26,
            'Omitted experiment needs its own BEST26 completion witness: '+arm)
        pin(record['BEST']);pin(record['FINAL'])
        require(record['BEST']['sha256']==proof.get('best_sha256')
            and record['FINAL']['sha256']==proof.get('final_training_witness_sha256'),
            'Retained own BEST or full250 FINAL witness bytes changed: '+arm)
        require(set(record['reports'])=={'evidence','text','json'},'All fixed-schema result reports must be retained')
        for name,reference in record['reports'].items():
            pin(reference)
            require(reference['sha256']==status['reports'][name]['sha256'],'Retained result report bytes changed')
        pin(record['archive']);request=read(pin(record['request']))
        require(request.get('arm')==arm and request.get('logical_slot')==record['logical_slot']
            and Path(record['status']['path'])==Path(record['output_root'])/'status.json'
            and Path(record['request']['path'])==Path(record['output_root'])/'request.json',
            'Retained viewer route must point to that exact original experiment')
        require(set(proof['validation_cases'])==set(request['split']['outer_val'])
            and Path(record['BEST']['path']).relative_to(Path(record['output_root']))==
                Path(proof['fold']).relative_to(Path(request['root']))/'checkpoint_best.pth'
            and Path(record['FINAL']['path']).relative_to(Path(record['output_root']))==
                Path(proof['fold']).relative_to(Path(request['root']))/'checkpoint_final.pth',
            'Retained BEST26 witness must belong to the same complete original experiment')
        require(record['archive']['sha256']==status['archive']['sha256'],'Retained complete archive marker changed')
    return tuple(arms)


def validate_cohort(value):
    arms=continuation_arms(value)
    require(value.get('format')==FORMAT and set(value.get('experiments',{}))==set(arms),
        'Exact unfinished experiments required; completed experiments are retained separately')
    require(Path(value['canonical_code']['path'])==ROOT,
        'Every process must execute the one explicit canonical Git checkout')
    require(Path(value['root']).is_absolute()and len({row['output_root']for row in value['experiments'].values()})==len(arms),
        'Distinct absolute unfinished result folders required')
    require(Path(value['root']).is_relative_to(Path('/dev/shm')),
        'The existing independent archive contract requires actual tmpfs output roots')
    bindings=read(pin(value['input_bindings'],owner=True))
    require(isinstance(bindings,dict)and set(bindings)==set(arms),
        'Exact SHA-pinned original or saved-resume unfinished input bindings required')
    for arm,row in value['experiments'].items():
        validate_bound_resume(arm,row,bindings[arm])
        require(row.get('physical_batch')==2
            and row.get('epochs')==250 and row.get('workers')==4,
            'Actual whole original checkpoint/B2/250epoch/four-worker contract differs: '+arm)
        require(Path(row['output_root']).is_relative_to(Path(value['root'])),'Results must remain in the fresh own cohort')
        require(Path(row['evaluator_root'])==ROOT,'Historical metrics must use the same canonical Git checkout')
        for name in('source_native','effective_bank','preflight','LATEST','BEST','log_prefix','calibration'):
            require(isinstance(row.get(name),dict)and set(row[name])=={'path','sha256'},
                'Explicit scientific artifact path/SHA required: '+arm+'/'+name)
    return value


def verify_historical_reference(cohort,expected_cases):
    proof=read(pin(cohort['historical_reference_inputs']))
    require(proof.get('format')=='basic_cp_historical_reference_actual26_readiness_v1'and proof.get('complete')is True
        and proof.get('ground_truth_files')==proof.get('basic_prediction_files')==26
        and set(proof['validation_cases'])==set(expected_cases)and proof.get('NIfTI_header_geometry_verified')is True
        and proof.get('full_evaluation_executed')is False and proof.get('inputs_modified')is False,
        'Actual full26 historical reference readiness required; no final-evaluation claim')
    for path,record in proof['files'].items():
        require(list(signature(path))==record['stat'],'Admitted historical reference changed: '+path)
    return proof


def verify_inputs(cohort,arm):
    """Verify the actual effective642 bank, never the stale native.bank field."""
    row=cohort['experiments'][arm];native=read(pin(row['source_native']))
    require(native['arm']==arm,'Original native snapshot belongs to another experiment')
    bank_path=pin(row['effective_bank']);bank=read(bank_path);preflight=read(pin(row['preflight']))
    expected=128 if arm.startswith('multi')else int(arm.removeprefix('candidates'))
    expanded=importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer_BasicCPExpanded')
    expanded.validate_metadata(bank,arm)
    require(bank['source_entries']==642 and bank['candidate_count']==expected
        and preflight['complete']is True and preflight['arm']==arm
        and preflight['index_sha256']==row['effective_bank']['sha256']
        and preflight['source_entries']==642 and preflight['candidate_count']==expected,
        'Exact full642 effective-bank/preflight family differs')
    names={name for entries in bank['entries_by_case'].values()for name in entries}
    require(len(names)==642 and set(preflight['entries_sha256'])==names,
        'Every actual642 source needs its original complete preflight SHA')
    for name,checksum in preflight['entries_sha256'].items():
        require(digest(bank_path.parent/name)==checksum,'Actual full source NPZ changed: '+name)
    if arm.startswith('multi'):
        require(native.get('primary_joint_resampling_policy')=='basic_cp_multin_same_source128_primary_rejection_v1'
            and preflight.get('all642_sources_have_joint_feasible_primary')is True,
            'Approved same-source128 impossible-primary rejection policy required')
    preprocessed=Path(row['shared_preprocessed_dataset_directory'])
    require(preprocessed.is_absolute()and preprocessed.name==native['baseline']['dataset_name'],
        'Original complete shared preprocessed Dataset730 required')
    for name,checksum in native['baseline']['source_files_sha256'].items():
        require(digest(preprocessed/name)==checksum,'Original shared dataset metadata changed: '+name)
    inventory=read(pin(row['shared_preprocessed_inventory']))
    require(inventory.get('complete')is True and inventory.get('case_count')==131
        and inventory.get('all131_data_SHA_verified')is True
        and set(inventory['cases'])==set(native['split']['outer_train'])|set(native['split']['outer_val']),
        'Shared actual131-case preprocessing inventory required')
    for path,record in inventory['files'].items():
        require(list(signature(exact(path)))==record['stat'],
            'Previously SHA-verified shared preprocessing file changed: '+path)
        require(record.get('sha256')and Path(path).is_relative_to(preprocessed),
            'Shared preprocessing provenance leaves the admitted readonly dataset')
    verify_historical_reference(cohort,native['split']['outer_val'])
    return native,bank_path,preprocessed


def installed_trainers(cohort):
    """Process-only package path extension; never install or copy trainer files."""
    import nnunetv2.training.nnUNetTrainer as package
    custom=str(ROOT/'custom_trainers')
    if custom not in package.__path__:package.__path__.append(custom)
    base=importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer')
    historical=importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCP')
    require(digest(base.__file__)==NATIVE_TRAINER_SHA and digest(historical.__file__)==HISTORICAL_SHA,
        'Installed full native trainer and historical CP source must match the original bytes')
    return package


def native_for(cohort,arm,source,bank,preprocessed,actor):
    row=cohort['experiments'][arm];root=Path(row['output_root'])
    native=dict(source,root=str(root),native_root=str(root),bank=str(bank),bank_sha256=digest(bank),
        private_pre=str(root/'nnUNet_preprocessed'/source['baseline']['dataset_name']),
        private_data_folder=str(preprocessed/source['baseline']['data_identifier']),
        private_runtime=None,canonical_code_root=str(ROOT),gpu=cohort['physical_parent_index'],
        archive_root=str(root/'completed_native_archive'),archive_storage_kind='tmpfs',persistent_archive=False)
    native['logical_slot']=source['gpu']
    allocation=read(pin(cohort['allocation'],owner=True))
    hardware_keys=('gpu','gpu_uuid','GPU_memory_total_MiB','cpu_affinity','controller_cgroup_admission',
        'shared_resource_registry','shared_resource_guard_version','shared_resource_observation_revision',
        'GPU_admission_retry_version','RAM_admission_pause_revision','RAM512_claim_ledger',
        'RAM512_source_pending_receipt','GPU_placement_receipt','GPU56_sharing_receipt',
        'execution_physical_GPU','execution_GPU_UUID')
    native['original_hardware']={name:source[name]for name in hardware_keys if name in source}
    for name in hardware_keys:native.pop(name,None)
    token=allocation['execution_UUID_by_arm'][arm];device=allocation['execution_device_observations'][token]
    cpus=allocation['CPU_affinity'];offset=execution_arms(allocation).index(arm)*8
    native.update(gpu=allocation['physical_parent_index'],gpu_uuid=allocation['physical_parent_GPU_UUID'],
        execution_physical_GPU=allocation['physical_parent_index'],execution_GPU_UUID=token,
        GPU_memory_total_MiB=device['total_bytes']//2**20,cpu_affinity=cpus[offset:offset+8],
        allocation_CPU_affinity=cpus,execution_device_observation=device,
        physical_parent_observation=allocation['physical_parent_observation'])
    for name in('GPU56_sharing_receipt','GPU_queue_deferred_transfer','RAM512_migration_state','RAM512_completed_best_validator'):
        native.pop(name,None)
    native.update(controller_identity=actor,commit=cohort['canonical_code']['commit'],
        code_sha256=cohort['canonical_code']['files_sha256'],single_parent_allocation=cohort['allocation'],
        host_lifetime=dict(cohort['allocation'],PBS_lifetime=allocation['PBS_lifetime']))
    return native


def inactive_gradient_proof(trainer):
    """Reuse the admitted AST proof for native zero-weight auxiliary heads only."""
    from hiercp_v1x.v24_native_gradient_runtime import clone_step
    from hiercp_v1x.v24_nnunet_cp import _native_clone_step_with_amp_retry
    admitted=clone_step(_native_clone_step_with_amp_retry)
    parameters={name:p for name,p in trainer.network.named_parameters()if p.requires_grad}
    proof=admitted.__globals__['_native_inactive_auxiliary_gradient_proof'](trainer,parameters)
    missing=[name for name,p in parameters.items()if p.grad is None]
    require(set(missing)<=set(proof['official_inactive_parameter_names']),
        'Actual native backward is missing gradients outside proven zero-weight auxiliary heads')
    return dict(proof,actual_missing_gradient_names=missing,
        unexpected_missing_gradient_names=[],all_loss_connected_parameter_gradients_present=True)


def trainable_weight_SHA(trainer):
    """One-time CPU streaming hash adds no duplicate full-model CUDA tensors."""
    checksum=hashlib.sha256()
    for name,value in trainer.network.named_parameters():
        if value.requires_grad:
            checksum.update(name.encode());checksum.update(value.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def train_arm(cohort,arm,actors):
    row=cohort['experiments'][arm];root=Path(row['output_root']);lease=Lease(cohort,arm,actors)
    lease.verify(force=True)
    # nnunetv2.paths binds these at first import, before the trainer modules.
    os.environ.update(nnUNet_preprocessed=str(Path(row['shared_preprocessed_dataset_directory']).parent),
        nnUNet_results=str(root/'nnUNet_results'),nnUNet_raw=str(root/'nnUNet_raw'),
        ONLINE_CP_BANK=str(pin(row['effective_bank'])),ONLINE_CP_SEED='42',BASIC_CP_EXPANDED_ARM=arm,
        BASIC_CP_EXPANDED_PREFLIGHT=str(root/'preflight.json'),BASIC_CP_MULTIN_JOINT_PREFLIGHT=str(root/'preflight.json'),
        nnUNet_n_proc_DA='4',nnUNet_def_n_proc='4',nnUNet_compile='false',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    installed_trainers(cohort)
    source,bank,preprocessed=verify_inputs(cohort,arm);native=native_for(cohort,arm,source,bank,preprocessed,actors[arm])
    require(read(root/'request.json')==native,'Own immutable displayed request differs from actual runtime')
    for name,target in(('preflight',root/'preflight.json'),('calibration',root/'calibration.json')):
        copy_exact(row[name],target)
    Path(native['private_pre']).mkdir(parents=True,exist_ok=True)
    for name in source['baseline']['source_files_sha256']:
        copy_exact({'path':str(preprocessed/name),'sha256':source['baseline']['source_files_sha256'][name]},
            Path(native['private_pre'])/name)
    seal(root/'native.json',native)
    import torch
    torch.set_num_threads(1);torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=True
    require(torch.cuda.is_available()and torch.cuda.device_count()==1,'Exactly the one authorized execution device required')
    name='nnUNetTrainer_BasicCPMultiNJoint'if arm.startswith('multi')else'nnUNetTrainer_BasicCPExpanded'
    module=importlib.import_module('nnunetv2.training.nnUNetTrainer.'+name)
    require(Path(module.__file__).parent==ROOT/'custom_trainers','Custom trainer came from a second code copy')
    cls=getattr(module,native['trainer'])
    from hiercp_v1x import v24_native_best_eval_runtime as best
    from hiercp_v1x.v24_direct_gpu_access import install_native_step_guards
    best_guard=best._install(cls,native['split']['outer_val'],bank_sha256=native['bank_sha256'])
    step_guard=install_native_step_guards(cls,lambda:lease.verify())
    plans=read(preprocessed/(native['baseline']['plans_name']+'.json'));plans['continue_training']=False
    trainer=cls(plans,'3d_fullres',0,read(preprocessed/'dataset.json'),torch.device('cuda'))
    active=[trainer]
    def status(stage,state,**values):
        arm_status(cohort,arm,stage,state,**values)
    try:
        trainer.initialize();train,val=trainer.do_split()
        require(trainer.batch_size==2 and trainer.num_epochs==250 and trainer.num_iterations_per_epoch==250
            and trainer.num_val_iterations_per_epoch==50
            and trainer.configuration_manager.patch_size==[128]*3 and set(train)==set(native['split']['outer_train'])
            and set(val)==set(native['split']['outer_val'])and len(train)==105 and len(val)==26,
            'Actual full native B2/250epochs/250batches/50valbatches/128cube/105train/26validation differs')
        parameters=sum(p.numel()for p in trainer.network.parameters())
        require(parameters==102350575 and {id(p)for p in trainer.network.parameters()if p.requires_grad}
            =={id(p)for group in trainer.optimizer.param_groups for p in group['params']},
            'Full model or optimizer parameter connectivity differs')
        model=dict(trainer=native['trainer'],architecture=trainer.configuration_manager.network_arch_init_kwargs,
            model_class=type(trainer.network).__name__,validation_steps_per_epoch=trainer.num_val_iterations_per_epoch,
            parameters=parameters,trainable_parameters=sum(p.numel()for p in trainer.network.parameters()if p.requires_grad),
            physical_batch=2,effective_batch=2,gradient_accumulation=1,epochs=250,patch_size=[128]*3,workers=4,
            train_steps_per_epoch=250,train_samples_per_epoch=500,actual_train_cases=train,actual_validation_cases=val,
            debug=False,subset=False,
            pretrained_segmentation_used=False,precision='native CUDA autocast/GradScaler',own_BEST_guard=best_guard,
            resource_step_guard=step_guard,canonical_code_root=str(ROOT),no_private_code_copy=True)
        require(torch.cuda.get_device_properties(0).total_memory==native['execution_device_observation']['total_bytes']
            and psutil.Process().cpu_affinity()==native['cpu_affinity'],'Actual CUDA memory or native CPU8 subset differs')
        model['actual_hardware']=dict(physical_parent=native['physical_parent_observation'],
            execution_UUID=native['execution_GPU_UUID'],execution_device=native['execution_device_observation'],
            native_CPU_affinity=psutil.Process().cpu_affinity(),allocation_CPU_affinity=native['allocation_CPU_affinity'],
            native_RSS_limit_GiB=32,aggregate_RAM_limit_GiB=512,auxiliary_RSS_limit_GiB=16,
            original_hardware_preserved_in_native_metadata=True,allocation_receipt=cohort['allocation'])
        allocation=lease.receipt
        model['actual_hardware'].update(execution_mode=allocation['mode'],
            PBS_GPU_owned_reservation=allocation.get('PBS_GPU_owned_reservation',True),
            PBS_CPU_owned_reservation=allocation.get('PBS_CPU_owned_reservation',True),
            PBS_job_role=allocation.get('PBS_job_role','execution_allocation'),
            host_allocation_process=allocation.get('host_allocation_process',allocation['holder_identity']))
        seal(root/'actual_model.json',model)
        goal=Path(trainer.output_folder);goal.mkdir(parents=True,exist_ok=True)
        latest=copy_exact(row['LATEST'],goal/'checkpoint_latest.pth');copy_exact(row['BEST'],goal/'checkpoint_best.pth')
        trainer.load_checkpoint(str(latest))
        require(trainer.current_epoch==row['completed_epochs'],'Loaded whole checkpoint epoch differs')
        original_step=trainer.train_step;first_step=[]
        def train_step(batch):
            require(tuple(batch['data'].shape)==(2,1,128,128,128),'Actual full nativeB2 input shape differs')
            if not first_step:
                before=trainable_weight_SHA(trainer);scale_before=float(trainer.grad_scaler.get_scale())
            result=original_step(batch)
            if not first_step:
                require(math.isfinite(float(result['loss'])),'Actual first native loss is nonfinite')
                gradient=inactive_gradient_proof(trainer);scale_after=float(trainer.grad_scaler.get_scale())
                require(math.isfinite(scale_after)and scale_after>0,'Actual native GradScaler state is invalid')
                after=trainable_weight_SHA(trainer);skipped=scale_after<scale_before
                require(not skipped or before==after,'Native AMP skipped step unexpectedly changed model weights')
                if not skipped:
                    require(before!=after,'Native successful optimizer path produced no trainable weight change')
                    seal(root/'actual_first_production_step.json',dict(actual_data=True,physical_batch=2,
                        shape=[2,1,128,128,128],loss=float(result['loss']),gradient_proof=gradient,
                        weight_SHA256_before=before,weight_SHA256_after=after,model_weights_changed=True,
                        scaler_before=scale_before,scaler_after=scale_after,native_AMP_skipped_update=False,
                        actual_successful_optimizer_update=True,debug=False,time=time.time()))
                    first_step.append(True)
            return result
        trainer.train_step=train_step
        original_end=trainer.on_epoch_end
        def epoch_end():
            result=original_end()
            if 0<trainer.current_epoch<250:
                from hiercp_v1x.basic_cp_completed_epoch_checkpoint import save_completed_epoch
                temporary=goal/('whole_epoch_'+str(time.time_ns())+'.pth')
                save_completed_epoch(trainer,str(temporary));os.replace(temporary,goal/'checkpoint_latest.pth')
            status('train','RUNNING',completed_epochs=int(trainer.current_epoch),
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_reserved_cuda_bytes=torch.cuda.max_memory_reserved())
            return result
        trainer.on_epoch_end=epoch_end
        status('train','RUNNING',completed_epochs=trainer.current_epoch)
        trainer.run_training()
        final=goal/'checkpoint_final.pth'
        require(final.exists()and torch.load(final,map_location='cpu',weights_only=False)['current_epoch']==250,
            'Full250 FINAL is required as training witness')
        lease.verify(force=True);trainer.load_checkpoint(str(goal/'checkpoint_best.pth'))
        status('train','RUNNING',completed_epochs=250,native_substage='own_BEST26')
        trainer.perform_actual_validation(save_probabilities=False)
        receipt=best.validate_best_validation_receipt(goal/best.RECEIPT,fold=goal,
            expected_cases=native['split']['outer_val'],bank_sha256=native['bank_sha256'])
        proof=dict(epochs=250,checkpoint='checkpoint_best.pth',best_sha256=receipt['checkpoint_sha256'],
            validation_cases=receipt['validation_cases'],fold=str(goal),final_training_witness_sha256=digest(final),
            evaluation_checkpoint_is_final=False)
        seal(root/'training_and_BEST26_complete.json',proof)
        status('historical_metrics','RUNNING',completed_epochs=250)
        verify_historical_reference(cohort,native['split']['outer_val'])
        command=[sys.executable,'-B','-u',str(ROOT/'tools/evaluate_v24_historical_metrics.py'),
            '--evaluator-root',row['evaluator_root'],'--prediction',str(goal/'validation'),'--output',str(root/'evaluation'),
            '--project',row['project'],'--checkpoint','BEST','--trainer',native['trainer'],
            '--model-label','Basic CP '+arm,'--gpu-label','GPU'+str(cohort['physical_parent_index'])]
        subprocess.run(command,check=True,env=dict(os.environ,PYTHONPATH=str(ROOT)))
        from tools.basic_cp_expanded_result import completed_report,archive_completed
        status('report','RUNNING',completed_epochs=250)
        reports=completed_report(native,proof);archive=archive_completed(native,proof,reports)
        status('complete','COMPLETE',completed_epochs=250,evaluated_cases=26,full_native_training=True,
            full_native_evaluation=True,historical_metrics_complete=True,reports=reports,archive=archive)
    except BaseException as error:
        record_child_failure(cohort,arm,error)
        raise
    finally:
        seen=set()
        for instance in active:
            for loader in(getattr(instance,'dataloader_train',None),getattr(instance,'dataloader_val',None)):
                if loader is not None and id(loader)not in seen:
                    seen.add(id(loader));finish=getattr(loader,'_finish',None)
                    if finish is not None:finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--cohort-sha256',required=True);parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    require(args.execute,'Explicit --execute required for production continuation')
    cohort=validate_cohort(read(pin({'path':str(args.cohort),'sha256':args.cohort_sha256},owner=True)))
    cohort.update(_execution_cohort_path=str(args.cohort),_execution_cohort_SHA256=args.cohort_sha256)
    allocation=read(pin(cohort['allocation'],owner=True));arms=execution_arms(allocation)
    require(arms==continuation_arms(cohort),'Allocation arms differ from SHA-pinned unfinished cohort')
    root=Path(cohort['root']);root.mkdir(mode=0o700,exist_ok=True)
    require(root.stat().st_uid==41833 and root.resolve()==root,'Own fresh cohort output root required')
    registry=root/'single_parent_actors.json'
    if args.arm:
        require(args.arm in arms,'Completed or unbound experiment cannot be dispatched')
        try:
            index=arms.index(args.arm);psutil.Process().cpu_affinity(allocation['CPU_affinity'][index*8:index*8+8])
            deadline=time.monotonic()+120
            while not registry.exists():
                require(time.monotonic()<deadline,'Exact actor registry not published within120seconds');time.sleep(.1)
            train_arm(cohort,args.arm,read(exact(registry,owner=True)))
        except BaseException as error:
            record_child_failure(cohort,args.arm,error)
            raise
        return
    require(not registry.exists(),'Existing production dispatch requires inspection; no duplicate launch')
    holder,memory,holder_observation=observe_holder(allocation)
    own_memory=_memory_cgroup(os.getpid())
    direct=allocation.get('mode')==DIRECT_MODE
    require(os.getuid()==41833 and cgroup_identity(memory)==cgroup_identity(own_memory)
        and(direct or allocation['job_id']in memory['path'])
        and set(allocation['CPU_affinity'])<=set(psutil.Process().cpu_affinity()),
        'Production launcher must retain its admitted own cgroup and48 allowed CPUs')
    query_started=time.monotonic()
    document=json.loads(subprocess.check_output(['/opt/pbs/bin/qstat','-f','-F','json',allocation['job_id']],text=True,timeout=20))
    require(set(document['Jobs'])=={allocation['job_id']},'Exact own actual PBS job required')
    devices=nvml_devices(allocation['execution_UUID_by_arm'].values())
    validate_allocation(allocation,dict(document['Jobs'][allocation['job_id']],job_id=allocation['job_id']),
        holder_observation,devices,now_monotonic=time.monotonic(),
        node_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),query_started_monotonic=query_started)
    require(all(not row['compute_actors']for row in devices.values()),
        'Initial owned execution devices must be free of unknown existing CUDA actors')
    if direct:
        from hiercp_v1x.basic_cp_direct_MIG_access import observe_direct_memory
        prospective=observe_direct_memory({arm:0 for arm in arms},
            {holder.pid,*[p.pid for p in holder.children(recursive=True)]},expected_arms=len(arms))
    processes={};actors={}
    for index,arm in enumerate(arms):
        row=cohort['experiments'][arm];output=Path(row['output_root']);output.mkdir(mode=0o700)
        arm_status(cohort,arm,'prepare','RUNNING',completed_epochs=row['completed_epochs'],
            native_substage='admission_and_full_source_verification')
        copy_exact(row['log_prefix'],output/'train.log')
        command=[sys.executable,'-B','-u',str(Path(__file__)), '--cohort',str(args.cohort),
            '--cohort-sha256',args.cohort_sha256,'--arm',arm,'--execute']
        env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES=allocation['execution_UUID_by_arm'][arm],
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with(output/'train.log').open('ab',buffering=0)as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        process=psutil.Process(child.pid);process.cpu_affinity(allocation['CPU_affinity'][index*8:index*8+8])
        actors[arm]=identity(process);processes[arm]=child
        seal(output/'controller_spawned_actor.json',actors[arm])
        original=read(pin(row['source_native']))
        displayed=native_for(cohort,arm,original,pin(row['effective_bank']),
            Path(row['shared_preprocessed_dataset_directory']),actors[arm])
        seal(output/'request.json',displayed)
        seal(output/'train_actor.json',dict(pid=process.pid,birth=actors[arm]['create_time'],UID=41833,
            command=command,log=str(output/'train.log'),stage='train'))
    seal(registry,actors)
    seal(root/'launch.json',dict(group=str(root),code=str(ROOT),commit=cohort['canonical_code']['commit'],
        code_sha256=cohort['canonical_code']['files_sha256'],registration={str(cohort['experiments'][arm]['logical_slot']):
            dict(root=cohort['experiments'][arm]['output_root'],arm=arm,controller_identity=actors[arm],
                request_sha256=digest(Path(cohort['experiments'][arm]['output_root'])/'request.json'))for arm in arms}|
            {str(record['logical_slot']):dict(root=record['output_root'],arm=arm,
                request_sha256=record['request']['sha256'],retained_complete=True)
                for arm,record in cohort.get('retained_completion_evidence',{}).items()},
        retained_complete_arms=cohort['retained_complete_arms'],UI_version=2,canonical_code_shared_by_all_arms=True,
        source_arrays_copied=False,private_runtime_copies_created=False,full_training_complete_claimed=False))
    while any(p.poll()is None for p in processes.values()):time.sleep(2)
    result={arm:p.returncode for arm,p in processes.items()}
    seal(root/'single_parent_returncodes.json',result)
    require(all(code==0 for code in result.values()),'Some continuations failed; inspect preserved native logs: '+str(result))


if __name__=='__main__':main()
