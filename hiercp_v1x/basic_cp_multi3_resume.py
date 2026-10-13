"""Continue the same GPU1 experiment from a verified completed BEST checkpoint."""
from pathlib import Path
import os
import shutil
from . import basic_cp_ablation_resume as base

TRAINER = 'nnUNetTrainer_250epochs_BasicCPMulti3'


def prepare_resume(native):
    import torch
    w = native['completed_native_resume']
    base.require(set(w) == {'root','request_sha256','controller_identity',
        'checkpoint_sha256','best_sha256','completed_epochs'} and
        type(w['completed_epochs']) is int and 0 < w['completed_epochs'] < 250,
        'Exact completed multi3 continuation witness required')
    epoch = w['completed_epochs']
    base.require(native['gpu'] == 1 and native['arm'] == 'multi3' and native['trainer'] == TRAINER,
                 'Only the original GPU1 multi3 experiment may resume here')
    root = base.exact(native['root'], directory=True)
    oldroot = base.exact(w['root'], directory=True)
    base.require(oldroot.is_relative_to(base.SOURCE_NAMESPACE) and root != oldroot
        and root not in oldroot.parents and oldroot not in root.parents,
        'Separate original and recovery namespaces required')
    observed = {}
    def admit(path, expected=None):
        path=base.exact(path,root=oldroot)
        before=base.snapshot(path); digest=base.sha(path)
        base.require(before==base.snapshot(path) and (expected is None or digest==expected),
                     'Original multi3 evidence changed: '+str(path))
        observed[str(path)]=before
        return path,digest
    request_path,_=admit(oldroot/'request.json',w['request_sha256'])
    previous=base.read(request_path)
    base.require(previous['controller_identity']==w['controller_identity'] and
        previous['gpu']==1 and previous['arm']=='multi3','Original controller/request differs')
    base.released(w['controller_identity'])
    actor_path,_=admit(oldroot/'train_actor.json');actor=base.read(actor_path)
    base.released(dict(pid=actor['pid'],create_time=actor['birth'],UID=actor['UID'],command=actor['command']))
    oldpath,_=admit(oldroot/'native.json');old=base.read(oldpath)
    for key in ('gpu','arm','trainer','bank','bank_sha256','baseline','split','historical_trainer_sha256',
                'readonly_admission','readonly_admission_sha256','shared_preprocessed_cases',
                'source_preprocessed_arrays_written','full_training_cases','full_validation_cases',
                'CP_probability','seed','sources','candidates_per_source','copies_per_CP_event'):
        base.require(old.get(key)==native.get(key),'Original multi3 scientific contract differs: '+key)
    base.require(native['bank_sha256']==base.BANK_SHA and native['historical_trainer_sha256']==base.HISTORICAL_SHA
        and native['full_training_cases']==105 and native['full_validation_cases']==26
        and native['shared_preprocessed_cases']==131 and native['source_preprocessed_arrays_written'] is False,
        'Original full105/26 and shared131 preprocessing required')
    modelpath,_=admit(oldroot/'actual_model.json');model=base.read(modelpath)
    base.require(model['trainer']==TRAINER and model['epochs']==250 and model['physical_batch']==2
        and model['effective_batch']==2 and model['gradient_accumulation']==1 and model['workers']==4
        and model['patch_size']==[128]*3 and model['debug'] is False and model['subset'] is False,
        'Original full native training settings required')
    for name in ('nnUNetTrainer_OnlinePairedCP.py','nnUNetTrainer_BasicCPAblations.py'):
        relative=Path('nnunetv2/training/nnUNetTrainer')/name
        _,digest=admit(Path(old['private_runtime'])/relative)
        base.require(base.sha(Path(native['private_runtime'])/relative)==digest,'Scientific trainer bytes changed')
    source,digest=admit(base.fold(old)/'checkpoint_best.pth',w['checkpoint_sha256'])
    base.require(digest==w['best_sha256'],'The completed BEST is the exact continuation source')
    saved=torch.load(source,map_location='cpu',weights_only=False)
    state=base.validate_checkpoint(saved,epoch=epoch,trainer=TRAINER)
    del saved
    log,logsha=admit(oldroot/'train.log')
    prefix,prefix_record=base.completed_log_prefix(log.read_bytes(),epoch)
    prefix_record['last_epoch']=epoch-1
    destination=base.fold(native)
    base.require(not destination.exists(),'Fresh independent continuation fold required')
    destination.mkdir(parents=True)
    copies={}
    for name in ('checkpoint_latest.pth','checkpoint_best.pth'):
        target=destination/name
        with source.open('rb') as f,target.open('xb') as out:
            shutil.copyfileobj(f,out,8*2**20);out.flush();os.fsync(out.fileno())
        base.require(base.sha(target)==digest and base.snapshot(target)[:2]!=observed[str(source)][:2],
                     'Exact independent checkpoint copy required')
        copies[name]=dict(path=str(target),sha256=digest)
    target=root/'resumed_training_prefix.log'
    with target.open('xb') as f:f.write(prefix);f.flush();os.fsync(f.fileno())
    base.require(all(base.snapshot(p)==v for p,v in observed.items()),'Original evidence changed during recovery')
    receipt=dict(format='basic_cp_multi3_completed_checkpoint_resume_v1',completed_epochs=epoch,
        native_goal_epochs=250,remaining_epochs=250-epoch,checkpoint=copies['checkpoint_latest.pth'],
        BEST=copies['checkpoint_best.pth'],training_prefix_sha256=base.sha(target),
        training_prefix=prefix_record,source_train_log_sha256=logsha,source_files=observed,
        optimizer_and_scaler_preserved=True,old_results_written=False,full_training_complete=False,
        latest_state=state,source_root=str(oldroot),partial_recovery_checkpoint_used=False)
    base.publish(root/'resume.json',receipt)
    return receipt
