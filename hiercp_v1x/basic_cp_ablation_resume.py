"""Verified GPU6 Basic blend continuation; original results stay read-only."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import stat
import os

import psutil

UID = 41833
SOURCE_NAMESPACE = Path('/dev/shm/aicompetition06_basic_cp_ablation_20261013')
HISTORICAL_SHA = 'a3df56435fedff46fbc42a502596fdade2632d0211ea412630dba1555f32567d'
BANK_SHA = '9e5be3e3ab8f04fd23c166b3267bc4efbfb4b321ef974920dfc08fbf5dd275e6'
TRAINER = 'nnUNetTrainer_250epochs_BasicCPBlend'
FORMAT = 'basic_cp_verified_native_epoch200_resume_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot(path):
    value = Path(path).stat()
    return [value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,
            value.st_nlink,value.st_uid,value.st_gid,value.st_mode]


def exact(path, *, root=None, directory=False):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and path.resolve(strict=True) == path
            and not any(p.is_symlink() for p in (path,*path.parents)), 'Exact non-symlink resume evidence path required')
    info = path.stat()
    require(info.st_uid == UID and (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)),
            'Own UID regular resume evidence required')
    if root is not None:
        require(path != root and path.is_relative_to(root), 'Resume evidence escapes own namespace')
    return path


def read(path):
    value = json.loads(Path(path).read_text(encoding='utf8'))
    require(isinstance(value,dict), 'Resume JSON object required')
    return value


def publish(path, value):
    with Path(path).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())


def released(identity):
    require(isinstance(identity,dict) and set(identity)=={'pid','create_time','UID','command'}
            and type(identity['pid']) is int and identity['pid']>0 and identity['UID']==UID
            and type(identity['create_time']) in (int,float) and math.isfinite(identity['create_time'])
            and identity['create_time']>0 and isinstance(identity['command'],list)
            and identity['command'] and all(isinstance(v,str) and v for v in identity['command']),
            'Exact previous own controller identity required')
    try:
        process=psutil.Process(identity['pid'])
        if process.create_time()!=identity['create_time']:
            return  # PID reused: the registered predecessor has disappeared.
        require(process.uids().real==UID, 'Registered previous controller UID changed')
        raise RuntimeError('Registered previous controller still present; resume refused')
    except psutil.NoSuchProcess:
        return


def fold(native):
    return Path(native['native_root'])/'nnUNet_results'/native['baseline']['dataset_name']/(
        native['trainer']+'__'+native['baseline']['plans_name']+'__3d_fullres')/'fold_0'


def validate_checkpoint(saved, *, epoch, trainer):
    import torch
    require(saved.get('current_epoch')==epoch and saved.get('trainer_name')==trainer,
            'Verified completed checkpoint epoch/trainer differs; partial recovery is forbidden')
    weights=saved.get('network_weights')
    require(isinstance(weights,dict) and weights and all(isinstance(k,str) and torch.is_tensor(v)
            and v.numel()>0 for k,v in weights.items()), 'Actual full network tensor state required')
    optimizer=saved.get('optimizer_state')
    require(isinstance(optimizer,dict) and isinstance(optimizer.get('state'),dict) and optimizer['state']
            and isinstance(optimizer.get('param_groups'),list) and optimizer['param_groups']
            and all(isinstance(group,dict) and group.get('params') for group in optimizer['param_groups']),
            'Actual optimizer momentum/state and parameter groups required')
    scaler=saved.get('grad_scaler_state')
    require(isinstance(scaler,dict) and scaler and 'scale' in scaler
            and math.isfinite(float(scaler['scale'])) and scaler['scale']>0,
            'Actual native CUDA GradScaler state required')
    require(isinstance(saved.get('init_args'),dict) and saved['init_args']
            and '_best_ema' in saved and math.isfinite(float(saved['_best_ema'])),
            'Native initialization and BEST-selection state required')
    logging=saved.get('logging')
    require(isinstance(logging,dict), 'Actual native checkpoint logger required')
    for key in ('train_losses','val_losses','dice_per_class_or_region',
                'epoch_start_timestamps','epoch_end_timestamps'):
        values=logging.get(key)
        require(isinstance(values,(list,tuple)) and len(values)==epoch,
                'Exactly completed checkpoint logger epochs required: '+key)
    starts,ends=logging['epoch_start_timestamps'],logging['epoch_end_timestamps']
    require(all(math.isfinite(float(a)) and math.isfinite(float(b)) and b>=a for a,b in zip(starts,ends)),
            'Checkpoint contains an unfinished epoch timestamp')
    return {'current_epoch':epoch,'logging_epochs':epoch,'optimizer_state_entries':len(optimizer['state']),
            'optimizer_param_groups':len(optimizer['param_groups']),
            'network_tensor_keys':sorted(weights),'grad_scaler_preserved':True,
            'best_selection_state_preserved':True}


def completed_log_prefix(data, epochs=200):
    text=data.decode('utf8')
    headers=list(re.finditer(r'(?m)^.*?\bEpoch\s+(\d+)\s*$',text))
    boundary=next((h.start() for h in headers if int(h.group(1))==epochs),None)
    require(boundary is not None, 'Original log must expose the complete Epoch200 boundary')
    prefix=text[:boundary]
    prefix_headers=[int(h.group(1)) for h in re.finditer(r'(?m)^.*?\bEpoch\s+(\d+)\s*$',prefix)]
    require(prefix_headers==list(range(epochs)), 'Original complete epoch headers must be unique0..199')
    counts={}
    for epoch,applied,total in re.findall(r'\[OnlineCP\] epoch=(\d+) applied=(\d+)/(\d+)',prefix):
        epoch,applied,total=int(epoch),int(applied),int(total)
        require(epoch not in counts and total>0 and 0<=applied<=total, 'Duplicate/invalid original OnlineCP epoch counter')
        counts[epoch]=(applied,total)
    require(set(counts)==set(range(epochs)), 'Exactly200 complete original OnlineCP counters required')
    # Slice bytes at the same UTF8 boundary; timestamps and original lines stay exact.
    return prefix.encode('utf8'), {'epochs':epochs,'first_epoch':0,'last_epoch':199,
        'CP_applied':sum(v[0] for v in counts.values()),'CP_attempted':sum(v[1] for v in counts.values())}


def prepare_resume(native):
    import torch
    witness=native.get('completed_native_resume')
    require(isinstance(witness,dict) and set(witness)=={'root','request_sha256','controller_identity',
        'checkpoint_sha256','best_sha256','completed_epochs'} and witness['completed_epochs']==200,
        'Explicit exact completed native epoch200 resume witness required')
    require(native.get('gpu')==6 and native.get('arm')=='blend' and native.get('trainer')==TRAINER
            and native.get('full_training_cases')==105 and native.get('full_validation_cases')==26
            and native.get('shared_preprocessed_cases')==131 and native.get('source_preprocessed_arrays_written') is False,
            'Only complete matched GPU6 blend105/26/B2 pipeline can resume')
    root=exact(native['root'],directory=True);new_native=exact(native['native_root'],root=root,directory=True)
    old_root=exact(witness['root'],directory=True)
    require(old_root.is_relative_to(SOURCE_NAMESPACE) and old_root!=root
            and old_root not in root.parents and root not in old_root.parents, 'Separate own historical/fresh resume namespaces required')
    observed={}
    def admit(path, expected=None, source_root=old_root):
        path=exact(path,root=source_root)
        before=snapshot(path);digest=sha(path)
        require(snapshot(path)==before and (expected is None or digest==expected), 'Read-only resume source SHA/stat changed: '+str(path))
        observed[str(path)]=before
        return path,digest
    request_path,_=admit(old_root/'request.json',witness['request_sha256'])
    old_request=read(request_path)
    require(old_request.get('uid')==UID and old_request.get('gpu')==6 and old_request.get('arm')=='blend'
            and old_request.get('controller_identity')==witness['controller_identity'], 'Previous own GPU6 request/controller differs')
    released(witness['controller_identity'])
    old_native_path,_=admit(old_root/'native.json');old=read(old_native_path)
    for key in ('gpu','arm','trainer','bank','bank_sha256','baseline','split','historical_trainer_sha256',
                'readonly_admission','readonly_admission_sha256','shared_preprocessed_cases',
                'source_preprocessed_arrays_written','full_training_cases','full_validation_cases'):
        require(old.get(key)==native.get(key), 'Original matched native scientific identity differs: '+key)
    require(old.get('root')==str(old_root) and old.get('native_root')==str(old_root/'native')
            and old.get('bank_sha256')==BANK_SHA and old.get('historical_trainer_sha256')==HISTORICAL_SHA,
            'Original Basic128/historical trainer/own native root differs')
    require(len(native['split']['outer_train'])==len(set(native['split']['outer_train']))==105
            and len(native['split']['outer_val'])==len(set(native['split']['outer_val']))==26,
            'Full original105train/26validation identities required')
    model_path,_=admit(old_root/'actual_model.json');model=read(model_path)
    require(model.get('trainer')==TRAINER and model.get('epochs')==250
            and model.get('physical_batch')==model.get('effective_batch')==2
            and model.get('gradient_accumulation')==1 and model.get('patch_size')==[128]*3
            and model.get('workers')==4 and model.get('debug') is False and model.get('subset') is False
            and model.get('pretrained_segmentation_used') is False
            and sorted(model.get('actual_train_cases',[]))==sorted(native['split']['outer_train'])
            and sorted(model.get('actual_validation_cases',[]))==sorted(native['split']['outer_val']),
            'Original actual full-model250/B2/105/26 evidence differs')
    for name,expected in [('nnUNetTrainer_OnlinePairedCP.py',HISTORICAL_SHA),
                          ('nnUNetTrainer_BasicCPAblations.py',None)]:
        oldfile=Path(old['private_runtime'])/'nnunetv2/training/nnUNetTrainer'/name
        _,digest=admit(oldfile,expected)
        newfile=exact(Path(native['private_runtime'])/'nnunetv2/training/nnUNetTrainer'/name,root=new_native)
        require(sha(newfile)==digest, 'Fresh native scientific trainer bytes must equal original')
    oldfold=exact(fold(old),root=old_root,directory=True)
    latest,latest_sha=admit(oldfold/'checkpoint_latest.pth',witness['checkpoint_sha256'])
    best,best_sha=admit(oldfold/'checkpoint_best.pth',witness['best_sha256'])
    saved=torch.load(latest,map_location='cpu',weights_only=False)
    latest_record=validate_checkpoint(saved,epoch=200,trainer=TRAINER)
    best_saved=torch.load(best,map_location='cpu',weights_only=False)
    require(type(best_saved.get('current_epoch')) is int and 0<best_saved['current_epoch']<=200,
            'Original BEST epoch must belong to the verified completed prefix')
    best_record=validate_checkpoint(best_saved,epoch=best_saved['current_epoch'],trainer=TRAINER)
    require(set(saved['network_weights'])==set(best_saved['network_weights'])
            and all(saved['network_weights'][key].shape==best_saved['network_weights'][key].shape
                    for key in saved['network_weights']), 'Original latest/BEST architecture differs')
    del saved,best_saved
    log,log_sha=admit(old_root/'train.log')
    prefix,prefix_record=completed_log_prefix(log.read_bytes())
    destination=fold(native)
    require(destination.is_relative_to(new_native) and not destination.exists(), 'Fresh untouched resume fold required after isolated debug')
    destination.mkdir(parents=True,exist_ok=False)
    copies={}
    for source,digest in [(latest,latest_sha),(best,best_sha)]:
        target=destination/source.name
        with source.open('rb') as stream,target.open('xb') as output:
            shutil.copyfileobj(stream,output,8*2**20);output.flush();os.fsync(output.fileno())
        require(sha(target)==digest and snapshot(target)[:2]!=observed[str(source)][:2], 'Resume copy must be byte exact and a separate inode')
        copies[source.name]={'path':str(target),'sha256':digest,'bytes':target.stat().st_size}
    prefix_path=root/'resumed_training_prefix.log'
    with prefix_path.open('xb') as stream:stream.write(prefix);stream.flush();os.fsync(stream.fileno())
    require(all(snapshot(Path(path))==identity for path,identity in observed.items()), 'Original resume evidence changed during copy')
    released(witness['controller_identity'])
    receipt={'format':FORMAT,'completed_epochs':200,'planned_epochs':250,'native_goal_epochs':250,'remaining_epochs':50,
        'previous_full_completed_epochs':215,'rollback_completed_epochs':15,'discarded_partial_raw_epoch':216,
        'rollback_reason':'Continue from verified periodic completed200 checkpoint; replay later15 complete epochs and unfinished raw216',
        'checkpoint':copies['checkpoint_latest.pth'],'BEST':copies['checkpoint_best.pth'],
        'checkpoint_current_epoch':200,'BEST_current_epoch_raw':best_record['current_epoch'],
        'latest_state':latest_record,'BEST_state':best_record,'source_root':str(old_root),
        'source_request_sha256':witness['request_sha256'],'source_controller_identity':witness['controller_identity'],
        'source_train_log_sha256':log_sha,'training_prefix':dict(prefix_record,path=str(prefix_path),sha256=sha(prefix_path)),
        'training_prefix_sha256':sha(prefix_path),
        'original_scientific_trainer_and_bank_preserved':True,'optimizer_and_scaler_preserved':True,
        'old_results_written':False,'new_copies_independent_inodes':True,'full_training_complete':False,
        'source_files':observed}
    publish(root/'resume.json',receipt)
    return receipt
