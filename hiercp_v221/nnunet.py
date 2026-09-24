"""Isolated native nnU-Net plan/train/predict stages; v1 runtime is untouched."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from . import PIPELINE_VERSION
from .contracts import ROOT,read_json,write_new,sha,safe_new_root,validate_split,verify_v1,validate_native

TRAINER='nnUNetTrainer_250epochs_OnlinePromptGraphV221'
PLANS='nnUNetResEncUNetMPlans'

def run(command,env,log):
    with Path(log).open('x',encoding='utf-8') as f:
        result=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
    if result.returncode: raise RuntimeError(f'Stage failed ({result.returncode}); preserved log: {log}')

def executable(name):
    value=shutil.which(name)
    if value is None: raise RuntimeError(f'Missing native environment command: {name}')
    return value

def runtime(root):
    import nnunetv2
    verify_v1()
    package=Path(nnunetv2.__file__).parent
    destination=Path(root)/'runtime/nnunetv2'
    if destination.exists(): raise ValueError('Private runtime already exists; refusing replacement')
    shutil.copytree(package,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    trainer=destination/'training/nnUNetTrainer'
    for name in ('nnUNetTrainer_OnlinePairedCP.py','onlinecp_raw_bank.py','onlinecp_raw_resampling.py'):
        path=trainer/name
        if path.exists():
            if sha(path)!=sha(ROOT/'custom_trainers'/name):
                raise ValueError(f'Conflicting private runtime source: {name}')
        else: shutil.copy2(ROOT/'custom_trainers'/name,path)
    path=trainer/'nnUNetTrainer_OnlinePromptGraphV221.py'
    if path.exists(): raise ValueError('Conflicting pre-existing v2 trainer')
    shutil.copy2(ROOT/'hiercp_v221/nnUNetTrainer_OnlinePromptGraphV221.py',path)
    return destination.parent

def environment(root,runtime_dir=None):
    root=Path(root); env=os.environ.copy()
    env.update(nnunet_raw=str(root/'nnUNet_raw'),nnunet_preprocessed=str(root/'nnUNet_preprocessed'),nnunet_results=str(root/'nnUNet_results'))
    # nnU-Net environment variable spelling is case-sensitive on Linux.
    env['nnUNet_raw']=env.pop('nnunet_raw'); env['nnUNet_preprocessed']=env.pop('nnunet_preprocessed'); env['nnUNet_results']=env.pop('nnunet_results')
    paths=[str(ROOT)]
    if runtime_dir: paths.insert(0,str(runtime_dir))
    if env.get('PYTHONPATH'): paths.append(env['PYTHONPATH'])
    env['PYTHONPATH']=os.pathsep.join(paths); env['nnUNet_n_proc_DA']='8'
    return env

def prepare(medical,split_path,output,dataset_id,cfg):
    from hiercp.common import discover_cases
    split=read_json(split_path); validate_split(split)
    paths={c.case_id:c for c in discover_cases(Path(medical)/'Data')}
    if set(paths)!=set(split['outer_train']+split['outer_val']): raise ValueError('Raw cohort/split mismatch')
    for name in ('nnUNetv2_extract_fingerprint','nnUNetv2_plan_experiment','nnUNetv2_preprocess'): executable(name)
    root=safe_new_root(output)
    if shutil.disk_usage(root).free<cfg['minimum_free_gb']*1024**3: raise RuntimeError('Native preparation disk reserve not available')
    name=f'Dataset{int(dataset_id):03d}_PromptGraphV22'; env=environment(root)
    full=root/'nnUNet_raw'/name; fitting=root/'planning_raw'/name
    records=[]
    for folder in (full,fitting):
        (folder/'imagesTr').mkdir(parents=True); (folder/'labelsTr').mkdir()
    for case in split['outer_train']+split['outer_val']:
        record={'case_id':case}
        for key,sub,file in (('image','imagesTr',paths[case].image_path),('label','labelsTr',paths[case].label_path)):
            dest=full/sub/file.name; shutil.copy2(file,dest)
            record[key]=str(file.resolve()); record[key+'_sha256']=sha(file)
            if sha(dest)!=record[key+'_sha256']: raise ValueError('Raw copy/source hash mismatch')
            if case in split['outer_train']: shutil.copy2(file,fitting/sub/file.name)
        records.append(record)
    for folder,ids in ((full,split['outer_train']+split['outer_val']),(fitting,split['outer_train'])):
        write_new(folder/'dataset.json',{'channel_names':{'0':'CT'},'labels':{'background':0,'liver':1,'tumor':2},
                   'numTraining':len(ids),'file_ending':'.nii.gz'})
    fit_env=dict(env,nnUNet_raw=str(root/'planning_raw'))
    run([executable('nnUNetv2_extract_fingerprint'),'-d',str(dataset_id),'--verify_dataset_integrity'],fit_env,root/'fingerprint.log')
    run([executable('nnUNetv2_plan_experiment'),'-d',str(dataset_id),'-pl','nnUNetPlannerResEncM'],fit_env,root/'plan.log')
    run([executable('nnUNetv2_preprocess'),'-d',str(dataset_id),'-plans_name',PLANS,'-c','3d_fullres','-np','4','--no_pbar'],env,root/'preprocess.log')
    pre=root/'nnUNet_preprocessed'/name
    write_new(pre/'splits_final.json',[{'train':split['outer_train'],'val':split['outer_val']}])
    # Native fold 0 here means this explicitly selected outer split, not all folds completed.
    payload={'format':PIPELINE_VERSION,'complete':True,'root':str(root),'medical':str(Path(medical).resolve()),
             'dataset_id':int(dataset_id),'dataset_name':name,'split':split,'preprocessed':str(pre),
             'plans':str(pre/f'{PLANS}.json'),'raw':str(full),'raw_records':records,
             'planning_patient_ids':split['outer_train'],'native_fold':0}
    payload['plans_sha256']=sha(payload['plans']); payload['splits_sha256']=sha(pre/'splits_final.json')
    write_new(root/'native.json',payload)
    return root/'native.json'

def train(native_path,bank_path):
    from .training import require_device
    require_device('cuda')
    native=validate_native(read_json(native_path)); bank=read_json(bank_path)
    if bank.get('debug'):raise ValueError('DEBUG bank cannot start production segmentation training')
    if native['format']!=PIPELINE_VERSION or bank.get('pipeline_version')!=PIPELINE_VERSION or bank['split']!=native['split']:
        raise ValueError('v2 native/bank identities differ')
    if bank['native_sha256']!=sha(native_path): raise ValueError('Native preparation changed after bank publication')
    root=Path(native['root']); private=runtime(root); env=environment(root,private)
    env.update(nnUNet_raw=str(Path(native['raw']).parent),nnUNet_preprocessed=str(Path(native['preprocessed']).parent))
    env.update(ONLINE_CP_BANK=str(Path(bank_path).resolve()),ONLINE_CP_SEED='42')
    model=root/'nnUNet_results'/native['dataset_name']/f'{TRAINER}__{PLANS}__3d_fullres'
    if model.exists(): raise ValueError('Existing v2 segmentation results are preserved; no fresh overwrite')
    write_new(root/'training_started.json',{'format':PIPELINE_VERSION,'bank':str(Path(bank_path).resolve()),'bank_sha256':sha(bank_path),
                                          'native_sha256':sha(native_path),'epochs':250,'runtime':str(private)})
    run([executable('nnUNetv2_train'),str(native['dataset_id']),'3d_fullres','0','-tr',TRAINER,'-p',PLANS],env,root/'train.log')
    checkpoint=model/'fold_0/checkpoint_final.pth'
    write_new(root/'training_complete.json',{'format':PIPELINE_VERSION,'checkpoint':str(checkpoint),'sha256':sha(checkpoint),
                                            'bank_sha256':sha(bank_path),'native_sha256':sha(native_path)})
    return checkpoint

def predict(native_path,output):
    native=validate_native(read_json(native_path)); root=Path(native['root']); receipt=read_json(root/'training_complete.json')
    if receipt['native_sha256']!=sha(native_path): raise ValueError('Native preparation differs from trained model')
    if receipt['sha256']!=sha(receipt['checkpoint']): raise ValueError('Completed segmentation checkpoint changed')
    target=safe_new_root(output); images=target/'inputs'; images.mkdir()
    for case in native['split']['outer_val']:
        shutil.copy2(Path(native['raw'])/'imagesTr'/f'{case}_0000.nii.gz',images/f'{case}_0000.nii.gz')
    env=environment(root,root/'runtime')
    env.update(nnUNet_raw=str(Path(native['raw']).parent),nnUNet_preprocessed=str(Path(native['preprocessed']).parent))
    run([executable('nnUNetv2_predict'),'-i',str(images),'-o',str(target/'predictions'),'-d',str(native['dataset_id']),
         '-c','3d_fullres','-f','0','-tr',TRAINER,'-p',PLANS,'-chk','checkpoint_final.pth'],env,target/'predict.log')
    outputs={case:sha(target/'predictions'/f'{case}.nii.gz') for case in native['split']['outer_val']}
    write_new(target/'prediction_complete.json',{'format':PIPELINE_VERSION,'checkpoint_sha256':receipt['sha256'],
               'native_sha256':sha(native_path),'predictions':outputs,
               'inputs':{p.name:sha(p) for p in images.glob('*.nii.gz')},
               'provenance_scope':'command/checkpoint/input/output files; loaded tensor identity not independently instrumented'})
    return target/'predictions'
