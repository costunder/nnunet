"""Prepare exact six-arm bindings and one shared131-case admission; no training.

Input snapshot is the captured old G2 host_inputs JSON. Bindings name already
transferred checkpoint/effective-bank/preflight/log artifacts by exact SHA.
This tool copies only the small native snapshot metadata and writes receipts.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hiercp_v1x.basic_cp_single_parent_resources import ARMS,EPOCHS,digest,pin,require,signature
from tools.run_basic_cp_single_parent import seal,FORMAT


def inventory(preprocessed,cases,data_identifier):
    data=preprocessed/data_identifier;files={};groups={}
    for case in sorted(cases):
        candidates=[(data/(case+'.b2nd'),data/(case+'_seg.b2nd'),data/(case+'.pkl')),
            (data/(case+'.npz'),data/(case+'.pkl')),
            (data/(case+'.npy'),data/(case+'_seg.npy'),data/(case+'.pkl'))]
        valid=[row for row in candidates if all(p.is_file()for p in row)]
        require(bool(valid),'Actual full preprocessed data/segmentation/properties missing: '+case)
        ground_truth=preprocessed/'gt_segmentations'/(case+'.nii.gz')
        require(ground_truth.is_file(),'Original whole-case validation ground truth missing: '+case)
        # Multiple equivalent stored formats are recorded, not silently dropped.
        paths={str(p):p for row in valid for p in row};paths[str(ground_truth)]=ground_truth;groups[case]=sorted(paths)
        files.update(paths)
    cpus=len(psutil.Process().cpu_affinity());workers=min(16,cpus)
    def record(path):return str(path),dict(sha256=digest(path),stat=list(signature(path)))
    with ThreadPoolExecutor(max_workers=workers)as pool:checks=dict(pool.map(record,files.values()))
    return dict(format='basic_cp_shared_preprocessed_actual131_v1',complete=True,case_count=len(cases),
        all131_data_SHA_verified=len(cases)==131,cases=groups,files=checks,
        source_arrays_written=False,raw_CT_preprocessing_executed=False,hash_workers=workers,observed_unix=time.time())


def completed_log_prefix(reference,epochs,target):
    """Keep raw evidence and admit only its unique fully completed epoch prefix."""
    from hiercp_v1x.basic_cp_verified_continuation import _completed_prefix
    source=pin(reference);before=signature(source)
    prefix,record=_completed_prefix(source.read_bytes(),epochs)
    require(signature(source)==before,'Original log changed while admitting completed prefix')
    with target.open('xb')as stream:stream.write(prefix);stream.flush();os.fsync(stream.fileno())
    return dict(path=str(target),sha256=digest(target)),dict(record,
        original_raw_log=reference,completed_prefix_bytes=len(prefix),
        original_raw_bytes=source.stat().st_size,raw_log_modified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('source-snapshot','bindings','allocation'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--shared-preprocessed-dataset-directory',type=Path,required=True)
    parser.add_argument('--evaluator-root',type=Path,required=True);parser.add_argument('--project',type=Path,required=True)
    args=parser.parse_args();require(os.getuid()==41833,'Own experiment account required')
    refs={name:dict(path=str(getattr(args,name)),sha256=getattr(args,name+'_sha256'))
        for name in('source_snapshot','bindings','allocation')}
    snapshot=json.loads(pin(refs['source_snapshot']).read_bytes());bindings=json.loads(pin(refs['bindings'],owner=True).read_bytes())
    allocation=json.loads(pin(refs['allocation'],owner=True).read_bytes())
    require(set(bindings)==set(ARMS),'Exactly six explicit transferred artifact bindings required')
    rows={row['arm']:row for row in snapshot['host_inputs']}
    require(set(rows)==set(ARMS)|{'blend50'},'Original all-seven captured metadata required; blend50 stays complete')
    root=args.root;require(root.is_absolute()and root.is_relative_to(Path('/dev/shm'))and not(root/'cohort.json').exists(),
        'Fresh own tmpfs cohort receipt required')
    root.mkdir(mode=0o700,parents=True,exist_ok=True)
    original_native=root/'original_native_snapshots';original_native.mkdir(mode=0o700)
    prefixes=root/'completed_log_prefixes';prefixes.mkdir(mode=0o700)
    pre=args.shared_preprocessed_dataset_directory
    native=rows['multi5']['native'];cases=set(native['split']['outer_train'])|set(native['split']['outer_val'])
    require(len(cases)==131,'All105train and26validation patients required')
    proof=inventory(pre,cases,native['baseline']['data_identifier'])
    inventory_path=root/'shared_preprocessed_actual131.json';seal(inventory_path,proof)
    installed={}
    for name in('nnunetv2.training.nnUNetTrainer.nnUNetTrainer','nnunetv2.training.dataloading.nnunet_dataset',
                'nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCP'):
        path=Path(importlib.util.find_spec(name).origin);installed[str(path)]=digest(path)
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True,timeout=10).strip()
    names=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z','--','custom_trainers','hiercp_v1x','tools','config'])
    sources={name.decode():digest(ROOT/name.decode())for name in names.split(b'\0')if name
        and(Path(name.decode()).suffix=='.py'or name.decode()=='config/nnunet_result_report_schema_v1.json')}
    experiments={}
    import torch
    from hiercp_v1x.basic_cp_verified_continuation import _state
    for arm in ARMS:
        old=rows[arm]['native'];path=original_native/(arm+'.json');seal(path,old)
        row=dict(bindings[arm])
        for name in('effective_bank','preflight','LATEST','BEST','log_prefix','calibration'):pin(row[name])
        raw_log=row['log_prefix']
        prefix,prefix_proof=completed_log_prefix(raw_log,EPOCHS[arm],prefixes/(arm+'.log'))
        row.update(log_prefix=prefix,raw_original_log=raw_log,completed_log_prefix_proof=prefix_proof)
        loaded=torch.load(pin(row['LATEST']),map_location='cpu',weights_only=False)
        require(loaded['current_epoch']==EPOCHS[arm],'Actual latest completed epoch differs: '+arm)
        resume=_state(loaded,EPOCHS[arm],old['trainer']);del loaded
        loaded=torch.load(pin(row['BEST']),map_location='cpu',weights_only=False)
        require(type(loaded['current_epoch'])is int and 0<loaded['current_epoch']<=EPOCHS[arm],
            'Own BEST epoch lies outside the completed original prefix')
        best_state=_state(loaded,loaded['current_epoch'],old['trainer']);del loaded
        expected=128 if arm.startswith('multi')else int(arm.removeprefix('candidates'))
        effective=json.loads(pin(row['effective_bank']).read_bytes());preflight=json.loads(pin(row['preflight']).read_bytes())
        require(effective['source_entries']==642 and effective['candidate_count']==expected
            and preflight['index_sha256']==row['effective_bank']['sha256']and preflight['arm']==arm
            and len(preflight['entries_sha256'])==642,'Exact effective full642 bank/preflight required')
        experiments[arm]=dict(row,logical_slot=int(rows[arm]['slot']),source_native=dict(path=str(path),sha256=digest(path)),
            completed_epochs=EPOCHS[arm],physical_batch=2,epochs=250,workers=4,
            output_root=str(root/arm),
            shared_preprocessed_dataset_directory=str(pre),shared_preprocessed_inventory=dict(path=str(inventory_path),sha256=digest(inventory_path)),
            evaluator_root=str(args.evaluator_root),project=str(args.project),
            original_effective_source_bank=(old['bank']if expected==128 else str(Path(old['shared_banks'])/str(expected)/'index.json')),
            actual_resume_checkpoint_proof=resume,actual_original_BEST_proof=best_state)
    cohort=dict(format=FORMAT,root=str(root),canonical_code=dict(path=str(ROOT),commit=commit,files_sha256=sources),
        allocation=refs['allocation'],physical_parent_index=allocation['physical_parent_index'],
        installed_runtime_files_sha256=installed,experiments=experiments,retained_complete_arms=['blend50'],debug=False,
        original_snapshot=refs['source_snapshot'],input_bindings=refs['bindings'],
        source_arrays_copied=False,preprocessing_executed=False,private_code_copies_created=False)
    path=root/'cohort.json';seal(path,cohort)
    print(json.dumps(dict(cohort=str(path),sha256=digest(path),canonical_commit=commit,
        unfinished_arms=list(ARMS),completed_blend50_retained=True,production_launched=False)),flush=True)


if __name__=='__main__':main()
