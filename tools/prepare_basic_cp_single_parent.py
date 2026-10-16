"""Prepare exact unfinished bindings and one shared131-case admission; no training.

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
from hiercp_v1x.basic_cp_single_parent_resources import ARMS,EPOCHS,digest,pin,require,signature,execution_arms
from tools.run_basic_cp_single_parent import seal,FORMAT,bound_completed_epochs,continuation_arms


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


def checkpoint_resume_state(saved,*,epoch,trainer,latest):
    """Validate an exact whole checkpoint; model/optimizer/scaler proof stays native."""
    from hiercp_v1x.basic_cp_verified_continuation import _state
    actual=saved.get('current_epoch')
    require(type(actual)is int and((actual==epoch)if latest else(0<actual<=epoch)),
        'Actual latest completed epoch differs from bound resume epoch'if latest
        else'Own BEST epoch lies outside the bound completed prefix')
    return _state(saved,epoch if latest else actual,trainer)


def historical_reference_inputs(evaluator_root,project,expected_cases):
    """Read-only availability, byte and NIfTI-header proof, not an evaluation."""
    from tools.evaluate_v24_historical_metrics import load_evaluator,EXPECTED
    evaluator=load_evaluator(evaluator_root)
    require(evaluator.nib is not None,'Native historical NIfTI dependency is required before production')
    project=project.resolve(strict=True);online=project/'work'/evaluator.DEFAULT_ONLINE_ROOT
    nnroot=online/'nnunetv2'
    paths=evaluator.Paths(project=project,medical=project.parent,data=project.parent/'Data',
        paired=project/'work'/evaluator.DEFAULT_PAIRED_ROOT,online=online,nnroot=nnroot,
        results=nnroot/'nnUNet_results',outer_splits=project/'work'/evaluator.DEFAULT_PAIRED_ROOT/'outer_splits.json',
        nnunet_config=project/'config/nnunet.json')
    split=evaluator.outer_split(paths.outer_splits,0)
    require(len(expected_cases)==26 and set(split['val'])==set(expected_cases),
        'Historical reference must use the same complete outer26 validation patients')
    config=evaluator.load_json(paths.nnunet_config)
    basic=evaluator.model_validation_dir(paths,730,0,config,evaluator.DEFAULT_BASIC_TRAINER)
    evaluator.verify_prediction_inventory(basic,expected_cases)
    files={};headers={}
    def record(path,**values):
        before=signature(path);checksum=evaluator.file_sha256(path)
        require(signature(path)==before,'Historical reference changed during read-only readiness check: '+str(path))
        files[str(path)]=dict(sha256=checksum,stat=list(before),**values)
    for path in(paths.nnunet_config,paths.outer_splits):record(path,role='configuration')
    for case in expected_cases:
        ground=paths.data/'labels'/(case+'.nii.gz');prediction=basic/(case+'.nii.gz')
        record(ground,role='ground_truth',case_id=case);record(prediction,role='historical_basic_prediction',case_id=case)
        reference=evaluator.nib.load(str(ground));candidate=evaluator.nib.load(str(prediction))
        evaluator.verify_geometry(reference,candidate,prediction)
        headers[case]=dict(shape=list(reference.shape),affine=reference.affine.tolist())
    return dict(format='basic_cp_historical_reference_actual26_readiness_v1',complete=True,
        validation_cases=list(expected_cases),ground_truth_files=26,basic_prediction_files=26,
        NIfTI_header_geometry_verified=True,files=files,headers=headers,evaluator_root=str(evaluator_root),
        evaluator_sha256=EXPECTED,full_evaluation_executed=False,metrics_computed=False,inputs_modified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('source-snapshot','bindings','allocation'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--shared-preprocessed-dataset-directory',type=Path,required=True)
    parser.add_argument('--evaluator-root',type=Path,required=True);parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--continuation-inputs',type=Path);parser.add_argument('--continuation-inputs-sha256')
    args=parser.parse_args();require(os.getuid()==41833,'Own experiment account required')
    refs={name:dict(path=str(getattr(args,name)),sha256=getattr(args,name+'_sha256'))
        for name in('source_snapshot','bindings','allocation')}
    snapshot=json.loads(pin(refs['source_snapshot']).read_bytes());bindings=json.loads(pin(refs['bindings'],owner=True).read_bytes())
    allocation=json.loads(pin(refs['allocation'],owner=True).read_bytes())
    arms=execution_arms(allocation)
    continuation=dict(retained_complete_arms=['blend50'],debug=False)
    if args.continuation_inputs:
        continuation=json.loads(pin(dict(path=str(args.continuation_inputs),sha256=args.continuation_inputs_sha256),owner=True).read_bytes())
        require(continuation.get('bindings')==refs['bindings']and continuation.get('source_snapshot')==refs['source_snapshot'],
            'Explicit continuation inputs must bind the identical scientific artifacts')
    require(continuation_arms(continuation)==arms,'Allocation and explicit unfinished experiment list differ')
    require(set(bindings)==set(arms),'Exact explicit unfinished artifact bindings required')
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
    require(args.evaluator_root==ROOT,'The historical evaluator must share the one canonical Git checkout')
    historical=historical_reference_inputs(args.evaluator_root,args.project,native['split']['outer_val'])
    historical_path=root/'historical_reference_actual26_readiness.json';seal(historical_path,historical)
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
    for arm in arms:
        old=rows[arm]['native'];path=original_native/(arm+'.json');seal(path,old)
        row=dict(bindings[arm])
        epoch=bound_completed_epochs(arm,row)
        for name in('effective_bank','preflight','LATEST','BEST','log_prefix','calibration'):pin(row[name])
        raw_log=row['log_prefix']
        prefix,prefix_proof=completed_log_prefix(raw_log,epoch,prefixes/(arm+'.log'))
        row.update(log_prefix=prefix,raw_original_log=raw_log,completed_log_prefix_proof=prefix_proof)
        loaded=torch.load(pin(row['LATEST']),map_location='cpu',weights_only=False)
        resume=checkpoint_resume_state(loaded,epoch=epoch,trainer=old['trainer'],latest=True);del loaded
        loaded=torch.load(pin(row['BEST']),map_location='cpu',weights_only=False)
        best_state=checkpoint_resume_state(loaded,epoch=epoch,trainer=old['trainer'],latest=False);del loaded
        expected=128 if arm.startswith('multi')else int(arm.removeprefix('candidates'))
        effective=json.loads(pin(row['effective_bank']).read_bytes());preflight=json.loads(pin(row['preflight']).read_bytes())
        require(effective['source_entries']==642 and effective['candidate_count']==expected
            and preflight['index_sha256']==row['effective_bank']['sha256']and preflight['arm']==arm
            and len(preflight['entries_sha256'])==642,'Exact effective full642 bank/preflight required')
        experiments[arm]=dict(row,logical_slot=int(rows[arm]['slot']),source_native=dict(path=str(path),sha256=digest(path)),
            completed_epochs=epoch,physical_batch=2,epochs=250,workers=4,
            output_root=str(root/arm),
            shared_preprocessed_dataset_directory=str(pre),shared_preprocessed_inventory=dict(path=str(inventory_path),sha256=digest(inventory_path)),
            evaluator_root=str(args.evaluator_root),project=str(args.project),
            original_effective_source_bank=(old['bank']if expected==128 else str(Path(old['shared_banks'])/str(expected)/'index.json')),
            actual_resume_checkpoint_proof=resume,actual_original_BEST_proof=best_state)
    cohort=dict(format=FORMAT,root=str(root),canonical_code=dict(path=str(ROOT),commit=commit,files_sha256=sources),
        allocation=refs['allocation'],physical_parent_index=allocation['physical_parent_index'],
        installed_runtime_files_sha256=installed,experiments=experiments,active_arms=list(arms),
        retained_complete_arms=continuation['retained_complete_arms'],
        retained_completion_evidence=continuation.get('retained_completion_evidence',{}),debug=False,
        historical_reference_inputs=dict(path=str(historical_path),sha256=digest(historical_path)),
        original_snapshot=refs['source_snapshot'],input_bindings=refs['bindings'],
        source_arrays_copied=False,preprocessing_executed=False,private_code_copies_created=False)
    path=root/'cohort.json';seal(path,cohort)
    print(json.dumps(dict(cohort=str(path),sha256=digest(path),canonical_commit=commit,
        unfinished_arms=list(arms),retained_complete_arms=continuation['retained_complete_arms'],production_launched=False)),flush=True)


if __name__=='__main__':main()
