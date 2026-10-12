"""Automatic own v2.5 complete CP -> native250 -> BEST26 -> Result schema1.

Every completed stage is admitted through saved bytes. Interrupted native
training resumes its own latest; metric failures use a fresh output. No old
experiment, original Basic bank or frozen scientific module is rewritten.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from . import v25_cp as cp
from . import v25_native_cp as native
from . import v25_native_storage as storage

FORMAT='v25_full_own_arm_downstream_request_v1'
STATUS_FORMAT='v25_full_own_arm_downstream_status_v1'
COMPLETE_FORMAT='v25_full_own_arm_downstream_completion_v1'
ROOT=Path(__file__).resolve().parents[1]
REPORTER='tools/render_nnunet_result_schema_v1.py'
EVALUATOR='tools/evaluate_v24_historical_metrics.py'
HISTORICAL_EVALUATOR={
    'online_eval_v2.py':'93c703edb20e074645ef443f6ca2bd7db03268bbb90fda8591f337993592270b',
    'online_eval_provenance.py':'f624ffe37adc708ba5495b1b6bb9c59df74b841109442994379acd175ba75d91'}


def _regular(path,*,directory=False):
    from .v24_readonly_native_storage import regular
    return regular(path,directory=directory)


def _actor():
    import psutil
    actor=psutil.Process()
    return dict(pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),
        UID=os.getuid() if hasattr(os,'getuid') else None)


def _status(root,request,stage,state,*,error=None,proof=None):
    value=dict(format=STATUS_FORMAT,arm=request['arm'],stage=stage,state=state,
        updated_at=time.time(),actor=_actor(),error=error,request_sha256=cp._sha(root/'request.json'),
        full_native_training_completed=False,full_native_evaluation_completed=False,
        full_lesion_metric_evaluation_completed=False,result_schema1_published=False,
        stage_log=str(root/'controller.log'))
    if proof is not None:
        for key in ('full_native_training_completed','full_native_evaluation_completed',
            'full_lesion_metric_evaluation_completed','result_schema1_published'):
            value[key]=proof.get(key) is True
        if proof.get('native_checkpoint') is not None:value['native_checkpoint']=proof['native_checkpoint']
    child=root/'native_child.json'
    if child.exists():
        child_record=cp._read(child)
        if child_record.get('request_sha256')==value['request_sha256']:
            value['child_actor']=child_record
            value['stage_log']=child_record['log_path']
    certificate=root/'completion.json'
    if state=='COMPLETE' and certificate.exists():
        value['completion_evidence']=dict(path=str(certificate),sha256=cp._sha(certificate))
    if native.ENV_EXECUTION in os.environ:
        value['execution_resource_evidence']=dict(path=os.environ[native.ENV_EXECUTION],
            sha256=os.environ[native.ENV_EXECUTION_SHA])
    temporary=root/('.status.'+str(time.time_ns())+'.tmp')
    try:
        with temporary.open('x',encoding='utf8') as stream:
            json.dump(value,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,root/'status.json')
    finally:
        if temporary.exists():temporary.unlink()
    with (root/'controller.log').open('a',encoding='utf8') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)


def _request(*,arm,gpu,output,score_overlay,storage_admission,project,basic_validation,
             cpu_affinity,allocation_request,evaluator_root):
    if arm not in storage.ARMS or type(gpu)is not int or gpu<0:
        raise ValueError('One explicit logical v2.5 arm and physical GPU required')
    if allocation_request is None:raise ValueError('Actual current own allocation request required before downstream')
    admission_path=_regular(storage_admission);admission=storage.verify_admission(admission_path)
    root=Path(output).absolute()
    if str(root)!=admission['arm_roots'][arm]:raise ValueError('Downstream output must be exact reserved four-arm root')
    _regular(root,directory=True)
    overlay_path=_regular(score_overlay);overlay=cp.validate_overlay(overlay_path,full_hash=True)['overlay']
    if overlay['best_provenance']['arm']!=arm:raise ValueError('CP scores must come from this completed own-arm GNN BEST')
    if overlay['split']!=admission['split'] or overlay['baseline']!=admission['baseline']:
        raise ValueError('CP and native original full105/26 preprocessing/split must match')
    allocation_path=_regular(allocation_request);allocation=cp._read(allocation_path)
    if (allocation.get('format')!='v25_owned_GPU_allocation_v1' or allocation.get('gpu')!=gpu
            or allocation.get('CPU_affinity')!=list(cpu_affinity)
            or allocation.get('UID')!=41833 or allocation.get('owner')!='aicompetition06'
            or not isinstance(allocation.get('GPU_UUID'),str)
            or type(allocation.get('expires_at_unix'))not in (int,float)):
        raise ValueError('Current exact owned GPU/CPU/UID/PBS proof required')
    if len(set(cpu_affinity))!=len(cpu_affinity) or len(cpu_affinity)<4:
        raise ValueError('At least measured original four DA workers plus exact allocated CPU set required')
    project_path=_regular(project,directory=True);basic=_regular(basic_validation,directory=True)
    if evaluator_root is None:raise ValueError('Explicit frozen historical v4 evaluator directory required')
    evaluator=_regular(evaluator_root,directory=True)
    for name,digest in HISTORICAL_EVALUATOR.items():
        if cp._sha(evaluator/'tools'/name)!=digest:raise ValueError('Exact previously completed historical v4 evaluator required')
    cases=admission['split']['outer_val']
    expected={case+'.nii.gz' for case in cases}
    if {path.name for path in basic.glob('*.nii.gz')}!=expected:
        raise ValueError('Historical Basic predictions must contain exactly the full26 outer-validation cases')
    files=(*native.FILES,EVALUATOR,REPORTER,'config/nnunet_result_report_schema_v1.json')
    return dict(format=FORMAT,arm=arm,gpu=gpu,output=str(root),score_overlay=str(overlay_path),
        score_overlay_sha256=cp._sha(overlay_path),storage_admission=str(admission_path),
        storage_admission_sha256=cp._sha(admission_path),project=str(project_path),basic_validation=str(basic),
        basic_predictions_sha256={name:cp._sha(basic/name) for name in sorted(expected)},
        evaluator_root=str(evaluator),evaluator_source_sha256=copy.deepcopy(HISTORICAL_EVALUATOR),
        CPU_affinity=list(cpu_affinity),allocation_request=str(allocation_path),
        allocation_request_sha256=cp._sha(allocation_path),process_and_children_RSS_bytes=96*2**30,
        shared_cache_memory_reserve_bytes=32*2**30,
        sources_sha256={name:cp._sha(ROOT/name) for name in files},
        debug=False,train_cases=105,validation_cases=26,source_entries=642,candidates_per_source=128,
        native_epochs=250,native_physical_batch=2,native_patch_size=[128]*3,cp_probability=.5,
        evaluation_checkpoint='checkpoint_best.pth',result_schema_version=1,
        model_data_scale_preserved=True)


def _guard_request(request,path):
    if cp._read(path)!=request:raise ValueError('Own immutable downstream request changed')
    for field in ('score_overlay','storage_admission','allocation_request'):
        if cp._sha(request[field])!=request[field+'_sha256']:
            raise ValueError('Own admitted artifact changed: '+field)
    for name,digest in request['sources_sha256'].items():
        if cp._sha(ROOT/name)!=digest:raise ValueError('Frozen v2.5 downstream implementation changed: '+name)
    for name,digest in request['basic_predictions_sha256'].items():
        if cp._sha(Path(request['basic_validation'])/name)!=digest:
            raise ValueError('Original full26 historical Basic prediction changed')
    for name,digest in request['evaluator_source_sha256'].items():
        if cp._sha(Path(request['evaluator_root'])/'tools'/name)!=digest:raise ValueError('Frozen historical v4 metric implementation changed')


def _bind_execution(scientific,candidate,path):
    """Renew resource admission without changing one scientific request byte.

    The existing actual physical device remains the calibration authority.
    Different-device execution needs an additional full B2/B4 admission; it
    cannot silently inherit the original physical-device measurements.
    """
    fields=set(native.RESOURCE_FIELDS)
    if ({key:value for key,value in scientific.items() if key not in fields}
            !={key:value for key,value in candidate.items() if key not in fields}):
        raise ValueError('Resource renewal changed the immutable scientific downstream contract')
    before=cp._read(scientific['allocation_request']);after=cp._read(candidate['allocation_request'])
    if before['gpu']!=after['gpu'] or before['GPU_UUID']!=after['GPU_UUID']:
        raise ValueError('Native renewal requires identical calibrated physical GPU index/UUID')
    execution=dict(format='v25_full_own_arm_downstream_execution_v1',arm=scientific['arm'],gpu=scientific['gpu'],
        scientific_request=str(path),scientific_request_sha256=cp._sha(path),
        resource_binding={key:candidate[key] for key in native.RESOURCE_FIELDS},
        same_scientific_model_data_optimizer_CP_and_BEST=True,
        retained_original_actual_B2_B4_physical_device=True)
    target=path.parent/('execution_request_'+cp.canonical_hash(execution)+'.json')
    if target.exists():
        if cp._read(target)!=execution:raise ValueError('Own resource renewal receipt changed')
    else:cp._publish(target,execution)
    os.environ.update({native.ENV_EXECUTION:str(target),native.ENV_EXECUTION_SHA:cp._sha(target)})
    return target


def _validate_metrics(folder,expected_cases,*,request=None,prediction=None):
    """The renderer validates every published artifact, not just summary.json."""
    from tools import render_nnunet_result_schema_v1 as renderer
    schema=cp._read(ROOT/'config/nnunet_result_report_schema_v1.json')
    bundle=renderer._online(dict(directory=str(folder),experiment='v2.5'),schema)
    if bundle is None or bundle['status']!='VERIFIED_COMPLETE':raise ValueError('Actual complete lesion metric certificate required')
    if bundle['summary'].get('validation_cases')!=len(expected_cases):
        raise ValueError('Lesion evaluation must retain all26 outer-validation cases')
    if request is not None and bundle['summary'].get('basic_validation')!=request['basic_validation']:
        raise ValueError('Frozen historical evaluator must use exact admitted original Basic26 directory')
    if prediction is not None and bundle['summary'].get('hier_validation')!=prediction:
        raise ValueError('Frozen evaluator must use actual own BEST26 prediction directory')
    six=cp._read(Path(folder)/'six_metric_comparison.json')
    if (six.get('evaluator_sha256')!=HISTORICAL_EVALUATOR
            or six.get('historical_basic_reproduced')is not True or six.get('patient_count')!=26
            or six.get('checkpoint')!='BEST' or six.get('wrapper_sha256')!=cp._sha(ROOT/EVALUATOR)):
        raise ValueError('Actual historical Basic reproduction and own BEST metric wrapper proof required')
    return bundle


def _metrics(request,completion):
    root=Path(request['output'])/'downstream';adoption=root/'metrics_verified.json'
    if adoption.exists():
        record=cp._read(adoption)
        if (record.get('request_sha256')!=cp._sha(root/'request.json')
                or cp._sha(Path(record['directory'])/'completion.json')!=record.get('completion_sha256')
                or cp._sha(Path(record['directory'])/'six_metric_comparison.json')!=record.get('six_metric_comparison_sha256')):
            raise ValueError('Own adopted metric output changed')
        _validate_metrics(record['directory'],completion['native']['split']['outer_val'],request=request,prediction=completion['prediction'])
        return Path(record['directory'])
    # An incomplete metric directory is immutable evidence of the prior error.
    # A new attempt never changes it; actual predictions are safely reused.
    folder=root/('metrics_'+str(time.time_ns()))
    command=[sys.executable,'-B','-u',str(ROOT/EVALUATOR),'--project',request['project'],
        '--evaluator-root',request['evaluator_root'],'--prediction',completion['prediction'],
        '--checkpoint','BEST','--trainer',completion['native']['trainer'],'--output',str(folder),
        '--model-label','v2.5 GNN CP -> nnUNet','--gpu-label','arm'+request['arm']+'/GPU'+str(request['gpu'])]
    native.private_modules()['NATIVE_PIPELINE']._child(command,os.environ.copy(),root/('metric_'+str(time.time_ns())+'.log'))
    _validate_metrics(folder,completion['native']['split']['outer_val'],request=request,prediction=completion['prediction'])
    cp._publish(adoption,dict(format='v25_own_BEST26_lesion_metrics_adoption_v1',
        request_sha256=cp._sha(root/'request.json'),directory=str(folder),
        completion_sha256=cp._sha(folder/'completion.json'),
        six_metric_comparison_sha256=cp._sha(folder/'six_metric_comparison.json'),actual_data_evaluation=True))
    return folder


def _training_record(completion):
    import torch
    saved=torch.load(completion['final_checkpoint']['path'],map_location='cpu',weights_only=False)
    logging=saved.get('logging',{})
    if not isinstance(logging,dict):raise ValueError('Actual native checkpoint logger dictionary required')
    result=dict(planned_epochs=250,completed_epochs=250,
        best_current_epoch_raw=completion['best_receipt'].get('best_current_epoch'),
        best_selection_rule='original nnU-Net best EMA foreground Dice',
        best_epoch_index_definition='raw current_epoch saved by original nnU-Net; not relabelled',
        measurement_scope='actual full250 final checkpoint logger and original OnlineCP console')
    for source,target in (('train_losses','train_loss'),('val_losses','val_loss'),
            ('dice_per_class_or_region','online_pseudo_dice')):
        values=logging.get(source,[])
        if values:
            last=values[-1]
            if hasattr(last,'tolist'):last=last.tolist()
            elif hasattr(last,'item'):last=last.item()
            result[target]=last
            result['latest_loss_epoch']=len(values)-1
    starts,ends=logging.get('epoch_start_timestamps',[]),logging.get('epoch_end_timestamps',[])
    if len(starts)==250 and len(ends)==250:
        result['training_seconds']=sum(float(end)-float(start) for start,end in zip(starts,ends))
        result.update(started_at=starts[0],ended_at=ends[-1],
            epoch_time=dict(measured_epochs=250,source='original saved epoch timestamps'))
    counts=[];root=Path(completion['native']['root'])
    for path in sorted(root.glob('train_*.log')):
        counts.extend(tuple(map(int,row)) for row in re.findall(r'\[OnlineCP\] epoch=(\d+) applied=(\d+)/(\d+)',path.read_text(encoding='utf8')))
    unique={}
    for epoch,applied,attempted in counts:
        if epoch in unique and unique[epoch]!=(applied,attempted):
            raise ValueError('Native log contains conflicting CP epoch records')
        unique[epoch]=(applied,attempted)
    if len(unique)==250:
        applied=sum(row[0] for row in unique.values());attempted=sum(row[1] for row in unique.values())
        result.update(cp_applied=applied,cp_attempted=attempted,cp_apply_rate=applied/attempted)
    return result


def _summary_evidence(path):
    """Represent original undefined native cells as null only in report copies."""
    path=Path(path);raw=cp._read(path);notes=[]
    def copied(value,location='$'):
        if isinstance(value,float) and not math.isfinite(value):
            notes.append(dict(path=location,original_value='NaN' if math.isnan(value) else str(value),
                reason='Original native summary nonfinite/undefined cell; only report evidence copy uses JSON null; raw bytes/SHA preserved'))
            return None
        if isinstance(value,dict):return {key:copied(item,location+'.'+str(key)) for key,item in value.items()}
        if isinstance(value,list):return [copied(item,location+'['+str(index)+']') for index,item in enumerate(value)]
        return value
    return dict(summary=copied(raw),summary_source=str(path),summary_sha256=cp._sha(path),
        summary_nonfinite_copy_normalization=notes)


def collect_evidence(request,completion,metrics):
    overlay=cp._read(request['score_overlay']);name='v2.5 '+request['arm']+' '+Path(request['output']).parent.name
    nat=completion['native'];native_summary=_summary_evidence(Path(completion['prediction'])/'summary.json')
    pre=Path(nat['root'])/'nnUNet_preprocessed'/nat['baseline']['dataset_name']
    dataset=cp._read(pre/'dataset.json')
    native_info=dict(status='VERIFIED_COMPLETE',receipt=completion['best_receipt'],**native_summary,
        dataset_labels=dataset['labels'],verification=dict(best_receipt_verified=True,
        source=str(Path(completion['prediction']).parent/'best_validation_complete.json')),
        training_receipt=completion['final_checkpoint'])
    basic=dict(checkpoint_origin_status='UNVERIFIED / original historical Basic BEST linkage',
        native=dict(dataset_labels=dataset['labels']))
    basic_summary=Path(request['basic_validation'])/'summary.json'
    if basic_summary.is_file():basic['native'].update(_summary_evidence(basic_summary))
    return dict(kind='ACTUAL_COMPLETED_FULL_V25_EVIDENCE',experiments=[dict(experiment=name,
        completion=dict(training_status='VERIFIED_COMPLETE',completed_epochs=250,planned_epochs=250,
            best_native_evaluation_status='VERIFIED_COMPLETE',predicted_cases=26,expected_cases=26),
        settings=dict(experiment=dict(version='v2.5',arm=request['arm'],physical_GPU=request['gpu']),
            gnn=dict(epochs=40,optimizer_updates=680,parameters=13331792,physical_patient_batch=4,
                context=overlay['context_contract']),gnn_best=overlay['best_provenance'],
            cp=dict(overlay['cp_contract'],complete_sources=642,complete_positions=82176,
                original_source_and_positions_preserved=True),
            nnunet=dict(model='nnUNetResEncUNetM',epochs=250,physical_batch=2,patch=[128]*3,
                train_cases=105,validation_cases=26,evaluation_checkpoint='checkpoint_best.pth')),
        training=_training_record(completion),native=native_info)],basic=basic,
        online_evaluation=dict(experiment=name,directory=str(metrics)),
        comparison=dict(status='HISTORICAL_REFERENCE',source=str(metrics/'completion.json'),
            reason='Historical Basic prediction bytes/cohort are verified; original Basic training-BEST linkage is unverified'),
        verification=dict(full_training=True,full_best_native_evaluation=True,lesion_metrics=True,
            static_checks='implementation source SHA sealed',
            unit_smoke_checks='Actual B2/B4 native full-model clone calibration saved separately'),
        limitations=['Historical Basic original BEST linkage remains unverified; no matched-comparison claim',
            'Unrecorded scalar telemetry is shown as NA; metric definitions and fixed Result schema1 are unchanged'],
        artifacts=[dict(path=request[field],sha256=request[field+'_sha256'])
            for field in ('score_overlay','storage_admission','allocation_request')])


def _report(request,completion,metrics):
    from tools.render_nnunet_result_schema_v1 import write_report,render_report
    root=Path(request['output'])/'downstream';evidence=root/'result_evidence.json';stem=root/'result_schema1'
    value=collect_evidence(request,completion,metrics)
    if evidence.exists():
        if cp._read(evidence)!=value:raise ValueError('Existing collected evidence differs; preserved')
    else:cp._publish(evidence,value)
    text,document=root/'result_schema1.txt',root/'result_schema1.json'
    if text.exists() or document.exists():
        expected_json,expected_text=render_report(value)
        if (text.exists() and text.read_text(encoding='utf8')!=expected_text
                or document.exists() and cp._read(document)!=expected_json):
            raise ValueError('Existing schema1 report differs from verified actual completed evidence')
        # Complete a torn publication only with the very same renderer output.
        for path,contents in ((document,json.dumps(expected_json,ensure_ascii=False,indent=2,allow_nan=False)+'\n'),
                (text,expected_text)):
            if not path.exists():
                with path.open('x',encoding='utf8',newline='\n') as stream:
                    stream.write(contents);stream.flush();os.fsync(stream.fileno())
    else:text,document=write_report(value,stem)
    return dict(evidence=dict(path=str(evidence),sha256=cp._sha(evidence)),
        text=dict(path=str(text),sha256=cp._sha(text)),json=dict(path=str(document),sha256=cp._sha(document)))


def validate_completion(root):
    root=Path(root);proof=cp._read(root/'completion.json');request=cp._read(root/'request.json')
    _guard_request(request,root/'request.json')
    os.environ.pop(native.ENV_EXECUTION,None);os.environ.pop(native.ENV_EXECUTION_SHA,None)
    os.environ.update({native.ENV_REQUEST:str(root/'request.json'),native.ENV_SHA:cp._sha(root/'request.json')})
    if (proof.get('format')!=COMPLETE_FORMAT or proof.get('request_sha256')!=cp._sha(root/'request.json')
            or any(proof.get(key)is not True for key in ('complete','actual_data_evaluation',
                'full_native_training_completed','full_native_evaluation_completed',
                'full_lesion_metric_evaluation_completed','result_schema1_published'))):
        raise ValueError('Actual own complete250/BEST26/metrics/schema1 certificate required')
    completion=native.native_completion(Path(request['output'])/'native/native.json')
    if (completion['final_checkpoint']!=proof.get('native_final_checkpoint')
            or cp._sha(Path(completion['prediction']).parent/'best_validation_complete.json')!=proof.get('native_best_receipt_sha256')
            or cp._sha(Path(proof['metrics_directory'])/'completion.json')!=proof.get('metrics_completion_sha256')):
        raise ValueError('Completed native/metric certificate bytes changed')
    _validate_metrics(proof['metrics_directory'],completion['native']['split']['outer_val'],request=request,prediction=completion['prediction'])
    for witness in proof['reports'].values():
        if cp._sha(witness['path'])!=witness['sha256']:raise ValueError('Completed schema1 report bytes changed')
    return proof


def verify_completion(path):
    path=Path(path)
    if path.name!='completion.json':raise ValueError('Exact own downstream completion certificate required')
    return validate_completion(path.parent)


def run_downstream(*,arm,gpu,output,score_overlay,storage_admission,project,basic_validation,
                   cpu_affinity,allocation_request=None,evaluator_root=None):
    """Run every full downstream stage, then publish the sole final certificate."""
    os.environ.pop(native.ENV_EXECUTION,None);os.environ.pop(native.ENV_EXECUTION_SHA,None)
    request=_request(arm=arm,gpu=gpu,output=output,score_overlay=score_overlay,
        storage_admission=storage_admission,project=project,basic_validation=basic_validation,
        cpu_affinity=cpu_affinity,allocation_request=allocation_request,evaluator_root=evaluator_root)
    armroot=Path(request['output']);root=armroot/'downstream';root.mkdir(exist_ok=True);path=root/'request.json'
    if path.exists():
        scientific=cp._read(path)
        if scientific!=request:
            _guard_request(scientific,path)
            _bind_execution(scientific,request,path)
        request=scientific
    else:cp._publish(path,request)
    os.environ.update({native.ENV_REQUEST:str(path),native.ENV_SHA:cp._sha(path)})
    if (root/'completion.json').exists():return validate_completion(root)
    stage='admission';completion=None
    try:
        native.early_stage_guard(stage_action='prepare-native',gpu=gpu)
        admission=storage.verify_admission(storage_admission)
        recovery={}
        if (Path(admission['group_root'])/'sequential_native_wave.lease.json').exists():
            recovery=dict(recover_released_same_arm=True)
            if (root/'native_child.json').exists():recovery['predecessor_child']=root/'native_child.json'
            else:recovery['predecessor_status']=root/'status.json'
        with storage.sequential_wave_lock(storage_admission,arm,request_sha256=cp._sha(path),**recovery):
            _guard_request(request,path);storage.check_aggregate_disk(storage_admission,armroot)
            stage='prepare-native';_status(root,request,stage,'RUNNING')
            native_path=armroot/'native/native.json'
            if native_path.exists():native.private_modules()['NATIVE_PIPELINE']._guard_native(native_path)
            else:native.prepare_native(score_overlay,armroot/'native',storage_admission,gpu=gpu)
            stage='calibrate-native';_status(root,request,stage,'RUNNING')
            if (native_path.parent/'calibration.json').exists():native.private_modules()['NATIVE_PIPELINE'].validate_calibration(native_path)
            else:native.calibrate_native(native_path,gpu=gpu)
            stage='train';_status(root,request,stage,'RUNNING')
            completion=native.train(native_path,gpu=gpu)
            stage='metrics';_status(root,request,stage,'RUNNING',proof=completion)
            _guard_request(request,path);metrics=_metrics(request,completion)
            stage='report';_status(root,request,stage,'RUNNING',proof=completion)
            reports=_report(request,completion,metrics)
            _guard_request(request,path)
            proof=dict(format=COMPLETE_FORMAT,arm=arm,complete=True,debug=False,actual_data_evaluation=True,
                request_sha256=cp._sha(path),full_native_training_completed=True,
                full_native_evaluation_completed=True,full_lesion_metric_evaluation_completed=True,
                result_schema1_published=True,native_checkpoint='checkpoint_best.pth',
                completed_epochs=250,evaluated_cases=26,metrics_directory=str(metrics),
                metrics_completion_sha256=cp._sha(metrics/'completion.json'),reports=reports,
                native_final_checkpoint=completion['final_checkpoint'],
                native_best_receipt_sha256=cp._sha(native_path.parent/'nnUNet_results'/
                    completion['native']['baseline']['dataset_name']/
                    (completion['native']['trainer']+'__nnUNetResEncUNetMPlans__3d_fullres')/'fold_0/best_validation_complete.json'),
                completed_at=time.time())
            cp._publish(root/'completion.json',proof)
            _status(root,request,'complete','COMPLETE',proof=proof)
            return proof
    except BaseException as error:
        _status(root,request,stage,'FAILED',error=type(error).__name__+': '+str(error),proof=completion)
        raise
