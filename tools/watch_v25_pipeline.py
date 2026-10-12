"""Read-only v2.5 arm observer using the existing GNN/CP task UI schema2.

No training imports, CUDA work, signals, job starts or result writes occur here.
Native output is the original timestamp/Epoch console, without a new layout.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import types

import psutil

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from tools import watch_v24_training as original
from tools import training_monitor_adapter as adapter
from tools import training_monitor_ui as ui

ARMS=('A','B','C','D')
CP_PROGRESS='v25_CP_durable_progress_v1'
CP_REQUEST='v25_owned_BEST_Basic642_durable_score_store_v1'
CP_OVERLAY='v25_owned_BEST_complete_Basic_81case642source128positions_v1'
GNN_FORMAT='v25_all_P_patient_balanced_shared_context_training_v1'
NATIVE_STATUS='v25_full_own_arm_downstream_status_v1'


def input_path(path):
    path=Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Explicit absolute observer path required: '+str(path))
    for item in (path,*path.parents):
        if item.is_symlink():
            raise ValueError('Observer input symlink is not admitted: '+str(item))
    if path.exists() and hasattr(os,'getuid') and path.stat().st_uid!=os.getuid():
        raise ValueError('Observer input belongs to another UID: '+str(path))
    return path


def read(path, *, required=False):
    path=input_path(path)
    value=original.read_json(path)
    if value is None and required:
        raise original.JsonRecordError(path,'Required metadata has not been published',pending=True)
    return value


def sha(path):
    digest=hashlib.sha256()
    with input_path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def actor_snapshot(pid):
    """Observe one owned process tree; zombie/dead actors are not RUNNING."""
    try:
        process=psutil.Process(pid)
        status=process.status()
        if status==psutil.STATUS_ZOMBIE:
            return None
        uid=process.uids().real if hasattr(process,'uids') else None
        rss=process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                if uid is None or child.uids().real==uid:
                    rss+=child.memory_info().rss
            except (psutil.NoSuchProcess,psutil.ZombieProcess):
                continue
        return dict(pid=pid,parent_pid=process.ppid(),create_time=process.create_time(),command=process.cmdline(),
            UID=uid,process_status=status,RSS_bytes=rss)
    except (psutil.NoSuchProcess,psutil.ZombieProcess):
        return None


def declared_actor(state):
    if isinstance(state.get('actor'),dict):
        return state['actor']
    if state.get('child_pid') is not None:
        return dict(pid=state.get('child_pid'),create_time=state.get('child_create_time'),
            command=state.get('child_command'),UID=state.get('UID'))
    if state.get('pid') is not None:
        return dict(pid=state.get('pid'),create_time=state.get('create_time'),
            command=state.get('command'),UID=state.get('UID'))
    return None


def observe_actor(declaration, *, commands, path):
    if declaration is None:
        return None,'PROCESS_IDENTITY_PENDING'
    if not isinstance(declaration,dict):
        raise ValueError(str(path)+': actor identity must be an object')
    pid,birth,command,uid=(declaration.get(key) for key in ('pid','create_time','command','UID'))
    if (type(pid) is not int or pid<=0 or type(birth) not in (int,float) or not math.isfinite(birth)
            or birth<=0 or not isinstance(command,list) or not command
            or any(not isinstance(item,str) or not item for item in command)
            or uid is not None and (type(uid) is not int or uid<0)):
        raise ValueError(str(path)+': complete PID/birth/argv/UID actor identity required')
    if hasattr(os,'getuid') and uid!=os.getuid():
        raise ValueError(str(path)+': actor belongs to another UID')
    if command not in commands:
        raise ValueError(str(path)+': actor argv differs from declared stage commands')
    actual=actor_snapshot(pid)
    if actual is None:
        return None,'EXITED_WITHOUT_COMPLETION'
    if (actual['pid']!=pid or abs(actual['create_time']-birth)>1e-3
            or actual['command']!=command or actual['UID']!=uid):
        raise ValueError(str(path)+': PID was reused or actual UID/birth/argv differs')
    return actual,None


class GNNObservation(original.Viewer):
    """Reuse the original phase/curve reader, routing only its console input."""
    def observe(self, root, state, log, actor):
        self.observed_state=state;self.observed_log=log;self.observed_actor=actor
        namespace=dict(original.Viewer._snapshot.__globals__,read_json=self._read,
            log_tail=self._tail)
        function=types.FunctionType(original.Viewer._snapshot.__code__,namespace,
            original.Viewer._snapshot.__name__)
        value=function(self,root)
        # An execution contract states planned epochs; no event means the
        # actual epoch/counter/progress has not yet been observed.
        if not any(self.events[name] for name in ('update_timing','validation_timing','curve','invocations')) and not value.get('reports'):
            value.update(epoch=None,completed_epochs=None,optimizer_updates=None)
            if state.get('stage') not in ('prepare_inputs','calibrate'):
                value['progress']=None
        value['snapshot_fresh']=True
        return value

    def _read(self,path):
        if Path(path)==self.root/'status.json':
            return self.observed_state
        return read(path)

    def _tail(self,path,byte_count=16384):
        if Path(path)==self.root/'console.log':
            path=self.observed_log
        lines = original.log_tail(input_path(path),byte_count) if path else []
        command = (self.observed_actor or {}).get('command', [])
        if '--parallel-prepare' in command and '--input-cache' in command:
            cache = input_path(command[command.index('--input-cache')+1])
            records = cache/'local/records'
            # Reuse the established canonical progress row. Count actual stored
            # records from all four writers, rather than one shard's log count.
            if records.is_dir() and not (cache/'local/index.json').exists():
                done = sum(1 for record in records.glob('*.pt.gz'))
                lines.append(f'v2.5 GT-free CPU canonical four_process: {done}/14102 records')
        return lines

    def _rss(self,status,latest):
        if self.observed_actor is not None:
            return self.observed_actor['RSS_bytes'],'live owned arm process tree'
        if latest and latest.get('RSS_bytes') is not None:
            return latest['RSS_bytes'],'last successful train update'
        return None,None

    def _calibration(self,root):
        state=None
        for line in self._tail(root/'console.log'):
            match=re.search(r'v25 CALIBRATION patients(\d+)/chunk(\d+)/allP\+128U',line)
            if match:
                state=dict(patients=int(match[1]),chunk=int(match[2]),repeats_completed=0)
            repeat=re.search(r'v25 CALIBRATION repeat(\d+) ([\d.]+)s',line)
            if repeat and state:
                state.update(repeats_completed=int(repeat[1]),last_seconds=float(repeat[2]))
        return state


class Viewer:
    def __init__(self,group,arm=None):
        self.group=input_path(group);self.selected_arm=arm
        if arm is not None and arm not in ARMS:
            raise ValueError('One exact v25 arm A/B/C/D required')
        self.gnn={};self.last_valid=None;self.problem=None;self.last_arm=None
        self.hashes={}

    def witness(self,path,checksum):
        path=input_path(path);info=path.stat()
        key=(str(path),info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
        if key not in self.hashes:
            self.hashes[key]=sha(path)
        if self.hashes[key]!=checksum:
            raise ValueError('Completion witness bytes changed: '+str(path))

    def snapshot(self):
        try:
            value=self._snapshot()
        except original.JsonRecordError as error:
            signature=(error.path,error.identity,error.detail)
            now=time.monotonic()
            if self.problem is None or self.problem['signature']!=signature:
                self.problem=dict(signature=signature,since=now)
            age=now-self.problem['since']
            return self.failure('READ_PENDING' if error.pending and age<10 else 'READ_ERROR',str(error),age)
        except (ValueError,OSError,psutil.AccessDenied) as error:
            return self.failure('READ_ERROR',type(error).__name__+': '+str(error))
        self.last_valid=copy.deepcopy(value);self.problem=None
        return value

    def failure(self,status,detail,age=None):
        value=copy.deepcopy(self.last_valid) if self.last_valid else dict(GPU=None,arm=self.selected_arm,
            group=str(self.group),active_stage='gnn',gnn=None,scoring=None,native=None,
            full_native_training=False,full_native_evaluation=False)
        value.update(status=status,snapshot_fresh=False,last_confirmed_status=value.get('status'),
            last_valid_snapshot_preserved=self.last_valid is not None,
            read_problem=dict(detail=detail,unchanged_seconds=age,retries_continue=True))
        return value

    def _snapshot(self):
        group=read(self.group/'status.json')
        arm=self.selected_arm or (group or {}).get('arm')
        if arm not in ARMS:
            if arm is not None:
                raise ValueError('Group status references an unknown v25 arm')
            return dict(group=str(self.group),arm=None,GPU=None,status='WAITING_METADATA',active_stage='gnn',
                snapshot_fresh=True,gnn=None,scoring=None,native=None,
                full_native_training=False,full_native_evaluation=False)
        if arm!=self.last_arm:
            self.last_valid=None;self.problem=None;self.last_arm=arm
        output=input_path(self.group/arm);monitor=output/'monitor';path=monitor/'status.json'
        state=read(path,required=True)
        request=state.get('request') or read(monitor/'request.json',required=True)
        if (not isinstance(request,dict) or request.get('arm')!=arm
                or type(request.get('GPU')) is not int or request['GPU']<0
                or Path(request.get('production_output',''))!=output
                or type(request.get('RAM_GiB')) not in (int,float)
                or request['RAM_GiB']!=(96 if state.get('stage')=='native' else 64)):
            raise ValueError(str(path)+': exact own arm/output/GPU/RAM request required')
        commands=[]
        stages=request.get('stages')
        if not isinstance(stages,list) or not stages:
            raise ValueError(str(path)+': explicit stage commands required')
        for stage in stages:
            if (not isinstance(stage,dict) or not isinstance(stage.get('command'),list)
                    or not stage['command'] or any(not isinstance(v,str) or not v for v in stage['command'])):
                raise ValueError(str(path)+': stage argv must be a nonempty string list')
            commands.append(stage['command'])
        if state.get('arm',arm)!=arm:
            raise ValueError(str(path)+': current controller arm differs')
        if group and group.get('arm')==arm and group.get('output') is not None and Path(group['output'])!=output:
            raise ValueError(str(self.group/'status.json')+': group output differs from its selected arm')
        actor,actor_problem=observe_actor(declared_actor(state),commands=commands,path=path)
        status=state.get('status','UNKNOWN')
        if not isinstance(status,str) or not isinstance(state.get('stage'),(str,type(None))):
            raise ValueError(str(path)+': controller status/stage must be textual')
        if status in ('RUNNING','PREPARING') and actor_problem:
            status=actor_problem
        log=input_path(state.get('stage_log') or monitor/'console.log')
        if not log.is_relative_to(output):
            raise ValueError(str(path)+': stage console escaped the selected arm')
        stage=state.get('stage')
        normal=copy.deepcopy(state);normal['status']=status
        normal['request']=request
        if stage in ('gnn','training'):
            normal['stage']='train'
        # A gnn-pipeline child first prepares/calibrates, before its immutable
        # execution contract exists. Observe its actual console phase.
        if normal.get('stage')=='train' and not (output/'training/training_identity.json').exists():
            for line in reversed(original.log_tail(log)):
                if 'v25 CALIBRATION' in line:
                    normal['stage']='calibrate';break
                if 'CPU canonical' in line or 'UPPER PREPARE' in line:
                    normal['stage']='prepare_inputs';break
        reader=self.gnn.setdefault(arm,GNNObservation())
        gnn=reader.observe(monitor,normal,log,actor)
        gnn.update(status=status,process_status=(actor or {}).get('process_status'))
        if gnn.get('epochs') is None:
            gnn['epochs']=request.get('epochs')
        invocations,_=original.JsonlCursor().read(input_path(output/'training/invocations.jsonl'))
        complete=invocations[-1] if invocations else {}
        best=complete.get('best')
        gnn_done=(complete.get('format')==GNN_FORMAT and complete.get('status')=='COMPLETE'
            and complete.get('full_training') is True and complete.get('debug') is False
            and complete.get('actual_CUDA') is True and complete.get('completed_epochs')==40
            and complete.get('optimizer_updates')==680 and complete.get('connected_parameter_tensors')==537
            and complete.get('expected_parameter_tensors')==537
            and isinstance(best,dict) and best.get('selected_by')=='fixed_full128_validation_only')
        gnn['full_training_completed']=gnn_done
        if gnn_done:
            gnn.update(epoch=40,epochs=40,completed_epochs=40,optimizer_updates=680,active_U=128,
                status='TRAINING_COMPLETE',stage='complete',stage_label='전체 GNN 학습 완료',progress=None)
        elif gnn.get('status')=='TRAINING_COMPLETE':
            gnn['status']='COMPLETION_UNVERIFIED'
        if not gnn_done and gnn.get('completed_epochs')==40:
            gnn.update(epoch=40,stage='completion_receipt_pending',stage_label='GNN 완료 증거 대기',progress=None)
        value=dict(group=str(self.group),arm=arm,GPU=request['GPU'],status=status,snapshot_fresh=True,
            active_stage='gnn',gnn=gnn,scoring=None,native=None,
            full_native_training=False,full_native_evaluation=False,
            native_pending_reason=state.get('waiting_reason') or state.get('reason'))
        if state.get('error'):
            gnn['error']=state['error']
        cp=self.scoring(output,arm,commands,log=log if stage in ('scoring','score','score-cp','cp') else None)
        if cp is not None:
            value['scoring']=cp
        native=self.native(output,arm,commands)
        if native is not None:
            value['native']=native
        if stage in ('scoring','score','score-cp','cp') or cp and not cp.get('complete'):
            value.update(active_stage='scoring',status=(cp or {}).get('status',status))
        elif stage in ('native','prepare_native','calibrate_native','nnunet_training','evaluation','report') or native:
            value.update(active_stage='native',status=(native or {}).get('status',status))
            value['full_native_training']=(native or {}).get('full_native_training') is True
            value['full_native_evaluation']=(native or {}).get('full_native_evaluation') is True
        elif gnn_done and status=='COMPLETE':
            value['status']='TRAINING_COMPLETE'
        return value

    def scoring(self,output,arm,commands,*,log=None):
        root=output/'cp_bank';path=root/'progress.json';progress=read(path)
        if progress is None:
            return None
        request=read(root/'request.json',required=True);pin=request.get('best_provenance') or {}
        if not isinstance(pin,dict):
            raise ValueError(str(root/'request.json')+': own BEST provenance must be an object')
        if (progress.get('format')!=CP_PROGRESS or progress.get('arm')!=arm
                or request.get('format')!=CP_REQUEST or pin.get('arm')!=arm
                or progress.get('request_sha256')!=sha(root/'request.json')
                or progress.get('total_sources')!=642 or progress.get('total_positions')!=82176
                or type(progress.get('completed_sources')) is not int or not 0<=progress['completed_sources']<=642
                or progress.get('completed_positions')!=128*progress['completed_sources']
                or pin.get('completed_epochs')!=40 or pin.get('optimizer_updates')!=680
                or pin.get('selected_by')!='fixed_full128_validation_only'
                or pin.get('full_training') is not True or pin.get('debug') is not False):
            raise ValueError(str(path)+': own full40/680 BEST and durable full642/82176 progress binding differs')
        actor,problem=observe_actor(declared_actor(progress),commands=commands,path=path)
        status=progress.get('status','UNKNOWN');complete=False
        overlay=read(root/'score_overlay.json')
        if status=='COMPLETE':
            if (not overlay or overlay.get('format')!=CP_OVERLAY or overlay.get('complete') is not True
                    or overlay.get('complete_sources')!=642 or overlay.get('complete_positions')!=82176
                    or overlay.get('debug') is not False or overlay.get('actual_CUDA') is not True
                    or overlay.get('full_CP_scoring_executed') is not True
                    or overlay.get('best_provenance')!=pin
                    or overlay.get('durable_score_request_sha256')!=progress['request_sha256']
                    or overlay.get('content_sha256')!=canonical({k:v for k,v in overlay.items() if k!='content_sha256'})):
                status='SCORING_COMPLETION_UNVERIFIED'
            else:
                complete=True;status='SCORING_COMPLETE'
        elif status=='RUNNING' and problem:
            status=problem
        elif status=='FAILED':
            status='SCORING_FAILED'
        return dict(status=status,complete=complete,source_done=progress['completed_sources'],source_total=642,
            CP_done=progress['completed_positions'],CP_total=82176,own_BEST_pin_verified=True,
            last_batch_seconds=progress.get('last_batch_seconds'),
            seconds_since_console_write=max(0,time.time()-log.stat().st_mtime) if log and log.is_file() else None,
            RSS_GiB=actor['RSS_bytes']/2**30 if actor else None,RAM_limit_GiB=64,
            process_status=(actor or {}).get('process_status'),detail=progress.get('error') or problem
                or ('실제 CP 실패 기록: '+str(path) if status=='SCORING_FAILED' else None))

    def native(self,output,arm,commands):
        path=output/'downstream/status.json';state=read(path)
        if state is None:
            return None
        if state.get('format')!=NATIVE_STATUS or state.get('arm')!=arm:
            raise ValueError(str(path)+': downstream format/own arm differs')
        request_path=path.parent/'request.json';request=read(request_path,required=True)
        if (request.get('format')!='v25_full_own_arm_downstream_request_v1' or request.get('arm')!=arm
                or request.get('output')!=str(output) or state.get('request_sha256')!=sha(request_path)
                or request.get('debug') is not False or request.get('train_cases')!=105
                or request.get('validation_cases')!=26 or request.get('source_entries')!=642
                or request.get('candidates_per_source')!=128 or request.get('native_epochs')!=250
                or request.get('native_physical_batch')!=2 or request.get('native_patch_size')!=[128]*3
                or request.get('cp_probability')!=.5 or request.get('evaluation_checkpoint')!='checkpoint_best.pth'
                or request.get('result_schema_version')!=1
                or request.get('process_and_children_RSS_bytes')!=96*2**30
                or request.get('shared_cache_memory_reserve_bytes')!=32*2**30):
            raise ValueError(str(path)+': downstream request/own output differs')
        actor,problem=observe_actor(declared_actor(state),commands=commands,path=path)
        status=state.get('state') or state.get('status') or 'UNKNOWN'
        if status=='RUNNING' and problem:
            status=problem
        log=None;lines=[]
        if state.get('stage_log'):
            log=input_path(state['stage_log'])
            if not log.is_relative_to(output):
                raise ValueError(str(path)+': native console escaped its own downstream folder')
            lines=original.log_tail(log,65536)
        child_path=path.parent/'native_child.json';child=read(child_path)
        if child is not None:
            if (child.get('format')!='v25_exact_owned_downstream_child_v1' or child.get('arm')!=arm
                    or child.get('request_sha256')!=sha(request_path)
                    or child.get('parent_actor')!=declared_actor(state)):
                raise ValueError(str(child_path)+': native child parent/request/arm differs')
            actual,child_problem=observe_actor(declared_actor(child),commands=[child.get('command')],path=child_path)
            if actual is not None and actual.get('parent_pid')!=child['parent_actor']['pid']:
                raise ValueError(str(child_path)+': actual child belongs to another process tree')
            if child.get('state')=='RUNNING' or state.get('stage')=='train':
                actor=actual
                if child_problem and status=='RUNNING':
                    status=child_problem
                log=input_path(child.get('log_path',''))
                metric_log=(state.get('stage')=='metrics' and log.parent==output/'downstream'
                    and re.fullmatch(r'metric_\d+\.log',log.name) is not None)
                if not log.is_relative_to(output/'native') and not metric_log:
                    raise ValueError(str(child_path)+': child log escaped its exact native/metric stage root')
                lines=original.log_tail(log,65536)
        # Metric/report children share the exact parent/request actor protocol,
        # but their stdout is not an nnU-Net Epoch logger. An exited prior train
        # child must not make a later phase appear to be actively training.
        metrics=original.parse_nnunet_lines(lines) if state.get('stage')=='train' else None
        result=dict(status='COMPLETION_UNVERIFIED' if status=='COMPLETE' else status,
            stage=state.get('stage'),epoch=metrics['epoch_index']+1 if metrics else None,epochs=250,
            completed_epochs=metrics['epoch_index']+int(metrics['epoch_finished']) if metrics else None,
            native_metrics=metrics,native_training_started=actor is not None and bool(metrics),
            full_native_training=False,full_native_evaluation=False,
            native_pending_reason=state.get('error') or state.get('waiting_reason') or problem
                or (child_problem if child is not None else None)
                or ('nnU-Net 후속 단계: '+str(state.get('stage')) if state.get('stage')!='train' else None),
            RSS_GiB=actor['RSS_bytes']/2**30 if actor else None,
            RAM_limit_GiB=request['process_and_children_RSS_bytes']/2**30,
            process_status=(actor or {}).get('process_status'),console_lines=lines,stage_log=str(log) if log else None)
        if status=='COMPLETE':
            proof=read(path.parent/'completion.json')
            if proof is not None:
                self.native_completion(output,arm,request,proof)
                result.update(status='COMPLETE',full_native_training=True,full_native_evaluation=True,
                    epoch=250,epochs=250,completed_epochs=250,completion_metadata_verified=True,
                    completion_verification_scope='own complete250/BEST26 certificate and immutable metadata/byte witnesses; no model deserialization')
            else:
                result['native_pending_reason']='실제 full250/BEST26 완료 certificate 기록 대기'
        return result

    def native_completion(self,output,arm,request,proof):
        root=output/'downstream';request_hash=sha(root/'request.json')
        if (proof.get('format')!='v25_full_own_arm_downstream_completion_v1' or proof.get('arm')!=arm
                or proof.get('request_sha256')!=request_hash or proof.get('debug') is not False
                or any(proof.get(key) is not True for key in ('complete','actual_data_evaluation',
                    'full_native_training_completed','full_native_evaluation_completed',
                    'full_lesion_metric_evaluation_completed','result_schema1_published'))
                or proof.get('native_checkpoint')!='checkpoint_best.pth'
                or proof.get('completed_epochs')!=250 or proof.get('evaluated_cases')!=26):
            raise ValueError(str(root/'completion.json')+': actual own full250/BEST26/metrics/schema1 certificate differs')
        native=read(output/'native/native.json',required=True)
        cases=(native.get('split') or {}).get('outer_val')
        baseline=native.get('baseline');trainer=native.get('trainer')
        if (native.get('format')!='v25_four_arm_Basic642_full250_native_v1' or native.get('arm')!=arm
                or native.get('root')!=str(output/'native') or not isinstance(cases,list)
                or any(not isinstance(case,str) or not case for case in cases)
                or len(cases)!=26 or len(set(cases))!=26 or not isinstance(baseline,dict)
                or not isinstance(baseline.get('dataset_name'),str) or not isinstance(trainer,str)):
            raise ValueError('Own native complete26 cohort/root metadata differs')
        fold=input_path(output/'native/nnUNet_results'/baseline['dataset_name']/(trainer+'__nnUNetResEncUNetMPlans__3d_fullres')/'fold_0')
        if not fold.is_relative_to(output/'native'):
            raise ValueError('Own native fold escaped its reserved native root')
        receipt_path=fold/'best_validation_complete.json';receipt=read(receipt_path,required=True)
        self.witness(receipt_path,proof.get('native_best_receipt_sha256'))
        if (receipt.get('format')!='v24_native_best_loaded_full26_validation_v1'
                or receipt.get('checkpoint_name')!='checkpoint_best.pth'
                or receipt.get('checkpoint_path')!=str(fold/'checkpoint_best.pth')
                or receipt.get('native_fold')!=str(fold) or receipt.get('validation_cases')!=sorted(cases)
                or receipt.get('bank_sha256')!=native.get('bank_sha256')
                or any(receipt.get(key) is not True for key in ('complete','native_validation_returned_successfully',
                    'original_checkpoint_load_called','original_full_validation_called'))
                or any(receipt.get(key) is not False for key in ('final_checkpoint_fallback','GT_passed_to_predictor','CP_at_inference'))
                or set(receipt.get('predictions_sha256',{}))!=set(cases)):
            raise ValueError(str(receipt_path)+': actual own BEST/full26 validation receipt differs')
        self.witness(fold/'checkpoint_best.pth',receipt.get('checkpoint_sha256'))
        final=proof.get('native_final_checkpoint') or {}
        if final.get('path')!=str(fold/'checkpoint_final.pth'):
            raise ValueError('FINAL full250 witness differs from own native fold')
        self.witness(final['path'],final.get('sha256'))
        self.witness(fold/'validation/summary.json',receipt.get('summary_sha256'))
        metrics=input_path(proof.get('metrics_directory',''))
        if not metrics.is_relative_to(root):
            raise ValueError('Own complete lesion metrics escaped the downstream folder')
        self.witness(metrics/'completion.json',proof.get('metrics_completion_sha256'))
        reports=proof.get('reports')
        if not isinstance(reports,dict) or set(reports)!={'evidence','text','json'}:
            raise ValueError('All three actual Result schema1 artifacts required')
        for item in reports.values():
            if not isinstance(item,dict):
                raise ValueError('Actual Result schema1 artifact proof must be an object')
            report=input_path(item.get('path',''))
            if not report.is_relative_to(root):
                raise ValueError('Own Result schema1 artifact escaped downstream folder')
            self.witness(report,item.get('sha256'))
        document=read(reports['json']['path'],required=True)
        if document.get('format')!='nnunet_completed_result_layout_v1' or document.get('result_schema_version')!=1:
            raise ValueError('Actual Result schema1 report format differs')


def task_model(value):
    model=adapter.from_pipeline(value,experiment_version='v2.5')
    # Arm labels occupy an existing value, keeping every UI2 row unchanged.
    label=model['gnn'].get('phase_label')
    model['gnn']['phase_label']='arm '+str(value.get('arm') or '-')+(' | '+label if label else '')
    if value.get('active_stage')=='native':
        model['state']['stage']='CP 후보 채점' if value.get('scoring') else 'GNN'
    elif model['state']['stage'] not in ('GNN','CP 후보 채점'):
        model['state']['stage']='GNN'
    if value.get('scoring'):
        model['cp']['input_basis']='arm '+str(value.get('arm'))+' | '+str(model['cp'].get('input_basis') or '-')
    if value.get('native'):
        model['waiting_reason']=value['native'].get('native_pending_reason') or model.get('waiting_reason')
    return model


def display(value):
    native=value.get('native') or {}
    if (value.get('snapshot_fresh') and value.get('active_stage')=='native'
            and native.get('stage')=='train' and native.get('console_lines')):
        # Preserve the actual logger's timestamp, Epoch, OnlineCP and NumPy
        # scalar representation. Never present pseudo-Dice as BEST26 metrics.
        console='\n'.join(native['console_lines'])
        if native.get('status') not in ('RUNNING','COMPLETE') or value.get('status')=='FAILED':
            reason=(value.get('gnn') or {}).get('error') or native.get('native_pending_reason')
            prefix='nnU-Net controller: '+str(native.get('status'))+' | 단계 '+str(native.get('stage'))
            if reason:
                prefix+='\nnnU-Net 오류/대기 이유: '+str(reason)
            return prefix+'\n'+console
        return console
    return ui.render_task(task_model(value))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--json',action='store_true')
    parser.add_argument('--interval',type=float,default=2.)
    args=parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval<=0:
        parser.error('--interval must be finite and positive')
    viewer=Viewer(args.group,args.arm)
    terminal=ui.TerminalView(sys.stdout,layout='task') if sys.stdout.isatty() and not(args.once or args.json) else None
    previous=None
    try:
        while True:
            value=viewer.snapshot()
            if args.json:
                value.update(ui_schema_version=2,experiment_version='v2.5')
                print(json.dumps(value,ensure_ascii=False,allow_nan=False));return
            native=value.get('native') or {}
            if (terminal and value.get('active_stage')=='native' and native.get('stage')=='train'
                    and native.get('console_lines')):
                terminal.close();terminal=None
            if terminal:
                terminal.render(task_model(value))
            else:
                shown=display(value)
                if shown!=previous:
                    print(shown,flush=True);previous=shown
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n진행 뷰어만 종료했습니다. 학습/scoring 작업은 계속됩니다.',flush=True)
    finally:
        if terminal:
            terminal.close()


if __name__=='__main__':
    main()
