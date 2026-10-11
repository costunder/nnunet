"""Read-only UI2 adapter for existing CP and fixed-code native adoption."""
from __future__ import annotations
import argparse,copy,json,math,sys,time
from pathlib import Path
import psutil
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).absolute().parents[1]))
from tools import watch_v24_direct_gnn_training as direct
from tools import watch_v24_gpu_pipeline as pipeline
from tools import watch_v24_matched_cp_scoring as scoring
from tools import watch_v24_early_stop_pipeline as early_view
from tools import training_monitor_ui as ui

REQUEST='v24_native_prestart_adoption_request_v1'
STATUS='v24_native_prestart_adoption_status_v1'

def proof(witness):
    path=pipeline.input_path(witness['path'])
    if pipeline._file_stat(path)!=witness['stat'] or direct.sha(path)!=witness['sha256']:
        raise ValueError('Prestart viewer immutable input changed: '+str(path))
    return path

def guard_snapshot(value,renderer,path):
    if not value.get('snapshot_fresh'):
        return value
    path=Path(path);request=pipeline.read(path,required=True)
    original_path=proof(request['original_request']);original=pipeline.read(original_path,required=True)
    resource=original.get('resource_contract',original);gpu=request.get('GPU_arm')
    if (request.get('format')!=REQUEST or gpu not in (4,5) or value.get('GPU')!=gpu
            or original.get('format')!=(pipeline.NATIVE_REQUEST if gpu==4 else early_view.REQUEST)
            or request.get('recovery_root')!=str(path.parent) or request.get('resource_contract')!=resource
            or resource.get('GPU_arm')!=gpu or resource.get('epochs')!=250
            or resource.get('physical_batch')!=2 or resource.get('train_cases')!=105
            or resource.get('validation_cases')!=26 or resource.get('source_entries')!=642
            or resource.get('candidates_per_source')!=128 or resource.get('evaluation_checkpoint')!='checkpoint_best.pth'):
        raise ValueError('Own GPU4/5 original full642/native250/B2/BEST26 binding differs')
    gnn=value.get('gnn') or {}
    if gpu==4 and (gnn.get('full_training_completed') is not True or gnn.get('completed_epochs')!=40):
        raise ValueError('Own GPU4 completed full40 GNN required')
    if gpu==5 and (original.get('target_epoch')!=27 or original.get('full_GNN_training') is not False
            or gnn.get('completed_epochs')!=27 or gnn.get('full_training_completed') is not False):
        raise ValueError('Own user-authorized GPU5 early27 GNN required')
    digest=direct.sha(path)
    reg=pipeline.read(path.parent/'registration.json',required=True)
    lease=pipeline.read(path.parent/'lease.json',required=True)
    state=pipeline.read(path.parent/'status.json',required=True)
    command=[request['python'],'-B','-u',str(Path(request['code'])/'tools/run_v24_native_prestart_guard.py'),'watch','--request',str(path)]
    if (state.get('format')!=STATUS or state.get('GPU_arm')!=gpu or state.get('request_sha256')!=digest
            or state.get('source_output')!=resource['source_output'] or reg.get('request_sha256')!=digest
            or reg.get('command')!=command or reg.get('UID')!=41833 or lease.get('command')!=command
            or lease.get('UID')!=41833 or lease.get('request_sha256')!=digest
            or state.get('supervisor_pid')!=reg.get('pid') or state.get('supervisor_create_time')!=reg.get('create_time')
            or lease.get('pid')!=reg.get('pid') or lease.get('create_time')!=reg.get('create_time')):
        raise ValueError('Prestart own controller/lease/process metadata differs')
    controller=scoring.process_snapshot(reg)
    stage=state.get('stage');child=dict(alive=False,process_status='absent')
    if stage not in (None,'score',*(entry['name'] for entry in request['stages'])):
        raise ValueError('Prestart unknown stage differs from sealed original/native flow')
    if state.get('child_pid') is not None:
        expected=request['scoring_actor'] if stage=='score' else next(
            (entry for entry in request['stages'] if entry['name']==stage),None)
        if expected is None or state.get('command')!=expected['command']:
            raise ValueError('Prestart exact adopted/future child command differs')
        if stage=='score' and (state['child_pid']!=expected['pid'] or state['child_create_time']!=expected['create_time']):
            raise ValueError('Prestart adopted CP identity differs')
        child=scoring.process_snapshot(dict(pid=state['child_pid'],create_time=state['child_create_time'],command=expected['command']))
    label=state['status']
    if not controller['alive'] and label not in ('COMPLETE','FAILED'):
        label='CONTROLLER_EXITED_CHILD_ALIVE' if child['alive'] else 'EXITED_WITHOUT_COMPLETION'
    reason=state.get('error') if label=='FAILED' else state.get('waiting_reason')
    training=label=='RUNNING' and stage=='train' and child['alive']
    native=dict(GPU=gpu,status=label,stage=stage,controller_alive=controller['alive'],child_alive=child['alive'],
        native_training_started=training,epochs=250,full_native_training=False,full_native_evaluation=False,
        native_pending_reason=reason,prestart_root=str(path.parent),snapshot_source='sealed native prestart adoption and exact process identity')
    value.pop('read_problem',None)
    value.update(native=native,native_pending_reason=reason,full_native_training=False,full_native_evaluation=False)
    cp=value.get('scoring')
    if cp and cp.get('source_done')!=642:
        cp['status']='SCORING_PROCESS_ALIVE' if cp.get('alive') else cp['status']
        value.update(active_stage='scoring',status=cp['status'])
    else:
        value.update(active_stage='scoring',status=label)
    if training:
        log=path.parent/'train.log'
        if state.get('stage_log')!=str(log):
            raise ValueError('Prestart actual native console binding differs')
        metrics=renderer.parse_nnunet_lines(renderer.log_tail(log,65536))
        native.update(child,RAM_limit_GiB=48)
        if state.get('child_process_tree_RSS_bytes') is not None:
            native['RSS_GiB']=state['child_process_tree_RSS_bytes']/2**30
        if metrics:
            native.update(epoch=metrics['epoch_index']+1,completed_epochs=metrics['epoch_index']+int(metrics['epoch_finished']),native_metrics=metrics)
        value.update(active_stage='native',status=label)
    if label=='COMPLETE':
        if state.get('full_native_training_completed') is not True or state.get('full_native_evaluation_completed') is not True:
            raise ValueError('Actual complete250/BEST26 evidence required')
        native.update(pipeline.native_completion(Path(resource['chain_root']),resource,state,
            Path(resource['source_output']),gpu,resource['source_code']),native_pending_reason=None)
        value.update(active_stage='native',status='COMPLETE',full_native_training=True,full_native_evaluation=True)
    elif label in ('FAILED','EXITED_WITHOUT_COMPLETION','CONTROLLER_EXITED_CHILD_ALIVE'):
        value.update(status=label,read_problem=dict(detail=reason or label))
    return value

class Viewer:
    def __init__(self,renderer,path):
        self.renderer=renderer;self.path=Path(path);self.last_valid=None
        request=pipeline.read(self.path,required=True)
        original_path=proof(request['original_request']);original=pipeline.read(original_path,required=True)
        if request['GPU_arm']==4:
            self.base=pipeline.PipelineViewer(renderer,None,original['scoring_root'],None,gnn_root=original['source_output'],gpu=4)
        else:
            self.base=early_view.Viewer(original_path,original_path.parent.parent/'pause_guardian',original['source_output'],renderer)
    def snapshot(self):
        try:
            value=guard_snapshot(self.base.snapshot(),self.renderer,self.path)
        except (OSError,ValueError,KeyError,TypeError,psutil.AccessDenied) as error:
            value=copy.deepcopy(self.last_valid) if self.last_valid else dict(GPU=None,active_stage='metadata',gnn=None,scoring=None,native=None,full_native_training=False,full_native_evaluation=False)
            value.update(last_confirmed_status=value.get('status'),status='READ_PENDING' if isinstance(error,scoring.PendingJSON) else 'READ_ERROR',snapshot_fresh=False,
                last_valid_snapshot_preserved=self.last_valid is not None,read_problem=dict(detail=type(error).__name__+': '+str(error)))
        if value.get('snapshot_fresh'):
            self.last_valid=copy.deepcopy(value)
        return value

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--renderer',type=Path,required=True)
    parser.add_argument('--renderer-sha',required=True)
    parser.add_argument('--once',action='store_true');parser.add_argument('--json',action='store_true')
    parser.add_argument('--no-clear',action='store_true');parser.add_argument('--interval',type=float,default=2.)
    args=parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval<=0:
        parser.error('--interval must be finite and positive')
    viewer=Viewer(direct.load_renderer(args.renderer,args.renderer_sha),args.request)
    terminal=ui.TerminalView(sys.stdout,layout='task') if sys.stdout.isatty() and not(args.no_clear or args.once or args.json) else None
    previous=None
    try:
        while True:
            value=viewer.snapshot()
            if args.json:
                print(json.dumps(dict(value,ui_schema_version=2,experiment_version='v2.4'),ensure_ascii=False,allow_nan=False));return
            if value.get('active_stage')=='native' and (value.get('native') or {}).get('native_training_started'):
                if terminal:terminal.close();terminal=None
                shown=pipeline.display(value,experiment_version='v2.4')
            elif terminal:
                terminal.render(pipeline.task_model(value,experiment_version='v2.4'));shown=None
            else:shown=pipeline.display(value,experiment_version='v2.4')
            if shown is not None and shown!=previous:print(shown,flush=True);previous=shown
            if args.once:return
            time.sleep(args.interval)
    except KeyboardInterrupt:print('\n진행 뷰어만 종료했습니다. 학습 작업은 계속됩니다.',flush=True)
    finally:
        if terminal:terminal.close()
if __name__=='__main__':main()
