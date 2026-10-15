"""Actual RAM512 limit values over the unchanged UI2 and native console."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import stat
import sys
from types import FunctionType

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import watch_basic_cp_admission as admission

QUEUE_ANCHOR = Path('/var/tmp/aicompetition06_basic_cp_seven_resume_1791934200100000000/queue_multi5_on5_candidates1024_on6')
QUEUE_ROUTES = [dict(arm='multi5',waiting_physical_GPU=0,target_physical_GPU=5,upstream_arm='candidates2048'),
    dict(arm='candidates1024',waiting_physical_GPU=4,target_physical_GPU=6,upstream_arm='multi7')]
_QUEUE_SHA_CACHE = {}


def _queue_require(condition, message):
    if not condition:
        raise ValueError(message)


def _queue_sha(path):
    """Hash each stable evidence inode once, including the original whole checkpoint."""
    path = Path(path); before = path.stat()
    _queue_require(path.is_absolute() and not path.is_symlink() and stat.S_ISREG(before.st_mode),
        'Queue viewer requires an absolute regular evidence file')
    if hasattr(os, 'getuid'):
        _queue_require(before.st_uid == 41833, 'Queue viewer evidence must belong to the experiment owner')
    key = (str(path),before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
    if key not in _QUEUE_SHA_CACHE:
        checksum = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(8*2**20), b''):
                checksum.update(chunk)
            after = os.fstat(source.fileno())
        _queue_require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
            == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
            'Queue evidence changed during viewer verification')
        _QUEUE_SHA_CACHE[key] = checksum.hexdigest()
    return _QUEUE_SHA_CACHE[key]


def _queue_pin(reference):
    _queue_require(isinstance(reference,dict) and set(reference)=={'path','sha256'}
        and _queue_sha(reference['path'])==reference['sha256'], 'Pinned queue viewer evidence changed')
    return Path(reference['path'])


def _queue_actor_released(actor):
    import psutil
    _queue_require(isinstance(actor,dict) and type(actor.get('pid')) is int and actor.get('UID')==41833
        and type(actor.get('create_time',actor.get('birth'))) in (int,float) and actor.get('command'),
        'Queue viewer requires the captured own actor identity')
    try:
        process = psutil.Process(actor['pid'])
        if process.create_time()!=actor.get('create_time',actor.get('birth')):
            return True
        _queue_require(process.uids().real==41833, 'Captured queue actor UID changed')
        return process.status() in (psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD)
    except (psutil.NoSuchProcess,psutil.ZombieProcess):
        return True


def _queue_identity(actor):
    return (actor['pid'],actor.get('create_time',actor.get('birth')),actor['UID'],tuple(actor['command']))


def _queue_waiting_value(value, *, phase, reason, completed_epochs, target, error=None):
    """Only data changes; the existing UI2 task renderer remains untouched."""
    status = 'FAILED' if error is not None else 'WAITING'
    if error is None and type(completed_epochs) is int:
        reason = str(completed_epochs)+'/250 epoch 저장 | '+str(reason or '')
    return dict(value,stage='scoring',stage_label='GPU 후속 학습 대기',status=status,
        controller_status=status,controller_stage='waiting_GPU_queue_transfer',
        completed_epochs=completed_epochs,waiting_reason=reason,waiting_reason_detail=reason,
        queued_physical_GPU=target,queue_phase=phase,error=error,native_started=False,
        full_training_completed=False,full_native_evaluation=False,
        RSS_GiB=None,RSS_source=None,process_state=None)


def _queue_snapshot_overlay(root, request, value):
    """Explain only the two actually paused original controllers through their saved queue."""
    root = Path(root)
    rules = [r for r in QUEUE_ROUTES if request.get('gpu')==r['waiting_physical_GPU']
             and request.get('arm')==r['arm']]
    status_path = QUEUE_ANCHOR/'status.json'
    if len(rules)!=1 or not status_path.exists():
        return value
    state = json.loads(status_path.read_bytes())
    _queue_require(state.get('format')=='basic_cp_RAM512_successor_queue_status_v1',
        'Actual successor queue status format required')
    config_path = _queue_pin(dict(path=str(QUEUE_ANCHOR/'config.json'),sha256=state.get('config_sha256')))
    config = json.loads(config_path.read_bytes())
    original_ref = config['initial_manifest']
    if state.get('current_manifest')!=original_ref:
        return value
    manifest = json.loads(_queue_pin(original_ref).read_bytes())
    rule = rules[0]; gpu = str(rule['waiting_physical_GPU'])
    registration = manifest['registration'][gpu]
    if registration['root']!=str(root):
        return value
    _queue_require(_queue_sha(root/'request.json')==registration['request_sha256']
        and request['controller_identity']==registration['controller_identity']
        and state.get('routes')==QUEUE_ROUTES and state.get('GPU0_GPU4_autostart_disabled') is True,
        'Original paused request and approved durable successor routes differ')
    paused = json.loads(_queue_pin(config['selected_pause_proof']).read_bytes())
    _queue_require(paused.get('format')=='basic_cp_two_selected_queue_paused_v1'
        and paused.get('only_selected_own_actors_signaled') is True
        and paused.get('other_running_experiments_signaled') is False
        and paused.get('original_checkpoint_bytes_changed') is False,
        'Actual selected own checkpoint pause proof required')
    actors = paused['released_actors'][gpu]
    _queue_require(any(_queue_identity(a)==_queue_identity(request['controller_identity']) for a in actors),
        'Paused release proof lacks the original registered controller')
    if not all(_queue_actor_released(a) for a in actors):
        return value
    checkpoints = paused['checkpoint_proofs'][gpu]
    expected_epoch = {0:116,4:66}[rule['waiting_physical_GPU']]
    for role in ('LATEST','BEST'):
        proof = checkpoints[role]
        _queue_require(proof.get('whole_completed_epoch') is True
            and proof.get('scaler_present') is True
            and type(proof.get('completed_epoch')) is int
            and 0<proof['completed_epoch']<=expected_epoch
            and _queue_sha(proof['path'])==proof['sha256'],
            'Actual paused whole LATEST/BEST checkpoint evidence changed')
    _queue_require(checkpoints['LATEST']['completed_epoch']==expected_epoch,
        'Original selected paused checkpoint epoch differs')
    phase = state.get('phase'); error = None
    reason = 'GPU'+str(rule['target_physical_GPU'])+' '+rule['upstream_arm']+' 완료 후 기존 checkpoint에서 재개 대기'
    if phase=='FAILED_REQUIRES_INSPECTION':
        error = state.get('error') or '후속 학습 대기 실행기 오류: 확인 필요'
        reason = error
    elif phase=='QUEUE_SAVED_ALLOCATION_ENDED':
        reason += ' | GPU 할당 종료: 대기 순서와 checkpoint 보존, 새 할당 필요'
    elif phase=='HANDOFF':
        reason += ' | 검증된 checkpoint를 새 GPU 실행에 연결 중'
    else:
        _queue_require(phase=='WAITING', 'Unknown successor queue phase requires inspection')
        if _queue_actor_released(state['actor']):
            error = '후속 학습 대기 실행기가 종료되어 확인 필요'
            reason = error
    return _queue_waiting_value(value,phase=phase,reason=reason,
        completed_epochs=expected_epoch,target=rule['target_physical_GPU'],error=error)


def build_entry(expanded,viewer):
    entry=admission.build_entry(expanded,viewer)
    original=entry.__globals__['snapshot']
    def snapshot(root):
        value=original(root);request=viewer.read(Path(root)/'request.json')
        required={'memory_budget_revision':512,'aggregate_memory_cap_GiB':512,
            'effective_native_RSS_limit_GiB':32,'effective_shared_bank_RSS_limit_GiB':320}
        if not isinstance(request,dict) or any(type(request.get(k))is not int or request[k]!=v for k,v in required.items()):
            raise ValueError('RAM512 viewer requires its actual approved request limits')
        value = _queue_snapshot_overlay(root,request,value)
        if value['controller_stage']=='waiting_GPU_queue_transfer' and value['stage']!='scoring':
            state = viewer.read(Path(root)/'status.json') or {}
            value = _queue_waiting_value(value,phase='WAITING',reason=state.get('reason'),
                completed_epochs=state.get('completed_epochs'),target=state.get('queued_physical_GPU'))
        cpu_source=value['physical_GPU']==5 and value['controller_stage']=='candidate_bank'
        result = dict(value,RAM_limit_GiB=request['effective_shared_bank_RSS_limit_GiB']if cpu_source
            else request['effective_native_RSS_limit_GiB'])
        if 'GPU56_sharing_receipt' in request:
            from hiercp_v1x.basic_cp_GPU56_sharing import validate_sharing_request
            validate_sharing_request(request)
            result.update(physical_GPU=request['execution_physical_GPU'],logical_slot=request['gpu'])
        return result
    scope=dict(entry.__globals__,snapshot=snapshot)
    return FunctionType(entry.__code__,scope,entry.__name__,entry.__defaults__,entry.__closure__)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--code-root',type=Path,required=True)
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--gpu',type=int,choices=range(7),
        help='Existing logical experiment slot; retained for saved commands')
    selection.add_argument('--arm',help='Exact scientific experiment name on a shared GPU')
    args,rest=parser.parse_known_args(argv)
    manifest=json.loads(args.manifest.read_bytes())
    registration=manifest['registration']
    if args.arm is None:
        row=registration[str(args.gpu)]
    else:
        matches=[row for row in registration.values()if row['arm']==args.arm]
        if len(matches)!=1:
            parser.error('Choose one registered --arm: '+', '.join(row['arm']for row in registration.values()))
        row=matches[0]
    code_root=args.code_root.resolve(strict=True)
    if code_root!=Path(manifest['code']).resolve(strict=True):
        raise ValueError('Explicit code root must equal the manifest pinned export')
    expanded,viewer=admission.load_pinned_viewers(code_root)
    return build_entry(expanded,viewer)(['--root',str(Path(row['root'])),*rest])


if __name__=='__main__':main()
