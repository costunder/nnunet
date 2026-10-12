"""One exact all-four continuation controller; GPU0 remains independent."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from . import v25_direct_resources as direct
from . import v25_terminal_descendant_resources as terminal

FORMAT='v25_exact_four_arm_u128_validation_reuse_v1'
ENV_PATH='V25_U128_REUSE_CONTINUATION_SEAL'
ENV_SHA='V25_U128_REUSE_CONTINUATION_SEAL_SHA256'
_BASE=direct.DirectAllocationLease
MAPPING={'A':(6,list(range(32,38))),'B':(5,list(range(42,48))),'C':(1,list(range(16,22))),'D':(4,list(range(6)))}
INCREMENTS={'A':8,'B':16,'C':32,'D':128}
PREVIOUS_TRAINING={'A':(2489145,1791808660.16),'B':(2489075,1791808659.79),
    'C':(2489077,1791808659.81),'D':(2489076,1791808659.79)}
_FIELDS={'format','group_root','private_code','previous_code','controller_identity','slots',
    'preserved_files','runtime_files',
    'created_at','scientific_configuration_unchanged','no_process_signaled'}


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(2**20),b''):result.update(block)
    return result.hexdigest()


def _guard_proof(proof):
    path=direct._owned_regular(proof['path'])
    if sha(path)!=proof['sha256']:raise ValueError('Immutable resource proof changed')
    return json.loads(path.read_bytes())


def _proof_reference(value,group):
    return (isinstance(value,dict) and set(value)=={'path','sha256'}
        and isinstance(value['path'],str) and direct._checksum(value['sha256'])
        and Path(value['path']).is_absolute() and '..'not in Path(value['path']).parts
        and Path(value['path']).is_relative_to(group))


def _recovery_declaration(value,slot,seal):
    group=Path(seal['group_root'])
    if (not isinstance(value,dict) or set(value)!={'mode','registration','monitor','failure_log','previous_controller_identity'}
            or value['mode']not in ('batch_paused','natural_failed')
            or not direct._identity_document(value['previous_controller_identity'])
            or not all(_proof_reference(value[key],group)for key in ('registration','monitor'))
            or value['mode']=='natural_failed'and not _proof_reference(value['failure_log'],group)
            or value['mode']=='batch_paused'and value['failure_log']is not None):
        raise ValueError('Exact owned terminal/pause recovery evidence required')
    expected='RUNNING'if value['mode']=='natural_failed'else'PAUSED'
    if slot['checkpoint']['status']!=expected:
        raise ValueError('Exact terminal recovery must preserve original checkpoint status bytes')
    for key in ('registration','monitor','failure_log'):
        proof=value[key]
        if proof is not None and seal['preserved_files'].get(proof['path'])!=proof['sha256']:
            raise ValueError('Immutable terminal recovery evidence must be sealed')


def validate_seal(value):
    if (not isinstance(value,dict) or set(value)!=_FIELDS or value['format']!=FORMAT
            or value['scientific_configuration_unchanged'] is not True or value['no_process_signaled'] is not True
            or not direct._identity_document(value['controller_identity']) or set(value['slots'])!=set('ABCD')):
        raise ValueError('Exact one-controller all-four continuation required')
    group=Path(value['group_root']);code=Path(value['private_code']);old=Path(value['previous_code'])
    if (not all(p.is_absolute() and '..'not in p.parts for p in (group,code,old)) or code.parent!=group
            or old.parent!=group or code==old):raise ValueError('Fresh all-four private code required')
    for arm,(gpu,cpus) in MAPPING.items():
        slot=value['slots'][arm]
        if (not isinstance(slot,dict) or set(slot)!={'arm','gpu','CPU_affinity','direct_proof','checkpoint','previous_actor','recovery'}
                or slot['arm']!=arm or slot['gpu']!=gpu or slot['CPU_affinity']!=cpus
                or not direct._identity_document(slot['previous_actor'])):raise ValueError('Exact original arm GPU/six-core slot required')
        proof=slot['direct_proof'];cp=slot['checkpoint']
        if (not isinstance(proof,dict) or set(proof)!={'path','sha256'} or not isinstance(proof['path'],str)
                or not direct._checksum(proof['sha256'])):raise ValueError('Exact immutable resource proof required')
        if (not isinstance(cp,dict) or set(cp)!={'path','sha256','identity_sha256','content_sha256','epoch','updates',
                'phase','train_position','evaluation_position','active_u','status'}
                or cp['path']!=str(group/arm/'training/checkpoint_latest.pt')
                or not all(direct._checksum(cp[k]) for k in ('sha256','identity_sha256','content_sha256'))
                or any(type(cp[k])is not int for k in ('epoch','updates','train_position','evaluation_position','active_u'))
                or not 1<=cp['epoch']<=40 or not 0<=cp['updates']<=680 or not 0<=cp['train_position']<=65
                or cp['phase']not in ('initial_full_validation','training','train_probe','stage_validation','full_validation','epoch_completion')
                or not 0<=cp['evaluation_position']<=(65 if cp['phase']=='train_probe'else 21)
                or cp['status']not in ('RUNNING','PAUSED') or cp['active_u']not in range(INCREMENTS[arm],129,INCREMENTS[arm])):
            raise ValueError('Exact full-scale owned optimizer/RNG checkpoint receipt required')
        _recovery_declaration(slot['recovery'],slot,value)
    for table in ('preserved_files','runtime_files'):
        if not isinstance(value[table],dict) or not value[table] or any(not direct._checksum(v) for v in value[table].values()):
            raise ValueError('Immutable source and input admissions required')
    return value


def _released(identity):
    try:process=direct.psutil.Process(identity['pid'])
    except direct.psutil.NoSuchProcess:return
    if process.create_time()!=identity['create_time']or process.uids().real!=identity['UID']:
        raise RuntimeError('Previous owned terminal actor PID was reused')
    raise RuntimeError('Previous owned terminal actor has not been reaped; no signal or restart')


def _old_argument(command,name):
    if command.count(name)!=1 or command.index(name)+1>=len(command):
        raise ValueError('Exact previous stage argument required: '+name)
    return command[command.index(name)+1]


def _validate_terminal_resume(slot,seal):
    """Read immutable real failure/pause evidence; never alter the checkpoint."""
    evidence=slot['recovery'];arm=slot['arm'];group=Path(seal['group_root']);old=Path(seal['previous_code'])
    registration=_guard_proof(evidence['registration']);monitor=_guard_proof(evidence['monitor'])
    previous=evidence['previous_controller_identity']
    if ((previous['pid'],previous['create_time'])!=(2488819,1791808638.21)
            or previous['command'][:4]!=['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u',
                str(old/'tools/run_v25_recovery_continuation.py')]
            or _old_argument(previous['command'],'--group')!=str(group)):
        raise ValueError('Exact previous owned controller evidence required')
    base={'format','arm','marker','child','controller','created_at','reason','full40_GNN_preserved','no_signal_sent','token'}
    failed=evidence['mode']=='natural_failed'
    fields=base|({'status_path','status_sha256','failure_log','failure_log_sha256'}if failed else set())
    if (not isinstance(registration,dict) or set(registration)!=fields
            or registration['format']!=('v25_owned_validation_reuse_terminal_v1'if failed else'v25_owned_validation_reuse_pause_v1')
            or registration['arm']!=arm or registration['controller']!=previous
            or registration['full40_GNN_preserved']is not True or registration['no_signal_sent']is not True
            or not direct._identity_document(registration['child'])):
        raise ValueError('Exact root-owned real terminal/pause registration required')
    training=registration['child'];command=training['command']
    if ((training['pid'],training['create_time'])!=PREVIOUS_TRAINING[arm]
            or command[:4]!=['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u',
                str(old/'tools/run_v25_recovery_execution.py')]
            or any(_old_argument(command,name)!=expected for name,expected in (
                ('--arm',arm),('--gpu',str(slot['gpu'])),('--output',str(group/arm)),('--mode','train')))):
        raise ValueError('Exact previous owned original training actor required')
    terminal_identity=dict(pid=monitor.get('child_pid'),create_time=monitor.get('child_create_time'),
        command=monitor.get('child_command'),UID=monitor.get('UID'))
    if (not direct._identity_document(terminal_identity) or terminal_identity!=slot['previous_actor']
            or monitor.get('controller_pid')!=previous['pid']or monitor.get('status')!='FAILED'):
        raise ValueError('Exact archived own FAILED terminal monitor required')
    if failed:
        proof=evidence['failure_log']
        if (registration['marker']is not None or registration['token']is not None
                or registration['status_path']!=str(group/arm/'monitor/status.json')
                or registration['status_sha256']!=evidence['monitor']['sha256']
                or registration['failure_log']!=proof['path']or registration['failure_log_sha256']!=proof['sha256']
                or monitor.get('stage_log')!=proof['path']or monitor.get('stage')!='train'
                or terminal_identity!=training):
            raise ValueError('Exact real original failure log/actor binding required')
        path=direct._owned_regular(proof['path'])
        if (sha(path)!=proof['sha256']or 'RuntimeError: Recorded stage descendant is outside exact controller tree'
                not in path.read_text()):raise ValueError('Actual observed owned stage-tree failure required')
    else:
        if (registration['marker']!=str(group/arm/'training/STOP_AFTER_BATCH')
                or registration['token']!='U128_REUSE_'+arm+'_20261012_'+str(training['pid'])):
            raise ValueError('Exact own batch-boundary pause token required')
        if monitor.get('stage')=='train':
            if terminal_identity!=training:raise ValueError('Exact own paused training actor required')
        elif monitor.get('stage')=='scoring':
            argv=terminal_identity['command']
            if (argv[:4]!=command[:4]or any(_old_argument(argv,name)!=expected for name,expected in (
                    ('--arm',arm),('--gpu',str(slot['gpu'])),('--output',str(group/arm)),('--mode','score-cp')))):
                raise ValueError('Exact post-pause owned scoring actor required')
        else:raise ValueError('Only own train/scoring terminal pause stages admitted')
    for actor in (training,terminal_identity,previous):_released(actor)
    return evidence

def load_seal(path=None):
    path=direct._owned_regular(path or os.environ[ENV_PATH])
    if sha(path)!=os.environ.get(ENV_SHA):raise ValueError('Four-arm continuation seal changed')
    value=validate_seal(json.loads(path.read_bytes()))
    for table in ('preserved_files','runtime_files'):
        for name,digest in value[table].items():
            if sha(direct._owned_regular(name))!=digest:raise ValueError('Sealed four-arm source/input changed: '+name)
    return value


def validate_own_contracts(lease):
    seal=load_seal();arm=lease.request['arm']
    if arm not in MAPPING:raise ValueError('All-four controller admits original arms only')
    slot=seal['slots'][arm]
    if (str(lease.path)!=slot['direct_proof']['path'] or sha(lease.path)!=slot['direct_proof']['sha256']
            or lease.request['gpu']!=slot['gpu'] or lease.request['CPU_affinity']!=slot['CPU_affinity']
            or lease.request['controller_identity']!=seal['controller_identity'] or str(lease.group)!=seal['group_root']):
        raise ValueError('Exact same-controller own slot proof required')
    for selected in 'ABCD':
        declared=seal['slots'][selected];proof=direct.validate_direct_document(_guard_proof(declared['direct_proof']))
        if (proof['arm']!=selected or proof['gpu']!=declared['gpu'] or proof['CPU_affinity']!=declared['CPU_affinity']
                or proof['controller_identity']!=seal['controller_identity'] or proof['group_root']!=seal['group_root']
                or proof['host_lifetime']!=lease.request['host_lifetime']
                or proof['aggregate_CPU_affinity']!=lease.request['aggregate_CPU_affinity']
                or proof['resource_guard_sha256']!=lease.request['resource_guard_sha256']
                or proof['cgroup_admission']!=lease.request['cgroup_admission']):
            raise ValueError('Exact original complete four-arm resource contracts required')
        _validate_terminal_resume(declared,seal)
    return seal

def _remember_live(lease,process,observed,tree,cpus):
    """Extend a stale snapshot only after an exact live lineage observation."""
    previous=lease.managed.get(process.pid)
    if previous is not None and previous!=observed:
        raise RuntimeError('Managed direct child PID/birth/argv identity changed')
    if process.cpu_affinity()!=cpus:raise RuntimeError('Per-arm descendant CPU affinity changed')
    lease.managed[process.pid]=observed
    tree[process.pid]=process


def _current_descendant(lease,child,origin,tree,cpus):
    """Prove current ancestry; preparation workers can postdate _tree()."""
    selected={};observations={};current=child;seen=set()
    while current.pid!=origin['pid']:
        if current.pid in seen:raise RuntimeError('Per-arm descendant ancestry cycle')
        seen.add(current.pid)
        observed=terminal._descendant(lease,current)
        if observed is None:return None
        if current.pid==lease.request['controller_identity']['pid']:
            raise RuntimeError('Recorded stage descendant is outside exact stage ancestry')
        parent=current.parent()
        if parent is None or current.ppid()!=parent.pid:
            raise RuntimeError('Per-arm descendant parent identity changed during observation')
        observations[current.pid]=observed
        selected[current.pid]=current
        current=parent
    if direct._actor(current)!=origin:
        raise RuntimeError('Exact per-arm stage origin PID/birth/UID/argv changed')
    for pid,process in selected.items():
        _remember_live(lease,process,observations[pid],tree,cpus)
    return selected


def _stage_tree(lease,tree):
    path=direct._owned_regular(lease.group/lease.request['arm']/'monitor/status.json')
    value=json.loads(path.read_bytes())
    if value.get('controller_pid')!=lease.request['controller_identity']['pid']:
        # Initial admission happens before new workers exist; the old failed
        # monitor cannot grant GPU ownership to any new/foreign actor.
        return {}
    identity=dict(pid=value['child_pid'],create_time=value['child_create_time'],
        command=value['child_command'],UID=value['UID'])
    if not direct._identity_document(identity):raise RuntimeError('Exact per-arm stage origin required')
    command=identity['command'];arm=lease.request['arm'];seal=load_seal();slot=seal['slots'][arm]
    if command[:4]!=['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10','-B','-u',
            str(Path(seal['private_code'])/'tools/run_v25_u128_reuse_execution.py')]:
        raise RuntimeError('Exact sealed per-arm wrapper entry required')
    def flag(name):
        if command.count(name)!=1 or command.index(name)+1>=len(command):raise RuntimeError('Exact per-arm stage argument required: '+name)
        return command[command.index(name)+1]
    if (flag('--arm')!=arm or flag('--gpu')!=str(slot['gpu']) or flag('--output')!=str(lease.group/arm)
            or flag('--allocation-request')!=slot['direct_proof']['path']
            or flag('--continuation-seal')!=os.environ[ENV_PATH]):raise RuntimeError('Own per-arm stage binding changed')
    stage=value.get('stage')
    if stage in ('train','scoring'):
        if flag('--mode')!=('train'if stage=='train'else'score-cp'):raise RuntimeError('Exact GNN stage mode required')
    elif stage=='native':
        if command[4]!='--native':raise RuntimeError('Exact native stage bootstrap required')
    else:raise RuntimeError('Only recorded train/scoring/native stages can own CUDA')
    try:root=direct._exact(identity)
    except direct.psutil.NoSuchProcess:return {}
    controller=direct._exact(lease.request['controller_identity'])
    parent=root.parent()
    if parent is None or parent.pid!=controller.pid or root.ppid()!=controller.pid:
        raise RuntimeError('Per-arm stage origin is outside exact controller ancestry')
    if direct._actor(parent)!=lease.request['controller_identity']:
        raise RuntimeError('Exact controller PID/birth/UID/argv changed')
    _remember_live(lease,root,identity,tree,slot['CPU_affinity'])
    selected={root.pid:root}
    for child in root.children(recursive=True):
        try:actual=_current_descendant(lease,child,identity,tree,slot['CPU_affinity'])
        except direct.psutil.NoSuchProcess:continue
        if actual is None:continue
        selected.update(actual)
    return selected


class U128ReuseAllocationLease(_BASE):
    def _resources(self,controller,own,tree):
        validate_own_contracts(self);_BASE._resources(self,controller,own,tree)
        self.actual_resources.update(all_four_RSS_members=len(tree),single_controller_tree=True,
            aggregate416GiB_unchanged=True,GNN_aggregate_admission_bytes=288*2**30,native_aggregate_admission_bytes=416*2**30)
    def _gpu(self,tree,*,initial):
        # Only the exact monitored PID/birth/argv/UID stage origin and actual
        # descendants may own this GPU. Forged matching sibling affinity alone
        # can never turn the other arm into an admitted CUDA actor.
        before=set(tree)
        selected=_stage_tree(self,tree)
        if set(tree)!=before:
            # qstat/cgroup queries take time after the original tree snapshot.
            # Newly proven workers must also pass the unchanged RAM admission.
            self._resources(direct._exact(self.request['controller_identity']),
                direct._exact(self.self_identity),tree)
        return _BASE._gpu(self,selected,initial=initial)


def install_from_environment():
    value=load_seal()
    if direct.DirectAllocationLease not in (_BASE,U128ReuseAllocationLease):raise ValueError('Unknown direct-resource override')
    direct.DirectAllocationLease=U128ReuseAllocationLease;terminal.install_runtime()
    return dict(format=FORMAT,source_sha256=sha(Path(__file__)),seal_sha256=os.environ[ENV_SHA],
        aggregate_RSS_bytes=416*2**30,GNN_RSS_bytes_per_arm=64*2**30,native_RSS_bytes_per_arm=96*2**30,
        GPU0_processes_modified=False,controller_identity=value['controller_identity'])
