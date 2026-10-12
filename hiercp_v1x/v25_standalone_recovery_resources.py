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

FORMAT='v25_exact_four_arm_recovery_continuation_v1'
ENV_PATH='V25_RECOVERY_CONTINUATION_SEAL'
ENV_SHA='V25_RECOVERY_CONTINUATION_SEAL_SHA256'
_BASE=direct.DirectAllocationLease
MAPPING={'A':(6,list(range(32,38))),'B':(5,list(range(42,48))),'C':(1,list(range(16,22))),'D':(4,list(range(6)))}
INCREMENTS={'A':8,'B':16,'C':32,'D':128}
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
        if (not isinstance(slot,dict) or set(slot)!={'arm','gpu','CPU_affinity','direct_proof','checkpoint','previous_actor'}
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
        if arm=='B' and (cp['epoch'],cp['updates'],cp['phase'],cp['train_position'],cp['evaluation_position'],cp['active_u'])!=(2,34,'full_validation',65,18,16):
            raise ValueError('B must retain original e2/full_validation/update34/eval18/U16')
        if arm=='D' and (cp['epoch'],cp['updates'],cp['phase'],cp['train_position'],cp['evaluation_position'],cp['active_u'])!=(2,34,'stage_validation',65,12,128):
            raise ValueError('D must retain actual paused e2/stage_validation/update34/eval12/U128')
        if arm=='D' and cp['status']!='PAUSED':raise ValueError('D requires actual safely paused own latest; no rewind')
    for table in ('preserved_files','runtime_files'):
        if not isinstance(value[table],dict) or not value[table] or any(not direct._checksum(v) for v in value[table].values()):
            raise ValueError('Immutable source and input admissions required')
    return value

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
    return seal

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
            str(Path(seal['private_code'])/'tools/run_v25_recovery_execution.py')]:
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
    if root.pid not in tree or root.cpu_affinity()!=slot['CPU_affinity']:
        raise RuntimeError('Per-arm stage origin is outside admitted tree or CPU slot')
    selected={root.pid:root}
    for child in root.children(recursive=True):
        try:actual=terminal._descendant(lease,child)
        except direct.psutil.NoSuchProcess:continue
        if actual is None:continue
        if child.pid not in tree:raise RuntimeError('Recorded stage descendant is outside exact controller tree')
        if child.cpu_affinity()!=slot['CPU_affinity']:raise RuntimeError('Per-arm descendant CPU affinity changed')
        selected[child.pid]=child
    return selected


class StandaloneRecoveryAllocationLease(_BASE):
    def _resources(self,controller,own,tree):
        validate_own_contracts(self);_BASE._resources(self,controller,own,tree)
        self.actual_resources.update(all_four_RSS_members=len(tree),single_controller_tree=True,
            aggregate416GiB_unchanged=True,GNN_aggregate_admission_bytes=288*2**30,native_aggregate_admission_bytes=416*2**30)
    def _gpu(self,tree,*,initial):
        # Only the exact monitored PID/birth/argv/UID stage origin and actual
        # descendants may own this GPU. Forged matching sibling affinity alone
        # can never turn the other arm into an admitted CUDA actor.
        selected=_stage_tree(self,tree)
        return _BASE._gpu(self,selected,initial=initial)


def install_from_environment():
    value=load_seal()
    if direct.DirectAllocationLease not in (_BASE,StandaloneRecoveryAllocationLease):raise ValueError('Unknown direct-resource override')
    direct.DirectAllocationLease=StandaloneRecoveryAllocationLease;terminal.install_runtime()
    return dict(format=FORMAT,source_sha256=sha(Path(__file__)),seal_sha256=os.environ[ENV_SHA],
        aggregate_RSS_bytes=416*2**30,GNN_RSS_bytes_per_arm=64*2**30,native_RSS_bytes_per_arm=96*2**30,
        GPU0_processes_modified=False,controller_identity=value['controller_identity'])
