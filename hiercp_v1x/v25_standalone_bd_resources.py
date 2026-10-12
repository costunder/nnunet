"""One exact B/D continuation controller; original A/C only contributes RSS."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from . import v25_direct_resources as direct
from . import v25_terminal_descendant_resources as terminal

FORMAT='v25_exact_BD_standalone_continuation_v1'
ENV_PATH='V25_BD_CONTINUATION_SEAL'
ENV_SHA='V25_BD_CONTINUATION_SEAL_SHA256'
_BASE=direct.DirectAllocationLease
MAPPING={'B':(5,list(range(42,48))),'D':(4,list(range(6)))}
_FIELDS={'format','group_root','private_code','previous_code','controller_identity','slots',
    'sibling_manifest','sibling_controller_identity','sibling_slots','preserved_files','runtime_files',
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
            or not direct._identity_document(value['controller_identity'])
            or not direct._identity_document(value['sibling_controller_identity'])
            or value['controller_identity']['pid']==value['sibling_controller_identity']['pid']
            or set(value['slots'])!=set('BD') or set(value['sibling_slots'])!=set('AC')):
        raise ValueError('Exact new BD and unchanged original AC controllers required')
    group=Path(value['group_root']);code=Path(value['private_code']);old=Path(value['previous_code'])
    if (not all(p.is_absolute() and '..'not in p.parts for p in (group,code,old)) or code.parent!=group
            or old.parent!=group or code==old):raise ValueError('Fresh BD private code required')
    proofs=[value['sibling_manifest'],*value['sibling_slots'].values()]
    for arm,(gpu,cpus) in MAPPING.items():
        slot=value['slots'][arm]
        if (not isinstance(slot,dict) or set(slot)!={'arm','gpu','CPU_affinity','direct_proof','checkpoint','previous_actor'}
                or slot['arm']!=arm or slot['gpu']!=gpu or slot['CPU_affinity']!=cpus
                or not direct._identity_document(slot['previous_actor'])):raise ValueError('Exact B5/D4 six-core slots required')
        proofs.append(slot['direct_proof']);cp=slot['checkpoint']
        if (not isinstance(cp,dict) or set(cp)!={'path','sha256','identity_sha256','content_sha256','epoch','updates',
                'phase','train_position','evaluation_position','active_u','status'}
                or cp['path']!=str(group/arm/'training/checkpoint_latest.pt')
                or not all(direct._checksum(cp[k]) for k in ('sha256','identity_sha256','content_sha256'))
                or any(type(cp[k])is not int for k in ('epoch','updates','train_position','evaluation_position','active_u'))):
            raise ValueError('Exact owned optimizer/RNG checkpoint receipt required')
        if arm=='B' and (cp['epoch'],cp['updates'],cp['phase'],cp['train_position'],cp['evaluation_position'],cp['active_u'])!=(2,34,'full_validation',65,18,16):
            raise ValueError('B must resume original e2/full_validation/update34/eval18/U16')
        if arm=='D' and (cp['status']!='PAUSED' or cp['active_u']!=128 or not 20<=cp['updates']<=680
                or not 2<=cp['epoch']<=40 or not 0<=cp['train_position']<=65
                or not 0<=cp['evaluation_position']<=(65 if cp['phase']=='train_probe'else 21)
                or cp['phase']not in ('training','train_probe','stage_validation','full_validation','epoch_completion')):
            raise ValueError('D requires actual safely paused latest full-U128 checkpoint; no rewind')
    for proof in proofs:
        if (not isinstance(proof,dict) or set(proof)!={'path','sha256'} or not isinstance(proof['path'],str)
                or not direct._checksum(proof['sha256'])):raise ValueError('Exact immutable resource proofs required')
    for table in ('preserved_files','runtime_files'):
        if not isinstance(value[table],dict) or not value[table] or any(not direct._checksum(v) for v in value[table].values()):
            raise ValueError('Immutable source and input admissions required')
    return value


def load_seal(path=None):
    path=direct._owned_regular(path or os.environ[ENV_PATH])
    if sha(path)!=os.environ.get(ENV_SHA):raise ValueError('BD continuation seal changed')
    value=validate_seal(json.loads(path.read_bytes()))
    for table in ('preserved_files','runtime_files'):
        for name,digest in value[table].items():
            if sha(direct._owned_regular(name))!=digest:raise ValueError('Sealed BD source/input changed: '+name)
    return value


def _completed_AC(lease):
    proofs=getattr(lease,'_completed_AC_files',None)
    if proofs is None:
        proofs={str(lease.group/arm/'downstream/completion.json'):sha(direct._owned_regular(lease.group/arm/'downstream/completion.json')) for arm in 'AC'}
        program=('import json,sys;from pathlib import Path;from hiercp_v1x.v25_downstream import verify_completion;'
            'values=[verify_completion(Path(sys.argv[1])/arm/"downstream/completion.json") for arm in "AC"];'
            'print(json.dumps(dict(actual_full_AC_completion_verified=True)))')
        output=subprocess.check_output([sys.executable,'-B','-c',program,str(lease.group)],text=True,timeout=180,env=os.environ.copy())
        if json.loads(output).get('actual_full_AC_completion_verified') is not True:raise RuntimeError('Actual full AC completion unverified')
        lease._completed_AC_files=proofs
    for name,digest in proofs.items():
        if sha(direct._owned_regular(name))!=digest:raise ValueError('Verified AC completion certificate changed')


def _survivors(lease,tree):
    union=dict(tree)
    for old in getattr(lease,'_observed_AC_descendants',{}).values():
        try:p=direct.psutil.Process(old['pid']);actual=terminal._descendant(lease,p)
        except direct.psutil.NoSuchProcess:continue
        if actual is None:continue
        if actual!=old:raise RuntimeError('Reparented AC identity changed')
        if p.pid in union:raise RuntimeError('Separate AC/BD descendants overlap')
        union[p.pid]=p
    return union


def combined_resource_tree(lease,controller,own,tree):
    seal=load_seal();arm=lease.request['arm']
    if arm not in MAPPING:raise ValueError('Standalone controller admits B/D only')
    slot=seal['slots'][arm]
    if (str(lease.path)!=slot['direct_proof']['path'] or sha(lease.path)!=slot['direct_proof']['sha256']
            or lease.request['gpu']!=slot['gpu'] or lease.request['CPU_affinity']!=slot['CPU_affinity']
            or lease.request['controller_identity']!=seal['controller_identity'] or str(lease.group)!=seal['group_root']):
        raise ValueError('Exact same-controller BD slot proof required')
    # Both live slots retain the original complete resource document schema.
    for selected in 'BD':
        declared=seal['slots'][selected];proof=direct.validate_direct_document(_guard_proof(declared['direct_proof']))
        if (proof['arm']!=selected or proof['gpu']!=declared['gpu'] or proof['CPU_affinity']!=declared['CPU_affinity']
                or proof['controller_identity']!=seal['controller_identity'] or proof['group_root']!=seal['group_root']
                or proof['host_lifetime']!=lease.request['host_lifetime']
                or proof['aggregate_CPU_affinity']!=lease.request['aggregate_CPU_affinity']
                or proof['resource_guard_sha256']!=lease.request['resource_guard_sha256']):
            raise ValueError('Exact original full BD resource contracts required')
    manifest=_guard_proof(seal['sibling_manifest']);cgroup=None
    if (manifest.get('format')!='v25_four_GPU_independent_execution_v1' or manifest.get('group_root')!=seal['group_root']
            or manifest.get('controller_identity')!=seal['sibling_controller_identity']):raise ValueError('Original AC controller manifest changed')
    for selected in 'AC':
        declared=manifest['slots'][selected];proof=seal['sibling_slots'][selected];value=direct.validate_direct_document(_guard_proof(proof))
        if (proof!=dict(path=declared['resource_request_path'],sha256=declared['resource_request_sha256'])
                or value['arm']!=selected or value['gpu']!=declared['gpu'] or value['CPU_affinity']!=declared['CPU_affinity']
                or value['controller_identity']!=seal['sibling_controller_identity'] or value['group_root']!=seal['group_root']
                or value['host_lifetime']!=lease.request['host_lifetime'] or value['aggregate_CPU_affinity']!=lease.request['aggregate_CPU_affinity']
                or sha(Path(seal['previous_code'])/'hiercp_v1x/v25_direct_resources.py')!=value['resource_guard_sha256']
                or cgroup is not None and value['cgroup_admission']!=cgroup):raise ValueError('Exact unchanged AC resource proof required')
        cgroup=value['cgroup_admission']
    try:peer=direct._exact(seal['sibling_controller_identity'])
    except direct.psutil.NoSuchProcess:
        _completed_AC(lease);return _survivors(lease,tree)
    if peer.cpu_affinity()!=lease.request['aggregate_CPU_affinity'] or direct.capture_cgroup_admission(peer.pid)!=cgroup:
        raise RuntimeError('Original AC controller CPU/cgroup changed')
    union=dict(tree)
    if peer.pid in union:raise RuntimeError('Separate controller trees overlap')
    union[peer.pid]=peer
    if not hasattr(lease,'_observed_AC_descendants'):lease._observed_AC_descendants={}
    for child in peer.children(recursive=True):
        try:actual=terminal._descendant(lease,child)
        except direct.psutil.NoSuchProcess:continue
        if actual is None:continue
        old=lease.managed.get(child.pid)
        if old is not None and old!=actual:raise RuntimeError('Exact AC child identity changed')
        if child.pid in union:raise RuntimeError('Separate controller descendants overlap')
        lease.managed[child.pid]=actual;lease._observed_AC_descendants[child.pid]=actual;union[child.pid]=child
    return union


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
            str(Path(seal['private_code'])/'tools/run_v25_bd_execution.py')]:
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


class StandaloneBDAllocationLease(_BASE):
    def _resources(self,controller,own,tree):
        union=combined_resource_tree(self,controller,own,tree);_BASE._resources(self,controller,own,union)
        self.actual_resources.update(BD_tree_RSS_members=len(tree),all_AC_BD_RSS_members=len(union),
            AC_only_used_for_RAM=True,aggregate416GiB_unchanged=True)
    def _gpu(self,tree,*,initial):
        # Only the exact monitored PID/birth/argv/UID stage origin and actual
        # descendants may own this GPU. Forged matching sibling affinity alone
        # can never turn the other arm into an admitted CUDA actor.
        selected=_stage_tree(self,tree)
        return _BASE._gpu(self,selected,initial=initial)


def install_from_environment():
    value=load_seal()
    if direct.DirectAllocationLease not in (_BASE,StandaloneBDAllocationLease):raise ValueError('Unknown direct-resource override')
    direct.DirectAllocationLease=StandaloneBDAllocationLease;terminal.install_runtime()
    return dict(format=FORMAT,source_sha256=sha(Path(__file__)),seal_sha256=os.environ[ENV_SHA],
        aggregate_RSS_bytes=416*2**30,GNN_RSS_bytes_per_arm=64*2**30,native_RSS_bytes_per_arm=96*2**30,
        AC_GPU0_processes_modified=False,controller_identity=value['controller_identity'])
