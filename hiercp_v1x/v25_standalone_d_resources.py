"""Admit one exact D continuation while observing the unchanged ABC tree.

The inherited tree/GPU guard admits only this standalone controller and its
children. A separately sealed original controller contributes to aggregate
RSS accounting only; its actors never become admitted GPU4 users.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from . import v25_direct_resources as direct
from . import v25_terminal_descendant_resources as terminal

FORMAT='v25_exact_D_standalone_continuation_v1'
ENV_PATH='V25_D_CONTINUATION_SEAL'
ENV_SHA='V25_D_CONTINUATION_SEAL_SHA256'
_BASE=direct.DirectAllocationLease
_SOURCE=Path(__file__).resolve()
_FIELDS={'format','group_root','private_code','previous_code','controller_identity',
    'direct_proof','sibling_manifest','sibling_controller_identity','sibling_slots',
    'preserved_files','runtime_files','checkpoint','previous_D_actor',
    'created_at','scientific_configuration_unchanged','no_process_signaled'}


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):result.update(block)
    return result.hexdigest()


def validate_seal(value):
    if (not isinstance(value,dict) or set(value)!=_FIELDS or value['format']!=FORMAT
            or value['scientific_configuration_unchanged'] is not True
            or value['no_process_signaled'] is not True
            or not direct._identity_document(value['controller_identity'])
            or not direct._identity_document(value['sibling_controller_identity'])
            or not direct._identity_document(value['previous_D_actor'])
            or value['controller_identity']['pid']==value['sibling_controller_identity']['pid']
            or set(value['sibling_slots'])!=set('ABC')):
        raise ValueError('Exact separate D controller, released D and unchanged ABC identities required')
    group=Path(value['group_root']);code=Path(value['private_code']);old=Path(value['previous_code'])
    if (not all(path.is_absolute() and '..' not in path.parts for path in (group,code,old))
            or code.parent!=group or old.parent!=group or code==old):
        raise ValueError('Fresh private D code must be alongside the unchanged live code')
    for proof in (value['direct_proof'],value['sibling_manifest'],*value['sibling_slots'].values()):
        if (not isinstance(proof,dict) or set(proof)!={'path','sha256'}
                or not isinstance(proof['path'],str) or not direct._checksum(proof['sha256'])):
            raise ValueError('Exact immutable resource file proof required')
    for name in ('preserved_files','runtime_files'):
        if not isinstance(value[name],dict) or not value[name]:raise ValueError('Immutable file admissions required')
        if any(not isinstance(path,str) or not direct._checksum(digest) for path,digest in value[name].items()):
            raise ValueError('Exact source and input SHA admissions required')
    checkpoint=value['checkpoint']
    if (not isinstance(checkpoint,dict)
            or set(checkpoint)!={'path','sha256','identity_sha256','content_sha256','epoch','updates','train_position'}
            or checkpoint['path']!=str(group/'D/training/checkpoint_latest.pt')
            or not all(direct._checksum(checkpoint[key]) for key in ('sha256','identity_sha256','content_sha256'))
            or (checkpoint['epoch'],checkpoint['updates'],checkpoint['train_position'])!=(2,20,12)):
        raise ValueError('Original owned D epoch2/update20/position12 continuation required')
    return value


def load_seal(path=None):
    path=direct._owned_regular(path or os.environ[ENV_PATH])
    if sha(path)!=os.environ.get(ENV_SHA):raise ValueError('Immutable standalone D seal changed')
    value=validate_seal(json.loads(path.read_bytes()))
    for table in ('preserved_files','runtime_files'):
        for name,digest in value[table].items():
            if sha(direct._owned_regular(name))!=digest:raise ValueError('Sealed D source/input changed: '+name)
    return value


def _guard_proof(proof):
    path=direct._owned_regular(proof['path'])
    if sha(path)!=proof['sha256']:raise ValueError('Original exact resource proof changed')
    return json.loads(path.read_bytes())


def _abc_complete(lease):
    proofs=getattr(lease,'_completed_ABC_files',None)
    if proofs is None:
        group=lease.group
        paths=[group/arm/'downstream/completion.json' for arm in 'ABC']
        proofs={str(path):sha(direct._owned_regular(path)) for path in paths}
        # Frozen validation loads real final/BEST receipts and changes native
        # environment bindings. A separate CPU interpreter keeps both outside
        # the running D model, RNG, CUDA state and hard process RSS workspace.
        program=('import json,sys;from pathlib import Path;'
            'from hiercp_v1x.v25_downstream import verify_completion;'
            'values=[verify_completion(Path(sys.argv[1])/arm/"downstream/completion.json") for arm in "ABC"];'
            'print(json.dumps(dict(actual_full_ABC_completion_verified=True)))')
        output=subprocess.check_output([sys.executable,'-B','-c',program,str(group)],
            text=True,timeout=180,env=os.environ.copy())
        if json.loads(output).get('actual_full_ABC_completion_verified') is not True:
            raise RuntimeError('Full ABC completion verification did not succeed')
        lease._completed_ABC_files=proofs
    for name,digest in proofs.items():
        if sha(direct._owned_regular(name))!=digest:raise ValueError('Verified ABC completion certificate changed')


def _surviving_ABC_descendants(lease,tree):
    union=dict(tree)
    for identity in getattr(lease,'_observed_ABC_descendants',{}).values():
        try:
            process=direct.psutil.Process(identity['pid'])
            observed=terminal._descendant(lease,process)
        except direct.psutil.NoSuchProcess:continue
        if observed is None:continue
        if observed!=identity:raise RuntimeError('Previously observed ABC descendant identity changed')
        # A previously observed active child that outlives its controller must
        # still be included. Identity reuse/mutation cannot masquerade as exit.
        union[process.pid]=process
    return union


def combined_resource_tree(lease,controller,own,tree):
    value=load_seal()
    if (str(lease.path)!=value['direct_proof']['path']
            or sha(lease.path)!=value['direct_proof']['sha256']
            or lease.request['arm']!='D' or lease.request['gpu']!=4
            or lease.request['CPU_affinity']!=list(range(6))
            or lease.request['controller_identity']!=value['controller_identity']
            or str(lease.group)!=value['group_root']):
        raise ValueError('Standalone resource adapter belongs only to this exact D proof')
    manifest=_guard_proof(value['sibling_manifest'])
    if (manifest.get('format')!='v25_four_GPU_independent_execution_v1'
            or manifest.get('group_root')!=value['group_root']
            or manifest.get('controller_identity')!=value['sibling_controller_identity']):
        raise ValueError('Original four-arm controller binding changed')
    for arm in 'ABC':
        slot=manifest['slots'][arm];proof=value['sibling_slots'][arm];document=_guard_proof(proof)
        direct.validate_direct_document(document)
        if (proof!=dict(path=slot['resource_request_path'],sha256=slot['resource_request_sha256'])
                or document['arm']!=arm or document['controller_identity']!=value['sibling_controller_identity']
                or document['CPU_affinity']!=slot['CPU_affinity'] or document['gpu']!=slot['gpu']
                or document['group_root']!=value['group_root']
                or document['host_lifetime']!=lease.request['host_lifetime']
                or document['aggregate_CPU_affinity']!=lease.request['aggregate_CPU_affinity']
                or sha(Path(value['previous_code'])/'hiercp_v1x/v25_direct_resources.py')!=document['resource_guard_sha256']):
            raise ValueError('Unchanged original ABC resource allocation required')
    original=value['sibling_controller_identity']
    try:peer=direct._exact(original)
    except direct.psutil.NoSuchProcess:
        # A missing original controller is valid only after every ABC chain has
        # actual full native250/BEST26 completion, independently revalidated.
        _abc_complete(lease)
        return _surviving_ABC_descendants(lease,tree)
    if peer.cpu_affinity()!=lease.request['aggregate_CPU_affinity']:
        raise RuntimeError('Original ABC controller CPU affinity changed')
    if direct.capture_cgroup_admission(peer.pid)!=document['cgroup_admission']:
        raise RuntimeError('Original ABC controller cgroup admission changed')
    union=dict(tree);union[peer.pid]=peer
    if not hasattr(lease,'_observed_ABC_descendants'):lease._observed_ABC_descendants={}
    for child in peer.children(recursive=True):
        try:identity=terminal._descendant(lease,child)
        except direct.psutil.NoSuchProcess:continue
        if identity is None:continue
        previous=lease.managed.get(child.pid)
        if previous is not None and previous!=identity:raise RuntimeError('Original ABC descendant identity changed')
        lease.managed[child.pid]=identity
        lease._observed_ABC_descendants[child.pid]=identity
        if child.pid in tree:raise RuntimeError('Separate controller trees unexpectedly overlap')
        union[child.pid]=child
    return union


class StandaloneDAllocationLease(_BASE):
    def _resources(self,controller,own,tree):
        union=combined_resource_tree(self,controller,own,tree)
        _BASE._resources(self,controller,own,union)
        self.actual_resources.update(standalone_D_tree_RSS_members=len(tree),
            combined_original_ABC_and_D_RSS_members=len(union),
            original_ABC_tree_only_used_for_RAM=True,
            GPU4_admitted_actors_only_standalone_D_tree=True)


def install_from_environment():
    value=load_seal()
    if direct.DirectAllocationLease not in (_BASE,StandaloneDAllocationLease):
        raise ValueError('Unknown direct-resource class override')
    direct.DirectAllocationLease=StandaloneDAllocationLease
    # The inherited original/terminal _tree remains compatible with every
    # frozen native guard's idempotent terminal-runtime installation.
    terminal.install_runtime()
    return dict(format=FORMAT,source_sha256=sha(_SOURCE),seal_sha256=os.environ[ENV_SHA],
        source_guard_unchanged=True,original_GPU4_tree_guard_unchanged=True,
        aggregate_RSS_bytes=416*2**30,D_GNN_RSS_bytes=64*2**30,
        ABC_processes_modified=False,controller_identity=value['controller_identity'])
