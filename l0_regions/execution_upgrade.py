"""Narrow, explicit continuation across reviewed execution-only fixes."""
import hashlib
import subprocess
from pathlib import Path
from functools import lru_cache

ROOT=Path(__file__).resolve().parents[1]
MODEL='hiercp_v222/model.py'
OLD_MODEL='906d1eeb7982ed9f01801a22fdea72de68a82a7d24b061295e14489052af6920'
REVIEWED_MODEL='53d4d95220eab2ba643424a3e161fac4b91ba6a44f8ec7a108fa44dd9dce0a7d'
REVISIONS=('87eafa6', 'f07b13f5cc656aa413fc1f89b4f66c0925158559',
           '2526466', 'ba87e71', 'b398b3d8c0044d883729b789e83d69c0ed841725')


def compatible_core(previous,current):
    if previous==current:return True
    if set(previous)!=set(current):return False
    return (previous.get(MODEL)==OLD_MODEL and current.get(MODEL)==REVIEWED_MODEL
            and all(previous[k]==current[k] for k in previous if k!=MODEL))


@lru_cache(maxsize=256)
def blob_hash(revision,path):
    value=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',
        'show',f'{revision}:{path}'],cwd=ROOT)
    return hashlib.sha256(value).hexdigest()


def verify_upgrade(previous,current,*,allow_cuda_budget_change=False):
    # Explicit device migration changes only the allocator ceiling, never batch,
    # model, dataset, precision, RAM budget or saved optimization state.
    if allow_cuda_budget_change:
        before=previous.get('resource_limits',{});after=current.get('resource_limits',{})
        if set(before)!=set(after) or 'cuda_bytes' not in before:
            raise ValueError('CUDA budget migration requires matching resource fields')
        if any(before[k]!=after[k] for k in before if k!='cuda_bytes'):
            raise ValueError('CUDA budget migration cannot change other resource limits')
        for value in (before['cuda_bytes'],after['cuda_bytes']):
            if type(value) is not int or value<=0:raise ValueError('Positive integer CUDA byte limits required')
        previous={**previous,'resource_limits':dict(after)}
    old={k:v for k,v in previous.items() if k not in ('source','activation_storage')}
    new={k:v for k,v in current.items() if k not in ('source','activation_storage')}
    if old!=new:raise ValueError('Execution upgrade changes dataset/config/precision/resources/batch candidates')
    if previous.get('activation_storage','checkpointed') not in ('checkpointed','retained'):
        raise ValueError('Unknown saved activation storage')
    if current.get('activation_storage')!='retained':
        raise ValueError('Reviewed upgrade targets retained activations only')
    a=previous['source'];b=current['source']
    if not compatible_core(a['core'],b['core']):raise ValueError('Unreviewed core change')
    if a==b:return 'same_source_execution_policy_change'
    reviewed={'l0_regions/data.py','l0_regions/resident.py','l0_regions/training.py',
        'l0_regions/training_data.py','l0_regions/preparation_reuse.py','l0_regions/final.py',
        'l0_regions/execution_upgrade.py','tools/run_fixed_regions.py'}
    # f07b13f predates preparation overlap. Accept its helper and caller only
    # when CURRENT bytes match the immutable, already reviewed Git blobs.
    # Later edits to either file are not covered by this compatibility rule.
    pinned_additions=set()
    for name in ('l0_regions/preparation_runtime.py','l0_regions/preparation.py'):
        if name in b['runtime'] and a['runtime'].get(name)!=b['runtime'][name]:
            if b['runtime'][name]!=blob_hash('b398b3d8c0044d883729b789e83d69c0ed841725',name):
                raise ValueError('Preparation execution differs from reviewed Git blob')
            pinned_additions.add(name)
    if set(a['runtime'])-set(b['runtime']):raise ValueError('Removed runtime source')
    if any(a['runtime'].get(k)!=b['runtime'][k] for k in set(b['runtime'])-reviewed-pinned_additions):
        raise ValueError('Unreviewed model/loss/graph runtime change')
    for revision in REVISIONS:
        if a['runtime'].get('l0_regions/training.py')!=blob_hash(revision,'l0_regions/training.py'):continue
        if all(value==blob_hash(revision,name) for name,value in a['runtime'].items()):return revision
    raise ValueError('Saved runtime is not a reviewed release')
