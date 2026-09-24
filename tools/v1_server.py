"""Code-only deployment: inspect resources and rebuild observations from server CT.

No transferred graph cache, SSH changes, installation or automatic training.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import psutil
import torch
from hiercp_v222.v1_cache import configuration,provenance,assign_pairs
from hiercp_v222.contracts import (read_json,write_new,sha,validate_identities,
    safe_new_root,CASE_BENCHMARK_IDENTITY)


def observation_metadata(inventory,split,identities,cfg):
    """Reuse the established raw-label anchor and seed42 comparison definition."""
    validate_identities(identities,split)
    if set(inventory)!=set(split['outer_train']+split['outer_val']):
        raise ValueError('Complete raw cohort inventory required')
    rows=[]
    for c in split['outer_train']:
        info=inventory[c]
        centers=[(r['center'],1,r['component']) for r in info['positives']]
        centers += [(point,0,None) for point in info['comparison']['centers']]
        for i,(center,target,component) in enumerate(centers):
            rows.append(dict(id=f'{c}:{i}',case_id=c,patient_group=identities['cases'][c]['patient_group'],
                             component=component,center=center,target=target))
    pool=[dict(case_id=c,component_id=r['component']) for c in sorted(split['inner_train']) for r in inventory[c]['positives']]
    meta=dict(format='v222_paired_observation_metadata_v1',complete=True,debug=False,
        split=split,identities=identities,records=sorted(rows,key=lambda r:r['id']),donor_pool=pool,
        raw_records=[inventory[c] for c in split['outer_train']],config=cfg,
        source_identity=provenance(),raw_CT_rebuilt=True,graph_cache_transferred=False)
    assign_pairs(meta,cfg)
    return meta


def observations(medical,split_path,output):
    from hiercp.common import discover_cases
    from hiercp_v222.data import inspect_case,assert_no_duplicate_ct
    from hiercp_v222.parallel import run_jobs
    from hiercp_v22.volumes import volume_memory_bound
    cfg,_=configuration();split=read_json(split_path)
    approved=read_json(ROOT/'config/split_cp80_fold0.json')
    if split!=approved:raise ValueError('Split must match the committed local CP80 comparison split')
    paths={p.case_id:p for p in discover_cases(Path(medical).resolve()/'Data')}
    if set(paths)!=set(split['outer_train']+split['outer_val']):
        raise ValueError('Expected the full original 131-case cohort in Data/image and Data/labels')
    identities=dict(format=CASE_BENCHMARK_IDENTITY,independence_scope='published_case_only',
        patient_independence_verified=False,annotation_scope='provided_masks_may_omit_lesions',
        cases={c:dict(patient_group='case:'+c,identity_basis='published_case_id_only',annotation_complete=None) for c in paths})
    validate_identities(identities,split);root=safe_new_root(output);inventory={}
    write_new(root/'started.json',dict(config=cfg,split=split,identities=identities,
        medical_root=str(Path(medical).resolve()),source_identity=provenance(),debug=False,subset=False))
    torch.set_num_threads(1)
    def scan(c):
        return c,inspect_case(paths[c],cfg['donor_max_diameter_mm'],cfg if c in split['outer_train'] else None)
    def commit(pair):
        inventory[pair[0]]=pair[1]
        print(json.dumps(dict(stage='raw_inventory',completed=len(inventory),total=len(paths))),flush=True)
    run_jobs(list(paths),scan,commit,cfg['preparation_workers'],root/'resources.json',
        memory_per_job=max(volume_memory_bound(p.image_path) for p in paths.values()),benchmark_function=scan)
    assert_no_duplicate_ct(inventory,split)
    write_new(root/'inventory.json',inventory)
    meta=observation_metadata(inventory,split,identities,cfg)
    meta['split_sha256']=sha(split_path)
    write_new(root/'index.json',meta)
    print(json.dumps(dict(index=str(root/'index.json'),raw_cases=len(inventory),observations=len(meta['records']),
                         donors=len(meta['donor_pool']),graph_tensors_created=False)),flush=True)


def resources(output):
    cfg,_=configuration()
    versions={p:importlib.metadata.version(p) for p in ('torch','torch-geometric','numpy','scipy','nibabel','psutil','scikit-learn')}
    gpus=[]
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i);free,total=torch.cuda.mem_get_info(i)
        gpus.append(dict(index=i,name=p.name,capability=list(torch.cuda.get_device_capability(i)),total=total,free=free))
    result=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv'],capture_output=True,text=True,check=True)
    report=dict(platform=platform.platform(),python=sys.version,packages=versions,cuda_build=torch.version.cuda,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpus=gpus,nvidia_smi=result.stdout,
        cpu_physical=psutil.cpu_count(logical=False),cpu_logical=psutil.cpu_count(),ram=psutil.virtual_memory()._asdict(),
        disk=shutil.disk_usage(ROOT)._asdict(),source_identity=provenance(),config=cfg,training_started=False)
    write_new(output,report);print(json.dumps({k:v for k,v in report.items() if k not in ('source_identity','config')},indent=2))
    if not gpus:raise RuntimeError('No allocated CUDA GPU; no CPU fallback')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('resources');r.add_argument('--output',required=True)
    o=sub.add_parser('observations');o.add_argument('--medical-root',required=True)
    o.add_argument('--split',default=str(ROOT/'config/split_cp80_fold0.json'));o.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='resources':resources(a.output)
    else:observations(a.medical_root,a.split,a.output)
if __name__=='__main__':main()
