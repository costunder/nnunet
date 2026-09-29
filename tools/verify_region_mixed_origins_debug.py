"""Build a DEBUG index over unchanged real items from two reviewed cache origins.

Reproduce the old mixed-source batch rejection, verify guarded collation and
integrated retained training/resume. Never regenerate partitions or relabel items.
"""
import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('first','second','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    import torch
    from l0_regions.training_data import RegionDataset,sha
    from l0_regions.preparation_reuse import verified_binding_sources
    from l0_regions.data import collate
    from l0_regions.resident import load
    first=RegionDataset(a.first,'inner_train',True,'research-report')
    second=RegionDataset(a.second,'inner_train',True,'research-report')
    for key in ('config','base','profile','view_epoch','cnn_sha256','original_cache_sha256'):
        if first.meta[key]!=second.meta[key]:raise ValueError('Different test cache: '+key)
    if first.rows!=second.rows:raise ValueError('Different test cohort/order')
    a.output.mkdir(parents=True,exist_ok=False)
    folder=a.output/'cache';folder.mkdir()
    meta=copy.deepcopy(first.meta)
    proof=meta['reused_preparation']
    proof['source_origins']=proof.get('source_origins',[proof['source_identity']])+[second.meta['source_identity']]
    before={}
    for part in meta['partitions']:
        left=first.meta['partitions'][part];right=second.meta['partitions'][part]
        if [v['row'] for v in left]!=[v['row'] for v in right]:raise ValueError('Different held-out cohort')
        meta['partitions'][part]=[]
        for i in range(len(left)):
            ds=first if i%2==0 else second
            entry=copy.deepcopy(ds.meta['partitions'][part][i])
            path=(ds.root/entry['file']).resolve();before[str(path)]=sha(path)
            entry['file']=str(path);meta['partitions'][part].append(entry)
    meta['diagnostic_mixed_origin_fixture']=True
    shutil.copyfile(first.root/'frozen_cnn.pt',folder/'frozen_cnn.pt')
    index=folder/'index.json';index.write_text(json.dumps(meta),encoding='utf-8')
    ds=RegionDataset(index,'inner_train',True,'research-report')
    paths,bindings,ids=ds.request([0,1]);items=[load(path,binding) for path,binding in zip(paths,bindings)]
    if len({it['binding']['preparation_source_sha256'] for it in items})!=2:
        raise ValueError('Fixture does not exercise heterogeneous sources')
    try:collate(items,ids)
    except ValueError as exc:
        if 'unverified preparation sources' not in str(exc):raise
    else:raise AssertionError('Unverified mixed origins must fail')
    batch=collate(items,ids,compatible_sources=ds.compatible_sources)
    if batch.bindings!=bindings:raise AssertionError('Historical bindings relabelled')
    try:collate(items,ids,compatible_sources=frozenset())
    except ValueError:pass
    else:raise AssertionError('Unknown source accepted')
    bad=copy.deepcopy(meta);bad['partitions']['inner_train'][0]['binding']['preparation_source_sha256']='unreviewed'
    from l0_regions.training_data import source_identity
    try:verified_binding_sources(bad,source_identity(preparation=True))
    except ValueError:pass
    else:raise AssertionError('Unreviewed index source accepted')
    subprocess.run([sys.executable,'-u',str(ROOT/'tools/verify_region_support_resume_debug.py'),
        '--cache',str(index),'--output',str(a.output/'retained_smoke'),
        '--activation-storage','retained'],cwd=ROOT,check=True)
    if any(sha(path)!=digest for path,digest in before.items()):raise AssertionError('Original tensor changed')
    result=dict(debug=True,actual_CT=True,status='PASS',partition_rebuilds=0,
        actual_mixed_sources=2,original_bindings_preserved=True,original_tensors_unchanged=True,
        unverified_mix_rejected=True,unknown_source_rejected=True,retained_fresh_process_resume=True,
        full_training=False,server_validation=False)
    (a.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
