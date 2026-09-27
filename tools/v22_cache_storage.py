"""Read-only size audit and explicit full-cache storage admission (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(index):
    index=Path(index).resolve();meta=json.loads(index.read_text(encoding='utf-8-sig'))
    categories={'graphs':set(),'shared_sources':set(),'donors':set()}
    for row in meta['records']:
        categories['graphs'].add(row['path'])
        categories['shared_sources'].add(row['shared_source']['path'])
    for row in meta['donor_files'].values():categories['donors'].add(row['path'])
    report={}
    for name,paths in categories.items():
        sizes=[]
        for relative in paths:
            p=(index.parent/relative).resolve()
            if not p.is_relative_to(index.parent):raise ValueError('Storage reference escapes cache')
            if not p.is_file():raise ValueError(f'Missing storage reference: {p}')
            sizes.append(p.stat().st_size)
        if not sizes:raise ValueError(f'Empty storage category: {name}')
        report[name]=dict(files=len(sizes),bytes=sum(sizes),maximum_bytes=max(sizes))
    return dict(reference_index=str(index),reference_index_sha256=digest(index),categories=report,
        referenced_logical_bytes=sum(r['bytes'] for r in report.values()),
        payload_hashes_validated=False,compatible_training_cache_claimed=False)


def make_plan(reference,index,output):
    audit=inventory(reference);index=Path(index).resolve();meta=json.loads(index.read_text(encoding='utf-8-sig'))
    counts=dict(graphs=len(meta['records']),shared_sources=len(meta['donor_pool']),donors=len(meta['donor_pool']))
    estimate=sum(audit['categories'][k]['maximum_bytes']*n for k,n in counts.items())
    return dict(format='v22_storage_admission_v1',source_index=str(index),source_index_sha256=digest(index),
        output=str(Path(output).resolve()),counts=counts,reference=audit,
        projected_payload_bytes=estimate,estimate_method='reference category maximum times complete target counts',
        guaranteed_upper_bound=False,includes_raw_CT=False,includes_checkpoints=False,
        limitation='Empirical storage estimate, not a guarantee for changed geometry; disk reserve is also checked while writing.')


def admit(plan,index,output,counts,reserve_bytes):
    if plan.get('format')!='v22_storage_admission_v1' or plan['source_index_sha256']!=digest(index):
        raise ValueError('Storage plan absent, wrong format or bound to another observation index')
    if plan['output']!=str(Path(output).resolve()) or plan['counts']!=counts:
        raise ValueError('Storage plan targets another output/cohort')
    estimate=plan['projected_payload_bytes']
    expected=sum(plan['reference']['categories'][k]['maximum_bytes']*n for k,n in counts.items())
    if type(estimate)!=int or estimate<=0 or estimate!=expected:
        raise ValueError('Invalid storage projection')
    parent=Path(output).resolve().parent
    while not parent.exists():parent=parent.parent
    free=shutil.disk_usage(parent).free
    if free<=estimate+reserve_bytes:
        raise OSError(f'Full cache storage not admitted: free={free}, projected={estimate}, reserve={reserve_bytes}; no cache created')
    return dict(free_before_bytes=free,projected_payload_bytes=estimate,reserved_free_bytes=reserve_bytes,
                empirical_estimate=True,full_cohort_count=counts['graphs'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-cache',required=True)
    parser.add_argument('--index',help='Full observation index for a preparation plan')
    parser.add_argument('--output',help='Intended new cache directory; not created by this audit')
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    if bool(args.index)!=bool(args.output):parser.error('--index and --output must be supplied together')
    report=make_plan(args.reference_cache,args.index,args.output) if args.index else inventory(args.reference_cache)
    with Path(args.report).open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)
