"""Two actual CT observations: retained scheduler + canonical equivalence.

Explicit DEBUG regression only. No production cohort/config is rewritten.
"""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from hiercp_v222.v1_cache import configuration,load_verified
from hiercp_v22.storage import GraphWriter,load_record
from hiercp.preparation_runtime import run_case_jobs
from tools.v222_prepare_optimized import build_pair


def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main():
    root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
    cache=ROOT/'work/v222_v1_recovered2_training_20260924/cache'
    meta=json.loads((cache/'index_execution_r6_final.json').read_text())
    _,base=configuration();torch.set_num_threads(1)
    case_id=next(r['case_id'] for r in meta['records'] if r['target']==1)
    rows=[next(r for r in meta['records'] if r['case_id']==case_id and r['target']==target) for target in (1,0)]
    loaded=load_verified(next(r for r in meta['raw_records'] if r['case_id']==case_id))
    donors={}
    for row in rows:
        key=f"{row['donor_case_id']}:{row['donor_component']}"
        donors[key]=torch.load(cache/meta['donor_files'][key]['path'],weights_only=False,map_location='cpu',mmap=True)
    writer=GraphWriter(root)
    built=[]
    def build(row):
        key=f"{row['donor_case_id']}:{row['donor_component']}"
        return build_pair(row,loaded,donors[key],base,writer)
    run_case_jobs(tasks=rows,function=build,commit=built.append,workers='auto',report_path=root/'scheduler.json')
    assert len(built)==len(rows)==len({r['id'] for r in built})
    for row in built:
        old=next(r for r in rows if r['id']==row['id'])
        assert same(load_record(root,row['path']),load_record(cache,old['path'])),row['id']
        assert row['bounds']==old['bounds']
    result=dict(debug=True,real_CT=True,full_preparation=False,observations=len(built),
        canonical_CT_nodes_edges_bitwise_equal=True,calibration_results_retained=True,
        duplicate_tasks=0,rows=[r['id'] for r in built])
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
