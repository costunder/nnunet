"""Read-only full-cohort anchor audit plus bounded real-CT GPU regression.

No training/checkpoint output. Detailed case data stays in the requested local
report; publish only aggregate summaries. Observation and mask geometry is fixed.
"""
import argparse,copy,json,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm
from l0_regions.donor_data import assignment
from l0_local_cnn.data import CropStore
from l0_local_cnn.model import LocalCNN
from l0_exploration.data import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--records',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,required=True)
    p.add_argument('--cuda-gib',type=float,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.workers<2 or not torch.cuda.is_available():raise ValueError('Parallel readers and CUDA required')
    torch.set_num_threads(a.workers)
    torch.cuda.set_per_process_memory_fraction(a.cuda_gib*2**30/torch.cuda.get_device_properties(0).total_memory)
    m=json.loads(a.metadata.read_text());m['records']=json.loads(a.records.read_text())
    rows=assignment(m,m['config']['seed']);by={}
    for r in rows:by.setdefault(r['case_id'],[]).append(r)
    assert set(by)==set(m['split']['outer_train'])
    def audit(raw):
        if sha(raw['label'])!=raw['label_sha256']:raise ValueError('Annotation hash changed')
        label=np.asarray(nib.load(raw['label']).dataobj)
        counts=Counter();outside=[];known={x['component']:x['center'] for x in raw['positives']}
        for row in by[raw['case_id']]:
            center=np.asarray(row['center'])
            if not np.isfinite(center).all() or not np.equal(center,np.round(center)).all() or ((center<0)|(center>=label.shape)).any():raise ValueError('Bad recorded center')
            v=int(label[tuple(center.astype(int))]);counts[f'target{row["target"]}_label{v}']+=1
            if row['target']==0 and v!=1:raise ValueError('Comparison is not liver')
            if row['target']==1 and not np.array_equal(known[row['component']],center):raise ValueError('Observed anchor binding changed')
            if v==0:outside.append(row['id'])
        return dict(counts=dict(counts),outside=outside)
    raw=[r for r in m['raw_records'] if r['case_id'] in by]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        audits=list(tqdm(pool.map(audit,raw),total=len(raw),desc='DEBUG full observation anchor audit'))
    total=Counter()
    for result in audits:total.update(result['counts'])
    # The exact 30th physical batch from the reported calibration failure.
    train=[r for r in rows if r['case_id'] in m['split']['inner_train']]
    selected=train[29*32:30*32]
    if len(selected)!=32:raise ValueError('Expected full reported physical batch')
    original=copy.deepcopy(selected);profiles=[];raw_cache=None
    for margin in (10,20,30):
        cfg=json.loads((ROOT/'config/v22_local_cnn.json').read_text());cfg['margin_mm']=margin;m['local_cnn']=cfg
        store=CropStore(m,a.workers,12*2**30,32*2**30)
        if raw_cache is not None:store.raw.cache=raw_cache
        batch=store.batch(selected,list(range(32)));raw_cache=store.raw.cache
        assert selected==original
        if torch.count_nonzero(batch.images[~batch.organ]):raise AssertionError('External CT leaked into crop')
        net=LocalCNN(cfg,dropout=0).cuda().eval();gpu=batch.to('cuda')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.no_grad():out=net(gpu)
        torch.cuda.synchronize()
        assert out.shape==(32,128) and torch.isfinite(out).all()
        profiles.append(dict(margin_mm=margin,physical_batch=32,input_shape=list(gpu.images.shape),
            background_anchor_crops=sum(not c['anchor_in_organ'] for c in batch.audit),
            forward_seconds=time.perf_counter()-start,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30))
        # Short gradient regression on an actual pair whose donor anchor is background.
        bad=next(r for r in selected if not store.raw.cache[r['donor_case_id']]['organ'][tuple(store.donor_bounds(r)[0])])
        del out,gpu,batch
        q=store.batch([bad],[0]).to('cuda');q.images.requires_grad_(True)
        opt=torch.optim.AdamW(net.parameters(),lr=1e-4);before=next(net.parameters()).detach().clone()
        loss=net(q).square().mean();loss.backward()
        if torch.count_nonzero(q.images.grad[~q.organ]):raise AssertionError('External gradient')
        if not all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters()):raise AssertionError('Disconnected gradient')
        opt.step()
        if torch.equal(before,next(net.parameters()).detach()):raise AssertionError('Optimizer did not update CNN')
        profiles[-1]['actual_pair_backward_optimizer']=True
        del net,q,opt,loss,before
        torch.cuda.empty_cache()
    report=dict(debug=True,full_training=False,full_evaluation=False,scope='Full observation anchor audit, reported 32-pair batch L0 forwards, one actual background-anchor pair gradient per FOV',
        cases=len(raw),records=len(rows),counts=dict(total),outside_observed_ids=[r for x in audits for r in x['outside']],
        gpu=torch.cuda.get_device_name(),profiles=profiles)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='outside_observed_ids'},indent=2))

if __name__=='__main__':main()
