"""Short real-pair L0 compute diagnostic. Never starts GNN/nnU-Net training.

Both regs, record IDs, physical batch, workers and repeats are explicit.
An admission rejection is a measured failure, never an automatic repair.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import psutil
import torch
from l0_ezsp.config import load_profile
from l0_ezsp.data import load_pairs
from l0_ezsp.encoder import EZSPEncoder
from l0_ezsp.partition import CoarseningConstraintError
from hiercp_v222.v1_cache import configuration,provenance
from hiercp_v222.v1_local import V1LocalEncoder
from tools.v222_review_contracts import installed

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def timed(fn):
    torch.cuda.synchronize();start=perf_counter();value=fn();torch.cuda.synchronize()
    return value,perf_counter()-start

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--debug-cache',action='store_true')
    p.add_argument('--indices',type=int,nargs='+',required=True)
    p.add_argument('--physical-batch',type=int,required=True)
    p.add_argument('--workers',type=int,required=True)
    p.add_argument('--repeats',type=int,required=True)
    p.add_argument('--reg-scale1',type=float,required=True)
    p.add_argument('--reg-scale2',type=float,required=True)
    p.add_argument('--sharding',type=int)
    p.add_argument('--baseline',action='store_true',help='Measure the same full fine pair batch separately')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    if a.physical_batch!=len(a.indices) or a.repeats<1 or a.workers<1 or len(set(a.indices))!=len(a.indices):
        raise ValueError('Explicit unique indices must equal physical batch; positive workers/repeats required')
    profile=load_profile(reg_scale1=a.reg_scale1,reg_scale2=a.reg_scale2,sharding=a.sharding)
    a.output.mkdir(parents=True,exist_ok=False)
    cfg,base=configuration();torch.set_num_threads(min(a.workers,os.cpu_count() or a.workers))
    if a.debug_cache:
        from tools.v22_debug_profile import RepairDataset as Dataset
    else:
        from tools.v222_runtime_cache import CachedPairDataset as Dataset
    dataset=Dataset(a.cache,'inner_train')
    if any(i<0 or i>=len(dataset.rows) for i in a.indices):raise ValueError('Record index outside validated cache')
    device=torch.cuda.get_device_properties(0)
    report={'debug':True,'training_ready':False,'full_training':False,'full_evaluation':False,
        'scope':'real-pair L0 compute only; scalar squared-output diagnostic loss, not observation rank training',
        'weights':'new seed42 initialization; not a trained feature/reg calibration',
        'profile':profile,'cache':str(a.cache.resolve()),'cache_sha256':sha(a.cache),
        'core_source_identity':provenance(),
        'adapter_source_identity':{str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'l0_ezsp').rglob('*') if f.suffix in ('.py','.json')},
        'gpu':device.name,'total_vram':device.total_memory,'free_vram_start':torch.cuda.mem_get_info()[0],
        'torch':torch.__version__,'cpu_physical':psutil.cpu_count(logical=False),'available_ram':psutil.virtual_memory().available,
        'physical_batch':a.physical_batch,'effective_batch':a.physical_batch,'gradient_accumulation':1,
        'workers':a.workers,'precision':'float32','indices':a.indices,'dataset_records':len(dataset.rows),
        'used_fraction':len(a.indices)/len(dataset.rows),'repeats':a.repeats,
        'unmeasured':['full support/L1/L2 update','production rank loss','support refresh','validation','checkpoint/hash/durable write','epoch duration','MIG 10GB physical32 admission'],
        'modes':{}}
    start=perf_counter();host=load_pairs(dataset,a.indices,workers=a.workers).pin_memory()
    report['loader_graph_seconds']=perf_counter()-start
    batch,report['h2d_seconds']=timed(lambda:host.to('cuda'))
    report['fine_nodes']={k:batch.graph[k].num_nodes for k in batch.graph.node_types}
    report['fine_edges']={'|'.join(e):batch.graph[e].edge_index.shape[1] for e in batch.graph.edge_types}
    report['input_shapes']={'source':list(batch.source_patches.shape),'target':list(batch.target_patches.shape)}
    rejected=False
    try:
        for mode in (['fine_baseline','ezsp'] if a.baseline else ['ezsp']):
            torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            if mode=='ezsp':net=EZSPEncoder(base,profile).cuda().train();net.timing_enabled=True
            else:
                with installed('stride4'):net=V1LocalEncoder(base).cuda().train()
            optimizer=torch.optim.Adam(net.parameters(),lr=1e-4)
            result={'parameters':sum(p.numel() for p in net.parameters()),'trainable_parameters':sum(p.numel() for p in net.parameters() if p.requires_grad),'runs':[]}
            report['modes'][mode]=result
            for repeat in range(a.repeats):
                optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
                start=perf_counter()
                try:
                    output,forward=timed(lambda:net(batch))
                    loss=output.float().square().mean()
                    _,backward=timed(loss.backward)
                    _,step=timed(optimizer.step)
                    row={'repeat':repeat,'forward_seconds':forward,'backward_seconds':backward,'optimizer_seconds':step,
                         'update_seconds':perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                         'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'loss':float(loss.detach()),'status':'DEBUG_compute_completed'}
                    if mode=='ezsp':row['stages']=net.last_audit
                    result['runs'].append(row)
                    print(json.dumps({'mode':mode,**{k:v for k,v in row.items() if k!='stages'}}),flush=True)
                    del output,loss
                except CoarseningConstraintError as exc:
                    rejected=True
                    result['runs'].append({'repeat':repeat,'status':'admission_rejected','error':str(exc),
                        'seconds_to_rejection':perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'stages':net.last_audit,'backward_seconds':None,'optimizer_seconds':None})
                    print(json.dumps({'mode':mode,'status':'admission_rejected','stats':exc.stats}),flush=True)
                    break  # Failed diagnostic, not a skipped training record.
            del net,optimizer
            gc.collect();torch.cuda.empty_cache()
        report['status']='admission_rejected' if rejected else 'DEBUG_compute_completed_not_production_admitted'
    except Exception as exc:
        report['status']='error';report['error']={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        report['rss_bytes']=psutil.Process().memory_info().rss
        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print('REPORT:',a.output/'report.json',flush=True)
    return 2 if rejected else 0

if __name__=='__main__':raise SystemExit(main())
