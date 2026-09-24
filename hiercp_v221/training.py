"""Full-cohort episodic training with measured physical batching and CSV logs."""
from __future__ import annotations
from pathlib import Path
import copy
import csv
import time
import math
import gc
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from hiercp.preparation_runtime import snapshot
from hiercp.training_resources import host_input_budget
from .resources import _local_bound
from . import PIPELINE_VERSION
from .contracts import read_json,write_new,sha,safe_new_root,source_identity,validate_split
from .data import LocalDataset,collate
from .model import PromptGraphModel,prompt_loss
from .storage import load_record

def require_device(name):
    device=torch.device(name)
    if device.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('Production v2 training/scoring requires the configured CUDA environment; no CPU fallback')
    if torch.cuda.device_count()>1:
        raise RuntimeError('v2 single-device trainer: select an allocated device explicitly with CUDA_VISIBLE_DEVICES; multi-GPU training is not implemented')
    return device

def loader(dataset,batch,workers,shuffle=False):
    return DataLoader(dataset,batch_size=batch,shuffle=shuffle,num_workers=workers,collate_fn=collate,
                      pin_memory=True,persistent_workers=workers>0,**({'prefetch_factor':1} if workers else {}))

def candidates(n):
    out=[]; b=1
    while b<=n: out.append(b); b*=2
    if out[-1]!=n: out.append(n)
    return out

def full_inventory(root,rows):
    result=[]
    for row in rows:
        if 'bounds' in row:
            result.append({'id':row['id'],**row['bounds']}); continue
        r=load_record(root,row['path'])
        a,b=_local_bound(r['source_local']),_local_bound(r['target_local'])
        result.append({'id':row['id'],'nodes':a[0]+b[0],'edges':a[1]+b[1],
                       'bytes':a[2]+b[2]+r['source_patch'].numel()*4+r['target_patch'].numel()*4})
    return result

def calibrate(model,dataset,device,cfg,objective=None,*,inference_only=False):
    """Largest/mixed full graphs, full loss/backward when objective is available.

    Failed resource candidates never drop samples from the final epoch. Trials
    restore RNG and do not change parameters or optimizer moments.
    """
    inventory=full_inventory(dataset.root,dataset.rows)
    order=sorted(range(len(dataset)),key=lambda i:inventory[i]['bytes'],reverse=True)
    mixed=np.random.default_rng(cfg['seed']).permutation(len(dataset)).tolist()
    trials=[]; best=None
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    cpu_rng=torch.get_rng_state(); gpu_rng=torch.cuda.get_rng_state_all()
    parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
    try:
        for b in candidates(len(dataset)):
            cost=sum(inventory[i]['bytes'] for i in order[:b])
            accepted_trials=[t for t in trials if t['status']=='accepted']
            # Prevent Windows/WDDM allocation spill before an oversized trial
            # performs its full forward. Use two measured full-graph points,
            # retaining the fixed support/model memory in the intercept.
            if len(accepted_trials)>=2:
                previous,last=accepted_trials[-2:]
                slope=max(0.,(last['peak_reserved_estimate']-previous['peak_reserved_estimate'])/
                          max(1,last['input_cost']-previous['input_cost']))
                predicted=last['peak_reserved_estimate']+slope*(cost-last['input_cost'])
                free,total=torch.cuda.mem_get_info(device)
                capacity=min(total*cfg['max_vram_fraction'],free+torch.cuda.memory_allocated(device))
                if predicted>capacity:
                    trials.append(dict(batch=b,status='measured_trend_vram_admission_rejected',predicted_peak=predicted,
                                       capacity=capacity,basis_batches=[previous['batch'],last['batch']],input_cost=cost))
                    print({'calibration':'VRAM admission','batch':b,'predicted_peak':predicted,'capacity':capacity},flush=True)
                    break
            budget=host_input_budget(sum(sorted((r['bytes'] for r in inventory),reverse=True)[:b])*2,
                                     workers=0,prefetch_factor=1,pin_memory=True)
            if not budget['accepted']:
                trials.append({'batch':b,'status':'host_memory_rejected','host':budget}); continue
            started=time.perf_counter(); peak=0; count=0; accepted=True
            for chosen in (order,mixed):
                for _ in range(cfg['batch_calibration_repeats']):
                    model.zero_grad(set_to_none=True)
                    batch=encoded=loss=None
                    try:
                        torch.cuda.reset_peak_memory_stats(device)
                        batch=collate([dataset[i] for i in chosen[:b]]).to(device)
                        with torch.set_grad_enabled(objective is not None and not inference_only), torch.autocast('cuda',dtype=torch.bfloat16):
                            encoded=model.encode_local(batch)
                            if objective is not None:
                                loss=objective(encoded,batch.indices)
                            else:
                                # Inference sizing bootstrap only, never a training target.
                                loss=None
                        if loss is not None and not inference_only: loss.backward()
                        torch.cuda.synchronize(device)
                        # AdamW states are reserved even though calibration does not update weights.
                        peak=max(peak,torch.cuda.max_memory_allocated(device)+(2*parameter_bytes if objective and not inference_only else 0))
                        free,total=torch.cuda.mem_get_info(device)
                        capacity=min(total*cfg['max_vram_fraction'],free+torch.cuda.memory_allocated(device))
                        if peak>capacity: accepted=False
                        count+=b
                    except torch.cuda.OutOfMemoryError:
                        accepted=False
                    finally:
                        batch=encoded=loss=None
                        if not accepted:
                            model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
                    if not accepted: break
                if not accepted: break
            elapsed=time.perf_counter()-started
            trial={'batch':b,'status':'accepted' if accepted else 'gpu_memory_rejected','peak_reserved_estimate':peak,
                   'input_cost':cost,'seconds':elapsed,
                   'graphs_per_second':count/max(elapsed,1e-9),'host':budget,'objective':('full_prompt_inference' if inference_only else 'actual_cross_patient_transfer_and_context') if objective else 'encoder_inference_bootstrap'}
            trials.append(trial)
            print({'calibration':'physical_batch',**{k:trial[k] for k in ('batch','status','seconds','peak_reserved_estimate','graphs_per_second')}},flush=True)
            if accepted and (best is None or trial['graphs_per_second']>best['graphs_per_second']): best=trial
            model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            # Calibration trials are not dataset traversal. Largest-graph
            # prefixes only grow; after a measured GPU rejection, bigger
            # physical batches cannot meet the same memory bound.
            if not accepted:break
        if best is None: raise RuntimeError('No full-scale batch fits; graphs/model were not reduced')
    finally:
        model.load_state_dict(state); torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(gpu_rng)
        model.zero_grad(set_to_none=True)
    return best['batch'],{'trials':trials,'selected_batch':best['batch'],'inventory':inventory,'resources':snapshot()}

def calibrate_workers(dataset,batch,cfg):
    inventory=full_inventory(dataset.root,dataset.rows); reports=[]
    # Calibration only: compare workers on identical full-size graphs, covering
    # every patient, each patient's largest graph, and seeded mixed batches.
    # Final memory refresh, SGD and validation still traverse ALL dataset rows.
    largest={}
    for i,row in enumerate(dataset.rows):
        case=row['case_id']
        if case not in largest or inventory[i]['bytes']>inventory[largest[case]]['bytes']:largest[case]=i
    mixed=np.random.default_rng(cfg['seed']).permutation(len(dataset))[:min(len(dataset),max(len(largest),2*batch*cfg['batch_calibration_repeats']))]
    indices=sorted(set(largest.values())|set(map(int,mixed)))
    probe=Subset(dataset,indices)
    for workers in cfg['worker_candidates']:
        budget=host_input_budget(sum(sorted((r['bytes'] for r in inventory),reverse=True)[:batch])*2,
                                 workers=workers,prefetch_factor=1,pin_memory=True)
        if not budget['accepted']:
            reports.append({'workers':workers,'status':'host_memory_rejected','host':budget}); continue
        start=time.perf_counter(); count=0
        for value in loader(probe,batch,workers): count+=len(value.indices)
        reports.append({'workers':workers,'status':'accepted','graphs_per_second':count/(time.perf_counter()-start),'host':budget,
            'scope':'worker timing only; full-patient coverage/largest graphs/mixed physical batches',
            'calibration_indices':indices,'dataset_graphs':len(dataset),'measured_graphs':count,'final_dataset_unchanged':True})
        print({'calibration':'loader_workers','workers':workers,'graphs':count,'seconds':time.perf_counter()-start,
               'graphs_per_second':reports[-1]['graphs_per_second']},flush=True)
    good=[r for r in reports if r['status']=='accepted']
    if not good: raise RuntimeError('No loader concurrency fits measured host memory')
    return max(good,key=lambda r:r['graphs_per_second'])['workers'],reports

@torch.no_grad()
def build_memory(model,dataset,batch,workers,device,case_ids,donor_ids):
    model.eval(); values=[]
    for item in loader(dataset,batch,workers):
        item=item.to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            values.append(model.encode_local(item).float().cpu())
    lookup={c:i for i,c in enumerate(case_ids)}
    owners=torch.tensor([lookup[r['case_id']] for r in dataset.rows],device=device)
    evidence=torch.tensor([r['evidence'] if r['case_id'] in donor_ids else -1 for r in dataset.rows],device=device)
    return {'case_ids':list(case_ids),'owners':owners,'embeddings':torch.cat(values).to(device),
            'evidence':evidence,
            'donor_allowed':torch.tensor([c in donor_ids for c in case_ids],device=device)}

def attach_context(memory,embeddings_,case_id):
    """Add/replace the target patient's unlabeled context without adding its T evidence."""
    if case_id in memory['case_ids']:
        index=memory['case_ids'].index(case_id)
        keep=memory['owners']!=index
        case_ids=list(memory['case_ids'])
        owners=memory['owners'][keep]
        allowed=memory['donor_allowed'].clone(); allowed[index]=False
    else:
        index=len(memory['case_ids']); case_ids=[*memory['case_ids'],case_id]
        keep=torch.ones(len(memory['owners']),dtype=torch.bool,device=memory['owners'].device)
        owners=memory['owners']; allowed=torch.cat((memory['donor_allowed'],torch.tensor([False],device=owners.device)))
    result=dict(case_ids=case_ids,owners=torch.cat((owners,owners.new_full((len(embeddings_),),index))),
        embeddings=torch.cat((memory['embeddings'][keep],embeddings_)),
        evidence=torch.cat((memory['evidence'][keep],owners.new_full((len(embeddings_),),-1))),
        donor_allowed=allowed)
    return result,index

def train(cache_path,output,cfg,base):
    from .contracts import require_training_objective
    require_training_objective()
