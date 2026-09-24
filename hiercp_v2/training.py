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
from hiercp.training_resources import _local_bound,host_input_budget
from . import PIPELINE_VERSION
from .contracts import read_json,write_new,sha,safe_new_root,source_identity,validate_split
from .data import LocalDataset,collate
from .model import PromptGraphModel,prompt_loss,context_descriptor
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
    model.eval(); values=[]; descriptors=[]
    for item in loader(dataset,batch,workers):
        item=item.to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            values.append(model.encode_local(item).float().cpu())
        descriptors.append(context_descriptor(item).cpu())
    lookup={c:i for i,c in enumerate(case_ids)}
    owners=torch.tensor([lookup[r['case_id']] for r in dataset.rows],device=device)
    evidence=torch.tensor([r['evidence'] if r['case_id'] in donor_ids else -1 for r in dataset.rows],device=device)
    return {'case_ids':list(case_ids),'owners':owners,'embeddings':torch.cat(values).to(device),
            'descriptors':torch.cat(descriptors).to(device),'evidence':evidence,
            'donor_allowed':torch.tensor([c in donor_ids for c in case_ids],device=device)}

def attach_context(memory,embeddings_,descriptors,case_id):
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
        descriptors=torch.cat((memory['descriptors'][keep],descriptors)),
        evidence=torch.cat((memory['evidence'][keep],owners.new_full((len(embeddings_),),-1))),
        donor_allowed=allowed)
    return result,index

def train(cache_path,output,cfg,base):
    device=require_device(cfg['training_device']); root=Path(cache_path).resolve().parent; index=read_json(cache_path)
    if index['format']!=PIPELINE_VERSION or not index['complete'] or index['config']!=cfg or index['base_config']!=base:
        raise ValueError('Cache/config/version mismatch; rebuild view-only caches')
    validate_split(index['split'])
    from .donors import validate_training_rows,reject_cross_split_duplicates
    validate_training_rows(index['records'],index['split'],index['donor_pool'])
    reject_cross_split_duplicates(index['all_raw_identities'],index['split'])
    tr=[r for r in index['records'] if r['case_id'] in index['split']['inner_train']]
    va=[r for r in index['records'] if r['case_id'] in index['split']['inner_val']]
    case_ids=list(index['split']['inner_train'])
    if set(r['case_id'] for r in tr)!=set(case_ids) or set(r['case_id'] for r in va)!=set(index['split']['inner_val']):
        raise ValueError('Every actual patient, including U-only patients, needs context records')
    if len({r['case_id'] for r in tr if r['evidence']==1})<2:
        raise ValueError('At least two observed-positive training patients required')
    out=safe_new_root(output); torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    model=PromptGraphModel(cfg,base,patient_ids=case_ids).to(device)
    dataset=LocalDataset(root,tr); validation=LocalDataset(root,va)
    model.eval(); bootstrap,bootstrap_report=calibrate(model,dataset,device,cfg)
    workers,worker_report=calibrate_workers(dataset,bootstrap,cfg)
    def make_memory(epoch):
        return build_memory(model,LocalDataset(root,tr,epoch=epoch,verify=False),bootstrap,workers,device,case_ids,set(case_ids))
    memory=make_memory(0)
    def objective(encoded,indices):
        output_=model.forward_tasks(memory,encoded,memory['owners'][indices])
        return prompt_loss(output_,memory['descriptors'][indices],memory['evidence'][indices],
                           cfg['loss_weights'],cfg['temperature'])[0]
    model.train(); batch,report=calibrate(model,dataset,device,cfg,objective)
    workers,worker_report=calibrate_workers(dataset,batch,cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'])
    write_new(out/'execution.json',{'format':PIPELINE_VERSION,'config':cfg,'base_config':base,'bootstrap':bootstrap_report,
        'calibration':report,'workers':worker_report,'selected_workers':workers,'physical_graph_batch':batch,
        'effective_graph_batch':batch,'gradient_accumulation':1,'parameters':sum(p.numel() for p in model.parameters()),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'training_graphs':len(tr),'validation_graphs':len(va),'patients':case_ids,
        'T_records':sum(r['evidence']==1 for r in tr),'U_records':sum(r['evidence']==-1 for r in tr),
        'F_contract':'hard geometry gate; normal liver is never F','source_identity':source_identity(),
        'gpu':torch.cuda.get_device_name(device),'device_count':torch.cuda.device_count(),'debug':False,
        'task_definition':'patient ID, not views','validation_interpretation':'compatibility/structural diagnostics, not medical accuracy'})
    total_steps=0
    with (out/'metrics.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','train_loss','validation_loss','graphs','steps','peak_vram','seconds','graphs_per_second'])
        writer.writeheader()
        for epoch in range(cfg['gnn_epochs']):
            if epoch: memory=make_memory(epoch)
            model.train(); total=torch.zeros((),device=device); seen=0; steps=0; start=time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            for item in loader(LocalDataset(root,tr,epoch=epoch,verify=False),batch,workers):
                item=item.to(device); ids=item.indices; optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16): loss=objective(model.encode_local(item),ids)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),base['training']['grad_clip'],error_if_nonfinite=True)
                optimizer.step(); total+=loss.detach()*len(ids); seen+=len(ids); steps+=1
            if seen!=len(tr): raise RuntimeError('Training omitted graphs')
            del item,loss
            optimizer.zero_grad(set_to_none=True)
            # All held-out tasks are batched together with U-only local evidence.
            validation_memory=build_memory(model,validation,bootstrap,workers,device,index['split']['inner_val'],set())
            offset=len(memory['case_ids'])
            joined=dict(case_ids=memory['case_ids']+validation_memory['case_ids'],
                owners=torch.cat((memory['owners'],validation_memory['owners']+offset)),
                embeddings=torch.cat((memory['embeddings'],validation_memory['embeddings'])),
                descriptors=torch.cat((memory['descriptors'],validation_memory['descriptors'])),
                evidence=torch.cat((memory['evidence'],validation_memory['evidence'])),
                donor_allowed=torch.cat((memory['donor_allowed'],validation_memory['donor_allowed'])))
            model.eval(); val=torch.zeros((),device=device); nv=0
            observed_validation=torch.tensor([r['evidence'] for r in va],device=device)
            validation_state=None
            def validation_objective(encoded,ids):
                result=model.forward_tasks(joined,encoded,validation_memory['owners'][ids]+offset,state=validation_state)
                observed=observed_validation[ids]
                return prompt_loss(result,validation_memory['descriptors'][ids],observed,cfg['loss_weights'],cfg['temperature'])[0]
            if epoch==0:
                validation_batch,validation_resources=calibrate(model,validation,device,cfg,validation_objective,inference_only=True)
                write_new(out/'validation_resources.json',validation_resources)
            with torch.no_grad():
                with torch.autocast('cuda',dtype=torch.bfloat16): validation_state=model.prepare_task_state(joined)
                for begin in range(0,len(va),validation_batch):
                    ids=torch.arange(begin,min(begin+validation_batch,len(va)),device=device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=validation_objective(validation_memory['embeddings'][ids],ids)
                    val+=loss*len(ids); nv+=len(ids)
            if nv!=len(va): raise RuntimeError('Validation omitted graphs')
            validation_state=None
            del joined,validation_memory,observed_validation
            total_steps+=steps; elapsed=time.perf_counter()-start
            row=dict(epoch=epoch+1,train_loss=float(total/seen),validation_loss=float(val/nv),graphs=seen,steps=steps,
                peak_vram=torch.cuda.max_memory_allocated(device),seconds=elapsed,graphs_per_second=seen/elapsed)
            writer.writerow(row); f.flush(); print(row,flush=True)
            payload={'format':PIPELINE_VERSION,'complete':False,'epoch':epoch+1,'total_steps':total_steps,
                'config':cfg,'base_config':base,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                'patient_ids':case_ids,'split':index['split'],'raw_records':index['cases'],'donor_pool':index['donor_pool'],'cache_sha256':sha(cache_path),
                'source_identity':source_identity(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'physical_batch':batch,'workers':workers}
            with (out/f'epoch_{epoch+1:03d}.pt').open('xb') as handle: torch.save(payload,handle)
    memory=make_memory(cfg['gnn_epochs'])
    payload['memory']={k:v.cpu() if torch.is_tensor(v) else v for k,v in memory.items()}; payload['complete']=True
    with (out/'model.pt').open('xb') as handle: torch.save(payload,handle)
    write_new(out/'completion.json',{'format':PIPELINE_VERSION,'epochs':cfg['gnn_epochs'],'checkpoint_sha256':sha(out/'model.pt'),
        'metrics_sha256':sha(out/'metrics.csv'),'actual_medical_training':True})
    return out/'model.pt'
