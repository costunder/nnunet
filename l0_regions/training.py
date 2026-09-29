"""Fixed-region SAGE research training with explicit full/DEBUG artifacts.

No diagnostic admission is silently promoted. Existing GAT checkpoints initialize
only the frozen/live CNN through the prepared cache, never resume this model.
"""
import copy
import gc
import json
import random
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import psutil
import torch
from tqdm import tqdm
from hiercp_v222.model import PromptGraphModel
from hiercp_v222.v1_local import V1LocalEncoder, support_for_recipient
from hiercp_v222.v1_training import groups, contiguous, gradient_check
from hiercp_v222.v1_execution import atomic_torch, rng_state, restore_rng, tree_to
from hiercp_v222.training import configure_runtime
from tools.v222_review_contracts import installed
from tools.v22_rank_objective import RankingContext,forward_loss,ranking_metrics,configuration as rank_config
from tools.v22_candidate_order import record_key
from tools.v22_artifacts import tree_hash
from .encoder import RegionSAGEEncoder
from .resident import RegionResidentCache
from .training_data import RegionDataset,sha,write_new,source_identity

FORMAT='fixed_region_sage_training_v1'

def hash_state(value):
    def canonical(v):
        if isinstance(v,np.ndarray):return torch.from_numpy(v.copy())
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,dict):return {k:canonical(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [canonical(x) for x in v]
        return v
    return tree_hash(canonical(value))

class TrainEncoder(RegionSAGEEncoder):
    def __init__(self,reference,*,budget,debug,contract):
        super().__init__(reference,seed=42,resource_budget=budget,allow_unvalidated_profile=debug)
        self.training_contract=copy.deepcopy(contract)
        for block in self.core.blocks:
            for conv in block.conv.convs.values():conv.stable_spmm=True
    def get_extra_state(self):return copy.deepcopy(self.training_contract)
    def set_extra_state(self,state):
        if state!=self.training_contract:raise ValueError('Different region model/profile/view contract')

def make_model(ds,budget,debug):
    m=ds.meta;configure_runtime(m['base'],m['config']['seed'])
    random.seed(m['config']['seed']);np.random.seed(m['config']['seed'])
    with installed('stride4'):ref=V1LocalEncoder(m['base'])
    weights=torch.load(ds.root/'frozen_cnn.pt',map_location='cpu',weights_only=True)
    if tree_hash(weights)!=m['cnn_sha256']:raise ValueError('Frozen CNN contents differ')
    ref.dense_encoder.load_state_dict(weights,strict=True)
    contract=dict(format=FORMAT,debug=debug,profile=m['profile'],frozen_cnn_sha256=m['cnn_sha256'],
                  view_epoch=m['view_epoch'],feature_coordinates='stride4')
    return PromptGraphModel(m['config'],m['base'],{},local_encoder=TrainEncoder(ref,budget=budget,debug=debug,contract=contract)).cuda()

class Loader:
    def __init__(self,ds,workers,resident_bytes):
        self.ds=ds;self.cache=RegionResidentCache(max_tensor_bytes=resident_bytes,workers=workers)
    def get(self,ids):return self.cache.get(*self.ds.request(ids))
    def batches(self,order):
        # One batch ahead; load uses independent per-record decode threads.
        with ThreadPoolExecutor(max_workers=1) as pool:
            iterator=iter(order);ids=next(iterator,None)
            if ids is None:return
            future=pool.submit(self.get,ids)
            for following in iterator:
                batch=future.result();future=pool.submit(self.get,following);yield batch
            yield future.result()

@contextmanager
def pause_signal():
    flag={'requested':False}
    old=signal.getsignal(signal.SIGINT)
    def request(signum,frame):
        flag['requested']=True
        print('\nPause requested; finishing the active batch and saving.',flush=True)
    signal.signal(signal.SIGINT,request)
    try:yield flag
    finally:signal.signal(signal.SIGINT,old)

def save_checkpoint(root,net,optimizer,state,identity):
    payload=dict(format=FORMAT,identity=identity,model=tree_to(net.state_dict(),'cpu'),
        optimizer=tree_to(optimizer.state_dict(),'cpu'),state=tree_to(state,'cpu'),rng=rng_state())
    payload['content_sha256']=hash_state(payload)
    atomic_torch(root/'checkpoint_latest.pt',payload)

def load_checkpoint(path,identity):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    digest=payload.pop('content_sha256',None)
    if payload.get('format')!=FORMAT or payload.get('identity')!=identity:raise ValueError('Not an exact region resume; GAT/DEBUG/profile/source changes forbidden')
    if digest!=hash_state(payload):raise ValueError('Checkpoint model/optimizer/state/RNG contents changed')
    return payload

def metadata(ds,embeddings):
    groups_=sorted({r['patient_group'] for r in ds.rows})
    return dict(embeddings=embeddings,record_ids=[r['id'] for r in ds.rows],patient_groups=groups_,
        donor_groups=[r['donor_group'] for r in ds.rows],
        owners=torch.tensor([groups_.index(r['patient_group']) for r in ds.rows],device='cuda'),
        classes=torch.tensor([r['target'] for r in ds.rows],device='cuda'))

@torch.no_grad()
def initial_memory(net,ds,loader,batch):
    net.eval();parts=[]
    for cpu in tqdm(loader.batches(contiguous(ds,batch)),total=(len(ds)+batch-1)//batch,desc='calibration support'):
        parts.append(net.local(cpu.to('cuda')).float())
    return metadata(ds,torch.cat(parts))

def calibrate(net,ds,loader,candidates,base,budget):
    # Complete support, graph-heaviest query group; clones leave train state intact.
    memory=initial_memory(net,ds,loader,min(candidates))
    context=RankingContext(ds,memory);weights=torch.bincount(memory['classes'],minlength=2).float()
    if bool((weights==0).any()):raise ValueError('Both observed classes required')
    weights=weights.sum()/(2*weights)
    group=max(memory['patient_groups'],key=lambda g:sum(r['bounds']['edges'] for r in ds.rows if r['patient_group']==g))
    ids=sorted([i for i,r in enumerate(ds.rows) if r['patient_group']==group],key=lambda i:ds.rows[i]['bounds']['edges'],reverse=True)
    reports=[];seen=set()
    for cap in candidates:
        selected=ids[:cap]
        if len(selected) in seen:continue
        from l0_sage.encoder import MeanAdjacencyCache
        # Sparse CSR storages cannot be deep-copied. They are execution caches,
        # not model state; cloned convs must share the clone's fresh caches.
        seen.add(len(selected));clone=copy.deepcopy(net,{id(c):MeanAdjacencyCache() for c in net.local.adjacencies}).train();clone.local.dense_batch_size=cap
        optimizer=torch.optim.AdamW(clone.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
        support=support_for_recipient(memory,group)
        plan=clone.fit_support_clusters(*support)
        query=loader.get(selected).to('cuda');times=[]
        try:
            torch.cuda.reset_peak_memory_stats()
            for trial in range(3):
                optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                loss,_=forward_loss(clone,query,support,plan,memory['classes'][selected],weights,context,rank_config(),indices=selected)
                loss.backward();gradient_check(clone)
                torch.nn.utils.clip_grad_norm_(clone.parameters(),base['training']['grad_clip'],error_if_nonfinite=True);optimizer.step()
                torch.cuda.synchronize()
                if trial:times.append(time.perf_counter()-start)
                budget.check()
            reports.append(dict(configured_batch=cap,actual_batch=len(selected),accepted=True,graphs_per_second=len(selected)/(sum(times)/len(times)),peak_bytes=torch.cuda.max_memory_allocated()))
        except torch.cuda.OutOfMemoryError as exc:
            reports.append(dict(configured_batch=cap,actual_batch=len(selected),accepted=False,error=str(exc)))
        finally:
            del clone,optimizer,query,plan,support
            gc.collect();torch.cuda.empty_cache()
        if not reports[-1]['accepted']:break
    accepted=[r for r in reports if r['accepted']]
    if not accepted:raise MemoryError('No full-graph batch admitted; no automatic model reduction')
    return max(accepted,key=lambda r:r['graphs_per_second'])['configured_batch'],reports

@torch.no_grad()
def evaluate(net,ds,loader,memory,batch):
    net.eval();seen=[];scores=[];truth=[];cases=[];last=None
    order=list(groups(ds,batch))
    for cpu in tqdm(loader.batches(order),total=len(order),desc='validation'):
        ids=cpu.indices.tolist();group=ds.rows[ids[0]]['patient_group']
        if group!=last:state=net.prepare_support(*support_for_recipient(memory,group));last=group
        output=net.predict_embeddings(net.local(cpu.to('cuda')),state)['logits'].float()
        scores.extend((output[:,1]-output[:,0]).cpu().tolist());seen.extend(ids)
        truth.extend(ds.rows[i]['target'] for i in ids);cases.extend(ds.rows[i]['case_id'] for i in ids)
    if sorted(seen)!=list(range(len(ds))):raise ValueError('Validation coverage changed')
    return ranking_metrics(scores,truth,cases,rank_config()['report_recall_at'],candidate_keys=[record_key(ds.rows[i]) for i in seen])

def train(index,output,*,workers,resident_bytes,candidates,budget,debug=False,resume=None,debug_pause_step=None):
    if debug_pause_step is not None and not debug:raise ValueError('Pause step is DEBUG only')
    if not candidates or min(candidates)<1 or candidates!=sorted(set(candidates)):raise ValueError('Explicit increasing physical batch candidates required')
    ds=RegionDataset(index,'inner_train',debug);val=RegionDataset(index,'inner_val',debug)
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(workers);net=make_model(ds,budget,debug)
    base=ds.meta['base'];cfg=ds.meta['config'];epochs=1 if debug else cfg['gnn_epochs']
    loader=Loader(ds,workers,resident_bytes);val_loader=Loader(val,workers,resident_bytes)
    optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
    identity=dict(format=FORMAT,debug=debug,cache_sha256=sha(index),source=source_identity(),epochs=epochs,
        config=cfg,base=base,precision='FP32',workers=workers,candidates=candidates,
        ranking=rank_config(),resident_budget_bytes=resident_bytes)
    identity['resource_limits']=dict(cuda_bytes=budget.cuda_bytes,rss_bytes=budget.rss_bytes)
    if resume:
        saved=load_checkpoint(resume,identity);net.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        state=tree_to(saved['state'],'cuda');restore_rng(saved['rng']);del saved
    else:
        batch,report=calibrate(net,ds,loader,candidates,base,budget)
        write_new(root/'batch_calibration.json',report)
        state=dict(epoch=0,step=0,phase='initial_memory',memory=None,memory_parts=[],memory_done=0,
            batch=batch,next_batch=0,plan=None,last_group=None,best=None,selected_epoch=None)
    net.local.dense_batch_size=state['batch']
    write_new(root/'execution_contract.json',dict(**identity,train_samples=len(ds),val_samples=len(val),usage_ratio=1.,
        physical_batch=state['batch'],effective_batch=state['batch'],accumulation=1,parameters=sum(p.numel() for p in net.parameters()),
        layers=dict(L0=[2,1],L1=2,L2=2),hidden=128,cnn=[12,24,32],candidate_count=128,
        gpu=torch.cuda.get_device_name(),cpu_logical=psutil.cpu_count(),ram=psutil.virtual_memory()._asdict(),
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
        L0_parameters=sum(p.numel() for p in net.local.parameters()),input_shape=[state['batch'],1,48,48,48],
        graph_counts=[dict(record=e['row']['id'],fine_nodes=e['fine_nodes'],region_nodes=e['region_nodes'],region_edges=e['region_edges']) for e in ds.entries],
        optimization_steps=sum(1 for _ in groups(ds,state['batch']))*epochs,
        actual_query_batch_sizes=[len(ids) for ids in groups(ds,state['batch'])],
        scope='GNN only; separate final model format, online CP adapter not yet connected',
        profile_exceeded=bool(ds.meta['admission_failures']),production_admitted=not debug))
    save_checkpoint(root,net,optimizer,state,identity)
    with pause_signal() as flag:
        def pause():
            if flag['requested'] or (root/'STOP_AFTER_BATCH').exists() or (debug_pause_step is not None and state['step']>=debug_pause_step):
                save_checkpoint(root,net,optimizer,state,identity)
                write_new(root/'paused.json',dict(step=state['step'],phase=state['phase'],debug=debug))
                print('PAUSED: checkpoint_latest.pt saved',flush=True);return True
            return False
        while True:
            budget.check()
            if state['phase'] in ('initial_memory','refresh_memory','final_memory'):
                net.eval();done=state['memory_done'];order=list(contiguous(ds,state['batch']))
                with torch.no_grad():
                    for cpu in tqdm(loader.batches([ids for ids in order if ids[0]>=done]),desc=state['phase'],total=sum(ids[0]>=done for ids in order)):
                        ids=cpu.indices.tolist()
                        if ids!=list(range(state['memory_done'],state['memory_done']+len(ids))):raise ValueError('Memory cursor mismatch')
                        state['memory_parts'].append(net.local(cpu.to('cuda')).detach().float())
                        state['memory_done']+=len(ids);save_checkpoint(root,net,optimizer,state,identity)
                        if pause():return root/'checkpoint_latest.pt'
                if state['memory_done']!=len(ds):raise ValueError('Incomplete support memory')
                state['memory']=metadata(ds,torch.cat(state['memory_parts']));state['memory_parts']=[];state['memory_done']=0
                if state['phase']=='final_memory':
                    if hash_state(net.state_dict())!=state['best']['model_sha256']:raise ValueError('Final memory is not bound to selected best model')
                    state['phase']='complete';save_checkpoint(root,net,optimizer,state,identity)
                    write_new(root/'training_complete.json',dict(debug=debug,epochs=epochs,steps=state['step'],selected_epoch=state['selected_epoch'],checkpoint_sha256=sha(root/'checkpoint_latest.pt'),nnunet_training=False))
                    return root/'checkpoint_latest.pt'
                state['phase']='optimization' if state['phase']=='initial_memory' else 'validation'
                save_checkpoint(root,net,optimizer,state,identity)
            if state['phase']=='optimization':
                net.train();order=list(groups(ds,state['batch'],cfg['seed'],state['epoch']))
                counts=torch.bincount(state['memory']['classes'],minlength=2).float()
                if bool((counts==0).any()):raise ValueError('Both observation classes required')
                weights=counts.sum()/(2*counts);context=RankingContext(ds,state['memory'])
                bar=tqdm(loader.batches(order[state['next_batch']:]),initial=state['next_batch'],total=len(order),desc=f"epoch {state['epoch']+1}/{epochs}")
                for cpu in bar:
                    budget.check();ids=cpu.indices.tolist()
                    if ids!=order[state['next_batch']]:raise ValueError('Saved query cursor mismatch')
                    group=ds.rows[ids[0]]['patient_group'];support=support_for_recipient(state['memory'],group)
                    if state['last_group']!=group:
                        net.eval()
                        with torch.no_grad():state['plan']=net.fit_support_clusters(*support)
                        net.train();state['last_group']=group
                    optimizer.zero_grad(set_to_none=True)
                    loss,terms=forward_loss(net,cpu.to('cuda'),support,state['plan'],state['memory']['classes'][ids],weights,context,rank_config(),indices=ids)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward();gradient_check(net)
                    torch.nn.utils.clip_grad_norm_(net.parameters(),base['training']['grad_clip'],error_if_nonfinite=True);optimizer.step()
                    state['next_batch']+=1;state['step']+=1
                    bar.set_postfix(step=state['step'],loss=float(loss.detach()))
                    del loss,terms,support
                    save_checkpoint(root,net,optimizer,state,identity)
                    if pause():return root/'checkpoint_latest.pt'
                state['phase']='refresh_memory';save_checkpoint(root,net,optimizer,state,identity)
            if state['phase']=='validation':
                metrics,details=evaluate(net,val,val_loader,state['memory'],state['batch'])
                write_new(root/f"epoch_{state['epoch']+1:03d}.json",dict(metrics=metrics,cases=details,debug=debug))
                score=metrics['ranking_pairwise_loss']
                if state['best'] is None or score<state['best']['metric']:
                    weights=tree_to(net.state_dict(),'cpu');state['best']=dict(weights=weights,metric=score,epoch=state['epoch']+1,model_sha256=hash_state(weights))
                state.update(epoch=state['epoch']+1,next_batch=0,last_group=None,plan=None,phase='optimization')
                if state['epoch']==epochs:
                    net.load_state_dict(state['best']['weights']);state['selected_epoch']=state['best']['epoch'];state['phase']='final_memory'
                save_checkpoint(root,net,optimizer,state,identity)
                if pause():return root/'checkpoint_latest.pt'
            if state['phase']=='complete':raise ValueError('Completed runs need no resume')
