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
from .profile_policy import allow_profile,validate_policy

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
        from .data import scale_count
        super().__init__(reference,seed=42,resource_budget=budget,
            allow_unvalidated_profile=allow_profile(contract.get('profile_policy','strict'),debug),levels=scale_count(contract['profile']))
        self.training_contract=copy.deepcopy(contract)
        for block in self.core.blocks:
            for conv in block.conv.convs.values():conv.stable_spmm=True
    def get_extra_state(self):return copy.deepcopy(self.training_contract)
    def set_extra_state(self,state):
        if state!=self.training_contract:raise ValueError('Different region model/profile/view contract')

def make_model(ds,budget,debug,activation_storage='checkpointed'):
    if activation_storage not in ('checkpointed','retained'):raise ValueError('Unknown activation storage')
    m=ds.meta;configure_runtime(m['base'],m['config']['seed'])
    random.seed(m['config']['seed']);np.random.seed(m['config']['seed'])
    with installed('stride4'):ref=V1LocalEncoder(m['base'])
    weights=torch.load(ds.root/'frozen_cnn.pt',map_location='cpu',weights_only=True)
    if tree_hash(weights)!=m['cnn_sha256']:raise ValueError('Frozen CNN contents differ')
    ref.dense_encoder.load_state_dict(weights,strict=True)
    contract=dict(format=FORMAT,debug=debug,profile=m['profile'],frozen_cnn_sha256=m['cnn_sha256'],
                  view_epoch=m['view_epoch'],feature_coordinates='stride4',profile_policy=m.get('profile_policy','strict'))
    net=PromptGraphModel(m['config'],m['base'],{},local_encoder=TrainEncoder(ref,budget=budget,debug=debug,contract=contract)).cuda()
    if activation_storage=='retained':
        net.local.core.checkpoint_dense_encoder=False
        net.local.core.checkpoint_local_blocks=False
        net.checkpoint_support=False
    return net

class Loader:
    def __init__(self,ds,workers,resident_bytes):
        self.ds=ds;self.cache=RegionResidentCache(max_tensor_bytes=resident_bytes,workers=workers,
            compatible_sources=ds.compatible_sources)
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

def load_checkpoint(path,identity,*,allow_execution_upgrade=False,allow_cuda_budget_change=False,allow_support_migration=False):
    if allow_cuda_budget_change and not allow_execution_upgrade:
        raise ValueError('CUDA budget migration requires explicit execution upgrade')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    digest=payload.pop('content_sha256',None)
    if payload.get('format')!=FORMAT:raise ValueError('Not an exact region resume; GAT/DEBUG/profile/source changes forbidden')
    if digest!=hash_state(payload):raise ValueError('Checkpoint model/optimizer/state/RNG contents changed')
    if allow_support_migration:
        if allow_execution_upgrade or allow_cuda_budget_change:raise ValueError('Separate learning migration from execution/budget migration')
        from .support_episodes import verify_migration
        revision=verify_migration(payload['identity'],identity)
        state=payload['state']
        if state['phase']!='optimization':raise ValueError('Support migration requires a paused optimization checkpoint')
        state['support_migration']=dict(previous_identity=payload['identity'],reviewed_revision=revision,
            step=state['step'],epoch=state['epoch'],next_batch=state['next_batch'],exact_resume=False,
            previous_best=None if state['best'] is None else {k:v for k,v in state['best'].items() if k!='weights'})
        state.update(plan=None,last_group=None,best=None,selected_epoch=None)
        return payload
    if payload.get('identity')!=identity:
        if not allow_execution_upgrade:raise ValueError('Not an exact region resume; GAT/DEBUG/profile/source changes forbidden')
        from .execution_upgrade import verify_upgrade
        revision=verify_upgrade(payload['identity'],identity,allow_cuda_budget_change=allow_cuda_budget_change)
        payload['execution_upgrade']=dict(reviewed_revision=revision,
            original_identity=payload['identity'],step=payload['state']['step'],
            preserved=['model','optimizer','RNG','support','cluster plan','epoch','next batch','physical batch'],
            activation_storage=identity['activation_storage'],exact_old_runtime_replay=False,
            cuda_budget_change=dict(authorized=allow_cuda_budget_change,
                previous_bytes=payload['identity']['resource_limits']['cuda_bytes'],
                current_bytes=identity['resource_limits']['cuda_bytes']))
    return payload

def metadata(ds,embeddings):
    groups_=sorted({r['patient_group'] for r in ds.rows})
    return dict(embeddings=embeddings,record_ids=[r['id'] for r in ds.rows],patient_groups=groups_,
        donor_groups=[r['donor_group'] for r in ds.rows],
        owners=torch.tensor([groups_.index(r['patient_group']) for r in ds.rows],device='cuda'),
        classes=torch.tensor([r['target'] for r in ds.rows],device='cuda'))

@torch.no_grad()
def initial_memory(net,ds,loader,batch,timing_path=None):
    net.eval();parts=[]
    net.local.dense_batch_size=batch
    order=contiguous(ds,batch);done=0;iterator=iter(loader.batches(order))
    bar=tqdm(total=(len(ds)+batch-1)//batch,desc='calibration support')
    while done<len(ds):
        start=time.perf_counter();cpu=next(iterator);wait=time.perf_counter()-start
        ids=cpu.indices.tolist()
        if ids!=list(range(done,done+len(ids))):raise ValueError('Calibration support coverage/order changed')
        torch.cuda.synchronize();start=time.perf_counter();query=cpu.to('cuda')
        torch.cuda.synchronize();transfer=time.perf_counter()-start
        start=time.perf_counter();embedding=net.local(query).float()
        torch.cuda.synchronize();forward=time.perf_counter()-start
        parts.append(embedding);done+=len(ids)
        row=dict(stage='calibration_support',completed=done,total=len(ds),physical_batch=len(ids),
            cnn_chunk=batch,loader_wait_seconds=wait,h2d_validation_seconds=transfer,forward_seconds=forward,
            rss_bytes=psutil.Process().memory_info().rss,cuda_allocated_bytes=torch.cuda.memory_allocated())
        if timing_path is not None:
            with Path(timing_path).open('a',encoding='utf8') as stream:stream.write(json.dumps(row)+'\n')
        bar.update();bar.set_postfix(wait_s=round(wait,2),transfer_s=round(transfer,2),forward_s=round(forward,2),cnn=batch)
        del query
    bar.close()
    return metadata(ds,torch.cat(parts))

def calibrate(net,ds,loader,candidates,base,budget,timing_path=None,support_patients=None):
    # Full reference memory; active support policy and heaviest query group.
    # Calibration clones leave the real training state intact.
    initial_model_hash=hash_state(net.state_dict())
    memory=initial_memory(net,ds,loader,min(candidates),timing_path)
    initial_memory_hash=hash_state(memory)
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
        if support_patients is None:
            support=support_for_recipient(memory,group)
        else:
            from .support_episodes import PatientEpisodes
            episodes=PatientEpisodes(ds.rows,list(groups(ds,cap,ds.meta['config']['seed'],0)),
                support_patients,ds.meta['config']['seed'],0).bind(memory)
            support=episodes.support(group)
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
    if hash_state(net.state_dict())!=initial_model_hash:raise RuntimeError('Calibration mutated the model bound to support')
    if hash_state(memory)!=initial_memory_hash:raise RuntimeError('Calibration mutated the reusable support')
    return max(accepted,key=lambda r:r['graphs_per_second'])['configured_batch'],reports,memory

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

def train(index,output,*,workers,resident_bytes,candidates,budget,debug=False,resume=None,debug_pause_step=None,profile_policy='strict',activation_storage='checkpointed',resume_execution_upgrade=False,resume_cuda_budget_change=False,support_patients=None,resume_support_minibatch=False):
    validate_policy(profile_policy)
    from .support_episodes import PatientEpisodes,contract as support_contract
    support_policy=None if support_patients is None else support_contract(support_patients)
    if resume_support_minibatch and (not resume or support_policy is None or resume_execution_upgrade or resume_cuda_budget_change):
        raise ValueError('Support migration requires resume + explicit support patients, without execution/budget upgrade')
    if resume_cuda_budget_change and not resume_execution_upgrade:
        raise ValueError('CUDA budget migration requires explicit execution upgrade')
    if resume_execution_upgrade and not resume:raise ValueError('Execution upgrade requires a saved checkpoint')
    if debug_pause_step is not None and not debug:raise ValueError('Pause step is DEBUG only')
    if not candidates or min(candidates)<1 or candidates!=sorted(set(candidates)):raise ValueError('Explicit increasing physical batch candidates required')
    ds=RegionDataset(index,'inner_train',debug,profile_policy);val=RegionDataset(index,'inner_val',debug,profile_policy)
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(workers);net=make_model(ds,budget,debug,activation_storage)
    base=ds.meta['base'];cfg=ds.meta['config'];epochs=1 if debug else cfg['gnn_epochs']
    loader=Loader(ds,workers,resident_bytes);val_loader=Loader(val,workers,resident_bytes)
    optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
    identity=dict(format=FORMAT,debug=debug,cache_sha256=sha(index),source=source_identity(),epochs=epochs,
        config=cfg,base=base,precision='FP32',workers=workers,candidates=candidates,
        ranking=rank_config(),resident_budget_bytes=resident_bytes,profile_policy=profile_policy,
        activation_storage=activation_storage)
    identity['resource_limits']=dict(cuda_bytes=budget.cuda_bytes,rss_bytes=budget.rss_bytes)
    if support_policy is not None:identity['support_training']=support_policy
    if resume:
        saved=load_checkpoint(resume,identity,allow_execution_upgrade=resume_execution_upgrade,
            allow_cuda_budget_change=resume_cuda_budget_change,allow_support_migration=resume_support_minibatch)
        net.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        if saved.get('execution_upgrade'):write_new(root/'execution_upgrade.json',saved['execution_upgrade'])
        state=tree_to(saved['state'],'cuda');restore_rng(saved['rng']);del saved
        if resume_support_minibatch:write_new(root/'support_migration.json',state['support_migration'])
    else:
        batch,report,memory=calibrate(net,ds,loader,candidates,base,budget,root/'support_timing.jsonl',support_patients)
        write_new(root/'batch_calibration.json',report)
        # Calibration updated clones only. Its complete, ordered support belongs
        # to this exact initial model; do not discard it and run another cohort pass.
        state=dict(epoch=0,step=0,phase='optimization',memory=memory,memory_parts=[],memory_done=0,
            batch=batch,next_batch=0,plan=None,last_group=None,best=None,selected_epoch=None)
    net.local.dense_batch_size=state['batch']
    write_new(root/'execution_contract.json',dict(**identity,train_samples=len(ds),val_samples=len(val),usage_ratio=1.,
        physical_batch=state['batch'],effective_batch=state['batch'],accumulation=1,parameters=sum(p.numel() for p in net.parameters()),
        layers=dict(L0=ds.meta['profile']['graph_encoder']['sage_layers'],L1=2,L2=2),hidden=128,cnn=[12,24,32],candidate_count=128,
        gpu=torch.cuda.get_device_name(),cpu_logical=psutil.cpu_count(),ram=psutil.virtual_memory()._asdict(),
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
        L0_parameters=sum(p.numel() for p in net.local.parameters()),input_shape=[state['batch'],1,48,48,48],
        graph_counts=[dict(record=e['row']['id'],fine_nodes=e['fine_nodes'],region_nodes=e['region_nodes'],region_edges=e['region_edges']) for e in ds.entries],
        optimization_steps=sum(1 for _ in groups(ds,state['batch']))*epochs,
        actual_query_batch_sizes=[len(ids) for ids in groups(ds,state['batch'])],
        cache_preparation_revision=ds.preparation_revision,cache_rebuilt=False,
        initial_support_policy='reuse full calibration support after verifying model unchanged',
        scope='GNN only; final artifact connects to the existing CP recommendation adapter',
        profile_exceeded=bool(ds.meta['admission_failures']),partition_quality_validated=False,
        research_training=profile_policy=='research-report',production_admitted=not debug and profile_policy=='strict'))
    print(json.dumps(dict(stage='region_training_policy',profile_policy=profile_policy,
        profile_violating_records=ds.meta['admission_failures'],partition_quality_validated=False,
        debug=debug,epochs=epochs,activation_storage=activation_storage,
        activation_checkpointing=dict(CNN=net.local.core.checkpoint_dense_encoder,
            L0=net.local.core.checkpoint_local_blocks,L1_L2=net.checkpoint_support),
        physical_batch=state['batch'],resumed_step=state['step'],support_training=support_policy or 'full_support',
        cuda_limit_gib=budget.cuda_bytes/2**30)),flush=True)
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
                    write_new(root/'training_complete.json',dict(debug=debug,epochs=epochs,steps=state['step'],selected_epoch=state['selected_epoch'],checkpoint_sha256=sha(root/'checkpoint_latest.pt'),nnunet_training=False,profile_policy=profile_policy,partition_quality_validated=False))
                    return root/'checkpoint_latest.pt'
                state['phase']='optimization' if state['phase']=='initial_memory' else 'validation'
                save_checkpoint(root,net,optimizer,state,identity)
            if state['phase']=='optimization':
                net.train();order=list(groups(ds,state['batch'],cfg['seed'],state['epoch']))
                counts=torch.bincount(state['memory']['classes'],minlength=2).float()
                if bool((counts==0).any()):raise ValueError('Both observation classes required')
                weights=counts.sum()/(2*counts);context=RankingContext(ds,state['memory'])
                episodes=None
                if support_policy is not None:
                    episodes=PatientEpisodes(ds.rows,order,support_patients,cfg['seed'],state['epoch']).bind(state['memory'])
                    audit_path=root/f"support_epoch_{state['epoch']+1:03d}.json"
                    if not audit_path.exists():write_new(audit_path,dict(episodes.audit,starting_query_batch=state['next_batch'],
                        coverage_scope='complete deterministic epoch schedule; actual consumed prefix is checkpoint next_batch'))
                bar=tqdm(loader.batches(order[state['next_batch']:]),initial=state['next_batch'],total=len(order),desc=f"epoch {state['epoch']+1}/{epochs}")
                previous_end=time.perf_counter()
                for cpu in bar:
                    load_wait=time.perf_counter()-previous_end;step_start=time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    budget.check();ids=cpu.indices.tolist()
                    if ids!=order[state['next_batch']]:raise ValueError('Saved query cursor mismatch')
                    group=ds.rows[ids[0]]['patient_group']
                    support=(support_for_recipient(state['memory'],group) if episodes is None else episodes.support(group))
                    support_records=len(support[0])
                    support_groups=(int(support[1].max())+1 if episodes is None else len(episodes.selections[group]['patients']))
                    if state['last_group']!=group:
                        net.eval()
                        with torch.no_grad():state['plan']=net.fit_support_clusters(*support)
                        net.train();state['last_group']=group
                    optimizer.zero_grad(set_to_none=True)
                    events=[torch.cuda.Event(enable_timing=True) for _ in range(5)]
                    events[0].record();query=cpu.to('cuda');events[1].record()
                    loss,terms=forward_loss(net,query,support,state['plan'],state['memory']['classes'][ids],weights,context,rank_config(),indices=ids)
                    events[2].record()
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward();events[3].record();gradient_check(net)
                    torch.nn.utils.clip_grad_norm_(net.parameters(),base['training']['grad_clip'],error_if_nonfinite=True);optimizer.step()
                    events[4].record();events[4].synchronize()
                    state['next_batch']+=1;state['step']+=1
                    loss_value=float(loss.detach());peak=torch.cuda.max_memory_allocated()
                    del loss,terms,support,query
                    save_start=time.perf_counter()
                    save_checkpoint(root,net,optimizer,state,identity)
                    save_seconds=time.perf_counter()-save_start
                    row=dict(step=state['step'],epoch=state['epoch']+1,physical_batch=len(ids),
                        activation_storage=activation_storage,loss=loss_value,loader_wait_seconds=load_wait,
                        support_records=support_records,support_patients=support_groups,support_policy=support_policy,
                        transfer_seconds=events[0].elapsed_time(events[1])/1000,
                        forward_seconds=events[1].elapsed_time(events[2])/1000,
                        backward_seconds=events[2].elapsed_time(events[3])/1000,
                        check_clip_optimizer_seconds=events[3].elapsed_time(events[4])/1000,
                        checkpoint_seconds=save_seconds,step_seconds=time.perf_counter()-step_start,
                        peak_cuda_bytes=peak,rss_bytes=psutil.Process().memory_info().rss)
                    with (root/'update_timing.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
                    bar.set_postfix(step=state['step'],loss=round(loss_value,4),seconds=round(row['step_seconds'],2),peak_GiB=round(peak/2**30,2),support=f'{support_groups}p/{support_records}r')
                    if pause():return root/'checkpoint_latest.pt'
                    previous_end=time.perf_counter()
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
