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
from hiercp_v222.v1_training import groups as legacy_groups, contiguous, gradient_check
from hiercp_v222.v1_execution import atomic_torch, rng_state, restore_rng, tree_to
from hiercp_v222.training import configure_runtime
from tools.v222_review_contracts import installed
from tools.v22_rank_objective import RankingContext,forward_loss as legacy_forward_loss,ranking_metrics,configuration as legacy_rank_config
from tools.v22_candidate_order import record_key
from tools.v22_artifacts import tree_hash
from .encoder import RegionSAGEEncoder
from .resident import RegionResidentCache
from .training_data import RegionDataset,sha,write_new,source_identity
from .profile_policy import allow_profile,validate_policy

FORMAT='fixed_region_sage_training_v1'

def groups(ds,batch,seed=None,epoch=0):
    if ds.meta.get('learning_policy'):
        from .donor_learning import groups as live_groups
        return live_groups(ds,batch,seed,epoch)
    return legacy_groups(ds,batch,seed,epoch)

def rank_config(ds=None):
    if ds is not None and ds.meta.get('learning_policy'):
        from .donor_learning import configuration
        return configuration()
    return legacy_rank_config()

def ranking_context(ds,memory,batch):
    if ds.meta.get('learning_policy'):
        from .donor_learning import LiveContext
        return LiveContext(ds,batch)
    return RankingContext(ds,memory)

def forward_loss(net,query,support,plan,targets,weights,context,settings,*,indices):
    from .donor_learning import LiveContext,forward_loss as live_loss
    fn=live_loss if isinstance(context,LiveContext) else legacy_forward_loss
    return fn(net,query,support,plan,targets,weights,context,settings,indices=indices)

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
    if 'local_cnn' in m:
        from l0_local_cnn.model import LocalCNN
        local=LocalCNN(m['local_cnn'],checkpointing=activation_storage=='checkpointed',budget=budget,dropout=m['base']['model']['dropout'])
        net=PromptGraphModel(m['config'],m['base'],{},local_encoder=local).cuda()
        net.checkpoint_support=activation_storage=='checkpointed'
        return net
    with installed('stride4'):ref=V1LocalEncoder(m['base'])
    weights=torch.load(ds.root/'frozen_cnn.pt',map_location='cpu',weights_only=True)
    if tree_hash(weights)!=m['cnn_sha256']:raise ValueError('Frozen CNN contents differ')
    ref.dense_encoder.load_state_dict(weights,strict=True)
    contract=dict(format=FORMAT,debug=debug,profile=m['profile'],frozen_cnn_sha256=m['cnn_sha256'],
                  view_epoch=m['view_epoch'],feature_coordinates='stride4',profile_policy=m.get('profile_policy','strict'))
    encoder=TrainEncoder
    if m.get('graph_representation'):
        from .fine_graph import FineTrainEncoder,MODE
        contract['graph_representation']=MODE;encoder=FineTrainEncoder
    net=PromptGraphModel(m['config'],m['base'],{},local_encoder=encoder(ref,budget=budget,debug=debug,contract=contract)).cuda()
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
        if 'execution_pipeline' in identity:
            if allow_cuda_budget_change:raise ValueError('Separate CUDA budget migration from pipeline upgrade')
            from .execution_pipeline import verify_pipeline_upgrade
            revision=verify_pipeline_upgrade(payload['identity'],identity)
        else:
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
    weights=torch.bincount(memory['classes'],minlength=2).float()
    if bool((weights==0).any()):raise ValueError('Both observed classes required')
    weights=weights.sum()/(2*weights)
    group=max(memory['patient_groups'],key=lambda g:sum(r['bounds']['edges'] for r in ds.rows if r['patient_group']==g))
    ids=sorted([i for i,r in enumerate(ds.rows) if r['patient_group']==group],key=lambda i:ds.rows[i]['bounds']['edges'],reverse=True)
    reports=[];seen=set()
    for cap in candidates:
        context=ranking_context(ds,memory,cap)
        selected=(max(list(groups(ds,cap)),key=lambda xs:sum(ds.rows[i]['bounds']['edges'] for i in xs)) if ds.meta.get('learning_policy') else ids[:cap])
        group=ds.rows[selected[0]]['patient_group']
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
            episodes=PatientEpisodes(ds.rows,list(legacy_groups(ds,cap,ds.meta['config']['seed'],0)),
                support_patients,ds.meta['config']['seed'],0).bind(memory)
            support=episodes.support(group)
        plan=clone.fit_support_clusters(*support)
        query=loader.get(selected).to('cuda');times=[]
        try:
            torch.cuda.reset_peak_memory_stats()
            for trial in range(3):
                optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                loss,_=forward_loss(clone,query,support,plan,memory['classes'][selected],weights,context,rank_config(ds),indices=selected)
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
    # Evaluation observes each candidate once, including in the live-pair policy.
    order=list(legacy_groups(ds,batch))
    for cpu in tqdm(loader.batches(order),total=len(order),desc='validation'):
        ids=cpu.indices.tolist();group=ds.rows[ids[0]]['patient_group']
        if group!=last:state=net.prepare_support(*support_for_recipient(memory,group));last=group
        output=net.predict_embeddings(net.local(cpu.to('cuda')),state)['logits'].float()
        scores.extend((output[:,1]-output[:,0]).cpu().tolist());seen.extend(ids)
        truth.extend(ds.rows[i]['target'] for i in ids);cases.extend(ds.rows[i]['case_id'] for i in ids)
    if sorted(seen)!=list(range(len(ds))):raise ValueError('Validation coverage changed')
    return ranking_metrics(scores,truth,cases,rank_config()['report_recall_at'],candidate_keys=[record_key(ds.rows[i]) for i in seen])

def train(index,output,*,workers,resident_bytes,candidates,budget,debug=False,resume=None,debug_pause_step=None,profile_policy='strict',activation_storage='checkpointed',resume_execution_upgrade=False,resume_cuda_budget_change=False,support_patients=None,resume_support_minibatch=False,execution_pipeline='synchronous',device_cache_bytes=0,sage_workspace_bytes=64*2**20,fine_cache=None,resume_without_coarsening=False,learning_policy=None,local_cnn=False):
    validate_policy(profile_policy)
    from .support_episodes import PatientEpisodes,contract as support_contract
    support_policy=None if support_patients is None else support_contract(support_patients)
    from .execution_pipeline import CheckpointPipeline,DeviceBatchCache,gradient_check_batched
    from .sparse import workspace
    from .learning_monitor import LearningMonitor,validation_line
    if type(sage_workspace_bytes) is not int or sage_workspace_bytes<=0 or sage_workspace_bytes>=budget.cuda_bytes:raise ValueError('Sparse workspace must fit the explicit CUDA budget')
    if execution_pipeline not in ('synchronous','overlapped'):raise ValueError('Unknown execution pipeline')
    if execution_pipeline=='synchronous' and device_cache_bytes:raise ValueError('Device cache requires explicit overlapped pipeline')
    if execution_pipeline=='synchronous' and sage_workspace_bytes!=64*2**20:raise ValueError('Explicit workspace change requires overlapped pipeline')
    if execution_pipeline=='overlapped' and activation_storage!='retained':raise ValueError('Measured pipeline requires retained activations')
    if execution_pipeline=='overlapped' and resume_support_minibatch:raise ValueError('Separate support learning migration from pipeline upgrade')
    if resume_support_minibatch and (not resume or support_policy is None or resume_execution_upgrade or resume_cuda_budget_change):
        raise ValueError('Support migration requires resume + explicit support patients, without execution/budget upgrade')
    if resume_cuda_budget_change and not resume_execution_upgrade:
        raise ValueError('CUDA budget migration requires explicit execution upgrade')
    if resume_execution_upgrade and not resume:raise ValueError('Execution upgrade requires a saved checkpoint')
    if debug_pause_step is not None and not debug:raise ValueError('Pause step is DEBUG only')
    if not candidates or min(candidates)<1 or candidates!=sorted(set(candidates)):raise ValueError('Explicit increasing physical batch candidates required')
    if resume_without_coarsening and (not resume or fine_cache is None or resume_execution_upgrade or resume_cuda_budget_change or resume_support_minibatch):
        raise ValueError('Coarsening removal requires fine cache + original checkpoint, with no other migration')
    if local_cnn:
        if fine_cache is not None or resume_without_coarsening or resume_execution_upgrade or resume_support_minibatch:raise ValueError('Local CNN is a new architecture; only matching local CNN resume allowed')
        from l0_local_cnn.data import Dataset as CNNDataset,Loader as CNNLoader
        from l0_regions.donor_learning import POLICY
        if learning_policy!=POLICY:raise ValueError('Local CNN retains same-donor live ranking')
        ds=CNNDataset(index,'inner_train',debug);val=CNNDataset(index,'inner_val',debug)
    elif fine_cache is None:
        ds=RegionDataset(index,'inner_train',debug,profile_policy);val=RegionDataset(index,'inner_val',debug,profile_policy)
    else:
        from .fine_graph import FineDataset,FineLoader,MODE
        if learning_policy is not None:
            from .donor_learning import POLICY
            if learning_policy!=POLICY or resume_without_coarsening or resume_support_minibatch:raise ValueError('Explicit new learning policy; no old-model transition')
            from .donor_data import DonorDataset as FineDataset,DonorLoader as FineLoader
        ds=FineDataset(index,'inner_train',debug,profile_policy,fine_cache);val=FineDataset(index,'inner_val',debug,profile_policy,fine_cache)
    if learning_policy is not None and fine_cache is None and not local_cnn:raise ValueError('Same-donor learning requires regenerated fine cache')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(workers);net=make_model(ds,budget,debug,activation_storage)
    base=ds.meta['base'];cfg=ds.meta['config'];epochs=1 if debug else cfg['gnn_epochs']
    if local_cnn:
        loader=CNNLoader(ds,workers,resident_bytes,budget.rss_bytes);val_loader=CNNLoader(val,workers,resident_bytes,budget.rss_bytes,store=loader.store)
    elif fine_cache is None:
        loader=Loader(ds,workers,resident_bytes);val_loader=Loader(val,workers,resident_bytes)
    else:
        loader=FineLoader(ds,workers,resident_bytes,rss_limit=budget.rss_bytes);val_loader=FineLoader(val,workers,resident_bytes,store=loader.store,rss_limit=budget.rss_bytes)
    optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
    identity=dict(format=FORMAT,debug=debug,cache_sha256=sha(index),source=source_identity(),epochs=epochs,
        config=cfg,base=base,precision='FP32',workers=workers,candidates=candidates,
        ranking=rank_config(ds),resident_budget_bytes=resident_bytes,profile_policy=profile_policy,
        activation_storage=activation_storage)
    identity['resource_limits']=dict(cuda_bytes=budget.cuda_bytes,rss_bytes=budget.rss_bytes)
    if local_cnn:identity.update(local_cnn=ds.meta['local_cnn'],graph_representation=ds.meta['local_cnn']['architecture'])
    if fine_cache is not None:identity.update(graph_representation=MODE,fine_cache_sha256=sha(fine_cache))
    if learning_policy is not None:identity['learning_policy']=learning_policy
    if support_policy is not None:identity['support_training']=support_policy
    if execution_pipeline=='overlapped':identity['execution_pipeline']=dict(mode=execution_pipeline,device_cache_bytes=device_cache_bytes,sage_workspace_bytes=sage_workspace_bytes,
        checkpoints='every update; immutable packed CPU snapshot; one ordered writer; flush on pause/phase/return',gradient_check='one finite decision per update')
    if resume:
        if resume_without_coarsening:
            from .fine_graph import transition
            saved,receipt=transition(resume,identity,net)
            write_new(root/'fine_graph_transition.json',receipt)
        else:
            saved=load_checkpoint(resume,identity,allow_execution_upgrade=resume_execution_upgrade,
                allow_cuda_budget_change=resume_cuda_budget_change,allow_support_migration=resume_support_minibatch)
        net.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        if saved.get('execution_upgrade'):write_new(root/'execution_upgrade.json',saved['execution_upgrade'])
        state=tree_to(saved['state'],'cuda');restore_rng(saved['rng']);del saved
        if resume_support_minibatch:write_new(root/'support_migration.json',state['support_migration'])
    else:
        with workspace(sage_workspace_bytes):
            batch,report,memory=calibrate(net,ds,loader,candidates,base,budget,root/'support_timing.jsonl',support_patients)
        write_new(root/'batch_calibration.json',report)
        # Calibration updated clones only. Its complete, ordered support belongs
        # to this exact initial model; do not discard it and run another cohort pass.
        state=dict(epoch=0,step=0,phase='optimization',memory=memory,memory_parts=[],memory_done=0,
            batch=batch,next_batch=0,plan=None,last_group=None,best=None,selected_epoch=None)
        if learning_policy is not None:
            state.update(phase='initial_validation',initial_validation=None,validation_history=[])
    net.local.dense_batch_size=state['batch']
    if learning_policy is not None:write_new(root/'learning_schedule.json',ranking_context(ds,state['memory'],state['batch']).audit)
    write_new(root/'execution_contract.json',dict(**{k:v for k,v in identity.items() if k!='graph_representation'},train_samples=len(ds),val_samples=len(val),usage_ratio=1.,
        physical_batch=state['batch'],effective_batch=state['batch'],accumulation=1,parameters=sum(p.numel() for p in net.parameters()),
        layers=dict(L0=8 if local_cnn else ds.meta['profile']['graph_encoder']['sage_layers'],L1=2,L2=2),hidden=128,cnn=[12,24,32],candidate_count=128,
        gpu=torch.cuda.get_device_name(),cpu_logical=psutil.cpu_count(),ram=psutil.virtual_memory()._asdict(),
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
        L0_parameters=sum(p.numel() for p in net.local.parameters()),input_shape='native donor bbox + explicit margin; padded unique CT crops [V,1,X,Y,Z]' if local_cnn else [state['batch'],1,48,48,48],
        graph_counts=[] if local_cnn else [dict(record=e['row']['id'],fine_nodes=e['fine_nodes'],region_nodes=e['region_nodes'],region_edges=e['region_edges']) for e in ds.entries] if fine_cache is None else
            [dict(record=e['row']['id'],fine_nodes=e['fine_nodes'],edge_upper_bound=e['row']['bounds']['edges'],scope='fixed-view node counts from original receipt; exact fine edges logged per consumed batch') for e in ds.entries],
        graph_representation=identity.get('graph_representation','fixed_regions'),
        optimization_steps=sum(1 for _ in groups(ds,state['batch']))*epochs,
        actual_query_batch_sizes=[len(ids) for ids in groups(ds,state['batch'])],
        cache_preparation_revision=ds.preparation_revision,cache_rebuilt=False,
        initial_support_policy='rebuild full support after explicit graph transition' if resume_without_coarsening else 'reuse saved memory or verified calibration support',
        scope='GNN only; final artifact connects to the existing CP recommendation adapter',
        profile_exceeded=bool(ds.meta['admission_failures']) if fine_cache is None and not local_cnn else None,partition_quality_validated=False,
        research_training=local_cnn or profile_policy=='research-report',production_admitted=not local_cnn and not debug and profile_policy=='strict'))
    print(json.dumps(dict(stage='region_training_policy',profile_policy=profile_policy,
        profile_violating_records=ds.meta['admission_failures'] if fine_cache is None and not local_cnn else None,partition_quality_validated=False,
        graph_representation=identity.get('graph_representation','fixed_regions'),coarsening_enabled=fine_cache is None and not local_cnn,
        debug=debug,epochs=epochs,activation_storage=activation_storage,
        activation_checkpointing=dict(CNN=net.local.cnn.checkpointing if local_cnn else net.local.core.checkpoint_dense_encoder,
            L0=False if local_cnn else net.local.core.checkpoint_local_blocks,L1_L2=net.checkpoint_support),
        physical_batch=state['batch'],resumed_step=state['step'],support_training=support_policy or 'full_support',
        cuda_limit_gib=budget.cuda_bytes/2**30,execution_pipeline=identity.get('execution_pipeline',dict(mode='synchronous')))),flush=True)
    device_cache=DeviceBatchCache(device_cache_bytes,budget) if execution_pipeline=='overlapped' else None
    checkpoint_io=CheckpointPipeline(root,execution_pipeline,save_checkpoint,hash_state,FORMAT)
    with checkpoint_io,pause_signal() as flag,workspace(sage_workspace_bytes):
        def save(wait=False):return checkpoint_io.save(net,optimizer,state,identity,wait=wait)
        def phase_timing(name,start,**details):
            with (root/'phase_timing.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(dict(phase=name,epoch=state['epoch']+1,step=state['step'],seconds=time.perf_counter()-start,execution_pipeline=execution_pipeline,**details))+'\n')
        save(wait=True)
        if learning_policy is not None:
            from .ranking_history import measurement,write_history,progress_line
            write_history(root,state['validation_history'])
        def pause():
            if flag['requested'] or (root/'STOP_AFTER_BATCH').exists() or (debug_pause_step is not None and state['step']>=debug_pause_step):
                save(wait=True)
                write_new(root/'paused.json',dict(step=state['step'],phase=state['phase'],debug=debug))
                print('PAUSED: checkpoint_latest.pt saved',flush=True);return True
            return False
        while True:
            budget.check()
            if state['phase']=='initial_validation':
                # Calibration support already belongs to the unchanged initial
                # model. Reuse it, and keep diagnostic evaluation out of RNG flow.
                if fine_cache is not None:
                    loader.release_batches();val_loader.retain_batches=False
                baseline_start=time.perf_counter();saved_rng=rng_state()
                try:metrics,details=evaluate(net,val,val_loader,state['memory'],state['batch'])
                finally:restore_rng(saved_rng)
                state['initial_validation']=metrics
                state['validation_history']=[measurement(0,0,metrics,metrics)]
                state['phase']='optimization'
                write_new(root/'validation_initial.json',dict(metrics=metrics,cases=details,debug=debug,
                    scope='Before optimizer updates; same full validation candidates as subsequent epochs'))
                save(wait=True);write_history(root,state['validation_history'])
                phase_timing('initial_validation',baseline_start)
                tqdm.write(validation_line(0,metrics,metrics['ranking_mrr'],False)+' | INITIAL baseline; no optimizer updates')
                if pause():return root/'checkpoint_latest.pt'
            if state['phase'] in ('initial_memory','refresh_memory','final_memory'):
                if fine_cache is not None:
                    if device_cache is not None:device_cache.clear()
                    loader.release_batches();loader.retain_batches=False
                phase_start=time.perf_counter();phase_name=state['phase']
                net.eval();done=state['memory_done'];order=list(contiguous(ds,state['batch']))
                with torch.no_grad():
                    for cpu in tqdm(loader.batches([ids for ids in order if ids[0]>=done]),desc=state['phase'],total=sum(ids[0]>=done for ids in order)):
                        memory_started=time.perf_counter()
                        ids=cpu.indices.tolist()
                        if ids!=list(range(state['memory_done'],state['memory_done']+len(ids))):raise ValueError('Memory cursor mismatch')
                        state['memory_parts'].append(net.local(cpu.to('cuda')).detach().float())
                        state['memory_done']+=len(ids);save()
                        with (root/'memory_timing.jsonl').open('a',encoding='utf8') as stream:
                            stream.write(json.dumps(dict(phase=state['phase'],completed=state['memory_done'],total=len(ds),
                                seconds=time.perf_counter()-memory_started,rss_bytes=psutil.Process().memory_info().rss,
                                input_cache=None if fine_cache is None else loader.store.cache.report(),
                                ram_guard=None if fine_cache is None else loader.last_ram_report))+'\n')
                        if pause():return root/'checkpoint_latest.pt'
                if state['memory_done']!=len(ds):raise ValueError('Incomplete support memory')
                state['memory']=metadata(ds,torch.cat(state['memory_parts']));state['memory_parts']=[];state['memory_done']=0
                if state['phase']=='final_memory':
                    if hash_state(net.state_dict())!=state['best']['model_sha256']:raise ValueError('Final memory is not bound to selected best model')
                    state['phase']='complete';save(wait=True)
                    phase_timing(phase_name,phase_start)
                    write_new(root/'training_complete.json',dict(debug=debug,epochs=epochs,steps=state['step'],selected_epoch=state['selected_epoch'],checkpoint_sha256=sha(root/'checkpoint_latest.pt'),nnunet_training=False,profile_policy=profile_policy,partition_quality_validated=False))
                    return root/'checkpoint_latest.pt'
                state['phase']='optimization' if state['phase']=='initial_memory' else 'validation'
                save(wait=True)
                phase_timing(phase_name,phase_start)
            if state['phase']=='optimization':
                if fine_cache is not None:loader.retain_batches=True
                optimization_start=time.perf_counter();starting_batch=state['next_batch']
                net.train();order=list(groups(ds,state['batch'],cfg['seed'],state['epoch']))
                counts=torch.bincount(state['memory']['classes'],minlength=2).float()
                if bool((counts==0).any()):raise ValueError('Both observation classes required')
                weights=counts.sum()/(2*counts);context=ranking_context(ds,state['memory'],state['batch'])
                episodes=None
                if support_policy is not None:
                    # Support selection still visits each observation once; query
                    # repeats required by exact live P/U tiling do not alter it.
                    episode_order=list(legacy_groups(ds,state['batch'],cfg['seed'],state['epoch']))
                    episodes=PatientEpisodes(ds.rows,episode_order,support_patients,cfg['seed'],state['epoch']).bind(state['memory'])
                    audit_path=root/f"support_epoch_{state['epoch']+1:03d}.json"
                    if not audit_path.exists():write_new(audit_path,dict(episodes.audit,starting_query_batch=state['next_batch'],
                        coverage_scope='complete deterministic epoch schedule; actual consumed prefix is checkpoint next_batch'))
                bar=tqdm(loader.batches(order[state['next_batch']:]),initial=state['next_batch'],total=len(order),desc=f"epoch {state['epoch']+1}/{epochs}")
                monitor=LearningMonitor(net)
                tqdm.write(f'Learning display: avg20=recent mean loss; grad=before clipping; probe=modules with sampled weight changes/{len(monitor.probes)}. Validation follows each epoch; loss window resets on resume.')
                previous_end=time.perf_counter()
                for cpu in bar:
                    load_wait=time.perf_counter()-previous_end;step_start=time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    budget.check();ids=cpu.indices.tolist()
                    if ids!=order[state['next_batch']]:raise ValueError('Saved query cursor mismatch')
                    group=ds.rows[ids[0]]['patient_group']
                    support_start=time.perf_counter()
                    support=(support_for_recipient(state['memory'],group) if episodes is None else episodes.support(group))
                    support_records=len(support[0])
                    support_groups=(int(support[1].max())+1 if episodes is None else len(episodes.selections[group]['patients']))
                    if state['last_group']!=group:
                        net.eval()
                        with torch.no_grad():state['plan']=net.fit_support_clusters(*support)
                        net.train();state['last_group']=group
                    support_preparation_seconds=time.perf_counter()-support_start
                    optimizer.zero_grad(set_to_none=True)
                    events=[torch.cuda.Event(enable_timing=True) for _ in range(5)]
                    events[0].record();query=cpu.to('cuda') if device_cache is None else device_cache.get(cpu)
                    if device_cache is not None:device_cache.activate(net,cpu)
                    events[1].record()
                    loss,terms=forward_loss(net,query,support,state['plan'],state['memory']['classes'][ids],weights,context,rank_config(ds),indices=ids)
                    events[2].record()
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward();events[3].record()
                    (gradient_check if execution_pipeline=='synchronous' else gradient_check_batched)(net)
                    grad_norm=torch.nn.utils.clip_grad_norm_(net.parameters(),base['training']['grad_clip'],error_if_nonfinite=True)
                    before=monitor.before_step();optimizer.step()
                    events[4].record();events[4].synchronize()
                    health=monitor.after_step(loss,terms,grad_norm,before,optimizer.param_groups[0]['lr'])
                    if device_cache is not None:device_cache.remember(net,cpu)
                    state['next_batch']+=1;state['step']+=1
                    loss_value=float(loss.detach());peak=torch.cuda.max_memory_allocated()
                    del loss,terms,support,query
                    save_start=time.perf_counter()
                    checkpoint_stats=save()
                    save_seconds=time.perf_counter()-save_start
                    peak=torch.cuda.max_memory_allocated();budget.check()
                    row=dict(step=state['step'],epoch=state['epoch']+1,physical_batch=len(ids),
                        fine_graph_counts=None if fine_cache is None else dict(nodes=cpu.graph.sampled_counts.tolist(),edges=cpu.graph.relation_edge_counts.tolist()),
                        learning=health,
                        activation_storage=activation_storage,loss=loss_value,loader_wait_seconds=load_wait,
                        support_records=support_records,support_patients=support_groups,support_policy=support_policy,
                        support_preparation_seconds=support_preparation_seconds,
                        checkpoint_pipeline=checkpoint_stats,execution_pipeline=execution_pipeline,
                        device_cache=None if device_cache is None else dict(bytes=device_cache.bytes,hits=device_cache.hits,misses=device_cache.misses,limit_bytes=device_cache.max_bytes),
                        transfer_seconds=events[0].elapsed_time(events[1])/1000,
                        forward_seconds=events[1].elapsed_time(events[2])/1000,
                        backward_seconds=events[2].elapsed_time(events[3])/1000,
                        check_clip_optimizer_seconds=events[3].elapsed_time(events[4])/1000,
                        checkpoint_seconds=save_seconds,step_seconds=time.perf_counter()-step_start,
                        peak_cuda_bytes=peak,rss_bytes=psutil.Process().memory_info().rss)
                    with (root/'update_timing.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
                    bar.set_postfix(**monitor.postfix(health,state['step']),sec=round(row['step_seconds'],2),GiB=round(peak/2**30,2))
                    if pause():return root/'checkpoint_latest.pt'
                    previous_end=time.perf_counter()
                state['phase']='refresh_memory';save(wait=True)
                phase_timing('optimization',optimization_start,completed_updates=len(order)-starting_batch,starting_batch=starting_batch,includes_final_checkpoint_flush=True)
            if state['phase']=='validation':
                if fine_cache is not None:
                    if device_cache is not None:device_cache.clear()
                    loader.release_batches();val_loader.retain_batches=False
                validation_start=time.perf_counter()
                metrics,details=evaluate(net,val,val_loader,state['memory'],state['batch'])
                phase_timing('validation',validation_start)
                write_new(root/f"epoch_{state['epoch']+1:03d}.json",dict(metrics=metrics,cases=details,debug=debug))
                score=metrics['ranking_pairwise_loss']
                improved=state['best'] is None or score<state['best']['metric']
                if learning_policy is not None:
                    from .donor_learning import selection
                    key=selection(metrics);score=metrics['ranking_mrr']
                    improved=state['best'] is None or key>tuple(state['best']['selection_key'])
                    row=measurement(state['epoch']+1,state['step'],metrics,state['initial_validation'])
                    state['validation_history'].append(row)
                if improved:
                    weights=tree_to(net.state_dict(),'cpu');state['best']=dict(weights=weights,metric=score,epoch=state['epoch']+1,model_sha256=hash_state(weights))
                    if learning_policy is not None:state['best']['selection_key']=list(key)
                tqdm.write(validation_line(state['epoch']+1,metrics,state['best']['metric'],improved)+(' | best metric=MRR, tie=R@1, rank_loss' if learning_policy is not None else ''))
                state.update(epoch=state['epoch']+1,next_batch=0,last_group=None,plan=None,phase='optimization')
                if state['epoch']==epochs:
                    net.load_state_dict(state['best']['weights']);state['selected_epoch']=state['best']['epoch'];state['phase']='final_memory'
                save(wait=True)
                if learning_policy is not None:
                    write_history(root,state['validation_history']);tqdm.write(progress_line(row))
                if pause():return root/'checkpoint_latest.pt'
            if state['phase']=='complete':raise ValueError('Completed runs need no resume')
