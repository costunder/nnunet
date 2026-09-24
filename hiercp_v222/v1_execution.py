"""Resumable execution; model, observations, loss and graph rules are unchanged."""
from pathlib import Path
import csv
import json
import os
import random
import time
import uuid
import numpy as np
import psutil
import torch
from .contracts import write_new,sha
from .v1_cache import PairDataset,PairLoader,configuration,provenance,emit,save_torch_new
from .v1_local import model,support_for_recipient
from .model import supervised_loss
from .training import configure_runtime,require_device

RESUME_FORMAT='v222_v1_exact_execution_resume_v1'

def tree_to(value,device):
    if torch.is_tensor(value):return value.detach().to(device)
    if isinstance(value,dict):return {k:tree_to(v,device) for k,v in value.items()}
    if isinstance(value,list):return [tree_to(v,device) for v in value]
    if isinstance(value,tuple):return tuple(tree_to(v,device) for v in value)
    return value

def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(),python=random.getstate())

def restore_rng(state):
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])

def atomic_torch(path,value):
    """Keep the previous valid checkpoint until a fully flushed replacement exists."""
    path=Path(path);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('xb') as f:
            torch.save(value,f);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink() # Only this call's explicitly owned temp.

def memory_metrics():
    stats=torch.cuda.memory_stats()
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        inactive_split=stats.get('inactive_split_bytes.all.current',0),
        allocation_retries=stats.get('num_alloc_retries',0),rss=psutil.Process().memory_info().rss)

def optimizer_step(net,optimizer,payload,support,plan,targets,weights,grad_clip,*,check_gradients=False):
    """CUDA events cover real transfers and full forward/backward/optimizer work."""
    from .v1_training import gradient_check
    events=[torch.cuda.Event(enable_timing=True) for _ in range(5)]
    torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
    events[0].record();query=payload.cuda(non_blocking=True);events[1].record()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result=net(query,*support,cluster_plan=plan)
        loss=supervised_loss(result,targets,weights)
    events[2].record();loss.backward();events[3].record()
    check=gradient_check(net) if check_gradients else None
    torch.nn.utils.clip_grad_norm_(net.parameters(),grad_clip,error_if_nonfinite=True)
    optimizer.step();events[4].record();events[4].synchronize()
    timing={key:events[i].elapsed_time(events[i+1])/1000
            for i,key in enumerate(('H2D_seconds','forward_seconds','backward_seconds','optimizer_seconds'))}
    values=dict(loss=float(loss.detach()),alignment_loss=float(result['alignment_loss'].detach()))
    usage=memory_metrics()
    # Release graph references before the next variable-size batch is allocated.
    del result,loss,query
    optimizer.zero_grad(set_to_none=True)
    return values,timing,usage,check

class Saver:
    def __init__(self,root,net,optimizer,identity):
        self.root=Path(root);self.net=net;self.optimizer=optimizer;self.identity=identity
        # A new run owns its rolling checkpoint; a resume always uses a new run.
        if (self.root/'checkpoint_latest.pt').exists():raise FileExistsError('New checkpoint owner required')
    def save(self,state):
        start=time.perf_counter()
        payload=dict(format=RESUME_FORMAT,**self.identity,state=tree_to(state,'cpu'),
            model=tree_to(self.net.state_dict(),'cpu'),optimizer=tree_to(self.optimizer.state_dict(),'cpu'),rng=rng_state())
        atomic_torch(self.root/'checkpoint_latest.pt',payload)
        receipt=dict(phase=state['phase'],epoch=state['epoch']+1,step=state['step'],
            next_batch=state.get('next_batch',0),memory_completed=state.get('memory_next',0),
            seconds=time.perf_counter()-start,bytes=(self.root/'checkpoint_latest.pt').stat().st_size,
            saved_at=time.time(),path=str(self.root/'checkpoint_latest.pt'))
        temporary=self.root/'checkpoint_status.tmp'
        temporary.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        os.replace(temporary,self.root/'checkpoint_status.json')
        return receipt
    def stop_requested(self):return (self.root/'STOP_AFTER_BATCH').exists()

def fresh_epoch(state):
    state.update(next_batch=0,seen=[],losses=[],alignment=[],last_group=None,plan=None,audits={},epoch_seconds=0.)

def memory_metadata(dataset,embeddings):
    cases=sorted({r['case_id'] for r in dataset.rows});lookup={c:i for i,c in enumerate(cases)}
    if not set(cases)<=set(dataset.meta['split']['inner_train']):raise ValueError('Validation entered memory')
    return dict(embeddings=embeddings,owners=torch.tensor([lookup[r['case_id']] for r in dataset.rows],device='cuda'),
        classes=torch.tensor([r['target'] for r in dataset.rows],device='cuda'),case_ids=cases,
        patient_groups=[dataset.meta['identities']['cases'][c]['patient_group'] for c in cases],
        donor_groups=[r['donor_group'] for r in dataset.rows],record_ids=[r['id'] for r in dataset.rows])

@torch.no_grad()
def encode_memory(net,dataset,state,saver,batch,workers,release_unused):
    """Checkpoint partial support encoding too; no hour-long unsaved preparation."""
    from .v1_training import contiguous
    net.eval();net.local.dense_batch_size=batch
    if state.get('memory_work') is None:
        # Only the completed prefix is serialized. Uninitialized rows never enter a checkpoint/model.
        embeddings=torch.empty((len(dataset),128),device='cuda',dtype=torch.float32)
        done=0
    else:
        done=state['memory_next'];embeddings=torch.empty((len(dataset),128),device='cuda',dtype=torch.float32)
        embeddings[:done]=state['memory_work']
    loader=PairLoader(dataset,workers);start=time.perf_counter();initial=done
    try:
        for payload in loader.batches((ids for ids in contiguous(dataset,batch) if ids[0]>=done),epoch=0):
            ids=payload.indices.tolist()
            if ids!=list(range(done,done+len(ids))):raise RuntimeError('Non-contiguous memory resume')
            with torch.autocast('cuda',dtype=torch.bfloat16):values=net.local(payload.cuda(non_blocking=True))
            embeddings[ids]=values.float();done+=len(ids);del values,payload
            state['memory_work']=embeddings[:done].clone();state['memory_next']=done
            receipt=saver.save(state)
            if release_unused:torch.cuda.empty_cache()
            emit(stage='support_memory',completed=done,total=len(dataset),physical_batch=batch,
                graphs_per_second=(done-initial)/(time.perf_counter()-start),checkpoint_seconds=receipt['seconds'],**memory_metrics())
            if saver.stop_requested():return False
    finally:loader.close()
    if done!=len(dataset):raise RuntimeError('Incomplete memory')
    state['memory']=memory_metadata(dataset,embeddings)
    state['memory_work']=None;state['memory_next']=0
    return True

def train(cache,output,*,resume=None,calibration=None,release_unused=True,debug=False):
    from .v1_training import groups,calibrate,calibrate_workers,evaluate,CHECKPOINT_FORMAT
    cfg,base=configuration();require_device('cuda');configure_runtime(base,cfg['seed'])
    torch.set_num_threads(psutil.cpu_count(logical=False))
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    dataset=PairDataset(cache,'inner_train');validation=PairDataset(cache,'inner_val')
    if dataset.meta['config']!=cfg or dataset.meta['base']!=base:raise ValueError('Cache config mismatch')
    net=model(cfg,base).cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
    identity=dict(config=cfg,base=base,source_identity=provenance(),cache_sha256=sha(cache),debug=bool(debug))
    write_new(root/'initialization.json',dict(**identity,parameters=sum(p.numel() for p in net.parameters()),
        train_samples=len(dataset),validation_samples=len(validation),subset=False,gpu=torch.cuda.get_device_name()))
    loaded=None
    if resume:
        loaded=torch.load(resume,map_location='cpu',weights_only=False)
        if loaded['format']!=RESUME_FORMAT or any(loaded.get(k)!=v for k,v in identity.items()):
            raise ValueError('Resume source/cache/config mismatch; no silent state conversion')
        net.load_state_dict(loaded['model']);optimizer.load_state_dict(loaded['optimizer'])
        state=tree_to(loaded['state'],'cuda')
        if bool(state['release_unused'])!=release_unused:raise ValueError('Resume must retain measured allocator policy')
        write_new(root/'resume_from.json',dict(path=str(Path(resume).resolve()),sha256=sha(resume),
            epoch=state['epoch']+1,step=state['step'],phase=state['phase'],next_batch=state['next_batch']))
    else:
        state=dict(phase='initial_memory',epoch=0,step=0,memory=None,memory_work=None,memory_next=0,
            best=float('inf'),best_path=None,release_unused=release_unused,training_calibrated=False)
        fresh_epoch(state)
        if calibration:
            # Execution-only transfer changes permit use of previously measured physical batch.
            previous=Path(calibration);old=json.loads((previous/'training_started.json').read_text(encoding='utf-8'))
            if old['config']!=cfg or old['base']!=base or old['parameters']!=sum(p.numel() for p in net.parameters()):
                raise ValueError('Calibration architecture mismatch')
            if old['train_samples']!=len(dataset) or old['val_samples']!=len(validation):raise ValueError('Calibration cohort mismatch')
            for filename in ('loader_calibration.json','training_batch_calibration.json','memory_batch_calibration.json'):
                write_new(root/filename,json.loads((previous/filename).read_text(encoding='utf-8')))
            state.update(batch=old['physical_batch'],memory_batch=old['memory_physical_batch'],workers=old['workers'],training_calibrated=True)
            write_new(root/'calibration_reuse.json',dict(previous=str(previous.resolve()),
                scope='same full model/graphs/cohort and physical batch; consecutive execution timings recorded anew',
                report_sha256=sha(previous/'training_started.json')))
        else:
            ids=sorted(range(len(dataset)),key=lambda i:dataset.rows[i]['bounds']['edges'],reverse=True)[:psutil.cpu_count()]
            state['workers']=calibrate_workers(dataset,ids,root)
            state['memory_batch'],_=calibrate(net,dataset,cfg,state['workers'],root)
            state['batch']=None
    saver=Saver(root,net,optimizer,identity)
    if loaded is not None:restore_rng(loaded['rng']);del loaded
    saver.save(state)
    report=dict(format=CHECKPOINT_FORMAT,**identity,parameters=sum(p.numel() for p in net.parameters()),
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),L0_parameters=sum(p.numel() for p in net.local.parameters()),
        physical_batch=state['batch'],effective_batch=state['batch'],gradient_accumulation=1,
        memory_physical_batch=state['memory_batch'],epochs=cfg['gnn_epochs'],train_samples=len(dataset),train_usage_ratio=1.,val_samples=len(validation),
        workers=state['workers'],worker_kind='persistent threads',prefetch_batches=1,pin_memory=True,
        precision='bfloat16 autocast; FP32 loss/optimizer',input_shape=[state['batch'],1,48,48,48],input_branches=['donor','recipient'],
        layers=dict(L0=3,L1=2,L2=2),hidden=128,heads=4,cnn_channels=[12,24,32],sampling=base['graph'],
        gpu=torch.cuda.get_device_name(),gpu_count=1,cpu_threads=torch.get_num_threads(),ram=psutil.virtual_memory()._asdict(),
        subset=False,fast_mode=False,changed_architecture=False,nnunet_online_adapter_ready=False,
        checkpoint='atomic replacement after every optimizer/support batch; exact RNG, optimizer and episode plan',
        release_unused_allocator_cache=release_unused,optimizer_started=False)
    write_new(root/'execution_contract.json',report)
    def stop():
        if not saver.stop_requested():return False
        write_new(root/'paused.json',dict(step=state['step'],epoch=state['epoch']+1,phase=state['phase'],checkpoint='checkpoint_latest.pt'))
        emit(stage='paused',step=state['step'],checkpoint=str(root/'checkpoint_latest.pt'));return True
    while state['epoch']<cfg['gnn_epochs']:
        if state['phase'] in ('initial_memory','refresh_memory'):
            initial=state['phase']=='initial_memory'
            if not encode_memory(net,dataset,state,saver,state['memory_batch'],state['workers'],release_unused):
                stop();return root/'checkpoint_latest.pt'
            if initial:
                if not state['training_calibrated']:
                    state['batch'],_=calibrate(net,dataset,cfg,state['workers'],root,state['memory'])
                    state['training_calibrated']=True
                configure_runtime(base,cfg['seed'])
                state['phase']='optimization'
            else:state['phase']='validation'
            saver.save(state)
        batch=state['batch'];workers=state['workers']
        if state['phase']=='optimization':
            total_steps=sum(1 for _ in groups(dataset,batch))*cfg['gnn_epochs']
            if not (root/'training_started.json').exists():
                write_new(root/'training_started.json',report|dict(physical_batch=batch,effective_batch=batch,
                    input_shape=[batch,1,48,48,48],dense_execution_chunk=batch,optimization_steps=total_steps))
            net.train();net.local.dense_batch_size=batch
            counts=torch.bincount(state['memory']['classes'],minlength=2).float()
            if bool((counts==0).any()):raise ValueError('Both observation classes required')
            weights=counts.sum()/(2*counts)
            order=list(groups(dataset,batch,cfg['seed'],state['epoch']))
            expected={i for ids in order[:state['next_batch']] for i in ids}
            if set(state['seen'])!=expected:raise ValueError('Resume query coverage disagrees with saved batch cursor')
            loader=PairLoader(dataset,workers);iterator=iter(loader.batches(order[state['next_batch']:],epoch=state['epoch']))
            support=None
            try:
                for position in range(state['next_batch'],len(order)):
                    started=time.perf_counter();payload=next(iterator);load_done=time.perf_counter()
                    ids=payload.indices.tolist();group=dataset.rows[ids[0]]['patient_group']
                    if ids!=order[position] or any(dataset.rows[i]['patient_group']!=group for i in ids):raise ValueError('Query order/group changed')
                    if group!=state['last_group']:
                        support=support_for_recipient(state['memory'],group)
                        with torch.autocast('cuda',dtype=torch.bfloat16):state['plan']=net.fit_support_clusters(*support)
                        state['audits'][group]=state['plan']['audit'];state['last_group']=group
                    elif support is None:support=support_for_recipient(state['memory'],group)
                    plan_done=time.perf_counter()
                    targets=torch.tensor([dataset.rows[i]['target'] for i in ids],device='cuda')
                    values,timing,usage,check=optimizer_step(net,optimizer,payload,support,state['plan'],targets,weights,
                        base['training']['grad_clip'],check_gradients=state['step']==0)
                    if check:write_new(root/'first_gradient_check.json',check)
                    nodes,edges=int(payload.graph.num_nodes),int(payload.graph.num_edges)
                    del payload,targets
                    state['step']+=1;state['next_batch']=position+1;state['seen'].extend(ids)
                    state['losses'].append(values['loss']);state['alignment'].append(values['alignment_loss'])
                    state['epoch_seconds']+=time.perf_counter()-started
                    receipt=saver.save(state)
                    if release_unused:torch.cuda.empty_cache()
                    seconds=time.perf_counter()-started
                    progress=dict(stage='optimization',epoch=state['epoch']+1,step=state['step'],total_steps=total_steps,
                        visited_queries=len(state['seen']),total_queries=len(dataset),physical_batch=len(ids),nodes=nodes,edges=edges,
                        **values,**timing,**usage,loader_wait_seconds=load_done-started,support_plan_seconds=plan_done-load_done,
                        checkpoint_seconds=receipt['seconds'],step_seconds=seconds,graphs_per_second=len(ids)/seconds,
                        checkpoint_step=state['step'])
                    with (root/'step_timings.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(progress)+'\n')
                    emit(**progress)
                    if not (root/'first_optimizer_step.json').exists():write_new(root/'first_optimizer_step.json',progress)
                    if stop():return root/'checkpoint_latest.pt'
            finally:loader.close()
            if set(state['seen'])!=set(range(len(dataset))) or len(state['seen'])!=len(dataset):raise RuntimeError('Incomplete/duplicate epoch')
            state['phase']='refresh_memory';saver.save(state)
            continue
        if state['phase']=='validation':
            net.local.dense_batch_size=batch
            metrics=evaluate(net,validation,state['memory'],batch,workers,
                batch_complete=torch.cuda.empty_cache if release_unused else None)
            row=dict(epoch=state['epoch']+1,optimization_steps=state['step'],train_loss=float(np.mean(state['losses'])),
                alignment_loss=float(np.mean(state['alignment'])),validation=metrics,all_train_queries_visited=len(state['seen']),
                seconds=state['epoch_seconds'],clusters=state['audits'],**memory_metrics())
            write_new(root/f'epoch_{state["epoch"]+1:03d}.json',row)
            with (root/'metrics.csv').open('a',newline='',encoding='utf-8') as f:
                flat={k:row[k] for k in ('epoch','optimization_steps','train_loss','alignment_loss','all_train_queries_visited','seconds')}
                flat.update(metrics);writer=csv.DictWriter(f,fieldnames=list(flat))
                if f.tell()==0:writer.writeheader()
                writer.writerow(flat)
            path=root/f'epoch_{state["epoch"]+1:03d}.pt'
            save_torch_new(path,dict(format=CHECKPOINT_FORMAT,completed_epochs=state['epoch']+1,
                state_dict=tree_to(net.state_dict(),'cpu'),optimizer=tree_to(optimizer.state_dict(),'cpu'),
                epoch=state['epoch']+1,step=state['step'],**identity))
            if metrics['observation_cross_entropy']<state['best']:
                state['best']=metrics['observation_cross_entropy'];state['best_path']=str(path)
            state['epoch']+=1;fresh_epoch(state);state['phase']='optimization'
            saver.save(state);emit(stage='epoch_complete',**{k:v for k,v in row.items() if k!='clusters'})
            if stop():return root/'checkpoint_latest.pt'
    if state['phase']!='final_memory':
        best=torch.load(state['best_path'],map_location='cpu',weights_only=False)
        net.load_state_dict(best['state_dict']);state['selected_epoch']=best['epoch']
        state.update(phase='final_memory',memory_work=None,memory_next=0);saver.save(state)
    if not encode_memory(net,dataset,state,saver,state['memory_batch'],state['workers'],release_unused):
        stop();return root/'checkpoint_latest.pt'
    payload=dict(format=CHECKPOINT_FORMAT,debug=bool(debug),completed_epochs=cfg['gnn_epochs'],selected_epoch=state['selected_epoch'],
        state_dict=tree_to(net.state_dict(),'cpu'),config=cfg,base=base,split=dataset.meta['split'],identities=dataset.meta['identities'],
        memory=tree_to(state['memory'],'cpu'),donor_pool=dataset.meta['donor_pool'],raw_records=dataset.meta['raw_records'],
        source_identity=provenance(),cache_sha256=sha(cache),physical_batch=state['batch'],workers=state['workers'],
        optimization_steps=state['step'],native_online_bank_integrated=False)
    save_torch_new(root/'checkpoint.pt',payload)
    write_new(root/'training_complete.json',dict(debug=bool(debug),epochs=cfg['gnn_epochs'],selected_epoch=state['selected_epoch'],
        checkpoint_sha256=sha(root/'checkpoint.pt'),segmentation_training_executed=False))
    return root/'checkpoint.pt'
