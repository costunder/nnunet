"""v2.2 ranking training; preserved full execution loop with an explicit new loss.

Based on hiercp_v222/v1_execution.py at 6480715. Calls its installed process
loader, durable saver and memory encoder; no forked graph/cache implementation.
Changes: rank loss, ranking validation/model selection, objective identity.
"""
from pathlib import Path
import csv
import json
import time
import gc
import numpy as np
import psutil
import torch
from hiercp_v222 import v1_execution as ex
from tools.v22_rank_objective import OBJECTIVE,configuration,RankingContext,forward_loss,ranking_metrics
from tools.v22_ranking_steps import optimizer_step,evaluate
RESUME_FORMAT=ex.RESUME_FORMAT

def calibrate(net,dataset,cfg,workers,root,memory=None):
    from hiercp_v222.v1_training import execution_budget,batch_admission,gradient_check
    training=memory is not None
    settings=configuration();context=RankingContext(dataset,memory) if training else None
    counts={}
    for row in dataset.rows:counts[row['patient_group']]=counts.get(row['patient_group'],0)+1
    maximum=max(counts.values())
    if training:
        # Use the case with highest canonical graph mass, not a uniform-graph assumption.
        group=max(counts,key=lambda g:sum(r['bounds']['edges'] for r in dataset.rows if r['patient_group']==g))
        indices=sorted([i for i,r in enumerate(dataset.rows) if r['patient_group']==group],
                       key=lambda i:dataset.rows[i]['bounds']['edges'],reverse=True)
        support=ex.support_for_recipient(memory,group)
        with torch.autocast('cuda',dtype=torch.bfloat16):plan=net.fit_support_clusters(*support)
    else:
        indices=sorted(range(len(dataset)),key=lambda i:dataset.rows[i]['bounds']['edges'],reverse=True)[:maximum]
    candidates=[2]
    while candidates[-1]<len(indices):candidates.append(min(2*candidates[-1],len(indices)))
    reports=[];loader=ex.PairLoader(dataset,workers)
    net.train(training)
    try:
        for count in candidates:
            net.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            baseline,budget=execution_budget(cfg)
            admission=batch_admission(count,reports,baseline,budget)
            if not admission['admitted']:
                reports.append(dict(physical_batch=count,accepted=False,executed=False,**admission));break
            cpu=loader.make(indices[:count],0);query=loss=output=None
            # Execution chunk only: CNN sees the full measured physical batch.
            net.local.dense_batch_size=count
            try:
                query=cpu.cuda(non_blocking=True)
                labels=torch.tensor([dataset.rows[i]['target'] for i in indices[:count]],device='cuda')
                def step():
                    with torch.set_grad_enabled(training),torch.autocast('cuda',dtype=torch.bfloat16):
                        if training:
                            value,_=forward_loss(net,query,support,plan,labels,None,context,settings,indices=indices[:count])
                        else:value=net.local(query)
                    if training:value.backward()
                    return value
                output=step();del output;output=None;net.zero_grad(set_to_none=True);torch.cuda.synchronize()
                start=time.perf_counter()
                for _ in range(cfg['batch_calibration_repeats']):
                    net.zero_grad(set_to_none=True);output=step();del output;output=None
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                if training:gradient_check(net)
                peak=torch.cuda.max_memory_allocated()
                report=dict(physical_batch=count,graphs_per_second=count*cfg['batch_calibration_repeats']/elapsed,
                    peak_vram_bytes=peak,accepted=peak<budget,executed=True,budget_bytes=budget,
                    nodes=int(cpu.graph.num_nodes),edges=int(cpu.graph.num_edges),training=training)
                reports.append(report)
            except torch.cuda.OutOfMemoryError as error:
                reports.append(dict(physical_batch=count,accepted=False,error=str(error),
                    policy='execution batch admission only; full graph/model/cohort retained'))
            finally:
                query=loss=output=None;del cpu;net.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            ex.emit(stage='paired_training_calibration' if training else 'paired_memory_calibration',**reports[-1])
            if not reports[-1]['accepted']:break
    finally:loader.close()
    name='training_batch' if training else 'memory_batch'
    ex.write_new(root/f'{name}_calibration.json',reports)
    accepted=[r for r in reports if r['accepted']]
    if not accepted:raise MemoryError('Full paired graph batch does not fit; inspect memory report before changing execution')
    selected=max(accepted,key=lambda r:r['graphs_per_second'])['physical_batch']
    net.local.dense_batch_size=selected
    return selected,reports


def train(cache,output,*,resume=None,calibration=None,release_unused=True,debug=False):
    from hiercp_v222.v1_training import groups,calibrate_workers,CHECKPOINT_FORMAT
    cfg,base=ex.configuration();ex.require_device('cuda');ex.configure_runtime(base,cfg['seed'])
    torch.set_num_threads(psutil.cpu_count(logical=False))
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    dataset=ex.PairDataset(cache,'inner_train');validation=ex.PairDataset(cache,'inner_val')
    if dataset.meta['config']!=cfg or dataset.meta['base']!=base:raise ValueError('Cache config mismatch')
    net=ex.model(cfg,base).cuda()
    if getattr(net.local,'feature_coordinate_contract',None)!='stride4':
        raise ValueError('Ranking trainer requires the corrected stride4 encoder adapter')
    optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
    settings=configuration()
    identity=dict(training_objective=OBJECTIVE,ranking_contract=settings,feature_coordinates='stride4',
                  support_task_contract='patient_group_v1',config=cfg,base=base,source_identity=ex.provenance(),cache_sha256=ex.sha(cache),debug=bool(debug))
    ex.write_new(root/'initialization.json',dict(**identity,parameters=sum(p.numel() for p in net.parameters()),
        train_samples=len(dataset),validation_samples=len(validation),subset=bool(debug),gpu=torch.cuda.get_device_name()))
    loaded=None
    if resume:
        loaded=torch.load(resume,map_location='cpu',weights_only=False)
        if loaded['format']!=RESUME_FORMAT or any(loaded.get(k)!=v for k,v in identity.items()):
            raise ValueError('Resume source/cache/config mismatch; no silent state conversion')
        net.load_state_dict(loaded['model']);optimizer.load_state_dict(loaded['optimizer'])
        state=ex.tree_to(loaded['state'],'cuda')
        if bool(state['release_unused'])!=release_unused:raise ValueError('Resume must retain measured allocator policy')
        ex.write_new(root/'resume_from.json',dict(path=str(Path(resume).resolve()),sha256=ex.sha(resume),
            epoch=state['epoch']+1,step=state['step'],phase=state['phase'],next_batch=state['next_batch']))
    else:
        state=dict(phase='initial_memory',epoch=0,step=0,memory=None,memory_work=None,memory_next=0,
            best=float('inf'),best_path=None,release_unused=release_unused,training_calibrated=False)
        ex.fresh_epoch(state)
        if calibration:
            # Execution-only transfer changes permit use of previously measured physical batch.
            previous=Path(calibration);old=json.loads((previous/'training_started.json').read_text(encoding='utf-8'))
            if old.get('ranking_contract')!=settings:raise ValueError('Ranking needs matching loss calibration')
            if old['config']!=cfg or old['base']!=base or old['parameters']!=sum(p.numel() for p in net.parameters()):
                raise ValueError('Calibration architecture mismatch')
            if old['train_samples']!=len(dataset) or old['val_samples']!=len(validation):raise ValueError('Calibration cohort mismatch')
            for filename in ('loader_calibration.json','training_batch_calibration.json','memory_batch_calibration.json'):
                ex.write_new(root/filename,json.loads((previous/filename).read_text(encoding='utf-8')))
            state.update(batch=old['physical_batch'],memory_batch=old['memory_physical_batch'],workers=old['workers'],training_calibrated=True)
            ex.write_new(root/'calibration_reuse.json',dict(previous=str(previous.resolve()),
                scope='same full model/graphs/cohort and physical batch; consecutive execution timings recorded anew',
                report_sha256=ex.sha(previous/'training_started.json')))
        else:
            ids=sorted(range(len(dataset)),key=lambda i:dataset.rows[i]['bounds']['edges'],reverse=True)[:psutil.cpu_count()]
            state['workers']=calibrate_workers(dataset,ids,root)
            state['memory_batch'],_=calibrate(net,dataset,cfg,state['workers'],root)
            state['batch']=None
    saver=ex.Saver(root,net,optimizer,identity)
    if loaded is not None:ex.restore_rng(loaded['rng']);del loaded
    saver.save(state)
    report=dict(format=CHECKPOINT_FORMAT,**identity,parameters=sum(p.numel() for p in net.parameters()),
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),L0_parameters=sum(p.numel() for p in net.local.parameters()),
        physical_batch=state['batch'],effective_batch=state['batch'],gradient_accumulation=1,
        memory_physical_batch=state['memory_batch'],epochs=cfg['gnn_epochs'],train_samples=len(dataset),train_usage_ratio=1.,val_samples=len(validation),
        workers=state['workers'],worker_kind='persistent threads',prefetch_batches=1,pin_memory=True,
        precision='bfloat16 autocast; FP32 loss/optimizer',input_shape=[state['batch'],1,48,48,48],input_branches=['donor','recipient'],
        layers=dict(L0=3,L1=2,L2=2),hidden=128,heads=4,cnn_channels=[12,24,32],sampling=base['graph'],
        gpu=torch.cuda.get_device_name(),gpu_count=1,cpu_threads=torch.get_num_threads(),ram=psutil.virtual_memory()._asdict(),
        subset=bool(debug),fast_mode=False,changed_architecture=False,nnunet_online_adapter_ready=False,
        checkpoint='atomic replacement after every optimizer/support batch; exact RNG, optimizer and episode plan',
        release_unused_allocator_cache=release_unused,optimizer_started=False)
    ex.write_new(root/'execution_contract.json',report)
    def stop():
        if not saver.stop_requested():return False
        ex.write_new(root/'paused.json',dict(step=state['step'],epoch=state['epoch']+1,phase=state['phase'],checkpoint='checkpoint_latest.pt'))
        ex.emit(stage='paused',step=state['step'],checkpoint=str(root/'checkpoint_latest.pt'));return True
    while state['epoch']<cfg['gnn_epochs']:
        if state['phase'] in ('initial_memory','refresh_memory'):
            initial=state['phase']=='initial_memory'
            if not ex.encode_memory(net,dataset,state,saver,state['memory_batch'],state['workers'],release_unused):
                stop();return root/'checkpoint_latest.pt'
            if initial:
                if not state['training_calibrated']:
                    state['batch'],_=calibrate(net,dataset,cfg,state['workers'],root,state['memory'])
                    state['training_calibrated']=True
                ex.configure_runtime(base,cfg['seed'])
                state['phase']='optimization'
            else:state['phase']='validation'
            saver.save(state)
        batch=state['batch'];workers=state['workers']
        if state['phase']=='optimization':
            total_steps=sum(1 for _ in groups(dataset,batch))*cfg['gnn_epochs']
            if not (root/'training_started.json').exists():
                ex.write_new(root/'training_started.json',report|dict(physical_batch=batch,effective_batch=batch,
                    input_shape=[batch,1,48,48,48],dense_execution_chunk=batch,optimization_steps=total_steps))
            net.train();net.local.dense_batch_size=batch
            rank_context=RankingContext(dataset,state['memory'])
            counts=torch.bincount(state['memory']['classes'],minlength=2).float()
            if bool((counts==0).any()):raise ValueError('Both observation classes required')
            weights=counts.sum()/(2*counts)
            order=list(groups(dataset,batch,cfg['seed'],state['epoch']))
            expected={i for ids in order[:state['next_batch']] for i in ids}
            if set(state['seen'])!=expected:raise ValueError('Resume query coverage disagrees with saved batch cursor')
            loader=ex.PairLoader(dataset,workers);iterator=iter(loader.batches(order[state['next_batch']:],epoch=state['epoch']))
            support=None
            try:
                for position in range(state['next_batch'],len(order)):
                    started=time.perf_counter();payload=next(iterator);load_done=time.perf_counter()
                    ids=payload.indices.tolist();group=dataset.rows[ids[0]]['patient_group']
                    if ids!=order[position] or any(dataset.rows[i]['patient_group']!=group for i in ids):raise ValueError('Query order/group changed')
                    if group!=state['last_group']:
                        support=ex.support_for_recipient(state['memory'],group)
                        with torch.autocast('cuda',dtype=torch.bfloat16):state['plan']=net.fit_support_clusters(*support)
                        state['audits'][group]=state['plan']['audit'];state['last_group']=group
                    elif support is None:support=ex.support_for_recipient(state['memory'],group)
                    plan_done=time.perf_counter()
                    targets=torch.tensor([dataset.rows[i]['target'] for i in ids],device='cuda')
                    values,timing,usage,check=optimizer_step(net,optimizer,payload,support,state['plan'],targets,weights,
                        base['training']['grad_clip'],context=rank_context,settings=settings,check_gradients=state['step']==0)
                    if check:ex.write_new(root/'first_gradient_check.json',check)
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
                    ex.emit(**progress)
                    if not (root/'first_optimizer_step.json').exists():ex.write_new(root/'first_optimizer_step.json',progress)
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
                seconds=state['epoch_seconds'],clusters=state['audits'],**ex.memory_metrics())
            ex.write_new(root/f'epoch_{state["epoch"]+1:03d}.json',row)
            with (root/'metrics.csv').open('a',newline='',encoding='utf-8') as f:
                flat={k:row[k] for k in ('epoch','optimization_steps','train_loss','alignment_loss','all_train_queries_visited','seconds')}
                flat.update(metrics);writer=csv.DictWriter(f,fieldnames=list(flat))
                if f.tell()==0:writer.writeheader()
                writer.writerow(flat)
            path=root/f'epoch_{state["epoch"]+1:03d}.pt'
            ex.save_torch_new(path,dict(format=CHECKPOINT_FORMAT,completed_epochs=state['epoch']+1,
                state_dict=ex.tree_to(net.state_dict(),'cpu'),optimizer=ex.tree_to(optimizer.state_dict(),'cpu'),
                epoch=state['epoch']+1,step=state['step'],**identity))
            ex.write_new(root/f'ranking_epoch_{state["epoch"]+1:03d}.json',net.ranking_validation_report)
            if metrics['ranking_pairwise_loss']<state['best']:
                state['best']=metrics['ranking_pairwise_loss'];state['best_path']=str(path)
            state['epoch']+=1;ex.fresh_epoch(state);state['phase']='optimization'
            saver.save(state);ex.emit(stage='epoch_complete',**{k:v for k,v in row.items() if k!='clusters'})
            if stop():return root/'checkpoint_latest.pt'
    if state['phase']!='final_memory':
        best=torch.load(state['best_path'],map_location='cpu',weights_only=False)
        net.load_state_dict(best['state_dict']);state['selected_epoch']=best['epoch']
        state.update(phase='final_memory',memory_work=None,memory_next=0);saver.save(state)
    if not ex.encode_memory(net,dataset,state,saver,state['memory_batch'],state['workers'],release_unused):
        stop();return root/'checkpoint_latest.pt'
    payload=dict(format=CHECKPOINT_FORMAT,debug=bool(debug),completed_epochs=cfg['gnn_epochs'],selected_epoch=state['selected_epoch'],
        state_dict=ex.tree_to(net.state_dict(),'cpu'),config=cfg,base=base,split=dataset.meta['split'],identities=dataset.meta['identities'],
        memory=ex.tree_to(state['memory'],'cpu'),donor_pool=dataset.meta['donor_pool'],raw_records=dataset.meta['raw_records'],
        training_objective=OBJECTIVE,ranking_contract=settings,feature_coordinates='stride4',support_task_contract='patient_group_v1',
        source_identity=ex.provenance(),cache_sha256=ex.sha(cache),physical_batch=state['batch'],workers=state['workers'],
        optimization_steps=state['step'],native_online_bank_integrated=False)
    ex.save_torch_new(root/'checkpoint.pt',payload)
    ex.write_new(root/'training_complete.json',dict(debug=bool(debug),epochs=cfg['gnn_epochs'],selected_epoch=state['selected_epoch'],
        checkpoint_sha256=ex.sha(root/'checkpoint.pt'),segmentation_training_executed=False))
    return root/'checkpoint.pt'
