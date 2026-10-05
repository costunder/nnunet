"""Full-observation scoring and actual CUDA execution for the crossed D arm."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import signal
import time
import uuid

import psutil
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .contracts import canonical_hash
from .transition_evaluation import PreparedL0Batch, SCORING_FORMAT, run_scoring
from .transition_model import full_support_for_case, tensor_hash

FORMAT = 'v17_crossed_D_exact_resume_v1'


def append(path,value):
    with Path(path).open('a',encoding='utf8') as stream:
        stream.write(json.dumps(value,allow_nan=False,ensure_ascii=False)+'\n')


@torch.no_grad()
def observation_memory(net,ds,loader,batch,budget,*,timing=None):
    modes=[(m,m.training) for m in net.modules()]
    net.eval();parts=[];seen=[];start=time.perf_counter()
    try:
        schedule=[list(range(i,min(i+batch,len(ds.rows)))) for i in range(0,len(ds.rows),batch)]
        for ids,cpu in tqdm(zip(schedule,loader.batches(schedule,epoch=0)),total=len(schedule),desc='full observation support',unit='batch'):
            pause=getattr(net,'_transition_pause_requested',None)
            if pause is not None and pause():
                from .transition_c_training import TrainingPauseRequested
                raise TrainingPauseRequested()
            if cpu.indices.tolist()!=ids:
                raise ValueError('Full observation support order changed')
            t=time.perf_counter();query=cpu.to('cuda');value=net.local(query).detach().float()
            torch.cuda.synchronize();budget.check()
            if value.shape!=(len(ids),128) or not bool(torch.isfinite(value).all()):
                raise ValueError('Real complete support L0 embeddings required')
            parts.append(value);seen.extend(ids)
            if timing is not None:
                append(timing,dict(stage='full_observation_memory',records=len(ids),seconds=time.perf_counter()-t,
                                   rss_bytes=psutil.Process().memory_info().rss))
            del query,value,cpu
        if seen!=list(range(len(ds.rows))):
            raise ValueError('Full observation support lost or repeated records')
        names=sorted({r['patient_group'] for r in ds.rows})
        device=parts[0].device
        result=dict(embeddings=torch.cat(parts),record_ids=[r['id'] for r in ds.rows],
                    patient_groups=names,row_groups=[r['patient_group'] for r in ds.rows],
                    donor_groups=[r['donor_group'] for r in ds.rows],
                    owners=torch.tensor([names.index(r['patient_group']) for r in ds.rows],device=device),
                    classes=torch.tensor([r['target'] for r in ds.rows],device=device),
                    full_training_observation_bank=True,seconds=time.perf_counter()-start)
        return result
    finally:
        for module,mode in modes:module.training=mode


class WholeCandidateEvaluation:
    """Common external-donor P/U audit, independently refreshed for each model.

    C's old curriculum bank is not relabeled as observation support. Both arms
    encode ALL actual native inner-train rows with their current L0 weights.
    Only L0 is chunked; the upper sees the entire recipient case in one call.
    """
    def __init__(self,inventory,train_ds,train_loader,val_ds,val_loader,*,batch,budget,debug=False):
        self.inventory=inventory;self.train_ds=train_ds;self.train_loader=train_loader
        self.val_ds=val_ds;self.val_loader=val_loader;self.batch=batch;self.budget=budget;self.debug=debug
        self.lookup={r['id']:i for i,r in enumerate(val_ds.rows)}
        self.case_ids=sorted({r['case_id'] for r in val_ds.rows})

    def __call__(self,net,*,memory=None,output=None):
        from hiercp_v222.v1_execution import rng_state,restore_rng
        modes=[(m,m.training) for m in net.modules()];rng=rng_state();net.eval()
        try:
            if memory is None:
                memory=observation_memory(net,self.train_ds,self.train_loader,self.batch,self.budget)
            def provider(rows):
                pause=getattr(net,'_transition_pause_requested',None)
                if pause is not None and pause():
                    from .transition_c_training import TrainingPauseRequested
                    raise TrainingPauseRequested()
                ids=[self.lookup[r['id']] for r in rows]
                batch=self.val_loader.get(ids,epoch=0)
                return PreparedL0Batch(batch,tuple(r['id'] for r in rows),False)
            def encode(cpu):
                result=net.local(cpu.to('cuda')).float();self.budget.check();return result
            def score(features,rows):
                group=rows[0]['patient_group'];support=full_support_for_case(memory,group)
                state=net.prepare_support(*support)
                logits=net.predict_embeddings(features,state)['logits'].float()
                return dict(scores=logits[:,1]-logits[:,0],contract=dict(format=SCORING_FORMAT,
                    scored_record_ids=[r['id'] for r in rows],l0_only_chunking=True,upper_chunking=False,
                    upper_execution='single_joint_case',query_GT_in_forward=False,upper_invocations=1))
            result=run_scoring(self.inventory,provider,encode,score,l0_batch_size=self.batch,
                               case_ids=self.case_ids if self.debug else None,debug=self.debug)
            result.update(raw_CT_execution_verified=True,callback_internal_execution_verified=True,
                          model_sha256=tensor_hash(net.state_dict()),initial_upper_sha256=net.initial_upper_sha256,
                          support_records=len(memory['record_ids']),support_GT='native observed P/unobserved U',
                          upper_support_policy='full inner-train bank; exclude query patient on both support recipient and donor sides',
                          C_curriculum_support_reused=False,quality_verified=False)
            result['full_evaluation']=not self.debug and len(self.case_ids)==21
            if output is not None:
                requested=Path(output)
                path=requested.with_name(requested.stem+'_'+uuid.uuid4().hex+requested.suffix)
                result['evaluation_artifact']=str(path.resolve())
                with path.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
            return result
        finally:
            for module,mode in modes:module.training=mode
            restore_rng(rng)


def native_D_loss(net,query,support,plan,targets,context,indices):
    """Native three objective terms unchanged, plus original input-owned .1 consistency."""
    rows=[context.rows[i] for i in indices]
    if len({r['case_id'] for r in rows})!=1:
        raise ValueError('D tile must have one recipient and its fixed donor')
    torch._assert_async((targets==torch.tensor([r['target'] for r in rows],device=targets.device)).all(),
                        'D GT binding changed')
    value,consistency=net.local.forward_with_consistency(query)
    out=net.predict_embeddings(value,net.prepare_support(*support,cluster_plan=plan))
    logits=out['logits'].float();score=logits[:,1]-logits[:,0]
    difference=score[targets==1,None]-score[None,targets==0]
    ranking=F.softplus(-difference).sum()*context.steps/context.pairs
    coef=logits.new_tensor([context.steps/(2*context.counts[r['target']]*context.uses[i])
                           for i,r in zip(indices,rows)])
    ce=(F.cross_entropy(logits,targets,reduction='none')*coef).sum()
    loss=ranking+ce+out['alignment_loss_weight']*out['alignment_loss']+.1*consistency
    return loss,dict(ranking_loss=ranking,observation_ce=ce,alignment=out['alignment_loss'],
                     input_view_consistency=consistency,ranking_pairs=difference.numel())


def gradient_receipt(net):
    groups=dict(L0=net.local,L1=net.core.l1,L2=net.core.l2,L2_updates=net.core.l2_updates)
    receipt={};flat=[]
    for name,module in groups.items():
        terms=[p.grad.detach().float().square().sum() for p in module.parameters()
               if p.requires_grad and p.grad is not None]
        if not terms:raise RuntimeError(f'Actual loss did not reach {name}')
        norm=torch.stack(terms).sum().sqrt();flat.append(norm);receipt[name]=norm
    values=torch.stack(flat)
    if not bool(torch.isfinite(values).all()):raise FloatingPointError('Nonfinite gradient')
    return dict(zip(receipt,values.cpu().tolist()))|dict(
        trainable_parameter_tensors=sum(p.requires_grad for p in net.parameters()),
        parameter_tensors_with_gradient=sum(p.requires_grad and p.grad is not None for p in net.parameters()),
        missing_parameter_gradients=[k for k,p in net.named_parameters() if p.requires_grad and p.grad is None])


def calibrate_D(net,ds,loader,memory,candidates,budget,base,support_patients,*,debug=False):
    """Full physical-row candidates, cloned model/optimizer; no production mutation."""
    from l0_regions.donor_learning import groups,LiveContext
    from l0_regions.training import legacy_groups
    from l0_regions.support_episodes import PatientEpisodes
    from hiercp_v222.v1_execution import rng_state,restore_rng
    import gc
    rng=rng_state();before=tensor_hash(net.state_dict());reports=[]
    try:
        for batch in candidates:
            context=LiveContext(ds,batch);order=list(groups(ds,batch,42,0))
            selected=max(order,key=lambda ids:(loader.measured_cost(ids)['sampled_two_view_edges'],
                                              loader.measured_cost(ids)['sampled_two_view_nodes']))
            episodes=PatientEpisodes(ds.rows,list(legacy_groups(ds,batch,42,0)),support_patients,42,0).bind(memory)
            support=episodes.support(ds.rows[selected[0]]['patient_group'])
            clone=copy.deepcopy(net).cuda().train()
            optimizer=torch.optim.AdamW(clone.parameters(),lr=base['training']['lr'],
                weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
            query=None
            try:
                clone.eval()
                with torch.no_grad():plan=clone.fit_support_clusters(*support)
                clone.train();query=loader.get(selected,epoch=0).to('cuda');seconds=[]
                torch.cuda.reset_peak_memory_stats()
                for trial in range(3):
                    optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                    loss,_=native_D_loss(clone,query,support,plan,memory['classes'][selected],context,selected)
                    loss.backward();gradient_receipt(clone)
                    torch.nn.utils.clip_grad_norm_(clone.parameters(),base['training']['grad_clip'],error_if_nonfinite=True)
                    optimizer.step();torch.cuda.synchronize();budget.check()
                    if trial:seconds.append(time.perf_counter()-start)
                    del loss
                reports.append(dict(configured_batch=batch,actual_observation_rows=len(selected),
                    actual_local_graphs=2*len(selected),accepted=True,seconds=sum(seconds)/len(seconds),
                    observations_per_second=len(selected)/(sum(seconds)/len(seconds)),
                    peak_bytes=torch.cuda.max_memory_allocated(),actual_input_cost=loader.measured_cost(selected),
                    scope='largest measured epoch0 sampled-edge tile; future stochastic views are not a worst-case guarantee'))
            except torch.cuda.OutOfMemoryError as exc:
                reports.append(dict(configured_batch=batch,accepted=False,error=str(exc)))
            finally:
                del clone,optimizer,query,support
                gc.collect();torch.cuda.empty_cache()
        if tensor_hash(net.state_dict())!=before:raise RuntimeError('Calibration changed production model')
        accepted=[r for r in reports if r['accepted']]
        if not accepted:raise MemoryError('No explicitly requested complete physical batch admitted: '+json.dumps(reports))
        return max(accepted,key=lambda r:r['observations_per_second'])['configured_batch'],reports
    finally:restore_rng(rng)


def train_D(net,ds,loader,evaluator,base,*,output,budget,candidates,support_patients,identity,
            debug=False,debug_updates=None,resume=None):
    from l0_regions.donor_learning import groups,LiveContext
    from l0_regions.training import legacy_groups,hash_state
    from l0_regions.support_episodes import PatientEpisodes
    from .transition_timing import WallCheckpointPipeline,restore_D_wall
    from .transition_c_training import ActiveEpochWall
    from hiercp_v222.v1_execution import tree_to,restore_rng
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    initial_neural_sha256=tensor_hash(net.state_dict())
    probe_before={name:{key:p.detach().clone() for key,p in module.named_parameters()}
                  for name,module in dict(L0=net.local,L1=net.core.l1,L2=net.core.l2,
                                          L2_updates=net.core.l2_updates).items()} if debug else None
    if debug_updates is not None and (not debug or debug_updates<1):raise ValueError('Explicit DEBUG updates only')
    epochs=1 if debug else 40
    opt=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],
                         weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
    state=dict(epoch=0,step=0,next_batch=0,phase='initial_memory',batch=None,memory=None,
               plan=None,last_group=None,best=None,epoch_seconds=0.)
    if resume:
        payload=torch.load(resume,map_location='cpu',weights_only=False)
        if (payload.get('format')!=FORMAT or payload.get('identity')!=identity
                or payload.get('content_sha256')!=hash_state({k:v for k,v in payload.items() if k!='content_sha256'})):
            raise ValueError('D exact resume requires its own bound arm/data/runtime/source checkpoint')
        net.load_state_dict(payload['model']);opt.load_state_dict(payload['optimizer'])
        state=tree_to(restore_D_wall(payload,root),'cuda');restore_rng(payload['rng'])
    elif (root/'checkpoint_latest.pt').exists():
        raise FileExistsError('An existing D checkpoint must be explicitly resumed')
    paused=[False];old_signal=signal.signal(signal.SIGINT,lambda *_:paused.__setitem__(0,True))
    try:
        clock=None;journal='d_checkpoint_wall_'+uuid.uuid4().hex+'.jsonl'
        with WallCheckpointPipeline(root,'overlapped',None,hash_state,FORMAT) as writer:
            def save(wait=False):
                if clock is None:
                    state['epoch_wall_checkpoint']=None
                    return writer.save(net,opt,state,identity,wait=wait)
                token=clock.begin('checkpoint_submit_and_wait')
                state['epoch_wall']=clock.snapshot('cursor_before_current_checkpoint_write')
                state['epoch_seconds']=state['epoch_wall']['elapsed_seconds']
                state['epoch_wall_checkpoint']=dict(checkpoint_id=uuid.uuid4().hex,journal_file=journal,
                                                    snapshot_monotonic=time.perf_counter())
                try:return writer.save(net,opt,state,identity,wait=wait)
                finally:clock.finish(token)
            if state['phase']=='initial_memory':
                state['memory']=observation_memory(net,ds,loader,min(candidates),budget,timing=root/'memory_timing.jsonl')
                batch,reports=calibrate_D(net,ds,loader,state['memory'],candidates,budget,base,support_patients,debug=debug)
                state.update(batch=batch,phase='initial_validation');evaluator.batch=batch
                append(root/'calibration.jsonl',dict(reports=reports,selected_observation_batch=batch))
                save(wait=True)
            evaluator.batch=state['batch']
            if state['phase']=='initial_validation':
                report=evaluator(net,memory=state['memory']);append(root/'curve.jsonl',dict(epoch=0,stage='validation',**report['metrics']))
                state['phase']='optimization';save(wait=True)
            if debug and debug_updates is not None and state['step']>=debug_updates:
                print(f"DEBUG already completed {state['step']} successful updates; no additional update.",flush=True)
                return root/'checkpoint_latest.pt'
            while state['epoch']<epochs:
                clock=ActiveEpochWall(state['epoch']+1,state.get('epoch_wall'))
                if state['phase']=='optimization':
                    order=list(groups(ds,state['batch'],42,state['epoch']));context=LiveContext(ds,state['batch'])
                    episodes=PatientEpisodes(ds.rows,list(legacy_groups(ds,state['batch'],42,state['epoch'])),
                                             support_patients,42,state['epoch']).bind(state['memory'])
                    net.train();bar=tqdm(loader.batches(order[state['next_batch']:],epoch=state['epoch']),
                        initial=state['next_batch'],total=len(order),desc=f"D epoch {state['epoch']+1}/{epochs}")
                    previous=time.perf_counter()
                    for cpu in bar:
                        wait=time.perf_counter()-previous;started=time.perf_counter();ids=cpu.indices.tolist()
                        clock.record_duration('loader_wait',wait)
                        update_wall=clock.begin('optimization_host_and_GPU')
                        graph_counts=dict(nodes={k:len(v) for k,v in cpu.graph.x_dict.items()},
                                          edges={'|'.join(k):v.shape[1] for k,v in cpu.graph.edge_index_dict.items()})
                        if ids!=order[state['next_batch']]:raise ValueError('Saved D cursor/order mismatch')
                        group=ds.rows[ids[0]]['patient_group'];support=episodes.support(group)
                        if state['last_group']!=group:
                            net.eval()
                            with torch.no_grad():state['plan']=net.fit_support_clusters(*support)
                            net.train();state['last_group']=group
                        opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
                        events=[torch.cuda.Event(enable_timing=True) for _ in range(5)]
                        events[0].record();query=cpu.to('cuda');events[1].record()
                        loss,terms=native_D_loss(net,query,support,state['plan'],state['memory']['classes'][ids],context,ids)
                        events[2].record();loss.backward();events[3].record();grad=gradient_receipt(net)
                        grad_norm=torch.nn.utils.clip_grad_norm_(net.parameters(),base['training']['grad_clip'],error_if_nonfinite=True)
                        opt.step();events[4].record();events[4].synchronize();budget.check()
                        value=float(loss.detach());terms={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in terms.items()}
                        state['next_batch']+=1;state['step']+=1
                        del query,loss,cpu,support
                        elapsed=time.perf_counter()-started;clock.finish(update_wall)
                        saved=save()
                        append(root/'updates.jsonl',dict(step=state['step'],epoch=state['epoch']+1,loss=value,terms=terms,
                            gradients=grad,gradient_clip_norm=float(grad_norm),physical_observations=len(ids),local_graphs=2*len(ids),
                            actual_batched_graph_counts=graph_counts,
                            data_wait_seconds=wait,seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated(),
                            rss_bytes=psutil.Process().memory_info().rss,checkpoint=saved,
                            **{k:events[i].elapsed_time(events[i+1])/1000 for i,k in enumerate(('H2D_s','forward_s','backward_s','optimizer_s'))}))
                        bar.set_postfix(loss=f'{value:.4g}',step=state['step'],GiB=round(torch.cuda.max_memory_allocated()/2**30,2))
                        if paused[0] or debug_updates is not None and state['step']>=debug_updates:
                            save(wait=True);append(root/'status.jsonl',dict(status='PAUSED',debug=debug,step=state['step'],full_training=False))
                            if debug and not paused[0]:
                                report=evaluator(net,output=root/'DEBUG_final_full128.json')
                                changes={}
                                for name,module in dict(L0=net.local,L1=net.core.l1,L2=net.core.l2,
                                                        L2_updates=net.core.l2_updates).items():
                                    squares=[(p.detach()-probe_before[name][key]).float().square().sum()
                                             for key,p in module.named_parameters()]
                                    norm=float(torch.stack(squares).sum().sqrt())
                                    if not norm>0:raise RuntimeError(f'Actual optimizer did not change {name}')
                                    changes[name]=norm
                                append(root/'DEBUG_neural_execution.jsonl',dict(debug=True,actual_CT=True,
                                    actual_CUDA=True,updates=state['step'],optimizer_weight_changes=changes,
                                    initial_neural_sha256=initial_neural_sha256,
                                    final_neural_sha256=tensor_hash(net.state_dict()),
                                    evaluation_artifact=report['evaluation_artifact'],metrics=report['metrics'],
                                    full_training=False,full_evaluation=False,quality_verified=False))
                            print(f"PAUSED: {root/'checkpoint_latest.pt'}",flush=True);return root/'checkpoint_latest.pt'
                        previous=time.perf_counter()
                    state['phase']='refresh_memory';state.update(plan=None,last_group=None);save(wait=True)
                if state['phase']=='refresh_memory':
                    with clock.phase('full_support_refresh'):
                        state['memory']=observation_memory(net,ds,loader,state['batch'],budget,timing=root/'memory_timing.jsonl')
                    state['phase']='validation';save(wait=True)
                if state['phase']=='validation':
                    with clock.phase('whole_candidate_validation'):
                        report=evaluator(net,memory=state['memory']);metrics=report['metrics']
                    key=(metrics['case_first_P_mrr'],metrics['observed_micro_recall_at_1'],-metrics['P_U_softplus_loss'])
                    improved=state['best'] is None or key>tuple(state['best']['key'])
                    if improved:
                        from l0_regions.execution_pipeline import packed_cpu_snapshot
                        state['best']=dict(key=key,epoch=state['epoch']+1,weights=packed_cpu_snapshot(net.state_dict()))
                    save(wait=True)
                    completed_epoch=state['epoch']+1;finished_clock=clock
                    with finished_clock.phase('next_epoch_cursor_checkpoint_publication'):
                        state.update(epoch=completed_epoch,next_batch=0,phase='optimization',epoch_seconds=0.,epoch_wall=None)
                        clock=None;save(wait=True)
                    wall=finished_clock.history_fields('through_validation_and_completed_next_epoch_cursor_checkpoint_publication')
                    append(root/'curve.jsonl',dict(epoch=completed_epoch,stage='validation',new_best=improved,
                        **wall,**metrics,denominators=report['denominators']))
            net.load_state_dict(state['best']['weights'])
            result=evaluator(net,output=root/'full_candidate_best.json')
            state['phase']='complete';save(wait=True)
            from tools.run_v17_crossed_training import sha
            from tools.run_local_cnn_experiment import write_json
            write_json(root/'training_complete.json',dict(debug=debug,full_training=not debug,full_evaluation=result['full_evaluation'],
                    arm='D',identity=identity,completed_epochs=epochs,steps=state['step'],selected_epoch=state['best']['epoch'],
                    nnunet_training=False,quality_verified=False,metrics=result['metrics'],
                    evaluation_artifact=result['evaluation_artifact'],
                    checkpoint_sha256=sha(root/'checkpoint_latest.pt'),evaluation_sha256=sha(result['evaluation_artifact'])))
            return root/'checkpoint_latest.pt'
    finally:signal.signal(signal.SIGINT,old_signal)
