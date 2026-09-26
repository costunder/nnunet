"""Rank update and full held-out observation ranking; no annotation score boost."""
import time
import torch
from scipy.stats import rankdata
from hiercp_v222 import v1_execution as ex
from tools.v22_rank_objective import forward_loss,ranking_metrics,configuration


def optimizer_step(net,optimizer,payload,support,plan,targets,weights,grad_clip,*,context,settings,check_gradients=False):
    from hiercp_v222.v1_training import gradient_check
    ids=payload.indices.tolist()
    events=[torch.cuda.Event(enable_timing=True) for _ in range(5)]
    torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
    events[0].record();query=payload.cuda(non_blocking=True);events[1].record()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss,parts=forward_loss(net,query,support,plan,targets,weights,context,settings,indices=ids)
    events[2].record()
    rank_check={}
    if check_gradients and int(parts['ranking_pairs'].detach())>0:
        probe=torch.autograd.grad(parts['ranking_loss'],next(net.local.parameters()),retain_graph=True)[0]
        if not bool(torch.isfinite(probe).all()) or not bool(probe.abs().sum()>0):
            raise RuntimeError('Rank loss does not provide finite nonzero L0 gradient')
        rank_check['ranking_loss_L0_gradient_nonzero']=True
        del probe
    loss.backward();events[3].record()
    check=gradient_check(net) if check_gradients else None
    if check is not None:check.update(rank_check)
    torch.nn.utils.clip_grad_norm_(net.parameters(),grad_clip,error_if_nonfinite=True)
    optimizer.step();events[4].record();events[4].synchronize()
    timing={key:events[i].elapsed_time(events[i+1])/1000
            for i,key in enumerate(('H2D_seconds','forward_seconds','backward_seconds','optimizer_seconds'))}
    values=dict(loss=float(loss.detach()),**{name:float(value.detach()) for name,value in parts.items()})
    values['ranking_pairs']=int(values['ranking_pairs'])
    values['ranking_available']=values['ranking_pairs']>0
    usage=ex.memory_metrics();del parts,loss,query
    optimizer.zero_grad(set_to_none=True)
    return values,timing,usage,check


@torch.no_grad()
def evaluate(net,dataset,memory,batch,workers,batch_complete=None):
    from hiercp_v222.v1_training import groups
    net.eval();loader=ex.PairLoader(dataset,workers);last_group=None;pred=[];truth=[];case_ids=[];seen=[]
    try:
        for payload in loader.batches(groups(dataset,batch),epoch=0):
            ids=payload.indices.tolist();group=dataset.rows[ids[0]]['patient_group']
            if any(dataset.rows[i]['patient_group']!=group for i in ids):raise ValueError('Mixed support exclusion groups')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if group!=last_group:
                    state=net.prepare_support(*ex.support_for_recipient(memory,group));last_group=group
                result=net.predict_embeddings(net.local(payload.cuda(non_blocking=True)),state)
            pred.append(result['logits'].float().cpu());truth.extend(dataset.rows[i]['target'] for i in ids)
            case_ids.extend(dataset.rows[i]['case_id'] for i in ids);seen.extend(ids)
            del result,payload
            if batch_complete is not None:batch_complete()
    finally:loader.close()
    if sorted(seen)!=list(range(len(dataset))):raise RuntimeError('Incomplete/duplicated ranking validation')
    logits=torch.cat(pred);y=torch.tensor(truth);prob=logits.softmax(-1)[:,1].numpy()
    pos=int(y.sum());neg=len(y)-pos
    if not pos or not neg:raise ValueError('Both observed classes required')
    metrics,report=ranking_metrics(logits[:,1]-logits[:,0],y,case_ids,configuration()['report_recall_at'])
    net.ranking_validation_report=dict(before_observation_exclusion=True,tie_policy='pessimistic',cases=report,
        record_ids=[dataset.rows[i]['id'] for i in seen],scores=(logits[:,1]-logits[:,0]).tolist())
    return dict(**metrics,observation_cross_entropy=float(torch.nn.functional.cross_entropy(logits,y)),
        observation_accuracy=float((logits.argmax(-1)==y).float().mean()),
        observation_auroc=float((rankdata(prob)[y.numpy()==1].sum()-pos*(pos+1)/2)/(pos*neg)),
        samples=len(y),positives=pos,unobserved=neg,
        scope='held-out observed-anchor ranking before exclusion; not validated CP efficacy')
