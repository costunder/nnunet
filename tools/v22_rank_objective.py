"""Observed-anchor ranking; query annotation enters loss/metrics, never scoring.

The auxiliary predicts observed presence, not paste validity. Epoch L0 references
are detached, as in the existing support bank; current query rows keep L0 grads.
This is memory-based minibatch ranking, not full-case end-to-end backpropagation.
"""
import json
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
OBJECTIVE='observed_rank_v1'
LEGACY='observation_ce'


def configuration():
    value=json.loads((ROOT/'config/v22_observed_ranking.json').read_text())
    if value['objective']!=OBJECTIVE or value['ranking_weight']<=0 or value['observation_auxiliary_weight']<=0:
        raise ValueError('Explicit positive ranking/auxiliary weights required')
    return value


def resolve_objective(saved=None,requested=None):
    allowed=(OBJECTIVE,LEGACY)
    if requested is not None and requested not in allowed:raise ValueError('Unknown training objective')
    if saved is None:return requested or OBJECTIVE
    previous=(saved.get('execution_policy') or {}).get('training_objective',saved.get('training_objective',LEGACY))
    if previous not in allowed:raise ValueError('Unknown checkpoint objective')
    if requested is not None and requested!=previous:
        raise ValueError('Ranking changes the training objective; use a new run, not exact resume')
    if previous==OBJECTIVE and saved.get('ranking_contract')!=configuration():
        raise ValueError('Saved ranking contract differs from current configuration')
    return previous


def rank_loss(scores,observed,cases,live):
    """All P/U pairs in one scan that touch at least one live query row."""
    if scores.ndim!=1 or observed.shape!=scores.shape or cases.shape!=scores.shape or live.shape!=scores.shape:
        raise ValueError('Aligned score, observation, case, live vectors required')
    if observed.dtype!=torch.long or live.dtype!=torch.bool:raise TypeError('Long observations and bool live mask required')
    if bool(((observed!=0)&(observed!=1)).any()):raise ValueError('Observation must be zero or one')
    pairs=(observed[:,None]==1)&(observed[None,:]==0)&(cases[:,None]==cases[None,:])&(live[:,None]|live[None,:])
    count=pairs.sum()
    difference=scores.float()[:,None]-scores.float()[None,:]
    # Empty valid relation set contributes no rank loss, explicitly counted.
    # The observation and L2 terms still train zero-positive cases.
    value=(F.softplus(-difference)*pairs).sum()/count.clamp_min(1)
    return value,count


class RankingContext:
    def __init__(self,dataset,memory):
        self.dataset=dataset;self.memory=memory
        lookup={name:i for i,name in enumerate(memory['record_ids'])}
        if len(lookup)!=len(memory['record_ids']) or set(lookup)!=set(r['id'] for r in dataset.rows):
            raise ValueError('Ranking reference must cover exactly all inner-train observations')
        self.lookup=lookup;self.by_case={}
        for i,row in enumerate(dataset.rows):self.by_case.setdefault(row['case_id'],[]).append(i)
        self.case_index={case:i for i,case in enumerate(sorted(self.by_case))}

    def reference(self,ids,device):
        if len(ids)!=len(set(ids)):raise ValueError('Duplicated live ranking query')
        rows=self.dataset.rows
        selected=sorted({rows[i]['case_id'] for i in ids})
        all_ids=[i for case in selected for i in self.by_case[case]]
        position={index:j for j,index in enumerate(all_ids)}
        tensor=lambda value:torch.tensor(value,device=device,dtype=torch.long)
        locations=tensor([position[i] for i in ids])
        reference=self.memory['embeddings'].index_select(0,tensor([self.lookup[rows[i]['id']] for i in all_ids])).detach()
        observed=tensor([rows[i]['target'] for i in all_ids])
        cases=tensor([self.case_index[rows[i]['case_id']] for i in all_ids])
        return reference,locations,observed,cases


def forward_loss(net,query,support,plan,targets,weights,context,settings,*,indices):
    ids=list(indices)
    reference,locations,observed,cases=context.reference(ids,query.target_patches.device)
    # Scoring receives embeddings and training-only support. No query GT edge,
    # score boost, rank override, or observation-specific encoder branch.
    embeddings=reference.index_copy(0,locations,net.local(query).to(reference.dtype))
    state=net.prepare_support(*support,cluster_plan=plan)
    output=net.predict_embeddings(embeddings,state)
    logits=output['logits'].float()
    live=torch.zeros(len(logits),device=logits.device,dtype=torch.bool);live[locations]=True
    if not torch.equal(observed[locations],targets):raise ValueError('Query observation/reference mismatch')
    ranks,pairs=rank_loss(logits[:,1]-logits[:,0],observed,cases,live)
    auxiliary=F.cross_entropy(logits[locations],targets,weight=weights)
    loss=(settings['ranking_weight']*ranks+settings['observation_auxiliary_weight']*auxiliary+
          output['alignment_loss_weight']*output['alignment_loss'])
    return loss,dict(ranking_loss=ranks,ranking_pairs=pairs,observation_auxiliary_loss=auxiliary,
                     alignment_loss=output['alignment_loss'])


def ranking_metrics(scores,observed,case_ids,top_k=(1,5,10)):
    """Full-case ranking before annotation exclusion, with pessimistic ties."""
    scores=torch.as_tensor(scores,dtype=torch.float64);observed=torch.as_tensor(observed,dtype=torch.long)
    if len(case_ids)!=len(scores) or not torch.isfinite(scores).all():raise ValueError('Complete finite scores required')
    names=sorted(set(case_ids));case_tensor=torch.tensor([names.index(c) for c in case_ids])
    loss,pairs=rank_loss(scores,observed,case_tensor,torch.ones(len(scores),dtype=torch.bool))
    reports=[];all_ranks=[]
    for name in names:
        mask=torch.tensor([c==name for c in case_ids]);values=scores[mask];truth=observed[mask].bool()
        positives=values[truth]
        if not len(positives):
            reports.append(dict(case_id=name,eligible_observed=0,rank_evaluable=False));continue
        ranks=(values[None,:]>=positives[:,None]).sum(1) # ties cannot improve rank using GT
        all_ranks.extend(ranks.tolist())
        reports.append(dict(case_id=name,eligible_observed=len(positives),rank_evaluable=True,
            observed_ranks=ranks.tolist(),first_observed_rank=int(ranks.min()),
            reciprocal_rank=float(1/ranks.min().double()),
            **{f'recall_at_{k}':float((ranks<=k).double().mean()) for k in top_k}))
    valid=[r for r in reports if r['rank_evaluable']]
    if not valid or not int(pairs):raise ValueError('Ranking validation needs observed/unobserved comparisons')
    return dict(ranking_pairwise_loss=float(loss),ranking_pairs=int(pairs),
        ranking_mrr=sum(r['reciprocal_rank'] for r in valid)/len(valid),
        ranking_mean_observed_rank=sum(all_ranks)/len(all_ranks),
        ranking_evaluable_cases=len(valid),ranking_cases_without_observed=len(names)-len(valid),
        **{f'ranking_recall_at_{k}':sum(rank<=k for rank in all_ranks)/len(all_ranks) for k in top_k}),reports
