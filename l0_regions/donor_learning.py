"""Explicit same-donor, live-gradient ranking contract; never an old-run resume."""
from collections import Counter
import math
import numpy as np
import torch
from torch.nn import functional as F

POLICY='same_donor_live_v1'

def configuration():
    return dict(objective=POLICY,ranking_weight=1.,observation_auxiliary_weight=1.,
        ranking='all within-case observed/unobserved pairs exactly once per epoch',
        donor='one fixed train-only donor per case; independent of observation class',
        reference='both sides live; no detached ranking-reference embeddings',
        auxiliary='global balanced CE; inverse schedule multiplicity per observation',
        normalization='epoch objective: global pair mean and global balanced observation mean',
        selection_metric='MRR then R@1 then negative pairwise loss',
        query_scope='all observations; retain cases without eligible positive',
        support='unchanged patient support and L2 policy',report_recall_at=[1,5,10])

def validate_rows(rows):
    by={}
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate observation')
    for i,r in enumerate(rows):
        if r['target'] not in (0,1):raise ValueError('Binary observation required')
        by.setdefault(r['case_id'],[]).append(i)
    for ids in by.values():
        if len({(rows[i]['donor_case_id'],rows[i]['donor_component'],rows[i]['donor_group']) for i in ids})!=1:
            raise ValueError('Ranking case contains different donors')
        if len({rows[i]['patient_group'] for i in ids})!=1:raise ValueError('Case identity mismatch')
        if any(rows[i]['donor_group']==rows[i]['patient_group'] for i in ids):raise ValueError('Self-patient donor')
    return by

def groups(dataset,batch,seed=None,epoch=0):
    """Tile the complete P x U comparison, without stale features or pair drops.

    Each tile uses up to the same physical batch; both sides are live. Repeated
    observation CE is inverse-weighted. Natural final partial tiles are explicit.
    """
    if batch<2:raise ValueError('Live comparison requires physical batch >=2')
    rows=dataset.rows;by=validate_rows(rows);rng=np.random.default_rng(None if seed is None else seed+epoch)
    cases=list(by)
    if seed is not None:rng.shuffle(cases)
    for case in cases:
        ids=sorted(by[case],key=lambda i:(rows[i]['bounds']['edges'],rows[i]['id']))
        pos=[i for i in ids if rows[i]['target']];neg=[i for i in ids if not rows[i]['target']]
        tiles=[]
        if pos and neg:
            width=min(len(pos),batch//2)
            for start in range(0,len(pos),width):
                p=pos[start:start+width];remaining=batch-len(p)
                tiles.extend(p+neg[j:j+remaining] for j in range(0,len(neg),remaining))
        else:tiles=[ids[j:j+batch] for j in range(0,len(ids),batch)]
        if seed is not None:rng.shuffle(tiles)
        yield from tiles

class LiveContext:
    def __init__(self,dataset,batch):
        self.rows=dataset.rows;by=validate_rows(self.rows)
        self.order=list(groups(dataset,batch));self.uses=Counter(i for ids in self.order for i in ids)
        if set(self.uses)!=set(range(len(self.rows))):raise ValueError('Observation coverage incomplete')
        self.pairs=sum(sum(self.rows[i]['target'] for i in ids)*sum(1-self.rows[i]['target'] for i in ids) for ids in by.values())
        if not self.pairs:raise ValueError('No observed/unobserved ranking pairs in training cohort')
        self.counts=Counter(r['target'] for r in self.rows)
        self.steps=len(self.order)
        self.audit=dict(policy=POLICY,unique_observations=len(self.rows),query_presentations=sum(self.uses.values()),
            optimization_steps=self.steps,physical_batch=batch,actual_batch_sizes=[len(ids) for ids in self.order],
            ranking_pairs=self.pairs,zero_positive_cases=sum(not any(self.rows[i]['target'] for i in ids) for ids in by.values()))

def forward_loss(net,query,support,plan,targets,weights,context,settings,*,indices):
    ids=list(indices);rows=[context.rows[i] for i in ids]
    if len({r['case_id'] for r in rows})!=1:raise ValueError('One recipient per ranking tile')
    validate_rows(rows)
    if not torch.equal(targets,torch.tensor([r['target'] for r in rows],device=targets.device)):raise ValueError('Observation binding mismatch')
    # Both sides are encoded in one disjoint-union physical batch.
    output=net.predict_embeddings(net.local(query),net.prepare_support(*support,cluster_plan=plan))
    logits=output['logits'].float();score=logits[:,1]-logits[:,0]
    p=targets==1;u=targets==0
    dif=score[p,None]-score[None,u]
    ranking=F.softplus(-dif).sum()*context.steps/context.pairs
    # Global class weights do not cancel on a pure-negative tile. Repeated
    # anchors do not receive repeated CE weight merely because P x U is tiled.
    coefficients=logits.new_tensor([context.steps/(2*context.counts[r['target']]*context.uses[i]) for i,r in zip(ids,rows)])
    auxiliary=(F.cross_entropy(logits,targets,reduction='none')*coefficients).sum()
    loss=settings['ranking_weight']*ranking+settings['observation_auxiliary_weight']*auxiliary+output['alignment_loss_weight']*output['alignment_loss']
    return loss,dict(ranking_loss=ranking,ranking_pairs=torch.tensor(dif.numel(),device=logits.device),
        observation_auxiliary_loss=auxiliary,alignment_loss=output['alignment_loss'])

def selection(metrics):
    values=(metrics['ranking_mrr'],metrics['ranking_recall_at_1'],-metrics['ranking_pairwise_loss'])
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite selection metric')
    return tuple(values)
