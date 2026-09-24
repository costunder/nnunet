"""Actual patient-local label graphs and cross-patient positive evidence."""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import softmax as segment_softmax
from hiercp.schema import LOCAL_NODE_TYPES,LOCAL_HANDCRAFTED_DIM
DESCRIPTOR_DIM=len(LOCAL_NODE_TYPES)*LOCAL_HANDCRAFTED_DIM+20

def context_descriptor(batch):
    count=batch.graph.num_graphs; parts=[]
    for kind in LOCAL_NODE_TYPES:
        node=batch.graph[kind]; x=node.x.float()
        totals=x.new_zeros(count,x.shape[-1]).index_add_(0,node.batch,x)
        parts.append(totals/torch.bincount(node.batch,minlength=count).clamp_min(1)[:,None])
    for patch in (batch.source_patches[batch.source_index],batch.target_patches):
        x=patch.float().flatten(2); parts.extend((x.mean(-1),x.std(-1,unbiased=False)))
    return torch.cat(parts,-1).detach()

class Residual(nn.Module):
    def __init__(self,dim,dropout):
        super().__init__(); self.out=nn.Linear(dim,dim); self.norm=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,4*dim),nn.SiLU(),nn.Dropout(dropout),nn.Linear(4*dim,dim))
        self.final=nn.LayerNorm(dim); self.drop=nn.Dropout(dropout)
    def forward(self,x,message):
        x=self.norm(x+self.drop(self.out(message)))
        return self.final(x+self.drop(self.ff(x)))

class PatientTaskGraph(nn.Module):
    """Segmented disjoint patient graphs; queries cannot update support."""
    def __init__(self,dim,heads,layers,dropout,temperature):
        super().__init__(); self.heads=heads; self.width=dim//heads; self.temperature=temperature
        self.to_label=nn.ModuleList([nn.ModuleDict({'q':nn.Linear(dim,dim,bias=False),'k':nn.Linear(dim,dim,bias=False),
            'v':nn.Linear(dim,dim,bias=False),'residual':Residual(dim,dropout)}) for _ in range(layers)])
        self.to_data=nn.ModuleList([nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True) for _ in range(layers)])
        self.data_update=nn.ModuleList([Residual(dim,dropout) for _ in range(layers)])
    def encode_support(self,support,owners,labels):
        p,k,d=labels.shape; n=len(support)
        if owners.shape!=(n,): raise ValueError('Task ownership shape mismatch')
        if bool((torch.bincount(owners,minlength=p)==0).any()): raise ValueError('Every patient needs actual context nodes')
        history=[]
        for index in range(len(self.to_label)):
            if torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                from functools import partial
                support,labels=checkpoint(partial(self._support_layer,owners=owners,index=index),support,labels,use_reentrant=False)
            else:support,labels=self._support_layer(support,labels,owners,index)
            history.append(labels)
        return labels,self.relations(support,labels[owners]),history

    def _support_layer(self,support,labels,owners,index):
        p,k,d=labels.shape; n=len(support)
        block,attention,update=self.to_label[index],self.to_data[index],self.data_update[index]
        q=block['q'](labels).view(p,k,self.heads,self.width)
        sk=block['k'](support).view(n,self.heads,self.width); sv=block['v'](support).view(n,self.heads,self.width)
        logits=torch.einsum('nkhd,nhd->nkh',q[owners],sk)/math.sqrt(self.width)
        weights=segment_softmax(logits,owners,num_nodes=p,dim=0)
        messages=torch.zeros(p,k,self.heads,self.width,device=sv.device,dtype=torch.promote_types(weights.dtype,sv.dtype))
        messages.index_add_(0,owners,weights[...,None]*sv[:,None])
        labels=block['residual'](labels,messages.reshape(p,k,d))
        # Torch 2.8 CUDA efficient-attention rejects >65535 batch rows with dropout.
        # Split execution only; every context and every label/edge is retained.
        pieces=[]
        for begin in range(0,n,65535):
            x=support[begin:begin+65535]; local_labels=labels[owners[begin:begin+65535]]
            message=attention(x[:,None],local_labels,local_labels,need_weights=False)[0].squeeze(1)
            pieces.append(update(x,message))
        return torch.cat(pieces),labels

    def relations(self,data,labels):
        return (torch.einsum('nd,nkd->nk',F.normalize(data,dim=-1),F.normalize(labels,dim=-1))/self.temperature).softmax(-1)

    def query_relations(self,query,owners,history):
        for attention,update,labels in zip(self.to_data,self.data_update,history):
            local_labels=labels[owners]
            query=update(query,attention(query[:,None],local_labels,local_labels,need_weights=False)[0].squeeze(1))
        return self.relations(query,history[-1][owners])

class CrossPatientAlignment(nn.Module):
    def __init__(self,dim,heads,layers,dropout,temperature):
        super().__init__(); self.temperature=temperature
        self.layers=nn.ModuleList([nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True) for _ in range(layers)])
        self.updates=nn.ModuleList([Residual(dim,dropout) for _ in range(layers)])
    def forward(self,labels):
        p,k,d=labels.shape
        if p<2: raise ValueError('L2 requires different actual patient tasks')
        owner=torch.arange(p,device=labels.device).repeat_interleave(k); same=owner[:,None]==owner[None,:]
        x=labels.reshape(1,p*k,d)
        for layer,update in zip(self.layers,self.updates):
            x=update(x,layer(x,x,x,attn_mask=same,need_weights=False)[0])
        aligned=x.reshape(p,k,d)
        score=torch.einsum('aid,bjd->abij',F.normalize(aligned,dim=-1),F.normalize(aligned,dim=-1))/self.temperature
        return aligned,score.softmax(-1)

def transport_positive(assignment,owners,evidence,matrix,donor_allowed):
    """Only T creates evidence; target-patient diagonal excluded; U is not F."""
    p=matrix.shape[0]; k=assignment.shape[-1]
    positives=(evidence==1).to(assignment.dtype)
    counts=assignment.new_zeros(p).index_add_(0,owners,positives)
    mass=assignment.new_zeros(p,k).index_add_(0,owners,assignment*positives[:,None])
    distribution=mass/counts.clamp_min(1)[:,None]
    available=(counts>0)&donor_allowed
    allowed=available[:,None].expand(p,p)&~torch.eye(p,dtype=torch.bool,device=owners.device)
    result=torch.einsum('abij,ai,ab->bj',matrix,distribution,allowed.to(matrix.dtype))
    denominator=allowed.sum(0)
    return result/denominator.clamp_min(1)[:,None],denominator>0,distribution

class PromptGraphModel(nn.Module):
    def __init__(self,cfg,base,*,patient_ids,with_local=True):
        super().__init__(); self.cfg=cfg; self.patient_ids=tuple(patient_ids)
        if len(set(patient_ids))!=len(patient_ids) or not patient_ids: raise ValueError('Unique actual training patients required')
        self.patient_lookup={name:i for i,name in enumerate(patient_ids)}
        m=base['model']; d=m['hidden_dim']; self.dim=d
        self.patient_labels=nn.Parameter(torch.stack([self.initial_labels(name) for name in patient_ids]))
        if with_local:
            from hiercp.model import LocalTumorContextPyGEncoder
            self.local=LocalTumorContextPyGEncoder(hidden_dim=d,heads=m['heads'],layers=m['local_layers'],dropout=m['dropout'],
                dense_base_channels=m['dense_base_channels'],dense_feature_dim=m['dense_feature_dim'],dense_batch_size=m['dense_batch_size'],
                channels_last_3d=m['channels_last_3d'],checkpoint_local_blocks=m['checkpoint_local_blocks'],checkpoint_dense_encoder=m['checkpoint_dense_encoder'])
            self.data_projection=nn.Sequential(nn.Linear(d*6,d),nn.LayerNorm(d),nn.SiLU())
        self.task=PatientTaskGraph(d,m['heads'],cfg['task_layers'],m['dropout'],cfg['temperature'])
        self.alignment=CrossPatientAlignment(d,m['heads'],cfg['alignment_layers'],m['dropout'],cfg['temperature'])
        self.local_decoder=nn.Sequential(nn.Linear(d,2*d),nn.SiLU(),nn.Linear(2*d,DESCRIPTOR_DIM))
        self.label_decoder=nn.Sequential(nn.Linear(d,2*d),nn.SiLU(),nn.Linear(2*d,DESCRIPTOR_DIM))
    def initial_labels(self,case_id):
        seed=int.from_bytes(hashlib.sha256(f"{self.cfg['seed']}:{case_id}".encode()).digest()[:8],'little')%(2**63-1)
        return torch.randn(self.cfg['label_count'],self.dim,generator=torch.Generator().manual_seed(seed))/math.sqrt(self.dim)
    def labels_for(self,case_ids):
        # Unseen patients: independent random initialization + contextual adaptation;
        # no fitting on validation labels, and no fabricated positive observation.
        return torch.stack([self.patient_labels[self.patient_lookup[c]] if c in self.patient_lookup
            else self.initial_labels(c).to(self.patient_labels) for c in case_ids])
    def encode_local(self,batch):
        fields=self.local(batch.graph,batch.source_patches,batch.source_index,batch.target_patches)
        names=('tumor','source_context','target_context','source_relation','target_relation','fused')
        return self.data_projection(torch.cat([fields[k] for k in names],-1))
    def prepare_task_state(self,memory):
        labels,assignment,history=self.task.encode_support(memory['embeddings'],memory['owners'],self.labels_for(memory['case_ids']))
        aligned,matrix=self.alignment(labels)
        transported,available,positive=transport_positive(assignment,memory['owners'],memory['evidence'],matrix,memory['donor_allowed'])
        p,k,_=labels.shape
        weights=assignment.float()
        mass=weights.new_zeros(p,k).index_add_(0,memory['owners'],weights)
        basis=weights.new_zeros(p,k,DESCRIPTOR_DIM).index_add_(0,memory['owners'],weights[:,:,None]*memory['descriptors'].float()[:,None])
        basis=basis/mass.clamp_min(1e-8)[...,None]
        return dict(labels=labels,aligned=aligned,correspondence=matrix,positive=positive,transported=transported,
                    descriptor_basis=basis,patient_available=available,label_history=history)

    def forward_tasks(self,memory,query,query_owners,*,state=None):
        # Reuse frozen support/alignment once per full scoring/validation task.
        state=self.prepare_task_state(memory) if state is None else state
        posterior=self.task.query_relations(query,query_owners,state['label_history'])
        score=F.cosine_similarity(posterior,state['transported'][query_owners],dim=-1)
        label_query=torch.einsum('nk,nkd->nd',posterior,state['labels'][query_owners])
        return dict(state,scores=score,available=state['patient_available'][query_owners],posterior=posterior,
                    reconstruction=self.local_decoder(query),label_reconstruction=self.label_decoder(label_query))

def observed_evidence_loss(scores,evidence,available):
    if not bool(torch.isin(evidence,torch.tensor([-1,0,1],device=evidence.device)).all()): raise ValueError('Evidence must be T/F/U')
    known=(evidence!=-1)&available; target=(evidence==1).to(scores.dtype)
    return (((scores-target)**2)*known).sum()/known.sum().clamp_min(1),known.sum()

def prompt_loss(output,descriptors,evidence,weights,temperature):
    observed,count=observed_evidence_loss(output['scores'],evidence,output['available'])
    reconstruction=F.mse_loss(output['reconstruction'].float(),descriptors.float())+F.mse_loss(output['label_reconstruction'].float(),descriptors.float())
    # Structural teacher: actual context statistics, not biological GT correspondence.
    basis=F.normalize(output['descriptor_basis'].detach().float(),dim=-1)
    target=(torch.einsum('aid,bjd->abij',basis,basis)/temperature).softmax(-1)
    p=len(basis); cross=~torch.eye(p,dtype=torch.bool,device=basis.device)
    alignment=-(target*output['correspondence'].float().clamp_min(1e-8).log()).sum(-1)[cross].mean()
    terms={'observed_transfer':observed,'context_reconstruction':reconstruction,'cross_patient_alignment':alignment}
    total=sum(weights[k]*v for k,v in terms.items())
    if not bool(torch.isfinite(total)): raise FloatingPointError('Nonfinite cross-patient objective')
    return total,dict(terms,supervised_observations=count)
