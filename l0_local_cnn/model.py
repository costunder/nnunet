import copy,math
from dataclasses import dataclass
import torch
from torch import nn
from l0_exploration.model import OrganPyramid
from l0_regions.resident import signature,check_verified

MODE='paired_native_local_cnn_v1'

def validate_config(c):
    if c['architecture']!=MODE or c['channels']!=[12,24,32] or c['convolutions']!=[2,3,3] or c['hidden_dim']!=128:
        raise ValueError('Unexpected local CNN architecture')
    if not isinstance(c['margin_mm'],(int,float)) or not math.isfinite(c['margin_mm']) or c['margin_mm']<=0 or c['input']!='native_spacing_organ_only' or c['readout']!='organ_masked_mean_each_scale' or c['fusion']!='donor_target_difference_product':
        raise ValueError('Different approved CNN input/readout contract')
    if c['initialization']!='fresh_seed42' or c['learning_policy']!='same_donor_live_v1':raise ValueError('Different initialization/learning policy')

@dataclass
class LocalBatch:
    images:torch.Tensor
    organ:torch.Tensor
    donor:torch.Tensor
    recipient:torch.Tensor
    indices:torch.Tensor
    audit:list
    def __len__(self):return len(self.donor)
    def validate(self):
        x,m=self.images,self.organ
        if x.ndim!=5 or x.shape[1]!=1 or m.shape!=x.shape or m.dtype!=torch.bool:raise ValueError('CT and boolean organ [V,1,X,Y,Z] required')
        if not len(self) or self.recipient.shape!=self.donor.shape or self.indices.shape!=self.donor.shape:raise ValueError('Nonempty bound pair batch required')
        if not bool(m.flatten(1).any(1).all()) or not bool(torch.isfinite(x[m]).all()):raise ValueError('Invalid organ/CT')
        for v in (self.donor,self.recipient):
            if v.dtype!=torch.long or bool(((v<0)|(v>=len(x))).any()):raise ValueError('Bad paired crop index')
        self._verified_signature=signature(self)
        return self
    def to(self,device,non_blocking=True):
        check_verified(self)
        return LocalBatch(**{k:getattr(self,k).to(device,non_blocking=non_blocking) for k in ('images','organ','donor','recipient','indices')},audit=self.audit).validate()

class LocalCNN(nn.Module):
    def __init__(self,config,*,checkpointing=False,budget=None,dropout=.1):
        super().__init__();validate_config(config);self.config=copy.deepcopy(config);self.resource_budget=budget
        self.cnn=OrganPyramid(config['channels'],checkpointing)
        self.project=nn.Sequential(nn.Linear(sum(config['channels']),128),nn.LayerNorm(128),nn.SiLU())
        self.fuse=nn.Sequential(nn.Linear(512,256),nn.LayerNorm(256),nn.SiLU(),nn.Dropout(dropout),nn.Linear(256,128))
        self.dense_batch_size=32
        self.adjacencies=[]  # No topology, sparse cache, or graph message passing.
        self.monitor_prefixes={'CNN':'local.cnn','readout':'local.project','fusion':'local.fuse','L1':'l1','L2':'l2'}
    def get_extra_state(self):return copy.deepcopy(self.config)
    def set_extra_state(self,state):
        if state!=self.config:raise ValueError('Local CNN checkpoint input/architecture mismatch')
    def forward(self,batch):
        if not isinstance(batch,LocalBatch):raise TypeError('Native local CT batch required')
        check_verified(batch)
        if self.resource_budget:self.resource_budget.check()
        maps=self.cnn(batch.images,batch.organ)
        pooled=[]
        for x,m in maps:
            count=m.sum((2,3,4))
            if not bool((count>0).all()):raise ValueError('Organ vanished at a CNN scale; no silent zero representation')
            pooled.append(torch.where(m,x,0).sum((2,3,4))/count)
        features=self.project(torch.cat(pooled,1));d=features[batch.donor];r=features[batch.recipient]
        output=self.fuse(torch.cat((d,r,r-d,r*d),1))
        if output.shape!=(len(batch),128) or not bool(torch.isfinite(output).all()):raise FloatingPointError('Invalid local CNN output')
        if self.resource_budget:self.resource_budget.check()
        return output
