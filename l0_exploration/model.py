"""Shared native-resolution masked CNN + learned exploration + mean GraphSAGE.

Positions are (D,H,W) native voxel indices, not grid_sample's reversed order.
Every volume is encoded once per forward; indexed trilinear sampling shares its
maps among all queries. Coordinate gradients remain connected to offset heads.
"""
from dataclasses import dataclass, replace
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class ExplorationBatch:
    images: torch.Tensor
    organ: torch.Tensor
    spacing: torch.Tensor
    recipient_index: torch.Tensor
    donor_index: torch.Tensor
    centers: torch.Tensor
    donor_centers: torch.Tensor
    def __len__(self):return len(self.centers)
    def to(self,device,non_blocking=True):
        return replace(self,**{k:v.to(device,non_blocking=non_blocking) for k,v in vars(self).items()})
    def validate(self):
        x=self.images;m=self.organ;v=len(x);q=len(self)
        if x.ndim!=5 or x.shape[1]!=1 or x.shape!=m.shape or m.dtype!=torch.bool:raise ValueError('Matching CT and boolean organ [V,1,D,H,W] required')
        if self.spacing.shape!=(v,3) or not bool(torch.isfinite(self.spacing).all()&(self.spacing>0).all()):raise ValueError('Physical spacing required')
        if self.centers.shape!=(q,3) or self.donor_centers.shape!=(q,3) or not q:raise ValueError('Nonempty 3D queries required')
        for ids in (self.recipient_index,self.donor_index):
            if ids.shape!=(q,) or ids.dtype!=torch.long or bool(((ids<0)|(ids>=v)).any()):raise ValueError('Bad volume index')
        if any(t.device!=x.device for t in vars(self).values()):raise ValueError('Mixed devices')
        if not bool(torch.isfinite(x[m]).all()):raise ValueError('Nonfinite internal CT')
        if not bool(m.flatten(1).any(1).all()):raise ValueError('Empty organ volume')
        for p,ids in ((self.centers,self.recipient_index),(self.donor_centers,self.donor_index)):
            if not bool(torch.isfinite(p).all()) or not bool(inside(m,p,ids).all()):raise ValueError('Center outside organ')

def inside(mask,pos,volume_ids):
    shape=torch.tensor(mask.shape[2:],device=pos.device)
    index=torch.floor(pos+.5).long();valid=((index>=0)&(index<shape)).all(-1)
    safe=index.maximum(torch.zeros_like(index)).minimum(shape-1)
    flat=(safe[...,0]*shape[1]+safe[...,1])*shape[2]+safe[...,2]
    ids=volume_ids.reshape((-1,)+(1,)*(pos.ndim-2)).expand(pos.shape[:-1])
    return valid & mask[:,0].flatten(1)[ids,flat]

def sample_map(features,mask,pos,volume_ids,stride=1):
    """All eight interpolation corners. No replicated V x Q feature volume.

Stride convolution centers are 0,s,2s in native coordinates. Invalid corners
have zero weight. A coarse level without organ support contributes zero; native
stride-1 support exists at admitted centers. This is not an error fallback.
"""
    shape=torch.tensor(features.shape[2:],device=pos.device)
    p=pos/stride;lower=p.floor().long();fraction=p-lower
    bits=torch.tensor([[a,b,c] for a in (0,1) for b in (0,1) for c in (0,1)],device=pos.device)
    corner=lower[...,None,:]+bits;valid=((corner>=0)&(corner<shape)).all(-1)
    safe=corner.maximum(torch.zeros_like(corner)).minimum(shape-1)
    flat=(safe[...,0]*shape[1]+safe[...,1])*shape[2]+safe[...,2]
    ids=volume_ids.reshape((-1,)+(1,)*(flat.ndim-1)).expand_as(flat)
    mass=mask[:,0].flatten(1)[ids,flat]&valid
    weights=torch.where(bits.bool(),fraction[...,None,:],1-fraction[...,None,:]).prod(-1)*mass
    values=features.flatten(2).transpose(1,2)[ids,flat]
    return (values*weights[...,None]).sum(-2)/weights.sum(-1,keepdim=True).clamp_min(1e-8)

class MaskedConv(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__();self.conv=nn.Conv3d(cin,cout,3,stride=stride,padding=1,bias=False)
        self.norm=nn.LayerNorm(cout);self.stride=stride
    def forward(self,x,m):
        x=self.conv(torch.where(m,x,0))
        if self.stride!=1:m=m[:,:,::self.stride,::self.stride,::self.stride]
        x=F.silu(self.norm(x.movedim(1,-1))).movedim(-1,1)
        return torch.where(m,x,0),m

class OrganPyramid(nn.Module):
    def __init__(self,channels,checkpointing):
        super().__init__();a,b,c=channels
        self.checkpointing=checkpointing
        self.levels=nn.ModuleList([nn.ModuleList([MaskedConv(1,a),MaskedConv(a,a)]),
            nn.ModuleList([MaskedConv(a,b,2),MaskedConv(b,b),MaskedConv(b,b)]),
            nn.ModuleList([MaskedConv(b,c,2),MaskedConv(c,c),MaskedConv(c,c)])])
    def forward(self,images,organ):
        x=torch.where(organ,images,0);m=organ;maps=[]
        for blocks in self.levels:
            for block in blocks:
                if self.checkpointing and self.training and torch.is_grad_enabled():x,m=checkpoint(block,x,m,use_reentrant=False)
                else:x,m=block(x,m)
            maps.append((x,m))
        return maps

class MeanSAGEResidual(nn.Module):
    """Root linear exactly once + neighbor mean; then residual FFN.

Each nonroot has one parent, so batched parent gather is exact neighbor mean
for this directed graph. Root has zero incoming message, without neighbor bias.
"""
    def __init__(self,dim,dropout):
        super().__init__();self.root=nn.Linear(dim,dim);self.neighbor=nn.Linear(dim,dim,bias=False)
        self.norm=nn.LayerNorm(dim);self.ff=nn.Sequential(nn.Linear(dim,4*dim),nn.SiLU(),nn.Dropout(dropout),nn.Linear(4*dim,dim))
        self.final=nn.LayerNorm(dim);self.drop=nn.Dropout(dropout)
    def forward(self,x,parents):
        message=self.neighbor(x[:,parents]);message=torch.cat((torch.zeros_like(message[:,:1]),message[:,1:]),1)
        h=self.norm(self.root(x)+message)
        return self.final(h+self.drop(self.ff(h)))

class ExplorationL0(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=copy.deepcopy(config);c=config;d=c['hidden_dim']
        if (c['node_budget'],c['new_nodes_per_round'],c['offset_heads'],d,c['sage_layers'])!=(48,[4,16,27],4,128,3):raise ValueError('Explicit reviewed 48-node /3-round schedule required')
        if c['cnn_channels']!=[12,24,32] or c['cnn_convolutions']!=[2,3,3]:raise ValueError('CNN depth/width changed')
        if not 0<c['max_path_mm'] or c['boundary_backtrack_steps']<1:raise ValueError('Explicit spatial limits required')
        self.cnn=OrganPyramid(c['cnn_channels'],c['checkpoint_cnn'])
        self.project=nn.Sequential(nn.Linear(sum(c['cnn_channels']),d),nn.LayerNorm(d),nn.SiLU())
        self.offset_head=nn.Sequential(nn.Linear(d*3,d),nn.SiLU(),nn.Linear(d,c['offset_heads']*3))
        self.spatial_offsets=c.get('offset_parameterization','legacy')=='spatial_state_spread_v2'
        if c.get('offset_parameterization','legacy') not in ('legacy','spatial_state_spread_v2'):
            raise ValueError('Unknown offset parameterization')
        if self.spatial_offsets:
            # Geometry conditions the search only; it is not a new CT descriptor.
            # Preserve the baseline RNG stream for all shared CNN/SAGE/L1/L2
            # parameters so the paired DEBUG comparison changes this path only.
            with torch.random.fork_rng(devices=[]):
                self.search_position=nn.Linear(6,d*3,bias=False)
            # Four maximally separated directions on a 3D sphere. This is an
            # initialization, NOT learned anatomy or a required sampling shape.
            directions=torch.tensor([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]])/3**.5
            with torch.no_grad():self.offset_head[-1].bias.copy_(torch.atanh(directions).flatten())
        self.blocks=nn.ModuleList([MeanSAGEResidual(d,c['dropout']) for _ in range(3)])
        self.fuse=nn.Sequential(nn.Linear(d*4,d*2),nn.LayerNorm(d*2),nn.SiLU(),nn.Dropout(c['dropout']),nn.Linear(d*2,d))
        parents=[0];previous=[0]
        for r,n in enumerate(c['new_nodes_per_round']):
            p=[previous[i%len(previous)] for i in range(n)];h=[i//len(previous) for i in range(n)]
            if max(h)>=c['offset_heads']:raise ValueError('Branch budget mismatch')
            self.register_buffer(f'parent_{r}',torch.tensor(p),persistent=False)
            self.register_buffer(f'head_{r}',torch.tensor(h),persistent=False)
            previous=list(range(len(parents),len(parents)+n));parents.extend(p)
        self.register_buffer('parents',torch.tensor(parents),persistent=False)
        self.register_buffer('edges',torch.stack((torch.tensor(parents[1:]),torch.arange(1,len(parents)))),persistent=False)
        self.register_buffer('factors',2.**-torch.arange(c['boundary_backtrack_steps']+1),persistent=False)
    def features(self,maps,pos,ids):
        return self.project(torch.cat([sample_map(x,m,pos,ids,stride) for (x,m),stride in zip(maps,self.config['feature_strides'])],-1))
    def forward(self,batch,*,return_trace=False):
        batch.validate();maps=self.cnn(batch.images,batch.organ)
        donor=self.features(maps,batch.donor_centers[:,None],batch.donor_index)[:,0]
        positions=batch.centers[:,None];x=self.features(maps,positions,batch.recipient_index)
        paths=[];scales=[];raw_offsets=[];step=self.config['max_path_mm']/3
        spacing=batch.spacing[batch.recipient_index][:,None,:]
        for r,block in enumerate(self.blocks):
            parents=getattr(self,f'parent_{r}');heads=getattr(self,f'head_{r}')
            context=x.mean(1,keepdim=True).expand(-1,len(parents),-1)
            inp=torch.cat((x[:,parents],donor[:,None].expand_as(context),context),-1)
            if self.spatial_offsets:
                relative=(positions[:,parents]-batch.centers[:,None])*spacing/self.config['max_path_mm']
                incoming=(positions[:,parents]-positions[:,self.parents[parents]])*spacing/step
                inp=inp+self.search_position(torch.cat((relative,incoming),-1))
            offsets=self.offset_head(inp).reshape(len(batch),len(parents),self.config['offset_heads'],3)
            delta=offsets[:,torch.arange(len(parents),device=x.device),heads].tanh()
            delta=delta/torch.linalg.vector_norm(delta,dim=-1,keepdim=True).clamp_min(1)*step
            origin=positions[:,parents]
            proposal=origin[:,:,None]+delta[:,:,None]/spacing[:,:,None]*self.factors[None,None,:,None]
            valid=inside(batch.organ,proposal,batch.recipient_index)
            if not bool(valid.any(-1).all()):raise ValueError('Organ-constrained exploration failed; no node drop/fallback')
            chosen=valid.to(torch.long).argmax(-1);scale=self.factors[chosen]
            new=origin+delta/spacing*scale[...,None]
            positions=torch.cat((positions,new),1)
            x=torch.cat((x,self.features(maps,positions[:,-len(parents):],batch.recipient_index)),1)
            x=block(x,self.parents[:x.shape[1]])
            if return_trace:paths.append(positions);scales.append(scale);raw_offsets.append(delta)
        pooled=x.mean(1);output=self.fuse(torch.cat((x[:,0],pooled,donor,(pooled-donor).abs()),-1))
        if not bool(torch.isfinite(output).all()):raise FloatingPointError('Nonfinite exploration output')
        if return_trace:return output,dict(positions=positions,round_positions=paths,step_scales=scales,offsets=raw_offsets,edges=self.edges,cnn_volumes=len(batch.images),queries=len(batch))
        return output
    def get_extra_state(self):return copy.deepcopy(self.config)
    def set_extra_state(self,state):
        if state!=self.config:raise ValueError('Exploration contract mismatch; old-model resume forbidden')
