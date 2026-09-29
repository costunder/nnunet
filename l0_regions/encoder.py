"""Live CNN -> fixed pooling -> SAGE2 -> fixed pooling -> SAGE1 -> 128D.

Only the offline artifact contains partition decisions. No fine edge argument,
no epoch-dependent resampling, no online partition/quotient construction.
"""
import copy
import torch
from torch import nn
from torch_geometric.nn import HeteroConv
from hiercp_v22.schema import LOCAL_NODE_TYPES as NT, LOCAL_EDGE_TYPES as ET, SOURCE_LOCAL_NODE_TYPES
from hiercp_v222.deterministic_sampling import sample_nodes
from tools.v222_review_contracts import cnn_lattice,input_grid_to_feature_grid
from l0_ezsp.ops import mass_mean
from l0_ezsp.encoder import EZSPEncoder
from l0_sage.encoder import MeanAdjacencyCache,MeanSAGEConv
from .data import RegionBatch


def validate_batch(batch):
    if not isinstance(batch,RegionBatch) or len(batch)<1 or len(batch.bindings)!=len(batch):raise ValueError('Fixed region batch required')
    if set(batch.fine)!=set(NT) or len(batch.regions)!=2 or len(batch.graphs)!=2:raise ValueError('Incomplete hierarchy')
    if batch.indices.shape!=(len(batch),) or len(torch.unique(batch.indices))!=len(batch):raise ValueError('Observation coverage')
    if len({b['record_id'] for b in batch.bindings})!=len(batch):raise ValueError('Duplicate record')
    if [b['dataset_index'] for b in batch.bindings]!=batch.indices.tolist():raise ValueError('Record/index mismatch')
    if len(batch.admission_flags)!=len(batch) or batch.profile_exceeded!=any(batch.admission_flags):raise ValueError('Admission flag changed')
    if batch.source_index.shape!=(len(batch),) or batch.source_index.dtype!=torch.long or bool(((batch.source_index<0)|(batch.source_index>=len(batch.source_patches))).any()):raise ValueError('Source identity')
    for t in (batch.source_patches,batch.target_patches):
        if t.ndim!=5 or t.shape[1:]!=(1,48,48,48) or not bool(torch.isfinite(t).all()):raise ValueError('Invalid CT')
    for role,f in batch.fine.items():
        n=len(f['grid'])
        if f['grid'].shape!=(n,3) or f['pos_mm'].shape!=(n,3) or not bool(torch.isfinite(f['grid']).all()&torch.isfinite(f['pos_mm']).all()) or bool((f['grid'].abs()>1.001).any()):raise ValueError('Fine coordinates')
        if f['owner'].shape!=(n,) or f['owner'].dtype!=torch.long or bool(((f['owner']<0)|(f['owner']>=len(batch))).any()):raise ValueError('Pair identity')
        if f['stable_id'].shape!=(n,) or f['stable_id'].dtype!=torch.long or len(torch.unique(torch.stack((f['owner'],f['stable_id']),1),dim=0))!=n:raise ValueError('Canonical coverage')
        if role!='target_context' and bool((torch.bincount(f['owner'],minlength=len(batch))==0).any()):raise ValueError('Role coverage')
        mass=torch.ones(n,device=f['grid'].device);parent=f['parent'];owner=f['owner'];shell=f['shell']
        for level in range(2):
            r=batch.regions[level][role]
            if parent.dtype!=torch.long or parent.shape!=mass.shape or bool(((parent<0)|(parent>=len(r.mass))).any()):raise ValueError('Assignment coverage')
            if not torch.equal(r.owner[parent],owner) or not torch.equal(r.shell[parent],shell):raise ValueError('Mixed pair/shell')
            if not torch.equal(torch.zeros_like(r.mass).index_add(0,parent,mass),r.mass) or not bool(torch.isfinite(r.mass).all()&(r.mass>0).all()):raise ValueError('Mass coverage')
            if batch.graphs[level][role].num_nodes!=len(r.mass):raise ValueError('Coarse graph coverage')
            owner=r.owner;shell=r.shell;mass=r.mass
            if level==0:parent=r.parent
    for level,graph in enumerate(batch.graphs):
        if set(graph.edge_types)!=set(ET) or set(graph.node_types)!=set(NT):raise ValueError('Coarse schema')
        for rel in ET:
            e=graph[rel].edge_index
            if e.dtype!=torch.long or e.ndim!=2 or e.shape[0]!=2 or (e.numel() and bool((e<0).any()|(e[0]>=graph[rel[0]].num_nodes).any()|(e[1]>=graph[rel[2]].num_nodes).any())):raise ValueError('Coarse edge endpoint')
            if not torch.equal(batch.regions[level][rel[0]].owner[e[0]],batch.regions[level][rel[2]].owner[e[1]]):raise ValueError('Cross-pair relation')
    from l0_ezsp.ops import quotient
    for rel in ET:
        expected=quotient(batch.graphs[0][rel].edge_index,batch.regions[0][rel[0]].parent,
            batch.regions[0][rel[2]].parent,same_type=rel[0]==rel[2])[0]
        if not torch.equal(expected,batch.graphs[1][rel].edge_index):raise ValueError('Batch stage2 quotient mismatch')


class RegionSAGEEncoder(nn.Module):
    def __init__(self,reference,*,seed,resource_budget,allow_unvalidated_profile):
        super().__init__()
        if resource_budget is None:raise ValueError('Explicit diagnostic resource budget required')
        self.resource_budget=resource_budget;self.allow_unvalidated_profile=allow_unvalidated_profile
        self.adjacencies=[MeanAdjacencyCache(),MeanAdjacencyCache()]
        with torch.random.fork_rng(devices=[]):
            self.core=copy.deepcopy(reference)
            torch.random.default_generator.manual_seed(seed)
            if len(self.core.blocks)!=3 or self.core.hidden_dim!=128:raise ValueError('Preserve 2+1 /128D')
            for i,block in enumerate(self.core.blocks):
                block.conv=HeteroConv({e:MeanSAGEConv(128,relation=e,cache=self.adjacencies[0 if i<2 else 1]) for e in ET},aggr='sum')
        self.feature_lattice=cnn_lattice(self.core.dense_encoder)

    @property
    def dense_encoder(self):return self.core.dense_encoder
    @property
    def dense_batch_size(self):return self.core.dense_batch_size
    @dense_batch_size.setter
    def dense_batch_size(self,value):self.core.dense_batch_size=value

    def get_extra_state(self):raise RuntimeError('Diagnostic region model cannot export a production checkpoint')

    def forward(self,batch):
        from .resident import check_verified
        self.resource_budget.check()
        check_verified(batch)
        if batch.profile_exceeded and not self.allow_unvalidated_profile:raise ValueError('Unvalidated partition admission failed')
        source,target=self.core.encode_dense_maps(batch.source_patches,batch.source_index,batch.target_patches)
        shape,jump,origin=self.feature_lattice;x={}
        for role,f in batch.fine.items():
            fmap=source if role in SOURCE_LOCAL_NODE_TYPES else target
            grid=input_grid_to_feature_grid(f['grid'],feature_shape=shape,stride=jump,origin=origin)
            fine=sample_nodes(fmap,grid,f['owner'])
            pooled,mass=mass_mean(fine,f['parent'],torch.ones(len(fine),device=fine.device))
            if not torch.equal(mass,batch.regions[0][role].mass):raise ValueError('Live pooling coverage')
            x[role]=self.core.project[role](pooled)
        for i in range(2):self.adjacencies[i].prepare(batch.graphs[i])
        e1=batch.graphs[0].edge_index_dict;e2=batch.graphs[1].edge_index_dict
        for block in self.core.blocks[:2]:x=self.core._run_local_block(block,x,e1,{e:None for e in ET})
        read1=EZSPEncoder._readout(self.core,x,batch.regions[0],len(batch))
        x={role:mass_mean(x[role],r.parent,r.mass)[0] for role,r in batch.regions[0].items()}
        x=self.core._run_local_block(self.core.blocks[2],x,e2,{e:None for e in ET})
        read2=EZSPEncoder._readout(self.core,x,batch.regions[1],len(batch))
        pooled={k:(v+read2[k])*.5 for k,v in read1.items()}
        sc=[pooled[f'source_context_c{i}'] for i in range(3)]
        tc=[pooled[f'target_context_c{i}'] for i in range(3)]
        core=self.core
        tumor=core.tumor_fuse(pooled['tumor_surface'])
        source_context=core.source_context_fuse(core._pair(core.context_shell_fuse['source_context'](torch.cat(sc,-1)),pooled['source_liver_surface']))
        target_context=core.target_context_fuse(core._pair(core.context_shell_fuse['target_context'](torch.cat(tc,-1)),pooled['target_liver_surface']))
        sr=core.source_relation(core._pair(tumor,sc[0]));tr=core.target_relation(core._pair(tumor,tc[0]))
        output=core.final_fuse(torch.cat((tumor,source_context,target_context,sr,tr,(sr-tr).abs()),-1))
        if output.shape!=(len(batch),128) or not bool(torch.isfinite(output).all()):raise FloatingPointError('Region embedding')
        self.resource_budget.check()
        return output
