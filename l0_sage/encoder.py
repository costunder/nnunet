"""Relation-wise PyG GraphSAGE mean on the unchanged complete fine graph.

Sparse D^-1 A multiplication is exactly neighbor mean, including duplicate
edge multiplicity. No sampling, coarsening or added learned inputs.
"""
import copy
import torch
from torch import nn
from torch_geometric.nn import SAGEConv,HeteroConv
from hiercp_v22.schema import LOCAL_EDGE_TYPES
from l0_ezsp.validation import validate_batch

def mean_adjacency(edge,source_count,target_count):
    if edge.dtype!=torch.long or edge.ndim!=2 or edge.shape[0]!=2:raise ValueError('Complete int64 COO required')
    if edge.numel() and bool((edge<0).any() | (edge[0]>=source_count).any() | (edge[1]>=target_count).any()):raise ValueError('Invalid endpoint')
    degree=torch.bincount(edge[1],minlength=target_count).float()
    values=degree[edge[1]].reciprocal()
    # Summing duplicate contributions preserves every original edge's mass.
    matrix=torch.sparse_coo_tensor(edge.flip(0),values,(target_count,source_count),device=edge.device).coalesce()
    return matrix.to_sparse_csr(),degree

class MeanAdjacencyCache:
    """One current batch, reused by three layers and backward recomputation."""
    def __init__(self):self.signature=None;self.matrices={};self.transposes={};self.edges={};self.builds=0

    def prepare(self,graph):
        signature=tuple((k,id(graph[k].edge_index),graph[k].edge_index._version,graph[k[0]].num_nodes,graph[k[2]].num_nodes) for k in LOCAL_EDGE_TYPES)
        if signature==self.signature:return
        matrices={}
        for k in LOCAL_EDGE_TYPES:
            matrices[k]=mean_adjacency(graph[k].edge_index,graph[k[0]].num_nodes,graph[k[2]].num_nodes)[0]
        self.matrices=matrices;self.transposes={};self.edges={k:graph[k].edge_index for k in LOCAL_EDGE_TYPES}
        self.signature=signature;self.builds+=1

class MeanSAGEConv(SAGEConv):
    def __init__(self,dim,*,relation,cache):
        # PyG linear/root computation. Sum on D^-1 A equals mean on raw COO
        # without constructing an E x hidden tensor.
        super().__init__((dim,dim),dim,aggr='sum',normalize=False,root_weight=True,project=False,bias=True)
        self.relation=relation;self.cache=cache

    def message_and_aggregate(self,adj_t,x):
        if not getattr(self,'stable_spmm',False):return super().message_and_aggregate(adj_t,x)
        from l0_regions.sparse import stable_spmm
        if self.relation not in self.cache.transposes:
            self.cache.transposes[self.relation]=adj_t.transpose(0,1).to_sparse_csr()
        return stable_spmm(adj_t,self.cache.transposes[self.relation],x[0])

    def forward(self,x,edge_index,edge_attr=None):
        if edge_attr is not None:raise ValueError('No handcrafted edge inputs')
        if self.cache.edges.get(self.relation) is not edge_index:raise ValueError('Full-batch adjacency not prepared')
        if any(t.dtype!=torch.float32 for t in x):raise TypeError('FP32 diagnostic; mixed precision not admitted')
        return super().forward(x,self.cache.matrices[self.relation])

class GraphSAGEEncoder(nn.Module):
    def __init__(self,reference,*,seed):
        super().__init__();self.adjacency=MeanAdjacencyCache()
        if len(reference.blocks)!=3 or reference.hidden_dim!=128:raise ValueError('Preserve 3 layers /128D')
        with torch.random.fork_rng(devices=[]):
            # PyG deepcopy can initialize temporary modules and consume CPU RNG.
            # Preserve the caller across copying as well as new layer creation.
            self.encoder=copy.deepcopy(reference)
            torch.random.default_generator.manual_seed(seed)
            for block in self.encoder.blocks:
                block.conv=HeteroConv({e:MeanSAGEConv(128,relation=e,cache=self.adjacency) for e in LOCAL_EDGE_TYPES},aggr='sum')

    @property
    def dense_encoder(self):return self.encoder.dense_encoder

    def forward(self,batch):
        validate_batch(batch);self.adjacency.prepare(batch.graph)
        output=self.encoder(batch)
        if output.shape!=(len(batch),128) or not bool(torch.isfinite(output).all()):raise FloatingPointError('Invalid SAGE embedding')
        return output

    def get_extra_state(self):raise RuntimeError('Diagnostic GraphSAGE checkpoint export is disabled')
