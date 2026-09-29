"""Mandatory integrity checks shared by admission and diagnostic-only paths."""
import torch
from hiercp_v22.schema import LOCAL_NODE_TYPES,LOCAL_EDGE_TYPES

def validate_batch(batch):
    graph=batch.graph;count=len(batch)
    if count<1 or set(graph.node_types)!=set(LOCAL_NODE_TYPES) or set(graph.edge_types)!=set(LOCAL_EDGE_TYPES):raise ValueError('Invalid roles/batch')
    if 'sampled_counts' not in graph or 'relation_edge_counts' not in graph:raise ValueError('Missing fine coverage manifest')
    if graph.sampled_counts.shape!=(count,len(LOCAL_NODE_TYPES)) or graph.relation_edge_counts.shape!=(count,len(LOCAL_EDGE_TYPES)):raise ValueError('Invalid coverage manifest shape')
    for patches in (batch.source_patches,batch.target_patches):
        if patches.ndim!=5 or tuple(patches.shape[1:])!=(1,48,48,48) or not bool(torch.isfinite(patches).all()):raise ValueError('Invalid CT data')
    if batch.indices.shape!=(count,) or len(torch.unique(batch.indices))!=count:raise ValueError('Invalid record coverage')
    if batch.source_index.shape!=(count,) or bool(((batch.source_index<0)|(batch.source_index>=len(batch.source_patches))).any()):raise ValueError('Invalid source index')
    for k in LOCAL_NODE_TYPES:
        node=graph[k];meta=batch.sidecar[k];n=node.num_nodes
        if node.grid.shape!=(n,3) or not bool(torch.isfinite(node.grid).all()):raise ValueError('Invalid coordinates')
        if bool((node.grid.abs()>1.001).any()):raise ValueError('Coordinates outside original normalized patch')
        if node.batch.shape!=(n,) or node.batch.dtype!=torch.long or bool(((node.batch<0)|(node.batch>=count)).any()):raise ValueError('Invalid pair ownership')
        if meta['pos_mm'].shape!=(n,3) or not bool(torch.isfinite(meta['pos_mm']).all()):raise ValueError('Invalid mm sidecar')
        if meta['stable_id'].shape!=(n,) or meta['stable_id'].dtype!=torch.long or bool((meta['stable_id']<0).any()):raise ValueError('Invalid canonical IDs')
        if len(torch.unique(torch.stack((node.batch,meta['stable_id']),1),dim=0))!=n:raise ValueError('Duplicate fine coverage')
        if 'full_id' in node and not torch.equal(node.full_id,meta['stable_id']):raise ValueError('Canonical sidecar identity mismatch')
        counts=torch.bincount(node.batch,minlength=count)
        if k!='target_context' and bool((counts==0).any()):raise ValueError('Required role absent')
        if 'sampled_counts' in graph:
            expected=graph.sampled_counts[:,LOCAL_NODE_TYPES.index(k)]
            if not torch.equal(counts,expected):raise ValueError('Fine node coverage differs from materialized view')
    for edge in LOCAL_EDGE_TYPES:
        a,_,b=edge;e=graph[edge].edge_index
        if e.dtype!=torch.long or e.ndim!=2 or e.shape[0]!=2:raise ValueError('Invalid edge index')
        if e.numel() and bool((e<0).any() | (e[0]>=graph[a].num_nodes).any() | (e[1]>=graph[b].num_nodes).any()):raise ValueError('Edge endpoint outside nodes')
        if not torch.equal(graph[a].batch[e[0]],graph[b].batch[e[1]]):raise ValueError('Cross-pair edge')
    if 'relation_edge_counts' in graph:
        expected=graph.relation_edge_counts.sum(0)
        actual=torch.tensor([graph[e].edge_index.shape[1] for e in LOCAL_EDGE_TYPES],device=expected.device)
        if not torch.equal(expected,actual):raise ValueError('Fine edge coverage differs from materialized view')
