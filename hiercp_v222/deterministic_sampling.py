"""Batched border trilinear interpolation without CUDA grid_sample backward.

Same align_corners=True interpolation and node coordinates. Indexed gather
uses PyTorch's deterministic accumulation when deterministic algorithms are on.
No random seed, input geometry, resolution or feature dimension is changed.
"""
import torch

def sample_nodes(feature_map,grid,node_batch):
    if feature_map.ndim!=5 or grid.ndim!=2 or grid.shape[1]!=3 or node_batch.shape!=(len(grid),):
        raise ValueError('Expected [B,C,D,H,W], [N,3], [N]')
    if not feature_map.is_floating_point() or not grid.is_floating_point():
        raise TypeError('Floating CT features/coordinates required')
    if node_batch.dtype!=torch.long:raise TypeError('Graph ownership must be int64')
    batch,channels,depth,height,width=feature_map.shape
    if grid.device!=feature_map.device or node_batch.device!=feature_map.device:
        raise ValueError('Features, coordinates and graph ownership must share device')
    # FP32 matches the existing bfloat16 grid_sample promotion.
    compute=torch.float32 if feature_map.dtype in (torch.bfloat16,torch.float16) else feature_map.dtype
    values=feature_map.to(compute).permute(0,2,3,4,1).reshape(-1,channels)
    size=grid.new_tensor([width-1,height-1,depth-1],dtype=compute)
    xyz=((grid.to(compute)+1)*.5*size).clamp(min=0)
    xyz=torch.minimum(xyz,size)
    lower=xyz.floor().long();fraction=xyz-lower
    corner=torch.tensor([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                         [0,0,1],[1,0,1],[0,1,1],[1,1,1]],device=grid.device)
    coords=lower[:,None,:]+corner[None,:,:]
    limit=torch.tensor([width-1,height-1,depth-1],device=grid.device)
    coords=torch.minimum(coords,limit)
    weights=torch.where(corner[None,:,:].bool(),fraction[:,None,:],1-fraction[:,None,:]).prod(-1)
    index=((node_batch[:,None]*depth+coords[:,:,2])*height+coords[:,:,1])*width+coords[:,:,0]
    output=(values[index]*weights[:,:,None]).sum(1)
    return output.to(feature_map.dtype)
