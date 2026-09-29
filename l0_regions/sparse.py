"""Fixed-order sparse sums for reproducible region training.

Rows are never split. Temporary edge features are bounded by a workspace target;
an individual longer row is processed intact. Backward multiplies by the stored
transpose with the same segmented reduction, avoiding atomic scatter sums.
"""
import torch


def segmented_mm(matrix, x, workspace_bytes=64*2**20):
    if matrix.layout != torch.sparse_csr or x.ndim != 2:
        raise ValueError('CSR and rank-two features required')
    ptr=matrix.crow_indices();cols=matrix.col_indices();values=matrix.values()
    # Structure is immutable: prepare row chunks once on this CSR, outside the
    # recurrent feature calculation. CPU lengths are topology, not learned data.
    topology=getattr(matrix,'_segment_chunks',None)
    key=(x.shape[1],x.element_size(),workspace_bytes)
    if topology is None or topology[0]!=key:
        offsets=ptr.cpu().tolist();limit=max(1,workspace_bytes//(x.shape[1]*x.element_size()))
        chunks=[];start=0
        for stop in range(1,len(offsets)):
            if offsets[stop]-offsets[start]>limit and stop>start+1:
                chunks.append((start,stop-1,offsets[start],offsets[stop-1]));start=stop-1
        if start<len(offsets)-1:chunks.append((start,len(offsets)-1,offsets[start],offsets[-1]))
        matrix._segment_chunks=(key,chunks)
    else:chunks=topology[1]
    out=x.new_empty((matrix.shape[0],x.shape[1]))
    for begin,end,left,right in chunks:
        messages=x.index_select(0,cols[left:right])*values[left:right,None]
        out[begin:end]=torch.segment_reduce(messages,'sum',lengths=ptr[begin+1:end+1]-ptr[begin:end])
    return out


class SparseMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx,matrix,transpose,x):
        ctx.save_for_backward(matrix,transpose)
        return segmented_mm(matrix,x)
    @staticmethod
    def backward(ctx,gradient):
        _,transpose=ctx.saved_tensors
        return None,None,segmented_mm(transpose,gradient)


def stable_spmm(matrix,transpose,x):
    return SparseMean.apply(matrix,transpose,x)
