"""Live aggregation and typed quotient; topology decisions have no gradient."""
import torch
from torch_geometric.utils import softmax

def compact(parent,n):
    if parent.dtype!=torch.long or parent.shape!=(n,) or (n and bool((parent<0).any())):
        raise ValueError('Invalid assignment')
    k=len(torch.unique(parent))
    if n and int(parent.max())!=k-1:raise ValueError('Noncompact assignment')
    return k

def mass_mean(x,parent,mass):
    k=compact(parent,len(x))
    if mass.shape!=(len(x),) or not bool(torch.isfinite(x).all() & torch.isfinite(mass).all() & (mass>0).all()):
        raise ValueError('Invalid live features/mass')
    dtype=torch.float64 if x.dtype==torch.float64 else torch.float32
    m=torch.zeros(k,device=x.device,dtype=dtype).index_add(0,parent,mass.to(dtype))
    sums=torch.zeros((k,x.shape[1]),device=x.device,dtype=dtype).index_add(0,parent,x.to(dtype)*mass[:,None].to(dtype))
    return (sums/m[:,None]).to(x.dtype),m

def reduce_bounds(values,parent,k,reduce):
    out=torch.full((k,*values.shape[1:]),float('inf') if reduce=='amin' else -float('inf'),device=values.device,dtype=values.dtype)
    index=parent.reshape(-1,*([1]*(values.ndim-1))).expand_as(values)
    return out.scatter_reduce(0,index,values,reduce=reduce,include_self=True)

def quotient(edge,ps,pd,*,same_type):
    if edge.dtype!=torch.long or edge.ndim!=2 or edge.shape[0]!=2:raise ValueError('COO expected')
    mapped=torch.stack((ps[edge[0]],pd[edge[1]]))
    keep=mapped[0]!=mapped[1] if same_type else torch.ones(edge.shape[1],device=edge.device,dtype=torch.bool)
    unique,inverse,counts=torch.unique(mapped[:,keep].T,sorted=True,dim=0,return_inverse=True,return_counts=True)
    witness=torch.full((len(unique),),edge.shape[1],device=edge.device,dtype=torch.long)
    witness.scatter_reduce_(0,inverse,torch.arange(edge.shape[1],device=edge.device)[keep],reduce='amin',include_self=True)
    return unique.T.contiguous(),counts,witness,(~keep).sum()

def undirected(edge,groups,weights=None):
    """Initial reverse duplicates count once; later supplied boundary weights sum."""
    keep=(edge[0]!=edge[1]) & (groups[edge[0]]==groups[edge[1]])
    pairs=edge[:,keep].sort(dim=0).values.T
    unique,inv=torch.unique(pairs,dim=0,sorted=True,return_inverse=True)
    w=torch.ones(len(unique),device=edge.device,dtype=torch.float32)
    if weights is not None:
        w=torch.zeros_like(w).index_add(0,inv,weights[keep].float())
    return unique.T.contiguous(),w

def mass_readout(x,mass,batch,gate,graph_count):
    if bool((mass<=0).any()):raise ValueError('Nonpositive readout mass')
    logits=gate(x).float()+mass.float().log()[:,None]
    alpha=softmax(logits,batch,num_nodes=graph_count)
    result=torch.zeros((graph_count,x.shape[1]),device=x.device,dtype=torch.float32)
    return result.index_add(0,batch,alpha*x.float()).to(x.dtype)
