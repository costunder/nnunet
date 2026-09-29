from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .backend import official
from .ops import mass_mean,reduce_bounds,compact,undirected

class CoarseningConstraintError(RuntimeError):
    def __init__(self,message,stats):super().__init__(message);self.stats=stats

class CoarseningBudgetError(CoarseningConstraintError):pass

@dataclass
class Region:
    owner:torch.Tensor
    shell:torch.Tensor
    mass:torch.Tensor
    lower:torch.Tensor
    upper:torch.Tensor
    stable_id:torch.Tensor
    adjacency:torch.Tensor
    boundary:torch.Tensor

@dataclass
class Partition:
    parent:torch.Tensor
    region:Region
    stats:dict
    violations:list

@torch.no_grad()
def partition(x,region,*,role,level,cfg,diagnostics=False,measure=False):
    if x.device.type!='cuda':raise ValueError('Official adapter requires CUDA; no CPU fallback')
    n=len(x)
    if level not in (1,2):raise ValueError('Exactly two coarsening scales required')
    if any(v.shape!=(n,) or v.dtype!=torch.long for v in (region.owner,region.shell,region.stable_id)):
        raise ValueError('Exact per-node owner/shell/canonical IDs required')
    if bool((region.owner<0).any() | (region.stable_id<0).any()):raise ValueError('Negative identity')
    if len(region.mass)!=n or not bool(torch.isfinite(x).all()):raise ValueError('Invalid partition feature input')
    if not bool(torch.isfinite(region.mass).all() & (region.mass>=1).all()):raise ValueError('Invalid original-node mass')
    groups=region.owner*4+region.shell+1
    e=region.adjacency;w=region.boundary
    if e.dtype!=torch.long or e.ndim!=2 or e.shape[0]!=2 or w.shape!=(e.shape[1],):raise ValueError('Malformed merge adjacency')
    if e.numel() and bool((e<0).any() | (e>=n).any()):raise ValueError('Adjacency endpoint outside nodes')
    if not bool(torch.isfinite(w).all() & (w>0).all()):raise ValueError('Invalid boundary mass')
    if region.lower.shape!=(n,3) or region.upper.shape!=(n,3) or not bool(torch.isfinite(region.lower).all() & torch.isfinite(region.upper).all()):
        raise ValueError('Finite original-member bounds required')
    if bool((region.lower>region.upper).any()):raise ValueError('Inverted physical bounds')
    if bool((groups[e[0]]!=groups[e[1]]).any()):raise ValueError('Mixed pair/role/shell partition edges')
    u=F.normalize(x.detach().float(),dim=1,eps=cfg['partition']['features']['eps'])
    merge,wcc=official()
    merge_seconds=0.
    if n<2 or e.shape[1]==0:
        parent=torch.arange(n,device=x.device);depth=0
    else:
        # Stable input order resolves index-based ties without changing the kernel.
        order=torch.argsort(region.stable_id,stable=True)
        order=order[torch.argsort(groups[order],stable=True)]
        inv=torch.empty_like(order);inv[order]=torch.arange(n,device=x.device)
        ordered_edges=inv[e].sort(dim=0).values
        edge_order=torch.argsort(ordered_edges[0]*n+ordered_edges[1],stable=True)
        if diagnostics or measure:
            from time import perf_counter
            torch.cuda.synchronize();merge_started=perf_counter()
        # Official WCC uses randperm. Isolate it from dropout/augmentation RNG.
        with torch.random.fork_rng(devices=[x.device.index]):
            with torch.cuda.device(x.device):torch.cuda.manual_seed(0)
            ordered_parent,depth,terminal=merge(u[order],region.mass[order].float(),ordered_edges[:,edge_order],w[edge_order],
                cfg['partition'][f'reg_scale{level}'],**cfg['backend']['options'])
        parent=ordered_parent[inv].long()
        if diagnostics or measure:torch.cuda.synchronize();merge_seconds=perf_counter()-merge_started
    k=compact(parent,n)
    means,mass=mass_mean(u,parent,region.mass)
    sse=torch.zeros(k,device=x.device).index_add_(0,parent,(u-means[parent]).square().sum(1)*region.mass)
    variance=sse/mass
    lower=reduce_bounds(region.lower,parent,k,'amin');upper=reduce_bounds(region.upper,parent,k,'amax')
    diagonal=(upper-lower).norm(dim=1)
    # Nonfloating reductions retain exact discrete group/ID information.
    def minimum(v):
        out=torch.full((k,),torch.iinfo(torch.long).max,device=x.device,dtype=torch.long)
        return out.scatter_reduce_(0,parent,v,reduce='amin',include_self=True)
    owner=minimum(region.owner);shell=minimum(region.shell);stable=minimum(region.stable_id)
    violations=[]
    if not torch.equal(owner[parent],region.owner) or not torch.equal(shell[parent],region.shell):violations.append('group_crossing')
    internal=e[:,parent[e[0]]==parent[e[1]]]
    if n:
        with torch.random.fork_rng(devices=[x.device.index]):
            with torch.cuda.device(x.device):torch.cuda.manual_seed(0)
            component,_=wcc(n,internal.clone())
        # With only within-parent edges, equal component counts imply connectivity.
        if len(torch.unique(component))!=k:violations.append('disconnected_cluster')
    limits=cfg['diagnostic_profile_from_original_NOT_VALIDATED'];j=level-1
    bound=limits['bbox_context_diagonal_mm' if 'context' in role else 'bbox_surface_diagonal_mm'][j]
    if bool((diagonal>bound).any()):violations.append('bbox')
    if bool((variance>limits['normalized_feature_sse_per_mass'][j]).any()):violations.append('variance')
    key=role+'_per_shell' if 'context' in role else role
    coarse_groups=owner*4+shell+1
    _,counts=torch.unique(coarse_groups,return_counts=True)
    if bool((counts>limits['caps_by_role'][key][j]).any()):violations.append('role_shell_nodes')
    ce,cw=undirected(parent[e],coarse_groups,w)
    result=Region(owner,shell,mass,lower,upper,stable,ce,cw)
    def summary(v):return dict(min=float(v.min()),max=float(v.max()),mean=float(v.float().mean())) if v.numel() else None
    from .vendor.torch_graph_components.merge import edge_merge_energy
    energy=edge_merge_energy(u,region.mass.float(),e,w,cfg['partition'][f'reg_scale{level}'],sharding=cfg['backend']['options']['sharding']) if e.shape[1] else w
    stats=dict(role=role,level=level,input_nodes=n,output_nodes=k,round_depth=int(depth),
        mass=summary(mass),boundary_weight=summary(w),initial_merge_energy=summary(energy),feature_norm=summary(x.detach().float().norm(dim=1)),
        near_zero_norms=int((x.detach().float().norm(dim=1)<cfg['partition']['features']['eps']).sum()),
        max_bbox_mm=float(diagonal.max()) if k else 0.,max_variance=float(variance.max()) if k else 0.,
        input_mass=float(region.mass.sum()),output_mass=float(mass.sum()),
        groups=[dict(group=int(g),nodes=int(c)) for g,c in zip(*torch.unique(coarse_groups,return_counts=True))],
        violations=violations)
    if diagnostics or measure:stats['official_merge_seconds']=merge_seconds
    if diagnostics:
        with torch.random.fork_rng(devices=[x.device.index]):
            with torch.cuda.device(x.device):torch.cuda.manual_seed(0)
            initial_cc,_=wcc(n,e.clone())
            final_cc,_=wcc(k,ce.clone())
        # Terminal energies inspect the returned partition, not a replacement algorithm.
        terminal_energy=edge_merge_energy(means,mass,ce,cw,cfg['partition'][f'reg_scale{level}']) if ce.shape[1] else cw
        details=[]
        for group in torch.unique(groups).tolist():
            before=groups==group;after=coarse_groups==group
            total_mass=float(region.mass[before].sum());sizes=mass[after]
            keys,frequency=torch.unique(sizes,return_counts=True)
            selected_edges=coarse_groups[ce[0]]==group
            values=terminal_energy[selected_edges]
            reason=('no_remaining_adjacency' if not len(values) else
                    'no_energy_admissible_merge' if bool((values>0).all()) else 'eligible_edges_at_return')
            if len(values) and bool((values<=0).any()) and depth>=cfg['backend']['options']['max_iterations']:reason='iteration_limit_with_eligible_edges'
            def excess(mask):
                bad=after & mask
                return {'clusters':int(bad.sum()),'fine_nodes':float(mass[bad].sum()),
                    'fine_node_fraction':float(mass[bad].sum())/total_mass if total_mass else 0.}
            details.append(dict(owner=group//4,shell=group%4-1,
                input_nodes=int(before.sum()),clusters=int(after.sum()),
                connected_components_before=len(torch.unique(initial_cc[before])),
                connected_components_after=len(torch.unique(final_cc[after])),
                cluster_size_histogram={str(int(v)):int(c) for v,c in zip(keys.tolist(),frequency.tolist())},
                cluster_size_quantiles=torch.quantile(sizes,torch.tensor([0.,.25,.5,.75,1.],device=x.device)).tolist(),
                bbox_excess=excess(diagonal>bound),variance_excess=excess(variance>limits['normalized_feature_sse_per_mass'][j]),
                either_excess=excess((diagonal>bound)|(variance>limits['normalized_feature_sse_per_mass'][j])),
                termination_condition=reason,termination_basis='returned partition energies/adjacency and official depth; no invented per-group iteration trace'))
        stats.update(group_diagnostics=details,official_merge_seconds=merge_seconds)
    return Partition(parent,result,stats,violations)
