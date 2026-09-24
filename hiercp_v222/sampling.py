"""CNN-conditioned PPR selection and batched A* path closure on a spatial graph.

Topology decisions are discrete/no-grad. Selected CNN values keep their gradient.
There is no target, annotation, case identity or donor argument in this module.
All candidate vertices and radius edges participate; no neighbor truncation.
"""
import itertools
import math
import torch
from torch import nn
from torch.nn import functional as F


def validate_sampler(cfg):
    if cfg.get('method') != 'cnn_ppr_astar_v1':
        raise ValueError('Explicit PPR/A* sampling contract required')
    if not 0 < cfg['continuation_probability'] < 1:
        raise ValueError('PPR continuation probability must be inside (0,1)')
    if not 0 < cfg['retained_ppr_mass'] <= 1:
        raise ValueError('Retained mass must be inside (0,1]')
    if not 0 < cfg['ppr_l1_error'] < 1e-3:
        raise ValueError('Explicit numerical accuracy required')
    if cfg['path_halo_hops'] != 1 or cfg['outer_targets'] != '26_lattice_directions':
        raise ValueError('Full 3D direction set and one radius-neighborhood halo required')
    if cfg['cost'] != 'distance_mm_times_2_minus_cosine':
        raise ValueError('Unsupported path/transition cost')


@torch.no_grad()
def personalized_rank(cost, neighbors, valid, seed, continuation, error):
    """Symmetric conductances 1/cost; deterministic gather-based Markov steps.

    Infinity marks nonedges. Contraction supplies a posteriori L1 error bound,
    not a graph/sample cap. Exhausting the mathematically sufficient iterations
    without convergence raises, rather than publishing approximate success.
    """
    conductance = torch.where(valid[None], cost.reciprocal(), 0.)
    degree = conductance.sum(-1)
    if not torch.isfinite(degree).all() or not (degree > 0).all():
        raise ValueError('Finite connected candidate graph with nonzero degrees required')
    teleport = torch.zeros_like(degree); teleport[:, seed] = 1.
    rank = teleport.clone()
    bound = math.inf
    limit = math.ceil(math.log(error / 2.) / math.log(continuation)) + 2
    for iteration in range(1, limit + 1):
        incoming = ((rank / degree)[:, neighbors] * conductance).sum(-1)
        updated = (1-continuation)*teleport + continuation*incoming
        # A scalar convergence check every 8 steps avoids a sync per iteration.
        if iteration % 8 == 0 or iteration == limit:
            bound = float((updated-rank).abs().sum(-1).max()) * continuation / (1-continuation)
            if bound <= error:
                rank = updated
                break
        rank = updated
    else:
        raise RuntimeError(f'PPR failed L1 error contract: {bound} > {error}')
    return rank, degree, dict(iterations=iteration, l1_error_bound=bound)


@torch.no_grad()
def select_mass(rank, degree, mass, seed):
    # PPR/degree is the local sweep density, not a global popularity score.
    order = torch.argsort(rank / degree, dim=1, descending=True, stable=True)
    ordered_mass = rank.gather(1, order)
    keep = ordered_mass.cumsum(1) - ordered_mass < mass
    selected = torch.zeros_like(keep).scatter(1, order, keep)
    selected[:, seed] = True
    return selected


@torch.no_grad()
def batched_astar(positions, neighbors, valid, cost, seed, targets):
    """Exact A*, parallel across every graph x outer target on the input device.

    Euclidean heuristic is consistent because each edge cost >= physical length.
    Each path expands at most N vertices. No CPU graph loop or hidden early cap.
    Returns path membership, path costs and parent chains for reproducible QA.
    """
    b, n, _ = cost.shape; t = len(targets); device = cost.device
    if not t:
        raise ValueError('Outer destinations required')
    graph = torch.arange(b, device=device).repeat_interleave(t)
    goal = targets.repeat(b); rows = torch.arange(len(graph), device=device)
    heuristic = torch.linalg.vector_norm(positions[None]-positions[targets, None], dim=-1)
    heuristic = F.pad(heuristic.repeat(b, 1),(0,1),value=torch.inf)
    g = torch.full_like(heuristic, torch.inf); g[:, seed] = 0
    frontier = torch.full_like(g, torch.inf); frontier[:, seed] = heuristic[:, seed]
    closed = torch.zeros_like(g, dtype=torch.bool)
    parent = torch.full(g.shape, -1, dtype=torch.long, device=device)
    done = torch.zeros(len(rows), dtype=torch.bool, device=device)
    for iteration in range(n):
        current = frontier.argmin(1)
        reachable = torch.isfinite(frontier[rows, current])
        active = ~done & reachable
        done |= active & (current == goal)
        expand = active & ~done
        # Nonedges address a dedicated sentinel column. Valid neighbors are
        # unique per row, so fixed-shape scatter replaces nonzero()/variable
        # indexing and its CPU synchronization in every A* expansion.
        nb = torch.where(valid[current],neighbors[current],n)
        candidate = g[rows, current, None] + cost[graph, current]
        improve = (expand[:, None] & valid[current] & ~closed[rows[:, None], nb]
                   & (candidate < g[rows[:, None], nb]))
        g.scatter_(1,nb,torch.where(improve,candidate,g.gather(1,nb)))
        parent.scatter_(1,nb,torch.where(improve,current[:,None],parent.gather(1,nb)))
        frontier.scatter_(1,nb,torch.where(improve,candidate+heuristic.gather(1,nb),frontier.gather(1,nb)))
        closed[rows, current] |= active
        frontier[rows, current] = torch.inf
        if iteration % 16 == 0 or iteration == n-1:
            if bool(done.all()):
                break
            if bool((~done & ~reachable).any()):
                raise ValueError('No spatial path to an outer destination')
    if not bool(done.all()):
        raise RuntimeError('A* failed despite exhausting all candidate vertices')
    membership = torch.zeros((b, n), dtype=torch.bool, device=device)
    current = goal.clone(); path_columns = []
    for _ in range(n):
        path_columns.append(current.clone())
        membership[graph, current] = True
        arrived = current == seed
        if bool(arrived.all()):
            break
        previous = parent[rows, current]
        if bool((~arrived & (previous < 0)).any()):
            raise RuntimeError('Broken A* predecessor chain')
        current = torch.where(arrived, current, previous)
    else:
        raise RuntimeError('Cyclic A* predecessor chain')
    return membership, dict(costs=g[rows, goal].reshape(b,t),
                           paths=torch.stack(path_columns, 1).reshape(b,t,-1),
                           expansions=closed.sum(1).reshape(b,t), iterations=iteration+1)


@torch.no_grad()
def pack_selection(selected, neighbors, valid):
    """Padded graph batch, exact induced edges, no graph truncation."""
    b, n = selected.shape; device = selected.device
    counts = selected.sum(1); width = int(counts.max())
    ids = torch.argsort(selected.long(), descending=True, stable=True, dim=1)[:, :width]
    alive = torch.arange(width, device=device)[None] < counts[:, None]
    mapping = torch.full((b,n), -1, dtype=torch.long, device=device)
    destination = torch.arange(width, device=device).expand(b,-1)
    mapping.scatter_(1, ids, torch.where(alive, destination, -1))
    original = neighbors[ids]
    batch = torch.arange(b, device=device)[:,None,None]
    mapped = mapping[batch, original]
    edge_valid = valid[ids] & (mapped >= 0) & alive[:,:,None]
    # One explicit self loop for each live node, not for padding.
    self_ids = destination[:,:,None]
    packed = torch.cat((mapped.clamp_min(0), self_ids), -1)
    mask = torch.cat((edge_valid, alive[:,:,None]), -1)
    return dict(ids=ids, alive=alive, neighbors=packed, valid=mask,
                node_counts=counts, edge_counts=edge_valid.sum((1,2)))


class PPRPathSampler(nn.Module):
    def __init__(self, positions, neighbors, valid, cfg):
        super().__init__(); validate_sampler(cfg); self.cfg = dict(cfg)
        self.register_buffer('positions', positions.float(), persistent=False)
        self.register_buffer('neighbors', neighbors, persistent=False)
        ids = torch.arange(len(positions))[:,None]
        valid = valid & (neighbors != ids)
        self.register_buffer('valid', valid, persistent=False)
        self.candidate_edges = int(valid.sum())
        distances = (positions[:,None]-positions[neighbors]).norm(dim=-1)
        self.register_buffer('distances', distances, persistent=False)
        self.seed = int(positions.norm(dim=-1).argmin())
        directions = torch.tensor([p for p in itertools.product((-1.,0.,1.), repeat=3)
                                   if p != (0.,0.,0.)])
        directions = F.normalize(directions, dim=-1)
        radius = positions.norm(dim=-1).max()
        # Nearest candidate to the sphere endpoint in each face/edge/corner direction.
        target = torch.cdist(directions*radius, positions).argmin(1).unique(sorted=True)
        target = target[target != self.seed]
        self.register_buffer('targets', target, persistent=False)
        self.last_audit = None
        self.capture = False
        self.last_capture = None

    @torch.no_grad()
    def forward(self, features):
        with torch.autocast(features.device.type, enabled=False):
            f = F.normalize(features.detach().float(), dim=-1)
            if not torch.isfinite(f).all():
                raise FloatingPointError('Nonfinite CNN features')
            b,n,c = f.shape
            # Execution tiles bound the temporary B*N*degree*C gather, not N/E.
            free = torch.cuda.mem_get_info(f.device)[0] if f.is_cuda else 512*1024**2
            tile = max(1, min(n, int(free*.1/(b*self.neighbors.shape[1]*c*4*3))))
            parts = []
            for start in range(0,n,tile):
                stop = min(n,start+tile)
                cosine = (f[:,start:stop,None]*f[:,self.neighbors[start:stop]]).sum(-1).clamp(-1,1)
                costs = self.distances[None,start:stop]*(2-cosine)
                parts.append(costs.masked_fill(~self.valid[None,start:stop], torch.inf))
            cost = torch.cat(parts,1)
            del parts, costs, cosine, f
            rank, degree, convergence = personalized_rank(cost,self.neighbors,self.valid,self.seed,
                self.cfg['continuation_probability'],self.cfg['ppr_l1_error'])
            core = select_mass(rank,degree,self.cfg['retained_ppr_mass'],self.seed)
            path, path_audit = batched_astar(self.positions,self.neighbors,self.valid,cost,self.seed,self.targets)
            # Radius halo gives paths surrounding context, not just a thin chain.
            halo = (path[:,self.neighbors] & self.valid[None]).any(-1)
            selected = core | path | halo
            packed = pack_selection(selected,self.neighbors,self.valid)
            self.last_audit = dict(candidate_nodes=n,candidate_edges=self.candidate_edges,
                node_counts=packed['node_counts'].detach(),edge_counts=packed['edge_counts'].detach(),
                ppr_core_nodes=core.sum(1),path_nodes=path.sum(1),halo_nodes=halo.sum(1),
                retained_mass=(rank*core).sum(1),outer_targets=len(self.targets),
                ppr=convergence,astar_iterations=path_audit['iterations'],
                astar_expansions=path_audit['expansions'].detach())
            if self.capture:
                self.last_capture = dict(selected=selected.detach(),core=core.detach(),path=path.detach(),
                    halo=halo.detach(),rank=rank.detach(),cost=cost.detach(),**path_audit)
            return packed


def padded_gat(layer, x, neighbors, valid, alive, chunk_nodes, checkpoint_chunks):
    """Batched per-graph adjacency. Exact GATv2, with padding excluded."""
    from torch.utils.checkpoint import checkpoint
    b,n,_ = x.shape
    left = layer.lin_l(x).reshape(b,n,layer.heads,layer.out_channels)
    right = layer.lin_r(x).reshape(b,n,layer.heads,layer.out_channels)
    batch = torch.arange(b,device=x.device)[:,None,None]
    pieces=[]
    for start in range(0,n,chunk_nodes):
        stop=min(n,start+chunk_nodes)
        def run(left,right,start=start,stop=stop):
            source=left[batch,neighbors[:,start:stop]]
            logits=F.leaky_relu(source+right[:,start:stop,None],layer.negative_slope)
            logits=(logits*layer.att).sum(-1).float()
            mask=valid[:,start:stop]
            # Dummy padding rows have zero output; no all-inf softmax/NaN.
            logits=logits.masked_fill(~mask[...,None],-torch.inf)
            logits=torch.where(alive[:,start:stop,None,None],logits,torch.zeros_like(logits))
            alpha=F.dropout(logits.softmax(2),p=layer.dropout,training=layer.training)
            message=(source*alpha[...,None]).sum(2).flatten(2)
            if layer.bias is not None:message=message+layer.bias
            return message*alive[:,start:stop,None]
        pieces.append(checkpoint(run,left,right,use_reentrant=False) if checkpoint_chunks else run(left,right))
    return torch.cat(pieces,1)
