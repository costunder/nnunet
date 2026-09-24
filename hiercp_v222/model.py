"""Full-scale CNN/GNN -> edge-conditioned L1 -> clustered L2 -> two CE losses.

Query targets have no forward argument. L1 support states are computed before
query application, with no query -> support/label edges, in every layer.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax
from hiercp.model import PatchFeatureEncoder3D
from .inputs import topology
from .clustering import fit_prototypes, alignment_loss, prototype_logits

def neighbor_table(edge, count):
    """All incoming edges plus self loops, padded only to the actual max degree."""
    pairs = edge.cpu().numpy()
    vertices = np.arange(count)
    source = np.concatenate((pairs[0], vertices)); dest = np.concatenate((pairs[1], vertices))
    order = np.lexsort((source, dest)); source, dest = source[order], dest[order]
    degrees = np.bincount(dest, minlength=count)
    start = np.cumsum(degrees)-degrees
    column = np.arange(len(dest))-start[dest]
    index = np.zeros((count, int(degrees.max())), dtype=np.int64)
    mask = np.zeros(index.shape, dtype=bool)
    index[dest,column] = source; mask[dest,column] = True
    return torch.from_numpy(index), torch.from_numpy(mask)

def chunked_gat(layer, x, neighbors, valid, chunk_nodes, *, checkpoint_chunks):
    """Exact GATv2 equations with bounded live edge tensors; no edge is dropped.

    The Python loop streams node ranges, while every range processes all graphs
    and neighbors in parallel. Each range is recomputed in backward to avoid
    retaining the full E*D attention tensor under deterministic CUDA indexing.
    """
    b, n, _ = x.shape
    left = layer.lin_l(x).reshape(b,n,layer.heads,layer.out_channels)
    right = layer.lin_r(x).reshape(b,n,layer.heads,layer.out_channels)
    pieces = []
    for start in range(0,n,chunk_nodes):
        stop = min(n,start+chunk_nodes)
        def run(left, right, start=start, stop=stop):
            source = left[:,neighbors[start:stop]]
            score = F.leaky_relu(source+right[:,start:stop,None], layer.negative_slope)
            score = (score*layer.att).sum(-1).float()
            score = score.masked_fill(~valid[None,start:stop,:,None], -torch.inf)
            alpha = F.dropout(score.softmax(2), p=layer.dropout, training=layer.training)
            return (source*alpha[...,None]).sum(2).flatten(2)
        value = checkpoint(run,left,right,use_reentrant=False) if checkpoint_chunks else run(left,right)
        pieces.append(value)
    result = torch.cat(pieces,dim=1)
    if layer.bias is not None: result = result+layer.bias
    return result

class Residual(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4*dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
        self.final = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, message):
        x = self.norm(x+self.drop(self.out(message)))
        return self.final(x+self.drop(self.ff(x)))

class LocalEncoder(nn.Module):
    def __init__(self, base, contract, sampling=None):
        super().__init__()
        m = base['model']; d = m['hidden_dim']
        grid, edge = topology(contract)
        self.register_buffer('grid', grid.clone(), persistent=False)
        self.register_buffer('edge', edge.clone(), persistent=False)
        neighbors, valid = neighbor_table(edge, len(grid))
        self.register_buffer('neighbors',neighbors,persistent=False)
        self.register_buffer('neighbor_valid',valid,persistent=False)
        self.dense = PatchFeatureEncoder3D(1, m['dense_base_channels'], m['dense_feature_dim'])
        self.project = nn.Sequential(nn.Linear(m['dense_feature_dim'], d), nn.LayerNorm(d), nn.SiLU())
        self.blocks = nn.ModuleList([GATv2Conv(d, d//m['heads'], heads=m['heads'],
                                             dropout=m['dropout'], add_self_loops=True) for _ in range(m['local_layers'])])
        self.updates = nn.ModuleList([Residual(d, m['dropout']) for _ in self.blocks])
        self.pool_gate = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        self.checkpointing = m['checkpoint_local_blocks']
        self.checkpoint_dense = m['checkpoint_dense_encoder']
        self._batched_edges = None
        self._sample_table = None
        self.execution_chunk_nodes = {}
        self.sampler = None
        if sampling is not None:
            from .sampling import PPRPathSampler
            positions = grid.flip(-1)*float(contract['outer_radius_mm'])
            self.sampler = PPRPathSampler(positions, neighbors, valid, sampling)
    def sample_features(self, fmap):
        """Exact align_corners=True trilinear sampling without CUDA grid_sample backward.

        Fixed topology means indices/weights are cached. PyTorch deterministic
        advanced-index backward handles shared feature-grid contributors.
        """
        shape = tuple(fmap.shape[2:]); key = (shape, str(fmap.device))
        if self._sample_table is None or self._sample_table[0] != key:
            sizes = torch.tensor(shape, device=fmap.device)
            pos = (self.grid.flip(-1)+1)*(sizes-1)/2
            lower = pos.floor().long(); upper = (lower+1).minimum(sizes-1)
            fraction = pos-lower
            bits = torch.tensor([[z,y,x] for z in (0,1) for y in (0,1) for x in (0,1)], device=fmap.device)
            corner = torch.where(bits[None].bool(), upper[:,None], lower[:,None])
            index = (corner[:,:,0]*shape[1]+corner[:,:,1])*shape[2]+corner[:,:,2]
            weights = torch.where(bits[None].bool(), fraction[:,None], 1-fraction[:,None]).prod(-1)
            self._sample_table = (key,index,weights)
        _, index, weights = self._sample_table
        # Stream the eight exact interpolation corners. Materializing
        # [B,C,N,8] and then casting it to FP32 caused nonlinear memory growth
        # at large physical batches. This visits every corner for every node.
        flat=fmap.flatten(2)
        sampled=None
        for corner in range(8):
            value=flat.index_select(2,index[:,corner]).float()*weights[None,None,:,corner]
            sampled=value if sampled is None else sampled+value
        return sampled.transpose(1,2).reshape(-1,fmap.shape[1]).to(fmap.dtype)
    def forward(self, patches):
        if patches.ndim != 5 or patches.shape[1:] != (1, 48, 48, 48):
            raise ValueError('CT-only [B,1,48,48,48] required')
        if self.checkpoint_dense and self.training:
            fmap = checkpoint(self.dense, patches, use_reentrant=False)
        else:
            fmap = self.dense(patches)
        b = len(patches); n = len(self.grid)
        features = self.sample_features(fmap).reshape(b,n,-1)
        if self.sampler is not None:
            from .sampling import padded_gat
            packed = self.sampler(features)
            batch = torch.arange(b,device=patches.device)[:,None]
            x = self.project(features[batch,packed['ids']])
            alive = packed['alive']; x = x*alive[:,:,None]
            width=x.shape[1]
            free=torch.cuda.mem_get_info(patches.device)[0] if patches.is_cuda else 512*1024**2
            live=b*packed['neighbors'].shape[-1]*x.shape[-1]*4*4
            chunk=max(1,min(width,int(free*.25/live)))
            for block,update in zip(self.blocks,self.updates):
                def run(value,block=block,update=update):
                    message=padded_gat(block,value,packed['neighbors'],packed['valid'],alive,chunk,
                        self.training and self.checkpointing)
                    return update(value,message)*alive[:,:,None]
                x=checkpoint(run,x,use_reentrant=False) if self.checkpointing and self.training else run(x)
            weights=self.pool_gate(x).masked_fill(~alive[:,:,None],-torch.inf).softmax(1)
            return (weights*x).sum(1)
        x = self.project(features)
        key = (b,str(patches.device))
        if key not in self.execution_chunk_nodes:
            # Explicit execution tiling, sized from current free VRAM. It is not
            # a node cap: every n is visited, including the final partial range.
            free = torch.cuda.mem_get_info(patches.device)[0] if patches.is_cuda else 512*1024**2
            # Full-graph RTX5070Ti measurements (execution_benchmark) showed
            # that 5% free memory / 32 copies created launch-bound tiny tiles.
            # Four live FP32-equivalent edge workspaces and a 25% free-memory
            # budget keep headroom for full-node states, L1/L2 and gradients.
            # Physical batch calibration still measures actual peak VRAM.
            live_bytes_per_node = b*self.neighbors.shape[1]*x.shape[-1]*4*4
            target = max(1,int(free*0.25/live_bytes_per_node))
            self.execution_chunk_nodes[key] = min(n,2**int(math.log2(target)))
        chunk = self.execution_chunk_nodes[key]
        for block, update in zip(self.blocks, self.updates):
            def run(value, block=block, update=update):
                message = chunked_gat(block,value,self.neighbors,self.neighbor_valid,chunk,
                                      checkpoint_chunks=self.training and self.checkpointing)
                return update(value,message)
            x = checkpoint(run, x, use_reentrant=False) if self.checkpointing and self.training else run(x)
        weights = self.pool_gate(x).softmax(1)
        return (weights*x).sum(1)

class RelationLayer(nn.Module):
    """Attention with explicit [is_support, true_relation] edge conditioning."""
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads; self.width = dim//heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.edge = nn.Linear(2, dim, bias=False)
        self.attn = nn.Sequential(nn.Linear(3*self.width, self.width), nn.LeakyReLU(0.2), nn.Linear(self.width, 1))
        self.update = Residual(dim, dropout)
    def messages(self, source, destination, src, dst, features):
        q = self.q(destination).reshape(-1, self.heads, self.width)
        k = self.k(source).reshape(-1, self.heads, self.width)
        v = self.v(source).reshape(-1, self.heads, self.width)
        e = self.edge(features.to(source.dtype)).reshape(-1, self.heads, self.width)
        logits = self.attn(torch.cat((q[dst], k[src], F.silu(e)), -1)).squeeze(-1)
        weights = softmax(logits.float(), dst, num_nodes=len(destination)).to(v.dtype)
        msg = (v[src]+e)*weights[..., None]
        out = msg.new_zeros(len(destination), self.heads, self.width)
        out.index_add_(0, dst, msg)
        return self.update(destination, out.flatten(1))
    def forward(self, nodes, edges, features):
        return self.messages(nodes, nodes, edges[0], edges[1], features)

class PromptGraphModel(nn.Module):
    def __init__(self, cfg, base, contract, *, local_encoder=None):
        super().__init__()
        m = base['model']; d = m['hidden_dim']
        self.dim = d; self.temperature = cfg['temperature']
        self.alignment_loss_weight = cfg['alignment_loss_weight']
        # Explicit injection supports L0 comparisons with exactly the same L1/L2.
        # The production constructor and its release gates are unchanged.
        if local_encoder is not None:
            self.local = local_encoder
        elif cfg.get('local_encoder') == 'v1_ct_only_context_pair':
            from .v1_local import V1LocalEncoder
            self.local = V1LocalEncoder(base)
        else:
            self.local = LocalEncoder(base, contract, cfg['graph_sampling'])
        # Shared class initializations; patient-specific values arise from support.
        # There is no learned patient ID table or query-GT-dependent initialization.
        self.label_seed = nn.Parameter(torch.randn(2, d)/math.sqrt(d))
        self.l1 = nn.ModuleList([RelationLayer(d, m['heads'], m['dropout']) for _ in range(cfg['task_layers'])])
        self.l2 = nn.ModuleList([nn.MultiheadAttention(d, m['heads'], dropout=m['dropout'], batch_first=True)
                                 for _ in range(cfg['alignment_layers'])])
        self.l2_updates = nn.ModuleList([Residual(d, m['dropout']) for _ in self.l2])

    def encode_support(self, embeddings, owners, classes):
        if embeddings.ndim != 2 or owners.shape != classes.shape or owners.shape != (len(embeddings),):
            raise ValueError('Malformed explicit support')
        if owners.dtype != torch.long or classes.dtype != torch.long or not len(owners):
            raise ValueError('Support owner/class int64 IDs required')
        p = int(owners.max())+1
        if p < 2 or bool((torch.bincount(owners, minlength=p) == 0).any()):
            raise ValueError('At least two nonempty other-patient support tasks required')
        if bool(((classes < 0) | (classes > 1)).any()) or len(classes.unique()) != 2:
            raise ValueError('Both observed classes must exist in training support')
        n = len(embeddings); device = embeddings.device
        data = torch.arange(n, device=device).repeat_interleave(2)
        cls = torch.arange(2, device=device).repeat(n)
        label = n+owners.repeat_interleave(2)*2+cls
        known = torch.stack((torch.ones_like(cls), (classes.repeat_interleave(2) == cls).long()), -1).float()
        edges = torch.stack((torch.cat((data, label)), torch.cat((label, data))))
        features = torch.cat((known, known))
        x = torch.cat((embeddings, self.label_seed.repeat(p, 1)))
        histories = []
        for layer in self.l1:
            # Cache layer inputs, not outputs: query read is simultaneous with support messages.
            histories.append(x[n:])
            x = checkpoint(layer, x, edges, features, use_reentrant=False) if self.training else layer(x, edges, features)
        local_labels = x[n:]
        return histories, local_labels.reshape(p, 2, self.dim)

    @torch.no_grad()
    def fit_support_clusters(self, embeddings, owners, classes):
        """Freeze deterministic L1 teacher/assignments once per support episode."""
        training = self.training
        self.eval()
        try:
            # A no-grad teacher inside the caller's autocast scope must not
            # populate its weight cache with detached casts used by the later
            # trainable/checkpointed L1 pass. Keep the same autocast precision.
            device_type=embeddings.device.type
            with torch.autocast(device_type,enabled=torch.is_autocast_enabled(device_type),
                                dtype=torch.get_autocast_dtype(device_type),cache_enabled=False):
                _, labels = self.encode_support(embeddings, owners, classes)
            plan = fit_prototypes(labels, owners, classes)
            plan['support_embeddings'] = embeddings.detach()
            return plan
        finally:
            self.train(training)

    def prepare_support(self, embeddings, owners, classes, cluster_plan=None):
        plan = self.fit_support_clusters(embeddings, owners, classes) if cluster_plan is None else cluster_plan
        if not torch.equal(plan['owners'], owners) or not torch.equal(plan['classes'], classes):
            raise ValueError('Cluster plan and explicit support ownership/classes differ')
        if not torch.equal(plan['support_embeddings'], embeddings):
            raise ValueError('Cluster plan was fitted on a different support memory')
        histories, local_labels = self.encode_support(embeddings, owners, classes)
        p = len(local_labels); device = embeddings.device
        task = torch.arange(p, device=device).repeat_interleave(2)
        forbidden = task[:, None] == task[None, :]
        aligned = local_labels.flatten(0,1)[None]
        for layer, update in zip(self.l2, self.l2_updates):
            def run(value, layer=layer, update=update):
                return update(value, layer(value, value, value, attn_mask=forbidden, need_weights=False)[0])
            aligned = checkpoint(run, aligned, use_reentrant=False) if self.training else run(aligned)
        labels = aligned[0].reshape(p, 2, self.dim)
        return dict(histories=histories, labels=labels, local_labels=local_labels,
                    cluster_plan=plan, alignment_loss=alignment_loss(labels,plan,self.temperature))

    def predict_embeddings(self, query, state):
        q = query
        for layer, labels in zip(self.l1, state['histories']):
            src = torch.arange(len(labels), device=q.device).repeat(len(q))
            dst = torch.arange(len(q), device=q.device).repeat_interleave(len(labels))
            # U: no known class relation. There is deliberately no query reverse edge.
            edge = q.new_zeros(len(src), 2)
            q = layer.messages(labels, q, src, dst, edge)
        logits = prototype_logits(q,state['labels'],state['cluster_plan'],self.temperature)
        return {'logits': logits, 'ranking_score': logits.softmax(-1)[:, 1],
                'alignment_loss': state['alignment_loss'], 'alignment_loss_weight': self.alignment_loss_weight}

    def forward(self, patches, support, owners, classes, cluster_plan=None):
        return self.predict_embeddings(self.local(patches), self.prepare_support(support, owners, classes, cluster_plan))

def supervised_loss(output, query_targets, class_weights=None):
    if query_targets.dtype != torch.long or query_targets.shape != (len(output['logits']),):
        raise ValueError('Explicit held-out query targets required only at loss time')
    return (F.cross_entropy(output['logits'], query_targets, weight=class_weights)
            +output['alignment_loss_weight']*output['alignment_loss'])
