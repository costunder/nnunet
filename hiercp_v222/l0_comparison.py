"""Early selection versus late learned pooling; no A*, labels or GT inputs.

All graph arms share the complete physical candidate graph and CNN. Explicit
retained_nodes is an experimental variable, never a hidden cap or OOM fallback.
SAG uses PyG's GCN scorer and SelectTopK; exact induced edges are packed for the
existing batched GAT implementation instead of materializing dense adjacency.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import SAGPooling, GCNConv
from .model import LocalEncoder, PromptGraphModel, chunked_gat
from .sampling import personalized_rank, pack_selection, padded_gat

MODES = ('cnn_only', 'full_graph', 'early_ppr', 'early_sag', 'late_sag')


class ComparisonEncoder(LocalEncoder):
    def __init__(self, base, contract, spec):
        super().__init__(base, contract, sampling=None)
        self.spec = dict(spec)
        self.mode = spec['mode']
        if self.mode not in MODES:
            raise ValueError(f'Unknown L0 comparison mode: {self.mode}')
        self.keep = spec['retained_nodes']
        if self.mode in ('cnn_only', 'full_graph'):
            if self.keep != len(self.grid):
                raise ValueError('Full graph must retain every candidate')
        elif type(self.keep) is not int or not 1 < self.keep < len(self.grid):
            raise ValueError('Explicit retained_nodes must be between 2 and N-1')
        self.pool_after = spec['pool_after_layers']
        expected = 1 if self.mode == 'late_sag' else 0
        if self.pool_after != expected or len(self.blocks) <= self.pool_after:
            raise ValueError('Early selection is before layer 1; late pooling is after layer 1')
        if self.mode == 'cnn_only':
            # Explicit GNN ablation: remove parameters, do not leave dead layers.
            self.blocks = nn.ModuleList()
            self.updates = nn.ModuleList()
        no_self = self.neighbor_valid & (self.neighbors != torch.arange(len(self.grid))[:, None])
        self.register_buffer('selection_valid', no_self, persistent=False)
        positions = self.grid.flip(-1) * float(contract['outer_radius_mm'])
        self.register_buffer('distances', (positions[:, None]-positions[self.neighbors]).norm(dim=-1), persistent=False)
        self.seed_node = int(positions.norm(dim=-1).argmin())
        # Construct scorer AFTER shared L0 modules to preserve their seeded init.
        self.sag = (SAGPooling(base['model']['hidden_dim'], ratio=self.keep, GNN=GCNConv)
                    if self.mode in ('early_sag', 'late_sag') else None)
        if self.mode == 'early_ppr':
            if not 0 < spec['ppr_continuation'] < 1 or not 0 < spec['ppr_error'] < 1e-3:
                raise ValueError('Explicit PPR numerical contract required')
        self.last_audit = None
        self.capture = False
        self.last_capture = None
        self._sag_batch_cache = None

    def _apply(self, fn, recurse=True):
        # Device/dtype transitions invalidate execution-only caches.
        self._sag_batch_cache = None
        self._sample_table = None
        return super()._apply(fn, recurse=recurse)

    def _sag_inputs(self, batch_size, device):
        key = (batch_size, str(device))
        if self._sag_batch_cache is None or self._sag_batch_cache[0] != key:
            n = len(self.grid)
            offsets = torch.arange(batch_size, device=device) * n
            edges = (self.edge[:, None, :] + offsets[None, :, None]).reshape(2, -1)
            owners = torch.arange(batch_size, device=device).repeat_interleave(n)
            self._sag_batch_cache = (key, edges, owners)
        return self._sag_batch_cache[1:]

    @torch.no_grad()
    def _ppr(self, features):
        with torch.autocast(features.device.type, enabled=False):
            f = F.normalize(features.detach().float(), dim=-1)
            b, n, c = f.shape
            free = torch.cuda.mem_get_info(f.device)[0] if f.is_cuda else 512*1024**2
            tile = max(1, min(n, int(free*.1/(b*self.neighbors.shape[1]*c*4*3))))
            costs = []
            for lo in range(0, n, tile):
                hi = min(n, lo+tile)
                cosine = (f[:, lo:hi, None]*f[:, self.neighbors[lo:hi]]).sum(-1).clamp(-1, 1)
                cost = self.distances[None, lo:hi]*(2-cosine)
                costs.append(cost.masked_fill(~self.selection_valid[None, lo:hi], torch.inf))
            rank, degree, audit = personalized_rank(torch.cat(costs, 1), self.neighbors,
                self.selection_valid, self.seed_node, self.spec['ppr_continuation'], self.spec['ppr_error'])
            score = rank/degree
            ids = torch.argsort(score, descending=True, stable=True, dim=1)[:, :self.keep]
            return ids, score, audit

    def select(self, x, features):
        b, n, d = x.shape
        if self.mode == 'early_ppr':
            ids, score, convergence = self._ppr(features)
            flat_weights = None
        else:
            edges, owners = self._sag_inputs(b, x.device)
            # FP32 score computation avoids BF16 ties controlling discrete top-k.
            with torch.autocast(x.device.type, enabled=False):
                logits = self.sag.gnn(x.float().reshape(-1, d), edges)
                selection = self.sag.select(logits, owners)
            perm = selection.node_index
            ids = perm.reshape(b, self.keep) % n
            flat_weights = x.new_zeros(b*n, dtype=torch.float32).scatter(0, perm, selection.weight)
            score = flat_weights.reshape(b, n)
            convergence = None
        chosen = torch.zeros((b, n), dtype=torch.bool, device=x.device).scatter(1, ids, True)
        packed = pack_selection(chosen, self.neighbors, self.selection_valid)
        batch = torch.arange(b, device=x.device)[:, None]
        values = x[batch, packed['ids']]
        if flat_weights is not None:
            # Differentiable selected score gate is essential: hard indices alone
            # would not train the scorer. No query target participates in selection.
            values = values * score[batch, packed['ids'], None].to(values.dtype)
        if self.capture:
            self.last_capture = dict(ids=packed['ids'].detach(), scores=score.detach(),
                                     neighbors=packed['neighbors'].detach(), valid=packed['valid'].detach())
        return values, packed, convergence

    def forward(self, patches):
        if patches.ndim != 5 or patches.shape[1:] != (1, 48, 48, 48):
            raise ValueError('CT-only [B,1,48,48,48] required')
        with torch.profiler.record_function('L0/CNN'):
            fmap = (checkpoint(self.dense, patches, use_reentrant=False)
                    if self.checkpoint_dense and self.training else self.dense(patches))
        b, n = len(patches), len(self.grid)
        with torch.profiler.record_function('L0/candidate_features'):
            features = self.sample_features(fmap).reshape(b, n, -1)
            x = self.project(features)
        packed = None
        convergence = None
        stages = []
        for index, (block, update) in enumerate(zip(self.blocks, self.updates)):
            if self.mode not in ('cnn_only', 'full_graph') and index == self.pool_after:
                with torch.profiler.record_function('L0/selection_and_induced_edges'):
                    x, packed, convergence = self.select(x, features)
            width = x.shape[1]
            degree = self.neighbors.shape[1] if packed is None else packed['neighbors'].shape[-1]
            free = torch.cuda.mem_get_info(x.device)[0] if x.is_cuda else 512*1024**2
            chunk = max(1, min(width, int(free*.25/(b*degree*x.shape[-1]*4*4))))
            def run(value, block=block, update=update, packed=packed, chunk=chunk):
                if packed is None:
                    message = chunked_gat(block, value, self.neighbors, self.neighbor_valid, chunk,
                                          checkpoint_chunks=self.training and self.checkpointing)
                else:
                    message = padded_gat(block, value, packed['neighbors'], packed['valid'],
                                         packed['alive'], chunk, self.training and self.checkpointing)
                return update(value, message)
            with torch.profiler.record_function(f'L0/GAT_{index+1}'):
                x = checkpoint(run, x, use_reentrant=False) if self.checkpointing and self.training else run(x)
            stages.append(dict(nodes=width, edges=(self.edge.shape[1] if packed is None
                                                  else packed['edge_counts'].detach())))
        with torch.profiler.record_function('L0/readout'):
            answer = (self.pool_gate(x).softmax(1)*x).sum(1)
        self.last_audit = dict(mode=self.mode, candidate_nodes=n, candidate_edges=self.edge.shape[1],
            retained_nodes=x.shape[1], pool_after_layers=self.pool_after, layers=stages, ppr=convergence)
        return answer


def comparison_model(cfg, base, contract, spec):
    """Shared L1/L2 seed is independent of scorer parameter initialization."""
    if spec['mode'] == 'graph_unet':
        from .l0_graph_unet import graph_unet_model
        return graph_unet_model(cfg, base, contract, spec)
    encoder = ComparisonEncoder(base, contract, spec)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['seed'])
        return PromptGraphModel(cfg, base, contract, local_encoder=encoder)
