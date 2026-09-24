"""U-shaped graph encoder/decoder: two learned pools, two unpools and skips.

This is a GAT + SAG U-Net variant, NOT the original Graph U-Nets GCN/gPool/A²
reproduction. Each level retains its exact induced spatial adjacency. Decoder
restores the saved parent topology and combines ALL parent nodes via skips.
Pool budgets are explicit architecture inputs; no hidden cap or fallback.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATv2Conv, SAGPooling, GCNConv
from .model import LocalEncoder, PromptGraphModel, Residual
from .sampling import padded_gat


def induced_level(parent, ids):
    """Vectorized induced subgraphs for B independent parent graphs.

    ids[B,K] maps each child row to its parent row. All parent edges between
    retained nodes survive, including one self loop per node. No neighbor cap.
    """
    b, n, degree = parent['neighbors'].shape
    k = ids.shape[1]
    mapping = torch.full((b, n), -1, dtype=torch.long, device=ids.device)
    mapping.scatter_(1, ids, torch.arange(k, device=ids.device).expand(b, -1))
    parent_neighbors = parent['neighbors'].gather(1, ids[:, :, None].expand(-1, -1, degree))
    parent_valid = parent['valid'].gather(1, ids[:, :, None].expand(-1, -1, degree))
    rows = torch.arange(b, device=ids.device)[:, None, None]
    mapped = mapping[rows, parent_neighbors]
    valid = parent_valid & (mapped >= 0)
    return dict(neighbors=mapped.clamp_min(0), valid=valid,
                alive=torch.ones((b, k), dtype=torch.bool, device=ids.device))


def unpool_nodes(child, ids, parent_nodes):
    """Restore selected rows at their saved parent indices; missing rows are zero.

    The skip supplies the missing fine-scale information. This operation alone
    does not reconstruct discarded features. Gradients propagate to child rows.
    """
    if ids.shape != child.shape[:2]:
        raise ValueError('Unpool requires exactly one parent index per child row')
    return child.new_zeros(child.shape[0], parent_nodes, child.shape[2]).scatter(
        1, ids[:, :, None].expand_as(child), child)


def level_edges(level):
    """Disjoint-union sparse edges, no per-graph Python loop, no self duplicates."""
    b, n, _ = level['neighbors'].shape
    destination = torch.arange(n, device=level['neighbors'].device)[None, :, None]
    mask = level['valid'] & (level['neighbors'] != destination)
    graph, dst, slot = mask.nonzero(as_tuple=True)
    src = level['neighbors'][graph, dst, slot]
    return torch.stack((src+graph*n, dst+graph*n))


class GraphUNetEncoder(LocalEncoder):
    def __init__(self, base, contract, spec):
        super().__init__(base, contract, sampling=None)
        self.spec = dict(spec)
        if spec['mode'] != 'graph_unet' or spec['pooling'] != 'sag_gcn':
            raise ValueError('Explicit graph_unet / sag_gcn architecture required')
        if spec['skip_merge'] != 'concat_linear' or spec['coarse_edges'] != 'induced_spatial':
            raise ValueError('Explicit concat skip and induced spatial edge policy required')
        self.pool_nodes = tuple(spec['pool_nodes'])
        if len(self.pool_nodes) != len(self.blocks)-1 or len(self.pool_nodes) < 2:
            raise ValueError('Hierarchical U-Net requires >=2 pool levels and matching encoder depth')
        n = len(self.grid)
        for k in self.pool_nodes:
            if type(k) is not int or not 1 < k < n:
                raise ValueError('Every explicit pool budget must be strictly smaller than its parent')
            n = k
        m = base['model']; d = m['hidden_dim']
        self.pools = nn.ModuleList([SAGPooling(d, ratio=k, GNN=GCNConv) for k in self.pool_nodes])
        self.decoder_blocks = nn.ModuleList([
            GATv2Conv(d, d//m['heads'], heads=m['heads'], dropout=m['dropout'], add_self_loops=True)
            for _ in self.pool_nodes])
        self.decoder_updates = nn.ModuleList([Residual(d, m['dropout']) for _ in self.pool_nodes])
        self.skip_merges = nn.ModuleList([
            nn.Sequential(nn.Linear(2*d, d), nn.LayerNorm(d), nn.SiLU()) for _ in self.pool_nodes])
        self._level0_cache = None
        self.capture = False
        self.last_capture = None
        self.last_audit = None

    def _apply(self, fn, recurse=True):
        self._level0_cache = None
        self._sample_table = None
        return super()._apply(fn, recurse=recurse)

    def initial_level(self, b, device):
        key = (b, str(device))
        if self._level0_cache is None or self._level0_cache[0] != key:
            n = len(self.grid)
            level = dict(neighbors=self.neighbors[None].expand(b, -1, -1),
                         valid=self.neighbor_valid[None].expand(b, -1, -1),
                         alive=torch.ones((b, n), dtype=torch.bool, device=device))
            offset = torch.arange(b, device=device)*n
            level['edges'] = (self.edge[:, None]+offset[None, :, None]).reshape(2, -1)
            self._level0_cache = key, level
        return self._level0_cache[1]

    def pool_level(self, x, level, pool):
        b, n, d = x.shape
        edges = level['edges'] if 'edges' in level else level_edges(level)
        owners = torch.arange(b, device=x.device).repeat_interleave(n)
        with torch.autocast(x.device.type, enabled=False):
            selected = pool.select(pool.gnn(x.float().reshape(-1, d), edges), owners)
        ids = selected.node_index.reshape(b, pool.ratio) % n
        score = selected.weight.reshape(b, pool.ratio)
        child = x.gather(1, ids[:, :, None].expand(-1, -1, d))*score[:, :, None].to(x.dtype)
        return child, induced_level(level, ids), ids, score

    def block(self, x, level, conv, update):
        b, n, d = x.shape
        free = torch.cuda.mem_get_info(x.device)[0] if x.is_cuda else 512*1024**2
        tile = max(1, min(n, int(free*.25/(b*level['neighbors'].shape[-1]*d*4*4))))
        def run(value):
            message = padded_gat(conv, value, level['neighbors'], level['valid'], level['alive'],
                                 tile, self.training and self.checkpointing)
            return update(value, message)
        return checkpoint(run, x, use_reentrant=False) if self.training and self.checkpointing else run(x)

    def forward(self, patches):
        if patches.ndim != 5 or patches.shape[1:] != (1, 48, 48, 48):
            raise ValueError('CT-only [B,1,48,48,48] required')
        with torch.profiler.record_function('UNet/CNN'):
            fmap = (checkpoint(self.dense, patches, use_reentrant=False)
                    if self.training and self.checkpoint_dense else self.dense(patches))
            x = self.project(self.sample_features(fmap).reshape(len(patches), len(self.grid), -1))
        level = self.initial_level(len(patches), patches.device)
        skips, topologies, mappings, counts, captures = [], [], [], [], []
        for stage, (conv, update) in enumerate(zip(self.blocks, self.updates)):
            with torch.profiler.record_function(f'UNet/encoder_{stage}'):
                x = self.block(x, level, conv, update)
            counts.append(x.shape[1])
            if stage < len(self.pools):
                skips.append(x)
                topologies.append(level)
                with torch.profiler.record_function(f'UNet/pool_{stage}'):
                    x, level, ids, score = self.pool_level(x, level, self.pools[stage])
                mappings.append(ids)
                if self.capture:
                    captures.append(dict(ids=ids.detach(), scores=score.detach(),
                        neighbors=level['neighbors'].detach(), valid=level['valid'].detach()))
        decoder_counts = []
        for stage, (conv, update, merge) in enumerate(zip(self.decoder_blocks, self.decoder_updates, self.skip_merges)):
            parent = len(skips)-1-stage
            with torch.profiler.record_function(f'UNet/unpool_skip_decoder_{parent}'):
                up = unpool_nodes(x, mappings[parent], skips[parent].shape[1])
                # Every fine node, including ones absent from the bottleneck,
                # receives its corresponding encoder feature via this skip.
                x = merge(torch.cat((skips[parent], up), dim=-1))
                x = self.block(x, topologies[parent], conv, update)
            decoder_counts.append(x.shape[1])
        if x.shape[1] != len(self.grid):
            raise RuntimeError('U-Net decoder did not restore the original node space')
        self.last_audit = dict(mode='graph_unet', encoder_nodes=counts, decoder_nodes=decoder_counts,
            pool_levels=len(self.pools), unpool_levels=len(self.pools), skip_connections=len(skips),
            candidate_nodes=len(self.grid), output_nodes=x.shape[1],
            encoder_gat_layers=len(self.blocks), decoder_gat_layers=len(self.decoder_blocks),
            cnn_feature_shape=list(fmap.shape[1:]), coarse_edges='induced_spatial')
        if self.capture:
            self.last_capture = captures
        with torch.profiler.record_function('UNet/decoded_graph_readout'):
            return (self.pool_gate(x).softmax(1)*x).sum(1)


def graph_unet_model(cfg, base, contract, spec):
    encoder = GraphUNetEncoder(base, contract, spec)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['seed'])
        return PromptGraphModel(cfg, base, contract, local_encoder=encoder)
