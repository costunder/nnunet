"""CT-only CNN node features and connectivity-only spatial GNN.

Structural node types and shell pooling retain the existing graph contract.
No handcrafted observation or geometric value enters learned node/edge inputs.
"""
import torch
from torch import nn
from hiercp.model import LocalTumorContextPyGEncoder, HeteroGATv2Block
from hiercp.schema import CONTEXT_SHELL_COUNT
from .schema import LOCAL_NODE_TYPES, LOCAL_EDGE_TYPES, SOURCE_LOCAL_NODE_TYPES
from torch.utils.checkpoint import checkpoint
from hiercp.model import PatchFeatureEncoder3D

class CTOnlyEncoder(LocalTumorContextPyGEncoder):
    def __init__(self, *, hidden_dim, heads, layers, dropout, checkpoint_local_blocks, dense_base_channels, dense_feature_dim, dense_batch_size, channels_last_3d, checkpoint_dense_encoder):
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.checkpoint_local_blocks = checkpoint_local_blocks
        self.dense_batch_size = dense_batch_size
        self.channels_last_3d = channels_last_3d
        self.checkpoint_dense_encoder = checkpoint_dense_encoder
        self.dense_encoder = PatchFeatureEncoder3D(in_channels=1, base_channels=dense_base_channels, out_channels=dense_feature_dim)
        if channels_last_3d:
            self.dense_encoder.to(memory_format=torch.channels_last_3d)
        self.project = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(dense_feature_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(inplace=True),
                )
                for node_type in LOCAL_NODE_TYPES
            }
        )
        self.blocks = nn.ModuleList(
            [
                HeteroGATv2Block(
                    node_types=LOCAL_NODE_TYPES,
                    edge_types=LOCAL_EDGE_TYPES,
                    dim=hidden_dim,
                    heads=heads,
                    edge_dim=None,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )
        self.pool = nn.ModuleDict(
            {
                node_type: self._attention_pool(hidden_dim)
                for node_type in LOCAL_NODE_TYPES
                if node_type not in {"source_context", "target_context"}
            }
        )
        self.context_shell_pool = nn.ModuleDict(
            {
                f"{node_type}_c{shell_id}": self._attention_pool(hidden_dim)
                for node_type in ("source_context", "target_context")
                for shell_id in range(CONTEXT_SHELL_COUNT)
            }
        )
        self.empty_context_shell = nn.ParameterDict(
            {
                f"{node_type}_c{shell_id}": nn.Parameter(torch.zeros(hidden_dim))
                for node_type in ("source_context", "target_context")
                for shell_id in range(CONTEXT_SHELL_COUNT)
            }
        )
        self.context_shell_fuse = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(hidden_dim * CONTEXT_SHELL_COUNT, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.SiLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for node_type in ("source_context", "target_context")
            }
        )
        self.tumor_fuse = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim*2),nn.LayerNorm(hidden_dim*2),
            nn.SiLU(inplace=True),nn.Dropout(dropout),nn.Linear(hidden_dim*2,hidden_dim))
        self.source_context_fuse = self._pair_fuser(hidden_dim, dropout)
        self.target_context_fuse = self._pair_fuser(hidden_dim, dropout)
        self.source_relation = self._pair_fuser(hidden_dim, dropout)
        self.target_relation = self._pair_fuser(hidden_dim, dropout)
        self.final_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def _pool_context_shells(self, node_type, x, shell_id, batch_index, graph_count):
        if shell_id.ndim != 1 or bool(((shell_id<0)|(shell_id>=CONTEXT_SHELL_COUNT)).any()):
            raise ValueError("Invalid structural shell membership")
        pooled=[]
        for index in range(CONTEXT_SHELL_COUNT):
            mask=shell_id==index
            counts=torch.bincount(batch_index[mask],minlength=graph_count)
            values=self.context_shell_pool[f"{node_type}_c{index}"](x[mask],index=batch_index[mask],dim_size=graph_count)
            empty=self.empty_context_shell[f"{node_type}_c{index}"].to(values.dtype)
            pooled.append(torch.where(counts[:,None]>0,values,empty[None].expand(graph_count,-1)))
        return self.context_shell_fuse[node_type](torch.cat(pooled,-1)),pooled

    def _run_local_block(self,block,x_dict,edge_index_dict,edge_attr_dict):
        if not (self.checkpoint_local_blocks and self.training and torch.is_grad_enabled()):
            return block(x_dict,edge_index_dict,edge_attr_dict)
        kinds=tuple(block.node_types)
        def run(*values):
            out=block(dict(zip(kinds,values)),edge_index_dict,edge_attr_dict)
            return tuple(out[k] for k in kinds)
        values=checkpoint(run,*(x_dict[k] for k in kinds),use_reentrant=False,preserve_rng_state=True)
        return dict(zip(kinds,values))

    def forward(self, local_batch):
        batch=local_batch.graph
        if set(batch.node_types)!=set(LOCAL_NODE_TYPES) or set(batch.edge_types)!=set(LOCAL_EDGE_TYPES):
            raise ValueError('Retired/unknown node or edge types; surface/context-only schema required')
        for patches in (local_batch.source_patches,local_batch.target_patches):
            if patches.ndim!=5 or patches.shape[1]!=1:
                raise ValueError("CNN-only L0 requires CT [B,1,D,H,W]; mask/statistic channels are forbidden")
        source_map,target_map=self.encode_dense_maps(local_batch.source_patches,local_batch.source_index,local_batch.target_patches)
        x_dict={}
        for kind in LOCAL_NODE_TYPES:
            forbidden={'x','observed','pos','pos_mm'}.intersection(batch[kind].keys())
            if forbidden: raise ValueError(f"Only CNN features enter L0; unexpected node inputs {forbidden}")
            feature_map=source_map if kind in SOURCE_LOCAL_NODE_TYPES else target_map
            sampled=self._sample_dense_features(feature_map,batch[kind].grid,batch[kind].batch)
            x_dict[kind]=self.project[kind](sampled)
        if any('edge_attr' in batch[edge] for edge in LOCAL_EDGE_TYPES):
            raise ValueError("CNN-only L0 uses connectivity only; edge attributes are forbidden")
        edge_attr_dict={edge:None for edge in LOCAL_EDGE_TYPES}
        for block in self.blocks:
            x_dict = self._run_local_block(
                block, x_dict, batch.edge_index_dict, edge_attr_dict
            )
        pooled = {
            node_type: self.pool[node_type](
                x_dict[node_type],
                index=batch[node_type].batch,
            )
            for node_type in LOCAL_NODE_TYPES
            if node_type not in {"source_context", "target_context"}
        }
        source_context_core, source_shells = self._pool_context_shells(
            "source_context",
            x_dict["source_context"],
            batch["source_context"].shell_id,
            batch["source_context"].batch,
            batch.num_graphs,
        )
        target_context_core, target_shells = self._pool_context_shells(
            "target_context",
            x_dict["target_context"],
            batch["target_context"].shell_id,
            batch["target_context"].batch,
            batch.num_graphs,
        )
        tumor = self.tumor_fuse(pooled["tumor_surface"])
        source_context = self.source_context_fuse(
            self._pair(source_context_core, pooled["source_liver_surface"])
        )
        target_context = self.target_context_fuse(
            self._pair(target_context_core, pooled["target_liver_surface"])
        )
        source_relation = self.source_relation(self._pair(tumor, source_shells[0]))
        target_relation = self.target_relation(self._pair(tumor, target_shells[0]))
        fused = self.final_fuse(
            torch.cat(
                [
                    tumor,
                    source_context,
                    target_context,
                    source_relation,
                    target_relation,
                    torch.abs(source_relation - target_relation),
                ],
                dim=-1,
            )
        )
        return {
            "tumor": tumor,
            "source_context": source_context,
            "target_context": target_context,
            "source_relation": source_relation,
            "target_relation": target_relation,
            "source_c0": source_shells[0],
            "source_c1": source_shells[1],
            "source_c2": source_shells[2],
            "target_c0": target_shells[0],
            "target_c1": target_shells[1],
            "target_c2": target_shells[2],
            "fused": fused,
        }
