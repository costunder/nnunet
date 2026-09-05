"""Three-level PyTorch Geometric placement model.

Regular-grid CT encoding uses a compact 3D CNN. Every graph message-passing
operation uses PyTorch Geometric ``HeteroConv`` with relation-specific
``GATv2Conv`` layers; there is no hand-written scatter/index-add GNN path.

The high-throughput path consumes a :class:`hiercp.data.HierarchicalBatch`
that is assembled in DataLoader workers. This avoids rebuilding and copying
three levels of PyG graphs on the GPU for every optimizer step.
"""

from __future__ import annotations


from hiercp.sample import sample_dense_features_variable
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from torch_geometric.data import Batch
    from torch_geometric.nn import AttentionalAggregation, GATv2Conv, HeteroConv
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch Geometric is required. Run: python -m tools.install"
    ) from exc

from hiercp.schema import (
    CONTEXT_SHELL_COUNT,
    CONTEXT_SHELL_FEATURE_INDEX,
    LOCAL_EDGE_DIM,
    LOCAL_EDGE_TYPES,
    LOCAL_HANDCRAFTED_DIM,
    LOCAL_NODE_TYPES,
    PATIENT_EDGE_DIM,
    PATIENT_EDGE_TYPES,
    PATIENT_NODE_TYPES,
    PROTOTYPE_EDGE_DIM,
    PROTOTYPE_EDGE_TYPES,
    PROTOTYPE_FEATURE_DIM,
    PROTOTYPE_NODE_TYPES,
    REGION_FEATURE_DIM,
    SOURCE_LOCAL_NODE_TYPES,
    UPPER_FORBIDDEN_RAW_COLUMNS,
    PATIENT_POSITION_EDGE_COLUMNS,
    UPPER_RAW_DIM,
)


@dataclass
class HierarchicalOutput:
    scores: list[Tensor]
    local_batch: Batch
    local_batch_view2: Batch | None
    patient_batch: Batch
    prototype_batch: Batch
    local_embeddings: dict[str, Tensor]
    local_embeddings_view2: dict[str, Tensor] | None
    consistency: Tensor


ABLATION_MODES: tuple[str, ...] = (
    "full",
    "no_local",
    "no_patient",
    "no_population",
)

_LOCAL_EMBEDDING_KEYS: tuple[str, ...] = (
    "tumor",
    "source_context",
    "target_context",
    "source_relation",
    "target_relation",
    "source_c0",
    "source_c1",
    "source_c2",
    "target_c0",
    "target_c1",
    "target_c2",
    "fused",
)


def _normalize_ablation_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in ABLATION_MODES:
        raise ValueError(
            f"Unknown ablation_mode={value!r}; expected one of {ABLATION_MODES}"
        )
    return mode


def _mask_upper_shortcuts(raw: Tensor) -> Tensor:
    """Remove label-only upper features while preserving the cached schema."""

    if raw.ndim != 2 or int(raw.shape[1]) != UPPER_RAW_DIM:
        raise ValueError(
            f"Expected upper raw features [N,{UPPER_RAW_DIM}], got {tuple(raw.shape)}"
        )
    mask = raw.new_ones((UPPER_RAW_DIM,))
    mask[list(UPPER_FORBIDDEN_RAW_COLUMNS)] = 0.0
    return raw * mask


def _mask_tumor_spatial_edge_attr(
    edge_type: tuple[str, str, str], edge_attr: Tensor
) -> Tensor:
    """Hide source-tumor coordinates from every patient-level relation."""

    if "tumor" not in (edge_type[0], edge_type[2]):
        return edge_attr
    if edge_attr.ndim != 2 or int(edge_attr.shape[1]) != PATIENT_EDGE_DIM:
        raise ValueError(
            f"Expected patient edge attributes [E,{PATIENT_EDGE_DIM}], "
            f"got {tuple(edge_attr.shape)} for {edge_type}"
        )
    mask = edge_attr.new_ones((PATIENT_EDGE_DIM,))
    mask[list(PATIENT_POSITION_EDGE_COLUMNS)] = 0.0
    return edge_attr * mask


def _group_count(channels: int) -> int:
    groups = min(8, int(channels))
    while channels % groups != 0:
        groups -= 1
    return groups


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class PatchFeatureEncoder3D(nn.Module):
    """Dense 3D CT/SDF encoder whose feature map is sampled at graph nodes."""

    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 12,
        out_channels: int = 32,
    ) -> None:
        super().__init__()
        self.stem = ConvBlock3D(in_channels, base_channels)
        self.down1 = nn.Sequential(
            nn.Conv3d(
                base_channels,
                base_channels * 2,
                3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(base_channels * 2), base_channels * 2),
            nn.SiLU(inplace=True),
            ConvBlock3D(base_channels * 2, base_channels * 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(
                base_channels * 2,
                out_channels,
                3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            ConvBlock3D(out_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down2(self.down1(self.stem(x)))


class HeteroGATv2Block(nn.Module):
    """Residual heterogeneous message passing implemented entirely by PyG."""

    def __init__(
        self,
        *,
        node_types: Sequence[str],
        edge_types: Sequence[tuple[str, str, str]],
        dim: int,
        heads: int,
        edge_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.node_types = tuple(node_types)
        self.edge_types = tuple(edge_types)
        self.conv = HeteroConv(
            {
                edge_type: GATv2Conv(
                    (dim, dim),
                    dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                    edge_dim=edge_dim,
                    share_weights=False,
                )
                for edge_type in self.edge_types
            },
            aggr="sum",
        )
        self.message_norm = nn.ModuleDict(
            {node_type: nn.LayerNorm(dim) for node_type in self.node_types}
        )
        self.ffn_norm = nn.ModuleDict(
            {node_type: nn.LayerNorm(dim) for node_type in self.node_types}
        )
        self.ffn = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.SiLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                )
                for node_type in self.node_types
            }
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
        edge_attr_dict: dict[tuple[str, str, str], Tensor],
    ) -> dict[str, Tensor]:
        messages = self.conv(
            x_dict,
            edge_index_dict,
            edge_attr_dict=edge_attr_dict,
        )
        output: dict[str, Tensor] = {}
        for node_type in self.node_types:
            current = x_dict[node_type]
            message = messages.get(node_type)
            if message is None:
                message = torch.zeros_like(current)
            current = self.message_norm[node_type](current + self.dropout(message))
            current = self.ffn_norm[node_type](
                current + self.dropout(self.ffn[node_type](current))
            )
            output[node_type] = current
        return output


class LocalTumorContextPyGEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        layers: int,
        dropout: float,
        dense_base_channels: int,
        dense_feature_dim: int,
        dense_batch_size: int,
        channels_last_3d: bool,
        checkpoint_local_blocks: bool,
        checkpoint_dense_encoder: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dense_batch_size = max(1, int(dense_batch_size))
        self.channels_last_3d = bool(channels_last_3d)
        self.checkpoint_local_blocks = bool(checkpoint_local_blocks)
        self.checkpoint_dense_encoder = bool(checkpoint_dense_encoder)
        self.dense_encoder = PatchFeatureEncoder3D(
            in_channels=5,
            base_channels=dense_base_channels,
            out_channels=dense_feature_dim,
        )
        if self.channels_last_3d:
            self.dense_encoder.to(memory_format=torch.channels_last_3d)
        self.project = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(LOCAL_HANDCRAFTED_DIM + dense_feature_dim, hidden_dim),
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
                    edge_dim=LOCAL_EDGE_DIM,
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
        self.tumor_fuse = self._pair_fuser(hidden_dim, dropout)
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

    @staticmethod
    def _attention_pool(dim: int) -> AttentionalAggregation:
        return AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(dim, max(1, dim // 2)),
                nn.Tanh(),
                nn.Linear(max(1, dim // 2), 1),
            )
        )

    @staticmethod
    def _pair_fuser(dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    @staticmethod
    def _pair(first: Tensor, second: Tensor) -> Tensor:
        return torch.cat(
            [first, second, torch.abs(first - second), first * second],
            dim=-1,
        )

    @staticmethod
    def _sample_dense_features(
        feature_map: Tensor,
        grid: Tensor,
        node_batch: Tensor,
    ) -> Tensor:
        return sample_dense_features_variable(feature_map, grid, node_batch)

    def _encode_dense(self, patches: Tensor) -> Tensor:
        if patches.ndim != 5:
            raise ValueError(
                "Dense patches must have shape [B,C,D,H,W], "
                f"got {tuple(patches.shape)}"
            )
        # Cached patches stay float16 in host memory. CPU execution and CUDA
        # execution outside autocast require a float32 convolution input.
        if patches.device.type == "cpu" or not torch.is_autocast_enabled():
            patches = patches.float()
        if self.channels_last_3d:
            patches = patches.contiguous(memory_format=torch.channels_last_3d)
        def encode(chunk: Tensor) -> Tensor:
            if (
                self.checkpoint_dense_encoder
                and self.training
                and torch.is_grad_enabled()
            ):
                return checkpoint(
                    self.dense_encoder,
                    chunk,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            return self.dense_encoder(chunk)

        count = int(patches.shape[0])
        if count <= self.dense_batch_size:
            return encode(patches)
        chunks = [
            encode(patches[start : start + self.dense_batch_size])
            for start in range(0, count, self.dense_batch_size)
        ]
        return torch.cat(chunks, dim=0)

    def _pool_context_shells(
        self,
        node_type: str,
        x: Tensor,
        raw_x: Tensor,
        batch_index: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        """Pool each physical shell without requiring fixed shell counts.

        A sampled view may legitimately omit a shell for one graph.  Missing
        shells use a learned token rather than making graph construction or
        batching depend on a fixed node allocation.
        """
        shell_value = raw_x[:, CONTEXT_SHELL_FEATURE_INDEX]
        shell_id = torch.clamp(
            torch.round(shell_value * CONTEXT_SHELL_COUNT).long(),
            0,
            CONTEXT_SHELL_COUNT - 1,
        )
        if batch_index.numel() == 0:
            raise RuntimeError(f"{node_type} contains no nodes")
        graph_count = int(batch_index.max().item()) + 1
        shell_pooled: list[Tensor] = []
        for index in range(CONTEXT_SHELL_COUNT):
            mask = shell_id == index
            counts = torch.bincount(batch_index[mask], minlength=graph_count)
            if bool(mask.any()):
                pooled = self.context_shell_pool[f"{node_type}_c{index}"](
                    x[mask],
                    index=batch_index[mask],
                    dim_size=graph_count,
                )
            else:
                pooled = x.new_zeros((graph_count, self.hidden_dim))
            empty = self.empty_context_shell[f"{node_type}_c{index}"].to(
                dtype=pooled.dtype
            )
            pooled = torch.where(
                counts[:, None] > 0,
                pooled,
                empty[None].expand(graph_count, -1),
            )
            shell_pooled.append(pooled)
        fused = self.context_shell_fuse[node_type](torch.cat(shell_pooled, dim=-1))
        return fused, shell_pooled

    def encode_dense_maps(
        self,
        source_patches: Tensor,
        source_graph_index: Tensor,
        target_patches: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode dense patches once and reuse them for both sampled views."""
        if source_patches.ndim != 5 or target_patches.ndim != 5:
            raise ValueError("source and target dense patches must be five-dimensional")
        if int(source_graph_index.numel()) != int(target_patches.shape[0]):
            raise ValueError("source_graph_index must contain one entry per local graph")
        source_unique = self._encode_dense(source_patches)
        return source_unique[source_graph_index], self._encode_dense(target_patches)

    def _run_local_block(
        self,
        block: HeteroGATv2Block,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
        edge_attr_dict: dict[tuple[str, str, str], Tensor],
    ) -> dict[str, Tensor]:
        if not (
            self.checkpoint_local_blocks
            and self.training
            and torch.is_grad_enabled()
        ):
            return block(x_dict, edge_index_dict, edge_attr_dict)

        node_types = tuple(LOCAL_NODE_TYPES)
        inputs = tuple(x_dict[node_type] for node_type in node_types)

        def run(*node_values: Tensor) -> tuple[Tensor, ...]:
            checkpoint_x = {
                node_type: value
                for node_type, value in zip(node_types, node_values)
            }
            output = block(checkpoint_x, edge_index_dict, edge_attr_dict)
            return tuple(output[node_type] for node_type in node_types)

        values = checkpoint(
            run,
            *inputs,
            use_reentrant=False,
            preserve_rng_state=True,
        )
        return {
            node_type: value
            for node_type, value in zip(node_types, values)
        }

    def forward_graph(
        self,
        batch: Batch,
        source_map: Tensor,
        target_map: Tensor,
    ) -> dict[str, Tensor]:
        """Run one sampled radius-graph view with precomputed dense maps."""
        x_dict: dict[str, Tensor] = {}
        for node_type in LOCAL_NODE_TYPES:
            feature_map = source_map if node_type in SOURCE_LOCAL_NODE_TYPES else target_map
            sampled = self._sample_dense_features(
                feature_map,
                batch[node_type].grid,
                batch[node_type].batch,
            )
            handcrafted = batch[node_type].x
            if handcrafted.dtype != sampled.dtype:
                handcrafted = handcrafted.to(sampled.dtype)
            x_dict[node_type] = self.project[node_type](
                torch.cat([handcrafted, sampled], dim=-1)
            )
        edge_attr_dict = {
            edge_type: batch[edge_type].edge_attr for edge_type in LOCAL_EDGE_TYPES
        }
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
            batch["source_context"].x,
            batch["source_context"].batch,
        )
        target_context_core, target_shells = self._pool_context_shells(
            "target_context",
            x_dict["target_context"],
            batch["target_context"].x,
            batch["target_context"].batch,
        )
        tumor = self.tumor_fuse(
            self._pair(pooled["tumor_surface"], pooled["tumor_interior"])
        )
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

    def forward(
        self,
        batch: Batch,
        source_patches: Tensor,
        source_graph_index: Tensor,
        target_patches: Tensor,
    ) -> dict[str, Tensor]:
        source_map, target_map = self.encode_dense_maps(
            source_patches, source_graph_index, target_patches
        )
        return self.forward_graph(batch, source_map, target_map)


class PatientRegionPyGEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.tumor_project = nn.Sequential(
            nn.Linear(hidden_dim * 3 + UPPER_RAW_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.candidate_project = nn.Sequential(
            nn.Linear(hidden_dim * 3 + UPPER_RAW_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.region_project = nn.Sequential(
            nn.Linear(REGION_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.lesion_project = nn.Sequential(
            nn.Linear(UPPER_RAW_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.liver_project = nn.Sequential(
            nn.Linear(UPPER_RAW_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                HeteroGATv2Block(
                    node_types=PATIENT_NODE_TYPES,
                    edge_types=PATIENT_EDGE_TYPES,
                    dim=hidden_dim,
                    heads=heads,
                    edge_dim=PATIENT_EDGE_DIM,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

    def forward_raw(
        self,
        raw_batch: Batch,
        local: dict[str, Tensor],
        counts: tuple[int, ...],
    ) -> dict[str, Tensor]:
        starts: list[int] = []
        offset = 0
        for count in counts:
            starts.append(offset)
            offset += int(count)
        if offset != int(local["fused"].shape[0]):
            raise ValueError("Local embedding count does not match patient candidates")
        start_index = torch.tensor(
            starts,
            dtype=torch.long,
            device=local["fused"].device,
        )
        tumor_raw = _mask_upper_shortcuts(raw_batch["tumor"].raw_x)
        candidate_raw = _mask_upper_shortcuts(raw_batch["candidate"].raw_x)
        x_dict: dict[str, Tensor] = {
            "tumor": self.tumor_project(
                torch.cat(
                    [
                        local["tumor"][start_index],
                        local["source_context"][start_index],
                        local["source_relation"][start_index],
                        tumor_raw,
                    ],
                    dim=-1,
                )
            ),
            "candidate": self.candidate_project(
                torch.cat(
                    [
                        local["fused"],
                        local["target_context"],
                        local["target_relation"],
                        candidate_raw,
                    ],
                    dim=-1,
                )
            ),
            "region": self.region_project(raw_batch["region"].raw_x),
            "lesion": self.lesion_project(raw_batch["lesion"].raw_x),
            "liver": self.liver_project(raw_batch["liver"].raw_x),
        }
        edge_attr_dict = {
            edge_type: _mask_tumor_spatial_edge_attr(
                edge_type, raw_batch[edge_type].edge_attr
            )
            for edge_type in PATIENT_EDGE_TYPES
        }
        for block in self.blocks:
            x_dict = block(x_dict, raw_batch.edge_index_dict, edge_attr_dict)
        return x_dict


class PrototypePyGEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.candidate_bridge = nn.Sequential(
            nn.Linear(hidden_dim + UPPER_RAW_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.region_bridge = nn.Sequential(
            nn.Linear(hidden_dim + REGION_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.prototype_project = nn.Sequential(
            nn.Linear(PROTOTYPE_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                HeteroGATv2Block(
                    node_types=PROTOTYPE_NODE_TYPES,
                    edge_types=PROTOTYPE_EDGE_TYPES,
                    dim=hidden_dim,
                    heads=heads,
                    edge_dim=PROTOTYPE_EDGE_DIM,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

    def forward_raw(
        self,
        raw_batch: Batch,
        patient_x: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        x_dict: dict[str, Tensor] = {
            "candidate": self.candidate_bridge(
                torch.cat(
                    [
                        patient_x["candidate"],
                        _mask_upper_shortcuts(raw_batch["candidate"].raw_x),
                    ],
                    dim=-1,
                )
            ),
            "region": self.region_bridge(
                torch.cat([patient_x["region"], raw_batch["region"].raw_x], dim=-1)
            ),
            "prototype": self.prototype_project(raw_batch["prototype"].raw_x),
        }
        edge_attr_dict = {
            edge_type: raw_batch[edge_type].edge_attr
            for edge_type in PROTOTYPE_EDGE_TYPES
        }
        for block in self.blocks:
            x_dict = block(x_dict, raw_batch.edge_index_dict, edge_attr_dict)
        return x_dict


class HierarchicalPyGPlacementModel(nn.Module):
    """Dense CT/SDF encoder + local PyG + patient PyG + prototype PyG."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        heads: int = 4,
        local_layers: int = 3,
        patient_layers: int = 3,
        prototype_layers: int = 2,
        dropout: float = 0.1,
        dense_base_channels: int = 12,
        dense_feature_dim: int = 32,
        dense_batch_size: int = 8,
        channels_last_3d: bool = True,
        checkpoint_local_blocks: bool = True,
        checkpoint_dense_encoder: bool = True,
        ablation_mode: str = "full",
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = int(hidden_dim)
        self.ablation_mode = _normalize_ablation_mode(ablation_mode)
        self.local_encoder = LocalTumorContextPyGEncoder(
            hidden_dim=hidden_dim,
            heads=heads,
            layers=local_layers,
            dropout=dropout,
            dense_base_channels=dense_base_channels,
            dense_feature_dim=dense_feature_dim,
            dense_batch_size=dense_batch_size,
            channels_last_3d=channels_last_3d,
            checkpoint_local_blocks=checkpoint_local_blocks,
            checkpoint_dense_encoder=checkpoint_dense_encoder,
        )
        self.patient_encoder = PatientRegionPyGEncoder(
            hidden_dim=hidden_dim,
            heads=heads,
            layers=patient_layers,
            dropout=dropout,
        )
        self.prototype_encoder = PrototypePyGEncoder(
            hidden_dim=hidden_dim,
            heads=heads,
            layers=prototype_layers,
            dropout=dropout,
        )
        active_hidden_blocks = {
            "full": 9,
            "no_local": 4,
            "no_patient": 7,
            "no_population": 7,
        }[self.ablation_mode]
        self.score_input_dim = hidden_dim * active_hidden_blocks + UPPER_RAW_DIM
        self.score_head = nn.Sequential(
            nn.Linear(self.score_input_dim, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )
        disabled_encoder = {
            "no_local": self.local_encoder,
            "no_patient": self.patient_encoder,
            "no_population": self.prototype_encoder,
        }.get(self.ablation_mode)
        if disabled_encoder is not None:
            disabled_encoder.requires_grad_(False)

    def trainable_parameters(self):
        """Iterate only parameters that can receive optimizer updates."""

        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    @property
    def local_edge_types(self) -> tuple[tuple[str, str, str], ...]:
        return LOCAL_EDGE_TYPES

    @property
    def patient_edge_types(self) -> tuple[tuple[str, str, str], ...]:
        return PATIENT_EDGE_TYPES

    @property
    def prototype_edge_types(self) -> tuple[tuple[str, str, str], ...]:
        return PROTOTYPE_EDGE_TYPES

    @staticmethod
    def _coerce_batch(payload: Any) -> Any:
        # Backward-compatible convenience for smoke tests and direct inference.
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            from hiercp.data import collate_samples

            return collate_samples(list(payload))
        required = (
            "source_patches",
            "target_patches",
            "local_batch",
            "patient_batch",
            "prototype_batch",
            "counts",
        )
        if not all(hasattr(payload, name) for name in required):
            raise TypeError("Expected HierarchicalBatch or a sequence of cached samples")
        return payload

    @staticmethod
    def _mean_embeddings(
        first: dict[str, Tensor], second: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        if first.keys() != second.keys():
            raise RuntimeError("Sampled local views produced different embedding schemas")
        return {key: 0.5 * (first[key] + second[key]) for key in first}

    @staticmethod
    def _view_consistency(
        first: dict[str, Tensor], second: dict[str, Tensor]
    ) -> Tensor:
        keys = (
            "tumor",
            "source_context",
            "target_context",
            "source_relation",
            "target_relation",
            "fused",
        )
        terms = [
            1.0 - F.cosine_similarity(first[key].float(), second[key].float(), dim=-1).mean()
            for key in keys
        ]
        return torch.stack(terms).mean()

    def _score_full(
        self, batch: Any, local_embeddings: dict[str, Tensor]
    ) -> list[Tensor]:
        """Run the patient/prototype hierarchy from complete local embeddings."""

        device = next(self.parameters()).device
        patient_x = self.patient_encoder.forward_raw(
            batch.patient_batch,
            local_embeddings,
            batch.counts,
        )
        prototype_x = self.prototype_encoder.forward_raw(
            batch.prototype_batch,
            patient_x,
        )

        candidate_patient = patient_x["candidate"]
        candidate_sample = batch.patient_batch["candidate"].batch
        tumor_for_candidate = patient_x["tumor"][candidate_sample]
        candidate_prototype = prototype_x["candidate"]

        membership = batch.prototype_batch[
            ("candidate", "belongs_to", "region")
        ].edge_index
        candidate_region = torch.empty(
            int(candidate_patient.shape[0]),
            dtype=torch.long,
            device=device,
        )
        candidate_region[membership[0]] = membership[1]
        region_for_candidate = prototype_x["region"][candidate_region]
        raw_candidate = _mask_upper_shortcuts(
            batch.patient_batch["candidate"].raw_x
        )

        pair = torch.cat(
            [
                candidate_patient,
                tumor_for_candidate,
                candidate_prototype,
                region_for_candidate,
                local_embeddings["fused"],
                local_embeddings["source_relation"],
                local_embeddings["target_relation"],
                torch.abs(
                    local_embeddings["source_relation"]
                    - local_embeddings["target_relation"]
                ),
                local_embeddings["source_relation"]
                * local_embeddings["target_relation"],
                raw_candidate,
            ],
            dim=-1,
        )
        flat_scores = self.score_head(pair).squeeze(-1)
        return list(torch.split(flat_scores, batch.counts, dim=0))

    def _zero_local_embeddings(self, raw_candidate: Tensor) -> dict[str, Tensor]:
        zero = raw_candidate.new_zeros(
            (int(raw_candidate.shape[0]), self.hidden_dim)
        )
        return {key: zero for key in _LOCAL_EMBEDDING_KEYS}

    def _zero_patient_embeddings(self, batch: Any) -> dict[str, Tensor]:
        """Return exact-zero patient embeddings for an independent L1 ablation."""

        output: dict[str, Tensor] = {}
        reference = batch.patient_batch["candidate"].raw_x
        for node_type in PATIENT_NODE_TYPES:
            count = int(batch.patient_batch[node_type].raw_x.shape[0])
            output[node_type] = reference.new_zeros((count, self.hidden_dim))
        return output

    def _prototype_for_candidates(
        self, batch: Any, prototype_x: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        candidate_prototype = prototype_x["candidate"]
        membership = batch.prototype_batch[
            ("candidate", "belongs_to", "region")
        ].edge_index
        candidate_region = torch.empty(
            int(candidate_prototype.shape[0]),
            dtype=torch.long,
            device=candidate_prototype.device,
        )
        candidate_region[membership[0]] = membership[1]
        return candidate_prototype, prototype_x["region"][candidate_region]

    def _score_ablation(
        self,
        batch: Any,
        local_embeddings: dict[str, Tensor] | None,
    ) -> list[Tensor]:
        """Run a one-factor, leave-one-level-out ablation of full M3.

        Each mode removes exactly one hierarchy level while retaining the other
        two levels. The score head receives only active feature blocks, so it
        has no permanently zero trainable input columns:

        - ``no_local``: Level 0 slots and Level-0 inputs to Level 1 are zeros;
          Levels 1 and 2 remain active from their raw graph features.
        - ``no_patient``: Level 1 slots and Level-1 inputs to Level 2 are zeros;
          Levels 0 and 2 remain active.
        - ``no_population``: Level 2 slots are zeros; Levels 0 and 1 remain active.

        Disabled encoders are not executed or trained. Exact zero tensors are
        used only at an active downstream encoder's removed-level interface;
        those tensors are never concatenated into the ablation score input.
        """

        mode = self.ablation_mode
        if mode == "full":
            raise RuntimeError("_score_ablation was called for full mode")

        raw_candidate = _mask_upper_shortcuts(
            batch.patient_batch["candidate"].raw_x
        )
        if mode == "no_local":
            local_embeddings = self._zero_local_embeddings(raw_candidate)
        elif local_embeddings is None:
            raise RuntimeError(f"{mode} requires Level-0 embeddings")

        if mode == "no_patient":
            patient_x = self._zero_patient_embeddings(batch)
        else:
            patient_x = self.patient_encoder.forward_raw(
                batch.patient_batch,
                local_embeddings,
                batch.counts,
            )
            candidate_patient = patient_x["candidate"]
            candidate_sample = batch.patient_batch["candidate"].batch
            tumor_for_candidate = patient_x["tumor"][candidate_sample]

        if mode != "no_population":
            prototype_x = self.prototype_encoder.forward_raw(
                batch.prototype_batch,
                patient_x,
            )
            candidate_prototype, region_for_candidate = (
                self._prototype_for_candidates(batch, prototype_x)
            )

        score_parts: list[Tensor] = []
        if mode != "no_patient":
            score_parts.extend([candidate_patient, tumor_for_candidate])
        if mode != "no_population":
            score_parts.extend([candidate_prototype, region_for_candidate])
        if mode != "no_local":
            if local_embeddings is None:  # defensive narrowing for type checkers
                raise RuntimeError(f"{mode} requires Level-0 embeddings")
            source_relation = local_embeddings["source_relation"]
            target_relation = local_embeddings["target_relation"]
            score_parts.extend(
                [
                    local_embeddings["fused"],
                    source_relation,
                    target_relation,
                    torch.abs(source_relation - target_relation),
                    source_relation * target_relation,
                ]
            )
        score_parts.append(raw_candidate)
        pair = torch.cat(score_parts, dim=-1)
        if int(pair.shape[1]) != self.score_input_dim:
            raise RuntimeError(
                f"Ablation score input mismatch for {mode}: "
                f"expected={self.score_input_dim}, actual={int(pair.shape[1])}"
            )
        flat_scores = self.score_head(pair).squeeze(-1)
        return list(torch.split(flat_scores, batch.counts, dim=0))

    def _score_upper(
        self,
        batch: Any,
        local_embeddings: dict[str, Tensor] | None,
    ) -> list[Tensor]:
        if self.ablation_mode == "full":
            if local_embeddings is None:
                raise RuntimeError("full mode requires Level-0 embeddings")
            return self._score_full(batch, local_embeddings)
        return self._score_ablation(batch, local_embeddings)

    def score_inference_chunked(
        self, payload: Any, *, local_chunk_size: int = 8
    ) -> list[Tensor]:
        """Score ordered inference hierarchies without moving all local graphs to CUDA.

        Candidate-local dense maps and both sampled GAT views are processed in
        bounded chunks.  The complete patient/prototype graphs are evaluated
        only after the local embeddings have been reassembled, so candidate
        interactions and ranking semantics remain identical to ``forward``.
        """

        if self.training or torch.is_grad_enabled():
            raise RuntimeError(
                "score_inference_chunked requires model.eval() and inference/no-grad mode"
            )
        batch = self._coerce_batch(payload)
        counts = tuple(int(value) for value in batch.counts)
        case_count = len(counts)
        candidate_count = sum(counts)
        if case_count < 1 or batch.sample_count != case_count:
            raise ValueError("Chunked inference requires a non-empty aligned case batch")
        if any(count < 1 for count in counts):
            raise ValueError("Chunked inference received a case with no candidates")
        if int(batch.source_patches.shape[0]) != case_count:
            raise ValueError("Source patch count does not match inference case count")
        if int(batch.target_patches.shape[0]) != candidate_count:
            raise ValueError("Target patch count does not match inference candidates")

        device = next(self.parameters()).device
        non_blocking = device.type == "cuda"

        def validate_scores(scores: list[Tensor]) -> list[Tensor]:
            if len(scores) != case_count:
                raise RuntimeError(
                    "Chunked inference output lost case order/count: "
                    f"expected={case_count}, actual={len(scores)}"
                )
            mismatches = [
                (index, counts[index], int(score.numel()))
                for index, score in enumerate(scores)
                if int(score.numel()) != counts[index]
            ]
            if mismatches:
                raise RuntimeError(
                    "Chunked inference output lost per-case candidate mapping: "
                    f"{mismatches}"
                )
            return scores

        if self.ablation_mode == "no_local":
            batch.patient_batch = batch.patient_batch.to(
                device, non_blocking=non_blocking
            )
            batch.prototype_batch = batch.prototype_batch.to(
                device, non_blocking=non_blocking
            )
            return validate_scores(self._score_upper(batch, None))

        chunk_size = max(1, int(local_chunk_size))
        first_graphs = batch.local_batch.to_data_list()
        second_graphs = (
            batch.local_batch_view2.to_data_list()
            if batch.local_batch_view2 is not None
            else None
        )
        if len(first_graphs) != candidate_count or (
            second_graphs is not None and len(second_graphs) != candidate_count
        ):
            raise ValueError("Local graph count does not match inference candidates")

        source_patches = batch.source_patches.to(
            device, non_blocking=non_blocking
        )
        source_unique = self.local_encoder._encode_dense(source_patches)
        source_graph_index = torch.repeat_interleave(
            torch.arange(case_count, dtype=torch.long, device=device),
            torch.tensor(counts, dtype=torch.long, device=device),
        )
        if int(source_graph_index.numel()) != candidate_count:
            raise RuntimeError("Chunked source-to-candidate mapping has invalid length")
        embedding_parts: dict[str, list[Tensor]] = {}

        for start in range(0, candidate_count, chunk_size):
            stop = min(candidate_count, start + chunk_size)
            current_count = stop - start
            target_patches = batch.target_patches[start:stop].to(
                device, non_blocking=non_blocking
            )
            target_map = self.local_encoder._encode_dense(target_patches)
            source_map = source_unique.index_select(
                0, source_graph_index[start:stop]
            )
            if int(source_map.shape[0]) != current_count:
                raise RuntimeError("Chunked source-map selection lost candidate mapping")
            first_batch = Batch.from_data_list(first_graphs[start:stop]).to(
                device, non_blocking=non_blocking
            )
            first = self.local_encoder.forward_graph(
                first_batch, source_map, target_map
            )
            if second_graphs is not None:
                second_batch = Batch.from_data_list(
                    second_graphs[start:stop]
                ).to(device, non_blocking=non_blocking)
                second = self.local_encoder.forward_graph(
                    second_batch, source_map, target_map
                )
                merged = self._mean_embeddings(first, second)
            else:
                second_batch = None
                second = None
                merged = first
            for key, value in merged.items():
                embedding_parts.setdefault(key, []).append(value)
            del target_patches, target_map, source_map, first_batch, first
            if second_batch is not None:
                del second_batch, second

        local_embeddings = {
            key: torch.cat(values, dim=0)
            for key, values in embedding_parts.items()
        }
        if any(int(value.shape[0]) != candidate_count for value in local_embeddings.values()):
            raise RuntimeError("Chunked local embeddings lost candidate order/count")
        batch.patient_batch = batch.patient_batch.to(
            device, non_blocking=non_blocking
        )
        batch.prototype_batch = batch.prototype_batch.to(
            device, non_blocking=non_blocking
        )
        return validate_scores(self._score_upper(batch, local_embeddings))

    def forward(self, payload: Any) -> HierarchicalOutput:
        batch = self._coerce_batch(payload)
        if not batch.counts:
            raise ValueError("batch must be non-empty")
        device = next(self.parameters()).device

        if self.ablation_mode == "no_local":
            non_blocking = device.type == "cuda"
            batch.patient_batch = batch.patient_batch.to(
                device, non_blocking=non_blocking
            )
            batch.prototype_batch = batch.prototype_batch.to(
                device, non_blocking=non_blocking
            )
            raw_candidate = _mask_upper_shortcuts(
                batch.patient_batch["candidate"].raw_x
            )
            local_embeddings = self._zero_local_embeddings(raw_candidate)
            consistency = raw_candidate.new_zeros(())
            scores = self._score_upper(batch, None)
            return HierarchicalOutput(
                scores=scores,
                local_batch=batch.local_batch,
                local_batch_view2=batch.local_batch_view2,
                patient_batch=batch.patient_batch,
                prototype_batch=batch.prototype_batch,
                local_embeddings=local_embeddings,
                local_embeddings_view2=None,
                consistency=consistency,
            )

        if batch.source_patches.device != device:
            batch.to(device, non_blocking=device.type == "cuda")

        source_graph_index = torch.repeat_interleave(
            torch.arange(batch.sample_count, device=device, dtype=torch.long),
            torch.tensor(batch.counts, device=device, dtype=torch.long),
        )
        source_map, target_map = self.local_encoder.encode_dense_maps(
            batch.source_patches,
            source_graph_index,
            batch.target_patches,
        )
        local_view1 = self.local_encoder.forward_graph(
            batch.local_batch, source_map, target_map
        )
        if batch.local_batch_view2 is not None:
            local_view2 = self.local_encoder.forward_graph(
                batch.local_batch_view2, source_map, target_map
            )
            consistency = self._view_consistency(local_view1, local_view2)
            local_embeddings = self._mean_embeddings(local_view1, local_view2)
        else:
            local_view2 = None
            consistency = local_view1["fused"].new_zeros(())
            local_embeddings = local_view1

        scores = self._score_upper(batch, local_embeddings)
        return HierarchicalOutput(
            scores=scores,
            local_batch=batch.local_batch,
            local_batch_view2=batch.local_batch_view2,
            patient_batch=batch.patient_batch,
            prototype_batch=batch.prototype_batch,
            local_embeddings=local_embeddings,
            local_embeddings_view2=local_view2,
            consistency=consistency,
        )
