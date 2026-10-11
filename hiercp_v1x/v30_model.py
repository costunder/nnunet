"""v3.0: full v2.5 hierarchy with a live, complete recipient-vessel branch.

The STU-Net-S L0 and original L1/L2 equations and learned score trunk remain
intact. Two sparse vessel-attention blocks retain every exported skeleton node
and real edge. Candidate attention reads every same-patient vessel node and
component; separate patient means supply an explicit whole-graph path. Sparse
message passing itself has a two-edge receptive field and is not called global.

All coordinates used for candidate-to-vessel relations are native physical
RAS millimetres. CT-normalized upper-graph positions are never reused here.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


FORMAT = "v30_full_stunet_hierarchy_complete_vessel_v1"
HIDDEN_DIM = 128
HEADS = 4
VESSEL_LAYERS = 2
NODE_FEATURE_DIM = 12
EDGE_FEATURE_DIM = 8
RADIUS_COLUMN = 3


def _require(condition: Tensor, message: str) -> None:
    """Assert tensor contracts without synchronizing CUDA to Python."""
    torch._assert_async(condition, message)


def _mean(values: Tensor, owners: Tensor, count: int) -> Tensor:
    # Long real skeletons require float32 accumulation under AMP as well;
    # bfloat16 counters cannot represent every point count above 256 exactly.
    precise = values.float()
    sums = precise.new_zeros((count, values.shape[-1])).index_add(0, owners, precise)
    counts = precise.new_zeros(count).index_add(0, owners, precise.new_ones(len(values)))
    _require((counts > 0).all(), "Every declared vessel patient/component requires real nodes")
    return (sums / counts[:, None]).to(values.dtype)


def _validate_vessel_batch(batch, device: torch.device) -> None:
    required = ("node_features", "pos_mm", "edge_index", "edge_attr",
                "node_patient_index", "component_index", "component_patient_index", "case_ids")
    if not all(hasattr(batch, name) for name in required):
        raise TypeError("A complete disjoint-union VesselBatch is required")
    nodes, components, patients = len(batch.node_features), len(batch.component_patient_index), len(batch.case_ids)
    if nodes < 1 or components < 1 or patients < 1:
        raise ValueError("Empty vessel graphs are not replaced with fabricated context")
    if batch.node_features.shape != (nodes, NODE_FEATURE_DIM) or batch.pos_mm.shape != (nodes, 3):
        raise ValueError("Vessel nodes require all 12 features and physical RAS-mm coordinates")
    if batch.edge_index.ndim != 2 or batch.edge_index.shape[0] != 2:
        raise ValueError("Vessel edges require [2,E] real native connectivity")
    if batch.edge_attr.shape != (batch.edge_index.shape[1], EDGE_FEATURE_DIM):
        raise ValueError("Every directed vessel edge requires all eight physical features")
    if batch.node_patient_index.shape != (nodes,) or batch.component_index.shape != (nodes,):
        raise ValueError("Every vessel node requires patient and component ownership")
    for name in required[:-1]:
        value = getattr(batch, name)
        if not isinstance(value, Tensor) or value.device != device:
            raise ValueError("All vessel tensors must share the actual score device")
    for name in ("edge_index", "node_patient_index", "component_index", "component_patient_index"):
        if getattr(batch, name).dtype != torch.long:
            raise ValueError("Vessel connectivity and ownership must use int64 indices")
    for name in ("node_features", "pos_mm", "edge_attr"):
        value = getattr(batch, name)
        if not value.is_floating_point():
            raise ValueError("Vessel geometry and features must be floating point")
        _require(torch.isfinite(value).all(), "Nonfinite physical vessel input")
    owner, component, component_owner = batch.node_patient_index, batch.component_index, batch.component_patient_index
    _require(((owner >= 0) & (owner < patients)).all(), "Vessel node patient outside batch")
    _require(((component >= 0) & (component < components)).all(), "Vessel component outside batch")
    _require(((component_owner >= 0) & (component_owner < patients)).all(), "Vessel component patient outside batch")
    _require((component_owner[component] == owner).all(), "Vessel component crosses patient boundaries")
    _require((batch.node_features[:, RADIUS_COLUMN] >= 0).all(), "Vessel EDT radius must be nonnegative")
    source, target = batch.edge_index
    _require(((batch.edge_index >= 0) & (batch.edge_index < nodes)).all(), "Vessel edge endpoint outside batch")
    _require((source != target).all(), "Only real skeleton edges are allowed; no fabricated self edges")
    _require((owner[source] == owner[target]).all(), "Vessel edge crosses patient boundaries")
    _require((component[source] == component[target]).all(), "Vessel edge joins disconnected native components")


class VesselMessageBlock(nn.Module):
    """128D/four-head sparse GATv2-style update, residual LN and 512D FFN."""

    def __init__(self, dropout: float):
        super().__init__()
        self.hidden_dim, self.heads = HIDDEN_DIM, HEADS
        self.channels = HIDDEN_DIM // HEADS
        self.source = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.target = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.edge = nn.Linear(EDGE_FEATURE_DIM, HIDDEN_DIM, bias=False)
        self.attention = nn.Parameter(torch.empty(HEADS, self.channels))
        nn.init.xavier_uniform_(self.attention)
        self.message_norm = nn.LayerNorm(HIDDEN_DIM)
        self.ffn_norm = nn.LayerNorm(HIDDEN_DIM)
        self.ffn = nn.Sequential(nn.Linear(HIDDEN_DIM, 4 * HIDDEN_DIM), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(4 * HIDDEN_DIM, HIDDEN_DIM))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        source, target = edge_index
        left = self.source(x).reshape(-1, HEADS, self.channels)
        right = self.target(x).reshape(-1, HEADS, self.channels)
        relation = self.edge(edge_attr).reshape(-1, HEADS, self.channels)
        logits = (F.leaky_relu(left[source] + right[target] + relation, .2)
                  * self.attention).sum(-1).float()
        index = target[:, None].expand(-1, HEADS)
        maxima = logits.new_full((len(x), HEADS), -torch.inf)
        maxima = maxima.scatter_reduce(0, index, logits.detach(), reduce="amax", include_self=True)
        exponent = (logits - maxima[target]).exp()
        denominator = logits.new_zeros((len(x), HEADS)).scatter_add(0, index, exponent)
        weights = exponent / denominator[target].clamp_min(torch.finfo(logits.dtype).tiny)
        # Compatibility gating matches the retained hierarchy's distinction
        # between relative neighbor competition and absolute pair compatibility.
        weights = self.dropout(weights * torch.sigmoid(logits))
        messages = left[source] * weights.to(left.dtype)[..., None]
        summed = messages.new_zeros((len(x), HEADS, self.channels), dtype=torch.float32).index_add(0, target, messages.float())
        current = self.message_norm(x + self.dropout(summed.flatten(1).to(x.dtype)))
        return self.ffn_norm(current + self.dropout(self.ffn(current)))


class VesselEncoder(nn.Module):
    def __init__(self, dropout: float = .1):
        super().__init__()
        self.hidden_dim, self.heads, self.layers = HIDDEN_DIM, HEADS, VESSEL_LAYERS
        self.project = nn.Sequential(nn.Linear(NODE_FEATURE_DIM, HIDDEN_DIM),
                                     nn.LayerNorm(HIDDEN_DIM), nn.SiLU())
        self.blocks = nn.ModuleList(VesselMessageBlock(dropout) for _ in range(VESSEL_LAYERS))

    def forward(self, batch) -> Tensor:
        x = self.project(batch.node_features)
        for block in self.blocks:
            if self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, batch.edge_index, batch.edge_attr, use_reentrant=False)
            else:
                x = block(x, batch.edge_index, batch.edge_attr)
        return x


class PhysicalVesselAttention(nn.Module):
    """Exact all-node attention; tiling changes working memory, never coverage."""

    def __init__(self, *, query_chunk_size: int = 128, node_chunk_size: int = 1024):
        super().__init__()
        if type(query_chunk_size) is not int or query_chunk_size < 1 or type(node_chunk_size) is not int or node_chunk_size < 1:
            raise ValueError("Attention working-memory tile sizes must be positive integers")
        self.query_chunk_size, self.node_chunk_size = query_chunk_size, node_chunk_size
        self.key = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.value = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.geometry = nn.Sequential(nn.Linear(5, HIDDEN_DIM), nn.SiLU(),
                                      nn.Linear(HIDDEN_DIM, HEADS, bias=False))

    def _tile(self, query, key, value, candidate_pos, node_pos, radius, candidate_owner, node_owner):
        delta = node_pos[None].float() - candidate_pos[:, None].float()
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        radius = radius.float()[None, :, None].expand(len(query), -1, -1)
        physical = torch.cat((delta.sign() * torch.log1p(delta.abs()), torch.log1p(distance),
                              torch.log1p(radius)), -1)
        geometry = self.geometry(physical).permute(0, 2, 1).float()
        with torch.autocast(query.device.type, enabled=False):
            logits = torch.einsum("qhd,khd->qhk", query.float(), key.float()) / math.sqrt(HIDDEN_DIM // HEADS)
        valid = candidate_owner[:, None] == node_owner[None]
        logits = (logits + geometry).masked_fill(~valid[:, None], -torch.inf)
        maximum = logits.amax(-1).detach()
        safe_maximum = maximum.masked_fill(~torch.isfinite(maximum), 0)
        exponent = (logits - safe_maximum[..., None]).exp()
        denominator = exponent.sum(-1)
        with torch.autocast(query.device.type, enabled=False):
            numerator = torch.einsum("qhk,khd->qhd", exponent, value.float())
        return maximum, denominator, numerator

    def forward(self, query, nodes, candidate_pos_mm, node_pos_mm, radius_mm, candidate_owner, node_owner):
        keys = self.key(nodes).reshape(-1, HEADS, HIDDEN_DIM // HEADS)
        values = self.value(nodes).reshape(-1, HEADS, HIDDEN_DIM // HEADS)
        outputs = []
        for begin in range(0, len(query), self.query_chunk_size):
            end = begin + self.query_chunk_size
            q = query[begin:end].reshape(-1, HEADS, HIDDEN_DIM // HEADS)
            maximum = q.new_full((len(q), HEADS), -torch.inf, dtype=torch.float32)
            denominator = q.new_zeros((len(q), HEADS), dtype=torch.float32)
            numerator = q.new_zeros((len(q), HEADS, HIDDEN_DIM // HEADS), dtype=torch.float32)
            for start in range(0, len(nodes), self.node_chunk_size):
                stop = start + self.node_chunk_size
                args = (q, keys[start:stop], values[start:stop], candidate_pos_mm[begin:end],
                        node_pos_mm[start:stop], radius_mm[start:stop],
                        candidate_owner[begin:end], node_owner[start:stop])
                tile = checkpoint(self._tile, *args, use_reentrant=False) if self.training and torch.is_grad_enabled() else self._tile(*args)
                next_maximum = torch.maximum(maximum, tile[0])
                safe_next = next_maximum.masked_fill(~torch.isfinite(next_maximum), 0)
                old_scale = (maximum - safe_next).exp()
                tile_scale = (tile[0] - safe_next).exp()
                denominator = denominator * old_scale + tile[1] * tile_scale
                numerator = numerator * old_scale[..., None] + tile[2] * tile_scale[..., None]
                maximum = next_maximum
            _require((denominator > 0).all(), "Every candidate must attend real vessels from its own patient")
            outputs.append((numerator / denominator[..., None]).flatten(1).to(query.dtype))
        return torch.cat(outputs, 0)


class VesselCandidateReadout(nn.Module):
    def __init__(self, query_dim: int, *, query_chunk_size=128, node_chunk_size=1024):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(query_dim, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.SiLU())
        self.nodes = PhysicalVesselAttention(query_chunk_size=query_chunk_size, node_chunk_size=node_chunk_size)
        self.components = PhysicalVesselAttention(query_chunk_size=query_chunk_size, node_chunk_size=node_chunk_size)
        self.fuse = nn.Sequential(nn.Linear(5 * HIDDEN_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.SiLU())

    def forward(self, pair, encoded, batch, candidate_pos_mm, candidate_patient_index):
        query = self.query(pair)
        components, patients = len(batch.component_patient_index), len(batch.case_ids)
        # Every node enters its actual component; every actual component is
        # retained independently even when it has no physical edges.
        component_states = _mean(encoded, batch.component_index, components)
        component_pos = _mean(batch.pos_mm, batch.component_index, components)
        radius = batch.node_features[:, RADIUS_COLUMN]
        component_radius = _mean(radius[:, None], batch.component_index, components).squeeze(-1)
        whole_nodes = _mean(encoded, batch.node_patient_index, patients)
        whole_components = _mean(component_states, batch.component_patient_index, patients)
        local = self.nodes(query, encoded, candidate_pos_mm, batch.pos_mm, radius,
                           candidate_patient_index, batch.node_patient_index)
        complete_components = self.components(query, component_states, candidate_pos_mm, component_pos,
                                              component_radius, candidate_patient_index, batch.component_patient_index)
        return self.fuse(torch.cat((local, complete_components, whole_nodes[candidate_patient_index],
                                   whole_components[candidate_patient_index], query * local), -1))


@dataclass(frozen=True)
class _VesselContext:
    batch: object
    candidate_pos_mm: Tensor
    candidate_patient_index: Tensor


class VesselAugmentedScoreHead(nn.Module):
    """Original 1546->512->256 score trunk plus learned vessel input at 512."""

    def __init__(self, original: nn.Sequential, *, dropout: float, query_chunk_size=128, node_chunk_size=1024):
        super().__init__()
        if not isinstance(original, nn.Sequential) or len(original) != 8 or not isinstance(original[0], nn.Linear):
            raise ValueError("The complete original eight-stage score trunk is required")
        if original[0].out_features != 512 or original[4].in_features != 512 or original[4].out_features != 256:
            raise ValueError("v3.0 preserves the original 512/256 score widths")
        self.original = original
        self.vessel_encoder = VesselEncoder(dropout)
        self.vessel_readout = VesselCandidateReadout(original[0].in_features,
            query_chunk_size=query_chunk_size, node_chunk_size=node_chunk_size)
        self.vessel_fusion = nn.Linear(HIDDEN_DIM, original[0].out_features, bias=False)
        self._context = None

    def __getitem__(self, index):
        return self.original[index]

    def __len__(self):
        return len(self.original)

    def forward(self, pair: Tensor) -> Tensor:
        context = self._context
        if context is None:
            raise RuntimeError("v3.0 scores require explicit complete vessels and native RAS-mm candidate coordinates")
        batch, pos, owner = context.batch, context.candidate_pos_mm, context.candidate_patient_index
        if pair.ndim != 2 or pair.shape[1] != self.original[0].in_features or len(pair) < 1:
            raise ValueError("Complete original candidate/donor-conditioned score features required")
        if pos.shape != (len(pair), 3) or owner.shape != (len(pair),):
            raise ValueError("One physical RAS-mm position and patient owner per original candidate required")
        if pos.device != pair.device or owner.device != pair.device or owner.dtype != torch.long or not pos.is_floating_point():
            raise ValueError("Explicit candidate physical coordinates and ownership must share the score device")
        _validate_vessel_batch(batch, pair.device)
        _require(torch.isfinite(pos).all(), "Nonfinite candidate physical RAS coordinates")
        _require(((owner >= 0) & (owner < len(batch.case_ids))).all(), "Candidate patient outside complete vessel batch")
        encoded = self.vessel_encoder(batch)
        vessels = self.vessel_readout(pair, encoded, batch, pos, owner)
        x = self.original[0](pair) + self.vessel_fusion(vessels)
        for layer in self.original[1:]:
            x = layer(x)
        return x


@contextmanager
def vessel_score_context(model, vessel_batch, candidate_pos_mm: Tensor, candidate_patient_index: Tensor):
    """Bind explicit physical inputs for exactly one scoring call; always release.

    No learned state or CUDA autograd graph is cached between calls, optimizer
    updates, inference, or calibration clones. Nested use restores its caller's
    context and exceptions cannot leave a patient's vessel graph attached.
    """
    head = model.score_head
    if not isinstance(head, VesselAugmentedScoreHead):
        raise TypeError("Install the v3.0 vessel branch before binding score inputs")
    previous = head._context
    head._context = _VesselContext(vessel_batch, candidate_pos_mm, candidate_patient_index)
    try:
        yield model
    finally:
        head._context = previous


def install_vessel_branch(model, *, query_chunk_size=128, node_chunk_size=1024):
    """Attach trainable vessels once, preserving every existing trunk weight."""
    if isinstance(model.score_head, VesselAugmentedScoreHead):
        raise ValueError("The v3.0 vessel branch is already installed")
    if model.hidden_dim != HIDDEN_DIM or model.ablation_mode != "full":
        raise ValueError("v3.0 retains the complete 128D hierarchy")
    for name in ("patient_encoder", "prototype_encoder"):
        blocks = getattr(model, name).blocks
        if len(blocks) != 2 or any(conv.heads != HEADS for block in blocks for conv in block.conv.convs.values()):
            raise ValueError("v3.0 retains both original two-layer/four-head L1 and L2")
    if not hasattr(model, "v24_stunet_contract"):
        raise ValueError("Install the full pretrained STU-Net-S L0 before the vessel branch")
    original = model.score_head
    dropout = original[3].p
    parameter = next(original.parameters())
    head = VesselAugmentedScoreHead(original, dropout=dropout,
        query_chunk_size=query_chunk_size, node_chunk_size=node_chunk_size)
    head.to(device=parameter.device, dtype=parameter.dtype)
    head.train(model.training)
    model.score_head = head
    model.architecture_version += "|v30_complete_native_vessel"
    parameters = sum(value.numel() for value in model.parameters())
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    model.v24_model_contract = dict(model.v24_model_contract, parameters=parameters, trainable_parameters=trainable)
    model.v30_model_contract = dict(format=FORMAT, version="3.0", parameters=parameters,
        trainable_parameters=trainable, recipient_GT_used_in_forward=False,
        original_retained_operator_equations=True, base_model_contract=copy.deepcopy(model.v24_model_contract),
        base_stunet_contract=copy.deepcopy(model.v24_stunet_contract),
        vessel_hidden_dim=HIDDEN_DIM, vessel_heads=HEADS, vessel_layers=VESSEL_LAYERS,
        vessel_ffn_dim=4 * HIDDEN_DIM, vessel_node_feature_dim=NODE_FEATURE_DIM,
        vessel_edge_feature_dim=EDGE_FEATURE_DIM,
        vessel_parameters=sum(value.numel() for name, value in head.named_parameters() if not name.startswith("original.")),
        whole_graph_readout="all_same_patient_nodes_and_components_plus_both_patient_means",
        candidate_geometry="native_physical_RAS_mm_delta_distance_EDT_radius",
        donor_conditioning="complete_original_fused_score_input",
        original_score_input_dim=original[0].in_features, original_score_widths=[512, 256],
        fusion="learned_128_to_512_residual_before_original_first_LayerNorm",
        skeleton_nodes_capped=False, skeleton_edges_pruned=False, disconnected_components_joined=False,
        learned_embeddings_cached=False, query_chunk_size=query_chunk_size, node_chunk_size=node_chunk_size,
        mixture_density_network=False, monoplacement=False)
    return model


def build_v30_model(model_config, *, original_snapshot_root=None, stunet_checkpoint,
                    query_chunk_size=128, node_chunk_size=1024):
    from hiercp_v1x.v24_model import build_gt_free_model
    from hiercp_v1x.v24_stunet import install_stunet_l0
    model = build_gt_free_model(model_config, original_snapshot_root=original_snapshot_root)
    install_stunet_l0(model, stunet_checkpoint)
    return install_vessel_branch(model, query_chunk_size=query_chunk_size, node_chunk_size=node_chunk_size)
