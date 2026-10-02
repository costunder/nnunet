"""Isolated cumulative model changes on a constructed ORIGINAL v1 model.

No import of current hiercp: the launcher first verifies/extracts original v1.
v1.0 is unchanged. v1.1 retains CNN activations while the large native graph
blocks keep their original checkpointing; v1.2 removes only handcrafted local
projection columns; v1.3 replaces local GAT with edge-conditioned mean GraphSAGE
on the unchanged full directed graph, keeping original residuals.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


STAGES = ("v1.0", "v1.1", "v1.2", "v1.3")
STAGE_REVISIONS = {stage: index for index, stage in enumerate(STAGES)}


def execution_overrides(stage: str) -> dict[str, bool]:
    """Constructor execution flags only, with no shape/batch/sampling change."""
    if stage not in STAGES:
        raise ValueError(f"Unknown v1.x stage {stage!r}; expected {STAGES}")
    if stage == "v1.0":
        return {}
    # The actual two-view CT batch has ~210k nodes/~14.2m edges per view.
    # Retaining every local GAT block exceeded the explicit 12 GiB DEBUG budget;
    # CNN-only retention preserves the entire graph and original block execution.
    return {"checkpoint_local_blocks": True, "checkpoint_dense_encoder": False}


class DenseSliceProjection(nn.Module):
    """Keep original concat API; allocate no discarded handcrafted weights."""

    def __init__(self, original: nn.Sequential, handcrafted_dim: int):
        super().__init__()
        linear = original[0]
        if not isinstance(linear, nn.Linear) or len(original) != 3:
            raise ValueError("Expected original v1 Linear/LayerNorm/SiLU projection")
        self.handcrafted_dim = int(handcrafted_dim)
        self.input_dim = int(linear.in_features)
        dense_dim = self.input_dim - self.handcrafted_dim
        if dense_dim <= 0:
            raise ValueError("Local projection has no dense CNN columns")
        self.dense = nn.Linear(dense_dim, linear.out_features, bias=linear.bias is not None,
                               device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            self.dense.weight.copy_(linear.weight[:, self.handcrafted_dim:])
            if linear.bias is not None:
                self.dense.bias.copy_(linear.bias)
        self.norm = copy.deepcopy(original[1])
        self.activation = copy.deepcopy(original[2])

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"Original local concat must have width {self.input_dim}")
        return self.activation(self.norm(self.dense(features[:, self.handcrafted_dim:])))


class MeanTopologyCache:
    """One complete batch CSR normalization, shared by all three local layers.

    Each original edge gets inverse destination degree; coalescing duplicates
    by SUM preserves multiplicity. Topology and immutable raw edge-attribute
    means are cached, never live learned features or gradient-bearing attrs.
    Changing/mutating edge_index invalidates the one-batch cache.
    """

    def __init__(self):
        self.signature = None
        self.entries = {}
        self.edge_references = {}
        self.attribute_means = {}
        self.builds = 0

    def prepare(self, node_types, edge_types, x_dict, edge_index_dict):
        if set(x_dict) != set(node_types) or set(edge_index_dict) != set(edge_types):
            raise ValueError("SAGE requires all original node types and relations")
        signature = tuple((relation, id(edge_index_dict[relation]),
                           edge_index_dict[relation]._version,
                           x_dict[relation[0]].shape[0], x_dict[relation[2]].shape[0],
                           str(edge_index_dict[relation].device)) for relation in edge_types)
        if signature == self.signature:
            return
        entries = {}
        for relation in edge_types:
            edges = edge_index_dict[relation]
            source, target = x_dict[relation[0]], x_dict[relation[2]]
            if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
                raise ValueError(f"Complete int64 [2,E] COO required for {relation}")
            if edges.device != source.device or edges.device != target.device:
                raise ValueError("Node and edge devices differ")
            if edges.numel():
                valid = ((edges >= 0).all() & (edges[0] < source.shape[0]).all()
                         & (edges[1] < target.shape[0]).all())
                if edges.device.type == "cuda":
                    torch._assert_async(valid, "SAGE edge endpoint outside its node type")
                elif not bool(valid):
                    raise ValueError("SAGE edge endpoint outside its node type")
            degree = torch.bincount(edges[1], minlength=target.shape[0]).float()
            inverse = degree.clamp_min(1).reciprocal()
            matrix = torch.sparse_coo_tensor(edges.flip(0), inverse[edges[1]],
                                            (target.shape[0], source.shape[0]),
                                            device=edges.device).coalesce().to_sparse_csr()
            entries[relation] = (matrix, inverse, degree > 0)
        self.entries = entries
        # Strong one-batch references prevent a freed tensor's Python id from
        # being recycled into a false cache hit on a later equal-sized batch.
        self.edge_references = dict(edge_index_dict)
        self.attribute_means = {}
        self.signature = signature
        self.builds += 1

    def mean_attributes(self, relation, edges, attributes, destination_count, dtype):
        """Reuse immutable graph attributes; keep live attribute gradients live."""
        signature = (id(attributes), attributes._version, dtype)
        cached = self.attribute_means.get(relation)
        if (not attributes.requires_grad and cached is not None
                and cached[0] == signature and cached[1] is attributes):
            return cached[2]
        inverse = self.entries[relation][1]
        mean = torch.zeros((destination_count, attributes.shape[1]),
                           device=attributes.device, dtype=dtype)
        mean.index_add_(0, edges[1], attributes.to(dtype))
        mean = mean * inverse.to(dtype)[:, None]
        if not attributes.requires_grad:
            self.attribute_means[relation] = (signature, attributes, mean)
        return mean


class EdgeConditionedMeanRelation(nn.Module):
    """mean(W_r h_source + E_r edge_attr + b_r); no per-relation root.

    Linear commutes with mean. Mean-first avoids E×128 feature expansion.
    Edge attributes remain the original E×10 input. Empty relation message is
    zero; the original block supplies one identity residual/root contribution.
    """

    def __init__(self, original, relation, cache: MeanTopologyCache):
        super().__init__()
        self.relation, self.cache = relation, cache
        self.dim = int(original.lin_l.out_channels)
        self.neighbor = nn.Linear(self.dim, self.dim, bias=True,
                                  device=original.lin_l.weight.device,
                                  dtype=original.lin_l.weight.dtype)
        self.edge = nn.Linear(int(original.lin_edge.in_channels), self.dim, bias=False,
                              device=original.lin_l.weight.device,
                              dtype=original.lin_l.weight.dtype)
        with torch.no_grad():
            # Controlled initialization transfer, NOT GAT functional parity.
            self.neighbor.weight.copy_(original.lin_l.weight)
            self.neighbor.bias.zero_()
            if original.lin_l.bias is not None:
                self.neighbor.bias.add_(original.lin_l.bias)
            if original.bias is not None:
                self.neighbor.bias.add_(original.bias)
            self.edge.weight.copy_(original.lin_edge.weight)

    def forward(self, source: Tensor, target: Tensor, edges: Tensor, attributes: Tensor):
        if attributes.ndim != 2 or attributes.shape != (edges.shape[1], self.edge.in_features):
            raise ValueError(f"Missing/incompatible original edge attrs for {self.relation}")
        if attributes.device != source.device or target.device != source.device:
            raise ValueError("Node and attribute devices differ")
        matrix, _, has_neighbors = self.cache.entries[self.relation]
        # Explicit FP32 sparse CUDA math remains GPU; outer CNN/MLP autocast
        # stays intact. FP64 is available for numerical UNIT parity.
        arithmetic_dtype = torch.float64 if source.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=source.device.type, enabled=False):
            mean_source = torch.sparse.mm(matrix.to(dtype=arithmetic_dtype), source.to(arithmetic_dtype))
            mean_attributes = self.cache.mean_attributes(
                self.relation, edges, attributes, target.shape[0], arithmetic_dtype)
            message = (F.linear(mean_source, self.neighbor.weight.to(arithmetic_dtype),
                                self.neighbor.bias.to(arithmetic_dtype))
                       + F.linear(mean_attributes, self.edge.weight.to(arithmetic_dtype)))
            message = message * has_neighbors[:, None]
        return message.to(source.dtype)


class RelationMeanComposition(nn.Module):
    """Original heterogeneous SUM composition with independent relation means."""

    def __init__(self, original_block, cache: MeanTopologyCache):
        super().__init__()
        self.node_types, self.edge_types = tuple(original_block.node_types), tuple(original_block.edge_types)
        self.cache = cache
        self.relations = nn.ModuleList([
            EdgeConditionedMeanRelation(original_block.conv.convs[relation], relation, cache)
            for relation in self.edge_types
        ])

    def forward(self, x_dict: Mapping[str, Tensor], edge_index_dict, *, edge_attr_dict):
        if set(edge_attr_dict) != set(self.edge_types):
            raise ValueError("All original local edge attributes are mandatory")
        self.cache.prepare(self.node_types, self.edge_types, x_dict, edge_index_dict)
        messages = {}
        for relation, conv in zip(self.edge_types, self.relations):
            source, _, destination = relation
            message = conv(x_dict[source], x_dict[destination],
                           edge_index_dict[relation], edge_attr_dict[relation])
            messages[destination] = messages.get(destination, 0) + message
        return messages


def _reject_stage_state(module, state_dict, prefix, local_metadata, strict,
                        missing_keys, unexpected_keys, error_msgs):
    saved = state_dict.get(prefix + "_v1x_stage_revision")
    expected = STAGE_REVISIONS[module.v1x_stage]
    if (saved is None or not isinstance(saved, Tensor) or saved.numel() != 1
            or int(saved.detach().cpu()) != expected):
        raise RuntimeError(f"Exact resume requires {module.v1x_stage} revision {expected}; "
                           "another stage is a new experiment, not resume")


def apply_variant(model: nn.Module, stage: str) -> nn.Module:
    """Apply cumulative stage to a freshly constructed preserved v1 model."""
    overrides = execution_overrides(stage)
    previous = getattr(model, "v1x_stage", None)
    if previous is not None:
        if previous == stage:
            return model
        raise ValueError("Construct a fresh original v1 model for each stage")
    if stage == "v1.0":
        return model
    local = model.local_encoder
    if (len(local.project) != 6 or len(local.blocks) != 3 or local.hidden_dim != 128
            or any(len(block.edge_types) != 16 for block in local.blocks)):
        raise ValueError("Expected preserved v1 6-role/16-relation/3-layer/128D L0")
    for name, value in overrides.items():
        setattr(local, name, value)
    if STAGE_REVISIONS[stage] >= 2:
        convolutions = [layer for layer in local.dense_encoder.modules() if isinstance(layer, nn.Conv3d)]
        if not convolutions or convolutions[0].in_channels != 5:
            raise ValueError("v1.x retains original five-channel CNN inputs")
        dense_dim = convolutions[-1].out_channels
        if {projection[0].in_features - dense_dim for projection in local.project.values()} != {16}:
            raise ValueError("Expected original sixteen handcrafted local columns")
        # Shape-only constructors otherwise perturb dropout RNG. Preserve it.
        devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
        with torch.random.fork_rng(devices=devices):
            local.project = nn.ModuleDict({role: DenseSliceProjection(projection, 16)
                                          for role, projection in local.project.items()})
            if stage == "v1.3":
                cache = MeanTopologyCache()
                for block in local.blocks:
                    block.conv = RelationMeanComposition(block, cache)
    model.v1x_stage = stage
    model.architecture_version = f"{model.architecture_version}|v1x_{stage}_r1"
    model.register_buffer("_v1x_stage_revision", torch.tensor(STAGE_REVISIONS[stage],
                                                           dtype=torch.int64,
                                                           device=next(model.parameters()).device))
    model.register_load_state_dict_pre_hook(_reject_stage_state)
    return model
