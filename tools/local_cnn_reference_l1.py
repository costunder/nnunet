"""Opt-in cloned PRODIGY basic-operator L1 control, never a release default.

The independent tensor implementation below follows ``MetaGNNLayer`` and the
inter-layer GELU in snap-stanford/prodigy commit
107ba57234d3188227cda5b78a2dbcfb84a1c694.  It does not import or install the
retained official source.  L0, the two patient-label nodes, L2, clustering,
prototype readout, temperature, and loss remain the project's own design.

Both support and query must execute together in training: the reference layer
uses one BatchNorm over the complete prompt graph.  Query targets have no input
argument.  Query features influence normalization statistics in train mode,
although no query-to-support/label message edges exist.  In eval mode, fixed BN
running statistics restore query-chunk independence.  Fresh BN statistics in a
transferred candidate are explicitly untrained; this is not an exact resume.
"""
import copy
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import softmax

from hiercp_v222.clustering import alignment_loss, prototype_logits
from hiercp_v222.model import PromptGraphModel, RelationLayer
from hiercp_v222.v1_execution import rng_state, restore_rng


ARCHITECTURE = 'diagnostic_l1_prodigy_basic_joint_v1'
OFFICIAL_COMMIT = '107ba57234d3188227cda5b78a2dbcfb84a1c694'
OFFICIAL_SOURCE_SHA256 = 'fd94b26236b0edddfa585fc69f2d2f03e5deafd29d2a2a1b2e30bbc4267d9074'


def _state_hash(state):
    """Use the project's mixed-state hash, including LocalCNN input contracts.

    LocalCNN exposes its configuration through ``get_extra_state``; that dict
    is part of the state identity just as much as parameter and buffer tensors.
    This admission check is outside the measured update path.
    """
    from l0_regions.training import hash_state
    return hash_state(state)


def _partition_hash(net, prefixes):
    return _state_hash({name: value for name, value in net.state_dict().items()
                        if name.startswith(prefixes)})


def source_identity():
    """Verify the pinned audit reference before constructing this control."""
    path = Path(__file__).resolve().parents[1] / 'validation/prompt_graph_audit_20261001/official/models/metaGNN.py'
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != OFFICIAL_SOURCE_SHA256:
        raise ValueError('Pinned official MetaGNN source changed; reference control not admitted')
    return dict(repository='https://github.com/snap-stanford/prodigy',
                commit=OFFICIAL_COMMIT, path='models/metaGNN.py', sha256=actual)


class ReferenceRelationLayer(nn.Module):
    """Basic official operator equations, with no unused FFN or LayerNorm.

    Self-loops and relation coding belong to ``joint_topology``, just as the
    official graph wrapper supplies them rather than ``MetaGNNLayer`` itself.
    Parameter names match the pinned basic layer for strict independent-oracle
    state transfer: mlp_kqv, lin_edge, att_mlp, out_proj, bn.
    """
    def __init__(self, dim, heads, dropout):
        super().__init__()
        if type(dim) is not int or type(heads) is not int or dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError('Positive dimension divisible by the original head count required')
        if isinstance(dropout, bool) or not math.isfinite(float(dropout)) or not 0 <= float(dropout) < 1:
            raise ValueError('Finite original attention/residual dropout required')
        self.emb_dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.width = self.head_dim
        self.dropout = float(dropout)
        self.mlp_kqv = nn.Linear(dim, 3*dim)
        self.lin_edge = nn.Linear(2, dim)
        self.att_mlp = nn.Sequential(nn.Linear(3*self.head_dim, self.head_dim),
                                     nn.ReLU(), nn.Linear(self.head_dim, 1))
        self.out_proj = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward_with_terms(self, nodes, edges, features):
        """Run each edge once; return optional instrumentation from that run."""
        if nodes.ndim != 2 or nodes.shape[1] != self.emb_dim or not len(nodes):
            raise ValueError('Complete prompt-node matrix with original dimension required')
        if (edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2
                or features.shape != (edges.shape[1], 2)
                or edges.device != nodes.device or features.device != nodes.device):
            raise ValueError('Bound int64 directed edges and two relation attributes required')
        if (not edges.shape[1] or bool(((edges < 0) | (edges >= len(nodes))).any())
                or not bool(torch.isfinite(nodes).all() & torch.isfinite(features).all())):
            raise ValueError('Finite prompt data and in-range complete edges required')
        source, destination = edges
        qkv = self.mlp_kqv(nodes)
        q = qkv[:, :self.emb_dim].reshape(-1, self.heads, self.head_dim)
        k = qkv[:, self.emb_dim:2*self.emb_dim].reshape(-1, self.heads, self.head_dim)
        v = qkv[:, 2*self.emb_dim:].reshape(-1, self.heads, self.head_dim)
        edge = self.lin_edge(features.to(nodes.dtype)).reshape(-1, self.heads, self.head_dim)
        logits = self.att_mlp(torch.cat((k[source]/math.sqrt(self.head_dim),
                                         q[destination], F.relu(edge)), dim=-1))
        # Keep the official softmax dtype; promoting it is a different control.
        probabilities = softmax(logits, destination, num_nodes=len(nodes))
        weights = F.dropout(probabilities, p=self.dropout, training=self.training)
        # Official bias is applied per edge, *after* weighting each head's value.
        # Moving this projection outside scatter changes degree*bias to one bias.
        per_edge = self.out_proj((weights*v[source]).reshape(-1, self.emb_dim))
        aggregate = per_edge.new_zeros(nodes.shape)
        aggregate.index_add_(0, destination, per_edge)
        pre_bn = nodes + F.dropout(aggregate, p=self.dropout, training=self.training)
        output = self.bn(pre_bn)
        return output, dict(logits=logits, probabilities=probabilities, weights=weights,
                            aggregate=aggregate, pre_bn=pre_bn)

    def forward(self, nodes, edges, features):
        return self.forward_with_terms(nodes, edges, features)[0]


def joint_topology(support, owners, classes, query, label_seed):
    """Build support↔patient labels, labels→queries, and [0,0] self-loops.

    Observed support labels alone define T=[0,1] and F=[0,-1].  Every query
    receives U=[1,0], independent of any held-out query class.  The existing
    project's two label nodes per patient are preserved rather than replaced
    by the official task-global label topology.
    """
    if (support.ndim != 2 or query.ndim != 2 or label_seed.shape != (2, support.shape[1])
            or query.shape[1] != support.shape[1] or not len(support)
            or owners.shape != classes.shape or owners.shape != (len(support),)
            or owners.dtype != torch.long or classes.dtype != torch.long
            or any(value.device != support.device for value in (owners, classes, query, label_seed))
            or query.dtype != support.dtype or label_seed.dtype != support.dtype):
        raise ValueError('Matched finite support/query/label tensors and int64 ownership required')
    if not bool(torch.isfinite(support).all() & torch.isfinite(query).all() & torch.isfinite(label_seed).all()):
        raise FloatingPointError('Nonfinite prompt input; no fallback')
    if bool((owners < 0).any()) or bool(((classes < 0) | (classes > 1)).any()):
        raise ValueError('Nonnegative owners and binary observed support classes required')
    patients = int(owners.max())+1
    if (patients < 2 or bool((torch.bincount(owners, minlength=patients) == 0).any())
            or len(classes.unique()) != 2):
        raise ValueError('Two contiguous nonempty support patients and both observed classes required')
    ns = len(support)
    nl = 2*patients
    nq = len(query)
    query_start = ns+nl
    nodes = torch.cat((support, label_seed.repeat(patients, 1), query))
    data = torch.arange(ns, device=support.device).repeat_interleave(2)
    cls = torch.arange(2, device=support.device).repeat(ns)
    label = ns+2*owners.repeat_interleave(2)+cls
    signed = torch.where(classes.repeat_interleave(2) == cls, 1., -1.)
    known = torch.stack((torch.zeros_like(signed), signed), -1).to(support.dtype)
    q_source = torch.arange(ns, ns+nl, device=support.device).repeat(nq)
    q_dest = torch.arange(query_start, query_start+nq, device=support.device).repeat_interleave(nl)
    unknown = support.new_zeros(len(q_source), 2)
    unknown[:, 0] = 1.
    self_nodes = torch.arange(len(nodes), device=support.device)
    edges = torch.stack((torch.cat((data, label, q_source, self_nodes)),
                         torch.cat((label, data, q_dest, self_nodes))))
    features = torch.cat((known, known, unknown, support.new_zeros(len(nodes), 2)))
    return dict(nodes=nodes, edge_index=edges, edge_attr=features, support_count=ns,
                label_count=nl, query_start=query_start, patients=patients,
                query_count=nq, self_loops=len(nodes), query_reverse_edges=0)


class ReferencePromptGraphModel(PromptGraphModel):
    """Cloned project model with joint basic-operator L1 and original live L2.

    ``prepare_support`` returns an immutable binding context, not precomputed
    live L1 labels.  ``predict_embeddings`` then sees the complete query tile.
    The eval support-only cluster teacher stays separate and is fitted once
    per existing support episode.  Live joint-BN labels may differ from that
    teacher during training; no hidden query class enters either path.
    """
    def encode_joint(self, query, embeddings, owners, classes, *, return_trace=False):
        graph = joint_topology(embeddings, owners, classes, query, self.label_seed)
        x = graph['nodes']
        ns = graph['support_count']
        stop = graph['query_start']
        histories = []
        traces = []
        for index, layer in enumerate(self.l1):
            histories.append(x[ns:stop])
            before = x
            # Activation checkpointing would replay BN running-stat updates.
            # The basic reference operator is therefore executed once jointly.
            if return_trace:
                x, terms = layer.forward_with_terms(x, graph['edge_index'], graph['edge_attr'])
            else:
                x = layer(x, graph['edge_index'], graph['edge_attr'])
            raw_output = x
            if index != len(self.l1)-1:
                x = F.gelu(x)
            if return_trace:
                traces.append(dict(layer=index+1, input=before, operator_output=raw_output,
                                   output=x, **terms))
        return dict(query=x[stop:], local_labels=x[ns:stop].reshape(graph['patients'], 2, self.dim),
                    support=x[:ns], histories=histories, graph=graph, trace=traces)

    def encode_support(self, embeddings, owners, classes):
        result = self.encode_joint(embeddings.new_empty(0, self.dim), embeddings, owners, classes)
        return result['histories'], result['local_labels']

    @torch.no_grad()
    def fit_support_clusters(self, embeddings, owners, classes):
        modes = [(module, module.training) for module in self.modules()]
        try:
            result = super().fit_support_clusters(embeddings, owners, classes)
            result['l1_reference_architecture'] = ARCHITECTURE
            result['l1_reference_teacher_scope'] = 'eval_support_only_fixed_running_statistics'
            return result
        finally:
            # Preserve intentionally mixed child-module modes, too. The
            # original helper restores the overall mode with train(bool).
            for module, training in modes:
                module.training = training

    def prepare_support(self, embeddings, owners, classes, cluster_plan=None):
        plan = self.fit_support_clusters(embeddings, owners, classes) if cluster_plan is None else cluster_plan
        if plan.get('l1_reference_architecture') != ARCHITECTURE:
            raise ValueError('Reference L1 requires its own support-only teacher plan; old L1 plan reuse forbidden')
        if (not torch.equal(plan['owners'], owners) or not torch.equal(plan['classes'], classes)
                or not torch.equal(plan['support_embeddings'], embeddings)):
            raise ValueError('Reference teacher plan differs from the bound complete support memory')
        return dict(reference_pending=True, support_embeddings=embeddings, owners=owners,
                    classes=classes, cluster_plan=plan, architecture=ARCHITECTURE)

    def align_labels(self, local_labels, plan):
        """Original project's L2 equations and alignment objective, unchanged."""
        patients = len(local_labels)
        task = torch.arange(patients, device=local_labels.device).repeat_interleave(2)
        forbidden = task[:, None] == task[None, :]
        aligned = local_labels.flatten(0, 1)[None]
        for layer, update in zip(self.l2, self.l2_updates):
            def run(value, layer=layer, update=update):
                return update(value, layer(value, value, value, attn_mask=forbidden, need_weights=False)[0])
            aligned = checkpoint(run, aligned, use_reentrant=False) if self.training and self.checkpoint_support else run(aligned)
        labels = aligned[0].reshape(patients, 2, self.dim)
        return dict(labels=labels, local_labels=local_labels, cluster_plan=plan,
                    alignment_loss=alignment_loss(labels, plan, self.temperature))

    def predict_embeddings(self, query, state, *, return_trace=False):
        if state.get('architecture') != ARCHITECTURE or state.get('reference_pending') is not True:
            raise ValueError('Joint reference support binding context required')
        # Check the immutable plan binding again if a caller retained the context.
        plan = state['cluster_plan']
        if (not torch.equal(plan['owners'], state['owners'])
                or not torch.equal(plan['classes'], state['classes'])
                or not torch.equal(plan['support_embeddings'], state['support_embeddings'])):
            raise ValueError('Reference support context was changed after teacher binding')
        result = self.encode_joint(query, state['support_embeddings'], state['owners'], state['classes'],
                                   return_trace=return_trace)
        aligned = self.align_labels(result['local_labels'], plan)
        logits = prototype_logits(result['query'], aligned['labels'], plan, self.temperature)
        output = dict(logits=logits, ranking_score=logits.softmax(-1)[:, 1],
                      alignment_loss=aligned['alignment_loss'], alignment_loss_weight=self.alignment_loss_weight)
        if return_trace:
            start = result['graph']['query_start']
            output['query_stages'] = [row['output'][start:] for row in result['trace']]
        return output


def clone_reference(net):
    """Return ``(candidate, transfer_metadata)``; source model and RNG untouched.

    The retained compatible projection tensors initialize a new model.  This
    transfers neither optimizer moments nor old LayerNorm/FFN parameters.
    New q/k/v and edge biases are zero; BN starts with identity affine, mean=0,
    variance=1 and counter=0.  This construction is an opt-in reference control,
    not identical numerical equations or an old-checkpoint exact resume.
    """
    if (not isinstance(net, PromptGraphModel) or isinstance(net, ReferencePromptGraphModel)
            or net.dim != 128 or len(net.l1) != 2
            or any(type(layer) is not RelationLayer or layer.heads != 4 for layer in net.l1)):
        raise ValueError('Unmodified project 128D, four-head, two-layer L1 required')
    identity = source_identity()
    caller_rng = rng_state()
    original_hash = _state_hash(net.state_dict())
    unchanged_before = _partition_hash(net, ('local.', 'l2.', 'l2_updates.', 'label_seed'))
    transfers = []
    try:
        candidate = copy.deepcopy(net)
        candidate.__class__ = ReferencePromptGraphModel
        replacement = []
        for index, old in enumerate(net.l1):
            reference = next(old.parameters())
            layer = ReferenceRelationLayer(net.dim, old.heads, old.update.drop.p).to(reference.device, dtype=reference.dtype)
            with torch.no_grad():
                layer.mlp_kqv.weight.copy_(torch.cat((old.q.weight, old.k.weight, old.v.weight), dim=0))
                layer.mlp_kqv.bias.zero_()
                layer.lin_edge.weight.copy_(old.edge.weight)
                layer.lin_edge.bias.zero_()
                width = old.width
                layer.att_mlp[0].weight.copy_(torch.cat((old.attn[0].weight[:, width:2*width],
                                                       old.attn[0].weight[:, :width],
                                                       old.attn[0].weight[:, 2*width:]), dim=1))
                layer.att_mlp[0].bias.copy_(old.attn[0].bias)
                layer.att_mlp[2].load_state_dict(old.attn[2].state_dict(), strict=True)
                layer.out_proj.load_state_dict(old.update.out.state_dict(), strict=True)
            layer.train(old.training)
            replacement.append(layer)
            prefix = f'l1.{index}.'
            transfers.append(dict(layer=index+1,
                copied={prefix+'mlp_kqv.weight': [prefix+'q.weight', prefix+'k.weight', prefix+'v.weight'],
                        prefix+'lin_edge.weight': prefix+'edge.weight',
                        prefix+'att_mlp.0.weight': prefix+'attn.0.weight [k,q,e] block permutation',
                        prefix+'att_mlp.0.bias': prefix+'attn.0.bias',
                        prefix+'att_mlp.2.weight': prefix+'attn.2.weight',
                        prefix+'att_mlp.2.bias': prefix+'attn.2.bias',
                        prefix+'out_proj.weight': prefix+'update.out.weight',
                        prefix+'out_proj.bias': prefix+'update.out.bias'},
                new_zero_bias=[prefix+'mlp_kqv.bias', prefix+'lin_edge.bias'],
                new_batch_norm=dict(affine_weight=1., affine_bias=0., running_mean=0., running_var=1.,
                                    num_batches_tracked=0, momentum=layer.bn.momentum, eps=layer.bn.eps),
                removed_parameters=[prefix+name for name, _ in old.named_parameters()
                                    if name.startswith(('update.norm.', 'update.ff.', 'update.final.'))]))
        candidate.l1 = nn.ModuleList(replacement)
        unchanged_after = _partition_hash(candidate, ('local.', 'l2.', 'l2_updates.', 'label_seed'))
        if unchanged_before != unchanged_after or _state_hash(net.state_dict()) != original_hash:
            raise AssertionError('Reference construction changed L0/L2/label seed or original model state')
        metadata = dict(architecture=ARCHITECTURE, official_source=identity, debug=True,
            production_default_changed=False, production_checkpoint_created=False, exact_resume=False,
            layers=2, hidden_dimension=128, heads=4, inter_layer_activation='GELU',
            relation_codes=dict(T=[0, 1], F=[0, -1], U=[1, 0], self_loop=[0, 0]),
            original_state_sha256=original_hash,
            unchanged_l0_l2_label_seed_sha256_before=unchanged_before,
            unchanged_l0_l2_label_seed_sha256_after=unchanged_after,
            original_parameters=sum(value.numel() for value in net.parameters()),
            candidate_parameters=sum(value.numel() for value in candidate.parameters()), transfers=transfers,
            batch_norm_contract=dict(train='one_joint_support_labels_query_graph',
                eval='fixed_running_statistics_query_chunks', fresh_running_statistics=True,
                query_ground_truth='not_an_input', query_normalization_influence='train_only',
                l1_activation_checkpointing=False),
            unchanged_equations=['L0', 'L2', 'cluster_assignment', 'prototype_logits', 'loss', 'temperature'],
            altered_equations=['biased_qkv_and_edge', 'k_scaling', 'relation_coding', 'ReLU_attention',
                'value_without_edge', 'per_edge_output_projection', 'attention_dropout',
                'self_loop_attention', 'joint_batch_norm_without_FFN', 'inter_layer_GELU'])
        candidate.reference_transfer = metadata
        return candidate, metadata
    finally:
        restore_rng(caller_rng)
