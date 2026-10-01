"""Opt-in L1 additive-plus-dot diagnostics; production equations are untouched.

All old projections, edge-conditioned MLP, value messages and Residual modules
remain in the executed path. The explicit interaction scale is not a learned
parameter. Same weights do not mean an exact production resume for scale != 0.
"""
import copy
import hashlib
import inspect
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch_geometric.utils import softmax

from hiercp_v222.clustering import prototype_logits
from hiercp_v222.model import RelationLayer
from hiercp_v222.v1_execution import rng_state, restore_rng
from l0_regions.training import hash_state
from tools.local_cnn_l1_probe import _attention_signal, candidate_signal, _score_signal


ARCHITECTURE = 'diagnostic_l1_additive_plus_dot_v1'


def _scale(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Explicit finite nonnegative attention interaction scale required')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Explicit finite nonnegative attention interaction scale required')
    return value


def attention_terms(layer, source, destination, src, dst, features, *, scale=None):
    """One shared equation for candidate forward and its query instrumentation.

    Softmax and the added dot score use FP32 like production softmax; resulting
    weights and messages retain the original value dtype and device. Scale zero
    uses the original logits exactly, rather than adding a rounded zero term.
    """
    scale = _scale(getattr(layer, 'interaction_scale', 0.) if scale is None else scale)
    if (source.ndim != 2 or destination.ndim != 2 or source.shape[1] != layer.heads*layer.width
            or destination.shape[1] != source.shape[1] or source.device != destination.device
            or source.dtype != destination.dtype or not len(source) or not len(destination)):
        raise ValueError('Matching nonempty L1 source/destination dimension, dtype and device required')
    if (src.dtype != torch.long or dst.dtype != torch.long or src.shape != dst.shape
            or src.ndim != 1 or not len(src) or features.shape != (len(src), 2)
            or any(value.device != source.device for value in (src, dst, features))):
        raise ValueError('Bound original L1 edge indices and two edge features required')
    if (bool(((src < 0) | (src >= len(source))).any())
            or bool(((dst < 0) | (dst >= len(destination))).any())):
        raise ValueError('L1 edge index outside its bound source/destination')
    if not bool(torch.isfinite(source).all() & torch.isfinite(destination).all() & torch.isfinite(features).all()):
        raise FloatingPointError('Nonfinite bound L1 input; no fallback')
    q = layer.q(destination).reshape(-1, layer.heads, layer.width)
    k = layer.k(source).reshape(-1, layer.heads, layer.width)
    v = layer.v(source).reshape(-1, layer.heads, layer.width)
    e = layer.edge(features.to(source.dtype)).reshape(-1, layer.heads, layer.width)
    legacy = layer.attn(torch.cat((q[dst], k[src], F.silu(e)), -1)).squeeze(-1)
    dot = (q[dst].float()*k[src].float()).sum(-1)/math.sqrt(layer.width)
    combined = legacy.float() if scale == 0 else legacy.float()+scale*dot
    if not bool(torch.isfinite(legacy).all() & torch.isfinite(dot).all() & torch.isfinite(combined).all()):
        raise FloatingPointError('Nonfinite interaction logits; no fallback')
    weights = softmax(combined, dst, num_nodes=len(destination)).to(v.dtype)
    message = (v[src]+e)*weights[..., None]
    aggregate = message.new_zeros(len(destination), layer.heads, layer.width)
    aggregate.index_add_(0, dst, message)
    if not bool(torch.isfinite(aggregate).all() & torch.isfinite(weights).all()):
        raise FloatingPointError('Nonfinite interaction message; no fallback')
    return dict(legacy_logits=legacy, dot_logits=dot, combined_logits=combined,
                weights=weights, aggregate=aggregate.flatten(1), interaction_scale=scale)


class CandidateRelationLayer(RelationLayer):
    """Same modules/parameters, explicit source-specific q-k score interaction."""
    def __init__(self, dim, heads, dropout, *, scale):
        super().__init__(dim, heads, dropout)
        self.interaction_scale = _scale(scale)

    def messages(self, source, destination, src, dst, features):
        terms = attention_terms(self, source, destination, src, dst, features)
        return self.update(destination, terms['aggregate'])


def clone_candidate(net, scale):
    """Deep copy; replace every L1 before any optimizer is constructed.

    Constructor RNG is restored even on failure. Parameter names, shapes,
    strides, dtype, device, requires_grad and module modes match the original.
    L0/L2 are untouched deep copies. No production checkpoint is created.
    """
    scale = _scale(scale)
    caller_rng = rng_state()
    try:
        if not net.l1 or any(type(layer) is not RelationLayer for layer in net.l1):
            raise ValueError('Original unmodified production RelationLayer sequence required')
        candidate = copy.deepcopy(net)
        for index, old in enumerate(net.l1):
            reference = next(old.parameters())
            layer = CandidateRelationLayer(old.heads*old.width, old.heads, old.update.drop.p,
                                           scale=scale).to(reference.device, dtype=reference.dtype)
            layer.load_state_dict(old.state_dict(), strict=True)
            old_named = dict(old.named_parameters())
            with torch.no_grad():
                for name, parameter in layer.named_parameters():
                    original = old_named[name]
                    if parameter.stride() != original.stride():
                        parameter.data = torch.empty_strided(original.shape, original.stride(),
                            device=original.device, dtype=original.dtype).copy_(original)
                    parameter.requires_grad_(original.requires_grad)
            modes = {name: module.training for name, module in old.named_modules()}
            for name, module in layer.named_modules():
                module.training = modes[name]
            candidate.l1[index] = layer
        candidate._diagnostic_architecture = ARCHITECTURE
        candidate._diagnostic_interaction_scale = scale
        old_named, new_named = dict(net.named_parameters()), dict(candidate.named_parameters())
        if old_named.keys() != new_named.keys():
            raise AssertionError('Interaction candidate changed registered parameter names')
        if len({id(p) for p in candidate.parameters()}) != len(new_named):
            raise AssertionError('Interaction candidate has aliased/orphaned registered parameters')
        for name, parameter in new_named.items():
            old = old_named[name]
            if (parameter.shape != old.shape or parameter.dtype != old.dtype or parameter.device != old.device
                    or parameter.stride() != old.stride() or parameter.requires_grad != old.requires_grad
                    or id(parameter) == id(old)):
                raise AssertionError(f'Interaction candidate changed parameter binding/layout: {name}')
        return candidate
    finally:
        restore_rng(caller_rng)
        if hash_state(rng_state()) != hash_state(caller_rng):
            raise AssertionError('Interaction constructor failed to restore caller RNG')


def _logit_magnitude(values):
    # [candidate, source, head], all entries retained in descriptive statistics.
    values = values.detach().float()
    return [dict(head=head, mean_abs=float(values[..., head].abs().mean()),
                 rms=float(values[..., head].square().mean().sqrt()),
                 maximum_abs=float(values[..., head].abs().max()),
                 std=float(values[..., head].std(unbiased=False)))
            for head in range(values.shape[-1])]


@torch.no_grad()
def _trace_query(net, embeddings, state, batch):
    query = embeddings
    layers = []
    for level, (layer, source) in enumerate(zip(net.l1, state['histories']), start=1):
        outputs, weights, legacy, dot, combined, messages = [], [], [], [], [], []
        for value in query.split(batch):
            src = torch.arange(len(source), device=query.device).repeat(len(value))
            dst = torch.arange(len(value), device=query.device).repeat_interleave(len(source))
            terms = attention_terms(layer, source, value, src, dst, value.new_zeros(len(src), 2))
            outputs.append(layer.update(value, terms['aggregate']))
            messages.append(terms['aggregate'])
            shape = (len(value), len(source), layer.heads)
            weights.append(terms['weights'].reshape(shape))
            legacy.append(terms['legacy_logits'].reshape(shape))
            dot.append(terms['dot_logits'].reshape(shape))
            combined.append(terms['combined_logits'].reshape(shape))
        final = torch.cat(outputs)
        layers.append(dict(layer=level, interaction_scale=getattr(layer, 'interaction_scale', 0.),
            support_labels=len(source), incoming_edges_per_query=len(source), heads=layer.heads,
            input_signal=candidate_signal(query), raw_message_signal=candidate_signal(torch.cat(messages)),
            output_signal=candidate_signal(final),
            attention=_attention_signal(torch.cat(weights), batch),
            legacy_mlp_logits=_logit_magnitude(torch.cat(legacy)),
            unscaled_dot_logits=_logit_magnitude(torch.cat(dot)),
            combined_logits=_logit_magnitude(torch.cat(combined))))
        query = final
    return prototype_logits(query, state['labels'], state['cluster_plan'], net.temperature), layers


@torch.no_grad()
def probe_interaction(net, query_embeddings, support_tuple, *, scales, truth, batch,
                      support_record_ids, query_group):
    """Same bound full support for all branches; own teacher plan per equation.

    The caller supplies the already-verified complete eligible support tuple,
    including donor-side exclusion. This helper checks its immutable ordered
    IDs/owner/class/vector binding; it cannot infer a donor ID from 128D vectors.
    """
    if any(module.training for module in net.modules()):
        raise ValueError('Read-only interaction diagnosis requires the whole model in eval mode')
    values = [_scale(value) for value in scales]
    if not values or len(set(values)) != len(values) or 0. not in values:
        raise ValueError('Distinct explicit interaction scales including original scale 0 required')
    if type(batch) is not int or batch < 1:
        raise ValueError('Original positive physical query batch required')
    if not isinstance(support_tuple, (tuple, list)) or len(support_tuple) != 3:
        raise ValueError('Original support embedding/owner/class tuple required')
    support, owners, classes = support_tuple
    if (query_embeddings.ndim != 2 or query_embeddings.shape[1] != net.dim or not len(query_embeddings)
            or support.ndim != 2 or support.shape[1] != net.dim or not len(support)
            or query_embeddings.dtype != support.dtype or query_embeddings.device != support.device
            or owners.dtype != torch.long or classes.dtype != torch.long
            or owners.shape != classes.shape or owners.shape != (len(support),)
            or owners.device != support.device or classes.device != support.device
            or not bool(torch.isfinite(query_embeddings).all() & torch.isfinite(support).all())):
        raise ValueError('Matching finite complete query/support with original owner/class IDs required')
    if (len(support_record_ids) != len(support) or len(set(support_record_ids)) != len(support)
            or not isinstance(query_group, str) or not query_group):
        raise ValueError('Complete unique ordered support record IDs and explicit query group required')
    caller_rng = rng_state()
    model_hash = hash_state(net.state_dict())
    support_hash = hash_state(dict(ids=support_record_ids, vectors=support, owners=owners, classes=classes))
    query_hash = hash_state(dict(embeddings=query_embeddings, truth=truth))
    reports = []
    reference = None
    budget = getattr(net.local, 'resource_budget', None)
    try:
        for name, scale in [('legacy_additive', None)]+[(ARCHITECTURE, value) for value in values]:
            started = time.perf_counter()
            current = net if scale is None else clone_candidate(net, scale)
            if hash_state(current.state_dict()) != model_hash:
                raise AssertionError('Interaction candidate changed original weight values')
            # The same input support is not sufficient to reuse an old plan:
            # candidate L1 changes its local-label teacher and second history.
            state = current.prepare_support(support, owners, classes)
            if not torch.equal(state['cluster_plan']['support_embeddings'], support):
                raise AssertionError('Interaction teacher plan lost original support binding')
            logits, layers = _trace_query(current, query_embeddings, state, batch)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('Nonfinite interaction logits; no branch fallback')
            if scale is None:
                production = torch.cat([net.predict_embeddings(value, state)['logits']
                                        for value in query_embeddings.split(batch)])
                torch.testing.assert_close(logits, production, atol=2e-6, rtol=2e-5)
                reference = logits.detach().clone()
            elif scale == 0.:
                torch.testing.assert_close(logits, reference, atol=2e-6, rtol=2e-5)
            reports.append(dict(branch=name, interaction_scale=scale, score=_score_signal(logits, truth),
                layers=layers, support_records=len(support), support_patients=int(owners.max())+1,
                support_history_and_teacher_refitted=True, support_query_equation_matched=True,
                support_ids_and_embeddings_binding_sha256=support_hash,
                local_labels_signal=candidate_signal(state['local_labels'].flatten(0, 1)),
                aligned_labels_signal=candidate_signal(state['labels'].flatten(0, 1)),
                original_logit_max_abs_difference=float((logits-reference).abs().max()),
                seconds=time.perf_counter()-started,
                timing_scope='clone/identity checks, fresh support teacher/L1/L2, query scoring and descriptive statistics'))
            if budget:
                budget.check()
            del state, logits
            if scale is not None:
                del current
        weights_unchanged = hash_state(net.state_dict()) == model_hash
        support_unchanged = hash_state(dict(ids=support_record_ids, vectors=support, owners=owners, classes=classes)) == support_hash
        query_unchanged = hash_state(dict(embeddings=query_embeddings, truth=truth)) == query_hash
        if not (weights_unchanged and support_unchanged and query_unchanged):
            raise AssertionError('Interaction probe mutated original model or bound data')
        return dict(diagnostic_only=True, architecture=ARCHITECTURE, scales=values,
            formula='edge-conditioned additive MLP + scale * dot(q_destination,k_source)/sqrt(head_width)',
            helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            baseline_source_sha256=hashlib.sha256(Path(inspect.getfile(RelationLayer)).read_bytes()).hexdigest(),
            query_group=query_group, support_record_ids=list(support_record_ids),
            query_candidates=len(query_embeddings), physical_batch=batch,
            all_query_candidates_and_bound_support_retained=True, support_tuple_identical_across_branches=True,
            support_exclusion_scope='Caller supplies complete eligible support; recipient/donor exclusion cannot be inferred from feature vectors',
            scale_zero_production_parity_passed=True, original_weights_unchanged=weights_unchanged,
            support_tuple_unchanged=support_unchanged, query_unchanged=query_unchanged,
            caller_rng_restored=True, L2_equations_and_weights_unchanged=True,
            parameters=sum(p.numel() for p in net.parameters()), new_trainable_parameters=0,
            production_optimizer_updates=0, production_checkpoint_written=False, production_ready=False,
            exact_production_resume_for_nonzero_scale=False, branches=reports,
            interpretation='Fixed-weight source-scoring candidate; attention sensitivity or variance alone is not ranking improvement')
    finally:
        restore_rng(caller_rng)
        if hash_state(rng_state()) != hash_state(caller_rng):
            raise AssertionError('Interaction probe failed to restore caller RNG')
