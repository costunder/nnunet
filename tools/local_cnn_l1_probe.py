"""Read-only query L1 message/FFN and complete-attention diagnostics.

Every query attends to the original prepared support histories. A diagnostic
scale multiplies only the projected query message before its residual addition;
attention edges, support/L2 states, layer count, FFNs and norms are unchanged.
Scale 1 must match the production prediction. Optional FFN scales intervene only
on the second query layer's FF residual; messages and support execution remain
unchanged. These fixed-weight counterfactuals are not production settings.
"""
import hashlib
import inspect
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch_geometric.utils import softmax

from hiercp_v222.clustering import prototype_logits
from hiercp_v222.model import RelationLayer


def candidate_signal(x):
    """Descriptive common/centered signal energies; no biological interpretation."""
    x = x.detach().float()
    if x.ndim != 2 or not len(x) or not bool(torch.isfinite(x).all()):
        raise ValueError('Finite nonempty candidate feature matrix required')
    mean = x.mean(0)
    centered = (x-mean).square().sum(1).mean()
    common = mean.square().sum()
    total = x.square().sum(1).mean()
    z = F.normalize(x, dim=1)
    normal_centered = (z-z.mean(0)).square().sum(1).mean()
    return dict(centered_energy=float(centered), normalized_centered_energy=float(normal_centered),
                common_energy=float(common), total_energy=float(total),
                common_energy_fraction=float(common/total) if total > 0 else None,
                mean_norm=float(x.norm(dim=1).mean()), common_norm=float(mean.norm()),
                finite=True)


def _mean_norm_ratio(left, right):
    numerator = left.float().norm(dim=1).mean()
    denominator = right.float().norm(dim=1).mean()
    return float(numerator/denominator) if denominator > 0 else None


def _mean_cosine(left, right):
    left, right = left.float(), right.float()
    valid = (left.norm(dim=1) > 0) & (right.norm(dim=1) > 0)
    return dict(mean=float(F.cosine_similarity(left[valid], right[valid], dim=1).mean())
                if bool(valid.any()) else None, nonzero_rows=int(valid.sum()))


def _relation_stages(layer, source, query, scale, *, ff_scale=1.0, include_attention=False):
    """Exact RelationLayer.messages equations, with explicit projected scaling."""
    src = torch.arange(len(source), device=query.device).repeat(len(query))
    dst = torch.arange(len(query), device=query.device).repeat_interleave(len(source))
    # The production query has U edges with two zero edge features.
    edge_features = query.new_zeros(len(src), 2)
    q = layer.q(query).reshape(-1, layer.heads, layer.width)
    k = layer.k(source).reshape(-1, layer.heads, layer.width)
    v = layer.v(source).reshape(-1, layer.heads, layer.width)
    e = layer.edge(edge_features.to(source.dtype)).reshape(-1, layer.heads, layer.width)
    logits = layer.attn(torch.cat((q[dst], k[src], F.silu(e)), -1)).squeeze(-1)
    weights = softmax(logits.float(), dst, num_nodes=len(query)).to(v.dtype)
    message = (v[src]+e)*weights[..., None]
    aggregate = message.new_zeros(len(query), layer.heads, layer.width)
    aggregate.index_add_(0, dst, message)
    aggregate = aggregate.flatten(1)
    projected = layer.update.out(aggregate)
    # Diagnostics require eval mode, but use the original dropout modules too.
    scaled = layer.update.drop(projected)*scale
    added = query+scaled
    normalized = layer.update.norm(added)
    feedforward = layer.update.ff(normalized)
    second_added = normalized+layer.update.drop(feedforward)*ff_scale
    final = layer.update.final(second_added)
    result = dict(input=query, raw_message=aggregate, projected_message=projected,
                scaled_message=scaled, residual_add=added, residual_norm=normalized,
                ff_output=feedforward, second_add=second_added, final_norm=final)
    if include_attention:
        # Ordering exactly matches the complete U edges constructed above.
        result['_attention_weights'] = weights.reshape(len(query), len(source), layer.heads)
    return result


def _attention_signal(weights, physical_batch):
    """Exact all-candidate statistics; original batches bound pairwise workspace.

    Weights are [candidate, support label, head]. Every distinct candidate pair
    contributes to the JS mean. This computes no sampled or capped pair subset.
    Cosine uses the equivalent complete-pair sum identity, avoiding N x N storage.
    """
    if weights.ndim != 3 or not weights.shape[0] or not weights.shape[1]:
        raise ValueError('Nonempty complete candidate/source/head attention required')
    if not isinstance(physical_batch, int) or isinstance(physical_batch, bool) or physical_batch < 1:
        raise ValueError('Original positive physical query batch is required')
    # Diagnostic distributions can differ only at ~1e-7 in stored FP32 weights;
    # FP64 statistics avoid entropy subtraction rounding those JS values away.
    # This changes descriptive statistics only, never attention execution.
    values = weights.detach().to(torch.float64).permute(0, 2, 1)
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError('Finite nonnegative attention probabilities required')
    torch.testing.assert_close(values.sum(-1), torch.ones_like(values[..., 0]), atol=1e-4, rtol=1e-4)
    n, heads, sources = values.shape
    variance = values.var(0, unbiased=False)
    normalized = F.normalize(values, dim=-1)
    cosine_sum = normalized.sum(0).square().sum(-1)-normalized.square().sum((0, 2))
    entropy = -(torch.where(values > 0, values*values.clamp_min(torch.finfo(values.dtype).tiny).log(), 0)).sum(-1)
    js_sum = values.new_zeros(heads)
    pair_count = 0
    tiny = torch.finfo(values.dtype).tiny
    for left_start in range(0, n, physical_batch):
        left = values[left_start:left_start+physical_batch]
        for right_start in range(left_start, n, physical_batch):
            right = values[right_start:right_start+physical_batch]
            mean = (left[:, None]+right[None])/2
            # JS = H((p+q)/2) - (H(p)+H(q))/2; zero terms use 0 log 0 = 0.
            mixture_entropy = -(torch.where(mean > 0, mean*mean.clamp_min(tiny).log(), 0)).sum(-1)
            js = mixture_entropy-(entropy[left_start:left_start+len(left), None]
                                   +entropy[None, right_start:right_start+len(right)])/2
            if left_start == right_start:
                valid = torch.ones(len(left), len(right), device=values.device, dtype=torch.bool).triu(1)
                js_sum += js[valid].sum(0)
                pair_count += len(left)*(len(left)-1)//2
            else:
                js_sum += js.sum((0, 1))
                pair_count += len(left)*len(right)
    expected_pairs = n*(n-1)//2
    if pair_count != expected_pairs:
        raise AssertionError('Attention statistics did not cover every distinct candidate pair')
    reports = []
    for head in range(heads):
        per_source = variance[head]
        # Only summaries leave the GPU. The complete probabilities are not serialized.
        reports.append(dict(head=head, variance_over_candidates_per_support_label=per_source.tolist(),
            candidate_weight_variance_mean=float(per_source.mean()),
            candidate_weight_variance_min=float(per_source.min()),
            candidate_weight_variance_max=float(per_source.max()),
            mean_pairwise_cosine=float((cosine_sum[head]/(n*(n-1))).clamp(-1, 1)) if n > 1 else None,
            mean_pairwise_jensen_shannon=float((js_sum[head]/pair_count).clamp_min(0)) if pair_count else None,
            entropy_mean=float(entropy[:, head].mean()), entropy_std=float(entropy[:, head].std(unbiased=False)),
            entropy_min=float(entropy[:, head].min()), entropy_max=float(entropy[:, head].max())))
    return dict(candidates=n, support_labels=sources, heads=heads, distinct_candidate_pairs=pair_count,
        physical_batch=physical_batch, all_candidates_and_support_labels_included=True,
        support_label_order='Prepared history order: patient-major, class 0 then class 1',
        entropy_and_js_unit='natural-log nats',
        interpretation='Source-scoring sensitivity on fixed weights; repeated seeds alone are not a coding error',
        per_head=reports)


def _explicit_scales(values, kind):
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f'Explicit finite nonnegative {kind} scales required')
    if len(set(values)) != len(values) or 1.0 not in values:
        raise ValueError(f'Distinct explicit {kind} scales including production scale 1 required')
    return values


def _score_signal(logits, truth):
    score = logits[:, 1].float()-logits[:, 0].float()
    result = dict(score_std=float(score.std(unbiased=False)), score_min=float(score.min()),
                  score_max=float(score.max()))
    if truth is not None:
        if truth.shape != score.shape or truth.device != score.device or truth.dtype != torch.long:
            raise ValueError('Query truth must be matching int64 IDs on the query device')
        if not bool(((truth == 0) | (truth == 1)).all()) or len(truth.unique()) != 2:
            raise ValueError('Both observation classes required for score diagnosis')
        delta = score[truth == 1, None]-score[None, truth == 0]
        result.update(mean_positive_minus_unobserved=float(delta.mean()),
                      pair_win_rate=float((delta > 0).float().mean()),
                      exact_tie_rate=float((delta == 0).float().mean()),
                      mean_pairwise_loss=float(F.softplus(-delta).mean()),
                      comparisons=delta.numel())
    return result


def shared_seed_evidence(net, state):
    """Describe seed source and prepared history without treating repeats as a bug."""
    first = state['histories'][0]
    if len(first) % 2:
        raise ValueError('Prepared support histories must preserve two labels per patient')
    repeated = net.label_seed.detach().repeat(len(first)//2, 1)
    source_path = Path(inspect.getfile(RelationLayer))
    return dict(source_module='hiercp_v222.model.RelationLayer / Residual',
                source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                seed_source='trained shared label_seed parameter from the loaded model snapshot',
                label_seed_is_parameter=isinstance(net.label_seed, torch.nn.Parameter),
                label_seed_requires_grad=net.label_seed.requires_grad,
                first_history_labels=len(first), support_patients=len(first)//2,
                first_history_matches_shared_seed=bool(torch.equal(first, repeated)),
                first_history_seed_max_abs_difference=float((first-repeated).abs().max()),
                shared_seed_signal=candidate_signal(first),
                scope='Prepared support histories are fixed across all query-only scale probes')


def _trace_layers(net, embeddings, state, batch, message_scale, *, ff_scale=1.0, ff_layers=(), attention=False):
    query = embeddings
    layers, attention_reports = [], []
    for level, (layer, source) in enumerate(zip(net.l1, state['histories']), start=1):
        current_ff_scale = ff_scale if level in ff_layers else 1.0
        chunks = [_relation_stages(layer, source, value, message_scale,
                  ff_scale=current_ff_scale, include_attention=attention) for value in query.split(batch)]
        if attention:
            weights = torch.cat([part.pop('_attention_weights') for part in chunks])
            attention_reports.append(dict(layer=level, **_attention_signal(weights, batch)))
        stage_values = {key: torch.cat([part[key] for part in chunks]) for key in chunks[0]}
        signals = {key: candidate_signal(value) for key, value in stage_values.items()}
        initial_centered = signals['input']['centered_energy']
        for signal in signals.values():
            signal['centered_energy_relative_to_layer_input'] = (
                signal['centered_energy']/initial_centered if initial_centered > 0 else None)
        centered_residual = stage_values['residual_norm']-stage_values['residual_norm'].mean(0)
        centered_ff = stage_values['ff_output']-stage_values['ff_output'].mean(0)
        layers.append(dict(layer=level, support_labels=len(source),
            incoming_edges_per_query=len(source), heads=layer.heads, ff_scale=current_ff_scale,
            stages=signals,
            projected_message_to_input_mean_norm=_mean_norm_ratio(
                stage_values['projected_message'], stage_values['input']),
            scaled_message_to_input_mean_norm=_mean_norm_ratio(
                stage_values['scaled_message'], stage_values['input']),
            projected_message_input_cosine=_mean_cosine(
                stage_values['projected_message'], stage_values['input']),
            ff_to_normalized_residual_mean_norm=_mean_norm_ratio(
                stage_values['ff_output'], stage_values['residual_norm']),
            centered_ff_residual_cosine=_mean_cosine(centered_ff, centered_residual)))
        query = stage_values['final_norm']
    return query, layers, attention_reports


@torch.no_grad()
def probe_l1(net, embeddings, state, batch, scales, truth=None, *, ff_scales=None):
    """JSON-ready substage traces and explicit query-only counterfactuals.

    Scales are required from the caller and must contain 1 for production parity.
    Optional FF scales require scale 1 and act only on query layer 2 with message
    scale 1. No support preparation, optimizer, mutation, or checkpoint write occurs.
    The same physical query batch is used by the trace and production reference.
    """
    if net.training or any(layer.training for layer in net.l1):
        raise ValueError('Read-only L1 diagnosis requires eval mode')
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError('Original positive physical query batch is required')
    if embeddings.ndim != 2 or not len(embeddings) or embeddings.shape[1] != net.dim:
        raise ValueError('Original candidate embedding matrix required')
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError('Nonfinite query embeddings')
    scales = _explicit_scales(scales, 'message')
    if ff_scales is not None:
        ff_scales = _explicit_scales(ff_scales, 'FF')
        if len(net.l1) != 2:
            raise ValueError('Layer-2 FF diagnostic requires the original two L1 layers')
    if len(net.l1) != len(state['histories']) or not net.l1:
        raise ValueError('Prepared support history/layer count differs')
    reference = torch.cat([net.predict_embeddings(value, state)['logits']
                           for value in embeddings.split(batch)])
    reports, attention_report = [], None
    for scale in scales:
        query, layers, attention_layers = _trace_layers(net, embeddings, state, batch, scale,
                                                       attention=scale == 1.0)
        if scale == 1.0:
            attention_report = attention_layers
        logits = prototype_logits(query, state['labels'], state['cluster_plan'], net.temperature)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Nonfinite diagnostic logits')
        if scale == 1.0:
            torch.testing.assert_close(logits, reference, atol=2e-6, rtol=2e-5)
        reports.append(dict(message_scale=scale, layers=layers, score=_score_signal(logits, truth),
                            production_logit_max_abs_difference=float((logits-reference).abs().max())))
    ff_reports = []
    if ff_scales is not None:
        for scale in ff_scales:
            query, layers, _ = _trace_layers(net, embeddings, state, batch, 1.0,
                                             ff_scale=scale, ff_layers=(2,))
            logits = prototype_logits(query, state['labels'], state['cluster_plan'], net.temperature)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError('Nonfinite diagnostic FF logits')
            if scale == 1.0:
                torch.testing.assert_close(logits, reference, atol=2e-6, rtol=2e-5)
            ff_reports.append(dict(ff_scale=scale, target_layers=[2], message_scale=1.0,
                layers=layers, score=_score_signal(logits, truth),
                production_logit_max_abs_difference=float((logits-reference).abs().max())))
    result = dict(diagnostic_only=True, production_scale=1.0,
                production_scale_parity_passed=True, physical_batch=batch,
                candidates=len(embeddings), layers=len(net.l1),
                intervention='Projected query message scaled before residual addition; support histories and L2 fixed',
                dropout='eval mode; production dropout modules retained',
                interpretation='Counterfactual execution on fixed weights; no claim of trained accuracy improvement',
                shared_label_seed=shared_seed_evidence(net, state), message_scale_sweep=reports,
                attention_sensitivity=dict(diagnostic_only=True, message_scale=1.0, ff_scale=1.0,
                    query_embeddings_and_support_fixed=True, layers=attention_report))
    if ff_scales is not None:
        result['query_ff_scale_sweep'] = dict(diagnostic_only=True, target_layers=[2],
            message_scale=1.0, support_ff_and_l2_unchanged=True, production_ff_scale_parity_passed=True,
            intervention='Only layer-2 query FF output scaled before its second residual addition',
            interpretation='Directional spread and ranking ordering are separate outcomes; no production patch implied',
            reports=ff_reports)
    return result
