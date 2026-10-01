"""Read-only query L1 substage and explicit message-scale diagnostics.

Every query attends to the original prepared support histories. A diagnostic
scale multiplies only the projected query message before its residual addition;
attention edges, support/L2 states, layer count, FFNs and norms are unchanged.
Scale 1 must match the production prediction. Scale 0 still executes the query
FFN and both norms, so it is not an L1 deletion or a proposed production setting.
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


def _relation_stages(layer, source, query, scale):
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
    second_added = normalized+layer.update.drop(feedforward)
    final = layer.update.final(second_added)
    return dict(input=query, raw_message=aggregate, projected_message=projected,
                scaled_message=scaled, residual_add=added, residual_norm=normalized,
                ff_output=feedforward, second_add=second_added, final_norm=final)


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


@torch.no_grad()
def probe_l1(net, embeddings, state, batch, scales, truth=None):
    """JSON-ready substage traces and explicit query-only counterfactuals.

    Scales are required from the caller and must contain 1 for production parity.
    No support preparation, optimizer, model mutation, or checkpoint write occurs.
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
    scales = [float(value) for value in scales]
    if not scales or any(not math.isfinite(value) or value < 0 for value in scales):
        raise ValueError('Explicit finite nonnegative message scales required')
    if len(set(scales)) != len(scales) or 1.0 not in scales:
        raise ValueError('Distinct explicit scales including production scale 1 required')
    if len(net.l1) != len(state['histories']) or not net.l1:
        raise ValueError('Prepared support history/layer count differs')
    reference = torch.cat([net.predict_embeddings(value, state)['logits']
                           for value in embeddings.split(batch)])
    reports = []
    for scale in scales:
        query = embeddings
        layers = []
        for level, (layer, source) in enumerate(zip(net.l1, state['histories']), start=1):
            chunks = [_relation_stages(layer, source, value, scale) for value in query.split(batch)]
            stage_values = {key: torch.cat([part[key] for part in chunks]) for key in chunks[0]}
            signals = {key: candidate_signal(value) for key, value in stage_values.items()}
            initial_centered = signals['input']['centered_energy']
            for signal in signals.values():
                signal['centered_energy_relative_to_layer_input'] = (
                    signal['centered_energy']/initial_centered if initial_centered > 0 else None)
            layers.append(dict(layer=level, support_labels=len(source),
                incoming_edges_per_query=len(source), heads=layer.heads,
                stages=signals,
                projected_message_to_input_mean_norm=_mean_norm_ratio(
                    stage_values['projected_message'], stage_values['input']),
                scaled_message_to_input_mean_norm=_mean_norm_ratio(
                    stage_values['scaled_message'], stage_values['input']),
                projected_message_input_cosine=_mean_cosine(
                    stage_values['projected_message'], stage_values['input']),
                ff_to_normalized_residual_mean_norm=_mean_norm_ratio(
                    stage_values['ff_output'], stage_values['residual_norm'])))
            query = stage_values['final_norm']
        logits = prototype_logits(query, state['labels'], state['cluster_plan'], net.temperature)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Nonfinite diagnostic logits')
        if scale == 1.0:
            torch.testing.assert_close(logits, reference, atol=2e-6, rtol=2e-5)
        reports.append(dict(message_scale=scale, layers=layers, score=_score_signal(logits, truth),
                            production_logit_max_abs_difference=float((logits-reference).abs().max())))
    return dict(diagnostic_only=True, production_scale=1.0,
                production_scale_parity_passed=True, physical_batch=batch,
                candidates=len(embeddings), layers=len(net.l1),
                intervention='Projected query message scaled before residual addition; support histories and L2 fixed',
                dropout='eval mode; production dropout modules retained',
                interpretation='Counterfactual execution on fixed weights; no claim of trained accuracy improvement',
                shared_label_seed=shared_seed_evidence(net, state), message_scale_sweep=reports)
