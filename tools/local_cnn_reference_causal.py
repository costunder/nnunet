"""Opt-in DEBUG observations of a real ranking update; never production training.

Every fitted case is evaluated with its complete original candidates. Tile
controls hold the supplied support/teacher fixed and disable dropout in an
isolated clone. These controls do not replace the original train-mode update.
"""
import copy
from contextlib import contextmanager

import torch

from hiercp_v222.v1_execution import rng_state, restore_rng
from l0_regions.training import hash_state
from tools.diagnose_local_cnn_learning import score_summary, spread
from tools.v22_candidate_order import record_key
from tools.v22_rank_objective import ranking_metrics


@contextmanager
def capture_prediction(model):
    """Observe the exact existing forward; restore the instance method on exit."""
    marker = object()
    prior = model.__dict__.get('predict_embeddings', marker)
    original = model.predict_embeddings
    captured = []
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result
    model.predict_embeddings = observe
    try:
        yield captured
    finally:
        if prior is marker:
            del model.predict_embeddings
        else:
            model.predict_embeddings = prior


def tile_scores(output, targets, rows):
    logits = output['logits'].detach().float()
    if (logits.shape != (len(rows), 2) or targets.shape != (len(rows),)
            or not torch.equal(targets, torch.tensor([r['target'] for r in rows], device=targets.device))
            or len({r['case_id'] for r in rows}) != 1
            or not bool(torch.isfinite(logits).all()) or set(targets.tolist()) != {0, 1}):
        raise ValueError('Finite, bound same-case P/U logits required')
    scores = logits[:, 1]-logits[:, 0]
    metrics, detail = ranking_metrics(scores.cpu(), targets.cpu(), [rows[0]['case_id']]*len(rows),
        candidate_keys=[record_key(r) for r in rows])
    return dict(case_id=rows[0]['case_id'], observed=int(targets.sum()),
        unobserved=int((targets==0).sum()), metrics=metrics, score=score_summary(scores, targets),
        observed_ranks=detail[0]['observed_ranks'],
        scope='this original physical tile only; not the complete case or CP efficacy')


@torch.no_grad()
def frozen_tile(model, query, support, plan, targets, context, indices, budget):
    """Evaluate the same input/teacher pre/post without advancing BN or RNG."""
    from l0_regions.donor_learning import configuration, forward_loss
    rng = rng_state()
    old_model = hash_state(model.state_dict())
    old_input = hash_state(dict(query=vars(query), support=support, plan=plan))
    clone = None
    try:
        budget.check()
        clone = copy.deepcopy(model).eval()
        with capture_prediction(clone) as captured:
            loss, terms = forward_loss(clone, query, support, plan, targets, None,
                context, configuration(), indices=indices)
        if len(captured) != 1:
            raise AssertionError('Exactly one original loss forward required')
        report = tile_scores(captured[0], targets, [context.rows[i] for i in indices])
        report.update(full_loss=float(loss), weighted_terms={k:float(v) for k,v in terms.items()},
            mode='eval BN + dropout off', fixed_teacher_sha256=hash_state(plan),
            support_sha256=hash_state(support), teacher_refitted=False)
        return report
    finally:
        del clone
        restore_rng(rng)
        if (hash_state(model.state_dict()) != old_model
                or hash_state(dict(query=vars(query), support=support, plan=plan)) != old_input):
            raise AssertionError('Frozen tile control mutated model/input/teacher')


@torch.no_grad()
def target_head(model, embeddings, support, physical_batch, truth, rows):
    """Original eval equations with stage P/U statistics; GT is metric-only."""
    from hiercp_v222.clustering import prototype_logits
    from tools.local_cnn_reference_target_signal import class_separation
    state = model.prepare_support(*support)
    stages = [embeddings]
    if hasattr(model, 'encode_joint'):
        logits, layers = [], [[] for _ in model.l1]
        for q in embeddings.split(physical_batch):
            output = model.predict_embeddings(q, state, return_trace=True)
            logits.append(output['logits'])
            for i, value in enumerate(output['query_stages']):
                layers[i].append(value)
        logits = torch.cat(logits)
        stages.extend(torch.cat(values) for values in layers)
    else:
        q = embeddings
        for layer, labels in zip(model.l1, state['histories']):
            parts = []
            for value in q.split(physical_batch):
                src = torch.arange(len(labels), device=q.device).repeat(len(value))
                dst = torch.arange(len(value), device=q.device).repeat_interleave(len(labels))
                parts.append(layer.messages(labels, value, src, dst, value.new_zeros(len(src), 2)))
            q = torch.cat(parts)
            stages.append(q)
        logits = prototype_logits(q, state['labels'], state['cluster_plan'], model.temperature)
    trace = dict(stages=[dict(stage='L0' if i==0 else f'L1_{i}', **spread(value),
        target_signal=class_separation(value, truth, rows=rows)) for i,value in enumerate(stages)],
        mode='eval; original scoring equations; target labels used only for metrics',
        cluster_audit=state['cluster_plan']['audit'])
    return logits[:, 1]-logits[:, 0], trace


def fitted_case_ids(selection):
    """Stable union of actual updated cases, never prediction-based selection."""
    return list(dict.fromkeys(tile['case_id'] for tile in selection['selected_tiles']))


from tools.local_cnn_causal_summary import format_causal_summary
