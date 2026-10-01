"""Matched short native-CT comparison of legacy and reference-equation L1.

Explicit DEBUG only. No production checkpoint, optimizer-history migration,
training-ready marker, or automatic long training. L0/L2/loss are preserved.
"""
import copy
from collections import Counter
import math
import random
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from hiercp_v222.v1_execution import rng_state, restore_rng
from l0_regions.donor_learning import LiveContext, configuration, forward_loss
from l0_regions.execution_pipeline import gradient_check_batched
from l0_regions.training import hash_state
from tools.local_cnn_interaction_updates import (
    _evaluate, _module_norms, _query, _support, _execution_runtime,
)
from tools.local_cnn_interaction_runtime import support_binding
from tools.v22_candidate_order import record_key
from tools.v22_rank_objective import ranking_metrics


def _head(model, embeddings, support, physical_batch):
    from tools.diagnose_local_cnn_learning import trace_head, spread
    if not hasattr(model, 'encode_joint'):
        return trace_head(model, embeddings, support, physical_batch)
    state = model.prepare_support(*support)
    logits, stages = [], [[], []]
    for q in embeddings.split(physical_batch):
        output = model.predict_embeddings(q, state, return_trace=True)
        logits.append(output['logits'])
        for i, value in enumerate(output['query_stages']):
            stages[i].append(value)
    logits = torch.cat(logits)
    report = dict(stages=[dict(stage='L0', **spread(embeddings))] + [
        dict(stage=f'L1_{i+1}', **spread(torch.cat(values))) for i, values in enumerate(stages)],
        mode='eval; reference joint graph with fixed BatchNorm running statistics',
        support_recomputed_each_query_chunk=True)
    return logits[:, 1]-logits[:, 0], report


def evaluator(cases, memory, train_rows, physical_batch, budget, original_local_hash):
    """Retain every candidate of each explicitly selected train/validation case."""
    from tools.diagnose_local_cnn_learning import score_summary
    if not cases or {case['split'] for case in cases} != {'train', 'validation'}:
        raise ValueError('Matched explicit train and validation case sets required')

    @torch.no_grad()
    def evaluate(model):
        modes = [(module, module.training) for module in model.modules()]
        parts = {split: [] for split in ('train', 'validation')}
        try:
            model.eval()
            reuse = hash_state(model.local.state_dict()) == original_local_hash
            for case in cases:
                rows, ids, ds = case['rows'], case['indices'], case['reader'].ds
                expected = [i for i, row in enumerate(ds.rows) if row['case_id'] == rows[0]['case_id']]
                if ids != expected or rows != [ds.rows[i] for i in ids]:
                    raise ValueError('Selected case lost or substituted native candidates')
                if not torch.equal(case['truth'].cpu(), torch.tensor([row['target'] for row in rows])):
                    raise ValueError('Evaluation truth and original observation binding differ')
                if reuse:
                    embeddings = case['embeddings']
                else:
                    vectors = []
                    for start in range(0, len(ids), physical_batch):
                        budget.check()
                        query = case['reader'].get(ids[start:start+physical_batch]).to('cuda')
                        vectors.append(model.local(query).detach())
                        del query
                    embeddings = torch.cat(vectors)
                support, records, _ = support_binding(memory, train_rows, rows[0]['patient_group'])
                scores, trace = _head(model, embeddings, support, physical_batch)
                truth = case['truth']
                keys = [record_key(row) for row in rows]
                names = [rows[0]['case_id']]*len(rows)
                metrics, detail = ranking_metrics(scores.cpu(), truth.cpu(), names, candidate_keys=keys)
                parts[case['split']].append(dict(case_id=names[0], records=len(rows),
                    all_case_candidates_retained=True, metrics=metrics,
                    score=score_summary(scores, truth), trace=trace,
                    observed_ranks=detail[0]['observed_ranks'],
                    support_record_ids=records, cached_query_basis_reused=reuse,
                    _scores=scores.cpu(), _truth=truth.cpu(), _names=names, _keys=keys))
                budget.check()
            result = {}
            for split, rows in parts.items():
                metrics, _ = ranking_metrics(torch.cat([r.pop('_scores') for r in rows]),
                    torch.cat([r.pop('_truth') for r in rows]),
                    [v for r in rows for v in r.pop('_names')],
                    candidate_keys=[v for r in rows for v in r.pop('_keys')])
                total = sum(r['score']['comparisons'] for r in rows)
                if total <= 0:
                    raise ValueError('Explicit selected cases require observed/unobserved comparisons')
                result[split] = dict(metrics=metrics, cases=rows,
                    pair_win_rate=sum(r['score']['pair_win_rate']*r['score']['comparisons'] for r in rows)/total)
            result.update(scope='all candidates of explicitly selected cases; not full validation or CP efficacy',
                support_scope='full eligible saved detached epoch L0; own L1/L2 and teacher plan',
                original_L0_basis_reused=reuse)
            return result
        finally:
            for module, mode in modes:
                module.training = mode
    return evaluate


def probe_updates(net, *, steps, train_tiles, batch_provider, support_provider,
                  loss_context, physical_batch, budget, training, seed,
                  evaluation_provider, progress=True):
    """Fresh matched AdamW, same CT prefix and complete-cohort loss coefficients."""
    from tools.local_cnn_reference_l1 import clone_reference
    if type(steps) is not int or not 0 < steps <= len(train_tiles):
        raise ValueError('Explicit positive DEBUG prefix within full schedule required')
    if not isinstance(loss_context, LiveContext) or loss_context.audit['physical_batch'] != physical_batch:
        raise ValueError('Original full-cohort live-ranking context and physical batch required')
    if Counter(tuple(sorted(v)) for v in train_tiles) != Counter(tuple(sorted(v)) for v in loss_context.order):
        raise ValueError('Full ranking schedule changed; no observations/pairs may be dropped')
    for key in ('lr', 'weight_decay', 'grad_clip'):
        if not math.isfinite(training[key]) or training[key] < 0 or (key != 'weight_decay' and not training[key]):
            raise ValueError('Invalid original optimizer settings')
    if type(training['fused_optimizer']) is not bool:
        raise ValueError('Original fused optimizer policy required')
    saved_rng = rng_state()
    saved_state = hash_state(net.state_dict())
    saved_grad = hash_state({n: p.grad for n, p in net.named_parameters()})
    saved_modes = [(m, m.training) for m in net.modules()]
    saved_l0 = hash_state(net.local.state_dict())
    saved_l2 = hash_state({n: v for n, v in net.state_dict().items() if n.startswith(('l2.', 'l2_updates.'))})
    results, input_hashes = [], []
    device = next(net.parameters()).device
    try:
        with torch.autocast(device.type, enabled=False):
            for branch in ('legacy', 'reference'):
                budget.check()
                clone, contract = ((copy.deepcopy(net), dict(architecture='unchanged_legacy_L1'))
                                   if branch == 'legacy' else clone_reference(net))
                if hash_state(clone.local.state_dict()) != saved_l0 or hash_state({
                    n: v for n, v in clone.state_dict().items() if n.startswith(('l2.', 'l2_updates.'))}) != saved_l2:
                    raise AssertionError('Reference initialization changed L0/L2 weights')
                named = [(n, p) for n, p in clone.named_parameters() if p.requires_grad]
                if not named or any(p.dtype != torch.float32 or p.device != device for _, p in named):
                    raise ValueError('Single-device FP32 native comparison required')
                initial = [p.detach().clone() for _, p in named]
                optimizer = torch.optim.AdamW([p for _, p in named], lr=training['lr'],
                    weight_decay=training['weight_decay'], fused=training['fused_optimizer'])
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                before = _evaluate(evaluation_provider, clone)
                clone.train()
                reports, last_group, plan = [], None, None
                bar = tqdm(range(steps), desc=f'CLONED {branch} L1', disable=not progress)
                for offset in bar:
                    ids = train_tiles[offset]
                    budget.check()
                    query = _query(batch_provider(ids), ids, loss_context, device)
                    support, records, group = _support(support_provider(ids), ids, loss_context, device)
                    native = {k: v for k, v in vars(query).items() if k != '_verified_signature'}
                    digest = hash_state(dict(query=native, support=support, records=records, group=group))
                    if branch == 'legacy': input_hashes.append(digest)
                    elif digest != input_hashes[offset]:
                        raise ValueError('Legacy/reference original CT/support inputs differ')
                    if group != last_group:
                        plan = clone.fit_support_clusters(*support)
                        last_group = group
                    targets = torch.tensor([loss_context.rows[i]['target'] for i in ids], device=device)
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize(device); started = time.perf_counter()
                    loss, terms = forward_loss(clone, query, support, plan, targets, None,
                        loss_context, configuration(), indices=ids)
                    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite full original objective')
                    loss.backward(); gradient_check_batched(clone)
                    norms = _module_norms(named, [p.grad for _, p in named])
                    grad = torch.nn.utils.clip_grad_norm_([p for _, p in named], training['grad_clip'], error_if_nonfinite=True)
                    optimizer.step(); torch.cuda.synchronize(device)
                    seconds = time.perf_counter()-started
                    budget.check()
                    if hash_state(dict(query=native, support=support, records=records, group=group)) != digest:
                        raise AssertionError('Native query/support mutated during comparison')
                    reports.append(dict(step=offset+1, indices=ids, physical_batch=len(ids),
                        input_content_sha256=digest, support_record_ids=records,
                        loss=float(loss.detach()), terms={k: float(v.detach()) for k, v in terms.items()},
                        module_gradient_norms=norms, gradient_norm_before_clip=float(grad),
                        synchronized_update_seconds=seconds))
                    bar.set_postfix(loss=f'{float(loss.detach()):.4g}', seconds=f'{seconds:.3g}', batch=len(ids))
                    del query, support, loss, terms
                bar.close()
                after = _evaluate(evaluation_provider, clone)
                changes = _module_norms(named, [p.detach()-old for (_, p), old in zip(named, initial)])
                if any(value is not None and not math.isfinite(value) for value in changes.values()):
                    raise FloatingPointError('Nonfinite cloned optimizer parameter delta')
                if not changes['CNN'] or not changes['L1'] or not changes['L2']:
                    raise AssertionError('Required CNN/L1/L2 module was not updated')
                results.append(dict(branch=branch, architecture_contract=contract,
                    optimizer='fresh AdamW; no saved moments reused', precision='FP32',
                    initial_parameters=sum(p.numel() for _, p in named),
                    before=before, after=after, updates=reports, parameter_delta_norms=changes))
                del clone, optimizer, named, initial
    finally:
        restore_rng(saved_rng)
        if hash_state(net.state_dict()) != saved_state or hash_state({n: p.grad for n, p in net.named_parameters()}) != saved_grad:
            raise AssertionError('Original model/gradient buffers changed')
        if any(m.training != mode for m, mode in saved_modes) or hash_state(rng_state()) != hash_state(saved_rng):
            raise AssertionError('Original module modes/caller RNG changed')
    return dict(diagnostic_only=True, branches=results, steps_per_branch=steps,
        physical_batch=physical_batch, full_objective=configuration(),
        execution=_execution_runtime(device), original_state_and_rng_preserved=True,
        timing_scope='synchronized forward/full loss/backward/gradient diagnostics/clip/AdamW only; excludes CT loading, transfer, input hashing, teacher preparation, evaluation and checkpoint IO; not epoch timing',
        same_native_inputs=True, original_L0_L2_initial_weights_preserved=True,
        exact_resume=False, production_optimizer_updates=0, production_checkpoint_written=False,
        production_ready=False, full_training=False, full_evaluation=False)
