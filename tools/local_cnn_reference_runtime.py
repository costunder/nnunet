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


SELECTION_POLICIES = ('complete_prefix', 'rankable_full_batch_prefix')


def select_update_tiles(train_tiles, *, steps, loss_context, physical_batch,
                        selection_policy='complete_prefix'):
    """Admit an explicit DEBUG selection using only the complete native schedule.

    Selection never constructs a new tile or changes ``LiveContext``.  In the
    rankable control the first full physical tiles containing both P and U are
    selected in existing schedule order, independently of model outputs.
    """
    if selection_policy not in SELECTION_POLICIES:
        raise ValueError(f'Explicit DEBUG tile selection required; choose from {SELECTION_POLICIES}')
    if type(steps) is not int or steps <= 0:
        raise ValueError('Explicit positive DEBUG update count required')
    if (type(physical_batch) is not int or physical_batch < 2
            or not isinstance(loss_context, LiveContext)
            or loss_context.audit['physical_batch'] != physical_batch):
        raise ValueError('Original full-cohort live-ranking context and physical batch required')
    if not isinstance(train_tiles, list) or len(train_tiles) != loss_context.steps:
        raise ValueError('Complete original ranking schedule required')
    if any(not tile or len(tile) > physical_batch or len(set(tile)) != len(tile)
           or any(type(i) is not int or i < 0 or i >= len(loss_context.rows) for i in tile)
           for tile in train_tiles):
        raise ValueError('Original unique in-range physical query tiles required')
    if Counter(tuple(sorted(tile)) for tile in train_tiles) != Counter(
            tuple(sorted(tile)) for tile in loss_context.order):
        raise ValueError('Full ranking schedule changed; no observations/pairs may be dropped')
    if Counter(i for tile in train_tiles for i in tile) != loss_context.uses:
        raise ValueError('Original observation coverage and schedule multiplicities required')

    tiles = []
    for schedule_index, ids in enumerate(train_tiles):
        rows = [loss_context.rows[i] for i in ids]
        if len({row['case_id'] for row in rows}) != 1:
            raise ValueError('One original recipient case per scheduled ranking tile required')
        observed = sum(int(row['target']) for row in rows)
        unobserved = len(ids)-observed
        tiles.append(dict(schedule_index=schedule_index,
            tile_id=f'schedule:{schedule_index}', indices=list(ids),
            observation_ids=[row['id'] for row in rows], case_id=rows[0]['case_id'],
            physical_batch=len(ids), observed=observed, unobserved=unobserved,
            ranking_pairs=observed*unobserved))
    eligible = [tile for tile in tiles if tile['physical_batch'] == physical_batch
                and tile['ranking_pairs'] > 0]
    choices = tiles if selection_policy == 'complete_prefix' else eligible
    if steps > len(choices):
        raise ValueError(f'DEBUG {selection_policy} requires {steps} native tiles; '
                         f'only {len(choices)} available within the unchanged complete schedule')
    selected = choices[:steps]
    return dict(selection_policy=selection_policy,
        selected_schedule_indices=[tile['schedule_index'] for tile in selected],
        selected_tiles=selected, eligible_tile_count=len(eligible),
        eligible_full_physical_rankable_tile_count=len(eligible),
        selection_basis='existing schedule order and original P/U metadata only; no score selection',
        full_schedule_audit=copy.deepcopy(loss_context.audit),
        full_schedule_tiles=len(train_tiles), full_cohort_observations=len(loss_context.rows),
        full_schedule_query_presentations=sum(loss_context.uses.values()),
        selected_unique_observations=len({i for tile in selected for i in tile['indices']}),
        complete_schedule_coverage_and_multiplicity_preserved=True,
        normalization=dict(steps=loss_context.steps, pairs=loss_context.pairs,
            class_counts=dict(loss_context.counts),
            observation_multiplicities_sha256=hash_state(dict(loss_context.uses)),
            ranking_coefficient=loss_context.steps/loss_context.pairs,
            observation_auxiliary_coefficient='steps / (2 * full_class_count * original_schedule_uses)',
            scope='unchanged full-cohort LiveContext; never selected-prefix normalization'),
        subset_scope='explicit DEBUG update selection; full cohort and production schedule retained')


def ranking_parameter_gradients(ranking_loss, named, *, require_nonzero=False):
    """Measure this forward's ranking-only parameter derivatives without .grad writes.

    Retaining the graph lets the unchanged full objective perform its single
    optimizer backward afterwards. Missing ranking dependencies stay explicit.
    """
    from tools.local_cnn_shadow_probe import MODULES
    if (not named or ranking_loss.ndim != 0 or not ranking_loss.requires_grad
            or not bool(torch.isfinite(ranking_loss))):
        raise ValueError('Finite differentiable scalar ranking loss and trainable parameters required')
    gradients = torch.autograd.grad(ranking_loss, [p for _, p in named],
                                    retain_graph=True, allow_unused=True)
    present = [value for value in gradients if value is not None]
    if present and not bool(torch.stack([torch.isfinite(value).all() for value in present]).all()):
        raise FloatingPointError('Nonfinite ranking-only parameter gradient')
    norms, connections = {}, {}
    for label, prefixes in MODULES.items():
        bound = [value for (name, _), value in zip(named, gradients) if name.startswith(prefixes)]
        connected = [value for value in bound if value is not None]
        norms[label] = (float(torch.stack([value.detach().float().square().sum()
                         for value in connected]).sum().sqrt()) if connected else None)
        connections[label] = dict(trainable_parameters=len(bound), connected_parameters=len(connected))
    norms['global'] = (float(torch.stack([value.detach().float().square().sum()
                         for value in present]).sum().sqrt()) if present else None)
    required = ['CNN', 'L1', 'L2'] if require_nonzero else []
    if any(norms[label] is None or not math.isfinite(norms[label]) or norms[label] <= 0
           for label in required):
        raise ValueError('Rankable DEBUG update requires nonzero finite ranking parameter gradients '
                         f'through CNN/L1/L2; measured {norms}')
    return dict(status='MEASURED', weighted_loss=float(ranking_loss.detach()),
        module_gradient_norms=norms, module_parameter_connections=connections,
        required_nonzero_modules=required,
        scope='ranking-only derivatives from this exact original full-objective forward',
        original_gradient_buffers_untouched=True)


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


def evaluator(cases, memory, train_rows, physical_batch, budget, original_local_hash, *, target_signal=False):
    """Retain every candidate of each explicitly selected train/validation case."""
    from tools.diagnose_local_cnn_learning import score_summary
    if not cases or {case['split'] for case in cases} != {'train', 'validation'}:
        raise ValueError('Matched explicit train and validation case sets required')

    @torch.no_grad()
    def evaluate(model, *, case_ids=None):
        modes = [(module, module.training) for module in model.modules()]
        parts = {split: [] for split in ('train', 'validation')}
        try:
            model.eval()
            reuse = hash_state(model.local.state_dict()) == original_local_hash
            selected = cases if case_ids is None else [c for c in cases if c['rows'][0]['case_id'] in case_ids]
            if not selected or (case_ids is not None and {c['rows'][0]['case_id'] for c in selected} != set(case_ids)):
                raise ValueError('Every actual fitted case must be available for evaluation')
            for case in selected:
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
                truth = case['truth']
                if target_signal:
                    from tools.local_cnn_reference_causal import target_head
                    scores, trace = target_head(model, embeddings, support, physical_batch, truth, rows)
                else:
                    scores, trace = _head(model, embeddings, support, physical_batch)
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
                if not rows:
                    continue
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
                  evaluation_provider, progress=True, selection_policy='complete_prefix',
                  transfer_policy='raw_columns', causal_probe=False, fitted_evaluation_provider=None):
    """Fresh matched AdamW on explicit original tiles with full-cohort coefficients.

    The historical complete-prefix/raw-column path remains the default.
    Rankable full-batch selection and alternative transfers are opt-in DEBUG
    controls; neither mode writes or updates the production model.
    """
    from tools.local_cnn_reference_l1 import clone_reference
    from tools.local_cnn_reference_transfer import POLICIES, clone_reference_control
    # All selection/transfer settings are admitted before GPU providers or clones.
    selection = select_update_tiles(train_tiles, steps=steps, loss_context=loss_context,
        physical_batch=physical_batch, selection_policy=selection_policy)
    if transfer_policy not in POLICIES:
        raise ValueError(f'Explicit reference transfer policy required; choose from {POLICIES}')
    rankable = selection_policy == 'rankable_full_batch_prefix'
    if type(causal_probe) is not bool or (causal_probe and (not rankable or fitted_evaluation_provider is None)):
        raise ValueError('Causal DEBUG control requires original full rankable tiles and complete fitted-case evaluator')
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
                if branch == 'legacy':
                    clone, contract = copy.deepcopy(net), dict(architecture='unchanged_legacy_L1')
                elif transfer_policy == 'raw_columns':
                    # Preserve the historical transfer implementation and metadata.
                    clone, contract = clone_reference(net)
                else:
                    clone, contract = clone_reference_control(net, policy=transfer_policy)
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
                if causal_probe:
                    from tools.local_cnn_reference_causal import fitted_case_ids
                    fitted_ids = fitted_case_ids(selection)
                    fitted_timeline = [dict(after_updates=0, evaluation=_evaluate(
                        lambda model: fitted_evaluation_provider(model, fitted_ids), clone))]
                clone.train()
                reports, last_group, plan = [], None, None
                bar = tqdm(range(steps), desc=f'CLONED {branch} L1', disable=not progress)
                for offset in bar:
                    tile = selection['selected_tiles'][offset]
                    ids = tile['indices']
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
                    plan_hash = hash_state(plan)
                    targets = torch.tensor([loss_context.rows[i]['target'] for i in ids], device=device)
                    causal = {}
                    if causal_probe:
                        from tools.local_cnn_reference_causal import frozen_tile, capture_prediction, tile_scores
                        causal['tile_before'] = frozen_tile(clone, query, support, plan, targets,
                            loss_context, ids, budget)
                        if branch == 'reference' and offset == 0:
                            from tools.local_cnn_reference_mode_probe import mode_probe
                            causal['mode_control'] = mode_probe(clone, query, support, plan, targets,
                                loss_context, ids, training, budget)
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize(device); started = time.perf_counter()
                    if causal_probe:
                        with capture_prediction(clone) as captured:
                            loss, terms = forward_loss(clone, query, support, plan, targets, None,
                                loss_context, configuration(), indices=ids)
                        if len(captured) != 1:
                            raise AssertionError('Causal observation must use exactly one original forward')
                        causal['train_forward_tile'] = tile_scores(captured[0], targets,
                            [loss_context.rows[i] for i in ids])
                        del captured
                    else:
                        loss, terms = forward_loss(clone, query, support, plan, targets, None,
                            loss_context, configuration(), indices=ids)
                    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite full original objective')
                    if int(terms['ranking_pairs'].detach()) != tile['ranking_pairs']:
                        raise AssertionError('Actual forward ranking pairs differ from original selected tile')
                    ranking_gradient = dict(status='NOT_RUN',
                        reason='Historical complete-prefix path; ranking-only gradients are required in rankable control')
                    if rankable:
                        if len(ids) != physical_batch or tile['ranking_pairs'] <= 0:
                            raise AssertionError('Rankable selection lost its full physical P/U comparison')
                        budget.check()
                        ranking_gradient = ranking_parameter_gradients(
                            configuration()['ranking_weight']*terms['ranking_loss'], named, require_nonzero=True)
                        budget.check()
                    if causal_probe:
                        from tools.local_cnn_reference_objective_probe import objective_probe
                        settings = configuration()
                        weighted = dict(ranking=settings['ranking_weight']*terms['ranking_loss'],
                            observation_ce=settings['observation_auxiliary_weight']*terms['observation_auxiliary_loss'],
                            alignment=clone.alignment_loss_weight*terms['alignment_loss'], full=loss)
                        causal['objective_direction'] = objective_probe(clone, named, weighted,
                            optimizer, training['grad_clip'])
                        step_initial = {name:p.detach().clone() for name,p in named}
                        budget.check()
                    loss.backward(); gradient_check_batched(clone)
                    norms = _module_norms(named, [p.grad for _, p in named])
                    grad = torch.nn.utils.clip_grad_norm_([p for _, p in named], training['grad_clip'], error_if_nonfinite=True)
                    optimizer.step(); torch.cuda.synchronize(device)
                    if causal_probe:
                        actual_delta = hash_state({name:p.detach()-step_initial[name] for name,p in named})
                        if actual_delta != causal['objective_direction']['full_delta_sha256']:
                            raise AssertionError('Shadow full AdamW differs from actual original full update')
                        causal['actual_full_delta_sha256'] = actual_delta
                        causal['shadow_full_matches_actual_update'] = True
                        del step_initial
                    seconds = time.perf_counter()-started
                    budget.check()
                    if hash_state(dict(query=native, support=support, records=records, group=group)) != digest:
                        raise AssertionError('Native query/support mutated during comparison')
                    if hash_state(plan) != plan_hash:
                        raise AssertionError('Own-branch teacher plan mutated during comparison')
                    reports.append(dict(step=offset+1, indices=ids, physical_batch=len(ids),
                        input_tensor_shape=list(query.images.shape), organ_tensor_shape=list(query.organ.shape),
                        native_crop_audit=query.audit,
                        schedule_index=tile['schedule_index'], tile_id=tile['tile_id'],
                        observation_ids=tile['observation_ids'], case_id=tile['case_id'],
                        observed=tile['observed'], unobserved=tile['unobserved'],
                        ranking_pairs=tile['ranking_pairs'], ranking_parameter_gradient=ranking_gradient,
                        teacher_plan_sha256=plan_hash, own_branch_teacher_plan=True,
                        input_content_sha256=digest, support_record_ids=records,
                        loss=float(loss.detach()), terms={k: float(v.detach()) for k, v in terms.items()},
                        module_gradient_norms=norms, gradient_norm_before_clip=float(grad),
                        synchronized_update_seconds=seconds))
                    if causal_probe:
                        causal['tile_after'] = frozen_tile(clone, query, support, plan, targets,
                            loss_context, ids, budget)
                        reports[-1]['causal'] = causal
                        fitted_timeline.append(dict(after_updates=offset+1, tile_id=tile['tile_id'],
                            evaluation=_evaluate(lambda model: fitted_evaluation_provider(model, fitted_ids), clone)))
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
                    reference_transfer_policy=transfer_policy if branch == 'reference' else None,
                    selection_policy=selection_policy,
                    optimizer='fresh AdamW; no saved moments reused', precision='FP32',
                    initial_parameters=sum(p.numel() for _, p in named),
                    before=before, after=after, updates=reports, parameter_delta_norms=changes))
                if causal_probe:
                    results[-1]['fitted_case_timeline'] = fitted_timeline
                    results[-1]['fitted_case_ids'] = fitted_ids
                del clone, optimizer, named, initial
    finally:
        restore_rng(saved_rng)
        if hash_state(net.state_dict()) != saved_state or hash_state({n: p.grad for n, p in net.named_parameters()}) != saved_grad:
            raise AssertionError('Original model/gradient buffers changed')
        if any(m.training != mode for m, mode in saved_modes) or hash_state(rng_state()) != hash_state(saved_rng):
            raise AssertionError('Original module modes/caller RNG changed')
    return dict(diagnostic_only=True, branches=results, steps_per_branch=steps,
        causal_probe=causal_probe,
        selection_policy=selection_policy, reference_transfer_policy=transfer_policy,
        update_selection=selection,
        selected_schedule_indices=selection['selected_schedule_indices'],
        ranking_parameter_gradients_required=rankable,
        physical_batch=physical_batch, full_objective=configuration(),
        execution=_execution_runtime(device), original_state_and_rng_preserved=True,
        timing_scope='synchronized forward/full loss/backward/gradient diagnostics/clip/AdamW only; excludes CT loading, transfer, input hashing, teacher preparation, evaluation and checkpoint IO; not epoch timing',
        same_native_inputs=True, original_L0_L2_initial_weights_preserved=True,
        exact_resume=False, production_optimizer_updates=0, production_checkpoint_written=False,
        production_ready=False, full_training=False, full_evaluation=False)
