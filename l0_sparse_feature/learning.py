"""Complete-case DEBUG learning bindings for the sparse L0 comparison.

This module does not replace a production dataset, objective, support policy or
optimizer. Native observations and the existing same-donor assignment are
preserved. Evaluation always re-encodes current native CT on CUDA; a detached
training-only support table is supplied explicitly by the caller.
"""
from collections import Counter
import copy
import time

import numpy as np
import torch
from torch.nn import functional as F

from l0_regions.donor_learning import validate_rows


DONOR_FIELDS = ('donor_case_id', 'donor_component', 'donor_group')
OBSERVATION_FIELDS = ('id', 'case_id', 'patient_group', 'component', 'center', 'target')


def _center_key(center):
    value = np.asarray(center)
    if (value.shape != (3,) or not np.issubdtype(value.dtype, np.integer)
            or not np.isfinite(value).all()):
        raise ValueError('Original finite integer native observation center required')
    return tuple(int(v) for v in value)


def _donor_contract(meta):
    known = meta['identities']['cases']
    train = set(meta['split']['inner_train'])
    pool = meta['donor_pool']
    if not pool or any(d['case_id'] not in train or d['case_id'] not in known for d in pool):
        raise ValueError('Nonempty original train-only donor pool required')
    keys = [(d['case_id'], d['component_id']) for d in pool]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate original donor component')
    return known, train, set(keys)


def _validate_binding(rows, meta):
    if not rows:
        raise ValueError('Explicit nonempty original observation rows required')
    known, _, pool = _donor_contract(meta)
    for row in rows:
        case = row['case_id']
        donor = row['donor_case_id']
        if case not in known or row['patient_group'] != known[case]['patient_group']:
            raise ValueError('Original recipient patient identity differs')
        if ((donor, row['donor_component']) not in pool
                or row['donor_group'] != known[donor]['patient_group']):
            raise ValueError('Original train-only donor component/identity differs')
        _center_key(row['center'])
    validate_rows(rows)


def expand_rows(meta, original_rows, cases, donor_overrides=None):
    """Retain every original P and all128 U of the explicit selected cases.

    ID and row order come from the original observation list, including its
    existing scheduling bounds. No observation or native anchor is regenerated.
    If donor overrides are absent, use the existing deterministic full-pool
    same-donor assignment; an override must bind each selected case explicitly.
    """
    requested = list(cases)
    if (not requested or any(not isinstance(c, str) or not c for c in requested)
            or len(set(requested)) != len(requested)):
        raise ValueError('Explicit distinct selected case IDs required')
    originals = list(original_rows)
    if not originals or len({r['id'] for r in originals}) != len(originals):
        raise ValueError('Complete unique original observation IDs required')
    wanted = set(requested)
    selected = [(i, row) for i, row in enumerate(originals) if row['case_id'] in wanted]
    if {row['case_id'] for _, row in selected} != wanted:
        raise ValueError('A selected case is missing from the original observations')
    known, _, _ = _donor_contract(meta)
    if not wanted <= set(known):
        raise ValueError('A selected case lacks original patient identity')
    raw_list = meta['raw_records']
    raw = {r['case_id']: r for r in raw_list}
    if len(raw) != len(raw_list) or not wanted <= set(raw):
        raise ValueError('Unique original raw records required for every selected case')
    for case in requested:
        observed = [r for _, r in selected if r['case_id'] == case]
        for row in observed:
            if any(k not in row for k in OBSERVATION_FIELDS):
                raise ValueError('An original observation field is missing')
            if row['target'] not in (0, 1) or row['patient_group'] != known[case]['patient_group']:
                raise ValueError('Original binary observation/patient identity differs')
        positives = [r for r in observed if r['target'] == 1]
        unobserved = [r for r in observed if r['target'] == 0]
        rp = raw[case]['positives']
        ru = raw[case]['comparison']['centers']
        if len(positives) != len(rp) or len(unobserved) != 128 or len(ru) != 128:
            raise ValueError('Every original positive and exactly128 original U required')
        pkey = lambda r: (r['component'], _center_key(r['center']))
        if (any(r['component'] is None for r in positives)
                or Counter(map(pkey, positives)) != Counter(map(pkey, rp))):
            raise ValueError('Original observed component/anchor coverage differs')
        if (any(r['component'] is not None for r in unobserved)
                or Counter(_center_key(r['center']) for r in unobserved)
                != Counter(_center_key(c) for c in ru)):
            raise ValueError('Original unobserved center coverage differs')
    if donor_overrides is None:
        from l0_regions.donor_data import assignment
        assignment_meta = dict(meta, records=originals)
        assigned = {r['id']: r for r in assignment(assignment_meta, meta['config']['seed'])}
        donors = {case: {k: assigned[next(r['id'] for _, r in selected
                        if r['case_id'] == case)][k] for k in DONOR_FIELDS}
                  for case in requested}
    else:
        if set(donor_overrides) != wanted:
            raise ValueError('Explicit donor overrides must cover exactly the selected cases')
        donors = {}
        for case in requested:
            value = donor_overrides[case]
            if any(k not in value for k in DONOR_FIELDS):
                raise ValueError('Complete original donor fields required in overrides')
            donors[case] = {k: copy.deepcopy(value[k]) for k in DONOR_FIELDS}
    rows = []
    for original_index, row in selected:
        new = copy.deepcopy(row)
        new.update(donors[row['case_id']])
        if 'bounds' not in new:
            new['bounds'] = {'edges': original_index}
        if not isinstance(new['bounds'], dict) or 'edges' not in new['bounds']:
            raise ValueError('Original scheduling bounds.edges required when bounds exist')
        rows.append(new)
    _validate_binding(rows, meta)
    return rows


def validate_train_only(rows, meta):
    """Guard the support whitelist separately from query observation expansion."""
    _validate_binding(rows, meta)
    train = set(meta['split']['inner_train'])
    if not {r['case_id'] for r in rows} <= train:
        raise ValueError('Validation or held-out observation entered support memory')
    train_groups = {meta['identities']['cases'][c]['patient_group'] for c in train}
    if not {r['patient_group'] for r in rows} <= train_groups:
        raise ValueError('Validation or held-out patient entered support memory')


def build_memory(rows, embeddings):
    """The existing metadata layout, plus bound source rows for strict checking.

    Pure metadata assembly can be tested on CPU. This is not CPU model inference;
    actual native CT embedding creation and evaluation require CUDA.
    """
    rows = list(rows)
    if not rows:
        raise ValueError('Nonempty complete support rows required')
    validate_rows(rows)
    if (not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2
            or embeddings.shape != (len(rows), 128) or embeddings.dtype != torch.float32):
        raise ValueError('Aligned original FP32 [observations,128] support embeddings required')
    if not bool(torch.isfinite(embeddings).all()):
        raise FloatingPointError('Nonfinite support embedding')
    groups = sorted({r['patient_group'] for r in rows})
    owners = {g: i for i, g in enumerate(groups)}
    return dict(embeddings=embeddings.detach(), record_ids=[r['id'] for r in rows],
        patient_groups=groups, donor_groups=[r['donor_group'] for r in rows],
        owners=torch.tensor([owners[r['patient_group']] for r in rows],
            device=embeddings.device, dtype=torch.long),
        classes=torch.tensor([r['target'] for r in rows], device=embeddings.device, dtype=torch.long),
        source_rows=copy.deepcopy(rows))


def validate_train_memory(memory, meta):
    """Check the complete detached table's order, ownership and train whitelist."""
    if 'source_rows' not in memory:
        raise ValueError('Support memory lacks its original source-row binding')
    rows = memory['source_rows']
    validate_train_only(rows, meta)
    if (memory['record_ids'] != [r['id'] for r in rows]
            or memory['donor_groups'] != [r['donor_group'] for r in rows]
            or memory['patient_groups'] != sorted({r['patient_group'] for r in rows})):
        raise ValueError('Training memory order/patient/donor identity differs')
    embedding, owners, classes = (memory[k] for k in ('embeddings', 'owners', 'classes'))
    if (not all(isinstance(v, torch.Tensor) for v in (embedding, owners, classes))
            or embedding.shape != (len(rows), 128) or embedding.dtype != torch.float32
            or embedding.requires_grad or owners.shape != (len(rows),)
            or classes.shape != (len(rows),) or owners.dtype != torch.long
            or classes.dtype != torch.long or owners.device != embedding.device
            or classes.device != embedding.device):
        raise ValueError('Detached FP32 128D memory and aligned int64 metadata required')
    if not bool(torch.isfinite(embedding).all()):
        raise FloatingPointError('Nonfinite original training memory')
    groups = {group: i for i, group in enumerate(memory['patient_groups'])}
    expected_owners = torch.tensor([groups[r['patient_group']] for r in rows],
        device=embedding.device, dtype=torch.long)
    expected_classes = torch.tensor([r['target'] for r in rows],
        device=embedding.device, dtype=torch.long)
    if not torch.equal(owners, expected_owners) or not torch.equal(classes, expected_classes):
        raise ValueError('Training memory original owner/class observation binding differs')


def _finite_tree(value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError('Nonfinite evaluation support/model output')
    elif isinstance(value, dict):
        for part in value.values():
            _finite_tree(part)
    elif isinstance(value, (tuple, list)):
        for part in value:
            _finite_tree(part)


def _score_diagnostics(scores, truth, logits):
    difference = scores[truth == 1, None] - scores[None, truth == 0]
    result = dict(records=len(scores), observed=int((truth == 1).sum()),
        unobserved=int((truth == 0).sum()), comparisons=difference.numel(),
        rank_evaluable=bool(difference.numel()), score_std=float(scores.std(unbiased=False)),
        score_min=float(scores.min()), score_max=float(scores.max()),
        logit_min=float(logits.min()), logit_max=float(logits.max()))
    if difference.numel():
        result.update(pair_win_rate=float((difference > 0).float().mean()),
            exact_tie_rate=float((difference == 0).float().mean()),
            pair_win_with_half_ties=float(((difference > 0).float()
                + .5 * (difference == 0).float()).mean()),
            mean_positive_minus_unobserved=float(difference.mean()),
            mean_pairwise_loss=float(F.softplus(-difference).mean()))
    else:
        result.update(pair_win_rate=None, exact_tie_rate=None,
            pair_win_with_half_ties=None, mean_positive_minus_unobserved=None,
            mean_pairwise_loss=None)
    return result


@torch.no_grad()
def evaluate_current(net, rows, store, memory, budget, batch=32):
    """Evaluate all case observations with the current CNN and unchanged head.

    One support-only teacher/state is prepared per case. No validation rows enter
    that state, and both CNN and query head run in physical32 chunks (including
    explicit natural final partial chunks). Candidate labels enter metrics only.
    """
    from hiercp_v222.v1_execution import rng_state, restore_rng
    from tools.local_cnn_interaction_runtime import support_binding
    from tools.verify_sparse_feature_ct_debug import bound_batch
    from tools.v22_candidate_order import record_key
    from tools.v22_rank_objective import ranking_metrics

    if batch != 32 or type(batch) is not int:
        raise ValueError('This DEBUG comparison preserves physical batch32')
    rows = list(rows)
    if not torch.cuda.is_available() or next(net.parameters()).device.type != 'cuda':
        raise RuntimeError('Native current-model evaluation requires actual CUDA; no CPU fallback')
    _validate_binding(rows, store.meta)
    # The standalone evaluator must reject an incomplete query case as well as
    # the runner. Validation reuses the same raw P/U multiset contract without
    # reassigning or moving any native query/donor anchor.
    case_names = list(dict.fromkeys(r['case_id'] for r in rows))
    donors = {case: {k: next(r for r in rows if r['case_id'] == case)[k]
        for k in DONOR_FIELDS} for case in case_names}
    expand_rows(store.meta, rows, case_names, donors)
    if 'source_rows' not in memory:
        raise ValueError('Support memory must retain its bound original source rows')
    support_rows = memory['source_rows']
    validate_train_memory(memory, store.meta)
    if (memory['embeddings'].device != next(net.parameters()).device
            or memory['embeddings'].requires_grad):
        raise ValueError('Detached same-device training-only support memory required')
    budget.check()
    modes = [(module, module.training) for module in net.modules()]
    original_rng = rng_state()
    by_case = validate_rows(rows)
    all_scores = torch.empty(len(rows), dtype=torch.float32, device='cpu')
    all_logits = torch.empty((len(rows), 2), dtype=torch.float32, device='cpu')
    details = {}
    started = time.perf_counter()
    cnn_batches = []
    try:
        net.eval()
        with torch.autocast('cuda', enabled=False):
            for case, indices in by_case.items():
                budget.check()
                case_rows = [rows[i] for i in indices]
                support, support_ids, _ = support_binding(memory, support_rows,
                    case_rows[0]['patient_group'])
                plan = net.fit_support_clusters(*support)
                state = net.prepare_support(*support, cluster_plan=plan)
                _finite_tree(state)
                local_scores, local_logits = [], []
                for start in range(0, len(indices), batch):
                    ids = indices[start:start + batch]
                    budget.check()
                    query = bound_batch(store, [rows[i] for i in ids], ids)
                    embedding = net.local(query)
                    if (embedding.shape != (len(ids), 128) or embedding.dtype != torch.float32
                            or embedding.device != memory['embeddings'].device):
                        raise ValueError('Current native query 128D FP32 binding differs')
                    output = net.predict_embeddings(embedding, state)
                    _finite_tree((embedding, output))
                    logits = output['logits'].float()
                    if logits.shape != (len(ids), 2):
                        raise ValueError('Original two-class scoring output shape differs')
                    local_scores.append((logits[:, 1] - logits[:, 0]).detach())
                    local_logits.append(logits.detach())
                    cnn_batches.append(len(ids))
                    budget.check()
                    del query, embedding, output, logits
                scores = torch.cat(local_scores)
                logits = torch.cat(local_logits)
                truth = torch.tensor([r['target'] for r in case_rows],
                    dtype=torch.long, device=scores.device)
                report = _score_diagnostics(scores, truth, logits)
                details[case] = dict(case_id=case, **report,
                    support_record_ids=support_ids, support_records=len(support_ids),
                    all_original_case_observations=True,
                    current_native_CNN_reencoded=True, cached_query_embeddings_reused=False,
                    support_teacher_fits=1, support_state_preparations=1)
                # Model inference stayed on CUDA. Only completed metric inputs
                # move to CPU; the original case/input order is restored here.
                all_scores[indices] = scores.cpu()
                all_logits[indices] = logits.cpu()
                del local_scores, local_logits, scores, logits, truth, plan, state, support
                budget.check()
        truth_cpu = torch.tensor([r['target'] for r in rows], dtype=torch.long)
        metrics, rank_cases = ranking_metrics(all_scores, truth_cpu,
            [r['case_id'] for r in rows], candidate_keys=[record_key(r) for r in rows])
        for rank_case in rank_cases:
            details[rank_case['case_id']].update(rank_case)
        comparisons = sum(r['comparisons'] for r in details.values())
        if not comparisons:
            raise ValueError('Complete selected evaluation needs original P/U comparisons')
        weighted = lambda key: sum(r[key] * r['comparisons'] for r in details.values()
            if r['comparisons']) / comparisons
        diagnostics = dict(comparisons=comparisons,
            pair_win_rate=weighted('pair_win_rate'), exact_tie_rate=weighted('exact_tie_rate'),
            pair_win_with_half_ties=weighted('pair_win_with_half_ties'),
            mean_positive_minus_unobserved=weighted('mean_positive_minus_unobserved'),
            score_std=float(all_scores.std(unbiased=False)), score_min=float(all_scores.min()),
            score_max=float(all_scores.max()), logit_min=float(all_logits.min()),
            logit_max=float(all_logits.max()))
        torch.cuda.synchronize()
        return dict(metrics=metrics, cases=[details[c] for c in by_case],
            scores=all_scores.tolist(), record_ids=[r['id'] for r in rows],
            diagnostics=diagnostics, evaluation=dict(current_native_CNN_reencoded=True,
                support_train_only=True, support_query_exclusion=True, eval_mode=True,
                dropout_disabled=True, physical_batch=32, actual_batch_sizes=cnn_batches,
                complete_selected_case_P_U=True, original_candidates=128,
                support_refresh_performed_inside_evaluator=False,
                support_scope='caller-bound detached training-only table; own support-only L1 teacher per case',
                seconds=time.perf_counter() - started))
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(original_rng)
