"""Confirmed v2.5 adaptive native-U curriculum, shared by the v3.0 arms.

This controller only admits cumulative prefixes of the unchanged native U128
bank. It neither chooses BEST nor changes model, data, graph, or batch scale.
All production receipts require complete train65/P527 and validation21/P135
probes, and seventeen real optimizer updates in every completed epoch.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

from .contracts import canonical_hash


FORMAT = 'v25_adaptive_mastery_overfit_remaining_budget_curriculum_v1'
POLICY_FIELDS = (
    'initial_u', 'increment_u', 'total_u', 'minimum_stage_epochs',
    'consecutive_passes', 'validation_mrr', 'validation_top1', 'train_mrr',
    'train_top1', 'overfit_mrr_gap', 'plateau_patience', 'plateau_min_delta',
    'maximum_stage_epochs', 'total_epochs', 'minimum_full_bank_epochs',
)
COMMON_POLICY = dict(
    total_u=128, minimum_stage_epochs=2, consecutive_passes=2,
    validation_mrr=.80, validation_top1=.70, train_mrr=.90, train_top1=.85,
    overfit_mrr_gap=.10, plateau_patience=2, plateau_min_delta=.005,
    maximum_stage_epochs=4, total_epochs=40, minimum_full_bank_epochs=8,
)
ARMS = {(8, 8): 'adaptive_U8_step8', (16, 16): 'adaptive_U16_step16',
        (32, 32): 'adaptive_U32_step32', (128, None): 'full_U128'}
TRAIN_PATIENTS, TRAIN_P = 65, 527
VAL_PATIENTS, VAL_P, VAL_ZERO_P = 21, 135, 5
UPDATES_PER_EPOCH = 17
METRIC_KEYS = ('per_P_patient_mrr', 'per_P_patient_top1',
               'patient_balanced_pair_loss')
GATE_PRECEDENCE = ('validation_mastery', 'train_mastery_overfit_plateau',
                   'stage_max4_deadline', 'remaining_budget_deadline')


def _latest_stage_end(policy, active_u):
    if active_u == policy['total_u']:
        return None
    remaining = (policy['total_u'] - active_u) // policy['increment_u']
    latest_full_start = policy['total_epochs'] - policy['minimum_full_bank_epochs'] + 1
    return latest_full_start - 1 - policy['minimum_stage_epochs'] * (remaining - 1)


def _schedule(policy, *, fastest):
    """Arithmetic proof only; never a claim that training actually ran."""
    active, started, result = policy['initial_u'], 1, []
    for epoch in range(1, policy['total_epochs'] + 1):
        result.append(active)
        if active == policy['total_u']:
            continue
        elapsed = epoch - started + 1
        deadline = _latest_stage_end(policy, active)
        if (elapsed >= policy['minimum_stage_epochs'] and
                (fastest or elapsed >= policy['maximum_stage_epochs'] or epoch >= deadline)):
            active += policy['increment_u']
            started = epoch + 1
    return result


def _all_schedule_first_full_epochs(policy):
    """Check every legal hold/advance path, including an admission's minimum stay."""
    states = {(policy['initial_u'], 1)}
    first_full_epochs = {1} if policy['initial_u'] == policy['total_u'] else set()
    for epoch in range(1, policy['total_epochs'] + 1):
        following = set()
        for active, started in states:
            if active == policy['total_u']:
                following.add((active, started))
                continue
            elapsed = epoch - started + 1
            deadline = _latest_stage_end(policy, active)
            if started + policy['minimum_stage_epochs'] - 1 > deadline:
                raise ValueError('Stage admission cannot retain minimum2 before remaining-budget deadline')
            if elapsed > policy['maximum_stage_epochs'] or epoch > deadline:
                raise ValueError('A legal schedule exceeded its coverage deadline; no silent extension')
            eligible = elapsed >= policy['minimum_stage_epochs']
            forced = elapsed >= policy['maximum_stage_epochs'] or epoch >= deadline
            if forced and not eligible:
                raise ValueError('Coverage requires an illegal transition before minimum2')
            if not forced:
                following.add((active, started))
            if eligible:
                next_u = active + policy['increment_u']
                if next_u > policy['total_u']:
                    raise ValueError('Exactly one increment must reach the native bank; no stage jump')
                if next_u == policy['total_u']:
                    first_full_epochs.add(epoch + 1)
                following.add((next_u, epoch + 1))
        if not following:
            raise ValueError('No complete forty-epoch curriculum is admissible')
        states = following
    if any(active != policy['total_u'] for active, _ in states):
        raise ValueError('A legal schedule did not actually train the full bank')
    latest = policy['total_epochs'] - policy['minimum_full_bank_epochs'] + 1
    if not first_full_epochs or max(first_full_epochs) > latest:
        raise ValueError('A legal schedule cannot guarantee eight full-bank training epochs')
    return sorted(first_full_epochs)


def validate_policy(policy):
    """Admit only the four confirmed arms and prove every forty-epoch schedule."""
    if not isinstance(policy, Mapping) or set(policy) != set(POLICY_FIELDS):
        raise ValueError('Explicit closed v25 curriculum policy required, including minimum_full_bank_epochs')
    integers = ('initial_u', 'total_u', 'minimum_stage_epochs', 'consecutive_passes',
                'plateau_patience', 'maximum_stage_epochs', 'total_epochs',
                'minimum_full_bank_epochs')
    if any(type(policy[key]) is not int or policy[key] < 1 for key in integers):
        raise ValueError('Positive explicit curriculum integer settings required')
    increment = policy['increment_u']
    if increment is not None and (type(increment) is not int or increment < 1):
        raise ValueError('Explicit positive adaptive increment or full-arm null required')
    for key in set(POLICY_FIELDS) - set(integers) - {'increment_u'}:
        value = policy[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Finite explicit unit-interval curriculum threshold required: ' + key)
    changed = [key for key, value in COMMON_POLICY.items() if policy[key] != value]
    if changed:
        raise ValueError('Confirmed v25 forty-epoch/minimum2/maximum4/full8 gates changed: ' + ', '.join(changed))
    arm = ARMS.get((policy['initial_u'], increment))
    if arm is None:
        raise ValueError('Exactly four confirmed arms required: U8/+8, U16/+16, U32/+32, fullU128/null')
    stages = 1 if increment is None else 1 + (policy['total_u'] - policy['initial_u']) // increment
    fastest, slowest = _schedule(policy, fastest=True), _schedule(policy, fastest=False)
    possible = _all_schedule_first_full_epochs(policy)
    earliest, latest = fastest.index(128) + 1, slowest.index(128) + 1
    if possible[0] != earliest or possible[-1] != latest:
        raise ValueError('Exhaustive schedule proof differs from earliest/worst-case schedules')
    return dict(policy=copy.deepcopy(dict(policy)), policy_sha256=canonical_hash(policy),
        arm=arm, stages=stages, earliest_first_full_bank_epoch=earliest,
        latest_first_full_bank_epoch=latest,
        guaranteed_full_bank_epochs=policy['total_epochs'] - latest + 1,
        minimum_full_bank_epochs=policy['minimum_full_bank_epochs'],
        full_bank_start_deadline_epoch=policy['total_epochs'] - policy['minimum_full_bank_epochs'] + 1,
        possible_first_full_bank_epochs=possible,
        earliest_active_u_by_epoch=fastest, worst_case_active_u_by_epoch=slowest,
        latest_current_stage_end_by_u={str(u): _latest_stage_end(policy, u) for u in sorted(set(fastest))})


def _initial_state(proof):
    policy = proof['policy']
    counts = {str(u): 0 for u in sorted(set(proof['earliest_active_u_by_epoch']))}
    return dict(format=FORMAT, **copy.deepcopy(proof), active_u=policy['initial_u'],
        stage_started_epoch=1, stage_epochs=0, validation_pass_streak=0,
        plateau_epochs=0, plateau_reference=None, last_completed_epoch=0,
        completed_optimizer_updates=0, actual_P_U_pairs=0,
        first_full_bank_training_epoch=None, full_bank_training_epochs=0,
        full_bank_optimizer_updates=0, u_training_epochs=copy.deepcopy(counts),
        u_training_updates=copy.deepcopy(counts), u_P_U_pairs=copy.deepcopy(counts),
        training_case_ids=None, validation_case_ids=None, history=[])


def new_curriculum(policy):
    return _initial_state(validate_policy(policy))


def _metrics(report, active_u):
    if not isinstance(report, Mapping) or not isinstance(report.get('metrics'), Mapping):
        raise ValueError('Actual finite patient-balanced probe metrics required; missing metrics are not zero')
    value = report['metrics']
    if any(key not in value or type(value[key]) not in (int, float) or
           not math.isfinite(value[key]) for key in METRIC_KEYS):
        raise ValueError('Actual finite patient-balanced probe metrics required; missing metrics are not zero')
    if (not 1 / (active_u + 1) <= value[METRIC_KEYS[0]] <= 1 or
            not 0 <= value[METRIC_KEYS[1]] <= 1 or value[METRIC_KEYS[2]] < 0):
        raise ValueError('Invalid native probe metric range; nonempty-P MRR cannot be zero')
    return {key: value[key] for key in METRIC_KEYS}


def _case_ids(cases, *, count, label):
    if (not isinstance(cases, (list, tuple)) or len(cases) != count or
            any(not isinstance(case, str) or not case for case in cases) or
            len(set(cases)) != len(cases)):
        raise ValueError(f'Complete nonduplicated {label} population of {count} patients required')
    return sorted(cases)


def _validate_report(report, *, epoch, active_u, phase, expected_cases=None):
    if (not isinstance(report, Mapping) or type(report.get('epoch')) is not int or
            report['epoch'] != epoch or type(report.get('active_u')) is not int or
            report['active_u'] != active_u or report.get('phase') != phase or
            type(report.get('evaluation_view_epoch')) is not int or
            report['evaluation_view_epoch'] != 29 or report.get('all_P_scored') is not True):
        raise ValueError('Same-stage fixed-view actual train/validation probes required')
    train = phase == 'train_probe'
    patients, positives, zero_p = (TRAIN_PATIENTS, TRAIN_P, 0) if train else (VAL_PATIENTS, VAL_P, VAL_ZERO_P)
    rows = report.get('cases')
    if not isinstance(rows, (list, tuple)) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError('Complete actual patient probe rows required')
    ids = _case_ids([row.get('case_id') for row in rows], count=patients, label=phase)
    if expected_cases is not None and ids != expected_cases:
        raise ValueError('Probe population differs from the complete bound patient split')
    row_metrics = []
    for row in rows:
        p, u = row.get('observed_P'), row.get('unobserved_U')
        if type(p) is not int or p < 0 or type(u) is not int or u != active_u:
            raise ValueError('Every patient must score all native P and the same cumulative active-U prefix')
        if p:
            per_p = row.get('per_P')
            if not isinstance(per_p, Mapping):
                raise ValueError('Observed-P patient is missing actual per-P ranking metrics')
            mapped = {'metrics': dict(per_P_patient_mrr=per_p.get('mrr'),
                per_P_patient_top1=per_p.get('top1'), patient_balanced_pair_loss=per_p.get('pair_loss'))}
            row_metrics.append(_metrics(mapped, active_u))
        elif row.get('per_P', 'MISSING') is not None:
            raise ValueError('Zero-P patient must remain in coverage with explicitly undefined ranking metrics')
    if sum(row['observed_P'] for row in rows) != positives or patients - len(row_metrics) != zero_p:
        raise ValueError('Entire train65/P527 or validation21/P135/zeroP5 population required')
    required = dict(patients=patients, ranking_patients=patients-zero_p,
        zero_P_patients=zero_p, observed_P=positives, unobserved_U=patients*active_u,
        P_U_pairs=positives*active_u)
    denominators = report.get('denominators')
    if (not isinstance(denominators, Mapping) or
            any(type(denominators.get(key)) is not int or denominators[key] != value
                for key, value in required.items())):
        raise ValueError('Actual complete patient/P/U/pair denominators required')
    metrics = _metrics(report, active_u)
    for key in METRIC_KEYS:
        measured = math.fsum(row[key] for row in row_metrics) / len(row_metrics)
        if not math.isclose(metrics[key], measured, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Patient-macro probe metrics differ from their complete per-P patient rows')
    return metrics, ids


def _advance(state, train, val, epoch):
    policy, active = state['policy'], state['active_u']
    nonfull = active < policy['total_u']
    deadline = _latest_stage_end(policy, active)
    if nonfull and state['stage_started_epoch'] + policy['minimum_stage_epochs'] - 1 > deadline:
        raise ValueError('Stage admission cannot retain minimum2 before remaining-budget deadline')
    if nonfull and (epoch > deadline or state['stage_epochs'] >= policy['maximum_stage_epochs']):
        raise ValueError('Coverage deadline was exceeded; no silent extension or stage jump')
    if epoch >= state['latest_first_full_bank_epoch'] and nonfull:
        raise ValueError('Coverage contract violated: full bank was not actually trained by its deadline')
    state['stage_epochs'] += 1
    state['last_completed_epoch'] = epoch
    state['completed_optimizer_updates'] += UPDATES_PER_EPOCH
    pairs = TRAIN_P * active
    state['actual_P_U_pairs'] += pairs
    state['u_training_epochs'][str(active)] += 1
    state['u_training_updates'][str(active)] += UPDATES_PER_EPOCH
    state['u_P_U_pairs'][str(active)] += pairs
    if not nonfull:
        if state['first_full_bank_training_epoch'] is None:
            state['first_full_bank_training_epoch'] = epoch
        state['full_bank_training_epochs'] += 1
        state['full_bank_optimizer_updates'] += UPDATES_PER_EPOCH
    reference = state['plateau_reference']
    improved = reference is None or (
        val[METRIC_KEYS[0]] > reference['mrr'] + policy['plateau_min_delta'] or
        val[METRIC_KEYS[1]] > reference['top1'] + policy['plateau_min_delta'])
    if improved:
        state['plateau_reference'] = dict(mrr=val[METRIC_KEYS[0]], top1=val[METRIC_KEYS[1]])
        state['plateau_epochs'] = 0
    else:
        state['plateau_epochs'] += 1
    val_pass = val[METRIC_KEYS[0]] >= policy['validation_mrr'] and val[METRIC_KEYS[1]] >= policy['validation_top1']
    state['validation_pass_streak'] = state['validation_pass_streak'] + 1 if val_pass else 0
    mastery = train[METRIC_KEYS[0]] >= policy['train_mrr'] and train[METRIC_KEYS[1]] >= policy['train_top1']
    gap = train[METRIC_KEYS[0]] - val[METRIC_KEYS[0]]
    # Compare via addition so the exact .90 versus .80 boundary is not lost
    # to subtractive floating-point roundoff of the recorded .10 gap.
    gap_pass = train[METRIC_KEYS[0]] >= val[METRIC_KEYS[0]] + policy['overfit_mrr_gap']
    plateau_pass = state['plateau_epochs'] >= policy['plateau_patience']
    eligible = state['stage_epochs'] >= policy['minimum_stage_epochs']
    gates = dict(validation_mastery=nonfull and eligible and state['validation_pass_streak'] >= policy['consecutive_passes'],
        train_mastery_overfit_plateau=nonfull and eligible and mastery and gap_pass and plateau_pass,
        stage_max4_deadline=nonfull and eligible and state['stage_epochs'] >= policy['maximum_stage_epochs'],
        remaining_budget_deadline=nonfull and eligible and epoch >= deadline)
    expanded = any(gates.values())
    reason = next((name for name in GATE_PRECEDENCE if gates[name]),
                  'keep_current_stage' if nonfull else 'full_bank_replay')
    receipt = dict(epoch=epoch, active_u=active, previous_active_u=active,
        next_active_u=active + policy['increment_u'] if expanded else active,
        expanded=expanded, reason=reason, gates=gates,
        minimum_stage_gate_passed=eligible, validation_epoch_passed=val_pass,
        validation_gate_passed=gates['validation_mastery'], train_mastery=mastery,
        overfit_mrr_gap_gate_passed=gap_pass, plateau_gate_passed=plateau_pass,
        train_mastery_overfit_plateau_gate_passed=gates['train_mastery_overfit_plateau'],
        stage_max4_deadline_gate_passed=gates['stage_max4_deadline'],
        remaining_budget_deadline_gate_passed=gates['remaining_budget_deadline'],
        latest_current_stage_end=deadline, stage_started_epoch=state['stage_started_epoch'],
        stage_epochs=state['stage_epochs'], plateau_epochs=state['plateau_epochs'],
        plateau_reference=copy.deepcopy(state['plateau_reference']),
        stage_validation_improved=improved, validation_pass_streak=state['validation_pass_streak'],
        train_minus_stage_val_mrr=gap, train_minus_stage_val_top1=train[METRIC_KEYS[1]]-val[METRIC_KEYS[1]],
        train_probe_metrics=copy.deepcopy(train), stage_validation_metrics=copy.deepcopy(val),
        training_probe_view_epoch=29, full_validation_used_for_curriculum=False,
        coverage_is_not_mastery=reason in GATE_PRECEDENCE[2:], policy_sha256=state['policy_sha256'],
        train_case_ids_sha256=canonical_hash(state['training_case_ids']),
        validation_case_ids_sha256=canonical_hash(state['validation_case_ids']),
        actual_train_patients=TRAIN_PATIENTS, actual_train_P=TRAIN_P,
        stage_validation_patients=VAL_PATIENTS, stage_validation_P=VAL_P,
        stage_validation_zero_P_patients=VAL_ZERO_P,
        optimizer_updates_in_epoch=UPDATES_PER_EPOCH,
        completed_optimizer_updates=state['completed_optimizer_updates'],
        P_U_pairs_in_epoch=pairs, actual_P_U_pairs=state['actual_P_U_pairs'],
        first_full_bank_training_epoch=state['first_full_bank_training_epoch'],
        full_bank_training_epochs=state['full_bank_training_epochs'],
        full_bank_optimizer_updates=state['full_bank_optimizer_updates'])
    if expanded:
        next_u = receipt['next_active_u']
        if next_u > policy['total_u']:
            raise ValueError('Exactly one increment required; no stage jump or omitted native U')
        next_deadline = _latest_stage_end(policy, next_u)
        if next_deadline is not None and epoch + policy['minimum_stage_epochs'] > next_deadline:
            raise ValueError('Next stage cannot retain minimum2 before remaining-budget deadline')
        state.update(active_u=next_u, stage_started_epoch=epoch+1, stage_epochs=0,
            validation_pass_streak=0, plateau_epochs=0, plateau_reference=None)
    state['history'].append(receipt)
    if epoch == policy['total_epochs'] and (state['full_bank_training_epochs'] < policy['minimum_full_bank_epochs'] or
            state['completed_optimizer_updates'] != policy['total_epochs'] * UPDATES_PER_EPOCH):
        raise ValueError('Completed run must retain forty epochs/680 updates and at least eight actual full128 epochs')
    return receipt


def _validate_state(state):
    if not isinstance(state, Mapping) or state.get('format') != FORMAT:
        raise ValueError('Explicit v25 curriculum state required; cross-version fallback is forbidden')
    proof = validate_policy(state.get('policy'))
    expected = _initial_state(proof)
    if set(state) != set(expected) or state.get('policy_sha256') != proof['policy_sha256']:
        raise ValueError('Resume curriculum policy identity or state schema differs')
    history = state['history']
    if not isinstance(history, list) or len(history) > proof['policy']['total_epochs']:
        raise ValueError('Complete monotonic curriculum history required')
    if history:
        expected['training_case_ids'] = _case_ids(state['training_case_ids'], count=TRAIN_PATIENTS, label='train')
        expected['validation_case_ids'] = _case_ids(state['validation_case_ids'], count=VAL_PATIENTS, label='validation')
        if set(expected['training_case_ids']) & set(expected['validation_case_ids']):
            raise ValueError('Train and validation probe patient populations must be disjoint')
    for epoch, saved in enumerate(history, 1):
        if not isinstance(saved, Mapping):
            raise ValueError('Complete actual curriculum transition receipt required')
        train = _metrics({'metrics': saved.get('train_probe_metrics')}, expected['active_u'])
        val = _metrics({'metrics': saved.get('stage_validation_metrics')}, expected['active_u'])
        receipt = _advance(expected, train, val, epoch)
        if canonical_hash(saved) != canonical_hash(receipt):
            raise ValueError('Resume curriculum history differs from deterministic gates, coverage, or increments')
    if canonical_hash(state) != canonical_hash(expected):
        raise ValueError('Resume curriculum counters or stages differ from complete actual epoch history')


def finish_curriculum_epoch(curriculum, train_probe, stage_validation, *, epoch,
                            actual_train_cases, expected_train_cases, updates_in_epoch):
    """Finish a complete production epoch; expansion applies to the next epoch.

    Priority is validation mastery, diagnosed train mastery/overfit/plateau,
    then stage-max4 and remaining-budget deadlines. Every gate is recorded.
    Missing evidence fails explicitly and the caller's state is never mutated.
    """
    _validate_state(curriculum)
    state = copy.deepcopy(dict(curriculum))
    if (type(epoch) is not int or epoch != state['last_completed_epoch'] + 1 or
            epoch > state['policy']['total_epochs']):
        raise ValueError('A complete monotonic epoch within the unchanged forty-epoch contract required')
    actual = _case_ids(actual_train_cases, count=TRAIN_PATIENTS, label='training')
    expected = _case_ids(expected_train_cases, count=TRAIN_PATIENTS, label='expected training')
    if actual != expected or type(updates_in_epoch) is not int or updates_in_epoch != UPDATES_PER_EPOCH:
        raise ValueError('Complete train65 population and seventeen actual optimizer updates per epoch required')
    if state['training_case_ids'] is not None and state['training_case_ids'] != expected:
        raise ValueError('Training split identity differs from the bound curriculum population')
    train, _ = _validate_report(train_probe, epoch=epoch, active_u=state['active_u'],
        phase='train_probe', expected_cases=expected)
    val, val_ids = _validate_report(stage_validation, epoch=epoch, active_u=state['active_u'],
        phase='stage_validation', expected_cases=state['validation_case_ids'])
    if set(expected) & set(val_ids):
        raise ValueError('Train and validation probe patient populations must be disjoint')
    state['training_case_ids'], state['validation_case_ids'] = expected, val_ids
    receipt = _advance(state, train, val, epoch)
    return state, receipt
