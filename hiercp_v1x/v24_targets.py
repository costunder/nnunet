"""Explicit v2.4 cumulative coverage curriculum; never chooses model BEST.

Fixed-view full training probes diagnose mastery/overfit. Stage validation
still measures generalization, while a separately named coverage deadline
ensures every native U is trained within the unchanged forty-epoch contract.
"""
from __future__ import annotations

import copy
import math
from .contracts import canonical_hash


POLICY_FIELDS = ('initial_u', 'increment_u', 'total_u', 'minimum_stage_epochs',
    'consecutive_passes', 'validation_mrr', 'validation_top1', 'train_mrr',
    'train_top1', 'overfit_mrr_gap', 'plateau_patience', 'plateau_min_delta',
    'maximum_stage_epochs', 'total_epochs')


def validate_policy(policy):
    if set(policy) != set(POLICY_FIELDS):
        raise ValueError('Explicit closed v24 curriculum policy required')
    integers = ('initial_u', 'increment_u', 'total_u', 'minimum_stage_epochs',
        'consecutive_passes', 'plateau_patience', 'maximum_stage_epochs', 'total_epochs')
    if any(type(policy[k]) is not int or policy[k] < 1 for k in integers):
        raise ValueError('Positive explicit curriculum integer settings required')
    if not (policy['initial_u'] <= policy['total_u'] and
            policy['minimum_stage_epochs'] <= policy['maximum_stage_epochs']):
        raise ValueError('Invalid cumulative stage interval')
    for key in set(POLICY_FIELDS) - set(integers):
        value = policy[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Finite explicit unit-interval curriculum threshold required: ' + key)
    pre_full_stages = math.ceil((policy['total_u'] - policy['initial_u']) / policy['increment_u'])
    full_epoch = 1 + pre_full_stages * policy['maximum_stage_epochs']
    if full_epoch + policy['minimum_stage_epochs'] - 1 > policy['total_epochs']:
        raise ValueError('Configured deadlines cannot train the full bank before total epochs; no silent extension')
    return dict(policy=copy.deepcopy(policy), policy_sha256=canonical_hash(policy),
        stages=pre_full_stages + 1, latest_first_full_bank_epoch=full_epoch,
        guaranteed_full_bank_epochs=policy['total_epochs'] - full_epoch + 1)


def new_curriculum(policy):
    proof = validate_policy(policy)
    return dict(format='v24_mastery_overfit_and_coverage_curriculum_v1', **proof,
        active_u=policy['initial_u'], stage_started_epoch=1, stage_epochs=0,
        validation_pass_streak=0, plateau_epochs=0, plateau_reference=None,
        last_completed_epoch=0, history=[])


def _metrics(report):
    value = report['metrics']
    keys = ('per_P_patient_mrr', 'per_P_patient_top1', 'patient_balanced_pair_loss')
    if any(not isinstance(value[k], (int, float)) or not math.isfinite(value[k]) for k in keys):
        raise ValueError('Actual finite patient-balanced probe metrics required')
    if not all(0 <= value[k] <= 1 for k in keys[:2]) or value[keys[2]] < 0:
        raise ValueError('Invalid native probe metric range')
    return value


def finish_curriculum_epoch(curriculum, train_probe, stage_validation, *, epoch,
                            actual_train_cases, expected_train_cases, updates_in_epoch):
    """A complete epoch can advance by val success, diagnosed overfit or deadline.

    Overfit is an AND: train MRR/top1 mastery AND MRR gap AND two stage-val
    plateau epochs. Its plateau reference resets at every admitted U stage.
    Deadline admission is recorded as coverage, never as fabricated mastery.
    """
    state = copy.deepcopy(curriculum)
    proof = validate_policy(state['policy']); policy = state['policy']
    if proof['policy_sha256'] != state['policy_sha256']:
        raise ValueError('Resume curriculum policy identity differs')
    if (epoch != state['last_completed_epoch'] + 1 or epoch > policy['total_epochs']
            or len(actual_train_cases) != len(set(actual_train_cases))
            or set(actual_train_cases) != set(expected_train_cases) or updates_in_epoch < 1):
        raise ValueError('Complete nonduplicated training population and monotonic epoch required')
    active = state['active_u']
    for report, phase in ((train_probe, 'train_probe'), (stage_validation, 'stage_validation')):
        if (report['epoch'] != epoch or report['active_u'] != active or report['phase'] != phase
                or report['evaluation_view_epoch'] != 29 or report['all_P_scored'] is not True):
            raise ValueError('Same-stage fixed-view actual train/validation probes required')
    probe_cases = [row['case_id'] for row in train_probe['cases']]
    if len(probe_cases) != len(set(probe_cases)) or set(probe_cases) != set(expected_train_cases):
        raise ValueError('Training mastery probe must cover every ranking patient')
    train, val = _metrics(train_probe), _metrics(stage_validation)
    state['stage_epochs'] += 1; state['last_completed_epoch'] = epoch
    reference = state['plateau_reference']
    improved = reference is None or (
        val['per_P_patient_mrr'] > reference['mrr'] + policy['plateau_min_delta'] or
        val['per_P_patient_top1'] > reference['top1'] + policy['plateau_min_delta'])
    if improved:
        state['plateau_reference'] = dict(mrr=val['per_P_patient_mrr'], top1=val['per_P_patient_top1'])
        state['plateau_epochs'] = 0
    else:
        state['plateau_epochs'] += 1
    val_pass = val['per_P_patient_mrr'] >= policy['validation_mrr'] and val['per_P_patient_top1'] >= policy['validation_top1']
    state['validation_pass_streak'] = state['validation_pass_streak'] + 1 if val_pass else 0
    mastery = train['per_P_patient_mrr'] >= policy['train_mrr'] and train['per_P_patient_top1'] >= policy['train_top1']
    gap = train['per_P_patient_mrr'] - val['per_P_patient_mrr']
    overfit = mastery and gap >= policy['overfit_mrr_gap'] and state['plateau_epochs'] >= policy['plateau_patience']
    eligible = state['stage_epochs'] >= policy['minimum_stage_epochs']
    validation_mastered = eligible and state['validation_pass_streak'] >= policy['consecutive_passes']
    coverage_deadline = state['stage_epochs'] >= policy['maximum_stage_epochs']
    expanded = active < policy['total_u'] and eligible and (validation_mastered or overfit or coverage_deadline)
    reason = ('validation_mastery' if validation_mastered else 'train_mastery_overfit_plateau' if overfit
        else 'coverage_deadline' if coverage_deadline else 'keep_current_stage') if expanded else 'full_bank_replay' if active == policy['total_u'] else 'keep_current_stage'
    receipt = dict(epoch=epoch, previous_active_u=active, next_active_u=active,
        expanded=expanded, reason=reason, validation_gate_passed=validation_mastered,
        train_mastery=mastery, train_minus_stage_val_mrr=gap,
        train_minus_stage_val_top1=train['per_P_patient_top1']-val['per_P_patient_top1'],
        stage_epochs=state['stage_epochs'], plateau_epochs=state['plateau_epochs'],
        validation_pass_streak=state['validation_pass_streak'],
        training_probe_view_epoch=29, full_validation_used_for_curriculum=False,
        coverage_is_not_mastery=reason == 'coverage_deadline', policy_sha256=state['policy_sha256'])
    if expanded:
        state['active_u'] = min(policy['total_u'], active + policy['increment_u'])
        receipt['next_active_u'] = state['active_u']
        state.update(stage_started_epoch=epoch+1, stage_epochs=0,
            validation_pass_streak=0, plateau_epochs=0, plateau_reference=None)
    state['history'].append(receipt)
    if epoch >= proof['latest_first_full_bank_epoch'] and active < policy['total_u']:
        raise ValueError('Coverage contract violated: full bank was not actually trained by its deadline')
    return state, receipt
