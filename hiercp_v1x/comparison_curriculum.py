"""Checkpointed candidate policy; no model, optimizer, loader, or I/O mutation.

Adoption finishes the current legacy epoch before a cumulative policy begins.
Historical successful exposure is retained. Validation chooses when a pool
grows; it never chooses the full-validation BEST checkpoint.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

POLICY_NAME = 'dual_validation_cumulative_v1'
FORMAT = 'comparison_curriculum_state_v1'
ARMS = ('selected', 'native', 'native_fixed', 'native_listwise')
ADAPTIVE = ('native', 'native_listwise')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid curriculum ' + name)
    return value


def _ids(values):
    result = list(values)
    if not result or any(type(v) is not int or v < 0 for v in result) or len(set(result)) != len(result):
        raise ValueError('Complete distinct original training source indices required')
    return result


def _valid_key(key, arm):
    prefix, count = ('S', 7) if arm == 'selected' else ('U', 128)
    return isinstance(key, str) and key in {f'{prefix}:{i}' for i in range(count)}


def _sort_keys(keys):
    return sorted(keys, key=lambda key: int(key.split(':')[1]))


def _legacy_keys(arm, epoch):
    if arm == 'selected':
        return [f'S:{i}' for i in range(7)]
    start = 0 if arm == 'native_fixed' else (epoch - 1) * 7 % 128
    return [f'U:{(start + i) % 128}' for i in range(7)]


@lru_cache(maxsize=4)
def _policy(arm):
    if arm not in ARMS:
        raise ValueError('Unknown curriculum arm')
    value = dict(name=POLICY_NAME, arm=arm, adaptive=arm in ADAPTIVE,
        training_comparisons_per_source=7, bank_size=128 if arm != 'selected' else 7,
        expansion_keys=7, stage_mrr_threshold=.90, stage_top1_threshold=.85,
        consecutive_completed_stage_validations=2, minimum_completed_stage_epochs=2,
        coverage='every active key in every train source successful optimizer history',
        sampling='new keys first, then checkpointed cyclic active pool; cursor advances only at epoch completion',
        adoption='finish current epoch with legacy keys; start cumulative union next epoch',
        stage_validation='actual subset hierarchy on every held-out source; no full-score slicing',
        best='unchanged full129 patient macro MRR, top1, negative pairwise loss',
        full_validation_candidates=129, production_epochs=40,
        optimizer_scheduler_model_batch_and_source_order_unchanged=True,
        no_forced_expansion_or_full_coverage_at_epoch40=True,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    value['policy_sha256'] = _digest(value)
    return value


def policy(arm):
    return copy.deepcopy(_policy(arm))


def _seen(state, arm, train_ids):
    seen = state.get('seen_comparisons')
    if not isinstance(seen, dict) or set(seen) != {str(i) for i in train_ids}:
        raise ValueError('Saved successful exposure must cover every original train source')
    result = {}
    for source in train_ids:
        values = seen[str(source)]
        if (not isinstance(values, list) or len(set(values)) != len(values)
                or any(not _valid_key(k, arm) for k in values)):
            raise ValueError('Malformed saved successful candidate exposure')
        if arm == 'native_fixed' and any(k not in _legacy_keys(arm, 1) for k in values):
            raise ValueError('Fixed-native history contains a nonfixed candidate')
        result[str(source)] = _sort_keys(values)
    return result


def _preserved_seen(current, seen):
    previous = current['last_successful_seen']
    if not isinstance(previous, dict) or set(previous) != set(seen):
        raise ValueError('Historical exposure source inventory changed')
    for source, keys in previous.items():
        if not set(keys) <= set(seen[source]):
            raise ValueError('Historical successful candidate exposure was lost')


def _plan(epoch, stage_id, mode, pool, cursor=0, priority=()):
    keys = list(priority)
    if len(keys) > 7 or len(set(keys)) != len(keys) or not set(keys) <= set(pool):
        raise ValueError('Invalid newly admitted candidate priority')
    if not 0 <= cursor < len(pool):
        raise ValueError('Candidate cursor lies outside the active pool')
    after = cursor
    while len(keys) < 7:
        candidate = pool[after]
        after = (after + 1) % len(pool)
        if candidate not in keys:
            keys.append(candidate)
    return dict(epoch=epoch, stage_id=stage_id, mode=mode, keys=keys,
                pool_sha256=_digest(pool), cursor_before=cursor, cursor_after=after,
                priority_keys=list(priority))


def _current(state, arm):
    current = state.get('comparison_curriculum')
    if (not isinstance(current, dict) or current.get('format') != FORMAT
            or current.get('policy') != policy(arm)):
        raise ValueError('Checkpoint curriculum policy differs from the active policy')
    pool = current.get('pool_keys')
    if (not isinstance(pool, list) or not 7 <= len(pool) <= (7 if arm == 'selected' else 128)
            or len(set(pool)) != len(pool) or any(not _valid_key(k, arm) for k in pool)):
        raise ValueError('Malformed checkpoint active candidate pool')
    mode = current.get('mode')
    if mode not in ('LEGACY', 'ADAPTIVE', 'FIXED'):
        raise ValueError('Unknown candidate policy mode')
    if mode == 'ADAPTIVE' and arm not in ADAPTIVE:
        raise ValueError('Fixed arm cannot acquire an adaptive candidate pool')
    if mode == 'FIXED' and (arm in ADAPTIVE or pool != _legacy_keys(arm, 1)):
        raise ValueError('Fixed comparison policy changed')
    stage = _integer(current.get('stage_id'), 'stage id')
    cursor = _integer(current.get('pool_cursor'), 'pool cursor')
    completed = _integer(current.get('completed_stage_epochs'), 'completed stage epochs')
    streak = _integer(current.get('gate_streak'), 'gate streak')
    if streak > completed:
        raise ValueError('Gate streak exceeds completed stage observations')
    plan = current.get('epoch_plan')
    if not isinstance(plan, dict):
        raise ValueError('Checkpoint has no exact candidate epoch plan')
    epoch = _integer(plan.get('epoch'), 'plan epoch', minimum=1)
    if plan != _plan(epoch, stage, mode, pool, cursor, plan.get('priority_keys', ())):
        raise ValueError('Checkpoint candidate epoch plan changed')
    if mode == 'LEGACY' and (pool != _legacy_keys(arm, epoch) or stage != 0 or completed or streak):
        raise ValueError('Adoption must preserve the exact current legacy epoch')
    return current


def adopt_or_validate(state, arm, train_ids, checkpoint_sha):
    """Add only a curriculum substate; never rewrite an existing core cursor."""
    train_ids = _ids(train_ids)
    if (not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64
            or any(c not in '0123456789abcdef' for c in checkpoint_sha)):
        raise ValueError('Verified source checkpoint content SHA256 required')
    if arm not in ARMS:
        raise ValueError('Unknown curriculum arm')
    seen = _seen(state, arm, train_ids)
    epoch = _integer(state.get('epoch'), 'epoch', minimum=1)
    if state.get('phase') not in ('initial_validation', 'training', 'validation', 'complete'):
        raise ValueError('Unknown original training phase')
    _integer(state.get('position'), 'source position')
    _integer(state.get('updates'), 'successful optimizer updates')
    if 'comparison_curriculum' not in state:
        if state.get('phase') == 'complete' or epoch > 40:
            raise ValueError('A completed experiment cannot newly adopt a curriculum')
        pool = _legacy_keys(arm, epoch)
        adoption = dict(checkpoint_content_sha256=checkpoint_sha,
            epoch=epoch, phase=state['phase'], position=state['position'], updates=state['updates'],
            seen_comparisons=copy.deepcopy(seen), train_ids_sha256=_digest(train_ids))
        state['comparison_curriculum'] = dict(format=FORMAT, policy=policy(arm),
            adoption=adoption, mode='LEGACY', stage_id=0, pool_keys=pool,
            stage_start_epoch=epoch, completed_stage_epochs=0, gate_streak=0,
            pool_cursor=0, epoch_plan=_plan(epoch, 0, 'LEGACY', pool),
            last_successful_seen=copy.deepcopy(seen),
            last_completed_epoch=epoch - 1, last_completed_update=None,
            last_report_sha256=None, last_decision=None, events=[])
        transition = 'adopt_after_verified_legacy_checkpoint'
    else:
        transition = 'same_policy_resume'
    current = _current(state, arm)
    adoption = current['adoption']
    if adoption['train_ids_sha256'] != _digest(train_ids):
        raise ValueError('Curriculum training source inventory changed')
    for source, historical in adoption['seen_comparisons'].items():
        if not set(historical) <= set(seen[source]):
            raise ValueError('Historical successful candidate exposure was lost')
    _preserved_seen(current, seen)
    if current['mode'] != 'LEGACY' and not set().union(*map(set, seen.values())) <= set(current['pool_keys']):
        raise ValueError('Active pool discarded an actually trained candidate')
    planned = current['epoch_plan']['epoch']
    pending = (state.get('phase') == 'validation' and current['last_completed_epoch'] == epoch
               and planned == epoch + 1)
    if planned != epoch and not pending:
        raise ValueError('Candidate plan does not match the exact resumed epoch')
    return dict(transition=transition, policy_sha256=current['policy']['policy_sha256'],
                adopted_checkpoint_sha256=adoption['checkpoint_content_sha256'],
                source_checkpoint_sha256=checkpoint_sha, mode=current['mode'],
                stage_id=current['stage_id'], pool_keys=copy.deepcopy(current['pool_keys']),
                core_model_optimizer_scheduler_rng_cursor_unchanged=True)


def training_keys(state, arm, epoch):
    current = _current(state, arm)
    if current['epoch_plan']['epoch'] != epoch or state['epoch'] != epoch:
        raise ValueError('Requested training epoch differs from checkpointed candidate plan')
    return ['P', *current['epoch_plan']['keys']]


def stage_keys(state, arm):
    return ['P', *_current(state, arm)['pool_keys']]


def _stage_report(report, current, state, validation_ids):
    if report.get('epoch') != state['epoch'] or report.get('update') != state['updates']:
        raise ValueError('Stage validation belongs to another epoch or optimizer state')
    rows = report.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Complete stage validation rows required')
    indices = [row['sample_index'] for row in rows]
    if len(set(indices)) != len(indices):
        raise ValueError('Repeated source in stage validation')
    if validation_ids is not None and sorted(indices) != sorted(_ids(validation_ids)):
        raise ValueError('Stage validation did not cover the complete held-out sources')
    cases = {}
    for row in rows:
        if row.get('candidate_keys') != ['P', *current['pool_keys']]:
            raise ValueError('Stage validation candidate set differs from the active pool')
        values = [row.get(name) for name in ('mrr', 'top1')]
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError('Nonfinite or invalid stage rank metrics')
        if values[1] not in (0, 1):
            raise ValueError('Per-source top1 must be binary')
        cases.setdefault(row['case_id'], []).append(values)
    expected = {name: sum(sum(v[i] for v in group) / len(group) for group in cases.values()) / len(cases)
                for i, name in enumerate(('mrr', 'top1'))}
    metrics = report.get('metrics', {})
    if any(type(metrics.get(k)) not in (int, float) or not math.isfinite(metrics[k])
           or abs(metrics[k] - v) > 1e-12 for k, v in expected.items()):
        raise ValueError('Stage metrics differ from complete patient-macro rows')
    if 'source_problems' in report and report['source_problems'] != len(rows):
        raise ValueError('Stage source count differs from actual rows')
    if 'patients' in report and report['patients'] != len(cases):
        raise ValueError('Stage patient count differs from actual rows')
    if 'stage_id' in report and report['stage_id'] != current['stage_id']:
        raise ValueError('Stage validation belongs to a different stage')
    return expected, _digest(dict(epoch=report['epoch'], update=report['update'], rows=rows, metrics=metrics))


def finish_epoch(state, arm, stage_report, train_ids, *, validation_ids=None):
    """Commit one completed epoch's candidate cursor and next-epoch plan.

    Call after full validation and actual stage validation, before advancing the
    core epoch/scheduler. A replay of the identical commit is idempotent.
    """
    train_ids = _ids(train_ids)
    current = _current(state, arm)
    epoch = state['epoch']
    if current['last_completed_epoch'] == epoch:
        fingerprint = _digest(dict(epoch=stage_report['epoch'], update=stage_report['update'],
                                   rows=stage_report['rows'], metrics=stage_report['metrics']))
        if current['last_report_sha256'] != fingerprint or stage_report['update'] != state['updates']:
            raise ValueError('An already committed stage validation cannot change')
        return copy.deepcopy(current['last_decision'])
    if (state.get('phase') != 'validation' or current['epoch_plan']['epoch'] != epoch
            or current['last_completed_epoch'] != epoch - 1 or not 1 <= epoch <= 40
            or state.get('position') != len(train_ids)):
        raise ValueError('Curriculum advances only after the complete current training epoch')
    seen = _seen(state, arm, train_ids)
    _preserved_seen(current, seen)
    rows = state.get('train_rows')
    plan_keys = ['P', *current['epoch_plan']['keys']]
    if (not isinstance(rows, list) or len(rows) != len(train_ids)
            or sorted(row['sample_index'] for row in rows) != sorted(train_ids)
            or any(row.get('candidate_keys') != plan_keys for row in rows)
            or any(not set(plan_keys[1:]) <= set(seen[str(i)]) for i in train_ids)):
        raise ValueError('Current epoch lacks exact successful training-key coverage')
    if current['last_completed_update'] is not None and state['updates'] <= current['last_completed_update']:
        raise ValueError('Repeated unchanged weights cannot satisfy consecutive stage evaluations')
    metrics, report_sha = _stage_report(stage_report, current, state, validation_ids)
    next_state = copy.deepcopy(current)
    old_pool = list(current['pool_keys'])
    added = []
    qualifies = False
    coverage = all(set(old_pool) <= set(seen[str(i)]) for i in train_ids)
    if current['mode'] == 'LEGACY':
        pool = (_sort_keys(set().union(*map(set, seen.values()))) if arm in ADAPTIVE
                else _legacy_keys(arm, 1))
        if len(pool) < 7:
            raise ValueError('Cumulative adoption cannot discard the completed legacy epoch')
        next_state.update(mode='ADAPTIVE' if arm in ADAPTIVE else 'FIXED',
            stage_id=1, pool_keys=pool, stage_start_epoch=epoch + 1,
            completed_stage_epochs=0, gate_streak=0, pool_cursor=0)
        action = 'begin_cumulative_pool' if arm in ADAPTIVE else 'begin_fixed_stage_diagnostics'
    else:
        next_state['completed_stage_epochs'] += 1
        qualifies = (metrics['mrr'] >= current['policy']['stage_mrr_threshold'] and
                     metrics['top1'] >= current['policy']['stage_top1_threshold'])
        next_state['gate_streak'] = current['gate_streak'] + 1 if qualifies else 0
        next_state['pool_cursor'] = current['epoch_plan']['cursor_after']
        action = 'retain_pool'
        if (arm in ADAPTIVE and epoch < 40 and len(old_pool) < 128
                and next_state['completed_stage_epochs'] >= 2
                and next_state['gate_streak'] >= 2 and coverage):
            added = [f'U:{i}' for i in range(128) if f'U:{i}' not in old_pool][:7]
            next_state.update(stage_id=current['stage_id'] + 1,
                pool_keys=_sort_keys([*old_pool, *added]), stage_start_epoch=epoch + 1,
                completed_stage_epochs=0, gate_streak=0)
            # Preserve the next old candidate when inserting previously unseen keys.
            next_old = old_pool[current['epoch_plan']['cursor_after']]
            next_state['pool_cursor'] = next_state['pool_keys'].index(next_old)
            action = 'expand_pool'
    next_state['epoch_plan'] = _plan(epoch + 1, next_state['stage_id'], next_state['mode'],
        next_state['pool_keys'], next_state['pool_cursor'], added)
    decision = dict(epoch=epoch, update=state['updates'], action=action,
        evaluated_stage_id=current['stage_id'], evaluated_mode=current['mode'],
        evaluated_pool_keys=old_pool, metrics=metrics, thresholds_met=qualifies,
        successful_active_pool_coverage=coverage, added_keys=added,
        next_stage_id=next_state['stage_id'], next_pool_keys=list(next_state['pool_keys']),
        next_epoch_plan=copy.deepcopy(next_state['epoch_plan']),
        full129_best_unchanged=True, production_epochs_unchanged=40,
        complete_native_bank_active=len(next_state['pool_keys']) == 128)
    next_state.update(last_completed_epoch=epoch, last_completed_update=state['updates'],
                      last_report_sha256=report_sha, last_decision=decision,
                      last_successful_seen=copy.deepcopy(seen))
    next_state['events'].append(copy.deepcopy(decision))
    state['comparison_curriculum'] = next_state
    return copy.deepcopy(decision)
