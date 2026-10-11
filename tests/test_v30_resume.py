"""CPU UNIT state admission only; synthetic65/21 reports are not training.

No model, optimizer, patient CT, output namespace or production run is used.
Fixtures retain the full40/680/curriculum contract to test checkpoint boundaries.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest

from hiercp_v1x.v25_targets import new_curriculum, finish_curriculum_epoch
from hiercp_v1x.v30_training import (validate_curriculum_progress, _validate_progress,
    patient_batches, best_selection_key)
from tests.test_v25_targets import policy, report, TRAIN_CASES, VAL_CASES


def owned_report(epoch, active, role, *, metrics=(.6, .4)):
    value = report(epoch, active, role, metrics)
    value.update(updates=17*epoch, debug=False, recipient_GT_used_in_forward=False)
    return value


def fresh_state(initial=8, increment=8):
    return dict(CPU_UNIT_SYNTHETIC_STATE=True, epoch=1, phase='initial_full_validation',
        updates=0, attempts=0, overflows=0, history=[], best=None, train_order=None,
        train_position=0, train_rows=[], evaluation_position=0, evaluation_rows=[],
        epoch_start_updates=0, curriculum=new_curriculum(policy(initial, increment)))


def after_initial(state):
    state['initial_validation'] = owned_report(0, 128, 'initial_full_validation')
    state['phase'] = 'training'
    return state


def train_batches(state, count=17):
    active = state['curriculum']['active_u']
    all_rows = {row['case_id']: row for row in owned_report(state['epoch'], active, 'train_probe')['cases']}
    state['train_order'] = list(reversed(TRAIN_CASES))
    state['epoch_start_updates'] = len(state['history'])*17
    complete = patient_batches(state['train_order'], 4)[:count]
    state['train_rows'] = [copy.deepcopy(all_rows[case]) for batch in complete for case in batch]
    state['train_position'] = len(state['train_rows'])
    state['updates'] = state['epoch_start_updates']+len(complete)
    state['attempts'] = state['updates']+state['overflows']
    return state


def pending_epoch(state, *, full_metrics=(.6, .4)):
    train_batches(state)
    epoch, active = state['epoch'], state['curriculum']['active_u']
    probe = owned_report(epoch, active, 'train_probe')
    stage = owned_report(epoch, active, 'stage_validation')
    full = owned_report(epoch, 128, 'full_validation', metrics=full_metrics)
    curriculum, transition = finish_curriculum_epoch(state['curriculum'], probe, stage,
        epoch=epoch, actual_train_cases=[row['case_id'] for row in state['train_rows']],
        expected_train_cases=TRAIN_CASES, updates_in_epoch=17)
    key = best_selection_key(full)
    if state['best'] is None or key > tuple(state['best']['selection_key']):
        state['best'] = dict(epoch=epoch, updates=state['updates'], selection_key=list(key),
            selected_by='fixed_full128_validation_only')
    state['pending_epoch_completion'] = dict(train_probe=probe, stage_validation=stage,
        full_validation=full, curriculum=curriculum, transition=transition)
    state['phase'] = 'epoch_completion'
    return state


def finish_epoch(state):
    pending = state.pop('pending_epoch_completion')
    state['history'].append(dict(epoch=state['epoch'], update=state['updates'],
        train=owned_report(state['epoch'], state['curriculum']['active_u'], 'train_probe'),
        train_probe=pending['train_probe'], stage_validation=pending['stage_validation'],
        full_validation=pending['full_validation'], curriculum_transition=pending['transition']))
    state.update(epoch=state['epoch']+1, phase='training', curriculum=pending['curriculum'],
        train_order=None, train_position=0, train_rows=[])
    return state


def completed_state(epochs=40, initial=8, increment=8):
    state = after_initial(fresh_state(initial, increment))
    for _ in range(epochs):
        pending_epoch(state); finish_epoch(state)
    if epochs == 40:
        state['phase'] = 'complete'
    return state


def validate(state):
    return validate_curriculum_progress(state, policy=state['curriculum']['policy'],
        train_cases=TRAIN_CASES, val_cases=VAL_CASES, physical_patient_batch=4)


class V30ResumeCPUUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = completed_state()
        cls.partial = completed_state(2)

    def assert_refused(self, state):
        with self.assertRaises(ValueError):
            validate(state)

    def test_fresh_initial_partial_initial_validation_and_after_initial_are_admitted(self):
        state = fresh_state()
        self.assertEqual(validate(state)['completed_epochs'], 0)
        state['evaluation_rows'] = owned_report(0, 128, 'initial_full_validation')['cases'][:4]
        state['evaluation_position'] = 4
        validate(state)
        state.update(evaluation_rows=[], evaluation_position=0)
        after_initial(state)
        validate(state)
        # Initial full128 scores are deliberately excluded from trained BEST.
        state['best'] = dict(epoch=0, updates=0, selection_key=[.6, .4, -.2],
            selected_by='fixed_full128_validation_only')
        self.assert_refused(state)

    def test_complete40_680_worst_A_replay_and_all_four_arms_are_admitted(self):
        summary = validate(self.full)
        self.assertEqual(summary['optimizer_updates'], 680)
        self.assertEqual(summary['active_u'], 128)
        self.assertEqual(summary['full_bank_training_epochs'], 8)
        for initial, increment in ((16, 16), (32, 32), (128, None)):
            with self.subTest(initial=initial):
                validate(completed_state(initial=initial, increment=increment))

    def test_partial_training_counts_real_batches_and_rebalanced_tail(self):
        for batches, patients in ((0, 0), (1, 4), (15, 60), (16, 62), (17, 65)):
            with self.subTest(batches=batches):
                state = train_batches(copy.deepcopy(self.partial), batches)
                self.assertEqual(state['train_position'], patients)
                self.assertEqual(validate(state)['optimizer_updates'], 34+batches)
        state = train_batches(copy.deepcopy(self.partial), 1)
        state['updates'] += 1; state['attempts'] += 1
        self.assert_refused(state)
        state = train_batches(copy.deepcopy(self.partial), 1)
        state['train_position'] = 3; state['train_rows'].pop()
        self.assert_refused(state)

    def test_explicit_amp_overflow_retry_is_admitted_without_counting_as_an_update(self):
        state = train_batches(copy.deepcopy(self.partial), 3)
        state['overflows'] = 2; state['attempts'] += 2
        self.assertEqual(validate(state)['optimizer_updates'], 37)
        state['attempts'] += 1
        self.assert_refused(state)

    def test_train_stage_full_evaluation_partial_boundaries_are_admitted(self):
        for phase in ('train_probe', 'stage_validation', 'full_validation'):
            with self.subTest(phase=phase):
                state = train_batches(copy.deepcopy(self.partial))
                state['phase'] = phase
                for role in ('train_probe', 'stage_validation'):
                    if phase == 'full_validation' or (phase == 'stage_validation' and role == 'train_probe'):
                        state[role] = owned_report(3, state['curriculum']['active_u'], role)
                state['evaluation_rows'] = owned_report(3, 128 if phase == 'full_validation' else
                    state['curriculum']['active_u'], phase)['cases'][:4]
                state['evaluation_position'] = 4
                validate(state)
                state['evaluation_rows'].reverse()
                self.assert_refused(state)

    def test_pending_transition_is_validated_once_before_outer_history_commit(self):
        state = pending_epoch(copy.deepcopy(self.partial))
        self.assertEqual(validate(state)['pending_epoch'], True)
        self.assertEqual(state['curriculum']['last_completed_epoch'], 2)
        self.assertEqual(state['pending_epoch_completion']['curriculum']['last_completed_epoch'], 3)
        for field in ('transition', 'curriculum', 'full_validation', 'train_probe'):
            changed = copy.deepcopy(state)
            if field == 'transition':
                changed['pending_epoch_completion'][field]['next_active_u'] += 8
            elif field == 'curriculum':
                changed['pending_epoch_completion'][field]['full_bank_training_epochs'] += 1
            elif field == 'full_validation':
                changed['pending_epoch_completion'][field]['active_u'] = 64
            else:
                changed['pending_epoch_completion'][field]['updates'] -= 1
            self.assert_refused(changed)
        state['phase'] = 'training'
        self.assert_refused(state)

    def test_prior_loose_complete_counter_reproduction_is_now_refused(self):
        state = copy.deepcopy(self.full)
        state['curriculum'] = new_curriculum(policy())
        # The pre-existing cursor/scheduler validator accepted this exact gap.
        _validate_progress(state, TRAIN_CASES, VAL_CASES, 4, 40, SimpleNamespace(last_epoch=40))
        self.assert_refused(state)
        state = copy.deepcopy(self.full)
        state['curriculum']['full_bank_training_epochs'] = 7
        self.assert_refused(state)

    def test_outer_history_reports_transitions_best_and_population_cannot_be_swapped(self):
        variants = []
        state = copy.deepcopy(self.full); state['history'][3]['update'] -= 1; variants.append(state)
        state = copy.deepcopy(self.full); state['history'][3]['curriculum_transition']['reason'] = 'validation_mastery'; variants.append(state)
        state = copy.deepcopy(self.full); state['history'][3]['train_probe']['metrics']['per_P_patient_mrr'] = .9; variants.append(state)
        state = copy.deepcopy(self.full); state['history'][3]['full_validation']['cases'].pop(); variants.append(state)
        state = copy.deepcopy(self.full); state['history'][3]['stage_validation']['cases'][-1]['per_P'] = {}; variants.append(state)
        state = copy.deepcopy(self.full); state['history'][3]['train']['cases'].pop(); variants.append(state)
        state = copy.deepcopy(self.full); state['best']['epoch'] = 40; variants.append(state)
        state = copy.deepcopy(self.full); state['updates'] = 679; state['attempts'] = 679; variants.append(state)
        state = copy.deepcopy(self.full); state['history'].pop(); variants.append(state)
        for index, state in enumerate(variants):
            with self.subTest(variant=index):
                self.assert_refused(state)
        with self.assertRaises(ValueError):
            validate_curriculum_progress(self.full, policy=policy(), train_cases=TRAIN_CASES,
                val_cases=['OTHER_'+case for case in VAL_CASES], physical_patient_batch=4)
        with self.assertRaises(ValueError):
            validate_curriculum_progress(self.full, policy=policy(), train_cases=TRAIN_CASES,
                val_cases=VAL_CASES, physical_patient_batch=8)

    def test_incomplete_probe_wrong_phase_stale_report_and_boolean_counters_are_refused(self):
        state = train_batches(copy.deepcopy(self.partial), 16); state['phase'] = 'train_probe'
        self.assert_refused(state)
        state = train_batches(copy.deepcopy(self.partial)); state['phase'] = 'full_validation'
        self.assert_refused(state)
        state = copy.deepcopy(self.partial); state['train_probe'] = owned_report(2, 8, 'train_probe')
        self.assert_refused(state)
        state = copy.deepcopy(self.partial); state['updates'] = True
        self.assert_refused(state)
        state = train_batches(copy.deepcopy(self.partial)); state['phase'] = 'train_probe'
        state['evaluation_rows'] = owned_report(3, state['curriculum']['active_u'], 'train_probe')['cases'][:4]
        state['evaluation_position'] = 4
        state['evaluation_rows'][0]['unobserved_U'] = 128
        self.assert_refused(state)

    def test_json_resume_is_deterministic_nonmutating_and_accepts_final_training_boundary(self):
        before = copy.deepcopy(self.full)
        restored = json.loads(json.dumps(self.full))
        self.assertEqual(validate(restored), validate(self.full))
        self.assertEqual(self.full, before)
        restored['phase'] = 'training'
        self.assertEqual(validate(restored)['completed_epochs'], 40)


if __name__ == '__main__':
    unittest.main()
