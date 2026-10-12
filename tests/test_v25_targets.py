"""CPU UNIT metadata/controller checks; no real model predictions or training.

The explicitly synthetic report fixtures represent the entire required65/21
population. Their metric values exercise gates, never performance claims.
"""
from __future__ import annotations

import copy
import json
import math
import unittest

from hiercp_v1x.v25_targets import (
    COMMON_POLICY, FORMAT, POLICY_FIELDS, finish_curriculum_epoch,
    new_curriculum, validate_policy,
)


TRAIN_CASES = [f'CPU_UNIT_TRAIN_{index:03d}' for index in range(65)]
VAL_CASES = [f'CPU_UNIT_VALIDATION_{index:03d}' for index in range(21)]
ARMS = ((8, 8, 31, 33), (16, 16, 15, 29), (32, 32, 7, 13), (128, None, 1, 1))


def policy(initial=8, increment=8):
    return dict(COMMON_POLICY, initial_u=initial, increment_u=increment)


def report(epoch, active_u, phase, metrics=(.6, .4), *, pair_loss=.2):
    train = phase == 'train_probe'
    case_ids = TRAIN_CASES if train else VAL_CASES
    # Complete P527 over65 ranking patients; P135 over16 of21 validation
    # patients. The last five validation patients retain zero-P rows.
    p_counts = ([15] + [8]*64) if train else ([15] + [8]*15 + [0]*5)
    rows = [dict(case_id=case, observed_P=p, unobserved_U=active_u,
                 per_P=dict(mrr=metrics[0], top1=metrics[1], pair_loss=pair_loss) if p else None)
            for case, p in zip(case_ids, p_counts)]
    ranking = sum(p > 0 for p in p_counts)
    return dict(CPU_UNIT_SYNTHETIC_METADATA=True, epoch=epoch, active_u=active_u,
        phase=phase, evaluation_view_epoch=29, all_P_scored=True, cases=rows,
        denominators=dict(patients=len(rows), ranking_patients=ranking,
            zero_P_patients=len(rows)-ranking, observed_P=sum(p_counts),
            unobserved_U=len(rows)*active_u, P_U_pairs=sum(p_counts)*active_u),
        metrics=dict(per_P_patient_mrr=metrics[0], per_P_patient_top1=metrics[1],
            patient_balanced_pair_loss=pair_loss))


def step(state, *, val=(.6, .4), train=(.7, .5)):
    epoch, active = state['last_completed_epoch'] + 1, state['active_u']
    return finish_curriculum_epoch(state, report(epoch, active, 'train_probe', train),
        report(epoch, active, 'stage_validation', val), epoch=epoch,
        actual_train_cases=TRAIN_CASES, expected_train_cases=TRAIN_CASES,
        updates_in_epoch=17)


def run_arm(initial, increment, *, mastered=False):
    state, trained = new_curriculum(policy(initial, increment)), []
    for _ in range(40):
        trained.append(state['active_u'])
        state, _ = step(state, val=(.8, .7) if mastered else (.6, .4))
    return state, trained


class V25PolicyCPUUnit(unittest.TestCase):
    def test_closed_four_arm_schedule_proofs_match_confirmed_plan(self):
        for initial, increment, earliest, latest in ARMS:
            with self.subTest(initial=initial):
                proof = validate_policy(policy(initial, increment))
                self.assertEqual(proof['earliest_first_full_bank_epoch'], earliest)
                self.assertEqual(proof['latest_first_full_bank_epoch'], latest)
                self.assertEqual(proof['guaranteed_full_bank_epochs'], 41-latest)
                self.assertEqual(proof['possible_first_full_bank_epochs'], list(range(earliest, latest+1)))
                self.assertEqual(proof['full_bank_start_deadline_epoch'], 33)
                self.assertEqual(len(proof['earliest_active_u_by_epoch']), 40)
                self.assertEqual(len(proof['worst_case_active_u_by_epoch']), 40)
                self.assertEqual(set(policy(initial, increment)), set(POLICY_FIELDS))
        proof = validate_policy(policy())
        self.assertEqual(proof['latest_current_stage_end_by_u']['8'], 4)
        self.assertEqual(proof['latest_current_stage_end_by_u']['16'], 6)
        self.assertEqual(proof['latest_current_stage_end_by_u']['120'], 32)
        self.assertIsNone(proof['latest_current_stage_end_by_u']['128'])

    def test_no_missing_extra_unapproved_arm_or_changed_gate_is_admitted(self):
        variants = [None, {}, dict(policy(), hidden_subset=True)]
        for field in POLICY_FIELDS:
            bad = policy(); del bad[field]; variants.append(bad)
        for field, value in (
                ('initial_u', 7), ('increment_u', 7), ('increment_u', None),
                ('total_u', 64), ('total_epochs', 41), ('minimum_stage_epochs', 1),
                ('maximum_stage_epochs', 5), ('minimum_full_bank_epochs', 7),
                ('consecutive_passes', 1), ('plateau_patience', 1),
                ('plateau_min_delta', .006), ('validation_mrr', .79),
                ('validation_top1', .69), ('train_mrr', .89), ('train_top1', .84),
                ('overfit_mrr_gap', .09), ('minimum_stage_epochs', True),
                ('validation_mrr', float('nan')), ('train_top1', True),
                ('train_mrr', float('inf'))):
            variants.append(dict(policy(), **{field: value}))
        variants.extend((policy(128, 16), policy(16, 8), policy(32, 16)))
        for bad in variants:
            with self.subTest(policy=bad):
                with self.assertRaises(ValueError):
                    validate_policy(bad)

    def test_constructor_and_proof_have_no_mutable_alias_to_policy(self):
        chosen = policy(); proof = validate_policy(chosen); state = new_curriculum(chosen)
        chosen['initial_u'] = 7
        self.assertEqual(proof['policy']['initial_u'], 8)
        self.assertEqual(state['policy']['initial_u'], 8)
        self.assertEqual(state['format'], FORMAT)
        self.assertIsNone(state['first_full_bank_training_epoch'])
        self.assertEqual(state['full_bank_training_epochs'], 0)


class V25ScheduleCPUUnit(unittest.TestCase):
    def test_A_worst_case_really_trains_full128_from33_without_skips(self):
        state, trained = run_arm(8, 8)
        expected = [8]*4 + [u for u in range(16, 128, 8) for _ in range(2)] + [128]*8
        self.assertEqual(trained, expected)
        self.assertEqual(state['last_completed_epoch'], 40)
        self.assertEqual(state['first_full_bank_training_epoch'], 33)
        self.assertEqual(state['full_bank_training_epochs'], 8)
        self.assertEqual(state['full_bank_optimizer_updates'], 136)
        self.assertEqual(state['completed_optimizer_updates'], 680)
        self.assertEqual(state['actual_P_U_pairs'], 527*sum(expected))
        self.assertEqual(len(state['history']), 40)
        for row in state['history']:
            self.assertEqual(row['actual_train_patients'], 65)
            self.assertEqual(row['actual_train_P'], 527)
            self.assertEqual(row['stage_validation_patients'], 21)
            self.assertEqual(row['stage_validation_P'], 135)
            self.assertEqual(row['stage_validation_zero_P_patients'], 5)
            self.assertEqual(row['optimizer_updates_in_epoch'], 17)
            if row['expanded']:
                self.assertEqual(row['next_active_u']-row['previous_active_u'], 8)
                self.assertGreaterEqual(row['stage_epochs'], 2)
                self.assertLessEqual(row['stage_epochs'], 4)
                self.assertTrue(row['coverage_is_not_mastery'])
                self.assertFalse(row['validation_gate_passed'])
                self.assertFalse(row['train_mastery_overfit_plateau_gate_passed'])
        at4, at6 = state['history'][3], state['history'][5]
        self.assertTrue(at4['gates']['stage_max4_deadline'])
        self.assertTrue(at4['gates']['remaining_budget_deadline'])
        self.assertEqual(at4['reason'], 'stage_max4_deadline')
        self.assertFalse(at6['gates']['stage_max4_deadline'])
        self.assertTrue(at6['gates']['remaining_budget_deadline'])
        self.assertEqual(at6['reason'], 'remaining_budget_deadline')
        self.assertIsNone(state['history'][31]['first_full_bank_training_epoch'])
        self.assertEqual(state['history'][31]['next_active_u'], 128)
        self.assertEqual(state['history'][32]['first_full_bank_training_epoch'], 33)
        for u in range(8, 129, 8):
            self.assertEqual(state['u_training_epochs'][str(u)], expected.count(u))
            self.assertEqual(state['u_training_updates'][str(u)], 17*expected.count(u))
            self.assertEqual(state['u_P_U_pairs'][str(u)], 527*u*expected.count(u))

    def test_all_arms_actual_earliest_and_latest_full_exposure(self):
        for initial, increment, earliest, latest in ARMS:
            for mastered, first in ((False, latest), (True, earliest)):
                with self.subTest(initial=initial, mastered=mastered):
                    state, trained = run_arm(initial, increment, mastered=mastered)
                    self.assertEqual(trained.index(128)+1, first)
                    self.assertEqual(state['first_full_bank_training_epoch'], first)
                    self.assertEqual(state['full_bank_training_epochs'], 41-first)
                    self.assertEqual(state['completed_optimizer_updates'], 680)
                    self.assertEqual(state['actual_P_U_pairs'], 527*sum(trained))
                    proof_key = 'earliest_active_u_by_epoch' if mastered else 'worst_case_active_u_by_epoch'
                    self.assertEqual(trained, state[proof_key])
                    for row in state['history']:
                        if row['expanded']:
                            self.assertEqual(row['next_active_u'], row['previous_active_u']+increment)
                            self.assertIn(row['stage_epochs'], (2, 3, 4))
                        if row['active_u'] == 128:
                            self.assertEqual(row['reason'], 'full_bank_replay')
                            self.assertFalse(row['expanded'])
                            self.assertFalse(any(row['gates'].values()))

    def test_full_arm_all_gate_outcomes_replay_all128_for_forty_epochs(self):
        state = new_curriculum(policy(128, None))
        for epoch in range(1, 41):
            state, receipt = step(state, val=(.95, .9) if epoch%2 else (.3, .1), train=(.99, .98))
            self.assertEqual(receipt['previous_active_u'], 128)
            self.assertEqual(receipt['next_active_u'], 128)
            self.assertEqual(receipt['reason'], 'full_bank_replay')
        self.assertEqual(state['full_bank_training_epochs'], 40)
        self.assertEqual(state['full_bank_optimizer_updates'], 680)


class V25GateCPUUnit(unittest.TestCase):
    def test_validation_mastery_has_exact_thresholds_consecutive2_and_minimum2(self):
        state = new_curriculum(policy())
        state, one = step(state, val=(.8, .7))
        self.assertFalse(one['minimum_stage_gate_passed'])
        self.assertFalse(one['expanded'])
        state, two = step(state, val=(.8, .7))
        self.assertEqual(two['reason'], 'validation_mastery')
        self.assertEqual(state['active_u'], 16)
        self.assertFalse(two['full_validation_used_for_curriculum'])
        self.assertFalse(two['coverage_is_not_mastery'])
        for val in ((.79999, .7), (.8, .69999)):
            bad = new_curriculum(policy())
            for _ in range(2): bad, row = step(bad, val=val)
            self.assertFalse(row['expanded'])
            self.assertEqual(row['validation_pass_streak'], 0)

    def test_failed_validation_epoch_resets_streak(self):
        state = new_curriculum(policy())
        state, _ = step(state, val=(.9, .8))
        state, row = step(state, val=(.79, .7))
        self.assertEqual(row['validation_pass_streak'], 0)
        state, row = step(state, val=(.9, .8))
        self.assertEqual(row['validation_pass_streak'], 1)
        self.assertFalse(row['expanded'])

    def test_train_mastery_requires_mrr_top1_gap_and_two_plateau_epochs(self):
        state = new_curriculum(policy())
        for _ in range(2): state, row = step(state, train=(.9, .85), val=(.8, .69))
        self.assertFalse(row['expanded'])
        self.assertEqual(row['plateau_epochs'], 1)
        state, row = step(state, train=(.9, .85), val=(.8, .69))
        self.assertEqual(row['reason'], 'train_mastery_overfit_plateau')
        self.assertTrue(row['overfit_mrr_gap_gate_passed'])
        self.assertAlmostEqual(row['train_minus_stage_val_mrr'], .1)
        self.assertEqual(row['plateau_epochs'], 2)
        for train, val in (((.89999, .85), (.6, .4)),
                           ((.95, .84999), (.6, .4)),
                           ((.9, .85), (.80001, .69))):
            bad = new_curriculum(policy())
            for _ in range(3): bad, row = step(bad, train=train, val=val)
            self.assertFalse(row['expanded'])
            self.assertFalse(row['train_mastery_overfit_plateau_gate_passed'])

    def test_plateau_reference_requires_more_than_delta_in_either_metric(self):
        for val in ((.605, .4), (.6, .405)):
            state = new_curriculum(policy())
            state, _ = step(state)
            state, row = step(state, val=val)
            self.assertEqual(row['plateau_epochs'], 1)
            self.assertFalse(row['stage_validation_improved'])
        for val in ((.606, .4), (.6, .406)):
            state = new_curriculum(policy())
            state, _ = step(state); state, _ = step(state)
            state, row = step(state, val=val, train=(.95, .9))
            self.assertEqual(row['plateau_epochs'], 0)
            self.assertTrue(row['stage_validation_improved'])
            self.assertFalse(row['expanded'])
        state = new_curriculum(policy())
        state, _ = step(state); state, _ = step(state, val=(.604, .404))
        state, row = step(state, val=(.608, .408))
        self.assertTrue(row['stage_validation_improved'])
        self.assertEqual(row['plateau_epochs'], 0)

    def test_precedence_preserves_every_simultaneously_true_gate(self):
        state = new_curriculum(policy())
        for val in ((.8, .7), (.799, .699), (.8, .7)):
            state, _ = step(state, val=val)
        state, row = step(state, val=(.8, .7), train=(.9, .85))
        self.assertTrue(all(row['gates'].values()))
        self.assertEqual(row['reason'], 'validation_mastery')
        self.assertFalse(row['coverage_is_not_mastery'])
        state = new_curriculum(policy())
        for _ in range(3): state, _ = step(state)
        state, row = step(state, train=(.95, .9))
        self.assertTrue(row['gates']['train_mastery_overfit_plateau'])
        self.assertTrue(row['gates']['stage_max4_deadline'])
        self.assertTrue(row['gates']['remaining_budget_deadline'])
        self.assertEqual(row['reason'], 'train_mastery_overfit_plateau')

    def test_each_expansion_resets_stage_counters_and_reference(self):
        for mode, count in (('validation', 2), ('overfit', 3), ('deadline', 4)):
            state = new_curriculum(policy())
            for _ in range(count):
                state, row = step(state, val=(.8, .7) if mode=='validation' else (.6, .4),
                    train=(.95, .9) if mode=='overfit' else (.7, .5))
            self.assertTrue(row['expanded'])
            self.assertEqual(state['stage_epochs'], 0)
            self.assertEqual(state['stage_started_epoch'], count+1)
            self.assertEqual(state['validation_pass_streak'], 0)
            self.assertEqual(state['plateau_epochs'], 0)
            self.assertIsNone(state['plateau_reference'])
            state, row = step(state)
            self.assertEqual(row['stage_epochs'], 1)
            self.assertEqual(row['plateau_epochs'], 0)
            self.assertFalse(row['expanded'])
        state = new_curriculum(policy())
        for _ in range(6): state, row = step(state)
        self.assertEqual(row['reason'], 'remaining_budget_deadline')
        self.assertEqual(state['stage_started_epoch'], 7)
        self.assertIsNone(state['plateau_reference'])


class V25CoverageAndResumeCPUUnit(unittest.TestCase):
    def inputs(self, state=None):
        state = new_curriculum(policy()) if state is None else state
        epoch = state['last_completed_epoch']+1
        return dict(curriculum=state, train_probe=report(epoch, state['active_u'], 'train_probe'),
            stage_validation=report(epoch, state['active_u'], 'stage_validation'), epoch=epoch,
            actual_train_cases=list(TRAIN_CASES), expected_train_cases=list(TRAIN_CASES), updates_in_epoch=17)

    def assert_rejected(self, inputs):
        before = copy.deepcopy(inputs['curriculum'])
        with self.assertRaises(ValueError): finish_curriculum_epoch(**inputs)
        self.assertEqual(inputs['curriculum'], before)

    def test_missing_zero_nonfinite_boolean_or_mismatched_metrics_never_fallback(self):
        for phase in ('train_probe', 'stage_validation'):
            for mutation in ('missing_report', 'missing_metrics', 'missing_mrr', 'zero_mrr',
                             'nan_mrr', 'inf_loss', 'bool_top1', 'negative_loss', 'mismatched_average'):
                with self.subTest(phase=phase, mutation=mutation):
                    inputs = self.inputs(); probe = inputs[phase]
                    if mutation == 'missing_report': inputs[phase] = None
                    elif mutation == 'missing_metrics': del probe['metrics']
                    elif mutation == 'missing_mrr': del probe['metrics']['per_P_patient_mrr']
                    elif mutation == 'zero_mrr': probe['metrics']['per_P_patient_mrr'] = 0
                    elif mutation == 'nan_mrr': probe['metrics']['per_P_patient_mrr'] = float('nan')
                    elif mutation == 'inf_loss': probe['metrics']['patient_balanced_pair_loss'] = float('inf')
                    elif mutation == 'bool_top1': probe['metrics']['per_P_patient_top1'] = False
                    elif mutation == 'negative_loss': probe['metrics']['patient_balanced_pair_loss'] = -.1
                    else: probe['cases'][0]['per_P']['mrr'] = .7
                    self.assert_rejected(inputs)

    def test_true_zero_top1_and_pair_loss_are_not_fabricated_missing_metrics(self):
        inputs = self.inputs()
        for phase in ('train_probe', 'stage_validation'):
            inputs[phase] = report(1, 8, phase, metrics=(1/9, 0), pair_loss=0)
        state, row = finish_curriculum_epoch(**inputs)
        self.assertFalse(row['expanded'])
        self.assertEqual(state['last_completed_epoch'], 1)

    def test_probe_stage_view_phase_and_allP_contract_are_required(self):
        for phase in ('train_probe', 'stage_validation'):
            for field, value in (('epoch', 0), ('epoch', True), ('active_u', 128),
                                 ('phase', 'full_validation'), ('evaluation_view_epoch', 1),
                                 ('all_P_scored', False)):
                inputs = self.inputs(); inputs[phase][field] = value
                self.assert_rejected(inputs)

    def test_complete_training_patients_and_actual17_updates_are_required(self):
        for field in ('actual_train_cases', 'expected_train_cases'):
            for mutation in ('missing', 'duplicate', 'wrong_id'):
                inputs = self.inputs()
                if mutation == 'missing': inputs[field].pop()
                elif mutation == 'duplicate': inputs[field][-1] = inputs[field][0]
                else: inputs[field][-1] = 'CPU_UNIT_OTHER'
                self.assert_rejected(inputs)
        for value in (0, 1, 16, 18, True, 17.0, None):
            inputs = self.inputs(); inputs['updates_in_epoch'] = value
            self.assert_rejected(inputs)

    def test_every_P_U_and_zeroP_validation_row_remains_in_probe_coverage(self):
        for phase in ('train_probe', 'stage_validation'):
            for mutation in ('missing_patient', 'duplicate_patient', 'missing_P', 'missing_U',
                             'missing_denominators', 'zero_denominator', 'missing_perP'):
                with self.subTest(phase=phase, mutation=mutation):
                    inputs = self.inputs(); probe = inputs[phase]
                    if mutation == 'missing_patient': probe['cases'].pop()
                    elif mutation == 'duplicate_patient': probe['cases'][-1]['case_id'] = probe['cases'][0]['case_id']
                    elif mutation == 'missing_P': probe['cases'][0]['observed_P'] -= 1
                    elif mutation == 'missing_U': probe['cases'][0]['unobserved_U'] -= 1
                    elif mutation == 'missing_denominators': del probe['denominators']
                    elif mutation == 'zero_denominator': probe['denominators']['observed_P'] = 0
                    else: del probe['cases'][0]['per_P']
                    self.assert_rejected(inputs)
        inputs = self.inputs(); inputs['stage_validation']['cases'][-1]['per_P'] = dict(mrr=0, top1=0, pair_loss=0)
        self.assert_rejected(inputs)
        inputs = self.inputs(); inputs['stage_validation']['denominators']['ranking_patients'] = 21
        self.assert_rejected(inputs)

    def test_validation_split_is_disjoint_and_bound_after_first_complete_epoch(self):
        inputs = self.inputs(); inputs['stage_validation']['cases'][0]['case_id'] = TRAIN_CASES[0]
        self.assert_rejected(inputs)
        state, _ = step(new_curriculum(policy()))
        inputs = self.inputs(state); inputs['stage_validation']['cases'][0]['case_id'] = 'CPU_UNIT_OTHER_VALIDATION'
        self.assert_rejected(inputs)
        inputs = self.inputs(state)
        for field in ('actual_train_cases', 'expected_train_cases'):
            inputs[field][0] = 'CPU_UNIT_OTHER_TRAIN'
        inputs['train_probe']['cases'][0]['case_id'] = 'CPU_UNIT_OTHER_TRAIN'
        self.assert_rejected(inputs)

    def test_success_never_mutates_input_state_or_reports(self):
        inputs = self.inputs(); before = copy.deepcopy(inputs)
        state, _ = finish_curriculum_epoch(**inputs)
        self.assertEqual(inputs, before)
        self.assertEqual(state['last_completed_epoch'], 1)

    def test_json_resume_is_deterministic_and_cannot_double_count(self):
        uninterrupted = new_curriculum(policy())
        resumed = new_curriculum(policy())
        for epoch in range(1, 41):
            val = (.8, .7) if epoch%3 == 0 else (.6, .4)
            train = (.95, .9) if epoch%2 == 0 else (.7, .5)
            uninterrupted, receipt = step(uninterrupted, val=val, train=train)
            resumed = json.loads(json.dumps(resumed))
            resumed, resumed_receipt = step(resumed, val=val, train=train)
            self.assertEqual(receipt, resumed_receipt)
            self.assertEqual(uninterrupted, resumed)
        self.assertEqual(resumed['completed_optimizer_updates'], 680)
        inputs = self.inputs(resumed)
        self.assert_rejected(inputs)
        inputs['epoch'] = 40
        self.assert_rejected(inputs)

    def test_resume_rejects_jumps_omitted_minimum_stays_and_tampered_history(self):
        state = new_curriculum(policy())
        for _ in range(5): state, _ = step(state)
        mutations = [('active_u', 32), ('stage_epochs', 0), ('last_completed_epoch', 4),
                     ('stage_started_epoch', 6), ('validation_pass_streak', 2),
                     ('plateau_epochs', 5), ('completed_optimizer_updates', 84),
                     ('full_bank_training_epochs', 1), ('latest_first_full_bank_epoch', 34),
                     ('policy_sha256', '0'*64), ('format', 'v24_mastery_overfit_and_coverage_curriculum_v1')]
        for field, value in mutations:
            inputs = self.inputs(copy.deepcopy(state)); inputs['curriculum'][field] = value
            self.assert_rejected(inputs)
        for field, value in (('next_active_u', 32), ('stage_epochs', 1),
                             ('reason', 'validation_mastery'), ('optimizer_updates_in_epoch', 16)):
            inputs = self.inputs(copy.deepcopy(state))
            inputs['curriculum']['history'][3][field] = value
            self.assert_rejected(inputs)
        inputs = self.inputs(copy.deepcopy(state))
        inputs['curriculum']['history'][3]['gates']['remaining_budget_deadline'] = False
        self.assert_rejected(inputs)
        inputs = self.inputs(copy.deepcopy(state)); inputs['curriculum']['history'].pop()
        self.assert_rejected(inputs)
        inputs = self.inputs(); inputs['epoch'] = 2
        self.assert_rejected(inputs)


if __name__ == '__main__':
    unittest.main()
