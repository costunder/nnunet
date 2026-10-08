"""CPU UNIT policy/state checks; no model or actual training is executed."""
from __future__ import annotations

import copy
import json
import unittest

from hiercp_v1x import comparison_curriculum as curriculum

TRAIN = [3, 7, 12]
VAL = [101, 102, 103, 104]
SHA = 'a' * 64


def legacy(arm, epoch):
    if arm == 'selected':
        return [f'S:{i}' for i in range(7)]
    start = 0 if arm == 'native_fixed' else (epoch - 1) * 7 % 128
    return [f'U:{(start + i) % 128}' for i in range(7)]


def original_state(arm='native', epoch=8, position=1):
    previous = sorted({key for e in range(1, epoch) for key in legacy(arm, e)})
    seen = {str(i): list(previous) for i in TRAIN}
    for i in TRAIN[:position]:
        seen[str(i)] = sorted(set(seen[str(i)]) | set(legacy(arm, epoch)))
    return dict(epoch=epoch, phase='training', position=position,
        updates=(epoch - 1) * len(TRAIN) + position, order=list(TRAIN),
        train_rows=[dict(sample_index=i, candidate_keys=['P', *legacy(arm, epoch)]) for i in TRAIN[:position]],
        seen_comparisons=seen, opaque_optimizer_marker={'preserve': [1, 2, 3]},
        best={'epoch': 2, 'selection_key': [.93, .89, -.02]})


def complete_training(state, arm):
    keys = curriculum.training_keys(state, arm, state['epoch'])
    state['updates'] += len(TRAIN) - state['position']
    state['position'] = len(TRAIN)
    state['phase'] = 'validation'
    state['train_rows'] = [dict(sample_index=i, candidate_keys=list(keys)) for i in TRAIN]
    for i in TRAIN:
        state['seen_comparisons'][str(i)] = sorted(set(state['seen_comparisons'][str(i)]) | set(keys[1:]))


def report(state, arm, *, good=True):
    mrr, top1 = (1., 1.) if good else (.5, 0.)
    rows = [dict(sample_index=i, case_id='patient_' + str(n // 2),
                 candidate_keys=curriculum.stage_keys(state, arm), mrr=mrr, top1=top1)
            for n, i in enumerate(VAL)]
    return dict(epoch=state['epoch'], update=state['updates'], rows=rows,
                metrics=dict(mrr=mrr, top1=top1), source_problems=4, patients=2)


def boundary(state, arm, *, good=True):
    complete_training(state, arm)
    stage = report(state, arm, good=good)
    decision = curriculum.finish_epoch(state, arm, stage, TRAIN, validation_ids=VAL)
    return decision, stage


def next_epoch(state):
    state.update(epoch=state['epoch'] + 1, phase='training', position=0,
                 order=list(TRAIN), train_rows=[])


def active_state(arm='native', epoch=8):
    state = original_state(arm, epoch)
    curriculum.adopt_or_validate(state, arm, TRAIN, SHA)
    boundary(state, arm)
    next_epoch(state)
    return state


class CurriculumStateUnit(unittest.TestCase):
    def test_policy_is_explicit_arm_specific_and_returns_an_independent_copy(self):
        for arm in curriculum.ARMS:
            p = curriculum.policy(arm)
            self.assertEqual(p['name'], curriculum.POLICY_NAME)
            self.assertEqual(p['adaptive'], arm in ('native', 'native_listwise'))
            self.assertEqual(p['training_comparisons_per_source'], 7)
            self.assertEqual(p['production_epochs'], 40)
            self.assertEqual(p['stage_mrr_threshold'], .90)
            self.assertEqual(p['stage_top1_threshold'], .85)
            self.assertEqual(len(p['policy_sha256']), 64)
            p['production_epochs'] = 2
            self.assertEqual(curriculum.policy(arm)['production_epochs'], 40)

    def test_mid_epoch_adoption_preserves_every_existing_core_field_and_plan(self):
        state = original_state()
        before = copy.deepcopy(state)
        receipt = curriculum.adopt_or_validate(state, 'native', TRAIN, SHA)
        self.assertEqual({k:v for k,v in state.items() if k != 'comparison_curriculum'}, before)
        self.assertEqual(curriculum.training_keys(state, 'native', 8), ['P', *legacy('native', 8)])
        self.assertEqual(curriculum.stage_keys(state, 'native'), ['P', *legacy('native', 8)])
        self.assertEqual(receipt['mode'], 'LEGACY')
        receipt['pool_keys'].clear()
        self.assertEqual(len(curriculum.stage_keys(state, 'native')), 8)

    def test_legacy_wrap_order_is_preserved_until_epoch_completion(self):
        state = original_state(epoch=19)
        curriculum.adopt_or_validate(state, 'native', TRAIN, SHA)
        self.assertEqual(curriculum.training_keys(state, 'native', 19),
                         ['P', 'U:126', 'U:127', 'U:0', 'U:1', 'U:2', 'U:3', 'U:4'])

    def test_adoption_rejects_unverified_sha_and_completed_new_migration(self):
        for checksum in (None, 'x', 'z' * 64):
            state = original_state()
            with self.assertRaises(ValueError):
                curriculum.adopt_or_validate(state, 'native', TRAIN, checksum)
            self.assertNotIn('comparison_curriculum', state)
        state = original_state(epoch=40)
        state.update(phase='complete', epoch=41)
        with self.assertRaisesRegex(ValueError, 'completed experiment'):
            curriculum.adopt_or_validate(state, 'native', TRAIN, SHA)

    def test_boundary_adopts_actual_union_without_inventing_per_source_exposure(self):
        state = original_state()
        state['seen_comparisons']['3'].append('U:100')
        curriculum.adopt_or_validate(state, 'native', TRAIN, SHA)
        decision, _ = boundary(state, 'native')
        current = state['comparison_curriculum']
        self.assertEqual(decision['action'], 'begin_cumulative_pool')
        self.assertEqual(len(current['pool_keys']), 57)
        self.assertIn('U:100', current['pool_keys'])
        self.assertNotIn('U:100', state['seen_comparisons']['7'])
        self.assertEqual(current['gate_streak'], 0)
        self.assertEqual(current['completed_stage_epochs'], 0)

    def test_plan_read_is_pure_and_cursor_advances_only_once_per_finished_epoch(self):
        state = active_state()
        before = copy.deepcopy(state)
        self.assertEqual(curriculum.training_keys(state, 'native', 9), ['P', *[f'U:{i}' for i in range(7)]])
        self.assertEqual(state, before)
        with self.assertRaises(ValueError):
            curriculum.training_keys(state, 'native', 10)
        decision, stage = boundary(state, 'native')
        after = copy.deepcopy(state)
        self.assertEqual(state['comparison_curriculum']['pool_cursor'], 7)
        self.assertEqual(curriculum.finish_epoch(state, 'native', stage, TRAIN, validation_ids=VAL), decision)
        self.assertEqual(state, after)
        stage['metrics']['mrr'] = .9
        with self.assertRaisesRegex(ValueError, 'already committed'):
            curriculum.finish_epoch(state, 'native', stage, TRAIN)

    def test_two_successful_stage_epochs_expand_and_prioritize_new_seven(self):
        for arm in ('native', 'native_listwise'):
            state = active_state(arm)
            best = copy.deepcopy(state['best'])
            decision, _ = boundary(state, arm)
            self.assertEqual(decision['action'], 'retain_pool')
            next_epoch(state)
            decision, _ = boundary(state, arm)
            self.assertEqual(decision['action'], 'expand_pool')
            self.assertEqual(decision['added_keys'], [f'U:{i}' for i in range(56, 63)])
            self.assertEqual(len(decision['next_pool_keys']), 63)
            next_epoch(state)
            self.assertEqual(curriculum.training_keys(state, arm, 11), ['P', *decision['added_keys']])
            boundary(state, arm)
            next_epoch(state)
            self.assertEqual(curriculum.training_keys(state, arm, 12), ['P', *[f'U:{i}' for i in range(14, 21)]])
            self.assertEqual(state['best'], best)

    def test_failed_gate_resets_consecutive_streak(self):
        state = active_state()
        boundary(state, 'native'); next_epoch(state)
        boundary(state, 'native', good=False); next_epoch(state)
        decision, _ = boundary(state, 'native')
        self.assertEqual(decision['action'], 'retain_pool')
        self.assertEqual(state['comparison_curriculum']['gate_streak'], 1)
        next_epoch(state)
        self.assertEqual(boundary(state, 'native')[0]['action'], 'expand_pool')

    def test_user_lowered_gate_accepts_point_nine_two_five_mrr_and_point_eight_five_top1(self):
        state = active_state()
        validation = list(range(101, 121))
        for observation in range(2):
            complete_training(state, 'native')
            keys = curriculum.stage_keys(state, 'native')
            rows = [dict(sample_index=index, case_id=f'patient_{index}', candidate_keys=keys,
                         mrr=1. if n < 17 else .5, top1=1. if n < 17 else 0.)
                    for n, index in enumerate(validation)]
            stage = dict(epoch=state['epoch'], update=state['updates'], rows=rows,
                         metrics=dict(mrr=sum(r['mrr'] for r in rows) / 20., top1=.85),
                         source_problems=20, patients=20)
            decision = curriculum.finish_epoch(state, 'native', stage, TRAIN, validation_ids=validation)
            self.assertTrue(decision['thresholds_met'])
            self.assertEqual(decision['action'], 'retain_pool' if observation == 0 else 'expand_pool')
            next_epoch(state)

    def test_union_key_missing_in_one_source_blocks_growth_without_erasing_history(self):
        state = original_state()
        state['seen_comparisons']['3'].append('U:100')
        curriculum.adopt_or_validate(state, 'native', TRAIN, SHA)
        boundary(state, 'native'); next_epoch(state)
        boundary(state, 'native'); next_epoch(state)
        decision, _ = boundary(state, 'native')
        self.assertTrue(decision['thresholds_met'])
        self.assertFalse(decision['successful_active_pool_coverage'])
        self.assertEqual(decision['action'], 'retain_pool')
        self.assertIn('U:100', decision['next_pool_keys'])

    def test_forward_or_overflow_without_successful_key_coverage_cannot_advance(self):
        state = active_state()
        complete_training(state, 'native')
        state['seen_comparisons']['7'].remove('U:0')
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ValueError, 'Historical successful'):
            curriculum.finish_epoch(state, 'native', report(state, 'native'), TRAIN)
        self.assertEqual(state, before)

    def test_rejects_partial_wrong_pool_or_inconsistent_stage_validation_atomically(self):
        for problem in ('partial', 'duplicate', 'pool', 'metric', 'nan', 'epoch', 'update', 'stage'):
            state = active_state()
            complete_training(state, 'native')
            stage = report(state, 'native')
            if problem == 'partial': stage['rows'].pop()
            if problem == 'duplicate': stage['rows'][0]['sample_index'] = VAL[1]
            if problem == 'pool': stage['rows'][0]['candidate_keys'] = ['P', 'U:0']
            if problem == 'metric': stage['metrics']['mrr'] = .8
            if problem == 'nan': stage['rows'][0]['mrr'] = float('nan')
            if problem == 'epoch': stage['epoch'] += 1
            if problem == 'update': stage['update'] -= 1
            if problem == 'stage': stage['stage_id'] = 999
            before = copy.deepcopy(state)
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                curriculum.finish_epoch(state, 'native', stage, TRAIN, validation_ids=VAL)
            self.assertEqual(state, before)

    def test_metrics_use_patient_macro_not_flat_source_average(self):
        state = active_state()
        complete_training(state, 'native')
        stage = report(state, 'native')
        for row in stage['rows'][:3]: row['case_id'] = 'many'
        stage['rows'][3].update(case_id='one', mrr=.5, top1=0.)
        stage['metrics'] = dict(mrr=.875, top1=.75)
        with self.assertRaisesRegex(ValueError, 'patient-macro'):
            curriculum.finish_epoch(state, 'native', stage, TRAIN, validation_ids=VAL)
        stage['metrics'] = dict(mrr=.75, top1=.5)
        self.assertEqual(curriculum.finish_epoch(state, 'native', stage, TRAIN)['metrics'], stage['metrics'])

    def test_corrupted_plan_policy_and_cohort_are_rejected_on_resume(self):
        for problem in ('plan', 'policy', 'cohort', 'pool'):
            state = active_state()
            if problem == 'plan': state['comparison_curriculum']['epoch_plan']['keys'].reverse()
            if problem == 'policy': state['comparison_curriculum']['policy']['stage_mrr_threshold'] = .5
            if problem == 'pool': state['comparison_curriculum']['pool_keys'].pop()
            ids = TRAIN[::-1] if problem == 'cohort' else TRAIN
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                curriculum.adopt_or_validate(state, 'native', ids, 'b' * 64)

    def test_successful_exposure_after_adoption_also_cannot_disappear(self):
        state = active_state()
        boundary(state, 'native'); next_epoch(state)
        boundary(state, 'native'); next_epoch(state)
        boundary(state, 'native'); next_epoch(state)
        self.assertIn('U:56', state['seen_comparisons']['3'])
        state['seen_comparisons']['3'].remove('U:56')
        with self.assertRaisesRegex(ValueError, 'Historical successful'):
            curriculum.adopt_or_validate(state, 'native', TRAIN, SHA)

    def test_fixed_arms_never_grow_even_after_repeated_perfect_stage_validation(self):
        for arm in ('selected', 'native_fixed'):
            state = active_state(arm)
            for _ in range(4):
                decision, _ = boundary(state, arm)
                self.assertEqual(decision['action'], 'retain_pool')
                self.assertEqual(curriculum.stage_keys(state, arm), ['P', *legacy(arm, 1)])
                next_epoch(state)

    def test_final_two_bank_keys_preserve_exact_seven_without_duplicate_or_cap_loss(self):
        state = active_state(epoch=18)
        self.assertEqual(len(state['comparison_curriculum']['pool_keys']), 126)
        boundary(state, 'native'); next_epoch(state)
        decision, _ = boundary(state, 'native')
        self.assertEqual(decision['added_keys'], ['U:126', 'U:127'])
        self.assertEqual(len(decision['next_pool_keys']), 128)
        next_epoch(state)
        keys = curriculum.training_keys(state, 'native', 21)
        self.assertEqual(keys[:3], ['P', 'U:126', 'U:127'])
        self.assertEqual(len(keys), 8)
        self.assertEqual(len(set(keys)), 8)
        boundary(state, 'native'); next_epoch(state)
        self.assertEqual(boundary(state, 'native')[0]['action'], 'retain_pool')

    def test_no_forced_full_pool_at_epoch40_and_completed_checkpoint_resumes(self):
        state = active_state()
        while state['epoch'] <= 40:
            decision, _ = boundary(state, 'native', good=False)
            next_epoch(state)
        state['phase'] = 'complete'
        self.assertEqual(len(state['comparison_curriculum']['pool_keys']), 56)
        self.assertFalse(decision['complete_native_bank_active'])
        self.assertEqual(curriculum.adopt_or_validate(state, 'native', TRAIN, SHA)['transition'], 'same_policy_resume')
        json.dumps(state, allow_nan=False)

    def test_incomplete_training_epoch_cannot_commit_stage_or_change_core_best(self):
        state = active_state()
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ValueError, 'complete current training epoch'):
            curriculum.finish_epoch(state, 'native', report(state, 'native'), TRAIN)
        self.assertEqual(state, before)


if __name__ == '__main__':
    unittest.main()
