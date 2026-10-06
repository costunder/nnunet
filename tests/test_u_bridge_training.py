"""UNIT score/checkpoint checks only; no model construction or CUDA execution."""
from pathlib import Path
import math
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

from hiercp_v1x.u_bridge_training import (
    aggregate_rows, atomic_save, capture_rng, cpu_copy, digest, pair_objective,
    permutation, record_comparisons, restore_optimizer_history, restore_rng,
    retry_amp_overflow, run_arm, score_row, selection_key, validate_cursor,
    validate_resume_progress,
)


def example(index=0, case='patient-A'):
    return dict(index=index, id=f'{case}:{index}', case_id=case, source_component=9,
        positive_center=(1, 2, 3), selected_centers=[(4+i, 5, 6) for i in range(7)],
        native_centers=[(10+i, 11, 12) for i in range(128)])


class ObjectiveUNIT(unittest.TestCase):
    def test_each_problem_and_all_seven_comparisons_have_equal_weight(self):
        first = torch.tensor([2., 0., 1., 2., 3., 4., 5., 6.], requires_grad=True)
        second = torch.tensor([-3., -1., -2., -3., -4., -5., -6., -7.], requires_grad=True)
        consistency = torch.tensor(.25, requires_grad=True)
        loss, terms = pair_objective([first, second], consistency)
        expected = sum(math.log1p(math.exp(float(s[j]-s[0])))
                       for s in (first.detach(), second.detach()) for j in range(1, 8)) / 14
        self.assertAlmostEqual(float(terms['ranking'].detach()), expected, places=6)
        self.assertAlmostEqual(float(loss.detach()), expected + .025, places=6)
        loss.backward()
        for scores in (first, second):
            self.assertLess(float(scores.grad[0]), 0.)
            self.assertTrue(bool((scores.grad[1:] > 0).all()))
            self.assertAlmostEqual(float(scores.grad.sum()), 0., places=7)
        self.assertAlmostEqual(float(consistency.grad), .1, places=7)

    def test_per_problem_translation_does_not_change_rank_loss(self):
        values = [torch.linspace(-2, 2, 8), torch.linspace(3, -1, 8)]
        base, _ = pair_objective(values, torch.tensor(0.))
        shifted, _ = pair_objective([values[0]+17, values[1]-9], torch.tensor(0.))
        self.assertAlmostEqual(float(base), float(shifted), places=6)

    def test_missing_or_extra_candidates_are_rejected(self):
        for count in (1, 7, 9, 129):
            with self.assertRaises(ValueError):
                pair_objective([torch.zeros(count)], torch.tensor(0.))
        with self.assertRaises(ValueError): pair_objective([], torch.tensor(0.))
        with self.assertRaises(ValueError): pair_objective([torch.zeros(8)], torch.zeros(2))


class MetricsUNIT(unittest.TestCase):
    def test_ties_cannot_gain_positive_first_advantage(self):
        row = score_row(torch.zeros(129), example(), ('P', *(f'U:{i}' for i in range(128))), expected_candidates=129)
        self.assertEqual(row['first_P_rank'], 129)
        self.assertEqual(row['top1'], 0.)
        self.assertEqual(row['pair_win'], 0.)
        self.assertEqual(row['pair_tie'], 1.)
        self.assertAlmostEqual(row['pair_loss'], math.log(2), places=12)
        self.assertEqual(row['candidate_centers'][0], [1, 2, 3])
        self.assertEqual(row['candidate_centers'][-1], [137, 11, 12])
        self.assertEqual(row['candidate_difficulties'], [0] + [1] * 128)

    def test_macro_weights_patients_equally_with_unequal_source_counts(self):
        keys = ('P', 'U:0')
        rows = [score_row(torch.tensor([2., 0.]), example(0), keys, expected_candidates=2),
                score_row(torch.tensor([2., 0.]), example(1), keys, expected_candidates=2),
                score_row(torch.tensor([0., 2.]), example(2, 'patient-B'), keys, expected_candidates=2)]
        report = aggregate_rows(rows)
        self.assertEqual(report['patients'], 2)
        self.assertEqual(report['source_problems'], 3)
        self.assertEqual(report['metrics']['top1'], .5)
        self.assertEqual(report['metrics']['mrr'], .75)
        self.assertAlmostEqual(report['metrics']['pair_loss'],
            (math.log1p(math.exp(-2)) + math.log1p(math.exp(2))) / 2)
        with self.assertRaises(ValueError): aggregate_rows(rows + [rows[0]])

    def test_score_order_and_finite_values_are_required(self):
        for scores, keys in (([0., float('nan')], ('P', 'U:0')),
                             ([0., 1.], ('P', 'P')), ([0., 1.], ('U:0', 'P')),
                             ([0., 1.], ('P', 'U:128'))):
            with self.assertRaises(ValueError): score_row(torch.tensor(scores), example(), keys, expected_candidates=2)

    def test_best_definition_has_stable_tiebreaks(self):
        def report(mrr, top1, loss): return {'metrics':dict(mrr=mrr, top1=top1, pair_loss=loss)}
        self.assertGreater(selection_key(report(.7, .3, 3)), selection_key(report(.6, .9, .1)))
        self.assertGreater(selection_key(report(.7, .4, 3)), selection_key(report(.7, .3, .1)))
        self.assertGreater(selection_key(report(.7, .4, .1)), selection_key(report(.7, .4, .2)))


class CheckpointUNIT(unittest.TestCase):
    def test_cpu_snapshot_is_immutable_and_value_hashed(self):
        live = dict(model={'weight':torch.arange(6.).reshape(2, 3)},
                    optimizer={'moments':[torch.tensor(2.)]}, epoch=2)
        snapshot = cpu_copy(live); fingerprint = digest(snapshot)
        live['model']['weight'].add_(10); live['optimizer']['moments'][0].zero_()
        self.assertEqual(float(snapshot['model']['weight'][0, 0]), 0.)
        self.assertEqual(float(snapshot['optimizer']['moments'][0]), 2.)
        self.assertEqual(digest(snapshot), fingerprint)
        self.assertNotEqual(digest(live), fingerprint)

    def test_atomic_latest_roundtrips_exact_rng_payload(self):
        with patch('torch.cuda.is_available', return_value=False):
            payload = dict(model={'scalar':torch.tensor(2.)}, rng=capture_rng(), phase='training', next_batch=4)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / 'latest.pt'
            atomic_save(path, payload)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(digest(saved), digest(payload))
            payload['next_batch'] = 6; atomic_save(path, payload)
            self.assertEqual(torch.load(path, weights_only=False)['next_batch'], 6)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['latest.pt'])

    def test_global_rng_restore_and_independent_permutation_cursor(self):
        with patch('torch.cuda.is_available', return_value=False):
            original = capture_rng()
            try:
                random.seed(93); np.random.seed(72); torch.manual_seed(81)
                state = capture_rng()
                first = (random.random(), np.random.random(), torch.rand(5))
                restore_rng(state)
                second = (random.random(), np.random.random(), torch.rand(5))
                self.assertEqual(first[:2], second[:2]); self.assertTrue(torch.equal(first[2], second[2]))
            finally: restore_rng(original)
        generator = torch.Generator().manual_seed(1845)
        permutation(list(range(11)), generator)
        saved = generator.get_state().clone()
        expected = permutation(list(range(11)), generator)
        restored = torch.Generator(); restored.set_state(saved)
        self.assertEqual(expected, permutation(list(range(11)), restored))

    def test_cursor_preserves_all_samples_and_batch_boundaries(self):
        values = [7, 3, 1, 8, 2]
        for position in (0, 2, 4, 5): validate_cursor(dict(order=values, position=position), values, 2)
        for order, position in ((values, 1), (values, 6), ([7, 3, 1, 8, 8], 0), (None, 2)):
            with self.assertRaises(ValueError): validate_cursor(dict(order=order, position=position), values, 2)

    def test_all_forty_epoch_orders_and_rng_states_match_original_random_sampler(self):
        # Noncontiguous global IDs verify the sampler operates on original
        # example positions; all151 production problems remain once per epoch.
        ids = [1000 + i * 3 for i in range(151)]
        original = torch.Generator().manual_seed(42 + 2003)
        bridge = torch.Generator().manual_seed(42 + 2003)
        for epoch in range(1, 41):
            expected_positions = list(torch.utils.data.RandomSampler(range(len(ids)), generator=original))
            actual = permutation(ids, bridge)
            self.assertEqual(actual, [ids[i] for i in expected_positions], f'epoch{epoch}')
            self.assertEqual(set(actual), set(ids)); self.assertEqual(len(actual), len(ids))
            self.assertTrue(torch.equal(bridge.get_state(), original.get_state()), f'epoch{epoch} RNG')
        saved = bridge.get_state().clone()
        expected_next = permutation(ids, bridge)
        restored = torch.Generator(); restored.set_state(saved)
        self.assertEqual(permutation(ids, restored), expected_next)


class OverflowAndExecutionUNIT(unittest.TestCase):
    def test_only_completed_updates_earn_exact_comparison_coverage(self):
        state = {'seen_comparisons': {'8': []}}
        rows = [{'sample_index':8, 'candidate_keys':['P', *(f'U:{i}' for i in range(7))]}]
        record_comparisons(state, rows, optimizer_updated=False)
        self.assertEqual(state['seen_comparisons']['8'], [])
        record_comparisons(state, rows, optimizer_updated=True)
        rows[0]['candidate_keys'] = ['P', *(f'U:{i}' for i in range(7, 14))]
        record_comparisons(state, rows, optimizer_updated=True)
        self.assertEqual(set(state['seen_comparisons']['8']), {f'U:{i}' for i in range(14)})
        self.assertNotIn('P', state['seen_comparisons']['8'])

    def test_overflow_uses_scaler_skip_and_lowers_scale(self):
        class Scaler:
            def __init__(self): self.scale=1024; self.skips=0
            def is_enabled(self): return True
            def get_scale(self): return self.scale
            def step(self, optimizer): self.skips+=1  # Recorded-inf scaler skip; optimizer not called.
            def update(self): self.scale/=2
        class Optimizer:
            def step(self): raise AssertionError('Overflow must not advance optimizer')
        scaler=Scaler()
        self.assertEqual(retry_amp_overflow(scaler, Optimizer()), (1024, 512))
        self.assertEqual(scaler.skips, 1)
        scaler.update=lambda:None
        with self.assertRaises(RuntimeError): retry_amp_overflow(scaler, Optimizer())
        scaler.is_enabled=lambda:False
        with self.assertRaises(FloatingPointError): retry_amp_overflow(scaler, Optimizer())

    def test_production_shortening_and_cpu_fallback_rejected_before_work(self):
        arguments = dict(arm='native', output='unused_UNIT', physical_batch=2, workers=2,
                         identity={}, budget=None)
        with self.assertRaisesRegex(ValueError, 'forty'):
            run_arm(None, None, {}, epochs=1, **arguments)
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'Actual CUDA'):
                run_arm(None, None, {}, epochs=40, **arguments)


class SchedulerResumeUNIT(unittest.TestCase):
    """Direct scalar tensor optimization; no neural model or CUDA execution."""

    @staticmethod
    def make(values=None):
        parameter = (torch.tensor([.4, -.2], dtype=torch.float64) if values is None
                     else values.detach().clone()).requires_grad_(True)
        optimizer = torch.optim.AdamW([parameter], lr=.01, weight_decay=.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        return parameter, optimizer, scheduler

    def trajectory(self, pause):
        parameter, optimizer, scheduler = self.make()
        learning_rates = []; warning_messages = []
        for epoch in range(1, 41):
            for batch in range(2):
                learning_rates.append(optimizer.param_groups[0]['lr'])
                optimizer.zero_grad(set_to_none=True)
                target = torch.tensor([epoch / 40, batch - .5], dtype=torch.float64)
                (parameter - target).square().sum().backward(); optimizer.step()
            if pause and epoch == 1:
                values = parameter.detach().clone()
                saved_optimizer = cpu_copy(optimizer.state_dict())
                saved_scheduler = cpu_copy(scheduler.state_dict())
                parameter, optimizer, scheduler = self.make(values)
                optimizer.load_state_dict(saved_optimizer); scheduler.load_state_dict(saved_scheduler)
                before = (digest(optimizer.state_dict()), digest(scheduler.state_dict()), parameter.detach().clone())
                state = dict(epoch=1, phase='validation', position=3, order=[8, 5, 3],
                    updates=2, attempts=2, overflows=0, history=[],
                    train_rows=[{'sample_index':i} for i in [8, 5, 3]],
                    validation_rows=[], validation_position=0)
                validate_resume_progress(state, [8, 5, 3], [10], 2, 40, scheduler)
                receipt = restore_optimizer_history(optimizer, 2)
                self.assertEqual(receipt['verified_genuine_updates'], 2)
                self.assertEqual(before[0], digest(optimizer.state_dict()))
                self.assertEqual(before[1], digest(scheduler.state_dict()))
                self.assertTrue(torch.equal(before[2], parameter.detach()))
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always'); scheduler.step()
            warning_messages += [str(item.message) for item in captured]
            learning_rates.append(optimizer.param_groups[0]['lr'])
        return learning_rates, digest(optimizer.state_dict()), parameter.detach(), warning_messages

    def test_paused_before_first_epoch_schedule_matches_all_forty_epochs(self):
        uninterrupted = self.trajectory(False); resumed = self.trajectory(True)
        self.assertEqual(uninterrupted[0], resumed[0])
        self.assertEqual(uninterrupted[1], resumed[1])
        self.assertTrue(torch.equal(uninterrupted[2], resumed[2]))
        self.assertEqual(resumed[3], [])

    def test_real_adamw_history_is_required_to_restore_transient_flag(self):
        parameter, optimizer, _ = self.make()
        receipt = restore_optimizer_history(optimizer, 0)
        self.assertFalse(receipt['transient_optimizer_called'])
        with self.assertRaises(ValueError): restore_optimizer_history(optimizer, 1)
        parameter.square().sum().backward(); optimizer.step()
        with self.assertRaises(ValueError): restore_optimizer_history(optimizer, 0)
        with self.assertRaises(ValueError): restore_optimizer_history(optimizer, 2)
        self.assertTrue(restore_optimizer_history(optimizer, 1)['transient_optimizer_called'])

    def test_fused_overflow_only_state_has_no_genuine_update_history(self):
        parameter = torch.tensor([1., 2.], requires_grad=True)
        optimizer = torch.optim.AdamW([parameter], lr=.01, fused=True)
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        scaler = torch.amp.GradScaler('cpu', init_scale=128.)
        before = parameter.detach().clone()
        scaler.scale((parameter * float('inf')).sum()).backward(); scaler.unscale_(optimizer)
        retry_amp_overflow(scaler, optimizer)
        self.assertTrue(torch.equal(before, parameter.detach()))
        self.assertEqual(float(optimizer.state[parameter]['step']), 0.)
        receipt = restore_optimizer_history(optimizer, 0)
        self.assertFalse(receipt['transient_optimizer_called'])
        self.assertEqual(receipt['verified_initialized_parameter_tensors'], 1)
        optimizer.state[parameter]['exp_avg'][0] = .1
        with self.assertRaises(ValueError): restore_optimizer_history(optimizer, 0)

    def test_completed_resume_metadata_does_not_repeat_or_mutate_work(self):
        state = dict(epoch=41, phase='complete', position=0, order=None,
            updates=80, attempts=83, overflows=3,
            history=[{'epoch':epoch} for epoch in range(1, 41)], train_rows=[],
            validation_rows=[], validation_position=0)
        scheduler = SimpleNamespace(last_epoch=40, T_max=40)
        before = digest(state)
        validate_resume_progress(state, [8, 5, 3], [10], 2, 40, scheduler)
        self.assertEqual(digest(state), before)
        for key, value in (('updates', 81), ('epoch', 40), ('phase', 'training'),
                           ('validation_position', 1), ('attempts', 80)):
            invalid = cpu_copy(state); invalid[key] = value
            with self.assertRaises(ValueError):
                validate_resume_progress(invalid, [8, 5, 3], [10], 2, 40, scheduler)


if __name__ == '__main__':
    unittest.main()
