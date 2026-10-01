"""Explicit UNIT frozen vectors plus CUDA head mechanics, not CT accuracy."""
import hashlib
from types import SimpleNamespace
import unittest

import torch

from hiercp_v222.v1_execution import rng_state
from l0_regions.training import hash_state
from tools.local_cnn_direct_head_probe import (
    DirectScalarHead, _cases, _evaluate, _groups, _v1_reference, probe_direct_head,
)


class Budget:
    def __init__(self, fail=None):
        self.calls = 0
        self.fail = fail
    def check(self):
        self.calls += 1
        if self.calls == self.fail:
            raise MemoryError('UNIT explicit budget')


def case(name, *, device='cpu', positives=16, unobserved=32):
    generator = torch.Generator().manual_seed(sum(map(ord, name)))
    truth = torch.tensor([1] * positives + [0] * unobserved, dtype=torch.long, device=device)
    features = torch.randn(len(truth), 128, generator=generator) * .05
    features[:, 0] = torch.where(truth.cpu() == 1, 2., -2.)
    return dict(features=features.to(device), truth=truth,
                case_id=name, donor_case_id='UNIT fixed train donor',
                record_ids=[f'{name}-record-{i}' for i in range(len(truth))],
                candidate_keys=[hashlib.sha256(f'UNIT geometry {name} {i}'.encode()).hexdigest()
                                for i in range(len(truth))])


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)

    def test_v1_reference_and_scalar_head_shape_have_explicit_source(self):
        reference = _v1_reference()
        self.assertEqual(reference['revision'], '74dcc2cf03d2d40d1f582223321d96004333f661')
        self.assertEqual(reference['config']['model']['hidden_dim'], 128)
        head = DirectScalarHead(128, reference['config']['model']['dropout'])
        self.assertEqual((head[0].in_features, head[0].out_features), (128, 512))
        self.assertEqual((head[4].in_features, head[4].out_features), (512, 256))
        self.assertEqual((head[-1].in_features, head[-1].out_features), (256, 1))
        self.assertIsNone(head[-1].bias)

    def test_every_pair_once_complete_epoch_and_pure_class_observations_retained(self):
        cases = [case('UNIT many', positives=35, unobserved=41),
                 case('UNIT few', positives=3, unobserved=29),
                 case('UNIT no positive', positives=0, unobserved=9)]
        data = _cases(cases, 'train')
        order, expected = _groups(data, 32, 42, 0)
        observed = []
        for ids in order:
            self.assertLessEqual(len(ids), 32)
            self.assertEqual(len({data.rows[i]['case_id'] for i in ids}), 1)
            self.assertEqual(len({data.rows[i]['donor_case_id'] for i in ids}), 1)
            observed.extend((p, u) for p in ids if data.rows[p]['target']
                            for u in ids if not data.rows[u]['target'])
        self.assertEqual(expected, 35 * 41 + 3 * 29)
        self.assertEqual(len(observed), expected)
        self.assertEqual(len(set(observed)), expected)
        self.assertEqual({i for ids in order for i in ids}, set(range(len(data.rows))))
        second, pairs = _groups(data, 32, 42, 1)
        self.assertEqual(len(second), len(order))
        self.assertEqual(pairs, expected)

    def test_short_frozen_head_learns_without_mutating_features_or_caller_rng(self):
        train, val = [case('UNIT train')], [case('UNIT validation')]
        inputs = hash_state(dict(train=train, validation=val))
        before_rng = hash_state(rng_state())
        report = probe_direct_head(train, val, steps=16, physical_batch=32, seed=42,
                                   lr=.001, hidden_dim=128, budget=Budget())
        self.assertEqual(hash_state(dict(train=train, validation=val)), inputs)
        self.assertEqual(hash_state(rng_state()), before_rng)
        self.assertEqual(report['head_optimizer_updates'], 16)
        self.assertEqual(report['production_optimizer_updates'], 0)
        self.assertEqual(report['cnn_updates'], 0)
        self.assertEqual(report['checkpoints_written'], 0)
        self.assertFalse(report['full_v1_reproduction'])
        self.assertTrue(report['learning_check']['all_parameter_gradients_present_and_finite'])
        self.assertGreater(report['learning_check']['head_parameter_delta_norm'], 0)
        self.assertLess(report['after']['train']['metrics']['ranking_pairwise_loss'],
                        report['before']['train']['metrics']['ranking_pairwise_loss'])
        self.assertGreater(report['after']['validation']['metrics']['pair_win_rate'], .95)
        self.assertEqual(set(report['actual_update_batch_sizes']), {32})
        self.assertEqual(report['schedule']['train_observations'], 48)
        self.assertTrue(report['schedule']['all_supplied_observations_retained'])

    def test_prefix_scope_and_missing_pair_cohort_are_explicit(self):
        train, val = [case('UNIT prefix')], [case('UNIT eval')]
        report = probe_direct_head(train, val, steps=1, physical_batch=32, seed=42,
                                   lr=.001, hidden_dim=128, budget=Budget())
        self.assertEqual(report['schedule']['complete_schedule_epochs'], 0)
        self.assertEqual(report['schedule']['unique_train_observations_presented'], 32)
        self.assertEqual(report['schedule']['train_observations'], 48)
        with self.assertRaisesRegex(ValueError, 'needs observed/unobserved ranking pairs'):
            probe_direct_head([case('UNIT no ranking', positives=0, unobserved=32)], val,
                              steps=1, physical_batch=32, seed=42, lr=.001, hidden_dim=128, budget=Budget())

    def test_no_train_validation_leakage_and_no_unfrozen_features(self):
        item = case('UNIT same')
        with self.assertRaisesRegex(ValueError, 'case leakage'):
            probe_direct_head([item], [item], steps=1, physical_batch=32, seed=42,
                              lr=.001, hidden_dim=128, budget=Budget())
        invalid = case('UNIT live tensor')
        invalid['features'].requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'Frozen finite'):
            probe_direct_head([invalid], [case('UNIT separate')], steps=1, physical_batch=32,
                              seed=42, lr=.001, hidden_dim=128, budget=Budget())

    def test_failure_restores_rng_and_explicit_steps_are_mandatory(self):
        train, val = [case('UNIT failure')], [case('UNIT failure val')]
        digest = hash_state(rng_state())
        with self.assertRaisesRegex(MemoryError, 'UNIT explicit budget'):
            probe_direct_head(train, val, steps=1, physical_batch=32, seed=42,
                              lr=.001, hidden_dim=128, budget=Budget(fail=3))
        self.assertEqual(hash_state(rng_state()), digest)
        with self.assertRaisesRegex(ValueError, 'Explicit steps'):
            probe_direct_head(train, val, steps=0, physical_batch=32, seed=42,
                              lr=.001, hidden_dim=128, budget=Budget())

    def test_tied_mrr_uses_actual_geometry_order_and_reports_ties(self):
        item = case('UNIT ties', positives=1, unobserved=3)
        item['candidate_keys'] = ['z', 'a', 'b', 'c']
        data = _cases([item], 'validation')
        head = DirectScalarHead(128, .1)
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.zero_()
        result = _evaluate(head, data, 32, Budget())
        self.assertEqual(result['metrics']['exact_tie_rate'], 1)
        self.assertEqual(result['metrics']['ranking_mrr'], .25)
        self.assertAlmostEqual(result['metrics']['ranking_pairwise_loss'], float(torch.log(torch.tensor(2., dtype=torch.float64))), places=7)

    def test_cuda_original_physical_batch_and_finite_update(self):
        if not torch.cuda.is_available():
            self.skipTest('Explicit CUDA scalar-head mechanics requires GPU')
        train, val = [case('UNIT CUDA train', device='cuda')], [case('UNIT CUDA val', device='cuda')]
        report = probe_direct_head(train, val, steps=3, physical_batch=32, seed=42,
                                   lr=.001, hidden_dim=128, budget=Budget())
        self.assertEqual(report['actual_update_batch_sizes'], [32, 32, 32])
        self.assertGreater(report['learning_check']['gradient_norm_min'], 0)
        self.assertGreater(report['learning_check']['head_parameter_delta_norm'], 0)
        self.assertEqual(report['architecture']['widths'], [512, 256, 1])
        self.assertTrue(report['frozen_inputs_unchanged'])


if __name__ == '__main__':
    unittest.main()
