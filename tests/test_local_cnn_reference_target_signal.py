"""Small explicit DEBUG fixtures; never real-data or full-training validation."""
import unittest
import os

import torch

from tools.local_cnn_reference_target_signal import class_separation, frozen_linear_pair_ranking_probe


def feature(values, device=None):
    device = device or os.environ.get('NNUNET_TEST_DEVICE', 'cpu')
    result = torch.zeros(len(values), 128, device=device, dtype=torch.float64)
    result[:, 0] = torch.tensor(values, device=device, dtype=result.dtype)
    return result


def case(name, patient, values, truth, split='train', device=None):
    device = device or os.environ.get('NNUNET_TEST_DEVICE', 'cpu')
    return dict(case_id=name, patient_group=patient, split=split,
        features=feature(values, device), truth=torch.tensor(truth, device=device, dtype=torch.long),
        all_case_candidates_retained=True,
        rows=[dict(id=f'{name}:{i}', case_id=name, patient_group=patient, target=t) for i, t in enumerate(truth)])


class TargetSignalSeparation(unittest.TestCase):
    def test_population_geometry_and_independent_label_binding(self):
        c = case('a', 'patient-a', [0., 2., 4., 6.], [1, 1, 0, 0])
        report = class_separation(c['features'], c['truth'], rows=c['rows'])
        self.assertTrue(report['label_binding_verified'])
        self.assertEqual((report['observed_count'], report['unobserved_count']), (2, 2))
        self.assertEqual(report['raw']['class_mean_difference_squared'], 16.)
        self.assertEqual(report['raw']['within_observed_variance'], 1.)
        self.assertEqual(report['raw']['within_unobserved_variance'], 1.)
        self.assertEqual(report['raw']['fisher_ratio'], 8.)
        self.assertEqual(report['zero_norm_candidates'], 1)
        self.assertIn('not held-out accuracy', report['scope'])

    def test_raw_and_normalized_reports_do_not_confuse_amplitude_with_ratio(self):
        x = feature([1., 3., 5., 7.])
        x[:, 1] = 2.
        y = torch.tensor([1, 1, 0, 0], device=x.device)
        small, large = class_separation(x, y), class_separation(100.*x, y)
        self.assertAlmostEqual(large['raw']['class_mean_difference_squared'], 10000.*small['raw']['class_mean_difference_squared'])
        self.assertAlmostEqual(large['raw']['fisher_ratio'], small['raw']['fisher_ratio'])
        for key in ('class_mean_difference_squared', 'within_variance_sum', 'fisher_ratio'):
            self.assertAlmostEqual(large['l2_normalized'][key], small['l2_normalized'][key])

    def test_zero_variance_and_missing_class_are_explicit(self):
        x = feature([-1., 1.])
        one_per_class = class_separation(x, torch.tensor([1, 0], device=x.device))
        self.assertIsNone(one_per_class['raw']['fisher_ratio'])
        self.assertEqual(one_per_class['raw']['fisher_ratio_status'], 'UNDEFINED_ZERO_WITHIN_CLASS_VARIANCE')
        x = feature([1., 2.])
        missing = class_separation(x, torch.tensor([0, 0], device=x.device))
        self.assertEqual(missing['status'], 'NOT_EVALUABLE')
        self.assertIsNone(missing['raw']['class_mean_difference_squared'])
        self.assertIsNone(missing['raw']['fisher_ratio'])
        self.assertEqual(missing['raw']['within_unobserved_variance'], .25)

    def test_rejects_nonfinite_shape_and_metadata_mismatch(self):
        c = case('a', 'patient-a', [0., 1.], [1, 0])
        with self.assertRaises(ValueError):
            class_separation(c['features'], c['truth'].float())
        with self.assertRaises(ValueError):
            class_separation(c['features'], 1-c['truth'], rows=c['rows'])
        with self.assertRaises(ValueError):
            class_separation(c['features'][:, :2], c['truth'])
        c['features'][0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            class_separation(c['features'], c['truth'])


class FrozenLinearProbe(unittest.TestCase):
    def test_all_cases_all_pairs_and_no_feature_grad_mutation(self):
        train = [case('a', 'pa', [1., 2., -1., -2.], [1, 1, 0, 0]),
                 case('b', 'pb', [3., -3., -4.], [1, 0, 0]),
                 case('no-pair', 'pc', [20.], [0])]
        heldout = [case('h', 'ph', [1.5, -1.5], [1, 0], 'validation')]
        original = train[0]['features'].clone()
        train[0]['features'].requires_grad_(True)
        train[0]['features'].grad = torch.ones_like(original)
        report = frozen_linear_pair_ranking_probe(train, heldout, steps=6, lr=.05, feature_space='raw')
        self.assertEqual(report['physical_batch'], 8)
        self.assertEqual(report['ranking_pairs'], 6)
        self.assertEqual(len(report['training']['after']['cases']), 3)
        self.assertEqual(report['training']['after']['cases'][2]['status'], 'NOT_EVALUABLE')
        self.assertGreater(report['parameter_delta_norm'], 0.)
        self.assertLess(report['training']['after']['mean_pairwise_loss'], report['training']['before']['mean_pairwise_loss'])
        self.assertEqual(report['evaluation']['after']['pair_win_rate'], 1.)
        torch.testing.assert_close(train[0]['features'].detach(), original)
        torch.testing.assert_close(train[0]['features'].grad, torch.ones_like(original))
        self.assertFalse(report['frozen_feature_source_model_updated'])
        self.assertFalse(report['checkpoint_written'])

    def test_eval_labels_do_not_change_fitted_head(self):
        train = [case('a', 'pa', [1., -1.], [1, 0])]
        heldout = [case('h', 'ph', [2., -2.], [1, 0], 'validation')]
        first = frozen_linear_pair_ranking_probe(train, heldout, steps=3, lr=.1, feature_space='l2_normalized')
        heldout[0]['truth'] = 1-heldout[0]['truth']
        for row in heldout[0]['rows']:
            row['target'] = 1-row['target']
        second = frozen_linear_pair_ranking_probe(train, heldout, steps=3, lr=.1, feature_space='l2_normalized')
        self.assertEqual(first['linear_weight'], second['linear_weight'])
        self.assertEqual(first['evaluation']['after']['pair_win_rate'], 1.)
        self.assertEqual(second['evaluation']['after']['pair_win_rate'], 0.)

    def test_cross_case_pairs_are_never_added(self):
        train = [case('a', 'pa', [100., 99.], [1, 0]), case('b', 'pb', [-99., -100.], [1, 0])]
        heldout = [case('h', 'ph', [1., 0.], [1, 0], 'validation')]
        report = frozen_linear_pair_ranking_probe(train, heldout, steps=1, lr=.1, feature_space='raw')
        self.assertEqual(report['ranking_pairs'], 2)
        self.assertEqual(report['training']['after']['pair_win_rate'], 1.)

    def test_leakage_and_required_probe_settings_are_rejected(self):
        train = [case('a', 'pa', [1., -1.], [1, 0])]
        heldout = [case('h', 'pa', [1., -1.], [1, 0], 'validation')]
        with self.assertRaisesRegex(ValueError, 'patient_groups overlap'):
            frozen_linear_pair_ranking_probe(train, heldout, steps=1, lr=.1, feature_space='raw')
        heldout = [case('h', 'ph', [1., -1.], [1, 0], 'validation')]
        with self.assertRaises(TypeError):
            frozen_linear_pair_ranking_probe(train, heldout, lr=.1, feature_space='raw')
        for value in (0, True, 1.):
            with self.assertRaises(ValueError):
                frozen_linear_pair_ranking_probe(train, heldout, steps=value, lr=.1, feature_space='raw')
        train[0]['all_case_candidates_retained'] = False
        with self.assertRaisesRegex(ValueError, 'retention'):
            frozen_linear_pair_ranking_probe(train, heldout, steps=1, lr=.1, feature_space='raw')

    @unittest.skipUnless(torch.cuda.is_available(), 'Small DEBUG CUDA fixture requires allocated CUDA')
    def test_cuda_frozen_feature_probe(self):
        train = [case('a', 'pa', [1., -1.], [1, 0], device='cuda')]
        heldout = [case('h', 'ph', [2., -2.], [1, 0], 'validation', device='cuda')]
        separation = class_separation(train[0]['features'], train[0]['truth'], rows=train[0]['rows'])
        self.assertTrue(separation['finite'])
        report = frozen_linear_pair_ranking_probe(train, heldout, steps=1, lr=.1, feature_space='raw')
        self.assertEqual(report['device'], 'cuda:0')
        self.assertEqual(report['evaluation']['after']['pair_win_rate'], 1.)


if __name__ == '__main__':
    unittest.main()
