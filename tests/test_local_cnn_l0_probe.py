"""Short synthetic operator tests; these do not validate trained CT accuracy."""
import copy
import unittest

import torch

from l0_local_cnn.model import LocalBatch, LocalCNN, MODE
from tools.local_cnn_l0_probe import _anchor_sphere_mean, trace_local_cnn


def config():
    return dict(architecture=MODE, channels=[12, 24, 32], convolutions=[2, 3, 3],
                hidden_dim=128, margin_mm=10.0, input='native_spacing_organ_only',
                readout='organ_masked_mean_each_scale', fusion='donor_target_difference_product',
                initialization='fresh_seed42', learning_policy='same_donor_live_v1')


def fixture(device='cpu'):
    torch.manual_seed(123)
    images = torch.randn(4, 1, 9, 9, 9, device=device)
    organ = torch.ones_like(images, dtype=torch.bool)
    # Only the native 7x8x9 crop is present in the last padded volume.
    organ[3, :, 7:] = False
    organ[3, :, :, 8:] = False
    audit = [dict(case=f'UNIT_FIXTURE_{i}', origin=[10, 20, 30],
                  shape=[7, 8, 9] if i == 3 else [9, 9, 9],
                  spacing=[2., 1., 3.], anchor_in_organ=True) for i in range(4)]
    batch = LocalBatch(images, organ, torch.tensor([0, 0, 0], device=device),
                       torch.tensor([1, 2, 3], device=device),
                       torch.arange(3, device=device), audit).validate()
    local = LocalCNN(config()).to(device).eval()
    centers = [[14, 24, 34]] * 3
    return local, batch, centers


class LocalL0ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def parity(self, device):
        local, batch, centers = fixture(device)
        before = copy.deepcopy(local.state_dict())
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
        with torch.no_grad():
            expected = local(batch)
            result, stages, audit = trace_local_cnn(local, batch,
                recipient_centers_native=centers, donor_centers_native=centers,
                anchor_radius_mm=3.)
        torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(result, stages['fusion_output'], rtol=0, atol=0)
        self.assertEqual(result.shape, (3, 128))
        self.assertFalse(result.requires_grad)
        self.assertTrue(audit['diagnostic_only'])
        self.assertEqual(audit['physical_pairs'], 3)
        self.assertEqual(audit['unique_cnn_crops'], 4)
        self.assertTrue(all(value is None or value.ndim == 2 for value in stages.values()))
        self.assertTrue(all(value is None or not value.requires_grad for value in stages.values()))
        self.assertEqual(stages['fusion_input'].shape, (3, 512))
        self.assertEqual(stages['global_mean_concat_recipient'].shape, (3, 68))
        torch.testing.assert_close(stages['project_donor'], stages['project_donor'][:1].expand(3, -1))
        for name, value in before.items():
            now = local.state_dict()[name]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, now, rtol=0, atol=0)
            else:
                self.assertEqual(value, now)
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        if cuda_rng is not None:
            self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))

    def test_cpu_operator_parity_and_read_only_state(self):
        self.parity('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA parity requires actual GPU')
    def test_cuda_operator_parity_and_read_only_state(self):
        self.parity('cuda')

    def test_anisotropic_physical_sphere_and_stride_centers(self):
        coordinates = torch.stack(torch.meshgrid(*[torch.arange(6.)] * 3, indexing='ij'))
        features = coordinates[:2][None]
        mask = torch.ones(1, 1, 6, 6, 6, dtype=torch.bool)
        ids = torch.tensor([0])
        mean, report = _anchor_sphere_mean(features, mask, torch.tensor([[2., 2., 2.]]),
            torch.tensor([[2., 1., 3.]]), ids, stride=1, radius_mm=2.)
        self.assertEqual(report['candidate_organ_cell_counts'], [7])
        torch.testing.assert_close(mean, torch.tensor([[2., 2.]]))
        coarse, report = _anchor_sphere_mean(features, mask, torch.tensor([[4., 4., 4.]]),
            torch.tensor([[2., 1., 3.]]), ids, stride=2, radius_mm=2.)
        self.assertEqual(report['candidate_organ_cell_counts'], [3])
        torch.testing.assert_close(coarse, torch.tensor([[2., 2.]]))

    def test_invalid_roi_is_not_run_for_every_candidate_not_zero_fallback(self):
        features = torch.ones(2, 3, 5, 5, 5)
        mask = torch.zeros(2, 1, 5, 5, 5, dtype=torch.bool)
        mask[0, 0, 2, 2, 2] = True
        mask[1, 0, 4, 4, 4] = True
        mean, report = _anchor_sphere_mean(features, mask,
            torch.tensor([[2., 2., 2.], [0., 0., 0.]]), torch.ones(2, 3),
            torch.tensor([0, 1]), stride=1, radius_mm=.1)
        self.assertIsNone(mean)
        self.assertEqual(report['status'], 'NOT_RUN')
        self.assertEqual(report['candidate_organ_cell_counts'], [1, 0])
        self.assertEqual(report['candidate_covered'], [True, False])
        self.assertTrue(report['all_candidates_retained'])

    def test_background_anchor_remains_bound_without_new_admission(self):
        local, batch, _ = fixture()
        batch.organ.zero_()
        batch.organ[:, :, 4, 4, 4] = True
        batch.validate()
        centers = [[10, 20, 30]] * 3
        result, stages, audit = trace_local_cnn(local, batch,
            recipient_centers_native=centers, donor_centers_native=centers,
            anchor_radius_mm=.1)
        self.assertTrue(torch.isfinite(result).all())
        self.assertIsNone(stages['scale1_anchor_roi_recipient'])
        self.assertEqual(audit['anchor_readouts']['scale1_anchor_roi_recipient']['anchors_in_native_organ'], [False] * 3)

    def test_nonempty_padding_is_rejected(self):
        local, batch, centers = fixture()
        batch.organ[3, :, 8, 0, 0] = True
        batch.validate()
        with self.assertRaisesRegex(ValueError, 'includes padding'):
            trace_local_cnn(local, batch, recipient_centers_native=centers,
                donor_centers_native=centers, anchor_radius_mm=3.)

    def test_radius_is_explicit_and_not_an_implicit_default(self):
        local, batch, centers = fixture()
        with self.assertRaises(TypeError):
            trace_local_cnn(local, batch, recipient_centers_native=centers,
                donor_centers_native=centers)
        with self.assertRaisesRegex(ValueError, 'finite positive'):
            trace_local_cnn(local, batch, recipient_centers_native=centers,
                donor_centers_native=centers, anchor_radius_mm=0.)

    def test_mixed_donor_comparison_is_rejected(self):
        local, batch, centers = fixture()
        batch.donor[2] = 1
        batch.validate()
        with self.assertRaisesRegex(ValueError, 'Same-donor'):
            trace_local_cnn(local, batch, recipient_centers_native=centers,
                donor_centers_native=centers, anchor_radius_mm=3.)


if __name__ == '__main__':
    unittest.main()
