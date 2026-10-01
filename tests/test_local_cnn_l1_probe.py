"""L1 diagnostic operator checks; no trained CT accuracy claims."""
import json
from pathlib import Path
import unittest

import torch

from tools.local_cnn_l1_probe import candidate_signal, probe_l1, _relation_stages


class L1ProbeChecks(unittest.TestCase):
    def model(self, device):
        from hiercp_v222.model import PromptGraphModel
        torch.manual_seed(372)
        cfg = json.loads(Path('config/prompt_graph_v222_v1_l0.json').read_text())
        base = json.loads(Path('config/train.json').read_text())
        net = PromptGraphModel(cfg, base, {}, local_encoder=torch.nn.Identity()).to(device).eval()
        # Full 128D operator inputs; synthetic examples are explicitly unit tests.
        support = (torch.randn(12, 128, device=device),
                   torch.arange(3, device=device).repeat_interleave(4),
                   torch.tensor([0, 1, 0, 1]*3, device=device))
        query = torch.randn(7, 128, device=device)
        with torch.no_grad():
            state = net.prepare_support(*support)
        return net, query, state

    def test_common_and_centered_signal_distinguished(self):
        signal = candidate_signal(torch.tensor([[1., 2.], [2., 4.]]))
        self.assertGreater(signal['centered_energy'], 0)
        self.assertLess(signal['normalized_centered_energy'], 1e-12)
        self.assertLess(signal['common_energy_fraction'], 1)
        same = candidate_signal(torch.ones(4, 128))
        self.assertEqual(same['common_energy_fraction'], 1)
        self.assertEqual(same['centered_energy'], 0)
        zero = candidate_signal(torch.zeros(4, 128))
        self.assertIsNone(zero['common_energy_fraction'])

    def test_nonfinite_signal_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Finite'):
            candidate_signal(torch.tensor([[float('nan'), 1.]]))

    def test_scale_zero_retains_residual_norm_and_ff(self):
        from hiercp_v222.model import RelationLayer
        torch.manual_seed(328)
        layer = RelationLayer(128, 4, .1).eval()
        source, query = torch.randn(6, 128), torch.randn(3, 128)
        traced = _relation_stages(layer, source, query, 0.)
        normalized = layer.update.norm(query)
        expected = layer.update.final(normalized+layer.update.ff(normalized))
        torch.testing.assert_close(traced['residual_add'], query)
        torch.testing.assert_close(traced['final_norm'], expected)
        self.assertGreater(float((expected-query).detach().abs().max()), .1)

    def test_invalid_scales_or_training_rejected(self):
        net, query, state = self.model('cpu')
        for scales in ([], [0, .5], [1, 1], [1, float('nan')], [1, -1]):
            with self.assertRaisesRegex(ValueError, 'scale'):
                probe_l1(net, query, state, 3, scales)
        net.train()
        with self.assertRaisesRegex(ValueError, 'eval'):
            probe_l1(net, query, state, 3, [1])

    def test_cuda_parity_all_stages_and_weights_unchanged(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA required for diagnostic operator parity')
        torch.set_num_threads(4)
        net, query, state = self.model('cuda')
        weights = {name: value.detach().clone() for name, value in net.state_dict().items()}
        histories = [value.clone() for value in state['histories']]
        labels = state['labels'].clone()
        truth = torch.tensor([1, 0, 0, 1, 0, 0, 0], device='cuda')
        report = probe_l1(net, query, state, 3, [0., .25, .5, 1.], truth)
        self.assertTrue(report['production_scale_parity_passed'])
        self.assertTrue(report['shared_label_seed']['first_history_matches_shared_seed'])
        self.assertEqual(report['physical_batch'], 3)
        self.assertEqual(report['candidates'], 7)
        self.assertEqual(len(report['message_scale_sweep']), 4)
        for scale in report['message_scale_sweep']:
            self.assertEqual(len(scale['layers']), 2)
            for layer in scale['layers']:
                self.assertEqual(len(layer['stages']), 9)
                self.assertEqual(layer['incoming_edges_per_query'], 6)
        for name, value in net.state_dict().items():
            self.assertTrue(torch.equal(weights[name], value), name)
        for old, value in zip(histories, state['histories']):
            self.assertTrue(torch.equal(old, value))
        self.assertTrue(torch.equal(labels, state['labels']))
        # Scale 1 internal final stage must independently match RelationLayer.
        layer = net.l1[0]
        stages = _relation_stages(layer, state['histories'][0], query, 1.)
        src = torch.arange(6, device='cuda').repeat(7)
        dst = torch.arange(7, device='cuda').repeat_interleave(6)
        with torch.no_grad():
            production = layer.messages(state['histories'][0], query, src, dst,
                                        query.new_zeros(len(src), 2))
        torch.testing.assert_close(stages['final_norm'], production, atol=2e-6, rtol=2e-5)
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
