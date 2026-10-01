"""L1 diagnostic operator checks; no trained CT accuracy claims."""
import json
from pathlib import Path
import unittest

import torch

from tools.local_cnn_l1_probe import candidate_signal, probe_l1, _relation_stages, _attention_signal


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

    def test_ff_zero_intervenes_only_second_residual_add(self):
        from hiercp_v222.model import RelationLayer
        torch.manual_seed(335)
        layer = RelationLayer(128, 4, .1).eval()
        source, query = torch.randn(6, 128), torch.randn(3, 128)
        baseline = _relation_stages(layer, source, query, 1.)
        probe = _relation_stages(layer, source, query, 1., ff_scale=0.)
        for name in ('input', 'raw_message', 'projected_message', 'scaled_message',
                     'residual_add', 'residual_norm', 'ff_output'):
            torch.testing.assert_close(probe[name], baseline[name])
        torch.testing.assert_close(probe['second_add'], probe['residual_norm'])
        torch.testing.assert_close(probe['final_norm'], layer.update.final(probe['residual_norm']))
        self.assertGreater(float((probe['final_norm']-baseline['final_norm']).detach().abs().max()), .01)

    def test_ff_explicit_scales_required(self):
        net, query, state = self.model('cpu')
        for scales in ([], [0, .5], [1, 1], [1, float('nan')], [1, -1]):
            with self.assertRaisesRegex(ValueError, 'FF scale'):
                probe_l1(net, query, state, 3, [1], ff_scales=scales)

    def test_attention_complete_pairs_matches_direct_definition(self):
        torch.manual_seed(419)
        weights = torch.randn(5, 3, 2).softmax(1)
        report = _attention_signal(weights, 2)
        self.assertEqual(report['distinct_candidate_pairs'], 10)
        self.assertTrue(report['all_candidates_and_support_labels_included'])
        for head, row in enumerate(report['per_head']):
            p = weights[:, :, head]
            expected_cos, expected_js = [], []
            for i in range(5):
                for j in range(i+1, 5):
                    expected_cos.append(torch.nn.functional.cosine_similarity(p[i], p[j], dim=0))
                    mean = (p[i]+p[j])/2
                    expected_js.append(((p[i]*(p[i]/mean).log()).sum()
                                        +(p[j]*(p[j]/mean).log()).sum())/2)
            self.assertAlmostEqual(row['mean_pairwise_cosine'], float(torch.stack(expected_cos).mean()), places=6)
            self.assertAlmostEqual(row['mean_pairwise_jensen_shannon'], float(torch.stack(expected_js).mean()), places=6)
            torch.testing.assert_close(torch.tensor(row['variance_over_candidates_per_support_label']), p.var(0, unbiased=False))
            self.assertAlmostEqual(row['entropy_mean'], float(-(p*p.log()).sum(-1).mean()), places=6)
        unchanged = _attention_signal(weights, 5)
        for a, b in zip(report['per_head'], unchanged['per_head']):
            self.assertAlmostEqual(a['mean_pairwise_jensen_shannon'], b['mean_pairwise_jensen_shannon'], places=6)
        single = _attention_signal(weights[:1], 2)
        self.assertEqual(single['distinct_candidate_pairs'], 0)
        self.assertIsNone(single['per_head'][0]['mean_pairwise_cosine'])
        self.assertIsNone(single['per_head'][0]['mean_pairwise_jensen_shannon'])

    def test_cuda_attention_additive_cancellation_and_nonlinear_sensitivity(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA required for attention scoring checks')
        from hiercp_v222.model import RelationLayer
        layer = RelationLayer(128, 4, .1).cuda().eval()
        with torch.no_grad():
            layer.q.weight.zero_(); layer.k.weight.zero_()
            layer.q.weight[0, 0] = 1.; layer.k.weight[0, 0] = 1.
            layer.attn[0].weight.zero_(); layer.attn[0].bias.zero_()
            layer.attn[0].weight[0, 0] = 1.
            layer.attn[0].weight[0, layer.width] = 1.
            layer.attn[2].weight.zero_(); layer.attn[2].bias.zero_()
            layer.attn[2].weight[0, 0] = 1.
        source, query = torch.zeros(2, 128, device='cuda'), torch.zeros(2, 128, device='cuda')
        source[:, 0] = torch.tensor([-1., 1.], device='cuda')
        query[:, 0] = torch.tensor([2., 4.], device='cuda')
        same = _relation_stages(layer, source, query, 1., include_attention=True)['_attention_weights']
        signal = _attention_signal(same, 1)
        self.assertGreater(float((query[0]-query[1]).abs().max()), 1)
        self.assertLess(signal['per_head'][0]['candidate_weight_variance_max'], 1e-12)
        self.assertAlmostEqual(signal['per_head'][0]['mean_pairwise_cosine'], 1., places=6)
        query[:, 0] = torch.tensor([-2., 2.], device='cuda')
        different = _relation_stages(layer, source, query, 1., include_attention=True)['_attention_weights']
        signal = _attention_signal(different, 1)
        self.assertGreater(signal['per_head'][0]['candidate_weight_variance_mean'], 1e-3)
        self.assertGreater(signal['per_head'][0]['mean_pairwise_jensen_shannon'], 1e-3)
        self.assertLess(signal['per_head'][0]['mean_pairwise_cosine'], .99)

    def test_cuda_parity_all_stages_and_weights_unchanged(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA required for diagnostic operator parity')
        torch.set_num_threads(4)
        net, query, state = self.model('cuda')
        weights = {name: value.detach().clone() for name, value in net.state_dict().items()}
        histories = [value.clone() for value in state['histories']]
        labels = state['labels'].clone()
        truth = torch.tensor([1, 0, 0, 1, 0, 0, 0], device='cuda')
        report = probe_l1(net, query, state, 3, [0., .25, .5, 1.], truth,
                          ff_scales=[0., .25, .5, 1.])
        self.assertTrue(report['production_scale_parity_passed'])
        self.assertTrue(report['shared_label_seed']['first_history_matches_shared_seed'])
        self.assertEqual(report['physical_batch'], 3)
        self.assertEqual(report['candidates'], 7)
        self.assertEqual(len(report['message_scale_sweep']), 4)
        ff = report['query_ff_scale_sweep']
        self.assertTrue(ff['production_ff_scale_parity_passed'])
        self.assertEqual(ff['target_layers'], [2])
        baseline = report['message_scale_sweep'][-1]
        for row in ff['reports']:
            self.assertEqual(row['message_scale'], 1)
            self.assertEqual(row['layers'][0]['ff_scale'], 1)
            self.assertEqual(row['layers'][1]['ff_scale'], row['ff_scale'])
            for stage, values in row['layers'][0]['stages'].items():
                for key, value in values.items():
                    reference = baseline['layers'][0]['stages'][stage][key]
                    if isinstance(value, float):
                        self.assertAlmostEqual(value, reference, delta=1e-7+abs(reference)*2e-5)
                    else:
                        self.assertEqual(value, reference)
        self.assertLess(ff['reports'][-1]['production_logit_max_abs_difference'], 2e-6)
        self.assertEqual(report['attention_sensitivity']['layers'][0]['candidates'], 7)
        self.assertEqual(report['attention_sensitivity']['layers'][0]['distinct_candidate_pairs'], 21)
        self.assertEqual(len(report['attention_sensitivity']['layers'][0]['per_head']), 4)
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
