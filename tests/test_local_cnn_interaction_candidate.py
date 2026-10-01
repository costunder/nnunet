"""Diagnostic full-size L1 operator checks, not trained CT performance claims."""
import copy
import json
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from hiercp_v222.model import PromptGraphModel, RelationLayer
from hiercp_v222.v1_execution import rng_state
from l0_regions.training import hash_state
from tools.local_cnn_interaction_candidate import (
    ARCHITECTURE, CandidateRelationLayer, attention_terms, clone_candidate, probe_interaction,
)
from tools.local_cnn_l1_probe import _attention_signal


def fixture(device='cpu', dtype=torch.float32):
    torch.manual_seed(865)
    cfg = json.loads(Path('config/prompt_graph_v222_v1_l0.json').read_text())
    base = json.loads(Path('config/train.json').read_text())
    net = PromptGraphModel(cfg, base, {}, local_encoder=torch.nn.Identity()).to(device, dtype=dtype).eval()
    query = torch.randn(7, 128, device=device, dtype=dtype)
    support = (torch.randn(12, 128, device=device, dtype=dtype),
               torch.arange(3, device=device).repeat_interleave(4),
               torch.tensor([0, 1, 0, 1]*3, device=device))
    truth = torch.tensor([1, 0, 0, 1, 0, 0, 0], device=device)
    return net, query, support, truth


def complete_edges(source, query):
    src = torch.arange(len(source), device=query.device).repeat(len(query))
    dst = torch.arange(len(query), device=query.device).repeat_interleave(len(source))
    return src, dst, query.new_zeros(len(src), 2)


class InteractionCandidateChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_explicit_scale_and_original_layers_required(self):
        net, query, support, truth = fixture()
        for scale in (-1, float('nan'), True, None):
            with self.assertRaisesRegex(ValueError, 'scale'):
                clone_candidate(net, scale)
        candidate = clone_candidate(net, 1.)
        with self.assertRaisesRegex(ValueError, 'unmodified'):
            clone_candidate(candidate, 1.)
        for scales in ([], [1], [0, 0], [0, float('inf')]):
            with self.assertRaisesRegex(ValueError, 'scale'):
                probe_interaction(net, query, support, scales=scales, truth=truth, batch=3,
                    support_record_ids=[str(i) for i in range(12)], query_group='query')

    def test_clone_preserves_rng_parameter_layout_and_optimizer_membership(self):
        net, _, _, _ = fixture(dtype=torch.float64)
        # Non-default dense layout verifies strict-load is not enough by itself.
        net.l1[0].q.weight.data = net.l1[0].q.weight.data.T.contiguous().T
        net.l1[0].q.weight.requires_grad_(False)
        before = hash_state(net.state_dict()); rng_before = hash_state(rng_state())
        candidate = clone_candidate(net, .5)
        self.assertEqual(hash_state(rng_state()), rng_before)
        self.assertEqual(hash_state(net.state_dict()), before)
        self.assertEqual(hash_state(candidate.state_dict()), before)
        self.assertEqual(candidate._diagnostic_architecture, ARCHITECTURE)
        original = dict(net.named_parameters())
        for name, value in candidate.named_parameters():
            self.assertEqual(value.dtype, torch.float64)
            self.assertEqual(value.stride(), original[name].stride())
            self.assertEqual(value.requires_grad, original[name].requires_grad)
            self.assertIsNot(value, original[name])
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-4)
        bound = [p for group in optimizer.param_groups for p in group['params']]
        self.assertEqual({id(p) for p in bound}, {id(p) for p in candidate.parameters()})
        self.assertEqual(len(bound), len(list(net.parameters())))
        self.assertEqual(set(candidate.state_dict()), set(net.state_dict()))

    def test_clone_failure_restores_constructor_rng(self):
        net, _, _, _ = fixture()
        original = net.l1[0].load_state_dict
        # Force an error after constructor initialization, not at input validation.
        from unittest.mock import patch
        before = hash_state(rng_state())
        with patch.object(CandidateRelationLayer, 'load_state_dict', side_effect=RuntimeError('injected strict load failure')):
            with self.assertRaisesRegex(RuntimeError, 'strict load'):
                clone_candidate(net, 1.)
        self.assertEqual(hash_state(rng_state()), before)
        self.assertTrue(callable(original))

    def test_scale_zero_output_and_gradient_parity_cpu_and_cuda(self):
        devices = ['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                net, query, support, truth = fixture(device)
                candidate = clone_candidate(net, 0.)
                outputs, derivatives = [], []
                for model in (net, candidate):
                    q = query.detach().clone().requires_grad_()
                    s = support[0].detach().clone().requires_grad_()
                    state = model.prepare_support(s, support[1], support[2])
                    result = model.predict_embeddings(q, state)
                    logits = result['logits']
                    score = logits[:, 1]-logits[:, 0]
                    loss = (F.softplus(-(score[truth == 1, None]-score[None, truth == 0])).mean()
                            +F.cross_entropy(logits, truth)+result['alignment_loss_weight']*result['alignment_loss'])
                    params = [q, s]+list(model.parameters())
                    derivatives.append(torch.autograd.grad(loss, params, allow_unused=True))
                    outputs.append(logits)
                torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-5)
                for old, new in zip(*derivatives):
                    if old is None:
                        self.assertIsNone(new)
                    else:
                        self.assertIsNotNone(new)
                        torch.testing.assert_close(old, new, atol=2e-6, rtol=2e-5)

    def _additive_cancellation_layer(self, device):
        layer = CandidateRelationLayer(128, 4, .1, scale=1.).to(device).eval()
        with torch.no_grad():
            layer.q.weight.zero_(); layer.k.weight.zero_()
            layer.q.weight[0, 0] = 1.; layer.k.weight[0, 0] = 1.
            layer.attn[0].weight.zero_(); layer.attn[0].bias.zero_()
            layer.attn[0].weight[0, 0] = 1.
            layer.attn[0].weight[0, layer.width] = 1.
            layer.attn[2].weight.zero_(); layer.attn[2].bias.zero_()
            layer.attn[2].weight[0, 0] = 1.
        return layer

    def test_cuda_dot_breaks_additive_shift_but_equal_keys_can_still_be_invariant(self):
        if not torch.cuda.is_available():
            self.skipTest('Real CUDA attention counterexample required')
        layer = self._additive_cancellation_layer('cuda')
        source, query = torch.zeros(2, 128, device='cuda'), torch.zeros(3, 128, device='cuda')
        source[:, 0] = torch.tensor([-1., 1.], device='cuda')
        query[:, 0] = torch.tensor([2., 4., 6.], device='cuda')
        src, dst, edge = complete_edges(source, query)
        old = attention_terms(layer, source, query, src, dst, edge, scale=0.)
        new = attention_terms(layer, source, query, src, dst, edge, scale=1.)
        old_signal = _attention_signal(old['weights'].reshape(3, 2, 4), 2)
        new_signal = _attention_signal(new['weights'].reshape(3, 2, 4), 2)
        self.assertLess(old_signal['per_head'][0]['candidate_weight_variance_mean'], 1e-12)
        self.assertGreater(new_signal['per_head'][0]['candidate_weight_variance_mean'], 1e-5)
        self.assertGreater(new_signal['per_head'][0]['mean_pairwise_jensen_shannon'], 1e-5)
        source[:, 0] = 1.
        same = attention_terms(layer, source, query, src, dst, edge, scale=1.)
        same_signal = _attention_signal(same['weights'].reshape(3, 2, 4), 2)
        self.assertEqual(same_signal['per_head'][0]['candidate_weight_variance_mean'], 0)
        # Queries varying only in q-k-orthogonal coordinates are also valid invariance.
        source[:, 0] = torch.tensor([-1., 1.], device='cuda')
        query[:, 0] = 2.; query[:, 1] = torch.tensor([1., 2., 3.], device='cuda')
        same = attention_terms(layer, source, query, src, dst, edge, scale=1.)
        same_signal = _attention_signal(same['weights'].reshape(3, 2, 4), 2)
        self.assertEqual(same_signal['per_head'][0]['candidate_weight_variance_mean'], 0)

    def test_support_and_query_both_use_candidate_while_l2_weights_are_fixed(self):
        net, query, support, _ = fixture()
        candidate = clone_candidate(net, 1.)
        visits = []
        handles = [layer.update.register_forward_pre_hook(
            lambda module, values, level=i: visits.append((level, len(values[0]))))
            for i, layer in enumerate(candidate.l1)]
        with torch.no_grad():
            state = candidate.prepare_support(*support)
            candidate.predict_embeddings(query, state)
            old_state = net.prepare_support(*support)
        for handle in handles:
            handle.remove()
        # Support destinations include data+2 labels per patient; query has 7.
        self.assertTrue(all((i, 18) in visits and (i, 7) in visits for i in range(2)))
        self.assertGreater(float((state['histories'][1]-old_state['histories'][1]).abs().max()), 1e-5)
        for name, value in net.state_dict().items():
            if name.startswith(('l2.', 'l2_updates.')):
                torch.testing.assert_close(value, candidate.state_dict()[name], atol=0, rtol=0)

    def test_every_original_l1_parameter_remains_connected_to_real_objective(self):
        net, query, support, truth = fixture()
        candidate = clone_candidate(net, 1.)
        state = candidate.prepare_support(*support)
        result = candidate.predict_embeddings(query, state)
        logits = result['logits'];score = logits[:, 1]-logits[:, 0]
        loss = (F.softplus(-(score[truth == 1, None]-score[None, truth == 0])).mean()
                +F.cross_entropy(logits, truth)+result['alignment_loss_weight']*result['alignment_loss'])
        named = list(candidate.l1.named_parameters())
        derivatives = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
        for (name, _), derivative in zip(named, derivatives):
            self.assertIsNotNone(derivative, name)
            self.assertTrue(bool(torch.isfinite(derivative).all()), name)
        self.assertTrue(any(float(derivative.norm()) > 0 for (name, _), derivative in zip(named, derivatives)
                            if '.attn.' in name))

    def test_probe_failure_preserves_weights_support_and_rng(self):
        net, query, support, truth = fixture()
        original_hash, support_hash, rng_hash = hash_state(net.state_dict()), hash_state(support), hash_state(rng_state())
        from unittest.mock import patch
        # A failure after the unchanged legacy branch must not mutate baseline.
        with patch('tools.local_cnn_interaction_candidate.clone_candidate', side_effect=RuntimeError('injected candidate failure')):
            with self.assertRaisesRegex(RuntimeError, 'candidate failure'):
                probe_interaction(net, query, support, scales=[0, 1], truth=truth, batch=3,
                    support_record_ids=[str(i) for i in range(12)], query_group='query')
        self.assertEqual(hash_state(net.state_dict()), original_hash)
        self.assertEqual(hash_state(support), support_hash)
        self.assertEqual(hash_state(rng_state()), rng_hash)

    def test_cuda_full_support_probe_own_plans_complete_coverage_and_immutable_state(self):
        if not torch.cuda.is_available():
            self.skipTest('Real CUDA downstream matched-support probe required')
        net, query, support, truth = fixture('cuda')
        before, support_before, rng_before = hash_state(net.state_dict()), hash_state(support), hash_state(rng_state())
        report = probe_interaction(net, query, support, scales=[0., .25, 1.], truth=truth, batch=3,
            support_record_ids=[str(i) for i in range(12)], query_group='query')
        self.assertEqual(report['architecture'], ARCHITECTURE)
        self.assertEqual(report['query_candidates'], 7); self.assertEqual(report['physical_batch'], 3)
        self.assertTrue(report['scale_zero_production_parity_passed'])
        self.assertFalse(report['production_checkpoint_written']); self.assertFalse(report['production_ready'])
        self.assertEqual(report['new_trainable_parameters'], 0)
        self.assertEqual(len(report['branches']), 4)
        for branch in report['branches']:
            self.assertTrue(branch['support_history_and_teacher_refitted'])
            self.assertTrue(branch['support_query_equation_matched'])
            self.assertEqual(branch['support_records'], 12)
            self.assertEqual(len(branch['layers']), 2)
            self.assertTrue(all(layer['attention']['distinct_candidate_pairs'] == 21 for layer in branch['layers']))
        self.assertLess(report['branches'][1]['original_logit_max_abs_difference'], 2e-6)
        self.assertEqual(hash_state(net.state_dict()), before)
        self.assertEqual(hash_state(support), support_before)
        self.assertEqual(hash_state(rng_state()), rng_before)
        json.dumps(report, allow_nan=False)

    def test_invalid_edge_finite_support_binding_and_training_rejected(self):
        net, query, support, truth = fixture()
        layer = clone_candidate(net, 1.).l1[0]
        src, dst, edge = complete_edges(support[0], query)
        bad = src.clone(); bad[0] = 99
        with self.assertRaisesRegex(ValueError, 'outside'):
            attention_terms(layer, support[0], query, bad, dst, edge)
        bad_query = query.clone(); bad_query[0, 0] = float('nan')
        with self.assertRaisesRegex(FloatingPointError, 'Nonfinite'):
            attention_terms(layer, support[0], bad_query, src, dst, edge)
        with self.assertRaisesRegex(ValueError, 'record IDs'):
            probe_interaction(net, query, support, scales=[0, 1], truth=truth, batch=3,
                support_record_ids=['duplicate']*12, query_group='query')
        net.train()
        with self.assertRaisesRegex(ValueError, 'eval'):
            probe_interaction(net, query, support, scales=[0, 1], truth=truth, batch=3,
                support_record_ids=[str(i) for i in range(12)], query_group='query')


if __name__ == '__main__':
    unittest.main()
