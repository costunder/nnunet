"""Independent UNIT checks for explicit warm-start relation controls.

Fixtures are 128D, two-layer, four-head models; these are not CT accuracy
results.  The official operator parity oracle remains in the existing separate
``test_local_cnn_reference_l1`` suite, which this change does not replace.
"""
import copy
import inspect
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from hiercp_v222.model import PromptGraphModel
from hiercp_v222.v1_execution import rng_state
from l0_local_cnn.model import LocalCNN
from l0_regions.training import hash_state
from tools.local_cnn_reference_l1 import clone_reference, ReferenceRelationLayer
from tools.local_cnn_reference_transfer import POLICIES, clone_reference_control
from tests.test_local_cnn_reference_l1 import official_class


ROOT = Path(__file__).resolve().parents[1]


def fixture(device='cpu', dtype=torch.float64):
    torch.manual_seed(7013)
    config = json.loads((ROOT/'config/prompt_graph_v222_v1_l0.json').read_text())
    base = json.loads((ROOT/'config/train.json').read_text())
    return PromptGraphModel(config, base, {}, local_encoder=nn.Identity()).to(device, dtype=dtype).eval()


class ReferenceTransferChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def _tfu_equivalence(self, device):
        for dtype in (torch.float32, torch.float64):
            original = fixture(device, dtype=dtype)
            old_codes = torch.tensor([[1., 1.], [1., 0.], [0., 0.]], dtype=dtype, device=device)
            new_codes = torch.tensor([[0., 1.], [0., -1.], [1., 0.]], dtype=dtype, device=device)
            tolerance = 16*torch.finfo(dtype).eps
            for state in ('random', 'learned'):
                if state == 'learned':
                    # Independently fit the old edge projection to a known target;
                    # remapping must work for trained values as well as initialization.
                    for old in original.l1:
                        optimizer = torch.optim.SGD(old.edge.parameters(), lr=.1)
                        target = torch.randn(3, 128, device=device, dtype=dtype)
                        for _ in range(3):
                            optimizer.zero_grad(set_to_none=True)
                            (old.edge(old_codes)-target).square().mean().backward()
                            optimizer.step()
                for policy in ('affine_relations', 'affine_relations_zero_out_bias'):
                    candidate, metadata = clone_reference_control(original, policy=policy)
                    for index, (old, new) in enumerate(zip(original.l1, candidate.l1)):
                        expected = torch.stack((old.edge.weight[:, 0]+old.edge.weight[:, 1],
                                                old.edge.weight[:, 0],
                                                old.edge.weight.new_zeros(128)))
                        projected = F.linear(new_codes, new.lin_edge.weight, new.lin_edge.bias)
                        torch.testing.assert_close(projected, expected, atol=tolerance, rtol=tolerance)
                        torch.testing.assert_close(projected, old.edge(old_codes), atol=tolerance, rtol=tolerance)
                        report = metadata['relation_transfer'][index]
                        self.assertTrue(report['preserved_projected_relations'])
                        self.assertFalse(report['full_layer_function_preserved'])
                        self.assertLessEqual(max(report['projected_TFU_max_abs_error'].values()),
                                             report['floating_absolute_tolerance'])
                        self.assertFalse(report['self_projection']['old_counterpart'])
                        # New self-loop relation is the affine bias, not old U.
                        torch.testing.assert_close(new.lin_edge(torch.zeros(1, 2, device=device, dtype=dtype))[0],
                                                    old.edge.weight[:, 0]+old.edge.weight[:, 1]/2,
                                                    atol=tolerance, rtol=tolerance)

    def test_tfu_projections_preserved_for_random_and_learned_weights_cpu(self):
        self._tfu_equivalence('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'Real CUDA unavailable')
    def test_tfu_projections_preserved_for_random_and_learned_weights_cuda(self):
        self._tfu_equivalence('cuda')

    def test_raw_columns_matches_historical_candidate_bitwise(self):
        original = fixture(dtype=torch.float32)
        historical, _ = clone_reference(original)
        candidate, metadata = clone_reference_control(original, policy='raw_columns')
        self.assertEqual(hash_state(historical.state_dict()), hash_state(candidate.state_dict()))
        self.assertTrue(all(not report['preserved_projected_relations'] for report in metadata['relation_transfer']))
        self.assertEqual(metadata['transfer_policy'], 'raw_columns')
        for old, new in zip(original.l1, candidate.l1):
            torch.testing.assert_close(new.lin_edge.weight, old.edge.weight, atol=0, rtol=0)
            torch.testing.assert_close(new.lin_edge.bias, torch.zeros_like(new.lin_edge.bias), atol=0, rtol=0)

    def test_policy_is_required_and_invalid_choices_are_not_inferred(self):
        original = fixture()
        self.assertIs(inspect.signature(clone_reference_control).parameters['policy'].default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            clone_reference_control(original)
        for policy in (None, '', 'automatic', 'affine', 3):
            with self.assertRaisesRegex(ValueError, 'Explicit reference transfer policy'):
                clone_reference_control(original, policy=policy)
        self.assertEqual(len(POLICIES), 3)

    def test_only_explicit_edge_projection_and_output_bias_tensors_differ(self):
        original = fixture()
        baseline, _ = clone_reference(original)
        for policy in POLICIES:
            candidate, metadata = clone_reference_control(original, policy=policy)
            self.assertEqual(type(candidate), type(baseline))
            self.assertIs(type(candidate.l1[0]).forward, ReferenceRelationLayer.forward)
            self.assertEqual(len(candidate.l1), 2)
            self.assertEqual(candidate.dim, 128)
            self.assertEqual([layer.heads for layer in candidate.l1], [4, 4])
            for key, old_value in baseline.state_dict().items():
                if policy != 'raw_columns' and key.endswith(('lin_edge.weight', 'lin_edge.bias')):
                    continue
                if policy == 'affine_relations_zero_out_bias' and key.endswith('out_proj.bias'):
                    continue
                torch.testing.assert_close(candidate.state_dict()[key], old_value, atol=0, rtol=0)
            self.assertFalse(metadata['exact_resume'])
            self.assertFalse(metadata['batch_norm_calibrated'])
            self.assertTrue(metadata['operator_equations_unchanged_from_reference'])

    def test_zero_output_bias_removes_degree_bias_without_reinterpreting_operator(self):
        original = fixture()
        copied, _ = clone_reference_control(original, policy='affine_relations')
        zeroed, metadata = clone_reference_control(original, policy='affine_relations_zero_out_bias')
        # Nodes have incoming degrees 3, 2, 1, 0. Both operators retain the
        # official edge-wise projection; only the copied bias differs.
        edges = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 2]])
        degree = torch.tensor([3., 2., 1., 0.], dtype=torch.float64)
        attributes = torch.tensor([[0., 1.], [0., -1.], [1., 0.], [0., 0.], [0., 1.], [1., 0.]], dtype=torch.float64)
        nodes = torch.randn(4, 128, dtype=torch.float64)
        for index, (left, right) in enumerate(zip(copied.l1, zeroed.l1)):
            left.eval(); right.eval()
            output_left, terms_left = left.forward_with_terms(nodes, edges, attributes)
            output_right, terms_right = right.forward_with_terms(nodes, edges, attributes)
            bias = original.l1[index].update.out.bias
            expected = degree[:, None]*bias[None]
            torch.testing.assert_close(terms_left['pre_bn']-terms_right['pre_bn'], expected, atol=2e-14, rtol=2e-13)
            torch.testing.assert_close(output_left-output_right,
                                       expected/math.sqrt(1+left.bn.eps), atol=2e-14, rtol=2e-13)
            # The old layer contributed one bias; zeroing official bias loses
            # that term and cannot be advertised as exact function transfer.
            audit = metadata['out_bias_transfer'][index]
            self.assertEqual(audit['shift_relative_old_once_per_node_bias'], '-b_old')
            self.assertFalse(audit['old_once_per_node_bias_effect_preserved'])
            self.assertFalse(audit['full_layer_function_preserved'])
            self.assertEqual(audit['new_output_bias_norm'], 0.)

    def test_original_state_gradients_rng_mixed_modes_extra_state_unchanged(self):
        original = fixture()
        original.local = LocalCNN(json.loads((ROOT/'config/v22_local_cnn.json').read_text())).double()
        original.train(); original.l1[0].eval(); original.local.eval()
        original.label_seed.grad = torch.full_like(original.label_seed, .17)
        original_state = hash_state(original.state_dict())
        original_gradients = hash_state({name: p.grad for name, p in original.named_parameters()})
        modes = tuple(module.training for module in original.modules())
        before_rng = hash_state(rng_state())
        for policy in POLICIES:
            candidate, _ = clone_reference_control(original, policy=policy)
            self.assertEqual(hash_state(original.state_dict()), original_state)
            self.assertEqual(hash_state({name: p.grad for name, p in original.named_parameters()}), original_gradients)
            self.assertEqual(tuple(module.training for module in original.modules()), modes)
            self.assertEqual(hash_state(rng_state()), before_rng)
            self.assertEqual(hash_state(candidate.local.state_dict()), hash_state(original.local.state_dict()))
            self.assertEqual(candidate.local.config, original.local.config)
            candidate.local.config['margin_mm'] = 20
            self.assertEqual(original.local.config['margin_mm'], 10)

    def test_failure_preserves_original_and_rng(self):
        original = fixture()
        state, random_state = hash_state(original.state_dict()), hash_state(rng_state())
        with patch('tools.local_cnn_reference_transfer.clone_reference', side_effect=RuntimeError('UNIT clone failure')):
            with self.assertRaisesRegex(RuntimeError, 'UNIT clone failure'):
                clone_reference_control(original, policy='affine_relations')
        self.assertEqual(hash_state(original.state_dict()), state)
        self.assertEqual(hash_state(rng_state()), random_state)

    def _official_equation_parity(self, device):
        original = fixture(device)
        Oracle = official_class()
        source = torch.arange(8, device=device).repeat_interleave(3)
        destination = (source + torch.arange(24, device=device).remainder(3)) % 8
        edges = torch.stack((source, destination))
        codes = torch.tensor([[0., 1.], [0., -1.], [1., 0.], [0., 0.]],
                             device=device, dtype=torch.float64).repeat(6, 1)
        coefficients = torch.randn(8, 128, device=device, dtype=torch.float64)
        for policy in POLICIES:
            candidate, _ = clone_reference_control(original, policy=policy)
            layer = candidate.l1[0].eval()
            oracle = Oracle(2, 128, heads=4, dropout=.1, batch_norm=True).to(device, dtype=torch.float64).eval()
            oracle.load_state_dict(layer.state_dict(), strict=True)
            nodes = torch.randn(8, 128, device=device, dtype=torch.float64)
            output, derivatives = [], []
            for module in (layer, oracle):
                live_nodes = nodes.detach().clone().requires_grad_()
                result = module(live_nodes, edges, codes)
                output.append(result)
                derivatives.append(torch.autograd.grad((result*coefficients).mean(),
                                    [live_nodes]+list(module.parameters())))
            torch.testing.assert_close(*output, atol=1e-12, rtol=1e-12)
            for left, right in zip(*derivatives):
                torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)

    def test_all_policies_keep_official_operator_output_and_gradient_cpu(self):
        self._official_equation_parity('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'Real CUDA unavailable')
    def test_all_policies_keep_official_operator_output_and_gradient_cuda(self):
        self._official_equation_parity('cuda')


if __name__ == '__main__':
    unittest.main()
