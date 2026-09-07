"""DEBUG operator/gradient checks; not medical-data, CUDA or final training.

Tiny synthetic graphs exercise the complete production attention operator with
an explicitly DEBUG-only small workspace to force destination-crossing chunks.
"""

import copy
import unittest
from unittest import mock

import torch
from torch_geometric.nn import GATv2Conv

from hiercp.model import CompatibilityGatedGATv2Conv


class EdgeAttentionStreamingDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.prior_threads)

    def setUp(self):
        torch.manual_seed(1701)
        self.model = CompatibilityGatedGATv2Conv(
            (6, 5), 3, heads=2, concat=True, edge_dim=4,
            add_self_loops=False, dropout=0.0,
        )
        # Each destination recurs in separated chunks, including parallel edges.
        self.edges = torch.stack((torch.arange(31) % 7, torch.arange(31) % 5))
        self.inputs = (torch.randn(7, 6), torch.randn(5, 5), torch.randn(31, 4))
        self.workspace = mock.patch(
            "hiercp.model.EDGE_ATTENTION_WORKSPACE_BYTES", 8 * 4 * 2 * 3 * 3,
        )
        self.workspace.start()
        self.addCleanup(self.workspace.stop)

    def compare_dense(self, model=None, edges=None, inputs=None):
        streamed = self.model if model is None else model
        dense = copy.deepcopy(streamed)
        edges = self.edges if edges is None else edges
        supplied = self.inputs if inputs is None else inputs
        left_inputs = tuple(value.detach().clone().requires_grad_() for value in supplied)
        right_inputs = tuple(value.detach().clone().requires_grad_() for value in supplied)
        torch.manual_seed(913)
        actual, (actual_edges, actual_attention) = streamed(
            left_inputs[:2], edges, left_inputs[2], return_attention_weights=True,
        )
        torch.manual_seed(913)
        expected, (expected_edges, expected_attention) = GATv2Conv.forward(
            dense, right_inputs[:2], edges, right_inputs[2], return_attention_weights=True,
        )
        torch.testing.assert_close(actual_edges, expected_edges, rtol=0, atol=0)
        torch.testing.assert_close(actual_attention, expected_attention, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
        coefficient = torch.linspace(-0.9, 1.3, actual.numel()).reshape_as(actual)
        (actual.square() * coefficient).sum().backward()
        (expected.square() * coefficient).sum().backward()
        for index, (left, right) in enumerate(zip(left_inputs, right_inputs)):
            with self.subTest(input_gradient=index):
                self.assertIsNotNone(left.grad)
                torch.testing.assert_close(left.grad, right.grad, rtol=3e-5, atol=3e-6)
        self.assertEqual(set(streamed.state_dict()), set(dense.state_dict()))
        for (name, left), (other, right) in zip(streamed.named_parameters(), dense.named_parameters()):
            self.assertEqual(name, other)
            with self.subTest(parameter_gradient=name):
                self.assertIsNotNone(left.grad)
                torch.testing.assert_close(left.grad, right.grad, rtol=3e-5, atol=3e-6)
        return streamed

    def test_destination_crossing_chunks_forward_and_every_gradient_match_dense(self):
        self.assertEqual(self.model._edge_chunk_size(torch.float32), 3)
        self.compare_dense()

    def test_dropout_mask_and_gradient_match_dense_across_chunks(self):
        self.model.dropout = 0.35
        self.model.train()
        self.compare_dense()

    def test_empty_relation_keeps_dense_zero_gradient_connectivity(self):
        self.compare_dense(edges=torch.empty((2, 0), dtype=torch.long),
                           inputs=(*self.inputs[:2], torch.empty((0, 4))))

    def test_workspace_bounds_forward_and_backward_edge_hidden_work(self):
        observed = []
        original = self.model._logit_chunk

        def record(left, right, edges, attributes):
            observed.append(int(edges.shape[1]))
            return original(left, right, edges, attributes)

        retained_shapes = []
        retained_edge_hidden_bytes = 0

        def pack(value):
            nonlocal retained_edge_hidden_bytes
            retained_shapes.append(tuple(value.shape))
            # The only legitimate 3-D saved tensors are shared node projections.
            # Count every saved edge-hidden chunk, not merely a full-E shape.
            if value.ndim == 3 and tuple(value.shape) not in {(7, 2, 3), (5, 2, 3)}:
                retained_edge_hidden_bytes += value.numel() * value.element_size()
            return value

        with mock.patch.object(self.model, "_logit_chunk", side_effect=record):
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
                self.compare_dense_streaming_only()
        self.assertEqual(sum(observed[:11]), 31)
        self.assertGreaterEqual(len(observed), 22)
        self.assertTrue(all(0 <= count <= 3 for count in observed))
        self.assertNotIn((31, 2, 3), retained_shapes)
        self.assertEqual(retained_edge_hidden_bytes, 0)

    def test_message_chunks_allocate_only_one_complete_destination_output(self):
        with mock.patch("hiercp.model.torch.zeros", wraps=torch.zeros) as allocations:
            self.compare_dense_streaming_only()
        output_allocations = [call for call in allocations.call_args_list
                              if call.args and call.args[0] == (5, 2, 3)]
        self.assertEqual(len(output_allocations), 1)

    def compare_dense_streaming_only(self):
        source, target, attributes = (value.clone().requires_grad_() for value in self.inputs)
        self.model((source, target), self.edges, attributes).square().sum().backward()

    def test_homogeneous_shared_projection_self_loops_and_mean_heads(self):
        model = CompatibilityGatedGATv2Conv(
            6, 3, heads=2, concat=False, share_weights=True,
            add_self_loops=True, edge_dim=None, dropout=0.0,
        )
        dense = copy.deepcopy(model)
        source = torch.randn(7, 6, requires_grad=True)
        reference = source.detach().clone().requires_grad_()
        edges = torch.stack((torch.arange(31) % 7, (torch.arange(31) * 3) % 7))
        actual = model(source, edges)
        expected = GATv2Conv.forward(dense, reference, edges)
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(source.grad, reference.grad, rtol=3e-5, atol=3e-6)

    def test_double_precision_numerical_gradcheck(self):
        model = CompatibilityGatedGATv2Conv(
            (2, 2), 2, heads=1, edge_dim=1, add_self_loops=False,
            dropout=0.0,
        ).double()
        edges = torch.tensor([[0, 1, 0, 1, 0], [0, 1, 1, 0, 0]])
        inputs = (torch.randn(2, 2, dtype=torch.float64, requires_grad=True),
                  torch.randn(2, 2, dtype=torch.float64, requires_grad=True),
                  torch.randn(5, 1, dtype=torch.float64, requires_grad=True))
        with mock.patch("hiercp.model.EDGE_ATTENTION_WORKSPACE_BYTES", 8 * 8 * 2 * 2):
            self.assertTrue(torch.autograd.gradcheck(
                lambda left, right, attributes: model((left, right), edges, attributes),
                inputs, eps=1e-6, atol=2e-5, rtol=1e-3,
            ))

    def test_cpu_autocast_dropout_backward_is_finite(self):
        self.model.dropout = 0.35
        self.model.train()
        source, target, attributes = (value.clone().requires_grad_() for value in self.inputs)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = self.model((source, target), self.edges, attributes)
            loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        for value in (*self.model.parameters(), source, target, attributes):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_complete_hierarchy_nested_checkpoint_dropout_replay_and_optimizer(self):
        from test_hierarchy_model_debug import debug_batch
        from hiercp.loss import CurriculumConfig, curriculum_ranking_loss
        from hiercp.model import HierarchicalPyGPlacementModel

        # DEBUG width/patch only; all configured 3/2/2 propagation depths and
        # every relation remain present in one physical two-patient graph batch.
        model = HierarchicalPyGPlacementModel(
            hidden_dim=16, heads=4, local_layers=3, patient_layers=2, prototype_layers=2,
            dropout=0.1, dense_base_channels=4, dense_feature_dim=8, dense_batch_size=4,
            channels_last_3d=False, checkpoint_local_blocks=True,
            checkpoint_dense_encoder=True,
        ).train()
        replay = copy.deepcopy(model)
        batch = debug_batch()
        initial = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=0.001)
        with mock.patch("hiercp.model.EDGE_ATTENTION_WORKSPACE_BYTES", 8 * 4 * 16 * 17):
            results = []
            for network in (model, replay):
                torch.manual_seed(551)
                output = network(copy.deepcopy(batch))
                ranking, _ = curriculum_ranking_loss(
                    output.scores, batch.difficulties, epoch=30, config=CurriculumConfig(),
                )
                loss = ranking + 0.1 * output.consistency
                loss.backward()
                results.append([score.detach() for score in output.scores])
            for actual, expected in zip(*results):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for (name, parameter), (other, reference) in zip(model.named_parameters(), replay.named_parameters()):
                self.assertEqual(name, other)
                if not parameter.requires_grad:
                    continue
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)
        optimizer.step()
        for prefix in ("local_encoder", "patient_encoder", "prototype_encoder", "score_head"):
            self.assertTrue(any(
                not torch.equal(initial[name], parameter.detach())
                for name, parameter in model.named_parameters() if name.startswith(prefix)
            ), prefix)


if __name__ == "__main__":
    unittest.main()
