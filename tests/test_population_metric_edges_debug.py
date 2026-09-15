"""CPU DEBUG checks for explicit L2 metrics, not medical/downstream validation.

The native upper-graph fixture retains H128, four heads, 24 regions, 16
prototypes and a physical batch of two. L0 embeddings are injected at their
documented boundary; this is not a production-size CNN or training run.
"""

import copy
import unittest
from unittest import mock

import numpy as np
import torch

from hiercp.hierarchy import PROTOTYPE_EDGE_DIMS, _prototype_edge_attributes
from hiercp.loss import CurriculumConfig, curriculum_ranking_loss
from hiercp.model import HierarchicalPyGPlacementModel
from hiercp.schema import POPULATION_METRIC_VERSION
from tests import test_source_content_contract_debug as native_debug
from tools import causality


class PopulationMetricMathDebugTests(unittest.TestCase):
    def test_new_diagnostic_preserves_existing_condition_seed_indices(self):
        previous = (
            "source_address_noise", "upper_node_order", "upper_context", "node_order",
            "upper_position_noise", "upper_clearance_noise", "target_context", "source_context",
            "edge_attr_zero", "topology_shuffle", "view1_only",
        )
        self.assertEqual(tuple(causality.CONDITIONS), (*previous, "population_metric"))

    def attributes(self, distances=None, dispersion=None, count=2):
        return _prototype_edge_attributes(
            np.zeros((1, 3), np.float32), np.zeros((count, 3), np.float32),
            np.stack([np.zeros(count, np.int64), np.arange(count)]),
            np.full(count, .5, np.float32), np.arange(count, dtype=np.float32),
            descriptor_distance=distances, dispersion_reference=dispersion,
        )

    def test_equal_relative_assignment_and_xyz_do_not_hide_absolute_novelty(self):
        # Two standardized 16D centers and an orthogonal query displacement.
        centers = np.zeros((2, 16), np.float32)
        centers[1, 3] = 2.
        queries = np.zeros((2, 16), np.float32)
        queries[:, 3] = 1.
        queries[1, 4] = 10.
        squared = ((queries[:, None] - centers[None]) ** 2).sum(-1)
        logits = -squared / .5
        weights = np.exp(logits - logits.max(1, keepdims=True))
        weights /= weights.sum(1, keepdims=True)
        np.testing.assert_array_equal(weights[0], weights[1])
        near = self.attributes(np.sqrt(squared[0]), np.array([.25, .5], np.float32))
        far = self.attributes(np.sqrt(squared[1]), np.array([.25, .5], np.float32))
        np.testing.assert_array_equal(near[:, :6], far[:, :6])
        self.assertTrue(np.all(far[:, 6:] > near[:, 6:]))

    def test_signed_excess_and_zero_dispersion_are_finite_without_epsilon(self):
        attributes = self.attributes(np.array([0., .25]), np.array([0., .5]))
        np.testing.assert_array_equal(attributes[:, 6], [0., .25])
        np.testing.assert_array_equal(attributes[:, 7], [0., -.25])
        self.assertTrue(np.isfinite(attributes).all())

    def test_nonmetric_and_empty_relations_have_typed_widths(self):
        self.assertEqual(self.attributes().shape, (2, 6))
        self.assertEqual(self.attributes(count=0).shape, (0, 6))
        self.assertEqual(self.attributes(np.empty(0), np.empty(0), count=0).shape, (0, 8))
        self.assertEqual(sorted(PROTOTYPE_EDGE_DIMS.values()), [6, 6, 8, 8, 8])

    def test_invalid_metrics_are_errors_not_zero_fallbacks(self):
        for distance, dispersion in (
            (np.ones(2), None), (None, np.ones(2)),
            (np.array([np.nan, 1.]), np.ones(2)),
            (np.array([-1., 1.]), np.ones(2)),
            (np.ones(2), np.array([np.inf, 0.])),
            (np.ones(2), np.array([-.1, 0.])),
            (np.ones(1), np.ones(2)),
        ):
            with self.subTest(distance=distance, dispersion=dispersion), self.assertRaises(ValueError):
                self.attributes(distance, dispersion)


class PopulationMetricNativeUpperDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        original_builder = native_debug.build_prototype_bank
        captured = {}

        def capture_bank(*args, **kwargs):
            bank = original_builder(*args, **kwargs)
            captured["bank"] = bank
            return bank

        with mock.patch.object(native_debug, "build_prototype_bank", side_effect=capture_bank):
            cls.batch, cls.local = native_debug.native_upper_fixture()
        cls.bank = captured["bank"]
        torch.manual_seed(381)
        cls.model = HierarchicalPyGPlacementModel(
            hidden_dim=128, heads=4, local_layers=3, patient_layers=2,
            prototype_layers=2, dropout=0., dense_base_channels=12,
            dense_feature_dim=32, dense_batch_size=4,
        ).eval()

    @classmethod
    def tearDownClass(cls):
        del cls.model, cls.batch, cls.local, cls.bank
        torch.set_num_threads(cls.previous_threads)

    def test_native_builder_preserves_topology_and_exact_metric_definitions(self):
        for graph in self.batch.prototype_batch.to_data_list():
            self.assertEqual(graph.population_metric_version, POPULATION_METRIC_VERSION)
            self.assertEqual(graph["region"].num_nodes, 24)
            self.assertEqual(graph["prototype"].num_nodes, 16)
            forward = graph[("region", "assigned_to", "prototype")]
            reverse = graph[("prototype", "represents", "region")]
            self.assertEqual(forward.edge_index.shape[1], 24 * 2)
            torch.testing.assert_close(reverse.edge_index, forward.edge_index.flip(0), rtol=0, atol=0)
            torch.testing.assert_close(reverse.edge_attr[:, 6:], forward.edge_attr[:, 6:], rtol=0, atol=0)
            distances = self.bank.standardized_distances(graph["region"].raw_x.numpy())
            source, target = forward.edge_index.numpy()
            np.testing.assert_allclose(forward.edge_attr[:, 6].numpy(), distances[source, target], rtol=1e-6)
            np.testing.assert_allclose(
                forward.edge_attr[:, 7].numpy(),
                distances[source, target] - self.bank.cluster_mean_distance[target], rtol=1e-6, atol=1e-7,
            )
            relation = graph[("prototype", "similar_to", "prototype")]
            np.testing.assert_array_equal(relation.edge_index.numpy(), self.bank.edge_index)
            source, target = relation.edge_index.numpy()
            distance = np.linalg.norm(
                self.bank.standardized_centers[source] - self.bank.standardized_centers[target], axis=1)
            reference = .5 * (self.bank.cluster_mean_distance[source] + self.bank.cluster_mean_distance[target])
            np.testing.assert_allclose(relation.edge_attr[:, 6].numpy(), distance, rtol=1e-6)
            np.testing.assert_allclose(relation.edge_attr[:, 7].numpy(), distance - reference, rtol=1e-6, atol=1e-7)
            for edge, width in PROTOTYPE_EDGE_DIMS.items():
                self.assertEqual(graph[edge].edge_attr.shape[1], width)

    def test_every_new_field_and_layer_column_receives_ranking_gradient(self):
        changed = copy.deepcopy(self.batch)
        self.model.zero_grad(set_to_none=True)
        for edge, width in PROTOTYPE_EDGE_DIMS.items():
            if width == 8:
                changed.prototype_batch[edge].edge_attr.requires_grad_()
        scores = self.model._score_full(changed, self.local)
        loss, _ = curriculum_ranking_loss(scores, changed.difficulties, epoch=30, config=CurriculumConfig())
        loss.backward()
        for edge, width in PROTOTYPE_EDGE_DIMS.items():
            if width == 8:
                gradient = changed.prototype_batch[edge].edge_attr.grad
                self.assertIsNotNone(gradient, edge)
                self.assertTrue(torch.isfinite(gradient).all(), edge)
                self.assertTrue((gradient[:, 6:].abs().sum(0) > 0).all(), edge)
        for block in self.model.prototype_encoder.blocks:
            for edge, width in PROTOTYPE_EDGE_DIMS.items():
                projection = block.conv.convs[edge].lin_edge
                self.assertEqual(projection.in_channels, width)
                if width == 8:
                    self.assertTrue((projection.weight.grad[:, 6:].abs().sum(0) > 0).all(), edge)
        self.model.zero_grad(set_to_none=True)

    def test_metric_intervention_preserves_spatial_fields_dispersion_and_reverse_edges(self):
        original = copy.deepcopy(self.batch)
        changed = causality._condition_population_metric(copy.deepcopy(self.batch), 314)
        for before, after in zip(original.prototype_batch.to_data_list(), changed.prototype_batch.to_data_list()):
            for edge, width in PROTOTYPE_EDGE_DIMS.items():
                torch.testing.assert_close(before[edge].edge_index, after[edge].edge_index, rtol=0, atol=0)
                torch.testing.assert_close(before[edge].edge_attr[:, :6], after[edge].edge_attr[:, :6], rtol=0, atol=0)
                if width == 8:
                    old, new = before[edge].edge_attr, after[edge].edge_attr
                    self.assertTrue((new[:, 6] > old[:, 6]).all(), edge)
                    torch.testing.assert_close(new[:, 6] - new[:, 7], old[:, 6] - old[:, 7], rtol=1e-5, atol=1e-6)
            forward = after[("region", "assigned_to", "prototype")]
            reverse = after[("prototype", "represents", "region")]
            torch.testing.assert_close(forward.edge_attr[:, 6:], reverse.edge_attr[:, 6:], rtol=0, atol=0)
            pair = after[("prototype", "similar_to", "prototype")]
            before_pair = before[("prototype", "similar_to", "prototype")]
            shifts = pair.edge_attr[:, 6] - before_pair.edge_attr[:, 6]
            values = {tuple(edge): value for edge, value in zip(pair.edge_index.T.tolist(), shifts)}
            for (left, right), value in values.items():
                if (right, left) in values:
                    torch.testing.assert_close(value, values[(right, left)], rtol=1e-5, atol=1e-6)
        # Caller-owned source graphs remain byte-identical.
        for before, current in zip(original.prototype_batch.to_data_list(), self.batch.prototype_batch.to_data_list()):
            for edge in PROTOTYPE_EDGE_DIMS:
                torch.testing.assert_close(before[edge].edge_attr, current[edge].edge_attr, rtol=0, atol=0)

    def test_isomorphic_node_reindexing_keeps_all_metric_fields_on_their_edges(self):
        changed = causality._condition_upper_node_order(copy.deepcopy(self.batch), 91)
        for before, after in zip(self.batch.prototype_batch.to_data_list(), changed.prototype_batch.to_data_list()):
            for edge in PROTOTYPE_EDGE_DIMS:
                # Endpoints are reindexed, not reordered: attributes stay with
                # the same logical directed edge, including new columns 6/7.
                torch.testing.assert_close(before[edge].edge_attr, after[edge].edge_attr, rtol=0, atol=0)
        with torch.no_grad():
            expected = self.model._score_full(self.batch, self.local)
            actual = self.model._score_full(changed, self.local)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=3e-4, atol=3e-5)

    def test_upper_context_is_explicitly_a_node_only_intervention(self):
        changed = causality._condition_upper_context(copy.deepcopy(self.batch), 818)
        self.assertFalse(torch.equal(changed.prototype_batch["region"].raw_x, self.batch.prototype_batch["region"].raw_x))
        for edge in PROTOTYPE_EDGE_DIMS:
            torch.testing.assert_close(changed.prototype_batch[edge].edge_attr, self.batch.prototype_batch[edge].edge_attr, rtol=0, atol=0)

    def test_each_metric_changes_relative_scores_without_cross_patient_leak(self):
        with torch.no_grad():
            expected = self.model._score_full(self.batch, self.local)
            for column in (6, 7):
                changed = copy.deepcopy(self.batch)
                for edge, width in PROTOTYPE_EDGE_DIMS.items():
                    if width == 8:
                        source = changed.prototype_batch[edge].edge_index[0]
                        selected = changed.prototype_batch[edge[0]].batch[source] == 0
                        values = changed.prototype_batch[edge].edge_attr
                        values[selected, column] += torch.linspace(.2, 2., int(selected.sum()))
                actual = self.model._score_full(changed, self.local)
                self.assertGreater(float(((actual[0] - actual[0][0]) - (expected[0] - expected[0][0])).abs().max()), 1e-7)
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_no_population_has_no_metric_effect_or_trainable_population_weights(self):
        torch.manual_seed(381)
        model = HierarchicalPyGPlacementModel(
            hidden_dim=128, heads=4, local_layers=3, patient_layers=2,
            prototype_layers=2, dropout=0., dense_base_channels=12,
            dense_feature_dim=32, dense_batch_size=4, ablation_mode="no_population",
        ).eval()
        changed = copy.deepcopy(self.batch)
        for edge, width in PROTOTYPE_EDGE_DIMS.items():
            if width == 8:
                changed.prototype_batch[edge].edge_attr[:, 6:] += 11.
        with torch.no_grad():
            expected = model._score_active(self.batch, self.local)
            actual = model._score_active(changed, self.local)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertIsNone(model.population_readout)
        self.assertFalse(any(parameter.requires_grad for parameter in model.prototype_encoder.parameters()))

    def test_uniform_absolute_distance_shift_survives_actual_full_readout(self):
        # Isolate the new representation: keep every node, old six edge fields,
        # top-M weight/rank and prototype-pair edge fixed. Increase every
        # first-patient region/prototype distance AND excess by exactly one
        # standardized-distance unit, not an edge-varying perturbation.
        changed = copy.deepcopy(self.batch)
        for edge in (("region", "assigned_to", "prototype"),
                     ("prototype", "represents", "region")):
            source = changed.prototype_batch[edge].edge_index[0]
            selected = changed.prototype_batch[edge[0]].batch[source] == 0
            changed.prototype_batch[edge].edge_attr[selected, 6:] += 1.
            torch.testing.assert_close(changed.prototype_batch[edge].edge_attr[:, :6],
                                       self.batch.prototype_batch[edge].edge_attr[:, :6], rtol=0, atol=0)
        with torch.no_grad():
            expected = self.model._score_full(self.batch, self.local)
            actual = self.model._score_full(changed, self.local)
        relative_delta = (actual[0] - actual[0][0]) - (expected[0] - expected[0][0])
        self.assertGreater(float(relative_delta.abs().max()), 1e-7)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_legacy_marker_width_and_weights_are_not_relabelled(self):
        with torch.no_grad():
            changed = copy.deepcopy(self.batch)
            changed.prototype_batch.population_metric_version = ["legacy"] * 2
            with self.assertRaisesRegex(ValueError, "metric contract"):
                self.model._score_full(changed, self.local)
            changed = copy.deepcopy(self.batch)
            edge = ("region", "assigned_to", "prototype")
            changed.prototype_batch[edge].edge_attr = changed.prototype_batch[edge].edge_attr[:, :6]
            with self.assertRaisesRegex(ValueError, "8 edge fields"):
                self.model._score_full(changed, self.local)
        legacy = self.model.state_dict()
        legacy["_architecture_revision"] = torch.tensor(4, dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "historical comparisons"):
            self.model.load_state_dict(legacy)


if __name__ == "__main__":
    unittest.main()
