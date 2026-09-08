"""DEBUG only: synthetic tensor/graph smoke, not clinical or GPU validation.

Reuse the existing explicit DEBUG widths/patches while retaining every relation
and the complete 3/2/2 hierarchy. Production configurations are never modified.
"""
from __future__ import annotations

import copy
import math
import unittest

import torch
from torch_geometric.data import Batch

from hiercp.data import HierarchicalBatch
from hiercp.schema import (
    LOCAL_EDGE_DIM, LOCAL_EDGE_TYPES, PATIENT_EDGE_DIM, PATIENT_EDGE_TYPES,
    PROTOTYPE_EDGE_DIM, PROTOTYPE_EDGE_TYPES,
)
from tools import causality
from test_hierarchy_model_debug import debug_batch, debug_model


EMPTY_LOCAL_RELATION = ("source_context", "corresponds_to", "target_context")


def debug_causality_batch(*, layout):
    source = debug_batch(empty_lesions=layout == "all_empty")
    patients = source.patient_batch.to_data_list()
    if layout == "mixed":
        patients[0]["lesion"].raw_x = patients[0]["lesion"].raw_x[:0].clone()
        patients[0]["lesion"].num_nodes = 0
        for relation in PATIENT_EDGE_TYPES:
            if "lesion" in (relation[0], relation[2]):
                patients[0][relation].edge_index = torch.empty((2, 0), dtype=torch.long)
                patients[0][relation].edge_attr = torch.empty((0, PATIENT_EDGE_DIM))
    elif layout != "all_empty":
        raise ValueError(f"Unknown DEBUG fixture layout: {layout}")
    # Exercise both raw_x and the optional explicit position matrix branches.
    for patient in patients:
        for node_type in ("tumor", "candidate"):
            patient[node_type].pos = patient[node_type].raw_x[:, :3].clone()
    populations = source.prototype_batch.to_data_list()
    for population in populations:
        population["candidate"].pos = population["candidate"].raw_x[:, :3].clone()
    local_views = []
    for view in (source.local_batch, source.local_batch_view2):
        graphs = view.to_data_list()
        # Optional correspondence may be empty. Retain its relation key, both
        # endpoint node sets, every other edge, and the exact attribute width.
        graphs[0][EMPTY_LOCAL_RELATION].edge_index = torch.empty((2, 0), dtype=torch.long)
        graphs[0][EMPTY_LOCAL_RELATION].edge_attr = torch.empty((0, LOCAL_EDGE_DIM))
        local_views.append(Batch.from_data_list(graphs))
    return HierarchicalBatch(
        source_patches=source.source_patches, target_patches=source.target_patches,
        local_batch=local_views[0], local_batch_view2=local_views[1],
        patient_batch=Batch.from_data_list(patients),
        prototype_batch=Batch.from_data_list(populations),
        difficulties=torch.cat(source.difficulties), counts=source.counts,
        case_ids=("DEBUG_patient_empty", "DEBUG_patient_second"),
    )


class CausalityTransformsDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)  # Explicit DEBUG CPU profile, not final runtime.

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_nested_equal(self, expected, actual):
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(expected, actual):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def assert_batch_unchanged(self, expected, actual):
        for name in ("source_patches", "target_patches", "difficulties", "counts", "case_ids"):
            self.assert_nested_equal(getattr(expected, name), getattr(actual, name))
        for name in ("local_batch", "local_batch_view2", "patient_batch", "prototype_batch"):
            before, after = getattr(expected, name), getattr(actual, name)
            if before is None:
                self.assertIsNone(after)
            else:
                self.assert_nested_equal(before.to_dict(), after.to_dict())

    def assert_graph_schema(self, batch):
        for graph, relations, width in (
            (batch.local_batch, LOCAL_EDGE_TYPES, LOCAL_EDGE_DIM),
            (batch.local_batch_view2, LOCAL_EDGE_TYPES, LOCAL_EDGE_DIM),
            (batch.patient_batch, PATIENT_EDGE_TYPES, PATIENT_EDGE_DIM),
            (batch.prototype_batch, PROTOTYPE_EDGE_TYPES, PROTOTYPE_EDGE_DIM),
        ):
            if graph is None:
                continue
            self.assertEqual(set(graph.edge_types), set(relations))
            self.assertTrue(graph.validate(raise_on_error=True))
            for relation in relations:
                self.assertEqual(graph[relation].edge_index.shape[0], 2)
                self.assertEqual(tuple(graph[relation].edge_attr.shape),
                                 (graph[relation].edge_index.shape[1], width))

    def test_zero_rows_preserve_width_dtype_input_and_rng(self):
        for dtype in (torch.float32, torch.float64):
            for columns in ((0,), (0, 1, 2, 3, 11), ()):
                with self.subTest(dtype=dtype, columns=columns):
                    value = torch.empty((0, 12), dtype=dtype)
                    rng = torch.get_rng_state().clone()
                    result = causality._noise_columns(value, columns, 42)
                    self.assertIsNot(result, value)
                    torch.testing.assert_close(result, value, rtol=0, atol=0)
                    self.assertEqual(result.dtype, dtype)
                    self.assertEqual(tuple(result.shape), (0, 12))
                    self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_zero_columns_preserve_empty_and_nonempty_matrices(self):
        for dtype in (torch.float32, torch.float64):
            for rows, width in ((0, 0), (0, 6), (1, 0), (4, 0), (1, 6), (4, 6)):
                with self.subTest(dtype=dtype, shape=(rows, width)):
                    value = torch.arange(rows * width, dtype=dtype).reshape(rows, width)
                    original = value.clone()
                    result = causality._noise_columns(value, (), 73)
                    self.assertIsNot(result, value)
                    torch.testing.assert_close(result, original, rtol=0, atol=0)
                    torch.testing.assert_close(value, original, rtol=0, atol=0)

    def test_nonempty_output_matches_original_seeded_math_exactly(self):
        for dtype in (torch.float32, torch.float64):
            for rows in (1, 7):
                for seed in (-17, 0, 42, 10_009):
                    for columns in ((0,), (3, 1), (0, 1, 2, 3, 4)):
                        with self.subTest(dtype=dtype, rows=rows, seed=seed, columns=columns):
                            value = torch.arange(rows * 5, dtype=dtype).reshape(rows, 5)
                            expected = value.clone()
                            base = torch.arange(rows * len(columns), dtype=torch.float32)
                            noise = torch.sin(base + float(int(seed) % 10_007)) * 1.75
                            expected[:, list(columns)] = noise.reshape(rows, -1).to(dtype)
                            result = causality._noise_columns(value, columns, seed)
                            torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_noncontiguous_input_other_columns_and_global_rng_are_untouched(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                value = torch.arange(35, dtype=dtype).reshape(5, 7).T
                self.assertFalse(value.is_contiguous())
                original, rng = value.clone(), torch.get_rng_state().clone()
                changed = causality._noise_columns(value, (3, 1), 91)
                torch.testing.assert_close(value, original, rtol=0, atol=0)
                torch.testing.assert_close(changed[:, [0, 2, 4]], original[:, [0, 2, 4]], rtol=0, atol=0)
                self.assertEqual(changed.dtype, dtype)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                self.assertNotEqual(changed.data_ptr(), value.data_ptr())

    def test_noise_keeps_unmodified_columns_connected_to_input_gradient(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                value = torch.arange(20, dtype=dtype).reshape(4, 5).requires_grad_()
                output = causality._noise_columns(value, (1, 3), 44)
                output.sum().backward()
                expected = torch.ones_like(value)
                expected[:, [1, 3]] = 0
                torch.testing.assert_close(value.grad, expected, rtol=0, atol=0)

    def test_invalid_dimensions_still_raise(self):
        for shape in ((), (0,), (5,), (0, 3, 2), (2, 3, 4)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "Expected a matrix"):
                    causality._noise_columns(torch.empty(shape), (0,), 42)

    def test_invalid_column_indices_keep_existing_nonempty_errors(self):
        for rows in (1, 3):
            for columns in ((5,), (-6,), (0, 5)):
                with self.subTest(rows=rows, columns=columns):
                    with self.assertRaises(IndexError):
                        causality._noise_columns(torch.empty((rows, 5)), columns, 42)

    def test_debug_fixtures_retain_every_relation_and_real_empty_stores(self):
        for layout in ("mixed", "all_empty"):
            with self.subTest(layout=layout):
                batch = debug_causality_batch(layout=layout)
                self.assertEqual(batch.sample_count, 2)
                self.assertEqual(batch.counts, (4, 3))
                self.assert_graph_schema(batch)
                patients = batch.patient_batch.to_data_list()
                self.assertEqual([patient["lesion"].num_nodes for patient in patients],
                                 [0, 2] if layout == "mixed" else [0, 0])
                empty = patients[0][("tumor", "coexists_with", "lesion")]
                self.assertEqual(tuple(empty.edge_attr.shape), (0, PATIENT_EDGE_DIM))
                for view in (batch.local_batch, batch.local_batch_view2):
                    graphs = view.to_data_list()
                    self.assertEqual(tuple(graphs[0][EMPTY_LOCAL_RELATION].edge_attr.shape), (0, LOCAL_EDGE_DIM))
                    self.assertGreater(graphs[1][EMPTY_LOCAL_RELATION].edge_index.shape[1], 0)

    def test_every_transform_real_full_hierarchy_forward_and_delta_on_empty_relations(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(770)
            model = debug_model().eval()
            for layout in ("mixed", "all_empty"):
                batch = debug_causality_batch(layout=layout)
                original = copy.deepcopy(batch)
                baseline = causality._evaluate(model, copy.deepcopy(batch), device=torch.device("cpu"), amp=False)
                for name, transform in causality.CONDITIONS.items():
                    with self.subTest(layout=layout, condition=name):
                        changed = transform(copy.deepcopy(batch), 123)
                        self.assertEqual(changed.counts, batch.counts)
                        self.assert_graph_schema(changed)
                        evaluated = causality._evaluate(model, changed, device=torch.device("cpu"), amp=False)
                        self.assertEqual(tuple(score.numel() for score in evaluated["scores"]), batch.counts)
                        for value in [*evaluated["scores"], *evaluated["embeddings"].values()]:
                            self.assertTrue(torch.isfinite(value).all())
                        delta = causality._condition_delta(baseline, evaluated)
                        self.assertEqual(delta["score_count"], sum(batch.counts))
                        self.assertEqual(delta["sample_count"], batch.sample_count)
                        for value in (delta["score_abs_sum"], delta["score_max_error"],
                                      delta["positive_delta_sum"], delta["margin_drop_sum"],
                                      *delta["embedding_cosine_sum"].values()):
                            self.assertTrue(math.isfinite(value))
                self.assert_batch_unchanged(original, batch)

    def test_actual_preflight_runs_baseline_and_all_conditions_without_mutating_inputs(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(771)
            model = debug_model().eval()
            self.assertEqual([len(getattr(model, f"{level}_encoder").blocks)
                              for level in ("local", "patient", "prototype")], [3, 2, 2])
            before_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
            for layout in ("mixed", "all_empty"):
                with self.subTest(layout=layout):
                    batch = debug_causality_batch(layout=layout)
                    original, calls = copy.deepcopy(batch), []
                    rng = torch.get_rng_state().clone()
                    def record_real_forward(_module, arguments, output):
                        self.assertEqual(arguments[0].sample_count, 2)
                        self.assertTrue(all(torch.isfinite(score).all() for score in output.scores))
                        calls.append(tuple(score.numel() for score in output.scores))
                    handle = model.register_forward_hook(record_real_forward)
                    try:
                        result = causality._run_preflight_workload(model, batch, device=torch.device("cpu"), seed=42)
                    finally:
                        handle.remove()
                    self.assertIsNone(result)
                    self.assertEqual(calls, [batch.counts] * (1 + len(causality.CONDITIONS)))
                    self.assert_batch_unchanged(original, batch)
                    self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, before_parameters[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
