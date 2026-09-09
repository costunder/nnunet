"""DEBUG copy/storage regression; synthetic graphs, not medical/GPU evidence."""
from __future__ import annotations

import contextlib
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref

import torch
from torch_geometric.data import Batch

from hiercp.schema import LOCAL_EDGE_TYPES, LOCAL_NODE_TYPES
from tools import causality
import test_causality_transforms_debug as fixtures
from test_hierarchy_model_debug import debug_model


LOCAL_HELPERS = (
    "_permute_graph_nodes", "_rotate_context_features",
    "_zero_edge_attributes", "_shuffle_topology",
)


def legacy_transform(transform, batch, seed):
    """Freeze the former deep-copy ownership around unchanged transform math."""
    originals = {name: getattr(causality, name) for name in LOCAL_HELPERS}
    with contextlib.ExitStack() as stack:
        for name, function in originals.items():
            def legacy(graph, *args, _function=function, **kwargs):
                return _function(copy.deepcopy(graph), *args, **kwargs)
            stack.enter_context(mock.patch.object(causality, name, new=legacy))
        return transform(copy.deepcopy(batch), seed)


def tensor_storages(graphs):
    result = {}
    for graph in graphs:
        for store in graph.stores:
            for value in store.values():
                if torch.is_tensor(value):
                    storage = value.untyped_storage()
                    result[storage.data_ptr()] = storage.nbytes()
    return result


def debug_overlap_batch(layout):
    batch = fixtures.debug_causality_batch(layout=layout)
    views = []
    for view_number, view in enumerate((batch.local_batch, batch.local_batch_view2)):
        graphs = view.to_data_list()
        for graph_number, graph in enumerate(graphs):
            counts = []
            for node_number, node_type in enumerate(LOCAL_NODE_TYPES):
                count = graph[node_type].num_nodes
                counts.append(count)
                graph[node_type].full_id = torch.arange(count) + node_number * 1000
            graph.canonical_counts = torch.tensor(counts) + 2
            graph.sampled_counts = torch.tensor(counts)
            if view_number == 1 and graph_number == 0:
                graph["target_context"].full_id[-1] += 100_000
        views.append(graphs)
    batch.local_batch, batch.local_batch_view2 = [Batch.from_data_list(view) for view in views]
    return batch


class CausalityTransformMemoryDebugTests(unittest.TestCase):
    assert_nested_equal = fixtures.CausalityTransformsDebugTests.assert_nested_equal
    assert_batch_unchanged = fixtures.CausalityTransformsDebugTests.assert_batch_unchanged

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)  # Explicit DEBUG CPU profile only.

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_fork_independent_stores_share_only_readonly_tensor_storage(self):
        batch = fixtures.debug_causality_batch(layout="mixed")
        original = copy.deepcopy(batch)
        fork = causality._fork_batch_for_audit(batch)
        self.assertIsNot(fork, batch)
        for name in ("source_patches", "target_patches", "difficulties"):
            self.assertIs(getattr(fork, name), getattr(batch, name))
        for name in ("local_batch", "local_batch_view2", "patient_batch", "prototype_batch"):
            left, right = getattr(batch, name), getattr(fork, name)
            self.assertIsNot(left, right)
            for first, second in zip(left.stores, right.stores):
                self.assertIsNot(first, second)
                for key, value in first.items():
                    if torch.is_tensor(value):
                        self.assertIs(value, second[key])
        # .to mutates PyG store mappings even though tensor conversion is out of
        # place. Meta exercises that ownership boundary without GPU allocation.
        fork.to("meta")
        self.assertEqual(fork.source_patches.device.type, "meta")
        self.assertEqual(fork.local_batch["source_context"].x.device.type, "meta")
        self.assertEqual(batch.source_patches.device.type, "cpu")
        self.assert_batch_unchanged(original, batch)

    def test_single_view_fork_preserves_none_and_original_graph(self):
        batch = fixtures.debug_causality_batch(layout="all_empty")
        batch.local_batch_view2 = None
        fork = causality._fork_batch_for_audit(batch)
        self.assertIsNone(fork.local_batch_view2)
        self.assertIsNot(fork.local_batch, batch.local_batch)
        self.assertIs(fork.local_batch["source_context"].x, batch.local_batch["source_context"].x)

    def test_every_condition_matches_legacy_tensors_and_actual_forward(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(873)
            model = debug_model().eval()
            for layout in ("mixed", "all_empty"):
                batch = fixtures.debug_causality_batch(layout=layout)
                original = copy.deepcopy(batch)
                for index, (name, transform) in enumerate(causality.CONDITIONS.items()):
                    with self.subTest(layout=layout, condition=name):
                        seed = -19 + index * 10_007
                        expected = legacy_transform(transform, batch, seed)
                        actual = transform(causality._fork_batch_for_audit(batch), seed)
                        self.assert_batch_unchanged(expected, actual)
                        left = causality._evaluate(model, expected, device=torch.device("cpu"), amp=False)
                        right = causality._evaluate(model, actual, device=torch.device("cpu"), amp=False)
                        self.assert_nested_equal(left, right)
                        self.assert_batch_unchanged(original, batch)

    def test_local_helper_new_storage_is_logical_sized_not_batched_backing_size(self):
        batch = fixtures.debug_causality_batch(layout="mixed")
        graphs = batch.local_batch.to_data_list()
        before = copy.deepcopy(batch)
        original_storage = tensor_storages(graphs)
        for name in LOCAL_HELPERS:
            with self.subTest(helper=name):
                function = getattr(causality, name)
                arguments = ((causality.SOURCE_CONTEXT_TYPES, 52)
                             if name == "_rotate_context_features" else
                             () if name == "_zero_edge_attributes" else (52,))
                changed = [function(graph, *arguments) for graph in graphs]
                for source, target in zip(graphs, changed):
                    for relation in source.edge_types:
                        if name != "_zero_edge_attributes":
                            self.assertIs(source[relation].edge_attr, target[relation].edge_attr)
                    for store in target.stores:
                        for value in store.values():
                            if torch.is_tensor(value) and value.untyped_storage().data_ptr() not in original_storage:
                                self.assertLessEqual(value.untyped_storage().nbytes(),
                                                     value.numel() * value.element_size())
                self.assert_batch_unchanged(before, batch)

    def test_legacy_view_deepcopy_amplification_is_absent_for_node_order(self):
        batch = fixtures.debug_causality_batch(layout="mixed")
        graphs = batch.local_batch.to_data_list()
        relation = ("source_context", "context_neighbor", "source_context")
        source = batch.local_batch[relation].edge_attr
        storage_bytes = source.untyped_storage().nbytes()
        self.assertGreater(storage_bytes, graphs[0][relation].edge_attr.numel() * source.element_size())
        legacy = [copy.deepcopy(graph) for graph in graphs]
        previous_storage = {graph[relation].edge_attr.untyped_storage().data_ptr():
                            graph[relation].edge_attr.untyped_storage().nbytes() for graph in legacy}
        self.assertEqual(sum(previous_storage.values()), len(graphs) * storage_bytes)
        changed = [causality._permute_graph_nodes(graph, 42 + index)
                   for index, graph in enumerate(graphs)]
        current_storage = {graph[relation].edge_attr.untyped_storage().data_ptr():
                           graph[relation].edge_attr.untyped_storage().nbytes() for graph in changed}
        self.assertEqual(sum(current_storage.values()), storage_bytes)

    def test_real_preflight_releases_each_fork_before_next_condition(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(874)
            batch = fixtures.debug_causality_batch(layout="all_empty")
            original = copy.deepcopy(batch)
            model = debug_model().eval()
            references, events = [], []
            def record(_module, arguments, _output):
                value = arguments[0]
                references.append(weakref.ref(value))
                references.extend(weakref.ref(getattr(value, name)) for name in
                                  ("local_batch", "local_batch_view2", "patient_batch", "prototype_batch")
                                  if getattr(value, name) is not None)
            def progress(**event):
                self.assertTrue(all(reference() is None for reference in references))
                events.append((event["event"], event["condition"]))
            handle = model.register_forward_hook(record)
            try:
                causality._run_preflight_workload(model, batch, device=torch.device("cpu"), seed=42, progress=progress)
            finally:
                handle.remove()
            expected = [("baseline_start", "baseline"), ("baseline_end", "baseline")]
            for name in causality.CONDITIONS:
                expected.extend([("condition_start", name), ("condition_end", name)])
            self.assertEqual(events, expected)
            self.assert_batch_unchanged(original, batch)

    def test_progress_guard_failure_prevents_fork_allocation(self):
        batch = fixtures.debug_causality_batch(layout="mixed")
        def refuse(**event):
            raise RuntimeError("DEBUG resource refusal")
        with mock.patch.object(causality, "_fork_batch_for_audit", side_effect=AssertionError("unexpected allocation")):
            with self.assertRaisesRegex(RuntimeError, "DEBUG resource refusal"):
                causality._run_preflight_workload(None, batch, device=torch.device("cpu"), seed=42, progress=refuse)

    def test_preflight_failed_evaluation_releases_its_fork(self):
        batch = fixtures.debug_causality_batch(layout="mixed")
        references = []
        def fail(_model, value, **kwargs):
            references.append(weakref.ref(value))
            raise RuntimeError("DEBUG evaluation failure")
        with mock.patch.object(causality, "_evaluate", new=fail):
            with self.assertRaisesRegex(RuntimeError, "DEBUG evaluation failure"):
                causality._run_preflight_workload(None, batch, device=torch.device("cpu"), seed=42)
        self.assertTrue(all(reference() is None for reference in references))

    def test_view_overlap_integrates_exact_helper_and_requested_work_directory(self):
        batch = debug_overlap_batch("mixed")
        views = [view.to_data_list() for view in (batch.local_batch, batch.local_batch_view2)]
        def legacy_edges(graph):
            result = set()
            for relation, kind in enumerate(LOCAL_EDGE_TYPES):
                edge = graph[kind].edge_index
                result.update((relation, int(source), int(destination)) for source, destination in zip(
                    graph[kind[0]].full_id[edge[0]].tolist(), graph[kind[2]].full_id[edge[1]].tolist()))
            return result
        expected = []
        for first, second in zip(*views):
            left, right = legacy_edges(first), legacy_edges(second)
            expected.append(len(left & right) / len(left | right) if left | right else 1.0)
        original = copy.deepcopy(batch)
        with tempfile.TemporaryDirectory(prefix="DEBUG_overlap_integration_") as directory:
            work = Path(directory)
            sentinel = work / "DEBUG_existing_file"
            sentinel.write_bytes(b"preserved")
            with mock.patch("hiercp.causality_overlap.TemporaryDirectory", wraps=tempfile.TemporaryDirectory) as calls:
                result = causality._view_overlap(batch, work_dir=work)
            self.assertEqual(calls.call_count, sum(batch.counts))
            self.assertTrue(all(call.kwargs["dir"] == work for call in calls.call_args_list))
            self.assertEqual(list(work.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_bytes(), b"preserved")
        self.assertEqual(result["edge_jaccard"], expected)
        self.assertEqual(result["source_context_jaccard"], [1.0] * sum(batch.counts))
        self.assertEqual(result["source_context_fraction"], [0.75] * sum(batch.counts))
        self.assertEqual(result["target_context_fraction"], [0.75] * sum(batch.counts))
        self.assertLess(result["edge_jaccard"][0], 1.0)
        self.assert_batch_unchanged(original, batch)

    def test_actual_audit_loader_matches_legacy_rows_and_releases_before_next_fetch(self):
        with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory(prefix="DEBUG_full_audit_") as directory:
            torch.manual_seed(875)
            model = debug_model().eval()
            originals, events, fork_references = [], [], []
            def batches():
                for layout in ("mixed", "all_empty"):
                    value = debug_overlap_batch(layout)
                    before = copy.deepcopy(value)
                    originals.append(before)
                    reference = weakref.ref(value)
                    yield value
                    self.assert_batch_unchanged(before, value)
                    del value
                    # In particular, enumerate(loader)'s cached result tuple
                    # must not keep the previous full input during next fetch.
                    self.assertIsNone(reference())
            def record(_module, arguments, _output):
                fork_references.append(weakref.ref(arguments[0]))
            def progress(**event):
                self.assertTrue(all(reference() is None for reference in fork_references))
                events.append(event)
            handle = model.register_forward_hook(record)
            try:
                actual = causality._audit_loader(
                    model, batches(), device=torch.device("cpu"), amp=False, seed=42,
                    max_batches=0, work_dir=Path(directory), progress=progress,
                )
            finally:
                handle.remove()
            clean_rows, condition_rows, delta_rows, overlap_rows = actual
            self.assertEqual(len(clean_rows), 2)
            expected_clean = []
            expected_conditions = {name: [] for name in causality.CONDITIONS}
            expected_deltas = {name: [] for name in causality.CONDITIONS}
            expected_overlap = {name: [] for name in overlap_rows}
            for batch_index, batch in enumerate(originals):
                clean = causality._evaluate(model, copy.deepcopy(batch), device=torch.device("cpu"), amp=False)
                expected_clean.append(causality._score_metrics(clean["scores"]))
                for condition_index, (name, transform) in enumerate(causality.CONDITIONS.items()):
                    changed_batch = legacy_transform(transform, batch, 42 + batch_index * 100_003 + condition_index * 10_007)
                    changed = causality._evaluate(model, changed_batch, device=torch.device("cpu"), amp=False)
                    expected_conditions[name].append(causality._score_metrics(changed["scores"]))
                    expected_deltas[name].append(causality._condition_delta(clean, changed))
                for key, values in causality._view_overlap(batch, work_dir=Path(directory)).items():
                    expected_overlap[key].extend(values)
            self.assert_nested_equal((expected_clean, expected_conditions, expected_deltas, expected_overlap), actual)
            for event in ("audit_batch_start", "audit_batch_end", "baseline_start", "baseline_end"):
                self.assertEqual(sum(row["event"] == event for row in events), 2)
            for event in ("condition_start", "condition_end"):
                self.assertEqual(sum(row["event"] == event for row in events), 2 * len(causality.CONDITIONS))
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
