"""CPU UNIT graph scheduling equivalence; analytic topology, no medical claims."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import random
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from hiercp import sample as original_sample, schema
from hiercp_v1x.comparison_views import materialize_parallel_views, parallel_view_provider


def canonical_sample(candidates=8):
    source_types = {'tumor_surface', 'tumor_interior', 'source_context', 'source_liver_surface'}
    nodes = {}
    for number, name in enumerate(schema.LOCAL_NODE_TYPES):
        count = 32 if 'context' in name else 4
        index = torch.arange(count, dtype=torch.float32)
        features = torch.stack([((index + column) % (column + 3)) / (column + 3)
                                for column in range(16)], dim=1)
        position = torch.stack((index / 17, index.remainder(5) / 7, index.remainder(11) / 13), dim=1)
        if 'context' in name:
            position += 2.
        nodes[name] = dict(x=features + number / 100., pos=position,
                          pos_mm=position * 10., grid=position / 5.)
    edges = {}
    for relation in schema.LOCAL_EDGE_TYPES:
        left = torch.arange(len(nodes[relation[0]]['x']), dtype=torch.int32)
        right = torch.arange(len(nodes[relation[2]]['x']), dtype=torch.int32)
        edges[relation] = torch.stack((left.repeat_interleave(len(right)), right.repeat(len(left))))
    source = dict(format='canonical-full-v22', geometry_contract=original_sample.LEVEL0_GEOMETRY_CONTRACT,
                  nodes={name: node for name, node in nodes.items() if name in source_types},
                  edges={relation: edge for relation, edge in edges.items()
                         if relation[0] in source_types and relation[2] in source_types})
    target = dict(format='canonical-full-v22', geometry_contract=original_sample.LEVEL0_GEOMETRY_CONTRACT,
                  nodes={name: node for name, node in nodes.items() if name not in source_types},
                  edges={relation: edge for relation, edge in edges.items() if relation not in source['edges']},
                  transform=torch.eye(3))
    targets = []
    for candidate in range(candidates):
        local = copy.deepcopy(target)
        local['nodes']['target_context']['x'][:, 0] += candidate / 100.
        targets.append(local)
    config = schema.GraphBuildConfig(sample_context_nodes=8, sample_interface_radius_mm=.001,
                                     sample_hops=0).to_dict()
    return dict(case_id='UNIT_ANALYTIC_TOPOLOGY', sample_index=17, graph_config=config,
                source_local=source, target_locals=targets, UNIT_metadata=['preserve', 17])


def assert_same(test, first, second):
    if torch.is_tensor(first):
        test.assertTrue(torch.is_tensor(second))
        test.assertEqual(first.dtype, second.dtype)
        test.assertEqual(first.shape, second.shape)
        test.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        test.assertEqual(set(first), set(second))
        for key in first:
            assert_same(test, first[key], second[key])
    elif isinstance(first, (list, tuple)):
        test.assertEqual(type(first), type(second))
        test.assertEqual(len(first), len(second))
        for one, two in zip(first, second):
            assert_same(test, one, two)
    elif hasattr(first, 'to_dict'):
        test.assertEqual(type(first), type(second))
        assert_same(test, first.to_dict(), second.to_dict())
    else:
        test.assertEqual(first, second)


class ParallelViewTests(unittest.TestCase):
    def test_all_graph_tensors_metadata_and_order_match_original_for_training_and_validation(self):
        for training, epoch in ((True, 3), (True, 29), (False, 11)):
            with self.subTest(training=training, epoch=epoch):
                expected_input = canonical_sample()
                actual_input = copy.deepcopy(expected_input)
                canonical_before = copy.deepcopy(actual_input)
                original_source = actual_input['source_local']
                original_targets = actual_input['target_locals']
                expected = original_sample.materialize_sample_views(expected_input,
                    training=training, epoch=epoch, global_seed=42)
                actual = materialize_parallel_views(original_sample, schema, actual_input,
                    training=training, epoch=epoch, global_seed=42, workers=4)
                self.assertIs(actual, actual_input)
                assert_same(self, actual, expected)
                assert_same(self, original_source, canonical_before['source_local'])
                assert_same(self, original_targets, canonical_before['target_locals'])
                self.assertEqual(len(actual['local_graphs']), 8)
                self.assertEqual(len(actual['local_graphs_view2']), 8)
                self.assertFalse(torch.equal(actual['local_graphs'][0].view_seed,
                                             actual['local_graphs_view2'][0].view_seed))

    def test_global_rng_and_original_module_functions_are_unchanged(self):
        state_python, state_numpy, state_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
        functions = (original_sample.materialize_sample_views, original_sample.build_local_view,
                     original_sample.stable_view_seed)
        materialize_parallel_views(original_sample, schema, canonical_sample(),
            training=True, epoch=17, global_seed=71, workers=4)
        self.assertEqual(random.getstate(), state_python)
        now_numpy = np.random.get_state()
        self.assertEqual(now_numpy[0], state_numpy[0])
        np.testing.assert_array_equal(now_numpy[1], state_numpy[1])
        self.assertEqual(now_numpy[2:], state_numpy[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), state_torch))
        self.assertEqual(functions, (original_sample.materialize_sample_views,
                                    original_sample.build_local_view, original_sample.stable_view_seed))

    def test_fixed_validation_epoch_and_seeded_training_epoch_are_preserved(self):
        zero = materialize_parallel_views(original_sample, schema, canonical_sample(3),
            training=False, epoch=0, global_seed=42, workers=2)
        later = materialize_parallel_views(original_sample, schema, canonical_sample(3),
            training=False, epoch=40, global_seed=42, workers=3)
        assert_same(self, zero, later)
        training = materialize_parallel_views(original_sample, schema, canonical_sample(3),
            training=True, epoch=40, global_seed=42, workers=2)
        self.assertFalse(torch.equal(zero['local_graphs'][0].view_seed,
                                     training['local_graphs'][0].view_seed))

    def test_already_materialized_sample_is_returned_without_rebuilding(self):
        value = {'local_graphs': ['UNIT existing first'], 'local_graphs_view2': ['UNIT existing second']}
        self.assertIs(materialize_parallel_views(None, None, value,
            training=True, epoch=1, global_seed=42, workers=2), value)

    def test_candidate_work_is_actually_parallel_but_results_keep_original_order(self):
        barrier = threading.Barrier(2, timeout=10)
        seen_threads = set()
        lock = threading.Lock()
        def seed(global_seed, case_id, sample_index, candidate_index, epoch, view_index):
            return candidate_index * 2 + view_index
        def build(source, target, config, *, seed):
            if seed % 2 == 0:
                with lock:
                    seen_threads.add(threading.get_ident())
                barrier.wait()
            return ('UNIT ordered worker output', seed)
        module = SimpleNamespace(stable_view_seed=seed, build_local_view=build)
        value = dict(source_local={}, target_locals=[{} for _ in range(8)], graph_config={})
        result = materialize_parallel_views(module, SimpleNamespace(graph_config_from_dict=dict), value,
            training=True, epoch=1, global_seed=42, workers=2)
        self.assertEqual(len(seen_threads), 2)
        self.assertEqual([item[1] for item in result['local_graphs']], list(range(0, 16, 2)))
        self.assertEqual([item[1] for item in result['local_graphs_view2']], list(range(1, 16, 2)))

    def test_build_failure_is_not_replaced_or_partially_published(self):
        value = canonical_sample(3)
        value['target_locals'][1]['format'] = 'UNIT_invalid_canonical_format'
        before = copy.deepcopy(value)
        with self.assertRaisesRegex(ValueError, 'canonical-full-v22'):
            materialize_parallel_views(original_sample, schema, value,
                training=True, epoch=1, global_seed=42, workers=3)
        assert_same(self, value, before)
        self.assertNotIn('local_graphs', value)

    def test_invalid_worker_count_never_falls_back_to_serial_execution(self):
        for workers in (0, 1, -1, True, 2.5):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, 'workers'):
                materialize_parallel_views(original_sample, schema, canonical_sample(2),
                    training=True, epoch=1, global_seed=42, workers=workers)

    def test_provider_proxy_is_local_lazy_threadsafe_and_exposes_timing_hook(self):
        runtime = SimpleNamespace(sample=original_sample, schema=schema, data=object(), cache=object())
        class OriginalProvider:
            def __init__(self, workers):
                self.workers = workers
                self.modules = None
                self.initializations = 0
            def _runtime(self):
                if self.modules is None:
                    self.initializations += 1
                    time.sleep(.01)
                    self.modules = runtime
                return self.modules
        class Timed(parallel_view_provider(OriginalProvider)):
            def _materialize_views(self, *args, **kwargs):
                self.timing_hook_called = True
                return super()._materialize_views(*args, **kwargs)
        original_function = original_sample.materialize_sample_views
        provider = Timed(4)
        second = Timed(2)
        with ThreadPoolExecutor(max_workers=8) as pool:
            proxies = list(pool.map(lambda _: provider._runtime(), range(32)))
        self.assertEqual(provider.initializations, 1)
        self.assertTrue(all(proxy is proxies[0] for proxy in proxies))
        self.assertIsNot(proxies[0], runtime)
        self.assertIsNot(proxies[0], second._runtime())
        self.assertIsNot(proxies[0].sample, runtime.sample)
        self.assertIs(proxies[0].data, runtime.data)
        self.assertIs(proxies[0].cache, runtime.cache)
        self.assertIs(proxies[0].sample.build_local_view, original_sample.build_local_view)
        value = canonical_sample(3)
        actual = proxies[0].sample.materialize_sample_views(value, training=True, epoch=3, global_seed=42)
        expected = original_function(canonical_sample(3), training=True, epoch=3, global_seed=42)
        assert_same(self, actual, expected)
        self.assertTrue(provider.timing_hook_called)
        self.assertIs(runtime.sample, original_sample)
        self.assertIs(original_sample.materialize_sample_views, original_function)


if __name__ == '__main__':
    unittest.main()
