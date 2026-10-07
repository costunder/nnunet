"""CPU UNIT admission/order checks; no production training or GPU claims."""
from contextlib import closing
from dataclasses import dataclass
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from hiercp_v1x.comparison_runtime import _prefetch, closing as close_staged
from hiercp_v1x.comparison_progress import PhaseProgress
from hiercp_v1x.comparison_views import parallel_view_provider
from hiercp_v1x.u_bridge_training import capture_rng, digest


@dataclass
class Batch:
    indices: tuple
    values: torch.Tensor

    def pin_memory(self):
        return self


class Budget:
    rss_bytes = 100_000

    def __init__(self):
        self.checks = 0

    def check(self):
        self.checks += 1


class Base:
    global_rng_free = True
    workers = 6
    resident_limit = 80_000

    def __init__(self, cold=()):
        self.budget = Budget()
        self.cold = set(cold)
        self.calls = []
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.started = {i: threading.Event() for i in range(10)}
        self.release = {i: threading.Event() for i in range(10)}
        self.blocked = set()
        self.failures = set()
        self.rss = 10_000

    def cached_batch_ready(self, indices, arm, epoch, *, training, full=False):
        return indices[0] not in self.cold

    def batch(self, indices, arm, epoch, training, full=False):
        index = indices[0]
        with self.lock:
            self.calls.append((tuple(indices), arm, epoch, training, full))
            self.active += 1
            self.peak = max(self.peak, self.active)
        self.started[index].set()
        try:
            if index in self.blocked and not self.release[index].wait(10):
                raise TimeoutError('UNIT did not release batch')
            if index in self.failures:
                raise ValueError(f'original failure {index}')
            self.rss = 12_000
            return Batch(tuple(indices), torch.zeros(256))
        finally:
            with self.lock:
                self.active -= 1


Provider = parallel_view_provider(Base)


class OrderedPrefetchTests(unittest.TestCase):
    def resources(self, provider, available=100_000):
        return patch('hiercp_v1x.comparison_runtime._prefetch_resources',
                     side_effect=lambda: (provider.rss, available))

    def test_measured_warm_jobs_overlap_with_exact_order_flags_and_rng(self):
        provider = Provider()
        provider.blocked = {1, 2, 3, 4}
        batches = [[0, 8], [1, 7], [2, 6], [3, 5], [4]]
        receipts = []
        with self.resources(provider), patch('torch.cuda.is_available', return_value=False), \
                patch('hiercp_v1x.comparison_runtime._append', side_effect=lambda path, row: receipts.append(row)):
            before = digest(capture_rng())
            with closing(_prefetch(provider, batches, 'native_fixed', 11,
                                   training=False, full=True, pin_memory=False,
                                   with_timing=True, receipt_path='UNIT-unused.jsonl')) as staged:
                first, timing = next(staged)
                self.assertEqual(first.indices, (0, 8))
                self.assertEqual(receipts[0]['slots'], 4)
                self.assertEqual(set(timing), {'loader_seconds', 'loader_wait_seconds'})
                try:
                    self.assertTrue(all(provider.started[i].wait(3) for i in range(1, 5)))
                    self.assertEqual(provider.peak, 4)
                    self.assertEqual(provider.workers, 6)
                finally:
                    for i in (4, 3, 2, 1):
                        provider.release[i].set()
                result = [first] + [batch for batch, _ in staged]
            self.assertEqual(digest(capture_rng()), before)
        self.assertEqual([x.indices for x in result], list(map(tuple, batches)))
        self.assertTrue(all(x[1:] == ('native_fixed', 11, False, True) for x in provider.calls))
        self.assertIsNone(provider._comparison_pool)
        self.assertEqual(receipts[-1]['peak_queued_batches'], 4)
        self.assertGreaterEqual(provider.budget.checks, len(batches) * 2)

    def test_cold_batch_runs_alone_between_warm_segments(self):
        provider = Provider(cold={3})
        provider.blocked = {1, 2, 3}
        with self.resources(provider), closing(_prefetch(provider, [[i] for i in range(5)],
                                                       'native', 2, pin_memory=False)) as staged:
            self.assertEqual(next(staged).indices, (0,))
            try:
                self.assertTrue(provider.started[1].wait(3))
                self.assertTrue(provider.started[2].wait(3))
                self.assertFalse(provider.started[3].is_set())
                provider.release[1].set()
                self.assertEqual(next(staged).indices, (1,))
                self.assertFalse(provider.started[3].is_set())
                provider.release[2].set()
                self.assertEqual(next(staged).indices, (2,))
                self.assertTrue(provider.started[3].wait(3))
                self.assertIsNone(provider._comparison_pool)
                self.assertFalse(provider.started[4].is_set())
                provider.release[3].set()
                self.assertEqual([batch.indices for batch in staged], [(3,), (4,)])
            finally:
                for event in provider.release.values():
                    event.set()

    def test_os_memory_and_declared_workspace_each_limit_admission(self):
        for low_os in (False, True):
            provider = Provider()
            if not low_os:
                provider.budget.rss_bytes = 20_000
                provider.resident_limit = 12_000
            provider.blocked = {1}
            receipts = []
            with self.subTest(low_os=low_os), self.resources(provider, 1_000 if low_os else 100_000), \
                    patch('hiercp_v1x.comparison_runtime._append', side_effect=lambda path, row: receipts.append(row)):
                with closing(_prefetch(provider, [[0], [1], [2]], 'selected', 1,
                                       pin_memory=False, with_timing=True, receipt_path='UNIT-unused.jsonl')) as staged:
                    _, timing = next(staged)
                    try:
                        self.assertEqual(receipts[0]['slots'], 1)
                        self.assertTrue(provider.started[1].wait(3))
                        self.assertFalse(provider.started[2].is_set())
                    finally:
                        provider.release[1].set()
                    self.assertEqual(len(list(staged)), 2)

    def test_observed_peak_rss_growth_controls_slot_estimate(self):
        provider = Provider()
        readings = iter([10_000, 50_000, 12_000])
        def resources():
            return next(readings, 12_000), 100_000
        receipts = []
        with patch('hiercp_v1x.comparison_runtime._prefetch_resources', side_effect=resources), \
                patch('hiercp_v1x.comparison_runtime._append', side_effect=lambda path, row: receipts.append(row)):
            values = list(_prefetch(provider, [[0], [1], [2]], 'selected', 1,
                                    pin_memory=False, with_timing=True, receipt_path='UNIT-unused.jsonl'))
        self.assertEqual(receipts[0]['slots'], 1)

    def test_failure_cursor_and_pause_drain_keep_primary_error(self):
        provider = Provider()
        provider.failures = {2}
        consumed = []
        with self.resources(provider), self.assertRaisesRegex(ValueError, 'original failure 2'):
            for batch in _prefetch(provider, [[0], [1], [2], [3]], 'native', 1, pin_memory=False):
                consumed.extend(batch.indices)
        self.assertEqual(consumed, [0, 1])
        self.assertEqual(provider.active, 0)
        self.assertIsNone(provider._comparison_pool)

        provider = Provider()
        provider.failures = {1}
        provider.blocked = {1, 2}
        with self.resources(provider), self.assertRaisesRegex(RuntimeError, 'consumer') as caught:
            with close_staged(_prefetch(provider, [[0], [1], [2]], 'native', 1,
                                        pin_memory=False)) as staged:
                next(staged)
                self.assertTrue(provider.started[1].wait(3))
                self.assertTrue(provider.started[2].wait(3))
                provider.release[1].set()
                provider.release[2].set()
                raise RuntimeError('consumer')
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertEqual(provider.active, 0)
        self.assertIsNone(provider._comparison_pool)

    def test_one_shared_candidate_pool_retains_saved_capacity_for_every_slot_count(self):
        provider = Provider()
        from concurrent.futures import ThreadPoolExecutor
        for slots in range(1, provider.workers - 1):
            barrier = threading.Barrier(provider.workers, timeout=5)
            names = set()
            lock = threading.Lock()
            def item(value):
                with lock:
                    names.add(threading.current_thread().name)
                barrier.wait()
                return value
            with self.subTest(slots=slots), patch('hiercp_v1x.comparison_views.ThreadPoolExecutor',
                                                  wraps=ThreadPoolExecutor) as factory:
                with provider.cpu_staging(slots):
                    inputs = [list(range(i * 12, (i + 1) * 12)) for i in range(slots)]
                    with ThreadPoolExecutor(max_workers=slots) as coordinators:
                        jobs = [coordinators.submit(provider._comparison_map, item, values) for values in inputs]
                        self.assertEqual([job.result() for job in jobs], inputs)
                    with self.assertRaisesRegex(RuntimeError, 'already active'):
                        with provider.cpu_staging(1):
                            pass
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(len(names), provider.workers)
                self.assertTrue(all(name.startswith('comparison-candidates') for name in names))
                self.assertIsNone(provider._comparison_pool)
        self.assertEqual(provider.workers, 6)

    def test_pause_observes_every_running_failure_on_python310(self):
        provider = Provider()
        provider.failures = {1, 2}
        provider.blocked = {1, 2}
        with self.resources(provider), self.assertRaisesRegex(RuntimeError, 'cleanup failures') as caught:
            with closing(_prefetch(provider, [[0], [1], [2]], 'native', 1,
                                   pin_memory=False)) as staged:
                next(staged)
                self.assertTrue(provider.started[1].wait(3))
                self.assertTrue(provider.started[2].wait(3))
                provider.release[1].set()
                provider.release[2].set()
        self.assertEqual([str(x) for x in caught.exception.failures],
                         ['original failure 1', 'original failure 2'])
        self.assertEqual(provider.active, 0)
        self.assertIsNone(provider._comparison_pool)

    def test_hard_budget_failure_is_never_a_smaller_batch_fallback(self):
        provider = Provider()
        original_check = provider.budget.check
        def check():
            original_check()
            if provider.calls:
                raise MemoryError('UNIT exact hard RSS limit')
        provider.budget.check = check
        with self.resources(provider), self.assertRaisesRegex(MemoryError, 'exact hard RSS'):
            list(_prefetch(provider, [[0, 8], [1, 7]], 'native', 1, pin_memory=False))
        self.assertEqual([x[0] for x in provider.calls], [(0, 8)])
        self.assertEqual(provider.active, 0)

    def test_readiness_error_drains_staged_jobs_and_remains_primary(self):
        provider = Provider()
        original_ready = provider.cached_batch_ready
        def ready(indices, *args, **kwargs):
            if indices == [2]:
                raise ValueError('corrupt signed hierarchy metadata')
            return original_ready(indices, *args, **kwargs)
        provider.cached_batch_ready = ready
        with self.resources(provider), self.assertRaisesRegex(ValueError, 'corrupt signed'):
            list(_prefetch(provider, [[0], [1], [2]], 'native', 1, pin_memory=False))
        self.assertEqual(provider.active, 0)
        self.assertIsNone(provider._comparison_pool)

    def test_actual_original_views_match_inside_shared_pool(self):
        from tests.test_comparison_views import canonical_sample, assert_same
        from hiercp import sample, schema
        from hiercp_v1x.comparison_views import materialize_parallel_views
        provider = Provider()
        for training in (True, False):
            expected = sample.materialize_sample_views(canonical_sample(8),
                training=training, epoch=7, global_seed=42)
            with provider.cpu_staging(4):
                actual = materialize_parallel_views(sample, schema, canonical_sample(8),
                    training=training, epoch=7, global_seed=42, workers=provider.workers,
                    ordered_map=provider._comparison_map)
            assert_same(self, actual, expected)

    def test_staged_train_and_validation_timing_flows_through_actual_progress(self):
        from tests.artifacts import unit_artifact_root
        for training in (True, False):
            provider = Provider()
            with self.subTest(training=training), self.resources(provider), \
                    tempfile.TemporaryDirectory(prefix='UNIT_prefetch_progress_', dir=unit_artifact_root()) as directory:
                root = Path(directory)
                totals = dict(loader_seconds=0., loader_wait_seconds=0.)
                with PhaseProgress(root=root, arm='native_fixed', phase='train7' if training else 'validation129',
                                   epoch=1, epochs=40, total=3, initial=0, physical_batch=1,
                                   stream=io.StringIO()) as progress:
                    with closing(_prefetch(provider, [[0], [1], [2]], 'native_fixed', 1,
                                           training=training, full=not training, pin_memory=False,
                                           with_timing=True, receipt_path=root / 'prefetch.jsonl')) as staged:
                        for position, (batch, timing) in enumerate(staged, 1):
                            progress.update(stage='forward', timings=timing)
                            progress.update(completed=position)
                            for key in totals:
                                totals[key] += timing[key]
                    progress.finish()
                receipt = json.loads((root / 'progress.json').read_text())
                self.assertEqual(receipt['status'], 'COMPLETE')
                for key, value in totals.items():
                    self.assertEqual(receipt['timings'][key], value)
                self.assertNotIn('prefetch_slots', receipt['timings'])
                scheduling = [json.loads(line) for line in (root / 'prefetch.jsonl').read_text().splitlines()]
                self.assertEqual(scheduling[0]['slots'], 4)
                self.assertEqual(scheduling[-1]['yielded'], 3)
                self.assertEqual(scheduling[-1]['warm_batches'], 2)


if __name__ == '__main__':
    unittest.main()
