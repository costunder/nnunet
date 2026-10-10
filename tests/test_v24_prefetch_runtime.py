"""CPU DEBUG: deterministic producer overlap and real original input parity.

No native CUDA utilization, model accuracy or complete training claim. Events
control the original CPU collator; no reduced production model is installed.
"""
from collections import OrderedDict
from contextlib import contextmanager
import copy
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hiercp_v1x import v24_memory_runtime as memory
from hiercp_v1x import v24_prefetch_runtime as runtime
from hiercp_v1x.v23_training import prepared_native_chunks
from tests import test_v24_provider as fixture


@contextmanager
def installed():
    owners = ((memory.MemorySafeCoordinator, '__init__'),
              (memory.MemorySafeCoordinator, 'trim'),
              (memory.MemorySafeInputProvider, '__init__'),
              (memory.MemorySafeInputProvider, 'get'))
    originals = [(owner, name, owner.__dict__.get(name)) for owner, name in owners]
    try:
        with patch.object(torch.cuda, 'is_initialized', return_value=False):
            runtime.install_runtime()
        yield
    finally:
        for owner, name, original in originals:
            if original is None:
                delattr(owner, name)
            else:
                setattr(owner, name, original)


class PrefetchPressurePolicyDebug(unittest.TestCase):
    def setUp(self):
        self.installation = installed(); self.installation.__enter__()

    def tearDown(self):
        self.installation.__exit__(None, None, None)

    def coordinator(self, rss=1024, actual=0):
        coordinator = memory.MemorySafeCoordinator(rss)
        coordinator._rss_process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=actual))
        coordinator._prefetch_runtime_state['resident_limits'] = (rss//2,)
        return coordinator

    def test_original_classes_pressure_code_and_one_future_are_unchanged(self):
        contract = runtime.runtime_contract()
        self.assertIs(runtime._COORDINATOR, memory.MemorySafeCoordinator)
        self.assertIs(runtime._PROVIDER, memory.MemorySafeInputProvider)
        self.assertIs(memory.MemorySafeCoordinator._trim_locked,
                      runtime._COORDINATOR._trim_locked)
        self.assertEqual(contract['producer_chunks_ahead'], 1)
        self.assertEqual(contract['additional_input_futures'], 0)
        self.assertFalse(contract['RSS_limit_increased'])
        self.assertFalse(contract['model_loss_views_data_or_batch_changed'])
        self.assertEqual(contract, runtime.install_runtime())

    def test_low_RSS_admission_does_not_wait_for_producer_cache_lock(self):
        coordinator = self.coordinator(actual=767)
        acquired = threading.Event(); release = threading.Event(); failures = []
        def producer():
            with coordinator.lock:
                acquired.set(); release.wait(5)
        worker = threading.Thread(target=producer); worker.start()
        self.assertTrue(acquired.wait(2))
        try:
            # Original inherited trim is confirmed blocked in this situation.
            old_done = threading.Event()
            old = threading.Thread(target=lambda: (runtime._BASE_TRIM(coordinator), old_done.set()))
            old.start(); self.assertFalse(old_done.wait(.05))
            started = time.perf_counter(); coordinator.trim()
            self.assertLess(time.perf_counter()-started, .5)
            snapshot = runtime.receipt(coordinator)
            self.assertFalse(snapshot['telemetry_acquires_coordinator_cache_lock'])
            self.assertEqual(snapshot['pressure_trigger_bytes'], 768)
            self.assertEqual(snapshot['producer_workspace_reserve_bytes'], 256)
            self.assertEqual(snapshot['profile']['low_RSS_fast_admissions'], 1)
        finally:
            release.set(); worker.join(3); old.join(3)
        self.assertFalse(worker.is_alive()); self.assertFalse(old.is_alive())

    def test_pressure_still_waits_for_original_lock_and_records_hidden_wait(self):
        coordinator = self.coordinator(actual=769); reclaimed = []
        coordinator._release_allocators = lambda: reclaimed.append(True)
        ready = threading.Event(); release = threading.Event(); done = threading.Event()
        def producer():
            with coordinator.lock:
                ready.set(); release.wait(5)
        worker = threading.Thread(target=producer); worker.start(); self.assertTrue(ready.wait(2))
        def consumer():
            coordinator.trim(); done.set()
        check = threading.Thread(target=consumer); check.start()
        try:
            self.assertFalse(done.wait(.05))
        finally:
            release.set(); worker.join(3); check.join(3)
        self.assertTrue(done.is_set()); self.assertEqual(reclaimed, [True])
        stats = runtime.receipt(coordinator)['profile']
        self.assertEqual(stats['low_RSS_fast_admissions'], 0)
        self.assertEqual(stats['pressure_lock_admissions'], 1)
        self.assertGreater(stats['pressure_lock_wait_seconds'], .04)
        self.assertEqual(coordinator._memory_stats['pressure_events'], 1)

    def test_RSS_crossing_on_second_sample_uses_original_pressure_path(self):
        coordinator = self.coordinator(); values = iter((767, 769))
        coordinator._rss_process = SimpleNamespace(memory_info=lambda:
            SimpleNamespace(rss=next(values, 769)))
        calls = []; coordinator._release_allocators = lambda: calls.append(True)
        coordinator.trim()
        self.assertEqual(calls, [True])
        self.assertEqual(runtime.receipt(coordinator)['profile']['low_RSS_fast_admissions'], 0)

    def test_hard_RSS_limit_and_protected_live_records_are_not_relaxed(self):
        coordinator = self.coordinator(actual=1025)
        protected = {'actual_record'}; views = {'actual_view'}; attempted = []
        class Ledger:
            def bytes(self, name): return 100
        def evict(category):
            attempted.append((category, coordinator.lock._is_owned(), set(protected), set(views)))
            return False
        coordinator.providers = [SimpleNamespace(resident_bytes=512, _ledger=Ledger(), evict=evict)]
        coordinator._release_allocators = lambda: None
        with self.assertRaisesRegex(MemoryError, 'Full active'):
            coordinator.trim()
        self.assertEqual(protected, {'actual_record'}); self.assertEqual(views, {'actual_view'})
        self.assertEqual(attempted, [('view', True, protected, views), ('record', True, protected, views)])
        self.assertEqual(coordinator.rss_bytes, 1024)
        self.assertEqual(coordinator._memory_stats['strict_failures'], 1)

    def test_low_RSS_runtime_source_guards_and_strict_type_still_fail(self):
        coordinator = self.coordinator()
        with patch.object(memory._ORIGINAL_PROVIDER, '_stat', return_value=(0, 0, 0, 0, 0)):
            with self.assertRaisesRegex(ValueError, 'runtime source changed'):
                coordinator.trim()
        with self.assertRaisesRegex(TypeError, 'Explicit strict'):
            coordinator.trim(strict=1)
        with patch.object(torch.cuda, 'is_initialized', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'hot-swap'):
                runtime.install_runtime()
        with patch.object(memory.MemorySafeCoordinator, 'trim', lambda *args: None):
            with self.assertRaisesRegex(ValueError, 'Foreign'):
                runtime.install_runtime()

    def test_installation_after_coordinator_creation_is_rejected(self):
        coordinator = self.coordinator(); del coordinator._prefetch_runtime_state
        with self.assertRaisesRegex(ValueError, 'before coordinator'):
            coordinator.trim()

    def test_admitted_RSS_budget_and_original_pressure_function_cannot_be_mutated(self):
        coordinator = self.coordinator(); coordinator.rss_bytes = 2048
        with self.assertRaisesRegex(ValueError, 'shared RSS budget changed'):
            coordinator.trim()
        with patch.object(memory.MemorySafeCoordinator, '_trim_locked', lambda *args, **kwargs: None):
            with self.assertRaisesRegex(ValueError, 'pressure reclamation'):
                runtime.install_runtime()


class PrefetchActualInputsDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.V24CPUProviderDebug.setUpClass(); cls.data = fixture.V24CPUProviderDebug

    @classmethod
    def tearDownClass(cls):
        fixture.V24CPUProviderDebug.tearDownClass()

    def setUp(self):
        self.installation = installed(); self.installation.__enter__()

    def tearDown(self):
        self.installation.__exit__(None, None, None)

    def provider(self, partition='inner_val'):
        import psutil
        coordinator = memory.MemorySafeCoordinator(psutil.Process().memory_info().rss+2**31)
        coordinator._rss_process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0))
        dataset = SimpleNamespace(rows=copy.deepcopy(self.data.fixture.rows),
            meta=dict(records=self.data.fixture.rows), partition=partition)
        return memory.MemorySafeInputProvider(dataset, self.data.index, workers=4,
            resident_bytes=2**29, coordinator=coordinator)

    def test_actual_fixed_views_and_epoch_views_full_tensor_RNG_and_private_parity(self):
        optimized = self.provider(); original = self.provider()
        try:
            rng = torch.get_rng_state().clone()
            expected = self.data.signature(runtime._BASE_GET(original, [0,1,2], epoch=29))
            batch = optimized.get([0,1,2], epoch=29)
            self.assertEqual(expected, self.data.signature(batch))
            batch.graph['tumor_surface'].x.add_(100); batch.target_patches.add_(100)
            warm = optimized.get([0,1,2], epoch=29)
            self.assertEqual(expected, self.data.signature(warm))
            sample = runtime.receipt(optimized.coordinator)['provider_last_completed_samples'][0]
            self.assertEqual(sample['scalar_deltas']['sampled_pair_hits'], 3)
            self.assertEqual(sample['scalar_deltas']['sampled_pair_misses'], 0)
            self.assertEqual(sample['scalar_deltas']['record_loads'], 0)
            self.assertEqual(sample['scalar_deltas']['sampled_pairs_materialized'], 0)
            self.assertEqual(sample['complete_ordered_observation_ids'], [0,1,2])
            self.assertGreaterEqual(sample['scalar_deltas']['collate_seconds'], 0)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertFalse(optimized._protected_ids); self.assertFalse(optimized._protected_views)
        finally:
            optimized.close(); original.close()
        train = self.provider('inner_train'); reference = self.provider('inner_train')
        try:
            for epoch in (1, 2, 29, 40):
                self.assertEqual(self.data.signature(train.get([0,1,2], epoch=epoch)),
                    self.data.signature(runtime._BASE_GET(reference, [0,1,2], epoch=epoch)))
            self.assertEqual(len(train._views), 0)
        finally:
            train.close(); reference.close()

    def test_real_next_chunk_collator_overlaps_current_consumer_and_one_future_only(self):
        provider = self.provider(); second_ready = threading.Event(); release = threading.Event()
        collate = memory._GET.__globals__['collate']; counts = []
        def observed_collate(items):
            ids = [index for _, index in items]; counts.append(ids)
            if ids == [1]:
                self.assertTrue(provider.coordinator.lock._is_owned())
                self.assertTrue(provider._protected_ids); self.assertTrue(provider._protected_views)
                second_ready.set(); self.assertTrue(release.wait(5))
            return collate(items)
        try:
            with patch.dict(memory._GET.__globals__, collate=observed_collate):
                with prepared_native_chunks(provider, [[0], [1], [2]], epoch=29,
                        prefetch=True, pin_memory=False) as chunks:
                    first, timing = next(chunks)
                    self.assertTrue(second_ready.wait(2))
                    # This is the exact consumer admission invoked before _encode.
                    provider.coordinator.trim()
                    self.assertEqual(first.indices.tolist(), [0])
                    self.assertEqual(counts, [[0], [1]])  # third was not queued
                    stats = runtime.receipt(provider.coordinator)['profile']
                    self.assertGreater(stats['fast_admissions_during_provider_get'], 0)
                    release.set()
                    remaining = list(chunks)
                    self.assertEqual([batch.indices.tolist() for batch, _ in remaining], [[1], [2]])
            self.assertEqual(counts, [[0], [1], [2]])
            self.assertEqual(runtime.receipt(provider.coordinator)['profile']['provider_get_calls'], 3)
        finally:
            release.set(); provider.close()

    def test_warm_file_proofs_closed_provider_and_original_validation_remain(self):
        provider = self.provider()
        try:
            provider.get([0], epoch=29)
            entry = provider.records[self.data.fixture.rows[0]['id']]
            file = provider.root/entry['path']; stat = file.stat()
            import os
            os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns+1000000))
            try:
                with self.assertRaisesRegex(ValueError, 'replaced or changed'):
                    provider.get([0], epoch=29)
            finally:
                os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, 'unique'):
                provider.get([0,0], epoch=29)
            with self.assertRaisesRegex(ValueError, 'unique'):
                provider.get(None, epoch=29)
            with self.assertRaisesRegex(ValueError, 'fixed29'):
                provider.get([0], epoch=1)
            provider.close()
            with self.assertRaisesRegex(RuntimeError, 'Closed'):
                provider.get([0], epoch=29)
        finally:
            provider.close()

    def test_provider_residency_budget_cannot_change_after_static_registration(self):
        provider = self.provider()
        try:
            provider.resident_bytes += 1
            with self.assertRaisesRegex(ValueError, 'residency budget changed'):
                provider.get([0], epoch=29)
        finally:
            provider.close()

    def test_source_only_prefetch_guard_rejects_without_disabling_original_memory_guard(self):
        provider = self.provider(); state = provider.coordinator._prefetch_runtime_state
        original = memory._ORIGINAL_PROVIDER._stat
        def modified(path):
            value = original(path)
            return (0, 0, 0, 0, 0) if str(path) == str(state['source']) else value
        try:
            with patch.object(memory._ORIGINAL_PROVIDER, '_stat', side_effect=modified):
                with self.assertRaisesRegex(ValueError, 'prefetch runtime source changed'):
                    provider.coordinator.trim()
        finally:
            provider.close()

    def test_returned_live_input_over_budget_is_rejected_after_original_finally(self):
        import weakref
        provider = self.provider(); live = []; collate = memory._GET.__globals__['collate']
        def observed_collate(items):
            result = collate(items); live.append(weakref.ref(result)); return result
        provider.coordinator._rss_process = SimpleNamespace(memory_info=lambda:
            SimpleNamespace(rss=provider.rss_bytes+1 if any(item() is not None for item in live) else 0))
        provider.coordinator._release_allocators = lambda: None
        try:
            with patch.dict(memory._GET.__globals__, collate=observed_collate):
                with self.assertRaisesRegex(MemoryError, 'Full active'):
                    provider.get([0], epoch=29)
            self.assertFalse(provider._protected_ids); self.assertFalse(provider._protected_views)
        finally:
            provider.close()

    def test_forward_binding_preserves_original_budget_gradient_and_does_not_wait_for_telemetry(self):
        from hiercp_v1x.historical_evaluation import ResourceBudget
        provider = self.provider()
        class Inputs:
            def _rss(self): pass
            def guard_source(self): self.guard_calls += 1
        inputs = Inputs(); inputs.memory_coordinator = provider.coordinator
        inputs.rss_bytes = provider.rss_bytes; inputs._input_file_proofs = {}; inputs._file_proofs = {}
        inputs.guard_calls = 0; inputs._lock = threading.RLock()
        inputs._raw_cache = OrderedDict(); inputs._regions = {}; inputs._donors = {}
        geometry = Inputs(); geometry.memory_guard = inputs._rss; geometry.rss_bytes = provider.rss_bytes
        geometry._lock = threading.RLock(); geometry._memo = OrderedDict(); geometry._proofs = {}; geometry._bytes = 0
        class Scorer(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.scale = torch.nn.Parameter(torch.tensor(2.))
                self.providers = dict(inner_val=provider); self.geometry = geometry
                self.budget = ResourceBudget(2**30, provider.rss_bytes)
            def _encode(self, values, *, training): return values*self.scale
            def forward(self, values, *, epoch, training):
                self.budget.check()
                return SimpleNamespace(values=self._encode(values, training=training), workload={})
        scorer = Scorer(); ready = threading.Event(); release = threading.Event()
        failures = []; collate = memory._GET.__globals__['collate']
        def blocked_collate(items):
            ready.set()
            if not release.wait(5): raise TimeoutError('DEBUG consumer did not release collator')
            return collate(items)
        def producer():
            try: provider.get([0], epoch=29)
            except BaseException as error: failures.append(error)
        worker = threading.Thread(target=producer)
        try:
            original_model = scorer.state_dict()['scale'].clone(); rng = torch.get_rng_state().clone()
            memory.bind_memory_runtime(scorer); runtime.bind(scorer); runtime.bind(scorer)
            with patch.dict(memory._GET.__globals__, collate=blocked_collate):
                worker.start(); self.assertTrue(ready.wait(2))
                with patch.object(provider, 'profile', side_effect=AssertionError('blocking telemetry forbidden')), \
                        patch.object(provider.coordinator, 'memory_runtime_receipt', side_effect=AssertionError('blocking receipt forbidden')):
                    result = scorer(torch.tensor([1., 2.]), epoch=29, training=False)
                    result.values.sum().backward()
                release.set(); worker.join(3)
            self.assertFalse(worker.is_alive()); self.assertEqual(failures, [])
            self.assertTrue(torch.equal(result.values, torch.tensor([2., 4.])))
            self.assertEqual(scorer.scale.grad.item(), 3.)
            self.assertTrue(torch.equal(original_model, scorer.scale))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(result.workload['CPU_prefetch_runtime']['forward_profile_delta']['local_encode_calls'], 1)
            self.assertEqual(result.workload['CPU_prefetch_runtime']['forward_profile_delta']['local_encode_calls_during_provider_get'], 1)
            self.assertGreater(result.workload['CPU_prefetch_runtime']['forward_profile_delta']['local_encode_host_overlap_seconds'], 0)
            self.assertEqual(len(scorer._forward_pre_hooks), 2); self.assertEqual(len(scorer._forward_hooks), 2)
            with patch.object(torch.cuda, 'memory_allocated', return_value=2**31):
                with self.assertRaisesRegex(MemoryError, 'Historical evaluation CUDA'):
                    scorer.budget.check()
        finally:
            release.set()
            if worker.ident is not None: worker.join(3)
            provider.close()


if __name__ == '__main__':
    unittest.main()
