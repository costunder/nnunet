"""UNIT host residency tests; synthetic cache objects are not CT evidence."""
from collections import OrderedDict
import gc
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

from hiercp_v1x import host_memory as host
from hiercp_v1x import u_bridge_fields as fields


class Provider:
    def __init__(self, budget, payload=None):
        self.budget = budget
        self.resident_limit = budget.resident_bytes
        self._lock = threading.RLock()
        self._cache = OrderedDict()
        self._resident_bytes = 0
        self.stats = dict(evictions=0)
        self.calls = []
        if payload is not None:
            self._cache["entry"] = (payload, 80)
            self._resident_bytes = 80

    def _get(self, key, factory):
        self.calls.append("factory")
        return factory()

    def batch(self, value):
        self.calls.append("batch")
        return value


def make_budget(rss=100, cuda=0, after=100, callback=None):
    state = SimpleNamespace(rss=rss, cuda=cuda, reclaim_count=0)
    def reclaim():
        state.reclaim_count += 1
        state.rss = after
        return dict(UNIT_reclaim_hook=True)
    budget = host.PressureBudget(40, 192, resident_bytes=128,
        rss_reader=lambda: state.rss, cuda_reader=lambda: state.cuda,
        reclaim_hook=reclaim, event_callback=callback)
    return budget, state


class HostMemoryTests(unittest.TestCase):
    def test_below_trigger_does_not_evict_or_run_reclamation(self):
        events = []
        budget, state = make_budget(callback=events.append)
        value = np.arange(12, dtype=np.float32)
        provider = Provider(budget, value)
        budget.register_provider(provider)
        budget.check()
        self.assertIs(provider._cache["entry"][0], value)
        self.assertEqual(provider._resident_bytes, 80)
        self.assertEqual((state.reclaim_count, events), (0, []))
        self.assertEqual(budget.pressure_trigger_bytes, 160)

    def test_equal_trigger_does_not_reclaim(self):
        budget, state = make_budget(rss=160)
        budget.check()
        self.assertEqual(state.reclaim_count, 0)

    def test_pressure_releases_cache_ownership_but_live_array_is_unchanged(self):
        events = []
        budget, state = make_budget(rss=180, after=110, callback=events.append)
        live_batch_array = np.arange(12, dtype=np.float32)
        expected = live_batch_array.copy()
        provider = Provider(budget, live_batch_array)
        budget.register_provider(provider)
        budget.register_provider(provider)  # One cache must not be double-counted.
        budget.check()
        self.assertFalse(provider._cache)
        self.assertEqual(provider._resident_bytes, 0)
        self.assertEqual(provider.stats["evictions"], 1)
        np.testing.assert_array_equal(live_batch_array, expected)
        self.assertEqual(state.reclaim_count, 1)
        event = events[0]
        self.assertEqual((event["rss_before_bytes"], event["rss_after_bytes"]), (180, 110))
        self.assertEqual(len(event["cache_providers"]), 1)
        self.assertTrue(event["target_reached"])
        self.assertFalse(event["model_or_input_changed"])
        json.dumps(event, allow_nan=False)

    def test_rss_above_hard_limit_can_reclaim_but_not_relax_limit(self):
        budget, state = make_budget(rss=210, after=170)
        budget.check()
        self.assertEqual(state.reclaim_count, 1)
        self.assertEqual(budget.rss_bytes, 192)
        self.assertEqual(budget.resident_bytes, 128)

    def test_live_batch_above_hard_limit_still_fails_with_actual_measurement(self):
        events = []
        budget, state = make_budget(rss=210, after=200, callback=events.append)
        provider = Provider(budget, object())
        budget.register_provider(provider)
        with self.assertRaisesRegex(MemoryError, "actual=200, limit=192"):
            budget.check()
        self.assertFalse(provider._cache)
        self.assertFalse(events[0]["measured_rss_within_hard_budget"])
        self.assertEqual(events[-1]["resource"], "RSS")

    def test_cuda_hard_limit_is_checked_before_any_host_reclamation(self):
        events = []
        budget, state = make_budget(rss=210, cuda=41, callback=events.append)
        provider = Provider(budget, object())
        budget.register_provider(provider)
        with self.assertRaisesRegex(MemoryError, "CUDA.*actual=41, limit=40"):
            budget.check()
        self.assertTrue(provider._cache)
        self.assertEqual(state.reclaim_count, 0)
        self.assertEqual(events[0]["resource"], "CUDA")

    def test_weak_registration_does_not_extend_provider_lifetime(self):
        budget, _ = make_budget(rss=180)
        provider = Provider(budget)
        reference = weakref.ref(provider)
        budget.register_provider(provider)
        del provider
        gc.collect()
        self.assertIsNone(reference())
        budget.check()
        self.assertFalse(budget._providers)

    def test_no_registered_provider_is_valid_during_original_initialization(self):
        budget, state = make_budget(rss=180)
        budget.check()
        self.assertEqual(state.reclaim_count, 1)

    def test_multiple_worker_checks_reclaim_once_after_actual_rss_drops(self):
        budget, state = make_budget(rss=180, after=100)
        provider = Provider(budget, object())
        budget.register_provider(provider)
        barrier = threading.Barrier(3)
        errors = []
        def worker():
            barrier.wait()
            try:
                budget.check()
            except (MemoryError, ValueError, TypeError) as error:
                errors.append(str(error))
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertEqual((state.reclaim_count, provider.stats["evictions"]), (1, 1))

    def test_provider_checks_before_factory_and_returns_identical_live_batch(self):
        original_get, original_batch = Provider._get, Provider.batch
        Managed = host.pressure_aware_provider(Provider)
        budget, state = make_budget(rss=180, after=100)
        provider = Managed(budget, object())
        def factory():
            self.assertFalse(provider._cache)
            self.assertEqual(state.rss, 100)
            return object()
        result = provider._get("new", factory)
        self.assertIsNotNone(result)
        batch = np.arange(9).reshape(3, 3)
        self.assertIs(provider.batch(batch), batch)
        self.assertIs(Provider._get, original_get)
        self.assertIs(Provider.batch, original_batch)
        self.assertEqual(provider.calls, ["factory", "batch"])

    def test_provider_post_batch_rejects_actual_live_batch_over_budget(self):
        budget, state = make_budget(rss=100, after=205)
        class GrowingProvider(Provider):
            def batch(self, value):
                state.rss = 205
                return value
        provider = host.pressure_aware_provider(GrowingProvider)(budget)
        with self.assertRaisesRegex(MemoryError, "actual=205, limit=192"):
            provider.batch(object())

    def test_readonly_mapping_hints_never_close_or_change_active_field_data(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(prefix="UNIT_host_memory_", dir=root) as directory:
            path = Path(directory) / "distance.npy"
            expected = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
            np.save(path, expected)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            mapped = np.load(path, mmap_mode="r")
            try:
                active = mapped[1]
                with patch.dict(fields._OPENED, {("UNIT", "binding"): ((mapped, mapped), {})}, clear=True):
                    budget = host.PressureBudget(40, 192, resident_bytes=128,
                        rss_reader=lambda: 180, cuda_reader=lambda: 0)
                    events = []
                    budget._event_callback = events.append
                    budget.check()
                np.testing.assert_array_equal(active, expected[1])
                np.testing.assert_array_equal(mapped, expected)
                self.assertFalse(mapped._mmap.closed)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
                hints = events[0]["reclamation"]["readonly_pages"]
                self.assertEqual(hints["mapped_arrays_attempted"], 1)
                self.assertEqual(hints["mappings_closed"], 0)
            finally:
                mapped._mmap.close()  # This UNIT test owns its disposable mapping.

    def test_writable_mapping_is_rejected_instead_of_advised(self):
        array = SimpleNamespace(_mmap=object(), flags=SimpleNamespace(writeable=True))
        with patch.dict(fields._OPENED, {("UNIT", "binding"): ((array,), {})}, clear=True):
            with self.assertRaisesRegex(ValueError, "immutable"):
                host._readonly_page_hints()

    def test_os_hint_rejection_is_recorded_and_does_not_hide_hard_failure(self):
        array = SimpleNamespace(_mmap=object(), flags=SimpleNamespace(writeable=False))
        receipt = dict(supported=True, applied=False, advised_bytes=0, reason="OS rejected UNIT hint")
        with patch.dict(fields._OPENED, {("UNIT", "binding"): ((array,), {})}, clear=True), \
                patch.object(fields, "_release_readonly_pages", return_value=receipt):
            events = []
            budget = host.PressureBudget(40, 192, resident_bytes=128,
                rss_reader=lambda: 200, cuda_reader=lambda: 0, event_callback=events.append)
            with self.assertRaisesRegex(MemoryError, "actual=200, limit=192"):
                budget.check()
        self.assertEqual(events[0]["reclamation"]["readonly_pages"]["rejected"], [receipt["reason"]])

    def test_allocator_trim_unsupported_platform_is_explicit(self):
        with patch.object(host.sys, "platform", "win32"):
            result = host._trim_heap()
        self.assertFalse(result["supported"])
        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "unsupported_platform")

    def test_frozen_core_files_are_not_modified_by_runtime_policy(self):
        root = Path(__file__).resolve().parents[1]
        paths = [root / "hiercp_v1x" / name for name in
                 ("u_bridge_experiment.py", "u_bridge_data.py", "u_bridge_fields.py", "u_bridge_training.py")]
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        budget, _ = make_budget(rss=180)
        provider = host.pressure_aware_provider(Provider)(budget, object())
        provider.batch(np.arange(3))
        after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.assertEqual(before, after)

    def test_invalid_settings_and_inconsistent_cache_accounting_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "headroom"):
            host.PressureBudget(40, 192, resident_bytes=192)
        budget, _ = make_budget(rss=180)
        provider = Provider(budget, object())
        provider._resident_bytes = 81
        budget.register_provider(provider)
        with self.assertRaisesRegex(ValueError, "accounting differs"):
            budget.check()
        budget, _ = make_budget()
        provider = Provider(budget)
        provider.resident_limit = 127
        with self.assertRaisesRegex(ValueError, "sealed execution limit"):
            budget.register_provider(provider)


if __name__ == "__main__":
    unittest.main()
