"""UNIT verified RAM hits and cold integrity; tiny arrays, no CT/GPU claims."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from hiercp_v1x import preparation_reuse as reuse
from hiercp_v1x import u_bridge_fields as fields
from hiercp_v1x import u_bridge_upper as upper
from hiercp_v1x.host_memory import PressureBudget, pressure_aware_provider
from hiercp_v1x.u_bridge_data import UBridgeData
from tests.test_preparation_reuse import (
    binding, close_mappings, directory, static_upper, upper_arrays,
    upper_binding, whole,
)


class UnitResidentProvider(UBridgeData):
    """Use the actual original _get/LRU and pressure adapter, without CT setup."""
    def __init__(self, root, *, resident_limit=2048):
        self.graph_dir = root / 'canonical_local'
        self.graph_dir.mkdir(parents=True)
        self._lock = threading.RLock()
        self._cache = OrderedDict()
        self._resident_bytes = 0
        self.resident_limit = resident_limit
        self.stats = dict(resident_hits=0, evictions=0)
        self.budget = PressureBudget(1024, resident_limit * 2,
            resident_bytes=resident_limit, rss_reader=lambda: 0,
            cuda_reader=lambda: 0)

    def _check_budget(self):
        self.budget.check()


def forbid_added_io(stack, *names):
    for name in names:
        stack.enter_context(patch.object(reuse, name,
            side_effect=AssertionError('Verified UNIT RAM hit must omit added '+name)))


class PreparationHotpathTests(unittest.TestCase):
    def local_fixture(self, root):
        signed = dict(fixture='UNIT_tiny_embedded_graph')
        key = reuse.publication._digest(signed)
        source = root / 'source'
        folder = source / 'canonical_local'
        folder.mkdir(parents=True)
        torch.save(dict(binding=signed, built=torch.arange(4, dtype=torch.float32)),
                   folder / (key+'.pt'))
        return source, signed, key

    def loader(self, provider, key, signed):
        def load():
            value = torch.load(provider.graph_dir / (key+'.pt'), weights_only=False)
            if value.get('binding') != signed:
                raise ValueError('UNIT original loader binding differs')
            return value['built']
        return load

    def test_local_verified_ram_omits_added_io_preserving_original_budget_and_lru(self):
        with directory() as root:
            source, signed, key = self.local_fixture(root)
            with reuse.preparation_reuse(pressure_aware_provider(UnitResidentProvider), [source]) as Reusing:
                provider = Reusing(root / 'own')
                original_loader = Mock(side_effect=self.loader(provider, key, signed))
                cold = provider._get(('local', key), original_loader)
                accounted = provider._resident_bytes
                with ExitStack() as probes, patch.object(provider.budget, 'check',
                        wraps=provider.budget.check) as check:
                    forbid_added_io(probes, '_ensure_local', '_publication_guard', '_safe')
                    hot = provider._get(('local', key), original_loader)
                self.assertIs(hot, cold)
                original_loader.assert_called_once()
                self.assertEqual(provider.stats['resident_hits'], 1)
                self.assertEqual(provider._resident_bytes, accounted)
                check.assert_called_once()
                self.assertTrue((provider.graph_dir / (key+'.pt')).samefile(
                    source / 'canonical_local' / (key+'.pt')))

    def test_local_original_lru_order_and_eviction_unchanged(self):
        with directory() as root:
            source = root / 'source'; source.mkdir()
            with reuse.preparation_reuse(pressure_aware_provider(UnitResidentProvider), [source]) as Reusing:
                provider = Reusing(root / 'own', resident_limit=32)
                keys = [('local', character*64) for character in 'abc']
                first = provider._get(keys[0], lambda: torch.zeros(4))
                provider._get(keys[1], lambda: torch.ones(4))
                with ExitStack() as probes:
                    forbid_added_io(probes, '_ensure_local', '_publication_guard', '_safe')
                    self.assertIs(provider._get(keys[0], lambda: self.fail('Unexpected factory')), first)
                self.assertEqual(list(provider._cache), [keys[1], keys[0]])
                provider._get(keys[2], lambda: torch.full((4,), 2.))
                self.assertEqual(list(provider._cache), [keys[0], keys[2]])
                self.assertEqual(provider._resident_bytes, 32)
                self.assertEqual(provider.stats['evictions'], 1)

    def test_local_ram_removal_returns_to_original_disk_binding_validation(self):
        with directory() as root:
            source, signed, key = self.local_fixture(root)
            with reuse.preparation_reuse(pressure_aware_provider(UnitResidentProvider), [source]) as Reusing:
                provider = Reusing(root / 'own')
                loader = self.loader(provider, key, signed)
                provider._get(('local', key), loader)
                provider._cache.clear(); provider._resident_bytes = 0
                torch.save(dict(binding=dict(signed, changed=True), built=torch.zeros(4)),
                           provider.graph_dir / (key+'.pt'))
                with patch.object(reuse, '_ensure_local', wraps=reuse._ensure_local) as ensure:
                    with self.assertRaisesRegex(ValueError, 'original loader binding differs'):
                        provider._get(('local', key), loader)
                ensure.assert_called_once()
                self.assertNotIn(('local', key), provider._cache)

    def test_local_cold_corrupted_source_rejected_before_original_factory(self):
        with directory() as root:
            source, signed, key = self.local_fixture(root)
            path = source / 'canonical_local' / (key+'.pt')
            path.write_bytes(b'UNIT invalid complete publication')
            with reuse.preparation_reuse(pressure_aware_provider(UnitResidentProvider), [source]) as Reusing:
                provider = Reusing(root / 'own')
                factory = Mock(side_effect=AssertionError('No corruption fallback'))
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    provider._get(('local', key), factory)
                factory.assert_not_called()
                self.assertFalse((provider.graph_dir / (key+'.pt')).exists())

    def test_local_concurrent_cold_import_keeps_guard_and_one_publication(self):
        with directory() as root:
            source, signed, key = self.local_fixture(root); events = []
            with reuse.preparation_reuse(pressure_aware_provider(UnitResidentProvider), [source], events.append) as Reusing:
                provider = Reusing(root / 'own')
                loader = self.loader(provider, key, signed)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(lambda _: provider._get(('local', key), loader), range(4)))
                self.assertTrue(all(value is results[0] for value in results))
                self.assertEqual([event['status'] for event in events], ['reused'])
                self.assertEqual(provider._resident_bytes, 16)

    def test_upper_both_resident_kinds_keep_budget_counts_and_defensive_copies(self):
        with directory() as root:
            source = root / 'source'
            for kind in ('source_raw', 'lesions'): static_upper(source, kind)
            check = Mock()
            with reuse.preparation_reuse(UnitResidentProvider, [source]):
                cache = upper.UpperCache(root / 'own/upper_static', check)
                for kind in ('source_raw', 'lesions'):
                    factory = Mock(side_effect=AssertionError('No exact helper recompute'))
                    cold = cache._obtain(kind, upper_binding(), factory)
                    before = check.call_count
                    with ExitStack() as probes:
                        forbid_added_io(probes, '_ensure_upper', '_publication_guard', '_safe')
                        hot = cache._obtain(kind, upper_binding(), factory)
                    expected = upper_arrays(kind)
                    for actual, original, previous in zip(
                            (hot,) if kind == 'source_raw' else hot,
                            (expected,) if kind == 'source_raw' else expected,
                            (cold,) if kind == 'source_raw' else cold):
                        np.testing.assert_array_equal(actual, original)
                        self.assertFalse(np.shares_memory(actual, previous))
                        actual.flat[0] = -100
                    factory.assert_not_called()
                    self.assertEqual(check.call_count, before+1)
                self.assertEqual(cache.stats['resident_hits'], 2)
                self.assertEqual(cache.stats['disk_reopens'], 2)
                self.assertEqual(cache.stats['original_helper_builds'], 0)

    def test_upper_removed_resident_rechecks_payload_without_fallback(self):
        with directory() as root:
            source = root / 'source'; static_upper(source)
            with reuse.preparation_reuse(UnitResidentProvider, [source]):
                cache = upper.UpperCache(root / 'own/upper_static', lambda: None)
                cache._obtain('source_raw', upper_binding(), lambda: self.fail('No recompute'))
                cache._resident.clear()
                payload = cache.root / 'UNIT_case/c1_a1_2_3/source_raw/arrays.npz'
                payload.write_bytes(payload.read_bytes()+b'UNIT altered bytes')
                with self.assertRaisesRegex(ValueError, 'payload SHA256 differs'):
                    cache._obtain('source_raw', upper_binding(), lambda: self.fail('No corruption fallback'))

    def test_upper_changed_binding_cannot_use_old_resident_value(self):
        with directory() as root:
            source = root / 'source'; static_upper(source)
            with reuse.preparation_reuse(UnitResidentProvider, [source]):
                cache = upper.UpperCache(root / 'own/upper_static', lambda: None)
                cache._obtain('source_raw', upper_binding(), lambda: self.fail('No recompute'))
                changed = dict(upper_binding(), image_sha256='e'*64)
                with self.assertRaisesRegex(ValueError, 'binding/metadata differs'):
                    cache._obtain('source_raw', changed, lambda: self.fail('No wrong-binding fallback'))
                self.assertEqual(cache.stats['resident_hits'], 0)

    def test_fields_verified_mapping_omits_added_io_and_calls_original_budget(self):
        with directory() as root:
            source = root / 'source'; whole(source)
            check = Mock()
            with reuse.preparation_reuse(UnitResidentProvider, [source]):
                args = (root / 'own/whole_case_fields', 'UNIT_case', binding(),
                        lambda: self.fail('No CPU field recompute'),
                        lambda: self.fail('No CPU field recompute'), check)
                cold = fields.cached_fields(*args)
                before = check.call_count
                with ExitStack() as probes:
                    forbid_added_io(probes, '_ensure_fields', '_publication_guard', '_safe')
                    probes.enter_context(patch.object(fields, '_load',
                        side_effect=AssertionError('Original registry must serve RAM mapping')))
                    hot = fields.cached_fields(*args)
                self.assertIs(hot[0], cold[0]); self.assertIs(hot[1], cold[1])
                self.assertEqual(hot[2]['status'], 'resident_mapping')
                self.assertEqual(check.call_count, before+1)
                self.assertFalse(hot[0].flags.writeable)
                self.assertEqual(hot[2]['fields'], cold[2]['fields'])

    def test_fields_registry_removal_restores_cold_sha_validation(self):
        with directory() as root:
            source = root / 'source'; whole(source)
            with reuse.preparation_reuse(UnitResidentProvider, [source]):
                args = (root / 'own/whole_case_fields', 'UNIT_case', binding(),
                        lambda: self.fail('No CPU recompute'), lambda: self.fail('No CPU recompute'), lambda: None)
                fields.cached_fields(*args)
                close_mappings()
                payload = args[0] / 'UNIT_case/depth.npy'
                payload.write_bytes(payload.read_bytes()+b'UNIT changed bytes')
                with patch.object(fields, '_load', wraps=fields._load) as load:
                    with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
                        fields.cached_fields(*args)
                load.assert_called_once()

    def test_fields_changed_binding_rejected_despite_other_open_mapping(self):
        with directory() as root:
            source = root / 'source'; whole(source)
            with reuse.preparation_reuse(UnitResidentProvider, [source]):
                args = (root / 'own/whole_case_fields', 'UNIT_case', binding(),
                        lambda: self.fail('No CPU recompute'), lambda: self.fail('No CPU recompute'), lambda: None)
                fields.cached_fields(*args)
                changed = (*args[:2], dict(binding(), image_sha256='d'*64), *args[3:])
                with self.assertRaisesRegex(ValueError, 'binding/metadata identity differs'):
                    fields.cached_fields(*changed)

    def test_concurrent_hot_fields_reuse_original_mapping_without_extra_imports(self):
        with directory() as root:
            source = root / 'source'; whole(source)
            with reuse.preparation_reuse(UnitResidentProvider, [source]):
                args = (root / 'own/whole_case_fields', 'UNIT_case', binding(),
                        lambda: self.fail('No recompute'), lambda: self.fail('No recompute'), lambda: None)
                first = fields.cached_fields(*args)
                with ExitStack() as probes:
                    forbid_added_io(probes, '_ensure_fields', '_publication_guard', '_safe')
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        results = list(pool.map(lambda _: fields.cached_fields(*args), range(8)))
                self.assertTrue(all(result[0] is first[0] for result in results))
                self.assertTrue(all(result[2]['status'] == 'resident_mapping' for result in results))


if __name__ == '__main__':
    unittest.main()
