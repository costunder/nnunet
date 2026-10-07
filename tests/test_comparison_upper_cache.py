"""CPU UNIT checks of exact original compact helpers; no model-quality claim."""
from collections import OrderedDict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hiercp import hierarchy
from hiercp_v1x.comparison_upper_cache import CompactUpperCache, compact_upper_provider, _digest
from hiercp_v1x.u_bridge_upper import UpperCache
from hiercp_v1x.u_bridge_data import UBridgeData
from tests.artifacts import unit_artifact_root


def fixture():
    shape = (7, 8, 9)
    label = np.zeros(shape, np.int16)
    label[1:6, 1:7, 1:8] = 1
    label[2:4, 3:5, 3:5] = 2
    case = SimpleNamespace(paths=SimpleNamespace(case_id='UNIT_original_helpers'), shape=shape,
        spacing=np.asarray([.7, 1.1, 2.], np.float32), image=np.arange(np.prod(shape),
        dtype=np.float32).reshape(shape) / 3, label=label)
    mask = label == 2
    source = SimpleNamespace(component_id=1, anchor_center=(3, 4, 4),
                             voxel_count=int(mask.sum()), full_mask=mask)
    regions = SimpleNamespace(full_organ_mask=label > 0,
                              organ_depth=np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 10)
    binding = dict(case_id=case.paths.case_id, image_sha256='a'*64, label_sha256='b'*64,
        original_core_sha256='c'*64, region_identity_sha256='d'*64,
        source_component=1, source_voxels=source.voxel_count, shape=list(shape),
        anchor=list(source.anchor_center), spacing=list(map(float, case.spacing)),
        ct_clip=[-200., 250.], tumor_label=2, max_lesions=None, upper_raw_dim=14)
    return case, source, regions, binding


class BaseProvider:
    def __init__(self, root, module, binding):
        import threading
        self.root, self.module, self.binding = Path(root), module, binding
        self._lock = threading.RLock()
        self._cache = OrderedDict()
        self._resident_bytes, self.resident_limit = 0, 2**20
        self.stats = dict(resident_hits=0, evictions=0)
        self._upper_bindings = {}
        self.budget_calls = 0
        self._upper_cache = UpperCache(self.root/'upper_static', self._check_budget)

    def _check_budget(self): self.budget_calls += 1
    def _runtime(self): return SimpleNamespace(hierarchy=self.module)
    _get = UBridgeData._get

    def _upper_context(self, example, case, source, regions):
        self._upper_bindings[example['id']] = self.binding
        return self._upper_cache.original_helpers(self.module, case, source, regions, self.binding)

    def report(self): return dict(original_report=True)


class CompactUpperTests(unittest.TestCase):
    def temporary(self):
        return TemporaryDirectory(prefix='UNIT_compact_upper_', dir=unit_artifact_root())

    def assert_exact(self, actual, expected):
        if isinstance(actual, tuple):
            self.assert_exact(actual[0], expected[0]); self.assertEqual(actual[1], expected[1]); return
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.strides, expected.strides)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_original_cold_hot_evicted_reopen_exact_and_budgeted(self):
        case, source, regions, binding = fixture()
        liver = hierarchy._liver_raw(case, regions, tumor_label=2, ct_clip=(-200., 250.))
        axis = hierarchy._principal_axis(source.full_mask, case.spacing)
        with self.temporary() as directory:
            provider = compact_upper_provider(BaseProvider)(directory, hierarchy, binding)
            with patch.object(hierarchy, '_liver_raw', wraps=hierarchy._liver_raw) as liver_call, \
                 patch.object(hierarchy, '_principal_axis', wraps=hierarchy._principal_axis) as axis_call:
                for attempt in range(3):
                    if attempt == 2:
                        provider._cache.clear(); provider._resident_bytes = 0
                    with provider._upper_context({'id': 'UNIT_source'}, case, source, regions):
                        self.assert_exact(hierarchy._liver_raw(case, regions, tumor_label=2,
                            ct_clip=(-200., 250.)), liver)
                        self.assert_exact(hierarchy._principal_axis(source.full_mask, case.spacing), axis)
                self.assertEqual((liver_call.call_count, axis_call.call_count), (1, 1))
                fresh = compact_upper_provider(BaseProvider)(directory, hierarchy, binding)
                with fresh._upper_context({'id': 'UNIT_source'}, case, source, regions):
                    self.assert_exact(hierarchy._liver_raw(case, regions, tumor_label=2,
                        ct_clip=(-200., 250.)), liver)
                    self.assert_exact(hierarchy._principal_axis(source.full_mask, case.spacing), axis)
                self.assertEqual((liver_call.call_count, axis_call.call_count), (1, 1))
            for kind, row in provider.report()['compact_upper']['kinds'].items():
                self.assertEqual((row['calls'], row['builds'], row['reopens'], row['resident_hits']), (3, 1, 1, 1))
            self.assertGreater(provider.budget_calls, 8)
            self.assertTrue(all(key[0] == 'compact_upper' for key in provider._cache))

    def test_defensive_copies_and_other_mask_never_uses_source_axis(self):
        case, source, regions, binding = fixture()
        with self.temporary() as directory:
            provider = compact_upper_provider(BaseProvider)(directory, hierarchy, binding)
            original = hierarchy._principal_axis
            with patch.object(hierarchy, '_principal_axis', wraps=original) as called:
                with provider._upper_context({'id': 'UNIT_source'}, case, source, regions):
                    first = hierarchy._principal_axis(source.full_mask, case.spacing)
                    expected = first[0].copy(); first[0][:] = 900
                    self.assert_exact(hierarchy._principal_axis(source.full_mask, case.spacing)[0], expected)
                    alternate = source.full_mask.copy()
                    self.assert_exact(hierarchy._principal_axis(alternate, case.spacing), original(alternate, case.spacing))
                    with self.assertRaisesRegex(ValueError, 'spacing binding'):
                        hierarchy._principal_axis(source.full_mask, case.spacing + 1)
                self.assertEqual(called.call_count, 2)

    def test_binding_changes_create_distinct_entries_and_liver_shares_across_sources(self):
        _, _, _, binding = fixture()
        values = np.arange(14, dtype=np.float32)
        memory = {}
        def get(key, factory):
            if key not in memory: memory[key] = factory()
            return memory[key]
        with self.temporary() as directory:
            cache = CompactUpperCache(directory, get, lambda: None)
            calls = []
            def factory(): calls.append(1); return values
            cache.obtain('liver_raw', binding, factory)
            other_source = dict(binding, source_component=2, anchor=[2, 3, 4])
            cache.obtain('liver_raw', other_source, factory)
            changed = dict(binding, label_sha256='e'*64)
            cache.obtain('liver_raw', changed, factory)
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(list(Path(directory).glob('liver_raw/*/metadata.json'))), 2)

    def test_corrupt_disk_output_is_rejected_without_original_recompute(self):
        _, _, _, binding = fixture()
        for corruption in ('payload', 'metadata', 'array_header'):
            with self.subTest(corruption=corruption), self.temporary() as directory:
                cache = CompactUpperCache(directory, lambda key, factory: factory(), lambda: None)
                value = np.arange(14, dtype=np.float32)
                cache.obtain('liver_raw', binding, lambda: value)
                metadata_path = next(Path(directory).glob('liver_raw/*/metadata.json'))
                payload = metadata_path.with_name('arrays.npz')
                if corruption == 'payload':
                    payload.write_bytes(payload.read_bytes() + b'UNIT_changed')
                elif corruption == 'metadata':
                    metadata_path.write_text('{}')
                else:
                    import hashlib
                    with payload.open('wb') as stream: np.savez(stream, a0=value.astype(np.float64))
                    metadata = json.loads(metadata_path.read_text()); metadata.pop('metadata_sha256')
                    metadata.update(payload_bytes=payload.stat().st_size,
                                    payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest())
                    metadata['metadata_sha256'] = _digest(metadata)
                    metadata_path.write_text(json.dumps(metadata))
                def forbidden(): raise AssertionError('Corrupt existing output must not recompute')
                with self.assertRaises(ValueError): cache.obtain('liver_raw', binding, forbidden)

    def test_context_and_factory_failures_restore_every_original_helper(self):
        case, source, regions, binding = fixture()
        names = ('_source_raw', '_lesions', '_liver_raw', '_principal_axis')
        saved = tuple(getattr(hierarchy, name) for name in names)
        with self.temporary() as directory:
            provider = compact_upper_provider(BaseProvider)(directory, hierarchy, binding)
            with self.assertRaisesRegex(RuntimeError, 'UNIT interrupted'):
                with provider._upper_context({'id': 'UNIT_source'}, case, source, regions):
                    raise RuntimeError('UNIT interrupted')
            self.assertEqual(tuple(getattr(hierarchy, name) for name in names), saved)
            with patch.object(hierarchy, '_liver_raw', side_effect=ValueError('UNIT original failure')) as fail:
                with self.assertRaisesRegex(ValueError, 'UNIT original failure'):
                    with provider._upper_context({'id': 'UNIT_source'}, case, source, regions):
                        hierarchy._liver_raw(case, regions, tumor_label=2, ct_clip=(-200., 250.))
                self.assertIs(hierarchy._liver_raw, fail)
            self.assertFalse(list((Path(directory)/'compact_upper_v1').glob('liver_raw/*/metadata.json')))


if __name__ == '__main__':
    unittest.main()
