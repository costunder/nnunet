"""CPU UNIT exact compact-cache tests; analytic data, no medical/GPU claim."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, is_dataclass
import copy
import json
import os
from pathlib import Path
import random
import socket
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hiercp import common, local
from hiercp_v1x.u_bridge_data import UBridgeData, _resident_size
from hiercp_v1x.comparison_source_cache import (source_cache_provider, source_binding,
    compact_source, restore_source, _digest, _sha)


ROOT = Path(__file__).resolve().parents[1]


def case_fixture(shape=(23, 25, 27), *, fortran=False):
    label = np.ones(shape, dtype=np.int16)
    label[0:2, 1:4, 0:2] = 2
    label[12:16, 15:18, 17:20] = 2
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 13
    if fortran:
        image = np.asfortranarray(image)
    return SimpleNamespace(paths=SimpleNamespace(case_id='UNIT_source'), image=image,
        label=label, shape=shape, spacing=np.asarray([1.1, .7, 2.], dtype=np.float32))


def assert_exact(test, left, right):
    test.assertEqual(type(left), type(right))
    if isinstance(left, np.ndarray):
        test.assertEqual(left.dtype, right.dtype)
        test.assertEqual(left.shape, right.shape)
        test.assertEqual(left.strides, right.strides)
        np.testing.assert_array_equal(left, right)
    elif torch.is_tensor(left):
        test.assertEqual(left.dtype, right.dtype)
        test.assertEqual(left.shape, right.shape)
        test.assertEqual(left.stride(), right.stride())
        test.assertTrue(torch.equal(left, right))
    elif is_dataclass(left):
        for field in fields(left):
            assert_exact(test, getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_exact(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for first, second in zip(left, right):
            assert_exact(test, first, second)
    else:
        test.assertEqual(left, right)


class OriginalProvider:
    """Use the real unmodified source selector and provider LRU/source method.

    Canonical values below are explicit analytic UNIT tensors; real CT canonical
    construction and complete train8/validation129 equivalence are separate.
    """
    _source = UBridgeData._source
    _get = UBridgeData._get

    def __init__(self, root, *, mode='random', pad=2, resident=2**28):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/'.data.lock').write_text(json.dumps(dict(host=socket.gethostname(),
            pid=os.getpid(), token='UNIT_owned_source_namespace')), encoding='utf8')
        self.config = dict(seed=42, cache=dict(source_selection=mode, source_pad=pad),
            graph=dict(adaptive_roi_margin_mm=10., context_outer_radius_mm=10.), ct_clip=[-200., 250.])
        self.raw = {'UNIT_source': dict(image_sha256='a'*64, label_sha256='b'*64)}
        self._scope = dict(source_archive_sha256='c'*64, contract_sha256='d'*64,
            original_module_sha256={'hiercp/common.py': 'e'*64, 'hiercp/local.py': 'f'*64})
        self._cache, self._lock = OrderedDict(), threading.RLock()
        self._resident_bytes, self.resident_limit = 0, resident
        self.stats = dict(resident_hits=0, evictions=0)
        self.original_calls = self.budget_checks = self.runtime_checks = 0
        self.runtime = SimpleNamespace(common=common,
            local=SimpleNamespace(PreparedLocalSource=local.PreparedLocalSource,
                                  prepare_local_source=self.prepare),
            schema=SimpleNamespace(graph_config_from_dict=lambda value: value))

    def _runtime(self):
        self.runtime_checks += 1
        return self.runtime

    def _check_budget(self):
        self.budget_checks += 1

    def report(self):
        return dict(UNIT=True, resident_bytes=self._resident_bytes)

    def prepare(self, case, source, **kwargs):
        self.original_calls += 1
        base = torch.arange(48, dtype=torch.float32).reshape(8, 6)
        node = dict(x=base[:, ::2], grid=base[:, 1::2], pos=base[:, ::2]/7)
        return local.PreparedLocalSource(source_footprint=source.patch_mask.copy(),
            source_patch=(np.arange(5*4*5*6, dtype=np.float32).reshape(5, 4, 5, 6)/17),
            canonical_nodes={'source_context': node},
            canonical_edges={('source_context', 'within', 'source_context'):
                torch.tensor([[0, 1, 6], [2, 3, 7]], dtype=torch.int32)},
            canonical_counts={'source_context': 8})

    def example(self, case, index=0):
        source, _, _ = common.choose_source_tumor(case.image, case.label, tumor_label=2,
            selection=self.config['cache']['source_selection'], pad=self.config['cache']['source_pad'],
            rng=np.random.default_rng(common.stable_case_seed(self.config['seed'],
                case.paths.case_id, f'sample_{index}')))
        return dict(id=f'UNIT_source:{index}', case_id='UNIT_source', sample_index=index,
            source_component=source.component_id, positive_center=source.anchor_center,
            original_sample_sha256='1'*64)


class SourceCacheTests(unittest.TestCase):
    def directory(self):
        return TemporaryDirectory(prefix='UNIT_compact_source_', dir=ROOT)

    def obtain(self, provider, case, example=None):
        example = provider.example(case) if example is None else example
        return provider._source(example, case, case.label > 0, np.ones(case.shape, dtype=np.float32))

    def test_cold_resident_evicted_and_reopened_equal_original_all_selection_modes(self):
        for mode in ('random', 'largest', 'size_weighted'):
            for pad in (0, 2):
                with self.subTest(mode=mode, pad=pad), self.directory() as folder:
                    case = case_fixture(fortran=True)
                    baseline = OriginalProvider(Path(folder)/'baseline', mode=mode, pad=pad)
                    current = source_cache_provider(OriginalProvider)(Path(folder)/'cached', mode=mode, pad=pad)
                    for index in (0, 1):
                        example = baseline.example(case, index)
                        wanted = self.obtain(baseline, case, example)
                        first = self.obtain(current, case, example)
                        assert_exact(self, wanted, first)
                        self.assertIs(first, self.obtain(current, case, example))
                        current._cache.clear(); current._resident_bytes = 0
                        assert_exact(self, wanted, self.obtain(current, case, example))
                    reopened = source_cache_provider(OriginalProvider)(Path(folder)/'cached', mode=mode, pad=pad)
                    for index in (0, 1):
                        example = baseline.example(case, index)
                        assert_exact(self, self.obtain(baseline, case, example), self.obtain(reopened, case, example))
                    self.assertEqual(current.original_calls, 2)
                    self.assertEqual(reopened.original_calls, 0)
                    self.assertEqual(reopened.source_cache_stats['disk_hits'], 2)
                    self.assertGreater(reopened.runtime_checks, 0)

    def test_publication_omits_full_volume_mask_and_preserves_exact_centroid(self):
        with self.directory() as folder:
            provider = source_cache_provider(OriginalProvider)(folder)
            case = case_fixture((70, 72, 74))
            example = provider.example(case)
            first = self.obtain(provider, case, example)
            path = next((Path(folder)/'source_prepared').glob('*/payload.pt'))
            payload = torch.load(path, weights_only=False)
            self.assertNotIn('full_mask', payload['source'])
            self.assertLess(path.stat().st_size, first[0].full_mask.nbytes)
            self.assertEqual(payload['source']['centroid'], first[0].centroid)
            provider._cache.clear(); provider._resident_bytes = 0
            with patch.object(common, 'choose_source_tumor', side_effect=AssertionError('must reuse original selection')):
                assert_exact(self, first, self.obtain(provider, case, example))
            report = provider.report()['source_cache_stats']
            self.assertEqual(report['original_builds'], 1)
            self.assertEqual(report['disk_hits'], 1)
            self.assertFalse(report['full_volume_mask_stored'])

    def test_global_rng_and_original_functions_unchanged(self):
        with self.directory() as folder:
            provider = source_cache_provider(OriginalProvider)(folder)
            case = case_fixture(); example = provider.example(case)
            py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
            original_choose, original_source = common.choose_source_tumor, UBridgeData._source
            self.obtain(provider, case, example)
            provider._cache.clear(); provider._resident_bytes = 0
            self.obtain(provider, case, example)
            self.assertEqual(py_state, random.getstate())
            after = np.random.get_state()
            self.assertEqual(np_state[0], after[0]); np.testing.assert_array_equal(np_state[1], after[1])
            self.assertEqual(np_state[2:], after[2:])
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
            self.assertIs(original_choose, common.choose_source_tumor)
            self.assertIs(original_source, UBridgeData._source)

    def test_binding_changes_for_raw_scope_graph_pad_seed_sample_and_spacing(self):
        with self.directory() as folder:
            provider = source_cache_provider(OriginalProvider)(folder)
            case = case_fixture(); example = provider.example(case)
            original = source_binding(provider, example, case)
            mutations = (
                lambda: provider.raw['UNIT_source'].update(label_sha256='2'*64),
                lambda: provider._scope.update(contract_sha256='3'*64),
                lambda: provider.config['graph'].update(adaptive_roi_margin_mm=20.),
                lambda: provider.config['cache'].update(source_pad=3),
                lambda: provider.config.update(seed=43),
                lambda: example.update(sample_index=7),
                lambda: example.update(original_sample_sha256='4'*64),
                lambda: setattr(case, 'spacing', np.asarray([1.2, .7, 2.], dtype=np.float32)),
            )
            for mutate in mutations:
                mutate()
                changed = source_binding(provider, example, case)
                self.assertNotEqual(_digest(original), _digest(changed))
                original = changed

    def test_original_lru_budget_owns_all_reconstructed_storage(self):
        with self.directory() as folder:
            provider = source_cache_provider(OriginalProvider)(folder, resident=1)
            case = case_fixture(); example = provider.example(case)
            first = self.obtain(provider, case, example)
            self.assertGreater(_resident_size(first), 1)
            self.assertEqual(provider._resident_bytes, 0)
            self.assertFalse(provider._cache)
            assert_exact(self, first, self.obtain(provider, case, example))
            self.assertEqual(provider.original_calls, 1)
            self.assertEqual(provider.source_cache_stats['disk_hits'], 1)
            self.assertGreaterEqual(provider.budget_checks, 4)

    def test_corruption_and_signed_wrong_geometry_are_not_rebuilt_or_hidden(self):
        with self.directory() as folder:
            provider = source_cache_provider(OriginalProvider)(folder)
            case = case_fixture(); example = provider.example(case)
            self.obtain(provider, case, example)
            directory = next(p for p in (Path(folder)/'source_prepared').iterdir() if p.is_dir())
            payload_path, metadata_path = directory/'payload.pt', directory/'metadata.json'
            provider._cache.clear(); provider._resident_bytes = 0
            before = payload_path.read_bytes()
            with payload_path.open('ab') as stream: stream.write(b'UNIT corruption')
            with self.assertRaisesRegex(ValueError, 'binding/content'):
                self.obtain(provider, case, example)
            payload_path.write_bytes(before)
            payload = torch.load(payload_path, weights_only=False)
            payload['source']['anchor_center'] = (0, 0, 0)
            torch.save(payload, payload_path)
            meta = json.loads(metadata_path.read_text())
            meta.update(payload_sha256=_sha(payload_path), payload_bytes=payload_path.stat().st_size)
            meta.pop('metadata_sha256'); meta['metadata_sha256'] = _digest(meta)
            metadata_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'component/anchor'):
                self.obtain(provider, case, example)
            self.assertEqual(provider.original_calls, 1)
            self.assertFalse(provider._cache)

    def test_compaction_rejects_source_voxels_outside_original_patch(self):
        with self.directory() as folder:
            provider = OriginalProvider(folder); case = case_fixture()
            example = provider.example(case)
            result = self.obtain(provider, case, example)
            source = copy.deepcopy(result[0])
            source.full_mask[-1, -1, -1] = True
            with self.assertRaisesRegex(ValueError, 'reconstructed exactly'):
                compact_source((source, result[1]), provider.runtime, source_binding(provider, example, case))

    def test_failed_original_factory_propagates_and_next_call_has_clean_context(self):
        with self.directory() as folder:
            provider = source_cache_provider(OriginalProvider)(folder)
            case = case_fixture(); example = provider.example(case)
            with patch.object(provider.runtime.local, 'prepare_local_source', side_effect=ValueError('UNIT original failure')):
                with self.assertRaisesRegex(ValueError, 'UNIT original failure'):
                    self.obtain(provider, case, example)
            self.assertFalse(list((Path(folder)/'source_prepared').glob('*/metadata.json')))
            self.obtain(provider, case, example)
            self.assertEqual(provider.original_calls, 1)

    def test_concurrent_requests_publish_once_and_share_the_original_lru(self):
        with self.directory() as folder:
            provider = source_cache_provider(OriginalProvider)(folder)
            case = case_fixture(); example = provider.example(case)
            with ThreadPoolExecutor(max_workers=4) as pool:
                values = list(pool.map(lambda _: self.obtain(provider, case, example), range(8)))
            self.assertEqual(provider.original_calls, 1)
            self.assertEqual(len(list((Path(folder)/'source_prepared').glob('*/metadata.json'))), 1)
            self.assertTrue(all(value is values[0] for value in values))


if __name__ == '__main__':
    unittest.main()
