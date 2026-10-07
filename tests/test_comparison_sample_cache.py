"""CPU UNIT referenced-sample equivalence; analytic topology, no CT claim."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import random
import runpy
import socket
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import HeteroData

from hiercp import data, local, sample, schema
from hiercp_v1x.comparison_data import ComparisonData
from hiercp_v1x.comparison_data_timing import timed_provider
from hiercp_v1x.comparison_sample_cache import sample_cache_provider, _same
from hiercp_v1x.comparison_views import parallel_view_provider
from hiercp_v1x.u_bridge_data import _hash, _sha, _resident_size


ROOT = Path(__file__).resolve().parents[1]
canonical_sample = runpy.run_path(str(ROOT / 'tests/test_comparison_views.py'))['canonical_sample']


class UnitProvider(ComparisonData):
    """Real original key policy/LRU/collation with explicit analytic preparation."""
    def __init__(self, root, *, resident=2**28):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        owner = self.root / '.data.lock'
        if not owner.exists():
            owner.write_text(json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), token='UNIT_layout')))
        self.graph_dir = self.root / 'canonical_local'
        self.graph_dir.mkdir(exist_ok=True)
        template = canonical_sample(1)
        self.config = dict(seed=42, graph=template['graph_config'], ct_clip=[-200., 250.])
        self.workers, self.resident_limit = 4, resident
        self._cache, self._lock = OrderedDict(), threading.RLock()
        self._resident_bytes = 0
        self.stats = dict(resident_hits=0, evictions=0, disk_hits=0)
        self.bank_fingerprint = '1' * 64
        self.bank = SimpleNamespace(fingerprint=lambda: self.bank_fingerprint)
        self._scope = dict(source_archive_sha256='2'*64, contract_sha256='3'*64,
            original_module_sha256={'hiercp/common.py': '4'*64})
        self.raw = {}
        self._examples = []
        self.calls = dict(original=0, case=0, source=0, upper=0, budget=0)
        for i in range(2):
            case_id = 'UNIT_ANALYTIC_TOPOLOGY_' + str(i)
            image, label = self.root / f'image_{i}.bin', self.root / f'label_{i}.bin'
            for path, value in ((image, b'UNIT explicit image identity'), (label, b'UNIT explicit label identity')):
                if not path.exists():
                    path.write_bytes(value + bytes([i]))
            self.raw[case_id] = dict(image=str(image), label=str(label),
                image_sha256=_sha(image), label_sha256=_sha(label), shape=[200, 200, 200], spacing=[1., 1., 1.])
            self._examples.append(dict(index=i, id=case_id+':17', case_id=case_id, sample_index=17,
                source_component=1, positive_center=(0, 0, 0),
                selected_centers=tuple((c, 1, 1) for c in range(1, 8)),
                native_centers=tuple((c, 2, 2) for c in range(1, 129)),
                original_sample_sha256='5'*64, split='train', partition='train'))
        self.runtime = SimpleNamespace(sample=sample, schema=schema, local=local, data=data,
            cache=SimpleNamespace(build_inference_sample=lambda: None))

    def _runtime(self):
        return self.runtime

    def _check_budget(self):
        self.calls['budget'] += 1

    def report(self):
        return dict(resident_bytes=self._resident_bytes, **self.stats)

    def _case(self, *args):
        self.calls['case'] += 1

    def _source(self, *args):
        self.calls['source'] += 1

    def _upper_context(self, *args):
        self.calls['upper'] += 1
        return nullcontext()

    def _local_map(self, built):
        return built

    def sample(self, index, center_keys, epoch, training):
        self.calls['original'] += 1
        self._case(index); self._source(index)
        with self._upper_context(index):
            example = self._example(index)
            centers = self._centers_for(example, center_keys)
            source = canonical_sample(1)['source_local']
            built = []
            for center in centers:
                binding = json.loads(json.dumps(self._binding(example, center)))
                key = _hash(binding)
                def build(center=center, binding=binding, key=key):
                    path = self.graph_dir / (key + '.pt')
                    if path.exists():
                        return torch.load(path, weights_only=False)['built']
                    target = canonical_sample(1)['target_locals'][0]
                    target['nodes']['target_context']['x'][:, 0] += sum(center)/1000.
                    patch_values = np.arange(3*4*5*6, dtype=np.float32).reshape(3, 4, 5, 6) / 17
                    value = local.BuiltLocalGraph(graph=None,
                        source_patch=np.asfortranarray(patch_values),
                        target_patch=patch_values + sum(center)/101.,
                        source_local=source, target_local=target)
                    torch.save(dict(binding=binding, built=value), path)
                    return value
                built.append(self._get(('local', key), build))
            built = self._local_map(built)
            canonical = dict(case_id=example['case_id'], sample_index=17, split='train', source_component=1,
                graph_config=self.config['graph'], prototype_fingerprint=self.bank_fingerprint,
                candidate_centers=torch.tensor(centers), difficulties=torch.tensor([0]+[1]*(len(centers)-1)),
                corruptions=torch.zeros(len(centers), dtype=torch.long), UNIT_metadata=('ordered', 17),
                source_patch=torch.from_numpy(built[0].source_patch.astype(np.float16)),
                target_patches=torch.from_numpy(np.stack([b.target_patch for b in built]).astype(np.float16)),
                source_local=built[0].source_local, target_locals=[b.target_local for b in built])
            for name in ('patient_graph', 'prototype_graph'):
                graph = HeteroData()
                graph['candidate'].raw_x = torch.arange(len(centers)*8, dtype=torch.float32).reshape(-1,8)[:,::2]
                graph['candidate'].num_nodes = len(centers)
                graph['candidate','ordered','candidate'].edge_index = torch.tensor([[0, 1], [1, 0]])
                graph.UNIT_metadata = {'name': name, 'row_order': list(range(len(centers)))}
                canonical[name] = graph
        result = self._runtime().sample.materialize_sample_views(canonical,
            training=training, epoch=epoch, global_seed=self.config['seed'])
        result['bridge_source_id'] = example['id']
        result['bridge_center_keys'] = tuple(center_keys)
        return result


def evict(provider):
    with provider._lock:
        provider._cache.clear()
        provider._resident_bytes = 0


class SampleCacheTests(unittest.TestCase):
    def directory(self):
        return TemporaryDirectory(prefix='UNIT_sample_layout_', dir=ROOT)

    def obtain(self, provider, arm='native_fixed', epoch=2, training=True, full=False, index=0):
        keys = provider.candidate_keys(index, arm, epoch, full=full)
        return provider.sample(index, keys, epoch, training)

    def assert_exact(self, first, second):
        self.assertTrue(_same(first, second), 'Values, strides, types or graph metadata differ')

    def forbid_preparation(self, provider):
        def fail(*args, **kwargs):
            raise AssertionError('Completed sample unexpectedly re-entered preparation')
        provider._case = provider._source = provider._upper_context = fail

    def test_cold_hot_evicted_reopened_preserve_every_value_layout_and_metadata(self):
        with self.directory() as folder:
            original = UnitProvider(Path(folder)/'original')
            klass = sample_cache_provider(parallel_view_provider(UnitProvider))
            provider = klass(Path(folder)/'adapted')
            expected = self.obtain(original)
            cold = self.obtain(provider)
            self.assert_exact(expected, cold)
            self.forbid_preparation(provider)
            self.assert_exact(expected, self.obtain(provider))
            evict(provider)
            self.assert_exact(expected, self.obtain(provider))
            fresh = klass(provider.root)
            self.forbid_preparation(fresh)
            self.assert_exact(expected, self.obtain(fresh))
            self.assertEqual(provider.report()['sample_cache']['original_builds'], 1)
            self.assertEqual(fresh.report()['sample_cache']['original_builds'], 0)
            self.assertEqual(fresh.report()['sample_cache']['raw_verifications'], 2)

    def test_all_four_arm_keysets_epochs_and_full129_validation(self):
        with self.directory() as folder:
            original = UnitProvider(Path(folder)/'original')
            provider = sample_cache_provider(parallel_view_provider(UnitProvider))(Path(folder)/'adapted')
            for arm, epoch, training, full in (
                ('selected',1,True,False), ('selected',2,True,False),
                ('native',1,True,False), ('native',2,True,False),
                ('native_fixed',2,True,False), ('native_listwise',2,True,False),
                ('native_fixed',2,False,True), ('native',3,False,True)):
                with self.subTest(arm=arm, epoch=epoch, full=full):
                    args = (arm, epoch, training, full)
                    self.assert_exact(self.obtain(original, *args), self.obtain(provider, *args))
                    self.assert_exact(self.obtain(original, *args), self.obtain(provider, *args))
            self.assertEqual(provider.report()['sample_cache']['original_builds'], 4)
            self.assertTrue(provider.report()['sample_cache']['rotating_candidate_sets_require_first_build'])

    def test_original_lru_owns_arrays_and_graphs_are_not_mutated_by_consumers(self):
        with self.directory() as folder:
            provider = sample_cache_provider(UnitProvider)(folder, resident=1)
            first = self.obtain(provider)
            self.assertEqual(provider._resident_bytes, 0)
            self.assertFalse(provider._cache)
            self.assert_exact(first, self.obtain(provider))
            provider.resident_limit = 2**28
            expected = self.obtain(provider)
            modified = self.obtain(provider)
            modified['patient_graph']['candidate'].raw_x.fill_(999)
            modified['prototype_graph'].UNIT_metadata['row_order'].reverse()
            self.assert_exact(expected, self.obtain(provider))
            self.assertLessEqual(provider._resident_bytes, provider.resident_limit)
            self.assertTrue(any(key[0]=='sample_layout' for key in provider._cache))

    def test_skeleton_contains_no_large_local_graph_or_patch_duplicates(self):
        with self.directory() as folder:
            provider = sample_cache_provider(UnitProvider)(folder)
            self.obtain(provider, full=True, training=False)
            directory = next(p for p in (provider.root/'sample_layout').iterdir() if p.is_dir())
            self.assertEqual({p.name for p in directory.iterdir()}, {'metadata.json','skeleton.pt'})
            skeleton = torch.load(directory/'skeleton.pt', weights_only=False)
            self.assertFalse(set(skeleton) & {'source_patch','target_patches','source_local','target_locals',
                                            'local_graphs','local_graphs_view2'})
            metadata = json.loads((directory/'metadata.json').read_text())
            self.assertEqual(len(metadata['references']), 129)
            self.assertFalse(metadata['local_graphs_and_patches_stored'])
            self.assertLess((directory/'skeleton.pt').stat().st_size,
                sum(path.stat().st_size for path in provider.graph_dir.glob('*.pt')))

    def test_global_rng_and_original_modules_are_unchanged(self):
        with self.directory() as folder:
            provider = sample_cache_provider(parallel_view_provider(UnitProvider))(folder)
            python, numpy, tensor = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
            original = sample.materialize_sample_views
            self.obtain(provider); self.obtain(provider); evict(provider); self.obtain(provider)
            self.assertEqual(python, random.getstate())
            current = np.random.get_state()
            self.assertEqual(numpy[0], current[0]); np.testing.assert_array_equal(numpy[1], current[1])
            self.assertEqual(numpy[2:], current[2:]); self.assertTrue(torch.equal(tensor,torch.get_rng_state()))
            self.assertIs(sample.materialize_sample_views, original)

    def test_missing_or_corrupt_reference_is_rejected_on_eviction_without_rebuild(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), self.directory() as folder:
                provider = sample_cache_provider(UnitProvider)(folder)
                expected = self.obtain(provider); self.obtain(provider)
                path = next(provider.graph_dir.glob('*.pt'))
                if corrupt:
                    with path.open('ab') as stream: stream.write(b'UNIT_corruption')
                else:
                    path.rename(path.with_suffix('.UNIT_missing'))
                self.forbid_preparation(provider)
                # Existing immutable RAM objects need no repeated filesystem probes.
                self.assert_exact(expected, self.obtain(provider))
                evict(provider)
                with self.assertRaises((ValueError,FileNotFoundError)):
                    self.obtain(provider)
                self.assertEqual(provider.report()['sample_cache']['original_builds'], 1)

    def test_fresh_process_raw_and_skeleton_corruption_are_rejected(self):
        for target in ('raw','skeleton','metadata'):
            with self.subTest(target=target), self.directory() as folder:
                klass = sample_cache_provider(UnitProvider)
                provider = klass(folder)
                self.obtain(provider)
                fresh = klass(folder)
                if target=='raw':
                    path = Path(provider.raw[provider._example(0)['case_id']]['image'])
                else:
                    directory = next(p for p in (provider.root/'sample_layout').iterdir() if p.is_dir())
                    path = directory/('skeleton.pt' if target=='skeleton' else 'metadata.json')
                with path.open('ab') as stream: stream.write(b'UNIT_corruption')
                self.forbid_preparation(fresh)
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    self.obtain(fresh)

    def test_hot_timing_parallel_runtime_and_complete_batch_metadata(self):
        with self.directory() as folder:
            klass = timed_provider(sample_cache_provider(parallel_view_provider(UnitProvider)),Path(folder)/'timing.jsonl')
            provider = klass(Path(folder)/'cache')
            first = provider.batch([0,1], 'native_fixed', 2, True)
            self.forbid_preparation(provider)
            second = provider.batch([0,1], 'native_fixed', 2, True)
            self.assert_exact(vars(first), vars(second))
            rows = [json.loads(row) for row in (Path(folder)/'timing.jsonl').read_text().splitlines()]
            self.assertEqual(rows[1]['status'], 'complete')
            self.assertGreater(rows[1]['regions']['sampled_views_seconds'],0.)
            self.assertGreater(rows[1]['regions']['collate_seconds'],0.)
            self.assertEqual(second.bridge_center_keys, first.bridge_center_keys)

    def test_readiness_never_loads_raw_or_tensors_and_never_treats_corruption_as_miss(self):
        with self.directory() as folder:
            provider = sample_cache_provider(UnitProvider)(folder)
            self.assertFalse(provider.cached_batch_ready([0], 'native_fixed', 2, training=True))
            self.obtain(provider)
            with patch('hiercp_v1x.comparison_sample_cache._sha', side_effect=AssertionError('readiness hashed input')), \
                 patch('torch.load', side_effect=AssertionError('readiness allocated tensor')):
                self.assertTrue(provider.cached_batch_ready([0], 'native_fixed', 2, training=True))
                self.assertFalse(provider.cached_batch_ready([0], 'native', 2, training=True))
            directory = next(p for p in (provider.root/'sample_layout').iterdir() if p.is_dir())
            with (directory/'metadata.json').open('ab') as stream: stream.write(b'UNIT_invalid')
            evict(provider)
            with self.assertRaises(ValueError):
                provider.cached_batch_ready([0], 'native_fixed', 2, training=True)

    def test_resident_sample_does_not_stat_hash_or_lock_files(self):
        with self.directory() as folder:
            provider = sample_cache_provider(UnitProvider)(folder)
            expected = self.obtain(provider)
            self.obtain(provider)
            with patch.object(provider, '_layout_file', side_effect=AssertionError('resident filesystem I/O')), \
                 patch('hiercp_v1x.comparison_sample_cache._safe', side_effect=AssertionError('resident path scan')):
                self.assertTrue(provider.cached_batch_ready([0], 'native_fixed', 2, training=True))
                self.assert_exact(expected, self.obtain(provider))

    def test_same_bytes_with_changed_file_metadata_reverify_after_eviction(self):
        with self.directory() as folder:
            provider = sample_cache_provider(UnitProvider)(folder)
            expected = self.obtain(provider)
            path = next(provider.graph_dir.glob('*.pt'))
            token = path.stat()
            os.utime(path, ns=(token.st_atime_ns, token.st_mtime_ns + 1_000_000))
            evict(provider)
            self.assert_exact(expected, self.obtain(provider))

    def test_concurrent_hot_reads_keep_capture_and_metadata_independent(self):
        with self.directory() as folder:
            provider = sample_cache_provider(parallel_view_provider(UnitProvider))(folder)
            expected = [self.obtain(provider,index=i) for i in range(2)]
            self.forbid_preparation(provider)
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda i: self.obtain(provider,index=i), (0,1,0,1)))
            for i,result in enumerate(results):
                self.assert_exact(expected[i%2],result)
            self.assertEqual(provider.report()['sample_cache']['original_builds'],2)


if __name__ == '__main__':
    unittest.main()
