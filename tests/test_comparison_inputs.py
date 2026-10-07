"""UNIT boundary/metadata equivalence, not a CT accuracy evaluation."""
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import contextmanager
from types import SimpleNamespace, ModuleType
import json
import unittest

import numpy as np

from hiercp import common
from hiercp_v1x.u_bridge_data import UBridgeData
from hiercp_v1x.comparison_inputs import local_mask_provider
from hiercp_v1x.comparison_data_timing import timed_provider


class InputTests(unittest.TestCase):
    def test_runtime_proxies_preserve_active_upper_wrappers_and_disjoint_totals(self):
        # Metadata-only functions expose call routing without constructing CT/graphs.
        module = ModuleType('UNIT_original_runtime')
        exec('''
def _source_raw(): return 'original'
def _lesions(): return 'lesions'
def _liver_raw(): return 'liver'
def _principal_axis(): return 'axis'
def build_generation_specs(): return 'specs'
def build_patient_graph(): return (_source_raw(), _lesions(), _liver_raw(), _principal_axis())
def build_prototype_graph(): return 'prototype'
def build_inference_sample(): return (build_generation_specs(), build_patient_graph(), build_prototype_graph())
def collate_samples(samples): return tuple(samples)
''', module.__dict__)
        class Base:
            def __init__(self): self.runtime = SimpleNamespace(cache=module, data=module)
            def _runtime(self): return self.runtime
            @contextmanager
            def _upper_context(self):
                original = module._source_raw
                module._source_raw = lambda: 'existing verified cache'
                try: yield
                finally: module._source_raw = original
            def batch(self, *args, **kwargs):
                runtime = self._runtime()  # Intentionally obtain it before upper wrappers are installed.
                with self._upper_context(): values = runtime.cache.build_inference_sample()
                return runtime.data.collate_samples([values])
        with TemporaryDirectory(prefix='UNIT_input_details_') as folder:
            path = Path(folder)/'input.jsonl'
            provider = timed_provider(Base, path)()
            expected = Base().batch()
            self.assertEqual(provider.batch([1], 'native_fixed', 2, True), expected)
            self.assertEqual(module._source_raw(), 'original')
            row = json.loads(path.read_text())
            self.assertEqual(set(row['regions']), {'upper_context_enter_seconds', 'generation_specs_seconds',
                'patient_graph_seconds', 'prototype_graph_seconds', 'collate_seconds'})
            self.assertEqual(row['inclusive_details']['source_raw_seconds']['calls'], 1)
            self.assertEqual(row['inclusive_details']['liver_raw_seconds']['calls'], 1)
            self.assertGreaterEqual(row['inclusive_details']['inference_assembly_inclusive_seconds']['seconds'],
                                    row['regions']['patient_graph_seconds'])
            self.assertAlmostEqual(row['input_seconds'], sum(row['regions'].values()) +
                                   row['other_assembly_views_collate_seconds'])

    def test_exact_candidate_metadata_at_borders_and_odd_even_footprints(self):
        class Base:
            config = {'ct_clip': [-200, 300]}
            def _runtime(self): return SimpleNamespace(common=common)
        provider = local_mask_provider(Base)()
        shape = (15, 17, 13)
        label = (np.arange(np.prod(shape)).reshape(shape) % 3).astype(np.uint8)
        image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 11
        case = SimpleNamespace(image=image, label=label, shape=shape)
        depth = np.ones(shape, np.float32)*3
        occupied = np.ones(shape, np.float32)*5
        organ = label > 0
        original_label = label.copy()
        for footprint in ((3, 3, 3), (4, 6, 2), (20, 18, 16)):
            mask = np.ones(footprint, dtype=bool)
            source = SimpleNamespace(patch_mask=mask, voxel_count=mask.sum())
            for center in ((0, 0, 0), (14, 16, 12), (7, 8, 6), (0, 8, 12)):
                with self.subTest(footprint=footprint, center=center):
                    args = (center, case, source, organ, depth, occupied)
                    old = UBridgeData._candidate(provider, *args)
                    new = provider._candidate(*args)
                    self.assertEqual(vars(old), vars(new))
        np.testing.assert_array_equal(label, original_label)

    def test_timing_regions_are_disjoint_and_provider_values_are_preserved(self):
        class Base:
            def _case(self, *args): return 2
            def _regions(self, *args): return 3
            def _source(self, *args): return 4
            def _candidate(self, *args): return 5
            def _positive_candidate(self, *args): return 6
            def _local_map(self, *args): return 7
            def batch(self, ids, arm, epoch, training, full=False):
                return (ids, self._case(), self._regions(), self._source(), self._candidate(),
                        self._positive_candidate(), self._local_map(), full)
        with TemporaryDirectory(prefix='UNIT_input_timing_') as folder:
            path = Path(folder)/'input.jsonl'
            provider = timed_provider(Base, path)()
            expected = Base().batch([7], 'native', 1, False, True)
            self.assertEqual(provider.batch([7], 'native', 1, False, True), expected)
            row = json.loads(path.read_text())
            self.assertEqual(row['status'], 'complete')
            self.assertEqual(row['source_indices'], [7])
            self.assertTrue(row['full129'])
            self.assertEqual(set(row['regions']), {'case_fields_seconds', 'regions_seconds',
                'source_seconds', 'candidate_metadata_seconds', 'local_graphs_seconds'})
            self.assertGreaterEqual(row['other_assembly_views_collate_seconds'], 0)
            self.assertAlmostEqual(row['input_seconds'], sum(row['regions'].values())+
                                   row['other_assembly_views_collate_seconds'])

    def test_failed_preparation_stays_failure_and_is_recorded(self):
        class Base:
            def batch(self, *args, **kwargs): raise ValueError('UNIT input failure')
        with TemporaryDirectory(prefix='UNIT_input_failure_') as folder:
            path = Path(folder)/'input.jsonl'
            provider = timed_provider(Base, path)()
            with self.assertRaisesRegex(ValueError, 'UNIT input failure'):
                provider.batch([7], 'native', 1, False, True)
            self.assertEqual(json.loads(path.read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
