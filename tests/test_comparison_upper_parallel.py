"""CPU UNIT concurrency/identity tests; no real-CT throughput claim."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import FunctionType, ModuleType, SimpleNamespace
import unittest

import numpy as np

from hiercp_v1x.comparison_data_timing import timed_provider
from hiercp_v1x.comparison_upper_cache import compact_upper_provider
from tests.artifacts import unit_artifact_root
from tests.test_comparison_upper_cache import BaseProvider, fixture


def runtime_module():
    module = ModuleType('UNIT_thread_local_original_upper')
    module.np = np
    # Explicit analytic helper functions expose routing and nested calls. The
    # companion compact-upper tests compare actual original helper arrays.
    exec('''
UPPER_RAW_DIM = 14
def _principal_axis(mask, spacing):
    return np.asarray([mask.sum(), spacing[0], spacing[1]], np.float32), float(spacing[2])
def _source_raw(case, source, regions, occupied_without_source, *, ct_clip):
    axis, anisotropy = _principal_axis(source.full_mask, case.spacing)
    return np.full(14, case.image.flat[0] + axis[0] + anisotropy, np.float32)
def _lesions(case, source, regions, *, tumor_label, max_lesions, ct_clip):
    return (np.full((1, 14), case.image.flat[0], np.float32),
            np.asarray([source.anchor_center], np.float32), np.asarray([source.component_id], np.int64))
def _liver_raw(case, regions, *, tumor_label, ct_clip):
    return np.full(14, case.image.flat[0] + 20, np.float32)
def build_generation_specs(case, source, regions):
    return case.paths.case_id
def build_patient_graph(case, source, regions, *, barrier=None, fail=False):
    if barrier is not None: barrier.wait(timeout=10)
    occupied = (case.label == 2) & ~source.full_mask
    first = _source_raw(case, source, regions, occupied, ct_clip=(-200., 250.))
    if fail: raise RuntimeError('UNIT original patient failure')
    return (first, _lesions(case, source, regions, tumor_label=2, max_lesions=None, ct_clip=(-200., 250.)),
            _liver_raw(case, regions, tumor_label=2, ct_clip=(-200., 250.)),
            _principal_axis(source.full_mask, case.spacing))
def build_prototype_graph(patient):
    return patient
def build_inference_sample(case, source, regions, *, barrier=None, fail=False):
    specs = build_generation_specs(case, source, regions)
    patient = build_patient_graph(case, source, regions, barrier=barrier, fail=fail)
    return specs, build_prototype_graph(patient)
def collate_samples(samples):
    return tuple(samples)
''', module.__dict__)
    return module


def contexts():
    rows = []
    for index in range(2):
        case, source, regions, binding = fixture()
        case.paths.case_id = 'UNIT_parallel_' + str(index)
        case.image = case.image + np.float32(index * 100)
        case.spacing = case.spacing + np.float32(index)
        binding.update(case_id=case.paths.case_id,
                       image_sha256=str(index + 1) * 64,
                       label_sha256=str(index + 3) * 64,
                       spacing=list(map(float, case.spacing)))
        rows.append((case, source, regions, binding))
    return rows


class Provider(BaseProvider):
    def __init__(self, root, module, rows):
        super().__init__(root, module, rows[0][3])
        self.rows = rows
        self.runtime = SimpleNamespace(hierarchy=module, cache=module, data=module)
        self.original_context_entries = 0
        self.barrier = None
        self.fail_index = None

    def _runtime(self):
        return self.runtime

    def _upper_context(self, example, case, source, regions):
        binding = self.rows[example['index']][3]
        with self._lock:
            self._upper_bindings[example['id']] = copy.deepcopy(binding)

        @contextmanager
        def forbidden_global_context():
            self.original_context_entries += 1
            raise AssertionError('Original module-patching context must never be entered')
            yield
        return forbidden_global_context()

    def batch(self, indices, arm, epoch, training, full=False):
        runtime = self._runtime()  # Deliberately acquired before context entry.
        values = []
        for index in indices:
            case, source, regions, _ = self.rows[index]
            with self._upper_context(dict(id=case.paths.case_id, index=index), case, source, regions):
                values.append(runtime.cache.build_inference_sample(case, source, regions,
                    barrier=self.barrier, fail=index == self.fail_index))
        return runtime.data.collate_samples(values)


class ParallelUpperTests(unittest.TestCase):
    def temporary(self):
        return TemporaryDirectory(prefix='UNIT_parallel_upper_', dir=unit_artifact_root())

    def assert_exact(self, actual, expected):
        if isinstance(expected, np.ndarray):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.strides, expected.strides)
            self.assertEqual(actual.tobytes(), expected.tobytes())
        elif isinstance(expected, tuple):
            self.assertIsInstance(actual, tuple)
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_exact(a, b)
        else:
            self.assertEqual(actual, expected)

    def test_two_contexts_overlap_without_module_mutation_and_timing_keeps_bindings(self):
        for timed in (False, True):
            with self.subTest(timed=timed), self.temporary() as directory:
                module, rows = runtime_module(), contexts()
                names = ('_source_raw', '_lesions', '_liver_raw', '_principal_axis',
                         'build_patient_graph', 'build_inference_sample')
                originals = {name: getattr(module, name) for name in names}
                expected = [(module.build_inference_sample(*row[:3]),) for row in rows]
                cls = compact_upper_provider(Provider)
                log = Path(directory) / 'input.jsonl'
                if timed:
                    cls = timed_provider(cls, log)
                provider = cls(directory, module, rows)
                self.assertTrue(provider.independent_cold_inputs)
                provider.barrier = threading.Barrier(2)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(provider.batch, [index], 'native', 3, True)
                               for index in range(2)]
                    actual = [future.result(timeout=20) for future in futures]
                for value, baseline in zip(actual, expected):
                    self.assert_exact(value, baseline)
                self.assertEqual(provider.original_context_entries, 0)
                for name, original in originals.items():
                    self.assertIs(getattr(module, name), original)
                self.assertIs(provider._runtime().hierarchy._source_raw, originals['_source_raw'])
                provider.barrier = None
                for index in range(2):
                    self.assert_exact(provider.batch([index], 'native', 4, True), expected[index])
                if timed:
                    records = [json.loads(line) for line in log.read_text().splitlines()]
                    self.assertEqual(len(records), 4)
                    for record in records:
                        self.assertGreater(record['regions']['patient_graph_seconds'], 0)
                        self.assertEqual(record['inclusive_details']['source_raw_seconds']['calls'], 1)

    def test_failure_clears_its_thread_context_without_affecting_other_context(self):
        with self.temporary() as directory:
            module, rows = runtime_module(), contexts()
            original = module.build_inference_sample
            provider = compact_upper_provider(Provider)(directory, module, rows)
            provider.barrier, provider.fail_index = threading.Barrier(2), 0

            def call(index):
                try:
                    return provider.batch([index], 'native', 3, True)
                finally:
                    self.assertIsNone(getattr(provider._upper_local, 'active', None))
                    self.assertIs(provider._runtime().cache.build_inference_sample, original)
            with ThreadPoolExecutor(max_workers=2) as pool:
                failed, succeeded = pool.submit(call, 0), pool.submit(call, 1)
                with self.assertRaisesRegex(RuntimeError, 'UNIT original patient failure'):
                    failed.result(timeout=20)
                self.assert_exact(succeeded.result(timeout=20), (original(*rows[1][:3]),))
            provider.barrier, provider.fail_index = None, None
            self.assert_exact(call(0), (original(*rows[0][:3]),))

    def test_bound_functions_preserve_code_and_reject_wrong_arguments(self):
        with self.temporary() as directory:
            module, rows = runtime_module(), contexts()
            provider = compact_upper_provider(Provider)(directory, module, rows)
            case, source, regions, _ = rows[0]
            example = dict(id=case.paths.case_id, index=0)
            with provider._upper_context(example, case, source, regions):
                runtime = provider._runtime()
                inference = runtime.cache.build_inference_sample
                self.assertIsInstance(inference, FunctionType)
                self.assertIs(inference.__code__, module.build_inference_sample.__code__)
                patient = inference.__globals__['build_patient_graph']
                self.assertIs(patient.__code__, module.build_patient_graph.__code__)
                occupied = (case.label == 2) & ~source.full_mask
                helpers = runtime.hierarchy
                with self.assertRaisesRegex(ValueError, 'another verified'):
                    helpers._source_raw(rows[1][0], source, regions, occupied, ct_clip=(-200., 250.))
                with self.assertRaisesRegex(ValueError, 'occupied-mask'):
                    helpers._source_raw(case, source, regions, occupied.astype(np.int16), ct_clip=(-200., 250.))
                with self.assertRaisesRegex(ValueError, 'annotation/limit'):
                    helpers._lesions(case, source, regions, tumor_label=2, max_lesions=1, ct_clip=(-200., 250.))
                with self.assertRaisesRegex(ValueError, 'another verified'):
                    helpers._liver_raw(case, rows[1][2], tumor_label=2, ct_clip=(-200., 250.))
                with self.assertRaisesRegex(ValueError, 'spacing binding'):
                    helpers._principal_axis(source.full_mask, rows[1][0].spacing)
                with self.assertRaisesRegex(RuntimeError, 'Nested static upper'):
                    with provider._upper_context(example, case, source, regions):
                        pass
            self.assertIs(runtime.cache.build_inference_sample, module.build_inference_sample)
            changed = copy.copy(case)
            changed.spacing = case.spacing + 1
            with self.assertRaisesRegex(ValueError, 'Actual source/CT identity'):
                with provider._upper_context(example, changed, source, regions):
                    pass


if __name__ == '__main__':
    unittest.main()
