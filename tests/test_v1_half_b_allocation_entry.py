"""CPU UNIT resource orchestration only; no neural or server quality claims."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x.half_b_entry import bind_execution_resources
from hiercp_v1x import half_b_training
from tests.test_v1_half_b_training import UnitFixture

ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30


def fixture(root):
    gpu = dict(logical_index=0, name='UNIT same measured GPU', total_vram_bytes=48*GIB,
               multiprocessors=84, compute_capability='8.6', mig_instance=False)
    actual = dict(selected_device='cuda', cuda_available=True, cuda_visible_device_count=1,
                  gpu_devices=[gpu], cuda_visible_devices_env='UNIT_new_uuid', platform='UNIT_new_host')
    expected = deepcopy(actual)
    expected.update(cuda_visible_devices_env='UNIT_old_uuid', platform='UNIT_old_host')
    receipt = dict(experiment=str(root), contract_sha256='ab'*32,
        baseline_experiment=str(root.parent/'UNIT_preserved_baseline'),
        allocation_policy='current_allocation', cuda_gib=40., rss_gib=192.,
        config=dict(training=dict(batch_size=2, num_workers=4, gradient_accumulation_steps=1)),
        baseline_proof=dict(execution=dict(resource_fingerprint=expected, physical_batch=2, workers=4),
            neural_baseline=dict(training_signature=dict(gradient_accumulation_steps=1))))
    report = dict(actual, cuda_free_bytes=44*GIB, cuda_allocated_bytes=GIB)
    host = dict(cpu_capacity=16, rss_bytes=GIB, available_memory_bytes=300*GIB)
    return receipt, report, host, actual, expected


class AllocationEntryUnits(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='half_B_allocation_UNIT_', dir=ROOT/'work')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.receipt, self.report, self.host, self.actual, self.expected = fixture(self.root)
        self.pipeline = SimpleNamespace(
            _calibration_resource_fingerprint=lambda report, *, device: {
                key:deepcopy(report[key]) for key in self.actual},
            _print_report=Mock())
        bind_execution_resources(self.pipeline, self.receipt)

    def run_checked(self):
        with patch('hiercp.preparation_runtime.snapshot', return_value=deepcopy(self.host)):
            return self.pipeline._calibration_resource_fingerprint(self.report, device='cuda')

    def test_migration_returns_actual_native_fingerprint_and_records_both(self):
        self.assertEqual(self.run_checked(), self.actual)
        self.assertNotEqual(self.actual, self.expected)
        lock = json.loads((self.root/'execution_lock.json').read_text())
        self.assertEqual(lock['resource_fingerprint'], self.actual)
        self.assertEqual(lock['baseline_resource_fingerprint'], self.expected)
        self.assertEqual((lock['physical_batch'],lock['workers'],lock['effective_batch']), (2,4,2))
        self.assertTrue(lock['allocation_changed_from_baseline'])
        self.assertFalse(lock['newly_calibrated'])
        self.assertFalse(lock['identical_throughput_measurement_claim'])
        path, = (self.root/'allocations').glob('admission_*.json')
        admission = json.loads(path.read_text())
        self.assertTrue(admission['accepted'])
        self.assertEqual(admission['half_B_contract_sha256'], self.receipt['contract_sha256'])
        self.assertEqual(set(admission['changed_fields']), {'platform','cuda_visible_devices_env'})
        self.pipeline._print_report.assert_called_once()

    def test_resume_same_allocation_tolerates_volatile_capacity_and_preserves_lock(self):
        self.run_checked()
        path = self.root/'execution_lock.json'; before = path.read_bytes()
        self.report['cuda_free_bytes'] -= GIB
        self.host['available_memory_bytes'] -= GIB
        self.run_checked()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(list((self.root/'allocations').glob('admission_*.json'))), 2)

    def test_started_allocation_lock_cannot_silently_change(self):
        self.run_checked()
        path = self.root/'execution_lock.json'; before = path.read_bytes()
        self.report['cuda_visible_devices_env'] = 'UNIT_third_uuid'
        with self.assertRaisesRegex(ValueError, 'execution lock changed'):
            self.run_checked()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(list((self.root/'allocations').glob('admission_*.json'))), 1)

    def test_insufficient_current_capacity_creates_no_execution_lock(self):
        self.report['cuda_free_bytes'] = GIB
        with self.assertRaises((MemoryError,ValueError)):
            self.run_checked()
        self.assertFalse((self.root/'execution_lock.json').exists())
        self.assertFalse((self.root/'allocations').exists())


class AllocationRecipeUnits(UnitFixture):
    def test_current_policy_is_explicit_bound_and_changes_no_training_config(self):
        _, original = self.initialized()
        root, current = half_b_training.initialize(self.baseline, self.directory/'UNIT_current_B',
                                                  3, 40., 192., 'current_allocation')
        self.assertEqual(current['config'], original['config'])
        self.assertEqual(current['model_contract'], original['model_contract'])
        self.assertEqual(current['baseline_proof'], original['baseline_proof'])
        self.assertEqual(current['allocation_policy'], 'current_allocation')
        self.assertEqual(half_b_training.verify(root), current)
        self.assertNotEqual(current['contract_sha256'], original['contract_sha256'])

    def test_policy_cannot_silently_change_an_existing_experiment(self):
        root, _ = self.initialized()
        before = (root/'manifest.json').read_bytes()
        with self.assertRaises(ValueError):
            half_b_training.initialize(self.baseline, root, 3, 40., 192., 'current_allocation')
        self.assertEqual((root/'manifest.json').read_bytes(), before)

    def test_invalid_policy_cannot_create_experiment(self):
        path = self.directory/'UNIT_invalid_B'
        with self.assertRaises(ValueError):
            half_b_training.initialize(self.baseline, path, 3, 40., 192., 'ignore_resources')
        self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
