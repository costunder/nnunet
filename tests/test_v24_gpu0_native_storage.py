"""CPU regression tests for execution-only output admission; no CUDA or server."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import v24_gpu0_native_storage as storage


class Root:
    def __init__(self, path, device):
        self.path, self.device = path, device
    def __str__(self):
        return self.path
    def stat(self):
        return SimpleNamespace(st_dev=self.device)


class SingleArmStorageTests(unittest.TestCase):
    def fixture(self):
        old = dict(baseline=dict(dataset_name='Dataset730', plans_name='Plans', configuration='3d_fullres'),
            budget=dict(new_writable_bytes_estimate=1000, measured_components={'full_native':1000}),
            aggregate_budget=dict(minimum_runtime_free_bytes=100, growth_margin_bytes=20,
                arms={'gpu4':{'root':'/old/gpu4'},'gpu5':{'root':'/old/gpu5'},'gpu6':{'root':'/old/gpu6'}}),
            raw_cases={'case':{'exact':'raw'}}, preprocessed={'cases':list(range(131))},
            split={'train':list(range(105)), 'val':list(range(26))},
            checkpoint_coordination={'old':True})
        binding = dict(execution_root='/shm/new', metadata_root='/tmp/new', arm_root='/shm/new/gpu4',
            arm_witness=dict(root='/shm/new/gpu4', existing_parent='/shm/new', filesystem_device=2),
            initial_arm_allocated_bytes=0, initial_metadata_allocated_bytes=0,
            initial_execution_free_bytes=10**15, created_at=123.0,
            lock_file={'path':'/shm/new/native_final_publication.lock'}, adapter_source={'path':'adapter'},
            persistent_archive_bytes=500, canonical_tmpfs_path='/shm', canonical_tmpfs_reserve_bytes=300)
        return old, binding

    def test_only_output_budget_and_coordination_are_rebound(self):
        old, binding = self.fixture(); before = copy.deepcopy(old)
        document = storage._derive(old, binding)
        self.assertEqual(old, before)
        for key in ('raw_cases','preprocessed','split','baseline','budget'):
            self.assertEqual(document[key], old[key])
        self.assertEqual(set(document['aggregate_budget']['arms']), {'gpu4'})
        self.assertEqual(document['aggregate_budget']['required_free_bytes'], 1100)
        self.assertEqual(document['checkpoint_coordination']['global_peak_checkpoint_slots'], 3)
        self.assertEqual(set(document['checkpoint_coordination']['folds']), {'gpu4'})

    def test_changed_scientific_or_input_payload_is_rejected(self):
        old, binding = self.fixture()
        document = storage._derive(old, binding)
        document['raw_cases']['case']['exact'] = 'changed'
        with patch.object(storage, '_old', return_value=({}, old)), patch.object(storage, '_validate_binding'):
            with self.assertRaisesRegex(ValueError, 'immutable science'):
                storage.verify_admission(document)

    def test_old_admission_uses_stat_checks_never_full_volume_rehash(self):
        core, static, matched = Mock(), Mock(), Mock()
        old_original = {'path':'/old/original','sha256':'x'}
        matched.verify_admission.return_value = {'original_readonly_admission':old_original}
        original_verify = Mock(return_value={'full':True})
        binding = {'old_matched_admission':{'path':'/old/matched'}, 'old_original_admission':old_original}
        with patch.object(storage, '_modules', return_value=(core, static, matched)), \
             patch.object(storage, '_ORIGINALS', {'verify':original_verify}):
            storage._old(binding)
        matched.verify_admission.assert_called_once_with('/old/matched', full_hash=False)
        original_verify.assert_called_once_with('/old/original', full_hash=False)
        self.assertEqual(core.guard.call_count, 2)  # Small JSON admissions are fully hashed.

    def roots(self):
        roots = {'/shm/new/gpu4':Root('/shm/new/gpu4',2), '/tmp/new':Root('/tmp/new',1),
                 '/shm':Root('/shm',2)}
        return lambda path: roots[str(path)]

    def test_cross_version_future_bytes_charged_only_to_matching_filesystem(self):
        old, binding = self.fixture(); document = storage._derive(old, binding)
        core, static, matched = Mock(), Mock(), Mock()
        static._allocated_bytes.return_value = 0
        cross = dict(filesystem_device=1, required_free_bytes=700, remaining_output_bytes=600)
        with patch.object(storage, '_modules', return_value=(core, static, matched)), \
             patch.object(storage, '_root', side_effect=self.roots()), \
             patch.object(storage, '_canonical_root', side_effect=self.roots()), \
             patch.object(storage, '_v25_reservation', return_value=cross), \
             patch.object(storage.shutil, 'disk_usage', return_value=SimpleNamespace(free=5000)):
            result = storage._disk_receipt(document)
        self.assertEqual(result['filesystems']['2']['required_free_bytes'], 1400)
        self.assertEqual(result['filesystems']['1']['required_free_bytes'], 1200)
        self.assertFalse(result['immutable_input_files_scanned_for_allocation'])

    def test_same_filesystem_sums_native_archive_canonical_and_v25(self):
        old, binding = self.fixture(); document = storage._derive(old, binding)
        core, static, matched = Mock(), Mock(), Mock(); static._allocated_bytes.return_value = 0
        with patch.object(storage, '_modules', return_value=(core, static, matched)), \
             patch.object(storage, '_root', side_effect=lambda p:Root(str(p),1)), \
             patch.object(storage, '_canonical_root', side_effect=lambda p:Root(str(p),1)), \
             patch.object(storage, '_v25_reservation', return_value={'filesystem_device':1,'required_free_bytes':700}), \
             patch.object(storage.shutil, 'disk_usage', return_value=SimpleNamespace(free=2599)):
            with self.assertRaisesRegex(OSError, 'reservations do not fit'):
                storage._disk_receipt(document)

    def test_actual_output_bytes_do_not_double_charge_completed_writes(self):
        old, binding = self.fixture(); document = storage._derive(old, binding)
        core, static, matched = Mock(), Mock(), Mock()
        static._allocated_bytes.side_effect = [400,100]
        with patch.object(storage, '_modules', return_value=(core, static, matched)), \
             patch.object(storage, '_root', side_effect=self.roots()), \
             patch.object(storage, '_canonical_root', side_effect=self.roots()), \
             patch.object(storage, '_v25_reservation', return_value={'filesystem_device':1,'required_free_bytes':700}), \
             patch.object(storage.shutil, 'disk_usage', return_value=SimpleNamespace(free=5000)):
            result = storage._disk_receipt(document)
        self.assertEqual(result['own_native_remaining_bytes'], 600)
        self.assertEqual(result['persistent_archive_remaining_bytes'], 400)

    def test_over_provision_native_output_fails_without_reducing_data(self):
        old, binding = self.fixture(); document = storage._derive(old, binding)
        core, static, matched = Mock(), Mock(), Mock(); static._allocated_bytes.return_value = 1001
        with patch.object(storage, '_modules', return_value=(core, static, matched)), \
             patch.object(storage, '_root', side_effect=self.roots()):
            with self.assertRaisesRegex(OSError, 'unchanged measured full provision'):
                storage._disk_receipt(document)

    def test_entire_actual_archive_above_estimate_is_readmitted_not_truncated(self):
        old, binding = self.fixture()
        binding['metadata_root'] = str(Path('/tmp/new').absolute())
        binding['arm_root'] = str(Path('/shm/new/gpu4').absolute())
        original = storage._derive(old, binding)
        core, static, matched = Mock(), Mock(), Mock()
        matched.verify_admission.return_value = {'original_readonly_admission':{'path':'proof'}}
        core.guard.return_value = 'original'
        request = dict(storage_admission='/tmp/new/storage_admission.json', metadata_root=binding['metadata_root'],
            execution_root=binding['arm_root'])
        cross = dict(filesystem_device=1,required_free_bytes=700)
        with patch.object(storage, '_modules', return_value=(core, static, matched)), \
             patch.object(storage, 'verify_admission', return_value=original), \
             patch.object(storage, '_root', side_effect=lambda p:Path(p)), \
             patch.object(Path, 'stat', return_value=SimpleNamespace(st_dev=1)), \
             patch.object(storage, '_v25_reservation', return_value=cross), \
             patch.object(storage.shutil, 'disk_usage', return_value=SimpleNamespace(free=1600)):
            result = storage.check_persistent_archive(request, 900)
        self.assertEqual(result['actual_archive_bytes'], 900)
        self.assertEqual(result['required_free_bytes'], 1600)
        self.assertFalse(result['archive_truncated'])

    def test_execution_two_field_admission_is_captured_as_full_proof_before_guard(self):
        old, binding = self.fixture()
        binding['metadata_root'] = str(Path('/tmp/new').absolute())
        binding['arm_root'] = str(Path('/shm/new/gpu4').absolute())
        original = storage._derive(old, binding)
        core, static, matched = Mock(), Mock(), Mock()
        matched.verify_admission.return_value = {'original_readonly_admission':{'path':'original'}}
        core.regular.return_value = '/tmp/new/storage_admission.json'
        full = dict(path='/tmp/new/storage_admission.json',sha256='sealed',stat=[1,2,3,4,5],bytes=3)
        core.proof.return_value = full
        core.guard.return_value = '/tmp/new/storage_admission.json'
        request = dict(storage_admission={'path':full['path'],'sha256':'sealed'},
            metadata_root=binding['metadata_root'], execution_root=binding['arm_root'])
        with patch.object(storage, '_modules', return_value=(core, static, matched)), \
             patch.object(storage, 'verify_admission', return_value=original), \
             patch.object(storage, '_root', side_effect=lambda p:Path(p)), \
             patch.object(Path, 'stat', return_value=SimpleNamespace(st_dev=1)), \
             patch.object(storage, '_v25_reservation', return_value={'filesystem_device':1,'required_free_bytes':700}), \
             patch.object(storage.shutil, 'disk_usage', return_value=SimpleNamespace(free=1600)):
            storage.check_persistent_archive(request,900)
            core.guard.assert_any_call(full, full_hash=True)
            core.proof.return_value = dict(full, sha256='changed')
            with self.assertRaisesRegex(ValueError, 'storage admission SHA changed'):
                storage.check_persistent_archive(request,900)
            request['storage_admission']['extra'] = True
            with self.assertRaisesRegex(ValueError, 'Exact path/SHA'):
                storage.check_persistent_archive(request,900)


if __name__ == '__main__':
    unittest.main()
