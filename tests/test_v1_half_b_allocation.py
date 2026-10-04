"""Pure UNIT resource metadata checks; no torch, GPU, CT or training execution."""
from copy import deepcopy
import json
import unittest

from hiercp_v1x.half_b_allocation import validate_allocation


GIB = 2**30


def unit_resources():
    gpu = dict(logical_index=0, name='UNIT NVIDIA RTX A6000',
               total_vram_bytes=48 * GIB, multiprocessors=84,
               compute_capability='8.6', mig_instance=False)
    fingerprint = dict(selected_device='cuda', cuda_available=True,
        cuda_visible_device_count=1, gpu_devices=[gpu], platform='UNIT baseline kernel',
        cuda_visible_devices_env='GPU-UNIT-baseline', nvidia_visible_devices_env='<unset>',
        cpu_logical_cores=128, cpu_affinity_cores=128, ram_total_bytes=512 * GIB,
        cgroup_memory_limit_bytes='unlimited', cgroup_cpu_limit_cores='unlimited',
        cgroup_cpuset='0-127', scheduler_allocation={'SLURM_JOB_ID':'UNIT-old'},
        container_hint='unavailable', mig_visibility='not identifiable from PyTorch/environment')
    receipt = dict(cuda_gib=40., rss_gib=192.,
        config={'training':dict(batch_size=16, num_workers=8, gradient_accumulation_steps=2)},
        baseline_proof=dict(execution=dict(physical_batch=16, workers=8,
            resource_fingerprint=deepcopy(fingerprint)),
            neural_baseline={'training_signature':{'gradient_accumulation_steps':2}}))
    report = dict(cuda_free_bytes=44 * GIB, cuda_allocated_bytes=2 * GIB)
    host = dict(cpu_capacity=32, rss_bytes=4 * GIB, available_memory_bytes=220 * GIB,
                cgroup={'UNIT':'already resolved effective limits'})
    return report, fingerprint, deepcopy(fingerprint), receipt, host


class HalfBAllocationUnits(unittest.TestCase):
    def setUp(self):
        self.report, self.expected, self.actual, self.receipt, self.host = unit_resources()

    def admit(self):
        return validate_allocation(self.report, self.expected, self.actual,
                                   self.receipt, self.host)

    def migrated(self):
        self.receipt['allocation_policy'] = 'current_allocation'
        self.actual.update(platform='UNIT current kernel', cuda_visible_devices_env='GPU-UNIT-new',
            cpu_logical_cores=96, cpu_affinity_cores=64, cgroup_cpuset='32-95',
            scheduler_allocation={'SLURM_JOB_ID':'UNIT-new','SLURM_JOB_GPUS':'3'},
            ram_total_bytes=256 * GIB)

    def test_default_same_allocation_retains_exact_guard_and_reports_changes(self):
        self.actual['cuda_visible_devices_env'] = 'GPU-UNIT-new'
        with self.assertRaisesRegex(ValueError, 'changed_fields=.*cuda_visible_devices_env'):
            self.admit()

    def test_same_allocation_admission_preserves_all_controls(self):
        value = self.admit()
        self.assertEqual(value['allocation_policy'], 'same_allocation')
        self.assertEqual(value['changed_fields'], {})
        self.assertEqual(value['preserved_execution'], dict(physical_batch=16, workers=8,
            gradient_accumulation_steps=2, data_parallel_workers=1, effective_batch=32))
        self.assertFalse(value['matched_performance_claim'])
        self.assertFalse(value['new_calibration'])

    def test_explicit_migration_accepts_identity_and_cpu_shape_changes_with_capacity(self):
        self.migrated()
        self.actual['gpu_devices'][0]['logical_index'] = 7
        value = self.admit()
        self.assertTrue(value['accepted'])
        self.assertEqual(value['actual_resource_fingerprint'], self.actual)
        self.assertEqual(value['baseline_resource_fingerprint'], self.expected)
        self.assertEqual(value['changed_fields']['cuda_visible_devices_env'],
                         dict(old='GPU-UNIT-baseline', new='GPU-UNIT-new'))
        self.assertIn('gpu_devices', value['changed_fields'])

    def test_return_is_json_safe_independent_and_validator_does_not_mutate_inputs(self):
        self.migrated()
        originals = deepcopy((self.report, self.expected, self.actual, self.receipt, self.host))
        value = self.admit()
        json.dumps(value, allow_nan=False)
        value['current_resources']['cuda_free_bytes'] = 0
        value['actual_resource_fingerprint']['gpu_devices'][0]['name'] = 'changed return'
        self.assertEqual((self.report, self.expected, self.actual, self.receipt, self.host), originals)

    def test_unknown_policy_rejected(self):
        for policy in ('automatic', '', None, True, 1):
            with self.subTest(policy=policy):
                self.receipt['allocation_policy'] = policy
                with self.assertRaisesRegex(ValueError, 'allocation policy'):
                    self.admit()

    def test_cuda_availability_requires_true_boolean(self):
        self.migrated()
        for value in (False, 1, 'true', None):
            with self.subTest(value=value):
                self.actual['cuda_available'] = value
                with self.assertRaises(ValueError):
                    self.admit()

    def test_cpu_or_other_selected_devices_rejected(self):
        self.migrated()
        for device in ('cpu', 'cuda:1', None, True):
            with self.subTest(device=device):
                self.actual['selected_device'] = device
                with self.assertRaisesRegex(ValueError, 'select CUDA'):
                    self.admit()

    def test_multiple_or_unknown_visible_devices_rejected(self):
        self.migrated()
        for count in (0, 2, True, 1., '1', None):
            with self.subTest(count=count):
                self.actual['cuda_visible_device_count'] = count
                with self.assertRaises(ValueError):
                    self.admit()
        self.actual['cuda_visible_device_count'] = 1
        for devices in ([], [self.expected['gpu_devices'][0]] * 2, None):
            with self.subTest(devices=devices):
                self.actual['gpu_devices'] = devices
                with self.assertRaises(ValueError):
                    self.admit()

    def test_every_stable_gpu_hardware_field_must_match(self):
        self.migrated()
        gpu = deepcopy(self.actual['gpu_devices'][0])
        differences = dict(name='UNIT other GPU', total_vram_bytes=47 * GIB,
                           multiprocessors=83, compute_capability='8.0', mig_instance=True)
        for key, value in differences.items():
            with self.subTest(field=key):
                self.actual['gpu_devices'][0] = dict(gpu, **{key:value})
                with self.assertRaisesRegex(ValueError, 'hardware differs'):
                    self.admit()

    def test_unknown_or_malformed_gpu_hardware_rejected(self):
        self.migrated()
        gpu = deepcopy(self.actual['gpu_devices'][0])
        for key, value in [('name','unavailable (query failed)'),('multiprocessors',True),
                ('total_vram_bytes','unavailable'),('compute_capability','unknown'),
                ('mig_instance',0)]:
            with self.subTest(field=key):
                self.actual['gpu_devices'][0] = dict(gpu, **{key:value})
                with self.assertRaises(ValueError):
                    self.admit()

    def test_declared_budgets_require_finite_positive_numbers(self):
        for key in ('cuda_gib','rss_gib'):
            for value in (True, False, 0, -1, float('nan'), float('inf'), '40', None, 1e-20):
                with self.subTest(field=key, value=value):
                    receipt = deepcopy(self.receipt); receipt[key] = value
                    with self.assertRaises(ValueError):
                        validate_allocation(self.report, self.expected, self.actual, receipt, self.host)

    def test_declared_cuda_budget_cannot_exceed_hardware(self):
        self.receipt['cuda_gib'] = 49.
        with self.assertRaisesRegex(ValueError, 'exceeds selected GPU'):
            self.admit()

    def test_actual_cpu_capacity_cannot_reduce_preserved_workers(self):
        self.migrated()
        self.host['cpu_capacity'] = 7
        with self.assertRaisesRegex(ValueError, 'fixed worker count'):
            self.admit()

    def test_unknown_invalid_host_facts_rejected(self):
        for key, values in [('cpu_capacity',[None,True,8.,'unavailable',0]),
                ('rss_bytes',[None,True,1.,'unavailable',-1]),
                ('available_memory_bytes',[None,True,1.,'unavailable',-1])]:
            for value in values:
                with self.subTest(field=key, value=value):
                    host = dict(self.host, **{key:value})
                    with self.assertRaises(ValueError):
                        validate_allocation(self.report,self.expected,self.actual,self.receipt,host)

    def test_host_budget_uses_process_rss_plus_effective_available_memory(self):
        self.host.update(rss_bytes=4 * GIB, available_memory_bytes=188 * GIB)
        self.assertTrue(self.admit()['accepted'])
        self.host['available_memory_bytes'] -= 1
        with self.assertRaisesRegex(ValueError, 'insufficient memory'):
            self.admit()

    def test_process_rss_over_budget_rejected_even_with_available_memory(self):
        self.host['rss_bytes'] = 192 * GIB + 1
        with self.assertRaisesRegex(ValueError, 'process RSS exceeds'):
            self.admit()

    def test_gpu_budget_uses_free_plus_allocated_without_changing_budget(self):
        self.report.update(cuda_free_bytes=38 * GIB, cuda_allocated_bytes=2 * GIB)
        self.assertTrue(self.admit()['accepted'])
        self.report['cuda_free_bytes'] -= 1
        with self.assertRaisesRegex(ValueError, 'insufficient memory'):
            self.admit()

    def test_gpu_unknown_negative_boolean_or_inconsistent_memory_rejected(self):
        for key in ('cuda_free_bytes','cuda_allocated_bytes'):
            for value in (None, True, 1., 'unavailable', -1):
                with self.subTest(field=key, value=value):
                    report = dict(self.report, **{key:value})
                    with self.assertRaises(ValueError):
                        validate_allocation(report,self.expected,self.actual,self.receipt,self.host)
        self.report['cuda_free_bytes'] = 48 * GIB
        with self.assertRaisesRegex(ValueError, 'exceed identified GPU'):
            self.admit()

    def test_gpu_allocated_over_declared_budget_rejected(self):
        self.report.update(cuda_allocated_bytes=40 * GIB + 1, cuda_free_bytes=GIB)
        with self.assertRaisesRegex(ValueError, 'allocation exceeds'):
            self.admit()

    def test_configured_batch_workers_and_accumulation_cannot_change(self):
        for key, bad in [('batch_size',8),('num_workers',4),('gradient_accumulation_steps',1),
                ('batch_size',True),('num_workers',8.),('gradient_accumulation_steps','auto')]:
            with self.subTest(field=key, value=bad):
                receipt = deepcopy(self.receipt); receipt['config']['training'][key] = bad
                with self.assertRaises(ValueError):
                    validate_allocation(self.report,self.expected,self.actual,receipt,self.host)

    def test_missing_or_invalid_baseline_proof_cannot_supply_fallback_controls(self):
        for branch, key, value in [('execution','physical_batch',None),
                ('execution','workers',True),('signature','gradient_accumulation_steps',None),
                ('signature','gradient_accumulation_steps','auto')]:
            with self.subTest(field=key, value=value):
                receipt = deepcopy(self.receipt)
                target = receipt['baseline_proof']['execution'] if branch=='execution' else receipt['baseline_proof']['neural_baseline']['training_signature']
                target[key] = value
                with self.assertRaises(ValueError):
                    validate_allocation(self.report,self.expected,self.actual,receipt,self.host)

    def test_supplied_baseline_fingerprint_must_match_verified_proof(self):
        self.migrated()
        self.receipt['baseline_proof']['execution']['resource_fingerprint']['platform'] = 'UNIT altered proof'
        with self.assertRaisesRegex(ValueError, 'differs from verified baseline execution'):
            self.admit()

    def test_small_explicit_unit_budgets_supported_without_production_changes(self):
        self.receipt.update(cuda_gib=2., rss_gib=4.)
        self.report.update(cuda_free_bytes=2 * GIB, cuda_allocated_bytes=0)
        self.host.update(rss_bytes=GIB, available_memory_bytes=3 * GIB)
        self.assertTrue(self.admit()['accepted'])

    def test_zero_workers_still_requires_one_effective_cpu(self):
        self.receipt['baseline_proof']['execution']['workers'] = 0
        self.receipt['config']['training']['num_workers'] = 0
        self.host['cpu_capacity'] = 1
        self.assertTrue(self.admit()['accepted'])
        self.host['cpu_capacity'] = 0
        with self.assertRaises(ValueError):
            self.admit()

    def test_nonfinite_or_non_json_snapshot_rejected(self):
        for invalid in (float('nan'), object()):
            with self.subTest(invalid=invalid):
                self.host['extra_observation'] = invalid
                with self.assertRaisesRegex(ValueError, 'JSON-safe'):
                    self.admit()


if __name__ == '__main__':
    unittest.main()
