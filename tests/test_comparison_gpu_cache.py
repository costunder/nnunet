"""CPU UNIT persistence/failure checks; mock probes are not CUDA evidence."""
from __future__ import annotations

import copy
import errno
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import comparison_gpu_cache as cache
from hiercp_v1x import comparison_gpu_runtime as runtime
from tests.test_comparison_gpu_runtime import ORIGINAL, SELECTED, unit_workload


def unit_context():
    return dict(identity_sha256='1' * 64, implementation_sha256='2' * 64,
                model_structure_sha256='3' * 64, loss_identity_sha256='4' * 64,
                torch_version='UNIT_TORCH', precision={'autocast_dtype': 'UNIT_FP16'},
                cuda_environment=dict(device_uuid='UNIT_GPU', device_name='UNIT_DEVICE',
                    total_memory_bytes=48 * 2**30, compute_capability='8.6', cuda_runtime='UNIT_CUDA'))


def unit_receipt():
    return dict(format='comparison_gpu_execution_policy_v1', calibration_status='measured',
                amp=True, declared_cuda_limit_bytes=40 * 2**30,
                optimizer_updates=0, model_gradients_rng_modes_and_settings_restored=True,
                optimizer_state_untouched=True, original_source_batch_and_candidate_coverage_unchanged=True,
                warmup_repetitions=1, measured_repetitions=2,
                original_policy=copy.deepcopy(ORIGINAL), selected_policy=copy.deepcopy(SELECTED),
                reports=[dict(accepted=True, policy=copy.deepcopy(SELECTED),
                              forward_backward_seconds=1.0, peak_cuda_bytes=2**30,
                              trials=[dict(equivalence=dict(accepted=True)) for _ in range(3)])],
                unit_only=True)


class DurableGpuPolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix='UNIT_durable_gpu_', dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.probe = patch.object(runtime.policy, 'calibrate_execution_policy',
            return_value=(copy.deepcopy(SELECTED), unit_receipt())).start()
        self.addCleanup(patch.stopall)

    def controller(self, context=None, **changes):
        arguments = dict(amp=True, cuda_limit_bytes=40 * 2**30, output=self.output,
                         cache_context=unit_context() if context is None else context, notify=lambda text: None)
        arguments.update(changes)
        return runtime.ComparisonGpuRuntime(SimpleNamespace(local_encoder=SimpleNamespace(**ORIGINAL)),
                                            lambda result: result, **arguments)

    def test_compatible_restart_reuses_envelope_without_forward_or_optimizer(self):
        first = self.controller()
        first.ensure(object(), unit_workload())
        opaque_checkpoint = self.output / 'checkpoint_latest.pt'
        opaque_checkpoint.write_bytes(b'UNIT opaque original checkpoint remains untouched')
        before = opaque_checkpoint.read_bytes()
        second = self.controller()
        self.assertEqual(second.current_settings(), ORIGINAL)
        self.assertFalse(second.path.exists())
        second.ensure(object(), unit_workload(80))
        self.probe.assert_called_once()
        self.assertEqual(second.current_settings(), SELECTED)
        self.assertEqual(opaque_checkpoint.read_bytes(), before)
        self.assertNotEqual(first.path, second.path)
        events = [json.loads(line) for line in second.path.read_text().splitlines()]
        self.assertEqual(events[0]['event'], 'context_bound_policy_cache_loaded')

    def test_context_changes_use_separate_cache_and_never_assume_compatibility(self):
        first = self.controller(); first.ensure(object(), unit_workload())
        original_bytes = first._cache.path.read_bytes()
        for key in ('identity_sha256', 'implementation_sha256', 'model_structure_sha256', 'loss_identity_sha256'):
            changed = unit_context(); changed[key] = 'f' * 64
            with self.subTest(key=key):
                second = self.controller(changed)
                self.assertEqual(second._measured, [])
                self.assertNotEqual(second._cache.path, first._cache.path)
        variants = []
        changed = unit_context(); changed['cuda_environment']['device_uuid'] = 'UNIT_OTHER'; variants.append(changed)
        changed = unit_context(); changed['cuda_environment']['total_memory_bytes'] += 1; variants.append(changed)
        changed = unit_context(); changed['precision']['autocast_dtype'] = 'UNIT_FP32'; variants.append(changed)
        changed = unit_context(); changed['torch_version'] = 'UNIT_NEW'; variants.append(changed)
        for context in variants:
            self.assertEqual(self.controller(context)._measured, [])
        for options in ({'cuda_limit_bytes': 39 * 2**30}, {'reserve_bytes': 2**30}, {'amp': False}):
            self.assertEqual(self.controller(**options)._measured, [])
        self.assertEqual(first._cache.path.read_bytes(), original_bytes)

    def test_legacy_truncated_audit_is_preserved_and_never_imported(self):
        historical = self.output / 'gpu_execution.jsonl'
        original = b'{"event":"calibration_completed","selected":'
        historical.write_bytes(original)
        controller = self.controller(); controller.ensure(object(), unit_workload())
        self.probe.assert_called_once()
        self.assertEqual(historical.read_bytes(), original)
        self.assertNotEqual(controller.path, historical)

    def test_oom_quarantine_survives_resume_but_does_not_drop_source(self):
        first = self.controller(); first.ensure(object(), unit_workload())
        self.assertTrue(first.reject_optimized_oom(unit_workload(), RuntimeError('UNIT injected OOM')))
        second = self.controller()
        second.ensure(object(), unit_workload())
        self.assertEqual(second.current_settings(), ORIGINAL)
        self.probe.assert_called_once()
        self.assertFalse(second.reject_optimized_oom(unit_workload(), RuntimeError('UNIT original OOM')))
        second.ensure(object(), unit_workload(80))
        self.assertEqual(second.current_settings(), SELECTED)

    def test_corrupt_cache_fails_clearly_and_preserves_existing_bytes(self):
        first = self.controller(); first.ensure(object(), unit_workload())
        first._cache.path.write_bytes(b'{"format":')
        with self.assertRaisesRegex(cache.PolicyCacheError, 'Invalid GPU execution sidecar'):
            self.controller()
        self.assertEqual(first._cache.path.read_bytes(), b'{"format":')

    def test_recomputed_checksum_does_not_hide_invalid_shapes_or_policy(self):
        first = self.controller(); first.ensure(object(), unit_workload())
        saved = json.loads(first._cache.path.read_text())
        for invalid in ('shape', 'policy', 'unmeasured_chunk', 'proof', 'duplicate'):
            payload = copy.deepcopy(saved)
            row = payload['measured'][0]
            if invalid == 'shape': row['inventory']['source_patch_shape.elements'] += 1
            elif invalid == 'policy': row['selected']['dense_batch_size'] = True
            elif invalid == 'unmeasured_chunk': row['selected']['dense_batch_size'] = 12345
            elif invalid == 'proof': row['proof']['optimizer_updates'] = 1
            else: payload['measured'].append(copy.deepcopy(row))
            unsigned = dict(payload); unsigned.pop('content_sha256')
            payload['content_sha256'] = cache.digest(unsigned)
            first._cache.path.write_text(json.dumps(payload), encoding='utf8')
            with self.subTest(invalid=invalid), self.assertRaises(cache.PolicyCacheError): self.controller()

    def test_incomplete_or_nonfinite_context_never_enables_reuse(self):
        variants = [dict(identity_sha256='1' * 64)]
        invalid = unit_context(); invalid['precision']['bad'] = float('nan'); variants.append(invalid)
        invalid = unit_context(); invalid['cuda_environment']['total_memory_bytes'] = True; variants.append(invalid)
        for context in variants:
            with self.subTest(context=context), self.assertRaises(cache.PolicyCacheError): self.controller(context)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_failed_atomic_publication_keeps_previous_cache_and_original_execution(self):
        controller = self.controller(); controller.ensure(object(), unit_workload())
        before = controller._cache.path.read_bytes()
        quota = OSError(getattr(errno, 'EDQUOT', 122), 'UNIT injected quota')
        with patch.object(cache.os, 'replace', side_effect=quota):
            with self.assertRaisesRegex(OSError, 'EDQUOT'):
                controller.ensure(object(), unit_workload(200))
        self.assertEqual(controller._cache.path.read_bytes(), before)
        self.assertEqual(len(controller._measured), 1)
        self.assertEqual(controller.current_settings(), ORIGINAL)
        self.assertEqual(list(controller._cache.path.parent.glob('*.lock')), [])
        self.assertEqual(list(controller._cache.path.parent.glob('*.tmp')), [])

    def test_concurrent_cache_writer_is_rejected_without_lost_updates(self):
        first = self.controller(); second = self.controller()
        first.ensure(object(), unit_workload())
        before = first._cache.path.read_bytes()
        with self.assertRaisesRegex(cache.PolicyCacheError, 'Concurrent'):
            second.ensure(object(), unit_workload(200))
        self.assertEqual(first._cache.path.read_bytes(), before)

    def test_partial_new_audit_is_not_reappended_after_error(self):
        controller = self.controller()
        quota = OSError(getattr(errno, 'EDQUOT', 122), 'UNIT fsync quota')
        with patch.object(runtime.os, 'fsync', side_effect=quota):
            with self.assertRaises(OSError): controller.ensure(object(), unit_workload())
        before = controller.path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'refusing to append'):
            controller.ensure(object(), unit_workload())
        self.assertEqual(controller.path.read_bytes(), before)
        self.probe.assert_not_called()

    def test_incomplete_calibration_cannot_become_durable_evidence(self):
        controller = self.controller()
        receipt = unit_receipt()
        receipt['reports'][0]['trials'].pop()
        self.probe.return_value = (copy.deepcopy(SELECTED), receipt)
        with self.assertRaisesRegex(cache.PolicyCacheError, 'trial evidence'):
            controller.ensure(object(), unit_workload())
        self.assertEqual(controller.current_settings(), ORIGINAL)
        self.assertEqual(controller._measured, [])
        self.assertFalse(controller._cache.path.exists())

    def test_original_no_headroom_remains_valid_and_survives_resume(self):
        receipt = unit_receipt()
        receipt.update(calibration_status='original_retained_no_headroom', selected_policy=copy.deepcopy(ORIGINAL))
        receipt['reports'][0]['policy'] = copy.deepcopy(ORIGINAL)
        self.probe.return_value = (copy.deepcopy(ORIGINAL), receipt)
        first = self.controller(); first.ensure(object(), unit_workload())
        self.assertEqual(first.current_settings(), ORIGINAL)
        self.assertTrue(first._cache.path.exists())
        second = self.controller(); second.ensure(object(), unit_workload())
        self.probe.assert_called_once()
        self.assertEqual(second.current_settings(), ORIGINAL)

    def test_incomplete_no_headroom_proof_keeps_original_without_persisting(self):
        receipt = unit_receipt()
        receipt.update(calibration_status='original_retained_no_headroom', selected_policy=copy.deepcopy(ORIGINAL))
        receipt['reports'][0]['policy'] = copy.deepcopy(ORIGINAL)
        receipt['reports'][0]['trials'].pop()
        self.probe.return_value = (copy.deepcopy(ORIGINAL), receipt)
        controller = self.controller()
        controller.ensure(object(), unit_workload())
        controller.ensure(object(), unit_workload())
        self.probe.assert_called_once()
        self.assertEqual(controller.current_settings(), ORIGINAL)
        self.assertFalse(controller._cache.path.exists())
        rows = [json.loads(line) for line in controller.path.read_text().splitlines()]
        self.assertIn('original_policy_cache_admission_skipped', [row['event'] for row in rows])

    def test_no_headroom_status_cannot_admit_an_optimized_policy(self):
        receipt = unit_receipt(); receipt['calibration_status'] = 'original_retained_no_headroom'
        self.probe.return_value = (copy.deepcopy(SELECTED), receipt)
        controller = self.controller()
        with self.assertRaises(cache.PolicyCacheError): controller.ensure(object(), unit_workload())
        self.assertEqual(controller.current_settings(), ORIGINAL)
        self.assertFalse(controller._cache.path.exists())


if __name__ == '__main__': unittest.main()
