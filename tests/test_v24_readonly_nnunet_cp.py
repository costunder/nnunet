"""CPU-only DEBUG entry/dispatch tests; no real model training is claimed."""
from __future__ import annotations

from pathlib import Path
import io
import json
import os
import tempfile
from types import FunctionType, SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_v24_readonly_nnunet_cp as entry


class ReadonlyEntryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='DEBUG_readonly_entry_', dir=entry.ROOT / 'outputs')
        self.root = Path(self.temporary.name)
        self.admission = self.root / 'admission.json'; self.admission.write_text('{}', encoding='utf8')
        self.native = self.root / 'native.json'
        self.native.write_text('{"format":"v24_frozen_current_GT_blind_GNN_native_nnunet_CP_v1","physical_GPU":4}', encoding='utf8')
        self.runtime = SimpleNamespace(WORKER_COMMAND='exact original calibration entry', TRAIN_COMMAND='exact original crop entry')
        self.common = dict(executable='DEBUG_python', native_path=self.native, admission_path=self.admission,
            runtime=self.runtime, trainer='original_full250_trainer', plans='original_plans')

    def tearDown(self):
        self.temporary.cleanup()

    def test_public_profile_requires_gpu4_full_current_action_and_admission(self):
        args = entry.parse(['train', '--native', str(self.native), '--gpu', '4', '--storage-admission', str(self.admission)])
        self.assertEqual(args.storage_admission, self.admission)
        self.assertEqual(args.gpu, 4); self.assertFalse(args.resume)
        for extra in (['--gpu', '5'], ['--resume'], ['--physical-batch', '1']):
            with self.subTest(extra=extra), patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                entry.parse(['train', '--native', str(self.native), '--gpu', '4',
                    '--storage-admission', str(self.admission), *extra])
        with patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
            entry.parse(['train', '--native', str(self.native), '--gpu', '4'])

    def test_redirect_installs_adapter_in_both_original_physical_clone_workers(self):
        for batch in ('2', '4'):
            original = ['DEBUG_python', '-B', '-c', self.runtime.WORKER_COMMAND, str(self.native), batch, str(self.root / ('batch_' + batch))]
            actual = entry.redirect(original, **self.common)
            args = entry.parse(actual[4:])
            self.assertEqual(args.readonly_worker, 'calibrate'); self.assertEqual(args.physical_batch, int(batch))
            self.assertEqual(args.native, self.native); self.assertEqual(args.worker_output, self.root / ('batch_' + batch))
            self.assertEqual(original[3], self.runtime.WORKER_COMMAND)

    def test_redirect_preserves_exact_full_native_cli_tail(self):
        tail = ['730', '3d_fullres', '0', '-tr', 'original_full250_trainer', '-p', 'original_plans']
        original = ['DEBUG_python', '-B', '-c', self.runtime.TRAIN_COMMAND, *tail]
        actual = entry.redirect(original, **self.common)
        args = entry.parse(actual[4:])
        self.assertEqual(args.readonly_worker, 'train'); self.assertEqual(args.training_args, tail)
        self.assertEqual(actual[-len(tail):], original[-len(tail):])

    def test_unknown_worker_smaller_batch_resume_or_other_dataset_cannot_be_redirected(self):
        commands = [
            ['DEBUG_python', '-B', '-c', 'unknown entry', str(self.native), '2', str(self.root)],
            ['DEBUG_python', '-B', '-c', self.runtime.WORKER_COMMAND, str(self.native), '1', str(self.root)],
            ['DEBUG_python', '-B', '-c', self.runtime.TRAIN_COMMAND, '730', '3d_fullres', '0', '-tr', 'original_full250_trainer', '-p', 'original_plans', '--c'],
            ['DEBUG_python', '-B', '-c', self.runtime.TRAIN_COMMAND, '731', '3d_fullres', '0', '-tr', 'original_full250_trainer', '-p', 'original_plans'],
        ]
        for command in commands:
            with self.subTest(command=command), self.assertRaises(ValueError):
                entry.redirect(command, **self.common)

    def test_storage_scope_keeps_materializer_code_and_restores_after_error(self):
        def materialize():return 'DEBUG original source code identity'
        pipeline = SimpleNamespace(_materialize_bank=materialize, require_project_budget=lambda: {'full_budget': True}, subprocess=object())
        original_budget, original_process = pipeline.require_project_budget, pipeline.subprocess
        def install(target, admission):
            target._materialize_bank = FunctionType(materialize.__code__, materialize.__globals__, materialize.__name__)
        storage = SimpleNamespace(install_pipeline_adapter=install)
        with patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 2**30)), \
                self.assertRaisesRegex(RuntimeError, 'DEBUG failure'):
            with entry.runtime_scope(pipeline, storage, self.admission, lambda: None, write_root=self.root, reserve_bytes=10 * 2**30):
                self.assertIs(pipeline._materialize_bank.__code__, materialize.__code__)
                self.assertEqual(pipeline.require_project_budget(), {'full_budget': True})
                raise RuntimeError('DEBUG failure')
        self.assertIs(pipeline._materialize_bank, materialize)
        self.assertIs(pipeline.require_project_budget, original_budget); self.assertIs(pipeline.subprocess, original_process)

    def test_original_budget_calls_enforce_runtime_disk_floor_without_population_checks(self):
        def materialize():return None
        original_budget = unittest.mock.Mock(return_value={'full_budget': True})
        pipeline = SimpleNamespace(_materialize_bank=materialize, require_project_budget=original_budget, subprocess=object())
        storage = SimpleNamespace(install_pipeline_adapter=lambda *_: None)
        guard = unittest.mock.Mock()
        with patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=9 * 2**30)):
            with entry.runtime_scope(pipeline, storage, self.admission, guard, write_root=self.root, reserve_bytes=10 * 2**30):
                with self.assertRaisesRegex(OSError, '10GiB'):
                    pipeline.require_project_budget()
        guard.assert_called_once(); original_budget.assert_not_called()

    def test_unknown_storage_materializer_replacement_is_rejected_and_restored(self):
        def materialize():return None
        pipeline = SimpleNamespace(_materialize_bank=materialize, require_project_budget=lambda: None, subprocess=object())
        def different(target, admission):target._materialize_bank = lambda: 'changed scientific implementation'
        with self.assertRaisesRegex(ValueError, 'materialization code'):
            with entry.runtime_scope(pipeline, SimpleNamespace(install_pipeline_adapter=different), self.admission,
                    lambda: None, write_root=self.root, reserve_bytes=10 * 2**30):
                self.fail('Unknown scientific replacement entered scope')
        self.assertIs(pipeline._materialize_bank, materialize)

    def test_guard_detects_admission_byte_replacement(self):
        from hiercp_v1x import v24_readonly_native_storage as storage
        document = dict(format=storage.FORMAT, profile=entry.PROFILE, debug=False,
            model_data_scale_preserved=True, scores_reused=False, learned_upper_reused=False)
        self.admission.write_text(json.dumps(document), encoding='utf8')
        checksum = entry._sha(self.admission)
        with patch.object(storage, 'verify_admission'), patch.dict(os.environ, {entry.CHECKSUM_ENV: checksum}):
            admitted, actual, guard = entry.admit(self.admission)
            self.assertEqual(admitted, document); self.assertEqual(actual, checksum)
            guard()
            self.admission.write_text('{"modified":true}', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'storage/source file changed'):
                guard()
        with patch.object(storage, 'verify_admission'), patch.dict(os.environ, {entry.CHECKSUM_ENV: checksum}):
            with self.assertRaisesRegex(ValueError, 'watcher-pinned'):
                entry.admit(self.admission)


if __name__ == '__main__':
    unittest.main()
