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

    def test_gpu56_public_stages_require_the_declared_static_extension(self):
        self.admission.write_text(json.dumps(dict(storage_extension=entry.EXTENSION)))
        for gpu in (5, 6):
            self.native.write_text(json.dumps(dict(format='v24_frozen_current_GT_blind_GNN_native_nnunet_CP_v1', physical_GPU=gpu)))
            args = entry.parse(['train', '--native', str(self.native), '--gpu', str(gpu),
                '--storage-admission', str(self.admission)])
            self.assertEqual(args.gpu, gpu); self.assertFalse(args.resume)
        self.admission.write_text('{}')
        with patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
            entry.parse(['train', '--native', str(self.native), '--gpu', '6', '--storage-admission', str(self.admission)])

    def test_static_extension_keeps_original_materializer_and_growth_floor(self):
        def materialize():return 'DEBUG full original CP operator equations'
        pipeline = SimpleNamespace(_materialize_bank=materialize, require_project_budget=lambda: 'full', subprocess=object())
        def install(target, base, admission):
            self.assertEqual(base, admission)
            target._materialize_bank = FunctionType(materialize.__code__, materialize.__globals__, materialize.__name__)
        extension = SimpleNamespace(install_pipeline_adapter=unittest.mock.Mock(side_effect=install))
        core = SimpleNamespace(install_pipeline_adapter=unittest.mock.Mock())
        with patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=int(10.25 * 2**30))):
            with entry.runtime_scope(pipeline, core, self.admission, lambda: None, write_root=self.root,
                    reserve_bytes=int(10.5 * 2**30), extension=extension):
                with self.assertRaisesRegex(OSError, 'checkpoint reserve'):
                    pipeline.require_project_budget()
                self.assertIs(pipeline._materialize_bank.__code__, materialize.__code__)
        extension.install_pipeline_adapter.assert_called_once(); core.install_pipeline_adapter.assert_not_called()
        self.assertIs(pipeline._materialize_bank, materialize)

    def test_static_extension_pins_the_actual_additional_runtime_module(self):
        from hiercp_v1x import v24_readonly_native_storage as storage
        from hiercp_v1x import v24_readonly_static_operator_storage as static
        document = dict(format=storage.FORMAT, profile=entry.PROFILE, storage_extension=entry.EXTENSION,
            debug=False, model_data_scale_preserved=True, scores_reused=False, learned_upper_reused=False)
        self.admission.write_text(json.dumps(document)); checksum = entry._sha(self.admission)
        with patch.object(storage, 'verify_admission'), patch.object(static, 'verify_admission') as verify, \
                patch.dict(os.environ, {entry.CHECKSUM_ENV: checksum}):
            actual, _, guard = entry.admit(self.admission)
            verify.assert_called_once_with(document, full_hash=False); guard()
            self.assertEqual(entry.runtime_files(actual), (*entry.FILES, entry.EXTENSION_FILE))
        document['storage_extension'] = 'unknown'; self.admission.write_text(json.dumps(document))
        with patch.object(storage, 'verify_admission'), patch.dict(os.environ, {entry.CHECKSUM_ENV: entry._sha(self.admission)}):
            with self.assertRaisesRegex(ValueError, 'Unknown explicit'):
                entry.admit(self.admission)

    def test_static_native_binding_checks_all105_roles_and_original_checkpoint_methods(self):
        from hiercp_v1x import v24_readonly_static_operator_storage as static
        checksum = 'e' * 64
        document, _ = self.coordination_fixture()
        self.admission.write_text(json.dumps(document))
        native = dict(format='DEBUG_current_full_model', physical_GPU=6, root=str(self.root / 'native'),
            bank=str(self.root / 'bank/index.json'), private_runtime=str(self.root / 'private_runtime'),
            storage_profile=entry.PROFILE, storage_extension=entry.EXTENSION,
            storage_admission=str(self.admission), storage_admission_sha256=checksum,
            checkpoint_coordination=document['checkpoint_coordination'],
            checkpoint_publication_proof=document['checkpoint_publication_proof'])
        bank = dict(physical_GPU=6, storage_profile=entry.PROFILE, storage_extension=entry.EXTENSION,
            storage_admission_sha256=checksum, baseline=dict(dataset_name='full131', data_identifier='original_fullres'))
        pipeline = SimpleNamespace(read=lambda path: native if Path(path) == self.native else bank,
            validate_bank=lambda value: value, CURRENT_FORMAT='DEBUG_current_full_model', TRAINER='original_full250_trainer')
        args = SimpleNamespace(native=self.native, storage_admission=self.admission, gpu=6)
        raw = dict(exact_original_store_class=True, readonly_role_references=210,
            readonly_static_operator_references=840, original_load_case_code_preserved=True, read_mode='r')
        dataset = dict(original_dataset_class='nnUNetDatasetBlosc2', original_load_case_code_preserved=True,
            readonly_mode='r', unpack_noop=True, full_preprocessed_cases=131)
        original_trainer = object()
        core = SimpleNamespace(install_dataset_adapter=unittest.mock.Mock(return_value=dataset))
        with patch.object(entry.importlib, 'import_module', return_value=SimpleNamespace(original_full250_trainer=original_trainer)), \
                patch.object(static, 'install_runtime_adapters', return_value=raw), \
                patch.object(static, 'verify_private_checkpoint_publication', return_value={'peak_checkpoint_slots': 3}) as checkpoints:
            result = entry._bind_native(pipeline, core, args, checksum)
            self.assertEqual(result[2]['raw_store']['readonly_static_operator_references'], 840)
            checkpoints.assert_called_once_with(self.admission, native['private_runtime'], trainer_class=original_trainer)
            raw['readonly_static_operator_references'] = 839
            with self.assertRaisesRegex(ValueError, 'eight static array roles'):
                entry._bind_native(pipeline, core, args, checksum)
            native.pop('storage_extension')
            with self.assertRaisesRegex(ValueError, 'own-arm read-only'):
                entry._bind_native(pipeline, core, args, checksum)

    def coordination_fixture(self):
        from hiercp_v1x import v24_readonly_native_storage as core
        from hiercp_v1x import v24_readonly_static_operator_storage as static
        source = ('class nnUNetTrainer:\n'
            '    def save_checkpoint(self, filename): return filename\n'
            '    def on_train_end(self): return self.save_checkpoint("DEBUG_final")\n'
            '    def on_epoch_end(self): return None\n')
        installed = self.root / 'DEBUG_installed_trainer.py'; installed.write_text(source)
        private = self.root / 'private_runtime/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py'
        private.parent.mkdir(parents=True, exist_ok=True); private.write_text(source)
        fold = str(self.root / 'native/fold_0')
        coordination = dict(format=entry.COORDINATION, global_peak_checkpoint_slots=7, per_arm_peak_checkpoint_slots=3,
            optimizer_work_locked=False, lock_file=dict(path=str(self.root / 'final.lock'), sha256='a' * 64),
            folds={'gpu6': fold})
        publication = dict(source=core.proof(installed), compiled_methods=static._compiled_methods(installed))
        document = dict(checkpoint_coordination=coordination, checkpoint_publication_proof=publication)
        proof = dict(format=entry.COORDINATION, active=True, coordination_active=True,
            global_peak_checkpoint_slots=7, per_arm_peak_checkpoint_slots=3, lock_file=coordination['lock_file'],
            original_on_train_end_code_preserved=True, original_on_train_end_called_once_per_completion=True,
            original_on_train_end_called_once_under_flock=True, successful_original_on_train_end_calls=0, bound_native_fold=None,
            optimizer_work_locked=False, save_checkpoint_unchanged=True, on_epoch_end_unchanged=True,
            retention_changed=False, failure_blocks_next_final=True, repeated_final_rejected=True,
            private_checkpoint_source=core.proof(private), actual_private_compiled_methods=static._compiled_methods(private),
            expected_private_compiled_methods=static._compiled_methods(private), admitted_source_compiled_methods=publication['compiled_methods'])
        return document, proof

    def test_final_coordination_requires_exact_lock_source_and_one_completed_own_fold(self):
        document, proof = self.coordination_fixture(); fold = document['checkpoint_coordination']['folds']['gpu6']
        private_runtime = self.root / 'private_runtime'
        self.assertNotEqual(proof['actual_private_compiled_methods'], document['checkpoint_publication_proof']['compiled_methods'])
        entry.validate_checkpoint_coordination(proof, document, completed=False, expected_fold=fold, private_runtime=private_runtime)
        with self.assertRaisesRegex(ValueError, 'single original final'):
            entry.validate_checkpoint_coordination(proof, document, completed=True, expected_fold=fold, private_runtime=private_runtime)
        proof.update(successful_original_on_train_end_calls=1, bound_native_fold=fold)
        entry.validate_checkpoint_coordination(proof, document, completed=True, expected_fold=fold, private_runtime=private_runtime)
        mutations = [dict(lock_file=dict(sha256='d' * 64)), dict(global_peak_checkpoint_slots=9),
            dict(optimizer_work_locked=True), dict(successful_original_on_train_end_calls=2),
            dict(bound_native_fold='other GPU fold'), dict(admitted_source_compiled_methods={})]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'single original final'):
                entry.validate_checkpoint_coordination(dict(proof, **mutation), document, completed=True, expected_fold=fold,
                    private_runtime=private_runtime)
        with self.assertRaisesRegex(ValueError, 'compiled methods differ'):
            entry.validate_checkpoint_coordination(dict(proof, actual_private_compiled_methods={}), document, completed=True,
                expected_fold=fold, private_runtime=private_runtime)

    def test_only_actual_training_worker_installs_coordination_before_original_entry(self):
        from hiercp_v1x import v24_readonly_static_operator_storage as static
        from hiercp_v1x import v24_native_crop_runtime as crop
        document, proof = self.coordination_fixture(); self.admission.write_text(json.dumps(document))
        native = dict(root=str(self.root), private_runtime=str(self.root / 'private_runtime'), physical_GPU=6,
            storage_extension=entry.EXTENSION, checkpoint_coordination=document['checkpoint_coordination'])
        adapters = {}; bank = dict(baseline=dict(epochs=250, physical_batch=2))
        pipeline = SimpleNamespace(TRAINER='original_full250_trainer', PLANS='original_full_plans',
            admit_native=unittest.mock.Mock(), new_json=unittest.mock.Mock())
        args = SimpleNamespace(native=self.native, storage_admission=self.admission, readonly_worker='train', physical_batch=None,
            training_args=['730', '3d_fullres', '0', '-tr', pipeline.TRAINER, '-p', pipeline.PLANS])
        trainer = object(); events = []
        def install(*args):events.append('install');return proof
        def run():
            self.assertEqual(events, ['install']);events.append('original_full250_entry')
            proof.update(successful_original_on_train_end_calls=1, bound_native_fold=document['checkpoint_coordination']['folds']['gpu6'])
            return 'original_result'
        with patch.object(entry, '_bind_native', return_value=(native, bank, adapters)), \
                patch.object(entry.importlib, 'import_module', return_value=SimpleNamespace(original_full250_trainer=trainer)), \
                patch.object(static, 'install_checkpoint_adapters', side_effect=install) as activation, \
                patch.object(crop, 'run_training_entry', side_effect=run), patch.object(entry.sys, 'argv', []):
            self.assertEqual(entry.worker(args, pipeline, object(), 'e' * 64, lambda: None), 'original_result')
        activation.assert_called_once_with(self.admission, native['private_runtime'], trainer)
        receipt = pipeline.new_json.call_args.args[1]
        self.assertEqual(receipt['adapters']['checkpoint_coordination']['successful_original_on_train_end_calls'], 1)
        self.assertEqual(receipt['physical_batch'], 2);self.assertEqual(receipt['native_epochs'], 250)

    def test_calibration_worker_never_activates_final_coordination(self):
        from hiercp_v1x import v24_readonly_static_operator_storage as static
        from hiercp_v1x import v24_native_calibration_runtime as runtime
        document, _ = self.coordination_fixture()
        native = dict(storage_extension=entry.EXTENSION, checkpoint_coordination=document['checkpoint_coordination'])
        pipeline = SimpleNamespace(admit_native=unittest.mock.Mock(), new_json=unittest.mock.Mock())
        args = SimpleNamespace(native=self.native, storage_admission=self.admission, readonly_worker='calibrate',
            physical_batch=2, worker_output=self.root)
        with patch.object(entry, '_bind_native', return_value=(native, {}, {})), \
                patch.object(static, 'install_checkpoint_adapters') as activation, \
                patch.object(runtime, '_calibrate_native_worker', return_value='actual clone calibration'):
            entry.worker(args, pipeline, object(), 'e' * 64, lambda: None)
        activation.assert_not_called()
        self.assertNotIn('checkpoint_coordination', pipeline.new_json.call_args.args[1]['adapters'])

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
