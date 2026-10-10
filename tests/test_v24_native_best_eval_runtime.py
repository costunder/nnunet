"""CPU DEBUG identity tests, never full training or medical evaluation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from hiercp_v1x import v24_native_best_eval_runtime as runtime
from hiercp_v1x import v24_nnunet_cp as pipeline


def _debug_trainer_class():
    class DebugTrainer:
        def load_checkpoint(self, filename):
            self.load_calls += 1
            self.current_epoch = json.loads(Path(filename).read_text())['current_epoch']
            if self.mutate_load:
                Path(filename).write_text('{"current_epoch":8}')
            return 'original load return'

        def perform_actual_validation(self, export_probabilities=False):
            self.validation_calls += 1
            if self.fail_validation:
                raise RuntimeError('Original DEBUG validation failed')
            output = Path(self.output_folder) / 'validation'
            output.mkdir()
            for case in self.cases[:self.produce_count]:
                (output / (case + '.nii.gz')).write_bytes(('DEBUG prediction ' + case).encode())
            (output / 'summary.json').write_text('{"DEBUG":true}')
            if self.mutate_validation:
                (Path(self.output_folder) / runtime.BEST).write_text('{"current_epoch":8}')
            return 'original validation return'

        def do_split(self):
            return ['DEBUG_train'], self.cases
    return DebugTrainer


class NativeBestGuardDebugTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='v24_best_guard_DEBUG_', dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.fold = Path(self.temp.name).resolve()
        self.cases = ['DEBUG_case_%02d' % index for index in range(26)]
        self.best = self.fold / runtime.BEST
        self.best.write_text('{"current_epoch":17}')
        self.final = self.fold / 'checkpoint_final.pth'
        self.final.write_text('{"current_epoch":250}')
        self.trainer_class = _debug_trainer_class()
        runtime._install(self.trainer_class, self.cases, bank_sha256='a' * 64)
        self.trainer = self.trainer_class()
        self.trainer.output_folder = str(self.fold)
        self.trainer.cases = self.cases.copy()
        self.trainer.load_calls = self.trainer.validation_calls = 0
        self.trainer.mutate_load = self.trainer.mutate_validation = self.trainer.fail_validation = False
        self.trainer.produce_count = 26

    def validate_receipt(self):
        return runtime.validate_best_validation_receipt(self.fold / runtime.RECEIPT, fold=self.fold,
            expected_cases=self.cases, bank_sha256='a' * 64)

    def test_actual_original_load_and_validation_called_once_then_completed_best_proof(self):
        self.assertEqual(self.trainer.load_checkpoint(self.best), 'original load return')
        self.assertEqual(self.trainer.perform_actual_validation(export_probabilities=True), 'original validation return')
        self.assertEqual((self.trainer.load_calls, self.trainer.validation_calls), (1, 1))
        proof = self.validate_receipt()
        self.assertEqual(proof['checkpoint_name'], runtime.BEST)
        self.assertEqual(proof['checkpoint_path'], str(self.best))
        self.assertEqual(proof['checkpoint_sha256'], runtime._sha(self.best))
        self.assertEqual(proof['best_current_epoch'], 17)
        self.assertEqual(set(proof['predictions_sha256']), set(self.cases))
        self.assertFalse(proof['final_checkpoint_fallback'])
        self.assertTrue(proof['original_full_validation_called'])

    def test_missing_best_never_falls_back_to_present_final(self):
        self.best.unlink()
        self.trainer.load_checkpoint(self.final)
        with self.assertRaisesRegex(ValueError, 'BEST checkpoint missing'):
            self.trainer.perform_actual_validation()
        self.assertEqual(self.trainer.validation_calls, 0)
        self.assertFalse((self.fold / runtime.RECEIPT).exists())

    def test_final_loaded_last_is_rejected_even_if_best_was_previously_loaded(self):
        self.trainer.load_checkpoint(self.best)
        self.trainer.load_checkpoint(self.final)
        with self.assertRaisesRegex(ValueError, 'BEST must be loaded last'):
            self.trainer.perform_actual_validation()
        self.assertEqual(self.trainer.validation_calls, 0)

    def test_best_changed_after_load_is_rejected(self):
        self.trainer.load_checkpoint(self.best)
        self.best.write_text('{"current_epoch":18}')
        with self.assertRaisesRegex(ValueError, 'unchanged'):
            self.trainer.perform_actual_validation()
        self.assertEqual(self.trainer.validation_calls, 0)

    def test_best_changed_during_original_load_is_rejected(self):
        self.trainer.mutate_load = True
        with self.assertRaisesRegex(ValueError, 'changed during original native loading'):
            self.trainer.load_checkpoint(self.best)
        with self.assertRaisesRegex(ValueError, 'loaded last'):
            self.trainer.perform_actual_validation()

    def test_complete_unchanged_26_case_split_is_required(self):
        self.trainer.load_checkpoint(self.best)
        self.trainer.cases[-1] = 'DEBUG_other'
        with self.assertRaisesRegex(ValueError, 'outer26 cohort'):
            self.trainer.perform_actual_validation()
        self.assertEqual(self.trainer.validation_calls, 0)

    def test_original_validation_failure_cannot_publish_success_provenance(self):
        self.trainer.load_checkpoint(self.best)
        self.trainer.fail_validation = True
        with self.assertRaisesRegex(RuntimeError, 'Original DEBUG validation failed'):
            self.trainer.perform_actual_validation()
        self.assertEqual(self.trainer.validation_calls, 1)
        self.assertFalse((self.fold / runtime.RECEIPT).exists())

    def test_missing_actual_prediction_cannot_publish_success_provenance(self):
        self.trainer.load_checkpoint(self.best)
        self.trainer.produce_count = 25
        with self.assertRaisesRegex(ValueError, 'all and only outer26 predictions'):
            self.trainer.perform_actual_validation()
        self.assertFalse((self.fold / runtime.RECEIPT).exists())

    def test_best_mutation_during_original_validation_cannot_publish_success_provenance(self):
        self.trainer.load_checkpoint(self.best)
        self.trainer.mutate_validation = True
        with self.assertRaisesRegex(ValueError, 'changed during original full validation'):
            self.trainer.perform_actual_validation()
        self.assertFalse((self.fold / runtime.RECEIPT).exists())

    def test_completed_proof_refuses_final_or_ambiguous_receipt(self):
        self.trainer.load_checkpoint(self.best)
        self.trainer.perform_actual_validation()
        path = self.fold / runtime.RECEIPT
        original = json.loads(path.read_text())
        mutations = [lambda value:value.update(checkpoint_name='checkpoint_final.pth'),
            lambda value:value.update(checkpoint_path=str(self.final)),
            lambda value:value.pop('original_checkpoint_load_called'),
            lambda value:value.update(native_validation_returned_successfully=False),
            lambda value:value.update(validation_cases=self.cases[:25])]
        for mutate in mutations:
            value = copy.deepcopy(original)
            mutate(value)
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                self.validate_receipt()
        path.write_text(json.dumps(original))
        (self.fold / 'validation' / (self.cases[0] + '.nii.gz')).write_bytes(b'changed DEBUG prediction')
        with self.assertRaises(ValueError):
            self.validate_receipt()

    def test_repeat_validation_cannot_overwrite_completed_results(self):
        self.trainer.load_checkpoint(self.best)
        self.trainer.perform_actual_validation()
        with self.assertRaises(FileExistsError):
            self.trainer.perform_actual_validation()
        self.assertEqual(self.trainer.validation_calls, 1)

    def test_existing_final_or_unknown_predictions_cannot_be_relabelled_best(self):
        self.trainer.load_checkpoint(self.best)
        directory = self.fold / 'validation'
        directory.mkdir()
        (directory / (self.cases[0] + '.nii.gz')).write_bytes(b'old FINAL or unknown prediction')
        with self.assertRaisesRegex(FileExistsError, 'Fresh empty validation output required'):
            self.trainer.perform_actual_validation()
        self.assertEqual(self.trainer.validation_calls, 0)
        self.assertFalse((self.fold / runtime.RECEIPT).exists())


class BestEntryDebugTests(unittest.TestCase):
    def test_explicit_val_best_required_before_guard_or_native_entry(self):
        with patch.object(sys, 'argv', ['-c', '730', '3d_fullres', '0']), patch.object(runtime, 'install_best_evaluation_guard') as install:
            with self.assertRaisesRegex(ValueError, '--val_best'):
                runtime.run_training_entry()
        install.assert_not_called()

    def test_full_native_cli_tail_and_return_are_preserved(self):
        module = types.ModuleType('nnunetv2.run.run_training')
        captured = []
        module.run_training_entry = lambda: captured.append(sys.argv[1:]) or 'native return'
        tail = ['730', '3d_fullres', '0', '--val', '--val_best']
        with patch.dict(sys.modules, {'nnunetv2.run.run_training':module}), patch.object(sys, 'argv', ['-c', *tail]), patch.object(runtime, 'install_best_evaluation_guard') as install:
            self.assertEqual(runtime.run_training_entry(), 'native return')
        self.assertEqual(captured, [tail])
        install.assert_called_once_with()


class BestPredictionReceiptDebugTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='v24_best_prediction_DEBUG_', dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.cases = ['DEBUG_case_%02d' % index for index in range(26)]
        self.bank = dict(baseline=dict(dataset_name='DEBUG_dataset'), split=dict(outer_val=self.cases))
        self.native = dict(root=str(self.root), bank_sha256='a' * 64)
        self.native_path = self.root / 'native.json'
        self.native_path.write_text(json.dumps(self.native))
        self.fold = self.root / 'nnUNet_results' / 'DEBUG_dataset' / (pipeline.TRAINER + '__' + pipeline.PLANS + '__3d_fullres') / 'fold_0'
        self.fold.mkdir(parents=True)
        self.best = self.fold / runtime.BEST
        self.best.write_bytes(b'BEST DEBUG checkpoint')
        self.predictions = self.root / 'DEBUG_prediction' / 'predictions'
        self.predictions.mkdir(parents=True)
        for case in self.cases:
            (self.predictions / (case + '.nii.gz')).write_bytes(('DEBUG ' + case).encode())
        self.receipt_path = self.predictions.parent / 'prediction_complete.json'
        self.proof = dict(format='v24_native_best_CT_only_prediction_v1', checkpoint_name=runtime.BEST,
            checkpoint_path=str(self.best), checkpoint_sha256=pipeline.sha(self.best),
            bank_sha256=self.native['bank_sha256'], native_metadata_path=str(self.native_path),
            native_metadata_sha256=pipeline.sha(self.native_path), command=['DEBUG_predict', '-chk', runtime.BEST],
            GT_passed_to_predictor=False, CP_at_inference=False,
            predictions_sha256={case:pipeline.sha(self.predictions / (case + '.nii.gz')) for case in self.cases})

    def admit(self, proof=None):
        self.receipt_path.write_text(json.dumps(self.proof if proof is None else proof))
        return pipeline.validate_best_prediction_receipt(self.native_path, self.native, self.bank, self.predictions)

    def test_exact_current_native_best_and_full26_prediction_checksums_accepted(self):
        path, proof = self.admit()
        self.assertEqual(path, self.receipt_path)
        self.assertEqual(proof['checkpoint_sha256'], pipeline.sha(self.best))

    def test_final_and_unattributed_receipts_rejected(self):
        mutations = [lambda value:value.update(checkpoint_name='checkpoint_final.pth'),
            lambda value:value.update(checkpoint_path=str(self.fold / 'checkpoint_final.pth')),
            lambda value:value.update(command=['DEBUG_predict', '-chk', 'checkpoint_final.pth']),
            lambda value:value.pop('checkpoint_name'), lambda value:value.pop('native_metadata_sha256'),
            lambda value:value.update(checkpoint_sha256='b' * 64)]
        for mutate in mutations:
            proof = copy.deepcopy(self.proof)
            mutate(proof)
            with self.assertRaisesRegex(ValueError, 'exact native BEST'):
                self.admit(proof)

    def test_missing_best_never_uses_final_file(self):
        self.best.unlink()
        (self.fold / 'checkpoint_final.pth').write_bytes(b'FINAL DEBUG checkpoint')
        with self.assertRaisesRegex(ValueError, 'BEST checkpoint missing'):
            self.admit()

    def test_missing_extra_or_mutated_predictions_rejected(self):
        first = self.predictions / (self.cases[0] + '.nii.gz')
        first.write_bytes(b'changed DEBUG')
        with self.assertRaisesRegex(ValueError, 'full26'):
            self.admit()
        first.write_bytes(('DEBUG ' + self.cases[0]).encode())
        (self.predictions / 'DEBUG_extra.nii.gz').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'full26'):
            self.admit()

    def test_predict_executes_best_checkpoint_cli_and_publishes_exact_best_identity(self):
        from tools import local_cnn_device
        inventory = self.root / 'inventory_DEBUG.json'
        raw = []
        for case in self.cases:
            image = self.root / (case + '_CT_DEBUG.nii.gz')
            image.write_bytes(('DEBUG CT ' + case).encode())
            raw.append(dict(case_id=case, image=str(image), image_sha256=pipeline.sha(image)))
        inventory.write_text(json.dumps(dict(split=self.bank['split'], raw_records=raw)))
        bank_path = self.root / 'bank_DEBUG.json'
        bank_path.write_text(json.dumps(self.bank))
        self.native.update(bank=str(bank_path), bank_sha256=pipeline.sha(bank_path))
        self.native_path.write_text(json.dumps(self.native))
        output = self.root / 'fresh_predict_DEBUG'
        commands = []
        def original_predict_debug(command, **kwargs):
            commands.append(command)
            directory = output / 'predictions'
            directory.mkdir()
            for case in self.cases:
                (directory / (case + '.nii.gz')).write_bytes(('new DEBUG ' + case).encode())
            return types.SimpleNamespace(returncode=0)
        with patch.object(local_cnn_device, 'select'), patch.object(pipeline, 'admit_native', return_value=(self.native, self.bank)), patch.object(pipeline, 'environment', return_value={}), patch.object(pipeline.subprocess, 'run', side_effect=original_predict_debug):
            result = pipeline.predict(self.native_path, inventory_path=inventory, output=output)
        command = commands[0]
        self.assertEqual(command[command.index('-chk') + 1], runtime.BEST)
        self.assertNotIn('checkpoint_final.pth', command)
        receipt = json.loads((output / 'prediction_complete.json').read_text())
        self.assertEqual(receipt['checkpoint_path'], str(self.best))
        self.assertEqual(receipt['checkpoint_sha256'], pipeline.sha(self.best))
        self.assertEqual(result, output / 'predictions')
        pipeline.validate_best_prediction_receipt(self.native_path, self.native, self.bank, result)

    def test_train_requests_val_best_and_requires_best_provenance_after_final_completion(self):
        from tools import local_cnn_device
        training_root = self.root / 'train_DEBUG'
        training_root.mkdir()
        native = dict(root=str(training_root), format='DEBUG', bank_sha256='b' * 64)
        bank = dict(self.bank, pin={'DEBUG':True})
        calibration = self.root / 'calibration_DEBUG.json'
        calibration.write_text('{"DEBUG":true}')
        fold = training_root / 'nnUNet_results' / 'DEBUG_dataset' / (pipeline.TRAINER + '__' + pipeline.PLANS + '__3d_fullres') / 'fold_0'
        commands = []
        def child_debug(command, **kwargs):
            commands.append(command)
            fold.mkdir(parents=True)
            (fold / 'checkpoint_final.pth').write_bytes(b'DEBUG FINAL full250 witness')
            (fold / runtime.BEST).write_bytes(b'DEBUG BEST')
            return types.SimpleNamespace(stdout=[], wait=lambda:0)
        with patch.object(local_cnn_device, 'select'), patch.object(pipeline, 'require_project_budget'), patch.object(pipeline, 'admit_native', return_value=(native, bank)), patch.object(pipeline, '_admit_arm_gpu'), patch.object(pipeline, 'admit_native_calibration', return_value=calibration), patch.object(pipeline, 'environment', return_value={}), patch.object(pipeline.subprocess, 'Popen', side_effect=child_debug):
            with self.assertRaises(FileNotFoundError):
                pipeline.train(self.native_path)
        self.assertIn('--val_best', commands[0])
        self.assertEqual(commands[0][commands[0].index('-c') + 1], runtime.TRAIN_COMMAND)
        self.assertEqual(list(training_root.glob('training_complete_*.json')), [])


if __name__ == '__main__':
    unittest.main()
