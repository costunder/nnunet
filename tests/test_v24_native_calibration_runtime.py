"""CPU UNIT: exact flag transport, isolation and fail-closed command guards.

These metadata fixtures never construct a model or claim a native GPU trial.
"""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import unittest
import uuid

from hiercp_v1x import v24_native_calibration_runtime as runtime


TRAINER = '''
class nnUNetTrainer:
    def __init__(self, plans, configuration, fold, dataset_json, device):
        continue_training = plans.pop("continue_training")
        self.plans_manager = PlansManager(plans)
        self.logger = MetaLogger(self.output_folder, continue_training)
'''
CLI = '''
def get_trainer_from_args(dataset_name_or_id, configuration, fold,
                         continue_training=False, device=None):
    plans = load_json(plans_file)
    plans["continue_training"] = continue_training
    return nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                         dataset_json=dataset_json, device=device)
'''


GLOBAL_VALUE = 'original'


@contextmanager
def unit_directory():
    path = Path(__file__).resolve().parents[1] / 'outputs' / ('v24_native_flags_CPU_UNIT_' + uuid.uuid4().hex)
    path.mkdir(parents=True)
    yield str(path)


def original_function(value='default', *, option='keyword'):
    return GLOBAL_VALUE, value, option


class NativeFreshCLITransportDebug(unittest.TestCase):
    def test_actual_flag_pattern_proves_fresh_default_logger_only(self):
        proof = runtime._prove_fresh_cli(TRAINER, CLI)
        self.assertIs(proof['fresh_CLI_default'], False)
        self.assertEqual(proof['constructor_flag_use'], 'MetaLogger_only')
        self.assertIs(proof['fresh_CLI_equivalent'], True)

    def test_changed_cli_default_missing_or_multiple_injection_fail(self):
        for source in (CLI.replace('continue_training=False', 'continue_training=True'),
                CLI.replace('plans["continue_training"] = continue_training', 'pass'),
                CLI.replace('plans["continue_training"] = continue_training',
                    'plans["continue_training"] = continue_training\n    plans["continue_training"] = continue_training')):
            with self.subTest(source=source), self.assertRaises(ValueError):
                runtime._prove_fresh_cli(TRAINER, source)

    def test_constructor_flag_use_in_model_or_changed_pop_fail(self):
        for source in (TRAINER.replace('self.plans_manager', 'self.extra = continue_training\n        self.plans_manager'),
                TRAINER.replace('plans.pop("continue_training")', 'plans.get("continue_training", False)'),
                TRAINER.replace('MetaLogger(self.output_folder, continue_training)', 'MetaLogger(self.output_folder, True)')):
            with self.subTest(source=source), self.assertRaises(ValueError):
                runtime._prove_fresh_cli(source, CLI)

    def test_only_exact_plans_is_deepcopied_and_flagged_without_disk_write(self):
        with unit_directory() as directory:
            plans = Path(directory) / 'plans.json'
            dataset = Path(directory) / 'dataset.json'
            values = {'model': {'channels': [32, 64]}, 'normalization': {'mean': 1.2}}
            plans.write_text(json.dumps(values))
            dataset.write_text('{"labels":{"background":0}}')
            original = copy.deepcopy(values)
            calls = []
            def read(path):
                calls.append(path)
                return values if Path(path) == plans else {'labels': {'background': 0}}
            reader, count = runtime._plans_reader(read, plans)
            flagged = reader(plans)
            self.assertIs(flagged.pop('continue_training'), False)
            flagged['model']['channels'][0] = 999
            self.assertEqual(values, original)
            self.assertEqual(json.loads(plans.read_text()), original)
            self.assertEqual(reader(dataset), {'labels': {'background': 0}})
            self.assertEqual(count, [1])
            self.assertEqual(calls, [plans, dataset])

    def test_existing_runtime_flags_are_never_silently_replaced(self):
        with unit_directory() as directory:
            plans = Path(directory) / 'plans.json'
            plans.write_text('{}')
            for values in ({'continue_training': True}, {'continue_training': False},
                    {'continue_training': 0}, {'only_run_validation': False}, []):
                reader, count = runtime._plans_reader(lambda path: values, plans)
                with self.subTest(values=values), self.assertRaises(ValueError):
                    reader(plans)
                self.assertEqual(count, [0])

    def test_private_function_keeps_code_and_defaults_without_global_mutation(self):
        clone = runtime._function(original_function, {'GLOBAL_VALUE': 'private'})
        self.assertIs(clone.__code__, original_function.__code__)
        self.assertEqual(clone(), ('private', 'default', 'keyword'))
        self.assertEqual(original_function(), ('original', 'default', 'keyword'))
        self.assertEqual(clone.__globals__['GLOBAL_VALUE'], 'private')
        self.assertEqual(original_function.__globals__['GLOBAL_VALUE'], 'original')
        self.assertIsNot(clone.__kwdefaults__, original_function.__kwdefaults__)

    def test_exact_original_worker_command_changes_only_c_code(self):
        with unit_directory() as directory:
            native = Path(directory) / 'native.json'
            native.write_text('{}')
            command = [sys.executable, '-B', '-c', runtime.ORIGINAL_WORKER_COMMAND,
                str(native), '4', str(Path(directory) / 'batch_4')]
            original = command.copy()
            changed = runtime._replace_worker_command(command, native, sys.executable)
            self.assertEqual(command, original)
            self.assertEqual(changed[:3] + changed[4:], original[:3] + original[4:])
            self.assertEqual(changed[3], runtime.WORKER_COMMAND)

    def test_unexpected_process_or_worker_arguments_fail_closed(self):
        with unit_directory() as directory:
            native = Path(directory) / 'native.json'; native.write_text('{}')
            other = Path(directory) / 'other.json'; other.write_text('{}')
            command = [sys.executable, '-B', '-c', runtime.ORIGINAL_WORKER_COMMAND, str(native), '4', 'batch_4']
            variants = [command + ['--resume'], tuple(command), ['different-python'] + command[1:]]
            for index, value in ((1, '-m'), (3, 'unrelated worker'), (4, str(other)), (5, '0'), (5, '-1'), (5, '4.0')):
                modified = command.copy(); modified[index] = value; variants.append(modified)
            for value in variants:
                with self.subTest(command=value), self.assertRaises(ValueError):
                    runtime._replace_worker_command(value, native, sys.executable)

    def test_gpu_assignment_is_guarded_before_any_pipeline_or_model_call(self):
        with self.assertRaisesRegex(ValueError, 'GPU1'):
            runtime.calibrate_native('nonexistent-native.json', gpu=5)

    def test_receipt_publication_refuses_overwriting_existing_proof(self):
        with unit_directory() as directory:
            path = Path(directory) / 'proof.json'
            runtime._publish(path, {'CPU_UNIT': True})
            with self.assertRaises(FileExistsError):
                runtime._publish(path, {'CPU_UNIT': False})
            self.assertEqual(json.loads(path.read_text()), {'CPU_UNIT': True})


if __name__ == '__main__':
    unittest.main()
