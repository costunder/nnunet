"""CPU UNIT: GPU5 actual tensor/parity admission rejection paths.

These explicitly synthetic tensor fixtures test receipt validation only.
They do not construct the research model, read clinical data, launch CUDA,
measure native throughput or claim full training/evaluation completion.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hiercp_v1x.u_bridge_training import digest
from tests import test_v24_performance_continuation as original_fixture
from tools import run_v24_gpu5_performance_continuation as gpu5


class GPU5ActualTensorGateCPUUnit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='DEBUG_gpu5_gate_UNIT_', dir=gpu5.ROOT / 'outputs')
        self.root = Path(self.temp.name)
        original = original_fixture.NativeProbeAdmissionUnit()
        original.setUp()
        self.rows = [copy.deepcopy(original.baseline), copy.deepcopy(original.optimized)]
        self.paths = []
        self.tensor_paths = []
        self.values = dict(scores=tuple(torch.arange(131, dtype=torch.float32) for _ in range(4)),
                           consistency=torch.tensor(0.5), loss=torch.tensor(1.25),
                           gradients={'CPU_UNIT_parameter_' + str(index): torch.tensor([index / 1024.])
                                      for index in range(gpu5.EXPECTED_GRADIENTS)})
        for index, row in enumerate(self.rows):
            directory = self.root / ('baseline' if index == 0 else 'optimized')
            directory.mkdir()
            path = directory / 'result.json'
            tensor_path = directory / 'numerical_tensors.pt'
            row.update(physical_candidate_chunk=32, GPU_arm=5,
                       original_named_trainable_parameter_count=981,
                       initial_optimizer_sha256='a' * 64, after_optimizer_sha256='a' * 64,
                       initial_scaler_sha256='b' * 64, after_scaler_sha256='b' * 64,
                       optimizer_contains_all_trainable_parameters_exactly_once=True,
                       scheduler_and_shuffle_state_unchanged=True,
                       GPU5_probe_source_sha256=gpu5._sha(gpu5.ROOT / 'tools/run_v24_gpu5_performance_probe.py'),
                       model_contract={'CPU_UNIT_only': True, 'no_model_constructed': True},
                       output_sha256=digest(self.values['scores']), gradient_sha256=digest(self.values['gradients']))
            row['source_checkpoint'].update(numerical_state_sha256={key: key for key in gpu5.NUMERICAL_STATES},
                                            optimizer_parameter_tensors=981)
            row['source_checkpoint']['numerical_state_sha256'].update(model=row['initial_model_sha256'],
                optimizer=row['initial_optimizer_sha256'], scaler=row['initial_scaler_sha256'])
            row['gradient'].update(trainable_parameter_tensors=981, gradient_present=981)
            if index == 1:
                row['input_runtime'] = dict(contract=dict(runtime_source_sha256=
                    gpu5._sha(gpu5.ROOT / 'hiercp_v1x/v24_input_runtime.py')))
                row['input_runtime_profile'] = {name: 0 for name in ('source_live_chunks',
                    'source_cache_live_entries', 'source_cache_live_key_bytes',
                    'source_cache_live_result_bytes', 'source_cache_live_tree_visible_array_bytes')}
            row['input_tensor_proof'].update(physical_candidate_chunk=32,
                observed_native_chunks=17, expected_native_chunks=17,
                content=dict(local_chunks=[{} for _ in range(17)], upper_graphs=[
                    {'record_ids': ['CPU_UNIT_record_' + str(i) for i in range(131)]} for _ in range(4)]))
            torch.save(self.values, tensor_path)
            row['numerical_tensor_file'] = dict(path=str(tensor_path.resolve()),
                size=tensor_path.stat().st_size, raw_sha256=gpu5._sha(tensor_path),
                content_sha256=digest(self.values), all_named_trainable_gradients=981,
                server_only=True, no_raw_CT=True, no_model_update=True)
            self.paths.append(path)
            self.tensor_paths.append(tensor_path)
        self.write_rows()

    def tearDown(self):
        if (self.root.resolve().parent != (gpu5.ROOT / 'outputs').resolve()
                or not self.root.name.startswith('DEBUG_gpu5_gate_UNIT_')):
            raise ValueError('Refusing cleanup outside this newly created CPU UNIT workspace directory')
        self.temp.cleanup()

    def write_rows(self):
        for path, row in zip(self.paths, self.rows):
            path.write_text(json.dumps(row, allow_nan=False), encoding='utf8')

    def compare(self):
        self.write_rows()
        return gpu5.compare_probes(*self.paths, 'actual-snapshot')

    def test_all981_actual_value_files_admit(self):
        result = self.compare()
        self.assertTrue(result['actual_numerical_tensor_values_bitwise_equal'])
        self.assertEqual(result['all_named_trainable_gradients'], 981)
        self.assertTrue(result['optimizer_and_scaler_unchanged'])
        self.assertEqual(result['physical_candidate_chunk'], 32)
        self.assertTrue(result['single_DEBUG_batch_not_steady_state_throughput'])

    def test_wrong_GPU_or_any_parameter_disconnected_rejected(self):
        saved = copy.deepcopy(self.rows[1])
        for changes in ({'GPU_arm': 6}, {'original_named_trainable_parameter_count': 980},
                        {'optimizer_contains_all_trainable_parameters_exactly_once': False},
                        {'scheduler_and_shuffle_state_unchanged': False}):
            self.rows[1] = copy.deepcopy(saved)
            self.rows[1].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.compare()
        self.rows[1] = saved
        self.rows[1]['gradient']['gradient_present'] = 980
        with self.assertRaises(ValueError):
            self.compare()

    def test_optimizer_scaler_or_original_scheduler_change_rejected(self):
        saved = copy.deepcopy(self.rows[1])
        for state in ('optimizer', 'scaler'):
            self.rows[1] = copy.deepcopy(saved)
            self.rows[1]['after_' + state + '_sha256'] = 'c' * 64
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, state):
                self.compare()
        self.rows[1] = copy.deepcopy(saved)
        self.rows[1]['source_checkpoint']['numerical_state_sha256']['scheduler'] = 'different'
        with self.assertRaisesRegex(ValueError, 'six-state'):
            self.compare()

    def test_wrong_chunk_incomplete_chunks_or_old_probe_source_rejected(self):
        saved = copy.deepcopy(self.rows[1])
        self.rows[1]['GPU5_probe_source_sha256'] = 'old-probe'
        with self.assertRaises(ValueError):
            self.compare()
        self.rows[1] = copy.deepcopy(saved)
        self.rows[1]['input_tensor_proof']['observed_native_chunks'] = 16
        with self.assertRaises(ValueError):
            self.compare()
        for row in self.rows:
            row['physical_candidate_chunk'] = 64
            row['input_tensor_proof'].update(physical_candidate_chunk=64, observed_native_chunks=9,
                expected_native_chunks=9, content=dict(local_chunks=[{} for _ in range(9)],
                    upper_graphs=row['input_tensor_proof']['content']['upper_graphs']))
        with self.assertRaisesRegex(ValueError, 'GPU5 chunk32'):
            self.compare()

    def test_actual_file_value_change_not_just_receipt_claim_rejected(self):
        changed = copy.deepcopy(self.values)
        changed['gradients']['CPU_UNIT_parameter_980'] += 1.
        torch.save(changed, self.tensor_paths[1])
        entry = self.rows[1]['numerical_tensor_file']
        entry.update(size=self.tensor_paths[1].stat().st_size, raw_sha256=gpu5._sha(self.tensor_paths[1]))
        with self.assertRaisesRegex(ValueError, 'tensor values'):
            self.compare()

    def test_truncated_gradient_file_rejected_even_with_updated_file_digest(self):
        changed = copy.deepcopy(self.values)
        del changed['gradients']['CPU_UNIT_parameter_980']
        torch.save(changed, self.tensor_paths[1])
        entry = self.rows[1]['numerical_tensor_file']
        entry.update(size=self.tensor_paths[1].stat().st_size, raw_sha256=gpu5._sha(self.tensor_paths[1]),
                     content_sha256=digest(changed))
        with self.assertRaisesRegex(ValueError, 'named981 gradients'):
            self.compare()

    def test_cross_directory_tensor_redirection_and_changed_bytes_rejected(self):
        self.rows[1]['numerical_tensor_file']['path'] = str(self.tensor_paths[0])
        with self.assertRaisesRegex(ValueError, 'server-local'):
            self.compare()
        self.rows[1]['numerical_tensor_file']['path'] = str(self.tensor_paths[1])
        with self.tensor_paths[1].open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'size/hash changed'):
            self.compare()

    def test_bit_comparison_detects_signed_zero_and_shape(self):
        self.assertTrue(torch.equal(torch.tensor([0.]), torch.tensor([-0.])))
        with self.assertRaisesRegex(ValueError, 'bits differ'):
            gpu5._same_bits(torch.tensor([0.]), torch.tensor([-0.]), 'signed_zero')
        with self.assertRaisesRegex(ValueError, 'dtype/shape'):
            gpu5._same_bits(torch.tensor([0.]), torch.tensor([[0.]]), 'shape')

    def test_actual524_score_coverage_required(self):
        self.rows[1]['input_tensor_proof']['content']['upper_graphs'][0]['record_ids'].pop()
        with self.assertRaisesRegex(ValueError, 'all524'):
            self.compare()

    def test_full_model_contract_parity_and_GPU5_execution_required(self):
        self.rows[1]['model_contract']['changed'] = True
        with self.assertRaisesRegex(ValueError, 'six-state'):
            self.compare()
        with self.assertRaisesRegex(ValueError, 'GPU5 arm'):
            gpu5._gpu5_args(SimpleNamespace(mode='train', gpu=6))
        with self.assertRaisesRegex(ValueError, 'GPU5 arm'):
            gpu5._gpu5_args(SimpleNamespace(mode='calibrate', gpu=5))

    def test_private_training_source_guard_preserves_shared_code_and_document(self):
        def template(args):
            return admit(args)
        shared = SimpleNamespace(train=template)
        document = {'runtime': {'files_sha256': {'shared_source': 'shared_sha'}}}
        with patch.object(gpu5, 'admit', return_value=('original_receipt', document)), \
                patch.object(gpu5, 'runtime_identity', return_value={
                    'GPU5_files_sha256': {'gpu5_gate_source': 'own_sha'}}):
            private = gpu5._guarded_training(shared)
            original, guarded = private(SimpleNamespace(GPU5_CPU_UNIT=True))
        self.assertIs(private.__code__, template.__code__)
        self.assertIsNot(private.__globals__, template.__globals__)
        self.assertEqual(original, 'original_receipt')
        self.assertEqual(document['runtime']['files_sha256'], {'shared_source': 'shared_sha'})
        self.assertEqual(guarded['runtime']['files_sha256'],
                         {'shared_source': 'shared_sha', 'gpu5_gate_source': 'own_sha'})
        self.assertTrue(guarded['runtime']['GPU5_exact_gate_source_guard'])

    def test_measured_runtime_source_and_complete_chunk_release_required(self):
        saved = copy.deepcopy(self.rows[1])
        self.rows[1]['input_runtime']['contract']['runtime_source_sha256'] = 'stale_source'
        with self.assertRaisesRegex(ValueError, 'admitted training source'):
            self.compare()
        for name in saved['input_runtime_profile']:
            self.rows[1] = copy.deepcopy(saved)
            self.rows[1]['input_runtime_profile'][name] = 1
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'fully released'):
                self.compare()
        self.rows[1] = copy.deepcopy(saved)
        del self.rows[1]['input_runtime_profile']['source_cache_live_entries']
        with self.assertRaisesRegex(ValueError, 'fully released'):
            self.compare()


if __name__ == '__main__':
    unittest.main()
