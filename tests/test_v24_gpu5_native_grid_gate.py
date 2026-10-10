"""CPU UNIT: explicitly synthetic receipts/tensors, never native research data.

Exercises actual files and all981 gradient identities, original-repeat bounds,
strict forward/state rejection and the one explicitly supported warning.
"""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import torch

from hiercp_v1x.u_bridge_training import digest
from tests import test_v24_gpu5_performance_continuation as fixture
from tools import run_v24_gpu5_performance_continuation as gate
from tools import run_v24_gpu5_performance_probe as probe
from tools import v24_gpu5_native_grid_gate as native

MESSAGE = native.OPERATOR + " does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue."


class NativeGridObservedEnvelopeCPUUnit(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.GPU5ActualTensorGateCPUUnit(); self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.rows = self.fixture.rows
        self.paths = self.fixture.paths
        directory = self.fixture.root / 'baseline_repeat'; directory.mkdir()
        self.repeat_path = directory / 'result.json'
        self.repeat_tensor = directory / 'numerical_tensors.pt'
        self.repeat = copy.deepcopy(self.rows[0])
        self.values = [copy.deepcopy(self.fixture.values) for _ in range(3)]
        self.all_rows = [self.rows[0], self.repeat, self.rows[1]]
        for row in self.all_rows:
            row['debug_numerics'].update(debug_native_grid_sampler_numerics=True,
                admitted_native_nondeterministic_operator=native.OPERATOR)
            row['debug_numerics']['actual_forward_flags'].update(deterministic_warn_only=True,
                cudnn_deterministic=True, cudnn_benchmark=False)
            row['native_grid_sampler_warning_receipt'] = dict(operator=native.OPERATOR,
                only_declared_operator=True, original_native_atomic_kernel=True, warning_count=1,
                warnings=[dict(category='UserWarning', message=MESSAGE)])
        self.all_paths = [self.paths[0], self.repeat_path, self.paths[1]]
        self.tensor_paths = [self.fixture.tensor_paths[0], self.repeat_tensor, self.fixture.tensor_paths[1]]
        self.persist()

    def persist(self):
        for row, path, tensor_path, values in zip(self.all_rows, self.all_paths, self.tensor_paths, self.values):
            torch.save(values, tensor_path)
            row['gradient_sha256'] = digest(values['gradients']); row['output_sha256'] = digest(values['scores'])
            row['loss'] = float(values['loss'])
            row['numerical_tensor_file'].update(path=str(tensor_path.resolve()), size=tensor_path.stat().st_size,
                raw_sha256=gate._sha(tensor_path), content_sha256=digest(values))
            path.write_text(json.dumps(row, allow_nan=False), encoding='utf8')

    def compare(self, policy='per-element'):
        self.persist()
        return gate.compare_probes(self.paths[0], self.paths[1], 'actual-snapshot',
            baseline_repeat_path=self.repeat_path, debug_native_grid_sampler_numerics=True,
            gradient_envelope_policy=policy)

    def test_three_actual_artifacts_exact_admit_without_relaxing_production(self):
        result = self.compare()
        self.assertTrue(result['actual_numerical_tensor_values_bitwise_equal'])
        self.assertTrue(result['input_forward_loss_and_model_RNG_state_bitwise_equal'])
        self.assertFalse(result['production_numerical_policy_changed'])
        self.assertEqual(len(result['gradient_envelope']['tensors']), 981)
        self.assertEqual(result['gradient_envelope']['epsilon'], 0)

    def test_per_element_bound_is_actual_original_difference_without_epsilon(self):
        name = 'CPU_UNIT_parameter_0'
        for values in self.values: values['gradients'][name] = torch.tensor([0., 1., 2.])
        self.values[1]['gradients'][name] += torch.tensor([0., .25, -.5])
        self.values[2]['gradients'][name] += torch.tensor([0., -.125, .5])
        result = self.compare()
        self.assertFalse(result['actual_numerical_tensor_values_bitwise_equal'])
        self.values[2]['gradients'][name][0] = torch.finfo(torch.float32).tiny
        with self.assertRaises(native.NativeGradientEnvelopeError) as raised: self.compare()
        self.assertEqual(raised.exception.evidence['per_element_failed_tensors'], 1)

    def test_tensor_max_and_RMS_both_required_and_element_failures_remain_visible(self):
        name = 'CPU_UNIT_parameter_0'
        for values in self.values: values['gradients'][name] = torch.zeros(3)
        self.values[1]['gradients'][name] = torch.tensor([2., 0., 0.])
        self.values[2]['gradients'][name] = torch.tensor([0., 1., 0.])
        with self.assertRaises(native.NativeGradientEnvelopeError): self.compare()
        result = self.compare('per-tensor-max-rms')
        self.assertEqual(result['gradient_envelope']['per_element_failed_tensors'], 1)
        self.values[2]['gradients'][name] = torch.tensor([2., 2., 0.])
        with self.assertRaises(native.NativeGradientEnvelopeError) as raised: self.compare('per-tensor-max-rms')
        self.assertEqual(raised.exception.evidence['per_tensor_max_RMS_failed_tensors'], 1)

    def test_zero_noise_tensor_requires_exact_bits_in_both_policies(self):
        self.values[2]['gradients']['CPU_UNIT_parameter_0'] = torch.tensor([-0.])
        for policy in native.POLICIES:
            with self.subTest(policy=policy), self.assertRaises(native.NativeGradientEnvelopeError): self.compare(policy)

    def test_native_support_global_max_is_explicit_and_keeps_tensor_failures_visible(self):
        first, second = 'CPU_UNIT_parameter_0', 'CPU_UNIT_parameter_1'
        for values in self.values:
            values['gradients'][first] = torch.zeros(1)
            values['gradients'][second] = torch.zeros(1)
        self.values[1]['gradients'][first] = torch.tensor([2.])
        self.values[1]['gradients'][second] = torch.tensor([.1])
        self.values[2]['gradients'][second] = torch.tensor([.2])
        result = self.compare('native-support-global-max')
        evidence = result['gradient_envelope']
        self.assertEqual(evidence['global_original_repeat_max_absolute_difference'], 2.)
        self.assertFalse(evidence['per_tensor_envelope_guaranteed'])
        self.assertEqual(evidence['per_tensor_max_RMS_failed_tensors'], 1)
        self.values[2]['gradients'][second] = torch.tensor([3.])
        with self.assertRaises(native.NativeGradientEnvelopeError): self.compare('native-support-global-max')

    def test_native_support_global_max_rejects_new_variable_parameter(self):
        self.values[1]['gradients']['CPU_UNIT_parameter_0'] = torch.tensor([2.])
        self.values[2]['gradients']['CPU_UNIT_parameter_1'] = torch.tensor([.01])
        with self.assertRaises(native.NativeGradientEnvelopeError): self.compare('native-support-global-max')

    def test_preserved_memory_selection_requires_actual_probe_contracts(self):
        selection=dict(actual_contract={'CPU_UNIT_source':'old_allocator'})
        for row in self.all_rows:
            row['provider_profiles_after']={'train':dict(memory_runtime=dict(
                contract=selection['actual_contract'],RSS_limit_bytes=64*2**30,profile=dict(strict_failures=0)))}
        self.persist()
        with patch.object(gate,'_PRESERVED_MEMORY_SELECTION',None), patch(
                'tools.v24_memory_reclamation_gate.route_memory_runtime',return_value=selection):
            self.assertEqual(gate.select_preserved_probed_memory(self.fixture.root,self.all_paths),selection)
            self.all_rows[2]['provider_profiles_after']['train']['memory_runtime']['contract']={'CPU_UNIT_source':'unmeasured_allocator'}
            self.persist()
            with self.assertRaisesRegex(ValueError,'actually measured'):
                gate.select_preserved_probed_memory(self.fixture.root,self.all_paths)
        with self.assertRaisesRegex(ValueError,'distinct'):
            gate.select_preserved_probed_memory(self.fixture.root,[self.paths[0]]*3)

    def test_forward_loss_and_consistency_never_use_gradient_envelope(self):
        saved = copy.deepcopy(self.values[2])
        for name in ('loss', 'consistency'):
            self.values[2] = copy.deepcopy(saved); self.values[2][name] += .0001
            with self.subTest(name=name), self.assertRaises(ValueError): self.compare('per-tensor-max-rms')
        self.values[2] = copy.deepcopy(saved); self.values[2]['scores'][0][0] += .0001
        with self.assertRaises(ValueError): self.compare('per-tensor-max-rms')

    def test_checkpoint_RNG_optimizer_scaler_and_proof_changes_fail(self):
        saved = copy.deepcopy(self.repeat)
        for key in ('initial_RNG_sha256', 'after_RNG_sha256', 'initial_model_sha256', 'after_optimizer_sha256', 'after_scaler_sha256'):
            self.repeat.clear(); self.repeat.update(copy.deepcopy(saved)); self.repeat[key] = 'different'
            with self.subTest(key=key), self.assertRaises(ValueError): self.compare()
        self.repeat.clear(); self.repeat.update(saved); self.repeat['source_checkpoint']['raw_sha256'] = 'different'
        with self.assertRaises(ValueError): self.compare()

    def test_missing_repeat_wrong_warning_unknown_policy_and_reused_receipt_fail(self):
        with self.assertRaisesRegex(ValueError, 'baseline repeat'):
            gate.compare_probes(*self.paths, 'actual-snapshot', debug_native_grid_sampler_numerics=True)
        with self.assertRaisesRegex(ValueError, 'opt-in'):
            gate.compare_probes(*self.paths, 'actual-snapshot', baseline_repeat_path=self.repeat_path)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            gate.compare_probes(*self.paths, 'actual-snapshot', baseline_repeat_path=self.paths[0], debug_native_grid_sampler_numerics=True)
        self.repeat['native_grid_sampler_warning_receipt']['warnings'][0]['message'] = 'different_operator_cuda warning'
        with self.assertRaisesRegex(ValueError, 'Unexpected operator'): self.compare()

    def test_nonfinite_disconnected_or_reordered_actual_gradients_fail(self):
        saved = copy.deepcopy(self.values[1])
        self.values[1]['gradients']['CPU_UNIT_parameter_0'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'): self.compare()
        self.values[1] = copy.deepcopy(saved); del self.values[1]['gradients']['CPU_UNIT_parameter_980']
        with self.assertRaisesRegex(ValueError, 'named981'): self.compare()
        self.values[1] = copy.deepcopy(saved)
        self.values[1]['gradients'] = dict(reversed(list(saved['gradients'].items())))
        with self.assertRaisesRegex(ValueError, 'identities'): self.compare()

    def test_declared_warning_only_and_explicit_parser_mutual_exclusion(self):
        with probe.native_grid_warning_scope(True) as receipt: warnings.warn(MESSAGE, UserWarning)
        self.assertEqual(receipt['warning_count'], 1)
        with self.assertRaisesRegex(ValueError, 'Unexpected warning'):
            with probe.native_grid_warning_scope(True): warnings.warn('different CUDA operator', UserWarning)
        with self.assertRaisesRegex(ValueError, 'actual declared'):
            with probe.native_grid_warning_scope(True): pass
        arguments = ['--variant', 'baseline', '--source-output', 'a', '--source-code', 'b', '--probe-output', 'c', '--debug-performance-probe']
        options, _ = probe.parse_probe_options(arguments + ['--debug-native-grid-sampler-numerics'])
        self.assertTrue(options.debug_native_grid_sampler_numerics)
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            probe.parse_probe_options(arguments + ['--debug-native-grid-sampler-numerics', '--debug-deterministic-numerics'])

    def test_private_production_comparator_preserves_original_code(self):
        def original(a, b, checkpoint): return compare_probes(a, b, checkpoint)
        shared = SimpleNamespace(train=original)
        comparator = lambda a, b, checkpoint: {'CPU_UNIT_actual_re_admitted': True}
        private = gate._guarded_training(shared, comparator)
        self.assertIs(private.__code__, original.__code__)
        self.assertEqual(private({}, {}, 'owned'), {'CPU_UNIT_actual_re_admitted': True})


if __name__ == '__main__': unittest.main()
