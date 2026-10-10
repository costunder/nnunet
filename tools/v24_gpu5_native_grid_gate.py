"""Explicit DEBUG admission for observed native GPU5 grid-backward variation.

No tolerance, epsilon, multiplier or production kernel setting is introduced.
Inputs, forward values, loss, model/RNG/checkpoint and optimizer/scaler remain
exact. Two fresh original baseline executions measure the gradient envelope.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

POLICIES = ('per-element', 'per-tensor-max-rms')
OPERATOR = 'grid_sampler_3d_backward_cuda'


class NativeGradientEnvelopeError(ValueError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__('Optimized gradients exceed the measured original native envelope: '
                         + evidence['policy'] + '; failed tensors=' + str(evidence['failed_tensors']))


def read_receipt(path, gate):
    path = Path(path)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.resolve(strict=True) != path.absolute()
            or hasattr(os, 'getuid') and path.stat().st_uid != os.getuid()):
        raise ValueError('Fresh regular owned probe receipt without symlinks required')
    before = path.stat(); checksum = gate._sha(path)
    row = json.loads(path.read_text(encoding='utf8')); after = path.stat()
    witness = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if witness(before) != witness(after) or gate._sha(path) != checksum:
        raise ValueError('Probe receipt changed during native admission')
    return row, dict(path=str(path), raw_sha256=checksum)


def validate_policy(row):
    policy = row.get('debug_numerics', {}); flags = policy.get('actual_forward_flags', {})
    warnings = row.get('native_grid_sampler_warning_receipt', {})
    if (policy.get('debug_native_grid_sampler_numerics') is not True
            or policy.get('debug_deterministic_numerics') is not True
            or policy.get('admitted_native_nondeterministic_operator') != OPERATOR
            or policy.get('unsupported_operator_fallback') is not False
            or policy.get('production_config_modified') is not False
            or flags.get('deterministic_algorithms') is not True
            or flags.get('deterministic_warn_only') is not True
            or flags.get('cudnn_deterministic') is not True
            or flags.get('cudnn_benchmark') is not False
            or flags.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'
            or warnings.get('operator') != OPERATOR
            or warnings.get('only_declared_operator') is not True
            or warnings.get('original_native_atomic_kernel') is not True
            or not warnings.get('warnings')
            or warnings.get('warning_count') != len(warnings['warnings'])):
        raise ValueError('Explicit native grid DEBUG policy and actual known warning required')
    for warning in warnings['warnings']:
        if (warning.get('category') != 'UserWarning'
                or not warning.get('message', '').startswith(OPERATOR + ' does not have a deterministic implementation, but you set ')
                or 'torch.use_deterministic_algorithms(True, warn_only=True)' not in warning['message']):
            raise ValueError('Unexpected operator warning cannot be admitted')


def metadata(rows, checkpoint, gate):
    a, b, o = rows
    if [row.get('variant') for row in rows] != ['baseline', 'baseline', 'optimized']:
        raise ValueError('Two separate original baselines and one optimized execution required')
    exact = ('original_training_identity_sha256', 'initial_model_sha256', 'initial_RNG_sha256',
             'output_sha256', 'loss', 'after_forward_model_sha256', 'after_RNG_sha256',
             'case_ids', 'physical_candidate_chunk', 'debug_numerics', 'ordered_CPU_input_sha256',
             'source_checkpoint', 'model_contract', 'initial_optimizer_sha256', 'after_optimizer_sha256',
             'initial_scaler_sha256', 'after_scaler_sha256')
    if any(not a.get(key) or any(row.get(key) != a[key] for row in rows[1:])
           for key in exact if key != 'loss') or any(row.get('loss') != a.get('loss') for row in rows[1:]):
        raise ValueError('Native input/forward/loss/model/RNG/checkpoint/runtime metadata differs')
    for row in rows:
        validate_policy(row)
        proof = row.get('input_tensor_proof', {})
        if (row.get('debug') is not True or row.get('GPU_arm') != 5
                or row.get('production_optimizer_updates') != 0
                or row.get('active_U') != 128 or row.get('all_P_included') is not True
                or row.get('physical_patient_batch') != 4 or row.get('physical_candidate_chunk') != 32
                or row.get('diagnostic_batches') != 1
                or row.get('case_ids') != ['liver_6', 'liver_129', 'liver_123', 'liver_69']
                or row.get('original_named_trainable_parameter_count') != gate.EXPECTED_GRADIENTS
                or row.get('optimizer_contains_all_trainable_parameters_exactly_once') is not True
                or row.get('scheduler_and_shuffle_state_unchanged') is not True
                or row.get('GPU5_probe_source_sha256') != gate._sha(gate.ROOT / 'tools/run_v24_gpu5_performance_probe.py')
                or row.get('source_checkpoint', {}).get('raw_sha256') != checkpoint
                or set(row['source_checkpoint'].get('numerical_state_sha256', {})) != gate.NUMERICAL_STATES
                or row['source_checkpoint'].get('optimizer_parameter_tensors') != gate.EXPECTED_GRADIENTS
                or row.get('gradient', {}).get('trainable_parameter_tensors') != gate.EXPECTED_GRADIENTS
                or row['gradient'].get('gradient_present') != gate.EXPECTED_GRADIENTS
                or row['gradient'].get('missing') != [] or row['gradient'].get('finite') is not True
                or proof.get('complete') is not True or proof.get('every_actual_value_hashed') is not True
                or proof.get('both_sampled_views') is not True or proof.get('ordered_records') != 524
                or proof.get('observed_native_chunks') != 17 or proof.get('expected_native_chunks') != 17
                or proof.get('observed_upper_graphs') != 4 or proof.get('expected_upper_graphs') != 4
                or proof.get('physical_candidate_chunk') != 32 or proof.get('epoch') != 1
                or len(proof.get('content', {}).get('local_chunks', [])) != 17
                or len(proof.get('content', {}).get('upper_graphs', [])) != 4
                or row.get('ordered_CPU_input_sha256') != proof.get('content_sha256')
                or type(proof.get('content_sha256')) is not str or len(proof['content_sha256']) != 64):
            raise ValueError('Complete original GPU5 all981-gradient/full128 input proof required')
        states = row['source_checkpoint']['numerical_state_sha256']
        if row['initial_model_sha256'] != states['model']:
            raise ValueError('Actual native model differs from the source checkpoint')
        for state in ('optimizer', 'scaler'):
            first = row['initial_' + state + '_sha256']
            if (type(first) is not str or len(first) != 64 or first != states[state]
                    or first != row['after_' + state + '_sha256']):
                raise ValueError('Actual native checkpoint ' + state + ' state changed')
    if o.get('input_runtime', {}).get('contract', {}).get('runtime_source_sha256') != gate._sha(gate.ROOT / 'hiercp_v1x/v24_input_runtime.py'):
        raise ValueError('Measured native input source differs from admitted training source')
    profile = o.get('input_runtime_profile', {})
    if any(profile.get(name) != 0 for name in ('source_live_chunks', 'source_cache_live_entries',
           'source_cache_live_key_bytes', 'source_cache_live_result_bytes', 'source_cache_live_tree_visible_array_bytes')):
        raise ValueError('Pure source chunk cache must be fully released')


def gradient_envelope(a, b, o, policy, gate):
    import torch
    if policy not in POLICIES:
        raise ValueError('Explicit supported empirical gradient envelope policy required')
    if list(a) != list(b) or list(a) != list(o) or len(a) != gate.EXPECTED_GRADIENTS:
        raise ValueError('All981 ordered original named gradient identities required')
    details = []; failed = []; all_bits = True
    for name, left in a.items():
        repeat = b[name]; optimized = o[name]
        if (left.dtype != repeat.dtype or left.dtype != optimized.dtype
                or left.shape != repeat.shape or left.shape != optimized.shape
                or not left.is_floating_point()):
            raise ValueError('Native gradient dtype/shape differs: ' + name)
        raw = lambda value: value.contiguous().reshape(-1).view(torch.uint8)
        same = bool(torch.equal(raw(left), raw(optimized))); all_bits &= same
        native = (repeat.to(torch.float64) - left.to(torch.float64)).abs()
        difference = (optimized.to(torch.float64) - left.to(torch.float64)).abs()
        if not bool(torch.isfinite(native).all() and torch.isfinite(difference).all()):
            raise ValueError('Nonfinite native gradient envelope: ' + name)
        native_max = float(native.max()) if native.numel() else 0.
        optimized_max = float(difference.max()) if difference.numel() else 0.
        native_rms = float(native.square().mean().sqrt()) if native.numel() else 0.
        optimized_rms = float(difference.square().mean().sqrt()) if difference.numel() else 0.
        zero = native == 0
        # Byte checks include signed zero where original measured variation is zero.
        zero_bytes_exact = bool(torch.equal(raw(left[zero]), raw(optimized[zero])))
        element_violations = int((difference > native).sum())
        element_pass = element_violations == 0 and zero_bytes_exact
        tensor_pass = optimized_max <= native_max and optimized_rms <= native_rms
        if native_max == 0:
            tensor_pass = tensor_pass and same and bool(torch.equal(raw(left), raw(repeat)))
        passed = element_pass if policy == 'per-element' else tensor_pass
        details.append(dict(name=name, shape=list(left.shape), dtype=str(left.dtype), elements=left.numel(),
            original_repeat_max_absolute_difference=native_max, optimized_max_absolute_difference=optimized_max,
            original_repeat_RMS_difference=native_rms, optimized_RMS_difference=optimized_rms,
            per_element_exceedances=element_violations, native_zero_variation_elements=int(zero.sum()),
            zero_variation_bits_exact=zero_bytes_exact, optimized_bits_exact=same,
            per_element_pass=element_pass, per_tensor_max_RMS_pass=tensor_pass, admitted=passed))
        if not passed: failed.append(name)
    evidence = dict(policy=policy, reference='two fresh original baseline executions; differences relative to baseline A',
        epsilon=0, multiplier=1, inferred_unobserved_noise_bound=False,
        bitwise_gradient_equality_guaranteed=all_bits, all_named_trainable_gradients=gate.EXPECTED_GRADIENTS,
        failed_tensors=failed, per_element_failed_tensors=sum(not row['per_element_pass'] for row in details),
        per_tensor_max_RMS_failed_tensors=sum(not row['per_tensor_max_RMS_pass'] for row in details),
        finite_all_actual_gradients=True, tensors=details,
        two_repeat_envelope_is_observed_diagnostic_not_a_universal_nondeterminism_bound=True)
    if failed:
        raise NativeGradientEnvelopeError(evidence)
    return evidence


def compare(baseline_path, repeat_path, optimized_path, checkpoint, *, policy, gate):
    paths = [Path(path).absolute() for path in (baseline_path, repeat_path, optimized_path)]
    if len(set(paths)) != 3:
        raise ValueError('Three distinct fresh probe receipt paths required')
    loaded = [read_receipt(path, gate) for path in paths]
    rows = [value[0] for value in loaded]; metadata(rows, checkpoint, gate)
    artifacts = [gate._read_actual_tensors(path, row) for path, row in zip(paths, rows)]
    a, b, o = [value[0] for value in artifacts]
    for values in (b, o):
        for index, (left, right) in enumerate(zip(a['scores'], values['scores'])):
            gate._same_bits(left, right, 'scores[' + str(index) + ']')
        for name in ('loss', 'consistency'):
            gate._same_bits(a[name], values[name], name)
    measured = gradient_envelope(a['gradients'], b['gradients'], o['gradients'], policy, gate)
    # Recheck reports and artifacts after the complete comparison.
    for path, (row, proof), (_, tensor_proof) in zip(paths, loaded, artifacts):
        if read_receipt(path, gate) != (row, proof) or gate._sha(tensor_proof['path']) != tensor_proof['raw_sha256']:
            raise ValueError('Native probe evidence changed during comparison')
    return dict(status='ACTUAL_FULL128_INPUT_FORWARD_LOSS_MODEL_RNG_EXACT_NATIVE_GRID_GRADIENT_ENVELOPE_PASS',
        GPU_arm=5, checkpoint_sha256=checkpoint, baseline_seconds=rows[0]['seconds'],
        baseline_repeat_seconds=rows[1]['seconds'], optimized_seconds=rows[2]['seconds'],
        ordered_CPU_input_sha256=rows[0]['ordered_CPU_input_sha256'],
        actual_numerical_tensor_values_bitwise_equal=measured['bitwise_gradient_equality_guaranteed'],
        input_forward_loss_and_model_RNG_state_bitwise_equal=True,
        all_named_trainable_gradients=gate.EXPECTED_GRADIENTS, gradient_envelope=measured,
        optimizer_and_scaler_unchanged=True, six_checkpoint_states_exact=True,
        measured_and_admitted_input_source_same=True, source_chunk_cache_fully_released=True,
        physical_candidate_chunk=32, physical_patient_batch=4, optimizer_updates=0,
        explicit_native_grid_sampler_DEBUG_policy=True, production_numerical_policy_changed=False,
        baseline_actual_tensor_file=artifacts[0][1], baseline_repeat_actual_tensor_file=artifacts[1][1],
        optimized_actual_tensor_file=artifacts[2][1], actual_probe_receipts=[value[1] for value in loaded],
        single_DEBUG_batch_not_steady_state_throughput=True, full_training_completion_claimed=False)
