"""GPU5 exact continuation with a complete981-gradient native parity gate.

The shared execution-only adapters retain the original numerical CLI, model,
data, sampling, physical patient batch4/candidate chunk32 and forty epochs.
The additional GPU5 gate reads the saved actual numerical tensor values and
requires exact inputs, forward values and checkpoint state. Strict mode also
requires exact gradients; the explicit native-grid mode bounds gradient
variation by measured original repeats. Disconnected parameters and state
updates are refused. Existing results and shared source are preserved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORMAT = 'v24_gpu5_execution_only_exact_continuation_v1'
RECEIPT = 'gpu5_performance_continuation.json'
EXPECTED_GRADIENTS = 981
NUMERICAL_STATES = {'model', 'optimizer', 'scheduler', 'scaler', 'rank_rng', 'shuffle_generator'}


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            result.update(block)
    return result.hexdigest()


def _publish(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _gpu5_args(args):
    if args.gpu != 5 or args.mode != 'train':
        raise ValueError('Exact original full training GPU5 arm required')


def runtime_identity():
    """Include the GPU5 gate/probe themselves in the immutable receipt."""
    from tools import run_v24_performance_continuation as shared
    files = (Path(__file__).resolve(), ROOT / 'tools/run_v24_gpu5_performance_probe.py',
             ROOT / 'tools/v24_gpu5_native_grid_gate.py')
    return dict(shared_runtime=shared.runtime_identity(),
                GPU5_files_sha256={str(path): _sha(path) for path in files},
                all_named_trainable_gradients=EXPECTED_GRADIENTS,
                physical_patient_batch=4, physical_candidate_chunk=32,
                epochs=40, production_numerical_policy_changed=False)


def prepare(args, source_output, source_code, source_interruption_proof=None):
    from tools import run_v24_performance_continuation as shared
    _gpu5_args(args)
    document = shared.prepare(args, source_output, source_code, source_interruption_proof)
    saved = document['source']['latest']
    if (set(saved['numerical_state_sha256']) != NUMERICAL_STATES
            or saved['optimizer_parameter_tensors'] != EXPECTED_GRADIENTS):
        raise ValueError('Original complete GPU5 six-state981-parameter checkpoint required')
    result = dict(format=FORMAT, destination=str(args.output.resolve()),
                  shared_receipt_sha256=_sha(args.output / shared.RECEIPT),
                  runtime=runtime_identity(), source_checkpoint=saved,
                  all_P_and128_U_preserved=True, production_optimizer_updates=0,
                  original_results_overwritten=False)
    _publish(args.output / RECEIPT, result)
    return result


def admit(args, source_output, source_code, source_interruption_proof=None):
    from tools import run_v24_performance_continuation as shared
    _gpu5_args(args)
    original, document = shared.admit(args, source_output, source_code, source_interruption_proof)
    own = json.loads((args.output / RECEIPT).read_text(encoding='utf8'))
    if (own.get('format') != FORMAT or own.get('destination') != str(args.output.resolve())
            or own.get('shared_receipt_sha256') != _sha(args.output / shared.RECEIPT)
            or own.get('runtime') != runtime_identity()
            or own.get('source_checkpoint') != document['source']['latest']
            or own.get('all_P_and128_U_preserved') is not True
            or own.get('production_optimizer_updates') != 0
            or own.get('original_results_overwritten') is not False):
        raise ValueError('Immutable GPU5 original checkpoint/runtime receipt differs')
    return original, document


def _read_actual_tensors(receipt_path, receipt):
    """Load only tensor dictionaries from an owned immutable probe output."""
    import torch
    from hiercp_v1x.u_bridge_training import digest
    receipt_path = Path(receipt_path).resolve(strict=True)
    entry = receipt.get('numerical_tensor_file', {})
    path = Path(entry.get('path', ''))
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.resolve(strict=True) != receipt_path.parent / 'numerical_tensors.pt'
            or entry.get('server_only') is not True or entry.get('no_raw_CT') is not True
            or entry.get('no_model_update') is not True
            or entry.get('all_named_trainable_gradients') != EXPECTED_GRADIENTS):
        raise ValueError('Actual complete GPU5 server-local981-gradient tensor file required')
    if hasattr(os, 'getuid') and path.stat().st_uid != os.getuid():
        raise ValueError('Probe tensors must be owned by the current server user')
    before = path.stat()
    expected_stat = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    raw_sha = _sha(path)
    if before.st_size != entry.get('size') or raw_sha != entry.get('raw_sha256'):
        raise ValueError('Actual numerical tensor file size/hash changed')
    values = torch.load(path, map_location='cpu', weights_only=True)
    after = path.stat()
    if expected_stat != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('Actual numerical tensor snapshot changed while reading')
    if (type(values) is not dict or set(values) != {'scores', 'consistency', 'loss', 'gradients'}
            or type(values['scores']) is not tuple or len(values['scores']) != 4
            or type(values['gradients']) is not dict or len(values['gradients']) != EXPECTED_GRADIENTS
            or any(type(key) is not str for key in values['gradients'])):
        raise ValueError('Complete native score/loss/consistency and named981 gradients required')
    leaves = (*values['scores'], values['consistency'], values['loss'], *values['gradients'].values())
    if any(type(value) is not torch.Tensor or value.layout != torch.strided
           or value.device.type != 'cpu' or not bool(torch.isfinite(value).all()) for value in leaves):
        raise ValueError('Every actual native numerical tensor must be finite and CPU-strided')
    if values['loss'].numel() != 1 or float(values['loss']) != receipt.get('loss'):
        raise ValueError('Saved actual numerical loss differs from probe receipt')
    upper = receipt['input_tensor_proof']['content']['upper_graphs']
    if (any(score.ndim != 1 or score.numel() != len(row.get('record_ids', []))
            for score, row in zip(values['scores'], upper))
            or sum(score.numel() for score in values['scores']) != 524):
        raise ValueError('Actual scores must cover all524 ordered native candidates')
    if (digest(values) != entry.get('content_sha256')
            or digest(values['scores']) != receipt.get('output_sha256')
            or digest(values['gradients']) != receipt.get('gradient_sha256')):
        raise ValueError('Actual tensor values differ from the numerical receipt')
    return values, dict(path=str(path), raw_sha256=raw_sha,
                        content_sha256=entry['content_sha256'], all_named_gradients=EXPECTED_GRADIENTS)


def _same_bits(left, right, name):
    import torch
    if left.dtype != right.dtype or left.shape != right.shape:
        raise ValueError('Actual native numerical dtype/shape differs: ' + name)
    a = left.contiguous().reshape(-1).view(torch.uint8)
    b = right.contiguous().reshape(-1).view(torch.uint8)
    if not torch.equal(a, b):
        raise ValueError('Actual native numerical bits differ: ' + name)


def compare_probes(baseline_path, optimized_path, checkpoint_sha256, *, baseline_repeat_path=None,
                   debug_native_grid_sampler_numerics=False, gradient_envelope_policy='per-element'):
    if debug_native_grid_sampler_numerics:
        if baseline_repeat_path is None:
            raise ValueError('Explicit native grid admission requires a separate original baseline repeat')
        from tools import v24_gpu5_native_grid_gate as native_grid
        return native_grid.compare(baseline_path, baseline_repeat_path, optimized_path, checkpoint_sha256,
                                   policy=gradient_envelope_policy, gate=sys.modules[__name__])
    if baseline_repeat_path is not None or gradient_envelope_policy != 'per-element':
        raise ValueError('Native repeat/envelope options require explicit native grid DEBUG opt-in')
    from tools import run_v24_performance_continuation as shared
    baseline = json.loads(Path(baseline_path).read_text(encoding='utf8'))
    optimized = json.loads(Path(optimized_path).read_text(encoding='utf8'))
    comparison = shared.compare_probes(baseline, optimized, checkpoint_sha256)
    if (optimized.get('input_runtime', {}).get('contract', {}).get('runtime_source_sha256')
            != _sha(ROOT / 'hiercp_v1x/v24_input_runtime.py')):
        raise ValueError('Measured GPU5 input runtime source must equal the admitted training source')
    profile = optimized.get('input_runtime_profile', {})
    if any(profile.get(name) != 0 for name in ('source_live_chunks', 'source_cache_live_entries',
           'source_cache_live_key_bytes', 'source_cache_live_result_bytes',
           'source_cache_live_tree_visible_array_bytes')):
        raise ValueError('GPU5 pure source chunk cache must be fully released before native gate admission')
    for row in (baseline, optimized):
        if (row.get('GPU_arm') != 5 or row.get('physical_candidate_chunk') != 32
                or row.get('original_named_trainable_parameter_count') != EXPECTED_GRADIENTS
                or row.get('optimizer_contains_all_trainable_parameters_exactly_once') is not True
                or row.get('scheduler_and_shuffle_state_unchanged') is not True
                or row.get('GPU5_probe_source_sha256') != _sha(ROOT / 'tools/run_v24_gpu5_performance_probe.py')
                or row.get('gradient', {}).get('trainable_parameter_tensors') != EXPECTED_GRADIENTS
                or row.get('gradient', {}).get('gradient_present') != EXPECTED_GRADIENTS
                or set(row.get('source_checkpoint', {}).get('numerical_state_sha256', {})) != NUMERICAL_STATES
                or row.get('source_checkpoint', {}).get('optimizer_parameter_tensors') != EXPECTED_GRADIENTS):
            raise ValueError('Original GPU5 chunk32/six-state/all981-gradient proof required')
        for state in ('optimizer', 'scaler'):
            first = row.get('initial_' + state + '_sha256')
            if (type(first) is not str or len(first) != 64
                    or first != row.get('after_' + state + '_sha256')
                    or first != row['source_checkpoint']['numerical_state_sha256'][state]):
                raise ValueError('DEBUG persisted ' + state + ' state must remain unchanged')
        if row['initial_model_sha256'] != row['source_checkpoint']['numerical_state_sha256']['model']:
            raise ValueError('Actual model must be restored exactly from the original GPU5 checkpoint')
    keys = ('initial_optimizer_sha256', 'after_optimizer_sha256',
            'initial_scaler_sha256', 'after_scaler_sha256')
    if (baseline['source_checkpoint'] != optimized['source_checkpoint']
            or not baseline.get('model_contract')
            or baseline['model_contract'] != optimized.get('model_contract')
            or any(baseline[key] != optimized[key] for key in keys)):
        raise ValueError('The exact original six-state checkpoint/AdamW/scaler differs between probes')
    left, left_proof = _read_actual_tensors(baseline_path, baseline)
    right, right_proof = _read_actual_tensors(optimized_path, optimized)
    if list(left['gradients']) != list(right['gradients']):
        raise ValueError('Original named gradient identity/order differs')
    for index, (a, b) in enumerate(zip(left['scores'], right['scores'])):
        _same_bits(a, b, 'scores[' + str(index) + ']')
    for name in ('loss', 'consistency'):
        _same_bits(left[name], right[name], name)
    for name in left['gradients']:
        _same_bits(left['gradients'][name], right['gradients'][name], name)
    comparison.update(GPU_arm=5, actual_numerical_tensor_values_bitwise_equal=True,
                      all_named_trainable_gradients=EXPECTED_GRADIENTS,
                      optimizer_and_scaler_unchanged=True, six_checkpoint_states_exact=True,
                      measured_and_admitted_input_source_same=True, source_chunk_cache_fully_released=True,
                      baseline_actual_tensor_file=left_proof, optimized_actual_tensor_file=right_proof,
                      physical_candidate_chunk=32, physical_patient_batch=4,
                      optimizer_updates=0, single_DEBUG_batch_not_steady_state_throughput=True)
    return comparison


def _guarded_training(shared, native_comparator=None):
    """Private original execution code; bind GPU5 file proofs to each forward."""
    def bounded_admit(*args, **kwargs):
        original, document = admit(*args, **kwargs)
        private = copy.deepcopy(document)
        own = runtime_identity()['GPU5_files_sha256']
        private['runtime']['files_sha256'].update(own)
        private['runtime']['GPU5_exact_gate_source_guard'] = True
        return original, private
    namespace = dict(shared.train.__globals__)
    namespace['admit'] = bounded_admit
    if native_comparator is not None:
        namespace['compare_probes'] = native_comparator
    return FunctionType(shared.train.__code__, namespace, shared.train.__name__,
                        shared.train.__defaults__, shared.train.__closure__)


def train(args, source_output, source_code, probe_baseline, probe_optimized,
          source_interruption_proof=None, *, probe_baseline_repeat=None,
          debug_native_grid_sampler_numerics=False, gradient_envelope_policy='per-element'):
    from tools import run_v24_performance_continuation as shared
    _, document = admit(args, source_output, source_code, source_interruption_proof)
    if probe_baseline is None or probe_optimized is None:
        raise ValueError('Both actual full128 GPU5 native probe receipts required')
    comparison_options = dict(baseline_repeat_path=probe_baseline_repeat,
        debug_native_grid_sampler_numerics=debug_native_grid_sampler_numerics,
        gradient_envelope_policy=gradient_envelope_policy)
    checkpoint = document['source']['latest']['raw_sha256']
    from tools.v24_gpu5_native_grid_gate import NativeGradientEnvelopeError
    try:
        comparison = compare_probes(probe_baseline, probe_optimized, checkpoint, **comparison_options)
    except NativeGradientEnvelopeError as error:
        _publish(args.output / ('gpu5_native_gradient_envelope_failure_' + str(time.time_ns()) + '.json'),
            dict(debug=True, admitted=False, checkpoint_sha256=checkpoint, evidence=error.evidence,
                 baseline_file_sha256=_sha(probe_baseline), baseline_repeat_file_sha256=_sha(probe_baseline_repeat),
                 optimized_file_sha256=_sha(probe_optimized), production_started=False,
                 production_config_modified=False, silent_fallback=False))
        raise
    _publish(args.output / 'gpu5_actual_native_probe_admission.json', dict(comparison=comparison,
        GPU5_receipt_sha256=_sha(args.output / RECEIPT),
        baseline_file_sha256=_sha(probe_baseline), optimized_file_sha256=_sha(probe_optimized),
        baseline_repeat_file_sha256=_sha(probe_baseline_repeat) if probe_baseline_repeat is not None else None))
    def native_comparator(baseline, optimized, expected_checkpoint):
        from tools import v24_gpu5_native_grid_gate as native_grid
        if (expected_checkpoint != checkpoint
                or native_grid.read_receipt(probe_baseline, sys.modules[__name__])[0] != baseline
                or native_grid.read_receipt(probe_optimized, sys.modules[__name__])[0] != optimized):
            raise ValueError('Shared production invocation received different native grid evidence')
        # Re-admit actual values; no DEBUG kernel flags are installed in production.
        return compare_probes(probe_baseline, probe_optimized, checkpoint, **comparison_options)
    native = _guarded_training(shared, native_comparator if debug_native_grid_sampler_numerics else None)
    if native.__code__ is not shared.train.__code__:
        raise ValueError('Shared original full training execution code must remain identical')
    return native(args, source_output, source_code, probe_baseline, probe_optimized,
                  source_interruption_proof)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--action', choices=('prepare', 'train'), required=True)
    parser.add_argument('--source-output', type=Path, required=True)
    parser.add_argument('--source-code', type=Path, required=True)
    parser.add_argument('--probe-baseline', type=Path)
    parser.add_argument('--probe-optimized', type=Path)
    parser.add_argument('--probe-baseline-repeat', type=Path)
    parser.add_argument('--debug-native-grid-sampler-numerics', action='store_true')
    parser.add_argument('--gradient-native-noise-policy', '--debug-gradient-envelope-policy',
                        dest='debug_gradient_envelope_policy', choices=('per-element', 'per-tensor-max-rms'),
                        default='per-element')
    parser.add_argument('--source-interruption-proof', type=Path)
    options, remaining = parser.parse_known_args(argv)
    from tools.run_v24_all_p import parse
    args = parse(remaining)
    _gpu5_args(args)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if options.action == 'prepare' else '5'
    if options.action == 'prepare':
        result = prepare(args, options.source_output, options.source_code, options.source_interruption_proof)
        print(json.dumps(result, allow_nan=False), flush=True)
        return result
    return train(args, options.source_output, options.source_code,
                 options.probe_baseline, options.probe_optimized, options.source_interruption_proof,
                 probe_baseline_repeat=options.probe_baseline_repeat,
                 debug_native_grid_sampler_numerics=options.debug_native_grid_sampler_numerics,
                 gradient_envelope_policy=options.debug_gradient_envelope_policy)


if __name__ == '__main__':
    main()
