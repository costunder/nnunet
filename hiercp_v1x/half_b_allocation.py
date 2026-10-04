"""Pure admission of explicitly selected half-B execution allocations.

The measured baseline batch, workers and accumulation stay fixed. Admission
does not measure throughput, initialize CUDA, change a model or create files.
"""
from __future__ import annotations

import json
import math
import re


POLICIES = ('same_allocation', 'current_allocation')
FORMAT = 'hiercp_half_B_allocation_admission_v1'
GPU_HARDWARE_FIELDS = ('name', 'total_vram_bytes', 'multiprocessors',
                       'compute_capability', 'mig_instance')
GPU_IDENTITY_FIELDS = ('name', 'multiprocessors', 'compute_capability', 'mig_instance')


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be an explicit resource object')
    return value


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}; got {value!r}')
    return value


def _budget(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or value <= 0):
        raise ValueError(f'{name} must be finite and positive')
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f'{name} must be finite and positive') from exc
    if not math.isfinite(number) or not math.isfinite(number * 2**30):
        raise ValueError(f'{name} must be finite and positive')
    size = int(number * 2**30)
    if size < 1:
        raise ValueError(f'{name} must declare at least one byte')
    return number, size


def _json_copy(value, name):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain finite JSON-safe resource facts') from exc


def _gpu(fingerprint, name):
    if fingerprint.get('selected_device') not in ('cuda', 'cuda:0'):
        raise ValueError(f'{name} must select CUDA device 0')
    if fingerprint.get('cuda_available') is not True:
        raise ValueError(f'{name} must report actual CUDA availability')
    if _integer(fingerprint.get('cuda_visible_device_count'),
                f'{name}.cuda_visible_device_count', minimum=1) != 1:
        raise ValueError(f'{name} must contain exactly one selected visible GPU')
    devices = fingerprint.get('gpu_devices')
    if not isinstance(devices, list) or len(devices) != 1:
        raise ValueError(f'{name}.gpu_devices must describe exactly one GPU')
    gpu = _object(devices[0], f'{name}.gpu_devices[0]')
    label = gpu.get('name')
    if (not isinstance(label, str) or not label.strip()
            or 'unavailable' in label.lower()):
        raise ValueError(f'{name} has no identified GPU name')
    _integer(gpu.get('total_vram_bytes'), f'{name}.total_vram_bytes', minimum=1)
    _integer(gpu.get('multiprocessors'), f'{name}.multiprocessors', minimum=1)
    capability = gpu.get('compute_capability')
    if (not isinstance(capability, str)
            or re.fullmatch(r'[1-9][0-9]*\.[0-9]+', capability) is None):
        raise ValueError(f'{name} has no identified compute capability')
    if type(gpu.get('mig_instance')) is not bool:
        raise ValueError(f'{name}.mig_instance must be an identified boolean')
    return {key: gpu[key] for key in GPU_HARDWARE_FIELDS}


def validate_allocation(report, expected_fingerprint, actual_fingerprint,
                        receipt, host_state):
    """Return an independent JSON-safe admission, or fail without a fallback.

    ``report`` is native TrainingResources, and ``host_state`` is the native
    preparation_runtime.snapshot() with effective cgroup/scheduler accounting.
    The full return includes volatile observations for an invocation receipt;
    an execution lock must instead bind the actual stable fingerprint and the
    preserved execution values. No hostname or UUID is treated as GPU capacity.
    """
    report = _object(report, 'Current resource report')
    expected_fingerprint = _object(expected_fingerprint, 'Baseline fingerprint')
    actual_fingerprint = _object(actual_fingerprint, 'Current fingerprint')
    receipt = _object(receipt, 'Half-B receipt')
    host_state = _object(host_state, 'Current host allocation')
    policy = receipt.get('allocation_policy', 'same_allocation')
    if policy not in POLICIES:
        raise ValueError(f'Unknown explicit allocation policy: {policy!r}')
    expected = _json_copy(expected_fingerprint, 'Baseline fingerprint')
    actual = _json_copy(actual_fingerprint, 'Current fingerprint')
    changed = {key: dict(old=expected.get(key), new=actual.get(key))
               for key in sorted(set(expected) | set(actual))
               if key not in expected or key not in actual or expected[key] != actual[key]}
    if policy == 'same_allocation' and changed:
        raise ValueError('Current allocation differs from baseline measurement; '
                         'changed_fields=' + json.dumps(changed, sort_keys=True,
                                                       allow_nan=False))

    baseline_gpu = _gpu(expected, 'Baseline fingerprint')
    current_gpu = _gpu(actual, 'Current fingerprint')
    # Capacity is an admission bound, not hardware identity. The current
    # device must fit the declared unchanged budget; a larger reported VRAM
    # total must not reject an otherwise matching GPU. same_allocation keeps
    # its exact fingerprint guard above, including the reported capacity.
    hardware_changes = {key: dict(old=baseline_gpu[key], new=current_gpu[key])
                        for key in GPU_IDENTITY_FIELDS
                        if baseline_gpu[key] != current_gpu[key]}
    if hardware_changes:
        raise ValueError('Selected GPU hardware differs from baseline; '
                         'changed_hardware=' + json.dumps(hardware_changes,
                                                         sort_keys=True))
    cuda_gib, cuda_bytes = _budget(receipt.get('cuda_gib'), 'cuda_gib')
    rss_gib, rss_bytes = _budget(receipt.get('rss_gib'), 'rss_gib')
    if cuda_bytes > current_gpu['total_vram_bytes']:
        raise ValueError('Declared CUDA budget exceeds selected GPU capacity')

    proof = _object(receipt.get('baseline_proof'), 'Verified baseline proof')
    execution = _object(proof.get('execution'), 'Verified baseline execution')
    bound_fingerprint = _object(execution.get('resource_fingerprint'),
                                'Verified baseline resource fingerprint')
    if _json_copy(bound_fingerprint, 'Verified baseline resource fingerprint') != expected:
        raise ValueError('Expected fingerprint differs from verified baseline execution')
    batch = _integer(execution.get('physical_batch'), 'Baseline physical batch', minimum=1)
    workers = _integer(execution.get('workers'), 'Baseline workers')
    config = _object(receipt.get('config'), 'Half-B config')
    training = _object(config.get('training'), 'Half-B training config')
    if _integer(training.get('batch_size'), 'Configured physical batch', minimum=1) != batch:
        raise ValueError('Configured physical batch differs from verified baseline')
    if _integer(training.get('num_workers'), 'Configured workers') != workers:
        raise ValueError('Configured workers differ from verified baseline')
    neural = _object(proof.get('neural_baseline'), 'Verified neural baseline')
    signature = _object(neural.get('training_signature'), 'Verified baseline training signature')
    accumulation = _integer(signature.get('gradient_accumulation_steps'),
                            'Baseline accumulation', minimum=1)
    if _integer(training.get('gradient_accumulation_steps'),
                'Configured accumulation', minimum=1) != accumulation:
        raise ValueError('Configured accumulation differs from verified baseline')

    capacity = _integer(host_state.get('cpu_capacity'), 'Current effective CPU capacity', minimum=1)
    rss = _integer(host_state.get('rss_bytes'), 'Current process RSS')
    available = _integer(host_state.get('available_memory_bytes'),
                         'Current available allocation memory')
    if capacity < max(1, workers):
        raise ValueError('Current CPU allocation cannot sustain the fixed worker count')
    if rss > rss_bytes:
        raise ValueError('Current process RSS exceeds the declared RSS budget')
    if rss + available < rss_bytes:
        raise ValueError('Current allocation has insufficient memory for the declared RSS budget')
    cuda_free = _integer(report.get('cuda_free_bytes'), 'Current CUDA free bytes')
    cuda_allocated = _integer(report.get('cuda_allocated_bytes'), 'Current CUDA allocated bytes')
    if cuda_free + cuda_allocated > current_gpu['total_vram_bytes']:
        raise ValueError('Current CUDA memory facts exceed identified GPU capacity')
    if cuda_allocated > cuda_bytes:
        raise ValueError('Current CUDA allocation exceeds the declared CUDA budget')
    if cuda_free + cuda_allocated < cuda_bytes:
        raise ValueError('Current selected GPU has insufficient memory for the declared CUDA budget')

    admission = dict(format=FORMAT, allocation_policy=policy, accepted=True,
        changed_fields=changed, baseline_resource_fingerprint=expected,
        actual_resource_fingerprint=actual,
        GPU_identity_verified=True,
        VRAM_capacity_comparison='current_declared_budget_not_baseline_equality',
        baseline_total_vram_bytes=baseline_gpu['total_vram_bytes'],
        current_total_vram_bytes=current_gpu['total_vram_bytes'],
        preserved_execution=dict(physical_batch=batch, workers=workers,
            gradient_accumulation_steps=accumulation, data_parallel_workers=1,
            effective_batch=batch * accumulation),
        declared_budgets=dict(cuda_gib=cuda_gib, rss_gib=rss_gib,
                              cuda_bytes=cuda_bytes, rss_bytes=rss_bytes),
        current_resources=report, current_host_allocation=host_state,
        matched_performance_claim=False, new_calibration=False)
    return _json_copy(admission, 'Allocation admission')
