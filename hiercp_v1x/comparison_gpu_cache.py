"""Bounded execution-policy reuse across resumes, separate from checkpoints.

This is an execution cache, not a learned model or a memory-safety proof. Only
new, context-bound measurements enter it; historical JSONL is never imported.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

from .comparison_storage import report_secondary_failure, storage_error

FORMAT = 'comparison_gpu_policy_cache_v1'
GRAPHS = ('local_batch', 'local_batch_view2', 'patient_batch', 'prototype_batch')
POLICY_KEYS = {'dense_batch_size', 'checkpoint_dense_encoder', 'checkpoint_local_blocks'}
CONTEXT_KEYS = {'identity_sha256', 'cuda_environment', 'torch_version', 'precision',
                'implementation_sha256', 'model_structure_sha256', 'loss_identity_sha256'}


class PolicyCacheError(ValueError):
    """A present cache is invalid; keep it intact and report the exact cause."""


def _json(value):
    def validate(item):
        if isinstance(item, dict):
            if any(type(key) is not str or not key for key in item):
                raise PolicyCacheError('Cache JSON requires nonempty string keys')
            for child in item.values(): validate(child)
        elif isinstance(item, list):
            for child in item: validate(child)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise PolicyCacheError('Cache values must be explicit JSON data')
        elif type(item) is float and not math.isfinite(item):
            raise PolicyCacheError('Cache values must be finite')
    validate(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(_json(value).encode('utf8')).hexdigest()


def _hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_policy(value):
    if (not isinstance(value, dict) or set(value) != POLICY_KEYS
            or type(value['dense_batch_size']) is not int or value['dense_batch_size'] < 1
            or any(type(value[key]) is not bool for key in POLICY_KEYS - {'dense_batch_size'})):
        raise PolicyCacheError('Cache policy requires a positive dense chunk and two boolean checkpoint flags')


def _validate_chunk(inventory, selected, original):
    choices = [original['dense_batch_size']]
    patches = max(inventory['source_patch_shape.0'], inventory['target_patch_shape.0'])
    while choices[-1] < patches:
        choices.append(min(choices[-1] * 2, patches))
    if selected['dense_batch_size'] not in choices:
        raise PolicyCacheError('Cached dense chunk was not an admissible complete-batch measurement candidate')


def validate_inventory(value):
    if (not isinstance(value, dict) or not value
            or any(type(key) is not str or type(count) is not int or count < 0
                   for key, count in value.items())):
        raise PolicyCacheError('Cache workload requires named nonnegative integer counts')
    expected = {'source_problems'}
    for name in ('source_patch_shape', 'target_patch_shape'):
        keys = [f'{name}.{i}' for i in range(5)]
        expected.update(keys + [name + '.elements'])
        if any(value.get(key, 0) < 1 for key in keys):
            raise PolicyCacheError('Cache workload requires complete positive five-dimensional patches')
        if math.prod(value[key] for key in keys) != value.get(name + '.elements'):
            raise PolicyCacheError('Cache patch element count differs from its shape')
    if (value.get('source_problems', 0) < 1
            or value['source_patch_shape.0'] != value['source_problems']
            or value['target_patch_shape.0'] != 8 * value['source_problems']):
        raise PolicyCacheError('Cache must preserve complete eight-candidate source batches')
    for name in GRAPHS:
        expected.add(name + '.graphs')
        if value.get(name + '.graphs', 0) < 1:
            raise PolicyCacheError('Cache requires all four actual graph batches')
        for kind, types in (('nodes', 'node_types'), ('edges', 'edge_types')):
            required = [f'{name}.{kind}.{stat}' for stat in ('total', 'max')]
            expected.update(required)
            prefix = f'{name}.{types}.'
            typed = [key for key in value if key.startswith(prefix) and len(key) > len(prefix)]
            expected.update(typed)
            if (not typed or any(key not in value for key in required)
                    or value[required[1]] > value[required[0]]
                    or sum(value[key] for key in typed) != value[required[0]]):
                raise PolicyCacheError('Cache graph totals, maxima and typed inventory disagree')
    if set(value) != expected:
        raise PolicyCacheError('Cache workload has unknown or incomplete shape/inventory fields')


def bind_context(context, *, amp, cuda_limit_bytes, reserve_bytes, original):
    if not isinstance(context, dict) or not CONTEXT_KEYS <= set(context):
        raise PolicyCacheError('Explicit identity, CUDA, precision, implementation, model and loss cache context required')
    _json(context)
    if any(not _hash(context[name]) for name in CONTEXT_KEYS if name.endswith('_sha256')):
        raise PolicyCacheError('Cache identity/model/loss/implementation fingerprints must be SHA256 values')
    environment = context['cuda_environment']
    capability = environment.get('compute_capability') if isinstance(environment, dict) else None
    valid_capability = (type(capability) is str and bool(re.fullmatch(r'\d+\.\d+', capability))
                        or isinstance(capability, list) and len(capability) == 2
                        and all(type(part) is int and part >= 0 for part in capability))
    if (not isinstance(environment, dict)
            or not {'device_uuid', 'device_name', 'total_memory_bytes', 'compute_capability', 'cuda_runtime'} <= set(environment)
            or any(type(environment[k]) is not str or not environment[k]
                   for k in ('device_uuid', 'device_name', 'cuda_runtime'))
            or not valid_capability
            or type(environment['total_memory_bytes']) is not int or environment['total_memory_bytes'] <= 0):
        raise PolicyCacheError('Actual GPU identity, total memory, capability and CUDA runtime required')
    if (type(context['torch_version']) is not str or not context['torch_version']
            or not isinstance(context['precision'], dict) or not context['precision']):
        raise PolicyCacheError('Explicit PyTorch version and precision context required')
    if (type(amp) is not bool or type(cuda_limit_bytes) is not int or cuda_limit_bytes <= 0
            or reserve_bytes is not None and (type(reserve_bytes) is not int or reserve_bytes < 0)):
        raise PolicyCacheError('Explicit valid AMP, CUDA budget and reserve required')
    validate_policy(original)
    return dict(context=copy.deepcopy(context), amp=amp, cuda_limit_bytes=cuda_limit_bytes,
                reserve_bytes=reserve_bytes, original_policy=copy.deepcopy(original))


def measured_entry(inventory, selected, receipt, binding):
    """Retain a compact proof; the complete numerical receipt stays in audit."""
    validate_inventory(inventory); validate_policy(selected)
    _validate_chunk(inventory, selected, binding['original_policy'])
    _json(receipt)
    required_true = ('model_gradients_rng_modes_and_settings_restored', 'optimizer_state_untouched',
                     'original_source_batch_and_candidate_coverage_unchanged')
    if (not isinstance(receipt, dict) or receipt.get('format') != 'comparison_gpu_execution_policy_v1'
            or receipt.get('calibration_status') not in ('measured', 'original_retained_no_headroom')
            or receipt.get('calibration_status') == 'original_retained_no_headroom'
               and selected != binding['original_policy']
            or type(receipt.get('optimizer_updates')) is not int or receipt['optimizer_updates'] != 0
            or any(receipt.get(key) is not True for key in required_true)
            or receipt.get('amp') is not binding['amp']
            or receipt.get('declared_cuda_limit_bytes') != binding['cuda_limit_bytes']
            or receipt.get('original_policy') != binding['original_policy']
            or receipt.get('selected_policy') != selected):
        raise PolicyCacheError('Only complete measured, state-preserving execution receipts may be cached')
    for key, minimum in (('warmup_repetitions', 1), ('measured_repetitions', 2)):
        if type(receipt.get(key)) is not int or receipt[key] < minimum:
            raise PolicyCacheError('Cache cannot admit reduced calibration evidence')
    reports = receipt.get('reports')
    passing = [row for row in reports if isinstance(row, dict)
               and row.get('accepted') is True and row.get('policy') == selected] if isinstance(reports, list) else []
    if len(passing) != 1:
        raise PolicyCacheError('Cached selection must have exactly one accepted measured policy report')
    report = passing[0]
    seconds = report.get('forward_backward_seconds')
    peak = report.get('peak_cuda_bytes')
    trials = report.get('trials')
    if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
            or type(peak) is not int or not 0 < peak <= binding['cuda_limit_bytes']
            or not isinstance(trials, list)
            or len(trials) != receipt['warmup_repetitions'] + receipt['measured_repetitions']
            or any(not isinstance(trial, dict) or not isinstance(trial.get('equivalence'), dict)
                   or trial['equivalence'].get('accepted') is not True for trial in trials)):
        raise PolicyCacheError('Cached policy lacks complete finite, accepted forward/backward trial evidence')
    proof = {key: receipt[key] for key in (*required_true, 'optimizer_updates', 'warmup_repetitions', 'measured_repetitions')}
    proof.update(calibration_status=receipt['calibration_status'],
                 selected_forward_backward_seconds=seconds, selected_peak_cuda_bytes=peak)
    return dict(inventory=copy.deepcopy(inventory), selected=copy.deepcopy(selected),
                workload_sha256=digest(inventory), calibration_sha256=digest(receipt), proof=proof)


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise PolicyCacheError(f'Duplicate cache JSON key: {key}')
        result[key] = value
    return result


class ExecutionPolicyCache:
    def __init__(self, output, binding):
        self.binding = copy.deepcopy(binding)
        self.context_sha256 = digest(binding)
        self.path = Path(output) / 'gpu_execution_policy_cache' / (self.context_sha256 + '.json')
        self._previous_digest = None

    def _decode(self):
        if self.path.parent.is_symlink():
            raise PolicyCacheError(f'Refusing symlink GPU cache directory: {self.path.parent}')
        if self.path.is_symlink(): raise PolicyCacheError(f'Refusing symlink policy cache: {self.path}')
        try:
            payload = json.loads(self.path.read_text(encoding='utf8'), object_pairs_hook=_no_duplicates)
            _json(payload)
        except (ValueError, UnicodeError) as error:
            raise PolicyCacheError(f'Invalid GPU execution sidecar {self.path}: {error}') from error
        if (not isinstance(payload, dict)
                or set(payload) != {'format', 'binding', 'context_sha256', 'measured', 'rejected', 'content_sha256'}):
            raise PolicyCacheError(f'Invalid GPU cache document schema: {self.path}')
        signed = dict(payload); signature = signed.pop('content_sha256')
        if (payload['format'] != FORMAT or payload['binding'] != self.binding
                or payload['context_sha256'] != self.context_sha256
                or not _hash(signature) or digest(signed) != signature):
            raise PolicyCacheError(f'GPU cache compatibility/checksum mismatch: {self.path}')
        self._validate_entries(payload['measured'], payload['rejected'])
        return payload

    def _validate_entries(self, measured, rejected):
        if not isinstance(measured, list) or not isinstance(rejected, list):
            raise PolicyCacheError('Measured/rejected cache entries must be explicit lists')
        seen = set()
        proof_keys = {'model_gradients_rng_modes_and_settings_restored', 'optimizer_state_untouched',
                      'original_source_batch_and_candidate_coverage_unchanged', 'optimizer_updates',
                      'warmup_repetitions', 'measured_repetitions',
                      'selected_forward_backward_seconds', 'selected_peak_cuda_bytes', 'calibration_status'}
        for row in measured:
            if not isinstance(row, dict) or set(row) != {'inventory', 'selected', 'workload_sha256', 'calibration_sha256', 'proof'}:
                raise PolicyCacheError('Invalid measured cache entry schema')
            validate_inventory(row['inventory']); validate_policy(row['selected'])
            _validate_chunk(row['inventory'], row['selected'], self.binding['original_policy'])
            proof = row['proof']
            if (not _hash(row['calibration_sha256']) or not isinstance(proof, dict) or set(proof) != proof_keys
                    or any(proof[k] is not True for k in proof_keys if k not in {'optimizer_updates', 'warmup_repetitions', 'measured_repetitions', 'selected_forward_backward_seconds', 'selected_peak_cuda_bytes', 'calibration_status'})
                    or type(proof['optimizer_updates']) is not int or proof['optimizer_updates'] != 0
                    or type(proof['warmup_repetitions']) is not int or proof['warmup_repetitions'] < 1
                    or type(proof['measured_repetitions']) is not int or proof['measured_repetitions'] < 2):
                raise PolicyCacheError('Invalid cache measurement proof')
            seconds, peak = proof['selected_forward_backward_seconds'], proof['selected_peak_cuda_bytes']
            if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
                    or type(peak) is not int or not 0 < peak <= self.binding['cuda_limit_bytes']):
                raise PolicyCacheError('Invalid cached finite timing or CUDA peak measurement')
            if (proof['calibration_status'] not in ('measured', 'original_retained_no_headroom')
                    or proof['calibration_status'] == 'original_retained_no_headroom'
                    and row['selected'] != self.binding['original_policy']):
                raise PolicyCacheError('No-headroom cache admission must retain the original execution policy')
            signature = row['workload_sha256']
            if signature != digest(row['inventory']) or signature in seen:
                raise PolicyCacheError('Duplicate or mismatched cache workload fingerprint')
            seen.add(signature)
        rejected_seen = set()
        for row in rejected:
            if not isinstance(row, dict) or set(row) != {'workload_sha256', 'inventory'}:
                raise PolicyCacheError('Invalid rejected cache entry schema')
            validate_inventory(row['inventory'])
            signature = row['workload_sha256']
            if signature != digest(row['inventory']) or signature in rejected_seen:
                raise PolicyCacheError('Duplicate or mismatched rejected-workload fingerprint')
            rejected_seen.add(signature)

    def load(self):
        if not self.path.exists() and not self.path.is_symlink(): return [], []
        payload = self._decode()
        self._previous_digest = payload['content_sha256']
        return copy.deepcopy(payload['measured']), copy.deepcopy(payload['rejected'])

    def publish(self, measured, rejected):
        self._validate_entries(measured, rejected)
        payload = dict(format=FORMAT, binding=self.binding, context_sha256=self.context_sha256,
                       measured=measured, rejected=rejected)
        payload['content_sha256'] = digest(payload)
        raw = (_json(payload) + '\n').encode('utf8')
        temporary = self.path.parent / ('.' + self.path.stem + '.' + uuid.uuid4().hex + '.tmp')
        lock = self.path.with_suffix('.lock')
        own_lock = own_temporary = False
        primary = None
        try:
            if self.path.parent.is_symlink():
                raise PolicyCacheError(f'Refusing symlink GPU cache directory: {self.path.parent}')
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with lock.open('xb'):
                own_lock = True
            if self.path.exists() or self.path.is_symlink():
                current = self._decode()
                if current['content_sha256'] != self._previous_digest:
                    raise PolicyCacheError(f'Concurrent GPU policy cache change; no overwrite: {self.path}')
            elif self._previous_digest is not None:
                raise PolicyCacheError(f'GPU policy cache disappeared; no replacement: {self.path}')
            with temporary.open('xb') as stream:
                own_temporary = True
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            own_temporary = False
            if os.name != 'nt':
                descriptor = os.open(self.path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
                try: os.fsync(descriptor)
                finally: os.close(descriptor)
            self._previous_digest = payload['content_sha256']
        except OSError as error:
            primary = storage_error(error, operation='GPU execution sidecar publish', path=self.path)
            raise primary from error
        except Exception as error:
            primary = error
            raise
        finally:
            for owned, path in ((own_temporary, temporary), (own_lock, lock)):
                if owned:
                    try: path.unlink()
                    except OSError as error:
                        if primary is None: raise
                        report_secondary_failure(primary, error, operation='own GPU cache temporary cleanup')
