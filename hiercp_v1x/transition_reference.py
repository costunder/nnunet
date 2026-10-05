"""Read-only metadata admission of the saved native comparison endpoint.

This module does not construct a network, transfer weights, run a forward pass,
or certify predictive quality. The trusted own-run checkpoint reader uses CPU
FakeTensorMode; the receipt binds file bytes and the recorded recipe/cursor.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from .native_transition_receipt import (
    checkpoint_metadata, native_execution_bindings, required_source_files, select_checkpoint,
)

FORMAT = 'v17_native_reference_admission_v1'
CHECKPOINT_FORMAT = 'fixed_region_sage_training_v1'
POLICY = 'same_donor_live_v1'
_SHA = re.compile(r'^[0-9a-f]{64}$')


class ReferenceContractError(ValueError):
    """Required metadata is absent, malformed, or does not bind this endpoint."""

    def __init__(self, missing=(), failed=()):
        self.required_fields_missing = sorted(set(missing))
        self.failed_bindings = sorted(set(failed))
        parts = []
        if self.required_fields_missing:
            parts.append('required metadata missing: ' + ', '.join(self.required_fields_missing))
        if self.failed_bindings:
            parts.append('reference binding failed: ' + ', '.join(self.failed_bindings))
        super().__init__('; '.join(parts))


def _canonical(value):
    # JSON identity distinguishes True from 1, unlike Python mapping equality.
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_reference_metadata(manifest, inventory, receipt, execution, schedule, *, inventory_sha256):
    """Pure admission of an actual CPU metadata-reader receipt.

    The inventory and manifest are the actual run files, not proposed defaults.
    Partial native training is admitted as a *partial reference*, with its exact
    saved cursor. Production recipe remains full40/debug=False. No tensor values
    or model/optimizer/state tensor summaries are returned.
    """
    missing, failed, checks = [], [], {}

    def mapping(value, label, keys):
        if not isinstance(value, dict):
            missing.append(label)
            return {}
        for key in keys:
            if key not in value:
                missing.append(label + '.' + key)
        return value

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            failed.append(name)

    def equal(name, actual, expected):
        try:
            matches = _canonical(actual) == _canonical(expected)
        except (TypeError, ValueError):
            matches = False
        check(name, matches)

    manifest = mapping(manifest, 'manifest', ('format', 'request', 'attempts'))
    request = mapping(manifest.get('request'), 'request', ('source', 'local_cnn', 'original_inventory_sha256', 'settings'))
    inventory = mapping(inventory, 'inventory', ('format', 'complete', 'debug', 'config', 'base',
                        'local_cnn', 'learning_policy', 'original_inventory_sha256'))
    cfg = mapping(inventory.get('config'), 'inventory.config', ('gnn_epochs',))
    base = mapping(inventory.get('base'), 'inventory.base', ())
    cnn = mapping(inventory.get('local_cnn'), 'inventory.local_cnn', ('architecture', 'margin_mm', 'learning_policy'))
    receipt = mapping(receipt, 'checkpoint_receipt', ('path', 'sha256', 'bytes', 'loading', 'tensor_values_exported', 'payload'))
    payload = mapping(receipt.get('payload'), 'checkpoint', ('format', 'identity', 'model', 'optimizer', 'state', 'rng', 'content_sha256'))
    identity = mapping(payload.get('identity'), 'checkpoint.identity', ('format', 'cache_sha256', 'local_cnn',
                       'config', 'base', 'learning_policy', 'source', 'epochs', 'debug', 'precision'))
    state = mapping(payload.get('state'), 'checkpoint.state', ('epoch', 'step', 'phase', 'next_batch', 'batch'))
    execution = mapping(execution, 'execution_contract', ())
    schedule = mapping(schedule, 'learning_schedule', ('policy', 'optimization_steps'))

    check('experiment_format', manifest.get('format') == 'local_cnn_experiment_v1')
    check('experiment_attempts_present', isinstance(manifest.get('attempts'), list) and bool(manifest['attempts']))
    check('inventory_format', inventory.get('format') == 'native_local_cnn_inventory_v1')
    check('inventory_complete', inventory.get('complete') is True)
    check('production_not_debug', inventory.get('debug') is False and identity.get('debug') is False)
    check('checkpoint_format', payload.get('format') == CHECKPOINT_FORMAT and identity.get('format') == CHECKPOINT_FORMAT)
    check('production_target40', type(cfg.get('gnn_epochs')) is int and cfg['gnn_epochs'] == 40
          and type(identity.get('epochs')) is int and identity['epochs'] == 40)
    check('native_local_cnn', cnn.get('architecture') == 'paired_native_local_cnn_v1')
    check('ten_mm_context', type(cnn.get('margin_mm')) in (int, float) and cnn['margin_mm'] == 10)
    check('inventory_config_nonempty', bool(cfg))
    check('inventory_base_nonempty', bool(base))
    check('native_learning_policy', inventory.get('learning_policy') == POLICY
          and identity.get('learning_policy') == POLICY and cnn.get('learning_policy') == POLICY
          and schedule.get('policy') == POLICY)
    check('native_precision', identity.get('precision') == 'FP32')
    check('inventory_sha256_valid', isinstance(inventory_sha256, str) and bool(_SHA.fullmatch(inventory_sha256)))
    check('checkpoint_file_sha256_valid', isinstance(receipt.get('sha256'), str)
          and bool(_SHA.fullmatch(receipt['sha256'])))
    check('checkpoint_recorded_content_sha256_valid', isinstance(payload.get('content_sha256'), str)
          and bool(_SHA.fullmatch(payload['content_sha256'])))
    check('checkpoint_bytes_valid', type(receipt.get('bytes')) is int and receipt['bytes'] > 0)
    check('checkpoint_path_present', isinstance(receipt.get('path'), str) and bool(receipt['path']))
    check('metadata_only_cpu_reader', receipt.get('loading') ==
          'FakeTensorMode; CPU metadata only; no neural execution; mmap=False'
          and receipt.get('tensor_values_exported') is False)
    check('original_inventory_sha256_valid', isinstance(inventory.get('original_inventory_sha256'), str)
          and bool(_SHA.fullmatch(inventory['original_inventory_sha256'])))
    equal('checkpoint_inventory_sha256_matches', identity.get('cache_sha256'), inventory_sha256)
    for key in ('local_cnn', 'config', 'base', 'learning_policy'):
        equal('checkpoint_' + key + '_matches_inventory', identity.get(key), inventory.get(key))
    equal('request_local_cnn_matches_inventory', request.get('local_cnn'), inventory.get('local_cnn'))
    equal('request_original_inventory_matches_inventory', request.get('original_inventory_sha256'), inventory.get('original_inventory_sha256'))
    equal('request_source_matches_checkpoint', request.get('source'), identity.get('source'))

    source = mapping(identity.get('source'), 'checkpoint.identity.source', ('core', 'runtime'))
    source_files = {}
    required = required_source_files()
    for group in ('core', 'runtime'):
        declared = mapping(source.get(group), 'checkpoint.identity.source.' + group, ())
        check('source_' + group + '_nonempty', bool(declared))
        missing.extend('checkpoint.identity.source.' + group + '.' + name
                       for name in sorted(required[group] - set(declared)))
        for name, digest in declared.items():
            path = PurePosixPath(name)
            safe = (bool(name) and not path.is_absolute() and '..' not in path.parts
                    and '\\' not in name and ':' not in name)
            check('source_path_' + group + ':' + name, safe)
            check('source_sha256_' + group + ':' + name, isinstance(digest, str) and bool(_SHA.fullmatch(digest)))
            if name in source_files:
                equal('source_duplicate_' + name, source_files[name], digest)
            source_files[name] = digest

    # Reuse the actual schema's wrapper/resource/execution/schedule/cursor checks.
    # No default batch size, missing setting, phase or step count is fabricated.
    bound, absent = native_execution_bindings(manifest, inventory, payload, execution, schedule)
    checks.update(bound)
    missing.extend(item for item in absent if not item.startswith('binding.'))
    failed.extend(name for name, passed in bound.items() if not passed)
    # Strengthen scalar equality and identity types used by that legacy exporter.
    for key, value in identity.items():
        if key in execution:
            equal('strict_execution_identity_' + key, execution[key], value)
    for key in ('physical_batch', 'effective_batch', 'accumulation', 'optimization_steps', 'train_samples', 'val_samples'):
        check('execution_integer_' + key, type(execution.get(key)) is int and execution[key] > 0)
    for key in ('workers', 'epochs'):
        check('identity_integer_' + key, type(identity.get(key)) is int and identity[key] > 0)
    check('identity_candidate_integers', isinstance(identity.get('candidates'), list) and bool(identity['candidates'])
          and all(type(value) is int and value >= 2 for value in identity['candidates']))
    if state.get('phase') == 'optimization':
        check('optimization_epoch_below_target', type(state.get('epoch')) is int and state['epoch'] < 40)
    elif state.get('phase') in ('refresh_memory', 'validation'):
        check('post_optimization_epoch_below_target', type(state.get('epoch')) is int and state['epoch'] < 40)
    if missing or failed:
        raise ReferenceContractError(missing, failed)

    phase, epoch, cursor, steps = (state['phase'], state['epoch'], state['next_batch'], schedule['optimization_steps'])
    # epoch is zero-based during optimization/validation and becomes40 only after
    # last validation. Completing optimizer updates is a distinct boundary.
    optimizer_epochs = epoch + int(phase in ('optimization', 'refresh_memory', 'validation') and cursor == steps)
    return dict(format=FORMAT, admitted=True, debug=False, inventory_sha256=inventory_sha256,
                checkpoint=receipt['path'], checkpoint_sha256=receipt['sha256'], checkpoint_bytes=receipt['bytes'],
                weights_transferred=False, tensor_values_exported=False, neural_execution=False,
                validation_scope='recorded metadata and file identity only; no predictive quality claim',
                checkpoint_content_hash_recomputed=False, checkpoint_recorded_content_sha256=payload['content_sha256'],
                target_epochs=40, optimization_steps_per_epoch=steps,
                saved_cursor={key: state[key] for key in ('epoch', 'step', 'phase', 'next_batch', 'batch')},
                optimizer_epochs_completed=optimizer_epochs,
                validated_epochs_completed=epoch, native_training_complete=phase == 'complete',
                recorded_source_files=len(source_files), checks=checks, required_fields_missing=[], failed_bindings=[])


def _read_json(path):
    path = Path(path)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Native metadata changed during reference admission: {path}')
    value = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(value, dict):
        raise ReferenceContractError(failed=[str(path) + ': JSON mapping required'])
    return value, dict(path=str(path.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def admit_native_reference(native_run, *, inventory_path=None):
    """Read the explicitly selected native attempt; leave all run files intact.

    The caller should place the returned admission in its new experiment identity.
    Reference admission accepts a saved partial endpoint and never resumes it.
    """
    root = Path(native_run).resolve(strict=True)
    inventory_path = root / 'inventory/index.json' if inventory_path is None else Path(inventory_path).resolve(strict=True)
    if inventory_path.resolve() != (root / 'inventory/index.json').resolve():
        raise ReferenceContractError(failed=['inventory_path must be native_run/inventory/index.json'])
    manifest, manifest_file = _read_json(root / 'experiment.json')
    inventory, inventory_file = _read_json(inventory_path)
    checkpoint = select_checkpoint(root, manifest)
    checkpoint_stat = checkpoint.stat()
    receipt = checkpoint_metadata(checkpoint)
    if Path(receipt['path']).resolve() != checkpoint.resolve():
        raise ReferenceContractError(failed=['metadata reader checkpoint path differs from selected attempt'])
    execution, execution_file = _read_json(checkpoint.parent / 'execution_contract.json')
    schedule, schedule_file = _read_json(checkpoint.parent / 'learning_schedule.json')
    result = validate_reference_metadata(manifest, inventory, receipt, execution, schedule,
                                         inventory_sha256=inventory_file['sha256'])
    files = dict(experiment=manifest_file, inventory=inventory_file, execution_contract=execution_file, learning_schedule=schedule_file)
    # Metadata must stay bound through the potentially long checkpoint hash read.
    for bound in files.values():
        _, current = _read_json(bound['path'])
        if current != bound:
            raise RuntimeError(f'Native metadata changed during reference admission: {bound["path"]}')
    if select_checkpoint(root, manifest).resolve() != checkpoint.resolve():
        raise RuntimeError('Selected native checkpoint changed during reference admission')
    if checkpoint.stat() != checkpoint_stat:
        raise RuntimeError('Native checkpoint changed during reference admission')
    result.update(experiment_sha256=manifest_file['sha256'], metadata_files=files)
    return result
