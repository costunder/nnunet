"""Read-only evidence export for the complete v1 to native v2.2 transition.

This is metadata collection, not a neural evaluation or training entrypoint.
The receipt never contains CT volumes, masks, or checkpoint tensor values.
Missing historical evidence is recorded rather than invented or repaired.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import pickle
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
from .contracts import V1_ARCHIVE_SHA256, V1_MANIFEST_SHA256

FORMAT = 'hiercp_native_transition_receipt_v1'
ROOT = Path(__file__).resolve().parents[1]


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(name):
    value = PurePosixPath(name)
    if not name or value.is_absolute() or '..' in value.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe evidence member: {name}')
    return value


def _inside(root, relative):
    relative = _relative(relative)
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Evidence path escapes its root: {relative}')
    return path


def select_checkpoint(root, manifest):
    """Same explicit attempt semantics as run_local_cnn_experiment; no glob/latest."""
    if manifest is None:
        path = root / 'training/checkpoint_latest.pt'
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if manifest.get('format') != 'local_cnn_experiment_v1':
        raise ValueError('Native experiment manifest format differs')
    attempts = manifest.get('attempts')
    if not isinstance(attempts, list) or not attempts:
        raise ValueError('Native experiment has no bound optimization attempt')
    current = attempts[-1]
    output = _inside(root, current['output'])
    path = output / 'checkpoint_latest.pt'
    if path.is_file():
        return path
    if (output / 'training_complete.json').exists() or (output / 'checkpoint_timing.jsonl').exists():
        raise FileNotFoundError('Recorded checkpoint missing; no older weights selected')
    previous = current.get('resume_from')
    if previous is None:
        raise ValueError('Current native attempt has no saved update')
    path = _inside(root, previous)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def metadata_value(value):
    """Preserve scalar/configuration metadata; summarize every tensor, never its data."""
    import torch
    if isinstance(value, torch.Tensor):
        return dict(kind='tensor_metadata_only', shape=list(value.shape), dtype=str(value.dtype),
                    layout=str(value.layout), numel=value.numel(), values_exported=False)
    if isinstance(value, dict):
        return {str(key): metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [metadata_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else dict(kind='nonfinite_float', value=str(value))
    # NumPy RNG arrays and scalars are metadata, not image arrays. Values are
    # deliberately not materialized; architecture/config dictionaries stay exact.
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return dict(kind=type(value).__name__, shape=list(value.shape), dtype=str(value.dtype),
                    values_exported=False)
    return dict(kind='unsupported_metadata_type', type=type(value).__module__ + '.' + type(value).__qualname__)


def checkpoint_metadata(path):
    """Inspect trusted own-run PyTorch pickle without allocating weight storage."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    before = Path(path).stat()
    hash_started = perf_counter()
    digest = sha_file(path)
    hash_seconds = perf_counter() - hash_started
    with FakeTensorMode():
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=False)
        if not isinstance(payload, dict):
            raise ValueError('Native checkpoint is not a mapping')
        summary = metadata_value(payload)
    after = Path(path).stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Checkpoint changed during read-only collection; retry after save/pause')
    return dict(path=str(Path(path).resolve()), sha256=digest, bytes=before.st_size,
                sha256_stream_seconds=hash_seconds,
                loading='FakeTensorMode; CPU metadata only; no neural execution; mmap=False',
                tensor_values_exported=False, payload=summary)


class Collector:
    def __init__(self):
        self.payload = {}
        self.inputs = []
        self.errors = []

    def add(self, path, member, required=False):
        _relative(member)
        path = Path(path)
        if not path.is_file():
            if required:
                self.errors.append(dict(field=member, reason='file_missing', path=str(path)))
            return None
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            self.errors.append(dict(field=member, reason='file_changed_during_collection', path=str(path)))
        self.payload[member] = raw
        self.inputs.append(dict(path=str(path.resolve()), member=member, bytes=len(raw),
                                sha256=hashlib.sha256(raw).hexdigest()))
        return raw

    def json(self, path, member, required=False):
        raw = self.add(path, member, required)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode('utf-8-sig'))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self.errors.append(dict(field=member, reason='invalid_json', error=str(error)))
            return None

    def generated(self, member, value):
        self.payload[member] = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf8')


def _source_hashes(identity):
    if not isinstance(identity, dict):
        return {}
    result = {}
    for group in ('core', 'runtime'):
        values = identity.get(group)
        if isinstance(values, dict):
            for name, digest in values.items():
                _relative(name)
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ValueError(f'Invalid bound source hash: {name}')
                if name in result and result[name] != digest:
                    raise ValueError(f'Conflicting bound source hashes: {name}')
                result[name] = digest
    return result


def required_source_files(root=ROOT):
    """File-name inventory of the real provenance()/source_identity() definition.

    The installed collector definition is used even when an old checkout is
    offered for byte matching. Historical inventory differences stay unknown.
    """
    core = {'run_v222_v1_l0.py', 'config/prompt_graph_v222_v1_l0.json', 'config/train.json'}
    for directory in ('hiercp', 'hiercp_v22', 'hiercp_v222'):
        core.update(path.relative_to(root).as_posix() for path in (root / directory).glob('*.py'))
    runtime = {'l0_exploration/model.py', 'l0_exploration/data.py', 'l0_sage/encoder.py',
               'tools/run_fixed_regions.py', 'tools/run_local_cnn.py', 'tools/v22_rank_objective.py',
               'tools/v222_review_contracts.py', 'tools/v22_rank_recommendation.py', 'tools/v22_online_rank_bank.py'}
    for directory in ('l0_regions', 'l0_local_cnn'):
        runtime.update(path.relative_to(root).as_posix() for path in (root / directory).glob('*.py'))
    return dict(core=core, runtime=runtime)


def collect_sources(collector, identity, roots):
    expected = _source_hashes(identity)
    required = required_source_files()
    inventory_missing = []
    for group, names in required.items():
        declared = identity.get(group) if isinstance(identity, dict) else None
        if not isinstance(declared, dict) or not declared:
            inventory_missing.append(group)
            declared = {}
        inventory_missing.extend(group + '/' + name for name in sorted(names - set(declared)))
    results = []
    for index, root in enumerate(roots):
        files, mismatches = {}, []
        for name, digest in sorted(expected.items()):
            raw = collector.add(_inside(root, name), f'sources/root_{index:02d}/{name}')
            actual = None if raw is None else hashlib.sha256(raw).hexdigest()
            files[name] = actual
            if actual != digest:
                mismatches.append(dict(file=name, expected_sha256=digest, actual_sha256=actual))
        archive = root / 'versions/v1/pipeline_v1_source.zip'
        raw = collector.add(archive, f'sources/root_{index:02d}/versions/v1/pipeline_v1_source.zip')
        results.append(dict(root=str(root), expected_files=len(expected), files=files,
                            matches_checkpoint_source=bool(expected) and not mismatches and not inventory_missing,
                            required_source_identity_files_missing=inventory_missing,
                            mismatches=mismatches,
                            original_archive_sha256=None if raw is None else hashlib.sha256(raw).hexdigest(),
                            original_archive_matches_preserved_v1=raw is not None and hashlib.sha256(raw).hexdigest() == V1_ARCHIVE_SHA256))
    return results


ROOT_METADATA = ('manifest.json', 'config.json', 'experiment_version.json', 'execution_lock.json',
                 'experiment.json', 'shared/split.json', 'shared/cache/index.json',
                 'shared/cache/config.json', 'shared/cache/manifest.csv', 'shared/cache/complete.json',
                 'shared/metadata.json', 'shared/manifest.csv', 'sampling/v1.0.json')
RESULT_METADATA = ('execution_contract.json', 'resource_preflight.json', 'batch_calibration.json',
                   'learning_schedule.json', 'validation_initial.json', 'validation_history.csv',
                   'validation_history.json', 'training_history.csv', 'training_complete.json',
                   'paused.json', 'epoch_records.jsonl', 'epoch_telemetry.jsonl', 'history.jsonl',
                   'phase_timing.jsonl', 'support_timing.jsonl', 'memory_timing.jsonl',
                   'timing.jsonl', 'checkpoint_timing.jsonl', 'optimization_timing.jsonl',
                   'learning_health.jsonl', 'execution_upgrade.json', 'support_migration.json')


def collect_result_metadata(collector, directory, prefix, required=()):
    for name in RESULT_METADATA:
        collector.add(directory / name, prefix + '/' + name, required=name in required)
    # Include all named invocation/epoch/preflight metadata, not only the latest
    # file or a shortened history. Checkpoint selection never uses this inventory.
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv'):
                member = prefix + '/' + path.name
                if member not in collector.payload:
                    collector.add(path, member)


def collect_comparison(collector, role, root, result_name, source_roots):
    for name in ROOT_METADATA:
        collector.add(root / name, f'comparisons/{role}/{name}', required=name == 'manifest.json')
    manifest = collector.json(root / 'manifest.json', f'comparisons/{role}/manifest.json')
    hashes, implementation, inventory_missing = {}, [], []
    if isinstance(manifest, dict):
        source = manifest.get('source')
        pinned = collector.json(ROOT / 'versions/v1/manifest.json', 'sources/preserved_v1_manifest.json')
        pinned_files = pinned.get('files', {}) if isinstance(pinned, dict) else {}
        if isinstance(source, str):
            hashes.update(_snapshot_files(collector, role, Path(source), manifest.get('source_hashes', pinned_files)))
        elif manifest.get('format') == 'v1_bounded_roi_full_learning_v1':
            hashes.update(_snapshot_files(collector, role, root / 'source/v1.0', pinned_files))
        for stage, entry in manifest.get('stages', {}).items():
            if isinstance(entry, dict):
                hashes.update(_snapshot_files(collector, role + '/' + stage, root / 'source' / stage,
                                             entry.get('source_hashes', {})))
                collector.add(root / 'configs' / (stage + '.json'), f'comparisons/{role}/configs/{stage}.json')
        expected_helpers = manifest.get('implementation', manifest.get('helpers', {}))
        required_field = ('helpers' if manifest.get('format') == 'v1_bounded_roi_full_learning_v1'
                          else 'implementation' if manifest.get('format') in
                          ('hiercp_v14_m10_half_A_learning_v1', 'hiercp_v14_m10_half_B_learning_v1') else None)
        if required_field is not None:
            named = manifest.get(required_field)
            if not isinstance(named, dict) or not named:
                inventory_missing.append(required_field)
        if isinstance(expected_helpers, dict) and expected_helpers:
            for index, source_root in enumerate(source_roots):
                checked = _snapshot_files(collector, role + f'/implementation_root_{index:02d}',
                                          source_root, expected_helpers)
                implementation.append(dict(root=str(source_root), checks=checked,
                                           matches=bool(checked) and all(item['matched'] for item in checked.values())))
    results = root / 'results' / result_name
    collect_result_metadata(collector, results, f'comparisons/{role}/results/{result_name}')
    # The original workers record telemetry in named directories. This is a
    # metadata-only file inventory, not newest-checkpoint selection.
    for folder in (results / 'telemetry', results / 'epoch_records', root / 'allocations'):
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv'):
                    collector.add(path, f'comparisons/{role}/{path.relative_to(root).as_posix()}')
    checkpoints = []
    for name in ('checkpoint_best.pt', 'checkpoint_best.last.pt', 'checkpoint_latest.pt'):
        path = results / name
        if path.is_file():
            stat = path.stat()
            checkpoints.append(dict(path=str(path), bytes=stat.st_size, sha256=sha_file(path), weights_exported=False))
    return dict(root=str(root), result_name=result_name, manifest_present=manifest is not None,
                source_snapshot_checks=hashes, implementation_sources=implementation,
                required_source_inventories_missing=inventory_missing, checkpoint_files=checkpoints)


def _snapshot_files(collector, role, root, expected):
    results = {}
    if not isinstance(expected, dict):
        return results
    for name, digest in sorted(expected.items()):
        raw = collector.add(_inside(root.resolve(), name), f'comparisons/{role}/source/{name}')
        actual = None if raw is None else hashlib.sha256(raw).hexdigest()
        results[name] = dict(expected_sha256=digest, actual_sha256=actual, matched=actual == digest)
    return results


def inventory_audit(inventory):
    if not isinstance(inventory, dict):
        return dict(available=False)
    rows = inventory.get('records')
    if not isinstance(rows, list):
        return dict(available=True, records_present=False)
    counts = Counter((row.get('case_id'), row.get('target')) for row in rows if isinstance(row, dict))
    cases = sorted({row.get('case_id') for row in rows if isinstance(row, dict) and isinstance(row.get('case_id'), str)})
    ids = [row.get('id') for row in rows if isinstance(row, dict)]
    return dict(available=True, records_present=True, records=len(rows), unique_ids=len(set(ids)),
                observed_positive=sum(row.get('target') == 1 for row in rows),
                unobserved_comparison=sum(row.get('target') == 0 for row in rows),
                per_case=[dict(case_id=case, P=counts[case, 1], U=counts[case, 0]) for case in cases],
                split=inventory.get('split'), local_cnn=inventory.get('local_cnn'),
                learning_policy=inventory.get('learning_policy'), labels='P observed tumor; U unobserved comparison; neither is donor-specific CP suitability')


def native_inventory_contract(inventory):
    """Presence/type gates taken from LocalCNN Dataset; no CT/mask evaluation."""
    if not isinstance(inventory, dict):
        return ['native.inventory']
    missing = []
    for name in ('format', 'complete', 'debug', 'config', 'base', 'split', 'identities', 'donor_pool',
                 'raw_records', 'source_identity', 'records', 'learning_policy', 'local_cnn',
                 'original_inventory_sha256'):
        if name not in inventory:
            missing.append('native.inventory.' + name)
    if inventory.get('format') != 'native_local_cnn_inventory_v1':
        missing.append('native.inventory.format_contract')
    if inventory.get('complete') is not True:
        missing.append('native.inventory.complete_contract')
    for name in ('config', 'base', 'split', 'identities', 'source_identity', 'local_cnn'):
        if name in inventory and not isinstance(inventory[name], dict):
            missing.append('native.inventory.' + name + '_mapping')
    for name in ('records', 'donor_pool', 'raw_records'):
        if name in inventory and not isinstance(inventory[name], list):
            missing.append('native.inventory.' + name + '_list')
    return missing


def native_execution_bindings(manifest, inventory, payload, execution, schedule):
    """Bind the actual wrapper request, saved controls, resolved batch and cursor.

    Field names follow run_local_cnn_experiment/run_local_cnn/training.py.
    A missing old field is a diagnostic finding, never a synthesized default.
    """
    checks, missing = {}, []

    def mapping(value, label, keys=()):
        if not isinstance(value, dict):
            missing.append(label)
            return {}
        for name in keys:
            if name not in value:
                missing.append(label + '.' + name)
        return value

    def equal(name, actual, expected, *, available=True):
        checks[name] = available and actual == expected
        if not checks[name]:
            missing.append('binding.' + name)

    request = mapping(manifest.get('request') if isinstance(manifest, dict) else None, 'native.request', ('settings',))
    settings = mapping(request.get('settings'), 'native.request.settings',
                       ('workers', 'cuda_gib', 'rss_gib', 'resident_gib', 'device_cache_gib',
                        'batch_candidates', 'support_patients', 'debug'))
    payload = mapping(payload, 'native.checkpoint.payload', ('format', 'identity', 'model', 'optimizer', 'state', 'rng', 'content_sha256'))
    identity = mapping(payload.get('identity'), 'native.checkpoint.identity',
                       ('format', 'debug', 'activation_storage', 'resident_budget_bytes', 'resource_limits',
                        'execution_pipeline', 'profile_policy', 'support_training', 'epochs', 'workers', 'candidates',
                        'graph_representation'))
    state = mapping(payload.get('state'), 'native.checkpoint.state',
                    ('batch', 'epoch', 'step', 'phase', 'next_batch', 'memory', 'memory_parts', 'memory_done',
                     'plan', 'last_group', 'best', 'selected_epoch', 'initial_validation', 'validation_history'))
    execution = mapping(execution, 'native.execution_contract',
                        ('physical_batch', 'effective_batch', 'accumulation', 'optimization_steps', 'train_samples', 'val_samples'))
    schedule = mapping(schedule, 'native.learning_schedule',
                       ('policy', 'unique_observations', 'physical_batch', 'actual_batch_sizes', 'optimization_steps'))
    limits = mapping(identity.get('resource_limits'), 'native.checkpoint.identity.resource_limits', ('cuda_bytes', 'rss_bytes'))
    pipeline = mapping(identity.get('execution_pipeline'), 'native.checkpoint.identity.execution_pipeline',
                       ('mode', 'device_cache_bytes', 'sage_workspace_bytes', 'checkpoints', 'gradient_check'))
    support = mapping(identity.get('support_training'), 'native.checkpoint.identity.support_training', ('policy', 'patients'))
    for setting, identity_key in (('workers', 'workers'), ('batch_candidates', 'candidates'), ('debug', 'debug')):
        equal('request_' + setting + '_matches_checkpoint', settings.get(setting), identity.get(identity_key),
              available=setting in settings and identity_key in identity)
    for setting, actual, present in (
        ('cuda_gib', limits.get('cuda_bytes'), 'cuda_bytes' in limits),
        ('rss_gib', limits.get('rss_bytes'), 'rss_bytes' in limits),
        ('resident_gib', identity.get('resident_budget_bytes'), 'resident_budget_bytes' in identity),
        ('device_cache_gib', pipeline.get('device_cache_bytes'), 'device_cache_bytes' in pipeline)):
        value = settings.get(setting)
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        equal('request_' + setting + '_matches_checkpoint', actual,
              int(value * 2**30) if valid else None, available=valid and present)
    equal('request_support_patients_matches_checkpoint', settings.get('support_patients'), support.get('patients'),
          available='support_patients' in settings and 'patients' in support)
    equal('request_debug_matches_inventory', settings.get('debug'), inventory.get('debug') if isinstance(inventory, dict) else None,
          available='debug' in settings and isinstance(inventory, dict) and 'debug' in inventory)
    equal('wrapper_activation_storage_retained', identity.get('activation_storage'), 'retained',
          available='activation_storage' in identity)
    equal('wrapper_execution_pipeline_overlapped', pipeline.get('mode'), 'overlapped', available='mode' in pipeline)
    cnn = inventory.get('local_cnn', {}) if isinstance(inventory, dict) else {}
    equal('checkpoint_graph_representation_matches_inventory', identity.get('graph_representation'),
          cnn.get('architecture') if isinstance(cnn, dict) else None,
          available='graph_representation' in identity and isinstance(cnn, dict) and 'architecture' in cnn)
    equal('execution_graph_representation_matches_checkpoint', execution.get('graph_representation'),
          identity.get('graph_representation'), available='graph_representation' in identity and 'graph_representation' in execution)
    equal('checkpoint_format', payload.get('format'), 'fixed_region_sage_training_v1', available='format' in payload)
    equal('checkpoint_identity_format', identity.get('format'), payload.get('format'),
          available='format' in identity and 'format' in payload)
    # Execution contains the unchanged identity plus measured/runtime fields.
    for key, value in identity.items():
        if key != 'graph_representation':
            equal('execution_identity_' + key, execution.get(key), value, available=key in execution)
    batch = state.get('batch')
    candidates = identity.get('candidates')
    valid_batch = type(batch) is int and batch >= 2 and isinstance(candidates, list) and batch in candidates
    checks['resolved_batch_is_declared_candidate'] = valid_batch
    if not valid_batch:
        missing.append('binding.resolved_batch_is_declared_candidate')
    equal('resolved_batch_matches_execution_physical', batch, execution.get('physical_batch'),
          available=valid_batch and 'physical_batch' in execution)
    equal('resolved_batch_matches_execution_effective', batch, execution.get('effective_batch'),
          available=valid_batch and 'effective_batch' in execution)
    equal('native_gradient_accumulation', execution.get('accumulation'), 1, available='accumulation' in execution)
    equal('resolved_batch_matches_schedule', batch, schedule.get('physical_batch'),
          available=valid_batch and 'physical_batch' in schedule)
    steps = schedule.get('optimization_steps')
    epochs = identity.get('epochs')
    valid_steps = type(steps) is int and steps > 0 and type(epochs) is int and epochs > 0
    equal('schedule_total_optimizer_steps_matches_execution', steps * epochs if valid_steps else None,
          execution.get('optimization_steps'), available=valid_steps and 'optimization_steps' in execution)
    equal('schedule_observation_coverage_matches_execution', schedule.get('unique_observations'), execution.get('train_samples'),
          available='unique_observations' in schedule and 'train_samples' in execution)
    sizes = schedule.get('actual_batch_sizes')
    valid_sizes = valid_steps and valid_batch and isinstance(sizes, list) and len(sizes) == steps and all(
        type(size) is int and 1 <= size <= batch for size in sizes)
    checks['schedule_actual_batch_sizes_valid'] = valid_sizes
    if not valid_sizes:
        missing.append('binding.schedule_actual_batch_sizes_valid')
    phase, epoch, step, cursor = (state.get(name) for name in ('phase', 'epoch', 'step', 'next_batch'))
    valid_cursor = (phase in ('initial_validation', 'initial_memory', 'optimization', 'refresh_memory', 'validation', 'final_memory', 'complete')
                    and type(epoch) is int and type(step) is int and type(cursor) is int and valid_steps
                    and 0 <= epoch <= epochs and step >= 0 and 0 <= cursor <= steps)
    checks['saved_training_cursor_valid'] = valid_cursor
    if not valid_cursor:
        missing.append('binding.saved_training_cursor_valid')
    equal('saved_step_matches_epoch_and_cursor', step, epoch * steps + cursor if valid_cursor else None,
          available=valid_cursor)
    if phase in ('final_memory', 'complete'):
        equal('final_phase_completed_epochs', epoch, epochs, available=valid_cursor)
        equal('final_phase_query_cursor_reset', cursor, 0, available=valid_cursor)
    elif phase in ('refresh_memory', 'validation'):
        equal('post_optimization_cursor_complete', cursor, steps, available=valid_cursor)
    elif phase in ('initial_validation', 'initial_memory'):
        equal('initial_phase_before_updates', (epoch, step, cursor), (0, 0, 0), available=valid_cursor)
    return checks, missing


def export_receipt(native_run, baseline, half_a, half_b, output, *, source_roots=(), checkpoint_reader=checkpoint_metadata,
                   create_archive=True):
    if type(create_archive) is not bool:
        raise ValueError('Archive selection must be an explicit boolean')
    collection_started = perf_counter()
    roots = {name: Path(path).resolve() for name, path in
             (('native', native_run), ('baseline', baseline), ('half_A', half_a), ('half_B', half_b))}
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists; preserved: {output}')
    for root in roots.values():
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError('Receipt output must be disjoint from all input experiment roots')
    collector = Collector()
    native = roots['native']
    manifest = collector.json(native / 'experiment.json', 'native/experiment.json', required=True)
    inventory = collector.json(native / 'inventory/index.json', 'native/inventory/index.json', required=True)
    checkpoint, metadata = None, None
    try:
        checkpoint = select_checkpoint(native, manifest)
        metadata = checkpoint_reader(checkpoint)
        collector.generated('native/checkpoint_metadata.json', metadata)
    except (FileNotFoundError, ValueError, RuntimeError, OSError, ImportError, EOFError, pickle.UnpicklingError) as error:
        collector.errors.append(dict(field='native.checkpoint', reason=type(error).__name__, error=str(error)))
    payload = metadata.get('payload', {}) if metadata is not None else {}
    identity = payload.get('identity')
    if not isinstance(identity, dict):
        identity = {}
        collector.errors.append(dict(field='native.checkpoint.identity', reason='missing'))
    source_results = []
    candidates = list(dict.fromkeys([Path(path).resolve() for path in (ROOT, *source_roots)]))
    try:
        source_results = collect_sources(collector, identity.get('source'), candidates)
    except (ValueError, OSError) as error:
        collector.errors.append(dict(field='native.source', reason=type(error).__name__, error=str(error)))
    if checkpoint is not None:
        collect_result_metadata(collector, checkpoint.parent, 'native/attempt',
                                required=('execution_contract.json', 'learning_schedule.json'))
    if isinstance(manifest, dict) and isinstance(manifest.get('attempts'), list):
        for index, attempt in enumerate(manifest['attempts']):
            try:
                directory = _inside(native, attempt['output'])
                collect_result_metadata(collector, directory, f'native/all_attempts/{index:04d}')
            except (KeyError, TypeError, ValueError, OSError) as error:
                collector.errors.append(dict(field=f'native.attempts.{index}', reason=type(error).__name__, error=str(error)))
    comparisons = []
    for role, result in (('baseline', 'v1.0'), ('half_A', 'half_A'), ('half_B', 'half_B')):
        try:
            comparisons.append(collect_comparison(collector, role, roots[role], result, candidates))
        except (ValueError, OSError, UnicodeDecodeError) as error:
            collector.errors.append(dict(field='comparisons.' + role, reason=type(error).__name__, error=str(error)))
    missing = [item['field'] for item in collector.errors]
    missing.extend(native_inventory_contract(inventory))
    for name in ('config', 'base', 'local_cnn', 'ranking', 'epochs', 'workers', 'candidates',
                 'resource_limits', 'precision', 'learning_policy', 'support_training', 'source', 'cache_sha256'):
        if name not in identity:
            missing.append('native.checkpoint.identity.' + name)
    if not any(value['matches_checkpoint_source'] for value in source_results):
        missing.append('native.source_root_matching_checkpoint')
    for value in source_results:
        missing.extend('native.source_identity.' + name for name in value['required_source_identity_files_missing'])
    if not any(value['original_archive_matches_preserved_v1'] for value in source_results):
        missing.append('preserved_v1_original_archive_sha256')
    pinned_raw = collector.payload.get('sources/preserved_v1_manifest.json')
    if pinned_raw is None or hashlib.sha256(pinned_raw).hexdigest() != V1_MANIFEST_SHA256:
        missing.append('preserved_v1_original_manifest_sha256')
    for comparison in comparisons:
        missing.extend('comparisons.' + comparison['result_name'] + '.' + name
                       for name in comparison['required_source_inventories_missing'])
        if not comparison['source_snapshot_checks']:
            missing.append('comparisons.' + comparison['result_name'] + '.source_snapshot_identity')
        if any(not item['matched'] for item in comparison['source_snapshot_checks'].values()):
            missing.append('comparisons.' + comparison['result_name'] + '.source_snapshot_bytes')
        if comparison['implementation_sources'] and not any(item['matches'] for item in comparison['implementation_sources']):
            missing.append('comparisons.' + comparison['result_name'] + '.implementation_source_root')
    bindings = {}
    if isinstance(inventory, dict):
        raw = collector.payload['native/inventory/index.json']
        bindings['checkpoint_inventory_sha256_matches'] = identity.get('cache_sha256') == hashlib.sha256(raw).hexdigest()
        bindings['checkpoint_local_cnn_matches_inventory'] = identity.get('local_cnn') == inventory.get('local_cnn')
        bindings['checkpoint_config_matches_inventory'] = identity.get('config') == inventory.get('config')
        bindings['checkpoint_base_matches_inventory'] = identity.get('base') == inventory.get('base')
        bindings['checkpoint_learning_policy_matches_inventory'] = identity.get('learning_policy') == inventory.get('learning_policy')
        if isinstance(manifest, dict):
            request = manifest.get('request', {})
            bindings['request_source_matches_checkpoint'] = request.get('source') == identity.get('source')
            bindings['request_local_cnn_matches_inventory'] = request.get('local_cnn') == inventory.get('local_cnn')
            bindings['request_original_inventory_matches_inventory'] = request.get('original_inventory_sha256') == inventory.get('original_inventory_sha256')
    for name, matches in bindings.items():
        if not matches:
            missing.append('binding.' + name)
    execution, schedule = None, None
    for member, destination in (('native/attempt/execution_contract.json', 'execution'),
                                ('native/attempt/learning_schedule.json', 'schedule')):
        raw = collector.payload.get(member)
        if raw is None:
            continue
        try:
            value = json.loads(raw.decode('utf-8-sig'))
            if destination == 'execution':
                execution = value
            else:
                schedule = value
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError) as error:
            collector.errors.append(dict(field=member, reason=type(error).__name__, error=str(error)))
            missing.append(member)
    setting_checks, setting_missing = native_execution_bindings(manifest, inventory, payload, execution, schedule)
    bindings.update(setting_checks)
    missing.extend(setting_missing)
    audit = inventory_audit(inventory)
    if audit.get('records') != 14102:
        missing.append('native.full_14102_observation_inventory')
    if audit.get('records') != audit.get('unique_ids'):
        missing.append('native.unique_record_ids')
    if any(row['U'] != 128 for row in audit.get('per_case', [])):
        missing.append('native.full_128_comparison_positions_per_case')
    checked_inputs = set()
    for item in collector.inputs:
        key = (item['path'], item['sha256'])
        if key in checked_inputs:
            continue
        checked_inputs.add(key)
        path = Path(item['path'])
        try:
            unchanged = path.is_file() and sha_file(path) == item['sha256']
        except OSError:
            unchanged = False
        if not unchanged:
            collector.errors.append(dict(field=item['member'], reason='input_changed_before_receipt_seal', path=str(path)))
            missing.append('input_changed.' + item['member'])
    artifacts = ([] if metadata is None else [dict(role='native', path=str(checkpoint),
                  sha256=metadata.get('sha256'), bytes=metadata.get('bytes'))])
    for comparison in comparisons:
        artifacts.extend(dict(role=comparison['result_name'], **item) for item in comparison['checkpoint_files'])
    artifact_checks = []
    for artifact in artifacts:
        path = Path(artifact['path'])
        started = perf_counter()
        try:
            before = path.stat()
            digest = sha_file(path)
            after = path.stat()
            unchanged = (digest == artifact['sha256'] and before.st_size == artifact['bytes']
                         and (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns))
            error = None
        except OSError as failure:
            unchanged, digest, error = False, None, str(failure)
        artifact_checks.append(dict(artifact, sha256_at_seal=digest, unchanged=unchanged,
                                    stream_sha_seconds=perf_counter() - started, error=error))
        if not unchanged:
            field = 'checkpoint_changed_before_receipt_seal.' + artifact['role'] + '.' + path.name
            collector.errors.append(dict(field=field, reason='artifact_changed_or_unreadable', path=str(path), error=error))
            missing.append(field)
    required = sorted(set(missing))
    report = dict(format=FORMAT, mode='read_only_metadata_export', inputs={name: str(root) for name, root in roots.items()},
                  native_checkpoint=None if checkpoint is None else str(checkpoint), inventory=audit,
                  checkpoint_identity=identity, checkpoint_state=payload.get('state'), bindings=bindings,
                  sources=source_results, comparisons=comparisons, input_files=collector.inputs,
                  checkpoint_artifact_checks=artifact_checks,
                  collection_seconds_before_serialization=perf_counter() - collection_started,
                  errors=collector.errors, required_fields_missing=required,
                  exact_native_recipe_bound=not required, tensor_values_exported=False,
                  checkpoint_content_hash_recomputed=False,
                  source_execution_validated=False, quality_verified=False, production_ready=False,
                  training_started=False, optimizer_updates=0, neural_forward_executed=False,
                  full_evaluation_executed=False, server_training_command_generated=False,
                  collector_platform=platform.platform(),
                  scope='Original metadata and source-byte collection only. SHA binding is not neural validation or recommended CP quality.')
    collector.generated('receipt.json', report)
    manifest_out = dict(format=FORMAT + '_manifest', files={name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                                                         for name, raw in sorted(collector.payload.items())})
    collector.generated('manifest.json', manifest_out)
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in sorted(collector.payload.items()):
        path = _inside(output, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    bundle = None
    if create_archive:
        bundle = output / 'transition_receipt.zip'
        with ZipFile(bundle, 'x', compression=ZIP_DEFLATED) as zipped:
            for name, raw in sorted(collector.payload.items()):
                zipped.writestr(name, raw)
    validate_receipt(bundle if bundle is not None else output)
    return report, bundle


def validate_receipt(path):
    """Standalone ZIP/directory SHA, size, inventory, path and truth-flag validation."""
    path = Path(path)
    if path.is_dir():
        members = {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob('*')
                   if p.is_file() and p.name != 'transition_receipt.zip'}
    else:
        with ZipFile(path) as zipped:
            names = zipped.namelist()
            if len(names) != len(set(names)) or zipped.testzip() is not None:
                raise ValueError('Duplicate ZIP members or failed CRC')
            members = {name: zipped.read(name) for name in names}
    for name in members:
        _relative(name)
    manifest = json.loads(members['manifest.json'])
    expected = manifest['files']
    if manifest.get('format') != FORMAT + '_manifest' or set(members) != set(expected) | {'manifest.json'}:
        raise ValueError('Receipt manifest inventory differs')
    for name, info in expected.items():
        raw = members[name]
        if len(raw) != info['bytes'] or hashlib.sha256(raw).hexdigest() != info['sha256']:
            raise ValueError(f'Receipt bytes differ: {name}')
    receipt = json.loads(members['receipt.json'])
    if receipt.get('format') != FORMAT or receipt.get('mode') != 'read_only_metadata_export':
        raise ValueError('Receipt format/mode differs')
    for name in ('quality_verified', 'production_ready', 'training_started', 'neural_forward_executed',
                 'full_evaluation_executed', 'tensor_values_exported', 'server_training_command_generated'):
        if receipt.get(name) is not False:
            raise ValueError(f'Metadata export cannot assert {name}')
    if receipt.get('optimizer_updates') != 0:
        raise ValueError('Metadata receipt cannot assert optimizer updates')
    if receipt.get('exact_native_recipe_bound') != (not receipt.get('required_fields_missing')):
        raise ValueError('Missing native fields cannot be marked bound')
    return receipt
