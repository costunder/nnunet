"""Verified same-policy singleton continuation across an input-runtime upgrade.

No source files or processes are changed. Publication requires an independently
verified paused source, numerical equivalence and real full128 engine evidence.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .contracts import canonical_hash
from .u_bridge_training import atomic_save, cpu_copy, digest
from .v23_training import (FORMAT, EXPECTED_PARAMETERS, _file_sha256, _validate_progress,
    _write_new, execution_identity, training_binding, remaining_training_updates)

PERFORMANCE_KEYS = frozenset(('batch_calibration', 'resume_checkpoint', 'pause_file',
    'prefetch_cpu_chunks', 'pin_cpu_batches', 'memoize_cpu_geometry',
    'cache_stage_admission', 'geometry_resident_gib_per_rank',
    'memoize_fixed_views', 'persistent_cpu_workers'))
NUMERICAL_KEYS = ('model', 'optimizer', 'scheduler', 'scaler', 'shuffle_generator', 'rank_rng')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _request_check(request):
    _require(canonical_hash({k: v for k, v in request.items() if k != 'request_sha256'})
        == request['request_sha256'], 'Exact request SHA required')


def _load_checkpoint(path, identity_sha):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), 'Regular source checkpoint required')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    _require(saved['format'] == FORMAT and saved['identity_sha256'] == identity_sha
        and saved['content_sha256'] == digest({k: v for k, v in saved.items() if k != 'content_sha256'}),
        'Source checkpoint ownership/content differs')
    return saved


def upgrade_single_gpu_checkpoint(source_checkpoint, source_identity, destination_request,
        config, calibration, output, *, source_proof, population, source_best_checkpoint=None,
        debug=False):
    """Keep all numerical state, candidate membership and partial-epoch cursors."""
    required = ('source_commit', 'source_request', 'source_files_sha256',
        'source_checkpoint_file_sha256', 'source_identity_file_sha256',
        'geometry_receipt_sha256', 'calibration_receipt_sha256',
        'core_equations_preserved', 'original_model_parameters', 'clean_pause_verified',
        'new_runtime_full128_backward_verified', 'numerical_equivalence_receipt_sha256')
    _require(all(key in source_proof for key in required), 'Complete explicit upgrade proof required')
    _require(source_proof['clean_pause_verified'] is True
        and source_proof['core_equations_preserved'] is True
        and source_proof['new_runtime_full128_backward_verified'] is True
        and (debug or source_proof['original_model_parameters'] == EXPECTED_PARAMETERS),
        'Paused original-model numerical/full128 evidence required')
    paths = dict(source_checkpoint_file_sha256=Path(source_checkpoint),
        source_identity_file_sha256=Path(source_identity))
    for key, path in paths.items():
        _require(path.is_file() and not path.is_symlink() and _file_sha256(path) == source_proof[key],
            'Immutable source file SHA differs: ' + key)
    for key in ('geometry_receipt_sha256', 'calibration_receipt_sha256',
                'numerical_equivalence_receipt_sha256'):
        value = source_proof[key]
        _require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), 'Exact upgrade SHA required: ' + key)
    ownership = json.loads(paths['source_identity_file_sha256'].read_text(encoding='utf8'))
    old = ownership['binding']
    _require(ownership['identity_sha256'] == digest(old), 'Original ownership binding differs')
    saved = _load_checkpoint(source_checkpoint, ownership['identity_sha256'])
    state = saved['state']
    _require(state['status'] == 'PAUSED' and state['phase'] != 'complete' and state['epoch'] <= 40,
        'Only the explicitly paused unfinished experiment can be upgraded')
    source_request = source_proof['source_request']
    _request_check(source_request); _request_check(destination_request)
    _require(old['identity']['request_sha256'] == source_request['request_sha256']
        and old['identity']['source'] == source_request['source']
        and source_proof['source_files_sha256'] == source_request['source'], 'Original source proof differs')
    _require(source_request['gpus'] == destination_request['gpus']
        and len(destination_request['gpus']) == 1 and destination_request['gpus'][0] in (1, 5, 6)
        and old['data_parallel_world_size'] == 1 and len(saved['rank_rng']) == 1
        and old['physical_patient_batch'] == 4 and old['physical_candidate_batch'] == 32,
        'Same GPU/world1/physical4/chunk32 upgrade required')
    _require(old['epochs'] == 40 and old['debug'] is debug
        and old['population'] == population.manifest(), 'Complete population/forty epochs must remain identical')
    for key in ('model', 'graph', 'training', 'ct_clip', 'runtime', 'labels', 'cache', 'generation'):
        _require(config.get(key) == old['config'].get(key), 'Numerical configuration changed: ' + key)
    old_runtime = {k: v for k, v in old['config']['v23_runtime'].items() if k not in PERFORMANCE_KEYS}
    new_runtime = {k: v for k, v in config['v23_runtime'].items() if k not in PERFORMANCE_KEYS}
    _require(old_runtime == new_runtime, 'Target policy, training gates or resource/model contract changed')
    _require(calibration['world_size'] == 1 and calibration['request_sha256'] == destination_request['request_sha256']
        and calibration['physical_GPUs'] == destination_request['gpus']
        and calibration['selected_physical_patient_batch'] == 4
        and calibration['selected_physical_candidate_batch'] == 32
        and calibration['initial_state_sha256'] == old['initial_state_sha256']
        and calibration['measured_full_P_U128_backward'] is True
        and calibration['original_model_and_RNG_preserved'] is True,
        'New runtime same-shape actual full128 calibration evidence required')
    _validate_progress(state, old['train_cases'], old['val_cases'], 4, 40,
        SimpleNamespace(last_epoch=saved['scheduler']['last_epoch']), 1)
    from .v23_training import validate_target_state
    validate_target_state(population, state['targets'],
        policy=config['v23_runtime']['target_selection_policy'], active_u=state['curriculum']['active_u'])
    for slot in saved['optimizer']['state'].values():
        step = slot['step']
        _require(int(step.item() if torch.is_tensor(step) else step) == state['updates'],
            'Source optimizer history differs from successful updates')
    continuation = dict(source_commit=source_proof['source_commit'],
        source_checkpoint=str(Path(source_checkpoint).resolve()), source_identity=str(Path(source_identity).resolve()),
        source_checkpoint_file_sha256=source_proof['source_checkpoint_file_sha256'],
        source_checkpoint_content_sha256=saved['content_sha256'],
        source_identity_sha256=ownership['identity_sha256'], source_request_sha256=source_request['request_sha256'],
        source_world_size=1, source_rank_rng_index=0, destination_world_size=1,
        destination_physical_GPU=destination_request['gpus'][0], inherited_epoch=state['epoch'],
        inherited_phase=state['phase'], inherited_updates=state['updates'],
        geometry_receipt_sha256=source_proof['geometry_receipt_sha256'],
        calibration_receipt_sha256=source_proof['calibration_receipt_sha256'],
        runtime_upgrade=True, numerical_equivalence_receipt_sha256=source_proof['numerical_equivalence_receipt_sha256'])
    identity = execution_identity(destination_request, continuation)
    config = copy.deepcopy(config); config['v23_runtime']['batch_calibration'] = cpu_copy(calibration)
    binding = training_binding(config, population, identity=identity, physical_patient_batch=4,
        physical_candidate_batch=32, world_size=1, workers=config['v23_runtime']['workers'],
        initial_hash=old['initial_state_sha256'], debug=debug)
    fork = cpu_copy(saved); fork['identity_sha256'] = digest(binding)
    fork['state']['handoff'] = continuation
    _require(digest({k: v for k, v in fork['state'].items() if k != 'handoff'})
        == digest({k: v for k, v in state.items() if k != 'handoff'}), 'Partial epoch or candidate state changed')
    for key in NUMERICAL_KEYS:
        _require(digest(fork[key]) == digest(saved[key]), 'Numerical state changed: ' + key)
    fork.pop('content_sha256'); fork['content_sha256'] = digest(fork)
    best_fork = None; best_proof = None
    if state['best'] is not None:
        _require(source_best_checkpoint is not None and 'source_best_checkpoint_file_sha256' in source_proof,
            'Actual source BEST required; latest cannot substitute')
        path = Path(source_best_checkpoint)
        _require(_file_sha256(path) == source_proof['source_best_checkpoint_file_sha256'], 'Source BEST file SHA differs')
        best = _load_checkpoint(path, ownership['identity_sha256'])
        _require(best['state']['best'] == state['best'] and len(best['rank_rng']) == 1,
            'Actual source BEST record differs')
        best_fork = cpu_copy(best); best_fork['identity_sha256'] = digest(binding)
        best_fork['state']['handoff'] = continuation
        best_fork.pop('content_sha256'); best_fork['content_sha256'] = digest(best_fork)
        _require(all(digest(best_fork[key]) == digest(best[key]) for key in NUMERICAL_KEYS), 'Actual BEST state changed')
        paths['source_best_checkpoint_file_sha256'] = path
        best_proof = dict(source_checkpoint=str(path.resolve()),
            source_file_sha256=source_proof['source_best_checkpoint_file_sha256'], model_sha256=digest(best['model']),
            best_record=cpu_copy(best['state']['best']), actual_best_weights_preserved=True, latest_weights_substituted=False)
    for key, path in paths.items():
        _require(_file_sha256(path) == source_proof[key], 'Source changed during upgrade: ' + key)
    output = Path(output)
    _require(not output.is_symlink(), 'Fresh regular output required')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Existing results preserved; fresh upgrade output required')
    receipt = dict(format='v23_verified_same_policy_singleton_runtime_upgrade_v1',
        destination_request_sha256=destination_request['request_sha256'], destination_identity=identity,
        destination_binding_sha256=digest(binding), continuation=continuation, source_proof=cpu_copy(source_proof),
        preserved_state_sha256={key: digest(saved[key]) for key in NUMERICAL_KEYS},
        preserved_progress_sha256=digest({k: v for k, v in state.items() if k != 'handoff'}),
        inherited_best_checkpoint=best_proof, source_best_record=cpu_copy(state['best']),
        destination_checkpoint_content_sha256=fork['content_sha256'],
        total_target_epochs=40, completed_epochs=len(state['history']),
        remaining_planned_updates=remaining_training_updates(state, old['train_cases'], 4, 1),
        no_new_optimizer_update=True, original_files_written=False, debug=debug)
    output.mkdir(parents=True, exist_ok=True)
    _write_new(output/'training_identity.json', dict(identity_sha256=digest(binding), binding=cpu_copy(binding)))
    if best_fork is not None: atomic_save(output/'checkpoint_best.pt', best_fork)
    atomic_save(output/'checkpoint_latest.pt', fork)
    _write_new(output/'handoff_receipt.json', receipt)
    return receipt
