"""Explicit, CPU-only admission of a user-authorized GPU5 epoch-boundary stop.

The original full40 pin format and admission rules are untouched. A PAUSED
checkpoint is never labelled full training: this separate format records the
actual completed history, all full128 validations and its own genuine BEST.
No model is constructed, RNG restored, training started or signal sent here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

EARLY_PIN_FORMAT = 'v24_user_authorized_epoch_boundary_early_stop_best_v1'
AUTHORIZATION_FORMAT = 'v24_user_authorized_GNN_epoch_boundary_stop_request_v1'
COMPLETION_KIND = 'USER_AUTHORIZED_EPOCH_BOUNDARY_EARLY_STOP'
STOP_POLICY = 'after_current_epoch_training_and_full128_validation'
SELECTION_METRIC = 'fixed full128 per-P patient-macro MRR, top1, negativepairloss'
NUMERICAL = ('model', 'optimizer', 'scheduler', 'scaler', 'rank_rng', 'shuffle_generator')


def _canonical(value):
    from .contracts import canonical_hash
    return canonical_hash(value)


def _regular(path, *, directory=False):
    path = Path(path)
    if (not path.is_absolute() or '..' in path.parts
            or any(item.is_symlink() for item in (path, *path.parents))
            or not (path.is_dir() if directory else path.is_file())
            or hasattr(os, 'getuid') and path.stat().st_uid != os.getuid()):
        raise ValueError('Owned absolute regular early-stop input required: ' + str(path))
    return path.resolve(strict=True)


def _identity(path):
    value = _regular(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def file_proof(path):
    path = _regular(path); before = _identity(path); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    if _identity(path) != before:
        raise ValueError('Early-stop input changed during read: ' + str(path))
    return dict(path=str(path), stat=before, sha256=digest.hexdigest())


def _read(path):
    value = json.loads(_regular(path).read_text(encoding='utf8'))
    if not isinstance(value, dict):
        raise ValueError('Early-stop metadata JSON object required: ' + str(path))
    return value


def _checksum(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set('0123456789abcdef')


def _actor_absent(actor):
    import psutil
    if (not isinstance(actor, dict) or set(actor) != {'pid', 'create_time', 'command'}
            or type(actor['pid']) is not int or actor['pid'] <= 0
            or type(actor['create_time']) not in (float, int) or not math.isfinite(actor['create_time'])
            or not isinstance(actor['command'], list) or not actor['command']
            or any(not isinstance(arg, str) or not arg for arg in actor['command'])):
        raise ValueError('Exact registered early-stop actor PID/birth/argv required')
    try:
        process = psutil.Process(actor['pid'])
        if (process.create_time() != actor['create_time']
                or hasattr(os, 'getuid') and process.uids().real != os.getuid()):
            raise ValueError('Early-stop actor PID reuse or UID mismatch')
        if process.status() == psutil.STATUS_ZOMBIE:
            return True
        if process.cmdline() != actor['command']:
            raise ValueError('Early-stop actor full command mismatch')
        raise ValueError('Early-stop actor is still live; no scoring/native admission')
    except psutil.NoSuchProcess:
        return True


def validate_authorization(value, *, source_output, source_code, target_epoch, gpu=5):
    if (type(gpu) is not int or gpu != 5 or type(target_epoch) is not int or not 1 <= target_epoch < 40
            or value.get('format') != AUTHORIZATION_FORMAT or value.get('GPU_arm') != gpu
            or value.get('source_output') != str(source_output) or value.get('source_code') != str(source_code)
            or value.get('target_epoch') != target_epoch or value.get('original_planned_epochs') != 40
            or value.get('user_authorized') is not True or value.get('stop_policy') != STOP_POLICY
            or type(value.get('created_at')) not in (int, float) or not math.isfinite(value['created_at'])
            or not isinstance(value.get('reason'), str) or not value['reason'].strip()):
        raise ValueError('Explicit user-authorized GPU5 completed-epoch stop required')
    actor = value.get('original_actor', {})
    if (not isinstance(actor, dict) or set(actor) != {'pid', 'create_time', 'command'}
            or type(actor.get('pid')) is not int or actor['pid'] <= 0
            or type(actor.get('create_time')) not in (float, int) or not math.isfinite(actor['create_time'])
            or not isinstance(actor.get('command'), list) or not actor['command']
            or any(not isinstance(arg, str) or not arg for arg in actor['command'])):
        raise ValueError('User authorization must identify the original single actor')
    return copy.deepcopy(value)


def _authorization(path, output, code, epoch, gpu):
    authorization = validate_authorization(_read(path), source_output=output, source_code=code,
                                         target_epoch=epoch, gpu=gpu)
    registration_path = output.parent / 'direct_training_registration.json'
    declaration_path = output.parent / 'direct_training_request.json'
    registration, declaration = _read(registration_path), _read(declaration_path)
    if (registration.get('format') != 'v24_existing_direct_GPU_exact_continuation_registration_v1'
            or declaration.get('format') != 'v24_existing_direct_GPU_exact_continuation_request_v1'
            or registration.get('GPU_arm') != gpu or declaration.get('GPU_arm') != gpu
            or registration.get('output') != str(output) or declaration.get('output') != str(output)
            or declaration.get('source_code') != str(code)
            or registration.get('request_sha256') != file_proof(declaration_path)['sha256']
            or {key: registration.get(key) for key in ('pid', 'create_time', 'command')} != authorization['original_actor']):
        raise ValueError('Authorization differs from the original own direct GNN registration')
    _actor_absent(authorization['original_actor'])
    return authorization


def validate_early_stop_pin(pin):
    """Metadata-only policy adapter; never projects this pin to full40/680."""
    selected = pin.get('selected', {}); contract = pin.get('model_contract', {})
    epoch = pin.get('completed_epochs'); updates = pin.get('optimizer_updates')
    if (pin.get('format') != EARLY_PIN_FORMAT or pin.get('physical_GPU') != 5
            or type(epoch) is not int or not 1 <= epoch < 40 or updates != 17 * epoch
            or pin.get('planned_early_stop_epoch') != epoch or pin.get('original_planned_epochs') != 40
            or pin.get('training_completion_kind') != COMPLETION_KIND or pin.get('status') != 'EARLY_STOPPED'
            or pin.get('full_training') is not False or pin.get('debug') is not False
            or pin.get('recipient_annotation_exposed') is not False
            or pin.get('selection_metric') != SELECTION_METRIC
            or pin.get('selection_scope') != 'this user-authorized early-stopped arm only; no cross-arm winner'
            or selected.get('gpu') != 5 or selected.get('best_record', {}).get('selected_by') != 'fixed_full128_validation_only'
            or selected.get('best_record', {}).get('candidate_universe') != 'all native P + fixed128U'
            or type(selected.get('best_record', {}).get('epoch')) is not int
            or not 1 <= selected['best_record']['epoch'] <= epoch
            or len(selected.get('best_record', {}).get('selection_key', [])) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in selected['best_record']['selection_key'])
            or contract.get('recipient_GT_used_in_forward') is not False
            or contract.get('trained_v23_weights_loaded') is not False or contract.get('encoder') != 'original_CNN_GAT'
            or type(contract.get('parameters')) is not int or contract['parameters'] <= 0
            or contract.get('trainable_parameters') != contract['parameters']
            or pin.get('stunet_checkpoint') is not None
            or pin.get('pin_sha256') != _canonical({k: v for k, v in pin.items() if k != 'pin_sha256'})):
        raise ValueError('Separate truthful GPU5 early-stop own-BEST pin required')
    for name in ('best_file_sha256', 'best_content_sha256', 'best_model_sha256', 'identity_sha256'):
        if not _checksum(selected.get(name)):
            raise ValueError('Actual early-stop BEST checksum missing: ' + name)
    authorization = pin.get('early_stop_authorization', {})
    if (set(authorization) != {'path', 'stat', 'sha256', 'request'} or not _checksum(authorization['sha256'])
            or not isinstance(authorization['stat'], list) or len(authorization['stat']) != 5):
        raise ValueError('Immutable early-stop authorization proof required')
    validate_authorization(authorization['request'], source_output=pin.get('source_output'),
                           source_code=pin.get('source_code'), target_epoch=epoch)
    for name in ('final_checkpoint_proof', 'BEST_checkpoint_proof'):
        proof = pin.get(name, {})
        if (set(proof.get('numerical_state_sha256', {})) != set(NUMERICAL)
                or any(not _checksum(value) for value in proof['numerical_state_sha256'].values())
                or not _checksum(proof.get('raw_sha256')) or not _checksum(proof.get('content_sha256'))
                or proof.get('identity_sha256') != selected['identity_sha256']):
            raise ValueError('Actual six-state early-stop checkpoint proof required')
    latest, best = pin['final_checkpoint_proof'], pin['BEST_checkpoint_proof']
    selected_record = {key: value for key, value in selected['best_record'].items() if key != 'candidate_universe'}
    exposure = pin.get('training_U_exposure', {})
    stage_u = exposure.get('stage_validation_active_u_by_epoch', [])
    if (latest.get('status') != 'PAUSED' or latest.get('phase') != 'training'
            or latest.get('epoch') != epoch + 1 or latest.get('updates') != updates
            or latest.get('history_epochs') != list(range(1, epoch + 1))
            or latest.get('optimizer_parameter_tensors') != 981
            or latest.get('train_position') != 0 or latest.get('evaluation_position') != 0
            or best.get('epoch') != selected_record['epoch'] or best.get('updates') != selected_record.get('updates')
            or selected_record.get('updates') != 17 * selected_record['epoch']
            or latest.get('best') != selected_record or best.get('best') != selected_record
            or best.get('raw_sha256') != selected['best_file_sha256']
            or best.get('content_sha256') != selected['best_content_sha256']
            or pin.get('original_actor_absent') is not True
            or len(stage_u) != epoch or any(type(value) is not int or not 1 <= value <= 128 for value in stage_u)
            or exposure.get('actual_full128_training_epochs') != sum(value == 128 for value in stage_u)
            or exposure.get('full128_validation_epochs') != epoch or exposure.get('next_active_u') != latest.get('active_u')):
        raise ValueError('Truthful paused boundary, own BEST and actual U exposure proofs required')
    return copy.deepcopy(pin)


def inspect_early_stop(source_output, source_code, inventory_path, *, input_cache,
                       target_epoch, authorization_path, gpu=5):
    """Read original six-state checkpoints on CPU after a verified epoch stop."""
    import torch
    from . import v24_nnunet_cp as original
    from .v24_training_continuation import _checkpoint
    from .u_bridge_training import digest
    from .v23_training import best_selection_key
    output = _regular(source_output, directory=True); code = _regular(source_code, directory=True)
    inventory = _regular(inventory_path); cache = _regular(input_cache, directory=True)
    authorization_path = _regular(authorization_path)
    authorization = _authorization(authorization_path, output, code, target_epoch, gpu)
    paths = [authorization_path, output.parent / 'direct_training_registration.json',
             output.parent / 'direct_training_request.json', output / 'request.json',
             output / 'calibration.json', output / 'training/training_identity.json',
             output / 'training/invocations.jsonl', output / 'training/checkpoint_latest.pt',
             output / 'training/checkpoint_best.pt', output / 'training/STOP_AFTER_BATCH', inventory]
    before = {str(path): file_proof(path) for path in paths}
    request = _read(output / 'request.json'); owner = _read(output / 'training/training_identity.json')
    binding = owner.get('binding', {})
    if (request.get('format') != 'v24_full_native_GT_free_execution_request_v1'
            or request.get('request_sha256') != _canonical({k: v for k, v in request.items() if k != 'request_sha256'})
            or request.get('physical_GPU') != gpu or request.get('inventory_sha256') != before[str(inventory)]['sha256']
            or _regular(request['inventory']) != inventory or _regular(request['input_cache'], directory=True) != cache
            or owner.get('identity_sha256') != digest(binding) or binding.get('identity') != request
            or binding.get('debug') is not False or binding.get('epochs') != 40):
        raise ValueError('Original immutable full40 planned scientific request/identity required')
    cli = original._completed_source_cli(code, request.get('source'), gpu)
    cli.validate_config(request['config'], gpu, None)
    interrupted = original.inspect_interrupted_provenance(output, code, gpu=gpu)
    calibration = _read(output / 'calibration.json')
    if (request.get('STU_checkpoint_sha256') is not None
            or calibration.get('request_sha256') != request['request_sha256'] or calibration.get('physical_GPU') != gpu
            or calibration != binding['config']['v24_runtime']['batch_calibration']
            or calibration.get('model_contract') != binding['model_contract']
            or calibration.get('initial_state_sha256') != binding['initial_state_sha256']
            or calibration.get('selected_physical_patient_batch') != 4
            or calibration.get('selected_physical_candidate_batch') != 32):
        raise ValueError('Original GPU5 CNN/GAT B4/chunk32 calibration must remain unchanged')
    train, val = binding['train_cases'], binding['val_cases']
    if (len(train) != 65 or len(set(train)) != 65 or len(val) != 21 or len(set(val)) != 21 or set(train) & set(val)):
        raise ValueError('Full65 ranking train and full21 validation populations required')
    latest, latest_proof = _checkpoint(output / 'training/checkpoint_latest.pt', owner)
    best, best_proof = _checkpoint(output / 'training/checkpoint_best.pt', owner)
    original._finite_checkpoint(latest); original._finite_checkpoint(best)
    state = latest['state']; history = state['history']; selected = state.get('best')
    if (state.get('status') != 'PAUSED' or state.get('phase') != 'training'
            or state.get('epoch') != target_epoch + 1 or len(history) != target_epoch
            or [row.get('epoch') for row in history] != list(range(1, target_epoch + 1))
            or state.get('updates') != 17 * target_epoch or len(set(state.get('connected', []))) != 981
            or len(state.get('connected', [])) != 981 or state.get('train_order') is not None
            or state.get('train_position') != 0 or state.get('train_rows') != []
            or state.get('evaluation_position') != 0 or state.get('evaluation_rows') != []
            or 'pending_epoch_completion' in state or selected is None
            or latest_proof.get('optimizer_parameter_tensors') != 981
            or state['curriculum'].get('last_completed_epoch') != target_epoch):
        raise ValueError('Actual PAUSED state after the requested whole epoch, before any next-epoch work required')
    reports = [row['full_validation'] for row in history]
    for epoch, report in enumerate(reports, 1):
        if (report.get('phase') != 'full_validation' or report.get('active_u') != 128 or report.get('epoch') != epoch
                or report.get('debug') is not False or report.get('recipient_GT_used_in_forward') is not False
                or report.get('all_P_scored') is not True or report.get('evaluation_view_epoch') != 29
                or [row.get('case_id') for row in report.get('cases', [])] != val
                or any(not math.isfinite(float(value)) for value in best_selection_key(report))):
            raise ValueError('Every completed epoch needs its actual complete fixed full128 validation')
    winner = max(range(target_epoch), key=lambda index: best_selection_key(reports[index]))
    if (selected.get('epoch') != winner + 1 or selected.get('updates') != 17 * (winner + 1)
            or selected.get('selection_key') != list(best_selection_key(reports[winner]))
            or selected.get('selected_by') != 'fixed_full128_validation_only'
            or best['state'].get('best') != selected or best['state'].get('epoch') != selected['epoch']
            or best['state'].get('updates') != selected['updates']
            or best['state'].get('pending_epoch_completion', {}).get('full_validation') != reports[winner]):
        raise ValueError('Actual own full128 BEST among all completed epochs required; no LAST or stage substitution')
    lines = (output / 'training/invocations.jsonl').read_bytes().splitlines(keepends=True)
    if not lines or any(line.strip() and not line.endswith(b'\n') for line in lines):
        raise ValueError('Stable complete PAUSED invocation JSONL required')
    rows = [json.loads(line) for line in lines if line.strip()]; invocation = rows[-1] if rows else {}
    if (invocation.get('status') != 'PAUSED' or invocation.get('full_training') is not False
            or invocation.get('debug') is not False or invocation.get('actual_CUDA') is not True
            or invocation.get('completed_epochs') != target_epoch or invocation.get('optimizer_updates') != 17 * target_epoch
            or invocation.get('connected_parameter_tensors') != 981 or invocation.get('expected_parameter_tensors') != 981
            or invocation.get('best') != selected or invocation.get('active_u') != state['curriculum']['active_u']):
        raise ValueError('Actual own PAUSED CUDA invocation at the requested epoch boundary required')
    record = copy.deepcopy(selected); record['candidate_universe'] = 'all native P + fixed128U'
    stage_u = [row.get('stage_validation', {}).get('active_u') for row in history]
    if any(type(value) is not int or not 1 <= value <= 128 for value in stage_u):
        raise ValueError('Actual training-stage U exposure required for every completed epoch')
    pin = dict(format=EARLY_PIN_FORMAT, physical_GPU=gpu, status='EARLY_STOPPED',
        training_completion_kind=COMPLETION_KIND, original_planned_epochs=40, planned_early_stop_epoch=target_epoch,
        completed_epochs=target_epoch, optimizer_updates=17 * target_epoch, full_training=False, debug=False,
        recipient_annotation_exposed=False, selection_metric=SELECTION_METRIC,
        selection_scope='this user-authorized early-stopped arm only; no cross-arm winner',
        source_output=str(output), source_code=str(code), inventory=str(inventory), input_cache=str(cache), stunet_checkpoint=None,
        model_contract=copy.deepcopy(binding['model_contract']), population=copy.deepcopy(binding['population']),
        final_checkpoint_proof=latest_proof, BEST_checkpoint_proof=best_proof,
        source_files_sha256=copy.deepcopy(request['source']),
        source_artifacts_sha256={name: before[str(output / name)]['sha256'] for name in
            ('request.json', 'calibration.json', 'training/training_identity.json', 'training/invocations.jsonl')},
        selected=dict(gpu=gpu, best_path=best_proof['path'], best_file_sha256=best_proof['raw_sha256'],
            best_content_sha256=best_proof['content_sha256'], best_model_sha256=digest(best['model']),
            identity_sha256=owner['identity_sha256'], best_record=record),
        admitted_source_experiment=str(output), admitted_source_request_sha256=request['request_sha256'],
        early_stop_authorization={**before[str(authorization_path)], 'request': authorization},
        early_stop_source_file_proofs=before, original_actor_absent=True,
        training_U_exposure=dict(stage_validation_active_u_by_epoch=stage_u,
            actual_full128_training_epochs=sum(value == 128 for value in stage_u),
            next_active_u=state['curriculum']['active_u'], full128_validation_epochs=target_epoch),
        interrupted_saved_state_provenance=dict(receipt=str(output / original.INTERRUPTED_RECEIPT),
            receipt_sha256=original.sha(output / original.INTERRUPTED_RECEIPT), original_source=interrupted['source'],
            execution_code=interrupted['execution_code'], execution_files=interrupted['execution_files']))
    for path, proof in before.items():
        if file_proof(path) != proof:
            raise ValueError('Original early-stop source changed during CPU admission: ' + path)
    _actor_absent(authorization['original_actor'])
    original._completed_source_cli(code, request['source'], gpu)
    original.inspect_interrupted_provenance(output, code, gpu=gpu)
    pin['pin_sha256'] = _canonical(pin)
    return validate_early_stop_pin(pin)


def admit_early_stop_pin(path, inventory):
    pin = validate_early_stop_pin(_read(path))
    current = inspect_early_stop(pin['source_output'], pin['source_code'], inventory,
        input_cache=pin['input_cache'], target_epoch=pin['completed_epochs'],
        authorization_path=pin['early_stop_authorization']['path'])
    if current != pin:
        raise ValueError('Pinned early-stop BEST/source/authorization proof changed')
    return pin, _read(Path(pin['source_output']) / 'request.json')


def pin_early_stopped(*, source_output, source_code, inventory_path, input_cache,
                      target_epoch, authorization_path, output, gpu=5):
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('Initial early-stop pin publication must precede CUDA initialization')
    pin = inspect_early_stop(source_output, source_code, inventory_path, input_cache=input_cache,
        target_epoch=target_epoch, authorization_path=authorization_path, gpu=gpu)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Fresh separate early-stop pin required; existing results preserved')
    for protected in (source_output, source_code, input_cache):
        if output.is_relative_to(Path(protected).resolve(strict=True)):
            raise ValueError('Early-stop pin cannot be written into original inputs/results')
    from .v24_nnunet_cp import new_json
    new_json(output, pin)
    return output
