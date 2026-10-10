"""Task-owned, CPU-only wait followed by one completed arm's native nnUNet.

No signal is sent to the running GNN or to a failed child. A durable exclusive
lease makes uncertain/repeated registration fail explicitly instead of replaying
stages or overwriting an experiment. This launcher never imports torch while
waiting; the pin-current-gnn child admits the actual completed own-arm BEST.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.dont_write_bytecode = True

FORMAT = 'v24_completed_gnn_to_native_chain_request_v1'
CURRENT_FORMAT = 'v24_frozen_current_GT_blind_GNN_native_nnunet_CP_v1'
CONTINUATION_PIPELINE_FORMAT = 'v24_GPU4_exact_saved_state_continuation_pipeline_v1'
CONTINUATION_FORMAT = 'v24_GPU4_exact_saved_state_continuation_v1'
CONTINUATION_RECEIPT = 'GPU4_exact_continuation.json'
CONTINUATION_FILES = ('tools/run_v24_gpu4_exact_continuation.py', 'tools/run_v24_gpu4_pipeline.py')
NUMERICAL_STATES = ('model', 'optimizer', 'scheduler', 'scaler', 'rank_rng', 'shuffle_generator')
READONLY_PROFILE = 'reuse_full105_static_raw_CTseg_and_full131_Blosc2_readonly'
READONLY_ENTRY = 'tools/run_v24_readonly_nnunet_cp.py'
READONLY_HELPER = 'hiercp_v1x/v24_readonly_native_storage.py'
STATIC_EXTENSION = 'reuse_full105_exact_static_raw_case_operators_readonly_v1'
STATIC_HELPER = 'hiercp_v1x/v24_readonly_static_operator_storage.py'
ACTIONS = ('pin-current-gnn', 'prepare-current-bank', 'prepare-native', 'calibrate-native', 'train')
AFFINITIES = {5: [42, 43, 44, 45], 6: [32, 33, 34, 35]}
PARAMETER_TENSORS = {4: 537, 5: 981, 6: 537}
POLL_SECONDS = 5


class AllocationUnavailable(RuntimeError):
    """An owned reservation is insufficient; wait without launching a GPU child."""


class AssignedGPUUnavailable(RuntimeError):
    """The assigned device is occupied; leave all existing applications alone."""


def owned(path, *, directory=False, must_exist=True):
    """Reject path aliases and require ownership of the actual input/output."""
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Absolute non-traversing path required: ' + str(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('Symlink path is not an owned immutable input: ' + str(part))
    if not path.exists():
        if must_exist:
            raise FileNotFoundError(path)
        return path
    info = path.stat()
    if hasattr(os, 'getuid') and info.st_uid != os.getuid():
        raise ValueError('Input/output belongs to a different UID: ' + str(path))
    if directory != stat.S_ISDIR(info.st_mode) or (not directory and not stat.S_ISREG(info.st_mode)):
        raise ValueError('Owned regular file/directory required: ' + str(path))
    return path


def sha(path):
    path = owned(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(owned(path).read_text(encoding='utf8'))


def last_jsonl(path):
    lines = [line for line in owned(path).read_text(encoding='utf8').splitlines() if line.strip()]
    if not lines:
        raise ValueError('Actual completion receipt is empty: ' + str(path))
    return json.loads(lines[-1])


def _checksum(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def _assigned_cpu_contract(gpu, affinity):
    if gpu in AFFINITIES:
        return affinity == AFFINITIES[gpu]
    return (gpu == 4 and isinstance(affinity, list) and len(affinity) == 4
            and all(type(core) is int and core >= 0 for core in affinity)
            and affinity == sorted(set(affinity)))


def _gpu4_full_u(runtime):
    curriculum = runtime.get('curriculum', {})
    return (runtime.get('all_U_from_epoch_one') is True and curriculum.get('initial_u') == 128
            and curriculum.get('total_u') == 128 and curriculum.get('total_epochs') == 40)


def _inside(path, parent):
    try:
        Path(path).relative_to(parent)
    except ValueError as error:
        raise ValueError('Stage output escapes the fresh arm directory: ' + str(path)) from error


def command_options(command, *, python, entry, action):
    if not isinstance(command, list) or any(not isinstance(x, str) or '\x00' in x for x in command):
        raise ValueError('Explicit argument array required; shell commands are not accepted')
    if not command or command[0] != python:
        raise ValueError('The pinned Python interpreter must execute every child')
    args = command[1:]
    flags = []
    while args and args[0] in ('-B', '-u'):
        flags.append(args.pop(0))
    if set(flags) != {'-B', '-u'} or len(flags) != 2 or args[:2] != [entry, action]:
        raise ValueError('Pinned unbuffered API entry/action required')
    args = args[2:]
    if len(args) % 2:
        raise ValueError('Native stages accept only explicit flag/value pairs')
    options = {}
    for flag, value in zip(args[::2], args[1::2]):
        if not flag.startswith('--') or flag in options:
            raise ValueError('Duplicate or invalid native argument: ' + flag)
        options[flag] = value
    return options


def _readonly_storage(request):
    profile = request.get('storage_profile')
    if profile is None:
        if any(key in request for key in ('storage_admission', 'storage_admission_sha256', 'storage_admission_stat',
                                          'source_execution_code', 'source_execution_commit', 'source_execution_files',
                                          'storage_extension', 'storage_handoff')):
            raise ValueError('Explicit reviewed read-only storage profile required for storage/execution overrides')
        return None
    if (profile != READONLY_PROFILE or request.get('GPU_arm') not in (4, 5, 6)
            or request.get('recovery') is not None):
        raise ValueError('The immutable full-cache storage profile requires an explicitly admitted fresh own-arm chain')
    path = owned(request['storage_admission'])
    if (not _checksum(request.get('storage_admission_sha256'))
            or sha(path) != request['storage_admission_sha256']
            or _file_stat(path) != request.get('storage_admission_stat')):
        raise ValueError('Reviewed full-cache storage admission SHA/stat changed')
    from hiercp_v1x.v24_readonly_native_storage import verify_admission
    document = read(path)
    verify_admission(document, full_hash=False)
    extension = request.get('storage_extension')
    if document.get('storage_extension') != extension or extension not in (None, STATIC_EXTENSION):
        raise ValueError('The supplemental static operator admission differs from its explicit request')
    if request['GPU_arm'] in (5, 6) and extension != STATIC_EXTENSION:
        raise ValueError('GPU5/6 require the explicit full105 static-operator extension')
    if extension is not None:
        from hiercp_v1x.v24_readonly_static_operator_storage import verify_admission as verify_static
        verify_static(document, full_hash=False)
        aggregate = document.get('aggregate_budget', {})
        if (type(request.get('min_free_disk_GiB')) not in (int, float)
                or not math.isfinite(request['min_free_disk_GiB'])
                or type(aggregate.get('required_free_bytes')) is not int
                or aggregate.get('global_peak_checkpoint_slots') != 7
                or document.get('checkpoint_coordination', {}).get('format') != 'original_native_final_publication_flock_global7_v1'
                or document['checkpoint_coordination'].get('global_peak_checkpoint_slots') != 7
                or document['checkpoint_coordination'].get('per_arm_peak_checkpoint_slots') != 3
                or document['checkpoint_coordination'].get('optimizer_work_locked') is not False
                or request['min_free_disk_GiB'] * 2**30 < aggregate['required_free_bytes']
                or aggregate.get('arms', {}).get('gpu' + str(request['GPU_arm']), {}).get('root') != request['chain_root']
                or STATIC_HELPER not in request.get('code_files', {})):
            raise ValueError('Complete three-arm original checkpoint/growth storage reservation required')
    if request['GPU_arm'] in (5, 6):
        if (request.get('source_execution_code') != request['source_code']
                or request.get('source_execution_commit') != request['source_commit']
                or request.get('source_execution_files') != request['source_files']):
            raise ValueError('GPU5/6 storage cannot retarget their original scientific execution checkout')
        if request.get('storage_handoff') is None:
            raise ValueError('Existing GPU5/6 exclusive source claims require an append-only storage handoff')
    elif request.get('storage_handoff') is not None:
        raise ValueError('GPU4 retains its ordinary fresh source claim')
    budget = document.get('budget', {})
    estimate, reserve, required = (budget.get(key) for key in
        ('new_writable_bytes_estimate', 'checkpoint_reserve_bytes', 'required_free_bytes'))
    if (document.get('format') != 'v24_native_immutable_full_cache_storage_v1'
            or document.get('profile') != READONLY_PROFILE
            or document.get('model_data_scale_preserved') is not True or document.get('debug') is not False
            or document.get('scores_reused') is not False or document.get('learned_upper_reused') is not False
            or document.get('inventory', {}).get('path') != request['inventory']
            or document['inventory'].get('sha256') != sha(request['inventory'])
            or document.get('baseline', {}).get('preprocessed') != request['baseline_preprocessed']
            or type(estimate) is not int or estimate < 0 or type(reserve) is not int or reserve != 10 * 2**30
            or type(required) is not int or required != estimate + reserve
            or budget.get('minimum_runtime_free_bytes') != 10 * 2**30
            or type(request.get('min_free_disk_GiB')) not in (int, float)
            or not math.isfinite(request['min_free_disk_GiB']) or request['min_free_disk_GiB'] * 2**30 < required
            or type(request.get('minimum_runtime_free_disk_GiB', 10)) not in (int, float)
            or not math.isfinite(request.get('minimum_runtime_free_disk_GiB', 10))
            or request.get('minimum_runtime_free_disk_GiB', 10) * 2**30 < budget['minimum_runtime_free_bytes']):
        raise ValueError('Read-only full105/full131 storage and actual new-write budget plus10GiB reserve required')
    if any(name not in request.get('code_files', {}) for name in (READONLY_ENTRY, READONLY_HELPER)):
        raise ValueError('The actual read-only native entry/helper must be SHA pinned')
    return document


def _source_execution(request):
    if request.get('storage_profile') == READONLY_PROFILE:
        return request['source_execution_code'], request['source_execution_commit'], request['source_execution_files']
    return request['code'], request['commit'], request['code_files']


def validate_request(request, *, fresh=True):
    gpu = request.get('GPU_arm')
    if (request.get('format') != FORMAT or type(gpu) is not int or gpu not in (4, 5, 6)
            or not _assigned_cpu_contract(gpu, request.get('CPU_affinity'))
            or request.get('scoring_RAM_GiB') != 64 or request.get('native_RAM_GiB') != 48
            or (request.get('storage_profile') is None and request.get('min_free_disk_GiB', 0) < 160)
            or request.get('minimum_runtime_free_disk_GiB', 10) < 10
            or not isinstance(request.get('GPU_UUID'), str) or not request['GPU_UUID'].startswith('GPU-')):
        raise ValueError('Explicitly assigned GPU4/5/6 four-CPU/full RAM/disk contract required')
    chain = owned(request['chain_root'], directory=True, must_exist=False)
    owned(chain.parent, directory=True)
    source = owned(request['source_output'], directory=True)
    storage = _readonly_storage(request)
    if chain == source or chain in source.parents or source in chain.parents:
        raise ValueError('Fresh chain output must not overlap a GNN source experiment')
    claim_path = source.parent / 'gnn_native_chain_claims_20261010' / (hashlib.sha256(str(source).encode()).hexdigest() + '.json')
    if request.get('source_claim_path') != str(claim_path):
        raise ValueError('One globally derived exclusive claim is required for each completed GNN source')
    owned(claim_path.parent, directory=True)
    owned(claim_path, must_exist=False)
    code_fields = [('code', 'commit', 'code_files'), ('source_code', 'source_commit', 'source_files')]
    if storage is not None:
        code_fields.append(('source_execution_code', 'source_execution_commit', 'source_execution_files'))
    for key, commit_key, files_key in code_fields:
        owned(request[key], directory=True)
        commit = request.get(commit_key)
        if not isinstance(commit, str) or len(commit) != 40 or set(commit) - set('0123456789abcdef'):
            raise ValueError('Pinned full Git commit required: ' + key)
        files = request.get(files_key)
        if not isinstance(files, dict) or not files:
            raise ValueError('Actual immutable source SHA witnesses required: ' + key)
        for name, checksum in files.items():
            if Path(name).is_absolute() or '..' in Path(name).parts or not _checksum(checksum):
                raise ValueError('Invalid source SHA witness')
            owned(Path(request[key]) / name)
    if 'tools/watch_v24_gnn_to_nnunet.py' not in request['code_files'] or 'tools/run_v24_nnunet_cp.py' not in request['code_files']:
        raise ValueError('The actual chain supervisor and native API must be SHA pinned')
    owned(request['python'])
    for name in ('inventory', 'baseline_preprocessed', 'input_cache'):
        owned(request[name], directory=name != 'inventory')
    if gpu in (4, 6):
        owned(request['stunet_checkpoint'])
        scheduler = request.get('scheduler')
        if gpu == 4:
            if not _checksum(request.get('stunet_checkpoint_sha256')) or sha(request['stunet_checkpoint']) != request['stunet_checkpoint_sha256']:
                raise ValueError('GPU4 exact original official STU checkpoint SHA required')
            if scheduler is not None and (not isinstance(scheduler, dict) or scheduler.get('state') != 'R'
                    or not isinstance(scheduler.get('job_id'), str) or not scheduler['job_id']
                    or scheduler['job_id'] == '129443.ECE-util1'
                    or scheduler.get('expected_owner') != 'aicompetition06'
                    or not isinstance(scheduler.get('expected_node'), str) or not scheduler['expected_node']
                    or scheduler.get('minimum_remaining_seconds', 0) < 28800
                    or scheduler.get('native_minimum_remaining_seconds', 25200) < 25200
                    or not isinstance(scheduler.get('expires_at_unix'), (int, float))):
                raise ValueError('GPU4 requires its own direct allocation or distinct owned PBS reservation')
        elif (not isinstance(scheduler, dict) or scheduler.get('state') != 'R'
                or scheduler.get('job_id') != '129443.ECE-util1'
                or scheduler.get('expected_owner') != 'aicompetition06'
                or scheduler.get('expected_node') != 'ece-a6gpu6'
                or scheduler.get('minimum_remaining_seconds', 0) < 28800
                or scheduler.get('native_minimum_remaining_seconds', 25200) < 25200
                or not isinstance(scheduler.get('expires_at_unix'), (int, float))):
            raise ValueError('GPU6 original owned PBS reservation and explicit remaining-time floor required')
    elif request.get('stunet_checkpoint') is not None or request.get('scheduler') is not None:
        raise ValueError('GPU5 must retain its own CNN source and direct owned reservation')
    job = request.get('source_job', {})
    owned(job['status'])
    if (type(job.get('worker_pid')) is not int or job['worker_pid'] <= 0
            or not isinstance(job.get('worker_create_time'), (float, int))
            or not isinstance(job.get('worker_command'), list) or not job['worker_command']
            or job.get('training_stage') != 'train' or not _checksum(job.get('pipeline_request_sha256'))):
        raise ValueError('Exact original owned worker identity and final train stage required')
    owned(job['worker_command'][-1])
    stages = request.get('stages')
    if not isinstance(stages, list) or tuple(row.get('action') for row in stages) != ACTIONS:
        raise ValueError('Exactly pin, complete bank, native preparation, calibration, full train required')
    entry = str(Path(request['code']) / (READONLY_ENTRY if storage is not None else 'tools/run_v24_nnunet_cp.py'))
    pin, bank, native = str(chain / 'pin.json'), str(chain / 'bank/index.json'), str(chain / 'native/native.json')
    common = {'--gpu': str(gpu)}
    if storage is not None:
        common['--storage-admission'] = request['storage_admission']
    expected = [dict(common, **{'--source-output': request['source_output'], '--source-code': request['source_code'],
                '--inventory': request['inventory'], '--input-cache': request['input_cache'], '--output': pin}),
        dict(common, **{'--pin': pin, '--inventory': request['inventory'], '--baseline-preprocessed': request['baseline_preprocessed'],
                '--input-cache': request['input_cache'], '--output': str(chain / 'bank')}),
        dict(common, **{'--bank': bank, '--output': str(chain / 'native')}),
        dict(common, **{'--native': native}), dict(common, **{'--native': native})]
    if gpu in (4, 6):
        for options in expected[:2]:
            options['--stunet-checkpoint'] = request['stunet_checkpoint']
    for action, row, options in zip(ACTIONS, stages, expected):
        if row.get('name') != action.replace('-', '_'):
            raise ValueError('Unambiguous stage log name required')
        actual = command_options(row.get('command'), python=request['python'], entry=entry, action=action)
        if actual != options:
            raise ValueError('A stage changed its arm, source, full data, output, or API: ' + action)
    if fresh and chain.exists():
        allowed = {'request.json', 'watcher.log', 'registration.json'}
        if any(p.name not in allowed for p in chain.iterdir()):
            raise FileExistsError('Existing lease/stage/output found; explicit recovery required, no replay')
        for path in chain.iterdir():
            owned(path)
    if request.get('recovery') is not None:
        _validate_recovery(request)
    if request.get('storage_handoff') is not None:
        _validate_storage_handoff(request)
    return request


def verify_code(request):
    fields = [('code', 'code_files', 'commit'), ('source_code', 'source_files', 'source_commit')]
    if request.get('storage_profile') == READONLY_PROFILE:
        fields.append(('source_execution_code', 'source_execution_files', 'source_execution_commit'))
    for root_key, files_key, commit_key in fields:
        code = owned(request[root_key], directory=True)
        actual = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != request[commit_key]:
            raise ValueError('Pinned immutable Git checkout changed: ' + str(code))
        for name, expected in request[files_key].items():
            if sha(code / name) != expected:
                raise ValueError('Actual source changed: ' + str(code / name))


def process_witness(pid, birth, command=None):
    """A zombie is still present; PID reuse or unknown identity fails closed."""
    import psutil
    try:
        process = psutil.Process(pid)
        actual_birth = process.create_time()
        if abs(actual_birth - birth) > .001 or (hasattr(os, 'getuid') and process.uids().real != os.getuid()):
            raise ValueError('Owned exact process identity changed: ' + str(pid))
        actual_status = process.status()
        if command is not None and actual_status != psutil.STATUS_ZOMBIE and process.cmdline() != command:
            raise ValueError('Exact GNN worker command changed: ' + str(pid))
        return dict(pid=pid, create_time=actual_birth, status=actual_status)
    except psutil.NoSuchProcess:
        return None


def source_completion(request, status, worker, child):
    job = request['source_job']
    if (status.get('worker_pid') != job['worker_pid']
            or abs(status.get('worker_create_time', -1) - job['worker_create_time']) > .001
            or status.get('actual_CPU_affinity') != request['CPU_affinity']):
        raise ValueError('Source worker ownership/CPU proof differs')
    source = Path(request['source_output'])
    for marker in (source / 'training/STOP_AFTER_BATCH', Path(job['status']).parent / 'STOP_BEFORE_NEXT_STAGE'):
        if marker.exists():
            raise RuntimeError('GNN has a cooperative pause marker; downstream was not started: ' + str(marker))
    outcome = status.get('status')
    if outcome in ('FAILED', 'PAUSED', 'PAUSED_BETWEEN_STAGES'):
        raise RuntimeError('GNN source is ' + outcome + '; downstream was not started')
    if outcome != 'COMPLETE':
        if worker is None:
            raise RuntimeError('GNN worker disappeared without COMPLETE; preserve source and inspect its log')
        return None
    if (status.get('stage') != job['training_stage'] or not status.get('stages_completed')
            or status['stages_completed'][-1] != job['training_stage']):
        raise ValueError('Source pipeline completed without completing the actual final GNN train stage')
    if any(witness is not None and witness['status'] != 'zombie' for witness in (worker, child)):
        return None
    receipt = last_jsonl(source / 'training/invocations.jsonl')
    count = PARAMETER_TENSORS[request['GPU_arm']]
    if (receipt.get('status') != 'COMPLETE' or receipt.get('full_training') is not True
            or receipt.get('debug') is not False or receipt.get('actual_CUDA') is not True
            or receipt.get('completed_epochs') != 40 or receipt.get('optimizer_updates') != 680
            or receipt.get('connected_parameter_tensors') != count
            or receipt.get('expected_parameter_tensors') != count
            or receipt.get('recipient_GT_used_in_forward') is not False
            or receipt.get('checkpoint') != str(source / 'training/checkpoint_latest.pt')
            or receipt.get('best', {}).get('selected_by') != 'fixed_full128_validation_only'):
        raise ValueError('Actual full40/680 CUDA connected own-arm training completion required')
    arm = read(source / 'request.json')
    if arm.get('physical_GPU') != request['GPU_arm']:
        raise ValueError('Completed GNN receipt belongs to another GPU arm')
    # No torch import/checkpoint unpickling here. The first pinned API child
    # independently proves every epoch, actual BEST, all six saved states.
    return dict(invocation_sha256=sha(source / 'training/invocations.jsonl'), invocation=receipt,
                checkpoint_validation_delegated_to='pin-current-gnn', worker_and_child_finished=True)


def _validate_gpu4_pipeline_inputs(request, pipeline):
    """Admit a fresh source before its calibration/training ownership exists."""
    from tools.run_v24_all_p import validate_config
    code = Path(request['source_code'])
    config_path = owned(pipeline['source_config'])
    _inside(config_path, code)
    name = config_path.relative_to(code).as_posix()
    config_checksum = sha(config_path)
    config = read(config_path)
    if (request['source_files'].get(name) != config_checksum
            or pipeline.get('source_config_sha256') != config_checksum
            or pipeline.get('gnn_config') != config
            or sha(request['stunet_checkpoint']) != request['stunet_checkpoint_sha256']):
        raise ValueError('GPU4 actual immutable full source configuration/SHA required before waiting')
    validate_config(config, 4, request['stunet_checkpoint'])
    if pipeline.get('format') == CONTINUATION_PIPELINE_FORMAT:
        _validate_gpu4_continuation_pipeline(request, pipeline, config_path)
        return
    stages = pipeline.get('stages', [])
    if tuple(stage.get('name') for stage in stages) != ('prepare_inputs', 'calibrate', 'train'):
        raise ValueError('GPU4 requires its exact fresh prepare_inputs/calibrate/train pipeline')
    entry = str(owned(code / 'tools/run_v24_gpu4_all_u.py'))
    native_experiment = None
    for stage, mode in zip(stages, ('prepare', 'calibrate', 'train')):
        command = stage.get('command')
        if (not isinstance(command, list) or any(not isinstance(arg, str) or '\x00' in arg for arg in command)
                or command[:4] != [request['python'], '-B', '-u', entry]
                or len(command[4:]) % 2):
            raise ValueError('GPU4 exact pinned fresh runner argument array required')
        options = {}
        for flag, value in zip(command[4::2], command[5::2]):
            if not flag.startswith('--') or flag in options:
                raise ValueError('GPU4 repeated/invalid fresh source option')
            options[flag] = value
        actual_native = str(owned(options['--native-experiment'], directory=True))
        if native_experiment is None:
            native_experiment = actual_native
        expected = {'--mode': mode, '--config': str(config_path), '--native-experiment': native_experiment,
            '--inventory': request['inventory'], '--input-cache': request['input_cache'],
            '--output': request['source_output'], '--gpu': '4', '--stunet-checkpoint': request['stunet_checkpoint'],
            '--cpu-affinity': ','.join(str(core) for core in request['CPU_affinity'])}
        if options != expected:
            raise ValueError('GPU4 fresh stage changed full source inputs/config/GPU/CPU or added an unapproved option')


def _file_stat(path):
    value = owned(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _validate_gpu4_continuation_pipeline(request, pipeline, config_path):
    """Bind a new execution overlay to one untouched failed scientific source."""
    execution_code, execution_commit, execution_files = _source_execution(request)
    if (request.get('recovery') is not None
            or pipeline.get('code') != execution_code or pipeline.get('commit') != execution_commit
            or pipeline.get('execution_code') != execution_code or pipeline.get('execution_commit') != execution_commit
            or pipeline.get('source_code') != request['source_code']
            or pipeline.get('source_commit') != request['source_commit']
            or not _checksum(pipeline.get('latest_checkpoint_sha256'))
            or type(pipeline.get('durable_optimizer_updates')) is not int
            or pipeline['durable_optimizer_updates'] != 1):
        raise ValueError('GPU4 exact continuation requires its new overlay, original science and durable update1; no claim recovery')
    files = pipeline.get('execution_files')
    if (not isinstance(files, dict) or set(files) != set(CONTINUATION_FILES)
            or any(not _checksum(value) or execution_files.get(name) != value
                   or sha(Path(execution_code) / name) != value for name, value in files.items())):
        raise ValueError('GPU4 continuation exact new execution source SHA manifest differs')
    if request.get('storage_profile') == READONLY_PROFILE:
        if (execution_files != files or subprocess.check_output(
                ['git', '-C', execution_code, 'rev-parse', 'HEAD'], text=True).strip() != execution_commit):
            raise ValueError('Original separately pinned live continuation execution checkout changed')
    previous = owned(pipeline['source_output'], directory=True)
    destination = Path(request['source_output'])
    if (previous == destination or previous in destination.parents or destination in previous.parents
            or previous == Path(request['chain_root']) or previous in Path(request['chain_root']).parents):
        raise ValueError('GPU4 continuation requires a fresh disjoint source and new source claim')
    previous_job = owned(pipeline['source_job'], directory=True)
    previous_request = read(previous_job / 'request.json')
    previous_status = read(previous_job / 'status.json')
    if (previous_status.get('request') != previous_request or previous_status.get('status') != 'FAILED'
            or previous_status.get('stage') != 'train' or previous_request.get('GPU') != 4
            or previous_request.get('GPU_uuid') != request['GPU_UUID']
            or previous_request.get('code') != request['source_code']
            or previous_request.get('commit') != request['source_commit']
            or previous_request.get('production_output') != str(previous)
            or previous_request.get('gnn_config') != pipeline['gnn_config']
            or previous_request.get('source_config') != str(config_path)
            or previous_request.get('source_config_sha256') != pipeline['source_config_sha256']
            or previous_request.get('CPU_affinity') != request['CPU_affinity']
            or tuple(previous_request.get(key) for key in ('epochs', 'physical_patient_batch', 'candidate_chunk')) != (40, 4, 64)):
        raise ValueError('GPU4 continuation original failed full source/config/job binding differs')
    for name in ('worker', 'child'):
        pid, birth = previous_status.get(name + '_pid'), previous_status.get(name + '_create_time')
        if (type(pid) is not int or type(birth) not in (int, float)
                or process_witness(pid, birth) is not None):
            raise ValueError('GPU4 continuation original exact worker/child must be absent; no overlapping source')
    stages = pipeline.get('stages', [])
    if tuple(stage.get('name') for stage in stages) != ('prepare_continuation', 'train'):
        raise ValueError('GPU4 continuation requires exact prepare_continuation/train stages')
    old_trains = [stage for stage in previous_request.get('stages', []) if stage.get('name') == 'train']
    if len(old_trains) != 1 or previous_request['stages'][-1] != old_trains[0]:
        raise ValueError('GPU4 continuation requires one original final science train command')
    old_command = old_trains[0].get('command', [])
    if old_command[:4] != [request['python'], '-B', '-u', str(Path(request['source_code']) / 'tools/run_v24_gpu4_all_u.py')]:
        raise ValueError('GPU4 continuation original scientific runner command differs')
    entry = str(owned(Path(execution_code) / CONTINUATION_FILES[0]))
    for stage, action in zip(stages, ('prepare', 'train')):
        command = stage.get('command')
        if (not isinstance(command, list) or any(not isinstance(arg, str) or '\x00' in arg for arg in command)
                or command[:4] != [request['python'], '-B', '-u', entry] or len(command[4:]) % 2):
            raise ValueError('GPU4 continuation exact overlay argument array required')
        options = {}
        for flag, value in zip(command[4::2], command[5::2]):
            if not flag.startswith('--') or flag in options:
                raise ValueError('GPU4 continuation repeated/invalid source option')
            options[flag] = value
        native = str(owned(options['--native-experiment'], directory=True))
        science = {'--mode': 'train', '--config': str(config_path), '--native-experiment': native,
            '--inventory': request['inventory'], '--input-cache': request['input_cache'],
            '--output': request['source_output'], '--gpu': '4', '--stunet-checkpoint': request['stunet_checkpoint'],
            '--cpu-affinity': ','.join(str(core) for core in request['CPU_affinity'])}
        expected = dict(science, **{'--action': action, '--source-output': str(previous),
            '--source-code': request['source_code'], '--source-job': str(previous_job)})
        original_science = dict(science, **{'--output': str(previous)})
        old_pairs = old_command[4:]
        if len(old_pairs) % 2 or len(set(old_pairs[::2])) != len(old_pairs[::2]):
            raise ValueError('GPU4 continuation original scientific arguments are invalid')
        if options != expected or dict(zip(old_pairs[::2], old_pairs[1::2])) != original_science:
            raise ValueError('GPU4 continuation changed original science/GPU/full128/B4/CPU or supplied resume/extra flags')


def _validate_gpu4_continuation_receipt(request, pipeline):
    """Inspect immutable small proofs; final actual checkpoint proof belongs to pin."""
    destination = Path(request['source_output'])
    document = read(destination / CONTINUATION_RECEIPT)
    source = document.get('source', {})
    previous = Path(pipeline['source_output'])
    previous_job = Path(pipeline['source_job'])
    flags = ('scientific_request_unchanged', 'checkpoint_identity_and_content_unchanged',
             'six_numerical_states_unchanged', 'curriculum_history_and_partial_cursors_unchanged')
    execution_code, execution_commit, _ = _source_execution(request)
    if (document.get('format') != CONTINUATION_FORMAT or document.get('status') != 'PREPARED_EXACT_SAVED_STATE'
            or document.get('destination') != str(destination)
            or document.get('source_code') != request['source_code'] or document.get('source_commit') != request['source_commit']
            or document.get('execution_code') != execution_code or document.get('execution_commit') != execution_commit
            or document.get('execution_files_sha256') != pipeline['execution_files']
            or document.get('original_files_modified') is not False
            or type(document.get('production_optimizer_updates_performed')) is not int
            or document['production_optimizer_updates_performed'] != 0
            or any(document.get(key) is not True for key in flags)
            or source.get('source_output') != str(previous) or source.get('source_code') != request['source_code']
            or source.get('source_commit') != request['source_commit'] or source.get('BEST', 'missing') is not None):
        raise ValueError('GPU4 continuation actual unchanged saved-state/source/overlay receipt required')
    for relative, raw_key in (('request.json', 'request_raw_sha256'), ('calibration.json', 'calibration_raw_sha256'),
                              ('training/training_identity.json', 'training_identity_raw_sha256')):
        checksum = sha(previous / relative)
        copied = document.get('copied_files', {}).get(relative, {})
        if (source.get(raw_key) != checksum or sha(destination / relative) != checksum
                or copied.get('sha256') != checksum or copied.get('source') != str(previous / relative)
                or copied.get('source_stat') != _file_stat(previous / relative)
                or copied.get('bytes') != (previous / relative).stat().st_size):
            raise ValueError('GPU4 continuation original/copied immutable metadata bytes changed: ' + relative)
    scientific = read(previous / 'request.json')
    owner = read(previous / 'training/training_identity.json')
    files = source.get('source_files_sha256')
    if (source.get('request_sha256') != scientific.get('request_sha256')
            or source.get('training_identity_sha256') != owner.get('identity_sha256')
            or scientific.get('config') != pipeline['gnn_config']
            or not isinstance(files, dict) or files != scientific.get('source')
            or any(request['source_files'].get(name) != value or not _checksum(value) for name, value in files.items())):
        raise ValueError('GPU4 continuation original scientific identity/source SHA proof differs')
    latest = source.get('latest', {})
    numerical = latest.get('numerical_state_sha256')
    if (latest.get('path') != str(previous / 'training/checkpoint_latest.pt')
            or latest.get('source_stat') != _file_stat(previous / 'training/checkpoint_latest.pt')
            or latest.get('raw_sha256') != pipeline['latest_checkpoint_sha256']
            or latest.get('identity_sha256') != owner.get('identity_sha256')
            or latest.get('updates') != pipeline['durable_optimizer_updates']
            or latest.get('epoch') != 1 or latest.get('phase') != 'training' or latest.get('status') != 'RUNNING'
            or latest.get('active_u') != 128 or latest.get('train_position') != 4
            or latest.get('optimizer_parameter_tensors') != 537 or latest.get('history_epochs') != []
            or latest.get('best', 'missing') is not None
            or not isinstance(numerical, dict) or set(numerical) != set(NUMERICAL_STATES)
            or any(not _checksum(value) for value in numerical.values())
            or any(not _checksum(latest.get(key)) for key in ('content_sha256', 'state_sha256', 'curriculum_sha256'))):
        raise ValueError('GPU4 continuation exact durable update1/U128/537/all six numerical states required')
    copied_latest = document.get('copied_files', {}).get('training/checkpoint_latest.pt', {})
    if (copied_latest.get('sha256') != latest['raw_sha256']
            or copied_latest.get('source') != latest['path'] or copied_latest.get('source_stat') != latest['source_stat']
            or copied_latest.get('bytes') != latest['source_stat'][2]):
        raise ValueError('GPU4 continuation exact original checkpoint copy proof differs')
    previous_status = read(previous_job / 'status.json')
    job = source.get('source_job', {})
    if (job.get('path') != str(previous_job) or job.get('status') != 'FAILED' or job.get('stage') != 'train'
            or job.get('quota_failure_verified') is not True
            or job.get('request_sha256') != sha(previous_job / 'request.json')
            or job.get('status_sha256') != sha(previous_job / 'status.json')
            or job.get('status_stat') != _file_stat(previous_job / 'status.json')):
        raise ValueError('GPU4 continuation failed original job proof changed')
    for name in ('worker', 'child'):
        expected = dict(pid=previous_status[name + '_pid'], create_time=previous_status[name + '_create_time'], absent=True)
        if job.get('processes', {}).get(name) != expected:
            raise ValueError('GPU4 continuation original worker/child absence proof differs')
    if (not isinstance(document.get('derived_logs'), dict) or not isinstance(document.get('original_full_logs'), dict)
            or set(document['derived_logs']) != set(document['original_full_logs'])):
        raise ValueError('GPU4 continuation requires actual saved-boundary/full-original log lineage')
    for name, row in document['derived_logs'].items():
        lineage = document['original_full_logs'].get(name, {})
        path = destination / row.get('lineage_path', '')
        _inside(path, destination / 'source_lineage/training')
        if (row.get('path') != 'training/' + name or row.get('boundary_updates') != 1 or row.get('boundary_epoch') != 1
                or not _checksum(row.get('sha256')) or row.get('source_sha256') != lineage.get('sha256')
                or lineage.get('source') != str(previous / 'training' / name)
                or sha(path) != lineage.get('sha256') or sha(previous / 'training' / name) != lineage.get('sha256')):
            raise ValueError('GPU4 continuation durable boundary/full log lineage differs')
    if not isinstance(document.get('excluded_partial_files'), list):
        raise ValueError('GPU4 continuation requires preserved partial-checkpoint inventory')
    actual_partial = {str(path) for path in (previous / 'training').iterdir()
                      if path.name.startswith('checkpoint_latest.pt.') and path.suffix == '.tmp'}
    declared_partial = []
    for row in document['excluded_partial_files']:
        path = owned(row['path'])
        if (str(path) not in actual_partial or row.get('stat') != _file_stat(path)
                or not _checksum(row.get('sha256')) or (destination / 'training' / path.name).exists()):
            raise ValueError('GPU4 continuation original partial checkpoint must remain preserved and excluded')
        declared_partial.append(str(path))
    if len(declared_partial) != len(set(declared_partial)) or set(declared_partial) != actual_partial:
        raise ValueError('GPU4 continuation partial-checkpoint inventory differs')


def inspect_source(request):
    _readonly_storage(request)
    status = read(request['source_job']['status'])
    job = request['source_job']
    pipeline_request_path = job['worker_command'][-1]
    if sha(pipeline_request_path) != job['pipeline_request_sha256']:
        raise ValueError('Registered original worker request changed')
    pipeline = read(pipeline_request_path)
    gpu = request['GPU_arm']
    continuation = pipeline.get('format') == CONTINUATION_PIPELINE_FORMAT
    if continuation and gpu != 4:
        raise ValueError('Exact saved-state continuation is explicitly GPU4 only')
    execution_code, execution_commit, _ = _source_execution(request)
    expected_code, expected_commit = ((execution_code, execution_commit) if continuation else
        (request['source_code'], request['source_commit']))
    if request.get('storage_profile') == READONLY_PROFILE and gpu == 4 and not continuation:
        raise ValueError('Read-only GPU4 chain requires its explicitly pinned exact saved-state source pipeline')
    scale_fields = ('epochs', 'physical_patient_batch', 'candidate_chunk') if gpu in (4, 5) else (
        'original_total_epochs', 'original_patient_batch', 'original_candidate_chunk')
    if (status.get('request') != pipeline or pipeline.get('GPU') != gpu
            or pipeline.get('GPU_uuid') != request['GPU_UUID'] or pipeline.get('code') != expected_code
            or pipeline.get('commit') != expected_commit or pipeline.get('production_output') != request['source_output']
            or pipeline.get('CPU_affinity') != request['CPU_affinity'] or pipeline.get('RAM_GiB') != 64
            or tuple(pipeline.get(name) for name in scale_fields) != (40, 4, 32 if gpu == 5 else 64)):
        raise ValueError('Registered GNN worker request does not match the exact full own-arm contract')
    if gpu == 4:
        _validate_gpu4_pipeline_inputs(request, pipeline)
    else:
        validate_source_metadata(request)
    trains = [stage for stage in pipeline.get('stages', []) if stage.get('name') == job['training_stage']]
    if len(trains) != 1 or pipeline['stages'][-1] != trains[0]:
        raise ValueError('One actual final train command required in original worker request')
    worker = process_witness(job['worker_pid'], job['worker_create_time'], job['worker_command'])
    child = None
    allowed_stages = ('prepare_continuation', 'train') if continuation else ('prepare_inputs', 'calibrate', 'train')
    if gpu == 4 and status.get('stage') not in allowed_stages:
        raise ValueError('GPU4 source is not in one of its exact declared stages')
    if status.get('child_pid') is not None:
        # A pipeline may still be in a preceding DEBUG admission stage at first
        # registration. Bind the exact live child to its actual declared stage.
        stages = [stage for stage in pipeline['stages'] if stage.get('name') == status.get('stage')]
        if len(stages) != 1:
            raise ValueError('Original live GNN child does not have one declared stage')
        child = process_witness(status['child_pid'], status['child_create_time'], stages[0]['command'])
        if child is not None and child['status'] != 'zombie':
            import psutil
            if psutil.Process(child['pid']).ppid() != job['worker_pid']:
                raise ValueError('Original GNN child is not parented by its exact owned worker')
    if gpu == 4:
        metadata = [Path(request['source_output']) / name for name in
                    ('request.json', 'calibration.json', 'training/training_identity.json')]
        missing = [path for path in metadata if not path.exists()]
        receipt_missing = continuation and not (Path(request['source_output']) / CONTINUATION_RECEIPT).exists()
        if continuation and not receipt_missing:
            _validate_gpu4_continuation_receipt(request, pipeline)
        if receipt_missing and status.get('stage') != 'prepare_continuation':
            raise FileNotFoundError('GPU4 continuation train requires its real prepared saved-state receipt')
        if missing or receipt_missing:
            # Existing malformed artifacts are real errors, even while another
            # required artifact has not been published. No synthetic identity
            # or calibration is substituted during fresh preparation.
            present = {}
            for path in metadata:
                if path.exists():
                    present[path] = read(path)
                    if not isinstance(present[path], dict):
                        raise ValueError('Actual GPU4 source metadata must be a JSON object: ' + str(path))
            scientific = present.get(metadata[0])
            if scientific is not None:
                declared = {key: value for key, value in scientific.items() if key != 'request_sha256'}
                checksum = hashlib.sha256(json.dumps(declared, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                if (scientific.get('config') != pipeline['gnn_config'] or scientific.get('physical_GPU') != 4
                        or scientific.get('STU_checkpoint_sha256') != request['stunet_checkpoint_sha256']
                        or scientific.get('request_sha256') != checksum):
                    raise ValueError('Published GPU4 scientific request differs from its immutable fresh full configuration')
            calibration = present.get(metadata[1])
            if calibration is not None and (scientific is None
                    or calibration.get('request_sha256') != scientific['request_sha256']
                    or calibration.get('physical_GPU') != 4
                    or calibration.get('selected_physical_patient_batch') != 4
                    or calibration.get('selected_physical_candidate_batch') != 64):
                raise ValueError('Published GPU4 calibration lacks its actual full B4/chunk64 source binding')
            owner = present.get(metadata[2])
            if owner is not None and (scientific is None or calibration is None
                    or owner.get('binding', {}).get('identity') != scientific
                    or owner.get('binding', {}).get('epochs') != 40 or owner.get('binding', {}).get('debug') is not False
                    or not _checksum(owner.get('identity_sha256'))):
                raise ValueError('Published GPU4 training ownership lacks its actual complete source binding')
            if status.get('status') == 'COMPLETE':
                missing_path = missing[0] if missing else Path(request['source_output']) / CONTINUATION_RECEIPT
                raise FileNotFoundError('Completed GPU4 source is missing required real metadata: ' + str(missing_path))
            source_completion(request, status, worker, child)
            if status.get('status') != 'RUNNING' or worker is None or worker['status'] == 'zombie':
                raise RuntimeError('GPU4 missing metadata may wait only for its exact live RUNNING source worker')
            return None
        validate_source_metadata(request)
    return source_completion(request, status, worker, child)


def validate_source_metadata(request):
    """Admit immutable real training metadata without loading a checkpoint."""
    source = Path(request['source_output'])
    scientific = read(source / 'request.json')
    declared = {key: value for key, value in scientific.items() if key != 'request_sha256'}
    checksum = hashlib.sha256(json.dumps(declared, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    owner = read(source / 'training/training_identity.json')
    calibration = read(source / 'calibration.json')
    binding = owner.get('binding', {})
    config = scientific.get('config', {})
    runtime = config.get('v24_runtime', {})
    if (scientific.get('format') != 'v24_full_native_GT_free_execution_request_v1'
            or scientific.get('request_sha256') != checksum or scientific.get('physical_GPU') != request['GPU_arm']
            or config.get('epochs') != 40 or runtime.get('debug') is not False or runtime.get('hidden_subset') is not False
            or runtime.get('workers') != 4 or runtime.get('torch_threads') != 1
            or binding.get('identity') != scientific or binding.get('epochs') != 40 or binding.get('debug') is not False
            or not _checksum(owner.get('identity_sha256'))
            or calibration.get('request_sha256') != checksum or calibration.get('physical_GPU') != request['GPU_arm']
            or calibration.get('selected_physical_patient_batch') != 4
            or calibration.get('selected_physical_candidate_batch') != (32 if request['GPU_arm'] == 5 else 64)
            or binding.get('config', {}).get('v24_runtime', {}).get('batch_calibration') != calibration):
        raise ValueError('Actual immutable checksum/identity/config/full40/B4/calibration metadata required')
    if request['GPU_arm'] == 4 and (config.get('encoder') != 'official_pretrained_STU_Net_S'
            or runtime.get('physical_GPU') != 4 or not _gpu4_full_u(runtime)
            or runtime.get('all_P_from_epoch_one') is not True or runtime.get('other_P_as_negative') is not False
            or not _gpu4_full_u(binding.get('config', {}).get('v24_runtime', {}))
            or scientific.get('STU_checkpoint_sha256') != request['stunet_checkpoint_sha256']
            or sha(request['stunet_checkpoint']) != request['stunet_checkpoint_sha256']):
        raise ValueError('GPU4 original STU asset and actual all-P/full128U from epoch one required')


def _run(command):
    return subprocess.check_output(command, text=True).strip()


def _seconds(text):
    fields = text.split(':')
    if len(fields) != 3 or any(not x.isdigit() for x in fields):
        raise ValueError('Explicit PBS walltime required: ' + text)
    hours, minutes, seconds = map(int, fields)
    return hours * 3600 + minutes * 60 + seconds


def scheduler_proof(request, *, stage_index=0):
    expected = request.get('scheduler')
    if expected is None:
        return None
    try:
        text = _run(['/opt/pbs/bin/qstat', '-f', expected['job_id']])
    except subprocess.CalledProcessError as error:
        raise AllocationUnavailable('The original owned PBS reservation is unavailable; no native launch') from error
    fields = {}
    for line in text.splitlines():
        if ' = ' in line:
            key, value = line.strip().split(' = ', 1)
            fields[key] = value
    walltime = _seconds(fields.get('Resource_List.walltime', ''))
    reported_remaining = walltime - _seconds(fields.get('resources_used.walltime', ''))
    started = time.mktime(time.strptime(fields.get('stime', ''), '%a %b %d %H:%M:%S %Y'))
    expires = started + walltime
    if abs(expires - expected['expires_at_unix']) > 1:
        raise AllocationUnavailable('Original owned PBS start/expiry changed; no native launch')
    # Scheduler resource counters can lag; the actual wall-clock expiry must
    # independently leave the required native training margin.
    remaining = min(reported_remaining, expires - time.time())
    floor = expected['minimum_remaining_seconds'] if stage_index < 2 else expected.get('native_minimum_remaining_seconds', 25200)
    if (fields.get('job_state') != 'R' or not fields.get('Job_Owner', '').startswith(expected['expected_owner'] + '@')
            or expected['expected_node'] not in fields.get('exec_host', '')
            or fields.get('Resource_List.ncpus') != '6' or fields.get('Resource_List.ngpus') != '1'
            or fields.get('Resource_List.mem', '').lower() != '128gb'
            or remaining < floor):
        raise AllocationUnavailable('Assigned owned PBS allocation or remaining walltime is insufficient; no native launch')
    return dict(job_id=expected['job_id'], remaining_seconds=remaining, required_remaining_seconds=floor,
                expires_at_unix=expires, reported_remaining_seconds=reported_remaining,
                ownership_and_allocation_checked=True)


def resources(request, *, initial=False, stage_index=0):
    storage = _readonly_storage(request)
    import psutil
    affinity = psutil.Process().cpu_affinity()
    if affinity != request['CPU_affinity']:
        raise ValueError('Watcher must inherit exactly its original four assigned CPU IDs')
    row = _run(['nvidia-smi', '--id=' + str(request['GPU_arm']),
        '--query-gpu=index,uuid,memory.total,memory.free,mig.mode.current', '--format=csv,noheader,nounits'])
    fields = [value.strip() for value in row.split(',')]
    if (len(fields) != 5 or fields[0] != str(request['GPU_arm']) or fields[1] != request['GPU_UUID']
            or float(fields[2]) < 40 * 1024 or float(fields[3]) < 40 * 1024
            or fields[4] not in ('N/A', 'Disabled', '[N/A]')):
        raise AssignedGPUUnavailable('The exact assigned full GPU is not free for the next native stage')
    apps = _run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'])
    if any(line.strip().split(',')[-1].strip() == request['GPU_UUID'] for line in apps.splitlines() if ',' in line):
        raise AssignedGPUUnavailable('Assigned GPU still has a compute application; no duplicate/downstream launch')
    free = shutil.disk_usage(request['chain_root']).free
    minimum = request['min_free_disk_GiB'] if initial else request.get('minimum_runtime_free_disk_GiB', 10)
    aggregate = None
    if request.get('storage_extension') == STATIC_EXTENSION:
        from hiercp_v1x.v24_readonly_static_operator_storage import check_aggregate_disk
        aggregate = check_aggregate_disk(storage, request['chain_root'], preparation=True)
        minimum = aggregate['required_free_bytes'] / 2**30
    elif free < minimum * 2**30:
        raise OSError('Full native data/checkpoint reserve does not fit; no subset or overwrite fallback')
    return dict(actual_CPU_affinity=affinity, GPU_UUID=request['GPU_UUID'], free_VRAM_MiB=float(fields[3]),
                free_disk_bytes=free, minimum_free_disk_GiB=minimum, aggregate_storage=aggregate,
                scheduler=scheduler_proof(request, stage_index=stage_index))


def stage_proof(request, action):
    root = Path(request['chain_root'])
    gpu = request['GPU_arm']
    if action == 'pin-current-gnn':
        path = root / 'pin.json'; value = read(path)
        if (value.get('format') != 'v24_completed_single_arm_GT_blind_best_v1'
                or value.get('physical_GPU') != gpu or value.get('completed_epochs') != 40
                or value.get('optimizer_updates') != 680 or value.get('full_training') is not True
                or value.get('debug') is not False or value.get('source_output') != request['source_output']
                or value.get('source_code') != request['source_code']
                or value.get('selection_scope') != 'this completed arm only; no cross-arm winner'
                or value.get('selected', {}).get('gpu') != gpu
                or value.get('selected', {}).get('best_record', {}).get('selected_by') != 'fixed_full128_validation_only'
                or not _checksum(value.get('selected', {}).get('best_file_sha256'))):
            raise ValueError('Native first stage did not publish the actual completed own-arm BEST pin')
        if gpu == 4 and (value.get('training_U_policy') != dict(all_U_from_epoch_one=True, initial_u=128, total_u=128, total_epochs=40)
                or value.get('model_contract', {}).get('encoder') != 'official_pretrained_STU_Net_S'
                or value.get('stunet_checkpoint') != request['stunet_checkpoint']):
            raise ValueError('GPU4 native pin must retain its original STU/full128U from epoch one contract')
    elif action == 'prepare-current-bank':
        path = root / 'bank/index.json'; value = read(path)
        if (value.get('pipeline_version') != CURRENT_FORMAT or value.get('physical_GPU') != gpu
                or value.get('complete') is not True or value.get('pin') != read(root / 'pin.json')):
            raise ValueError('Complete own-arm native bank proof required')
    elif action == 'prepare-native':
        path = root / 'native/native.json'; value = read(path)
        if (value.get('format') != CURRENT_FORMAT or value.get('physical_GPU') != gpu
                or value.get('bank') != str(root / 'bank/index.json')
                or value.get('baseline', {}).get('epochs') != 250
                or value.get('model_weights_fresh') is not True):
            raise ValueError('Fresh full250 own-arm native nnUNet proof required')
    elif action == 'calibrate-native':
        path = root / 'native/calibration.json'; value = read(path)
        if (value.get('debug') is not True or value.get('production_updates') != 0
                or value.get('production_epochs') != 250 or value.get('production_physical_batch') != 2
                or value.get('production_cp_probability') != .5):
            raise ValueError('Actual native clone calibration must preserve the full production contract')
    elif action == 'train':
        candidates = list((root / 'native').glob('training_complete_*.json'))
        if len(candidates) != 1:
            raise ValueError('Exactly one successful full native training receipt required')
        path = candidates[0]; value = read(path)
        if value.get('epochs') != 250 or not _checksum(value.get('checkpoint_sha256')):
            raise ValueError('Actual full250 native completion checkpoint proof required')
        checkpoint = owned(value['checkpoint']); _inside(checkpoint, root / 'native')
        if sha(checkpoint) != value['checkpoint_sha256']:
            raise ValueError('Actual final native checkpoint checksum differs')
    else:
        raise ValueError('Unknown chain stage')
    proof = dict(path=str(path), sha256=sha(path), action=action)
    if request.get('storage_profile') == READONLY_PROFILE:
        storage_document = _readonly_storage(request)
        receipt = root / ('readonly_storage_' + action.replace('-', '_') + '.json')
        declared = read(receipt)
        if (declared.get('format') != 'v24_readonly_native_stage_v1' or declared.get('action') != action
                or declared.get('storage_profile') != READONLY_PROFILE or declared.get('status') != 'COMPLETE'
                or declared.get('storage_admission') != request['storage_admission']
                or declared.get('storage_admission_sha256') != request['storage_admission_sha256']
                or declared.get('cached_inputs_written') is not False
                or declared.get('model_data_scale_preserved') is not True):
            raise ValueError('Explicit read-only stage/cache/admission completion proof differs')
        runtime_files = (READONLY_ENTRY, READONLY_HELPER, 'hiercp_v1x/v24_native_calibration_runtime.py',
                         'hiercp_v1x/v24_native_crop_runtime.py', 'hiercp_v1x/v24_native_gradient_runtime.py')
        if request.get('storage_extension') == STATIC_EXTENSION:
            runtime_files = (*runtime_files, STATIC_HELPER)
        if declared.get('storage_extension') != request.get('storage_extension'):
            raise ValueError('The actual completed stage static-operator extension differs')
        sources = declared.get('runtime_sources_sha256')
        if (not isinstance(sources, dict) or set(sources) != set(runtime_files)
                or any(request['code_files'].get(name) != checksum or sha(Path(request['code']) / name) != checksum
                       for name, checksum in sources.items())):
            raise ValueError('Actual read-only stage runtime source proof differs')
        if action in ('prepare-current-bank', 'prepare-native') and (
                value.get('storage_profile') != READONLY_PROFILE
                or value.get('storage_admission') != request['storage_admission']
                or value.get('storage_admission_sha256') != request['storage_admission_sha256']):
            raise ValueError('Bank/native metadata does not declare its actual read-only cache profile')
        if action in ('prepare-current-bank', 'prepare-native') and value.get('storage_extension') != request.get('storage_extension'):
            raise ValueError('Bank/native metadata does not declare its actual static-operator extension')
        if request.get('storage_extension') == STATIC_EXTENSION:
            coordination = storage_document['checkpoint_coordination']
            if declared.get('checkpoint_coordination') != coordination:
                raise ValueError('Actual stage must retain the admitted original global7 final-publication coordination')
            if action == 'prepare-native' and (value.get('checkpoint_coordination') != coordination
                    or value.get('checkpoint_publication_proof') != storage_document['checkpoint_publication_proof']):
                raise ValueError('Actual native metadata must seal its admitted original final-publication source and lock')
            if action == 'train':
                launches = declared.get('launches')
                if not isinstance(launches, list) or len(launches) != 1:
                    raise ValueError('One actual full native worker with original final-publication proof required')
                worker_receipt = launches[0].get('readonly_worker_receipt', {})
                worker_path = owned(worker_receipt.get('path')); _inside(worker_path, root / 'native')
                if sha(worker_path) != worker_receipt.get('sha256'):
                    raise ValueError('Actual final-publication worker completion proof changed')
                worker = read(worker_path)
                if (worker.get('format') != 'v24_readonly_native_worker_v1' or worker.get('status') != 'COMPLETE'
                        or worker.get('worker') != 'train' or worker.get('physical_batch') != 2 or worker.get('native_epochs') != 250
                        or worker.get('storage_admission_sha256') != request['storage_admission_sha256']
                        or worker.get('checkpoint_coordination') != coordination
                        or worker.get('native') != str(root / 'native/native.json')
                        or worker.get('native_sha256') != sha(root / 'native/native.json')):
                    raise ValueError('Actual full native training worker final-publication binding differs')
                from tools.run_v24_readonly_nnunet_cp import validate_checkpoint_coordination
                validate_checkpoint_coordination(worker.get('adapters', {}).get('checkpoint_coordination'), storage_document, completed=True,
                    expected_fold=coordination['folds']['gpu' + str(request['GPU_arm'])],
                    private_runtime=read(root / 'native/native.json')['private_runtime'])
        proof['storage_receipt'] = dict(path=str(receipt), sha256=sha(receipt))
    return proof


def publish(root, status):
    path = root / 'status.json'
    if path.exists():
        owned(path)
    temporary = root / ('status.' + str(os.getpid()) + '.' + str(time.time_ns()) + '.tmp')
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(status, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def claim(root, request_sha256):
    owned(root, directory=True, must_exist=False)
    if not root.exists():
        root.mkdir()
    lease = root / 'launch.lease.json'
    with lease.open('x', encoding='utf8') as stream:
        json.dump(dict(request_sha256=request_sha256, supervisor_pid=os.getpid(), created=time.time(),
                       replay_policy='Existing lease requires explicit recovery; completed stages are never replayed'), stream)
        stream.flush(); os.fsync(stream.fileno())


def _validate_recovery(request):
    """Only an owned GPU4 import failure before every native stage may recover."""
    recovery = request.get('recovery')
    if request.get('GPU_arm') != 4 or not isinstance(recovery, dict) or set(recovery) != {'proof', 'proof_sha256', 'lease'}:
        raise ValueError('Only explicit GPU4 prelaunch registration-failure recovery is supported')
    proof_path = owned(recovery['proof'])
    if not _checksum(recovery['proof_sha256']) or sha(proof_path) != recovery['proof_sha256']:
        raise ValueError('Immutable GPU4 recovery proof changed')
    proof = read(proof_path)
    previous_root = owned(proof['previous_chain_root'], directory=True)
    root = Path(request['chain_root'])
    lease_path = previous_root / 'recovery.lease.json'
    if (proof.get('format') != 'v24_GPU4_prelaunch_chain_recovery_v1'
            or proof.get('chain_root') != str(root) or root.parent != previous_root
            or not root.name.startswith('recovery_') or root.name == 'recovery_'
            or proof.get('source_output') != request['source_output']
            or proof_path.parent != previous_root or not proof_path.name.startswith('recovery_proof_')
            or proof_path.suffix != '.json' or recovery['lease'] != str(lease_path)
            or set(proof.get('previous_files_sha256', {})) != {'request.json', 'status.json', 'launch.lease.json'}
            or not _checksum(proof.get('source_claim_sha256'))):
        raise ValueError('GPU4 recovery must append one new root/proof/lease to its original failed chain')
    owned(lease_path, must_exist=False)
    for name, checksum in proof['previous_files_sha256'].items():
        if not _checksum(checksum) or sha(previous_root / name) != checksum:
            raise ValueError('Original failed GPU4 request/status/lease changed: ' + name)
    previous = read(previous_root / 'request.json')
    if previous.get('recovery') is not None or previous.get('chain_root') != str(previous_root):
        raise ValueError('Recovery requires the original direct GPU4 chain, never another recovery')
    validate_request(previous, fresh=False)
    same_source = ('GPU_arm', 'GPU_UUID', 'source_output', 'source_code', 'source_commit', 'source_files',
        'CPU_affinity', 'scoring_RAM_GiB', 'native_RAM_GiB', 'min_free_disk_GiB', 'minimum_runtime_free_disk_GiB',
        'python', 'inventory', 'baseline_preprocessed', 'input_cache', 'stunet_checkpoint',
        'stunet_checkpoint_sha256', 'scheduler', 'source_job', 'source_claim_path')
    if any(previous.get(key) != request.get(key) for key in same_source):
        raise ValueError('GPU4 recovery changed its original GNN/source/assets/assignment contract')
    previous_checksum = proof['previous_files_sha256']['request.json']
    status = read(previous_root / 'status.json')
    prior_lease = read(previous_root / 'launch.lease.json')
    pid, birth = proof.get('previous_supervisor_pid'), proof.get('previous_supervisor_create_time')
    if (type(pid) is not int or pid <= 0 or type(birth) not in (int, float)
            or status.get('status') != 'FAILED' or status.get('GPU_arm') != 4
            or status.get('error') != "ModuleNotFoundError: No module named 'tools'"
            or status.get('stage') is not None or status.get('child_pid') is not None
            or status.get('child_create_time') is not None or status.get('gnn_completion') is not None
            or status.get('stages_completed') != [] or status.get('stage_proofs') != []
            or status.get('signals_sent') is not False or status.get('request_sha256') != previous_checksum
            or status.get('source_job') != previous['source_job']
            or status.get('source_claim_path') != request['source_claim_path']
            or status.get('source_claim_sha256') != proof['source_claim_sha256']
            or status.get('supervisor_pid') != pid or status.get('supervisor_create_time') != birth
            or prior_lease.get('supervisor_pid') != pid or prior_lease.get('request_sha256') != previous_checksum):
        raise ValueError('Original GPU4 failure was not the proven tools import failure before native launch')
    witness = process_witness(pid, birth)
    if witness is not None and witness['status'] != 'zombie':
        raise RuntimeError('Prior GPU4 chain supervisor is still live; recovery cannot launch')
    allowed = {'request.json', 'status.json', 'launch.lease.json', 'watcher.log', 'registration.json',
               proof_path.name, lease_path.name, root.name}
    for path in previous_root.iterdir():
        if path.name not in allowed:
            raise ValueError('Original failed GPU4 chain contains stage/output files; no prelaunch recovery')
        owned(path, directory=path.name == root.name)
    return proof, previous_checksum


def _validate_storage_handoff(request):
    """Append a storage-only controller to a closed, never-launched own waiter."""
    handoff = request.get('storage_handoff')
    if (request.get('GPU_arm') not in (5, 6) or request.get('storage_profile') != READONLY_PROFILE
            or request.get('storage_extension') != STATIC_EXTENSION
            or request.get('recovery') is not None or not isinstance(handoff, dict)
            or set(handoff) != {'proof', 'proof_sha256', 'lease'}):
        raise ValueError('Only the explicit GPU5/6 static-storage controller supersession is supported')
    claim_path = owned(request['source_claim_path'])
    proof_path = owned(handoff['proof'])
    lease = claim_path.with_name(claim_path.stem + '.storage_handoff_lease.json')
    if (not _checksum(handoff['proof_sha256']) or sha(proof_path) != handoff['proof_sha256']
            or proof_path.parent != claim_path.parent
            or not proof_path.name.startswith(claim_path.stem + '.storage_handoff_proof_')
            or proof_path.suffix != '.json' or handoff['lease'] != str(lease)):
        raise ValueError('Storage handoff must append one SHA-pinned proof/exclusive lease to the original registry')
    owned(lease, must_exist=False)
    proof = read(proof_path)
    previous_root = owned(proof['previous_chain_root'], directory=True)
    root = Path(request['chain_root'])
    if (proof.get('format') != 'v24_native_storage_only_supersession_proof_v1'
            or proof.get('chain_root') != str(root) or proof.get('source_output') != request['source_output']
            or previous_root == root or previous_root in root.parents or root in previous_root.parents
            or set(proof.get('previous_files_sha256', {})) !=
                {'request.json', 'status.json', 'launch.lease.json', 'registration.json'}
            or not _checksum(proof.get('source_claim_sha256'))
            or proof.get('old_waiter_closed') is not True or proof.get('old_waiter_children') != []
            or proof.get('old_waiter_cuda_visible_devices') != ''
            or proof.get('original_GNN_processes_signaled') is not False
            or proof.get('previous_results_written') is not False):
        raise ValueError('Immutable prelaunch CPU-only storage handoff proof required')
    for name, checksum in proof['previous_files_sha256'].items():
        if not _checksum(checksum) or sha(previous_root / name) != checksum:
            raise ValueError('Original GPU5/6 controller artifact changed: ' + name)
    previous = read(previous_root / 'request.json')
    if (previous.get('chain_root') != str(previous_root) or previous.get('storage_profile') is not None
            or previous.get('storage_handoff') is not None or previous.get('recovery') is not None):
        raise ValueError('Storage handoff requires the original full-copy waiter, never another handoff')
    validate_request(previous, fresh=False)
    same = ('GPU_arm', 'GPU_UUID', 'source_output', 'source_code', 'source_commit', 'source_files', 'source_job',
            'CPU_affinity', 'scoring_RAM_GiB', 'native_RAM_GiB', 'python', 'inventory', 'baseline_preprocessed',
            'input_cache', 'stunet_checkpoint', 'stunet_checkpoint_sha256', 'scheduler', 'source_claim_path')
    if any(previous.get(name) != request.get(name) for name in same):
        raise ValueError('Storage supersession changed original GNN/assets/assignment/PBS contract')
    checksum = proof['previous_files_sha256']['request.json']
    status, registration, old_lease = (read(previous_root / name) for name in
                                       ('status.json', 'registration.json', 'launch.lease.json'))
    pid, birth, command = (proof.get('previous_supervisor_' + key) for key in ('pid', 'create_time', 'command'))
    expected_command = [previous['python'], '-B', '-u', str(Path(previous['code']) / 'tools/watch_v24_gnn_to_nnunet.py'),
                        '--request', str(previous_root / 'request.json')]
    if (type(pid) is not int or pid <= 0 or type(birth) not in (int, float) or command != expected_command
            or registration.get('GPU_arm') != request['GPU_arm'] or registration.get('root') != str(previous_root)
            or registration.get('pid') != pid or registration.get('create_time') != birth
            or registration.get('command') != command or registration.get('request_sha256') != checksum
            or registration.get('CPU_affinity') != request['CPU_affinity']
            or registration.get('CUDA_visible_devices_while_waiting') != ''
            or status.get('GPU_arm') != request['GPU_arm'] or status.get('status') != 'WAITING_FOR_FULL_GNN'
            or status.get('stage') is not None or status.get('child_pid') is not None
            or status.get('child_create_time') is not None or status.get('gnn_completion') is not None
            or status.get('stages_completed') != [] or status.get('stage_proofs') != []
            or status.get('signals_sent') is not False or status.get('request_sha256') != checksum
            or status.get('source_job') != request['source_job'] or status.get('supervisor_pid') != pid
            or status.get('supervisor_create_time') != birth
            or status.get('actual_CPU_affinity') != request['CPU_affinity']
            or status.get('source_claim_path') != request['source_claim_path']
            or status.get('source_claim_sha256') != proof['source_claim_sha256']
            or old_lease.get('supervisor_pid') != pid or old_lease.get('request_sha256') != checksum):
        raise ValueError('Original controller is not its registered CPU-only never-launched waiter')
    witness = process_witness(pid, birth, command)
    if witness is not None and witness['status'] != 'zombie':
        raise RuntimeError('Original own CPU waiter remains live; no storage controller may launch')
    allowed = {'request.json', 'status.json', 'launch.lease.json', 'registration.json', 'watcher.log'}
    for path in previous_root.iterdir():
        if path.name not in allowed:
            raise ValueError('Original waiter contains native stage/output artifacts; storage-only handoff refused')
        owned(path)
    return proof, checksum


def verify_source_claim(request, request_checksum):
    path = request['source_claim_path']
    if request.get('storage_handoff') is not None:
        proof, previous_checksum = _validate_storage_handoff(request)
        handoff = request['storage_handoff']
        expected = dict(format='v24_owned_gnn_to_native_source_claim_v1', source_output=request['source_output'],
            GPU_arm=request['GPU_arm'], chain_root=proof['previous_chain_root'], request_sha256=previous_checksum)
        if read(path) != expected or sha(path) != proof['source_claim_sha256']:
            raise ValueError('Original exclusive source claim changed; storage handoff cannot replace it')
        expected_lease = dict(format='v24_native_storage_only_supersession_lease_v1', GPU_arm=request['GPU_arm'],
            source_output=request['source_output'], source_claim_path=path, source_claim_sha256=proof['source_claim_sha256'],
            previous_chain_root=proof['previous_chain_root'], chain_root=request['chain_root'], request_sha256=request_checksum,
            proof=handoff['proof'], proof_sha256=handoff['proof_sha256'])
        if read(handoff['lease']) != expected_lease:
            raise ValueError('Exclusive storage handoff lease belongs to another request/root; no replay')
        return sha(path)
    if request.get('recovery') is not None:
        proof, previous_checksum = _validate_recovery(request)
        recovery = request['recovery']
        expected = dict(format='v24_owned_gnn_to_native_source_claim_v1', source_output=request['source_output'],
            GPU_arm=4, chain_root=proof['previous_chain_root'], request_sha256=previous_checksum)
        if read(path) != expected or sha(path) != proof['source_claim_sha256']:
            raise ValueError('Original exclusive GPU4 source claim changed; recovery cannot replace it')
        expected_lease = dict(format='v24_GPU4_prelaunch_chain_recovery_lease_v1', GPU_arm=4,
            source_output=request['source_output'], source_claim_path=path,
            source_claim_sha256=proof['source_claim_sha256'], previous_chain_root=proof['previous_chain_root'],
            chain_root=request['chain_root'], request_sha256=request_checksum,
            proof=recovery['proof'], proof_sha256=recovery['proof_sha256'])
        if read(recovery['lease']) != expected_lease:
            raise ValueError('Exclusive GPU4 recovery lease belongs to another request/root; no replay')
        return sha(path)
    expected = dict(format='v24_owned_gnn_to_native_source_claim_v1', source_output=request['source_output'],
                    GPU_arm=request['GPU_arm'], chain_root=request['chain_root'], request_sha256=request_checksum)
    if read(path) != expected:
        raise ValueError('This completed GNN source is claimed by another native chain; no duplicate launch')
    return sha(path)


def _tree_rss(process):
    import psutil
    total = 0
    try:
        children = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return total
    for child in children:
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total


def run_stage(request, stage, status, request_path, request_checksum):
    """Direct file output avoids a blocked stdout reader and preserves every log."""
    import psutil
    root = Path(request['chain_root'])
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               V24_ARM_RSS_GIB=str(request['scoring_RAM_GiB'] if stage['action'] == 'prepare-current-bank' else request['native_RAM_GiB']))
    env['CUDA_VISIBLE_DEVICES'] = str(request['GPU_arm'])
    if request.get('storage_profile') == READONLY_PROFILE:
        _readonly_storage(request)
        env['V24_READONLY_STORAGE_ADMISSION_SHA256'] = request['storage_admission_sha256']
    status.update(status='RUNNING', stage=stage['name'], stage_started=time.time(), command=stage['command'])
    publish(root, status)
    print(json.dumps(dict(stage=stage['name'], command=stage['command'])), flush=True)
    log_path = root / (stage['name'] + '.log')
    budget = (request['scoring_RAM_GiB'] if stage['action'] in ('pin-current-gnn', 'prepare-current-bank') else request['native_RAM_GiB']) * 2**30
    violation = None
    with log_path.open('xb', buffering=0) as log:
        process = subprocess.Popen(stage['command'], cwd=request['code'], env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        own = psutil.Process(process.pid)
        status.update(child_pid=process.pid, child_create_time=own.create_time(), stage_log=str(log_path))
        publish(root, status)
        while process.poll() is None:
            rss = _tree_rss(own)
            if rss > budget:
                violation = 'Owned native child process tree exceeded its fixed RSS budget; no downstream stage will run'
            if sha(request_path) != request_checksum:
                violation = 'Registered immutable request changed while its owned child was running'
            status.update(child_process_tree_RSS_bytes=rss, RAM_budget_bytes=budget, resource_violation=violation,
                          checked_at=time.time())
            publish(root, status)
            time.sleep(POLL_SECONDS)
        result = process.wait()
    if result or violation:
        raise RuntimeError('Stage ' + stage['name'] + ' failed (' + str(result) + '): ' + str(violation)
                           + '; logs preserved; no process was signaled and no downstream stage was launched')
    proof = stage_proof(request, stage['action'])
    status['stages_completed'].append(stage['name'])
    status['stage_proofs'].append(proof)
    status.update(child_pid=None, child_create_time=None)
    publish(root, status)


def run(request_path):
    import psutil
    request_path = owned(request_path)
    request_checksum = sha(request_path)
    request = validate_request(read(request_path))
    verify_code(request)
    source_claim_checksum = verify_source_claim(request, request_checksum)
    root = Path(request['chain_root'])
    claim(root, request_checksum)
    status = dict(format='v24_completed_gnn_to_native_chain_status_v1', request_sha256=request_checksum,
                  GPU_arm=request['GPU_arm'], source_job=request['source_job'], source_output=request['source_output'],
                  source_code=request['source_code'], source_commit=request['source_commit'], code=request['code'], commit=request['commit'],
                  supervisor_pid=os.getpid(), supervisor_create_time=psutil.Process().create_time(),
                  actual_CPU_affinity=psutil.Process().cpu_affinity(), status='WAITING_FOR_FULL_GNN', stage=None,
                  stages_completed=[], stage_proofs=[], started=time.time(), signals_sent=False,
                  source_claim_path=request['source_claim_path'], source_claim_sha256=source_claim_checksum)
    publish(root, status)
    try:
        if status['actual_CPU_affinity'] != request['CPU_affinity']:
            raise ValueError('Watcher did not inherit its exact assigned CPU affinity')
        while True:
            if sha(request_path) != request_checksum or verify_source_claim(request, request_checksum) != source_claim_checksum:
                raise ValueError('Registered request changed while waiting')
            completion = inspect_source(request)
            if completion is not None:
                status.update(gnn_completion=completion, status='GNN_COMPLETE', gnn_completed_detected=time.time())
                publish(root, status)
                break
            status.update(checked_at=time.time())
            publish(root, status)
            time.sleep(POLL_SECONDS)
        for index, stage in enumerate(request['stages']):
            verify_code(request)
            if sha(request_path) != request_checksum or verify_source_claim(request, request_checksum) != source_claim_checksum:
                raise ValueError('Registered request changed before a native stage')
            if inspect_source(request) != completion:
                raise ValueError('Completed GNN source/receipt changed before a native stage')
            while True:
                try:
                    status['resources'] = resources(request, initial=index == 0, stage_index=index)
                    break
                except (AllocationUnavailable, AssignedGPUUnavailable) as error:
                    status.update(status='WAITING_ALLOCATION' if isinstance(error, AllocationUnavailable)
                                  else 'WAITING_FOR_ASSIGNED_GPU', waiting_reason=str(error), checked_at=time.time())
                    publish(root, status)
                    time.sleep(POLL_SECONDS)
                    if (sha(request_path) != request_checksum or verify_source_claim(request, request_checksum) != source_claim_checksum
                            or inspect_source(request) != completion):
                        raise ValueError('Request or completed GNN changed while waiting for its owned allocation')
            run_stage(request, stage, status, request_path, request_checksum)
        status.update(status='COMPLETE', completed=time.time(), full_native_training_completed=True)
        publish(root, status)
    except Exception as error:
        status.update(status='FAILED', error=type(error).__name__ + ': ' + str(error), failed=time.time(),
                      recovery='Keep this lease and every log/result. Inspect the exact failed stage; an explicit new recovery request is required.')
        publish(root, status)
        raise


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Read-only request/code admission; never starts a child or creates a lease')
    return parser.parse_args(argv)


def main(args):
    if args.check:
        request = validate_request(read(args.request))
        verify_code(request)
        if request['GPU_arm'] == 4:
            _validate_gpu4_pipeline_inputs(request, read(request['source_job']['worker_command'][-1]))
        print(json.dumps(dict(status='REQUEST_ADMITTED', GPU_arm=request['GPU_arm'], server_files_written=False,
                              GPU_child_started=False, GNN_checkpoint_loaded=False)), flush=True)
    else:
        run(args.request)


if __name__ == '__main__':
    main(parse())
