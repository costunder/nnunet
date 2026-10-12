"""Explicit GPU6 zero-update startup retry; original science/output stay sealed.

The failed original trainer never entered on_train_start: its only fold file is
the citation header. This adapter does not relax the original fresh-fold gate.
It uses that exact frozen runtime with one process-local resource-observer name
binding, and publishes completion only after original full250/BEST26 proofs.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

FORMAT = 'v24_zero_update_native_startup_retry_request_v1'
STATUS_FORMAT = 'v24_zero_update_native_startup_retry_status_v1'
REGISTRATION_FORMAT = 'v24_zero_update_native_startup_retry_registration_v1'
FILES = ('hiercp_v1x/v24_native_startup_retry.py', 'tools/run_v24_native_startup_retry.py')
UID = 41833
AFFINITY = [32, 33, 34, 35]
RAM_BYTES = 48 * 2**30
FAILED_ERROR = 'RuntimeError: Native stage train failed (1): None; original logs preserved; no process signaled'
IMPORT_ERROR = "ImportError: cannot import name 'snapshot' from 'hiercp_v1x.v24_nnunet_cp'"
NATIVE_ARGV = ['730', '3d_fullres', '0', '-tr', 'nnUNetTrainer_250epochs_FrozenV23CP',
               '-p', 'nnUNetResEncUNetMPlans', '--val_best']
BINDING = 'original.v24_nnunet_cp.snapshot = original.v24_nnunet_cp.require_project_budget'
COMPLETION_FORMAT = 'v24_post_training_existing_best26_completion_request_v1'
COMPLETION_STATUS = 'v24_post_training_existing_best26_completion_status_v1'
POST_PROOF_ERROR = 'ValueError: Scientific module was already imported from a different checkout: hiercp_v1x._readonly_admitted_base_'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            value.update(block)
    return value.hexdigest()


def checksum(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set('0123456789abcdef')


def absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('Absolute non-traversing path required: ' + repr(value))
    return Path(value)


def regular(path, *, directory=False):
    path = Path(path)
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('Symlink is not an admitted recovery path: ' + str(parent))
    info = path.stat()
    if (hasattr(os, 'getuid') and info.st_uid != UID
            or directory and not stat.S_ISDIR(info.st_mode)
            or not directory and not stat.S_ISREG(info.st_mode)):
        raise ValueError('Exact owned regular recovery input required: ' + str(path))
    return path


def identity(path):
    info = Path(path).stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def proof(path):
    path = regular(path); before = identity(path); value = digest(path)
    if identity(path) != before:
        raise ValueError('Recovery input changed while hashing: ' + str(path))
    return dict(path=str(path), sha256=value, stat=before)


def verify_proof(value):
    if (not isinstance(value, dict) or set(value) != {'path', 'sha256', 'stat'}
            or not checksum(value['sha256']) or not isinstance(value['stat'], list) or len(value['stat']) != 5):
        raise ValueError('Exact path/raw SHA/full stat recovery proof required')
    path = regular(absolute(value['path']))
    if identity(path) != value['stat'] or digest(path) != value['sha256'] or identity(path) != value['stat']:
        raise ValueError('Sealed recovery input changed: ' + str(path))
    return path


def read(path):
    value = json.loads(regular(path).read_text(encoding='utf8'))
    if not isinstance(value, dict):
        raise ValueError('Recovery metadata object required: ' + str(path))
    return value


def publish(path, value, *, replace=False):
    path = Path(path)
    temporary = path.parent / ('.' + path.name + '.' + str(os.getpid()) + '.' + str(time.time_ns()) + '.tmp')
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    try:
        if replace:
            if path.exists():
                regular(path)
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_original(request, *, admitted_modules=None):
    """Use old package files, including deferred relative imports, without editing."""
    code = regular(request['original_declaration']['code'], directory=True)
    package = importlib.import_module('hiercp_v1x')
    # Storage admission may deliberately load its SHA-bound historical helper
    # under _readonly_admitted_base_<SHA>.  A later proof pass must reuse the
    # exact modules admitted before that deferred import, rather than replay
    # the first-import gate over those legitimate transitive dependencies.
    if admitted_modules is None:
        for name, module in list(sys.modules.items()):
            if name.startswith('hiercp_v1x.') and name != __name__ and getattr(module, '__file__', None):
                if not Path(module.__file__).absolute().is_relative_to(code):
                    raise ValueError('Scientific module was already imported from a different checkout: ' + name)
    elif (not isinstance(admitted_modules, tuple) or len(admitted_modules) != 5
            or list(package.__path__) != [str(code / 'hiercp_v1x')]):
        raise ValueError('Exact original admitted module tuple/package path required')
    package.__path__ = [str(code / 'hiercp_v1x')]
    if 'tools' in sys.modules:
        sys.modules['tools'].__path__ = [str(code / 'tools')]
    sys.path.insert(0, str(code))
    names = (
        'v24_scored_native_chain', 'v24_matched_basic_cp_pipeline', 'v24_matched_basic_cp_runtime',
        'v24_nnunet_cp', 'v24_native_best_eval_runtime')
    modules = (tuple(importlib.import_module('hiercp_v1x.' + name) for name in names)
        if admitted_modules is None else admitted_modules)
    for name, module in zip(names, modules):
        if (not Path(module.__file__).absolute().is_relative_to(code)
                or hasattr(module, 'ROOT') and Path(module.ROOT).absolute() != code
                or sys.modules.get('hiercp_v1x.' + name) is not module
                or module.__name__ != 'hiercp_v1x.' + name):
            raise ValueError('Frozen original module path differs: ' + module.__name__)
    return modules


def verify_new_code(request):
    code = regular(request['code'], directory=True)
    if subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() != request['commit']:
        raise ValueError('Recovery checkout Git commit changed')
    for name, value in request['code_files'].items():
        path = regular(code / name)
        if digest(path) != value:
            raise ValueError('Recovery checkout source SHA changed: ' + name)


def validate_declaration(request):
    old = request.get('original_declaration', {})
    if (request.get('format') != FORMAT or request.get('GPU_arm') != 6
            or request.get('CPU_affinity') != AFFINITY or request.get('native_RAM_GiB') != 48
            or request.get('binding') != BINDING or request.get('zero_production_updates_before_retry') is not True
            or request.get('rescore') is not False or request.get('resume') is not False
            or request.get('epochs') != 250 or request.get('physical_batch') != 2
            or request.get('evaluation_checkpoint') != 'checkpoint_best.pth'
            or old.get('format') != 'v24_scored_bank_to_native_chain_request_v1' or old.get('GPU_arm') != 6):
        raise ValueError('Explicit GPU6 zero-update frozen full250/B2/BEST26 retry required')
    root = absolute(request.get('chain_root')); old_root = absolute(old.get('chain_root'))
    if root.parent != old_root or root.name in ('native', 'bank'):
        raise ValueError('Recovery supervisor needs a fresh direct subdirectory of original GPU6 arm')
    for key in ('code', 'native_path', 'fold'):
        absolute(request.get(key))
    if request['native_path'] != str(old_root / 'native/native.json'):
        raise ValueError('Recovery must retain the exact admitted native output')
    commit = request.get('commit'); manifest = request.get('code_files')
    if (not isinstance(commit, str) or len(commit) != 40 or set(commit) - set('0123456789abcdef')
            or not isinstance(manifest, dict) or not set(FILES) <= set(manifest)):
        raise ValueError('Fresh recovery Git commit and complete source manifest required')
    for name, value in manifest.items():
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts or not checksum(value):
            raise ValueError('Pinned relative recovery source SHA required')
    required = {'request', 'status', 'registration', 'lease', 'admitted_scoring', 'parent_log',
        'native', 'calibration', 'native_log', 'training_started', 'fold_log', 'prepare_receipt',
        'calibrate_receipt', 'storage_binding', 'storage_admission', 'source_claim'}
    if not isinstance(request.get('proofs'), dict) or set(request['proofs']) != required:
        raise ValueError('All exact original failure/native/storage/claim witnesses required')
    if (request.get('original_request') != request['proofs']['request']
            or request.get('recovery_root') != str(root) or request.get('python') != old.get('python')
            or request.get('old_code') != old.get('code')):
        raise ValueError('Recovery viewer original proof/root/Python/code binding differs')
    if request.get('command') != child_command(request):
        raise ValueError('Exact fresh entry command and unchanged native CLI required')
    return request


def child_command(request):
    return [request['original_declaration']['python'], '-B', '-u',
        str(Path(request['code']) / 'tools/run_v24_native_startup_retry.py'), 'entry',
        '--request', str(Path(request['chain_root']) / 'request.json'), '--', *NATIVE_ARGV]


def _zero_fold(request):
    old_root = Path(request['original_declaration']['chain_root']); native_root = old_root / 'native'
    fold = regular(request['fold'], directory=True)
    files = [path for path in (native_root / 'nnUNet_results').rglob('*') if path.is_file() or path.is_symlink()]
    expected = verify_proof(request['proofs']['fold_log'])
    if files != [expected] or expected.parent != fold or not expected.name.startswith('training_log_'):
        raise ValueError('Zero-update retry requires only the sealed failed citation log in all native results')
    text = expected.read_text(encoding='utf8')
    if (expected.stat().st_size != 404 or 'Please cite the following paper when using nnU-Net:' not in text
            or '2026-' in text or 'Epoch' in text or 'train_loss' in text):
        raise ValueError('Failed fold log is not the original citation-only zero-update header')
    for pattern in ('actual_nnunet_configuration.json', 'training_complete_*.json'):
        if list(native_root.glob(pattern)):
            raise ValueError('Native configuration/completion already exists; zero-update replay refused')
    if (old_root / 'matched_storage_train.json').exists():
        raise ValueError('Completed native train receipt already exists; no replay')


def verify_failure(request, chain, *, zero=True):
    old = request['original_declaration']; root = Path(old['chain_root']); values = {}
    expected = dict(request=root/'request.json', status=root/'status.json', registration=root/'registration.json',
        lease=root/'lease.json', admitted_scoring=root/'admitted_scoring.json', parent_log=root/'train.log',
        native=root/'native/native.json', calibration=root/'native/calibration.json',
        prepare_receipt=root/'matched_storage_prepare_native.json', calibrate_receipt=root/'matched_storage_calibrate_native.json',
        storage_binding=Path(old['storage_binding']), storage_admission=Path(old['storage_admission']),
        source_claim=Path(old['native_claim_path']))
    for key, witness in request['proofs'].items():
        path = verify_proof(witness)
        if key in expected and path != expected[key]:
            raise ValueError('Original recovery witness path changed: ' + key)
        if key in ('request', 'status', 'registration', 'lease', 'native', 'training_started'):
            values[key] = read(path)
    if values['request'] != old or request['proofs']['request']['path'] != str(root / 'request.json'):
        raise ValueError('Original sealed controller request changed')
    state, registration, lease = values['status'], values['registration'], values['lease']
    expected_actor = [old['python'], '-B', '-u', str(Path(old['code']) / 'tools/watch_v24_scored_bank_to_nnunet.py'),
                      '--request', str(root / 'request.json')]
    if (state.get('status') != 'FAILED' or state.get('stage') != 'train' or state.get('error') != FAILED_ERROR
            or state.get('GPU_arm') != 6 or state.get('UID') != UID or state.get('resource_violation') is not None
            or state.get('stages_completed') != ['prepare_native', 'calibrate_native']
            or state.get('full_native_training_completed') is not False or state.get('full_native_evaluation_completed') is not False
            or state.get('request_sha256') != request['proofs']['request']['sha256']
            or state.get('command') != old['stages'][2]['command'] or state.get('stage_log') != str(root / 'train.log')
            or registration.get('format') != chain.REGISTRATION_FORMAT or registration.get('root') != str(root)
            or registration.get('GPU_arm') != 6 or registration.get('command') != expected_actor
            or registration.get('request_sha256') != state['request_sha256']
            or registration.get('pid') != state.get('supervisor_pid') or registration.get('create_time') != state.get('supervisor_create_time')
            or registration.get('CPU_affinity') != AFFINITY or state.get('actual_CPU_affinity') != AFFINITY
            or lease.get('supervisor_pid') != registration['pid'] or lease.get('supervisor_create_time') != registration['create_time']
            or lease.get('command') != expected_actor or lease.get('request_sha256') != state['request_sha256']
            or lease.get('claim') != old['native_claim_path'] or lease.get('claim_sha256') != request['proofs']['source_claim']['sha256']):
        raise ValueError('Exact original registered failed train supervisor/child required')
    for pid, birth, command in ((registration['pid'], registration['create_time'], expected_actor),
            (state.get('child_pid'), state.get('child_create_time'), state['command'])):
        if chain.process_snapshot(pid, birth, command) is not None:
            raise RuntimeError('Exact original native actor is still live; retry refused')
    log = Path(request['proofs']['native_log']['path']).read_text(encoding='utf8')
    if ('Traceback (most recent call last):' not in log
            or 'from .v24_nnunet_cp import require_project_budget, snapshot' not in log
            or IMPORT_ERROR not in log.splitlines()[-1]
            or str(Path(old['code']) / 'hiercp_v1x/v24_nnunet_cp.py') not in log.splitlines()[-1]):
        raise ValueError('Only preserved exact missing-snapshot pre-start ImportError may be retried')
    if values['native'].get('checkpoint_coordination', {}).get('folds', {}).get('gpu6') != request['fold']:
        raise ValueError('Original globally coordinated native fold must be preserved')
    started = values['training_started']
    start_path = Path(request['proofs']['training_started']['path']); native_log = Path(request['proofs']['native_log']['path'])
    if (start_path.parent != root/'native' or not start_path.name.startswith('training_started_')
            or native_log != start_path.with_name(start_path.name.replace('training_started_', 'train_')).with_suffix('.log')):
        raise ValueError('Exact paired original training-start record/log paths required')
    if (started.get('command', [])[4:] != ['from hiercp_v1x.v24_matched_basic_cp_runtime import run_training_entry; run_training_entry()', *NATIVE_ARGV]
            or started.get('epochs') != 250 or started.get('physical_batch') != 2
            or started.get('debug') is not False or started.get('bank_sha256') != values['native'].get('bank_sha256')):
        raise ValueError('Original fresh full native invocation changed')
    if zero:
        _zero_fold(request)
    return state


def build_request(original_request_path, recovery_root, code, commit, code_files, *, inspection=None):
    """Read-only actual-host assembly; no leases, paths or jobs are created."""
    old_path = regular(original_request_path); old = read(old_path); root = old_path.parent
    native = root / 'native'; metadata = read(native / 'native.json')
    starts = list(native.glob('training_started_*.json')); logs = list(native.glob('train_*.log'))
    fold = Path(metadata['checkpoint_coordination']['folds']['gpu6'])
    fold_logs = list(fold.glob('training_log_*.txt'))
    if len(starts) != 1 or len(logs) != 1 or len(fold_logs) != 1:
        raise ValueError('Exactly one failed zero-update native invocation/header required')
    paths = dict(request=old_path, status=root/'status.json', registration=root/'registration.json',
        lease=root/'lease.json', admitted_scoring=root/'admitted_scoring.json', parent_log=root/'train.log',
        native=native/'native.json', calibration=native/'calibration.json', native_log=logs[0],
        training_started=starts[0], fold_log=fold_logs[0], prepare_receipt=root/'matched_storage_prepare_native.json',
        calibrate_receipt=root/'matched_storage_calibrate_native.json', storage_binding=Path(old['storage_binding']),
        storage_admission=Path(old['storage_admission']), source_claim=Path(old['native_claim_path']))
    request = dict(format=FORMAT, GPU_arm=6, CPU_affinity=AFFINITY, native_RAM_GiB=48,
        chain_root=str(absolute(str(recovery_root))), native_path=str(native/'native.json'), fold=str(fold),
        code=str(absolute(str(code))), commit=commit, code_files=copy.deepcopy(code_files), original_declaration=old,
        original_request=proof(old_path), recovery_root=str(absolute(str(recovery_root))),
        python=old['python'], old_code=old['code'],
        proofs={key:proof(path) for key,path in paths.items()}, binding=BINDING,
        zero_production_updates_before_retry=True, rescore=False, resume=False, epochs=250, physical_batch=2,
        evaluation_checkpoint='checkpoint_best.pth')
    request['command'] = child_command(request)
    if inspection is not None:
        relative = fold_logs[0].relative_to(native).as_posix()
        files = inspection.get('files', [])
        if (inspection.get('native') != metadata or len(files) != 1
                or files[0].get('path') != relative or files[0].get('size') != 404
                or files[0].get('sha256') != request['proofs']['fold_log']['sha256']):
            raise ValueError('Captured actual zero-update native inspection differs from current files')
    return validate_declaration(request)


def seal_request(request):
    validate_declaration(request)
    root = absolute(request['chain_root'])
    if root.exists():
        regular(root, directory=True)
        if list(root.iterdir()):
            raise FileExistsError('Fresh empty recovery root required; no automatic lease replay')
    else:
        regular(root.parent, directory=True); root.mkdir(mode=0o700)
    publish(root / 'request.json', request)
    return root / 'request.json'


def admit(request, *, zero=True, resources=False, admitted_modules=None):
    validate_declaration(request); verify_new_code(request)
    chain, pipeline, runtime, original, best = _load_original(request, admitted_modules=admitted_modules)
    cache = chain.FileProofCache(); old = request['original_declaration']
    chain.validate_declaration(old); chain.verify_code(old, cache); chain.preserve_predecessors(old, cache)
    verify_failure(request, chain, zero=zero)
    completion = chain.source_complete(old)
    scoring = chain.scoring_complete(old, cache)
    if completion is None or scoring is None:
        raise RuntimeError('Own full40/680 source/full642 scoring or original actor release is missing')
    if read(request['proofs']['admitted_scoring']['path']) != scoring:
        raise ValueError('Previously admitted full642 scoring proof changed')
    pipeline._guard_native(request['native_path']); pipeline.validate_calibration(request['native_path'])
    for action in ('prepare-native', 'calibrate-native'):
        pipeline.stage_proof(old, action)
    storage = chain.storage_ready(old, cache)
    resource = chain.resources_ready(old, 'train') if resources else None
    return (chain, pipeline, runtime, original, best), storage, resource


def bind_snapshot(original):
    if hasattr(original, 'snapshot'):
        raise ValueError('Original missing name is already bound; unknown runtime repair refused')
    original.snapshot = original.require_project_budget
    return original.snapshot


def entry(request_path, native_argv):
    if list(native_argv) != NATIVE_ARGV:
        raise ValueError('Only original full250/B2/fold0/BEST CLI may execute')
    request = read(request_path)
    modules, _, _ = admit(request, zero=True, resources=True)
    chain, pipeline, runtime, original, best = modules
    lease = read(Path(request['chain_root'])/'lease.json')
    if lease.get('request_sha256') != digest(request_path):
        raise ValueError('Entry requires this exclusive supervisor lease')
    if chain.process_snapshot(lease['supervisor_pid'], lease['supervisor_create_time'], lease['command']) is None:
        raise RuntimeError('Exact recovery supervisor is no longer alive')
    import psutil
    actor = psutil.Process()
    if actor.cmdline() != request['command']:
        raise ValueError('Exact sealed retry entry process command required')
    publish(Path(request['chain_root'])/'entry.lease.json', dict(request_sha256=digest(request_path),
        pid=actor.pid, create_time=actor.create_time(), command=actor.cmdline(), UID=UID))
    bind_snapshot(original)
    sys.argv = [sys.argv[0], *NATIVE_ARGV]
    return runtime.run_training_entry()


def publish_completion(request, pipeline, best):
    native = read(request['native_path']); root = Path(native['root']); fold = Path(request['fold'])
    configuration = read(root/'actual_nnunet_configuration.json')
    if (configuration.get('epochs') != 250 or configuration.get('physical_batch') != 2
            or configuration.get('train_cases') != 105 or configuration.get('validation_cases') != 26
            or configuration.get('patch_size') != [128]*3 or configuration.get('CPU_workers') != 4
            or configuration.get('source_entries') != 642 or configuration.get('candidate_count') != 128
            or configuration.get('validation_checkpoint') != 'checkpoint_best.pth' or configuration.get('debug') is not False):
        raise ValueError('Actual original full250/B2/105+26/whole642 native configuration proof required')
    validated = best.validate_best_validation_receipt(fold/best.RECEIPT, fold=fold,
        expected_cases=native['split']['outer_val'], bank_sha256=native['bank_sha256'])
    final = regular(fold/'checkpoint_final.pth')
    if list(root.glob('training_complete_*.json')):
        raise FileExistsError('Native completion already exists; no replay')
    path = root/('training_complete_'+str(time.time_ns())+'.json')
    publish(path, dict(format=pipeline.NATIVE_FORMAT, epochs=250, GPU=6, checkpoint=str(final),
        checkpoint_sha256=digest(final), checkpoint_role='full250_training_completion_witness',
        evaluation_checkpoint_name=best.BEST, evaluation_checkpoint_path=validated['checkpoint_path'],
        evaluation_checkpoint_sha256=validated['checkpoint_sha256'], best_validation_receipt=str(fold/best.RECEIPT),
        best_validation_receipt_sha256=digest(fold/best.RECEIPT), bank_sha256=native['bank_sha256']))
    old = request['original_declaration']; binding = read(old['storage_binding'])
    publish(Path(old['chain_root'])/'matched_storage_train.json', dict(format=pipeline.STAGE_FORMAT,
        status='COMPLETE', action='train', storage_profile=pipeline.PROFILE,
        storage_admission=old['storage_admission'], storage_admission_sha256=binding['admission']['sha256'],
        runtime_sources_sha256={name:pipeline.sha(pipeline.ROOT/name) for name in pipeline.FILES},
        cached_inputs_written=False, model_data_scale_preserved=True, native_epochs=250,
        native_physical_batch=2, cp_probability=.5, source_entries=642, candidates_per_source=128,
        source_payloads_written=False, evaluation_checkpoint='checkpoint_best.pth', completed_at=time.time()))
    return pipeline.stage_proof(old, 'train')


def watch(request_path):
    import psutil
    if not hasattr(os, 'getuid') or os.getuid() != UID:
        raise ValueError('Actual UID41833 Linux recovery controller required')
    path = regular(request_path); request = read(path); root = regular(path.parent, directory=True)
    if request['chain_root'] != str(root) or psutil.Process().cpu_affinity() != AFFINITY:
        raise ValueError('Exact sealed root and original four CPUs required')
    modules, storage, resources = admit(request, zero=True, resources=True)
    chain, pipeline, runtime, original, best = modules
    actor = psutil.Process(); lease_path = Path(request['original_declaration']['chain_root'])/'native_startup_retry.lease.json'
    lease = dict(format=FORMAT, chain_root=str(root), request_sha256=digest(path),
        pid=actor.pid, create_time=actor.create_time(),
        supervisor_pid=actor.pid, supervisor_create_time=actor.create_time(), command=actor.cmdline(), UID=UID)
    publish(lease_path, lease)  # One global arm lease, even across different new roots.
    publish(root/'lease.json', lease)
    state = dict(format=STATUS_FORMAT, GPU_arm=6, request_sha256=digest(path), source_output=request['original_declaration']['source_output'],
        source_code=request['original_declaration']['source_code'], supervisor_pid=actor.pid,
        supervisor_create_time=actor.create_time(), UID=UID, actual_CPU_affinity=AFFINITY,
        status='READY_FOR_NATIVE_STAGE', stage='train', child_pid=None, child_create_time=None,
        storage=storage, resources=resources, stages_completed=['prepare_native', 'calibrate_native'],
        full_native_training_completed=False, full_native_evaluation_completed=False, rescore=False,
        signals_sent=False, original_failed_records_overwritten=False, binding=BINDING)
    def update(**values):
        state.update(values, checked_at=time.time()); publish(root/'status.json', state, replace=True)
    update()
    old_cache = chain.FileProofCache()
    chain.verify_code(request['original_declaration'], old_cache)
    own_request_proof = proof(path)
    try:
        native = read(request['native_path']); env = pipeline.environment(native, request['native_path'])
        env.update(CUDA_VISIBLE_DEVICES='', CUDA_DEVICE_ORDER='PCI_BUS_ID', V24_ARM_RSS_GIB='48',
            V24_CONTEXT_DEPTH1_CHAIN_REQUEST=request['proofs']['request']['path'],
            V24_CONTEXT_DEPTH1_CHAIN_REQUEST_SHA256=request['proofs']['request']['sha256'],
            V24_READONLY_STORAGE_ADMISSION_SHA256=storage['sha256'])
        log = root/'train.log'; command = request['command']; violation = None
        publish(root/'training_started.json', dict(format=FORMAT, command=command, native_path=request['native_path'],
            original_request=request['original_request'], original_calibration=request['proofs']['calibration'],
            zero_update_startup_retry=True, epochs=250, physical_batch=2, evaluation_checkpoint='checkpoint_best.pth',
            binding=BINDING, started=time.time()))
        with log.open('xb', buffering=0) as stream:
            child = subprocess.Popen(command, cwd=request['code'], env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT)
            process = psutil.Process(child.pid)
            chain.process_snapshot(child.pid, process.create_time(), command)
            update(status='RUNNING', command=command, stage_log=str(log), child_pid=child.pid,
                child_create_time=process.create_time(), stage_started=time.time())
            while child.poll() is None:
                try:
                    verify_proof(own_request_proof)
                    verify_failure(request, chain, zero=False)
                    old_cache.check()
                    verify_new_code(request)
                except (OSError, ValueError, RuntimeError) as error:
                    violation = type(error).__name__+': '+str(error)
                try:
                    rss = process.memory_info().rss
                    for descendant in process.children(recursive=True):
                        try:
                            rss += descendant.memory_info().rss
                        except psutil.NoSuchProcess:
                            continue
                except psutil.NoSuchProcess:
                    if child.poll() is None:
                        raise RuntimeError('Exact own recovery child disappeared without recorded exit')
                    break
                if rss > RAM_BYTES:
                    violation = 'Own native child tree exceeded original48GiB; completion refused'
                update(child_process_tree_RSS_bytes=rss, RAM_budget_bytes=RAM_BYTES, resource_violation=violation)
                time.sleep(5)
            result = child.wait()
        if result or violation:
            raise RuntimeError('Explicit startup retry failed ('+str(result)+'): '+str(violation)+'; preserved log '+str(log))
        admit(request, zero=False, admitted_modules=modules)
        pipeline.verify_admission(request['original_declaration']['storage_admission'])
        pipeline.check_aggregate_disk(pipeline.verify_admission(request['original_declaration']['storage_admission']),
            Path(request['original_declaration']['chain_root']), preparation=False)
        completion = publish_completion(request, pipeline, best)
        publish(root/'completion.json', dict(format=FORMAT, status='COMPLETE', completion_proof=completion,
            zero_update_startup_retry=True, full_native_training_completed=True, full_native_evaluation_completed=True,
            original_request_sha256=request['proofs']['request']['sha256'], binding=BINDING))
        update(status='COMPLETE', stage=None, child_pid=None, child_create_time=None,
            stages_completed=['prepare_native', 'calibrate_native', 'train'], completion_proof=completion,
            full_native_training_completed=True, full_native_evaluation_completed=True, completed=time.time())
    except Exception as error:
        update(status='FAILED', error=type(error).__name__+': '+str(error), failed=time.time())
        raise
    return state


def validate_completion_request(request):
    old = validate_declaration(request['failed_retry_declaration'])
    root = absolute(request['recovery_root']); original_arm = Path(old['original_declaration']['chain_root'])
    if (request.get('format') != COMPLETION_FORMAT or request.get('GPU_arm') != 6
            or request.get('training_rerun') is not False or request.get('evaluation_rerun') is not False
            or request.get('CUDA_compute') is not False or request.get('epochs') != 250
            or request.get('physical_batch') != 2 or request.get('evaluation_checkpoint') != 'checkpoint_best.pth'
            or request.get('python') != old['python'] or root.parent != original_arm
            or root == Path(old['chain_root']) or root.name in ('native','bank')):
        raise ValueError('Fresh CPU-only original full250/BEST26 artifact completion required')
    required = {'failed_retry_request','status','registration','lease','global_retry_lease','entry_lease',
        'training_started','train_log','actual_configuration','best_receipt','best_checkpoint','final_checkpoint'}
    if not isinstance(request.get('proofs'), dict) or set(request['proofs']) != required:
        raise ValueError('Exact failed retry and whole existing training/BEST evidence required')
    code = absolute(request['code']); commit = request.get('commit'); manifest = request.get('code_files')
    if (not isinstance(commit,str) or len(commit)!=40 or set(commit)-set('0123456789abcdef')
            or not isinstance(manifest,dict) or not set(FILES)<=set(manifest)):
        raise ValueError('Fresh completion code commit and manifest required')
    for name, value in manifest.items():
        if not isinstance(name,str) or Path(name).is_absolute() or '..' in Path(name).parts or not checksum(value):
            raise ValueError('Exact tracked completion-code sources required')
    if code == Path(old['code']):
        raise ValueError('Completion repair must use a distinct fixed checkout')
    if request.get('failed_retry_request') != request['proofs']['failed_retry_request']:
        raise ValueError('Exact failed retry request witness required')
    return request


def build_completion_request(failed_retry_request_path, recovery_root, code, commit, code_files):
    """Read-only declaration; training, predictions and failures remain intact."""
    path = regular(failed_retry_request_path); old = validate_declaration(read(path))
    root = path.parent; native = Path(old['native_path']).parent; fold = Path(old['fold'])
    paths = dict(failed_retry_request=path,status=root/'status.json',registration=root/'registration.json',
        lease=root/'lease.json',global_retry_lease=Path(old['original_declaration']['chain_root'])/'native_startup_retry.lease.json',
        entry_lease=root/'entry.lease.json',training_started=root/'training_started.json',train_log=root/'train.log',
        actual_configuration=native/'actual_nnunet_configuration.json',best_receipt=fold/'best_validation_complete.json',
        best_checkpoint=fold/'checkpoint_best.pth',final_checkpoint=fold/'checkpoint_final.pth')
    request = dict(format=COMPLETION_FORMAT,GPU_arm=6,recovery_root=str(absolute(str(recovery_root))),
        code=str(absolute(str(code))),commit=commit,code_files=copy.deepcopy(code_files),python=old['python'],
        failed_retry_declaration=old,proofs={key:proof(path) for key,path in paths.items()},
        training_rerun=False,evaluation_rerun=False,CUDA_compute=False,epochs=250,physical_batch=2,
        evaluation_checkpoint='checkpoint_best.pth')
    request['failed_retry_request']=request['proofs']['failed_retry_request']
    return validate_completion_request(request)


def seal_completion_request(request):
    validate_completion_request(request);root=absolute(request['recovery_root'])
    if root.exists():
        regular(root,directory=True)
        if list(root.iterdir()):raise FileExistsError('Fresh empty completion namespace required')
    else:
        regular(root.parent,directory=True);root.mkdir(mode=0o700)
    publish(root/'request.json',request)
    return root/'request.json'


def verify_completed_retry_evidence(request, chain, pipeline, best):
    """Require the successful full CLI artifacts behind the post-proof failure."""
    validate_completion_request(request);old=request['failed_retry_declaration'];root=Path(old['chain_root'])
    fold=Path(old['fold']);native_path=Path(old['native_path']);arm=Path(old['original_declaration']['chain_root'])
    paths=dict(failed_retry_request=root/'request.json',status=root/'status.json',registration=root/'registration.json',
        lease=root/'lease.json',global_retry_lease=arm/'native_startup_retry.lease.json',entry_lease=root/'entry.lease.json',
        training_started=root/'training_started.json',train_log=root/'train.log',
        actual_configuration=native_path.parent/'actual_nnunet_configuration.json',
        best_receipt=fold/best.RECEIPT,best_checkpoint=fold/best.BEST,final_checkpoint=fold/'checkpoint_final.pth')
    for key, expected in paths.items():
        if verify_proof(request['proofs'][key]) != expected:
            raise ValueError('Existing completion witness path differs: '+key)
    if read(root/'request.json') != old:raise ValueError('Preserved failed retry request changed')
    state=read(root/'status.json');reg=read(root/'registration.json');lease=read(root/'lease.json')
    expected_parent=[old['python'],'-B','-u',str(Path(old['code'])/'tools/run_v24_native_startup_retry.py'),
        'watch','--request',str(root/'request.json')]
    digest_old=request['failed_retry_request']['sha256'];error=state.get('error','')
    if (state.get('format')!=STATUS_FORMAT or state.get('status')!='FAILED' or state.get('stage')!='train'
            or not error.startswith(POST_PROOF_ERROR) or not checksum(error[len(POST_PROOF_ERROR):])
            or state.get('GPU_arm')!=6 or state.get('UID')!=UID or state.get('actual_CPU_affinity')!=AFFINITY
            or state.get('resource_violation') is not None or state.get('request_sha256')!=digest_old
            or state.get('source_output')!=old['original_declaration']['source_output']
            or state.get('full_native_training_completed') is not False
            or state.get('full_native_evaluation_completed') is not False
            or state.get('stages_completed')!=['prepare_native','calibrate_native']
            or state.get('command')!=old['command'] or state.get('stage_log')!=str(root/'train.log')
            or reg.get('pid')!=state.get('supervisor_pid') or reg.get('create_time')!=state.get('supervisor_create_time')
            or reg.get('command')!=expected_parent or reg.get('request_sha256')!=digest_old
            or reg.get('root')!=str(root) or reg.get('GPU_arm')!=6 or reg.get('UID')!=UID
            or reg.get('CPU_affinity')!=AFFINITY or reg.get('commit')!=old['commit']
            or lease.get('supervisor_pid')!=reg.get('pid') or lease.get('supervisor_create_time')!=reg.get('create_time')
            or lease.get('pid')!=reg.get('pid') or lease.get('create_time')!=reg.get('create_time')
            or lease.get('UID')!=UID or lease.get('command')!=expected_parent or lease.get('request_sha256')!=digest_old
            or read(paths['global_retry_lease'])!=lease):
        raise ValueError('Exact completed child/post-proof failed retry identity required')
    entry=read(root/'entry.lease.json');started=read(root/'training_started.json')
    if (entry.get('pid')!=state.get('child_pid') or entry.get('create_time')!=state.get('child_create_time')
            or entry.get('command')!=old['command'] or entry.get('UID')!=UID or entry.get('request_sha256')!=digest_old
            or started.get('format')!=FORMAT or started.get('command')!=old['command']
            or started.get('native_path')!=old['native_path'] or started.get('original_request')!=old['original_request']
            or started.get('original_calibration')!=old['proofs']['calibration']
            or started.get('zero_update_startup_retry') is not True or started.get('epochs')!=250
            or started.get('physical_batch')!=2 or started.get('evaluation_checkpoint')!=best.BEST
            or started.get('binding')!=BINDING):
        raise ValueError('Actual original full250/BEST retry entry/start binding required')
    for pid,birth,command in ((reg['pid'],reg['create_time'],expected_parent),
            (entry['pid'],entry['create_time'],old['command'])):
        if chain.process_snapshot(pid,birth,command) is not None:
            raise RuntimeError('Exact prior retry actor is still active; completion refused')
    text=paths['train_log'].read_text(encoding='utf8')
    epochs=[int(value) for value in re.findall(r'^\d{4}-\d\d-\d\d [^\n]*: Epoch (\d+)\s*$',text,re.MULTILINE)]
    if (epochs!=list(range(250)) or 'Training done.' not in text or 'Validation complete' not in text
            or 'Traceback (most recent call last):' in text):
        raise ValueError('Actual complete successful epochs0..249 and full validation log required')
    config=read(paths['actual_configuration'])
    if (config.get('epochs')!=250 or config.get('physical_batch')!=2 or config.get('train_cases')!=105
            or config.get('validation_cases')!=26 or config.get('patch_size')!=[128]*3
            or config.get('CPU_workers')!=4 or config.get('source_entries')!=642
            or config.get('candidate_count')!=128 or config.get('validation_checkpoint')!=best.BEST
            or config.get('debug') is not False):
        raise ValueError('Actual whole-scale completed native configuration required')
    native=read(native_path)
    validated=best.validate_best_validation_receipt(paths['best_receipt'],fold=fold,
        expected_cases=native['split']['outer_val'],bank_sha256=native['bank_sha256'])
    pipeline._guard_native(native_path);pipeline.validate_calibration(native_path)
    for value in request['proofs'].values():verify_proof(value)
    return validated


def complete_existing_artifacts(request_path):
    """Publish proofs for existing full250/BEST26 only; no training/evaluation CLI."""
    import psutil
    path=regular(request_path);request=validate_completion_request(read(path));root=regular(path.parent,directory=True)
    if path!=Path(request['recovery_root'])/'request.json':raise ValueError('Exact fresh completion request path required')
    if Path(__file__).resolve()!=Path(request['code'])/FILES[0]:raise ValueError('Exact fixed completion module required')
    actor=psutil.Process()
    if actor.uids().real!=UID or actor.cpu_affinity()!=AFFINITY:
        raise ValueError('Own UID41833/original four CPUs required for CPU completion')
    own_request_proof=proof(path)
    if read(path)!=request:raise ValueError('Completion request changed during admission')
    verify_new_code(request)
    old=request['failed_retry_declaration']
    # One first-import admission only: separately admitted historical storage
    # helpers are retained for all later proofs in this process.
    modules,storage,_=admit(old,zero=False,resources=False)
    chain,pipeline,runtime,original,best=modules
    validated=verify_completed_retry_evidence(request,chain,pipeline,best)
    for value in request['proofs'].values():verify_proof(value)
    arm=Path(old['original_declaration']['chain_root'])
    if list(Path(old['native_path']).parent.glob('training_complete_*.json')) or (arm/'matched_storage_train.json').exists():
        raise FileExistsError('Canonical completion already exists; explicit review required, no overwrite')
    lease=dict(format=COMPLETION_FORMAT,pid=actor.pid,create_time=actor.create_time(),command=actor.cmdline(),
        UID=UID,request_sha256=digest(path),recovery_root=str(root),training_rerun=False,evaluation_rerun=False,CUDA_compute=False)
    publish(arm/'native_post_training_completion.lease.json',lease);publish(root/'lease.json',lease)
    state=dict(format=COMPLETION_STATUS,GPU_arm=6,request_sha256=digest(path),source_output=old['original_declaration']['source_output'],
        source_code=old['original_declaration']['source_code'],supervisor_pid=actor.pid,supervisor_create_time=actor.create_time(),
        UID=UID,status='FINALIZING_EXISTING_ARTIFACTS',stage='train',child_pid=None,child_create_time=None,
        stages_completed=['prepare_native','calibrate_native'],full_native_training_completed=False,
        full_native_evaluation_completed=False,training_rerun=False,evaluation_rerun=False,CUDA_compute=False,
        signals_sent=False,original_failed_records_overwritten=False,storage=storage)
    publish(root/'status.json',state)
    try:
        verify_proof(own_request_proof);verify_new_code(request)
        pipeline.check_aggregate_disk(pipeline.verify_admission(old['original_declaration']['storage_admission']),arm,preparation=False)
        completion=publish_completion(old,pipeline,best)
        for value in request['proofs'].values():verify_proof(value)
        verify_proof(own_request_proof)
        publish(root/'completion.json',dict(format=COMPLETION_FORMAT,status='COMPLETE',completion_proof=completion,
            failed_retry_request=request['failed_retry_request'],best_validation_receipt=request['proofs']['best_receipt'],
            full_native_training_completed=True,full_native_evaluation_completed=True,
            training_rerun=False,evaluation_rerun=False,CUDA_compute=False))
        state.update(status='COMPLETE',stage=None,stages_completed=['prepare_native','calibrate_native','train'],
            full_native_training_completed=True,full_native_evaluation_completed=True,completion_proof=completion,
            evaluation_checkpoint=best.BEST,completed_epochs=250,evaluated_cases=len(validated['validation_cases']),completed=time.time())
    except Exception as error:
        state.update(status='FAILED',error=type(error).__name__+': '+str(error),failed=time.time())
        publish(root/'status.json',state,replace=True);raise
    publish(root/'status.json',state,replace=True)
    return state
