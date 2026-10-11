"""Observe one authorized GPU5 epoch boundary and request a cooperative pause.

This observer never sends a process signal and never writes a checkpoint or a
scientific result. Its sole scientific-directory write is an exclusive
STOP_AFTER_BATCH sentinel. Polling is not an atomic fence: a late boundary is a
failure, never a successful early stop. Final verification uses the unchanged
CPU checkpoint reader after the exact original worker has stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import time

FORMAT = 'v24_gpu5_pause_after_current_epoch_guardian_request_v1'
RECEIPT = 'v24_gpu5_observed_complete_epoch_cooperative_pause_receipt_v1'
AUTH = 'v24_user_authorized_GNN_epoch_boundary_stop_request_v1'
BASE = Path('/var/tmp/aicompetition06_v24_20261010')
OUTPUT = BASE / 'GPU45_interrupted_exact_CPU_prepared_20261011_1791705559311834000/gpu5/gnn'
UID, PID, TARGET, UPDATES = 41833, 1822677, 27, 459
SOURCE_NAMES = ('hiercp_v1x/v24_training.py',
                'hiercp_v1x/v24_training_continuation.py',
                'hiercp_v1x/v23_training.py', 'hiercp_v1x/u_bridge_training.py')
SOURCE_SHAS = dict(zip(SOURCE_NAMES, (
    'd74822a9429666d48561d2295dd739f2402791cbf9c543e101426a06be631ca5',
    'b58ac32b81b901e04f29824881b87f83536629b6740aefdc90d1555e5f97e0d5',
    'e8d488a95f0950382c464aab755bab7be5f2f1b0c33b86029df779a6b347af34',
    '9bcb9d67fc2b69861028b5241110270a144ee0ea849c573ab757da7ba391a8cc')))


def _fail_constant(value):
    raise ValueError('Nonfinite JSON constant is forbidden: ' + value)


def _loads(raw):
    return json.loads(raw, parse_constant=_fail_constant)


def _stat(value):
    return [value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns]


def _owned(path, *, directory=False):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('Absolute owned path required: ' + str(path))
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('Symlink paths are forbidden: ' + str(parent))
    value = path.lstat()
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if value.st_uid != UID or not expected(value.st_mode):
        raise ValueError('Exact owned regular file/directory required: ' + str(path))
    if not directory and value.st_nlink != 1:
        raise ValueError('An independent regular file is required: ' + str(path))
    return path


def _bytes(path):
    path = _owned(path)
    before = _stat(path.stat())
    with path.open('rb') as stream:
        if _stat(os.fstat(stream.fileno())) != before:
            raise ValueError('File changed before read: ' + str(path))
        raw = stream.read()
    if _stat(path.stat()) != before:
        raise ValueError('File changed during read: ' + str(path))
    return raw, dict(path=str(path), stat=before, sha256=hashlib.sha256(raw).hexdigest())


def _verify(proof):
    raw, actual = _bytes(proof['path'])
    expected_stat = proof.get('stat', proof.get('source_stat'))
    if actual['sha256'] != proof['sha256'] or actual['stat'] != list(expected_stat):
        raise ValueError('Exact file witness changed: ' + proof['path'])
    return raw, actual


def _exclusive(path, value):
    path = Path(path)
    raw = json.dumps(value, indent=2, allow_nan=False).encode('utf8') + b'\n'
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return _bytes(path)[1]


def _status(root, value):
    # Only this newly created guardian namespace is replaceable.
    temporary = root / ('status.' + str(time.time_ns()) + '.json')
    _exclusive(temporary, value)
    os.replace(temporary, root / 'status.json')
    print(json.dumps(value, allow_nan=False), flush=True)


def _worker(expected):
    import psutil
    try:
        process = psutil.Process(expected['pid'])
        with process.oneshot():
            actual = dict(pid=process.pid, create_time=process.create_time(),
                          command=process.cmdline(), uid=process.uids().real)
            zombie = process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    if actual != expected:
        raise ValueError('Original worker UID/PID/birth/full argv changed or PID reused')
    return not zombie


def _rows(raw):
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Complete newline-terminated curve witness required')
    rows = [_loads(line) for line in raw.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Each curve record must be a JSON object')
    if [row.get('epoch') for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Curve epochs are missing, duplicated, or reordered')
    return rows


def _target(row, owner):
    if row.get('epoch') != TARGET or row.get('update') != UPDATES:
        raise ValueError('Target epoch/update boundary differs from authorization')
    train, val = owner['binding']['train_cases'], owner['binding']['val_cases']
    checks = ((row['train'], train, 527), (row['train_probe'], train, 527),
              (row['stage_validation'], val, 135), (row['full_validation'], val, 135))
    for report, cases, observed_p in checks:
        actual_cases = [entry['case_id'] for entry in report['cases']]
        coverage_matches = (set(actual_cases) == set(cases) if report is row['train']
                            else actual_cases == cases)
        if not coverage_matches:
            raise ValueError('Target epoch lost actual complete train/validation population')
        if (len(report['cases']) != len(cases)
                or report['denominators']['patients'] != len(cases)
                or report['denominators']['observed_P'] != observed_p):
            raise ValueError('Target epoch complete population/P denominator differs')
    for phase in ('train_probe', 'stage_validation', 'full_validation'):
        report = row[phase]
        if (report.get('epoch') != TARGET or report.get('updates') != UPDATES
                or report.get('phase') != phase or report.get('all_P_scored') is not True
                or report.get('debug') is not False
                or report.get('recipient_GT_used_in_forward') is not False):
            raise ValueError('Target actual evaluation report differs: ' + phase)
    if row['full_validation']['active_u'] != 128:
        raise ValueError('Target fixed full128 validation is incomplete')


def _admit(request, request_proof):
    if (request.get('format') != FORMAT or request.get('UID') != UID
            or request.get('target_epoch') != TARGET
            or request.get('expected_updates') != UPDATES
            or Path(request.get('output_root', '')) != OUTPUT
            or request.get('initial_history_epochs') != list(range(1, TARGET))):
        raise ValueError('Exact authorized GPU5 e27/update459 observer request required')
    if os.getuid() != UID:
        raise ValueError('Original UID41833 required')
    output = _owned(OUTPUT, directory=True)
    training = _owned(output / 'training', directory=True)
    source = _owned(request['source_code'], directory=True)
    if not source.is_relative_to(BASE) and not source.is_relative_to(Path('/dev/shm/aicompetition06_v24_gpu6_DEBUG_20261011')):
        raise ValueError('Actual previously frozen task code required')
    worker = request['worker']
    if (set(worker) != {'pid', 'create_time', 'command', 'uid'}
            or worker['pid'] != PID or worker['uid'] != UID
            or not isinstance(worker['create_time'], (int, float))
            or not math.isfinite(worker['create_time'])
            or not isinstance(worker['command'], list) or not worker['command']
            or not _worker(worker)):
        raise ValueError('The exact original live GPU5 actor is required before arming')
    for option, value in (('--action', 'train'), ('--gpu', '5'), ('--output', str(output)),
                          ('--source-code', str(source))):
        if worker['command'].count(option) != 1:
            raise ValueError('Exact original CLI flag required: ' + option)
        index = worker['command'].index(option)
        if index + 1 == len(worker['command']) or worker['command'][index + 1] != value:
            raise ValueError('Original worker flag/value differs: ' + option)
    if '/.conda/envs/nnunet/bin/python' not in worker['command'][0]:
        raise ValueError('The original nnunet Python interpreter is required')
    proofs = request['source_files']
    if not set(SOURCE_NAMES) <= set(proofs):
        raise ValueError('Sealed trainer/checkpoint/publisher/hash sources are required')
    if any(proofs[name]['sha256'] != SOURCE_SHAS[name] for name in SOURCE_NAMES):
        raise ValueError('The explicitly audited original cooperative pause protocol changed')
    for name, proof in proofs.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or Path(proof['path']) != source / name:
            raise ValueError('Explicit exact source-code witness required: ' + name)
        _verify(proof)
    owner_raw, _ = _verify(request['training_identity'])
    if Path(request['training_identity']['path']) != training / 'training_identity.json':
        raise ValueError('Actual training identity path required')
    owner = _loads(owner_raw)
    binding = owner['binding']
    if (binding['epochs'] != 40 or binding['debug'] is not False
            or len(binding['train_cases']) != 65 or len(binding['val_cases']) != 21
            or binding['physical_patient_batch'] != 4 or binding['workers'] != 4):
        raise ValueError('The unchanged actual full40/B4/whole population binding is required')
    for name, expected in binding['identity']['source'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Original scientific source manifest path is invalid')
        if _bytes(source / name)[1]['sha256'] != expected:
            raise ValueError('Original actual scientific source differs: ' + name)
    contract_raw, _ = _verify(request['execution_contract'])
    if Path(request['execution_contract']['path']) != training / 'execution_contract.json':
        raise ValueError('Actual execution contract path required')
    contract = _loads(contract_raw)
    if (contract['epochs'] != 40 or contract['debug'] is not False
            or contract['full_training_completed'] is not False):
        raise ValueError('Original full40 contract must remain honest and unchanged')
    auth_raw, auth_proof = _verify(request['authorization'])
    authorization = _loads(auth_raw)
    if (authorization.get('format') != AUTH or authorization.get('GPU_arm') != 5
            or Path(authorization.get('source_output', '')) != output
            or Path(authorization.get('source_code', '')) != source
            or authorization.get('target_epoch') != TARGET
            or authorization.get('original_planned_epochs') != 40
            or authorization.get('user_authorized') is not True
            or authorization.get('stop_policy') != 'after_current_epoch_training_and_full128_validation'
            or authorization.get('original_actor') != {key: worker[key] for key in ('pid', 'create_time', 'command')}
            or not isinstance(authorization.get('reason'), str) or not authorization['reason'].strip()
            or not isinstance(authorization.get('created_at'), (int, float))
            or not math.isfinite(authorization['created_at'])):
        raise ValueError('Exact human-authorized epoch-boundary stop document required')
    if Path(request['initial_curve']['path']) != training / 'curve.jsonl':
        raise ValueError('Actual current training curve path required')
    curve_raw, curve_proof = _verify(request['initial_curve'])
    if len(_rows(curve_raw)) != TARGET - 1:
        raise ValueError('Observer arrived after the target boundary or initial history differs')
    if (training / 'STOP_AFTER_BATCH').exists() or (training / 'STOP_AFTER_BATCH').is_symlink():
        raise FileExistsError('An existing pause sentinel must not be replaced or reinterpreted')
    root = Path(request['guardian_root'])
    if not root.is_relative_to(BASE) or root == output or root.is_relative_to(output):
        raise ValueError('A fresh disjoint task-owned guardian namespace is required')
    _owned(root.parent, directory=True)
    root.mkdir(mode=0o700, exist_ok=False)
    _owned(root, directory=True)
    _exclusive(root / 'request_bound.json', dict(request=request, proof=request_proof))
    return root, training, source, owner, curve_raw, curve_proof, auth_proof


def _checkpoint_verify(request, source, owner, curve_row):
    # The worker is absent: CPU readers cannot race its atomic checkpoint writer.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    for proof in request['source_files'].values():
        _verify(proof)
    for name, expected in owner['binding']['identity']['source'].items():
        if _bytes(source / name)[1]['sha256'] != expected:
            raise ValueError('Original source changed before CPU inspection: ' + name)
    sys.path.insert(0, str(source))
    import torch
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise ValueError('Checkpoint verification must not initialize CUDA')
    from hiercp_v1x.v24_training_continuation import _checkpoint
    from hiercp_v1x.u_bridge_training import digest
    from hiercp_v1x.v23_training import best_selection_key
    training = OUTPUT / 'training'
    if digest(owner['binding']) != owner['identity_sha256']:
        raise ValueError('Actual owner binding checksum differs')
    saved, latest = _checkpoint(training / 'checkpoint_latest.pt', owner)
    state = saved['state']
    if (state['status'] != 'PAUSED' or state['phase'] != 'training'
            or state['epoch'] != TARGET + 1 or state['updates'] != UPDATES
            or state['train_position'] != 0 or state['train_rows']
            or state['evaluation_position'] != 0 or state['evaluation_rows']
            or 'pending_epoch_completion' in state
            or len(state['history']) != TARGET
            or [row['epoch'] for row in state['history']] != list(range(1, TARGET + 1))
            or digest(state['history'][-1]) != digest(curve_row)
            or latest['optimizer_parameter_tensors'] != 981):
        raise ValueError('LATE_OR_INCOMPLETE_BOUNDARY: actual PAUSED checkpoint is not e28/history27/update459/cursor0')
    selected = state['best']
    if selected is None:
        raise ValueError('The actual owned full128 BEST checkpoint is required')
    best, best_proof = _checkpoint(training / 'checkpoint_best.pt', owner)
    full = best['state']['pending_epoch_completion']['full_validation']
    if (best['state']['best'] != selected or best['state']['epoch'] != selected['epoch']
            or best['state']['updates'] != selected['updates']
            or selected['epoch'] > TARGET or full['phase'] != 'full_validation'
            or full['active_u'] != 128
            or list(best_selection_key(full)) != selected['selection_key']):
        raise ValueError('Actual saved owned BEST winning full128 state differs')
    if torch.cuda.is_initialized():
        raise ValueError('CPU checkpoint inspection unexpectedly initialized CUDA')
    for proof in request['source_files'].values():
        _verify(proof)
    for name, expected in owner['binding']['identity']['source'].items():
        if _bytes(source / name)[1]['sha256'] != expected:
            raise ValueError('Original source changed during CPU inspection: ' + name)
    return latest, best_proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--wait-seconds', type=float, default=7200.)
    parser.add_argument('--pause-wait-seconds', type=float, default=600.)
    args = parser.parse_args(argv)
    if (not math.isfinite(args.wait_seconds) or args.wait_seconds <= 0
            or not math.isfinite(args.pause_wait_seconds) or args.pause_wait_seconds <= 0):
        raise ValueError('Explicit positive finite waiting timeouts required')
    raw, request_proof = _bytes(args.request)
    if request_proof['sha256'] != args.request_sha256:
        raise ValueError('Exact guardian request SHA required')
    request = _loads(raw)
    root = None
    sentinel = None
    try:
        root, training, source, owner, prefix, initial, auth = _admit(request, request_proof)
        started = time.time()
        common = dict(format=RECEIPT, GPU_arm=5, target_epoch=TARGET,
                      expected_updates=UPDATES, original_planned_epochs=40,
                      full_training_completed=False, atomic_boundary_guaranteed=False,
                      polling_seconds=.02, signals_sent=False,
                      source_output=str(OUTPUT), source_code=str(source),
                      request=request_proof, authorization=auth, worker=request['worker'])
        _status(root, dict(common, status='WAITING_COMPLETE_EPOCH_CURVE', verified=False, created_at=started))
        curve = training / 'curve.jsonl'
        training_inode = tuple(_stat(training.stat())[:2])
        buffered = b''
        offset = len(prefix)
        deadline = time.monotonic() + args.wait_seconds
        target_row = None
        while target_row is None:
            if not _worker(request['worker']):
                raise ValueError('Original worker stopped before the authorized completed epoch curve')
            if time.monotonic() > deadline:
                raise TimeoutError('Target completed epoch was not observed before the explicit deadline')
            _owned(training, directory=True)
            _owned(curve)
            if tuple(_stat(training.stat())[:2]) != training_inode:
                raise ValueError('Training directory inode changed during observation')
            current = curve.stat()
            if _stat(current)[:2] != initial['stat'][:2] or current.st_size < offset:
                raise ValueError('Training curve replaced or truncated during observation')
            if current.st_size > offset:
                with curve.open('rb') as stream:
                    if _stat(os.fstat(stream.fileno()))[:2] != initial['stat'][:2]:
                        raise ValueError('Training curve inode changed before tail read')
                    stream.seek(offset)
                    chunk = stream.read()
                offset += len(chunk)
                buffered += chunk
                if b'\n' in buffered:
                    completed, separator, remaining = buffered.rpartition(b'\n')
                    appended = _rows(completed + separator) if not prefix else [
                        _loads(line) for line in completed.splitlines() if line.strip()]
                    if len(appended) != 1:
                        raise ValueError('LATE_OR_INCOMPLETE_BOUNDARY: more than the target completed epoch appeared')
                    target_row = appended[0]
                    _target(target_row, owner)
                    if remaining:
                        raise ValueError('LATE_BOUNDARY: next curve already began publication')
                    if not _worker(request['worker']):
                        raise ValueError('Original worker stopped before cooperative pause request')
                    _owned(training, directory=True)
                    if tuple(_stat(training.stat())[:2]) != training_inode:
                        raise ValueError('Training directory changed before sentinel creation')
                    sentinel = _exclusive(training / 'STOP_AFTER_BATCH',
                        dict(format='v24_owned_single_actor_cooperative_pause_sentinel_v1',
                             guardian=str(root), authorization=auth,
                             target_epoch=TARGET, expected_updates=UPDATES,
                             requested_at=time.time(), original_actor=request['worker']))
                    _status(root, dict(common, status='PAUSE_SENTINEL_CREATED', verified=False,
                                       sentinel=sentinel, observed_curve_epoch=TARGET, observed_at=time.time()))
            if target_row is None:
                time.sleep(.02)
        deadline = time.monotonic() + args.pause_wait_seconds
        while _worker(request['worker']):
            if time.monotonic() > deadline:
                raise TimeoutError('Cooperative pause did not stop the exact original worker before deadline; no signal sent')
            time.sleep(.02)
        _verify(request['training_identity'])
        _verify(request['execution_contract'])
        _verify(request['authorization'])
        curve_raw, final_curve = _bytes(curve)
        if not curve_raw.startswith(prefix) or len(_rows(curve_raw)) != TARGET:
            raise ValueError('LATE_OR_INCOMPLETE_BOUNDARY: immutable preceding history or target count differs')
        final_rows = _rows(curve_raw)
        if final_rows[-1] != target_row:
            raise ValueError('Observed target curve changed after pause')
        latest, best = _checkpoint_verify(request, source, owner, target_row)
        invocation_raw, invocation_proof = _bytes(training / 'invocations.jsonl')
        invocations = [_loads(line) for line in invocation_raw.splitlines() if line.strip()]
        if not invocations or any(invocations[-1].get(key) != value for key, value in
                dict(status='PAUSED', completed_epochs=TARGET, optimizer_updates=UPDATES,
                     full_training=False, debug=False, actual_CUDA=True,
                     recipient_GT_used_in_forward=False, expected_parameter_tensors=981).items()):
            raise ValueError('The actual original final PAUSED invocation receipt is missing or differs')
        if _worker(request['worker']):
            raise ValueError('Original worker unexpectedly became live during final verification')
        result = dict(common, status='VERIFIED_PAUSED_AT_EPOCH_BOUNDARY', verified=True,
                      created_at=started, verified_at=time.time(), sentinel=sentinel,
                      final_curve=final_curve, checkpoint_latest=latest, checkpoint_best=best,
                      invocation=invocation_proof, completed_epochs=TARGET,
                      optimizer_updates=UPDATES, next_epoch=TARGET + 1,
                      next_epoch_optimizer_updates=0, CPU_only_checkpoint_inspection=True,
                      six_saved_numerical_states_verified=True,
                      scientific_files_written=['training/STOP_AFTER_BATCH'])
        _exclusive(root / 'receipt.json', result)
        _status(root, result)
        return result
    except Exception as error:
        failure = dict(format=RECEIPT, status='FAILED', verified=False,
                       error=type(error).__name__ + ': ' + str(error),
                       target_epoch=TARGET, expected_updates=UPDATES,
                       full_training_completed=False, atomic_boundary_guaranteed=False,
                       signals_sent=False, sentinel=sentinel, created_at=time.time(),
                       recovery='Preserve original checkpoints and this sentinel; inspect exact state explicitly. No success receipt was issued.')
        if root is not None:
            _exclusive(root / ('failure_' + str(time.time_ns()) + '.json'), failure)
            _status(root, failure)
        else:
            print(json.dumps(failure, allow_nan=False), flush=True)
        raise


if __name__ == '__main__':
    main()
