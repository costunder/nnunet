"""Inspect, preserve, then explicitly resume one existing comparison arm.

Default inspection is read-only. This tool never requests a pause, signals an
existing process, removes a lock, or edits an existing checkpoint. A resume requires a
complete, verified byte-copy backup of the currently quiescent arm.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from tools import run_comparison_arm as preserved
from tools.run_allocated_comparison_arm import (CURRICULUM_POLICY, _curriculum_dispatch,
                                               parse as parse_arm)

FORMAT = 'comparison_curriculum_full_backup_v1'
BACKUPS = 'curriculum_migration_backups'
KNOWN_LAUNCHERS = {'run_v18_u_bridge.py', 'run_v19_comparison.py', 'run_v18_independent.py',
                   'resume_comparison_cached.py', 'run_comparison_arm.py',
                   'run_allocated_comparison_arm.py'}


def parse(argv=None):
    supplied = list(sys.argv[1:] if argv is None else argv)
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument('--action', choices=('inspect', 'backup', 'resume'), default='inspect')
    extra.add_argument('--backup', type=Path)
    selected, remainder = extra.parse_known_args(supplied)
    if '--help' in remainder or '-h' in remainder:
        print(__doc__)
        print('Migration options: --action {inspect,backup,resume} (default inspect); --backup PATH')
    arguments = parse_arm(remainder)
    arguments.action, arguments.backup = selected.action, selected.backup
    if arguments.owned_child and selected.action != 'resume':
        extra.error('Only an explicit resume can be an owned child')
    if selected.action == 'resume' and (selected.backup is None or arguments.curriculum_policy is None):
        extra.error('Resume requires --backup and explicit --curriculum-policy')
    if selected.action != 'resume' and selected.backup is not None:
        extra.error('--backup selects a complete backup only for resume')
    return arguments


def _option(command, name, value):
    return any(command[i:i+2] == [name, value] for i in range(len(command) - 1))


def _matching_processes(request):
    """Read matching process metadata only, including jobs before lock creation."""
    import psutil
    result = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'username']):
        row = process.info
        command = row.get('cmdline') or []
        if row['pid'] == os.getpid() or not str(row.get('name') or '').lower().startswith('python'):
            continue
        known = any(Path(item).name in KNOWN_LAUNCHERS for item in command)
        migration_child = (any(Path(item).name == Path(__file__).name for item in command)
                           and '--owned-child' in command)
        if ((known or migration_child) and _option(command, '--experiment', str(request['experiment']))
                and _option(command, '--arm', request['arm'])):
            result.append(dict(pid=row['pid'], create_time=row['create_time'], user=row['username'], command=command))
    return result


def inspect(request):
    root, arm = request['experiment'], request['arm']
    owner = preserved._active_owner(request)
    locks = [root / '.pipeline.lock', root / '.setup.lock', root / ('.' + arm + '.lock'),
             request['data_root'] / '.data.lock']
    present = [str(path) for path in locks if path.exists() or path.is_symlink()]
    processes = _matching_processes(request)
    pause = root / arm / 'STOP_AFTER_BATCH'
    return dict(arm=arm, experiment=str(root), requested_physical_gpu=request['gpu'],
                active_owner=owner, matching_processes=processes, locks=present,
                pause_file_present=pause.exists() or pause.is_symlink(),
                quiescent=owner is None and not present and not processes,
                no_process_signaled=True, no_existing_file_modified=True)


def _quiescent(request):
    state = inspect(request)
    if not state['quiescent']:
        raise RuntimeError('Arm must already be stopped with no active writer or lock; nothing was stopped: '
                           + json.dumps(state, allow_nan=False))
    if request['requires_clone']:
        raise ValueError('Migration requires an existing independent arm; no clone or fresh training is allowed')
    return state


def _files(request):
    root, arm = request['experiment'], request['arm']
    required = {'experiment.json', 'initial.pt', 'calibration.json',
                f'{arm}/training_identity.json', f'{arm}/checkpoint_latest.pt', f'{arm}/checkpoint_best.pt'}
    if arm in ('selected', 'native'):
        required.add('continuation.json')
    files = set(required)
    for directory in (root, root / arm, root / 'invocations', root / 'execution_overrides'):
        files.update(path.relative_to(root).as_posix() for path in directory.glob('*.json'))
    for relative in files:
        path = root / relative
        preserved._path(path)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Complete regular preserved checkpoint/metadata file required: {path}')
    return sorted(files)


def _stat(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _record(path):
    before = _stat(path)
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            checksum.update(block)
    if _stat(path) != before:
        raise RuntimeError(f'File changed while reading; no stable backup can be certified: {path}')
    return dict(bytes=before[2], sha256=checksum.hexdigest())


def _write_new(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_new(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    checksum, size = hashlib.sha256(), 0
    with source.open('rb') as src, destination.open('xb') as dst:
        for block in iter(lambda: src.read(4 * 2**20), b''):
            dst.write(block)
            checksum.update(block)
            size += len(block)
        dst.flush()
        os.fsync(dst.fileno())
    if dict(bytes=size, sha256=checksum.hexdigest()) != expected or _record(destination) != expected:
        raise RuntimeError(f'Full backup byte verification failed; partial copy retained: {destination}')
    if os.path.samefile(source, destination):
        raise RuntimeError('Checkpoint backup must be an independent full copy, never a hard link')


def resume_command(arguments, request, backup):
    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--action', 'resume',
               '--backup', str(backup), '--curriculum-policy', CURRICULUM_POLICY,
               '--gpu', str(request['gpu']), '--arm', request['arm'],
               '--experiments-dir', str(arguments.experiments_dir), '--experiment', str(request['experiment']),
               '--source-experiment', str(request['source_experiment']), '--inventory', str(request['inventory']),
               '--cache-sources', *map(str, request['cache_sources'])]
    for name in ('fixture', 'config', 'source', 'bank'):
        value = getattr(arguments, 'debug_' + name)
        if value is not None:
            command += ['--debug-' + name, str(value)]
    return command


def backup(arguments, request):
    state = _quiescent(request)
    root = request['experiment']
    names = _files(request)
    originals = {name: _record(root / name) for name in names}
    _quiescent(request)
    base = preserved._path(root / BACKUPS)
    base.mkdir(exist_ok=True)
    destination = base / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex)
    destination.mkdir(exist_ok=False)
    _write_new(destination / 'backup_in_progress.json',
               dict(format=FORMAT, experiment=str(root), arm=request['arm'], files=names))
    for name in names:
        _copy_new(root / name, destination / 'files' / name, originals[name])
    if _files(request) != names or any(_record(root / name) != originals[name] for name in names):
        raise RuntimeError(f'Original files changed during backup; incomplete copy retained: {destination}')
    _quiescent(request)
    command = resume_command(arguments, request, destination)
    receipt = dict(format=FORMAT, complete=True, experiment=str(root), arm=request['arm'],
        curriculum_policy=CURRICULUM_POLICY, full_independent_byte_copies=True,
        files=originals, inspection=state, resume_argv=command,
        restore_rule='copy files back to recorded relative paths only after explicitly stopping the exact arm')
    receipt['sha256'] = preserved._digest(receipt)
    _write_new(destination / 'backup_complete.json', receipt)
    return dict(status='BACKUP_COMPLETE', backup=str(destination), files=len(names),
                bytes=sum(row['bytes'] for row in originals.values()), resume_argv=command,
                resume_shell=shlex.join(command), no_process_started=True,
                no_existing_file_modified=True)


def verify_backup(arguments, request):
    state = _quiescent(request)
    if state['pause_file_present']:
        raise RuntimeError('The existing STOP_AFTER_BATCH flag is preserved; clear it explicitly before resume')
    destination = preserved._path(arguments.backup)
    if destination.parent != request['experiment'] / BACKUPS:
        raise ValueError('Backup must be the unique full-copy directory under this experiment')
    receipt = preserved._read(destination / 'backup_complete.json')
    checksum = receipt.get('sha256')
    if checksum != preserved._digest({k: v for k, v in receipt.items() if k != 'sha256'}):
        raise ValueError('Backup receipt digest changed')
    if (receipt.get('format') != FORMAT or receipt.get('complete') is not True
            or receipt.get('full_independent_byte_copies') is not True
            or receipt.get('experiment') != str(request['experiment'])
            or receipt.get('arm') != request['arm']
            or receipt.get('curriculum_policy') != arguments.curriculum_policy):
        raise ValueError('Backup does not belong to this exact continuation policy/arm/experiment')
    files = receipt['files']
    if not isinstance(files, dict) or sorted(files) != _files(request):
        raise ValueError('Backup file inventory differs from the current preserved experiment')
    for name, expected in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('Invalid backup relative file identity')
        source = request['experiment'] / name
        target = destination / 'files' / name
        preserved._path(target)
        if (target.is_symlink() or not target.is_file() or os.path.samefile(source, target)
                or _record(source) != expected or _record(target) != expected):
            raise ValueError(f'Backup/current byte identity differs; make a new backup after quiescence: {name}')
    _quiescent(request)
    return receipt


def run(arguments):
    request = preserved.resolve_request(arguments)
    if arguments.action == 'inspect':
        result = inspect(request)
    elif arguments.action == 'backup':
        result = backup(arguments, request)
    elif arguments.action == 'resume':
        verify_backup(arguments, request)
        options = ('--action', 'resume', '--backup', str(arguments.backup))
        # Preserve the existing Linux supervisor and current-environment GPU
        # selector. No resource reservation or scheduler override is introduced.
        with _curriculum_dispatch(arguments, child_entrypoint=__file__, child_options=options):
            result = preserved.run(arguments)
    else:
        raise ValueError('Unknown explicit migration action')
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    run(parse())
