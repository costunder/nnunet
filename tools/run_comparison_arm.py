"""Run one preserved comparison arm with its own output and bounded stopping."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import signal
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
ARMS = ('selected', 'native', 'native_fixed', 'native_listwise')
DEFAULT_BASE = Path('/home/aicompetition06/Medical/experiments')


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--experiments-dir', type=Path, default=DEFAULT_BASE)
    p.add_argument('--experiment', type=Path)
    p.add_argument('--source-experiment', type=Path)
    p.add_argument('--inventory', type=Path)
    p.add_argument('--cache-sources', type=Path, nargs='+')
    p.add_argument('--owned-child', action='store_true', help=argparse.SUPPRESS)
    for name in ('fixture', 'config', 'source', 'bank'):
        p.add_argument('--debug-' + name, type=Path,
                       help='Original inputs only for an already sealed actual-CT DEBUG run')
    return p.parse_args(argv)


def _path(value):
    path = Path(value).absolute()
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError(f'Arm ownership cannot traverse a symlink: {ancestor}')
    return path.resolve()


def _read(path):
    with path.open(encoding='utf8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'Arm ownership must be a mapping: {path}')
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def resolve_request(a, environ=None):
    """Only explicit CLI options route an arm; stale shell variables cannot.

    This reads routing metadata, never a checkpoint or GPU. The owned child
    performs all original source, tensor and resume checks before training.
    """
    if a.arm not in ARMS or type(a.gpu) is not int or a.gpu < 0:
        raise ValueError('One explicit arm and nonnegative physical GPU number required')
    base = _path(a.experiments_dir)
    source = _path(a.source_experiment or base/'v18_u_bridge_m10_seed42')
    name = (f'v18_{a.arm}_m10_seed42_memory' if a.arm in ('selected', 'native')
            else f'v19_{a.arm}_m10_seed42')
    root = _path(a.experiment or base/name)
    if root == source or root in source.parents or source in root.parents:
        raise ValueError('Independent arm output must be separate from the shared source experiment')
    inventory = _path(a.inventory or base/'v22_cnn_m10_seed42/inventory/index.json')
    manifest_path = root/'experiment.json'
    continuation_path = root/'continuation.json'
    requires_clone = not continuation_path.exists() and a.arm in ('selected', 'native')
    if manifest_path.exists():
        manifest = _read(manifest_path)
        if manifest.get('sha256') != _digest({k:v for k,v in manifest.items() if k != 'sha256'}):
            raise ValueError('Saved experiment contract changed')
        expected_format = ('v18_u_bridge_matched_experiment_v1' if a.arm in ('selected', 'native')
                           else 'v19_matched_comparison_controls_v1')
        if manifest.get('format') != expected_format:
            raise ValueError('Requested arm does not belong to this preserved experiment')
        if not manifest.get('debug') and manifest.get('epochs') != 40:
            raise ValueError('Preserved forty-epoch production contract required')
        if continuation_path.exists():
            request = _read(continuation_path)['request']
            if (request.get('arm') != a.arm or request.get('destination_root') != str(root)
                    or request.get('contract_sha256') != manifest['sha256']
                    or request.get('data_root') != str(root/'data')):
                raise ValueError('Independent continuation belongs to another arm or namespace')
            saved_source = _path(request['source_root'])
            if a.source_experiment is not None and saved_source != source:
                raise ValueError('Explicit continuation source differs from its preserved receipt')
            source = saved_source
        elif a.arm not in ('selected', 'native'):
            if _path(manifest.get('prepared_data_root', root/'data')) != root/'data':
                raise ValueError('Concurrent arms require their own data namespace; shared data was not changed')
        for other in ARMS:
            if other != a.arm and (root/other).exists():
                raise ValueError(f'This output also contains {other}; do not share mutable arm output')
    elif a.arm not in ('selected', 'native'):
        raise FileNotFoundError(f'Existing {a.arm} experiment required; no fresh-training fallback: {root}')
    if requires_clone and not (source/'experiment.json').is_file():
        raise FileNotFoundError(f'Preserved selected/native source experiment required: {source}')
    defaults = [base/'v18_u_bridge_m10_seed42/data',
                base/'v18_selected_m10_seed42_memory/data',
                base/'v18_native_m10_seed42_memory/data',
                base/'v19_native_fixed_m10_seed42/data',
                base/'v19_native_listwise_m10_seed42/data']
    cache_sources = list(dict.fromkeys(_path(p) for p in (a.cache_sources or defaults)))
    return dict(arm=a.arm, gpu=a.gpu, experiment=root, source_experiment=source,
                inventory=inventory, data_root=root/'data', checkpoint=root/a.arm/'checkpoint_latest.pt',
                cache_sources=cache_sources, requires_clone=requires_clone)


def _active_owner(request):
    """Report an existing matching arm; never remove a live lock or signal it."""
    import psutil
    lock = request['experiment']/'.pipeline.lock'
    if not lock.is_file():
        return None
    owner = _read(lock)
    if owner.get('host') != socket.gethostname():
        raise RuntimeError(f'Arm lock belongs to another host: {lock}; {owner}')
    pid = owner.get('pid')
    if type(pid) is not int or pid <= 0:
        raise ValueError(f'Malformed arm lock owner: {lock}')
    try:
        process = psutil.Process(pid)
        created = process.create_time()
        args = process.cmdline()
        executable = Path(process.exe()).name.lower()
        same_user = (process.uids().real == os.getuid() if hasattr(os, 'getuid')
                     else process.username() == psutil.Process().username())
    except psutil.NoSuchProcess:
        return None  # The unchanged lock helper preserves and recovers stale locks.
    def option(name, value):
        return any(args[i] == name and args[i+1] == value for i in range(len(args)-1))
    known = {'run_v18_u_bridge.py', 'run_v19_comparison.py', 'run_v18_independent.py',
             'resume_comparison_cached.py', 'run_comparison_arm.py'}
    if (not same_user or not executable.startswith('python')
            or not any(Path(x).name in known for x in args)
            or not option('--experiment', str(request['experiment']))
            or not option('--arm', request['arm']) or process.create_time() != created
            or _read(lock) != owner):
        raise RuntimeError(f'Live lock PID is not the verified requested arm; preserved: {lock}; PID={pid}; command={args}')
    return dict(status='ALREADY_RUNNING', arm=request['arm'], pid=pid, create_time=created,
                command=args, experiment=str(request['experiment']),
                checkpoint=str(request['checkpoint']), duplicate_started=False)


def _child(a, request):
    from hiercp_v1x.owned_continuation import independent_continuation_execution
    from hiercp_v1x.u_bridge_experiment import lock
    from tools import resume_comparison_cached
    if request['requires_clone']:
        from hiercp_v1x.u_bridge_continuation import prepare_continuation
        with lock(request['experiment']/'.pipeline.lock'):
            print(f"Preserve {a.arm} progress in its own namespace: {request['experiment']}", flush=True)
            prepare_continuation(request['source_experiment'], request['experiment'], a.arm)
    supplied = SimpleNamespace(gpu=a.gpu, arm=a.arm, experiment=request['experiment'],
        inventory=request['inventory'], cache_sources=request['cache_sources'],
        **{'debug_'+name:getattr(a, 'debug_'+name) for name in ('fixture', 'config', 'source', 'bank')})
    with independent_continuation_execution():
        return resume_comparison_cached.run(supplied)


def run(a):
    request = resolve_request(a)
    if a.owned_child:
        return _child(a, request)
    active = _active_owner(request)
    if active is not None:
        print(json.dumps(active, allow_nan=False), flush=True)
        print('This arm already runs in its own folder. Start other arms with their own --arm and --gpu.', flush=True)
        return active
    command = [sys.executable, '-B', '-u', str(ROOT/'tools/run_comparison_arm.py'),
               '--owned-child', '--gpu', str(a.gpu), '--arm', a.arm,
               '--experiment', str(request['experiment']), '--source-experiment', str(request['source_experiment']),
               '--experiments-dir', str(a.experiments_dir), '--inventory', str(request['inventory']),
               '--cache-sources', *(str(p) for p in request['cache_sources'])]
    for name in ('fixture', 'config', 'source', 'bank'):
        value = getattr(a, 'debug_'+name)
        if value is not None:
            command.extend(('--debug-'+name, str(value)))
    print(json.dumps(dict(stage='independent_arm_launch', arm=a.arm, physical_GPU=a.gpu,
        experiment=str(request['experiment']), data=str(request['data_root']),
        checkpoint=str(request['checkpoint']), preserve_saved_model_Adam_RNG_cursor=True,
        preserve_measured_batch_workers_and_40_epochs=True, shared_mutable_arm_namespace=False,
        stop_grace_seconds=10, stop_recovery='last published atomic checkpoint; unsaved work replayed'),
        allow_nan=False), flush=True)
    from hiercp_v1x.arm_process import run_owned
    result = run_owned(command, checkpoint=request['checkpoint'], grace_seconds=10, cwd=ROOT)
    print('ARM PROCESS RESULT: '+json.dumps(result, allow_nan=False), flush=True)
    expected_pause = result['paused'] and result['child_returncode'] in (0, -signal.SIGINT, -signal.SIGTERM)
    if result['child_returncode'] and not expected_pause:
        raise RuntimeError(f"{a.arm} child failed ({result['child_returncode']}); outputs preserved: {request['experiment']}")
    return result


if __name__ == '__main__':
    run(parse())
