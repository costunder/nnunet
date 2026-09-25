"""Read-only admission guard for a still-active source run. Never sends signals."""
import json
from pathlib import Path
import os
import psutil

TRAINERS = {'run_v222_v1_l0.py', 'run_v222_optimized.py'}


def output_argument(command):
    if not any(Path(token).name in TRAINERS for token in command):
        return None
    for i, token in enumerate(command):
        if token == '--output' and i+1 < len(command):
            return command[i+1]
        if token.startswith('--output='):
            return token.split('=', 1)[1]
    return None


def assert_source_runs_idle(cache=None, resume=None):
    roots = set()
    if resume is not None:
        roots.add(Path(resume).resolve().parent)
    if cache is not None:
        # Server layout: RUN/paired_cache/index.json and RUN/training/.
        path = Path(cache).resolve()
        if (path.parent.parent/'worker.json').is_file():
            roots.add(path.parent.parent/'training')
    blockers = []
    own_pid = os.getpid()
    for training in roots:
        receipt = training.parent/'worker.json'
        if receipt.is_file():
            record = json.loads(receipt.read_text(encoding='utf-8'))
            try:
                process = psutil.Process(record['pid'])
                if (process.pid != own_pid and process.create_time() == record['created_at']
                        and process.status() != psutil.STATUS_ZOMBIE):
                    blockers.append(dict(pid=process.pid,source=str(receipt),kind='source worker'))
            except psutil.NoSuchProcess:
                pass # A completed source worker is expected, not a fallback.
    # A worker can die while its training child survives. Check the exact
    # trainer output too; an old paused marker alone is not liveness evidence.
    if roots:
        username = psutil.Process(own_pid).username()
        for process in psutil.process_iter(['pid', 'name']):
            if process.pid == own_pid or 'python' not in (process.info['name'] or '').lower():
                continue
            try:
                if process.username() != username:
                    continue
                argument = output_argument(process.cmdline())
                if argument is None:
                    continue
                output = Path(argument)
                if not output.is_absolute():
                    output = Path(process.cwd())/output
                if output.resolve() in roots and process.status() != psutil.STATUS_ZOMBIE:
                    blockers.append(dict(pid=process.pid,source=str(output.resolve()),kind='source trainer'))
            except psutil.NoSuchProcess:
                continue # Process finished during this read-only snapshot.
    if blockers:
        raise RuntimeError('Source run is still active; refusing duplicate/resume launch. '
            'No process was stopped and no checkpoint was changed. '
            + json.dumps(blockers, ensure_ascii=False))
    return dict(source_runs_idle=True,checked_training_roots=sorted(map(str,roots)))
