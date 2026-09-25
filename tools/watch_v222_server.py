"""Read-only, attachable tqdm console for existing v2.22 server runs.

Self-contained: may be fetched to /tmp while an older checkout keeps running.
Ctrl+C closes only this viewer. It never signals or changes a training process.
"""
import argparse
import ast
from collections import deque
import json
from pathlib import Path
import sys
import time

import psutil
from tqdm import tqdm


class LogReader:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.partial = b''
        self.tail = deque(maxlen=16)

    def read(self):
        if not self.path.exists():
            return
        with self.path.open('rb') as stream:
            if self.path.stat().st_size < self.offset:
                self.offset = 0
                self.partial = b''
            stream.seek(self.offset)
            while chunk := stream.read(65536):
                self.offset = stream.tell()
                lines = (self.partial + chunk).split(b'\n')
                self.partial = lines.pop()
                for raw in lines:
                    line = raw.decode('utf-8', errors='replace').strip()
                    self.tail.append(line)
                    start = line.find('{')
                    if start < 0:
                        continue
                    try:
                        value = json.loads(line[start:])
                    except json.JSONDecodeError:
                        if not line.startswith("{'stage':"):
                            continue  # Ordinary text/tracebacks stay in the log.
                        try:
                            value = ast.literal_eval(line)
                        except (ValueError, SyntaxError):
                            continue
                    if isinstance(value, dict):
                        yield value


class Progress:
    def __init__(self):
        self.stage = 'starting'
        self.phase = 'waiting for first event'
        self.epoch = None
        self.done = 0
        self.total = None
        self.detail = ''

    def phase_to(self, phase):
        if phase != self.phase:
            self.phase, self.done, self.total, self.detail = phase, 0, None, ''

    def event(self, row):
        stage = row.get('stage')
        lifecycle = row.get('event')
        if lifecycle is not None:
            # Runner lifecycle envelopes and child progress share stage names,
            # but only child progress carries per-item completed/total fields.
            if lifecycle == 'stage_started':
                self.stage = stage
                self.phase_to(stage)
            elif lifecycle == 'stage_complete':
                self.stage = stage
                self.detail = 'stage complete; waiting for next stage'
            elif lifecycle == 'stage_failed':
                self.stage = stage
                self.phase_to(f'{stage} FAILED')
                self.detail = 'see failure receipt and log'
            return
        if stage in ('raw_inventory', 'raw_inventory_heartbeat', 'paired_cache', 'support_memory'):
            self.phase_to(stage)
            self.done, self.total = row['completed'], row['total']
        elif stage == 'optimization':
            self.epoch = row['epoch']
            self.phase_to(f'epoch {self.epoch} optimization')
            self.done, self.total = row['visited_queries'], row['total_queries']
            self.detail = f"step {row['step']}/{row['total_steps']} loss={row['loss']:.4f}"
        elif stage == 'paired_cache_complete':
            self.phase_to('paired_cache')
            self.done = self.total = row['observations']
        elif stage and ('calibration' in stage or 'benchmark' in stage or stage == 'preparation_admission'):
            self.phase_to(stage)
            self.detail = ' '.join(f'{k}={row[k]}' for k in ('workers', 'tasks', 'pairs', 'physical_batch') if k in row)
        elif stage in ('paired_donors', 'all_pair_transport_preflight_complete', 'epoch_complete', 'paused'):
            self.phase_to(stage)
            self.detail = ' '.join(f'{k}={row[k]}' for k in ('case', 'donors', 'epoch', 'step') if k in row)
        elif row.get('debug') and 'step' in row and 'allocator_policy' in row:
            self.phase_to('profile DEBUG')
            self.done, self.total = row['step'], 10
        if stage == 'raw_inventory_heartbeat':
            self.detail = 'active=' + ','.join(row['active_cases'])
        elif stage == 'raw_case_started':
            self.detail = 'CT ' + row['case']
        elif stage in ('paired_cache', 'support_memory'):
            self.detail = ' '.join(f'{k}={row[k]}' for k in ('case', 'workers', 'physical_batch') if k in row)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None  # A receipt may currently be being written; retry next refresh.


def terminal_status(root):
    failures = sorted(root.glob('*.failed.json'))
    if failures or (root/'pipeline_failed.json').exists():
        return 'FAILED', failures[-1] if failures else root/'pipeline_failed.json'
    for filename, state in (('pipeline_complete.json', 'COMPLETE (GNN only)'),
                            ('pipeline_paused.json', 'PAUSED')):
        if (root/filename).exists():
            return state, root/filename
    worker = read_json(root/'worker.json')
    if worker:
        try:
            process = psutil.Process(worker['pid'])
            alive = process.create_time() == worker['created_at'] and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            alive = False
        if not alive:
            return 'WORKER ENDED WITHOUT COMPLETION', root/'worker.json'
    return None


def watch(root, log=None, interval=.5, stream=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    log = Path(log) if log else root/'console.log'
    if not log.exists() and log == root/'console.log':
        log = root.parent.parent/(root.name + '.log')  # Previous nohup runner.
    if not log.exists():
        raise FileNotFoundError(f'No log at {log}; provide --log')
    stream = stream or sys.stderr
    if not stream.isatty():
        raise RuntimeError('Run the viewer in a terminal without nohup/redirection; worker logs already go to a file')
    reader, state = LogReader(log), Progress()
    for row in reader.read():
        state.event(row)
    print(f'Live progress | {root.name}\nLog: {log}\nCtrl+C: close viewer only; training continues.', file=stream)
    # Stages have unequal durations. A stage-count ETA/rate is not meaningful,
    # especially when reattaching and replaying completed receipts instantly.
    requested = read_json(root/'requested.json')
    overall = tqdm(total=len(requested['stages']) if requested else 7, desc='Pipeline stages', position=0, dynamic_ncols=True, file=stream,
                   bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}| {postfix}')
    current = tqdm(total=state.total, initial=state.done, desc=state.phase, position=1,
                   dynamic_ncols=True, file=stream, unit='item')
    key = None
    checkpoint_stamp = None
    try:
        while True:
            if requested is None:
                requested = read_json(root/'requested.json')
                if requested is not None:
                    overall.total = len(requested['stages'])
            for row in reader.read():
                state.event(row)
            # Phase receipts cover validation/final-memory transitions without
            # changing the frozen model sources or inventing validation counts.
            saved = read_json(root/'training/checkpoint_status.json')
            if saved and saved['saved_at'] != checkpoint_stamp:
                checkpoint_stamp = saved['saved_at']
                phase = saved['phase']
                if phase in ('validation', 'initial_memory', 'refresh_memory', 'final_memory'):
                    if phase == 'validation':
                        state.phase_to(f"epoch {saved['epoch']} validation")
                    elif state.phase != 'support_memory':
                        state.phase_to(phase)
            newkey = (state.stage, state.phase, state.total, saved['epoch'] if saved else state.epoch)
            if newkey != key or state.done < current.n:
                current.total = state.total
                current.reset(total=state.total)
                # Never derive ETA from replaying historical logs in a millisecond.
                current.n = current.initial = state.done
                current.last_print_n = state.done
                key = newkey
            elif state.done > current.n:
                current.update(state.done-current.n)
            overall.n = len(list(root.glob('[0-9][0-9]_*.complete.json')))
            overall.set_postfix_str(state.stage, refresh=False)
            current.set_description_str(state.phase, refresh=False)
            current.set_postfix_str(state.detail, refresh=False)
            overall.refresh()
            current.refresh()
            result = terminal_status(root)
            if result:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        result = ('Viewer closed; training was not signalled', None)
    finally:
        current.close()
        overall.close()
    print(f'{result[0]} | {result[1] or log}', file=stream)
    if result[0] in ('FAILED', 'WORKER ENDED WITHOUT COMPLETION'):
        print('\n'.join(reader.tail), file=stream)
        return 1
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--latest', action='store_true', help='Attach newest v222_mig10gb_r6* run in ./work')
    p.add_argument('--log', type=Path)
    a = p.parse_args()
    if a.latest:
        candidates = list(Path('work').glob('v222_mig10gb_r6*/requested.json'))
        if not candidates:
            raise FileNotFoundError('No recorded v222_mig10gb_r6 run in ./work')
        a.output = max(candidates, key=lambda x: x.stat().st_mtime).parent
    return watch(a.output, a.log)


if __name__ == '__main__':
    raise SystemExit(main())
