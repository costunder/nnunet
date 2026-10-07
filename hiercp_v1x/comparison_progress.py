"""Execution-only comparison progress; no torch, sampling, or model state."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time


class RunningPatientMetrics:
    """Patient macro on observed sources, including rows restored at resume."""
    keys = ('mrr', 'top1', 'pair_win')

    def __init__(self, rows=()):
        self.cases = defaultdict(lambda: [0, [0.] * len(self.keys)])
        self.sources = set()
        self.add(rows)

    def add(self, rows):
        for row in rows:
            source = row['source_id']
            if source in self.sources:
                raise ValueError('Duplicate observed source in progress metrics')
            values = [float(row[key]) for key in self.keys]
            if not all(math.isfinite(value) for value in values):
                raise ValueError('Nonfinite observed progress metric')
            self.sources.add(source)
            case = self.cases[row['case_id']]
            case[0] += 1
            for index, value in enumerate(values):
                case[1][index] += value

    def metrics(self):
        if not self.sources:
            return {}
        return {key: sum(sums[index] / count for count, sums in self.cases.values()) / len(self.cases)
                for index, key in enumerate(self.keys)}


class PhaseProgress:
    """One compact TTY line or bounded newline heartbeats for redirected logs.

    The heartbeat waits on an Event and touches telemetry only. It is not a
    data worker, never reads tensors, and never consumes a random generator.
    ``progress.json`` is a replaceable live status receipt, not a checkpoint.
    """
    def __init__(self, *, root, arm, phase, epoch, epochs, total, initial,
                 physical_batch, metrics=None, stream=None, interval=15.,
                 heartbeat_interval=1.):
        if total < 1 or not 0 <= initial <= total or interval <= 0 or heartbeat_interval <= 0:
            raise ValueError('Positive progress size/interval and valid resume cursor required')
        self.root = Path(root)
        self.stream = sys.stderr if stream is None else stream
        self.interval = float(interval)
        self.heartbeat_interval = float(heartbeat_interval)
        self.tty = bool(self.stream.isatty())
        self.started = self.stage_started = time.perf_counter()
        self.initial = initial
        self.last_console = self.last_receipt = -math.inf
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.failure = None
        self.bar = None
        self.thread = None
        self.closed = False
        self.data = dict(format='comparison_progress_v1', arm=arm, phase=phase,
            epoch=epoch, epochs=epochs, stage='load', status='RUNNING',
            completed_sources=initial, total_sources=total, physical_batch=physical_batch,
            metrics=dict(metrics or {}), metrics_scope='partial patient macro of observed sources',
            timings=dict(loader_seconds=0., loader_wait_seconds=0., forward_seconds=0., checkpoint_seconds=0.),
            timing_scope='current phase segment in this invocation; excludes prior resumed segments')

    @staticmethod
    def write(message, *, stream=None):
        from tqdm import tqdm
        stream = sys.stderr if stream is None else stream
        tqdm.write(str(message), file=stream)
        stream.flush()

    def __enter__(self):
        if self.tty:
            from tqdm import tqdm
            self.write(f"{self.data['arm']} | * partial patient macro; M=MRR, T1=top1, PW=pair-win", stream=self.stream)
            self.bar = tqdm(total=self.data['total_sources'], initial=self.data['completed_sources'],
                            file=self.stream, bar_format='{desc}', ascii=True, leave=True)
        self._publish(force=True)
        self.thread = threading.Thread(target=self._heartbeat, name='comparison-progress', daemon=True)
        self.thread.start()
        return self

    def _line(self, now):
        data = self.data
        # Eight total candidates = one P plus seven U. Keep the historical
        # machine phase key separate from the unambiguous visible count.
        phase = 'val129' if data['phase'] == 'validation129' else 'train8'
        stage = f"{data['stage']} {now - self.stage_started:.0f}s"
        fraction = data['completed_sources'] / data['total_sources']
        eta = self._eta(now)
        eta_text = '--' if eta is None else f'{eta / 60:.1f}m'
        counts = f"{data['completed_sources']}/{data['total_sources']} {100 * fraction:.0f}%"
        prefix = (f"{data['arm']} e{data['epoch']:02d}/{data['epochs']} {phase} "
                  f"{counts} {stage} ETA={eta_text}")
        metrics = data['metrics']
        scope = 'full' if data['status'] == 'COMPLETE' else 'partial'
        if metrics:
            suffix = (f" {scope} MRR={metrics['mrr']:.3f} T1={metrics['top1']:.3f}"
                      f" PW={metrics['pair_win']:.3f}")
        else:
            suffix = ' partial metrics=pending'
        if 'loss' in data:
            suffix += f" loss={data['loss']:.4f}"
        if self.tty:
            # ASCII only: each character has width one; reserve the last column
            # to prevent terminal auto-wrap, including narrow SSH/MobaXterm panes.
            width = max(1, min(120, shutil.get_terminal_size(fallback=(80, 24)).columns - 1))
            short_stage = {'forward': 'fwd', 'complete': 'done', 'paused': 'pause', 'failed': 'fail'}.get(data['stage'], data['stage'])
            brief = f"{phase} e{data['epoch']:02d} {counts} {short_stage} {now - self.stage_started:.0f}s"
            marker = '' if scope == 'full' else '*'
            number = lambda value: f'{value:.3f}'.lstrip('0')
            compact_metrics = (f" M{marker}={number(metrics['mrr'])} T1={number(metrics['top1'])} PW={number(metrics['pair_win'])}"
                               if metrics else ' M*=-- T1=-- PW=--')
            compact_loss = f" L={data['loss']:.4f}" if 'loss' in data else ''
            # Drop ETA/loss before metrics. At narrow widths keep MRR/top1 and
            # the source cursor/stage instead of silently clipping all scores.
            options = [brief + f' ETA={eta_text}' + compact_metrics + compact_loss,
                       brief + compact_metrics + compact_loss,
                       brief + compact_metrics]
            if metrics:
                options.append(brief + f" M{marker}={number(metrics['mrr'])} T1={number(metrics['top1'])}")
                options.append(f"{counts} {short_stage} M{marker}{number(metrics['mrr'])} "
                               f"T{number(metrics['top1'])} P{number(metrics['pair_win'])}")
                options.append(f"{counts} M{marker}{number(metrics['mrr'])} "
                               f"T{number(metrics['top1'])} P{number(metrics['pair_win'])}")
            for line in options:
                if len(line) <= width:
                    return line
            return options[-1][:width]
        return prefix + suffix

    def _eta(self, now):
        completed = self.data['completed_sources'] - self.initial
        if completed <= 0:
            return None
        return (now - self.started) * (self.data['total_sources'] - self.data['completed_sources']) / completed

    def _publish(self, *, force=False):
        with self.lock:
            now = time.perf_counter()
            if force or now - self.last_console >= (self.heartbeat_interval if self.tty else self.interval):
                line = self._line(now)
                if self.bar is not None:
                    self.bar.n = self.data['completed_sources']
                    self.bar.desc = line
                    self.bar.refresh()
                else:
                    self.write(line, stream=self.stream)
                self.last_console = now
            if force or now - self.last_receipt >= self.interval:
                payload = dict(self.data, elapsed_seconds=now - self.started,
                               stage_elapsed_seconds=now - self.stage_started,
                               percent=100 * self.data['completed_sources'] / self.data['total_sources'],
                               eta_seconds=self._eta(now),
                               updated_at=datetime.now(timezone.utc).isoformat())
                path = self.root / 'progress.json'
                temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
                with temporary.open('w', encoding='utf8') as output:
                    json.dump(payload, output, allow_nan=False)
                    output.write('\n')
                os.replace(temporary, path)
                self.last_receipt = now

    def _heartbeat(self):
        try:
            while not self.stop.wait(self.heartbeat_interval):
                self._publish()
        except Exception as error:
            self.failure = error
            self.stop.set()

    def update(self, *, stage=None, completed=None, metrics=None, timings=None, loss=None):
        if self.failure is not None:
            raise RuntimeError('Comparison progress writer failed') from self.failure
        with self.lock:
            if stage is not None and stage != self.data['stage']:
                self.data['stage'] = stage
                self.stage_started = time.perf_counter()
            if completed is not None:
                if not self.data['completed_sources'] <= completed <= self.data['total_sources']:
                    raise ValueError('Progress source cursor cannot regress or exceed full data')
                self.data['completed_sources'] = completed
            if metrics is not None:
                self.data['metrics'] = dict(metrics)
            if loss is not None:
                self.data['loss'] = float(loss)
            for key, value in (timings or {}).items():
                self.data['timings'][key] += float(value)
            # Publish the phase/stage immediately to the status file without
            # forcing extra newline output into redirected logs.
            self.last_receipt = -math.inf
            self._publish()

    def finish(self, status='COMPLETE'):
        if status not in ('COMPLETE', 'PAUSED', 'FAILED'):
            raise ValueError('Explicit progress completion status required')
        with self.lock:
            self.data['status'] = status
            self.data['stage'] = status.lower()
            self.stage_started = time.perf_counter()
            if status == 'COMPLETE':
                self.data['metrics_scope'] = 'patient macro of observed sources; phase complete'
        self.close()

    def close(self):
        if self.closed:
            return
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        try:
            self._publish(force=True)
        finally:
            if self.bar is not None:
                self.bar.close()
            self.closed = True
        if self.failure is not None:
            raise RuntimeError('Comparison progress writer failed') from self.failure

    def __exit__(self, kind, value, traceback):
        if not self.closed:
            try:
                self.finish('FAILED' if kind is not None else 'PAUSED')
            except Exception as progress_error:
                if value is not None:
                    raise value.with_traceback(traceback) from progress_error
                raise
