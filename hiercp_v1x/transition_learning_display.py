"""Read existing D learning journals without loading checkpoints or models."""
from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path


def number(row, key):
    value = row.get(key)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Missing/nonfinite numeric learning record: ' + key)
    return value


def integer(row, key):
    value = row.get(key)
    if type(value) is not int or value < 0:
        raise ValueError('Missing/invalid learning cursor: ' + key)
    return value


class JsonlTail:
    """Read every complete appended row once; an unfinished final row is pending."""
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.line = 0
        self.pending = False
        self.identity = None

    def read(self):
        self.pending = False
        if not self.path.exists():
            if self.identity is not None:
                raise ValueError('Previously read learning journal disappeared: ' + str(self.path))
            return []
        stat = self.path.stat()
        identity = stat.st_dev, stat.st_ino
        if self.identity is not None and (identity != self.identity or stat.st_size < self.offset):
            raise ValueError('Learning journal replaced or truncated: ' + str(self.path))
        self.identity = identity
        rows = []
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                raw = stream.readline()
                if not raw:
                    break
                if not raw.endswith(b'\n'):
                    self.pending = True
                    break
                self.offset = stream.tell()
                self.line += 1
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw.decode('utf-8'))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(f'Invalid complete learning journal row: {self.path}:{self.line}') from error
                if not isinstance(value, dict):
                    raise ValueError(f'Learning journal row is not an object: {self.path}:{self.line}')
                rows.append(value)
        return rows


def validation_line(row, *, best_epoch=None):
    epoch = integer(row, 'epoch')
    for key in ('case_first_P_mrr', 'case_hit_at_1', 'observed_micro_recall_at_1', 'P_U_pair_win_rate'):
        if not 0 <= number(row, key) <= 1:
            raise ValueError('Out-of-range validation metric: ' + key)
    loss = number(row, 'P_U_softplus_loss')
    if loss < 0:
        raise ValueError('Negative P/U softplus loss')
    label = 'INITIAL' if epoch == 0 else f'epoch {epoch:02d}/40'
    best = ' NEW_BEST' if row.get('new_best') is True else ''
    text = (f"D {label} | MRR={row['case_first_P_mrr']:.6f}"
            f" Hit@1={row['case_hit_at_1']:.6f} R@1={row['observed_micro_recall_at_1']:.6f}"
            f" pair-win={row['P_U_pair_win_rate']:.6f} P/U_loss={loss:.6f}{best}")
    if best_epoch is not None:
        text += f' | selected BEST epoch={best_epoch}'
    if 'epoch_wall_seconds' in row:
        text += f" | wall={number(row, 'epoch_wall_seconds') / 60:.2f}min"
    return text


class DLearningDisplay:
    """Full saved validation history plus a display-only recent20 update window."""
    def __init__(self, experiment):
        self.root = Path(experiment).resolve(strict=True)
        self.training = self.root / 'training'
        if not self.training.is_dir():
            raise ValueError('D experiment/training directory is missing: ' + str(self.training))
        self.curve = JsonlTail(self.training / 'curve.jsonl')
        self.updates = JsonlTail(self.training / 'updates.jsonl')
        self.history = {}
        self.best_epoch = None
        self.recent = deque(maxlen=20)  # Display window only, never a training/data cap.
        self.last_step = None

    def scope_line(self):
        path = self.root / 'manifest.json'
        if not path.is_file():
            return 'D run scope: UNKNOWN (manifest absent); displayed journal metrics are not a newly verified evaluation.'
        value = json.loads(path.read_text(encoding='utf8'))
        if not isinstance(value, dict) or value.get('arm') != 'D':
            raise ValueError('Requested experiment manifest is not D')
        binding = value.get('original_curriculum_binding', {})
        cases = binding.get('validation_case_ids')
        count = len(cases) if isinstance(cases, list) else 'UNKNOWN'
        return f"D saved run scope | debug={value.get('debug', 'UNKNOWN')} | declared validation cases={count} | P+128U"

    def poll(self):
        lines = []
        for row in self.curve.read():
            if row.get('stage') != 'validation':
                raise ValueError('Unexpected D curve stage')
            epoch = integer(row, 'epoch')
            if epoch > 40 or epoch in self.history or self.history and epoch <= max(self.history):
                raise ValueError('Duplicate/backwards/out-of-range D validation epoch')
            if epoch > 0 and type(row.get('new_best')) is not bool:
                raise ValueError('Missing actual D BEST selection flag')
            previous = max(self.history) if self.history else -1
            if epoch != previous + 1:
                lines.append(f'WARNING: saved validation epochs {previous + 1}..{epoch - 1} are missing; no metrics fabricated.')
            if row.get('new_best') is True:
                self.best_epoch = epoch
            lines.append(validation_line(row, best_epoch=self.best_epoch))
            self.history[epoch] = row
        changed = False
        for row in self.updates.read():
            step, epoch = integer(row, 'step'), integer(row, 'epoch')
            if step < 1 or not 1 <= epoch <= 40 or self.last_step is not None and step <= self.last_step:
                raise ValueError('Duplicate/backwards/invalid D update cursor')
            number(row, 'loss')
            if self.recent and epoch != self.recent[-1]['epoch']:
                self.recent.clear()
            self.recent.append(row)
            self.last_step = step
            changed = True
        return lines, changed

    def progress_line(self):
        if not self.recent:
            return 'D latest update: NOT_YET_RECORDED'
        row = self.recent[-1]
        count = len(self.recent)
        average = math.fsum(number(item, 'loss') for item in self.recent) / count
        text = (f"D update | epoch={row['epoch']}/40 step={row['step']}"
                f" loss={row['loss']:.6f} avg{count}={average:.6f}"
                f" | latest completed validation={max(self.history) if self.history else 'UNKNOWN'}")
        terms = row.get('terms', {})
        for name, key in (('rank', 'ranking_loss'), ('CE', 'observation_ce'), ('align', 'alignment')):
            if key in terms:
                text += f' {name}={number(terms, key):.5f}'
        if 'gradient_clip_norm' in row:
            text += f" grad={number(row, 'gradient_clip_norm'):.4g}"
        gradients = row.get('gradients', {})
        names = ('L0', 'L1', 'L2', 'L2_updates')
        if all(name in gradients for name in names):
            text += ' | module_grad=' + '/'.join(f'{number(gradients, name):.3g}' for name in names)
        if 'seconds' in row:
            text += f" sec={number(row, 'seconds'):.2f}"
        if 'peak_bytes' in row:
            text += f" peak_GiB={number(row, 'peak_bytes') / 2**30:.2f}"
        return text

    def delta_line(self):
        if 0 not in self.history or max(self.history) == 0:
            return 'D learning change: completed initial/epoch validation pair is not yet available.'
        initial, latest = self.history[0], self.history[max(self.history)]
        text = 'D INITIAL -> LATEST | '
        text += ' | '.join(f"{label} {number(initial, key):.6f} -> {number(latest, key):.6f}"
                          f" (delta={number(latest, key) - number(initial, key):+.6f})"
                          for label, key in (('MRR', 'case_first_P_mrr'), ('Hit@1', 'case_hit_at_1'),
                                             ('pair-win', 'P_U_pair_win_rate')))
        return text

    def completion(self):
        path = self.training / 'training_complete.json'
        if not path.is_file():
            return None
        # The existing writer publishes JSON atomically; malformed complete JSON
        # is an error, not a made-up completion state.
        value = json.loads(path.read_text(encoding='utf8'))
        if value.get('arm') != 'D' or value.get('completed_epochs') != 40 or value.get('full_training') is not True:
            raise ValueError('D completion receipt is not the full40 production result')
        return value
