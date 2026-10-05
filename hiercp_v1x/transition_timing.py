"""D checkpoint wall receipts; async phases never sum into epoch wall."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
import time

from .contracts import canonical_hash
from l0_regions.execution_pipeline import CheckpointPipeline


class WallCheckpointPipeline(CheckpointPipeline):
    def _write(self, payload):
        result = super()._write(payload)
        state = payload['state']
        reference = state.get('epoch_wall_checkpoint')
        if reference is None:
            return result
        wall = copy.deepcopy(state['epoch_wall'])
        tail = time.perf_counter() - reference['snapshot_monotonic']
        wall['elapsed_seconds'] += tail
        phases = wall['phase_wall_seconds']
        phases['asynchronous_checkpoint_write'] = phases.get('asynchronous_checkpoint_write', 0.) + tail
        wall['scope'] = 'through_atomic_checkpoint_write_before_its_timing_receipt_write'
        row = dict(format='crossed_D_checkpoint_wall_receipt_v1',
                   checkpoint_id=reference['checkpoint_id'], epoch=state['epoch'],
                   step=state['step'], phase=state['phase'],
                   identity_sha256=canonical_hash(payload['identity']), active_epoch_wall=wall)
        row['content_sha256'] = canonical_hash(row)
        with (self.root / reference['journal_file']).open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        return result


def restore_D_wall(payload, output):
    """Recover only a receipt bound to the saved scientific state/token.

    If interruption prevented its complete receipt, the saved pre-write elapsed
    value is retained and explicitly marked as a lower bound. No downtime is
    added. The timing receipt's own write is included by the next live snapshot.
    """
    state = copy.deepcopy(payload['state'])
    state['wall_resume_scope'] = 'saved_cursor_before_checkpoint_write_lower_bound'
    reference = state.get('epoch_wall_checkpoint')
    if reference is None:
        return state
    name = reference.get('journal_file')
    if (not isinstance(name, str) or not re.fullmatch(r'd_checkpoint_wall_[0-9a-f]{32}\.jsonl', name)
            or not isinstance(reference.get('checkpoint_id'), str)):
        raise ValueError('D wall journal reference changed')
    path = Path(output) / name
    if not path.exists():
        return state
    if path.is_symlink():
        raise ValueError('D wall journal cannot be a symlink')
    for line in path.read_text(encoding='utf8').splitlines(keepends=True):
        if not line.endswith('\n'):
            break
        row = json.loads(line)
        if row.get('checkpoint_id') != reference['checkpoint_id']:
            continue
        if row.get('content_sha256') != canonical_hash({k: v for k, v in row.items() if k != 'content_sha256'}):
            raise ValueError('D wall journal content changed')
        if (row.get('format') != 'crossed_D_checkpoint_wall_receipt_v1'
                or row.get('identity_sha256') != canonical_hash(payload['identity'])
                or any(row.get(key) != state[key] for key in ('epoch', 'step', 'phase'))):
            raise ValueError('D wall journal belongs to another saved scientific state')
        wall = row['active_epoch_wall']
        from .transition_c_training import ActiveEpochWall
        ActiveEpochWall(state['epoch'] + 1, wall)
        old = state['epoch_wall']['elapsed_seconds']
        if not math.isfinite(old) or wall['elapsed_seconds'] < old:
            raise ValueError('D wall receipt moves saved elapsed time backwards')
        state['epoch_wall'] = wall
        state['epoch_seconds'] = wall['elapsed_seconds']
        state['wall_resume_scope'] = wall['scope']
        return state
    return state
