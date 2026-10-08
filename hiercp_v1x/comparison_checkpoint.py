"""Synchronous, wire-compatible checkpoints with owned CPU snapshots.

Every call still writes and fsyncs a complete ordinary ``torch.save`` checkpoint
before publishing it atomically. No checkpoint interval or resume state changes.
An explicit generation may reuse static state only during no-update evaluation;
normal training must pass ``None``. Tensor version/storage witnesses catch normal
in-place changes, but callers must never mutate state through ``.data`` or raw
storage while promising that an evaluation generation is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import torch

from .u_bridge_training import cpu_copy
from .comparison_storage import report_secondary_failure


_DYNAMIC = frozenset(('state', 'rng', 'shuffle_generator'))


def _visit(result, item):
    """The frozen digest's exact byte stream, without a tensor ``tobytes`` copy."""
    if torch.is_tensor(item):
        tensor = item.detach().cpu().contiguous()
        result.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        result.update(tensor.reshape(-1).view(torch.uint8).numpy())
    elif isinstance(item, np.ndarray):
        _visit(result, torch.from_numpy(item.copy()))
    elif isinstance(item, dict):
        result.update(b'dict')
        for key in sorted(item, key=str):
            _visit(result, str(key))
            _visit(result, item[key])
    elif isinstance(item, (tuple, list)):
        result.update(type(item).__name__.encode())
        for child in item:
            _visit(result, child)
    else:
        result.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())


def _witness(item):
    """Cheap live-state mutation witness; no CUDA transfer or synchronization."""
    if torch.is_tensor(item):
        return ('tensor', str(item.device), str(item.dtype), tuple(item.shape),
                tuple(item.stride()), item.storage_offset(),
                item.untyped_storage().data_ptr(), item._version)
    if isinstance(item, np.ndarray):
        return ('numpy', str(item.dtype), item.shape, item.tobytes())
    if isinstance(item, dict):
        return ('dict', tuple((str(key), _witness(item[key]))
                              for key in sorted(item, key=str)))
    if isinstance(item, (tuple, list)):
        return (type(item).__name__, tuple(_witness(child) for child in item))
    return ('scalar', json.dumps(item, sort_keys=True, allow_nan=False))


def _atomic_owned_snapshot(path, snapshot):
    """Publish a private immutable snapshot without cloning it a second time."""
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    created = False
    try:
        with temporary.open('xb') as stream:
            created = True
            torch.save(snapshot, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException as error:
        # Only this call's exclusively-created partial file is eligible. Preserve
        # the previously committed checkpoint and propagate the original failure.
        if created:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                report_secondary_failure(error, cleanup_error,
                    operation=f'remove own partial checkpoint {temporary}')
        raise


class ComparisonCheckpointWriter:
    """Own CPU state and hash prefixes across explicitly immutable evaluations.

    ``payload`` has the original checkpoint fields, including any arm policy,
    without ``content_sha256``. The caller may supply live state dictionaries:
    this writer detaches and clones every captured tensor before returning.
    Reusing a generation requires the same model/optimizer/scaler/scheduler state.
    A changed witness refreshes the snapshot even when the caller's token matches.
    """

    def __init__(self):
        self._generation = None
        self._signature = None
        self._static = None
        self._prefix = None
        self._prefix_keys = ()
        self._lock = threading.Lock()

    def invalidate(self):
        with self._lock:
            self._generation = self._signature = self._static = self._prefix = None
            self._prefix_keys = ()

    def save(self, paths, payload, *, static_generation=None):
        """Write every requested full checkpoint synchronously; return timings."""
        paths = tuple(Path(path) for path in paths)
        if not paths:
            raise ValueError('At least one checkpoint destination is required')
        if len(set(paths)) != len(paths):
            raise ValueError('Checkpoint destinations must be distinct')
        if 'content_sha256' in payload:
            raise ValueError('Writer computes content_sha256 from the complete original payload')
        if not _DYNAMIC.issubset(payload):
            raise ValueError('Checkpoint requires state, rng, and shuffle_generator')
        with self._lock:
            began = time.perf_counter()
            static = {key: value for key, value in payload.items() if key not in _DYNAMIC}
            signature = _witness(static) if static_generation is not None else None
            reused = (static_generation is not None and self._static is not None
                      and static_generation == self._generation and signature == self._signature)
            if not reused:
                self._static = cpu_copy(static)
                self._generation = static_generation
                self._signature = signature
                self._prefix = None
            snapshot = dict(self._static)
            snapshot.update({key: cpu_copy(payload[key]) for key in _DYNAMIC})
            snapshot_seconds = time.perf_counter() - began
            hashing = time.perf_counter()
            keys = tuple(sorted(snapshot, key=str))
            prefix_keys = tuple(key for key in keys if str(key) < min(_DYNAMIC))
            if self._prefix is None or prefix_keys != self._prefix_keys:
                self._prefix = hashlib.sha256()
                self._prefix.update(b'dict')
                for key in prefix_keys:
                    _visit(self._prefix, str(key))
                    _visit(self._prefix, snapshot[key])
                self._prefix_keys = prefix_keys
            result = self._prefix.copy()
            for key in keys[len(prefix_keys):]:
                _visit(result, str(key))
                _visit(result, snapshot[key])
            snapshot['content_sha256'] = result.hexdigest()
            hash_seconds = time.perf_counter() - hashing
            writing = time.perf_counter()
            for path in paths:
                _atomic_owned_snapshot(path, snapshot)
            save_seconds = time.perf_counter() - writing
            return dict(static_snapshot_reused=reused, snapshot_seconds=snapshot_seconds,
                        hash_seconds=hash_seconds, save_seconds=save_seconds,
                        total_seconds=time.perf_counter() - began, paths=len(paths),
                        content_sha256=snapshot['content_sha256'])
