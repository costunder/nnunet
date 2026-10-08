"""Execution-only caching of compact outputs from the verified original helpers."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from .preparation_reuse import _publication_guard, _rename_new
from .u_bridge_upper import _HELPER_LOCK, _binding


FORMAT = 'comparison_compact_original_upper_v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(kind, binding):
    binding = _binding(binding)
    if kind == 'liver_raw':
        # This original helper depends on the recipient case, not its selected source.
        names = ('case_id', 'image_sha256', 'label_sha256', 'original_core_sha256',
                 'region_identity_sha256', 'shape', 'spacing', 'ct_clip',
                 'tumor_label', 'upper_raw_dim')
        binding = {name: binding[name] for name in names}
    elif kind != 'source_axis':
        raise ValueError('Unknown compact original upper helper')
    return dict(format=FORMAT, kind=kind, binding=binding)


def _arrays(kind, result, identity):
    if kind == 'liver_raw':
        arrays = (result,)
    else:
        if (not isinstance(result, tuple) or len(result) != 2
                or type(result[1]) is not float):
            raise ValueError('Original source axis must be an array and Python float')
        arrays = (result[0], np.asarray(result[1], dtype=np.float64))
    return _validate_arrays(kind, arrays, identity)


def _validate_arrays(kind, arrays, identity):
    shapes = ((identity['binding']['upper_raw_dim'],),) if kind == 'liver_raw' else ((3,), ())
    dtypes = (np.dtype('float32'),) if kind == 'liver_raw' else (np.dtype('float32'), np.dtype('float64'))
    if (len(arrays) != len(shapes) or any(not isinstance(a, np.ndarray)
            or a.shape != shape or a.dtype != dtype or not np.isfinite(a).all()
            for a, shape, dtype in zip(arrays, shapes, dtypes))):
        raise ValueError('Original compact upper helper shape/dtype/finite contract differs')
    return arrays


def _result(kind, arrays):
    if kind == 'liver_raw':
        return arrays[0].copy(order='K')
    return arrays[0].copy(order='K'), float(arrays[1])


class CompactUpperCache:
    """Persist tiny original results; the provider owns LRU memory and budgets."""
    def __init__(self, root, get, check):
        self.root = Path(root)
        if self.root.is_symlink():
            raise ValueError('Compact upper namespace cannot be a symlink')
        if not callable(get) or not callable(check):
            raise TypeError('Original resident-cache and budget callbacks required')
        self.get, self.check = get, check
        self._lock = threading.RLock()
        self._stats = {kind: dict(calls=0, builds=0, reopens=0, resident_hits=0,
                                 original_seconds=0., lookup_seconds=0.)
                       for kind in ('liver_raw', 'source_axis')}

    def obtain(self, kind, binding, factory):
        identity = _identity(kind, binding)
        key = _digest(identity)
        directory = self.root / kind / key
        began = time.perf_counter()
        loaded = False
        self.check()

        def load_or_build():
            nonlocal loaded
            loaded = True
            with _publication_guard(directory):
                if directory.is_symlink():
                    raise ValueError('Compact upper entry cannot be a symlink')
                if directory.exists():
                    metadata_path, payload = directory / 'metadata.json', directory / 'arrays.npz'
                    if (not directory.is_dir() or {p.name for p in directory.iterdir()}
                            != {'metadata.json', 'arrays.npz'} or metadata_path.is_symlink()
                            or payload.is_symlink() or not metadata_path.is_file() or not payload.is_file()
                            or not 0 < metadata_path.stat().st_size <= 16384
                            or not 0 < payload.stat().st_size <= 65536):
                        raise ValueError('Incomplete or oversized compact upper publication')
                    metadata = json.loads(metadata_path.read_text(encoding='utf8'))
                    signed = dict(metadata); checksum = signed.pop('metadata_sha256', None)
                    if (checksum != _digest(signed) or metadata.get('identity') != identity
                            or metadata.get('payload_bytes') != payload.stat().st_size
                            or metadata.get('payload_sha256') != _sha(payload)):
                        raise ValueError('Compact upper binding/payload SHA256 differs')
                    with np.load(payload, allow_pickle=False) as saved:
                        names = ['a0'] if kind == 'liver_raw' else ['a0', 'a1']
                        if sorted(saved.files) != names:
                            raise ValueError('Compact upper array inventory differs')
                        arrays = tuple(saved[name] for name in names)
                    arrays = _validate_arrays(kind, arrays, identity)
                    status = 'reopens'
                else:
                    compute = time.perf_counter()
                    arrays = _arrays(kind, factory(), identity)
                    arrays = tuple(a.copy(order='K') for a in arrays)
                    original_seconds = time.perf_counter() - compute
                    self.check()
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    attempt = directory.with_name('.' + key + '.attempt.' + uuid.uuid4().hex)
                    attempt.mkdir(exist_ok=False)
                    payload = attempt / 'arrays.npz'
                    with payload.open('xb') as stream:
                        np.savez(stream, **{f'a{i}': a for i, a in enumerate(arrays)})
                    metadata = dict(identity=identity, payload_bytes=payload.stat().st_size,
                                    payload_sha256=_sha(payload))
                    metadata['metadata_sha256'] = _digest(metadata)
                    with (attempt / 'metadata.json').open('x', encoding='utf8') as stream:
                        json.dump(metadata, stream, allow_nan=False)
                    _rename_new(attempt, directory)
                    with self._lock:
                        self._stats[kind]['original_seconds'] += original_seconds
                    status = 'builds'
                for array in arrays:
                    array.setflags(write=False)
                self.check()
                with self._lock:
                    self._stats[kind][status] += 1
                return arrays

        arrays = self.get(('compact_upper', kind, key), load_or_build)
        result = _result(kind, arrays)
        self.check()
        with self._lock:
            self._stats[kind]['calls'] += 1
            self._stats[kind]['resident_hits'] += int(not loaded)
            self._stats[kind]['lookup_seconds'] += time.perf_counter() - began
        return result

    def report(self):
        with self._lock:
            return dict(format=FORMAT, kinds=copy.deepcopy(self._stats),
                        memory_ownership='original provider resident LRU and hard budget',
                        original_formulas_preserved=True)


def compact_upper_provider(original):
    """Memoize original liver/axis outputs inside the existing serialized context."""
    class CompactUpperProvider(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._compact_upper_cache = CompactUpperCache(
                self.root / 'compact_upper_v1', self._get, self._check_budget)

        @contextmanager
        def _upper_context(self, example, case, source, regions):
            # Enter the original context before binding anything: it supplies
            # verified identities and its existing source/lesion helper wrappers.
            with _HELPER_LOCK, super()._upper_context(example, case, source, regions):
                binding = _binding(self._upper_bindings[example['id']])
                hierarchy = self._runtime().hierarchy
                liver_original, axis_original = hierarchy._liver_raw, hierarchy._principal_axis

                def liver(actual_case, actual_regions, *, tumor_label, ct_clip):
                    if (actual_case is not case or actual_regions is not regions
                            or tumor_label != binding['tumor_label']
                            or list(map(float, ct_clip)) != binding['ct_clip']):
                        raise ValueError('Compact liver helper requested another verified case/region/config')
                    return self._compact_upper_cache.obtain('liver_raw', binding,
                        lambda: liver_original(actual_case, actual_regions,
                                               tumor_label=tumor_label, ct_clip=ct_clip))

                def axis(mask, spacing):
                    if mask is not source.full_mask:
                        return axis_original(mask, spacing)
                    if (not isinstance(mask, np.ndarray) or mask.dtype != np.dtype(bool)
                            or mask.shape != tuple(binding['shape'])
                            or list(map(float, spacing)) != binding['spacing']):
                        raise ValueError('Compact source axis mask/spacing binding differs')
                    return self._compact_upper_cache.obtain('source_axis', binding,
                        lambda: axis_original(mask, spacing))

                hierarchy._liver_raw, hierarchy._principal_axis = liver, axis
                try:
                    yield
                finally:
                    hierarchy._liver_raw, hierarchy._principal_axis = liver_original, axis_original

        def report(self):
            value = super().report()
            value['compact_upper'] = self._compact_upper_cache.report()
            return value

    CompactUpperProvider.__name__ = 'CompactUpper' + original.__name__
    return CompactUpperProvider
