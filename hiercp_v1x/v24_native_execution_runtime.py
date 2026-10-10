"""Reuse one event's identical lossless segmentation crop during native CP.

The unchanged loader first compares the bound baseline segmentation with the
actual nnUNet patch. The unchanged raw engine then decodes the same baseline
crop again. This execution-only runtime keeps that first decoded crop until
the event finishes. Every cache hit still performs the original file/stat
verification and returns an independent read-only array. Nothing survives
the event, and no CT, source mask, resampling operator or model is cached.
"""
from __future__ import annotations

from contextvars import ContextVar
import copy
import hashlib
import importlib
import inspect
import marshal
from pathlib import Path
import time

import numpy as np
import torch

from . import v24_lossless_raw_storage as storage
from . import v24_native_crop_runtime as crop

FORMAT = 'v24_native_CP_exact_event_segmentation_crop_reuse_v1'
_BASE_GETITEM = storage._CropArray.__getitem__
_ACTIVE = ContextVar('v24_native_CP_segmentation_crop_scope', default=None)
_SCIENTIFIC_FILES = ('custom_trainers/nnUNetTrainer_OnlinePairedCP.py',
                     'custom_trainers/onlinecp_raw_resampling.py',
                     'custom_trainers/nnUNetTrainer_FrozenV23CP.py',
                     'hiercp_v1x/v24_lossless_raw_storage.py',
                     'hiercp_v1x/v24_native_crop_runtime.py')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stat(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def runtime_contract():
    return dict(format=FORMAT, runtime_source_sha256=_sha(__file__),
        scientific_source_sha256={name:_sha(crop.ROOT/name) for name in _SCIENTIFIC_FILES},
        original_crop_getitem_source_sha256=hashlib.sha256(inspect.getsource(_BASE_GETITEM).encode()).hexdigest(),
        reused_role='only exact bound baseline_seg lossless int16 proxy',
        cache_scope='one unchanged native raw CP paste call; finally releases all arrays',
        key='proxy object identity and exact basic index; file/stat verified on every access',
        independent_readonly_returned_arrays=True, CT_or_source_or_model_output_cache=False,
        raw_resampling_engine_changed=False, actual_segmentation_comparison_preserved=True,
        candidate_geometry_source_mask_jitter_and_RNG_unchanged=True,
        physical_batch_patch_model_loss_optimizer_or_dataset_changed=False,
        producer_processes_epoch_seeds_queue_order_and_warmup_unchanged=True,
        timing_kind='CPU wall including cache snapshot/copy and repeated original stat verification')


def _source_guard(proofs):
    for path, expected in proofs:
        if _stat(path) != expected:
            raise ValueError('Admitted native execution source changed: '+str(path))


def _basic_index(index, dimensions):
    """Conservative exact keys, with invalid inputs delegated to original code."""
    if not isinstance(index, tuple):
        index = (index,)
    if len(index) > dimensions:
        return None
    key = []
    for item in index:
        if isinstance(item, (int, np.integer)):
            key.append(('int', int(item)))
        elif isinstance(item, slice) and item.step in (None, 1):
            if any(value is not None and not isinstance(value, (int, np.integer))
                   for value in (item.start, item.stop, item.step)):
                return None
            key.append(('slice', *(None if value is None else int(value)
                                   for value in (item.start, item.stop, item.step))))
        else:
            # No broader ndarray indexing rules are admitted by this runtime.
            return None
    key.extend([('slice', None, None, None)]*(dimensions-len(index)))
    return tuple(key)


def _cached_getitem(self, index):
    state = _ACTIVE.get()
    if state is None or self is not state['proxy'] or type(self) is not storage._CropArray:
        return _BASE_GETITEM(self, index)
    _source_guard(state['source_proofs'])
    spec_signature = tuple(self._spec.get(name) for name in ('path','sha256','storage','dtype')) + (tuple(self._spec.get('shape', ())),)
    if spec_signature != state['spec_signature']:
        raise ValueError('Bound native segmentation crop descriptor changed within event')
    key = _basic_index(index, self.ndim)
    if key is None:
        return _BASE_GETITEM(self, index)
    stats = state['stats']; stats['crop_accesses'] += 1
    started = time.perf_counter()
    if key not in state['cache']:
        result = _BASE_GETITEM(self, index)
        stats['original_decode_seconds'] += time.perf_counter()-started
        stats['original_decodes'] += 1
        copied = time.perf_counter()
        snapshot = np.array(result, copy=True, order='K')
        snapshot.flags.writeable = False
        state['cache'][key] = snapshot
        stats['snapshot_copy_seconds'] += time.perf_counter()-copied
        stats['snapshot_bytes'] += snapshot.nbytes
        stats['peak_live_crop_bytes'] = max(stats['peak_live_crop_bytes'],
                                          sum(value.nbytes for value in state['cache'].values()))
        return result
    # Original getter checks closed owner and immutable payload on every call.
    # Preserve those checks on hits rather than relying on the first decode.
    if self._closed:
        raise ValueError('Lossless crop array is closed')
    owner = self._store()
    if owner is None:
        raise ValueError('Lossless crop storage owner is unavailable')
    owner._check(self._spec['path'], self._spec['sha256'])
    stats['hit_original_stat_seconds'] += time.perf_counter()-started
    copied = time.perf_counter()
    result = np.array(state['cache'][key], copy=True, order='K')
    result.flags.writeable = False
    stats['hit_return_copy_seconds'] += time.perf_counter()-copied
    stats['crop_hits'] += 1
    stats['hit_return_bytes'] += result.nbytes
    return result


def _empty_stats():
    return dict(events=0, completed_events=0, failed_events=0, crop_accesses=0,
        original_decodes=0, crop_hits=0, snapshot_bytes=0, hit_return_bytes=0,
        original_decode_seconds=0., snapshot_copy_seconds=0., hit_original_stat_seconds=0.,
        hit_return_copy_seconds=0., event_wall_seconds=0., peak_live_crop_bytes=0,
        live_crop_bytes=0, closed_event_scopes=0)


def wrap_paste_method(original, *, source_proofs=None):
    """Wrap the admitted loader call; the algorithm itself runs unchanged."""
    if source_proofs is None:
        files = (Path(__file__).resolve(), *(crop.ROOT/name for name in _SCIENTIFIC_FILES))
        source_proofs = tuple((path, _stat(path)) for path in files)
    else:
        source_proofs = tuple(source_proofs)
    # The admitted AST clone has a synthetic source filename, so inspect cannot
    # locate its source. Its immutable code is recorded separately from the
    # original file source guards; no guessed source reconstruction is used.
    original_sha = hashlib.sha256(marshal.dumps(original.__code__)).hexdigest()
    def apply(self, data_cropped, seg_cropped, bbox_lbs, plan, case_id):
        _source_guard(source_proofs)
        proxy = plan.get('raw_case', {}).get('baseline_seg')
        if type(proxy) is not storage._CropArray or not crop._admitted_reference(proxy, storage):
            return original(self, data_cropped, seg_cropped, bbox_lbs, plan, case_id)
        if _ACTIVE.get() is not None:
            raise ValueError('Nested native CP crop scopes are not admitted')
        if storage._CropArray.__getitem__ is not _cached_getitem:
            raise ValueError('Native crop execution runtime must be installed before paste')
        stats = _empty_stats(); stats['events'] = 1
        signature = tuple(proxy._spec.get(name) for name in ('path','sha256','storage','dtype')) + (tuple(proxy._spec.get('shape', ())),)
        state = dict(proxy=proxy, cache={}, stats=stats, source_proofs=source_proofs, spec_signature=signature)
        token = _ACTIVE.set(state); started = time.perf_counter(); success = False
        try:
            result = original(self, data_cropped, seg_cropped, bbox_lbs, plan, case_id)
            success = True
            return result
        finally:
            stats['event_wall_seconds'] = time.perf_counter()-started
            stats['completed_events'] = int(success); stats['failed_events'] = int(not success)
            state['cache'].clear(); state['proxy'] = None
            stats['live_crop_bytes'] = 0; stats['closed_event_scopes'] = 1
            _ACTIVE.reset(token)
            cumulative = getattr(self, '_v24_native_crop_reuse_stats', None)
            if cumulative is None:
                cumulative = self._v24_native_crop_reuse_stats = _empty_stats()
            for name, value in stats.items():
                if name == 'peak_live_crop_bytes':
                    cumulative[name] = max(cumulative[name], value)
                elif name == 'live_crop_bytes':
                    cumulative[name] = value
                else:
                    cumulative[name] += value
            self._v24_native_crop_reuse_last = copy.deepcopy(stats)
    apply.__name__ = original.__name__
    apply._v24_native_execution_original = original
    apply._v24_native_execution_original_sha256 = original_sha
    apply._v24_native_execution_source_proofs = source_proofs
    return apply


def install_runtime():
    """Install in a new process before CUDA/augmentation construction."""
    if torch.cuda.is_initialized():
        raise RuntimeError('Native execution runtime cannot hot-swap initialized CUDA')
    if storage._CropArray.__getitem__ not in (_BASE_GETITEM, _cached_getitem):
        raise ValueError('Foreign lossless crop getter replacement refused')
    frozen = importlib.import_module(crop.FROZEN_MODULE)
    loader = frozen.FrozenV23Loader
    current = loader.__dict__.get(crop.METHOD)
    receipt = getattr(loader, '_v24_native_execution_contract', None)
    if receipt is not None:
        if (receipt != runtime_contract() or current is not loader.__dict__.get('_v24_native_execution_method')
                or getattr(current, '_v24_native_execution_original', None) is not loader.__dict__.get('_v24_crop_protocol_method')):
            raise ValueError('Native execution runtime identity changed')
        if storage._CropArray.__getitem__ is not _cached_getitem:
            raise ValueError('Native execution crop getter changed')
        _source_guard(current._v24_native_execution_source_proofs)
        return receipt
    crop.install_crop_protocol()
    current = loader.__dict__.get(crop.METHOD)
    if current is None or getattr(loader, '_v24_crop_protocol_contract', None) != crop.crop_protocol_contract():
        raise ValueError('Exact admitted frozen crop protocol required')
    replacement = wrap_paste_method(current)
    storage._CropArray.__getitem__ = _cached_getitem
    setattr(loader, crop.METHOD, replacement)
    loader._v24_native_execution_method = replacement
    receipt = runtime_contract(); loader._v24_native_execution_contract = receipt
    return receipt


def run_training_entry():
    install_runtime()
    from nnunetv2.run.run_training import run_training_entry as native_entry
    return native_entry()
