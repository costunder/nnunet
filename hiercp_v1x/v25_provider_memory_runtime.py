"""Execution-only allocator reclamation for the unchanged V24 CPU provider.

Inactive graph cycles and allocator arenas can retain RSS after the original
coordinator evicts its cache ownership. Collect/release them at its existing
pressure boundary and then remeasure actual RSS. Live/protected inputs, complete
graphs, prefetch, four workers, 32-GiB residency and 64-GiB RSS stay unchanged.
"""
from __future__ import annotations

import copy
import gc
import hashlib
from pathlib import Path
import time

import psutil

from . import v24_provider as frozen
from .host_memory import _trim_heap

FORMAT='v25_provider_inactive_RSS_reclamation_execution_v1'
_SOURCE=Path(__file__).resolve()
_CLASS=frozen.V24InputCoordinator
_ORIGINAL_TRIM=_CLASS._trim_locked


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _profile(coordinator):
    profile=getattr(coordinator,'_v25_provider_memory_profile',None)
    if profile is None:
        profile=dict(checks=0,pressure_events=0,allocator_releases=0,
            original_cache_evictions=0,gc_collected=0,strict_failures=0,
            maximum_observed_RSS_bytes=0,allocator_release_seconds=0.,
            last_reclamation=None)
        coordinator._v25_provider_memory_profile=profile
    return profile


def _rss(coordinator):
    current=int(psutil.Process().memory_info().rss)
    profile=_profile(coordinator)
    profile['maximum_observed_RSS_bytes']=max(profile['maximum_observed_RSS_bytes'],current)
    return current


def _evictions(coordinator):
    # Scalar-only telemetry cannot retain an evicted graph/record payload.
    return sum(provider._stats['view_evictions']+provider._stats['record_evictions']
               for provider in coordinator.providers)


def _reclaim(coordinator):
    began=time.perf_counter();before=_rss(coordinator)
    collected=gc.collect();heap=_trim_heap();after=_rss(coordinator)
    profile=_profile(coordinator)
    profile['allocator_releases']+=1;profile['gc_collected']+=collected
    profile['allocator_release_seconds']+=time.perf_counter()-began
    profile['last_reclamation']=dict(before_RSS_bytes=before,after_RSS_bytes=after,
        reclaimed_RSS_bytes=before-after,gc_collected=collected,heap_trim=heap,
        active_input_values_changed=False,protected_cache_entries_removed=False,
        CUDA_initialized_by_reclamation=False)
    print('V2.5 PROVIDER RSS RECLAIM before_bytes=%d after_bytes=%d '
          'limit_bytes=%d reclaimed_RSS_bytes=%d gc_collected=%d '
          'heap_trim_supported=%s heap_trim_applied=%s' %
          (before,after,coordinator.rss_bytes,before-after,collected,
           heap.get('supported'),heap.get('applied')),flush=True)
    return after


def _trim_locked(coordinator,*,strict):
    profile=_profile(coordinator);profile['checks']+=1
    # Exactly the original coordinator's boundary, rather than a new cache cap.
    target=coordinator.rss_bytes-min(coordinator.rss_bytes//64,3*2**30)
    current=_rss(coordinator)
    if current>target:
        profile['pressure_events']+=1
        _reclaim(coordinator)
    before=_evictions(coordinator)
    # Preserve every original eviction category, protection and LRU rule.
    # Original strict admission is deferred only until the allocator hint has
    # run; a fresh real-RSS check below retains the identical hard budget.
    _ORIGINAL_TRIM(coordinator,strict=False)
    profile['original_cache_evictions']+=_evictions(coordinator)-before
    current=_rss(coordinator)
    if current>target:
        current=_reclaim(coordinator)
    if strict and current>coordinator.rss_bytes:
        profile['strict_failures']+=1
        print('V2.5 PROVIDER RSS FAILED actual_bytes=%d limit_bytes=%d '
              'original_cache_evictions=%d' %
              (current,coordinator.rss_bytes,profile['original_cache_evictions']),flush=True)
        raise MemoryError('Full active GT-free input exceeds shared RSS; no graph or data reduction')


def runtime_contract():
    return dict(format=FORMAT,runtime_source_sha256=_sha(_SOURCE),
        frozen_provider_source_sha256=_sha(Path(frozen.__file__)),
        allocator_hint_source_sha256=_sha(_SOURCE.with_name('host_memory.py')),
        original_cache_eviction_function_preserved=True,
        pressure_boundary='original RSS-min(RSS//64,3GiB)',
        actual_RSS_remeasured_after_allocator_release=True,
        active_protected_inputs_unchanged=True,physical_batch_unchanged=True,
        workers_and_prefetch_unchanged=True,resident_and_RSS_limits_unchanged=True,
        model_optimizer_RNG_checkpoint_unchanged=True,
        frozen_scientific_files_modified=False,running_processes_hot_swapped=False)


def install_runtime():
    """Install in one new execution process before constructing its inputs."""
    if _CLASS._trim_locked not in (_ORIGINAL_TRIM,_trim_locked):
        raise ValueError('Unknown existing provider coordinator override')
    _CLASS._trim_locked=_trim_locked
    return runtime_contract()


def receipt(coordinator):
    with coordinator.lock:
        return dict(contract=runtime_contract(),RSS_limit_bytes=coordinator.rss_bytes,
            actual_RSS_bytes=_rss(coordinator),profile=copy.deepcopy(_profile(coordinator)))
