"""Reclaim inactive upper-preparation allocations without changing sealed inputs.

The frozen preparation file participates in canonical source identity. This
sidecar changes only its execution lifetime: eviction frames release every
payload reference before actual GC/allocator reclamation and a fresh RSS check.
All complete inputs, graph equations, four workers and 8/64-GiB limits remain.
"""
from __future__ import annotations

import copy
import gc
import hashlib
from pathlib import Path
import time

import psutil

from . import v25_preparation as frozen
from .host_memory import _trim_heap

FORMAT='v25_upper_preparation_inactive_RSS_reclamation_execution_v1'
_CLASS=frozen.AcceleratedNativeInputs
_ORIGINAL_INIT=_CLASS.__init__
_ORIGINAL_RSS=_CLASS._rss_locked
_ORIGINAL_RECEIPT=_CLASS.preparation_runtime_receipt
_SOURCE=Path(__file__).resolve()
_EXECUTION_RUNTIME_FILES=()


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _profile(inputs):
    profile=getattr(inputs,'_upper_memory_profile',None)
    if profile is None:
        profile=dict(checks=0,pressure_events=0,allocator_releases=0,
            cache_evictions=0,released_logical_bytes=0,gc_collected=0,
            strict_failures=0,maximum_observed_RSS_bytes=0,
            allocator_release_seconds=0.,last_reclamation=None)
        inputs._upper_memory_profile=profile
    return profile


def _rss(inputs):
    value=int(psutil.Process().memory_info().rss)
    profile=_profile(inputs)
    profile['maximum_observed_RSS_bytes']=max(profile['maximum_observed_RSS_bytes'],value)
    return value


def _resident(inputs):return inputs._raw_bytes+inputs._region_bytes+inputs._source_bytes


def _pressure_trigger(inputs,coordinator):
    # Training providers retain their declared 32-GiB cache. Reuse the same
    # half-workspace boundary as their coordinator, while CPU-only preparation
    # still reserves workspace around the original 8-GiB upper-input cache.
    resident=max((provider.resident_bytes for provider in
        getattr(coordinator,'providers',())),default=inputs.raw_resident_bytes)
    resident=max(inputs.raw_resident_bytes,resident)
    return inputs.rss_bytes-(inputs.rss_bytes-resident)//2


def _evict_one(inputs):
    """Return scalar accounting only; no popped payload escapes this frame."""
    released=0
    if inputs._raw_cache:
        case,item=inputs._raw_cache.popitem(last=False)
        released+=item[1];inputs._raw_bytes-=item[1]
        del item
        inputs._regions.pop(case,None)
        for key in [key for key in inputs._donors if key[0]==case]:inputs._donors.pop(key)
        for key in [key for key in inputs._upper_sources if key[0]==case]:
            item=inputs._upper_sources.pop(key)
            released+=item[1];inputs._source_bytes-=item[1]
            del item
        category='raw'
    elif inputs._upper_sources:
        _,item=inputs._upper_sources.popitem(last=False)
        released+=item[1];inputs._source_bytes-=item[1]
        del item
        category='source'
    elif inputs._static_regions:
        _,item=inputs._static_regions.popitem(last=False)
        released+=item[1];inputs._region_bytes-=item[1]
        del item
        inputs._region_profile['evictions']+=1
        category='region'
    else:
        return None
    # Weak region admissions remain valid for active builders. Cache ownership
    # is the only reference removed; immutable arrays held by a caller survive.
    profile=_profile(inputs)
    profile['cache_evictions']+=1;profile['released_logical_bytes']+=released
    return category,int(released)


def _reclaim(inputs):
    began=time.perf_counter();before=_rss(inputs)
    collected=gc.collect()
    heap=_trim_heap()
    after=_rss(inputs)
    profile=_profile(inputs)
    profile['allocator_releases']+=1;profile['gc_collected']+=collected
    profile['allocator_release_seconds']+=time.perf_counter()-began
    profile['last_reclamation']=dict(before_RSS_bytes=before,after_RSS_bytes=after,
        reclaimed_RSS_bytes=before-after,RSS_reclaimed_fraction=(before-after)/before,
        gc_collected=collected,heap_trim=heap,live_inputs_changed=False,
        CUDA_initialized_by_reclamation=False)
    print('V2.5 UPPER RSS RECLAIM before_bytes=%d after_bytes=%d limit_bytes=%d '
          'reclaimed_RSS_bytes=%d reclaim_fraction=%.6f '
          'released_logical_bytes=%d gc_collected=%d heap_trim_supported=%s '
          'heap_trim_applied=%s' % (before,after,inputs.rss_bytes,
            before-after,(before-after)/before,
            profile['released_logical_bytes'],collected,heap.get('supported'),
            heap.get('applied')),flush=True)
    return after


def _rss_locked(inputs):
    profile=_profile(inputs);profile['checks']+=1
    coordinator=getattr(inputs,'memory_coordinator',None)
    if coordinator is not None:coordinator.trim(strict=False)
    # The original PressureBudget execution policy reserves half the uncached
    # workspace before the hard RSS ceiling. This changes no budget or batch.
    trigger=_pressure_trigger(inputs,coordinator)
    current=_rss(inputs)
    if current>trigger:
        profile['pressure_events']+=1
        current=_reclaim(inputs)
    while _resident(inputs)>inputs.raw_resident_bytes or current>trigger:
        deficit=max(_resident(inputs)-inputs.raw_resident_bytes,current-trigger,0)
        released=0;changed=False
        while released<deficit:
            eviction=_evict_one(inputs)
            if eviction is None:break
            changed=True;released+=eviction[1]
        if not changed:break
        # _evict_one's payload-owning frame has returned. Logical byte ledgers
        # choose a collection boundary; only freshly measured RSS admits work.
        current=_reclaim(inputs)
    current=_rss(inputs)
    if current>inputs.rss_bytes:
        profile['strict_failures']+=1
        print('V2.5 UPPER RSS FAILED actual_bytes=%d limit_bytes=%d '
              'resident_raw_region_source_bytes=%d cache_evictions=%d' %
              (current,inputs.rss_bytes,_resident(inputs),profile['cache_evictions']),flush=True)
        raise MemoryError('Actual complete v24 workspace exceeds process RSS; no subset or fallback')


def _init(inputs,*args,**kwargs):
    _ORIGINAL_INIT(inputs,*args,**kwargs)
    # The frozen admission method replaces its SHA map. Carry its admitted
    # preparation/CLI files forward, alongside the separate execution adapters.
    files=tuple(dict.fromkeys((*inputs._runtime_file_sha256,_SOURCE,
        *_EXECUTION_RUNTIME_FILES)))
    inputs._admit_runtime_files(files)
    _profile(inputs)


def _receipt(inputs):
    value=_ORIGINAL_RECEIPT(inputs)
    with inputs._lock:
        value['upper_memory_reclamation']=dict(format=FORMAT,
            source_sha256=_sha(_SOURCE),RSS_limit_bytes=inputs.rss_bytes,
            raw_region_source_limit_bytes=inputs.raw_resident_bytes,
            actual_RSS_bytes=_rss(inputs),profile=copy.deepcopy(_profile(inputs)),
            source_identity_unchanged=True,full_input_and_graphs_preserved=True,
            scientific_file_bytes_changed=False,physical_batch_changed=False,
            allocator_hint_never_relaxes_strict_RSS=True)
    return value


def install_upper_memory_runtime(*,runtime_files=()):
    """Call in run_v25_execution before frozen.main installs its constructor."""
    global _EXECUTION_RUNTIME_FILES
    _EXECUTION_RUNTIME_FILES=tuple(dict.fromkeys((*_EXECUTION_RUNTIME_FILES,
        *(Path(path).resolve(strict=True) for path in runtime_files))))
    actual=(_CLASS.__init__,_CLASS._rss_locked,_CLASS.preparation_runtime_receipt)
    original=(_ORIGINAL_INIT,_ORIGINAL_RSS,_ORIGINAL_RECEIPT)
    installed=(_init,_rss_locked,_receipt)
    if actual not in (original,installed):
        raise ValueError('Unknown existing upper-memory execution override')
    _CLASS.__init__,_CLASS._rss_locked,_CLASS.preparation_runtime_receipt=installed
    root=_SOURCE.parent
    return dict(format=FORMAT,runtime_source_sha256=_sha(_SOURCE),
        frozen_preparation_source_sha256=_sha(Path(frozen.__file__)),
        unchanged_scientific_files_sha256={name:_sha(root/name) for name in
            ('v25_factory.py','v24_geometry.py','v24_inputs.py','v24_model.py','v24_provider.py')},
        workers=4,raw_region_source_limit_bytes=8*2**30,RSS_limit_bytes=64*2**30,
        source_identity_unchanged=True,full_input_and_graphs_preserved=True,
        disk_cache_added=False,active_weak_region_admissions_preserved=True)
