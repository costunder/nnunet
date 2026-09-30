"""Reclaim immutable input caches, never training samples or model state."""
import gc
import sys
import ctypes
import psutil

def trim(cache,*,bytes_to_release=0,prefixes=()):
    released=0;removed=0
    with cache.lock:
        for key in list(cache.values):
            if not (isinstance(key,tuple) and key[0] in prefixes) and released>=bytes_to_release:continue
            value,storage=cache.values.pop(key)
            for identity,size in storage.items():
                cache.references[identity]-=1
                if not cache.references[identity]:
                    del cache.references[identity];cache.bytes-=size;released+=size
            del value
            cache.stats['evictions']+=1;removed+=1
    if removed:
        gc.collect()
        # glibc can otherwise retain freed multithreaded CPU work buffers. This
        # releases free heap pages only; no file, tensor or checkpoint is changed.
        if sys.platform.startswith('linux'):
            libc=ctypes.CDLL(None);fn=getattr(libc,'malloc_trim',None)
            if fn is not None:fn.argtypes=[ctypes.c_size_t];fn.restype=ctypes.c_int;fn(0)
    return dict(evicted=removed,released_tensor_bytes=released)

def guard(cache,rss_limit):
    if rss_limit is None:return None
    if cache.budget>=rss_limit:raise ValueError('Resident cache must leave CPU workspace inside RSS budget')
    before=psutil.Process().memory_info().rss
    # Preserve the user's declared RSS-minus-cache workspace, rather than letting
    # raw tensors, materialized batches and native allocations fill RSS together.
    target=cache.budget
    report=trim(cache,bytes_to_release=max(0,before-target)) if before>target else dict(evicted=0,released_tensor_bytes=0)
    after=psutil.Process().memory_info().rss
    report.update(rss_before=before,rss_after=after,rss_limit=rss_limit,resident_bytes=cache.bytes)
    if after>rss_limit:raise MemoryError(f'RSS remains over budget after cache reclamation: {report}')
    return report
