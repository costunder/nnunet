"""Release rebuildable host-cache references before the existing hard budgets.

This is an execution policy, not a change to a neural or data contract.  It
does not close mappings, alter arrays, drop candidates, or reduce a batch.
Active batch references remain valid and still have to fit the declared RSS.
"""
from __future__ import annotations

import ctypes
import gc
import sys
import threading
import time
import weakref

FORMAT = "v18_pressure_aware_host_cache_execution_v1"


def process_rss():
    import psutil
    return int(psutil.Process().memory_info().rss)


def cuda_allocation():
    import torch
    return int(torch.cuda.memory_allocated())


def _trim_heap():
    """An optional allocator residency hint; rejection never relaxes a budget."""
    result = dict(method="malloc_trim(0)", supported=False, applied=False)
    if not sys.platform.startswith("linux"):
        result["reason"] = "unsupported_platform"
        return result
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except AttributeError:
        result["reason"] = "allocator_does_not_export_malloc_trim"
        return result
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    result.update(supported=True, applied=bool(trim(0)))
    result["reason"] = "OS_allocator_hint_applied" if result["applied"] else "allocator_released_no_pages"
    return result


def _readonly_page_hints():
    """Advise immutable distance fields without closing any active mapping."""
    from . import u_bridge_fields as fields
    with fields._REGISTRY_LOCK:
        opened = tuple(fields._OPENED.values())
    mappings = set()
    applied, attempted, advised_bytes, unsupported = 0, 0, 0, 0
    rejected = []
    for arrays, _ in opened:
        for array in arrays:
            mapping = getattr(array, "_mmap", None)
            if mapping is None:
                raise TypeError("Distance field registry contains a non-mapped array")
            identity = id(mapping)
            if identity in mappings:
                continue
            mappings.add(identity)
            # The original helper validates readonly state on Linux.  Enforce
            # the invariant on all platforms before invoking that helper.
            if array.flags.writeable:
                raise ValueError("Pressure reclamation requires immutable field mappings")
            receipt = fields._release_readonly_pages(array)
            attempted += 1
            applied += int(receipt["applied"])
            advised_bytes += int(receipt["advised_bytes"])
            unsupported += int(not receipt["supported"])
            if receipt["supported"] and not receipt["applied"]:
                rejected.append(receipt["reason"])
    return dict(mapped_arrays_attempted=attempted, applied=applied,
                advised_bytes=advised_bytes, unsupported=unsupported, rejected=rejected,
                mappings_closed=0, array_values_changed=False)


def _evict_provider(provider):
    """Remove LRU ownership only.  Objects held by live batches stay alive."""
    with provider._lock:
        before = int(provider._resident_bytes)
        count, accounted = 0, 0
        while provider._cache:
            _, entry = provider._cache.popitem(last=False)
            accounted += int(entry[1])
            count += 1
            del entry
        # These are exactly the entries tracked by the original provider.
        # Avoid silently resetting an inconsistent accounting ledger.
        if accounted != before:
            raise ValueError(f"Resident cache accounting differs: ledger={before}, entries={accounted}")
        provider._resident_bytes -= accounted
        provider.stats["evictions"] += count
        remaining = int(provider._resident_bytes)
    return dict(entries_released=count, accounted_bytes_released=accounted,
                resident_bytes_before=before, resident_bytes_after=remaining,
                active_batch_references_changed=False)


class PressureBudget:
    """The original hard CUDA/RSS limits plus pressure-triggered cache release.

    The trigger and target use the original uncached workspace headroom:
    trigger = RSS limit - (RSS limit - resident limit) / 2; target = resident
    limit.  Every cache is cleared under pressure because entry byte estimates
    cannot predict when aliases or allocator arenas actually release RSS.
    The measured process RSS, not the cache accounting estimate, decides pass.

    Reader and reclaim hooks support deterministic UNIT tests; production uses
    the actual process, CUDA allocator, readonly mapping hints and heap trim.
    """
    def __init__(self, cuda_bytes, rss_bytes, *, resident_bytes,
                 event_callback=None, rss_reader=None, cuda_reader=None,
                 reclaim_hook=None):
        self.cuda_bytes, self.rss_bytes = int(cuda_bytes), int(rss_bytes)
        self.resident_bytes = int(resident_bytes)
        if min(self.cuda_bytes, self.rss_bytes, self.resident_bytes) <= 0:
            raise ValueError("Explicit positive CUDA, RSS and resident cache budgets required")
        if self.resident_bytes >= self.rss_bytes:
            raise ValueError("Resident cache budget must leave declared host workspace headroom")
        self.workspace_headroom_bytes = self.rss_bytes - self.resident_bytes
        self.pressure_trigger_bytes = self.rss_bytes - self.workspace_headroom_bytes // 2
        self.pressure_target_bytes = self.resident_bytes
        self._rss_reader = process_rss if rss_reader is None else rss_reader
        self._cuda_reader = cuda_allocation if cuda_reader is None else cuda_reader
        self._reclaim_hook = reclaim_hook
        if any(not callable(reader) for reader in (self._rss_reader, self._cuda_reader)):
            raise TypeError("Resource measurement readers must be callable")
        if event_callback is not None and not callable(event_callback):
            raise TypeError("Resource event callback must be callable")
        if reclaim_hook is not None and not callable(reclaim_hook):
            raise TypeError("UNIT reclaim hook must be callable")
        self._event_callback = event_callback
        self._providers = []
        self._pressure_lock = threading.RLock()
        self.reclaim_events = 0

    def register_provider(self, provider):
        for name in ("_lock", "_cache", "_resident_bytes", "stats"):
            if not hasattr(provider, name):
                raise TypeError(f"Provider has no original resident-cache attribute: {name}")
        if int(provider.resident_limit) != self.resident_bytes:
            raise ValueError("Provider resident limit differs from the sealed execution limit")
        with self._pressure_lock:
            self._providers = [ref for ref in self._providers if ref() is not None]
            if all(ref() is not provider for ref in self._providers):
                self._providers.append(weakref.ref(provider))

    def _measure(self):
        rss, cuda = int(self._rss_reader()), int(self._cuda_reader())
        if rss < 0 or cuda < 0:
            raise ValueError("Measured resource bytes cannot be negative")
        return rss, cuda

    def _event(self, value):
        if self._event_callback is not None:
            self._event_callback(value)

    def check(self):
        rss, cuda = self._measure()
        if cuda > self.cuda_bytes:
            self._event(dict(format=FORMAT, stage="hard_budget_failed", resource="CUDA",
                             actual_bytes=cuda, limit_bytes=self.cuda_bytes, rss_bytes=rss))
            raise MemoryError(f"Declared CUDA allocation budget exceeded: actual={cuda}, limit={self.cuda_bytes}")
        if rss > self.pressure_trigger_bytes:
            with self._pressure_lock:
                # Another worker may have finished reclamation while we waited.
                rss, cuda = self._measure()
                if rss > self.pressure_trigger_bytes:
                    started = time.perf_counter()
                    released = []
                    alive = []
                    provider = None
                    for ref in self._providers:
                        provider = ref()
                        if provider is not None:
                            alive.append(ref)
                            released.append(_evict_provider(provider))
                    self._providers = alive
                    del provider
                    if self._reclaim_hook is None:
                        hints = _readonly_page_hints()
                        collected = gc.collect()
                        trim = _trim_heap()
                        reclaim = dict(readonly_pages=hints, garbage_objects_collected=collected,
                                       heap_trim=trim)
                    else:
                        reclaim = self._reclaim_hook()
                        if not isinstance(reclaim, dict):
                            raise TypeError("UNIT reclaim hook must return a diagnostic mapping")
                    after, cuda = self._measure()
                    self.reclaim_events += 1
                    self._event(dict(format=FORMAT, stage="host_cache_pressure", sequence=self.reclaim_events,
                        rss_before_bytes=rss, rss_after_bytes=after, rss_limit_bytes=self.rss_bytes,
                        pressure_trigger_bytes=self.pressure_trigger_bytes,
                        pressure_target_bytes=self.pressure_target_bytes,
                        target_reached=after <= self.pressure_target_bytes,
                        measured_rss_within_hard_budget=after <= self.rss_bytes,
                        cache_providers=released, reclamation=reclaim,
                        wall_seconds=time.perf_counter() - started,
                        model_or_input_changed=False, physical_batch_changed=False))
                    rss = after
        if cuda > self.cuda_bytes:
            self._event(dict(format=FORMAT, stage="hard_budget_failed", resource="CUDA",
                             actual_bytes=cuda, limit_bytes=self.cuda_bytes, rss_bytes=rss))
            raise MemoryError(f"Declared CUDA allocation budget exceeded: actual={cuda}, limit={self.cuda_bytes}")
        if rss > self.rss_bytes:
            self._event(dict(format=FORMAT, stage="hard_budget_failed", resource="RSS",
                             actual_bytes=rss, limit_bytes=self.rss_bytes, cuda_bytes=cuda))
            raise MemoryError(f"Declared process RSS budget exceeded after cache reclamation: "
                              f"actual={rss}, limit={self.rss_bytes}; active batch or workspace must fit")


def pressure_aware_provider(original):
    """Add registration; keep the original provider's preparation/model APIs."""
    class RegisteredProvider(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not isinstance(self.budget, PressureBudget):
                raise TypeError("Pressure-aware provider requires the explicit PressureBudget execution policy")
            self.budget.register_provider(self)

        def _get(self, *args, **kwargs):
            # Reclaim before a new allocation, rather than waiting until the
            # cache, factory scratch and new item coexist after construction.
            self.budget.check()
            return super()._get(*args, **kwargs)

        def batch(self, *args, **kwargs):
            self.budget.check()
            batch = super().batch(*args, **kwargs)
            # This intentionally keeps the active batch alive during the hard
            # check: a batch that alone exceeds RSS must remain an error.
            self.budget.check()
            return batch
    RegisteredProvider.__name__ = "PressureAware" + original.__name__
    RegisteredProvider.__qualname__ = RegisteredProvider.__name__
    return RegisteredProvider
