"""Lossless CPU worker/view reuse for the original native input provider.

Only immutable CPU canonical records and fixed-validation sampled graph views
are cached. Every observation, actual epoch and both original view seeds remain
unchanged. No neural embedding, prediction, target subset or graph cap exists.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
import copy
import threading
import time

import psutil

from . import transition_v1_data as original
from .transition_v1_data import OriginalInputProvider
from . import transition_v1_local as local


def _view_owner(payload):
    graphs, source, target = payload
    return (tuple(graph.to_dict() for graph in graphs), source, target)


def _view_signature(payload):
    return local._tensor_signature(_view_owner(payload))


class V23InputMemoryCoordinator:
    """One actual process-RSS guard for both native partition caches.

    Per-provider resident limits share this RSS ceiling. Eviction first drops
    inactive sampled views, then inactive canonical cache owners. The current
    complete observation chunk remains protected; eviction changes residency,
    never which observations or graph nodes are processed.
    """
    def __init__(self, rss_bytes):
        if type(rss_bytes) is not int or rss_bytes <= 0:
            raise ValueError('Explicit shared input RSS budget required')
        self.rss_bytes = rss_bytes
        self.lock = threading.RLock()
        self._providers = []

    def register(self, provider):
        with self.lock:
            if provider.rss_bytes != self.rss_bytes:
                raise ValueError('Every native partition must share the exact process RSS ceiling')
            self._providers.append(provider)

    def trim(self):
        with self.lock:
            headroom = min(self.rss_bytes // 64, 3 * 1024**3)
            target = self.rss_bytes - headroom
            for category in ('view', 'record'):
                while psutil.Process().memory_info().rss > target:
                    changed = False
                    for provider in self._providers:
                        changed |= provider._evict_one(category)
                        if psutil.Process().memory_info().rss <= target:
                            break
                    if not changed:
                        break
            if psutil.Process().memory_info().rss > self.rss_bytes:
                raise MemoryError('Complete active native input exceeds shared process RSS budget; no data/graph reduction')


class V23InputProvider(OriginalInputProvider):
    """Original provider plus persistent workers and exact-epoch CPU view LRU."""
    def __init__(self, dataset, *, workers, resident_bytes, rss_bytes, cache_index=None,
                 persistent_cpu_workers=True, cache_sampled_views=True,
                 fixed_validation_epoch=29, memory_coordinator=None):
        if type(persistent_cpu_workers) is not bool or type(cache_sampled_views) is not bool:
            raise ValueError('Explicit CPU worker/view-cache policy required')
        if type(fixed_validation_epoch) is not int or not 0 <= fixed_validation_epoch <= 40:
            raise ValueError('Exact original fixed-validation epoch required')
        super().__init__(dataset, workers=workers, resident_bytes=resident_bytes,
                         rss_bytes=rss_bytes, cache_index=cache_index)
        self.persistent_cpu_workers = persistent_cpu_workers
        self.cache_sampled_views = cache_sampled_views
        self.fixed_validation_epoch = fixed_validation_epoch
        self.memory_coordinator = memory_coordinator or V23InputMemoryCoordinator(rss_bytes)
        self._resident._groups['view'] = original._StorageIntervalUnion()
        self._views = original._AccountedCache(self._resident, 'view', lambda entry: _view_owner(entry['payload']))
        self._protected_ids, self._protected_view_keys = set(), set()
        self._executor = (ThreadPoolExecutor(max_workers=workers, thread_name_prefix='v23_cpu_native')
                          if persistent_cpu_workers else None)
        self._closed = False
        self._stats = dict(get_calls=0, canonical_load_calls=0, canonical_load_seconds=0.,
            sampled_pair_hits=0, sampled_pair_misses=0, sampled_pairs_materialized=0,
            materialize_seconds=0., collate_seconds=0., get_seconds=0.,
            sampled_view_evictions=0, canonical_record_evictions=0)
        self.memory_coordinator.register(self)

    def _parallel(self, function, items):
        if self._executor is None:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                return list(executor.map(function, items))
        futures = [self._executor.submit(function, item) for item in items]
        try:
            return [future.result() for future in futures]
        finally:
            for future in futures: future.cancel()
            # Running tasks own complete immutable records until they finish.
            # An originating error cannot leave detached materialization jobs.
            wait(futures)

    def _evict_one(self, category):
        cache = self._views if category == 'view' else self._records
        protected = self._protected_view_keys if category == 'view' else self._protected_ids
        key = next((key for key in cache if key not in protected), None)
        if key is None:
            return False
        cache.pop(key)
        self._stats['sampled_view_evictions' if category == 'view' else 'canonical_record_evictions'] += 1
        return True

    def _guard(self):
        self.memory_coordinator.trim()
        super()._guard()

    def _make_room(self, active_cases=(), active_records=()):
        while self._views and self._resident.bytes('all') > self.resident_bytes:
            if not self._evict_one('view'): break
        self.memory_coordinator.trim()
        super()._make_room(active_cases=active_cases, active_records=active_records)

    def _records_for(self, rows):
        if not self.persistent_cpu_workers:
            return super()._records_for(rows)
        # Identical native admission/cache logic, with the worker executor kept
        # alive instead of creating a new pool for every candidate chunk.
        self._guard()
        if self.cache_index is not None and original._checked_sha(self.cache_index) != self._cache_index_sha256:
            raise ValueError('Complete D canonical index changed after admission')
        missing = [row for row in rows if row['id'] not in self._records]
        if self._canonical is None and missing:
            self._initialize_raw()
            active = {row[key] for row in missing for key in ('case_id', 'donor_case_id')}
            self._make_room(active)
            self._raw.preload(missing)
            self._make_room(active)
            for row in missing: self._prepare_donor(row)
            build = self._build
        else:
            build = self._load
        if missing:
            began = time.perf_counter()
            records = self._parallel(build, missing)
            self._stats['canonical_load_calls'] += len(missing)
            self._stats['canonical_load_seconds'] += time.perf_counter()-began
            for row, record in zip(missing, records):
                self._records[row['id']] = record, original._bytes(record)
            self._cached_bytes = self._resident.bytes('canonical')
        result = []
        for row in rows:
            record, _ = self._records[row['id']]
            self._records.move_to_end(row['id']); result.append(record)
        self._make_room(active_records=result)
        return result

    def get(self, ids, *, epoch=0):
        with self.memory_coordinator.lock:
            if self._closed:
                raise RuntimeError('Closed native input provider cannot accept another batch')
            if (not isinstance(ids, (list, tuple)) or not ids
                    or any(type(index) is not int or not 0 <= index < len(self.ds) for index in ids)):
                raise ValueError('Exact actual native observation row indices required')
            if type(epoch) is not int or not 0 <= epoch <= 40:
                raise ValueError('Exact actual sampling epoch0..40 required')
            if self.ds.partition == 'inner_val' and epoch not in (0, self.fixed_validation_epoch):
                raise ValueError('Validation sampled views must use the original fixed epoch')
            began = time.perf_counter()
            self._protected_ids = {self.ds.rows[index]['id'] for index in ids}
            try:
                if not self.persistent_cpu_workers and not self.cache_sampled_views:
                    # Explicit benchmark reference: the original get path.
                    return super().get(ids, epoch=epoch)
                records = self._records_for([self.ds.rows[index] for index in ids])
                should_cache = self.cache_sampled_views and self.ds.partition == 'inner_val'
                keys = []
                for record in records:
                    local.validate_record(record)
                    keys.append((record['input_provenance']['observation_id'], epoch,
                                 record['content_binding']['graph_sha256']))
                self._protected_view_keys = set(keys)
                payloads, missing = [None]*len(records), []
                for position, (record, key) in enumerate(zip(records, keys)):
                    entry = self._views.get(key) if should_cache else None
                    if entry is None:
                        missing.append((position, record, key)); self._stats['sampled_pair_misses'] += 1
                    else:
                        if _view_signature(entry['payload']) != entry['signature']:
                            raise ValueError('Cached original sampled CPU view changed')
                        payloads[position] = entry['payload']; self._views.move_to_end(key)
                        self._stats['sampled_pair_hits'] += 1
                materialize_started = time.perf_counter()
                built = self._parallel(lambda item: local.materialize_pair(item[1], epoch=epoch), missing)
                self._stats['materialize_seconds'] += time.perf_counter()-materialize_started
                self._stats['sampled_pairs_materialized'] += len(missing)
                for (position, _, key), payload in zip(missing, built):
                    payloads[position] = payload
                    if should_cache:
                        self._views[key] = dict(payload=payload, signature=_view_signature(payload))
                self._make_room(active_records=records)
                # Return a private collated graph. Source/target patches are
                # stacked by the original collator, preserving storage sharing
                # inside the resident cache without exposing mutable owners.
                private = ([((first.clone(), second.clone()), source, target)
                            for (first, second), source, target in payloads]
                           if should_cache else payloads)
                collate_started = time.perf_counter()
                batch = local.collate(list(zip(private, ids)))
                self._stats['collate_seconds'] += time.perf_counter()-collate_started
                self._guard()
                return batch
            finally:
                self._protected_ids.clear(); self._protected_view_keys.clear()
                self._stats['get_calls'] += 1; self._stats['get_seconds'] += time.perf_counter()-began
                # Once private collation owns the active batch, every cache
                # owner can be evicted without invalidating returned tensors.
                while self._resident.bytes('all') > self.resident_bytes and self._evict_one('view'):
                    continue

    def profile(self):
        with self.memory_coordinator.lock:
            return dict(**copy.deepcopy(self._stats), persistent_CPU_workers=self.persistent_cpu_workers,
                CPU_workers=self.workers,
                CPU_sampled_view_cache=self.cache_sampled_views and self.ds.partition == 'inner_val',
                CPU_sampled_view_cache_requested=self.cache_sampled_views,
                fixed_validation_epoch=self.fixed_validation_epoch,
                sampled_view_entries=len(self._views), canonical_record_entries=len(self._records),
                sampled_view_resident_bytes=self._resident.bytes('view'),
                canonical_record_resident_bytes=self._resident.bytes('record'),
                total_unique_resident_bytes=self._resident.bytes('all'),
                resident_limit_bytes=self.resident_bytes, shared_process_RSS_limit_bytes=self.rss_bytes,
                actual_RSS_bytes=psutil.Process().memory_info().rss,
                neural_features_cached=False, changed_sampling_epoch=False, hidden_subset=False)

    def close(self):
        with self.memory_coordinator.lock:
            if self._closed: return
            self._closed = True
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            self._views.clear(); self._records.clear(); self._donors.clear()
            self.memory_coordinator._providers.remove(self)

    def __enter__(self): return self

    def __exit__(self, error_type, error, traceback): self.close()
