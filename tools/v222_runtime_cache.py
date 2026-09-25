"""Bounded, single-flight caches for immutable v2.22 inputs.

No learned embeddings are cached here. Epoch-dependent graph views keep the
original epoch seed. Only cache capacity changes with RAM, never data coverage.
"""
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import fields, is_dataclass
import gzip
import hashlib
import io
from pathlib import Path
import threading

import numpy as np
import torch
from hiercp.preparation_runtime import snapshot as resource_snapshot
from hiercp_v222.v1_cache import PairDataset as OriginalDataset, PairLoader as OriginalLoader
from hiercp_v222.v1_local import materialize, LocalBatch
from torch_geometric.data import Batch


def storages(value, found=None):
    """Count unique CPU tensor storage, including aliases across cache entries."""
    found = {} if found is None else found
    if torch.is_tensor(value):
        storage = value.untyped_storage()
        found[(str(value.device), storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    elif isinstance(value, np.ndarray):
        allocation = value
        while isinstance(allocation.base, np.ndarray):
            allocation = allocation.base
        found[('cpu', allocation.__array_interface__['data'][0], allocation.nbytes)] = allocation.nbytes
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            storages(getattr(value, field.name), found)
    elif hasattr(value, 'stores'):
        for store in value.stores:
            storages(store.to_dict(), found)
    elif isinstance(value, dict):
        for item in value.values():
            storages(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            storages(item, found)
    return found


class TensorCache:
    def __init__(self, budget):
        if budget <= 0:
            raise ValueError('Positive cache RAM budget required')
        self.budget = int(budget)
        self.lock = threading.Lock()
        self.values = OrderedDict()
        self.pending = {}
        self.references = {}
        self.bytes = 0
        self.stats = dict(hits=0, misses=0, shared_waits=0, evictions=0)

    def get(self, key, produce):
        with self.lock:
            if key in self.values:
                self.values.move_to_end(key)
                self.stats['hits'] += 1
                return self.values[key][0]
            if key in self.pending:
                future = self.pending[key]
                owner = False
                self.stats['shared_waits'] += 1
            else:
                future = self.pending[key] = Future()
                owner = True
                self.stats['misses'] += 1
        if not owner:
            return future.result()
        try:
            value = produce()  # I/O, decoding, graph construction OUTSIDE lock.
            storage = storages(value)
            with self.lock:
                additional = lambda: sum(n for k, n in storage.items() if k not in self.references)
                if sum(storage.values()) <= self.budget:
                    while self.values and self.bytes + additional() > self.budget:
                        _, (_, old) = self.values.popitem(last=False)
                        for k, n in old.items():
                            self.references[k] -= 1
                            if not self.references[k]:
                                del self.references[k]
                                self.bytes -= n
                        self.stats['evictions'] += 1
                    for k, n in storage.items():
                        if k not in self.references:
                            self.bytes += n
                            self.references[k] = 0
                        self.references[k] += 1
                    self.values[key] = (value, storage)
                future.set_result(value)
                del self.pending[key]
            return value
        except BaseException as error:
            with self.lock:
                future.set_exception(error)
                self.pending.pop(key, None)
            raise

    def report(self):
        with self.lock:
            return dict(**self.stats, resident_bytes=self.bytes, budget_bytes=self.budget,
                        resident_entries=len(self.values), pending=len(self.pending))


class CanonicalStore:
    def __init__(self, root, budget):
        self.root = Path(root).resolve()
        self.cache = TensorCache(budget)
        self.verified = set()
        self.verify_lock = threading.Lock()

    def path_key(self, relative, digest):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Graph reference escapes immutable cache root')
        stat = path.stat()
        return path, (str(path), digest, stat.st_size, stat.st_mtime_ns)

    def read(self, relative, digest):
        path, key = self.path_key(relative, digest)

        def produce():
            raw = path.read_bytes()
            with self.verify_lock:
                verified = key in self.verified
            if not verified:
                if hashlib.sha256(raw).hexdigest() != digest:
                    raise ValueError(f'Graph SHA mismatch: {path}')
                with self.verify_lock:
                    self.verified.add(key)
            if str(path).endswith('.pt.gz'):
                raw = gzip.decompress(raw)
            return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)

        return self.cache.get(('file', key), produce), key

    def record(self, row):
        value, key = self.read(row['path'], row['sha256'])
        value = dict(value)  # Never mutate the canonical cache entry.
        reference = value.pop('shared_source', None)
        if reference != row.get('shared_source'):
            raise ValueError('Shared donor identity differs between index and graph payload')
        source_key = None
        if reference is not None:
            source, source_key = self.read(reference['path'], reference['sha256'])
            value.update(source)
        return value, (key, source_key)


class CachedPairDataset(OriginalDataset):
    """Original cohort/provenance checks, shared store, seed-aware graph reuse."""
    stores = {}
    stores_lock = threading.Lock()

    def __init__(self, index, partition):
        super().__init__(index, partition)
        key = str(self.path)
        with self.stores_lock:
            if key not in self.stores:
                self.stores[key] = CanonicalStore(self.root, int(resource_snapshot()['available_memory_bytes'] * .20))
            self.store = self.stores[key]
        self.budget = self.store.cache.budget

    def record(self, i):
        return self.store.record(self.rows[i])[0]

    def item(self, i, epoch=0):
        row = self.rows[i]
        # File stat identity invalidates stale views if an input was modified.
        _, key = self.store.path_key(row['path'], row['sha256'])
        ref = row.get('shared_source')
        source_key = self.store.path_key(ref['path'], ref['sha256'])[1] if ref else None
        value = self.store.cache.get(('view', key, source_key, int(epoch)),
            lambda: materialize(self.record(i), epoch=epoch))
        return value, i


class CachedPairLoader(OriginalLoader):
    def make(self, indices, epoch):
        args = [(int(i), epoch) for i in indices]
        values = list(self.pool.map(lambda a: self.dataset.item(*a), args)) if self.pool else [self.dataset.item(*a) for a in args]
        graphs, targets, sources, source_ids, lookup = [], [], [], [], {}
        for (graph, source, target), index in values:
            reference = self.dataset.rows[index].get('shared_source')
            # Verified content identity survives cache eviction/reloading. A
            # duplicate donor never needs a second CNN map within this batch.
            key = ('sha', reference['content_sha256']) if reference else ('ptr', source.data_ptr(), tuple(source.shape), source.dtype)
            if key not in lookup:
                lookup[key] = len(sources)
                sources.append(source)
            source_ids.append(lookup[key])
            graphs.append(graph)
            targets.append(target)
        return LocalBatch(Batch.from_data_list(graphs), torch.stack(sources), torch.stack(targets),
            torch.tensor(source_ids), torch.tensor(indices)).pin_memory()
