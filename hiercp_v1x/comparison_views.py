"""Build complete comparison graph views on the measured CPU worker pool.

Only independent candidate scheduling changes. Original canonical graphs,
stable seeds, view builders and output order remain the source of truth.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
from types import FunctionType
import math


def materialize_parallel_views(sample_module, schema_module, sample, *,
                               training, epoch, global_seed, workers, ordered_map=None):
    """Run each candidate's original two views together, preserving their order."""
    if type(workers) is not int or workers < 2:
        raise ValueError('Measured parallel graph workers >=2 required')
    if 'local_graphs' in sample and 'local_graphs_view2' in sample:
        return sample
    source_local = sample.get('source_local')
    targets = sample.get('target_locals')
    if not isinstance(source_local, Mapping) or not isinstance(targets, Sequence):
        raise ValueError('V22 cache payload has no canonical source/target node tables')
    config = schema_module.graph_config_from_dict(dict(sample['graph_config']))
    case_id = str(sample.get('case_id', ''))
    sample_index = int(sample.get('sample_index', -1))
    effective_epoch = int(epoch) if training else 0

    def obtain(item):
        candidate_index, target_local = item
        seed1 = sample_module.stable_view_seed(global_seed, case_id, sample_index,
            candidate_index, effective_epoch, 0)
        seed2 = sample_module.stable_view_seed(global_seed, case_id, sample_index,
            candidate_index, effective_epoch, 1)
        first = sample_module.build_local_view(source_local, target_local, config, seed=seed1)
        second = sample_module.build_local_view(source_local, target_local, config, seed=seed2)
        return first, second

    if ordered_map is None:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='comparison-views') as pool:
            pairs = list(pool.map(obtain, enumerate(targets)))
    else:
        pairs = list(ordered_map(obtain, enumerate(targets)))
    # Publish only after every supplied candidate and both views have succeeded.
    sample['local_graphs'] = [pair[0] for pair in pairs]
    sample['local_graphs_view2'] = [pair[1] for pair in pairs]
    sample.pop('source_local', None)
    sample.pop('target_locals', None)
    return sample


class _SampleProxy:
    def __init__(self, original, schema, provider):
        self._original = original
        self._schema = schema
        self._provider = provider

    def __getattr__(self, name):
        return getattr(self._original, name)

    def materialize_sample_views(self, sample, *, training, epoch, global_seed):
        return self._provider._materialize_views(self._original, self._schema, sample,
            training=training, epoch=epoch, global_seed=global_seed)


class _RuntimeProxy:
    def __init__(self, original, provider):
        self._original = original
        self.sample = _SampleProxy(original.sample, original.schema, provider)

    def __getattr__(self, name):
        return getattr(self._original, name)


def parallel_view_provider(original):
    """Wrap only this provider's runtime; imported archived modules stay intact."""
    class ParallelViewProvider(original):
        def __init__(self, *args, **kwargs):
            self._parallel_view_lock = threading.RLock()
            self._parallel_view_runtime = None
            self._comparison_pool = None
            self._comparison_staging_lock = threading.Lock()
            self._comparison_keys_lock = threading.Lock()
            self._comparison_keys = {}
            super().__init__(*args, **kwargs)
            if type(self.workers) is not int or self.workers < 2:
                raise ValueError('Measured parallel graph workers >=2 required')

        def _runtime(self):
            # Canonical graph workers also request this runtime. Serialize its
            # original lazy verification, then share one provider-local proxy.
            with self._parallel_view_lock:
                current = super()._runtime()
                proxy = self._parallel_view_runtime
                if proxy is None or proxy._original is not current:
                    proxy = _RuntimeProxy(current, self)
                    self._parallel_view_runtime = proxy
                return proxy

        def _materialize_views(self, sample_module, schema_module, sample, *,
                               training, epoch, global_seed):
            # Kept as a method so an outer timing wrapper can measure this
            # complete phase without modifying any original sample function.
            return materialize_parallel_views(sample_module, schema_module, sample,
                training=training, epoch=epoch, global_seed=global_seed, workers=self.workers,
                ordered_map=self._comparison_map)

        def _comparison_map(self, function, values):
            # A certified provider can share this pool for cold canonical
            # construction too; batch coordinators never execute in this pool.
            pool = self._comparison_pool
            if pool is not None:
                return list(pool.map(function, values))
            with ThreadPoolExecutor(max_workers=self.workers,
                                    thread_name_prefix='comparison-views') as own_pool:
                return list(own_pool.map(function, values))

        def _get(self, key, factory):
            # Concurrent source preparations may need the same case/region.
            # Keep the original LRU semantics but build/load each key once.
            with self._comparison_keys_lock:
                entry = self._comparison_keys.setdefault(key, [threading.RLock(), 0])
                entry[1] += 1
            try:
                with entry[0]:
                    return super()._get(key, factory)
            finally:
                with self._comparison_keys_lock:
                    entry[1] -= 1
                    if entry[1] == 0:
                        del self._comparison_keys[key]

        def _local_map(self, *args, **kwargs):
            original = super()._local_map
            pool = self._comparison_pool
            if pool is None:
                return original(*args, **kwargs)
            if getattr(self, 'independent_cold_inputs', False) is not True:
                raise RuntimeError('Cold local construction requires independent upper helpers')

            @contextmanager
            def shared_executor(*, max_workers):
                if max_workers != self.workers:
                    raise ValueError('Original canonical worker capacity changed')
                yield pool

            # Run the original implementation, including binding, publication,
            # shape/type checks and ordering. Only executor ownership changes.
            function = original.__func__
            namespace = dict(function.__globals__, ThreadPoolExecutor=shared_executor)
            bound = FunctionType(function.__code__, namespace, function.__name__,
                                 function.__defaults__, function.__closure__)
            bound.__kwdefaults__ = function.__kwdefaults__
            return bound(self, *args, **kwargs)

        def cold_staging_bytes(self, indices):
            """Conservative dense-array inventory, additional to batch storage.

            Per voxel: image/label 4+2, organ 1, two fields 8, region
            labels/depth/organ 7, source and excluded masks 1+1, connected
            labels 4, distance-transform indices 12 and distances 8 bytes.
            This admission estimate is not a hard bound; the original RSS
            guard and declared workspace reserve remain in force.
            """
            sizes = []
            for index in indices:
                row = self.raw[self._example(index)['case_id']]
                shape = row.get('shape')
                if shape is None:
                    # Older valid inventories omitted optional shape metadata.
                    # They still use the complete original serial preparation;
                    # do not invent an estimate or break checkpoint resume.
                    return None
                if (not isinstance(shape, (tuple, list)) or len(shape) != 3
                        or any(type(n) is not int or n <= 0 for n in shape)):
                    raise ValueError('Cold CPU staging requires verified full CT shape')
                sizes.append(math.prod(shape) * 48)
            if not sizes:
                raise ValueError('Cold CPU staging requires the complete nonempty source batch')
            # Count repeated patients per source too: independently active
            # source workspaces must not borrow each other's allocation.
            return sum(sizes)

        @contextmanager
        def cpu_staging(self, slots):
            """Share candidate workers across ordered independent preparations.

            One shared pool retains the saved candidate-worker capacity.
            Memory-admitted batch coordinators mostly wait for this pool;
            they are counted separately rather than removing candidate workers.
            The caller must drain batch jobs before leaving.
            """
            if type(slots) is not int or not 1 <= slots <= self.workers - 2:
                raise ValueError('Staging slots exceed the bounded coordinator ceiling')
            if not self._comparison_staging_lock.acquire(blocking=False):
                raise RuntimeError('A CPU staging scope is already active for this provider')
            try:
                with ThreadPoolExecutor(max_workers=self.workers,
                                        thread_name_prefix='comparison-candidates') as pool:
                    self._comparison_pool = pool
                    try:
                        yield
                    finally:
                        self._comparison_pool = None
            finally:
                self._comparison_staging_lock.release()

    ParallelViewProvider.__name__ = 'ParallelView' + original.__name__
    return ParallelViewProvider
