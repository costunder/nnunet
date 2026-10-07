"""Measure preserved CPU input preparation without changing its result or cache."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path
import threading
import time
from types import FunctionType


def _with_globals(function, replacements):
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    value = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    value.__kwdefaults__ = function.__kwdefaults__
    return value


class _ModuleProxy:
    def __init__(self, original, replacements):
        self._original, self._replacements = original, replacements

    def __getattr__(self, name):
        return self._replacements[name] if name in self._replacements else getattr(self._original, name)


class _TimingRuntime:
    def __init__(self, original, provider):
        self._original = original

        def measured(name, function, *, detail=False):
            @wraps(function)
            def call(*args, **kwargs):
                timer = provider._timed_detail if detail else provider._timed_input
                return timer(name, function, *args, **kwargs)
            return call

        def patient(function, *args, **kwargs):
            # Resolve globals while the verified upper-helper context is active.
            # Earlier capture would bypass its source/lesion caches.
            replacements = {name: measured(name.lstrip('_') + '_seconds',
                                           function.__globals__[name], detail=True)
                            for name in ('_source_raw', '_lesions', '_liver_raw', '_principal_axis')}
            instrumented = _with_globals(function, replacements)
            return provider._timed_input('patient_graph_seconds', instrumented, *args, **kwargs)

        def inference(*args, **kwargs):
            function = original.cache.build_inference_sample
            replacements = {name: measured(label, function.__globals__[name])
                            for name, label in (('build_generation_specs', 'generation_specs_seconds'),
                                                ('build_prototype_graph', 'prototype_graph_seconds'))}
            original_patient = function.__globals__['build_patient_graph']
            replacements['build_patient_graph'] = lambda *a, **kw: patient(original_patient, *a, **kw)
            return provider._timed_detail('inference_assembly_inclusive_seconds',
                _with_globals(function, replacements), *args, **kwargs)

        self.cache = _ModuleProxy(original.cache, {'build_inference_sample': inference})
        self.data = _ModuleProxy(original.data, {'collate_samples':
            measured('collate_seconds', original.data.collate_samples)})

    def __getattr__(self, name):
        return getattr(self._original, name)


def timed_provider(original, log_path):
    """Record wall-time regions on the batch-building thread, once per batch.

    Timed regions are disjoint method calls; the remainder includes upper graph
    assembly and collation. Sampled views have their own timed region. Work inside local_map
    is measured as wall time, never as a sum of worker durations.
    """
    path = Path(log_path)
    write_lock = threading.Lock()

    class TimedProvider(original):
        def __init__(self, *args, **kwargs):
            self._input_timing = threading.local()
            self._timing_runtime = None
            self._timing_runtime_lock = threading.RLock()
            super().__init__(*args, **kwargs)

        def _runtime(self):
            with self._timing_runtime_lock:
                original = super()._runtime()
                if self._timing_runtime is None or self._timing_runtime._original is not original:
                    self._timing_runtime = _TimingRuntime(original, self)
                return self._timing_runtime

        def _timed_input(self, name, function, *args, **kwargs):
            receipt = getattr(self._input_timing, 'active', None)
            if receipt is None:
                return function(*args, **kwargs)
            began = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                receipt[name] = receipt.get(name, 0.) + time.perf_counter() - began

        def _timed_detail(self, name, function, *args, **kwargs):
            receipt = getattr(self._input_timing, 'details', None)
            if receipt is None:
                return function(*args, **kwargs)
            began = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                item = receipt.setdefault(name, dict(calls=0, seconds=0.))
                item['calls'] += 1
                item['seconds'] += time.perf_counter() - began

        @contextmanager
        def _upper_context(self, *args, **kwargs):
            began = time.perf_counter()
            with super()._upper_context(*args, **kwargs):
                receipt = getattr(self._input_timing, 'active', None)
                if receipt is not None:
                    name = 'upper_context_enter_seconds'
                    receipt[name] = receipt.get(name, 0.) + time.perf_counter() - began
                yield

        def _case(self, *args, **kwargs):
            return self._timed_input('case_fields_seconds', super()._case, *args, **kwargs)

        def _regions(self, *args, **kwargs):
            return self._timed_input('regions_seconds', super()._regions, *args, **kwargs)

        def _source(self, *args, **kwargs):
            return self._timed_input('source_seconds', super()._source, *args, **kwargs)

        def _candidate(self, *args, **kwargs):
            return self._timed_input('candidate_metadata_seconds', super()._candidate, *args, **kwargs)

        def _positive_candidate(self, *args, **kwargs):
            return self._timed_input('candidate_metadata_seconds', super()._positive_candidate, *args, **kwargs)

        def _local_map(self, *args, **kwargs):
            return self._timed_input('local_graphs_seconds', super()._local_map, *args, **kwargs)

        def _materialize_views(self, *args, **kwargs):
            return self._timed_input('sampled_views_seconds', super()._materialize_views, *args, **kwargs)

        def batch(self, indices, arm, epoch, training, full=False):
            import psutil
            if getattr(self._input_timing, 'active', None) is not None:
                raise RuntimeError('Nested batch construction cannot be timed as independent input')
            ids = list(indices)
            receipt = {}
            self._input_timing.active = receipt
            details = {}
            self._input_timing.details = details
            compact = getattr(self, '_compact_upper_cache', None)
            compact_before = compact.report()['kinds'] if compact is not None else None
            process = psutil.Process()
            cpu_before = process.cpu_times()
            began = time.perf_counter()
            succeeded = False
            try:
                result = super().batch(ids, arm, epoch, training, full=full)
                succeeded = True
                return result
            finally:
                elapsed = time.perf_counter() - began
                self._input_timing.active = None
                self._input_timing.details = None
                cpu_after = process.cpu_times()
                accounted = sum(receipt.values())
                row = dict(format='comparison_input_wall_timing_v1', observed_at=time.time(),
                    arm=arm, view_epoch=epoch, training=training, full129=full,
                    source_indices=ids, status='complete' if succeeded else 'failed',
                    input_seconds=elapsed, regions=receipt,
                    inclusive_details=details,
                    inclusive_details_scope='nested wall times; overlap parent regions and each other; do not sum',
                    other_assembly_views_collate_seconds=elapsed-accounted,
                    process_cpu_seconds=cpu_after.user+cpu_after.system-cpu_before.user-cpu_before.system,
                    process_cpu_scope='whole process during input; may overlap main-thread CUDA work',
                    rss_bytes=process.memory_info().rss,
                    timing_scope='CPU batch construction, excluding queue wait and memory pinning')
                if compact is not None:
                    current = compact.report()['kinds']
                    row['compact_upper_cache'] = {kind: {key: value - compact_before[kind][key]
                        for key, value in values.items()} for kind, values in current.items()}
                # Failed preparation must remain the primary error, even if its
                # diagnostic filesystem write also fails.
                try:
                    with write_lock, path.open('a', encoding='utf8') as stream:
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                except OSError:
                    if succeeded:
                        raise
                    import warnings
                    warnings.warn(f'Input failure diagnostic could not be written: {path}', RuntimeWarning)
    TimedProvider.__name__ = 'Timed' + original.__name__
    return TimedProvider
