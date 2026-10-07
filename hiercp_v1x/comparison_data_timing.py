"""Measure preserved CPU input preparation without changing its result or cache."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time


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
            super().__init__(*args, **kwargs)

        def _timed_input(self, name, function, *args, **kwargs):
            receipt = getattr(self._input_timing, 'active', None)
            if receipt is None:
                return function(*args, **kwargs)
            began = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                receipt[name] = receipt.get(name, 0.) + time.perf_counter() - began

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
                cpu_after = process.cpu_times()
                accounted = sum(receipt.values())
                row = dict(format='comparison_input_wall_timing_v1', observed_at=time.time(),
                    arm=arm, view_epoch=epoch, training=training, full129=full,
                    source_indices=ids, status='complete' if succeeded else 'failed',
                    input_seconds=elapsed, regions=receipt,
                    other_assembly_views_collate_seconds=elapsed-accounted,
                    process_cpu_seconds=cpu_after.user+cpu_after.system-cpu_before.user-cpu_before.system,
                    process_cpu_scope='whole process during input; may overlap main-thread CUDA work',
                    rss_bytes=process.memory_info().rss,
                    timing_scope='CPU batch construction, excluding queue wait and memory pinning')
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
