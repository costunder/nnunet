"""Execution-only admission for the unchanged one-future CPU input producer.

Below the existing RSS pressure trigger the original trim is a no-op. Avoid
waiting for the next producer's long-held coordinator lock in that case only.
Pressure reclamation, protected inputs and the actual ResourceBudget remain
unchanged. Telemetry uses a separate short lock, never the cache lock.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path
import threading
import time
from types import MethodType
import weakref

import torch

from . import v24_memory_runtime as memory

FORMAT = 'v24_bounded_CPU_prefetch_pressure_admission_runtime_v1'
_COORDINATOR = memory.MemorySafeCoordinator
_PROVIDER = memory.MemorySafeInputProvider
_BASE_COORDINATOR_INIT = _COORDINATOR.__init__
_BASE_PROVIDER_INIT = _PROVIDER.__init__
_BASE_TRIM = _COORDINATOR.trim
_BASE_GET = _PROVIDER.get
_BASE_PRESSURE = _COORDINATOR._trim_locked
_STATS = ('record_loads', 'sampled_pair_hits', 'sampled_pair_misses',
          'sampled_pairs_materialized', 'record_load_seconds',
          'materialize_seconds', 'collate_seconds', 'get_seconds', 'get_calls')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _code_sha(function):
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def runtime_contract():
    from .v23_training import prepared_native_chunks, V23Scorer
    return dict(format=FORMAT, runtime_source_sha256=_sha(__file__),
        memory_runtime_source_sha256=_sha(memory.__file__),
        unchanged_scientific_files_sha256={name: _sha(Path(__file__).parent/name)
                                           for name in memory._SCIENCE_NAMES},
        original_locked_pressure_function_sha256=_code_sha(_COORDINATOR._trim_locked),
        original_provider_get_sha256=_code_sha(_BASE_GET),
        unchanged_one_future_producer_sha256=_code_sha(prepared_native_chunks),
        unchanged_scorer_forward_sha256=_code_sha(V23Scorer.forward),
        unchanged_local_encode_sha256=_code_sha(V23Scorer._encode),
        producer_chunks_ahead=1, additional_input_futures=0,
        pressure_trigger='RSS_limit - (RSS_limit - maximum_static_provider_resident_limit)/2',
        pressure_reserve='unchanged half of declared uncached workspace; 16GiB for RSS64/resident32',
        low_RSS_admission_requires_two_actual_RSS_samples=True,
        original_ResourceBudget_check_after_trim_preserved=True,
        original_locked_pressure_reclamation_preserved=True,
        cache_protection_and_returned_input_ownership_unchanged=True,
        RSS_limit_increased=False, model_loss_views_data_or_batch_changed=False,
        timing_kind='CPU wall and scalar provider counters; not CUDA kernel or utilization measurements',
        process_RSS_admission_is_sampled_not_atomic=True)


def _coordinator_init(self, rss_bytes):
    _BASE_COORDINATOR_INIT(self, rss_bytes)
    source = Path(__file__).resolve()
    self._prefetch_runtime_state = dict(lock=threading.Lock(), source=source,
        source_stat=memory._ORIGINAL_PROVIDER._stat(source), source_sha256=_sha(source),
        resident_limits=(), admitted_RSS_limit_bytes=rss_bytes,
        active_gets={}, bound=weakref.WeakKeyDictionary(),
        provider_samples={}, counters=dict(trim_calls=0, low_RSS_fast_admissions=0,
            pressure_lock_admissions=0, pressure_lock_wait_seconds=0.,
            maximum_pressure_lock_wait_seconds=0., pressure_reclaim_seconds=0.,
            fast_admissions_during_provider_get=0, provider_get_calls=0,
            provider_get_failures=0, provider_get_wall_seconds=0.,
            local_encode_calls=0, local_encode_host_wall_seconds=0.,
            local_encode_calls_during_provider_get=0,
            local_encode_host_overlap_seconds=0.))


def _state(coordinator):
    value = getattr(coordinator, '_prefetch_runtime_state', None)
    if not isinstance(value, dict):
        raise ValueError('Prefetch runtime must be installed before coordinator creation')
    if coordinator.rss_bytes != value['admitted_RSS_limit_bytes']:
        raise ValueError('Admitted shared RSS budget changed')
    coordinator._guard_runtime()
    if memory._ORIGINAL_PROVIDER._stat(value['source']) != value['source_stat']:
        raise ValueError('Admitted prefetch runtime source changed')
    return value


def _provider_init(self, *args, **kwargs):
    _BASE_PROVIDER_INIT(self, *args, **kwargs)
    value = _state(self.coordinator)
    self._prefetch_admitted_resident_bytes = self.resident_bytes
    # Register immutable scalar budgets at construction, never inspect mutable
    # cache/ledger/provider lists from the lockless consumer admission path.
    with value['lock']:
        value['resident_limits'] += (self.resident_bytes,)


def _trim(self, *, strict=True):
    if type(strict) is not bool:
        raise TypeError('Explicit strict RSS admission required')
    value = _state(self)
    with value['lock']:
        limits = value['resident_limits']
    resident = max(limits, default=self.rss_bytes//2)
    if not 0 < resident < self.rss_bytes:
        raise ValueError('Unchanged provider residency and actual RSS budget required')
    trigger = self.rss_bytes-(self.rss_bytes-resident)//2
    first = self._rss()
    if first <= trigger and self._rss() <= trigger:
        # No cache/ledger/protected IDs are read or modified. The producer's
        # _room and post-finally admission still run, and consumer budget.check
        # immediately executes its original RSS/CUDA checks after this trim.
        with value['lock']:
            stats = value['counters']; stats['trim_calls'] += 1
            stats['low_RSS_fast_admissions'] += 1
            stats['fast_admissions_during_provider_get'] += bool(value['active_gets'])
        self._memory_stats['trim_calls'] += 1
        return
    started = time.perf_counter()
    self.lock.acquire()
    acquired = time.perf_counter()
    try:
        # Exact original pressure implementation, including GC/malloc release,
        # protected cache eviction and unchanged strict hard-limit failure.
        return self._trim_locked(strict=strict)
    finally:
        finished = time.perf_counter()
        self.lock.release()
        with value['lock']:
            stats = value['counters']; stats['trim_calls'] += 1
            stats['pressure_lock_admissions'] += 1
            waited = acquired-started
            stats['pressure_lock_wait_seconds'] += waited
            stats['maximum_pressure_lock_wait_seconds'] = max(
                stats['maximum_pressure_lock_wait_seconds'], waited)
            stats['pressure_reclaim_seconds'] += finished-acquired


def _scalar_stats(provider):
    # These scalar fields are written by the original single producer. They
    # are observational GIL snapshots; no OrderedDict/ledger is inspected.
    return {key: provider._stats[key] for key in _STATS}


def _get(self, ids, *, epoch=0):
    value = _state(self.coordinator)
    if self.resident_bytes != self._prefetch_admitted_resident_bytes:
        raise ValueError('Admitted native input residency budget changed')
    before = _scalar_stats(self); started = time.perf_counter(); success = False
    identity = id(self)
    with value['lock']:
        if identity in value['active_gets']:
            raise ValueError('Single producer required for each native input provider')
        value['active_gets'][identity] = dict(start=started, partition=self.ds.partition,
            thread_ident=threading.get_ident())
    try:
        batch = _BASE_GET(self, ids, epoch=epoch)
        success = True
        return batch
    finally:
        finished = time.perf_counter(); after = _scalar_stats(self)
        with value['lock']:
            value['active_gets'].pop(identity)
            stats = value['counters']; stats['provider_get_calls'] += 1
            stats['provider_get_failures'] += not success
            stats['provider_get_wall_seconds'] += finished-started
            value['provider_samples'][identity] = dict(partition=self.ds.partition,
                epoch=epoch, complete_ordered_observation_ids=(list(ids)
                    if isinstance(ids, (list, tuple)) else None),
                success=success, get_wall_seconds=finished-started,
                scalar_deltas={key: after[key]-before[key] for key in _STATS},
                cumulative_scalar_counters=after,
                sampling='original producer scalar snapshots; no cache-lock acquisition',
                cache_limit_bytes=self.resident_bytes, workers=self.workers)


def install_runtime(memory_module=None):
    """Install before inputs/CUDA creation; keep original class identities."""
    if memory_module is not None and memory_module is not memory:
        raise ValueError('Exact admitted memory runtime module required')
    if torch.cuda.is_initialized():
        raise RuntimeError('Prefetch runtime cannot hot-swap an initialized CUDA process')
    if _COORDINATOR._trim_locked is not _BASE_PRESSURE:
        raise ValueError('Foreign original pressure reclamation refused')
    expected = ((_COORDINATOR, '__init__', _BASE_COORDINATOR_INIT, _coordinator_init),
                (_COORDINATOR, 'trim', _BASE_TRIM, _trim),
                (_PROVIDER, '__init__', _BASE_PROVIDER_INIT, _provider_init),
                (_PROVIDER, 'get', _BASE_GET, _get))
    for owner, name, original, replacement in expected:
        if getattr(owner, name) not in (original, replacement):
            raise ValueError('Foreign prefetch lifecycle replacement refused: '+name)
    for owner, name, _, replacement in expected:
        setattr(owner, name, replacement)
    return runtime_contract()


def receipt(coordinator):
    """Small nonblocking-to-cache-lock telemetry; no cache-owned tensors."""
    value = _state(coordinator)
    with value['lock']:
        limits = value['resident_limits']
        sampled = dict(profile=copy.deepcopy(value['counters']),
            active_provider_get_count=len(value['active_gets']),
            provider_last_completed_samples=copy.deepcopy(list(value['provider_samples'].values())))
    resident = max(limits, default=coordinator.rss_bytes//2)
    return dict(format=FORMAT, runtime_source_sha256=value['source_sha256'],
        RSS_limit_bytes=coordinator.rss_bytes, actual_RSS_bytes=coordinator._rss(),
        pressure_trigger_bytes=coordinator.rss_bytes-(coordinator.rss_bytes-resident)//2,
        producer_workspace_reserve_bytes=(coordinator.rss_bytes-resident)//2,
        telemetry_acquires_coordinator_cache_lock=False,
        local_encode_overlap_scope='host invocation while provider.get is active; excludes completed get intervals and CUDA kernel claims',
        timing_kind='CPU wall and scalar snapshots; CUDA kernels not timed', **sampled)


def bind(scorer):
    """Bind telemetry and source guards after unchanged memory binding."""
    providers = tuple(scorer.providers.values())
    if not providers or any(type(p) is not _PROVIDER for p in providers):
        raise TypeError('Exact memory-safe original provider instances required')
    coordinator = providers[0].coordinator
    if any(p.coordinator is not coordinator for p in providers):
        raise ValueError('One actual shared process RSS coordinator required')
    value = _state(coordinator)
    if scorer not in coordinator._bound_scorers:
        raise ValueError('Original memory/ResourceBudget binding must precede prefetch binding')
    if scorer.budget.rss_bytes != coordinator.rss_bytes:
        raise ValueError('Prefetch runtime cannot change the actual ResourceBudget')
    inputs = getattr(scorer.geometry.memory_guard, '__self__', None)
    if inputs is None or inputs.memory_coordinator is not coordinator:
        raise ValueError('Actual original raw/geometry input ownership required')
    proof = value['source_stat']; path = str(value['source'])
    inputs._input_file_proofs[path] = proof; inputs._file_proofs[path] = proof
    inputs.guard_source()
    if scorer not in value['bound']:
        original_encode = getattr(scorer, '_encode', None)
        if not callable(original_encode):
            raise ValueError('Original local encode path required for overlap telemetry')
        def encode(module, *args, **kwargs):
            started = time.perf_counter()
            try:
                return original_encode(*args, **kwargs)
            finally:
                finished = time.perf_counter()
                with value['lock']:
                    stats = value['counters']; stats['local_encode_calls'] += 1
                    stats['local_encode_host_wall_seconds'] += finished-started
                    active = tuple(value['active_gets'].values())
                    if active:
                        stats['local_encode_calls_during_provider_get'] += 1
                        overlap_start = max(started, min(item['start'] for item in active))
                        stats['local_encode_host_overlap_seconds'] += max(0., finished-overlap_start)
        scorer._encode = MethodType(encode, scorer)
        baseline = {}
        provider_baseline = {}
        def before_forward(module, args, kwargs):
            baseline.clear(); baseline.update(receipt(coordinator)['profile'])
            provider_baseline.clear()
            provider_baseline.update({p.ds.partition: _scalar_stats(p) for p in providers})
        def after_forward(module, args, kwargs, result):
            current = receipt(coordinator)
            current['forward_profile_delta'] = {
                key: count-baseline.get(key, 0) for key, count in current['profile'].items()
                if key != 'maximum_pressure_lock_wait_seconds'}
            current['provider_forward_scalar_deltas'] = {
                p.ds.partition: {key: count-provider_baseline.get(p.ds.partition, {}).get(key, 0)
                    for key, count in _scalar_stats(p).items()} for p in providers}
            if result is not None and hasattr(result, 'workload'):
                result.workload['CPU_prefetch_runtime'] = current
        hooks = (scorer.register_forward_pre_hook(before_forward, with_kwargs=True),
                 scorer.register_forward_hook(after_forward, with_kwargs=True))
        value['bound'][scorer] = (hooks, original_encode)
    return receipt(coordinator)
