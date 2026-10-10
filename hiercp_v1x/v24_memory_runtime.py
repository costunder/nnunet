"""RSS reclamation adapter; immutable v24 inputs and model arithmetic stay intact.

Eviction changes residency only. The original provider's final RSS check is
performed after its temporary collator frame and protected cache IDs have been
released. A genuine live-input overflow still raises at the original limit.
"""
from __future__ import annotations

import ast
import copy
import ctypes
import gc
import hashlib
import inspect
import os
from pathlib import Path
import textwrap
import time
import weakref

import psutil
import torch

from . import v24_provider as provider_module
from .contracts import canonical_hash

FORMAT = 'v24_exact_input_lifetime_RSS_reclamation_runtime_v2'
_ORIGINAL_COORDINATOR = provider_module.V24InputCoordinator
_ORIGINAL_PROVIDER = provider_module.V24InputProvider
_SCIENCE_NAMES = ('v24_factory.py', 'v24_geometry.py', 'v24_inputs.py',
                  'v24_model.py', 'v24_provider.py')
_OVERFLOW = 'Full active GT-free input exceeds shared RSS; no graph or data reduction'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _deferred_get():
    """Replace exactly the final, post-collate trim call; no tensor operation."""
    original = _ORIGINAL_PROVIDER.get
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'trim'
             and ast.dump(node.func.value) == ast.dump(ast.parse('self.coordinator', mode='eval').body)]
    if len(calls) != 1 or calls[0].args or calls[0].keywords:
        raise ValueError('Exactly one original final provider RSS call required')
    body = tree.body[0].body[0].body  # original with coordinator lock
    guarded = next((node for node in body if isinstance(node, ast.Try)), None)
    if (guarded is None or len(guarded.body) < 3
            or not isinstance(guarded.body[-1], ast.Return)
            or not isinstance(guarded.body[-2], ast.Expr)
            or guarded.body[-2].value is not calls[0]
            or not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                       and node.func.id == 'collate' for node in ast.walk(guarded.body[-4]))):
        raise ValueError('Original post-collate/finally provider lifetime changed')
    before = ast.dump(tree, include_attributes=False)
    calls[0].keywords = [ast.keyword(arg='strict', value=ast.Constant(False))]
    after = ast.dump(tree, include_attributes=False)
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), str(Path(__file__)), 'exec'), namespace)
    return namespace[original.__name__], dict(
        original_get_AST_sha256=hashlib.sha256(before.encode()).hexdigest(),
        deferred_get_AST_sha256=hashlib.sha256(after.encode()).hexdigest(),
        final_trim_expressions_changed=1, tensor_operations_changed=False,
        strict_check_after_original_finally=True)


_GET, _GET_PROOF = _deferred_get()


def memory_runtime_contract():
    directory = Path(__file__).resolve().parent
    return dict(format=FORMAT, runtime_source_sha256=_sha(__file__),
        unchanged_scientific_files_sha256={name: _sha(directory/name) for name in _SCIENCE_NAMES},
        provider_lifetime=dict(_GET_PROOF),
        limits_increased=False, graphs_or_data_reduced=False, model_or_loss_changed=False,
        candidate_chunk_and_patient_batch_changed=False, prefetch_chunks_changed=False,
        inactive_cache_eviction_only=True, active_pinned_checkpoint_inputs_preserved=True,
        allocator_release_on_RSS_pressure_only=True,
        allocator_release_coalescing='actual_RSS_minus_original_target_by_released_ledger_bytes_then_fresh_RSS',
        original_initial_pressure_collection_preserved=True,
        garbage_collection_disabled=False, fixed_eviction_batch_or_collection_cap=False,
        original_eviction_category_provider_order_and_protection_preserved=True,
        inactive_eviction_count_may_change=True, strict_RSS_check_after_coalescing=True,
        pressure_trigger='RSS_limit - (RSS_limit - per_provider_resident_limit)/2',
        pressure_headroom_basis='existing host_memory.PressureBudget execution policy; collate/prefetch/checkpoint scratch')


def _malloc_api():
    # Only glibc supplies this optional release API. Absence is explicit metadata,
    # and never changes the strict RSS limit or introduces a CPU/GPU fallback.
    if os.name != 'posix':
        return None, None
    libc = ctypes.CDLL(None)
    trim = getattr(libc, 'malloc_trim', None)
    if trim is not None:
        trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
    fields = ('arena', 'ordblks', 'smblks', 'hblks', 'hblkhd', 'usmblks',
              'fsmblks', 'uordblks', 'fordblks', 'keepcost')
    class MallInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_size_t) for name in fields]
    info = getattr(libc, 'mallinfo2', None)
    if info is not None:
        info.argtypes, info.restype = [], MallInfo
    return trim, info


class MemorySafeCoordinator(_ORIGINAL_COORDINATOR):
    def __init__(self, rss_bytes):
        super().__init__(rss_bytes)
        self._rss_process = psutil.Process()
        self._malloc_trim, self._malloc_info = _malloc_api()
        self._external = []
        self._bound_scorers = weakref.WeakKeyDictionary()
        self._runtime_source = Path(__file__).resolve()
        self._runtime_sha256 = _sha(self._runtime_source)
        self._runtime_stat = _ORIGINAL_PROVIDER._stat(self._runtime_source)
        self._memory_stats = dict(trim_calls=0, pressure_events=0, allocator_releases=0,
            gc_collected=0, host_cache_releases=0, malloc_trim_calls=0,
            view_evictions=0, record_evictions=0, geometry_evictions=0,
            raw_evictions=0, region_evictions=0, donor_evictions=0,
            released_cache_logical_bytes=0, strict_failures=0,
            allocator_release_wall_seconds=0., gc_wall_seconds=0.,
            inactive_host_release_wall_seconds=0., malloc_trim_wall_seconds=0.,
            provider_coalescing_batches=0, provider_coalesced_evictions=0,
            provider_coalesced_logical_bytes=0, external_coalescing_batches=0,
            external_coalesced_logical_bytes=0,
            maximum_observed_RSS_bytes=0, last_reclamation=None)

    def _guard_runtime(self):
        if _ORIGINAL_PROVIDER._stat(self._runtime_source) != self._runtime_stat:
            raise ValueError('Admitted memory runtime source changed')

    def _rss(self):
        value = self._rss_process.memory_info().rss
        self._memory_stats['maximum_observed_RSS_bytes'] = max(
            value, self._memory_stats['maximum_observed_RSS_bytes'])
        return value

    def _allocator_snapshot(self):
        info = self._malloc_info() if self._malloc_info is not None else None
        host = getattr(torch.cuda.memory, 'host_memory_stats', None)
        return dict(glibc=None if info is None else dict(
            arena_bytes=info.arena, used_arena_bytes=info.uordblks,
            free_arena_bytes=info.fordblks, mmap_bytes=info.hblkhd),
            pinned_host=None if not callable(host) or not torch.cuda.is_initialized() else dict(host()))

    def _release_allocators(self):
        began = time.perf_counter()
        before = self._rss(); before_allocators = self._allocator_snapshot()
        started = time.perf_counter()
        collected = gc.collect()
        self._memory_stats['gc_wall_seconds'] += time.perf_counter()-started
        host_empty = getattr(torch._C, '_host_emptyCache', None)
        host_released = callable(host_empty) and torch.cuda.is_initialized()
        if host_released:
            # The native caching host allocator releases only inactive blocks;
            # its outstanding stream events and live Tensor owners stay intact.
            started = time.perf_counter(); host_empty()
            self._memory_stats['inactive_host_release_wall_seconds'] += time.perf_counter()-started
            self._memory_stats['host_cache_releases'] += 1
        started = time.perf_counter()
        trimmed = None if self._malloc_trim is None else int(self._malloc_trim(0))
        if trimmed is not None:
            self._memory_stats['malloc_trim_wall_seconds'] += time.perf_counter()-started
            self._memory_stats['malloc_trim_calls'] += 1
        self._memory_stats['allocator_releases'] += 1
        self._memory_stats['gc_collected'] += collected
        self._memory_stats['last_reclamation'] = dict(before_RSS_bytes=before,
            after_RSS_bytes=self._rss(), before_allocators=before_allocators,
            after_allocators=self._allocator_snapshot(), gc_collected=collected,
            inactive_pinned_cache_release_called=host_released, malloc_trim_result=trimmed)
        self._memory_stats['allocator_release_wall_seconds'] += time.perf_counter()-began

    def bind_inputs(self, geometry, inputs):
        with self.lock:
            pair = (weakref.ref(geometry), weakref.ref(inputs))
            if not any(a() is geometry and b() is inputs for a, b in self._external):
                self._external.append(pair)

    def _evict_external(self):
        """Never wait for raw/geometry locks while holding coordinator.lock."""
        changed = False
        for geometry_ref, inputs_ref in self._external:
            geometry, inputs = geometry_ref(), inputs_ref()
            if geometry is not None and geometry._lock.acquire(blocking=False):
                try:
                    if geometry._memo:
                        key, item = geometry._memo.popitem(last=False)
                        geometry._bytes -= item[3]; geometry._proofs.pop(key, None)
                        self._memory_stats['geometry_evictions'] += 1
                        self._memory_stats['released_cache_logical_bytes'] += item[3]
                        del item
                        changed = True
                finally:
                    geometry._lock.release()
            if inputs is None or not inputs._lock.acquire(blocking=False):
                continue
            try:
                if inputs._raw_cache:
                    case, item = inputs._raw_cache.popitem(last=False)
                    inputs._raw_bytes -= item[1]
                    self._memory_stats['released_cache_logical_bytes'] += item[1]
                    inputs._regions.pop(case, None)
                    for key in [key for key in inputs._donors if key[0] == case]:
                        inputs._donors.pop(key)
                    sources = getattr(inputs, '_upper_sources', {})
                    for key in [key for key in sources if key[0] == case]:
                        source = sources.pop(key)
                        inputs._source_bytes -= source[1]
                        del source
                    del item
                    self._memory_stats['raw_evictions'] += 1; changed = True
                elif getattr(inputs, '_upper_sources', None):
                    _, item = inputs._upper_sources.popitem(last=False)
                    inputs._source_bytes -= item[1]
                    self._memory_stats['released_cache_logical_bytes'] += item[1]
                    del item
                    self._memory_stats['donor_evictions'] += 1; changed = True
                elif getattr(inputs, '_static_regions', None):
                    _, item = inputs._static_regions.popitem(last=False)
                    inputs._region_bytes -= item[1]
                    self._memory_stats['released_cache_logical_bytes'] += item[1]
                    del item
                    self._memory_stats['region_evictions'] += 1; changed = True
            finally:
                inputs._lock.release()
        return changed

    def _trim_locked(self, *, strict):
        if type(strict) is not bool:
            raise TypeError('Explicit strict RSS admission required')
        self._guard_runtime(); self._memory_stats['trim_calls'] += 1
        resident = max((p.resident_bytes for p in self.providers), default=self.rss_bytes//2)
        # The already declared uncached workspace is kept available before a
        # complete collate/pinned prefetch is allocated, using the existing
        # host_memory.PressureBudget half-workspace pressure trigger.
        trigger = self.rss_bytes-(self.rss_bytes-resident)//2
        target = trigger-min(self.rss_bytes//64, 3*2**30)
        if self._rss() > trigger:
            self._memory_stats['pressure_events'] += 1
            self._release_allocators()
        else:
            if strict and self._rss() > self.rss_bytes:
                self._memory_stats['strict_failures'] += 1
                raise MemoryError(_OVERFLOW)
            return
        for category in ('view', 'record'):
            current = self._rss()
            while current > target:
                # Cache ownership is removed in the original category/provider
                # order. No materialization or new tensor/patch buffer occurs.
                # Logical bytes select a release boundary only; they never
                # replace measured RSS or declare successful admission.
                deficit = current-target; freed = 0; evictions = 0; exhausted = False
                while freed < deficit:
                    changed = False
                    for provider in self.providers:
                        before = provider._ledger.bytes('all')
                        evicted = provider.evict(category)
                        if evicted:
                            released = before-provider._ledger.bytes('all')
                            if released < 0:
                                raise ValueError('Inactive cache eviction increased resident ledger bytes')
                            self._memory_stats[category+'_evictions'] += 1
                            self._memory_stats['released_cache_logical_bytes'] += released
                            freed += released; evictions += 1
                        changed |= evicted
                    if not changed:
                        exhausted = True; break
                if not evictions: break
                self._memory_stats['provider_coalescing_batches'] += 1
                self._memory_stats['provider_coalesced_evictions'] += evictions
                self._memory_stats['provider_coalesced_logical_bytes'] += freed
                self._release_allocators()
                current = self._rss()
                if exhausted: break
        current = self._rss()
        while current > target:
            deficit = current-target
            before = self._memory_stats['released_cache_logical_bytes']
            changed = False; exhausted = False
            while self._memory_stats['released_cache_logical_bytes']-before < deficit:
                if not self._evict_external():
                    exhausted = True; break
                changed = True
            if not changed: break
            self._memory_stats['external_coalescing_batches'] += 1
            self._memory_stats['external_coalesced_logical_bytes'] += self._memory_stats['released_cache_logical_bytes']-before
            self._release_allocators()
            current = self._rss()
            if exhausted: break
        if strict and self._rss() > self.rss_bytes:
            self._memory_stats['strict_failures'] += 1
            raise MemoryError(_OVERFLOW)

    def memory_runtime_receipt(self):
        with self.lock:
            self._guard_runtime()
            return dict(contract=memory_runtime_contract(), RSS_limit_bytes=self.rss_bytes,
                actual_RSS_bytes=self._rss(), profile=copy.deepcopy(self._memory_stats),
                providers=[dict(partition=p.ds.partition, protected_record_count=len(p._protected_ids),
                    protected_view_count=len(p._protected_views), resident_bytes=p._ledger.bytes('all'),
                    cache_limit_bytes=p.resident_bytes) for p in self.providers],
                glibc_malloc_trim_available=self._malloc_trim is not None,
                pinned_host_release_available=callable(getattr(torch._C, '_host_emptyCache', None)))


class MemorySafeInputProvider(_ORIGINAL_PROVIDER):
    def get(self, ids, *, epoch=0):
        if not isinstance(self.coordinator, MemorySafeCoordinator):
            raise TypeError('Shared memory runtime coordinator required')
        self.coordinator.trim()
        batch = _GET(self, ids, epoch=epoch)
        # The private original frame is now gone, including canonical records,
        # sampled payloads/private clones and its finally-released protected IDs.
        # The returned collated tensors still own every required input voxel/node.
        self.coordinator.trim()
        return batch

    def profile(self):
        return dict(super().profile(), memory_runtime=self.coordinator.memory_runtime_receipt())


def install_memory_runtime(factory_module=None):
    if factory_module is None:
        from . import v24_factory as factory_module
    allowed = ((_ORIGINAL_COORDINATOR, _ORIGINAL_PROVIDER),
               (MemorySafeCoordinator, MemorySafeInputProvider))
    if (factory_module.V24InputCoordinator, factory_module.V24InputProvider) not in allowed:
        raise ValueError('Foreign input/coordinator replacement refused')
    factory_module.V24InputCoordinator = MemorySafeCoordinator
    factory_module.V24InputProvider = MemorySafeInputProvider
    return memory_runtime_contract()


def bind_memory_runtime(scorer):
    providers = tuple(scorer.providers.values())
    if not providers or any(not isinstance(p, MemorySafeInputProvider) for p in providers):
        raise TypeError('Memory runtime must be installed before build_runtime')
    coordinators = {id(p.coordinator): p.coordinator for p in providers}
    if len(coordinators) != 1:
        raise ValueError('All providers must retain one actual process RSS budget')
    coordinator = next(iter(coordinators.values()))
    geometry = scorer.geometry
    guard = geometry.memory_guard
    inputs = getattr(guard, '__self__', None)
    if inputs is None or inputs.memory_coordinator is not coordinator:
        raise ValueError('Exact factory raw/geometry shared memory ownership required')
    if not (coordinator.rss_bytes == inputs.rss_bytes == geometry.rss_bytes):
        raise ValueError('Memory adapter cannot increase any existing RSS budget')
    coordinator.bind_inputs(geometry, inputs)
    path = Path(__file__).resolve()
    proof = _ORIGINAL_PROVIDER._stat(path)
    inputs._input_file_proofs[str(path)] = proof
    inputs._file_proofs[str(path)] = proof
    inputs.guard_source()
    if scorer not in coordinator._bound_scorers:
        budget = scorer.budget
        if budget.rss_bytes != coordinator.rss_bytes or not callable(getattr(budget, 'check', None)):
            raise ValueError('Actual original callable ResourceBudget at the unchanged RSS limit required')
        original_check = budget.check
        def memory_checked_budget():
            coordinator.trim()
            return original_check()
        budget.check = memory_checked_budget
        def before_forward(module, args, kwargs):
            budget.check()
        def after_forward(module, args, kwargs, result):
            coordinator.trim()
            if result is not None and hasattr(result, 'workload'):
                result.workload['CPU_memory_runtime'] = dict(
                    format=FORMAT, RSS_limit_bytes=coordinator.rss_bytes,
                    actual_RSS_bytes=coordinator._rss(),
                    pressure_trigger_bytes=coordinator.rss_bytes-(coordinator.rss_bytes-max(
                        p.resident_bytes for p in coordinator.providers))//2,
                    profile=copy.deepcopy(coordinator._memory_stats))
        hooks = (scorer.register_forward_pre_hook(before_forward, with_kwargs=True),
                 scorer.register_forward_hook(after_forward, with_kwargs=True))
        coordinator._bound_scorers[scorer] = (hooks, original_check)
    return coordinator.memory_runtime_receipt()
