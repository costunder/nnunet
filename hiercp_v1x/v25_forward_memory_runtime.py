"""Reclaim inactive CPU ownership before the original strict forward budget.

This execution binding does not alter the frozen scientific files, active
checkpoint inputs, model, batch, workers, cache declarations or RSS/CUDA limits.
The existing input/coordinator reclamation runs only at its pressure boundary;
the original budget check then measures actual memory and may still fail.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import threading
import time
from types import MethodType

import psutil

FORMAT = 'v25_forward_inactive_RSS_reclamation_execution_v1'
SOURCE = Path(__file__).resolve()


def bind(budget, inputs):
    if int(budget.rss_bytes) != int(inputs.rss_bytes):
        raise ValueError('Exact original input and forward RSS budgets required')
    if not callable(getattr(inputs, '_rss', None)):
        raise ValueError('Admitted accelerated input reclamation required')
    previous = getattr(budget, '_v25_forward_memory_binding', None)
    if previous is not None:
        if previous['inputs'] is not inputs:
            raise ValueError('Forward budget already belongs to different inputs')
        return receipt(budget)
    original = budget.check
    lock = threading.RLock()
    profile = dict(checks=0, pressure_events=0, strict_failures=0,
        maximum_observed_RSS_bytes=0, reclamation_seconds=0., last_reclamation=None)
    binding = dict(inputs=inputs, original=original, lock=lock, profile=profile)
    budget._v25_forward_memory_binding = binding

    def check(instance):
        with lock:
            profile['checks'] += 1
            before = int(psutil.Process().memory_info().rss)
            profile['maximum_observed_RSS_bytes'] = max(
                profile['maximum_observed_RSS_bytes'], before)
            # Identical to the original provider coordinator's pressure target.
            target = instance.rss_bytes - min(instance.rss_bytes // 64, 3 * 2**30)
            if before > target:
                profile['pressure_events'] += 1
                began = time.perf_counter()
                try:
                    # Uses the original input->coordinator lock ordering. Only
                    # cache ownership is evicted; live CPU checkpoint/prefetch
                    # objects remain referenced and keep their complete values.
                    inputs._rss()
                except MemoryError:
                    profile['strict_failures'] += 1
                    raise
                after = int(psutil.Process().memory_info().rss)
                profile['reclamation_seconds'] += time.perf_counter() - began
                profile['last_reclamation'] = dict(before_RSS_bytes=before,
                    after_RSS_bytes=after, reclaimed_RSS_bytes=before-after,
                    RSS_limit_bytes=int(instance.rss_bytes), active_inputs_changed=False)
                print('V2.5 FORWARD RSS RECLAIM before_bytes=%d after_bytes=%d '
                      'limit_bytes=%d reclaimed_RSS_bytes=%d' %
                      (before, after, instance.rss_bytes, before-after), flush=True)
            try:
                return original()
            except MemoryError:
                profile['strict_failures'] += 1
                raise

    budget.check = MethodType(check, budget)
    return receipt(budget)


def receipt(budget):
    binding = budget._v25_forward_memory_binding
    with binding['lock']:
        return dict(format=FORMAT, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            RSS_limit_bytes=int(budget.rss_bytes), CUDA_limit_bytes=int(budget.cuda_bytes),
            profile=copy.deepcopy(binding['profile']), original_budget_check_preserved=True,
            active_inputs_and_checkpoint_graph_preserved=True,
            physical_batch_workers_and_declared_cache_unchanged=True,
            scientific_file_bytes_changed=False, limit_relaxed=False)
