"""Bounded CPU producer processes; model execution remains in the main process.

Four producers split the requested decode workers and prefetch four complete
physical batches in order. Their combined immutable-cache budget is 20% of
available RAM at creation. Producers persist across support/train/validation
phases so closing a loader does not discard the canonical graph cache.
One parent-process thread pins completed CPU batches ahead of GPU consumption.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
import multiprocessing
import time
from unittest.mock import patch

import torch
import torch.multiprocessing  # Register CPU tensor shared-storage reducers.
from hiercp.preparation_runtime import snapshot
from hiercp_v222.v1_local import LocalBatch
from tools.v222_runtime_cache import CachedPairDataset, CachedPairLoader

_datasets = {}
_loaders = {}
_path = None
_budget = None
_pools = {}


def producer_layout(workers):
    if workers < 0:raise ValueError('Decode worker count must be nonnegative')
    depth=min(ProcessPairLoader.producer_count,max(1,workers))
    return depth,workers//depth if workers else 0


def initialize(path, cache_budget):
    global _path, _budget
    _path, _budget = path, cache_budget


def produce(partition, row_ids, epoch, workers):
    if _path is None:
        raise RuntimeError('CPU producer was not initialized')
    if partition not in _datasets:
        dataset = CachedPairDataset(_path, partition)
        dataset.store.cache.budget = _budget
        dataset.budget = _budget
        _datasets[partition] = (dataset, {r['id']:i for i,r in enumerate(dataset.rows)})
    dataset, indices = _datasets[partition]
    current = _loaders.get(partition)
    if current is None or current.workers != workers:
        if current is not None:
            current.close()
        current = _loaders[partition] = CachedPairLoader(dataset, workers)
    torch.set_num_threads(max(1, workers))
    # Pinning belongs to the GPU process; children never initialize CUDA.
    with patch.object(LocalBatch, 'pin_memory', lambda self:self):
        return current.make([indices[key] for key in row_ids], epoch)


class ProcessPairLoader:
    producer_count = 4

    def __init__(self, dataset, workers):
        if workers < 0:
            raise ValueError('Decode worker count must be nonnegative')
        cases = {row['case_id'] for row in dataset.rows}
        partitions = [key for key in ('inner_train','inner_val')
                      if cases and cases.issubset(dataset.meta['split'][key])]
        if len(partitions) != 1:
            raise ValueError('Each loader must use one existing inner partition')
        self.dataset, self.workers, self.partition = dataset, workers, partitions[0]
        self.depth,self.decode_width = producer_layout(workers)
        key = (str(dataset.path), self.depth)
        if key not in _pools:
            total_budget = int(snapshot()['available_memory_bytes'] * .20)
            _pools[key] = ProcessPoolExecutor(max_workers=self.depth,
                mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
                initargs=(str(dataset.path), total_budget//self.depth))
        self.pool = _pools[key]
        self.pending = deque()
        self.closed = False
        self.pin_pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='graph-pin-prefetch')

    def submit(self, ids, epoch, workers):
        if self.closed:
            raise RuntimeError('Loader is closed')
        indices = list(map(int, ids))
        keys = [self.dataset.rows[i]['id'] for i in indices]
        return self.pool.submit(produce, self.partition, keys, epoch, workers), indices

    @staticmethod
    def receive(job):
        future, indices = job
        value = future.result()  # Propagate decode, integrity and IPC errors.
        # Child indices are in the complete partition; caller may be a DEBUG
        # diagnostic view. Only row-address metadata is remapped, never data.
        value.indices = torch.tensor(indices)
        return value.pin_memory()

    def make(self, indices, epoch):
        return self.receive(self.submit(indices, epoch, self.workers))

    def batches(self, groups, epoch=0):
        if self.pending:
            raise RuntimeError('Previous prefetch must be drained before reuse')
        iterator = iter(groups)
        # Split decode threads across active producers, not across samples.
        width = self.decode_width
        for _ in range(self.depth):
            ids = next(iterator, None)
            if ids is not None:
                self.pending.append(self.pin_pool.submit(self.receive,self.submit(ids, epoch, width)))
        while self.pending:
            job = self.pending.popleft()
            value = job.result()
            ids = next(iterator, None)
            if ids is not None:
                self.pending.append(self.pin_pool.submit(self.receive,self.submit(ids, epoch, width)))
            yield value

    def close(self):
        self.closed = True
        try:
            while self.pending:
                self.pending.popleft().result()
        finally:
            self.pin_pool.shutdown(wait=True)


def close_producers():
    pools = list(_pools.values())
    _pools.clear()
    for pool in pools:
        pool.shutdown(wait=True)


def calibrate_workers(dataset,indices,root):
    """Measure the production prefetch path, not single-producer make().

    Same complete diagnostic graphs at each candidate width; four batches in
    both cold and warm passes. Pool teardown bounds memory across candidates.
    This is a loader-only measurement, not overlapped GPU throughput.
    """
    from hiercp_v222.v1_training import write_new,emit
    reports=[]
    for workers in (0,2,4,8):
        close_producers()
        loader=ProcessPairLoader(dataset,workers)
        report=dict(workers=workers,graphs_per_batch=len(indices),batches_per_pass=4,
                    producers=loader.depth,decode_threads_per_producer=loader.decode_width,
                    debug_calibration=True,path='ProcessPairLoader.batches')
        try:
            for key in ('seconds','warm_seconds'):
                start=time.perf_counter();count=0
                for payload in loader.batches([indices]*4,0):
                    count+=1
                    report['nodes_per_batch']=int(payload.graph.num_nodes)
                    report['edges_per_batch']=int(payload.graph.num_edges)
                    del payload
                if count!=4:raise RuntimeError('Incomplete calibration stream')
                report[key]=time.perf_counter()-start
            reports.append(report);emit(stage='paired_loader_calibration',**report)
        finally:
            loader.close();close_producers()
    selected=min(reports,key=lambda r:r['warm_seconds'])['workers']
    write_new(root/'loader_calibration.json',dict(reports=reports,selected=selected,
        cache_policy='one producer pool at a time; combined budget 20% available RAM',
        worker_kind='production process prefetch; parent pin thread',gpu_overlap_measured=False))
    return selected


@contextmanager
def installed():
    """Called inside the existing execution backend's installation context."""
    from hiercp_v222 import v1_execution, v1_training, v1_cache
    from tools import v222_runtime_execution
    modules = (v1_execution, v1_training, v1_cache)
    bindings = [module.PairLoader for module in modules]
    previous = v222_runtime_execution.CachedPairLoader
    for module in modules:
        module.PairLoader = ProcessPairLoader
    v222_runtime_execution.CachedPairLoader = ProcessPairLoader
    try:
        with patch.object(v1_training,'calibrate_workers',calibrate_workers):
            yield
    finally:
        try:
            close_producers()
        finally:
            for module, prior in zip(modules, bindings):
                module.PairLoader = prior
            v222_runtime_execution.CachedPairLoader = previous
