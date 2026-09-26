"""DEBUG candidate: isolate threaded CPU graph preparation from GPU Python.

One persistent CPU producer keeps the bounded canonical cache and the existing
eight-way (or requested count) per-sample worker pool. Its tensors cross the
process boundary through PyTorch shared storage. Only the main process pins
memory or invokes CUDA. One batch is prefetched, never an unbounded queue.
"""
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import multiprocessing
from unittest.mock import patch

import torch
import torch.multiprocessing  # Register tensor shared-storage reducers.

from hiercp_v222.v1_local import LocalBatch
from tools.v222_runtime_cache import CachedPairDataset, CachedPairLoader

_loader = None


def initialize(path, partition, row_ids, workers, cache_budget=None):
    global _loader
    torch.set_num_threads(workers or 1)
    dataset = CachedPairDataset(path, partition)
    if cache_budget is not None:
        dataset.store.cache.budget=cache_budget
        dataset.budget=cache_budget
    by_id = {row['id']: row for row in dataset.rows}
    dataset.rows = [by_id[key] for key in row_ids]
    _loader = CachedPairLoader(dataset, workers)


def produce(indices, epoch):
    if _loader is None:
        raise RuntimeError('CPU producer was not initialized')
    with patch.object(LocalBatch, 'pin_memory', lambda self: self):
        return _loader.make(indices, epoch)


class ProcessPairLoader:
    producer_count=1
    def __init__(self, dataset, workers):
        cases = {row['case_id'] for row in dataset.rows}
        partitions = [key for key in ('inner_train', 'inner_val')
                      if cases and cases.issubset(dataset.meta['split'][key])]
        if len(partitions) != 1:
            raise ValueError('A producer must belong to exactly one existing inner partition')
        from hiercp.preparation_runtime import snapshot
        producers=self.producer_count
        if producers<1 or workers<producers:
            raise ValueError('At least one decode thread per producer is required')
        cache_budget=int(snapshot()['available_memory_bytes']*.20)//producers
        self.depth=producers
        self.pool = ProcessPoolExecutor(max_workers=producers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(str(dataset.path), partitions[0], [row['id'] for row in dataset.rows], workers//producers, cache_budget))

    def batches(self, groups, epoch=0):
        iterator = iter(groups)
        futures=deque()
        for _ in range(self.depth):
            ids=next(iterator,None)
            if ids is not None:futures.append(self.pool.submit(produce,list(ids),epoch))
        while futures:
            batch=futures.popleft().result()
            ids=next(iterator,None)
            if ids is not None:futures.append(self.pool.submit(produce,list(ids),epoch))
            yield batch.pin_memory()

    def close(self):
        self.pool.shutdown(wait=True)
