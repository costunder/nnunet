"""Execution backend: snapshot once, serialize concurrently, retain exact state.

Each requested checkpoint is committed; a single writer applies backpressure.
GPU/optimizer state is frozen on CPU before the training loop may mutate it.
The durable step is never confused with the submitted/pending step.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time

import torch
from hiercp_v222 import v1_execution as original
from hiercp_v222 import v1_training as training
from hiercp_v222 import v1_cache as cache_module
from hiercp_v222.v1_cache import PairLoader, emit
from tools.v222_runtime_cache import CachedPairDataset, CachedPairLoader


def snapshot(value, memo=None):
    """Freeze mutable CPU tensors too; reuse a CPU copy for repeated objects."""
    memo = {} if memo is None else memo
    if torch.is_tensor(value):
        key = id(value)
        if key not in memo:
            memo[key] = value.detach().to('cpu', copy=True)
        return memo[key]
    if isinstance(value, dict):
        return {k: snapshot(v, memo) for k, v in value.items()}
    if isinstance(value, list):
        return [snapshot(v, memo) for v in value]
    if isinstance(value, tuple):
        return tuple(snapshot(v, memo) for v in value)
    return value


class AsyncSaver(original.Saver):
    instances = []
    execution_policy = None

    def __init__(self, root, net, optimizer, identity):
        super().__init__(root, net, optimizer, identity)
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='checkpoint-writer')
        self.future = None
        self.committed = None
        self.receipt_lock = threading.Lock()
        self.closed = False
        self.clock = None
        self.last_phase = None
        self.phase_seconds = {}
        (self.root/'runtime_log_schema.json').write_text(json.dumps(dict(
            durable_checkpoint='checkpoint_status.json; updated only after atomic checkpoint commit',
            runtime_events='runtime_events.jsonl; checkpoint_step is committed, checkpoint_pending is separate',
            legacy_step_timings='step_timings.jsonl and first_optimizer_step.json: checkpoint_step is submitted step',
            legacy_epoch_seconds='epoch_NNN.json and metrics.csv: seconds is optimization time only',
            complete_epoch_seconds='runtime_events.jsonl epoch_complete: optimization + support refresh + validation',
            abrupt_process_loss='the last pending write may be lost; resume from the last atomic checkpoint'),indent=2),encoding='utf-8')
        self.instances.append(self)

    def flush(self):
        if self.future is not None:
            self.future.result()  # Never suppress disk/serialization failures.

    def close(self):
        if not self.closed:
            try:
                self.flush()
            finally:
                self.writer.shutdown(wait=True)
                self.closed = True

    def save(self, state):
        if self.closed:
            raise RuntimeError('Checkpoint writer is closed')
        began = time.perf_counter()
        if self.clock is None:
            self.phase_seconds = dict(state.get('runtime_phase_seconds', {}))
        else:
            self.phase_seconds[self.last_phase] = self.phase_seconds.get(self.last_phase, 0.) + began - self.clock
        self.clock = began
        self.last_phase = f"{state['epoch']}:{state['phase']}"
        state['runtime_phase_seconds'] = dict(self.phase_seconds)
        self.flush()  # At most one immutable CPU snapshot is queued.
        wait_seconds = time.perf_counter() - began
        payload = snapshot(dict(format=original.RESUME_FORMAT, **self.identity,
            execution_policy=self.execution_policy, state=state, model=self.net.state_dict(),
            optimizer=self.optimizer.state_dict(), rng=original.rng_state()))
        snapshot_seconds = time.perf_counter() - began - wait_seconds
        receipt = dict(phase=state['phase'], epoch=state['epoch'] + 1, step=state['step'],
            next_batch=state.get('next_batch', 0), memory_completed=state.get('memory_next', 0),
            snapshot_seconds=snapshot_seconds, writer_wait_seconds=wait_seconds,
            path=str(self.root / 'checkpoint_latest.pt'))

        def commit():
            start = time.perf_counter()
            original.atomic_torch(self.root / 'checkpoint_latest.pt', payload)
            done = dict(**receipt, saved_at=time.time(), seconds=time.perf_counter()-start,
                        bytes=(self.root / 'checkpoint_latest.pt').stat().st_size, durable=True)
            temporary = self.root / 'checkpoint_status.tmp'
            temporary.write_text(json.dumps(done, indent=2), encoding='utf-8')
            original.os.replace(temporary, self.root / 'checkpoint_status.json')
            with self.receipt_lock:
                self.committed = done
            with (self.root / 'checkpoint_durability.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(done) + '\n')
            return done

        self.future = self.writer.submit(commit)
        return dict(**receipt, seconds=time.perf_counter()-began, pending=True)

    def stop_requested(self):
        stopping = super().stop_requested()
        if stopping:
            self.flush()
        return stopping


@torch.no_grad()
def encode_memory(net, dataset, state, saver, batch, workers, release_unused):
    """Same order, weights, precision and coverage; no growing-prefix GPU clone."""
    net.eval()
    net.local.dense_batch_size = batch
    done = 0 if state.get('memory_work') is None else state['memory_next']
    embeddings = torch.empty((len(dataset), 128), device='cuda', dtype=torch.float32)
    if done:
        embeddings[:done] = state['memory_work']
    loader = CachedPairLoader(dataset, workers)
    started = time.perf_counter()
    initial = done
    try:
        for payload in loader.batches((ids for ids in training.contiguous(dataset, batch) if ids[0] >= done), epoch=0):
            ids = payload.indices.tolist()
            if ids != list(range(done, done + len(ids))):
                raise RuntimeError('Non-contiguous memory resume')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                values = net.local(payload.cuda(non_blocking=True))
            embeddings[ids] = values.float()
            done += len(ids)
            del values, payload
            state['memory_work'] = embeddings[:done]
            state['memory_next'] = done
            receipt = saver.save(state)  # Copies the prefix before future writes.
            if release_unused:
                torch.cuda.empty_cache()
            progress(stage='support_memory', completed=done, total=len(dataset), physical_batch=batch,
                graphs_per_second=(done-initial)/(time.perf_counter()-started),
                checkpoint_seconds=receipt['seconds'], **original.memory_metrics())
            if saver.stop_requested():
                saver.flush()
                return False
    finally:
        loader.close()
    if done != len(dataset):
        raise RuntimeError('Incomplete memory')
    state['memory'] = original.memory_metadata(dataset, embeddings)
    state['memory_work'] = None
    state['memory_next'] = 0
    return True


def progress(**value):
    active = [s for s in AsyncSaver.instances if not s.closed]
    if active and value.get('stage') in ('optimization', 'support_memory'):
        saver = active[-1]
        with saver.receipt_lock:
            committed = saver.committed
        value['checkpoint_step'] = committed['step'] if committed else None
        value['checkpoint_memory_completed'] = committed['memory_completed'] if committed else 0
        value['checkpoint_durable'] = committed is not None
        value['checkpoint_pending'] = saver.future is not None and not saver.future.done()
    if active and value.get('stage') == 'epoch_complete':
        saver = active[-1]
        epoch = int(value['epoch']) - 1
        parts = {phase: saver.phase_seconds.get(f'{epoch}:{phase}', 0.)
                 for phase in ('optimization', 'refresh_memory', 'validation')}
        value['legacy_optimization_seconds'] = value['seconds']
        value['seconds'] = sum(parts.values())
        value['runtime_phase_seconds'] = parts
        value['time_scope'] = 'active execution segments, excluding initial memory and paused downtime'
    if active:
        with (active[-1].root / 'runtime_events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value) + '\n')
    emit(**value)


@contextmanager
def installed(policy):
    """Explicit backend installation; preserved model/cache code is untouched."""
    bindings = dict(PairDataset=original.PairDataset, PairLoader=original.PairLoader, Saver=original.Saver,
                    encode_memory=original.encode_memory, emit=original.emit)
    training_loader = training.PairLoader
    cache_bindings = cache_module.PairDataset, cache_module.PairLoader
    cache_module.PairDataset, cache_module.PairLoader = CachedPairDataset, CachedPairLoader
    original.PairDataset = CachedPairDataset
    original.PairLoader = CachedPairLoader
    training.PairLoader = CachedPairLoader
    original.Saver = AsyncSaver
    original.encode_memory = encode_memory
    original.emit = progress
    AsyncSaver.execution_policy = policy
    start = len(AsyncSaver.instances)
    try:
        yield
    finally:
        failures = []
        try:
            for saver in AsyncSaver.instances[start:]:
                try:
                    saver.close()
                except Exception as error:
                    failures.append(error)
        finally:
            for name, value in bindings.items():
                setattr(original, name, value)
            training.PairLoader = training_loader
            cache_module.PairDataset, cache_module.PairLoader = cache_bindings
        if failures:
            raise RuntimeError('Checkpoint writer failed during shutdown') from failures[0]
