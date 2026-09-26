"""Reuse immutable model/Adam CPU snapshots within one no-update support pass.

Rolling checkpoints remain complete, standalone files committed every batch.
No learned tensor survives into the next support pass through this cache.
"""
from contextlib import contextmanager
from unittest.mock import patch
import torch
from tools import v222_runtime_execution as backend

CachedPairLoader=backend.CachedPairLoader
progress=backend.progress
_snapshot=backend.snapshot


class AsyncSaver(backend.AsyncSaver):
    frozen_payload=None

    def versions(self):
        tensors=[*self.net.parameters(),*self.net.buffers()]
        tensors.extend(value for state in self.optimizer.state.values() for value in state.values() if torch.is_tensor(value))
        return tuple((id(t),t._version) for t in tensors)

    @contextmanager
    def fixed_support(self):
        if self.frozen_payload is not None:
            raise RuntimeError('Nested support snapshot scope')
        versions=self.versions()
        self.frozen_payload=dict(model=_snapshot(self.net.state_dict()),optimizer=_snapshot(self.optimizer.state_dict()))
        self.frozen_versions=versions
        self.support_snapshots=getattr(self,'support_snapshots',0)+1
        try:
            yield
        finally:
            try:self.flush()
            finally:
                self.frozen_payload=None
                self.frozen_versions=None

    def freeze(self,value,memo=None):
        if isinstance(value,dict) and {'format','model','optimizer','state','rng'}<=value.keys():
            if value['state']['phase'] not in ('initial_memory','refresh_memory','final_memory'):
                raise RuntimeError('Fixed snapshot cannot be used for an optimizer update')
            if self.versions()!=self.frozen_versions:
                raise RuntimeError('Model changed inside no-update support pass')
            result=_snapshot({key:item for key,item in value.items() if key not in ('model','optimizer')},memo)
            return dict(result,**self.frozen_payload)
        return _snapshot(value,memo)

    def save(self,state):
        if self.frozen_payload is None:
            return super().save(state)
        with patch.object(backend,'snapshot',self.freeze):
            return super().save(state)


@torch.no_grad()
def encode_memory(net,dataset,state,saver,batch,workers,release_unused):
    net.eval()
    with saver.fixed_support(),patch.object(backend,'CachedPairLoader',CachedPairLoader),patch.object(backend,'progress',progress):
        return backend.encode_memory(net,dataset,state,saver,batch,workers,release_unused)


@contextmanager
def installed():
    from hiercp_v222 import v1_execution
    with patch.object(v1_execution,'Saver',AsyncSaver),patch.object(v1_execution,'encode_memory',encode_memory), \
         patch(__name__+'.CachedPairLoader',backend.CachedPairLoader):
        yield
