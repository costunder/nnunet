"""Explicitly bounded, verified resident batches for diagnostic reuse.

No eviction, sample dropping or unchecked repair. A mutated retained object is
rejected. Callers must not bypass PyTorch version tracking with .data/raw storage.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import torch
from .data import VerifiedItem,collate


def load(path,expected_binding):
    """Deserialize the exact bytes just hashed, without a second file read.

    All original binding, receipt and structural checks still run in VerifiedItem.
    This runtime reader does not modify preparation or any stored cache record.
    """
    path=Path(path)
    manifest=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if manifest['binding']!=expected_binding:raise ValueError('Different record/view/geometry/partition model')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['sha256']:raise ValueError('Corrupt region cache')
    item=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if item['binding']!=expected_binding:raise ValueError('Cache identity mismatch')
    return VerifiedItem(item)


def signature(value):
    if isinstance(value,torch.Tensor):
        return ('tensor',id(value),value._version,tuple(value.shape),str(value.dtype),str(value.device))
    if hasattr(value,'to_dict'):return signature(value.to_dict())
    if isinstance(value,dict):return tuple((str(k),signature(v)) for k,v in sorted(value.items(),key=lambda p:str(p[0])) if k!='_verified_signature')
    if isinstance(value,(tuple,list)):return tuple(signature(v) for v in value)
    if hasattr(value,'__dict__'):return signature(vars(value))
    return value


def storage_bytes(values):
    seen={}
    def walk(v):
        if isinstance(v,torch.Tensor):
            storage=v.untyped_storage();seen[(str(v.device),storage.data_ptr())]=storage.nbytes()
        elif hasattr(v,'to_dict'):walk(v.to_dict())
        elif isinstance(v,dict):
            for k,x in v.items():
                if k!='_verified_signature':walk(x)
        elif isinstance(v,(list,tuple)):
            for x in v:walk(x)
        elif hasattr(v,'__dict__'):walk(vars(v))
    walk(values)
    return sum(seen.values())


def mark_verified(batch):
    from .encoder import validate_batch
    validate_batch(batch)
    batch._verified_signature=signature(batch)
    return batch


def check_verified(batch):
    # Missing evidence must not certify the current, possibly changed contents.
    if not hasattr(batch,'_verified_signature'):
        raise ValueError('Verified resident batch signature missing; reload through verified load/collate')
    if batch._verified_signature!=signature(batch):raise ValueError('Verified resident batch mutated')
    return True


class RegionResidentCache:
    def __init__(self,*,max_tensor_bytes,workers,compatible_sources=None):
        if type(max_tensor_bytes)!=int or max_tensor_bytes<=0:raise ValueError('Explicit resident tensor budget required')
        if type(workers)!=int or workers<1:raise ValueError('Explicit I/O workers required')
        self.workers=workers;self.max_tensor_bytes=max_tensor_bytes;self.entries={};self.hits=0;self.misses=0
        self.compatible_sources=None if compatible_sources is None else frozenset(compatible_sources)

    def get(self,paths,bindings,indices,*,device='cpu'):
        paths=[Path(p) for p in paths]
        if not paths or len(paths)!=len(bindings) or len(paths)!=len(indices):raise ValueError('Resident request coverage')
        identities=tuple((str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns,
            p.with_suffix('.json').stat().st_mtime_ns) for p in paths)
        key=(identities,signature(bindings),tuple(indices),str(device))
        if key in self.entries:
            self.hits+=1;batch=self.entries[key]
            check_verified(batch)
            return batch
        self.misses+=1
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            items=list(pool.map(lambda pair:load(*pair),zip(paths,bindings)))
        batch=collate(items,indices,compatible_sources=self.compatible_sources)
        if str(device)!='cpu':batch=batch.to(device)
        check_verified(batch)
        if storage_bytes([*self.entries.values(),batch])>self.max_tensor_bytes:
            raise MemoryError('Resident tensor budget exceeded; no sample dropped')
        self.entries[key]=batch
        return batch
