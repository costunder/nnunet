"""Lossless canonical graph storage; shared source tensors are stored once."""
from collections import OrderedDict
import gzip
import hashlib
import io
from pathlib import Path
import threading
import shutil
import torch
from .resources import _local_bound
from .contracts import sha

_lock=threading.Lock()
_source_cache=OrderedDict()
_cache_bytes=0
# Per-process byte budget for memoization, never a graph/data inclusion limit.
SOURCE_CACHE_BYTES=128*1024**2

def encode(value):
    stream=io.BytesIO(); torch.save(value,stream)
    return stream.getvalue()

class GraphWriter:
    def __init__(self,root,minimum_free_bytes=80*1024**3):
        self.root=Path(root); self.sources={}; self.lock=threading.Lock()
        self.minimum_free_bytes=minimum_free_bytes

    def write(self,relative,record,source_key):
        if shutil.disk_usage(self.root).free<=self.minimum_free_bytes:
            raise OSError('Graph storage disk reserve reached; partial output preserved, no data omitted')
        source={k:record[k] for k in ('source_local','source_patch')}
        raw=encode(source); digest=hashlib.sha256(raw).hexdigest()
        with self.lock:
            if source_key not in self.sources:
                source_path=f'shared_sources/{digest}.pt.gz'; file=self.root/source_path
                file.parent.mkdir(parents=True,exist_ok=True)
                if not file.exists():
                    with file.open('xb') as f:f.write(gzip.compress(raw,compresslevel=1,mtime=0))
                self.sources[source_key]={'path':source_path,'sha256':sha(file),'content_sha256':digest}
            reference=self.sources[source_key]
            if reference['content_sha256']!=digest:raise ValueError('Source tensors changed within a declared shared-source group')
        payload={k:v for k,v in record.items() if k not in ('source_local','source_patch')}
        payload['shared_source']=reference
        file=self.root/relative; file.parent.mkdir(parents=True,exist_ok=True)
        with file.open('xb') as f:f.write(gzip.compress(encode(payload),compresslevel=1,mtime=0))
        a,b=_local_bound(record['source_local']),_local_bound(record['target_local'])
        bounds={'nodes':a[0]+b[0],'edges':a[1]+b[1],
            'bytes':a[2]+b[2]+record['source_patch'].numel()*4+record['target_patch'].numel()*4}
        return {'path':relative,'sha256':sha(file),'bounds':bounds,'shared_source':reference}

def load_record(root,relative):
    global _cache_bytes
    root=Path(root).resolve(); path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Non-local graph reference')
    if not str(path).endswith('.pt.gz'):
        return torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    with gzip.open(path,'rb') as stream:record=torch.load(io.BytesIO(stream.read()),map_location='cpu',weights_only=False)
    ref=record.pop('shared_source'); source=(root/ref['path']).resolve()
    if not source.is_relative_to(root):raise ValueError('Non-local shared source reference')
    stat=source.stat(); key=(str(source),ref['sha256'],stat.st_size,stat.st_mtime_ns)
    with _lock:
        if key in _source_cache:
            value,size=_source_cache.pop(key); _source_cache[key]=(value,size)
        else:
            if sha(source)!=ref['sha256']:raise ValueError('Shared source SHA mismatch')
            with gzip.open(source,'rb') as stream:raw=stream.read()
            size=len(raw); value=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
            while _source_cache and _cache_bytes+size>SOURCE_CACHE_BYTES:
                _,(_,old_size)=_source_cache.popitem(last=False); _cache_bytes-=old_size
            if size<=SOURCE_CACHE_BYTES:_source_cache[key]=(value,size); _cache_bytes+=size
    record.update(value)
    return record
