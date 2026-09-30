"""Bounded execution overlap. No change to graph, loss, batch or update schedule."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import time
import torch
from .resident import signature,storage_bytes,check_verified


def packed_cpu_snapshot(value):
    """One D2H copy per CUDA dtype/device, independent storage for all snapshots."""
    tensors={}
    def collect(v):
        if torch.is_tensor(v):
            if v.layout!=torch.strided:raise TypeError('Checkpoint snapshot requires dense tensors')
            tensors[id(v)]=v
        elif isinstance(v,dict):
            for x in v.values():collect(x)
        elif isinstance(v,(list,tuple)):
            for x in v:collect(x)
    collect(value);copies={};groups={}
    for key,t in tensors.items():
        if t.device.type=='cpu':copies[key]=t.detach().clone()
        else:groups.setdefault((t.device,t.dtype),[]).append((key,t))
    for entries in groups.values():
        packed=torch.cat([t.detach().reshape(-1) for _,t in entries]).cpu()
        offset=0
        for key,t in entries:
            view=packed[offset:offset+t.numel()].view(t.shape);offset+=t.numel()
            # Fused Adam requires the parameter/moment memory formats to match,
            # including channels_last_3d. Values alone are not sufficient.
            copies[key]=view if t.is_contiguous() else torch.empty_like(t,device='cpu',memory_format=torch.preserve_format).copy_(view)
    def rebuild(v):
        if torch.is_tensor(v):return copies[id(v)]
        if isinstance(v,dict):return {k:rebuild(x) for k,x in v.items()}
        if isinstance(v,list):return [rebuild(x) for x in v]
        if isinstance(v,tuple):return tuple(rebuild(x) for x in v)
        return copy.deepcopy(v)
    return rebuild(value)


class CheckpointPipeline:
    """At most one writer. Never skip saves or expose mutable training tensors."""
    def __init__(self,root,mode,legacy_save,hash_fn,format_name):
        if mode not in ('synchronous','overlapped'):raise ValueError('Unknown checkpoint execution')
        self.root=root;self.mode=mode;self.legacy_save=legacy_save;self.hash_fn=hash_fn;self.format=format_name
        self.pool=None;self.pending=None;self.static={};self.parts={};self.last_saved=None
    def __enter__(self):
        if self.mode=='overlapped':self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='checkpoint-writer')
        return self
    def flush(self):
        if self.pending is not None:
            result=self.pending.result();self.pending=None;self.last_saved=result
        return self.last_saved
    def __exit__(self,*exc):
        try:self.flush()
        finally:
            if self.pool is not None:self.pool.shutdown(wait=True)
    def _write(self,payload):
        from hiercp_v222.v1_execution import atomic_torch
        start=time.perf_counter();payload['content_sha256']=self.hash_fn(payload)
        hashed=time.perf_counter();atomic_torch(self.root/'checkpoint_latest.pt',payload)
        row=dict(step=payload['state']['step'],phase=payload['state']['phase'],
                 hash_seconds=hashed-start,write_fsync_seconds=time.perf_counter()-hashed,
                 checkpoint_bytes=(self.root/'checkpoint_latest.pt').stat().st_size)
        with (self.root/'checkpoint_timing.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
        return row
    def save(self,net,optimizer,state,identity,*,wait=False):
        start=time.perf_counter()
        if self.mode=='synchronous':
            self.legacy_save(self.root,net,optimizer,state,identity)
            return dict(wait_seconds=0.,snapshot_seconds=time.perf_counter()-start,asynchronous=False)
        if self.pool is None:raise RuntimeError('Checkpoint writer context not active')
        self.flush();waited=time.perf_counter()-start;snap_start=time.perf_counter()
        from hiercp_v222.v1_execution import rng_state
        frozen={}
        for key in ('memory','plan','best'):
            original=state.get(key);sig=signature(original);cached=self.static.get(key)
            if cached is None or cached[0]!=sig:
                cached=(sig,packed_cpu_snapshot(original),original);self.static[key]=cached
            frozen[key]=cached[1]
        parts={};frozen['memory_parts']=[]
        for part in state.get('memory_parts',[]):
            key=signature(part);entry=self.parts.get(key)
            if entry is None:entry=(packed_cpu_snapshot(part),part)
            parts[key]=entry;frozen['memory_parts'].append(entry[0])
        self.parts=parts
        dynamic={k:v for k,v in state.items() if k not in frozen}
        payload=packed_cpu_snapshot(dict(format=self.format,identity=identity,model=net.state_dict(),
                                         optimizer=optimizer.state_dict(),state=dynamic,rng=rng_state()))
        payload['state'].update(frozen)
        snapshot=time.perf_counter()-snap_start
        self.pending=self.pool.submit(self._write,payload)
        if wait:self.flush()
        return dict(wait_seconds=waited,snapshot_seconds=snapshot,asynchronous=True)


class DeviceBatchCache:
    """LRU of fully validated GPU batches, bounded by explicit tensor bytes.

    A miss executes the original transfer and complete GPU validation. A hit
    verifies both source and GPU tensor identities/versions, never learned values.
    """
    def __init__(self,max_bytes,budget):
        if type(max_bytes) is not int or max_bytes<0:raise ValueError('Explicit nonnegative device cache budget required')
        if max_bytes>=budget.cuda_bytes:raise ValueError('Device cache must leave room inside the CUDA training budget')
        self.max_bytes=max_bytes;self.budget=budget;self.items=OrderedDict();self.bytes=0;self.hits=0;self.misses=0
    def get(self,cpu):
        check_verified(cpu);key=id(cpu)
        if key in self.items:
            entry=self.items.pop(key)
            if entry['source'] is not cpu:raise RuntimeError('Device cache identity collision')
            if entry['source_signature']!=signature(cpu):raise ValueError('Device cache source changed')
            check_verified(entry['gpu']);self.items[key]=entry;self.hits+=1
            return entry['gpu']
        self.misses+=1;size=storage_bytes(cpu)
        while self.items and (self.bytes+size>self.max_bytes or torch.cuda.memory_allocated()+size>self.budget.cuda_bytes):
            _,old=self.items.popitem(last=False);self.bytes-=old['bytes'];del old
        gpu=cpu.to('cuda');self.budget.check()
        # Oversize batches still execute whole; only their reuse is not retained.
        if size<=self.max_bytes and self.max_bytes:
            self.items[key]=dict(source=cpu,gpu=gpu,bytes=size,source_signature=signature(cpu),adjacency=None);self.bytes+=size
        return gpu
    def activate(self,net,cpu):
        entry=self.items.get(id(cpu))
        if entry is None or entry['adjacency'] is None:return
        states=entry['adjacency']
        if signature(states)!=entry['adjacency_signature']:raise ValueError('Cached SAGE topology mutated')
        for cache,saved in zip(net.local.adjacencies,states):
            for key,value in saved.items():setattr(cache,key,value)
    def remember(self,net,cpu):
        entry=self.items.get(id(cpu))
        if entry is None or entry['adjacency'] is not None:return
        states=[{k:getattr(c,k) for k in ('signature','matrices','transposes','edges')} for c in net.local.adjacencies]
        # Count sparse topology buffers; original COO edges already belong to input.
        seen={}
        for saved in states:
            for name in ('matrices','transposes'):
                for matrix in saved[name].values():
                    for tensor in (matrix.crow_indices(),matrix.col_indices(),matrix.values()):
                        storage=tensor.untyped_storage();seen[storage.data_ptr()]=storage.nbytes()
        extra=sum(seen.values())
        if entry['bytes']+extra>self.max_bytes:return
        while self.bytes+extra>self.max_bytes:
            victim=next(k for k in self.items if k!=id(cpu))
            old=self.items.pop(victim);self.bytes-=old['bytes'];del old
        entry['adjacency']=states;entry['adjacency_signature']=signature(states)
        entry['bytes']+=extra;self.bytes+=extra
    def clear(self):self.items.clear();self.bytes=0


def gradient_check_batched(net):
    params=[(n,p) for n,p in net.named_parameters() if p.requires_grad]
    missing=[n for n,p in params if p.grad is None]
    if missing:raise RuntimeError(f'Gradient path failure: missing={missing}')
    finite=torch.stack([torch.isfinite(p.grad).all() for _,p in params])
    if not bool(finite.all()):
        invalid=[n for (n,_),ok in zip(params,finite.cpu().tolist()) if not ok]
        raise RuntimeError(f'Gradient path failure: nonfinite={invalid}')
    return dict(all_parameter_gradients_present=True,all_parameter_gradients_finite=True)


def verify_pipeline_upgrade(previous,current):
    from .execution_upgrade import blob_hash
    if {k:v for k,v in previous.items() if k not in ('source','execution_pipeline')}!={
            k:v for k,v in current.items() if k not in ('source','execution_pipeline')}:
        raise ValueError('Pipeline upgrade cannot change model/data/support/batch/precision/resources')
    a,b=previous['source'],current['source']
    if a['core']!=b['core']:raise ValueError('Pipeline upgrade cannot change core')
    if a==b:return 'same_source_pipeline_policy_change'
    allowed={'l0_regions/training.py','l0_regions/execution_pipeline.py','l0_regions/learning_monitor.py','l0_regions/sparse.py','tools/run_fixed_regions.py',
        'l0_regions/fine_graph.py','l0_regions/final.py','l0_regions/ram_pressure.py','l0_regions/donor_learning.py','l0_regions/donor_data.py'}
    if set(a['runtime'])-set(b['runtime']) or any(a['runtime'].get(k)!=v for k,v in b['runtime'].items() if k not in allowed):
        raise ValueError('Unreviewed pipeline runtime change')
    for revision in ('4389182','10287dd','8fadbd0'):
        # This monitor did not exist in either reviewed predecessor.
        if revision!='4389182' and 'l0_regions/learning_monitor.py' in a['runtime']:continue
        if revision=='10287dd' and 'l0_regions/execution_pipeline.py' in a['runtime']:continue
        if all(v==blob_hash(revision,k) for k,v in a['runtime'].items()):return revision
    raise ValueError('Pipeline upgrade requires the reviewed patient-episode runtime')
