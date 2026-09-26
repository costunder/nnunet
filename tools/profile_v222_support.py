"""Actual-cohort support profiling, not optimization training or accuracy.

Default covers all 11,279 inner-train observations, preserving original order,
physical batch32, complete model/graphs and per-batch checkpoint durability.
--debug-batches is explicitly a separate short diagnostic, never production.
"""
import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend-file',type=Path)
    p.add_argument('--debug-batches',type=int)
    p.add_argument('--release-unused',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--loader',choices=('thread','process','serial-prefetch'),default='thread')
    p.add_argument('--producer-processes',type=int,choices=(1,2),default=1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from hiercp_v222.v1_cache import configuration
    from hiercp_v222.v1_cache import provenance
    from hiercp_v222.contracts import sha
    from hiercp_v222.v1_local import model,LocalBatch
    from hiercp_v222.training import configure_runtime
    from tools.v222_runtime_cache import CachedPairDataset,CachedPairLoader
    if a.loader=='process':
        from tools.v222_process_loader import ProcessPairLoader,close_producers
        ProcessPairLoader.producer_count=a.producer_processes
        loader_base=ProcessPairLoader
    elif a.loader=='serial-prefetch':
        class SerialLoader(CachedPairLoader):
            def batches(self,groups,epoch=0):
                for ids in groups:yield self.make(ids,epoch)
        loader_base=SerialLoader
    else:loader_base=CachedPairLoader
    from tools import v222_runtime_execution as backend
    import hiercp.model as implementation
    if a.backend_file:
        spec=importlib.util.spec_from_file_location('profile_preserved_backend',a.backend_file)
        backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
    cfg,base=configuration();configure_runtime(base,42);torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(9e9/torch.cuda.get_device_properties(0).total_memory)
    implementation.EDGE_ATTENTION_WORKSPACE_BYTES=256*1024**2
    data=CachedPairDataset(ROOT/'work/v222_v1_recovered2_training_20260924/cache/index_execution_r6_final.json','inner_train')
    original_count=len(data)
    if a.debug_batches is not None:
        if a.debug_batches<1:raise ValueError('Positive DEBUG batches required')
        data.rows=data.rows[:32*a.debug_batches] # Explicit diagnostic only.
    checkpoint=ROOT/'work/v222_runtime_20260925_DEBUG/optimized256/checkpoint_latest.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    net=model(cfg,base).cuda();net.load_state_dict(saved['model'])
    optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=True)
    optimizer.load_state_dict(saved['optimizer']);del saved
    evidence=dict(debug=True,full_training=False,real_CT=True,
        requested_observations=len(data),total_observations=original_count,
        cache_sha256=sha(data.path),checkpoint_sha256=sha(checkpoint),source_identity=provenance(),
        physical_batch=32,workers=8,loader=a.loader,release_unused=a.release_unused,
        GPU=torch.cuda.get_device_name(),input_shape=[32,1,48,48,48],
        full_model=base['model'],model_parameters=sum(p.numel() for p in net.parameters()))
    (a.output/'started.json').write_text(json.dumps(evidence,indent=2))
    rows=[];events={};hooks=[];timing={}
    def event():
        value=torch.cuda.Event(enable_timing=True);value.record();return value
    def install_hooks(module,name):
        hooks.append(module.register_forward_pre_hook(lambda *args:events.setdefault(name,[]).append([event(),None])))
        hooks.append(module.register_forward_hook(lambda *args:events[name][-1].__setitem__(1,event())))
    install_hooks(net.local,'L0')
    install_hooks(net.local.dense_encoder,'CNN')
    for block in net.local.blocks:install_hooks(block,'GNN')
    original_cuda=LocalBatch.cuda;empty_cache=torch.cuda.empty_cache
    def transfer(self,non_blocking=False):
        start=event();value=original_cuda(self,non_blocking=non_blocking)
        events['H2D']=[[start,event()]];return value
    def release():
        start=time.perf_counter();empty_cache();timing['allocator_release_seconds']=time.perf_counter()-start
    class Loader(loader_base):
        def batches(self,groups,epoch=0):
            iterator=super().batches(groups,epoch)
            while True:
                start=time.perf_counter()
                try:value=next(iterator)
                except StopIteration:return
                timing['loader_wait_seconds']=time.perf_counter()-start
                timing['batch_nodes']=int(value.graph.num_nodes);timing['batch_edges']=int(value.graph.num_edges)
                timing['batch_start']=start
                yield value
    saver=backend.AsyncSaver(a.output,net,optimizer,dict(debug=True))
    state=dict(phase='initial_memory',epoch=0,step=6,memory_work=None,memory_next=0)
    def progress(**values):
        torch.cuda.synchronize()
        metrics={name+'_seconds':sum(start.elapsed_time(end)/1000 for start,end in pairs) for name,pairs in events.items()}
        metrics['batch_wall_seconds']=time.perf_counter()-timing['batch_start']
        record=dict(**values,**{k:v for k,v in timing.items() if k!='batch_start'},**metrics)
        record['other_L0_seconds']=metrics['L0_seconds']-metrics['CNN_seconds']-metrics['GNN_seconds']
        rows.append(record);events.clear()
        with (a.output/'profile.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps({k:record[k] for k in ('completed','total','batch_wall_seconds','loader_wait_seconds','CNN_seconds','GNN_seconds','other_L0_seconds','checkpoint_seconds')}),flush=True)
    started=time.perf_counter()
    try:
        with patch.object(backend,'CachedPairLoader',Loader),patch.object(backend,'progress',progress),patch.object(LocalBatch,'cuda',transfer),patch.object(torch.cuda,'empty_cache',release):
            completed=backend.encode_memory(net,data,state,saver,32,8,a.release_unused)
        saver.close()
        if a.loader=='process':close_producers()
        elapsed=time.perf_counter()-started
        values=state['memory']['embeddings'] if completed else state['memory_work']
        torch.save(values.cpu(),a.output/'embeddings_DEBUG.pt')
        result=dict(debug=True,full_support=completed and len(data)==original_count,
            paused=not completed,observations=len(values),total_observations=original_count,
            full_training=False,model_parameters=sum(p.numel() for p in net.parameters()),physical_batch=32,
            loader=a.loader,producer_processes=a.producer_processes if a.loader=='process' else 0,
            precision='bfloat16',workspace_mib=256,allocator_cap_bytes=9000000000,release_unused=a.release_unused,workers=8,
            gpu=torch.cuda.get_device_name(),seconds=elapsed,
            seconds_by_component={k:sum(r.get(k,0) for r in rows) for k in ('loader_wait_seconds','H2D_seconds','CNN_seconds','GNN_seconds','other_L0_seconds','checkpoint_seconds','allocator_release_seconds')},
            rows=len(rows),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:
        saver.close()
        if a.loader=='process':close_producers()
        for handle in hooks:handle.remove()


if __name__=='__main__':main()
