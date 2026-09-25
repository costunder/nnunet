"""DEBUG: time one real full-size cached batch, preserving the entire model.

Uses verified real support prefix, not final/full support. Execution workspace
changes only edge chunking; all nodes/edges, precision, batch and loss are fixed.
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import psutil
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workspace-mib',type=int,required=True)
    p.add_argument('--reference',type=Path)
    p.add_argument('--group-rank',type=int,default=0,help='DEBUG recipient group, ranked by full graph size')
    p.add_argument('--operators',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from hiercp_v222.v1_cache import PairDataset,PairLoader,configuration,provenance
    from hiercp_v222.v1_local import model,support_for_recipient
    from hiercp_v222.v1_execution import memory_metadata,restore_rng,rng_state,Saver,memory_metrics
    from hiercp_v222.model import supervised_loss
    from hiercp_v222.training import configure_runtime
    from hiercp_v222.contracts import sha
    import hiercp.model as original
    cache=ROOT/'work/v222_v1_recovered2_training_20260924/cache/index_execution_r6_final.json'
    saved=torch.load(ROOT/'work/v222_v1_resumable_training_20260924_r6/checkpoint_latest.pt',weights_only=False,map_location='cpu')
    cfg,base=configuration()
    if saved['source_identity']!=provenance() or saved['cache_sha256']!=sha(cache):raise ValueError('Checkpoint provenance differs')
    configure_runtime(base,42);torch.set_num_threads(8)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(9e9/total)
    original.EDGE_ATTENTION_WORKSPACE_BYTES=a.workspace_mib*1024**2
    dataset=PairDataset(cache,'inner_train');prefix=saved['state']['memory_next']
    class Prefix:
        rows=dataset.rows[:prefix]
        meta=dataset.meta
    memory=memory_metadata(Prefix,saved['state']['memory_work'].cuda())
    by_group={}
    for i,r in enumerate(dataset.rows):by_group.setdefault(r['patient_group'],[]).append(i)
    for ids in by_group.values():ids.sort(key=lambda i:dataset.rows[i]['bounds']['edges'],reverse=True)
    ranked=sorted(by_group,key=lambda g:sum(dataset.rows[i]['bounds']['edges'] for i in by_group[g][:32]),reverse=True)
    group=ranked[a.group_rank]
    ids=by_group[group][:32]
    net=model(cfg,base).cuda();net.load_state_dict(saved['model']);net.local.dense_batch_size=32;net.train()
    del saved
    support=support_for_recipient(memory,group)
    started=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):plan=net.fit_support_clusters(*support)
    torch.cuda.synchronize();plan_seconds=time.perf_counter()-started
    loader=PairLoader(dataset,8)
    started=time.perf_counter()
    try:cpu=loader.make(ids,epoch=0)
    finally:loader.close()
    loading=time.perf_counter()-started
    classes=torch.tensor([r['target'] for r in dataset.rows],device='cuda')
    counts=torch.bincount(classes,minlength=2).float();weights=counts.sum()/(2*counts);targets=classes[ids]
    optimizer=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],weight_decay=base['training']['weight_decay'],fused=True)
    info=dict(debug=True,full_training=False,actual_CT=True,workspace_mib=a.workspace_mib,allocator_cap_bytes=9000000000,
        model_parameters=sum(p.numel() for p in net.parameters()),physical_batch=32,workers=8,
        nodes=int(cpu.graph.num_nodes),edges=int(cpu.graph.num_edges),support_prefix=prefix,
        full_support=len(dataset),cpu_logical=psutil.cpu_count(),ram_available=psutil.virtual_memory().available,
        gpu=torch.cuda.get_device_name(),loading_seconds=loading,cluster_fit_seconds=plan_seconds,
        query_ids=[dataset.rows[i]['id'] for i in ids],model_provenance=provenance())
    (a.output/'started.json').write_text(json.dumps(info,indent=2))
    print(json.dumps({k:v for k,v in info.items() if k not in ('model_provenance','query_ids')}),flush=True)
    rng=rng_state();rows=[];gradients=None
    for step in range(2):
        restore_rng(rng);optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        prof=(torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
              if a.operators and step==1 else contextlib.nullcontext())
        events=[torch.cuda.Event(enable_timing=True) for _ in range(6)]
        began=time.perf_counter()
        with prof as trace:
            events[0].record();query=cpu.cuda(non_blocking=True);events[1].record()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                with torch.profiler.record_function('L0'):
                    embedding=net.local(query)
                events[2].record()
                with torch.profiler.record_function('L1_L2'):
                    result=net.predict_embeddings(embedding,net.prepare_support(*support,cluster_plan=plan))
                    loss=supervised_loss(result,targets,weights)
            events[3].record()
            with torch.profiler.record_function('backward'):loss.backward()
            events[4].record();events[4].synchronize()
            if step==1:
                gradients={n:p.grad.detach().cpu().clone() for n,p in net.named_parameters() if p.grad is not None}
                if len(gradients)!=sum(p.requires_grad for p in net.parameters()):raise AssertionError('Missing gradient')
                if not all(torch.isfinite(g).all() for g in gradients.values()):raise AssertionError('Invalid gradient')
            clip_start=torch.cuda.Event(enable_timing=True);clip_start.record()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
            if step==1:optimizer.step()
            events[5].record();events[5].synchronize()
        row=dict(step=step,loss=float(loss.detach()),wall_seconds=time.perf_counter()-began,
            **{k:events[i].elapsed_time(events[i+1])/1000 for i,k in enumerate(('H2D','L0','L1_L2','backward'))},
            optimizer=clip_start.elapsed_time(events[5])/1000,**memory_metrics())
        rows.append(row);print(json.dumps(row),flush=True)
        with (a.output/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if trace is not None:
            table=trace.key_averages().table(sort_by='self_cuda_time_total',row_limit=45)
            (a.output/'operators.txt').write_text(table,encoding='utf-8')
        del query,embedding,result,loss
        torch.cuda.empty_cache()
    if a.reference:
        ref=torch.load(a.reference,map_location='cpu',weights_only=False)
        comparisons={n:dict(max_abs=float((g-ref[n]).abs().max()),
                            relative_l2=float((g-ref[n]).double().norm()/ref[n].double().norm().clamp_min(1e-12))) for n,g in gradients.items()}
        (a.output/'gradient_comparison.json').write_text(json.dumps(comparisons,indent=2))
    else:torch.save(gradients,a.output/'reference_gradients_DEBUG.pt')
    saver=Saver(a.output,net,optimizer,dict(debug=True,workspace_mib=a.workspace_mib))
    receipt=saver.save(dict(phase='optimization',epoch=0,step=1,next_batch=1,memory=memory,plan=plan))
    (a.output/'result.json').write_text(json.dumps(dict(**info,steps=rows,checkpoint_seconds=receipt['seconds'],
        scope='Two real query passes, one optimizer update; verified support prefix; not final training or MIG speed emulation'),indent=2))
    print(json.dumps(dict(stage='profile_complete',checkpoint_seconds=receipt['seconds'])),flush=True)


if __name__=='__main__':main()
