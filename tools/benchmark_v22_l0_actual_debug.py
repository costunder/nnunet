"""Three short same-state L0 update comparisons on complete actual DEBUG graphs.

Untrained full-size production L0, squared fused-output probe; not ranking loss,
not full-support training, not epoch timing. All saved DEBUG CT graphs retained.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policies',nargs='+',choices=('baseline','direct','fused','combined'),
                   default=['baseline','direct','fused'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.policies[0]!='baseline':raise ValueError('Baseline must run first')
    import torch
    import psutil
    import hiercp.model as implementation
    from tools.v22_debug_profile import RepairDataset
    from tools.v222_runtime_cache import CachedPairLoader
    from tools.v222_review_contracts import installed as coordinates
    from tools.v22_l0_logit_stream import installed
    from tools.benchmark_v22_recompute_debug import frozen,exact
    from hiercp_v222.v1_local import V1LocalEncoder
    from hiercp_v222.v1_cache import configuration
    from hiercp_v222.training import configure_runtime
    from hiercp_v222.v1_execution import rng_state,restore_rng
    cfg,base=configuration();configure_runtime(base,cfg['seed'])
    torch.set_num_threads(min(8,psutil.cpu_count(logical=False)))
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    implementation.EDGE_ATTENTION_WORKSPACE_BYTES=256*1024**2
    dataset=RepairDataset(a.cache,'inner_train');loader=CachedPairLoader(dataset,4)
    try:payload=loader.make(list(range(len(dataset))),0).cuda(non_blocking=True)
    finally:loader.close()
    with coordinates('stride4'):net=V1LocalEncoder(base).cuda().train()
    net.dense_batch_size=len(dataset)
    initial=frozen(net.state_dict());random=rng_state();reference=[];rows=[]
    for policy in a.policies:
        net.load_state_dict(initial);restore_rng(random)
        opt=torch.optim.AdamW(net.parameters(),lr=base['training']['lr'],
              weight_decay=base['training']['weight_decay'],fused=base['training']['fused_optimizer'])
        with installed(net,fused=policy in ('fused','combined'),node_cast=policy=='combined') if policy!='baseline' else nullcontext():
            for i in range(3):  # explicit DEBUG: warm-up + two measured updates
                opt.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
                start=time.perf_counter();events[0].record()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=net(payload);loss=output.float().square().mean()
                events[1].record();loss.backward();events[2].record()
                # Preserve the gradients before optimizer can mutate buffers.
                torch.nn.utils.clip_grad_norm_(net.parameters(),base['training']['grad_clip'],error_if_nonfinite=True)
                opt.step();events[3].record();events[3].synchronize()
                row=dict(policy=policy,update=i+1,warmup=i==0,seconds=time.perf_counter()-start,
                    forward_seconds=events[0].elapsed_time(events[1])/1000,
                    backward_seconds=events[1].elapsed_time(events[2])/1000,
                    optimizer_seconds=events[2].elapsed_time(events[3])/1000,
                    peak_allocated=torch.cuda.max_memory_allocated())
                evidence=frozen(dict(output=output,loss=loss,gradients={k:v.grad for k,v in net.named_parameters()},
                    model=net.state_dict(),adam=opt.state_dict(),rng=rng_state()))
                if policy=='baseline':
                    if i < len(reference):
                        if not exact(reference[i],evidence):raise RuntimeError('Repeated baseline changed numerically')
                    else:reference.append(evidence)
                else:row['bitwise_equal']={k:exact(v,reference[i][k]) for k,v in evidence.items()}
                rows.append(row);print(json.dumps(row),flush=True)
                del evidence,output,loss
        del opt
    report=dict(debug=True,actual_CT=True,trained=False,full_training=False,full_evaluation=False,
        scope=__doc__,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),
        cache=str(a.cache.resolve()),source_identity=dataset.meta['source_identity'],
        physical_batch=len(dataset),nodes=payload.graph.num_nodes,edges=payload.graph.num_edges,
        L0_parameters=sum(p.numel() for p in net.parameters()),rows=rows,
        means={p:statistics.mean(r['seconds'] for r in rows if r['policy']==p and not r['warmup'])
               for p in a.policies},
        epoch_speedup_verified=False,production_migration_approved=False)
    (a.output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['means']),flush=True)


if __name__=='__main__':main()
