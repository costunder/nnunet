"""Short actual-CT L0 operator diagnostic using an existing explicit DEBUG cache.

Every record/node/edge in that DEBUG partition is used together. Full production
L0 width/depth/CT resolution, untrained weights, squared-output diagnostic loss.
This is neither the ranking objective nor an epoch/capacity/medical score test.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', choices=('baseline','fused'), default='baseline')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    import torch
    import psutil
    import hiercp.model as implementation
    from tools.v22_debug_profile import RepairDataset
    from tools.v222_runtime_cache import CachedPairLoader
    from tools.v222_review_contracts import installed
    from hiercp_v222.v1_local import V1LocalEncoder
    from hiercp_v222.v1_cache import configuration
    from hiercp_v222.training import configure_runtime
    cfg, base = configuration()
    configure_runtime(base, cfg['seed'])
    torch.set_num_threads(min(8, psutil.cpu_count(logical=False)))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    implementation.EDGE_ATTENTION_WORKSPACE_BYTES = 256 * 1024**2
    dataset = RepairDataset(args.cache, 'inner_train')
    loader = CachedPairLoader(dataset, 4)
    try:
        payload = loader.make(list(range(len(dataset))), 0).cuda(non_blocking=True)
    finally:
        loader.close()
    with installed('stride4'):
        net = V1LocalEncoder(base).cuda().train()
    net.dense_batch_size = len(dataset)
    report = dict(debug=True, actual_CT=True, trained=False, full_training=False,
        cache=str(args.cache.resolve()), source_identity=dataset.meta['source_identity'],
        physical_batch=len(dataset), nodes=payload.graph.num_nodes, edges=payload.graph.num_edges,
        parameters=sum(p.numel() for p in net.parameters()), gpu=torch.cuda.get_device_name(),
        ram_available=psutil.virtual_memory().available, cpu_threads=torch.get_num_threads(),
        precision='BF16 autocast', workspace_mib=256, policy=args.policy, scope=__doc__)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_identity','scope')}),flush=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        net(payload).float().square().mean().backward()
    net.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print('Warm-up complete; one profiled full DEBUG batch.',flush=True)
    records = {}; phase = ['forward']
    def timed(key, function):
        def call(*a, **kw):
            begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            begin.record(); value = function(*a, **kw); end.record()
            records.setdefault(phase[0]+':'+key, []).append((begin,end))
            return value
        return call
    # Checkpoint early-stop can interrupt calls; timers count completed calls.
    with ExitStack() as stack:
        if args.policy == 'fused':
            from tools.v22_l0_logit_stream import installed as kernels_installed, FusedEdgeLogit, NodeCastAggregation
            stack.enter_context(kernels_installed(net, fused=True, node_cast=True))
            # Compile outside the profiled region, then discard its gradients.
            with torch.autocast('cuda',dtype=torch.bfloat16):
                net(payload).float().square().mean().backward()
            net.zero_grad(set_to_none=True);torch.cuda.synchronize()
            stack.enter_context(patch.object(FusedEdgeLogit, 'apply', timed('fused_logits', FusedEdgeLogit.apply)))
        for name in ('encode_dense_maps','_sample_dense_features','_run_local_block'):
            stack.enter_context(patch.object(net,name,timed(name,getattr(net,name))))
        cls = implementation.CompatibilityGatedGATv2Conv
        stack.enter_context(patch.object(cls,'_logit_chunk',timed('logits',cls._logit_chunk)))
        aggregation = NodeCastAggregation if args.policy == 'fused' else implementation._StreamedEdgeAggregation
        stack.enter_context(patch.object(aggregation,'apply',timed('aggregation',aggregation.apply)))
        torch.cuda.reset_peak_memory_stats()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        began = time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as trace:
            events[0].record()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss = net(payload).float().square().mean()
            events[1].record();phase[0]='backward';loss.backward();events[2].record()
            torch.cuda.synchronize()
        report.update(profiled_wall_seconds=time.perf_counter()-began,
            profiled_forward_seconds=events[0].elapsed_time(events[1])/1000,
            profiled_backward_seconds=events[1].elapsed_time(events[2])/1000,
            peak_allocated=torch.cuda.max_memory_allocated(),
            completed_calls={k:dict(calls=len(v),gpu_seconds=sum(a.elapsed_time(b)/1000 for a,b in v)) for k,v in records.items()},
            timing_caveat='Profiler overhead included; nested timers overlap; incomplete early-stop calls excluded')
        for sort, name in [('self_cuda_time_total','gpu'),('self_cpu_time_total','cpu')]:
            (args.output/f'{name}_operators.txt').write_text(trace.key_averages().table(sort_by=sort,row_limit=35),encoding='utf-8')
    (args.output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_identity','scope')},indent=2),flush=True)


if __name__ == '__main__':
    main()
