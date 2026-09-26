"""Actual heavy support graphs, paired evaluation-only execution comparison.

No medical score, optimizer training, cohort reduction or production change.
The same full-size batch32 payload and model weights are used by both paths.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import hiercp.model as impl
    from hiercp_v222.v1_cache import configuration, provenance
    from hiercp_v222.v1_local import model
    from hiercp_v222.training import configure_runtime
    from hiercp_v222.contracts import sha
    from tools.v222_runtime_cache import CachedPairDataset, CachedPairLoader
    from tools.v222_support_sparse_debug import installed
    cfg, base = configuration()
    configure_runtime(base, 42)
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(9e9/torch.cuda.get_device_properties(0).total_memory)
    impl.EDGE_ATTENTION_WORKSPACE_BYTES = 256 * 1024**2
    cache = ROOT/'work/v222_v1_recovered2_training_20260924/cache/index_execution_r6_final.json'
    checkpoint = ROOT/'work/v222_runtime_20260925_DEBUG/optimized256/checkpoint_latest.pt'
    data = CachedPairDataset(cache, 'inner_train')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    net = model(cfg, base).cuda().eval()
    net.load_state_dict(saved['model'])
    net.local.dense_batch_size = 32
    del saved
    groups = {}
    for i, row in enumerate(data.rows):
        groups.setdefault(row['patient_group'], []).append(i)
    for ids in groups.values():
        ids.sort(key=lambda i: data.rows[i]['bounds']['edges'], reverse=True)
    ranked = sorted(groups.values(), key=lambda ids:sum(data.rows[i]['bounds']['edges'] for i in ids[:32]), reverse=True)
    selected = [ids[:32] for ids in ranked[:3]]
    if any(len(ids) != 32 for ids in selected):
        raise ValueError('Expected complete physical batches')
    loader = CachedPairLoader(data, 8)
    results = []
    try:
        for batch_number, host in enumerate(loader.batches(selected, epoch=0)):
            payload = host.cuda(non_blocking=True)
            torch.cuda.synchronize()
            reference = None
            for name in ('baseline_warmup', 'baseline', 'sparse', 'sparse_repeat'):
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                events = {}
                def timed(key, function):
                    def call(*a, **kw):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record();value=function(*a, **kw);end.record()
                        events.setdefault(key, []).append((start,end))
                        return value
                    return call
                sparse = name.startswith('sparse')
                began = time.perf_counter()
                with torch.no_grad(), (installed() if sparse else nullcontext()):
                    aggregate = impl._StreamedEdgeAggregation.apply
                    with patch.object(impl.CompatibilityGatedGATv2Conv, '_logit_chunk', timed('logits', impl.CompatibilityGatedGATv2Conv._logit_chunk)), \
                         patch.object(impl, 'pyg_softmax', timed('softmax', impl.pyg_softmax)), \
                         patch.object(impl._StreamedEdgeAggregation, 'apply', timed('aggregation', aggregate)):
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            output = net.local(payload)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-began
                if name == 'baseline':reference=output.detach().clone()
                record = dict(batch=batch_number, mode=name, seconds=elapsed,
                    nodes=host.graph.num_nodes, edges=host.graph.num_edges,
                    peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                    components={key:sum(a.elapsed_time(b)/1000 for a,b in values) for key,values in events.items()})
                if sparse:
                    difference=(output.float()-reference.float()).abs()
                    record.update(max_abs_difference=float(difference.max()), mean_abs_difference=float(difference.mean()),
                        finite=bool(output.isfinite().all()))
                results.append(record)
                with (args.output/'profile.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
            del payload, host, output, reference
    finally:
        loader.close()
    report=dict(debug=True, full_training=False, support_observations=96, total_support=len(data),
        batch_size=32, model_parameters=sum(p.numel() for p in net.parameters()),
        checkpoint_sha256=sha(checkpoint), cache_sha256=sha(cache), source_identity=provenance(),
        gpu=torch.cuda.get_device_name(), allocator_cap_bytes=9000000000,
        numerical_change='FP32 sum order changes; candidate is not a bitwise continuation', results=results)
    (args.output/'result.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
