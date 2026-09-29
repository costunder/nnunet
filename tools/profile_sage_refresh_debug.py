"""Short real-data refresh audit; no training, checkpoints or admission changes."""
import argparse
import copy
import gc
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
import torch
import torch_geometric
import l0_sage.encoder as sage_module
from hiercp_v22.schema import LOCAL_EDGE_TYPES, LOCAL_NODE_TYPES
from hiercp_v222.v1_cache import configuration, provenance
from hiercp_v222.v1_local import V1LocalEncoder
from l0_ezsp.data import load_pairs
from l0_ezsp.diagnostic import ResourceBudget
from l0_ezsp.validation import validate_batch
from tools.profile_l0_ezsp_debug import timed, sha
from tools.v222_review_contracts import installed
from tools.v22_debug_profile import RepairDataset
from tools.diagnose_l0_ezsp_pairs import tensor_hash


def measured_forward(net, batch, is_sage):
    """Instrument the real wrapper; all original validation still executes."""
    parts = {}

    def validation(value):
        result, parts['integrity_seconds'] = timed(lambda: validate_batch(value))
        return result

    if is_sage:
        original_prepare = net.adjacency.prepare
        original_forward = net.encoder.forward

        def prepare(graph):
            result, parts['adjacency_seconds'] = timed(lambda: original_prepare(graph))
            return result

        def encoder(value):
            result, parts['encoder_seconds'] = timed(lambda: original_forward(value))
            return result

        with patch.object(sage_module, 'validate_batch', validation), \
             patch.object(net.adjacency, 'prepare', prepare), \
             patch.object(net.encoder, 'forward', encoder):
            output, parts['whole_seconds'] = timed(lambda: net(batch))
        parts['adjacency_build_count'] = net.adjacency.builds
        parts['other_and_instrumentation_seconds'] = parts['whole_seconds'] - sum(
            parts[k] for k in ('integrity_seconds', 'adjacency_seconds', 'encoder_seconds'))
    else:
        output, parts['whole_seconds'] = timed(lambda: net(batch))
        parts['encoder_seconds'] = parts['whole_seconds']
    if output.shape != (len(batch), 128) or not bool(torch.isfinite(output).all()):
        raise FloatingPointError('Invalid measured embedding')
    return output, parts


def module_measured_forward(net, batch, is_sage):
    """Nested synchronized timings; module instrumentation is diagnostic only."""
    encoder = net.encoder if is_sage else net
    modules = {}

    def wrapper(original, key):
        def call(*args, **kwargs):
            result, seconds = timed(lambda: original(*args, **kwargs))
            modules[key] = modules.get(key, 0.) + seconds
            return result
        return call

    with ExitStack() as stack:
        stack.enter_context(patch.object(encoder, 'encode_dense_maps',
            wrapper(encoder.encode_dense_maps, 'cnn_seconds')))
        for i, block in enumerate(encoder.blocks):
            stack.enter_context(patch.object(block, 'forward',
                wrapper(block.forward, f'block_{i}_seconds')))
        output, parts = measured_forward(net, batch, is_sage)
    parts.update(modules)
    parts['sample_project_readout_and_instrumentation_seconds'] = parts['encoder_seconds'] - sum(modules.values())
    return output, parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    for key in ('physical-batch', 'workers', 'repeats', 'warmup'):
        parser.add_argument('--' + key, type=int, required=True)
    for key in ('cuda-gib', 'rss-gib', 'seconds'):
        parser.add_argument('--' + key, type=float, required=True)
    args = parser.parse_args()
    if min(args.physical_batch, args.workers, args.repeats, args.warmup) < 1:
        raise ValueError('Explicit positive batch/workers/repeats/warmup required')
    args.output.mkdir(parents=True, exist_ok=False)
    budget = ResourceBudget(int(args.cuda_gib * 2**30), int(args.rss_gib * 2**30), args.seconds)
    total = torch.cuda.get_device_properties(0).total_memory
    if budget.cuda_bytes >= total:
        raise ValueError('Device headroom required')
    previous = torch.cuda.get_per_process_memory_fraction()
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes / total)
    report = dict(debug=True, training_started=False, production_admitted=False,
        scope='FP32 eval/no_grad, all actual DEBUG records; no backward/update/loader/save/validation metric in refresh timer',
        timing='CUDA synchronized subphases; instrumentation adds synchronization, not the earlier uninstrumented benchmark',
        cold='new encoder instance, adjacency absent; CNN/kernel warmed by explicit prior trials',
        hot='same encoder and exact same immutable batch; reuse diagnostic, not a whole support pass',
        resources=dict(gpu=torch.cuda.get_device_name(), total_vram=total,
            free_vram=torch.cuda.mem_get_info()[0], cpu_physical=psutil.cpu_count(logical=False),
            ram_available=psutil.virtual_memory().available, cuda_budget=budget.cuda_bytes,
            rss_budget=budget.rss_bytes, seconds_budget=budget.seconds),
        torch_version=torch.__version__, pyg_version=torch_geometric.__version__,
        physical_batch=args.physical_batch, workers=args.workers, precision='FP32',
        repeats=args.repeats, warmup=args.warmup, core_identity=provenance(),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            (Path(__file__), ROOT/'l0_sage/encoder.py', ROOT/'l0_ezsp/validation.py')},
        cache_sha256=sha(args.cache), rows=[])
    try:
        torch.set_num_threads(args.workers)
        ds = RepairDataset(args.cache, 'inner_train')
        if len(ds.rows) != args.physical_batch:
            raise ValueError('Physical batch must contain the whole explicitly DEBUG cache; no slicing')
        batch = load_pairs(ds, list(range(len(ds.rows))), workers=args.workers).pin_memory()
        batch, report['h2d_seconds'] = timed(lambda: batch.to('cuda'))
        validate_batch(batch)
        report['records'] = [r['id'] for r in ds.rows]
        report['nodes'] = sum(batch.graph[k].num_nodes for k in LOCAL_NODE_TYPES)
        report['edges'] = sum(batch.graph[k].edge_index.shape[1] for k in LOCAL_EDGE_TYPES)
        report['incoming_relation_root_counts'] = dict(Counter(e[2] for e in LOCAL_EDGE_TYPES))
        _, base = configuration()
        torch.manual_seed(42)
        with installed('stride4'):
            reference = V1LocalEncoder(base)
        masters = {'gat':reference, 'sage':sage_module.GraphSAGEEncoder(reference, seed=42)}
        hashes = {k:tensor_hash(v) for k,v in masters.items()}
        report['weight_source'] = 'same seed42 initialized local masters, no trained checkpoint'
        report['master_hashes'] = hashes
        report['dense_batch_size'] = reference.dense_batch_size
        with torch.no_grad():
            for trial in range(args.warmup + args.repeats):
                for name in (('gat','sage') if trial % 2 == 0 else ('sage','gat')):
                    budget.check()
                    net = copy.deepcopy(masters[name]).cuda().eval()
                    torch.cuda.reset_peak_memory_stats()
                    cold, cold_parts = module_measured_forward(net, batch, name=='sage')
                    hot, hot_parts = module_measured_forward(net, batch, name=='sage')
                    torch.testing.assert_close(cold, hot, rtol=1e-5, atol=1e-6)
                    if name=='sage' and net.adjacency.builds != 1:
                        raise RuntimeError('CSR rebuilt for unchanged batch')
                    report['rows'].append(dict(arm=name, trial=trial, warmup=trial<args.warmup,
                        cold=cold_parts, hot=hot_parts,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                    print(name, trial, 'cold', round(cold_parts['whole_seconds'],4),
                        'hot', round(hot_parts['whole_seconds'],4), flush=True)
                    del net, cold, hot
                    gc.collect()
                    torch.cuda.empty_cache()
                    budget.check()
        report['means'] = {}
        for name in masters:
            rows = [r for r in report['rows'] if r['arm']==name and not r['warmup']]
            report['means'][name] = {mode:{k:statistics.mean(r[mode][k] for r in rows)
                for k in rows[0][mode] if k.endswith('_seconds')} for mode in ('cold','hot')}
            if tensor_hash(masters[name]) != hashes[name]:
                raise RuntimeError('Master modified')
        report['status'] = 'DEBUG_REFRESH_AUDIT_COMPLETE'
    except Exception as exc:
        report['status'] = 'ERROR'
        report['error'] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        torch.cuda.set_per_process_memory_fraction(previous)
        report['rss_end'] = psutil.Process().memory_info().rss
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
