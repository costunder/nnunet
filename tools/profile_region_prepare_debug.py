"""Explicit short preparation profile; no training, cache index or ready marker.

Reads selected real records without changing the running server or its outputs.
The instrumented preparation path includes integrity checks and diagnostics.
Nested timings are inclusive and must not be added to their parent timings.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'prepared-frozen', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--indices', type=int, nargs='+', required=True)
    p.add_argument('--partition', choices=('inner_train', 'inner_val'), required=True)
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--cuda-gib', type=float, required=True)
    p.add_argument('--rss-gib', type=float, required=True)
    p.add_argument('--debug', action='store_true', required=True)
    p.add_argument('--debug-cache', action='store_true')
    a = p.parse_args()
    import psutil
    import torch
    import l0_regions.preparation as prep
    import l0_regions.materialization as material
    from l0_regions.training_data import Budget, dataset, reference_from_prepared, sha
    from l0_regions.data import save_new
    from tools.v22_artifacts import tree_hash

    if min(a.workers, a.cuda_gib, a.rss_gib) <= 0:
        raise ValueError('Explicit positive resources required')
    torch.set_num_threads(a.workers)
    budget = Budget(int(a.cuda_gib * 2**30), int(a.rss_gib * 2**30))
    total = torch.cuda.get_device_properties(0).total_memory
    if budget.cuda_bytes >= total:
        raise ValueError('Leave explicit device memory headroom')
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes / total)
    a.output.mkdir(parents=True, exist_ok=False)
    ds = dataset(a.cache, a.partition, a.debug_cache)
    if not a.indices or len(set(a.indices)) != len(a.indices) or any(i < 0 or i >= len(ds.rows) for i in a.indices):
        raise ValueError('Explicit distinct indices must be within the dataset')
    ref, origin, _ = reference_from_prepared(a.prepared_frozen, a.debug_cache)
    frozen = ref.cuda().eval().requires_grad_(False)
    profile = origin['profile']
    if profile.get('region_scales') != 1:
        raise ValueError('This probe is for the current single-scale preparation')
    timings = {}
    def measured(name, fn, gpu=False):
        def wrapped(*args, **kwargs):
            if gpu:
                torch.cuda.synchronize()
            start = time.perf_counter()
            value = fn(*args, **kwargs)
            if gpu:
                torch.cuda.synchronize()
            timings.setdefault(name, []).append(time.perf_counter() - start)
            return value
        return wrapped

    process = psutil.Process()
    cpu_start = process.cpu_times()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with patch.object(material, 'collate', measured('cpu_collate', material.collate)), \
         patch.object(material, 'content_hash', measured('materialization_hash', material.content_hash)):
        fine = measured('cpu_load_materialize_collate_hash', material.load_pairs)(
            ds, a.indices, workers=a.workers, epoch=origin['view_epoch'], cache_path=a.cache)
    records = [dict(ds.rows[i], donor_component=r['donor_component']) for i, r in zip(a.indices, fine.receipts)]
    bindings = measured('binding', prep.batch_bindings)(records, a.indices,
        cache_sha256=sha(a.cache), frozen_cnn_sha256=origin['cnn_sha256'], profile=profile,
        view_epoch=origin['view_epoch'], view_index=0,
        feature_evidence='checkpoint_partition_quality_unverified')
    fine = measured('h2d', fine.to, True)('cuda')
    with patch.object(prep, 'validate_batch', measured('input_validation', prep.validate_batch, True)), \
         patch.object(prep, 'verify_materialization', measured('gpu_materialization_hash_validation', prep.verify_materialization, True)), \
         patch.object(frozen, 'encode_dense_maps', measured('cnn', frozen.encode_dense_maps, True)), \
         patch.object(prep.OfflinePartition, '_coarsen', measured('partition_quotient_diagnostics', prep.OfflinePartition._coarsen, True)), \
         patch.object(prep, '_single_items', measured('quotient_recheck_packaging_receipts', prep._single_items, True)):
        items, audit = measured('prepare_inclusive', prep.prepare, True)(fine, frozen, profile, bindings, budget, allow_unvalidated_profile=True)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        byte_count = sum(pool.map(lambda v: save_new(*v), [(a.output / f'debug_{i}.pt', it) for i, it in zip(a.indices, items)]))
    timings['save_validate_hash_write'] = [time.perf_counter() - start]
    wall = time.perf_counter() - started
    cpu_end = process.cpu_times()
    budget.check()
    report = dict(debug=True, full_training=False, production_ready=False,
        physical_batch=len(items), indices=a.indices, partition=a.partition, workers=a.workers,
        gpu=torch.cuda.get_device_name(0), cpu_logical=psutil.cpu_count(),
        rss_bytes=process.memory_info().rss, available_ram_bytes=psutil.virtual_memory().available,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), serialized_bytes=byte_count,
        source_cache_sha256=sha(a.cache), cnn_sha256=origin['cnn_sha256'],
        timings=timings, whole_batch_seconds=wall,
        process_cpu_seconds=(cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system),
        official_merge_seconds=sum(v['official_merge_seconds'] for v in audit['scale1']['roles'].values()),
        fine_nodes=sum(fine.graph[k].num_nodes for k in fine.graph.node_types),
        fine_edges=sum(fine.graph[e].edge_index.shape[1] for e in fine.graph.edge_types),
        region_nodes=audit['scale1']['nodes_per_pair'], region_edges=audit['scale1']['edges_per_pair'],
        tensor_payload_sha256=[tree_hash({k:it[k] for k in ('source_patch','target_patch','fine','scales','edges')}) for it in items],
        scope='One explicit DEBUG batch; cold instrumented timings, inclusive nested spans. Not steady-state or server epoch throughput.')
    (a.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    (a.output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
