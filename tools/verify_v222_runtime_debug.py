"""Actual full-size cached queries: runtime equivalence, reuse and throughput.

Explicit DEBUG: six batches from three heavy recipient groups, actual saved
1,216-observation support prefix, unchanged full model. Not full-support or
A100-MIG speed validation. Modes run in separate fresh processes.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('legacy', 'optimized'), required=True)
    parser.add_argument('--workspace-mib', type=int, choices=(64, 256), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=(16, 32), default=32)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    import torch
    import psutil
    import hiercp.model as local_model
    from hiercp_v222.v1_cache import PairDataset, PairLoader, configuration, provenance
    from hiercp_v222.v1_execution import memory_metadata, optimizer_step, Saver, tree_to, restore_rng, rng_state
    from hiercp_v222.v1_local import model, support_for_recipient
    from hiercp_v222.training import configure_runtime
    from hiercp_v222.contracts import sha
    from tools.v222_runtime_cache import CachedPairDataset, CachedPairLoader
    from tools.v222_runtime_execution import AsyncSaver
    from tools.verify_v1_resume import equal
    cfg, base = configuration()
    configure_runtime(base, 42)
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(9e9/torch.cuda.get_device_properties(0).total_memory)
    local_model.EDGE_ATTENTION_WORKSPACE_BYTES = args.workspace_mib * 1024**2
    cache = ROOT/'work/v222_v1_recovered2_training_20260924/cache/index_execution_r6_final.json'
    saved = torch.load(ROOT/'work/v222_v1_resumable_training_20260924_r6/checkpoint_latest.pt',
                       map_location='cpu', weights_only=False)
    if saved['source_identity'] != provenance() or saved['cache_sha256'] != sha(cache):
        raise ValueError('Real model/support/cache provenance mismatch')
    optimized = args.mode == 'optimized'
    dataset = (CachedPairDataset if optimized else PairDataset)(cache, 'inner_train')
    loader_class = CachedPairLoader if optimized else PairLoader
    prefix = saved['state']['memory_next']
    class Prefix:
        rows = dataset.rows[:prefix]
        meta = dataset.meta
    memory = memory_metadata(Prefix, saved['state']['memory_work'].cuda())
    net = model(cfg, base).cuda()
    net.load_state_dict(saved['model'])
    net.local.dense_batch_size = args.batch
    del saved
    optimizer = torch.optim.AdamW(net.parameters(), lr=base['training']['lr'],
        weight_decay=base['training']['weight_decay'], fused=True)
    by_group = {}
    for i, row in enumerate(dataset.rows):
        by_group.setdefault(row['patient_group'], []).append(i)
    for indices in by_group.values():
        indices.sort(key=lambda i: dataset.rows[i]['bounds']['edges'], reverse=True)
    ranked = sorted(by_group, key=lambda group: sum(dataset.rows[i]['bounds']['edges']
        for i in by_group[group][:args.batch]), reverse=True)
    chosen = [(g, by_group[g][k*args.batch:(k+1)*args.batch]) for g in ranked[:3] for k in range(2)]
    if any(len(indices) != args.batch for _, indices in chosen):
        raise ValueError('Full declared physical batch is required')
    classes = torch.tensor([r['target'] for r in dataset.rows], device='cuda')
    counts = torch.bincount(classes, minlength=2).float()
    weights = counts.sum()/(2*counts)
    saver = (AsyncSaver if optimized else Saver)(root, net, optimizer, dict(debug=True))
    loader = loader_class(dataset, 8)
    details = dict(debug=True, mode=args.mode, workspace_mib=args.workspace_mib,
        model_parameters=sum(p.numel() for p in net.parameters()), batch=args.batch,
        allocator_cap_bytes=9000000000, support_prefix=prefix, full_support=len(dataset),
        gpu=torch.cuda.get_device_name(), cpu_logical=psutil.cpu_count(),
        available_ram=psutil.virtual_memory().available, precision='bfloat16',
        workers=8, gradient_accumulation=1, full_training=False, real_CT=True,
        input_shape=[args.batch, 1, 48, 48, 48], model=base['model'], graph_config=base['graph'],
        queries=[[dataset.rows[i]['id'] for i in ids] for _, ids in chosen])
    (root/'started.json').write_text(json.dumps(details, indent=2))
    print(json.dumps({k:v for k,v in details.items() if k not in ('queries','graph_config','model')}), flush=True)
    rows = []
    reference_step = None
    net.train()
    try:
        iterator = iter(loader.batches([ids for _,ids in chosen], epoch=0))
        for number, (group, indices) in enumerate(chosen):
            start = time.perf_counter()
            cpu = next(iterator)
            loaded = time.perf_counter()
            support = support_for_recipient(memory, group)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                plan = net.fit_support_clusters(*support)
            values, timing, usage, gradients = optimizer_step(net, optimizer, cpu, support, plan,
                classes[indices], weights, 5, check_gradients=number==0)
            state = dict(phase='optimization', epoch=0, step=number+1, next_batch=number+1,
                         memory=memory, plan=plan)
            receipt = saver.save(state)
            if not optimized:
                torch.cuda.empty_cache()
            row = dict(step=number+1, loader_wait_seconds=loaded-start,
                step_wall_seconds=time.perf_counter()-start, checkpoint_submit_seconds=receipt['seconds'],
                nodes=int(cpu.graph.num_nodes), edges=int(cpu.graph.num_edges),
                unique_source_maps=len(cpu.source_patches), **values, **timing, **usage)
            if gradients:
                row.update(gradients)
            rows.append(row)
            print(json.dumps(row), flush=True)
            with (root/'steps.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            # Save a frozen real CPU state for cross-process exact comparison.
            if number == 1:
                reference_step = dict(model=tree_to(net.state_dict(),'cpu'),
                    optimizer=tree_to(optimizer.state_dict(),'cpu'), losses=[r['loss'] for r in rows],
                    queries=details['queries'][:2])
                torch.save(reference_step, root/'state_step2_DEBUG.pt')
                if args.reference:
                    old = torch.load(args.reference, map_location='cpu', weights_only=False)
                    if not equal(reference_step, old):
                        raise AssertionError('Runtime changed step-2 model/optimizer/loss for same workspace and CT inputs')
                    del old
            del cpu, plan, support, state
        if isinstance(saver, AsyncSaver):
            saver.flush()
        # Reload the same epoch-0 inputs, then prove a cache hit avoids graph
        # materialization. The equality check is on actual full-size graphs.
        began = time.perf_counter()
        first = loader.make(chosen[0][1], 0)
        first_reload = time.perf_counter()-began
        before_cache = dataset.store.cache.report() if optimized else None
        began = time.perf_counter()
        second = loader.make(chosen[0][1], 0)
        warm_reload = time.perf_counter()-began
        if not equal(first.graph.to_dict(), second.graph.to_dict()):
            raise AssertionError('Repeated graph view changed')
        if not torch.equal(first.source_patches, second.source_patches) or not torch.equal(first.target_patches, second.target_patches):
            raise AssertionError('Repeated CT input changed')
        after_cache = dataset.store.cache.report() if optimized else None
        if optimized and after_cache['misses'] != before_cache['misses']:
            raise AssertionError('Warm graph access unnecessarily recomputed inputs')
        del first, second
        final = dict(**details, steps=rows, same_workspace_reference_bitwise_equal=bool(args.reference),
            first_reload_seconds=first_reload, warm_reload_seconds=warm_reload,
            warm_cache_before=before_cache, warm_cache_after=after_cache,
            peak_allocated=max(r['peak_allocated'] for r in rows),
            peak_reserved=max(r['peak_reserved'] for r in rows),
            scope='Six real full-size batches and native loss updates; partial real support; not a full epoch or final accuracy')
        (root/'result.json').write_text(json.dumps(final, indent=2))
        print(json.dumps(dict(stage='runtime_debug_complete', mode=args.mode,
            peak_reserved=final['peak_reserved'], warm_reload_seconds=warm_reload,
            reference_bitwise_equal=bool(args.reference))), flush=True)
    finally:
        loader.close()
        if isinstance(saver, AsyncSaver):
            saver.close()


if __name__ == '__main__':
    main()
