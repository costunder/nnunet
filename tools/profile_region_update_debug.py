"""One real next update from a saved region run, in disposable diagnostic state.

No cache generation, support refresh, production checkpoint, resume migration or
long training. Keep the saved physical batch and complete saved support. Run only
when the source training is paused: concurrent GPU work invalidates the timing.
"""
import argparse
import io
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('cuda-gib', 'rss-gib', 'resident-gib'):
        p.add_argument('--' + name, type=float, required=True)
    p.add_argument('--workers', type=int, required=True)
    a = p.parse_args()
    import torch
    import psutil
    from torch.profiler import profile, ProfilerActivity, record_function
    from l0_regions import training as t, sparse, encoder
    from l0_regions.training_data import RegionDataset, Budget, sha, source_identity

    if a.workers < 1 or min(a.cuda_gib, a.rss_gib, a.resident_gib) <= 0:
        raise ValueError('Positive explicit resource limits required')
    budget = Budget(int(a.cuda_gib * 2**30), int(a.rss_gib * 2**30))
    total = torch.cuda.get_device_properties(0).total_memory
    if budget.cuda_bytes >= total:
        raise ValueError('Leave device headroom')
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes / total)
    torch.set_num_threads(a.workers)
    checkpoint_hash = sha(a.checkpoint)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    digest = saved.pop('content_sha256')
    if t.hash_state(saved) != digest or saved['format'] != t.FORMAT:
        raise ValueError('Checkpoint content/format mismatch')
    identity = saved['identity']
    saved_rng = saved['rng']
    if identity['cache_sha256'] != sha(a.cache):
        raise ValueError('Different cache')
    if saved['state']['phase'] != 'optimization':
        raise ValueError('An optimization checkpoint is required; no support rebuild')
    ds = RegionDataset(a.cache, 'inner_train', identity['debug'], identity['profile_policy'])
    if identity['base'] != ds.meta['base'] or identity['config'] != ds.meta['config']:
        raise ValueError('Checkpoint configuration differs from cache')
    net = t.make_model(ds, budget, identity['debug'])
    net.load_state_dict(saved['model'], strict=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=ds.meta['base']['training']['lr'],
        weight_decay=ds.meta['base']['training']['weight_decay'],
        fused=ds.meta['base']['training']['fused_optimizer'])
    optimizer.load_state_dict(saved['optimizer'])
    state = t.tree_to(saved['state'], 'cuda')
    del saved
    batch = state['batch']
    net.local.dense_batch_size = batch
    order = list(t.groups(ds, batch, ds.meta['config']['seed'], state['epoch']))
    if state['next_batch'] >= len(order):
        raise ValueError('No next optimization batch in this epoch')
    ids = order[state['next_batch']]
    memory = state['memory']
    context = t.RankingContext(ds, memory)
    counts = torch.bincount(memory['classes'], minlength=2).float()
    if bool((counts == 0).any()):
        raise ValueError('Both observed classes required')
    weights = counts.sum() / (2 * counts)
    loader = t.Loader(ds, a.workers, int(a.resident_gib * 2**30))
    times = {}

    @contextmanager
    def timed(name):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with record_function(name):
            yield
        torch.cuda.synchronize()
        times[name] = time.perf_counter() - start

    # Ranges do not change tensor computations. Nested times are inclusive;
    # checkpoint recomputations appear again inside backward.
    def annotate(obj, name, label):
        original = getattr(obj, name)
        def wrapped(*args, **kwargs):
            with record_function(label):
                return original(*args, **kwargs)
        setattr(obj, name, wrapped)

    annotate(net.local, 'forward', 'L0')
    annotate(net.local.dense_encoder, 'forward', 'CNN')
    annotate(net, 'prepare_support', 'support_L1_L2')
    annotate(net, 'predict_embeddings', 'query_L1_rank')
    annotate(sparse, 'segmented_mm', 'SAGE_segmented_mm')
    annotate(encoder, 'sample_nodes', 'fine_feature_sampling')
    annotate(encoder, 'mass_mean', 'region_pooling')
    a.output.mkdir(parents=True, exist_ok=False)
    with timed('load_cold'):
        cpu = loader.get(ids)
    with timed('transfer_validation'):
        query = cpu.to('cuda')
    group = ds.rows[ids[0]]['patient_group']
    with timed('support_selection_and_plan'):
        support = t.support_for_recipient(memory, group)
        if state['last_group'] != group:
            net.eval()
            with torch.no_grad():
                state['plan'] = net.fit_support_clusters(*support)
    net.train()
    # Restore the actual saved RNG only after input preparation, as the normal
    # next update does not re-initialize the model. No state is written back.
    t.restore_rng(saved_rng)
    torch.cuda.reset_peak_memory_stats()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        optimizer.zero_grad(set_to_none=True)
        with timed('forward'):
            loss, terms = t.forward_loss(net, query, support, state['plan'],
                memory['classes'][ids], weights, context, t.rank_config(), indices=ids)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite loss')
        with timed('backward'):
            loss.backward()
        with timed('gradient_check_clip'):
            t.gradient_check(net)
            torch.nn.utils.clip_grad_norm_(net.parameters(), ds.meta['base']['training']['grad_clip'], error_if_nonfinite=True)
        with timed('optimizer'):
            optimizer.step()
    budget.check()
    # Measure checkpoint copy/hash/serialization without a disk artifact. Disk
    # completion and original network filesystem latency remain unmeasured.
    with timed('checkpoint_copy_hash_serialize_no_disk'):
        payload = dict(model=t.tree_to(net.state_dict(), 'cpu'),
            optimizer=t.tree_to(optimizer.state_dict(), 'cpu'), state=t.tree_to(state, 'cpu'), rng=t.rng_state())
        payload['content_sha256'] = t.hash_state(payload)
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        serialized_bytes = buffer.tell()
    if sha(a.checkpoint) != checkpoint_hash:
        raise ValueError('Source checkpoint changed while profiling; pause training first')
    result = dict(debug=True, production_ready=False, full_training=False,
        production_writes=False, cache_rebuilt=False, checkpoint_unchanged=True,
        source_checkpoint_sha256=checkpoint_hash, runtime_source=source_identity(),
        saved_source=identity['source'], configured_batch=batch, actual_batch=len(ids),
        support_records=len(support[0]), total_memory_records=len(memory['record_ids']),
        next_step=state['step'] + 1, records=[ds.rows[i]['id'] for i in ids],
        gpu=torch.cuda.get_device_name(), times=times, loss=float(loss.detach()),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), rss_bytes=psutil.Process().memory_info().rss,
        checkpoint_serialized_bytes=serialized_bytes,
        scope='One cold next update, profiler overhead included; current diagnostic runtime, not exact production resume or epoch prediction; no disk-save timing')
    table = prof.key_averages()
    (a.output / 'gpu.txt').write_text(table.table(sort_by='self_cuda_time_total', row_limit=40), encoding='utf-8')
    (a.output / 'cpu.txt').write_text(table.table(sort_by='self_cpu_time_total', row_limit=40), encoding='utf-8')
    (a.output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items()
                     if k not in ('runtime_source', 'saved_source')}, indent=2))


if __name__ == '__main__':
    main()
