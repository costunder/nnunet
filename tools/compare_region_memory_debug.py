"""Compare recomputation versus retained activations on the saved next update.

Diagnostic only: identical model/optimizer/RNG, real query, complete saved support,
FP32, physical batch and loss. No production files or refreshed caches are made.
Use a paused checkpoint and an otherwise free GPU for meaningful timings.
"""
import argparse
import copy
import gc
import json
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def assert_state_close(actual, expected, path='state'):
    import torch
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6,
                                   msg=lambda message: path+': '+message)
    elif isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(path+': different keys')
        for key in actual:
            assert_state_close(actual[key], expected[key], path+'.'+str(key))
    elif isinstance(actual, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(path+': different sequence')
        for i, (left, right) in enumerate(zip(actual, expected)):
            assert_state_close(left, right, path+'.'+str(i))
    elif type(actual) is not type(expected) or actual != expected:
        raise AssertionError(path+': different metadata')


@contextmanager
def execution(net, *, retain, workspace_bytes):
    from l0_regions import sparse
    from hiercp_v222 import model as prompt
    core = net.local.core
    previous = (core.checkpoint_dense_encoder, core.checkpoint_local_blocks)
    original_mm = sparse.segmented_mm
    original_checkpoint = prompt.checkpoint
    def direct(function, *args, use_reentrant=False, **kwargs):
        if use_reentrant or kwargs:
            raise ValueError('Unexpected checkpoint options; explicit review required')
        return function(*args)
    def mm(matrix, x):
        return original_mm(matrix, x, workspace_bytes=workspace_bytes)
    core.checkpoint_dense_encoder = False if retain else previous[0]
    core.checkpoint_local_blocks = False if retain else previous[1]
    try:
        with patch.object(sparse, 'segmented_mm', mm), patch.object(
                prompt, 'checkpoint', direct if retain else original_checkpoint):
            yield
    finally:
        core.checkpoint_dense_encoder, core.checkpoint_local_blocks = previous


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('cuda-gib', 'rss-gib', 'resident-gib'):
        p.add_argument('--'+name, type=float, required=True)
    for name in ('workers', 'workspace-mib', 'warmup', 'repeats'):
        p.add_argument('--'+name, type=int, required=True)
    a = p.parse_args()
    if min(a.workers, a.workspace_mib, a.repeats) < 1 or a.warmup < 1:
        raise ValueError('Positive explicit sizes and at least one warmup required')
    import torch
    from l0_regions import training as t
    from l0_regions.training_data import RegionDataset, Budget, sha
    budget = Budget(int(a.cuda_gib*2**30), int(a.rss_gib*2**30))
    total = torch.cuda.get_device_properties(0).total_memory
    if not 0 < budget.cuda_bytes < total or a.resident_gib <= 0:
        raise ValueError('Explicit limits must leave GPU headroom')
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/total)
    torch.set_num_threads(a.workers)
    before = sha(a.checkpoint)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    digest = saved.pop('content_sha256')
    if t.hash_state(saved) != digest or saved['format'] != t.FORMAT:
        raise ValueError('Checkpoint integrity/format mismatch')
    identity = saved['identity']
    if identity['cache_sha256'] != sha(a.cache) or saved['state']['phase'] != 'optimization':
        raise ValueError('Matching cache and optimization checkpoint required')
    ds = RegionDataset(a.cache, 'inner_train', identity['debug'], identity['profile_policy'])
    if identity['base'] != ds.meta['base'] or identity['config'] != ds.meta['config']:
        raise ValueError('Checkpoint/cache configuration mismatch')
    state = t.tree_to(saved['state'], 'cuda')
    batch = state['batch']
    schedule = list(t.groups(ds, batch, ds.meta['config']['seed'], state['epoch']))
    if state['next_batch'] >= len(schedule):
        raise ValueError('No next optimization batch')
    ids = schedule[state['next_batch']]
    loader = t.Loader(ds, a.workers, int(a.resident_gib*2**30))
    query = loader.get(ids).to('cuda')
    memory = state['memory']
    context = t.RankingContext(ds, memory)
    counts = torch.bincount(memory['classes'], minlength=2).float()
    if bool((counts == 0).any()):
        raise ValueError('Both observed classes required')
    weights = counts.sum()/(2*counts)
    group = ds.rows[ids[0]]['patient_group']
    support = t.support_for_recipient(memory, group)
    net = t.make_model(ds, budget, identity['debug'])
    net.local.dense_batch_size = batch
    net.load_state_dict(saved['model'], strict=True)
    if state['last_group'] == group:
        plan = state['plan']
    else:
        net.eval()
        with torch.no_grad():
            plan = net.fit_support_clusters(*support)
    a.output.mkdir(parents=True, exist_ok=False)
    arms = [('checkpointed_64MiB', False, 64*2**20),
            ('retained_64MiB', True, 64*2**20)]
    if a.workspace_mib != 64:
        arms.append(('retained_explicit_workspace', True, a.workspace_mib*2**20))
    rows = {name: [] for name, _, _ in arms}
    reference = None
    checks = {}
    for trial in range(a.warmup+a.repeats):
        for name, retain, workspace in (arms if trial % 2 == 0 else list(reversed(arms))):
            net.load_state_dict(saved['model'], strict=True)
            optimizer = torch.optim.AdamW(net.parameters(), lr=ds.meta['base']['training']['lr'],
                weight_decay=ds.meta['base']['training']['weight_decay'], fused=ds.meta['base']['training']['fused_optimizer'])
            optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
            net.train(); optimizer.zero_grad(set_to_none=True)
            t.restore_rng(saved['rng'])
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            times = {}
            def timed(key, function):
                torch.cuda.synchronize(); start = time.perf_counter()
                value = function(); torch.cuda.synchronize()
                times[key] = time.perf_counter()-start
                return value
            start = time.perf_counter()
            with execution(net, retain=retain, workspace_bytes=workspace):
                loss, terms = timed('forward', lambda: t.forward_loss(net, query, support, plan,
                    memory['classes'][ids], weights, context, t.rank_config(), indices=ids))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite loss')
                timed('backward', loss.backward)
                def clip():
                    t.gradient_check(net)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), ds.meta['base']['training']['grad_clip'], error_if_nonfinite=True)
                timed('check_clip', clip)
                timed('optimizer', optimizer.step)
            elapsed = time.perf_counter()-start
            budget.check()
            peak = torch.cuda.max_memory_allocated()
            # Outside timing: every trainable gradient, model/Adam tensor and
            # post-update RNG is compared. No sampled gradient subset.
            current = dict(loss=loss.detach().cpu(), terms=t.tree_to(terms, 'cpu'),
                gradients={k: v.grad.detach().cpu().clone() for k, v in net.named_parameters() if v.requires_grad},
                model=t.tree_to(net.state_dict(), 'cpu'), optimizer=t.tree_to(optimizer.state_dict(), 'cpu'))
            rng_hash = t.hash_state(t.rng_state())
            if reference is None:
                reference, reference_rng = current, rng_hash
            else:
                assert_state_close(current, reference)
                if rng_hash != reference_rng:
                    raise ValueError('Execution change altered RNG advancement')
            checks[name] = True
            row = dict(trial=trial, warmup=trial<a.warmup, whole_update_seconds=elapsed,
                times=times, peak_cuda_bytes=peak, workspace_bytes=workspace)
            rows[name].append(row)
            print(f'{name} trial={trial} update={elapsed:.3f}s peak={peak/2**30:.3f}GiB', flush=True)
            del current, optimizer, loss, terms
            net.zero_grad(set_to_none=True); gc.collect()
    if before != sha(a.checkpoint) or identity['cache_sha256'] != sha(a.cache):
        raise ValueError('Source artifacts changed during diagnostic; use a paused run')
    summary = {name: dict(update_mean_seconds=statistics.mean(r['whole_update_seconds'] for r in values if not r['warmup']),
        peak_cuda_bytes=max(r['peak_cuda_bytes'] for r in values)) for name, values in rows.items()}
    result = dict(debug=True, full_training=False, production_ready=False, production_writes=False,
        checkpoint_unchanged=True, cache_rebuilt=False, status='PASS', checkpoint_sha256=before,
        gpu=torch.cuda.get_device_name(), configured_batch=batch, actual_batch=len(ids),
        total_memory_records=len(memory['record_ids']), support_records=len(support[0]),
        next_step=state['step']+1, parity=checks, parity_rtol=2e-5, parity_atol=2e-6,
        scope='Same saved next update, full saved support; timing excludes loading, plan fitting, parity checks and checkpoint saving; not an epoch estimate',
        rows=rows, summary=summary)
    (a.output/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
