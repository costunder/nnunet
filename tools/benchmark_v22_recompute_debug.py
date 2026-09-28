"""DEBUG A/B execution probe from a PAUSED production ranking checkpoint.

Reads full saved support/Adam/RNG and the next actual bucket batches. No cache
preparation, support recomputation, production checkpoint writes or auto-resume.
Default candidates disable the outer checkpoint of ONE L0 block each. The CNN
and inner edge checkpoints, graph, physical batch, loss and weights are kept.
The previous all-block policy remains opt-in for reproducibility, not default.
"""
import argparse
from contextlib import contextmanager, nullcontext
import copy
import gc
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@contextmanager
def execution_policy(net, name):
    selective = {f'no_outer_l0_block{i}': i for i in range(3)}
    if name not in ('baseline', 'no_outer_l0', *selective):
        raise ValueError('Unknown diagnostic execution policy')
    old = net.local.checkpoint_local_blocks
    if old is not True:
        raise ValueError('Expected the preserved outer L0 checkpoint baseline')
    try:
        if name in selective:
            if len(net.local.blocks) != 3:
                raise ValueError('Selective policy requires the unchanged three-block L0')
            selected = net.local.blocks[selective[name]]
            original = net.local._run_local_block
            def run(block, x, edges, attributes):
                if block is selected:
                    return block(x, edges, attributes)
                return original(block, x, edges, attributes)
            with patch.object(net.local, '_run_local_block', run):
                yield
            return
        if name == 'no_outer_l0':
            net.local.checkpoint_local_blocks = False
        yield
    finally:
        net.local.checkpoint_local_blocks = old


@contextmanager
def stage_events(local):
    """GPU event intervals; gradient arrival is a boundary, not an exclusive L1/L2 attribution."""
    import torch
    stamps = {key: torch.cuda.Event(enable_timing=True) for key in ('begin', 'end', 'gradient')}
    calls = []
    def before(module, args):
        stamps['begin'].record()
    def after(module, args, output):
        stamps['end'].record()
        calls.append(1)
        def gradient_arrives(gradient):
            stamps['gradient'].record()
        output.register_hook(gradient_arrives)
    handles = [local.register_forward_pre_hook(before), local.register_forward_hook(after)]
    try:
        yield stamps, calls
    finally:
        for handle in handles:
            handle.remove()


def probe_batches(dataset, state, seed, count):
    from hiercp_v222.v1_training import groups
    if count < 2:
        raise ValueError('At least one warm-up and one measured DEBUG batch required')
    if state['phase'] != 'optimization' or not state['training_calibrated']:
        raise ValueError('Pause during optimization, after full support and calibration')
    if state.get('memory') is None:
        raise ValueError('Full saved support is required; no prefix fallback')
    order = list(groups(dataset, state['batch'], seed, state['epoch']))
    start = state['next_batch']
    selected = order[start:start + count]
    if len(selected) != count:
        raise ValueError('Too few remaining batches in this epoch; do not wrap or repeat batches')
    return selected


def frozen(value):
    import numpy as np
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: frozen(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(frozen(v) for v in value)
    return copy.deepcopy(value)


def exact(a, b):
    import numpy as np
    import torch
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def update(net, optimizer, payload, support, plan, context, settings, targets, weights, clip):
    """Same objective/update as production; capture gradients AFTER timed work.

Parity copies deliberately occur after GPU events and peak capture. This is a
compute probe, not a measurement of production loading/checkpoint serialization.
"""
    import torch
    from hiercp_v222.v1_execution import memory_metrics, rng_state
    from tools.v22_rank_objective import forward_loss
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    began = time.perf_counter()
    events[0].record()
    query = payload.cuda(non_blocking=True)
    events[1].record()
    with stage_events(net.local) as (stamps, calls):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, parts = forward_loss(net, query, support, plan, targets, weights,
                                       context, settings, indices=payload.indices.tolist())
        events[2].record()
        loss.backward()
    events[3].record()
    torch.nn.utils.clip_grad_norm_(net.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    events[4].record()
    events[4].synchronize()
    timings = {name: events[i].elapsed_time(events[i+1])/1000 for i, name in
               enumerate(('H2D_seconds', 'forward_seconds', 'backward_seconds', 'optimizer_seconds'))}
    timings.update(wall_compute_seconds=time.perf_counter()-began, **memory_metrics())
    if len(calls) != 1:
        raise RuntimeError('Expected exactly one live L0 forward per optimizer update')
    timings.update(L0_forward_seconds=stamps['begin'].elapsed_time(stamps['end'])/1000,
                   forward_outside_L0_seconds=(events[1].elapsed_time(stamps['begin'])+
                                                stamps['end'].elapsed_time(events[2]))/1000,
                   backward_before_L0_gradient_seconds=events[2].elapsed_time(stamps['gradient'])/1000,
                   backward_after_L0_gradient_seconds=stamps['gradient'].elapsed_time(events[3])/1000)
    gradients = {name: p.grad for name, p in net.named_parameters() if p.requires_grad}
    if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients.values()):
        raise RuntimeError('Missing or nonfinite parameter gradient')
    evidence = frozen(dict(loss=loss, parts=parts, gradients=gradients,
                           model=net.state_dict(), adam=optimizer.state_dict(), rng=rng_state()))
    optimizer.zero_grad(set_to_none=True)
    return timings, evidence


def run_policy(name, saved, dataset, order, reference, output, bar, *, operators=False):
    import psutil
    import torch
    from hiercp_v222.v1_local import model
    from hiercp_v222.v1_execution import tree_to, restore_rng
    from tools.v222_process_loader import ProcessPairLoader, close_producers
    from tools.v222_review_contracts import grouped_support
    from tools.v22_rank_objective import RankingContext, configuration
    state = tree_to(copy.deepcopy(saved['state']), 'cuda')
    base = saved['base']
    net = model(saved['config'], base).cuda()
    net.load_state_dict(saved['model'])
    net.local.dense_batch_size = state['batch']
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=base['training']['lr'],
                                 weight_decay=base['training']['weight_decay'],
                                 fused=base['training']['fused_optimizer'])
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    context = RankingContext(dataset, state['memory'])
    counts = torch.bincount(state['memory']['classes'], minlength=2).float()
    if bool((counts == 0).any()):
        raise ValueError('Both support observation classes required')
    weights = counts.sum()/(2*counts)
    loader = ProcessPairLoader(dataset, state['workers'])
    rows, collected = [], []
    restore_rng(saved['rng'])
    try:
        with execution_policy(net, name):
            for number, payload in enumerate(loader.batches(order, epoch=state['epoch'])):
                ids = payload.indices.tolist()
                if ids != order[number]:
                    raise RuntimeError('Probe batch order differs from production cursor')
                group = dataset.rows[ids[0]]['patient_group']
                support = grouped_support(state['memory'], group)
                if group != state['last_group']:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        state['plan'] = net.fit_support_clusters(*support)
                    state['last_group'] = group
                targets = torch.tensor([dataset.rows[i]['target'] for i in ids], device='cuda')
                bar.set_description(f'{name} {number+1}/{len(order)}')
                # Profile only baseline warm-up; measured A/B batches have no profiler.
                profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) if operators and name == 'baseline' and number == 0 else nullcontext()
                with profile as trace:
                    timing, evidence = update(net, optimizer, payload, support, state['plan'], context,
                                              configuration(), targets, weights, base['training']['grad_clip'])
                if trace is not None:
                    table = trace.key_averages().table(sort_by='self_cuda_time_total', row_limit=40)
                    (output/'baseline_operators.txt').write_text(table, encoding='utf-8')
                checks = None if reference is None else {
                    key: exact(evidence[key], reference[number][key]) for key in evidence}
                row = dict(policy=name, batch_number=number+1, warmup=number == 0,
                           physical_batch=len(ids), nodes=int(payload.graph.num_nodes),
                           edges=int(payload.graph.num_edges), full_memory_records=len(dataset),
                           eligible_support_records=len(support[0]), query_ids=[dataset.rows[i]['id'] for i in ids],
                           rss_with_verification_copies=psutil.Process().memory_info().rss,
                           comparisons=checks, **timing)
                rows.append(row)
                if reference is None:
                    collected.append(evidence)
                del evidence, payload, targets
                with (output/'steps.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row) + '\n')
                bar.update(1)
                bar.set_postfix(seconds=round(timing['wall_compute_seconds'], 2),
                                GiB=round(timing['peak_allocated']/1024**3, 2))
                if saved['state']['release_unused']:
                    torch.cuda.empty_cache()
    finally:
        loader.close()
        close_producers()
    return rows, collected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batches', type=int, default=3, help='Explicit DEBUG batches only; no production cap')
    p.add_argument('--candidates', nargs='+', choices=('no_outer_l0', 'no_outer_l0_block0',
                   'no_outer_l0_block1', 'no_outer_l0_block2'),
                   default=['no_outer_l0_block2', 'no_outer_l0_block1', 'no_outer_l0_block0'])
    p.add_argument('--operators', action='store_true', help='Profile baseline warm-up operators; timing batches stay unprofiled')
    a = p.parse_args(argv)
    if len(set(a.candidates)) != len(a.candidates):
        raise ValueError('Duplicate diagnostic candidates')
    from tools.v222_resume_guard import assert_source_runs_idle
    assert_source_runs_idle(a.cache, a.checkpoint)
    if a.output.exists():
        raise FileExistsError('New DEBUG output required; existing results are preserved')
    import psutil
    import torch
    from tqdm import tqdm
    from hiercp_v222.contracts import sha
    from hiercp_v222.training import configure_runtime
    from tools.v222_runtime_cache import CachedPairDataset
    from tools.v222_review_contracts import installed
    from tools.v22_artifacts import validate_artifact, validate_memory
    import hiercp.model as implementation
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One allocated CUDA device required; no CPU fallback')
    before = sha(a.checkpoint)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if sha(a.checkpoint) != before:
        raise RuntimeError('Source checkpoint changed during read; pause the source run first')
    dataset = CachedPairDataset(a.cache, 'inner_train')
    identity_keys = ('training_objective','ranking_contract','feature_coordinates','artifact_contract',
                     'geometry_contract','run_id','support_task_contract','config','base','source_identity',
                     'cache_sha256','debug','optimizer_contract','rng_contract')
    identity = {key: saved[key] for key in identity_keys}
    validate_artifact(saved, 'resume', identity=identity, resume_rows=dataset.rows)
    if saved['cache_sha256'] != sha(a.cache):
        raise ValueError('Checkpoint/cache identity mismatch')
    order = probe_batches(dataset, saved['state'], saved['config']['seed'], a.batches)
    validate_memory(saved['state']['memory'], dataset.rows, dataset.meta['identities'],
                    dataset.meta['split'], dataset.meta['donor_pool'])
    workspace = saved['execution_policy']['workspace_mib']
    implementation.EDGE_ATTENTION_WORKSPACE_BYTES = workspace*1024**2
    configure_runtime(saved['base'], saved['config']['seed'])
    torch.set_num_threads(psutil.cpu_count(logical=False))
    a.output.mkdir(parents=True, exist_ok=False)
    info = dict(debug=True, full_training=False, automatic_production_migration=False,
                checkpoint_sha256=before, cache_sha256=sha(a.cache),
                source_identity=saved['source_identity'], runtime_sha256=saved['execution_policy']['runtime_sha256'],
                tool_sha256=sha(Path(__file__)), torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                free_bytes_at_start=torch.cuda.mem_get_info()[0], available_ram=psutil.virtual_memory().available,
                epoch=saved['state']['epoch']+1, saved_step=saved['state']['step'],
                physical_batch=saved['state']['batch'], workers=saved['state']['workers'],
                full_support=len(dataset), workspace_mib=workspace, debug_batches=a.batches,
                candidates=a.candidates, baseline_warmup_operator_profile=a.operators,
                backward_intervals='before/after gradient reaches L0 output; shared autograd scheduling is not exclusive module attribution',
                gradient_comparison='all trainable parameter gradients after production clipping',
                timing_excludes='loading, cluster plan preparation, parity CPU copies, production checkpoint writing',
                scope='same next batches and saved full support; compute-only A/B, not full-epoch or worst-case admission')
    (a.output/'started.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f"DEBUG comparison | saved step {info['saved_step']} | full support {len(dataset)} | "
          f"physical batch {info['physical_batch']} | no production writes", flush=True)
    results = []
    with installed('stride4'), tqdm(total=(1+len(a.candidates))*a.batches, unit='batch', dynamic_ncols=True) as bar:
        baseline, reference = run_policy('baseline', saved, dataset, order, None, a.output, bar, operators=a.operators)
        old = statistics.mean(row['wall_compute_seconds'] for row in baseline[1:])
        for name in a.candidates:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                candidate, _ = run_policy(name, saved, dataset, order, reference, a.output, bar)
            except torch.cuda.OutOfMemoryError as error:
                # Keep only text; do not retain the failed autograd graph via traceback.
                summary = dict(policy=name, candidate_rejected='OOM', error=str(error),
                               eligible_for_further_validation=False)
            else:
                new = statistics.mean(row['wall_compute_seconds'] for row in candidate[1:])
                parity = all(all(row['comparisons'].values()) for row in candidate)
                peak = max(row['peak_allocated'] for row in candidate)
                memory_ok = peak < saved['config']['max_vram_fraction']*info['device_total_bytes']
                summary = dict(policy=name, candidate_mean_compute_seconds=new, speedup=old/new,
                               candidate_peak_allocated=peak, all_loss_gradient_model_adam_rng_bitwise_equal=parity,
                               measured_memory_budget_passed=memory_ok,
                               eligible_for_further_validation=parity and memory_ok and new < old*.95)
            results.append(summary)
            (a.output/f'{name}_result.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            bar.write(json.dumps(summary))
    result = dict(**info, baseline_mean_compute_seconds=old, candidate_results=results,
                  production_migration_approved=False,
                  decision_rule='exact tested updates, configured VRAM budget, >5% compute gain; not production approval')
    if sha(a.checkpoint) != before:
        raise RuntimeError('Source checkpoint changed during probe; comparison invalid')
    (a.output/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_identity','runtime_sha256')},
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
