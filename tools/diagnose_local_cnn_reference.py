"""Explicit short legacy/reference L1 comparison on cloned native CT models.

Never writes a production checkpoint or changes an experiment. Reads the bound
latest snapshot once; all candidates of explicitly chosen cases are evaluated.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def select_cases(rows, count):
    """Fixed spread of P/U-eligible cases, independent of model predictions."""
    groups = {}
    for row in rows:
        groups.setdefault(row['case_id'], set()).add(row['target'])
    names = sorted(name for name, classes in groups.items() if classes == {0, 1})
    if type(count) is not int or count <= 0 or len(names) < count:
        raise ValueError('Explicit requested count exceeds available P/U diagnostic cases')
    return [names[(i*len(names))//count] for i in range(count)]


def diagnose(a):
    import psutil
    import torch
    from tqdm.auto import tqdm
    from hiercp_v222.v1_execution import tree_to
    from l0_local_cnn.data import Dataset, Loader
    from l0_regions.donor_learning import LiveContext, groups
    from l0_regions.support_episodes import PatientEpisodes
    from l0_regions.training import FORMAT, hash_state, legacy_groups, make_model
    from l0_regions.training_data import Budget, sha, source_identity
    from tools.diagnose_local_cnn_learning import run_root, checkpoint_for
    from tools.local_cnn_interaction_runtime import support_binding
    from tools.local_cnn_reference_runtime import evaluator, probe_updates
    from tools.local_cnn_reference_fixed_probe import probe_fixed_weights
    from tools.local_cnn_interaction_updates import _query, _support

    if a.output.exists():
        raise FileExistsError(a.output)
    # Admit report storage before doing GPU work; never overwrite a result.
    a.output.parent.mkdir(parents=True, exist_ok=True)
    root = run_root(a.run)
    cp = checkpoint_for(root) if a.checkpoint is None else a.checkpoint.resolve()
    if not cp.is_file() or not cp.is_relative_to(root):
        raise ValueError('Checkpoint must belong to the bound local CNN experiment')
    print(f'CLONED DEBUG comparison | run={root}\ncheckpoint={cp}', flush=True)
    # One snapshot load is valid even if another worker atomically replaces latest.
    saved = torch.load(cp, map_location='cpu', weights_only=False)
    if saved['format'] != FORMAT or hash_state({k: v for k, v in saved.items()
                                                if k != 'content_sha256'}) != saved['content_sha256']:
        raise ValueError('Checkpoint integrity mismatch')
    identity, state = saved['identity'], saved['state']
    if identity['source'] != source_identity():
        raise ValueError('Checkpoint runtime source differs; matching production source required')
    index = root/'inventory/index.json'
    if identity.get('local_cnn') is None or identity['cache_sha256'] != sha(index):
        raise ValueError('Checkpoint/input is not the bound local CNN experiment')
    if identity['debug'] and not a.allow_debug:
        raise ValueError('DEBUG checkpoint requires explicit --allow-debug')
    if identity['precision'] != 'FP32':
        raise ValueError('This comparison requires the original FP32 precision contract')
    ds = Dataset(index, 'inner_train', identity['debug'])
    val = Dataset(index, 'inner_val', identity['debug'])
    torch.set_num_threads(a.workers)
    budget = Budget(int(a.cuda_gib*2**30), int(a.rss_gib*2**30))
    total = torch.cuda.get_device_properties(0).total_memory
    if not 0 < budget.cuda_bytes < total:
        raise ValueError('CUDA budget must leave device headroom')
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/total)
    torch.cuda.reset_peak_memory_stats()
    net = make_model(ds, budget, identity['debug'], 'retained')
    net.load_state_dict(saved['model'], strict=True)
    net.eval()
    memory = tree_to(state['memory'], 'cuda')
    batch = state['batch']
    schedule = list(groups(ds, batch, ds.meta['config']['seed'], state['epoch']))
    if not a.fixed_only and a.steps > len(schedule):
        raise ValueError(f'DEBUG steps exceed full schedule ({len(schedule)})')
    context = LiveContext(ds, batch)
    counts = identity.get('support_training', {}).get('patients')
    episodes = None
    if counts:
        episodes = PatientEpisodes(ds.rows, list(legacy_groups(ds, batch,
            ds.meta['config']['seed'], state['epoch'])), counts,
            ds.meta['config']['seed'], state['epoch']).bind(memory)
    resident = int(a.resident_gib*2**30)
    loader = Loader(ds, a.workers, resident, budget.rss_bytes)
    vl = Loader(val, a.workers, resident, budget.rss_bytes, store=loader.store)
    model_hash = hash_state(net.state_dict())
    memory_hash = hash_state(memory)
    cases = []
    started = time.perf_counter()
    with torch.no_grad():
        for split, data, reader in [('train', ds, loader), ('validation', val, vl)]:
            for name in select_cases(data.rows, a.cases_per_split):
                ids = [i for i, row in enumerate(data.rows) if row['case_id'] == name]
                rows = [data.rows[i] for i in ids]
                if {r['target'] for r in rows} != {0, 1}:
                    raise ValueError(f'Selected case {name} lacks P/U; no nicer case substituted')
                support_binding(memory, ds.rows, rows[0]['patient_group'])
                parts = []
                for start in tqdm(range(0, len(ids), batch), desc=f'CT {split} {name}'):
                    budget.check()
                    query = reader.get(ids[start:start+batch]).to('cuda')
                    parts.append(net.local(query).detach())
                    del query
                cases.append(dict(split=split, indices=ids, rows=rows, reader=reader,
                    embeddings=torch.cat(parts), truth=torch.tensor([r['target'] for r in rows], device='cuda')))
    evaluate = evaluator(cases, memory, ds.rows, batch, budget, hash_state(net.local.state_dict()))
    def support(ids):
        return support_binding(memory, ds.rows, ds.rows[ids[0]]['patient_group'], episodes)
    if a.fixed_only:
        cursor = state.get('next_batch')
        is_next = (state['phase'] == 'optimization' and type(cursor) is int
                   and 0 <= cursor < len(schedule))
        position = cursor if is_next else 0
        ids = schedule[position]
        query = _query(loader.get(ids).to('cuda'), ids, context, torch.device('cuda:0'))
        bound_support, records, group = _support(support(ids), ids, context, torch.device('cuda:0'))
        with torch.no_grad():
            embeddings = net.local(query).detach()
        truth = torch.tensor([ds.rows[i]['target'] for i in ids], device='cuda', dtype=torch.long)
        print(f'FIXED-WEIGHT ONLY | zero optimizer updates | tile={position}/{len(schedule)} '
              f'| actual batch={len(ids)} | saved next tile={is_next}', flush=True)
        comparison = probe_fixed_weights(net, embeddings, bound_support,
            policies=a.transfer_policies, truth=truth, budget=budget,
            loss_context=context, indices=ids, evaluation_provider=evaluate)
        comparison['native_tile_binding'] = dict(indices=ids, record_ids=[ds.rows[i]['id'] for i in ids],
            schedule_position=position, saved_next_optimization_tile=is_next,
            selection=('exact saved next optimization tile' if is_next else
                'first deterministic complete-schedule tile; no saved next optimization tile available in this phase'),
            saved_phase=state['phase'], saved_next_batch=cursor, actual_batch=len(ids),
            positive_count=int((truth==1).sum()), unobserved_count=int((truth==0).sum()),
            original_physical_batch=batch, input_tensor_shape=list(query.images.shape),
            input_sha256=hash_state({k:v for k,v in vars(query).items() if k!='_verified_signature'}),
            support_record_ids=records, query_group=group)
        del query
    else:
        print(f"DEBUG | {a.steps} updates/branch, original batch={batch}, "
              f"CNN channels={ds.meta['local_cnn']['channels']}, L1=2x128D/4heads, L2={len(net.l2)} "
              f"| {torch.cuda.get_device_name()}", flush=True)
        comparison = probe_updates(net, steps=a.steps, train_tiles=schedule,
            batch_provider=lambda ids: loader.get(ids).to('cuda'), support_provider=support,
            loss_context=context, physical_batch=batch, budget=budget,
            training=ds.meta['base']['training'], seed=ds.meta['config']['seed'],
            evaluation_provider=evaluate)
    if hash_state(net.state_dict()) != model_hash or hash_state(memory) != memory_hash:
        raise AssertionError('Original loaded model or epoch support memory changed')
    report = dict(format='local_cnn_reference_comparison_debug_v1', actual_CT=True,
        checkpoint=str(cp), checkpoint_content_sha256=saved['content_sha256'],
        snapshot=dict(epoch=state['epoch'], step=state['step'], phase=state['phase']),
        checkpoint_debug=identity['debug'], run=str(root), comparison=comparison,
        source=dict(production=source_identity(), diagnostic={p: sha(ROOT/p) for p in (
            'tools/diagnose_local_cnn_reference.py', 'tools/local_cnn_reference_l1.py',
            'tools/local_cnn_reference_runtime.py', 'tools/local_cnn_reference_transfer.py',
            'tools/local_cnn_reference_fixed_probe.py')}),
        input_contract=dict(margin_mm=identity['local_cnn']['margin_mm'],
            local_cnn=identity['local_cnn'], train_observations=len(ds), validation_observations=len(val),
            optimization_schedule=context.audit, candidate_subset='explicit diagnostic cases; every candidate retained',
            case_selection='fixed sorted spread of cases with both P/U, independent of model scores; cases without positives remain in the complete training schedule',
            selected_cases=[dict(split=c['split'], case_id=c['rows'][0]['case_id'], records=len(c['indices'])) for c in cases],
            support='saved detached epoch L0; unchanged episode selection during updates; full eligible support during evaluation',
            Basic_CP_changed=False, original_masks_changed=False,
            L0_L2_initial_weights_and_loss_formulas_preserved=True,
            alignment_query_gradient_route_changed_by_joint_BN=True,
            saved_teacher_and_optimizer_exact_resume=False),
        resources=dict(gpu=torch.cuda.get_device_name(), gpu_count=torch.cuda.device_count(),
            gpu_total_bytes=total, cuda_budget_bytes=budget.cuda_bytes, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), rss_bytes=psutil.Process().memory_info().rss,
            rss_budget_bytes=budget.rss_bytes, resident_bytes=resident, workers=a.workers,
            cpu_logical=psutil.cpu_count(), available_ram_bytes=psutil.virtual_memory().available,
            original_physical_batch=batch, gradient_accumulation=1, effective_batch=batch),
        wall_seconds=time.perf_counter()-started, original_model_and_memory_preserved=True,
        production_checkpoint_written=False, production_training_started=False,
        production_ready=False, full_training=False, full_evaluation=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    from tools.summarize_local_cnn_reference import format_summary
    print(format_summary(report), flush=True)
    print(f'REPORT: {a.output}\nShort cloned comparison completed; original experiment unchanged. Not a full accuracy evaluation.', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases-per-split', type=int, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--fixed-only', action='store_true', help='Zero optimizer updates; fixed same-tile transfer/BN controls')
    parser.add_argument('--transfer-policies', nargs='+', choices=('raw_columns', 'affine_relations', 'affine_relations_zero_out_bias'))
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--cuda-gib', type=float, required=True)
    parser.add_argument('--rss-gib', type=float, required=True)
    parser.add_argument('--resident-gib', type=float, required=True)
    parser.add_argument('--allow-debug', action='store_true')
    args = parser.parse_args()
    if args.fixed_only:
        if args.steps is not None or not args.transfer_policies or len(set(args.transfer_policies))!=len(args.transfer_policies):
            parser.error('--fixed-only requires explicit unique --transfer-policies and no --steps')
    elif args.steps is None or args.steps <= 0 or args.transfer_policies is not None:
        parser.error('Update comparison requires positive --steps; --transfer-policies belongs to --fixed-only')
    if (min(args.cases_per_split, args.workers) <= 0
            or not all(math.isfinite(v) and v > 0 for v in (args.cuda_gib, args.rss_gib, args.resident_gib))
            or args.resident_gib >= args.rss_gib):
        parser.error('Explicit positive diagnostic counts and resource limits required')
    from tools.local_cnn_device import select
    select(args.gpu)
    diagnose(args)


if __name__ == '__main__':
    main()
