"""Short actual CUDA control of ranking-active native CT updates.

Full physical32 and every original candidate of one actual case are retained.
The small saved support cohort is explicitly DEBUG, never final evaluation.
Both cloned branches use the new runtime selection/gradient path. No production
checkpoint, ready marker, long training or model-default change is made.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil
import torch

from hiercp_v222.v1_execution import tree_to
from l0_local_cnn.data import Dataset, CropStore
from l0_regions.donor_learning import LiveContext, validate_rows
from l0_regions.training import FORMAT, hash_state, make_model
from l0_regions.training_data import Budget, sha, source_identity
from tools.diagnose_local_cnn_learning import checkpoint_for
from tools.local_cnn_interaction_runtime import support_binding
from tools.local_cnn_reference_runtime import probe_updates, select_update_tiles
from tools.local_cnn_reference_transfer import POLICIES
from tools.verify_reference_l1_ct_debug import evaluate


def verify(a):
    if a.output.exists():
        raise FileExistsError('Previous report preserved')
    if (a.steps <= 0 or a.workers < 2 or not 0 < a.resident_gib < a.rss_gib
            or a.cuda_gib <= 0):
        raise ValueError('Explicit short DEBUG updates and resource headroom required')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; no CPU fallback')
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    budget = Budget(int(a.cuda_gib*2**30), int(a.rss_gib*2**30))
    if not 0 < budget.cuda_bytes < min(total, free):
        raise ValueError('Explicit smoke CUDA limit exceeds current available headroom')
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/total)
    torch.set_num_threads(a.workers)
    cp = checkpoint_for(a.run.resolve())
    cp_sha = sha(cp)
    assignment_sha = sha(a.assignment)
    saved = torch.load(cp, map_location='cpu', weights_only=False)
    if (saved.get('format') != FORMAT or not saved['identity'].get('debug')
            or saved['identity'].get('source') != source_identity()
            or hash_state({k:v for k,v in saved.items() if k!='content_sha256'}) != saved['content_sha256']):
        raise ValueError('Matching integrity-verified local CNN DEBUG checkpoint required')
    index = a.run/'inventory/index.json'
    if saved['identity']['cache_sha256'] != sha(index):
        raise ValueError('Checkpoint and native CT inventory binding differs')
    ds = Dataset(index, 'inner_train', True)
    existing = [r for r in ds.rows if r['case_id']==a.case]
    if not existing:
        raise ValueError('Requested actual CT case is not in this DEBUG checkpoint')
    donor = {k:existing[0][k] for k in ('donor_case_id','donor_component','donor_group')}
    assignment = json.loads(a.assignment.read_text(encoding='utf-8'))
    native = [r for r in assignment if r['case_id']==a.case]
    positives = [r for r in native if r['target']==1]
    unobserved = [r for r in native if r['target']==0]
    raw = next(r for r in ds.meta['raw_records'] if r['case_id']==a.case)
    if len(unobserved)!=128 or len(positives)!=len(raw['positives']) or not positives:
        raise ValueError('Every original positive and all 128 original comparison centers required')
    # Retain native IDs/coordinates/components. The explicitly reported DEBUG
    # same-donor binding is shared by both branches, as in the previous CT smoke.
    case_rows = [dict(r, **donor, bounds={'edges':i})
                 for i,r in enumerate(positives+unobserved)]
    # Other bound DEBUG training cases stay in the context. This retains their
    # eligible support IDs and their schedule/class/multiplicity normalization.
    rows = case_rows+[r for r in ds.rows if r['case_id']!=a.case]
    validate_rows(rows)
    context = LiveContext(SimpleNamespace(rows=rows), 32)
    schedule = list(context.order)
    selection = select_update_tiles(schedule, steps=a.steps, loss_context=context,
        physical_batch=32, selection_policy='rankable_full_batch_prefix')
    if any(tile['case_id']!=a.case for tile in selection['selected_tiles']):
        raise ValueError('Short local CT control must use its explicit complete-case native tiles')
    print('ACTUAL CUDA DEBUG | physical32 | all case observations=' + str(len(case_rows))
          + ' | original DEBUG cohort=' + str(len(rows))
          + ' | selected original schedule=' + str(selection['selected_schedule_indices']), flush=True)
    net = make_model(ds, budget, True, 'retained')
    net.load_state_dict(saved['model'], strict=True)
    net.eval()
    memory = tree_to(saved['state']['memory'], 'cuda')
    original_model = hash_state(net.state_dict())
    original_memory = hash_state(memory)
    support, support_ids, query_group = support_binding(memory, ds.rows, case_rows[0]['patient_group'])
    store = CropStore(ds.meta, a.workers, int(a.resident_gib*2**30), budget.rss_bytes)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    causal_probe = getattr(a, 'causal_probe', False)
    def fitted_evaluate(model, names):
        from tools.local_cnn_reference_causal import target_head
        from tools.diagnose_local_cnn_learning import score_summary
        from tools.v22_candidate_order import record_key
        from tools.v22_rank_objective import ranking_metrics
        if names != [a.case]:
            raise ValueError('Local causal smoke must retain its explicit complete fitted case')
        vectors=[]
        for start in range(0,len(case_rows),32):
            budget.check()
            ids=list(range(start,min(start+32,len(case_rows))))
            query=store.batch([case_rows[i] for i in ids],ids).to('cuda')
            vectors.append(model.local(query).detach())
        truth=torch.tensor([r['target'] for r in case_rows],device='cuda')
        scores,trace=target_head(model,torch.cat(vectors),support,32,truth,case_rows)
        metrics,details=ranking_metrics(scores.cpu(),truth.cpu(),[a.case]*len(case_rows),
            candidate_keys=[record_key(r) for r in case_rows])
        return dict(train=dict(cases=[dict(case_id=a.case, records=len(case_rows),
            all_case_candidates_retained=True,metrics=metrics,score=score_summary(scores,truth),
            observed_ranks=details[0]['observed_ranks'],trace=trace)]),
            scope='complete actual fitted case; explicit DEBUG saved support; not full evaluation')
    comparison = probe_updates(net, steps=a.steps, train_tiles=schedule,
        batch_provider=lambda ids: store.batch([rows[i] for i in ids], ids).to('cuda'),
        support_provider=lambda ids: (support, support_ids, query_group),
        loss_context=context, physical_batch=32, budget=budget,
        training=ds.meta['base']['training'], seed=ds.meta['config']['seed'],
        evaluation_provider=lambda model: evaluate(model, case_rows, store, support, budget),
        selection_policy='rankable_full_batch_prefix', transfer_policy=a.transfer_policy,
        causal_probe=causal_probe, fitted_evaluation_provider=fitted_evaluate if causal_probe else None)
    if (hash_state(net.state_dict())!=original_model or hash_state(memory)!=original_memory
            or sha(cp)!=cp_sha or sha(a.assignment)!=assignment_sha):
        raise AssertionError('Original model/support/checkpoint/native assignment changed')
    for branch in comparison['branches']:
        for step in branch['updates']:
            if (step['physical_batch']!=32 or step['ranking_pairs']<=0
                    or step['terms']['ranking_loss']<=0
                    or step['ranking_parameter_gradient']['status']!='MEASURED'):
                raise AssertionError('Actual ranking-active full physical update not verified')
            if causal_probe and not step['causal']['shadow_full_matches_actual_update']:
                raise AssertionError('Exact full AdamW counterfactual parity not verified')
        print('DEBUG '+branch['branch']+' | rank pairs=' + str([s['ranking_pairs'] for s in branch['updates']])
              + ' | rank-only CNN/L1/L2=' + str({k:branch['updates'][-1]['ranking_parameter_gradient']
                    ['module_gradient_norms'][k] for k in ('CNN','L1','L2')})
              + ' | optimizer deltas=' + str(branch['parameter_delta_norms']), flush=True)
    report = dict(debug=True, diagnostic_only=True, actual_CT=True, actual_CUDA=True,
        full_training=False, full_evaluation=False, production_ready=False,
        production_default_changed=False, production_checkpoint_written=False,
        production_training_started=False, original_model_support_rng_preserved=True,
        checkpoint=str(cp), checkpoint_sha256=cp_sha, assignment_sha256=assignment_sha,
        case=a.case, all_original_positive_anchors=len(positives), comparison_centers=128,
        all_case_observations=len(case_rows), full_debug_cohort_observations=len(rows),
        support_records=len(support_ids), support_record_ids=support_ids,
        support_scope='verified saved DEBUG memory; not complete production cohort',
        donor_binding=donor, original_masks_changed=False, Basic_CP_changed=False,
        loss_normalization_scope='unchanged complete DEBUG LiveContext, not selected update subset',
        physical_batch=32, gradient_accumulation=1, effective_batch=32,
        CNN_config=ds.meta['local_cnn'], L1_layers=2, L2_layers=len(net.l2),
        hidden_dim=128, heads=4, total_parameters=sum(p.numel() for p in net.parameters()),
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
        comparison=comparison, causal_probe=causal_probe, elapsed_seconds=time.perf_counter()-started,
        resources=dict(gpu=torch.cuda.get_device_name(), gpu_count=torch.cuda.device_count(),
            gpu_total_bytes=total, free_bytes_at_start=free, cuda_limit_bytes=budget.cuda_bytes,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), rss_bytes=psutil.Process().memory_info().rss,
            rss_limit_bytes=budget.rss_bytes, resident_limit_bytes=int(a.resident_gib*2**30),
            workers=a.workers, cpu_logical=psutil.cpu_count(), available_ram_bytes=psutil.virtual_memory().available),
        torch_version=torch.__version__, python=platform.python_version(),
        source=dict(production=source_identity(), diagnostic={p:sha(ROOT/p) for p in (
            'tools/verify_reference_rankable_ct_debug.py', 'tools/local_cnn_reference_runtime.py',
            'tools/local_cnn_reference_l1.py', 'tools/local_cnn_reference_transfer.py') + ((
            'tools/local_cnn_reference_causal.py','tools/local_cnn_reference_objective_probe.py',
            'tools/local_cnn_reference_mode_probe.py','tools/local_cnn_reference_target_signal.py') if causal_probe else ())}),
        limitations=['One complete case with DEBUG support is mechanical smoke, not full accuracy.',
            'Fresh optimizer and reference BatchNorm: not exact resume or learned production replacement.',
            'Ranking-only gradient measurement adds a diagnostic derivative pass; timing is not production speed.'])
    with a.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print('REPORT: '+str(a.output), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'work/local_cnn_experiment_resume_DEBUG_20261001/resumed')
    parser.add_argument('--assignment', type=Path, default=ROOT/'work/v222_v1_full_training_20260924/cache/pair_assignment.json')
    parser.add_argument('--case', default='liver_66')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--transfer-policy', choices=POLICIES, required=True)
    parser.add_argument('--causal-probe', action='store_true')
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--cuda-gib', type=float, required=True)
    parser.add_argument('--rss-gib', type=float, required=True)
    parser.add_argument('--resident-gib', type=float, required=True)
    verify(parser.parse_args())


if __name__=='__main__':
    main()
