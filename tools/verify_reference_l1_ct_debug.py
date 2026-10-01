"""Actual native-CT physical32 DEBUG integration, never production training.

Use all five observed anchors and all 128 original comparison centers of the
specified case. The saved eight-observation DEBUG support table is explicitly
kept as DEBUG evidence. Each cloned arm performs one full-objective update;
no checkpoint/ready artifact or long training is created.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import torch

from hiercp_v222.v1_execution import rng_state, restore_rng, tree_to
from l0_local_cnn.data import Dataset, CropStore
from l0_regions.donor_learning import LiveContext, forward_loss, configuration, validate_rows
from l0_regions.execution_pipeline import gradient_check_batched
from l0_regions.resident import check_verified
from l0_regions.training import FORMAT, hash_state, make_model
from l0_regions.training_data import Budget, sha, source_identity
from tools.diagnose_local_cnn_learning import checkpoint_for, score_summary
from tools.local_cnn_interaction_runtime import support_binding
from tools.local_cnn_interaction_updates import _module_norms
from tools.local_cnn_reference_l1 import clone_reference
from tools.local_cnn_reference_transfer import POLICIES, clone_reference_control
from tools.v22_candidate_order import record_key
from tools.v22_rank_objective import ranking_metrics


def evaluate(model, rows, store, support, budget):
    """Every case observation, physical32 query encoding; no query GT input."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        scores = []
        with torch.no_grad():
            state = model.prepare_support(*support)
            for start in range(0, len(rows), 32):
                budget.check()
                ids = list(range(start, min(start+32, len(rows))))
                batch = store.batch([rows[i] for i in ids], ids).to('cuda')
                output = model.predict_embeddings(model.local(batch), state)
                logits = output['logits'].float()
                if logits.shape != (len(ids), 2) or not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError('Invalid native CT evaluation logits')
                scores.append((logits[:,1]-logits[:,0]).cpu())
                del batch, output
        scores = torch.cat(scores)
        truth = torch.tensor([row['target'] for row in rows], dtype=torch.long)
        metrics, detail = ranking_metrics(scores, truth, [rows[0]['case_id']]*len(rows),
                                          candidate_keys=[record_key(row) for row in rows])
        return dict(observations=len(rows), positives=int(truth.sum()),
                    comparison_centers=int((truth==0).sum()), all_candidates_retained=True,
                    physical_query_chunk=32, metrics=metrics, score=score_summary(scores,truth),
                    observed_ranks=detail[0]['observed_ranks'], finite=True,
                    scope='one complete original case; DEBUG support; not full validation')
    finally:
        for module, mode in modes:
            module.training = mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path,
                        default=ROOT/'work/local_cnn_experiment_resume_DEBUG_20261001/resumed')
    parser.add_argument('--assignment', type=Path,
                        default=ROOT/'work/v222_v1_full_training_20260924/cache/pair_assignment.json')
    parser.add_argument('--case', default='liver_66')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'work/reference_l1_physical32_DEBUG_20261001/report.json')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cuda-gib', type=float, default=12.)
    parser.add_argument('--rss-gib', type=float, default=32.)
    parser.add_argument('--resident-gib', type=float, default=8.)
    parser.add_argument('--transfer-policy', choices=POLICIES,
                        help='Explicit new reference initialization; omission preserves historical raw transfer')
    parser.add_argument('--fixed-probe-policies', nargs='+', choices=POLICIES,
                        help='Optional zero-update controls on this exact actual physical32 tile')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Existing DEBUG report preserved')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; no CPU fallback')
    if args.workers < 2 or not 0 < args.resident_gib < args.rss_gib:
        raise ValueError('Parallel workers and explicit resident/RSS headroom required')
    total = torch.cuda.get_device_properties(0).total_memory
    budget = Budget(int(args.cuda_gib*2**30), int(args.rss_gib*2**30))
    if not 0 < budget.cuda_bytes < total:
        raise ValueError('Explicit CUDA budget must leave device headroom')
    torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/total)
    torch.set_num_threads(args.workers)
    cp = checkpoint_for(args.run.resolve())
    checkpoint_before = sha(cp)
    assignment_sha = sha(args.assignment)
    saved = torch.load(cp, map_location='cpu', weights_only=False)
    if (saved.get('format') != FORMAT or not saved['identity'].get('debug')
            or saved['identity'].get('source') != source_identity()
            or hash_state({key:value for key,value in saved.items() if key!='content_sha256'}) != saved['content_sha256']):
        raise ValueError('Matching byte-verified local CNN DEBUG checkpoint required')
    index = args.run/'inventory/index.json'
    if saved['identity']['cache_sha256'] != sha(index):
        raise ValueError('DEBUG checkpoint and original inventory differ')
    ds = Dataset(index, 'inner_train', True)
    meta = ds.meta
    debug_case_rows = [row for row in ds.rows if row['case_id']==args.case]
    if not debug_case_rows:
        raise ValueError('Explicit actual CT case is absent from the bound DEBUG inventory')
    donor = {key:debug_case_rows[0][key] for key in ('donor_case_id','donor_component','donor_group')}
    assignment = json.loads(args.assignment.read_text(encoding='utf-8'))
    original_rows = [row for row in assignment if row['case_id']==args.case]
    positives = [row for row in original_rows if row['target']==1]
    negatives = [row for row in original_rows if row['target']==0]
    raw = next(row for row in meta['raw_records'] if row['case_id']==args.case)
    if len(negatives)!=128 or len(positives)!=len(raw['positives']) or not positives:
        raise ValueError('All 128 native comparison centers and every original positive required')
    # Keep actual original observation ID/center/component. Only the fixed
    # donor is bound to the DEBUG checkpoint's explicitly reported contract.
    rows = [dict(row, **donor, bounds={'edges':i})
            for i,row in enumerate(positives+negatives)]
    validate_rows(rows)
    context = LiveContext(SimpleNamespace(rows=rows), 32)
    tiles = [ids for ids in context.order if len(ids)==32 and
             any(rows[i]['target'] for i in ids) and any(not rows[i]['target'] for i in ids)]
    if not tiles:
        raise ValueError('Complete original-case schedule has no physical32 P/U tile')
    ids = tiles[0]
    net = make_model(ds, budget, True, 'retained')
    net.load_state_dict(saved['model'], strict=True); net.eval()
    memory = tree_to(saved['state']['memory'], 'cuda')
    support, support_ids, query_group = support_binding(memory, ds.rows, rows[0]['patient_group'])
    before_net = hash_state(net.state_dict())
    before_memory = hash_state(memory)
    before_grad = hash_state({name:value.grad for name,value in net.named_parameters()})
    modes = [(module,module.training) for module in net.modules()]
    caller_rng = rng_state()
    store = CropStore(meta, args.workers, int(args.resident_gib*2**30), budget.rss_bytes)
    query = store.batch([rows[i] for i in ids], ids).to('cuda')
    check_verified(query)
    query_hash = hash_state({key:value for key,value in vars(query).items() if key!='_verified_signature'})
    if len(query)!=32 or query.indices.tolist()!=ids:
        raise ValueError('Exact physical32 native CT tile binding failed')
    branches = []
    settings = meta['base']['training']
    started = time.perf_counter()
    fixed = None
    if args.fixed_probe_policies:
        from tools.local_cnn_reference_fixed_probe import probe_fixed_weights
        if len(set(args.fixed_probe_policies))!=len(args.fixed_probe_policies):
            raise ValueError('Duplicate fixed control policies')
        with torch.no_grad():
            embeddings = net.local(query).detach()
        truth = torch.tensor([rows[i]['target'] for i in ids], device='cuda', dtype=torch.long)
        fixed = probe_fixed_weights(net,embeddings,support,policies=args.fixed_probe_policies,
            truth=truth,budget=budget,loss_context=context,indices=ids)
    try:
        for name in ('legacy','reference'):
            budget.check()
            if name=='reference':
                candidate, transfer = (clone_reference(net) if args.transfer_policy is None else
                    clone_reference_control(net,policy=args.transfer_policy))
            else:
                candidate, transfer = copy.deepcopy(net), dict(architecture='unchanged_legacy_L1')
            if candidate.checkpoint_support:
                raise ValueError('This physical32 control requires the original retained-activation policy')
            named = [(key,value) for key,value in candidate.named_parameters() if value.requires_grad]
            initial = [value.detach().clone() for _,value in named]
            optimizer = torch.optim.AdamW([value for _,value in named], lr=settings['lr'],
                                          weight_decay=settings['weight_decay'], fused=settings['fused_optimizer'])
            before = evaluate(candidate, rows, store, support, budget)
            candidate.train()
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            plan = candidate.fit_support_clusters(*support)
            targets = torch.tensor([rows[i]['target'] for i in ids], device='cuda', dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            step_start = time.perf_counter()
            loss, terms = forward_loss(candidate,query,support,plan,targets,None,context,configuration(),indices=ids)
            torch.cuda.synchronize(); forward_end = time.perf_counter()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite original full objective')
            loss.backward(); torch.cuda.synchronize(); backward_end=time.perf_counter()
            gradient_check_batched(candidate)
            gradients = _module_norms(named, [value.grad for _,value in named])
            grad_norm = torch.nn.utils.clip_grad_norm_([value for _,value in named],settings['grad_clip'],error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize(); step_end=time.perf_counter()
            update_peak = torch.cuda.max_memory_allocated()
            deltas = _module_norms(named, [value.detach()-old for (_,value),old in zip(named,initial)])
            if not all(gradients[key]>0 and deltas[key]>0 for key in ('CNN','readout_fusion','L1','L2')):
                raise AssertionError('Native CNN/readout/L1/L2 did not receive gradients and updates')
            budget.check()
            after = evaluate(candidate, rows, store, support, budget)
            if hash_state({key:value for key,value in vars(query).items() if key!='_verified_signature'})!=query_hash:
                raise AssertionError('Native CT tile was changed by cloned update')
            branches.append(dict(branch=name,transfer=transfer,optimizer='fresh AdamW; no saved moments',
                precision='FP32',physical_batch=32,effective_batch=32,gradient_accumulation=1,
                update_indices=ids,observed_in_batch=int(targets.sum()),loss=float(loss.detach()),
                terms={key:float(value.detach()) for key,value in terms.items()},
                all_trainable_parameters_finite_gradient=True,module_gradient_norms=gradients,
                gradient_norm_before_clip=float(grad_norm),module_parameter_delta_norms=deltas,
                parameters=sum(value.numel() for value in candidate.parameters()),
                timing=dict(forward_seconds=forward_end-step_start,backward_seconds=backward_end-forward_end,
                    gradient_clip_optimizer_seconds=step_end-backward_end,update_seconds=step_end-step_start,
                    excludes='CT loading/transfer, input hashing, teacher fit, evaluation, checkpoint IO'),
                update_peak_cuda_bytes=update_peak,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),before=before,after=after))
            print(f'DEBUG {name} | physical32 | loss={float(loss.detach()):.6g} | '
                  f'update={step_end-step_start:.3f}s | all{len(rows)} observations evaluated',flush=True)
            del candidate,optimizer,named,initial,loss,terms
    finally:
        restore_rng(caller_rng)
    if (hash_state(net.state_dict())!=before_net or hash_state(memory)!=before_memory
            or hash_state({name:value.grad for name,value in net.named_parameters()})!=before_grad
            or any(module.training!=mode for module,mode in modes)
            or hash_state(rng_state())!=hash_state(caller_rng) or sha(cp)!=checkpoint_before):
        raise AssertionError('Original checkpoint/model/support/mode/grad/RNG changed')
    report = dict(debug=True,diagnostic_only=True,actual_CT=True,full_training=False,full_evaluation=False,
        production_default_changed=False,production_checkpoint_written=False,production_ready=False,
        original_checkpoint_unchanged=True,original_model_support_rng_preserved=True,
        case=args.case,checkpoint=str(cp),checkpoint_sha256=checkpoint_before,
        assignment_sha256=assignment_sha,margin_mm=meta['local_cnn']['margin_mm'],
        CNN_config=meta['local_cnn'],L1_layers=2,L2_layers=2,hidden_dim=128,heads=4,
        case_observations=len(rows),all_original_positive_anchors=len(positives),comparison_centers=128,
        support_records=len(support_ids),support_patients=int(support[1].max())+1,
        support_record_ids=support_ids,query_group=query_group,donor_binding=donor,
        support_scope='verified saved DEBUG epoch memory; not full training cohort support',
        loss=configuration(),normalization_scope='all P x U pairs and observations of this complete case; DEBUG only',
        context_audit=context.audit,physical_batch=32,cloned_optimizer_updates_per_arm=1,
        branches=branches,elapsed_seconds=time.perf_counter()-started,
        explicit_transfer_policy=args.transfer_policy, fixed_weight_comparison=fixed,
        input_tensor_shape=list(query.images.shape),native_crop_audit=query.audit,
        GPU=torch.cuda.get_device_name(),gpu_count=torch.cuda.device_count(),gpu_total_bytes=total,
        cuda_budget_bytes=budget.cuda_bytes,rss_budget_bytes=budget.rss_bytes,
        resident_budget_bytes=int(args.resident_gib*2**30),workers=args.workers,
        CPU_logical=psutil.cpu_count(),RSS_bytes=psutil.Process().memory_info().rss,
        RAM_available_bytes=psutil.virtual_memory().available,python=platform.python_version(),
        torch_version=torch.__version__,no_new_graphs=True,
        limitations=['One cloned update per arm is mechanical smoke evidence, not learning improvement.',
                     'Candidate BatchNorm running statistics start fresh; not an exact resume.',
                     'DEBUG support and one-case loss normalization do not represent production cohort.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print('REPORT:',args.output,flush=True)


if __name__=='__main__':
    main()
