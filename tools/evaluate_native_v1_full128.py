"""Read-only actual native30 V1 BEST on the retained complete P+128U cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    for name in ('original-source', 'inventory', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--prototype', type=Path, required=True)
    p.add_argument('--region-cache', type=Path, help='Explicit read-only original-config region cache; missing cases fail')
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--physical-batch-candidates', type=int, nargs='+', required=True)
    for name in ('cuda-gib', 'rss-gib', 'resident-gib'):
        p.add_argument('--' + name, type=float, required=True)
    p.add_argument('--debug-fresh-model', action='store_true')
    p.add_argument('--debug-case-ids', nargs='+')
    a = p.parse_args(argv)
    if (a.gpu < 0 or a.workers < 2 or not 0 < a.resident_gib < a.rss_gib
            or a.cuda_gib <= 0 or a.physical_batch_candidates != sorted(set(a.physical_batch_candidates))
            or min(a.physical_batch_candidates) < 1):
        p.error('Explicit GPU, parallel workers, ordered physical batches and positive budgets required')
    if a.debug_fresh_model:
        if a.checkpoint is not None or not a.debug_case_ids:
            p.error('Fresh untrained DEBUG requires explicit cases and forbids a production checkpoint')
    elif a.checkpoint is None or a.debug_case_ids is not None:
        p.error('Production requires the actual saved BEST and all retained validation cases')
    return a


def execution_source():
    from hiercp_v1x.historical_evaluation import sha
    names = ('tools/evaluate_native_v1_full128.py', 'tools/local_cnn_device.py',
             'hiercp_v1x/native30_checkpoint.py', 'hiercp_v1x/native30_geometry.py',
             'hiercp_v1x/native30_upper.py', 'hiercp_v1x/native30_data_contract.py', 'hiercp_v1x/transition_evaluation.py',
             'hiercp_v1x/historical_evaluation.py', 'tools/v22_candidate_order.py',
             'tools/v22_rank_objective.py', 'l0_regions/donor_data.py', 'l0_regions/donor_learning.py',
             'hiercp_v22/data.py', 'hiercp/preparation_runtime.py', 'config/server_gpu_allocations.json')
    return {name: sha(ROOT / name) for name in names}


def evaluate(a):
    from tools.local_cnn_device import select
    select(a.gpu)
    import numpy as np
    import psutil
    import torch
    from torch_geometric.data import Batch
    from hiercp_v1x.historical_evaluation import (ResourceBudget, assert_new_destination,
        sha, write_new, calibrate, encode_original_fields, unpack_fields)
    from hiercp_v1x.native30_checkpoint import (activate_native30_source, load_native30,
        load_debug_native30, verify_native30_source)
    from hiercp_v1x.native30_geometry import Native30Geometry
    from hiercp_v1x.native30_upper import build_native30_upper
    from hiercp_v1x.transition_evaluation import (validate_cohort, run_scoring,
        PreparedL0Batch, SCORING_FORMAT)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one selected actual CUDA device required; no CPU fallback')
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    cuda_bytes, rss_bytes = int(a.cuda_gib * 2**30), int(a.rss_gib * 2**30)
    if not 0 < cuda_bytes < min(total, free):
        raise ValueError('CUDA budget exceeds current available allocation or leaves no headroom')
    process = psutil.Process()
    if a.workers > len(process.cpu_affinity()) or rss_bytes > psutil.virtual_memory().available + process.memory_info().rss:
        raise ValueError('Requested CPU/RAM budget exceeds current effective host availability')
    preserved = [a.original_source, a.prototype, a.inventory]
    if a.checkpoint is not None:
        preserved.append(a.checkpoint)
    assert_new_destination(a.output, preserved)
    a.output.mkdir(parents=True, exist_ok=False)
    code = execution_source()
    before = {str(path.resolve()): sha(path) for path in (a.inventory, a.prototype)}
    if a.checkpoint is not None:
        before[str(a.checkpoint.resolve())] = sha(a.checkpoint)
    activate_native30_source(a.original_source)
    inventory = json.loads(a.inventory.read_text(encoding='utf8'))
    debug = a.debug_fresh_model
    cohort = validate_cohort(inventory, case_ids=a.debug_case_ids, debug=debug)
    write_new(a.output / 'request.json', dict(settings={key: str(value) if isinstance(value, Path) else value
        for key, value in vars(a).items()}, source_sha256=code, preserved_files=before,
        cohort={key: value for key, value in cohort.items() if key not in ('rows', 'by_case')},
        checkpoint_selection='fresh untrained DEBUG' if debug else 'explicit original own-task BEST',
        training_started=False, optimizer_updates=0))
    torch.cuda.set_per_process_memory_fraction(cuda_bytes / total)
    torch.set_num_threads(a.workers)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    budget = ResourceBudget(cuda_bytes, rss_bytes)
    original_config = json.loads((a.original_source / 'config/train.json').read_text(encoding='utf8'))
    runtime = original_config.get('runtime', {})
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime.get('allow_tf32', False))
    torch.backends.cudnn.allow_tf32 = bool(runtime.get('allow_tf32', False))
    torch.backends.cudnn.benchmark = bool(runtime.get('cudnn_benchmark', False))
    bundle = (load_debug_native30(a.original_source, original_config, a.prototype, budget)
              if debug else load_native30(a.checkpoint, a.prototype, a.original_source, budget))
    write_new(a.output / 'checkpoint_receipt.json', bundle.receipt)
    if not debug:
        print('NATIVE30 CHECKPOINT | BEST epoch=' + str(bundle.receipt['selected_epoch'])
              + ' | saved training seed=' + str(bundle.receipt['training_seed'])
              + ' | evaluation seed=' + str(bundle.receipt['evaluation_seed'])
              + ' | strict weights/prototype=PASS', flush=True)
    model = bundle.model.eval()
    geometry = Native30Geometry(a.inventory, bundle.config, bundle.source,
        workers=a.workers, resident_bytes=int(a.resident_gib * 2**30), rss_bytes=rss_bytes,
        output=a.output / 'geometry', debug=debug, debug_case_ids=a.debug_case_ids)
    # Resource accounting is separate from the activated historical hiercp
    # namespace; the legacy checkout legitimately has no preparation_runtime.
    import importlib.util
    spec = importlib.util.spec_from_file_location('native30_resource_accounting', ROOT / 'hiercp/preparation_runtime.py')
    accounting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(accounting)
    allocation = accounting.snapshot()
    if a.workers > allocation['cpu_capacity'] or rss_bytes > allocation['rss_bytes'] + allocation['available_memory_bytes']:
        raise ValueError('Requested resources exceed measured affinity/cgroup allocation')
    preparation = geometry.prepare()
    resources = dict(gpu=torch.cuda.get_device_name(), GPU_count=1, total_VRAM_bytes=total,
        free_VRAM_bytes=free, cuda_limit_bytes=cuda_bytes, rss_limit_bytes=rss_bytes,
        resident_bytes=int(a.resident_gib * 2**30), workers=a.workers, effective_allocation=allocation,
        model=type(model).__name__, model_configuration=bundle.config['model'],
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        ROI_margin_mm=30, context_outer_mm=28, context_shells_mm=[4, 12, 28],
        dense_input_shape_per_crop=[5, 48, 48, 48], precision='saved-source AMP' if bundle.config['training']['amp'] else 'FP32',
        validation_cases=len(cohort['case_ids']), records=len(geometry), observed_P=cohort['observed_P'],
        unobserved_U=cohort['unobserved_U'], all_128U=True, two_genuine_sampled_views=True,
        graph_size_statistics=preparation.get('sampled_two_view_summary'),
        geometry_cache='byte-accounted resident canonical tensors and file-backed mmap',
        input_resolution='native patient CT spacing; saved fixed48 dense crop',
        sampling_hops=bundle.config['graph'].get('sample_hops'), time_window='not applicable',
        inference_only=True, optimizer_updates=0, debug=debug)
    write_new(a.output / 'resources.json', resources)
    print('NATIVE30 ACTUAL CUDA | ' + json.dumps({key: resources[key] for key in
        ('gpu', 'parameters', 'validation_cases', 'records', 'ROI_margin_mm', 'context_outer_mm', 'debug')}), flush=True)
    use_amp = bool(bundle.config['training']['amp'])
    timings = []; audits = []; lookup = {r['id']: i for i, r in enumerate(geometry.rows)}
    def encode(cpu):
        budget(); torch.cuda.synchronize(); start = time.perf_counter()
        gpu = cpu.to('cuda')
        with torch.no_grad(), torch.autocast('cuda', enabled=use_amp):
            result = encode_original_fields(model, gpu).float()
        torch.cuda.synchronize(); budget()
        timings.append(dict(stage='L0_H2D_forward', observations=len(cpu), seconds=time.perf_counter()-start))
        return result
    batch, calibration = calibrate(encode, geometry, geometry, a.physical_batch_candidates, budget, arm='native30')
    write_new(a.output / 'calibration.json', calibration)
    def provider(rows):
        cpu = geometry.get([lookup[r['id']] for r in rows], epoch=0)
        return PreparedL0Batch(cpu, tuple(r['id'] for r in rows), False)
    def score(features, rows):
        graph, prototype, audit = build_native30_upper(bundle, geometry, rows,
            output=a.output / 'regions', region_cache=a.region_cache)
        audits.append(audit)
        upper = SimpleNamespace(patient_batch=Batch.from_data_list([graph]).to('cuda'),
            prototype_batch=Batch.from_data_list([prototype]).to('cuda'), counts=(len(rows),),
            case_ids=(rows[0]['case_id'],))
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.autocast('cuda', enabled=use_amp):
            values = model._score_upper(upper, unpack_fields(features))[0].float()
        torch.cuda.synchronize(); budget()
        timings.append(dict(stage='joint_whole_case_L1_L2', case_id=rows[0]['case_id'],
                            observations=len(rows), seconds=time.perf_counter()-start))
        return dict(scores=values, contract=dict(format=SCORING_FORMAT,
            scored_record_ids=[r['id'] for r in rows], l0_only_chunking=True, upper_chunking=False,
            upper_execution='single_joint_case', query_GT_in_forward=False, upper_invocations=1,
            annotation_derived_recipient_inputs=True, class_target_argument_passed=False))
    torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    report = run_scoring(inventory, provider, encode, score, l0_batch_size=batch,
        case_ids=a.debug_case_ids, debug=debug)
    source_proof = verify_native30_source(bundle.source_proof)
    if any(sha(path) != digest for path, digest in before.items()) or execution_source() != code:
        raise ValueError('Original data/checkpoint/prototype or evaluator changed during evaluation')
    overlap = sorted(set(bundle.bank.training_case_ids) & set(cohort['case_ids']))
    report.update(checkpoint=bundle.receipt, resources=resources, preparation=preparation,
        calibration=calibration, physical_L0_batch=batch, upper_physical_batch='one complete case',
        actual_CUDA=True, raw_CT_execution_verified=True, callback_internal_execution_verified=True,
        optimizer_updates=0, training_started=False, full_evaluation=not debug,
        quality_verified=False, production_ready=False, source_proof=source_proof,
        source_provenance_at_training='UNKNOWN', preserved_files=before, old_evidence_preserved=True,
        prototype_validation_overlap=overlap, independently_heldout_quality=False,
        annotation_derived_recipient_inputs=True, blind_recommendation_quality_verified=False,
        original_learned_operators_preserved=True, original_single_patient_topology_equivalent=False,
        external_donor_input_adaptation=audits, elapsed_seconds=time.perf_counter()-start,
        stage_timings=timings, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        final_process_RSS_bytes=process.memory_info().rss, CP_started=False, nnunet_started=False)
    write_new(a.output / 'report.json', report)
    metrics = report['metrics']
    print('RESULT | native30 existing BEST' if not debug else 'RESULT | untrained actual-CT CUDA DEBUG', flush=True)
    print('MRR={:.6f} Hit@1={:.6f} pair-win={:.6f} loss={:.6f}'.format(
        metrics['case_first_P_mrr'], metrics['case_hit_at_1'], metrics['P_U_pair_win_rate'], metrics['P_U_softplus_loss']), flush=True)
    print('PROTOTYPE/CURRENT-VAL OVERLAP: ' + json.dumps(overlap), flush=True)
    print('REPORT: ' + str((a.output / 'report.json').resolve()), flush=True)
    return report


def main():
    evaluate(parse())


if __name__ == '__main__':
    main()
