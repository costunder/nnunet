"""Bounded physical ROI / original-v1 CT and CUDA cost probe, DEBUG only.

Replays the fixture's complete eight-candidate matrices and GT. Each physical
margin runs in a fresh process; no production checkpoint or learning gate is
created. Preparation and neural update times are reported separately.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def current_adapter():
    # Bind the current package before the archived source gets sys.path priority.
    # The adapter's relative imports must resolve beside this implementation.
    import hiercp_v1x
    if Path(hiercp_v1x.__file__).resolve().parent != ROOT / 'hiercp_v1x':
        raise ImportError('Archived helpers shadowed the current scope adapter package')
    path = ROOT / 'hiercp_v1x/bounded_scope.py'
    spec = importlib.util.spec_from_file_location('hiercp_v1x._bounded_scope_probe_current', path)
    if spec is None or spec.loader is None:
        raise ImportError('Bounded scope implementation cannot be loaded')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, {'path': str(path), 'sha256': sha(path)}


def child(request):
    if (request['margin'] not in (10, 20, 30) or not 1 <= request['updates'] <= 4
            or request['workers'] < 2 or request['physical_batch'] < 2):
        raise ValueError('Internal request must preserve the explicit short parallel DEBUG limits')
    from hiercp_v1x.scope_probe_support import activate_original, _gradient_groups, state_digest
    adapter, adapter_identity = current_adapter()
    source = Path(request['baseline_source']).resolve(strict=True)
    provenance = activate_original(source)
    if request.get('gpu') is not None:
        from tools.local_cnn_device import select
        select(request['gpu'])
    import numpy as np
    import psutil
    import torch
    from hiercp import local
    from hiercp.common import CasePaths, load_case, choose_source_tumor, stable_case_seed, organ_depth_mm
    from hiercp.data import collate_samples, summarize_hierarchical_batch
    from hiercp.loss import CurriculumConfig, curriculum_ranking_loss, ranking_metrics
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.sample import materialize_sample_views
    from hiercp.tensor import configure_runtime, set_seed
    margin = request['margin']
    report = dict(debug=True, scope='bounded physical ROI short original-v1 cost probe',
        training_started=False, debug_optimizer_updates=True, full_training=False,
        full_evaluation=False, production_ready=False, quality_verified=False,
        margin_mm=margin, source=provenance, adapter=adapter_identity, phase='CUDA preflight')
    report['timing_contract'] = dict(warmup_updates=0, first_update_includes_kernel_startup=True,
        scope='bounded short DEBUG; initialization and diagnostic work included where named',
        excluded=['production loader queue', 'production checkpoint save', 'complete epoch'],
        RSS_enforcement='100ms sampled peak plus explicit phase checks; not an OS hard limit')
    report['probe_tool_sha256'] = sha(__file__)
    report['requested_execution'] = {k:v for k,v in request.items() if k not in ('_request', 'report')}
    output = Path(request['report'])
    stop = threading.Event()
    peak = [psutil.Process().memory_info().rss]
    process = psutil.Process()
    rss_limit = int(request['rss_gib'] * 2**30)
    def monitor():
        while not stop.wait(.1):
            peak[0] = max(peak[0], process.memory_info().rss)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    def budget():
        peak[0] = max(peak[0], process.memory_info().rss)
        if peak[0] > rss_limit:
            raise MemoryError(f'Explicit DEBUG RSS limit exceeded: {peak[0]} > {rss_limit}')
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one actual CUDA device required; no fallback')
        total = torch.cuda.get_device_properties(0).total_memory
        cuda_limit = int(request['cuda_gib'] * 2**30)
        if not 0 < cuda_limit < total:
            raise ValueError('CUDA budget must be positive and leave device headroom')
        torch.cuda.set_per_process_memory_fraction(cuda_limit / total)
        torch.set_num_threads(request['workers'])
        config = json.loads((source / 'config/train.json').read_text())
        configure_runtime(
            deterministic=config['runtime']['deterministic'], allow_tf32=config['runtime']['allow_tf32'],
            cudnn_benchmark=config['runtime']['cudnn_benchmark'])
        if config['cache']['total_candidates'] != 8 or config['cache']['candidate_pool_size'] != 128:
            raise ValueError('Original eight-candidate / 128-pool contract required')
        report['scope_contract'] = adapter.install(margin, expected_snapshot_root=source)
        root = Path(request['fixture_dir']).resolve(strict=True)
        manifest = json.loads((root / 'fixture_manifest.json').read_text())
        raw_records = {row['case_id']: row for row in manifest['source_records']}
        raw, samples = {}, []
        report['phase'] = 'verified raw CT and canonical fixture load'
        print(f"SCOPE DEBUG | margin={margin:g} | checking raw CT and original candidates", flush=True)
        start = time.perf_counter()
        for item in manifest['files']:
            path = root / item['name']
            if sha(path) != item['sha256']:
                raise ValueError(f'Canonical fixture SHA mismatch: {path}')
            sample = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            if sample['split'] != item['split'] or len(sample['target_locals']) != 8:
                raise ValueError('Fixture split or complete candidate count mismatch')
            row = raw_records[sample['case_id']]
            if row['case_id'] not in raw:
                for kind in ('image', 'label'):
                    if sha(row[kind]) != row[kind + '_sha256']:
                        raise ValueError(f'Raw CT provenance changed: {row[kind]}')
                case = load_case(CasePaths(row['case_id'], Path(row['image']), Path(row['label'])))
                organ = (case.label == config['labels']['liver']) | (case.label == config['labels']['tumor'])
                raw[row['case_id']] = (case, organ, organ_depth_mm(organ, case.spacing))
                print(f"SCOPE DEBUG | margin={margin:g} | loaded {row['case_id']}", flush=True)
            samples.append(sample)
            budget()
        report['raw_load_and_depth_seconds'] = time.perf_counter() - start
        train = [sample for sample in samples if sample['split'] == 'train']
        validation = [sample for sample in samples if sample['split'] == 'val']
        if len(train) != request['physical_batch'] or not validation:
            raise ValueError('DEBUG fixture must contain one exact physical train batch and held-out CT')
        if set(s['case_id'] for s in train) & set(s['case_id'] for s in validation):
            raise ValueError('Train and validation patient overlap')
        report.update(GPU=torch.cuda.get_device_name(0), device_total_bytes=total,
            cuda_budget_bytes=cuda_limit, rss_budget_bytes=rss_limit, workers=request['workers'],
            physical_sample_batch=request['physical_batch'], physical_candidate_graph_batch=8*request['physical_batch'],
            effective_sample_batch=request['physical_batch'], gradient_accumulation_steps=1,
            CPU_logical=psutil.cpu_count(), RAM_available=psutil.virtual_memory().available,
            cache_candidate_pool=128, candidates_per_sample=8, epoch_for_debug_loss=29,
            model_config=config['model'], loss_and_training_config=config['training'], seed=config['seed'])
        records, lock, context = [], threading.Lock(), threading.local()
        original_payload = local.build_patch_payload
        def measured_payload(**kwargs):
            began = time.perf_counter()
            result = original_payload(**kwargs)
            mask_count = int(np.count_nonzero(result['footprint']))
            expected = int(np.count_nonzero(kwargs['footprint']))
            if mask_count != expected:
                raise AssertionError('Complete donor footprint changed inside bounded ROI')
            shape = result['ct_norm'].shape
            footprint_shape = np.asarray(kwargs['footprint'].shape)
            expected_shape = footprint_shape + 2*np.ceil(margin/kwargs['spacing']).astype(np.int64)
            expected_shape += expected_shape % 2 == 0
            if tuple(shape) != tuple(expected_shape):
                raise AssertionError('Bounded ROI grew beyond requested voxel-rounded margin')
            with lock:
                records.append(dict(case_id=context.case_id, candidate=context.candidate,
                    center=list(kwargs['center']), native_shape=list(shape), native_voxels=int(np.prod(shape)),
                    footprint_shape=footprint_shape.tolist(), full_footprint_voxels=mask_count,
                    spacing_mm=np.asarray(kwargs['spacing']).astype(float).tolist(),
                    actual_padding_mm=((np.asarray(shape)-footprint_shape)/2*kwargs['spacing']).tolist(),
                    padding_low_mm=(((np.asarray(shape)-footprint_shape)//2)*kwargs['spacing']).tolist(),
                    padding_high_mm=(((np.asarray(shape)-footprint_shape+1)//2)*kwargs['spacing']).tolist(),
                    fields_seconds=time.perf_counter()-began))
            budget()
            return result
        report['phase'] = 'bounded ROI and complete canonical graphs'
        start = time.perf_counter()
        rebuilt = []
        local.build_patch_payload = measured_payload
        try:
            for sample in samples:
                case, organ, depth = raw[sample['case_id']]
                graph_config = adapter.configure(sample['graph_config'], margin)
                source_tumor, _, _ = choose_source_tumor(case.image, case.label,
                    tumor_label=config['labels']['tumor'], selection=config['cache']['source_selection'],
                    pad=config['cache']['source_pad'], rng=np.random.default_rng(stable_case_seed(
                        config['seed'], sample['case_id'], f"sample_{sample['sample_index']}")))
                if (source_tumor.component_id != sample['source_component'] or
                    source_tumor.anchor_center != tuple(sample['candidate_centers'][0].tolist())):
                    raise ValueError('Replayed raw source differs from scored original anchor')
                context.case_id, context.candidate = sample['case_id'], 'source'
                prepared = local.prepare_local_source(case, source_tumor, full_organ_mask=organ,
                    organ_depth=depth, config=graph_config, rng=np.random.default_rng(config['seed']),
                    ct_clip=tuple(sample['ct_clip']))
                def target(index):
                    context.case_id, context.candidate = sample['case_id'], index
                    matrix = sample['target_locals'][index]['transform'].numpy()
                    spec = SimpleNamespace(center=tuple(sample['candidate_centers'][index].tolist()),
                        rotation_matrix=matrix, scale_array=np.ones(3, dtype=np.float32))
                    built = local.build_local_graph(case, source_tumor, spec, full_organ_mask=organ,
                        organ_depth=depth, config=graph_config, rng=np.random.default_rng(config['seed']),
                        ct_clip=tuple(sample['ct_clip']), prepared_source=prepared)
                    if not torch.equal(built.target_local['transform'], sample['target_locals'][index]['transform']):
                        raise AssertionError('Full cached candidate matrix changed')
                    return built
                with ThreadPoolExecutor(max_workers=request['workers']) as pool:
                    built = list(pool.map(target, range(8)))
                changed = {**sample, 'source_patch': torch.from_numpy(prepared.source_patch.astype(np.float16)),
                    'target_patches': torch.from_numpy(np.stack([item.target_patch for item in built]).astype(np.float16)),
                    'source_local': built[0].source_local, 'target_locals': [item.target_local for item in built],
                    'graph_config': graph_config.to_dict()}
                for key in ('candidate_centers', 'difficulties', 'corruptions', 'candidate_regions', 'candidate_prototypes'):
                    if not torch.equal(changed[key], sample[key]):
                        raise AssertionError(f'Original candidate / GT metadata changed: {key}')
                if changed['patient_graph'] is not sample['patient_graph'] or changed['prototype_graph'] is not sample['prototype_graph']:
                    raise AssertionError('Nonlocal hierarchy changed')
                rebuilt.append(changed)
                print(f"SCOPE DEBUG | margin={margin:g} | prepared {sample['case_id']} | 8/8 original candidates", flush=True)
        finally:
            local.build_patch_payload = original_payload
        report['ROI_and_canonical_preparation_seconds'] = time.perf_counter()-start
        report['ROI_records'] = records
        report['phase'] = 'two native views, collate and pin'
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=request['workers']) as pool:
            materialized = list(pool.map(lambda sample: materialize_sample_views(sample,
                training=sample['split']=='train', epoch=29, global_seed=config['seed']), rebuilt))
        train_cpu = collate_samples([s for s in materialized if s['split']=='train']).pin_memory()
        val_cpu = collate_samples([s for s in materialized if s['split']=='val']).pin_memory()
        report['views_collate_pin_seconds'] = time.perf_counter()-start
        report['train_input'] = summarize_hierarchical_batch(train_cpu)
        report['validation_input'] = summarize_hierarchical_batch(val_cpu)
        start = time.perf_counter()
        train_gpu, val_gpu = train_cpu.to('cuda', non_blocking=True), val_cpu.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        report['H2D_seconds'] = time.perf_counter()-start
        set_seed(config['seed'], deterministic=config['runtime']['deterministic'])
        net = HierarchicalPyGPlacementModel(**config['model']).cuda()
        report['initial_state_sha256'] = state_digest(net.state_dict())
        report['model_parameters'] = sum(p.numel() for p in net.parameters())
        if report['model_parameters'] != 10434532:
            raise AssertionError('Original neural architecture / parameter count changed')
        objective = CurriculumConfig(**{k:v for k,v in config['training'].items()
            if k in {f.name for f in dataclasses.fields(CurriculumConfig)}})
        optimizer = torch.optim.AdamW(net.parameters(), lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay'], fused=config['training']['fused_optimizer'])
        scaler = torch.amp.GradScaler('cuda', enabled=config['training']['amp'])
        report['updates'] = []
        report['phase'] = 'original full objective CUDA update'
        for step in range(request['updates']):
            budget(); net.train(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
            with torch.autocast('cuda', enabled=config['training']['amp']):
                out = net(train_gpu)
                base, _ = curriculum_ranking_loss(out.scores, train_gpu.difficulty_list(), epoch=29, config=objective)
                loss = base + config['training']['consistency_weight'] * out.consistency
            torch.cuda.synchronize(); forward=time.perf_counter()-start
            began=time.perf_counter(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.cuda.synchronize(); backward=time.perf_counter()-began
            gradients = _gradient_groups(net)
            connectivity = dict(trainable=sum(p.requires_grad for p in net.parameters()),
                with_gradient=sum(p.requires_grad and p.grad is not None for p in net.parameters()),
                scope='Core module norms checked; genuinely absent surface roles can contribute zero gradients')
            if not all(bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.grad is not None):
                raise AssertionError('Nonfinite complete objective gradient')
            torch.nn.utils.clip_grad_norm_(net.parameters(), config['training']['grad_clip'])
            probe = next(net.score_head.parameters()); before=probe.detach().clone()
            began=time.perf_counter(); scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
            if torch.equal(before, probe.detach()):
                raise AssertionError('Optimizer did not change scalar ranking head')
            row=dict(step=step+1, loss=float(loss.detach()), forward_seconds=forward, backward_seconds=backward,
                optimizer_seconds=time.perf_counter()-began, update_with_diagnostics_seconds=time.perf_counter()-start,
                gradient_groups=gradients, gradient_connectivity=connectivity,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())
            report['updates'].append(row)
            print(json.dumps(dict(DEBUG=True, margin_mm=margin, **row)), flush=True)
            del out, base, loss
        report['phase'] = 'held-out CT short evaluation'
        net.eval(); torch.cuda.synchronize(); start=time.perf_counter()
        with torch.no_grad(), torch.autocast('cuda', enabled=config['training']['amp']):
            out=net(val_gpu); top1,mrr=ranking_metrics(out.scores)
            scores=[value.detach().float().cpu().tolist() for value in out.scores]
        torch.cuda.synchronize()
        report['validation']=dict(seconds=time.perf_counter()-start, top1=top1, MRR=mrr, scores=scores,
            case_ids=list(val_gpu.case_ids), quality_verified=False)
        budget(); report.update(status='PASS', phase='complete', actual_CT=True, actual_CUDA=True,
            process_peak_rss_bytes=peak[0], fixture_manifest_sha256=sha(root/'fixture_manifest.json'))
        write_new(output, report)
    except Exception:
        report.update(status='FAILED', error=traceback.format_exc(), process_peak_rss_bytes=peak[0])
        write_new(output, report)
        raise
    finally:
        stop.set(); watcher.join()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--_request', help=argparse.SUPPRESS)
    parser.add_argument('--baseline-source'); parser.add_argument('--fixture-dir'); parser.add_argument('--output')
    parser.add_argument('--margin-mm', type=float, nargs='+'); parser.add_argument('--physical-batch', type=int)
    parser.add_argument('--workers', type=int); parser.add_argument('--updates', type=int)
    parser.add_argument('--cuda-gib', type=float); parser.add_argument('--rss-gib', type=float)
    parser.add_argument('--gpu', type=int)
    args=parser.parse_args()
    if args._request:
        child(json.loads(Path(args._request).read_text())); return
    required=('baseline_source','fixture_dir','output','margin_mm','physical_batch','workers','updates','cuda_gib','rss_gib')
    if any(getattr(args,key) is None for key in required):
        parser.error('All scope, input, physical batch, workers, DEBUG update and resource arguments are required')
    if args.physical_batch<2 or args.workers<2 or not 1<=args.updates<=4 or len(set(args.margin_mm))!=len(args.margin_mm):
        parser.error('Explicit parallel DEBUG batch/workers >=2, unique margins and 1..4 updates required')
    if any(m not in (10,20,30) for m in args.margin_mm):
        parser.error('This explicit initial physical-scope probe accepts 10/20/30 mm only')
    root=Path(args.output).resolve(); root.mkdir(parents=True,exist_ok=False)
    reports=[]
    for margin in args.margin_mm:
        request={**vars(args), 'margin':margin, 'report':str(root/f'm{margin:g}.json')}
        request_path=root/f'm{margin:g}.request.json'; write_new(request_path,request)
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPYCACHEPREFIX':str(root/f'isolated_pyc_m{margin:g}')}
        result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--_request',str(request_path)],env=env)
        if result.returncode:
            raise RuntimeError(f'Physical scope m{margin:g} failed; diagnostics preserved in {request["report"]}; no training started')
        reports.append(json.loads(Path(request['report']).read_text()))
    if len({r['initial_state_sha256'] for r in reports})!=1:
        raise AssertionError('Scope branches did not use identical initial neural weights')
    write_new(root/'report.json',dict(debug=True, status='PASS', full_training=False, full_evaluation=False,
        production_ready=False, quality_verified=False, branches=reports,
        timing_scope='Actual bounded ROI preparation, complete canonical graphs, two original views and full objective updates; no epoch extrapolation'))
    print(f'DEBUG REPORT: {root / "report.json"}',flush=True)


if __name__=='__main__':
    main()
