"""Actual-CT DEBUG: reopened compact sources must build unseen epoch3 targets.

Uses the immutable three-case fixture, original physical2/train8/full model,
and actual listwise CUDA backward/update. It neither resumes server training
nor claims full-cohort evaluation or quality. Every output directory is new.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', type=Path, default=ROOT/'work/v19_checked_DEBUG')
    parser.add_argument('--inventory', type=Path,
        default=ROOT/'work/v22_cumulative_u16_DEBUG_20261006/inventory.json')
    parser.add_argument('--bank', type=Path,
        default=ROOT/'work/v1x_real_CT_DEBUG_20261003_prepare/fixture.pt')
    parser.add_argument('--legacy-data', type=Path,
        default=ROOT/'work/runs/v1.9/preparation/debug-20261008/data')
    parser.add_argument('--legacy-ref', default='3b119fc',
        help='Immutable Git revision containing the buggy compact-source helper')
    return parser.parse_args(argv)


def actual_update(batch, initial, config, budget):
    import psutil
    import torch
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp_v1x import u_bridge_training as engine
    from hiercp_v1x.comparison_training import objective
    from tools.verify_comparison_gpu_debug import gradient_norms
    net = HierarchicalPyGPlacementModel(**config['model']).cuda()
    parameters = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if parameters != 10434532 or trainable != parameters:
        raise AssertionError('The complete original model is required')
    net.load_state_dict(initial['model'], strict=True)
    engine.restore_rng(initial['rng'])
    net.train()
    training = config['training']
    optimizer = torch.optim.AdamW(net.parameters(), lr=training['lr'],
        weight_decay=training['weight_decay'], fused=training['fused_optimizer'])
    if {id(p) for group in optimizer.param_groups for p in group['params']} != {
            id(p) for p in net.parameters() if p.requires_grad}:
        raise AssertionError('Optimizer must include every trainable parameter')
    scaler = torch.amp.GradScaler('cuda', enabled=training['amp'])
    before = engine.digest(net.state_dict())
    budget.check()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    began = time.perf_counter()
    times[0].record()
    batch = batch.pin_memory().to('cuda', non_blocking=True)
    times[1].record()
    with torch.autocast('cuda', enabled=training['amp']):
        result = net(batch)
        loss, terms = objective(result.scores, result.consistency, arm='native_listwise')
    times[2].record()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Actual CT listwise loss is not finite')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    times[3].record()
    receipt = engine.gradient_receipt(net, engine._model_contract(net, True))
    norms = gradient_norms(net)
    if receipt['gradient_present'] != 1085 or receipt['missing'] or not receipt['finite']:
        raise AssertionError('All1085 original parameter gradients must be finite')
    clipped = torch.nn.utils.clip_grad_norm_(net.parameters(), training['grad_clip'],
        error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    times[4].record()
    times[4].synchronize()
    after = engine.digest(net.state_dict())
    history = engine.restore_optimizer_history(optimizer, 1)
    if before == after:
        raise AssertionError('Real optimizer update did not change model weights')
    budget.check()
    timings = {name: times[i].elapsed_time(times[i+1])/1000 for i, name in enumerate(
        ('host_to_device_seconds', 'forward_seconds', 'backward_seconds', 'optimizer_seconds'))}
    scores = [row.detach().float().cpu().tolist() for row in result.scores]
    ranks = [1 + sum(value > row[0] for value in row[1:]) for row in scores]
    report = dict(actual_CUDA=True, actual_CT=True, parameters=parameters,
        trainable_parameters=trainable, model_config=config['model'],
        physical_source_batch=2, candidates_per_source=8, accumulation_steps=1,
        data_parallel_workers=1, effective_source_batch=2, genuine_optimizer_updates=1,
        scores=scores, listwise_loss=float(loss.detach()),
        ranking_term=float(terms['ranking'].detach()),
        two_view_consistency=float(result.consistency.detach()),
        training_batch_rank_diagnostic=dict(ranks=ranks, mrr=sum(1/r for r in ranks)/len(ranks),
            quality_metric=False), gradient=receipt, gradient_norms=norms,
        gradient_before_clip=float(clipped), optimizer_history=history,
        initial_model_state_sha256=before, updated_model_state_sha256=after,
        optimizer_state_sha256=engine.digest(optimizer.state_dict()),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        process_rss_bytes=psutil.Process().memory_info().rss,
        available_ram_bytes=psutil.virtual_memory().available, **timings,
        wall_seconds=time.perf_counter()-began,
        full_training=False, full_evaluation=False, checkpoint_saved=False)
    report['physical_sources_per_compute_second'] = 2/sum(timings.values())
    return report


def verify(a):
    from tools.current_gpu import select_record
    selected = select_record(a.gpu)
    import numpy as np
    import torch
    from tools.verify_comparison_cache_debug import read, sha, canonical
    from tools.verify_comparison_gpu_debug import tree_stats
    from tools.verify_comparison_preparation_debug import fingerprint, compare
    from hiercp_v1x.comparison_experiment import FORMAT, FILES, digest
    from hiercp_v1x.scope_probe_support import activate_original, state_digest
    from hiercp_v1x import bounded_scope, comparison_source_cache
    from hiercp_v1x.comparison_data import ComparisonData
    from hiercp_v1x.comparison_preparation import prepared_provider
    from hiercp_v1x.comparison_inputs import local_mask_provider
    from hiercp_v1x.comparison_upper_cache import compact_upper_provider
    from hiercp_v1x.comparison_views import parallel_view_provider
    from hiercp_v1x.comparison_sample_cache import sample_cache_provider
    from hiercp_v1x.host_memory import PressureBudget, pressure_aware_provider, _evict_provider
    from hiercp_v1x.preparation_reuse import preparation_reuse
    from hiercp_v1x import u_bridge_training as engine
    from hiercp_v1x.u_bridge_data import _hash

    a.reference = a.reference.resolve(strict=True)
    a.legacy_data = a.legacy_data.resolve(strict=True)
    a.output = a.output.resolve()
    manifest = read(a.reference/'experiment.json')
    source_cache = Path(manifest['prepared_data_root']).resolve(strict=True)
    if a.output.exists() or any(a.output.is_relative_to(p) or p.is_relative_to(a.output)
            for p in (a.reference, a.legacy_data, source_cache)):
        raise FileExistsError('Use a new DEBUG output disjoint from every reference cache')
    if (manifest['format'] != FORMAT or manifest['debug'] is not True or manifest['workers'] != 4
            or manifest['explicit_batch_candidates'] != [2]
            or manifest['sha256'] != digest({k:v for k,v in manifest.items() if k != 'sha256'})):
        raise ValueError('Sealed actual three-CT physical2/worker4 fixture required')
    frozen_before = {name:sha(ROOT/name) for name in FILES}
    if (frozen_before != manifest['helpers']
            or sha(a.inventory) != manifest['baseline']['inventory_sha256']
            or sha(a.bank) != manifest['baseline']['bank_fixture_sha256']):
        raise ValueError('Frozen source helpers, actual CT inventory or bank differs')
    original = Path(manifest['original']['source'])
    if (canonical(activate_original(original)) != manifest['original']
            or canonical(bounded_scope.install(10, expected_snapshot_root=original)) != manifest['scope']):
        raise ValueError('Original neural source or exact10mm bounded scope differs')
    from hiercp.tensor import collect_runtime_resources, configure_runtime
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one explicitly selected actual CUDA device is required')
    torch.set_num_threads(manifest['workers'])
    config = copy.deepcopy(manifest['config'])
    configure_runtime(deterministic=config['runtime']['deterministic'],
        allow_tf32=config['runtime']['allow_tf32'], cudnn_benchmark=config['runtime']['cudnn_benchmark'])
    cuda_bytes = int(manifest['cuda_gib']*2**30)
    capacity = torch.cuda.get_device_properties(0).total_memory
    if not 0 < cuda_bytes < capacity:
        raise ValueError('Original CUDA budget must leave actual driver headroom')
    torch.cuda.set_per_process_memory_fraction(cuda_bytes/capacity)
    budget = PressureBudget(cuda_bytes, int(manifest['rss_gib']*2**30),
        resident_bytes=int(manifest['resident_gib']*2**30))
    initial = torch.load(a.reference/'initial.pt', map_location='cpu', weights_only=False)
    if (initial['contract_sha256'] != manifest['sha256']
            or initial['model_sha256'] != state_digest(initial['model'])):
        raise ValueError('Full original initial model state differs')
    fixture = torch.load(a.bank, map_location='cpu', weights_only=False, mmap=True)
    bank = fixture['prototype_bank']
    if bank.fingerprint() != manifest['prototype_fingerprint']:
        raise ValueError('Original actual-CT prototype bank differs')
    del fixture
    a.output.mkdir(parents=True, exist_ok=False)
    legacy_bytes = subprocess.check_output(['git', '-c', 'safe.directory='+ROOT.as_posix(),
        'show', a.legacy_ref+':hiercp_v1x/comparison_source_cache.py'], cwd=ROOT)
    legacy_path = a.output/'legacy_comparison_source_cache.py'
    with legacy_path.open('xb') as stream:
        stream.write(legacy_bytes)
    spec = importlib.util.spec_from_file_location('hiercp_v1x._source_scope_debug_legacy', legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    references = [a.reference/'experiment.json', a.reference/'initial.pt', a.inventory, a.bank,
        *[Path(row['path']) for row in manifest['baseline']['source_samples']]]
    reference_before = {str(p):sha(p) for p in references}
    cache_before = {str(p):tree_stats(p) for p in (a.legacy_data, source_cache)}
    helper_before = sha(ROOT/'hiercp_v1x/comparison_source_cache.py')
    raw = read(a.inventory)['raw_records']
    resources = collect_runtime_resources('cuda', storage_path=a.output)
    request = dict(debug=True, actual_CT=True, selected_device=selected, resources=resources,
        original_config=config, original_manifest_sha256=manifest['sha256'],
        legacy_git_ref=a.legacy_ref, legacy_helper_sha256=sha(legacy_path),
        current_helper_sha256=helper_before, harness_sha256=sha(Path(__file__)),
        raw_inventory_sha256=sha(a.inventory), bank_sha256=sha(a.bank),
        fixture_sources=manifest['baseline']['source_samples'], fixture_source_count=3,
        selected_training_sources=['liver_5:0','liver_6:0'], actual_source_count=2,
        physical_batch=2, effective_batch=2, workers=4, view_epoch=3,
        arm='native_listwise', candidate_keys=['P']+[f'U:{i}' for i in range(14,21)],
        trigger=dict(case_id='liver_5', sample_index=0, key='U:14', center=[328,178,416]),
        production_source_samples=187, debug_source_fraction=2/187,
        planned_optimizer_updates=1, full_training=False, full_evaluation=False,
        quality_verified=False, original_40_epoch_config_preserved=True,
        physical_batch_basis='Original immutable physical2 calibration; no change to physical batch',
        preprocessing='Borrow only verified whole-case fields/static upper; never canonical targets or sample layouts')
    write_new(a.output/'request.json', request)
    print(json.dumps(request, allow_nan=False), flush=True)
    result = dict(debug=True, actual_CT=True, full_training=False, full_evaluation=False,
        quality_verified=False, current_helper_sha256=helper_before,
        harness_sha256=request['harness_sha256'])
    providers = []
    began = time.perf_counter()
    try:
        core = pressure_aware_provider(ComparisonData)
        fresh_type = sample_cache_provider(parallel_view_provider(compact_upper_provider(local_mask_provider(core))))
        fixed_type = prepared_provider(core)
        legacy_type = legacy.source_cache_provider(core)
        kwargs = dict(source_samples=manifest['baseline']['source_samples'], raw_records=raw,
            config=config, bank=bank, workers=4, resident_bytes=int(manifest['resident_gib']*2**30),
            budget=budget, regions_dir=source_cache/'regions')
        # Deliberately instantiate the unwrapped classes. This context borrows
        # fields/upper only: its yielded ReusingPreparation local-graph wrapper
        # is never used, so unseen canonical candidates really must be built.
        with preparation_reuse(core, [source_cache, a.legacy_data]):
            fresh = fresh_type(root=a.output/'fresh_data', **kwargs)
            fixed = fixed_type(root=a.output/'reopened_data', **kwargs)
            old = legacy_type(root=a.output/'legacy_data', **kwargs)
            providers.extend((fresh, fixed, old))
            if canonical(fresh.examples('train')+fresh.examples('val')) != manifest['samples']:
                raise ValueError('Complete immutable three-case fixture inventory differs')
            ids = [row['index'] for row in fresh.examples('train')]
            if len(ids) != 2 or [fresh._example(i)['id'] for i in ids] != ['liver_5:0','liver_6:0']:
                raise AssertionError('Original physical2 source identities required')
            expected_sources = {}
            source_publications = {}
            for index in ids:
                ex = fresh._example(index)
                case, organ, depth, _ = fresh._case(ex['case_id'])
                source, prepared = fresh._source(ex, case, organ, depth)
                # vars(), not dataclass fields(): the dynamic marker is part
                # of the regression's exact source contract.
                expected_sources[ex['id']] = fingerprint(dict(source=vars(source), prepared=vars(prepared)))
                binding = comparison_source_cache.source_binding(fresh, ex, case)
                directory = a.legacy_data/'source_prepared'/comparison_source_cache._digest(binding)
                if not directory.is_dir():
                    raise FileNotFoundError('Preserved v1 compact source is required: '+str(directory))
                source_publications[ex['id']] = dict(directory=str(directory), binding=binding,
                    metadata_sha256=sha(directory/'metadata.json'), payload_sha256=sha(directory/'payload.pt'))
                for provider in (fixed, old):
                    target = provider.root/'source_prepared'/directory.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(directory, target)
                    if (sha(target/'metadata.json') != sha(directory/'metadata.json')
                            or sha(target/'payload.pt') != sha(directory/'payload.pt')):
                        raise AssertionError('Byte-exact immutable legacy source copy failed')
            write_new(a.output/'source_publications.json', source_publications)
            ex = old._example(ids[0])
            case, organ, depth, occupied = old._case(ex['case_id'])
            source, prepared = old._source(ex, case, organ, depth)
            center = tuple(ex['native_centers'][14])
            if center != (328,178,416) or getattr(prepared,'v1x_bounded_scope_contract',None) is not None:
                raise AssertionError('Exact old missing-marker/U14 failure precondition not observed')
            runtime = old._runtime()
            graph_config = runtime.schema.graph_config_from_dict(config['graph'])
            candidate = old._candidate(center, case, source, organ, depth, occupied)
            specs = runtime.cache.build_generation_specs([candidate], old._regions(case), bank, config=graph_config)
            failure = None
            try:
                runtime.local.build_local_graph(case, source, specs[0], full_organ_mask=organ,
                    organ_depth=depth, config=graph_config, rng=np.random.default_rng(config['seed']),
                    ct_clip=tuple(config['ct_clip']), prepared_source=prepared)
            except ValueError as error:
                if str(error) != 'Native or other-scope prepared source cannot enter bounded construction':
                    raise
                failure = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            if failure is None or old.source_cache_stats['disk_hits'] != 1 or old.source_cache_stats['original_builds'] != 0:
                raise AssertionError('The exact legacy disk-reopen construction failure did not reproduce')
            failure.update(actual_CT=True, debug=True, candidate_key='U:14', center=list(center),
                legacy_helper_sha256=sha(legacy_path), source_disk_hits=1, source_rebuilds=0,
                marker_missing=True, canonical_targets_before=len(list(old.graph_dir.glob('*.pt'))))
            result['legacy_reproduced'] = failure
            write_new(a.output/'legacy_failure.json', failure)
            print('DEBUG legacy reproduced: liver_5 source0 U:14 (328,178,416), missing scope marker after disk reopen', flush=True)
            del source, prepared, case, organ, depth, occupied, candidate, specs
            _evict_provider(old)
            source_checks = {}
            for index in ids:
                ex = fixed._example(index)
                case, organ, depth, _ = fixed._case(ex['case_id'])
                source, prepared = fixed._source(ex, case, organ, depth)
                observed = fingerprint(dict(source=vars(source), prepared=vars(prepared)))
                check = compare(expected_sources[ex['id']], observed)
                check['marker'] = getattr(prepared, 'v1x_bounded_scope_contract', None)
                if not check['exact_values_layout_metadata_equal'] or check['marker'] != manifest['scope']['contract_sha256']:
                    raise AssertionError('Restored complete source values/layout/runtime marker differ')
                source_checks[ex['id']] = check
            if fixed.source_cache_stats['disk_hits'] != 2 or fixed.source_cache_stats['original_builds'] != 0:
                raise AssertionError('Both genuine sources must reopen v1 cache with no rebuild')
            write_new(a.output/'source_comparison.json', source_checks)
            del source, prepared, case, organ, depth
            target_paths = [fixed.graph_dir/(_hash(fixed._binding(fixed._example(i), center))+'.pt')
                for i in ids for center in fixed._centers_for(fixed._example(i),
                    fixed.candidate_keys(i,'native_listwise',3))]
            if any(path.exists() for path in target_paths) or list(fixed.graph_dir.glob('*.pt')):
                raise AssertionError('All16 fixed candidates must be absent before construction')
            print('DEBUG fixed source markers exact; building16 previously absent epoch3 canonical targets', flush=True)
            start = time.perf_counter()
            fixed_batch = fixed.batch(ids,'native_listwise',3,True,full=False)
            fixed_seconds = time.perf_counter()-start
            actual_fingerprint = fingerprint(vars(fixed_batch))
            if (fixed.stats['local_builds'] != 16 or not all(path.is_file() for path in target_paths)
                    or fixed.stats['disk_hits'] != 0):
                raise AssertionError('Fixed regression must construct every unseen candidate, with no canonical cache hits')
            write_new(a.output/'fixed_inputs.json', actual_fingerprint)
            print('DEBUG independently building fresh-source reference16 canonical targets', flush=True)
            start = time.perf_counter()
            fresh_batch = fresh.batch(ids,'native_listwise',3,True,full=False)
            fresh_seconds = time.perf_counter()-start
            input_comparison = compare(fingerprint(vars(fresh_batch)), actual_fingerprint)
            write_new(a.output/'input_comparison.json', input_comparison)
            if not input_comparison['exact_values_layout_metadata_equal'] or fresh.stats['local_builds'] != 16:
                raise AssertionError('Fresh/reopened whole batches differ in tensor bytes/layout/graph metadata')
            graph_comparisons = []
            for path in target_paths:
                left = torch.load(fresh.graph_dir/path.name, map_location='cpu', weights_only=False)
                right = torch.load(path, map_location='cpu', weights_only=False)
                check = compare(fingerprint(left), fingerprint(right))
                if not check['exact_values_layout_metadata_equal']:
                    raise AssertionError('Fresh/reopened canonical graph differs: '+path.name)
                graph_comparisons.append(dict(binding=right['binding'], comparison=check,
                    fresh_publication_sha256=sha(fresh.graph_dir/path.name), reopened_publication_sha256=sha(path)))
                del left, right
            write_new(a.output/'canonical_comparison.json', graph_comparisons)
            result.update(exact_sources=source_checks, exact_inputs=input_comparison,
                previously_unbuilt_canonical_targets=16, canonical_cache_hits=0,
                fixed_input_seconds=fixed_seconds, fresh_input_seconds=fresh_seconds,
                fixed_provider=fixed.report(), fresh_provider=fresh.report())
            del fresh_batch, actual_fingerprint
            for provider in providers:
                _evict_provider(provider)
            gc.collect()
        print('DEBUG actual CUDA full10434532 listwise forward/backward/optimizer, physical2 x8, view epoch3', flush=True)
        result['cuda'] = actual_update(fixed_batch, initial, config, budget)
        result['original_references_preserved'] = reference_before == {str(p):sha(p) for p in references}
        result['existing_cache_inventory_size_mtime_preserved'] = cache_before == {
            str(p):tree_stats(p) for p in (a.legacy_data, source_cache)}
        result['frozen_helpers_preserved'] = frozen_before == {name:sha(ROOT/name) for name in FILES}
        result['current_helper_unchanged_during_run'] = helper_before == sha(ROOT/'hiercp_v1x/comparison_source_cache.py')
        result['harness_unchanged_during_run'] = request['harness_sha256'] == sha(Path(__file__))
        result['wall_seconds'] = time.perf_counter()-began
        result['limitations'] = 'Actual original three-CT fixture; two training sources at one DEBUG epoch3 update. No server resume, full187-source training, validation129 evaluation or production quality claim.'
        if not all(result[key] for key in ('original_references_preserved',
                'existing_cache_inventory_size_mtime_preserved', 'frozen_helpers_preserved',
                'current_helper_unchanged_during_run', 'harness_unchanged_during_run')):
            raise AssertionError('Reference/source preservation check failed')
        write_new(a.output/'report.json', result)
        print(json.dumps(dict(report=str(a.output/'report.json'), legacy_reproduced=True,
            newly_built_targets=16, exact_sources=True, exact_inputs=True, actual_CUDA_updates=1,
            full_model_parameters=10434532, full_training=False, full_evaluation=False)), flush=True)
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(),
            wall_seconds=time.perf_counter()-began)
        write_new(a.output/'failure.json', result)
        raise
    finally:
        for provider in providers:
            _evict_provider(provider)


if __name__ == '__main__':
    verify(parse())
