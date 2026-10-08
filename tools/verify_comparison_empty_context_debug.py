"""Isolated actual-CT DEBUG of epoch8 empty recipient context and CUDA resume.

Keep all187 signed source indices, full10434532 parameters and real16 candidates.
Only indices76/77 run in this explicit physical2 diagnostic; production remains
untouched. Each requested loss independently restores the same native_listwise
checkpoint, performs one genuine update and roundtrips a DEBUG checkpoint.
This is a state-continuation test, not a production training-cursor migration.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
INDICES = (76, 77)
SOURCE_IDS = ('liver_46:0', 'liver_46:1')
KEYS = ('P', *(f'U:{i}' for i in range(49, 56)))


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--cache-sources', type=Path, nargs='+', required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True, help='New /tmp/*DEBUG* directory')
    p.add_argument('--objectives', nargs='+', choices=('native', 'native_listwise'),
                   default=['native', 'native_listwise'])
    p.add_argument('--geometry-sweep', action='store_true',
                   help='Also check full canonical target geometry for both sources and all128 native U')
    return p.parse_args(argv)


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


@contextmanager
def readonly_inputs(roots):
    """Reject this process's Python filesystem mutations in preserved inputs."""
    roots = tuple(Path(p).resolve() for p in roots)
    active = [True]

    def protected(value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        path = Path(os.fsdecode(value)).resolve()
        return any(path == root or path.is_relative_to(root) for root in roots)

    def audit(event, args):
        if not active[0]:
            return
        paths = ()
        if event == 'open':
            path, mode, flags = args
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                    isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
                paths = (path,)
        elif event == 'os.rename':
            paths = args[:2]
        elif event == 'os.link':
            # The reuse helper first tries a link, then copies on EXDEV. A
            # cross-filesystem attempt cannot mutate the source's link count.
            source, destination = args[:2]
            if protected(source) and Path(source).stat().st_dev == Path(destination).parent.stat().st_dev:
                raise PermissionError('DEBUG refuses hardlinks that change preserved source metadata')
            paths = (destination,)
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.chown', 'os.utime'):
            paths = args[:1]
        elif event == 'os.symlink':
            paths = args[1:2]
        if any(protected(path) for path in paths):
            raise PermissionError('DEBUG refused mutation of a preserved production input: ' + event)

    sys.addaudithook(audit)
    try:
        yield
    finally:
        active[0] = False


def empty_receipt(batch):
    import torch
    from hiercp_v1x.comparison_empty_context import GRAPH_PROOF_KEY, GRAPH_POLICY_KEY
    rows = {}
    for name in ('local_batch', 'local_batch_view2'):
        graph = getattr(batch, name)
        if graph is None:
            raise AssertionError('Both genuine training views are required')
        counts = torch.bincount(graph['target_context'].batch, minlength=16).tolist()
        if len(counts) != 16:
            raise AssertionError('Exactly16 original candidate graphs are required')
        proof = graph.get(GRAPH_PROOF_KEY)
        if (not torch.is_tensor(proof) or tuple(proof.shape) != (16, 4)
                or any((count == 0) != bool(row[0]) for count,row in zip(counts, proof.tolist()))):
            raise AssertionError('Actual empty geometry must carry the exact per-candidate proof')
        rows[name] = dict(target_context_nodes=counts,
            canonical_absence_evidence=proof.tolist(), policy=graph.get(GRAPH_POLICY_KEY),
            empty_candidate_rows=[i for i, n in enumerate(counts) if n == 0],
            nonempty_candidate_rows=[i for i, n in enumerate(counts) if n > 0])
    if not any(row['empty_candidate_rows'] for row in rows.values()):
        raise AssertionError('The actual empty-recipient regression did not reproduce')
    if (tuple(batch.counts) != (8, 8) or tuple(batch.bridge_indices) != INDICES
            or tuple(batch.bridge_source_ids) != SOURCE_IDS
            or tuple(tuple(k) for k in batch.bridge_candidate_keys) != (KEYS, KEYS)):
        raise AssertionError('The exact original source/candidate order changed')
    return rows


def geometry_sweep(provider):
    """Full geometry only: no feature/edge construction, sampling or truncation."""
    if provider.workers != 16:
        raise ValueError('This optional geometry sweep requires the recorded16 workers')
    runtime = provider._runtime()
    config = runtime.schema.graph_config_from_dict(provider.config['graph'])
    runtime.local._require_full_graph(config)
    result = []
    started = time.perf_counter()
    for index in INDICES:
        ex = provider._example(index)
        case, organ, depth, occupied = provider._case(ex['case_id'])
        source, prepared = provider._source(ex, case, organ, depth)
        centers = [tuple(center) for center in ex['native_centers']]
        if len(centers) != 128 or len(set(centers)) != 128:
            raise AssertionError('The complete frozen128 native centers are required')
        candidates = [provider._candidate(c, case, source, organ, depth, occupied) for c in centers]
        specs = runtime.cache.build_generation_specs(candidates, provider._regions(case),
                                                     provider.bank, config=config)
        if len(specs) != 128:
            raise AssertionError('Generation changed the full128 center inventory')

        def check(item):
            number, spec = item
            if tuple(spec.center) != centers[number]:
                raise AssertionError('Canonical geometry center order changed')
            target = runtime.local._prepare_local_target(case, spec, full_organ_mask=organ,
                organ_depth=depth, config=config, ct_clip=tuple(provider.config['ct_clip']),
                prepared_source=prepared)
            provider._check_budget()
            return dict(key=f'U:{number}', center=list(centers[number]),
                        canonical_coordinate_counts={k:len(v) for k,v in target.coordinates.items()})

        with ThreadPoolExecutor(max_workers=16, thread_name_prefix='DEBUG-geometry') as pool:
            rows = list(pool.map(check, enumerate(specs)))
        result.append(dict(source_id=ex['id'], native_candidates=rows))
    return dict(sources=result, checked_targets=256, workers=16,
        wall_seconds=time.perf_counter()-started, full_geometry=True,
        graph_edges_built=False, graph_rules_changed=False,
        scope='These two original sources only; full cohort and full129 GPU evaluation not tested')


def resumed_update(batch, saved, config, budget, objective_arm, output):
    import psutil
    import torch
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp_v1x import u_bridge_training as engine
    from hiercp_v1x.comparison_training import objective
    from hiercp_v1x.comparison_geometry_execution import current_policy, validate_resume_policy

    net = HierarchicalPyGPlacementModel(**config['model']).cuda()
    groups = engine._model_contract(net, True)
    training = config['training']
    optimizer = torch.optim.AdamW(net.parameters(), lr=training['lr'],
        weight_decay=training['weight_decay'], fused=training['fused_optimizer'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    scaler = torch.amp.GradScaler('cuda', enabled=training['amp'])
    if {id(p) for g in optimizer.param_groups for p in g['params']} != {
            id(p) for p in net.parameters() if p.requires_grad}:
        raise AssertionError('Every trainable parameter must belong to AdamW')
    net.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    scheduler.load_state_dict(copy.deepcopy(saved['scheduler']))
    scaler.load_state_dict(copy.deepcopy(saved['scaler']))
    updates = saved['state']['updates']
    history_before = engine.restore_optimizer_history(optimizer, updates)
    engine.restore_rng(saved['rng'])
    restored = dict(model=engine.digest(net.state_dict()), optimizer=engine.digest(optimizer.state_dict()),
        scheduler=engine.digest(scheduler.state_dict()), scaler=engine.digest(scaler.state_dict()),
        rng=engine.digest(engine.capture_rng()))
    if restored != {key:engine.digest(saved[key]) for key in restored}:
        raise AssertionError('Source checkpoint model/Adam/scheduler/scaler/RNG did not restore exactly')
    transition = validate_resume_policy(saved.get('recipient_context_policy'), current_policy())
    net.train()
    optimizer.zero_grad(set_to_none=True)
    budget.check()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    events[0].record()
    with torch.autocast('cuda', enabled=training['amp']):
        prediction = net(batch)
        loss, terms = objective(prediction.scores, prediction.consistency, arm=objective_arm)
    events[1].record()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Actual DEBUG loss is nonfinite')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    events[2].record()
    gradient = engine.gradient_receipt(net, groups)
    if gradient['gradient_present'] != 1085 or gradient['missing'] or not gradient['finite']:
        raise AssertionError('All1085 original parameter tensors require finite gradients')
    token_gradients = {name:float(parameter.grad.detach().float().norm())
        for name,parameter in net.named_parameters() if 'empty_context_shell.target_context_' in name}
    if not token_gradients or not any(value > 0 for value in token_gradients.values()):
        raise AssertionError('Actual empty recipient tokens did not receive a nonzero gradient')
    clipped = torch.nn.utils.clip_grad_norm_(net.parameters(), training['grad_clip'], error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    events[3].record()
    events[3].synchronize()
    history_after = engine.restore_optimizer_history(optimizer, updates+1)
    if engine.digest(net.state_dict()) == restored['model']:
        raise AssertionError('A genuine optimizer update must change model weights')
    if engine.digest(scheduler.state_dict()) != restored['scheduler']:
        raise AssertionError('A diagnostic batch must not advance the production epoch scheduler')
    budget.check()
    payload = dict(format='empty_context_state_continuation_DEBUG_v1', debug=True,
        model=engine.cpu_copy(net.state_dict()), optimizer=engine.cpu_copy(optimizer.state_dict()),
        scheduler=engine.cpu_copy(scheduler.state_dict()), scaler=engine.cpu_copy(scaler.state_dict()),
        rng=engine.cpu_copy(engine.capture_rng()),
        original_training_state=engine.cpu_copy(saved['state']),
        original_shuffle_generator=engine.cpu_copy(saved['shuffle_generator']),
        original_identity_sha256=saved['identity_sha256'], diagnostic_updates=1,
        diagnostic_objective=objective_arm, recipient_context_policy=current_policy(),
        production_resume_compatible=False)
    payload['content_sha256'] = engine.digest(payload)
    path = output/(objective_arm+'_checkpoint_DEBUG.pt')
    with path.open('xb') as stream:
        torch.save(payload, stream)
    reopened = torch.load(path, map_location='cpu', weights_only=False)
    checksum = reopened.pop('content_sha256')
    if checksum != engine.digest(reopened) or checksum != payload['content_sha256']:
        raise AssertionError('DEBUG checkpoint serialization changed continuation state')
    net.load_state_dict(reopened['model'], strict=True)
    optimizer.load_state_dict(reopened['optimizer'])
    scheduler.load_state_dict(reopened['scheduler'])
    scaler.load_state_dict(reopened['scaler'])
    engine.restore_rng(reopened['rng'])
    for key,value in dict(model=net.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=engine.capture_rng()).items():
        if engine.digest(value) != engine.digest(reopened[key]):
            raise AssertionError('DEBUG checkpoint reload changed '+key)
    engine.restore_optimizer_history(optimizer, updates+1)
    report = dict(objective_arm=objective_arm, source_checkpoint_arm='native_listwise',
        diagnostic_loss_crosscheck=objective_arm != 'native_listwise',
        actual_CUDA=True, parameters=10434532, trainable_parameters=10434532,
        physical_source_batch=2, candidate_rows=16, two_views=True,
        accumulation_steps=1, data_parallel_workers=1, effective_source_batch=2,
        genuine_optimizer_updates=1, source_updates=updates, debug_updates=updates+1,
        source_state_restored=restored, resume_policy=transition,
        loss=float(loss.detach()), loss_terms={k:float(v.detach()) for k,v in terms.items()},
        gradient=gradient, target_empty_token_gradient_norms=token_gradients,
        clipped_gradient_norm=float(clipped), optimizer_history_before=history_before,
        optimizer_history_after=history_after, checkpoint_roundtrip_exact=True,
        checkpoint=str(path), model_state_sha256=engine.digest(net.state_dict()),
        scheduler_not_advanced=True, production_cursor_not_modified=True,
        forward_seconds=events[0].elapsed_time(events[1])/1000,
        backward_seconds=events[1].elapsed_time(events[2])/1000,
        optimizer_seconds=events[2].elapsed_time(events[3])/1000,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        process_rss_bytes=psutil.Process().memory_info().rss, wall_seconds=time.perf_counter()-began)
    del reopened, payload, optimizer, scheduler, scaler, prediction, loss, net
    gc.collect()
    torch.cuda.empty_cache()
    return report


def verify(a):
    from tools.current_gpu import select_record
    selected = select_record(a.gpu)  # Complete device selection before importing torch.
    import torch
    from tools.resume_comparison_cached import sealed_arguments
    from tools.verify_comparison_cache_debug import read, sha, canonical
    from tools.verify_comparison_preparation_debug import fingerprint, compare
    from hiercp_v1x import bounded_scope, u_bridge_training as engine
    from hiercp_v1x.scope_probe_support import activate_original
    from hiercp_v1x.comparison_geometry_execution import comparison_geometry_execution
    from hiercp_v1x.comparison_data import ComparisonData
    from hiercp_v1x.comparison_preparation import prepared_provider
    from hiercp_v1x.host_memory import PressureBudget, pressure_aware_provider, _evict_provider
    from hiercp_v1x.preparation_reuse import preparation_reuse

    a.experiment = a.experiment.resolve(strict=True)
    a.inventory = a.inventory.resolve(strict=True)
    a.cache_sources = list(dict.fromkeys(p.resolve(strict=True) for p in a.cache_sources))
    a.output = a.output.resolve()
    if (a.output.exists() or not a.output.is_relative_to(Path('/tmp'))
            or 'DEBUG' not in a.output.name):
        raise ValueError('Use a new explicitly named /tmp/*DEBUG* directory')
    manifest, family, _, namespace = sealed_arguments(SimpleNamespace(
        gpu=a.gpu, arm='native_listwise', experiment=a.experiment, inventory=a.inventory,
        debug_fixture=None, debug_config=None, debug_source=None, debug_bank=None))
    if family != 'comparison' or manifest['debug'] or len(manifest['baseline']['source_samples']) != 187:
        raise ValueError('The complete signed187-source v1.9 production manifest is required')
    baseline = Path(manifest['baseline']['baseline']).resolve(strict=True)
    protected = [a.experiment, a.inventory, baseline, namespace, *a.cache_sources]
    if any(a.output.is_relative_to(p) or p.is_relative_to(a.output) for p in protected):
        raise ValueError('DEBUG output must be disjoint from every preserved input')
    if len(set(a.objectives)) != len(a.objectives):
        raise ValueError('Duplicate diagnostic objectives are not permitted')
    a.output.mkdir(parents=True, exist_ok=False)
    checkpoint = a.experiment/'native_listwise/checkpoint_latest.pt'
    checkpoint_sha = sha(checkpoint)
    result = dict(debug=True, status='RUNNING', actual_CT=True, full_training=False,
        full_evaluation=False, quality_verified=False, output=str(a.output),
        source_checkpoint=str(checkpoint), source_checkpoint_sha256=checkpoint_sha,
        production_physical_batch_unchanged=True, production_epochs=40,
        debug_physical_batch=2, source_indices=list(INDICES), source_ids=list(SOURCE_IDS),
        source_inventory_count=187, actual_debug_source_count=2, actual_debug_source_fraction=2/187,
        view_epoch=8, candidate_keys=list(KEYS), selected_device=selected,
        geometry_sweep_requested=a.geometry_sweep, harness_sha256=sha(Path(__file__)))
    write_new(a.output/'request.json', result)
    try:
        with readonly_inputs(protected), comparison_geometry_execution():
            original = Path(manifest['original']['source'])
            if canonical(activate_original(original)) != manifest['original']:
                raise ValueError('Original neural source differs')
            if canonical(bounded_scope.install(10, expected_snapshot_root=original)) != manifest['scope']:
                raise ValueError('The original exact10mm scope differs')
            from hiercp.prototype import PrototypeBank
            from hiercp.tensor import configure_runtime, collect_runtime_resources
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('Exactly one explicitly selected CUDA device is required')
            config = copy.deepcopy(manifest['config'])
            configure_runtime(deterministic=config['runtime']['deterministic'],
                allow_tf32=config['runtime']['allow_tf32'], cudnn_benchmark=config['runtime']['cudnn_benchmark'])
            torch.set_num_threads(manifest['workers'])
            capacity = torch.cuda.get_device_properties(0).total_memory
            cuda_bytes = int(manifest['cuda_gib']*2**30)
            if not 0 < cuda_bytes < capacity:
                raise ValueError('The original CUDA budget requires actual driver headroom')
            torch.cuda.set_per_process_memory_fraction(cuda_bytes/capacity)
            budget = PressureBudget(cuda_bytes, int(manifest['rss_gib']*2**30),
                                    resident_bytes=int(manifest['resident_gib']*2**30))
            result['resources'] = collect_runtime_resources('cuda', storage_path=a.output)
            bank_path = baseline/'shared/prototype_bank.pt'
            if sha(bank_path) != manifest['baseline']['bank_sha256']:
                raise ValueError('Original bank file changed')
            bank = PrototypeBank.load(bank_path)
            if bank.fingerprint() != manifest['prototype_fingerprint']:
                raise ValueError('Original bank fingerprint changed')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            content_sha = saved.pop('content_sha256', None)
            ownership = read(checkpoint.parent/'training_identity.json')
            if (content_sha != engine.digest(saved) or saved['identity_sha256'] != ownership['identity_sha256']
                    or ownership['binding']['arm'] != 'native_listwise'
                    or saved.get('comparison_policy', {}).get('arm') != 'native_listwise'):
                raise ValueError('Original native_listwise checkpoint content/ownership changed')
            result['source_checkpoint_state'] = {key:saved['state'][key]
                for key in ('epoch', 'phase', 'position', 'updates', 'attempts', 'overflows')}
            core = pressure_aware_provider(ComparisonData)
            provider_type = prepared_provider(core)
            kwargs = dict(source_samples=manifest['baseline']['source_samples'],
                raw_records=read(a.inventory)['raw_records'], config=config, bank=bank,
                workers=manifest['workers'], resident_bytes=int(manifest['resident_gib']*2**30),
                budget=budget, regions_dir=baseline/'shared/regions')
            # Only fields/static upper are borrowed. Deliberately ignore the
            # yielded local-graph subclass: all16 canonical targets build cold.
            # DEBUG destination is absent from sources, so no shared-parent
            # cold-field coordination lock can be formed in production.
            with preparation_reuse(core, a.cache_sources):
                provider = provider_type(root=a.output/'data', **kwargs)
                if (canonical(provider.examples('train')+provider.examples('val')) != manifest['samples']
                        or tuple(provider._example(i)['id'] for i in INDICES) != SOURCE_IDS):
                    raise AssertionError('Complete source list or original indices changed')
                print('DEBUG cold build: liver_46:0/1, epoch8, P+U49..55, physical2/full model', flush=True)
                before_rng = engine.digest(engine.capture_rng())
                batch = provider.batch(INDICES, 'native_listwise', 8, True, full=False)
                if engine.digest(engine.capture_rng()) != before_rng:
                    raise AssertionError('Input construction changed global model RNG')
                result['empty_context'] = empty_receipt(batch)
                cold = fingerprint(vars(batch))
                result['cold_provider'] = provider.report()
                if result['cold_provider']['local_builds'] != 16:
                    raise AssertionError('All16 canonical targets must be genuinely cold-built in DEBUG')
                result['workload'] = engine.batch_workload(batch)
                if a.geometry_sweep:
                    result['geometry_sweep'] = geometry_sweep(provider)
                    write_new(a.output/'geometry_sweep.json', result['geometry_sweep'])
                _evict_provider(provider)
                del batch, provider
                gc.collect()
                reopened = provider_type(root=a.output/'data', **kwargs)
                reopened._runtime()
                if not reopened.cached_batch_ready(INDICES, 'native_listwise', 8, training=True):
                    raise AssertionError('The exact epoch8 sample publications did not reopen')

                def denied(*args, **kwargs):
                    raise AssertionError('Completed layout replay attempted cold preparation')

                with patch.object(reopened, '_case', denied), patch.object(reopened, '_source', denied), \
                        patch.object(reopened, '_upper_context', denied), patch.object(reopened, '_layout_publish', denied):
                    batch = reopened.batch(INDICES, 'native_listwise', 8, True, full=False)
                result['cache_reopen'] = compare(cold, fingerprint(vars(batch)))
                if not result['cache_reopen']['exact_values_layout_metadata_equal']:
                    raise AssertionError('Cold/reopened exact real batch differs')
                result['reopened_provider'] = reopened.report()
                empty_receipt(batch)
                _evict_provider(reopened)
                del reopened, cold
                gc.collect()
            write_new(a.output/'input_receipt.json', result)
            torch.cuda.synchronize()
            started = time.perf_counter()
            batch = batch.pin_memory().to('cuda', non_blocking=True)
            torch.cuda.synchronize()
            result['host_to_device_seconds'] = time.perf_counter()-started
            result['updates'] = []
            for arm in a.objectives:
                print('DEBUG checkpoint restoration and one full CUDA update: '+arm, flush=True)
                report = resumed_update(batch, saved, config, budget, arm, a.output)
                write_new(a.output/(arm+'_update.json'), report)
                result['updates'].append(report)
            result['prior_nonempty_view_equality'] = dict(verified=False,
                reason='No unpatched view execution claimed; cold/reopened exact current inputs checked')
            result['source_checkpoint_unchanged'] = sha(checkpoint) == checkpoint_sha
            if not result['source_checkpoint_unchanged']:
                raise AssertionError('Source checkpoint changed during DEBUG verification')
            result['status'] = 'DEBUG_PASS'
            write_new(a.output/'result.json', result)
            print(json.dumps(dict(status=result['status'], output=str(a.output),
                objectives=a.objectives, source_checkpoint_unchanged=True), allow_nan=False), flush=True)
    except Exception as error:
        result.update(status='DEBUG_FAILED', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), source_checkpoint_unchanged=sha(checkpoint)==checkpoint_sha)
        write_new(a.output/'failure.json', result)
        raise
    return result


if __name__ == '__main__':
    verify(parse())
