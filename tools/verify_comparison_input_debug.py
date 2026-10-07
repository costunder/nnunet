"""Actual CT/CUDA DEBUG input equivalence and repeated-forward diagnostics.

No optimizer or training is run. Full fixture inputs and full129 inference use
unchanged original parameters and precision; no production quality claim.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def cache_inventory(root):
    return {p.relative_to(root).as_posix(): sha(p)
            for p in sorted(Path(root).rglob('*')) if p.is_file()}


def verify(arguments):
    from tools.local_cnn_device import select
    select(arguments.gpu)
    import torch
    from hiercp_v1x.comparison_experiment import FORMAT, FILES, digest, write_new
    from hiercp_v1x.comparison_data import ComparisonData
    from hiercp_v1x.comparison_training import run_arm, restore_rng, digest as neural_digest
    from hiercp_v1x.scope_probe_support import activate_original, state_digest
    from hiercp_v1x import bounded_scope
    from hiercp_v1x.host_memory import PressureBudget, pressure_aware_provider
    from hiercp_v1x.preparation_reuse import preparation_reuse

    reference = arguments.reference.resolve(strict=True)
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError('DEBUG output must be new; all old results are preserved')
    if output.is_relative_to(reference) or reference.is_relative_to(output):
        raise ValueError('DEBUG output must be disjoint from the reference')
    manifest = read(reference / 'experiment.json')
    if (manifest.get('format') != FORMAT or manifest.get('debug') is not True
            or manifest.get('epochs') != 2 or manifest.get('workers') != 4
            or manifest.get('explicit_batch_candidates') != [2]
            or manifest.get('sha256') != digest({k:v for k,v in manifest.items() if k != 'sha256'})):
        raise ValueError('The sealed actual-CT two-epoch, batch2, worker4 fixture is required')
    if set(manifest['helpers']) != set(FILES):
        raise ValueError('Frozen17-helper inventory differs')
    helpers_before = {name: sha(ROOT / name) for name in FILES}
    if helpers_before != manifest['helpers']:
        raise ValueError('Frozen input/model/training source bytes changed')
    if sha(arguments.inventory) != manifest['baseline']['inventory_sha256']:
        raise ValueError('Frozen128 candidate inventory changed')
    if sha(arguments.bank) != manifest['baseline']['bank_fixture_sha256']:
        raise ValueError('Actual-CT prototype bank fixture changed')
    source = Path(manifest['original']['source'])
    original = activate_original(source)
    scope = bounded_scope.install(10, expected_snapshot_root=source)
    if canonical(original) != manifest['original'] or canonical(scope) != manifest['scope']:
        raise ValueError('Original neural source or10mm graph scope changed')
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.tensor import configure_runtime, collect_runtime_resources
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one actual CUDA GPU required; no neural CPU fallback')
    torch.set_num_threads(manifest['workers'])
    capacity = torch.cuda.get_device_properties(0).total_memory
    cuda_bytes = int(manifest['cuda_gib'] * 2**30)
    if cuda_bytes >= capacity:
        raise ValueError('Preserved12GiB CUDA budget must leave actual driver headroom')
    torch.cuda.set_per_process_memory_fraction(cuda_bytes / capacity)
    config = copy.deepcopy(manifest['config'])
    configure_runtime(deterministic=bool(config['runtime'].get('deterministic', True)),
                      allow_tf32=bool(config['runtime'].get('allow_tf32', False)),
                      cudnn_benchmark=bool(config['runtime'].get('cudnn_benchmark', False)))
    source_cache = Path(manifest['prepared_data_root']).resolve(strict=True)
    original_files = [reference/'experiment.json', reference/'initial.pt', reference/'calibration.json',
                      reference/'native_fixed/checkpoint_latest.pt']
    originals_before = {str(path): sha(path) for path in original_files}
    source_cache_before = cache_inventory(source_cache)
    initial = torch.load(reference/'initial.pt', map_location='cpu', weights_only=False)
    if (initial['contract_sha256'] != manifest['sha256']
            or initial['model_sha256'] != state_digest(initial['model'])):
        raise ValueError('Preserved initial neural state changed')
    calibration = read(reference/'calibration.json')
    if calibration['physical_batch'] != 2 or calibration['contract_sha256'] != manifest['sha256']:
        raise ValueError('Actual physicalbatch2 calibration changed')
    fixture = torch.load(arguments.bank, map_location='cpu', weights_only=False, mmap=True)
    bank = fixture['prototype_bank']
    if bank.training_case_ids != ('liver_5', 'liver_6') or bank.fingerprint() != manifest['prototype_fingerprint']:
        raise ValueError('Preserved actual-CT population bank changed')
    del fixture
    output.mkdir(parents=True, exist_ok=False)
    events, event_lock = [], threading.Lock()
    def event(row):
        with event_lock:
            events.append(copy.deepcopy(row))
            with (output/'cache_events.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
    budget = PressureBudget(cuda_bytes, int(manifest['rss_gib'] * 2**30),
                            resident_bytes=int(manifest['resident_gib'] * 2**30), event_callback=event)
    resources = collect_runtime_resources('cuda', storage_path=output)
    print(json.dumps(dict(scope='actual_CT_CUDA_DEBUG_only', resources=resources,
        parameters=10434532, physical_batch=2, workers=4, optimizer_updates=0,
        training_candidates=8, validation_candidates=129, actual_train_cases=['liver_5','liver_6'],
        actual_validation_cases=['liver_31'], production_training_started=False,
        quality_verified=False), allow_nan=False), flush=True)

    from hiercp_v1x.comparison_inputs import local_mask_provider
    from hiercp_v1x.comparison_views import parallel_view_provider
    from hiercp_v1x.comparison_runtime import _prefetch, closing
    from hiercp_v1x.u_bridge_training import capture_rng
    from tools.current_gpu import current_device_selection

    def leaves(value, prefix='root'):
        if torch.is_tensor(value):
            yield prefix, value
        elif isinstance(value, dict):
            for key in sorted(value, key=repr):
                yield from leaves(value[key], prefix + '/' + repr(key))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                yield from leaves(item, prefix + '/' + str(index))
        elif hasattr(value, 'to_dict'):
            yield from leaves(value.to_dict(), prefix + '/graph')
            for field in ('_slice_dict', '_inc_dict', '_num_graphs'):
                if hasattr(value, field):
                    yield from leaves(getattr(value, field), prefix + '/' + field)
        elif value is None or isinstance(value, (str, int, float, bool)):
            yield prefix, value
        else:
            raise TypeError('Unrecognized actual batch payload ' + prefix + ': ' + str(type(value)))

    def payload(batch):
        return dict(vars(batch))

    def describe(value):
        if not torch.is_tensor(value):
            return dict(kind='metadata', value=value)
        cpu = value.detach().cpu().contiguous()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        return dict(kind='tensor', sha256=hashlib.sha256(raw).hexdigest(), shape=list(value.shape),
            dtype=str(value.dtype), stride=list(value.stride()), storage_offset=value.storage_offset(),
            layout=str(value.layout), device=str(value.device), pinned=value.is_pinned(),
            bytes=value.numel() * value.element_size())

    def compare_payload(left, right):
        first, second = dict(leaves(payload(left))), dict(leaves(payload(right)))
        keys = sorted(set(first) | set(second))
        differences = []
        for key in keys:
            if key not in first or key not in second:
                differences.append(dict(path=key, problem='missing leaf',
                                        baseline_present=key in first, new_present=key in second))
                continue
            a, b = describe(first[key]), describe(second[key])
            if a != b:
                row = dict(path=key, baseline=a, new=b)
                if (torch.is_tensor(first[key]) and torch.is_tensor(second[key])
                        and first[key].shape == second[key].shape):
                    row['values_equal'] = torch.equal(first[key], second[key])
                    row['max_absolute_difference'] = float(
                        (first[key].double() - second[key].double()).abs().max()) if first[key].numel() else 0.
                differences.append(row)
        return dict(leaf_count=len(keys), tensor_count=sum(torch.is_tensor(v) for v in first.values()),
                    baseline_digest=neural_digest({k:describe(v) for k,v in first.items()}),
                    new_digest=neural_digest({k:describe(v) for k,v in second.items()}),
                    exact_values_and_layout_equal=not differences, differences=differences)

    def score_difference(a, b):
        if len(a) != len(b):
            return dict(equal=False, problem='source count changed')
        differences = [float((torch.tensor(x, dtype=torch.float64)
                              - torch.tensor(y, dtype=torch.float64)).abs().max()) for x,y in zip(a,b)]
        return dict(equal=a == b, max_absolute_difference=max(differences), by_source=differences)

    started = time.perf_counter()
    reports = []
    traces = []
    net = HierarchicalPyGPlacementModel(**config['model']).cuda().eval()
    net.load_state_dict(initial['model'], strict=True)
    model_before = neural_digest(net.state_dict())
    hook_names = {'local_encoder.dense_encoder', 'local_encoder.blocks.0', 'local_encoder.blocks.1',
                  'local_encoder.blocks.2', 'patient_encoder.blocks.0', 'patient_encoder.blocks.1',
                  'prototype_encoder.blocks.0', 'prototype_encoder.blocks.1', 'score_head'}
    current_trace = []
    handles = []
    def hook(name):
        def observe(module, inputs, result):
            current_trace.append(dict(module=name, outputs={k:describe(v) for k,v in leaves(result)}))
        return observe
    observed_modules = {name:module for name,module in net.named_modules() if name in hook_names}
    if len(observed_modules) != len(hook_names):
        raise ValueError('Original module inventory changed before diagnostic hooks')

    try:
        with preparation_reuse(pressure_aware_provider(ComparisonData), [source_cache], event) as Base:
            New = parallel_view_provider(local_mask_provider(Base))
            kwargs = dict(source_samples=manifest['baseline']['source_samples'],
                raw_records=read(arguments.inventory)['raw_records'], config=config, bank=bank,
                workers=manifest['workers'], resident_bytes=int(manifest['resident_gib'] * 2**30),
                budget=budget, regions_dir=source_cache/'regions')
            old_provider = Base(root=output/'baseline_data', **kwargs)
            new_provider = New(root=output/'new_data', **kwargs)
            for partition, full, epoch in (('train', False, 1), ('val', True, config['training']['fixed_validation_epoch'])):
                ids = [r['index'] for r in old_provider.examples(partition)]
                if canonical(old_provider.examples(partition)) != canonical(new_provider.examples(partition)):
                    raise ValueError('Provider source inventories differ')
                batches = [ids[i:i+2] for i in range(0, len(ids), 2)]
                with closing(_prefetch(new_provider, batches, 'native_fixed', epoch,
                                      training=not full, full=full, pin_memory=False)) as staged:
                    for sample_ids in batches:
                        restore_rng(initial['rng'])
                        before_rng = neural_digest(capture_rng())
                        begin = time.perf_counter()
                        baseline_batch = old_provider.batch(sample_ids, 'native_fixed', epoch, not full, full=full)
                        original_seconds = time.perf_counter() - begin
                        baseline_rng_preserved = neural_digest(capture_rng()) == before_rng
                        new_batch = next(staged)
                        new_rng_preserved = neural_digest(capture_rng()) == before_rng
                        comparison = compare_payload(baseline_batch, new_batch)
                        comparison.update(partition=partition, sample_indices=sample_ids,
                            candidate_count=129 if full else 8, baseline_seconds=original_seconds,
                            baseline_rng_preserved=baseline_rng_preserved, new_rng_preserved=new_rng_preserved)
                        reports.append(comparison)
                        print(json.dumps(dict(partition=partition, sample_indices=sample_ids,
                            input_equal=comparison['exact_values_and_layout_equal'],
                            differing_leaves=len(comparison['differences'])), allow_nan=False), flush=True)
                        if full:
                            runs = []
                            for name, batch in (('baseline_same_payload', baseline_batch), ('new_same_payload', new_batch)):
                                for repetition in range(3):
                                    restore_rng(initial['rng'])
                                    current_trace.clear()
                                    torch.cuda.synchronize()
                                    begin = time.perf_counter()
                                    with torch.no_grad(), torch.autocast('cuda', enabled=config['training']['amp']):
                                        scores = net.score_inference_chunked(
                                            batch, local_chunk_size=manifest['validation_local_chunk'])
                                    torch.cuda.synchronize()
                                    runs.append(dict(provider=name, repetition=repetition,
                                        scores=[score.detach().cpu().double().tolist() for score in scores],
                                        seconds=time.perf_counter()-begin, trace=copy.deepcopy(current_trace)))
                                    print(f'{name} full129 forward {repetition+1}/3 completed', flush=True)
                            comparisons = []
                            for index in range(1, len(runs)):
                                baseline = runs[0]
                                row = dict(left=0, right=index, **score_difference(baseline['scores'], runs[index]['scores']))
                                paired = zip(baseline['trace'], runs[index]['trace'])
                                row['first_different_module_call'] = next(
                                    (dict(call=call, baseline=a, new=b) for call,(a,b) in enumerate(paired) if a != b), None)
                                row['trace_call_counts'] = [len(baseline['trace']), len(runs[index]['trace'])]
                                comparisons.append(row)
                            traced_runs = []
                            if any(not row['equal'] for row in comparisons):
                                handles.extend(module.register_forward_hook(hook(name))
                                               for name,module in observed_modules.items())
                                for name, batch in (('baseline', baseline_batch), ('baseline_repeat', baseline_batch), ('new', new_batch)):
                                    restore_rng(initial['rng'])
                                    current_trace.clear()
                                    with torch.no_grad(), torch.autocast('cuda', enabled=config['training']['amp']):
                                        diagnostic = net.score_inference_chunked(batch,
                                            local_chunk_size=manifest['validation_local_chunk'])
                                    traced_runs.append(dict(provider=name,
                                        scores=[score.detach().cpu().double().tolist() for score in diagnostic],
                                        trace=copy.deepcopy(current_trace)))
                                    print(f'{name} additional traced full129 forward completed', flush=True)
                                for handle in handles:
                                    handle.remove()
                                handles.clear()
                            trace_differences = []
                            for index in range(1, len(traced_runs)):
                                paired = zip(traced_runs[0]['trace'], traced_runs[index]['trace'])
                                trace_differences.append(dict(left=0, right=index,
                                    **score_difference(traced_runs[0]['scores'], traced_runs[index]['scores']),
                                    first_different_module_call=next((dict(call=call, baseline=a, new=b)
                                        for call,(a,b) in enumerate(paired) if a != b), None)))
                            traces.append(dict(sample_indices=sample_ids, runs=runs, comparisons_to_first_baseline=comparisons,
                                extra_traced_runs=traced_runs, extra_traced_comparisons=trace_differences,
                                baseline_repeat_equal=all(score_difference(runs[0]['scores'], r['scores'])['equal'] for r in runs[1:3]),
                                new_repeat_equal=all(score_difference(runs[3]['scores'], r['scores'])['equal'] for r in runs[4:6])))
                        del baseline_batch, new_batch
                        budget.check()
            provider_stats = dict(baseline=old_provider.report(), new=new_provider.report())
    finally:
        for handle in handles:
            handle.remove()
    preserved = dict(model_unchanged=neural_digest(net.state_dict()) == model_before,
        references_unchanged={str(path):sha(path) for path in original_files} == originals_before,
        source_cache_unchanged=cache_inventory(source_cache) == source_cache_before,
        frozen_helpers_unchanged={name:sha(ROOT/name) for name in FILES} == helpers_before)
    result = dict(scope='actual_CT_CUDA_DEBUG_input_and_repeated_forward_only', debug=True,
        optimizer_updates=0, quality_verified=False, full_training=False, full_production_evaluation=False,
        parameters=10434532, physical_train_batch=2, workers=4,
        actual_fixture_train_sources=len(old_provider.examples('train')),
        actual_fixture_validation_sources=len(old_provider.examples('val')),
        inputs=reports, forward_diagnostics=traces, preserved=preserved,
        providers=provider_stats, resources=resources,
        cuda_runtime=dict(cudnn_deterministic=torch.backends.cudnn.deterministic,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark, amp=config['training']['amp'],
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
        diagnostic_hooks='First baseline/new repeats have no hooks. Differences trigger three additional full129 forwards with full-value hashes at dense encoder, L0/L1/L2 blocks and score head; these extra forwards synchronize in DEBUG only.',
        exact_inputs=all(r['exact_values_and_layout_equal'] and r['baseline_rng_preserved'] and r['new_rng_preserved'] for r in reports),
        wall_seconds=time.perf_counter()-started)
    write_new(output/'report.json', canonical(result))
    print(json.dumps(dict(report=str(output/'report.json'), exact_inputs=result['exact_inputs'], preserved=preserved,
        repeated_identical_payloads_stable=[dict(baseline=r['baseline_repeat_equal'], new=r['new_repeat_equal']) for r in traces]),
        allow_nan=False), flush=True)
    if not result['exact_inputs'] or not all(preserved.values()):
        raise AssertionError('Exact input/preservation diagnostic differs; inspect report.json')
    return result


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--reference', type=Path, default=ROOT/'work/v19_checked_DEBUG')
    p.add_argument('--inventory', type=Path, default=ROOT/'work/v22_cumulative_u16_DEBUG_20261006/inventory.json')
    p.add_argument('--bank', type=Path, default=ROOT/'work/v1x_real_CT_DEBUG_20261003_prepare/fixture.pt')
    p.add_argument('--output', type=Path, required=True)
    return p.parse_args(argv)


if __name__ == '__main__':
    from tools.current_gpu import current_device_selection
    with current_device_selection():
        verify(parse())
