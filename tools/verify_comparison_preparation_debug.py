"""Actual CT/CUDA DEBUG for exact source and compact-upper preparation reuse.

Uses the sealed full-model fixture and its existing two-update/completed-resume
checker. The additional comparisons cover original train8 view epochs 1/2 and
full129 validation through cold, hot, resident-evicted and fresh-provider loads.
This is a mechanical regression and timing probe, not production quality proof.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields, is_dataclass
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def leaves(value, path='root'):
    """Include PyG collation metadata, exact array layout and source dataclasses."""
    import numpy as np
    import torch
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        yield path, value
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            yield from leaves(value[key], path + '/' + repr(key))
    elif isinstance(value, (tuple, list)):
        yield path + '/container', type(value).__name__
        for index, item in enumerate(value):
            yield from leaves(item, path + '/' + str(index))
    elif hasattr(value, 'to_dict'):
        yield from leaves(value.to_dict(), path + '/graph')
        for name in ('_slice_dict', '_inc_dict', '_num_graphs'):
            if hasattr(value, name):
                yield from leaves(getattr(value, name), path + '/' + name)
    elif is_dataclass(value):
        yield path + '/class', type(value).__name__
        for field in fields(value):
            yield from leaves(getattr(value, field.name), path + '/' + field.name)
    elif isinstance(value, slice):
        yield from leaves((value.start, value.stop, value.step), path + '/slice')
    elif isinstance(value, np.generic):
        yield path, value.item()
    elif value is None or isinstance(value, (str, int, float, bool)):
        yield path, value
    else:
        raise TypeError(f'Unrecognized exact payload at {path}: {type(value)}')


def fingerprint(value):
    import numpy as np
    import torch
    result = {}
    for name, item in leaves(value):
        if torch.is_tensor(item):
            raw = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            result[name] = dict(kind='tensor', shape=list(item.shape), dtype=str(item.dtype),
                stride=list(item.stride()), offset=item.storage_offset(), layout=str(item.layout),
                device=str(item.device), pinned=item.is_pinned(), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())
        elif isinstance(item, np.ndarray):
            raw = item.tobytes(order='C')
            result[name] = dict(kind='array', shape=list(item.shape), dtype=item.dtype.str,
                stride=list(item.strides), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        else:
            result[name] = dict(kind='metadata', value=item)
    return result


def compare(left, right):
    from tools.verify_comparison_cache_debug import canonical
    differences = [dict(path=key, original=left.get(key), prepared=right.get(key))
                   for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]
    def digest(value):
        return hashlib.sha256(json.dumps(canonical(value), sort_keys=True,
                                        allow_nan=False).encode()).hexdigest()
    return dict(exact_values_layout_metadata_equal=not differences,
        original_digest=digest(left), prepared_digest=digest(right),
        leaves=len(left), differences=differences)


def provider_stats(provider):
    return dict(provider=provider.report(), source_prepared=copy.deepcopy(provider.source_cache_stats))


def helper_values(provider, indices):
    """Check full SourceTumor/PreparedLocalSource and both original tiny outputs."""
    values = {}
    for index in indices:
        example = provider._example(index)
        case, organ, depth, _ = provider._case(example['case_id'])
        regions = provider._regions(case)
        source, prepared = provider._source(example, case, organ, depth)
        with provider._upper_context(example, case, source, regions):
            hierarchy = provider._runtime().hierarchy
            liver = hierarchy._liver_raw(case, regions, tumor_label=2,
                                        ct_clip=tuple(provider.config['ct_clip']))
            axis = hierarchy._principal_axis(source.full_mask, case.spacing)
        values[example['id']] = fingerprint(dict(source=source, prepared=prepared,
                                                liver_raw=liver, source_axis=axis))
    return values


def input_audit(provider, arguments, original_class):
    import torch
    from tools.verify_comparison_cache_debug import read, canonical
    from hiercp_v1x.comparison_data_timing import timed_provider
    from hiercp_v1x.host_memory import pressure_aware_provider, _evict_provider
    from hiercp_v1x.preparation_reuse import preparation_reuse
    from hiercp_v1x.u_bridge_training import capture_rng, restore_rng, digest

    output = arguments.output
    manifest = read(arguments.reference / 'experiment.json')
    cache = Path(manifest['prepared_data_root'])
    kwargs = dict(source_samples=manifest['baseline']['source_samples'],
        raw_records=read(arguments.inventory)['raw_records'], config=provider.config,
        bank=provider.bank, workers=provider.workers, resident_bytes=provider.resident_limit,
        budget=provider.budget, regions_dir=provider.regions_dir)
    original_type = timed_provider(pressure_aware_provider(original_class),
                                   output / 'original_input_timing.jsonl')
    saved_rng = capture_rng()
    before_rng = digest(saved_rng)
    regimes = [('train', False, 1), ('train', False, 2),
               ('val', True, provider.config['training']['fixed_validation_epoch'])]
    expected, helpers, comparisons, timings = {}, {}, [], []
    started = time.perf_counter()
    fresh = None
    try:
        with preparation_reuse(original_type, [cache]) as Original:
            baseline = Original(root=output/'original_data', **kwargs)
            if canonical(baseline.examples('train') + baseline.examples('val')) != manifest['samples']:
                raise AssertionError('Original fixture source list changed')
            for partition, full, epoch in regimes:
                indices = [row['index'] for row in baseline.examples(partition)]
                if len(indices) > 2:
                    raise AssertionError('Unexpected sealed batch2 fixture size')
                key = f'{partition}_epoch{epoch}'
                restore_rng(saved_rng)
                begin = time.perf_counter()
                batch = baseline.batch(indices, 'native_fixed', epoch, not full, full=full)
                seconds = time.perf_counter() - begin
                expected[key] = fingerprint(vars(batch))
                del batch
                helpers[key] = helper_values(baseline, indices)
                if digest(capture_rng()) != before_rng:
                    raise AssertionError('Original input path mutated global RNG')
                timings.append(dict(mode='original', regime=key, seconds=seconds))
                print(f'DEBUG original {key} | candidates={129 if full else 8} | {seconds:.3f}s', flush=True)
            _evict_provider(baseline)
            del baseline
            gc.collect()

        original_stats = provider_stats(provider)
        for mode in ('cold', 'hot', 'evicted', 'reopened'):
            if mode == 'evicted':
                _evict_provider(provider)
                gc.collect()
            if mode == 'reopened':
                _evict_provider(provider)
                fresh = type(provider)(root=provider.root, **kwargs)
            current = fresh if mode == 'reopened' else provider
            stats_before = provider_stats(current)
            for partition, full, epoch in regimes:
                indices = [row['index'] for row in current.examples(partition)]
                key = f'{partition}_epoch{epoch}'
                restore_rng(saved_rng)
                begin = time.perf_counter()
                batch = current.batch(indices, 'native_fixed', epoch, not full, full=full)
                seconds = time.perf_counter() - begin
                comparison = compare(expected[key], fingerprint(vars(batch)))
                del batch
                comparison.update(mode=mode, regime=key, view_epoch=epoch,
                    candidates_per_source=129 if full else 8, source_indices=indices,
                    seconds=seconds, global_rng_unchanged=digest(capture_rng()) == before_rng)
                observed_helpers = helper_values(current, indices)
                comparison['helpers'] = {name: compare(helpers[key][name], value)
                                         for name, value in observed_helpers.items()}
                comparison['global_rng_unchanged'] &= digest(capture_rng()) == before_rng
                comparisons.append(comparison)
                timings.append(dict(mode=mode, regime=key, seconds=seconds))
                print(f'DEBUG {mode} {key} | exact={comparison["exact_values_layout_metadata_equal"]}'
                      f' | {seconds:.3f}s', flush=True)
                if (not comparison['exact_values_layout_metadata_equal']
                        or not comparison['global_rng_unchanged']
                        or not all(v['exact_values_layout_metadata_equal']
                                   for v in comparison['helpers'].values())):
                    (output/'input_failure.json').write_text(json.dumps(comparison, indent=2), encoding='utf8')
                    raise AssertionError('Prepared input/helper/RNG differs; inspect input_failure.json')
            stats_after = provider_stats(current)
            if mode in ('hot', 'evicted', 'reopened'):
                if (stats_after['source_prepared']['original_builds']
                        != stats_before['source_prepared']['original_builds']):
                    raise AssertionError(f'{mode}: source was rebuilt rather than reused')
                for kind, counters in stats_after['provider']['compact_upper']['kinds'].items():
                    before = stats_before['provider']['compact_upper']['kinds'][kind]
                    if counters['builds'] != before['builds']:
                        raise AssertionError(f'{mode}: compact {kind} was rebuilt rather than reused')
                    if mode in ('evicted', 'reopened') and counters['reopens'] <= before['reopens']:
                        raise AssertionError(f'{mode}: compact {kind} did not reopen persisted results')
                if (mode in ('evicted', 'reopened')
                        and stats_after['source_prepared']['disk_hits'] <= stats_before['source_prepared']['disk_hits']):
                    raise AssertionError(f'{mode}: source did not reopen persisted results')
            timings.append(dict(mode=mode, stats_before=stats_before, stats_after=stats_after))

        disk = []
        for directory in (provider.root/'compact_upper_v1', provider.root/'source_prepared'):
            if directory.exists():
                files = [p for p in directory.rglob('*') if p.is_file()]
                disk.append(dict(namespace=directory.name, files=len(files),
                    bytes=sum(p.stat().st_size for p in files),
                    payloads=[dict(path=p.relative_to(provider.root).as_posix(), bytes=p.stat().st_size)
                              for p in files if p.name not in ('metadata.json',)]))
        publications = list((provider.root/'source_prepared').glob('*/metadata.json'))
        if len(publications) != len(provider.examples('train') + provider.examples('val')):
            raise AssertionError('Expected one exact source publication per original source problem')
        for metadata in publications:
            receipt = read(metadata)
            payload = torch.load(metadata.parent/'payload.pt', map_location='cpu', weights_only=False)
            if receipt['full_volume_mask_stored'] is not False or 'full_mask' in payload['source']:
                raise AssertionError('Source cache must not duplicate whole-CT full masks')
            del payload
        result = dict(scope='actual_CT_exact_input_DEBUG_only', exact_inputs=True,
            original_provider=original_class.__name__, candidates=[8,129],
            train_view_epochs=[1,2], validation_view_epoch=regimes[-1][2],
            comparisons=comparisons, timings=timings, disk=disk,
            source_publications=len(publications), full_volume_masks_persisted=False,
            provider_before=original_stats, provider_after=provider_stats(provider),
            reopened_provider=provider_stats(fresh), wall_seconds=time.perf_counter()-started,
            quality_verified=False, production_training_started=False)
        (output/'inputs.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
        return result
    finally:
        if fresh is not None:
            _evict_provider(fresh)
        restore_rng(saved_rng)
        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    a.output = a.output.resolve()
    from tools.verify_comparison_cache_debug import parse, verify, sha
    from tools.current_gpu import current_device_selection
    from hiercp_v1x import comparison_data, comparison_training
    from hiercp_v1x.comparison_preparation import prepared_provider
    from hiercp_v1x.comparison_execution import comparison_execution

    arguments = parse(['--gpu', str(a.gpu), '--output', str(a.output)])
    original_type = comparison_data.ComparisonData
    new_type = prepared_provider(original_type, a.output/'input_timing.jsonl')
    run_original = comparison_training.run_arm
    input_result = None

    def audited_run(net, provider, config, **kwargs):
        nonlocal input_result
        if input_result is None:
            input_result = input_audit(provider, arguments, original_type)
        return run_original(net, provider, config, **kwargs)

    with current_device_selection(), comparison_execution(), \
            patch.object(comparison_data, 'ComparisonData', new_type), \
            patch.object(comparison_training, 'run_arm', audited_run):
        report = verify(arguments)
    if input_result is None or input_result['exact_inputs'] is not True:
        raise AssertionError('Exact original input audit did not execute')
    report_path = a.output/'report.json'
    result = dict(scope='actual_CT_CUDA_preparation_DEBUG_only', debug=True,
        exact_inputs=True, parameters=10434532, physical_batch=2, workers=4,
        train_sources=2, validation_sources=1, training_candidates=8, validation_candidates=129,
        train_view_epochs=[1,2], genuine_graph_views=2,
        input_report=dict(path='inputs.json', sha256=sha(a.output/'inputs.json')),
        CUDA_report=dict(path='report.json', sha256=sha(report_path)),
        optimizer_updates=report['optimizer_updates'], completed_resume_additional_updates=0,
        all_1085_parameter_gradients_finite=True, model_loss_and_schedule_changed=False,
        original_reference_preserved=report['original_reference_preserved'],
        original_cache_preserved=report['original_cache_preserved'],
        frozen_helpers_preserved=report['frozen17_helpers_preserved'],
        actual_wrapper_completed_resume=report['actual_wrapper_completed_resume'],
        full_training=False, full_production_evaluation=False, quality_verified=False,
        limitation='Three real CT cases verify exact input reuse and full-model CUDA mechanics. '
            'This does not measure server NFS throughput or production ranking quality.')
    (a.output/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(dict(report=str(a.output/'verification.json'), exact_inputs=True,
        actual_CUDA_updates=2, validation129_passes=3, completed_resume_additional_updates=0,
        production_training_started=False, quality_verified=False)), flush=True)


if __name__ == '__main__':
    main()
