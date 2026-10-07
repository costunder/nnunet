"""Actual-CT CPU input timing: existing parallel provider versus exact caches.

No model/checkpoint is loaded, no CUDA forward or optimizer is run. Both branches
use the same full sealed train8 and validation129 source problems, view epochs,
workers, resident budget, original operators and borrowed completed geometry.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verified', type=Path, default=ROOT/'work/runs/v1.9/preparation/debug-20261008')
    a = p.parse_args()
    from tools.verify_comparison_cache_debug import parse, read, sha, cache_inventory, canonical
    from tools.verify_comparison_preparation_debug import fingerprint, compare
    from hiercp_v1x.comparison_experiment import FILES, FORMAT, digest
    from hiercp_v1x.scope_probe_support import activate_original
    from hiercp_v1x import bounded_scope
    from hiercp_v1x.comparison_data import ComparisonData
    from hiercp_v1x.comparison_inputs import local_mask_provider
    from hiercp_v1x.comparison_views import parallel_view_provider
    from hiercp_v1x.comparison_preparation import prepared_provider
    from hiercp_v1x.comparison_data_timing import timed_provider
    from hiercp_v1x.host_memory import PressureBudget, pressure_aware_provider, _evict_provider
    from hiercp_v1x.preparation_reuse import preparation_reuse
    from hiercp_v1x.u_bridge_training import capture_rng, restore_rng, digest as rng_digest
    import numpy as np
    import psutil
    import torch

    output = a.output.resolve()
    if output.exists():
        raise FileExistsError('Timing output must be new; all completed evidence is preserved')
    args = parse(['--gpu', '0', '--output', str(output)])
    manifest = read(args.reference/'experiment.json')
    if (manifest['format'] != FORMAT or manifest['debug'] is not True
            or manifest['workers'] != 4 or manifest['explicit_batch_candidates'] != [2]
            or manifest['sha256'] != digest({k:v for k,v in manifest.items() if k != 'sha256'})
            or {name:sha(ROOT/name) for name in FILES} != manifest['helpers']
            or sha(args.inventory) != manifest['baseline']['inventory_sha256']
            or sha(args.bank) != manifest['baseline']['bank_fixture_sha256']):
        raise ValueError('Sealed real-CT/full-input fixture identity differs')
    verified = a.verified.resolve(strict=True)
    receipt = read(verified/'verification.json')
    if (receipt.get('exact_inputs') is not True or receipt.get('optimizer_updates') != 2
            or sha(verified/'inputs.json') != receipt['input_report']['sha256']):
        raise ValueError('Completed actual-CT/CUDA exact input evidence required')
    source = Path(manifest['original']['source'])
    original = activate_original(source)
    scope = bounded_scope.install(10, expected_snapshot_root=source)
    if canonical(original) != manifest['original'] or canonical(scope) != manifest['scope']:
        raise ValueError('Original source/10mm scope differs')
    frozen_before = {name:sha(ROOT/name) for name in FILES}
    references = [args.reference/'experiment.json', args.reference/'initial.pt',
                  args.reference/'native_fixed/checkpoint_latest.pt', verified/'verification.json',
                  verified/'report.json', verified/'inputs.json']
    preserved = {str(path):sha(path) for path in references}
    source_cache = Path(manifest['prepared_data_root'])
    source_inventory = cache_inventory(source_cache)
    bank = torch.load(args.bank, map_location='cpu', weights_only=False, mmap=True)['prototype_bank']
    if bank.fingerprint() != manifest['prototype_fingerprint']:
        raise ValueError('Original prototype bank identity differs')
    output.mkdir(parents=True)
    new_root = output/'prepared'
    new_root.mkdir()
    copied = {}
    for name in ('source_prepared', 'compact_upper_v1'):
        origin = verified/'data'/name
        copied[name] = cache_inventory(origin)
        shutil.copytree(origin, new_root/name)
        if cache_inventory(new_root/name) != copied[name]:
            raise AssertionError('Verified execution-cache copy bytes differ')
    torch.set_num_threads(manifest['workers'])
    torch.manual_seed(manifest['config']['seed'])
    np.random.seed(manifest['config']['seed'])
    random.seed(manifest['config']['seed'])
    # This process performs CPU preparation only; no CUDA allocator is touched.
    budget = PressureBudget(int(manifest['cuda_gib']*2**30), int(manifest['rss_gib']*2**30),
        resident_bytes=int(manifest['resident_gib']*2**30), cuda_reader=lambda: 0)
    resources = dict(cpu_logical=psutil.cpu_count(), available_RAM_bytes=psutil.virtual_memory().available,
        physical_source_batch=2, workers=4, resident_bytes=budget.resident_bytes,
        rss_budget_bytes=budget.rss_bytes, CUDA_model_loaded=False, GPU_forward=False)
    print(json.dumps(dict(scope='actual_CT_CPU_input_performance_DEBUG_only', **resources)), flush=True)
    rows = []
    started = time.perf_counter()
    with preparation_reuse(pressure_aware_provider(ComparisonData), [source_cache]) as Base:
        Current = timed_provider(parallel_view_provider(local_mask_provider(Base)), output/'current_timing.jsonl')
        Prepared = prepared_provider(Base, output/'prepared_timing.jsonl')
        kwargs = dict(source_samples=manifest['baseline']['source_samples'],
            raw_records=read(args.inventory)['raw_records'], config=copy.deepcopy(manifest['config']),
            bank=bank, workers=4, resident_bytes=budget.resident_bytes, budget=budget,
            regions_dir=source_cache/'regions')
        providers = dict(current=Current(root=output/'current', **kwargs),
                         prepared=Prepared(root=new_root, **kwargs))
        rng = capture_rng()
        schedules = [('cold_resident', 'train', 2, ('current','prepared'))]
        schedules += [('warm', 'train', 2, order) for order in
                      (('prepared','current'), ('current','prepared'), ('prepared','current'))]
        schedules += [('evicted', 'train', 2, ('current','prepared')),
                      ('validation', 'val', manifest['config']['training']['fixed_validation_epoch'],
                       ('prepared','current'))]
        for sequence, (mode, partition, epoch, order) in enumerate(schedules):
            if mode == 'evicted':
                for provider in providers.values():
                    _evict_provider(provider)
                gc.collect()
            pair = {}
            for name in order:
                provider = providers[name]
                indices = [r['index'] for r in provider.examples(partition)]
                restore_rng(rng)
                before = time.perf_counter()
                batch = provider.batch(indices, 'native_fixed', epoch, partition == 'train', full=partition == 'val')
                seconds = time.perf_counter()-before
                pair[name] = fingerprint(vars(batch))
                if rng_digest(capture_rng()) != rng_digest(rng):
                    raise AssertionError('Global RNG changed during preparation')
                del batch
                rows.append(dict(sequence=sequence, mode=mode, provider=name, view_epoch=epoch,
                    source_indices=indices, candidates=8 if partition == 'train' else 129,
                    input_seconds=seconds, rss_bytes=psutil.Process().memory_info().rss))
                print(f'DEBUG {mode} {name} {partition} | {seconds:.3f}s', flush=True)
            exact = compare(pair['current'], pair['prepared'])
            if not exact['exact_values_layout_metadata_equal']:
                (output/'failure.json').write_text(json.dumps(exact, indent=2), encoding='utf8')
                raise AssertionError('Existing release/prepared input differences')
            rows[-1]['pair_exact'] = exact
        provider_reports = {name:provider.report() for name,provider in providers.items()}
        for provider in providers.values():
            _evict_provider(provider)
    preserved_ok = ({str(path):sha(path) for path in references} == preserved
        and cache_inventory(source_cache) == source_inventory
        and {name:sha(ROOT/name) for name in FILES} == frozen_before
        and all(cache_inventory(verified/'data'/name) == inventory for name,inventory in copied.items()))
    if not preserved_ok:
        raise AssertionError('Reference/source/cache evidence changed')
    warm = {name:statistics.median(r['input_seconds'] for r in rows
        if r['mode'] == 'warm' and r['provider'] == name) for name in providers}
    result = dict(scope='actual_CT_CPU_input_performance_DEBUG_only', debug=True,
        current_control='Existing parallel_view_provider(local_mask_provider(original))',
        prepared='same + exact source preparation and compact original upper caches',
        exact_inputs=True, current_vs_prepared_warm_train_median_seconds=warm,
        warm_input_time_ratio=warm['current']/warm['prepared'], measurements=rows,
        provider_reports=provider_reports, resources=resources, references_preserved=True,
        wall_seconds=time.perf_counter()-started, quality_verified=False,
        production_training_started=False, GPU_forward=False, optimizer_updates=0,
        limitation='Same three real CT cases on local Windows SSD. Not a server/NFS throughput '
            'claim. Cold resident starts from completed immutable preprocessing on disk; '
            'prepared branch already has its verified exact source/upper disk publications.')
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(dict(report=str(output/'report.json'), warm_medians=warm,
        ratio=result['warm_input_time_ratio'], exact_inputs=True, references_preserved=True)), flush=True)


if __name__ == '__main__':
    main()
