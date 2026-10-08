"""Actual-CT CPU-only DEBUG: first use, hot repeat, rotation and provider reopen.

Uses unchanged production provider and immutable source/cache publications.
No model, forward, optimizer, CUDA work, final training or evaluation is run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import cProfile
import io
import json
from pathlib import Path
import pstats
import shutil
import subprocess
import sys
import threading
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def stats_report(profiles, prefix):
    stream = io.StringIO()
    stats = pstats.Stats(*profiles, stream=stream).sort_stats('tottime')
    stats.dump_stats(str(prefix)+'.prof')
    stats.print_stats(70)
    with Path(str(prefix)+'.txt').open('x', encoding='utf8') as target:
        target.write(stream.getvalue())
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, callers) in stats.stats.items():
        rows.append(dict(file=filename, line=line, function=name, primitive_calls=primitive,
            calls=calls, tottime_seconds=own, cumtime_seconds=cumulative))
    return sorted(rows, key=lambda row: row['tottime_seconds'], reverse=True)[:70]


@contextmanager
def worker_profiles():
    original = ThreadPoolExecutor.submit
    records, lock = [], threading.Lock()

    def submit(executor, function, /, *args, **kwargs):
        def measured():
            profile = cProfile.Profile()
            began, cpu = time.perf_counter(), time.thread_time()
            try:
                return profile.runcall(function, *args, **kwargs)
            finally:
                row = dict(thread=threading.current_thread().name,
                    wall_seconds=time.perf_counter()-began,
                    thread_cpu_seconds=time.thread_time()-cpu, profile=profile)
                with lock:
                    records.append(row)
        return original(executor, measured)

    with patch.object(ThreadPoolExecutor, 'submit', submit):
        yield records


def run(output):
    import psutil
    import torch
    from tools.verify_comparison_cache_debug import read, sha, canonical
    from tools.verify_comparison_gpu_debug import tree_stats
    from tools.verify_comparison_preparation_debug import fingerprint, compare
    from hiercp_v1x.comparison_experiment import FORMAT, FILES, digest
    from hiercp_v1x.scope_probe_support import activate_original
    from hiercp_v1x import bounded_scope
    from hiercp_v1x.comparison_data import ComparisonData
    from hiercp_v1x.comparison_preparation import prepared_provider
    from hiercp_v1x.host_memory import PressureBudget, pressure_aware_provider
    from hiercp_v1x.preparation_reuse import preparation_reuse

    reference = ROOT/'work/v19_checked_DEBUG'
    inventory = ROOT/'work/v22_cumulative_u16_DEBUG_20261006/inventory.json'
    bank_path = ROOT/'work/v1x_real_CT_DEBUG_20261003_prepare/fixture.pt'
    verified = ROOT/'work/runs/v1.9/source-scope/debug-20261008/verified'
    manifest = read(reference/'experiment.json')
    source_root = Path(manifest['prepared_data_root']).resolve(strict=True)
    completed = verified/'reopened_data'
    roots = [completed, source_root]
    output = output.resolve()
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in roots):
        raise FileExistsError('New disjoint DEBUG output required')
    helpers = {name: sha(ROOT/name) for name in FILES}
    if (manifest['format'] != FORMAT or manifest['debug'] is not True
            or manifest['workers'] != 4 or manifest['explicit_batch_candidates'] != [2]
            or manifest['sha256'] != digest({k:v for k,v in manifest.items() if k!='sha256'})
            or helpers != manifest['helpers'] or sha(inventory) != manifest['baseline']['inventory_sha256']
            or sha(bank_path) != manifest['baseline']['bank_fixture_sha256']):
        raise ValueError('Sealed actual CT physical2/worker4 input fixture differs')
    original = Path(manifest['original']['source'])
    if (canonical(activate_original(original)) != manifest['original']
            or canonical(bounded_scope.install(10, expected_snapshot_root=original)) != manifest['scope']):
        raise ValueError('Preserved original neural source or m10 scope differs')
    torch.set_num_threads(4)
    fixture = torch.load(bank_path, map_location='cpu', weights_only=False, mmap=True)
    bank = fixture['prototype_bank']
    if bank.fingerprint() != manifest['prototype_fingerprint']:
        raise ValueError('Real CT bank identity differs')
    del fixture
    config = copy.deepcopy(manifest['config'])
    budget = PressureBudget(int(manifest['cuda_gib']*2**30), int(manifest['rss_gib']*2**30),
        resident_bytes=int(manifest['resident_gib']*2**30))
    references = [reference/'experiment.json', inventory, bank_path,
        *[Path(row['path']) for row in manifest['baseline']['source_samples']]]
    ref_before = {str(path): sha(path) for path in references}
    cache_before = {str(path): tree_stats(path) for path in roots}
    output.mkdir(parents=True, exist_ok=False)
    copied_source = completed/'source_prepared'
    shutil.copytree(copied_source, output/'data/source_prepared')
    source_copies = {str(p.relative_to(copied_source)): sha(p) for p in copied_source.rglob('*') if p.is_file()}
    if source_copies != {str(p.relative_to(output/'data/source_prepared')): sha(p)
            for p in (output/'data/source_prepared').rglob('*') if p.is_file()}:
        raise AssertionError('Compact source byte copies differ')
    request = dict(debug=True, actual_CT=True, CPU_input_only=True, full_training=False,
        full_evaluation=False, GPU_training=False, optimizer_updates=0, arm='native_listwise',
        view_epoch=4, profiled_view_epochs=[4,4,5,4], candidates_per_source=8, physical_source_batch=2, workers=4,
        torch_threads=4, candidate_keys=['P']+[f'U:{i}' for i in range(21,28)],
        original_model_config=config['model'], model_instantiated=False,
        original_config=config, fixture_sources=manifest['baseline']['source_samples'],
        sources_used=['liver_5:0','liver_6:0'], debug_source_fraction=2/187,
        reuse_roots=list(map(str, roots)), copied_source_publications=source_copies,
        preserved_reference_hashes=ref_before, harness_sha256=sha(Path(__file__)),
        memory=psutil.virtual_memory()._asdict(), logical_cpu_count=psutil.cpu_count(),
        physical_cpu_count=psutil.cpu_count(logical=False),
        hardware_gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used',
            '--format=csv,noheader'], text=True).strip(),
        initial_rss_bytes=psutil.Process().memory_info().rss,
        profiler_scope='cProfile wall timing separately for main and each executor task; worker aggregates overlap, not batch wall; per-thread CPU excludes native library child threads',
        limitation='DEBUG 2 actual CT training sources, full graphs; not production/server throughput. Profiling overhead included.')
    write_new(output/'request.json', request)
    result = dict(request=request, trials=[])
    reuse_events = []
    started = time.perf_counter()
    try:
        with preparation_reuse(pressure_aware_provider(ComparisonData), roots, event_callback=reuse_events.append) as reusable:
            provider_type = prepared_provider(reusable, output/'input_timing.jsonl')
            provider_kwargs = dict(source_samples=manifest['baseline']['source_samples'],
                raw_records=read(inventory)['raw_records'], config=config, bank=bank,
                root=output/'data', workers=4, resident_bytes=int(manifest['resident_gib']*2**30),
                budget=budget, regions_dir=source_root/'regions')
            provider = provider_type(**provider_kwargs)
            if canonical(provider.examples('train')+provider.examples('val')) != manifest['samples']:
                raise ValueError('Actual immutable three-case source fixture differs')
            indices = [row['index'] for row in provider.examples('train')]
            if len(indices) != 2 or [provider._example(i)['id'] for i in indices] != request['sources_used']:
                raise ValueError('Original physical2 source identities required')
            for index in indices:
                if list(provider.candidate_keys(index, 'native_listwise', 4)) != request['candidate_keys']:
                    raise ValueError('Full epoch4 rotating candidate contract differs')
            previous_fingerprint = None
            trials = (('epoch4_new_combination', 4, False),
                      ('epoch4_identical_repeat', 4, False),
                      ('epoch5_rotation_warm_case_and_upper', 5, False),
                      ('epoch4_reopened_provider', 4, True))
            for label, epoch, reopened in trials:
                if reopened:
                    # Fresh provider LRU, same owned completed disk publications.
                    # Do not delete/evict/mutate any source-cache namespace.
                    provider = provider_type(**provider_kwargs)
                profile = cProfile.Profile()
                cpu, main_cpu = psutil.Process().cpu_times(), time.thread_time()
                before = provider.report()
                print('CPU DEBUG start '+label, flush=True)
                began = time.perf_counter()
                with worker_profiles() as records:
                    batch = profile.runcall(provider.batch, indices, 'native_listwise', epoch, True, full=False)
                elapsed = time.perf_counter()-began
                after_cpu = psutil.Process().cpu_times()
                row = dict(label=label, view_epoch=epoch, fresh_provider_LRU=reopened, wall_seconds=elapsed,
                    process_cpu_seconds=after_cpu.user+after_cpu.system-cpu.user-cpu.system,
                    main_thread_cpu_seconds=time.thread_time()-main_cpu,
                    worker_task_thread_cpu_seconds=sum(r['thread_cpu_seconds'] for r in records),
                    worker_task_wall_seconds_inclusive_sum=sum(r['wall_seconds'] for r in records),
                    worker_tasks=len(records), rss_bytes=psutil.Process().memory_info().rss,
                    provider_before=before, provider_after=provider.report(),
                    counts=list(batch.counts), source_ids=list(batch.bridge_source_ids),
                    candidate_keys=batch.bridge_candidate_keys)
                if list(batch.counts) != [8,8]:
                    raise AssertionError('Complete physical2/train8 batch required')
                row['main_tottime'] = stats_report([profile], output/(label+'_main'))
                row['workers_tottime'] = stats_report([r['profile'] for r in records], output/(label+'_workers'))
                row['worker_tasks_detail'] = [{k:v for k,v in r.items() if k!='profile'} for r in records]
                observed = fingerprint(vars(batch))
                if epoch == 4 and previous_fingerprint is not None:
                    row['exact_repeat'] = compare(previous_fingerprint, observed)
                    if not row['exact_repeat']['exact_values_layout_metadata_equal']:
                        raise AssertionError('Identical epoch4 batch differs on cache reuse')
                if epoch == 4 and previous_fingerprint is None:
                    previous_fingerprint = observed
                write_new(output/(label+'.json'), row)
                result['trials'].append(row)
                print(json.dumps({key:row[key] for key in ('label','wall_seconds','process_cpu_seconds',
                    'main_thread_cpu_seconds','worker_task_thread_cpu_seconds','worker_tasks','rss_bytes')}), flush=True)
                del batch
        result.update(reuse_events=reuse_events, elapsed_seconds=time.perf_counter()-started,
            original_references_preserved=ref_before=={str(path):sha(path) for path in references},
            existing_cache_inventory_size_mtime_preserved=cache_before=={str(path):tree_stats(path) for path in roots},
            frozen_helpers_preserved=helpers=={name:sha(ROOT/name) for name in FILES})
        if not all(result[key] for key in ('original_references_preserved','existing_cache_inventory_size_mtime_preserved','frozen_helpers_preserved')):
            raise AssertionError('Original reference/cache/helper preservation failed')
        write_new(output/'report.json', result)
        print('CPU DEBUG complete '+str(output/'report.json'), flush=True)
    except Exception as error:
        write_new(output/'failure.json', dict(error=str(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)