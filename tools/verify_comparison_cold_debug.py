"""Real CT DEBUG: serial vs certified concurrent complete cold input batches.

Two repeated scheduling-probe batches [0,1] and [1,0], each physical2/train8,
share the same epoch6. This is not a production epoch order or speed benchmark.
Preserve original source/cache namespaces; run one complete CUDA update only
following exact CPU-input/RNG comparisons and actual concurrency verification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import gc
import importlib.util
import json
from pathlib import Path
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


def traced_provider(original, events):
    lock = threading.Lock()

    class TracedProvider(original):
        @contextmanager
        def _trace(self, stage, identity):
            began = time.perf_counter()
            try:
                yield
            finally:
                with lock:
                    events.append(dict(stage=stage, identity=identity,
                        thread=threading.current_thread().name, began=began,
                        ended=time.perf_counter()))

        def _case(self, case_id):
            with self._trace('case_fields', case_id):
                return super()._case(case_id)

        @contextmanager
        def _upper_context(self, example, *args, **kwargs):
            with self._trace('upper_context', example['id']):
                with super()._upper_context(example, *args, **kwargs):
                    yield

        def batch(self, indices, *args, **kwargs):
            with self._trace('full_batch', list(indices)):
                return super().batch(indices, *args, **kwargs)

    return TracedProvider


def overlap(events, stage):
    selected = [event for event in events if event['stage'] == stage]
    return [dict(left=a['identity'], right=b['identity'], left_thread=a['thread'],
        right_thread=b['thread'], seconds=min(a['ended'],b['ended'])-max(a['began'],b['began']))
        for index,a in enumerate(selected) for b in selected[index+1:]
        if a['thread'] != b['thread'] and a['identity'] != b['identity']
        and min(a['ended'],b['ended']) > max(a['began'],b['began'])]


def run(args):
    from tools.current_gpu import select_record
    selected_gpu = select_record(args.gpu)
    import psutil
    import torch
    from tools.verify_comparison_cache_debug import read, sha, canonical
    from tools.verify_comparison_gpu_debug import tree_stats
    from tools.verify_comparison_preparation_debug import fingerprint, compare
    from tools.verify_source_scope_debug import actual_update
    from hiercp_v1x.comparison_experiment import FORMAT, FILES, digest
    from hiercp_v1x.scope_probe_support import activate_original, state_digest
    from hiercp_v1x import bounded_scope, u_bridge_training as engine
    from hiercp_v1x.comparison_data import ComparisonData
    from hiercp_v1x.comparison_preparation import prepared_provider
    from hiercp_v1x.host_memory import PressureBudget, pressure_aware_provider, _evict_provider
    from hiercp_v1x.preparation_reuse import preparation_reuse
    from hiercp_v1x.comparison_runtime import _prefetch

    reference = ROOT/'work/v19_checked_DEBUG'
    inventory = ROOT/'work/v22_cumulative_u16_DEBUG_20261006/inventory.json'
    bank_path = ROOT/'work/v1x_real_CT_DEBUG_20261003_prepare/fixture.pt'
    manifest = read(reference/'experiment.json')
    source_root = Path(manifest['prepared_data_root']).resolve(strict=True)
    completed = ROOT/'work/runs/v1.9/source-scope/debug-20261008/verified/reopened_data'
    roots = [completed, source_root]
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in roots):
        raise FileExistsError('New disjoint actual-CT DEBUG output required')
    frozen_before = {name:sha(ROOT/name) for name in FILES}
    if (manifest['format'] != FORMAT or manifest['debug'] is not True
            or manifest['workers'] != 4 or manifest['explicit_batch_candidates'] != [2]
            or manifest['sha256'] != digest({k:v for k,v in manifest.items() if k!='sha256'})
            or frozen_before != manifest['helpers']
            or sha(inventory) != manifest['baseline']['inventory_sha256']
            or sha(bank_path) != manifest['baseline']['bank_fixture_sha256']):
        raise ValueError('Sealed real CT full physical2/train8/worker4 fixture differs')
    original = Path(manifest['original']['source'])
    if (canonical(activate_original(original)) != manifest['original']
            or canonical(bounded_scope.install(10, expected_snapshot_root=original)) != manifest['scope']):
        raise ValueError('Original byte-exact V1 or bounded m10 source contract differs')
    from hiercp.tensor import configure_runtime, collect_runtime_resources
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One explicitly selected CUDA device required')
    config = copy.deepcopy(manifest['config'])
    torch.set_num_threads(4)
    configure_runtime(deterministic=config['runtime']['deterministic'],
        allow_tf32=config['runtime']['allow_tf32'], cudnn_benchmark=config['runtime']['cudnn_benchmark'])
    cuda_bytes = int(manifest['cuda_gib']*2**30)
    capacity = torch.cuda.get_device_properties(0).total_memory
    if not 0 < cuda_bytes < capacity:
        raise ValueError('Original CUDA memory budget requires physical headroom')
    torch.cuda.set_per_process_memory_fraction(cuda_bytes/capacity)
    budget = PressureBudget(cuda_bytes, int(manifest['rss_gib']*2**30),
        resident_bytes=int(manifest['resident_gib']*2**30))
    initial = torch.load(reference/'initial.pt', map_location='cpu', weights_only=False)
    if (initial['contract_sha256'] != manifest['sha256']
            or initial['model_sha256'] != state_digest(initial['model'])):
        raise ValueError('Original complete model initial state differs')
    fixture = torch.load(bank_path, map_location='cpu', weights_only=False, mmap=True)
    bank = fixture['prototype_bank']
    if bank.fingerprint() != manifest['prototype_fingerprint']:
        raise ValueError('Real CT prototype bank identity differs')
    del fixture
    refs = [reference/'experiment.json', reference/'initial.pt', inventory, bank_path,
        *[Path(row['path']) for row in manifest['baseline']['source_samples']]]
    refs_before = {str(path):sha(path) for path in refs}
    cache_before = {str(path):tree_stats(path) for path in roots}
    execution_before = {str(path.relative_to(ROOT)):sha(path) for path in (ROOT/'hiercp_v1x').glob('*.py')}
    output.mkdir(parents=True, exist_ok=False)
    legacy_ref = subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),
        'rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    legacy_bytes = subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),
        'show',legacy_ref+':hiercp_v1x/comparison_upper_cache.py'],cwd=ROOT)
    legacy_path = output/'legacy_comparison_upper_cache.py'
    with legacy_path.open('xb') as stream:
        stream.write(legacy_bytes)
    spec = importlib.util.spec_from_file_location('hiercp_v1x._cold_debug_legacy_upper',legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    from hiercp_v1x import comparison_upper_cache as compact_module
    request = dict(debug=True, actual_CT=True, view_epoch=6, arm='native_listwise',
        candidate_keys=['P']+[f'U:{i}' for i in range(35,42)], physical_batch=2,
        candidates_per_source=8, workers=4, torch_threads=4,
        repeated_DEBUG_batches=[[0,1],[1,0]], training_epoch_order=False,
        full_training=False, full_evaluation=False, production_speed_benchmark=False,
        comparison='HEAD legacy global upper helper sequential vs new isolated upper helper certified staging(2)',
        legacy_git_ref=legacy_ref, legacy_upper_sha256=sha(legacy_path),
        model_config=config['model'], original_config=config, original_manifest_sha256=manifest['sha256'],
        selected_device=selected_gpu, resources=collect_runtime_resources('cuda',storage_path=output),
        original_reference_sha256=refs_before, execution_helper_sha256=execution_before,
        harness_sha256=sha(Path(__file__)), reuse_roots=list(map(str,roots)),
        planned_genuine_optimizer_updates=1, effective_batch=2)
    write_new(output/'request.json',request)
    results = dict(request=request, runs={})
    providers = []
    gpu_batch = None
    fingerprints = {}
    began = time.perf_counter()
    try:
        with preparation_reuse(pressure_aware_provider(ComparisonData),roots) as reusable:
            for mode in ('serial','concurrent'):
                directory = output/mode
                directory.mkdir()
                shutil.copytree(completed/'source_prepared',directory/'data/source_prepared')
                events = []
                if mode == 'serial':
                    with patch.object(compact_module,'compact_upper_provider',legacy.compact_upper_provider):
                        base_type = prepared_provider(reusable,directory/'input_timing.jsonl')
                else:
                    base_type = prepared_provider(reusable,directory/'input_timing.jsonl')
                provider_type = traced_provider(base_type,events)
                provider = provider_type(source_samples=manifest['baseline']['source_samples'],
                    raw_records=read(inventory)['raw_records'], config=config, bank=bank,
                    root=directory/'data', workers=4, resident_bytes=int(manifest['resident_gib']*2**30),
                    budget=budget, regions_dir=source_root/'regions')
                providers.append(provider)
                if canonical(provider.examples('train')+provider.examples('val')) != manifest['samples']:
                    raise ValueError('Immutable three-case fixture inventory differs')
                indices = [row['index'] for row in provider.examples('train')]
                if indices != [0,1] or [provider._example(i)['id'] for i in indices] != ['liver_5:0','liver_6:0']:
                    raise AssertionError('Original physical2 actual source identity required')
                batches = [indices,list(reversed(indices))]
                for index in indices:
                    if list(provider.candidate_keys(index,'native_listwise',6)) != request['candidate_keys']:
                        raise ValueError('Exact epoch6 rotating candidates required')
                if mode == 'concurrent' and getattr(provider,'independent_cold_inputs',False) is not True:
                    raise RuntimeError('New provider must explicitly certify independent cold input contexts')
                engine.restore_rng(initial['rng'])
                rng_before = engine.digest(engine.capture_rng())
                cpu_before = psutil.Process().cpu_times()
                started = time.perf_counter()
                print('REAL CT COLD DEBUG '+mode+' physical2 x8, two ordered scheduling-probe batches',flush=True)
                if mode == 'serial':
                    built = [provider.batch(ids,'native_listwise',6,True,full=False) for ids in batches]
                else:
                    with provider.cpu_staging(2):
                        with ThreadPoolExecutor(max_workers=2,thread_name_prefix='cold-debug-batch') as executor:
                            futures = [executor.submit(provider.batch,ids,'native_listwise',6,True,full=False) for ids in batches]
                            built = [future.result() for future in futures]
                seconds = time.perf_counter()-started
                cpu_after = psutil.Process().cpu_times()
                rng_after = engine.digest(engine.capture_rng())
                if rng_before != rng_after:
                    raise AssertionError('CPU preparation changed original global RNG states')
                if any(list(batch.counts)!=[8,8] for batch in built):
                    raise AssertionError('Complete physical2/train8 inputs required')
                observed = [fingerprint(vars(batch)) for batch in built]
                fingerprints[mode] = observed
                for number,value in enumerate(observed):
                    write_new(directory/f'batch_{number}_fingerprint.json',value)
                row = dict(wall_seconds=seconds,
                    process_cpu_seconds=cpu_after.user+cpu_after.system-cpu_before.user-cpu_before.system,
                    rng_before=rng_before,rng_after=rng_after,events=events,
                    case_overlap=overlap(events,'case_fields'),upper_overlap=overlap(events,'upper_context'),
                    batch_overlap=overlap(events,'full_batch'),provider=provider.report(),
                    rss_bytes=psutil.Process().memory_info().rss)
                write_new(directory/'report.json',row)
                results['runs'][mode]=row
                if mode == 'concurrent':
                    if not row['batch_overlap'] or not row['case_overlap'] or not row['upper_overlap']:
                        raise AssertionError('Distinct-source batch/case/upper context overlap was not observed')
                    results['cpu_comparisons']=[compare(left,right) for left,right in zip(fingerprints['serial'],observed)]
                    if not all(check['exact_values_layout_metadata_equal'] for check in results['cpu_comparisons']):
                        raise AssertionError('Concurrent complete batch values/layout/metadata differ')
                    gpu_batch=built[0]
                    print('CPU exact complete batch fingerprints and global RNG verified',flush=True)
                    # Actual prefetch receipt, same epoch and full physical2.
                    # These layouts are already warm; unit tests own cold admission.
                    prefetched=list(_prefetch(provider,batches,'native_listwise',6,pin_memory=False,
                        receipt_path=directory/'prefetch.jsonl'))
                    checks=[compare(left,fingerprint(vars(right))) for left,right in zip(observed,prefetched)]
                    if len(prefetched)!=2 or not all(check['exact_values_layout_metadata_equal'] for check in checks):
                        raise AssertionError('Actual ordered prefetch changed complete CPU inputs')
                    results['prefetch_warm_comparisons']=checks
                    del prefetched
                del built,observed
                _evict_provider(provider)
                gc.collect()
        print('REAL CT DEBUG full10434532 CUDA forward/backward/optimizer physical2 x8',flush=True)
        results['cuda']=actual_update(gpu_batch,initial,config,budget)
        results.update(original_references_preserved=refs_before=={str(path):sha(path) for path in refs},
            existing_cache_inventory_size_mtime_preserved=cache_before=={str(path):tree_stats(path) for path in roots},
            frozen_helpers_preserved=frozen_before=={name:sha(ROOT/name) for name in FILES},
            execution_helpers_unchanged_during_run=execution_before=={str(path.relative_to(ROOT)):sha(path) for path in (ROOT/'hiercp_v1x').glob('*.py')},
            wall_seconds=time.perf_counter()-began)
        if not all(results[key] for key in ('original_references_preserved','existing_cache_inventory_size_mtime_preserved',
                'frozen_helpers_preserved','execution_helpers_unchanged_during_run')):
            raise AssertionError('Source/reference/cache/helper preservation failed')
        write_new(output/'report.json',results)
        print('REAL CT COLD DEBUG complete '+str(output/'report.json'),flush=True)
    except Exception as error:
        write_new(output/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),
            completed=results,wall_seconds=time.perf_counter()-began))
        raise
    finally:
        for provider in providers:
            _evict_provider(provider)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())