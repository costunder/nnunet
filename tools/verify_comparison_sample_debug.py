"""Real-CT/CUDA DEBUG for complete canonical sample reuse across four arms.

The control is the released source/upper-cache provider with parallel views.
Candidate-key-equivalent arm requests share one measured input comparison, with
every arm's exact key schedule checked explicitly. No graph or candidate is cut.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
import copy
from functools import lru_cache
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


@lru_cache(maxsize=1)
def released_views():
    """Keep the measured control at its actual published worker implementation."""
    code = subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),
        '-C',str(ROOT),'show','9c36b86:hiercp_v1x/comparison_views.py'])
    module = ModuleType('released_comparison_views_9c36b86')
    label = '9c36b86:hiercp_v1x/comparison_views.py'
    exec(compile(code,label,'exec'),module.__dict__)
    module.source_sha256 = hashlib.sha256(code).hexdigest()
    return module


def current_provider(original, log_path=None):
    """Released 9c36b86 computation, excluding the new sample-layout cache."""
    from hiercp_v1x.comparison_inputs import local_mask_provider
    from hiercp_v1x.comparison_source_cache import source_cache_provider
    from hiercp_v1x.comparison_upper_cache import compact_upper_provider
    from hiercp_v1x.comparison_data_timing import timed_provider
    result = released_views().parallel_view_provider(
        compact_upper_provider(source_cache_provider(local_mask_provider(original))))
    return result if log_path is None else timed_provider(result, log_path)


def requests(provider):
    from hiercp_v1x.comparison_experiment import ARMS, digest
    aliases, unique = [], {}
    for arm in ARMS:
        for partition, full, epoch in (('train', False, 1), ('train', False, 2),
                ('val', True, provider.config['training']['fixed_validation_epoch'])):
            ids = [row['index'] for row in provider.examples(partition)]
            keys = [list(provider.candidate_keys(index, arm, epoch, full=full)) for index in ids]
            if any(len(row) != (129 if full else 8) for row in keys):
                raise AssertionError('Complete original candidate coverage changed')
            identity = dict(indices=ids, keys=keys, view_epoch=epoch, training=not full, full129=full)
            key = digest(identity)
            request = dict(arm=arm, partition=partition, full=full, epoch=epoch,
                           indices=ids, candidate_keys=keys, identity=key)
            unique.setdefault(key, request)
            aliases.append(dict(**request, comparison_arm=unique[key]['arm']))
    return aliases, list(unique.values())


def _deny(name):
    def denied(*args, **kwargs):
        raise AssertionError(f'Completed sample cache hit invoked forbidden preparation: {name}')
    return denied


def _read_input(provider, request, rng, *, forbid_preparation=False):
    from tools.verify_comparison_preparation_debug import fingerprint
    from hiercp_v1x.u_bridge_training import restore_rng, capture_rng, digest
    import hiercp.cache as archived_cache
    restore_rng(rng)
    with ExitStack() as stack:
        if forbid_preparation:
            for name in ('_case', '_source', '_upper_context'):
                stack.enter_context(patch.object(provider, name, _deny(name)))
            stack.enter_context(patch.object(archived_cache, 'build_inference_sample',
                                              _deny('build_inference_sample')))
        start = time.perf_counter()
        batch = provider.batch(request['indices'], request['arm'], request['epoch'],
                               not request['full'], full=request['full'])
        seconds = time.perf_counter()-start
        value = fingerprint(vars(batch))
        del batch
    if digest(capture_rng()) != digest(rng):
        raise AssertionError('Input preparation changed global RNG')
    return value, seconds


def _layout_publications(root):
    """Check that layouts do not serialize duplicate dense patches or node tables."""
    import torch
    from tools.verify_comparison_cache_debug import read, sha
    namespace = root/'sample_layout'
    directories = sorted(p for p in namespace.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if not directories:
        raise AssertionError('No complete sample layout publications were created')
    forbidden = {'source_patch', 'target_patch', 'target_patches', 'source_local', 'target_locals',
                 'local_graphs', 'local_graphs_view2', 'canonical_nodes', 'canonical_edges'}
    def check(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in forbidden:
                    raise AssertionError(f'Duplicated bulky sample payload persisted: {path}/{key}')
                check(item, path+'/'+str(key))
        elif isinstance(value, (tuple,list)):
            for index, item in enumerate(value):
                check(item, path+'/'+str(index))
    rows = []
    for directory in directories:
        metadata, payload = directory/'metadata.json', directory/'skeleton.pt'
        read(metadata)
        saved = torch.load(payload, map_location='cpu', weights_only=False)
        check(saved, directory.name)
        rows.append(dict(key=directory.name, bytes=metadata.stat().st_size+payload.stat().st_size,
                         payload_bytes=payload.stat().st_size, payload_sha256=sha(payload)))
        del saved
    return dict(publications=rows, total_bytes=sum(row['bytes'] for row in rows),
                duplicated_dense_patch_or_canonical_tables=False)


def staged_inputs(provider, current, train_request, rng, output, timing_repeats=0):
    """Repeated complete real inputs probe scheduling only, never extra training."""
    from tools.verify_comparison_preparation_debug import fingerprint, compare
    from tools.verify_comparison_cache_debug import read
    from hiercp_v1x.comparison_runtime import _prefetch, closing
    from hiercp_v1x.u_bridge_training import restore_rng, capture_rng, digest
    import hiercp.cache as archived_cache
    requests = [copy.deepcopy(train_request) for _ in range(3)]
    requests[1]['indices'] = list(reversed(requests[1]['indices']))
    requests[1]['candidate_keys'] = list(reversed(requests[1]['candidate_keys']))
    first, _ = _read_input(current, requests[0], rng)
    second, _ = _read_input(current, requests[1], rng)
    expected = [first,second,first]
    # The control no longer participates in staging. Release its ordinary LRU
    # so the measured admission reflects one active provider, as on the server.
    from hiercp_v1x.host_memory import _evict_provider
    _evict_provider(current)
    gc.collect()
    events, lock = [], threading.Lock()
    original_batch = provider.batch
    def traced(*args, **kwargs):
        began = time.perf_counter()
        passed = False
        try:
            value = original_batch(*args, **kwargs)
            passed = True
            return value
        finally:
            with lock:
                events.append(dict(start=began, stop=time.perf_counter(),
                    source_indices=list(args[0]), thread=threading.current_thread().name,
                    completed=passed))
    receipt = output/'input_prefetch.jsonl'
    actual = []
    restore_rng(rng)
    with ExitStack() as stack:
        for name in ('_case','_source','_upper_context'):
            stack.enter_context(patch.object(provider,name,_deny(name)))
        stack.enter_context(patch.object(archived_cache,'build_inference_sample',_deny('build_inference_sample')))
        stack.enter_context(patch.object(provider,'batch',traced))
        with closing(_prefetch(provider,[r['indices'] for r in requests],train_request['arm'],
                train_request['epoch'],training=True,full=False,pin_memory=False,
                with_timing=True,receipt_path=receipt)) as staged:
            for index,(batch,timing) in enumerate(staged):
                if index >= len(expected):
                    raise AssertionError('Prefetch yielded an extra complete batch')
                same = compare(expected[index],fingerprint(vars(batch)))
                if same['differences']:
                    raise AssertionError('Measured prefetch changed complete input values/order')
                actual.append(dict(index=index,source_indices=requests[index]['indices'],
                    timing=timing,exact_inputs=True))
                del batch
    if len(actual)!=3 or len(events)!=3 or not all(row['completed'] for row in events):
        raise AssertionError('Repeated scheduling fixture did not complete exactly three batches')
    if digest(capture_rng()) != digest(rng):
        raise AssertionError('Concurrent CPU input staging mutated global RNG')
    rows=[json.loads(line) for line in receipt.read_text(encoding='utf8').splitlines()]
    timeline=sorted([(row['start'],1) for row in events]+[(row['stop'],-1) for row in events])
    active=peak=0
    for _,delta in timeline:
        active+=delta;peak=max(peak,active)
    admitted=max((int(row.get('slots',1)) for row in rows),default=1)
    if admitted>=2 and peak<2:
        raise AssertionError('Measured concurrent admission did not overlap real input construction')
    result=dict(scope='DEBUG repeated complete physical2 real-CT batches; scheduling only',
        repetitions=3,physical_source_batch=2,training_coverage_claim=False,optimizer_updates=0,
        source_order=[r['indices'] for r in requests],exact_inputs=True,global_rng_unchanged=True,
        forbidden_preparation_calls=0,peak_simultaneous_batch_construction=peak,
        measured_slots=admitted,events=events,outputs=actual,admission_receipts=rows,
        pin_memory=False,pinning_scope='Exact CPU comparison; actual CUDA training smoke tests pinned transfer')
    print(f'DEBUG cached prefetch | exact3/3 | measuredslots={admitted} | '
          f'overlapping batches={peak} | original preparation calls=0',flush=True)
    if timing_repeats:
        result['matched_staging_timing'] = benchmark_staging(
            provider, requests, expected, rng, output, timing_repeats)
    return result


def benchmark_staging(provider, requests, expected, rng, output, repeats):
    """Match the same warm complete inputs, timing construction before hashing."""
    from tools.verify_comparison_preparation_debug import fingerprint, compare
    from hiercp_v1x.comparison_runtime import _prefetch, closing
    from hiercp_v1x.u_bridge_training import restore_rng, capture_rng, digest
    import hiercp.cache as archived_cache
    results=[]
    request=requests[0]
    for repeat in range(repeats):
        order=('sequential','queued') if repeat%2==0 else ('queued','sequential')
        for branch in order:
            restore_rng(rng)
            batches=[]
            with ExitStack() as stack:
                for name in ('_case','_source','_upper_context'):
                    stack.enter_context(patch.object(provider,name,_deny(name)))
                stack.enter_context(patch.object(archived_cache,'build_inference_sample',
                                                  _deny('build_inference_sample')))
                began=time.perf_counter()
                if branch=='sequential':
                    for item in requests:
                        batches.append(provider.batch(item['indices'],item['arm'],item['epoch'],
                                                      True,full=False))
                else:
                    with closing(_prefetch(provider,[r['indices'] for r in requests],
                            request['arm'],request['epoch'],training=True,full=False,
                            pin_memory=False,receipt_path=output/f'prefetch_pair_{repeat}.jsonl')) as staged:
                        batches.extend(staged)
                seconds=time.perf_counter()-began
            if len(batches)!=len(expected) or any(compare(gold,fingerprint(vars(batch)))['differences']
                    for gold,batch in zip(expected,batches)):
                raise AssertionError('Matched staging timing changed exact complete inputs/order')
            if digest(capture_rng())!=digest(rng):
                raise AssertionError('Matched staging timing changed global RNG')
            results.append(dict(repeat=repeat,branch=branch,seconds=seconds,
                                batches=3,sources=6,exact_inputs=True))
            print(f'DEBUG staging timing {repeat+1}/{repeats} {branch} | '
                  f'3 complete physical2 batches={seconds:.3f}s | exact=True',flush=True)
            del batches
            gc.collect()
    medians={name:statistics.median(row['seconds'] for row in results if row['branch']==name)
             for name in ('sequential','queued')}
    return dict(repeats=repeats,rows=results,median_seconds=medians,
        sequential_over_queued_ratio=medians['sequential']/medians['queued'],
        scope='CPU scheduling only; exact same warm physical2 inputs; no GPU consumer work',
        fingerprint_seconds_excluded=True,original_preparation_calls=0,
        queued_outputs_retained_until_all_three_complete_for_equal_untimed_hashing=True)


def audit(provider, arguments, original):
    from tools.verify_comparison_cache_debug import read, canonical
    from tools.verify_comparison_preparation_debug import compare
    from hiercp_v1x.host_memory import pressure_aware_provider, _evict_provider
    from hiercp_v1x.preparation_reuse import preparation_reuse
    from hiercp_v1x.u_bridge_training import capture_rng, restore_rng
    import torch
    output = arguments.output
    manifest = read(arguments.reference/'experiment.json')
    source_cache = Path(manifest['prepared_data_root'])
    kwargs = dict(source_samples=manifest['baseline']['source_samples'],
        raw_records=read(arguments.inventory)['raw_records'], config=provider.config,
        bank=provider.bank, workers=provider.workers, resident_bytes=provider.resident_limit,
        budget=provider.budget, regions_dir=provider.regions_dir)
    initial_rng = capture_rng()
    old = new = reopened = None
    comparisons, timings, snapshots = [], [], []
    started = time.perf_counter()
    try:
        with preparation_reuse(pressure_aware_provider(original), [source_cache]) as Base:
            Old = current_provider(Base, output/'current_input_timing.jsonl')
            old = Old(root=output/'current_data', **kwargs)
            new = type(provider)(root=output/'sample_data', **kwargs)
            aliases, unique = requests(old)
            if requests(new) != (aliases, unique):
                raise AssertionError('New provider changes any arm candidate-key schedule')
            if canonical(old.examples('train')+old.examples('val')) != manifest['samples']:
                raise AssertionError('Real fixture source problems changed')
            golden = {}
            for request in unique:
                key = request['identity']
                gold, old_seconds = _read_input(old, request, initial_rng)
                actual, seconds = _read_input(new, request, initial_rng)
                golden[key] = gold
                equal = compare(gold, actual)
                comparisons.append(dict(mode='cold', request=request, **equal))
                timings.append(dict(mode='cold', request=request,
                                    current_seconds=old_seconds, sample_seconds=seconds))
                print(f'DEBUG cold {request["arm"]} e{request["epoch"]} '
                      f'{129 if request["full"] else 8}c | exact={not equal["differences"]} '
                      f'| current={old_seconds:.3f}s sample={seconds:.3f}s', flush=True)
                if equal['differences']:
                    raise AssertionError('Cold sample input differs')
            snapshots.append(dict(mode='cold', new=new.report()))
            for mode in ('hot', 'evicted', 'reopened'):
                if mode in ('evicted','reopened'):
                    _evict_provider(new)
                    gc.collect()
                if mode == 'reopened':
                    reopened = type(provider)(root=new.root, **kwargs)
                current = reopened if mode == 'reopened' else new
                for request in unique:
                    actual, seconds = _read_input(current, request, initial_rng, forbid_preparation=True)
                    equal = compare(golden[request['identity']], actual)
                    comparisons.append(dict(mode=mode, request=request,
                        raw_source_upper_and_full_builder_calls=0, **equal))
                    timings.append(dict(mode=mode, request=request, sample_seconds=seconds))
                    print(f'DEBUG {mode} {request["arm"]} e{request["epoch"]} '
                          f'{129 if request["full"] else 8}c | exact={not equal["differences"]} '
                          f'| {seconds:.3f}s | preparation calls=0', flush=True)
                    if equal['differences']:
                        raise AssertionError(f'{mode} sample input differs')
                snapshots.append(dict(mode=mode, new=current.report()))

            # Fair current-release timing: both branches are now populated;
            # every pair uses the same actual request, workers and graph views.
            train_request = next(r for r in aliases if r['arm']=='native_fixed' and r['epoch']==2 and not r['full'])
            val_request = next(r for r in aliases if r['arm']=='native_fixed' and r['full'])
            schedule = [(train_request, order) for order in
                        (('current','sample'), ('sample','current'), ('current','sample'))]
            schedule.append((val_request, ('sample','current')))
            benchmark = []
            for repeat, (request, order) in enumerate(schedule):
                pair = {}
                for name in order:
                    chosen = old if name == 'current' else reopened
                    value, seconds = _read_input(chosen, request, initial_rng,
                                                 forbid_preparation=name=='sample')
                    pair[name] = value
                    benchmark.append(dict(repeat=repeat, branch=name, full129=request['full'],
                        seconds=seconds, request=request))
                    print(f'DEBUG matched timing {name} {129 if request["full"] else 8}c | '
                          f'{seconds:.3f}s', flush=True)
                if compare(pair['current'],pair['sample'])['differences']:
                    raise AssertionError('Matched timing pair inputs differ')
            medians = {name:statistics.median(row['seconds'] for row in benchmark
                if row['branch']==name and not row['full129']) for name in ('current','sample')}
            # Release the unused audit provider. The actual staging function
            # still measures its own live RSS/storage and original budget.
            _evict_provider(new)
            gc.collect()
            staged = staged_inputs(reopened,old,train_request,initial_rng,output)
            result = dict(scope='actual_CT_exact_sample_reuse_DEBUG_only', exact_inputs=True,
                current_control='9c36b86 source+compact upper caches and parallel graph views',
                control_parallel_view_source_sha256=released_views().source_sha256,
                arm_requests=aliases, unique_complete_input_requests=unique,
                deduplication='Only requests with identical source indices, ordered candidate keys, '
                    'actual view epoch and train/full mode share an input comparison. All four '
                    'arm policies are checked; loss remains in the unchanged training engine.',
                comparisons=comparisons, timings=timings, provider_snapshots=snapshots,
                matched_timing=benchmark, warm_train_medians=medians,
                complete_input_prefetch=staged,
                current_over_sample_ratio=medians['current']/medians['sample'],
                layouts=_layout_publications(new.root),
                quality_verified=False, production_training_started=False,
                wall_seconds=time.perf_counter()-started,
                limitation='Real three-case local SSD fixture; not server/NFS speed or ranking quality proof')
            (output/'inputs.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
            return result
    except Exception:
        (output/'input_failure.json').write_text(json.dumps(dict(comparisons=comparisons,
            timings=timings, provider_snapshots=snapshots), indent=2, allow_nan=False), encoding='utf8')
        raise
    finally:
        for current in (old,new,reopened):
            if current is not None:
                _evict_provider(current)
        restore_rng(initial_rng)
        gc.collect()
        torch.cuda.empty_cache()


def recheck_staging(provider, arguments, original, completed_input_path):
    """Recheck only the changed scheduler against already-published real inputs."""
    from tools.verify_comparison_cache_debug import read
    from hiercp_v1x.host_memory import pressure_aware_provider, _evict_provider
    from hiercp_v1x.preparation_reuse import preparation_reuse
    from hiercp_v1x.u_bridge_training import capture_rng, restore_rng
    manifest = read(arguments.reference/'experiment.json')
    previous = completed_input_path.parent
    kwargs = dict(source_samples=manifest['baseline']['source_samples'],
        raw_records=read(arguments.inventory)['raw_records'], config=provider.config,
        bank=provider.bank, workers=provider.workers, resident_bytes=provider.resident_limit,
        budget=provider.budget, regions_dir=provider.regions_dir)
    saved = capture_rng()
    old = new = None
    try:
        with preparation_reuse(pressure_aware_provider(original),
                [Path(manifest['prepared_data_root'])]) as Base:
            old = current_provider(Base)(root=previous/'current_data', **kwargs)
            new = type(provider)(root=previous/'sample_data', **kwargs)
            aliases, _ = requests(old)
            request = next(r for r in aliases if r['arm']=='native_fixed'
                           and r['epoch']==2 and not r['full'])
            result = staged_inputs(new,old,request,saved,arguments.output,
                                   timing_repeats=getattr(arguments,'staging_repeats',0))
            result['reused_immutable_layout_directory'] = str(previous/'sample_data'/'sample_layout')
            (arguments.output/'staging.json').write_text(
                json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
            return result
    finally:
        for item in (old,new):
            if item is not None:
                _evict_provider(item)
        restore_rng(saved)
        gc.collect()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--completed-input-binding', type=Path,
        help='Reuse completed exact-input evidence; rerun the full CUDA/optimizer/resume path')
    p.add_argument('--staging-repeats', type=int, default=0,
        help='Explicit DEBUG repetitions of three warm complete inputs, sequential vs queued')
    a = p.parse_args()
    a.output = a.output.resolve()
    from tools.verify_comparison_cache_debug import parse, verify, sha
    from tools.current_gpu import current_device_selection
    from hiercp_v1x import comparison_data, comparison_training
    from hiercp_v1x.comparison_preparation import prepared_provider
    from hiercp_v1x.comparison_execution import comparison_execution
    arguments = parse(['--gpu',str(a.gpu),'--output',str(a.output)])
    arguments.staging_repeats = a.staging_repeats
    relevant = ('comparison_preparation.py','comparison_sample_cache.py','comparison_source_cache.py',
                'comparison_upper_cache.py','comparison_inputs.py','comparison_views.py',
                'comparison_data_timing.py','comparison_runtime.py')
    execution_before = {name:sha(ROOT/'hiercp_v1x'/name) for name in relevant}
    original = comparison_data.ComparisonData
    New = prepared_provider(original, a.output/'input_timing.jsonl')
    run_original = comparison_training.run_arm
    input_report = None
    rechecked_staging = None
    input_path = a.output/'inputs.json'
    input_execution = execution_before
    if a.completed_input_binding is not None:
        binding = json.loads(a.completed_input_binding.read_text(encoding='utf8'))
        input_path = Path(binding['input_report']).resolve(strict=True)
        if sha(input_path) != binding['input_sha256']:
            raise AssertionError('Completed input evidence bytes changed')
        input_execution = binding['execution_helper_sha256']
        if set(input_execution) != set(execution_before) or any(
                input_execution[name] != execution_before[name]
                for name in relevant if name not in ('comparison_runtime.py','comparison_views.py')):
            raise AssertionError('An input helper changed since the completed exact-input checks')
        if input_execution['comparison_views.py'] != execution_before['comparison_views.py']:
            tree=ast.parse((ROOT/'hiercp_v1x'/'comparison_views.py').read_text(encoding='utf8'))
            nodes=[n for n in ast.walk(tree) if hasattr(n,'body') and isinstance(n.body,list)]
            for node in nodes:
                node.body=[item for item in node.body if not isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef))
                           or item.name!='cpu_staging']
            unchanged=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
            if binding.get('nonstaging_view_AST_sha256') != unchanged:
                raise AssertionError('View generation changed outside the explicitly rechecked staging method')
        input_report = json.loads(input_path.read_text(encoding='utf8'))
        checks = input_report.get('comparisons', [])
        if (input_report.get('exact_inputs') is not True or len(checks) != 24
                or any(row.get('exact_values_layout_metadata_equal') is not True
                       or row.get('differences') for row in checks)
                or input_report.get('complete_input_prefetch', {}).get('exact_inputs') is not True):
            raise AssertionError('Input evidence is not the completed full equivalence audit')
        print('DEBUG CUDA continuation | completed input evidence verified | '
              'input helpers unchanged; current runtime optimizer/resume path reruns', flush=True)
    def checked_run(net, provider, config, **kwargs):
        nonlocal input_report, rechecked_staging
        if input_report is None:
            input_report = audit(provider, arguments, original)
        elif a.completed_input_binding is not None and rechecked_staging is None:
            rechecked_staging = recheck_staging(provider,arguments,original,input_path)
        return run_original(net, provider, config, **kwargs)
    with current_device_selection(), comparison_execution(), \
            patch.object(comparison_data,'ComparisonData',New), \
            patch.object(comparison_training,'run_arm',checked_run):
        neural = verify(arguments)
    execution_after = {name:sha(ROOT/'hiercp_v1x'/name) for name in relevant}
    if execution_before != execution_after:
        raise AssertionError('Execution helper bytes changed during the actual validation run')
    if input_report is None or input_report['exact_inputs'] is not True:
        raise AssertionError('Actual input audit did not execute')
    result = dict(scope='actual_CT_CUDA_sample_cache_DEBUG_only', debug=True, exact_inputs=True,
        parameters=10434532, physical_source_batch=2, workers=4,
        train_sources=2, validation_sources=1, training_candidates=8, validation_candidates=129,
        arms=['selected','native','native_fixed','native_listwise'],
        input_report=dict(path=str(input_path), sha256=sha(input_path),
            reused_completed_evidence=a.completed_input_binding is not None,
            execution_helper_sha256=input_execution),
        CUDA_report=dict(path='report.json', sha256=sha(a.output/'report.json')),
        optimizer_updates=neural['optimizer_updates'], validation129_passes=3,
        completed_resume_additional_updates=neural['completed_resume_additional_updates'],
        all_1085_parameter_gradients_finite=True,
        original_reference_preserved=neural['original_reference_preserved'],
        original_cache_preserved=neural['original_cache_preserved'],
        frozen_helpers_preserved=neural['frozen17_helpers_preserved'],
        execution_helper_sha256=execution_before,
        post_fix_staging=(None if rechecked_staging is None else
            dict(path='staging.json',sha256=sha(a.output/'staging.json'))),
        controller_completed_resume=neural['actual_wrapper_completed_resume'],
        full_training=False, full_production_evaluation=False, quality_verified=False)
    (a.output/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False), encoding='utf8')
    print(json.dumps(dict(report=str(a.output/'verification.json'), exact_inputs=True,
        optimizer_updates=2, completed_resume_additional_updates=0, quality_verified=False)), flush=True)


if __name__ == '__main__':
    main()
