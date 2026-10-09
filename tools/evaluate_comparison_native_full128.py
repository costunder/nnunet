"""Read-only native full-P+128U evaluation of current four-arm checkpoints.

Uses the existing historical real-CT geometry and whole-case score path. Only
the additive current-checkpoint loader differs. BEST and latest are distinct
predeclared selections; no hard-validation score chooses a new checkpoint.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

EXPERIMENTS = {
    'selected': 'v18_selected_m10_seed42_memory',
    'native': 'v18_native_m10_seed42_memory',
    'native_fixed': 'v19_native_fixed_m10_seed42',
    'native_listwise': 'v19_native_listwise_m10_seed42',
}


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--experiments-dir', type=Path, required=True)
    parser.add_argument('--native-run', type=Path, required=True)
    parser.add_argument('--prepared-cache', type=Path, required=True)
    parser.add_argument('--region-cache-source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--physical-batch-candidates', type=int, nargs='+', required=True)
    parser.add_argument('--cuda-gib', type=float, required=True)
    parser.add_argument('--rss-gib', type=float, required=True)
    parser.add_argument('--resident-gib', type=float, required=True)
    parser.add_argument('--minimum-free-gib', type=float, required=True)
    parser.add_argument('--selections', choices=('best', 'latest'), nargs='+', default=['best', 'latest'])
    parser.add_argument('--worker-job', help=argparse.SUPPRESS)
    parser.add_argument('--worker-gpu', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (len(set(args.gpus)) != len(args.gpus) or any(gpu not in (1, 5, 6) for gpu in args.gpus)
            or args.workers < 2 or not 0 < args.resident_gib < args.rss_gib
            or min(args.cuda_gib, args.minimum_free_gib) <= 0
            or args.physical_batch_candidates != sorted(set(args.physical_batch_candidates))
            or min(args.physical_batch_candidates) < 1
            or len(set(args.selections)) != len(args.selections)):
        raise ValueError('Use only authorized GPUs1/5/6 and explicit parallel/resource/batch settings')
    if bool(args.worker_job) != (args.worker_gpu is not None):
        raise ValueError('Worker job and assigned authorized GPU must be provided together')
    if args.worker_gpu is not None and args.worker_gpu not in args.gpus:
        raise ValueError('Worker GPU is outside the authorized request')
    return args


def source_receipt():
    from hiercp_v1x.historical_evaluation import sha
    directories = ('hiercp_v1x', 'hiercp_v22', 'hiercp_v222', 'l0_regions',
                   'l0_local_cnn', 'l0_sage', 'l0_ezsp', 'l0_exploration', 'tools')
    return {path.relative_to(ROOT).as_posix(): sha(path)
            for directory in directories for path in sorted((ROOT / directory).glob('*.py'))}


def verify_request(args, request):
    from hiercp_v1x.historical_evaluation import sha
    if any(sha(ROOT / name) != digest for name, digest in request['source'].items()):
        raise ValueError('Evaluation source changed during the bound run')
    if sha(args.native_run / 'inventory/index.json') != request['inventory_sha256']:
        raise ValueError('Native hard-evaluation inventory changed')
    if sha(args.prepared_cache) != request['prepared_cache_sha256']:
        raise ValueError('Preserved original-input geometry index changed')
    for job in request['jobs']:
        if sha(job['checkpoint']) != job['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed; evaluation must not follow a moving model')


def worker(args):
    from tools.local_cnn_device import select
    select(args.worker_gpu)
    import torch
    import psutil
    import random
    import numpy as np
    from hiercp_v1x.historical_evaluation import (
        ResourceBudget, calibrate, encode_original_fields, evaluate_historical,
        original_geometry_provider, sha, write_new,
    )
    from hiercp_v1x.comparison_native_checkpoint import load_comparison
    from hiercp_v1x.transition_evaluation import validate_cohort
    from hiercp_v1x.transition_v1_data import NativeObservationDataset
    from hiercp_v1x.contracts import canonical_hash

    request = json.loads((args.output / 'request.json').read_text())
    verify_request(args, request)
    job = next(row for row in request['jobs'] if row['name'] == args.worker_job)
    destination = args.output / job['name']
    attempt = destination / 'attempts' / uuid.uuid4().hex
    attempt.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one verified actual CUDA GPU is required')
    properties = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    if not 0 < args.cuda_gib * 2**30 < min(free, total):
        raise ValueError('Requested CUDA budget must fit actual free VRAM with headroom')
    if args.rss_gib * 2**30 > psutil.virtual_memory().available + psutil.Process().memory_info().rss:
        raise ValueError('Requested RSS exceeds actual available RAM')
    if args.workers > len(psutil.Process().cpu_affinity()):
        raise ValueError('Requested workers exceed the actual CPU affinity')
    torch.cuda.set_per_process_memory_fraction(args.cuda_gib * 2**30 / total)
    torch.set_num_threads(args.workers)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    budget = ResourceBudget(int(args.cuda_gib * 2**30), int(args.rss_gib * 2**30))
    bundle = load_comparison(job['arm'], job['experiment'], job['selection'], budget=budget)
    if bundle.receipt['native_inventory_sha256'] != request['inventory_sha256']:
        raise ValueError('Checkpoint and hard evaluation do not share the sealed native inventory')
    from hiercp.tensor import configure_runtime
    runtime = bundle.config['runtime']
    configure_runtime(deterministic=bool(runtime.get('deterministic', True)),
        allow_tf32=bool(runtime.get('allow_tf32', False)),
        cudnn_benchmark=bool(runtime.get('cudnn_benchmark', False)))
    inventory = json.loads((args.native_run / 'inventory/index.json').read_text())
    cohort = validate_cohort(inventory, debug=False)
    if (cohort['records'], cohort['observed_P'], cohort['unobserved_U']) != (2823, 135, 2688):
        raise ValueError('The historical full21/P135/U2688 benchmark contract changed')
    ds = NativeObservationDataset(args.native_run / 'inventory/index.json', 'inner_val', False)
    loader, preparation = original_geometry_provider(ds,
        root=attempt / 'geometry_request', workers=args.workers,
        resident_bytes=int(args.resident_gib * 2**30), rss_bytes=budget.rss_bytes,
        prepared_cache=args.prepared_cache, minimum_free_bytes=int(args.minimum_free_gib * 2**30))
    model = bundle.model
    model.eval()
    resources = dict(physical_gpu=args.worker_gpu, gpu_name=properties.name,
        total_VRAM_bytes=total, initial_free_VRAM_bytes=free,
        cpu_affinity_cores=len(psutil.Process().cpu_affinity()), available_RAM_bytes=psutil.virtual_memory().available,
        workers=args.workers, CUDA_limit_bytes=budget.cuda_bytes, RSS_limit_bytes=budget.rss_bytes,
        resident_bytes=int(args.resident_gib * 2**30),
        model=type(model).__name__, model_configuration=bundle.config['model'],
        parameters=sum(value.numel() for value in model.parameters()),
        trainable_parameters=sum(value.numel() for value in model.parameters() if value.requires_grad),
        margin_mm=10, input_representation='unchanged original five-channel/48-cube two-view GAT/CNN',
        hidden_dim=128, heads=4, local_layers=3, task_layers=2, alignment_layers=2,
        validation_cases=21, validation_records=2823, observed_P=135, unobserved_U=2688,
        precision_AMP=bool(bundle.config['training']['amp']),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        evaluation_fraction=1., DEBUG=False, model_training=False, optimizer_updates=0,
        checkpoint=bundle.checkpoint)
    write_new(attempt / 'resources.json', resources)

    def encode(cpu):
        budget()
        with torch.no_grad(), torch.autocast('cuda', enabled=bool(bundle.config['training']['amp'])):
            result = encode_original_fields(model, cpu.to('cuda'))
        budget()
        return result

    measured_batch, calibration = calibrate(encode, loader, ds,
        args.physical_batch_candidates, budget, arm='V1')
    write_new(attempt / 'calibration.json', calibration)
    print(f"{job['name']} FULL21 START | observations2823 P135 U2688 | physical L0 batch{measured_batch}", flush=True)
    report, _ = evaluate_historical(bundle, inventory, ds, loader,
        batch=measured_batch, budget=budget, output=attempt, debug=False,
        region_output=args.output / 'shared_regions')
    verify_request(args, request)
    report.update(arm=job['arm'], selection=job['selection'],
        benchmark='native full21 observed-P +128U; external training donor fixed per recipient',
        evaluation_request_sha256=canonical_hash(request),
        resources=resources, calibration=calibration, preparation=preparation,
        training_started=False, optimizer_updates=0, DEBUG=False,
        prior_source_anchor_validation_is_this_benchmark=False,
        checkpoint_selected_before_native_evaluation=True)
    report_path = destination / 'report.json'
    write_new(report_path, report)
    write_new(destination / 'complete.json', dict(name=job['name'],
        request_sha256=canonical_hash(request), checkpoint_sha256=job['checkpoint_sha256'],
        report_sha256=sha(report_path), report=str(report_path)))
    print(f"{job['name']} COMPLETE | {json.dumps(report['metrics'])}", flush=True)


def main(args):
    from hiercp_v1x.historical_evaluation import assert_new_destination, sha, write_new
    from hiercp_v1x.contracts import canonical_hash
    if args.worker_job:
        worker(args)
        return
    preserved = [args.native_run, args.prepared_cache,
                 *(args.experiments_dir / name for name in EXPERIMENTS.values())]
    if args.region_cache_source is not None:
        preserved.append(args.region_cache_source)
    assert_new_destination(args.output, preserved)
    if shutil.disk_usage(args.output.parent).free < args.minimum_free_gib * 2**30:
        raise OSError('Insufficient free space in the new evaluation filesystem')
    jobs = []
    for selection in args.selections:
        for arm, dirname in EXPERIMENTS.items():
            experiment = (args.experiments_dir / dirname).resolve(strict=True)
            checkpoint = experiment / arm / ('checkpoint_best.pt' if selection == 'best' else 'checkpoint_latest.pt')
            if checkpoint.is_symlink() or not checkpoint.is_file():
                raise ValueError('Immutable regular checkpoint is required: ' + str(checkpoint))
            jobs.append(dict(name=arm + '_' + selection, arm=arm, selection=selection,
                experiment=str(experiment), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint)))
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items() if not key.startswith('worker_')}
    request = dict(format='four_arm_native_full128_readonly_evaluation_v1', settings=settings,
        jobs=jobs, inventory_sha256=sha(args.native_run / 'inventory/index.json'),
        prepared_cache_sha256=sha(args.prepared_cache), source=source_receipt(),
        selection_policy='predeclared original full129 BEST and saved latest; no hard-score selection',
        training_started=False, original_experiment_files_written=False,
        validation_cases=21, observations=2823, observed_P=135, U_per_case=128,
        annotation_aware_original_V1_upper=True, blind_recommendation_quality_verified=False)
    args.output.mkdir(parents=True, exist_ok=True)
    request_path = args.output / 'request.json'
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError('Existing evaluation request changed; previous evidence preserved')
    else:
        write_new(request_path, request)
    verify_request(args, request)
    from tools.run_local_cnn_experiment import exclusive
    with exclusive(args.output):
        regions = args.output / 'shared_regions'
        if args.region_cache_source is not None and not regions.exists():
            if any(path.is_symlink() for path in args.region_cache_source.rglob('*')):
                raise ValueError('Region reuse source must contain only ordinary files/directories')
            shutil.copytree(args.region_cache_source, regions, copy_function=shutil.copy2)
            for path in args.region_cache_source.rglob('*'):
                if path.is_file() and sha(path) != sha(regions / path.relative_to(args.region_cache_source)):
                    raise ValueError('Independent evaluation region copy checksum differs')
        assignments = {gpu: [] for gpu in args.gpus}
        # Round-robin only assigns jobs to the user's three named GPUs. A slot
        # runs one child at a time and retains its entire original model/cohort.
        for position, job in enumerate(jobs):
            assignments[args.gpus[position % len(args.gpus)]].append(job)

        def slot(gpu, assigned):
            completed = []
            for job in assigned:
                destination = args.output / job['name']; destination.mkdir(exist_ok=True)
                report_path = destination / 'report.json'
                if report_path.exists():
                    seal = json.loads((destination / 'complete.json').read_text())
                    if (seal['request_sha256'] != canonical_hash(request)
                            or seal['checkpoint_sha256'] != job['checkpoint_sha256']
                            or seal['report_sha256'] != sha(report_path)):
                        raise ValueError('Existing completed report identity differs')
                    completed.append(job['name']); continue
                command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                           *sys.argv[1:], '--worker-job', job['name'], '--worker-gpu', str(gpu)]
                print(f"START {job['name']} GPU{gpu}", flush=True)
                with (destination / 'run.log').open('ab') as log:
                    process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                    code = process.wait()
                if code != 0:
                    raise RuntimeError(f"{job['name']} returned{code}; preserved log {destination/'run.log'}; no process signaled")
                report = json.loads(report_path.read_text())
                print(f"COMPLETE {job['name']} GPU{gpu} | {json.dumps(report['metrics'])}", flush=True)
                completed.append(job['name'])
            return completed

        failures = []
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            pending = {pool.submit(slot, gpu, assigned): gpu for gpu, assigned in assignments.items()}
            for future in as_completed(pending):
                try:
                    future.result()
                except Exception as error:
                    failures.append(dict(gpu=pending[future], exception=type(error).__name__, error=str(error)))
                    print('FAILED SLOT | ' + json.dumps(failures[-1]), flush=True)
        if failures:
            write_new(args.output / ('failure_' + uuid.uuid4().hex + '.json'), failures)
            raise RuntimeError('Evaluation incomplete; all existing outputs preserved; no child or session signaled')
        reports = {job['name']: json.loads((args.output / job['name'] / 'report.json').read_text()) for job in jobs}
        if len({report['cohort']['cohort_sha256'] for report in reports.values()}) != 1:
            raise ValueError('Models were not scored on exactly the same native hard cohort')
        verify_request(args, request)
        summary = dict(format=request['format'], complete=True, full_evaluation=True,
            training_started=False, optimizer_updates=0, original_files_preserved=True,
            benchmark_cohort='same21 validation / P135 / U2688 / fixed external training donor',
            annotation_aware=True, blind_recommendation_quality_verified=False,
            jobs={name: dict(metrics=report['metrics'], denominators=report['denominators'],
                checkpoint=report['checkpoint'], calibration=report['calibration'],
                report=str(args.output / name / 'report.json')) for name, report in reports.items()})
        write_new(args.output / 'summary.json', summary)
        print(f"ALL {len(jobs)} EVALUATIONS COMPLETE: " + str(args.output / 'summary.json'), flush=True)


if __name__ == '__main__':
    main(parse())
