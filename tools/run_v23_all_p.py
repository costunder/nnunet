"""Prepare, measure and run the complete v2.3 native all-P experiment.

The archived model, actual native inputs and old experiments are read only.
Each new experiment owns its CPU geometry, calibration and training directory.
Each invocation uses exactly one of the user's GPUs 1/5/6 independently.
No NCCL/DDP process group or GPU/CPU fallback is created.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
FILES = ('config/v23_all_p_native.json', 'hiercp_v1x/v23_data.py',
         'hiercp_v1x/v23_geometry.py', 'hiercp_v1x/v23_training.py',
         'hiercp_v1x/v23_targets.py', 'hiercp_v1x/v23_inputs.py',
         'hiercp_v1x/v23_runtime_upgrade.py', 'tools/run_v23_all_p.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('all', 'prepare', 'calibrate', 'train'), default='all')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/v23_gpu1_prefix.json')
    parser.add_argument('--native-experiment', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--prepared-cache', type=Path, required=True)
    parser.add_argument('--full-validation-upper-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker-rank', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--invocation', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if len(args.gpus) != 1 or args.gpus[0] not in (1,5,6):
        raise ValueError('Each independent v2.3 run requires exactly one physical GPU1,5 or6')
    if args.worker_rank is not None and not 0 <= args.worker_rank < len(args.gpus):
        raise ValueError('Worker rank is outside this exact GPU request')
    if args.resume and args.mode != 'train':
        raise ValueError('Resume is explicitly --mode train --resume; preparation/calibration are preserved')
    if args.invocation is not None and (len(args.invocation) != 32 or any(
            character not in '0123456789abcdef' for character in args.invocation)):
        raise ValueError('Exact fresh invocation UUID required')
    if args.worker_rank is not None and args.invocation is None:
        raise ValueError('Workers must belong to a parent-owned invocation')
    return args


def contract(args):
    config = read(args.config)
    if (config['version'] != 'v2.3' or config['seed'] != 42 or config['epochs'] != 40
            or config['v23_runtime']['debug'] is not False
            or config['v23_runtime']['hidden_subset'] is not False):
        raise ValueError('Complete explicitly non-DEBUG v2.3 contract required')
    runtime = config['v23_runtime']
    if (runtime.get('independent_single_gpu') is not True or runtime['data_parallel_gpus'] != args.gpus
            or runtime.get('target_selection_policy') not in ('native_prefix','hard_score_top','score_stratified_mix')):
        raise ValueError('Independent GPU and target policy must match this explicit config')
    source = {name: sha(ROOT / name) for name in FILES}
    config_path = args.config.resolve()
    if not config_path.is_relative_to(ROOT):
        raise ValueError('The selected variant config must belong to the bound source checkout')
    source[config_path.relative_to(ROOT).as_posix()] = sha(config_path)
    value = dict(format='v23_native_all_P_execution_request_v1', config=config,
        config_sha256=sha(args.config), source=source,
        native_experiment=str(args.native_experiment.resolve(strict=True)),
        inventory=str(args.inventory.resolve(strict=True)), inventory_sha256=sha(args.inventory),
        prepared_cache=str(args.prepared_cache.resolve(strict=True)),
        prepared_cache_sha256=sha(args.prepared_cache),
        full_validation_upper_cache=str(args.full_validation_upper_cache.resolve(strict=True)),
        full_validation_upper_cache_sha256=sha(args.full_validation_upper_cache / 'index.json'),
        initial_model=str(args.native_experiment / 'initial.pt'),
        initial_model_file_sha256=sha(args.native_experiment / 'initial.pt'),
        gpus=args.gpus, original_experiment_written=False, trained_checkpoint_used_as_initial=False)
    from hiercp_v1x.contracts import canonical_hash
    value['request_sha256'] = canonical_hash(value)
    return value


def admit_output(args):
    from hiercp_v1x.historical_evaluation import assert_new_destination
    output = args.output.resolve()
    assert_new_destination(output, [args.native_experiment, args.inventory,
        args.prepared_cache.parent, args.full_validation_upper_cache])
    request = contract(args)
    path = output / 'request.json'
    if path.exists():
        if path.is_symlink() or read(path) != request:
            raise ValueError('Existing v2.3 experiment has different source/data/settings; preserve it')
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Existing directory has no v2.3 ownership request; no overwrite')
        new_json(path, request)
    return request


def inputs(args, *, gpu=False):
    # Activating the byte-exact archive precedes imports that touch hiercp.
    from hiercp_v1x.comparison_native_upper_cache import _load_geometry_inputs
    from hiercp_v1x.v23_data import V23Population
    from hiercp_v1x.v23_geometry import V23UpperGeometryCache
    bundle, inventory, _, _, preserved = _load_geometry_inputs(
        args.native_experiment, 'native', args.inventory)
    population = V23Population(inventory, debug=False)
    runtime = read(args.config)['v23_runtime']
    geometry = V23UpperGeometryCache(bundle, population, args.output / 'upper_geometry',
        workers=runtime['workers'], rss_bytes=int(runtime['rss_gib_per_rank'] * 2**30),
        resident_bytes=int(runtime['geometry_resident_gib_per_rank'] * 2**30),
        memoize_cpu_geometry=runtime['memoize_cpu_geometry'],
        cache_stage_admission=runtime['cache_stage_admission'],
        full_validation_cache=args.full_validation_upper_cache)
    if any(sha(path) != checksum for path, checksum in preserved.items()):
        raise ValueError('Preserved native model, bank or source changed during admission')
    return bundle, population, geometry


def prepare(args, request):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    import psutil
    import shutil
    runtime = request['config']['v23_runtime']
    if len(psutil.Process().cpu_affinity()) < runtime['workers']:
        raise ValueError('Parallel CPU preparation exceeds actual CPU affinity')
    if shutil.disk_usage(args.output).free < runtime['minimum_free_disk_gib'] * 2**30:
        raise OSError('New checkpoint/geometry filesystem lacks the declared disk reserve')
    torch.set_num_threads(1)
    bundle, population, geometry = inputs(args)
    started = time.perf_counter()
    for count in (runtime['initial_u'], runtime['total_u']):
        print(f'v2.3 CPU GEOMETRY START | U{count} | all P527 train + P135 validation', flush=True)
        geometry.prepare(count)
        print(f'v2.3 CPU GEOMETRY COMPLETE | U{count}', flush=True)
    receipt = dict(request_sha256=request['request_sha256'], population=population.manifest(),
        geometry=geometry.receipt, seconds=time.perf_counter() - started,
        CPU_affinity=psutil.Process().cpu_affinity(), available_RAM_bytes=psutil.virtual_memory().available,
        RSS_bytes=psutil.Process().memory_info().rss, storage_free_bytes=shutil.disk_usage(args.output).free,
        training_started=False, optimizer_updates=0)
    path = args.output / 'preparation.json'
    if not path.exists():
        new_json(path, receipt)


def gpu_setup(args, request):
    from tools.local_cnn_device import select
    rank = args.worker_rank
    select(args.gpus[rank])
    import torch
    import psutil
    import random
    import numpy as np
    from hiercp_v1x.historical_evaluation import ResourceBudget
    runtime = request['config']['v23_runtime']
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one physical A6000 must be visible per rank')
    free, total = torch.cuda.mem_get_info()
    properties = torch.cuda.get_device_properties(0)
    if 'A6000' not in properties.name or not runtime['cuda_gib_per_gpu'] * 2**30 < min(free, total):
        raise ValueError('Actual GPU model/free VRAM differs from the measured46GB A6000 contract')
    if runtime['rss_gib_per_rank'] * 2**30 > psutil.virtual_memory().available:
        raise ValueError('Rank RAM budget exceeds available system RAM')
    torch.cuda.set_per_process_memory_fraction(runtime['cuda_gib_per_gpu'] * 2**30 / total)
    torch.set_num_threads(4)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    budget = ResourceBudget(int(runtime['cuda_gib_per_gpu'] * 2**30), int(runtime['rss_gib_per_rank'] * 2**30))
    bundle, population, geometry = inputs(args, gpu=True)
    from hiercp.tensor import configure_runtime
    configure_runtime(deterministic=bundle.config['runtime']['deterministic'],
        allow_tf32=bundle.config['runtime']['allow_tf32'],
        cudnn_benchmark=bundle.config['runtime']['cudnn_benchmark'])
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp_v1x.u_bridge_training import digest
    from hiercp_v1x.scope_probe_support import state_digest
    saved = torch.load(args.native_experiment / 'initial.pt', map_location='cpu', weights_only=False, mmap=True)
    if saved['contract_sha256'] != read(args.native_experiment / 'experiment.json')['sha256']:
        raise ValueError('Original shared random initialization ownership differs')
    if saved['model_sha256'] != state_digest(saved['model']):
        raise ValueError('Original recorded initialization tensor checksum differs')
    model = HierarchicalPyGPlacementModel(**bundle.config['model'])
    model.load_state_dict(saved['model'], strict=True)
    if sum(value.numel() for value in model.parameters()) != request['config']['model_parameters']:
        raise ValueError('Original full model parameter count differs')
    initial_sha = digest(model.state_dict())
    if initial_sha != digest(saved['model']):
        raise ValueError('v2.3 did not restore the byte-exact shared random initialization')
    del saved
    model.to('cuda')
    geometry.admit(runtime['initial_u']); geometry.admit(runtime['total_u'])
    from hiercp_v1x.transition_v1_data import NativeObservationDataset
    from hiercp_v1x.v23_inputs import V23InputProvider, V23InputMemoryCoordinator
    coordinator = V23InputMemoryCoordinator(budget.rss_bytes)
    providers = {partition: V23InputProvider(
        NativeObservationDataset(args.inventory, partition, False),
        workers=runtime['workers'], resident_bytes=int(runtime['resident_gib_per_rank'] * 2**30),
        rss_bytes=budget.rss_bytes, cache_index=args.prepared_cache,
        persistent_cpu_workers=runtime['persistent_cpu_workers'],
        cache_sampled_views=runtime['memoize_fixed_views'],
        fixed_validation_epoch=bundle.config['training']['fixed_validation_epoch'],
        memory_coordinator=coordinator)
        for partition in ('inner_train', 'inner_val')}
    config = copy.deepcopy(bundle.config)
    config['v23_runtime'] = copy.deepcopy(runtime)
    resources = dict(rank=rank, physical_GPU=args.gpus[rank], GPU=properties.name,
        total_VRAM_bytes=total, free_VRAM_bytes=free, CPU_affinity=psutil.Process().cpu_affinity(),
        available_RAM_bytes=psutil.virtual_memory().available, workers=runtime['workers'],
        model=config['model'], parameters=sum(value.numel() for value in model.parameters()),
        trainable_parameters=sum(value.numel() for value in model.parameters() if value.requires_grad),
        graph_config=config['graph'], precision_AMP=config['training']['amp'],
        seed=42, epochs=40, population=population.manifest(), initial_model_tensor_sha256=initial_sha,
        actual_native_inputs=True, original_initialization_restored=True, trained_weights_used=False)
    resources['initialization_phase'] = 'initial model restoration for input/calibration admission; continuation loads later'
    resources['production_continued_state_reported_by'] = 'training/execution_contract.json and training/handoff_receipt.json'
    path = args.output / ('resources_' + args.mode + '_rank' + str(rank) + '_' + args.invocation + '.json')
    if not path.exists():
        new_json(path, resources)
    from hiercp_v1x.v23_training import V23Scorer
    scorer = V23Scorer(model, providers, geometry,
        physical_candidate_batch=min(runtime['physical_candidate_batch_candidates']),
        checkpoint_local_chunks=runtime['checkpoint_local_chunks'], amp=config['training']['amp'], budget=budget,
        prefetch_cpu_chunks=runtime['prefetch_cpu_chunks'], pin_cpu_batches=runtime['pin_cpu_batches'])
    return model, scorer, population, config, budget


def worker(args, request):
    model, scorer, population, config, budget = gpu_setup(args, request)
    try:
        _run_worker(args, request, model, scorer, population, config, budget)
    finally:
        for provider in scorer.providers.values():
            provider.close()


def _run_worker(args, request, model, scorer, population, config, budget):
    from hiercp_v1x.v23_training import calibrate_training_batches, run_training
    runtime = request['config']['v23_runtime']
    if args.mode == 'calibrate':
        report = calibrate_training_batches(model, scorer, population, config,
            patient_batches=runtime['physical_patient_batch_candidates'],
            candidate_batches=runtime['physical_candidate_batch_candidates'], budget=budget, debug=False)
        report['physical_GPU'] = args.gpus[args.worker_rank]
        report['request_sha256'] = request['request_sha256']
        new_json(args.output / ('calibration_rank' + str(args.worker_rank) + '.json'), report)
        print(f"v2.3 FULL128 BACKWARD CALIBRATION COMPLETE | rank{args.worker_rank}", flush=True)
        return
    calibration = read(args.output / 'calibration.json')
    validate_single_gpu_calibration(calibration, request)
    config['v23_runtime']['batch_calibration'] = calibration
    scorer.physical_candidate_batch = calibration['selected_physical_candidate_batch']
    if args.resume:
        config['v23_runtime']['resume_checkpoint'] = str(args.output / 'training/checkpoint_latest.pt')
    from hiercp_v1x.v23_training import execution_identity
    identity = execution_identity(request)
    handoff_path = args.output / 'training/handoff_receipt.json'
    if handoff_path.exists():
        if not args.resume:
            raise ValueError('Explicit checkpoint continuation requires --mode train --resume')
        handoff = read(handoff_path)
        if (handoff['destination_request_sha256'] != request['request_sha256']
                or handoff['destination_identity'] != execution_identity(request, handoff['continuation'])):
            raise ValueError('Checkpoint handoff belongs to another singleton/source request')
        identity = handoff['destination_identity']
    report = run_training(model, scorer, population, config, output=args.output / 'training',
        physical_patient_batch=calibration['selected_physical_patient_batch'], workers=runtime['workers'],
        identity=identity,
        budget=budget, epochs=40, debug=False)
    if args.worker_rank == 0:
        final_inputs = scorer.geometry.finish()
        new_json(args.output / ('final_inputs_' + args.invocation + '.json'), final_inputs)
        path = args.output / ('completion_' + args.invocation + '.json')
        new_json(path, report)


def validate_single_gpu_calibration(calibration, request):
    if (calibration.get('world_size') != 1 or calibration.get('request_sha256') != request['request_sha256']
            or calibration.get('physical_GPUs') != request['gpus'] or calibration.get('debug') is not False
            or not calibration.get('measured_full_P_U128_backward')
            or not calibration.get('original_model_and_RNG_preserved')):
        raise ValueError('Actual per-GPU full128 calibration must belong to this independent request')
    reports = calibration['GPU_reports']
    if (len(reports) != 1 or reports[0]['physical_GPU'] != request['gpus'][0]
            or reports[0]['initial_state_sha256'] != calibration['initial_state_sha256']):
        raise ValueError('Single GPU calibration hardware/initialization provenance differs')


def child_command(args, mode, rank):
    result = [sys.executable, '-B', '-u', str(ROOT / 'tools/run_v23_all_p.py'),
        '--mode', mode, '--config', str(args.config), '--native-experiment', str(args.native_experiment),
        '--inventory', str(args.inventory), '--prepared-cache', str(args.prepared_cache),
        '--full-validation-upper-cache', str(args.full_validation_upper_cache),
        '--output', str(args.output), '--gpus', *map(str, args.gpus), '--worker-rank', str(rank),
        '--invocation', args.invocation]
    if args.resume:
        result.append('--resume')
    return result


def launch_children(args, mode):
    processes, readers = [], []
    log_paths = [args.output / (mode + '_rank' + str(rank) + '_' + args.invocation + '.log')
                 for rank in range(len(args.gpus))]
    launch_path = args.output / (mode + '_launch_' + args.invocation + '.json')
    for path in [*log_paths, launch_path]:
        if path.exists():
            raise FileExistsError('Owned previous invocation evidence preserved: ' + str(path))
    def forward(process, path, rank):
        with path.open('xb') as target:
            for line in process.stdout:
                target.write(line); target.flush()
                # Rank0 emits the common training progress; calibration logs
                # remain visible for every GPU and are preserved separately.
                if rank == 0 or mode == 'calibrate':
                    print(line.decode(errors='replace'), end='', flush=True)
    for rank in range(len(args.gpus)):
        command = child_command(args, mode, rank)
        log = log_paths[rank]
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        reader = threading.Thread(target=forward, args=(process, log, rank))
        reader.start(); processes.append(process); readers.append(reader)
    new_json(launch_path,
        dict(mode=mode, pids=[process.pid for process in processes],
             commands=[child_command(args, mode, rank) for rank in range(len(args.gpus))],
             started_unix=time.time(), physical_GPUs=args.gpus))
    codes = [process.wait() for process in processes]
    for reader in readers:
        reader.join()
    if any(codes):
        raise RuntimeError(f'v2.3 {mode} worker errors {codes}; exact rank logs preserved under {args.output}')


def common_calibration(args, request):
    reports = [read(args.output / ('calibration_rank' + str(rank) + '.json')) for rank in range(len(args.gpus))]
    if any(report['request_sha256'] != request['request_sha256'] for report in reports):
        raise ValueError('GPU calibrations belong to different v2.3 contracts')
    hashes = {report['initial_state_sha256'] for report in reports}
    if len(hashes) != 1 or any(not report['original_model_and_RNG_preserved'] for report in reports):
        raise ValueError('Calibration changed production initialization/RNG or ranks disagree')
    trials = []
    runtime = request['config']['v23_runtime']
    for patients in runtime['physical_patient_batch_candidates']:
        for candidates in runtime['physical_candidate_batch_candidates']:
            rows = [next(row for row in report['trials'] if row['physical_patient_batch'] == patients
                         and row['physical_candidate_batch'] == candidates) for report in reports]
            trials.append(dict(physical_patient_batch=patients, physical_candidate_batch=candidates,
                accepted=all(row['accepted'] for row in rows), GPU_trials=rows,
                patients_per_second=min(row.get('patients_per_second', 0.) for row in rows)))
    accepted = [row for row in trials if row['accepted']]
    if not accepted:
        raise MemoryError('No common full128 backward batch fits all allocated GPUs; no model/input reduction')
    selected = max(accepted, key=lambda row: row['patients_per_second'])
    report = dict(format=reports[0]['format'], trials=trials, GPU_reports=reports,
        selected_physical_patient_batch=selected['physical_patient_batch'],
        selected_physical_candidate_batch=selected['physical_candidate_batch'],
        initial_state_sha256=next(iter(hashes)), original_model_and_RNG_preserved=True,
        measured_full_P_U128_backward=True, world_size=len(args.gpus),
        effective_patient_batch=selected['physical_patient_batch'] * len(args.gpus),
        gradient_accumulation_steps=1, debug=False)
    report.update(request_sha256=request['request_sha256'], physical_GPUs=args.gpus,
                  independent_single_gpu=True, new_measurement=True)
    new_json(args.output / 'calibration.json', report)
    print('v2.3 COMMON MEASURED BATCH | patients/rank=' + str(selected['physical_patient_batch'])
          + ' | local candidates=' + str(selected['physical_candidate_batch'])
          + ' | effective patients=' + str(report['effective_patient_batch']), flush=True)


def main(args):
    if args.invocation is None:
        args.invocation = uuid.uuid4().hex
    request = admit_output(args)
    if args.worker_rank is not None:
        worker(args, request)
        return
    if args.mode in ('all', 'prepare'):
        prepare(args, request)
    if args.mode in ('all', 'calibrate'):
        launch_children(args, 'calibrate'); common_calibration(args, request)
    if args.mode in ('all', 'train'):
        if not (args.output / 'calibration.json').is_file():
            raise ValueError('Actual complete full128 backward calibration must precede production')
        launch_children(args, 'train')


if __name__ == '__main__':
    main(parse())
