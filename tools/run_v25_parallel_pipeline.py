"""Four independent full V25 chains on explicitly requested physical GPUs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ('A', 'B', 'C', 'D')
GPU_MAPPING = {'A': 6, 'B': 5, 'C': 1, 'D': 4}


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--execution-manifest', type=Path, required=True)
    parser.add_argument('--storage-admission', type=Path, required=True)
    parser.add_argument('--native-experiment', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--stunet-checkpoint', type=Path, required=True)
    parser.add_argument('--input-cache', type=Path, required=True)
    parser.add_argument('--source-bank', type=Path, required=True)
    parser.add_argument('--baseline-preprocessed', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--basic-validation', type=Path, required=True)
    parser.add_argument('--evaluator-root', type=Path, required=True)
    return parser.parse_args(argv)


def validate_manifest(manifest, group):
    if (manifest.get('format') != 'v25_four_GPU_independent_execution_v1'
            or manifest.get('group_root') != str(group) or manifest.get('debug') is not False
            or set(manifest.get('slots', {})) != set(ARMS)):
        raise ValueError('Exact full four-arm direct execution manifest required')
    occupied = set()
    cpus = set()
    for arm in ARMS:
        slot = manifest['slots'][arm]
        if slot.get('gpu') != GPU_MAPPING[arm] or slot.get('arm') != arm:
            raise ValueError('Authorized GPU6/5/1/4 curriculum mapping differs')
        if slot['gpu'] in occupied or len(set(slot['CPU_affinity'])) != 6 or cpus.intersection(slot['CPU_affinity']):
            raise ValueError('Four distinct GPUs and non-overlapping six-core CPU sets required')
        occupied.add(slot['gpu'])
        cpus.update(slot['CPU_affinity'])
        proof_path = Path(slot['resource_request_path'])
        if hashlib.sha256(proof_path.read_bytes()).hexdigest() != slot['resource_request_sha256']:
            raise ValueError('Immutable direct resource proof changed')
    return manifest


def stage_command(args, slot, mode, *, parallel_prepare=False):
    argv = ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
        str(ROOT/'tools/run_v25_execution.py')]
    if parallel_prepare:
        argv.append('--parallel-prepare')
    return argv + ['--mode', mode, '--arm', slot['arm'], '--gpu', str(slot['gpu']),
        '--native-experiment', str(args.native_experiment), '--inventory', str(args.inventory),
        '--stunet-checkpoint', str(args.stunet_checkpoint), '--input-cache', str(args.input_cache),
        '--output', str(args.group/slot['arm']), '--cpu-affinity', ','.join(map(str, slot['CPU_affinity'])),
        '--allocation-request', slot['resource_request_path'], '--storage-admission', str(args.storage_admission),
        '--source-bank', str(args.source_bank), '--baseline-preprocessed', str(args.baseline_preprocessed)]


def native_command(args, slot):
    return ['/home/aicompetition06/.conda/envs/nnunet/bin/python3.10', '-B', '-u',
        str(ROOT/'tools/run_v25_downstream.py'), '--arm', slot['arm'], '--gpu', str(slot['gpu']),
        '--score-overlay', str(args.group/slot['arm']/'cp_bank/score_overlay.json'),
        '--output', str(args.group/slot['arm']), '--storage-admission', str(args.storage_admission),
        '--project', str(args.project), '--basic-validation', str(args.basic_validation),
        '--evaluator-root', str(args.evaluator_root), '--cpu-affinity', ','.join(map(str, slot['CPU_affinity'])),
        '--allocation-request', slot['resource_request_path']]


def main(argv=None):
    import psutil
    from tools.run_v25_pipeline import actor, atomic_status, cache_memory_admission, validate_preparation
    from hiercp_v1x.u_bridge_experiment import lock
    args = parse(argv)
    args.group = args.group.resolve(strict=True)
    deadline = time.monotonic() + 120
    while not args.execution_manifest.is_file():
        if time.monotonic() > deadline:
            raise RuntimeError('Actual controller registration not published within120 seconds')
        time.sleep(1)
    manifest = validate_manifest(json.loads(args.execution_manifest.read_text()), args.group)
    from hiercp_v1x.v25_direct_resources import validate_resource_proof
    aggregate_cpus = sorted({cpu for slot in manifest['slots'].values() for cpu in slot['CPU_affinity']})
    psutil.Process().cpu_affinity(aggregate_cpus)
    from hiercp_v1x.v25_native_storage import verify_admission
    admission = verify_admission(args.storage_admission, full_hash=True)
    if admission['group_root'] != str(args.group):
        raise ValueError('Exact four-arm storage namespace required')
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        V25_PARALLEL_EXECUTION_MANIFEST=str(args.execution_manifest))
    def publish(slot, stage, state, command=None, child=None, log=None, error=None):
        arm = slot['arm']
        declarations = [dict(name=name, command=stage_command(args, slot, mode)) for name, mode in
            (('prepare_inputs', 'prepare'), ('calibrate', 'calibrate'), ('train', 'train'), ('scoring', 'score-cp'))]
        declarations.append(dict(name='native', command=native_command(args, slot)))
        if command is not None and not any(item['command'] == command for item in declarations):
            declarations.append(dict(name=stage, command=command))
        request = dict(GPU=slot['gpu'], arm=arm, experiment_version='v2.5', production_output=str(args.group/arm),
            RAM_GiB=96 if stage == 'native' else 64, epochs=40, updates=680, stages=declarations)
        value = dict(status=state, stage=stage, arm=arm, output=str(args.group/arm), GPU=slot['gpu'], request=request,
            updated_at=time.time(), controller_pid=os.getpid(), allocation_request=slot['resource_request_path'],
            storage_admission=str(args.storage_admission), stage_log=str(log) if log else None, error=error)
        if child:
            value.update(child)
        monitor = args.group/arm/'monitor'
        monitor.mkdir(exist_ok=True)
        atomic_status(monitor/'status.json', value)
        if arm == 'A':
            atomic_status(args.group/'status.json', value)
        return value
    def run_stage(slot, stage, command):
        monitor = args.group/slot['arm']/'monitor'
        log = monitor/(stage + '_parallel_' + str(time.time_ns()) + '.log')
        with log.open('xb') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        identity = actor(process.pid)
        publish(slot, stage, 'RUNNING', command, identity, log)
        while process.poll() is None:
            time.sleep(2)
        if process.returncode:
            publish(slot, stage, 'FAILED', command, identity, log,
                error=f'Exit{process.returncode}; exact child and original log preserved: {log}')
            raise RuntimeError(f'Arm{slot["arm"]} {stage} failed({process.returncode}); {log}')
        publish(slot, stage, 'STAGE_COMPLETE', command, identity, log)
    def chain(slot):
        try:
            if (args.group/slot['arm']/'downstream/completion.json').exists():
                from hiercp_v1x.v25_downstream import verify_completion
                verify_completion(args.group/slot['arm']/'downstream/completion.json')
                return
            if not (args.group/slot['arm']/'calibration.json').exists():
                run_stage(slot, 'calibrate', stage_command(args, slot, 'calibrate'))
            run_stage(slot, 'train', stage_command(args, slot, 'train'))
            run_stage(slot, 'scoring', stage_command(args, slot, 'score-cp'))
            run_stage(slot, 'native', native_command(args, slot))
            from hiercp_v1x.v25_downstream import verify_completion
            verify_completion(args.group/slot['arm']/'downstream/completion.json')
            publish(slot, 'complete', 'COMPLETE')
        except Exception as error:
            path = args.group/slot['arm']/'monitor/status.json'
            previous = json.loads(path.read_text())
            child = {key: previous[key] for key in ('child_pid', 'child_create_time', 'child_command', 'UID') if key in previous}
            publish(slot, previous.get('stage', 'pipeline'), 'FAILED', previous.get('child_command'), child,
                previous.get('stage_log'), f'{type(error).__name__}: {error}')
            raise
    with lock(args.group/'.pipeline.lock'):
        for slot in manifest['slots'].values():
            (args.group/slot['arm']).mkdir(exist_ok=True)
            publish(slot, 'prepare_inputs', 'WAITING_FOR_COMMON_PREPARATION')
        first = manifest['slots']['A']
        if not (args.group/'A/preparation.json').is_file():
            run_stage(first, 'prepare_inputs', stage_command(args, first, 'prepare', parallel_prepare=True))
        validate_preparation(args.group, args.input_cache)
        atomic_status(args.group/'cache_memory_admission.json', cache_memory_admission(args.input_cache))
        # Each arm admits the SAME complete canonical index, then only its own U prefix.
        for arm in 'BCD':
            slot = manifest['slots'][arm]
            if not (args.group/arm/'preparation.json').exists():
                run_stage(slot, 'prepare_inputs', stage_command(args, slot, 'prepare'))
        failures = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(chain, manifest['slots'][arm]): arm for arm in ARMS}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    failures.append(dict(arm=futures[future], error=f'{type(error).__name__}: {error}'))
                    atomic_status(args.group/'parallel_failures.json', dict(failures=failures, observed_at=time.time(),
                        other_arms_continue=True, signals_sent=False))
        if failures:
            raise RuntimeError('Independent V25 chain failures; unaffected arms completed: ' + json.dumps(failures))
        atomic_status(args.group/'completion.json', dict(status='COMPLETE', arms=list(ARMS),
            complete_GNN_epochs_per_arm=40, complete_GNN_updates_per_arm=680, native_epochs_per_arm=250,
            evaluated_checkpoint='checkpoint_best.pth', evaluation_cases_per_arm=26, completed_at=time.time(),
            GPU_mapping=GPU_MAPPING, independent_chains=True))


if __name__ == '__main__':
    main()
