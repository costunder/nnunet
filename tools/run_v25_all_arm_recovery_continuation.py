"""Four independent full continuations in a new exact controller namespace."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def stage_command(args, slot, mode):
    from tools.run_v25_parallel_pipeline import stage_command as original
    declared = dict(arm=slot['arm'], gpu=slot['gpu'], CPU_affinity=slot['CPU_affinity'],
        resource_request_path=slot['direct_proof']['path'])
    command = original(args, declared, mode)
    command[3] = str(ROOT/'tools/run_v25_all_arm_recovery_execution.py')
    command[4:4] = ['--continuation-seal', str(args.execution_manifest)]
    if mode == 'train':
        command += ['--resume', slot['checkpoint']['path']]
    return command


def native_command(args, slot):
    from tools.run_v25_parallel_pipeline import native_command as original
    declared = dict(arm=slot['arm'], gpu=slot['gpu'], CPU_affinity=slot['CPU_affinity'],
        resource_request_path=slot['direct_proof']['path'])
    command = original(args, declared)
    command[3] = str(ROOT/'tools/run_v25_all_arm_recovery_execution.py')
    command[4:4] = ['--native', '--continuation-seal', str(args.execution_manifest)]
    return command


def run_chains(chain, on_failure):
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(chain, arm): arm for arm in ('A', 'B', 'C', 'D')}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(dict(arm=futures[future], error=f'{type(error).__name__}: {error}'))
                on_failure(list(failures))
    return failures


def main(argv=None):
    import psutil
    from tools.run_v25_parallel_pipeline import parse
    from tools.run_v25_pipeline import actor, atomic_status, same_actor
    from tools.run_v25_u128_reuse_continuation import verify_arm_completion
    from hiercp_v1x import v25_all_arm_recovery_resources as resources
    from hiercp_v1x.u_bridge_experiment import lock
    args = parse(argv); args.group = args.group.resolve(strict=True)
    deadline = time.monotonic()+120
    while not args.execution_manifest.is_file():
        if time.monotonic() > deadline:
            raise RuntimeError('Exact all-four registration was not published within120seconds')
        time.sleep(1)
    os.environ[resources.ENV_PATH] = str(args.execution_manifest)
    os.environ[resources.ENV_SHA] = resources.sha(args.execution_manifest)
    seal = resources.load_seal()
    if seal['group_root'] != str(args.group) or seal['private_code'] != str(ROOT):
        raise ValueError('Exact sealed all-four private code/output required')
    if resources.direct._actor(psutil.Process()) != seal['controller_identity']:
        raise RuntimeError('Actual new all-four controller PID/birth/UID/argv differs')
    first = resources._guard_proof(seal['slots']['A']['direct_proof'])
    psutil.Process().cpu_affinity(first['aggregate_CPU_affinity'])
    resources.install_from_environment()
    from hiercp_v1x.v25_native_storage import verify_admission
    if verify_admission(args.storage_admission, full_hash=True)['group_root'] != str(args.group):
        raise ValueError('Original complete four-arm storage namespace required')
    from hiercp_v1x.v25_independent_preparation import validate_shared_preparation, validate_arm_preparation
    validate_shared_preparation(args.group, args.input_cache)
    released = resources.verify_initial_release(seal)
    for arm in seal['selected_arms']:
        slot = seal['slots'][arm]
        validate_arm_preparation(args.group, args.input_cache, arm)
        lease = resources.direct.DirectAllocationLease(slot['direct_proof']['path']); lease.verify(force=True)
        previous = resources._guard_proof(slot['failed_monitor'])
        if previous.get('status') != 'FAILED' or same_actor(previous):
            raise RuntimeError('Exact naturally failed and released own stage required')
        if resources.sha(slot['checkpoint']['path']) != slot['checkpoint']['sha256']:
            raise ValueError('Exact captured own latest checkpoint changed before resume')
        if (args.group/arm/'training/STOP_AFTER_BATCH').exists():
            raise RuntimeError('An existing own pause marker requires explicit recovery; no bypass')
        monitor = Path(slot['monitor_path']).parent; monitor.mkdir(exist_ok=True)
        if Path(slot['monitor_path']).exists():
            raise RuntimeError('Fresh per-arm recovery monitor required; no existing monitor overwrite')
        with (monitor/'original_FAILED_monitor.json').open('xb') as stream:
            stream.write(Path(slot['failed_monitor']['path']).read_bytes())
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (args.group/('ALL_release_before_launch_'+str(time.time_ns())+'.json')).open('x') as stream:
        json.dump(released, stream, allow_nan=False)

    def chain(arm):
        slot = seal['slots'][arm]; monitor_path = Path(slot['monitor_path']); monitor = monitor_path.parent
        declarations = [dict(name=name, command=stage_command(args, slot, mode))
            for name, mode in (('train', 'train'), ('scoring', 'score-cp'))]
        declarations.append(dict(name='native', command=native_command(args, slot)))

        def publish(stage, status, identity=None, log=None, error=None):
            value = dict(status=status, stage=stage, arm=arm, output=str(args.group/arm),
                GPU=slot['gpu'], controller_pid=os.getpid(), updated_at=time.time(), error=error,
                stage_log=str(log) if log else None, allocation_request=slot['direct_proof']['path'],
                storage_admission=str(args.storage_admission), all_arm_recovery=str(args.execution_manifest),
                original_canonical_monitor_preserved=True, GPU0_preserved=True,
                request=dict(GPU=slot['gpu'], arm=arm, experiment_version='v2.5', production_output=str(args.group/arm),
                    RAM_GiB=96 if stage == 'native' else 64, epochs=40, updates=680, stages=declarations))
            if identity:
                value.update(identity)
            atomic_status(monitor_path, value)

        for item in declarations:
            lease = resources.direct.DirectAllocationLease(slot['direct_proof']['path']); lease.verify(force=True)
            stage = item['name']; log = monitor/(stage+'_ALL_continuation_'+str(time.time_ns())+'.log')
            with log.open('xb') as stream:
                child = subprocess.Popen(item['command'], cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            identity = actor(child.pid); publish(stage, 'RUNNING', identity, log)
            while child.poll() is None:
                time.sleep(2)
            if child.returncode:
                publish(stage, 'FAILED', identity, log,
                    f'Exit{child.returncode}; exact own checkpoint/log preserved: {log}')
                raise RuntimeError(f'{arm} {stage} failed({child.returncode}); {log}')
            publish(stage, 'STAGE_COMPLETE', identity, log)
        verify_arm_completion(args.group, arm)
        publish('complete', 'COMPLETE')

    with lock(args.group/'.all-arm-terminal-recovery.lock'):
        failures = run_chains(chain, lambda failures: atomic_status(
            args.group/('ALL_failures_'+str(time.time_ns())+'.json'),
            dict(failures=failures, other_arms_continue=True, GPU0_preserved=True, signals_sent=False)))
        if failures:
            raise RuntimeError('Independent all-four continuation failure: '+json.dumps(failures))
        atomic_status(args.group/('ALL_completion_'+str(time.time_ns())+'.json'),
            dict(status='COMPLETE', selected_arms=['A', 'B', 'C', 'D'],
                complete_GNN_epochs_per_arm=40, complete_GNN_updates_per_arm=680,
                native_epochs_per_arm=250, evaluated_checkpoint='checkpoint_best.pth',
                evaluation_cases_per_arm=26, GPU0_preserved=True, signals_sent=False))


if __name__ == '__main__':
    main()
