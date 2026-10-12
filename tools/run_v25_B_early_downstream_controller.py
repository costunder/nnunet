"""B's authorized own full128 BEST -> full CP -> native250 -> BEST26.

No GNN training is restarted, no historical monitor is overwritten, and no
process is signalled. A successful child exit is followed by independently
validated saved completion bytes before the next stage can start.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def stage_command(args, seal):
    from tools.run_v25_parallel_pipeline import stage_command as original
    slot = seal['slot']
    declared = dict(arm='B', gpu=5, CPU_affinity=slot['CPU_affinity'],
        resource_request_path=slot['direct_proof']['path'])
    command = original(args, declared, 'score-cp')
    command[3] = str(ROOT/'tools/run_v25_B_early_downstream_execution.py')
    command[4:4] = ['--controller-seal', str(args.execution_manifest),
        '--early-request', seal['scientific_request']['path']]
    science = json.loads(Path(seal['scientific_request']['path']).read_bytes())
    command += ['--cp-output', science['cp_output']]
    return command


def native_command(args, seal):
    from tools.run_v25_parallel_pipeline import native_command as original
    slot = seal['slot']
    declared = dict(arm='B', gpu=5, CPU_affinity=slot['CPU_affinity'],
        resource_request_path=slot['direct_proof']['path'])
    command = original(args, declared)
    command[3] = str(ROOT/'tools/run_v25_B_early_downstream_execution.py')
    command[4:4] = ['--native', '--controller-seal', str(args.execution_manifest),
        '--early-request', seal['scientific_request']['path']]
    science = json.loads(Path(seal['scientific_request']['path']).read_bytes())
    command[command.index('--score-overlay')+1] = str(Path(science['cp_output'])/'score_overlay.json')
    return command


def run_chain(declarations, run_stage, verify_stage):
    if [item['name'] for item in declarations] != ['scoring', 'native']:
        raise ValueError('B-only downstream starts with full CP then native, never train')
    proofs = {}
    for item in declarations:
        run_stage(item)
        proofs[item['name']] = verify_stage(item['name'])
    return proofs


def verify_stage_completion(stage, output, cp_output):
    if stage not in ('scoring', 'native'):
        raise ValueError('Only exact CP/native saved completion can be admitted')
    # A fresh CPU verifier independently reads the overlay/native completion
    # with the same authorized BEST admission, without creating CUDA state.
    program = (
        'import json,sys;from hiercp_v1x.v25_authorized_early_downstream import '
        'install_from_environment,verify_stage_completion;install_from_environment();'
        'value=verify_stage_completion(sys.argv[1],sys.argv[2],sys.argv[3]);'
        'print(json.dumps(dict(actual_saved_completion_verified=True,stage=sys.argv[1])))')
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ''
    text = subprocess.check_output([sys.executable, '-B', '-c', program, stage,
        str(output), str(cp_output)], cwd=ROOT, env=env, text=True, timeout=600)
    value = json.loads(text)
    if value != dict(actual_saved_completion_verified=True, stage=stage):
        raise RuntimeError('Actual saved B-only full downstream stage completion was not verified')
    return value


def validate_inputs(args, seal, science):
    from hiercp_v1x import v25_B_early_downstream_resources as resources
    if (seal['group_root'] != str(args.group) or seal['private_code'] != str(ROOT)
            or science.get('source_gnn_output') != str(args.group/'B')):
        raise ValueError('Exact original own B GNN namespace and fresh sealed code required')
    cp_output = Path(science['cp_output'])
    transition = Path(seal['authorization_request']['path']).parent
    if (not cp_output.is_absolute() or '..' in cp_output.parts
            or not cp_output.is_relative_to(transition) or cp_output.name != 'cp_bank'):
        raise ValueError('Fresh CP score namespace inside the authorized B transition required')
    if resources.immutable_digest(args.storage_admission,
            seal['preserved_files'].get(str(args.storage_admission))) is None:
        raise ValueError('Original complete storage proof required')
    from hiercp_v1x.v25_native_storage import verify_admission
    if verify_admission(args.storage_admission, full_hash=True)['group_root'] != str(args.group):
        raise ValueError('Original full250 native storage admission required')
    from hiercp_v1x.v25_independent_preparation import validate_shared_preparation, validate_arm_preparation
    validate_shared_preparation(args.group, args.input_cache)
    validate_arm_preparation(args.group, args.input_cache, 'B')
    # The unchanged scientific entry independently verifies exact bank,
    # baseline, source geometry, context, model and full128 BEST bytes.
    return cp_output


def main(argv=None):
    import psutil
    from tools.run_v25_parallel_pipeline import parse
    from tools.run_v25_pipeline import actor, atomic_status
    from hiercp_v1x import v25_B_early_downstream_resources as resources
    from hiercp_v1x.u_bridge_experiment import lock
    args = parse(argv); args.group = args.group.resolve(strict=True)
    deadline = time.monotonic()+120
    while not args.execution_manifest.is_file():
        if time.monotonic() > deadline:
            raise RuntimeError('Exact B-only registration was not published within120seconds')
        time.sleep(1)
    os.environ[resources.ENV_PATH] = str(args.execution_manifest)
    os.environ[resources.ENV_SHA] = resources.sha(args.execution_manifest)
    seal = resources.load_seal()
    if resources.direct._actor(psutil.Process()) != seal['controller_identity']:
        raise RuntimeError('Actual new B-only controller PID/birth/UID/argv differs')
    if not set(resources.CPUS) <= set(psutil.Process().cpu_affinity()):
        raise RuntimeError('Original B six-core CPU set is not allowed for this controller')
    psutil.Process().cpu_affinity(resources.CPUS)
    resources.install_from_environment()
    released = resources.verify_initial_release(seal)
    science = resources._guard_proof(seal['scientific_request'])
    cp_output = validate_inputs(args, seal, science)
    slot = seal['slot']; monitor_path = Path(slot['monitor_path']); monitor = monitor_path.parent
    monitor.mkdir(parents=True, exist_ok=True)
    if monitor_path.exists():
        raise RuntimeError('Fresh B-only monitor required; no prior status overwrite')
    guard = resources.direct.DirectAllocationLease(slot['direct_proof']['path'])
    guard.verify(force=True)
    with (monitor/'release_before_downstream.json').open('x') as stream:
        json.dump(released, stream, allow_nan=False)
    declarations = [dict(name='scoring', command=stage_command(args, seal)),
        dict(name='native', command=native_command(args, seal))]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    gnn = released['actual_B_GNN']; current = dict(stage='admission', identity=None, log=None)

    def publish(stage, status, identity=None, log=None, error=None):
        value = dict(status=status, stage=stage, arm='B', output=str(args.group/'B'), GPU=5,
            controller_pid=os.getpid(), updated_at=time.time(), error=error,
            stage_log=str(log) if log else None, allocation_request=slot['direct_proof']['path'],
            storage_admission=str(args.storage_admission), B_early_downstream_seal=str(args.execution_manifest),
            authorization_request=seal['authorization_request'], release_receipt=seal['release_receipt'],
            original_monitors_preserved=True, GPU0_preserved=True, processes_signaled_by_this_adapter=False,
            request=dict(GPU=5, arm='B', experiment_version='v2.5', production_output=str(args.group/'B'),
                RAM_GiB=96 if stage == 'native' else 64, GNN_planned_epochs=40,
                GNN_planned_updates=680, GNN_full_training=False,
                GNN_completed_epochs=gnn['completed_epochs'], GNN_actual_updates=gnn['updates'],
                GNN_phase=gnn['phase'], GNN_train_position=gnn['train_position'],
                CP_sources=642, CP_candidates_per_source=128, native_epochs=250,
                evaluated_checkpoint='checkpoint_best.pth', evaluation_cases=26, stages=declarations))
        if identity:
            value.update(identity)
        atomic_status(monitor_path, value)

    def run_stage(item):
        guard.verify(force=True)
        stage = item['name']; log = monitor/(stage+'_B_early_'+str(time.time_ns())+'.log')
        current.update(stage=stage, identity=None, log=log)
        with log.open('xb') as stream:
            child = subprocess.Popen(item['command'], cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        identity = actor(child.pid); current['identity'] = identity
        publish(stage, 'RUNNING', identity, log)
        while child.poll() is None:
            time.sleep(2)
            guard.verify()
        if child.returncode:
            raise RuntimeError(f'B {stage} failed({child.returncode}); original log preserved: {log}')

    def verify_stage(stage):
        proof = verify_stage_completion(stage, args.group/'B', cp_output)
        with (monitor/(stage+'_verified_completion.json')).open('x') as stream:
            json.dump(proof, stream, allow_nan=False)
        publish(stage, 'STAGE_COMPLETE', current['identity'], current['log'])
        return proof

    with lock(Path(seal['authorization_request']['path']).parent/'.B-only-downstream.lock'):
        try:
            proofs = run_chain(declarations, run_stage, verify_stage)
        except Exception as error:
            publish(current['stage'], 'FAILED', current['identity'], current['log'],
                f'{type(error).__name__}: {error}; no process signalled; original checkpoints/logs preserved')
            raise
        with (monitor/'completion.json').open('x') as stream:
            json.dump(dict(status='COMPLETE', selected_arms=['B'],
                GNN_full_training=False, actual_B_GNN=gnn, native_epochs=250,
                CP_sources=642, CP_positions=82176, evaluated_checkpoint='checkpoint_best.pth',
                evaluation_cases=26, stages=proofs, GPU0_preserved=True,
                processes_signaled_by_this_adapter=False), stream, allow_nan=False)
        publish('complete', 'COMPLETE')


if __name__ == '__main__':
    main()
