"""Resume only D's owned full40/680 -> own-BEST CP -> native250/BEST26 chain."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))


def stage_command(args,slot,mode):
    from tools.run_v25_parallel_pipeline import stage_command as original
    command=original(args,slot,mode)
    command[3]=str(ROOT/'tools/run_v25_d_execution.py')
    command[4:4]=['--continuation-seal',str(args.execution_manifest)]
    if mode=='train':command+=['--resume',str(args.group/'D/training/checkpoint_latest.pt')]
    return command


def native_command(args,slot):
    from tools.run_v25_parallel_pipeline import native_command as original
    command=original(args,slot);command[3]=str(ROOT/'tools/run_v25_d_execution.py')
    command[4:4]=['--native','--continuation-seal',str(args.execution_manifest)]
    return command


def main(argv=None):
    import psutil
    from tools.run_v25_parallel_pipeline import parse
    from tools.run_v25_pipeline import actor,atomic_status,same_actor
    from hiercp_v1x import v25_standalone_d_resources as resources
    from hiercp_v1x.u_bridge_experiment import lock
    args=parse(argv);args.group=args.group.resolve(strict=True)
    deadline=time.monotonic()+120
    while not args.execution_manifest.is_file():
        if time.monotonic()>deadline:raise RuntimeError('Standalone D registration not published within120seconds')
        time.sleep(1)
    os.environ[resources.ENV_PATH]=str(args.execution_manifest)
    os.environ[resources.ENV_SHA]=resources.sha(args.execution_manifest)
    seal=resources.load_seal()
    if seal['group_root']!=str(args.group) or seal['private_code']!=str(ROOT):
        raise ValueError('Standalone D controller must run the exact sealed private code/output')
    if resources.direct._actor(psutil.Process())!=seal['controller_identity']:
        raise RuntimeError('Actual standalone controller identity differs from launch receipt')
    proof=json.loads(Path(seal['direct_proof']['path']).read_bytes())
    psutil.Process().cpu_affinity(proof['aggregate_CPU_affinity'])
    resources.install_from_environment()
    lease=resources.direct.DirectAllocationLease(seal['direct_proof']['path']);lease.verify(force=True)
    from hiercp_v1x.v25_native_storage import verify_admission
    if verify_admission(args.storage_admission,full_hash=True)['group_root']!=str(args.group):
        raise ValueError('Original unchanged four-arm storage admission required')
    from hiercp_v1x.v25_independent_preparation import validate_shared_preparation,validate_arm_preparation
    validate_shared_preparation(args.group,args.input_cache);validate_arm_preparation(args.group,args.input_cache,'D')
    slot=dict(arm='D',gpu=4,CPU_affinity=proof['CPU_affinity'],resource_request_path=str(lease.path))
    monitor=args.group/'D/monitor';previous=json.loads((monitor/'status.json').read_bytes())
    if previous.get('status')!='FAILED' or previous.get('child_pid')!=seal['previous_D_actor']['pid'] or same_actor(previous):
        raise RuntimeError('Only the observed failed and actually released D actor may be continued')
    if resources.sha(seal['checkpoint']['path'])!=seal['checkpoint']['sha256']:
        raise ValueError('Exact latest D checkpoint changed before launch')
    # Frozen run_training verifies complete content/identity, optimizer, scaler,
    # scheduler, curriculum and RNG again before executing a resumed batch.
    env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    declarations=[dict(name=stage,command=stage_command(args,slot,mode))
        for stage,mode in (('train','train'),('scoring','score-cp'))]
    declarations.append(dict(name='native',command=native_command(args,slot)))
    def publish(stage,status,*,identity=None,log=None,error=None):
        value=dict(status=status,stage=stage,arm='D',output=str(args.group/'D'),GPU=4,
            controller_pid=os.getpid(),updated_at=time.time(),error=error,stage_log=str(log) if log else None,
            allocation_request=str(lease.path),storage_admission=str(args.storage_admission),
            standalone_D_continuation=str(args.execution_manifest),original_ABC_controller_preserved=True,
            request=dict(GPU=4,arm='D',experiment_version='v2.5',production_output=str(args.group/'D'),
                RAM_GiB=96 if stage=='native' else 64,epochs=40,updates=680,stages=declarations))
        if identity:value.update(identity)
        atomic_status(monitor/'status.json',value)
    with lock(args.group/'D/.standalone-continuation.lock'):
        saved=monitor/('status_before_D_continuation_'+str(time.time_ns())+'.json')
        with saved.open('xb') as stream:stream.write((monitor/'status.json').read_bytes())
        for item in declarations:
            stage=item['name'];log=monitor/(stage+'_D_continuation_'+str(time.time_ns())+'.log')
            lease.verify(force=True)
            with log.open('xb') as stream:
                child=subprocess.Popen(item['command'],cwd=ROOT,env=env,stdout=stream,
                    stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            identity=actor(child.pid);publish(stage,'RUNNING',identity=identity,log=log)
            while child.poll() is None:time.sleep(2)
            if child.returncode:
                publish(stage,'FAILED',identity=identity,log=log,error=f'Exit{child.returncode}; exact D checkpoint/log preserved: {log}')
                raise RuntimeError(f'D {stage} failed({child.returncode}); {log}')
            publish(stage,'STAGE_COMPLETE',identity=identity,log=log)
        from hiercp_v1x.v25_downstream import verify_completion
        completed=verify_completion(args.group/'D/downstream/completion.json')
        publish('complete','COMPLETE')
        with (args.group/('D_standalone_completion_'+str(time.time_ns())+'.json')).open('x') as stream:
            json.dump(dict(format='v25_exact_D_standalone_completion_v1',own_completion=completed,
                continuation_seal_sha256=os.environ[resources.ENV_SHA],original_ABC_controller_preserved=True,
                GNN_epochs=40,GNN_updates=680,native_epochs=250,evaluation_checkpoint='checkpoint_best.pth',
                evaluated_cases=26,completed_at=time.time()),stream,allow_nan=False)


if __name__=='__main__':main()
