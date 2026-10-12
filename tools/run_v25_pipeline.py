"""Sequential complete A/B/C/D waves within a genuine single-GPU PBS lease."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def atomic_status(path,value):
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temporary.open('x',encoding='utf8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
    os.replace(temporary,path)


def actor(pid):
    import psutil
    process=psutil.Process(pid)
    return dict(child_pid=pid,child_create_time=process.create_time(),child_command=process.cmdline(),UID=process.uids().real)


def same_actor(value):
    import psutil
    try:process=psutil.Process(value['child_pid'])
    except psutil.NoSuchProcess:return False
    if process.status()==psutil.STATUS_ZOMBIE:return False
    return (process.uids().real==value['UID'] and process.create_time()==value['child_create_time']
        and process.cmdline()==value['child_command'])


def validate_preparation(group,input_cache):
    prepared=json.loads((group/'A/preparation.json').read_text())
    requested=json.loads((group/'A/request.json').read_text())
    index_path=input_cache/'local/index.json'
    cache_index=json.loads(index_path.read_text())
    request_contents={key:value for key,value in requested.items() if key!='request_sha256'}
    request_sha=hashlib.sha256(json.dumps(request_contents,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (prepared['request_sha256']!=requested['request_sha256'] or requested['request_sha256']!=request_sha
            or prepared['local_index_sha256']!=hashlib.sha256(index_path.read_bytes()).hexdigest()
            or cache_index.get('complete') is not True
            or len(cache_index.get('records',[]))!=14102 or cache_index.get('debug') is not False):
        raise ValueError('Complete source-bound full14102 preparation proof required')
    return prepared


def cache_memory_admission(input_cache):
    """Charge actual tmpfs allocation to the PBS memory budget before CUDA."""
    if not input_cache.is_dir() or any(path.is_symlink() for path in (input_cache,*input_cache.parents)):
        raise ValueError('Fresh canonical cache requires an owned directory with no symlink ancestors')
    allocated=0
    for path in input_cache.rglob('*'):
        if path.is_symlink():raise ValueError('Fresh canonical cache cannot contain external symlinks')
        if path.is_file():
            value=path.stat()
            allocated+=getattr(value,'st_blocks',(value.st_size+511)//512)*512
    if allocated>32*2**30:
        raise MemoryError('Actual complete canonical/upper tmpfs exceeds explicit32GiB PBS reserve; no cap/fallback')
    return dict(format='v25_actual_canonical_tmpfs_memory_admission_v1',input_cache=str(input_cache),
        allocated_bytes=allocated,shared_cache_memory_reserve_bytes=32*2**30,
        native_process_and_children_RSS_bytes=96*2**30,PBS_memory_bytes=128*2**30,
        full_cache_preserved=True,debug=False)


def parse(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',type=Path,required=True)
    parser.add_argument('--allocation-request',type=Path,required=True)
    parser.add_argument('--storage-admission',type=Path,required=True)
    parser.add_argument('--prepare-launch',type=Path,required=True)
    parser.add_argument('--native-experiment',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--stunet-checkpoint',type=Path,required=True)
    parser.add_argument('--input-cache',type=Path,required=True)
    parser.add_argument('--source-bank',type=Path,required=True)
    parser.add_argument('--baseline-preprocessed',type=Path,required=True)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--basic-validation',type=Path,required=True)
    parser.add_argument('--evaluator-root',type=Path,required=True)
    return parser.parse_args(argv)


def main(argv=None):
    import psutil
    from hiercp_v1x.v25_server_resources import AllocationLease
    from hiercp_v1x.v25_native_storage import verify_admission
    from hiercp_v1x.u_bridge_experiment import lock
    args=parse(argv)
    group=args.group.resolve(strict=True)
    allocation=json.loads(args.allocation_request.read_text())
    psutil.Process().cpu_affinity(allocation['CPU_affinity'])
    lease=AllocationLease(args.allocation_request);lease.verify(force=True)
    storage=verify_admission(args.storage_admission,full_hash=True)
    if storage['group_root']!=str(group) or any(storage['arm_roots'][arm]!=str(group/arm) for arm in 'ABCD'):
        raise ValueError('Pipeline must own all four exact storage-reserved arm roots')
    python='/home/aicompetition06/.conda/envs/nnunet/bin/python3.10'
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    active=None;current=None
    def publish(arm,stage,state,command=None,child=None,log=None,error=None,update_group=True):
        output=group/arm
        declarations=[dict(name=name,command=stage_command(arm,mode)) for name,mode in
            (('prepare_inputs','prepare'),('calibrate','calibrate'),('train','train'),('scoring','score-cp'))]
        declarations.append(dict(name='native',command=native_command(arm)))
        if command is not None and not any(item['command']==command for item in declarations):
            declarations.append(dict(name=stage,command=command))
        request=dict(GPU=allocation['gpu'],arm=arm,experiment_version='v2.5',production_output=str(output),
            RAM_GiB=96 if stage=='native' else 64,epochs=40,updates=680,stages=declarations)
        value=dict(status=state,stage=stage,arm=arm,output=str(output),GPU=allocation['gpu'],request=request,
            updated_at=time.time(),controller_pid=os.getpid(),allocation_request=str(args.allocation_request),
            storage_admission=str(args.storage_admission),stage_log=str(log) if log else None,error=error)
        if child:value.update(child)
        if update_group:atomic_status(group/'status.json',value)
        monitor=output/'monitor';monitor.mkdir(exist_ok=True)
        atomic_status(monitor/'status.json',value)
        return value
    def run_stage(arm,stage,command):
        nonlocal current
        lease.verify(force=True)
        output=group/arm;monitor=output/'monitor';monitor.mkdir(exist_ok=True)
        log=monitor/(stage+'_'+str(time.time_ns())+'.log')
        with log.open('xb') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        current=publish(arm,stage,'RUNNING',command,actor(process.pid),log)
        while process.poll() is None:
            time.sleep(10)
        code=process.returncode
        if code:
            publish(arm,stage,'FAILED',command,
                {key:current[key] for key in ('child_pid','child_create_time','child_command','UID')},
                log=log,error=f'Exit{code}; original log preserved: {log}')
            raise RuntimeError(f'V25 arm{arm} {stage} failed({code}); {log}')
        publish(arm,stage,'STAGE_COMPLETE',command,
            {key:current[key] for key in ('child_pid','child_create_time','child_command','UID')},log=log)
    def stage_command(arm,mode):
        return [python,'-B','-u',str(ROOT/'tools/run_v25.py'),'--mode',mode,'--arm',arm,
            '--gpu',str(allocation['gpu']),'--native-experiment',str(args.native_experiment),
            '--inventory',str(args.inventory),'--stunet-checkpoint',str(args.stunet_checkpoint),
            '--input-cache',str(args.input_cache),'--output',str(group/arm),
            '--cpu-affinity',','.join(map(str,allocation['CPU_affinity'])),
            '--allocation-request',str(args.allocation_request),'--storage-admission',str(args.storage_admission),
            '--source-bank',str(args.source_bank),'--baseline-preprocessed',str(args.baseline_preprocessed)]
    def native_command(arm):
        return [python,'-B','-u',str(ROOT/'tools/run_v25_downstream.py'),'--arm',arm,
            '--gpu',str(allocation['gpu']),'--score-overlay',str(group/arm/'cp_bank/score_overlay.json'),
            '--output',str(group/arm),'--storage-admission',str(args.storage_admission),
            '--project',str(args.project),'--basic-validation',str(args.basic_validation),
            '--evaluator-root',str(args.evaluator_root),
            '--cpu-affinity',','.join(map(str,allocation['CPU_affinity'])),
            '--allocation-request',str(args.allocation_request)]
    with lock(group/'.pipeline.lock'):
        try:
            for queued in 'BCD':
                if not (group/queued/'monitor/status.json').exists():
                    (group/queued).mkdir(exist_ok=True)
                    publish(queued,'queued','QUEUED',update_group=False)
            # The separately launched full preparation is adopted, never duplicated.
            launch=json.loads(args.prepare_launch.read_text())
            if not (group/'A/preparation.json').is_file():
                initial=json.loads((group/'A/monitor/status.json').read_text())
                if initial.get('child_pid')!=launch['PID'] or initial.get('child_command')!=launch['command']:
                    raise ValueError('Exact originally launched full preparation ownership required')
                current=publish('A','prepare_inputs','PREPARING',launch['command'],
                    {key:initial[key] for key in ('child_pid','child_create_time','child_command','UID')},Path(launch['log']))
                while same_actor(current):
                    lease.verify();time.sleep(10)
            if not (group/'A/preparation.json').is_file():
                raise RuntimeError('Original full preparation exited without complete receipt; original log preserved')
            validate_preparation(group,args.input_cache)
            atomic_status(group/'cache_memory_admission.json',cache_memory_admission(args.input_cache))
            for arm in 'ABCD':
                active=arm;(group/arm).mkdir(exist_ok=True)
                if (group/arm/'downstream/completion.json').is_file():
                    # Completion is a verified contract, never inferred from a filename alone.
                    from hiercp_v1x.v25_downstream import verify_completion
                    verify_completion(group/arm/'downstream/completion.json')
                    continue
                if arm!='A':run_stage(arm,'prepare_inputs',stage_command(arm,'prepare'))
                atomic_status(group/'cache_memory_admission.json',cache_memory_admission(args.input_cache))
                if not (group/arm/'calibration.json').is_file():
                    run_stage(arm,'calibrate',stage_command(arm,'calibrate'))
                run_stage(arm,'train',stage_command(arm,'train'))
                run_stage(arm,'scoring',stage_command(arm,'score-cp'))
                run_stage(arm,'native',native_command(arm))
                from hiercp_v1x.v25_downstream import verify_completion
                verify_completion(group/arm/'downstream/completion.json')
                publish(arm,'complete','COMPLETE')
            atomic_status(group/'completion.json',dict(status='COMPLETE',arms=list('ABCD'),
                complete_GNN_epochs_per_arm=40,complete_GNN_updates_per_arm=680,native_epochs_per_arm=250,
                evaluated_checkpoint='checkpoint_best.pth',evaluation_cases_per_arm=26,completed_at=time.time()))
        except Exception as error:
            arm=active or 'A'
            owned=current or {}
            child={key:owned[key] for key in ('child_pid','child_create_time','child_command','UID') if key in owned}
            publish(arm,owned.get('stage','pipeline'),'FAILED',command=owned.get('child_command'),
                child=child,log=owned.get('stage_log'),error=f'{type(error).__name__}: {error}')
            raise


if __name__=='__main__':main()
