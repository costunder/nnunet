"""One new GPU4 pipeline; validate the exact free GPU before each CUDA stage."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def publish(path,value):
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temporary.open('x',encoding='utf8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def assigned_GPU(request):
    value=subprocess.check_output(['nvidia-smi','--id=4',
        '--query-gpu=uuid,name,memory.free,memory.used','--format=csv,noheader,nounits'],text=True).strip()
    parts=[item.strip() for item in value.split(',')]
    if parts[0]!=request['GPU_uuid'] or 'A6000' not in parts[1] or float(parts[2])*2**20<=40*2**30 or float(parts[3])>=10:
        raise RuntimeError('Assigned GPU4 UUID/free40GiB admission failed: '+value)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    if any(row.strip().endswith(request['GPU_uuid']) for row in apps.splitlines()):
        raise RuntimeError('GPU4 has another compute application; stage was not started')
    return value


def main():
    import psutil
    request_path=Path(sys.argv[1]).resolve(strict=True);request=json.loads(request_path.read_text())
    affinity=request['CPU_affinity'];root=request_path.parent
    if (request['GPU']!=4 or request['epochs']!=40 or request['physical_patient_batch']!=4
            or request['candidate_chunk']!=64 or request['RAM_GiB']!=64
            or len(affinity)!=4 or len(set(affinity))!=4 or not set(affinity)<=os.sched_getaffinity(0)):
        raise ValueError('Exact GPU4/fourCPU/full40/B4/chunk64/64GiB pipeline required')
    process=psutil.Process();process.cpu_affinity(affinity);process.nice(10)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',V24_ARM_RSS_GIB='64')
    status=dict(request=request,worker_pid=os.getpid(),worker_create_time=process.create_time(),
        actual_CPU_affinity=process.cpu_affinity(),nice=process.nice(),status='PREPARING',stage=None,
        stages_completed=[],started=time.time())
    publish(root/'status.json',status)
    try:
        for stage in request['stages']:
            if (root/'STOP_BEFORE_NEXT_STAGE').exists():
                status['status']='PAUSED_BETWEEN_STAGES';publish(root/'status.json',status);return
            if stage['name'] in ('calibrate','train'):
                status['GPU_admission']=assigned_GPU(request)
            status.update(status='RUNNING',stage=stage['name'],stage_started=time.time())
            print(json.dumps(dict(stage=stage['name'],command=stage['command'],CPU_affinity=affinity,
                RAM_GiB=64,nice=process.nice(),time=time.time())),flush=True)
            with (root/(stage['name']+'.log')).open('x',encoding='utf8') as log:
                child=subprocess.Popen(stage['command'],cwd=request['code'],env=env,stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                status.update(child_pid=child.pid,child_create_time=psutil.Process(child.pid).create_time())
                publish(root/'status.json',status)
                for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
                result=child.wait()
            if result:raise RuntimeError('Stage '+stage['name']+' failed with code '+str(result)+'; logs/results preserved')
            status['stages_completed'].append(stage['name']);publish(root/'status.json',status)
        status.update(status='COMPLETE',completed=time.time());publish(root/'status.json',status)
    except Exception as error:
        status.update(status='FAILED',error=type(error).__name__+': '+str(error),failed=time.time())
        publish(root/'status.json',status)
        raise


if __name__=='__main__':main()
