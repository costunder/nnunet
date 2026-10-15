"""Exact owned allocation for six full Basic CP continuations on one parent GPU.

No scheduler, driver, session or foreign process is modified. A full GPU and
six separately allocated MIG instances have distinct, explicit contracts.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import uuid

import psutil

GIB=2**30
UID=41833
FORMAT='basic_cp_single_parent_allocation_v1'
DIRECT_MODE='direct_idle_MIG_same_parent'
ARMS=('multi5','multi7','multi10','candidates512','candidates1024','candidates2048')
EPOCHS={'multi5':130,'multi7':181,'multi10':115,'candidates512':111,'candidates1024':80,'candidates2048':46}
_HASHED={}


def require(value,message):
    if not value:raise ValueError(message)


def signature(path):
    s=Path(path).stat()
    return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_uid,s.st_mode)


def exact(path,*,owner=False):
    p=Path(path)
    require(p.is_absolute()and p.resolve(strict=True)==p and not any(x.is_symlink()for x in(p,*p.parents)),
        'Absolute nonsymlink evidence required: '+str(p))
    require(p.is_file()and(not owner or p.stat().st_uid==UID),'Regular readable evidence/owner differs: '+str(p))
    return p


def digest(path):
    p=exact(path);before=signature(p);key=(str(p),before)
    if key not in _HASHED:
        checksum=hashlib.sha256()
        with p.open('rb')as stream:
            for block in iter(lambda:stream.read(8*2**20),b''):checksum.update(block)
        require(signature(p)==before,'Evidence changed during hashing: '+str(p))
        _HASHED[key]=checksum.hexdigest()
    return _HASHED[key]


def pin(reference,*,owner=False):
    require(isinstance(reference,dict)and set(reference)=={'path','sha256'}
        and isinstance(reference['sha256'],str)and re.fullmatch('[0-9a-f]{64}',reference['sha256']),
        'Explicit path/SHA256 evidence reference required')
    p=exact(reference['path'],owner=owner)
    require(digest(p)==reference['sha256'],'Pinned evidence SHA changed: '+str(p))
    return p


def identity(process):
    return dict(pid=process.pid,create_time=process.create_time(),UID=process.uids().real,command=process.cmdline())


def exact_actor(expected):
    require(isinstance(expected,dict)and set(expected)=={'pid','create_time','UID','command'}and expected['UID']==UID,
        'Exact own PID/birth/UID/argv required')
    p=psutil.Process(expected['pid'])
    require(identity(p)==expected and p.status()not in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD),
        'Original actor identity changed or actor released')
    return p


def live_actor(expected):
    """PID reuse is release of the old actor, never permission for its replacement."""
    try:process=psutil.Process(expected['pid'])
    except psutil.NoSuchProcess:
        require(_same_live_process(expected['pid'],expected['create_time'],[],'registered actor')is None,
            'Live registered actor observation failed')
        return None
    observation=process_snapshot(process,expected=expected,resources=False)
    return None if observation is None else psutil.Process(expected['pid'])


def _same_live_process(pid,birth,released,context):
    """Read fresh kernel state; a stale Process object is not release evidence."""
    absent=[]
    for attempt in range(2):
        try:
            process=psutil.Process(pid);actual_birth=process.create_time();state=process.status()
        except psutil.ZombieProcess:
            # On Linux this exception is itself an observed terminal state.
            released.append(dict(pid=pid,create_time=birth,state=psutil.STATUS_ZOMBIE,context=context,
                evidence='fresh psutil terminal-state exception'))
            return None
        except psutil.NoSuchProcess:
            absent.append('fresh Process/status observation '+str(attempt+1));continue
        if birth is not None and actual_birth!=birth:
            released.append(dict(pid=pid,create_time=birth,replacement_create_time=actual_birth,
                state='PID_reused',context=context,evidence='fresh kernel start time changed'))
            return None
        if state in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD):
            released.append(dict(pid=pid,create_time=actual_birth,state=state,context=context,
                evidence='fresh kernel process status'))
            return None
        return process
    released.append(dict(pid=pid,create_time=birth,state='reaped',context=context,
        evidence=absent))
    return None


def process_snapshot(process,*,expected=None,validator=None,released=None,context='',resources=True):
    """Validate one known descendant while bracketing reads with kernel lifetime checks.

    A worker exiting during cmdline/cgroup/RSS reads may expose empty argv or
    the root cgroup. Only a fresh terminal/reaped/PID-reuse observation permits
    exclusion. Live UID/CPU/cgroup/identity mismatches remain fatal.
    """
    if released is None:released=[]
    try:birth=expected['create_time']if expected is not None else process.create_time()
    except psutil.NoSuchProcess:
        require(_same_live_process(process.pid,None,released,context)is None,
            'Live descendant start-time observation failed')
        return None
    current=_same_live_process(process.pid,birth,released,context)
    if current is None:return None
    if birth is None:birth=current.create_time()
    try:
        value=dict(identity=identity(current),PPID=current.ppid())
        if resources:
            value.update(CPU_affinity=current.cpu_affinity(),
                memory_cgroup=cgroup_identity(_memory_cgroup(current.pid)),RSS_bytes=current.memory_info().rss)
        if _same_live_process(current.pid,birth,released,context)is None:return None
        if expected is not None:require(value['identity']==expected,'Original live actor identity changed')
        if validator is not None:validator(value)
        if _same_live_process(current.pid,birth,released,context)is None:return None
        return value
    except(psutil.NoSuchProcess,FileNotFoundError,ProcessLookupError,ValueError):
        if _same_live_process(current.pid,birth,released,context)is None:return None
        raise


def _visible(value,inventory):
    index={str(row['index']):key for key,row in inventory.items()if not key.startswith('MIG-')}
    tokens=[x.strip()for x in(value or'').split(',')if x.strip()]
    return[index.get(x,x)for x in tokens]


def cgroup_identity(value):return {'path':value['path'],'limit':value['limit']}


def observe_holder(receipt):
    """Keep the direct execution holder separate from its host-lifetime PBS shell."""
    holder=exact_actor(receipt['holder_identity']);memory=_memory_cgroup(holder.pid)
    observation=dict(identity=identity(holder),environment=holder.environ(),CPU_affinity=holder.cpu_affinity(),
        memory_limit_bytes=memory['limit'],memory_cgroup=cgroup_identity(memory))
    if receipt.get('mode')==DIRECT_MODE:
        host=exact_actor(receipt['host_allocation_process'])
        observation['host_allocation_holder']=dict(identity=identity(host),environment=host.environ())
    return holder,memory,observation


def duration(value):
    require(isinstance(value,str)and re.fullmatch(r'\d+:\d{2}:\d{2}',value),'Actual PBS elapsed/walltime required')
    hours,minutes,seconds=map(int,value.split(':'))
    require(minutes<60 and seconds<60,'Invalid actual PBS elapsed/walltime')
    return hours*3600+minutes*60+seconds


def validate_lifetime(receipt,job,*,now_monotonic,node_boot_id,query_started_monotonic=None):
    """PBS elapsed plus same-boot monotonic time; node and server clocks may differ."""
    proof=receipt.get('PBS_lifetime',{})
    require(proof.get('node_boot_id')==node_boot_id and isinstance(node_boot_id,str)and bool(node_boot_id),
        'Actual same-node boot lifetime proof required')
    started=proof.get('query_started_monotonic');wall=proof.get('walltime_seconds');elapsed=proof.get('elapsed_seconds')
    require(type(started)in(int,float)and math.isfinite(started)and 0<=started<=now_monotonic
        and type(wall)is int and type(elapsed)is int and 0<=elapsed<wall
        and proof.get('remaining_seconds')==wall-elapsed
        and isinstance(proof.get('raw_PBS_response_sha256'),str)
        and re.fullmatch('[0-9a-f]{64}',proof['raw_PBS_response_sha256']),
        'Pinned genuine PBS elapsed/monotonic admission required')
    actual_wall=duration(job['Resource_List']['walltime'])
    actual_elapsed=duration(job.get('resources_used',{}).get('walltime'))
    require(actual_wall==wall and elapsed<=actual_elapsed<wall,'Actual PBS elapsed moved backwards or allocation walltime changed')
    deadline=started+wall-elapsed
    if query_started_monotonic is not None:
        require(type(query_started_monotonic)in(int,float)and started<=query_started_monotonic<=now_monotonic,
            'Actual PBS query monotonic provenance differs')
        deadline=min(deadline,query_started_monotonic+actual_wall-actual_elapsed)
    require(now_monotonic<deadline-180,'Actual monotonic PBS lifetime reached180second reserve')
    return deadline


def validate_allocation(receipt,job,holder,devices,*,now_monotonic,node_boot_id,query_started_monotonic=None):
    """Pure validator; every production observation is obtained independently."""
    require(receipt.get('format')==FORMAT and receipt.get('UID')==UID
        and receipt.get('owner')=='aicompetition06'and isinstance(receipt.get('node'),str),
        'Explicit same-owner allocation receipt required')
    require(receipt.get('mode')in('full_GPU_shared6','six_MIG_same_parent',DIRECT_MODE),
        'Explicit allocated or previously authorized direct execution mode required')
    selected=receipt.get('execution_UUID_by_arm',{})
    require(set(selected)==set(ARMS)and all(isinstance(v,str)for v in selected.values()),
        'All six unfinished experiments need explicit execution UUIDs')
    parent=receipt.get('physical_parent_GPU_UUID');mode=receipt['mode'];assigned=set(selected.values())
    require(isinstance(parent,str)and parent.startswith('GPU-')and
        ((mode=='full_GPU_shared6'and assigned=={parent})or
         (mode in('six_MIG_same_parent',DIRECT_MODE)and len(assigned)==6 and all(v.startswith('MIG-')for v in assigned))),
        'One parent full GPU or six distinct MIG UUIDs required')
    require(receipt.get('job_id')==job.get('job_id')and job.get('job_state')=='R'
        and job.get('Job_Owner','').split('@')[0]=='aicompetition06'
        and job.get('exec_host')and all(x.split('/')[0]==receipt['node']for x in job['exec_host'].split('+')),
        'Actual own running same-host PBS allocation required')
    resources=job['Resource_List']
    if mode!=DIRECT_MODE:
        require(int(resources.get('ngpus',0))==(1 if mode=='full_GPU_shared6'else 6)
            and int(resources.get('ncpus',0))>=48,'Actual allocation must provide the requested GPU(s) and48CPUs')
    validate_lifetime(receipt,job,now_monotonic=now_monotonic,node_boot_id=node_boot_id,
        query_started_monotonic=query_started_monotonic)
    cpus=receipt.get('CPU_affinity')
    require(isinstance(cpus,list)and len(cpus)>=48 and len(set(cpus))==len(cpus)
        and holder['identity'].get('UID')==UID
        and set(cpus)<=set(holder['CPU_affinity'])and holder['identity']==receipt['holder_identity'],
        'Original own execution holder and48 actually allowed CPUs required')
    if mode==DIRECT_MODE:
        require(receipt.get('PBS_GPU_owned_reservation')is False and receipt.get('PBS_CPU_owned_reservation')is False
            and receipt.get('PBS_job_role')=='host_lifetime_only'
            and receipt.get('explicit_existing_direct_execution_requested')is True,
            'Direct execution cannot claim PBS GPU or CPU ownership')
        host=holder.get('host_allocation_holder',{})
        require(host.get('identity')==receipt.get('host_allocation_process')
            and host.get('identity',{}).get('UID')==UID
            and host.get('environment',{}).get('PBS_JOBID')==receipt['job_id'],
            'Separate exact own running host-lifetime shell required for direct execution')
        require(holder.get('memory_cgroup')==receipt.get('direct_launcher_cgroup')
            and set(receipt.get('direct_launcher_cgroup',{}))=={'path','limit'},
            'Actual direct launcher cgroup and inherited limit must remain unchanged')
    else:
        require(holder['environment'].get('PBS_JOBID')==receipt['job_id'],
            'Original owned PBS execution holder required')
        observed=holder['environment'].get('CUDA_VISIBLE_DEVICES')
        require(set(_visible(observed,devices))==assigned,'Unmodified scheduler visibility does not authorize the execution devices')
    require(holder['memory_limit_bytes']>=208*GIB,'Allocation cgroup cannot reserve six32GiB native trees plus16GiB auxiliary')
    if mode!= 'full_GPU_shared6':
        require(len({devices[token]['GI']for token in assigned if token in devices})==6,
            'Six independent MIG memory partitions required; compute instances cannot double-charge one shared GI')
    for token in assigned:
        require(token in devices and devices[token]['parent_UUID']==parent,
            'Assigned execution devices must independently map to one physical parent')
        row=devices[token]
        admitted=receipt.get('execution_device_observations',{}).get(token,{})
        require(all(admitted.get(name)==row[name]for name in('parent_UUID','index','MIG','GI','CI','MIG_enabled','total_bytes')),
            'Current execution hardware differs from its original independently observed allocation')
        if mode=='full_GPU_shared6':
            require(row['MIG']is False and row['MIG_enabled']is False and row['total_bytes']>=79*GIB,'Owned full80GB GPU required')
        else:
            require(row['MIG']is True and row['GI']is not None and row['CI']is not None,
                'Actual distinct GPU/compute-instance identity required')
    return assigned


def validate_capacity(receipt,devices,*,native_RSS,auxiliary_RSS,cgroup_usage_bytes):
    require(set(native_RSS)==set(ARMS)and all(type(x)is int and x>=0 for x in native_RSS.values()),
        'Actual per-experiment RSS required')
    require(all(x<=32*GIB for x in native_RSS.values())and 0<=auxiliary_RSS<=16*GIB,
        'Native32GiB or auxiliary16GiB RAM ceiling exceeded')
    require(type(cgroup_usage_bytes)is int and 0<=cgroup_usage_bytes<=512*GIB,
        'Actual allocation RAM including file cache exceeded512GiB')
    peaks=receipt.get('VRAM_reservation_bytes_by_arm',{})
    require(type(receipt.get('VRAM_floor_bytes'))is int and receipt['VRAM_floor_bytes']>=256*2**20,
        'Explicit positive CUDA floor of at least256MiB required')
    require(set(peaks)==set(ARMS)and all(type(v)is int and v>0 for v in peaks.values()),
        'Measured per-experiment VRAM reservations required')
    for token in set(receipt['execution_UUID_by_arm'].values()):
        row=devices[token];reserved=sum(peaks[a]for a in ARMS if receipt['execution_UUID_by_arm'][a]==token)
        require(reserved+receipt['VRAM_floor_bytes']<=row['total_bytes'],
            'Reserved native stages and CUDA floor exceed actual execution memory')
        require(row['used_bytes']<=row['total_bytes']-receipt['VRAM_floor_bytes'],
            'Execution device lost its reserved CUDA memory floor')
    return dict(native_RSS_bytes=sum(native_RSS.values()),auxiliary_RSS_bytes=auxiliary_RSS,
        actual_cgroup_RAM_bytes=cgroup_usage_bytes,aggregate_RAM_cap_bytes=512*GIB)


def validate_execution_actors(receipt,devices,owners):
    for token,row in devices.items():
        for actor in row['compute_actors']:
            owner=owners.get(actor['pid'])
            require(owner is not None and owner.get('UID')==UID,
                'Unknown own or foreign CUDA actor uses an execution device: '+json.dumps(
                    dict(observed_token=token,observed_GPU_instance=row.get('GI'),observed_compute_instance=row.get('CI'),
                        actor=actor,owner=owner),sort_keys=True))
            require(receipt['execution_UUID_by_arm'][owner['arm']]==token,
                'A native actor created CUDA work on another execution device: '+json.dumps(
                    dict(observed_token=token,expected_token=receipt['execution_UUID_by_arm'][owner['arm']],
                        observed_GPU_instance=row.get('GI'),observed_compute_instance=row.get('CI'),actor=actor,owner=owner),sort_keys=True))


def refresh_terminal_CUDA_observations(receipt,devices,owners,known,*,released):
    """A pre-tree NVML snapshot may still contain an exactly known exited actor.

    Only that terminal identity triggers one new execution-handle observation.
    The caller must still reject every unknown actor in the refreshed result.
    """
    stale=[]
    for row in devices.values():
        for actor in row['compute_actors']:
            pid=actor['pid']
            if pid not in owners and pid in known:
                expected=known[pid]
                if _same_live_process(pid,expected['create_time'],released,'known CUDA actor lifetime')is None:
                    stale.append(pid)
    if not stale:return devices
    return nvml_devices(receipt['execution_UUID_by_arm'].values())


def validate_native_child(observation,*,arm,expected_CPUs,expected_cgroup):
    failures=[]
    if observation['identity']['UID']!=UID:failures.append('UID')
    if not set(observation['CPU_affinity'])<=set(expected_CPUs):failures.append('CPU_affinity')
    if observation['memory_cgroup']!=expected_cgroup:failures.append('cgroup')
    require(not failures,'Native descendant guard rejected: '+json.dumps(dict(arm=arm,failed=failures,
        expected_CPUs=expected_CPUs,expected_cgroup=expected_cgroup,actual=observation),sort_keys=True))


def _memory_cgroup(pid):
    text=Path('/proc',str(pid),'cgroup').read_text()
    rows=[line.split(':',2)[2]for line in text.splitlines()if 'memory'in line.split(':',2)[1].split(',')]
    require(len(rows)==1,'Exact memory cgroupv1 observation required')
    folder=Path('/sys/fs/cgroup/memory')/rows[0].lstrip('/')
    return dict(path=rows[0],limit=int((folder/'memory.limit_in_bytes').read_text()),
        usage=int((folder/'memory.usage_in_bytes').read_text()))


def cuda_execution_UUID(expected):
    lib=ctypes.CDLL('libcuda.so.1');device=ctypes.c_int();value=(ctypes.c_ubyte*16)()
    require(lib.cuInit(0)==0 and lib.cuDeviceGet(ctypes.byref(device),0)==0
        and lib.cuDeviceGetUuid_v2(ctypes.byref(value),device)==0,'Actual CUDA execution UUID unavailable')
    actual=str(uuid.UUID(bytes=bytes(value)))
    require(actual==expected.removeprefix('MIG-').removeprefix('GPU-'),
        'Actual CUDA logical0 differs from the allocated execution UUID')
    return actual


def actual_PBS(cohort,receipt,actors):
    """Share real query timestamps; a transient failure never refreshes old R."""
    import fcntl
    root=Path(cohort['root']);path=root/'single_parent_PBS_observation.json'
    def existing():
        if not path.exists():return None
        value=json.loads(exact(path,owner=True).read_bytes())
        require(value['job_id']==receipt['job_id']and value['actor']in actors.values()
            and hashlib.sha256(value['raw'].encode()).hexdigest()==value['raw_sha256'],
            'Shared actual PBS observation provenance changed')
        require(value['node_boot_id']==receipt['PBS_lifetime']['node_boot_id'],
            'Shared actual PBS observation belongs to another node boot')
        age=time.monotonic()-value['query_started_monotonic']
        require(age>=0,'Shared PBS observation clock is in the future')
        return value,age
    prior=existing()
    if prior is not None and prior[1]<60:return prior[0]
    with(root/'single_parent_PBS_observation.lock').open('a+')as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        prior=existing()
        if prior is not None and prior[1]<60:return prior[0]
        started,mono=time.time(),time.monotonic()
        try:raw=subprocess.check_output(['/opt/pbs/bin/qstat','-f','-F','json',receipt['job_id']],text=True,timeout=20)
        except subprocess.TimeoutExpired:
            if prior is not None and prior[1]<300:return prior[0]
            raise RuntimeError('Initial/fresh actual running PBS observation unavailable within300seconds')
        document=json.loads(raw)
        require(set(document['Jobs'])=={receipt['job_id']}and document['Jobs'][receipt['job_id']]['job_state']=='R',
            'Actual selected PBS allocation is no longer running')
        value=dict(job_id=receipt['job_id'],raw=raw,raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            query_started_unix=started,query_started_monotonic=mono,
            node_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),actor=identity(psutil.Process()))
        temporary=path.with_name('.PBS_'+str(os.getpid())+'_'+str(time.time_ns()))
        with temporary.open('x')as stream:json.dump(value,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        return value


def nvml_devices(tokens):
    """Use NVML execution UUID and parent/GI/CI; never Torch's legacy parent UUID."""
    lib=ctypes.CDLL('libnvidia-ml.so.1')
    require(lib.nvmlInit_v2()==0,'NVML initialization failed')
    class Memory(ctypes.Structure):_fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
    class Process(ctypes.Structure):_fields_=[('pid',ctypes.c_uint),('used',ctypes.c_ulonglong),('GI',ctypes.c_uint),('CI',ctypes.c_uint)]
    def call(name,*args):
        code=getattr(lib,name)(*args);require(code==0,'Actual NVML observation failed: '+name+' '+str(code))
    def uuid_for(handle):
        out=ctypes.create_string_buffer(96);call('nvmlDeviceGetUUID',handle,out,len(out));return out.value.decode()
    process_name='nvmlDeviceGetComputeRunningProcesses_v2'
    process_method=getattr(lib,process_name)
    # Preserve the documented24byte v2 ProcessInfo ABI. Query the exact MIG
    # handle rather than infer execution scope from a parent-level GI filter.
    process_method.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint),ctypes.POINTER(Process)]
    process_method.restype=ctypes.c_int
    def actors(execution_handle):
        count=ctypes.c_uint(0);code=process_method(execution_handle,ctypes.byref(count),None)
        require(code in(0,7),'NVML CUDA process count unavailable')
        if not count.value:return[]
        for attempt in range(3):
            capacity=count.value+8;values=(Process*capacity)();count=ctypes.c_uint(capacity)
            code=process_method(execution_handle,ctypes.byref(count),values)
            if code==7:continue
            require(code==0,'Exact execution-scoped NVML CUDA process query failed: '+str(code))
            return[{'pid':int(v.pid),'used_bytes':int(v.used),'reported_GPU_instance':int(v.GI),
                'reported_compute_instance':int(v.CI),'NVML_function':process_name,
                'scope':'exact execution UUID handle'}for v in values[:count.value]]
        raise RuntimeError('Execution-scoped NVML process list changed during all bounded retries')
    result={}
    for token in set(tokens):
        handle=ctypes.c_void_p();call('nvmlDeviceGetHandleByUUID',token.encode(),ctypes.byref(handle))
        actual=uuid_for(handle);require(actual==token,'Actual NVML execution UUID changed')
        memory=Memory();call('nvmlDeviceGetMemoryInfo',handle,ctypes.byref(memory))
        mig=ctypes.c_uint();call('nvmlDeviceIsMigDeviceHandle',handle,ctypes.byref(mig))
        gi=ci=None;parent=handle
        if mig.value:
            parent=ctypes.c_void_p();call('nvmlDeviceGetDeviceHandleFromMigDeviceHandle',handle,ctypes.byref(parent))
            g=ctypes.c_uint();c=ctypes.c_uint();call('nvmlDeviceGetGpuInstanceId',handle,ctypes.byref(g));call('nvmlDeviceGetComputeInstanceId',handle,ctypes.byref(c));gi,ci=g.value,c.value
        index=ctypes.c_uint();call('nvmlDeviceGetIndex',parent,ctypes.byref(index))
        current=ctypes.c_uint();pending=ctypes.c_uint()
        mode_code=lib.nvmlDeviceGetMigMode(parent,ctypes.byref(current),ctypes.byref(pending))
        require(mode_code in(0,3),'Actual parent MIG mode unavailable')
        result[token]=dict(parent_UUID=uuid_for(parent),index=index.value,MIG=bool(mig.value),GI=gi,CI=ci,
            MIG_enabled=bool(current.value)if mode_code==0 else False,
            total_bytes=memory.total,used_bytes=memory.used,free_bytes=memory.free,compute_actors=actors(handle))
    return result


class Lease:
    def __init__(self,cohort,arm,actors):
        self.cohort,self.arm,self.actors=cohort,arm,actors
        self.receipt=json.loads(pin(cohort['allocation'],owner=True).read_bytes())
        self.own=identity(psutil.Process());self.last=0;self.files={};self.previous_trees={}
        self.last_probe={}
        self.canonical=Path(cohort['canonical_code']['path'])
        required={'tools/run_basic_cp_single_parent.py','hiercp_v1x/basic_cp_single_parent_resources.py',
            'custom_trainers/nnUNetTrainer_BasicCPExpanded.py','custom_trainers/nnUNetTrainer_BasicCPMultiNJoint.py',
            'custom_trainers/nnUNetTrainer_BasicCPAblations.py','custom_trainers/nnUNetTrainer_OnlinePairedCP.py'}
        if self.receipt.get('mode')==DIRECT_MODE:required.add('hiercp_v1x/basic_cp_direct_MIG_access.py')
        require(required<=set(cohort['canonical_code']['files_sha256']),
            'One canonical code manifest must bind the actual runner, guards and scientific trainers')
        actual=subprocess.check_output(['git','-C',str(self.canonical),'rev-parse','HEAD'],text=True,timeout=10).strip()
        require(actual==cohort['canonical_code']['commit'],'Canonical Git HEAD differs from the admitted commit')
        for name,checksum in cohort['canonical_code']['files_sha256'].items():
            p=pin({'path':str(self.canonical/name),'sha256':checksum},owner=True);self.files[str(p)]=signature(p)
        for name,checksum in cohort['installed_runtime_files_sha256'].items():
            p=pin({'path':name,'sha256':checksum});self.files[str(p)]=signature(p)
        evidence=json.loads(pin(self.receipt['VRAM_measurement_evidence']).read_bytes())
        evidence=evidence.get('DEBUG_result.json',evidence)
        require(evidence.get('accepted')is True and evidence.get('physical_batch')==2
            and evidence.get('parameters')==102350575 and evidence.get('patch_size')==[128]*3,
            'Actual full-model B2 CUDA evidence required for VRAM admission')
        measured=max(row['peak_reserved_bytes']for row in evidence['measurements'])
        require(all(value>=measured for value in self.receipt['VRAM_reservation_bytes_by_arm'].values()),
            'VRAM reservations cannot fall below the measured actual cold-stage peak')
        require(set(actors)==set(ARMS),'Exact six independently captured native actor identities required')
        require(actors[arm]==self.own,'Calling native process differs from its own admitted actor')
        for peer in ARMS:
            expected=[sys.executable,'-B','-u',str(self.canonical/'tools/run_basic_cp_single_parent.py'),
                '--cohort',cohort['_execution_cohort_path'],'--cohort-sha256',cohort['_execution_cohort_SHA256'],
                '--arm',peer,'--execute']
            require(actors[peer]['command']==expected,'A registered peer is outside this exact canonical cohort command')

    def _publish_probe(self):
        root=Path(self.cohort['experiments'][self.arm]['output_root']);path=root/'last_resource_guard.json'
        temporary=path.with_name('.resource_guard_'+str(os.getpid())+'_'+str(time.time_ns()))
        with temporary.open('x',encoding='utf8')as stream:
            json.dump(self.last_probe,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)

    def verify(self,*,force=False):
        try:
            result=self._verify(force=force)
            if force:
                self.last_probe['status']='PASS';self._publish_probe()
            return result
        except BaseException as error:
            self.last_probe.update(status='FAILED',error=type(error).__name__+': '+str(error),observed_unix=time.time())
            self._publish_probe()
            raise

    def _verify(self,*,force=False):
        now=time.monotonic()
        lifetime=self.receipt['PBS_lifetime']
        require(now<lifetime['query_started_monotonic']+lifetime['remaining_seconds']-180,
            'Actual monotonic allocation lifetime reached180second reserve')
        if not force and time.monotonic()-self.last<5:return
        self.last_probe=dict(format='basic_cp_exact_native_resource_guard_observation_v1',arm=self.arm,
            calling_actor=self.own,phase='actual_source_and_host',observed_unix=time.time(),native_children={})
        require(os.getuid()==UID and socket.gethostname().split('.')[0]==self.receipt['node'],
            'Actual native owner or host changed')
        for name,value in self.files.items():require(signature(exact(name))==value,'Admitted canonical/runtime source changed: '+name)
        holder,mem,holder_observation=observe_holder(self.receipt)
        own_cgroup=_memory_cgroup(os.getpid())
        direct=self.receipt.get('mode')==DIRECT_MODE
        require(cgroup_identity(mem)==cgroup_identity(own_cgroup)
            and(direct or self.receipt['job_id']in mem['path']),
            'Native process moved outside its unchanged admitted execution cgroup')
        observation=actual_PBS(self.cohort,self.receipt,self.actors);document=json.loads(observation['raw'])
        require(set(document['Jobs'])=={self.receipt['job_id']},'Exact own current PBS job required')
        job=dict(document['Jobs'][self.receipt['job_id']],job_id=self.receipt['job_id'])
        devices=nvml_devices(self.receipt['execution_UUID_by_arm'].values())
        self.last_probe.update(phase='execution_devices',devices=devices)
        selected=validate_allocation(self.receipt,job,holder_observation,devices,
            now_monotonic=time.monotonic(),node_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            query_started_monotonic=observation['query_started_monotonic'])
        expected_UUID=self.receipt['execution_UUID_by_arm'][self.arm]
        require(os.environ.get('CUDA_VISIBLE_DEVICES')==expected_UUID,'Native CUDA visibility changed')
        cuda_execution_UUID(expected_UUID)
        self.last_probe.update(own_CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
            own_CUDA_v2_UUID_verified=expected_UUID,phase='native_process_trees')
        known_actor_identities={value['pid']:value for value in self.actors.values()}
        for previous in self.previous_trees.values():known_actor_identities.update(previous)
        cpus=self.receipt['CPU_affinity'];trees={};native_RSS={};all_PIDs=set();released=[]
        self.last_probe['naturally_released_processes']=released
        for index,arm in enumerate(ARMS):
            expected=cpus[index*8:index*8+8];tree={};native_RSS[arm]=0
            self.last_probe['native_children'][arm]=dict(expected_CPU_affinity=expected,
                expected_cgroup=cgroup_identity(mem),observations=[])
            def validate_child(value):
                validate_native_child(value,arm=arm,expected_CPUs=expected,expected_cgroup=cgroup_identity(mem))
            def capture(child,known=None):
                def validate(value):
                    validate_child(value)
                    if known==self.actors[arm]:require(value['CPU_affinity']==expected,'Actual native eight-CPU affinity changed')
                value=process_snapshot(child,expected=known,validator=validate,released=released,context='native '+arm)
                if value is not None:
                    tree[child.pid]=value
                    self.last_probe['native_children'][arm]['observations'].append(value)
                return value
            root=None
            try:root=capture(psutil.Process(self.actors[arm]['pid']),self.actors[arm])
            except psutil.NoSuchProcess:
                require(_same_live_process(self.actors[arm]['pid'],self.actors[arm]['create_time'],released,
                    'native root '+arm)is None,'Live native root observation failed')
            if root is not None:
                try:
                    children=psutil.Process(root['identity']['pid']).children(recursive=True)
                except psutil.NoSuchProcess:
                    require(_same_live_process(root['identity']['pid'],root['identity']['create_time'],released,
                        'native child enumeration '+arm)is None,'Live native child enumeration failed')
                    children=[]
                for child in children:capture(child)
            # Retain exact previously observed workers even if a parent exits
            # and Linux reparents them during this observation.
            for pid,known_worker_identity in self.previous_trees.get(arm,{}).items():
                if pid in tree:continue
                try:capture(psutil.Process(pid),known_worker_identity)
                except psutil.NoSuchProcess:
                    require(_same_live_process(pid,known_worker_identity['create_time'],released,
                        'previous native '+arm)is None,'Live previous native observation failed')
            native_RSS[arm]=sum(row['RSS_bytes']for row in tree.values());all_PIDs.update(tree)
            trees[arm]=tree
            self.previous_trees[arm]={pid:value['identity']for pid,value in tree.items()}
        owners={}
        for arm,tree in trees.items():
            for pid,value in tree.items():
                require(pid not in owners or owners[pid]['arm']==arm,
                    'Exact native process trees overlap: '+json.dumps(dict(pid=pid,first=owners.get(pid),second_arm=arm,
                        identity=value['identity'],PPID=value['PPID']),sort_keys=True))
                owners[pid]=dict(arm=arm,UID=value['identity']['UID'],identity=value['identity'],PPID=value['PPID'])
        prior_devices=devices
        devices=refresh_terminal_CUDA_observations(self.receipt,devices,owners,known_actor_identities,released=released)
        if devices is not prior_devices:
            self.last_probe['pre_lifetime_refresh_devices']=prior_devices
            validate_allocation(self.receipt,job,holder_observation,devices,
                now_monotonic=time.monotonic(),node_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                query_started_monotonic=observation['query_started_monotonic'])
        self.last_probe.update(phase='CUDA_actor_ownership',owners=owners,devices=devices)
        validate_execution_actors(self.receipt,devices,owners)
        if direct:
            from .basic_cp_direct_MIG_access import observe_direct_memory
            cohort_PIDs={holder.pid}
            for process in holder.children(recursive=True):
                def validate_auxiliary(value):
                    require(value['identity']['UID']==UID and value['memory_cgroup']==cgroup_identity(mem),
                        'Direct cohort live child moved outside the original own UID/cgroup: '+json.dumps(value,sort_keys=True))
                value=process_snapshot(process,validator=validate_auxiliary,released=released,context='direct cohort child')
                if value is not None:cohort_PIDs.add(process.pid)
            # Already captured exact native descendants remain charged after a
            # peer's natural release, even if Linux reparents a finishing worker.
            cohort_PIDs.update(all_PIDs)
            memory=observe_direct_memory(native_RSS,cohort_PIDs,
                native_PIDs_by_arm={arm:set(tree)for arm,tree in trees.items()})
            self.last_probe.update(phase='direct_memory',memory=memory)
            native_RSS=memory['native_RSS_by_arm']
            global_RAM=memory['current_uid_RSS_bytes']+memory['own_tmpfs_allocated_bytes']
            result=validate_capacity(self.receipt,devices,native_RSS=native_RSS,
                auxiliary_RSS=memory['cohort_auxiliary_RSS_bytes'],cgroup_usage_bytes=global_RAM)
            result['actual_global_own_UID_and_tmpfs_RAM_bytes']=result.pop('actual_cgroup_RAM_bytes')
            result.update(direct_memory=memory,PBS_GPU_owned_reservation=False,PBS_CPU_owned_reservation=False,
                PBS_job_role='host_lifetime_only',RAM_accounting='own UID RSS plus owned tmpfs allocated blocks')
            self.last=time.monotonic();return result
        # Cgroup file cache is counted by the total512GiB limit; only actual
        # non-native process RSS belongs to the separate16GiB auxiliary budget.
        auxiliary=0
        for p in psutil.process_iter():
            try:
                if p.pid not in all_PIDs and p.uids().real==UID and _memory_cgroup(p.pid)['path']==mem['path']:
                    auxiliary+=p.memory_info().rss
            except(psutil.NoSuchProcess,FileNotFoundError):continue
        result=validate_capacity(self.receipt,devices,native_RSS=native_RSS,auxiliary_RSS=auxiliary,cgroup_usage_bytes=mem['usage'])
        self.last=time.monotonic();return result
