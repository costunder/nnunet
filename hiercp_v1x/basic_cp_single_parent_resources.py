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
    try:
        p=psutil.Process(expected['pid'])
        if p.create_time()!=expected['create_time']:return None
        if p.status()in(psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD):return None
        return exact_actor(expected)
    except psutil.NoSuchProcess:return None


def _visible(value,inventory):
    index={str(row['index']):key for key,row in inventory.items()if not key.startswith('MIG-')}
    tokens=[x.strip()for x in(value or'').split(',')if x.strip()]
    return[index.get(x,x)for x in tokens]


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
    require(receipt.get('mode')in('full_GPU_shared6','six_MIG_same_parent'),
        'Full GPU or six separately owned MIG instances required')
    selected=receipt.get('execution_UUID_by_arm',{})
    require(set(selected)==set(ARMS)and all(isinstance(v,str)for v in selected.values()),
        'All six unfinished experiments need explicit execution UUIDs')
    parent=receipt.get('physical_parent_GPU_UUID');mode=receipt['mode'];assigned=set(selected.values())
    require(isinstance(parent,str)and parent.startswith('GPU-')and
        ((mode=='full_GPU_shared6'and assigned=={parent})or
         (mode=='six_MIG_same_parent'and len(assigned)==6 and all(v.startswith('MIG-')for v in assigned))),
        'One parent full GPU or six distinct MIG UUIDs required')
    require(receipt.get('job_id')==job.get('job_id')and job.get('job_state')=='R'
        and job.get('Job_Owner','').split('@')[0]=='aicompetition06'
        and job.get('exec_host')and all(x.split('/')[0]==receipt['node']for x in job['exec_host'].split('+')),
        'Actual own running same-host PBS allocation required')
    resources=job['Resource_List']
    require(int(resources.get('ngpus',0))==(1 if mode=='full_GPU_shared6'else 6)
        and int(resources.get('ncpus',0))>=48,'Actual allocation must provide the requested GPU(s) and48CPUs')
    validate_lifetime(receipt,job,now_monotonic=now_monotonic,node_boot_id=node_boot_id,
        query_started_monotonic=query_started_monotonic)
    cpus=receipt.get('CPU_affinity')
    require(isinstance(cpus,list)and len(cpus)>=48 and len(set(cpus))==len(cpus)
        and holder['identity'].get('UID')==UID
        and set(cpus)<=set(holder['CPU_affinity'])and holder['identity']==receipt['holder_identity']
        and holder['environment'].get('PBS_JOBID')==receipt['job_id'],
        'Original own allocation holder and48 allowed CPUs required')
    observed=holder['environment'].get('CUDA_VISIBLE_DEVICES')
    require(set(_visible(observed,devices))==assigned,'Unmodified scheduler visibility does not authorize the execution devices')
    require(holder['memory_limit_bytes']>=208*GIB,'Allocation cgroup cannot reserve six32GiB native trees plus16GiB auxiliary')
    for token in assigned:
        require(token in devices and devices[token]['parent_UUID']==parent,
            'Assigned execution devices must independently map to one physical parent')
        row=devices[token]
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
                'Unknown own or foreign CUDA actor uses an allocated execution device')
            require(receipt['execution_UUID_by_arm'][owner['arm']]==token,
                'A native actor created CUDA work on another execution device')


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
    def actors(parent,gi,ci):
        count=ctypes.c_uint(0);code=lib.nvmlDeviceGetComputeRunningProcesses_v2(parent,ctypes.byref(count),None)
        require(code in(0,7),'NVML CUDA process count unavailable')
        if not count.value:return[]
        values=(Process*count.value)();call('nvmlDeviceGetComputeRunningProcesses_v2',parent,ctypes.byref(count),values)
        return[{'pid':int(v.pid),'used_bytes':int(v.used)}for v in values[:count.value]
            if gi is None or(v.GI==gi and v.CI==ci)]
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
            total_bytes=memory.total,used_bytes=memory.used,free_bytes=memory.free,compute_actors=actors(parent,gi,ci))
    return result


class Lease:
    def __init__(self,cohort,arm,actors):
        self.cohort,self.arm,self.actors=cohort,arm,actors
        self.receipt=json.loads(pin(cohort['allocation'],owner=True).read_bytes())
        self.own=identity(psutil.Process());self.last=0;self.files={};self.previous_trees={}
        self.canonical=Path(cohort['canonical_code']['path'])
        required={'tools/run_basic_cp_single_parent.py','hiercp_v1x/basic_cp_single_parent_resources.py',
            'custom_trainers/nnUNetTrainer_BasicCPExpanded.py','custom_trainers/nnUNetTrainer_BasicCPMultiNJoint.py',
            'custom_trainers/nnUNetTrainer_BasicCPAblations.py','custom_trainers/nnUNetTrainer_OnlinePairedCP.py'}
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

    def verify(self,*,force=False):
        now=time.monotonic()
        lifetime=self.receipt['PBS_lifetime']
        require(now<lifetime['query_started_monotonic']+lifetime['remaining_seconds']-180,
            'Actual monotonic allocation lifetime reached180second reserve')
        if not force and time.monotonic()-self.last<5:return
        require(os.getuid()==UID and socket.gethostname().split('.')[0]==self.receipt['node'],
            'Actual native owner or host changed')
        for name,value in self.files.items():require(signature(exact(name))==value,'Admitted canonical/runtime source changed: '+name)
        holder=exact_actor(self.receipt['holder_identity']);mem=_memory_cgroup(holder.pid)
        own_cgroup=_memory_cgroup(os.getpid())
        require(mem['path']==own_cgroup['path']and self.receipt['job_id']in mem['path'],
            'Native process is outside its actual owned allocation cgroup')
        observation=actual_PBS(self.cohort,self.receipt,self.actors);document=json.loads(observation['raw'])
        require(set(document['Jobs'])=={self.receipt['job_id']},'Exact own current PBS job required')
        job=dict(document['Jobs'][self.receipt['job_id']],job_id=self.receipt['job_id'])
        devices=nvml_devices(self.receipt['execution_UUID_by_arm'].values())
        selected=validate_allocation(self.receipt,job,dict(identity=identity(holder),environment=holder.environ(),
            CPU_affinity=holder.cpu_affinity(),memory_limit_bytes=mem['limit']),devices,
            now_monotonic=time.monotonic(),node_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            query_started_monotonic=observation['query_started_monotonic'])
        expected_UUID=self.receipt['execution_UUID_by_arm'][self.arm]
        require(os.environ.get('CUDA_VISIBLE_DEVICES')==expected_UUID,'Native CUDA visibility changed')
        cuda_execution_UUID(expected_UUID)
        cpus=self.receipt['CPU_affinity'];trees={};native_RSS={};all_PIDs=set()
        for index,arm in enumerate(ARMS):
            p=live_actor(self.actors[arm]);expected=cpus[index*8:index*8+8]
            if p is not None:
                require(p.cpu_affinity()==expected,'Actual native eight-CPU affinity changed')
                tree={p.pid:p,**{c.pid:c for c in p.children(recursive=True)}}
            else:
                tree={}
                for known in self.previous_trees.get(arm,{}).values():
                    survivor=live_actor(known)
                    if survivor is not None:tree[survivor.pid]=survivor
            native_RSS[arm]=0
            for child in tree.values():
                try:
                    require(child.uids().real==UID and set(child.cpu_affinity())<=set(expected)
                        and _memory_cgroup(child.pid)['path']==mem['path'],'Foreign UID/CPU/cgroup appeared in native tree')
                    native_RSS[arm]+=child.memory_info().rss;all_PIDs.add(child.pid)
                except psutil.NoSuchProcess:continue
            trees[arm]=tree
            self.previous_trees[arm]={pid:identity(value)for pid,value in tree.items()}
        owners={}
        for arm,tree in trees.items():
            for pid,p in tree.items():
                try:owners[pid]=dict(arm=arm,UID=p.uids().real)
                except psutil.NoSuchProcess:continue
        validate_execution_actors(self.receipt,devices,owners)
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
