"""GPU4 scientific source, explicitly sealed native execution on physical GPU0.

Only process admission, physical CUDA selection and child entry points change.
The admitted original preparation, augmentation, trainer, optimizer and BEST
evaluation functions execute from their exact frozen source checkout.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
import types

import psutil

FORMAT = 'v24_gpu4_source_gpu0_native_execution_v1'
RESOURCE_FORMAT = 'v24_gpu0_native_resource_proof_v1'
STATUS_FORMAT = 'v24_gpu4_source_gpu0_native_execution_status_v1'
UID = 41833
GIB = 2**30
FILES = ('hiercp_v1x/v24_gpu0_native_execution.py', 'tools/run_v24_gpu0_native.py',
         'hiercp_v1x/v24_gpu0_native_storage.py', 'hiercp_v1x/v24_gpu0_native_persistence.py')
SCIENCE = dict(epochs=250, physical_batch=2, gradient_accumulation=1,
    patch=[128,128,128], training_cases=105, validation_cases=26,
    source_entries=642, candidate_count=128, cp_probability=.5,
    CPU_workers=4, evaluation_checkpoint='checkpoint_best.pth')
NATIVE_ARGV = ['730','3d_fullres','0','-tr','nnUNetTrainer_250epochs_FrozenV23CP',
               '-p','nnUNetResEncUNetMPlans','--val_best']
TRAIN_ENTRY = 'from hiercp_v1x.v24_matched_basic_cp_runtime import run_training_entry; run_training_entry()'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            result.update(block)
    return result.hexdigest()


def checksum(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('Exact absolute non-traversing path required')
    return Path(value)


def owned(value, *, directory=False):
    path = absolute(str(value))
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('Execution path must have no symlink components')
    info = path.stat()
    if (not hasattr(os, 'getuid') or os.getuid() != UID or info.st_uid != UID
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))):
        raise ValueError('Actual own Linux UID41833 regular execution path required')
    return path


def read(path):
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError('Actual JSON object required')
    return value


def witness(value):
    if not isinstance(value, dict) or set(value) != {'path','sha256'} or not checksum(value['sha256']):
        raise ValueError('Exact immutable path/SHA witness required')
    path = owned(value['path'])
    if digest(path) != value['sha256']:
        raise ValueError('Immutable input SHA changed: '+str(path))
    return path


def publish(path, value, *, replace=False):
    path = absolute(str(path)); owned(path.parent, directory=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf8')
    if not replace:
        with path.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        return
    temporary = path.parent/('.'+path.name+'.'+str(os.getpid())+'.'+str(time.time_ns())+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    if path.exists():
        owned(path)
    os.replace(temporary, path)


def validate_request(value):
    required = {'format','source_arm','execution_physical_gpu','execution_gpu_uuid',
        'source_code','source_code_files','adapter_code','adapter_code_files','python',
        'metadata_root','execution_root','native_path','overlay','storage_admission',
        'resource_proof_path','science'}
    if (not isinstance(value, dict) or not required <= set(value) or value.get('format') != FORMAT
            or type(value.get('source_arm')) is not int or value['source_arm'] != 4
            or type(value.get('execution_physical_gpu')) is not int or value['execution_physical_gpu'] != 0
            or not isinstance(value.get('execution_gpu_uuid'), str)
            or re.fullmatch(r'GPU-[0-9a-f-]+', value['execution_gpu_uuid']) is None
            or value.get('science') != SCIENCE):
        raise ValueError('Explicit source GPU4 / physical GPU0 full250/B2/BEST26 contract required')
    for key in ('source_code','adapter_code','python','metadata_root','execution_root',
                'native_path','resource_proof_path'):
        absolute(value[key])
    if (Path(value['native_path']) != Path(value['execution_root'])/'native/native.json'
            or Path(value['resource_proof_path']).parent != Path(value['metadata_root'])):
        raise ValueError('Separate persistent controller and fresh native namespace binding required')
    for key in ('source_code_files','adapter_code_files'):
        entries = value[key]
        if not isinstance(entries, dict) or not entries:
            raise ValueError('Frozen source SHA manifest required')
        for name, sha in entries.items():
            if (not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts
                    or not checksum(sha)):
                raise ValueError('Pinned relative source path/SHA required')
    if not set(FILES) <= set(value['adapter_code_files']):
        raise ValueError('All execution/storage adapter sources must be sealed')
    for key in ('overlay','storage_admission'):
        if not isinstance(value[key], dict) or set(value[key]) != {'path','sha256'} or not checksum(value[key]['sha256']):
            raise ValueError('Exact source overlay/storage admission witnesses required')
        absolute(value[key]['path'])
    return value


def verify_sources(request):
    for folder, manifest in (('source_code','source_code_files'), ('adapter_code','adapter_code_files')):
        root = owned(request[folder], directory=True)
        for name, sha in request[manifest].items():
            if digest(owned(root/name)) != sha:
                raise ValueError('Frozen implementation SHA changed: '+name)
    if Path(__file__).resolve() != Path(request['adapter_code'])/'hiercp_v1x/v24_gpu0_native_execution.py':
        raise ValueError('Adapter imported from another source checkout')


def actor(process):
    if process.status() == psutil.STATUS_ZOMBIE:
        raise RuntimeError('Current actor/controller cannot be a zombie')
    return dict(pid=process.pid, create_time=process.create_time(), command=process.cmdline(), UID=process.uids().real)


def exact(value):
    if (not isinstance(value, dict) or set(value) != {'pid','create_time','command','UID'}
            or type(value['pid']) is not int or value['pid'] <= 0 or value['UID'] != UID
            or type(value['create_time']) not in (int,float) or not math.isfinite(value['create_time'])
            or value['create_time'] <= 0 or not isinstance(value['command'], list) or not value['command']):
        raise ValueError('Exact own PID/birth/argv/UID identity required')
    process = psutil.Process(value['pid'])
    if actor(process) != value:
        raise RuntimeError('Actor/controller PID, birth, command or UID changed')
    return process


def capture_cgroup_admission(pid):
    text = (Path('/proc')/str(pid)/'cgroup').read_text(); limits = {}
    for line in text.splitlines():
        hierarchy, controllers, relative = line.split(':', 2)
        if hierarchy == '0' and controllers == '':
            root, name = Path('/sys/fs/cgroup'), 'memory.max'
        elif 'memory' in controllers.split(','):
            root, name = Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes'
        else:
            continue
        if not relative.startswith('/') or '..' in Path(relative).parts:
            raise ValueError('Actual absolute cgroup path required')
        folder = root/relative.lstrip('/')
        while folder == root or root in folder.parents:
            path = folder/name
            if path.is_file():
                raw = path.read_text().strip()
                if raw != 'max' and not raw.isdigit():
                    raise ValueError('Actual cgroup limit must be bytes or max')
                limits[str(path)] = None if raw == 'max' else int(raw)
            if folder == root:
                break
            folder = folder.parent
    if not limits:
        raise RuntimeError('Actual cgroup ancestor memory limits unavailable')
    return dict(controller_cgroup=text, memory_limits=[dict(path=name,limit_bytes=limits[name]) for name in sorted(limits)])


def query(command):
    return subprocess.check_output(command, text=True, timeout=20).strip()


class ExecutionGuard:
    def __init__(self, request, request_path, request_sha256):
        self.request = request; self.path = owned(request_path); self.sha = request_sha256
        self.proof_path = owned(request['resource_proof_path'])
        self.proof_sha = digest(self.proof_path); self.proof = read(self.proof_path)
        value = self.proof
        if (value.get('format') != RESOURCE_FORMAT or value.get('request_sha256') != self.sha
                or value.get('RSS_limit_bytes') != 48*GIB or value.get('initial_remaining_seconds') != 25200
                or value.get('PBS_owned_reservation') is not False
                or value.get('explicit_direct_execution_authorized') is not True
                or value.get('aggregate_RSS_reservation_bytes') != 464*GIB
                or not isinstance(value.get('CPU_affinity'), list) or len(value['CPU_affinity']) != 4
                or any(type(cpu) is not int or cpu < 0 for cpu in value['CPU_affinity'])
                or value['CPU_affinity'] != sorted(set(value['CPU_affinity']))):
            raise ValueError('Exact actual GPU0 fourCPU/48GiB and concurrent464GiB admission required')
        lifetime = value.get('host_lifetime')
        if (not isinstance(lifetime, dict) or set(lifetime) != {'allocation_request','allocation_request_sha256','expires_at_unix'}
                or not checksum(lifetime['allocation_request_sha256'])):
            raise ValueError('Unchanged owned GPU6 host-lifetime witness required')
        self.lifetime_path = owned(lifetime['allocation_request'])
        if digest(self.lifetime_path) != lifetime['allocation_request_sha256']:
            raise ValueError('Actual host lifetime request SHA changed')
        self.lifetime = read(self.lifetime_path)
        if (self.lifetime.get('format') != 'v25_owned_GPU_allocation_v1'
                or self.lifetime.get('gpu') != 6 or self.lifetime.get('UID') != UID
                or self.lifetime.get('owner') != 'aicompetition06' or self.lifetime.get('node') != 'ece-a6gpu6'
                or self.lifetime.get('expires_at_unix') != lifetime['expires_at_unix']):
            raise ValueError('Host job is only the exact owned GPU6 lifetime witness')
        self.self_identity = actor(psutil.Process()); self.managed = {}; self.last_checked = 0
        self.monotonic_expiry = time.monotonic()+lifetime['expires_at_unix']-time.time()
        self(initial=True)

    def tree(self):
        controller = exact(self.proof['controller_identity']); own = exact(self.self_identity)
        tree = {controller.pid: controller}
        for process in controller.children(recursive=True):
            try:
                if process.uids().real != UID:
                    raise RuntimeError('Foreign UID descendant cannot be admitted')
                if process.status() == psutil.STATUS_ZOMBIE:
                    continue
                current = actor(process)
            except psutil.NoSuchProcess:
                continue
            if process.pid in self.managed and self.managed[process.pid] != current:
                raise RuntimeError('Managed descendant PID/birth/argv changed')
            self.managed[process.pid] = current; tree[process.pid] = process
        if own.pid not in tree:
            raise RuntimeError('Native stage must be actual controller or descendant')
        if controller.cpu_affinity() != self.proof['CPU_affinity'] or own.cpu_affinity() != self.proof['CPU_affinity']:
            raise RuntimeError('Exact fourCPU controller/native affinity changed')
        return controller, own, tree

    def lifetime_check(self, initial):
        remaining = min(self.lifetime['expires_at_unix']-time.time(), self.monotonic_expiry-time.monotonic())
        if remaining < (25200 if initial else 180):
            raise RuntimeError('Actual host lifetime insufficient for native stage')
        document = json.loads(query(['/opt/pbs/bin/qstat','-f','-F','json',self.lifetime['job_id']]))
        jobs = document.get('Jobs', {})
        if list(jobs) != [self.lifetime['job_id']]:
            raise RuntimeError('Exact host lifetime PBS job is unavailable')
        job = jobs[self.lifetime['job_id']]
        if (job.get('job_state') != 'R' or job.get('Job_Owner','').split('@')[0] != 'aicompetition06'
                or int(job.get('Resource_List',{}).get('ngpus',0)) != 1
                or not job.get('exec_host')
                or any(part.split('/')[0] != 'ece-a6gpu6' for part in job['exec_host'].split('+'))):
            raise RuntimeError('Actual own host lifetime job identity changed')
        started = job.get('stime')
        if isinstance(started, str):
            try:
                started = float(started)
            except ValueError:
                started = time.mktime(time.strptime(started,'%a %b %d %H:%M:%S %Y'))
        hours, minutes, seconds = map(int,job['Resource_List']['walltime'].split(':'))
        expiry = float(started)+hours*3600+minutes*60+seconds
        if abs(expiry-self.lifetime['expires_at_unix']) > 1:
            raise RuntimeError('Actual PBS lifetime start/walltime changed')

    def gpu_check(self, tree, *, initial=False):
        fields = [part.strip() for part in query(['nvidia-smi','--id=0',
            '--query-gpu=index,uuid,memory.total,memory.free,mig.mode.current','--format=csv,noheader,nounits']).split(',')]
        if (len(fields) != 5 or fields[0] != '0' or fields[1] != self.request['execution_gpu_uuid']
                or fields[4] not in ('Disabled','N/A','[N/A]') or float(fields[2]) < 39000):
            raise RuntimeError('Actual full physical GPU0 UUID/40GiB/MIG admission changed')
        if initial and float(fields[3]) < 38000:
            raise RuntimeError('Native startup requires actually free full physical GPU0 memory')
        apps = query(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'])
        for row in apps.splitlines():
            if not row.strip():
                continue
            uuid, pid = [part.strip() for part in row.split(',')]
            if uuid == self.request['execution_gpu_uuid'] and int(pid) not in tree:
                raise RuntimeError('GPU0 has a CUDA actor outside this exact controller tree')
        cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cvd is not None and cvd != self.request['execution_gpu_uuid']:
            raise RuntimeError('CUDA visibility differs from explicit GPU0 UUID')

    def __call__(self, *, initial=False, force=False):
        if not initial and not force and time.monotonic()-self.last_checked < 30:
            return self.receipt
        if (not hasattr(os,'getuid') or os.getuid() != UID or socket.gethostname().split('.')[0] != 'ece-a6gpu6'
                or digest(self.path) != self.sha or digest(self.proof_path) != self.proof_sha
                or digest(self.lifetime_path) != self.proof['host_lifetime']['allocation_request_sha256']):
            raise RuntimeError('Actual host/UID or sealed execution/resource proof changed')
        controller, own, tree = self.tree(); self.lifetime_check(initial); self.gpu_check(tree,initial=initial)
        observed = capture_cgroup_admission(controller.pid)
        if observed != self.proof.get('cgroup_admission'):
            raise RuntimeError('Controller cgroup/ancestor limits changed')
        stage_cgroup = capture_cgroup_admission(own.pid)
        for document in (observed, stage_cgroup):
            for row in document['memory_limits']:
                if row['limit_bytes'] is not None and row['limit_bytes'] < 464*GIB:
                    raise MemoryError('Cgroup cannot admit concurrent V25 416GiB plus native48GiB')
        rss = 0
        for process in tree.values():
            try:
                rss += process.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        memory = psutil.virtual_memory()
        if rss > 48*GIB or memory.available+rss < 48*GIB:
            raise MemoryError('Actual native controller tree exceeds original48GiB admission')
        self.receipt = dict(format=RESOURCE_FORMAT,source_arm=4,execution_physical_gpu=0,
            execution_gpu_uuid=self.request['execution_gpu_uuid'],CPU_affinity=self.proof['CPU_affinity'],
            process_and_children_RSS_bytes=rss,RSS_limit_bytes=48*GIB,
            request_sha256=self.sha,resource_proof_sha256=self.proof_sha,
            controller_identity=self.proof['controller_identity'],PBS_owned_reservation=False,
            host_job_role='host_lifetime_only',actual_cgroup=stage_cgroup)
        self.last_checked = time.monotonic()
        return self.receipt

    def select(self, source_arm):
        if type(source_arm) is not int or source_arm != 4:
            raise ValueError('Only sealed source GPU4 may select physical GPU0')
        self(force=True)
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
            raise RuntimeError('Physical GPU selection must precede CUDA initialization')
        os.environ['CUDA_VISIBLE_DEVICES'] = self.request['execution_gpu_uuid']
        return self.request['execution_gpu_uuid']


def command(request, request_path, request_sha256, action, *tail):
    return [request['python'],'-B','-u',str(Path(request['adapter_code'])/'tools/run_v24_gpu0_native.py'),
        action,'--request',str(request_path),'--request-sha256',request_sha256,*tail]


def rewrite_child(original, request, request_path, request_sha256):
    prefix = [request['python'],'-B','-u']
    if original[:3] != prefix:
        raise ValueError('Only exact original admitted native interpreter child may execute')
    if original[3:5] == [str(Path(request['source_code'])/'tools/run_v24_matched_basic_cp.py'),'clone-debug-worker']:
        expected = ['--native',request['native_path'],'--physical-batch']
        if original[5:8] != expected or len(original) != 11 or original[8] not in ('2','4') or original[9] != '--output':
            raise ValueError('Only exact original B2/B4 clone worker may execute')
        return command(request,request_path,request_sha256,'clone-debug-worker',*original[5:])
    if original[3:5] == ['-c',TRAIN_ENTRY] and original[5:] == NATIVE_ARGV:
        return command(request,request_path,request_sha256,'train-entry','--',*NATIVE_ARGV)
    raise ValueError('Unrecognized native child entry point; no permissive command fallback')


def load_original(request):
    verify_sources(request)
    source = Path(request['source_code']); adapter = Path(request['adapter_code'])
    for name, module in tuple(sys.modules.items()):
        if name.startswith(('hiercp_v1x.v24_','tools.local_cnn_device')) and name not in (
                __name__, 'hiercp_v1x.v24_gpu0_native_storage'):
            filename = getattr(module,'__file__',None)
            if filename and source not in Path(filename).resolve().parents:
                raise ValueError('Original scientific module imported from another checkout: '+name)
    import hiercp_v1x, tools
    hiercp_v1x.__path__ = [str(source/'hiercp_v1x'),str(adapter/'hiercp_v1x')]
    tools.__path__ = [str(source/'tools'),str(adapter/'tools')]
    sys.path.insert(0,str(source))
    storage = importlib.import_module('hiercp_v1x.v24_gpu0_native_storage')
    if Path(storage.__file__).resolve() != adapter/'hiercp_v1x/v24_gpu0_native_storage.py':
        raise ValueError('Storage execution sidecar imported from another checkout')
    storage.install_runtime()
    pipeline = importlib.import_module('hiercp_v1x.v24_matched_basic_cp_pipeline')
    runtime = importlib.import_module('hiercp_v1x.v24_matched_basic_cp_runtime')
    original = importlib.import_module('hiercp_v1x.v24_nnunet_cp')
    device = importlib.import_module('tools.local_cnn_device')
    if pipeline.ROOT != source or not set(pipeline.FILES) <= set(request['source_code_files']):
        raise ValueError('Complete original source/runtime manifest required')
    witness(request['overlay']); witness(request['storage_admission'])
    admission = pipeline.verify_admission(request['storage_admission']['path'])
    if admission['arm_roots'].get('gpu4') != request['execution_root']:
        raise ValueError('Fresh storage output namespace differs from sealed GPU4 source arm')
    return pipeline, runtime, original, device


def install(request, request_path, request_sha256, guard):
    pipeline, runtime, original, device = load_original(request)
    code_objects = {name:getattr(pipeline,name).__code__ for name in (
        'prepare_native','calibrate_native','calibration_worker','train','environment','_child')}
    def early_stage_guard(*,stage_action,gpu=None,native_path=None,chain_root=None):
        if stage_action not in ('prepare-native','calibrate-native','train','clone-debug-worker') or gpu not in (None,4):
            raise ValueError('Unexpected source-arm native stage')
        if native_path is not None and str(native_path) != request['native_path']:
            raise ValueError('Native guard output path differs from sealed experiment')
        if chain_root is not None and str(chain_root) != request['execution_root']:
            raise ValueError('Native guard arm root differs from sealed experiment')
        guard(force=True)
        return guard
    pipeline.early_stage_guard = early_stage_guard
    device.select = guard.select
    real_popen = subprocess.Popen
    def redirected(command_args,*args,**kwargs):
        rewritten = rewrite_child(command_args,request,request_path,request_sha256)
        env = dict(kwargs.get('env') or os.environ)
        env['V24_GPU0_EXECUTION_REQUEST'] = str(request_path)
        env['V24_GPU0_EXECUTION_REQUEST_SHA256'] = request_sha256
        env['V24_READONLY_STORAGE_ADMISSION_SHA256'] = request['storage_admission']['sha256']
        env['CUDA_VISIBLE_DEVICES'] = request['execution_gpu_uuid']
        env.pop('V24_CONTEXT_DEPTH1_CHAIN_REQUEST',None)
        env['PYTHONPATH'] = str(Path(request['adapter_code']))+os.pathsep+env.get('PYTHONPATH','')
        kwargs['env'] = env
        return real_popen(rewritten,*args,**kwargs)
    pipeline.subprocess = types.SimpleNamespace(Popen=redirected,PIPE=subprocess.PIPE,STDOUT=subprocess.STDOUT)
    binding = 'existing_original_snapshot'
    if not hasattr(original,'snapshot'):
        original.snapshot = original.require_project_budget
        binding = 'original.snapshot = original.require_project_budget'
    if any(getattr(pipeline,name).__code__ is not value for name,value in code_objects.items()):
        raise RuntimeError('Original preparation/model/trainer child code object changed')
    return pipeline,runtime,dict(original_native_functions_unchanged=True,resource_snapshot_binding=binding,
        source_arm=4,execution_physical_gpu=0,source_code_files=request['source_code_files'],
        execution_adapter_files=request['adapter_code_files'])


def stage(request,request_path,request_sha256,action,*,native_argv=(),physical_batch=None,output=None):
    guard = ExecutionGuard(request,request_path,request_sha256)
    pipeline,runtime,binding = install(request,request_path,request_sha256,guard)
    os.environ['V24_READONLY_STORAGE_ADMISSION_SHA256'] = request['storage_admission']['sha256']
    if action == 'prepare-native':
        result = pipeline.prepare_native(request['overlay']['path'],Path(request['native_path']).parent,
            request['storage_admission']['path'],gpu=4)
    elif action == 'calibrate-native':
        result = pipeline.calibrate_native(request['native_path'],gpu=4)
    elif action == 'train':
        result = pipeline.train(request['native_path'],gpu=4)
    elif action == 'clone-debug-worker':
        if physical_batch not in (2,4) or output is None or native_argv:
            raise ValueError('Explicit B2/B4 DEBUG clone output required')
        result = pipeline.calibration_worker(request['native_path'],physical_batch=physical_batch,output=output)
    elif action == 'train-entry':
        if list(native_argv) != NATIVE_ARGV:
            raise ValueError('Only unchanged full250/B2/fold0/BEST native CLI accepted')
        sys.argv = [sys.argv[0],*NATIVE_ARGV]
        result = runtime.run_training_entry()
    else:
        raise ValueError('Unknown native execution action')
    if action in ('prepare-native','calibrate-native','train'):
        publish(Path(request['metadata_root'])/(action+'_complete.json'),dict(
            format=STATUS_FORMAT,status='COMPLETE',stage=action,result=None if result is None else str(result),
            request_sha256=request_sha256,resource=guard(force=True),runtime_binding=binding))
    return result


def watch(request,request_path,request_sha256):
    root = owned(request['metadata_root'],directory=True)
    if Path(request_path).parent != root:
        raise ValueError('Controller request must live in its persistent metadata namespace')
    if (root/'lease.json').exists() or (root/'status.json').exists():
        raise FileExistsError('Controller execution namespace already used; no replay')
    current = dict(format=STATUS_FORMAT,status='REGISTERING',source_arm=4,execution_physical_gpu=0,
        execution_gpu_uuid=request['execution_gpu_uuid'],request_sha256=request_sha256,
        controller_pid=os.getpid(),controller_create_time=psutil.Process().create_time(),
        stage=None,stage_log=None,child_pid=None,child_create_time=None,error=None)
    def update(**values):
        current.update(values); current['updated_at_unix'] = time.time()
        publish(root/'status.json',current,replace=True)
    try:
        update()
        deadline = time.monotonic()+120
        while not Path(request['resource_proof_path']).exists():
            if time.monotonic() > deadline:
                raise RuntimeError('Actual controller resource registration did not arrive within120s')
            time.sleep(.2)
        guard = ExecutionGuard(request,request_path,request_sha256)
        if actor(psutil.Process()) != guard.proof['controller_identity']:
            raise ValueError('watch requires the exact resource-proof controller identity')
        publish(root/'lease.json',dict(format=STATUS_FORMAT,request_sha256=request_sha256,
            controller_identity=guard.proof['controller_identity'],resource_proof_sha256=guard.proof_sha))
        update(status='RUNNING')
        verify_sources(request)
        for action,viewstage in (('prepare-native','prepare-native'),('calibrate-native','calibrate-native'),('train','nnunet_training')):
            guard(force=True)
            logfile = root/(action+'.log'); cmd = command(request,request_path,request_sha256,action)
            env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = request['execution_gpu_uuid']
            env.pop('V24_CONTEXT_DEPTH1_CHAIN_REQUEST',None)
            env['PYTHONPATH'] = request['adapter_code']+os.pathsep+env.get('PYTHONPATH','')
            for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
                env[key] = '1'
            env['nnUNet_n_proc_DA']='4'; env['nnUNet_def_n_proc']='4'
            with logfile.open('xb') as stream:
                child = subprocess.Popen(cmd,cwd=request['adapter_code'],env=env,stdout=stream,stderr=subprocess.STDOUT)
            child_actor = psutil.Process(child.pid)
            update(stage=viewstage,stage_action=action,stage_log=str(logfile),child_pid=child.pid,
                child_create_time=child_actor.create_time(),child_command=cmd)
            while child.poll() is None:
                guard(); time.sleep(1)
            guard(force=True)
            if child.returncode != 0:
                raise RuntimeError('Native '+action+' failed ('+str(child.returncode)+'); actual log '+str(logfile))
            receipt = read(root/(action+'_complete.json'))
            if receipt.get('status') != 'COMPLETE' or receipt.get('request_sha256') != request_sha256:
                raise RuntimeError('Actual native child completion receipt missing')
        load_original(request)
        from .v24_gpu0_native_persistence import archive_completed
        archive = archive_completed(request)
        update(status='COMPLETE',stage='nnunet_complete',child_pid=None,child_create_time=None,
            persistent_archive=archive['archive_root'],
            evaluation_checkpoint='checkpoint_best.pth',full_training_epochs=250,full_validation_cases=26)
    except Exception as error:
        update(status='FAILED',error=type(error).__name__+': '+str(error))
        raise
    return root/'status.json'


def dispatch(request_path,request_sha256,action,**kwargs):
    path = owned(request_path)
    if not checksum(request_sha256) or digest(path) != request_sha256:
        raise ValueError('Exact immutable execution request SHA required')
    request = validate_request(read(path)); verify_sources(request)
    return watch(request,path,request_sha256) if action == 'watch' else stage(request,path,request_sha256,action,**kwargs)
