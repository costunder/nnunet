"""Explicit direct GPU execution with truthful host and process admission.

This adapter does not reserve GPUs or claim ownership of their PBS entries.
The owned GPU6 job is read only as the host lifetime witness. All CUDA users
must belong to the exact declared controller tree; no process is signalled.
"""
from __future__ import annotations

import hashlib
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

from .v25_server_resources import AllocationLease as OwnedAllocationLease

DIRECT_FORMAT = 'v25_explicit_direct_GPU_execution_v1'
OWNED_FORMAT = 'v25_owned_GPU_allocation_v1'
GIB = 2**30
DIRECT_GPUS = frozenset((1, 4, 5, 6))
EXPIRY_RESERVE_SECONDS = 180
POLL_SECONDS = 30
_FIELDS = frozenset(('format', 'arm', 'gpu', 'GPU_UUID', 'CPU_affinity', 'UID', 'owner', 'node',
    'PBS_owned_reservation', 'explicit_direct_execution_authorized', 'controller_identity', 'group_root',
    'host_lifetime', 'resource_guard_sha256', 'aggregate_CPU_affinity',
    'aggregate_native_and_cache_RSS_bytes', 'aggregate_GNN_and_cache_RSS_bytes',
    'canonical_cache_reserve_bytes', 'cgroup_admission'))


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checksum(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _integer_list(value, length):
    return (isinstance(value, list) and len(value) == length
        and all(type(item) is int and item >= 0 for item in value)
        and len(set(value)) == length and value == sorted(value))


def _identity_document(value):
    return (isinstance(value, dict) and set(value) == {'pid', 'create_time', 'command', 'UID'}
        and type(value['pid']) is int and value['pid'] > 0
        and type(value['create_time']) in (int, float) and math.isfinite(value['create_time'])
        and value['create_time'] > 0 and value['UID'] == 41833
        and isinstance(value['command'], list) and bool(value['command'])
        and all(isinstance(item, str) and bool(item) and '\x00' not in item for item in value['command']))


def validate_direct_document(value):
    """Validate a closed declaration; this alone is never live GPU admission."""
    if (not isinstance(value, dict) or set(value) != _FIELDS or value.get('format') != DIRECT_FORMAT
            or value.get('arm') not in tuple('ABCD') or type(value.get('gpu')) is not int
            or value['gpu'] not in DIRECT_GPUS or not isinstance(value.get('GPU_UUID'), str)
            or not value['GPU_UUID'].startswith('GPU-') or value.get('UID') != 41833
            or value.get('owner') != 'aicompetition06' or value.get('node') != 'ece-a6gpu6'
            or value.get('PBS_owned_reservation') is not False
            or value.get('explicit_direct_execution_authorized') is not True
            or not _integer_list(value.get('CPU_affinity'), 6)
            or not _integer_list(value.get('aggregate_CPU_affinity'), 24)
            or not set(value['CPU_affinity']) <= set(value['aggregate_CPU_affinity'])
            or not _identity_document(value.get('controller_identity'))
            or not _checksum(value.get('resource_guard_sha256'))
            or value.get('aggregate_native_and_cache_RSS_bytes') != 416*GIB
            or value.get('aggregate_GNN_and_cache_RSS_bytes') != 288*GIB
            or value.get('canonical_cache_reserve_bytes') != 32*GIB
            or not isinstance(value.get('group_root'), str)):
        raise ValueError('Exact authorized four-arm direct GPU/CPU/416GiB host declaration required; no PBS ownership claim')
    lifetime = value['host_lifetime']
    if (not isinstance(lifetime, dict)
            or set(lifetime) != {'allocation_request', 'allocation_request_sha256', 'expires_at_unix'}
            or not isinstance(lifetime['allocation_request'], str)
            or not _checksum(lifetime['allocation_request_sha256'])
            or type(lifetime['expires_at_unix']) not in (int, float)
            or not math.isfinite(lifetime['expires_at_unix'])):
        raise ValueError('Separate unchanged owned GPU6 allocation is required only as host lifetime witness')
    admission = value['cgroup_admission']
    if (not isinstance(admission, dict) or set(admission) != {'controller_cgroup', 'memory_limits'}
            or not isinstance(admission['controller_cgroup'], str) or not admission['controller_cgroup']
            or not isinstance(admission['memory_limits'], list)):
        raise ValueError('Actual controller cgroup and ancestor memory-limit observation required')
    for row in admission['memory_limits']:
        if (not isinstance(row, dict) or set(row) != {'path', 'limit_bytes'}
                or not isinstance(row['path'], str)
                or not row['path'].startswith('/sys/fs/cgroup/')
                or row['limit_bytes'] is not None and (type(row['limit_bytes']) is not int
                    or row['limit_bytes'] < 416*GIB)):
            raise ValueError('Direct execution cannot exceed an actual cgroup memory limit')
    return value


def _owned_regular(path, *, directory=False):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('Absolute own non-symlink direct proof or group path required')
    value = path.stat()
    mode = stat.S_ISDIR if directory else stat.S_ISREG
    if not mode(value.st_mode) or value.st_uid != 41833 or os.getuid() != 41833:
        raise ValueError('Actual own Linux UID41833 direct proof/group required')
    return path


def capture_cgroup_admission(pid):
    """Read Linux cgroup v1/v2 ancestor memory limits without changing them."""
    body = Path('/proc')/str(pid)/'cgroup'
    text = body.read_text()
    limits = {}
    for line in text.splitlines():
        hierarchy, controllers, relative = line.split(':', 2)
        if hierarchy == '0' and controllers == '':
            root = Path('/sys/fs/cgroup'); name = 'memory.max'
        elif 'memory' in controllers.split(','):
            root = Path('/sys/fs/cgroup/memory'); name = 'memory.limit_in_bytes'
        else:
            continue
        if not relative.startswith('/') or '..' in Path(relative).parts:
            raise ValueError('Actual cgroup path must be absolute without traversal')
        folder = root/relative.lstrip('/')
        while folder == root or root in folder.parents:
            path = folder/name
            if path.is_file():
                raw = path.read_text().strip()
                if raw == 'max':
                    limit = None
                elif raw.isdigit():
                    limit = int(raw)
                else:
                    raise ValueError('Actual cgroup memory limit is not a byte count or max')
                limits[str(path)] = limit
            if folder == root:
                break
            folder = folder.parent
    if not limits:
        raise RuntimeError('Actual Linux cgroup memory limits could not be read; no assumed host-only fallback')
    return dict(controller_cgroup=text,
        memory_limits=[dict(path=path, limit_bytes=limits[path]) for path in sorted(limits)])


def _actor(process):
    if process.status() == psutil.STATUS_ZOMBIE:
        raise RuntimeError('Direct managed actor is a zombie')
    return dict(pid=process.pid, create_time=process.create_time(), command=process.cmdline(), UID=process.uids().real)


def _exact(identity):
    if not _identity_document(identity):
        raise ValueError('Exact own PID/birth/argv/UID controller identity required')
    process = psutil.Process(identity['pid'])
    if _actor(process) != identity:
        raise RuntimeError('Exact direct controller or child identity changed')
    return process


def _query(command):
    return subprocess.check_output(command, text=True, timeout=20).strip()


def _pbs_start_time(value):
    """PBS JSON may expose Unix seconds or the server-local PBS date text."""
    if type(value) in (int, float):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            try:
                result = time.mktime(time.strptime(value, '%a %b %d %H:%M:%S %Y'))
            except (ValueError, OverflowError) as error:
                raise RuntimeError('Actual PBS start time must be Unix seconds or valid server-local PBS date text') from error
    else:
        raise RuntimeError('Actual PBS start time must be Unix seconds or valid server-local PBS date text')
    if not math.isfinite(result) or result <= 0:
        raise RuntimeError('Actual PBS start time must be finite positive seconds')
    return result


class DirectAllocationLease:
    """Compatibility interface for unchanged v2.5 stage entry points."""
    def __init__(self, path):
        self.path = _owned_regular(path)
        self.bytes = self.path.read_bytes()
        self.request = validate_direct_document(json.loads(self.bytes))
        self.group = _owned_regular(self.request['group_root'], directory=True)
        _owned_regular(self.group/self.request['arm'], directory=True)
        self.guard_path = Path(__file__).resolve()
        if _digest(self.guard_path) != self.request['resource_guard_sha256']:
            raise ValueError('Tracked direct resource guard SHA changed')
        self.lifetime_path = _owned_regular(self.request['host_lifetime']['allocation_request'])
        if _digest(self.lifetime_path) != self.request['host_lifetime']['allocation_request_sha256']:
            raise ValueError('Original owned host-lifetime allocation SHA changed')
        self.lifetime = json.loads(self.lifetime_path.read_bytes())
        lifetime = self.lifetime
        if (lifetime.get('format') != OWNED_FORMAT or lifetime.get('UID') != 41833
                or lifetime.get('owner') != self.request['owner'] or lifetime.get('node') != self.request['node']
                or lifetime.get('gpu') != 6
                or not re.fullmatch(r'\d+\.ECE-util1', lifetime.get('job_id', ''))
                or lifetime.get('expires_at_unix') != self.request['host_lifetime']['expires_at_unix']):
            raise ValueError('Genuine unchanged owned GPU6 host lifetime proof required; direct GPU ownership remains false')
        self.self_identity = _actor(psutil.Process())
        self.managed = {}
        self.last_verified = None
        self.monotonic_expiry = time.monotonic()+lifetime['expires_at_unix']-time.time()
        self.actual_resources = None
        self.verify(force=True)

    def _tree(self):
        controller = _exact(self.request['controller_identity'])
        own = _exact(self.self_identity)
        tree = {controller.pid: controller}
        for child in controller.children(recursive=True):
            try:
                identity = _actor(child)
            except psutil.NoSuchProcess:
                continue
            if identity['UID'] != 41833:
                raise RuntimeError('Foreign-UID controller descendant cannot be an admitted direct actor')
            previous = self.managed.get(child.pid)
            if previous is not None and previous != identity:
                raise RuntimeError('Managed direct child PID/birth/argv identity changed')
            self.managed[child.pid] = identity
            tree[child.pid] = child
        if own.pid not in tree:
            raise RuntimeError('Direct stage must be the exact declared controller or its actual descendant')
        if controller.cpu_affinity() != self.request['aggregate_CPU_affinity']:
            raise RuntimeError('Controller aggregate24 CPU affinity changed')
        if own.cpu_affinity() != self.request['CPU_affinity'] and own.pid != controller.pid:
            raise RuntimeError('Own direct stage six-CPU affinity differs from its arm proof')
        return controller, own, tree

    def _host_lifetime(self):
        jobs = json.loads(_query(['/opt/pbs/bin/qstat', '-f', '-F', 'json', self.lifetime['job_id']]))['Jobs']
        if list(jobs) != [self.lifetime['job_id']]:
            raise RuntimeError('Exact owned host-lifetime job is unavailable')
        job = next(iter(jobs.values()))
        if (job.get('job_state') != 'R' or job.get('Job_Owner', '').split('@')[0] != 'aicompetition06'
                or int(job.get('Resource_List', {}).get('ngpus', 0)) != 1
                or any(part.split('/')[0] != self.request['node'] for part in job.get('exec_host', '').split('+'))):
            raise RuntimeError('Owned job is only a current host lifetime witness; its identity changed')
        hours, minutes, seconds = map(int, job['Resource_List']['walltime'].split(':'))
        expiration = _pbs_start_time(job['stime'])+hours*3600+minutes*60+seconds
        if abs(expiration-self.lifetime['expires_at_unix']) > 1:
            raise RuntimeError('Actual host lifetime differs from its sealed start/walltime proof')

    def _resources(self, controller, own, tree):
        declared = self.request
        observed = capture_cgroup_admission(controller.pid)
        if observed != declared['cgroup_admission']:
            raise RuntimeError('Observed direct controller cgroup or ancestor memory limit changed')
        current = capture_cgroup_admission(own.pid)
        for row in current['memory_limits']:
            if row['limit_bytes'] is not None and row['limit_bytes'] < 416*GIB:
                raise MemoryError('Actual direct stage cgroup cannot admit four native96GiB plus shared32GiB')
        rss = 0
        for process in tree.values():
            try:
                if process.status() != psutil.STATUS_ZOMBIE:
                    rss += process.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        if rss > 384*GIB:
            raise MemoryError('Aggregate exact direct controller tree exceeds four96GiB process budget')
        memory = psutil.virtual_memory()
        if memory.total < 416*GIB or memory.available+rss < 416*GIB:
            raise MemoryError('Actual available host RAM cannot admit aggregate416GiB; no resource or data shrink fallback')
        self.actual_resources = dict(aggregate_process_and_children_RSS_bytes=rss,
            aggregate_process_and_children_RSS_limit_bytes=384*GIB,
            canonical_cache_memory_reserve_bytes=32*GIB,
            available_host_RAM_bytes=memory.available, host_RAM_bytes=memory.total,
            actual_controller_cgroup=observed, actual_stage_cgroup=current,
            PBS_GPU_owned_reservation=False, PBS_job_role='host_lifetime_only')

    def _gpu(self, tree, *, initial):
        value = self.request
        row = _query(['nvidia-smi', '--id='+str(value['gpu']),
            '--query-gpu=index,uuid,memory.total,memory.free,mig.mode.current', '--format=csv,noheader,nounits'])
        fields = [item.strip() for item in row.split(',')]
        if (len(fields) != 5 or fields[0] != str(value['gpu']) or fields[1] != value['GPU_UUID']
                or fields[4] not in ('N/A', 'Disabled', '[N/A]')
                or not math.isfinite(float(fields[2])) or float(fields[2]) < 40*1024
                or not math.isfinite(float(fields[3])) or initial and float(fields[3]) < 40*1024):
            raise RuntimeError('Exact full physical GPU UUID/MIG/initial40GiB free admission failed')
        apps = _query(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'])
        for line in apps.splitlines():
            if not line.strip():
                continue
            pid, uuid = [item.strip() for item in line.split(',')]
            if not pid.isdigit():
                raise RuntimeError('Actual CUDA process query did not contain an exact PID')
            if uuid == value['GPU_UUID'] and int(pid) not in tree:
                raise RuntimeError('Foreign or unrelated CUDA process on direct target GPU: '+pid)

    def verify(self, *, force=False):
        if (self.path.read_bytes() != self.bytes or _digest(self.guard_path) != self.request['resource_guard_sha256']
                or _digest(self.lifetime_path) != self.request['host_lifetime']['allocation_request_sha256']):
            raise ValueError('Immutable direct request/guard/host lifetime proof changed')
        if os.getuid() != 41833 or socket.gethostname().split('.')[0] != self.request['node']:
            raise RuntimeError('Actual direct process UID or host differs')
        remaining = min(self.lifetime['expires_at_unix']-time.time(), self.monotonic_expiry-time.monotonic())
        if remaining < EXPIRY_RESERVE_SECONDS:
            raise RuntimeError('Host lifetime has less than180seconds; preserve exact checkpoint and resume')
        controller = _exact(self.request['controller_identity'])
        own = _exact(self.self_identity)
        if controller.cpu_affinity() != self.request['aggregate_CPU_affinity']:
            raise RuntimeError('Controller aggregate24 CPU affinity changed')
        if own.pid != controller.pid and own.cpu_affinity() != self.request['CPU_affinity']:
            raise RuntimeError('Own direct stage six-CPU affinity differs from its arm proof')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible not in ('', str(self.request['gpu']), self.request['GPU_UUID']):
            raise RuntimeError('CUDA visibility differs from the exact direct physical GPU')
        torch = sys.modules.get('torch')
        if not visible and torch is not None and torch.cuda.is_initialized():
            raise RuntimeError('Direct GPU must be selected before CUDA initialization')
        now = time.monotonic()
        if force or self.last_verified is None or now-self.last_verified >= POLL_SECONDS:
            controller, own, tree = self._tree()
            self._host_lifetime()
            self._resources(controller, own, tree)
            self._gpu(tree, initial=self.last_verified is None)
            self.last_verified = now
        return self.actual_resources

    def select_cuda(self):
        self.verify()
        os.environ['CUDA_VISIBLE_DEVICES'] = self.request['GPU_UUID']
        return self.actual_resources

    def install_scorer_guard(self, scorer):
        self.verify(force=True)
        original = scorer._encode
        def encode(instance, cpu, *, training):
            self.verify()
            return original(cpu, training=training)
        scorer._encode = types.MethodType(encode, scorer)
        return lambda: setattr(scorer, '_encode', original)

    def install_preparation_guard(self, factory):
        original = factory.V25NativeInputs._rss
        lease = self
        def checked(instance, *args, **kwargs):
            lease.verify()
            return original(instance, *args, **kwargs)
        factory.V25NativeInputs._rss = checked


def validate_resource_proof(path, *, gpu=None, cpu_affinity=None, force=True, initial_cuda=False):
    """Return a live direct or original owned guard, preserving both meanings."""
    value = json.loads(Path(path).read_bytes())
    if value.get('format') == DIRECT_FORMAT:
        guard = DirectAllocationLease(path)
    elif value.get('format') == OWNED_FORMAT:
        guard = OwnedAllocationLease(path)
    else:
        raise ValueError('Explicit direct execution or genuine owned allocation proof required')
    if gpu is not None and guard.request['gpu'] != gpu:
        raise ValueError('Exact resource proof physical GPU differs from the selected stage')
    if cpu_affinity is not None and guard.request['CPU_affinity'] != list(cpu_affinity):
        raise ValueError('Exact resource proof CPU set differs from the selected stage')
    guard.verify(force=force)
    if initial_cuda:
        if isinstance(guard, DirectAllocationLease):
            guard.select_cuda()
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = guard.request['GPU_UUID']
    return guard
