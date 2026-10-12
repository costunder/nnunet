"""Read-only lease for each explicitly authorized Basic CP single-factor arm.

The original owned GPU6 job supplies the host lifetime only. This guard does
not reserve GPU1 or GPU4 and never signals, launches, or modifies a process.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys
import time

import psutil

FORMAT = 'basic_cp_one_factor_experiment_request_v1'
ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30
UID = 41833
NODE = 'ece-a6gpu6'
CPUS = {1: list(range(16, 22)), 4: list(range(6)), 6: list(range(32, 38))}
POLL_SECONDS = 5
EXPIRY_RESERVE_SECONDS = 180
RSS_LIMIT = 96 * GIB
AGGREGATE_CAPACITY = 416 * GIB
STORAGE_FLOOR = 10 * GIB
FROZEN_SCIENCE_FILES = ('custom_trainers/onlinecp_raw_bank.py',
    'custom_trainers/onlinecp_raw_resampling.py', 'tools/online_raw_bank_preparation.py',
    'hiercp_v1x/v24_lossless_raw_storage.py')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            value.update(block)
    return value.hexdigest()


def _checksum(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _identity(value):
    return (isinstance(value, dict) and set(value) == {'pid', 'create_time', 'UID', 'command'}
        and type(value['pid']) is int and value['pid'] > 0 and value['UID'] == UID
        and type(value['create_time']) in (int, float) and math.isfinite(value['create_time'])
        and value['create_time'] > 0 and isinstance(value['command'], list)
        and bool(value['command']) and all(isinstance(x, str) and x and '\x00' not in x
            for x in value['command']))


def validate_request(value):
    required = {'format', 'uid', 'root', 'native_root', 'gpu', 'gpu_uuid',
        'cpu_affinity', 'controller_identity', 'code_sha256', 'bank', 'bank_sha256',
        'host_lifetime', 'runtime_reserve_bytes', 'RSS_limit_bytes', 'GPU_memory_total_MiB'}
    if (not isinstance(value, dict) or not required <= set(value)
            or value['format'] != FORMAT or value['uid'] != UID
            or type(value['gpu']) is not int or value['gpu'] not in CPUS
            or value['cpu_affinity'] != CPUS[value['gpu']]
            or not isinstance(value['gpu_uuid'], str) or not value['gpu_uuid'].startswith('GPU-')
            or type(value['GPU_memory_total_MiB']) not in (int, float)
            or not math.isfinite(value['GPU_memory_total_MiB'])
            or value['GPU_memory_total_MiB'] < 44 * 1024
            or not _identity(value['controller_identity'])
            or value['runtime_reserve_bytes'] != STORAGE_FLOOR
            or value['RSS_limit_bytes'] != RSS_LIMIT):
        raise ValueError('Own fresh controller, exact physical GPU/six CPUs/96GiB/full-storage contract required')
    for name in ('root', 'native_root', 'bank'):
        path = Path(value[name])
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('Absolute experiment/native/bank paths without traversal required')
    if not Path(value['native_root']).is_relative_to(Path('/dev/shm')):
        raise ValueError('Full native checkpoints and results require the admitted tmpfs namespace')
    code = value['code_sha256']
    if (not isinstance(code, dict) or not code or any(not isinstance(p, str)
            or Path(p).is_absolute() or '..' in Path(p).parts or not _checksum(h)
            for p, h in code.items())
            or 'hiercp_v1x/basic_cp_ablation_resources.py' not in code
            or not _checksum(value['bank_sha256'])):
        raise ValueError('Original Basic bank and every private runtime guard require exact SHA admissions')
    lifetime = value['host_lifetime']
    if (not isinstance(lifetime, dict) or set(lifetime) != {'path', 'sha256', 'expires_at_unix'}
            or not isinstance(lifetime['path'], str) or not Path(lifetime['path']).is_absolute()
            or '..' in Path(lifetime['path']).parts or not _checksum(lifetime['sha256'])
            or type(lifetime['expires_at_unix']) not in (int, float)
            or not math.isfinite(lifetime['expires_at_unix'])):
        raise ValueError('Unchanged owned GPU6 job is required as a separate host-lifetime witness')
    if ('source_bank' in value) != ('source_bank_sha256' in value):
        raise ValueError('Original bank path and SHA must be admitted together')
    if 'source_bank' in value:
        original = Path(value['source_bank'])
        if (not original.is_absolute() or '..' in original.parts
                or not _checksum(value['source_bank_sha256'])):
            raise ValueError('Admitted256bank retains a separately pinned original128bank')
    if ('readonly_admission' in value) != ('readonly_admission_sha256' in value):
        raise ValueError('Original full read-only admission path and SHA must be pinned together')
    if 'readonly_admission' in value:
        admission = Path(value['readonly_admission'])
        if (not admission.is_absolute() or '..' in admission.parts
                or not _checksum(value['readonly_admission_sha256'])):
            raise ValueError('Exact original read-only admission path/SHA required')
    return value


def _owned(path, *, directory=False):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Absolute non-symlink experiment evidence required')
    row = path.stat()
    if row.st_uid != UID or not (stat.S_ISDIR(row.st_mode) if directory else stat.S_ISREG(row.st_mode)):
        raise ValueError('Own UID41833 regular evidence or private namespace required')
    return path


def _snapshot(path):
    row = Path(path).stat()
    return (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns,
        row.st_uid, row.st_gid, row.st_mode)


def _metadata(process):
    return dict(pid=process.pid, create_time=process.create_time(), UID=process.uids().real)


def _fork_no_exec(process):
    """Linux records an inherited fork image until that child first executes.

    Field9 is the kernel task flags. PF_FORKNOEXEC is cleared by the first
    exec, so a parent's temporarily inherited argv is not a final child argv.
    The caller binds PID/birth/UID before and after reading this observation.
    """
    try:
        body = (Path('/proc') / str(process.pid) / 'stat').read_text()
    except FileNotFoundError as failure:
        if process.is_running():
            raise RuntimeError('Live Basic descendant lost its Linux fork/exec observation') from failure
        raise psutil.NoSuchProcess(process.pid) from failure
    prefix, separator, tail = body.rpartition(') ')
    fields = tail.split()
    if (not separator or not prefix.startswith(str(process.pid) + ' (')
            or len(fields) < 7 or not fields[6].isdigit()):
        raise RuntimeError('Exact Linux descendant fork/exec observation required')
    return bool(int(fields[6]) & 0x40)


def _actor(process):
    if process.status() == psutil.STATUS_ZOMBIE:
        raise RuntimeError('Exact Basic controller or stage is already terminal')
    value = dict(_metadata(process), command=process.cmdline())
    if not _identity(value):
        raise RuntimeError('Live Basic process requires exact nonempty argv/PID/birth/UID')
    return value


def _exact(identity):
    process = psutil.Process(identity['pid'])
    if _actor(process) != identity:
        raise RuntimeError('Exact Basic controller or stage PID/birth/UID/argv changed')
    return process


def capture_cgroup_admission(pid):
    """Read all memory ancestor limits; never infer a missing limit from RAM."""
    body = (Path('/proc') / str(pid) / 'cgroup').read_text()
    limits = {}
    for line in body.splitlines():
        hierarchy, controllers, relative = line.split(':', 2)
        if hierarchy == '0' and controllers == '':
            root, filename = Path('/sys/fs/cgroup'), 'memory.max'
        elif 'memory' in controllers.split(','):
            root, filename = Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes'
        else:
            continue
        if not relative.startswith('/') or '..' in Path(relative).parts:
            raise ValueError('Actual cgroup membership path is invalid')
        folder = root / relative.lstrip('/')
        while folder == root or root in folder.parents:
            path = folder / filename
            if path.is_file():
                raw = path.read_text().strip()
                if raw == 'max':
                    limit = None
                elif raw.isdigit():
                    limit = int(raw)
                else:
                    raise ValueError('Actual cgroup limit must contain bytes or max')
                limits[str(path)] = limit
            if folder == root:
                break
            folder = folder.parent
    if not limits:
        raise RuntimeError('Actual inherited cgroup memory limits unavailable; no assumed capacity fallback')
    return dict(controller_cgroup=body,
        memory_limits=[dict(path=p, limit_bytes=limits[p]) for p in sorted(limits)])


def _query(command):
    return subprocess.check_output(command, text=True, timeout=20).strip()


def _pbs_start(value):
    if type(value) in (int, float):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            result = time.mktime(time.strptime(value, '%a %b %d %H:%M:%S %Y'))
    else:
        raise RuntimeError('Owned PBS host-lifetime start time is unavailable')
    if not math.isfinite(result) or result <= 0:
        raise RuntimeError('Owned PBS host-lifetime start time must be finite positive seconds')
    return result


class BasicLease:
    """Hash admitted bytes once, check metadata per batch, poll resources at5s."""

    def __init__(self, request):
        self.request = copy.deepcopy(validate_request(request))
        self.request_bytes = json.dumps(self.request, sort_keys=True, separators=(',', ':'))
        self._files = {}
        _owned(self.request['root'], directory=True)
        _owned(self.request['native_root'], directory=True)
        for relative, digest in self.request['code_sha256'].items():
            self._admit(ROOT / relative, digest)
        self._admit(self.request['bank'], self.request['bank_sha256'])
        if 'source_bank' in self.request:
            self._admit(self.request['source_bank'], self.request['source_bank_sha256'])
        if 'readonly_admission' in self.request:
            self._admit_frozen_storage()
        ref = self.request['host_lifetime']
        self._admit(ref['path'], ref['sha256'])
        self.lifetime = json.loads(Path(ref['path']).read_bytes())
        value = self.lifetime
        if (value.get('format') != 'v25_owned_GPU_allocation_v1' or value.get('UID') != UID
                or value.get('owner') != 'aicompetition06' or value.get('node') != NODE
                or value.get('gpu') != 6 or not re.fullmatch(r'\d+\.ECE-util1', value.get('job_id', ''))
                or value.get('expires_at_unix') != ref['expires_at_unix']):
            raise ValueError('Genuine unchanged original GPU6-owned host lifetime required; otherGPU ownership is not claimed')
        self.self_identity = _actor(psutil.Process())
        self.monotonic_expiry = time.monotonic() + value['expires_at_unix'] - time.time()
        self.last_verified = None
        self.actual_resources = None
        self._controller_cgroup = None
        self._managed = {}
        self._fork_pending = {}
        self._cached_tree = {}

    def _admit(self, path, digest):
        path = _owned(path)
        before = _snapshot(path)
        if sha(path) != digest or _snapshot(path) != before:
            raise ValueError('Original admitted Basic source/code/lifetime changed: ' + str(path))
        self._files[str(path)] = before

    def _admit_frozen_storage(self):
        """Keep original cache provenance; new ablation code never replaces it."""
        path = self.request['readonly_admission']
        self._admit(path, self.request['readonly_admission_sha256'])
        document = json.loads(Path(path).read_bytes())
        if (document.get('format') != 'v24_native_immutable_full_cache_storage_v1'
                or document.get('model_data_scale_preserved') is not True
                or document.get('debug') is not False or document.get('source_arrays_written') is not False):
            raise ValueError('Actual original complete read-only native admission required')
        witness = document['adapter_source']
        adapter = Path(witness['path'])
        if (adapter.name != 'v24_readonly_native_storage.py' or adapter.parent.name != 'hiercp_v1x'
                or not _checksum(witness['sha256'])):
            raise ValueError('Exact original frozen read-only adapter path and SHA required')
        self._admit(adapter, witness['sha256'])
        scientific = document['source_bank']['scientific_source_files_sha256']
        if set(scientific) != set(FROZEN_SCIENCE_FILES) or any(
                not _checksum(v) for v in scientific.values()):
            raise ValueError('All four original scientific preparation sources must remain pinned')
        original_root = adapter.parents[1]
        for relative, digest in scientific.items():
            self._admit(original_root / relative, digest)

    def _inputs(self):
        if json.dumps(self.request, sort_keys=True, separators=(',', ':')) != self.request_bytes:
            raise ValueError('Admitted Basic experiment request changed in memory')
        for path, observed in self._files.items():
            if _snapshot(_owned(path)) != observed:
                raise ValueError('Original admitted Basic source/code/lifetime metadata changed: ' + path)

    def _descendant(self, process):
        """A genuinely terminal worker may disappear without pretending it is live."""
        previous = self._managed.get(process.pid)
        for attempt in range(3):
            metadata = _metadata(process)
            if metadata['UID'] != UID or previous is not None and any(
                    metadata[k] != previous[k] for k in ('pid', 'create_time', 'UID')):
                raise RuntimeError('Basic descendant PID/birth/UID changed or became foreign')
            if process.status() == psutil.STATUS_ZOMBIE:
                return None
            fork_before = _fork_no_exec(process)
            command = process.cmdline()
            if process.status() == psutil.STATUS_ZOMBIE:
                return None
            if _fork_no_exec(process) != fork_before:
                if attempt < 2:
                    time.sleep(.01)
                    continue
                raise RuntimeError('Descendant first exec did not stabilize during bounded observation')
            if not command:
                if attempt < 2:
                    time.sleep(.01)
                    continue
                raise RuntimeError('Basic live descendant retained unknown empty argv')
            pending = self._fork_pending.get(process.pid)
            if fork_before:
                parent = process.parent()
                if parent is None:
                    raise RuntimeError('Unexecuted Basic fork lost its exact parent lineage')
                inherited = _actor(parent)
                if command != inherited['command'] or pending is not None and pending != inherited:
                    raise RuntimeError('Unexecuted Basic fork changed its exact inherited parent argv/identity')
                if previous is not None and pending is None:
                    raise RuntimeError('Finalized Basic descendant regressed to an unexecuted fork')
            else:
                inherited = None
                if pending is not None:
                    parent = process.parent()
                    if parent is None or _actor(parent) != pending:
                        raise RuntimeError('First Basic child exec changed its exact parent lineage')
                elif previous is not None and command != previous['command']:
                    raise RuntimeError('Basic descendant live argv changed')
            identity = dict(metadata, command=command)
            final_metadata = _metadata(process)
            final_command = process.cmdline()
            final_fork = _fork_no_exec(process)
            if (fork_before and not final_fork and final_metadata == metadata and attempt < 2):
                time.sleep(.01)
                continue
            if (not _identity(identity) or final_metadata != metadata
                    or final_command != command or final_fork != fork_before):
                raise RuntimeError('Exact Basic descendant changed during live observation')
            self._managed[process.pid] = identity
            if fork_before:
                self._fork_pending[process.pid] = inherited
            else:
                self._fork_pending.pop(process.pid, None)
            return process
        raise RuntimeError('Exact Basic worker observation could not complete')

    def _tree(self, controller, own):
        tree = {controller.pid: controller}
        for process in controller.children(recursive=True):
            try:
                process = self._descendant(process)
                if process is not None:
                    if process.cpu_affinity() != self.request['cpu_affinity']:
                        raise RuntimeError('Basic descendant left the exact six-CPU allocation')
                    tree[process.pid] = process
            except psutil.NoSuchProcess:
                continue
        if own.pid not in tree:
            raise RuntimeError('Basic stage is outside its exact authorized fresh controller tree')
        return tree

    def _host_lifetime(self):
        jobs = json.loads(_query(['/opt/pbs/bin/qstat', '-f', '-F', 'json', self.lifetime['job_id']]))['Jobs']
        if list(jobs) != [self.lifetime['job_id']]:
            raise RuntimeError('Exact owned host-lifetime job unavailable')
        value = next(iter(jobs.values()))
        hosts = value.get('exec_host', '').split('+')
        if (value.get('job_state') != 'R' or value.get('Job_Owner', '').split('@')[0] != 'aicompetition06'
                or int(value.get('Resource_List', {}).get('ngpus', 0)) != 1
                or not hosts or any(x.split('/')[0] != NODE for x in hosts)):
            raise RuntimeError('Owned PBS job no longer proves the admitted current host lifetime')
        duration = value['Resource_List']['walltime'].split(':')
        if len(duration) != 3 or any(not x.isdigit() for x in duration):
            raise RuntimeError('Actual PBS host-lifetime walltime is invalid')
        hours, minutes, seconds = map(int, duration)
        expiration = _pbs_start(value['stime']) + hours * 3600 + minutes * 60 + seconds
        if abs(expiration - self.lifetime['expires_at_unix']) > 1:
            raise RuntimeError('Actual PBS start/walltime deadline changed from its original immutable proof')

    def _rss(self, tree):
        total = 0
        for process in tree.values():
            try:
                if process.status() != psutil.STATUS_ZOMBIE:
                    if process.uids().real != UID:
                        raise RuntimeError('Foreign UID appeared in the admitted Basic process tree')
                    total += process.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        if total > RSS_LIMIT:
            raise MemoryError('Own Basic native controller/worker tree exceeds its96GiB RSS contract')
        return total

    def _resources(self, controller, own, tree):
        observed = capture_cgroup_admission(controller.pid)
        if self._controller_cgroup is None:
            self._controller_cgroup = observed
        elif observed != self._controller_cgroup:
            raise RuntimeError('Exact Basic controller cgroup/ancestor memory limits changed')
        declared = self.request.get('controller_cgroup_admission')
        if declared is not None and declared != observed:
            raise RuntimeError('Fresh Basic controller differs from its observed cgroup admission')
        current = capture_cgroup_admission(own.pid)
        for admission in (observed, current):
            for row in admission['memory_limits']:
                if row['limit_bytes'] is not None and row['limit_bytes'] < AGGREGATE_CAPACITY:
                    raise MemoryError('Actual inherited cgroup cannot admit four96GiB plus shared32GiB')
        rss = self._rss(tree)
        memory = psutil.virtual_memory()
        if memory.total < AGGREGATE_CAPACITY or memory.available + rss < AGGREGATE_CAPACITY:
            raise MemoryError('Actual host RAM cannot admit aggregate416GiB without changing experiment scale')
        free = shutil.disk_usage(self.request['native_root']).free
        if free < STORAGE_FLOOR:
            raise OSError('Full native tmpfs output has less than its admitted10GiB free floor')
        return dict(own_process_tree_RSS_bytes=rss, own_process_tree_RSS_limit_bytes=RSS_LIMIT,
            aggregate_native_and_cache_capacity_bytes=AGGREGATE_CAPACITY,
            canonical_cache_reserve_bytes=32 * GIB, host_RAM_bytes=memory.total,
            host_available_RAM_bytes=memory.available, native_free_bytes=free,
            native_free_floor_bytes=STORAGE_FLOOR, actual_controller_cgroup=observed,
            actual_stage_cgroup=current, GPU5_guard_modified=False, processes_signaled=False,
            PBS_GPU_owned_reservation=False, PBS_job_role='host_lifetime_only',
            own_GPU=self.request['gpu'], own_CPU_affinity=self.request['cpu_affinity'])

    def _gpu(self, tree, *, idle):
        value = _query(['nvidia-smi', '--id=' + str(self.request['gpu']),
            '--query-gpu=index,uuid,name,memory.total,memory.free,mig.mode.current',
            '--format=csv,noheader,nounits'])
        row = [x.strip() for x in value.split(',')]
        if (len(row) != 6 or row[0] != str(self.request['gpu']) or row[1] != self.request['gpu_uuid']
                or row[5] not in ('N/A', 'Disabled', '[N/A]') or not math.isfinite(float(row[3]))
                or float(row[3]) != self.request['GPU_memory_total_MiB']
                or not math.isfinite(float(row[4]))
                or idle and float(row[4]) < 40 * 1024):
            raise RuntimeError('Exact full GPU UUID/MIG/40GiB initial free admission failed')
        apps = _query(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'])
        own_pids = []
        for line in apps.splitlines():
            if not line.strip():
                continue
            cells = [x.strip() for x in line.split(',')]
            if len(cells) != 2 or not cells[0].isdigit():
                raise RuntimeError('Actual CUDA actors require exact numeric PID evidence')
            pid, uuid = int(cells[0]), cells[1]
            if uuid == self.request['gpu_uuid']:
                if idle or pid not in tree:
                    raise RuntimeError('Target Basic GPU is occupied by a non-admitted CUDA actor')
                own_pids.append(pid)
        return dict(GPU_name=row[2], GPU_UUID=row[1], GPU_total_MiB=float(row[3]),
            GPU_free_MiB=float(row[4]), MIG=row[5], own_GPU_compute_PIDs=own_pids)

    def verify(self, *, force=False, idle=False):
        self._inputs()
        if os.getuid() != UID or socket.gethostname().split('.')[0] != NODE:
            raise RuntimeError('Actual Basic execution UID or node changed')
        remaining = min(self.lifetime['expires_at_unix'] - time.time(),
            self.monotonic_expiry - time.monotonic())
        if remaining < EXPIRY_RESERVE_SECONDS:
            raise RuntimeError('Host lifetime has less than180seconds; preserve actual checkpoint and resume')
        controller = _exact(self.request['controller_identity'])
        own = _exact(self.self_identity)
        if controller.cpu_affinity() != self.request['cpu_affinity'] or own.cpu_affinity() != self.request['cpu_affinity']:
            raise RuntimeError('Exact authorized Basic six-CPU affinity changed')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible not in ('', str(self.request['gpu']), self.request['gpu_uuid']):
            raise RuntimeError('CUDA visibility differs from the admitted Basic physical GPU')
        torch = sys.modules.get('torch')
        if not visible and torch is not None and torch.cuda.is_initialized():
            raise RuntimeError('Basic physical GPU must be selected before CUDA initialization')
        now = time.monotonic()
        if force or idle or self.last_verified is None or now - self.last_verified >= POLL_SECONDS:
            tree = self._tree(controller, own)
            self._host_lifetime()
            resources = self._resources(controller, own, tree)
            resources.update(self._gpu(tree, idle=idle))
            self._cached_tree = tree
            self.actual_resources = resources
            self.last_verified = now
        else:
            self.actual_resources['own_process_tree_RSS_bytes'] = self._rss(self._cached_tree)
            free = shutil.disk_usage(self.request['native_root']).free
            if free < STORAGE_FLOOR:
                raise OSError('Full native tmpfs output dropped below its10GiB free floor')
            self.actual_resources['native_free_bytes'] = free
        self.actual_resources['host_lifetime_remaining_seconds'] = remaining
        self.actual_resources['host_lifetime_expires_at_unix'] = self.lifetime['expires_at_unix']
        self.actual_resources['host_lifetime_job_id'] = self.lifetime['job_id']
        return dict(self.actual_resources)
