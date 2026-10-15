"""Read-only resource evidence for explicitly authorized direct MIG execution.

This module does not reserve devices or claim PBS CPU/GPU ownership. It never
initializes CUDA, changes affinity/cgroups, or signals a process.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import re
import socket
import stat
import time

GIB = 2 ** 30
UID = 41833
# Linux sets PF_EXITING before exit_mm(), while the task can still report R.
# https://github.com/torvalds/linux/blob/v5.15/include/linux/sched.h#L1569
# https://github.com/torvalds/linux/blob/v5.15/kernel/exit.c
PF_EXITING = 0x00000004


def _positive(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(name + ' must be an explicit integer byte/count value')
    return value


def parse_CPU_list(text):
    result = set()
    for piece in text.strip().split(','):
        if not piece:
            continue
        if '-' in piece:
            first, last = piece.split('-', 1)
            if not first.isdigit() or not last.isdigit() or int(last) < int(first):
                raise ValueError('Invalid actual Linux CPU range: ' + piece)
            result.update(range(int(first), int(last) + 1))
        elif piece.isdigit():
            result.add(int(piece))
        else:
            raise ValueError('Invalid actual Linux CPU: ' + piece)
    return sorted(result)


def _proc_stat(text, pid):
    """Read identity/exit flags without splitting the parenthesized comm field."""
    close = text.rfind(')')
    if close < 0 or text.split('(', 1)[0].strip() != str(pid):
        raise RuntimeError('Actual proc stat PID/comm is invalid: ' + str(pid))
    fields = text[close+1:].split()
    if len(fields) < 20 or len(fields[0]) != 1:
        raise RuntimeError('Actual proc stat fields are incomplete: ' + str(pid))
    flags, start = int(fields[6]), int(fields[19])
    if flags < 0 or start < 0:
        raise RuntimeError('Actual proc stat identity/flags are invalid: ' + str(pid))
    return {'PID': pid, 'state': fields[0], 'flags': flags, 'starttime_ticks': start}


def _process_RSS_sample(path):
    """Sample one PID coherently; never infer a live task's missing RSS as zero.

    A stat/status/stat sandwich detects PID reuse during the read.  A bounded
    resample observes the replacement rather than mixing its UID/RSS with the
    original identity. Missing VmRSS is released memory only with kernel exit
    evidence; proc status omits its memory section after the task loses its mm.
    """
    pid = int(path.name)
    for attempt in range(3):
        try:
            before = _proc_stat((path/'stat').read_text(), pid)
            fields = dict(line.split(':', 1) for line in (path/'status').read_text().splitlines() if ':' in line)
            after = _proc_stat((path/'stat').read_text(), pid)
        except (FileNotFoundError, ProcessLookupError):
            return {'PID': pid, 'disposition': 'reaped_during_sample'}
        if before['starttime_ticks'] != after['starttime_ticks']:
            continue
        if int(fields['Pid'].strip()) != pid:
            raise RuntimeError('Actual proc status PID differs: ' + str(pid))
        observed_uid = int(fields['Uid'].split()[0])
        if observed_uid < 0:
            raise RuntimeError('Actual process UID is invalid: ' + str(pid))
        result = dict(after, UID=observed_uid, name=fields.get('Name', '').strip())
        if 'VmRSS' not in fields:
            if after['state'] in ('Z', 'X', 'x') or after['flags'] & PF_EXITING:
                return dict(result, disposition='kernel_exit_without_mm',
                            exit_evidence='PF_EXITING' if after['flags'] & PF_EXITING else 'terminal_state')
            if attempt < 2:
                # exec can briefly detach the old mm without exiting. Read a
                # complete coherent observation again; no RSS value is invented.
                continue
            raise RuntimeError('Actual own process RSS is unavailable for live PID: ' + str(pid))
        values = fields['VmRSS'].split()
        if len(values) != 2 or values[1] != 'kB' or not values[0].isdigit():
            raise RuntimeError('Actual own process RSS units/value differ: ' + str(pid))
        return dict(result, RSS_bytes=int(values[0])*1024, disposition='RSS_observed')
    raise RuntimeError('Process identity changed throughout coherent RSS sampling: ' + str(pid))


def _own_process_RSS(uid, *, proc_root=None, lifecycle=None):
    rows = []
    released = []
    for path in (Path('/proc') if proc_root is None else Path(proc_root)).iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != uid:
                continue
            sample = _process_RSS_sample(path)
            if lifecycle is not None:
                lifecycle[int(path.name)] = sample
            if sample['disposition'] == 'RSS_observed' and sample['UID'] == uid:
                rows.append(sample)
            elif sample['disposition'] != 'RSS_observed':
                released.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            released.append(int(path.name))
            if lifecycle is not None:
                lifecycle[int(path.name)] = {'PID': int(path.name), 'disposition': 'reaped_during_sample'}
    return rows, released


def _identity_starttime(identity):
    if type(identity.get('starttime_ticks')) is int:
        return _positive(identity['starttime_ticks'], 'registered process starttime', zero=True)
    birth = identity.get('create_time')
    if type(birth) not in (int, float) or birth <= 0:
        raise ValueError('Registered process needs its actual starttime/create_time')
    import psutil
    return round((birth-psutil.boot_time())*os.sysconf('SC_CLK_TCK'))


def _cohort_RSS_members(rows, requested, lifecycle, uid, identities=None, *, proc_root=None):
    """Keep replacement RSS globally charged, without adopting a reused actor PID."""
    root = Path('/proc') if proc_root is None else Path(proc_root)
    requested = set(requested)
    if any(type(pid) is not int or pid <= 0 for pid in requested):
        raise ValueError('Exact positive cohort PIDs required')
    if identities is not None and not requested <= set(identities):
        raise ValueError('Every registered cohort PID needs an exact process identity')
    observed = {row['PID']: row for row in rows}
    active, released, reuse = set(), [], []
    for pid in sorted(requested):
        if identities is not None and identities[pid].get('pid', pid) != pid:
            raise ValueError('Registered process identity PID differs: ' + str(pid))
        sample = observed.get(pid) or lifecycle.get(pid)
        if sample is None or sample['disposition'] == 'reaped_during_sample':
            # A PID absent from an earlier enumeration can have been born or
            # reused since then. Resolve it; exists() alone is not identity proof.
            sample = _process_RSS_sample(root/str(pid))
            lifecycle[pid] = sample
            if sample['disposition'] == 'RSS_observed' and sample['UID'] == uid:
                observed[pid] = sample
        if sample['disposition'] == 'reaped_during_sample':
            released.append({'PID': pid, 'disposition': 'reaped_during_sample'})
            continue
        expected = None if identities is None else _identity_starttime(identities[pid])
        if expected is not None and sample['starttime_ticks'] != expected:
            reuse.append({'PID': pid, 'registered_starttime_ticks': expected,
                          'observed_starttime_ticks': sample['starttime_ticks'],
                          'observed_UID': sample['UID'], 'disposition': sample['disposition']})
            continue
        if sample['UID'] != uid:
            raise RuntimeError('Registered cohort process UID changed: ' + str(pid))
        if sample['disposition'] == 'kernel_exit_without_mm':
            released.append(sample)
        elif sample['disposition'] == 'RSS_observed':
            active.add(pid)
        else:
            raise RuntimeError('Unrecognized registered process lifecycle: ' + str(pid))
    return list(observed.values()), active, released, reuse


def _owned_tmpfs(uid, roots):
    seen, total, files, unavailable, released = set(), 0, 0, [], []
    pending = [Path(root) for root in roots]
    while pending:
        path = pending.pop()
        try:
            value = path.lstat()
            identity = (value.st_dev, value.st_ino)
            if identity in seen or stat.S_ISLNK(value.st_mode):
                continue
            seen.add(identity)
            if stat.S_ISDIR(value.st_mode):
                try:
                    with os.scandir(path) as entries:
                        pending.extend(Path(entry.path) for entry in entries)
                except PermissionError as error:
                    if value.st_uid == uid:
                        raise RuntimeError('Cannot measure own tmpfs directory: ' + str(path)) from error
                    unavailable.append(str(path))
            elif value.st_uid == uid and stat.S_ISREG(value.st_mode):
                total += value.st_blocks * 512
                files += 1
        except FileNotFoundError:
            released.append(str(path))
    return {'allocated_bytes': total, 'regular_files': files, 'inode_deduplication': 'st_dev/st_ino',
            'allocation_measurement': 'st_blocks * 512; symlinks not followed',
            'unreadable_foreign_directories': unavailable, 'naturally_released_paths': released}


def _cgroup_observation():
    text = Path('/proc/self/cgroup').read_text()
    limits = {}
    for line in text.splitlines():
        hierarchy, controllers, relative = line.split(':', 2)
        if hierarchy == '0' and not controllers:
            root, filename = Path('/sys/fs/cgroup'), 'memory.max'
        elif 'memory' in controllers.split(','):
            root, filename = Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes'
        else:
            continue
        if not relative.startswith('/') or '..' in Path(relative).parts:
            raise ValueError('Invalid actual cgroup path')
        folder = root / relative.lstrip('/')
        while folder == root or root in folder.parents:
            path = folder / filename
            if path.is_file():
                raw = path.read_text().strip()
                if raw != 'max' and not raw.isdigit():
                    raise RuntimeError('Actual cgroup memory limit is unavailable: ' + str(path))
                limits[str(path)] = None if raw == 'max' else int(raw)
            if folder == root:
                break
            folder = folder.parent
    if not limits:
        raise RuntimeError('No actual inherited memory limits could be read')
    return {'current_cgroup': text, 'memory_limits': [{'path': name, 'limit_bytes': value} for name, value in sorted(limits.items())]}


def _memory_accounting(rows, native_RSS_by_arm, cohort_PIDs, *, expected_arms, native_reserve_bytes,
                       auxiliary_reserve_bytes, max_user_RAM_bytes, own_tmpfs_bytes, host_available_bytes):
    if len(native_RSS_by_arm) != expected_arms:
        raise ValueError('Every unfinished experiment needs its own native RSS observation')
    values = [_positive(value, 'native RSS', zero=True) for value in native_RSS_by_arm.values()]
    process_map = {row['PID']: row['RSS_bytes'] for row in rows}
    cohort = set(cohort_PIDs)
    if any(type(pid) is not int or pid <= 0 for pid in cohort):
        raise ValueError('Exact positive cohort PIDs required')
    missing = sorted(cohort - set(process_map))
    if missing:
        raise RuntimeError('Exact cohort PIDs not present in current own UID RSS observations: ' + str(missing))
    cohort_RSS = sum(process_map[pid] for pid in cohort)
    auxiliary = cohort_RSS - sum(values)
    if auxiliary < 0:
        raise RuntimeError('Native tree RSS exceeds exact complete cohort RSS; observations inconsistent')
    if any(value > native_reserve_bytes for value in values) or auxiliary > auxiliary_reserve_bytes:
        raise RuntimeError('Actual native/auxiliary RSS exceeds the explicit per-experiment envelope')
    additional = sum(native_reserve_bytes-value for value in values) + auxiliary_reserve_bytes-auxiliary
    current_RSS = sum(process_map.values())
    prospective = current_RSS + own_tmpfs_bytes + additional
    if prospective > max_user_RAM_bytes or host_available_bytes < additional:
        raise RuntimeError('Direct user aggregate/host available RAM cannot admit the unchanged full experiments')
    return {'current_uid_RSS_bytes': current_RSS, 'own_tmpfs_allocated_bytes': own_tmpfs_bytes,
            'cohort_RSS_bytes': cohort_RSS, 'cohort_auxiliary_RSS_bytes': auxiliary,
            'native_RSS_by_arm': dict(native_RSS_by_arm), 'additional_reserved_bytes': additional,
            'prospective_user_RAM_bytes': prospective, 'host_available_bytes': host_available_bytes,
            'max_user_RAM_bytes': max_user_RAM_bytes, 'native_reserve_bytes_per_experiment': native_reserve_bytes,
            'auxiliary_reserve_bytes': auxiliary_reserve_bytes,
            'RSS_accounting': 'sum actual own UID process RSS; shared resident pages conservatively counted per process'}


def observe_direct_memory(native_RSS_by_arm, cohort_PIDs, *, uid=UID, expected_arms=6,
                          native_reserve_bytes=32*GIB, auxiliary_reserve_bytes=16*GIB,
                          max_user_RAM_bytes=512*GIB, tmpfs_roots=('/dev/shm',), native_PIDs_by_arm=None,
                          cohort_identities=None):
    if os.getuid() != uid or uid != UID:
        raise RuntimeError('Actual Linux UID41833 required')
    for name, value in [('expected_arms', expected_arms), ('native_reserve_bytes', native_reserve_bytes),
                        ('auxiliary_reserve_bytes', auxiliary_reserve_bytes), ('max_user_RAM_bytes', max_user_RAM_bytes)]:
        _positive(value, name)
    lifecycle = {}
    rows, released = _own_process_RSS(uid, lifecycle=lifecycle)
    requested = set(cohort_PIDs)
    rows, cohort_PIDs, registered_released, reused = _cohort_RSS_members(
        rows, requested, lifecycle, uid, cohort_identities)
    observed = {row['PID']: row['RSS_bytes'] for row in rows}
    if native_PIDs_by_arm is not None:
        if set(native_PIDs_by_arm) != set(native_RSS_by_arm):
            raise ValueError('Exact same native arm keys required for coherent RSS accounting')
        all_native = set()
        actual = {}
        for arm, pids in native_PIDs_by_arm.items():
            pids = set(pids)
            if all_native & pids or not pids <= requested:
                raise RuntimeError('Native process trees overlap or escape the exact cohort')
            all_native.update(pids)
            actual[arm] = sum(observed[pid] for pid in pids & cohort_PIDs)
        native_RSS_by_arm = actual
    tmpfs = _owned_tmpfs(uid, tmpfs_roots)
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    raw = meminfo['MemAvailable'].split()
    if len(raw) != 2 or raw[1] != 'kB':
        raise RuntimeError('Actual host MemAvailable is unavailable')
    result = _memory_accounting(rows, native_RSS_by_arm, cohort_PIDs, expected_arms=expected_arms,
        native_reserve_bytes=native_reserve_bytes, auxiliary_reserve_bytes=auxiliary_reserve_bytes,
        max_user_RAM_bytes=max_user_RAM_bytes, own_tmpfs_bytes=tmpfs['allocated_bytes'], host_available_bytes=int(raw[0])*1024)
    cgroup = _cgroup_observation()
    for row in cgroup['memory_limits']:
        if row['limit_bytes'] is not None and row['limit_bytes'] < result['prospective_user_RAM_bytes']:
            raise RuntimeError('Actual inherited cgroup limit cannot admit prospective direct user memory')
    result.update(cgroup, UID=uid, PID=os.getpid(), node=socket.gethostname(), observed_unix=time.time(),
                  own_process_RSS=rows, naturally_released_PIDs=released, tmpfs_observation=tmpfs,
                  registered_PIDs_proven_released=sorted(row['PID'] for row in registered_released),
                  registered_process_release_observations=registered_released,
                  registered_PID_reuse_observations=reused,
                  process_lifecycle_observations=list(lifecycle.values()),
                  native_RSS_from_same_process_snapshot=native_PIDs_by_arm is not None,
                  resource_mode='explicit_direct_access', PBS_CPU_GPU_ownership_claimed=False)
    return result


def capture_direct_resources(native_RSS_by_arm, cohort_PIDs, *, allowed_CPU_affinity=None,
                             required_slots=6, cpus_per_slot=8, **memory_options):
    required = _positive(required_slots, 'required_slots') * _positive(cpus_per_slot, 'cpus_per_slot')
    result = observe_direct_memory(native_RSS_by_arm, cohort_PIDs, **memory_options)
    actual = sorted(os.sched_getaffinity(0))
    allowed = set(actual if allowed_CPU_affinity is None else allowed_CPU_affinity)
    if not allowed <= set(actual):
        raise RuntimeError('Requested direct CPUs exceed actual SSH process affinity')
    PBS_assigned, observations, unreadable = set(), [], []
    root = Path('/sys/fs/cgroup/cpuset/pbs_jobs.service/jobid')
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            try:
                cpus = parse_CPU_list((path/'cpuset.cpus').read_text())
                PBS_assigned.update(cpus)
                observations.append({'path': str(path/'cpuset.cpus'), 'CPUs': cpus})
            except (FileNotFoundError, PermissionError) as error:
                unreadable.append({'path': str(path), 'error': type(error).__name__+': '+str(error)})
    else:
        unreadable.append({'path': str(root), 'error': 'PBS cpuset hierarchy not readable/present'})
    # This is explicitly shared direct execution, not a PBS CPU reservation.
    # The caller's actual inherited cpuset authorizes CPU access. Observed PBS
    # assignments describe sharing; they do not make those allowed CPUs absent.
    available = sorted(allowed)
    if len(available) < required:
        raise RuntimeError('Actual allowed SSH affinity lacks '+str(required)+' disjoint CPUs')
    import psutil
    utilization = psutil.cpu_percent(interval=.1, percpu=True)
    if len(utilization) <= max(available):
        raise RuntimeError('Actual CPU utilization observations do not cover available CPU IDs')
    ranked = sorted(available, key=lambda cpu: (utilization[cpu], cpu))
    selected = ranked[:required]
    result.update(CPU_affinity=actual, allowed_CPU_affinity=sorted(allowed), excluded_PBS_CPUs=[],
                  observed_PBS_assigned_CPUs=sorted(PBS_assigned),
                  selected_CPUs_shared_with_observed_PBS_jobs=sorted(set(selected)&PBS_assigned),
                  PBS_CPU_owned_reservation=False, CPU_exclusive_reservation_claimed=False,
                  PBS_cpuset_observations=observations, PBS_cpuset_unavailable=unreadable,
                  available_direct_CPUs=available, selected_CPU_groups=[selected[i:i+cpus_per_slot] for i in range(0, required, cpus_per_slot)],
                  observed_CPU_utilization_percent={str(cpu): utilization[cpu] for cpu in available},
                  CPU_selection='least observed utilization within actual allowed SSH affinity; disjoint fixed-size groups; observed PBS assignments are shared, not reserved')
    return result


def choose_empty_MIGs(observations, parent_UUID, *, required=6, min_memory_bytes=9*GIB,
                     exclude_UUIDs=(), max_idle_used_bytes=64*2**20):
    _positive(required,'required');_positive(min_memory_bytes,'min_memory_bytes')
    _positive(max_idle_used_bytes,'max_idle_used_bytes',zero=True)
    excluded = set(exclude_UUIDs)
    rows = [row for row in observations if row['parent_UUID'] == parent_UUID and row['MIG_UUID'] not in excluded
            and row['compute_process_count'] == 0 and row['graphics_process_count'] == 0
            and row['total_bytes'] >= min_memory_bytes and row['free_bytes'] >= min_memory_bytes
            and row['used_bytes'] <= max_idle_used_bytes]
    rows.sort(key=lambda row: (row['GPU_instance_id'], row['compute_instance_id'], row['MIG_UUID']))
    if len(rows) < required:
        raise RuntimeError('Fewer than '+str(required)+' empty sufficiently sized MIG devices on the exact same parent')
    selected = rows[:required]
    if len({row['MIG_UUID'] for row in selected}) != required or len({(row['GPU_instance_id'], row['compute_instance_id']) for row in selected}) != required:
        raise RuntimeError('Actual MIG identities overlap')
    return selected


def select_empty_MIGs(parent_UUID, *, required=6, min_memory_bytes=9*GIB,
                      exclude_UUIDs=(), max_idle_used_bytes=64*2**20):
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', parent_UUID):
        raise ValueError('Exact physical parent GPU UUID required')
    lib = ctypes.CDLL('libnvidia-ml.so.1')
    class Memory(ctypes.Structure):
        _fields_ = [('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
    def bind(name, args):
        method=getattr(lib,name);method.argtypes=args;method.restype=ctypes.c_int;return method
    def check(code,label):
        if code != 0:raise RuntimeError('NVML '+label+' failed with actual returncode '+str(code))
    ptr=ctypes.c_void_p;uint=ctypes.c_uint
    init=bind('nvmlInit_v2',[]);shutdown=bind('nvmlShutdown',[])
    by_uuid=bind('nvmlDeviceGetHandleByUUID',[ctypes.c_char_p,ctypes.POINTER(ptr)])
    get_uuid=bind('nvmlDeviceGetUUID',[ptr,ctypes.c_void_p,uint])
    maximum=bind('nvmlDeviceGetMaxMigDeviceCount',[ptr,ctypes.POINTER(uint)])
    mig_handle=bind('nvmlDeviceGetMigDeviceHandleByIndex',[ptr,uint,ctypes.POINTER(ptr)])
    parent_handle=bind('nvmlDeviceGetDeviceHandleFromMigDeviceHandle',[ptr,ctypes.POINTER(ptr)])
    memory=bind('nvmlDeviceGetMemoryInfo',[ptr,ctypes.POINTER(Memory)])
    gi_fn=bind('nvmlDeviceGetGpuInstanceId',[ptr,ctypes.POINTER(uint)])
    ci_fn=bind('nvmlDeviceGetComputeInstanceId',[ptr,ctypes.POINTER(uint)])
    def uuid(handle):
        value=ctypes.create_string_buffer(128);check(get_uuid(handle,value,len(value)),'UUID');return value.value.decode()
    def count(handle, kind):
        name=next((prefix for prefix in [kind+'_v3',kind+'_v2',kind]if hasattr(lib,prefix)),None)
        if name is None:raise RuntimeError('Actual NVML process observation symbol unavailable')
        method=bind(name,[ptr,ctypes.POINTER(uint),ctypes.c_void_p]);value=uint(0);code=method(handle,ctypes.byref(value),None)
        if code not in (0,7,3):check(code,name)
        return value.value,code,name
    check(init(),'init')
    try:
        parent=ptr();check(by_uuid(parent_UUID.encode(),ctypes.byref(parent)),'parent handle');assert uuid(parent)==parent_UUID
        parent_graphics,parent_graphics_code,parent_graphics_fn=count(parent,'nvmlDeviceGetGraphicsRunningProcesses')
        check(parent_graphics_code if parent_graphics_code!=7 else 0,'parent graphics process count')
        total=uint();check(maximum(parent,ctypes.byref(total)),'maximum MIG devices');rows=[]
        for index in range(total.value):
            handle=ptr();code=mig_handle(parent,index,ctypes.byref(handle))
            if code==6:continue
            check(code,'MIG handle')
            mapped=ptr();check(parent_handle(handle,ctypes.byref(mapped)),'MIG parent');actual_parent=uuid(mapped)
            if actual_parent!=parent_UUID:raise RuntimeError('NVML MIG parent mapping changed')
            mig_UUID=uuid(handle);gi=uint();ci=uint();mem=Memory()
            check(gi_fn(handle,ctypes.byref(gi)),'GPU instance');check(ci_fn(handle,ctypes.byref(ci)),'compute instance');check(memory(handle,ctypes.byref(mem)),'MIG memory')
            compute,compute_code,compute_fn=count(handle,'nvmlDeviceGetComputeRunningProcesses')
            if compute_code==3:raise RuntimeError('Cannot prove empty MIG compute processes: NVML not supported')
            graphics,graphics_code,graphics_fn=count(handle,'nvmlDeviceGetGraphicsRunningProcesses')
            if graphics_code==3:
                if parent_graphics:graphics=parent_graphics
                else:graphics=0
            rows.append({'MIG_UUID':mig_UUID,'parent_UUID':actual_parent,'GPU_instance_id':gi.value,'compute_instance_id':ci.value,
                         'total_bytes':mem.total,'free_bytes':mem.free,'used_bytes':mem.used,
                         'compute_process_count':compute,'graphics_process_count':graphics,
                         'NVML_compute_count_returncode':compute_code,'NVML_graphics_count_returncode':graphics_code,
                         'NVML_compute_count_function':compute_fn,'NVML_graphics_count_function':graphics_fn,
                         'graphics_not_supported_parent_count':parent_graphics if graphics_code==3 else None})
        selected=choose_empty_MIGs(rows,parent_UUID,required=required,min_memory_bytes=min_memory_bytes,
                                  exclude_UUIDs=exclude_UUIDs,max_idle_used_bytes=max_idle_used_bytes)
        return {'format':'explicit_direct_same_parent_empty_MIG_observation_v1','parent_UUID':parent_UUID,'observed_unix':time.time(),
                'selected':selected,'all_parent_MIGs':rows,'excluded_UUIDs':list(exclude_UUIDs),
                'CUDA_initialized':False,'NVML_only':True,'PBS_CPU_GPU_ownership_claimed':False,
                'parent_graphics_process_count':parent_graphics,'parent_graphics_count_function':parent_graphics_fn}
    finally:
        check(shutdown(),'shutdown own NVML reference')
