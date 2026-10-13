"""Seven independently authorized Basic CP arms on physical GPU0..6.

The proven original lease is copied into this module's independent globals.
Its PBS, exact process identity, frozen source and CUDA ownership checks remain
unchanged.  The old module and its running experiments are never patched.
Eight disjoint CPU cores and the original 96GiB process-tree limit apply to
each arm.  Seven full limits plus the shared 32GiB cache reserve require an
actual 704GiB cgroup and host admission; the science is never reduced to fit.
"""
from __future__ import annotations

import copy as _copy
import types as _types

from . import basic_cp_ablation_resources as _original


# Rebinding functions is deliberate: a normal subclass still resolves the
# original functions' CPUS/AGGREGATE_CAPACITY globals and would admit the wrong
# allocation.  Every inherited helper/method instead sees this module's own
# admitted constants, including when mocked in the read-only OS tests.
for _name, _value in vars(_original).items():
    if not _name.startswith('__') and not isinstance(_value, _types.FunctionType) and _name != 'BasicLease':
        globals()[_name] = _value
def _rebind_function(value):
    if value.__globals__ is not vars(_original):
        wrapped = getattr(value, '__wrapped__', None)
        if wrapped is not None and wrapped.__globals__ is vars(_original):
            # The contextmanager wrapper belongs to contextlib; its captured
            # generator must be rebound rather than its standard-library code.
            return _original.contextmanager(_rebind_function(wrapped))
        return value
    function = _types.FunctionType(value.__code__, globals(), value.__name__,
                                   value.__defaults__, value.__closure__)
    function.__kwdefaults__ = _copy.deepcopy(value.__kwdefaults__)
    function.__doc__ = value.__doc__
    function.__qualname__ = value.__qualname__
    function.__annotations__ = dict(value.__annotations__)
    return function


for _name, _value in vars(_original).items():
    if isinstance(_value, _types.FunctionType):
        globals()[_name] = _rebind_function(_value)

CPUS = {gpu: list(range(gpu * 8, (gpu + 1) * 8)) for gpu in range(7)}
AGGREGATE_CAPACITY = 704 * GIB
SHARED_CPU_AFFINITY = list(range(56, 64))
SEVEN_GUARD_FILE = 'hiercp_v1x/basic_cp_seven_resources.py'
ORIGINAL_GUARD_FILE = 'hiercp_v1x/basic_cp_ablation_resources.py'

_original_validate_request = validate_request


def validate_request(value):
    """Require both immutable guards and the exact seven-arm resource map."""
    result = _original_validate_request(value)
    if SEVEN_GUARD_FILE not in value['code_sha256']:
        raise ValueError('Seven-arm guard and original immutable guard require exact SHA admissions')
    return result


_class_body = dict(vars(_original.BasicLease))
_class_body.pop('__dict__', None)
_class_body.pop('__weakref__', None)
_class_body['__module__'] = __name__
for _name, _value in tuple(_class_body.items()):
    if isinstance(_value, _types.FunctionType):
        _class_body[_name] = _rebind_function(_value)

_InheritedLease = type('_InheritedLease', _original.BasicLease.__bases__, _class_body)


class BasicLease(_InheritedLease):
    """Full seven-arm admission plus a CPU-only wait for a free target GPU."""

    def _resources(self, controller, own, tree):
        observed = capture_cgroup_admission(controller.pid)
        if self._controller_cgroup is None:
            self._controller_cgroup = observed
        elif observed != self._controller_cgroup:
            raise RuntimeError('Exact seven-arm controller cgroup/ancestor memory limits changed')
        declared = self.request.get('controller_cgroup_admission')
        if declared is not None and declared != observed:
            raise RuntimeError('Fresh seven-arm controller differs from its observed cgroup admission')
        current = capture_cgroup_admission(own.pid)
        for admission in (observed, current):
            for row in admission['memory_limits']:
                if row['limit_bytes'] is not None and row['limit_bytes'] < AGGREGATE_CAPACITY:
                    raise MemoryError('Actual inherited cgroup cannot admit seven96GiB plus shared32GiB')
        rss = self._rss(tree)
        memory = psutil.virtual_memory()
        if memory.total < AGGREGATE_CAPACITY or memory.available + rss < AGGREGATE_CAPACITY:
            raise MemoryError('Actual host RAM cannot admit aggregate704GiB without changing experiment scale')
        free = shutil.disk_usage(self.request['native_root']).free
        if free < STORAGE_FLOOR:
            raise OSError('Full native tmpfs output has less than its admitted10GiB free floor')
        return dict(own_process_tree_RSS_bytes=rss, own_process_tree_RSS_limit_bytes=RSS_LIMIT,
            aggregate_native_and_cache_capacity_bytes=AGGREGATE_CAPACITY,
            canonical_cache_reserve_bytes=32 * GIB, host_RAM_bytes=memory.total,
            host_available_RAM_bytes=memory.available, native_free_bytes=free,
            native_free_floor_bytes=STORAGE_FLOOR, actual_controller_cgroup=observed,
            actual_stage_cgroup=current, original_experiment_guards_modified=False,
            processes_signaled=False, PBS_GPU_owned_reservation=False,
            PBS_job_role='host_lifetime_only', own_GPU=self.request['gpu'],
            own_CPU_affinity=self.request['cpu_affinity'],
            seven_independent_GPU_arms=True, scientific_scale_preserved=True)

    def observe_waiting(self):
        """Read GPU occupancy while validating this waiting CPU actor and PBS.

        No CUDA context is created, no foreign actor is admitted or signaled,
        and no existing driver row is ignored.  Full normal verify(idle=True)
        is still mandatory immediately before dispatching a training stage.
        """
        self._inputs()
        if os.getuid() != UID or socket.gethostname().split('.')[0] != NODE:
            raise RuntimeError('Actual seven-arm waiting UID or node changed')
        controller = _exact(self.request['controller_identity'])
        own = _exact(self.self_identity)
        expected = self.request['cpu_affinity']
        if controller.cpu_affinity() != expected or own.cpu_affinity() != expected:
            raise RuntimeError('Exact seven-arm waiting eight-CPU affinity changed')
        remaining = min(self.lifetime['expires_at_unix'] - time.time(),
                        self.monotonic_expiry - time.monotonic())
        if remaining < EXPIRY_RESERVE_SECONDS:
            raise RuntimeError('Host lifetime has less than180seconds while waiting for GPU')
        self._host_lifetime(force=True)
        remaining = min(self.lifetime['expires_at_unix'] - time.time(),
                        self.monotonic_expiry - time.monotonic())
        if remaining < EXPIRY_RESERVE_SECONDS:
            raise RuntimeError('Host lifetime crossed its180second boundary while waiting for GPU')
        tree = self._tree(controller, own)
        resources = self._resources(controller, own, tree)
        row = [cell.strip() for cell in _query(['nvidia-smi', '--id=' + str(self.request['gpu']),
            '--query-gpu=index,uuid,name,memory.total,memory.free,mig.mode.current',
            '--format=csv,noheader,nounits'], timeout=5).split(',')]
        if (len(row) != 6 or row[0] != str(self.request['gpu'])
                or row[1] != self.request['gpu_uuid'] or row[5] not in ('N/A', 'Disabled', '[N/A]')
                or not math.isfinite(float(row[3]))
                or float(row[3]) != self.request['GPU_memory_total_MiB']
                or not math.isfinite(float(row[4]))):
            raise RuntimeError('Exact waiting GPU UUID/MIG/full-memory identity changed')
        actors = []
        apps = _query(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                       '--format=csv,noheader,nounits'], timeout=5)
        for line in apps.splitlines():
            if not line.strip():
                continue
            cells = [cell.strip() for cell in line.split(',')]
            if len(cells) != 2 or not cells[0].isdigit():
                raise RuntimeError('Waiting CUDA actors require exact numeric PID evidence')
            if cells[1] == self.request['gpu_uuid']:
                actors.append(dict(pid=int(cells[0]), GPU_UUID=cells[1],
                                   admitted=False, processes_signaled=False))
        resources.update(GPU_name=row[2], GPU_UUID=row[1], GPU_total_MiB=float(row[3]),
            GPU_free_MiB=float(row[4]), MIG=row[5], occupied_GPU_compute_actors=actors,
            GPU_available=not actors and float(row[4]) >= 40 * 1024,
            host_lifetime_remaining_seconds=remaining,
            host_lifetime_expires_at_unix=self.lifetime['expires_at_unix'],
            host_lifetime_job_id=self.lifetime['job_id'],
            host_lifetime_actual_PBS_observation=copy.deepcopy(self._pbs_observation),
            host_lifetime_PBS_observation_age_seconds=time.monotonic() - self._pbs_verified_at,
            host_lifetime_PBS_warning=copy.deepcopy(self._pbs_timeout))
        return resources

    def wait_until_available(self, *, on_wait=None, poll_seconds=POLL_SECONDS):
        """Wait for actual release, then make the complete idle admission."""
        if type(poll_seconds) not in (int, float) or not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError('Finite positive waiting poll interval required')
        while True:
            observation = self.observe_waiting()
            if observation['GPU_available']:
                # A race is checked again by the original exact GPU guard;
                # another process appearing here is an explicit failure.
                return self.verify(force=True, idle=True)
            if on_wait is not None:
                on_wait(copy.deepcopy(observation))
            time.sleep(poll_seconds)


__all__ = ['BasicLease', 'validate_request', 'capture_cgroup_admission', 'sha',
           'FORMAT', 'UID', 'NODE', 'CPUS', 'RSS_LIMIT', 'AGGREGATE_CAPACITY',
           'STORAGE_FLOOR', 'SHARED_CPU_AFFINITY', 'GIB']
