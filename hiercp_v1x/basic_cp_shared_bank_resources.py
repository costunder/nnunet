"""Explicit temporary CPU/RAM admission for the shared full642 bank producer.

The producer uses the three candidate arms' combined 288GiB budget.  The
other four native arms retain 96GiB each, plus 32GiB shared reserve: 704GiB.
Native requests and the live seven-GPU guard are never edited.  After this
CPU-only stage, its controller must restore GPU5's original eight CPU cores
and validate the original BasicLease before running any CUDA training.
"""
from __future__ import annotations

import copy as _copy
import types as _types

from . import basic_cp_ablation_resources as _original
from . import basic_cp_seven_resources as _seven


for _name, _value in vars(_original).items():
    if not _name.startswith('__') and not isinstance(_value, _types.FunctionType) and _name != 'BasicLease':
        globals()[_name] = _value


def _rebind_function(value):
    if value.__globals__ is not vars(_original):
        wrapped = getattr(value, '__wrapped__', None)
        if wrapped is not None and wrapped.__globals__ is vars(_original):
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

SHARED_CPU_AFFINITY = [*range(24, 48), *range(56, 96)]
CPUS = {5: SHARED_CPU_AFFINITY}
RSS_LIMIT = 288 * GIB
AGGREGATE_CAPACITY = 704 * GIB
SHARED_GUARD_FILE = 'hiercp_v1x/basic_cp_shared_bank_resources.py'
_runtime_validate_request = validate_request


def validate_shared_request(value):
    """Validate the unchanged native request plus its explicit CPU-only budget."""
    _seven.validate_request(value)
    if (value['gpu'] != 5 or value.get('shared_bank_CPU_affinity') != SHARED_CPU_AFFINITY
            or type(value.get('shared_bank_RSS_limit_GiB')) is not int
            or value['shared_bank_RSS_limit_GiB'] != 288
            or SHARED_GUARD_FILE not in value['code_sha256']):
        raise ValueError('Explicit shared-bank GPU5/64CPU/288GiB and exact guard SHA admission required')
    return value


def validate_request(value):
    """Internal execution view only; public native CPU/RSS values remain pinned."""
    result = _runtime_validate_request(value)
    if (value.get('native_CPU_affinity_preserved') != _seven.CPUS[5]
            or value.get('native_RSS_limit_bytes_preserved') != _seven.RSS_LIMIT
            or value.get('shared_bank_CPU_affinity') != SHARED_CPU_AFFINITY
            or value.get('shared_bank_RSS_limit_GiB') != 288
            or _seven.SEVEN_GUARD_FILE not in value['code_sha256']
            or SHARED_GUARD_FILE not in value['code_sha256']):
        raise ValueError('Shared CPU execution view lost the unchanged native eight-CPU/96GiB contract')
    return result


_class_body = dict(vars(_original.BasicLease))
_class_body.pop('__dict__', None)
_class_body.pop('__weakref__', None)
_class_body['__module__'] = __name__
for _name, _value in tuple(_class_body.items()):
    if isinstance(_value, _types.FunctionType):
        _class_body[_name] = _rebind_function(_value)
_InheritedLease = type('_InheritedSharedBankLease', _original.BasicLease.__bases__, _class_body)


class SharedBankLease(_InheritedLease):
    """Exact CPU-only process guard; quick checks between <=5s full checks."""

    def __init__(self, request):
        ordinary = _copy.deepcopy(validate_shared_request(request))
        self.native_request = ordinary
        self.native_request_bytes = json.dumps(ordinary, sort_keys=True, separators=(',', ':'))
        effective = dict(ordinary, cpu_affinity=list(SHARED_CPU_AFFINITY), RSS_limit_bytes=RSS_LIMIT,
                         native_CPU_affinity_preserved=list(ordinary['cpu_affinity']),
                         native_RSS_limit_bytes_preserved=ordinary['RSS_limit_bytes'])
        super().__init__(effective)
        self._request_path = _owned(Path(ordinary['root']) / 'request.json')
        self._admit(self._request_path, sha(self._request_path))
        if json.loads(self._request_path.read_bytes()) != ordinary:
            raise ValueError('Shared bank admission must match its exact original request.json bytes')
        self._request_snapshot = _snapshot(self._request_path)
        self._quick_admission = _copy.deepcopy(self._small_contract(self.request))
        self._quick_self_identity = _copy.deepcopy(self.self_identity)

    @staticmethod
    def _small_contract(value):
        keys = ('root', 'native_root', 'uid', 'gpu', 'gpu_uuid', 'cpu_affinity',
                'controller_identity', 'bank', 'bank_sha256', 'host_lifetime',
                'RSS_limit_bytes', 'runtime_reserve_bytes', 'GPU_memory_total_MiB',
                'shared_bank_CPU_affinity', 'shared_bank_RSS_limit_GiB',
                'native_CPU_affinity_preserved', 'native_RSS_limit_bytes_preserved')
        return {key: value[key] for key in keys}

    def _inputs(self):
        if json.dumps(self.native_request, sort_keys=True, separators=(',', ':')) != self.native_request_bytes:
            raise ValueError('Original native request changed during temporary shared bank admission')
        return super()._inputs()

    def _tree(self, controller, own):
        tree = {controller.pid: controller}
        for process in controller.children(recursive=True):
            try:
                process = self._descendant(process)
                if process is not None:
                    if process.cpu_affinity() != SHARED_CPU_AFFINITY:
                        raise RuntimeError('Shared bank descendant left the exact admitted64CPU affinity')
                    tree[process.pid] = process
            except psutil.NoSuchProcess:
                continue
        if own.pid not in tree:
            raise RuntimeError('Shared bank stage is outside its exact authorized controller tree')
        return tree

    def _rss(self, tree):
        total = 0
        for process in tree.values():
            try:
                if process.status() != psutil.STATUS_ZOMBIE:
                    if process.uids().real != UID:
                        raise RuntimeError('Foreign UID appeared in the exact shared bank process tree')
                    total += process.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        if total > RSS_LIMIT:
            raise MemoryError('Shared bank controller/producer/worker tree exceeds its explicit288GiB RSS budget')
        return total

    def _resources(self, controller, own, tree):
        observed = capture_cgroup_admission(controller.pid)
        if self._controller_cgroup is None:
            self._controller_cgroup = observed
        elif observed != self._controller_cgroup:
            raise RuntimeError('Exact shared bank controller cgroup/ancestor memory limits changed')
        declared = self.request.get('controller_cgroup_admission')
        if declared is not None and declared != observed:
            raise RuntimeError('Shared bank controller differs from its observed cgroup admission')
        current = capture_cgroup_admission(own.pid)
        for admission in (observed, current):
            for row in admission['memory_limits']:
                if row['limit_bytes'] is not None and row['limit_bytes'] < AGGREGATE_CAPACITY:
                    raise MemoryError('Actual cgroup cannot admit shared288GiB plus four96GiB and shared32GiB')
        rss = self._rss(tree)
        memory = psutil.virtual_memory()
        if memory.total < AGGREGATE_CAPACITY or memory.available + rss < AGGREGATE_CAPACITY:
            raise MemoryError('Actual host RAM cannot admit aggregate704GiB without changing experiment scale')
        free = shutil.disk_usage(self.request['native_root']).free
        if free < STORAGE_FLOOR:
            raise OSError('Full native tmpfs output has less than its admitted10GiB free floor')
        logical_cpus = psutil.cpu_count(logical=True)
        if logical_cpus is None or logical_cpus <= max(SHARED_CPU_AFFINITY):
            raise RuntimeError('Actual host CPU inventory cannot admit the measured64CPU shared affinity')
        return dict(own_process_tree_RSS_bytes=rss, own_process_tree_RSS_limit_bytes=RSS_LIMIT,
            aggregate_native_and_cache_capacity_bytes=AGGREGATE_CAPACITY,
            shared_bank_RAM_budget_bytes=RSS_LIMIT, other_four_native_RSS_budget_bytes=4 * _seven.RSS_LIMIT,
            canonical_cache_reserve_bytes=32 * GIB, host_RAM_bytes=memory.total,
            host_available_RAM_bytes=memory.available, native_free_bytes=free,
            native_free_floor_bytes=STORAGE_FLOOR, actual_controller_cgroup=observed,
            actual_stage_cgroup=current, actual_host_logical_CPU_count=logical_cpus,
            processes_signaled=False, original_native_request_modified=False,
            native_CPU_affinity_preserved=list(self.native_request['cpu_affinity']),
            native_RSS_limit_bytes_preserved=self.native_request['RSS_limit_bytes'],
            PBS_GPU_owned_reservation=False, PBS_CPU_owned_reservation=False,
            PBS_job_role='host_lifetime_only', own_GPU=self.request['gpu'],
            own_CPU_affinity=list(SHARED_CPU_AFFINITY), scientific_scale_preserved=True,
            execution_stage='shared_candidate_bank_CPU_only')

    def _gpu(self, tree, *, idle):
        result = super()._gpu(tree, idle=True)
        # CPU workers are forbidden from creating a CUDA context on ANY GPU,
        # including a GPU other than the producer controller's physicalGPU5.
        rows = _query(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                       '--format=csv,noheader,nounits'], timeout=5)
        for line in rows.splitlines():
            if not line.strip():
                continue
            cells = [value.strip() for value in line.split(',')]
            if len(cells) != 2 or not cells[0].isdigit():
                raise RuntimeError('Shared bank CUDA observation requires exact numeric PID evidence')
            if int(cells[0]) in tree:
                raise RuntimeError('CPU-only shared bank actor created a forbidden CUDA context: ' + line)
        result['shared_bank_CPU_only_CUDA_verified'] = True
        return result

    def quick_verify(self):
        """Constant-tree-size source/identity/affinity/expiry check per candidate."""
        # Hundreds of immutable source/code metadata entries are checked by
        # force/full verification at <=5s and before publishing any source.
        # Candidate-level checks read only the pinned request file and the
        # small execution contract, rather than repeating that whole scan.
        if (not stat.S_ISREG(self._request_path.lstat().st_mode)
                or _snapshot(self._request_path) != self._request_snapshot):
            raise ValueError('Exact shared bank original request.json metadata changed')
        if self._small_contract(self.request) != self._quick_admission or self.self_identity != self._quick_self_identity:
            raise ValueError('Shared bank execution identity/request contract changed during quick verification')
        if os.getuid() != UID or socket.gethostname().split('.')[0] != NODE:
            raise RuntimeError('Actual shared bank execution UID or node changed')
        controller = _exact(self.request['controller_identity'])
        own = _exact(self.self_identity)
        if controller.cpu_affinity() != SHARED_CPU_AFFINITY or own.cpu_affinity() != SHARED_CPU_AFFINITY:
            raise RuntimeError('Exact shared bank controller/stage64CPU affinity changed')
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            raise RuntimeError('Shared bank producer must remain CPU-only with uninitialized CUDA')
        remaining = min(self.lifetime['expires_at_unix'] - time.time(),
                        self.monotonic_expiry - time.monotonic())
        if remaining < EXPIRY_RESERVE_SECONDS:
            raise RuntimeError('Host lifetime has less than180seconds during shared CPU bank preparation')
        if self.last_verified is None or time.monotonic() - self.last_verified >= POLL_SECONDS:
            return self.verify(force=True)
        result = dict(self.actual_resources)
        result['host_lifetime_remaining_seconds'] = remaining
        result['last_full_resource_verification_age_seconds'] = time.monotonic() - self.last_verified
        return result

    def verify(self, *, force=False, idle=False):
        if not force and self.last_verified is not None and time.monotonic() - self.last_verified < POLL_SECONDS:
            return self.quick_verify()
        # The inherited verification keeps all exact PBS, code SHA, monotonic
        # expiry, controller/stage and GPU/cgroup checks in the independent view.
        result = super().verify(force=True, idle=True)
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            raise RuntimeError('Shared bank stage initialized CUDA during full CPU resource verification')
        result['last_full_resource_verification_age_seconds'] = 0.
        return result


BasicLease = SharedBankLease
__all__ = ['SharedBankLease', 'BasicLease', 'validate_shared_request', 'SHARED_CPU_AFFINITY',
           'RSS_LIMIT', 'AGGREGATE_CAPACITY', 'GIB', 'sha', 'capture_cgroup_admission']
