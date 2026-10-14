"""Approved 512GiB operational accounting; complete Basic CP science is retained.

Legacy request declarations remain native96/shared384 for their immutable
validation schema. This new SHA-admitted execution guard enforces the actual
32GiB native / 320GiB source ceilings. Owned tmpfs, including deleted open
files, is counted by allocated inode blocks, and auxiliary processes are
measured rather than invented as zero.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

import psutil

from . import basic_cp_ablation_resources as common
from . import basic_cp_group_resources_v5 as group
from . import basic_cp_shared_bank_resources_v6 as shared
from . import basic_cp_RAM_admission_pause as pause
from .basic_cp_group_resources import RAMAdmissionUnavailable

GUARD_FILE = 'hiercp_v1x/basic_cp_512_resources.py'
CLAIMS_FILE = 'hiercp_v1x/basic_cp_512_claims.py'
FORMAT = 'basic_cp_measured_RAM512_v1'
GIB = 2 ** 30
AGGREGATE_CAPACITY = 512 * GIB
NATIVE_RSS_LIMIT = 32 * GIB
SOURCE_RSS_LIMIT = 320 * GIB
SOURCE_STORE_LIMIT = 144 * GIB
POST_SOURCE_STORE_LIMIT = 272 * GIB
AUX_RSS_LIMIT = 16 * GIB
GROUP_RSS_LIMIT = 480 * GIB
SHARED_CPU_AFFINITY = list(shared.SHARED_CPU_AFFINITY)
FIELDS = {
    'memory_budget_revision': 512,
    'aggregate_memory_cap_GiB': 512,
    'effective_native_RSS_limit_GiB': 32,
    'effective_shared_bank_RSS_limit_GiB': 320,
    'owned_tmpfs_envelope_GiB': 144,
    'post_source_owned_tmpfs_envelope_GiB': 272,
    'auxiliary_RAM_envelope_GiB': 16,
}


def validate_request(request):
    for key, expected in FIELDS.items():
        if type(request.get(key)) is not int or request[key] != expected:
            raise ValueError('Explicit approved RAM512 integer field required: ' + key)
    pins = request.get('code_sha256', {})
    if not {GUARD_FILE, CLAIMS_FILE, group.GUARD_FILE, shared.SHARED_GUARD_FILE,
            pause.GUARD_FILE} <= set(pins):
        raise ValueError('Exact RAM512/claims/Group5/Shared6/pause code SHA pins required')
    for key in ('owned_tmpfs_root', 'RAM512_claim_ledger'):
        value = request.get(key)
        if (not isinstance(value, str) or not Path(value).is_absolute()
                or '..' in Path(value).parts):
            raise ValueError('Absolute exact RAM512 path required: ' + key)
    root = Path(request['owned_tmpfs_root'])
    if not Path(request['root']).is_relative_to(root):
        raise ValueError('Registered experiment must belong to its owned tmpfs accounting root')
    if (not isinstance(request.get('native_root'), str)
            or not Path(request['native_root']).is_absolute()
            or not Path(request['native_root']).is_relative_to(root)):
        raise ValueError('Native files must belong to the same measured owned tmpfs accounting root')
    if 'shared_banks' in request and not Path(request['shared_banks']).is_relative_to(root):
        raise ValueError('Generated candidate files must belong to the owned tmpfs accounting root')
    if not Path(request['RAM512_claim_ledger']).is_relative_to(root):
        raise ValueError('Claim ledger must belong to the same owned tmpfs accounting root')
    reference = request.get('RAM512_source_pending_receipt')
    if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}
            or not isinstance(reference['path'], str) or not Path(reference['path']).is_absolute()
            or not Path(reference['path']).is_relative_to(root)
            or not common._checksum(reference['sha256'])):
        raise ValueError('Exact owned RAM512 source-bootstrap receipt path and SHA required')
    return request


def _identity(process):
    metadata = common._metadata(process)
    command = process.cmdline()
    if common._metadata(process) != metadata:
        raise RuntimeError('RAM512 task changed PID/birth/UID during exact observation')
    return dict(metadata, command=command)


def _same_metadata(process, before):
    after = common._metadata(process)
    if any(after[key] != before[key] for key in ('pid', 'create_time', 'UID')):
        raise RuntimeError('RAM512 PID/birth/UID changed during memory observation')


def outside_inode_store_bound(root, *, excluded_inode_keys=frozenset()):
    """Coherent conservative charge: device allocation minus held outside files.

    O_PATH references prevent an outside inode's unlink from invalidating the
    subtraction. Inaccessible directories stay in the charge, which can only
    overstate owned memory. Every allocated outside regular inode is considered;
    no file contents are read and no model/data/sample limit is introduced.
    """
    root = Path(root)
    device_root = root
    device = root.stat().st_dev
    while device_root.parent != device_root and device_root.parent.stat().st_dev == device:
        device_root = device_root.parent
    if device_root == root or not hasattr(os, 'O_PATH'):
        raise RuntimeError('Linux tmpfs device-root/O_PATH accounting is required')
    held = {}; denied = []; stack = [device_root]
    try:
        while stack:
            folder = stack.pop()
            try:
                with os.scandir(folder) as entries:
                    for entry in entries:
                        path = Path(entry.path)
                        if path == root:
                            continue
                        try:
                            metadata = entry.stat(follow_symlinks=False)
                        except FileNotFoundError:
                            continue
                        if metadata.st_dev != device:
                            raise RuntimeError('Named tmpfs observation crossed another device')
                        if stat.S_ISDIR(metadata.st_mode):
                            stack.append(path)
                        elif stat.S_ISREG(metadata.st_mode) and metadata.st_blocks > 0:
                            try:
                                fd = os.open(path, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
                            except FileNotFoundError:
                                continue
                            value = os.fstat(fd)
                            key = (value.st_dev, value.st_ino)
                            if value.st_dev != device or not stat.S_ISREG(value.st_mode):
                                os.close(fd)
                                raise RuntimeError('O_PATH outside observation changed device/file type')
                            if key in held or key in excluded_inode_keys:
                                os.close(fd)
                            else:
                                held[key] = (fd, value)
            except PermissionError:
                info = folder.stat()
                denied.append(dict(UID=info.st_uid, mode=stat.S_IMODE(info.st_mode)))
        before = {key: os.fstat(fd) for key, (fd, _) in held.items()}
        filesystem = os.statvfs(device_root)
        device_used = (filesystem.f_blocks - filesystem.f_bfree) * filesystem.f_frsize
        stable = {}
        for key, (fd, first) in held.items():
            middle = before[key]; final = os.fstat(fd)
            # In-place mutation is never subtracted from device allocation.
            if (first.st_size, first.st_mtime_ns, first.st_ctime_ns) != (
                    final.st_size, final.st_mtime_ns, final.st_ctime_ns):
                continue
            stable[key] = min(first.st_blocks, middle.st_blocks, final.st_blocks) * 512
        outside = sum(stable.values())
        if device_used < outside:
            raise RuntimeError('Coherent outside inode charge exceeded actual device allocation')
        return dict(store_upper_bound_bytes=device_used - outside,
            whole_tmpfs_device_allocated_bytes=device_used,
            held_stable_outside_allocated_bytes=outside,
            held_outside_inode_count=len(held), stable_outside_inode_count=len(stable),
            unreadable_outside_directory_metadata=denied,
            accounting='actual_device_used_minus_O_PATH_held_stable_allocated_outside_inodes',
            foreign_files_read_for_content=False, observer_environment_modified=False)
    finally:
        for fd, _ in held.values():
            os.close(fd)


def allocated_tmpfs(root, processes, *, uid=common.UID, proc_root=Path('/proc')):
    """Count owned filesystem blocks once, including unlinked open tmpfs files.

    Permission/I/O errors are fatal. Only an actually vanished inode or task is
    re-observed/skipped; a failed inventory never supplies fabricated capacity.
    ``proc_root`` is injectable only for CPU filesystem regression tests.
    """
    root = Path(root)
    metadata = root.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != uid:
        raise RuntimeError('RAM512 accounting root is not the exact owned directory')
    device = metadata.st_dev
    inodes = {}
    deleted = []
    denied_FDs = []

    def admit(value, path):
        if value.st_uid != uid:
            raise RuntimeError('Foreign-owned inode in RAM512 accounting root: ' + str(path))
        key = (value.st_dev, value.st_ino)
        blocks = value.st_blocks * 512
        if blocks < 0:
            raise RuntimeError('Allocated inode blocks cannot be negative')
        # A coherent scan can observe an atomically replaced inode twice; the
        # larger allocation is conservative and neither creates free capacity.
        inodes[key] = max(blocks, inodes.get(key, 0))

    admit(metadata, root)
    stack = [root]
    while stack:
        folder = stack.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    value = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    # A named file can disappear after scandir during an
                    # atomic checkpoint replacement; held old FDs follow below.
                    continue
                admit(value, entry.path)
                if stat.S_ISDIR(value.st_mode):
                    stack.append(Path(entry.path))
                elif not (stat.S_ISREG(value.st_mode) or stat.S_ISLNK(value.st_mode)):
                    raise RuntimeError('Unexpected special file in RAM512 accounting root')
    root_text = str(root).rstrip('/') + '/'
    for process in processes.values():
        try:
            before = common._metadata(process)
            if before['UID'] != uid:
                raise RuntimeError('Foreign process supplied a RAM512 own open-file observation')
            directory = Path(proc_root) / str(process.pid) / 'fd'
            try:
                names = list(directory.iterdir())
            except PermissionError:
                _same_metadata(process, before)
                denied_FDs.append(dict(before, command=process.cmdline(), observation='PermissionDenied'))
                continue
            except FileNotFoundError:
                # /proc disappearance is accepted only with real task release.
                process.status()
                raise RuntimeError('Live own process has no observable fd inventory')
            for descriptor in names:
                try:
                    target = os.readlink(descriptor)
                    if not (target.startswith(root_text) or target == str(root)):
                        continue
                    value = descriptor.stat()
                except PermissionError:
                    denied_FDs.append(dict(before, command=process.cmdline(), fd=descriptor.name,
                        observation='PermissionDenied'))
                    continue
                except FileNotFoundError:
                    # Individual FD close is an ordinary producer/helper IO
                    # transition. Its inode is no longer held by this FD.
                    continue
                if value.st_dev != device:
                    raise RuntimeError('Owned tmpfs FD resolved onto a different device')
                admit(value, descriptor)
                if target.endswith(' (deleted)'):
                    deleted.append(dict(pid=process.pid, fd=descriptor.name,
                        target=target, inode=value.st_ino, allocated_bytes=value.st_blocks * 512))
            _same_metadata(process, before)
        except psutil.NoSuchProcess:
            continue
    measured = sum(inodes.values())
    result = dict(owned_tmpfs_allocated_bytes=measured,
                inode_count=len(inodes), deleted_open_inode_observations=deleted,
                accounting='unique_owned_allocated_inode_blocks_including_open_deleted_FDs')
    if denied_FDs:
        bound = outside_inode_store_bound(root, excluded_inode_keys=set(inodes))
        upper = max(measured, bound['store_upper_bound_bytes'])
        result.update(bound, owned_tmpfs_visible_and_readable_allocated_bytes=measured,
            owned_tmpfs_unknown_file_upper_bound_bytes=upper-measured,
            owned_tmpfs_allocated_bytes=upper, unreadable_own_FD_actors=denied_FDs,
            unknown_file_charge_is_actual_owned_attribution=False,
            denied_FD_zero_fallback_used=False)
    return result


def own_process_inventory(*, uid=common.UID):
    """Actual same-UID RSS inventory; AccessDenied is never treated as zero."""
    processes = {}
    rss = {}
    identities = {}
    for process in psutil.process_iter():
        try:
            if process.uids().real != uid:
                continue
            before = common._metadata(process)
            if process.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                continue
            value = process.memory_info().rss
            _same_metadata(process, before)
            processes[process.pid] = process
            rss[process.pid] = value
            identities[process.pid] = before
        except psutil.NoSuchProcess:
            continue
    return processes, rss, identities


def archive_allocated_bytes(row, accounting_root):
    """Only actual own writes under the exact reserved fresh destination count."""
    if row.get('destination') is None:
        return 0
    destination = Path(row['destination'])
    if (not destination.is_absolute() or not destination.is_relative_to(Path(accounting_root))
            or any(parent.is_symlink() for parent in destination.parents)):
        raise RuntimeError('Reserved independent archive destination left its exact namespace')
    parent = destination.parent.stat()
    identity = [parent.st_dev, parent.st_ino, parent.st_uid]
    if identity != row.get('destination_parent_identity') or parent.st_uid != common.UID:
        raise RuntimeError('Reserved archive destination parent inode/UID changed')
    if not destination.exists():
        return 0
    info = destination.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != common.UID:
        raise RuntimeError('Reserved archive destination is not its own nonsymlink directory')
    stack = [destination]; inodes = {}
    while stack:
        folder = stack.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                value = entry.stat(follow_symlinks=False)
                if value.st_uid != common.UID or stat.S_ISLNK(value.st_mode):
                    raise RuntimeError('Independent archive acquired foreign/symlink content')
                if stat.S_ISDIR(value.st_mode):
                    stack.append(Path(entry.path))
                elif not stat.S_ISREG(value.st_mode):
                    raise RuntimeError('Independent archive acquired special file content')
                inodes[(value.st_dev, value.st_ino)] = value.st_blocks * 512
    return sum(inodes.values())


def enforce_accounting(*, store_bytes, native_rss, source_rss, auxiliary_rss,
                       source_pending, host_total, host_available, registered_rss,
                       projected_storage_bytes=0):
    """Pure approved ceilings, used by the real observation and CPU tests."""
    quantities = [store_bytes, source_rss, auxiliary_rss, host_total,
                  host_available, registered_rss, projected_storage_bytes, *native_rss.values()]
    if (type(source_pending) is not bool or any(type(value) is not int or value < 0
                                              for value in quantities)):
        raise ValueError('Actual nonnegative integer bytes and exact source phase required')
    store_limit = SOURCE_STORE_LIMIT if source_pending else POST_SOURCE_STORE_LIMIT
    projected_store = store_bytes + projected_storage_bytes
    if store_bytes > store_limit:
        raise MemoryError('Owned tmpfs exceeds its approved ' + str(store_limit // GIB) + 'GiB phase ceiling')
    if projected_store > store_limit:
        from .basic_cp_512_claims import ResourceClaimUnavailable
        raise ResourceClaimUnavailable('Independent archive awaits approved ' + str(store_limit // GIB) + 'GiB file envelope',
            dict(owned_tmpfs_allocated_bytes=store_bytes, projected_store_bytes=projected_store,
                 owned_tmpfs_limit_bytes=store_limit, source_pending=source_pending))
    if source_rss > SOURCE_RSS_LIMIT:
        raise MemoryError('Exact CPU source tree exceeds its approved320GiB ceiling')
    for gpu, value in native_rss.items():
        if value > NATIVE_RSS_LIMIT:
            raise MemoryError('Exact native GPU' + str(gpu) + ' tree exceeds its approved32GiB ceiling')
    if auxiliary_rss > AUX_RSS_LIMIT:
        raise MemoryError('Actual own auxiliary RSS exceeds its approved16GiB ceiling')
    if source_pending and len([value for value in native_rss.values() if value > 0]) > 1:
        raise MemoryError('CPU source phase admits only one complete native tree')
    if source_pending and any(native_rss.get(gpu, 0) for gpu in (3, 4, 5)):
        raise MemoryError('Candidate native actors must await complete CPU source release')
    if not source_pending and source_rss:
        raise RuntimeError('Released source phase still contains a live source actor')
    group_rss = source_rss + sum(native_rss.values())
    if group_rss > GROUP_RSS_LIMIT:
        raise MemoryError('Registered compute RSS exceeds approved480GiB ceiling')
    actual_total = store_bytes + group_rss + auxiliary_rss
    total = projected_store + group_rss + auxiliary_rss
    if actual_total > AGGREGATE_CAPACITY:
        raise MemoryError('Actual owned tmpfs plus compute and auxiliary RSS exceeds approved512GiB total')
    if total > AGGREGATE_CAPACITY:
        from .basic_cp_512_claims import ResourceClaimUnavailable
        raise ResourceClaimUnavailable('Independent archive awaits approved512GiB total',
            dict(actual_total_RAM_bytes=actual_total, projected_total_RAM_bytes=total,
                 effective_RAM_cap_bytes=AGGREGATE_CAPACITY))
    if host_total < AGGREGATE_CAPACITY:
        raise MemoryError('Actual total host RAM cannot admit approved512GiB')
    if host_available + store_bytes + registered_rss + auxiliary_rss < AGGREGATE_CAPACITY:
        raise RAMAdmissionUnavailable(host_total_bytes=host_total,
            available_bytes=host_available,
            group_resident_bytes=store_bytes + registered_rss + auxiliary_rss,
            required_bytes=AGGREGATE_CAPACITY)
    return dict(format=FORMAT, effective_RAM_cap_bytes=AGGREGATE_CAPACITY,
        owned_tmpfs_allocated_bytes=store_bytes, owned_tmpfs_limit_bytes=store_limit,
        projected_owned_tmpfs_bytes=projected_store,
        projected_storage_reservation_bytes=projected_storage_bytes,
        source_phase_pending=source_pending, source_process_tree_RSS_bytes=source_rss,
        effective_shared_bank_RSS_limit_bytes=SOURCE_RSS_LIMIT,
        effective_native_RSS_limit_bytes=NATIVE_RSS_LIMIT,
        native_RSS_by_GPU_bytes={str(key): value for key, value in native_rss.items()},
        auxiliary_RSS_bytes=auxiliary_rss, auxiliary_RSS_limit_bytes=AUX_RSS_LIMIT,
        accounted_total_RAM_bytes=actual_total, projected_total_RAM_bytes=total,
        registered_compute_RSS_bytes=group_rss,
        group_process_tree_RSS_limit_bytes=GROUP_RSS_LIMIT,
        host_RAM_bytes=host_total, host_available_RAM_bytes=host_available,
        measured_available_plus_owned_charge_bytes=host_available + store_bytes + registered_rss + auxiliary_rss,
        allocation_bytes_are_conservative=True, legacy_native_RSS_declaration_bytes=96 * GIB,
        legacy_shared_RSS_declaration_bytes=384 * GIB, scientific_scale_preserved=True,
        processes_signaled=False)


def _normalize_identity(value):
    if not isinstance(value, dict):
        raise ValueError('Exact actor identity required')
    result = dict(value)
    if 'create_time' not in result and 'birth' in result:
        result['create_time'] = result.pop('birth')
    result = {key: result[key] for key in ('pid', 'create_time', 'UID', 'command') if key in result}
    if not common._identity(result):
        raise ValueError('Exact PID/birth/UID/argv claim identity required')
    return result


class ProcessResolver:
    """Only exact own actors and their actual descendants can release claims."""
    def __init__(self, lease):
        self.lease = lease

    def inspect(self, identity):
        identity = _normalize_identity(identity)
        try:
            process = psutil.Process(identity['pid'])
            metadata = common._metadata(process)
            if metadata['create_time'] != identity['create_time']:
                return dict(state='gone', identity=identity, actual_identity=dict(metadata,
                    command=process.cmdline()), terminal=True, gone=True,
                    original_actor_released_and_numeric_PID_reused=True,
                    replacement_actor_admitted=False, processes_signaled=False)
            if any(metadata[key] != identity[key] for key in ('pid', 'UID')):
                raise RuntimeError('Claim actor PID/birth/UID changed')
            state = process.status()
            terminal = state in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
            actual = dict(metadata, command=process.cmdline())
            _same_metadata(process, metadata)
            if not terminal and actual != identity and not (
                    not actual['command'] and common._kernel_exiting(process)):
                raise RuntimeError('Live claim actor argv changed')
            return dict(state='terminal' if terminal else 'live', identity=identity,
                        actual_identity=actual, terminal=terminal, gone=False,
                        identity_is_original_bound_capture=True)
        except psutil.NoSuchProcess:
            return dict(state='gone', identity=identity, actual_identity=None,
                        terminal=True, gone=True)

    def descendants(self, identity):
        identity = _normalize_identity(identity)
        observation = self.inspect(identity)
        if observation['state'] != 'live':
            return []
        result = []
        try:
            parent = common._exact(identity)
            for process in parent.children(recursive=True):
                try:
                    admitted = self.lease._descendant(process)
                    if admitted is not None:
                        current = _identity(admitted)
                        if not current['command']:
                            # The original guard keeps a PF_EXITING task in
                            # its RSS/CPU tree. Never turn that observation
                            # into an invented nonempty allocation identity.
                            metadata = {key: current[key] for key in ('pid', 'create_time', 'UID')}
                            previous = self.lease._managed.get(current['pid'])
                            if previous is not None:
                                if any(previous[key] != metadata[key] for key in metadata):
                                    raise RuntimeError('Exiting claim descendant PID/birth/UID changed')
                                current = _normalize_identity(previous)
                                self.inspect(current)  # Original exact argv remains bound until release.
                            else:
                                terminal = self.lease._terminal_observations.get(current['pid'])
                                if (terminal is None or terminal.get('status') != 'PF_EXITING'
                                        or any(terminal.get(key) != metadata[key] for key in metadata)
                                        or common._metadata(admitted) != metadata
                                        or not common._kernel_exiting(admitted)):
                                    raise RuntimeError('Uncaptured empty-argv descendant lacks exact kernel-exit evidence')
                                if not self.lease._terminal_after_change(admitted, metadata,
                                        reason='uncaptured allocation descendant completed kernel exit'):
                                    raise RuntimeError('Uncaptured empty-argv descendant is still live after bounded exit observation')
                                continue  # Actual terminal/NoSuch proof is retained by the original guard.
                        result.append(current)
                except psutil.NoSuchProcess:
                    continue
            common._exact(identity)
        except psutil.NoSuchProcess:
            # A parent may naturally disappear while its already captured
            # workers still run. The ledger retains earlier captures.
            return result
        return result

    def scan(self, command):
        found = []
        for process in psutil.process_iter():
            try:
                if process.uids().real != common.UID or process.cmdline() != command:
                    continue
                identity = _identity(process)
                if identity['command'] != command:
                    raise RuntimeError('Claim command changed during exact fast-start observation')
                if process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    found.append(identity)
            except psutil.NoSuchProcess:
                continue
        return found

    def has_cuda(self, identities):
        # Numeric PID reuse cannot keep a released original source allocation
        # alive or grant its replacement any execution permission.
        target = {_normalize_identity(value)['pid'] for value in identities
                  if self.inspect(value)['state'] != 'gone'}
        return _pids_have_cuda(target)


def _pids_have_cuda(target):
        query = common._query(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                               '--format=csv,noheader,nounits'], timeout=5)
        for line in query.splitlines():
            if not line.strip():
                continue
            cells = [part.strip() for part in line.split(',')]
            if len(cells) != 2 or not cells[0].isdigit():
                raise RuntimeError('Claim release requires exact numeric actual CUDA rows')
            if int(cells[0]) in target:
                return True
        return False


class _RAM512:
    def __init__(self, request):
        validate_request(request)
        super().__init__(request)
        view = getattr(self, '_group_view', self)
        self._validate_common(view)
        reference = self.request['RAM512_source_pending_receipt']
        self._admit(reference['path'], reference['sha256'])
        self._bootstrap_reference = copy.deepcopy(reference)
        self._bootstrap_receipt = json.loads(Path(reference['path']).read_bytes())
        if type(self._bootstrap_receipt.get('source_initially_pending')) is not bool:
            raise ValueError('Actual source-bootstrap pending phase must be explicit')
        self._claim_override = None
        self._completion_cache = {}
        self._ledger = None
        self._shared_observer_lease = None
        self._RAM512_contract_bytes = json.dumps({key: self.request[key]
            for key in (*FIELDS, 'owned_tmpfs_root', 'RAM512_claim_ledger', 'RAM512_source_pending_receipt')}, sort_keys=True)

    def _inputs(self):
        super()._inputs()
        if hasattr(self, '_RAM512_contract_bytes'):
            validate_request(self.request)
            if json.dumps({key: self.request[key] for key in
                    (*FIELDS, 'owned_tmpfs_root', 'RAM512_claim_ledger', 'RAM512_source_pending_receipt')}, sort_keys=True) != self._RAM512_contract_bytes:
                raise ValueError('Approved RAM512 execution contract changed in memory')

    def _validate_common(self, view):
        if getattr(view, '_previous_registry_manifest', {}):
            raise ValueError('RAM512 requires one fresh all-seven registry without previous-ledger permissions')
        for peer in view.registered_requests.values():
            validate_request(peer)
            if (peer.get('shared_resource_registry') != str(view.registry_path)
                    or peer['owned_tmpfs_root'] != self.request['owned_tmpfs_root']
                    or peer['RAM512_claim_ledger'] != self.request['RAM512_claim_ledger']
                    or peer['RAM512_source_pending_receipt'] != self.request['RAM512_source_pending_receipt']):
                raise ValueError('All seven RAM512 peers require the same current ledger and accounting root')
            for filename in (GUARD_FILE, CLAIMS_FILE, group.GUARD_FILE,
                             shared.SHARED_GUARD_FILE, pause.GUARD_FILE):
                if peer['code_sha256'][filename] != self.request['code_sha256'][filename]:
                    raise ValueError('All seven RAM512 peers require the same exact operational guard code')
        producer = view.registered_requests[5]
        if (type(producer.get('shared_bank_workers')) is not int or producer['shared_bank_workers'] != 24
                or producer.get('shared_bank_CPU_affinity') != SHARED_CPU_AFFINITY
                or producer.get('shared_bank_RSS_limit_GiB') != 384):
            raise ValueError('Exact full24-worker/64CPU source keeps its original384 declaration')

    def claim_binding(self):
        view = getattr(self, '_group_view', self)
        return dict(registry_path=str(view.registry_path), registry_sha256=view.registry_sha256,
            request_sha256={str(gpu): row['request_sha256'] for gpu, row in view.registration.items()},
            host_lifetime_sha256=self.native_request['host_lifetime']['sha256'] if hasattr(self, 'native_request')
                else self.request['host_lifetime']['sha256'],
            expires_at_unix=self.lifetime['expires_at_unix'],
            owned_tmpfs_root=self.request['owned_tmpfs_root'],
            source_initially_pending=self._bootstrap_receipt['source_initially_pending'])

    def claim_ledger(self):
        if self._ledger is None:
            from .basic_cp_512_claims import Ledger
            native = self.native_request if hasattr(self, 'native_request') else self.request
            self._ledger = Ledger(native, self.claim_binding(), ProcessResolver(self))
            self._ledger.bootstrap(source_pending_receipt=self._bootstrap_reference)
        return self._ledger

    def _claim_state(self):
        return copy.deepcopy(self._claim_override) if self._claim_override is not None else self.claim_ledger().snapshot()

    def _bind_self(self):
        if self.self_identity == self.request['controller_identity']:
            return
        state = self.claim_ledger().snapshot()
        for claim in state['claims'].values():
            if (claim['gpu'] == self.request['gpu'] and claim['command'] == self.self_identity['command']
                    and claim['child_identity'] is None):
                own = common._exact(self.self_identity)
                parent = own.parent()
                if parent is not None:
                    actual_parent = _identity(parent)
                    if actual_parent == self.request['controller_identity']:
                        self.claim_ledger().assert_current_child(claim['stage'], claim['command'],
                            self.self_identity, actual_parent)
                return

    def verify(self, *, force=False, idle=False):
        self._bind_self()
        return super().verify(force=force, idle=idle)

    def _rss(self, tree):
        rss = 0
        for process in tree.values():
            try:
                if process.uids().real != common.UID:
                    raise RuntimeError('Foreign UID appeared in RAM512 exact own process tree')
                if process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    rss += process.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        limit = SOURCE_RSS_LIMIT if isinstance(self, (SharedBankLease, _CPUStageGroupLease)) else NATIVE_RSS_LIMIT
        if rss > limit:
            raise MemoryError('Exact own tree exceeds approved' + str(limit // GIB) + 'GiB execution ceiling')
        return rss

    def _completion(self, path, validator):
        path = Path(path)
        if not path.exists():
            return False
        metadata = common._snapshot(path)
        cached = self._completion_cache.get(str(path))
        if cached is not None:
            if metadata != cached:
                raise ValueError('Sealed RAM512 completion proof changed')
            return True
        value = json.loads(path.read_bytes())
        validator(value)
        self._admit(path, common.sha(path))
        self._completion_cache[str(path)] = metadata
        return True

    def source_complete(self):
        view = getattr(self, '_group_view', self)
        request = view.registered_requests[5]
        banks = Path(request['shared_banks'])
        def validate(value):
            if (value.get('complete') is not True or value.get('full_sources') != 642
                    or set(value.get('prefixes', {})) != {'512', '1024', '2048'}
                    or value.get('CUDA_initialized') is not False):
                raise ValueError('Actual complete full642 CPU-only staged bank proof required')
            for count in (512, 1024, 2048):
                prefix = banks / str(count)
                index = prefix / 'index.json'
                done = prefix / 'complete.json'
                metadata = json.loads(index.read_bytes())
                receipt = json.loads(done.read_bytes())
                if (metadata.get('source_entries') != 642 or metadata.get('candidate_count') != count
                        or receipt.get('complete') is not True
                        or receipt.get('index_sha256') != common.sha(index)):
                    raise ValueError('All complete642 candidate prefix indices must be exact')
                self._admit(index, common.sha(index)); self._admit(done, common.sha(done))
        return self._completion(banks / 'complete.json', validate)

    def priority_complete(self):
        view = getattr(self, '_group_view', self)
        path = Path(view.registered_requests[6]['root']) / 'training_and_BEST26_complete.json'
        def validate(value):
            if (value.get('epochs') != 250 or value.get('checkpoint') != 'checkpoint_best.pth'
                    or value.get('evaluation_checkpoint_is_final') is not False
                    or not isinstance(value.get('validation_cases'), list) or len(value['validation_cases']) != 26
                    or len(set(value['validation_cases'])) != 26):
                raise ValueError('GPU6 priority release needs real250epoch and full26 BEST validation')
            fold = Path(value['fold'])
            for filename, field in (('checkpoint_best.pth', 'best_sha256'),
                                    ('checkpoint_final.pth', 'final_training_witness_sha256')):
                if common.sha(fold / filename) != value[field]:
                    raise ValueError('GPU6 actual whole BEST/final training witness SHA changed')
        if not self._completion(path, validate):
            return False
        evidence_path = path.parent / 'result_evidence.json'
        if not evidence_path.exists():
            return False
        evidence = json.loads(evidence_path.read_bytes())
        verification = evidence.get('verification', {})
        experiments = evidence.get('experiments', [])
        if (verification.get('full_training') is not True
                or verification.get('full_best_native_evaluation') is not True
                or verification.get('lesion_metrics') is not True
                or len(experiments) != 1):
            raise ValueError('GPU6 priority needs completed full native and historical metrics evidence')
        completion = experiments[0].get('completion', {})
        if (completion.get('completed_epochs') != 250 or completion.get('predicted_cases') != 26
                or completion.get('expected_cases') != 26
                or completion.get('lesion_metrics_status') != 'VERIFIED_COMPLETE'):
            raise ValueError('GPU6 actual250/BEST26 historical completion required')
        return True

    def _phase_receipt(self, name, value):
        """New immutable operational evidence; never replaces scientific files."""
        raw = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
        digest = hashlib.sha256(raw).hexdigest()
        directory = Path(self.request['RAM512_claim_ledger']).parent
        path = directory / ('RAM512_' + name + '_' + digest + '.json')
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError('Immutable RAM512 phase evidence bytes changed')
        else:
            temporary = path.with_name('.' + path.name + '_' + str(os.getpid()) + '_' + str(time.time_ns()))
            try:
                with temporary.open('xb') as stream:
                    stream.write(raw); stream.flush(); os.fsync(stream.fileno())
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    if path.read_bytes() != raw:
                        raise ValueError('Concurrent RAM512 phase evidence differs')
            finally:
                if temporary.exists():
                    temporary.unlink()
        return dict(path=str(path), sha256=digest)

    def _phase_observations(self, state):
        source_ready = self.source_complete()
        captured = state.get('source_captured', [])
        resolver = ProcessResolver(self)
        source_released = (source_ready and bool(captured) and state.get('source_actor_identity') is not None
            and not any(claim['kind'] == 'cpu_source'
            for claim in state['claims'].values())
            and all(resolver.inspect(actor)['state'] == 'gone' for actor in captured)
            and not resolver.has_cuda(captured))
        source_proof = None
        if source_released:
            source_path = Path(getattr(self, '_group_view', self).registered_requests[5]['shared_banks']) / 'complete.json'
            source_proof = self._phase_receipt('source_release', dict(
                format='basic_cp_RAM512_actual_source_release_v1',
                candidate_counts={'512': 642, '1024': 642, '2048': 642},
                producer_fully_released=True, captured=captured,
                scientific_completion=dict(path=str(source_path), sha256=common.sha(source_path)),
                group_registry_sha256=getattr(self, '_group_view', self).registry_sha256))
        priority_ready = self.priority_complete()
        priority_proof = None
        if priority_ready:
            root = Path(getattr(self, '_group_view', self).registered_requests[6]['root'])
            priority_proof = self._phase_receipt('priority_complete', dict(
                format='basic_cp_RAM512_actual_priority_complete_v1', completed_epochs=250,
                evaluated_cases=26, checkpoint='BEST', historical_metrics_complete=True,
                training_proof_sha256=common.sha(root / 'training_and_BEST26_complete.json'),
                result_evidence_sha256=common.sha(root / 'result_evidence.json'),
                group_registry_sha256=getattr(self, '_group_view', self).registry_sha256))
        return dict(source_complete=source_released, source_completion_proof=source_proof,
                    priority_complete=priority_ready, priority_completion_proof=priority_proof)

    def _completed_source_controller_aux(self, state, view, tree, descendants):
        """Recognize cleanup after a real bound producer has fully stopped.

        Ledger refresh may remove its finished child claim before the parent
        finally restores CPU8. This grants no source/model work: only the exact
        controller and literal read-only GPU/PBS probes remain auxiliary RSS.
        The original phase proof still requires every captured PID fully gone.
        """
        captured = state.get('source_captured')
        actor = state.get('source_actor_identity')
        if not isinstance(captured, list) or not captured or actor is None:
            raise RuntimeError('CPU64 source actor has no atomic prospective/bound source reservation: '
                               'cleanup lacks its exact previously bound source provenance')
        actor = _normalize_identity(actor)
        identities = [_normalize_identity(value) for value in captured]
        if actor not in identities or not self.source_complete():
            raise RuntimeError('CPU64 cleanup requires the original bound actor and actual full642 banks')
        resolver = ProcessResolver(self)
        observed = [resolver.inspect(value) for value in identities]
        if any(value['state'] == 'live' for value in observed) or resolver.has_cuda(identities):
            raise RuntimeError('Live source or CUDA actors cannot lose their atomic source reservation')
        allowed = {
            ('nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'),
            ('nvidia-smi', '--id=5',
             '--query-gpu=index,uuid,name,memory.total,memory.free,mig.mode.current',
             '--format=csv,noheader,nounits'),
            ('/opt/pbs/bin/qstat', '-f', '-F', 'json', self.lifetime['job_id']),
        }
        probes = []
        for process in descendants.values():
            try:
                metadata = common._metadata(process)
                command = process.cmdline()
                if tuple(command) not in allowed:
                    previous = view._managed.get(process.pid)
                    known_probe = (previous is not None
                        and all(previous[key] == metadata[key] for key in ('pid', 'create_time', 'UID'))
                        and tuple(previous['command']) in allowed)
                    if not (not command and common._kernel_exiting(process) and known_probe):
                        if (not command and common._kernel_exiting(process)
                                and view._terminal_after_change(process, metadata,
                                    reason='finished source read-only probe naturally exiting')):
                            continue
                        raise RuntimeError('Unexpected live child remains after the completed source claim')
                _same_metadata(process, metadata)
                probes.append(dict(metadata, command=command,
                    source_work_authorized=False, auxiliary_RSS_still_accounted=True))
            except psutil.NoSuchProcess:
                continue
        if _pids_have_cuda(set(tree)):
            raise RuntimeError('Completed source controller cleanup created a forbidden CUDA context')
        return dict(completed_source_controller_auxiliary_only=True,
            bound_source_actor_identity=actor,
            captured_source_actual_states=[value['state'] for value in observed],
            completed_source_readonly_probe_observations=probes,
            source_work_or_CUDA_permission_granted=False,
            controller_CPU_restoration_not_yet_complete=True)

    def _resources(self, controller, own, tree):
        view = getattr(self, '_group_view', self)
        view._inputs()
        observed = common.capture_cgroup_admission(controller.pid)
        if self._controller_cgroup is None:
            self._controller_cgroup = observed
        elif observed != self._controller_cgroup:
            raise RuntimeError('RAM512 exact controller cgroup/ancestor limits changed')
        declared = self.request.get('controller_cgroup_admission')
        if declared is not None and declared != observed:
            raise RuntimeError('RAM512 controller differs from actual cgroup declaration')
        current = common.capture_cgroup_admission(own.pid)
        for admission in (observed, current):
            for row in admission['memory_limits']:
                if row['limit_bytes'] is not None and row['limit_bytes'] < AGGREGATE_CAPACITY:
                    raise MemoryError('Actual inherited cgroup cannot admit approved512GiB')
        state = self._claim_state()
        if (not isinstance(state, dict) or type(state.get('source_pending')) is not bool
                or not isinstance(state.get('claims'), dict)):
            raise ValueError('Exact atomic RAM512 claim state required')
        pending = state['source_pending']
        if not pending and not self.source_complete():
            raise ValueError('Released source reservation lacks actual complete full642 banks')
        claims = state['claims']
        native_claims = {}
        source_claims = []
        for claim in claims.values():
            if claim['kind'] == 'cpu_source':
                source_claims.append(claim)
            elif claim['kind'] == 'native':
                gpu = claim['gpu']
                if gpu in native_claims:
                    raise RuntimeError('Duplicate prospective/native allocation on the same GPU')
                native_claims[gpu] = claim
            else:
                raise ValueError('Unknown RAM512 allocation kind')
        if len(source_claims) > 1 or (not pending and source_claims):
            raise RuntimeError('Exact source reservation lifecycle changed')
        if pending and (len(native_claims) > 1 or set(native_claims) & {3, 4, 5}):
            raise MemoryError('Source phase requires one native slot and no candidate native actors')
        inventory, inventory_rss, inventory_ids = own_process_inventory()
        registered = {}; observations = {}; native_rss = {}; source_rss = 0
        compute_pids = set()
        closing_view = None
        bound_source = [claim for claim in source_claims if claim.get('child_identity')]
        if bound_source:
            source_request = view.registered_requests[5]
            source_controller, _, source_affinity, _ = view._controller_stage_snapshot(5)
            if source_controller is None or source_affinity == source_request['cpu_affinity']:
                closing_view = (view if isinstance(view, _ClosingSourceObservation)
                    else _ClosingSourceObservation.from_admitted_view(view, state))
        for gpu in range(7):
            row_tree, observation = (closing_view._group_tree(5)
                if gpu == 5 and closing_view is not None else view._group_tree(gpu))
            observations[str(gpu)] = observation
            descendants = {pid: process for pid, process in row_tree.items()
                           if pid != view.registration[str(gpu)]['controller_identity']['pid']}
            for pid, process in row_tree.items():
                try:
                    before = common._metadata(process)
                    if before['UID'] != common.UID:
                        raise RuntimeError('Foreign UID in RAM512 registered group')
                    if process.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                        continue
                    rss = process.memory_info().rss
                    _same_metadata(process, before)
                    if pid in registered:
                        raise RuntimeError('Overlapping registered RAM512 process trees')
                    registered[pid] = rss
                    inventory[pid] = process; inventory_rss[pid] = rss; inventory_ids[pid] = before
                except psutil.NoSuchProcess:
                    continue
            claim = source_claims[0] if gpu == 5 and observation['shared_CPU_bank_stage'] and source_claims else native_claims.get(gpu)
            if observation['shared_CPU_bank_stage'] and claim is None:
                observation.update(self._completed_source_controller_aux(
                    state, view, row_tree, descendants))
            claimed = set()
            if claim is not None:
                child = claim.get('child_identity')
                captured = claim.get('captured', [])
                roots = [_normalize_identity(value) for value in captured]
                if child is not None:
                    roots.append(_normalize_identity(child))
                identities = {value['pid']: value for value in roots}
                for pid, process in descendants.items():
                    if pid in identities:
                        before = common._metadata(process)
                        identity = identities[pid]
                        if any(before[key] != identity[key] for key in ('pid', 'create_time', 'UID')):
                            raise RuntimeError('Bound claim PID/birth/UID was reused')
                        claimed.add(pid)
                    elif child is not None:
                        try:
                            if any(parent.pid == child['pid'] and parent.create_time() == child['create_time']
                                   for parent in process.parents()):
                                claimed.add(pid)
                        except psutil.NoSuchProcess:
                            continue
                # Exact pending command identifies a fast-start child before
                # the parent .bind_child; child self-binding closes this race.
                if child is None:
                    for pid, process in descendants.items():
                        try:
                            if process.cmdline() == claim['command']:
                                claimed.add(pid)
                        except psutil.NoSuchProcess:
                            continue
            # qstat and other controller-only observation children are AUX.
            # A model/source argv without its published claim cannot run Torch.
            for pid, process in descendants.items():
                if pid in claimed or pid not in registered:
                    continue
                try:
                    command = process.cmdline()
                    if '--stage' in command or any('basic_cp_large_bank' in item for item in command):
                        raise RuntimeError('Live model/source actor is outside its atomic RAM512 allocation')
                except psutil.NoSuchProcess:
                    continue
            compute_pids.update(claimed)
            total = sum(registered[pid] for pid in claimed if pid in registered)
            if observation['shared_CPU_bank_stage']:
                source_rss += total
                if claimed and _pids_have_cuda(claimed):
                    raise RuntimeError('CPU-only source actor created a forbidden CUDA context')
            elif claim is not None:
                native_rss[gpu] = total
        if self.self_identity != self.request['controller_identity'] and own.pid not in compute_pids:
            raise RuntimeError('Torch/source child must bind its exact prospective RAM512 allocation before work')
        auxiliary_pids = set(inventory_rss) - compute_pids
        aux_rss = sum(inventory_rss[pid] for pid in auxiliary_pids)
        store = allocated_tmpfs(self.request['owned_tmpfs_root'], inventory)
        memory = psutil.virtual_memory()
        reservations = state.get('storage_reservations', {})
        if not isinstance(reservations, dict):
            raise ValueError('Exact projected archive reservations required')
        projected = 0
        for reservation in reservations.values():
            value = reservation.get('bytes')
            if type(value) is not int or value < 0:
                raise ValueError('Exact nonnegative projected archive bytes required')
            written = archive_allocated_bytes(reservation, self.request['owned_tmpfs_root'])
            projected += max(0, value-written)
        result = enforce_accounting(store_bytes=store['owned_tmpfs_allocated_bytes'],
            native_rss=native_rss, source_rss=source_rss, auxiliary_rss=aux_rss,
            source_pending=pending, host_total=memory.total, host_available=memory.available,
            registered_rss=sum(registered[pid] for pid in compute_pids if pid in registered),
            projected_storage_bytes=projected)
        free = shutil.disk_usage(self.request['native_root']).free
        if free < common.STORAGE_FLOOR:
            raise OSError('Native output has less than unchanged10GiB free floor')
        result.update(store, own_process_tree_RSS_bytes=self._rss(tree),
            own_process_tree_RSS_limit_bytes=SOURCE_RSS_LIMIT if isinstance(self, SharedBankLease) else NATIVE_RSS_LIMIT,
            aggregate_native_and_cache_capacity_bytes=AGGREGATE_CAPACITY,
            actual_controller_cgroup=observed, actual_stage_cgroup=current,
            native_free_bytes=free, native_free_floor_bytes=common.STORAGE_FLOOR,
            group_process_tree_RSS_bytes=sum(registered.values()),
            group_RSS_deduplicated_PIDs=sorted(registered),
            group_controller_observations=observations,
            auxiliary_process_PIDs=sorted(auxiliary_pids), auxiliary_process_identities={str(pid): inventory_ids[pid] for pid in auxiliary_pids},
            group_registry_path=str(view.registry_path), group_registry_sha256=view.registry_sha256,
            RAM_admission_formula='actual_available_RAM + owned_tmpfs_allocated_blocks + exact_registered_RSS + actual_auxiliary_RSS >= 512GiB',
            scientific_scale_preserved=True, original_native_request_modified=False,
            PBS_GPU_owned_reservation=False, PBS_CPU_owned_reservation=False, PBS_job_role='host_lifetime_only',
            own_GPU=self.request['gpu'], own_CPU_affinity=list(self.request['cpu_affinity']))
        result.update(self._phase_observations(state))
        return result

    def reservation_observer(self, claims_snapshot, prospective_claim):
        """Single raw observation under the claims mutex; never sleeps/pauses."""
        from .basic_cp_512_claims import ResourceClaimUnavailable, reservation_bytes
        self._inputs()
        common._exact(self.request['controller_identity']); common._exact(self.self_identity)
        self._host_lifetime(force=True)
        remaining = min(self.lifetime['expires_at_unix'] - time.time(),
                        self.monotonic_expiry - time.monotonic())
        if remaining < common.EXPIRY_RESERVE_SECONDS:
            raise RuntimeError('RAM512 prospective admission crossed actual180second lifetime boundary')
        state = copy.deepcopy(claims_snapshot)
        if prospective_claim is not None:
            if any(claim['gpu'] == prospective_claim['gpu'] for claim in state['claims'].values()):
                raise ResourceClaimUnavailable('Exact GPU allocation is retained until full child release', state)
            if (prospective_claim['kind'] == 'native' and state['source_pending']
                    and prospective_claim['gpu'] in (3, 4, 5)):
                raise ResourceClaimUnavailable('Candidate native awaits complete CPU source release', state)
            state['claims']['__prospective__'] = copy.deepcopy(prospective_claim)
        reservation_bytes(state)
        self._claim_override = state
        try:
            if prospective_claim is None:
                controller = common._exact(self.request['controller_identity'])
                native = self.native_request if hasattr(self, 'native_request') else self.request
                if controller.cpu_affinity() == SHARED_CPU_AFFINITY:
                    if self.request['gpu'] != 5:
                        raise RuntimeError('Only exact GPU5 source may use CPU64 observation')
                    if self._shared_observer_lease is None:
                        self._shared_observer_lease = SharedBankLease(native)
                    cpu = self._shared_observer_lease
                    cpu._claim_override = state
                    try:
                        # Direct raw inherited CPU verification: no pause or
                        # ledger snapshot while this observer holds its mutex.
                        return shared.SharedBankLease.verify(cpu, force=True)
                    finally:
                        cpu._claim_override = None
                source_claims = [claim for claim in state['claims'].values()
                    if claim['kind'] == 'cpu_source']
                if (self.request['gpu'] == 5 and any(
                        claim.get('child_identity') for claim in source_claims)):
                    if (self.request['gpu'] != 5 or len(source_claims) != 1
                            or controller.cpu_affinity() != native['cpu_affinity']
                            or self.self_identity != native['controller_identity']):
                        raise RuntimeError('Only exact GPU5 native8 controller may observe its retained CPU source release')
                    closing = _ClosingSourceObservation(native, state)
                    own = common._exact(closing.self_identity)
                    if own.pid != controller.pid or own.cpu_affinity() != native['cpu_affinity']:
                        raise RuntimeError('Source release observer must remain its exact native8 controller')
                    closing._claim_override = state
                    # The parent finally restores native8 before the claim
                    # closes. Remaining bound source descendants keep CPU64,
                    # source320 and noCUDA until their real natural release.
                    tree, detail = closing._group_tree(5)
                    result = _RAM512._resources(closing, controller, own,
                                               {controller.pid: controller})
                    result.update(SharedBankLease._gpu(closing, tree, idle=False))
                    result.update(source_release_observation=detail)
                    return result
                if hasattr(self, 'native_request'):
                    # A SharedLease's effective request remains CPU64. Once
                    # its source is actually gone, inspect native8 through a
                    # fresh private native lease and the same held state.
                    native_view = BasicLease(native)
                    native_view._claim_override = state
                    return group.BasicLease.observe_waiting(native_view)
                # CPU preparation needs exact UUID evidence, not CUDA ownership.
                return group.BasicLease.observe_waiting(self)
            return group.BasicLease.verify(self, force=True, idle=True)
        finally:
            self._claim_override = None

    def allocation_reservation(self, stage, command):
        from .basic_cp_512_claims import AllocationReservation
        native = self.native_request if hasattr(self, 'native_request') else self.request
        return AllocationReservation(self.claim_ledger(), native, stage, command,
                                     self.reservation_observer)

    def archive_reservation(self, byte_count, *, destination=None):
        return self.claim_ledger().reserve_storage(byte_count, self.reservation_observer,
                                                  destination=destination)


class BasicLease(_RAM512, pause._RAMPause, group.BasicLease):
    """Native32 operational ceiling with original identity/CPU/PBS/GPU guard."""


class _CPUStageGroupLease(_RAM512, group.BasicLease):
    """Raw all-seven observer only; CPU64 own RSS may use source320."""


class _ClosingSourceObservation(BasicLease):
    """Retain exactly bound CPU64 survivors after their parent restores CPU8."""
    def __init__(self, request, state):
        super().__init__(request)
        self._closing_source_state = copy.deepcopy(state)

    @classmethod
    def from_admitted_view(cls, view, state):
        # Every peer request/SHA/registry has already been admitted by this
        # exact group view. Observe its peer without creating another stage
        # lease or pretending the current native actor belongs to GPU5.
        result = object.__new__(cls)
        result.__dict__.update(view.__dict__)
        for name in ('_managed', '_fork_pending', '_terminal_observations'):
            if hasattr(view, name):
                setattr(result, name, copy.deepcopy(getattr(view, name)))
        result._closing_source_state = copy.deepcopy(state)
        return result

    def _group_tree(self, gpu):
        if gpu != 5:
            return super()._group_tree(gpu)
        claims = [claim for claim in self._closing_source_state['claims'].values()
                  if claim['kind'] == 'cpu_source']
        if len(claims) != 1:
            raise RuntimeError('Exactly one retained source claim is required for closing CPU observation')
        claim = claims[0]
        control_identity = self.registered_requests[5]['controller_identity']
        if (claim['gpu'] != 5 or claim['stage'] != 'candidate_bank'
                or claim['controller_identity'] != _normalize_identity(control_identity)
                or not claim.get('child_identity')
                or claim['child_identity']['command'] != claim['command']):
            raise RuntimeError('Closing source must retain its exact bound producer/controller command')
        controller, control_observation, affinity, control_children = self._controller_stage_snapshot(5)
        native_cpu = self.registered_requests[5]['cpu_affinity']
        if controller is not None and affinity != native_cpu:
            raise RuntimeError('Closing source controller must preserve its real native8 affinity')
        identities = {_normalize_identity(value)['pid']: _normalize_identity(value)
                      for value in [claim['child_identity'], *claim['captured']]}
        root_identity = _normalize_identity(claim['child_identity'])
        candidates = {}
        for expected in identities.values():
            try:
                process = psutil.Process(expected['pid'])
                actual = common._metadata(process)
                if actual['create_time'] != expected['create_time']:
                    continue # The original birth released; no new PID permission.
                if actual['UID'] != expected['UID'] or actual['UID'] != common.UID:
                    raise RuntimeError('Closing source retained PID/birth changed its exact UID')
                if process.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    continue
                command = process.cmdline()
                if command != expected['command'] and not (
                        not command and common._kernel_exiting(process)):
                    raise RuntimeError('Closing source retained live actor changed its exact argv')
                _same_metadata(process, actual)
                candidates[process.pid] = process
            except psutil.NoSuchProcess:
                continue
        producer = candidates.get(root_identity['pid'])
        producer_descendants = {}
        if producer is not None:
            for process in producer.children(recursive=True):
                producer_descendants[process.pid] = process
            candidates.update(producer_descendants)
        for process in control_children:
            if process.pid not in identities and process.pid not in producer_descendants:
                raise RuntimeError('Unexpected live actor accompanies the exact closing source claim')
            candidates[process.pid] = process
        tree = {} if controller is None else {controller.pid: controller}
        terminal = []
        for process in candidates.values():
            try:
                process = self._descendant(process)
                if process is None:
                    continue
                metadata = common._metadata(process); actual = process.cpu_affinity()
                if actual != SHARED_CPU_AFFINITY:
                    detail = self._affinity_details(5, process, metadata, actual,
                                                    SHARED_CPU_AFFINITY, 'closing_source_descendant')
                    if not self._retain_exiting_affinity(process, metadata, detail):
                        raise RuntimeError('Exact retained source descendant left its admitted64CPU affinity')
                    terminal.append(process.pid)
                _same_metadata(process, metadata)
                tree[process.pid] = process
            except psutil.NoSuchProcess:
                continue
        detail = dict(controller_identity=control_identity,
            actual_CPU_affinity=None if controller is None else list(native_cpu),
            shared_CPU_bank_stage=True,
            closing_source_controller_native8=controller is not None,
            released_source_controller_observation=control_observation,
            retained_source_descendant_CPU_affinity=list(SHARED_CPU_AFFINITY),
            surviving_exact_descendant_PIDs=sorted(tree),
            exact_kernel_exiting_unadmitted_CPU_mask_PIDs=terminal,
            source_claim_token=claim['token'], captured_source_identity_count=len(identities))
        return tree, detail


class SharedBankLease(_RAM512, pause._RAMPause, shared.SharedBankLease):
    """CPU64/24-worker source320; every source remains fully generated."""
    def __init__(self, request):
        super().__init__(request)
        self._group_view = _CPUStageGroupLease(self.native_request)

    @staticmethod
    def _small_contract(value):
        result = shared.SharedBankLease._small_contract(value)
        result.update({key: value[key] for key in (*FIELDS, 'owned_tmpfs_root', 'RAM512_claim_ledger', 'RAM512_source_pending_receipt')})
        return result

    def _gpu(self, tree, *, idle):
        """CPU source may observe an occupied GPU but never creates CUDA."""
        row = [part.strip() for part in common._query(['nvidia-smi', '--id=' + str(self.request['gpu']),
            '--query-gpu=index,uuid,name,memory.total,memory.free,mig.mode.current',
            '--format=csv,noheader,nounits'], timeout=5).split(',')]
        if (len(row) != 6 or row[0] != str(self.request['gpu']) or row[1] != self.request['gpu_uuid']
                or row[5] not in ('N/A', 'Disabled', '[N/A]')
                or not common.math.isfinite(float(row[3]))
                or float(row[3]) != self.request['GPU_memory_total_MiB']
                or not common.math.isfinite(float(row[4]))):
            raise RuntimeError('CPU source exact GPU UUID/MIG/full-memory identity changed')
        if _pids_have_cuda(set(tree)):
            raise RuntimeError('CPU-only shared source initialized a forbidden CUDA context')
        return dict(GPU_name=row[2], GPU_UUID=row[1], GPU_total_MiB=float(row[3]),
            GPU_free_MiB=float(row[4]), MIG=row[5], own_GPU_compute_PIDs=[],
            shared_bank_CPU_only_CUDA_verified=True, CUDA_ownership_not_granted=True)

    def _resources(self, controller, own, tree):
        result = _RAM512._resources(self, controller, own, tree)
        cpus = psutil.cpu_count(logical=True)
        if cpus is None or cpus <= max(SHARED_CPU_AFFINITY):
            raise RuntimeError('Actual host CPU inventory cannot admit measured64CPU affinity')
        result.update(actual_host_logical_CPU_count=cpus,
            native_CPU_affinity_preserved=list(self.native_request['cpu_affinity']),
            native_RSS_limit_bytes_preserved=self.native_request['RSS_limit_bytes'],
            shared_bank_RAM_budget_bytes=SOURCE_RSS_LIMIT,
            execution_stage='shared_candidate_bank_CPU_only', own_CPU_affinity=SHARED_CPU_AFFINITY,
            original_native_request_modified=False)
        return result


def allocation_reservation(native, stage, command):
    """Serialize prospective admission before the runner creates any child."""
    lease = BasicLease(native)
    return lease.allocation_reservation(stage, command)


def check_initial_GPU_idle(native):
    """Second actual idle check immediately before Popen, without another claim.

    A CPU producer has already expanded to64 CPUs at this point. The original
    GPU method receives its real CPU64 view, rather than forging native8 CPU
    evidence. Runtime CPU preparation subsequently permits outside CUDA while
    still forbidding all own source CUDA contexts.
    """
    validate_request(native)
    controller = common._exact(native['controller_identity'])
    shared_cpu = controller.cpu_affinity() == SHARED_CPU_AFFINITY
    if shared_cpu and native['gpu'] != 5:
        raise RuntimeError('Only exact GPU5 CPU producer may use shared64 affinity')
    lease = SharedBankLease(native) if shared_cpu else BasicLease(native)
    own = common._exact(lease.self_identity)
    lease._inputs(); lease._host_lifetime(force=True)
    remaining = min(lease.lifetime['expires_at_unix'] - time.time(),
                    lease.monotonic_expiry - time.monotonic())
    if remaining < common.EXPIRY_RESERVE_SECONDS:
        raise RuntimeError('Pre-Popen actual host lifetime crossed180second boundary')
    tree = lease._tree(controller, own)
    lease._rss(tree)
    return common.BasicLease._gpu(lease, tree, idle=True)


__all__ = ['BasicLease', 'SharedBankLease', 'allocation_reservation',
           'check_initial_GPU_idle',
           'validate_request', 'enforce_accounting', 'allocated_tmpfs',
           'own_process_inventory', 'ProcessResolver', 'GUARD_FILE', 'CLAIMS_FILE',
           'RAMAdmissionUnavailable', 'AGGREGATE_CAPACITY', 'NATIVE_RSS_LIMIT',
           'SOURCE_RSS_LIMIT', 'SOURCE_STORE_LIMIT', 'POST_SOURCE_STORE_LIMIT', 'AUX_RSS_LIMIT']
