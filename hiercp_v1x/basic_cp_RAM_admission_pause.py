"""Explicit runtime RAM pause for unchanged Group5/Shared6 full-scale guards.

Only temporary available-RAM admission is retryable. Every paused observation
keeps exact processes, source/code, CPU, PBS expiry, GPU and hard resource caps.
The caller returns to its next native step/source only after full readmission.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time

from . import basic_cp_ablation_resources as common
from . import basic_cp_group_resources_v5 as group
from . import basic_cp_shared_bank_resources_v6 as shared
from .basic_cp_group_resources import RAMAdmissionUnavailable

GUARD_FILE = 'hiercp_v1x/basic_cp_RAM_admission_pause.py'
FORMAT = 'basic_cp_exact_same_actor_RAM_pause_v1'
PAUSE_SECONDS = 5
_STATUS_BASE = {'format', 'arm', 'physical_GPU', 'stage', 'status', 'time', 'epochs', 'checkpoint'}


def validate_pause_request(request):
    if (type(request.get('RAM_admission_pause_revision')) is not int
            or request['RAM_admission_pause_revision'] != 1
            or GUARD_FILE not in request.get('code_sha256', {})):
        raise ValueError('Explicit RAM admission pause revision1 and exact new guard SHA required')


def _atomic(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.' + str(time.time_ns()) + '.tmp')
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class _RAMPause:
    def __init__(self, request):
        validate_pause_request(request)
        super().__init__(request)
        view = getattr(self, '_group_view', self)
        for peer in view.registered_requests.values():
            validate_pause_request(peer)
        self._RAM_pause_state = None

    def _pause_boundary(self, *, idle, waiting_GPU=False):
        # _resources has already checked every cgroup and deduplicated hard
        # RSS cap before raising the typed shortage. It raises before _gpu and
        # the storage floor, so check those too while the caller is blocked.
        self._inputs()
        state_path = Path(self.request['root']) / 'status.json'
        if state_path.exists():
            state = json.loads(state_path.read_bytes())
            if state.get('status') == 'FAILED':
                raise RuntimeError('Same native/source actor failed during RAM pause: ' + str(state.get('error')))
        if common.os.getuid() != common.UID or common.socket.gethostname().split('.')[0] != common.NODE:
            raise RuntimeError('RAM pause exact execution UID or node changed')
        controller = common._exact(self.request['controller_identity'])
        own = common._exact(self.self_identity)
        remaining = min(self.lifetime['expires_at_unix'] - time.time(),
                        self.monotonic_expiry - time.monotonic())
        if remaining < common.EXPIRY_RESERVE_SECONDS:
            raise RuntimeError('RAM pause host lifetime crossed its unchanged180second boundary')
        tree = self._tree(controller, own)
        rss = self._rss(tree)
        if waiting_GPU:
            query = self._gpu.__func__.__globals__['_query']
            row = [cell.strip() for cell in query(['nvidia-smi', '--id=' + str(self.request['gpu']),
                '--query-gpu=index,uuid,name,memory.total,memory.free,mig.mode.current',
                '--format=csv,noheader,nounits'], timeout=5).split(',')]
            if (len(row) != 6 or row[0] != str(self.request['gpu']) or row[1] != self.request['gpu_uuid']
                    or row[5] not in ('N/A', 'Disabled', '[N/A]')
                    or not common.math.isfinite(float(row[3])) or float(row[3]) != self.request['GPU_memory_total_MiB']
                    or not common.math.isfinite(float(row[4]))):
                raise RuntimeError('RAM-pause waiting GPU UUID/MIG/full-memory identity changed')
            gpu = dict(GPU_UUID=row[1], GPU_total_MiB=float(row[3]), GPU_free_MiB=float(row[4]),
                       CUDA_ownership_not_granted=True)
        else:
            gpu = self._gpu(tree, idle=idle)
        free = common.shutil.disk_usage(self.request['native_root']).free
        if free < common.STORAGE_FLOOR:
            raise OSError('RAM pause native storage dropped below its unchanged10GiB floor')
        remaining = min(self.lifetime['expires_at_unix'] - time.time(),
                        self.monotonic_expiry - time.monotonic())
        if remaining < common.EXPIRY_RESERVE_SECONDS:
            raise RuntimeError('RAM pause host lifetime crossed its unchanged180second boundary')
        return dict(own_process_tree_RSS_bytes=rss, native_free_bytes=free,
                    host_lifetime_remaining_seconds=remaining, GPU=copy.deepcopy(gpu),
                    processes_signaled=False, same_actor_preserved=True)

    def _pause_file(self):
        digest = hashlib.sha256(json.dumps(self.self_identity, sort_keys=True).encode()).hexdigest()[:16]
        return Path(self.request['root']) / ('RAM_pause_actor_' + str(self.self_identity['pid']) + '_' + digest + '.json')

    def _publish_pause(self, failure, boundary, *, resumed=False):
        root = Path(self.request['root']); state_path = root / 'status.json'
        current = json.loads(state_path.read_bytes()) if state_path.exists() else None
        if current and current.get('status') == 'FAILED':
            raise RuntimeError('Same native/source actor failed during RAM pause: ' + str(current.get('error')))
        if self._RAM_pause_state is None:
            self._RAM_pause_state = copy.deepcopy(current)
        evidence = dict(format=FORMAT, status='READMITTED' if resumed else 'PAUSED',
            actor_identity=copy.deepcopy(self.self_identity),
            controller_identity=copy.deepcopy(self.request['controller_identity']),
            RAM_admission=copy.deepcopy(failure.observation), pause_boundary=copy.deepcopy(boundary),
            original_stage=self._RAM_pause_state.get('stage') if self._RAM_pause_state else None,
            original_status=self._RAM_pause_state.get('status') if self._RAM_pause_state else None,
            observed_at=time.time(), resource_caps_changed=False, optimizer_or_source_work_admitted=resumed,
            processes_signaled=False, RAM_admission_pause_revision=1)
        _atomic(self._pause_file(), evidence)
        # Workers never race over controller status. Its own guard observes the
        # same shortage and publishes the existing waiting_RAM state/UI fields.
        if self.self_identity == self.request['controller_identity']:
            from tools.run_basic_cp_ablation import status
            original = self._RAM_pause_state or {}
            metadata = {key:copy.deepcopy(value) for key,value in original.items() if key not in _STATUS_BASE}
            if current and type(current.get('completed_epochs')) is int:
                previous = metadata.get('completed_epochs')
                if previous is None or type(previous) is int and current['completed_epochs'] > previous:
                    metadata['completed_epochs'] = current['completed_epochs']
            if resumed and original:
                # Restore the exact prior phase/status; a GPU queue stays WAITING.
                status(self.request, original['stage'], original['status'], **dict(metadata,
                       RAM_pause_readmitted_same_actor=True, RAM_pause_actor_evidence=str(self._pause_file())))
            elif not resumed:
                status(self.request, 'waiting_RAM', 'WAITING', **dict(metadata, reason=str(failure),
                       RAM_admission=copy.deepcopy(failure.observation),
                       RAM_pause_original_stage=original.get('stage'),
                       RAM_pause_actor_evidence=str(self._pause_file())))
        return evidence

    def _run_with_pause(self, operation, arguments, *, waiting_GPU=False):
        failure = None
        while True:
            try:
                result = operation(**arguments)
            except RAMAdmissionUnavailable as shortage:
                boundary = (self._pause_boundary(idle=False, waiting_GPU=True) if waiting_GPU
                            else self._pause_boundary(idle=arguments.get('idle', False)))
                self._publish_pause(shortage, boundary)
                failure = shortage
                time.sleep(PAUSE_SECONDS)
                if 'force' in arguments:
                    arguments['force'] = True  # Never reuse a cached RAM admission.
                continue
            if failure is not None:
                boundary = dict(same_actor_preserved=True, processes_signaled=False,
                                readmitted_resources=copy.deepcopy(result))
                self._publish_pause(failure, boundary, resumed=True)
                self._RAM_pause_state = None
                result = dict(result, RAM_pause_readmitted_same_actor=True,
                              RAM_pause_actor_evidence=str(self._pause_file()))
            return result

    def verify(self, *, force=False, idle=False):
        return self._run_with_pause(super().verify, dict(force=force, idle=idle))

    def observe_waiting(self):
        # Bank-prefix/GPU queues also retain their same CPU actor on shortage.
        # This grants no CUDA permission and restores the prior WAITING state.
        return self._run_with_pause(super().observe_waiting, {}, waiting_GPU=True)


class BasicLease(_RAMPause, group.BasicLease):
    """Same native8CPU/96GiB, candidate384, total800 guard with a typed pause."""


class SharedBankLease(_RAMPause, shared.SharedBankLease):
    """Same CPU64/shared384/800 guard; per-candidate checks keep dynamic verify."""


__all__ = ['BasicLease', 'SharedBankLease', 'RAMAdmissionUnavailable', 'GUARD_FILE', 'FORMAT', 'PAUSE_SECONDS']
