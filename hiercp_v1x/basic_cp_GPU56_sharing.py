"""Explicit physical GPU5/6 sharing; logical experiments retain their contracts.

No process is started, signalled, or adopted here.  CUDA peers must have both an
immutable same-cohort registration and an atomic native allocation.  The RAM
ledger continues to charge native32 per logical experiment, not per device.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import psutil

from .basic_cp_512_claims import ResourceClaimUnavailable, identity, _regular
from .basic_cp_gpu_admission import GPUAdmissionUnavailable

FILE = 'hiercp_v1x/basic_cp_GPU56_sharing.py'
FIELD = 'GPU56_sharing_receipt'
FORMAT = 'basic_cp_GPU56_sharing_v1'
OPERATION = 'resume_six_unfinished_on_GPU5_GPU6'
ARMS = {0: 'multi5', 1: 'blend50', 2: 'multi10', 3: 'candidates512',
        4: 'candidates1024', 5: 'candidates2048', 6: 'multi7'}
CPU_STAGES = {'preflight', 'historical_metrics'}
CUDA_STAGES = {'train', 'debug_B2', 'debug_B4'}
MIB = 2 ** 20


def _checksum(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _read_reference(reference, admit=None):
    if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}
            or not isinstance(reference['path'], str) or not Path(reference['path']).is_absolute()
            or '..' in Path(reference['path']).parts or not _checksum(reference['sha256'])):
        raise ValueError('Exact absolute GPU56 evidence path/SHA required')
    path = Path(reference['path'])
    if path.resolve(strict=True) != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('GPU56 evidence must retain its exact nonsymlink path')
    before = _regular(path)  # Exact UID41833/regular/nonsymlink even without lease._admit.
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_uid', 'st_gid', 'st_mode')
    signature = lambda value: tuple(getattr(value, name) for name in fields)
    if admit is not None:
        admit(path, reference['sha256'])
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0))
    with os.fdopen(descriptor, 'rb') as stream:
        if signature(os.fstat(stream.fileno())) != signature(before):
            raise RuntimeError('GPU56 evidence inode changed before its pinned read')
        raw = stream.read()
        if signature(os.fstat(stream.fileno())) != signature(before):
            raise RuntimeError('GPU56 held evidence inode changed during its pinned read')
    if signature(_regular(path)) != signature(before):
        raise RuntimeError('GPU56 evidence path was replaced during its pinned read')
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise RuntimeError('Immutable GPU56 receipt/evidence SHA changed')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('GPU56 receipt/evidence must be a JSON mapping')
    return value


def validate_sharing_request(request, admit=None):
    if FIELD not in request:
        if 'execution_physical_GPU' in request or 'execution_GPU_UUID' in request:
            raise ValueError('Execution GPU overrides require the explicit GPU56 receipt')
        return None
    if not _checksum(request.get('code_sha256', {}).get(FILE)):
        raise ValueError('GPU56 execution requires its exact sharing module SHA')
    reference = request[FIELD]
    if (not isinstance(reference, dict) or not isinstance(reference.get('path'), str)
            or Path(reference['path']).parent != Path(request['root']).parent):
        raise ValueError('All GPU56 peers must reference their same-group sharing receipt')
    value = _read_reference(reference, admit)
    required = {'format', 'operation', 'logical_GPU_arms', 'execution_physical_GPUs',
        'execution_GPU_UUIDs', 'max_concurrent_CUDA_experiments_per_GPU',
        'VRAM_reservation_MiB_by_stage', 'VRAM_driver_floor_MiB', 'VRAM_measurement_evidence'}
    if (set(value) != required or value['format'] != FORMAT or value['operation'] != OPERATION
            or value['logical_GPU_arms'] != {str(k): v for k, v in ARMS.items()}
            or type(value['max_concurrent_CUDA_experiments_per_GPU']) is not int
            or value['max_concurrent_CUDA_experiments_per_GPU'] != 3
            or type(value['VRAM_driver_floor_MiB']) is not int
            or value['VRAM_driver_floor_MiB'] < 1024):
        raise ValueError('Exact six-experiment GPU5/6 sharing contract required')
    mapping, uuids = value['execution_physical_GPUs'], value['execution_GPU_UUIDs']
    if (not isinstance(mapping, dict) or set(mapping) != {str(i) for i in range(7)}
            or mapping['1'] is not None or not isinstance(uuids, dict) or set(uuids) != {'5', '6'}
            or any(type(mapping[str(i)]) is not int or mapping[str(i)] not in (5, 6)
                   for i in range(7) if i != 1)
            or any(not isinstance(u, str) or re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', u) is None
                   for u in uuids.values()) or uuids['5'] == uuids['6']):
        raise ValueError('Only exact GPU5/6 UUID mappings and completed logical1 are permitted')
    reservations = value['VRAM_reservation_MiB_by_stage']
    if (not isinstance(reservations, dict) or not {'train', *CPU_STAGES} <= set(reservations)
            or not set(reservations) <= CUDA_STAGES | CPU_STAGES
            or any(type(amount) is not int or amount < 0 for amount in reservations.values())
            or any(reservations[stage] != 0 for stage in CPU_STAGES)
            or any(reservations[stage] < 14 * 1024 for stage in set(reservations) & CUDA_STAGES)):
        raise ValueError('Measured stage VRAM reservations and CPU-only zero reservations required')
    evidence = _read_reference(value['VRAM_measurement_evidence'], admit)
    measurements = []
    # Actual read-only profiler evidence, tied to the unchanged full model.
    for row in evidence.get('experiments', {}).values():
        model = row.get('model', {})
        peak = row.get('peak_cuda_bytes')
        if peak is not None:
            if (type(peak) is not int or peak <= 0 or model.get('parameters') != 102350575
                    or model.get('physical_batch') != 2 or model.get('epochs') != 250
                    or model.get('patch_size') != [128, 128, 128]):
                raise ValueError('VRAM evidence must measure the actual full B2/128cubed/250epoch model')
            measurements.append(peak)
    if not measurements or max(measurements) + 2 * 1024 * MIB > reservations['train'] * MIB:
        raise ValueError('Training VRAM reservation must cover actual full-model peak plus2GiB')
    # DEBUG stages need their own measured witness; absence is an error, never
    # permission to rerun a larger calibration beneath a training reservation.
    for stage in set(reservations) & {'debug_B2', 'debug_B4'}:
        rows = evidence.get('DEBUG_stage_peak_cuda_bytes', {})
        peak = rows.get(stage)
        if type(peak) is not int or peak <= 0 or peak + 2 * 1024 * MIB > reservations[stage] * MIB:
            raise ValueError('Each additional DEBUG stage requires its measured peak plus2GiB')
    slot = request.get('gpu')
    if type(slot) is not int or slot not in ARMS or request.get('arm') != ARMS[slot]:
        raise ValueError('GPU56 request changed its logical experiment identity')
    target = mapping[str(slot)]
    uuid = None if target is None else uuids[str(target)]
    if request.get('execution_physical_GPU') != target or request.get('execution_GPU_UUID') != uuid:
        raise ValueError('Request physical execution target differs from its immutable sharing receipt')
    if 'GPU_queue_deferred_transfer' in request:
        raise ValueError('Superseded successor queues cannot coexist with simultaneous GPU56 sharing')
    return value


def execution_gpu(request):
    return (request['execution_physical_GPU'] if validate_sharing_request(request) is not None
            else request['gpu'])


def execution_uuid(request):
    return (request['execution_GPU_UUID'] if validate_sharing_request(request) is not None
            else request['gpu_uuid'])


def claim_fields(native, stage):
    receipt = validate_sharing_request(native)
    if receipt is None:
        return {}
    if native['gpu'] == 1 or stage not in receipt['VRAM_reservation_MiB_by_stage']:
        raise ValueError('Completed blend50, source regeneration, and unmeasured GPU stages cannot dispatch')
    return dict(execution_physical_GPU=native['execution_physical_GPU'],
        execution_GPU_UUID=native['execution_GPU_UUID'],
        GPU56_VRAM_reservation_MiB=receipt['VRAM_reservation_MiB_by_stage'][stage],
        GPU56_sharing_receipt_sha256=native[FIELD]['sha256'])


def verify_claim_capacity(snapshot, native):
    receipt = validate_sharing_request(native)
    if receipt is None:
        return {}
    if snapshot.get('source_pending') is not False:
        raise ValueError('GPU56 sharing requires actually completed full candidate preparation')
    per_gpu = {5: [], 6: []}
    for claim in snapshot['claims'].values():
        slot = claim.get('gpu')
        if type(slot) is not int or slot not in ARMS or slot == 1 or claim.get('kind') != 'native':
            raise ValueError('Only six exact unfinished native experiment claims are allowed')
        peer = dict(native, gpu=slot, arm=ARMS[slot],
            execution_physical_GPU=receipt['execution_physical_GPUs'][str(slot)],
            execution_GPU_UUID=receipt['execution_GPU_UUIDs'][str(receipt['execution_physical_GPUs'][str(slot)])])
        expected = claim_fields(peer, claim.get('stage'))
        if any(claim.get(key) != value for key, value in expected.items()):
            raise RuntimeError('Atomic GPU56 claim execution UUID, stage reservation, or receipt changed')
        if expected['GPU56_VRAM_reservation_MiB']:
            per_gpu[expected['execution_physical_GPU']].append(claim)
    result = {}
    for gpu, claims in per_gpu.items():
        reserved = sum(c['GPU56_VRAM_reservation_MiB'] for c in claims)
        if len(claims) > receipt['max_concurrent_CUDA_experiments_per_GPU']:
            raise ResourceClaimUnavailable('Physical GPU already has its three experiment slots',
                dict(physical_GPU=gpu, logical_slots=[c['gpu'] for c in claims]))
        result[str(gpu)] = dict(CUDA_experiments=len(claims), reserved_MiB=reserved,
            logical_slots=[c['gpu'] for c in claims])
    return dict(GPU56_VRAM_reservations=result,
        GPU56_driver_floor_MiB=receipt['VRAM_driver_floor_MiB'])


def _claimed_actor(lease, process, claim, tree, peer):
    """Admit an exact captured actor or a current descendant of its bound root."""
    from . import basic_cp_ablation_resources as common
    if process.pid not in tree or process.cpu_affinity() != peer['cpu_affinity']:
        return None
    observed = identity(common._actor(process))
    captures = [*claim.get('captured', [])]
    child = claim.get('child_identity')
    if child is not None:
        captures.append(child)
    direct = [identity(value) for value in captures if value['pid'] == process.pid]
    if direct:
        if len({json.dumps(value, sort_keys=True) for value in direct}) != 1 or observed != direct[0]:
            raise RuntimeError('CUDA captured PID/birth/UID/argv changed')
    elif child is not None:
        root = common._exact(child)
        if not any(parent.pid == root.pid and common._metadata(parent) == common._metadata(root)
                   for parent in process.parents()):
            return None
    else:
        parent = process.parent()
        if (observed['command'] != claim['command'] or parent is None
                or identity(common._actor(parent)) != identity(claim['controller_identity'])):
            return None
    if identity(common._actor(process)) != observed:
        raise RuntimeError('CUDA actor identity changed during exact sharing observation')
    return observed


def verify_gpu(lease, tree, *, idle, _headroom_attempt=0):
    """Observe same-cohort CUDA peers and prospective stage headroom exactly.

    Callers performing admission supply ``_claim_override`` from the existing
    ledger mutex.  Runtime callers read one complete atomic ledger snapshot.
    """
    from . import basic_cp_ablation_resources as common
    receipt = validate_sharing_request(lease.request, lease._admit)
    if receipt is None:
        raise ValueError('Sharing GPU verification requires its explicit receipt')
    target, uuid = lease.request['execution_physical_GPU'], lease.request['execution_GPU_UUID']
    if target is None:
        if lease.request['gpu'] != 1:
            raise RuntimeError('Only completed blend50 may lack a CUDA execution target')
        apps = common._query(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], timeout=5)
        if any(line.split(',')[0].strip().isdigit() and int(line.split(',')[0]) in tree
               for line in apps.splitlines() if line.strip()):
            raise RuntimeError('Completed blend50 controller unexpectedly owns CUDA work')
        return dict(GPU_available=False, physical_GPU=None, CUDA_ownership_not_granted=True,
            completed_experiment_GPU_dispatch_disabled=True)
    row = [x.strip() for x in common._query(['nvidia-smi', '--id=' + str(target),
        '--query-gpu=index,uuid,name,memory.total,memory.free,mig.mode.current',
        '--format=csv,noheader,nounits'], timeout=5).split(',')]
    if (len(row) != 6 or row[0] != str(target) or row[1] != uuid
            or row[5] not in ('N/A', 'Disabled', '[N/A]')
            or not all(math.isfinite(float(row[i])) for i in (3, 4))
            or float(row[3]) < 44 * 1024 or not 0 <= float(row[4]) <= float(row[3])):
        raise RuntimeError('GPU56 exact actual physical UUID/full VRAM/MIG observation changed')
    state = lease._claim_state()
    capacity = verify_claim_capacity(state, lease.request)
    claims = [c for c in state['claims'].values() if c['execution_physical_GPU'] == target]
    reserved = sum(c['GPU56_VRAM_reservation_MiB'] for c in claims)
    floor = receipt['VRAM_driver_floor_MiB']
    if reserved + floor > float(row[3]):
        raise ResourceClaimUnavailable('Physical GPU VRAM reservations plus driver floor exceed capacity',
            dict(physical_GPU=target, reserved_MiB=reserved, floor_MiB=floor, total_MiB=float(row[3])))
    view = getattr(lease, '_group_view', lease)
    admitted_trees = {}
    for claim in claims:
        slot = claim['gpu']; peer = view.registered_requests[slot]
        validate_sharing_request(peer, lease._admit)
        if peer[FIELD] != lease.request[FIELD]:
            raise RuntimeError('Sharing CUDA peer uses another immutable placement receipt')
        admitted_trees[slot] = view._group_tree(slot)[0]
    command = ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader,nounits']
    released, pending = {}, {}
    for attempt in range(3):
        used_by_slot = {c['gpu']: 0.0 for c in claims}; peers = {}; target_pids = set(); retry = False
        for line in common._query(command, timeout=5).splitlines():
            if not line.strip():
                continue
            cells = [x.strip() for x in line.split(',')]
            if len(cells) != 3 or not cells[0].isdigit():
                raise RuntimeError('Exact CUDA PID/UUID/used VRAM observation required')
            pid, actual_uuid = int(cells[0]), cells[1]
            if pid in tree and actual_uuid != uuid:
                raise RuntimeError('Registered CUDA actor initialized another physical GPU')
            if actual_uuid != uuid:
                continue
            if pid in target_pids:
                raise RuntimeError('Duplicate CUDA actor rows on one physical device')
            target_pids.add(pid)
            try:
                process = psutil.Process(pid)
                if process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    matched = []
                    for claim in claims:
                        if not claim['GPU56_VRAM_reservation_MiB']:
                            continue
                        slot = claim['gpu']
                        actor = _claimed_actor(lease, process, claim, admitted_trees[slot], view.registered_requests[slot])
                        if actor is not None:
                            matched.append((slot, actor))
                    if len(matched) != 1:
                        outside = lease._live_outside_cuda_actor(pid, uuid, tree)
                        if outside is not None:
                            outside.update(physical_GPU=target, GPU56_unknown_actor_rejected=True)
                            raise GPUAdmissionUnavailable(outside)
                    else:
                        amount = float(cells[2])
                        if not math.isfinite(amount) or amount < 0:
                            raise RuntimeError('Exact per-actor CUDA VRAM measurement required')
                        slot, actor = matched[0]; used_by_slot[slot] += amount; peers[pid] = actor
                        continue
            except psutil.NoSuchProcess:
                pass  # Only an actual vanished task reaches the bounded driver-release path.
            terminal = lease._terminal_cuda_actor(pid)
            if terminal is None:
                raise RuntimeError('CUDA actor changed identity before terminal release')
            pending[pid] = terminal; retry = True
        for pid in list(pending):
            if pid not in target_pids:
                released[pid] = dict(pending.pop(pid), driver_row_disappeared=True)
        if not retry:
            break
        if attempt == 2:
            raise RuntimeError('GPU56 stale CUDA driver row did not actually clear')
        time.sleep(.05)
    headroom = sum(max(0.0, c['GPU56_VRAM_reservation_MiB'] - used_by_slot[c['gpu']]) for c in claims)
    for claim in claims:
        if used_by_slot[claim['gpu']] > claim['GPU56_VRAM_reservation_MiB']:
            raise MemoryError('Exact GPU56 experiment exceeded its measured VRAM reservation: logical '
                + str(claim['gpu']))
    own_claims = [c for c in claims if c['gpu'] == lease.request['gpu']]
    # Free and per-process usage are separate driver observations. A live,
    # already-bound model retains its admitted future-peak reservation; a
    # train/validation allocation race must not stop that healthy model.
    admission = idle or not any(c.get('child_identity') is not None for c in own_claims)
    if admission and float(row[4]) < headroom + floor:
        if _headroom_attempt < 2:
            time.sleep(.05)
            return verify_gpu(lease, tree, idle=idle, _headroom_attempt=_headroom_attempt+1)
        raise ResourceClaimUnavailable('Actual free CUDA memory cannot cover reserved future stage peaks',
            dict(physical_GPU=target, free_MiB=float(row[4]), future_peak_headroom_MiB=headroom, driver_floor_MiB=floor))
    cuda_count = sum(bool(c['GPU56_VRAM_reservation_MiB']) for c in claims)
    available = bool(own_claims) or cuda_count < receipt['max_concurrent_CUDA_experiments_per_GPU']
    return dict(capacity, GPU_name=row[2], GPU_UUID=uuid, physical_GPU=target,
        logical_GPU=lease.request['gpu'], GPU_total_MiB=float(row[3]), GPU_free_MiB=float(row[4]),
        MIG=row[5], GPU_available=available, own_GPU_compute_PIDs=[pid for pid in peers if pid in tree],
        future_peak_headroom_MiB=headroom, free_headroom_checked_for_new_admission=admission,
        admitted_same_cohort_GPU_compute_identities=peers,
        naturally_released_GPU_actor_observations=released, scientific_scale_preserved=True)
