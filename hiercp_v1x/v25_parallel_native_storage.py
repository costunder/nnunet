"""Explicit direct four-arm native leases around the unchanged storage admission.

The original full131 data admission, global9 checkpoint publication methods
and retention are reused byte for byte. Independent arm leases guard their
own GPU and CPU set; this module does not acquire a PBS reservation.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import time

from . import v25_native_storage as storage

core = storage.core
ARMS = storage.ARMS
canonical_hash = storage.canonical_hash
_OWNED_WAVE = None


def verify_admission(admission):
    return storage.verify_admission(admission)


def _publication_lock(path):
    return storage._publication_lock(path)


def _recovery_predecessor(document, arm, lease, *, predecessor_child, predecessor_status):
    if lease.get('format') != 'v25_distinct_arm_downstream_wave_v1':
        raise ValueError('Only exact released same-arm direct native lease can be recovered')
    # Reuse the exact original PID/birth/argv/UID and orphan-child release proof.
    view = dict(lease, format='v25_exclusive_downstream_wave_v1')
    return storage._recovery_predecessor(document, arm, view,
        predecessor_child=predecessor_child, predecessor_status=predecessor_status)


def _wave_binding(document, arm, allocation_request):
    """Bind a separate explicit direct arm lease to the original admission."""
    group = Path(document['group_root'])
    if allocation_request is None:
        raise ValueError('Exact explicit direct resource path required for parallel native wave')
    path = core.regular(allocation_request)
    binding = core.read(path)
    if binding.get('format') != 'v25_explicit_direct_GPU_execution_v1':
        raise ValueError('Exact explicit direct native resource mode required')
    from .v25_direct_resources import DirectAllocationLease
    lease = DirectAllocationLease(path)
    lease.verify(force=True)
    if (binding.get('arm') != arm or binding.get('group_root') != str(group)
            or binding.get('PBS_owned_reservation') is not False
            or binding.get('explicit_direct_execution_authorized') is not True
            or binding.get('aggregate_native_and_cache_RSS_bytes') != 416*2**30
            or binding.get('canonical_cache_reserve_bytes') != 32*2**30):
        raise ValueError('Exact distinct own-arm direct full four-native/cache admission required')
    return group/('parallel_native_arm_'+arm+'.lease.json'), dict(
        direct_resource_proof=core.proof(path), gpu=binding['gpu'], GPU_UUID=binding['GPU_UUID'],
        CPU_affinity=binding['CPU_affinity'], aggregate_native_and_cache_RSS_bytes=416*2**30,
        PBS_owned_reservation=False)


def native_wave_path(admission, arm, *, allocation_request=None):
    document = verify_admission(admission)
    if arm not in ARMS:
        raise ValueError('One explicit V25 arm required')
    if allocation_request is None or core.read(core.regular(allocation_request)).get('format') == 'v25_owned_GPU_allocation_v1':
        return Path(document['group_root'])/'sequential_native_wave.lease.json'
    return _wave_binding(document, arm, allocation_request)[0]


def _check_wave_peers(document, arm, value):
    group = Path(document['group_root'])
    exclusive = group/'sequential_native_wave.lease.json'
    if exclusive.exists() or exclusive.is_symlink():
        raise RuntimeError('Existing exclusive native wave preserved; parallel mode cannot overlap it')
    for peer_arm in ARMS:
        candidate = group/('parallel_native_arm_'+peer_arm+'.lease.json')
        if peer_arm == arm or not (candidate.exists() or candidate.is_symlink()):
            continue
        peer = core.read(core.regular(candidate))
        if (peer.get('format') != 'v25_distinct_arm_downstream_wave_v1'
                or peer.get('arm') != peer_arm or peer.get('arm_root') != document['arm_roots'][peer_arm]
                or peer.get('admission_sha256') != canonical_hash(document) or peer.get('UID') != value['UID']):
            raise ValueError('Exact distinct peer-arm native wave identity required')
        proof = core.guard(peer['direct_resource_proof'], full_hash=True)
        binding = core.read(proof)
        if (binding.get('format') != 'v25_explicit_direct_GPU_execution_v1'
                or binding.get('arm') != peer_arm or binding.get('group_root') != str(group)
                or binding.get('gpu') != peer.get('gpu') or binding.get('GPU_UUID') != peer.get('GPU_UUID')
                or binding.get('CPU_affinity') != peer.get('CPU_affinity')
                or peer['gpu'] == value['gpu'] or peer['GPU_UUID'] == value['GPU_UUID']
                or set(peer['CPU_affinity']) & set(value['CPU_affinity'])):
            raise ValueError('Parallel native arms need distinct GPUs and disjoint exact CPU sets')


def _wave_receipt_path(path, action):
    prefix = path.name.removesuffix('.lease.json')
    return path.with_name(prefix+'_'+action+'_'+str(time.time_ns())+'.json')


@contextmanager
def parallel_wave_lock(admission, arm, *, request_sha256=None,
                         recover_released_same_arm=False, predecessor_child=None,
                         predecessor_status=None, allocation_request=None):
    """Four exact direct arm leases; only global checkpoint publication is serialized."""
    document = verify_admission(admission)
    if arm not in ARMS:
        raise ValueError('One explicit V25 arm required')
    import psutil
    actor = psutil.Process()
    path, direct = _wave_binding(document, arm, allocation_request)
    value = dict(format='v25_distinct_arm_downstream_wave_v1', arm=arm, pid=actor.pid,
        create_time=actor.create_time(), command=actor.cmdline(), UID=os.getuid() if hasattr(os, 'getuid') else None,
        admission_sha256=canonical_hash(document), arm_root=document['arm_roots'][arm])
    value.update(direct)
    if request_sha256 is not None:
        if (not isinstance(request_sha256, str) or len(request_sha256) != 64
                or any(item not in '0123456789abcdef' for item in request_sha256)):
            raise ValueError('Exact immutable downstream request SHA256 required')
        value['request_sha256'] = request_sha256
    lock = document['checkpoint_coordination']['lock_file']['path']
    with _publication_lock(lock):
        _check_wave_peers(document, arm, value)
        if path.exists() or path.is_symlink():
            if recover_released_same_arm is not True:
                raise FileExistsError('Existing native wave lease preserved; explicit same-arm recovery required')
            old_proof = core.proof(path); old = core.read(path)
            if old.get('format') != value['format']:
                raise ValueError('Same-arm native recovery cannot relabel the resource mode')
            prior_resource = core.read(core.guard(old['direct_resource_proof'], full_hash=True))
            if (prior_resource.get('arm') != arm or prior_resource.get('group_root') != document['group_root']
                    or prior_resource.get('gpu') != value['gpu'] or prior_resource.get('GPU_UUID') != value['GPU_UUID']):
                raise ValueError('Direct same-arm recovery must retain its exact calibrated physical GPU')
            child_proof = _recovery_predecessor(document, arm, old,
                predecessor_child=predecessor_child, predecessor_status=predecessor_status)
            archive = _wave_receipt_path(path, 'released')
            core.publish_json(archive, dict(original_lease=old, original_file=old_proof,
                predecessor_release=child_proof, verified_released_at=time.time(),
                recovery_arm=arm, signals_sent=False))
            core.guard(old_proof, full_hash=True)
            path.unlink()
        core.publish_json(path, value)
    witness = core.proof(path)
    try:
        yield copy.deepcopy(witness)
    except BaseException:
        # An uncertain failed child must not permit the next native arm.
        raise
    else:
        with _publication_lock(lock):
            core.guard(witness, full_hash=True)
            receipt = _wave_receipt_path(path, 'finished')
            core.publish_json(receipt, dict(lease=value, completed_at=time.time()))
            path.unlink()


@contextmanager
def _exclusive_publication_lock(path):
    """Reject mode overlap atomically before the original owned lease publishes."""
    with storage._publication_lock(path):
        group = Path(path).parent
        if any((group/('parallel_native_arm_'+arm+'.lease.json')).exists()
                or (group/('parallel_native_arm_'+arm+'.lease.json')).is_symlink() for arm in ARMS):
            raise RuntimeError('Existing direct native arm wave preserved; exclusive mode cannot overlap it')
        yield


def native_wave_lock(admission, arm, *, allocation_request=None, **kwargs):
    """Select truthful execution mode while preserving the original owned code."""
    if allocation_request is not None:
        binding = core.read(core.regular(allocation_request))
        if binding.get('format') == 'v25_explicit_direct_GPU_execution_v1':
            return parallel_wave_lock(admission, arm, allocation_request=allocation_request, **kwargs)
        if binding.get('format') != 'v25_owned_GPU_allocation_v1':
            raise ValueError('Truthful owned or explicit direct native resource mode required')
    global _OWNED_WAVE
    if _OWNED_WAVE is None:
        # The original contextmanager body is unchanged; only its publication
        # lock checks that the other explicit execution mode is absent.
        original = storage.sequential_wave_lock.__wrapped__
        _OWNED_WAVE = contextmanager(core._clone(original,
            dict(_publication_lock=_exclusive_publication_lock)))
    return _OWNED_WAVE(admission, arm, **kwargs)
