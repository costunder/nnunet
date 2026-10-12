"""Observe exited sibling stages without weakening live direct admission.

The frozen direct guard rejects a zombie while enumerating all controller
descendants. A normally completed sibling can remain a zombie until its
controller reaps it; it owns no runnable workload. This execution sidecar
excludes only an actually observed own terminal descendant from the live tree.
It leaves the controller, current actor, resource and CUDA checks unchanged.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

from . import v25_direct_resources as direct

FORMAT = 'v25_terminal_descendant_resource_runtime_v1'
_SOURCE = Path(__file__).resolve()
_ORIGINAL_TREE = direct.DirectAllocationLease._tree


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_SOURCE_SHA256 = _sha(_SOURCE)


def _metadata(process):
    value = dict(pid=process.pid, create_time=process.create_time(),
        UID=process.uids().real)
    if (type(value['pid']) is not int or value['pid'] <= 0
            or type(value['create_time']) not in (int, float)
            or not math.isfinite(value['create_time']) or value['create_time'] <= 0):
        raise RuntimeError('Exact terminal descendant PID and birth time required')
    if value['UID'] != 41833:
        raise RuntimeError('Foreign-UID controller descendant cannot be an admitted direct actor')
    return value


def _terminal(lease, process, *, expected=None):
    """Verify terminal UID/birth; a missing process is handled by the caller."""
    if process.status() != direct.psutil.STATUS_ZOMBIE:
        raise RuntimeError('Descendant ZombieProcess must have actual zombie status')
    observed = _metadata(process)
    if expected is not None and observed != expected:
        raise RuntimeError('Observed descendant PID/birth/UID changed during terminal transition')
    if process.status() != direct.psutil.STATUS_ZOMBIE:
        raise RuntimeError('Terminal descendant status changed during identity observation')
    previous = lease.managed.get(process.pid)
    if previous is not None and any(previous[key] != observed[key]
            for key in ('pid', 'create_time', 'UID')):
        raise RuntimeError('Managed direct child PID/birth/UID identity changed')
    terminals = getattr(lease, '_terminal_descendants', None)
    if terminals is None:
        terminals = lease._terminal_descendants = {}
    prior = terminals.get(process.pid)
    if prior is not None and prior != observed:
        raise RuntimeError('Terminal direct child PID/birth/UID identity changed')
    if prior is None:
        terminals[process.pid] = observed
        print('V2.5 DIRECT TERMINAL DESCENDANT pid=%d create_time=%.6f UID=%d '
              'status=zombie active_tree_member=False no_process_signaled=True' %
              (observed['pid'], observed['create_time'], observed['UID']), flush=True)
    return None


def _descendant(lease, process):
    """Take a live identity, checking both sides of the terminal transition."""
    observed = None
    # Linux clears an exiting worker's cmdline before its zombie status becomes
    # observable. Retry only that incomplete transition; empty argv alone never
    # grants live-tree or CUDA admission. Every reread must retain the exact
    # same owned PID/birth/UID, including identities already in managed.
    for attempt in range(3):
        if process.status() == direct.psutil.STATUS_ZOMBIE:
            return _terminal(lease, process, expected=observed)
        try:
            metadata = _metadata(process)
            if observed is not None and metadata != observed:
                raise RuntimeError('Observed descendant PID/birth/UID changed during terminal transition')
            previous = lease.managed.get(process.pid)
            if previous is not None and any(previous[key] != metadata[key]
                    for key in ('pid', 'create_time', 'UID')):
                raise RuntimeError('Managed direct child PID/birth/argv identity changed')
            observed = metadata
            command = process.cmdline()
        except direct.psutil.ZombieProcess:
            return _terminal(lease, process, expected=observed)
        if process.status() == direct.psutil.STATUS_ZOMBIE:
            return _terminal(lease, process, expected=observed)
        if command == []:
            if attempt < 2:
                # Give the exiting kernel task a scheduling opportunity. The
                # two bounded yields total at most20ms; ordinary live actors
                # never wait and persistent empty argv still fails below.
                time.sleep(.01)
            continue
        identity = dict(**metadata, command=command)
        if not direct._identity_document(identity):
            raise RuntimeError('Exact live descendant PID/birth/argv/UID identity required')
        if previous is not None and previous != identity:
            raise RuntimeError('Managed direct child PID/birth/argv identity changed')
        if getattr(lease, '_terminal_descendants', {}).get(process.pid) is not None:
            raise RuntimeError('Previously terminal direct child PID became an active actor')
        return identity
    raise RuntimeError('Exact live descendant PID/birth/argv/UID identity required')


def _tree(lease):
    if _sha(_SOURCE) != _SOURCE_SHA256:
        raise ValueError('Immutable terminal-descendant execution source changed')
    # The actual controller and current stage retain the original strict check.
    controller = direct._exact(lease.request['controller_identity'])
    own = direct._exact(lease.self_identity)
    tree = {controller.pid: controller}
    for child in controller.children(recursive=True):
        try:
            identity = _descendant(lease, child)
        except direct.psutil.ZombieProcess as error:
            raise RuntimeError('Terminal descendant UID/birth could not be verified') from error
        except direct.psutil.NoSuchProcess:
            continue
        if identity is None:
            continue
        previous = lease.managed.get(child.pid)
        if previous is not None and previous != identity:
            raise RuntimeError('Managed direct child PID/birth/argv identity changed')
        lease.managed[child.pid] = identity
        tree[child.pid] = child
    if own.pid not in tree:
        raise RuntimeError('Direct stage must be the exact declared controller or its actual descendant')
    if controller.cpu_affinity() != lease.request['aggregate_CPU_affinity']:
        raise RuntimeError('Controller aggregate24 CPU affinity changed')
    if own.cpu_affinity() != lease.request['CPU_affinity'] and own.pid != controller.pid:
        raise RuntimeError('Own direct stage six-CPU affinity differs from its arm proof')
    return controller, own, tree


def install_runtime():
    """Install before constructing a lease; return a separately sealed receipt."""
    if _sha(_SOURCE) != _SOURCE_SHA256:
        raise ValueError('Immutable terminal-descendant execution source changed')
    if direct.DirectAllocationLease._tree not in (_ORIGINAL_TREE, _tree):
        raise ValueError('Unknown existing direct-resource tree execution override')
    direct.DirectAllocationLease._tree = _tree
    return dict(format=FORMAT, runtime_source_sha256=_SOURCE_SHA256,
        original_resource_guard_sha256=_sha(Path(direct.__file__)),
        original_resource_guard_file_bytes_changed=False,
        current_actor_and_controller_exact_identity_preserved=True,
        terminal_descendants_require_actual_zombie_UID41833_and_same_birth=True,
        terminal_descendants_admitted_as_CUDA_actors=False,
        CPU_RAM_GPU_cgroup_and_host_lifetime_guards_unchanged=True,
        scientific_files_changed=False, no_process_signaled=True)
