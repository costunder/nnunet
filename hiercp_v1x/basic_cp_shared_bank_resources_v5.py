"""Bounded CPU-bank observations on fresh common ESRCH-safe task evidence.

The native request remains eightCPU/96GiB. Only the explicit CPU-only bank
stage uses64CPU/384GiB inside the unchanged800GiB group admission.
"""
from __future__ import annotations

import copy
import json
import time

from . import basic_cp_shared_bank_resources_v3 as prior
from . import basic_cp_group_resources_v4 as observation

resource = observation.resource
SHARED_GUARD_FILE = 'hiercp_v1x/basic_cp_shared_bank_resources_v5.py'
SHARED_CPU_AFFINITY = list(prior.SHARED_CPU_AFFINITY)
RSS_LIMIT = prior.RSS_LIMIT
AGGREGATE_CAPACITY = prior.AGGREGATE_CAPACITY
GIB = prior.GIB


class SharedBankLease(prior.SharedBankLease):
    """Actual64CPU descendants with bounded observation, exact release and RSS."""

    def __init__(self, request):
        if (type(request.get('group_observation_revision')) is not int
                or request['group_observation_revision'] != 5
                or not {SHARED_GUARD_FILE, observation.GUARD_FILE, prior._exit_fix.GUARD_FILE}
                <= set(request.get('code_sha256', {}))):
            raise ValueError('Explicit shared observation revision5 and exact new guard SHA required')
        super().__init__(request)
        self._shared_affinity_reobservations = {}

    _affinity_details = observation.BasicLease._affinity_details
    _retain_exiting_affinity = observation.BasicLease._retain_exiting_affinity

    def _tree_once(self):
        controller = prior._exact(self.request['controller_identity'])
        own = prior._exact(self.self_identity)
        for process, role in ((controller, 'controller'), (own, 'stage')):
            metadata = prior._metadata(process)
            actual = process.cpu_affinity()
            if actual != SHARED_CPU_AFFINITY:
                raise observation._AffinityMismatch(self._affinity_details(
                    5, process, metadata, actual, SHARED_CPU_AFFINITY, role))
        tree = {controller.pid: controller}
        terminal = []
        for process in controller.children(recursive=True):
            try:
                process = self._descendant(process)
                if process is None:
                    continue
                metadata = prior._metadata(process)
                actual = process.cpu_affinity()
                if prior._metadata(process) != metadata:
                    raise RuntimeError('Shared descendant changed exact PID/birth/UID during CPU observation')
                if actual != SHARED_CPU_AFFINITY:
                    details = self._affinity_details(5, process, metadata, actual,
                                                     SHARED_CPU_AFFINITY, 'descendant')
                    if not self._retain_exiting_affinity(process, metadata, details):
                        raise observation._AffinityMismatch(details)
                    terminal.append(process.pid)
                tree[process.pid] = process
            except prior.psutil.NoSuchProcess:
                continue
        if own.pid not in tree:
            raise RuntimeError('Shared bank stage is outside its exact authorized controller tree')
        return tree, terminal

    def _tree(self, controller, own):
        deadline = time.monotonic() + observation.AFFINITY_DEADLINE_SECONDS
        mismatches = []
        for attempt in range(observation.MAX_AFFINITY_REOBSERVATIONS + 1):
            if attempt and time.monotonic() >= deadline:
                raise self._affinity_failure(attempt - 1, mismatches)
            try:
                tree, terminal = self._tree_once()
                if attempt and time.monotonic() > deadline:
                    raise self._affinity_failure(attempt, mismatches)
                self._shared_affinity_reobservations = dict(reobservations=attempt,
                    recovered=bool(mismatches), mismatches=copy.deepcopy(mismatches),
                    exact_kernel_exiting_unadmitted_CPU_mask_PIDs=terminal,
                    maximum_seconds=observation.AFFINITY_DEADLINE_SECONDS)
                return tree
            except observation._AffinityMismatch as error:
                mismatches.append(error.details)
                remaining = deadline - time.monotonic()
                if attempt == observation.MAX_AFFINITY_REOBSERVATIONS or remaining <= 0:
                    raise self._affinity_failure(attempt, mismatches) from error
                time.sleep(min(observation.AFFINITY_REOBSERVATION_SECONDS, remaining))
        raise AssertionError('Bounded shared affinity observation must return or fail explicitly')

    def _affinity_failure(self, attempt, mismatches):
        self._shared_affinity_reobservations = dict(reobservations=attempt, recovered=False,
            mismatches=copy.deepcopy(mismatches), maximum_seconds=observation.AFFINITY_DEADLINE_SECONDS)
        return RuntimeError('Persistent exact shared CPU affinity mismatch after bounded observation: ' +
                            json.dumps(dict(attempts=attempt + 1, observations=mismatches), sort_keys=True))


BasicLease = SharedBankLease
__all__ = ['SharedBankLease', 'BasicLease', 'SHARED_GUARD_FILE', 'SHARED_CPU_AFFINITY',
           'RSS_LIMIT', 'AGGREGATE_CAPACITY', 'GIB']
