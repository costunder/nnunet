"""Bounded exact CPU-affinity observations with the unchanged V3 RAM budget.

Live guards remain immutable.  Every retry re-reads the same registered
controller and its descendants; persistent unadmitted masks still fail.
An exact kernel-exiting task stays in RSS accounting with an explicit terminal
observation, rather than pretending that its CPU mask or RSS is admitted/zero.
"""
from __future__ import annotations

import copy
import json
import time

from . import basic_cp_group_resources as original
from . import basic_cp_group_resources_v3 as prior

resource = original.resource
GUARD_FILE = 'hiercp_v1x/basic_cp_group_resources_v4.py'
MAX_AFFINITY_REOBSERVATIONS = 3
AFFINITY_REOBSERVATION_SECONDS = .05
AFFINITY_DEADLINE_SECONDS = .3
AGGREGATE_CAPACITY = prior.AGGREGATE_CAPACITY
CANDIDATE_RSS_LIMIT = prior.CANDIDATE_RSS_LIMIT
GROUP_RSS_LIMIT = prior.GROUP_RSS_LIMIT
NATIVE_RSS_LIMIT = prior.NATIVE_RSS_LIMIT
SHARED_CPU_AFFINITY = list(original.SHARED_CPU_AFFINITY)


class _AffinityMismatch(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__('Exact group CPU affinity observation mismatch: ' +
                         json.dumps(details, sort_keys=True))


class BasicLease(prior.BasicLease):
    """V3 resources and identities plus <=300ms coherent affinity re-reading."""

    def __init__(self, request):
        if (type(request.get('group_observation_revision')) is not int
                or request['group_observation_revision'] != 4
                or GUARD_FILE not in request.get('code_sha256', {})):
            raise ValueError('Explicit group observation revision4 and exact new guard SHA required')
        super().__init__(request)
        self._affinity_reobservations = {}

    def _affinity_details(self, gpu, process, metadata, actual, expected, role):
        command = process.cmdline()
        exiting = resource._kernel_exiting(process)
        final = resource._metadata(process)
        if final != metadata:
            raise RuntimeError('CPU affinity observation changed exact PID/birth/UID: ' +
                               json.dumps(dict(gpu=gpu, before=metadata, after=final), sort_keys=True))
        return dict(gpu=gpu, role=role, pid=metadata['pid'],
                    birth=metadata['create_time'], UID=metadata['UID'], argv=command,
                    actual_CPU_affinity=list(actual), expected_CPU_affinity=list(expected),
                    kernel_exiting=exiting, status=process.status())

    def _retain_exiting_affinity(self, process, metadata, details):
        # A true exit flag is proof that this exact task is ending, never
        # permission for an active task to use CPUs outside its admitted mask.
        if not details['kernel_exiting']:
            return False
        if resource._metadata(process) != metadata:
            raise RuntimeError('Kernel-exiting affinity task changed exact PID/birth/UID')
        self._terminal_observations[process.pid] = dict(metadata, status='PF_EXITING',
            reason='exact terminal task had an unadmitted CPU mask during resource observation',
            observed_at=time.time(), RSS_still_accounted=True, argv_invented=False,
            CPU_mask_admitted=False, actual_CPU_affinity=details['actual_CPU_affinity'],
            expected_CPU_affinity=details['expected_CPU_affinity'])
        return True

    def _group_tree_once(self, gpu):
        controller, observation, affinity, candidates = self._controller_stage_snapshot(gpu)
        admitted = self.registered_requests[gpu]
        expected = list(resource.CPUS[gpu])
        shared = False
        if controller is not None and gpu == 5 and affinity == SHARED_CPU_AFFINITY:
            if (admitted.get('shared_bank_CPU_affinity') != SHARED_CPU_AFFINITY
                    or type(admitted.get('shared_bank_RSS_limit_GiB')) is not int
                    or admitted['shared_bank_RSS_limit_GiB'] != 384):
                raise RuntimeError('GPU5 shared producer lacks explicit64CPU/384GiB admission')
            expected = list(SHARED_CPU_AFFINITY)
            shared = True
        if controller is not None:
            if affinity != expected:
                metadata = resource._metadata(controller)
                details = self._affinity_details(gpu, controller, metadata, affinity, expected, 'controller')
                # A controller must be coherently re-observed as terminal by
                # _controller_stage_snapshot; its mismatched mask is not accepted.
                raise _AffinityMismatch(details)
            tree = {controller.pid: controller}
        else:
            tree = {}
            candidates = list(self._group_previous_trees.get(gpu, {}).values())
            if gpu == 5 and self._group_previous_shared_stage.get(gpu, False):
                expected = list(SHARED_CPU_AFFINITY)
                shared = True
        terminal_affinity = []
        for process in candidates:
            try:
                process = self._descendant(process)
                if process is None:
                    continue
                metadata = resource._metadata(process)
                actual = process.cpu_affinity()
                if resource._metadata(process) != metadata:
                    raise RuntimeError('Group descendant changed exact PID/birth/UID during CPU observation')
                if actual != expected:
                    details = self._affinity_details(gpu, process, metadata, actual, expected, 'descendant')
                    if not self._retain_exiting_affinity(process, metadata, details):
                        raise _AffinityMismatch(details)
                    terminal_affinity.append(process.pid)
                tree[process.pid] = process
            except resource.psutil.NoSuchProcess:
                continue
        self._group_previous_trees[gpu] = dict(tree)
        self._group_previous_shared_stage[gpu] = shared
        observation.update(actual_CPU_affinity=expected, shared_CPU_bank_stage=shared,
                           surviving_exact_descendant_PIDs=sorted(tree),
                           exact_kernel_exiting_unadmitted_CPU_mask_PIDs=terminal_affinity)
        return tree, observation

    def _group_tree(self, gpu):
        deadline = time.monotonic() + AFFINITY_DEADLINE_SECONDS
        mismatches = []
        for attempt in range(MAX_AFFINITY_REOBSERVATIONS + 1):
            if attempt and time.monotonic() >= deadline:
                raise self._persistent_affinity_failure(gpu, attempt - 1, mismatches)
            try:
                tree, observation = self._group_tree_once(gpu)
                if attempt and time.monotonic() > deadline:
                    raise self._persistent_affinity_failure(gpu, attempt, mismatches)
                record = dict(reobservations=attempt, recovered=bool(mismatches),
                              mismatches=copy.deepcopy(mismatches), maximum_seconds=AFFINITY_DEADLINE_SECONDS)
                self._affinity_reobservations[gpu] = record
                observation['bounded_CPU_affinity_reobservation'] = record
                return tree, observation
            except _AffinityMismatch as error:
                mismatches.append(error.details)
                remaining = deadline - time.monotonic()
                if attempt == MAX_AFFINITY_REOBSERVATIONS or remaining <= 0:
                    raise self._persistent_affinity_failure(gpu, attempt, mismatches) from error
                time.sleep(min(AFFINITY_REOBSERVATION_SECONDS, remaining))
        raise AssertionError('Bounded affinity observation must return or fail explicitly')

    def _persistent_affinity_failure(self, gpu, attempt, mismatches):
        self._affinity_reobservations[gpu] = dict(reobservations=attempt, recovered=False,
            mismatches=copy.deepcopy(mismatches), maximum_seconds=AFFINITY_DEADLINE_SECONDS)
        return RuntimeError('Persistent exact group CPU affinity mismatch after bounded observation: ' +
                            json.dumps(dict(attempts=attempt + 1, observations=mismatches), sort_keys=True))


SharedGroupLease = BasicLease
FORMAT = original.FORMAT
__all__ = ['BasicLease', 'SharedGroupLease', 'FORMAT', 'GUARD_FILE',
           'AGGREGATE_CAPACITY', 'CANDIDATE_RSS_LIMIT', 'GROUP_RSS_LIMIT', 'NATIVE_RSS_LIMIT']
