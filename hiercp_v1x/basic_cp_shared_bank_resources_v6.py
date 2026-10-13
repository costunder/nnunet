"""CPU-bank admission adds back all seven exactly pinned resident trees.

The scientific/resource contract remains observation5, marker3, native96GiB,
candidate384GiB and aggregate800GiB. Only the shared RAM formula is corrected;
old exports remain immutable and the producer still cannot initialize CUDA.
"""
from __future__ import annotations

from . import basic_cp_shared_bank_resources_v5 as prior
from . import basic_cp_group_resources_v5 as group

SHARED_GUARD_FILE = 'hiercp_v1x/basic_cp_shared_bank_resources_v6.py'
SHARED_CPU_AFFINITY = list(prior.SHARED_CPU_AFFINITY)
RSS_LIMIT = prior.RSS_LIMIT
AGGREGATE_CAPACITY = prior.AGGREGATE_CAPACITY
GIB = prior.GIB


class _CPUStageGroupLease(group.BasicLease):
    # Only the producer's own CPU tree borrows the candidate384 budget.
    # Group accounting still enforces96 for every native tree,384 across
    # candidate3/4/5 and768 across all deduplicated registered processes.
    _rss = prior.SharedBankLease._rss


class SharedBankLease(prior.SharedBankLease):
    """Shared CPU guard composed with the complete immutable seven-row ledger."""

    def __init__(self, request):
        if (type(request.get('shared_resource_observation_revision')) is not int
                or request['shared_resource_observation_revision'] != 6
                or not {SHARED_GUARD_FILE, prior.SHARED_GUARD_FILE, group.GUARD_FILE}
                <= set(request.get('code_sha256', {}))):
            raise ValueError('Explicit shared resource observation6 and exact Shared6/Shared5/Group5 SHAs required')
        super().__init__(request)
        # Construct using the original native8 request. Group.verify() would
        # require native8 execution and is intentionally not called in the
        # CPU64 stage. Its immutable inputs and resource accounting are used.
        self._group_view = _CPUStageGroupLease(self.native_request)
        for admitted in self._group_view.registered_requests.values():
            if (type(admitted.get('shared_resource_observation_revision')) is not int
                    or admitted['shared_resource_observation_revision'] != 6
                    or type(admitted.get('group_observation_revision')) is not int
                    or admitted.get('group_observation_revision') != 5
                    or type(admitted.get('shared_resource_guard_version')) is not int
                    or admitted.get('shared_resource_guard_version') != 3
                    or admitted.get('shared_resource_registry') != str(self._group_view.registry_path)
                    or not {SHARED_GUARD_FILE, group.GUARD_FILE} <= set(admitted.get('code_sha256', {}))):
                raise ValueError('All seven current peers must bind the same fresh ledger and explicit observation6/5/marker3 SHAs')

    def _inputs(self):
        super()._inputs()
        if hasattr(self, '_group_view'):
            self._group_view._inputs()

    def _resources(self, controller, own, tree):
        self._group_view._inputs()
        result = self._group_view._resources(controller, own, tree)
        result.update(own_process_tree_RSS_limit_bytes=RSS_LIMIT,
            shared_bank_RAM_budget_bytes=RSS_LIMIT,
            other_four_native_RSS_budget_bytes=4 * group.NATIVE_RSS_LIMIT,
            native_CPU_affinity_preserved=list(self.native_request['cpu_affinity']),
            native_RSS_limit_bytes_preserved=self.native_request['RSS_limit_bytes'],
            own_CPU_affinity=list(SHARED_CPU_AFFINITY), execution_stage='shared_candidate_bank_CPU_only',
            original_native_request_modified=False,
            shared_resource_observation_revision=6,
            RAM_admission_formula='actual_available_RAM + deduplicated_all_registered_group_tree_RSS >= 800GiB')
        # The original CPU inventory admission remains independent of PBS
        # reservation claims and is checked from the actual host inventory.
        logical_cpus = prior.prior.psutil.cpu_count(logical=True)
        if logical_cpus is None or logical_cpus <= max(SHARED_CPU_AFFINITY):
            raise RuntimeError('Actual host CPU inventory cannot admit the measured64CPU shared affinity')
        result['actual_host_logical_CPU_count'] = logical_cpus
        return result


BasicLease = SharedBankLease
__all__ = ['SharedBankLease', 'BasicLease', 'SHARED_GUARD_FILE', 'SHARED_CPU_AFFINITY',
           'RSS_LIMIT', 'AGGREGATE_CAPACITY', 'GIB']
