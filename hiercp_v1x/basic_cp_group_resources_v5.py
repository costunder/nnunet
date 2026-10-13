"""Final future-export group observation guard; scientific/resources stay V3."""
from __future__ import annotations

from . import basic_cp_group_resources_v3 as resource_v3
from . import basic_cp_group_resources_v4 as prior

GUARD_FILE = 'hiercp_v1x/basic_cp_group_resources_v5.py'
AGGREGATE_CAPACITY = prior.AGGREGATE_CAPACITY
CANDIDATE_RSS_LIMIT = prior.CANDIDATE_RSS_LIMIT
GROUP_RSS_LIMIT = prior.GROUP_RSS_LIMIT
NATIVE_RSS_LIMIT = prior.NATIVE_RSS_LIMIT


class BasicLease(prior.BasicLease):
    """V4 bounded observations with explicit revision5 and fresh common code."""

    def __init__(self, request):
        if (type(request.get('group_observation_revision')) is not int
                or request['group_observation_revision'] != 5
                or not {GUARD_FILE, prior.GUARD_FILE} <= set(request.get('code_sha256', {}))):
            raise ValueError('Explicit group observation revision5 and exact new guard SHA required')
        # V4's constructor intentionally requires revision4. Its methods are
        # retained, while V3 still performs every immutable registry/admission.
        resource_v3.BasicLease.__init__(self, request)
        self._affinity_reobservations = {}


SharedGroupLease = BasicLease
FORMAT = prior.FORMAT
__all__ = ['BasicLease', 'SharedGroupLease', 'FORMAT', 'GUARD_FILE',
           'AGGREGATE_CAPACITY', 'CANDIDATE_RSS_LIMIT', 'GROUP_RSS_LIMIT', 'NATIVE_RSS_LIMIT']
