"""Independent800GiB group guard with384GiB shared candidate pool.

Native arms retain96GiB/eightCPU.  v2 exact worker-exit and immutable previous
ledger checks remain intact.  No original live guard bytes are modified.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

from . import basic_cp_group_resources as original
from . import basic_cp_group_resources_v2 as prior

GUARD_FILE='hiercp_v1x/basic_cp_group_resources_v3.py'
AGGREGATE_CAPACITY=800*original.GIB
CANDIDATE_RSS_LIMIT=384*original.GIB
GROUP_RSS_LIMIT=768*original.GIB
NATIVE_RSS_LIMIT=original.NATIVE_RSS_LIMIT


def _scope_clone(function):
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant):
            if type(node.value)is int and node.value==288:node.value=384
            elif isinstance(node.value,str):node.value=node.value.replace('288GiB','384GiB').replace('672GiB','768GiB').replace('704GiB','800GiB')
    scope=dict(original.__dict__,AGGREGATE_CAPACITY=AGGREGATE_CAPACITY,
        CANDIDATE_RSS_LIMIT=CANDIDATE_RSS_LIMIT,GROUP_RSS_LIMIT=GROUP_RSS_LIMIT)
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),scope)
    return scope[function.__name__]


class BasicLease(prior.BasicLease):
    """Explicit v3 group admission without changing native science or limits."""
    def __init__(self,request):
        if (type(request.get('shared_resource_guard_version'))is not int or request['shared_resource_guard_version']!=3
                or GUARD_FILE not in request.get('code_sha256',{})):
            raise ValueError('Explicit v3 group admission and exact new group guard SHA required')
        super().__init__(request)
    _resources=_scope_clone(original.BasicLease._resources)
    _group_tree=_scope_clone(original.BasicLease._group_tree)


SharedGroupLease=BasicLease
FORMAT=original.FORMAT
__all__=['BasicLease','SharedGroupLease','FORMAT','GUARD_FILE','AGGREGATE_CAPACITY',
         'CANDIDATE_RSS_LIMIT','GROUP_RSS_LIMIT','NATIVE_RSS_LIMIT']
