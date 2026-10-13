"""Measured384GiB CPU-bank admission; prior live guards remain immutable.

Four native96GiB budgets + candidate384GiB + shared32GiB require800GiB.
The same64 CPU cores, full source/candidate universe and24 workers are retained.
"""
from __future__ import annotations

import ast as _ast
import copy as _copy
import inspect as _inspect
import textwrap as _textwrap
import types as _types

from . import basic_cp_shared_bank_resources as _prior
from . import basic_cp_group_resources_v2 as _exit_fix

for _name,_value in vars(_prior).items():
    if not _name.startswith('__')and not isinstance(_value,_types.FunctionType)and _name not in('SharedBankLease','BasicLease','_InheritedLease'):
        globals()[_name]=_value


class _ResourceConstants(_ast.NodeTransformer):
    def visit_Constant(self,node):
        if type(node.value)is int and node.value in(288,704):node.value={288:384,704:800}[node.value]
        elif isinstance(node.value,str):node.value=node.value.replace('288GiB','384GiB').replace('704GiB','800GiB')
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        if (isinstance(node.func,_ast.Attribute)and isinstance(node.func.value,_ast.Call)
                and isinstance(node.func.value.func,_ast.Name)and node.func.value.func.id=='super'
                and not node.func.value.args):
            node.func.value=_ast.Name(id='_InheritedLease',ctx=_ast.Load())
            node.args.insert(0,_ast.Name(id='self',ctx=_ast.Load()))
        return node


def _rebind(value):
    if value.__globals__ is not vars(_prior):
        wrapped=getattr(value,'__wrapped__',None)
        if wrapped is not None and wrapped.__globals__ is vars(_prior):return _prior.contextmanager(_rebind(wrapped))
        return value
    source=_textwrap.dedent(_inspect.getsource(value));tree=_ast.parse(source)
    function=tree.body[0];function.decorator_list=[]
    tree=_ResourceConstants().visit(tree)
    # An aliased function can have a different source name: the original
    # ``_runtime_validate_request`` is itself named ``validate_request``.
    # Compile privately so that this alias cannot overwrite the stricter
    # execution-view validator already installed under that public name.
    scope=dict(globals())
    exec(compile(_ast.fix_missing_locations(tree),__file__,'exec'),scope)
    compiled=scope[function.name]
    rebound=_types.FunctionType(compiled.__code__,globals(),value.__name__,
                                compiled.__defaults__,compiled.__closure__)
    rebound.__kwdefaults__=_copy.deepcopy(compiled.__kwdefaults__)
    rebound.__doc__=value.__doc__
    rebound.__qualname__=value.__qualname__
    rebound.__annotations__=dict(compiled.__annotations__)
    return rebound


# Each clone resolves384/800 here; no imported live module is monkey-patched.
for _name,_value in vars(_prior).items():
    if isinstance(_value,_types.FunctionType)and _name!='_rebind_function':globals()[_name]=_rebind(_value)

RSS_LIMIT=384*GIB
AGGREGATE_CAPACITY=800*GIB
SHARED_GUARD_FILE='hiercp_v1x/basic_cp_shared_bank_resources_v3.py'
ORIGINAL_SHARED_GUARD_FILE='hiercp_v1x/basic_cp_shared_bank_resources.py'
resource=_seven

_old_shared_validation=validate_shared_request
_old_runtime_validation=validate_request


def validate_shared_request(value):
    _old_shared_validation(value)
    if (type(value.get('shared_resource_guard_version'))is not int or value['shared_resource_guard_version']!=3
            or ORIGINAL_SHARED_GUARD_FILE not in value['code_sha256']):
        raise ValueError('Explicit v3 CPU-bank admission and immutable original shared guard SHA required')
    return value


def validate_request(value):
    _old_runtime_validation(value)
    if (type(value.get('shared_resource_guard_version'))is not int or value['shared_resource_guard_version']!=3
            or ORIGINAL_SHARED_GUARD_FILE not in value['code_sha256']):
        raise ValueError('CPU-bank execution view requires the explicit v3 admission')
    return value


_body=dict(vars(_prior._InheritedLease));_body.pop('__dict__',None);_body.pop('__weakref__',None)
_body['__module__']=__name__
for _name,_value in tuple(_body.items()):
    if isinstance(_value,_types.FunctionType):_body[_name]=_rebind(_value)
_InheritedLease=type('_InheritedSharedBankLeaseV3',_prior._InheritedLease.__bases__,_body)

_body=dict(vars(_prior.SharedBankLease));_body.pop('__dict__',None);_body.pop('__weakref__',None)
_body['__module__']=__name__
for _name,_value in tuple(_body.items()):
    if isinstance(_value,_types.FunctionType):_body[_name]=_rebind(_value)
    elif isinstance(_value,staticmethod):_body[_name]=staticmethod(_rebind(_value.__func__))
_body['_descendant']=_types.FunctionType(_exit_fix.BasicLease._descendant.__code__,globals(),
    '_descendant',_exit_fix.BasicLease._descendant.__defaults__,_exit_fix.BasicLease._descendant.__closure__)
_body['_final_exit_snapshot']=_types.FunctionType(_exit_fix.BasicLease._final_exit_snapshot.__code__,globals(),
    '_final_exit_snapshot',_exit_fix.BasicLease._final_exit_snapshot.__defaults__,_exit_fix.BasicLease._final_exit_snapshot.__closure__)
SharedBankLease=type('SharedBankLease',(_InheritedLease,),_body)
BasicLease=SharedBankLease
__all__=['SharedBankLease','BasicLease','validate_shared_request','SHARED_CPU_AFFINITY',
         'RSS_LIMIT','AGGREGATE_CAPACITY','SHARED_GUARD_FILE','GIB']
