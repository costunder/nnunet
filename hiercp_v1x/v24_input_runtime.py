"""Exact input execution: pair-local attribute reuse and final pinned collation.

No neural features or epoch views are cached here. Attribute reuse lasts only
for the two views of one record and returns independent storage on every hit.
The original sampler, collator, offsets, input validation and RNG run intact.
"""
from __future__ import annotations

import builtins
import copy
import hashlib
import inspect
import json
from pathlib import Path
import threading
import time
from types import FunctionType, SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Batch as _Batch
from torch_geometric.data.collate import collate as _PYG_COLLATE,_collate as _PYG_LEAF

from . import v24_inputs as inputs
from . import v24_memory_runtime as memory
from . import transition_v1_local as local

FORMAT='v24_exact_pair_local_attributes_and_pinned_collate_runtime_v1'
_PAIR=inputs.materialize_pair
_COLLATE=inputs.collate
_LOCAL_COLLATE=local.collate
_PYG_FROM_LIST=_Batch.from_data_list.__func__
_ORIGINAL_CODES={function:function.__code__ for function in
    (_PAIR,_COLLATE,_LOCAL_COLLATE,_PYG_COLLATE,_PYG_LEAF,_PYG_FROM_LIST)}
_ORIGINAL_SOURCE_STATS={str(Path(inspect.getsourcefile(function)).resolve()):
    memory._ORIGINAL_PROVIDER._stat(inspect.getsourcefile(function)) for function in _ORIGINAL_CODES}
_LOCK=threading.RLock()
_SAMPLER=None
_PIN_COLLATE=None
_PIN_OUTPUTS=True
_STATS=dict(materialize_calls=0,attribute_calls=0,attribute_memo_hits=0,
    attribute_memo_misses=0,attribute_reused_output_bytes=0,
    attribute_key_hash_seconds=0.,attribute_original_compute_seconds=0.,
    attribute_private_copy_seconds=0.,materialize_wall_seconds=0.,
    collate_calls=0,collate_wall_seconds=0.,direct_pinned_cat_calls=0,
    direct_pinned_stack_calls=0,direct_pinned_output_bytes=0)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _guard_originals():
    from torch_geometric.data import Batch
    from torch_geometric.data.collate import collate as pyg_collate,_collate as pyg_leaf
    if (inputs.materialize_pair is not _PAIR or inputs.collate is not _COLLATE
            or local.collate is not _LOCAL_COLLATE or Batch is not _Batch
            or pyg_collate is not _PYG_COLLATE or pyg_leaf is not _PYG_LEAF
            or Batch.from_data_list.__func__ is not _PYG_FROM_LIST
            or _PYG_COLLATE.__globals__.get('_collate') is not _PYG_LEAF
            or _PYG_FROM_LIST.__globals__.get('collate') is not _PYG_COLLATE
            or _PYG_LEAF.__globals__.get('_collate') is not _PYG_LEAF
            or any(function.__code__ is not code for function,code in _ORIGINAL_CODES.items())):
        raise ValueError('Foreign original input or PyG implementation refused')
    if any(memory._ORIGINAL_PROVIDER._stat(path)!=proof
            for path,proof in _ORIGINAL_SOURCE_STATS.items()):
        raise ValueError('Original input or PyG source changed')


def _clone(original,namespace,closure=None):
    result=FunctionType(original.__code__,namespace,original.__name__,
        original.__defaults__,original.__closure__ if closure is None else closure)
    result.__kwdefaults__=copy.deepcopy(original.__kwdefaults__)
    result.__dict__.update(original.__dict__)
    return result


def _cell(value):
    return (lambda:value).__closure__[0]


def _sampling_functions(original):
    """The admitted absence/scope wrappers call private original leaf views."""
    functions=[];seen=set()
    def visit(function):
        if id(function) in seen:return
        seen.add(id(function));functions.append(function)
        for name,cell in zip(function.__code__.co_freevars,function.__closure__ or ()):
            if name in ('bounded_view','allowed_view','patched_view'):
                if not isinstance(cell.cell_contents,FunctionType):
                    raise ValueError('Original sampling wrapper branch changed')
                visit(cell.cell_contents)
    visit(original)
    return tuple(functions)


def _sampling_proof(original,attributes):
    functions=_sampling_functions(original)
    leaves=[value for value in functions if '_edge_attributes' in value.__code__.co_names]
    if not leaves or any(value.__globals__.get('_edge_attributes') is not attributes for value in leaves):
        raise ValueError('Original sampling attribute branch changed')
    return tuple((function,function.__code__,tuple(
        (name,cell.cell_contents) for name,cell in zip(function.__code__.co_freevars,
            function.__closure__ or ()) if name in ('bounded_view','allowed_view','patched_view')),
        function.__globals__.get('_edge_attributes') if function in leaves else None)
        for function in functions)


def _private_sampling_chain(original,attributes):
    replacements={}
    def clone(function):
        if id(function) in replacements:return replacements[id(function)]
        namespace=dict(function.__globals__)
        if '_edge_attributes' in function.__code__.co_names:namespace['_edge_attributes']=attributes
        closure=tuple(_cell(clone(cell.cell_contents))
            if name in ('bounded_view','allowed_view','patched_view') else cell
            for name,cell in zip(function.__code__.co_freevars,function.__closure__ or ()))
        private=_clone(function,namespace,closure if function.__closure__ is not None else None)
        replacements[id(function)]=private
        return private
    return clone(original)


def _bump(**counts):
    with _LOCK:
        for key,value in counts.items():_STATS[key]+=value


def _private_import(original,replacements):
    namespace=dict(original.__globals__)
    original_builtins=namespace.get('__builtins__',builtins)
    private=dict(original_builtins if isinstance(original_builtins,dict) else vars(original_builtins))
    importer=private['__import__']
    def importing(name,globals=None,locals=None,fromlist=(),level=0):
        key=(name,tuple(fromlist),level)
        if key in replacements:return replacements[key]
        return importer(name,globals,locals,fromlist,level)
    private['__import__']=importing;namespace['__builtins__']=private
    return namespace


def _sampler():
    global _SAMPLER
    _guard_originals()
    with _LOCK:
        admitted=local._runtime()
        module=admitted['sample']
        if _SAMPLER is None:
            _SAMPLER=dict(module=module,build=module.build_local_view,
                attributes=module._edge_attributes,
                source=Path(module.__file__).resolve(),stat=memory._ORIGINAL_PROVIDER._stat(module.__file__),
                scope_sha256=admitted['scope']['contract_sha256'],
                chain=_sampling_proof(module.build_local_view,module._edge_attributes))
        if (_SAMPLER['module'] is not module or _SAMPLER['build'] is not module.build_local_view
                or _SAMPLER['attributes'] is not module._edge_attributes
                or _SAMPLER['scope_sha256']!=admitted['scope']['contract_sha256']
                or memory._ORIGINAL_PROVIDER._stat(_SAMPLER['source'])!=_SAMPLER['stat']
                or _sampling_proof(module.build_local_view,module._edge_attributes)!=_SAMPLER['chain']):
            raise ValueError('Admitted original sampling implementation changed')
        return dict(_SAMPLER)


def _attribute_key(bound):
    value=hashlib.sha256()
    def put(payload):
        value.update(len(payload).to_bytes(8,'little'));value.update(payload)
    for name,item in bound.arguments.items():
        put(name.encode())
        if isinstance(item,np.ndarray):
            array=np.ascontiguousarray(item)
            put(json.dumps([str(array.dtype),list(array.shape)],separators=(',',':')).encode())
            put(memoryview(array).cast('B') if array.size else b'')
        elif item is None:put(b'None')
        elif type(item)is int:put(str(item).encode())
        else:
            # The original supports ndarray numerical inputs here. Unknown
            # invalid arguments are still passed to its exact validator.
            return None
    return value.digest()


def _attribute_reuse(original):
    table={};signature=inspect.signature(original)
    def attributes(*args,**kwargs):
        started=time.perf_counter();bound=signature.bind(*args,**kwargs);bound.apply_defaults()
        key=_attribute_key(bound);hashed=time.perf_counter();_bump(attribute_calls=1,
            attribute_key_hash_seconds=hashed-started)
        if key is not None and key in table:
            result=table[key].copy(order='C');finished=time.perf_counter()
            _bump(attribute_memo_hits=1,attribute_reused_output_bytes=result.nbytes,
                attribute_private_copy_seconds=finished-hashed)
            return result
        result=original(*args,**kwargs);finished=time.perf_counter()
        _bump(attribute_memo_misses=1,attribute_original_compute_seconds=finished-hashed)
        if key is not None:table[key]=result
        return result
    return attributes


def materialize_pair(record,*,epoch):
    admitted=_sampler()
    build=_private_sampling_chain(admitted['build'],_attribute_reuse(admitted['attributes']))
    pair_namespace=_private_import(_PAIR,{('hiercp.sample',('build_local_view',),0):
        SimpleNamespace(build_local_view=build)})
    pair=_clone(_PAIR,pair_namespace)
    started=time.perf_counter()
    try:return pair(record,epoch=epoch)
    finally:_bump(materialize_calls=1,materialize_wall_seconds=time.perf_counter()-started)


def _allocate(shape,dtype):
    return torch.empty(shape,dtype=dtype,device='cpu',pin_memory=True)


class _PinnedTorch:
    """Private original torch namespace; only final plain CPU cat/stack differ."""
    def __getattr__(self,name):return getattr(torch,name)

    @staticmethod
    def _ordinary(values,out):
        return (out is None and isinstance(values,(list,tuple)) and bool(values)
            and all(type(value)is torch.Tensor and value.device.type=='cpu'
                and value.layout==torch.strided and not value.requires_grad
                and value.is_contiguous()
                and not (value.ndim==4 and value.is_contiguous(memory_format=torch.channels_last))
                and not (value.ndim==5 and value.is_contiguous(memory_format=torch.channels_last_3d))
                for value in values)
            and all(value.dtype==values[0].dtype for value in values))

    def cat(self,values,dim=0,*,out=None):
        if not self._ordinary(values,out):return torch.cat(values,dim=dim,out=out)
        # Let the original operation produce its original exception on malformed
        # ranks/shapes/dimensions instead of inventing a permissive allocation.
        rank=values[0].ndim
        if type(dim)is not int or not -rank<=dim<rank:return torch.cat(values,dim=dim,out=out)
        axis=dim%rank;shape=list(values[0].shape)
        if any(value.ndim!=rank or any(value.shape[i]!=shape[i] for i in range(rank) if i!=axis) for value in values):
            return torch.cat(values,dim=dim,out=out)
        shape[axis]=sum(value.shape[axis] for value in values)
        result=_allocate(shape,values[0].dtype)
        torch.cat(values,dim=dim,out=result)
        _bump(direct_pinned_cat_calls=1,direct_pinned_output_bytes=result.numel()*result.element_size())
        return result

    def stack(self,values,dim=0,*,out=None):
        if not self._ordinary(values,out):return torch.stack(values,dim=dim,out=out)
        rank=values[0].ndim
        if (type(dim)is not int or not -rank-1<=dim<=rank
                or any(value.shape!=values[0].shape for value in values)):
            return torch.stack(values,dim=dim,out=out)
        axis=dim%(rank+1);shape=list(values[0].shape);shape.insert(axis,len(values))
        result=_allocate(shape,values[0].dtype)
        torch.stack(values,dim=dim,out=result)
        _bump(direct_pinned_stack_calls=1,direct_pinned_output_bytes=result.numel()*result.element_size())
        return result


def _pinned_collator():
    global _PIN_COLLATE
    _guard_originals()
    with _LOCK:
        if _PIN_COLLATE is not None:return _PIN_COLLATE
        from torch_geometric.data import Batch
        from torch_geometric.data.collate import collate as pyg_collate,_collate as pyg_leaf
        proxy=_PinnedTorch()
        leaf_namespace=dict(pyg_leaf.__globals__);leaf_namespace['torch']=proxy
        leaf=_clone(pyg_leaf,leaf_namespace);leaf_namespace['_collate']=leaf
        outer_namespace=dict(pyg_collate.__globals__);outer_namespace['_collate']=leaf
        outer=_clone(pyg_collate,outer_namespace)
        from_namespace=dict(Batch.from_data_list.__func__.__globals__);from_namespace['collate']=outer
        from_list=_clone(Batch.from_data_list.__func__,from_namespace)
        facade=SimpleNamespace(from_data_list=lambda *args,**kwargs:from_list(Batch,*args,**kwargs))
        namespace=_private_import(_LOCAL_COLLATE,{('torch_geometric.data',('Batch',),0):SimpleNamespace(Batch=facade)})
        namespace['torch']=proxy
        _PIN_COLLATE=_clone(_LOCAL_COLLATE,namespace)
        return _PIN_COLLATE


def collate(items):
    started=time.perf_counter()
    try:return _pinned_collator()(items) if _PIN_OUTPUTS else _COLLATE(items)
    finally:_bump(collate_calls=1,collate_wall_seconds=time.perf_counter()-started)


def runtime_contract():
    _guard_originals()
    from torch_geometric.data import Batch
    from torch_geometric.data.collate import collate as pyg_collate,_collate as pyg_leaf
    return dict(format=FORMAT,runtime_source_sha256=_sha(__file__),
        original_inputs_source_sha256=_sha(inputs.__file__),
        original_local_collate_source_sha256=_sha(local.__file__),
        PyG_collate_source_sha256=_sha(inspect.getsourcefile(pyg_collate)),
        original_materialize_code_preserved=True,original_build_local_view_code_preserved=True,
        admitted_empty_context_and_bounded_scope_closure_chain_preserved=True,
        original_input_and_PyG_function_identity_and_source_stat_guards=True,
        original_collator_and_PyG_code_preserved=True,
        PyG_leaf_code_sha256=hashlib.sha256(pyg_leaf.__code__.co_code).hexdigest(),
        PyG_from_list_code_sha256=hashlib.sha256(Batch.from_data_list.__func__.__code__.co_code).hexdigest(),
        attribute_memo_scope='two views of one record only; discarded before return',
        attribute_memo_key='all exact argument bytes, dtype, shape, chunk size and overrides',
        memo_hits_return_independent_C_order_storage=True,
        all_edges_nodes_sampling_seeds_metadata_and_exceptions_preserved=True,
        pinned_final_outputs=_PIN_OUTPUTS,no_epoch_or_neural_cache=True,
        workers_data_model_or_batch_changed=False,shared_RSS_limit_changed=False)


def install_runtime(*,pin_final_outputs=True):
    global _PIN_OUTPUTS
    _guard_originals()
    if type(pin_final_outputs)is not bool:raise TypeError('Explicit original pin policy required')
    if torch.cuda.is_initialized():raise RuntimeError('Input runtime requires a fresh pre-CUDA process')
    replacements={'materialize_pair':(_PAIR,materialize_pair),'collate':(_COLLATE,collate)}
    for key,(original,replacement) in replacements.items():
        if memory._GET.__globals__.get(key) not in (original,replacement):
            raise ValueError('Foreign original input helper refused: '+key)
    for key,(_,replacement) in replacements.items():memory._GET.__globals__[key]=replacement
    _PIN_OUTPUTS=pin_final_outputs
    return runtime_contract()


def profile():
    with _LOCK:return dict(_STATS)


def bind(scorer):
    if bool(scorer.pin_cpu_batches)!=_PIN_OUTPUTS:
        raise ValueError('Actual original scorer pin-memory policy must match input runtime')
    raw=scorer.geometry.memory_guard.__self__;path=Path(__file__).resolve()
    proof=memory._ORIGINAL_PROVIDER._stat(path)
    raw._input_file_proofs[str(path)]=proof;raw._file_proofs[str(path)]=proof;raw.guard_source()
    if getattr(scorer,'_v24_input_runtime_binding',None)is None:
        before={}
        def pre(module,args,kwargs):before.clear();before.update(profile())
        def post(module,args,kwargs,result):
            if memory._ORIGINAL_PROVIDER._stat(path)!=proof:raise ValueError('Input runtime source changed')
            current=profile()
            if result is not None and hasattr(result,'workload'):
                result.workload['CPU_input_runtime']=dict(format=FORMAT,profile=current,
                    forward_delta={key:value-before.get(key,0) for key,value in current.items()},
                    timing_kind='CPU wall; materialization task durations can overlap across four workers')
        scorer._v24_input_runtime_binding=(scorer.register_forward_pre_hook(pre,with_kwargs=True),
            scorer.register_forward_hook(post,with_kwargs=True))
    return dict(contract=runtime_contract(),profile=profile())
