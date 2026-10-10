"""Exact v24 content SHA with direct CPU byte buffers, without leaf memoization.

The sealed inputs file and all its binding formats remain unchanged. Every
tensor value is read on every call; only the redundant flat tensor allocation
and Python bytes copy are removed. Installation precedes input construction.
"""
from __future__ import annotations

import hashlib
import ctypes
import inspect
import json
from pathlib import Path
import sys
import threading
from types import CodeType

import numpy as np
import torch

from . import v24_inputs as inputs

FORMAT='v24_exact_direct_buffer_content_hash_runtime_v1'
REFERENCE_MODULES=('hiercp_v1x.v24_inputs','hiercp_v1x.v24_geometry',
    'hiercp_v1x.v24_provider','hiercp_v1x.v24_factory','hiercp_v1x.v24_upper_reuse')
_ORIGINAL_DIGEST=inputs.tensor_digest
_ORIGINAL_ARRAY_DIGEST=inputs.array_digest
_LOCK=threading.RLock()


def _stat(path):
    value=Path(path).stat()
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_SOURCE_PATHS=(Path(__file__).resolve(),Path(inputs.__file__).resolve(),
    Path(inputs.__file__).resolve().with_name('v24_memory_runtime.py'))
_SOURCE_PROOFS={str(path):(_stat(path),_sha(path)) for path in _SOURCE_PATHS}


def _guard_sources():
    for path,(identity,_) in _SOURCE_PROOFS.items():
        if _stat(path)!=identity:
            raise ValueError('Admitted exact content-hash source changed: '+path)
    if (inputs.tensor_digest not in (_ORIGINAL_DIGEST,tensor_digest)
            or inputs.array_digest is not _ORIGINAL_ARRAY_DIGEST):
        raise ValueError('Foreign v24 digest replacement refused')


def _prove_original():
    """Match actual function code to its unchanged admitted source, not its name."""
    _guard_sources()
    compiled=compile(_SOURCE_PATHS[1].read_text(encoding='utf8'),
        _ORIGINAL_DIGEST.__code__.co_filename,'exec',dont_inherit=True)
    for name,original in (('tensor_digest',_ORIGINAL_DIGEST),('array_digest',_ORIGINAL_ARRAY_DIGEST)):
        matches=[value for value in compiled.co_consts if isinstance(value,CodeType) and value.co_name==name]
        if (len(matches)!=1 or original.__code__!=matches[0]
                or original.__globals__ is not vars(inputs) or original.__closure__ is not None
                or original.__defaults__ is not None or original.__kwdefaults__ is not None
                or original.__module__!='hiercp_v1x.v24_inputs'):
            raise ValueError('Exact original source/function identity required: '+name)
    if (inputs.__dict__['torch'] is not torch or inputs.__dict__['np'] is not np
            or inputs.__dict__['hashlib'] is not hashlib or inputs.__dict__['json'] is not json):
        raise ValueError('Original hash primitive globals changed')


def tensor_digest(value):
    """Identical headers/order/actual value bytes, with no hash-result cache."""
    _guard_sources()
    digest=hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            tensor=item.detach().cpu().contiguous()
            digest.update(json.dumps(['tensor',str(tensor.dtype),list(tensor.shape)],
                separators=(',',':')).encode())
            # Original copy_ resolves logical conjugate/negative views. Resolve
            # only those flags before the same flattened dtype reinterpretation.
            # BF16 never goes through NumPy's unsupported BF16 scalar dtype.
            logical=tensor.resolve_conj().resolve_neg()
            byte_array=logical.reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(byte_array))
        elif isinstance(item,np.ndarray):
            array=np.asarray(item)
            contiguous=np.ascontiguousarray(array)
            inner=hashlib.sha256()
            inner.update(json.dumps([str(array.dtype),list(array.shape)],
                separators=(',',':')).encode())
            if contiguous.nbytes:
                if contiguous.dtype.hasobject:
                    # NumPy prohibits object dtype reinterpretation but the
                    # original tobytes hashes its raw pointer representation.
                    # This read-only buffer covers exactly the actual owned
                    # contiguous allocation, whose owner stays live here.
                    raw=(ctypes.c_ubyte*contiguous.nbytes).from_address(contiguous.ctypes.data)
                    buffer=memoryview(raw).cast('B')
                else:
                    # Byte reinterpretation also covers datetime, structured
                    # and endian dtypes unsupported by Python's typed buffer.
                    buffer=memoryview(contiguous.reshape(-1).view(np.uint8))
                inner.update(buffer)
            digest.update(inner.hexdigest().encode())
        elif isinstance(item,dict):
            for key in sorted(item,key=repr):
                visit(key);visit(item[key])
        elif isinstance(item,(tuple,list)):
            digest.update(type(item).__name__.encode())
            for child in item:visit(child)
        elif isinstance(item,(str,int,float,bool)) or item is None:
            digest.update(json.dumps(item,allow_nan=False,separators=(',',':')).encode())
        else:
            raise TypeError('Unsupported input binding type: '+type(item).__name__)
    visit(value)
    return digest.hexdigest()


def runtime_contract():
    _prove_original()
    return dict(format=FORMAT,runtime_source_sha256=_SOURCE_PROOFS[str(_SOURCE_PATHS[0])][1],
        unchanged_input_source_sha256=_SOURCE_PROOFS[str(_SOURCE_PATHS[1])][1],
        original_digest_function_source_sha256=hashlib.sha256(
            inspect.getsource(_ORIGINAL_DIGEST).encode()).hexdigest(),
        original_function_code_matches_unchanged_source=True,
        reference_modules=list(REFERENCE_MODULES),
        private_reference_allowlist=['hiercp_v1x.v24_memory_runtime._GET.__globals__[tensor_digest]'],
        unchanged_memory_source_sha256=_SOURCE_PROOFS[str(_SOURCE_PATHS[2])][1],
        every_actual_tensor_value_hashed_each_call=True,hash_result_memoization=False,
        SHA_headers_order_and_value_bytes_unchanged=True,
        NumPy_dtype_shape_and_value_bytes_unchanged=True,
        NumPy_reference_function_unchanged=True,
        conjugate_and_negative_logical_values_preserved=True,
        redundant_flat_tensor_copy_removed=True,redundant_tensor_Python_bytes_copy_removed=True,
        redundant_NumPy_Python_bytes_copy_removed=True,
        scientific_source_files_modified=False,cache_or_query_format_changed=False,
        model_data_sampling_seed_RNG_changed=False)


def _private_memory_reference():
    """Admit only the known AST-proved deferred-get private global copy."""
    name='hiercp_v1x.v24_memory_runtime'
    module=sys.modules.get(name)
    if module is None:return None
    path=_SOURCE_PATHS[2]
    if Path(module.__file__).resolve()!=path:
        raise ValueError('Private memory digest reference is outside the admitted package')
    compiled=compile(path.read_text(encoding='utf8'),
        module._deferred_get.__code__.co_filename,'exec',dont_inherit=True)
    expected=[value for value in compiled.co_consts
        if isinstance(value,CodeType) and value.co_name=='_deferred_get']
    provider=sys.modules.get('hiercp_v1x.v24_provider')
    if (len(expected)!=1 or module._deferred_get.__code__!=expected[0]
            or module._deferred_get.__globals__ is not vars(module)
            or module.provider_module is not provider
            or module._ORIGINAL_PROVIDER is not provider.V24InputProvider):
        raise ValueError('Private deferred-get source/function/provider identity differs')
    reconstructed,proof=module._deferred_get()
    if (module._GET.__code__!=reconstructed.__code__ or module._GET_PROOF!=proof
            or proof['final_trim_expressions_changed']!=1
            or proof['tensor_operations_changed'] is not False
            or module._GET.__globals__.get('materialize_pair') is not inputs.materialize_pair
            or module._GET.__globals__.get('collate') is not inputs.collate):
        raise ValueError('Private deferred-get original tensor path changed')
    reference=module._GET.__globals__.get('tensor_digest')
    if reference not in (_ORIGINAL_DIGEST,tensor_digest):
        raise ValueError('Foreign private deferred-get digest replacement refused')
    return module._GET.__globals__,dict(
        name=name+'._GET.__globals__[tensor_digest]',
        unchanged_memory_source_sha256=_SOURCE_PROOFS[str(path)][1],
        private_get_code_reconstructed_from_original=True,
        private_get_AST_proof=dict(proof))


def install():
    """Patch only verified module references, before new inputs are created.

    Preflight all references before changing any. Newly imported modules then
    receive the same implementation through the original v24_inputs import.
    """
    with _LOCK:
        if torch.cuda.is_initialized():
            raise RuntimeError('Exact content-hash runtime must precede CUDA/input construction')
        contract=runtime_contract();patches=[]
        for name in REFERENCE_MODULES:
            module=sys.modules.get(name)
            if module is None or 'tensor_digest' not in vars(module):continue
            expected=_SOURCE_PATHS[1].with_name(name.rsplit('.',1)[-1]+'.py')
            if Path(module.__file__).resolve()!=expected:
                raise ValueError('Digest-reference module is outside the admitted package: '+name)
            reference=vars(module)['tensor_digest']
            if reference not in (_ORIGINAL_DIGEST,tensor_digest):
                raise ValueError('Foreign imported digest replacement refused: '+name)
            patches.append((name,module))
        private=_private_memory_reference()
        for _,module in patches:module.tensor_digest=tensor_digest
        if private is not None:private[0]['tensor_digest']=tensor_digest
        return dict(contract=contract,installed_reference_modules=[name for name,_ in patches],
            installed_private_global_references=[] if private is None else [private[1]],
            original_input_source_bytes_unchanged=True,
            all_loaded_references_preflighted_before_install=True,
            instructions='Install before input/provider/cache construction; no running-worker hot swap')
