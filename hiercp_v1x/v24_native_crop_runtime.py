"""Explicit native loader protocol for admitted lossless segmentation crops.

The pinned parent paste equations and its copied private package stay intact.
Only FrozenV23Loader overrides one ndarray admission predicate in memory.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import inspect
from pathlib import Path
import textwrap

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FORMAT='v24_exact_native_lossless_crop_protocol_v1'
METHOD='_apply_raw_paste_to_crop'
FROZEN_MODULE='nnunetv2.training.nnUNetTrainer.nnUNetTrainer_FrozenV23CP'
PARENT_FILE='custom_trainers/nnUNetTrainer_OnlinePairedCP.py'
FROZEN_FILE='custom_trainers/nnUNetTrainer_FrozenV23CP.py'
LOSSLESS_FILE='hiercp_v1x/v24_lossless_raw_storage.py'
_PREDICATE=ast.parse('isinstance(reference, np.ndarray)',mode='eval').body
_REPLACEMENT=ast.parse('_v24_native_reference_admitted(reference)',mode='eval').body


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _method_tree(source):
    tree=ast.parse(source)
    classes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='nnUNetDataLoaderOnlineCP']
    methods=[] if len(classes)!=1 else [node for node in classes[0].body if isinstance(node,ast.FunctionDef) and node.name==METHOD]
    if len(methods)!=1:raise ValueError('Exactly one actual original raw-paste loader method required')
    return ast.Module(body=[methods[0]],type_ignores=[])


def _replace_predicate(tree):
    tree=copy.deepcopy(tree);count=0
    class Admission(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal count
            if ast.dump(node)==ast.dump(_PREDICATE):
                count+=1
                return ast.copy_location(copy.deepcopy(_REPLACEMENT),node)
            return self.generic_visit(node)
    result=Admission().visit(tree)
    if count!=1:raise ValueError('Exactly one original isinstance(reference, np.ndarray) predicate required')
    return ast.fix_missing_locations(result)


def _helper_proof(source):
    tree=ast.parse(source)
    classes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='FrozenV23Bank']
    methods=[] if len(classes)!=1 else [node for node in classes[0].body if isinstance(node,ast.FunctionDef) and node.name=='_get_raw_store']
    if len(methods)!=1:raise ValueError('Actual frozen lossless storage helper required')
    imports=[node for node in ast.walk(methods[0]) if isinstance(node,ast.ImportFrom)
        and node.module=='hiercp_v1x.v24_lossless_raw_storage' and node.level==0]
    calls=[node for node in ast.walk(methods[0]) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
        and node.func.id=='LosslessRawBankStore' and len(node.args)==1
        and ast.dump(node.args[0])==ast.dump(ast.parse('self.root',mode='eval').body)]
    if len(imports)!=1 or [(name.name,name.asname) for name in imports[0].names]!=[('LosslessRawBankStore',None)] or len(calls)!=1:
        raise ValueError('Frozen helper must select the exact ROOT lossless raw storage class')
    return methods[0]


def crop_protocol_contract():
    """Pure source/AST proof; no native framework import or model construction."""
    before=_method_tree((ROOT/PARENT_FILE).read_text(encoding='utf8'))
    after=_replace_predicate(before)
    helper=_helper_proof((ROOT/FROZEN_FILE).read_text(encoding='utf8'))
    return dict(format=FORMAT,source_files_sha256={name:_sha(ROOT/name) for name in
        (PARENT_FILE,FROZEN_FILE,LOSSLESS_FILE,'custom_trainers/onlinecp_raw_bank.py','hiercp_v1x/v24_native_crop_runtime.py')},
        original_method_AST_sha256=hashlib.sha256(ast.dump(before).encode()).hexdigest(),
        adapted_method_AST_sha256=hashlib.sha256(ast.dump(after).encode()).hexdigest(),
        frozen_storage_helper_AST_sha256=hashlib.sha256(ast.dump(helper).encode()).hexdigest(),
        original_predicate='isinstance(reference, np.ndarray)',
        replacement_predicate='_v24_native_reference_admitted(reference)',replaced_predicates=1,
        admitted_proxy='exact ROOT _CropArray registered as baseline_seg by exact LosslessRawBankStore',
        parent_method_changed=False,private_package_files_changed=False,
        crop_equations_shape_dtype_baseline_equality_support_audit_unchanged=True,
        full_volume_conversion_enabled=False,model_or_loss_changed=False)


def _admitted_reference(reference,storage):
    if isinstance(reference,np.ndarray):return True
    if type(reference) is not storage._CropArray:return False
    store=reference._store()
    if (type(store) is not storage.LosslessRawBankStore or reference._closed
            or reference.dtype!=np.dtype('int16') or tuple(reference._spec.get('shape',()))!=reference.shape
            or np.dtype(reference._spec.get('dtype'))!=reference.dtype
            or reference._spec.get('storage')!=storage.FORMAT):return False
    # Exact class alone is insufficient: require the object admitted by the
    # actual load_case role validator and its SHA/stat-witnessed volume file.
    if not any(result.get('baseline_seg') is reference for _,result in store._cases.values()):return False
    spec=reference._spec
    if (spec['path'],spec['sha256']) not in store._witnesses:return False
    store._check(spec['path'],spec['sha256'])
    return True


def _adapt_method(original,storage,source_guard=lambda:None):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    expected=_method_tree((ROOT/PARENT_FILE).read_text(encoding='utf8'))
    if ast.dump(tree)!=ast.dump(expected):raise ValueError('Actual private parent paste method differs from exact ROOT source')
    changed=_replace_predicate(tree)
    def admitted(reference):
        source_guard()
        return _admitted_reference(reference,storage)
    namespace=dict(original.__globals__,_v24_native_reference_admitted=admitted)
    exec(compile(changed,inspect.getsourcefile(original)+':v24_admitted_lossless_crop','exec'),namespace)
    result=namespace[METHOD]
    result.__module__=original.__module__
    result.__defaults__=original.__defaults__;result.__kwdefaults__=copy.deepcopy(original.__kwdefaults__)
    return result


def install_crop_protocol():
    contract=crop_protocol_contract()
    frozen=importlib.import_module(FROZEN_MODULE)
    parent=importlib.import_module(frozen.nnUNetDataLoaderOnlineCP.__module__)
    storage=importlib.import_module('hiercp_v1x.v24_lossless_raw_storage')
    checks={Path(frozen.__file__).resolve():contract['source_files_sha256'][FROZEN_FILE],
        Path(parent.__file__).resolve():contract['source_files_sha256'][PARENT_FILE],
        Path(storage.__file__).resolve():contract['source_files_sha256'][LOSSLESS_FILE],
        Path(storage.ORIGINAL_SOURCE_PATH).resolve():contract['source_files_sha256']['custom_trainers/onlinecp_raw_bank.py']}
    if (Path(storage.__file__).resolve()!=ROOT/LOSSLESS_FILE
            or Path(storage.ORIGINAL_SOURCE_PATH).resolve()!=ROOT/'custom_trainers/onlinecp_raw_bank.py'
            or storage.ORIGINAL_SOURCE_SHA256!=checks[Path(storage.ORIGINAL_SOURCE_PATH).resolve()]
            or frozen.FrozenV23Loader.__bases__!=(frozen.nnUNetDataLoaderOnlineCP,)):
        raise ValueError('Actual Frozen loader and exact ROOT lossless backend binding required')
    checks.update({ROOT/name:checksum for name,checksum in contract['source_files_sha256'].items()})
    witnesses={path:(path.stat().st_dev,path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns,path.stat().st_ctime_ns)
        for path in checks}
    for path,checksum in checks.items():
        if _sha(path)!=checksum:raise ValueError('Actual private/native crop source differs from ROOT: '+str(path))
    def guard():
        for path,witness in witnesses.items():
            stat=path.stat()
            if (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)!=witness:
                raise ValueError('Admitted native crop protocol source changed: '+str(path))
    loader=frozen.FrozenV23Loader
    existing=loader.__dict__.get('_v24_crop_protocol_contract')
    if existing is not None:
        if existing!=contract or loader.__dict__.get(METHOD) is not loader.__dict__.get('_v24_crop_protocol_method'):
            raise ValueError('Existing native crop protocol override differs')
        guard();return contract
    if METHOD in loader.__dict__:raise ValueError('Frozen loader already has an unrecognized paste override')
    original=frozen.nnUNetDataLoaderOnlineCP.__dict__[METHOD]
    replacement=_adapt_method(original,storage,guard)
    setattr(loader,METHOD,replacement)
    loader._v24_crop_protocol_contract=copy.deepcopy(contract);loader._v24_crop_protocol_method=replacement
    guard()
    return contract


def run_training_entry():
    """Preserve the native CLI's exact argument tail and trainer/epoch behavior."""
    install_crop_protocol()
    from nnunetv2.run.run_training import run_training_entry as original_entry
    return original_entry()
