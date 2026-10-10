"""Explicit byte-exact static operator reuse for the complete native CP arms.

The original full-cache profile remains intact. This supplemental admission
adds only the six resampling matrices and both clipping arrays, independently
of learned scores, donors, selected candidates, or upper representations.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat as stat_module
import sys
import time
import types

import numpy as np

from . import v24_readonly_native_storage as core

EXTENSION = 'reuse_full105_exact_static_raw_case_operators_readonly_v1'
FORMAT = core.FORMAT
PROFILE = core.PROFILE
RESERVE = core.RESERVE
GROWTH_MARGIN = 512 * 2**20
OPERATOR_ROLES = ('data_operators/0', 'data_operators/1', 'data_operators/2',
                  'seg_operators/0', 'seg_operators/1', 'seg_operators/2', 'clip_min', 'clip_max')
CHECKPOINT_METHODS = ('save_checkpoint', 'on_train_end', 'on_epoch_end')
CHECKPOINT_SOURCE = 'training/nnUNetTrainer/nnUNetTrainer.py'
_BINDINGS = {}
_ORIGINAL_CHECK = None
_READER_ORIGINALS = None
_CHECKPOINT_BINDINGS = {}
COORDINATION = 'original_native_final_publication_flock_global7_v1'


class _UnaccountedHardlink(OSError):
    """A bounded namespace rescan distinguishes own publication from external links."""


def _role_keys(tree):
    result = {}
    for name in ('data_operators', 'seg_operators'):
        values = tree.get(name)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError('All three original resampling axes required: ' + name)
        for axis, value in enumerate(values):
            if not isinstance(value, dict) or set(value) != {'$array'}:
                raise ValueError('Exact original static array reference required')
            result[name + '/' + str(axis)] = value['$array']
    for name in ('clip_min', 'clip_max'):
        value = tree.get(name)
        if not isinstance(value, dict) or set(value) != {'$array'}:
            raise ValueError('Both original clipping arrays required')
        result[name] = value['$array']
    if len(set(result.values())) != 8:
        raise ValueError('Exactly eight distinct static operator/clipping arrays required')
    return result


def _array_bytes(array):
    return hashlib.sha256(np.asarray(array).tobytes(order='C')).hexdigest()


def _npy_header(path):
    with Path(path).open('rb') as stream:
        version=np.lib.format.read_magic(stream)
        shape,fortran,dtype=np.lib.format._read_array_header(stream,version)
    return dict(shape=list(shape),dtype=dtype.str,fortran_order=bool(fortran))


def _serialized_layout(array):
    # This is exactly np.save's existing array order rule. It does not copy,
    # cast, reshape, or change the original scientific preparation array.
    fortran=bool(array.flags.f_contiguous and not array.flags.c_contiguous)
    strides=[];stride=array.dtype.itemsize
    for axis in (range(array.ndim) if fortran else range(array.ndim-1,-1,-1)):
        strides.append((axis,stride));stride*=array.shape[axis]
    return dict(fortran_order=fortran,strides=[value for _,value in sorted(strides)])


def _old_base(path):
    """Verify the admitted old helper and all old witnesses without rehashing volumes."""
    file_proof = core.proof(path)
    document = core.read(path)
    adapter = document['adapter_source']
    adapter_path = core.guard(adapter, full_hash=True)
    name = 'hiercp_v1x._readonly_admitted_base_' + adapter['sha256']
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, adapter_path)
        if spec is None or spec.loader is None:
            raise ValueError('The exact admitted immutable base helper must be loadable')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != adapter_path or core.sha(adapter_path) != adapter['sha256']:
        raise ValueError('Admitted historical base helper identity differs')
    module.verify_admission(document, full_hash=False)
    core.guard(file_proof, full_hash=True)
    return document, file_proof, copy.deepcopy(adapter)


def _compiled_methods(path, *, filename=None):
    source = core.regular(path).read_text(encoding='utf8')
    compiled = compile(source, str(path) if filename is None else str(filename), 'exec', dont_inherit=True)
    classes = [code for code in compiled.co_consts
               if isinstance(code, types.CodeType) and code.co_name == 'nnUNetTrainer']
    if len(classes) != 1:
        raise ValueError('Exactly one admitted original nnUNetTrainer class required')
    methods = {code.co_name: code for code in classes[0].co_consts if isinstance(code, types.CodeType)}
    if not set(CHECKPOINT_METHODS) <= methods.keys():
        raise ValueError('All original checkpoint publication methods required')
    return {name: core.code_fingerprint(methods[name]) for name in CHECKPOINT_METHODS}


def _checkpoint_source_proof(base, checkpoint_paths, checkpoint_max_bytes):
    """Prove original direct publication and final-before-latest-removal ordering."""
    value = base['runtime_files'][CHECKPOINT_SOURCE]
    path = core.guard(value, full_hash=True)
    tree = ast.parse(path.read_text(encoding='utf8'))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'nnUNetTrainer']
    methods = {} if len(classes) != 1 else {node.name: node for node in classes[0].body if isinstance(node, ast.FunctionDef)}
    if not set(CHECKPOINT_METHODS) <= methods.keys():
        raise ValueError('Original checkpoint source methods missing')
    saves = [node for node in ast.walk(methods['save_checkpoint']) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
             and node.func.value.id == 'torch' and node.func.attr == 'save']
    if (len(saves) != 1 or saves[0].keywords or len(saves[0].args) != 2
            or not isinstance(saves[0].args[0], ast.Name) or saves[0].args[0].id != 'checkpoint'
            or not isinstance(saves[0].args[1], ast.Name) or saves[0].args[1].id != 'filename'):
        raise ValueError('Exact original direct torch.save(checkpoint, filename) required')
    forbidden={'mkstemp','NamedTemporaryFile','TemporaryFile','replace','rename','link'}
    if any(isinstance(node,ast.Call) and (
            isinstance(node.func,ast.Attribute) and node.func.attr in forbidden
            or isinstance(node.func,ast.Name) and node.func.id in forbidden)
            for name in CHECKPOINT_METHODS for node in ast.walk(methods[name])):
        raise ValueError('Original direct checkpoint publication cannot add temporary/replacement operations')
    end = methods['on_train_end']
    calls = [node for node in ast.walk(end) if isinstance(node, ast.Call)]
    final = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'save_checkpoint'
             and any(isinstance(item, ast.Constant) and item.value == 'checkpoint_final.pth' for item in ast.walk(node))]
    removals = [node for node in calls if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'os' and node.func.attr == 'remove'
                and any(isinstance(item, ast.Constant) and item.value == 'checkpoint_latest.pth' for item in ast.walk(node))]
    epoch_saves = [node for node in ast.walk(methods['on_epoch_end']) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == 'save_checkpoint']
    filenames = {item.value for node in epoch_saves for item in ast.walk(node)
                 if isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value.endswith('.pth')}
    if (len(final) != 1 or len(removals) != 1 or final[0].lineno >= removals[0].lineno
            or len(epoch_saves) != 2 or filenames != {'checkpoint_latest.pth', 'checkpoint_best.pth'}):
        raise ValueError('Original best/latest/final publication ordering must remain exact')
    for trainer in base['custom_trainer_files'].values():
        source = ast.parse(core.guard(trainer, full_hash=True).read_text(encoding='utf8'))
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in CHECKPOINT_METHODS
               for node in ast.walk(source)):
            raise ValueError('Custom trainers must inherit the admitted checkpoint methods unchanged')
    if type(checkpoint_max_bytes) is not int or checkpoint_max_bytes <= 0:
        raise ValueError('Positive measured full original checkpoint byte bound required')
    samples = []
    for sample in checkpoint_paths:
        sample = core.regular(sample)
        witness = core.stat(sample)
        if sample.name not in ('checkpoint_best.pth', 'checkpoint_final.pth') or witness[2] > checkpoint_max_bytes:
            raise ValueError('Measured original complete checkpoint exceeds the explicit byte bound')
        samples.append(dict(path=str(sample), stat=witness, bytes=witness[2]))
    if {row['path'].rsplit('/', 1)[-1].rsplit('\\', 1)[-1] for row in samples} != {'checkpoint_best.pth', 'checkpoint_final.pth'}:
        raise ValueError('Both complete original best/final checkpoint size witnesses required')
    return dict(source=copy.deepcopy(value), compiled_methods=_compiled_methods(path),
        code_fingerprint_format=core.CODE_FINGERPRINT, direct_torch_save=True, atomic_temporary_checkpoint=False,
        final_published_before_latest_removed=True, peak_checkpoint_slots=3, completed_checkpoint_slots=2,
        checkpoint_max_bytes=checkpoint_max_bytes, measured_complete_checkpoint_files=samples,
        retention_changed=False)


def _arm_roots(roots):
    if set(roots) != {'gpu4', 'gpu5', 'gpu6'}:
        raise ValueError('Explicit complete GPU4/GPU5/GPU6 output reservations required')
    result = {}
    devices = set()
    for arm, text in roots.items():
        path = Path(text)
        if not path.is_absolute() or '..' in path.parts or any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError('Exact independent own-arm output roots required')
        parent = path
        while not parent.exists():
            parent = parent.parent
        core.regular(parent, directory=True)
        devices.add(parent.stat().st_dev)
        result[arm] = dict(root=str(path.absolute()), existing_parent=str(parent), filesystem_device=parent.stat().st_dev)
    paths = [Path(row['root']) for row in result.values()]
    if len(devices) != 1 or any(a.is_relative_to(b) or b.is_relative_to(a)
            for index, a in enumerate(paths) for b in paths[index+1:]):
        raise ValueError('Three disjoint roots on the one admitted writable filesystem required')
    return result


def _allocated_bytes(path):
    """Measure only explicitly reserved owned output trees, never shared inputs."""
    path=Path(path)
    if not path.exists():return 0
    core.regular(path,directory=True)
    for attempt in range(3):
        inodes={}
        try:
            for entry in path.rglob('*'):
                value=entry.lstat()
                if stat_module.S_ISLNK(value.st_mode):raise ValueError('Reserved native output tree cannot contain symlinks')
                if value.st_dev!=path.stat().st_dev:raise ValueError('Reserved output must remain on its admitted filesystem')
                if stat_module.S_ISDIR(value.st_mode):core.regular(entry,directory=True)
                elif stat_module.S_ISREG(value.st_mode):
                    core.regular(entry);key=(value.st_dev,value.st_ino)
                    if key not in inodes:inodes[key]=dict(paths=[],bytes=value.st_blocks*512 if hasattr(value,'st_blocks') else value.st_size)
                    inodes[key]['paths'].append(entry)
                else:raise ValueError('Only own regular native output files/directories allowed')
            for value in inodes.values():
                # Atomic exclusive publication may expose two names briefly.
                # Only links wholly inside this reserved tree are admissible.
                links=value['paths'][0].stat().st_nlink
                if links>len(value['paths']):
                    raise _UnaccountedHardlink('Unaccounted output hardlink during namespace scan')
            return sum(value['bytes'] for value in inodes.values())
        except (FileNotFoundError,_UnaccountedHardlink) as error:
            if attempt==2:
                if isinstance(error,_UnaccountedHardlink):
                    raise ValueError('External hardlinks cannot count as fresh private native writes') from error
                raise OSError('Owned native publication files kept changing during allocation measurement') from error
            time.sleep(.01)
    raise RuntimeError('Native output measurement retry exhausted')


def build_admission(base_admission_path, *, arm_roots, checkpoint_paths, checkpoint_max_bytes,
                    growth_margin_bytes=GROWTH_MARGIN, workers=4, progress=None):
    """Reuse all validated base witnesses and hash only the additional 840 NPYs."""
    if workers != 4 or type(workers) is not int:
        raise ValueError('Four assigned independent hash workers required')
    if type(growth_margin_bytes) is not int or growth_margin_bytes < GROWTH_MARGIN:
        raise ValueError('At least512MiB explicit aggregate growth margin required')
    base, base_proof, old_adapter = _old_base(base_admission_path)
    roots = _arm_roots(arm_roots)
    protected=(Path(base['source_bank']['root']),Path(base['baseline']['preprocessed']),
               Path(base['inventory']['path']),Path(base['adapter_source']['path']),Path(core.__file__).resolve())
    for value in roots.values():
        path=Path(value['root'])
        if any(path.is_relative_to(source) or source.is_relative_to(path) for source in protected):
            raise ValueError('Reserved new arm output must be disjoint from authoritative source/code/cache')
    root = Path(base['source_bank']['root'])
    pending = {}; paths = []
    for case_id, cached in base['raw_cases'].items():
        manifest = core.read(core.guard(cached['manifest']))
        keys = _role_keys(manifest['tree'])
        operators = {}
        for role, key in keys.items():
            spec = copy.deepcopy(manifest['arrays'][key])
            path = core._relative(root, spec['path'])
            if (path.suffix != '.npy' or set(spec)!={'shape','dtype','path','sha256'}
                    or spec['path']!='raw_cases/'+case_id+'.'+key+'.npy'
                    or np.dtype(spec['dtype']) != np.dtype('float64')):
                raise ValueError('Only original exact float64 NPY operators/clipping arrays can be reused')
            if (role.startswith(('data_operators/', 'seg_operators/')) and len(spec['shape']) != 2
                    or role in ('clip_min', 'clip_max') and len(spec['shape']) != 1
                    or any(type(n) is not int or n <= 0 for n in spec['shape'])):
                raise ValueError('Exact original resampling matrix/clipping shapes required')
            operators[key] = dict(role=role, spec=spec); paths.append(path)
        pending[case_id] = operators
    if len(paths)!=840 or len({str(path) for path in paths})!=840:
        raise ValueError('Exactly840 distinct original per-case static NPY files required')
    hashes = core._parallel_proofs(paths, workers=workers, progress=progress, phase='full105_static_operator_NPY_SHA')
    document = copy.deepcopy(base); reused = 0
    for case_id, operators in pending.items():
        for row in operators.values():
            value = hashes[str(core._relative(root, row['spec']['path']).absolute())]
            core.guard(value)
            if value['sha256'] != row['spec']['sha256']:
                raise ValueError('Original static operator SHA differs from its completed manifest')
            array = np.load(value['path'], mmap_mode='r', allow_pickle=False)
            try:
                header=_npy_header(value['path']);layout=_serialized_layout(array)
                if (array.dtype.str != row['spec']['dtype'] or list(array.shape) != row['spec']['shape']
                        or header!={'shape':row['spec']['shape'],'dtype':row['spec']['dtype'],'fortran_order':layout['fortran_order']}
                        or list(array.strides)!=layout['strides']):
                    raise ValueError('Actual full static operator NPY header differs')
                row['value_bytes_sha256'] = _array_bytes(array)
                row['array_bytes'] = int(array.nbytes)
                row['npy_header']=header;row['serialized_layout']=layout
                row['file'] = copy.deepcopy(value); reused += value['bytes']
            finally:
                del array
            core.guard(value)
        document['raw_cases'][case_id]['static_operator_arrays'] = operators
    checkpoint = _checkpoint_source_proof(base, checkpoint_paths, checkpoint_max_bytes)
    components = copy.deepcopy(base['budget']['measured_components'])
    if components['private_small_bank_artifacts_bytes'] < reused:
        raise ValueError('Measured full small-bank budget cannot be smaller than its static arrays')
    components['private_small_bank_artifacts_bytes'] -= reused
    components['explicit_additional_fresh_outputs_bytes'] = 3 * checkpoint_max_bytes + 16*2**20 + 64*2**20
    writable = sum(components.values())
    global_peak=3*(writable-3*checkpoint_max_bytes)+7*checkpoint_max_bytes
    aggregate_required = global_peak + RESERVE + growth_margin_bytes
    available = shutil.disk_usage(next(iter(roots.values()))['existing_parent']).free
    if available < aggregate_required:
        raise OSError('Complete three-arm original native writes plus10GiB reserve/growth do not fit')
    parents={str(Path(row['root']).parent) for row in roots.values()}
    if len(parents)!=1:raise ValueError('Final-publication coordination requires one own sibling-arm namespace')
    group=core.regular(next(iter(parents)),directory=True);lock=group/'native_final_publication.lock'
    try:
        descriptor=os.open(lock,os.O_RDWR|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    except FileExistsError:
        core.regular(lock)
    else:os.close(descriptor)
    if _lock_mode(lock)!=0o600 or lock.stat().st_nlink!=1 or lock.stat().st_size!=0:
        raise ValueError('Exact own empty0600 native final-publication lock required')
    from . import v24_nnunet_cp as pipeline
    fold_relative=('native/nnUNet_results/'+base['baseline']['dataset_name']+'/'+pipeline.TRAINER+'__'
                   +base['baseline']['plans_name']+'__'+base['baseline']['configuration']+'/fold_0')
    coordination=dict(format=COORDINATION,group_root=str(group),lock_file=core.proof(lock),mode=0o600,
        global_peak_checkpoint_slots=7,per_arm_peak_checkpoint_slots=3,
        original_final_before_latest_removal=True,optimizer_work_locked=False,retention_changed=False,
        failure_blocks_next_final=True,repeated_final_rejected=True,
        folds={arm:str(Path(row['root'])/fold_relative) for arm,row in roots.items()},adapter_source=core.proof(__file__))
    document.update(storage_extension=EXTENSION, static_operator_adapter_source=core.proof(__file__),
        base_admission_proof=base_proof, base_adapter_source=old_adapter,
        checkpoint_publication_proof=checkpoint,checkpoint_coordination=coordination, adapter_source=core.proof(core.__file__),
        static_operator_roles=list(OPERATOR_ROLES), static_operator_arrays_reused=840,
        donors_reused=False, selected_candidates_reused=False, static_operator_value_bytes_verified=True,
        aggregate_budget=dict(arms=roots,per_arm_new_writable_bytes={arm:writable for arm in roots},
            initial_allocated_bytes={arm:_allocated_bytes(value['root']) for arm,value in roots.items()},
            all_three_uncoordinated_new_writable_bytes=3*writable,all_three_new_writable_bytes=global_peak,
            global_peak_checkpoint_slots=7,minimum_runtime_free_bytes=RESERVE+growth_margin_bytes,
            growth_margin_bytes=growth_margin_bytes,required_free_bytes=aggregate_required,
            initial_available_free_bytes=available,initial_admitted_at=time.time()),
        budget=dict(base['budget'], measured_components=components,new_writable_bytes_estimate=writable,
            private_geometry_scoring_bytes=components['explicit_additional_fresh_outputs_bytes'],
            measured_private_small_bank_bytes=components['private_small_bank_artifacts_bytes'],
            required_free_bytes=writable+RESERVE,native_checkpoint_slots=3,checkpoint_max_bytes=checkpoint_max_bytes,
            estimated_static_operator_reuse_bytes=reused,reused_readonly_bytes=base['budget']['reused_readonly_bytes']+reused),
        extension_created_at=time.time())
    verify_admission(document)
    return document


def verify_admission(document, *, full_hash=False):
    document = core.read(document) if isinstance(document, (str, Path)) else document
    core.verify_admission(document, full_hash=full_hash)
    if (document.get('storage_extension') != EXTENSION or document.get('static_operator_roles') != list(OPERATOR_ROLES)
            or document.get('static_operator_arrays_reused') != 840
            or document.get('donors_reused') is not False or document.get('selected_candidates_reused') is not False
            or document.get('static_operator_value_bytes_verified') is not True):
        raise ValueError('Explicit exact full105 static-only operator extension required')
    core.guard(document['base_admission_proof'], full_hash=True)
    base, base_proof, old_adapter = _old_base(document['base_admission_proof']['path'])
    if base_proof != document['base_admission_proof'] or old_adapter != document['base_adapter_source']:
        raise ValueError('Supplemental admission historical base binding differs')
    core.guard(document['static_operator_adapter_source'], full_hash=True)
    if core.sha(__file__) != document['static_operator_adapter_source']['sha256']:
        raise ValueError('Static operator adapter source changed')
    for key in ('inventory', 'split', 'baseline', 'source_bank', 'raw_source_files', 'preprocessed',
                'private_metadata_files', 'ground_truth_files', 'runtime_package', 'runtime_files', 'custom_trainer_files'):
        if document[key] != base[key]:
            raise ValueError('Full authoritative base source proofs changed: ' + key)
    count = 0; reused = 0
    for case_id, cached in document['raw_cases'].items():
        if {key:value for key,value in cached.items() if key != 'static_operator_arrays'} != base['raw_cases'][case_id]:
            raise ValueError('Authoritative raw case volume/metadata proof changed')
        manifest = core.read(core.guard(cached['manifest']))
        keys = _role_keys(manifest['tree']); operators = cached.get('static_operator_arrays', {})
        if set(operators) != set(keys.values()):
            raise ValueError('Exactly all eight original per-case static arrays required')
        for role, key in keys.items():
            row = operators[key]; spec = row['spec']
            if (row['role'] != role or spec != manifest['arrays'][key]
                    or set(spec)!={'shape','dtype','path','sha256'}
                    or spec['path']!='raw_cases/'+case_id+'.'+key+'.npy'
                    or np.dtype(spec['dtype']) != np.dtype('float64')
                    or row['file']['path'] != str(core._relative(base['source_bank']['root'], spec['path']))
                    or row['file']['sha256'] != spec['sha256']
                    or row['array_bytes'] != int(np.prod(spec['shape'])) * np.dtype(spec['dtype']).itemsize
                    or row.get('npy_header')!={'shape':spec['shape'],'dtype':spec['dtype'],
                                              'fortran_order':row.get('serialized_layout',{}).get('fortran_order')}
                    or set(row.get('serialized_layout',{}))!={'fortran_order','strides'}
                    or not isinstance(row['value_bytes_sha256'], str) or len(row['value_bytes_sha256']) != 64):
                raise ValueError('Static role/content/shape/path binding differs')
            core.guard(row['file'], full_hash=full_hash); count += 1; reused += row['file']['bytes']
    checkpoint = document['checkpoint_publication_proof']
    rebuilt = _checkpoint_source_proof(base, [row['path'] for row in checkpoint['measured_complete_checkpoint_files']],
                                       checkpoint['checkpoint_max_bytes'])
    if rebuilt != checkpoint:
        raise ValueError('Original native three-slot source/publication proof changed')
    components = copy.deepcopy(base['budget']['measured_components'])
    components['private_small_bank_artifacts_bytes'] -= reused
    components['explicit_additional_fresh_outputs_bytes'] = 3*checkpoint['checkpoint_max_bytes'] + 80*2**20
    budget = document['budget']; aggregate = document['aggregate_budget']; writable = sum(components.values())
    roots = _arm_roots({arm:value['root'] for arm,value in aggregate['arms'].items()})
    coordination=document['checkpoint_coordination'];_coordination_guard(document)
    peak=3*(writable-3*checkpoint['checkpoint_max_bytes'])+7*checkpoint['checkpoint_max_bytes']
    recorded_roots={arm:{key:value for key,value in row.items()} for arm,row in aggregate['arms'].items()}
    if (count != 840 or components['private_small_bank_artifacts_bytes'] < 0
            or budget['measured_components'] != components or budget['new_writable_bytes_estimate'] != writable
            or budget['native_checkpoint_slots'] != 3 or budget['checkpoint_max_bytes'] != checkpoint['checkpoint_max_bytes']
            or budget['estimated_static_operator_reuse_bytes'] != reused
            or budget['reused_readonly_bytes'] != base['budget']['reused_readonly_bytes']+reused
            or set(recorded_roots)!=set(roots)
            or any(row['root']!=roots[arm]['root'] or row['filesystem_device']!=roots[arm]['filesystem_device']
                   or not Path(row['root']).is_relative_to(Path(row['existing_parent']))
                   for arm,row in recorded_roots.items())
            or aggregate['per_arm_new_writable_bytes'] != {arm:writable for arm in roots}
            or aggregate['all_three_uncoordinated_new_writable_bytes'] != 3*writable
            or aggregate['all_three_new_writable_bytes'] != peak or aggregate['global_peak_checkpoint_slots']!=7
            or set(aggregate['initial_allocated_bytes'])!=set(roots)
            or any(type(value)is not int or value<0 for value in aggregate['initial_allocated_bytes'].values())
            or type(aggregate['growth_margin_bytes']) is not int or aggregate['growth_margin_bytes'] < GROWTH_MARGIN
            or aggregate['minimum_runtime_free_bytes'] != RESERVE+aggregate['growth_margin_bytes']
            or aggregate['required_free_bytes'] != peak+RESERVE+aggregate['growth_margin_bytes']
            or aggregate['initial_available_free_bytes'] < aggregate['required_free_bytes']):
        raise ValueError('Explicit original three-arm write budget/reserve/growth proof differs')
    return document


def check_aggregate_disk(admission,chain_root,*,preparation=True):
    """Admit remaining writes across all3 without charging completed bytes twice."""
    document=verify_admission(admission);aggregate=document['aggregate_budget'];root=Path(chain_root).absolute()
    roots=aggregate['arms'];matches=[arm for arm,row in roots.items() if Path(row['root'])==root]
    if len(matches)!=1:raise ValueError('Exact own arm output root must match the aggregate admission')
    row=roots[matches[0]];parent=root
    while not parent.exists():parent=parent.parent
    core.regular(parent,directory=True)
    if parent.stat().st_dev!=row['filesystem_device']:raise ValueError('Reserved native filesystem changed')
    floor=aggregate['minimum_runtime_free_bytes']
    remaining={};written={}
    if preparation:
        for arm,value in roots.items():
            allocated=_allocated_bytes(value['root'])
            delta=max(0,allocated-aggregate['initial_allocated_bytes'][arm])
            provision=aggregate['per_arm_new_writable_bytes'][arm]
            if delta>provision:raise OSError('Actual own-arm writes exceed the measured full native provision: '+arm)
            written[arm]=delta;remaining[arm]=provision-delta
        required=max(0,aggregate['all_three_new_writable_bytes']-sum(written.values()))+floor
    else:required=floor
    # Concurrent writes after an earlier inode measurement make this later
    # free-space reading conservative. Sampling before the scan is unsafe.
    available=shutil.disk_usage(parent).free
    if available<required:raise OSError('Complete remaining three-arm writes plus10GiB/growth reserve do not fit')
    return dict(storage_extension=EXTENSION,arm=matches[0],actual_free_bytes=available,required_free_bytes=required,
        minimum_runtime_free_bytes=floor,remaining_new_writable_bytes=remaining,actual_new_allocated_bytes=written,
        expensive_output_scan=bool(preparation))


def make_case_preparer(pipeline, admission):
    """Keep original preparation bytecode; compare every fresh static array byte."""
    from . import v24_lossless_raw_storage as storage
    document = verify_admission(admission)
    original = pipeline._root_raw_preparation()
    def save_case(bank_root, relative_manifest, case):
        case_id = case['metadata']['case_id']; cached = document['raw_cases'][case_id]
        if case['metadata'] != cached['metadata']:
            raise ValueError('Fresh original case metadata differs from static cache')
        value = {key:item for key,item in case.items() if key != 'preparation'}; arrays = {}
        tree = storage.original._encode_tree(value, arrays)
        roles = {tree[role]['$array']:role for role in core.ROLES}
        operators = {key:role for role,key in _role_keys(tree).items()}
        cached_by_role={row['role']:row for row in cached['static_operator_arrays'].values()}
        manifest = dict(format=storage.original.STORAGE_FORMAT,kind='case',tree=tree,arrays={},
            v24_storage=storage.FORMAT,compression=dict(storage.COMPRESSION),readonly_storage_profile=PROFILE,
            readonly_source_volumes={},readonly_static_operator_profile=EXTENSION,readonly_static_operator_arrays={})
        for key,array in arrays.items():
            if key in roles:
                role=roles[key];row=cached['volumes'][role];core.guard(row['file']);spec=copy.deepcopy(row['spec'])
                if array.dtype.str != spec['dtype'] or list(array.shape) != spec['shape']:
                    raise ValueError('Fresh original full-precision volume grid differs')
                if storage._verify_all(array,row['file']['path'],spec) != spec['roundtrip']:
                    raise ValueError('Fresh original full-volume bytes differ')
                relative=str(PurePosixPath(relative_manifest).with_suffix(''))+'.'+key+'.b2nd';spec['path']=relative
                manifest['arrays'][key]=spec
                manifest['readonly_source_volumes'][relative]=dict(case_id=case_id,role=role,source=copy.deepcopy(row['file']))
            elif key in operators:
                row=cached_by_role[operators[key]];path=core.guard(row['file']);spec=copy.deepcopy(row['spec'])
                source=np.load(path,mmap_mode='r',allow_pickle=False)
                try:
                    if (array.dtype.str != spec['dtype'] or list(array.shape) != spec['shape']
                            or source.dtype.str != spec['dtype'] or list(source.shape) != spec['shape']
                            or _serialized_layout(array)!=row['serialized_layout']
                            or _serialized_layout(source)!=row['serialized_layout']
                            or list(source.strides)!=row['serialized_layout']['strides']
                            or _array_bytes(array) != row['value_bytes_sha256']
                            or _array_bytes(source) != row['value_bytes_sha256']
                            or array.tobytes(order='C') != source.tobytes(order='C')):
                        raise ValueError('Fresh complete original static operator/clipping bytes differ: '+case_id+'/'+row['role'])
                finally:
                    del source
                core.guard(row['file'])
                relative=str(PurePosixPath(relative_manifest).with_suffix(''))+'.'+key+'.npy';spec['path']=relative
                manifest['arrays'][key]=spec
                manifest['readonly_static_operator_arrays'][relative]=dict(case_id=case_id,role=row['role'],source=copy.deepcopy(row['file']))
            else:
                relative=str(PurePosixPath(relative_manifest).with_suffix(''))+'.'+key+'.npy'
                target=storage.original._relative_path(bank_root,relative,must_exist=False)
                checksum=storage.original._publish(target,lambda stream,a=array:np.save(stream,a,allow_pickle=False))
                manifest['arrays'][key]=dict(shape=list(array.shape),dtype=array.dtype.str,path=relative,sha256=checksum)
        manifest['lossless_storage']=copy.deepcopy(cached['lossless_storage'])
        path=storage.original._relative_path(bank_root,relative_manifest,must_exist=False)
        return storage.original._publish(path,lambda stream:stream.write((json.dumps(manifest,sort_keys=True,
            allow_nan=False,separators=(',',':'))+'\n').encode('utf8')))
    inner = core._clone(original._prepare_raw_case, dict(save_case=save_case))
    return core._clone(original.prepare_raw_case, dict(_prepare_raw_case=inner))


def install_pipeline_adapter(pipeline, base_admission_path, extension_admission_path=None):
    extension_admission_path = base_admission_path if extension_admission_path is None else extension_admission_path
    document = verify_admission(extension_admission_path)
    if Path(base_admission_path).resolve() != Path(extension_admission_path).resolve():
        raise ValueError('Single complete supplemental storage admission required')
    original = pipeline._materialize_bank; publisher = original.__globals__['new_json']
    def new_json(path, value):
        if Path(path).name == 'index.json':
            value=copy.deepcopy(value);value.update(storage_profile=PROFILE,
                storage_admission=str(Path(extension_admission_path).resolve()),
                storage_admission_sha256=core.sha(extension_admission_path),storage_extension=EXTENSION,
                static_raw_volume_reuse_only=True,static_operator_byte_exact_reuse=True,
                static_operator_adapter_sha256=document['static_operator_adapter_source']['sha256'])
            verify_admission(document)
        return publisher(path,value)
    def preparer():return make_case_preparer(pipeline,document)
    adapted=core._clone(original,dict(lossless_raw_case_preparer=preparer,new_json=new_json))
    pipeline._materialize_bank=adapted
    return dict(profile=PROFILE,storage_extension=EXTENSION,original_materializer_code_preserved=adapted.__code__ is original.__code__,
        globals_replaced=['lossless_raw_case_preparer','new_json'],scores_reused=False,learned_upper_reused=False,
        donors_reused=False,selected_candidates_reused=False)


def prepare_native(bank_path, output, admission_path, pipeline):
    document=verify_admission(admission_path);bank=core.read(bank_path)
    if bank.get('storage_extension') != EXTENSION or bank.get('static_operator_adapter_sha256') != document['static_operator_adapter_source']['sha256']:
        raise ValueError('Own fresh bank must declare its exact static operator extension')
    original=core.prepare_native;publisher=original.__globals__['publish_json']
    def publish_json(path,value):
        if Path(path).name == 'native.json':
            value=copy.deepcopy(value);value.update(storage_extension=EXTENSION,
                static_operator_adapter_sha256=document['static_operator_adapter_source']['sha256'],
                checkpoint_publication_proof=copy.deepcopy(document['checkpoint_publication_proof']),
                checkpoint_coordination=copy.deepcopy(document['checkpoint_coordination']),
                static_operator_byte_exact_reuse=True)
        return publisher(path,value)
    return core._clone(original,dict(publish_json=publish_json))(bank_path,output,admission_path,pipeline)


def _reader_codes(storage):
    return dict(load_case=storage.LosslessRawBankStore.load_case.__code__,
        crop_getitem=storage._CropArray.__getitem__.__code__,numpy_load=np.load.__code__)


def _reader_code_guard(storage):
    current=_reader_codes(storage)
    if _READER_ORIGINALS is None or any(current[name] is not code for name,code in _READER_ORIGINALS.items()):
        raise ValueError('Original raw crop/store/NPY reader code changed')


def install_raw_store_adapter(admission,bank_root):
    global _ORIGINAL_CHECK,_READER_ORIGINALS
    from . import v24_lossless_raw_storage as storage
    document=verify_admission(admission);root=core.regular(Path(bank_root).resolve(),directory=True)
    base_proof=core.install_raw_store_adapter(document,root);references={}
    manifests={path.stem:path for path in (root/'raw_cases').glob('*.json')}
    if set(manifests) != set(document['raw_cases']):raise ValueError('Fresh full105 own-bank manifests required')
    for case_id,path in manifests.items():
        manifest=core.read(path);cached=document['raw_cases'][case_id];keys=_role_keys(manifest['tree'])
        values=manifest.get('readonly_static_operator_arrays',{})
        if manifest.get('readonly_static_operator_profile') != EXTENSION or len(values) != 8:
            raise ValueError('Exactly eight explicit readonly static operator references required per case')
        expected={}
        for role,key in keys.items():
            cached_by_role={row['role']:row for row in cached['static_operator_arrays'].values()}
            row=cached_by_role[role];relative='raw_cases/'+case_id+'.'+key+'.npy'
            if (manifest['arrays'][key] != dict(row['spec'],path=relative) or (root/relative).exists()
                    or values.get(relative) != dict(case_id=case_id,role=role,source=row['file'])):
                raise ValueError('Exact static operator source-role reference differs')
            expected[relative]=copy.deepcopy(row['file'])
        if set(values) != set(expected):raise ValueError('Extra learned/donor/candidate reference prohibited')
        references.update(expected)
    binding=dict(document=document,references=references)
    if str(root) in _BINDINGS and _BINDINGS[str(root)] != binding:raise ValueError('Static operator bank root bound differently')
    _BINDINGS[str(root)]=binding
    if _READER_ORIGINALS is None:_READER_ORIGINALS=_reader_codes(storage)
    _reader_code_guard(storage)
    if _ORIGINAL_CHECK is None:
        _ORIGINAL_CHECK=storage.LosslessRawBankStore._check
        def check(self,relative,digest):
            admitted=_BINDINGS.get(str(self.root))
            if admitted is None or str(relative) not in admitted['references']:
                return _ORIGINAL_CHECK(self,relative,digest)
            _reader_code_guard(storage)
            value=admitted['references'][str(relative)]
            if digest != value['sha256']:raise ValueError('External static operator SHA differs')
            path=core.guard(value);witness=(value['stat'][2],value['stat'][3],value['stat'][4],value['stat'][1])
            previous=self._witnesses.get((str(relative),digest))
            if previous is not None and previous != witness:raise ValueError('Static operator source witness differs')
            self._witnesses[(str(relative),digest)]=witness
            return path
        storage.LosslessRawBankStore._check=check
    return dict(base_proof,storage_extension=EXTENSION,readonly_static_operator_references=len(references),
        original_load_case_code_preserved=True,original_crop_code_preserved=True,original_np_load_code_preserved=True,
        static_operator_read_mode='r',static_operator_adapter_sha256=document['static_operator_adapter_source']['sha256'])


def install_runtime_adapters(admission_path,bank_root,private_data_folder=None):
    result=install_raw_store_adapter(admission_path,bank_root)
    result['read_mode']='r'
    if private_data_folder is not None:
        return dict(raw_store=result,preprocessed=core.install_dataset_adapter(admission_path,private_data_folder))
    return result


def verify_private_checkpoint_publication(admission,private_runtime,trainer_class=None):
    """Verify the actual private inherited methods before any model construction."""
    document=verify_admission(admission);publication=document['checkpoint_publication_proof']
    path=Path(private_runtime)/'nnunetv2'/CHECKPOINT_SOURCE
    if core.sha(core.regular(path)) != publication['source']['sha256']:
        raise ValueError('Private original native checkpoint source differs')
    if trainer_class is None:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        trainer_class=nnUNetTrainer
    expected=_compiled_methods(path);actual={}
    for name in CHECKPOINT_METHODS:
        method=getattr(trainer_class,name)
        bound=_CHECKPOINT_BINDINGS.get(trainer_class)
        if name=='on_train_end' and bound is not None:
            if method is not bound['wrapper'] or method.__code__ is not bound['wrapper_code']:
                raise ValueError('Coordinated original on_train_end wrapper changed')
            method=bound['original']
        if Path(inspect.getsourcefile(method)).resolve() != path.resolve():
            raise ValueError('Actual native checkpoint method must inherit the admitted private source')
        actual[name]=core.code_fingerprint(method.__code__)
    if actual != expected:raise ValueError('Actual private checkpoint compiled methods changed')
    admitted=_compiled_methods(path,filename=publication['source']['path'])
    if admitted!=publication['compiled_methods']:
        raise ValueError('Verified private checkpoint bytes differ at the admitted source filename')
    return dict(storage_extension=EXTENSION,private_checkpoint_source=core.proof(path),
        actual_private_compiled_methods=actual,expected_private_compiled_methods=expected,
        admitted_source_compiled_methods=admitted,code_fingerprint_format=core.CODE_FINGERPRINT,
        peak_checkpoint_slots=3,completed_checkpoint_slots=2,
        retention_changed=False,trainer_constructor_called=False)


def _lock_mode(path):
    return stat_module.S_IMODE(Path(path).stat().st_mode)


def _coordination_guard(document):
    value=document['checkpoint_coordination'];roots=document['aggregate_budget']['arms']
    from . import v24_nnunet_cp as pipeline
    relative=('native/nnUNet_results/'+document['baseline']['dataset_name']+'/'+pipeline.TRAINER+'__'
              +document['baseline']['plans_name']+'__'+document['baseline']['configuration']+'/fold_0')
    group=core.regular(value['group_root'],directory=True);lock=core.guard(value['lock_file'],full_hash=True)
    if (value.get('format')!=COORDINATION or value.get('mode')!=0o600
            or lock!=group/'native_final_publication.lock' or lock.stat().st_nlink!=1 or lock.stat().st_size!=0
            or _lock_mode(lock)!=0o600
            or value.get('global_peak_checkpoint_slots')!=7 or value.get('per_arm_peak_checkpoint_slots')!=3
            or value.get('optimizer_work_locked')is not False or value.get('retention_changed')is not False
            or value.get('original_final_before_latest_removal')is not True
            or value.get('failure_blocks_next_final')is not True or value.get('repeated_final_rejected')is not True
            or value.get('folds')!={arm:str(Path(row['root'])/relative) for arm,row in roots.items()}
            or any(Path(row['root']).parent!=group for row in roots.values())
            or value['adapter_source']!=document['static_operator_adapter_source']):
        raise ValueError('Explicit original native final-publication global7 coordination proof required')
    return lock


def install_checkpoint_adapters(admission,private_runtime,trainer_class):
    """Serialize only original on_train_end, including its original latest removal."""
    import fcntl
    document=verify_admission(admission);lock=_coordination_guard(document)
    proof=verify_private_checkpoint_publication(document,private_runtime,trainer_class)
    existing=_CHECKPOINT_BINDINGS.get(trainer_class)
    if existing is not None:
        if existing['coordination']!=document['checkpoint_coordination']:
            raise ValueError('Native trainer final-publication coordination already differs')
        return existing['proof']
    original=trainer_class.on_train_end;original_code=original.__code__
    untouched={name:getattr(trainer_class,name) for name in ('save_checkpoint','on_epoch_end')}
    if hasattr(trainer_class,'train_step'):untouched['train_step']=trainer_class.train_step
    untouched_codes={name:function.__code__ for name,function in untouched.items()}
    private_source=proof['private_checkpoint_source'];coordination=copy.deepcopy(document['checkpoint_coordination'])
    bound_fold=None;completed=False
    def guard():
        core.guard(private_source)
        _coordination_guard(document)
        if original.__code__ is not original_code or any(getattr(trainer_class,name)is not function
                or function.__code__ is not untouched_codes[name] for name,function in untouched.items()):
            raise ValueError('Original native checkpoint/optimizer method changed during coordination')
        if trainer_class.on_train_end is not on_train_end or on_train_end.__code__ is not wrapper_code:
            raise ValueError('Native final-publication wrapper changed')
    def on_train_end(self):
        nonlocal bound_fold,completed
        guard();folder=core.regular(Path(self.output_folder).absolute(),directory=True)
        matches=[arm for arm,path in coordination['folds'].items() if Path(path)==folder]
        if len(matches)!=1:raise ValueError('Native final publication must use its exact admitted own fold')
        descriptor=os.open(lock,os.O_RDWR|getattr(os,'O_NOFOLLOW',0))
        try:
            opened=os.fstat(descriptor)
            if (opened.st_dev,opened.st_ino)!=(lock.stat().st_dev,lock.stat().st_ino):
                raise ValueError('Native final-publication lock inode changed')
            with os.fdopen(descriptor,'rb',closefd=False) as stream:
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
                try:
                    guard()
                    if completed or result['successful_original_on_train_end_calls']!=0:
                        raise ValueError('Original native final publication cannot be repeated')
                    if bound_fold is not None and bound_fold!=str(folder):
                        raise ValueError('One native trainer class must retain its admitted own fold')
                    bound_fold=str(folder)
                    if (folder/'checkpoint_final.pth').is_symlink():
                        raise ValueError('Native checkpoint publication cannot use a symlink')
                    if (folder/'checkpoint_final.pth').exists():
                        raise ValueError('Fresh native final publication cannot be repeated')
                    for arm,path in coordination['folds'].items():
                        candidate=Path(path)
                        if candidate.is_symlink():raise ValueError('Native checkpoint fold cannot use a symlink')
                        if candidate.exists():
                            core.regular(candidate,directory=True)
                            for name in ('checkpoint_final.pth','checkpoint_latest.pth','checkpoint_best.pth'):
                                checkpoint=candidate/name
                                if checkpoint.is_symlink():raise ValueError('Native checkpoint publication cannot use a symlink')
                                if checkpoint.exists():core.regular(checkpoint)
                            if (candidate/'checkpoint_final.pth').exists() and (candidate/'checkpoint_latest.pth').exists():
                                core.regular(candidate/'checkpoint_final.pth');core.regular(candidate/'checkpoint_latest.pth')
                                raise RuntimeError('Previous original final publication failed before latest removal; global7 preserved: '+arm)
                    answer=original(self)
                    completed=True;result['successful_original_on_train_end_calls']=1
                    result['bound_native_fold']=bound_fold
                    return answer
                finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        finally:os.close(descriptor)
    wrapper_code=on_train_end.__code__;trainer_class.on_train_end=on_train_end
    result=dict(proof,coordination=COORDINATION,format=COORDINATION,active=True,coordination_active=True,
        successful_original_on_train_end_calls=0,bound_native_fold=None,
        lock_file=copy.deepcopy(coordination['lock_file']),
        original_on_train_end_code_preserved=True,original_on_train_end_called_once_under_flock=True,
        original_on_train_end_called_once_per_completion=True,
        optimizer_work_locked=False,save_checkpoint_unchanged=True,on_epoch_end_unchanged=True,
        global_peak_checkpoint_slots=7,per_arm_peak_checkpoint_slots=3,
        failure_blocks_next_final=True,repeated_final_rejected=True)
    _CHECKPOINT_BINDINGS[trainer_class]=dict(wrapper=on_train_end,wrapper_code=wrapper_code,
        original=original,coordination=coordination,proof=result)
    return result
