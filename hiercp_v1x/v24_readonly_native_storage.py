"""Explicit full native storage profile; only static volumes are shared read-only.

The full-copy native profile stays unchanged. This profile verifies all 105 raw
CT/segmentation pairs and all 131 preprocessed cases, publishes fresh private
metadata/candidates/scores, and preserves the original preparation and reader
code. Shared files are never linked, unpacked, replaced, or opened for writing.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import hashlib
import inspect
import json
import marshal
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import textwrap
import types
import uuid

import numpy as np

FORMAT='v24_native_immutable_full_cache_storage_v1'
PROFILE='reuse_full105_static_raw_CTseg_and_full131_Blosc2_readonly'
RECEIPT='readonly_storage.json'
ROOT=Path(__file__).resolve().parents[1]
RESERVE=10*2**30
ROLES=('baseline_unclipped','baseline_seg')
SCIENCE_FILES=('custom_trainers/onlinecp_raw_bank.py','custom_trainers/onlinecp_raw_resampling.py',
    'tools/online_raw_bank_preparation.py','hiercp_v1x/v24_lossless_raw_storage.py')
CUSTOM_TRAINERS=('nnUNetTrainer_OnlinePairedCP.py','onlinecp_raw_bank.py','onlinecp_raw_resampling.py','nnUNetTrainer_FrozenV23CP.py')
_STORE_BINDINGS={}
_STORE_ORIGINAL=None
_DATASET_BINDINGS={}
_DATASET_ORIGINALS=None


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):result.update(block)
    return result.hexdigest()


def stat(path):
    value=Path(path).stat()
    return [value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns]


def regular(path,*,directory=False):
    path=Path(path)
    if not path.is_absolute() or path.resolve(strict=True)!=path.absolute():
        raise ValueError('Exact absolute native storage path required: '+str(path))
    for parent in (path,*path.parents):
        if parent.is_symlink():raise ValueError('Native shared/private storage cannot use symlinks: '+str(parent))
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError('Regular native storage artifact required: '+str(path))
    if hasattr(os,'getuid') and path.stat().st_uid!=os.getuid():
        raise ValueError('Only own native cache artifacts are admitted: '+str(path))
    return path


def read(path):return json.loads(regular(path).read_text(encoding='utf8'))


def proof(path):
    path=regular(path);before=stat(path);checksum=sha(path)
    if stat(path)!=before:raise ValueError('Native storage source changed during admission: '+str(path))
    return dict(path=str(path),sha256=checksum,stat=before,bytes=before[2])


def _parallel_proofs(paths,*,workers,progress,phase):
    """Hash independent complete files on the four explicitly assigned CPUs."""
    paths=list(dict.fromkeys(str(Path(path).absolute()) for path in paths));result={}
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='native_readonly_SHA') as executor:
        futures={executor.submit(proof,path):path for path in paths}
        for future in as_completed(futures):
            value=future.result();result[value['path']]=value
            if progress is not None:progress(dict(phase=phase,completed=len(result),total=len(paths),path=value['path']))
    return result


def guard(value,*,full_hash=False):
    path=regular(value['path'])
    if stat(path)!=value['stat'] or path.stat().st_size!=value['bytes']:
        raise ValueError('Admitted native read-only source changed: '+str(path))
    if full_hash and sha(path)!=value['sha256']:
        raise ValueError('Admitted native read-only content changed: '+str(path))
    if stat(path)!=value['stat']:raise ValueError('Native source changed while checking: '+str(path))
    return path


def publish(path,content):
    """Atomic exclusive publication, never replacement of existing results."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.readonly_'+uuid.uuid4().hex+'.tmp');created=False
    try:
        with temporary.open('xb') as stream:
            created=True;stream.write(content);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:
        if created:temporary.unlink()


def publish_json(path,value):
    publish(path,json.dumps(value,indent=2,allow_nan=False).encode('utf8'))


def copy_private(value,target):
    source=guard(value,full_hash=True);target=Path(target)
    if target.exists() or target.is_symlink():raise FileExistsError('Fresh private native file required: '+str(target))
    publish(target,source.read_bytes())
    if sha(target)!=value['sha256'] or stat(source)!=value['stat']:
        raise ValueError('Private native metadata/GT copy differs: '+str(target))


def _relative(root,name):
    text=str(name);parts=PurePosixPath(text).parts
    if (not parts or '\\' in text or ':' in text or text.startswith('/')
            or any(part in ('.','..') for part in text.split('/'))):
        raise ValueError('Exact contained native cache path required: '+text)
    path=Path(root).joinpath(*parts)
    if not path.absolute().is_relative_to(Path(root).absolute()):raise ValueError('Native cache path escapes root')
    return path


def _case_ids(split):
    from .v24_nnunet_cp import validate_split
    split=validate_split(split)
    return set(split['outer_train']),set(split['outer_train'])|set(split['outer_val'])


def _open_header(path):
    import blosc2
    array=blosc2.open(str(path),mode='r',dparams={'nthreads':1})
    return dict(shape=list(array.shape),dtype=array.dtype.str)


def build_admission(*,source_bank,inventory_path,baseline_preprocessed,additional_writable_bytes,
        runtime_package=None,workers=4,progress=None):
    """Hash the complete authoritative cache and measure new writable assets.

    additional_writable_bytes is a measured/provisioned private geometry budget,
    never a sample/node/data cap. It must include the newly built zero-P upper
    cache and any other fresh scoring artifacts not present in the donor bank.
    """
    from . import v24_nnunet_cp as pipeline
    from . import v24_lossless_raw_storage as storage
    if type(additional_writable_bytes)is not int or additional_writable_bytes<0:
        raise ValueError('Explicit nonnegative measured private geometry/scoring write budget required')
    if type(workers)is not int or workers!=4:
        raise ValueError('This full native profile requires the four assigned hash/I/O workers')
    root=regular(source_bank,directory=True);inventory_proof=proof(inventory_path);meta=read(inventory_path)
    bank=pipeline.validate_bank(read(root/'index.json'));request=read(root/'request.json')
    expected,all_cases=_case_ids(meta['split'])
    baseline=pipeline.validate_baseline(baseline_preprocessed,meta['split'])
    if (bank['split']!=meta['split'] or bank['baseline']!=baseline or request['inventory_sha256']!=inventory_proof['sha256']
            or set(bank['entries_by_case'])!=expected or len(meta['raw_records'])!=131
            or {row['case_id'] for row in meta['raw_records']}!=all_cases):
        raise ValueError('Cache and authoritative full105/131 inventory/plans must coincide')
    scientific={name:sha(ROOT/name) for name in SCIENCE_FILES}
    if any(bank['source_identity'].get(name)!=checksum for name,checksum in scientific.items()):
        raise ValueError('Static cache original raw preparation/resampling source differs')
    manifests={path.stem:path for path in (root/'raw_cases').glob('*.json')}
    if set(manifests)!=expected:raise ValueError('Exactly full105 regular raw manifests required')
    raw={row['case_id']:row for row in meta['raw_records']};cases={};raw_source_files={};reused_bytes=0
    hash_paths=[raw[case][role] for case in sorted(expected) for role in ('image','label')]
    for case,path in manifests.items():
        document=read(path);hash_paths.append(path)
        for role in ROLES:
            hash_paths.append(_relative(root,document['arrays'][document['tree'][role]['$array']]['path']))
    hashed=_parallel_proofs(hash_paths,workers=workers,progress=progress,phase='full105_raw_source_and_CTseg_SHA')
    def raw_proof(path):
        value=hashed[str(Path(path).absolute())];guard(value);return copy.deepcopy(value)
    for case_id in sorted(expected):
        original=raw[case_id]
        for role in ('image','label'):
            row=raw_proof(original[role])
            if row['sha256']!=original[role+'_sha256']:raise ValueError('Authoritative raw source hash differs: '+case_id+'/'+role)
            raw_source_files[case_id+'/'+role]=row
        manifest_path=manifests[case_id];document=read(manifest_path)
        lossless=pipeline.validate_lossless_case_receipt(document)
        if bank['CP_audits'][case_id]['lossless_storage']!=lossless:
            raise ValueError('Completed source bank static case proof differs from its published manifest')
        if (document.get('v24_storage')!=storage.FORMAT or document.get('kind')!='case'
                or document['tree']['metadata']['case_id']!=case_id or document['compression']!=storage.COMPRESSION):
            raise ValueError('Complete independent static raw case cache required: '+case_id)
        volumes={}
        for role,dtype in ((ROLES[0],np.dtype('float64')),(ROLES[1],np.dtype('int16'))):
            key=document['tree'][role]['$array'];spec=copy.deepcopy(document['arrays'][key])
            if (np.dtype(spec['dtype'])!=dtype or len(spec['shape'])!=4 or spec['shape'][0]!=1
                    or spec.get('storage')!=storage.FORMAT or spec['compression']!=storage.COMPRESSION
                    or spec['roundtrip']['all_voxels_verified']is not True
                    or spec['roundtrip']['verified_voxels']!=int(np.prod(spec['shape']))
                    or spec['roundtrip']['source_tile_bytes_sha256']!=spec['roundtrip']['decoded_tile_bytes_sha256']):
                raise ValueError('Complete exact64-bit CT/16-bit segmentation voxel proof required')
            value=raw_proof(_relative(root,spec['path']))
            if value['sha256']!=spec['sha256'] or value['bytes']!=spec['stored_bytes'] or _open_header(value['path'])!={k:spec[k] for k in ('shape','dtype')}:
                raise ValueError('Actual static raw volume SHA/header differs: '+case_id+'/'+role)
            volumes[role]=dict(spec=spec,file=value);reused_bytes+=value['bytes']
        if volumes[ROLES[0]]['spec']['shape']!=volumes[ROLES[1]]['spec']['shape']:
            raise ValueError('Raw cached CT/segmentation grids differ')
        cases[case_id]=dict(manifest=raw_proof(manifest_path),metadata=copy.deepcopy(document['tree']['metadata']),
            volumes=volumes,lossless_storage=copy.deepcopy(lossless))
    pre=regular(baseline_preprocessed,directory=True);data=regular(pre/baseline['data_identifier'],directory=True)
    expected_files={case+suffix for case in all_cases for suffix in ('.b2nd','_seg.b2nd','.pkl')}
    actual_files={path.name for path in data.iterdir()}
    if actual_files!=expected_files:raise ValueError('Exactly full131 Blosc2 CT/seg/property inventory required; no unpacked copies')
    pre_hashes=_parallel_proofs([data/name for name in sorted(expected_files)],workers=workers,progress=progress,
        phase='full131_preprocessed_Blosc2_properties_SHA')
    files={}
    for name in sorted(expected_files):
        value=copy.deepcopy(pre_hashes[str((data/name).absolute())]);guard(value)
        if name.endswith('.b2nd'):value.update(_open_header(data/name))
        files[name]=value;reused_bytes+=value['bytes']
    metadata={name:proof(pre/name) for name in baseline['source_files_sha256']}
    if any(row['sha256']!=baseline['source_files_sha256'][name] for name,row in metadata.items()):
        raise ValueError('Native metadata/plans changed during storage admission')
    gt=regular(pre/'gt_segmentations',directory=True)
    if {path.name for path in gt.iterdir()}!={case+'.nii.gz' for case in all_cases}:
        raise ValueError('Complete original131 GT evaluator annotations required')
    ground_truth={path.name:proof(path) for path in sorted(gt.iterdir())}
    package=runtime_package
    if package is None:
        import nnunetv2
        package=Path(nnunetv2.__file__).resolve().parent
    package=regular(package,directory=True)
    runtime_files={p.relative_to(package).as_posix():proof(p) for p in package.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    custom_trainers={name:proof(ROOT/'custom_trainers'/name) for name in CUSTOM_TRAINERS}
    volume_paths={Path(row['file']['path']) for case in cases.values() for row in case['volumes'].values()}
    # All measured prior small bank assets are conservatively provisioned fresh;
    # none of its scores, selected candidates or learned upper states are reused.
    small_bank_bytes=sum(regular(p).stat().st_size for p in root.rglob('*') if p.is_file() and p not in volume_paths)
    components=dict(private_metadata_bytes=sum(row['bytes'] for row in metadata.values()),
        private_full131_GT_bytes=sum(row['bytes'] for row in ground_truth.values()),
        private_runtime_package_bytes=sum(row['bytes'] for name,row in runtime_files.items()
            if PurePosixPath(name).name not in CUSTOM_TRAINERS)+sum(row['bytes'] for row in custom_trainers.values()),
        private_small_bank_artifacts_bytes=small_bank_bytes,explicit_additional_fresh_outputs_bytes=additional_writable_bytes)
    writable=sum(components.values())
    document=dict(format=FORMAT,profile=PROFILE,inventory=inventory_proof,split=copy.deepcopy(meta['split']),baseline=baseline,
        source_bank=dict(root=str(root),index=proof(root/'index.json'),request=proof(root/'request.json'),
            scientific_source_files_sha256=scientific),raw_source_files=raw_source_files,raw_cases=cases,
        preprocessed=dict(data_directory=str(data),cases=sorted(all_cases),files=files),
        private_metadata_files=metadata,ground_truth_files=ground_truth,runtime_package=str(package),runtime_files=runtime_files,
        custom_trainer_files=custom_trainers,
        budget=dict(new_writable_bytes_estimate=writable,private_geometry_scoring_bytes=additional_writable_bytes,
            measured_components=components,measured_private_small_bank_bytes=small_bank_bytes,checkpoint_reserve_bytes=RESERVE,
            required_free_bytes=writable+RESERVE,minimum_runtime_free_bytes=RESERVE,reused_readonly_bytes=reused_bytes),
        adapter_source=proof(Path(__file__).resolve()),admission_hash_workers=workers,model_data_scale_preserved=True,debug=False,
        scores_reused=False,learned_upper_reused=False,source_arrays_written=False,created_at=time.time())
    verify_admission(document)
    return document


def verify_admission(document,*,full_hash=False):
    """Check exact full inventories and persistent file witnesses before use."""
    expected,all_cases=_case_ids(document['split']);budget=document['budget']
    if (document.get('format')!=FORMAT or document.get('profile')!=PROFILE
            or document.get('model_data_scale_preserved')is not True or document.get('debug')is not False
            or document.get('scores_reused')is not False or document.get('learned_upper_reused')is not False
            or document.get('source_arrays_written')is not False or set(document['raw_cases'])!=expected
            or set(document['preprocessed']['cases'])!=all_cases
            or set(document['raw_source_files'])!={case+'/'+role for case in expected for role in ('image','label')}
            or set(document['ground_truth_files'])!={case+'.nii.gz' for case in all_cases}
            or set(document['preprocessed']['files'])!={case+suffix for case in all_cases for suffix in ('.b2nd','_seg.b2nd','.pkl')}
            or budget['checkpoint_reserve_bytes']!=RESERVE or budget['minimum_runtime_free_bytes']!=RESERVE
            or type(budget['new_writable_bytes_estimate'])is not int or budget['new_writable_bytes_estimate']<0
            or set(document['source_bank']['scientific_source_files_sha256'])!=set(SCIENCE_FILES)
            or set(document['custom_trainer_files'])!=set(CUSTOM_TRAINERS)
            or budget['new_writable_bytes_estimate']!=sum(budget['measured_components'].values())
            or budget['required_free_bytes']!=budget['new_writable_bytes_estimate']+RESERVE):
        raise ValueError('Explicit complete105/131 read-only native profile and measured reserve required')
    rows=[document['inventory'],document['source_bank']['index'],document['source_bank']['request'],document['adapter_source']]
    rows.extend(document['raw_source_files'].values());rows.extend(document['preprocessed']['files'].values())
    rows.extend(document['private_metadata_files'].values());rows.extend(document['ground_truth_files'].values())
    rows.extend(document['runtime_files'].values())
    rows.extend(document['custom_trainer_files'].values())
    for case in document['raw_cases'].values():
        if set(case['volumes'])!=set(ROLES):raise ValueError('Only both static CT/seg volume roles may be reused')
        rows.append(case['manifest'])
        for role,dtype in ((ROLES[0],np.dtype('float64')),(ROLES[1],np.dtype('int16'))):
            row=case['volumes'][role];spec=row['spec']
            if np.dtype(spec['dtype'])!=dtype or spec['sha256']!=row['file']['sha256']:
                raise ValueError('Original exact static CT/seg dtype/SHA required')
            rows.append(row['file'])
    for row in rows:guard(row,full_hash=full_hash)
    if any(sha(ROOT/name)!=value for name,value in document['source_bank']['scientific_source_files_sha256'].items()):
        raise ValueError('Read-only cache scientific preparation source changed')
    if sha(__file__)!=document['adapter_source']['sha256']:raise ValueError('Explicit read-only storage adapter source changed')
    return document


def _admission(value):
    return verify_admission(read(value) if isinstance(value,(str,Path)) else value)


def check_disk(path,document,*,preparation=False):
    document=_admission(document);free=shutil.disk_usage(path).free
    required=document['budget']['required_free_bytes'] if preparation else document['budget']['minimum_runtime_free_bytes']
    if free<required:raise OSError('Exact read-only native profile writable artifacts/checkpoint reserve do not fit')
    return dict(actual_free_bytes=free,required_free_bytes=required,profile=PROFILE)


def _clone(function,replacements):
    result=types.FunctionType(function.__code__,dict(function.__globals__,**replacements),function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=copy.deepcopy(function.__kwdefaults__)
    return result


def make_case_preparer(pipeline,admission):
    """Retain full scientific preparation; change only private array publication."""
    from . import v24_lossless_raw_storage as storage
    document=_admission(admission);original=pipeline._root_raw_preparation()
    def save_case(bank_root,relative_manifest,case):
        case_id=case['metadata']['case_id'];cached=document['raw_cases'][case_id]
        if case['metadata']!=cached['metadata']:raise ValueError('New native static case metadata differs from cache')
        value={key:item for key,item in case.items() if key!='preparation'};arrays={}
        tree=storage.original._encode_tree(value,arrays)
        roles={tree[role]['$array']:role for role in ROLES};manifest=dict(format=storage.original.STORAGE_FORMAT,
            kind='case',tree=tree,arrays={},v24_storage=storage.FORMAT,compression=dict(storage.COMPRESSION),
            readonly_storage_profile=PROFILE,readonly_source_volumes={})
        for key,array in arrays.items():
            if key in roles:
                role=roles[key];row=cached['volumes'][role];guard(row['file']);spec=copy.deepcopy(row['spec'])
                if array.dtype.str!=spec['dtype'] or list(array.shape)!=spec['shape']:
                    raise ValueError('New original full-precision baseline grid differs')
                # Traverse every freshly prepared voxel against the authoritative
                # existing cache; no subset, approximation, or dtype conversion.
                verified=storage._verify_all(array,row['file']['path'],spec)
                if verified!=spec['roundtrip']:raise ValueError('New raw baseline/cache full-voxel proof differs')
                relative=str(PurePosixPath(relative_manifest).with_suffix(''))+'.'+key+'.b2nd'
                spec['path']=relative;manifest['arrays'][key]=spec
                manifest['readonly_source_volumes'][relative]=dict(case_id=case_id,role=role,source=copy.deepcopy(row['file']))
            else:
                relative=str(PurePosixPath(relative_manifest).with_suffix(''))+'.'+key+'.npy'
                target=storage.original._relative_path(bank_root,relative,must_exist=False)
                checksum=storage.original._publish(target,lambda stream,a=array:np.save(stream,a,allow_pickle=False))
                manifest['arrays'][key]=dict(shape=list(array.shape),dtype=array.dtype.str,path=relative,sha256=checksum)
        manifest['lossless_storage']=copy.deepcopy(cached['lossless_storage'])
        path=storage.original._relative_path(bank_root,relative_manifest,must_exist=False)
        return storage.original._publish(path,lambda stream:stream.write((json.dumps(manifest,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode('utf8')))
    inner=_clone(original._prepare_raw_case,dict(save_case=save_case))
    return _clone(original.prepare_raw_case,dict(_prepare_raw_case=inner))


def install_pipeline_adapter(pipeline,admission_path):
    """Preserve original full105 scoring/donor/materialization implementation."""
    document=_admission(admission_path);original=pipeline._materialize_bank
    publisher=original.__globals__['new_json']
    def new_json(path,value):
        if Path(path).name=='index.json':
            value=copy.deepcopy(value);value.update(storage_profile=PROFILE,storage_admission=str(Path(admission_path).resolve()),
                storage_admission_sha256=sha(admission_path),static_raw_volume_reuse_only=True)
            verify_admission(document)
        return publisher(path,value)
    def preparer():return make_case_preparer(pipeline,document)
    adapted=_clone(original,dict(lossless_raw_case_preparer=preparer,new_json=new_json))
    pipeline._materialize_bank=adapted
    return dict(profile=PROFILE,original_materializer_code_preserved=adapted.__code__ is original.__code__,
        globals_replaced=['lossless_raw_case_preparer','new_json'],scores_reused=False,learned_upper_reused=False)


def prepare_native(bank_path,output,admission_path,pipeline):
    """Fresh runtime/plans/GT/output; all131 immutable preprocessed arrays shared."""
    pipeline.require_project_budget()
    document=_admission(admission_path);bank_path=regular(Path(bank_path).resolve());bank=pipeline.validate_bank(read(bank_path))
    if (bank.get('storage_profile')!=PROFILE or bank.get('storage_admission_sha256')!=sha(admission_path)
            or bank.get('storage_admission')!=str(Path(admission_path).resolve()) or bank['baseline']!=document['baseline']
            or bank['split']!=document['split']):raise ValueError('Fresh own-arm bank must bind explicit read-only storage admission')
    output=Path(output).absolute()
    if output.exists():raise FileExistsError('Fresh private native output required')
    for protected in (Path(document['source_bank']['root']),Path(document['baseline']['preprocessed']),bank_path.parent):
        if output.is_relative_to(protected) or protected.is_relative_to(output):raise ValueError('Private native output must be disjoint from inputs/bank')
    regular(output.parent,directory=True);check_disk(output.parent,document,preparation=True)
    output.mkdir();private=output/'runtime/nnunetv2';private.mkdir(parents=True)
    custom=CUSTOM_TRAINERS
    for name,value in document['runtime_files'].items():
        if PurePosixPath(name).name in custom:continue
        copy_private(value,private/name)
    trainers=private/'training/nnUNetTrainer';trainers.mkdir(parents=True,exist_ok=True)
    for name in custom:copy_private(document['custom_trainer_files'][name],trainers/name)
    pre=output/'nnUNet_preprocessed'/bank['baseline']['dataset_name'];pre.mkdir(parents=True)
    data=pre/bank['baseline']['data_identifier'];data.mkdir()
    copied={}
    for name,value in document['private_metadata_files'].items():
        copy_private(value,pre/name);copied[name]=value['sha256']
    for name,value in document['ground_truth_files'].items():
        relative='gt_segmentations/'+name;copy_private(value,pre/relative);copied[relative]=value['sha256']
    # The original baseline validator needs this real private directory; the
    # explicit dataset adapter supplies all131 arrays from its admitted source.
    if pipeline.validate_baseline(pre,bank['split'])['source_files_sha256']!=bank['baseline']['source_files_sha256']:
        raise ValueError('Fresh private native metadata differs from original comparison')
    value=dict(format=pipeline.pin_format(bank['pin']),root=str(output),bank=str(bank_path),bank_sha256=sha(bank_path),
        baseline=bank['baseline'],private_runtime=str(private.parent),trainer=pipeline.TRAINER,
        native_preprocessed_source_written=False,model_weights_fresh=True,preprocessed_storage=PROFILE,
        preprocessing_copy_bytes=sum(regular(pre/name).stat().st_size for name in copied),copied_native_array_sha256=copied,
        private_trainer_sha256={name:sha(trainers/name) for name in custom},pretrained_segmentation_used=False,
        prediction_inputs='outer26 CT images only',source_files_sha256={name:sha(pipeline.ROOT/name) for name in pipeline.FILES},
        resource_contract=pipeline.resource_contract(),storage_profile=PROFILE,storage_admission=str(Path(admission_path).resolve()),
        storage_admission_sha256=sha(admission_path),readonly_native_array_inventory=copy.deepcopy(document['preprocessed']),
        readonly_source_files_written=False,read_mode='r',unpack_policy='original nnUNetDatasetBlosc2 no-op')
    if bank['pin'].get('format')==pipeline.CURRENT_PIN_FORMAT:value['physical_GPU']=bank['physical_GPU']
    publish_json(output/'native.json',value);verify_admission(document);pipeline.require_project_budget()
    return output/'native.json'


def install_raw_store_adapter(admission,bank_root):
    """Keep the exact crop/store classes; admit only role-specific source refs."""
    global _STORE_ORIGINAL
    from . import v24_lossless_raw_storage as storage
    document=_admission(admission);root=regular(Path(bank_root).resolve(),directory=True);key=str(root)
    expected=set(document['raw_cases']);manifests={path.stem:path for path in (root/'raw_cases').glob('*.json')}
    if set(manifests)!=expected:raise ValueError('Fresh complete own-bank105 manifests required for reader binding')
    references={}
    for case_id,path in manifests.items():
        manifest=read(path)
        if manifest.get('readonly_storage_profile')!=PROFILE:raise ValueError('Explicit raw case read-only reference profile required')
        values=manifest.get('readonly_source_volumes',{})
        if len(values)!=2 or {value.get('role') for value in values.values()}!=set(ROLES):
            raise ValueError('Only static complete CT/seg references allowed')
        for relative,value in values.items():
            role=value['role'];cached=document['raw_cases'][case_id]['volumes'].get(role)
            array_key=manifest['tree'][role]['$array'];spec=manifest['arrays'][array_key]
            original_spec=None if cached is None else dict(cached['spec'],path=relative)
            if (cached is None or value['case_id']!=case_id or value['source']!=cached['file']
                    or relative!='raw_cases/'+case_id+'.'+array_key+'.b2nd' or spec!=original_spec
                    or (root/relative).exists()):raise ValueError('Role-specific immutable source mapping differs')
            references[relative]=copy.deepcopy(cached['file'])
    binding=dict(document=document,references=references)
    if key in _STORE_BINDINGS and _STORE_BINDINGS[key]!=binding:raise ValueError('Read-only store root already bound differently')
    _STORE_BINDINGS[key]=binding
    if _STORE_ORIGINAL is None:
        _STORE_ORIGINAL=storage.LosslessRawBankStore._check
        def check(self,relative,digest):
            admitted=_STORE_BINDINGS.get(str(self.root))
            if admitted is None or str(relative) not in admitted['references']:
                return _STORE_ORIGINAL(self,relative,digest)
            value=admitted['references'][str(relative)]
            if digest!=value['sha256']:raise ValueError('External static volume hash differs')
            path=guard(value);witness=(value['stat'][2],value['stat'][3],value['stat'][4],value['stat'][1])
            previous=self._witnesses.get((str(relative),digest))
            if previous is not None and previous!=witness:raise ValueError('External source witness differs')
            self._witnesses[(str(relative),digest)]=witness
            return path
        storage.LosslessRawBankStore._check=check
    return dict(profile=PROFILE,exact_original_store_class=True,readonly_role_references=len(references))


def _prove_blosc_reader(cls):
    tree=ast.parse(textwrap.dedent(inspect.getsource(cls.load_case)));opens=[node for node in ast.walk(tree)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='open']
    if len(opens)!=3 or any(next((kw.value for kw in call.keywords if kw.arg=='mode'),None).__class__ is not ast.Constant
            or next(kw.value.value for kw in call.keywords if kw.arg=='mode')!='r' for call in opens):
        raise ValueError('Original complete nnUNetDatasetBlosc2 reader must open all array roles in mode r')
    unpack=ast.parse(textwrap.dedent(inspect.getsource(cls.unpack_dataset)));functions=[n for n in ast.walk(unpack) if isinstance(n,ast.FunctionDef)]
    body=[] if len(functions)!=1 else [node for node in functions[0].body
        if not (isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str))]
    if len(functions)!=1 or len(body)!=1 or not isinstance(body[0],ast.Pass):
        raise ValueError('Original Blosc2 unpack_dataset must be the established no-op')


def _dataset_source_proof(cls,module,document,folder):
    """Bind the actual private class and compiled methods to admitted source bytes."""
    name='training/dataloading/nnunet_dataset.py'
    admitted=document['runtime_files'].get(name)
    path=folder.parents[2]/'runtime/nnunetv2'/name
    if (admitted is None or cls.__name__!='nnUNetDatasetBlosc2'
            or cls.__module__!=module.__name__
            or Path(module.__file__).resolve()!=path.resolve(strict=True)
            or Path(inspect.getsourcefile(cls.load_case)).resolve()!=path.resolve(strict=True)
            or sha(regular(path))!=admitted['sha256']):
        raise ValueError('Exact private dataset class/source must match admitted runtime SHA')
    guard(admitted)
    compiled=compile(path.read_text(encoding='utf8'),str(path),'exec',dont_inherit=True)
    class_code=[code for code in compiled.co_consts if isinstance(code,types.CodeType) and code.co_name=='nnUNetDatasetBlosc2']
    if len(class_code)!=1:raise ValueError('Exactly one original compiled nnUNetDatasetBlosc2 class required')
    hashes={}
    for name in ('__init__','load_case','unpack_dataset','get_identifiers'):
        function=getattr(cls,name);actual=function.__code__
        owner=function.__qualname__.rsplit('.',1)[0].split('.')[-1]
        owners=[code for code in compiled.co_consts if isinstance(code,types.CodeType) and code.co_name==owner]
        codes={} if len(owners)!=1 else {code.co_name:code for code in owners[0].co_consts if isinstance(code,types.CodeType)}
        if name not in codes or marshal.dumps(actual)!=marshal.dumps(codes[name]):
            raise ValueError('Actual private dataset compiled method differs from original source: '+name)
        hashes[name]=hashlib.sha256(marshal.dumps(actual)).hexdigest()
    return dict(private_dataset_source=str(path),original_dataset_source_sha256=admitted['sha256'],
        private_dataset_file=proof(path),original_compiled_function_sha256=hashes,
        actual_private_class_and_bytecode_verified=True)


def install_dataset_adapter(admission,private_data_folder,*,dataset_module=None,trainer_module=None):
    """Preserve the actual Blosc2 class/reader; redirect only this private folder."""
    global _DATASET_ORIGINALS
    document=_admission(admission);folder=regular(Path(private_data_folder).resolve(),directory=True)
    if any(folder.iterdir()):raise ValueError('Private read-only data directory must contain no duplicate/unpacked arrays')
    binding=dict(document=document,source=regular(document['preprocessed']['data_directory'],directory=True))
    if str(folder) in _DATASET_BINDINGS and _DATASET_BINDINGS[str(folder)]!=binding:
        raise ValueError('Private dataset folder already has a different source')
    _DATASET_BINDINGS[str(folder)]=binding
    if dataset_module is None:
        from nnunetv2.training.dataloading import nnunet_dataset as dataset_module
    if trainer_module is None:
        from nnunetv2.training.nnUNetTrainer import nnUNetTrainer as trainer_module
    cls=dataset_module.nnUNetDatasetBlosc2
    if _DATASET_ORIGINALS is None:
        _prove_blosc_reader(cls)
        source_proof=_dataset_source_proof(cls,dataset_module,document,folder)
        originals=dict(cls=cls,init=cls.__init__,load=cls.load_case,unpack=cls.unpack_dataset,
            identifiers=cls.get_identifiers,infer=dataset_module.infer_dataset_class)
        originals['source_proof']=source_proof
        _DATASET_ORIGINALS=originals
        def dataset_guard():
            guard(source_proof['private_dataset_file'])
            for role,name in (('init','__init__'),('load','load_case'),('unpack','unpack_dataset'),('identifiers','get_identifiers')):
                if hashlib.sha256(marshal.dumps(originals[role].__code__)).hexdigest()!=source_proof['original_compiled_function_sha256'][name]:
                    raise ValueError('Admitted original native reader bytecode changed: '+name)
        def init(self,folder,identifiers=None,folder_with_segs_from_previous_stage=None):
            bound=_DATASET_BINDINGS.get(str(Path(folder).resolve()))
            if bound is None:return originals['init'](self,folder,identifiers,folder_with_segs_from_previous_stage)
            dataset_guard()
            if folder_with_segs_from_previous_stage is not None:raise ValueError('Unrequested previous-stage data cannot enter this fullres profile')
            allowed=set(bound['document']['preprocessed']['cases'])
            if identifiers is not None and (len(set(identifiers))!=len(identifiers) or not set(identifiers)<=allowed):
                raise ValueError('Native identifiers are outside authoritative full131 cache')
            self._v24_readonly_binding=bound
            return originals['init'](self,str(bound['source']),identifiers,None)
        def load(self,identifier):
            bound=getattr(self,'_v24_readonly_binding',None)
            if bound is not None:
                dataset_guard()
                for suffix in ('.b2nd','_seg.b2nd','.pkl'):
                    guard(bound['document']['preprocessed']['files'][identifier+suffix])
            return originals['load'](self,identifier)
        def unpack(folder,overwrite_existing=False,num_processes=1,verify=True):
            bound=_DATASET_BINDINGS.get(str(Path(folder).resolve()))
            if bound is None:return originals['unpack'](folder,overwrite_existing,num_processes,verify)
            dataset_guard()
            for row in bound['document']['preprocessed']['files'].values():guard(row)
            return None
        def infer(folder):
            if str(Path(folder).resolve()) in _DATASET_BINDINGS:return cls
            return originals['infer'](folder)
        def identifiers(folder):
            bound=_DATASET_BINDINGS.get(str(Path(folder).resolve()))
            if bound is None:return originals['identifiers'](folder)
            dataset_guard()
            values=originals['identifiers'](str(bound['source']))
            if len(values)!=131 or set(values)!=set(bound['document']['preprocessed']['cases']):
                raise ValueError('Original read-only dataset enumeration must retain all131 cases')
            return values
        cls.__init__=init;cls.load_case=load;cls.unpack_dataset=staticmethod(unpack)
        cls.get_identifiers=staticmethod(identifiers)
        dataset_module.infer_dataset_class=infer
    elif _DATASET_ORIGINALS['cls']is not cls:raise ValueError('Different native dataset class already installed')
    trainer_module.infer_dataset_class=dataset_module.infer_dataset_class
    return dict(profile=PROFILE,original_dataset_class=cls.__name__,original_load_case_code_preserved=True,
        readonly_mode='r',unpack_noop=True,full_preprocessed_cases=131,**_DATASET_ORIGINALS['source_proof'])


def install_runtime_adapters(admission_path,bank_root,private_data_folder):
    document=_admission(admission_path)
    return dict(raw_store=install_raw_store_adapter(document,bank_root),
        preprocessed=install_dataset_adapter(document,private_data_folder))
