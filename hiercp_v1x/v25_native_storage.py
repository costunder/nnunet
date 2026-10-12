"""Four fresh V25 arms, immutable full131 inputs, and measured global9 writes.

This is a storage admission, not evidence of completed V25 training. Native
and GNN checkpoint publication retain the original functions and retention.
The four arms share static preprocessed arrays only; learned artifacts do not.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import functools
import json
import os
from pathlib import Path
import shutil
import time

from . import v24_readonly_native_storage as core
from . import v24_readonly_static_operator_storage as original_checkpoint
from .contracts import canonical_hash

FORMAT = 'v25_four_arm_full131_readonly_native_storage_v1'
PROFILE = 'v25_fresh_ABCD_native_full131_Blosc2_readonly_global9'
COORDINATION = 'v25_original_native_final_publication_global9_v1'
EXTENSION = PROFILE
ARMS = ('A', 'B', 'C', 'D')
NATIVE_CHECKPOINT_BYTES = 819237798
RESERVE = 2 * 2**30
GROWTH_MARGIN = 2**30
GNN_PROBE_FORMAT = 'v25_genuine_full_optimizer_checkpoint_byte_measurement_v1'
_DATASET_ADAPTER = None
_CHECKPOINT_BINDINGS = {}
_GNN_BINDINGS = {}


def _model_tensor_measurement(model,count):
    import torch
    from .v24_stunet import CHECKPOINT_SHA256,CHANNELS,DEPTHS,POOL_STRIDES
    extras={name:value for name,value in model.items() if not isinstance(value,torch.Tensor)}
    if set(extras)!={'local_encoder._extra_state'}:
        raise ValueError('Only original STU encoder contract extra_state is admitted')
    state=extras['local_encoder._extra_state']
    expected=dict(format='v24_stunet_small_L0_v1',checkpoint_sha256=CHECKPOINT_SHA256,
        encoder_channels=list(CHANNELS),encoder_depths=list(DEPTHS),pool_strides=[list(row) for row in POOL_STRIDES],
        pretrained=True,encoder_trainable=True,segmentation_decoder_constructed=False,
        local_message_passing=False,all_six_encoder_scales_used=True,recipient_tumor_GT_available=False)
    if not isinstance(state,dict) or any(state.get(key)!=value for key,value in expected.items()):
        raise ValueError('Original official full STU encoder metadata required')
    total=sum(value.numel() for value in model.values() if isinstance(value,torch.Tensor))
    if total<count:raise ValueError('Measured checkpoint cannot omit original model tensors')
    return dict(model_tensor_numel_including_buffers=total,STU_extra_state_sha256=canonical_hash(state))


def _positive(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError('Explicit measured integer bytes required: ' + name)
    return value


def measure_gnn_checkpoint(path):
    """CPU inspect a real full-STU model and populated AdamW checkpoint.

    An older complete-model measurement is labelled with its actual format;
    it is never labelled an actual V25 checkpoint or training completion.
    """
    import torch
    witness = core.proof(path)
    saved = torch.load(witness['path'], map_location='cpu', weights_only=False)
    model, optimizer = saved.get('model'), saved.get('optimizer')
    if not isinstance(model, dict) or not isinstance(optimizer, dict):
        raise ValueError('Genuine model and populated optimizer checkpoint required')
    states = optimizer.get('state', {})
    parameters = [item for group in optimizer.get('param_groups', []) for item in group['params']]
    if len(parameters) != 537 or len(set(parameters)) != 537 or set(parameters) != set(states):
        raise ValueError('All537 original full-STU trainable optimizer states required')
    count = 0
    for value in states.values():
        first, second, step = value.get('exp_avg'), value.get('exp_avg_sq'), value.get('step')
        if (not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor)
                or first.shape != second.shape or not torch.isfinite(first).all()
                or not torch.isfinite(second).all() or step is None
                or not torch.isfinite(torch.as_tensor(step)).all() or float(step) < 1):
            raise ValueError('Genuine finite populated AdamW moments required')
        count += first.numel()
    if count != 13331792:
        raise ValueError('Original13331792-parameter full-STU checkpoint required')
    model_metadata=_model_tensor_measurement(model,count)
    core.guard(witness, full_hash=True)
    return dict(format=GNN_PROBE_FORMAT, file=witness, actual_serialization=True,
        full_optimizer_state=True, optimizer_parameters=537, model_parameter_count=count,
        serialized_checkpoint_bytes=witness['bytes'], observed_training_format=saved.get('format'),
        actual_v25_checkpoint=saved.get('format') == 'v25_all_P_patient_balanced_shared_context_training_v1',
        training_completion_claim=False,**model_metadata)


def _gnn_probe(value, *, full_hash=False):
    if (value.get('format') != GNN_PROBE_FORMAT or value.get('actual_serialization') is not True
            or value.get('full_optimizer_state') is not True
            or value.get('optimizer_parameters') != 537
            or value.get('model_parameter_count') != 13331792
            or value.get('training_completion_claim') is not False
            or value.get('serialized_checkpoint_bytes') != value['file']['bytes']):
        raise ValueError('Genuine full optimizer serialization byte measurement required')
    _positive(value['serialized_checkpoint_bytes'], 'GNN checkpoint')
    core.guard(value['file'], full_hash=full_hash)
    return value


def _roots(group_root, arm_roots):
    group = core.regular(group_root, directory=True)
    if set(arm_roots) != set(ARMS):
        raise ValueError('Exactly four fresh A/B/C/D arm roots required')
    roots = {arm: str(core.regular(arm_roots[arm], directory=True)) for arm in ARMS}
    if len(set(roots.values())) != 4 or any(Path(path).parent != group for path in roots.values()):
        raise ValueError('Four distinct own immediate-child arm roots required')
    return group, roots


def _folds(document):
    from .v24_nnunet_cp import TRAINER
    baseline = document['baseline']
    relative = ('native/nnUNet_results/' + baseline['dataset_name'] + '/' + TRAINER + '__'
                + baseline['plans_name'] + '__' + baseline['configuration'] + '/fold_0')
    return {arm: str(Path(path)/relative) for arm, path in document['arm_roots'].items()}


def _components(document):
    private = sum(row['bytes'] for row in document['private_metadata_files'].values())
    private += sum(row['bytes'] for row in document['ground_truth_files'].values())
    custom = core.CUSTOM_TRAINERS
    private += sum(row['bytes'] for name, row in document['runtime_files'].items()
                   if Path(name).name not in custom)
    private += sum(row['bytes'] for row in document['custom_trainer_files'].values())
    return dict(four_private_metadata_GT_runtime_bytes=4*private,
        native_global9_checkpoint_bytes=9*max(row['bytes'] for row in document['native_checkpoint_samples']),
        gnn_global9_checkpoint_bytes=9*document['gnn_checkpoint_measurement']['serialized_checkpoint_bytes'],
        four_additional_writable_bytes=sum(document['additional_writable_bytes'].values()))


def build_admission(*, group_root, arm_roots, inventory_path, baseline_preprocessed,
                    runtime_package, native_checkpoint_paths, gnn_checkpoint_measurement,
                    additional_writable_bytes, workers=4, progress=None):
    """Hash all131 CT/seg/properties/GT and provision every four-arm output.

    additional_writable_bytes explicitly includes each fresh CP/score bank,
    prediction, metric, log and other persistent output budget. Canonical caches
    on /dev/shm need a separate measured tmpfs/RAM admission, not disk credit.
    """
    from .v24_nnunet_cp import validate_baseline, validate_split
    if workers != 4 or type(workers) is not int:
        raise ValueError('Exactly four assigned hash workers required')
    group, roots = _roots(group_root, arm_roots)
    if set(additional_writable_bytes) != set(ARMS):
        raise ValueError('Explicit output byte budget for each A/B/C/D arm required')
    additional = {arm: _positive(additional_writable_bytes[arm], arm+' outputs') for arm in ARMS}
    inventory = core.proof(inventory_path)
    split = validate_split(core.read(inventory_path)['split'])
    baseline = validate_baseline(baseline_preprocessed, split)
    pre = core.regular(baseline_preprocessed, directory=True)
    data = core.regular(pre/baseline['data_identifier'], directory=True)
    cases = sorted(split['outer_train']+split['outer_val'])
    names = {case+suffix for case in cases for suffix in ('.b2nd', '_seg.b2nd', '.pkl')}
    if {path.name for path in data.iterdir()} != names:
        raise ValueError('Exactly full131 Blosc2 CT/seg/properties required; no duplicate unpacked arrays')
    if {path.name for path in (pre/'gt_segmentations').iterdir()} != {case+'.nii.gz' for case in cases}:
        raise ValueError('Full131 original evaluator annotation files required')
    package = core.regular(runtime_package, directory=True)
    protected = (pre, package, Path(inventory['path']))
    if any(Path(root).is_relative_to(path) or path.is_relative_to(Path(root))
           for root in roots.values() for path in protected):
        raise ValueError('Four fresh results must be disjoint from immutable inputs/runtime')
    hashes = core._parallel_proofs([data/name for name in sorted(names)], workers=4,
                                   progress=progress, phase='V25_full131_readonly_SHA')
    files = {name: hashes[str(data/name)] for name in sorted(names)}
    for name, row in files.items():
        if name.endswith('.b2nd'):
            row.update(core._open_header(row['path']))
    metadata = {name: core.proof(pre/name) for name in baseline['source_files_sha256']}
    ground_truth = {case+'.nii.gz': core.proof(pre/'gt_segmentations'/(case+'.nii.gz')) for case in cases}
    runtime = {path.relative_to(package).as_posix(): core.proof(path) for path in package.rglob('*')
               if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    custom = {name: core.proof(core.ROOT/'custom_trainers'/name) for name in core.CUSTOM_TRAINERS}
    checkpoint_paths = [str(core.regular(path)) for path in native_checkpoint_paths]
    samples = [core.proof(path) for path in checkpoint_paths]
    if (len(samples) != 2 or {Path(row['path']).name for row in samples}
            != {'checkpoint_best.pth', 'checkpoint_final.pth'}
            or max(row['bytes'] for row in samples) < NATIVE_CHECKPOINT_BYTES):
        raise ValueError('Genuine complete BEST/final native checkpoint byte measurements required')
    for path in (group/'gnn_checkpoint_publication.lock', group/'native_final_publication.lock'):
        if path.exists() or path.is_symlink():
            raise FileExistsError('Fresh V25 publication coordination required: '+str(path))
    document = dict(format=FORMAT, profile=PROFILE, group_root=str(group), arm_roots=roots,
        inventory=inventory, split=split, baseline=baseline,
        preprocessed=dict(data_directory=str(data), cases=cases, files=files),
        private_metadata_files=metadata, ground_truth_files=ground_truth,
        runtime_package=str(package), runtime_files=runtime, custom_trainer_files=custom,
        native_checkpoint_samples=samples,
        gnn_checkpoint_measurement=copy.deepcopy(_gnn_probe(gnn_checkpoint_measurement, full_hash=True)),
        additional_writable_bytes=additional, adapter_source=core.proof(__file__),
        core_source=core.proof(core.__file__), checkpoint_core_source=core.proof(original_checkpoint.__file__),
        scores_reused=False, learned_upper_reused=False, source_arrays_written=False,
        model_data_scale_preserved=True, debug=False, admission_hash_workers=4)
    document['checkpoint_publication_proof'] = original_checkpoint._checkpoint_source_proof(
        document, checkpoint_paths, max(row['bytes'] for row in samples))
    components = _components(document)
    required = sum(components.values())+RESERVE+GROWTH_MARGIN
    available = shutil.disk_usage(group).free
    if available < required:
        raise OSError('All four complete V25 outputs/checkpoints plus2GiB reserve/1GiB growth do not fit')
    document['budget'] = dict(measured_components=components, new_writable_bytes_estimate=sum(components.values()),
        checkpoint_reserve_bytes=RESERVE, growth_margin_bytes=GROWTH_MARGIN,
        minimum_runtime_free_bytes=RESERVE+GROWTH_MARGIN, required_free_bytes=required,
        initial_available_free_bytes=available, native_checkpoint_slots=9, gnn_checkpoint_slots=9,
        canonical_tmpfs_included=False, readonly_dataset_copies=0)
    for name in ('gnn_checkpoint_publication.lock', 'native_final_publication.lock'):
        descriptor = os.open(group/name, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        os.close(descriptor)
    document['checkpoint_coordination'] = dict(format=COORDINATION, group_root=str(group),
        lock_file=core.proof(group/'native_final_publication.lock'),
        gnn_lock_file=core.proof(group/'gnn_checkpoint_publication.lock'),
        global_peak_checkpoint_slots=9, per_arm_peak_checkpoint_slots=3,
        folds=_folds(document), retention_changed=False, original_final_before_latest_removal=True,
        optimizer_work_locked=False, failure_blocks_next_final=True, repeated_final_rejected=True)
    return verify_admission(document)


def verify_admission(value, *, full_hash=False):
    from .v24_nnunet_cp import validate_baseline, validate_split
    document = core.read(value) if isinstance(value, (str, Path)) else value
    if not isinstance(document, dict) or document.get('format') != FORMAT or document.get('profile') != PROFILE:
        raise ValueError('Explicit V25 four-arm storage format required; V24 global7 is not admitted')
    group, roots = _roots(document['group_root'], document['arm_roots'])
    split = validate_split(document['split'])
    cases = sorted(split['outer_train']+split['outer_val'])
    pre = document['preprocessed']
    names = {case+suffix for case in cases for suffix in ('.b2nd', '_seg.b2nd', '.pkl')}
    if (pre['cases'] != cases or set(pre['files']) != names
            or set(document['ground_truth_files']) != {case+'.nii.gz' for case in cases}
            or set(document['additional_writable_bytes']) != set(ARMS)
            or document.get('scores_reused') is not False or document.get('learned_upper_reused') is not False
            or document.get('source_arrays_written') is not False or document.get('debug') is not False
            or document.get('model_data_scale_preserved') is not True):
        raise ValueError('Complete untouched full131/four-arm storage identity required')
    for arm in ARMS:
        _positive(document['additional_writable_bytes'][arm], arm+' outputs')
    _gnn_probe(document['gnn_checkpoint_measurement'], full_hash=full_hash)
    rows = [document['inventory'], document['adapter_source'], document['core_source'], document['checkpoint_core_source']]
    for key in ('private_metadata_files', 'ground_truth_files', 'runtime_files', 'custom_trainer_files'):
        rows.extend(document[key].values())
    rows.extend(pre['files'].values()); rows.extend(document['native_checkpoint_samples'])
    for row in rows:
        core.guard(row, full_hash=full_hash)
    if (document['adapter_source']['sha256'] != core.sha(__file__)
            or document['core_source']['sha256'] != core.sha(core.__file__)
            or document['checkpoint_core_source']['sha256'] != core.sha(original_checkpoint.__file__)
            or validate_baseline(document['baseline']['preprocessed'], split) != document['baseline']
            or {path.name for path in Path(pre['data_directory']).iterdir()} != names
            or any(Path(row['path']) != Path(pre['data_directory'])/name for name, row in pre['files'].items())):
        raise ValueError('Admitted immutable baseline/runtime/adapter binding changed')
    baseline_root = Path(document['baseline']['preprocessed'])
    if (core.read(document['inventory']['path'])['split'] != split
            or set(document['private_metadata_files']) != set(document['baseline']['source_files_sha256'])
            or set(document['custom_trainer_files']) != set(core.CUSTOM_TRAINERS)
            or Path(pre['data_directory']) != baseline_root/document['baseline']['data_identifier']
            or any(Path(row['path']) != baseline_root/name
                   or row['sha256'] != document['baseline']['source_files_sha256'][name]
                   for name, row in document['private_metadata_files'].items())
            or any(Path(row['path']) != baseline_root/'gt_segmentations'/name
                   for name, row in document['ground_truth_files'].items())
            or any(Path(row['path']) != Path(document['runtime_package'])/name
                   for name, row in document['runtime_files'].items())):
        raise ValueError('Full131 static files must bind the exact original split and source paths')
    samples = document['native_checkpoint_samples']
    if (len(samples) != 2 or max(row['bytes'] for row in samples) < NATIVE_CHECKPOINT_BYTES
            or {Path(row['path']).name for row in samples} != {'checkpoint_best.pth', 'checkpoint_final.pth'}):
        raise ValueError('Actual complete native checkpoint size witnesses changed')
    rebuilt = original_checkpoint._checkpoint_source_proof(document, [row['path'] for row in samples], max(row['bytes'] for row in samples))
    if rebuilt != document['checkpoint_publication_proof']:
        raise ValueError('Original native checkpoint publication code changed')
    budget, components = document['budget'], _components(document)
    if (budget['measured_components'] != components or budget['new_writable_bytes_estimate'] != sum(components.values())
            or budget['native_checkpoint_slots'] != 9 or budget['gnn_checkpoint_slots'] != 9
            or budget['checkpoint_reserve_bytes'] != RESERVE or budget['growth_margin_bytes'] != GROWTH_MARGIN
            or budget['minimum_runtime_free_bytes'] != RESERVE+GROWTH_MARGIN
            or budget['required_free_bytes'] != sum(components.values())+RESERVE+GROWTH_MARGIN
            or budget['initial_available_free_bytes'] < budget['required_free_bytes']
            or budget['canonical_tmpfs_included'] is not False or budget['readonly_dataset_copies'] != 0):
        raise ValueError('Measured four-arm global9 write/reserve budget changed')
    coordination = document['checkpoint_coordination']
    expected = dict(format=COORDINATION, group_root=str(group), lock_file=coordination['lock_file'],
        gnn_lock_file=coordination['gnn_lock_file'], global_peak_checkpoint_slots=9, per_arm_peak_checkpoint_slots=3,
        folds=_folds(document), retention_changed=False, original_final_before_latest_removal=True,
        optimizer_work_locked=False, failure_blocks_next_final=True, repeated_final_rejected=True)
    if coordination != expected:
        raise ValueError('Exact V25 global9 publication coordination required')
    for name, key in (('native_final_publication.lock', 'lock_file'), ('gnn_checkpoint_publication.lock', 'gnn_lock_file')):
        path = core.guard(coordination[key], full_hash=True)
        if path != group/name or path.stat().st_nlink != 1 or path.stat().st_size != 0:
            raise ValueError('Fresh single-link V25 publication lock required')
    return document


def seal_admission(path, document):
    document = verify_admission(document, full_hash=True)
    path = Path(path).absolute()
    if path.parent != Path(document['group_root']):
        raise ValueError('Admission belongs directly in its fresh four-arm group')
    core.publish_json(path, document)
    return core.proof(path)


def _allocated(roots):
    total, seen = 0, set()
    for root in roots:
        for path in Path(root).rglob('*'):
            if path.is_symlink():
                raise ValueError('Reserved own result trees cannot contain symlinks')
            if not path.is_file():
                continue
            core.regular(path)
            value = path.stat(); identity = (value.st_dev, value.st_ino)
            if value.st_nlink != 1:
                raise ValueError('Reserved result tree cannot credit unaccounted hardlinks')
            if identity not in seen:
                seen.add(identity); total += getattr(value, 'st_blocks', (value.st_size+511)//512)*512
    return total


def check_aggregate_disk(admission, armroot, *, preparation=True):
    document = verify_admission(admission)
    root = core.regular(armroot, directory=True)
    if str(root) not in document['arm_roots'].values():
        raise ValueError('Storage checks must bind one exact reserved arm')
    written = _allocated(document['arm_roots'].values())
    remaining = max(0, document['budget']['new_writable_bytes_estimate']-written)
    required = remaining+RESERVE+GROWTH_MARGIN
    free = shutil.disk_usage(root).free
    if free < required:
        raise OSError('Remaining complete four-arm writes plus2GiB reserve/1GiB growth do not fit')
    return dict(actual_free_bytes=free, required_free_bytes=required, observed_own_output_bytes=written,
                remaining_projected_output_bytes=remaining, arm_root=str(root), profile=PROFILE)


@contextmanager
def _publication_lock(path):
    """Linux flock serializes only publication, preserving original computation."""
    import fcntl
    path = core.regular(path)
    descriptor = os.open(path, os.O_RDWR|getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = path.stat(); actual = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (actual.st_dev, actual.st_ino):
            raise ValueError('Publication lock inode changed')
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _released(actor):
    """A reused PID cannot resurrect its old exact process identity."""
    import psutil
    if (type(actor.get('pid')) is not int or actor['pid'] <= 0
            or type(actor.get('create_time')) not in (int, float)
            or actor['create_time'] <= 0 or not isinstance(actor.get('command'), list)
            or not actor['command'] or any(not isinstance(item, str) for item in actor['command'])):
        raise ValueError('Exact previous actor PID/birth/full argv required')
    try:
        process = psutil.Process(actor['pid'])
        if process.create_time() != actor['create_time']:
            return True
        uid = process.uids().real if hasattr(process, 'uids') else None
        if uid != actor['UID'] or process.cmdline() != actor['command']:
            raise ValueError('Previous same-birth actor UID/argv changed; release is unproven')
        return process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _recovery_predecessor(document, arm, lease, *, predecessor_child, predecessor_status):
    uid = os.getuid() if hasattr(os, 'getuid') else None
    if (lease.get('format') != 'v25_exclusive_downstream_wave_v1'
            or lease.get('arm') != arm or lease.get('arm_root') != document['arm_roots'][arm]
            or lease.get('admission_sha256') != canonical_hash(document) or lease.get('UID') != uid):
        raise ValueError('Only the exact released same-arm admission lease can be recovered')
    if not _released(lease):
        raise RuntimeError('Previous exact native wave actor is still alive')
    if predecessor_child is not None:
        proof = core.proof(predecessor_child); child = core.read(predecessor_child)
        identity = {key: lease[key] for key in ('pid', 'create_time', 'command', 'UID')}
        if (child.get('parent_actor') != identity or child.get('UID') != uid
                or child.get('state') not in ('RUNNING', 'EXITED')
                or not isinstance(lease.get('request_sha256'), str)
                or child.get('request_sha256') != lease['request_sha256']):
            raise ValueError('Previous native child must bind the exact old parent and immutable request')
        if not _released(child):
            raise RuntimeError('Previous exact native child remains alive; orphan recovery refused')
        core.guard(proof, full_hash=True)
        return proof
    if predecessor_status is None:
        raise ValueError('Previous child absence proof or exact pre-child status is required')
    proof = core.proof(predecessor_status); status = core.read(predecessor_status)
    identity = {key: lease[key] for key in ('pid', 'create_time', 'command', 'UID')}
    expected_status = Path(document['arm_roots'][arm])/'downstream/status.json'
    if (Path(proof['path']) != expected_status
            or (expected_status.parent/'native_child.json').exists()
            or status.get('format') != 'v25_full_own_arm_downstream_status_v1'
            or status.get('actor') != identity or status.get('arm') != arm
            or status.get('request_sha256') != lease.get('request_sha256')
            or status.get('child_actor') is not None
            or status.get('stage') not in ('admission', 'prepare-native')
            or status.get('state') not in ('RUNNING', 'READY', 'FAILED', 'WAITING_FOR_NATIVE_STORAGE')):
        raise ValueError('Exact old status must prove no native subprocess was started')
    core.guard(proof, full_hash=True)
    return proof


@contextmanager
def sequential_wave_lock(admission, arm, *, request_sha256=None,
                         recover_released_same_arm=False, predecessor_child=None,
                         predecessor_status=None):
    document = verify_admission(admission)
    if arm not in ARMS:
        raise ValueError('One explicit V25 arm required')
    import psutil
    actor = psutil.Process()
    path = Path(document['group_root'])/'sequential_native_wave.lease.json'
    value = dict(format='v25_exclusive_downstream_wave_v1', arm=arm, pid=actor.pid,
        create_time=actor.create_time(), command=actor.cmdline(), UID=os.getuid() if hasattr(os, 'getuid') else None,
        admission_sha256=canonical_hash(document), arm_root=document['arm_roots'][arm])
    if request_sha256 is not None:
        if (not isinstance(request_sha256, str) or len(request_sha256) != 64
                or any(item not in '0123456789abcdef' for item in request_sha256)):
            raise ValueError('Exact immutable downstream request SHA256 required')
        value['request_sha256'] = request_sha256
    lock = document['checkpoint_coordination']['lock_file']['path']
    with _publication_lock(lock):
        if path.exists() or path.is_symlink():
            if recover_released_same_arm is not True:
                raise FileExistsError('Existing native wave lease preserved; explicit same-arm recovery required')
            old_proof = core.proof(path); old = core.read(path)
            child_proof = _recovery_predecessor(document, arm, old,
                predecessor_child=predecessor_child, predecessor_status=predecessor_status)
            archive = path.with_name('sequential_native_wave_released_'+str(time.time_ns())+'.json')
            core.publish_json(archive, dict(original_lease=old, original_file=old_proof,
                predecessor_release=child_proof, verified_released_at=time.time(),
                recovery_arm=arm, signals_sent=False))
            core.guard(old_proof, full_hash=True)
            path.unlink()
        core.publish_json(path, value)
    witness = core.proof(path)
    try:
        yield copy.deepcopy(witness)
    except BaseException:
        # An uncertain failed child must not permit the next native arm.
        raise
    else:
        with _publication_lock(lock):
            core.guard(witness, full_hash=True)
            receipt = path.with_name('sequential_native_wave_finished_'+str(time.time_ns())+'.json')
            core.publish_json(receipt, dict(lease=value, completed_at=time.time()))
            path.unlink()


def install_dataset_adapter(admission, private_data_folder, **kwargs):
    global _DATASET_ADAPTER
    if _DATASET_ADAPTER is None:
        _DATASET_ADAPTER = core._clone(core.install_dataset_adapter,
            dict(_admission=verify_admission, PROFILE=PROFILE, _DATASET_BINDINGS={}, _DATASET_ORIGINALS=None))
    return _DATASET_ADAPTER(admission, private_data_folder, **kwargs)


def verify_private_checkpoint_publication(admission, private_runtime, trainer_class=None):
    adapted = core._clone(original_checkpoint.verify_private_checkpoint_publication,
        dict(verify_admission=verify_admission, EXTENSION=EXTENSION, _CHECKPOINT_BINDINGS=_CHECKPOINT_BINDINGS))
    return adapted(admission, private_runtime, trainer_class)


def install_checkpoint_adapters(admission, private_runtime, trainer_class):
    """Retain original on_train_end; coordinate only its final/latest overlap."""
    document = verify_admission(admission)
    proof = verify_private_checkpoint_publication(document, private_runtime, trainer_class)
    if trainer_class in _CHECKPOINT_BINDINGS:
        bound = _CHECKPOINT_BINDINGS[trainer_class]
        if bound['document_hash'] != canonical_hash(document):
            raise ValueError('Native class already belongs to another admission')
        return bound['proof']
    original = trainer_class.on_train_end
    original_code = original.__code__
    untouched = {name: getattr(trainer_class, name) for name in ('save_checkpoint', 'on_epoch_end')}
    if hasattr(trainer_class, 'train_step'):
        untouched['train_step'] = trainer_class.train_step
    untouched_codes = {name: method.__code__ for name, method in untouched.items()}
    completed = False
    @functools.wraps(original)
    def on_train_end(self):
        nonlocal completed
        verify_admission(document)
        folder = core.regular(Path(self.output_folder).absolute(), directory=True)
        if str(folder) not in document['checkpoint_coordination']['folds'].values() or completed:
            raise ValueError('Original native final must publish once to its own admitted fold')
        if (original.__code__ is not original_code
                or trainer_class.on_train_end is not on_train_end
                or any(getattr(trainer_class, name) is not method or method.__code__ is not untouched_codes[name]
                       for name, method in untouched.items())):
            raise ValueError('Original native checkpoint/optimizer methods changed')
        with _publication_lock(document['checkpoint_coordination']['lock_file']['path']):
            for root in document['checkpoint_coordination']['folds'].values():
                candidate = Path(root)
                if candidate.exists() or candidate.is_symlink():
                    core.regular(candidate, directory=True)
                    for name in ('checkpoint_best.pth', 'checkpoint_latest.pth', 'checkpoint_final.pth'):
                        target = candidate/name
                        if target.exists() or target.is_symlink():
                            core.regular(target)
                    if (candidate/'checkpoint_final.pth').exists() and (candidate/'checkpoint_latest.pth').exists():
                        raise RuntimeError('Previous native final failed before latest removal; all records preserved')
            if (folder/'checkpoint_final.pth').exists():
                raise FileExistsError('Existing own native final is preserved')
            answer = original(self)
            completed = True
            return answer
    result = dict(proof, format=COORDINATION, global_peak_checkpoint_slots=9,
        original_on_train_end_code_preserved=True, save_checkpoint_unchanged=True,
        on_epoch_end_unchanged=True, optimizer_work_locked=False, retention_changed=False)
    trainer_class.on_train_end = on_train_end
    _CHECKPOINT_BINDINGS[trainer_class] = dict(wrapper=on_train_end, wrapper_code=on_train_end.__code__,
        original=original, document_hash=canonical_hash(document), proof=result)
    return result


def install_gnn_checkpoint_adapter(training_module, admission, *, arm):
    """Serialize the unchanged atomic_save call, retaining8 plus one temp slot."""
    document = verify_admission(admission)
    if arm not in ARMS:
        raise ValueError('Explicit current V25 arm required for GNN publication')
    if training_module in _GNN_BINDINGS:
        if (_GNN_BINDINGS[training_module]['document_hash'] != canonical_hash(document)
                or _GNN_BINDINGS[training_module]['arm'] != arm):
            raise ValueError('GNN publication module already bound differently')
        return _GNN_BINDINGS[training_module]['proof']
    from .u_bridge_training import atomic_save
    original = training_module.atomic_save
    if original is not atomic_save:
        raise ValueError('Only original full checkpoint atomic_save can be coordinated')
    @functools.wraps(original)
    def save(path, payload):
        verify_admission(document)
        path = Path(path).absolute()
        parents = [Path(root)/'training' for root in document['arm_roots'].values()]
        if path.parent != Path(document['arm_roots'][arm])/'training' or path.name not in ('checkpoint_best.pt', 'checkpoint_latest.pt'):
            raise ValueError('GNN checkpoint must bind its own reserved full training namespace')
        with _publication_lock(document['checkpoint_coordination']['gnn_lock_file']['path']):
            for parent in parents:
                if parent.exists() and any(parent.glob('checkpoint_*.pt.*.tmp')):
                    raise RuntimeError('Failed original GNN publication temp preserved; explicit recovery required')
            check_aggregate_disk(document, path.parent.parent)
            answer = original(path, payload)
            if path.stat().st_size > document['gnn_checkpoint_measurement']['serialized_checkpoint_bytes']+GROWTH_MARGIN//9:
                raise OSError('Actual full V25 checkpoint exceeds measured sample plus explicit growth allowance')
            return answer
    training_module.atomic_save = save
    result = dict(global_peak_checkpoint_slots=9, original_atomic_save_code_preserved=True,
                  retention_changed=False, optimizer_work_locked=False)
    _GNN_BINDINGS[training_module] = dict(document_hash=canonical_hash(document), arm=arm, proof=result)
    return result
