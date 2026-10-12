"""Execution-only single GPU4 native namespace on physical GPU0.

All immutable 105 raw/131 preprocessed witnesses and Basic642 input bindings
come from the previously admitted experiment. Only output reservations and the
final-publication lock are rebound. No volume is rehashed or materialized.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import time

FORMAT = 'v24_GPU4_GPU0_single_arm_storage_rebinding_v1'
COORDINATION = 'v24_original_single_arm_native_final_publication_global3_v1'
KEY = 'gpu0_storage_rebinding'
_ORIGINALS = None


def _modules():
    from . import v24_readonly_native_storage as core
    from . import v24_readonly_static_operator_storage as static
    from . import v24_matched_basic_cp_pipeline as matched
    return core, static, matched


def _positive(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError('Explicit nonnegative integer byte reservation required: '+name)
    return value


def _root(path):
    core, _, _ = _modules()
    raw = Path(path)
    if not raw.is_absolute() or '..' in raw.parts or any(p.is_symlink() for p in (raw, *raw.parents)):
        raise ValueError('Own absolute output path without symlink components required')
    return core.regular(raw, directory=True)


def _canonical_root(path):
    # /dev/shm itself is a root-owned shared mount; only our new output
    # directories require the experiment UID. Never reinterpret another path
    # as this explicit original v2.5 canonical tmpfs reservation.
    raw = Path(path)
    if str(raw) != '/dev/shm' or raw.is_symlink() or not raw.is_dir():
        raise ValueError('Original explicit /dev/shm canonical reservation required')
    return raw


def _fold(old, armroot):
    from . import v24_nnunet_cp as pipeline
    base = old['baseline']
    return str(Path(armroot)/'native/nnUNet_results'/base['dataset_name']/(
        pipeline.TRAINER+'__'+base['plans_name']+'__'+base['configuration'])/'fold_0')


def _coordination(old, binding):
    return dict(format=COORDINATION, group_root=binding['execution_root'],
        lock_file=copy.deepcopy(binding['lock_file']), mode=0o600,
        global_peak_checkpoint_slots=3, per_arm_peak_checkpoint_slots=3,
        original_final_before_latest_removal=True, optimizer_work_locked=False,
        retention_changed=False, failure_blocks_next_final=True,
        repeated_final_rejected=True, folds={'gpu4':_fold(old, binding['arm_root'])},
        adapter_source=copy.deepcopy(binding['adapter_source']))


def _derive(old, binding):
    document = copy.deepcopy(old)
    writable = old['budget']['new_writable_bytes_estimate']
    floor = old['aggregate_budget']['minimum_runtime_free_bytes']
    document['aggregate_budget'] = dict(arms={'gpu4':copy.deepcopy(binding['arm_witness'])},
        per_arm_new_writable_bytes={'gpu4':writable},
        initial_allocated_bytes={'gpu4':binding['initial_arm_allocated_bytes']},
        single_arm_new_writable_bytes=writable, global_peak_checkpoint_slots=3,
        minimum_runtime_free_bytes=floor,
        growth_margin_bytes=old['aggregate_budget']['growth_margin_bytes'],
        required_free_bytes=writable+floor,
        initial_available_free_bytes=binding['initial_execution_free_bytes'],
        initial_admitted_at=binding['created_at'])
    document['checkpoint_coordination'] = _coordination(old, binding)
    document[KEY] = copy.deepcopy(binding)
    return document


def _old(binding):
    core, _, matched = _modules()
    core.guard(binding['old_matched_admission'], full_hash=True)
    old_matched = matched.verify_admission(binding['old_matched_admission']['path'], full_hash=False)
    core.guard(binding['old_original_admission'], full_hash=True)
    if old_matched['original_readonly_admission'] != binding['old_original_admission']:
        raise ValueError('Fresh storage must bind the exact prior matched GPU4 admission')
    old = _ORIGINALS['verify'](binding['old_original_admission']['path'], full_hash=False)
    return old_matched, old


def _validate_binding(binding, old):
    core, static, _ = _modules()
    if (binding.get('format') != FORMAT or binding.get('source_arm') != 4
            or binding.get('execution_physical_gpu') != 0
            or binding.get('immutable_input_verification') != 'original_SHA_witnesses_and_fresh_stat_no_volume_rehash'
            or binding.get('full_preprocessed_cases') != 131 or binding.get('full_training_cases') != 105
            or binding.get('full_validation_cases') != 26 or binding.get('source_entries') != 642
            or binding.get('rescore') is not False or binding.get('source_arrays_written') is not False
            or binding.get('all_other_arms_reserved_by_this_receipt') is not False):
        raise ValueError('Explicit GPU4-only immutable full131/Basic642 rebinding required')
    execution = _root(binding['execution_root'])
    metadata = _root(binding['metadata_root'])
    arm = _root(binding['arm_root'])
    if arm != execution/'gpu4':
        raise ValueError('Only the fresh immediate-child gpu4 output root may be admitted')
    protected = [Path(old['source_bank']['root']), Path(old['baseline']['preprocessed']),
        Path(old['inventory']['path']), Path(old['runtime_package'])]
    protected += [Path(row['root']) for row in old['aggregate_budget']['arms'].values()]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for root in (execution, metadata) for p in protected):
        raise ValueError('New output namespaces must be disjoint from original inputs/results/runtime')
    witness = binding['arm_witness']
    if (witness != dict(root=str(arm), existing_parent=str(execution), filesystem_device=arm.stat().st_dev)
            or execution.stat().st_dev != arm.stat().st_dev
            or binding['metadata_filesystem_device'] != metadata.stat().st_dev):
        raise ValueError('Fresh writable filesystem identity changed')
    core.guard(binding['adapter_source'], full_hash=True)
    if core.sha(__file__) != binding['adapter_source']['sha256']:
        raise ValueError('Single-arm execution storage source changed')
    _positive(binding['initial_arm_allocated_bytes'], 'initial arm bytes', zero=True)
    _positive(binding['initial_metadata_allocated_bytes'], 'initial metadata bytes', zero=True)
    _positive(binding['persistent_archive_bytes'], 'persistent BEST/26 predictions')
    _positive(binding['canonical_tmpfs_reserve_bytes'], 'canonical tmpfs reservation')
    if binding['canonical_tmpfs_reserve_bytes'] != 32*2**30:
        raise ValueError('Original v2.5 canonical tmpfs32GiB reservation must be preserved')
    _canonical_root(binding['canonical_tmpfs_path'])
    if binding['initial_execution_free_bytes'] < old['budget']['new_writable_bytes_estimate']+old['aggregate_budget']['minimum_runtime_free_bytes']:
        raise OSError('Full unchanged single-arm native provision plus10GiB/growth was not admitted')
    lock = core.guard(binding['lock_file'], full_hash=True)
    if (lock != execution/'native_final_publication.lock' or lock.stat().st_nlink != 1
            or lock.stat().st_size != 0 or static._lock_mode(lock) != 0o600):
        raise ValueError('Fresh own empty0600 single-link native publication lock required')


def verify_admission(value, *, full_hash=False):
    core, _, _ = _modules()
    document = core.read(value) if isinstance(value, (str, Path)) else value
    if KEY not in document:
        return _ORIGINALS['verify'](value, full_hash=full_hash)
    binding = document[KEY]
    _, old = _old(binding)
    _validate_binding(binding, old)
    if document != _derive(old, binding):
        raise ValueError('Rebinding may change only fresh output budget/coordination; immutable science differs')
    return document


def _coordination_guard(document):
    if KEY not in document:
        return _ORIGINALS['coordination'](document)
    core, _, _ = _modules()
    binding = document[KEY]
    # verify_admission already checked every input. Guard the small lock/source
    # on every final-publication call without traversing the static caches.
    core.guard(binding['adapter_source'], full_hash=True)
    old_baseline = {'baseline':document['baseline']}
    if document['checkpoint_coordination'] != _coordination(old_baseline, binding):
        raise ValueError('Single-arm original final-publication coordination changed')
    lock = core.guard(binding['lock_file'], full_hash=True)
    _, static, _ = _modules()
    if (lock != _root(binding['execution_root'])/'native_final_publication.lock'
            or lock.stat().st_nlink != 1 or lock.stat().st_size != 0 or static._lock_mode(lock) != 0o600):
        raise ValueError('Single-arm final-publication lock changed')
    return lock


def _v25_reservation(binding):
    core, static, _ = _modules()
    from . import v25_native_storage as v25
    proof = binding['v25_storage_admission']
    core.guard(proof, full_hash=True)
    document = v25.verify_admission(proof['path'], full_hash=False)
    roots = [_root(root) for root in document['arm_roots'].values()]
    devices = {root.stat().st_dev for root in roots}
    if len(devices) != 1:
        raise ValueError('Original four-arm v2.5 writable filesystem must remain one device')
    if any(root.is_relative_to(other) or other.is_relative_to(root)
           for root in roots for other in (_root(binding['execution_root']), _root(binding['metadata_root']))):
        raise ValueError('GPU0 output reservation cannot overlap any live v2.5 arm')
    written = sum(static._allocated_bytes(root) for root in roots)
    remaining = max(0, document['budget']['new_writable_bytes_estimate']-written)
    return dict(filesystem_device=next(iter(devices)), observed_output_bytes=written,
        remaining_output_bytes=remaining, reserve_bytes=document['budget']['minimum_runtime_free_bytes'],
        required_free_bytes=remaining+document['budget']['minimum_runtime_free_bytes'],
        admission=copy.deepcopy(proof))


def _disk_receipt(document):
    _, static, _ = _modules()
    binding = document[KEY]
    arm = _root(binding['arm_root']); metadata = _root(binding['metadata_root'])
    own_written = max(0, static._allocated_bytes(arm)-binding['initial_arm_allocated_bytes'])
    provision = document['aggregate_budget']['single_arm_new_writable_bytes']
    if own_written > provision:
        raise OSError('Actual single-arm native writes exceed its unchanged measured full provision')
    archive_written = max(0, static._allocated_bytes(metadata)-binding['initial_metadata_allocated_bytes'])
    archive_remaining = max(0, binding['persistent_archive_bytes']-archive_written)
    v25 = _v25_reservation(binding)
    requirements = {}
    def add(path, bytes_required, label, *, canonical=False):
        root = _canonical_root(path) if canonical else _root(path)
        device = root.stat().st_dev
        row = requirements.setdefault(device, dict(path=str(root), required_free_bytes=0, components={}))
        row['required_free_bytes'] += bytes_required
        row['components'][label] = bytes_required
    add(arm, provision-own_written+document['aggregate_budget']['minimum_runtime_free_bytes'], 'GPU4_full_native_remaining_and_reserve')
    add(metadata, archive_remaining, 'persistent_BEST_and_26_prediction_archive_remaining')
    add(binding['canonical_tmpfs_path'], binding['canonical_tmpfs_reserve_bytes'], 'V25_canonical_tmpfs_reservation', canonical=True)
    # Charge only the filesystem actually used by the four v2.5 output roots.
    for device, row in requirements.items():
        if device == v25['filesystem_device']:
            row['required_free_bytes'] += v25['required_free_bytes']
            row['components']['V25_remaining_writes_and_reserve'] = v25['required_free_bytes']
        row['actual_free_bytes'] = shutil.disk_usage(row['path']).free
        if row['actual_free_bytes'] < row['required_free_bytes']:
            raise OSError('GPU0 plus unchanged v2.5 complete output reservations do not fit: '+str(row))
    return dict(format=FORMAT, source_arm=4, execution_physical_gpu=0,
        own_native_output_bytes=own_written, own_native_remaining_bytes=provision-own_written,
        persistent_archive_remaining_bytes=archive_remaining, V25=v25,
        filesystems={str(key):row for key,row in requirements.items()},
        immutable_input_files_scanned_for_allocation=False, checked_at=time.time())


def check_aggregate_disk(admission, chain_root, *, preparation=True):
    core, _, _ = _modules()
    document = core.read(admission) if isinstance(admission, (str, Path)) else admission
    if KEY not in document:
        return _ORIGINALS['disk'](admission, chain_root, preparation=preparation)
    document = verify_admission(document)
    if Path(chain_root).absolute() != Path(document[KEY]['arm_root']):
        raise ValueError('Storage checks require exact admitted GPU4 own native root')
    return _disk_receipt(document)


def check_persistent_archive(request, required_bytes):
    """Check all actual archive bytes against unchanged future v2.5 writes."""
    _positive(required_bytes, 'actual complete BEST/26 archive bytes')
    core, _, matched = _modules()
    admission = request['storage_admission']
    if isinstance(admission, dict):
        if set(admission) == {'path', 'sha256'}:
            # Execution requests seal paths/content only. Capture the full
            # current owned-file stat proof before passing it to core.guard.
            witness = core.proof(core.regular(admission['path']))
            if witness['sha256'] != admission['sha256']:
                raise ValueError('Execution request storage admission SHA changed')
            admission = core.guard(witness, full_hash=True)
        elif set(admission) == {'path', 'sha256', 'stat', 'bytes'}:
            admission = core.guard(admission, full_hash=True)
        else:
            raise ValueError('Exact path/SHA or complete storage admission witness required')
    document = matched.verify_admission(admission, full_hash=False)
    original = verify_admission(core.guard(document['original_readonly_admission'], full_hash=True))
    binding = original[KEY]
    metadata = _root(binding['metadata_root'])
    if (Path(request['metadata_root']).absolute() != metadata
            or Path(request['execution_root']).absolute() != Path(binding['arm_root'])):
        raise ValueError('Archive must bind its exact admitted GPU4 tmpfs and persistent namespaces')
    v25 = _v25_reservation(binding)
    # Actual bytes may exceed the estimate: never omit predictions or clip the
    # result. Re-admit the entire archive with the exact measured required size.
    required = required_bytes
    if metadata.stat().st_dev == v25['filesystem_device']:
        required += v25['required_free_bytes']
    free = shutil.disk_usage(metadata).free
    if free < required:
        raise OSError('Entire actual BEST/26 archive plus v2.5 remaining outputs do not fit')
    return dict(format=FORMAT, actual_archive_bytes=required_bytes,
        actual_free_bytes=free, required_free_bytes=required, V25=v25,
        archive_truncated=False, checked_at=time.time())


def _checkpoint_adapters(admission, private_runtime, trainer_class):
    result = _ORIGINALS['checkpoint'](admission, private_runtime, trainer_class)
    core, _, _ = _modules()
    document = core.read(admission) if isinstance(admission, (str, Path)) else admission
    if KEY in document:
        # The original save/on_epoch_end/on_train_end code and flock wrapper are
        # reused. Only its truthful single active arm peak is reported here.
        result.update(coordination=COORDINATION, format=COORDINATION, global_peak_checkpoint_slots=3)
    return result


def install_runtime():
    """Install after selecting the exact historical source package path."""
    global _ORIGINALS
    _, static, _ = _modules()
    if _ORIGINALS is None:
        _ORIGINALS = dict(verify=static.verify_admission, disk=static.check_aggregate_disk,
            coordination=static._coordination_guard, checkpoint=static.install_checkpoint_adapters)
        static.verify_admission = verify_admission
        static.check_aggregate_disk = check_aggregate_disk
        static._coordination_guard = _coordination_guard
        static.install_checkpoint_adapters = _checkpoint_adapters
    elif static.verify_admission is not verify_admission or static.check_aggregate_disk is not check_aggregate_disk:
        raise ValueError('Single-arm storage execution installation changed')
    return dict(format=FORMAT, original_science_files_changed=False,
        immutable_input_materialization=False, all_other_arms_runtime_changed=False)


def build_admission(old_matched_admission, *, group_root=None, execution_root=None,
                    metadata_root=None, v25_admission, canonical_tmpfs_path='/dev/shm',
                    canonical_tmpfs_reserve_bytes=32*2**30, persistent_archive_bytes=2**30):
    """Publish fresh receipts only; caller supplies new own empty namespaces."""
    install_runtime()
    core, static, matched = _modules()
    execution = _root(execution_root if execution_root is not None else group_root)
    metadata = _root(metadata_root if metadata_root is not None else execution)
    arm = _root(execution/'gpu4')
    if any(arm.iterdir()):
        raise FileExistsError('Single-arm native admission requires a fresh empty gpu4 namespace')
    old_proof = core.proof(old_matched_admission)
    old_matched = matched.verify_admission(old_proof['path'], full_hash=False)
    old = _ORIGINALS['verify'](old_matched['original_readonly_admission']['path'], full_hash=False)
    lock = execution/'native_final_publication.lock'
    descriptor = os.open(lock, os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os, 'O_NOFOLLOW', 0), 0o600)
    os.close(descriptor)
    binding = dict(format=FORMAT, source_arm=4, execution_physical_gpu=0,
        old_matched_admission=old_proof, old_original_admission=copy.deepcopy(old_matched['original_readonly_admission']),
        execution_root=str(execution), metadata_root=str(metadata), arm_root=str(arm),
        arm_witness=dict(root=str(arm), existing_parent=str(execution), filesystem_device=arm.stat().st_dev),
        metadata_filesystem_device=metadata.stat().st_dev,
        initial_arm_allocated_bytes=static._allocated_bytes(arm),
        initial_metadata_allocated_bytes=static._allocated_bytes(metadata),
        initial_execution_free_bytes=shutil.disk_usage(execution).free,
        v25_storage_admission=core.proof(v25_admission), canonical_tmpfs_path=str(_canonical_root(canonical_tmpfs_path)),
        canonical_tmpfs_reserve_bytes=canonical_tmpfs_reserve_bytes,
        persistent_archive_bytes=persistent_archive_bytes,
        lock_file=core.proof(lock), adapter_source=core.proof(__file__), created_at=time.time(),
        immutable_input_verification='original_SHA_witnesses_and_fresh_stat_no_volume_rehash',
        full_preprocessed_cases=131, full_training_cases=105, full_validation_cases=26,
        source_entries=642, rescore=False, source_arrays_written=False,
        all_other_arms_reserved_by_this_receipt=False)
    original = _derive(old, binding)
    verify_admission(original)
    initial_disk = _disk_receipt(original)
    original_path = metadata/'storage_original_single_arm.json'
    core.publish_json(original_path, original)
    document = copy.deepcopy(old_matched)
    document['original_readonly_admission'] = core.proof(original_path)
    for key in ('aggregate_budget', 'checkpoint_coordination'):
        document[key] = copy.deepcopy(original[key])
    document['arm_roots'] = {'gpu4':str(arm)}
    document[KEY] = copy.deepcopy(binding)
    path = metadata/'storage_admission.json'
    core.publish_json(path, document)
    matched.verify_admission(path, full_hash=False)
    core.publish_json(metadata/'storage_fresh_disk_admission.json', initial_disk)
    return path
