"""Explicit full-cache read-only CP stages; historical copy stages stay intact."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

PROFILE = 'reuse_full105_static_raw_CTseg_and_full131_Blosc2_readonly'
EXTENSION = 'reuse_full105_exact_static_raw_case_operators_readonly_v1'
COORDINATION = 'original_native_final_publication_flock_global7_v1'
EXTENSION_FILE = 'hiercp_v1x/v24_readonly_static_operator_storage.py'
ACTIONS = ('pin-current-gnn', 'prepare-current-bank', 'prepare-native', 'calibrate-native', 'train')
CHECKSUM_ENV = 'V24_READONLY_STORAGE_ADMISSION_SHA256'
FILES = ('tools/run_v24_readonly_nnunet_cp.py', 'hiercp_v1x/v24_readonly_native_storage.py',
         'hiercp_v1x/v24_native_calibration_runtime.py', 'hiercp_v1x/v24_native_crop_runtime.py',
         'hiercp_v1x/v24_native_gradient_runtime.py', 'hiercp_v1x/v24_native_best_eval_runtime.py')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def _stat(path):
    path = Path(path)
    if (not path.is_absolute() or '..' in path.parts or path.is_symlink() or not path.is_file()
            or any(parent.is_symlink() for parent in path.parents)
            or hasattr(os, 'getuid') and path.stat().st_uid != os.getuid()):
        raise ValueError('Exact owned regular immutable read-only profile input required: ' + str(path))
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--storage-admission', type=Path, required=True)
    parser.add_argument('--readonly-worker', choices=('calibrate', 'train'))
    parser.add_argument('--physical-batch', type=int)
    parser.add_argument('--worker-output', type=Path)
    options, remaining = parser.parse_known_args(argv)
    if options.readonly_worker is not None:
        worker = argparse.ArgumentParser(description='Internal explicitly admitted original native worker')
        worker.add_argument('--native', type=Path, required=True)
        if options.readonly_worker == 'calibrate':
            arguments = worker.parse_args(remaining)
            if options.physical_batch not in (2, 4) or options.worker_output is None:
                worker.error('Original baseline/doubled B2/B4 calibration and owned output required')
            arguments.training_args = None
        else:
            arguments, tail = worker.parse_known_args(remaining)
            if options.physical_batch is not None or options.worker_output is not None or not tail or tail[0] != '--':
                worker.error('Original full fresh native CLI argument tail after -- required')
            arguments.training_args = tail[1:]
        arguments.readonly_worker = options.readonly_worker
        arguments.storage_admission = options.storage_admission
        arguments.physical_batch = options.physical_batch
        arguments.worker_output = options.worker_output
        return arguments
    if options.physical_batch is not None or options.worker_output is not None:
        parser.error('Worker-only options cannot alter a public native stage')
    from tools.run_v24_nnunet_cp import parse as original_parse
    arguments = original_parse(remaining)
    if arguments.action not in ACTIONS or arguments.gpu not in (4, 5, 6) or arguments.resume:
        parser.error('This explicit read-only profile requires fresh own-arm GPU4/5/6 and its original five stages')
    if arguments.gpu in (5, 6):
        _stat(options.storage_admission)
        if json.loads(options.storage_admission.read_text(encoding='utf8')).get('storage_extension') != EXTENSION:
            parser.error('GPU5/6 require the explicit full105 static-operator extension')
    arguments.storage_admission = options.storage_admission
    arguments.readonly_worker = None
    return arguments


def admit(path):
    """Check the externally pinned admission without repeating the full69GiB hash."""
    from hiercp_v1x import v24_readonly_native_storage as storage
    path = Path(path)
    before = _stat(path)
    checksum = _sha(path)
    if os.environ.get(CHECKSUM_ENV) != checksum:
        raise ValueError('The watcher-pinned read-only storage admission SHA is required')
    document = json.loads(path.read_text(encoding='utf8'))
    storage.verify_admission(document, full_hash=False)
    extension = document.get('storage_extension')
    if extension is not None:
        if extension != EXTENSION:
            raise ValueError('Unknown explicit full-static-array storage extension')
        from hiercp_v1x import v24_readonly_static_operator_storage as static
        static.verify_admission(document, full_hash=False)
    if (document.get('profile') != PROFILE or document.get('format') != storage.FORMAT
            or document.get('debug') is not False or document.get('model_data_scale_preserved') is not True
            or document.get('scores_reused') is not False or document.get('learned_upper_reused') is not False
            or _stat(path) != before):
        raise ValueError('Actual unchanged full105/full131 static-only storage admission required')
    witnesses = {ROOT / name: _stat(ROOT / name) for name in runtime_files(document)}
    witnesses[path] = before
    def guard():
        for filename, stat in witnesses.items():
            if _stat(filename) != stat:
                raise ValueError('Admitted read-only storage/source file changed: ' + str(filename))
    return document, checksum, guard


def runtime_files(document):
    return (*FILES, EXTENSION_FILE) if document.get('storage_extension') == EXTENSION else FILES


def validate_checkpoint_coordination(proof, document, *, completed, expected_fold, private_runtime):
    """Require the actual original final hook, with no optimizer serialization."""
    admitted = document['checkpoint_coordination']
    publication = document['checkpoint_publication_proof']
    if (not isinstance(proof, dict) or proof.get('format') != COORDINATION or proof.get('active') is not True
            or proof.get('coordination_active') is not True
            or proof.get('global_peak_checkpoint_slots') != 7 or proof.get('per_arm_peak_checkpoint_slots') != 3
            or proof.get('lock_file') != admitted['lock_file']
            or proof.get('original_on_train_end_code_preserved') is not True
            or proof.get('original_on_train_end_called_once_per_completion') is not True
            or proof.get('original_on_train_end_called_once_under_flock') is not True
            or proof.get('successful_original_on_train_end_calls') != (1 if completed else 0)
            or expected_fold not in admitted['folds'].values()
            or proof.get('bound_native_fold') != (expected_fold if completed else None)
            or proof.get('optimizer_work_locked') is not False
            or proof.get('save_checkpoint_unchanged') is not True or proof.get('on_epoch_end_unchanged') is not True
            or proof.get('retention_changed') is not False
            or proof.get('failure_blocks_next_final') is not True or proof.get('repeated_final_rejected') is not True
            or proof.get('private_checkpoint_source', {}).get('sha256') != publication['source']['sha256']
            or proof.get('admitted_source_compiled_methods') != publication['compiled_methods']):
        raise ValueError('Actual single original final publication under the admitted global7 lock is required')
    from hiercp_v1x import v24_readonly_native_storage as core
    from hiercp_v1x import v24_readonly_static_operator_storage as static
    private_source = core.guard(proof['private_checkpoint_source'], full_hash=True)
    expected_path = Path(private_runtime) / 'nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py'
    if private_source != expected_path:
        raise ValueError('Actual final-publication proof must bind the exact private native trainer source')
    methods = static._compiled_methods(private_source)
    if (proof.get('actual_private_compiled_methods') != methods
            or proof.get('expected_private_compiled_methods') != methods):
        raise ValueError('Actual private final-publication compiled methods differ at their exact private filename')
    return proof


@contextmanager
def runtime_scope(pipeline, storage, admission_path, guard, *, write_root, reserve_bytes, extension=None):
    """Adapt storage only inside this dedicated process and restore its namespace."""
    original = {name: getattr(pipeline, name) for name in ('_materialize_bank', 'require_project_budget', 'subprocess')}
    try:
        if extension is None:
            storage.install_pipeline_adapter(pipeline, admission_path)
        else:
            extension.install_pipeline_adapter(pipeline, admission_path, admission_path)
        if pipeline._materialize_bank.__code__ is not original['_materialize_bank'].__code__:
            raise ValueError('Original complete CP materialization code must be preserved')
        def budget():
            guard()
            if shutil.disk_usage(write_root).free < reserve_bytes:
                raise OSError('Original10GiB native checkpoint reserve exhausted; full scale is preserved')
            return original['require_project_budget']()
        pipeline.require_project_budget = budget
        yield
    finally:
        for name, value in original.items():
            setattr(pipeline, name, value)


def redirect(command, *, executable, native_path, admission_path, runtime, trainer, plans):
    """Only the original fresh calibration/production children may be redirected."""
    head = [executable, '-B', '-c']
    prefix = [executable, '-B', '-u', str(ROOT / FILES[0]), '--storage-admission', str(admission_path)]
    if (isinstance(command, list) and len(command) == 7 and command[:4] == [*head, runtime.WORKER_COMMAND]
            and command[4] == str(native_path) and command[5] in ('2', '4')):
        return [*prefix, '--readonly-worker', 'calibrate', '--native', str(native_path),
                '--physical-batch', command[5], '--worker-output', command[6]]
    tail = ['730', '3d_fullres', '0', '-tr', trainer, '-p', plans, '--val_best']
    if isinstance(command, list) and command == [*head, runtime.TRAIN_COMMAND, *tail]:
        return [*prefix, '--readonly-worker', 'train', '--native', str(native_path), '--', *tail]
    raise ValueError('Only the exact original fresh B2/B4 clone or full250 native child is admitted')


def _bind_native(pipeline, storage, args, checksum):
    native = pipeline.read(args.native)
    bank = pipeline.validate_bank(pipeline.read(native['bank']))
    extended = native.get('storage_extension') == EXTENSION
    gpu = native.get('physical_GPU')
    if (native.get('format') != pipeline.CURRENT_FORMAT or gpu not in (4, 5, 6)
            or bank.get('physical_GPU') != gpu
            or not extended and gpu != 4
            or getattr(args, 'gpu', gpu) != gpu
            or native.get('storage_profile') != PROFILE or bank.get('storage_profile') != PROFILE
            or native.get('storage_admission') != str(args.storage_admission)
            or native.get('storage_admission_sha256') != checksum or bank.get('storage_admission_sha256') != checksum):
        raise ValueError('Native/bank must declare this exact own-arm read-only full-cache admission')
    private = Path(native['root']) / 'nnUNet_preprocessed' / bank['baseline']['dataset_name'] / bank['baseline']['data_identifier']
    if extended:
        from hiercp_v1x import v24_readonly_static_operator_storage as static
        if bank.get('storage_extension') != EXTENSION:
            raise ValueError('Native and bank must declare the same full-static-array extension')
        document = json.loads(Path(args.storage_admission).read_text(encoding='utf8'))
        if (not isinstance(document.get('checkpoint_coordination'), dict)
                or native.get('checkpoint_coordination') != document['checkpoint_coordination']
                or native.get('checkpoint_publication_proof') != document.get('checkpoint_publication_proof')):
            raise ValueError('Native must seal its exact admitted original global7 final-publication coordination')
        trainer = getattr(importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer_FrozenV23CP'), pipeline.TRAINER)
        adapters = dict(raw_store=static.install_runtime_adapters(args.storage_admission, Path(native['bank']).parent),
            preprocessed=storage.install_dataset_adapter(args.storage_admission, private),
            checkpoint_publication=static.verify_private_checkpoint_publication(args.storage_admission, native['private_runtime'],
                trainer_class=trainer))
    else:
        adapters = storage.install_runtime_adapters(args.storage_admission, Path(native['bank']).parent, private)
    if (not isinstance(adapters, dict)
            or adapters.get('raw_store', {}).get('exact_original_store_class') is not True
            or adapters.get('raw_store', {}).get('readonly_role_references') != 210
            or adapters.get('preprocessed', {}).get('original_dataset_class') != 'nnUNetDatasetBlosc2'
            or adapters.get('preprocessed', {}).get('original_load_case_code_preserved') is not True
            or adapters.get('preprocessed', {}).get('readonly_mode') != 'r'
            or adapters.get('preprocessed', {}).get('unpack_noop') is not True
            or adapters.get('preprocessed', {}).get('full_preprocessed_cases') != 131):
        raise ValueError('Actual read-only dataset/raw-store installation proof required')
    if extended and (adapters['raw_store'].get('readonly_static_operator_references') != 840
            or adapters['raw_store'].get('original_load_case_code_preserved') is not True
            or adapters['raw_store'].get('read_mode') != 'r'):
        raise ValueError('All105 original eight static array roles must use the exact read-only loader')
    return native, bank, adapters


def worker(args, pipeline, storage, checksum, guard):
    from hiercp_v1x import v24_native_calibration_runtime as runtime
    native, bank, adapters = _bind_native(pipeline, storage, args, checksum)
    pipeline.admit_native(args.native)
    guard()
    if args.readonly_worker == 'calibrate':
        result = runtime._calibrate_native_worker(args.native, args.physical_batch, args.worker_output)
        receipt = args.worker_output / 'readonly_storage_worker.json'
    else:
        expected = ['730', '3d_fullres', '0', '-tr', pipeline.TRAINER, '-p', pipeline.PLANS, '--val_best']
        if args.training_args != expected or bank['baseline']['epochs'] != 250 or bank['baseline']['physical_batch'] != 2:
            raise ValueError('Original fresh full250/physicalB2 Dataset730 native CLI tail required')
        if native.get('storage_extension') == EXTENSION:
            from hiercp_v1x import v24_readonly_static_operator_storage as static
            trainer = getattr(importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer_FrozenV23CP'), pipeline.TRAINER)
            document = json.loads(Path(args.storage_admission).read_text(encoding='utf8'))
            adapters['checkpoint_coordination'] = static.install_checkpoint_adapters(args.storage_admission,
                native['private_runtime'], trainer)
            fold = document['checkpoint_coordination']['folds']['gpu' + str(native['physical_GPU'])]
            validate_checkpoint_coordination(adapters['checkpoint_coordination'], document, completed=False, expected_fold=fold,
                private_runtime=native['private_runtime'])
        from hiercp_v1x.v24_native_crop_runtime import run_training_entry
        sys.argv = [str(Path(native['private_runtime']) / 'nnunetv2/run/run_training.py'), *expected]
        result = run_training_entry()
        if native.get('storage_extension') == EXTENSION:
            validate_checkpoint_coordination(adapters['checkpoint_coordination'], document, completed=True, expected_fold=fold,
                private_runtime=native['private_runtime'])
        receipt = Path(native['root']) / ('readonly_storage_training_worker_' + str(time.time_ns()) + '.json')
    guard()
    pipeline.new_json(receipt, dict(format='v24_readonly_native_worker_v1', status='COMPLETE',
        worker=args.readonly_worker, storage_profile=PROFILE, storage_admission=str(args.storage_admission),
        storage_admission_sha256=checksum, native=str(args.native), native_sha256=_sha(args.native),
        adapters=adapters, cached_inputs_written=False, model_data_scale_preserved=True,
        storage_extension=native.get('storage_extension'),
        checkpoint_coordination=native.get('checkpoint_coordination'),
        physical_batch=args.physical_batch if args.readonly_worker == 'calibrate' else 2,
        native_epochs=250, completed_at=time.time()))
    return result


def _chain_root(args):
    if args.action == 'pin-current-gnn':
        return args.output.parent
    if args.action in ('prepare-current-bank', 'prepare-native'):
        return args.output.parent
    return args.native.parent.parent


def main(args):
    if getattr(args, 'native', None) is not None:
        _stat(args.native)
        native = json.loads(args.native.read_text(encoding='utf8'))
        private = Path(native['private_runtime'])
        if (not private.is_absolute() or private.is_symlink() or not private.is_dir()
                or any(parent.is_symlink() for parent in private.parents)):
            raise ValueError('The exact owned private nnU-Net runtime directory is required')
        sys.path.insert(0, str(private))
    from hiercp_v1x import v24_nnunet_cp as pipeline
    from hiercp_v1x import v24_readonly_native_storage as storage
    from hiercp_v1x import v24_native_calibration_runtime as runtime
    document, checksum, guard = admit(args.storage_admission)
    extension = None
    if document.get('storage_extension') == EXTENSION:
        from hiercp_v1x import v24_readonly_static_operator_storage as extension
    if args.readonly_worker is None and args.gpu in (5, 6) and extension is None:
        raise ValueError('GPU5/6 require the explicitly admitted complete static-operator extension')
    launches = []
    write_root = (Path(native['root']) if args.readonly_worker is not None else _chain_root(args))
    chain_root = write_root.parent if args.readonly_worker is not None else write_root
    growth = 0 if extension is None else document['aggregate_budget']['growth_margin_bytes']
    if extension is not None:
        extension.check_aggregate_disk(document, chain_root, preparation=True)
    with runtime_scope(pipeline, storage, args.storage_admission, guard, write_root=write_root,
            reserve_bytes=document['budget']['minimum_runtime_free_bytes'] + growth, extension=extension):
        if args.readonly_worker is not None:
            return worker(args, pipeline, storage, checksum, guard)
        if args.action in ('pin-current-gnn', 'prepare-current-bank') and (
                str(args.inventory) != document['inventory']['path'] or _sha(args.inventory) != document['inventory']['sha256']):
            raise ValueError('The admitted complete native inventory is required')
        if args.action == 'pin-current-gnn':
            result = pipeline.pin_completed_current(source_output=args.source_output, source_code=args.source_code,
                inventory_path=args.inventory, input_cache=args.input_cache, output=args.output, gpu=args.gpu,
                stunet_checkpoint=args.stunet_checkpoint)
        elif args.action == 'prepare-current-bank':
            if str(args.baseline_preprocessed) != document['baseline']['preprocessed']:
                raise ValueError('The admitted complete original native preprocessing is required')
            result = pipeline.prepare_current_bank(pin_path=args.pin, inventory_path=args.inventory,
                baseline_preprocessed=args.baseline_preprocessed, input_cache=args.input_cache, output=args.output,
                gpu=args.gpu, stunet_checkpoint=args.stunet_checkpoint)
        elif args.action == 'prepare-native':
            pipeline._admit_arm_gpu(pipeline.validate_bank(pipeline.read(args.bank))['pin'], args.gpu)
            preparer = storage if extension is None else extension
            result = preparer.prepare_native(args.bank, args.output, args.storage_admission, pipeline)
        else:
            _bind_native(pipeline, storage, args, checksum)
            original = pipeline.subprocess
            def popen(command, *positional, **keywords):
                guard()
                actual = redirect(command, executable=pipeline.sys.executable, native_path=args.native,
                    admission_path=args.storage_admission, runtime=runtime, trainer=pipeline.TRAINER, plans=pipeline.PLANS)
                launches.append(dict(original_command=command.copy(), actual_command=actual.copy()))
                return original.Popen(actual, *positional, **keywords)
            pipeline.subprocess = SimpleNamespace(Popen=popen, PIPE=original.PIPE, STDOUT=original.STDOUT)
            if args.action == 'calibrate-native':
                result = runtime.calibrate_native(args.native, gpu=args.gpu)
                if len(launches) != 2:
                    raise ValueError('Both original B2/B4 native clone workers are required')
            else:
                result = runtime.train_native(args.native, gpu=args.gpu)
                if len(launches) != 1:
                    raise ValueError('One actual full250 fresh native training worker is required')
            worker_paths = ([Path(row['actual_command'][-1]) / 'readonly_storage_worker.json' for row in launches]
                if args.action == 'calibrate-native' else list(args.native.parent.glob('readonly_storage_training_worker_*.json')))
            if len(worker_paths) != len(launches):
                raise ValueError('Every actual native child must publish its read-only adapter completion proof')
            for launch, path in zip(launches, worker_paths):
                declared = pipeline.read(path)
                kind = 'calibrate' if args.action == 'calibrate-native' else 'train'
                batch = (int(launch['actual_command'][launch['actual_command'].index('--physical-batch') + 1])
                         if kind == 'calibrate' else 2)
                if (declared.get('format') != 'v24_readonly_native_worker_v1' or declared.get('status') != 'COMPLETE'
                        or declared.get('worker') != kind or declared.get('physical_batch') != batch
                        or declared.get('native_epochs') != 250 or declared.get('native') != str(args.native)
                        or declared.get('storage_admission_sha256') != checksum
                        or declared.get('storage_extension') != document.get('storage_extension')
                        or declared.get('native_sha256') != _sha(args.native)
                        or declared.get('cached_inputs_written') is not False
                        or not isinstance(declared.get('adapters'), dict)):
                    raise ValueError('Actual native child lacks its admitted read-only cache/runtime proof')
                if extension is not None:
                    if declared.get('checkpoint_coordination') != document['checkpoint_coordination']:
                        raise ValueError('Actual native child final-publication admission differs')
                    if kind == 'train':
                        validate_checkpoint_coordination(declared['adapters'].get('checkpoint_coordination'), document, completed=True,
                            expected_fold=document['checkpoint_coordination']['folds']['gpu' + str(args.gpu)],
                            private_runtime=native['private_runtime'])
                    elif declared['adapters'].get('checkpoint_coordination') is not None:
                        raise ValueError('Native clone calibration must not activate final-publication coordination')
            for launch, path in zip(launches, worker_paths):
                launch['readonly_worker_receipt'] = dict(path=str(path), sha256=_sha(path))
        guard()
        storage.verify_admission(document, full_hash=False)
        if extension is not None:
            extension.verify_admission(document, full_hash=False)
            extension.check_aggregate_disk(document, chain_root, preparation=False)
    proof = dict(format='v24_readonly_native_stage_v1', status='COMPLETE', action=args.action,
        storage_profile=PROFILE, storage_admission=str(args.storage_admission), storage_admission_sha256=checksum,
        cached_inputs_written=False, model_data_scale_preserved=True, scores_reused=False, learned_upper_reused=False,
        storage_extension=document.get('storage_extension'),
        checkpoint_coordination=document.get('checkpoint_coordination'),
        runtime_sources_sha256={name: _sha(ROOT / name) for name in runtime_files(document)}, launches=launches,
        native_epochs=250, native_physical_batch=2, cp_probability=.5, completed_at=time.time())
    pipeline.new_json(_chain_root(args) / ('readonly_storage_' + args.action.replace('-', '_') + '.json'), proof)
    print(result)
    return result


if __name__ == '__main__':
    main(parse())
