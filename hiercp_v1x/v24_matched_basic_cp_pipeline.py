"""Full current-GNN downstream experiment with the immutable historical Basic bank.

The original historical source loader, source RNG and preprocessed paste are
retained. Only the ordered 128 scores change. The old raw-donor experiment is
kept as a separate, reproducible profile.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'reuse_historical_Basic642_scores_only_and_full131_Blosc2_readonly'
FORMAT = 'v24_matched_Basic642_storage_admission_v1'
NATIVE_FORMAT = 'v24_matched_Basic642_native_v1'
STAGE_FORMAT = 'v24_matched_basic_native_stage_v1'
TRAINER = 'nnUNetTrainer_250epochs_FrozenV23CP'
PLANS = 'nnUNetResEncUNetMPlans'
HISTORICAL_TRAINER_SHA = 'a3df56435fedff46fbc42a502596fdade2632d0211ea412630dba1555f32567d'
FILES = ('tools/run_v24_matched_basic_cp.py', 'hiercp_v1x/v24_matched_basic_cp_pipeline.py',
         'hiercp_v1x/v24_matched_basic_cp_scoring.py', 'hiercp_v1x/v24_matched_basic_cp_runtime.py',
         'hiercp_v1x/v24_matched_native_gradient.py', 'hiercp_v1x/v24_native_gradient_runtime.py',
         'hiercp_v1x/v24_native_calibration_runtime.py',
         'hiercp_v1x/v24_native_best_eval_runtime.py', 'hiercp_v1x/v24_readonly_native_storage.py',
         'hiercp_v1x/v24_readonly_static_operator_storage.py', 'hiercp_v1x/v24_nnunet_cp.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def publish(path, document):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)


def verify_admission(path, *, full_hash=False):
    """Validate both original full131 witnesses and the new full642 contract."""
    from . import v24_readonly_native_storage as core
    from . import v24_readonly_static_operator_storage as static
    document = core.read(path)
    if (document.get('format') != FORMAT or document.get('profile') != PROFILE
            or document.get('full_preprocessed_cases') != 131
            or document.get('full_training_cases') != 105 or document.get('full_validation_cases') != 26
            or document.get('source_entries') != 642 or document.get('eligible_cases') != 81
            or document.get('candidate_count') != 128 or document.get('debug') is not False
            or document.get('model_data_scale_preserved') is not True):
        raise ValueError('Explicit full historical Basic642/131 matched admission required')
    original = static.verify_admission(core.guard(document['original_readonly_admission'], full_hash=True),
                                       full_hash=full_hash)
    basic = document['historical_basic_bank']
    index = core.guard(basic['index'], full_hash=full_hash)
    manifest = core.guard(basic['manifest'], full_hash=True)
    entries = read(manifest)
    if (index != Path(basic['path']) / 'index.json' or basic['index_sha256'] != basic['index']['sha256']
            or basic['entries_manifest_sha256'] != basic['manifest']['sha256']
            or basic['manifest_sha256'] != basic['manifest']['sha256']
            or basic['index_stat'] != core.stat(index)
            or len(entries.get('entries', {})) != 642
            or entries.get('index_sha256') != basic['index_sha256']):
        raise ValueError('Original Basic index/all642 immutable witnesses required')
    for witness in entries['entries'].values():
        core.guard(witness, full_hash=full_hash)
    for key in ('baseline', 'split', 'inventory', 'aggregate_budget', 'checkpoint_coordination'):
        if document.get(key) != original[key]:
            raise ValueError('Matched admission changed original full-scale storage contract: ' + key)
    if document.get('arm_roots') != {name: row['root'] for name, row in original['aggregate_budget']['arms'].items()}:
        raise ValueError('Matched private arm namespace differs')
    return document


def check_aggregate_disk(document, chain_root, *, preparation=True):
    from . import v24_readonly_native_storage as core
    from . import v24_readonly_static_operator_storage as static
    original = static.verify_admission(core.guard(document['original_readonly_admission']))
    return static.check_aggregate_disk(original, chain_root, preparation=preparation)


def _guard_native(native_path):
    from . import v24_matched_basic_cp_runtime as runtime
    from . import v24_readonly_native_storage as core
    native = read(core.regular(native_path))
    admission = verify_admission(native['storage_admission'])
    overlay = runtime.validate_overlay(native['overlay'])
    if (native.get('format') != NATIVE_FORMAT or native.get('storage_profile') != PROFILE
            or native.get('overlay_sha256') != sha(native['overlay'])
            or native.get('storage_admission_sha256') != sha(native['storage_admission'])
            or native.get('baseline') != admission['baseline'] or native.get('split') != admission['split']
            or native.get('physical_GPU') not in (4, 5, 6)
            or native.get('model_weights_fresh') is not True
            or native.get('pretrained_segmentation_used') is not False):
        raise ValueError('Sealed fresh matched own-arm full native runtime required')
    for name, checksum in native['source_files_sha256'].items():
        if sha(ROOT / name) != checksum:
            raise ValueError('Matched implementation changed: ' + name)
    pre = Path(native['root']) / 'nnUNet_preprocessed' / native['baseline']['dataset_name']
    for name, checksum in native['copied_native_array_sha256'].items():
        if sha(core.regular(pre / name)) != checksum:
            raise ValueError('Private original metadata/validation annotation changed: ' + name)
    trainers = Path(native['private_runtime']) / 'nnunetv2/training/nnUNetTrainer'
    for name, checksum in native['private_trainer_sha256'].items():
        if sha(core.regular(trainers / name)) != checksum:
            raise ValueError('Exact private historical trainer changed: ' + name)
    for name, checksum in native['private_runtime_files_sha256'].items():
        if sha(core.regular(Path(native['private_runtime'])/'nnunetv2'/name)) != checksum:
            raise ValueError('Original private native runtime changed: ' + name)
    return native, overlay, admission


def prepare_native(overlay_path, output, admission_path, *, gpu):
    from . import v24_matched_basic_cp_runtime as runtime
    from . import v24_readonly_native_storage as core
    from . import v24_readonly_static_operator_storage as static
    from . import v24_nnunet_cp as original
    original.require_project_budget()
    document = verify_admission(admission_path)
    overlay = runtime.validate_overlay(overlay_path, full_hash=True)
    output = Path(output).absolute()
    arm = Path(document['arm_roots']['gpu' + str(gpu)])
    if output != arm / 'native' or output.exists():
        raise FileExistsError('Fresh exact matched native namespace required')
    check_aggregate_disk(document, arm)
    original_document = static.verify_admission(core.guard(document['original_readonly_admission']))
    package = Path(importlib.import_module('nnunetv2').__file__).resolve().parent
    historical = package / 'training/nnUNetTrainer/nnUNetTrainer_OnlinePairedCP.py'
    if sha(historical) != HISTORICAL_TRAINER_SHA:
        raise ValueError('Actual original preprocessed historical Basic trainer required')
    output.mkdir()
    private = output / 'runtime/nnunetv2'
    shutil.copytree(package, private, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'nnUNetTrainer_FrozenV23CP.py'))
    trainers = private / 'training/nnUNetTrainer'
    alias = trainers / 'nnUNetTrainer_FrozenV23CP.py'
    alias.write_text('from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCP import '
                     'nnUNetTrainer_250epochs_OnlineHierCPExactArgmax\n\n'
                     'class nnUNetTrainer_250epochs_FrozenV23CP(nnUNetTrainer_250epochs_OnlineHierCPExactArgmax):\n'
                     '    """Historical complete trainer; name retains proven checkpoint coordination."""\n', encoding='utf8')
    from .v24_native_calibration_runtime import _prove_fresh_cli
    native_base = private/'training/nnUNetTrainer/nnUNetTrainer.py'
    native_cli = private/'run/run_training.py'
    fresh_cli = _prove_fresh_cli(native_base.read_text(encoding='utf8'),native_cli.read_text(encoding='utf8'))
    private_sources = {}
    for name,witness in original_document['runtime_files'].items():
        if Path(name).name == alias.name:
            continue
        target = core.regular(private/name)
        if sha(target) != witness['sha256']:
            raise ValueError('Private actual installed native source changed: '+name)
        private_sources[name] = witness['sha256']
    pre = output / 'nnUNet_preprocessed' / document['baseline']['dataset_name']
    pre.mkdir(parents=True)
    data = pre / document['baseline']['data_identifier']
    data.mkdir()
    copied = {}
    for name, witness in original_document['private_metadata_files'].items():
        core.copy_private(witness, pre / name)
        copied[name] = witness['sha256']
    for name, witness in original_document['ground_truth_files'].items():
        relative = 'gt_segmentations/' + name
        core.copy_private(witness, pre / relative)
        copied[relative] = witness['sha256']
    if original.validate_baseline(pre, document['split'])['source_files_sha256'] != document['baseline']['source_files_sha256']:
        raise ValueError('Original full-scale plans/split/normalization changed')
    native = dict(format=NATIVE_FORMAT, storage_profile=PROFILE, root=str(output), physical_GPU=gpu,
        overlay=str(Path(overlay_path).absolute()), overlay_sha256=sha(overlay_path),
        bank=str(Path(overlay_path).absolute()), bank_sha256=sha(overlay_path),
        baseline=document['baseline'], split=document['split'], private_runtime=str(private.parent),
        private_data_folder=str(data), trainer=TRAINER, model_weights_fresh=True,
        pretrained_segmentation_used=False, source_runtime=str(package),
        storage_admission=str(Path(admission_path).absolute()), storage_admission_sha256=sha(admission_path),
        original_storage_admission=document['original_readonly_admission']['path'],
        original_storage_admission_sha256=document['original_readonly_admission']['sha256'],
        checkpoint_coordination=document['checkpoint_coordination'],
        private_trainer_sha256={name: sha(trainers / name) for name in ('nnUNetTrainer_OnlinePairedCP.py', alias.name)},
        private_runtime_files_sha256=private_sources,fresh_CLI_semantics=fresh_cli,
        copied_native_array_sha256=copied, source_files_sha256={name: sha(ROOT / name) for name in FILES},
        resource_contract=original.resource_contract(), native_preprocessed_source_written=False,
        full_preprocessed_cases=131, train_cases=105, validation_cases=26, epochs=250,
        physical_batch=2, gradient_accumulation=1, effective_batch=2, patch_size=[128]*3,
        cp_probability=.5, source_entries=642, eligible_cases=81, candidates_per_source=128,
        paste_contract='unchanged historical preprocessed source_data/source_mask paste',
        source_selection='unchanged historical uniform own-patient source RNG',
        location_selection='argmax of current own-arm GNN scores for selected source',
        evaluation_checkpoint='checkpoint_best.pth', debug=False)
    publish(output / 'native.json', native)
    _guard_native(output / 'native.json')
    check_aggregate_disk(document, arm, preparation=False)
    return output / 'native.json'


def environment(native, native_path):
    env = os.environ.copy()
    root = Path(native['root'])
    env.update(nnUNet_preprocessed=str(root/'nnUNet_preprocessed'), nnUNet_results=str(root/'nnUNet_results'),
        nnUNet_raw=str(root/'nnUNet_raw'), ONLINE_CP_BANK=native['overlay'], ONLINE_CP_SEED='42',
        V24_MATCHED_OVERLAY=native['overlay'], V24_MATCHED_NATIVE=str(native_path),
        nnUNet_n_proc_DA='4', nnUNet_def_n_proc='4', nnUNet_compile='false',
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join((native['private_runtime'], str(ROOT))))
    return env


def calibration_worker(native_path, physical_batch, output):
    """DEBUG: four actual complete clone updates, with the unchanged full CP loader."""
    from tools.local_cnn_device import select
    from . import v24_nnunet_cp as original
    native, overlay, admission = _guard_native(native_path)
    select(native['physical_GPU'])
    sys.path.insert(0, native['private_runtime'])
    from . import v24_matched_basic_cp_runtime as runtime
    import numpy as np
    import torch
    if physical_batch not in (2, 4) or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Actual singleton CUDA full B2/B4 clone required')
    torch.set_num_threads(1)
    adapters = runtime.install_native_runtime(native_path, training=False)
    cls = getattr(importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer_FrozenV23CP'), TRAINER)
    pre = Path(native['root'])/'nnUNet_preprocessed'/native['baseline']['dataset_name']
    plans_path = pre/(PLANS+'.json')
    plans = read(plans_path)
    if any(name in plans for name in ('continue_training','only_run_validation')):
        raise ValueError('Original disk plans must not contain CLI runtime flags')
    plans_sha = sha(plans_path)
    plans['continue_training'] = False
    trainer = cls(plans, '3d_fullres', 0, read(pre/'dataset.json'), torch.device('cuda'))
    trainer.initialize()
    if trainer.num_epochs != 250 or trainer.batch_size != 2 or trainer.configuration_manager.patch_size != [128]*3:
        raise ValueError('Original complete historical native model/plans changed')
    trainer.batch_size = physical_batch
    from .v24_matched_native_gradient import clone_step
    step = clone_step(original._native_clone_step_with_amp_retry,
        historical_trainer_source=Path(native['private_runtime'])/'nnunetv2/training/nnUNetTrainer/nnUNetTrainer_OnlinePairedCP.py')
    loaders = trainer.get_dataloaders()
    parameters = {name: p for name, p in trainer.network.named_parameters() if p.requires_grad}
    if not parameters or {id(p) for group in trainer.optimizer.param_groups for p in group['params']} != {id(p) for p in parameters.values()}:
        raise ValueError('Native trainable parameters are disconnected from optimizer')
    trainer.network.train()
    rows = []
    cp_events = 0
    try:
        for iteration in range(4):
            started = time.perf_counter()
            batch = next(loaders[0])
            loading = time.perf_counter() - started
            shape = list(batch['data'].shape)
            if shape[0] != physical_batch or shape[-3:] != [128]*3:
                raise ValueError('Full physical batch/input resolution changed')
            flags = np.asarray(batch['online_cp_applied']).reshape(-1).tolist()
            cp_events += sum(bool(value) for value in flags)
            before = {name:p.detach().clone() for name,p in parameters.items()}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            result, gradient = step(trainer, batch, parameters, before)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            loss = float(np.asarray(result['loss']))
            changed = [name for name,p in parameters.items() if not torch.equal(before[name],p.detach())]
            if (not math.isfinite(loss) or not changed
                    or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters.values())
                    or not any('encoder' in name for name in changed) or not any('decoder' in name for name in changed)):
                raise ValueError('Full native forward/loss/gradient/update admission failed')
            row = dict(iteration=iteration, input_shape=shape, CP_flags=flags, loss=loss,
                loading_seconds=loading, forward_backward_optimizer_seconds=elapsed,
                full_step_seconds=loading+elapsed, gradient_finite=True, native_gradient_admission=gradient,
                changed_parameter_tensors=len(changed), encoder_and_decoder_updated=True,
                peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated(),
                resources=original.require_project_budget())
            rows.append(row)
            publish(Path(output)/('update_'+str(iteration)+'.json'),row)
            print(json.dumps(dict(phase='matched_Basic_CP_clone_DEBUG', batch=physical_batch, **row)), flush=True)
            del before, batch
        if cp_events <= 0:
            raise ValueError('No actual preprocessed CP sample observed in clone; training remains unadmitted')
        report = dict(format='v24_matched_Basic642_clone_trial_v1', debug=True, accepted=True,
            physical_batch=physical_batch, production_physical_batch=2, production_epochs=250,
            production_updates=0, clone_optimizer_updates=4, cp_probability=.5, patch_size=[128]*3,
            bank_sha256=native['bank_sha256'], model_parameter_count=sum(p.numel() for p in parameters.values()),
            full_training_cases=105, full_validation_cases=26, source_entries=642, candidates_per_source=128,
            cp_events=cp_events, warmup=rows[0], measurements=rows[1:], adapters=adapters,
            original_model_architecture_unchanged=True, original_source_rng_and_paste_unchanged=True)
    except torch.cuda.OutOfMemoryError as error:
        if physical_batch == 2:
            raise
        report = dict(format='v24_matched_Basic642_clone_trial_v1', debug=True, accepted=False,
            rejection='actual_torch_CUDA_OutOfMemoryError', error=str(error), physical_batch=physical_batch,
            production_physical_batch=2, production_epochs=250, production_updates=0,
            bank_sha256=native['bank_sha256'], completed_clone_updates=len(rows))
    finally:
        for loader in loaders:
            finish = getattr(loader, '_finish', None)
            if finish is not None:
                finish()
    publish(Path(output)/'trial.json', report)
    if sha(plans_path) != plans_sha:
        raise ValueError('Original immutable plans changed during clone calibration')
    return Path(output)/'trial.json'


def _child(command, env, log_path):
    with Path(log_path).open('x', encoding='utf8') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in child.stdout:
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        result = child.wait()
    if result:
        raise RuntimeError('Matched native stage failed; logs/results preserved: '+str(log_path))


def validate_calibration(native_path):
    native, overlay, admission = _guard_native(native_path)
    path = Path(native['root'])/'calibration.json'
    proof = read(path)
    if (proof.get('format') != 'v24_matched_Basic642_clone_calibration_v1'
            or proof.get('bank_sha256') != native['bank_sha256'] or proof.get('debug') is not True
            or proof.get('production_updates') != 0 or proof.get('production_epochs') != 250
            or proof.get('production_physical_batch') != 2 or proof.get('cp_probability') != .5
            or {row['report']['physical_batch'] for row in proof.get('trials', [])} != {2, 4}):
        raise ValueError('Both complete actual native physical batch trials required')
    for row in proof['trials']:
        report = row['report']
        if sha(row['path']) != row['sha256'] or read(row['path']) != report or report['bank_sha256'] != native['bank_sha256']:
            raise ValueError('Matched calibration proof changed')
        if report['accepted'] is True:
            if (report['clone_optimizer_updates'] != 4 or report['cp_events'] <= 0
                    or len(report['measurements']) != 3 or report['original_source_rng_and_paste_unchanged'] is not True):
                raise ValueError('Actual full CP clone updates missing')
        elif report['physical_batch'] != 4 or report.get('rejection') != 'actual_torch_CUDA_OutOfMemoryError':
            raise ValueError('Original production physical batch must pass')
    return path


def calibrate_native(native_path, *, gpu):
    native, overlay, admission = _guard_native(native_path)
    if native['physical_GPU'] != gpu:
        raise ValueError('Calibration belongs to another arm')
    root = Path(native['root'])
    if (root/'calibration.json').exists():
        raise FileExistsError('Completed native calibration is immutable')
    output = root/('matched_clone_DEBUG_'+str(time.time_ns()))
    output.mkdir()
    reports = []
    for batch in (2, 4):
        trial = output/('batch_'+str(batch))
        trial.mkdir()
        env = environment(native, native_path)
        env['nnUNet_results'] = str(trial/'nnUNet_results')
        command = [sys.executable, '-B', '-u', str(ROOT/'tools/run_v24_matched_basic_cp.py'),
            'clone-debug-worker', '--native', str(native_path), '--physical-batch', str(batch), '--output', str(trial)]
        _child(command, env, trial/'clone.log')
        path = trial/'trial.json'
        reports.append(dict(path=str(path), sha256=sha(path), report=read(path)))
    path = root/'calibration.json'
    publish(path, dict(format='v24_matched_Basic642_clone_calibration_v1', debug=True,
        bank_sha256=native['bank_sha256'], production_updates=0, production_epochs=250,
        production_physical_batch=2, cp_probability=.5, measured_physical_batches=[2,4], trials=reports,
        selection='Historical physical B2 retained for comparison; full B4 measured independently'))
    validate_calibration(native_path)
    return path


def train(native_path, *, gpu):
    from . import v24_native_best_eval_runtime as best
    native, overlay, admission = _guard_native(native_path)
    if native['physical_GPU'] != gpu:
        raise ValueError('Native training belongs to another arm')
    calibration = validate_calibration(native_path)
    root = Path(native['root'])
    fold = root/'nnUNet_results'/native['baseline']['dataset_name']/(TRAINER+'__'+PLANS+'__3d_fullres')/'fold_0'
    if fold.exists():
        raise FileExistsError('Fresh full native run required; no replay or implicit recovery')
    command = [sys.executable,'-B','-u','-c',
        'from hiercp_v1x.v24_matched_basic_cp_runtime import run_training_entry; run_training_entry()',
        '730','3d_fullres','0','-tr',TRAINER,'-p',PLANS,'--val_best']
    stamp = str(time.time_ns())
    publish(root/('training_started_'+stamp+'.json'), dict(format=NATIVE_FORMAT, command=command,
        epochs=250, physical_batch=2, cp_probability=.5, calibration_sha256=sha(calibration),
        bank_sha256=native['bank_sha256'], source_entries=642, candidates_per_source=128,
        evaluation_checkpoint=best.BEST, debug=False))
    _child(command, environment(native,native_path), root/('train_'+stamp+'.log'))
    receipt = fold/best.RECEIPT
    proof = best.validate_best_validation_receipt(receipt, fold=fold,
        expected_cases=native['split']['outer_val'], bank_sha256=native['bank_sha256'])
    final = fold/'checkpoint_final.pth'
    publish(root/('training_complete_'+stamp+'.json'),dict(format=NATIVE_FORMAT,epochs=250,GPU=gpu,
        checkpoint=str(final),checkpoint_sha256=sha(final),checkpoint_role='full250_training_completion_witness',
        evaluation_checkpoint_name=best.BEST,evaluation_checkpoint_path=proof['checkpoint_path'],
        evaluation_checkpoint_sha256=proof['checkpoint_sha256'],best_validation_receipt=str(receipt),
        best_validation_receipt_sha256=sha(receipt),bank_sha256=native['bank_sha256']))
    return final


def stage_proof(request, action):
    """Evidence read by the unchanged resource-aware chain after each child."""
    from . import v24_nnunet_cp as original
    from . import v24_matched_basic_cp_runtime as runtime
    root = Path(request['chain_root'])
    stage = root/('matched_storage_'+action.replace('-','_')+'.json')
    value = read(stage)
    if (value.get('format') != STAGE_FORMAT or value.get('status') != 'COMPLETE'
            or value.get('action') != action or value.get('storage_profile') != PROFILE
            or value.get('storage_admission_sha256') != request['storage_admission_sha256']
            or value.get('cached_inputs_written') is not False or value.get('model_data_scale_preserved') is not True
            or value.get('runtime_sources_sha256') != {name:sha(ROOT/name) for name in FILES}):
        raise ValueError('Actual matched native child completion proof required')
    path = root/'pin.json'
    pin = original.validate_current_pin(read(path))
    if pin['physical_GPU'] != request['GPU_arm']:
        raise ValueError('Wrong completed GNN arm')
    if action != 'pin-current-gnn':
        path = root/'bank/score_overlay.json'
        runtime.validate_overlay(path)
    if action in ('prepare-native','calibrate-native','train'):
        path = root/'native/native.json'
        _guard_native(path)
    if action in ('calibrate-native','train'):
        validate_calibration(path)
    if action == 'train':
        from . import v24_native_best_eval_runtime as best
        native = read(path)
        fold = Path(native['checkpoint_coordination']['folds']['gpu'+str(request['GPU_arm'])])
        receipt = fold/best.RECEIPT
        best_proof = best.validate_best_validation_receipt(receipt,fold=fold,
            expected_cases=native['split']['outer_val'],bank_sha256=native['bank_sha256'])
        complete = list((root/'native').glob('training_complete_*.json'))
        final = fold/'checkpoint_final.pth'
        if len(complete)!=1:
            raise ValueError('One completed full250 native training proof required')
        completion = read(complete[0])
        expected = dict(format=NATIVE_FORMAT,epochs=250,GPU=request['GPU_arm'],checkpoint=str(final),
            checkpoint_sha256=sha(final),checkpoint_role='full250_training_completion_witness',
            evaluation_checkpoint_name=best.BEST,evaluation_checkpoint_path=best_proof['checkpoint_path'],
            evaluation_checkpoint_sha256=best_proof['checkpoint_sha256'],best_validation_receipt=str(receipt),
            best_validation_receipt_sha256=sha(receipt),bank_sha256=native['bank_sha256'])
        if completion != expected:
            raise ValueError('Actual own final training witness and BEST evaluation receipt differ')
        path = complete[0]
    return dict(action=action,path=str(path),sha256=sha(path),matched_stage_receipt=str(stage),
                matched_stage_receipt_sha256=sha(stage),physical_GPU=request['GPU_arm'])
