"""Independent full Basic CP ablation; shared immutable preprocessing, own BEST."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

ROOT = Path(__file__).absolute().parents[1]
HISTORICAL_SHA = 'a3df56435fedff46fbc42a502596fdade2632d0211ea412630dba1555f32567d'
TRAINERS = {
    'multi3': 'nnUNetTrainer_250epochs_BasicCPMulti3',
    'candidates256': 'nnUNetTrainer_250epochs_BasicCPCandidates256',
    'blend': 'nnUNetTrainer_250epochs_BasicCPBlend',
}
_LEASES = {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            value.update(block)
    return value.hexdigest()


def publish(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def status(request, phase, state, **values):
    root = Path(request['root'])
    value = dict(format='basic_cp_ablation_full250_status_v1', arm=request['arm'],
        physical_GPU=request['gpu'], stage=phase, status=state, time=time.time(),
        epochs=250, checkpoint='checkpoint_best.pth', **values)
    temporary = root / ('status_' + str(time.time_ns()) + '.tmp')
    publish(temporary, value)
    os.replace(temporary, root / 'status.json')
    return value


def resource_guard(request, *, idle=False):
    from hiercp_v1x.basic_cp_ablation_resources import BasicLease
    key = (request['root'], request['bank'], request['bank_sha256'])
    if key not in _LEASES:
        _LEASES[key] = BasicLease(request)
    return _LEASES[key].verify(force=idle, idle=idle)


def prepare_bank(request):
    if request['arm'] != 'candidates256':
        return request
    status(request, 'candidate_bank', 'RUNNING', source_completed=0, source_total=642,
        candidate_count=256, GNN_scoring=False, CT_preprocessing=False)
    binding = Path(request['candidate_binding'])
    if sha(binding) != request['candidate_binding_sha256']:
        raise RuntimeError('Frozen historical candidate binding changed')
    output = Path(request['root']) / 'bank256'
    command = [sys.executable, '-B', '-u', '-m', 'hiercp_v1x.basic_cp_ablation_bank',
        '--binding', str(binding), '--output', str(output), '--workers', '4',
        '--rss-limit-gib', '64']
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT),
        BASIC_CP_LEASE_REQUEST=str(Path(request['root']) / 'request.json'))
    run_actor(request, command, 'candidate_bank', env)
    index = output / 'index.json'
    value = read(index)
    if value['source_entries'] != 642 or value['candidate_count'] != 256:
        raise RuntimeError('Full642 x256 candidate preparation did not complete')
    return dict(request, bank=str(index), bank_sha256=sha(index),
        source_bank=request['bank'], source_bank_sha256=request['bank_sha256'])


def prepare(request):
    from hiercp_v1x import v24_readonly_native_storage as core
    document = core._admission(request['readonly_admission'])
    if sha(request['readonly_admission']) != request['readonly_admission_sha256']:
        raise RuntimeError('Complete original read-only admission changed')
    root = Path(request['native_root'])
    runtime = root / 'runtime' / 'nnunetv2'
    if runtime.exists():
        raise FileExistsError('Fresh private native runtime required')
    shutil.copytree(document['runtime_package'], runtime,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    historical = runtime / 'training/nnUNetTrainer/nnUNetTrainer_OnlinePairedCP.py'
    if sha(historical) != HISTORICAL_SHA:
        raise RuntimeError('Historical Basic trainer bytes differ')
    trainer = runtime / 'training/nnUNetTrainer/nnUNetTrainer_BasicCPAblations.py'
    with trainer.open('xb') as stream:
        stream.write((ROOT / 'custom_trainers/nnUNetTrainer_BasicCPAblations.py').read_bytes())
    baseline = document['baseline']
    pre = root / 'nnUNet_preprocessed' / baseline['dataset_name']
    data = pre / baseline['data_identifier']
    data.mkdir(parents=True)
    for name, witness in document['private_metadata_files'].items():
        core.copy_private(witness, pre / name)
    gt = pre / 'gt_segmentations'
    gt.mkdir()
    for name, witness in document['ground_truth_files'].items():
        core.copy_private(witness, gt / name)
    result = dict(request, baseline=baseline, split=document['split'],
        private_runtime=str(runtime.parent), private_pre=str(pre), private_data_folder=str(data),
        trainer=TRAINERS[request['arm']], historical_trainer_sha256=HISTORICAL_SHA,
        shared_preprocessed_cases=131, source_preprocessed_arrays_written=False,
        full_training_cases=105, full_validation_cases=26)
    publish(Path(request['root']) / 'native.json', result)
    return result


def environment(native, *, output=None):
    root = Path(native['native_root'])
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(native['gpu']),
        nnUNet_preprocessed=str(root / 'nnUNet_preprocessed'),
        nnUNet_results=str(output or (root / 'nnUNet_results')),
        nnUNet_raw=str(root / 'nnUNet_raw'), ONLINE_CP_BANK=native['bank'], ONLINE_CP_SEED='42',
        BASIC_CP_ABLATION=native['arm'], BASIC_CP_ABLATION_PREFLIGHT=str(Path(native['root']) / 'preflight.json'),
        nnUNet_n_proc_DA='4', nnUNet_def_n_proc='4', nnUNet_compile='false',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join((native['private_runtime'], str(ROOT))))
    return env


def initialize_native(native):
    from hiercp_v1x import v24_readonly_native_storage as core
    sys.path.insert(0, native['private_runtime'])
    adapter = core.install_dataset_adapter(native['readonly_admission'], native['private_data_folder'])
    module = importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer_BasicCPAblations')
    return module, adapter


def preflight(native):
    resource_guard(native)
    module, adapter = initialize_native(native)
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
    dataset = nnUNetDatasetBlosc2(native['private_data_folder'], native['split']['outer_train'])
    result = module.preflight_bank(native['bank'], dataset, arm=native['arm'],
        network_patch_size=(128, 128, 128))
    result.update(readonly_adapter=adapter)
    publish(Path(native['root']) / 'preflight.json', result)
    print('BASIC_CP_PREFLIGHT_COMPLETE ' + json.dumps(result.get('summary', {})), flush=True)


def debug_trial(native, batch, output):
    resource_guard(native)
    import numpy as np
    import torch
    from hiercp_v1x import v24_nnunet_cp as original
    from hiercp_v1x.v24_matched_native_gradient import _clone_step
    module, adapter = initialize_native(native)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one actual CUDA GPU required; no CPU fallback')
    torch.set_num_threads(1)
    plans = read(Path(native['private_pre']) / (native['baseline']['plans_name'] + '.json'))
    plans['continue_training'] = False
    trainer = getattr(module, native['trainer'])(plans, '3d_fullres', 0,
        read(Path(native['private_pre']) / 'dataset.json'), torch.device('cuda'))
    trainer.initialize()
    if trainer.batch_size != 2 or trainer.num_epochs != 250:
        raise RuntimeError('Historical physical batch2/full250 contract changed')
    trainer.batch_size = batch
    step = _clone_step(original._native_clone_step_with_amp_retry,
        Path(native['private_runtime']) / 'nnunetv2/training/nnUNetTrainer/nnUNetTrainer_OnlinePairedCP.py',
        expected_sha256=HISTORICAL_SHA, verify_actual_constructor=False)
    parameters = {name:p for name,p in trainer.network.named_parameters() if p.requires_grad}
    if {id(p) for group in trainer.optimizer.param_groups for p in group['params']} != {id(p) for p in parameters.values()}:
        raise RuntimeError('Actual trainable parameters are missing from optimizer')
    loaders = trainer.get_dataloaders()
    trainer.network.train()
    rows = []
    cp_events = 0
    accepted = True
    error = None
    try:
        for iteration in range(4):
            resource_guard(native)
            started = time.perf_counter()
            data = next(loaders[0])
            flags = np.asarray(data['online_cp_applied'], dtype=np.int64).reshape(-1)
            pastes = np.asarray(data['online_cp_pasted_lesions'], dtype=np.int64).reshape(-1)
            if (flags.shape != (batch,) or not np.array_equal(pastes,
                    flags * (3 if native['arm'] == 'multi3' else 1))):
                raise RuntimeError('Actual CUDA batch CP event/paste count differs')
            cp_events += int(flags.sum())
            loading = time.perf_counter() - started
            before = {name:p.detach().clone() for name,p in parameters.items()}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            result, proof = step(trainer, data, parameters, before)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            loss = float(np.asarray(result['loss']))
            if not math.isfinite(loss):
                raise RuntimeError('Nonfinite actual native CUDA loss')
            row = dict(iteration=iteration, input_shape=list(data['data'].shape), loss=loss,
                load_seconds=loading, compute_seconds=elapsed, step_seconds=loading + elapsed,
                CP_flags=flags.tolist(), pasted_lesions=pastes.tolist(),
                peak_cuda_bytes=torch.cuda.max_memory_allocated(), gradient_proof=proof)
            rows.append(row)
            publish(Path(output) / ('update_' + str(iteration) + '.json'), row)
            print('BASIC_CP_CUDA_DEBUG ' + json.dumps(dict(batch=batch, iteration=iteration,
                loss=loss, seconds=row['step_seconds'], peak_cuda_bytes=row['peak_cuda_bytes'])), flush=True)
            del before, data
    except torch.cuda.OutOfMemoryError as failure:
        if batch == 2:
            raise
        accepted = False
        error = str(failure)
    finally:
        for loader in loaders:
            finish = getattr(loader, '_finish', None)
            if finish is not None:
                finish()
    report = dict(debug=True, accepted=accepted, physical_batch=batch, measurements=rows,
        CP_events=cp_events,
        error=error, production_optimizer_updates=0, production_epochs_executed=0,
        actual_CUDA=True, parameters=sum(p.numel() for p in trainer.network.parameters()),
        architecture=trainer.configuration_manager.network_arch_init_kwargs,
        trainable_parameters=sum(p.numel() for p in parameters.values()), adapter=adapter)
    publish(Path(output) / 'trial.json', report)
    if accepted and cp_events == 0:
        raise RuntimeError('Actual CUDA debug batches did not exercise Basic CP')


def train(native):
    resource_guard(native)
    import psutil
    import torch
    module, adapter = initialize_native(native)
    from hiercp_v1x import v24_native_best_eval_runtime as best
    from hiercp_v1x.v24_direct_gpu_access import install_native_step_guards
    cls = getattr(module, native['trainer'])
    guard = best._install(cls, native['split']['outer_val'], bank_sha256=native['bank_sha256'])
    step_guard = install_native_step_guards(cls, lambda: resource_guard(native))
    previous_initialize = cls.initialize
    previous_epoch_end = cls.on_epoch_end
    active = []

    def initialize(self):
        previous_initialize(self)
        active.append(self)
        tr, val = self.do_split()
        if (self.batch_size != 2 or self.num_epochs != 250 or len(tr) != 105 or len(val) != 26
                or self.configuration_manager.patch_size != [128]*3):
            raise RuntimeError('Full matched native configuration changed')
        publish(Path(native['root']) / 'actual_model.json', dict(trainer=native['trainer'],
            architecture=self.configuration_manager.network_arch_init_kwargs,
            model_class=type(self.network).__name__, parameters=sum(p.numel() for p in self.network.parameters()),
            trainable_parameters=sum(p.numel() for p in self.network.parameters() if p.requires_grad),
            physical_batch=2, gradient_accumulation=1, effective_batch=2, epochs=250,
            train_steps_per_epoch=self.num_iterations_per_epoch,
            validation_steps_per_epoch=self.num_val_iterations_per_epoch,
            actual_train_cases=tr, actual_validation_cases=val, patch_size=[128]*3,
            workers=4, precision='native CUDA autocast/GradScaler', debug=False, subset=False,
            pretrained_segmentation_used=False, own_BEST_guard=guard,
            resource_step_guard=step_guard, readonly_adapter=adapter))

    def epoch_end(self):
        result = previous_epoch_end(self)
        actor = psutil.Process()
        status(native, 'train', 'RUNNING', completed_epochs=int(self.current_epoch),
            RSS_bytes=actor.memory_info().rss, peak_cuda_bytes=torch.cuda.max_memory_allocated())
        return result

    cls.initialize = initialize
    cls.on_epoch_end = epoch_end
    sys.argv = ['basic_cp_ablation', '730', '3d_fullres', '0', '-tr', native['trainer'],
        '-p', native['baseline']['plans_name'], '--val_best']
    from nnunetv2.run.run_training import run_training_entry
    try:
        run_training_entry()
    except Exception as failure:
        if active:
            recovery = Path(native['native_root']) / ('recovery_partial_' + str(time.time_ns()) + '.pth')
            try:
                active[0].save_checkpoint(str(recovery))
                publish(recovery.with_suffix('.json'), dict(path=str(recovery), sha256=sha(recovery),
                    reason=type(failure).__name__ + ': ' + str(failure),
                    raw_current_epoch=int(active[0].current_epoch), partial_epoch_possible=True,
                    full_training_complete=False, existing_results_preserved=True))
            except Exception as backup_failure:
                publish(Path(native['root']) / ('recovery_failed_' + str(time.time_ns()) + '.json'),
                    dict(original_error=type(failure).__name__ + ': ' + str(failure),
                        recovery_error=type(backup_failure).__name__ + ': ' + str(backup_failure),
                        recovery_traceback=traceback.format_exc(), target=str(recovery),
                        full_training_complete=False, existing_results_preserved=True))
        raise
    finally:
        for instance in active:
            for loader in (getattr(instance, 'dataloader_train', None),
                    getattr(instance, 'dataloader_val', None)):
                finish = getattr(loader, '_finish', None)
                if finish is not None:
                    finish()
    fold = Path(native['native_root']) / 'nnUNet_results' / native['baseline']['dataset_name'] / (
        native['trainer'] + '__' + native['baseline']['plans_name'] + '__3d_fullres') / 'fold_0'
    receipt = best.validate_best_validation_receipt(fold / best.RECEIPT, fold=fold,
        expected_cases=native['split']['outer_val'], bank_sha256=native['bank_sha256'])
    final = torch.load(fold / 'checkpoint_final.pth', map_location='cpu', weights_only=False)
    if final['current_epoch'] != 250:
        raise RuntimeError('Actual full250 checkpoint witness missing')
    publish(Path(native['root']) / 'training_and_BEST26_complete.json', dict(epochs=250,
        checkpoint='checkpoint_best.pth', best_sha256=receipt['checkpoint_sha256'],
        validation_cases=receipt['validation_cases'], fold=str(fold),
        final_training_witness_sha256=sha(fold / 'checkpoint_final.pth'), evaluation_checkpoint_is_final=False))


def run_actor(native, command, label, env):
    resource_guard(native)
    log = Path(native['root']) / (label + '.log')
    with log.open('xb', buffering=0) as stream:
        actor = subprocess.Popen(command, env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT)
        import psutil
        process = psutil.Process(actor.pid)
        publish(Path(native['root']) / (label + '_actor.json'), dict(pid=actor.pid,
            birth=process.create_time(), UID=os.getuid(), command=command, log=str(log), stage=label))
        while actor.poll() is None:
            resource_guard(native)
            time.sleep(2)
        code = actor.returncode
    if code:
        raise RuntimeError(label + ' failed (' + str(code) + '); original log: ' + str(log))


def child(native, stage, *, batch=None, output=None):
    command = [sys.executable, '-B', '-u', str(Path(__file__).absolute()),
        '--native', str(Path(native['root']) / 'native.json'), '--stage', stage]
    if batch is not None:
        command += ['--batch', str(batch), '--output', str(output)]
    run_actor(native, command, stage + ('_B' + str(batch) if batch else ''),
        environment(native, output=str(Path(output) / 'nnUNet_results') if output else None))


def run(request):
    from tools.basic_cp_ablation_completed_result import completed_report, archive_completed
    resource_guard(request, idle=True)
    request = prepare_bank(request)
    status(request, 'prepare', 'RUNNING')
    native = prepare(request)
    status(native, 'preflight', 'RUNNING')
    child(native, 'preflight')
    status(native, 'CUDA_DEBUG', 'RUNNING')
    trials = []
    for batch in (2, 4):
        output = Path(native['root']) / ('B' + str(batch) + '_DEBUG')
        output.mkdir()
        child(native, 'debug', batch=batch, output=output)
        trials.append(read(output / 'trial.json'))
    if trials[0]['accepted'] is not True:
        raise RuntimeError('Original physical batch2 actual CUDA check failed')
    publish(Path(native['root']) / 'calibration.json', dict(debug=True, trials=trials,
        production_physical_batch=2, production_effective_batch=2,
        decision='Preserve historical Basic batch2 and optimization steps for one-factor comparison',
        production_optimizer_updates=0))
    resource_guard(native, idle=True)
    status(native, 'train', 'RUNNING', completed_epochs=0)
    child(native, 'train')
    status(native, 'historical_metrics', 'RUNNING', completed_epochs=250)
    proof = read(Path(native['root']) / 'training_and_BEST26_complete.json')
    command = [sys.executable, '-B', '-u', str(ROOT / 'tools/evaluate_v24_historical_metrics.py'),
        '--evaluator-root', native['evaluator_root'], '--prediction', str(Path(proof['fold']) / 'validation'),
        '--output', str(Path(native['root']) / 'evaluation'), '--project', native['project'],
        '--checkpoint', 'BEST', '--trainer', native['trainer'],
        '--model-label', 'Basic CP ' + native['arm'], '--gpu-label', 'GPU' + str(native['gpu'])]
    run_actor(native, command, 'historical_metrics', environment(native))
    status(native, 'report', 'RUNNING', completed_epochs=250)
    reports = completed_report(native, proof)
    archive = archive_completed(native, proof, reports)
    status(native, 'complete', 'COMPLETE', completed_epochs=250, evaluated_cases=26,
        full_native_training=True, full_native_evaluation=True, historical_metrics_complete=True,
        reports=reports, archive=archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--native', type=Path)
    parser.add_argument('--stage', choices=('preflight', 'debug', 'train'))
    parser.add_argument('--batch', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    target = args.request or args.native
    if args.request:
        deadline = time.monotonic() + 120
        while not target.is_file() and time.monotonic() < deadline:
            time.sleep(.1)
    request = read(target)
    import psutil
    psutil.Process().cpu_affinity(request['cpu_affinity'])
    try:
        if args.stage == 'preflight':
            preflight(request)
        elif args.stage == 'debug':
            debug_trial(request, args.batch, args.output)
        elif args.stage == 'train':
            train(request)
        else:
            run(request)
    except Exception as error:
        # Errors remain visible; no process or session is signaled.
        prior = read(Path(request['root']) / 'status.json') if (Path(request['root']) / 'status.json').is_file() else {}
        status(request, args.stage or prior.get('stage', 'controller'), 'FAILED',
            error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
