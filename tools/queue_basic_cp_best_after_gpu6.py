"""Resume a completed Multi3 run's BEST26 evaluation after the GPU6 experiment.

Original results remain immutable. This controller never trains or signals.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_basic_cp_ablation as runner


def released(identity):
    import psutil
    try:
        actor = psutil.Process(identity['pid'])
        return (actor.create_time() != identity.get('create_time', identity.get('birth'))
                or actor.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD))
    except psutil.NoSuchProcess:
        return True


def ready(state, actors_released, gpu_idle):
    return (state.get('status') == 'COMPLETE' and state.get('stage') == 'complete'
            and state.get('completed_epochs') == 250 and state.get('evaluated_cases') == 26
            and state.get('full_native_evaluation') is True
            and state.get('historical_metrics_complete') is True
            and actors_released and gpu_idle)


def pinned(request):
    for path, digest in request['source_files'].items():
        if runner.sha(path) != digest:
            raise RuntimeError('Pinned completed Multi3 input changed: ' + path)


def copy_exact(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as inp, target.open('xb') as out:
        shutil.copyfileobj(inp, out, 8 * 2**20)
    if runner.sha(source) != runner.sha(target):
        raise RuntimeError('Independent artifact copy differs: ' + str(target))


def prepare(request):
    import torch
    from hiercp_v1x.basic_cp_ablation_resume import fold
    from tools.basic_cp_ablation_completed_result import training_record
    pinned(request)
    oldroot = Path(request['source_root'])
    old = runner.read(oldroot/'native.json')
    original = runner.read(oldroot/'request.json')
    if not released(original['controller_identity']) or not released(runner.read(oldroot/'train_actor.json')):
        raise RuntimeError('Original Multi3 process remains active')
    if old['arm'] != 'multi3' or old['gpu'] != 1 or request['gpu'] != 6:
        raise RuntimeError('Only explicitly authorized GPU1 Multi3 to GPU6 evaluation is supported')
    saved = torch.load(fold(old)/'checkpoint_final.pth', map_location='cpu', weights_only=False)
    best = torch.load(fold(old)/'checkpoint_best.pth', map_location='cpu', weights_only=False)
    if saved['current_epoch'] != 250 or not 0 < best['current_epoch'] <= 250:
        raise RuntimeError('Full250 training and own BEST required')
    training_record(saved, {'best_current_epoch': best['current_epoch']}, oldroot/'train.log')
    del saved, best
    native = runner.prepare(request)
    for key in ('trainer', 'bank_sha256', 'split', 'CP_probability', 'physical_batch',
                'copies_per_CP_event', 'candidates_per_source', 'epochs'):
        if native[key] != old[key]:
            raise RuntimeError('Scientific identity changed: ' + key)
    for name in ('nnUNetTrainer_OnlinePairedCP.py', 'nnUNetTrainer_BasicCPAblations.py'):
        relative = Path('nnunetv2/training/nnUNetTrainer')/name
        if runner.sha(Path(old['private_runtime'])/relative) != runner.sha(Path(native['private_runtime'])/relative):
            raise RuntimeError('Scientific trainer bytes changed')
    preflight = runner.read(oldroot/'preflight.json')
    if preflight.get('complete') is not True or preflight.get('source_entries') != 642:
        raise RuntimeError('Full original CP preflight required')
    entries = preflight['entries_sha256']
    if len(entries) != 642:
        raise RuntimeError('All642 original sources required')
    for name, digest in entries.items():
        if runner.sha(Path(native['bank']).parent/name) != digest:
            raise RuntimeError('Original CP source changed: ' + name)
    for name in ('preflight.json', 'actual_model.json', 'calibration.json', 'train.log'):
        copy_exact(oldroot/name, Path(native['root'])/name)
    for name in ('checkpoint_best.pth', 'checkpoint_final.pth'):
        copy_exact(fold(old)/name, fold(native)/name)
    pinned(request)
    runner.publish(Path(native['root'])/'evaluation_migration.json', dict(
        training_GPU=1, evaluation_GPU=6, full_training_epochs=250, retraining=False,
        original_root=str(oldroot), original_files=request['source_files'], checkpoint='checkpoint_best.pth'))
    return native


def evaluate(native):
    from hiercp_v1x import v24_native_best_eval_runtime as best
    from hiercp_v1x.v24_direct_gpu_access import install_native_step_guards
    runner.resource_guard(native)
    module, _ = runner.initialize_native(native)
    cls = getattr(module, native['trainer'])
    best._install(cls, native['split']['outer_val'], bank_sha256=native['bank_sha256'])
    install_native_step_guards(cls, lambda: runner.resource_guard(native))
    sys.argv = ['Multi3_BEST26_only', '730', '3d_fullres', '0', '-tr', native['trainer'],
                '-p', native['baseline']['plans_name'], '--val', '--val_best']
    from nnunetv2.run.run_training import run_training_entry
    run_training_entry()


def run(request):
    from hiercp_v1x.basic_cp_ablation_resources import BasicLease
    from hiercp_v1x.basic_cp_ablation_resume import fold
    from hiercp_v1x import v24_native_best_eval_runtime as best
    from tools.basic_cp_ablation_completed_result import completed_report, archive_completed
    lease = BasicLease(request)
    lease._host_lifetime(force=True)
    native = prepare(request)
    runner.status(native, 'waiting_GPU6_completion', 'WAITING', completed_epochs=250,
                  reason='GPU6 full250 + BEST26 + results completion and GPU release required')
    upstream = Path(request['upstream_root'])
    while True:
        lease._inputs()
        lease._host_lifetime()
        if time.time() >= request['host_lifetime']['expires_at_unix'] - 180:
            raise RuntimeError('Original allocation expired before queued evaluation')
        if runner.sha(upstream/'request.json') != request['upstream_request_sha256']:
            raise RuntimeError('GPU6 upstream identity changed')
        state = runner.read(upstream/'status.json')
        identities = [runner.read(upstream/'request.json')['controller_identity']]
        identities += [runner.read(p) for p in upstream.glob('*_actor.json')]
        actors_released = all(released(v) for v in identities)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                       '--format=csv,noheader'], text=True, timeout=20)
        if ready(state, actors_released, request['gpu_uuid'] not in apps):
            break
        if state.get('status') == 'FAILED' and actors_released:
            raise RuntimeError('GPU6 failed before successful completion: ' + str(state.get('error')))
        time.sleep(20)
    runner.resource_guard(native, idle=True)
    runner.status(native, 'BEST26_validation', 'RUNNING', completed_epochs=250)
    runner.run_actor(native, [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
        '--native', str(Path(native['root'])/'native.json')], 'BEST26_validation', runner.environment(native))
    checkpoint_fold = fold(native)
    receipt = best.validate_best_validation_receipt(checkpoint_fold/best.RECEIPT, fold=checkpoint_fold,
        expected_cases=native['split']['outer_val'], bank_sha256=native['bank_sha256'])
    proof = dict(epochs=250, checkpoint='checkpoint_best.pth', best_sha256=receipt['checkpoint_sha256'],
        validation_cases=receipt['validation_cases'], fold=str(checkpoint_fold),
        final_training_witness_sha256=runner.sha(checkpoint_fold/'checkpoint_final.pth'),
        evaluation_checkpoint_is_final=False)
    runner.publish(Path(native['root'])/'training_and_BEST26_complete.json', proof)
    runner.status(native, 'historical_metrics', 'RUNNING', completed_epochs=250)
    command = [sys.executable, '-B', '-u', str(runner.ROOT/'tools/evaluate_v24_historical_metrics.py'),
        '--evaluator-root', native['evaluator_root'], '--prediction', str(checkpoint_fold/'validation'),
        '--output', str(Path(native['root'])/'evaluation'), '--project', native['project'],
        '--checkpoint', 'BEST', '--trainer', native['trainer'],
        '--model-label', 'Basic CP multi3', '--gpu-label', 'GPU1 training -> GPU6 evaluation']
    runner.run_actor(native, command, 'historical_metrics', runner.environment(native))
    reports = completed_report(native, proof)
    archive = archive_completed(native, proof, reports)
    runner.status(native, 'complete', 'COMPLETE', completed_epochs=250, evaluated_cases=26,
        full_native_training=True, full_native_evaluation=True, historical_metrics_complete=True,
        reports=reports, archive=archive)


def main():
    import psutil
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path)
    parser.add_argument('--native', type=Path)
    args = parser.parse_args()
    target = args.request or args.native
    deadline = time.monotonic() + 60
    while not target.exists() and time.monotonic() < deadline:
        time.sleep(.1)
    value = runner.read(target)
    psutil.Process().cpu_affinity(value['cpu_affinity'])
    try:
        if args.native:
            evaluate(value)
        else:
            run(value)
    except Exception as error:
        runner.status(value, 'queued_BEST26', 'FAILED', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
