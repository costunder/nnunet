"""One bounded v1 ROI: actual short-cohort learning curves on CUDA.

Uses all original forty curriculum epochs in the explicitly selected DEBUG cohort.
No production checkpoint, full84/21 claim, CP generation or nnU-Net launch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def child(request):
    # Import current helpers before the archived snapshot gains path priority.
    from hiercp_v1x import bounded_scope
    from hiercp_v1x.scope_learning_inputs import load_samples, rebuild_scope, supervision_digest
    from hiercp_v1x.scope_learning_loop import run_learning
    from hiercp_v1x.scope_probe_support import activate_original, state_digest, _sha
    from tools.local_cnn_device import select
    select(request['gpu'])
    source = Path(request['source']).resolve(strict=True)
    proof = activate_original(source)
    import psutil
    import torch
    from hiercp.model import HierarchicalPyGPlacementModel
    from hiercp.tensor import configure_runtime, set_seed
    report = dict(debug=True, full_training=False, full_evaluation=False,
        production_ready=False, quality_verified=False, branch=request['branch'],
        actual_CT=False, actual_CUDA=False, original_source=proof, phase='preflight')
    process = psutil.Process(); peak = [process.memory_info().rss]; stop = threading.Event()
    def monitor():
        while not stop.wait(.1): peak[0] = max(peak[0], process.memory_info().rss)
    watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
    def budget():
        peak[0] = max(peak[0], process.memory_info().rss)
        if peak[0] > int(request['rss_gib'] * 2**30):
            raise MemoryError('Explicit diagnostic RSS budget exceeded; cohort/model unchanged')
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one real CUDA device required; no CPU fallback')
        total = torch.cuda.get_device_properties(0).total_memory
        cuda_bytes = int(request['cuda_gib'] * 2**30)
        if not 0 < cuda_bytes < total: raise ValueError('Explicit CUDA budget must leave device headroom')
        torch.cuda.set_per_process_memory_fraction(cuda_bytes / total)
        torch.set_num_threads(request['workers'])
        config = json.loads((source / 'config/train.json').read_text(encoding='utf8'))
        if config['cache']['total_candidates'] != 8 or config['cache']['candidate_pool_size'] != 128:
            raise ValueError('Original eight-candidate/128-pool contract required')
        configure_runtime(**{name: config['runtime'][name]
            for name in ('deterministic', 'allow_tf32', 'cudnn_benchmark')})
        samples, manifest = load_samples(request['samples'], request['train_cases'], request['validation_cases'], budget)
        report.update(actual_CT=True, actual_CUDA=True, GPU=torch.cuda.get_device_name(0),
            device_total_bytes=total, CUDA_limit_bytes=cuda_bytes, workers=request['workers'],
            CPU_logical=psutil.cpu_count(), RAM_available=psutil.virtual_memory().available,
            physical_sample_batch=request['physical_batch'], candidates_per_sample=8, candidate_pool=128,
            original_config=config, supervision_sha256=supervision_digest(samples),
            source_manifest_sha256=_sha(Path(request['samples']) / 'fixture_manifest.json'),
            train_case_ids=[s['case_id'] for s in samples if s['split'] == 'train'],
            validation_case_ids=[s['case_id'] for s in samples if s['split'] == 'val'],
            comparison_question='Does the explicitly bounded v1 arm learn its original ranking target? Existing native/30mm evidence is a reference and is not rerun.')
        if request['branch'] == 'bounded':
            report['phase'] = 'bounded replay'
            report['scope_contract'] = bounded_scope.install(request['margin_mm'], source)
            samples, preparation = rebuild_scope(samples, manifest, config, bounded_scope,
                request['margin_mm'], request['workers'], budget)
            report['preparation'] = preparation
        elif request['branch'] != 'native':
            raise ValueError('Unknown branch')
        set_seed(config['seed'], deterministic=config['runtime']['deterministic'])
        net = HierarchicalPyGPlacementModel(**config['model']).cuda()
        report['initial_neural_sha256'] = state_digest(net.state_dict())
        report['model_parameters'] = sum(p.numel() for p in net.parameters())
        if report['model_parameters'] != 10434532: raise AssertionError('Native model size changed')
        epochs = request['smoke_epochs'] if request['smoke_epochs'] is not None else config['training']['epochs']
        report['phase'] = 'actual learning and fixed evaluation'
        report['learning'] = run_learning(net,
            [s for s in samples if s['split'] == 'train'], [s for s in samples if s['split'] == 'val'], config,
            physical_batch=request['physical_batch'], workers=request['workers'], epochs=epochs,
            output_dir=Path(request['report']).parent, smoke=request['smoke_epochs'] is not None,
            rss_budget_check=budget)
        budget()
        report.update(status='PASS', phase='complete', process_peak_rss_bytes=peak[0],
            learning_diagnostic_complete=True, full84_21_accuracy_evaluation=False)
        write_new(request['report'], report)
    except Exception:
        report.update(status='FAILED', error=traceback.format_exc(), process_peak_rss_bytes=peak[0])
        write_new(request['report'], report)
        raise
    finally:
        stop.set(); watcher.join()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--_request', help=argparse.SUPPRESS)
    p.add_argument('--gpu', type=int)
    p.add_argument('--experiment')
    p.add_argument('--margin-mm', type=float)
    p.add_argument('--output')
    p.add_argument('--train-cases', type=int)
    p.add_argument('--validation-cases', type=int)
    p.add_argument('--physical-batch', type=int)
    p.add_argument('--workers', type=int)
    p.add_argument('--cuda-gib', type=float)
    p.add_argument('--rss-gib', type=float)
    p.add_argument('--smoke-epochs', type=int, help='Explicit one/two epoch wiring smoke, never learning-quality approval')
    p.add_argument('--fixture-dir', help='Already verified actual-CT DEBUG publication, with --baseline-source; local smoke only')
    p.add_argument('--baseline-source', help='Byte-exact archived snapshot; local smoke only')
    a = p.parse_args()
    if a._request: child(json.loads(Path(a._request).read_text(encoding='utf8'))); return
    fields = ('gpu', 'margin_mm', 'output', 'train_cases', 'validation_cases', 'physical_batch', 'workers', 'cuda_gib', 'rss_gib')
    if any(getattr(a, name) is None for name in fields): p.error('All GPU, cohort, physical batch, margin and resource settings are required')
    if (a.margin_mm not in (10, 20, 30) or a.train_cases < 2 or a.validation_cases < 1
            or a.physical_batch < 2 or a.physical_batch > a.train_cases or a.train_cases % a.physical_batch
            or a.workers < 2 or a.cuda_gib <= 0 or a.rss_gib <= 0
            or (a.smoke_epochs is not None and a.smoke_epochs not in (1, 2))):
        p.error('Explicit complete DEBUG minibatches/cohort/workers and a single physical margin required')
    output = Path(a.output).resolve()
    if output.exists(): raise FileExistsError('Existing learning evidence is preserved; use a new output')
    if a.fixture_dir is not None or a.baseline_source is not None:
        if not (a.fixture_dir and a.baseline_source and a.smoke_epochs is not None and a.experiment is None):
            p.error('Direct fixture input is restricted to explicit local smoke with both source and fixture paths')
        source, samples = Path(a.baseline_source).resolve(strict=True), Path(a.fixture_dir).resolve(strict=True)
        if output.is_relative_to(source) or output.is_relative_to(samples): raise ValueError('Smoke output must be disjoint from original inputs')
        output.mkdir(parents=True, exist_ok=False)
    else:
        if not a.experiment: p.error('--experiment is required for server learning inputs')
        from hiercp_v1x.scope_inputs import prepare_debug_inputs
        source, samples = prepare_debug_inputs(a.experiment, output, a.train_cases, validation_cases=a.validation_cases)
    reports = []; failures = []
    # Existing native/30mm evidence is preserved, never rerun automatically.
    for branch in ('bounded',):
        directory = output / branch; directory.mkdir(exist_ok=False)
        request = {**vars(a), 'source': str(source), 'samples': str(samples),
            'branch': branch, 'report': str(directory / 'report.json')}
        path = directory / 'request.json'; write_new(path, request)
        from hiercp_v1x.snapshot_inventory import isolated_snapshot_bytecode_env
        env = isolated_snapshot_bytecode_env(directory / 'unused_bytecode_lookup')
        env.pop('PYTHONPATH', None); env.pop('HIERCP_V1X_SAMPLING_CONTRACT', None)
        print(f'LEARNING DEBUG | {branch} | {a.train_cases} train / {a.validation_cases} held-out | physical={a.physical_batch} | margin={a.margin_mm:g}mm', flush=True)
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--_request', str(path)], env=env)
        try: code = process.wait()
        except KeyboardInterrupt:
            print(f'Foreground interruption | directly launched PID {process.pid} | waiting for child; no session termination', flush=True)
            process.wait(); raise
        if code: failures.append(dict(branch=branch, report=request['report'], returncode=code))
        else: reports.append(json.loads(Path(request['report']).read_text(encoding='utf8')))
    result = dict(debug=True, full_training=False, full_evaluation=False, production_ready=False,
        quality_verified=False, branches=reports, failures=failures, status='FAILED' if failures else 'PASS',
        original_v1_GT='source original anchor versus original curriculum candidates',
        v22_GT_is_different='observed P versus unobserved U; never relabeled as CP suitability')
    if not failures:
        curve = reports[0]['learning']['history']
        result['learning_delta'] = {
            split: {name: curve[-1][split][name] - curve[0][split][name]
                    for name in ('MRR', 'top1', 'margin', 'pair_win')}
            for split in ('train', 'validation')}
    write_new(output / 'report.json', result)
    print(f'LEARNING REPORT: {output / "report.json"}', flush=True)
    if failures: raise RuntimeError('Learning diagnostic failed; all branch evidence preserved, no fallback or training-quality claim')
    print(json.dumps(result['learning_delta'], indent=2), flush=True)


if __name__ == '__main__': main()
