"""v1.4: one bounded L0 scope experiment branching directly from native v1.0.

Reuse the frozen bounded-training implementation without changing its source,
manifest, checkpoint identity or legacy results/v1.0 storage location.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_v1_bounded_training as controller
from hiercp_v1x.snapshot_inventory import isolated_snapshot_bytecode_env

VERSION = 'v1.4'


def version_record(manifest):
    return dict(format='hiercp_scope_experiment_version_v1',
        experiment_version=VERSION, baseline_model_version='v1.0',
        forked_from='v1.0', includes_v11_v12_v13_changes=False,
        changed_factor='physical_L0_scope', margin_mm=manifest['margin_mm'],
        scope_contract_sha256=manifest['contract_sha256'],
        training_manifest='manifest.json', results_relative_path='results/v1.0',
        storage_version_meaning='preserved native model stage; not the experiment version')


def record_version(root, manifest):
    controller.verify_bound_experiment(root, manifest)
    path = root / 'experiment_version.json'
    expected = version_record(manifest)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or controller.read(path) != expected:
            raise ValueError('Existing experiment version record differs; preserved without overwrite')
    else:
        controller.write_new(path, expected)
    return path


def run(root, manifest):
    with controller.owned_lock(root / 'run.lock'):
        record_version(root, manifest)
        print(f"{VERSION} | native v1.0 branch | margin={manifest['margin_mm']:g}mm | full84train/21val | seed42 |40epochs", flush=True)
        print('Ctrl+C interrupts foreground work; resume uses the last completed epoch.', flush=True)
        for phase in ('prepare', 'train'):
            invocation = root / 'invocations' / (phase + '_' + uuid.uuid4().hex)
            invocation.mkdir(parents=True)
            request = controller.request_for(root, manifest, phase)
            path = invocation / 'request.json'
            controller.write_new(path, request)
            env = isolated_snapshot_bytecode_env(invocation / 'unused_bytecode_lookup')
            env.pop('PYTHONPATH', None)
            env.pop('HIERCP_V1X_SAMPLING_CONTRACT', None)
            command = [sys.executable, '-u', '-m', 'hiercp_v1x.scope_training_entry', '--request', str(path)]
            process = subprocess.Popen(command, cwd=ROOT, env=env)
            try:
                code = process.wait()
            except KeyboardInterrupt:
                print(f'Foreground interruption | directly launched PID {process.pid} | waiting for child; no shell/session termination', flush=True)
                process.wait()
                raise
            if code:
                raise RuntimeError(f'{VERSION} {phase} failed({code}); completed work preserved; later phase not started')
    print(f"{VERSION} RESULTS: {root / 'results/v1.0'}", flush=True)
    print(f"EXPERIMENT VERSION: {root / 'experiment_version.json'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--margin-mm', type=float, required=True)
    p.add_argument('--source-experiment', required=True)
    p.add_argument('--experiment', required=True)
    p.add_argument('--cuda-gib', type=float, required=True)
    p.add_argument('--rss-gib', type=float, required=True)
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if a.gpu < 0 or not all(math.isfinite(v) and v > 0 for v in (a.cuda_gib, a.rss_gib)):
        raise ValueError('Explicit positive finite resource budgets required')
    if a.plan_only:
        old = controller.load_suite(a.source_experiment)
        config = controller.bounded_config(controller.read(Path(a.source_experiment) / 'configs/v1.0.json'), a.margin_mm)
        print(json.dumps(dict(experiment_version=VERSION, baseline_model_version='v1.0',
            forked_from='v1.0', includes_v11_v12_v13_changes=False, config=config,
            train_cases=len(old['split']['train']), validation_cases=len(old['split']['val']),
            epochs=40, margin_mm=a.margin_mm, GPU=a.gpu,
            training_started=False, native_or_30mm_rerun=False), indent=2))
        return
    root, manifest = controller.initialize(a.source_experiment, a.experiment,
        a.margin_mm, a.gpu, a.cuda_gib, a.rss_gib)
    run(root, manifest)


if __name__ == '__main__':
    main()
