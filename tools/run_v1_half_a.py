"""Run ONLY the first binary-search half on the successful10mm v1 baseline."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(root, receipt):
    from hiercp_v1x.experiment import owned_lock, write_new
    from hiercp_v1x.half_a_training import verify
    from hiercp_v1x.snapshot_inventory import isolated_snapshot_bytecode_env
    with owned_lock(root / 'run.lock'):
        verify(root, receipt)
        invocation = root / 'invocations' / uuid.uuid4().hex
        invocation.mkdir(parents=True)
        request = invocation / 'request.json'
        write_new(request, dict(experiment=str(root), contract_sha256=receipt['contract_sha256']))
        env = isolated_snapshot_bytecode_env(invocation / 'unused_bytecode_lookup')
        env.pop('PYTHONPATH', None)
        env.pop('HIERCP_V1X_SAMPLING_CONTRACT', None)
        print('v1.5 half A |10mm | original GT8/pool128 |40epochs | L0 only | no cache preparation', flush=True)
        print('Ctrl+C stops foreground work; resume uses the last completed epoch. Existing baseline preserved.', flush=True)
        process = subprocess.Popen([sys.executable, '-u', '-m', 'hiercp_v1x.half_a_entry',
                                    '--request', str(request)], cwd=ROOT, env=env)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            print(f'Foreground interruption | directly launched PID {process.pid} | waiting for child; no shell/session termination', flush=True)
            process.wait()
            raise
        if code:
            raise RuntimeError(f'Half-A failed({code}); outputs preserved; no B/combined experiment started')
    print(f"HALF-A RESULTS: {root / 'results/half_A'}", flush=True)
    print(f"BASELINE PRESERVED: {receipt['baseline_experiment']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--experiment', required=True)
    p.add_argument('--cuda-gib', type=float, required=True)
    p.add_argument('--rss-gib', type=float, required=True)
    a = p.parse_args()
    from hiercp_v1x.half_a_training import initialize
    root, receipt = initialize(a.baseline, a.experiment, a.gpu, a.cuda_gib, a.rss_gib)
    run(root, receipt)


if __name__ == '__main__':
    main()
