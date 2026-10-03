"""Foreground spatial-scope DEBUG; never launches long ranking or nnU-Net."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', required=True, type=int)
    p.add_argument('--experiment', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--margin-mm', required=True, type=float, nargs='+')
    p.add_argument('--physical-batch', required=True, type=int)
    p.add_argument('--workers', required=True, type=int)
    p.add_argument('--updates', required=True, type=int)
    p.add_argument('--cuda-gib', required=True, type=float)
    p.add_argument('--rss-gib', required=True, type=float)
    a = p.parse_args()
    if not 1 <= a.updates <= 4 or a.workers < 2:
        raise ValueError('Explicit DEBUG only: one to four updates and parallel workers required')
    from hiercp_v1x.scope_inputs import prepare_debug_inputs
    root = Path(a.output).resolve()
    source, samples = prepare_debug_inputs(a.experiment, root, a.physical_batch)
    command = [sys.executable, '-u', str(ROOT / 'tools/probe_v1_scope_time.py'),
               '--gpu', str(a.gpu), '--baseline-source', str(source), '--fixture-dir', str(samples),
               '--output', str(root / 'measurement'), '--margin-mm', *map(str, a.margin_mm),
               '--physical-batch', str(a.physical_batch), '--workers', str(a.workers),
               '--updates', str(a.updates), '--cuda-gib', str(a.cuda_gib), '--rss-gib', str(a.rss_gib)]
    print('SPATIAL SCOPE DEBUG | existing verified candidates; rebuilding bounded ROI from raw CT', flush=True)
    print('No production checkpoint, 40-epoch comparison or nnU-Net will start.', flush=True)
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPYCACHEPREFIX'] = str(root / 'isolated_bytecode_lookup')
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode:
        raise RuntimeError(f'Scope diagnostic failed ({result.returncode}); inputs and reports preserved at {root}')


if __name__ == '__main__':
    main()
