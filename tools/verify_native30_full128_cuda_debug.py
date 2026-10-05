"""Actual CT/CUDA full-size UNTRAINED smoke; no production checkpoint."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-source', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-cases', nargs='+', required=True)
    p.add_argument('--validation-cases', nargs='+', required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--cuda-gib', type=float, required=True)
    p.add_argument('--rss-gib', type=float, required=True)
    p.add_argument('--resident-gib', type=float, required=True)
    a = p.parse_args()
    if a.workers < 2:
        p.error('Explicit parallel preparation workers required')
    from hiercp_v1x.native30_checkpoint import activate_native30_source
    activate_native30_source(a.original_source)
    from hiercp.common import CasePaths, load_case, stable_case_seed
    from hiercp.region import load_or_build_patient_regions, REGION_CACHE_SEED_SALT
    from hiercp.prototype import build_prototype_bank
    from hiercp.schema import graph_config_from_dict
    import numpy as np
    import torch
    import psutil
    from hiercp_v1x.historical_evaluation import ResourceBudget, assert_new_destination, sha
    budget = ResourceBudget(int(a.cuda_gib * 2**30), int(a.rss_gib * 2**30))
    if not 0 < a.resident_gib < a.rss_gib or a.workers > len(psutil.Process().cpu_affinity()):
        raise ValueError('Explicit DEBUG resident/RSS budget and available parallel workers required')
    assert_new_destination(a.output, (a.original_source, a.inventory))
    torch.set_num_threads(a.workers)
    config = json.loads((a.original_source / 'config/train.json').read_text(encoding='utf8'))
    inventory = json.loads(a.inventory.read_text(encoding='utf8'))
    if (not set(a.train_cases) <= set(inventory['split']['inner_train'])
            or not set(a.validation_cases) <= set(inventory['split']['inner_val'])
            or set(a.train_cases) & set(a.validation_cases)):
        raise ValueError('Explicit real train-only prototype fitting and separate real validation cases required')
    a.output.mkdir(parents=True, exist_ok=False)
    graph = graph_config_from_dict(config['graph'])
    raw = {r['case_id']: r for r in inventory['raw_records']}
    def prepare(case_id):
        budget()
        r = raw[case_id]
        if any(sha(r[key]) != r[key + '_sha256'] for key in ('image', 'label')):
            raise ValueError('Actual DEBUG prototype CT/annotation differs from signed inventory')
        case = load_case(CasePaths(case_id, Path(r['image']), Path(r['label'])))
        regions = load_or_build_patient_regions(case, liver_label=1, tumor_label=2, config=graph,
            ct_clip=tuple(config['ct_clip']), seed=stable_case_seed(42, case_id, REGION_CACHE_SEED_SALT),
            cache_dir=a.output / 'fit_regions', overwrite=False, mmap=True)
        budget()
        return case_id, regions.region_features
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        descriptors = list(pool.map(prepare, a.train_cases))
    bank = build_prototype_bank(descriptors, config=graph, rng=np.random.default_rng(42 + 1009))
    bank.save(a.output / 'prototype_DEBUG.pt')
    command = [sys.executable, '-B', '-u', str(ROOT / 'tools/evaluate_native_v1_full128.py'),
        '--gpu', str(a.gpu), '--original-source', str(a.original_source),
        '--inventory', str(a.inventory), '--prototype', str(a.output / 'prototype_DEBUG.pt'),
        '--output', str(a.output / 'evaluation'), '--workers', str(a.workers),
        '--physical-batch-candidates', '2', '4', '--cuda-gib', str(a.cuda_gib),
        '--rss-gib', str(a.rss_gib), '--resident-gib', str(a.resident_gib),
        '--debug-fresh-model', '--debug-case-ids', *a.validation_cases]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
