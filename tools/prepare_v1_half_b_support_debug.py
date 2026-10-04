"""CPU-only real-CT DEBUG support extension for the preserved-v1 HALF B smoke.

Build one explicitly named additional inner-training case using the verified
original native builder and unchanged original DEBUG prototype bank. Existing
query fixture/results are read-only. No training, GPU context or checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from hiercp_v1x import bounded_scope
from hiercp_v1x.contracts import canonical_hash
from hiercp_v1x.scope_learning_inputs import rebuild_scope, supervision_digest
from hiercp_v1x.scope_probe_support import activate_original, _sha
from tools.verify_v14_learning_replay import write_new

FORMAT = 'hiercp_v1_half_b_support_extension_DEBUG_v1'
CACHE_FIELDS = ('source_selection', 'source_pad', 'total_candidates', 'candidate_pool_size',
    'easy_fraction', 'inter_fraction', 'intra_fraction', 'max_draws', 'min_liver_coverage',
    'occupied_clearance_vox', 'min_center_separation_mm', 'min_center_separation_vox')


def check_extension_case(case_id, train_cases, validation_cases, raw):
    """Admit a new real original-training case, never a held-out/support duplicate."""
    if (raw.get('complete') is not True or raw.get('debug') is not False
            or not isinstance(case_id, str) or not case_id
            or case_id in set(train_cases) | set(validation_cases)
            or case_id not in raw['split']['inner_train']
            or case_id in raw['split']['inner_val']):
        raise ValueError('New support-only case must belong exclusively to verified inner_train')
    records = [row for row in raw['raw_records'] if row['case_id'] == case_id]
    if len(records) != 1:
        raise ValueError('One signed real CT/annotation record required for extra support case')
    return records[0]


def prepare(a, root, manifest, inputs):
    proof = activate_original(a.source)
    import numpy as np
    import psutil
    import torch
    from hiercp.cache import build_training_sample
    from hiercp.common import CasePaths, load_case, stable_case_seed
    from hiercp.region import build_patient_regions, REGION_CACHE_SEED_SALT
    from hiercp.schema import graph_config_from_dict
    config = manifest['config']
    if (config['seed'] != 42 or config['cache']['total_candidates'] != 8
            or config['cache']['candidate_pool_size'] != 128 or config['training']['epochs'] != 40):
        raise ValueError('Exact native seed42/eight/pool128/production40 contract required')
    raw = json.loads(a.raw_index.read_text(encoding='utf8'))
    row = check_extension_case(a.case, manifest['train_cases'], manifest['validation_cases'], raw)
    for kind in ('image', 'label'):
        if _sha(Path(row[kind])) != row[kind + '_sha256']:
            raise ValueError('Actual added-case CT/annotation differs from signed inventory')
        inputs[str(Path(row[kind]).resolve())] = row[kind + '_sha256']
    fixture = torch.load(a.fixture, map_location='cpu', weights_only=False, mmap=True)
    if (fixture['config'] != config or fixture['train_cases'] != manifest['train_cases']
            or fixture['validation_cases'] != manifest['validation_cases']):
        raise ValueError('Original packed query fixture identity/cohort differs')
    bank = fixture['prototype_bank']
    if set(bank.training_case_ids) != set(fixture['train_cases']):
        raise ValueError('Unchanged native bank must have original training-only fit scope')
    bank_before = bank.fingerprint()
    original_supervision = supervision_digest(fixture['samples'])
    torch.set_num_threads(a.workers)
    process, started = psutil.Process(), time.perf_counter()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def monitor():
        while not stop.wait(.1):
            peak[0] = max(peak[0], process.memory_info().rss)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    def budget():
        peak[0] = max(peak[0], process.memory_info().rss)
        if peak[0] > int(a.rss_gib * 2**30):
            raise MemoryError('Explicit CPU DEBUG RSS budget exceeded; native sample rules preserved')
        if time.perf_counter() - started > a.wall_seconds:
            raise TimeoutError('Explicit CPU DEBUG preparation wall budget exceeded')
    try:
        case = load_case(CasePaths(a.case, Path(row['image']), Path(row['label'])))
        graph = graph_config_from_dict(config['graph'])
        regions = build_patient_regions(case, liver_label=config['labels']['liver'],
            tumor_label=config['labels']['tumor'], config=graph,
            rng=np.random.default_rng(stable_case_seed(42, a.case, REGION_CACHE_SEED_SALT)),
            ct_clip=tuple(config['ct_clip']))
        budget()
        print(f'CPU DEBUG native support sample | {a.case} | full8/pool128 | bank unchanged', flush=True)
        sample = build_training_sample(case, bank, regions, sample_index=0, split_name='train',
            graph_config=graph, liver_label=config['labels']['liver'], tumor_label=config['labels']['tumor'],
            ct_clip=tuple(config['ct_clip']), seed=42,
            **{key:config['cache'][key] for key in CACHE_FIELDS})
        budget()
        if (sample['split'] != 'train' or len(sample['target_locals']) != 8
                or sample['difficulties'].numel() != 8 or int(sample['difficulties'][0]) != 0):
            raise AssertionError('Native additional support sample lost positive-first complete8')
        del case, regions
        receipt = bounded_scope.install(10.0, a.source)
        additional_manifest = dict(source_records=[{key:row[key] for key in
            ('case_id', 'image', 'label', 'image_sha256', 'label_sha256')}])
        bounded, timing = rebuild_scope([sample], additional_manifest, config, bounded_scope,
            10.0, a.workers, budget)
        if (bank.fingerprint() != bank_before
                or supervision_digest(fixture['samples']) != original_supervision
                or supervision_digest(bounded) != supervision_digest([sample])):
            raise AssertionError('Original fixture/bank or additional-case GT was altered')
        path = root/'support_fixture_DEBUG.pt'
        with path.open('xb') as stream:
            torch.save(dict(format=FORMAT, debug=True, native_samples=[sample],
                bounded_samples=bounded, config=config, raw_records=additional_manifest['source_records']), stream)
        result = dict(format=FORMAT, status='COMPLETE', debug=True, actual_CT=True, actual_CUDA=False,
            full_training=False, full_evaluation=False, quality_verified=False, production_ready=False,
            checkpoint_written=False, support_fixture_sha256=_sha(path),
            original_fixture_sha256=manifest['fixture_sha256'], original_supervision_sha256=original_supervision,
            original_configuration_sha256=canonical_hash(config), configuration=config,
            query_train_cases=manifest['train_cases'], validation_cases=manifest['validation_cases'],
            additional_support_only_cases=[a.case], support_train_cases=[*manifest['train_cases'], a.case],
            prototype_training_cases=list(bank.training_case_ids), prototype_bank_fingerprint=bank_before,
            prototype_bank_unchanged=True, candidate_count=8, candidate_pool_size=128,
            native_supervision_sha256=supervision_digest([sample]), bounded_scope=receipt,
            source_records=additional_manifest['source_records'], raw_index_sha256=_sha(a.raw_index),
            raw_training_split=raw['split']['inner_train'], source_proof=proof, preparation=timing,
            process_peak_rss_bytes=peak[0], total_wall_seconds=time.perf_counter()-started,
            resources=dict(cpu_physical=psutil.cpu_count(logical=False), cpu_logical=psutil.cpu_count(),
                RAM_total_bytes=psutil.virtual_memory().total, workers=a.workers,
                rss_limit_bytes=int(a.rss_gib*2**30)), GPU_context_initialized=torch.cuda.is_initialized(),
            input_sha256=inputs, original_inputs_preserved=all(_sha(Path(p)) == h for p,h in inputs.items()),
            source_preserved=all(_sha(a.source/p) == h for p,h in proof['verified_files'].items()))
        if result['GPU_context_initialized'] or not result['original_inputs_preserved'] or not result['source_preserved']:
            raise AssertionError('CPU-only preparation changed preserved input/source or opened GPU context')
        result['identity_sha256'] = canonical_hash(result)
        write_new(root/'support_manifest_DEBUG.json', result)
        return result
    finally:
        stop.set(); watcher.join()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--raw-index', type=Path, required=True)
    p.add_argument('--case', required=True, help='Explicit extra inner_train support-only patient')
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--rss-gib', type=float, required=True)
    p.add_argument('--wall-seconds', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.workers < 2 or not all(math.isfinite(v) and v > 0 for v in (a.rss_gib,a.wall_seconds)):
        p.error('Explicit parallel workers>=2 and finite positive CPU resource bounds required')
    a.source = a.source.resolve(strict=True)
    manifest_path = a.fixture.with_name('fixture_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if not manifest.get('debug') or _sha(a.fixture) != manifest['fixture_sha256']:
        raise ValueError('Exact existing actual-CT DEBUG packed fixture required')
    root = a.output.resolve()
    if root.exists():
        raise FileExistsError('Preserve existing evidence: choose a fresh DEBUG output')
    if root.is_relative_to(a.source) or any(root.is_relative_to(p.resolve().parent)
            for p in (a.fixture,manifest_path,a.raw_index)):
        raise ValueError('Additional output must be disjoint from preserved inputs/source')
    inputs = {str(p.resolve()):_sha(p) for p in (a.fixture,manifest_path,a.raw_index)}
    root.mkdir(parents=True)
    write_new(root/'execution_contract.json', dict(debug=True, CPU_only=True, full_training=False,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}, input_sha256=inputs))
    try:
        result = prepare(a,root,manifest,inputs)
    except Exception:
        write_new(root/'failed.json', dict(debug=True, full_training=False, quality_verified=False,
            error=traceback.format_exc()))
        raise
    print(f"SUPPORT MANIFEST: {root/'support_manifest_DEBUG.json'}",flush=True)
    print(json.dumps({k:result[k] for k in ('support_train_cases','process_peak_rss_bytes',
        'total_wall_seconds','actual_CUDA','prototype_bank_unchanged')},allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
