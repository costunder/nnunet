"""First half only: v2-style CNN L0 on the completed, preserved 10mm v1 task.

No preparation, baseline rerun, complement, combined experiment or target change.
The baseline's actually measured batch/workers are comparison controls, not new
throughput-optimal settings for this different local encoder.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import math
import re

from .contracts import canonical_hash
from .experiment import digest, read, write_new

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'hiercp_v14_m10_half_A_learning_v1'
MARKER = 'v1x_half_a_digest'
FILES = ('hiercp_v1x/half_a_training.py', 'hiercp_v1x/half_a_entry.py',
         'hiercp_v1x/half_a_model.py', 'hiercp_v1x/half_a_case_metrics.py', 'l0_exploration/model.py',
         'tools/run_v1_half_a.py', 'tools/local_cnn_device.py')


def validation_cohort(files, configured_cases, cache_index):
    """Bind the original materialized/eligible cohort, without fabricating rows."""
    from collections import Counter
    if not isinstance(files, list) or not files or len(set(files)) != len(files):
        raise ValueError('Actual unique baseline validation cache files required')
    counts = Counter()
    entries = cache_index.get('entries') if isinstance(cache_index, dict) else None
    if not isinstance(entries, list):
        raise ValueError('Actual bound cache index entries required')
    rows = [row for row in entries if isinstance(row, dict) and row.get('split') == 'val']
    if sorted(row.get('path','') for row in rows) != sorted(files):
        raise ValueError('Measured validation files differ from actual materialized cache index')
    for row in rows:
        sample = row.get('sample_index')
        if (type(sample) is not int or sample < 0 or row.get('case_id') not in configured_cases
                or row.get('path') != f"{row['case_id']}__{sample:03d}.pt"):
            raise ValueError('Actual validation cache row case/sample binding is different')
    for name in files:
        match = re.fullmatch(r'([A-Za-z0-9_-]+)__(\d{3})\.pt', name) if isinstance(name, str) else None
        if match is None or match.group(1) not in configured_cases:
            raise ValueError('Validation filename is not an original in-split materialized sample')
        counts[match.group(1)] += 1
    return dict(case_ids=[case for case in configured_cases if counts[case]],
        samples=len(files), case_sample_counts=dict(counts),
        configured_cases=len(configured_cases), materialized_cases=len(counts),
        configured_but_not_materialized_cases=[case for case in configured_cases if not counts[case]])


def measured_execution(preflight, baseline):
    identity = preflight.get('identity', {})
    batch, workers = preflight.get('selected_batch_size'), preflight.get('selected_num_workers')
    if (preflight.get('format') != 'hiercp_preflight_calibration_v2'
            or identity.get('seed') != 42
            or identity.get('run_mode') != 'production'
            or Path(identity.get('cache_dir', '')).resolve() != baseline / 'shared/cache'
            or Path(identity.get('checkpoint_path', '')).resolve() != baseline / 'results/v1.0/checkpoint_best.pt'
            or type(batch) is not int or batch < 1
            or type(workers) is not int or workers < 0
            or not isinstance(preflight.get('resource_fingerprint'), dict)
            or not preflight['resource_fingerprint']):
        raise ValueError('Baseline measured batch/worker identity is missing or different')
    canonical_hash(preflight)
    return dict(physical_batch=batch, workers=workers,
                resource_fingerprint=copy.deepcopy(preflight['resource_fingerprint']))


def validate_completed_state(payload, baseline, cfg, execution, scope_digest, publication):
    """Validate real baseline weights and metadata; never load its Adam into A."""
    import torch
    marker = payload.get('state_dict', {}).get('v1x_bounded_scope_digest')
    expected_marker = torch.tensor(list(bytes.fromhex(scope_digest)), dtype=torch.uint8)
    signature = payload.get('training_signature', {})
    calibration = payload.get('preflight_calibration', {})
    calibrated_identity = calibration.get('identity', {})
    training = cfg['training']
    expected_signature = dict(format='hiercp_training_signature_v1', run_mode='production',
        target_epochs=40, seed=42, batch_setting=training['batch_size'],
        batch_size=execution['physical_batch'], worker_setting=training['num_workers'],
        num_workers=execution['workers'], gradient_accumulation_setting=training['gradient_accumulation_steps'],
        gradient_accumulation_steps=training['gradient_accumulation_steps'],
        target_effective_batch_size=training.get('target_effective_batch_size'),
        resolved_effective_batch_size=execution['physical_batch'] * training['gradient_accumulation_steps'],
        calibration_resource_fingerprint=execution['resource_fingerprint'],
        consistency_weight=training['consistency_weight'],
        optimizer=dict(name='AdamW', lr=training['lr'], weight_decay=training['weight_decay'],
                       fused=training['fused_optimizer']),
        scheduler=dict(name='CosineAnnealingLR', t_max=40), amp=training['amp'],
        grad_clip=training['grad_clip'], deterministic=cfg['runtime']['deterministic'],
        allow_tf32=cfg['runtime']['allow_tf32'],
        curriculum={key:training[key] for key in ('easy_epochs','inter_epochs','intra_epochs',
            'model_mine_start_epoch','semi_hard_low_percentile','semi_hard_high_percentile',
            'cross_entropy_weight','pairwise_weight','ordinal_weight','mined_weight')})
    expected_policy = dict(format='hiercp_fixed_validation_v1',
        epoch=training['fixed_validation_epoch'],
        checkpoint_order=['mrr','acc','margin','-ranking','-consistency'],
        metric_precision=training['checkpoint_metric_precision'])
    if (payload.get('format') != 'hiercp_training_state_v1'
            or payload.get('epoch') != 40 or payload.get('target_epochs') != 40
            or payload.get('training_complete') is not True
            or not isinstance(marker, torch.Tensor) or marker.dtype != torch.uint8
            or marker.shape != expected_marker.shape or not torch.equal(marker.cpu(), expected_marker)
            or payload.get('architecture_version') != 'hiercp_source_content_population_metric_v5|bounded_scope_' + scope_digest
            or payload.get('model_kwargs') != cfg['model']
            or payload.get('geometry_contract') != cfg['graph']['geometry_contract']
            or payload.get('ct_clip') != tuple(float(v) for v in cfg['ct_clip'])
            or any(signature.get(key) != value for key,value in expected_signature.items())
            or 'ablation_mode' in signature
            or payload.get('validation_policy') != expected_policy
            or payload.get('best_checkpoint') != str(baseline/'results/v1.0/checkpoint_best.pt')
            or calibration.get('format') != 'hiercp_preflight_calibration_v2'
            or calibration.get('resource_fingerprint') != execution['resource_fingerprint']
            or any(not isinstance(calibrated_identity.get(key), list) or not calibrated_identity[key]
                or signature.get(key) != calibrated_identity[key]
                for key in ('train_cache_files','val_cache_files'))
            or payload.get('graph_config') != cfg['graph']
            or payload.get('cache_publication') != publication
            or signature.get('seed') != 42 or signature.get('target_epochs') != 40
            or signature.get('run_mode') != 'production'
            or signature.get('batch_size') != execution['physical_batch']
            or signature.get('num_workers') != execution['workers']
            or signature.get('gradient_accumulation_steps') != cfg['training']['gradient_accumulation_steps']
            or payload.get('gradient_connectivity', {}).get('verified') is not True):
        raise ValueError('Baseline is not the complete, bound 10mm v1 training state')
    return dict(completed_epoch=40, best_epoch=payload.get('best_epoch'),
                best_mrr=payload.get('best_mrr'), selection=payload.get('best_selection'),
                scope_digest=scope_digest, training_signature=signature,
                architecture_version=payload['architecture_version'])


def baseline_proof(baseline):
    from tools.run_v1_bounded_training import verify_bound_experiment
    import torch
    baseline = Path(baseline).resolve(strict=True)
    manifest = verify_bound_experiment(baseline)
    cfg = manifest['config']
    if (manifest['margin_mm'] != 10 or cfg['seed'] != 42 or cfg['training']['epochs'] != 40
            or cfg['cache']['total_candidates'] != 8 or cfg['cache']['candidate_pool_size'] != 128
            or len(manifest['split']['train']) != 84 or len(manifest['split']['val']) != 21
            or len(manifest['split']['outer_validation_excluded']) != 26):
        raise ValueError('Completed original10mm seed42/full84/21/source-anchor8/pool128 baseline required')
    result = baseline / 'results/v1.0'
    paths = [baseline / 'manifest.json', baseline / 'config.json',
             result / 'initial_validation.json', result / 'checkpoint_best.pt.preflight.json',
             result / 'checkpoint_best.last.pt', result / 'checkpoint_best.pt',
             *(baseline / 'shared/cache' / name for name in ('config.json', 'index.json', 'complete.json', 'manifest.csv')),
             *(baseline / 'shared' / name for name in ('split.json', 'prototype_bank.pt', 'metadata.json', 'manifest.csv'))]
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Actual complete baseline evidence is missing: {path}')
    before = {str(path): digest(path) for path in paths}
    preflight = read(result / 'checkpoint_best.pt.preflight.json')
    execution = measured_execution(preflight, baseline)
    if preflight['identity'].get('graph_config') != cfg['graph']:
        raise ValueError('Baseline calibration graph differs from its10mm manifest')
    initial = read(result / 'initial_validation.json')
    scope = initial.get('scope_contract_sha256')
    if (initial.get('format') != 'hiercp_v1_bounded_initial_validation_v1'
            or initial.get('full_validation') is not True
            or not isinstance(scope, str) or len(scope) != 64
            or any(c not in '0123456789abcdef' for c in scope)):
        raise ValueError('Initial full-validation scope identity missing')
    publication = {name + '_sha256': before[str(baseline / 'shared/cache' / (name + '.json'))]
                   for name in ('config', 'index', 'complete')}
    if preflight['identity'].get('cache_publication') != publication:
        raise ValueError('Baseline cache publication differs from its calibration')
    payload = torch.load(result / 'checkpoint_best.last.pt', map_location='cpu', weights_only=False, mmap=True)
    if payload.get('preflight_calibration') != preflight:
        raise ValueError('Baseline checkpoint calibration differs from its measured preflight file')
    summary = validate_completed_state(payload, baseline, cfg, execution, scope, publication)
    best = torch.load(result / 'checkpoint_best.pt', map_location='cpu', weights_only=False, mmap=True)
    best_marker = best.get('state_dict', {}).get('v1x_bounded_scope_digest')
    if (best.get('training_complete') is not True or best.get('completed_epoch') != 40
            or best.get('target_epochs') != 40
            or best.get('architecture_version') != summary['architecture_version']
            or best.get('graph_config') != cfg['graph'] or best.get('cache_publication') != publication
            or best.get('training_signature') != summary['training_signature']
            or best.get('preflight_calibration') != preflight
            or best.get('epoch') != summary['best_epoch']
            or not isinstance(best_marker, torch.Tensor) or best_marker.dtype != torch.uint8
            or best_marker.shape != (32,)
            or not torch.equal(best_marker.cpu(), payload['state_dict']['v1x_bounded_scope_digest'])):
        raise ValueError('Baseline best is not bound to the same actual scope/cache/recipe')
    del best, payload
    if {str(path): digest(path) for path in paths} != before:
        raise ValueError('Baseline publication changed while reading it')
    return manifest, dict(files=before, execution=execution, neural_baseline=summary,
                          initial_validation=initial, cache_reused_without_preparation=True)


def initialize(baseline, experiment, gpu, cuda_gib, rss_gib):
    baseline = Path(baseline).resolve(strict=True)
    root = Path(experiment).resolve()
    if root == baseline or root.is_relative_to(baseline) or baseline.is_relative_to(root):
        raise ValueError('Half-A output must be disjoint from the preserved baseline')
    if type(gpu) is not int or gpu < 0 or any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in (cuda_gib, rss_gib)):
        raise ValueError('Explicit physical GPU and positive finite CUDA/RSS budgets required')
    native, proof = baseline_proof(baseline)
    from .half_a_model import model_contract
    cfg = copy.deepcopy(native['config'])
    cfg['training']['batch_size'] = proof['execution']['physical_batch']
    cfg['training']['num_workers'] = proof['execution']['workers']
    # These are the ONLY config changes. Model replacement is a bound runtime
    # adapter; cache fingerprints intentionally do not include optimizer fields.
    restored = copy.deepcopy(cfg)
    for key in ('batch_size', 'num_workers'):
        restored['training'][key] = native['config']['training'][key]
    if restored != native['config']:
        raise ValueError('Unexpected change outside measured batch/worker resolution')
    receipt = dict(format=FORMAT, experiment_version='v1.5_half_A',
        baseline_experiment=str(baseline), experiment=str(root),
        baseline_contract_sha256=native['contract_sha256'], baseline_proof=proof,
        config=cfg, source=str(baseline / 'source/v1.0'),
        model_contract=model_contract(),
        implementation={name: digest(ROOT / name) for name in FILES},
        gpu=gpu, cuda_gib=float(cuda_gib), rss_gib=float(rss_gib),
        learning_target='v1_source_original_anchor_vs_curriculum_candidates',
        target_supervision_changed=False, L1_L2_scorer_loss_unchanged=True,
        preparation_started=False, baseline_retraining=False,
        B_automatic_start=False, combined_automatic_start=False,
        epochs=40, quality_verified=False, production_CP_started=False, nnunet_started=False)
    receipt['contract_sha256'] = canonical_hash(receipt)
    if (root / 'manifest.json').exists():
        if read(root / 'manifest.json') != receipt:
            raise ValueError('Existing half-A source/baseline/resource/recipe differs; no silent migration')
        verify(root, receipt)
        return root, receipt
    if root.exists():
        raise FileExistsError('Existing unbound half-A directory is preserved')
    root.mkdir(parents=True)
    write_new(root / 'config.json', cfg)
    write_new(root / 'manifest.json', receipt)
    return root, receipt


def verify(root, receipt=None):
    root = Path(root).resolve(strict=True)
    receipt = read(root / 'manifest.json') if receipt is None else receipt
    if (receipt.get('format') != FORMAT or receipt.get('experiment') != str(root)
            or canonical_hash({k:v for k,v in receipt.items() if k != 'contract_sha256'}) != receipt.get('contract_sha256')
            or read(root / 'config.json') != receipt['config']):
        raise ValueError('Half-A actual manifest/config identity changed')
    if set(receipt.get('implementation', {})) != set(FILES):
        raise ValueError('Half-A implementation inventory changed')
    for name, expected in receipt['implementation'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Half-A actual runtime source changed: {name}')
    native, proof = baseline_proof(receipt['baseline_experiment'])
    if native['contract_sha256'] != receipt['baseline_contract_sha256'] or proof != receipt['baseline_proof']:
        raise ValueError('Baseline weights/cache/execution evidence changed')
    baseline = Path(receipt['baseline_experiment']).resolve(strict=True)
    cfg = copy.deepcopy(native['config'])
    for key, value in (('batch_size', proof['execution']['physical_batch']),
                       ('num_workers', proof['execution']['workers'])):
        cfg['training'][key] = value
    if (receipt['config'] != cfg or receipt.get('source') != str(baseline/'source/v1.0')
            or root == baseline or root.is_relative_to(baseline) or baseline.is_relative_to(root)
            or receipt.get('experiment_version') != 'v1.5_half_A'
            or receipt.get('learning_target') != 'v1_source_original_anchor_vs_curriculum_candidates'
            or receipt.get('epochs') != 40 or receipt.get('L1_L2_scorer_loss_unchanged') is not True
            or any(receipt.get(key) is not False for key in ('target_supervision_changed',
                'preparation_started', 'baseline_retraining', 'B_automatic_start',
                'combined_automatic_start', 'quality_verified', 'production_CP_started', 'nnunet_started'))):
        raise ValueError('Half-A frozen recipe/scope/control experiment changed')
    if (type(receipt.get('gpu')) is not int or receipt['gpu'] < 0
            or any(isinstance(receipt.get(key), bool) or not isinstance(receipt.get(key), (int,float))
                or not math.isfinite(receipt[key]) or receipt[key] <= 0 for key in ('cuda_gib','rss_gib'))):
        raise ValueError('Half-A explicit resources changed to invalid values')
    from .half_a_model import model_contract
    if receipt['model_contract'] != model_contract():
        raise ValueError('Half-A encoder/bridge recipe changed')
    return receipt
