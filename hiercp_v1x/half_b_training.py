"""Complement B only, against the completed original 10mm/source-anchor v1 task.

The unchanged complete v1 objective is deliberately different from native
v2.2 observation CE/alignment training. No A replacement is installed here.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

from .contracts import canonical_hash
from .experiment import digest, read, write_new
from .half_a_training import baseline_proof, validation_cohort

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'hiercp_v14_m10_half_B_learning_v1'
MARKER = 'v1x_half_b_digest'
FILES = ('hiercp_v1x/half_b_training.py', 'hiercp_v1x/half_b_entry.py',
    'hiercp_v1x/half_b_model.py', 'hiercp_v1x/half_b_support.py',
    'tools/run_v1_half_b.py', 'hiercp_v1x/half_a_training.py',
    'hiercp_v1x/half_a_case_metrics.py', 'hiercp_v1x/half_a_entry.py',
    'hiercp_v1x/scope_training_entry.py', 'hiercp_v1x/epoch_telemetry.py',
    'hiercp_v222/model.py', 'hiercp_v222/clustering.py',
    'l0_regions/donor_learning.py', 'l0_local_cnn/recommendation.py',
    'config/prompt_graph_v222_v1_l0.json', 'tools/local_cnn_device.py')


def recipe(baseline, experiment, gpu, cuda_gib, rss_gib):
    baseline = Path(baseline).resolve(strict=True)
    root = Path(experiment).resolve()
    if root == baseline or root.is_relative_to(baseline) or baseline.is_relative_to(root):
        raise ValueError('Half-B output must be disjoint from the preserved baseline')
    if (type(gpu) is not int or gpu < 0 or any(isinstance(v, bool)
            or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for v in (cuda_gib, rss_gib))):
        raise ValueError('Explicit physical GPU and positive finite CUDA/RSS budgets required')
    native, proof = baseline_proof(baseline)
    cfg = copy.deepcopy(native['config'])
    cfg['training']['batch_size'] = proof['execution']['physical_batch']
    cfg['training']['num_workers'] = proof['execution']['workers']
    index = read(baseline / 'shared/cache/index.json')
    files = proof['neural_baseline']['training_signature']['train_cache_files']
    from .half_b_support import signed_training_cohort
    cohort = signed_training_cohort(files, native['split']['train'], native['split']['val'], index)
    from .half_b_model import model_contract
    receipt = dict(format=FORMAT, experiment_version='v1.6_half_B',
        baseline_experiment=str(baseline), experiment=str(root),
        baseline_contract_sha256=native['contract_sha256'], baseline_proof=proof,
        config=cfg, source=str(baseline / 'source/v1.0'),
        model_contract=model_contract(),
        implementation={name: digest(ROOT / name) for name in FILES},
        gpu=gpu, cuda_gib=float(cuda_gib), rss_gib=float(rss_gib),
        learning_target='v1_source_original_anchor_vs_curriculum_candidates',
        target_supervision_changed=False, L0_loss_unchanged=True,
        native_v22_objective_equivalence=False,
        support_label_definition='original_v1_candidate_index0_class1_other7_class0',
        support_training_samples=len(files),
        support_materialized_patients=cohort['materialized_cases'],
        preparation_started=False, baseline_retraining=False,
        A_automatic_start=False, combined_automatic_start=False,
        epochs=40, quality_verified=False, production_CP_started=False, nnunet_started=False)
    receipt['contract_sha256'] = canonical_hash(receipt)
    return root, receipt


def initialize(baseline, experiment, gpu, cuda_gib, rss_gib):
    root, receipt = recipe(baseline, experiment, gpu, cuda_gib, rss_gib)
    if (root / 'manifest.json').exists():
        if read(root / 'manifest.json') != receipt or read(root / 'config.json') != receipt['config']:
            raise ValueError('Existing B source/baseline/resource/recipe differs; no silent migration')
        return root, receipt
    if root.exists():
        raise FileExistsError('Existing unbound B directory is preserved')
    root.mkdir(parents=True)
    write_new(root / 'config.json', receipt['config'])
    write_new(root / 'manifest.json', receipt)
    return root, receipt


def verify(root, receipt=None):
    root = Path(root).resolve(strict=True)
    actual = read(root / 'manifest.json')
    if receipt is not None and receipt != actual:
        raise ValueError('Half-B supplied manifest differs from saved bytes')
    if actual.get('format') != FORMAT:
        raise ValueError('Half-B manifest required')
    _, expected = recipe(actual['baseline_experiment'], root,
        actual['gpu'], actual['cuda_gib'], actual['rss_gib'])
    if actual != expected or read(root / 'config.json') != expected['config']:
        raise ValueError('Half-B actual source/baseline/config/support/recipe identity changed')
    return actual
