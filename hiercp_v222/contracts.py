"""Version, patient-group and provenance checks used by every v2.22 stage."""
from pathlib import Path
import hashlib
import json
from hiercp_v221.contracts import ROOT, read_json, write_new, sha, safe_new_root, validate_split, verify_v1
from . import PIPELINE_VERSION

CASE_BENCHMARK_IDENTITY = 'hiercp_public_case_benchmark_v1'

def load_config(path=None):
    cfg = read_json(path or ROOT / 'config/prompt_graph_v222.json')
    base = read_json(ROOT / cfg['base_config'])
    expected = dict(format=PIPELINE_VERSION, release='v2.22', seed=42, comparison_centers_per_patient=128,
                    label_count=2, task_layers=2,
                    alignment_layers=2, gnn_epochs=40, nnunet_epochs=250, candidate_count=128,
                    cp_probability=0.8, donor_max_diameter_mm=20.0, gradient_accumulation=1,
                    comparison_center_rule='annotated_liver_center_excluding_exact_positive_anchors_v1',
                    input_policy='raw_CT_full_candidate_ball_CNN_PPR_Astar_sampled_graph',
                    cluster_method='cosine_average_linkage',
                    cluster_selection='maximum_positive_silhouette_all_non_singleton_cuts_else_one',
                    cluster_refresh='detached_eval_L1_once_per_query_patient_episode',
                    cluster_min_patients_per_submode=2, alignment_loss_weight=1.0,
                    prototype_score='patient_mass_weighted_logsumexp_within_observation_class')
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('v2.22 task/scale differs from the approved configuration')
    from .sampling import validate_sampler
    validate_sampler(cfg['graph_sampling'])
    m = base['model']
    if (m['hidden_dim'], m['local_layers'], m['heads'], m['dense_base_channels'],
            m['dense_feature_dim'], base['graph']['patch_size']) != (128, 3, 4, 12, 32, 48):
        raise ValueError('Full CNN/GNN scale must be preserved')
    return cfg, base

def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def source_identity():
    paths = [ROOT/'run_v222.py', ROOT/'config/prompt_graph_v222.json', ROOT/'config/train.json',
             ROOT/'comparison_randomness.py']
    for directory in ('hiercp_v222', 'hiercp', 'custom_trainers', 'hiercp_v22', 'hiercp_v221', 'hiercp_v2'):
        paths.extend((ROOT/directory).glob('*.py'))
    paths.append(ROOT/'tools/online_raw_bank_preparation.py')
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))}

def validate_identities(identity, split):
    validate_split(split)
    cases = set(split['outer_train']) | set(split['outer_val'])
    benchmark = identity.get('format') == CASE_BENCHMARK_IDENTITY
    if identity.get('format') not in ('hiercp_patient_identity_v1',CASE_BENCHMARK_IDENTITY) or set(identity.get('cases', {})) != cases:
        raise ValueError('An explicit complete patient-group identity manifest is required')
    if benchmark and (identity.get('independence_scope') != 'published_case_only'
                      or identity.get('patient_independence_verified') is not False
                      or identity.get('annotation_scope') != 'provided_masks_may_omit_lesions'):
        raise ValueError('Public case benchmark must state its patient/annotation limitations')
    seen = {}
    for partition in ('inner_train', 'inner_val', 'outer_val'):
        for case in split[partition]:
            row = identity['cases'][case]
            if benchmark:
                if (row.get('patient_group') != 'case:'+case or row.get('identity_basis') != 'published_case_id_only'
                        or 'annotation_complete' not in row or row['annotation_complete'] is not None):
                    raise ValueError(f'Case benchmark cannot assert verified patients or complete annotations: {case}')
            elif not row.get('patient_group') or not row.get('identity_basis') or row.get('annotation_complete') is not True:
                raise ValueError(f'Missing patient identity/annotation provenance: {case}')
            group = row['patient_group']
            if group in seen and seen[group] != partition:
                raise ValueError(f'Patient-group leakage: {group}')
            seen[group] = partition
    return identity

def validate_checkpoint(payload):
    from . import TRAINING_READY, DESIGN_BLOCK_REASON
    if not TRAINING_READY:
        raise ValueError(DESIGN_BLOCK_REASON)
    import torch
    from .inputs import validate_contract
    if payload.get('format') != PIPELINE_VERSION or payload.get('debug') is not False:
        raise ValueError('Production v2.22 checkpoint required; legacy/debug weights rejected')
    if payload.get('completed_epochs') != payload['config']['gnn_epochs']:
        raise ValueError('Incomplete GNN training is not a production checkpoint')
    validate_identities(payload['identities'], payload['split'])
    if payload['source_identity'] != source_identity():
        raise ValueError('Checkpoint source provenance changed')
    if not set(payload['memory']['case_ids']) <= set(payload['split']['inner_train']):
        raise ValueError('Non-training patient in frozen support memory')
    memory = payload['memory']; cases = memory['case_ids']
    if len(cases) != len(set(cases)) or memory['patient_groups'] != [payload['identities']['cases'][c]['patient_group'] for c in cases]:
        raise ValueError('Support patient identity mismatch')
    owners, classes, embeddings = memory['owners'], memory['classes'], memory['embeddings']
    if owners.dtype != torch.long or classes.dtype != torch.long or owners.shape != classes.shape or embeddings.shape != (len(owners), 128):
        raise ValueError('Invalid frozen support tensor schema')
    if not len(owners) or bool(((owners < 0) | (owners >= len(cases))).any()) or not torch.isfinite(embeddings).all():
        raise ValueError('Invalid support ownership/values')
    if bool(((classes < 0) | (classes > 1)).any()):
        raise ValueError('Unknown support cannot be used as a known negative')
    validate_contract(payload['input_contract'])
    if set(payload['input_contract']['fit_cases']) != set(payload['split']['inner_train']):
        raise ValueError('Physical FOV was not fitted on exactly the training partition')
    return payload

def validate_native(native, *, verify_raw=True):
    from hiercp_v221.contracts import validate_native as previous
    from hiercp_v221 import PIPELINE_VERSION as old_format
    if native.get('format') != PIPELINE_VERSION:
        raise ValueError('v2.22 native metadata required')
    previous(dict(native, format=old_format), verify_raw=verify_raw)
    if set(native['planning_patient_ids']) != set(native['split']['outer_train']):
        raise ValueError('nnU-Net planning must use outer-training patients only')
    return native
