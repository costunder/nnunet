"""Separate version/provenance namespace; no v1 artifacts relabelled as v2."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from . import PIPELINE_VERSION

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()

def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)

def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)

def verify_v1(root=ROOT):
    from zipfile import ZipFile
    root=Path(root)
    manifest = read_json(root / 'versions/v1/manifest.json')
    archive=root / 'versions/v1/pipeline_v1_source.zip'
    if sha(archive)!=manifest['archive_sha256']:
        raise ValueError('Preserved v1 archive changed')
    # User-requested current handoff documents evolve; their original bytes
    # remain verified in the immutable archive. Runtime/config guards stay exact.
    handoff_documents={'gpt_handoff.md','code.txt'}
    with ZipFile(archive) as preserved:
        for name in handoff_documents:
            if hashlib.sha256(preserved.read(name)).hexdigest()!=manifest['files'][name]:
                raise ValueError(f'Preserved v1 handoff differs from manifest: {name}')
    for name, digest in manifest['files'].items():
        if name not in handoff_documents and sha(root / name) != digest:
            raise ValueError(f'v1 source changed; preserve/review dependency before v2: {name}')
    predecessor=root/'versions/v2.1/before_v22_20260920'
    preserved=read_json(predecessor/'manifest.json')
    if sha(predecessor/'source.zip')!=preserved['archive_sha256']:
        raise ValueError('Preserved v2.1 archive changed')
    for name,digest in preserved['files'].items():
        if name.startswith('hiercp_v2/') or name in ('run_v2.py','config/prompt_graph_v2.json'):
            if sha(root/name)!=digest:raise ValueError(f'v2.1 source changed: {name}')
    return manifest['revision']

def load_config(path=None):
    path = Path(path or ROOT / 'config/prompt_graph_v221.json').resolve()
    cfg = read_json(path)
    validate_encoder_config(cfg)
    from .donors import POLICY,CONTEXT
    if cfg.get('donor_policy')!=POLICY or cfg.get('gnn_context_schedule')!=CONTEXT:
        raise ValueError('Shared training-only donor policy required; rebuild inconsistent old caches')
    if cfg['format'] != PIPELINE_VERSION or cfg.get('task_definition')!='actual_patient_id' or 'task_views' in cfg:
        raise ValueError('Actual patient tasks required; view-only v2 is incompatible')
    if cfg['support_policy'] != 'all_actual_patient_contexts_cross_patient_T_only':
        raise ValueError('Unsupported support leakage policy')
    if cfg['alignment_target'] != 'cross_patient_context_correspondence_and_positive_transfer':
        raise ValueError('L2 must align provisional task labels, not anatomical prototypes')
    if cfg['label_definition'] != 'independent_patient_random_trainable_latent_nodes':
        raise ValueError('Unsupported provisional label semantics')
    base_path = ROOT / cfg['base_config']
    base = read_json(base_path)
    if cfg['gnn_epochs'] != 40 or cfg['nnunet_epochs'] != 250 or cfg['candidate_count'] != 128:
        raise ValueError('This version preserves the full 40/250/128 experiment contract')
    if cfg['task_layers'] != 2 or cfg['alignment_layers'] != 2 or cfg['label_count'] < 2:
        raise ValueError('Explicit full-scale task/alignment configuration required')
    if base['model']['hidden_dim'] != 128 or base['model']['local_layers'] != 3:
        raise ValueError('Unexpected L0 model scale')
    if cfg['seed']!=base['seed'] or cfg['cp_probability']!=.5 or cfg['donor_max_diameter_mm']!=20:
        raise ValueError('Seed/CP probability/donor diameter must match this experiment contract')
    if cfg['selection']!='cross_patient_positive_transport_argmax' or cfg['label_initialization']!='independent_seeded_random_trainable':
        raise ValueError('Unsupported v2 scoring/label initialization')
    if cfg['gradient_accumulation']!=1 or cfg['batch_size']!='auto' or cfg['num_workers']!='auto':
        raise ValueError('This trainer requires measured physical batches/workers and accumulation=1')
    if cfg['temperature']<=0 or not 0<cfg['max_vram_fraction']<1:
        raise ValueError('Invalid clustering/temperature/resource settings')
    if (base['model']['dense_base_channels'],base['model']['dense_feature_dim'],base['model']['heads'],base['graph']['patch_size'])!=(12,32,4,48):
        raise ValueError('Preserve full CNN/GNN width and input resolution')
    return cfg, base

def source_identity():
    paths = sorted(set((ROOT / 'hiercp_v221').glob('*.py')) | set((ROOT/'hiercp').glob('*.py')) |
                   set((ROOT/'custom_trainers').glob('*.py')) |
                   {ROOT/'run_v221.py',ROOT/'config/prompt_graph_v221.json',ROOT/'config/train.json',
                    ROOT/'tools/online_raw_bank_preparation.py'})
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def validate_encoder_config(cfg):
    expected={'format':PIPELINE_VERSION,'release':'v2.21','version_status':'in_progress',
              'implementation_scope':'relation_contract_only','tumor_interior_nodes':False,
              'cnn_input_channels':1,'node_auxiliary_feature_dim':0,'edge_attribute_dim':0,
              'evidence_contract':'per_data_label_T_relation_F_nonrelation_U_unknown',
              'geometry_contract':'candidate_filter_only_never_relation_F'}
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('v2.21 relation contract mismatch; legacy geometry-F is not accepted')


def require_training_objective():
    raise RuntimeError('v2.21 is in progress: relation construction is implemented, '
                       'but the medical task/class definition and full L1/L2 integration are not complete. '
                       'A random embedding does not define a class. Legacy v2.2 is not a v2.21 trained model.')


def safe_new_root(path):
    path = Path(path).resolve()
    if path == ROOT or not path.name or path.exists():
        raise ValueError(f'A new, nonexistent output directory is required: {path}')
    path.mkdir(parents=True)
    return path

def validate_split(split):
    required = ('inner_train', 'inner_val', 'outer_train', 'outer_val')
    for key in required:
        ids = split[key]
        if any(not isinstance(c,str) or not c or c in ('.','..') or '/' in c or chr(92) in c for c in ids):
            raise ValueError('Patient IDs must be nonempty local names')
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or not ids:
            raise ValueError(f'Nonempty, unique patient IDs required: {key}')
    a,b,c,d = [set(split[k]) for k in required]
    if a & b or c & d or a | b != c:
        raise ValueError('Inner partitions must exactly partition outer train; outer validation is disjoint')

def validate_checkpoint(payload, config=None):
    if payload.get('format')!=PIPELINE_VERSION:
        raise ValueError('Incompatible checkpoint: rebuild CT-only L0/L1, never relabel r1/r2 weights')
    require_training_objective()


def validate_native(native, *, verify_raw=True):
    if native.get('format')!=PIPELINE_VERSION or native.get('complete') is not True:
        raise ValueError('Completed v2 native preparation is required')
    validate_split(native['split'])
    if 'preprocessing_origin' in native:
        origin=native['preprocessing_origin']
        if sha(origin['path'])!=origin['sha256']:raise ValueError('Reused preprocessing receipt changed')
    if sha(native['plans'])!=native['plans_sha256'] or sha(Path(native['preprocessed'])/'splits_final.json')!=native['splits_sha256']:
        raise ValueError('Native plans/split changed since preparation')
    records={r['case_id']:r for r in native['raw_records']}
    if set(records)!=set(native['split']['outer_train']+native['split']['outer_val']):
        raise ValueError('Native raw inventory is incomplete')
    if verify_raw:
        for case,row in records.items():
            for key,sub,suffix in (('image','imagesTr','_0000.nii.gz'),('label','labelsTr','.nii.gz')):
                if sha(Path(native['raw'])/sub/(case+suffix))!=row[key+'_sha256']:
                    raise ValueError(f'Native copied {key} changed: {case}')
    return native
