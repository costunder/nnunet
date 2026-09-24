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
    return manifest['revision']

def load_config(path=None):
    path = Path(path or ROOT / 'config/prompt_graph_v2.json').resolve()
    cfg = read_json(path)
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
    return cfg, base

def source_identity():
    paths = sorted((ROOT / 'hiercp_v2').glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}

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
    if payload.get('format') != PIPELINE_VERSION or payload.get('complete') is not True:
        raise ValueError('A completed v2 checkpoint is required; v1 is not compatible')
    if config is not None and payload['config'] != config:
        raise ValueError('v2 checkpoint/config mismatch')
    if payload['source_identity'] != source_identity():
        raise ValueError('v2 checkpoint code identity differs; do not silently migrate it')
    from .donors import POLICY,validate_pool,reject_cross_split_duplicates
    validate_split(payload['split'])
    if payload['config'].get('donor_policy')!=POLICY:
        raise ValueError('Checkpoint donor policy mismatch')
    validate_pool(payload['donor_pool'],payload['split'])
    expected=set(payload['split']['inner_train'])
    if set(payload['patient_ids'])!=expected or set(payload['memory']['case_ids'])!=expected:
        raise ValueError('Checkpoint support/parameters include held-out patients')
    if len(payload['memory']['case_ids'])!=len(expected):
        raise ValueError('Duplicate checkpoint memory patient')
    import torch
    memory=payload['memory']; n=len(memory['embeddings']); patients=len(expected)
    if (memory['owners'].shape!=(n,) or memory['evidence'].shape!=(n,) or len(memory['descriptors'])!=n
            or memory['donor_allowed'].shape!=(patients,) or not bool(memory['donor_allowed'].all())
            or not bool(((memory['owners']>=0)&(memory['owners']<patients)).all())
            or not bool(torch.isin(memory['evidence'],torch.tensor([-1,0,1],device=memory['evidence'].device)).all())):
        raise ValueError('Malformed or contaminated checkpoint support memory')
    reject_cross_split_duplicates(payload['raw_records'],payload['split'])
    return payload


def validate_native(native, *, verify_raw=True):
    if native.get('format')!=PIPELINE_VERSION or native.get('complete') is not True:
        raise ValueError('Completed v2 native preparation is required')
    validate_split(native['split'])
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
