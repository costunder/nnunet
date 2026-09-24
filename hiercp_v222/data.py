"""Full-cohort preparation; targets and model inputs are separate artifacts."""
from pathlib import Path
import hashlib
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage as ndi
from hiercp.common import discover_cases, load_case, stable_case_seed
from hiercp_v22.data import sources, donor_in_target_spacing, candidate_pool
from .parallel import run_jobs
from hiercp_v22.volumes import volume_memory_bound
from .contracts import read_json, write_new, safe_new_root, sha, digest_object, validate_identities, source_identity
from .inputs import context_patch
from . import PIPELINE_VERSION

def select_comparison_centers(case, positives, cfg):
    """Annotation-negative centers, not biologically impossible locations.

    Exclude identical positive anchors: bounding-box centers of nonconvex
    tumors can lie outside their tumor annotation. No distance exclusion.
    """
    mask = case.label == 1
    for row in positives:
        center = np.asarray(row['center'], dtype=np.int64)
        if center.shape != (3,) or np.any(center < 0) or np.any(center >= mask.shape):
            raise ValueError('Positive anchor outside raw annotation')
        mask[tuple(center)] = False
    eligible = np.flatnonzero(mask)
    count = cfg['comparison_centers_per_patient']
    if len(eligible) < count:
        raise ValueError(f'{case.paths.case_id}: only {len(eligible)} annotated comparison centers, need {count}; no fallback')
    rng = np.random.default_rng(stable_case_seed(cfg['seed'], case.paths.case_id, 'v222-comparison'))
    chosen = eligible[rng.choice(len(eligible), count, replace=False)]
    return dict(centers=np.column_stack(np.unravel_index(chosen, mask.shape)).tolist(),
                eligible_count=int(len(eligible)), selected_count=count,
                rule=cfg['comparison_center_rule'], biological_absence_asserted=False)


def preflight_comparison_centers(paths, inventory, split, contract, cfg, root):
    """Check every training/inner-validation case before costly patch creation.

    Reuse sampling performed during the full inventory scan, avoiding a second
    full CT read or distance transform. No outer-validation sampling is used.
    """
    centers = {}
    for c in split['outer_train']:
        audit = inventory[c]['comparison']; selected = audit['centers']
        positives = {tuple(r['center']) for r in inventory[c]['positives']}
        if (audit['rule'] != cfg['comparison_center_rule'] or len(selected) != cfg['comparison_centers_per_patient']
                or len({tuple(p) for p in selected}) != len(selected)
                or any(tuple(p) in positives for p in selected)):
            raise ValueError(f'Invalid/conflicting comparison sample: {c}')
        centers[c] = selected
    write_new(Path(root)/'comparison_preflight.json', dict(complete=True, centers=centers,
        cases=len(centers), rule=cfg['comparison_center_rule'],
        all_cases_have_requested_centers=True, center_masking=False, outer_val_used_for_training=False,
        eligible_counts={c:inventory[c]['comparison']['eligible_count'] for c in centers}))
    print({'stage':'comparison_preflight_complete','cases':len(centers),
           'centers_per_case':cfg['comparison_centers_per_patient']},flush=True)
    return centers


def inspect_case(paths, maximum, comparison_cfg=None):
    case = load_case(paths)
    components, count = ndi.label(case.label == 2, structure=ndi.generate_binary_structure(3, 1))
    sizes = np.bincount(components.ravel(), minlength=count+1)
    boxes = ndi.find_objects(components)
    positives = []; excluded = []
    for component, box in enumerate(boxes, 1):
        diameter = 2*(3*int(sizes[component])*float(np.prod(case.spacing))/(4*math.pi))**(1/3)
        if diameter > maximum:
            excluded.append({'component': component, 'equivalent_diameter_mm': diameter, 'reason': 'outside_existing_small_lesion_task'})
            continue
        position = np.argwhere(components[box] == component)+np.array([s.start for s in box])
        center = np.array([s.start+(s.stop-s.start)//2 for s in box])
        extent = float(np.linalg.norm((position-center)*case.spacing, axis=1).max()+np.linalg.norm(case.spacing)/2)
        positives.append(dict(component=component, center=center.tolist(), extent_mm=extent, equivalent_diameter_mm=diameter))
    # Hash decoded CT to catch recompression/renaming, not only file bytes.
    digest = hashlib.sha256(np.ascontiguousarray(case.image).view(np.uint8)).hexdigest()
    result = dict(case_id=paths.case_id, positives=positives, excluded=excluded,
                all_tumor_components=count, image=str(paths.image_path), label=str(paths.label_path),
                image_sha256=sha(paths.image_path), label_sha256=sha(paths.label_path), decoded_ct_sha256=digest,
                shape=list(case.shape), spacing=case.spacing.tolist())
    if comparison_cfg is not None:
        result['comparison'] = select_comparison_centers(case, positives, comparison_cfg)
    return result

def fit_contract(inventory, split, base):
    rows = [r for c in split['inner_train'] for r in inventory[c]['positives']]
    if not rows:
        raise ValueError('No annotated small-lesion support in the inner-training cohort')
    margin = float(base['graph']['context_inner_radius_mm'])
    envelope = max(r['extent_mm'] for r in rows)+margin
    return dict(format='v222_raw_ct_full_ball_v1', fit_partition='inner_train',
                fit_cases=list(split['inner_train']), fit_positive_count=len(rows),
                outer_radius_mm=envelope+float(base['graph']['context_outer_radius_mm']),
                preserved_outer_envelope_mm=envelope, center_masking=False,
                node_policy='full_uniform_physical_ball',
                node_spacing_mm=float(base['graph']['canonical_context_spacing_mm']),
                edge_radius_mm=float(base['graph']['context_edge_radius_mm']),
                patch_size=int(base['graph']['patch_size']), ct_clip=base['ct_clip'],
                rationale='Preserve previous outer FOV fitted on all inner-train lesions; remove central CT/node exclusion. No query-specific shape.')

def assert_no_duplicate_ct(inventory, split):
    seen = {}
    for partition in ('inner_train', 'inner_val', 'outer_val'):
        for case_id in split[partition]:
            digest = inventory[case_id]['decoded_ct_sha256']
            if digest in seen:
                # Duplicates inside train also corrupt episode independence and coverage.
                raise ValueError(f'Duplicate decoded CT: {seen[digest]} / {case_id}; resolve identity before preparation')
            seen[digest] = case_id

def prepare(medical, split_path, identities_path, output, cfg, base):
    split = read_json(split_path); identities = validate_identities(read_json(identities_path), split)
    paths = {p.case_id: p for p in discover_cases(Path(medical)/'Data')}
    if set(paths) != set(split['outer_train']+split['outer_val']):
        raise ValueError('Split/identity manifest must describe the complete cohort')
    root = safe_new_root(output)
    write_new(root/'started.json', dict(config=cfg, split=split, identities=identities, source_identity=source_identity()))
    inventory = {}
    def scan(c):
        return c, inspect_case(paths[c], cfg['donor_max_diameter_mm'], cfg if c in split['outer_train'] else None)
    run_jobs(list(paths), scan, lambda pair: inventory.__setitem__(*pair), cfg['preparation_workers'],
             root/'inventory_resources.json', memory_per_job=max(volume_memory_bound(p.image_path) for p in paths.values()),
             benchmark_function=scan)
    write_new(root/'inventory.json', inventory)
    assert_no_duplicate_ct(inventory, split)
    contract = fit_contract(inventory, split, base)
    violations = [dict(case=c, **r) for c in split['outer_train'] for r in inventory[c]['positives']
                  if r['extent_mm'] > contract['outer_radius_mm']]
    write_new(root/'field_of_view_audit.json', dict(contract=contract, violations=violations,
              total_components=sum(inventory[c]['all_tumor_components'] for c in split['outer_train']),
              eligible_components=sum(len(inventory[c]['positives']) for c in split['outer_train']),
              larger_lesions_retained_in_raw_segmentation=True))
    if violations:
        raise ValueError('Eligible lesion exceeds preserved physical FOV; see field_of_view_audit.json. No auto-resize or dropped samples.')
    comparison = preflight_comparison_centers(paths, inventory, split, contract, cfg, root)
    rows = []
    def build(c, publish=True):
        case = load_case(paths[c]); info = inventory[c]
        if sha(paths[c].image_path) != info['image_sha256'] or sha(paths[c].label_path) != info['label_sha256']:
            raise ValueError(f'Raw source changed after comparison preflight: {c}')
        centers = [(r['center'], 1, r['component']) for r in info['positives']]
        centers += [(point, 0, None) for point in comparison[c]]
        result = []
        directory = root/'patches'/c
        if publish: directory.mkdir(parents=True)
        for index, (center, target, component) in enumerate(centers):
            patch = context_patch(case.image, case.spacing, center, contract)
            path = directory/f'{index:05d}.npy'
            if not publish: continue
            with path.open('xb') as stream: np.save(stream, patch, allow_pickle=False)
            result.append(dict(id=f'{c}:{index}', case_id=c, patient_group=identities['cases'][c]['patient_group'],
                               component=component, center=center, target=target,
                               patch=path.relative_to(root).as_posix(), patch_sha256=sha(path)))
        return result
    completed_cases=0
    def commit_case(result):
        nonlocal completed_cases
        rows.extend(result);completed_cases+=1
        print(f'CONTEXT_CACHE {completed_cases}/{len(split["outer_train"])} cases; {len(rows)} patches',flush=True)
    run_jobs(split['outer_train'], build, commit_case, cfg['preparation_workers'], root/'patch_resources.json',
             memory_per_job=max(volume_memory_bound(paths[c].image_path) for c in split['outer_train']),
             benchmark_function=lambda c:build(c,publish=False))
    pool = [dict(case_id=c, component_id=r['component']) for c in sorted(split['inner_train']) for r in inventory[c]['positives']]
    from hiercp_v22.donors import validate_pool
    validate_pool(pool, split)
    write_new(root/'index.json', dict(format=PIPELINE_VERSION, complete=True, debug=False, config=cfg,
              base=base, split=split, identities=identities, input_contract=contract, input_contract_sha256=digest_object(contract),
              records=sorted(rows, key=lambda r: r['id']), donor_pool=pool,
              raw_records=[inventory[c] for c in split['outer_train']], source_identity=source_identity(),
              targets_separate_from_inputs=True, no_outer_val_patches=True))
    return root/'index.json'

class ContextDataset(Dataset):
    def __init__(self, index, partition):
        self.path = Path(index).resolve(); self.meta = read_json(self.path)
        if self.meta.get('format') != PIPELINE_VERSION or not self.meta.get('complete') or self.meta.get('debug'):
            raise ValueError('Complete production v2.22 context cache required')
        validate_identities(self.meta['identities'], self.meta['split'])
        if digest_object(self.meta['input_contract']) != self.meta['input_contract_sha256']:
            raise ValueError('Frozen input contract changed')
        all_rows = self.meta['records']
        if len({r['id'] for r in all_rows}) != len(all_rows):
            raise ValueError('Duplicate cached context identity')
        for row in all_rows:
            if row['case_id'] not in self.meta['split']['outer_train'] or row['target'] not in (0, 1):
                raise ValueError('Invalid case/target in training context cache')
            if row['patient_group'] != self.meta['identities']['cases'][row['case_id']]['patient_group']:
                raise ValueError('Cached context patient group changed')
        if self.meta['source_identity'] != source_identity():
            raise ValueError('Cache source differs from current implementation')
        if partition not in ('inner_train', 'inner_val'):
            raise ValueError('Outer evaluation cannot enter GNN cache')
        allowed = set(self.meta['split'][partition])
        self.rows = [r for r in self.meta['records'] if r['case_id'] in allowed]
        if set(r['case_id'] for r in self.rows) != allowed:
            raise ValueError('Missing patients in context cache')
        # Verify once before training, not in each loading iteration.
        for row in self.rows:
            if sha(self.path.parent/row['patch']) != row['patch_sha256']:
                raise ValueError('Context patch changed')
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        patch = np.load(self.path.parent/row['patch'], mmap_mode='r', allow_pickle=False)
        return torch.from_numpy(np.array(patch)), index

def support_for_query(memory, query_group):
    """Exclude the whole group before L1/L2, not only a same-patient attention edge."""
    groups = memory['patient_groups']
    distinct = sorted(set(groups)-{query_group})
    if len(distinct) < 2:
        raise ValueError('Need at least two other training-patient tasks for L2')
    device = memory['owners'].device
    remap = torch.full((len(groups),), -1, dtype=torch.long, device=device)
    lookup = {group: i for i, group in enumerate(distinct)}
    remap[:] = torch.tensor([lookup.get(group, -1) for group in groups], device=device)
    owners = remap[memory['owners']]; keep = owners >= 0
    return memory['embeddings'][keep], owners[keep], memory['classes'][keep]
