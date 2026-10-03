"""Replay only native v1 L0 for an explicit physical-scope learning diagnostic.

No candidate search, GT changes, new patient/prototype graph or production cache.
The native arm uses the original serialized canonical samples byte-for-byte.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from .scope_probe_support import _sha


def supervision_digest(samples):
    """Include the complete fixed candidate and nonlocal hierarchy, not L0."""
    import torch
    from torch_geometric.data import HeteroData
    h = hashlib.sha256()
    def add(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != 'cpu':
                raise ValueError('Input provenance must be checked before GPU transfer')
            tensor = value.detach().contiguous()
            h.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            # Singleton metadata tensors may legally have zero strides even
            # when contiguous(). Allocate a canonical flat CPU byte view.
            flat = torch.empty(tensor.numel(), dtype=tensor.dtype)
            flat.copy_(tensor.reshape(-1))
            h.update(flat.view(torch.uint8).numpy().tobytes())
        elif isinstance(value, HeteroData):
            add(value.to_dict())
        elif isinstance(value, dict):
            for key in sorted(value, key=lambda k: json.dumps(k, sort_keys=True)):
                h.update(json.dumps(key, sort_keys=True).encode()); add(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value: add(item)
        elif value is None or isinstance(value, (str, bool, int, float)):
            h.update(json.dumps(value, allow_nan=False).encode())
        else:
            raise TypeError(f'Unrecognized supervision provenance type: {type(value)}')
    for sample in samples:
        add({key: sample[key] for key in ('case_id', 'sample_index', 'split', 'source_component',
            'candidate_centers', 'difficulties', 'corruptions', 'candidate_regions',
            'candidate_prototypes', 'patient_graph', 'prototype_graph')})
        add([local['transform'] for local in sample['target_locals']])
    return h.hexdigest()


def load_samples(directory, train_cases, validation_cases, budget):
    import torch
    root = Path(directory).resolve(strict=True)
    manifest = json.loads((root / 'fixture_manifest.json').read_text(encoding='utf8'))
    if manifest.get('debug') is not True or manifest.get('quality_verified') is not False:
        raise ValueError('An explicit verified DEBUG input publication is required')
    samples = []
    for item in manifest['files']:
        path = (root / item['name']).resolve(strict=True)
        if path.parent != root or path.is_symlink() or _sha(path) != item['sha256']:
            raise ValueError('Original canonical sample identity/path changed')
        sample = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if (sample['split'] != item['split'] or len(sample['target_locals']) != 8
                or sample['case_id'] != item.get('case_id', sample['case_id'])
                or sample['sample_index'] != item.get('sample_index', sample['sample_index'])):
            raise ValueError('Original case/split/sample/full-eight-candidate identity changed')
        samples.append(sample); budget()
    train = [s for s in samples if s['split'] == 'train']
    val = [s for s in samples if s['split'] == 'val']
    if len(train) != train_cases or len(val) != validation_cases:
        raise ValueError('Explicit selected learning cohort was not fully loaded; no smaller fallback')
    if (len({s['case_id'] for s in train}) != train_cases
            or len({s['case_id'] for s in val}) != validation_cases
            or {s['case_id'] for s in train} & {s['case_id'] for s in val}):
        raise ValueError('Learning diagnostic requires distinct and disjoint train/validation patients')
    for row in manifest['source_records']:
        for name in ('image', 'label'):
            if _sha(Path(row[name])) != row[name + '_sha256']:
                raise ValueError('Raw CT/label no longer matches the selected original sample')
    return samples, manifest


def rebuild_scope(samples, manifest, config, adapter, margin, workers, budget, *, raw_cases=None, profile_payload=True):
    import numpy as np
    import torch
    from hiercp import local
    from hiercp.common import CasePaths, load_case, choose_source_tumor, stable_case_seed, organ_depth_mm
    before = supervision_digest(samples)
    raw, records, lock, context = ({} if raw_cases is None else raw_cases), [], threading.Lock(), threading.local()
    started = time.perf_counter()
    for row in manifest['source_records']:
        if row['case_id'] in raw:
            continue
        case = load_case(CasePaths(row['case_id'], Path(row['image']), Path(row['label'])))
        organ = (case.label == config['labels']['liver']) | (case.label == config['labels']['tumor'])
        raw[row['case_id']] = (case, organ, organ_depth_mm(organ, case.spacing))
        budget()
    raw_seconds = time.perf_counter() - started
    original_payload = local.build_patch_payload
    def measured_payload(**kwargs):
        result = original_payload(**kwargs)
        mask_count = int(np.count_nonzero(result['footprint']))
        footprint_shape = np.asarray(kwargs['footprint'].shape)
        expected_shape = footprint_shape + 2 * np.ceil(margin / kwargs['spacing']).astype(np.int64)
        expected_shape += expected_shape % 2 == 0
        if (mask_count != int(np.count_nonzero(kwargs['footprint']))
                or tuple(result['ct_norm'].shape) != tuple(expected_shape)):
            raise AssertionError('Full donor footprint or explicit bounded physical scope changed')
        with lock:
            records.append(dict(case_id=context.case_id, candidate=context.candidate,
                shape=list(result['ct_norm'].shape), voxels=int(np.prod(expected_shape)),
                spacing_mm=np.asarray(kwargs['spacing']).tolist(), full_footprint_voxels=mask_count))
        budget()
        return result
    rebuilt = []
    started = time.perf_counter()
    if profile_payload:
        local.build_patch_payload = measured_payload
    try:
        for sample in samples:
            case, organ, depth = raw[sample['case_id']]
            graph_config = adapter.configure(sample['graph_config'], margin)
            source, _, _ = choose_source_tumor(case.image, case.label,
                tumor_label=config['labels']['tumor'], selection=config['cache']['source_selection'],
                pad=config['cache']['source_pad'], rng=np.random.default_rng(stable_case_seed(
                    config['seed'], sample['case_id'], f"sample_{sample['sample_index']}")))
            if (source.component_id != sample['source_component']
                    or source.anchor_center != tuple(sample['candidate_centers'][0].tolist())):
                raise ValueError('Actual raw tumor source/anchor differs from native sample')
            context.case_id, context.candidate = sample['case_id'], 'source'
            prepared = local.prepare_local_source(case, source, full_organ_mask=organ,
                organ_depth=depth, config=graph_config, rng=np.random.default_rng(config['seed']),
                ct_clip=tuple(sample['ct_clip']))
            def target(index):
                context.case_id, context.candidate = sample['case_id'], index
                spec = SimpleNamespace(center=tuple(sample['candidate_centers'][index].tolist()),
                    rotation_matrix=sample['target_locals'][index]['transform'].numpy(),
                    scale_array=np.ones(3, dtype=np.float32))
                result = local.build_local_graph(case, source, spec, full_organ_mask=organ,
                    organ_depth=depth, config=graph_config, rng=np.random.default_rng(config['seed']),
                    ct_clip=tuple(sample['ct_clip']), prepared_source=prepared)
                if not torch.equal(result.target_local['transform'], sample['target_locals'][index]['transform']):
                    raise AssertionError('Original candidate matrix changed during bounded replay')
                return result
            with ThreadPoolExecutor(max_workers=workers) as pool:
                built = list(pool.map(target, range(8)))
            changed = {**sample, 'source_patch': torch.from_numpy(prepared.source_patch.astype(np.float16)),
                'target_patches': torch.from_numpy(np.stack([b.target_patch for b in built]).astype(np.float16)),
                'source_local': built[0].source_local, 'target_locals': [b.target_local for b in built],
                'graph_config': graph_config.to_dict()}
            rebuilt.append(changed); budget()
            print(f"Bounded learning preparation | {sample['case_id']} | margin={margin:g}mm | original 8/8 candidates", flush=True)
    finally:
        if profile_payload:
            local.build_patch_payload = original_payload
    if supervision_digest(rebuilt) != before:
        raise AssertionError('Scope replay altered original GT/candidate/patient/prototype content')
    return rebuilt, dict(raw_load_and_depth_seconds=raw_seconds,
        bounded_canonical_preparation_seconds=time.perf_counter() - started,
        ROI_records=records, supervision_sha256=before)
