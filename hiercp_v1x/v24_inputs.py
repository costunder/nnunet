"""Explicit recipient CT/anatomy inputs for the GT-blind v2.4 experiment.

Recipient tumor annotations and P/U supervision are deliberately absent from
these objects. The organ mask is an explicit conditioning input, not a mask
recomputed from the recipient tumor annotation during a forward/cache lookup.
The unchanged annotated donor mask remains an allowed Copy-Paste source.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib
import json
from types import SimpleNamespace
import numpy as np
import torch

FORMAT = 'v24_recipient_CT_organ_no_tumor_annotation_v1'
LOCAL_FORMAT = 'v24_original_local_CT_organ_donor_footprint_v1'
QUERY_KEYS = ('id', 'case_id', 'center', 'donor_case_id', 'donor_component')


def array_digest(value):
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps([str(array.dtype), list(array.shape)], separators=(',', ':')).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def tensor_digest(value):
    """Hash actual values, including zero-sized/expanded tensors, without aliases."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(json.dumps(['tensor', str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode())
            flat = torch.empty(tensor.numel(), dtype=tensor.dtype)
            flat.copy_(tensor.reshape(-1))
            digest.update(flat.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(array_digest(item).encode())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item: visit(child)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(json.dumps(item, allow_nan=False, separators=(',', ':')).encode())
        else:
            raise TypeError('Unsupported input binding type: ' + type(item).__name__)
    visit(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class RecipientContext:
    case_id: str
    image: np.ndarray
    organ_mask: np.ndarray
    spacing: np.ndarray
    image_affine: np.ndarray

    @property
    def shape(self):
        return self.image.shape

    @property
    def paths(self):
        # Original physical geometry requires case identity, never label_path.
        return SimpleNamespace(case_id=self.case_id)

    @property
    def label(self):
        raise AttributeError('Recipient tumor annotation is not a v2.4 model input')

    def binding(self):
        return dict(format=FORMAT, case_id=self.case_id, CT_sha256=array_digest(self.image),
                    organ_sha256=array_digest(self.organ_mask), spacing=self.spacing.tolist(),
                    image_affine=self.image_affine.tolist(), recipient_tumor_GT_used=False)


def recipient_context(case_id, CT, organ_mask, spacing, image_affine):
    if not isinstance(case_id, str) or not case_id:
        raise ValueError('Explicit recipient case identity required')
    image, organ = np.asarray(CT), np.asarray(organ_mask)
    physical, affine = np.asarray(spacing), np.asarray(image_affine)
    if (image.ndim != 3 or image.dtype.kind != 'f' or not np.isfinite(image).all()
            or organ.shape != image.shape or organ.dtype != bool or not organ.any()
            or physical.shape != (3,) or not np.isfinite(physical).all() or (physical <= 0).any()
            or affine.shape != (4, 4) or not np.isfinite(affine).all()):
        raise ValueError('Finite real CT, explicit full organ mask, native spacing/affine required')
    arrays = [np.array(value, copy=True) for value in (image, organ, physical, affine)]
    for array in arrays: array.flags.writeable = False
    return RecipientContext(case_id, *arrays)


def query_inputs(rows):
    """Freeze locations/donor assignments without consulting targets/components."""
    if not rows:
        raise ValueError('Complete nonempty candidate set required')
    answers = []
    for row in rows:
        values = {key: row[key] for key in QUERY_KEYS}
        center = tuple(values['center'])
        if (len(center) != 3 or any(type(x) is not int or x < 0 for x in center)
                or type(values['donor_component']) is not int or values['donor_component'] < 1):
            raise ValueError('Native integer candidate center/donor component required')
        # Explicit lists survive JSON publication/admission without changing
        # equality. Physical builders convert to tuples at their API boundary.
        values['center'] = list(center)
        answers.append(values)
    if len({row['id'] for row in answers}) != len(answers):
        raise ValueError('Duplicate candidate observation identity')
    if len({(row['case_id'], row['donor_case_id'], row['donor_component']) for row in answers}) != 1:
        raise ValueError('One independent fixed donor per complete recipient graph required')
    return tuple(answers)


def prepare_donor(donor_case, source, *, config, seed, ct_clip):
    """Original donor branch, with actual complete annotated donor mask validation."""
    from hiercp.common import organ_depth_mm
    from hiercp_v1x.historical_patient_graph import _source
    _source(donor_case, source, 2)
    organ = np.isin(donor_case.label, (1, 2))
    local = importlib.import_module('hiercp.local')
    return local.prepare_local_source(donor_case, source, full_organ_mask=organ,
        organ_depth=organ_depth_mm(organ, donor_case.spacing), config=config,
        rng=np.random.default_rng(seed), ct_clip=ct_clip)


def build_local_record(recipient, donor_case, source, prepared, row, *, config, seed, ct_clip,
                       scope_contract):
    """Original complete physical L0 construction, with no recipient label access."""
    from hiercp.common import organ_depth_mm
    from hiercp_v22.data import donor_in_target_spacing, candidate_spec
    from hiercp_v1x import transition_v1_empty_context
    local = importlib.import_module('hiercp.local')
    spatial = importlib.import_module('hiercp.spatial')
    query = query_inputs([row])[0]
    if (query['case_id'] != recipient.case_id or query['donor_case_id'] != donor_case.paths.case_id
            or query['donor_component'] != source.component_id
            or recipient.case_id == donor_case.paths.case_id
            or any(x >= extent for x, extent in zip(query['center'], recipient.shape))):
        raise ValueError('Actual independent donor/recipient/native center binding differs')
    # Only the donor footprint is regridded; source CT/topology stay donor-native.
    target_source, target_mask = donor_in_target_spacing(source, donor_case.spacing, recipient.spacing)
    transported = replace(prepared, source_footprint=spatial.exact_source_footprint(target_source))
    transported.v1x_bounded_scope_contract = scope_contract
    spec = candidate_spec(target_source, query['center'])
    built = local.build_local_graph(recipient, target_source, spec,
        full_organ_mask=recipient.organ_mask,
        organ_depth=organ_depth_mm(recipient.organ_mask, recipient.spacing), config=config,
        rng=np.random.default_rng(seed), ct_clip=ct_clip, prepared_source=transported)
    if int(transported.source_footprint.sum()) != int(target_mask.sum()):
        raise ValueError('Native transport lost actual donor footprint voxels')
    proof = transition_v1_empty_context.validate_local_proof(built.source_local, built.target_local)
    record = dict(format=LOCAL_FORMAT, case_id=recipient.case_id,
        donor_case_id=donor_case.paths.case_id, component_id=int(source.component_id),
        observation_id=query['id'], center=list(query['center']), seed=seed,
        graph_config=config.to_dict(), scope_contract=scope_contract,
        recipient_binding=recipient.binding(),
        donor_mask_sha256=array_digest(source.full_mask),
        source_patch=torch.from_numpy(prepared.source_patch.astype(np.float16)),
        target_patch=torch.from_numpy(built.target_patch.astype(np.float16)),
        source_local=built.source_local, target_local=built.target_local,
        recipient_GT_used_in_forward=False, P_U_labels_in_forward=False,
        recipient_context_absence=proof)
    record['tensor_sha256'] = tensor_digest({key: record[key] for key in
        ('source_patch', 'target_patch', 'source_local', 'target_local')})
    return record


def materialize_pair(record, *, epoch):
    """Unchanged two-view graph sampling seed/topology; supervision is absent."""
    from hiercp.common import stable_case_seed
    from hiercp.sample import build_local_view
    from hiercp.schema import graph_config_from_dict
    if (record.get('format') != LOCAL_FORMAT or record.get('recipient_GT_used_in_forward') is not False
            or record.get('P_U_labels_in_forward') is not False or type(epoch) is not int or not 0 <= epoch <= 40):
        raise ValueError('Bound recipient-GT-free original local record required')
    actual = tensor_digest({key: record[key] for key in
        ('source_patch', 'target_patch', 'source_local', 'target_local')})
    if actual != record['tensor_sha256']:
        raise ValueError('Immutable v2.4 local tensor content changed')
    views = []
    for view in (0, 1):
        identity = f"{record['observation_id']}:{record['donor_case_id']}:{record['component_id']}:{view}:{epoch}"
        seed = stable_case_seed(record['seed'], record['case_id'], identity)
        graph = build_local_view(record['source_local'], record['target_local'],
            graph_config_from_dict(record['graph_config']), seed=seed)
        graph.transition_record_sha256 = actual
        graph.transition_observation_id = record['observation_id']
        graph.transition_recipient_case = record['case_id']
        graph.transition_donor_case = record['donor_case_id']
        graph.transition_view_id = torch.tensor([view], dtype=torch.long)
        graph.transition_epoch = torch.tensor([epoch], dtype=torch.long)
        views.append(graph)
    return tuple(views), record['source_patch'], record['target_patch']


def collate(items):
    # This original collator only combines provided tensors/views and identities.
    from hiercp_v1x.transition_v1_local import collate as original_collate
    return original_collate(items)
