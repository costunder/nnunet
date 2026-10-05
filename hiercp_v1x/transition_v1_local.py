"""D input block: actual archived v1 L0 on the unchanged native P/U cohort.

This is NOT the later CTOnlyEncoder/V1LocalEncoder. All six original semantic
roles, 16 handcrafted inputs, five-channel fixed48 dense CNN, three original
heterogeneous GAT layers and original shell/role/pair readouts are retained.
The recipient footprint is erased exactly as in original build_local_graph;
native observed/unobserved labels stay external and are never forward inputs.

An external donor is prepared in its own CT, mask, anchor and native spacing.
Only its complete virtual recipient footprint is regridded. Source node mm
positions remain donor-native; both physical frames and the explicit anchor
transport are audited. One observation owns two original sampled views, run
as one disjoint graph batch, with dense maps shared across those two views.
The actual fused128 view mean is the native upper input. The exact archived
six-key cosine consistency is exposed for the D runner's original .1 term.

No observed GT, candidate suitability label, truncation, query cap, dummy
feature, new trainable bridge, or CT-only substitution is introduced here.
The provider owns the signed full14102 assignments/all observed P/128 U
coverage. Any missing original mandatory geometry raises instead of skipping.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
import copy
import hashlib
import importlib
import json
from pathlib import Path
import re
import weakref

import numpy as np
import torch
from torch import Tensor, nn


FORMAT = "native_PU_actual_original_v1_local_two_view_10mm_v1"
ENCODER_ID = "actual_archived_v1_six_role_local_encoder_on_native_PU"
ROOT = Path(__file__).resolve().parents[1]
_RUNTIME = None
_LEAF_HASHES = {}
_ROLE_NAMES = ("tumor_surface", "tumor_interior", "source_context",
               "source_liver_surface", "target_context", "target_liver_surface")
_FIELD_NAMES = ("tumor", "source_context", "target_context", "source_relation",
                "target_relation", "source_c0", "source_c1", "source_c2",
                "target_c0", "target_c1", "target_c2", "fused")
_PROVENANCE_HASHES = ("input_inventory_sha256", "assignment_sha256", "donor_image_sha256",
                      "donor_label_sha256", "recipient_image_sha256", "recipient_label_sha256")


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def local_contract() -> dict:
    from .contracts import V1_ARCHIVE_SHA256
    return dict(format=FORMAT, encoder=ENCODER_ID, original_archive_sha256=V1_ARCHIVE_SHA256,
        width=128, heads=4, local_layers=3, dense_base_channels=12, dense_feature_dim=32,
        dense_execution_chunk_size=4, dense_input_shape=[5, 48, 48, 48], dropout=.1,
        checkpoint_dense_encoder=False, checkpoint_local_blocks=False,
        activation_storage="native endpoint retained; actual explicit model flags recorded in checkpoint extra_state",
        roles=list(_ROLE_NAMES), handcrafted_dim=16, ROI_margin_mm=10.,
        original_coordinate_normalization_mm=28., original_shells_mm=[4., 12., 28.],
        source_CT="actual donor CT; original normalization and five channels",
        target_CT="actual recipient CT with original donor-footprint target erasure",
        footprint="full original donor mask regridded about explicit anchor to recipient spacing",
        graph="actual original canonical radius edges and native two-hop sampled views; no new caps",
        views_per_observation=2, dense_maps_shared_between_views=True,
        bridge="mean of the two genuine original semantic fused128 vectors",
        view_consistency="actual original six-key cosine mean; runner adds original weight0.1",
        view_consistency_weight=.1, query_GT_in_forward=False,
        observed_unobserved_labels="unchanged native labels supplied only to external objective",
        native_total_assignment_count=14102, native_comparisons_per_recipient=128,
        native_equivalence=False, original_task_equivalence=False,
        differences=["native all-observed-P/128U task replaces original anchor/curriculum task",
                     "external training donor/recipient replace original same-case placement",
                     "native upper consumes one real fused128 view mean",
                     "original erasure/five-channel/graph input block remains distinct from native raw CT CNN"])


def source_identity() -> dict:
    return dict(contract=local_contract(), module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                contract_sha256=hashlib.sha256(_json(local_contract())).hexdigest())


def _runtime(*, expected_snapshot_root=None, scope_contract=None):
    """Require an already activated byte-exact archive and installed10mm scope."""
    global _RUNTIME
    from . import bounded_scope
    active = bounded_scope._ACTIVE
    if not isinstance(active, dict) or float(active.get("margin_mm", -1)) != 10.:
        raise RuntimeError("Activate the actual original archive and install explicit10mm scope first")
    identity = active["contract_sha256"]
    if scope_contract is not None and identity != scope_contract:
        raise ValueError("D encoder/record and active10mm scope identities differ")
    if _RUNTIME is None:
        local = importlib.import_module("hiercp.local")
        snapshot = Path(local.__file__).resolve().parents[1]
        if expected_snapshot_root is not None and snapshot != Path(expected_snapshot_root).resolve(strict=True):
            raise ValueError("D requires the calling worker's explicit original snapshot root")
        bounded_scope._verify_snapshot(snapshot)
        schema, sample, model, spatial = (importlib.import_module("hiercp." + name)
                                         for name in ("schema", "sample", "model", "spatial"))
        if tuple(schema.LOCAL_NODE_TYPES) != _ROLE_NAMES:
            raise ValueError("Actual original six-role schema required")
        _RUNTIME = dict(local=local, schema=schema, sample=sample, model=model, spatial=spatial,
                        snapshot=snapshot, scope=copy.deepcopy(active))
    elif (_RUNTIME["scope"] != active or expected_snapshot_root is not None
          and _RUNTIME["snapshot"] != Path(expected_snapshot_root).resolve(strict=True)):
        raise RuntimeError("Use a fresh isolated worker for a different original input contract")
    return _RUNTIME


def _config(base):
    runtime = _runtime()
    if not isinstance(base, Mapping) or base.get("seed") != 42:
        raise ValueError("Original seed42 base configuration required")
    model = base.get("model", {})
    expected = dict(hidden_dim=128, heads=4, local_layers=3, dropout=.1,
                    dense_base_channels=12, dense_feature_dim=32, dense_batch_size=4)
    if any(model.get(key) != value for key, value in expected.items()):
        raise ValueError("Original128D/4heads/GAT3/CNN12/32/chunk4/dropout.1 input block required")
    config = runtime["schema"].graph_config_from_dict(dict(base["graph"]))
    runtime["local"]._require_full_graph(config)
    from .bounded_scope import _scope_config
    _scope_config(config, 10.)
    clip = tuple(base["ct_clip"])
    if clip != (-200., 250.):
        raise ValueError("Original CT clipping/normalization contract required")
    return config, clip


def _hash_array(value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((array.dtype.str, array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _file_sha(path, size, mtime_ns):
    # The signature is part of the key. A changed source file is rehashed and
    # then refused against the signed provider digest, never silently trusted.
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_files(case) -> dict:
    result = {}
    for name in ("image", "label"):
        path = Path(getattr(case.paths, name + "_path")).resolve(strict=True)
        stat = path.stat()
        result[name + "_path"] = str(path)
        result[name + "_sha256"] = _file_sha(str(path), stat.st_size, stat.st_mtime_ns)
    return result


def _case_geometry(case, organ, depth) -> dict:
    arrays = (case.image, case.label, organ, depth, case.spacing, case.image_affine, case.label_affine)
    immutable = all(isinstance(value, np.ndarray) and not value.flags.writeable for value in arrays)
    signature = tuple((id(value), tuple(value.shape), value.dtype.str, value.flags.writeable) for value in arrays)
    saved = getattr(case, "_transition_geometry_checked", None)
    if immutable and saved is not None and saved[0] == signature:
        return copy.deepcopy(saved[1])
    from hiercp_v222.placement import validate_grid
    validate_grid(case)
    organ, depth = np.asarray(organ), np.asarray(depth)
    if (organ.dtype != bool or organ.shape != case.image.shape or depth.shape != organ.shape
            or not np.array_equal(organ, np.isin(case.label, [1, 2]))
            or not organ.any() or not np.isfinite(case.image).all()
            or not np.isfinite(depth).all() or (depth < 0).any()):
        raise ValueError("Real aligned full CT/organ/depth arrays required; no cropped annotation proxy")
    result = dict(case_id=case.paths.case_id, shape=list(case.image.shape),
                spacing=list(map(float, case.spacing)),
                image_affine=np.asarray(case.image_affine, dtype=float).tolist(),
                label_affine=np.asarray(case.label_affine, dtype=float).tolist())
    if immutable:
        case._transition_geometry_checked = signature, copy.deepcopy(result)
    return result


def _source_audit(case, source) -> dict:
    full = np.asarray(source.full_mask)
    patch = np.asarray(source.patch_mask)
    slices = source.patch_slices
    if (full.dtype != bool or full.shape != case.image.shape or patch.dtype != bool or patch.ndim != 3
            or len(slices) != 3 or not full.any() or not patch.any()
            or int(full.sum()) != int(source.voxel_count) or int(patch.sum()) != int(source.voxel_count)
            or not np.array_equal(patch, full[slices])
            or not np.all(np.asarray(case.label)[full] == 2)
            or not np.array_equal(np.asarray(source.patch_image), np.asarray(case.image)[slices])):
        raise ValueError("Complete actual donor tumor mask/CT patch required; no cropped or recipient-derived donor")
    origin = np.array([sl.start for sl in slices], dtype=np.int64)
    anchor = np.asarray(source.anchor_center)
    expected = origin + np.asarray(patch.shape) // 2
    if (anchor.shape != (3,) or not np.issubdtype(anchor.dtype, np.integer)
            or not np.array_equal(anchor, expected)):
        raise ValueError("Original Copy-Paste shape//2 donor anchor and real donor crop origin must agree")
    physical_anchor = np.asarray(case.image_affine) @ np.r_[anchor, 1.]
    return dict(donor=case.paths.case_id, component=int(source.component_id),
                original_anchor_native_ijk=anchor.tolist(), original_anchor_world_mm=physical_anchor[:3].tolist(),
                original_patch_origin_native_ijk=origin.tolist(), original_patch_shape=list(patch.shape),
                original_full_mask_sha256=_hash_array(full), original_patch_mask_sha256=_hash_array(patch),
                original_patch_CT_sha256=_hash_array(source.patch_image), original_mask_voxels=int(full.sum()),
                donor_spacing=list(map(float, case.spacing)), donor_affine=np.asarray(case.image_affine).tolist())


def _tensor_signature(value):
    if isinstance(value, Tensor):
        return (id(value), value._version, tuple(value.shape), str(value.dtype), str(value.device))
    if isinstance(value, np.ndarray):
        return _hash_array(value)
    if isinstance(value, Mapping):
        return tuple((repr(key), _tensor_signature(item)) for key, item in sorted(value.items(), key=lambda x: repr(x[0]))
                     if key != "_transition_checked")
    if isinstance(value, (tuple, list)):
        return tuple(_tensor_signature(item) for item in value)
    return value


@dataclass
class PreparedDonor:
    original: object
    source_patch: Tensor
    audit: dict
    files: dict
    scope_contract: str
    graph_config: dict
    signature: object


def prepare_donor(case, source, organ, depth, base) -> PreparedDonor:
    """Prepare donor nodes ONLY in actual donor CT/mask/anchor/native spacing."""
    runtime = _runtime()
    config, clip = _config(base)
    _case_geometry(case, organ, depth)
    audit = _source_audit(case, source)
    prepared = runtime["local"].prepare_local_source(case, source, full_organ_mask=organ,
        organ_depth=depth, config=config, rng=np.random.default_rng(base["seed"]), ct_clip=clip)
    if int(prepared.source_footprint.sum()) != audit["original_mask_voxels"]:
        raise RuntimeError("Original donor canonical footprint lost occupied voxels")
    patch = torch.from_numpy(prepared.source_patch.astype(np.float16))
    signature = _tensor_signature((prepared.source_footprint, prepared.source_patch,
                                   prepared.canonical_nodes, prepared.canonical_edges))
    return PreparedDonor(prepared, patch, audit, _case_files(case), runtime["scope"]["contract_sha256"],
                         config.to_dict(), signature)


def _provenance(value, *, donor_files, recipient_files):
    required = {"observation_id", "donor_group", "recipient_group", "debug", *_PROVENANCE_HASHES}
    if not isinstance(value, Mapping) or required - set(value):
        raise ValueError("Explicit native observation/assignment/raw-file provenance required")
    if (any(not isinstance(value[key], str) or not value[key] for key in
            ("observation_id", "donor_group", "recipient_group"))
            or value["donor_group"] == value["recipient_group"] or type(value["debug"]) is not bool
            or any(not isinstance(value[key], str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None
                   for key in _PROVENANCE_HASHES)):
        raise ValueError("Signed independent-patient native observation provenance required")
    for branch, files in (("donor", donor_files), ("recipient", recipient_files)):
        for kind in ("image", "label"):
            if value[branch + "_" + kind + "_sha256"] != files[kind + "_sha256"]:
                raise ValueError("D raw CT/label source differs from signed native assignment: " + branch)
    # P/U class and any other caller fields never become graph/neural metadata.
    return {key: copy.deepcopy(value[key]) for key in required}


def _physical_transport(donor_audit, target, center, transform):
    donor_spacing = np.asarray(donor_audit["donor_spacing"], dtype=float)
    donor_affine = np.asarray(donor_audit["donor_affine"], dtype=float)
    target_spacing = np.asarray(target.spacing, dtype=float)
    target_affine = np.asarray(target.image_affine, dtype=float)
    donor_frame = donor_affine[:3, :3] / donor_spacing[None]
    target_frame = target_affine[:3, :3] / target_spacing[None]
    matrix = target_frame @ transform @ np.linalg.inv(donor_frame)
    from_anchor = np.asarray(donor_audit["original_anchor_world_mm"])
    to_anchor = (target_affine @ np.r_[center, 1.])[:3]
    world = np.eye(4)
    world[:3, :3] = matrix
    world[:3, 3] = to_anchor - matrix @ from_anchor
    if not np.allclose(world @ np.r_[from_anchor, 1.], np.r_[to_anchor, 1.], atol=1e-7):
        raise RuntimeError("Donor-to-recipient physical anchor transport failed")
    return dict(axis_order="native_ijk", graph_relative_coordinates="native_axis_mm",
                donor_relative_mm_to_recipient_relative_mm=transform.tolist(),
                donor_world_to_recipient_world_mm=world.tolist(),
                recipient_anchor_native_ijk=list(map(int, center)), recipient_anchor_world_mm=to_anchor.tolist(),
                convention="actual native CP native-axis transport; explicit world frame audit; no hidden reorientation")


def _graph_hash(record):
    # Same content equation as native record_binding.graph_hash, with weak,
    # version-checked leaf memoization. Shared donor million-edge tables are
    # hashed once per tensor version rather than once per recipient candidate.
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, Tensor):
            if value.device.type != "cpu":
                raise ValueError("Canonical D content binding requires CPU tensors")
            key, signature = id(value), _tensor_signature(value)
            cached = _LEAF_HASHES.get(key)
            if cached is not None and cached[0]() is value and cached[1] == signature:
                leaf = cached[2]
            else:
                leaf = _hash_array(value.detach().numpy())
                def remove(reference, key=key):
                    existing = _LEAF_HASHES.get(key)
                    if existing is not None and existing[0] is reference:
                        _LEAF_HASHES.pop(key)
                _LEAF_HASHES[key] = weakref.ref(value, remove), signature, leaf
            digest.update(leaf.encode())
        elif isinstance(value, np.ndarray):
            digest.update(_hash_array(value).encode())
        elif isinstance(value, Mapping):
            for key in sorted(value, key=repr):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(str(len(value)).encode())
            for item in value:
                visit(item)
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
        digest.update(b"\0")
    visit({key: value for key, value in record.items() if key not in ("content_binding", "_transition_checked")})
    return digest.hexdigest()


def pair_record(target, source, donor_spacing, prepared: PreparedDonor, center, organ, depth, base,
                *, donor_id, provenance, placement=None) -> dict:
    """Regrid full donor footprint; build original erased recipient graph.

    ``source`` is the ORIGINAL donor source, never a recipient tumor/anchor.
    No query observation class is accepted or used in graph construction.
    """
    runtime = _runtime()
    config, clip = _config(base)
    recipient_geometry = _case_geometry(target, organ, depth)
    if (not isinstance(prepared, PreparedDonor) or prepared.scope_contract != runtime["scope"]["contract_sha256"]
            or prepared.graph_config != config.to_dict() or donor_id != prepared.audit["donor"]
            or int(source.component_id) != prepared.audit["component"]
            or not np.array_equal(donor_spacing, prepared.audit["donor_spacing"])
            or not np.array_equal(source.anchor_center, prepared.audit["original_anchor_native_ijk"])
            or _hash_array(source.patch_mask) != prepared.audit["original_patch_mask_sha256"]
            or _hash_array(source.patch_image) != prepared.audit["original_patch_CT_sha256"]
            or prepared.signature != _tensor_signature((prepared.original.source_footprint,
                prepared.original.source_patch, prepared.original.canonical_nodes, prepared.original.canonical_edges))):
        raise ValueError("External original donor identity/spacing/anchor/prepared canonical content changed")
    center = np.asarray(center)
    if (center.shape != (3,) or not np.issubdtype(center.dtype, np.integer)
            or ((center < 0) | (center >= target.image.shape)).any()):
        raise ValueError("Actual recipient-native integer observation center required")
    files = _case_files(target)
    bound = _provenance(provenance, donor_files=prepared.files, recipient_files=files)
    from hiercp_v22.data import donor_in_target_spacing, candidate_spec
    from hiercp_v222.placement import placement_spec
    target_source, target_mask = donor_in_target_spacing(source, np.asarray(donor_spacing), np.asarray(target.spacing))
    expected = placement_spec(target, target_source, center, donor_id,
                             forward_mm=None if placement is None else placement.transform, graph_config=config)
    if placement is not None and expected.metadata() != placement.metadata():
        raise ValueError("Caller placement differs from the complete donor/recipient native transport")
    placement = expected
    # Source branch remains the original donor nodes/features/spacing. Only the
    # authoritative footprint used to construct the target is recipient-regridded.
    regridded = replace(prepared.original, source_footprint=runtime["spatial"].exact_source_footprint(target_source))
    regridded.v1x_bounded_scope_contract = prepared.scope_contract
    spec = candidate_spec(target_source, center)
    transform = np.asarray(placement.transform, dtype=np.float32)
    if not np.array_equal(transform, np.eye(3, dtype=np.float32)):
        # Native observations and comparison banks use identity placement.
        # Unsupported pre-score transforms must not silently enter original
        # CandidateSpec identity geometry. Augmentation is a later paste action.
        raise ValueError("D observed/comparison graph requires the native identity placement transform")
    transformed, graph_transform = runtime["local"]._transform_footprint(
        regridded.source_footprint, spec, spacing=target.spacing, config=config)
    if (not np.array_equal(transformed, placement.mask) or not np.array_equal(graph_transform, transform)
            or not np.array_equal(target_mask, target_source.patch_mask)):
        raise RuntimeError("Original graph and native CP complete recipient footprint disagree")
    built = runtime["local"].build_local_graph(target, target_source, spec, full_organ_mask=organ,
        organ_depth=depth, config=config, rng=np.random.default_rng(base["seed"]),
        ct_clip=clip, prepared_source=regridded)
    if not np.array_equal(built.target_local["transform"].numpy(), transform):
        raise RuntimeError("Original cross-role graph changed the native physical transform")
    # Original build reports the authoritative TARGET-grid footprint count in
    # its generic source payload. Replace that audit field with the actual donor
    # count while retaining every original source node/edge/feature object.
    source_local = {**built.source_local, "footprint_voxels": prepared.audit["original_mask_voxels"]}
    native_shape = runtime["spatial"].adaptive_native_shape(transformed.shape, target.spacing, config,
                                      center_liver_depth_mm=float(depth[tuple(center)]))
    audit = dict(source=copy.deepcopy(prepared.audit), recipient=recipient_geometry,
                 source_files=copy.deepcopy(prepared.files), recipient_files=files,
                 placement=placement.metadata(), physical_transport=_physical_transport(prepared.audit, target, center, transform),
                 recipient_regridded_footprint_voxels=int(transformed.sum()),
                 recipient_regridded_footprint_sha256=_hash_array(transformed),
                 recipient_native_ROI_shape=list(native_shape),
                 source_native_footprint_shape=list(prepared.original.source_footprint.shape),
                 source_occupied_voxels_preserved=True, recipient_occupied_voxels_preserved=True,
                 target_erasure=True, observed_GT_used=False,
                 original_build_local_graph_used=True, complete_original_radius_topology=True)
    record = dict(format=FORMAT, encoder_id=ENCODER_ID, case_id=target.paths.case_id,
        donor_case_id=donor_id, component_id=int(source.component_id), center=center.tolist(), seed=base["seed"],
        graph_config=config.to_dict(), scope_contract=prepared.scope_contract, input_provenance=bound,
        target_erasure=True, views_per_observation=2, audit=audit,
        source_patch=prepared.source_patch, target_patch=torch.from_numpy(built.target_patch.astype(np.float16)),
        source_local=source_local, target_local=built.target_local)
    record["content_binding"] = dict(format=FORMAT, graph_sha256=_graph_hash(record))
    return record


def validate_record(record):
    runtime = _runtime()
    if (not isinstance(record, dict) or record.get("format") != FORMAT or record.get("encoder_id") != ENCODER_ID
            or record.get("target_erasure") is not True or record.get("views_per_observation") != 2
            or record.get("scope_contract") != runtime["scope"]["contract_sha256"]
            or not isinstance(record.get("content_binding"), dict)
            or record["content_binding"].get("format") != FORMAT):
        raise ValueError("D requires a new explicitly bound original-input/two-view record")
    signature = _tensor_signature(record)
    if record.get("_transition_checked") == signature:
        return
    if record["content_binding"].get("graph_sha256") != _graph_hash(record):
        raise ValueError("D canonical CT/graph/assignment content changed")
    for name in ("source_patch", "target_patch"):
        value = record.get(name)
        if (not isinstance(value, Tensor) or value.device.type != "cpu" or value.shape != (5, 48, 48, 48)
                or value.dtype != torch.float16 or not torch.isfinite(value).all()):
            raise ValueError("Exact original finite host-float16 five-channel48 payload required")
    # In-object version signatures avoid rehashing millions of immutable edges
    # on every materialization; reload/replacement/mutation forces a full check.
    record["_transition_checked"] = signature


def materialize(record, *, epoch=0, view=0):
    """One original sampled view; graph schema/sampling rules stay unchanged."""
    validate_record(record)
    if type(epoch) is not int or not 0 <= epoch <= 40 or type(view) is not int or view not in (0, 1):
        raise ValueError("Explicit original epoch0..40 and view0/1 required")
    runtime = _runtime()
    common = importlib.import_module("hiercp.common")
    identity = f"{record['input_provenance']['observation_id']}:{record['donor_case_id']}:{record['component_id']}:{view}:{epoch}"
    seed = common.stable_case_seed(record["seed"], record["case_id"], identity)
    graph = runtime["sample"].build_local_view(record["source_local"], record["target_local"],
        runtime["schema"].graph_config_from_dict(record["graph_config"]), seed=seed)
    graph.transition_record_sha256 = record["content_binding"]["graph_sha256"]
    graph.transition_observation_id = record["input_provenance"]["observation_id"]
    graph.transition_recipient_case = record["case_id"]
    graph.transition_donor_case = record["donor_case_id"]
    graph.transition_view_id = torch.tensor([view], dtype=torch.long)
    graph.transition_epoch = torch.tensor([epoch], dtype=torch.long)
    return graph, record["source_patch"], record["target_patch"]


def materialize_pair(record, *, epoch=0):
    """Both original views of one actual observation, without duplicate CNN inputs."""
    first, second = materialize(record, epoch=epoch, view=0), materialize(record, epoch=epoch, view=1)
    return ((first[0], second[0]), first[1], first[2])


@dataclass
class TransitionLocalBatch:
    graph: object
    source_patches: Tensor
    target_patches: Tensor
    source_index: Tensor
    graph_observation_index: Tensor
    indices: Tensor

    def __len__(self):
        return len(self.target_patches)

    def to(self, device, non_blocking=True):
        from copy import copy as shallow_copy
        return TransitionLocalBatch(shallow_copy(self.graph).to(device, non_blocking=non_blocking),
            *[getattr(self, name).to(device, non_blocking=non_blocking) for name in
              ("source_patches", "target_patches", "source_index", "graph_observation_index", "indices")])

    def cuda(self, non_blocking=False):
        return self.to("cuda", non_blocking=non_blocking)

    def pin_memory(self):
        from copy import copy as shallow_copy
        return TransitionLocalBatch(shallow_copy(self.graph).pin_memory(),
            *[getattr(self, name).pin_memory() for name in
              ("source_patches", "target_patches", "source_index", "graph_observation_index", "indices")])


def collate(items):
    """Batch complete two-view observations; no serial per-query GPU forwards."""
    from torch_geometric.data import Batch
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("Nonempty full physical observation batch required")
    graphs, sources, targets, source_ids, indices, lookup = [], [], [], [], [], {}
    for payload, index in items:
        views, source, target = payload
        if type(index) is not int or index < 0 or len(views) != 2:
            raise ValueError("Original observation index and both sampled graph views required")
        first, second = views
        for name in ("transition_record_sha256", "transition_observation_id", "transition_recipient_case", "transition_donor_case"):
            if first.get(name) != second.get(name):
                raise ValueError("Two views must own the same actual donor/recipient observation")
        if (not torch.equal(first.transition_view_id, torch.tensor([0]))
                or not torch.equal(second.transition_view_id, torch.tensor([1]))
                or not torch.equal(first.transition_epoch, second.transition_epoch)):
            raise ValueError("Views0/1 at the same actual epoch required")
        for patch in (source, target):
            if not isinstance(patch, Tensor) or patch.shape != (5, 48, 48, 48) or patch.device.type != "cpu":
                raise ValueError("Original host five-channel48 source/target patches required")
        key = (source.data_ptr(), tuple(source.shape), source.dtype)
        if key not in lookup:
            lookup[key] = len(sources)
            sources.append(source)
        source_ids.append(lookup[key])
        graphs.extend(views)
        targets.append(target)
        indices.append(index)
    return TransitionLocalBatch(Batch.from_data_list(graphs), torch.stack(sources), torch.stack(targets),
        torch.tensor(source_ids, dtype=torch.long), torch.arange(len(items)).repeat_interleave(2),
        torch.tensor(indices, dtype=torch.long))


@dataclass
class LocalFieldsOutput:
    fields: dict[str, Tensor]
    view_fields: tuple[dict[str, Tensor], dict[str, Tensor]]
    view_consistency: Tensor


class PreservedV1LocalEncoder(nn.Module):
    """Actual archived L0 owner; native upper receives the real fused view mean."""
    def __init__(self, base, *, scope_contract, expected_snapshot_root):
        super().__init__()
        runtime = _runtime(expected_snapshot_root=expected_snapshot_root, scope_contract=scope_contract)
        _config(base)
        model = base["model"]
        keys = ("hidden_dim", "heads", "dropout", "dense_base_channels", "dense_feature_dim",
                "dense_batch_size", "channels_last_3d", "checkpoint_local_blocks", "checkpoint_dense_encoder")
        self.core = runtime["model"].LocalTumorContextPyGEncoder(**{key: model[key] for key in keys},
                                                               layers=model["local_layers"])
        self.scope_contract = scope_contract
        self.snapshot_root = str(Path(expected_snapshot_root).resolve())
        self._consistency = runtime["model"].HierarchicalPyGPlacementModel._view_consistency
        self._activation_storage = {name: bool(model[name]) for name in
                                    ("checkpoint_dense_encoder", "checkpoint_local_blocks")}

    def get_extra_state(self):
        return {**local_contract(), **self._activation_storage, "scope_contract": self.scope_contract}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("Checkpoint belongs to another local input/scope/task bridge")

    def _validate_batch(self, batch):
        if not isinstance(batch, TransitionLocalBatch) or not len(batch):
            raise TypeError("D requires a complete TransitionLocalBatch, not a CT-only tensor")
        n = len(batch)
        device = next(self.core.parameters()).device
        if device.type != "cuda":
            raise RuntimeError("D neural execution requires CUDA; no CPU fallback")
        for name in ("source_patches", "target_patches"):
            value = getattr(batch, name)
            if value.ndim != 5 or tuple(value.shape[1:]) != (5, 48, 48, 48) or value.device != device:
                raise ValueError("Full original source/target5x48 patches on the encoder CUDA device required")
            torch._assert_async(torch.isfinite(value).all(), "Nonfinite original D dense inputs")
        if (batch.source_index.shape != (n,) or batch.source_index.dtype != torch.long
                or batch.graph_observation_index.shape != (2 * n,) or batch.graph_observation_index.dtype != torch.long
                or batch.indices.shape != (n,) or batch.indices.dtype != torch.long
                or any(getattr(batch, key).device != device for key in
                       ("source_index", "graph_observation_index", "indices"))
                or batch.graph.num_graphs != 2 * n or tuple(batch.graph.node_types) != _ROLE_NAMES):
            raise ValueError("Complete two-view disjoint graph ownership and original six roles required")
        torch._assert_async(((batch.source_index >= 0) & (batch.source_index < len(batch.source_patches))).all(),
                            "Original donor source index out of range")
        expected = torch.arange(n, device=device).repeat_interleave(2)
        torch._assert_async((batch.graph_observation_index == expected).all(), "Two-view observation ordering changed")
        torch._assert_async((batch.graph.transition_view_id == torch.tensor([0, 1], device=device).repeat(n)).all(),
                            "Original paired view0/1 order changed")
        markers = batch.graph.get("v1x_bounded_scope_contract")
        if not isinstance(markers, list) or len(markers) != 2 * n or any(marker != self.scope_contract for marker in markers):
            raise ValueError("Native/another-scope graph cannot enter the original10mm D encoder")

    def forward_fields(self, batch) -> LocalFieldsOutput:
        self._validate_batch(batch)
        source, target = self.core.encode_dense_maps(batch.source_patches, batch.source_index, batch.target_patches)
        graph_ids = batch.graph_observation_index
        fields = self.core.forward_graph(batch.graph, source[graph_ids], target[graph_ids])
        if tuple(fields) != _FIELD_NAMES:
            raise RuntimeError("Original semantic role/shell/readout output schema changed")
        n = len(batch)
        views = tuple({name: value.reshape(n, 2, 128)[:, view] for name, value in fields.items()}
                      for view in (0, 1))
        means = {name: (views[0][name] + views[1][name]) * .5 for name in fields}
        consistency = self._consistency(*views)
        torch._assert_async(torch.isfinite(means["fused"]).all() & torch.isfinite(consistency),
                            "Nonfinite original D readout/view consistency")
        return LocalFieldsOutput(means, views, consistency)

    def forward_with_consistency(self, batch):
        output = self.forward_fields(batch)
        return output.fields["fused"], output.view_consistency

    def forward(self, batch):
        return self.forward_with_consistency(batch)[0]
