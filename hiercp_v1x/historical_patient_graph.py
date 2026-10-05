"""Explicit foreign-donor inputs for the unchanged historical v1 upper model.

The archived builder accepts a tumor in its own patient's coordinate frame.
This adapter is therefore an evaluation adaptation, not an exact replay of
v1's source-anchor task. Donor descriptors stay donor-native; recipient
descriptors and all recipient lesion annotations stay recipient-native.
No mask is invented in the recipient to satisfy the original API. The original
feature equations, upper graph relations and population builder are reused.
Annotation exposure is returned as evidence, never concealed as blind CP
evaluation. P/U labels are not inputs to this module.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from scipy import ndimage as ndi
import torch


FORMAT = "historical_v1_external_donor_upper_graph_v1"
ROOT = Path(__file__).resolve().parents[1]
_SOURCE_MODULES = ("common", "contracts", "curriculum", "hierarchy", "prototype", "region", "schema")


def _runtime():
    """Use byte-exact original equations without switching global imports."""
    from .contracts import V1_ARCHIVE_SHA256
    archive = ROOT / "versions/v1/pipeline_v1_source.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != V1_ARCHIVE_SHA256:
        raise ValueError("Pinned original source archive differs")
    modules, hashes = {}, {}
    with ZipFile(archive) as bundle:
        for name in _SOURCE_MODULES:
            module = importlib.import_module("hiercp." + name)
            expected = hashlib.sha256(bundle.read("hiercp/" + name + ".py")).hexdigest()
            actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError("Historical upper graph requires original source bytes: " + name)
            modules[name], hashes[name] = module, actual
    return modules, hashes


def _case(case):
    image, label, spacing = np.asarray(case.image), np.asarray(case.label), np.asarray(case.spacing)
    if (image.ndim != 3 or label.shape != image.shape or spacing.shape != (3,)
            or not np.isfinite(image).all() or not np.isfinite(label).all()
            or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or not str(case.paths.case_id)):
        raise ValueError("Actual aligned CT, labels, positive spacing and case identity required")
    if hasattr(case, "shape"):
        if tuple(case.shape) != tuple(image.shape):
            raise ValueError("Case shape metadata differs from actual CT")
        return case
    # OriginalRawStore exposes real arrays but omits this derived shape property.
    return SimpleNamespace(image=image, label=label, spacing=spacing, paths=case.paths,
                           shape=tuple(image.shape))


def _regions(case, regions, width):
    if (regions.full_organ_mask.shape != case.shape or regions.organ_depth.shape != case.shape
            or regions.region_labels.shape != case.shape or not regions.full_organ_mask.any()
            or regions.full_organ_mask.dtype != bool
            or not np.array_equal(regions.full_organ_mask, np.isin(case.label, (1, 2)))
            or regions.region_features.shape != (regions.num_regions, width)
            or regions.region_positions.shape != (regions.num_regions, 3)
            or not np.isfinite(regions.region_features).all()
            or not np.isfinite(regions.region_positions).all()
            or not np.isfinite(regions.organ_depth).all()):
        raise ValueError("Actual case-aligned original region descriptors required")


def _source(case, source, tumor_label):
    mask = np.asarray(source.full_mask)
    if (mask.dtype != bool or mask.shape != case.shape or not mask.any()
            or np.any(mask & (case.label != tumor_label))
            or int(source.voxel_count) != int(mask.sum())):
        raise ValueError("Donor source must be its actual annotated full component")
    center = np.asarray(source.anchor_center)
    if (center.shape != (3,) or not np.issubdtype(center.dtype, np.integer)
            or np.any(center < 0) or np.any(center >= case.shape)):
        raise ValueError("Donor anchor must preserve its actual donor coordinate")
    if (not np.array_equal(source.patch_mask, mask[source.patch_slices])
            or int(source.patch_mask.sum()) != int(mask.sum())
            or not np.array_equal(source.patch_image, case.image[source.patch_slices])
            or not np.array_equal(center, np.array([sl.start for sl in source.patch_slices])
                                  + np.asarray(source.patch_mask.shape) // 2)):
        raise ValueError("Donor patch must preserve actual donor image and mask bytes")
    components, _ = ndi.label(case.label == tumor_label, structure=ndi.generate_binary_structure(3, 1))
    if int(source.component_id) < 1 or not np.array_equal(mask, components == int(source.component_id)):
        raise ValueError("Donor source must preserve one complete actual annotated component")


def _recipient_lesions(case, regions, *, hierarchy, schema, tumor_label, max_lesions, ct_clip):
    """Original _lesions equations on ALL actual recipient components.

    A foreign donor removes no recipient component. This sole occupancy change
    is explicit; no synthetic empty SourceTumor/full_mask is passed downstream.
    """
    occupied = case.label == int(tumor_label)
    components, count = ndi.label(occupied, structure=ndi.generate_binary_structure(3, 1))
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    component_ids = sorted(range(1, count + 1), key=lambda value: int(sizes[value]), reverse=True)
    if max_lesions is not None and int(max_lesions) > 0 and count > int(max_lesions):
        raise RuntimeError("Patient graph lesion count exceeds graph.max_lesions; no lesions were dropped. "
                           f"case_id={case.paths.case_id}, detected_recipient_lesions={count}, "
                           f"configured_max_lesions={max_lesions}")
    rows, positions, region_ids = [], [], []
    for component in component_ids:
        mask = components == component
        center = tuple(int(np.clip(round(value), 0, case.shape[axis] - 1))
                       for axis, value in enumerate(ndi.center_of_mass(mask)))
        mean, std = hierarchy._context_stats(case, mask, regions.full_organ_mask)
        depth = float(regions.organ_depth[center])
        rows.append(hierarchy.upper_geometry_vector(center=center, shape=case.shape,
            border_distance_mm=depth, occupied_distance_mm=0., context_mean_hu=mean,
            context_std_hu=std, volume_vox=int(mask.sum()), coverage=1.,
            local_thickness_mm=max(2. * depth, float(np.min(case.spacing))),
            surface_alignment=0., scale_mean=1., anisotropy=1., valid=1., ct_clip=ct_clip))
        positions.append(hierarchy.normalized_position(center, case.shape).astype(np.float32))
        region_ids.append(regions.region_at(center))
    if not rows:
        return (np.empty((0, schema.UPPER_RAW_DIM), np.float32),
                np.empty((0, 3), np.float32), np.empty((0,), np.int64), component_ids)
    return (np.stack(rows).astype(np.float32), np.stack(positions).astype(np.float32),
            np.asarray(region_ids, np.int64), component_ids)


def build_external_hierarchy(*, recipient_case, donor_case, donor_source, recipient_source,
                             specs, recipient_regions, donor_regions, bank, graph_config,
                             ct_clip, training_case_ids, tumor_label=2, debug=False):
    """Return (patient_graph, prototype_graph, audit) for all ordered candidates.

    ``recipient_source`` is the real complete donor footprint regridded by the
    native CP spacing convention, not a recipient lesion or invented mask.
    ``specs`` must contain the whole caller cohort with genuine recipient CT,
    depth, region assignment and placement statistics; labels remain external.
    The caller preserves candidate order and binds the full P+128U inventory.
    Explicit ``debug=True`` accepts a genuine prototype bank fitted on a
    nonempty subset of the declared training cases for mechanical smoke only.
    It never changes that bank's actual case IDs or production admission.
    """
    runtime, source_hashes = _runtime()
    h, schema = runtime["hierarchy"], runtime["schema"]
    recipient_case, donor_case = _case(recipient_case), _case(donor_case)
    if recipient_case.paths.case_id == donor_case.paths.case_id:
        raise ValueError("External-donor benchmark requires independent donor and recipient")
    training = tuple(map(str, training_case_ids))
    fitted = tuple(map(str, bank.training_case_ids))
    if type(debug) is not bool:
        raise ValueError("DEBUG mode must be an explicit boolean")
    bank_allowed = bool(fitted) and len(fitted) == len(set(fitted)) and (
        set(fitted).issubset(training) if debug else set(fitted) == set(training))
    if (not training or len(training) != len(set(training)) or not bank_allowed
            or str(donor_case.paths.case_id) not in training
            or str(recipient_case.paths.case_id) in training):
        raise ValueError("Exact training-only prototype bank, training donor and held-out recipient required")
    bank.validate()
    graph_config.validate()
    if (len(ct_clip) != 2 or not np.isfinite(ct_clip).all() or not ct_clip[0] < ct_clip[1]
            or int(tumor_label) != 2
            or donor_regions.num_regions != graph_config.num_regions
            or recipient_regions.num_regions != graph_config.num_regions
            or bank.num_prototypes != graph_config.num_prototypes):
        raise ValueError("Exact historical CT clip, organ labels, region and prototype configuration required")
    _regions(donor_case, donor_regions, schema.REGION_FEATURE_DIM)
    _regions(recipient_case, recipient_regions, schema.REGION_FEATURE_DIM)
    _source(donor_case, donor_source, int(tumor_label))
    target_mask = np.asarray(recipient_source.patch_mask)
    if (target_mask.ndim != 3 or target_mask.dtype != bool or not target_mask.any()
            or int(recipient_source.voxel_count) != int(target_mask.sum())
            or recipient_source.patch_image.shape != target_mask.shape
            or not np.isfinite(recipient_source.patch_image).all()):
        raise ValueError("Complete real recipient-regridded donor footprint required")
    from hiercp_v22.data import donor_in_target_spacing
    expected, _ = donor_in_target_spacing(donor_source, donor_case.spacing, recipient_case.spacing)
    if (not np.array_equal(target_mask, expected.patch_mask)
            or not np.array_equal(recipient_source.patch_image, expected.patch_image)
            or tuple(recipient_source.anchor_center) != tuple(expected.anchor_center)):
        raise ValueError("Recipient footprint differs from complete actual donor spacing transport")
    specs = tuple(specs)
    if not specs:
        raise ValueError("Complete ordered candidate cohort required")
    for spec in specs:
        center = np.asarray(spec.center)
        if (center.shape != (3,) or not np.issubdtype(center.dtype, np.integer)
                or np.any(center < 0) or np.any(center >= recipient_case.shape)
                or not recipient_regions.full_organ_mask[tuple(center)]
                or int(spec.region_id) != recipient_regions.region_at(center)):
            raise ValueError("Candidate must preserve its actual recipient center and region")
        finite = (spec.liver_coverage, spec.border_distance_mm, spec.context_mean_hu,
                  spec.context_std_hu, *spec.scale_array, *spec.rotation_matrix.ravel())
        if (not np.isfinite(finite).all() or not 0 <= spec.liver_coverage <= 1
                or spec.border_distance_mm < 0 or spec.context_std_hu < 0
                or not (np.isfinite(spec.occupied_distance_mm) or np.isposinf(spec.occupied_distance_mm))
                or spec.occupied_distance_mm < 0 or np.any(spec.scale_array <= 0)):
            raise ValueError("Actual finite candidate geometry/context statistics required")
    axis, anisotropy = h._principal_axis(donor_source.full_mask, donor_case.spacing)
    source_raw = h._source_raw(donor_case, donor_source, donor_regions,
        (donor_case.label == int(tumor_label)) & ~donor_source.full_mask, ct_clip=ct_clip)[None]
    candidate_raw = np.stack([h._candidate_raw(recipient_case, recipient_source, spec,
        recipient_regions, source_axis=axis, source_anisotropy=anisotropy, ct_clip=ct_clip)
        for spec in specs]).astype(np.float32)
    candidate_position = np.stack([h.normalized_position(spec.center, recipient_case.shape)
                                   for spec in specs]).astype(np.float32)
    candidate_regions = np.asarray([int(spec.region_id) for spec in specs], np.int64)
    lesion_raw, lesion_position, lesion_regions, lesion_ids = _recipient_lesions(
        recipient_case, recipient_regions, hierarchy=h, schema=schema,
        tumor_label=tumor_label, max_lesions=graph_config.max_lesions, ct_clip=ct_clip)
    graph = h.HeteroData()
    graph.patient_graph_contract = runtime["contracts"].PATIENT_GRAPH_CONTRACT
    graph["tumor"].raw_x = torch.from_numpy(source_raw.astype(np.float32))
    graph["tumor"].pos = torch.from_numpy(h.normalized_position(
        donor_source.anchor_center, donor_case.shape)[None].astype(np.float32))
    graph["tumor"].source_region_provenance = torch.tensor(
        [donor_regions.region_at(donor_source.anchor_center)], dtype=torch.long)
    graph["tumor"].region_index = torch.tensor([-1], dtype=torch.long)
    for name, raw, position, region_ids in (
        ("candidate", candidate_raw, candidate_position, candidate_regions),
        ("region", recipient_regions.region_features.astype(np.float32),
         recipient_regions.region_positions.astype(np.float32), np.arange(recipient_regions.num_regions)),
        ("lesion", lesion_raw, lesion_position, lesion_regions),
        ("liver", h._liver_raw(recipient_case, recipient_regions, tumor_label=tumor_label,
         ct_clip=ct_clip)[None].astype(np.float32), np.zeros((1, 3), np.float32), np.array([-1]))):
        graph[name].raw_x = torch.from_numpy(raw)
        graph[name].pos = torch.from_numpy(position)
        graph[name].region_index = torch.from_numpy(np.asarray(region_ids, np.int64))
    for node_type in schema.PATIENT_NODE_TYPES:
        graph[node_type].meta = torch.from_numpy(h._node_meta(graph[node_type].raw_x.numpy(), node_type))
    n, r, l = len(specs), recipient_regions.num_regions, len(lesion_ids)
    tc, tr, tl = h._full_bipartite(1, n), h._full_bipartite(1, r), h._full_bipartite(1, l)
    cr = np.stack([np.arange(n, dtype=np.int64), candidate_regions])
    cl = h._full_bipartite(n, l)
    lr = np.stack([np.arange(l, dtype=np.int64), lesion_regions])
    rl = h._full_bipartite(r, 1)
    relations = {
        ("tumor", "compatible_with", "candidate"): tc,
        ("candidate", "matched_to", "tumor"): tc[[1, 0]],
        ("candidate", "spatial_neighbor", "candidate"): h._knn(candidate_position, graph_config.candidate_k),
        ("candidate", "belongs_to", "region"): cr,
        ("region", "contains_candidate", "candidate"): cr[[1, 0]],
        ("tumor", "conditions", "region"): tr,
        ("region", "context_for", "tumor"): tr[[1, 0]],
        ("candidate", "near", "lesion"): cl,
        ("lesion", "near", "candidate"): cl[[1, 0]],
        ("lesion", "hosted_by", "region"): lr,
        ("region", "hosts_lesion", "lesion"): lr[[1, 0]],
        ("tumor", "coexists_with", "lesion"): tl,
        ("lesion", "coexists_with", "tumor"): tl[[1, 0]],
        ("region", "adjacent_to", "region"): recipient_regions.region_edge_index.astype(np.int64),
        ("region", "inside", "liver"): rl,
        ("liver", "contains", "region"): rl[[1, 0]],
    }
    for edge_type in schema.PATIENT_EDGE_TYPES:
        h._set_patient_relation(graph, edge_type, relations[edge_type])
    prototype = h.build_prototype_graph(specs, graph, recipient_regions, bank, config=graph_config)
    audit = dict(format=FORMAT, exact_original_external_donor_path=False,
        debug=debug, quality_verified=False,
        prototype_training_subset_in_DEBUG=debug and set(fitted) != set(training),
        declared_training_case_count=len(training), prototype_fitted_training_case_count=len(fitted),
        prototype_admission="explicit DEBUG training subset; mechanical smoke only" if debug
                            else "exact declared training case inventory",
        original_learned_operators_unchanged=True, P_U_labels_in_forward=False,
        donor_case_id=str(donor_case.paths.case_id), recipient_case_id=str(recipient_case.paths.case_id),
        donor_native_spacing_mm=donor_case.spacing.tolist(), recipient_native_spacing_mm=recipient_case.spacing.tolist(),
        donor_mask_voxels=int(donor_source.voxel_count), recipient_regridded_mask_voxels=int(recipient_source.voxel_count),
        donor_anchor_ijk=list(map(int, donor_source.anchor_center)), candidate_count=n,
        source_host_incidence=False, donor_position_used_only_as_original_masked_provenance=True,
        transport="native-axis physical mm; original rotation/scale on donor principal axis; no hidden reorientation",
        bank_fingerprint=bank.fingerprint(), prototype_training_case_ids=list(bank.training_case_ids),
        original_source_module_sha256=source_hashes,
        adaptation=["actual donor CT/regions describe source token in donor native frame",
                    "candidate volume uses complete recipient-regridded donor footprint",
                    "foreign donor excludes no recipient lesion; all original recipient annotation context retained"],
        annotation_blind=False,
        annotation_exposure=dict(recipient_label_available_to_graph=True, recipient_lesion_nodes=l,
            recipient_component_ids_in_size_order=lesion_ids, candidate_to_lesion_edges=n*l,
            exposed_fields=["lesion positions, volumes, CT context and region membership",
                            "liver raw tumor-count field", "candidate occupied-clearance bookkeeping"],
            observed_anchor_labels_supplied=False,
            warning="Observed P anchors can coincide with annotated lesion nodes; this is annotation-exposed ranking, not blind CP quality."))
    return graph, prototype, audit
