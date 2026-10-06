"""Paired real donor/recipient upper graphs for the legacy native30 model.

This is an explicit external-donor input adaptation of a single-patient
training task. It retains the activated source's learned relation schema and
feature equations, but it does not claim an exact replay of that training
topology. Both patients retain their own CT, coordinates, regions and liver;
only the original tumor/candidate compatibility relation crosses frames.
P/U targets never enter CandidateSpec or either upper graph.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import ndimage as ndi
import torch


FORMAT = "legacy_native30_paired_external_donor_upper_v1"
_SOURCE_MODULES = ("common", "curriculum", "hierarchy", "prototype", "region", "schema")
_LEGACY_HOST = ("tumor", "hosted_by", "region")
_LEGACY_REVERSE_HOST = ("region", "hosts_tumor", "tumor")


def _runtime(source):
    """Verify the already activated original source, never activate an archive."""
    root = Path(source).resolve(strict=True)
    modules, hashes = {}, {}
    for name in _SOURCE_MODULES:
        module = importlib.import_module("hiercp." + name)
        expected = (root / "hiercp" / (name + ".py")).resolve(strict=True)
        actual = Path(module.__file__).resolve(strict=True)
        if actual != expected or not actual.is_relative_to(root):
            raise ValueError(f"Native30 upper module is not the explicit original source: {name}: {actual}")
        modules[name] = module
        hashes[name] = hashlib.sha256(actual.read_bytes()).hexdigest()
    schema, hierarchy = modules["schema"], modules["hierarchy"]
    if (_LEGACY_HOST not in schema.PATIENT_EDGE_TYPES
            or _LEGACY_REVERSE_HOST not in schema.PATIENT_EDGE_TYPES
            or ("tumor", "conditions", "region") in schema.PATIENT_EDGE_TYPES):
        raise ValueError("Native30 upper requires the loaded legacy hosted_by/hosts_tumor schema")
    for name in ("_candidate_raw", "_lesions", "_liver_raw", "_node_meta",
                 "_set_patient_relation", "_full_bipartite", "_knn", "build_prototype_graph"):
        function = getattr(hierarchy, name, None)
        filename = inspect.getsourcefile(function) if callable(function) else None
        if filename is None or Path(filename).resolve(strict=True) != root / "hiercp/hierarchy.py":
            raise ValueError(f"Native30 upper requires the actual original hierarchy operator: {name}")
    return modules, hashes


def _verify_source_proof(bundle, source_hashes):
    proof = getattr(bundle, "source_proof", None)
    if proof is None:
        proof = getattr(bundle, "receipt", {}).get("source_proof")
    if not isinstance(proof, dict) or not isinstance(proof.get("source_python_sha256"), dict):
        raise ValueError("Loaded native30 source inventory proof required before upper construction")
    inventory = proof["source_python_sha256"]
    root = Path(bundle.source).resolve(strict=True)
    for name, digest in source_hashes.items():
        if inventory.get("hiercp/" + name + ".py") != digest:
            raise ValueError(f"Original source changed since native30 loading: {name}")
    for relative, digest in inventory.items():
        filename = root / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts or filename.is_symlink()
                or not filename.resolve(strict=True).is_relative_to(root / "hiercp")
                or hashlib.sha256(filename.read_bytes()).hexdigest() != digest):
            raise ValueError(f"Original source inventory changed since native30 loading: {relative}")


def _whole_case(geometry, rows):
    rows = tuple(rows)
    if not rows:
        raise ValueError("Complete ordered native P+128U case required")
    identities = {(row["case_id"], row["donor_case_id"], int(row["donor_component"])) for row in rows}
    if len(identities) != 1:
        raise ValueError("One joint case must preserve its exact one native donor")
    case_id, donor_id, _ = next(iter(identities))
    if str(case_id) == str(donor_id):
        raise ValueError("Native external donor and recipient must be distinct patients")
    expected = tuple(row for row in geometry.rows if row["case_id"] == case_id)
    # Inventory order and actual forward assignments are checked without
    # reading class labels, observed-positive flags or P/U target fields.
    keys = ("id", "case_id", "donor_case_id", "donor_component", "center")
    if (not expected or len(rows) != len(expected)
            or len({row["id"] for row in rows}) != len(rows)
            or any(any(row[key] != original[key] for key in keys)
                   for row, original in zip(rows, expected))):
        raise ValueError("Upper construction must use the entire unchanged ordered case, never an L0 chunk")
    return rows


def _case(case):
    image, label, spacing = np.asarray(case.image), np.asarray(case.label), np.asarray(case.spacing)
    if (image.ndim != 3 or label.shape != image.shape or spacing.shape != (3,)
            or not np.isfinite(image).all() or not np.isfinite(label).all()
            or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or not str(case.paths.case_id)):
        raise ValueError("Actual aligned CT, annotations, positive spacing and case identity required")
    if hasattr(case, "shape"):
        if tuple(case.shape) != tuple(image.shape):
            raise ValueError("Case shape metadata differs from actual CT")
        return case
    # Preserve all source-signature metadata needed by the original region
    # cache validator when a provider exposes arrays without a shape property.
    values = dict(vars(case))
    values["shape"] = tuple(image.shape)
    return SimpleNamespace(**values)


def _source(case, source, tumor_label):
    mask = np.asarray(source.full_mask)
    center = np.asarray(source.anchor_center)
    if (mask.dtype != bool or mask.shape != case.shape or not mask.any()
            or np.any(mask & (case.label != tumor_label))
            or int(source.voxel_count) != int(mask.sum())
            or center.shape != (3,) or not np.issubdtype(center.dtype, np.integer)
            or np.any(center < 0) or np.any(center >= case.shape)):
        raise ValueError("Actual complete donor component and donor-native anchor required")
    if (not np.array_equal(source.patch_mask, mask[source.patch_slices])
            or int(source.patch_mask.sum()) != int(mask.sum())
            or not np.array_equal(source.patch_image, case.image[source.patch_slices])
            or not np.array_equal(center, np.array([sl.start for sl in source.patch_slices])
                                  + np.asarray(source.patch_mask.shape) // 2)):
        raise ValueError("Donor patch must preserve the complete actual donor CT and component")
    components, _ = ndi.label(case.label == tumor_label, structure=ndi.generate_binary_structure(3, 1))
    if int(source.component_id) < 1 or not np.array_equal(mask, components == int(source.component_id)):
        raise ValueError("Donor source must be one complete actual annotated component")


def _regions(case, regions, config, schema):
    if (tuple(regions.full_organ_mask.shape) != case.shape
            or tuple(regions.organ_depth.shape) != case.shape
            or tuple(regions.region_labels.shape) != case.shape
            or regions.full_organ_mask.dtype != bool
            or not np.array_equal(regions.full_organ_mask, np.isin(case.label, (1, 2)))
            or not regions.full_organ_mask.any()
            or regions.num_regions != config.num_regions
            or regions.region_features.shape != (config.num_regions, schema.REGION_FEATURE_DIM)
            or regions.region_positions.shape != (config.num_regions, 3)
            or not np.isfinite(regions.region_features).all()
            or not np.isfinite(regions.region_positions).all()
            or not np.isfinite(regions.organ_depth).all()):
        raise ValueError("Actual case-aligned original region descriptors required")


def _bank_audit(bank, meta, *, debug=False):
    if type(debug) is not bool:
        raise ValueError("Native30 DEBUG profile must be an explicit boolean")
    bank.validate()
    fitted = tuple(map(str, bank.training_case_ids))
    if not fitted or len(set(fitted)) != len(fitted):
        raise ValueError("Actual nonempty unique prototype fitted-case inventory required")
    if not debug and len(fitted) != 105:
        raise ValueError("Native30 evaluation retains the complete original 105-case prototype bank")
    validation = tuple(map(str, meta["split"]["inner_val"]))
    if not validation or len(validation) != len(set(validation)):
        raise ValueError("Actual current validation inventory required for prototype overlap disclosure")
    overlap = sorted(set(fitted) & set(validation))
    if debug:
        training = set(map(str, meta["split"]["inner_train"]))
        if not set(fitted).issubset(training) or overlap:
            raise ValueError("Explicit DEBUG mechanical bank must retain only actual declared training cases")
    return dict(prototype_training_case_ids=list(fitted), prototype_fitted_case_count=len(fitted),
                current_validation_case_count=len(validation),
                prototype_current_validation_overlap_case_ids=overlap,
                prototype_current_validation_overlap_case_count=len(overlap),
                prototype_bank_preserved=True, prototype_bank_refitted=False,
                prototype_training_subset_in_DEBUG=debug and len(fitted) != 105,
                prototype_overlap_policy="explicit DEBUG training-only mechanical bank; production bank admission unchanged" if debug
                    else "retain actual saved full bank; disclose current validation overlap",
                prototype_bank_current_validation_disjoint=not bool(overlap),
                strict_current_validation_held_out_claim=False)


def _recipient_lesions(case, regions, *, hierarchy, schema, tumor_label, max_lesions, ct_clip):
    """Original lesion feature equations on every actual recipient component.

    The original _lesions API subtracts an in-patient SourceTumor. A foreign
    donor removes no recipient lesion, so enumerate real components directly;
    do not manufacture a recipient SourceTumor with an empty/fake full_mask.
    """
    components, count = ndi.label(case.label == int(tumor_label),
                                  structure=ndi.generate_binary_structure(3, 1))
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


def _output_root(bundle, output):
    root = Path(output).resolve()
    preserved = [Path(bundle.source).resolve(strict=True)]
    for name in ("baseline", "experiment"):
        value = getattr(bundle, name, None)
        if value is not None:
            preserved.append(Path(value).resolve(strict=True))
    for directory in preserved:
        if root == directory or root.is_relative_to(directory) or directory.is_relative_to(root):
            raise ValueError("Native30 region output must be disjoint from original source/experiment")
    root.mkdir(parents=True, exist_ok=True)
    return root / "native30_regions"


def _candidate_specs(runtime, case, transported_source, regions, rows, bank, config, clip):
    common, curriculum = runtime["common"], runtime["curriculum"]
    mask = np.asarray(transported_source.patch_mask)
    if (mask.ndim != 3 or mask.dtype != bool or not mask.any()
            or int(transported_source.voxel_count) != int(mask.sum())
            or transported_source.patch_image.shape != mask.shape
            or not np.isfinite(transported_source.patch_image).all()):
        raise ValueError("Complete actual donor footprint transported to recipient spacing required")
    assignments, _ = bank.assign(regions.region_features, top_k=config.prototype_top_m,
                                 temperature=config.prototype_temperature)
    occupied = common.distance_to_mask_mm(case.label == 2, case.spacing)
    specs = []
    for row in rows:
        coordinate = np.asarray(row["center"])
        if (coordinate.shape != (3,) or not np.issubdtype(coordinate.dtype, np.integer)
                or np.any(coordinate < 0) or np.any(coordinate >= case.shape)):
            raise ValueError("Candidate must retain its actual recipient-native integer center")
        center = tuple(map(int, coordinate))
        if not regions.full_organ_mask[center]:
            raise ValueError("Native candidate center is outside the actual recipient organ")
        ct = common.extract_centered_patch(case.image, center, mask.shape, pad_value=clip[0])
        organ = common.extract_centered_patch(regions.full_organ_mask, center, mask.shape, pad_value=False)
        mean, std = common.context_stats_for_local_mask(ct, organ, mask)
        region_id = int(regions.region_at(center))
        # Fixed neutral bookkeeping is independent of observed P/U labels.
        specs.append(curriculum.CandidateSpec(center=center, difficulty=runtime["schema"].DIFFICULTY_EASY,
            corruption=runtime["schema"].CORRUPTION_NONE, region_id=region_id,
            prototype_id=int(assignments[region_id, 0]),
            liver_coverage=float((organ & mask).sum() / mask.sum()),
            border_distance_mm=float(regions.organ_depth[center]),
            occupied_distance_mm=float(occupied[center]),
            context_mean_hu=float(mean), context_std_hu=float(std)))
    return tuple(specs)


def build_native30_upper(bundle, geometry, rows, *, output, region_cache=None,
                         lesion_policy="saved_guard"):
    """Return (patient_graph, prototype_graph, audit) for one entire case.

    ``bundle`` owns the actual saved full model/config/bank and explicit source.
    ``geometry.case`` and ``geometry.donor_source`` expose real native arrays and
    the complete physical-spacing donor transport. L0 may be encoded in batches;
    this upper graph is always built from the complete ordered case inventory.

    ``all_observed`` explicitly retains every recipient component and every
    donor component other than the selected source. The saved max_lesions is a
    construction guard, not a learned tensor dimension. This external-donor
    benchmark may use that policy without changing the saved L0 config/cache,
    but it must disclose the admission change rather than claim exact replay.
    """
    if lesion_policy not in ("saved_guard", "all_observed"):
        raise ValueError("Explicit native30 lesion policy must be saved_guard or all_observed")
    runtime, source_hashes = _runtime(bundle.source)
    _verify_source_proof(bundle, source_hashes)
    if Path(geometry.source).resolve(strict=True) != Path(bundle.source).resolve(strict=True):
        raise ValueError("Upper and local geometry must use the same explicit original source")
    rows = _whole_case(geometry, rows)
    h, schema, region_module = runtime["hierarchy"], runtime["schema"], runtime["region"]
    config = schema.graph_config_from_dict(bundle.config["graph"])
    lesion_limit = config.max_lesions if lesion_policy == "saved_guard" else None
    clip = tuple(map(float, bundle.config["ct_clip"]))
    if len(clip) != 2 or not np.isfinite(clip).all() or clip[0] >= clip[1]:
        raise ValueError("Actual saved CT clip required")
    for name, saved in bundle.config["graph"].items():
        if hasattr(config, name):
            actual = getattr(config, name)
            if isinstance(actual, tuple):
                saved = tuple(saved)
            if actual != saved:
                raise ValueError(f"Original upper graph parser changed actual saved configuration: {name}")
    bank = getattr(bundle, "prototype_bank", None)
    if bank is None:
        bank = bundle.bank
    debug = getattr(bundle, "receipt", {}).get("debug", False)
    bank_audit = _bank_audit(bank, geometry.meta, debug=debug)
    if bank.num_prototypes != config.num_prototypes:
        raise ValueError("Actual saved prototype bank and graph configuration differ")
    row = rows[0]
    recipient, donor = _case(geometry.case(row["case_id"])), _case(geometry.case(row["donor_case_id"]))
    if (str(recipient.paths.case_id) != str(row["case_id"])
            or str(donor.paths.case_id) != str(row["donor_case_id"])):
        raise ValueError("Actual native case identity differs from ordered inventory")
    donor_source, transported_source = geometry.donor_source(row)
    _source(donor, donor_source, 2)
    if int(donor_source.component_id) != int(row["donor_component"]):
        raise ValueError("Native donor component differs from its exact inventory assignment")
    cache_root = _output_root(bundle, output)
    if region_cache is not None:
        cache_root = Path(region_cache).resolve(strict=True)
        for case in (recipient, donor):
            cached = cache_root / case.paths.case_id
            if cached.is_symlink() or not (cached / 'metadata.json').is_file():
                raise ValueError('Explicit read-only original region cache lacks actual recipient/donor; no rebuild')
    def load_regions(case):
        result = region_module.load_or_build_patient_regions(case, liver_label=1, tumor_label=2,
            config=config, ct_clip=clip, seed=runtime["common"].stable_case_seed(
                int(bundle.config["seed"]), case.paths.case_id, region_module.REGION_CACHE_SEED_SALT),
            cache_dir=cache_root, overwrite=False, mmap=True)
        _regions(case, result, config, schema)
        return result
    recipient_regions, donor_regions = load_regions(recipient), load_regions(donor)
    specs = _candidate_specs(runtime, recipient, transported_source, recipient_regions, rows, bank, config, clip)
    axis, anisotropy = h._principal_axis(donor_source.full_mask, donor.spacing)
    source_raw_function = getattr(h, "_source_raw", None)
    if source_raw_function is None:
        source_raw_function = getattr(h, "_tumor_raw", None)
    if not callable(source_raw_function):
        raise ValueError("Loaded original source lacks its source/tumor raw feature operator")
    if Path(inspect.getsourcefile(source_raw_function)).resolve(strict=True) != Path(bundle.source).resolve() / "hiercp/hierarchy.py":
        raise ValueError("Source/tumor raw operator is not the actual original source")
    source_raw = source_raw_function(donor, donor_source, donor_regions,
        (donor.label == 2) & ~donor_source.full_mask, ct_clip=clip)[None].astype(np.float32)
    candidate_raw = np.stack([h._candidate_raw(recipient, transported_source, spec, recipient_regions,
        source_axis=axis, source_anisotropy=anisotropy, ct_clip=clip) for spec in specs]).astype(np.float32)
    candidate_pos = np.stack([h.normalized_position(spec.center, recipient.shape) for spec in specs]).astype(np.float32)
    candidate_regions = np.asarray([spec.region_id for spec in specs], np.int64)
    recipient_lesion_raw, recipient_lesion_pos, recipient_lesion_regions, recipient_lesion_ids = _recipient_lesions(
        recipient, recipient_regions, hierarchy=h, schema=schema, tumor_label=2,
        max_lesions=lesion_limit, ct_clip=clip)
    donor_lesion_raw, donor_lesion_pos, donor_lesion_regions = h._lesions(
        donor, donor_source, donor_regions, tumor_label=2, max_lesions=lesion_limit, ct_clip=clip)
    nr, nd = recipient_regions.num_regions, donor_regions.num_regions
    lr, ld = len(recipient_lesion_ids), len(donor_lesion_raw)
    combined_regions = SimpleNamespace(num_regions=nr + nd,
        region_features=np.concatenate((recipient_regions.region_features, donor_regions.region_features)),
        region_positions=np.concatenate((recipient_regions.region_positions, donor_regions.region_positions)))
    liver_raw = np.stack([h._liver_raw(case, regions, tumor_label=2, ct_clip=clip)
        for case, regions in ((recipient, recipient_regions), (donor, donor_regions))]).astype(np.float32)
    # Region and lesion positions remain in their own real normalized frames.
    # Liver positions follow the activated original builder's zero convention.
    graph = h.HeteroData()
    tumor_region = nr + int(donor_regions.region_at(donor_source.anchor_center))
    node_values = (
        ("tumor", source_raw, h.normalized_position(donor_source.anchor_center, donor.shape)[None].astype(np.float32),
         np.asarray([tumor_region], np.int64), np.ones(1, np.int64)),
        ("candidate", candidate_raw, candidate_pos, candidate_regions, np.zeros(len(rows), np.int64)),
        ("region", combined_regions.region_features.astype(np.float32), combined_regions.region_positions.astype(np.float32),
         np.arange(nr + nd, dtype=np.int64), np.concatenate((np.zeros(nr, np.int64), np.ones(nd, np.int64)))),
        ("lesion", np.concatenate((recipient_lesion_raw, donor_lesion_raw)),
         np.concatenate((recipient_lesion_pos, donor_lesion_pos)),
         np.concatenate((recipient_lesion_regions, donor_lesion_regions + nr)),
         np.concatenate((np.zeros(lr, np.int64), np.ones(ld, np.int64)))),
        ("liver", liver_raw, np.zeros((2, 3), np.float32), np.full(2, -1, np.int64), np.arange(2, dtype=np.int64)),
    )
    if set(schema.PATIENT_NODE_TYPES) != {value[0] for value in node_values}:
        raise ValueError("Unsupported loaded original patient node schema; no node types removed")
    for name, raw, pos, region_ids, frames in node_values:
        graph[name].raw_x = torch.from_numpy(raw.astype(np.float32))
        graph[name].pos = torch.from_numpy(pos.astype(np.float32))
        graph[name].region_index = torch.from_numpy(region_ids)
        graph[name].frame_index = torch.from_numpy(frames)
        graph[name].meta = torch.from_numpy(h._node_meta(raw, name))
    n = len(rows)
    tc = h._full_bipartite(1, n)
    cr = np.stack((np.arange(n, dtype=np.int64), candidate_regions))
    tr = np.asarray([[0], [tumor_region]], np.int64)
    cl = h._full_bipartite(n, lr)
    tl = h._full_bipartite(1, ld)
    tl[1] += lr
    lesion_region = np.stack((np.arange(lr + ld, dtype=np.int64),
                              np.concatenate((recipient_lesion_regions, donor_lesion_regions + nr))))
    region_liver = np.stack((np.arange(nr + nd, dtype=np.int64),
                             np.concatenate((np.zeros(nr, np.int64), np.ones(nd, np.int64)))))
    region_edges = np.concatenate((recipient_regions.region_edge_index,
                                    donor_regions.region_edge_index + nr), axis=1).astype(np.int64)
    relations = {
        ("tumor", "compatible_with", "candidate"): tc,
        ("candidate", "matched_to", "tumor"): tc[[1, 0]],
        ("candidate", "spatial_neighbor", "candidate"): h._knn(candidate_pos, config.candidate_k),
        ("candidate", "belongs_to", "region"): cr,
        ("region", "contains_candidate", "candidate"): cr[[1, 0]],
        _LEGACY_HOST: tr, _LEGACY_REVERSE_HOST: tr[[1, 0]],
        ("candidate", "near", "lesion"): cl,
        ("lesion", "near", "candidate"): cl[[1, 0]],
        ("lesion", "hosted_by", "region"): lesion_region,
        ("region", "hosts_lesion", "lesion"): lesion_region[[1, 0]],
        ("tumor", "coexists_with", "lesion"): tl,
        ("lesion", "coexists_with", "tumor"): tl[[1, 0]],
        ("region", "adjacent_to", "region"): region_edges,
        ("region", "inside", "liver"): region_liver,
        ("liver", "contains", "region"): region_liver[[1, 0]],
    }
    if set(schema.PATIENT_EDGE_TYPES) != set(relations):
        raise ValueError("Unsupported actual legacy relation schema; no relations replaced or dropped")
    for edge_type in schema.PATIENT_EDGE_TYPES:
        h._set_patient_relation(graph, edge_type, relations[edge_type])
    prototype = h.build_prototype_graph(specs, graph, combined_regions, bank, config=config)
    prototype["candidate"].frame_index = graph["candidate"].frame_index.clone()
    prototype["region"].frame_index = graph["region"].frame_index.clone()
    _verify_source_proof(bundle, source_hashes)
    audit = dict(format=FORMAT, debug=debug,
        input_adaptation="paired_donor_recipient_frames_for_external_donor",
        original_single_patient_topology_equivalent=False, exact_original_external_donor_path=False,
        original_learned_operators_unchanged=True, source_host_incidence=True,
        source_host_frame="actual donor region; recipient region-count offset",
        cross_frame_relations=[["tumor", "compatible_with", "candidate"],
                               ["candidate", "matched_to", "tumor"]],
        geometric_relationships_within_actual_patient_frames=True,
        donor_case_id=str(donor.paths.case_id), recipient_case_id=str(recipient.paths.case_id),
        frame_case_ids=[str(recipient.paths.case_id), str(donor.paths.case_id)],
        candidate_record_ids=[item["id"] for item in rows], candidate_count=n,
        recipient_region_count=nr, donor_region_count=nd, combined_region_count=nr + nd,
        recipient_lesion_count=lr, donor_other_lesion_count=ld,
        lesion_admission=dict(policy=lesion_policy, saved_max_lesions=config.max_lesions,
            effective_max_lesions=lesion_limit, recipient_components_retained=lr,
            donor_other_components_retained=ld, components_dropped=0,
            saved_guard_exceeded=config.max_lesions is not None and config.max_lesions > 0
                and max(lr, ld) > config.max_lesions,
            saved_configuration_unchanged=True,
            scope="explicit external-donor upper-input policy; no L0/cache/weights/P-U change"),
        donor_native_spacing_mm=donor.spacing.tolist(), recipient_native_spacing_mm=recipient.spacing.tolist(),
        donor_anchor_ijk=list(map(int, donor_source.anchor_center)),
        donor_mask_voxels=int(donor_source.voxel_count),
        recipient_regridded_mask_voxels=int(transported_source.voxel_count),
        source_region_index=tumor_region, source_region_native_index=tumor_region - nr,
        P_U_labels_in_forward=False, query_GT_in_forward=False,
        upper_execution="single_joint_case", upper_chunking=False,
        annotation_blind=False, annotation_aware_evaluation=True,
        quality_verified=False, strict_held_out_quality_verified=False,
        bank_fingerprint=bank.fingerprint(), original_source_module_sha256=source_hashes,
        original_source=str(Path(bundle.source).resolve()), graph_config=config.to_dict(), ct_clip=list(clip),
        patient_node_counts={name: int(graph[name].raw_x.shape[0]) for name in schema.PATIENT_NODE_TYPES},
        patient_edge_counts={"|".join(edge): int(graph[edge].edge_index.shape[1]) for edge in schema.PATIENT_EDGE_TYPES},
        prototype_edge_counts={"|".join(edge): int(prototype[edge].edge_index.shape[1]) for edge in schema.PROTOTYPE_EDGE_TYPES},
        recipient_component_ids_in_size_order=recipient_lesion_ids,
        region_cache_output=str(cache_root), existing_region_cache_overwritten=False,
        annotation_exposure=dict(recipient_label_available_to_graph=True,
            recipient_lesion_nodes=lr, donor_other_lesion_nodes=ld,
            candidate_to_recipient_lesion_edges=n * lr, tumor_to_donor_other_lesion_edges=ld,
            observed_P_U_targets_supplied=False,
            warning="All actual recipient lesions remain visible. This is annotation-aware ranking; prototype overlap is disclosed in this audit."),
        training_started=False, optimizer_updates=0, production_ready=False, **bank_audit)
    return graph, prototype, audit
