"""Opt-in matched-CP inference profile: context liver depth strictly above1mm.

Only the context depth predicate changes. The original2..10mm tumor annulus,
full10mm ROI, <=3mm liver-surface role, grids, edges and CT remain intact.
Context and liver-surface may share actual voxels as distinct typed roles.
Source context is still mandatory. Existing target-only semantic absence is
revalidated against the new predicate. Archived files/checkpoints are untouched.
This is a changed inference input definition for an unchanged completed BEST;
it does not claim that BEST was trained under this input definition.
"""
from __future__ import annotations

import ast
from contextvars import ContextVar
import copy
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys
import threading
import weakref

import numpy as np

FORMAT = "v24_matched_context_liver_depth_gt1mm_inference_v1"
PROOF_KEY = "v24_context_depth1_geometry"
GRAPH_KEY = "v24_context_depth1_contract"
POLICY = "context liver depth>1mm; tumor distance2..10mm; full ROI10mm; liver surface<=3mm; typed-role overlap; source context mandatory"
_LAST = ContextVar("v24_context_depth1_last_coordinates", default=None)
_INSTALLED = None


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash_array(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(_json([array.dtype.str, list(array.shape)]))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _node_hash(nodes):
    import torch
    digest = hashlib.sha256()
    for role in sorted(nodes):
        digest.update(role.encode())
        for key in sorted(nodes[role]):
            value = nodes[role][key]
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise ValueError("Profile canonical proofs require genuine CPU tensors before CUDA transfer")
            digest.update(key.encode())
            digest.update(_hash_array(value.detach().contiguous().numpy()).encode())
    return digest.hexdigest()


def identity():
    payload = dict(format=FORMAT, policy=POLICY,
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        enabled_by_default=False, minimum_liver_depth_exclusive_mm=1.,
        context_inner_radius_mm=2., context_outer_radius_mm=10.,
        ROI_margin_mm=10., liver_surface_depth_inclusive_mm=3.,
        coordinate_overlap="same actual voxel may occur once in each distinct typed role",
        source_context_required=True, model_state_changed=False,
        completed_BEST_training_definition="depth>4mm; input distribution changes for this inference profile",
        actual_CUDA_validation_claim=False, full_642_validation_claim=False)
    payload["contract_sha256"] = hashlib.sha256(_json(payload)).hexdigest()
    return payload


def _scope(config, runtime):
    from .bounded_scope import _scope_config
    _scope_config(config, 10.)
    values = (float(config.context_inner_radius_mm), float(config.context_outer_radius_mm),
              float(config.boundary_depth_mm), float(config.context_liver_surface_separation_mm))
    if values != (2., 10., 3., 1.):
        raise ValueError("Depth1 profile changes only context depth; original tumor2..10/surface3/config must remain")
    if float(runtime["scope"]["margin_mm"]) != 10.:
        raise ValueError("Depth1 profile requires the exact original bounded10mm runtime")


def _canonical_transform(node):
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Compare) and ast.unparse(child) == "liver_depth > boundary_depth + separation":
            child.comparators = [ast.Constant(1.)]
            count += 1
    return count


def _validator_transform(node):
    count = 0
    class Change(ast.NodeTransformer):
        def visit_Compare(self, child):
            nonlocal count
            if ast.unparse(child) == "liver_depth[tuple(context.T)] <= boundary + separation - 0.0001":
                # Exact strict predicate: depth==1 is excluded, independently
                # of the old tolerance intended for its old disjoint bands.
                child.comparators = [ast.Constant(1.)]
                count += 1
            return self.generic_visit(child)
        def visit_Assign(self, child):
            nonlocal count
            if len(child.targets) == 1 and isinstance(child.targets[0], ast.Name) and child.targets[0].id == "context_rows":
                count += 1
                return None
            return self.generic_visit(child)
        def visit_If(self, child):
            nonlocal count
            expression = ast.unparse(child.test)
            if expression == "any((tuple((int(v) for v in row)) in context_rows for row in liver))":
                count += 1
                return None
            if expression == "values.shape[0] == 0":
                child.test = ast.parse("values.shape[0] == 0 and name not in {'context', 'liver_surface'}", mode="eval").body
                count += 1
            return self.generic_visit(child)
    Change().visit(node)
    return count


def semantic_masks(fields, spatial):
    footprint = np.asarray(spatial._field(fields, "footprint"), dtype=bool)
    organ = np.asarray(spatial._field(fields, "organ_mask", "organ"), dtype=bool)
    outside = np.asarray(spatial._field(fields, "outside_tumor_mm", "tumor_outer"), dtype=np.float32)
    depth = np.asarray(spatial._field(fields, "organ_depth", "organ_boundary_mm", "liver_depth_mm"), dtype=np.float32)
    if (footprint.ndim != 3 or any(size < 1 for size in footprint.shape)
            or not (footprint.shape == organ.shape == outside.shape == depth.shape)
            or not np.isfinite(outside).all() or not np.isfinite(depth).all()
            or np.any(outside < 0) or np.any(depth < 0)):
        raise ValueError("Depth1 semantic fields must be complete finite nonnegative physical geometry")
    context = organ & ~footprint & (outside >= 2.) & (outside <= 10.) & (depth > 1.)
    surface = organ & ~footprint & (depth > 0.) & (depth <= 3.)
    return context, surface, dict(footprint=footprint, organ=organ, outside=outside, depth=depth)


def geometry_proof(fields, points, config, spacing, spatial):
    """Bind complete semantic-mask coverage and unchanged dense CT to the profile."""
    context, surface, values = semantic_masks(fields, spatial)
    for role, mask, pitch in (("context", context, config.canonical_context_spacing_mm),
                               ("liver_surface", surface, config.canonical_liver_spacing_mm)):
        actual = np.asarray(points[role])
        expected = spatial.voxel_grid_coordinates(np.argwhere(mask), np.asarray(spacing, np.float32), float(pitch))
        if (actual.dtype.kind not in "iu" or actual.shape != expected.shape or not np.array_equal(actual, expected)):
            raise ValueError("Depth1 canonical role must retain the complete original-grid semantic coverage: " + role)
    payload = dict(format=FORMAT, profile_sha256=identity()["contract_sha256"],
        semantic_context_voxels=int(np.count_nonzero(context)),
        semantic_liver_surface_voxels=int(np.count_nonzero(surface)),
        typed_overlap_voxels=int(np.count_nonzero(context & surface)),
        canonical_context_nodes=len(points["context"]),
        canonical_liver_surface_nodes=len(points["liver_surface"]),
        minimum_liver_depth_exclusive_mm=1., context_inner_radius_mm=2., context_outer_radius_mm=10.,
        roi_shape=list(map(int, context.shape)), spacing_mm=list(map(float, spacing)),
        full_fields_sha256={key: _hash_array(value) for key, value in values.items()},
        canonical_coordinate_sha256={key: _hash_array(value) for key, value in points.items()},
        dense_CT_patch_sha256=_hash_array(np.asarray(fields.model_input, dtype=np.float32)))
    payload["geometry_sha256"] = hashlib.sha256(_json(payload)).hexdigest()
    return payload


def _check_local(branch, profile):
    proof = branch.get(PROOF_KEY)
    if (not isinstance(proof, dict) or proof.get("format") != FORMAT
            or proof.get("profile_sha256") != profile
            or proof.get("canonical_nodes_sha256") != _node_hash(branch["nodes"])):
        raise ValueError("Depth1 canonical input has mixed profile or changed actual node tensors")
    if proof.get("geometry_sha256") != hashlib.sha256(_json({key: value for key, value in proof.items()
            if key not in ("geometry_sha256", "canonical_nodes_sha256", "source_identity")})).hexdigest():
        raise ValueError("Depth1 geometry proof content changed")
    role = "source" if "source_context" in branch["nodes"] else "target"
    if int(branch["nodes"][role + "_context"]["x"].shape[0]) != proof["canonical_context_nodes"]:
        raise ValueError("Depth1 context count differs from complete canonical geometry")
    if role == "source" and proof["canonical_context_nodes"] <= 0:
        raise ValueError("Depth1 donor source context remains mandatory; no empty-source fallback")
    return proof


def _target_proof_transform(node):
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and ast.unparse(child) == "boundary + separation":
            child.left, child.op, child.right = ast.Constant(1.), ast.Add(), ast.Constant(0.)
            count += 1
        if isinstance(child, ast.Name) and child.id == "FORMAT":
            # Changes are copied in-memory functions only; old module FORMAT
            # and its default identity stay untouched.
            pass
    class Format(ast.NodeTransformer):
        def visit_Name(self, child):
            if child.id == "FORMAT":
                return ast.copy_location(ast.Constant(FORMAT + "_target_absence"), child)
            return child
    Format().visit(node)
    return count


def _target_validator_transform(node):
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Compare) and ast.unparse(child.left) == "proof.get('minimum_liver_depth_exclusive_mm')":
            if ast.unparse(child.comparators[0]) != "4.0":
                raise ValueError("Original target absence depth marker differs")
            child.comparators = [ast.Constant(1.)]
            count += 1
    class Format(ast.NodeTransformer):
        def visit_Name(self, child):
            if child.id == "FORMAT":
                return ast.copy_location(ast.Constant(FORMAT + "_target_absence"), child)
            return child
    Format().visit(node)
    return count


def install(runtime, *, approved_profile=False):
    """Install once in a fresh original-snapshot process after the D adapter.

    Call after exact BEST loading and before new source/target preparation.
    No original cache may be reused as profile input. The model state remains
    byte-identical; callers must log this profile identity with new score output.
    """
    global _INSTALLED
    if approved_profile is not True:
        raise PermissionError("User-authorized depth>1mm inference profile must be explicitly selected")
    if _INSTALLED is not None:
        raise RuntimeError("Depth1 profile requires one fresh isolated process; do not switch existing profiles")
    from . import bounded_scope as scope
    from . import transition_v1_empty_context as target_adapter
    absence = sys.modules.get("hiercp_v1x.v24_matched_source_absence")
    if absence is not None and absence._INSTALLED is not None:
        raise RuntimeError("Depth1 profile cannot coexist with the unapproved empty-source proposal")
    if target_adapter._INSTALLED is None:
        raise RuntimeError("Install the exact original bounded10mm and D target adapter before depth1 profile")
    spatial, local, sample = (runtime[name] for name in ("spatial", "local", "sample"))
    # The strict runtime already verified module ownership before constructing
    # the model. PyG subsequently registers generated *_propagate modules with
    # temporary __file__ paths; they are not archived implementation modules.
    # Recheck every actual archive file without misclassifying generated code.
    snapshot = Path(runtime["snapshot"]).resolve(strict=True)
    for relative, expected in runtime["scope"]["original_module_sha256"].items():
        path = snapshot / relative
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Depth1 profile requires untouched exact original source: " + relative)
    selected = identity()
    profile = selected["contract_sha256"]
    original_coordinates = spatial.validate_canonical_coordinates
    raw_coordinates = inspect.unwrap(original_coordinates)
    validate = scope._patched(raw_coordinates, _validator_transform, 4)
    original_sets = spatial.canonical_coordinate_sets
    new_sets = scope._patched(original_sets, _canonical_transform, 1)
    original_prepare, original_build, original_view = local.prepare_local_source, local.build_local_graph, sample.build_local_view
    registry, lock = {}, threading.RLock()

    @functools.wraps(original_coordinates)
    def coordinates(fields, points, config):
        _scope(config, runtime)
        context, surface, _ = semantic_masks(fields, spatial)
        if len(points["context"]) == 0:
            if not target_adapter._TARGET.get():
                raise spatial.EmptyCanonicalNodeError("context", (0, 3))
            if np.any(context) or not np.any(surface) or len(points["liver_surface"]) == 0:
                raise ValueError("Depth1 target absence requires complete semantic absence and actual surface")
            target_adapter._PROOF.set(target_adapter.geometry_proof(fields, points, config, spatial))
        validate(fields, points, config)
        if len(points["liver_surface"]) == 0 and np.any(surface):
            raise ValueError("A present actual liver-surface role cannot be omitted")

    @functools.wraps(original_sets)
    def canonical(fields, config, spacing):
        _scope(config, runtime)
        points = new_sets(fields, config, spacing)
        _LAST.set(geometry_proof(fields, points, config, spacing, spatial))
        return points

    @functools.wraps(original_prepare)
    def prepare(case, source, **kwargs):
        token = _LAST.set(None)
        try:
            result = original_prepare(case, source, **kwargs)
            proof = copy.deepcopy(_LAST.get())
            if proof is None:
                raise ValueError("Depth1 prepared source lacks actual complete canonical geometry")
            proof["canonical_nodes_sha256"] = _node_hash(result.canonical_nodes)
            proof["source_identity"] = dict(case_id=case.paths.case_id, component_id=int(source.component_id),
                full_source_mask_sha256=_hash_array(source.full_mask))
            if proof["dense_CT_patch_sha256"] != _hash_array(result.source_patch):
                raise ValueError("Depth1 preparation changed original dense CT patch")
            key = result.canonical_nodes["source_context"]["x"]
            with lock:
                registry[id(key)] = (weakref.ref(key, lambda ref, item=id(key): registry.pop(item, None)), proof)
            result.v24_context_depth1_contract = profile
            result.v24_context_depth1_geometry = copy.deepcopy(proof)
            return result
        finally:
            _LAST.reset(token)

    @functools.wraps(original_build)
    def build(case, source, spec, **kwargs):
        token = _LAST.set(None)
        try:
            result = original_build(case, source, spec, **kwargs)
            key = result.source_local["nodes"]["source_context"]["x"]
            with lock:
                item = registry.get(id(key))
            if item is None or item[0]() is not key:
                raise ValueError("Old or foreign prepared source cannot enter depth1 construction")
            source_proof = copy.deepcopy(item[1])
            actual_id = dict(case_id=case.paths.case_id, component_id=int(source.component_id),
                full_source_mask_sha256=_hash_array(source.full_mask))
            if source_proof["source_identity"] != actual_id:
                raise ValueError("Depth1 matched CP must preserve exact same-patient source component")
            target_proof = copy.deepcopy(_LAST.get())
            if target_proof is None:
                raise ValueError("Depth1 target lacks actual complete canonical geometry")
            target_proof["canonical_nodes_sha256"] = _node_hash(result.target_local["nodes"])
            for role, proof in (("source", source_proof), ("target", target_proof)):
                if proof["dense_CT_patch_sha256"] != _hash_array(getattr(result, role + "_patch")):
                    raise ValueError("Depth1 construction changed original dense CT patch: " + role)
                branch = getattr(result, role + "_local")
                branch[PROOF_KEY] = proof
                _check_local(branch, profile)
            return result
        finally:
            _LAST.reset(token)

    @functools.wraps(original_view)
    def view(source_local, target_local, config, *, seed):
        _scope(config, runtime)
        source_proof, target_proof = (_check_local(branch, profile) for branch in (source_local, target_local))
        graph = original_view(source_local, target_local, config, seed=seed)
        graph[GRAPH_KEY] = profile
        graph.v24_context_depth1_geometry_sha256 = [source_proof["geometry_sha256"], target_proof["geometry_sha256"]]
        return graph

    target_adapter.geometry_proof = scope._patched(target_adapter.geometry_proof, _target_proof_transform, 2)
    target_adapter.validate_local_proof = scope._patched(target_adapter.validate_local_proof, _target_validator_transform, 1)
    for original, replacement in ((original_coordinates, coordinates), (original_sets, canonical),
                                   (original_prepare, prepare), (original_build, build), (original_view, view)):
        scope._rebind(original, replacement)
    _INSTALLED = dict(selected, snapshot=str(runtime["snapshot"]), scope_contract=runtime["scope"]["contract_sha256"])
    return copy.deepcopy(_INSTALLED)
