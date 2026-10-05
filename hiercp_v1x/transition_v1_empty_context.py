"""Bound D-only recipient context absence to real10mm geometry.

No source/target coordinate is manufactured, removed or moved. An empty
recipient deep-parenchyma role is admitted only when its complete semantic
mask is empty and a real liver-surface role remains. The archived shell tokens
represent that observed absence. The unchanged bounded-scope adapter remains
the authority for every nonempty construction and for source geometry.
"""
from __future__ import annotations

import ast
from contextvars import ContextVar
import copy
import functools
import hashlib
from pathlib import Path

import numpy as np

FORMAT = "actual_original_v1_recipient_context_absence_v1"
PROOF_KEY = "transition_recipient_context_absence"
GRAPH_PROOF_KEY = "transition_recipient_context_absence_counts"
POLICY = "target-only full semantic mask absent; real target liver surface; original learned empty-shell tokens; source context mandatory"
_TARGET = ContextVar("D_original_recipient_coordinate_admission", default=False)
_PROOF = ContextVar("D_original_recipient_geometry_proof", default=None)
_INSTALLED = None


def identity():
    return dict(format=FORMAT, module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), policy=POLICY)


def _coordinate_transform(node):
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.If) and ast.unparse(child.test) == "values.shape[0] == 0":
            child.test = ast.parse("values.shape[0] == 0 and name not in {'context', 'liver_surface'}", mode="eval").body
            count += 1
    return count


def _view_transform(node):
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.If) and ast.unparse(child.test) == "ids.size == 0":
            child.test = ast.parse("ids.size == 0 and node_type not in {'source_liver_surface', 'target_liver_surface', 'target_context'}", mode="eval").body
            count += 1
    return count


def _shell_transform(node):
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.If) and ast.unparse(child.test) == "batch_index.numel() == 0":
            child.test = ast.parse("batch_index.numel() == 0 and node_type != 'target_context'", mode="eval").body
            count += 1
    return count


def geometry_proof(fields, points, config, spatial):
    """Recompute the complete unchanged semantic masks before allowing absence."""
    footprint = np.asarray(spatial._field(fields, "footprint"), dtype=bool)
    organ = np.asarray(spatial._field(fields, "organ_mask", "organ"), dtype=bool)
    outside = np.asarray(spatial._field(fields, "outside_tumor_mm", "tumor_outer"), dtype=np.float32)
    depth = np.asarray(spatial._field(fields, "organ_depth", "organ_boundary_mm", "liver_depth_mm"), dtype=np.float32)
    if not (footprint.shape == organ.shape == outside.shape == depth.shape):
        raise ValueError("Recipient context proof fields have inconsistent shapes")
    if not np.isfinite(outside).all() or not np.isfinite(depth).all():
        raise ValueError("Nonfinite recipient semantic context geometry")
    inner, outer = float(config.context_inner_radius_mm), float(config.context_outer_radius_mm)
    boundary = float(config.boundary_depth_mm)
    separation = float(config.context_liver_surface_separation_mm)
    context = organ & ~footprint & (outside >= inner) & (outside <= outer) & (depth > boundary + separation)
    surface = organ & ~footprint & (depth > 0.) & (depth <= boundary)
    if np.count_nonzero(context):
        raise ValueError("Present real recipient context cannot be omitted")
    values = np.asarray(points["context"])
    liver = np.asarray(points["liver_surface"])
    if values.shape != (0, 3) or liver.ndim != 2 or liver.shape[1:] != (3,) or len(liver) == 0:
        raise ValueError("Empty recipient context requires a nonempty real liver-surface role")
    if not np.count_nonzero(surface):
        raise ValueError("Empty recipient context has no actual liver-surface CT evidence")
    return dict(format=FORMAT, target_context_observed_absent=True,
        semantic_context_voxels=0, liver_surface_voxels=int(np.count_nonzero(surface)),
        canonical_liver_surface_nodes=int(len(liver)), roi_shape=list(map(int, footprint.shape)),
        context_inner_radius_mm=inner, context_outer_radius_mm=outer,
        minimum_liver_depth_exclusive_mm=boundary + separation, source_context_required=True)


def validate_local_proof(source_local, target_local):
    """Content-bound canonical proof; source context is mandatory for all rows."""
    source_count = int(source_local["nodes"]["source_context"]["x"].shape[0])
    target_count = int(target_local["nodes"]["target_context"]["x"].shape[0])
    surface_count = int(target_local["nodes"]["target_liver_surface"]["x"].shape[0])
    proof = target_local.get(PROOF_KEY)
    if source_count <= 0:
        raise ValueError("Original donor source context remains mandatory")
    if target_count > 0:
        if proof is not None:
            raise ValueError("Nonempty recipient context cannot carry an absence proof")
        return None
    if (not isinstance(proof, dict) or proof.get("format") != FORMAT
            or proof.get("target_context_observed_absent") is not True
            or proof.get("source_context_required") is not True
            or type(proof.get("semantic_context_voxels")) is not int or proof["semantic_context_voxels"] != 0
            or type(proof.get("liver_surface_voxels")) is not int
            or type(proof.get("canonical_liver_surface_nodes")) is not int
            or surface_count <= 0 or proof["canonical_liver_surface_nodes"] != surface_count
            or proof["liver_surface_voxels"] < surface_count
            or proof.get("context_inner_radius_mm") != 2.
            or proof.get("context_outer_radius_mm") != 10.
            or proof.get("minimum_liver_depth_exclusive_mm") != 4.
            or not isinstance(proof.get("roi_shape"), list) or len(proof["roi_shape"]) != 3
            or any(type(value) is not int or value <= 0 for value in proof["roi_shape"])):
        raise ValueError("Recipient context absence is not bound to complete10mm geometry and real liver surface")
    return proof


def install(runtime):
    """Install once after the exact10mm adapter, only from the actual D owner."""
    global _INSTALLED
    from . import bounded_scope as scope
    spatial, local, sample, model = (runtime[name] for name in ("spatial", "local", "sample", "model"))
    contract = runtime["scope"]["contract_sha256"]
    if float(runtime["scope"]["margin_mm"]) != 10.:
        raise ValueError("D recipient context absence requires the unchanged explicit10mm scope")
    selected = (contract, str(runtime["snapshot"]))
    if _INSTALLED is not None:
        if _INSTALLED != selected:
            raise RuntimeError("Recipient absence adapter cannot change original source or scope in one process")
        return
    bounded_coordinates = spatial.validate_canonical_coordinates
    allowed_coordinates = scope._patched(bounded_coordinates.__wrapped__, _coordinate_transform, 1)
    bounded_view = sample.build_local_view
    allowed_view = scope._patched(bounded_view.__wrapped__, _view_transform, 1)
    bounded_forward = model.LocalTumorContextPyGEncoder.forward_graph
    allowed_forward = scope._patched(bounded_forward.__wrapped__, scope._pool_all_graphs, 1)
    original_shells = model.LocalTumorContextPyGEncoder._pool_context_shells
    allowed_shells = scope._patched(original_shells, _shell_transform, 1)
    original_target = local._prepare_local_target
    bounded_build = local.build_local_graph

    @functools.wraps(bounded_coordinates)
    def coordinates(fields, points, config):
        scope._scope_config(config, 10.)
        context = np.asarray(points["context"])
        if context.shape[0] != 0 or not _TARGET.get():
            return bounded_coordinates(fields, points, config)
        proof = geometry_proof(fields, points, config, spatial)
        allowed_coordinates(fields, points, config)
        _PROOF.set(proof)

    @functools.wraps(original_target)
    def target(*args, **kwargs):
        token = _TARGET.set(True)
        try:
            return original_target(*args, **kwargs)
        finally:
            _TARGET.reset(token)

    @functools.wraps(bounded_build)
    def build(*args, **kwargs):
        token = _PROOF.set(None)
        try:
            result = bounded_build(*args, **kwargs)
            proof = _PROOF.get()
            if proof is not None:
                result.target_local[PROOF_KEY] = copy.deepcopy(proof)
            validate_local_proof(result.source_local, result.target_local)
            return result
        finally:
            _PROOF.reset(token)

    @functools.wraps(bounded_view)
    def view(source_local, target_local, config, *, seed):
        import torch
        proof = validate_local_proof(source_local, target_local)
        if proof is None:
            graph = bounded_view(source_local, target_local, config, seed=seed)
            values = [0, -1, -1, -1]
        else:
            scope._scope_config(config, 10.)
            for branch in (source_local, target_local):
                if (branch.get("v1x_bounded_scope_contract") != contract
                        or branch.get("v1x_bounded_scope_margin_mm") != 10.):
                    raise ValueError("Unbound original10mm recipient context proof")
            graph = allowed_view(source_local, target_local, config, seed=seed)
            graph.v1x_bounded_scope_contract = contract
            graph.v1x_bounded_scope_margin_mm = 10.
            values = [1, 0, proof["liver_surface_voxels"], proof["canonical_liver_surface_nodes"]]
        graph[GRAPH_PROOF_KEY] = torch.tensor([values], dtype=torch.long)
        return graph

    @functools.wraps(bounded_forward)
    def forward(self, batch, source_map, target_map):
        import torch
        graph_count = int(target_map.shape[0])
        proof = batch.get(GRAPH_PROOF_KEY)
        if proof is None:
            # Historical nonempty graphs retain exactly the previous path.
            return bounded_forward(self, batch, source_map, target_map)
        if not isinstance(proof, torch.Tensor) or proof.dtype != torch.long or proof.shape != (graph_count, 4):
            raise ValueError("Per-view recipient absence proof does not match disjoint graph ownership")
        absent = proof[:, 0] == 1
        if bool(torch.any((proof[:, 0] != 0) & ~absent)):
            raise ValueError("Malformed recipient context absence flag")
        if not bool(torch.all(absent[:, None] | (proof[:, 1:] == -1))):
            raise ValueError("Nonempty recipient graphs must retain the explicit no-absence proof row")
        if not bool(torch.any(absent)):
            return bounded_forward(self, batch, source_map, target_map)
        markers = batch.get("v1x_bounded_scope_contract")
        if not isinstance(markers, (list, tuple)) or len(markers) != graph_count or any(value != contract for value in markers):
            raise ValueError("Recipient context absence graph is outside the unchanged original10mm scope")
        for role in runtime["schema"].LOCAL_NODE_TYPES:
            counts = torch.bincount(batch[role].batch, minlength=graph_count)
            if counts.shape != (graph_count,):
                raise ValueError("Original semantic node ownership exceeds physical graph count")
            if role == "target_context":
                if not bool(torch.all((counts == 0) == absent)):
                    raise ValueError("Recipient context count differs from its observed absence proof")
            elif role == "target_liver_surface":
                if not bool(torch.all(~absent | ((counts > 0) & (counts <= proof[:, 3]) & (proof[:, 2] >= proof[:, 3]) & (proof[:, 1] == 0)))):
                    raise ValueError("Empty recipient context lacks its complete real liver-surface evidence")
            elif role not in scope.OPTIONAL_ROLES and bool(torch.any(counts == 0)):
                raise ValueError("Original mandatory semantic role is missing: " + role)
        return allowed_forward(self, batch, source_map, target_map)

    for original, replacement in ((bounded_coordinates, coordinates), (original_target, target),
                                   (bounded_build, build), (bounded_view, view)):
        scope._rebind(original, replacement)
    model.LocalTumorContextPyGEncoder.forward_graph = forward
    model.LocalTumorContextPyGEncoder._pool_context_shells = allowed_shells
    _INSTALLED = selected
