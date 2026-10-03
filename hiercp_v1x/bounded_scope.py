"""Explicit bounded v1 spatial-scope adapter for isolated DEBUG snapshots.

Only the footprint bounding-box margin and context-annulus outer radius change.
The archived files stay byte-identical. Missing *real* liver-surface regions are
represented by empty semantic roles, never manufactured anchors or skipped data.
This adapter is not an admission or a production/quality-readiness declaration.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
import functools
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
import textwrap
from typing import Mapping
from zipfile import ZipFile

FORMAT = "hiercp_v1_bounded_bbox_scope_debug_v1"
OPTIONAL_ROLES = frozenset(("source_liver_surface", "target_liver_surface"))
ROOT = Path(__file__).resolve().parents[1]
_ACTIVE = None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _margin(value, inner=2.0):
    if isinstance(value, bool):
        raise ValueError("An explicit finite physical margin is required")
    value = float(value)
    if not math.isfinite(value) or value <= float(inner):
        raise ValueError("Bounded margin must exceed the preserved inner context radius")
    return value


def configure(base_graph, margin_mm):
    """Return the original graph configuration with only margin/outer changed."""
    from hiercp.schema import GraphBuildConfig, graph_config_from_dict
    if isinstance(base_graph, Mapping):
        base = graph_config_from_dict(copy.deepcopy(dict(base_graph)))
    elif isinstance(base_graph, GraphBuildConfig):
        base = base_graph
        base.validate()
    else:
        raise TypeError("Exact original GraphBuildConfig or mapping is required")
    margin = _margin(margin_mm, base.context_inner_radius_mm)
    selected = replace(base, adaptive_roi_margin_mm=margin,
                       context_outer_radius_mm=margin)
    selected.validate()  # Never alter the original resource caps to make it pass.
    return selected


def _scope_config(config, margin):
    config.validate()
    if (float(config.adaptive_roi_margin_mm) != margin
            or float(config.context_outer_radius_mm) != margin):
        raise ValueError("Graph configuration differs from the installed bounded scope")
    if (int(config.patch_size) != 48 or float(config.context_radius_mm) != 28.0
            or tuple(float(x) for x in config.context_shells_mm) != (4.0, 12.0, 28.0)):
        raise ValueError("Bounded scope must preserve CNN48 and native shell/coordinate scales")


def _function_ast(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    nodes = [x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(nodes) != 1:
        raise ValueError("One exact source function is required for an in-memory adapter")
    return nodes[0]


def _patched(function, transform, expected):
    node = _function_ast(function)
    count = transform(node)
    if count != expected:
        raise ValueError(f"Archived AST patch occurrence mismatch: {function.__qualname__}: {count}")
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(function.__globals__)
    exec(compile(module, function.__code__.co_filename, "exec"), namespace)
    patched = namespace[node.name]
    # The copied source references its live original module globals, including
    # the explicit bounded adaptive-shape function installed below.
    import types
    result = types.FunctionType(patched.__code__, function.__globals__, function.__name__,
                                function.__defaults__, function.__closure__)
    result.__kwdefaults__ = copy.deepcopy(function.__kwdefaults__)
    result.__annotations__ = copy.deepcopy(function.__annotations__)
    return result


def _remove_surface_expansion(node):
    count = 0
    class Remove(ast.NodeTransformer):
        def visit_If(self, child):
            nonlocal count
            if any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                   and call.func.id == "_expanded_surface_shape" for call in ast.walk(child)):
                count += 1
                return None
            return self.generic_visit(child)
    Remove().visit(node)
    return count


def _fix_sdf_normalization(node):
    count = 0
    for child in ast.walk(node):
        if (isinstance(child, ast.Assign) and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Name)
                and child.targets[0].id == "context_radius"):
            expected = 'float(_cfg(config, \'context_outer_radius_mm\', _cfg(config, \'context_radius_mm\', 24.0)))'
            if ast.unparse(child.value) != expected:
                raise ValueError("Archived SDF normalization expression differs")
            child.value = ast.parse("float(config.context_radius_mm)", mode="eval").body
            count += 1
    return count


def _bound_payload(node):
    expansion = _remove_surface_expansion(node)
    normalization = _fix_sdf_normalization(node)
    if expansion != 1 or normalization != 1:
        raise ValueError("Exactly one surface expansion and SDF normalization site required")
    return 2


def _allow_empty_coordinates(node):
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.If) and ast.unparse(child.test) == "values.shape[0] == 0":
            child.test = ast.BoolOp(op=ast.And(), values=[child.test,
                ast.parse("name != 'liver_surface'", mode="eval").body])
            count += 1
    return count


def _allow_empty_sample_roles(node):
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.If) and ast.unparse(child.test) == "ids.size == 0":
            child.test = ast.BoolOp(op=ast.And(), values=[child.test,
                ast.parse("node_type not in {'source_liver_surface', 'target_liver_surface'}",
                          mode="eval").body])
            count += 1
    return count


def _allow_missing_dense_owners(node):
    count = 0
    class Remove(ast.NodeTransformer):
        def visit_If(self, child):
            nonlocal count
            if ast.unparse(child.test) == "bool(torch.any(counts == 0))":
                count += 1
                return None
            return self.generic_visit(child)
    Remove().visit(node)
    return count


def _pool_all_graphs(node):
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and ast.unparse(child.func) == "self.pool[node_type]":
            if any(keyword.arg == "dim_size" for keyword in child.keywords):
                raise ValueError("Original role pooling unexpectedly already declares dim_size")
            child.keywords.append(ast.keyword(arg="dim_size",
                value=ast.parse("int(target_map.shape[0])", mode="eval").body))
            count += 1
    return count


def _rebind(original, replacement):
    """Update only aliases in the byte-verified original hiercp package."""
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "hiercp" or name.startswith("hiercp.")):
            continue
        for key, value in list(vars(module).items()):
            if value is original:
                setattr(module, key, replacement)


def _verify_snapshot(source):
    from .contracts import V1_ARCHIVE_SHA256
    source = Path(source).resolve(strict=True)
    archive = ROOT / "versions/v1/pipeline_v1_source.zip"
    if _sha(archive) != V1_ARCHIVE_SHA256:
        raise ValueError("Preserved v1 archive SHA256 differs")
    files = {}
    with ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if name.startswith("hiercp/") and name.endswith(".py"):
                path = source / name
                expected = hashlib.sha256(bundle.read(name)).hexdigest()
                if path.is_symlink() or not path.is_file() or _sha(path) != expected:
                    raise ValueError(f"Bounded adapter requires byte-exact original source: {name}")
                files[name] = expected
    for name, module in list(sys.modules.items()):
        if (name == "hiercp" or name.startswith("hiercp.")) and module is not None:
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(source):
                raise ValueError(f"Another hiercp implementation is already imported: {name}")
    return source, files


def install(margin_mm, expected_snapshot_root):
    """Install one explicit scope in a fresh process; never edit snapshot files."""
    global _ACTIVE
    margin = _margin(margin_mm)
    if _ACTIVE is not None:
        raise RuntimeError("Use a fresh isolated process for every bounded-scope branch")
    source, verified = _verify_snapshot(expected_snapshot_root)
    modules = {name: importlib.import_module("hiercp." + name)
               for name in ("schema", "spatial", "local", "sample", "model", "cache", "data")}
    _verify_snapshot(source)  # Verify actual loaded paths after all imports.
    spatial, local, sample, model = (modules[name] for name in ("spatial", "local", "sample", "model"))
    payload = dict(format=FORMAT, margin_mm=margin, context_outer_radius_mm=margin,
        ROI="full transformed footprint bounding box + ceil(margin/spacing) on each side; odd native shape",
        implicit_depth_expansion=False, mandatory_surface_search=False,
        optional_roles=sorted(OPTIONAL_ROLES), CNN_shape=[48, 48, 48],
        coordinate_normalization_mm=28.0, tumor_sdf_normalization_mm=28.0,
        context_shells_mm=[4.0, 12.0, 28.0],
        source_archive_sha256=_sha(ROOT / "versions/v1/pipeline_v1_source.zip"),
        adapter_sha256=_sha(__file__), original_module_sha256=verified,
        debug=True, production_ready=False, quality_verified=False)
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()
    payload["contract_sha256"] = identity

    patched_payload = _patched(spatial.build_patch_payload, _bound_payload, 2)
    patched_coordinates = _patched(spatial.validate_canonical_coordinates, _allow_empty_coordinates, 1)
    patched_view = _patched(sample.build_local_view, _allow_empty_sample_roles, 1)
    patched_dense = _patched(sample.sample_dense_features_variable, _allow_missing_dense_owners, 1)
    patched_forward = _patched(model.LocalTumorContextPyGEncoder.forward_graph, _pool_all_graphs, 1)

    def bounded_shape(footprint_shape, spacing, config, *, center_liver_depth_mm):
        import numpy as np
        _scope_config(config, margin)
        # Center depth does not choose the scope. Reject bad data rather than
        # allowing a NaN to disappear simply because this policy is bounded.
        if not math.isfinite(float(center_liver_depth_mm)) or float(center_liver_depth_mm) < 0:
            raise ValueError("Center liver depth must be finite and nonnegative")
        shape = np.asarray(footprint_shape, dtype=np.int64)
        pitch = spatial._positive_spacing(spacing)
        if shape.shape != (3,) or np.any(shape < 1):
            raise ValueError("Nonempty three-dimensional full footprint shape required")
        shape = spatial._odd(shape + 2 * np.ceil(margin / pitch).astype(np.int64))
        voxels = math.prod(int(x) for x in shape)
        if voxels > int(config.adaptive_roi_max_voxels):
            raise spatial.AdaptiveRoiBudgetError(
                f"Bounded ROI resource guard exceeded: margin_mm={margin}; requested_shape={tuple(shape)}; "
                f"requested_voxels={voxels}; voxel_budget={config.adaptive_roi_max_voxels}; no reduction/fallback")
        return tuple(int(x) for x in shape)

    @functools.wraps(spatial.validate_canonical_coordinates)
    def coordinates(fields, points, config):
        import numpy as np
        _scope_config(config, margin)
        patched_coordinates(fields, points, config)
        if len(points["liver_surface"]) == 0:
            organ = np.asarray(spatial._field(fields, "organ_mask", "organ"), dtype=bool)
            footprint = np.asarray(spatial._field(fields, "footprint"), dtype=bool)
            depth = np.asarray(spatial._field(fields, "organ_depth", "organ_boundary_mm", "liver_depth_mm"))
            if np.any(organ & ~footprint & (depth > 0) & (depth <= float(config.boundary_depth_mm))):
                raise ValueError("A present real liver surface cannot be silently omitted")

    original_prepare = local.prepare_local_source
    @functools.wraps(original_prepare)
    def prepare(*args, **kwargs):
        _scope_config(kwargs["config"], margin)
        result = original_prepare(*args, **kwargs)
        result.v1x_bounded_scope_contract = identity
        return result

    original_build = local.build_local_graph
    @functools.wraps(original_build)
    def build(*args, **kwargs):
        _scope_config(kwargs["config"], margin)
        prepared = kwargs.get("prepared_source")
        if prepared is not None and getattr(prepared, "v1x_bounded_scope_contract", None) != identity:
            raise ValueError("Native or other-scope prepared source cannot enter bounded construction")
        result = original_build(*args, **kwargs)
        for branch in (result.source_local, result.target_local):
            branch["v1x_bounded_scope_contract"] = identity
            branch["v1x_bounded_scope_margin_mm"] = margin
        return result

    @functools.wraps(sample.build_local_view)
    def view(source_local, target_local, config, *, seed):
        _scope_config(config, margin)
        for branch in (source_local, target_local):
            if (branch.get("v1x_bounded_scope_contract") != identity
                    or branch.get("v1x_bounded_scope_margin_mm") != margin):
                raise ValueError("Unbound native or mixed-scope canonical local payload")
        graph = patched_view(source_local, target_local, config, seed=seed)
        graph.v1x_bounded_scope_contract = identity
        graph.v1x_bounded_scope_margin_mm = margin
        return graph

    original_forward = model.LocalTumorContextPyGEncoder.forward_graph
    @functools.wraps(original_forward)
    def forward(self, batch, source_map, target_map):
        import torch
        marker = batch.get("v1x_bounded_scope_contract")
        markers = marker if isinstance(marker, (list, tuple)) else [marker]
        if not markers or any(value != identity for value in markers):
            raise ValueError("Local model batch is native or belongs to another bounded scope")
        graph_count = int(target_map.shape[0])
        for role in modules["schema"].LOCAL_NODE_TYPES:
            if role in OPTIONAL_ROLES:
                continue
            counts = torch.bincount(batch[role].batch, minlength=graph_count)
            if counts.numel() != graph_count or bool(torch.any(counts == 0)):
                raise ValueError(f"Bounded graph has a missing mandatory semantic role: {role}")
        return patched_forward(self, batch, source_map, target_map)

    replacements = ((spatial.adaptive_native_shape, bounded_shape),
        (spatial.build_patch_payload, patched_payload),
        (spatial.validate_canonical_coordinates, coordinates),
        (sample.sample_dense_features_variable, patched_dense),
        (local.prepare_local_source, prepare), (local.build_local_graph, build),
        (sample.build_local_view, view))
    for original, replacement in replacements:
        _rebind(original, replacement)
    model.LocalTumorContextPyGEncoder.forward_graph = forward
    model.HierarchicalPyGPlacementModel.architecture_version += "|bounded_scope_" + identity
    _ACTIVE = payload
    return copy.deepcopy(payload)
