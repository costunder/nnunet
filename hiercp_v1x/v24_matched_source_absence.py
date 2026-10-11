"""Unenabled proposal: admit proven donor-context absence for matched CP inference.

This changes the completed GNN's *input admission*, not its learned equations.
It must therefore be explicitly approved before ``install`` is called. No
production entry point imports or enables it. All source entries and all CP
positions remain mandatory. Real context nodes may never be omitted.

An absent context uses only the original, already registered shell tokens.
No coordinate, CT feature, surface node, parameter or zero embedding is added.
The archived source files and the completed checkpoint remain unchanged.
CPU UNIT checks are not actual-CT, CUDA or complete-642 validation.
"""
from __future__ import annotations

import ast
from contextvars import ContextVar
import copy
from dataclasses import dataclass
import functools
import hashlib
import inspect
import json
from pathlib import Path
import threading
import types
import weakref

import numpy as np

FORMAT = "v24_matched_Basic_proven_source_target_context_absence_inference_v1"
PROOF_KEY = "v24_matched_context_absence"
GRAPH_PROOF_KEY = "v24_matched_context_absence_counts"
GEOMETRY_KEYS = {role: "v24_matched_" + role + "_absence_geometry_sha256"
                 for role in ("source", "target")}
POLICY = ("inference only; complete unchanged semantic context absent; "
          "nonempty actual liver surface; existing checkpoint shell tokens; no source or candidate reduction")
_SOURCE_REQUEST = ContextVar("matched_source_absence_source_request", default=False)
_SOURCE_PROOF = ContextVar("matched_source_absence_source_proof", default=None)
_TARGET_PROOF = ContextVar("matched_source_absence_target_proof", default=None)
_BUILD_ADMISSION = ContextVar("matched_source_absence_owned_build", default=False)
_BUILD_SOURCE_ID = ContextVar("matched_source_absence_same_patient_source", default=None)
_FORWARD = ContextVar("matched_source_absence_forward", default=None)
_INSTALLED = None


def identity():
    return dict(format=FORMAT, policy=POLICY,
                module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                enabled_by_default=False, changes_training_admission=True,
                actual_CT_validated=False, complete_642_validated=False,
                token_learning_claim="none; preserve exact checkpoint values, including unchanged zero initialization",
                scientific_limitation="jointly absent source shells were excluded by original training admission; inference extrapolation requires approval")


def checkpoint_token_audit(encoder):
    """Expose unchanged zero-initialized tokens; do not label them all trained."""
    import torch
    result = {}
    for role in ("source_context", "target_context"):
        for index in range(3):
            name = role + "_c" + str(index)
            value = encoder.empty_context_shell[name].detach().cpu()
            if value.shape != (128,) or not bool(torch.isfinite(value).all()):
                raise ValueError("Exact finite128D existing checkpoint shell token required")
            nonzero = int(torch.count_nonzero(value))
            result[name] = dict(nonzero_count=nonzero, l2_norm=float(torch.linalg.vector_norm(value)),
                equals_original_zero_initialization=nonzero == 0,
                token_values_sha256=_array_sha(value.numpy()), new_parameter=False,
                evidence_of_having_learned_from_values=nonzero > 0)
    return result


def _array_sha(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([array.dtype.str, list(array.shape)],
                                      separators=(",", ":")).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _node_sha(nodes):
    """Bind the proof to actual canonical node tensors before any sampling."""
    import torch
    digest = hashlib.sha256()
    for role in sorted(nodes):
        digest.update(role.encode())
        for key in sorted(nodes[role]):
            value = nodes[role][key]
            if not isinstance(value, torch.Tensor):
                raise TypeError("Canonical node proof requires real tensor fields")
            if value.device.type != "cpu":
                raise ValueError("Canonical source/target proof is computed before CUDA transfer")
            digest.update(key.encode())
            digest.update(_array_sha(value.detach().contiguous().numpy()).encode())
    return digest.hexdigest()


def _is_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _scope_geometry(config):
    actual = (float(config.context_inner_radius_mm), float(config.context_outer_radius_mm),
              float(config.boundary_depth_mm), float(config.context_liver_surface_separation_mm))
    if actual != (2., 10., 3., 1.):
        raise ValueError("Absence proposal preserves context2..10mm and liver depth strictly above4mm")
    return actual


def geometry_proof(fields, points, config, spatial, *, role):
    """Prove full semantic-mask absence; retain every original geometry check."""
    if role not in ("source", "target"):
        raise ValueError("Explicit source or target absence role required")
    inner, outer, boundary, separation = _scope_geometry(config)
    footprint = np.asarray(spatial._field(fields, "footprint"), dtype=bool)
    organ = np.asarray(spatial._field(fields, "organ_mask", "organ"), dtype=bool)
    outside = np.asarray(spatial._field(fields, "outside_tumor_mm", "tumor_outer"), dtype=np.float32)
    depth = np.asarray(spatial._field(fields, "organ_depth", "organ_boundary_mm", "liver_depth_mm"), dtype=np.float32)
    if (footprint.ndim != 3 or any(size <= 0 for size in footprint.shape)
            or not (footprint.shape == organ.shape == outside.shape == depth.shape)):
        raise ValueError("Complete source/target semantic proof fields have inconsistent shapes")
    if not np.isfinite(outside).all() or not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("Nonfinite or negative source/target semantic geometry")
    context = organ & ~footprint & (outside >= inner) & (outside <= outer) & (depth > boundary + separation)
    surface = organ & ~footprint & (depth > 0.) & (depth <= boundary)
    if np.any(context):
        raise ValueError("Present real " + role + " context cannot be omitted")
    values, liver = np.asarray(points["context"]), np.asarray(points["liver_surface"])
    if (values.shape != (0, 3) or liver.ndim != 2 or liver.shape[1:] != (3,)
            or len(liver) == 0 or not np.issubdtype(liver.dtype, np.integer)):
        raise ValueError("Absent context requires a nonempty real canonical liver-surface role")
    shape = np.asarray(footprint.shape)
    if (np.any(liver < 0) or np.any(liver >= shape[None])
            or np.unique(liver, axis=0).shape[0] != len(liver)
            or not np.all(surface[tuple(liver.T)])):
        raise ValueError("Absence proof liver-surface coordinates are not unique actual surface voxels")
    surface_count = int(np.count_nonzero(surface))
    if not surface_count:
        raise ValueError("Absent context requires actual liver-surface CT evidence")
    fields_sha = {name: _array_sha(value) for name, value in
                  (("footprint", footprint), ("organ", organ), ("outside", outside), ("depth", depth))}
    payload = dict(format=FORMAT, role=role, context_observed_absent=True,
        semantic_context_voxels=0, liver_surface_voxels=surface_count,
        canonical_liver_surface_nodes=int(len(liver)), roi_shape=list(map(int, footprint.shape)),
        context_inner_radius_mm=inner, context_outer_radius_mm=outer,
        minimum_liver_depth_exclusive_mm=boundary + separation,
        full_field_sha256=fields_sha, canonical_surface_coordinates_sha256=_array_sha(liver))
    payload["geometry_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return payload


def bind_proof(proof, nodes, *, source_binding=None):
    result = copy.deepcopy(proof)
    if result["role"] == "source":
        if (not isinstance(source_binding, dict) or set(source_binding) != {
                "case_id", "component_id", "source_mask_sha256", "source_CT_patch_sha256", "source_dense_patch_sha256"}
                or not isinstance(source_binding["case_id"], str) or not source_binding["case_id"]
                or type(source_binding["component_id"]) is not int or source_binding["component_id"] < 1
                or not _is_sha(source_binding["source_mask_sha256"])
                or not _is_sha(source_binding["source_CT_patch_sha256"])
                or not _is_sha(source_binding["source_dense_patch_sha256"])):
            raise ValueError("Absent source proof requires actual case/component/full-mask/CT binding")
        result["source_binding"] = copy.deepcopy(source_binding)
        geometry = {key: value for key, value in result.items() if key != "geometry_sha256"}
        result["geometry_sha256"] = hashlib.sha256(json.dumps(geometry, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    result["canonical_nodes_sha256"] = _node_sha(nodes)
    return result


def validate_local_proof(local, *, role, contract):
    """An empty canonical role must carry its complete, unchanged geometry proof."""
    if role not in ("source", "target"):
        raise ValueError("Explicit source or target canonical role required")
    if (local.get("v1x_bounded_scope_contract") != contract
            or local.get("v1x_bounded_scope_margin_mm") != 10.):
        raise ValueError("Absence canonical payload is outside the unchanged original10mm scope")
    nodes = local["nodes"]
    count = int(nodes[role + "_context"]["x"].shape[0])
    proof = local.get(PROOF_KEY)
    if count > 0:
        if proof is not None:
            raise ValueError("Present canonical context cannot carry an absence proof")
        return None
    surface_count = int(nodes[role + "_liver_surface"]["x"].shape[0])
    if (not isinstance(proof, dict) or proof.get("format") != FORMAT or proof.get("role") != role
            or proof.get("context_observed_absent") is not True
            or type(proof.get("semantic_context_voxels")) is not int or proof["semantic_context_voxels"] != 0
            or type(proof.get("liver_surface_voxels")) is not int
            or type(proof.get("canonical_liver_surface_nodes")) is not int
            or surface_count <= 0 or proof["canonical_liver_surface_nodes"] != surface_count
            or proof["liver_surface_voxels"] < surface_count
            or proof.get("context_inner_radius_mm") != 2.
            or proof.get("context_outer_radius_mm") != 10.
            or proof.get("minimum_liver_depth_exclusive_mm") != 4.
            or not isinstance(proof.get("roi_shape"), list) or len(proof["roi_shape"]) != 3
            or any(type(value) is not int or value <= 0 for value in proof["roi_shape"])
            or not isinstance(proof.get("full_field_sha256"), dict)
            or set(proof["full_field_sha256"]) != {"footprint", "organ", "outside", "depth"}
            or any(not _is_sha(value) for value in proof["full_field_sha256"].values())
            or not _is_sha(proof.get("canonical_surface_coordinates_sha256"))
            or not _is_sha(proof.get("geometry_sha256"))
            or proof.get("canonical_nodes_sha256") != _node_sha(nodes)):
        raise ValueError("Absent " + role + " context lacks complete real geometry and canonical tensor binding")
    geometry = {key: value for key, value in proof.items()
                if key not in ("geometry_sha256", "canonical_nodes_sha256")}
    expected = hashlib.sha256(json.dumps(geometry, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if expected != proof["geometry_sha256"]:
        raise ValueError("Absence geometry proof content changed")
    if role == "source":
        binding = proof.get("source_binding")
        if (not isinstance(binding, dict) or set(binding) != {
                "case_id", "component_id", "source_mask_sha256", "source_CT_patch_sha256", "source_dense_patch_sha256"}
                or not isinstance(binding["case_id"], str) or not binding["case_id"]
                or type(binding["component_id"]) is not int or binding["component_id"] < 1
                or not _is_sha(binding["source_mask_sha256"])
                or not _is_sha(binding["source_CT_patch_sha256"])
                or not _is_sha(binding["source_dense_patch_sha256"])):
            raise ValueError("Absent source canonical proof lost real donor component/CT binding")
    return proof


def proof_row(source_proof, target_proof):
    result = []
    for proof in (source_proof, target_proof):
        result.extend([0, -1, -1, -1] if proof is None else
                      [1, 0, proof["liver_surface_voxels"], proof["canonical_liver_surface_nodes"]])
    return result


@dataclass(frozen=True)
class ForwardAdmission:
    encoder_identity: int
    graph_count: int
    absent: dict


def validate_batch_provenance(batch, *, graph_count, node_types, contract):
    """Check each disjoint graph, including mixed absent/present source batches."""
    import torch
    proof = batch.get(GRAPH_PROOF_KEY)
    if (type(graph_count) is not int or graph_count < 1 or not isinstance(proof, torch.Tensor)
            or proof.dtype != torch.long or proof.shape != (graph_count, 8)):
        raise ValueError("Source/target absence proof must have one exact eight-field row per graph")
    markers = batch.get("v1x_bounded_scope_contract")
    if (not isinstance(markers, (list, tuple)) or len(markers) != graph_count
            or any(value != contract for value in markers)):
        raise ValueError("Absence graph batch has mixed or unbound original10mm ownership")
    canonical = batch.get("canonical_counts")
    if (not isinstance(canonical, torch.Tensor) or canonical.dtype != torch.long
            or canonical.shape != (graph_count, len(node_types))):
        raise ValueError("Complete canonical role counts are required for each sampled graph")
    absent = {}
    for role, offset in (("source", 0), ("target", 4)):
        rows = proof[:, offset:offset + 4]
        flags = rows[:, 0]
        missing = flags == 1
        if bool(torch.any((flags != 0) & ~missing)):
            raise ValueError("Malformed source/target observed-absence flag")
        if not bool(torch.all(missing[:, None] | (rows[:, 1:] == -1))):
            raise ValueError("Present context must retain the explicit no-absence sentinel row")
        if not bool(torch.all(~missing | ((rows[:, 1] == 0) & (rows[:, 2] >= rows[:, 3]) & (rows[:, 3] > 0)))):
            raise ValueError("Absent context requires zero semantic mask and nonempty actual canonical surface")
        checksums = batch.get(GEOMETRY_KEYS[role])
        if (not isinstance(checksums, (list, tuple)) or len(checksums) != graph_count
                or any(not (_is_sha(value) if flag else value == "present")
                       for value, flag in zip(checksums, missing.detach().cpu().tolist()))):
            raise ValueError("Each absent source/target graph requires its complete geometry digest")
        context_column = node_types.index(role + "_context")
        surface_column = node_types.index(role + "_liver_surface")
        if not bool(torch.all((canonical[:, context_column] == 0) == missing)):
            raise ValueError("Canonical context omission differs from full-semantic absence proof")
        if not bool(torch.all(~missing | (canonical[:, surface_column] == rows[:, 3]))):
            raise ValueError("Canonical liver-surface evidence differs from absence proof")
        absent[role + "_context"] = missing
    for node_type in node_types:
        owners = batch[node_type].batch
        if owners.dtype != torch.long or owners.ndim != 1 or bool(torch.any(owners < 0)):
            raise ValueError("Semantic node owners must be nonnegative disjoint graph indices")
        counts = torch.bincount(owners, minlength=graph_count)
        if counts.shape != (graph_count,):
            raise ValueError("Semantic node ownership exceeds physical graph count")
        if node_type in absent:
            if not bool(torch.all((counts == 0) == absent[node_type])):
                raise ValueError("Sampled context omission differs from canonical semantic absence")
        elif node_type in ("source_liver_surface", "target_liver_surface"):
            role = node_type.split("_", 1)[0]
            rows = proof[:, 0:4] if role == "source" else proof[:, 4:8]
            if not bool(torch.all(~absent[role + "_context"] | ((counts > 0) & (counts <= rows[:, 3])))):
                raise ValueError("Absent context has no retained actual sampled liver surface")
        elif bool(torch.any(counts == 0)):
            raise ValueError("Original mandatory semantic role missing: " + node_type)
    return absent


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
            child.test = ast.parse("ids.size == 0 and node_type not in {'source_liver_surface', 'target_liver_surface', 'source_context', 'target_context'}", mode="eval").body
            count += 1
    return count


def _shell_transform(node):
    """Remove only the global-empty guard, after per-graph evidence is checked."""
    count = 0
    class Remove(ast.NodeTransformer):
        def visit_If(self, child):
            nonlocal count
            if ast.unparse(child.test) == "batch_index.numel() == 0":
                count += 1
                return None
            return self.generic_visit(child)
    Remove().visit(node)
    return count


def _target_validation_transform(node):
    """Retain every existing target-proof obligation, remove donor guard only."""
    count = 0
    class Remove(ast.NodeTransformer):
        def visit_If(self, child):
            nonlocal count
            if ast.unparse(child.test) == "source_count <= 0":
                count += 1
                return None
            return self.generic_visit(child)
    Remove().visit(node)
    return count


def _archived_permitted_shell(runtime):
    """Read exact archived method; an installed AST clone has synthetic lines."""
    module = runtime["model"]
    source = Path(module.__file__)
    expected = runtime["scope"]["original_module_sha256"]["hiercp/model.py"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Permitted shell equations require byte-exact original archived model")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [method for cls in tree.body if isinstance(cls, ast.ClassDef)
             and cls.name == "LocalTumorContextPyGEncoder" for method in cls.body
             if isinstance(method, ast.FunctionDef) and method.name == "_pool_context_shells"]
    if len(nodes) != 1 or _shell_transform(nodes[0]) != 1:
        raise ValueError("Exactly one archived global-empty shell guard is required")
    selected = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), nodes[0]], type_ignores=[])
    namespace = dict(vars(module))
    exec(compile(ast.fix_missing_locations(selected), str(source), "exec"), namespace)
    return namespace["_pool_context_shells"]


def _guarded_shell_call(self, original, permitted, node_type, x, raw_x, batch_index, graph_count):
    import torch
    if node_type not in ("source_context", "target_context"):
        raise ValueError("Only original source/target context shell pooling is allowed")
    counts = torch.bincount(batch_index, minlength=graph_count)
    if counts.shape != (graph_count,):
        raise ValueError("Shell owners exceed physical graph count")
    if bool(torch.any(counts == 0)):
        admission = _FORWARD.get()
        if (not isinstance(admission, ForwardAdmission) or admission.encoder_identity != id(self)
                or admission.graph_count != graph_count
                or not bool(torch.equal(counts == 0, admission.absent[node_type]))):
            raise ValueError("Empty context shell pooling requires this graph's validated full-semantic proof")
        return permitted(self, node_type, x, raw_x, batch_index, graph_count)
    return original(self, node_type, x, raw_x, batch_index, graph_count)


def install(runtime, encoder, *, approved_inference_policy=False):
    """Explicit opt-in only, in a fresh matched-inference worker after BEST load.

    This is a proposal API. Callers must still record approval/identity in a new
    scoring admission and overlay; existing completed-run proofs cannot be
    relabelled as having trained under the new donor-absence policy.
    """
    global _INSTALLED
    if approved_inference_policy is not True:
        raise PermissionError("Explicit approval is required to change completed GNN donor-context inference admission")
    from . import bounded_scope as scope
    from . import transition_v1_empty_context as target_adapter
    from .v24_stunet import STUNetL0Encoder
    if not isinstance(encoder, STUNetL0Encoder) or encoder.training:
        raise ValueError("The actual completed-BEST STU L0 must already be installed in eval mode")
    contract = runtime["scope"]["contract_sha256"]
    selected = (contract, str(runtime["snapshot"]), id(encoder))
    if _INSTALLED is not None:
        if _INSTALLED != selected:
            raise RuntimeError("Source-absence inference proposal requires a fresh original-runtime worker")
        return dict(identity(), checkpoint_tokens=checkpoint_token_audit(encoder))
    if (float(runtime["scope"]["margin_mm"]) != 10.
            or target_adapter._INSTALLED != (contract, str(runtime["snapshot"]))):
        raise ValueError("Install exact original10mm and original recipient-absence runtime first")
    spatial, local, sample = (runtime[name] for name in ("spatial", "local", "sample"))
    allowed_coordinates = scope._patched(inspect.unwrap(spatial.validate_canonical_coordinates), _coordinate_transform, 1)
    allowed_view = scope._patched(inspect.unwrap(sample.build_local_view), _view_transform, 1)
    original_shell = encoder._pool_context_shells.__func__
    permitted_shell = _archived_permitted_shell(runtime)
    original_coordinates, original_prepare, original_build, original_view = (
        spatial.validate_canonical_coordinates, local.prepare_local_source,
        local.build_local_graph, sample.build_local_view)
    original_validate = target_adapter.validate_local_proof
    original_target_obligations = scope._patched(original_validate, _target_validation_transform, 1)
    registry, registry_lock = {}, threading.RLock()

    @functools.wraps(original_coordinates)
    def coordinates(fields, points, config):
        if np.asarray(points["context"]).shape[0] != 0:
            return original_coordinates(fields, points, config)
        role = "source" if _SOURCE_REQUEST.get() else "target" if target_adapter._TARGET.get() else None
        if role is None:
            return original_coordinates(fields, points, config)
        scope._scope_config(config, 10.)
        proof = geometry_proof(fields, points, config, spatial, role=role)
        allowed_coordinates(fields, points, config)
        if role == "source":
            _SOURCE_PROOF.set(proof)
        else:
            # Keep the existing target adapter's independently validated proof.
            original_coordinates(fields, points, config)
            _TARGET_PROOF.set(proof)

    @functools.wraps(original_prepare)
    def prepare(*args, **kwargs):
        request_token = _SOURCE_REQUEST.set(True)
        proof_token = _SOURCE_PROOF.set(None)
        try:
            result = original_prepare(*args, **kwargs)
            proof = _SOURCE_PROOF.get()
            if int(result.canonical_nodes["source_context"]["x"].shape[0]) == 0:
                if proof is None:
                    raise ValueError("Prepared empty donor lacks full original source geometry evidence")
                arguments = inspect.signature(inspect.unwrap(original_prepare)).bind(*args, **kwargs).arguments
                case, source = arguments["case"], arguments["source"]
                bound = bind_proof(proof, result.canonical_nodes, source_binding=dict(
                    case_id=case.paths.case_id, component_id=int(source.component_id),
                    source_mask_sha256=_array_sha(source.full_mask),
                    source_CT_patch_sha256=_array_sha(source.patch_image),
                    source_dense_patch_sha256=_array_sha(result.source_patch)))
                # dataclasses.replace(prepared) retains canonical_nodes but drops
                # extra attributes. Weak tensor identity follows the real nodes
                # across that unchanged transport, without keeping CT caches alive.
                tensor = result.canonical_nodes["source_context"]["x"]
                key = id(tensor)
                def release(reference, key=key):
                    with registry_lock:
                        if registry.get(key, (None,))[0] is reference:
                            registry.pop(key, None)
                with registry_lock:
                    registry[key] = weakref.ref(tensor, release), bound
            return result
        finally:
            _SOURCE_PROOF.reset(proof_token)
            _SOURCE_REQUEST.reset(request_token)

    @functools.wraps(original_validate)
    def validate(source_local, target_local):
        if _BUILD_ADMISSION.get():
            if int(source_local["nodes"]["source_context"]["x"].shape[0]) == 0:
                tensor = source_local["nodes"]["source_context"]["x"]
                with registry_lock:
                    item = registry.get(id(tensor))
                    if item is None or item[0]() is not tensor:
                        raise ValueError("Transported absent donor is not this worker's genuinely prepared source")
                    binding = item[1]["source_binding"]
                    if _BUILD_SOURCE_ID.get() != (binding["case_id"], binding["component_id"]):
                        raise ValueError("Absent donor admission is restricted to unchanged Basic same-patient component")
                    source_local[PROOF_KEY] = copy.deepcopy(item[1])
            proof = _TARGET_PROOF.get()
            if proof is not None:
                target_local[PROOF_KEY] = bind_proof(proof, target_local["nodes"])
        source_proof = validate_local_proof(source_local, role="source", contract=contract)
        target_proof = validate_local_proof(target_local, role="target", contract=contract)
        if source_proof is None:
            original_validate(source_local, target_local)
        else:
            # Donor admission was separately proven above. Keep ALL original
            # target count, surface, ROI, policy and nonempty/no-proof checks.
            original_target_obligations(source_local, target_local)
        return target_local.get(target_adapter.PROOF_KEY)

    @functools.wraps(original_build)
    def build(*args, **kwargs):
        token = _BUILD_ADMISSION.set(True)
        proof_token = _TARGET_PROOF.set(None)
        arguments = inspect.signature(inspect.unwrap(original_build)).bind(*args, **kwargs).arguments
        identity_token = _BUILD_SOURCE_ID.set((arguments["case"].paths.case_id,
                                               int(arguments["source"].component_id)))
        try:
            result = original_build(*args, **kwargs)
            validate(result.source_local, result.target_local)
            proof = result.source_local.get(PROOF_KEY)
            if (proof is not None and _array_sha(result.source_patch)
                    != proof["source_binding"]["source_dense_patch_sha256"]):
                raise ValueError("Transported absent donor changed its actual five-channel dense CT input")
            return result
        finally:
            _TARGET_PROOF.reset(proof_token)
            _BUILD_SOURCE_ID.reset(identity_token)
            _BUILD_ADMISSION.reset(token)

    @functools.wraps(original_view)
    def view(source_local, target_local, config, *, seed):
        scope._scope_config(config, 10.)
        validate(source_local, target_local)
        source_proof = validate_local_proof(source_local, role="source", contract=contract)
        target_proof = validate_local_proof(target_local, role="target", contract=contract)
        graph = allowed_view(source_local, target_local, config, seed=seed)
        import torch
        graph.v1x_bounded_scope_contract = contract
        graph.v1x_bounded_scope_margin_mm = 10.
        graph[GRAPH_PROOF_KEY] = torch.tensor([proof_row(source_proof, target_proof)], dtype=torch.long)
        graph[target_adapter.GRAPH_PROOF_KEY] = torch.tensor(
            [proof_row(None, target_proof)[4:]], dtype=torch.long)
        for role, proof in (("source", source_proof), ("target", target_proof)):
            graph[GEOMETRY_KEYS[role]] = proof["geometry_sha256"] if proof else "present"
        return graph

    original_forward = encoder.forward_graph
    @functools.wraps(original_forward)
    def forward(self, batch, source_map, target_map):
        import torch
        if self.training or torch.is_grad_enabled():
            raise ValueError("Proposed donor absence is approved only for no-grad BEST inference")
        graph_count = int(target_map.shape[0])
        if int(source_map.shape[0]) != graph_count:
            raise ValueError("Source and target dense ownership must match sampled physical graphs")
        absent = validate_batch_provenance(batch, graph_count=graph_count,
            node_types=self.node_types, contract=contract)
        token = _FORWARD.set(ForwardAdmission(id(self), graph_count, absent))
        try:
            return original_forward(batch, source_map, target_map)
        finally:
            _FORWARD.reset(token)

    def shells(self, node_type, x, raw_x, batch_index, graph_count):
        return _guarded_shell_call(self, original_shell, permitted_shell,
                                  node_type, x, raw_x, batch_index, graph_count)

    for original, replacement in ((original_coordinates, coordinates), (original_prepare, prepare),
                                   (original_build, build), (original_view, view)):
        scope._rebind(original, replacement)
    target_adapter.validate_local_proof = validate
    encoder.forward_graph = types.MethodType(forward, encoder)
    encoder._pool_context_shells = types.MethodType(shells, encoder)
    _INSTALLED = selected
    return dict(identity(), checkpoint_tokens=checkpoint_token_audit(encoder))
