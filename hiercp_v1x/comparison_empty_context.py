"""Scoped comparison admission of genuinely absent recipient context.

The complete 2--10 mm/depth>4 mm semantic mask must be empty and a real
recipient liver surface must remain. Existing learned empty-shell parameters
represent absence. Source context, every existing node, and the original
nonempty computation are preserved. No D runtime installation is performed.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import copy
import functools
import hashlib
import json
from pathlib import Path
import sys
import threading

import numpy as np

from . import transition_v1_empty_context as _geometry

FORMAT = "comparison_original_v1_recipient_context_absence_v1"
PROOF_KEY = "comparison_recipient_context_absence"
GRAPH_PROOF_KEY = "comparison_recipient_context_absence_counts"
GRAPH_POLICY_KEY = "comparison_recipient_context_policy"
POLICY = _geometry.POLICY
_TARGET = ContextVar("comparison_recipient_coordinate_admission", default=False)
_PROOF = ContextVar("comparison_recipient_geometry_proof", default=None)
_LOCK = threading.Lock()
_ACTIVE = None


def identity():
    """Bind both this adapter and its unchanged shared geometry validator."""
    value = dict(format=FORMAT, policy=POLICY,
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        geometry_validator_sha256=hashlib.sha256(Path(_geometry.__file__).read_bytes()).hexdigest())
    value["policy_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return value


def geometry_proof(fields, points, config, spatial, *, policy_sha256=None):
    """Prove absence from all unchanged semantic voxels, not sampled nodes."""
    # The shared function is pure; its D-only global installation is never used.
    proof = _geometry.geometry_proof(fields, points, config, spatial)
    proof["comparison_format"] = FORMAT
    proof["policy_sha256"] = policy_sha256 or identity()["policy_sha256"]
    return proof


def validate_local_proof(source_local, target_local, *, policy_sha256=None):
    """Admit old nonempty records and require exact policy for empty records."""
    proof = target_local.get(PROOF_KEY)
    if proof is not None:
        expected = policy_sha256 or identity()["policy_sha256"]
        if (not isinstance(proof, dict) or proof.get("comparison_format") != FORMAT
                or proof.get("policy_sha256") != expected):
            raise ValueError("Comparison recipient absence proof has a different geometry policy")
    translated = dict(target_local)
    # Do not accept a D-only proof as comparison provenance.
    translated.pop(_geometry.PROOF_KEY, None)
    if proof is not None:
        translated[_geometry.PROOF_KEY] = proof
    return _geometry.validate_local_proof(source_local, translated)


def _replacements(runtime, receipt):
    from . import bounded_scope as scope
    spatial, local, sample, model = (runtime[name] for name in ("spatial", "local", "sample", "model"))
    contract = runtime["scope"]["contract_sha256"]
    policy = receipt["policy_sha256"]
    if (float(runtime["scope"]["margin_mm"]) != 10.
            or scope._ACTIVE is None or scope._ACTIVE["contract_sha256"] != contract):
        raise ValueError("Comparison recipient absence requires the active exact 10 mm scope")
    snapshot = Path(runtime["snapshot"]).resolve(strict=True)
    for name in ("spatial", "local", "sample", "model", "schema"):
        module = runtime[name]
        if (sys.modules.get("hiercp." + name) is not module
                or not Path(module.__file__).resolve().is_relative_to(snapshot)):
            raise ValueError("Comparison recipient absence requires the bound original modules")
    if _geometry._INSTALLED is not None:
        raise RuntimeError("A D recipient adapter cannot share the comparison execution scope")

    bounded_coordinates = spatial.validate_canonical_coordinates
    allowed_coordinates = scope._patched(bounded_coordinates.__wrapped__, _geometry._coordinate_transform, 1)
    bounded_view = sample.build_local_view
    allowed_view = scope._patched(bounded_view.__wrapped__, _geometry._view_transform, 1)
    cls = model.LocalTumorContextPyGEncoder
    bounded_forward = cls.forward_graph
    allowed_forward = scope._patched(bounded_forward.__wrapped__, scope._pool_all_graphs, 1)
    original_shells = cls._pool_context_shells
    allowed_shells = scope._patched(original_shells, _geometry._shell_transform, 1)
    original_target = local._prepare_local_target
    bounded_build = local.build_local_graph

    @functools.wraps(bounded_coordinates)
    def coordinates(fields, points, config):
        scope._scope_config(config, 10.)
        context = np.asarray(points["context"])
        if context.shape[0] != 0 or not _TARGET.get():
            return bounded_coordinates(fields, points, config)
        proof = geometry_proof(fields, points, config, spatial, policy_sha256=policy)
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
            validate_local_proof(result.source_local, result.target_local, policy_sha256=policy)
            return result
        finally:
            _PROOF.reset(token)

    @functools.wraps(bounded_view)
    def view(source_local, target_local, config, *, seed):
        import torch
        proof = validate_local_proof(source_local, target_local, policy_sha256=policy)
        if proof is None:
            graph = bounded_view(source_local, target_local, config, seed=seed)
            values = [0, -1, -1, -1]
        else:
            scope._scope_config(config, 10.)
            for branch in (source_local, target_local):
                if (branch.get("v1x_bounded_scope_contract") != contract
                        or branch.get("v1x_bounded_scope_margin_mm") != 10.):
                    raise ValueError("Unbound original 10 mm recipient context proof")
            graph = allowed_view(source_local, target_local, config, seed=seed)
            graph.v1x_bounded_scope_contract = contract
            graph.v1x_bounded_scope_margin_mm = 10.
            values = [1, 0, proof["liver_surface_voxels"], proof["canonical_liver_surface_nodes"]]
        # Identical metadata schema is needed for nonempty/empty disjoint batches.
        graph[GRAPH_PROOF_KEY] = torch.tensor([values], dtype=torch.long)
        graph[GRAPH_POLICY_KEY] = policy
        return graph

    @functools.wraps(bounded_forward)
    def forward(self, batch, source_map, target_map):
        import torch
        graph_count = int(target_map.shape[0])
        proof = batch.get(GRAPH_PROOF_KEY)
        if proof is None:
            # Historical nonempty materializations retain the original path.
            return bounded_forward(self, batch, source_map, target_map)
        if not isinstance(proof, torch.Tensor) or proof.dtype != torch.long or proof.shape != (graph_count, 4):
            raise ValueError("Per-view recipient absence proof differs from disjoint graph ownership")
        policies = batch.get(GRAPH_POLICY_KEY)
        if (not isinstance(policies, (list, tuple)) or len(policies) != graph_count
                or any(value != policy for value in policies)):
            raise ValueError("Sampled recipient absence proof has a different comparison policy")
        absent = proof[:, 0] == 1
        if bool(torch.any((proof[:, 0] != 0) & ~absent)):
            raise ValueError("Malformed recipient context absence flag")
        if not bool(torch.all(absent[:, None] | (proof[:, 1:] == -1))):
            raise ValueError("Nonempty recipient graphs require the no-absence proof row")
        if not bool(torch.any(absent)):
            return bounded_forward(self, batch, source_map, target_map)
        markers = batch.get("v1x_bounded_scope_contract")
        if (not isinstance(markers, (list, tuple)) or len(markers) != graph_count
                or any(value != contract for value in markers)):
            raise ValueError("Recipient absence graph is outside the original 10 mm scope")
        if int(source_map.shape[0]) != graph_count:
            raise ValueError("Recipient absence requires complete paired source/target dense maps")
        for role in runtime["schema"].LOCAL_NODE_TYPES:
            counts = torch.bincount(batch[role].batch, minlength=graph_count)
            if counts.shape != (graph_count,):
                raise ValueError("Original semantic node ownership exceeds physical graph count")
            if role == "target_context":
                if not bool(torch.all((counts == 0) == absent)):
                    raise ValueError("Recipient context count differs from its observed absence proof")
            elif role == "target_liver_surface":
                if not bool(torch.all(~absent | ((counts > 0) & (counts <= proof[:, 3])
                        & (proof[:, 2] >= proof[:, 3]) & (proof[:, 1] == 0)))):
                    raise ValueError("Empty recipient context lacks complete real liver-surface evidence")
            elif role not in scope.OPTIONAL_ROLES and bool(torch.any(counts == 0)):
                raise ValueError("Original mandatory semantic role is missing: " + role)
        return allowed_forward(self, batch, source_map, target_map)

    aliases = ((bounded_coordinates, coordinates), (original_target, target),
               (bounded_build, build), (bounded_view, view))
    methods = ((cls, "forward_graph", bounded_forward, forward),
               (cls, "_pool_context_shells", original_shells, allowed_shells))
    return aliases, methods


@contextmanager
def activated(runtime):
    """Temporarily activate after bounded scope installation; restore on error.

    The owner must finish its workers before leaving this process-wide scope.
    Target-coordinate permissions/proofs remain thread-local ContextVars. A
    second or nested owner is rejected rather than weakening another execution.
    """
    global _ACTIVE
    from . import bounded_scope as scope
    token = object()
    with _LOCK:
        if _ACTIVE is not None:
            raise RuntimeError("A comparison recipient absence scope is already active")
        _ACTIVE = token
    applied_aliases, applied_methods = [], []
    try:
        receipt = identity()
        aliases, methods = _replacements(runtime, receipt)
        for original, replacement in aliases:
            applied_aliases.append((original, replacement))
            scope._rebind(original, replacement)
        for cls, name, original, replacement in methods:
            applied_methods.append((cls, name, original, replacement))
            setattr(cls, name, replacement)
        yield copy.deepcopy(receipt)
    finally:
        # Includes aliases imported while active; their old behavior is restored.
        for cls, name, original, replacement in reversed(applied_methods):
            setattr(cls, name, original)
        for original, replacement in reversed(applied_aliases):
            scope._rebind(replacement, original)
        with _LOCK:
            _ACTIVE = None
