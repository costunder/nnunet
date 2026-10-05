"""Exact current data-contract definitions without importing current graph code.

Only named pure AST definitions are compiled. The original file's imports,
module-level statements and graph preparers are never executed. Transport uses
the explicitly activated original ``hiercp.common`` data types and operators.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import importlib
import inspect
import math
from pathlib import Path


POLICY = "same_donor_live_v1"
FORMAT = "native30_exact_AST_current_data_contract_v1"
_ROOT = Path(__file__).resolve().parents[1]
_CACHE = {}
_DEFINITIONS = {"l0_regions/donor_data.py": ("assignment",),
                "hiercp_v22/data.py": ("SourceCollection", "sources", "donor_in_target_spacing")}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _definitions(path, names):
    """Select unmodified complete definitions; do not execute source imports."""
    tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
    if len(selected) != len(names) or {node.name for node in selected} != set(names):
        raise ValueError("Exactly one original AST definition required for every data operator: " + str(path))
    definitions = ast.Module(body=selected, type_ignores=[])
    proof = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
             for node in selected}
    return compile(definitions, str(path), "exec"), proof


def _binding(source_root=None):
    import numpy as np
    from scipy import ndimage as ndi
    common = importlib.import_module("hiercp.common")
    learning = importlib.import_module("l0_regions.donor_learning")
    if learning.POLICY != POLICY:
        raise ValueError("Current exact same-donor policy changed")
    root = _ROOT if source_root is None else Path(source_root).resolve(strict=True)
    paths = {name: (root / name).resolve(strict=True) for name in _DEFINITIONS}
    if any(not path.is_relative_to(root) or not path.is_file() for path in paths.values()):
        raise ValueError("Explicit regular current data-contract source files required")
    common_path = Path(common.__file__).resolve(strict=True)
    learning_path = Path(inspect.getsourcefile(learning.validate_rows)).resolve(strict=True)
    for name in ("SourceTumor", "bbox_of_mask", "stable_case_seed"):
        if inspect.getsourcefile(getattr(common, name)) is None:
            raise ValueError("Actual activated original common definition required: " + name)
        if Path(inspect.getsourcefile(getattr(common, name))).resolve(strict=True) != common_path:
            raise ValueError("Original common data operator is replaced or belongs to another source: " + name)
    files = {name: {"path": str(path), "sha256": _sha(path)} for name, path in paths.items()}
    dependencies = {"hiercp.common": {"path": str(common_path), "sha256": _sha(common_path)},
                    "l0_regions.donor_learning.validate_rows": {"path": str(learning_path), "sha256": _sha(learning_path)}}
    key = (str(root), str(common_path))
    signature = (files, dependencies)
    cached = _CACHE.get(key)
    if cached is not None:
        if cached["signature"] != signature:
            raise ValueError("Bound exact data-contract source/dependency bytes changed during evaluation")
        return cached
    environment = {"__name__": __name__ + "._exact_definitions", "np": np, "math": math,
                   "ndi": ndi, "replace": replace, "SourceTumor": common.SourceTumor,
                   "bbox_of_mask": common.bbox_of_mask, "stable_case_seed": common.stable_case_seed,
                   "POLICY": POLICY, "validate_rows": learning.validate_rows}
    hashes = {}
    for name, names in _DEFINITIONS.items():
        code, definition_hashes = _definitions(paths[name], names)
        exec(code, environment)
        hashes[name] = definition_hashes
    receipt = dict(format=FORMAT, source_root=str(root), source_files=files,
                   exact_AST_definition_sha256=hashes, actual_dependencies=dependencies,
                   policy=POLICY, original_file_imports_executed=False,
                   algorithm_rewritten=False, current_graph_module_imported=False)
    cached = dict(signature=signature, environment=environment, receipt=receipt)
    _CACHE[key] = cached
    return cached


def assignment(meta, seed, *, source_root=None):
    return _binding(source_root)["environment"]["assignment"](meta, seed)


def sources(case, pad, maximum, *, source_root=None):
    return _binding(source_root)["environment"]["sources"](case, pad, maximum)


def donor_in_target_spacing(source, donor_spacing, target_spacing, *, source_root=None):
    return _binding(source_root)["environment"]["donor_in_target_spacing"](source, donor_spacing, target_spacing)


def source_receipt(*, source_root=None):
    """Freshly verify consumed bytes and return their independent receipt."""
    import copy
    return copy.deepcopy(_binding(source_root)["receipt"])
