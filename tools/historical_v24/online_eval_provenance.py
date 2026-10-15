"""Byte-level contracts for online evaluation; no training lineage is inferred.

Historical comparisons validate the recorded contract, whereas verification reads
the original input paths again. Symlinks are followed without rewriting inputs or
replacing their recorded paths with their resolved targets.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

FORMAT = "online_eval_inputs_v1"
SIDES = ("basic", "hier")
ROLES = ("ground_truth", *SIDES)
_CASE_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class EvaluationProvenanceError(ValueError):
    """An evaluation contract or its referenced input bytes are invalid."""


def _canonical(value: Any) -> bytes:
    def check_keys(item: Any) -> None:
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise EvaluationProvenanceError("Contract JSON keys must be strings")
            for child in item.values():
                check_keys(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check_keys(child)

    check_keys(value)
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise EvaluationProvenanceError(f"Contract must contain finite JSON values: {exc}") from exc


def _cohort(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise EvaluationProvenanceError("cohort must be a nonempty ordered sequence")
    if any(not isinstance(item, str) or not _CASE_ID.fullmatch(item) for item in value):
        raise EvaluationProvenanceError("cohort contains an unsafe case ID")
    if len(set(value)) != len(value):
        raise EvaluationProvenanceError("cohort contains duplicate case IDs")
    return list(value)


def _keys(value: Any, expected: Sequence[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise EvaluationProvenanceError(f"{label} must contain exactly {list(expected)}")


def _sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise EvaluationProvenanceError(f"{label} must be a lowercase SHA-256 digest")


def _recorded_path(value: Any, label: str) -> PurePosixPath | PureWindowsPath:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise EvaluationProvenanceError(f"{label} must be an absolute path")
    windows = PureWindowsPath(value)
    path = windows if windows.is_absolute() else PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise EvaluationProvenanceError(f"{label} must be an absolute path without traversal")
    return path


def _absolute_input(value: Any) -> Path:
    try:
        path = Path(os.path.abspath(os.fspath(value)))
    except (TypeError, ValueError, OSError) as exc:
        raise EvaluationProvenanceError(f"Invalid input path: {value!r}") from exc
    _recorded_path(str(path), "input path")
    return path


def _file_signature(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _hash_file(path: Path) -> str:
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise EvaluationProvenanceError(f"Evaluation input is not a regular file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            finished = os.fstat(stream.fileno())
        after = path.stat()
    except (OSError, ValueError) as exc:
        if isinstance(exc, EvaluationProvenanceError):
            raise
        raise EvaluationProvenanceError(f"Cannot hash evaluation input {path}: {exc}") from exc
    # Windows stat() and fstat() can expose different ctime semantics for the
    # same unchanged file. Keep strict change detection within each API, and
    # bind the opened descriptor to the recorded path by file identity and size.
    same_path_version = _file_signature(before) == _file_signature(after)
    same_open_version = _file_signature(opened) == _file_signature(finished)
    same_file = (before.st_dev, before.st_ino, before.st_size) == (
        opened.st_dev, opened.st_ino, opened.st_size
    )
    if not (same_path_version and same_open_version and same_file):
        raise EvaluationProvenanceError(f"Evaluation input changed while hashing: {path}")
    return digest.hexdigest()


def _validate_recorded_contract(contract: Any) -> None:
    _keys(contract, (
        "format", "cohort", "ground_truth", "methods", "evaluation_definition",
        "file_inventory", "contract_sha256",
    ), "contract")
    if contract["format"] != FORMAT:
        raise EvaluationProvenanceError(f"Unsupported evaluation contract format: {contract['format']!r}")
    cohort = _cohort(contract["cohort"])
    if not isinstance(contract["evaluation_definition"], dict):
        raise EvaluationProvenanceError("evaluation_definition must be a JSON object")
    _keys(contract["ground_truth"], cohort, "ground_truth")
    _keys(contract["methods"], SIDES, "methods")
    recorded_hashes = {"ground_truth": contract["ground_truth"]}
    for side in SIDES:
        method = contract["methods"][side]
        _keys(method, ("trainer", "validation_dir", "predictions"), f"methods.{side}")
        if not isinstance(method["trainer"], str) or not method["trainer"].strip():
            raise EvaluationProvenanceError(f"methods.{side}.trainer is missing")
        _recorded_path(method["validation_dir"], f"methods.{side}.validation_dir")
        _keys(method["predictions"], cohort, f"methods.{side}.predictions")
        recorded_hashes[side] = method["predictions"]
    for role, hashes in recorded_hashes.items():
        for case_id, digest in hashes.items():
            _sha(digest, f"{role}.{case_id}")

    inventory = contract["file_inventory"]
    if not isinstance(inventory, list) or len(inventory) != len(cohort) * len(ROLES):
        raise EvaluationProvenanceError("file_inventory has an incomplete or extra cohort/role inventory")
    seen: set[tuple[str, str]] = set()
    role_paths: dict[str, set[PurePosixPath | PureWindowsPath]] = {role: set() for role in ROLES}
    for entry in inventory:
        _keys(entry, ("role", "case_id", "path", "sha256"), "file_inventory entry")
        role, case_id = entry["role"], entry["case_id"]
        if not isinstance(role, str) or role not in ROLES or case_id not in cohort:
            raise EvaluationProvenanceError("file_inventory contains an unknown role/case")
        key = (role, case_id)
        if key in seen:
            raise EvaluationProvenanceError(f"Duplicate file_inventory role/case: {key}")
        seen.add(key)
        path = _recorded_path(entry["path"], "file_inventory.path")
        if path in role_paths[role]:
            raise EvaluationProvenanceError(f"Multiple {role} cases reference the same input path: {path}")
        role_paths[role].add(path)
        if entry["sha256"] != recorded_hashes[role][case_id]:
            raise EvaluationProvenanceError(f"file_inventory SHA-256 disagrees with {role}.{case_id}")
        if role in SIDES:
            directory = _recorded_path(contract["methods"][role]["validation_dir"], "validation_dir")
            if path.parent != directory:
                raise EvaluationProvenanceError(f"Prediction inventory path is outside its validation_dir: {path}")
    expected = {(role, case_id) for role in ROLES for case_id in cohort}
    if seen != expected:
        raise EvaluationProvenanceError("file_inventory does not exactly cover the declared cohort and roles")
    _sha(contract["contract_sha256"], "contract_sha256")
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if hashlib.sha256(_canonical(payload)).hexdigest() != contract["contract_sha256"]:
        raise EvaluationProvenanceError("Evaluation contract checksum mismatch")


def build_evaluation_contract(
    *, cohort: Sequence[str], ground_truth_files: Mapping[str, Path],
    prediction_files: Mapping[str, Mapping[str, Path]], trainers: Mapping[str, str],
    evaluation_definition: dict[str, Any],
) -> dict[str, Any]:
    """Hash all declared files and bind exact case, method, and metric identities."""
    case_ids = _cohort(cohort)
    _keys(ground_truth_files, case_ids, "ground_truth_files")
    _keys(prediction_files, SIDES, "prediction_files")
    _keys(trainers, SIDES, "trainers")
    if not isinstance(evaluation_definition, dict):
        raise EvaluationProvenanceError("evaluation_definition must be a JSON object")
    for side in SIDES:
        _keys(prediction_files[side], case_ids, f"prediction_files.{side}")
    files = {"ground_truth": ground_truth_files, **prediction_files}
    hashes: dict[str, dict[str, str]] = {role: {} for role in ROLES}
    paths: dict[str, dict[str, Path]] = {role: {} for role in ROLES}
    inventory: list[dict[str, str]] = []
    for role in ROLES:
        for case_id in case_ids:
            path = _absolute_input(files[role][case_id])
            digest = _hash_file(path)
            paths[role][case_id] = path
            hashes[role][case_id] = digest
            inventory.append({"role": role, "case_id": case_id, "path": str(path), "sha256": digest})
    methods = {}
    for side in SIDES:
        directories = {path.parent for path in paths[side].values()}
        if len(directories) != 1:
            raise EvaluationProvenanceError(f"{side} predictions must share one validation directory")
        methods[side] = {
            "trainer": trainers[side], "validation_dir": str(next(iter(directories))),
            "predictions": hashes[side],
        }
    contract = {
        "format": FORMAT, "cohort": case_ids, "ground_truth": hashes["ground_truth"],
        "methods": methods,
        "evaluation_definition": json.loads(_canonical(evaluation_definition)),
        "file_inventory": inventory,
    }
    contract["contract_sha256"] = hashlib.sha256(_canonical(contract)).hexdigest()
    _validate_recorded_contract(contract)
    return contract


def verify_evaluation_contract(contract: Mapping[str, Any]) -> None:
    """Validate the entire contract and rehash bytes at every original input path."""
    _validate_recorded_contract(contract)
    for entry in contract["file_inventory"]:
        path = Path(entry["path"])
        if not path.is_absolute():
            raise EvaluationProvenanceError(
                f"Recorded evaluation input is not an absolute path on this host: {path}"
            )
        if _hash_file(path) != entry["sha256"]:
            raise EvaluationProvenanceError(
                f"Evaluation input bytes changed: {entry['role']}/{entry['case_id']}: {path}"
            )


def full_method_identity(contract: Mapping[str, Any], side: str) -> dict[str, Any]:
    """Return the recorded method identity without comparator or location fields."""
    _validate_recorded_contract(contract)
    if side not in SIDES:
        raise EvaluationProvenanceError(f"Unknown method side: {side!r}")
    method = contract["methods"][side]
    return copy.deepcopy({
        "cohort": contract["cohort"], "ground_truth": contract["ground_truth"],
        "evaluation_definition": contract["evaluation_definition"],
        "method": {"trainer": method["trainer"], "predictions": method["predictions"]},
    })


def contract_comparability(reference: Any, current: Any) -> dict[str, Any]:
    """Compare recorded identities without reading historical source paths.

Missing/unsupported provenance is explicitly unverified. Invalid contracts that
claim this schema raise; they are never relabeled as legacy results.
"""
    unavailable = []
    for label, contract in (("reference", reference), ("current", current)):
        if contract is None:
            unavailable.append(f"{label} has no recorded evaluation contract")
        elif not isinstance(contract, Mapping):
            raise EvaluationProvenanceError(f"{label} contract must be an object or None")
        elif contract.get("format") != FORMAT:
            unavailable.append(f"{label} has unsupported or missing provenance format")
        else:
            _validate_recorded_contract(contract)
    if unavailable:
        return {"status": "unverified_provenance", "reasons": unavailable}
    reasons = []
    for key in ("cohort", "ground_truth", "evaluation_definition"):
        if reference[key] != current[key]:
            reasons.append(f"{key} changed")
    for side in SIDES:
        for key in ("trainer", "predictions"):
            if reference["methods"][side][key] != current["methods"][side][key]:
                reasons.append(f"methods.{side}.{key} changed")
    return {
        "status": "not_comparable" if reasons else "matched_inputs", "reasons": reasons,
        "reference_contract_sha256": reference["contract_sha256"],
        "current_contract_sha256": current["contract_sha256"],
    }


def prepare_new_output(path: str | Path) -> Path:
    """Create a new output directory, refusing every existing destination."""
    output = _absolute_input(path)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise EvaluationProvenanceError(f"Cannot create new evaluation output {output}: {exc}") from exc
    return output
