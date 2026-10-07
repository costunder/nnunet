"""Verify an already independent v1.8 continuation without its former locks.

An initial migration still requires the original inactive-source checks. Once
sealed, each independent arm owns its checkpoint and cache namespace; another
arm using the historical source cannot block this arm's exact resume.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import threading

from . import u_bridge_continuation as publication

FORMAT = "v18_owned_continuation_execution_v1"
_CONTEXT = threading.RLock()


def _safe(path):
    path = Path(path)
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError(f"Owned continuation cannot traverse a symlink: {ancestor}")
    return path.resolve()


def verify_owned_continuation(source_root, destination_root, arm):
    """Verify own immutable copies; leave current mutable state to the engine.

    No old source file, process, lock or data directory is inspected. The exact
    historical source identity remains in the receipt. The unchanged engine
    verifies model/Adam/RNG/checkpoint cursor and the measured execution lock.
    """
    if arm not in ("selected", "native"):
        raise ValueError("One declared v1.8 arm is required")
    root = _safe(destination_root)
    source = Path(source_root).resolve()
    publication._disjoint(source, root, "the historical experiment")
    receipt_path = root / publication.RECEIPT
    receipt = publication._read(receipt_path)
    signed = dict(receipt)
    checksum = signed.pop("receipt_sha256", None)
    if checksum != publication._digest(signed):
        raise ValueError("Owned continuation receipt identity differs")
    request = receipt.get("request")
    expected = dict(format=publication.FORMAT, source_root=str(source),
                    destination_root=str(root), data_root=str(root / "data"), arm=arm)
    if not isinstance(request, dict) or any(request.get(key) != value for key, value in expected.items()):
        raise ValueError("Owned continuation source/arm/destination identity differs")
    records = receipt.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Owned continuation requires its complete publication inventory")
    seen = set()
    required = {"experiment.json", "initial.pt", "calibration.json"}
    copied_sealed = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("Owned continuation file record is malformed")
        name = record["path"]
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name
                or name in seen or not relative.parts):
            raise ValueError("Owned continuation file path is unsafe or duplicated")
        seen.add(name)
        role = record.get("role")
        if role == "sealed":
            if (len(relative.parts) != 1 or name not in required
                    and not (name.startswith("calibration_") and name.endswith(".json"))):
                raise ValueError("Owned continuation sealed inventory differs")
            copied_sealed.add(name)
        elif role == "arm_identity":
            if name != arm + "/training_identity.json":
                raise ValueError("Owned continuation arm identity path differs")
        elif role == "arm_mutable":
            if len(relative.parts) < 2 or relative.parts[0] != arm:
                raise ValueError("Owned continuation mutable arm path differs")
        elif role == "immutable_cache":
            if (len(relative.parts) < 3 or relative.parts[0] != "data"
                    or relative.parts[1] not in ("canonical_local", "whole_case_fields", "upper_static", "regions")):
                raise ValueError("Owned continuation immutable cache path differs")
        else:
            raise ValueError("Owned continuation file role differs")
        path = _safe(root / relative)
        if role == "arm_mutable":
            continue
        if (not path.is_file() or path.stat().st_size != record.get("bytes")
                or publication._sha(path) != record.get("sha256")):
            raise ValueError(f"Owned continuation immutable bytes changed: {path}")
    if not required.issubset(copied_sealed):
        raise ValueError("Owned continuation is missing sealed initial/calibration/experiment bytes")
    contract = publication._sealed_contract(root)
    if request.get("contract_sha256") != contract["sha256"]:
        raise ValueError("Owned continuation sealed experiment identity differs")
    latest = _safe(root / arm / "checkpoint_latest.pt")
    if receipt.get("resumed") is True and not latest.is_file():
        raise ValueError("Owned continuation saved progress is missing; no initial-state fallback")
    return dict(request, receipt=str(receipt_path), reused=True, resumed=latest.is_file(),
                file_count=len(records), methods=receipt["methods"],
                source_lock_dependency=False, own_immutable_bytes_verified=True)


@contextmanager
def independent_continuation_execution():
    """Use owned receipt validation in the unchanged independent runner.

    New migrations retain every original source inactivity check. All aliases
    are restored even if checkpoint restoration or training raises an error.
    """
    with _CONTEXT:
        original = publication.prepare_continuation
        def prepare(source_root, destination_root, arm):
            if (Path(destination_root) / publication.RECEIPT).exists():
                return verify_owned_continuation(source_root, destination_root, arm)
            return original(source_root, destination_root, arm)
        publication.prepare_continuation = prepare
        try:
            yield
        finally:
            publication.prepare_continuation = original
