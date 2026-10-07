"""Copy an inactive, sealed v1.8 arm into an independent execution namespace.

The original neural identity stays sealed. Published cache payloads may share
read-only file bytes through hardlinks; checkpoints and logs never do. This
module changes execution locations only and performs no training or prediction.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import uuid
import zipfile

FORMAT = "v18_independent_arm_continuation_v1"
RECEIPT = "continuation.json"
PENDING = ".continuation.pending.json"
_HEX = re.compile(r"[0-9a-f]{64}")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Regular published JSON required: {path}")
    return json.loads(path.read_text(encoding="utf8"))


def _new_json(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def _pid_alive(pid):
    import psutil
    return psutil.pid_exists(pid)


def _inactive(path):
    if not path.exists() and not path.is_symlink():
        return
    owner = _read(path)
    if owner.get("host") != socket.gethostname() or type(owner.get("pid")) is not int or owner["pid"] <= 0:
        raise RuntimeError(f"Continuation source lock has another/unknown host or owner: {path}; {owner}")
    if _pid_alive(owner["pid"]):
        raise RuntimeError(f"Continuation source is still active: {path}; {owner}")
    if _read(path) != owner:
        raise RuntimeError(f"Continuation source lock changed during inspection: {path}")


def _regular_tree(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Regular cache directory required: {root}")
    # A resolved path must not hide an ancestor symlink.
    for parent in (root, *root.parents):
        if parent.is_symlink():
            raise ValueError(f"Cache/source path cannot traverse a symlink: {parent}")


def _disjoint(left, right, label):
    if left == right or left.is_relative_to(right) or right.is_relative_to(left):
        raise ValueError(f"Continuation namespaces must be disjoint from {label}: {left}; {right}")


def _record(path, relative, *, role, verified_sha=None):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Regular published file required: {path}")
    return dict(path=relative.as_posix(), sha256=verified_sha or _sha(path), bytes=path.stat().st_size, role=role)


def _sealed_contract(source):
    contract = _read(source / "experiment.json")
    body = dict(contract); checksum = body.pop("sha256", None)
    if (contract.get("format") != "v18_u_bridge_matched_experiment_v1"
            or checksum != _digest(body)):
        raise ValueError("The source v1.8 experiment is not an intact sealed contract")
    calibration = _read(source / "calibration.json")
    if (calibration.get("contract_sha256") != checksum
            or type(calibration.get("physical_batch")) is not int or calibration["physical_batch"] < 1
            or set(calibration.get("reports", {})) != {"selected", "native"}):
        raise ValueError("The exact matched source calibration receipt is required")
    if not (source / "initial.pt").is_file() or (source / "initial.pt").is_symlink():
        raise ValueError("The saved common initial state is required")
    return contract


def _torch_publication(path):
    """Check publication structure without unpickling or copying a graph to RAM.

    The native provider still verifies its full binding/graph on actual reuse.
    SHA256 here binds exact cache bytes; ZIP directory parsing rejects a truncated
    atomic publication. This is not a new neural or geometry validation.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if (len(names) != len(set(names)) or not any(n.endswith("/data.pkl") for n in names)
                    or not any(n.endswith("/version") for n in names)):
                raise ValueError(f"Canonical torch cache publication is malformed: {path}")
    except zipfile.BadZipFile as error:
        raise ValueError(f"Canonical torch cache publication is incomplete: {path}") from error


def _metadata_publication(directory, kind):
    metadata = _read(directory / "metadata.json")
    signed = dict(metadata); checksum = signed.pop("metadata_sha256", None)
    if checksum != _digest(signed) or metadata.get("binding_sha256") != _digest(metadata.get("binding")):
        raise ValueError(f"Published {kind} cache metadata identity differs: {directory}")
    if kind == "whole_case_fields":
        if (metadata.get("format") != "v18_exact_whole_case_distance_fields_v1"
                or set(metadata.get("fields", {})) != {"depth", "occupied"}):
            raise ValueError(f"Incomplete exact whole-case cache publication: {directory}")
        records = []
        for name in ("depth", "occupied"):
            item = metadata["fields"][name]
            if item.get("file") != name + ".npy" or item.get("units") != "mm":
                raise ValueError(f"Whole-case field payload record differs: {directory}")
            records.append((name + ".npy", item.get("file_bytes"), item.get("sha256")))
    else:
        if (metadata.get("format") != "v18_exact_original_static_upper_v1"
                or metadata.get("kind") not in ("source_raw", "lesions")):
            raise ValueError(f"Incomplete exact static upper cache publication: {directory}")
        records = [("arrays.npz", metadata.get("payload_bytes"), metadata.get("payload_sha256"))]
    expected = {"metadata.json", *(name for name, _, _ in records)}
    if set(p.name for p in directory.iterdir()) != expected:
        raise ValueError(f"Unexpected/incomplete published cache inventory: {directory}")
    for name, size, checksum in records:
        payload = directory / name
        if (payload.is_symlink() or not payload.is_file() or payload.stat().st_size != size
                or not isinstance(checksum, str) or _HEX.fullmatch(checksum) is None
                or _sha(payload) != checksum):
            raise ValueError(f"Published exact cache payload SHA/size differs: {payload}")
    checksums = {name: checksum for name, _, checksum in records}
    return [(directory / name, checksums.get(name)) for name in sorted(expected)]


def _region_publication(directory):
    metadata = _read(directory / "metadata.json")
    payload = directory / "regions.npz"
    checksum = metadata.get("artifact_sha256")
    if (metadata.get("format") != "hiercp_patient_regions_v3"
            or metadata.get("integrity_format") != "sha256_v1"
            or metadata.get("storage") != "compact_crop_npz_v1"
            or set(p.name for p in directory.iterdir()) != {"metadata.json", "regions.npz"}
            or payload.is_symlink() or not payload.is_file()
            or not isinstance(checksum, str) or _HEX.fullmatch(checksum) is None
            or _sha(payload) != checksum):
        raise ValueError(f"Published original region cache identity differs: {directory}")
    return [(directory / "metadata.json", None), (payload, checksum)]


def _cache_files(data):
    if not data.exists():
        return []
    _regular_tree(data)
    files = []
    for entry in sorted(data.iterdir()):
        if entry.name.startswith("."):
            if entry.is_symlink():
                raise ValueError(f"Hidden cache symlink rejected: {entry}")
            continue
        if entry.name not in ("canonical_local", "whole_case_fields", "upper_static", "regions"):
            raise ValueError(f"Unknown v1.8 cache namespace: {entry}")
        _regular_tree(entry)
        if entry.name == "canonical_local":
            for path in sorted(entry.iterdir()):
                if path.is_symlink():
                    raise ValueError(f"Canonical cache symlink rejected: {path}")
                if path.name.startswith(".") or path.name.endswith(".tmp"):
                    continue
                if path.suffix != ".pt" or _HEX.fullmatch(path.stem) is None or not path.is_file():
                    raise ValueError(f"Unrecognized canonical cache publication: {path}")
                _torch_publication(path)
                files.append((path, None))
        else:
            def visit(directory):
                if directory.is_symlink() or not directory.is_dir():
                    raise ValueError(f"Regular published cache directory required: {directory}")
                children = list(directory.iterdir())
                if any(p.name == "metadata.json" for p in children):
                    files.extend(_region_publication(directory) if entry.name == "regions"
                                 else _metadata_publication(directory, entry.name))
                    return
                for child in sorted(children):
                    if child.is_symlink():
                        raise ValueError(f"Cache symlink rejected: {child}")
                    if child.name.startswith("."):
                        continue
                    if not child.is_dir():
                        raise ValueError(f"Unpublished exact cache payload rejected: {child}")
                    visit(child)
            visit(entry)
    return files


def _publish(source, target, record, *, immutable):
    for ancestor in (target, *target.parents):
        if ancestor.is_symlink():
            raise ValueError(f"Continuation destination cannot traverse a symlink: {ancestor}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError(f"Continuation destination symlink rejected: {target}")
    if target.exists():
        if not target.is_file() or target.stat().st_size != record["bytes"] or _sha(target) != record["sha256"]:
            raise ValueError(f"Existing continuation file differs; nothing overwritten: {target}")
        return "existing"
    if immutable:
        try:
            os.link(source, target)
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EPERM):
                raise
        else:
            return "hardlink"
    temporary = target.with_name("." + target.name + ".copy." + uuid.uuid4().hex)
    with source.open("rb") as reader, temporary.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024**2)
    if temporary.stat().st_size != record["bytes"] or _sha(temporary) != record["sha256"]:
        raise ValueError(f"Continuation byte copy differs; temporary preserved: {temporary}")
    # New namespaces are locked by the caller. Avoid replacing any publication.
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Continuation target appeared during copy: {target}; temporary preserved: {temporary}")
    temporary.rename(target)
    return "copy"


def prepare_continuation(source_root, destination_root, arm):
    """Preserve one inactive v1.8 arm, calibration and published cache bytes.

    Call while holding the new experiment's pipeline lock. Repeated calls never
    copy source checkpoints over a destination that has already progressed.
    Original and cloned caches remain immutable; newly needed cache entries are
    published only in destination/data by the original provider.
    """
    if arm not in ("selected", "native"):
        raise ValueError("One explicit selected/native arm is required")
    source_arg, destination_arg = Path(source_root), Path(destination_root)
    _regular_tree(source_arg)
    if destination_arg.is_symlink():
        raise ValueError("Continuation destination must not be a symlink")
    for parent in destination_arg.parents:
        if parent.is_symlink():
            raise ValueError(f"Continuation destination cannot traverse a symlink: {parent}")
    source = source_arg.resolve(strict=True)
    destination = destination_arg.resolve()
    _disjoint(source, destination, "the original experiment")
    contract = _sealed_contract(source)
    baseline_value = contract.get("baseline", {}).get("baseline")
    if baseline_value is not None:
        _disjoint(Path(baseline_value).resolve(), destination, "the preserved baseline")
    source_data_arg = Path(contract.get("prepared_data_root", source / "data"))
    if source_data_arg.exists():
        _regular_tree(source_data_arg)
    source_data = source_data_arg.resolve()
    _disjoint(source_data, destination, "the original preparation cache")
    for name in (".pipeline.lock", ".selected.lock", ".native.lock"):
        _inactive(source / name)
    _inactive(source_data / ".data.lock")
    if source_data.parent != source:
        _inactive(source_data.parent / ".pipeline.lock")
    request = dict(format=FORMAT, source_root=str(source), destination_root=str(destination),
                   source_data_root=str(source_data), data_root=str(destination / "data"),
                   arm=arm, contract_sha256=contract["sha256"])
    receipt_path, pending_path = destination / RECEIPT, destination / PENDING
    if receipt_path.exists():
        receipt = _read(receipt_path)
        if receipt.get("request") != request:
            raise ValueError("Existing independent continuation belongs to a different source/arm")
        signed = dict(receipt); checksum = signed.pop("receipt_sha256", None)
        if checksum != _digest(signed):
            raise ValueError("Independent continuation receipt identity differs")
        for record in receipt["files"]:
            relative = Path(record["path"])
            if (relative.is_absolute() or ".." in relative.parts or "\\" in record["path"]
                    or ":" in record["path"]
                    or record.get("role") not in ("sealed", "arm_identity", "arm_mutable", "immutable_cache")):
                raise ValueError("Independent continuation receipt path/role differs")
            path = destination / relative
            for ancestor in (path, *path.parents):
                if ancestor.is_symlink():
                    raise ValueError(f"Independent continuation path cannot traverse a symlink: {ancestor}")
            # Arm checkpoint/log files are intentionally allowed to advance.
            if record["role"] == "arm_mutable":
                continue
            if (path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"]
                    or _sha(path) != record["sha256"]):
                raise ValueError(f"Independent continuation immutable bytes changed: {path}")
        return dict(request, receipt=str(receipt_path), reused=True, resumed=receipt["resumed"],
                    file_count=len(receipt["files"]), methods=receipt["methods"])
    if destination.exists():
        _regular_tree(destination)
        if pending_path.exists():
            if _read(pending_path) != request:
                raise ValueError("Interrupted continuation belongs to a different source/arm")
        elif any(p.name not in (".pipeline.lock",) for p in destination.iterdir()):
            raise FileExistsError("Unowned independent experiment files preserved; choose a new destination")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    if not pending_path.exists():
        _new_json(pending_path, request)
    files, methods = [], {"hardlink": 0, "copy": 0, "existing": 0}
    root_names = ["experiment.json", "initial.pt", "calibration.json"]
    root_names += sorted(p.name for p in source.glob("calibration_*.json") if p.is_file())
    for name in root_names:
        origin = source / name
        record = _record(origin, Path(name), role="sealed")
        methods[_publish(origin, destination / name, record, immutable=False)] += 1
        files.append(record)
    source_arm = source / arm
    resumed = (source_arm / "checkpoint_latest.pt").is_file()
    if source_arm.exists():
        _regular_tree(source_arm)
        ordinary = [p for p in source_arm.iterdir() if not p.name.startswith(".") and p.name != "STOP_AFTER_BATCH"]
        if ordinary and not resumed:
            raise ValueError("Original arm files exist without a latest checkpoint; saved progress cannot be inferred")
        for directory, dirnames, filenames in os.walk(source_arm, followlinks=False):
            directory = Path(directory)
            for name in list(dirnames):
                path = directory / name
                if path.is_symlink():
                    raise ValueError(f"Arm directory symlink rejected: {path}")
                if name.startswith("."):
                    dirnames.remove(name)
            for name in sorted(filenames):
                origin = directory / name
                if origin.is_symlink():
                    raise ValueError(f"Arm file symlink rejected: {origin}")
                if name.startswith(".") or name.endswith(".tmp") or name.endswith(".lock") or name == "STOP_AFTER_BATCH":
                    continue
                relative = origin.relative_to(source)
                role = "arm_identity" if name == "training_identity.json" else "arm_mutable"
                record = _record(origin, relative, role=role)
                methods[_publish(origin, destination / relative, record, immutable=False)] += 1
                files.append(record)
    for origin, verified_sha in _cache_files(source_data):
        relative = Path("data") / origin.relative_to(source_data)
        record = _record(origin, relative, role="immutable_cache", verified_sha=verified_sha)
        methods[_publish(origin, destination / relative, record, immutable=True)] += 1
        files.append(record)
    # Recheck ownership and source bytes before sealing the migration receipt.
    for name in (".pipeline.lock", ".selected.lock", ".native.lock"):
        _inactive(source / name)
    _inactive(source_data / ".data.lock")
    if _read(source / "experiment.json") != contract:
        raise ValueError("Original experiment changed during continuation; copied files preserved")
    receipt = dict(request=request, files=files, methods=methods, resumed=resumed,
                   source_preserved=True, neural_contract_changed=False,
                   old_sealed_data_root_preserved=True, independent_data_namespace=True,
                   cache_validation="exact SHA/size and publication structure; original provider verifies geometry on reuse")
    receipt["receipt_sha256"] = _digest(receipt)
    _new_json(receipt_path, receipt)
    return dict(request, receipt=str(receipt_path), reused=False, resumed=resumed,
                file_count=len(files), methods=methods)
