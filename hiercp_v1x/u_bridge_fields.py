"""Exact, immutable whole-case distance fields for the V1 U bridge.

This cache changes storage and repeat computation only. Factory arrays keep
their native shape, floating dtype, values, and millimetre units unchanged.
Interrupted attempts remain available for inspection and are never reused.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import mmap
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import uuid

import numpy as np

FORMAT = "v18_exact_whole_case_distance_fields_v1"
_OPENED = {}
_PAGE_HINTS = {}
_LOCKS = {}
_REGISTRY_LOCK = threading.RLock()
_HEADER_RESERVE = 1024 ** 2


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            value.update(block)
    return value.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _canonical_binding(binding):
    if not isinstance(binding, dict):
        raise TypeError("Signed whole-case field binding must be a mapping")
    value = json.loads(json.dumps(binding, allow_nan=False))
    for key in ("image_sha256", "label_sha256", "common_sha256"):
        if not isinstance(value.get(key), str) or re.fullmatch(r"[0-9a-fA-F]{64}", value[key]) is None:
            raise ValueError(f"Whole-case fields require original {key}")
    shape, spacing = value.get("shape"), value.get("spacing")
    if (not isinstance(shape, list) or len(shape) != 3
            or any(type(n) is not int or n <= 0 for n in shape)):
        raise ValueError("Whole-case fields require the exact positive integer CT shape")
    if (not isinstance(spacing, list) or len(spacing) != 3
            or any(type(n) not in (int, float) or not math.isfinite(n) or n <= 0 for n in spacing)):
        raise ValueError("Whole-case fields require three positive finite native spacings")
    return value


def _check_array(array, shape, name):
    if (not isinstance(array, np.ndarray) or array.shape != tuple(shape)
            or array.dtype.kind != "f" or array.dtype.hasobject):
        raise ValueError(f"{name}: exact whole-case floating array shape/dtype required")
    # Bound validation workspace, rather than copying a whole-CT boolean mask.
    for block in np.nditer(array, flags=["external_loop", "buffered"],
                           op_flags=["readonly"], buffersize=262144):
        if not np.isfinite(block).all() or (block < 0).any():
            raise ValueError(f"{name}: nonfinite or negative whole-case distance field")


def _room(root, required):
    free = shutil.disk_usage(root).free
    if free < required:
        raise OSError(f"Whole-case field cache needs {required} additional bytes; "
                      f"free={free}; no field cropped, quantized, or replaced")
    return free


def _release_readonly_pages(array):
    """Request Linux page eviction after validation; array bytes stay on disk.

    This is an OS residency hint, not an input or graph size limit. A failed
    hint remains visible and the caller's actual RSS budget still applies.
    """
    result = dict(supported=False, applied=False, advised_bytes=0,
                  reason="unsupported_platform", method="madvise(MADV_DONTNEED)")
    if not sys.platform.startswith("linux"):
        return result
    advise = getattr(array._mmap, "madvise", None)
    option = getattr(mmap, "MADV_DONTNEED", None)
    if not callable(advise) or option is None:
        result["reason"] = "madvise_api_unavailable"
        return result
    if array.flags.writeable:
        raise ValueError("Page-release hints require a read-only distance-field mapping")
    result["supported"] = True
    try:
        advise(option)
    except (OSError, ValueError) as error:
        result["reason"] = f"OS hint rejected: {error}"
    else:
        result.update(applied=True, advised_bytes=int(array.nbytes), reason="OS_hint_applied")
    return result


def _load(directory, binding, budget_check):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Whole-case field cache is not a real directory: {directory}")
    metadata_path = directory / "metadata.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError(f"Incomplete whole-case field publication: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf8"))
    signed = dict(metadata)
    saved_digest = signed.pop("metadata_sha256", None)
    if (metadata.get("format") != FORMAT or saved_digest != _digest(signed)
            or metadata.get("binding") != binding
            or metadata.get("binding_sha256") != _digest(binding)):
        raise ValueError(f"Whole-case field cache binding/metadata identity differs: {directory}")
    if set(metadata.get("fields", {})) != {"depth", "occupied"}:
        raise ValueError(f"Incomplete whole-case field publication: {directory}")
    arrays = []
    hints = []
    try:
        for name in ("depth", "occupied"):
            record = metadata["fields"][name]
            path = directory / (name + ".npy")
            if (record.get("file") != path.name or record.get("units") != "mm"
                    or path.is_symlink() or not path.is_file()):
                raise ValueError(f"Incomplete whole-case field publication: {path}")
            if path.stat().st_size != record.get("file_bytes") or _sha(path) != record.get("sha256"):
                raise ValueError(f"Whole-case field cache SHA256 differs: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            arrays.append(array)
            if (list(array.shape) != binding["shape"] or list(array.shape) != record.get("shape")
                    or array.dtype.str != record.get("dtype") or array.nbytes != record.get("array_bytes")
                    or bool(array.flags.f_contiguous and not array.flags.c_contiguous) != record.get("fortran_order")
                    or array.flags.writeable):
                raise ValueError(f"Whole-case field cache NumPy header differs: {path}")
            _check_array(array, binding["shape"], name)
            hints.append(_release_readonly_pages(array))
            budget_check()
    except Exception:
        # Release only mappings opened here; preserve every cache artifact.
        for array in arrays:
            array._mmap.close()
        raise
    _PAGE_HINTS[str(directory)] = dict(supported=all(h["supported"] for h in hints),
        applied=all(h["applied"] for h in hints), advised_bytes=sum(h["advised_bytes"] for h in hints),
        fields=dict(zip(("depth", "occupied"), hints)))
    return tuple(arrays), metadata


def _receipt(directory, metadata, status, free_before=None, started=None):
    cold_hint = copy.deepcopy(_PAGE_HINTS[str(directory)])
    current_hint = copy.deepcopy(cold_hint)
    if status == "resident_mapping":
        current_hint.update(applied=False, advised_bytes=0)
        for field_hint in current_hint["fields"].values():
            field_hint.update(applied=False, advised_bytes=0, reason="existing_mapping_reused_no_new_hint")
    return dict(format=FORMAT, case_directory=str(directory), status=status,
                binding_sha256=metadata["binding_sha256"], metadata_sha256=metadata["metadata_sha256"],
                fields=copy.deepcopy(metadata["fields"]),
                array_bytes=sum(r["array_bytes"] for r in metadata["fields"].values()),
                disk_bytes=sum(r["file_bytes"] for r in metadata["fields"].values())
                           + (directory / "metadata.json").stat().st_size,
                disk_free_bytes=shutil.disk_usage(directory).free,
                disk_free_bytes_before_build=free_before, readonly=True,
                whole_case=True, recomputed=status == "built",
                page_release_hint=current_hint, last_cold_load_page_release_hint=cold_hint,
                wall_seconds=time.perf_counter() - started if started is not None else None)


def cached_fields(root, case_id, binding, depth_factory, occupied_factory, budget_check):
    """Return (depth read-only memmap, occupied read-only memmap, receipt).

    Binding must contain image_sha256, label_sha256, common_sha256 (the exact
    original common.py), whole-case shape and native spacing. Extra identity
    fields remain signed. The two factories are called once on first creation.
    Existing complete caches are never overwritten or silently rebuilt.
    """
    started = time.perf_counter()
    if (not isinstance(case_id, str) or not case_id or case_id in (".", "..")
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id) is None):
        raise ValueError("A human-readable single-directory case identity is required")
    if not all(callable(x) for x in (depth_factory, occupied_factory, budget_check)):
        raise TypeError("Two actual field factories and a resource budget callback are required")
    binding = _canonical_binding(binding)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = root / case_id
    identity = (str(directory), _digest(binding))
    with _REGISTRY_LOCK:
        lock = _LOCKS.setdefault(str(directory), threading.RLock())
    with lock:
        budget_check()
        if identity in _OPENED:
            arrays, metadata = _OPENED[identity]
            return *arrays, _receipt(directory, metadata, "resident_mapping", started=started)
        free_before = None
        if directory.exists() or directory.is_symlink():
            arrays, metadata = _load(directory, binding, budget_check)
            status = "reopened"
        else:
            # The original distance operators return float32. This is a space
            # preflight estimate only; exact factory dtype and bytes are checked
            # again before each write, and never converted to that estimate.
            estimate = math.prod(binding["shape"]) * 2 * np.dtype(np.float32).itemsize
            free_before = _room(root, estimate + _HEADER_RESERVE)
            attempt = root / ("." + case_id + ".attempt." + uuid.uuid4().hex)
            attempt.mkdir(exist_ok=False)
            records = {}
            for name, factory in (("depth", depth_factory), ("occupied", occupied_factory)):
                budget_check()
                array = factory()
                _check_array(array, binding["shape"], name)
                budget_check()
                _room(root, array.nbytes + _HEADER_RESERVE)
                path = attempt / (name + ".npy")
                with path.open("xb") as stream:
                    np.save(stream, array, allow_pickle=False)
                    stream.flush()
                records[name] = dict(file=path.name, shape=list(array.shape), dtype=array.dtype.str,
                    array_bytes=int(array.nbytes), file_bytes=path.stat().st_size, sha256=_sha(path),
                    fortran_order=bool(array.flags.f_contiguous and not array.flags.c_contiguous), units="mm")
                del array
                budget_check()
            metadata = dict(format=FORMAT, binding=binding, binding_sha256=_digest(binding), fields=records)
            metadata["metadata_sha256"] = _digest(metadata)
            with (attempt / "metadata.json").open("x", encoding="utf8") as stream:
                json.dump(metadata, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.flush()
            if directory.exists() or directory.is_symlink():
                raise FileExistsError(f"Concurrent complete field publication; owned attempt preserved: {attempt}")
            attempt.rename(directory)
            arrays, metadata = _load(directory, binding, budget_check)
            status = "built"
        _OPENED[identity] = (arrays, metadata)
        return *arrays, _receipt(directory, metadata, status, free_before, started)
