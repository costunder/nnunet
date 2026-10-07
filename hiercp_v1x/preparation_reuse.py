"""Reuse exact completed preprocessing publications across independent arms.

Only immutable cache files are shared. Each arm owns its publication namespace,
locks, checkpoints and logs; source experiments may still be running. Original
provider loaders retain their binding and numerical checks after byte reuse.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import errno
import json
import math
import os
import platform
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import uuid

import numpy as np

from . import u_bridge_continuation as publication
from . import u_bridge_fields as fields
from . import u_bridge_upper as upper

FORMAT = "v18_v19_completed_preparation_reuse_v1"
_HEX = re.compile(r"[0-9a-f]{64}")
_REGISTRY = threading.RLock()
_LOCKS = {}
_CONTEXT = threading.RLock()


def _safe(path, *, directory=False):
    path = Path(path)
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError(f"Preparation reuse cannot traverse a symlink: {ancestor}")
    if path.exists() and ((directory and not path.is_dir()) or (not directory and not path.is_file())):
        raise ValueError(f"Unexpected preparation publication type: {path}")
    return path.resolve()


def _lock(path):
    # On Windows resolve() may adopt on-disk letter case once publication
    # appears. Derive one lexical case-normalized key before filesystem races.
    identity = _path_identity(path)
    with _REGISTRY:
        return _LOCKS.setdefault(identity, threading.RLock())


def _path_identity(path):
    value = os.path.abspath(path)
    # Path.resolve() can return an extended path if a Windows directory is
    # created while resolving it. Both spellings identify the same namespace.
    if sys.platform == "win32":
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
    return os.path.normcase(os.path.normpath(value))


def _emit(callback, **record):
    if callback is not None:
        callback(dict(format=FORMAT, **record))


def _rename_new(source, target):
    """Atomically publish without replacing an artifact created by another writer."""
    if sys.platform == "win32":
        os.rename(source, target)  # Windows rename refuses an existing destination.
        return
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameat2", None)
        if rename is None:
            # Older Singularity glibc may lack the symbol while the host kernel
            # supports the identical renameat2 syscall. Use only known ABI IDs.
            architecture = platform.machine().lower()
            number = {"x86_64": 316, "amd64": 316, "aarch64": 276, "arm64": 276}.get(architecture)
            syscall = getattr(library, "syscall", None)
            if number is None or syscall is None:
                raise OSError(errno.ENOSYS, f"Atomic no-replace renameat2 ABI unavailable: {architecture}")
            syscall.restype = ctypes.c_long
            result = syscall(ctypes.c_long(number), ctypes.c_int(-100), ctypes.c_char_p(os.fsencode(source)),
                             ctypes.c_int(-100), ctypes.c_char_p(os.fsencode(target)), ctypes.c_uint(1))
        else:
            rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            result = rename(-100, os.fsencode(source), -100, os.fsencode(target), 1)
        if result:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(target))
        return
    raise OSError(errno.ENOSYS, "Atomic no-replace cache publication is unavailable on this platform")


def _file(source, target, *, checksum):
    """Install already validated immutable bytes, with copy only across filesystems."""
    _safe(source); _safe(target)
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EPERM):
            raise
    else:
        return "hardlink"
    attempt = target.with_name("." + target.name + ".reuse." + uuid.uuid4().hex)
    with source.open("rb") as reader, attempt.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024**2)
        writer.flush()
    if publication._sha(attempt) != checksum:
        raise ValueError(f"Preparation byte copy differs; attempt preserved: {attempt}")
    _rename_new(attempt, target)
    return "copy"


def _fields_metadata(directory):
    files = publication._metadata_publication(directory, "whole_case_fields")
    metadata = publication._read(directory / "metadata.json")
    binding = fields._canonical_binding(metadata["binding"])
    for name in ("depth", "occupied"):
        record = metadata["fields"][name]
        with (directory / record["file"]).open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported exact field NumPy header version: {version}")
        if (list(shape) != binding["shape"] or list(shape) != record.get("shape")
                or dtype.kind != "f" or dtype.hasobject or dtype.str != record.get("dtype")
                or math.prod(shape) * dtype.itemsize != record.get("array_bytes")
                or fortran != record.get("fortran_order")):
            raise ValueError(f"Preparation field payload header differs: {directory / record['file']}")
    return metadata, files


def _ensure_fields(target_root, case_id, binding, source_roots, callback):
    root = _safe(target_root, directory=True)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / case_id
    _safe(destination, directory=True)
    if destination.exists():
        return  # The original loader checks the existing arm-owned publication.
    started = time.perf_counter()
    for source_root in source_roots:
        if source_root == root.parent:
            continue
        source = source_root / "whole_case_fields" / case_id
        _safe(source, directory=True)
        if not source.exists():
            continue
        # A completed signed publication is required. Dot-attempt directories
        # are never enumerated, nor is another arm's lock read or modified.
        metadata = publication._read(source / "metadata.json")
        signed = dict(metadata); saved_sha = signed.pop("metadata_sha256", None)
        if (metadata.get("format") != fields.FORMAT or saved_sha != publication._digest(signed)
                or metadata.get("binding_sha256") != publication._digest(metadata.get("binding"))):
            raise ValueError(f"Published field metadata was altered: {source}")
        if metadata.get("binding") != binding:
            _emit(callback, kind="whole_case_fields", status="binding_miss",
                  case_id=case_id, source=str(source), destination=str(destination))
            continue
        metadata, files = _fields_metadata(source)
        attempt = root / ("." + case_id + ".reuse." + uuid.uuid4().hex)
        attempt.mkdir(exist_ok=False)
        methods = []
        for payload, saved_sha in files:
            methods.append(_file(payload, attempt / payload.name,
                                 checksum=saved_sha or publication._sha(payload)))
        _rename_new(attempt, destination)
        _emit(callback, kind="whole_case_fields", status="reused", case_id=case_id,
              source=str(source), destination=str(destination), methods=methods,
              bytes=sum(payload.stat().st_size for payload, _ in files),
              binding_sha256=metadata["binding_sha256"],
              wall_seconds=time.perf_counter() - started)
        return
    _emit(callback, kind="whole_case_fields", status="cache_miss", case_id=case_id,
          destination=str(destination), wall_seconds=time.perf_counter() - started)


def _ensure_local(graph_dir, key, source_roots, callback):
    if not isinstance(key, str) or _HEX.fullmatch(key) is None:
        raise ValueError("Canonical reuse requires the original SHA256 binding key")
    root = _safe(graph_dir, directory=True)
    root.mkdir(parents=True, exist_ok=True)
    target = root / (key + ".pt")
    _safe(target)
    if target.exists():
        return  # Never replace an existing graph; original loader verifies it.
    started = time.perf_counter()
    for source_root in source_roots:
        if source_root == root.parent:
            continue
        source = source_root / "canonical_local" / target.name
        _safe(source)
        if not source.exists():
            continue
        before = source.stat()
        publication._torch_publication(source)
        checksum = publication._sha(source)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError(f"Completed canonical source changed during verification: {source}")
        method = _file(source, target, checksum=checksum)
        _emit(callback, kind="canonical_local", status="reused", key=key,
              source=str(source), destination=str(target), method=method,
              artifact_sha256=checksum, bytes=after.st_size,
              wall_seconds=time.perf_counter() - started)
        return
    _emit(callback, kind="canonical_local", status="cache_miss", key=key,
          destination=str(target), wall_seconds=time.perf_counter() - started)


def _ensure_upper(target_root, kind, binding, source_roots, callback):
    if kind not in ("source_raw", "lesions"):
        raise ValueError("Preparation reuse requires the original static upper helper kind")
    canonical = upper._binding(binding)
    root = _safe(target_root, directory=True)
    anchor = "_".join(map(str, canonical["anchor"]))
    relative = Path(canonical["case_id"]) / f"c{canonical['source_component']}_a{anchor}" / kind
    destination = root / relative
    _safe(destination, directory=True)
    if destination.exists():
        return  # The original _obtain checks every exact numeric array/header.
    started = time.perf_counter()
    for source_root in source_roots:
        if source_root == root.parent:
            continue
        source = source_root / "upper_static" / relative
        _safe(source, directory=True)
        if not source.exists():
            continue
        metadata = publication._read(source / "metadata.json")
        signed = dict(metadata); saved_sha = signed.pop("metadata_sha256", None)
        if (metadata.get("format") != upper.FORMAT or metadata.get("kind") != kind
                or saved_sha != publication._digest(signed)
                or metadata.get("binding_sha256") != publication._digest(metadata.get("binding"))):
            raise ValueError(f"Published static upper metadata was altered: {source}")
        if metadata["binding"] != canonical:
            _emit(callback, kind="upper_static", helper_kind=kind, status="binding_miss",
                  source=str(source), destination=str(destination))
            continue
        files = publication._metadata_publication(source, "upper_static")
        destination.parent.mkdir(parents=True, exist_ok=True)
        attempt = destination.with_name("." + kind + ".reuse." + uuid.uuid4().hex)
        attempt.mkdir(exist_ok=False)
        methods = [_file(payload, attempt / payload.name,
                         checksum=saved_sha or publication._sha(payload))
                   for payload, saved_sha in files]
        _rename_new(attempt, destination)
        _emit(callback, kind="upper_static", helper_kind=kind, status="reused",
              source=str(source), destination=str(destination), methods=methods,
              bytes=sum(payload.stat().st_size for payload, _ in files),
              binding_sha256=metadata["binding_sha256"], payload_sha256=metadata["payload_sha256"],
              wall_seconds=time.perf_counter() - started)
        return
    _emit(callback, kind="upper_static", helper_kind=kind, status="cache_miss",
          destination=str(destination), wall_seconds=time.perf_counter() - started)


@contextmanager
def preparation_reuse(provider_class, source_data_roots, event_callback=None):
    """Yield an additive provider subclass while exact field reuse is active.

    Callers retain their own experiment/data namespace locks. Source data roots
    can belong to active arms: completed publications are immutable and atomic.
    This wrapper never reads source checkpoints or mutates source namespaces.
    The original graph load verifies binding after file reuse, and the original
    field load verifies all array values. No data factory is replaced by dummy
    values; a genuine cache miss follows the original construction path.
    """
    if event_callback is not None and not callable(event_callback):
        raise TypeError("Preparation reuse event callback must be callable")
    roots = tuple(dict.fromkeys(_safe(root, directory=True) for root in source_data_roots))
    with _CONTEXT:
        original_fields = fields.cached_fields
        original_upper = upper.UpperCache._obtain

        def cached_fields(root, case_id, binding, depth_factory, occupied_factory, budget_check):
            if (not isinstance(case_id, str) or not case_id
                    or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id) is None):
                raise ValueError("Preparation reuse requires a single-directory case identity")
            canonical = fields._canonical_binding(binding)
            directory = _safe(Path(root) / case_id, directory=True)
            lock_identity = _path_identity(Path(root) / case_id)
            with fields._REGISTRY_LOCK:
                lock = fields._LOCKS.setdefault(lock_identity, threading.RLock())
            with lock:
                _ensure_fields(root, case_id, canonical, roots, event_callback)
                return original_fields(root, case_id, binding, depth_factory, occupied_factory, budget_check)

        class ReusingPreparation(provider_class):
            def _get(self, key, factory):
                if isinstance(key, tuple) and len(key) == 2 and key[0] == "local":
                    with _lock(Path(self.graph_dir) / (str(key[1]) + ".pt")):
                        _ensure_local(self.graph_dir, key[1], roots, event_callback)
                        return super()._get(key, factory)
                return super()._get(key, factory)

        def obtain_static(cache, kind, binding, factory):
            canonical = upper._binding(binding)
            anchor = "_".join(map(str, canonical["anchor"]))
            directory = Path(cache.root) / canonical["case_id"] / f"c{canonical['source_component']}_a{anchor}" / kind
            with _lock(directory):
                _ensure_upper(cache.root, kind, binding, roots, event_callback)
                return original_upper(cache, kind, binding, factory)

        fields.cached_fields = cached_fields
        upper.UpperCache._obtain = obtain_static
        try:
            yield ReusingPreparation
        finally:
            fields.cached_fields = original_fields
            upper.UpperCache._obtain = original_upper
