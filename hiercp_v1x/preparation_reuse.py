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
import socket
import stat
import sys
import threading
import time
import uuid
import weakref

import numpy as np

from . import u_bridge_continuation as publication
from . import u_bridge_fields as fields
from . import u_bridge_upper as upper

FORMAT = "v18_v19_completed_preparation_reuse_v1"
_HEX = re.compile(r"[0-9a-f]{64}")
_REGISTRY = threading.RLock()
_LOCKS = {}
_CONTEXT = threading.RLock()
_GUARDS = threading.local()
_UNSUPPORTED_RENAME = {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}


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


def _namespace_owner(target):
    """Capture the existing task's exclusive data-namespace ownership receipt."""
    for parent in Path(target).parents:
        path = parent / ".data.lock"
        if not os.path.lexists(path):
            continue
        _safe(path)
        raw = path.read_bytes()
        owner = json.loads(raw)
        if (owner.get("host") != socket.gethostname() or owner.get("pid") != os.getpid()
                or type(owner.get("pid")) is not int
                or not isinstance(owner.get("token"), str) or not owner["token"]):
            raise RuntimeError(f"Directory publication requires this process's own data namespace lock: {path}")
        return path, raw
    return None


def _verify_namespace_owner(receipt, target):
    if receipt is None:
        raise RuntimeError(f"Unsupported no-replace directory rename requires an owned data namespace lock: {target}")
    path, raw = receipt
    _safe(path)
    if not path.is_file() or path.read_bytes() != raw:
        raise RuntimeError(f"Data namespace ownership changed before directory publication: {path}")
    # Recheck identity as well as bytes: forked processes cannot inherit proof.
    if _namespace_owner(target) != receipt:
        raise RuntimeError(f"Data namespace ownership differs before directory publication: {path}")


@contextmanager
def _publication_guard(target):
    """Serialize cooperating publishers; lock files persist to avoid inode ABA.

    Linux POSIX record locks coordinate distinct processes and NFS lock managers.
    The thread RLock is also required because record locks belong to a process.
    Nested calls reuse the same descriptor; closing another descriptor for the
    same inode would otherwise release an outer process's record lock.
    """
    target = Path(target)
    with _lock(target):
        # Published immutable entries need only their original loader checks.
        # Avoid a network lock/stat/owner roundtrip on every hot-cache query.
        if os.path.lexists(target):
            yield None
            return
        if not sys.platform.startswith("linux"):
            yield None
            return
        active = getattr(_GUARDS, "active", None)
        if active is None:
            active = _GUARDS.active = {}
        identity = _path_identity(target)
        if identity in active:
            yield active[identity]
            return
        import fcntl
        target.parent.mkdir(parents=True, exist_ok=True)
        path = target.with_name("." + target.name + ".reuse.lock")
        _safe(path)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        locked = False
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError(f"Publication guard must be a regular lock file: {path}")
            fcntl.lockf(descriptor, fcntl.LOCK_EX)
            locked = True
            _safe(path)
            current = path.stat()
            if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
                raise RuntimeError(f"Publication guard inode changed: {path}")
            state = dict(descriptor=descriptor, owner=_namespace_owner(target))
            active[identity] = state
            try:
                yield state
            finally:
                del active[identity]
        finally:
            try:
                if locked:
                    fcntl.lockf(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _guarded_posix_publication(source, target):
    """Preserve atomic visibility on filesystems lacking RENAME_NOREPLACE.

    Directory rename is safe within a task-exclusive namespace and among writers
    using this guard. POSIX has no no-replace directory primitive against an
    unrelated writer ignoring ownership/locks; no such guarantee is claimed.
    """
    source, target = Path(source), Path(target)
    _safe(source, directory=source.is_dir())
    if not source.is_dir():
        os.link(source, target)  # Atomic exclusive link, including empty-target races.
        source.unlink()  # Only the successful publisher's private staging name.
        return "exclusive_link"
    with _publication_guard(target) as state:
        if os.path.lexists(target):
            raise FileExistsError(errno.EEXIST, "Existing publication preserved", str(target))
        _verify_namespace_owner(state["owner"], target)
        os.rename(source, target)  # Complete staging remains hidden until this rename.
        return "owned_namespace_posix_rename"


def _linux_noreplace(source, target):
    """Use the real no-replace syscall, preserving errno for filesystem fallback."""
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
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


def _rename_new(source, target):
    """Publish complete bytes; native no-replace first, guarded POSIX if unsupported."""
    if sys.platform == "win32":
        os.rename(source, target)  # Windows rename refuses an existing destination.
        return "native_windows_noreplace"
    if sys.platform.startswith("linux"):
        try:
            _linux_noreplace(source, target)
        except OSError as error:
            if error.errno not in _UNSUPPORTED_RENAME:
                raise
            return _guarded_posix_publication(source, target)
        return "renameat2_noreplace"
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
        publish_method = _rename_new(attempt, destination)
        _emit(callback, kind="whole_case_fields", status="reused", case_id=case_id,
              source=str(source), destination=str(destination), methods=methods,
              publication_method=publish_method,
              bytes=sum(payload.stat().st_size for payload, _ in files),
              binding_sha256=metadata["binding_sha256"],
              wall_seconds=time.perf_counter() - started)
        return
    _emit(callback, kind="whole_case_fields", status="cache_miss", case_id=case_id,
          destination=str(destination), wall_seconds=time.perf_counter() - started)


@contextmanager
def _shared_fields_guard(target_root, case_id, binding, source_roots):
    """One cold field build per case/binding across the configured arm roots.

    Arm-owned paths and signed bytes remain unchanged. The small coordination
    locks live beside, not inside, experiment data roots. Every participating
    destination must be in the same source-root set so a waiter can discover the
    first completed publication. Ordinary one-way imports retain their existing
    behavior. Different cases/bindings never share a lock.
    """
    data_root = _safe(Path(target_root).parent, directory=True)
    roots = tuple(dict.fromkeys(source_roots))
    destination = Path(target_root) / case_id
    if destination.exists() or data_root not in roots or len(roots) < 2:
        yield
        return
    try:
        shared_parent = Path(os.path.commonpath([str(path) for path in roots]))
    except ValueError:
        # Cross-volume imports cannot hardlink and do not form a shared store.
        yield
        return
    if shared_parent == Path(shared_parent.anchor) or shared_parent in roots:
        # Do not create coordination files in a filesystem root or inside a
        # source's signed data inventory. These are ordinary one-way imports.
        yield
        return
    key = publication._digest(dict(case_id=case_id, binding=binding))
    target = _safe(shared_parent / ".field_publication" / key)
    # This target itself is never created, only its persistent lock file. Thus
    # _publication_guard cannot take its existing-publication fast path here.
    with _publication_guard(target):
        yield


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
        publish_method = _rename_new(attempt, destination)
        _emit(callback, kind="upper_static", helper_kind=kind, status="reused",
              source=str(source), destination=str(destination), methods=methods,
              publication_method=publish_method,
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
    This wrapper never reads source checkpoints or mutates source payloads.
    Cold fields across participating arms share a per-case coordination lock
    beside their data namespaces; existing complete fields stay untouched.
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
        # Keep only identities returned by the original verified mapping loader.
        # A remembered identity is useful while its exact registry entry lives;
        # eviction/removal must take the full publication/loader path again.
        verified_fields = {}

        def cached_fields(root, case_id, binding, depth_factory, occupied_factory, budget_check):
            if (not isinstance(case_id, str) or not case_id
                    or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id) is None):
                raise ValueError("Preparation reuse requires a single-directory case identity")
            canonical = fields._canonical_binding(binding)
            lock_identity = _path_identity(Path(root) / case_id)
            with fields._REGISTRY_LOCK:
                lock = fields._LOCKS.setdefault(lock_identity, threading.RLock())
            with lock:
                remembered = verified_fields.get((lock_identity, fields._digest(canonical)))
                if remembered is not None:
                    identity, mappings = remembered
                    with fields._REGISTRY_LOCK:
                        entry = fields._OPENED.get(identity)
                        still_open = (entry is not None and len(entry[0]) == len(mappings)
                                      and all(array is reference() for array, reference in zip(entry[0], mappings)))
                    if still_open:
                        # Preserve the original budget, readonly mapping and
                        # receipt behavior; omit only our added disk probes.
                        return original_fields(root, case_id, binding, depth_factory, occupied_factory, budget_check)
                directory = _safe(Path(root) / case_id, directory=True)
                with _publication_guard(directory):
                    with _shared_fields_guard(root, case_id, canonical, roots):
                        # Recheck after the shared cold-build lock: a different
                        # arm may now have published the exact fields. Import
                        # its bytes before considering the genuine factories.
                        _ensure_fields(root, case_id, canonical, roots, event_callback)
                        result = original_fields(root, case_id, binding, depth_factory, occupied_factory, budget_check)
                identity = (str(directory), fields._digest(canonical))
                with fields._REGISTRY_LOCK:
                    entry = fields._OPENED.get(identity)
                if entry is not None:
                    # Do not extend mapping lifetime if the original registry
                    # is released by a future pressure policy or caller.
                    verified_fields[(lock_identity, identity[1])] = (identity, tuple(weakref.ref(a) for a in entry[0]))
                return result

        class ReusingPreparation(provider_class):
            def _get(self, key, factory):
                if isinstance(key, tuple) and len(key) == 2 and key[0] == "local":
                    if not isinstance(key[1], str) or _HEX.fullmatch(key[1]) is None:
                        raise ValueError("Canonical reuse requires the original SHA256 binding key")
                    def verified_factory():
                        # The original _get calls its factory only on a RAM
                        # miss. Import still precedes its disk load/build, and
                        # the guard spans both exact reuse and that loader.
                        with _publication_guard(Path(self.graph_dir) / (key[1] + ".pt")):
                            _ensure_local(self.graph_dir, key[1], roots, event_callback)
                            return factory()
                    return super()._get(key, verified_factory)
                return super()._get(key, factory)

        def obtain_static(cache, kind, binding, factory):
            if kind not in ("source_raw", "lesions"):
                raise ValueError("Preparation reuse requires the original static upper helper kind")
            canonical = upper._binding(binding)
            with cache._lock:
                if (kind, upper._digest(canonical)) in cache._resident:
                    # The original method retains its check(), hit accounting
                    # and defensive array copies under the same reentrant lock.
                    return original_upper(cache, kind, binding, factory)
            anchor = "_".join(map(str, canonical["anchor"]))
            directory = Path(cache.root) / canonical["case_id"] / f"c{canonical['source_component']}_a{anchor}" / kind
            with _publication_guard(directory):
                _ensure_upper(cache.root, kind, binding, roots, event_callback)
                return original_upper(cache, kind, binding, factory)

        fields.cached_fields = cached_fields
        upper.UpperCache._obtain = obtain_static
        try:
            yield ReusingPreparation
        finally:
            fields.cached_fields = original_fields
            upper.UpperCache._obtain = original_upper
