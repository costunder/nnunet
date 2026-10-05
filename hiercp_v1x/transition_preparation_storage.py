"""Lossless D preparation storage with checked immutable-donor reuse.

Only the shared donor serialization is memoized. Each observation payload is
still independently serialized, compressed, stored and SHA-bound in the exact
existing GraphWriter format. This module does not reduce graph/data coverage.

The provider owns immutable CPU donor tensors. Reuse requires the same live
tensor objects, storage/version/layout and recursively identical metadata.
Replacements and mutations take the original serialize-and-compare path.
NumPy arrays are content-hashed on EVERY check, never trusted by object ID.
Unsupported objects/layouts disable reuse rather than bypass integrity checks.
The bounded memo holds weak references/signatures, not donor tensor storage.
"""
from __future__ import annotations

from collections import OrderedDict
import gzip
import hashlib
from pathlib import Path
import shutil
import struct
import weakref

import numpy as np
import torch

from hiercp_v22 import storage as original


def _source_signature(value):
    """Return a reusable signature plus live weak tensor refs, or None.

    Dict/list order and scalar types are preserved because the existing byte
    serialization contract distinguishes those structures. Tensor byte access
    through .data or a NumPy alias must not occur during provider-owned immutable
    preparation; PyTorch version counters cannot observe such external writes.
    """
    references = []

    def visit(item):
        if torch.is_tensor(item):
            if type(item) is not torch.Tensor or item.device.type != "cpu" or item.layout != torch.strided:
                raise TypeError("Nonstandard tensor disables serialization memo")
            try:
                version = item._version
            except RuntimeError as error:
                raise TypeError("Unversioned inference tensor disables serialization memo") from error
            storage = item.untyped_storage()
            references.append((weakref.ref(item), id(item)))
            return ("tensor", id(item), storage._cdata, storage.data_ptr(), storage.nbytes(), version,
                    tuple(item.shape), tuple(item.stride()), item.storage_offset(), str(item.dtype),
                    str(item.device), item.requires_grad, item.is_conj(), item.is_neg())
        if isinstance(item, np.ndarray):
            # An object array may hide mutable objects/pickle metadata. It is
            # fully serialized every time instead of receiving a byte memo.
            if item.dtype.hasobject:
                raise TypeError("Object NumPy array disables serialization memo")
            digest = hashlib.sha256(item.tobytes(order="A")).hexdigest()
            return ("numpy", id(item), item.dtype.str, repr(item.dtype.descr), tuple(item.shape), tuple(item.strides),
                    item.flags.c_contiguous, item.flags.f_contiguous, digest)
        if type(item) is dict:
            return ("dict", tuple((visit(key), visit(nested)) for key, nested in item.items()))
        if type(item) is list:
            return ("list", tuple(visit(nested) for nested in item))
        if type(item) is tuple:
            return ("tuple", tuple(visit(nested) for nested in item))
        if item is None or type(item) in (str, bytes, int, bool):
            return (type(item).__name__, item)
        if type(item) is float:
            # Preserve -0.0 and even different NaN payloads in pickle metadata.
            return ("float", struct.pack("!d", item))
        if isinstance(item, np.generic) and not item.dtype.hasobject:
            return ("numpy_scalar", item.dtype.str, item.tobytes())
        raise TypeError("Unsupported donor metadata disables serialization memo")

    try:
        return visit(value), tuple(references)
    except TypeError:
        return None


def _same_signature(saved, current):
    if saved is None or current is None or saved[0] != current[0]:
        return False
    return (len(saved[1]) == len(current[1])
            and all(old_id == new_id and old() is new() and old() is not None
                    for (old, old_id), (new, new_id) in zip(saved[1], current[1])))


class GraphWriter(original.GraphWriter):
    """Existing lossless writer format, with bounded version-checked memo.

    ``source_memo_entries`` is only a CPU bookkeeping budget. An eviction means
    serialization is repeated; no graph, observation or donor is omitted.
    Shared-source publication is locked. Independent observation payload writes
    can safely run concurrently when the caller supplies distinct file paths.
    """
    def __init__(self, root, minimum_free_bytes=80 * 1024**3, *, source_memo_entries=128):
        super().__init__(root, minimum_free_bytes=minimum_free_bytes)
        if type(source_memo_entries) is not int or source_memo_entries < 1:
            raise ValueError("Positive serialization memo bookkeeping budget required")
        self.source_memo_entries = source_memo_entries
        self._source_memos = OrderedDict()
        self.source_encode_calls = 0
        self.source_memo_hits = 0

    def _reference(self, source, source_key):
        with self.lock:
            current = _source_signature(source)
            saved = self._source_memos.get(source_key)
            if source_key in self.sources and _same_signature(saved, current):
                self._source_memos.move_to_end(source_key)
                self.source_memo_hits += 1
                return self.sources[source_key]
            self.source_encode_calls += 1
            raw = original.encode(source)
            # A version/metadata change DURING serialization must not publish a
            # source bound to an earlier signature. Unsupported values always
            # use the original byte digest and remain outside the memo.
            after = _source_signature(source)
            if current is not None and not _same_signature(current, after):
                raise ValueError("Source tensors/metadata changed during serialization")
            digest = hashlib.sha256(raw).hexdigest()
            if source_key not in self.sources:
                source_path = f"shared_sources/{digest}.pt.gz"
                file = self.root / source_path
                file.parent.mkdir(parents=True, exist_ok=True)
                if not file.exists():
                    with file.open("xb") as stream:
                        stream.write(gzip.compress(raw, compresslevel=1, mtime=0))
                self.sources[source_key] = dict(path=source_path, sha256=original.sha(file),
                                                content_sha256=digest)
            reference = self.sources[source_key]
            if reference["content_sha256"] != digest:
                raise ValueError("Source tensors changed within a declared shared-source group")
            if after is not None:
                self._source_memos[source_key] = after
                self._source_memos.move_to_end(source_key)
                while len(self._source_memos) > self.source_memo_entries:
                    self._source_memos.popitem(last=False)
            else:
                self._source_memos.pop(source_key, None)
            return reference

    def write(self, relative, record, source_key):
        if shutil.disk_usage(self.root).free <= self.minimum_free_bytes:
            raise OSError("Graph storage disk reserve reached; partial output preserved, no data omitted")
        source = {key: record[key] for key in ("source_local", "source_patch")}
        reference = self._reference(source, source_key)
        payload = {key: value for key, value in record.items() if key not in ("source_local", "source_patch")}
        payload["shared_source"] = reference
        file = self.root / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("xb") as stream:
            stream.write(gzip.compress(original.encode(payload), compresslevel=1, mtime=0))
        a, b = original._local_bound(record["source_local"]), original._local_bound(record["target_local"])
        bounds = dict(nodes=a[0] + b[0], edges=a[1] + b[1],
                      bytes=a[2] + b[2] + record["source_patch"].numel() * 4
                      + record["target_patch"].numel() * 4)
        return dict(path=relative, sha256=original.sha(file), bounds=bounds, shared_source=reference)
