"""Persist exact original source/other-lesion upper features once per source.

Only deterministic, candidate-independent original helper results are cached.
No original expression, candidate, graph relation, or annotation is changed.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

import numpy as np

FORMAT = "v18_exact_original_static_upper_v1"
_HELPER_LOCK = threading.RLock()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            value.update(block)
    return value.hexdigest()


def _binding(value):
    value = json.loads(json.dumps(value, allow_nan=False))
    for key in ("image_sha256", "label_sha256", "original_core_sha256", "region_identity_sha256"):
        if not isinstance(value.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None:
            raise ValueError(f"Static upper cache requires verified {key}")
    case = value.get("case_id")
    if not isinstance(case, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case) is None:
        raise ValueError("Static upper cache requires a safe original case identity")
    for key in ("source_component", "source_voxels", "upper_raw_dim"):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError(f"Static upper cache requires exact positive {key}")
    shape, center = value.get("shape"), value.get("anchor")
    if (not isinstance(shape, list) or len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape)
            or not isinstance(center, list) or len(center) != 3
            or any(type(n) is not int or n < 0 or n >= size for n, size in zip(center, shape))):
        raise ValueError("Static upper cache requires an exact in-volume source anchor")
    for key, size in (("spacing", 3), ("ct_clip", 2)):
        items = value.get(key)
        if (not isinstance(items, list) or len(items) != size
                or any(type(n) not in (int, float) or not math.isfinite(n) for n in items)):
            raise ValueError(f"Static upper cache requires exact finite {key}")
    if min(value["spacing"]) <= 0 or value["ct_clip"][0] >= value["ct_clip"][1]:
        raise ValueError("Static upper cache requires positive spacing and ordered CT clipping")
    if value.get("tumor_label") != 2 or (value.get("max_lesions") is not None
                                        and type(value["max_lesions"]) is not int):
        raise ValueError("Original tumor-label/max-lesions contract required")
    return value


def _arrays(kind, result, binding):
    arrays = (result,) if kind == "source_raw" else result
    if not isinstance(arrays, tuple) or len(arrays) != (1 if kind == "source_raw" else 3):
        raise ValueError("Original static upper helper returned an unexpected structure")
    if any(not isinstance(a, np.ndarray) or a.dtype.hasobject for a in arrays):
        raise ValueError("Exact original numeric NumPy arrays required")
    dimension = binding["upper_raw_dim"]
    if kind == "source_raw":
        valid = arrays[0].shape == (dimension,) and arrays[0].dtype.kind == "f"
    else:
        n = arrays[0].shape[0] if arrays[0].ndim == 2 else -1
        valid = (arrays[0].shape == (n, dimension) and arrays[1].shape == (n, 3)
                 and arrays[2].shape == (n,) and all(a.dtype.kind == "f" for a in arrays[:2])
                 and arrays[2].dtype.kind in "iu")
    if not valid or any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Original static upper array shape/dtype/finite contract differs")
    return arrays


def _copies(kind, arrays):
    copied = tuple(array.copy(order="K") for array in arrays)
    return copied[0] if kind == "source_raw" else copied


class UpperCache:
    """Small signed CPU outputs; each original factory runs only on a cache miss."""
    def __init__(self, root, budget_check):
        if not callable(budget_check):
            raise TypeError("Actual resource budget callback required")
        self.root, self.check = Path(root).resolve(), budget_check
        self.root.mkdir(parents=True, exist_ok=True)
        self._resident = {}
        self._receipts = {}
        self._lock = threading.RLock()
        self.stats = dict(original_helper_builds=0, disk_reopens=0, resident_hits=0,
                          original_build_seconds=0., cache_read_seconds=0., resident_bytes=0)

    def _obtain(self, kind, binding, factory):
        started = time.perf_counter()
        key = (kind, _digest(binding))
        with self._lock:
            self.check()
            if key in self._resident:
                self.stats["resident_hits"] += 1
                return _copies(kind, self._resident[key])
            anchor = "_".join(map(str, binding["anchor"]))
            directory = self.root / binding["case_id"] / f"c{binding['source_component']}_a{anchor}" / kind
            if directory.exists() or directory.is_symlink():
                if directory.is_symlink() or not directory.is_dir():
                    raise ValueError("Static upper cache directory is not regular")
                metadata_path, payload_path = directory / "metadata.json", directory / "arrays.npz"
                if (set(p.name for p in directory.iterdir()) != {"metadata.json", "arrays.npz"}
                        or metadata_path.is_symlink() or payload_path.is_symlink()
                        or not metadata_path.is_file() or not payload_path.is_file()):
                    raise ValueError("Incomplete static upper cache; originals preserved")
                metadata = json.loads(metadata_path.read_text(encoding="utf8"))
                signed = dict(metadata); checksum = signed.pop("metadata_sha256", None)
                if (checksum != _digest(signed) or metadata.get("format") != FORMAT
                        or metadata.get("kind") != kind or metadata.get("binding") != binding
                        or metadata.get("binding_sha256") != key[1]):
                    raise ValueError("Static upper cache binding/metadata differs; no recomputation")
                if payload_path.stat().st_size != metadata.get("payload_bytes") or _sha(payload_path) != metadata.get("payload_sha256"):
                    raise ValueError("Static upper cache payload SHA256 differs; no recomputation")
                names = [f"a{i}" for i in range(1 if kind == "source_raw" else 3)]
                with np.load(payload_path, allow_pickle=False) as payload:
                    if sorted(payload.files) != names or len(set(payload.files)) != len(payload.files):
                        raise ValueError("Static upper cache array inventory differs")
                    arrays = tuple(payload[name] for name in names)
                _arrays(kind, arrays[0] if kind == "source_raw" else arrays, binding)
                records = [dict(shape=list(a.shape), dtype=a.dtype.str, bytes=int(a.nbytes)) for a in arrays]
                if records != metadata.get("arrays"):
                    raise ValueError("Static upper cache array headers differ")
                status = "reopened"
                self.stats["disk_reopens"] += 1
                self.stats["cache_read_seconds"] += time.perf_counter() - started
            else:
                compute = time.perf_counter()
                arrays = _arrays(kind, factory(), binding)
                arrays = tuple(a.copy(order="K") for a in arrays)
                self.check()
                elapsed = time.perf_counter() - compute
                required = sum(a.nbytes for a in arrays) + 1024 ** 2
                directory.parent.mkdir(parents=True, exist_ok=True)
                if shutil.disk_usage(directory.parent).free < required:
                    raise OSError("Insufficient static upper cache disk space; no output changed")
                attempt = directory.with_name("." + kind + ".attempt." + uuid.uuid4().hex)
                attempt.mkdir(exist_ok=False)
                payload_path = attempt / "arrays.npz"
                with payload_path.open("xb") as stream:
                    np.savez(stream, **{f"a{i}": a for i, a in enumerate(arrays)})
                    stream.flush()
                metadata = dict(format=FORMAT, kind=kind, binding=binding, binding_sha256=key[1],
                    payload_sha256=_sha(payload_path), payload_bytes=payload_path.stat().st_size,
                    arrays=[dict(shape=list(a.shape), dtype=a.dtype.str, bytes=int(a.nbytes)) for a in arrays],
                    original_compute_seconds=elapsed)
                metadata["metadata_sha256"] = _digest(metadata)
                with (attempt / "metadata.json").open("x", encoding="utf8") as stream:
                    json.dump(metadata, stream, indent=2, allow_nan=False); stream.flush()
                if directory.exists() or directory.is_symlink():
                    raise FileExistsError("Concurrent static upper cache publication; owned attempt preserved")
                attempt.rename(directory)
                status = "built"
                self.stats["original_helper_builds"] += 1
                self.stats["original_build_seconds"] += elapsed
            for array in arrays:
                array.setflags(write=False)
            self._resident[key] = arrays
            self.stats["resident_bytes"] += sum(a.nbytes for a in arrays)
            self._receipts[key] = dict(kind=kind, status=status, directory=str(directory),
                binding_sha256=key[1], payload_sha256=metadata["payload_sha256"],
                array_bytes=sum(a.nbytes for a in arrays),
                disk_bytes=metadata["payload_bytes"] + (directory / "metadata.json").stat().st_size,
                original_compute_seconds=metadata["original_compute_seconds"],
                wall_seconds=time.perf_counter() - started, exact_original_arrays=True)
            self.check()
            return _copies(kind, arrays)

    @contextmanager
    def original_helpers(self, hierarchy, case, source, regions, binding):
        """Serialize the temporary module wrappers and always restore originals."""
        binding = _binding(binding)
        if (getattr(getattr(case, "paths", None), "case_id", None) != binding["case_id"]
                or tuple(case.shape) != tuple(binding["shape"])
                or list(map(float, case.spacing)) != binding["spacing"]
                or int(source.component_id) != binding["source_component"]
                or tuple(source.anchor_center) != tuple(binding["anchor"])
                or int(source.voxel_count) != binding["source_voxels"]
                or int(hierarchy.UPPER_RAW_DIM) != binding["upper_raw_dim"]):
            raise ValueError("Actual source/CT identity differs from static upper binding")
        with _HELPER_LOCK:
            original_source, original_lesions = hierarchy._source_raw, hierarchy._lesions
            if getattr(original_source, "_u_bridge_upper_wrapper", False):
                raise RuntimeError("Nested static upper wrapper is not allowed")
            def validate(actual_case, actual_source, actual_regions, ct_clip):
                if (actual_case is not case or actual_source is not source or actual_regions is not regions
                        or list(map(float, ct_clip)) != binding["ct_clip"]):
                    raise ValueError("Original upper helper requested another verified CT/source/region")
            def source_raw(actual_case, actual_source, actual_regions, occupied_without_source, *, ct_clip):
                validate(actual_case, actual_source, actual_regions, ct_clip)
                if (not isinstance(occupied_without_source, np.ndarray)
                        or occupied_without_source.shape != tuple(binding["shape"])
                        or occupied_without_source.dtype != np.dtype(bool)):
                    raise ValueError("Original source-excluded occupied-mask contract differs")
                return self._obtain("source_raw", binding, lambda: original_source(
                    actual_case, actual_source, actual_regions, occupied_without_source, ct_clip=ct_clip))
            def lesions(actual_case, actual_source, actual_regions, *, tumor_label, max_lesions, ct_clip):
                validate(actual_case, actual_source, actual_regions, ct_clip)
                if tumor_label != binding["tumor_label"] or max_lesions != binding["max_lesions"]:
                    raise ValueError("Original lesion annotation/limit contract differs")
                return self._obtain("lesions", binding, lambda: original_lesions(
                    actual_case, actual_source, actual_regions, tumor_label=tumor_label,
                    max_lesions=max_lesions, ct_clip=ct_clip))
            source_raw._u_bridge_upper_wrapper = True
            hierarchy._source_raw, hierarchy._lesions = source_raw, lesions
            try:
                yield
            finally:
                hierarchy._source_raw, hierarchy._lesions = original_source, original_lesions

    def report(self):
        with self._lock:
            return dict(format=FORMAT, **self.stats, entries=len(self._receipts),
                        disk_bytes=sum(r["disk_bytes"] for r in self._receipts.values()),
                        receipts=copy.deepcopy(list(self._receipts.values())),
                        original_formulas_preserved=True, candidates_changed=False)
