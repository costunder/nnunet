"""Immutable, pickle-free storage for target-conditioned raw Copy-Paste.

Case-sized arrays are separate NPY files and are memory mapped by workers.
Only the selected candidate's small NPZ payload is materialized. Content hashes
bind every array to its manifest; a cached load still checks file witnesses.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

import numpy as np


PASTE_CONTRACT = "onlinecp_raw_target_paste_v1"
SOURCE_MAPPING_FORMAT = "online_cp_raw_target_resampling_v2"
STORAGE_FORMAT = "onlinecp_raw_target_payload_v1"
ENTRY_STORAGE = "npz_candidate_refs_raw_target_v1"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(root, relative, *, must_exist=True):
    text = str(relative)
    parts = PurePosixPath(text).parts
    if (not parts or "\\" in text or ":" in text or text.startswith("/")
            or any(part in (".", "..") for part in text.split("/"))):
        raise ValueError(f"Unsafe raw-bank relative path: {text!r}")
    root = Path(root).resolve()
    path = root.joinpath(*parts)
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"Raw-bank artifacts cannot be symlinks: {parent}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Raw-bank artifact escapes bank: {path}")
    if must_exist and not path.is_file():
        raise ValueError(f"Missing raw-bank artifact: {path}")
    return path


def _publish(path, writer):
    """Publish a new immutable file, or accept byte-identical resumed output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".pending.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha(temporary)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.is_symlink() or _sha(path) != digest:
                raise ValueError(f"Refusing to replace different raw-bank artifact: {path}")
        return digest
    finally:
        # Only our explicitly created temporary file, never an existing artifact.
        temporary.unlink(missing_ok=True)


def _encode_tree(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("Object arrays are forbidden in raw-bank payloads")
        key = f"array_{len(arrays):04d}"
        arrays[key] = value
        return {"$array": key}
    if isinstance(value, np.generic):
        return _encode_tree(value.item(), arrays)
    if isinstance(value, dict):
        if not all(isinstance(key, str) and not key.startswith("$") for key in value):
            raise ValueError("Raw-bank metadata requires non-reserved string keys")
        return {key: _encode_tree(item, arrays) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_tree(item, arrays) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported raw-bank metadata: {type(value).__name__}")


def _publish_source_array(path, array):
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Unsafe shared source array: {path}")

        class HashWriter:
            def __init__(self):
                self.digest = hashlib.sha256()

            def write(self, data):
                self.digest.update(data)
                return len(data)

        # Hash the exact NPY stream without rewriting a temporary donor file
        # for every target. np.save streams array buffers into this sink.
        sink = HashWriter()
        np.save(sink, array, allow_pickle=False)
        digest = sink.digest.hexdigest()
        if _sha(path) != digest:
            raise ValueError(f"Refusing to replace different shared source array: {path}")
        return digest
    return _publish(path, lambda handle: np.save(handle, array, allow_pickle=False))


def _save(root, relative, value, kind):
    path = _relative_path(root, relative, must_exist=False)
    arrays = {}
    tree = _encode_tree(value, arrays)
    manifest = {"format": STORAGE_FORMAT, "kind": kind, "tree": tree, "arrays": {}}
    for key, array in arrays.items():
        manifest["arrays"][key] = {"shape": list(array.shape), "dtype": array.dtype.str}
    if kind == "case":
        for key, array in arrays.items():
            relative_array = str(PurePosixPath(relative).with_suffix("")) + f".{key}.npy"
            array_path = _relative_path(root, relative_array, must_exist=False)
            digest = _publish(array_path, lambda handle, a=array: np.save(handle, a, allow_pickle=False))
            manifest["arrays"][key].update(path=relative_array, sha256=digest)
    else:
        # The same raw donor appears at every target. Keep its immutable CT
        # and mask once, instead of replicating them in all 128 archives.
        shared = {tree[name]["$array"] for name in ("source_ct", "source_mask")
                  if isinstance(tree.get(name), dict) and "$array" in tree[name]}
        for key in sorted(shared):
            array = np.ascontiguousarray(arrays[key])
            identity = hashlib.sha256()
            identity.update(json.dumps(manifest["arrays"][key], sort_keys=True).encode("ascii"))
            identity.update(array.tobytes(order="C"))
            relative_array = f"raw_sources/{identity.hexdigest()}.npy"
            array_path = _relative_path(root, relative_array, must_exist=False)
            digest = _publish_source_array(array_path, array)
            manifest["arrays"][key].update(path=relative_array, sha256=digest)
        relative_archive = str(PurePosixPath(relative).with_suffix(".npz"))
        archive_path = _relative_path(root, relative_archive, must_exist=False)
        private_arrays = {key: array for key, array in arrays.items() if key not in shared}
        digest = _publish(archive_path, lambda handle: np.savez(handle, **private_arrays))
        manifest["archive"] = {"path": relative_archive, "sha256": digest}
    encoded = (json.dumps(manifest, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    return _publish(path, lambda handle: handle.write(encoded))


def save_case(bank_root, relative_manifest, case):
    """Store runtime state only; raw preparation volumes are never duplicated."""
    return _save(bank_root, relative_manifest,
                 {key: value for key, value in case.items() if key != "preparation"}, "case")


def save_candidate(bank_root, relative_manifest, candidate):
    return _save(bank_root, relative_manifest, candidate, "candidate")


class RawBankStore:
    def __init__(self, bank_root):
        self.root = Path(bank_root).resolve()
        self._witnesses = {}
        # This bounds open mmap descriptors, not cases/samples or graph scale.
        # Eviction never discards disk data or its verified content witness.
        self._cases = OrderedDict()
        self._sources = OrderedDict()

    def __getstate__(self):
        # Spawned DataLoader workers reopen NPY mappings. Pickling a memmap
        # itself would serialize patient-sized arrays into each worker.
        return {**self.__dict__, "_cases": OrderedDict(), "_sources": OrderedDict()}

    def close(self):
        """Release owned mappings after consumers have finished using them."""
        for arrays, _ in self._cases.values():
            for array in arrays.values():
                mapped = getattr(array, "_mmap", None)
                if mapped is not None and not mapped.closed:
                    mapped.close()
        self._cases.clear()
        for array in self._sources.values():
            mapped = getattr(array, "_mmap", None)
            if mapped is not None and not mapped.closed:
                mapped.close()
        self._sources.clear()

    def _check(self, relative, digest):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Missing raw-bank content hash: {relative}")
        path = _relative_path(self.root, relative)
        stat = path.stat()
        witness = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        key = (str(relative), digest)
        previous = self._witnesses.get(key)
        if previous is not None and previous != witness:
            raise ValueError(f"Raw-bank artifact changed after verification: {path}")
        if previous is None:
            if _sha(path) != digest:
                raise ValueError(f"Raw-bank content hash mismatch: {path}")
            after = path.stat()
            if witness != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
                raise ValueError(f"Raw-bank artifact changed while verifying: {path}")
            self._witnesses[key] = witness
        return path

    def _load(self, relative, digest, kind):
        path = self._check(relative, digest)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("format") != STORAGE_FORMAT or manifest.get("kind") != kind:
            raise ValueError(f"Incompatible raw-bank payload: {path}")
        specs = manifest["arrays"]
        arrays = {}
        if kind == "case":
            key = (str(relative), digest)
            cached = self._cases.get(key)
            for name, spec in specs.items():
                array_path = self._check(spec["path"], spec["sha256"])
                arrays[name] = (cached[0][name] if cached is not None else
                                np.load(array_path, mmap_mode="r", allow_pickle=False))
        else:
            archive = manifest["archive"]
            archive_path = self._check(archive["path"], archive["sha256"])
            private = {name for name, spec in specs.items() if "path" not in spec}
            with np.load(archive_path, allow_pickle=False) as loaded:
                if set(loaded.files) != private:
                    raise ValueError(f"Raw-bank archive inventory mismatch: {archive_path}")
                arrays = {name: loaded[name] for name in private}
            for name in set(specs) - private:
                spec = specs[name]
                source_path = self._check(spec["path"], spec["sha256"])
                key = (spec["path"], spec["sha256"])
                if key not in self._sources:
                    self._sources[key] = np.load(source_path, mmap_mode="r", allow_pickle=False)
                self._sources.move_to_end(key)
                arrays[name] = self._sources[key]
                # Descriptor cache only; no donor/candidate/data is omitted.
                while len(self._sources) > 8:
                    self._sources.popitem(last=False)
        for name, array in arrays.items():
            spec = specs[name]
            if array.dtype.hasobject or list(array.shape) != spec["shape"] or array.dtype.str != spec["dtype"]:
                raise ValueError(f"Raw-bank array shape/dtype mismatch: {path}/{name}")
            array.flags.writeable = False
        referenced = set()

        def decode(value):
            if isinstance(value, dict):
                if "$array" in value:
                    if set(value) != {"$array"} or value["$array"] not in arrays:
                        raise ValueError(f"Invalid array reference in {path}")
                    referenced.add(value["$array"])
                    return arrays[value["$array"]]
                return {name: decode(item) for name, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return value

        result = decode(manifest["tree"])
        # The recursive function otherwise holds a self-referential closure,
        # retaining mmap arrays until cyclic GC rather than normal eviction.
        decode = None
        if referenced != set(arrays) or not isinstance(result, dict):
            raise ValueError(f"Unused arrays or malformed root in {path}")
        if kind == "case":
            self._cases[(str(relative), digest)] = (arrays, result)
            self._cases.move_to_end((str(relative), digest))
            while len(self._cases) > 8:
                self._cases.popitem(last=False)
        return result

    def load_case(self, relative, sha256):
        return self._load(relative, sha256, "case")

    def load_candidate(self, relative, sha256):
        return self._load(relative, sha256, "candidate")


def audit_source_entry(bank_root, payload, candidate_count, *, store=None):
    """Verify the entire candidate cohort and its shared case reference."""
    store = RawBankStore(bank_root) if store is None else store
    required = {"paste_contract", "case_id", "candidate_centers", "candidate_raw_centers", "scores",
                "candidate_payloads", "candidate_payload_sha256", "raw_case_reference",
                "raw_case_reference_sha256", "source_component", "source_diameter_mm"}
    if not required.issubset(payload):
        raise ValueError(f"Incomplete raw-target source entry: {sorted(required - set(payload))}")
    if {"source_data", "source_mask", "anchor_offset"}.intersection(payload):
        raise ValueError("Raw-target entry contains ambiguous legacy source-anchored tensors")

    def scalar(name):
        value = np.asarray(payload[name])
        if value.shape != (1,):
            raise ValueError(f"Raw-target source entry requires one {name}")
        return value[0].item()

    if scalar("paste_contract") != PASTE_CONTRACT:
        raise ValueError("Wrong raw-target paste contract")
    case_id = scalar("case_id")
    component = int(scalar("source_component"))
    case_digest = scalar("raw_case_reference_sha256")
    case = store.load_case(scalar("raw_case_reference"), case_digest)
    if case["metadata"].get("case_id") != case_id:
        raise ValueError("Raw-target source/case identity mismatch")
    centers = np.asarray(payload["candidate_centers"])
    raw_centers = np.asarray(payload["candidate_raw_centers"])
    scores = np.asarray(payload["scores"])
    references = np.asarray(payload["candidate_payloads"])
    digests = np.asarray(payload["candidate_payload_sha256"])
    if (centers.shape != (candidate_count, 3) or raw_centers.shape != centers.shape
            or centers.dtype.kind not in "iu" or raw_centers.dtype.kind not in "iu"
            or scores.shape != (candidate_count,) or not np.isfinite(scores).all()
            or references.shape != (candidate_count,) or digests.shape != references.shape
            or references.dtype.kind != "U" or digests.dtype.kind != "U"):
        raise ValueError("Invalid raw-target candidate arrays")
    if len(np.unique(raw_centers, axis=0)) != candidate_count or len(set(references)) != candidate_count:
        raise ValueError("Duplicate raw candidates or candidate payload references")
    shape = np.asarray(case["metadata"]["preprocessed_shape"])
    metadata = case["metadata"]
    transpose = np.asarray(metadata["transpose_forward"], dtype=np.int64)
    crop = np.asarray(metadata["crop_bbox"], dtype=np.int64)
    cropped_shape = np.asarray(metadata["cropped_shape"], dtype=np.int64)
    if (transpose.shape != (3,) or sorted(transpose.tolist()) != [0, 1, 2]
            or crop.shape != (3, 2) or cropped_shape.shape != (3,) or np.any(cropped_shape <= 0)):
        raise ValueError("Invalid native case mapping metadata")
    cropped = raw_centers[:, ::-1][:, transpose] - crop[:, 0]
    mapped = np.rint((cropped + .5) * shape / cropped_shape - .5).astype(np.int64)
    if not np.array_equal(centers, mapped):
        raise ValueError("Raw-target/native candidate anchor mapping mismatch")
    # An inactive raw anchor need not be inside the cropped native grid.
    # Preserve its coordinates; actual candidate output boxes below must fit.
    source_digest = None
    for index, (relative, digest) in enumerate(zip(references, digests)):
        candidate = store.load_candidate(str(relative), str(digest))
        if (candidate.get("case_reference_sha256") != case_digest or candidate.get("case_id") != case_id
                or candidate.get("source_component") != component
                or candidate.get("raw_target_center") != raw_centers[index].tolist()):
            raise ValueError("Raw-target candidate/source identity mismatch")
        bbox = np.asarray(candidate["output_bbox"])
        mask = np.asarray(candidate["source_mask"])
        support = np.asarray(candidate["pasted_support"])
        seg = np.asarray(candidate["seg_patch"])
        if (bbox.shape != (3, 2) or np.any(bbox[:, 0] < 0) or np.any(bbox[:, 1] > shape)
                or np.any(bbox[:, 1] <= bbox[:, 0]) or mask.ndim != 3 or not mask.any()
                or mask.dtype != np.dtype(bool) or support.dtype != np.dtype(bool)
                or support.shape != tuple(bbox[:, 1] - bbox[:, 0])
                or seg.shape != (1, *support.shape) or np.any(support & (seg[0] != 2))):
            raise ValueError("Malformed raw-target candidate geometry/support")
        raw_source = np.asarray(candidate["source_ct"])
        if raw_source.shape != (1, *mask.shape) or not np.isfinite(raw_source).all():
            raise ValueError("Invalid raw-target source CT")
        identity = hashlib.sha256(mask.tobytes() + raw_source.tobytes()).hexdigest()
        if source_digest is not None and identity != source_digest:
            raise ValueError("Candidate pool substitutes different source components")
        source_digest = identity
    return case
