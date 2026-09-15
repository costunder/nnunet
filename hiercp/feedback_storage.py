"""Immutable graph generations with one no-clobber publication record.

Readers ignore uncommitted generations. They are deliberately preserved after
failure; only a kernel-owned lock is released, never another process or file.
Torch payloads are trusted project artifacts, not an untrusted interchange API.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import errno
import json
import os
import shutil
import time
from pathlib import Path, PurePosixPath
import uuid

import torch

from hiercp.feedback_resources import content_digest, sample_inventory, snapshot_guard

FORMAT = "hiercp_feedback_graph_commit_v2"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _fsync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@contextmanager
def _entry_lock(path, *, wait=False):
    if path.is_symlink():
        raise ValueError(f"Unsafe feedback graph lock: {path}")
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            lock, unlock = msvcrt.LK_NBLCK, msvcrt.LK_UNLCK
            acquire = lambda: msvcrt.locking(handle.fileno(), lock, 1)
            release = lambda: msvcrt.locking(handle.fileno(), unlock, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            release = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        announced = time.monotonic()
        while True:
            try:
                acquire()
                break
            except OSError as error:
                if not wait or error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise RuntimeError(f"Feedback graph entry is owned by another live publisher: {path}") from error
                if time.monotonic() - announced >= 30:
                    print(f"[FeedbackGraphStorage] Waiting for owned cache I/O lock: {path}", flush=True)
                    announced = time.monotonic()
                time.sleep(.05)
        try:
            yield
        finally:
            handle.seek(0)
            release()


class FeedbackGraphStore:
    def __init__(self, root, *, minimum_free_bytes):
        if type(minimum_free_bytes) is not int or minimum_free_bytes < 0:
            raise ValueError("Feedback graph storage requires an explicit nonnegative free-space reserve")
        self.minimum_free_bytes = minimum_free_bytes
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.witnesses = {}

    def _path(self, relative):
        parts = PurePosixPath(relative).parts
        if not parts or relative.startswith("/") or "\\" in relative or ":" in relative or any(p in {".", ".."} for p in parts):
            raise ValueError("Unsafe feedback graph generation path")
        path = self.root.joinpath(*parts)
        for ancestor in (path, *path.parents):
            if ancestor == self.root:
                break
            if ancestor.is_symlink():
                raise ValueError(f"Feedback graph generation cannot contain a symlink: {ancestor}")
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Feedback graph generation escaped its cache")
        return path

    @staticmethod
    def key(entry, binding):
        return hashlib.sha256(json.dumps({"entry": entry, "binding": binding},
                                         sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def committed(self, entry, binding):
        path = self._path(self.key(entry, binding) + ".json")
        return path.exists() or path.is_symlink()

    def read(self, entry, binding, *, count, case_id, with_sample=True):
        key = self.key(entry, binding)
        commit = self._path(key + ".json")
        required = {"format", "entry_id", "binding", "generation", "sha256", "content_sha256", "inventory"}
        try:
            record = json.loads(commit.read_text(encoding="utf-8"))
            if (set(record) != required or record["format"] != FORMAT or record["entry_id"] != entry
                    or record["binding"] != binding or record["inventory"]["candidate_count"] != count
                    or record["inventory"]["case_id"] != case_id):
                raise ValueError("commit identity/schema differs")
            path = self._path(record["generation"])
            if not path.is_file():
                raise ValueError("committed generation is missing")
            signature = (_stat(commit), _stat(path))
            witness = self.witnesses.get(key)
            if witness is not None and witness != (signature, record["sha256"], record["content_sha256"]):
                raise ValueError("committed file changed after verification")
            if witness is None:
                before = _stat(path)
                if _sha(path) != record["sha256"] or _stat(path) != before:
                    raise ValueError("payload SHA or stable identity differs")
            if not with_sample and witness is not None:
                return record["inventory"]
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            if set(payload) != {"binding", "entry_id", "sample"} or payload["binding"] != binding or payload["entry_id"] != entry:
                raise ValueError("payload identity differs")
            sample = payload["sample"]
            if sample["case_id"] != case_id:
                raise ValueError("payload patient differs")
            inventory = sample_inventory(sample, entry_id=entry, candidate_count=count)
            if inventory != record["inventory"]:
                raise ValueError("materialized inventory differs")
            if witness is None and content_digest(sample) != record["content_sha256"]:
                raise ValueError("logical graph content differs")
            if signature != (_stat(commit), _stat(path)):
                raise ValueError("graph changed while reading")
            self.witnesses[key] = (signature, record["sha256"], record["content_sha256"])
            return sample if with_sample else inventory
        except (KeyError, TypeError, json.JSONDecodeError, OSError, ValueError) as error:
            raise ValueError(f"Incomplete or changed immutable feedback graph cache: {commit}: {error}") from error

    def get_or_build(self, entry, binding, *, count, case_id, build):
        key = self.key(entry, binding)
        if self.committed(entry, binding):
            return self.read(entry, binding, count=count, case_id=case_id)
        with _entry_lock(self._path(key + ".lock")):
            if self.committed(entry, binding):
                return self.read(entry, binding, count=count, case_id=case_id)
            generation = self._path("generations/" + uuid.uuid4().hex)
            generation.mkdir(parents=True, exist_ok=False)
            payload_path = generation / "payload.pt"
            sample = build()
            inventory = sample_inventory(sample, entry_id=entry, candidate_count=count)
            snapshot_guard(inventory["input_bytes_upper_bound"], phase="feedback_graph_serialization_and_native_verification")
            logical_digest = content_digest(sample)
            payload = {"binding": binding, "entry_id": entry, "sample": sample}
            class Counter:
                def __init__(self):
                    self.position = 0
                def write(self, value):
                    self.position += len(value)
                    return len(value)
                def flush(self):
                    return None
                def tell(self):
                    return self.position
            counter = Counter()
            torch.save(payload, counter)
            # Parallel patient builders must not all approve the same free
            # bytes. Only this cache's I/O admission+write is serialized; graph
            # construction remains patient-parallel. External writers remain
            # outside our ownership and can still change filesystem headroom.
            with _entry_lock(self.root / "disk-publication.lock", wait=True):
                free = shutil.disk_usage(self.root).free
                if free < counter.position + self.minimum_free_bytes:
                    raise RuntimeError("Feedback graph disk preflight refused before payload publication: "
                                       f"free={free}, payload_bytes={counter.position}, reserved_free={self.minimum_free_bytes}; "
                                       "existing and incomplete generations preserved")
                with payload_path.open("xb") as handle:
                    torch.save(payload, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
            if payload_path.stat().st_size != counter.position:
                raise ValueError("Feedback graph serialized size differed from its measured disk admission")
            del sample, payload
            # Reopen the actual serialized payload before publishing anything.
            reopened = torch.load(payload_path, map_location="cpu", mmap=True, weights_only=False)
            if (reopened["binding"] != binding or reopened["entry_id"] != entry
                    or content_digest(reopened["sample"]) != logical_digest
                    or sample_inventory(reopened["sample"], entry_id=entry, candidate_count=count) != inventory):
                raise ValueError("Feedback graph serialization did not preserve complete native content")
            del reopened
            record = {"format": FORMAT, "entry_id": entry, "binding": binding,
                      "generation": payload_path.relative_to(self.root).as_posix(),
                      "sha256": _sha(payload_path), "content_sha256": logical_digest, "inventory": inventory}
            staged_commit = generation / "commit.json"
            with staged_commit.open("x", encoding="utf-8") as handle:
                json.dump(record, handle, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(generation)
            destination = self._path(key + ".json")
            try:
                os.link(staged_commit, destination)
            except FileExistsError:
                existing = self.read(entry, binding, count=count, case_id=case_id)
                if content_digest(existing) != logical_digest:
                    raise ValueError("Concurrent feedback graph publishers produced different native content")
                return existing
            _fsync_directory(self.root)
            # The committed read is the same verifier used on later resumes.
            return self.read(entry, binding, count=count, case_id=case_id)
