"""Fetch the complete 210-case labelled KiTS19 cohort from its official mirrors.

Every compressed file is checked against a pinned upstream digest. Existing
different files are rejected and never replaced. Partial HTTP downloads resume.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import requests

HF_REVISION = "65f1f295873a326230153c7e1de0c7dba10f0b29"
LABEL_REVISION = "456b293435b25303bbc7995f094f83a0afdff0eb"
COUNT = 210


def fingerprint(path: Path, algorithm="sha256", git_blob=False):
    h = hashlib.new(algorithm)
    if git_blob:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, item, *, image):
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        raise ValueError(f"Upstream byte size differs: {path}")
    digest = fingerprint(path) if image else fingerprint(path, "sha1", True)
    expected = item["lfs"]["oid"] if image else item["git_blob_sha1"]
    if digest != expected:
        raise ValueError(f"Upstream digest differs; original file preserved: {path}")
    return digest


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".new")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def receive(item, destination, *, image=True):
    if destination.exists():
        return verify(destination, item, image=image), True
    partial = destination.with_name(destination.name + ".download.part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == item["bytes"]:
            break
        if offset > item["bytes"]:
            raise ValueError(f"Oversized partial file preserved: {partial}")
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(item["url"], headers=headers, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                if offset:
                    expected = f"bytes {offset}-"
                    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                        raise ValueError(f"Server did not honor exact resume range: {destination}")
                with partial.open("ab" if offset else "xb") as stream:
                    for block in response.iter_content(chunk_size=2**20):
                        if block:
                            stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
            break
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    digest = verify(partial, item, image=image)
    if destination.exists():
        raise FileExistsError(f"Destination appeared during download: {destination}")
    # The dataset directory is dedicated to this run; never replace an existing CT.
    partial.rename(destination)
    return digest, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--fetch-labels", action="store_true",
                        help="Fetch missing labels from the pinned official Git repository; existing labels are checked")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("Explicit HTTP concurrency must be between 1 and 16")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    cases = metadata["files"]
    expected = [f"case_{i:05d}" for i in range(COUNT)]
    if (metadata["HF_revision"] != HF_REVISION or metadata["kits19_revision"] != LABEL_REVISION
            or len(cases) != COUNT or sorted(row["case_id"] for row in cases) != expected):
        raise ValueError("The pinned complete 210-case paper cohort is required")
    args.output.mkdir(parents=True, exist_ok=False)
    args.data.mkdir(parents=True, exist_ok=True)
    labels = {}
    def label_one(row):
        path = args.data / row["case_id"] / "segmentation.nii.gz"
        if args.fetch_labels:
            digest, _ = receive(row["segmentation"], path, image=False)
        else:
            digest = verify(path, row["segmentation"], image=False)
        return row["case_id"], digest
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="kits19_labels") as pool:
        for case, digest in pool.map(label_one, cases):
            labels[case] = digest
            if len(labels) % 25 == 0 or len(labels) == COUNT:
                print(f"KiTS19 official labels verified {len(labels)}/{COUNT}", flush=True)
    required = sum(row["image"]["bytes"] for row in cases
                   if not (args.data / row["case_id"] / "imaging.nii.gz").exists())
    free = shutil.disk_usage(args.data).free
    if free < required + 10 * 2**30:
        raise OSError(f"Raw CT needs {required / 2**30:.2f} GiB plus 10 GiB headroom; free {free / 2**30:.2f}")
    started = time.monotonic()
    state = dict(format="kits19_pinned_download_v1", complete=False, metadata_sha256=fingerprint(args.metadata),
                 cases=COUNT, completed=0, image_bytes=sum(x["image"]["bytes"] for x in cases),
                 HTTP_workers=args.workers, free_disk_bytes=free, files=[], errors=[])
    atomic_json(args.output / "progress.json", state)
    print(f"KiTS19 official labelled cohort: {COUNT} cases; {state['image_bytes']/2**30:.2f} GiB images; HTTP workers={args.workers}", flush=True)
    def one(row):
        case = row["case_id"]
        path = args.data / case / "imaging.nii.gz"
        digest, reused = receive(row["image"], path)
        import nibabel as nib
        import numpy as np
        image, label = nib.load(path), nib.load(args.data / case / "segmentation.nii.gz")
        if image.shape != label.shape or len(image.shape) != 3 or not np.allclose(image.affine, label.affine, atol=1e-5):
            raise ValueError(f"Official CT/GT geometry mismatch: {case}")
        return dict(case_id=case, CT_sha256=digest, label_git_blob_sha1=labels[case],
                    image_bytes=path.stat().st_size, shape=list(image.shape), reused=reused)
    last_print = 0.0
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="kits19_HTTP") as pool:
        futures = {pool.submit(one, row): row["case_id"] for row in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                state["files"].append(future.result())
                state["completed"] += 1
            except (requests.RequestException, OSError, ValueError, RuntimeError) as error:
                state["errors"].append(dict(case_id=case, type=type(error).__name__, error=str(error)))
            elapsed = time.monotonic() - started
            state["wall_seconds"] = elapsed
            atomic_json(args.output / "progress.json", state)
            if elapsed - last_print >= 1 or len(state["files"]) + len(state["errors"]) == COUNT:
                print(f"KiTS19 {state['completed']}/{COUNT} ({state['completed']/COUNT:.1%}) | {elapsed:.1f}s | failed={len(state['errors'])} | {case}", flush=True)
                last_print = elapsed
    state["files"].sort(key=lambda row: row["case_id"])
    state["complete"] = state["completed"] == COUNT and not state["errors"]
    atomic_json(args.output / "progress.json", state)
    if not state["complete"]:
        raise RuntimeError(f"The complete KiTS19 cohort is not ready; see {args.output / 'progress.json'}")
    atomic_json(args.output / "complete.json", state)
    print(f"COMPLETE: all {COUNT} CT/GT files verified; {args.output / 'complete.json'}", flush=True)


if __name__ == "__main__":
    main()
