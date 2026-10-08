"""Recover duplicate whole-case field storage without changing any cache bytes.

Standalone Python 3.10+ standard library tool. Default is read-only. Interactive
mode verifies candidate bytes, prints a reviewable plan, then asks for approval.
Only depth.npy/occupied.npy in the five named experiment caches are eligible.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import tempfile
import threading
import time
import uuid


EXPERIMENTS = (
    "v18_u_bridge_m10_seed42",
    "v18_selected_m10_seed42_memory",
    "v18_native_m10_seed42_memory",
    "v19_native_fixed_m10_seed42",
    "v19_native_listwise_m10_seed42",
)
FORMAT = "v18_exact_whole_case_distance_fields_v1"
GIB = 1024 ** 3


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def checked_path(path, directory=False):
    path = Path(os.path.abspath(path))
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"Symlink is outside the maintenance contract: {parent}")
    info = path.stat()
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError(f"Unexpected path type: {path}")
    return path, info


def witness(info):
    return dict(device=info.st_dev, inode=info.st_ino, size=info.st_size,
                mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns,
                links=info.st_nlink, uid=info.st_uid, gid=info.st_gid,
                mode=stat.S_IMODE(info.st_mode),
                allocated_bytes=(info.st_blocks * 512 if hasattr(info, "st_blocks") else None))


def stable_key(value):
    # Linking changes ctime/nlink, but never content mtime/size or permissions.
    return tuple(value[k] for k in ("device", "inode", "size", "mtime_ns", "uid", "gid", "mode"))


def metadata_records(case_dir):
    checked_path(case_dir, directory=True)
    path, _ = checked_path(case_dir / "metadata.json")
    raw = path.read_bytes()
    metadata = json.loads(raw)
    signed = dict(metadata)
    signature = signed.pop("metadata_sha256", None)
    binding = metadata.get("binding")
    if (metadata.get("format") != FORMAT or signature != digest(signed)
            or not isinstance(binding, dict)
            or metadata.get("binding_sha256") != digest(binding)
            or set(metadata.get("fields", {})) != {"depth", "occupied"}):
        raise ValueError(f"Invalid signed field metadata: {path}")
    shape = binding.get("shape")
    if (not isinstance(shape, list) or len(shape) != 3
            or any(type(n) is not int or n <= 0 for n in shape)):
        raise ValueError(f"Invalid native shape: {path}")
    records = []
    for name in ("depth", "occupied"):
        field = metadata["fields"][name]
        if not isinstance(field, dict):
            raise ValueError(f"Invalid field record: {path}: {name}")
        payload, info = checked_path(case_dir / (name + ".npy"))
        if (field.get("file") != payload.name or field.get("units") != "mm"
                or field.get("file_bytes") != info.st_size or field.get("shape") != shape
                or re.fullmatch(r"[0-9a-f]{64}", str(field.get("sha256"))) is None):
            raise ValueError(f"Invalid field publication: {payload}")
        records.append(dict(path=str(payload), case=case_dir.name, field=name,
                            binding_sha256=metadata["binding_sha256"],
                            field_record=field, sha256=field["sha256"],
                            metadata_path=str(path), metadata_file_sha256=hashlib.sha256(raw).hexdigest(),
                            stat=witness(info)))
    return records


def scan(root):
    root, _ = checked_path(root, directory=True)
    records, errors, roots = [], [], []
    for experiment in EXPERIMENTS:
        directory = root / experiment / "data" / "whole_case_fields"
        if not directory.exists() and not directory.is_symlink():
            roots.append(dict(experiment=experiment, status="missing", files=0))
            continue
        checked_path(directory, directory=True)
        start = len(records)
        for case in sorted(directory.iterdir()):
            if case.name.startswith("."):
                continue  # Unpublished attempts/locks are never cleanup targets.
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case.name) is None:
                errors.append(dict(path=str(case), error="Invalid case directory name"))
                continue
            try:
                records.extend(metadata_records(case))
            except (OSError, ValueError, TypeError, KeyError) as error:
                errors.append(dict(path=str(case), error=str(error)))
        roots.append(dict(experiment=experiment, status="scanned", files=len(records)-start))
    groups = {}
    for record in records:
        info = record["stat"]
        key = (record["case"], record["binding_sha256"], record["field"],
               digest(record["field_record"]), info["device"], info["uid"], info["gid"], info["mode"])
        groups.setdefault(key, []).append(record)
    pairs = []
    for group in groups.values():
        by_inode = {}
        for record in sorted(group, key=lambda x: x["path"]):
            by_inode.setdefault((record["stat"]["device"], record["stat"]["inode"]), record)
        independent = list(by_inode.values())
        if len(independent) < 2:
            continue
        # Keep the already shared inode, otherwise the largest allocated copy.
        independent.sort(key=lambda r: (-r["stat"]["links"], -(r["stat"]["allocated_bytes"] or 0), r["path"]))
        source = independent[0]
        for target in independent[1:]:
            # Unknown external links cannot yield a definite recovery estimate.
            if target["stat"]["links"] == 1:
                pairs.append(dict(source=source, target=target,
                                  reclaim_estimate_bytes=target["stat"]["allocated_bytes"]))
    return dict(format="exact_field_dedup_v1", experiments_root=str(root), roots=roots,
                metadata_errors=errors, metadata_valid_files=len(records), pairs=pairs,
                verified=False, applied=False)


def current(record, exact=True):
    _, info = checked_path(record["path"])
    value = witness(info)
    if (value != record["stat"] if exact else stable_key(value) != stable_key(record["stat"])):
        raise RuntimeError(f"File changed since inventory: {record['path']}")
    meta, _ = checked_path(record["metadata_path"])
    if hashlib.sha256(meta.read_bytes()).hexdigest() != record["metadata_file_sha256"]:
        raise RuntimeError(f"Metadata changed since inventory: {meta}")
    return info


def verify_record(record, cancelled=None):
    current(record)
    value = hashlib.sha256()
    with open(record["path"], "rb") as stream:
        if witness(os.fstat(stream.fileno())) != record["stat"]:
            raise RuntimeError(f"File replaced before verification: {record['path']}")
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("Field verification cancelled; no cache files changed")
            value.update(block)
        if witness(os.fstat(stream.fileno())) != record["stat"]:
            raise RuntimeError(f"File changed during verification: {record['path']}")
    current(record)
    if value.hexdigest() != record["sha256"]:
        raise ValueError(f"Actual payload SHA256 differs: {record['path']}; no replacement")


def verify(plan, progress=print):
    unique = {}
    for pair in plan["pairs"]:
        for record in (pair["source"], pair["target"]):
            unique.setdefault((record["stat"]["device"], record["stat"]["inode"]), record)
    total_bytes = sum(r["stat"]["size"] for r in unique.values())
    progress(f"Verifying actual bytes: {len(unique)} files, {total_bytes/GIB:.2f} GiB; 2 bounded readers")
    completed = 0
    started = time.monotonic()
    cancelled = threading.Event()
    # Two readers bound shared-storage pressure and use at most 16 MiB buffers.
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(verify_record, r, cancelled) for r in unique.values()}
        try:
            while pending:
                done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    completed += 1
                progress(f"SHA256 {completed}/{len(unique)} verified | {time.monotonic()-started:.0f}s", flush=True)
        except BaseException:
            cancelled.set()
            for future in pending:
                future.cancel()
            raise
    if completed != len(unique):
        raise RuntimeError("Incomplete payload verification")
    plan["verified"] = True
    return plan


def active_owners(root):
    owners = []
    for name in EXPERIMENTS:
        experiment = root / name
        paths = [experiment / ".pipeline.lock", experiment / ".setup.lock", experiment / "data" / ".data.lock"]
        paths += [experiment / ("." + arm + ".lock") for arm in ("selected", "native", "native_fixed", "native_listwise")]
        for path in paths:
            if not path.exists() and not path.is_symlink():
                continue
            try:
                checked_path(path)
                value = json.loads(path.read_text())
                host, pid = value.get("host"), value.get("pid")
                if host != socket.gethostname() or type(pid) is not int or pid <= 0:
                    owners.append(dict(path=str(path), owner=value, reason="remote_or_unknown_owner"))
                    continue
                if os.name != "posix":
                    owners.append(dict(path=str(path), owner=value, reason="non_posix_owner_check_unavailable"))
                    continue
                try:
                    os.kill(pid, 0)  # Existence check only; sends no signal.
                except ProcessLookupError:
                    continue
                except PermissionError:
                    owners.append(dict(path=str(path), owner=value, reason="owner_not_inspectable"))
                else:
                    owners.append(dict(path=str(path), owner=value, reason="owner_alive"))
            except (OSError, ValueError, TypeError, AttributeError) as error:
                owners.append(dict(path=str(path), reason=f"unreadable_owner: {error}"))
    return owners


def _same_inode(left, right):
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def replace_pair(pair):
    source, target = pair["source"], pair["target"]
    current(source, exact=False)
    old = current(target)
    if old.st_nlink != 1:
        raise RuntimeError(f"Target has additional links: {target['path']}")
    temporary = Path(target["path"]).with_name(".field-link-" + uuid.uuid4().hex)
    result = dict(source=source["path"], target=target["path"],
                  reclaim_estimate_bytes=pair["reclaim_estimate_bytes"])
    try:
        try:
            os.link(source["path"], temporary)
        except OSError:
            # NFS can report failure after completing the operation. Only an
            # exact owned temporary link is a successful, recoverable outcome.
            if not temporary.exists() or not _same_inode(temporary.stat(), Path(source["path"]).stat()):
                raise
        current(source, exact=False)
        current(target)
        if not _same_inode(temporary.stat(), Path(source["path"]).stat()):
            raise RuntimeError(f"Unexpected temporary inode: {temporary}")
        try:
            os.replace(temporary, target["path"])
        except OSError:
            if not _same_inode(Path(target["path"]).stat(), Path(source["path"]).stat()):
                raise
        if not _same_inode(Path(target["path"]).stat(), Path(source["path"]).stat()):
            raise RuntimeError(f"Replacement did not publish expected inode: {target['path']}")
        result["status"] = "deduplicated"
        return result
    finally:
        if temporary.exists():
            # Never remove another file if an unexpected path was substituted.
            if not temporary.is_symlink() and _same_inode(temporary.stat(), Path(source["path"]).stat()):
                temporary.unlink()


def apply(plan, journal, progress=print):
    if not plan.get("verified"):
        raise ValueError("Actual payload verification is required")
    if hasattr(os, "getuid"):
        for pair in plan["pairs"]:
            if any(record["stat"]["uid"] != os.getuid() for record in (pair["source"], pair["target"])):
                raise PermissionError("Only the current user's own field files can be consolidated")
    owners = active_owners(Path(plan["experiments_root"]))
    if owners:
        raise RuntimeError("Cache maintenance needs inactive experiments; owners=" + json.dumps(owners))
    # A final all-file preflight avoids partial changes on already stale plans.
    for pair in plan["pairs"]:
        current(pair["source"])
        current(pair["target"])
    completed = []
    for index, pair in enumerate(plan["pairs"], 1):
        # Guard against a new recorded trainer being started during maintenance.
        owners = active_owners(Path(plan["experiments_root"]))
        if owners:
            raise RuntimeError("Experiment started during maintenance; remaining paths unchanged: " + json.dumps(owners))
        event = dict(stage="before_replace", target=pair["target"]["path"])
        journal.write(json.dumps(event) + "\n"); journal.flush(); os.fsync(journal.fileno())
        result = replace_pair(pair)
        completed.append(result)
        journal.write(json.dumps(result) + "\n"); journal.flush(); os.fsync(journal.fileno())
        progress(f"Shared exact cache {index}/{len(plan['pairs'])}", flush=True)
    plan["applied"] = True
    return completed


def probe_writes(root):
    """Test actual previously blocked locations, without creating output trees."""
    locations = [("Git objects", root.parent / "HierCP-regions-8580e59" / ".git" / "objects")]
    for name, arm in zip(EXPERIMENTS[1:], ("selected", "native", "native_fixed", "native_listwise")):
        locations.append((arm, root / name / arm))
    results = []
    for label, directory in locations:
        if not directory.exists():
            results.append(dict(location=label, status="NOT_FOUND", path=str(directory)))
            continue
        directory, _ = checked_path(directory, directory=True)
        path = directory / (".space-check-" + uuid.uuid4().hex)
        created = False
        result = dict(location=label, path=str(directory))
        try:
            with path.open("xb") as stream:
                created = True
                stream.write(b"quota recovery write check\n".ljust(4096, b"."))
                stream.flush()
                os.fsync(stream.fileno())
            result["status"] = "PASS"
        except OSError as error:
            result.update(status="FAIL", errno=error.errno, error=str(error))
        finally:
            if created:
                try:
                    path.unlink()
                except OSError as error:
                    result.update(status="FAIL", cleanup_error=str(error))
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=Path.home() / "Medical" / "experiments")
    parser.add_argument("--interactive", action="store_true", help="verify bytes, review, then request explicit approval")
    args = parser.parse_args()
    plan = scan(args.experiments)
    estimate = sum(p["reclaim_estimate_bytes"] or 0 for p in plan["pairs"])
    print("Only completed whole_case_fields depth.npy/occupied.npy caches are in scope.")
    for row in plan["roots"]:
        print(f"{row['experiment']}: {row['status']}, {row['files']} field files")
    print(f"Independent duplicate candidates: {len(plan['pairs'])}; allocated estimate={estimate/GIB:.2f} GiB")
    print("Checkpoints, metadata, source CT, graphs and experiment settings are preserved.")
    for error in plan["metadata_errors"]:
        print("EXCLUDED invalid publication: " + json.dumps(error, ensure_ascii=False))
    if not plan["pairs"]:
        print("No eligible independent duplicates. No files changed; do not repeat old609 cleanup.")
        return
    if not args.interactive:
        print("READ ONLY: metadata candidates only, not yet payload-verified. Use --interactive to review/apply.")
        return
    if os.name != "posix":
        raise RuntimeError("Interactive server maintenance requires Linux/POSIX; read-only scan is available here")
    owners = active_owners(Path(plan["experiments_root"]))
    if owners:
        raise RuntimeError("Pause the listed experiments before maintenance. No process will be stopped: " + json.dumps(owners))
    verify(plan)
    # The home quota must not prevent recording an exact plan and journal.
    fd, plan_name = tempfile.mkstemp(prefix="hiercp-fields-", suffix=".json", dir="/tmp")
    with os.fdopen(fd, "w") as stream:
        json.dump(plan, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    print(f"Verified plan (every exact source/target path): {plan_name}")
    for name in EXPERIMENTS:
        selected = [p for p in plan["pairs"] if Path(p["target"]["path"]).parts[-5] == name]
        if selected:
            amount = sum(p["reclaim_estimate_bytes"] or 0 for p in selected)
            print(f"REPLACE {plan['experiments_root']}/{name}/data/whole_case_fields: {len(selected)} files, {amount/GIB:.2f} GiB estimated")
    print("This replaces only verified duplicate file inodes with hardlinks; all paths and bytes remain identical.")
    print("All five experiments must stay stopped until this finishes. Open mappings can delay reclaimed space.")
    answer = input("Approve this plan and confirm no listed experiment is running on any host: type SHARE: ").strip()
    if answer != "SHARE":
        print("Not approved. No cache files changed.")
        return
    journal_name = plan_name + ".journal"
    with open(journal_name, "x") as journal:
        results = apply(plan, journal)
    print(f"DONE: {len(results)} duplicate paths shared; estimated released allocation={estimate/GIB:.2f} GiB")
    print(f"Journal: {journal_name}")
    for result in probe_writes(Path(plan["experiments_root"])):
        print("WRITE CHECK " + json.dumps(result))
    print("Write checks cover current 4-KiB writes only; they do not reserve future checkpoint/cache capacity.")


if __name__ == "__main__":
    main()
