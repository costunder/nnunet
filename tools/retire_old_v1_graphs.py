"""Remove only the 609 historical graph samples approved by the user.

Indexes, model weights, prototypes, results, raw data and active arm caches are
preserved. No directory is removed. A partial run can be retried: absent indexed
samples are counted separately, while every remaining file is rechecked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

APPROVED = (
    ("HierCP/work/full/graphs", 235),
    ("HierCP/work/paired_basic_vs_hiercp/folds/fold_0/gnn/graphs", 187),
    ("HierCP/work/paired_basic_vs_hiercp/folds/fold_1/gnn/graphs", 187),
)
DEFAULT_ROOT = Path("/home/aicompetition06/Medical")


def _safe(path):
    path = Path(path).absolute()
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"Symlink path refused: {parent}")
    return path.resolve(strict=True)


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode, info.st_nlink, info.st_uid)


def _owned_regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Only regular files are eligible: {path}")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise ValueError(f"File belongs to another user: {path}")
    return info


def plan(medical_root):
    root = _safe(medical_root)
    groups = []
    for relative, expected in APPROVED:
        directory = _safe(root / relative)
        if directory != root.joinpath(*relative.split("/")) or root not in directory.parents:
            raise ValueError(f"Target differs from the approved directory: {directory}")
        if not directory.is_dir():
            raise ValueError(f"Expected graph directory: {directory}")
        index = directory / "index.json"
        index_info = _owned_regular(index)
        raw = index.read_bytes()
        if _identity(index_info) != _identity(index.lstat()):
            raise ValueError(f"Index changed during read: {index}")
        entries = json.loads(raw).get("entries")
        if not isinstance(entries, list) or len(entries) != expected:
            raise ValueError(f"Approved inventory count differs: {directory}; expected={expected}")
        records, names = [], set()
        for row in entries:
            case, number = row.get("case_id"), row.get("sample_index")
            if (not isinstance(case, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", case) is None
                    or type(number) is not int or number < 0):
                raise ValueError(f"Malformed indexed sample identity: {index}")
            name = f"{case}__{number:03d}.pt"
            if row.get("path") != name or name in names:
                raise ValueError(f"Unexpected or duplicate indexed filename: {index}")
            names.add(name)
            target = directory / name
            try:
                info = _owned_regular(target)
            except FileNotFoundError:
                records.append(dict(name=name, absent=True))
                continue
            if info.st_nlink != 1:
                raise ValueError(f"Shared hardlink is outside this approval: {target}; links={info.st_nlink}")
            if "file_size" in row and info.st_size != row["file_size"]:
                raise ValueError(f"Indexed sample size changed: {target}")
            records.append(dict(name=name, absent=False, identity=list(_identity(info)),
                bytes=info.st_size, allocated_bytes=getattr(info, "st_blocks", 0) * 512))
        groups.append(dict(directory=str(directory), index_sha256=hashlib.sha256(raw).hexdigest(),
                           index_identity=list(_identity(index_info)), samples=records))
    return dict(format="approved_old_v1_graph_retirement_v1", root=str(root),
                approved_samples=609, groups=groups)


def apply(planned, progress=print):
    # Check all targets before the first deletion; then recheck each individual
    # file immediately before unlink. No wildcard, recursive deletion or rename.
    root = _safe(planned["root"])
    expected = [(str(root / relative), count) for relative, count in APPROVED]
    if (planned.get("approved_samples") != 609
            or [(g["directory"], len(g["samples"])) for g in planned["groups"]] != expected):
        raise ValueError("Cleanup plan is not the approved three-directory inventory")
    for group in planned["groups"]:
        directory = _safe(group["directory"])
        index = directory / "index.json"
        if (list(_identity(_owned_regular(index))) != group["index_identity"]
                or hashlib.sha256(index.read_bytes()).hexdigest() != group["index_sha256"]):
            raise ValueError(f"Index changed after planning: {index}")
        entries = json.loads(index.read_bytes())["entries"]
        if [row["name"] for row in group["samples"]] != [row["path"] for row in entries]:
            raise ValueError(f"Planned sample names differ from the approved index: {index}")
        for row in group["samples"]:
            target = directory / row["name"]
            if target.parent != directory or target.name != row["name"]:
                raise ValueError("Unexpected cleanup path")
            if row["absent"]:
                if os.path.lexists(target):
                    raise ValueError(f"Previously absent sample appeared: {target}")
            elif list(_identity(_owned_regular(target))) != row["identity"]:
                raise ValueError(f"Sample changed after planning: {target}")
    removed = absent = allocated = logical = 0
    for group in planned["groups"]:
        directory = _safe(group["directory"])
        for row in group["samples"]:
            if row["absent"]:
                absent += 1
                continue
            target = directory / row["name"]
            if list(_identity(_owned_regular(target))) != row["identity"]:
                raise ValueError(f"Sample changed before unlink: {target}; partial cleanup can be retried")
            target.unlink()
            removed += 1
            allocated += row["allocated_bytes"]
            logical += row["bytes"]
            if removed % 64 == 0:
                progress(f"Old graph cleanup: removed={removed}/609")
    return dict(removed=removed, already_absent=absent, allocated_bytes_estimate=allocated,
                logical_bytes=logical, directories_removed=0, indexes_preserved=True,
                checkpoints_preserved=True, active_arm_caches_touched=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medical-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    planned = plan(args.medical_root)
    present = [s for g in planned["groups"] for s in g["samples"] if not s["absent"]]
    print(f"Approved old graph caches: present={len(present)}/609; "
          f"allocated estimate={sum(s['allocated_bytes'] for s in present)/2**30:.2f} GiB", flush=True)
    if not args.apply:
        print("Read only. Use --apply to remove only these indexed samples.", flush=True)
        return
    # Keep the small plan in the system temporary directory, using /tmp on the
    # Linux server instead of an inherited TMPDIR pointing into the account.
    # The temporary filesystem's free space is not assumed or guaranteed.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", prefix="hiercp_old_graph_cleanup_",
                                     suffix=".json", delete=False,
                                     dir="/tmp" if sys.platform.startswith("linux") else None) as stream:
        json.dump(planned, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
        audit = Path(stream.name)
    print(f"Cleanup inventory: {audit}", flush=True)
    result = apply(planned, lambda line: print(line, flush=True))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    print("Historical graph caches retired; their indexes and results remain. "
          "Rebuild graph samples before rerunning those historical experiments.", flush=True)


if __name__ == "__main__":
    main()
