#!/usr/bin/env python3
"""Install the two HierCP OnlineCP custom trainer modules into nnU-Net v2.

The script is intentionally portable. By default it locates the active
``nnunetv2`` package. Use ``--nnunet-root`` to target a source checkout, for
example on Windows::

    python install_onlinecp_custom_trainers.py apply \
      --nnunet-root C:\\path\\to\\nnunetv2

The two source modules must be in the same directory as this installer:

- nnUNetTrainer_OnlinePairedCP.py
- nnUNetTrainer_OnlinePairedCPArgmaxV3.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODULES = {
    "nnUNetTrainer_OnlinePairedCP.py": (
        "75ed5fbb8230f9f4da241905759be0ca7613ab3b45db76396cd6f5c560c50154"
    ),
    "nnUNetTrainer_OnlinePairedCPArgmaxV3.py": (
        "0fb1b1f5e7f602fcc6be57bcf205b673d2749190ca686aa919ebd66bea8a00b2"
    ),
}
TARGET_RELATIVE = Path("training") / "nnUNetTrainer"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_nnunet_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if root.name != "nnunetv2" and (root / "nnunetv2").is_dir():
            root = root / "nnunetv2"
        if not (root / "training" / "nnUNetTrainer").is_dir():
            raise RuntimeError(
                "--nnunet-root must point to the nnunetv2 package directory "
                f"(missing training/nnUNetTrainer): {root}"
            )
        return root

    spec = importlib.util.find_spec("nnunetv2")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(
            "nnunetv2 is not importable. Activate the nnU-Net environment or "
            "pass --nnunet-root."
        )
    locations = [Path(value).resolve() for value in spec.submodule_search_locations]
    if len(locations) != 1:
        raise RuntimeError(f"Ambiguous nnunetv2 package roots: {locations}")
    return locations[0]


def source_dir() -> Path:
    return Path(__file__).resolve().parent


def audit_sources() -> dict[str, Path]:
    root = source_dir()
    output: dict[str, Path] = {}
    for name, expected in MODULES.items():
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"Required trainer source is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Trainer source hash mismatch for {name}:\n"
                f"  actual:   {actual}\n"
                f"  expected: {expected}"
            )
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        output[name] = path
    return output


def atomic_write_bytes(data: bytes, target: Path, *, overwrite: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as dst:
            dst.write(data)
            dst.flush()
            os.fsync(dst.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_copy(source: Path, target: Path, *, overwrite: bool = False) -> None:
    atomic_write_bytes(source.read_bytes(), target, overwrite=overwrite)


def import_smoke(nnunet_root: Path) -> None:
    parent = nnunet_root.parent
    code = r'''
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCP import (
    nnUNetTrainer_250epochs_OnlineBasicCP,
    nnUNetTrainer_250epochs_OnlineHierCPExactArgmax,
    nnUNetTrainer_250epochs_OnlineHierCPNoPatientExactArgmax,
    nnUNetTrainer_250epochs_OnlineHierCPNoPopulationExactArgmax,
    _smoke_policy as smoke_v2,
    _smoke_paste as paste_v2,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCPArgmaxV3 import (
    nnUNetTrainer_250epochs_OnlineBasicCPSharedPoolsV3,
    nnUNetTrainer_250epochs_OnlineHierCPArgmaxV3,
    _smoke_policy as smoke_v3,
    _smoke_paste as paste_v3,
)
assert nnUNetTrainer_250epochs_OnlineBasicCP.online_policy == "basic"
assert nnUNetTrainer_250epochs_OnlineHierCPExactArgmax.online_policy == "hier_argmax"
assert nnUNetTrainer_250epochs_OnlineHierCPNoPatientExactArgmax.expected_ablation_mode == "no_patient"
assert nnUNetTrainer_250epochs_OnlineHierCPNoPopulationExactArgmax.expected_ablation_mode == "no_population"
assert nnUNetTrainer_250epochs_OnlineBasicCPSharedPoolsV3.online_policy == "basic"
assert nnUNetTrainer_250epochs_OnlineHierCPArgmaxV3.online_policy == "hier"
print("[OK] v2 policy", smoke_v2())
print("[OK] v2 paste", paste_v2())
print("[OK] v3 policy", smoke_v3())
print("[OK] v3 paste", paste_v3())
print("[OK] OnlineCP custom trainer imports")
'''
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(parent) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        raise RuntimeError("Installed trainer import smoke failed")


def check(nnunet_root: Path) -> int:
    sources = audit_sources()
    target_dir = nnunet_root / TARGET_RELATIVE
    print(f"python:  {Path(sys.executable).resolve()}")
    print(f"nnunet:  {nnunet_root}")
    print(f"target:  {target_dir}")
    for name, source in sources.items():
        target = target_dir / name
        state = "missing"
        if target.is_file():
            state = "same" if sha256(target) == MODULES[name] else "different"
        print(f"  {name}: source=OK target={state}")
    print("[OK] source syntax and SHA-256 audit")
    return 0


def apply(nnunet_root: Path, *, overwrite: bool = False) -> int:
    sources = audit_sources()
    target_dir = nnunet_root / TARGET_RELATIVE
    if not target_dir.resolve().is_relative_to(nnunet_root.resolve()):
        raise RuntimeError(f"Trainer target escapes the requested package root: {target_dir}")
    originals: dict[Path, tuple[bytes | None, int | None]] = {}
    conflicts: list[Path] = []
    for name in MODULES:
        target = target_dir / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise RuntimeError(f"Trainer target must be a regular file or absent: {target}")
        if target.is_file():
            originals[target] = (target.read_bytes(), target.stat().st_mode & 0o777)
            if hashlib.sha256(originals[target][0]).hexdigest() != MODULES[name]:
                conflicts.append(target)
        else:
            originals[target] = (None, None)
        print(f"[InstallPlan] {sources[name]} -> {target}; overwrite={overwrite}")
    if conflicts and not overwrite:
        raise FileExistsError(
            "Existing trainer implementations differ; no trainer files were changed. "
            "Review these paths and use --overwrite only to authorize their replacement: "
            + ", ".join(str(path) for path in conflicts)
        )

    changed: list[Path] = []
    try:
        for name, source in sources.items():
            target = target_dir / name
            original_bytes, _ = originals[target]
            if original_bytes is not None and hashlib.sha256(original_bytes).hexdigest() == MODULES[name]:
                print(f"[REUSE] identical trainer: {target}")
                continue
            if original_bytes is not None:
                backup = target.with_name(target.name + f".pre_hiercp_onlinecp.{time.time_ns()}")
                atomic_write_bytes(original_bytes, backup)
                print(f"[BACKUP] {backup}")
            atomic_copy(source, target, overwrite=original_bytes is not None)
            changed.append(target)
            if sha256(target) != MODULES[name]:
                raise RuntimeError(f"Post-copy hash mismatch: {target}")
            print(f"[INSTALL] {target}")

        for name in MODULES:
            subprocess.run(
                [sys.executable, "-m", "py_compile", str(target_dir / name)],
                check=True,
            )
        import_smoke(nnunet_root)
    except Exception:
        for target in reversed(changed):
            data, mode = originals[target]
            if data is None:
                if target.exists():
                    target.unlink()
            else:
                atomic_write_bytes(data, target, overwrite=True)
                if mode is not None:
                    os.chmod(target, mode)
        print("[ROLLBACK] Trainer installation rolled back", file=sys.stderr)
        raise

    print("[OK] Both OnlineCP custom trainer modules installed and imported")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "apply"))
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Explicitly authorize replacing different existing trainers after a unique backup.",
    )
    parser.add_argument(
        "--nnunet-root",
        help="Path to the nnunetv2 package directory; defaults to the active environment.",
    )
    args = parser.parse_args()
    try:
        root = locate_nnunet_root(args.nnunet_root)
        if args.action == "check":
            raise SystemExit(check(root))
        raise SystemExit(apply(root, overwrite=args.overwrite))
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
