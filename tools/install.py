#!/usr/bin/env python3
"""Install a pinned PyTorch Geometric release without replacing PyTorch.

The project uses only base PyG operators (HeteroData, HeteroConv, GATv2Conv,
AttentionalAggregation).  Optional torch-scatter/torch-cluster wheels are not
required by this code path.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import re
import subprocess
import sys


def _major_minor(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise RuntimeError(f"Cannot parse version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def _recommended_pyg(torch_version: str) -> str:
    major, minor = _major_minor(torch_version)
    if major == 1 and minor >= 13:
        return "2.6.1"
    if major != 2:
        raise RuntimeError(
            f"No recorded PyG pin for PyTorch {torch_version}. "
            "Pass --pyg-version only after checking the official compatibility information."
        )
    if 0 <= minor <= 5:
        return "2.6.1"
    if 6 <= minor <= 8:
        return "2.7.0"
    if 9 <= minor <= 12:
        return "2.8.0.post1"
    raise RuntimeError(
        f"No recorded PyG pin for PyTorch {torch_version}. "
        "Pass --pyg-version only after checking the official compatibility information."
    )


def _installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyg-version",
        "--version",
        dest="pyg_version",
        default=None,
        help="Explicit torch-geometric version. Default: infer from active PyTorch.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-reinstall", action="store_true")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install the CUDA-matched PyTorch build first; "
            "this script deliberately does not install or replace torch."
        ) from exc

    pyg_version = args.pyg_version or _recommended_pyg(torch.__version__)
    if _major_minor(pyg_version) >= (2, 7) and sys.version_info < (3, 10):
        raise SystemExit(
            f"torch-geometric {pyg_version} requires Python 3.10+ for this project, "
            f"but the active interpreter is {sys.version.split()[0]}."
        )

    installed = _installed_version("torch-geometric")
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("torch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("selected torch-geometric:", pyg_version)
    print("currently installed torch-geometric:", installed or "not installed")

    if installed == pyg_version and not args.force_reinstall:
        print("[OK] Compatible pinned PyG version is already installed.")
        return

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"torch-geometric=={pyg_version}",
    ]
    if args.force_reinstall:
        command.extend(["--force-reinstall", "--no-cache-dir"])
    print("command:", " ".join(command))
    if args.dry_run:
        return
    subprocess.check_call(command)

    installed_after = _installed_version("torch-geometric")
    if installed_after != pyg_version:
        raise RuntimeError(
            f"Installation completed but torch-geometric={installed_after!r}; "
            f"expected {pyg_version!r}."
        )

    import torch_geometric
    from torch_geometric.data import Batch, HeteroData  # noqa: F401
    from torch_geometric.nn import AttentionalAggregation, GATv2Conv, HeteroConv  # noqa: F401

    print(f"[OK] torch-geometric {torch_geometric.__version__} imported successfully")


if __name__ == "__main__":
    main()
