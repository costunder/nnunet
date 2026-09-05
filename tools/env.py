#!/usr/bin/env python3
"""Validate the runtime, PyG imports, and the Medical data layout."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


UNAVAILABLE = "unavailable"


def _read_optional(path: str) -> tuple[str | None, str | None]:
    try:
        return Path(path).read_text(encoding="utf-8").strip(), None
    except FileNotFoundError:
        return None, "not present"
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _memory_report() -> dict[str, int | str]:
    payload, error = _read_optional("/proc/meminfo")
    if payload is not None:
        values: dict[str, int] = {}
        for line in payload.splitlines():
            key, separator, raw = line.partition(":")
            fields = raw.strip().split()
            if not separator or not fields:
                continue
            try:
                values[key] = int(fields[0]) * (
                    1024 if len(fields) > 1 and fields[1].lower() == "kb" else 1
                )
            except ValueError:
                continue
        return {
            "total_bytes": values.get("MemTotal", f"{UNAVAILABLE} (MemTotal absent)"),
            "available_bytes": values.get(
                "MemAvailable", f"{UNAVAILABLE} (MemAvailable absent)"
            ),
        }
    if platform.system() == "Windows":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx returned false")
            return {
                "total_bytes": int(status.ullTotalPhys),
                "available_bytes": int(status.ullAvailPhys),
            }
        except (AttributeError, ImportError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    return {
        "total_bytes": f"{UNAVAILABLE} ({error})",
        "available_bytes": f"{UNAVAILABLE} ({error})",
    }


def _resource_report(medical_root: Path, torch_module) -> dict[str, object]:
    try:
        affinity: int | str = len(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        affinity = f"{UNAVAILABLE} ({type(exc).__name__}: {exc})"
    storage_root = medical_root
    while not storage_root.exists() and storage_root != storage_root.parent:
        storage_root = storage_root.parent
    try:
        disk = shutil.disk_usage(storage_root)
        storage: dict[str, object] = {
            "path": str(storage_root),
            "total_bytes": int(disk.total),
            "used_bytes": int(disk.used),
            "free_bytes": int(disk.free),
        }
    except OSError as exc:
        reason = f"{UNAVAILABLE} ({type(exc).__name__}: {exc})"
        storage = {"path": str(storage_root), "total_bytes": reason, "used_bytes": reason, "free_bytes": reason}

    scheduler_keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_GPUS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "PBS_JOBID",
        "PBS_NP",
        "LSB_JOBID",
        "LSB_DJOB_NUMPROC",
        "NSLOTS",
    )
    scheduler = {key: os.environ[key] for key in scheduler_keys if key in os.environ}
    cgroup: dict[str, object] = {}
    for name, path in (
        ("memory_max", "/sys/fs/cgroup/memory.max"),
        ("memory_current", "/sys/fs/cgroup/memory.current"),
        ("cpu_max", "/sys/fs/cgroup/cpu.max"),
        ("cpuset", "/sys/fs/cgroup/cpuset.cpus.effective"),
    ):
        value, error = _read_optional(path)
        cgroup[name] = value if value else f"{UNAVAILABLE} ({error})"

    gpu_devices: list[dict[str, object]] = []
    cuda_available = bool(torch_module is not None and torch_module.cuda.is_available())
    cuda_count = int(torch_module.cuda.device_count()) if cuda_available else 0
    mig_detected = "mig-" in (
        os.environ.get("CUDA_VISIBLE_DEVICES", "")
        + os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    ).lower()
    if cuda_available:
        for index in range(cuda_count):
            try:
                properties = torch_module.cuda.get_device_properties(index)
                name = str(properties.name)
                mig = "mig" in name.lower()
                mig_detected = mig_detected or mig
                gpu_devices.append(
                    {
                        "logical_index": index,
                        "name": name,
                        "total_vram_bytes": int(properties.total_memory),
                        "compute_capability": f"{properties.major}.{properties.minor}",
                        "mig_instance": mig,
                    }
                )
            except (AssertionError, RuntimeError) as exc:
                gpu_devices.append(
                    {
                        "logical_index": index,
                        "name": f"{UNAVAILABLE} ({type(exc).__name__}: {exc})",
                        "total_vram_bytes": UNAVAILABLE,
                        "mig_instance": UNAVAILABLE,
                    }
                )

    nvidia_smi = shutil.which("nvidia-smi")
    nvidia_smi_listing: str = f"{UNAVAILABLE} (nvidia-smi not found)"
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [nvidia_smi, "-L"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            listing = (completed.stdout or completed.stderr).strip()
            nvidia_smi_listing = listing or f"{UNAVAILABLE} (nvidia-smi returned no text)"
            mig_detected = mig_detected or "MIG" in listing
        except (OSError, subprocess.SubprocessError) as exc:
            nvidia_smi_listing = f"{UNAVAILABLE} ({type(exc).__name__}: {exc})"

    return {
        "platform": platform.platform(),
        "cpu": {"logical_cores": os.cpu_count() or UNAVAILABLE, "affinity_cores": affinity},
        "ram": _memory_report(),
        "storage": storage,
        "scheduler_allocation": scheduler or UNAVAILABLE,
        "container": {
            "hint": os.environ.get("container") or ("docker" if Path("/.dockerenv").exists() else UNAVAILABLE),
            "cgroup": cgroup,
        },
        "cuda": {
            "available": cuda_available,
            "visible_device_count": cuda_count,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES", "<unset>"),
            "devices": gpu_devices,
            "mig_visibility": "MIG visible" if mig_detected else "not detected",
            "nvidia_smi_L": nvidia_smi_listing,
            "utilization_percent": f"{UNAVAILABLE} (not sampled by environment check)",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medical-root", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--require-pyg",
        action="store_true",
        help="fail when torch_geometric or required PyG symbols cannot be imported",
    )
    args = parser.parse_args()

    failures: list[str] = []
    medical_root = Path(args.medical_root).expanduser().resolve()
    image_dir = medical_root / "Data" / "image"
    label_dir = medical_root / "Data" / "labels"
    if not image_dir.is_dir():
        failures.append(f"missing image directory: {image_dir}")
    if not label_dir.is_dir():
        failures.append(f"missing label directory: {label_dir}")

    try:
        import numpy as np
        print("numpy:", np.__version__)
    except Exception as exc:  # pragma: no cover
        failures.append(f"numpy import failed: {exc}")
    try:
        import scipy
        print("scipy:", scipy.__version__)
    except Exception as exc:  # pragma: no cover
        failures.append(f"scipy import failed: {exc}")
    try:
        import nibabel as nib
        print("nibabel:", nib.__version__)
    except Exception as exc:  # pragma: no cover
        failures.append(f"nibabel import failed: {exc}")
    torch_module = None
    try:
        import torch
        torch_module = torch
        print("python:", sys.version.split()[0])
        print("torch:", torch.__version__)
        print("torch CUDA:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        if args.require_cuda and not torch.cuda.is_available():
            failures.append("CUDA was required but torch.cuda.is_available() is False")
    except Exception as exc:  # pragma: no cover
        failures.append(f"torch import failed: {exc}")
    try:
        import torch_geometric
        from torch_geometric.data import Batch, HeteroData  # noqa: F401
        from torch_geometric.nn import AttentionalAggregation, GATv2Conv, HeteroConv  # noqa: F401
        print("torch_geometric:", torch_geometric.__version__)
    except Exception as exc:
        message = f"torch_geometric import failed: {exc}"
        print("torch_geometric:", f"{UNAVAILABLE} ({exc})")
        if args.require_pyg:
            failures.append(message)

    print("resources:")
    print(json.dumps(_resource_report(medical_root, torch_module), indent=2, sort_keys=True))

    if image_dir.is_dir() and label_dir.is_dir():
        image_ids = {
            path.name[: -len("_0000.nii.gz")]
            for path in image_dir.glob("*_0000.nii.gz")
            if not path.name.startswith("._")
        }
        label_ids = {
            path.name[: -len(".nii.gz")]
            for path in label_dir.glob("*.nii.gz")
            if not path.name.startswith("._")
        }
        paired = image_ids & label_ids
        print("images:", len(image_ids))
        print("labels:", len(label_ids))
        print("paired cases:", len(paired))
        if not paired:
            failures.append("no paired NIfTI cases were found")

    if failures:
        print("\n[FAIL]")
        for failure in failures:
            print(" -", failure)
        raise SystemExit(1)
    print("\n[OK] Environment and data layout are valid")


if __name__ == "__main__":
    main()
