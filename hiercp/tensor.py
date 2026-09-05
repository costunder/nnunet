"""Torch runtime helpers."""

from __future__ import annotations

import os
import platform
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


UNAVAILABLE = "unavailable"


class CUDAOutOfMemoryError(RuntimeError):
    """CUDA OOM with a stable project-specific type and diagnostics."""


def _read_optional_text(path: str | os.PathLike[str]) -> tuple[str | None, str | None]:
    source = Path(path)
    try:
        return source.read_text(encoding="utf-8").strip(), None
    except FileNotFoundError:
        return None, "not present"
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _memory_info() -> dict[str, int | str]:
    payload, error = _read_optional_text("/proc/meminfo")
    if payload is not None:
        values: dict[str, int] = {}
        for line in payload.splitlines():
            key, separator, raw = line.partition(":")
            fields = raw.strip().split()
            if not separator or not fields:
                continue
            try:
                amount = int(fields[0])
            except ValueError:
                continue
            values[key] = amount * (
                1024 if len(fields) > 1 and fields[1].lower() == "kb" else 1
            )
        return {
            "ram_total_bytes": values.get(
                "MemTotal", f"{UNAVAILABLE} (MemTotal absent)"
            ),
            "ram_available_bytes": values.get(
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
                "ram_total_bytes": int(status.ullTotalPhys),
                "ram_available_bytes": int(status.ullAvailPhys),
            }
        except (AttributeError, ImportError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    return {
        "ram_total_bytes": f"{UNAVAILABLE} ({error})",
        "ram_available_bytes": f"{UNAVAILABLE} ({error})",
    }


def _parse_limit(value: str | None, reason: str | None) -> int | str:
    if value is None:
        return f"{UNAVAILABLE} ({reason})"
    token = value.strip().split()[0] if value.strip() else ""
    if token in {"", "max", "-1"}:
        return "unlimited"
    try:
        return int(token)
    except ValueError:
        return f"{UNAVAILABLE} (invalid value {value!r})"


def _cgroup_info() -> dict[str, int | float | str]:
    memory_max, memory_error = _read_optional_text("/sys/fs/cgroup/memory.max")
    memory_current, current_error = _read_optional_text("/sys/fs/cgroup/memory.current")
    cpu_max, cpu_error = _read_optional_text("/sys/fs/cgroup/cpu.max")
    cpuset, cpuset_error = _read_optional_text("/sys/fs/cgroup/cpuset.cpus.effective")
    cpu_limit: int | float | str = f"{UNAVAILABLE} ({cpu_error})"
    if cpu_max:
        parts = cpu_max.split()
        if len(parts) >= 2 and parts[0] == "max":
            cpu_limit = "unlimited"
        elif len(parts) >= 2:
            try:
                quota, period = int(parts[0]), int(parts[1])
                cpu_limit = quota / period if quota >= 0 and period > 0 else "unlimited"
            except ValueError:
                cpu_limit = f"{UNAVAILABLE} (invalid cpu.max {cpu_max!r})"
    return {
        "cgroup_memory_limit_bytes": _parse_limit(memory_max, memory_error),
        "cgroup_memory_current_bytes": _parse_limit(memory_current, current_error),
        "cgroup_cpu_limit_cores": cpu_limit,
        "cgroup_cpuset": cpuset or f"{UNAVAILABLE} ({cpuset_error})",
    }


def process_memory_snapshot() -> dict[str, int | str]:
    """Return current and peak process RAM where the host exposes them."""

    payload, error = _read_optional_text("/proc/self/status")
    if payload is None:
        return {
            "process_rss_bytes": f"{UNAVAILABLE} ({error})",
            "process_peak_rss_bytes": f"{UNAVAILABLE} ({error})",
        }
    values: dict[str, int] = {}
    for line in payload.splitlines():
        key, separator, raw = line.partition(":")
        if not separator or key not in {"VmRSS", "VmHWM"}:
            continue
        fields = raw.strip().split()
        try:
            values[key] = int(fields[0]) * 1024
        except (IndexError, ValueError):
            continue
    return {
        "process_rss_bytes": values.get("VmRSS", f"{UNAVAILABLE} (VmRSS absent)"),
        "process_peak_rss_bytes": values.get(
            "VmHWM", f"{UNAVAILABLE} (VmHWM absent)"
        ),
    }


def cuda_memory_snapshot(device: torch.device | str | None = None) -> dict[str, Any]:
    """Collect CUDA allocator/device memory facts without changing allocations."""

    try:
        if not torch.cuda.is_available():
            return {"cuda_memory": f"{UNAVAILABLE} (CUDA unavailable)"}
        resolved = torch.device(device or f"cuda:{torch.cuda.current_device()}")
        if resolved.type != "cuda":
            return {"cuda_memory": f"{UNAVAILABLE} (selected device is {resolved})"}
        index = torch.cuda.current_device() if resolved.index is None else int(resolved.index)
        result: dict[str, Any] = {
            "cuda_device": f"cuda:{index}",
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated(index)),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved(index)),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        }
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            result["cuda_free_bytes"] = int(free_bytes)
            result["cuda_total_bytes"] = int(total_bytes)
        except (AttributeError, RuntimeError) as exc:
            reason = f"{UNAVAILABLE} ({type(exc).__name__}: {exc})"
            result["cuda_free_bytes"] = reason
            result["cuda_total_bytes"] = reason
        return result
    except Exception as exc:
        return {
            "cuda_memory": (
                f"{UNAVAILABLE} (CUDA diagnostic query failed: "
                f"{type(exc).__name__}: {exc})"
            )
        }


def collect_runtime_resources(
    device: torch.device | str | None = None,
    *,
    storage_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Collect host, allocation, storage, and visible accelerator resources."""

    try:
        affinity_cores: int | str = len(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        affinity_cores = f"{UNAVAILABLE} ({type(exc).__name__}: {exc})"
    storage_root = Path(storage_path or Path.cwd()).expanduser()
    if storage_root.is_file():
        storage_root = storage_root.parent
    while not storage_root.exists() and storage_root != storage_root.parent:
        storage_root = storage_root.parent
    try:
        usage = shutil.disk_usage(storage_root)
        storage_report: dict[str, Any] = {
            "storage_path": str(storage_root.resolve()),
            "storage_total_bytes": int(usage.total),
            "storage_used_bytes": int(usage.used),
            "storage_free_bytes": int(usage.free),
        }
    except OSError as exc:
        reason = f"{UNAVAILABLE} ({type(exc).__name__}: {exc})"
        storage_report = {
            "storage_path": str(storage_root),
            "storage_total_bytes": reason,
            "storage_used_bytes": reason,
            "storage_free_bytes": reason,
        }
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
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "cpu_logical_cores": os.cpu_count() if os.cpu_count() is not None else UNAVAILABLE,
        "cpu_affinity_cores": affinity_cores,
        **_memory_info(),
        **_cgroup_info(),
        **storage_report,
        "scheduler_allocation": scheduler or UNAVAILABLE,
        "container_hint": (
            os.environ.get("container")
            or ("docker" if Path("/.dockerenv").exists() else UNAVAILABLE)
        ),
        "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "nvidia_visible_devices_env": os.environ.get("NVIDIA_VISIBLE_DEVICES", "<unset>"),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_utilization_percent": f"{UNAVAILABLE} (not exposed by PyTorch)",
    }
    if not torch.cuda.is_available():
        report.update(
            {
                "cuda_visible_device_count": 0,
                "gpu_devices": [],
                "mig_visibility": f"{UNAVAILABLE} (CUDA unavailable)",
            }
        )
        return report
    count = int(torch.cuda.device_count())
    devices: list[dict[str, Any]] = []
    mig_detected = "mig-" in (
        os.environ.get("CUDA_VISIBLE_DEVICES", "")
        + os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    ).lower()
    for index in range(count):
        try:
            properties = torch.cuda.get_device_properties(index)
            name = str(properties.name)
            mig = "mig" in name.lower()
            mig_detected = mig_detected or mig
            devices.append(
                {
                    "logical_index": index,
                    "name": name,
                    "total_vram_bytes": int(properties.total_memory),
                    "multiprocessors": int(properties.multi_processor_count),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "mig_instance": mig,
                }
            )
        except (AssertionError, RuntimeError) as exc:
            devices.append(
                {
                    "logical_index": index,
                    "name": f"{UNAVAILABLE} ({type(exc).__name__}: {exc})",
                    "total_vram_bytes": UNAVAILABLE,
                    "mig_instance": UNAVAILABLE,
                }
            )
    report["cuda_visible_device_count"] = count
    report["gpu_devices"] = devices
    report["mig_visibility"] = (
        "MIG instance visible"
        if mig_detected
        else "not identifiable from PyTorch/environment"
    )
    if device is not None:
        report.update(cuda_memory_snapshot(device))
    return report


def enforce_single_device_execution(device: torch.device, *, context: str) -> None:
    """Reject final runs that were assigned unsupported extra CUDA devices."""

    if device.type != "cuda":
        return
    count = int(torch.cuda.device_count())
    if count > 1:
        raise RuntimeError(
            f"[MULTI-GPU UNSUPPORTED] {context} has one-device semantics, but this "
            f"process sees {count} CUDA devices. No validated DDP or case-distribution "
            "path exists, so the final run is refused instead of silently leaving "
            "assigned GPUs idle. Request one scheduler-assigned GPU exposed as logical "
            "cuda:0, or implement and validate explicit distributed execution. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}."
        )


def is_cuda_out_of_memory(exc: BaseException) -> bool:
    """Identify native and older string-only PyTorch CUDA OOM exceptions."""

    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom_type is not None and isinstance(exc, cuda_oom_type):
        return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def raise_cuda_out_of_memory(
    exc: BaseException,
    *,
    device: torch.device | str | None,
    context: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Raise a loud diagnostic error without implicitly reducing the workload."""

    if not is_cuda_out_of_memory(exc):
        raise TypeError("Non-CUDA-OOM exception passed to OOM reporter") from exc
    diagnostics = cuda_memory_snapshot(device)
    diagnostics.update(process_memory_snapshot())
    if extra:
        diagnostics.update(extra)
    raise CUDAOutOfMemoryError(
        f"[CUDA OOM] {context} failed. No model, graph, data, resolution, candidate, "
        f"or physical-batch reduction was attempted. diagnostics={diagnostics}; "
        f"original_error={type(exc).__name__}: {exc}"
    ) from exc


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs while allowing a fixed-shape cuDNN fast path.

    ``cudnn.benchmark`` remains disabled in deterministic mode to preserve the
    original experiment's reproducibility contract. Users can opt into the
    faster fixed-shape autotuner through the runtime config without changing
    model/loss/curriculum definitions.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG stream needed for epoch-boundary training resume.

    The training dataset itself is deterministic, but DataLoader shuffling,
    dropout, and CUDA kernels consume PyTorch RNG state.  Saving all streams at
    the end of each completed epoch lets a restarted process continue with the
    same next-epoch random sequence instead of silently changing the run.
    """

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a payload produced by :func:`capture_rng_state`."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Resume checkpoint RNG state is incomplete: {missing}")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())

    cuda_states = list(state.get("torch_cuda", []))
    if torch.cuda.is_available():
        visible = int(torch.cuda.device_count())
        if len(cuda_states) != visible:
            raise ValueError(
                "Resume checkpoint CUDA RNG count does not match this process: "
                f"saved={len(cuda_states)} visible={visible}. Keep the same "
                "CUDA_VISIBLE_DEVICES mapping when resuming."
            )
        torch.cuda.set_rng_state_all([tensor.cpu() for tensor in cuda_states])
    elif cuda_states:
        raise ValueError(
            "Resume checkpoint contains CUDA RNG state, but CUDA is unavailable."
        )


def training_state_path(checkpoint_path: str | os.PathLike[str]) -> Path:
    """Return the sidecar path that stores the resumable last-epoch state."""

    checkpoint = Path(checkpoint_path)
    return checkpoint.with_name(f"{checkpoint.stem}.last{checkpoint.suffix}")


def configure_runtime(
    *,
    deterministic: bool,
    allow_tf32: bool,
    cudnn_benchmark: bool | None = None,
) -> None:
    """Configure speed/precision knobs explicitly.

    TF32 is disabled by default because it changes float32 arithmetic. AMP still
    uses Tensor Cores on A100. ``cudnn_benchmark`` may be enabled for fixed
    patch sizes; deterministic experiments leave it off unless explicitly
    requested.
    """

    torch.backends.cudnn.deterministic = bool(deterministic)
    if cudnn_benchmark is None:
        torch.backends.cudnn.benchmark = not bool(deterministic)
    else:
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    try:
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    except (AttributeError, RuntimeError) as exc:
        print(
            "[RuntimeWarning] Could not set float32 matmul precision explicitly: "
            f"{type(exc).__name__}: {exc}"
        )


def resolve_device(requested: str) -> torch.device:
    """Resolve and initialize a concrete runtime device.

    Bare ``cuda`` is deliberately normalized to logical ``cuda:0``. On GPU
    clusters, ``CUDA_VISIBLE_DEVICES`` remaps an allocated physical GPU to
    logical index 0; using an inherited current CUDA device can otherwise
    select an out-of-range index during lazy initialization.
    """

    value = str(requested).strip().lower()
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif value == "cuda":
        value = "cuda:0"

    device = torch.device(value)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}"
        )

    index = 0 if device.index is None else int(device.index)
    count = int(torch.cuda.device_count())
    if count <= 0:
        raise RuntimeError(
            "CUDA was reported available, but torch.cuda.device_count() returned 0. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}"
        )
    if index < 0 or index >= count:
        raise RuntimeError(
            f"Invalid logical CUDA index cuda:{index}; this process sees {count} GPU(s), "
            f"so valid indices are 0..{count - 1}. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}. "
            "When a scheduler exposes one physical GPU, use --device cuda:0."
        )

    concrete = torch.device(f"cuda:{index}")
    try:
        torch.cuda.set_device(index)
        torch.empty(1, device=concrete)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize {concrete}. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}, "
            f"torch.cuda.device_count()={count}. Restrict the process to one valid "
            "allocated GPU and use logical --device cuda:0."
        ) from exc
    return concrete


def save_checkpoint_atomic(payload: dict[str, Any], path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def torch_load_compat(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    mmap: bool = False,
) -> Any:
    """Load a torch payload with optional zero-copy file-backed storages.

    PyTorch 2.6 supports both ``weights_only`` and ``mmap``. Fallbacks keep the
    project usable on older environments without silently changing payload
    semantics.
    """

    source = Path(path)
    kwargs: dict[str, Any] = {
        "map_location": map_location,
        "weights_only": False,
    }
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(source, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        try:
            return torch.load(source, **kwargs)
        except TypeError:
            kwargs.pop("weights_only", None)
            return torch.load(source, **kwargs)


def load_checkpoint(path: str | os.PathLike[str], device: torch.device) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    payload = torch_load_compat(source, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload is not a dictionary: {source}")
    return payload
