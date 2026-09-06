"""Measured CPU preparation scheduling. No cohort or graph-size reduction.

Calibration waves perform real pending work and retain their outputs. Different
cases are not identical microbenchmarks: the report explicitly records this
limitation and never labels the selected concurrency globally optimal.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid


def _unescape_mount(value):
    for escaped, plain in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, plain)
    return value


@lru_cache(maxsize=1)
def _cgroup_locations():
    """Resolve this process's v1/v2 leaves and all applicable ancestors.

    Mount roots matter in containers: a mounted cgroup subtree is not the
    host's filesystem root. No guessed host-wide path replaces a known leaf.
    """
    membership_path, mounts_path = Path("/proc/self/cgroup"), Path("/proc/self/mountinfo")
    if not membership_path.is_file() or not mounts_path.is_file():
        return ()
    memberships = []
    for line in membership_path.read_text(encoding="utf-8").splitlines():
        _, controllers, relative = line.split(":", 2)
        if ".." in Path(relative).parts or not relative.startswith("/"):
            raise RuntimeError("Cannot safely resolve this process's cgroup namespace membership")
        memberships.append((set(controllers.split(",")) if controllers else set(), Path(relative)))
    locations = []
    for line in mounts_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        separator = fields.index("-")
        filesystem = fields[separator + 1]
        if filesystem not in {"cgroup", "cgroup2"}:
            continue
        mount_root, mount_point = Path(_unescape_mount(fields[3])), Path(_unescape_mount(fields[4]))
        supported = set(fields[separator + 3].split(","))
        for controllers, member in memberships:
            kinds = {"v2"} if filesystem == "cgroup2" and not controllers else supported & controllers & {"memory", "cpu"}
            if not kinds or not member.is_relative_to(mount_root):
                continue
            leaf = mount_point / member.relative_to(mount_root)
            directory = leaf
            while True:
                for kind in sorted(kinds):
                    item = (kind, str(directory))
                    if item not in locations:
                        locations.append(item)
                if directory == mount_point:
                    break
                directory = directory.parent
    return tuple(locations)


def _allocation_limits(available, cpu_limit):
    evidence, missing = {}, []
    locations = _cgroup_locations()
    for kind, directory in locations:
        names = (("memory.max", "memory.current", "cpu.max") if kind == "v2" else
                 ("memory.limit_in_bytes", "memory.usage_in_bytes") if kind == "memory" else
                 ("cpu.cfs_quota_us", "cpu.cfs_period_us"))
        values = {}
        for name in names:
            path = Path(directory) / name
            if path.is_file():
                values[name] = path.read_text(encoding="ascii").strip()
            else:
                missing.append(str(path))
        evidence[f"{kind}:{directory}"] = values
        if kind in {"v2", "memory"}:
            limit_name, current_name = names[:2]
            if limit_name in values and values[limit_name] != "max" and current_name not in values:
                raise RuntimeError(f"Incomplete cgroup memory accounting: {directory}")
            if limit_name in values and values[limit_name] != "max":
                limit, current = int(values[limit_name]), int(values[current_name])
                # v1 uses a near-LONG_MAX sentinel for an unlimited group.
                if limit < 0 or current < 0:
                    raise ValueError(f"Invalid cgroup memory accounting: {directory}")
                if kind != "memory" or limit < (1 << 60):
                    available = min(available, max(0, limit - current))
        if kind == "v2" and "cpu.max" in values:
            quota, period = values["cpu.max"].split()
            if int(period) <= 0:
                raise ValueError("cgroup CPU period must be positive")
            if quota != "max":
                if int(quota) <= 0:
                    raise ValueError("cgroup CPU quota must be positive")
                cpu_limit = min(cpu_limit, int(quota) / int(period))
        elif kind == "cpu":
            if bool(values) and len(values) != 2:
                raise RuntimeError(f"Incomplete cgroup CPU accounting: {directory}")
            if values:
                quota, period = int(values[names[0]]), int(values[names[1]])
                if period <= 0 or quota == 0 or quota < -1:
                    raise ValueError("Invalid cgroup-v1 CPU quota/period")
                if quota > 0:
                    cpu_limit = min(cpu_limit, quota / period)
    return available, cpu_limit, {
        "locations_resolved": bool(locations), "values": evidence, "missing_files": missing,
        "limitation": "Host availability alone is not a scheduler/allocation guarantee; unenforced or inaccessible limits may remain unknown"}


@lru_cache(maxsize=1)
def _process_for_pid(pid):
    import psutil
    return psutil.Process(pid)


def snapshot():
    import psutil
    process = _process_for_pid(os.getpid())
    memory = psutil.virtual_memory()
    affinity = len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else os.cpu_count()
    if affinity is None or affinity < 1:
        raise RuntimeError("Cannot determine a usable CPU affinity/count for preparation")
    available = int(memory.available)
    cpu_limit = float(affinity)
    available, cpu_limit, allocation = _allocation_limits(available, cpu_limit)
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        if os.environ.get(name):
            requested = int(os.environ[name])
            if requested <= 0:
                raise ValueError(f"{name} must be positive")
            cpu_limit = min(cpu_limit, requested)
    times = process.cpu_times()
    return {"rss_bytes": int(process.memory_info().rss),
            "available_memory_bytes": available, "host_total_memory_bytes": int(memory.total),
            "cpu_capacity": max(1, math.floor(cpu_limit)), "cpu_affinity_cores": affinity,
            "cpu_allocation_cores": cpu_limit,
            "process_cpu_seconds": float(times.user + times.system),
            "cgroup": allocation}


class Measurement(AbstractContextManager):
    """Sample process RSS during full-size work, including failed work."""
    def __init__(self):
        self.stop = threading.Event()
        self.sampling_error = None
        self.report = {"status": "not_started"}

    def __enter__(self):
        try:
            self.before = snapshot()
        except Exception as error:
            self.report = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                           "measurement_error": "initial resource snapshot failed"}
            raise
        self.peak_rss = self.before["rss_bytes"]
        self.minimum_available = self.before["available_memory_bytes"]
        self.started = time.perf_counter()
        self.monitor_samples = 0
        self.monitor_cpu_seconds = 0.
        self.monitor_wall_seconds = 0.

        def monitor():
            try:
                while not self.stop.wait(0.1):
                    cpu_started, wall_started = time.thread_time(), time.perf_counter()
                    state = snapshot()
                    self.monitor_cpu_seconds += time.thread_time() - cpu_started
                    self.monitor_wall_seconds += time.perf_counter() - wall_started
                    self.monitor_samples += 1
                    self.peak_rss = max(self.peak_rss, state["rss_bytes"])
                    self.minimum_available = min(self.minimum_available, state["available_memory_bytes"])
            except Exception as error:
                self.sampling_error = error

        self.thread = threading.Thread(target=monitor, name="prepare-resource-monitor", daemon=True)
        try:
            self.thread.start()
        except Exception as error:
            self.report = {"status": "failed", "before": self.before,
                           "error": f"{type(error).__name__}: {error}",
                           "measurement_error": "resource monitor could not start"}
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()
        after_error = None
        try:
            after = snapshot()
        except Exception as error:
            after, after_error = None, error
        if after is not None:
            self.peak_rss = max(self.peak_rss, after["rss_bytes"])
            self.minimum_available = min(self.minimum_available, after["available_memory_bytes"])
        elapsed = time.perf_counter() - self.started
        self.report = {"elapsed_seconds": elapsed, "before": self.before, "after": after,
                       "sampled_peak_rss_bytes": self.peak_rss,
                       "minimum_available_memory_bytes": self.minimum_available,
                       "average_cpu_cores": None if after is None else (after["process_cpu_seconds"] - self.before["process_cpu_seconds"]) / max(elapsed, 1e-12),
                       "rss_sampling_interval_seconds": 0.1,
                       "monitor_samples": self.monitor_samples,
                       "monitor_cpu_seconds": self.monitor_cpu_seconds,
                       "monitor_wall_seconds": self.monitor_wall_seconds,
                       "status": "failed" if exc is not None or self.sampling_error is not None or after_error is not None else "complete",
                       "error": None if exc is None else f"{type(exc).__name__}: {exc}",
                       "sampling_error": None if self.sampling_error is None else str(self.sampling_error),
                       "final_snapshot_error": None if after_error is None else str(after_error)}
        if exc is None and (self.sampling_error is not None or after_error is not None):
            raise RuntimeError("Preparation resource measurement failed") from (self.sampling_error or after_error)
        return False


def _exclusive_report(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = target.with_name(f"{target.stem}.{uuid.uuid4().hex}{target.suffix or '.json'}")
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def run_case_jobs(*, tasks, function, commit, workers, report_path):
    """Execute every task exactly once; measure increasing safe concurrency.

    ``auto`` starts with one *calibration* case, then tests powers of two up to
    CPU allocation, pending work and measured memory headroom. It selects the
    best observed throughput after a decrease. No case is skipped or truncated.
    Explicit integer worker counts are respected and still measured.
    """
    tasks = list(tasks)
    if workers != "auto" and (type(workers) is not int or workers < 1):
        raise ValueError("Preparation workers must be 'auto' or a positive integer")
    automatic = workers == "auto"
    offset, best_width, best_rate = 0, 0, -1.0
    incremental_peak = 0
    calibrating = automatic
    waves = []
    report = {"format": "hiercp_preparation_measurement_v1", "workers": workers,
              "configured_tasks": len(tasks), "completed_tasks": 0, "attempted_tasks": 0,
              "failed_tasks": 0, "submission_failures": 0, "waves": waves,
              "selection_basis": "real full-size pending-case throughput and sampled RSS",
              "comparison_limitation": "waves contain different cases; not an identical-input microbenchmark or a proof of optimality",
              "completion_semantics": "executor tasks returned and commit succeeded; domain/cache validity is checked separately by the caller",
              "memory_admission_estimate_only": True,
              "provided_measured_case_peak_rss_bytes": None,
              "provided_peak_semantics": "optional isolated whole-process pilot RSS, used as a conservative per-case admission floor",
              "memory_admission_basis": "conservative observed whole-wave RSS increments; future case shapes and sub-interval peaks may require more",
              "cohort_reduced": False, "graph_reduced": False, "resources_before": None,
              "status": "running"}
    primary_error = None
    try:
        provided = os.environ.get("HIERCP_PREPARE_MEASURED_CASE_RSS_BYTES")
        if provided is not None:
            try:
                measured_case_peak = int(provided)
            except ValueError as error:
                raise ValueError("HIERCP_PREPARE_MEASURED_CASE_RSS_BYTES must be a positive integer") from error
            if measured_case_peak <= 0:
                raise ValueError("HIERCP_PREPARE_MEASURED_CASE_RSS_BYTES must be a positive integer")
            incremental_peak = measured_case_peak
            report["provided_measured_case_peak_rss_bytes"] = measured_case_peak
        state = snapshot()
        report["resources_before"] = state
        maximum = min(len(tasks), state["cpu_capacity"] if automatic else workers)
        width = 1 if automatic else max(1, maximum)
        while offset < len(tasks):
            current = snapshot()
            if current["available_memory_bytes"] <= 0:
                raise MemoryError("No remaining allocation memory; preparation work was not started")
            if incremental_peak > 0:
                # Reserve at least 20% of current availability. This controls
                # concurrency, not the size of any case, graph or tensor.
                safe = math.floor(0.8 * current["available_memory_bytes"] / incremental_peak)
                if safe < 1:
                    raise MemoryError("Measured full-size case memory no longer fits available headroom; no graph was reduced")
                if automatic:
                    width = min(width, safe)
                elif min(width, len(tasks) - offset) > safe:
                    raise MemoryError("Explicit preparation worker count exceeds measured memory headroom; no worker/model/graph setting was silently changed")
            batch = tasks[offset:offset + min(width, len(tasks) - offset)]
            error = None
            measurement = Measurement()
            with measurement:
                with ThreadPoolExecutor(max_workers=len(batch), thread_name_prefix="hiercp-prepare") as executor:
                    futures = {}
                    for task in batch:
                        try:
                            future = executor.submit(function, task)
                        except Exception as exc:
                            report["submission_failures"] += 1
                            error = exc
                            break
                        futures[future] = task
                        report["attempted_tasks"] += 1
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            commit(result)
                            report["completed_tasks"] += 1
                        except Exception as exc:
                            report["failed_tasks"] += 1
                            # Retain other completed outputs, then propagate the
                            # first failure. Never mark the cohort complete.
                            if error is None:
                                error = exc
                    if error is not None:
                        raise error
            elapsed = measurement.report["elapsed_seconds"]
            rate = len(batch) / max(elapsed, 1e-12)
            wave = {**measurement.report, "workers": len(batch), "tasks": len(batch),
                    "cases_per_second": rate, "calibration_wave": calibrating}
            waves.append(wave)
            delta = max(0, measurement.peak_rss - measurement.before["rss_bytes"])
            # Dividing a wave's peak by its worker count estimates an average,
            # not a safe per-case high-water mark when case sizes vary.
            incremental_peak = max(incremental_peak, delta)
            print("[PrepareWave] " + json.dumps(wave, allow_nan=False), flush=True)
            offset += len(batch)
            if calibrating:
                if rate > best_rate:
                    best_width, best_rate = len(batch), rate
                    width = min(maximum, width * 2)
                    calibrating = width > len(batch)
                else:
                    width, calibrating = best_width, False
        report["status"] = "complete"
        report["selected_workers"] = best_width if automatic else (workers if tasks else 0)
        report["selection_semantics"] = "best observed throughput preference, subject to the per-wave memory admission clamp"
        report["applied_worker_counts"] = [wave["workers"] for wave in waves]
    except Exception as exc:
        primary_error = exc
        if "measurement" in locals() and hasattr(measurement, "report") and measurement.report["status"] == "failed":
            waves.append({**measurement.report, "workers": len(batch), "tasks": len(batch)})
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["pending_tasks"] = len(tasks) - report["attempted_tasks"]
        final_error = None
        try:
            report["resources_after"] = snapshot()
        except Exception as error:
            final_error = error
            report.update(status="failed", resources_after=None,
                          final_snapshot_error=f"{type(error).__name__}: {error}")
        try:
            target = _exclusive_report(report_path, report)
        except Exception as error:
            if primary_error is not None:
                raise RuntimeError(f"Preparation failed; resource report could not be saved: {error}") from primary_error
            raise
        print(f"[PreparationResources] {target}", flush=True)
        if final_error is not None and primary_error is None:
            raise RuntimeError("Preparation final resource snapshot failed; failed report was retained") from final_error
    return report
