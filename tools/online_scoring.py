"""Measured, ordered physical batches for online candidate-bank scoring."""
from __future__ import annotations

import os
import time
import math
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hiercp.preparation_runtime import Measurement, snapshot
from hiercp.training_resources import _local_bound, _tensor_bytes
from hiercp.tensor import torch_load_compat


SCORING_FORMAT = "hiercp_online_scoring_v1"
BANK_HOST_FORMAT = "hiercp_bank_scoring_host_v1"


def _storage_bytes(value: Any) -> int:
    """Count actual Torch backing stores once, including larger shared views."""
    import torch
    stores: dict[tuple[Any, ...], int] = {}

    def visit(item):
        if torch.is_tensor(item):
            storage = item.untyped_storage()
            stores[(str(item.device), storage.data_ptr(), storage.nbytes())] = storage.nbytes()
        elif isinstance(item, np.ndarray):
            stores[("numpy", id(item))] = item.nbytes
        elif isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
        elif hasattr(item, "to_dict"):
            visit(item.to_dict())

    visit(value)
    return sum(stores.values())


def _sample_inventory(sample: Mapping[str, Any]) -> dict[str, Any]:
    if "source_local" in sample and "target_locals" in sample:
        source = _local_bound(sample["source_local"])
        targets = [_local_bound(target) for target in sample["target_locals"]]
        if not targets:
            raise ValueError("Bank input has no canonical target graphs")
        bound = 2 * (len(targets) * source[2] + sum(row[2] for row in targets))
        bound += sum(_tensor_bytes(value) for key, value in sample.items()
                     if key not in {"source_local", "target_locals"})
    else:
        # Compatibility for explicitly materialized callers. Canonical bank
        # builders use the conservative complete-topology formula above.
        graphs = [*sample.get("local_graphs", ()), *sample.get("local_graphs_view2", ())]
        bound = _tensor_bytes(sample) + 512 * len(graphs)
    if bound <= 0:
        raise ValueError("Bank input has no measurable tensor storage")
    return {"case_id": str(sample["case_id"]), "canonical_bytes": _storage_bytes(sample),
            "input_bytes_upper_bound": int(bound)}


def _bank_host_budget(rows, *, pin_memory: bool, load_canonical: bool) -> dict[str, Any]:
    """Bound additional live input copies against fresh host AND cgroup space.

    Three view-sized copies cover materialized views, collated batches and the
    chunked model's graph rebatching workspace; pinning can add a fourth copy.
    Disk-backed canonical tensors can fault in concurrently and are charged
    separately. Already-resident direct inputs are not charged a second time.
    This is not an activation/Python-memory guarantee; RSS is also measured.
    """
    state = snapshot()
    bound = sum(row["input_bytes_upper_bound"] for row in rows)
    canonical = sum(row["canonical_bytes"] for row in rows) if load_canonical else 0
    copies = 3 + int(pin_memory)
    required = canonical + copies * bound
    available = int(state["available_memory_bytes"])
    return {"format": BANK_HOST_FORMAT, "source_count": len(rows),
            "input_bytes_upper_bound": bound, "canonical_load_bytes": canonical,
            "input_copies_accounted": copies, "pin_memory": bool(pin_memory),
            "estimated_input_bytes": required, "available_allocation_bytes": available,
            "permitted_input_bytes": available * 4 // 5, "reserved_headroom_fraction": 0.2,
            "accepted": required <= available * 4 // 5, "allocation": state,
            "graph_or_data_reduction": False, "is_input_bound_not_total_memory_guarantee": True}


def _validate_host_budget(budget):
    if not isinstance(budget, dict) or budget.get("format") != BANK_HOST_FORMAT:
        raise ValueError("Bank scoring trial lacks its host allocation evidence")
    for key in ("source_count", "input_bytes_upper_bound", "canonical_load_bytes",
                "estimated_input_bytes", "available_allocation_bytes", "permitted_input_bytes"):
        value = budget.get(key)
        if type(value) is not int or value < (1 if key in {"source_count", "input_bytes_upper_bound"} else 0):
            raise ValueError(f"Bank scoring host budget has invalid {key}")
    if type(budget.get("pin_memory")) is not bool:
        raise ValueError("Bank scoring host budget pinning is invalid")
    expected = budget["canonical_load_bytes"] + (3 + int(budget["pin_memory"])) * budget["input_bytes_upper_bound"]
    permitted = budget["available_allocation_bytes"] * 4 // 5
    if (budget.get("input_copies_accounted") != 3 + int(budget["pin_memory"])
            or budget["estimated_input_bytes"] != expected or budget["permitted_input_bytes"] != permitted
            or budget.get("accepted") is not (expected <= permitted)
            or budget.get("reserved_headroom_fraction") != 0.2
            or budget.get("graph_or_data_reduction") is not False
            or budget.get("is_input_bound_not_total_memory_guarantee") is not True
            or not isinstance(budget.get("allocation"), dict)
            or budget["allocation"].get("available_memory_bytes") != budget["available_allocation_bytes"]):
        raise ValueError("Bank scoring host budget contradicts its input/allocation arithmetic")


class BankHostBudgetError(RuntimeError):
    def __init__(self, budget):
        self.budget = budget
        super().__init__(f"Bank physical batch exceeds current host/cgroup input budget: {budget}; "
                         "no graph, candidate, model or selected batch reduction was applied")


def validate_scoring_report(report: Any, config: Mapping[str, Any]) -> None:
    if not isinstance(report, dict) or report.get("format") != SCORING_FORMAT:
        raise ValueError("Bank has no measured scoring report; rebuild with --overwrite")
    host_report = any(key in report for key in ("host_contract_format", "spool_io", "candidate_count", "calibration_sample_count"))
    if host_report:
        if report.get("host_contract_format") != BANK_HOST_FORMAT:
            raise ValueError("Bank scoring host contract is missing or unsupported")
        for key in ("candidate_count", "calibration_sample_count", "scored_samples"):
            if type(report.get(key)) is not int or report[key] < 1:
                raise ValueError(f"Bank scoring report has invalid {key}")
        if report.get("scored_candidates") != report["scored_samples"] * report["candidate_count"]:
            raise ValueError("Bank scoring candidate count arithmetic is inconsistent")
    selected = report.get("physical_batch_size")
    candidates = config["scoring_batch_size_candidates"]
    if not isinstance(selected, int) or isinstance(selected, bool) or selected not in candidates:
        raise ValueError("Bank scoring report has no valid measured physical batch")
    trials = report.get("calibration_trials")
    if not isinstance(trials, list) or len(trials) != len(candidates):
        raise ValueError("Bank scoring calibration does not cover the configured candidates")
    if sorted(trial.get("batch_size", -1) for trial in trials) != sorted(candidates):
        raise ValueError("Bank scoring calibration candidate identities differ")
    safe = []
    for trial in trials:
        if host_report:
            if report["host_contract_format"] != BANK_HOST_FORMAT:
                raise ValueError("Bank scoring host contract is unsupported")
            if trial.get("status") not in {"safe", "cuda_oom", "vram_headroom_exceeded",
                                           "unavailable_cohort_smaller_than_batch", "host_input_budget_exceeded"}:
                raise ValueError("Bank scoring trial has an unsupported resource status")
            budgets = trial.get("host_budgets")
            if trial.get("status") != "unavailable_cohort_smaller_than_batch":
                if not isinstance(budgets, list) or not budgets:
                    raise ValueError("Bank scoring trial has no host budgets")
                for budget in budgets:
                    _validate_host_budget(budget)
                if trial["status"] == "host_input_budget_exceeded":
                    if all(budget["accepted"] for budget in budgets):
                        raise ValueError("Bank host-rejected trial has no measured rejection")
                elif not all(budget["accepted"] for budget in budgets):
                    raise ValueError("Bank scoring trial ran with a rejected host allocation")
        if trial.get("status") == "safe":
            seconds = trial.get("seconds")
            throughput = trial.get("samples_per_second")
            if (not isinstance(seconds, list) or len(seconds) != config["scoring_batch_calibration_repeats"]
                    or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in seconds)
                    or not isinstance(throughput, (int, float)) or not math.isfinite(throughput) or throughput <= 0):
                raise ValueError("Bank scoring calibration timing is invalid")
            if host_report:
                expected_rate = report["calibration_sample_count"] / float(np.median(seconds))
                if not math.isclose(float(throughput), expected_rate, rel_tol=1e-12, abs_tol=0.0):
                    raise ValueError("Bank scoring throughput contradicts its measured cohort/timing")
                measurements = trial.get("host_measurements")
                if (not isinstance(measurements, list) or len(measurements) != config["scoring_batch_calibration_repeats"]
                        or any(not isinstance(row, dict) or row.get("status") != "complete" for row in measurements)):
                    raise ValueError("Bank scoring safe trial lacks completed host measurements")
            safe.append((throughput, trial["batch_size"]))
    if not safe or max(safe)[1] != selected:
        raise ValueError("Bank scoring batch does not match measured safe throughput selection")
    batches = report.get("actual_batches")
    if (not isinstance(batches, list) or not batches
            or any(not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= selected for value in batches)
            or sum(batches) != report.get("scored_samples")):
        raise ValueError("Bank scoring execution sample count is incomplete")


class PendingBankScorer:
    def __init__(self, model: Any, device: Any, config: Mapping[str, Any], candidate_count: int,
                 *, spool_directory: Any = None, progress: Callable[..., None] | None = None):
        import torch
        from hiercp.data import collate_samples

        candidates = config.get("scoring_batch_size_candidates")
        if config.get("scoring_batch_size") != "auto" or not isinstance(candidates, list):
            raise ValueError("Bank scoring requires configured automatic physical-batch measurement")
        if (
            len(candidates) < 2
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in candidates)
            or len(set(candidates)) != len(candidates)
            or max(candidates) <= 1
        ):
            raise ValueError("Scoring batch candidates must be distinct positive integers including a batch > 1")
        self.candidates = sorted(candidates)
        self.repeats = config.get("scoring_batch_calibration_repeats")
        if not isinstance(self.repeats, int) or isinstance(self.repeats, bool) or self.repeats < 3:
            raise ValueError("Scoring calibration requires at least three measured repetitions")
        self.memory_fraction = float(config["scoring_batch_max_vram_fraction"])
        if not 0.0 < self.memory_fraction < 1.0:
            raise ValueError("Scoring VRAM fraction must be in (0, 1)")
        self.chunk_size = int(config["local_candidate_chunk_size"])
        if self.chunk_size < 1 or candidate_count < 1:
            raise ValueError("Scoring chunk size and candidate count must be positive")
        self.torch, self.collate = torch, collate_samples
        self.model, self.device = model, device
        self.amp = bool(config.get("amp", True) and device.type == "cuda")
        self.pin_memory = bool(config.get("pin_memory", True) and device.type == "cuda")
        self.candidate_count = int(candidate_count)
        self.pending: list[tuple[tuple[Any, ...], Callable[[list[np.ndarray]], None]]] = []
        self.progress = progress
        self._spool = None
        if spool_directory is not None:
            parent = Path(spool_directory).resolve()
            if not parent.is_dir():
                raise ValueError(f"Bank scoring spool parent must already exist: {parent}")
            self._spool = tempfile.TemporaryDirectory(prefix=".bank-scoring-", dir=parent)
        self.spool_root = Path(self._spool.name) if self._spool is not None else None
        self._serial = 0
        self._closed = False
        self.spool_io = {"written_bytes": 0, "write_seconds": 0.0, "read_seconds": 0.0,
                         "sample_loads": 0, "mode": "disk_mmap" if self._spool else "direct_memory",
                         "read_timing_scope": "mmap deserialization/setup; page faults are included in scoring time"}
        self.calibration_sample_count = 0
        self.selected_batch_size: int | None = None
        self.trials: list[dict[str, Any]] = []
        self.scored_samples = 0
        self.scoring_seconds = 0.0
        self.peak_vram_bytes = 0
        self.actual_batches: list[int] = []
        self.resources: dict[str, Any] = {"cpu_cores": os.cpu_count(), "device": str(device),
                                          "initial_allocation": snapshot()}
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            free, total = torch.cuda.mem_get_info(device)
            self.resources.update(gpu=props.name, visible_gpu_count=torch.cuda.device_count(),
                                  vram_bytes=int(total), free_vram_bytes=int(free))
        print(f"[BankScoringResources] {self.resources}", flush=True)

    def _emit(self, event: str, **fields) -> None:
        payload = {"stage": "bank_scoring", "event": event, **fields, "allocation_snapshot": snapshot()}
        if self.progress is not None:
            self.progress(**payload)
        else:
            print("[BankScoringProgress] " + json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)

    def close(self) -> None:
        if self._closed:
            return
        # TemporaryDirectory owns a unique new child, never a caller directory.
        if self._spool is not None:
            self._spool.cleanup()
        self.pending.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def submit(self, samples: Sequence[Any], publish: Callable[[list[np.ndarray]], None]) -> None:
        if self._closed:
            raise RuntimeError("Bank scorer is closed")
        if not samples:
            raise ValueError("Cannot queue an empty bank-scoring group")
        queued = []
        for sample in samples:
            row = _sample_inventory(sample)
            if self.spool_root is None:
                queued.append({"sample": sample, "inventory": row})
                continue
            self._serial += 1
            path = self.spool_root / f"sample_{self._serial:08d}.pt"
            available_disk = shutil.disk_usage(self.spool_root).free
            # Tensor backing bytes are a lower bound, not a promise that all
            # pickle metadata fits. ENOSPC still propagates with no fallback.
            if available_disk < row["canonical_bytes"]:
                raise OSError(f"Insufficient bank spool disk space: free={available_disk}, tensor_bytes={row['canonical_bytes']}")
            self._emit("spool_write_start", case_id=row["case_id"], canonical_bytes=row["canonical_bytes"])
            started = time.perf_counter()
            with path.open("xb") as handle:
                self.torch.save(sample, handle)
                handle.flush()
                os.fsync(handle.fileno())
            elapsed = time.perf_counter() - started
            self.spool_io["written_bytes"] += path.stat().st_size
            self.spool_io["write_seconds"] += elapsed
            queued.append({"path": path, "inventory": row})
            self._emit("spool_write_end", case_id=row["case_id"], elapsed_seconds=elapsed,
                       file_bytes=path.stat().st_size, queued_sources=sum(len(group) for group, _ in self.pending) + len(queued))
        self.pending.append((tuple(queued), publish))

    def flush_ready(self) -> None:
        target = self.selected_batch_size if self.selected_batch_size is not None else max(self.candidates)
        if sum(len(samples) for samples, _ in self.pending) >= target:
            self.flush()

    def _infer(self, samples: Sequence[Any]) -> list[np.ndarray]:
        case_ids = tuple(str(sample["case_id"]) for sample in samples)
        # Bank builders submit canonical samples; collate materializes their
        # inference views in place below. Validate counts before that operation
        # without requiring a field that only exists after materialization.
        counts = tuple(len(sample["local_graphs"] if "local_graphs" in sample
                           else sample["target_locals"]) for sample in samples)
        if any(not value.strip() for value in case_ids) or any(count != self.candidate_count for count in counts):
            raise ValueError("Bank scoring input has an invalid case identity or candidate count")
        # materialize_sample_views replaces canonical keys in place. Independent
        # root containers prevent the pending samples becoming expanded graphs
        # and keep every calibration candidate's materialization work equal.
        batch = self.collate([dict(sample) for sample in samples])
        if tuple(batch.case_ids) != case_ids or tuple(batch.counts) != counts or batch.sample_count != len(samples):
            raise RuntimeError("Bank collate changed case order or per-source/pool candidate counts")
        if self.pin_memory:
            batch.pin_memory()
        with self.torch.inference_mode(), self.torch.autocast(device_type=self.device.type, enabled=self.amp):
            tensors = self.model.score_inference_chunked(batch, local_chunk_size=self.chunk_size)
        if len(tensors) != len(samples):
            raise RuntimeError("Bank scoring lost sample/component/pool order or cardinality")
        result = [value.detach().float().cpu().numpy().astype(np.float32) for value in tensors]
        if any(value.shape != (self.candidate_count,) or not np.all(np.isfinite(value)) for value in result):
            raise RuntimeError("Bank scoring returned invalid per-pool candidate scores")
        return result

    def _run(self, samples: Sequence[Any], batch_size: int) -> tuple[list[np.ndarray], float, int]:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
            self.torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        output: list[np.ndarray] = []
        for start in range(0, len(samples), batch_size):
            group = samples[start:start + batch_size]
            budget = _bank_host_budget([entry["inventory"] for entry in group],
                pin_memory=self.pin_memory, load_canonical=self.spool_root is not None)
            self._emit("score_batch_start", batch_size=len(group), host_budget=budget)
            if not budget["accepted"]:
                raise BankHostBudgetError(budget)
            loaded = []
            try:
                started_loading = time.perf_counter()
                for entry in group:
                    loaded.append(torch_load_compat(entry["path"], map_location="cpu", mmap=True)
                                  if "path" in entry else entry["sample"])
                self.spool_io["read_seconds"] += time.perf_counter() - started_loading
                self.spool_io["sample_loads"] += sum("path" in entry for entry in group)
                output.extend(self._infer(loaded))
            finally:
                loaded.clear()
                del loaded, group
            self._emit("score_batch_end", batch_size=min(batch_size, len(samples) - start))
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
            peak = int(self.torch.cuda.max_memory_allocated(self.device))
        else:
            peak = 0
        return output, time.perf_counter() - started, peak

    def _score(self, samples: Sequence[Any]) -> list[np.ndarray]:
        if self.selected_batch_size is None:
            self.calibration_sample_count = len(samples)
            best: tuple[float, int, list[np.ndarray], float] | None = None
            for size in self.candidates:
                if size > len(samples):
                    self.trials.append({"batch_size": size, "status": "unavailable_cohort_smaller_than_batch"})
                    continue
                seconds: list[float] = []
                peaks: list[int] = []
                budgets = [_bank_host_budget([entry["inventory"] for entry in samples[start:start + size]],
                    pin_memory=self.pin_memory, load_canonical=self.spool_root is not None)
                    for start in range(0, len(samples), size)]
                self._emit("calibration_trial_start", batch_size=size, repeats=self.repeats, host_budgets=budgets)
                if any(not budget["accepted"] for budget in budgets):
                    trial = {"batch_size": size, "status": "host_input_budget_exceeded", "host_budgets": budgets}
                    self.trials.append(trial)
                    self._emit("calibration_trial_end", **trial)
                    continue
                measurements = []
                try:
                    for repeat in range(self.repeats):
                        self._emit("calibration_repeat_start", batch_size=size, repeat=repeat)
                        with Measurement() as measured:
                            output, elapsed, peak = self._run(samples, size)
                        measurements.append(measured.report)
                        seconds.append(elapsed)
                        peaks.append(peak)
                except BankHostBudgetError as exc:
                    trial = {"batch_size": size, "status": "host_input_budget_exceeded",
                             "host_budgets": [*budgets, exc.budget], "completed_repeats": len(seconds)}
                    self.trials.append(trial)
                    self._emit("calibration_trial_end", **trial)
                    continue
                except self.torch.cuda.OutOfMemoryError:
                    trial = {"batch_size": size, "status": "cuda_oom", "host_budgets": budgets,
                             "completed_repeats": len(seconds)}
                    self.trials.append(trial)
                    self._emit("calibration_trial_end", **trial)
                    self.torch.cuda.empty_cache()
                    continue
                self.peak_vram_bytes = max(self.peak_vram_bytes, max(peaks))
                safe = self.device.type != "cuda" or max(peaks) <= self.memory_fraction * self.resources["vram_bytes"]
                median_seconds = float(np.median(seconds))
                throughput = len(samples) / median_seconds
                self.trials.append({"batch_size": size, "status": "safe" if safe else "vram_headroom_exceeded",
                                    "seconds": seconds, "peak_vram_bytes": max(peaks), "samples_per_second": throughput,
                                    "host_budgets": budgets, "host_measurements": measurements})
                self._emit("calibration_trial_end", **self.trials[-1])
                if safe and (best is None or (throughput, size) > (best[0], best[1])):
                    best = (throughput, size, output, median_seconds)
            if best is None:
                raise RuntimeError(f"No measured safe scoring batch; no model/graph reduction was applied: {self.trials}")
            _, self.selected_batch_size, output, elapsed = best
            print(f"[BankScoringCalibration] physical_batch_size={self.selected_batch_size} trials={self.trials}", flush=True)
        else:
            with Measurement() as measured:
                output, elapsed, peak = self._run(samples, self.selected_batch_size)
            self._emit("scoring_measurement", measurement=measured.report)
            self.peak_vram_bytes = max(self.peak_vram_bytes, peak)
        if len(output) != len(samples):
            raise RuntimeError("Scoring batch output count differs from the queued sample count")
        self.scored_samples += len(samples)
        self.scoring_seconds += elapsed
        self.actual_batches.extend(min(self.selected_batch_size, len(samples) - start)
                                   for start in range(0, len(samples), self.selected_batch_size))
        return output

    def flush(self) -> None:
        if not self.pending:
            return
        self._emit("flush_start", queued_sources=sum(len(group) for group, _ in self.pending),
                   selected_batch_size=self.selected_batch_size)
        groups = self.pending
        samples = [sample for group, _ in groups for sample in group]
        output = self._score(samples)
        self.pending = []
        offset = 0
        for group, publish in groups:
            stop = offset + len(group)
            publish(output[offset:stop])
            offset = stop
        if offset != len(output):
            raise RuntimeError("Scoring publication lost source/pool alignment")
        for group, _ in groups:
            for entry in group:
                if "path" in entry:
                    entry["path"].unlink()
        self._emit("flush_end", scored_samples=len(samples), selected_batch_size=self.selected_batch_size)

    def report(self) -> dict[str, Any]:
        if self.pending:
            raise RuntimeError("Cannot report completed scoring while samples are pending")
        self.resources["final_allocation"] = snapshot()
        self.close()
        return {"format": SCORING_FORMAT, "host_contract_format": BANK_HOST_FORMAT,
                "physical_batch_size": self.selected_batch_size, "actual_batches": self.actual_batches,
                "candidate_count": self.candidate_count, "calibration_sample_count": self.calibration_sample_count,
                "calibration_trials": self.trials, "resources": self.resources,
                "spool_io": dict(self.spool_io),
                "scored_samples": self.scored_samples, "scored_candidates": self.scored_samples * self.candidate_count,
                "scoring_seconds": self.scoring_seconds, "peak_vram_bytes": self.peak_vram_bytes,
                "samples_per_second": self.scored_samples / self.scoring_seconds if self.scoring_seconds else None}
