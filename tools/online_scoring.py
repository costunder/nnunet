"""Measured, ordered physical batches for online candidate-bank scoring."""
from __future__ import annotations

import os
import time
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCORING_FORMAT = "hiercp_online_scoring_v1"


def validate_scoring_report(report: Any, config: Mapping[str, Any]) -> None:
    if not isinstance(report, dict) or report.get("format") != SCORING_FORMAT:
        raise ValueError("Bank has no measured scoring report; rebuild with --overwrite")
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
        if trial.get("status") == "safe":
            seconds = trial.get("seconds")
            throughput = trial.get("samples_per_second")
            if (not isinstance(seconds, list) or len(seconds) != config["scoring_batch_calibration_repeats"]
                    or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in seconds)
                    or not isinstance(throughput, (int, float)) or not math.isfinite(throughput) or throughput <= 0):
                raise ValueError("Bank scoring calibration timing is invalid")
            safe.append((throughput, trial["batch_size"]))
    if not safe or max(safe)[1] != selected:
        raise ValueError("Bank scoring batch does not match measured safe throughput selection")
    batches = report.get("actual_batches")
    if (not isinstance(batches, list) or not batches
            or any(not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= selected for value in batches)
            or sum(batches) != report.get("scored_samples")):
        raise ValueError("Bank scoring execution sample count is incomplete")


class PendingBankScorer:
    def __init__(self, model: Any, device: Any, config: Mapping[str, Any], candidate_count: int):
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
        self.selected_batch_size: int | None = None
        self.trials: list[dict[str, Any]] = []
        self.scored_samples = 0
        self.scoring_seconds = 0.0
        self.peak_vram_bytes = 0
        self.actual_batches: list[int] = []
        self.resources: dict[str, Any] = {"cpu_cores": os.cpu_count(), "device": str(device)}
        try:
            import psutil
        except ImportError:
            self.psutil = None
            self.resources.update(cpu_utilization_percent=None, ram_total_bytes=None,
                                  ram_available_bytes=None, process_rss_bytes=None,
                                  host_telemetry_status="unavailable: psutil is not installed")
        else:
            self.psutil = psutil
            memory = psutil.virtual_memory()
            self.resources.update(cpu_utilization_percent=psutil.cpu_percent(interval=0.1),
                                  ram_total_bytes=int(memory.total), ram_available_bytes=int(memory.available),
                                  process_rss_bytes=int(psutil.Process().memory_info().rss),
                                  host_telemetry_status="measured")
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            free, total = torch.cuda.mem_get_info(device)
            self.resources.update(gpu=props.name, visible_gpu_count=torch.cuda.device_count(),
                                  vram_bytes=int(total), free_vram_bytes=int(free))
        print(f"[BankScoringResources] {self.resources}", flush=True)

    def submit(self, samples: Sequence[Any], publish: Callable[[list[np.ndarray]], None]) -> None:
        if not samples:
            raise ValueError("Cannot queue an empty bank-scoring group")
        self.pending.append((tuple(samples), publish))

    def flush_ready(self) -> None:
        if sum(len(samples) for samples, _ in self.pending) >= max(self.candidates):
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
        batch = self.collate(list(samples))
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
            output.extend(self._infer(samples[start:start + batch_size]))
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
            peak = int(self.torch.cuda.max_memory_allocated(self.device))
        else:
            peak = 0
        return output, time.perf_counter() - started, peak

    def _score(self, samples: Sequence[Any]) -> list[np.ndarray]:
        if self.selected_batch_size is None:
            best: tuple[float, int, list[np.ndarray], float] | None = None
            for size in self.candidates:
                if size > len(samples):
                    self.trials.append({"batch_size": size, "status": "unavailable_cohort_smaller_than_batch"})
                    continue
                seconds: list[float] = []
                peaks: list[int] = []
                try:
                    for _ in range(self.repeats):
                        output, elapsed, peak = self._run(samples, size)
                        seconds.append(elapsed)
                        peaks.append(peak)
                except self.torch.cuda.OutOfMemoryError:
                    self.trials.append({"batch_size": size, "status": "cuda_oom"})
                    self.torch.cuda.empty_cache()
                    continue
                self.peak_vram_bytes = max(self.peak_vram_bytes, max(peaks))
                safe = self.device.type != "cuda" or max(peaks) <= self.memory_fraction * self.resources["vram_bytes"]
                median_seconds = float(np.median(seconds))
                throughput = len(samples) / median_seconds
                self.trials.append({"batch_size": size, "status": "safe" if safe else "vram_headroom_exceeded",
                                    "seconds": seconds, "peak_vram_bytes": max(peaks), "samples_per_second": throughput})
                if safe and (best is None or (throughput, size) > (best[0], best[1])):
                    best = (throughput, size, output, median_seconds)
            if best is None:
                raise RuntimeError(f"No measured safe scoring batch; no model/graph reduction was applied: {self.trials}")
            _, self.selected_batch_size, output, elapsed = best
            print(f"[BankScoringCalibration] physical_batch_size={self.selected_batch_size} trials={self.trials}", flush=True)
        else:
            output, elapsed, peak = self._run(samples, self.selected_batch_size)
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

    def report(self) -> dict[str, Any]:
        if self.pending:
            raise RuntimeError("Cannot report completed scoring while samples are pending")
        if self.psutil is not None:
            memory = self.psutil.virtual_memory()
            self.resources.update(cpu_utilization_percent=self.psutil.cpu_percent(interval=None),
                                  ram_available_bytes=int(memory.available),
                                  process_rss_bytes=int(self.psutil.Process().memory_info().rss))
        return {"format": SCORING_FORMAT, "physical_batch_size": self.selected_batch_size, "actual_batches": self.actual_batches,
                "calibration_trials": self.trials, "resources": self.resources,
                "scored_samples": self.scored_samples, "scored_candidates": self.scored_samples * self.candidate_count,
                "scoring_seconds": self.scoring_seconds, "peak_vram_bytes": self.peak_vram_bytes,
                "samples_per_second": self.scored_samples / self.scoring_seconds if self.scoring_seconds else None}
