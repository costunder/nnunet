"""Pinned, runtime-only epoch timing for the unchanged native v1 training loop.

The archived pipeline, model, sampler and loss files are never rewritten. Only
``run_train`` receives explicit timer calls in memory, after exact source checks.
Sampling timers add Python metadata to samples/batches; graph/tensor values and
the original materializer/collator calls are preserved. CUDA events are read
after the pipeline's existing pass-end synchronization, never per batch.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import functools
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from types import FunctionType
from typing import Any, Mapping
import uuid


FORMAT = "hiercp_v1_epoch_telemetry_v1"
PIPELINE_SHA256 = "260f4acbef2adf61b5a55e47ae05c0bd2733fc40535d1da45bae137ef7b4dcb5"
RUN_TRAIN_SHA256 = "6bab3f98dce399461922a9f0a631237a1dffa535b34b89acb6652d71d17aebcd"
SAMPLE_TIMING_KEY = "_v1x_epoch_telemetry_sampling_seconds"
BATCH_TIMING_KEY = "v1x_epoch_telemetry_sampling"
_ORIGINAL_MATERIALIZER_KEY = "_v1x_epoch_telemetry_original_materializer"
_GLOBAL_NAME = "_v1x_epoch_telemetry"


class TelemetryError(ValueError):
    """Fail closed when the pinned training or instrumentation contract differs."""


# Every replacement must occur exactly once in the pinned run_train source.
# Existing computational statements keep their text and relative order.
SOURCE_REWRITES = (
    (
        "    # Use dedicated generators so resume restores the next epoch's shuffle\n",
        "    _v1x_epoch_telemetry.prepare_sampling_capture()\n"
        "    # Use dedicated generators so resume restores the next epoch's shuffle\n",
    ),
    (
        "        collate_fn=collate_samples,\n        **loader_options,\n",
        "        collate_fn=_v1x_epoch_telemetry.collate_samples,\n"
        "        worker_init_fn=_v1x_epoch_telemetry.worker_init,\n"
        "        **loader_options,\n",
    ),
    (
        "            collate_fn=collate_samples,\n            **loader_options,\n",
        "            collate_fn=_v1x_epoch_telemetry.collate_samples,\n"
        "            worker_init_fn=_v1x_epoch_telemetry.worker_init,\n"
        "            **loader_options,\n",
    ),
    (
        "        pass_started = time.perf_counter()\n        cpu_started = time.process_time()\n",
        "        pass_started = time.perf_counter()\n"
        "        _v1x_pass = _v1x_epoch_telemetry.new_pass(torch, device, training_mode)\n"
        "        cpu_started = time.process_time()\n",
    ),
    (
        "        for batch_index, batch in enumerate(loader):\n",
        "        for batch_index, batch in enumerate(_v1x_pass.iter_loader(loader)):\n"
        "            _v1x_pass.begin_compute()\n",
    ),
    (
        "            accuracy_sum, reciprocal_rank_sum, sample_count = ranking_metric_sums(\n",
        "            _v1x_pass.end_compute()\n"
        "            accuracy_sum, reciprocal_rank_sum, sample_count = ranking_metric_sums(\n",
    ),
    (
        "            result[\"cuda_peak_allocated_bytes\"] = f\"{UNAVAILABLE} (CPU run)\"\n"
        "        return result\n",
        "            result[\"cuda_peak_allocated_bytes\"] = f\"{UNAVAILABLE} (CPU run)\"\n"
        "        result.update(_v1x_pass.finish(expected_samples=int(count), expected_batches=total_batches))\n"
        "        return result\n",
    ),
    (
        "    for epoch in range(start_epoch, epochs + 1):\n        train_dataset.set_epoch(epoch)\n",
        "    for epoch in range(start_epoch, epochs + 1):\n"
        "        _v1x_epoch_telemetry.begin_epoch(epoch)\n"
        "        train_dataset.set_epoch(epoch)\n",
    ),
    (
        "            save_checkpoint_atomic(\n                {\n",
        "            _v1x_epoch_telemetry.save_checkpoint(save_checkpoint_atomic,\n                {\n",
    ),
    (
        "        save_checkpoint_atomic(\n            {\n",
        "        _v1x_epoch_telemetry.save_checkpoint(save_checkpoint_atomic,\n            {\n",
    ),
    (
        "        _print_report(\n            \"EpochPostRun\",\n",
        "        _v1x_epoch_telemetry.report(_print_report,\n            \"EpochPostRun\",\n",
    ),
    (
        "    save_checkpoint_atomic(completed_best, checkpoint_path)\n",
        "    _v1x_epoch_telemetry.save_checkpoint(save_checkpoint_atomic, completed_best, checkpoint_path)\n",
    ),
    (
        "    _print_report(\n        \"PostRun\",\n",
        "    _v1x_epoch_telemetry.report(_print_report,\n        \"PostRun\",\n",
    ),
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seconds(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TelemetryError(f"{label} must be a measured number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise TelemetryError(f"{label} must be finite and nonnegative")
    return result


def overlay_source(source: str) -> tuple[str, dict[str, Any]]:
    """Build a checked timer overlay without importing any neural dependencies."""
    if _sha256(source.encode("utf-8")) != RUN_TRAIN_SHA256:
        raise TelemetryError("Pinned native v1 run_train source hash mismatch; telemetry was not installed")
    rewritten = source
    for index, (before, after) in enumerate(SOURCE_REWRITES):
        occurrences = rewritten.count(before)
        if occurrences != 1:
            raise TelemetryError(
                f"Telemetry insertion {index} expected one source pattern, found {occurrences}; no fallback"
            )
        rewritten = rewritten.replace(before, after, 1)
    tree = ast.parse(rewritten)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != "run_train":
        raise TelemetryError("Telemetry overlay must contain only the original run_train function")
    # Compilation verifies the complete resulting function before installation.
    compile(tree, "<native-v1-epoch-telemetry-check>", "exec")
    return rewritten, {
        "format": FORMAT,
        "pipeline_source_sha256": PIPELINE_SHA256,
        "run_train_source_sha256": RUN_TRAIN_SHA256,
        "instrumented_run_train_sha256": _sha256(rewritten.encode("utf-8")),
        "source_rewrite_count": len(SOURCE_REWRITES),
        "archived_source_files_changed": False,
        "model_sampler_loss_values_changed": False,
        "per_batch_cuda_synchronization_added": False,
        "rng_operations_added": False,
    }


def timed_materialize_sample_views(*args, **kwargs):
    """Call the installed native/strict materializer exactly once in this process."""
    from hiercp import data

    original = getattr(data, _ORIGINAL_MATERIALIZER_KEY, None)
    if original is None:
        raise TelemetryError("Sampling timer has no verified original materializer in this worker")
    started = time.perf_counter()
    sample = original(*args, **kwargs)
    elapsed = time.perf_counter() - started
    if not isinstance(sample, dict) or SAMPLE_TIMING_KEY in sample:
        raise TelemetryError("Materializer result has an invalid or duplicate sampling telemetry field")
    sample[SAMPLE_TIMING_KEY] = _seconds(elapsed, "sampling seconds")
    return sample


def prepare_sampling_capture() -> None:
    """Install metadata-only sampling capture after native resource calibration."""
    from hiercp import data

    current = data.materialize_sample_views
    saved = getattr(data, _ORIGINAL_MATERIALIZER_KEY, None)
    if current is timed_materialize_sample_views:
        if saved is None:
            raise TelemetryError("Sampling telemetry installation is incomplete")
        return
    if saved is not None:
        raise TelemetryError("Sampling materializer changed after telemetry installation")
    setattr(data, _ORIGINAL_MATERIALIZER_KEY, current)
    data.materialize_sample_views = timed_materialize_sample_views


def telemetry_worker_init(worker_id: int) -> None:
    """Picklable worker hook; PyTorch's existing worker seeds are left untouched."""
    prepare_sampling_capture()


def timed_collate_samples(samples):
    """Preserve the original collator and attach only measured Python metadata."""
    from hiercp import data

    if not samples:
        raise TelemetryError("Sampling telemetry cannot collate an empty batch")
    durations = []
    for sample in samples:
        if SAMPLE_TIMING_KEY not in sample:
            raise TelemetryError("Sampling timing missing from an actual loader sample; no fallback")
        durations.append(_seconds(sample[SAMPLE_TIMING_KEY], "sample sampling seconds"))
    started = time.perf_counter()
    batch = data.collate_samples(samples)
    elapsed = time.perf_counter() - started
    if hasattr(batch, BATCH_TIMING_KEY):
        raise TelemetryError("Original batch unexpectedly already contains telemetry")
    setattr(batch, BATCH_TIMING_KEY, {
        "sampling_worker_seconds_sum": sum(durations),
        "sampling_worker_sample_seconds_min": min(durations),
        "sampling_worker_sample_seconds_max": max(durations),
        "sampling_measured_samples": len(durations),
        "collate_worker_seconds_sum": _seconds(elapsed, "collate seconds"),
    })
    return batch


class PassTelemetry:
    """Loader wait, worker sampling and GPU compute for one existing epoch pass."""

    def __init__(self, torch_module, device, training_mode: bool):
        self.torch = torch_module
        self.device = device
        self.training_mode = bool(training_mode)
        self.loader_wait_seconds = 0.0
        self.sampling_worker_seconds_sum = 0.0
        self.collate_worker_seconds_sum = 0.0
        self.sampling_measured_samples = 0
        self.batches = 0
        self.sample_min: float | None = None
        self.sample_max: float | None = None
        self._events = []
        self._pending_event = None
        self._compute_started: float | None = None
        self.compute_host_enqueue_seconds = 0.0

    def iter_loader(self, loader):
        from tqdm import tqdm

        started = time.perf_counter()
        iterator = iter(loader)
        self.loader_wait_seconds += time.perf_counter() - started
        progress = tqdm(total=len(loader), desc="v1 optimization" if self.training_mode else "v1 validation",
                        unit="batch", mininterval=1.0, leave=False)
        try:
            while True:
                started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    self.loader_wait_seconds += time.perf_counter() - started
                    return
                self.loader_wait_seconds += time.perf_counter() - started
                sampling = getattr(batch, BATCH_TIMING_KEY, None)
                if not isinstance(sampling, dict):
                    raise TelemetryError("Loader batch has no actual worker sampling timing; no fallback")
                count = sampling.get("sampling_measured_samples")
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    raise TelemetryError("Measured sampling sample count must be a positive integer")
                if count != batch.sample_count:
                    raise TelemetryError("Measured sampling sample count disagrees with the original batch")
                minimum = _seconds(sampling.get("sampling_worker_sample_seconds_min"), "sampling minimum")
                maximum = _seconds(sampling.get("sampling_worker_sample_seconds_max"), "sampling maximum")
                if minimum > maximum:
                    raise TelemetryError("Measured sampling minimum exceeds its maximum")
                self.sampling_worker_seconds_sum += _seconds(
                    sampling.get("sampling_worker_seconds_sum"), "sampling worker total"
                )
                self.collate_worker_seconds_sum += _seconds(
                    sampling.get("collate_worker_seconds_sum"), "collate worker total"
                )
                self.sampling_measured_samples += count
                self.sample_min = minimum if self.sample_min is None else min(self.sample_min, minimum)
                self.sample_max = maximum if self.sample_max is None else max(self.sample_max, maximum)
                self.batches += 1
                yield batch
                # The original epoch loop releases this batch before fetching
                # the next one. Do not extend graph/tensor lifetimes here.
                del batch
                # Update only when the original loop has finished this batch;
                # no loss reads, .item() calls or GPU synchronization are added.
                progress.update(1)
        finally:
            progress.close()

    def begin_compute(self) -> None:
        if self._compute_started is not None:
            raise TelemetryError("GPU compute timing interval was opened twice")
        self._compute_started = time.perf_counter()
        if self.device.type == "cuda":
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record(self.torch.cuda.current_stream(self.device))
            self._pending_event = (start, end)

    def end_compute(self) -> None:
        if self._compute_started is None:
            raise TelemetryError("GPU compute timing interval was never opened")
        if self.device.type == "cuda":
            start, end = self._pending_event
            end.record(self.torch.cuda.current_stream(self.device))
            self._events.append((start, end))
            self._pending_event = None
        self.compute_host_enqueue_seconds += time.perf_counter() - self._compute_started
        self._compute_started = None

    def finish(self, *, expected_samples: int, expected_batches: int) -> dict[str, Any]:
        if self._compute_started is not None or self._pending_event is not None:
            raise TelemetryError("GPU compute timing interval remains unfinished")
        if self.sampling_measured_samples != expected_samples or self.batches != expected_batches:
            raise TelemetryError("Epoch sampling measurements do not cover every original loader sample/batch")
        compute_seconds = None
        if self.device.type == "cuda":
            if len(self._events) != expected_batches:
                raise TelemetryError("CUDA event pairs do not cover every original batch")
            # The original pipeline has already synchronized this device at the
            # end of the pass. No new synchronization is needed here.
            compute_seconds = sum(_seconds(start.elapsed_time(end), "CUDA event milliseconds")
                                  for start, end in self._events) / 1000.0
        self._events.clear()
        return {
            "loader_wait_seconds": self.loader_wait_seconds,
            "sampling_worker_seconds_sum": self.sampling_worker_seconds_sum,
            "sampling_worker_sample_seconds_min": self.sample_min,
            "sampling_worker_sample_seconds_max": self.sample_max,
            "sampling_measured_samples": self.sampling_measured_samples,
            "collate_worker_seconds_sum": self.collate_worker_seconds_sum,
            "gpu_optimization_seconds": compute_seconds if self.training_mode else None,
            "gpu_validation_compute_seconds": compute_seconds if not self.training_mode else None,
            "gpu_compute_timing_status": "MEASURED_CUDA_EVENTS" if compute_seconds is not None else "NOT_APPLICABLE_CPU",
            "compute_host_enqueue_seconds": self.compute_host_enqueue_seconds,
            "timing_scope": {
                "loader_wait": "Host iter/next wait; CUDA prefetch may fetch the following batch before yielding",
                "sampling_worker": "Sum of actual materialize_sample_views calls, including both views and strict subset work",
                "sampling_overlap": "Worker sampling/collation overlap with GPU work; worker-second sums are not additive epoch wall time",
                "gpu_compute": "Current-stream forward, loss, backward and optimizer update; excludes ranking metrics and H2D prefetch stream",
                "host_enqueue": "Host time scheduling compute; not a measurement of GPU execution time",
                "validation_wall": "Original elapsed_seconds includes validation loader, transfer, forward, loss and metrics",
            },
        }


class EpochTelemetry:
    """Exclusive invocation JSONL plus additions to original printed reports."""

    prepare_sampling_capture = staticmethod(prepare_sampling_capture)
    collate_samples = staticmethod(timed_collate_samples)
    worker_init = staticmethod(telemetry_worker_init)

    def __init__(self, output_dir: Path, metadata: Mapping[str, Any], identity: Mapping[str, Any]):
        # Validate serializability/finite JSON before creating any output.
        self.metadata = json.loads(json.dumps(dict(metadata), allow_nan=False))
        self.identity = dict(identity)
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        invocation = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex
        self.record_path = output_dir / f"epoch_telemetry_{invocation}.jsonl"
        self.identity["record_path"] = str(self.record_path)
        self._stream = self.record_path.open("x", encoding="utf-8", newline="\n")
        self._epoch: int | None = None
        self._epoch_started: float | None = None
        self._checkpoint_saves = []
        self._finalization_saves = []
        self._write({"event": "TelemetryInstalled", "identity": self.identity, "request": self.metadata})

    def _write(self, value: dict[str, Any]) -> None:
        if self._stream.closed:
            raise TelemetryError("Epoch telemetry output was closed before the training report")
        self._stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()

    def new_pass(self, torch_module, device, training_mode: bool) -> PassTelemetry:
        return PassTelemetry(torch_module, device, training_mode)

    def begin_epoch(self, epoch: int) -> None:
        if self._epoch is not None:
            raise TelemetryError("Previous epoch did not complete its report; refusing to reset timing")
        self._epoch = int(epoch)
        self._epoch_started = time.perf_counter()
        self._checkpoint_saves = []

    def save_checkpoint(self, original, payload, path) -> None:
        started = time.perf_counter()
        original(payload, path)
        record = {"path": str(Path(path).resolve()), "elapsed_seconds": time.perf_counter() - started}
        if self._epoch is not None:
            self._checkpoint_saves.append(record)
        else:
            self._finalization_saves.append(record)

    def report(self, original, name: str, payload: dict[str, Any]) -> None:
        expanded = dict(payload)
        if name == "EpochPostRun":
            if self._epoch is None or payload.get("epoch") != self._epoch:
                raise TelemetryError("Original epoch report disagrees with telemetry's active epoch")
            elapsed = time.perf_counter() - self._epoch_started
            peaks = [part.get("cuda_peak_allocated_bytes") for part in
                     (payload.get("train", {}), payload.get("validation", {}))]
            peaks = [value for value in peaks if isinstance(value, int) and not isinstance(value, bool)]
            expanded["telemetry"] = {
                "format": FORMAT,
                "epoch_wall_seconds": elapsed,
                "checkpoint_save_seconds": sum(record["elapsed_seconds"] for record in self._checkpoint_saves),
                "checkpoint_saves": list(self._checkpoint_saves),
                "epoch_peak_vram_allocated_bytes": max(peaks) if peaks else None,
                "scope": "Train, validation, scheduler, connectivity and best/last saves; ends before report serialization",
                "record_path": str(self.record_path),
            }
            self._write({"event": name, **expanded})
            original(name, expanded)
            self._epoch = None
            self._epoch_started = None
        elif name == "PostRun":
            expanded["telemetry"] = {
                "format": FORMAT,
                "record_path": str(self.record_path),
                "finalization_checkpoint_save_seconds": sum(record["elapsed_seconds"] for record in self._finalization_saves),
                "finalization_checkpoint_saves": list(self._finalization_saves),
            }
            self._write({"event": name, **expanded})
            original(name, expanded)
        else:
            raise TelemetryError(f"Unexpected telemetry report name {name!r}")


def install(pipeline, output_dir, metadata: Mapping[str, Any]) -> EpochTelemetry:
    """Install checked timing in memory; caller closes the returned session."""
    if not isinstance(metadata, Mapping):
        raise TelemetryError("Telemetry contract metadata must be a mapping")
    if _GLOBAL_NAME in pipeline.__dict__:
        raise TelemetryError("Epoch telemetry was already installed in this pipeline")
    pipeline_path = Path(pipeline.__file__).resolve(strict=True)
    if _sha256(pipeline_path.read_bytes()) != PIPELINE_SHA256:
        raise TelemetryError("Pinned native v1 pipeline file hash mismatch; no instrumentation installed")
    source = inspect.getsource(pipeline.run_train)
    rewritten, identity = overlay_source(source)
    future_tree = ast.parse("from __future__ import annotations\n" + rewritten)
    code = compile(future_tree, str(pipeline_path) + "::<epoch-telemetry>", "exec")
    temporary_globals = dict(pipeline.__dict__)
    exec(code, temporary_globals)
    compiled = temporary_globals["run_train"]
    original = pipeline.run_train
    instrumented = FunctionType(compiled.__code__, pipeline.__dict__, original.__name__,
                                original.__defaults__, original.__closure__)
    functools.update_wrapper(instrumented, original)
    instrumented.__kwdefaults__ = original.__kwdefaults__
    session = EpochTelemetry(Path(output_dir), metadata, identity)
    pipeline.__dict__[_GLOBAL_NAME] = session
    pipeline.run_train = instrumented
    return session
