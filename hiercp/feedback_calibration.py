"""Measured full-pool feedback workloads, isolated from actual stream state."""
from __future__ import annotations

import copy
import json
import os
import time
import uuid

import numpy as np
import torch

from hiercp.feedback_resources import (EXECUTION_VERSION, allocation_guard, calibration_batches,
                                      content_digest, optimizer_probe_gap, require_allocation,
                                      snapshot_guard, tensor_bytes)
from hiercp.preparation_runtime import Measurement, snapshot


def close_loader(loader):
    """Close only workers created by this specific owned DataLoader."""
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def calibrate_feedback(runtime, entries, grouped=None):
    from hiercp.feedback import _capture_rng, _restore_rng, tensor_state_sha256
    if not entries or (grouped is not None and set(entries) != set(grouped)):
        raise ValueError("Feedback calibration requires exactly its real observed or prediction cohort")
    rows = runtime._ensure_inventory()
    allocation = snapshot()
    if runtime.num_workers > allocation["cpu_capacity"]:
        raise ValueError("Inherited feedback workers exceed the current CPU allocation; remeasure explicitly")
    if any(parameter.grad is not None for parameter in runtime.model.parameters()):
        raise ValueError("Feedback calibration must start at a cleared epoch boundary")
    sizes, power = [], 1
    while power <= len(entries):
        sizes.append(power)
        power *= 2
    if len(entries) not in sizes:
        sizes.append(len(entries))
    rng, was_training = _capture_rng(), runtime.model.training
    optimizer_before = content_digest(runtime.optimizer.state_dict())
    before = tensor_state_sha256(runtime.model.state_dict())
    model_bytes = tensor_bytes(runtime.model.state_dict())
    snapshot_proof = snapshot_guard(model_bytes, phase="feedback_calibration_model_snapshot")
    # One direct owned CPU copy, not a device->CPU copy followed by a second
    # clone. The admitted snapshot size therefore covers its tensor payload.
    saved_model = {name: value.detach().to(device="cpu", copy=True)
                   for name, value in runtime.model.state_dict().items()}
    actual_steps = runtime.optimizer_steps
    connected = set(runtime.connected_parameters)
    trials, best = [], None
    progress_path = (runtime.provider.cache / ("calibration." + uuid.uuid4().hex + ".jsonl")
                     if hasattr(runtime.provider, "cache") else None)
    progress = None if progress_path is None else progress_path.open("x", encoding="utf-8")
    def event(stage, **values):
        if progress is not None:
            progress.write(json.dumps({"format": EXECUTION_VERSION, "stage": stage,
                                       "pid": os.getpid(), "resources": snapshot(), **values},
                                      sort_keys=True, allow_nan=False) + "\n")
            progress.flush()
            os.fsync(progress.fileno())
    try:
        event("begin", mode="update" if grouped is not None else "predict", entries=len(entries),
              inventory_sha256=runtime.inventory_sha256)
        for size in sizes:
            batches = calibration_batches(rows, entries, size)
            names_to_measure = [entry for batch in batches for entry in batch]
            budget = allocation_guard(rows, entries, size, workers=runtime.num_workers,
                                      prefetch_factor=runtime.config["prefetch_factor"],
                                      pin_memory=runtime.device.type == "cuda")
            row = {"physical_batch_size": size, "host_budget": budget,
                   "measured_entry_batches": batches, "candidate_pools_unchanged": True}
            event("trial_begin", **row)
            if not budget["accepted"]:
                row["status"] = "host_headroom_exceeded"
                trials.append(row)
                event("trial_end", **row)
                continue
            loader = probe = batch = logits = loss = None
            seconds, allocated_peaks, reserved_peaks, external_peaks, missing_peaks = [], [], [], [], []
            shapes, optimizer_gaps = [], []
            measurement = Measurement()
            try:
                require_allocation(budget)
                loader = runtime._loader(names_to_measure, size)
                if grouped is not None:
                    probe = torch.optim.AdamW(runtime.model.parameters(), lr=0.,
                                              weight_decay=runtime.config["weight_decay"])
                with measurement:
                    for repetition in range(runtime.config["calibration_repeats"]):
                        _restore_rng(rng)
                        runtime.model.train(grouped is not None)
                        external = 0
                        if runtime.device.type == "cuda":
                            torch.cuda.synchronize(runtime.device)
                            free, total = torch.cuda.mem_get_info(runtime.device)
                            external = max(0, total - free - torch.cuda.memory_reserved(runtime.device))
                            torch.cuda.reset_peak_memory_stats(runtime.device)
                        start = time.perf_counter()
                        for names, batch in loader:
                            if repetition == 0:
                                shapes.append({"source_patch_shape": list(batch.source_patches.shape),
                                               "target_patch_shape": list(batch.target_patches.shape),
                                               "local_nodes": batch.local_batch.num_nodes,
                                               "local_edges": batch.local_batch.num_edges,
                                               "patient_nodes": batch.patient_batch.num_nodes,
                                               "patient_edges": batch.patient_batch.num_edges,
                                               "population_nodes": batch.prototype_batch.num_nodes,
                                               "population_edges": batch.prototype_batch.num_edges})
                            if probe is not None:
                                probe.zero_grad(set_to_none=True)
                                loss = runtime._forward_loss(batch, names, grouped)
                                runtime._backward(loss, probe, record=False)
                                loss = None
                            else:
                                with torch.inference_mode():
                                    logits = runtime.model.predict_logits(batch, local_chunk_size=runtime.local_chunk_size)
                                logits = None
                            batch = None
                        if runtime.device.type == "cuda":
                            torch.cuda.synchronize(runtime.device)
                        seconds.append(time.perf_counter() - start)
                        allocated_peaks.append(torch.cuda.max_memory_allocated(runtime.device) if runtime.device.type == "cuda" else 0)
                        reserved_peaks.append(torch.cuda.max_memory_reserved(runtime.device) if runtime.device.type == "cuda" else 0)
                        external_peaks.append(external)
                        if probe is not None:
                            gap = optimizer_probe_gap(runtime.optimizer, probe, runtime.model.parameters())
                            optimizer_gaps.append(gap)
                            missing_peaks.append(gap["additional_bytes"])
                        else:
                            missing_peaks.append(0)
                total = torch.cuda.get_device_properties(runtime.device).total_memory if runtime.device.type == "cuda" else None
                estimated = max(a + e + m for a, e, m in zip(reserved_peaks, external_peaks, missing_peaks))
                safe = total is None or estimated <= total * runtime.config["max_vram_fraction"]
                throughput = len(names_to_measure) / float(np.median(seconds))
                row.update(status="safe" if safe else "vram_headroom_exceeded", seconds=seconds,
                           samples_per_second=throughput, peak_vram_bytes=max(allocated_peaks),
                           peak_reserved_vram_bytes=max(reserved_peaks),
                           estimated_external_vram_bytes=max(external_peaks),
                           actual_optimizer_state_missing_from_probe_bytes=max(
                               (gap["historical_moment_bytes"] for gap in optimizer_gaps), default=0),
                           unmeasured_optimizer_gradient_workspace_bytes=max(missing_peaks),
                           optimizer_probe_gaps=optimizer_gaps,
                           estimated_actual_workload_reserved_vram_bytes=estimated,
                           input_graph_shapes=shapes, measurement=measurement.report,
                           persistent_loader_reused_between_repeats=runtime.num_workers > 0)
                if safe and (best is None or (throughput, size) > best):
                    best = throughput, size
            except torch.cuda.OutOfMemoryError as error:
                row.update(status="cuda_oom", error=str(error), measurement=getattr(measurement, "report", None))
            finally:
                batch = logits = loss = None
                runtime.model.zero_grad(set_to_none=True)
                if loader is not None:
                    close_loader(loader)
                loader = probe = None
                _restore_rng(rng)
                runtime.model.train(was_training)
                # The lr=0 probe is not allowed to carry mutable model buffers
                # into the next candidate, including when a CUDA OOM occurs.
                changed = before != tensor_state_sha256(runtime.model.state_dict())
                runtime.model.load_state_dict(saved_model, strict=True)
                if runtime.device.type == "cuda":
                    torch.cuda.empty_cache()
                if changed:
                    raise RuntimeError("Feedback probe changed model weights/buffers; original state restored")
            trials.append(row)
            event("trial_end", **row)
        if before != tensor_state_sha256(runtime.model.state_dict()):
            raise RuntimeError("Feedback calibration changed model weights/buffers; restored snapshot, no training continued")
        if (optimizer_before != content_digest(runtime.optimizer.state_dict())
                or actual_steps != runtime.optimizer_steps or connected != runtime.connected_parameters):
            raise RuntimeError("Feedback calibration changed actual optimizer or stream counters")
        if best is None:
            raise RuntimeError(f"No measured safe full-hierarchy feedback batch: {trials}; no scale reduction applied")
        report = {"format": EXECUTION_VERSION,
                  "mode": "actual_observation_bce_backward_adamw_lr0" if grouped is not None else "full_pool_prediction",
                  "physical_batch_size": best[1], "effective_batch_size": best[1], "trials": trials,
                  "available_source_entries": len(entries), "calibration_entry_order": list(entries),
                  "all_bank_inventory_sha256": runtime.inventory_sha256,
                  "all_bank_entries": len(rows), "cold_preparation": copy.deepcopy(runtime.preparation_report),
                  "calibration_policy": "largest_then_high_low_mixed_real_entries",
                  "observation_distribution_unchanged": True,
                  "num_workers": runtime.num_workers, "prefetch_factor": runtime.config["prefetch_factor"],
                  "worker_policy": "inherited_measured_quality_workers_revalidated_on_actual_feedback_workload",
                  "device": str(runtime.device), "allocation": allocation,
                  "model_snapshot_bytes": model_bytes, "snapshot_admission": snapshot_proof,
                  "actual_optimizer_parked_host_bytes": tensor_bytes(runtime.optimizer.state_dict()),
                  "model_kwargs": runtime.identity.get("model_kwargs", "explicit_debug_fixture"),
                  "precision": "bfloat16_autocast" if grouped is not None and runtime.config["amp"] and runtime.device.type == "cuda" else "float32",
                  "parameter_count": sum(p.numel() for p in runtime.model.parameters()),
                  "trainable_parameter_count": sum(p.numel() for p in runtime.model.parameters() if p.requires_grad),
                  "candidate_counts": sorted(set(runtime.provider.counts.values())),
                  "representative_calibration_only": True,
                  "model_optimizer_rng_and_stream_state_preserved": True}
        report["progress_path"] = None if progress_path is None else str(progress_path)
        event("complete", selected_physical_batch_size=best[1])
        print(f"[FeedbackGNNCalibration] {report}", flush=True)
        return report
    finally:
        runtime.model.load_state_dict(saved_model, strict=True)
        runtime.model.zero_grad(set_to_none=True)
        runtime.model.train(was_training)
        _restore_rng(rng)
        if progress is not None:
            progress.close()
