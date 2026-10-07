"""Measure execution settings on the unchanged, complete comparison batch.

Only existing activation checkpoint switches and the dense CNN execution chunk
are varied. Source batching, model structure, candidate coverage, precision,
objective and optimizer state are never changed. Calibration has no optimizer.
"""
from __future__ import annotations

import gc
import math
import statistics
import time
import traceback

import torch

from .u_bridge_training import capture_rng, cpu_copy, digest, restore_rng


FORMAT = "comparison_gpu_execution_policy_v1"
FIELDS = ("dense_batch_size", "checkpoint_dense_encoder", "checkpoint_local_blocks")


class ExecutionCalibrationRejected(RuntimeError):
    def __init__(self, reports):
        super().__init__("Original execution policy failed measured CUDA calibration; no model/data/batch fallback")
        self.reports = reports


def execution_settings(net):
    """Read only the original local encoder's three execution attributes."""
    local = net.local_encoder
    result = {name: getattr(local, name) for name in FIELDS}
    _validate_settings(result)
    return result


def _validate_settings(settings):
    if set(settings) != set(FIELDS):
        raise ValueError("Exactly the original dense chunk and two checkpoint switches are required")
    if type(settings["dense_batch_size"]) is not int or settings["dense_batch_size"] < 1:
        raise ValueError("Positive dense execution chunk required")
    if any(type(settings[name]) is not bool for name in FIELDS[1:]):
        raise ValueError("Checkpoint settings must be explicit booleans")


def apply_execution_settings(net, settings):
    """Apply existing execution attributes; no modules or parameters are replaced."""
    _validate_settings(settings)
    execution_settings(net)
    for name in FIELDS:
        setattr(net.local_encoder, name, settings[name])


def _flag_candidates(original):
    result = [dict(original)]
    for dense, local in ((False, True), (True, False), (False, False), (True, True)):
        row = dict(original, checkpoint_dense_encoder=dense, checkpoint_local_blocks=local)
        if row not in result:
            result.append(row)
    return result


def dense_chunk_candidates(original, patch_count):
    """Measure doubling through the full actual patch count, without a graph cap."""
    if type(original) is not int or original < 1 or type(patch_count) is not int or patch_count < 1:
        raise ValueError("Positive original chunk and actual patch count required")
    result = [original]
    while result[-1] < patch_count:
        result.append(min(result[-1] * 2, patch_count))
    return result


def _cuda_device(net, batch):
    devices = {value.device for value in (*net.parameters(), *net.buffers())}
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        raise ValueError("One existing CUDA model device required; calibration never moves the model")
    device = next(iter(devices))
    if batch.source_patches.device != device or batch.target_patches.device != device:
        raise ValueError("The actual complete batch must already reside on the model CUDA device")
    return device


def _batch_contract(batch):
    counts = list(batch.counts)
    if not counts or counts != [8] * len(counts) or batch.local_batch_view2 is None:
        raise ValueError("Complete original physical source batch with eight candidates and genuine two views required")
    if batch.source_patches.ndim != 5 or batch.target_patches.ndim != 5:
        raise ValueError("Original five-dimensional dense patches required")
    if batch.target_patches.shape[0] != sum(counts):
        raise ValueError("Dense target patches must cover every candidate in the source batch")
    return dict(physical_source_batch=len(counts), candidate_graphs=sum(counts),
        sampled_graph_views=2 * sum(counts), counts=counts,
        source_patch_shape=list(batch.source_patches.shape),
        target_patch_shape=list(batch.target_patches.shape),
        actual_source_indices=list(getattr(batch, "bridge_indices", ())))


def _restore_buffers(net, buffers):
    with torch.no_grad():
        for name, value in net.named_buffers():
            value.copy_(buffers[name])


def _probe_once(net, batch, loss_fn, *, amp, device):
    """Time forward/backward only; finite checks and CPU snapshots are untimed."""
    net.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", enabled=amp):
        output = net(batch)
        loss = loss_fn(output)
    if not torch.is_tensor(loss) or loss.numel() != 1:
        raise ValueError("The exact bound comparison objective must return a scalar Tensor")
    torch.cuda.synchronize(device)
    forward_done = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    backward_done = time.perf_counter()
    peak = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    scores = getattr(output, "scores", None)
    consistency = getattr(output, "consistency", None)
    if (not isinstance(scores, (tuple, list)) or len(scores) != len(batch.counts)
            or any(not torch.is_tensor(value) or value.ndim != 1 or value.numel() != 8 for value in scores)
            or not torch.is_tensor(consistency) or consistency.numel() != 1):
        raise ValueError("Original ordered eight-score outputs and genuine consistency scalar required")
    observed = dict(scores=torch.stack(scores).detach().float().cpu(),
                    consistency=consistency.detach().float().reshape(()).cpu(),
                    loss=loss.detach().float().reshape(()).cpu())
    gradients = {}
    for name, parameter in net.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise RuntimeError("Disconnected calibration gradient: " + name)
            gradients[name] = parameter.grad.detach().to("cpu", copy=True)
    if not all(bool(torch.isfinite(value).all()) for value in (*observed.values(), *gradients.values())):
        raise FloatingPointError("Nonfinite calibration output, loss, or gradients")
    return dict(forward_seconds=forward_done-started, backward_seconds=backward_done-forward_done,
                forward_backward_seconds=backward_done-started, peak_cuda_bytes=peak,
                peak_reserved_bytes=reserved, loss=float(observed["loss"]),
                gradient_tensors=len(gradients), rng_after_sha256=digest(capture_rng())), observed, gradients


def _equivalence(observed, gradients, reference, *, rtol, atol, noise=None):
    """Compare complete vectors against tolerances plus measured repeat noise.

    The global gradient test deliberately has no ``atol * sqrt(total_numel)``
    term: that term would hide material drift across ten million parameters.
    Individual tensor tests use the L2 norm of their elementwise tolerance.
    """
    reference_output, reference_gradients = reference
    failed_output, failed_gradients = [], []
    elementwise_failed = []
    output_max, gradient_max = 0., 0.
    reference_squared, difference_squared, actual_squared, inner_product = 0., 0., 0., 0.
    failed_gradient_details = {}
    tensor_differences = {}
    if set(gradients) != set(reference_gradients):
        raise RuntimeError("Execution policy changed gradient connectivity")
    for name, value in observed.items():
        other = reference_output[name]
        output_max = max(output_max, float((value-other).abs().max()))
        repeat_noise = 0 if noise is None else noise["outputs"][name]
        if not bool(((value-other).abs() <= repeat_noise + atol + rtol*other.abs()).all()):
            failed_output.append(name)
    for name, value in gradients.items():
        other = reference_gradients[name]
        difference = value.double()-other.double()
        maximum = float(difference.abs().max())
        reference_norm = float(other.double().norm())
        actual_norm = float(value.double().norm())
        difference_norm = float(difference.norm())
        reference_squared += reference_norm**2
        actual_squared += actual_norm**2
        difference_squared += difference_norm**2
        inner_product += float((value.double()*other.double()).sum())
        gradient_max = max(gradient_max, maximum)
        tensor_differences[name] = difference_norm
        if not torch.allclose(value, other, rtol=rtol, atol=atol):
            elementwise_failed.append(name)
        repeat_noise = 0 if noise is None else noise["gradient_tensors_l2"].get(name, 0.)
        bound = repeat_noise + rtol*reference_norm + atol*math.sqrt(value.numel())
        if difference_norm > bound:
            failed_gradients.append(name)
            failed_gradient_details[name] = dict(elements=value.numel(), max_abs=maximum,
                reference_l2=reference_norm, actual_l2=actual_norm, difference_l2=difference_norm,
                relative_l2=difference_norm/max(reference_norm, 1e-30), bound_l2=bound,
                baseline_repeat_noise_l2=repeat_noise)
    global_repeat_noise = 0 if noise is None else noise["gradient_global_l2"]
    global_bound = max(1e-6, global_repeat_noise + rtol*math.sqrt(reference_squared))
    global_accepted = math.sqrt(difference_squared) <= global_bound
    return dict(accepted=not failed_output and not failed_gradients and global_accepted,
        output_max_abs_difference=output_max, gradient_max_abs_difference=gradient_max,
        failed_outputs=failed_output, failed_gradient_parameters=failed_gradients,
        elementwise_gradient_failed_parameters=elementwise_failed,
        gradient_reference_l2=math.sqrt(reference_squared), gradient_actual_l2=math.sqrt(actual_squared),
        gradient_difference_l2=math.sqrt(difference_squared),
        gradient_relative_l2=math.sqrt(difference_squared/max(reference_squared, 1e-60)),
        gradient_cosine=inner_product/max(math.sqrt(reference_squared*actual_squared), 1e-60),
        gradient_global_bound_l2=global_bound, gradient_global_accepted=global_accepted,
        gradient_tensor_difference_l2=tensor_differences, failed_gradient_details=failed_gradient_details)


def calibrate_execution_policy(net, batch, loss_fn, *, amp, cuda_limit_bytes,
                               reserve_bytes=None, warmup=1, repeats=2, notify=None):
    """Return ``(selected_settings, receipt)`` after restoring all caller state.

    Four checkpoint combinations are measured at the original dense chunk, then
    the fastest passing combination is measured at chunks doubling through the
    actual patch count. Every trial uses the exact same full batch and RNG.
    Optimizer updates, loader timing and full-cohort worst-case claims are out
    of scope. The caller applies a selected policy separately and must retain
    conservative execution for uncalibrated larger workloads.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA required for execution calibration; no CPU fallback")
    if type(amp) is not bool or type(cuda_limit_bytes) is not int or cuda_limit_bytes <= 0:
        raise ValueError("Explicit precision and positive declared CUDA byte budget required")
    if type(warmup) is not int or warmup < 1 or type(repeats) is not int or repeats < 2:
        raise ValueError("At least one warmup and two measured forward/backward repetitions required")
    if notify is not None and not callable(notify):
        raise TypeError("notify must be callable")
    workload = _batch_contract(batch)
    device = _cuda_device(net, batch)
    original = execution_settings(net)
    parameter_bytes = sum(p.numel()*p.element_size() for p in net.parameters() if p.requires_grad)
    if reserve_bytes is None:
        reserve_bytes = 3 * parameter_bytes
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("Nonnegative explicit optimizer/temporary memory reserve required")
    torch.cuda.synchronize(device)
    # Validation can leave a large unused caching-allocator pool. Return those
    # blocks before asking the driver for free memory; live allocations remain.
    torch.cuda.empty_cache()
    free, total = map(int, torch.cuda.mem_get_info(device))
    allocated = int(torch.cuda.memory_allocated(device))
    headroom = max(cuda_limit_bytes // 10, 1024**3)
    hard_limit = min(cuda_limit_bytes, free + allocated)
    ceiling = hard_limit - headroom - reserve_bytes
    if hard_limit < allocated:
        raise MemoryError("Existing allocation exceeds declared CUDA budget")
    rtol, atol = ((1e-2, 1e-4) if amp else (1e-4, 1e-6))
    state = cpu_copy(net.state_dict())
    original_hash = digest(state)
    buffers = {name: value.detach().to("cpu", copy=True) for name, value in net.named_buffers()}
    old_grads = {name: None if parameter.grad is None else parameter.grad.detach().to("cpu", copy=True)
                 for name, parameter in net.named_parameters()}
    modes = [(module, module.training) for module in net.modules()]
    rng = capture_rng()
    reports, reference, reference_rng = [], None, None
    baseline_noise = dict(outputs={}, gradient_tensors_l2={}, gradient_global_l2=0.)

    def measure(settings, stage):
        nonlocal reference, reference_rng
        apply_execution_settings(net, settings)
        row = dict(policy=dict(settings), stage=stage, accepted=False, trials=[],
                   peak_cuda_bytes=0, peak_reserved_bytes=0)
        try:
            for index in range(warmup + repeats):
                _restore_buffers(net, buffers)
                restore_rng(rng)
                metrics, observed, gradients = _probe_once(net, batch, loss_fn, amp=amp, device=device)
                row["peak_cuda_bytes"] = max(row["peak_cuda_bytes"], metrics["peak_cuda_bytes"])
                row["peak_reserved_bytes"] = max(row["peak_reserved_bytes"], metrics["peak_reserved_bytes"])
                if reference is None:
                    reference = observed, gradients
                    reference_rng = metrics["rng_after_sha256"]
                    baseline_noise["outputs"] = {name: torch.zeros_like(value) for name, value in observed.items()}
                check = _equivalence(observed, gradients, reference, rtol=rtol, atol=atol,
                                     noise=None if stage == "baseline" else baseline_noise)
                tensor_differences = check.pop("gradient_tensor_difference_l2")
                check["rng_consumption_matches_reference"] = metrics["rng_after_sha256"] == reference_rng
                if stage == "baseline":
                    # Repeated unchanged execution measures the noise floor. It
                    # is not an optimization candidate compared against itself.
                    for name, value in observed.items():
                        baseline_noise["outputs"][name] = torch.maximum(
                            baseline_noise["outputs"][name], (value-reference[0][name]).abs())
                    for name, difference in tensor_differences.items():
                        baseline_noise["gradient_tensors_l2"][name] = max(
                            baseline_noise["gradient_tensors_l2"].get(name, 0.), difference)
                    baseline_noise["gradient_global_l2"] = max(
                        baseline_noise["gradient_global_l2"], check["gradient_difference_l2"])
                    check["within_zero_noise_tolerance"] = check["accepted"]
                    check["role"] = "original_unchanged_repeat_noise_measurement"
                    check["accepted"] = check["rng_consumption_matches_reference"]
                else:
                    check["accepted"] = check["accepted"] and check["rng_consumption_matches_reference"]
                metrics.update(warmup=index < warmup, equivalence=check)
                row["trials"].append(metrics)
                del observed, gradients
                limit = hard_limit if stage == "baseline" else ceiling
                if row["peak_cuda_bytes"] > limit:
                    row["error"] = ("Original measured peak exceeds declared available CUDA budget" if stage == "baseline"
                                    else "Measured peak exceeds declared CUDA budget minus headroom and optimizer reserve")
                    break
                if not check["accepted"]:
                    row["error"] = "Output/loss/gradient/RNG equivalence rejected at recorded tolerances plus original repeat noise"
                    break
            else:
                measured = row["trials"][warmup:]
                row.update(accepted=True,
                    forward_backward_seconds=statistics.median(value["forward_backward_seconds"] for value in measured),
                    forward_seconds=statistics.median(value["forward_seconds"] for value in measured),
                    backward_seconds=statistics.median(value["backward_seconds"] for value in measured))
                row["samples_per_second"] = workload["physical_source_batch"] / row["forward_backward_seconds"]
                row["within_preferred_headroom"] = row["peak_cuda_bytes"] <= ceiling
        except torch.cuda.OutOfMemoryError as error:
            row.update(error=f"CUDA OOM rejected: {error}",
                peak_cuda_bytes=max(row["peak_cuda_bytes"], int(torch.cuda.max_memory_allocated(device))),
                peak_reserved_bytes=max(row["peak_reserved_bytes"], int(torch.cuda.max_memory_reserved(device))))
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
        except FloatingPointError as error:
            row["error"] = str(error)
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
        finally:
            net.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        reports.append(row)
        if notify is not None:
            status = (f"accepted {row['forward_backward_seconds']:.3f}s"
                      if row["accepted"] else "rejected: " + row["error"].splitlines()[0])
            notify(f"GPU trial {len(reports)} chunk={settings['dense_batch_size']} "
                   f"checkpoint(dense/local)={int(settings['checkpoint_dense_encoder'])}/"
                   f"{int(settings['checkpoint_local_blocks'])} "
                   f"peak={row['peak_cuda_bytes']/1024**3:.2f}GiB {status}")
        return row

    try:
        net.train()
        baseline = measure(original, "baseline")
        if not baseline["accepted"]:
            raise ExecutionCalibrationRejected(reports)
        if baseline["within_preferred_headroom"]:
            for settings in _flag_candidates(original)[1:]:
                measure(settings, "checkpoint_flags")
            best_flags = min((row for row in reports if row["accepted"]),
                             key=lambda row: row["forward_backward_seconds"])["policy"]
            count = max(workload["source_patch_shape"][0], workload["target_patch_shape"][0])
            for chunk in dense_chunk_candidates(original["dense_batch_size"], count)[1:]:
                measure(dict(best_flags, dense_batch_size=chunk), "dense_chunk")
        selected = min((row for row in reports if row["accepted"]),
                       key=lambda row: row["forward_backward_seconds"])
    finally:
        net.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        net.load_state_dict(state, strict=True)
        _restore_buffers(net, buffers)  # Includes nonpersistent buffers omitted from state_dict.
        for name, parameter in net.named_parameters():
            saved = old_grads[name]
            parameter.grad = None if saved is None else saved.to(device=parameter.device, copy=True)
        for module, training in modes:
            module.training = training
        apply_execution_settings(net, original)
        restore_rng(rng)
        if digest(net.state_dict()) != original_hash:
            raise RuntimeError("Execution calibration failed to restore model state")
    receipt = dict(format=FORMAT, original_policy=original, selected_policy=dict(selected["policy"]),
        workload=workload, reports=reports, baseline_seconds=baseline["forward_backward_seconds"],
        selected_seconds=selected["forward_backward_seconds"],
        measured_speedup=baseline["forward_backward_seconds"]/selected["forward_backward_seconds"],
        baseline_peak_cuda_bytes=baseline["peak_cuda_bytes"], selected_peak_cuda_bytes=selected["peak_cuda_bytes"],
        cuda_device=str(device), cuda_device_name=torch.cuda.get_device_name(device),
        device_total_bytes=total, device_free_bytes=free, initially_allocated_bytes=allocated,
        declared_cuda_limit_bytes=cuda_limit_bytes, hard_cuda_limit_bytes=hard_limit, peak_budget_bytes=ceiling,
        headroom_bytes=headroom, optimizer_temporary_reserve_bytes=reserve_bytes,
        amp=amp, warmup_repetitions=warmup, measured_repetitions=repeats,
        numerical_equivalence=dict(rtol=rtol, atol=atol,
            scope="all scores, consistency, loss, global gradient vector and every trainable gradient tensor",
            output_bound="original pointwise repeat noise + atol + rtol*abs(reference)",
            gradient_global_bound="max(1e-6, original global L2 repeat noise + rtol*reference L2)",
            gradient_tensor_bound="original tensor L2 repeat noise + rtol*reference L2 + atol*sqrt(tensor elements)",
            baseline_repetitions=warmup+repeats, baseline_noise_multiplier=1,
            baseline_noise=dict(outputs={name: value.tolist() for name, value in baseline_noise["outputs"].items()},
                gradient_global_l2=baseline_noise["gradient_global_l2"],
                gradient_tensors_l2=baseline_noise["gradient_tensors_l2"])),
        backward_loss_scale=1.0,
        backward_precision_scope="AMP autocast with unscaled no-update backward; separate actual GradScaler update verification required",
        calibration_status="measured" if baseline["within_preferred_headroom"] else "original_retained_no_headroom",
        model_state_sha256=original_hash, model_gradients_rng_modes_and_settings_restored=True,
        optimizer_updates=0, optimizer_state_untouched=True,
        original_source_batch_and_candidate_coverage_unchanged=True,
        calibration_scope="same complete actual batch; forward/backward only; no full-cohort worst-case guarantee")
    if notify is not None:
        if not baseline["within_preferred_headroom"]:
            notify("GPU execution: original full batch fits the declared budget; retaining original settings because optimization headroom is unavailable")
        notify(f"GPU forward/backward calibration: {baseline['samples_per_second']:.3f} -> "
               f"{selected['samples_per_second']:.3f} sources/s ({receipt['measured_speedup']:.2f}x); "
               f"peak {baseline['peak_cuda_bytes']/1024**3:.2f} -> "
               f"{selected['peak_cuda_bytes']/1024**3:.2f}GiB; source batch unchanged")
    return dict(selected["policy"]), receipt
