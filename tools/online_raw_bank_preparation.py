"""Raw-CP/native-preprocessing bridge; no donor or candidate downsampling."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

from custom_trainers.onlinecp_raw_bank import save_case, save_candidate, SOURCE_MAPPING_FORMAT
from custom_trainers.onlinecp_raw_resampling import prepare_case, prepare_candidate, baseline_output
from hiercp.preparation_runtime import Measurement, run_case_jobs


def prepare_raw_case(bank_root, case_id, raw_image, raw_label, properties, plans,
                     pre_data, pre_seg, *, configuration_name, raw_spacing, raw_spatial_unit,
                     minimum_free_bytes):
    measurement = Measurement()
    lower_bound = int(np.prod(pre_data.shape[1:])) * 10
    report_path = Path(bank_root) / "preparation_resources" / (
        f"raw_case.{case_id}.{uuid.uuid4().hex}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with measurement:
            # This is an unavoidable persistent-array lower bound, NOT a
            # claim that cubic prefiltering or native temporary arrays fit.
            if measurement.before["available_memory_bytes"] <= lower_bound:
                raise RuntimeError(
                    f"Insufficient allocated RAM for raw-target baseline {case_id}: "
                    f"available={measurement.before['available_memory_bytes']}, "
                    f"persistent_arrays_alone={lower_bound}; no source/candidate was dropped")
            return _prepare_raw_case(
                bank_root, case_id, raw_image, raw_label, properties, plans, pre_data, pre_seg,
                configuration_name=configuration_name, raw_spacing=raw_spacing,
                raw_spatial_unit=raw_spatial_unit, minimum_free_bytes=minimum_free_bytes)
    finally:
        report = {"case_id": str(case_id), "format": "raw_target_case_resources_v1",
                  "persistent_array_lower_bound_bytes": lower_bound,
                  "limitation": "Array lower bound is not total peak RAM; sampled RSS includes actual native preparation, validation and publication",
                  **measurement.report}
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        print(f"[RawTargetCaseResources] {report_path}", flush=True)


def _prepare_raw_case(bank_root, case_id, raw_image, raw_label, properties, plans,
                      pre_data, pre_seg, *, configuration_name, raw_spacing, raw_spatial_unit,
                      minimum_free_bytes):
    started = time.perf_counter()
    shape = np.asarray(pre_data.shape[1:], dtype=np.int64)
    # The persistent unclipped float64 image and int16 full segmentation are
    # deliberate storage, not another full native preprocessing per candidate.
    minimum_payload_bytes = int(np.prod(shape)) * 10
    free = shutil.disk_usage(bank_root).free
    if free <= minimum_payload_bytes + int(minimum_free_bytes):
        raise RuntimeError(f"Insufficient disk for raw-target case {case_id}: "
                           f"free={free}, baseline_payload_alone={minimum_payload_bytes}, "
                           f"reserved_free={minimum_free_bytes}; "
                           "existing bank/preprocessing are preserved")
    case = prepare_case(raw_image, raw_label, properties, plans,
                        configuration_name=configuration_name, raw_spacing_xyz=raw_spacing,
                        raw_spatial_unit=raw_spatial_unit)
    case["metadata"]["case_id"] = str(case_id)
    actual = baseline_output(case)
    if tuple(actual.shape) != tuple(pre_data.shape) or tuple(case["baseline_seg"].shape) != tuple(pre_seg.shape):
        raise ValueError(f"Raw-target/native baseline shape mismatch: {case_id}")
    maximum_error = 0.0
    # Verification traverses ALL voxels in bounded I/O slabs, never a subset.
    for start in range(0, int(shape[0]), 16):
        selection = (slice(None), slice(start, start + 16), slice(None), slice(None))
        expected = np.asarray(pre_data[selection])
        predicted = np.asarray(actual[selection])
        if not np.isfinite(predicted).all() or not np.isfinite(expected).all():
            raise ValueError(f"Nonfinite raw-target/native baseline CT: {case_id}")
        error = float(np.max(np.abs(predicted - expected)))
        maximum_error = max(maximum_error, error)
        if not np.allclose(predicted, expected, rtol=1e-6, atol=1e-5):
            raise ValueError(f"Raw-target/native baseline CT mismatch: {case_id}; max_abs={error}")
        if not np.array_equal(case["baseline_seg"][selection], pre_seg[selection]):
            raise ValueError(f"Raw-target/native full-label baseline mismatch: {case_id}")
    del actual
    relative = f"raw_cases/{case_id}.json"
    digest = save_case(bank_root, relative, case)
    print("[RawTargetCase] " + json.dumps({"case_id": case_id, "format": SOURCE_MAPPING_FORMAT,
          "preprocessed_shape": shape.tolist(), "baseline_ct_max_abs_error": maximum_error,
          "baseline_seg_exact": True, "baseline_payload_bytes": minimum_payload_bytes,
          "disk_free_before_bytes": free, "seconds": time.perf_counter() - started}), flush=True)
    return case, relative, digest


def prepare_source_candidates(bank_root, case_id, component_id, case, case_digest,
                              source_ct, source_mask, source_anchor, raw_centers,
                              network_patch_size, *, candidate_map=None):
    """Publish all selected raw placements through a measured ordered mapper."""
    started = time.perf_counter()
    patch_limit = np.asarray(network_patch_size, dtype=np.int64)
    if patch_limit.shape != (3,) or np.any(patch_limit <= 0):
        raise ValueError("Native training patch size must contain three positive dimensions")

    def prepare_one(item):
        index, center = item
        candidate = prepare_candidate(case, source_ct, source_mask, source_anchor, center)
        # Preserve the entire candidate. nnU-Net trains on its original fixed
        # size crops, which may legitimately intersect only part of a lesion.
        # The engine applies the exact full-native result within that crop.
        candidate.update(case_id=str(case_id), source_component=int(component_id),
                         raw_target_center=np.asarray(center, dtype=np.int64).tolist(),
                         case_reference_sha256=case_digest)
        relative = f"raw_candidates/{case_id}__component_{component_id:03d}/{index:04d}.json"
        digest = save_candidate(bank_root, relative, candidate)
        return (relative, digest, int(np.count_nonzero(candidate["pasted_support"])),
                int(candidate["audit"]["extrema_complement_scans"]))

    if candidate_map is None:
        def candidate_map(function, tasks):
            completed = {}

            def execute(task):
                index, value = task
                return index, function(value)

            def commit(result):
                index, value = result
                if index in completed or not 0 <= index < len(tasks):
                    raise ValueError("Raw-target preparation returned a duplicate/invalid candidate")
                completed[index] = value

            report = Path(bank_root) / "preparation_resources" / (
                f"raw_targets.{case_id}.{component_id}.{uuid.uuid4().hex}.json")
            report.parent.mkdir(parents=True, exist_ok=True)
            run_case_jobs(tasks=list(enumerate(tasks)), function=execute, commit=commit,
                          workers="auto", report_path=report)
            if set(completed) != set(range(len(tasks))):
                raise ValueError("Raw-target preparation did not complete every candidate")
            print(f"[RawTargetResources] {report}", flush=True)
            return [completed[index] for index in range(len(tasks))]

    results = list(candidate_map(prepare_one, list(enumerate(raw_centers))))
    if len(results) != len(raw_centers):
        raise ValueError("Raw-target candidate mapper lost a selected candidate")
    audit = {"format": SOURCE_MAPPING_FORMAT, "raw_source_voxels": int(np.count_nonzero(source_mask)),
             "candidates": len(results), "native_support_voxels": [row[2] for row in results],
             "zero_native_support_candidates": sum(row[2] == 0 for row in results),
             "extrema_complement_scans": sum(row[3] for row in results),
             "seconds": time.perf_counter() - started,
             "training_patch_size": patch_limit.tolist(),
             "full_candidate_support_retained": True,
             "training_crop_may_intersect_partial_support": True,
             "source_origin_resampling_used": False, "all_selected_raw_candidates_retained": True}
    print("[RawTargetSource] " + json.dumps({"case_id": case_id, "component": component_id, **audit}), flush=True)
    return np.asarray([row[0] for row in results]), np.asarray([row[1] for row in results]), audit
