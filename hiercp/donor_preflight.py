"""Whole-cohort, header-only donor geometry preflight.

This audit does not read voxel arrays or hash volume files. Expected input
failures are retained for every case, then reported together before expensive
donor eligibility work. Passing headers establish grid equivalence only, not
anatomical correctness or donor label eligibility.
"""
from __future__ import annotations

from pathlib import Path
import zlib

import nibabel as nib
import numpy as np

from hiercp import nifti_geometry
from hiercp.preparation_runtime import _exclusive_report, run_case_jobs


_INPUT_ERRORS = (OSError, EOFError, ValueError, zlib.error, nib.filebasedimages.ImageFileError)


def _stat_signature(path):
    state = Path(path).stat()
    return {"device": state.st_dev, "inode": state.st_ino,
            "size_bytes": state.st_size, "mtime_ns": state.st_mtime_ns,
            "ctime_ns": state.st_ctime_ns}


def _numeric_evidence(value):
    """Retain invalid numeric header evidence without invalid JSON tokens."""
    array = np.asarray(value)
    if array.ndim:
        return [_numeric_evidence(item) for item in array]
    number = float(array)
    if np.isnan(number):
        return "NaN"
    if np.isposinf(number):
        return "+Infinity"
    if np.isneginf(number):
        return "-Infinity"
    return number


def _header_summary(image):
    header = image.header
    summary = {"shape": [int(size) for size in image.shape],
               "header_type": type(header).__name__,
               "selected_affine": _numeric_evidence(image.affine),
               "spatial_pixdim": _numeric_evidence(header["pixdim"][1:4]),
               "qform_code": int(header["qform_code"]),
               "sform_code": int(header["sform_code"])}
    try:
        summary["xyzt_units"] = list(header.get_xyzt_units())
    except (KeyError, ValueError) as error:
        summary["xyzt_units_error"] = f"{type(error).__name__}: {error}"
    summary["selected_form"] = ("sform" if summary["sform_code"] else
                                "qform" if summary["qform_code"] else "base")
    for name in ("qform", "sform"):
        if summary[name + "_code"] == 0:
            summary[name + "_affine"] = None
            summary[name + "_status"] = "unset"
            continue
        try:
            summary[name + "_affine"] = _numeric_evidence(getattr(header, "get_" + name)())
        except (ValueError, np.linalg.LinAlgError) as error:
            summary[name + "_affine"] = None
            summary[name + "_status"] = f"invalid: {type(error).__name__}: {error}"
        else:
            summary[name + "_status"] = "available"
    return summary


def audit_donor_headers(*, case_paths, selected_case_ids, workers, report_path):
    """Audit every selected image/label header, or fail with all invalid cases.

    Returns a geometry-report mapping only when the entire cohort passes.
    Expected corrupt/missing input and geometry errors are explicit failed
    outcomes, not ignored exceptions. Unexpected implementation/runtime errors
    propagate through the measured executor with its failed resource report.
    """
    paths = list(case_paths)
    selected = list(selected_case_ids)
    if (not selected or any(not isinstance(value, str) or not value for value in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("Donor header preflight selected cohort must be nonempty and unique")
    paths_by_id = {item.case_id: item for item in paths}
    if len(paths_by_id) != len(paths) or set(paths_by_id) != set(selected):
        raise ValueError("Donor header preflight input case cohort is not exact")
    base = Path(report_path)

    def inspect(paths):
        row = {"case_id": paths.case_id, "status": "failed", "errors": [],
               "geometry_audit": None, "headers": {}, "files": {}}
        images = {}
        for name in ("image", "label"):
            path = Path(getattr(paths, name + "_path")).absolute()
            file_evidence = {"path": str(path), "before": None, "after": None}
            row["files"][name] = file_evidence
            try:
                file_evidence["before"] = _stat_signature(path)
                images[name] = nib.load(str(path), mmap=True)
                row["headers"][name] = _header_summary(images[name])
            except _INPUT_ERRORS as error:
                row["errors"].append(f"{name} header: {type(error).__name__}: {error}")
        # The guard spans both reads; mutation of the image while reading its
        # label must not go unnoticed. Full content SHA checks follow later.
        for name, evidence in row["files"].items():
            if evidence["before"] is None:
                continue
            try:
                evidence["after"] = _stat_signature(evidence["path"])
                if evidence["after"] != evidence["before"]:
                    row["errors"].append(f"{name} file changed during header preflight")
            except OSError as error:
                row["errors"].append(f"{name} post-header stat: {type(error).__name__}: {error}")
        if not row["errors"]:
            try:
                row["geometry_audit"] = nifti_geometry.validate_donor_geometry(
                    images["image"], images["label"], case_id=paths.case_id)
            except _INPUT_ERRORS as error:
                row["errors"].append(f"geometry: {type(error).__name__}: {error}")
        if not row["errors"]:
            row["status"] = "passed"
        return row

    rows = {}

    def commit(row):
        if row["case_id"] not in paths_by_id or row["case_id"] in rows:
            raise RuntimeError("Donor header preflight returned an unknown or duplicate case")
        rows[row["case_id"]] = row

    resources = base.with_name(f"{base.stem}.resources{base.suffix or '.json'}")
    run_case_jobs(tasks=[paths_by_id[value] for value in selected], function=inspect,
                  commit=commit, workers=workers, report_path=resources)
    if set(rows) != set(selected):
        raise RuntimeError("Donor header preflight executor did not return the full cohort")
    ordered = [rows[value] for value in selected]
    failed = [row for row in ordered if row["status"] != "passed"]
    report = {"format": "hiercp_donor_header_preflight_v1",
              "geometry_policy_version": nifti_geometry.GEOMETRY_POLICY_VERSION,
              "status": "failed" if failed else "complete",
              "selected_case_ids": selected, "configured_cases": len(selected),
              "audited_cases": len(ordered), "passed_cases": len(ordered) - len(failed),
              "failed_case_ids": [row["case_id"] for row in failed],
              "cases": ordered, "workers": workers, "cohort_reduced": False,
              "voxel_arrays_read": False, "full_file_hashes_read": False,
              "completion_semantics": "complete only when every selected header pair passes geometry validation",
              "limitation": "header/grid equivalence is not anatomical registration or donor label eligibility; full source SHA validation follows"}
    target = _exclusive_report(base, report)
    print(f"[DonorHeaderAudit] {target}", flush=True)
    if failed:
        details = "; ".join(f"{row['case_id']}: {'; '.join(row['errors'])}" for row in failed)
        raise ValueError(f"Donor header preflight failed for {len(failed)}/{len(selected)} cases: {details}; report={target}")
    return {value: rows[value]["geometry_audit"] for value in selected}
