"""Bounded, read-only CPU geometry diagnostics for failed native-v1 ROI requests.

This executes no neural model, graph edge construction, cache preparation or
training. The proposed voxel ceiling exists only in the child process's copy of
GraphBuildConfig. Existing source, config, masks and prepared results stay intact.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

FORMAT = "v1_roi_budget_probe_v1"
SCOPE = "debug_geometry_only"


def parse_roi_failure(message: str) -> dict:
    """Extract only the original allocation-guard diagnostic, without eval/code."""
    if not message.startswith("Adaptive ROI request exceeds graph.adaptive_roi_max_voxels;"):
        raise ValueError("Only the original adaptive-native-shape voxel failure can be replayed")
    result = {}
    for name in ("footprint_shape", "spacing_mm", "requested_shape", "effective_shape"):
        found = re.search(rf"\b{name}=(\([^)]*\))", message)
        if found is None:
            raise ValueError(f"ROI diagnostic lacks {name}")
        values = ast.literal_eval(found.group(1))
        if not isinstance(values, tuple) or len(values) != 3:
            raise ValueError(f"Invalid ROI diagnostic {name}")
        if name == "spacing_mm":
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("Invalid ROI spacing")
        elif any(type(v) is not int or v <= 0 for v in values):
            raise ValueError(f"Invalid ROI dimensions: {name}")
        result[name] = list(values)
    for name in ("requested_margin_mm", "requested_voxels", "effective_voxels", "voxel_budget"):
        found = re.search(rf"\b{name}=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", message)
        if found is None:
            raise ValueError(f"ROI diagnostic lacks {name}")
        value = float(found.group(1)) if name == "requested_margin_mm" else int(found.group(1))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid ROI diagnostic {name}")
        result[name] = value
    if result["requested_shape"] != result["effective_shape"]:
        raise ValueError("Historical ROI diagnostic reports reduced geometry")
    if math.prod(result["requested_shape"]) != result["requested_voxels"]:
        raise ValueError("ROI shape/voxel diagnostic mismatch")
    if result["requested_voxels"] != result["effective_voxels"]:
        raise ValueError("Historical ROI diagnostic reports reduced voxel count")
    return result


def failure_requests(path: Path) -> list[dict]:
    """Replay every sample-level ROI resource failure; never silently skip one."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    requests = []
    identities = set()
    for row in rows:
        if row.get("status") != "resource_budget_error":
            continue
        if row.get("sample_index", "") == "":
            raise ValueError("Case-level resource failure cannot be replayed as a source ROI")
        try:
            index = int(row["sample_index"])
        except (ValueError, KeyError) as error:
            raise ValueError("Invalid failed sample index") from error
        case_id = row.get("case_id", "")
        if index < 0 or re.fullmatch(r"liver_[0-9]+", case_id) is None:
            raise ValueError("Invalid failed sample identity")
        identity = (case_id, index)
        if identity in identities:
            raise ValueError("Duplicate failed sample identity")
        identities.add(identity)
        requests.append({"case_id": case_id, "sample_index": index,
                         "original_failure": row["message"],
                         "geometry": parse_roi_failure(row["message"]),
                         "source_image_sha256": row.get("source_image_sha256"),
                         "source_label_sha256": row.get("source_label_sha256")})
    if not requests:
        raise ValueError("No failed sample-level ROI allocation requests in manifest")
    return sorted(requests, key=lambda r: (r["case_id"], r["sample_index"]))


def _write_new(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def summarize_probe_failure(row: dict, *, message_limit: int = 480) -> dict:
    """Summarize recorded child output; never infer an exception from ROI shape.

    The full traceback and stdout remain in ``row['error']`` / ``row['stdout']``.
    Only this terminal summary is bounded. Unclassified child output is labeled
    explicitly, and a resource stop is distinguished from a child exception.
    """
    if type(message_limit) is not int or message_limit < 1:
        raise ValueError("Failure display message limit must be a positive integer")
    if row.get("status") == "PASS":
        raise ValueError("A successful probe has no failure summary")
    stderr, stdout = row.get("error", ""), row.get("stdout", "")
    if not isinstance(stderr, str) or not isinstance(stdout, str):
        raise TypeError("Recorded child stderr/stdout must be text")
    source, error_class, message = None, None, None
    marker = "Traceback (most recent call last):"
    if marker in stderr:
        # In a chained traceback the final block contains the propagated error.
        lines = stderr.rsplit(marker, 1)[1].strip().splitlines()
        pattern = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?::(?:\s*(.*))?)?$")
        matches = [(index, pattern.match(line)) for index, line in enumerate(lines)]
        matches = [(index, match) for index, match in matches
                   if match is not None and (":" in lines[index] or
                       re.fullmatch(r"(?:[A-Za-z_]\w*\.)*(?:[A-Za-z_]\w*(?:Error|Exception)|KeyboardInterrupt|SystemExit)", match.group(1)))]
        if matches:
            # Later lines belong to a multiline exception message. A line such
            # as ``Tip: ...`` must not replace the actual exception header.
            index, match = matches[0]
            error_class = match.group(1)
            message = "\n".join([match.group(2) or "", *lines[index+1:]]).strip()
            source = "stderr_traceback"
    if source is None:
        if stderr.strip():
            source, message = "stderr_unclassified", stderr.strip().splitlines()[-1]
        elif stdout.strip():
            source, message = "stdout_unclassified", stdout.strip().splitlines()[-1]
        elif row.get("status") in ("RSS_BUDGET", "TIME_BUDGET"):
            source = "diagnostic_resource_budget"
            message = "Child stopped at the explicitly configured " + row["status"]
        elif row.get("answer_missing"):
            source = "child_exit_without_answer"
            message = f"Child returned {row.get('returncode')} without an answer file or diagnostic output"
        else:
            source = "child_exit_without_output"
            message = f"Child returned {row.get('returncode')} without diagnostic output; cause is unknown"
    # Collapse whitespace for one terminal line; this does not alter raw output.
    compact = " ".join(message.split())
    truncated = len(compact) > message_limit
    if truncated:
        compact = compact[:message_limit] + " ... [full output in report]"
    return dict(source=source, error_class=error_class, message=compact,
                message_truncated=truncated)


def format_probe_result(row: dict) -> str:
    """Format one result, including an honest failure reason when available."""
    line = (f"CPU ROI DEBUG | {row['case_id']}[{row['sample_index']}] | {row['status']} | "
            f"RSS={row['peak_rss_bytes']/2**30:.3f} GiB | wall={row['wall_seconds']:.2f}s")
    if row["status"] != "PASS":
        details = summarize_probe_failure(row)
        label = details["error_class"] or details["source"]
        line += f" | {label}"
        if details["message"]:
            line += ": " + details["message"]
    return line


def worker(request_path: str) -> None:
    """Child entry: import the verified frozen snapshot before any hiercp module."""
    from dataclasses import replace
    import hashlib
    import numpy as np
    import psutil
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    snapshot = Path(request["snapshot"]).resolve()
    sys.path.insert(0, str(snapshot))
    from hiercp.common import (CasePaths, choose_source_tumor, load_case,
                               stable_case_seed, ct_normalize, erase_mask_with_context,
                               verify_loaded_case_source_signatures)
    from hiercp.region import (load_patient_regions, _region_cache_metadata,
                              _metadata_equal, REGION_CACHE_SEED_SALT)
    from hiercp.schema import graph_config_from_dict
    from hiercp.spatial import (adaptive_native_shape, AdaptiveRoiBudgetError,
                               build_patch_payload, exact_source_footprint)
    import hiercp.spatial
    if Path(hiercp.spatial.__file__).resolve() != snapshot / "hiercp/spatial.py":
        raise ValueError("Worker imported a non-frozen spatial implementation")
    spatial_hash = hashlib.sha256((snapshot / "hiercp/spatial.py").read_bytes()).hexdigest()
    if spatial_hash != request["spatial_sha256"]:
        raise ValueError("Frozen spatial source changed before child execution")
    cfg = request["config"]
    original = graph_config_from_dict(cfg["graph"])
    proposed = replace(original, adaptive_roi_max_voxels=request["candidate_roi_max_voxels"])
    proposed.validate()
    row = request["failure"]
    case_id, index = row["case_id"], row["sample_index"]
    data = Path(request["medical_root"]) / "Data"
    load_start = time.perf_counter()
    case = load_case(CasePaths(case_id, data / "image" / f"{case_id}_0000.nii.gz",
                              data / "labels" / f"{case_id}.nii.gz"))
    for name in ("image", "label"):
        if getattr(case, f"{name}_source_signature")["sha256"] != row[f"source_{name}_sha256"]:
            raise ValueError(f"Failed manifest {name} content does not match actual CT")
    region_root = Path(request["shared"]) / "regions" / case_id
    regions, metadata = load_patient_regions(region_root, mmap=True)
    expected = _region_cache_metadata(case, liver_label=cfg["labels"]["liver"],
        tumor_label=cfg["labels"]["tumor"], config=original, ct_clip=tuple(cfg["ct_clip"]),
        seed=stable_case_seed(cfg["seed"], case_id, REGION_CACHE_SEED_SALT))
    if not _metadata_equal(metadata, expected):
        raise ValueError("Existing region cache does not match original case/config contract")
    source, components, _ = choose_source_tumor(case.image, case.label,
        tumor_label=cfg["labels"]["tumor"],
        rng=np.random.default_rng(stable_case_seed(cfg["seed"], case_id, f"sample_{index}")),
        selection=cfg["cache"]["source_selection"], pad=cfg["cache"]["source_pad"])
    del components
    footprint = exact_source_footprint(source)
    geometry = row["geometry"]
    if list(footprint.shape) != geometry["footprint_shape"]:
        raise ValueError("Failed ROI is not the replayed original source footprint; target replay is required")
    depth = float(regions.organ_depth[source.anchor_center])
    try:
        adaptive_native_shape(footprint.shape, case.spacing, original,
                              center_liver_depth_mm=depth)
    except AdaptiveRoiBudgetError as error:
        replayed = parse_roi_failure(str(error))
    else:
        raise ValueError("Original failed ROI guard did not reproduce on this source")
    if replayed != geometry:
        raise ValueError("Original source ROI does not exactly match failed manifest geometry")
    requested = adaptive_native_shape(footprint.shape, case.spacing, proposed,
                                      center_liver_depth_mm=depth)
    if list(requested) != geometry["requested_shape"]:
        raise ValueError("Proposed admission changed requested geometry")
    load_seconds = time.perf_counter() - load_start
    before_rss = psutil.Process().memory_info().rss
    start = time.perf_counter()
    payload = build_patch_payload(image=case.image, center=source.anchor_center,
        footprint=footprint, full_organ=regions.full_organ_mask,
        organ_depth=regions.organ_depth, spacing=case.spacing, config=proposed,
        erase_target=False, ct_clip=tuple(cfg["ct_clip"]),
        ct_normalize_fn=ct_normalize, erase_fn=erase_mask_with_context)
    elapsed = time.perf_counter() - start
    # This source probe must not silently claim a bigger liver-anchor expansion
    # matches the originally rejected shape.
    actual_shape = list(payload["footprint"].shape)
    geometry_equal = actual_shape == geometry["requested_shape"]
    if not geometry_equal:
        raise ValueError("Actual liver-anchor expansion differs from the initial failed ROI; measure a separate explicit budget")
    original_count = int(np.count_nonzero(footprint))
    if int(np.count_nonzero(payload["footprint"])) != original_count:
        raise ValueError("ROI probe changed original full paste footprint")
    unique_arrays = {id(value): value for value in payload.values() if isinstance(value, np.ndarray)}
    if any(not np.all(np.isfinite(value)) for value in unique_arrays.values()):
        raise ValueError("Non-finite original ROI geometry payload")
    verify_loaded_case_source_signatures(case)
    peak_rss = None
    if sys.platform != "win32":
        import resource
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
    answer = dict(case_id=case_id, sample_index=index, status="PASS",
        cpu_geometry_only=True, source_component=int(source.component_id),
        source_anchor=list(source.anchor_center), original_guard_reproduced=True,
        original_requested_shape=geometry["requested_shape"], requested_shape=list(requested),
        effective_shape=actual_shape, geometry_equal=geometry_equal,
        original_requested_voxels=geometry["requested_voxels"],
        roi_voxels=math.prod(actual_shape), full_mask_voxels=original_count,
        model_input_shape=list(payload["model_input"].shape), load_seconds=load_seconds,
        geometry_seconds=elapsed, retained_payload_bytes=sum(v.nbytes for v in unique_arrays.values()),
        rss_before_geometry_bytes=before_rss,
        rss_after_geometry_bytes=psutil.Process().memory_info().rss,
        process_peak_rss_bytes=peak_rss,
        spatial_sha256=spatial_hash, source_image_sha256=row["source_image_sha256"],
        source_label_sha256=row["source_label_sha256"],
        limitation="Source ROI fields only; target transforms, canonical nodes/edges and full-cache admission are not measured")
    _write_new(Path(request["answer_path"]), answer)


def run_probe(experiment: str, output: str, *, candidate_voxels: int,
              rss_bytes: int, case_timeout_seconds: float) -> dict:
    from .contracts import V1_ARCHIVE_SHA256
    from .experiment import load_suite, preparation_root, digest
    import psutil
    if type(candidate_voxels) is not int or candidate_voxels <= 0:
        raise ValueError("Candidate ROI voxel budget must be an explicit positive integer")
    if type(rss_bytes) is not int or rss_bytes <= 0:
        raise ValueError("Explicit RSS resource budget must be positive")
    if not math.isfinite(case_timeout_seconds) or case_timeout_seconds <= 0:
        raise ValueError("Explicit per-case timeout must be finite and positive")
    root = Path(experiment).resolve()
    manifest = load_suite(root)
    if manifest.get("sampling_contract", {}).get("mode") != "native":
        raise ValueError("ROI diagnostic must use the original native preparation owner")
    shared = preparation_root(root, manifest)
    progress = shared / "cache" / "manifest.csv"
    requests = failure_requests(progress)
    cfg_path = root / "configs" / "v1.0.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    original_voxels = cfg["graph"]["adaptive_roi_max_voxels"]
    if original_voxels != 8_000_000 or candidate_voxels <= original_voxels:
        raise ValueError("Probe requires original 8000000 guard and an explicit larger candidate")
    if any(r["geometry"]["voxel_budget"] != original_voxels for r in requests):
        raise ValueError("Failed manifest belongs to a different ROI allocation guard")
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError("Existing probe report is preserved; choose a new output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = destination.with_name(destination.stem + "_requests")
    work.mkdir(exist_ok=False)
    snapshot = root / "source" / "v1.0"
    spatial_sha = digest(snapshot / "hiercp/spatial.py")
    watched = [root / "manifest.json", cfg_path, progress]
    signatures = {str(p): digest(p) for p in watched}
    rows = []
    tool = Path(__file__).resolve().parents[1] / "tools" / "probe_v1_roi_budget.py"
    for ordinal, failure in enumerate(requests):
        request_path = work / f"{ordinal:03d}_request.json"
        answer_path = work / f"{ordinal:03d}_answer.json"
        _write_new(request_path, dict(snapshot=str(snapshot), config=cfg,
            candidate_roi_max_voxels=candidate_voxels, failure=failure,
            medical_root=manifest["medical_root"], shared=str(shared),
            spatial_sha256=spatial_sha, answer_path=str(answer_path)))
        command = [sys.executable, "-u", str(tool), "--worker-request", str(request_path)]
        print(f"CPU ROI DEBUG | {ordinal+1}/{len(requests)} | {failure['case_id']}[{failure['sample_index']}]", flush=True)
        started = time.perf_counter()
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace")
        observed_peak = 0
        stopped = None
        try:
            while proc.poll() is None:
                try:
                    observed_peak = max(observed_peak, psutil.Process(proc.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    break  # Collect the child's actual return code/output below.
                elapsed = time.perf_counter() - started
                if observed_peak > rss_bytes or elapsed > case_timeout_seconds:
                    stopped = "RSS_BUDGET" if observed_peak > rss_bytes else "TIME_BUDGET"
                    print(f"Stopping owned diagnostic child PID {proc.pid}: {stopped}; command={command!r}", flush=True)
                    proc.terminate()
                    break
                time.sleep(0.05)
            stdout, stderr = proc.communicate()
        except KeyboardInterrupt:
            if proc.poll() is None:
                print(f"Stopping owned diagnostic child PID {proc.pid}: user interrupted ROI probe", flush=True)
                proc.terminate()
            proc.communicate()
            raise
        if stopped or proc.returncode or not answer_path.is_file():
            row = dict(case_id=failure["case_id"], sample_index=failure["sample_index"],
                       status=stopped or "FAILED", returncode=proc.returncode,
                       peak_rss_bytes=observed_peak, wall_seconds=time.perf_counter()-started,
                       error=stderr, stdout=stdout, answer_missing=not answer_path.is_file())
        else:
            row = json.loads(answer_path.read_text(encoding="utf-8"))
            row["peak_rss_bytes"] = max(observed_peak, row["process_peak_rss_bytes"] or 0)
            row["wall_seconds"] = time.perf_counter()-started
            if row["peak_rss_bytes"] > rss_bytes:
                row["status"] = "RSS_BUDGET"
        if row["status"] != "PASS":
            row["failure_summary"] = summarize_probe_failure(row)
        rows.append(row)
        print(format_probe_result(row), flush=True)
    preserved = all(digest(Path(name)) == sha for name, sha in signatures.items())
    report = dict(format=FORMAT, scope=SCOPE, archived_source_sha256=V1_ARCHIVE_SHA256,
        original_roi_max_voxels=original_voxels, candidate_roi_max_voxels=candidate_voxels,
        experiment=str(root), failed_manifest_sha256=signatures[str(progress)],
        resource_budget=dict(rss_bytes=rss_bytes, case_timeout_seconds=case_timeout_seconds,
                             rss_sampling_interval_seconds=0.05, concurrent_cases=1,
                             concurrency_reason="Isolated whole-process peak measurement per full failing source ROI"),
        failed_sample_requests=len(requests), measurements=rows,
        completed=preserved and all(row["status"] == "PASS" for row in rows),
        originals_preserved=preserved, training_started=False, cache_publication_created=False,
        production_ready=False, quality_verified=False,
        limitation="Bounds source geometry cost only; full candidates, graph N/E and full-cache capacity remain unverified")
    _write_new(destination, report)
    return report
