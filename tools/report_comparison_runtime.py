"""Read comparison runtime logs without opening checkpoints or changing a run.

The report separates epoch timings, shared cache history, partial live scores,
and completed full-validation scores. Old logs without an explicit phase keep
the exact phase unknown. Timings are observed work, not a remaining-time promise.
This utility uses only the Python standard library and writes only to stdout.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


ARMS = ("selected", "native", "native_fixed", "native_listwise")
UPDATE_KEYS = ("status", "epoch", "update", "attempt", "sample_indices", "physical_samples",
               "candidate_rows", "loader_wait_seconds", "loader_seconds", "batch_wall_seconds", "step_seconds", "checkpoint_seconds",
               "transfer_seconds", "forward_seconds", "backward_seconds", "check_clip_optimizer_seconds",
               "retry_seconds", "peak_cuda_bytes", "peak_reserved_bytes", "rss_bytes", "cpu_percent",
               "loss", "ranking", "consistency", "margin", "train7_pair_win")
VALIDATION_KEYS = ("epoch", "source_indices", "source_problems", "candidate_rows", "loader_seconds",
                   "loader_wait_seconds", "batch_wall_seconds", "full_joint_forward_seconds", "checkpoint_seconds", "peak_cuda_bytes", "rss_bytes")
CACHE_KEYS = ("stage", "kind", "helper_kind", "status", "case_id", "wall_seconds", "bytes",
              "array_bytes", "disk_bytes", "rss_before_bytes", "rss_after_bytes")
INPUT_KEYS = ("format", "observed_at", "arm", "view_epoch", "training", "full129", "source_indices",
              "status", "input_seconds", "regions", "other_assembly_views_collate_seconds",
              "process_cpu_seconds", "process_cpu_scope", "rss_bytes", "timing_scope",
              "inclusive_details", "inclusive_details_scope", "compact_upper_cache",
              "overlapping_input_batches_possible")
INPUT_REGIONS = ("case_fields_seconds", "regions_seconds", "source_seconds",
                 "candidate_metadata_seconds", "local_graphs_seconds")


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant is not valid telemetry: {value}")


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _json(text):
    return json.loads(text, parse_constant=_invalid_constant)


def _compact(row, kind):
    if kind == "update":
        return {key: row[key] for key in UPDATE_KEYS if key in row}
    if kind == "validation":
        return {key: row[key] for key in VALIDATION_KEYS if key in row}
    if kind == "cache":
        return {key: row[key] for key in CACHE_KEYS if key in row}
    if kind == "input":
        return {key: row[key] for key in INPUT_KEYS if key in row}
    if kind == "curve":
        result = {key: row[key] for key in ("epoch", "update", "optimization_loader_save_seconds",
                  "validation_seconds", "epoch_wall_seconds", "wall_scope") if key in row}
        validation = row.get("validation129", {})
        result["validation129"] = {key: validation[key] for key in
            ("metrics", "patients", "source_problems", "metric_weighting", "scope", "epoch", "wall_seconds")
            if key in validation}
        return result
    return row


def parse_jsonl(stream, *, origin="UNIT stream", kind=None):
    """Ignore only an unfinished last append; reject corruption in complete lines."""
    rows, warnings = [], []
    for line_number, line in enumerate(stream, 1):
        if not line.strip():
            continue
        try:
            row = _json(line)
        except json.JSONDecodeError as error:
            if not line.endswith("\n") and stream.read(1) == "":
                warnings.append(f"{origin}:{line_number}: unfinished final append ignored")
                continue
            raise ValueError(f"{origin}:{line_number}: invalid complete JSONL line: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{origin}:{line_number}: expected a JSON object")
        rows.append(_compact(row, kind))
    return rows, warnings


def _read_jsonl(path, kind, warnings):
    if not path.exists():
        warnings.append(f"Missing optional log: {path}")
        return []
    with path.open("r", encoding="utf-8") as stream:
        rows, notices = parse_jsonl(stream, origin=str(path), kind=kind)
    warnings.extend(notices)
    return rows


def _read_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        row = json.load(stream, parse_constant=_invalid_constant)
    if not isinstance(row, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return row


def _duration(rows, key):
    values = [_number(row.get(key)) for row in rows]
    present = [float(value) for value in values if value is not None]
    if any(value < 0 for value in present):
        raise ValueError(f"Negative duration in {key}")
    return {"seconds": sum(present) if present else None, "records_with_value": len(present),
            "records_total": len(rows), "mean_seconds": statistics.mean(present) if present else None,
            "max_seconds": max(present) if present else None,
            "complete_measurement": bool(rows) and len(present) == len(rows)}


def _complete_seconds(duration):
    return duration["seconds"] if duration["complete_measurement"] else None


def _sum_complete(*values):
    return sum(values) if all(value is not None for value in values) else None


def _percent(value, total):
    return 100. * value / total if value is not None and total is not None and total > 0 else None


def _coverage(rows, ids_key, count_key, total):
    observations = []
    explicit = all(isinstance(row.get(ids_key), list) for row in rows)
    for row in rows:
        if isinstance(row.get(ids_key), list):
            observations.extend(row[ids_key])
    # A coverage percentage requires source identities, not a sum that might
    # count a repeated batch after an interrupted append/checkpoint boundary.
    unique = {json.dumps(value, sort_keys=True) for value in observations}
    count = len(unique) if explicit and rows else None
    declared = [_number(row.get(count_key)) for row in rows]
    return {"unique_logged_sources": count, "total_configured_sources": total,
            "percent": _percent(count, total), "source_observations": len(observations),
            "repeated_source_observations": len(observations) - len(unique),
            "summed_declared_source_count": sum(v for v in declared if v is not None) if any(v is not None for v in declared) else None,
            "scope": "logged source coverage; checkpoint cursor may be ahead of the last appended timing row"}


def _peak(rows, key):
    values = [_number(row.get(key)) for row in rows]
    return max(value for value in values if value is not None) if any(value is not None for value in values) else None


def summarize_training(rows, total_sources):
    successful = [row for row in rows if row.get("status") == "OPTIMIZER_UPDATED"]
    retries = [row for row in rows if row.get("status") == "AMP_OVERFLOW_SKIPPED_RETRY_SAME_INPUT"]
    timing = {key: _duration(successful, key) for key in
              ("loader_wait_seconds", "loader_seconds", "batch_wall_seconds", "step_seconds", "checkpoint_seconds", "transfer_seconds",
               "forward_seconds", "backward_seconds", "check_clip_optimizer_seconds")}
    observed = _sum_complete(_complete_seconds(timing["loader_wait_seconds"]), _complete_seconds(timing["step_seconds"]))
    for key, value in timing.items():
        value["percent_of_observed_loader_plus_step"] = _percent(value["seconds"], observed) if key != "loader_seconds" else None
    coverage = _coverage(successful, "sample_indices", "physical_samples", total_sources)
    physical_sources = coverage["summed_declared_source_count"]
    return {"successful_update_records": len(successful), "overflow_retry_records": len(retries),
            "coverage": coverage, "timings": timing, "observed_loader_plus_step_seconds": observed,
            "observed_sources_per_second": physical_sources / observed if physical_sources is not None and observed and observed > 0 else None,
            "retry_seconds": _duration(retries, "retry_seconds"),
            "peak_cuda_bytes": _peak(successful, "peak_cuda_bytes"), "peak_rss_bytes": _peak(successful, "rss_bytes"),
            "max_process_cpu_percent": _peak(successful, "cpu_percent"),
            "latest_loss": successful[-1].get("loss") if successful else None,
            "timing_semantics": ["step_seconds starts after the loader wait and already includes checkpoint_seconds.",
                "Add loader_wait_seconds + step_seconds once; do not add checkpoint or CUDA substage durations again.",
                "Loader wait is observed blocking time with prefetch; it is not total CPU preparation time.",
                "A successful step already includes preceding AMP retry attempts for that batch; retry_seconds is nested detail, not an added total.",
                "A paused/failed batch with retry rows but no successful step is absent from successful-step totals.",
                "Process CPU percent can exceed 100 when several cores are used; it is not whole-machine utilization."]}


def summarize_validation(rows, total_sources, completed_curve=None):
    timing = {key: _duration(rows, key) for key in ("loader_seconds", "loader_wait_seconds",
        "batch_wall_seconds", "full_joint_forward_seconds", "checkpoint_seconds")}
    blocking_rows = []
    modern = 0
    for row in rows:
        if "loader_wait_seconds" in row or "batch_wall_seconds" in row:
            modern += 1
            blocking = row.get("loader_wait_seconds")
        else:
            # Frozen execution used synchronous provider.batch. Its loader
            # duration is blocking work. Prefetch execution records both values.
            blocking = row.get("loader_seconds")
        blocking_rows.append(dict(blocking_loader_seconds=blocking))
    timing["blocking_loader_seconds"] = _duration(blocking_rows, "blocking_loader_seconds")
    observed = _sum_complete(_complete_seconds(timing["blocking_loader_seconds"]),
                             _complete_seconds(timing["full_joint_forward_seconds"]))
    for key, value in timing.items():
        value["percent_of_observed_blocking_load_plus_forward"] = _percent(value["seconds"], observed) if key in ("blocking_loader_seconds", "full_joint_forward_seconds") else None
    complete = completed_curve or {}
    return {"batch_records": len(rows), "coverage": _coverage(rows, "source_indices", "source_problems", total_sources),
            "timings": timing, "observed_blocking_load_plus_forward_seconds": observed,
            "observed_batch_wall_seconds": _complete_seconds(timing["batch_wall_seconds"]),
            "prefetch_batch_records": modern, "legacy_synchronous_batch_records": len(rows) - modern,
            "reported_completed_validation_active_seconds": complete.get("validation_seconds"),
            "peak_cuda_bytes": _peak(rows, "peak_cuda_bytes"), "peak_rss_bytes": _peak(rows, "rss_bytes"),
            "timing_semantics": ["Legacy validation loader_seconds is synchronous blocking loading; prefetch rows separate CPU construction from blocking loader_wait_seconds.",
                "Prefetch loader_seconds can overlap GPU work and must not be added to forward or batch_wall_seconds.",
                "Blocking load + forward is a lower bound excluding metric bookkeeping and checkpoint time.",
                "New batch_wall_seconds already includes blocking wait, forward, metric bookkeeping, and checkpoint time.",
                "Completed validation_seconds includes the active validation/checkpoint time; it is a separate total.",
                "Summed rows include any repeated work; unique source coverage is reported separately."]}


def summarize_cache(rows, *, available=True):
    if not available:
        return {"available": False, "scope": "cache history unavailable"}
    fields = [row for row in rows if row.get("stage") == "whole_case_fields_read"]
    stages = Counter(str(row.get("stage", "unlabelled")) for row in rows)
    field_statuses = Counter(str(row.get("status", "unknown")) for row in fields)
    reuse_statuses = Counter(str(row.get("status", "unknown")) for row in rows if row.get("stage") != "whole_case_fields_read")
    statuses = {}
    for status in field_statuses:
        matching = [row for row in fields if str(row.get("status", "unknown")) == status]
        statuses[status] = {"calls": len(matching), "timing": _duration(matching, "wall_seconds")}
    return {"available": True, "scope": "entire selected JSONL history, possibly multiple invocations/arms; not this epoch or ETA",
            "events": len(rows), "stages": dict(stages), "reuse_statuses": dict(reuse_statuses),
            "field_calls": len(fields), "unique_field_case_ids": len({r["case_id"] for r in fields if "case_id" in r}),
            "field_statuses": statuses, "summed_field_helper_seconds": _duration(fields, "wall_seconds"),
            "pressure_events": stages.get("host_cache_pressure", 0),
            "notes": ["field call counts are neither unique cases nor new distance-field builds.",
                "reopened includes full field-file integrity/finite-value checks; resident_mapping is a different path.",
                "Summed helper times can overlap other work and must not be added to epoch loader timing."]}


def _input_details(rows):
    names = set()
    observed = 0
    for row in rows:
        details = row.get('inclusive_details')
        if details is None:
            continue
        if not isinstance(details, dict) or any(not isinstance(item, dict) for item in details.values()):
            raise ValueError('Inclusive input details must map names to measurements')
        observed += 1
        names.update(details)
    timings = {}
    for name in sorted(names):
        items = [row.get('inclusive_details', {}).get(name, {}) for row in rows]
        counts = [item['calls'] for item in items if 'calls' in item]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError('Inclusive helper call counts must be nonnegative integers')
        timings[name] = dict(timing=_duration(items, 'seconds'),
                             observed_calls=sum(counts) if counts else None)
    return dict(scope='inclusive nested wall times; overlap parent regions and each other; do not add',
                records_with_value=observed, records_total=len(rows), timings=timings)


def _compact_upper_summary(rows):
    totals, observed = {}, 0
    for row in rows:
        cache = row.get('compact_upper_cache')
        if cache is None:
            continue
        if not isinstance(cache, dict):
            raise ValueError('Compact upper cache deltas must be a mapping')
        observed += 1
        for kind, metrics in cache.items():
            if not isinstance(metrics, dict):
                raise ValueError('Compact upper cache kind metrics must be a mapping')
            current = totals.setdefault(kind, {})
            for name, value in metrics.items():
                value = _number(value)
                if value is None or value < 0:
                    raise ValueError('Compact upper cache deltas must be finite and nonnegative')
                if not name.endswith('_seconds') and int(value) != value:
                    raise ValueError('Compact upper cache counts must be integers')
                current[name] = current.get(name, 0) + value
    return dict(scope='sums of observed per-batch deltas; cache times overlap helper/input times',
                records_with_value=observed, records_total=len(rows), kinds=totals)


def _input_mode(rows, scope):
    successful = [row for row in rows if row.get("status") == "complete"]
    failed = [row for row in rows if row.get("status") == "failed"]
    totals = _duration(successful, "input_seconds")
    denominator = _complete_seconds(totals)
    # Preserve additional measured regions, including sampled_views_seconds.
    # The instrumented provider can add disjoint regions without discarding
    # their time here or silently folding it into the remainder.
    extra_regions = set()
    accounting_rows = []
    remainder_key = "other_assembly_views_collate_seconds"
    for row in successful:
        regions = row.get("regions")
        if regions is not None and not isinstance(regions, dict):
            raise ValueError("Input timing regions must be a JSON object")
        extra_regions.update((regions or {}).keys())
        values = [_number(value) for value in (regions or {}).values()]
        remainder, elapsed = _number(row.get(remainder_key)), _number(row.get("input_seconds"))
        accounted = sum(values) + remainder if regions and all(value is not None for value in values) and remainder is not None else None
        if elapsed is not None and accounted is not None and not math.isclose(accounted, elapsed, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Disjoint input timing regions and remainder do not sum to input_seconds")
        accounting_rows.append(dict(accounted_seconds=accounted))
    timings = {}
    region_keys = (*INPUT_REGIONS, *sorted(extra_regions.difference(INPUT_REGIONS)))
    for key in region_keys:
        region_rows = [{key: row.get("regions", {}).get(key)} for row in successful]
        timings[key] = _duration(region_rows, key)
    timings[remainder_key] = _duration(successful, remainder_key)
    for duration in timings.values():
        duration["percent_of_cpu_input_construction"] = _percent(_complete_seconds(duration), denominator)
    # Missing region entries remain unknown: a zero is justified only by a
    # logged zero, not by an absent field in an older instrumentation version.
    # Audit every row before summing; positive and negative inconsistencies
    # must not cancel across batches. A newer optional region absent from old
    # rows remains unmeasured there, while each row's own disjoint sum is exact.
    accounting = _duration(accounting_rows, "accounted_seconds")
    accounted = _complete_seconds(accounting)
    return {"scope": scope, "completed_batch_records": len(successful), "failed_batch_records": len(failed),
            "view_epochs": sorted({row["view_epoch"] for row in rows if type(row.get("view_epoch")) is int}),
            "source_coverage": _coverage(successful, "source_indices", "source_problems", None),
            "input_construction": totals, "timings": timings,
            "inclusive_details": _input_details(successful),
            "compact_upper_cache": _compact_upper_summary(successful),
            "accounted_region_plus_remainder_seconds": accounted,
            "input_accounting": accounting,
            "failed_input_construction": _duration(failed, "input_seconds"),
            "peak_rss_bytes": _peak(rows, "rss_bytes")}


def summarize_input(rows, arm, selected_epoch, *, available=True):
    if not available:
        return {"available": False, "scope": "CPU input construction history unavailable"}
    selected_arm = [row for row in rows if arm is None or row.get("arm") == arm]
    training = [row for row in selected_arm if row.get("training") is True and row.get("view_epoch") == selected_epoch]
    validation = [row for row in selected_arm if row.get("training") is False and row.get("full129") is True]
    return {"available": True,
            "training_epoch": _input_mode(training, "training-mode input history for the selected view_epoch; may include calibration and repeated/resumed work"),
            "validation_history": _input_mode(validation, "full129 input rows across the selected log history; fixed view_epoch is not the outer training epoch"),
            "other_or_unclassified_arm_records": len(selected_arm) - len(training) - len(validation),
            "timing_semantics": ["CPU batch construction excludes memory pinning and queue wait.",
                "Disjoint region wall times plus assembly/view/collation remainder sum to input construction time.",
                "Input construction may overlap GPU work with prefetch; do not add this total to loader wait or step time.",
                "Memory-admitted cached batches can also overlap one another; summed input durations are work totals, not epoch elapsed time.",
                "Training-mode view_epoch=1 construction can include calibration probes; records do not certify optimizer-only input time.",
                "Full129 validation uses a fixed view_epoch across outer epochs; its input history is cumulative, not an epoch total.",
                "Process CPU seconds describe the whole process during construction and may overlap the training thread."]}


def _snapshot_age(progress):
    updated = progress.get("updated_at")
    if not isinstance(updated, str):
        return None
    try:
        observed = datetime.fromisoformat(updated.replace("Z", "+00:00"))
    except ValueError:
        return None
    if observed.tzinfo is None:
        return None
    return (datetime.now(timezone.utc) - observed).total_seconds()


def _expected_phase(training, validation, completed_epoch):
    train = training["coverage"]
    val = validation["coverage"]
    if completed_epoch:
        return "epoch metrics completed; a later phase needs an explicit progress snapshot"
    if val["unique_logged_sources"] is not None:
        if val["total_configured_sources"] and val["unique_logged_sources"] >= val["total_configured_sources"]:
            return "all validation sources logged; metric aggregation/checkpoint/transition may remain"
        return "validation source work observed; current exact phase is unconfirmed"
    if train["unique_logged_sources"] is not None:
        if train["total_configured_sources"] and train["unique_logged_sources"] >= train["total_configured_sources"]:
            return "optimizer source coverage complete; validation/transition expected"
        return "optimizer source work observed; current exact phase is unconfirmed"
    return "unknown"


def build_report(contract, updates, validations, curves, cache_rows, *, progress=None,
                 final_report=None, epoch=None, latest_validation=None, input_rows=None,
                 cache_available=True, input_available=False):
    contract = contract or {}
    progress = progress or {}
    final_report = final_report or {}
    observed_epochs = [row["epoch"] for row in updates + validations + curves if type(row.get("epoch")) is int]
    if type(progress.get("epoch")) is int:
        observed_epochs.append(progress["epoch"])
    selected_epoch = epoch if epoch is not None else max(observed_epochs, default=None)
    epoch_updates = [row for row in updates if row.get("epoch") == selected_epoch]
    epoch_validation = [row for row in validations if row.get("epoch") == selected_epoch]
    completed = next((row for row in reversed(curves) if row.get("epoch") == selected_epoch), None)
    training = summarize_training(epoch_updates, _number(contract.get("training_samples")))
    validation = summarize_validation(epoch_validation, _number(contract.get("validation_samples")), completed)
    latest_scores = None
    if latest_validation:
        latest_scores = {key: latest_validation[key] for key in
                         ("epoch", "metrics", "metric_weighting", "scope", "patients", "source_problems", "wall_seconds")
                         if key in latest_validation}
        latest_scores["scope_note"] = "completed full-validation file; source-anchor ranking proxy, not CP/clinical accuracy"
    elif curves:
        latest_scores = dict(curves[-1].get("validation129", {}))
        latest_scores["epoch"] = curves[-1].get("epoch")
        latest_scores["scope_note"] = "completed curve validation; source-anchor ranking proxy, not CP/clinical accuracy"
    # inspect_arm has already rejected stale final reports. A valid terminal
    # report is newer evidence than the preceding live phase snapshot.
    explicit_phase = final_report.get("final_phase") or progress.get("phase")
    successful = [row for row in updates if row.get("status") == "OPTIMIZER_UPDATED"]
    update_values = [_number(row.get("update")) for row in successful]
    last_update = max((value for value in update_values if value is not None), default=None)
    planned = _number(contract.get("total_planned_updates"))
    live = {key: progress[key] for key in ("format", "arm", "phase", "epoch", "epochs", "stage", "status",
            "completed_sources", "total_sources", "physical_batch", "metrics_scope", "metrics", "timings",
            "timing_scope", "elapsed_seconds", "stage_elapsed_seconds", "updated_at") if key in progress}
    live["percent"] = _percent(_number(progress.get("completed_sources")), _number(progress.get("total_sources")))
    live["snapshot_age_seconds"] = _snapshot_age(progress)
    return {"selected_epoch": selected_epoch,
            "phase": explicit_phase or "unknown", "phase_source": "final report" if final_report.get("final_phase") else
                "progress.json" if progress.get("phase") else "no explicit phase in legacy logs",
            "process_liveness": "not checked; phase is the latest reported observation",
            "expected_next_phase_from_legacy_records": _expected_phase(training, validation, completed),
            "live_progress": live, "arm": contract.get("arm"), "debug": contract.get("debug"),
            "terminal_status": final_report.get("status"),
            "physical_sample_batch": contract.get("physical_sample_batch"), "workers": contract.get("workers"),
            "optimizer_history": {"last_logged_update": last_update, "total_planned_updates": planned,
                                  "percent": _percent(last_update, planned), "successful_update_records": len(successful)},
            "training_epoch": training, "validation_epoch": validation,
            "latest_completed_validation": latest_scores, "cache_history": summarize_cache(cache_rows, available=cache_available),
            "input_preparation": summarize_input(input_rows or [], contract.get("arm"), selected_epoch, available=input_available)}


def inspect_arm(arm_directory, *, cache_log=None, input_log=None, epoch=None):
    directory = Path(arm_directory).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Arm output must be a directory")
    warnings = []
    contract = _read_json(directory / "execution_contract.json")
    if contract is None:
        warnings.append("Missing execution_contract.json: configured totals/resources remain unknown")
    progress = _read_json(directory / "progress.json")
    updates = _read_jsonl(directory / "update_timing.jsonl", "update", warnings)
    validations = _read_jsonl(directory / "validation_timing.jsonl", "validation", warnings)
    curves = _read_jsonl(directory / "curve.jsonl", "curve", warnings)
    completed_path = directory / "training_complete.json"
    final_report = _read_json(completed_path)
    # Old paused/completed reports can survive a resumed run. Do not let an old
    # terminal report override newer timing/progress observations.
    observed_paths = [directory / name for name in ("progress.json", "update_timing.jsonl", "validation_timing.jsonl", "curve.jsonl")]
    newest_observation = max((p.stat().st_mtime_ns for p in observed_paths if p.exists()), default=0)
    if final_report is not None and completed_path.stat().st_mtime_ns < newest_observation:
        warnings.append("training_complete.json predates newer observations; terminal phase ignored")
        final_report = None
    validation_files = []
    for path in directory.glob("validation_epoch_*.json"):
        token = path.stem.removeprefix("validation_epoch_")
        if token.isdigit():
            validation_files.append((int(token), path))
    latest_validation = None
    for _, path in sorted(validation_files, reverse=True):
        try:
            latest_validation = _read_json(path)
        except json.JSONDecodeError:
            warnings.append(f"Validation report is not yet complete JSON: {path}; trying older completed report")
            continue
        break
    if cache_log is None:
        candidates = [directory / "preparation_reuse.jsonl", directory.parent / "preparation_reuse.jsonl",
                      directory.parent.parent / "preparation_reuse.jsonl"]
        cache_log = next((path for path in candidates if path.is_file()), None)
    if cache_log is None:
        cache_rows = []
        warnings.append("No preparation_reuse.jsonl found: cache history unavailable")
    else:
        cache_log = Path(cache_log).resolve(strict=True)
        cache_rows = _read_jsonl(cache_log, "cache", warnings)
    if input_log is None:
        candidates = [directory / "input_timing.jsonl", directory.parent / "input_timing.jsonl",
                      directory.parent.parent / "input_timing.jsonl"]
        input_log = next((path for path in candidates if path.is_file()), None)
    if input_log is None:
        input_rows = []
        warnings.append("No input_timing.jsonl found: CPU input construction history unavailable")
    else:
        input_log = Path(input_log).resolve(strict=True)
        input_rows = _read_jsonl(input_log, "input", warnings)
    result = build_report(contract, updates, validations, curves, cache_rows, progress=progress,
                          final_report=final_report, epoch=epoch, latest_validation=latest_validation,
                          input_rows=input_rows, cache_available=cache_log is not None, input_available=input_log is not None)
    result.update(arm_directory=str(directory), cache_log=str(cache_log) if cache_log else None,
                  input_log=str(input_log) if input_log else None, warnings=warnings,
                  read_only=True, checkpoint_loaded=False)
    return result


def _fmt(value, digits=2):
    return f"{value:.{digits}f}" if _number(value) is not None else "unknown"


def _coverage_text(coverage):
    return f"{coverage['unique_logged_sources'] if coverage['unique_logged_sources'] is not None else 'unknown'}/{coverage['total_configured_sources'] if coverage['total_configured_sources'] is not None else 'unknown'} ({_fmt(coverage['percent'])}%)"


def render_text(report):
    live, training, validation = report["live_progress"], report["training_epoch"], report["validation_epoch"]
    lines = [f"Arm: {report.get('arm') or 'unknown'} | epoch {report['selected_epoch']} | latest reported phase: {report['phase']}",
             f"Phase evidence: {report['phase_source']} | process liveness not checked",
             f"Physical source batch={report['physical_sample_batch']} CPU workers={report['workers']} DEBUG={report['debug']}"]
    if live.get("phase"):
        lines.append(f"Live observation {live.get('status')} {live['phase']}/{live.get('stage')} | sources {live.get('completed_sources')}/{live.get('total_sources')} ({_fmt(live['percent'])}%) | stage elapsed {_fmt(live.get('stage_elapsed_seconds'))}s | updated {live.get('updated_at', 'unknown')} (age {_fmt(live.get('snapshot_age_seconds'))}s)")
        if live.get("metrics"):
            lines.append(f"Live scores ({live.get('metrics_scope', 'scope unknown')}): " + " ".join(f"{k}={_fmt(v, 6)}" for k, v in live['metrics'].items()))
    else:
        if report['phase_source'] == 'final report':
            lines.append(f"Latest terminal report status: {report.get('terminal_status') or 'unknown'}")
        else:
            lines.append("Legacy inference: " + report["expected_next_phase_from_legacy_records"])
    history = report["optimizer_history"]
    lines.append(f"Logged optimizer updates: {history['last_logged_update']}/{history['total_planned_updates']} ({_fmt(history['percent'])}%), separate from validation progress")
    lines.append(f"Epoch optimizer source coverage: {_coverage_text(training['coverage'])}; validation coverage: {_coverage_text(validation['coverage'])}")
    tt, vt = training['timings'], validation['timings']
    lines.append(f"Training observed loader+step: {_fmt(training['observed_loader_plus_step_seconds'])}s | wait {_fmt(tt['loader_wait_seconds']['seconds'])}s ({_fmt(tt['loader_wait_seconds']['percent_of_observed_loader_plus_step'])}%) | step {_fmt(tt['step_seconds']['seconds'])}s | checkpoint inside step {_fmt(tt['checkpoint_seconds']['seconds'])}s")
    lines.append(f"Training CUDA substage times: transfer {_fmt(tt['transfer_seconds']['seconds'])}s forward {_fmt(tt['forward_seconds']['seconds'])}s backward {_fmt(tt['backward_seconds']['seconds'])}s optimizer/check {_fmt(tt['check_clip_optimizer_seconds']['seconds'])}s | observed sources/s {_fmt(training['observed_sources_per_second'], 4)}")
    lines.append(f"Training latest logged loss {_fmt(training['latest_loss'], 6)} | overflow retry records {training['overflow_retry_records']} (retry time is nested in a later successful step)")
    lines.append(f"Validation blocking load+forward: {_fmt(validation['observed_blocking_load_plus_forward_seconds'])}s | blocking load {_fmt(vt['blocking_loader_seconds']['seconds'])}s ({_fmt(vt['blocking_loader_seconds']['percent_of_observed_blocking_load_plus_forward'])}%) | forward {_fmt(vt['full_joint_forward_seconds']['seconds'])}s | logged batch wall {_fmt(validation['observed_batch_wall_seconds'])}s | completed phase active total {_fmt(validation['reported_completed_validation_active_seconds'])}s")
    if validation['prefetch_batch_records']:
        lines.append(f"Validation loader construction {_fmt(vt['loader_seconds']['seconds'])}s overlaps prefetch; {validation['prefetch_batch_records']} new prefetch rows and {validation['legacy_synchronous_batch_records']} legacy synchronous rows")
    to_gib = lambda value: value / 2**30 if value is not None else None
    lines.append(f"Logged peaks: train CUDA {_fmt(to_gib(training['peak_cuda_bytes']))} GiB RSS {_fmt(to_gib(training['peak_rss_bytes']))} GiB process CPU {_fmt(training['max_process_cpu_percent'])}% | validation CUDA {_fmt(to_gib(validation['peak_cuda_bytes']))} GiB RSS {_fmt(to_gib(validation['peak_rss_bytes']))} GiB")
    scores = report['latest_completed_validation']
    if scores:
        lines.append(f"Completed full-validation scores epoch {scores.get('epoch')} ({scores.get('metric_weighting', 'weighting unknown')}): " + " ".join(f"{k}={_fmt(v, 6)}" for k, v in scores.get('metrics', {}).items()))
        lines.append("Score scope: source-anchor ranking proxy; not CP/clinical accuracy")
    else:
        lines.append("Completed full-validation scores: unavailable")
    cache = report['cache_history']
    if cache['available']:
        lines.append(f"Cache entire-log history (not per epoch/ETA): field calls={cache['field_calls']} summed helper={_fmt(cache['summed_field_helper_seconds']['seconds'])}s pressure events={cache['pressure_events']} statuses=" + json.dumps({key: value['calls'] for key, value in cache['field_statuses'].items()}))
    else:
        lines.append("Cache entire-log history: unavailable")
    inputs = report['input_preparation']
    if inputs['available']:
        for key, label in (("training_epoch", "Training-mode CPU input history for selected view_epoch (may include calibration)"), ("validation_history", "Full129 validation CPU input across log history")):
            mode = inputs[key]
            lines.append(f"{label}: {_fmt(mode['input_construction']['seconds'])}s in {mode['completed_batch_records']} completed batches; failed {mode['failed_batch_records']}; excludes pinning/queue wait")
            lines.append("  Disjoint construction: " + " | ".join(f"{name.removesuffix('_seconds')} {_fmt(value['seconds'])}s ({_fmt(value['percent_of_cpu_input_construction'])}%)" for name, value in mode['timings'].items()))
            details = mode['inclusive_details']
            if details['records_with_value']:
                lines.append(f"  Inclusive details (overlap; do not add; {details['records_with_value']}/{details['records_total']} batches): " +
                    ' | '.join(f"{name.removesuffix('_seconds')} {_fmt(value['timing']['seconds'])}s/{value['observed_calls']} calls"
                               for name, value in details['timings'].items()))
            compact = mode['compact_upper_cache']
            if compact['records_with_value']:
                lines.append(f"  Compact upper cache ({compact['records_with_value']}/{compact['records_total']} batch deltas): " +
                    ' | '.join(f"{kind} builds={value.get('builds', 'unknown')} reopens={value.get('reopens', 'unknown')} RAM hits={value.get('resident_hits', 'unknown')} original={_fmt(value.get('original_seconds'))}s"
                               for kind, value in compact['kinds'].items()))
        lines.append("Validation input view_epoch is fixed; cumulative input preparation can overlap GPU work and is not added to epoch wall time.")
    else:
        lines.append("CPU input construction breakdown: unavailable")
    lines.append("Timing rule: checkpoint/CUDA substage times are already inside training step; cache helper sums can overlap loader work. Missing measurements stay unknown.")
    lines.extend("Notice: " + warning for warning in report['warnings'])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--arm-dir", type=Path)
    source.add_argument("--experiment", type=Path)
    parser.add_argument("--arm", choices=ARMS, default="native_listwise")
    parser.add_argument("--cache-log", type=Path)
    parser.add_argument("--input-log", type=Path)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--json", action="store_true", help="Emit structured report instead of concise text")
    args = parser.parse_args()
    if args.epoch is not None and args.epoch < 0:
        raise ValueError("Epoch must be nonnegative (zero is initial validation)")
    directory = args.arm_dir
    if directory is None:
        candidates = [args.experiment / args.arm, args.experiment / "arms" / args.arm]
        matches = [path for path in candidates if (path / "execution_contract.json").is_file()]
        if len(matches) != 1:
            raise ValueError("Cannot uniquely locate arm output; pass its exact --arm-dir")
        directory = matches[0]
    result = inspect_arm(directory, cache_log=args.cache_log, input_log=args.input_log, epoch=args.epoch)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) if args.json else render_text(result), flush=True)


if __name__ == "__main__":
    main()
