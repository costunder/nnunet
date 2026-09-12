"""Admit only a proven, pre-write raw-case disk-reserve failure for retry.

This is not a generic error-row reset. The native preflight runs before
prepare_case/save_case; legacy progress and resource reports must independently
identify that failure. They have no common invocation ID, so ambiguous matches
are refused. Existing successful rows still require the builder's native audit.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid


_DISK_FAILURE = re.compile(
    r"RuntimeError: Insufficient disk for raw-target case ([A-Za-z0-9_-]+): "
    r"free=([0-9]+), baseline_payload_alone=([0-9]+), reserved_free=([0-9]+); "
    r"existing bank/preprocessing are preserved"
)


def admit_disk_preflight_retry(bank_root, failed_row, *, expected_row, contract,
                              baseline_payload_bytes, minimum_free_bytes,
                              manifest_path=None) -> Path:
    """Verify and archive failure evidence without changing the active manifest.

    The caller owns the experiment lock and commits a replacement row only as
    normal processing succeeds/fails. No payload, checkpoint, log or error row
    is deleted here. The original native disk check remains in the retry path.
    """
    root = Path(bank_root).resolve()
    manifest = Path(manifest_path) if manifest_path is not None else root / "manifest.csv"
    evidence = {}

    def refuse(reason):
        raise ValueError(
            f"Bank disk-preflight retry refused: {reason}. Existing bank and failure "
            "records are preserved; inspect the evidence, do not delete rows or use --overwrite."
        )

    def read(path):
        path = Path(path)
        relative = path.relative_to(root)
        if path.is_symlink() or not path.is_file() or path.resolve() != root / relative:
            refuse(f"missing or unsafe evidence file: {relative}")
        data = path.read_bytes()
        evidence[relative.as_posix()] = data
        return data

    match = _DISK_FAILURE.fullmatch(str(failed_row.get("reason", "")))
    if failed_row.get("status") != "error" or match is None:
        refuse("row is not the native pre-write disk-reserve failure")
    case_id, old_free, old_payload, old_reserve = match.groups()
    payload, reserve = int(baseline_payload_bytes), int(minimum_free_bytes)
    if payload <= 0 or reserve < 0 or (int(old_payload), int(old_reserve)) != (payload, reserve):
        refuse("recorded payload/reserve differs from the current case/configuration")
    if int(old_free) > payload + reserve:
        refuse("recorded free space would not have failed the native preflight")
    component = int(expected_row["source_component"])
    if component <= 0 or case_id != str(expected_row["case_id"]):
        refuse("failure case/component identity differs from the current source")
    for name in ("case_id", "source_component", "diameter_mm"):
        if str(failed_row.get(name, "")) != str(expected_row[name]):
            refuse(f"source identity changed: {name}")
    for name in ("candidate_count", "rejected_geometry", "rejected_preprocessed"):
        if str(failed_row.get(name, "")) != "0":
            refuse(f"failure row contains post-preflight state: {name}")
    for name in ("entry", "entry_sha256", "candidate_pool_sha256", "source_mapping_json",
                 "candidate_search_json", "score_min", "score_max", "score_std"):
        if failed_row.get(name, "") != "":
            refuse(f"failure row contains published/processed state: {name}")
    if json.loads(read(root / "config.json")) != contract:
        refuse("bank configuration differs from the verified current contract")
    for name in ("index.json", "complete.json"):
        if (root / name).exists() or (root / name).is_symlink():
            refuse(f"incomplete failure row conflicts with {name}")

    rows = list(csv.DictReader(io.StringIO(read(manifest).decode("utf-8"))))
    keys = [(row["case_id"], int(row["source_component"])) for row in rows]
    if len(keys) != len(set(keys)):
        refuse("manifest contains duplicate source identities")
    matches = [row for row, key in zip(rows, keys) if key == (case_id, component)]
    if len(matches) != 1 or any(
        str(matches[0].get(key, "")) != str(failed_row.get(key, ""))
        for key in set(matches[0]) | set(failed_row)
    ):
        refuse("manifest no longer contains the exact failed row")

    # At this failure boundary no raw case or candidate payload was published.
    # Shared raw_sources from OTHER completed sources are deliberately allowed.
    source_name = f"{case_id}__component_{component:03d}"
    for directory, prefix in (("raw_cases", case_id + "."),
                              ("raw_candidates", source_name),
                              ("entries", source_name + ".")):
        parent = root / directory
        if parent.is_symlink():
            refuse(f"unsafe artifact directory: {directory}")
        if parent.exists():
            artifacts = [path.name for path in parent.iterdir() if path.name.startswith(prefix)]
            if artifacts:
                refuse(f"partial or published artifacts for failed source: {directory}/{artifacts[0]}")

    reason = str(failed_row["reason"])
    resource_matches = []
    for path in (root / "preparation_resources").glob(f"raw_case.{case_id}.*.json"):
        data = path.read_bytes()
        report = json.loads(data)
        if report.get("error") != reason:
            continue
        if (report.get("format") != "raw_target_case_resources_v1"
                or report.get("case_id") != case_id or report.get("status") != "failed"
                or report.get("persistent_array_lower_bound_bytes") != payload
                or report.get("sampling_error") is not None
                or report.get("final_snapshot_error") is not None
                or report.get("measurement_error") is not None):
            refuse(f"inconsistent native failure measurement: {path.name}")
        elapsed = report.get("elapsed_seconds")
        if (not isinstance(report.get("before"), dict)
                or not isinstance(report.get("after"), dict)
                or isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed) or elapsed < 0
                or "sampling_error" not in report or "final_snapshot_error" not in report):
            refuse(f"incomplete native failure measurement: {path.name}")
        resource_matches.append(path)
    if len(resource_matches) != 1:
        refuse("native failure measurement is missing or ambiguous")
    read(resource_matches[0])

    progress_matches = []
    for path in root.glob("preparation_progress.*.jsonl"):
        data = path.read_bytes()
        # The current invocation may still be writing. Only the closed failed
        # invocation containing this exact error is parsed/bound as evidence.
        if reason.encode("utf-8") not in data:
            continue
        if not data.endswith(b"\n"):
            refuse(f"incomplete failure progress log: {path.name}")
        events = [json.loads(line) for line in data.splitlines()]
        failures = [event for event in events if event.get("event") == "failed"
                    and event.get("error") == reason]
        if not failures:
            continue
        if len(failures) != 1 or failures[0] != events[-1]:
            refuse(f"nonterminal or ambiguous failure progress: {path.name}")
        starts = [event for event in events[:-1] if event.get("event") == "phase_start"]
        if not starts:
            refuse(f"missing raw-case phase start: {path.name}")
        for event in (starts[-1], events[-1]):
            if (event.get("format") != "hiercp_bank_progress_v1"
                    or event.get("phase") != "raw_target_case"
                    or event.get("case_id") != case_id or event.get("component") != component):
                refuse(f"failure progress does not identify the pre-write phase: {path.name}")
        progress_matches.append(path)
    if len(progress_matches) != 1:
        refuse("native failure progress is missing or ambiguous")
    read(progress_matches[0])

    free = int(shutil.disk_usage(root).free)
    if free <= payload + reserve:
        refuse(f"disk is still insufficient: free={free}, baseline_payload_alone={payload}, "
               f"reserved_free={reserve}; required_more_than={payload + reserve}")
    # Refuse a concurrent edit rather than archive a mixture of old and new
    # evidence. This supplements, but does not replace, the launcher's lock.
    for name, data in evidence.items():
        if (root / name).read_bytes() != data:
            refuse(f"failure evidence changed during admission: {name}")
    archive = root / "disk_retry_history"
    if archive.is_symlink():
        refuse("unsafe disk retry history directory")
    archive.mkdir(exist_ok=True)
    receipt = archive / f"{uuid.uuid4().hex}.json"
    record = {
        "format": "hiercp_bank_disk_preflight_retry_v1",
        "case_id": case_id, "source_component": component,
        "failed_row": dict(failed_row), "baseline_payload_bytes": payload,
        "minimum_free_bytes": reserve, "current_free_bytes": free,
        "evidence": {name: {"sha256": hashlib.sha256(data).hexdigest(),
                            "content_base64": base64.b64encode(data).decode("ascii")}
                     for name, data in evidence.items()},
    }
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"[BankDiskRetry] {case_id} component={component} free={free} "
          f"baseline_payload_alone={payload} reserved_free={reserve} evidence={receipt}", flush=True)
    return receipt
