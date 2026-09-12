"""DEBUG disk-preflight retry proofs; synthetic arrays, no patient or training.

The original failure is emitted by the real case preflight and progress writer.
Only available disk bytes are mocked. Native resampling is never entered while
creating the failure evidence; the separate wiring fixture covers real payloads.
"""
from __future__ import annotations

import base64
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools import online_raw_bank_preparation as raw
from tools import online_bank_disk_retry as retry
from tools.online_bank_progress import BankProgress


FIELDS = ("case_id", "source_component", "source_mapping_json", "candidate_search_json",
          "diameter_mm", "status", "reason", "candidate_count", "rejected_geometry",
          "rejected_preprocessed", "entry", "entry_sha256", "candidate_pool_sha256",
          "score_min", "score_max", "score_std")


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def inventory(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def disk_failure_fixture(root, *, case_id="DEBUG_DISK", shape=(2, 2, 2), contract=None,
                         minimum_free_bytes=1024, diameter_mm="1.240701"):
    """Generate the exact historical error/proofs through production writers."""
    root.mkdir(parents=True, exist_ok=True)
    contract = dict(contract or {"DEBUG": "disk-retry-only", "candidate_count": 128})
    (root / "config.json").write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
    payload_bytes = int(np.prod(shape)) * 10
    measured_free = payload_bytes + minimum_free_bytes - 1
    data = np.zeros((1, *shape), dtype=np.float32)
    seg = np.zeros((1, *shape), dtype=np.int16)
    with redirect_stdout(io.StringIO()), \
         patch.object(raw.shutil, "disk_usage", return_value=SimpleNamespace(free=measured_free)), \
         patch.object(raw, "prepare_case", side_effect=AssertionError("DEBUG native work before disk check")):
        try:
            with BankProgress(root, heartbeat_seconds=3600) as progress:
                progress.update("raw_target_case", case_id=case_id, component=1)
                raw.prepare_raw_case(root, case_id, data[0], seg[0], {}, {}, data, seg,
                                     configuration_name="3d_fullres", raw_spacing=[1., 1., 1.],
                                     raw_spatial_unit="mm", minimum_free_bytes=minimum_free_bytes)
        except RuntimeError as exc:
            reason = f"{type(exc).__name__}: {exc}"
        else:
            raise AssertionError("DEBUG preflight unexpectedly succeeded")
    if not reason.startswith("RuntimeError: Insufficient disk for raw-target case "):
        raise AssertionError(reason)
    expected = dict(case_id=case_id, source_component=1, diameter_mm=diameter_mm,
                    status="error", reason="", candidate_count=0, rejected_geometry=0,
                    rejected_preprocessed=0, entry="")
    failed = dict(expected, reason=reason)
    manifest = root / "manifest.csv"
    write_manifest(manifest, [failed])
    with manifest.open(encoding="utf-8", newline="") as handle:
        failed = next(csv.DictReader(handle))
    return SimpleNamespace(root=root, failed=failed, expected=expected, contract=contract,
                           baseline=payload_bytes, reserve=minimum_free_bytes, manifest=manifest,
                           resource=next((root / "preparation_resources").glob("raw_case.*.json")),
                           progress=next(root.glob("preparation_progress.*.jsonl")))


class OnlineBankDiskRetryDebugTests(unittest.TestCase):
    def admit(self, fixture, *, free=None):
        available = fixture.baseline + fixture.reserve + 1 if free is None else free
        with redirect_stdout(io.StringIO()), patch.object(retry.shutil, "disk_usage", return_value=SimpleNamespace(free=available)):
            return retry.admit_disk_preflight_retry(
                fixture.root, fixture.failed, expected_row=fixture.expected,
                contract=fixture.contract, baseline_payload_bytes=fixture.baseline,
                minimum_free_bytes=fixture.reserve)

    def assert_rejected_without_mutation(self, fixture, *, free=None):
        before = inventory(fixture.root)
        with self.assertRaises((ValueError, RuntimeError, OSError)):
            self.admit(fixture, free=free)
        self.assertEqual(inventory(fixture.root), before)

    def mutate_json(self, path, mutate):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_real_legacy_failure_is_admitted_with_exact_immutable_evidence(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_disk_retry_") as directory:
            f = disk_failure_fixture(Path(directory))
            completed = f.root / "entries/DEBUG_COMPLETE__component_001.npz"
            completed.parent.mkdir()
            completed.write_bytes(b"DEBUG opaque existing completed entry; never loaded")
            source = f.root / "raw_sources/DEBUG_SHARED.npy"
            source.parent.mkdir()
            source.write_bytes(b"DEBUG unrelated shared source; never loaded")
            before = inventory(f.root)
            receipt_path = self.admit(f)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt_path.parent, f.root / "disk_retry_history")
            self.assertEqual(receipt["format"], "hiercp_bank_disk_preflight_retry_v1")
            self.assertEqual((receipt["case_id"], receipt["source_component"]), ("DEBUG_DISK", 1))
            self.assertEqual(receipt["failed_row"], f.failed)
            self.assertEqual(receipt["baseline_payload_bytes"], f.baseline)
            self.assertEqual(receipt["minimum_free_bytes"], f.reserve)
            expected_paths = {f.manifest.relative_to(f.root).as_posix(), "config.json",
                              f.resource.relative_to(f.root).as_posix(), f.progress.name}
            self.assertEqual(set(receipt["evidence"]), expected_paths)
            for relative, evidence in receipt["evidence"].items():
                self.assertEqual(base64.b64decode(evidence["content_base64"], validate=True), before[relative])
                self.assertEqual(evidence["sha256"], hashlib.sha256(before[relative]).hexdigest())
            for relative, original in before.items():
                self.assertEqual((f.root / relative).read_bytes(), original)
            original_receipt = receipt_path.read_bytes()
            second = self.admit(f)
            self.assertNotEqual(second, receipt_path)
            self.assertEqual(receipt_path.read_bytes(), original_receipt)

    def test_still_insufficient_space_and_exact_boundary_are_rejected(self):
        for extra in (-1, 0):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory(prefix="DEBUG_disk_still_full_") as directory:
                f = disk_failure_fixture(Path(directory))
                self.assert_rejected_without_mutation(f, free=f.baseline + f.reserve + extra)

    def test_row_identity_publication_and_nonzero_counters_never_become_retryable(self):
        changes = {"case_id": "DEBUG_OTHER", "source_component": "2", "diameter_mm": "2.000000",
                   "status": "no_placement", "candidate_count": "1", "rejected_geometry": "1",
                   "rejected_preprocessed": "1", "entry": "entries/DEBUG.npz",
                   "entry_sha256": "a" * 64, "candidate_pool_sha256": "b" * 64,
                   "source_mapping_json": "{}", "candidate_search_json": "{}", "score_min": "0"}
        for field, value in changes.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory(prefix="DEBUG_disk_row_") as directory:
                f = disk_failure_fixture(Path(directory))
                f.failed[field] = value
                write_manifest(f.manifest, [f.failed])
                self.assert_rejected_without_mutation(f)

    def test_generic_error_and_reason_parameter_mismatches_are_rejected(self):
        for problem in ("generic", "case", "baseline", "reserve"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory(prefix="DEBUG_disk_reason_") as directory:
                f = disk_failure_fixture(Path(directory))
                reason = f.failed["reason"]
                if problem == "generic":
                    reason = "RuntimeError: DEBUG unexpected implementation failure"
                elif problem == "case":
                    reason = reason.replace("case DEBUG_DISK:", "case DEBUG_OTHER:")
                elif problem == "baseline":
                    reason = reason.replace(f"baseline_payload_alone={f.baseline}", "baseline_payload_alone=1")
                else:
                    reason = reason.replace(f"reserved_free={f.reserve}", "reserved_free=1")
                f.failed["reason"] = reason
                write_manifest(f.manifest, [f.failed])
                self.assert_rejected_without_mutation(f)

    def test_contract_and_duplicate_or_changed_manifest_are_rejected(self):
        for problem in ("contract", "duplicate", "changed", "missing"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory(prefix="DEBUG_disk_manifest_") as directory:
                f = disk_failure_fixture(Path(directory))
                if problem == "contract":
                    self.mutate_json(f.root / "config.json", lambda value: value.update(candidate_count=127))
                elif problem == "duplicate":
                    write_manifest(f.manifest, [f.failed, f.failed])
                elif problem == "changed":
                    write_manifest(f.manifest, [dict(f.failed, reason="RuntimeError: DEBUG different failure")])
                else:
                    f.manifest.unlink()
                self.assert_rejected_without_mutation(f)

    def test_resource_proof_is_required_unique_and_free_of_measurement_errors(self):
        problems = ("missing", "duplicate", "format", "status", "error", "bound",
                    "sampling_error", "final_snapshot_error", "missing_before", "missing_after",
                    "missing_elapsed", "missing_sampling_error", "missing_final_snapshot_error",
                    "elapsed_negative", "elapsed_bool", "elapsed_string", "elapsed_nonfinite")
        for problem in problems:
            with self.subTest(problem=problem), tempfile.TemporaryDirectory(prefix="DEBUG_disk_resource_") as directory:
                f = disk_failure_fixture(Path(directory))
                if problem == "missing":
                    f.resource.unlink()
                elif problem == "duplicate":
                    f.resource.with_name("raw_case.DEBUG_DISK.DEBUG_DUPLICATE.json").write_bytes(f.resource.read_bytes())
                elif problem.startswith("missing_"):
                    key = {"missing_elapsed": "elapsed_seconds"}.get(problem, problem.removeprefix("missing_"))
                    self.mutate_json(f.resource, lambda value: value.pop(key))
                elif problem.startswith("elapsed_"):
                    elapsed = {"elapsed_negative": -1., "elapsed_bool": True,
                               "elapsed_string": "1", "elapsed_nonfinite": float("inf")}[problem]
                    self.mutate_json(f.resource, lambda value: value.update(elapsed_seconds=elapsed))
                else:
                    change = {"format": ("format", "DEBUG_wrong"), "status": ("status", "completed"),
                              "error": ("error", "RuntimeError: DEBUG other"),
                              "bound": ("persistent_array_lower_bound_bytes", f.baseline + 1),
                              "sampling_error": ("sampling_error", "DEBUG measurement failed"),
                              "final_snapshot_error": ("final_snapshot_error", "DEBUG measurement failed")}[problem]
                    self.mutate_json(f.resource, lambda value: value.update({change[0]: change[1]}))
                self.assert_rejected_without_mutation(f)

    def test_progress_requires_matching_started_phase_and_terminal_failure(self):
        for problem in ("missing", "duplicate", "no_phase_start", "wrong_phase", "component", "after_failure"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory(prefix="DEBUG_disk_progress_") as directory:
                f = disk_failure_fixture(Path(directory))
                if problem == "missing":
                    f.progress.unlink()
                elif problem == "duplicate":
                    f.progress.with_name("preparation_progress.DEBUG_DUPLICATE.jsonl").write_bytes(f.progress.read_bytes())
                else:
                    rows = [json.loads(line) for line in f.progress.read_text(encoding="utf-8").splitlines()]
                    if problem == "no_phase_start":
                        rows = [row for row in rows if row["event"] != "phase_start"]
                    elif problem == "wrong_phase":
                        rows[-1]["phase"] = "raw_target_candidates"
                    elif problem == "component":
                        rows[-1]["component"] = 2
                    else:
                        rows.append(dict(rows[-1], event="heartbeat"))
                    f.progress.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                self.assert_rejected_without_mutation(f)

    def test_partial_raw_case_candidate_and_entry_artifacts_are_not_overwritten(self):
        names = ("raw_cases/DEBUG_DISK.json", "raw_cases/DEBUG_DISK.a0000.npy",
                 "raw_candidates/DEBUG_DISK__component_001/0000.json",
                 "entries/DEBUG_DISK__component_001.npz", "entries/DEBUG_DISK__component_001.npz.tmp")
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="DEBUG_disk_artifact_") as directory:
                f = disk_failure_fixture(Path(directory))
                path = f.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"DEBUG partial artifact; never loaded or deleted")
                self.assert_rejected_without_mutation(f)


if __name__ == "__main__":
    unittest.main()
