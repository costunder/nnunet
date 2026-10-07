"""Metadata-only UNITs for preflight binding and logging, without neural work."""
from __future__ import annotations

from tests.artifacts import unit_artifact_root
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid

from hiercp_v1x import telemetry_entry as entry


ROOT = Path(__file__).resolve().parents[1]


class TelemetryPublicationMetadataUnits(unittest.TestCase):
    def setUp(self):
        # Preserve exact task-owned metadata fixtures for inspection. A failed
        # mkdir is reported directly, never retried as a Windows tempfile name.
        self.root = unit_artifact_root() / ("v1_telemetry_entry_metadata_UNIT_" + uuid.uuid4().hex)
        self.root.mkdir(parents=True, exist_ok=False)
        self.native = self.root / "native"
        source = self.native / "source/v1.0"
        source.mkdir(parents=True)
        (source / "metadata_UNIT.txt").write_text("metadata only, no model", encoding="utf-8")
        self.contract = dict(mode="native", profile=None,
                             source_identity={"files": {"metadata_UNIT.txt": entry._digest(source / "metadata_UNIT.txt")}})
        self.contract["contract_sha256"] = entry._canonical_hash(self.contract)
        self.manifest = dict(sampling_contract={"mode": "native"}, reference_experiment=None,
                             sampling_contracts={"v1.0": self.contract})
        self.manifest["manifest_sha256"] = entry._canonical_hash(self.manifest)
        self.write(self.native / "manifest.json", self.manifest)
        self.write(self.native / "sampling/v1.0.json", self.contract)
        self.request = dict(format=entry.FORMAT, output_dir=str(self.native / "results/v1.0/telemetry"),
                            mode="native", native_reference=str(self.native), stage="v1.0",
                            baseline_manifest_sha256=self.manifest["manifest_sha256"],
                            baseline_sampling_contract_sha256=self.contract["contract_sha256"])
        self.preflight = self.native / "results/v1.0/checkpoint_best.pt.preflight.json"
        self.calibration = dict(format="hiercp_preflight_calibration_v2", selected_batch_size=32,
                                selected_num_workers=8, resource_fingerprint={"metadata_UNIT": True},
                                identity=dict(seed=42, cache_dir=str(self.native / "shared/cache"),
                                              checkpoint_path=str(self.native / "results/v1.0/checkpoint_best.pt")))

    @staticmethod
    def write(path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")

    def test_publication_creates_exact_native_lock_after_original_writer(self):
        calls = []
        def writer(path, value):
            self.assertFalse((self.native / "shared/execution_lock.json").exists())
            calls.append((path, copy.deepcopy(value)))
            self.write(path, value)
        pipeline = SimpleNamespace(_write_json_atomic=writer)
        entry.install_preflight_publication_guard(pipeline, self.request)
        pipeline._write_json_atomic(self.preflight, self.calibration)
        lock = entry._read(self.native / "shared/execution_lock.json")
        self.assertEqual(lock, dict(selected_batch_size=32, selected_num_workers=8,
                                  baseline_calibration_sha256=entry._digest(self.preflight),
                                  resource_fingerprint=self.calibration["resource_fingerprint"],
                                  baseline_manifest_sha256=self.request["baseline_manifest_sha256"],
                                  baseline_sampling_contract_sha256=self.request["baseline_sampling_contract_sha256"]))
        self.assertEqual(calls, [(self.preflight, self.calibration)])

    def test_nonpreflight_writes_keep_exact_arguments_and_return_value(self):
        calls = []
        def writer(path, value):
            calls.append((path, value))
            return "metadata UNIT result"
        pipeline = SimpleNamespace(_write_json_atomic=writer)
        entry.install_preflight_publication_guard(pipeline, self.request)
        value = {"unchanged": True}
        path = self.root / "other.json"
        self.assertEqual(pipeline._write_json_atomic(path, value), "metadata UNIT result")
        self.assertIs(calls[0][1], value)
        self.assertEqual(calls[0][0], path)

    def test_changed_existing_preflight_rejected_before_original_writer(self):
        self.write(self.preflight, self.calibration)
        before = self.preflight.read_bytes()
        def unexpected(*args):
            raise AssertionError("Mismatched existing preflight must not be overwritten")
        pipeline = SimpleNamespace(_write_json_atomic=unexpected)
        entry.install_preflight_publication_guard(pipeline, self.request)
        changed = {**self.calibration, "selected_batch_size": 64}
        with self.assertRaisesRegex(ValueError, "no overwrite"):
            pipeline._write_json_atomic(self.preflight, changed)
        self.assertEqual(self.preflight.read_bytes(), before)

    def test_invalid_new_calibration_rejected_before_publication(self):
        pipeline = SimpleNamespace(_write_json_atomic=lambda *args: self.fail("Invalid calibration must not be published"))
        entry.install_preflight_publication_guard(pipeline, self.request)
        with self.assertRaises(ValueError):
            pipeline._write_json_atomic(self.preflight, {**self.calibration, "selected_batch_size": True})
        self.assertFalse(self.preflight.exists())
        self.assertFalse((self.native / "shared/execution_lock.json").exists())

    def test_identical_existing_preflight_preserves_exact_bytes(self):
        self.write(self.preflight, self.calibration)
        before = self.preflight.read_bytes()
        pipeline = SimpleNamespace(_write_json_atomic=lambda *args: self.fail("Existing preflight must not be rewritten"))
        entry.install_preflight_publication_guard(pipeline, self.request)
        pipeline._write_json_atomic(self.preflight, copy.deepcopy(self.calibration))
        self.assertEqual(self.preflight.read_bytes(), before)

    def test_wrong_calibration_seed_path_or_measured_values_rejected(self):
        changes = [dict(selected_batch_size=True), dict(selected_batch_size=0),
                   dict(selected_num_workers=-1), dict(selected_num_workers=True),
                   dict(resource_fingerprint={}),
                   dict(identity={**self.calibration["identity"], "seed": 41}),
                   dict(identity={**self.calibration["identity"], "cache_dir": str(self.root / "other")}),
                   dict(identity={**self.calibration["identity"], "checkpoint_path": str(self.root / "other.pt")})]
        for change in changes:
            with self.subTest(change=change):
                self.write(self.preflight, {**self.calibration, **change})
                with self.assertRaises(ValueError):
                    entry.verify_execution_lock(self.request, create=True)
                self.assertFalse((self.native / "shared/execution_lock.json").exists())

    def test_existing_lock_verified_without_rewriting_or_mismatch_adoption(self):
        self.write(self.preflight, self.calibration)
        expected = entry.verify_execution_lock(self.request, create=True)
        path = self.native / "shared/execution_lock.json"
        before = path.read_bytes()
        self.assertEqual(entry.verify_execution_lock(self.request, create=True), expected)
        self.assertEqual(path.read_bytes(), before)
        self.write(path, {**expected, "selected_batch_size": 64})
        changed = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "no overwrite"):
            entry.verify_execution_lock(self.request, create=True)
        self.assertEqual(path.read_bytes(), changed)

    def test_nested_requires_existing_lock_and_never_creates_or_writes_native_preflight(self):
        self.write(self.preflight, self.calibration)
        nested = {**self.request, "mode": "strict_nested"}
        with self.assertRaisesRegex(ValueError, "required"):
            entry.verify_execution_lock(nested)
        with self.assertRaisesRegex(ValueError, "cannot publish"):
            entry.verify_execution_lock(nested, create=True)
        expected = entry.verify_execution_lock(self.request, create=True)
        self.assertEqual(entry.verify_execution_lock(nested), expected)
        before = self.preflight.read_bytes()
        pipeline = SimpleNamespace(_write_json_atomic=lambda *args: self.fail("Nested must not write native preflight"))
        entry.install_preflight_publication_guard(pipeline, nested)
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            pipeline._write_json_atomic(self.preflight, self.calibration)
        self.assertEqual(self.preflight.read_bytes(), before)

    def test_baseline_manifest_sampling_or_actual_source_mutation_rejected(self):
        for kind in ("request", "sampling", "source"):
            with self.subTest(kind=kind):
                request = copy.deepcopy(self.request)
                original_contract = (self.native / "sampling/v1.0.json").read_bytes()
                source = self.native / "source/v1.0/metadata_UNIT.txt"
                original_source = source.read_bytes()
                try:
                    if kind == "request":
                        request["baseline_manifest_sha256"] = "0" * 64
                    elif kind == "sampling":
                        self.write(self.native / "sampling/v1.0.json", {**self.contract, "mode": "strict_nested"})
                    else:
                        source.write_bytes(b"changed metadata UNIT")
                    with self.assertRaises(ValueError):
                        entry.verify_baseline_binding(request)
                finally:
                    (self.native / "sampling/v1.0.json").write_bytes(original_contract)
                    source.write_bytes(original_source)

    def test_request_validates_native_identity_and_output_containment(self):
        contract_path = self.native / "sampling/v1.0.json"
        self.assertEqual(entry.validate_request(self.request, contract_path, self.contract), self.native)
        for change in ({"output_dir": str(self.root / "elsewhere")}, {"stage": "v1.1"},
                       {"mode": "strict_nested"}, {"output_dir": "relative"}, {"extra": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                entry.validate_request({**self.request, **change}, contract_path, self.contract)


class TelemetryTeeMetadataUnits(unittest.TestCase):
    def test_tee_preserves_terminal_isatty_and_logs_progress_text(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        terminal, log = Terminal(), io.StringIO()
        tee = entry.Tee(terminal, log)
        self.assertTrue(tee.isatty())
        self.assertEqual(tee.write("\rprogress 1/2"), len("\rprogress 1/2"))
        tee.flush()
        self.assertEqual(terminal.getvalue(), "\rprogress 1/2")
        self.assertEqual(log.getvalue(), terminal.getvalue())


if __name__ == "__main__":
    unittest.main()
