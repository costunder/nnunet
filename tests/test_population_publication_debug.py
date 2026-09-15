"""CPU DEBUG: immutable prototype publication and interruption recovery.

The cohort/region loader is an explicit synthetic boundary. The prototype fit,
serialization, reload verification and final publication are real operators.
No medical data or model-training result is represented by these tests.
"""
from __future__ import annotations

import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from hiercp import cache
from hiercp.schema import GraphBuildConfig


class PopulationPublicationDebug(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hiercp-publication-debug-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "prototype/bank.pt"
        self.paths = [SimpleNamespace(case_id=name) for name in ("case_a", "case_b")]
        rng = np.random.default_rng(20260913)
        self.descriptors = {path.case_id: rng.normal(size=(24, 16)).astype(np.float32)
                            for path in self.paths}
        self.sources = [{"case_id": path.case_id, "image_sha256": "a" * 64,
                         "label_sha256": "b" * 64} for path in self.paths]
        self.arguments = dict(data_dir=self.root / "raw", output_path=self.output,
                              region_cache_dir=self.root / "regions",
                              training_case_ids=[path.case_id for path in self.paths],
                              graph_config=GraphBuildConfig(), liver_label=1, tumor_label=2,
                              ct_clip=(-200., 250.), seed=42, overwrite=False, workers=2)

    def prepare(self, **changes):
        def regions(case, **kwargs):
            return SimpleNamespace(region_features=self.descriptors[case.case_id], num_regions=24)

        def jobs(*, tasks, function, commit, **kwargs):
            # Deliberately reversed completion order tests deterministic cohort
            # assembly, not production process-pool throughput.
            for task in reversed(tasks):
                commit(function(task))

        with mock.patch.object(cache, "discover_cases", return_value=self.paths), \
             mock.patch.object(cache, "_source_contract", return_value=self.sources), \
             mock.patch.object(cache, "load_case", side_effect=lambda case: case), \
             mock.patch.object(cache, "load_or_build_patient_regions", side_effect=regions), \
             mock.patch("hiercp.preparation_runtime.run_case_jobs", side_effect=jobs):
            return cache.prepare_prototype_bank(**{**self.arguments, **changes})

    def test_create_retry_and_collision_preserve_bytes(self):
        path = self.root / "immutable.json"
        cache._publish_prototype_bytes(path, b"first")
        cache._publish_prototype_bytes(path, b"first")
        with self.assertRaises(FileExistsError):
            cache._publish_prototype_bytes(path, b"different")
        self.assertEqual(path.read_bytes(), b"first")

    def test_failure_after_bank_can_resume_without_overwrite(self):
        original_publish = cache._publish_prototype_bytes

        def fail_manifest(path, payload):
            if path.name == "manifest.csv":
                raise OSError("DEBUG injected interruption after bank publication")
            return original_publish(path, payload)

        with mock.patch.object(cache, "_publish_prototype_bytes", side_effect=fail_manifest):
            with self.assertRaisesRegex(OSError, "DEBUG injected"):
                self.prepare()
        before = self.output.read_bytes()
        self.assertFalse((self.output.parent / "metadata.json").exists())
        resumed = self.prepare()
        self.assertEqual(self.output.read_bytes(), before)
        metadata = json.loads((self.output.parent / "metadata.json").read_text())
        self.assertEqual(metadata["state"], "ready")
        self.assertEqual(metadata["prototype_fingerprint"], resumed.fingerprint())
        completed = {path.name: path.read_bytes() for path in self.output.parent.iterdir()
                     if path.is_file()}
        self.prepare(overwrite=True)
        self.assertEqual(completed, {path.name: path.read_bytes()
                                    for path in self.output.parent.iterdir() if path.is_file()})

    def test_partial_different_intent_is_not_adopted(self):
        original_publish = cache._publish_prototype_bytes

        def fail_manifest(path, payload):
            if path.name == "manifest.csv":
                raise OSError("DEBUG interruption")
            return original_publish(path, payload)

        with mock.patch.object(cache, "_publish_prototype_bytes", side_effect=fail_manifest):
            with self.assertRaises(OSError):
                self.prepare()
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare(seed=43, overwrite=True)
        self.assertEqual(before, self.output.read_bytes())

    def test_historical_partial_without_intent_is_preserved(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b"DEBUG historical unclassified artifact")
        before = self.output.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "preserved"):
            self.prepare(overwrite=True)
        self.assertEqual(before, self.output.read_bytes())
        self.assertFalse((self.output.parent / "publication_intent.json").exists())

    def test_audit_cli_is_read_only_and_report_is_create_only(self):
        from tools.audit_population_bank import main

        self.prepare()
        before = self.output.read_bytes()
        stdout = io.StringIO()
        report_path = self.root / "audit.json"
        with redirect_stdout(stdout):
            main(["--prototype-bank", str(self.output), "--output", str(report_path)])
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["membership_verified"])
        self.assertTrue(report["requires_external_training_cohort_provenance"])
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), report)
        report_before = report_path.read_bytes()
        with self.assertRaises(FileExistsError):
            main(["--prototype-bank", str(self.output), "--output", str(report_path)])
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(report_path.read_bytes(), report_before)


if __name__ == "__main__":
    unittest.main()
