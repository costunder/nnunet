"""DEBUG recovery contracts: temporary fixtures, no medical training/GPU runs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import feedback_preparation_recovery as recovery


class RecoveryDebugTests(unittest.TestCase):
    def fixture(self, root):
        project = root / "DEBUG checkout"
        source = project / "work/failed"
        target = project / "work/recovered"
        medical = root / "DEBUG medical"
        plan = {"project_root": project, "run_root": target, "medical_root": medical,
                "outer_fold": 0, "dataset_id": 760, "seed": 42}
        gnn = source / "paired/folds/fold_0/gnn"
        old = {"project_root": str(project), "run_root": str(source), "medical_root": str(medical),
               "commands": [{"name": "gnn-prepare", "argv": ["DEBUG", "--outer-fold", "0"]},
                            {"name": "plan", "argv": ["DEBUG", "--dataset-id", "760"]},
                            {"name": "train_full", "argv": ["DEBUG", "--seed", "42"]}]}
        split = {"train": ["DEBUG_A", "DEBUG_B"], "val": ["DEBUG_C"], "outer_validation_excluded": ["DEBUG_TEST"]}
        config = {"state": "failed", "run_mode": "benchmark", "subset_active": False,
                  "selected_case_ids": split["train"] + split["val"],
                  "train_case_ids": split["train"], "val_case_ids": split["val"]}
        recovery.write_new(source / "launch_plan.json", old)
        recovery.write_new(gnn / "split.json", split)
        recovery.write_new(gnn / "graphs/config.json", config)
        for name in ("paired/outer_splits.json", "paired/case_profiles.csv",
                     "paired/folds/fold_0/gnn/prototype.pt", "paired/folds/fold_0/gnn/metadata.json",
                     "paired/folds/fold_0/gnn/manifest.csv", "paired/folds/fold_0/gnn/graphs/manifest.csv"):
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("DEBUG opaque fixture: never loaded as a model", encoding="utf-8")
        return plan, source, gnn

    def test_legacy_source_identity_is_read_only(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_recovery_") as directory:
            plan, source, _ = self.fixture(Path(directory))
            before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            identity = recovery.validate_recovery_source(plan, source)
            recovery.verify_identity(identity)
            self.assertEqual(identity["seed"], 42)
            self.assertEqual(len(identity["files"]), 9)
            self.assertFalse(plan["run_root"].exists())
            self.assertEqual(before, {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()})

    def test_rejects_identity_changes(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_recovery_") as directory:
            plan, source, _ = self.fixture(Path(directory))
            for key, value in (("outer_fold", 1), ("dataset_id", 761), ("seed", 7)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    recovery.validate_recovery_source({**plan, key: value}, source)

    def test_rejects_test_leakage_and_published_cache(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_recovery_") as directory:
            plan, source, gnn = self.fixture(Path(directory))
            split = recovery.read_json(gnn / "split.json")
            split["outer_validation_excluded"].append("DEBUG_A")
            (gnn / "split.json").write_text(json.dumps(split), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Outer validation"):
                recovery.validate_recovery_source(plan, source)
            split["outer_validation_excluded"].remove("DEBUG_A")
            (gnn / "split.json").write_text(json.dumps(split), encoding="utf-8")
            recovery.write_new(gnn / "graphs/complete.json", {})
            with self.assertRaisesRegex(ValueError, "published"):
                recovery.validate_recovery_source(plan, source)

    def test_rejects_overlapping_paths(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_recovery_") as directory:
            plan, source, _ = self.fixture(Path(directory))
            for target in (source, source / "child", source.parent):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    recovery.validate_recovery_source({**plan, "run_root": target}, source)

    def test_changed_source_receipt_fails(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_recovery_") as directory:
            plan, source, gnn = self.fixture(Path(directory))
            identity = recovery.validate_recovery_source(plan, source)
            (gnn / "prototype.pt").write_bytes(b"DEBUG changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                recovery.verify_identity(identity)

    def test_copy_is_independent_and_non_overwriting(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_copy_") as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            source.write_bytes(b"DEBUG original bytes")
            recovery._copy_verified(source, target)
            recovery._copy_verified(source, target)
            self.assertEqual(source.read_bytes(), target.read_bytes())
            target.write_bytes(b"DEBUG only destination changed")
            self.assertEqual(source.read_bytes(), b"DEBUG original bytes")
            with self.assertRaises(ValueError):
                recovery._copy_verified(source, target)
            self.assertEqual(target.read_bytes(), b"DEBUG only destination changed")

    def test_interrupted_copy_does_not_publish_partial_destination(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_copy_") as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            source.write_bytes(b"DEBUG original")
            with mock.patch.object(recovery.shutil, "copyfileobj", side_effect=OSError("DEBUG interruption")):
                with self.assertRaises(OSError):
                    recovery._copy_verified(source, target)
            self.assertFalse(target.exists())
            self.assertEqual(source.read_bytes(), b"DEBUG original")
            recovery._copy_verified(source, target)
            self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_roi_envelope_includes_all_lesions_not_only_selected_sample(self):
        config = {"cache": {"source_pad": 4}, "graph": {"adaptive_roi_max_radius_mm": 64,
            "adaptive_roi_margin_mm": 30, "context_outer_radius_mm": 28}}
        eligibility = {"cases": [
            {"case_id": "DEBUG_A", "spacing_mm": [0.7, 0.8, 1.2], "component_bbox_shapes": [[3, 5, 7], [81, 91, 85]]},
            {"case_id": "DEBUG_NEG", "spacing_mm": [1, 1, 1], "component_bbox_shapes": []}]}
        result = recovery.geometry_envelope(eligibility, config)
        self.assertFalse(result["memory_measured"])
        self.assertGreater(result["maximum_scale"], 1.60)
        self.assertEqual(result["cases"][0]["components"], 2)
        self.assertEqual(result["cases"][1]["roi_envelope_voxels"], 0)
        reduced = copy.deepcopy(eligibility)
        reduced["cases"][0]["component_bbox_shapes"].pop()
        self.assertGreater(result["maximum_voxels"], recovery.geometry_envelope(reduced, config)["maximum_voxels"])

    def test_roi_envelope_rejects_empty_or_invalid_physical_contract(self):
        config = {"cache": {"source_pad": 4}, "graph": {"adaptive_roi_max_radius_mm": 64,
            "adaptive_roi_margin_mm": 30, "context_outer_radius_mm": 28}}
        with self.assertRaisesRegex(ValueError, "No eligible"):
            recovery.geometry_envelope({"cases": []}, config)
        eligibility = {"cases": [{"case_id": "DEBUG", "spacing_mm": [0, 1, 1], "component_bbox_shapes": [[3, 3, 3]]}]}
        with self.assertRaisesRegex(ValueError, "spacing"):
            recovery.geometry_envelope(eligibility, config)

    def test_json_outputs_refuse_replacement(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_json_") as directory:
            path = Path(directory) / "receipt.json"
            recovery.write_new(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                recovery.write_new(path, {"value": 2})
            self.assertEqual(recovery.read_json(path), {"value": 1})

    def test_pilot_reuse_requires_complete_current_request_and_candidate_count(self):
        request = {"case_id": "DEBUG", "sample_index": 0,
                   "config": {"graph": {"adaptive_roi_max_voxels": 9000000}, "cache": {"total_candidates": 8}}}
        result = {"format": "hiercp_full_size_resource_pilot_v1", "calibration_only": True,
                  "training_performed": False, "case_id": "DEBUG", "sample_index": 0,
                  "request_sha256": recovery.value_sha(request), "roi_budget": 9000000,
                  "candidate_count": 8, "measurement": {"status": "complete", "sampled_peak_rss_bytes": 4096,
                                                        "elapsed_seconds": 0.1}}
        recovery.validate_pilot(result, request)
        for field, wrong in (("request_sha256", "stale"), ("candidate_count", 7), ("training_performed", True)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                recovery.validate_pilot({**result, field: wrong}, request)
        for field, wrong in (("status", "failed"), ("sampling_error", "failed"), ("elapsed_seconds", 0),
                             ("sampled_peak_rss_bytes", float("nan"))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                recovery.validate_pilot({**result, "measurement": {**result["measurement"], field: wrong}}, request)

    def test_json_rejects_duplicate_keys_and_nonfinite_evidence(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_json_") as directory:
            path = Path(directory) / "receipt.json"
            for text in ('{"sha": 1, "sha": 2}', '{"rss": NaN}', '{"rss": Infinity}'):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    recovery.read_json(path)


if __name__ == "__main__":
    unittest.main()
