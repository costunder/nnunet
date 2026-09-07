"""DEBUG recovery contracts: temporary fixtures, no medical training/GPU runs."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
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

    @staticmethod
    def policy_fixture():
        current = recovery.read_json(Path(__file__).resolve().parents[1] / "config/train.json")
        current["cache"].update(source_pad=2, min_center_separation_mm=0.0,
            min_center_separation_vox=12.0, no_placement_policy="retain_original")
        old = {"graph_config": copy.deepcopy(current["graph"]), "labels": current["labels"],
               "ct_clip": current["ct_clip"], "seed": current["seed"] + 2,
               **current["cache"], "difficulty_fractions": {
                   "easy": current["cache"]["easy_fraction"], "inter": current["cache"]["inter_fraction"],
                   "intra_corrupted": current["cache"]["intra_fraction"]}}
        old.update(source_pad=4, min_center_separation_mm=12.0)
        old.pop("min_center_separation_vox")
        old.pop("no_placement_policy")
        return current, old

    def test_medical_aug_policy_restoration_rebuilds_cp_graphs_but_reuses_shared_features(self):
        current, old = self.policy_fixture()
        original = copy.deepcopy(old)
        contract = recovery.recovery_cache_contract(current, old, fold=2)
        self.assertFalse(contract["graph_cache_reuse"])
        self.assertTrue(contract["shared_regions_and_prototype_reuse"])
        self.assertEqual(contract["old_policy"]["no_placement_policy"], "error")
        self.assertEqual(contract["old_policy"]["min_center_separation_vox"], 0)
        self.assertEqual(old, original)
        unchanged = copy.deepcopy(old)
        unchanged.update(current["cache"])
        self.assertTrue(recovery.recovery_cache_contract(current, unchanged, fold=2)["graph_cache_reuse"])

    def test_policy_restoration_does_not_authorize_graph_cohort_or_curriculum_reduction(self):
        current, old = self.policy_fixture()
        for group, name, value in (("cache", "total_candidates", 4), ("cache", "samples_per_case", 1),
                                   ("cache", "candidate_pool_size", 16), ("graph", "num_regions", 2),
                                   ("cache", "min_liver_coverage", 0.5),
                                   ("cache", "no_placement_policy", "ignore_any_error")):
            changed = copy.deepcopy(current)
            changed[group][name] = value
            with self.subTest(group=group, name=name), self.assertRaises(ValueError):
                recovery.recovery_cache_contract(changed, old, fold=2)
        with self.assertRaisesRegex(ValueError, "seed"):
            recovery.recovery_cache_contract(current, old, fold=0)

    @staticmethod
    def no_placement_error():
        from hiercp.common import CandidatePreparationError, CANDIDATE_SEARCH_VERSION
        return CandidatePreparationError("insufficient_valid_candidate_pool", {
            "case_id": "DEBUG", "sample_index": 0, "source_component": 1,
            "candidate_search_version": CANDIDATE_SEARCH_VERSION,
            "pool": {"source_component": 1, "search_version": CANDIDATE_SEARCH_VERSION,
                     "fullsearch_exhausted": True, "exhaustive_used": True, "accepted": 0,
                     "required_candidates": 7, "required_candidates_met": False, "excluded_center_count": 0,
                     "max_draws": 50000, "min_liver_coverage": 0.85, "occupied_clearance_vox": 2,
                     "min_center_separation_mm": 0.0, "min_center_separation_vox": 12.0,
                     "target_candidates": 128},
            "rejected_centers": [], "geometry_rejections": [], "curriculum_failure": None})

    def test_no_placement_pilot_requires_exhaustive_evidence_and_explicit_policy(self):
        from hiercp.cache import no_placement_evidence
        current, _ = self.policy_fixture()
        request = {"config": current, "case_id": "DEBUG", "sample_index": 0}
        evidence = no_placement_evidence(self.no_placement_error(), case_id="DEBUG", sample_index=0,
                                         required_candidates=7)
        self.assertIsNotNone(evidence)
        result = {"format": "hiercp_full_size_resource_pilot_v1", "calibration_only": True,
                  "training_performed": False, "case_id": "DEBUG", "sample_index": 0,
                  "request_sha256": recovery.value_sha(request),
                  "roi_budget": current["graph"]["adaptive_roi_max_voxels"],
                  "sample_outcome": "retain_original", "candidate_count": 0, "required_candidate_count": 8,
                  "no_placement_evidence": evidence,
                  "measurement": {"status": "complete", "sampled_peak_rss_bytes": 4096, "elapsed_seconds": 0.1}}
        recovery.validate_pilot(result, request)
        for field, value in (("accepted", 1), ("fullsearch_exhausted", False), ("required_candidates", 6),
                             ("min_center_separation_mm", 12.0)):
            changed = copy.deepcopy(result)
            changed["no_placement_evidence"]["diagnostics"]["pool"][field] = value
            body = {k: v for k, v in changed["no_placement_evidence"].items() if k != "evidence_sha256"}
            changed["no_placement_evidence"]["evidence_sha256"] = recovery.value_sha(body)
            with self.subTest(field=field), self.assertRaises(ValueError):
                recovery.validate_pilot(changed, request)
        legacy = copy.deepcopy(request)
        legacy["config"]["cache"]["no_placement_policy"] = "error"
        with self.assertRaisesRegex(ValueError, "not authorized"):
            recovery.validate_pilot({**result, "request_sha256": recovery.value_sha(legacy)}, legacy)

    def test_profile_records_zero_placement_without_fabricating_a_graph(self):
        from hiercp import cache, common, preparation_runtime, prototype, region
        current, _ = self.policy_fixture()

        class DebugMeasurement:
            """DEBUG context-status double, never claimed as a resource measurement."""
            def __enter__(self):
                return self

            def __exit__(self, error_type, error, traceback):
                self.report = {"status": "complete" if error_type is None else "failed",
                               "sampled_peak_rss_bytes": 4096, "elapsed_seconds": 0.1,
                               "debug_double": True}

        with tempfile.TemporaryDirectory(prefix="DEBUG_no_placement_pilot_") as directory:
            root = Path(directory)
            request = {"config": current, "split": {"train": ["DEBUG"], "val": []},
                       "gnn_root": str(root / "gnn"), "medical_root": str(root / "medical"),
                       "case_id": "DEBUG", "sample_index": 0, "code_sha256": {"DEBUG": "hash"},
                       "source": {"image_sha256": "DEBUG-sha", "label_sha256": "DEBUG-sha"},
                       "prototype_sha256": "DEBUG-sha"}
            output = root / "pilot.json"
            case = SimpleNamespace(paths=SimpleNamespace(image_path="DEBUG-image", label_path="DEBUG-label"))
            with mock.patch.object(recovery, "preparation_code_identity", return_value=request["code_sha256"]), \
                    mock.patch.object(recovery, "digest", return_value="DEBUG-sha"), \
                    mock.patch.object(common, "discover_cases", return_value=["DEBUG-path"]), \
                    mock.patch.object(common, "load_case", return_value=case), \
                    mock.patch.object(region, "load_or_build_patient_regions", return_value=object()), \
                    mock.patch.object(prototype.PrototypeBank, "load", return_value=object()), \
                    mock.patch.object(preparation_runtime, "Measurement", DebugMeasurement), \
                    mock.patch.object(cache, "build_training_sample", side_effect=self.no_placement_error()) as build:
                recovery._profile(request, output)
            build.assert_called_once()
            result = recovery.read_json(output)
            self.assertEqual(result["sample_outcome"], "retain_original")
            self.assertEqual(result["candidate_count"], 0)
            self.assertEqual(result["measurement"]["status"], "complete")
            self.assertFalse(result["training_performed"])
            recovery.validate_pilot(result, request)
            self.assertEqual(list(root.rglob("*.pt")), [])


if __name__ == "__main__":
    unittest.main()
