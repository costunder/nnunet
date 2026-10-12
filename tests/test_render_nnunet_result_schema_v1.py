"""CPU DEBUG report fixtures; no scientific metrics, training, or GPU execution."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("result_schema1_renderer", ROOT / "tools/render_nnunet_result_schema_v1.py")
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)
SCHEMA = json.loads(renderer.SCHEMA_PATH.read_text(encoding="utf-8"))
NAME = "CPU DEBUG fixture / GPU6 / report-layout-only"


def base_evidence():
    return {
        "kind": "CPU_DEBUG_report_fixture_not_actual_experiment",
        "experiments": [{
            "experiment": NAME,
            "completion": {"training_status": "VERIFIED_COMPLETE", "completed_epochs": 250,
                           "planned_epochs": 250, "expected_cases": 26},
            "training": {"best_current_epoch_raw": 236, "best_selection_rule": "native EMA Pseudo dice",
                         "latest_loss_epoch": 249, "train_loss": -0.731234567891,
                         "val_loss": -0.721987654321, "online_pseudo_dice": [0.969134567891, 0.781987654321],
                         "cp_applied": 207, "cp_attempted": 500, "cp_apply_rate": 0.414,
                         "training_seconds": 18920.125, "measurement_scope": "CPU DEBUG fixture only"},
            "settings": {"cp": {"source_count": 642, "positions_per_source": 128, "probability": 0.5},
                         "nnunet": {"train_cases": 105, "validation_cases": 26, "physical_batch": 2,
                                    "planned_epochs": 250}},
            "native": {
                "summary": {"mean": {"3": {"Dice": 0.965432198765}, "7": {"Dice": 0.69123456789}}},
                "dataset_labels": {"background": 0, "liver": 3, "tumor": 7},
                "receipt": {"checkpoint_name": "checkpoint_best.pth", "best_current_epoch": 236},
            },
        }],
        "comparison": {"status": "HISTORICAL_REFERENCE", "reason": "Basic original training-BEST linkage unverified"},
        "limitations": ["CPU DEBUG fixture tests the report only; no actual model/data execution"],
    }


def summary_fixture():
    summary = {
        "version": "CPU_DEBUG_fixture_online_eval_v2",
        "evaluation_contract": {"contract_sha256": "c" * 64},
        "source_files": {"CPU_DEBUG_metric_fixture.py": "e" * 64},
        "prediction_origin_status": "existing_prediction_bytes_verified; original_training_checkpoint_link_unverified",
        "validation_cases": 26,
        "case_tumor_dice": {"basic_mean": 0.651234567891, "hier_mean": 0.713456789123,
                            "statistics": {"difference": 0.062222221232, "ci_low": 0.002345678912,
                                           "ci_high": 0.102345678912, "permutation_p": 0.034567891234,
                                           "wilcoxon_p": 0.023456789123}},
        "lesion_dice_quality": {"basic_mean": 0.511234567891, "hier_mean": 0.613456789123,
                                "difference": 0.102222221232, "ci_low": 0.042345678912,
                                "ci_high": 0.152345678912, "permutation_p": 0.045678912345},
        "criteria": {},
        "resampling": {"cluster_unit": "validation patient", "bootstrap_iterations": 17,
                       "permutation_iterations": 19},
    }
    for criterion in SCHEMA["detection_criteria_order"]:
        summary["criteria"][criterion] = {
            "basic_cp": {"tp": 12, "fp": 9, "fn": 8, "recall": 0.6,
                         "precision": 0.5714285714285714, "f1": 0.5853658536585366,
                         "fp_per_case": 0.34615384615384615},
            "hiercp": {"tp": 14, "fp": 8, "fn": 6, "recall": 0.7,
                       "precision": 0.6363636363636364, "f1": 0.6666666666666666,
                       "fp_per_case": 0.3076923076923077},
            "statistics": {metric: {"difference": 0.012345678912, "ci_low": -0.012345678912,
                                     "ci_high": 0.112345678912, "permutation_p": 0.234567891234}
                           for metric in ("recall", "precision", "f1", "fp_per_case")},
            "by_size": {bin_name: {"gt": 8, "basic_detected": 3, "hier_detected": 4,
                                   "basic_recall": 0.375, "hier_recall": 0.5,
                                   "statistics": {"difference": 0.125, "ci_low": -0.1,
                                                  "ci_high": 0.4, "permutation_p": 0.3}}
                        for bin_name in SCHEMA["size_bins_order"]},
        }
    return summary


def completed_outputs(folder, summary=None):
    """Seal DEBUG fixture bytes to exercise production admission, not metric math."""
    summary = summary or summary_fixture()
    for name in SCHEMA["existing_metric_artifact_names"]:
        if name == "completion.json":
            continue
        if name == "summary.json":
            content = json.dumps(summary, ensure_ascii=False)
        elif name.endswith(".csv"):
            content = "CPU_DEBUG_fixture,scope\nreport-only,not-a-real-result\n"
        elif name.endswith(".json"):
            content = json.dumps({"kind": "CPU_DEBUG_report_fixture"})
        else:
            content = "CPU DEBUG fixture; report validation only\n"
        (folder / name).write_text(content, encoding="utf-8")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()
              if p.is_file() and p.name != "completion.json"}
    completion = {"format": "online_eval_completion_v1", "complete": True,
                  "actual_data_evaluation": True, "training_executed": False,
                  "summary_sha256": hashes["summary.json"], "outputs": hashes,
                  "evaluation_contract_sha256": summary["evaluation_contract"]["contract_sha256"]}
    (folder / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
    return completion


class ResultSchema1RendererTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="CPU_DEBUG_schema1_")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def with_metrics(self, summary=None):
        completed_outputs(self.folder, summary)
        evidence = base_evidence()
        evidence["online_evaluation"] = {"experiment": NAME, "directory": str(self.folder)}
        return evidence

    def test_exact_schema_sections_columns_and_missing_basic_rows(self):
        report, text = renderer.render_report(base_evidence())
        self.assertEqual(report["section_order"], SCHEMA["section_order"])
        self.assertEqual(report["table_columns"], SCHEMA["tables"])
        positions = [text.index(f"## {index:02d} ") for index in range(1, 9)]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(text.count("## 작업 완료 체크리스트"), 1)
        self.assertEqual(len(report["tables"]["whole_tumor_segmentation"]), 2)
        self.assertEqual(report["tables"]["whole_tumor_segmentation"][0]["experiment"], "Basic CP")
        detection = report["tables"]["whole_lesion_detection"]
        self.assertEqual(len(detection), 12)
        self.assertEqual([r["criterion"] for r in detection[:6]], SCHEMA["detection_criteria_order"])
        sizes = report["tables"]["size_stratified_detection"]
        self.assertEqual(len(sizes), 36)
        self.assertEqual([(r["criterion"], r["size_bin"]) for r in sizes[:18]],
                         [(c, b) for c in SCHEMA["detection_criteria_order"] for b in SCHEMA["size_bins_order"]])
        comparisons = report["tables"]["basic_cp_comparison"]
        self.assertEqual(len(comparisons), 44)
        self.assertTrue(all(row["experiment_minus_basic"] is None for row in comparisons))
        self.assertTrue(all(row["_missing"]["experiment_minus_basic"] for row in comparisons))
        self.assertTrue(all(row["recall"] is None for row in detection))
        self.assertIn("NA [N", text)
        self.assertIn("NA 사유:", text)
        self.assertIn(renderer.NO_METRICS, text)
        self.assertEqual(text.count("- [ ]"), 13)

    def test_preserves_raw_precision_epoch_index_and_observed_label_semantics(self):
        report, text = renderer.render_report(base_evidence())
        train = report["tables"]["nnunet_training_record"][0]
        self.assertEqual(train["best_current_epoch_raw"], 236)
        self.assertEqual(train["latest_loss_epoch"], 249)
        self.assertEqual(train["train_loss"], -0.731234567891)
        self.assertEqual(train["online_pseudo_dice"], [0.969134567891, 0.781987654321])
        seg = report["tables"]["whole_tumor_segmentation"][1]
        self.assertEqual(seg["native_liver_dice"], 0.965432198765)
        self.assertEqual(seg["native_tumor_dice"], 0.69123456789)
        self.assertIsNone(seg["case_macro_tumor_dice"])
        self.assertIn("0/1-based 정의 근거 없음; 추정하지 않음", text)
        self.assertIn("시작/종료 시각: NA", text)
        self.assertNotIn("Epoch 237", text)
        self.assertIn("0.9654", text)
        self.assertIn("-0.7312", text)

    def test_native_label_mapping_missing_or_foreign_never_guessed(self):
        evidence = base_evidence()
        native = evidence["experiments"][0]["native"]
        native.pop("dataset_labels")
        report, _ = renderer.render_report(evidence)
        self.assertIsNone(report["tables"]["whole_tumor_segmentation"][1]["native_tumor_dice"])
        native["dataset_labels"] = {"background": 0, "liver": 3, "tumor": 7}
        native["tumor_label"] = 2
        with self.assertRaisesRegex(renderer.EvidenceError, "label is not"):
            renderer.render_report(evidence)

    def test_nested_settings_keep_each_explicit_missing_reason_in_text(self):
        evidence = base_evidence()
        settings = evidence["experiments"][0]["settings"]
        settings["experiment"] = {"native_seed": {"value": None, "reason": "native seed was not captured; GNN seed is separate"}}
        settings["cp"]["HU_jitter"] = {"value": None, "reason": "HU jitter numeric configuration was not collected"}
        settings["nnunet"]["precision"] = {"value": None, "reason": "full native training precision was not measured"}
        report, text = renderer.render_report(evidence)
        for label in ("native seed was not captured; GNN seed is separate",
                      "HU jitter numeric configuration was not collected",
                      "full native training precision was not measured"):
            self.assertIn(label, text)
        metadata = report["experiment_metadata"][0]["settings"]
        self.assertIsNone(metadata["nnunet"]["precision"]["value"])
        self.assertIn("precision=NA [N", text)

    def test_full_native_complete_requires_best_load_and_own_receipt_verification(self):
        evidence = base_evidence()
        arm = evidence["experiments"][0]
        arm["completion"]["best_native_evaluation_status"] = "VERIFIED_COMPLETE"
        with self.assertRaisesRegex(renderer.EvidenceError, "verified actual BEST receipt"):
            renderer.render_report(evidence)
        arm["native"]["receipt"].update(complete=True, original_checkpoint_load_called=True,
            original_full_validation_called=True, final_checkpoint_fallback=False,
            checkpoint_sha256="a" * 64, summary_sha256="b" * 64,
            validation_cases=[f"DEBUG_case_{index:02d}" for index in range(26)],
            predictions_sha256={f"DEBUG_case_{index:02d}": "c" * 64 for index in range(26)})
        arm["native"]["verification"] = {"best_receipt_verified": True, "source": "CPU_DEBUG collector fixture"}
        report, _ = renderer.render_report(evidence)
        self.assertEqual(report["tables"]["completion_and_best_provenance"][0]["predicted_cases"], 26)
        for key, value in (("checkpoint_name", "checkpoint_final.pth"), ("final_checkpoint_fallback", True),
                           ("original_checkpoint_load_called", False)):
            changed = copy.deepcopy(evidence)
            changed["experiments"][0]["native"]["receipt"][key] = value
            with self.assertRaises(renderer.EvidenceError):
                renderer.render_report(changed)

    def test_completed_metric_bytes_keep_producer_metrics_and_historical_stats_separate(self):
        evidence = self.with_metrics()
        report, text = renderer.render_report(evidence)
        completion = report["tables"]["completion_and_best_provenance"][0]
        self.assertEqual(completion["lesion_metrics_status"], "VERIFIED_COMPLETE")
        seg = report["tables"]["whole_tumor_segmentation"][1]
        self.assertEqual(seg["case_macro_tumor_dice"], 0.713456789123)
        self.assertEqual(seg["lesion_mean_dice"], 0.613456789123)
        self.assertNotEqual(seg["case_macro_tumor_dice"], seg["native_tumor_dice"])
        det = report["tables"]["whole_lesion_detection"][6]
        self.assertEqual(det["gt"], 20)
        self.assertEqual(det["precision"], 0.6363636363636364)
        self.assertEqual(det["fp_per_case"], 0.3076923076923077)
        size = report["tables"]["size_stratified_detection"][18]
        self.assertEqual((size["gt"], size["detected"], size["missed"]), (8, 4, 4))
        rows = report["tables"]["basic_cp_comparison"]
        self.assertTrue(all(row["experiment_minus_basic"] is None for row in rows))
        refs = report["historical_reference_comparison"]
        self.assertEqual(len(refs), 44)
        self.assertEqual(refs[0]["experiment_minus_basic"], 0.062222221232)
        self.assertEqual(refs[0]["wilcoxon_p"], 0.023456789123)
        self.assertTrue(all(row["wilcoxon_p"] is None for row in refs[1:]))
        self.assertEqual(report["online_evaluation"]["summary"]["prediction_origin_status"],
                         summary_fixture()["prediction_origin_status"])
        self.assertIn("역사적 참고 차이", text)
        self.assertIn("0.308", text)
        self.assertEqual(len(report["online_evaluation"]["artifacts"]), 11)

    def test_matched_comparison_admission_cannot_be_claimed_from_filename(self):
        evidence = self.with_metrics()
        evidence["comparison"] = {"status": "MATCHED_VERIFIED", "source": "CPU_DEBUG admission fixture"}
        with self.assertRaisesRegex(renderer.EvidenceError, "explicit comparison admissions"):
            renderer.render_report(evidence)
        checks = {key: True for key in ("cohort", "gt_spacing", "own_best", "basic_best", "metric_implementation",
                                       "aggregation_matching_empty_mask", "native_training", "cp_policy")}
        evidence["comparison"]["admission"] = checks
        report, _ = renderer.render_report(evidence)
        self.assertEqual(report["tables"]["basic_cp_comparison"][0]["experiment_minus_basic"], 0.062222221232)
        self.assertEqual(report["historical_reference_comparison"], [])
        for key in checks:
            changed = copy.deepcopy(evidence)
            changed["comparison"]["admission"][key] = False
            with self.assertRaises(renderer.EvidenceError):
                renderer.render_report(changed)

    def test_incomplete_and_tampered_publication_are_not_completed_metrics(self):
        evidence = base_evidence()
        (self.folder / "summary.json").write_text(json.dumps(summary_fixture()), encoding="utf-8")
        evidence["online_evaluation"] = {"experiment": NAME, "directory": str(self.folder)}
        report, _ = renderer.render_report(evidence)
        self.assertEqual(report["tables"]["completion_and_best_provenance"][0]["lesion_metrics_status"], "INCOMPLETE")
        self.assertIsNone(report["tables"]["whole_tumor_segmentation"][1]["case_macro_tumor_dice"])
        completed_outputs(self.folder)
        (self.folder / "criterion_summary.csv").write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(renderer.EvidenceError, "SHA mismatch"):
            renderer.render_report(evidence)
        completed_outputs(self.folder)
        completion = json.loads((self.folder / "completion.json").read_text(encoding="utf-8"))
        completion["evaluation_contract_sha256"] = "foreign"
        (self.folder / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
        with self.assertRaisesRegex(renderer.EvidenceError, "binding differs"):
            renderer.render_report(evidence)

    def test_undefined_stats_preserve_null_reason_and_resampling_counts(self):
        summary = summary_fixture()
        stats = summary["criteria"]["dice_ge_0p10"]["statistics"]["precision"]
        stats.update(ci_low=None, ci_high=None, permutation_p=None,
                     bootstrap_inference={"reason": "undefined_metric_in_resamples", "valid_resamples": 14, "invalid_resamples": 3})
        report, text = renderer.render_report(self.with_metrics(summary))
        row = next(r for r in report["historical_reference_comparison"]
                   if r["criterion"] == "dice_ge_0p10" and r["metric"] == "precision")
        self.assertIsNone(row["ci95_low"])
        self.assertIsNone(row["patient_swap_p"])
        self.assertEqual(row["_original_statistics"]["bootstrap_inference"]["invalid_resamples"], 3)
        self.assertIn("undefined_metric_in_resamples", row["_missing"]["ci95_low"])
        self.assertIn("valid_resamples", text)

    def test_inline_bundle_requires_bound_collector_verification_and_names(self):
        evidence = base_evidence()
        summary = summary_fixture()
        completion = completed_outputs(self.folder, summary)
        evidence["online_evaluation"] = {"experiment": NAME, "summary": summary, "completion": completion,
                                         "status": "VERIFIED_COMPLETE"}
        with self.assertRaisesRegex(renderer.EvidenceError, "collector output-hash proof"):
            renderer.render_report(evidence)
        evidence["online_evaluation"]["verification"] = {
            "outputs_sha256_verified": True, "source": "CPU_DEBUG collector verification witness"}
        report, _ = renderer.render_report(evidence)
        self.assertIn("collector", report["online_evaluation"]["verification_scope"])
        evidence["online_evaluation"]["experiment"] = "foreign GPU/run"
        with self.assertRaisesRegex(renderer.EvidenceError, "exact report experiment"):
            renderer.render_report(evidence)

    def test_no_metric_library_or_gpu_execution_and_zero_is_real_zero(self):
        evidence = base_evidence()
        evidence["experiments"][0]["training"]["cp_applied"] = 0
        evidence["experiments"][0]["training"]["cp_apply_rate"] = 0.0
        report, text = renderer.render_report(evidence)
        self.assertFalse(report["metric_math_executed"])
        self.assertFalse(report["training_executed_by_renderer"])
        self.assertEqual(report["tables"]["nnunet_training_record"][0]["cp_apply_rate"], 0.0)
        self.assertIn("0/500(0.0000)", text)
        source = Path(renderer.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("import numpy", source)
        evidence["experiments"][0]["training"]["val_loss"] = float("nan")
        with self.assertRaisesRegex(renderer.EvidenceError, "Nonfinite"):
            renderer.render_report(evidence)

    def test_fresh_publication_preserves_precision_and_refuses_every_existing_result(self):
        stem = self.folder / "fresh_report"
        text_path, json_path = renderer.write_report(base_evidence(), stem)
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["tables"]["nnunet_training_record"][0]["train_loss"], -0.731234567891)
        original = {p: p.read_bytes() for p in (text_path, json_path)}
        with self.assertRaises(FileExistsError):
            renderer.write_report(base_evidence(), stem)
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)

    def test_second_output_failure_cleans_only_own_json_and_preserves_foreign_text(self):
        stem = self.folder / "race_report"
        json_path, text_path = Path(str(stem) + ".json"), Path(str(stem) + ".txt")
        original_open = Path.open

        def exclusive_race(path, *args, **kwargs):
            if path == text_path and args and args[0] == "x":
                text_path.write_text("foreign report must survive", encoding="utf-8")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", exclusive_race):
            with self.assertRaises(FileExistsError):
                renderer.write_report(base_evidence(), stem)
        self.assertFalse(json_path.exists())
        self.assertEqual(text_path.read_text(encoding="utf-8"), "foreign report must survive")


if __name__ == "__main__":
    unittest.main()
