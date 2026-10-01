"""Saved diagnostic summaries must not execute or alter the experiment."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.summarize_local_cnn_diagnosis import format_summary, main, read_summary


def fixture():
    def spread(value):
        return {"normalized_centered_energy": value}
    def stats(std):
        return {"score_std": std, "mean_positive_minus_unobserved": -std,
                "pair_win_rate": 0.25, "mean_pairwise_loss": 0.693}
    layers = [{"layer": layer, "stages": {key: spread(value) for key, value in (
        ("input", 1e-6), ("projected_message", 1e-9), ("residual_add", 2e-7),
        ("residual_norm", 3e-8), ("ff_output", 4e-9), ("second_add", 6e-9), ("final_norm", 5e-9))}}
        for layer in (1, 2)]
    case = {"split": "train", "case_id": "liver_1", "records": 129, "positives": 1,
            "full_support": {"score": stats(0.0001), "metrics": {"ranking_mrr": 0.16}},
            "l0_substages": {"stages": {key: {"status": "MEASURED", **spread(value)}
                for key, value in (("global_mean_concat_recipient", 1e-5),
                                   ("project_recipient", 2e-6), ("fusion_input", 7e-6), ("fusion_output", 1e-6),
                                   ("scale1_anchor_roi_recipient", 3e-5))}},
            "l1_substages_and_message_sweep": {
                "production_scale_parity_passed": True,
                "message_scale_sweep": [{"message_scale": scale, "score": stats(0.0001),
                    "layers": copy.deepcopy(layers)} for scale in (0, 0.25, 0.5, 1)]}}
    case["l0_substages"]["stages"]["scale2_anchor_roi_recipient"] = {
        "status": "NOT_RUN", "reason": "No organ-supported cell; no candidate was skipped"}
    case["l0_substages"]["stages"]["scale3_anchor_roi_recipient"] = {
        "status": "NOT_RUN", "reason": "No organ-supported cell; no candidate was skipped"}
    direction = {"cosine": -0.2}
    shadow = {"production_optimizer_updates": 0, "cloned_optimizer_steps": 4,
              "losses": {"ranking": 0.6, "observation_ce": 0.7, "alignment": 0.1, "full": 1.4},
              "gradients": {}, "branches": {}, "optimizer_state_reused": True,
              "decomposition_full_backward_verified": True,
              "same_forward_realization_for_all_branches": True,
              "saved_payload_unchanged": True, "support_plan_unchanged": True,
              "caller_rng_restored": True}
    for module in ("CNN", "readout_fusion", "L1", "L2", "global"):
        shadow["gradients"][module] = {key: direction for key in (
            "observation_ce_vs_ranking", "alignment_vs_ranking", "auxiliary_sum_vs_ranking")}
    for branch in ("ranking", "observation_ce", "alignment", "full"):
        shadow["branches"][branch] = {
            "clipping_factor": 0.5,
            "modules": {module: {"delta_norm": 0.01, "vs_negative_ranking_gradient": direction}
                        for module in shadow["gradients"]},
            "vs_full_update": {module: {"cosine": 0.8} for module in shadow["gradients"]}}
    return {"run": "/Medical/experiments/v22_cnn_m10_seed42", "epoch": 17, "step": 7000,
            "phase": "optimization", "margin_mm": 10, "physical_batch": 32,
            "checkpoint_content_sha256": "a" * 64, "cases": [case], "shadow_update": shadow,
            "diagnostic_only": True, "optimizer_updates": 0, "full_evaluation": False,
            "weights_unchanged": True, "peak_cuda_gib": 4.2}


class SummaryTests(unittest.TestCase):
    def test_four_cases_fit_compact_output_without_metric_omission(self):
        report = fixture()
        report["cases"] = [copy.deepcopy(report["cases"][0]) for _ in range(4)]
        for index, case in enumerate(report["cases"]):
            case["case_id"] = f"liver_{index}"
        original = copy.deepcopy(report)
        result = format_summary(report)
        self.assertLessEqual(len(result.splitlines()), 50)
        self.assertLessEqual(max(map(len, result.splitlines())), 140)
        for part in ("epoch=17 step=7000", "margin_mm=10 batch=32", "L1_1", "L1_2",
                     "s=0.25", "s=0.5", "s=1", "grad cos CE/align/aux=-0.2/-0.2/-0.2",
                     "Adam d R/C/A/F=0.01/0.01/0.01/0.01", "cos(rank,full)=0.8",
                     "mean/project/fusion_in/out=1e-05/2e-06/7e-06/1e-06", "in/msg/add/LN/FF/add2/final=",
                     "Clip R/C/A/F=0.5/0.5/0.5/0.5",
                     "weights=true saved-payload=true support-plan=true caller-RNG=true"):
            self.assertIn(part, result)
        self.assertEqual(report, original)

    def test_missing_fields_are_unavailable_and_never_pass_or_zero(self):
        result = format_summary({"cases": [{"case_id": "missing"}]})
        self.assertIn("epoch=unavailable step=unavailable", result)
        self.assertIn("std=unavailable", result)
        self.assertIn("weights=unavailable saved-payload=unavailable", result)
        self.assertIn("production-updates=unavailable", result)
        self.assertNotIn("PASS", result)
        self.assertNotIn("weights=true", result)

    def test_zero_null_missing_and_nonfinite_are_distinct(self):
        report = fixture()
        report["shadow_update"]["gradients"]["CNN"]["alignment_vs_ranking"] = {"cosine": None}
        report["shadow_update"]["branches"]["alignment"]["modules"]["CNN"]["delta_norm"] = 0
        report["cases"][0]["full_support"]["score"]["score_std"] = float("nan")
        for row in report["cases"][0]["l1_substages_and_message_sweep"]["message_scale_sweep"]:
            row["layers"][0]["stages"]["input"]["normalized_centered_energy"] = 0
        result = format_summary(report)
        self.assertIn("std=nonfinite", result)
        self.assertIn("grad cos CE/align/aux=-0.2/undefined/-0.2", result)
        self.assertIn("Adam d R/C/A/F=0.01/0.01/0/0.01", result)
        self.assertIn("final/input=undefined", result)

    def test_critical_loss_differences_and_exact_counts_are_preserved(self):
        report = fixture()
        report['step'] = 123456789
        report['cases'][0]['records'] = 14102
        sweep = report['cases'][0]['l1_substages_and_message_sweep']['message_scale_sweep']
        sweep[0]['score']['mean_pairwise_loss'] = 0.6932707
        sweep[1]['score']['mean_pairwise_loss'] = 0.6931957
        result = format_summary(report)
        self.assertIn('step=123456789', result)
        self.assertIn('N=14102', result)
        self.assertIn('0.6932707', result)
        self.assertIn('0.6931957', result)

    def test_not_run_reason_retained_and_no_invented_shadow(self):
        report = fixture()
        del report["shadow_update"]
        report["shadow_update_request"] = {"status": "NOT_RUN", "reason": "Snapshot phase=complete"}
        result = format_summary(report)
        self.assertIn("SHADOW NOT_RUN: Snapshot phase=complete", result)
        self.assertIn("No organ-supported cell; no candidate was skipped", result)
        self.assertIn("NOT_RUN/NOT_RUN", result)
        self.assertNotIn("cloned steps=4", result)

    def test_original_report_read_only_new_output_and_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "report.json"
            output = Path(directory) / "summary.txt"
            source.write_text(json.dumps(fixture()), encoding="utf8")
            before = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()) as console:
                main([str(source), "--output", str(output)])
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(output.read_text(encoding="utf8"), read_summary(source) + "\n")
            self.assertIn("SUMMARY SAVED:", console.getvalue())
            with self.assertRaises(FileExistsError):
                main([str(source), "--output", str(output)])
            with self.assertRaises(FileExistsError):
                main([str(source), "--output", str(source)])
            self.assertEqual(source.read_bytes(), before)

    def test_invalid_json_or_root_rejected_without_output(self):
        with self.assertRaises(ValueError):
            format_summary([])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "invalid.json"
            output = Path(directory) / "summary.txt"
            source.write_text("{broken", encoding="utf8")
            with self.assertRaises(json.JSONDecodeError):
                main([str(source), "--output", str(output)])
            self.assertFalse(output.exists())

    def test_summary_and_cli_help_do_not_import_torch(self):
        code = """import sys
class RejectTorch:
    def find_spec(self, name, *args):
        if name == 'torch' or name.startswith('torch.'):
            raise AssertionError('Summary must not import torch')
sys.meta_path.insert(0, RejectTorch())
from tools.summarize_local_cnn_diagnosis import format_summary
print(format_summary({}))
assert 'torch' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        helper = Path(__file__).resolve().parents[1] / "tools/summarize_local_cnn_diagnosis.py"
        help_result = subprocess.run([sys.executable, "-B", str(helper), "--help"], text=True, capture_output=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--output", help_result.stdout)
        self.assertIn("report", help_result.stdout)


if __name__ == "__main__":
    unittest.main()
