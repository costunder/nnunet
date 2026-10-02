"""UNIT metadata/statistics checks only: no neural CPU or CUDA execution.

Synthetic metric rows here are explicitly statistical fixtures, never measured
CT outcomes or completed training checkpoints. Actual CUDA score algebra is
checked separately by the parent's GPU smoke.
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.evaluate_v1_sampling_pair import aggregate_cases, paired_summary, main, evaluate


def row(case, sample, mrr, top1, margin):
    return dict(case_id=case, sample_id=sample, mrr=mrr, top1=top1, margin=margin)


class PairedCaseStatisticsTests(unittest.TestCase):
    def fixture(self):
        # Unequal samples per case deliberately reveal candidate/sample pseudo-replication.
        baseline = [row("case_a", "a1", .25, 0., -.2), row("case_a", "a2", .75, 1., .2),
                    row("case_b", "b1", .5, 0., -.1)]
        candidate = [row("case_a", "a1", .5, 1., .3), row("case_a", "a2", 1., 1., .7),
                     row("case_b", "b1", .5, 0., -.1)]
        return aggregate_cases(baseline, ["case_a", "case_b"]), aggregate_cases(candidate, ["case_a", "case_b"])

    def test_average_within_case_before_pairing(self):
        baseline, candidate = self.fixture()
        self.assertEqual(baseline[0]["mrr"], .5)
        self.assertEqual(baseline[0]["samples"], 2)
        result = paired_summary(baseline, candidate, resamples=1000, confidence=.95, seed=42)
        self.assertEqual(result["resampling_unit"], "case_id")
        self.assertEqual(result["cases"], 2)
        self.assertEqual(result["metrics"]["mrr"]["delta_nested_minus_native"], .125)
        self.assertEqual(result["metrics"]["top1"]["delta_nested_minus_native"], .25)
        self.assertAlmostEqual(result["metrics"]["margin"]["delta_nested_minus_native"], .25)
        self.assertEqual(result["metrics"]["mrr"]["interval"], [0., .25])

    def test_determinism_and_no_quality_verdict(self):
        old, new = self.fixture()
        a = paired_summary(old, new, resamples=137, confidence=.9, seed=7)
        self.assertEqual(a, paired_summary(old, new, resamples=137, confidence=.9, seed=7))
        self.assertFalse(a["graph_quality_passed"])
        self.assertFalse(a["automatic_promotion"])
        self.assertIsNone(a["quality_tolerance"])

    def test_constant_case_delta_interval(self):
        old, new = self.fixture()
        for before, after in zip(old, new):
            for key in ("mrr", "top1", "margin"):
                after[key] = before[key] + .125
        result = paired_summary(old, new, resamples=200, confidence=.95, seed=42)
        self.assertEqual(result["metrics"]["margin"]["interval"], [.125, .125])

    def test_aggregate_rejects_missing_unknown_duplicate_nonfinite(self):
        examples = [([], ["a"]), ([row("b", "b1", .5, 0, 0)], ["a"]),
                    ([row("a", "a1", .5, 0, 0)] * 2, ["a"]),
                    ([row("a", "a1", float("nan"), 0, 0)], ["a"]),
                    ([row("a", "a1", .5, 2, 0)], ["a"]),
                    ([row("a", "a1", .5, 0, 0)], ["a", "a"])]
        for rows, cohort in examples:
            with self.subTest(rows=rows, cohort=cohort), self.assertRaises(ValueError):
                aggregate_cases(rows, cohort)

    def test_paired_rejects_changed_ids_denominators_cohorts(self):
        old, new = self.fixture()
        mutations = []
        for field, value in (("sample_ids", ["other"]), ("samples", 3), ("case_id", "other"), ("margin", float("inf"))):
            changed = copy.deepcopy(new)
            changed[0][field] = value
            mutations.append(changed)
        mutations += [new[:1], new + new[:1]]
        for changed in mutations:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                paired_summary(old, changed, resamples=10, confidence=.95, seed=42)

    def test_explicit_bootstrap_contract_rejected(self):
        old, new = self.fixture()
        for count, confidence in ((1, .95), (0, .95), (True, .95), (10, 0), (10, 1), (10, float("nan"))):
            with self.subTest(count=count, confidence=confidence), self.assertRaises(ValueError):
                paired_summary(old, new, resamples=count, confidence=confidence, seed=42)

    def test_cli_has_no_implicit_uncertainty_or_gpu_budget(self):
        with patch("tools.evaluate_v1_sampling_pair.evaluate") as execute, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                main(["--native-experiment", "UNIT_NATIVE", "--nested-experiment", "UNIT_NESTED", "--output", "UNIT_ONLY"])
        execute.assert_not_called()

    def test_incomplete_result_cannot_launch_neural_evaluation(self):
        # Existing workspace paths only; collector refuses before output creation.
        args = SimpleNamespace(native_experiment=str(Path.cwd()), nested_experiment=str(Path.cwd()),
                               output="UNIT_NOT_CREATED", gpu=0, cuda_gib=8, rss_gib=16,
                               bootstrap_resamples=100, confidence=.95, bootstrap_seed=42)
        with patch("tools.local_cnn_device.select"), \
             patch("hiercp_v1x.results.collect_result", side_effect=ValueError("Incomplete full40 result")), \
             patch("tools.evaluate_v1_sampling_pair.subprocess.Popen") as process:
            with self.assertRaisesRegex(ValueError, "Incomplete full40"):
                evaluate(args)
        process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
