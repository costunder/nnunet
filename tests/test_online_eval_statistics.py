"""Debug-only numerical unit tests; no medical images, training, or evaluation run.

The production functions are compiled directly from their AST so these focused
NumPy tests do not require the separate SciPy/NIfTI imaging dependencies. This
does not test full evaluator imports, lesion matching, or real-data evaluation.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import unittest

import numpy as np


def load_statistics_functions() -> dict:
    source = Path(__file__).resolve().parents[1] / "tools" / "online_eval_v2.py"
    selected_names = {
        "EvaluationError",
        "_safe_ratio",
        "metric_from_counts",
        "_validated_count_resampling_inputs",
        "_count_resampling_diagnostics",
        "cluster_bootstrap_count_difference",
        "cluster_permutation_count_difference",
        "cluster_permutation_count_inference",
        "cluster_bootstrap_lesion_mean",
    }
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected_names
    ]
    found = {node.name for node in selected}
    if found != selected_names:
        raise AssertionError(f"Production statistical functions missing: {selected_names - found}")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    namespace = {"np": np, "math": math}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace


FUNCTIONS = load_statistics_functions()
bootstrap = FUNCTIONS["cluster_bootstrap_count_difference"]
permutation = FUNCTIONS["cluster_permutation_count_inference"]
compatibility_p = FUNCTIONS["cluster_permutation_count_difference"]
lesion_bootstrap = FUNCTIONS["cluster_bootstrap_lesion_mean"]
EvaluationError = FUNCTIONS["EvaluationError"]


class DebugOnlineCountStatisticsTests(unittest.TestCase):
    def test_equal_precision_does_not_count_undefined_swaps_as_non_extreme(self):
        basic = np.array([[1, 0, 0], [0, 0, 1]])
        hier = np.array([[0, 0, 1], [1, 0, 0]])
        result = permutation(basic, hier, "precision", seed=17, iterations=5000)
        self.assertIsNone(result["permutation_p"])
        self.assertIsNone(compatibility_p(basic, hier, "precision", 17, 5000))
        diagnostics = result["permutation_diagnostics"]
        self.assertEqual(diagnostics["status"], "unavailable")
        self.assertEqual(diagnostics["reason"], "undefined_metric_in_resamples")
        self.assertEqual(diagnostics["completed_resamples"], 5000)
        self.assertEqual(diagnostics["valid_resamples"] + diagnostics["invalid_resamples"], 5000)
        self.assertGreater(diagnostics["valid_resamples"], 0)
        self.assertGreater(diagnostics["invalid_resamples"], 0)
        json.dumps(result, allow_nan=False)

    def test_finite_permutations_preserve_patient_swap_and_plus_one_p(self):
        basic = np.array([[4, 2, 1], [2, 1, 3]])
        hier = np.array([[5, 1, 0], [3, 1, 2]])
        seed, iterations = 71, 4096
        result = permutation(basic, hier, "precision", seed, iterations)
        # Independent finite-only reference for the original patient-swap test.
        swaps = np.random.default_rng(seed).integers(
            0, 2, size=(iterations, len(basic)), dtype=np.int8
        ).astype(bool)
        first = np.where(swaps[:, :, None], hier[None], basic[None]).sum(axis=1)
        second = np.where(swaps[:, :, None], basic[None], hier[None]).sum(axis=1)
        null = second[:, 0] / second[:, :2].sum(axis=1) - first[:, 0] / first[:, :2].sum(axis=1)
        observed = hier[:, 0].sum() / hier[:, :2].sum() - basic[:, 0].sum() / basic[:, :2].sum()
        expected = (np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15) + 1) / (iterations + 1)
        self.assertEqual(result["permutation_p"], expected)
        self.assertEqual(result["permutation_diagnostics"]["status"], "available")
        self.assertEqual(result["permutation_diagnostics"]["invalid_resamples"], 0)
        self.assertEqual(compatibility_p(basic, hier, "precision", seed, iterations), expected)

    def test_seeded_resampling_is_deterministic_including_chunk_tail(self):
        basic = np.array([[4, 2, 1], [2, 1, 3]])
        hier = np.array([[5, 1, 0], [3, 1, 2]])
        for metric in ("recall", "precision", "f1", "fp_per_case"):
            with self.subTest(metric=metric):
                self.assertEqual(permutation(basic, hier, metric, 99, 5003), permutation(basic, hier, metric, 99, 5003))
                self.assertEqual(bootstrap(basic, hier, metric, 99, 1000), bootstrap(basic, hier, metric, 99, 1000))

    def test_bootstrap_does_not_silently_condition_on_defined_precision(self):
        counts = np.array([[1, 0, 0], [0, 0, 1]])
        result = bootstrap(counts, counts, "precision", seed=4, iterations=1000)
        self.assertEqual(result["difference"], 0.0)
        self.assertIsNone(result["ci_low"])
        self.assertIsNone(result["ci_high"])
        diagnostics = result["bootstrap_diagnostics"]
        self.assertEqual(diagnostics["reason"], "undefined_metric_in_resamples")
        self.assertGreater(diagnostics["invalid_resamples"], 0)
        self.assertEqual(diagnostics["valid_resamples"] + diagnostics["invalid_resamples"], 1000)
        # The paired-swap distribution remains defined for this same cohort.
        self.assertEqual(permutation(counts, counts, "precision", 4, 1000)["permutation_p"], 1.0)

    def test_finite_bootstrap_preserves_original_percentile_interval(self):
        basic = np.array([[4, 2, 1], [2, 1, 3]])
        hier = np.array([[5, 1, 0], [3, 1, 2]])
        seed, iterations = 5, 1000
        result = bootstrap(basic, hier, "precision", seed, iterations)
        indices = np.random.default_rng(seed).integers(0, 2, size=(iterations, 2))
        first, second = basic[indices].sum(axis=1), hier[indices].sum(axis=1)
        differences = second[:, 0] / second[:, :2].sum(axis=1) - first[:, 0] / first[:, :2].sum(axis=1)
        self.assertAlmostEqual(result["difference"], 8 / 10 - 6 / 9)
        self.assertEqual(result["ci_low"], float(np.quantile(differences, 0.025)))
        self.assertEqual(result["ci_high"], float(np.quantile(differences, 0.975)))
        self.assertEqual(result["bootstrap_diagnostics"]["status"], "available")

    def test_observed_zero_denominator_is_unavailable_not_zero(self):
        basic = np.array([[0, 0, 1], [0, 0, 2]])
        hier = np.array([[1, 0, 0], [1, 0, 1]])
        boot_result = bootstrap(basic, hier, "precision", 9, 1000)
        perm_result = permutation(basic, hier, "precision", 9, 1000)
        self.assertIsNone(boot_result["difference"])
        self.assertIsNone(boot_result["ci_low"])
        self.assertIsNone(perm_result["permutation_p"])
        self.assertEqual(boot_result["bootstrap_diagnostics"]["reason"], "observed_metric_undefined")
        self.assertEqual(perm_result["permutation_diagnostics"]["reason"], "observed_metric_undefined")
        json.dumps([boot_result, perm_result], allow_nan=False)

    def test_empty_cohort_and_invalid_inputs_are_explicit(self):
        empty = np.empty((0, 3))
        self.assertEqual(bootstrap(empty, empty, "recall", 1, 1000)["bootstrap_diagnostics"]["reason"], "no_patients")
        self.assertEqual(permutation(empty, empty, "recall", 1, 1000)["permutation_diagnostics"]["reason"], "no_patients")
        for invalid in (np.ones((2, 2)), np.array([[1, -1, 0]]), np.array([[np.nan, 0, 1]])):
            for function in (bootstrap, permutation):
                with self.subTest(function=function.__name__, invalid=invalid.tolist()):
                    with self.assertRaises(EvaluationError):
                        function(invalid, invalid, "precision", 1, 1000)
        for iterations in (0, -1, 2.5):
            for function in (bootstrap, permutation):
                with self.assertRaises(EvaluationError):
                    function(np.ones((2, 3)), np.ones((2, 3)), "recall", 1, iterations)

    def test_lesion_bootstrap_reports_undefined_lesion_free_resamples(self):
        result = lesion_bootstrap(np.array([0.5, 0.0]), np.array([0.6, 0.0]), np.array([1, 0]), 4, 1000)
        self.assertAlmostEqual(result["difference"], 0.1)
        self.assertIsNone(result["ci_low"])
        self.assertIsNone(result["ci_high"])
        self.assertGreater(result["bootstrap_diagnostics"]["invalid_resamples"], 0)
        self.assertEqual(result["bootstrap_diagnostics"]["reason"], "undefined_metric_in_resamples")


if __name__ == "__main__":
    unittest.main()
