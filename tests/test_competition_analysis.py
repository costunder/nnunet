"""UNIT synthetic-score fixtures, not CT/model quality evidence."""
import copy
import hashlib
import itertools
import math
import unittest

from hiercp_v1x.competition_analysis import (
    DEFAULT_U_SIZES, SOURCE_FORMAT, TIE_POLICY,
    analyze_competition, hypergeometric_expectation,
)


def unit_report(*, production=False):
    cases = []
    for index in range(21 if production else 3):
        name = f"case_{index}"
        # All U above P, perfect P, and zero-P are present in every fixture.
        p_count = (2, 1, 0)[index % 3]
        scores = (1.0, 3.0, 0.0)[index % 3]
        rows = [dict(record_id=f"{name}:P:{j}", candidate_key=hashlib.sha256(
            f"{name}:P:{j}".encode()).hexdigest(), score=scores-j/2, observed=1)
            for j in range(p_count)]
        rows += [dict(record_id=f"{name}:U:{j}", candidate_key=hashlib.sha256(
            f"{name}:U:{j}".encode()).hexdigest(), score=2.0, observed=0)
            for j in range(128)]
        ordered = sorted(rows, key=lambda row: (-row["score"], row["candidate_key"]))
        ranks = [rank for rank, row in enumerate(ordered, 1) if row["observed"]]
        case = dict(case_id=name, case_scores=rows, scored_record_ids=[r["record_id"] for r in rows],
                    ordered_record_ids=[r["record_id"] for r in ordered], records=len(rows),
                    observed_P=p_count, unobserved_U=128, eligible_observed=p_count,
                    rank_evaluable=bool(p_count))
        if ranks:
            case.update(observed_ranks=ranks, first_observed_rank=ranks[0], reciprocal_rank=1/ranks[0])
        cases.append(case)
    valid = [case for case in cases if case["rank_evaluable"]]
    positives = sum(case["observed_P"] for case in cases)
    return dict(format=SOURCE_FORMAT, task="native_observed_P_vs_unobserved_U",
        GT_is_donor_compatibility=False, original_eight_candidate_metrics=False,
        full_128_U_per_case=True, tie_policy=TIE_POLICY, debug=not production,
        production_full_inner_val=production, UNIT_synthetic_scores_only=True,
        cases=cases, cohort=dict(case_ids=[case["case_id"] for case in cases],
            debug=not production, production_full_inner_val=production,
            records=sum(case["records"] for case in cases), observed_P=positives,
            unobserved_U=128*len(cases)),
        denominators=dict(cases=len(cases), rank_evaluable_cases=len(valid),
            zero_P_cases=len(cases)-len(valid), observed_P=positives,
            unobserved_U=128*len(cases), P_U_pairs=128*positives,
            case_hit_at_1=len(valid), observed_micro_recall=positives),
        metrics=dict(case_first_P_mrr=math.fsum(case["reciprocal_rank"] for case in valid)/len(valid),
            case_hit_at_1=sum(case["first_observed_rank"] == 1 for case in valid)/len(valid)))


def rerank(report):
    """Update only a synthetic fixture after changing its score/tie arrangement."""
    valid = []
    for case in report["cases"]:
        rows = sorted(case["case_scores"], key=lambda row: (-row["score"], row["candidate_key"]))
        case["ordered_record_ids"] = [row["record_id"] for row in rows]
        ranks = [rank for rank, row in enumerate(rows, 1) if row["observed"]]
        if ranks:
            case.update(observed_ranks=ranks, first_observed_rank=ranks[0], reciprocal_rank=1/ranks[0])
            valid.append(case)
    report["metrics"].update(
        case_first_P_mrr=math.fsum(case["reciprocal_rank"] for case in valid)/len(valid),
        case_hit_at_1=sum(case["first_observed_rank"] == 1 for case in valid)/len(valid))


class ExactHypergeometricUnit(unittest.TestCase):
    def test_matches_exhaustive_uniform_subsets(self):
        for population in range(8):
            for ahead in range(population+1):
                for draws in range(population+1):
                    subsets = list(itertools.combinations(range(population), draws))
                    counts = [sum(index < ahead for index in subset) for subset in subsets]
                    expected_mrr = math.fsum(1/(1+count) for count in counts)/len(counts)
                    expected_hit = sum(count == 0 for count in counts)/len(counts)
                    result = hypergeometric_expectation(population, ahead, draws)
                    with self.subTest(population=population, ahead=ahead, draws=draws):
                        self.assertAlmostEqual(result["expected_case_first_P_mrr"], expected_mrr, places=14)
                        self.assertAlmostEqual(result["expected_case_hit_at_1"], expected_hit, places=14)

    def test_full_population_is_original_rank_and_empty_draw_is_one(self):
        for ahead in range(129):
            full = hypergeometric_expectation(128, ahead, 128)
            self.assertEqual(full["expected_case_first_P_mrr"], 1/(ahead+1))
            self.assertEqual(full["expected_case_hit_at_1"], float(ahead == 0))
            self.assertEqual(hypergeometric_expectation(128, ahead, 0),
                dict(expected_case_first_P_mrr=1.0, expected_case_hit_at_1=1.0))

    def test_rejects_nonliteral_or_impossible_counts(self):
        for arguments in ((True, 0, 0), (8, False, 1), (8, 1, 1.0), (-1, 0, 0),
                          (8, 9, 1), (8, 1, 9), (8, -1, 1), (8, 1, -1)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                hypergeometric_expectation(*arguments)


class ScoreFixedCompetitionUnit(unittest.TestCase):
    def test_exact_curves_retain_all_scores_GT_and_zero_P_cases(self):
        report = unit_report()
        original = copy.deepcopy(report)
        result = analyze_competition(report, debug=True)
        self.assertEqual(report, original)
        self.assertEqual(result["diagnostic_U_sizes"], list(DEFAULT_U_SIZES))
        self.assertEqual(result["denominators"], report["denominators"])
        self.assertEqual(len(result["cases"]), 3)
        for saved, derived in zip(report["cases"], result["cases"]):
            self.assertEqual(derived["case_scores"], saved["case_scores"])
            self.assertEqual(derived["ordered_record_ids"], saved["ordered_record_ids"])
        for curve in result["cases"][0]["curves"]:
            self.assertEqual(curve["expected_case_first_P_mrr"], 1/(curve["unobserved_U"]+1))
            self.assertEqual(curve["hypothetical_candidate_count"], 2+curve["unobserved_U"])
        for curve in result["cases"][1]["curves"]:
            self.assertEqual(curve["expected_case_first_P_mrr"], 1.0)
            self.assertEqual(curve["expected_case_hit_at_1"], 1.0)
        for curve in result["cases"][2]["curves"]:
            self.assertIsNone(curve["expected_case_first_P_mrr"])
            self.assertIsNone(curve["expected_case_hit_at_1"])
        self.assertTrue(result["endpoint_128_matches_source"])
        self.assertFalse(result["neural_forward_executed"])
        self.assertFalse(result["training_executed"])
        self.assertFalse(result["random_sampling_executed"])
        self.assertFalse(result["original_eight_candidate_replay"])
        self.assertFalse(result["quality_verified"])
        result["cases"][0]["case_scores"][0]["observed"] = 0
        self.assertEqual(report, original)

    def test_tied_U_before_P_counts_as_competitor_by_saved_key(self):
        report = unit_report()
        case = report["cases"][0]
        for row in case["case_scores"]:
            row["score"] = 0.0
        rerank(report)
        result = analyze_competition(report, debug=True)["cases"][0]
        self.assertEqual(result["U_ahead_of_first_P"], case["first_observed_rank"]-1)
        self.assertGreater(result["U_ahead_of_first_P"], 0)
        self.assertEqual(result["U_strictly_higher_score_than_first_P"], 0)
        self.assertEqual(result["U_tied_before_first_P"], result["U_ahead_of_first_P"])
        swapped = copy.deepcopy(report)
        order = swapped["cases"][0]["ordered_record_ids"]
        order[0], order[1] = order[1], order[0]
        with self.assertRaisesRegex(ValueError, "tie policy"):
            analyze_competition(swapped, debug=True)

    def test_same_valid_report_row_permutation_preserves_macro_curve(self):
        report = unit_report()
        expected = analyze_competition(report, debug=True)["curve"]
        for case in report["cases"]:
            case["case_scores"].reverse()
            case["scored_record_ids"].reverse()
        report["cases"].reverse()
        self.assertEqual(analyze_competition(report, debug=True)["curve"], expected)

    def test_debug_requires_explicit_flag_and_production_preserves_all21(self):
        with self.assertRaises(ValueError):
            analyze_competition(unit_report())
        report = unit_report(production=True)
        result = analyze_competition(report)
        self.assertFalse(result["debug"])
        self.assertTrue(result["production_full_inner_val"])
        self.assertEqual(result["denominators"]["cases"], 21)
        self.assertEqual(result["denominators"]["unobserved_U"], 2688)
        with self.assertRaises(ValueError):
            analyze_competition(report, debug=True)
        report["cases"].pop()
        with self.assertRaises(ValueError):
            analyze_competition(report)

    def test_rejects_invalid_GT_score_and_keys(self):
        for field, value in (("observed", True), ("observed", 2), ("observed", 1.0),
                             ("score", True), ("score", float("nan")),
                             ("score", float("inf")), ("score", 10**1000),
                             ("candidate_key", "invented")):
            report = unit_report()
            report["cases"][0]["case_scores"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                analyze_competition(report, debug=True)

    def test_rejects_missing_duplicate_extra_or_inconsistent_coverage(self):
        changes = (
            lambda r: r["cases"][0]["case_scores"].pop(),
            lambda r: r["cases"][0]["case_scores"].append(copy.deepcopy(r["cases"][0]["case_scores"][0])),
            lambda r: r["cases"][0]["ordered_record_ids"].pop(),
            lambda r: r["cases"][0]["scored_record_ids"].reverse(),
            lambda r: r["cases"][0].update(unobserved_U=127),
            lambda r: r["cases"][0].update(observed_P=3),
            lambda r: r["cases"][0].update(records=1),
            lambda r: r["cohort"]["case_ids"].pop(),
            lambda r: r["cohort"].update(records=1),
            lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])),
        )
        for index, change in enumerate(changes):
            report = unit_report()
            change(report)
            with self.subTest(change=index), self.assertRaises(ValueError):
                analyze_competition(report, debug=True)

    def test_rejects_stale_reported_rank_metric_and_denominator(self):
        changes = (
            lambda r: r["cases"][0].update(first_observed_rank=1),
            lambda r: r["cases"][0].update(first_observed_rank=True),
            lambda r: r["cases"][0].update(observed_ranks=[129]),
            lambda r: r["cases"][0].update(reciprocal_rank=1.0),
            lambda r: r["cases"][0].update(rank_evaluable=1),
            lambda r: r["cases"][2].update(first_observed_rank=0),
            lambda r: r["cases"][2].update(observed_ranks=[1]),
            lambda r: r["metrics"].update(case_first_P_mrr=1.0),
            lambda r: r["metrics"].update(case_hit_at_1=1.0),
            lambda r: r["denominators"].update(case_hit_at_1=3),
            lambda r: r["denominators"].update(zero_P_cases=0),
            lambda r: r["denominators"].update(P_U_pairs=0),
        )
        for index, change in enumerate(changes):
            report = unit_report()
            change(report)
            with self.subTest(change=index), self.assertRaises(ValueError):
                analyze_competition(report, debug=True)

    def test_rejects_misrepresented_task_or_tie_contract(self):
        for field, value in (("GT_is_donor_compatibility", True),
                             ("original_eight_candidate_metrics", True),
                             ("full_128_U_per_case", False), ("tie_policy", "GT_first"),
                             ("task", "CP_compatibility"), ("debug", False)):
            report = unit_report()
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze_competition(report, debug=True)


if __name__ == "__main__":
    unittest.main()
