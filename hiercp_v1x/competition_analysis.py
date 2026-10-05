"""Exact competitor-count diagnostics conditional on saved full-case scores.

All P scores, all 128 U scores and the original GT-independent total order are
retained. The hypothetical competition draws k U uniformly without replacement;
it never rescores a smaller graph. This is not an original curriculum8 replay,
an independent donor/task contrast, or a new neural evaluation. Only stdlib is
used, and no model, tensor, checkpoint or training code is imported.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
import math


FORMAT = "historical_score_fixed_U_competition_v1"
SOURCE_FORMAT = "hiercp_transition_whole128_evaluation_v1"
TIE_POLICY = "score_desc_geometry_sha256_v1"
DEFAULT_U_SIZES = (7, 15, 31, 63, 128)
_ABS_TOL = 1e-12
_REL_TOL = 1e-12


def _integer(value, label, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Literal integer >= {minimum} required: {label}")
    return value


def _finite(value, label):
    if type(value) not in (int, float):
        raise ValueError(f"Finite numeric value required: {label}")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"Finite numeric value required: {label}") from error
    if not math.isfinite(numeric):
        raise ValueError(f"Finite numeric value required: {label}")
    return numeric


def _same_number(actual, expected, label):
    value = _finite(actual, label)
    if not math.isclose(value, expected, rel_tol=_REL_TOL, abs_tol=_ABS_TOL):
        raise ValueError(f"Saved full-case value disagrees with scores: {label}")


def _ids(value, label):
    if (not isinstance(value, list) or not value
            or any(not isinstance(item, str) or not item for item in value)
            or len(value) != len(set(value))):
        raise ValueError(f"Nonempty unique string identity list required: {label}")
    return value


def hypergeometric_expectation(population, ahead, draws):
    """Return exact-probability expectations for rank 1+H and Hit@1.

    H ~ Hypergeometric(population, ahead, draws). Integer combinations and
    Fraction arithmetic compute the expectation exactly before its final float
    conversion; there is no Monte Carlo sampling or approximation. ``ahead``
    includes ties ordered before the first P by the saved total order.
    """
    population = _integer(population, "population")
    ahead = _integer(ahead, "ahead")
    draws = _integer(draws, "draws")
    if ahead > population or draws > population:
        raise ValueError("Hypergeometric ahead/draws cannot exceed population")
    denominator = math.comb(population, draws)
    lower = max(0, draws - (population - ahead))
    upper = min(ahead, draws)
    reciprocal = Fraction(0)
    probability_mass = 0
    for count in range(lower, upper + 1):
        weight = math.comb(ahead, count) * math.comb(population - ahead, draws - count)
        probability_mass += weight
        reciprocal += Fraction(weight, denominator * (count + 1))
    if probability_mass != denominator:
        raise ArithmeticError("Exact hypergeometric probability mass is incomplete")
    hit = (Fraction(math.comb(population - ahead, draws), denominator)
           if draws <= population - ahead else Fraction(0))
    return dict(expected_case_first_P_mrr=float(reciprocal),
                expected_case_hit_at_1=float(hit))


def _case_analysis(case):
    if not isinstance(case, Mapping) or not isinstance(case.get("case_id"), str) or not case["case_id"]:
        raise ValueError("Each saved case needs a nonempty case_id")
    name = case["case_id"]
    rows = case.get("case_scores")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"All saved scores required: {name}")
    by_id = {}
    keys = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"Saved score row must be a mapping: {name}")
        identity, key = row.get("record_id"), row.get("candidate_key")
        if not isinstance(identity, str) or not identity or identity in by_id:
            raise ValueError(f"Unique nonempty record_id required: {name}")
        if (not isinstance(key, str) or len(key) != 64
                or any(char not in "0123456789abcdef" for char in key) or key in keys):
            raise ValueError(f"Unique original geometry SHA256 candidate_key required: {name}")
        if type(row.get("observed")) is not int or row["observed"] not in (0, 1):
            raise ValueError(f"Literal observed P=1 / unobserved U=0 GT required: {name}")
        _finite(row.get("score"), f"{name}:{identity} score")
        by_id[identity] = row
        keys.add(key)
    scored = _ids(case.get("scored_record_ids"), f"{name} scored_record_ids")
    ordered = _ids(case.get("ordered_record_ids"), f"{name} ordered_record_ids")
    if scored != [row["record_id"] for row in rows] or set(ordered) != set(scored):
        raise ValueError(f"Saved score and ordered identity coverage differ: {name}")
    # The original policy orders by descending score then ascending geometry key.
    # GT is never used to resolve ties. Check the stored order before reusing it.
    expected_order = [row["record_id"] for row in sorted(
        rows, key=lambda row: (-row["score"], row["candidate_key"]))]
    if ordered != expected_order:
        raise ValueError(f"Stored order violates declared GT-independent tie policy: {name}")
    positives = sum(row["observed"] for row in rows)
    negatives = len(rows) - positives
    if negatives != 128:
        raise ValueError(f"Every saved full case must contain exactly 128 U: {name}")
    for field, expected in (("observed_P", positives), ("unobserved_U", negatives),
                            ("eligible_observed", positives), ("records", len(rows))):
        if _integer(case.get(field), f"{name} {field}") != expected:
            raise ValueError(f"Saved case count disagrees with scores: {name} {field}")
    evaluable = positives > 0
    if type(case.get("rank_evaluable")) is not bool or case["rank_evaluable"] is not evaluable:
        raise ValueError(f"Saved rank-evaluable flag disagrees with GT: {name}")
    ranks = [rank for rank, identity in enumerate(ordered, 1) if by_id[identity]["observed"]]
    first = ranks[0] if ranks else None
    best = by_id[ordered[first - 1]] if first is not None else None
    if evaluable:
        if _integer(case.get("first_observed_rank"), f"{name} first_observed_rank", minimum=1) != first:
            raise ValueError(f"Saved first P rank disagrees with total order: {name}")
        if (not isinstance(case.get("observed_ranks"), list)
                or any(type(rank) is not int for rank in case["observed_ranks"])
                or case["observed_ranks"] != ranks):
            raise ValueError(f"Saved observed ranks disagree with total order: {name}")
        _same_number(case.get("reciprocal_rank"), 1 / first, f"{name} reciprocal_rank")
    elif any(case.get(field) is not None for field in ("first_observed_rank", "reciprocal_rank")):
        raise ValueError(f"Zero-P case cannot contain a fabricated P rank: {name}")
    elif case.get("observed_ranks") not in (None, []):
        raise ValueError(f"Zero-P case cannot contain observed ranks: {name}")
    ahead = first - 1 if first is not None else None
    curves = []
    for draws in DEFAULT_U_SIZES:
        value = (hypergeometric_expectation(128, ahead, draws) if evaluable else
                 dict(expected_case_first_P_mrr=None, expected_case_hit_at_1=None))
        curves.append(dict(unobserved_U=draws, observed_P=positives,
                           hypothetical_candidate_count=positives + draws,
                           rank_evaluable=evaluable, **value))
    return dict(case_id=name, observed_P=positives, source_unobserved_U=128,
                source_records=len(rows), rank_evaluable=evaluable,
                first_observed_record_id=best["record_id"] if best else None,
                source_first_observed_rank=first, U_ahead_of_first_P=ahead,
                U_strictly_higher_score_than_first_P=(sum(
                    row["observed"] == 0 and row["score"] > best["score"] for row in rows)
                    if best else None),
                U_tied_before_first_P=(sum(
                    by_id[identity]["score"] == best["score"] for identity in ordered[:ahead])
                    if best else None),
                # Preserve every original numerical score/GT, including zero-P cases.
                case_scores=deepcopy(rows), scored_record_ids=list(scored),
                ordered_record_ids=list(ordered), curves=curves)


def analyze_competition(report, *, debug=False):
    """Validate a complete common report and derive its score-fixed U curve.

    Production requires all 21 cases. DEBUG reports require explicit
    ``debug=True`` and remain labeled DEBUG; they cannot become trained-quality
    or full-cohort evidence through this diagnostic.
    """
    if type(debug) is not bool or not isinstance(report, Mapping):
        raise ValueError("Explicit DEBUG boolean and saved report mapping required")
    if (report.get("format") != SOURCE_FORMAT
            or report.get("task") != "native_observed_P_vs_unobserved_U"
            or report.get("GT_is_donor_compatibility") is not False
            or report.get("original_eight_candidate_metrics") is not False
            or report.get("full_128_U_per_case") is not True
            or report.get("debug") is not debug
            or report.get("production_full_inner_val") is not (not debug)
            or report.get("tie_policy") != TIE_POLICY):
        raise ValueError("Complete native P/U report with original tie/DEBUG contract required")
    saved_cases = report.get("cases")
    if not isinstance(saved_cases, list) or not saved_cases or (not debug and len(saved_cases) != 21):
        raise ValueError("Production requires all 21 saved cases; smaller fixtures need explicit DEBUG")
    cases = [_case_analysis(case) for case in saved_cases]
    names = [case["case_id"] for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate saved case identities")
    identities = [identity for case in cases for identity in case["scored_record_ids"]]
    if len(set(identities)) != len(identities):
        raise ValueError("Saved record identities must be unique across all cases")
    valid = [case for case in cases if case["rank_evaluable"]]
    if not valid:
        raise ValueError("Saved common rank metrics require at least one observed P case")
    positives = sum(case["observed_P"] for case in cases)
    negatives = 128 * len(cases)
    denominators = dict(cases=len(cases), rank_evaluable_cases=len(valid),
                        zero_P_cases=len(cases) - len(valid), observed_P=positives,
                        unobserved_U=negatives, P_U_pairs=128 * positives,
                        case_hit_at_1=len(valid), observed_micro_recall=positives)
    saved_denominators = report.get("denominators")
    if not isinstance(saved_denominators, Mapping):
        raise ValueError("Saved common metric denominators required")
    for field, expected in denominators.items():
        if _integer(saved_denominators.get(field), field) != expected:
            raise ValueError(f"Saved denominator disagrees with full score coverage: {field}")
    cohort = report.get("cohort")
    if (not isinstance(cohort, Mapping) or cohort.get("debug") is not debug
            or cohort.get("production_full_inner_val") is not (not debug)
            or set(_ids(cohort.get("case_ids"), "cohort case_ids")) != set(names)):
        raise ValueError("Saved cohort identities/DEBUG flags disagree with cases")
    for field, expected in (("records", len(identities)), ("observed_P", positives),
                            ("unobserved_U", negatives)):
        if _integer(cohort.get(field), f"cohort {field}") != expected:
            raise ValueError(f"Saved cohort count disagrees with scores: {field}")
    curves = []
    for position, draws in enumerate(DEFAULT_U_SIZES):
        curves.append(dict(unobserved_U=draws, rank_evaluable_cases=len(valid),
            zero_P_cases=len(cases) - len(valid), observed_P_retained=positives,
            expected_case_first_P_mrr=math.fsum(
                case["curves"][position]["expected_case_first_P_mrr"] for case in valid) / len(valid),
            expected_case_hit_at_1=math.fsum(
                case["curves"][position]["expected_case_hit_at_1"] for case in valid) / len(valid)))
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Saved full-case MRR and Hit@1 metrics required")
    endpoint = curves[-1]
    _same_number(metrics.get("case_first_P_mrr"), endpoint["expected_case_first_P_mrr"], "case_first_P_mrr")
    _same_number(metrics.get("case_hit_at_1"), endpoint["expected_case_hit_at_1"], "case_hit_at_1")
    return dict(format=FORMAT, debug=debug, production_full_inner_val=not debug,
                quality_verified=False, GT_is_donor_compatibility=False,
                score_fixed=True, all_original_P_retained=True,
                all_original_128_U_scores_retained=True,
                neural_forward_executed=False, training_executed=False,
                random_sampling_executed=False, original_eight_candidate_replay=False,
                subset_neural_upper_evaluation=False, full_evaluation_replaced=False,
                method="Exact hypergeometric expectation over uniform U subsets without replacement; full-case scores and total order stay fixed",
                interpretation="Conditional competitor-count effect only; donor/GT/task and joint-upper candidate-context effects are not separated",
                tie_policy=TIE_POLICY, diagnostic_U_sizes=list(DEFAULT_U_SIZES),
                source_report=dict(format=report["format"], task=report["task"],
                    debug=report["debug"], scores_sha256=report.get("scores_sha256"),
                    cohort_sha256=cohort.get("cohort_sha256")),
                source_metrics=dict(case_first_P_mrr=metrics["case_first_P_mrr"],
                                    case_hit_at_1=metrics["case_hit_at_1"]),
                denominators=denominators, endpoint_128_matches_source=True,
                endpoint_tolerance=dict(relative=_REL_TOL, absolute=_ABS_TOL),
                curve=curves, cases=cases)
