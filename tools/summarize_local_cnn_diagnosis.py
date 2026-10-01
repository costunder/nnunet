"""Read an existing diagnostic JSON without importing torch or executing a model."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import textwrap
from typing import Any


_MISSING = object()


def _get(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _num(value: Any) -> str:
    if value is _MISSING:
        return "unavailable"
    if value is None:
        return "undefined"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "invalid"
    if not math.isfinite(value):
        return "nonfinite"
    return f"{value:.3g}"


def _metric(value: Any) -> str:
    if (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value)):
        return f"{value:.7g}"
    return _num(value)


def _text(value: Any) -> str:
    if value is _MISSING:
        return "unavailable"
    if value is None:
        return "undefined"
    return str(value).replace("\r", " ").replace("\n", " ")


def _flag(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unavailable" if value is _MISSING else "invalid"


def _variance(stage: Any) -> str:
    status = _get(stage, "status")
    if status == "NOT_RUN":
        return "NOT_RUN"
    return _num(_get(stage, "normalized_centered_energy"))


def _ratio(numerator: Any, denominator: Any) -> str:
    if numerator is _MISSING or denominator is _MISSING:
        return "unavailable"
    if numerator is None or denominator is None or denominator == 0:
        return "undefined"
    if (isinstance(numerator, bool) or isinstance(denominator, bool)
            or not isinstance(numerator, (int, float))
            or not isinstance(denominator, (int, float))):
        return "invalid"
    return _num(numerator / denominator)


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _case_lines(case: dict, reasons: dict[str, set[str]]) -> list[str]:
    score = _get(case, "full_support", "score")
    trace = _get(case, "full_support", "trace")
    metrics = _get(case, "full_support", "metrics")
    label = f"{_text(_get(case, 'split'))}/{_text(_get(case, 'case_id'))}"
    lines = [
        f"CASE {label} N={_text(_get(case, 'records'))} P={_text(_get(case, 'positives'))} "
        f"std={_metric(_get(score, 'score_std'))} gap={_metric(_get(score, 'mean_positive_minus_unobserved'))} "
        f"win={_metric(_get(score, 'pair_win_rate'))} loss={_metric(_get(score, 'mean_pairwise_loss'))} "
        f"MRR={_metric(_get(metrics, 'ranking_mrr'))}"
    ]
    map_means = []
    for scale in ("scale1", "scale2", "scale3"):
        values = []
        for measured in _as_list(_get(case, "l0_substages", "batches")):
            values.extend(_as_list(_get(measured, "map_spatial_variance", scale, "by_unique_crop")))
        if values and all(not isinstance(value, bool) and isinstance(value, (int, float))
                          and math.isfinite(value) for value in values):
            map_means.append(_num(math.fsum(values) / len(values)))
        else:
            map_means.append("unavailable" if not values else "invalid")
    lines.append("  CNN map spatial-var batch-crop mean s1/s2/s3=" + "/".join(map_means)
                 + " (not candidate variance)")
    stages = _get(case, "l0_substages", "stages")
    if not isinstance(stages, dict):
        lines.append("  L0 readout/anchor substage: unavailable (no deep trace in report)")
    else:
        selected = ["global_mean_concat_recipient", "project_recipient", "fusion_input", "fusion_output"]
        readout = "/".join(_variance(_get(stages, key)) for key in selected)
        roi = "/".join(_variance(_get(stages, f"scale{i}_anchor_roi_recipient")) for i in (1, 2, 3))
        lines.append(f"  L0 variance mean/project/fusion_in/out={readout}; anchor(scale1/2/3)={roi}")
        for key, stage in stages.items():
            if _get(stage, "status") == "NOT_RUN":
                reason = _text(_get(stage, "reason"))
                reasons.setdefault(reason, set()).add(label)

    probe = _get(case, "l1_substages_and_message_sweep")
    sweep = _as_list(_get(probe, "message_scale_sweep"))
    baseline = next((row for row in sweep if _get(row, "message_scale") == 1), None)
    if baseline is None:
        lines.append("  L1 baseline stage trace: unavailable (scale=1 not recorded)")
        fine_stages = {row.get("stage"): row for row in _as_list(_get(trace, "stages"))
                       if isinstance(row, dict)}
        if fine_stages:
            lines.append("  Candidate variance L0/L1_1/L1_2=" + "/".join(
                _variance(_get(fine_stages, key)) for key in ("L0", "L1_1", "L1_2")))
    else:
        for layer in _as_list(_get(baseline, "layers")):
            stats = _get(layer, "stages")
            keys = ("input", "projected_message", "residual_add", "residual_norm", "ff_output", "second_add", "final_norm")
            values = "/".join(_variance(_get(stats, key)) for key in keys)
            ratio = _ratio(_get(stats, "final_norm", "normalized_centered_energy"),
                           _get(stats, "input", "normalized_centered_energy"))
            lines.append(f"  L1_{_text(_get(layer, 'layer'))} var in/msg/add/LN/FF/add2/final={values}; final/input={ratio}")
        lines.append("  L1 scale=1 parity=" + _flag(_get(probe, "production_scale_parity_passed")))

    if not sweep:
        lines.append("  Message-scale sweep: unavailable")
    else:
        compact = []
        for row in sweep:
            stats = _get(row, "score")
            values = "/".join(_metric(_get(stats, key)) for key in (
                "score_std", "mean_positive_minus_unobserved", "pair_win_rate", "mean_pairwise_loss"))
            compact.append(f"s={_num(_get(row, 'message_scale'))} {values}")
        for start in range(0, len(compact), 2):
            lines.append("  Sweep std/gap/win/loss: " + " | ".join(compact[start:start + 2]))
    ff_sweep = _as_list(_get(probe, "query_ff_scale_sweep", "reports"))
    for start in range(0, len(ff_sweep), 2):
        compact = []
        for row in ff_sweep[start:start+2]:
            stats = _get(row, "score")
            values = "/".join(_metric(_get(stats, key)) for key in (
                "score_std", "mean_positive_minus_unobserved", "pair_win_rate", "mean_pairwise_loss"))
            compact.append(f"ff={_num(_get(row, 'ff_scale'))} {values}")
        lines.append("  L1_2 FF-only std/gap/win/loss: " + " | ".join(compact))
    for layer in _as_list(_get(probe, "attention_sensitivity", "layers")):
        values = ["/".join(_metric(_get(head, key)) for key in (
            "candidate_weight_variance_mean", "mean_pairwise_cosine", "mean_pairwise_jensen_shannon", "entropy_mean"))
            for head in _as_list(_get(layer, "per_head"))]
        lines.append(f"  Attention L1_{_text(_get(layer, 'layer'))} head var/cos/JS/entropy: " + " | ".join(values))
    for row in _as_list(_get(case, "fusion_identity_bypass", "branches")):
        stats = _get(row, "score")
        values = "/".join(_metric(_get(stats, key)) for key in (
            "score_std", "mean_positive_minus_unobserved", "pair_win_rate", "mean_pairwise_loss"))
        lines.append(f"  Fusion {_text(_get(row, 'branch'))} lambda={_text(_get(row, 'fusion_scale'))} "
                     f"std/gap/win/loss={values}; support+query re-encoded")
    lines.extend(_interaction_case_lines(case))
    return lines


def _interaction_case_lines(case: dict) -> list[str]:
    candidate = _get(case, "l1_interaction_candidate")
    if candidate is _MISSING:
        return []
    if not isinstance(candidate, dict):
        return ["  L1 INTERACTION: unavailable (invalid candidate result)"]
    if _get(candidate, "status") == "NOT_RUN":
        return ["  L1 INTERACTION NOT_RUN: " + _text(_get(candidate, "reason"))]
    branches = _as_list(_get(candidate, "branches"))
    matched_flags = [_get(branch, "support_query_equation_matched") for branch in branches]
    matched = (True if matched_flags and all(v is True for v in matched_flags)
               else False if any(v is False for v in matched_flags) else _MISSING)
    lines = ["  L1 INTERACTION fixed-weight; support+query-matched=" + _flag(matched)
             + " beta0-parity=" + _flag(_get(candidate, "scale_zero_production_parity_passed"))
             + " original-weights=" + _flag(_get(candidate, "original_weights_unchanged"))]
    if not branches:
        lines.append("  Interaction branches: unavailable")
    for branch in branches:
        name, scale = _get(branch, "branch"), _get(branch, "interaction_scale")
        beta = "legacy" if name == "legacy_additive" and scale is None else _metric(scale)
        score = _get(branch, "score")
        values = "/".join(_metric(_get(score, key)) for key in (
            "score_std", "pair_win_rate", "mean_pairwise_loss"))
        lines.append(f"  Interaction {_text(name)} beta={beta} std/win/loss={values}")
        layer = next((row for row in _as_list(_get(branch, "layers"))
                      if _get(row, "layer") == 2), None)
        heads = _as_list(_get(layer, "attention", "per_head"))
        if not heads:
            lines.append("    L1_2 attention: unavailable")
            continue
        variance = "/".join(_metric(_get(head, "candidate_weight_variance_mean")) for head in heads)
        js = "/".join(_metric(_get(head, "mean_pairwise_jensen_shannon")) for head in heads)
        cosines = [_get(head, "mean_pairwise_cosine") for head in heads]
        cosine_status = _saved_range(cosines)
        cosine_min = (cosine_status if cosine_status in ("unavailable", "undefined", "invalid", "nonfinite")
                      else _metric(min(cosines)))
        lines.append(f"    L1_2 attention head-var={variance} head-JS={js} min(head-mean cos)={cosine_min}")
    return lines


def _saved_range(values: list) -> str:
    """Never reduce only the present/finite subset of a saved update series."""
    if not values or any(value is _MISSING for value in values):
        return "unavailable"
    if any(value is None for value in values):
        return "undefined"
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        return "invalid"
    if any(not math.isfinite(value) for value in values):
        return "nonfinite"
    return _metric(min(values)) + "/" + _metric(max(values))


def _interaction_evaluation(value: Any, split: str) -> str:
    if _get(value, "status") == "NOT_RUN":
        return "NOT_RUN: " + _text(_get(value, "reason"))
    result = _get(value, split)
    if _get(result, "status") == "NOT_RUN":
        return "NOT_RUN: " + _text(_get(result, "reason"))
    # The explicit evaluation callback can return metrics directly or under
    # 'metrics'; this reader does not infer omitted values from another split.
    nested = _get(result, "metrics")
    metrics = nested if isinstance(nested, dict) else result
    values = []
    for key in ("ranking_mrr", "pair_win_rate", "ranking_pairwise_loss"):
        # The matched prefix evaluator saves aggregate pair win on split root,
        # while ranking_metrics saves MRR/loss inside 'metrics'. Older explicit
        # layouts with a nested win or fully flat metrics remain readable.
        value = _get(result, key) if key == "pair_win_rate" else _get(metrics, key)
        if key == "pair_win_rate" and value is _MISSING:
            value = _get(metrics, key)
        values.append(_metric(value))
    return "/".join(values)


def _interaction_update_lines(report: dict) -> list[str]:
    updates = _get(report, "l1_interaction_updates")
    if updates is _MISSING:
        return []
    if not isinstance(updates, dict):
        return ["L1 INTERACTION UPDATES: unavailable (invalid result)"]
    if _get(updates, "status") == "NOT_RUN":
        return ["L1 INTERACTION UPDATES NOT_RUN: " + _text(_get(updates, "reason"))]
    lines = [f"L1 INTERACTION UPDATES: cloned fresh-AdamW full-objective prefix; steps/branch="
             f"{_text(_get(updates, 'cloned_optimizer_updates_per_branch'))} "
             f"batch={_text(_get(updates, 'physical_batch'))} "
             f"prefix-observations={_text(_get(updates, 'prefix_unique_observations'))}/"
             f"{_text(_get(updates, 'full_cohort_observations'))}",
             "  Optimizer history=" + _text(_get(updates, "optimizer", "history"))
             + "; exact-resume=" + _flag(_get(updates, "exact_resume"))
             + " next-saved-update=" + _flag(_get(updates, "next_saved_update"))]
    branches = _as_list(_get(updates, "branches"))
    if not branches:
        lines.append("  Update branches: unavailable")
    for branch in branches:
        name = _text(_get(branch, "branch"))
        lines.append(f"  Update {name} beta={_metric(_get(branch, 'scale'))} "
                     f"CNN grad min/max={_saved_range([_get(row, 'module_gradient_norms', 'CNN') for row in _as_list(_get(branch, 'updates'))])} "
                     f"CNN parameter-delta={_metric(_get(branch, 'module_parameter_delta_norms', 'CNN'))} "
                     f"prefix-seconds={_metric(_get(branch, 'prefix_seconds'))}")
        for split in ("train", "validation"):
            before = _interaction_evaluation(_get(branch, "before"), split)
            after = _interaction_evaluation(_get(branch, "after"), split)
            lines.append(f"    {name} {split} MRR/win/loss before -> after: {before} -> {after}")
    lines.append("  Production updates=" + _text(_get(updates, "production_optimizer_updates"))
                 + " checkpoints=" + _text(_get(updates, "checkpoints_written"))
                 + " original-preserved=" + _flag(_get(updates, "original_weights_types_methods_modes_preserved"))
                 + " caller-RNG=" + _flag(_get(updates, "caller_rng_restored")))
    lines.append("  Fresh optimizer cloned prefix, not production continuation or final CP performance.")
    return lines


def _shadow_lines(report: dict) -> list[str]:
    shadow = _get(report, "shadow_update")
    request = _get(report, "shadow_update_request")
    if not isinstance(shadow, dict):
        if _get(request, "status") == "NOT_RUN":
            return ["SHADOW NOT_RUN: " + _text(_get(request, "reason"))]
        return ["SHADOW: unavailable (cloned next-update result not recorded)"]
    losses = _get(shadow, "losses")
    loss_text = "/".join(_metric(_get(losses, key)) for key in ("ranking", "observation_ce", "alignment", "full"))
    lines = [f"SHADOW loss rank/CE/align/full={loss_text}; cloned steps={_text(_get(shadow, 'cloned_optimizer_steps'))}"]
    lines.append("  Clip R/C/A/F=" + "/".join(_num(_get(shadow, "branches", key, "clipping_factor"))
        for key in ("ranking", "observation_ce", "alignment", "full")))
    for module in ("CNN", "readout_fusion", "L1", "L2", "global"):
        grads = _get(shadow, "gradients", module)
        cosines = "/".join(_num(_get(grads, key, "cosine")) for key in (
            "observation_ce_vs_ranking", "alignment_vs_ranking", "auxiliary_sum_vs_ranking"))
        deltas = "/".join(_num(_get(shadow, "branches", key, "modules", module, "delta_norm"))
                          for key in ("ranking", "observation_ce", "alignment", "full"))
        full_cos = _num(_get(shadow, "branches", "full", "modules", module,
                            "vs_negative_ranking_gradient", "cosine"))
        rank_cos = _num(_get(shadow, "branches", "ranking", "vs_full_update", module, "cosine"))
        lines.append(f"  {module}: grad cos CE/align/aux={cosines}; Adam d R/C/A/F={deltas}; "
                     f"cos(full,-rank)={full_cos} cos(rank,full)={rank_cos}")
    lines.append("  Shadow verified: Adam-state=" + _flag(_get(shadow, "optimizer_state_reused"))
                 + " gradient-sum=" + _flag(_get(shadow, "decomposition_full_backward_verified"))
                 + " shared-forward=" + _flag(_get(shadow, "same_forward_realization_for_all_branches")))
    history = _get(shadow, "history_isolation", "modules")
    if isinstance(history, dict):
        for module in ("CNN", "readout_fusion", "L1", "L2", "global"):
            row = _get(history, module)
            values = "/".join(_num(_get(row, key)) for key in ("history_delta_norm", "ratio_to_full_delta_norm"))
            lines.append(f"  Adam history {module}: norm/full-ratio={values} "
                         f"cos(full,history)={_num(_get(row, 'vs_full_delta', 'cosine'))} "
                         f"norm(full-history)={_num(_get(row, 'delta_full_minus_history', 'delta_norm'))} "
                         f"norm(rank-history)={_num(_get(row, 'delta_ranking_minus_history', 'delta_norm'))}")
        lines.append("  Adam history uses explicit zero gradients and weight decay; nonlinear contrasts, not additive attribution.")
    return lines


def format_summary(report: dict) -> str:
    """Format saved values only; absent data stays unavailable, null stays undefined."""
    if not isinstance(report, dict):
        raise ValueError("Diagnostic report must be a JSON object")
    lines = [
        "SAVED DIAGNOSTIC SUMMARY (saved values only; full details in JSON)",
        "Run: " + _text(_get(report, "run")),
        f"Snapshot epoch={_text(_get(report, 'epoch'))} step={_text(_get(report, 'step'))} "
        f"phase={_text(_get(report, 'phase'))} margin_mm={_text(_get(report, 'margin_mm'))} "
        f"batch={_text(_get(report, 'physical_batch'))}",
        "Checkpoint content SHA256: " + _text(_get(report, "checkpoint_content_sha256")),
    ]
    cases = _get(report, "cases")
    reasons: dict[str, set[str]] = {}
    if not isinstance(cases, list) or not cases:
        lines.append("Cases: unavailable (no case results recorded)")
    else:
        for case in cases:
            if not isinstance(case, dict):
                lines.append("CASE: unavailable (invalid case object)")
            else:
                lines.extend(_case_lines(case, reasons))
    lines.extend(_shadow_lines(report))
    audit = _get(report, "alignment_schedule_audit")
    if isinstance(audit, dict):
        dist = _get(audit, "tile_count_distribution")
        lines.append(f"ALIGNMENT audit: groups={_text(_get(audit, 'patient_group_count'))} "
                     f"tiles={_text(_get(audit, 'optimization_steps'))} "
                     f"K_g min/max={_text(_get(dist, 'minimum'))}/{_text(_get(dist, 'maximum'))} "
                     f"current/equal-group min/max={_metric(_get(dist, 'minimum_current_to_equal_ratio'))}/"
                     f"{_metric(_get(dist, 'maximum_current_to_equal_ratio'))}; production loss unchanged")
    encoding = _get(report, "fusion_support_reencoding")
    if isinstance(encoding, dict):
        lines.append(f"FUSION full-support union: records={_text(_get(encoding, 'required_unique_support_records'))} "
                     f"new/reused={_text(_get(encoding, 'newly_encoded_records'))}/"
                     f"{_text(_get(encoding, 'reused_query_trace_records'))} "
                     f"seconds={_metric(_get(encoding, 'elapsed_seconds'))}; CNN outputs shared across lambdas")
    direct = _get(report, "direct_scalar_head")
    if isinstance(direct, dict):
        lines.append(f"DIRECT HEAD: diagnostic updates={_text(_get(direct, 'head_optimizer_updates'))}; "
                     f"CNN updates={_text(_get(direct, 'cnn_updates'))}; fresh head/Adam, frozen recipient features, rank-only")
        for split in ("train", "validation"):
            values = []
            for phase in ("before", "after"):
                stats = _get(direct, phase, split, "metrics")
                values.append("/".join(_metric(_get(stats, key)) for key in (
                    "ranking_mrr", "pair_win_rate", "ranking_pairwise_loss")))
            lines.append(f"  Direct {split} MRR/win/loss before -> after: " + " -> ".join(values))
        lines.append("  Short frozen-feature control; train fit alone is not CP validity or proof of the original failure cause.")
    lines.extend(_interaction_update_lines(report))
    lines.append("Preserved: weights=" + _flag(_get(report, "weights_unchanged"))
                 + " saved-payload=" + _flag(_get(report, "shadow_update", "saved_payload_unchanged"))
                 + " support-plan=" + _flag(_get(report, "shadow_update", "support_plan_unchanged"))
                 + " caller-RNG=" + _flag(_get(report, "shadow_update", "caller_rng_restored")))
    lines.append(f"Scope: diagnostic={_flag(_get(report, 'diagnostic_only'))} "
                 f"production-updates={_text(_get(report, 'optimizer_updates'))} "
                 f"full-evaluation={_flag(_get(report, 'full_evaluation'))}; "
                 f"peak GPU GiB={_num(_get(report, 'peak_cuda_gib'))}")
    for reason, labels in reasons.items():
        lines.append(f"NOT_RUN anchor ({len(labels)} cases): {reason}")
    lines.append("Variance is not accuracy. Sweep uses fixed weights. unavailable=missing; undefined=null/zero denominator.")
    return "\n".join(part for line in lines for part in textwrap.wrap(
        line, width=140, subsequent_indent="    ", replace_whitespace=False,
        drop_whitespace=True, break_long_words=False, break_on_hyphens=False))


def read_summary(path: Path) -> str:
    with path.open("r", encoding="utf8") as handle:
        report = json.load(handle)
    return format_summary(report)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Existing completed diagnostic JSON")
    parser.add_argument("--output", type=Path, help="Optional NEW UTF-8 text file; never overwrite an existing file")
    args = parser.parse_args(argv)
    summary = read_summary(args.report)
    if args.output is not None:
        with args.output.open("x", encoding="utf8", newline="\n") as handle:
            handle.write(summary + "\n")
    print(summary, flush=True)
    if args.output is not None:
        print("SUMMARY SAVED: " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
