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
