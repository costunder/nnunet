"""Frozen cumulative native-U admission using U logits alone.

Mining is performed by the original full-bank joint model outside this module.
These pure functions consume only its 128 U scores in the original native bank
order; neither P scores nor labels/validation metrics enter admission.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import math
from numbers import Real


POLICIES = ("native_prefix", "hard_score_top", "score_stratified_mix")
FORMAT = "v23_frozen_cumulative_U_score_admission_v1"
STAGE_FORMAT = "v23_complete_frozen_U_selection_v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _indices(indices, count=None):
    if (not isinstance(indices, (tuple, list))
            or any(type(i) is not int or not 0 <= i < 128 for i in indices)
            or len(set(indices)) != len(indices) or not 1 <= len(indices) <= 128
            or (count is not None and len(indices) != count)):
        raise ValueError("Distinct original native-U bank indices and exact active count required")
    return tuple(sorted(indices))


def _scores(scores):
    if (not isinstance(scores, (tuple, list)) or len(scores) != 128
            or any(isinstance(s, bool) or not isinstance(s, Real) or not math.isfinite(float(s))
                   for s in scores)):
        raise ValueError("Exactly 128 finite native-bank-ordered U scores required; P scores are excluded")
    return tuple(float(s) for s in scores)


def _quotas(count, capacities):
    """Weighted fair apportionment: seven slots yield 3 high, 2 middle, 2 low."""
    weights = (3, 2, 2)
    result = [0, 0, 0]
    for slot in range(count):
        eligible = [i for i in range(3) if result[i] < capacities[i]]
        if not eligible:
            raise ValueError("Stratified admission capacity cannot cover the requested additions")
        winner = max(eligible, key=lambda i: (weights[i] * (slot + 1) / 7 - result[i], -i))
        result[winner] += 1
    return result


def admit_targets(policy, active_indices, u_scores, add_count=7, *, seed=42,
                  case_id="", epoch=None, model_sha256=None):
    """Freeze one gate's cumulative membership and signed U-only provenance.

    For the stratified policy, descending unseen-U ranks are divided into three
    contiguous, balanced thirds. Admission within each third follows descending
    score, with original bank position breaking ties. The final partial stage
    redistributes the 3:2:2 quota without losing any remaining U.
    """
    if policy not in POLICIES or type(add_count) is not int or not 1 <= add_count <= 128:
        raise ValueError("Declared v2.3 admission policy and positive native-U expansion required")
    if (type(seed) is not int or not isinstance(case_id, str)
            or (epoch is not None and (type(epoch) is not int or epoch < 0))
            or (model_sha256 is not None and (not isinstance(model_sha256, str)
                or len(model_sha256) != 64
                or any(c not in "0123456789abcdef" for c in model_sha256)))):
        raise ValueError("Explicit deterministic admission source provenance required")
    active, scores = _indices(active_indices), _scores(u_scores)
    unseen = [i for i in range(128) if i not in set(active)]
    count = min(add_count, len(unseen))
    ranked = sorted(unseen, key=lambda i: (-scores[i], i))
    strata, quotas = [], []
    if policy == "native_prefix":
        if active != tuple(range(len(active))):
            raise ValueError("Native-prefix experiment must retain its declared original prefix")
        additions = unseen[:count]
    elif policy == "hard_score_top":
        additions = ranked[:count]
    else:
        base, remainder = divmod(len(ranked), 3)
        sizes = [base + (i < remainder) for i in range(3)]
        cursor = 0
        for size in sizes:
            strata.append(ranked[cursor:cursor + size])
            cursor += size
        quotas = _quotas(count, sizes)
        additions = [index for group, quota in zip(strata, quotas) for index in group[:quota]]
    indices = tuple(sorted((*active, *additions)))
    if len(indices) != len(active) + count or not set(active) <= set(indices):
        raise ValueError("Cumulative admission lost retained native U membership")
    receipt = dict(format=FORMAT, policy=policy, case_id=case_id, seed=seed,
        epoch=epoch, model_sha256=model_sha256, previous_indices=list(active),
        added_indices=list(additions), indices=list(indices), active_U=len(indices),
        requested_additions=add_count, actual_additions=count,
        selected_U_scores=[dict(bank_index=i, score=scores[i]) for i in additions],
        full_U_scores_sha256=_digest(list(scores)), U_scores_only=True,
        P_scores_used=False, P_or_validation_labels_used=False,
        validation_metrics_used=False, selection_frozen_until_next_gate=True,
        tie_break="original_native_U_bank_position_ascending",
        bank_order_preserved=True, retained_U_preserved=True,
        score_strata=[list(group) for group in strata], stratum_quotas=quotas)
    receipt["sha256"] = _digest(receipt)
    return dict(indices=list(indices), receipt=receipt)


def validate_admission(receipt):
    """Verify serialized U-only admission membership and signed provenance."""
    if not isinstance(receipt, Mapping):
        raise ValueError("Signed U admission receipt required")
    value = copy.deepcopy(dict(receipt))
    signature = value.pop("sha256", None)
    if signature != _digest(value) or value.get("format") != FORMAT:
        raise ValueError("Frozen U admission receipt changed")
    previous, selected = _indices(value["previous_indices"]), _indices(value["indices"])
    added = value["added_indices"]
    scores = value["selected_U_scores"]
    if (not isinstance(added, list)
            or any(type(i) is not int or not 0 <= i < 128 for i in added)
            or len(set(added)) != len(added) or set(previous) & set(added)
            or set(selected) != set(previous) | set(added)
            or value["actual_additions"] != len(added) or value["active_U"] != len(selected)
            or value["policy"] not in POLICIES
            or not isinstance(scores, list) or len(scores) != len(added)
            or any(not isinstance(row, Mapping) or row.get("bank_index") != index
                   or isinstance(row.get("score"), bool)
                   or not isinstance(row.get("score"), Real)
                   or not math.isfinite(float(row["score"])) for row, index in zip(scores, added))
            or value["U_scores_only"] is not True or value["P_scores_used"] is not False
            or value["P_or_validation_labels_used"] is not False
            or value["validation_metrics_used"] is not False
            or value["selection_frozen_until_next_gate"] is not True):
        raise ValueError("Frozen U admission membership or target-free policy differs")
    return copy.deepcopy(dict(receipt))


def validate_target_selection(population, active_u_count, selection):
    """Return the complete immutable-stage case map (training loss + all val)."""
    cases = tuple(population.partition_cases("inner_train", ranking_only=True)) + tuple(
        population.partition_cases("inner_val"))
    if (type(active_u_count) is not int or not 1 <= active_u_count <= 128
            or not isinstance(selection, Mapping) or set(selection) != set(cases)):
        raise ValueError("Frozen selection must cover every ranking train case and every validation case")
    result = {case: list(_indices(selection[case], active_u_count)) for case in cases}
    for case in cases:
        population.case(case, active_u_count, active_u_indices=result[case])
    return result


def target_selection_manifest(population, active_u_count, selection):
    selected = validate_target_selection(population, active_u_count, selection)
    value = dict(format=STAGE_FORMAT, population_sha256=population.manifest()["sha256"],
                 active_U=active_u_count, selections=selected, all_P_retained=True,
                 retained_native_bank=True, P_scores_or_labels_used=False,
                 prefix_equivalent=all(indices == list(range(active_u_count))
                                       for indices in selected.values()))
    value["sha256"] = _digest(value)
    return value


def patient_epoch_order(case_ids, prior_train_rows, policy, random_order):
    """Order every training patient using only preceding completed TRAIN loss.

    The original seeded permutation supplies tie order. No patient is omitted,
    duplicated, weighted more often, or selected using validation information.
    """
    if (policy not in POLICIES or not isinstance(case_ids, (tuple, list))
            or not case_ids or any(not isinstance(c, str) or not c for c in case_ids)
            or len(set(case_ids)) != len(case_ids)
            or not isinstance(random_order, (tuple, list))
            or len(random_order) != len(case_ids) or set(random_order) != set(case_ids)):
        raise ValueError("Complete unique training patient set and original permutation required")
    losses, groups = {}, []
    if policy == "native_prefix":
        order = list(random_order)
    else:
        if not isinstance(prior_train_rows, (tuple, list)) or len(prior_train_rows) != len(case_ids):
            raise ValueError("Every preceding completed TRAIN patient loss is required")
        for row in prior_train_rows:
            if (not isinstance(row, Mapping) or row.get("case_id") not in set(case_ids)
                    or row["case_id"] in losses
                    or type(row.get("observed_P")) is not int or row["observed_P"] <= 0
                    or row.get("partition", "inner_train") != "inner_train"
                    or not isinstance(row.get("per_P"), Mapping)):
                raise ValueError("Unique P-bearing previous TRAIN cases required; validation is excluded")
            loss = row["per_P"].get("pair_loss")
            if (isinstance(loss, bool) or not isinstance(loss, Real)
                    or not math.isfinite(float(loss)) or loss < 0):
                raise ValueError("Finite nonnegative preceding TRAIN per_P pair_loss required")
            losses[row["case_id"]] = float(loss)
        if set(losses) != set(case_ids):
            raise ValueError("Previous TRAIN loss coverage differs from the full patient set")
        ties = {case: i for i, case in enumerate(random_order)}
        ranked = sorted(case_ids, key=lambda case: (-losses[case], ties[case]))
        if policy == "hard_score_top":
            order = ranked
        else:
            base, remainder = divmod(len(ranked), 3)
            cursor = 0
            for i in range(3):
                size = base + (i < remainder)
                groups.append(ranked[cursor:cursor + size])
                cursor += size
            order = [group[i] for i in range(max(map(len, groups)))
                     for group in groups if i < len(group)]
    if len(order) != len(case_ids) or len(set(order)) != len(case_ids) or set(order) != set(case_ids):
        raise ValueError("Patient ordering changed complete training coverage")
    receipt = dict(format="v23_complete_previous_train_loss_patient_order_v1",
        policy=policy, order=order, random_order=list(random_order), patient_count=len(case_ids),
        previous_train_rows_sha256=None if prior_train_rows is None else _digest(prior_train_rows),
        previous_train_per_P_pair_loss=losses, train_loss_strata=groups,
        source="provided_original_seeded_permutation" if policy == "native_prefix"
               else "preceding_completed_training_epoch", validation_used=False,
        validation_labels_or_metrics_used=False,
        training_P_U_labels_used=policy != "native_prefix",
        train_patient_difficulty=policy != "native_prefix", full_patient_coverage=True,
        omissions=0, duplicates=0, repeated_patients=0,
        tie_break="provided_original_seeded_permutation")
    receipt["sha256"] = _digest(receipt)
    return dict(order=list(order), receipt=receipt)
