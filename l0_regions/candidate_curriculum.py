"""Deterministic native P/U query curriculum; no neural or training operations.

All observed P remain active. Each case's existing U are ordered once by a
seeded geometry/identity hash, then retained cumulatively in groups of sixteen.
The full inventory and its global indices never change. Support and validation
scope are caller-owned and must not be inferred from this query schedule.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

FORMAT = "native_candidate_curriculum_v1"
STATE_FORMAT = "native_candidate_curriculum_state_v1"
CONFIG_KEYS = {"format", "group_size", "required_pair_win", "required_hit_at_1",
               "minimum_mean_margin", "sustained_epochs"}
METRIC_KEYS = {"pair_win_rate", "hit_at_1", "mean_margin", "rank_evaluable_cases"}


def _hash(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf8")
    except (TypeError, ValueError) as error:
        raise ValueError("Literal finite JSON curriculum evidence required") from error
    return hashlib.sha256(encoded).hexdigest()


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Finite literal numeric curriculum field required: " + name)
    return float(value)


def validate_config(cfg):
    if not isinstance(cfg, Mapping) or set(cfg) != CONFIG_KEYS:
        raise ValueError("Exact explicit curriculum configuration fields required")
    result = copy.deepcopy(dict(cfg))
    if result["format"] != FORMAT or type(result["group_size"]) is not int or result["group_size"] != 16:
        raise ValueError("Native curriculum requires its declared format and group_size=16")
    for key in ("required_pair_win", "required_hit_at_1"):
        result[key] = _number(result[key], key)
        if not 0 <= result[key] <= 1:
            raise ValueError("Curriculum rate thresholds must lie in [0,1]")
    result["minimum_mean_margin"] = _number(result["minimum_mean_margin"], "minimum_mean_margin")
    if result["minimum_mean_margin"] < 0:
        raise ValueError("Mean best-P minus best-U must exceed a nonnegative margin threshold")
    if type(result["sustained_epochs"]) is not int or result["sustained_epochs"] < 1:
        raise ValueError("Explicit positive sustained_epochs required")
    return result


@dataclass
class ActiveContext:
    """Active loss denominators with FULL rows and original global indices."""
    rows: tuple
    order: list
    uses: Counter
    counts: Counter
    pairs: int
    steps: int
    audit: dict


class Curriculum:
    def __init__(self, rows, cfg, seed=42, debug=False):
        if type(seed) is not int or seed < 0 or type(debug) is not bool:
            raise ValueError("Explicit nonnegative integer seed and boolean DEBUG required")
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError("Complete ordered native observation rows required")
        self.config = validate_config(cfg)
        self.seed, self.debug = seed, debug
        self.rows = tuple(copy.deepcopy(rows))
        self.by_case = {}
        ids = set()
        for index, row in enumerate(self.rows):
            if (not isinstance(row, Mapping) or not isinstance(row.get("id"), str)
                    or not row["id"] or row["id"] in ids
                    or not isinstance(row.get("case_id"), str) or not row["case_id"]
                    or type(row.get("target")) is not int or row["target"] not in (0, 1)):
                raise ValueError("Unique native row IDs, case IDs and literal observed P/unobserved U required")
            center = row.get("center")
            if (not isinstance(center, (list, tuple)) or len(center) != 3
                    or any(type(value) is not int or value < 0 for value in center)
                    or not isinstance(row.get("patient_group"), str) or not row["patient_group"]
                    or not isinstance(row.get("donor_case_id"), str) or not row["donor_case_id"]
                    or not isinstance(row.get("donor_group"), str) or not row["donor_group"]
                    or row["donor_group"] == row["patient_group"]
                    or type(row.get("donor_component")) is not int or row["donor_component"] < 1):
                raise ValueError("Exact native geometry and independent donor provenance required")
            ids.add(row["id"])
            self.by_case.setdefault(row["case_id"], []).append(index)
        self.case_ids = tuple(sorted(self.by_case))
        self.positive_indices, self.unobserved_order = {}, {}
        for case in self.case_ids:
            indices = self.by_case[case]
            if len({(self.rows[i]["patient_group"], self.rows[i]["donor_case_id"],
                     self.rows[i]["donor_group"], self.rows[i]["donor_component"]) for i in indices}) != 1:
                raise ValueError("All observations of one case must retain the same recipient and donor")
            positives = [i for i in indices if self.rows[i]["target"] == 1]
            unobserved = [i for i in indices if self.rows[i]["target"] == 0]
            if not unobserved or (not self.debug and len(unobserved) != 128):
                raise ValueError("Production requires exactly 128U per case; actual smaller counts are DEBUG only")
            if self.debug and len(unobserved) > 128:
                raise ValueError("DEBUG preserves an actual native U population of at most 128")
            if len({tuple(self.rows[i]["center"]) for i in unobserved}) != len(unobserved):
                raise ValueError("Duplicate unobserved geometry within a native case")
            self.positive_indices[case] = tuple(positives)
            self.unobserved_order[case] = tuple(sorted(unobserved, key=self._geometry_key))
        self.rank_evaluable_cases = sum(bool(value) for value in self.positive_indices.values())
        self.binding = dict(rows_sha256=_hash(self.rows), config_sha256=_hash(self.config),
            group_order_sha256=_hash({case: [self.rows[i]["id"] for i in self.unobserved_order[case]]
                                     for case in self.case_ids}), seed=self.seed, debug=self.debug)
        self.stage = 0
        self.streak = 0
        self.last_observed_epoch = 0
        self.final_stage_gate_passed = False
        self.history = []

    def _geometry_key(self, index):
        row = self.rows[index]
        # No bounds/edge count, score, difficulty or target determines the U order.
        return (_hash(dict(seed=self.seed, case_id=row["case_id"], id=row["id"],
            center=row["center"], donor_case_id=row["donor_case_id"],
            donor_component=row["donor_component"])), row["id"])

    @property
    def active_U(self):
        return (self.stage + 1) * self.config["group_size"]

    @property
    def all_candidates_active(self):
        return all(len(indices) <= self.active_U for indices in self.unobserved_order.values())

    @property
    def curriculum_complete(self):
        # Completing a DEBUG protocol can never prove full production training.
        return not self.debug and self.active_U == 128 and self.final_stage_gate_passed

    def active_indices(self):
        active = {i for case in self.case_ids for i in
                  (*self.positive_indices[case], *self.unobserved_order[case][:self.active_U])}
        return [i for i in range(len(self.rows)) if i in active]

    def _schedule(self, batch, epoch=None):
        if type(batch) is not int or batch < 2:
            raise ValueError("Explicit physical observation batch >=2 required")
        if epoch is not None and (type(epoch) is not int or epoch < 0):
            raise ValueError("Explicit nonnegative integer epoch required")
        cases = list(self.case_ids)
        rng = None if epoch is None else np.random.default_rng(self.seed + epoch)
        if rng is not None:
            rng.shuffle(cases)
        result = []
        for case in cases:
            positives = list(self.positive_indices[case])
            unobserved = list(self.unobserved_order[case][:self.active_U])
            tiles = []
            if positives:
                width = min(len(positives), batch // 2)
                for start in range(0, len(positives), width):
                    p = positives[start:start+width]
                    remaining = batch - len(p)
                    tiles.extend(p + unobserved[offset:offset+remaining]
                                 for offset in range(0, len(unobserved), remaining))
            else:
                tiles = [unobserved[offset:offset+batch] for offset in range(0, len(unobserved), batch)]
            if rng is not None:
                rng.shuffle(tiles)
            result.extend(tiles)
        return result

    def groups(self, batch, epoch):
        """Same-case live tiles; all selected cases remain contiguous episodes."""
        return iter(self._schedule(batch, epoch))

    def context(self, batch):
        order = self._schedule(batch)
        active = self.active_indices()
        uses = Counter(i for tile in order for i in tile)
        if set(uses) != set(active):
            raise RuntimeError("Native curriculum active observation coverage changed")
        counts = Counter(self.rows[i]["target"] for i in active)
        pairs = sum(len(self.positive_indices[case]) * min(self.active_U, len(self.unobserved_order[case]))
                    for case in self.case_ids)
        if pairs == 0:
            raise ValueError("A P/U training objective requires observed P; zero-P cases remain CE-only")
        audit = dict(format=FORMAT, debug=self.debug, stage=self.stage, stage_U_per_case=self.active_U,
            full_rows=len(self.rows), active_rows=len(active), deferred_rows=len(self.rows)-len(active),
            all_observed_P=counts[1], active_unobserved_U=counts[0],
            full_unobserved_U=sum(len(value) for value in self.unobserved_order.values()),
            active_U_per_case={case: min(self.active_U, len(self.unobserved_order[case])) for case in self.case_ids},
            full_cases=len(self.case_ids), active_cases=len(self.case_ids),
            rank_evaluable_cases=self.rank_evaluable_cases,
            zero_positive_cases=len(self.case_ids)-self.rank_evaluable_cases,
            active_global_indices=active, query_presentations=sum(uses.values()),
            unique_observations=len(active), physical_batch=batch, actual_batch_sizes=[len(tile) for tile in order],
            optimization_steps=len(order), ranking_pairs=pairs,
            all_P_retained=True, all_active_P_U_pairs_once=True,
            cumulative_U_groups=True, group_order_sha256=self.binding["group_order_sha256"],
            loss_normalization="active global pair mean and active globally balanced observation mean",
            support_scope="caller-owned full support; curriculum changes query schedule only",
            all_candidates_active=self.all_candidates_active, curriculum_complete=self.curriculum_complete)
        return ActiveContext(self.rows, order, uses, counts, pairs, len(order), audit)

    def _metrics(self, metrics):
        if not isinstance(metrics, Mapping) or set(metrics) != METRIC_KEYS:
            raise ValueError("Explicit active-training pair-win, Hit@1, mean-margin and rank-case count required")
        result = copy.deepcopy(dict(metrics))
        count = result["rank_evaluable_cases"]
        if type(count) is not int or count != self.rank_evaluable_cases:
            raise ValueError("Curriculum gate must cover all cases with observed P; zero-P excluded only from gate")
        if count == 0:
            if any(result[key] is not None for key in METRIC_KEYS - {"rank_evaluable_cases"}):
                raise ValueError("Absent ranking evidence must be None, never fabricated zero metrics")
        else:
            for key in ("pair_win_rate", "hit_at_1", "mean_margin"):
                result[key] = _number(result[key], key)
            if any(not 0 <= result[key] <= 1 for key in ("pair_win_rate", "hit_at_1")):
                raise ValueError("Curriculum gate rates must lie in [0,1]")
        return result

    def observe(self, metrics, epoch):
        """Gate a completed epoch; promotion takes effect in the next epoch only."""
        if type(epoch) is not int or epoch != self.last_observed_epoch + 1:
            raise ValueError("Observe each completed training epoch exactly once, consecutively from epoch1")
        metrics = self._metrics(metrics)
        previous_stage, previous_U, previous_streak = self.stage, self.active_U, self.streak
        passed = bool(metrics["rank_evaluable_cases"] and
            metrics["pair_win_rate"] >= self.config["required_pair_win"] and
            metrics["hit_at_1"] >= self.config["required_hit_at_1"] and
            metrics["mean_margin"] > self.config["minimum_mean_margin"])
        self.streak = self.streak + 1 if passed else 0
        promoted = self.streak >= self.config["sustained_epochs"] and self.active_U < 128
        if promoted:
            self.stage += 1
            self.streak = 0
        elif self.active_U == 128 and self.streak >= self.config["sustained_epochs"]:
            self.final_stage_gate_passed = True
        self.last_observed_epoch = epoch
        receipt = dict(format=FORMAT, epoch=epoch, metrics=metrics, gate_passed=passed,
            completed_stage=previous_stage, completed_epoch_active_U=previous_U,
            previous_streak=previous_streak, next_stage=self.stage,
            next_epoch_active_U=self.active_U, next_epoch=epoch+1, promoted=bool(promoted),
            streak=self.streak, final_stage_gate_passed=self.final_stage_gate_passed,
            all_candidates_active=self.all_candidates_active, curriculum_complete=self.curriculum_complete,
            debug=self.debug, gate_scope="all P and cumulative active U on training cases only",
            zero_P_ranking_gate_excluded=True,
            required_validation_and_BEST_scope="full original P+128U; verified separately by caller")
        self.history.append(copy.deepcopy(receipt))
        return receipt

    def state_dict(self):
        return dict(format=STATE_FORMAT, binding=copy.deepcopy(self.binding), config=copy.deepcopy(self.config),
            stage=self.stage, active_U=self.active_U, streak=self.streak,
            last_observed_epoch=self.last_observed_epoch, final_stage_gate_passed=self.final_stage_gate_passed,
            all_candidates_active=self.all_candidates_active, curriculum_complete=self.curriculum_complete,
            active_global_indices=self.active_indices(), history=copy.deepcopy(self.history))

    def load_state_dict(self, state):
        """Replay bound gate evidence before accepting any cursor or active set."""
        if (not isinstance(state, Mapping) or state.get("format") != STATE_FORMAT
                or state.get("binding") != self.binding or state.get("config") != self.config
                or not isinstance(state.get("history"), list)):
            raise ValueError("Curriculum state must bind exact rows/config/seed/DEBUG/group identity")
        replay = Curriculum(self.rows, self.config, self.seed, self.debug)
        for receipt in state["history"]:
            if not isinstance(receipt, Mapping) or "metrics" not in receipt or "epoch" not in receipt:
                raise ValueError("Complete curriculum epoch/gate receipts required")
            expected = replay.observe(receipt["metrics"], receipt["epoch"])
            if _hash(expected) != _hash(receipt):
                raise ValueError("Curriculum history differs from its actual declared gate decisions")
        if _hash(replay.state_dict()) != _hash(dict(state)):
            raise ValueError("Curriculum stage/cursor/active indices differ from replayed bound history")
        self.stage, self.streak = replay.stage, replay.streak
        self.last_observed_epoch = replay.last_observed_epoch
        self.final_stage_gate_passed = replay.final_stage_gate_passed
        self.history = copy.deepcopy(replay.history)
