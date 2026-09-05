"""Actual nnU-Net difficulty feedback, separate from frozen compatibility scores.

Only pre-update training observations are labels. A sampler is frozen for an
entire epoch; absent/stale labels remain explicitly unknown and are explored.
Configuration values are unvalidated research hypotheses, not efficacy claims.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np

CONFIG_FORMAT = "onlinecp_segmentation_feedback_v1"
STATE_FORMAT = "onlinecp_actual_feedback_state_v1"
SNAPSHOT_FORMAT = "onlinecp_actual_feedback_snapshot_v1"
MEASUREMENT_DEFINITION = "onlinecp_surviving_lesion_feedback_v1"
PREDICTION_FORMAT = "hiercp_feedback_prediction_v1"


class FeedbackError(ValueError):
    pass


def _exact(value, keys, name):
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise FeedbackError(f"{name} requires exactly {sorted(keys)}")


def _int(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise FeedbackError(f"Invalid integer {name}: {value!r}")
    return value


def _float(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeedbackError(f"{name} must be a finite number")
    value = float(value)
    if (not math.isfinite(value) or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise FeedbackError(f"Invalid finite {name}: {value!r}")
    return value


def _name(value, name):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise FeedbackError(f"Invalid {name}")
    return value


def canonical_sha256(value):
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FeedbackError(f"Feedback state must be finite JSON: {exc}") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise FeedbackError(f"{name} must be a SHA256 hex digest")


def validate_feedback_config(config):
    _exact(config, {"format", "experiment_id", "num_epochs", "candidate_count", "cp_probability",
                    "quality_gate", "difficulty", "metrics", "mixture", "predictions", "stages",
                    "design_notes"}, "Feedback configuration")
    if config["format"] != CONFIG_FORMAT:
        raise FeedbackError("Unknown feedback configuration format")
    _name(config["experiment_id"], "experiment_id")
    _name(config["design_notes"], "design_notes")
    if (_int(config["num_epochs"], "num_epochs") != 250
            or _int(config["candidate_count"], "candidate_count") != 128
            or _float(config["cp_probability"], "cp_probability") != .5):
        raise FeedbackError("Feedback retains 250 epochs, 128 candidates and CP probability 0.5")
    quality = config["quality_gate"]
    _exact(quality, {"score_floor", "max_score_drop", "minimum_choices"}, "Quality gate")
    floor = None if quality["score_floor"] is None else _float(quality["score_floor"], "score_floor")
    gap = _float(quality["max_score_drop"], "max_score_drop", 0)
    if gap == 0:
        raise FeedbackError("max_score_drop must be positive")
    minimum = _int(quality["minimum_choices"], "minimum_choices", 2, 128)
    difficulty = config["difficulty"]
    _exact(difficulty, {"statistic", "ema_alpha", "minimum_observations", "stale_after_epochs",
                        "measurement_definition"}, "Difficulty estimator")
    if difficulty["statistic"] not in {"mean", "ema"} or difficulty["measurement_definition"] != MEASUREMENT_DEFINITION:
        raise FeedbackError("Unknown actual-error statistic or measurement definition")
    alpha = _float(difficulty["ema_alpha"], "ema_alpha", 0, 1)
    if alpha == 0:
        raise FeedbackError("EMA alpha must be positive")
    observations = _int(difficulty["minimum_observations"], "minimum_observations", 1)
    stale = _int(difficulty["stale_after_epochs"], "stale_after_epochs", 1, 250)
    _exact(config["metrics"], {"foreground_ce", "boundary_error", "adjacent_fp"}, "Metric weights")
    metrics = {key: _float(value, key, 0) for key, value in config["metrics"].items()}
    if any(value <= 0 for value in metrics.values()) or not math.isfinite(sum(metrics.values())):
        raise FeedbackError("Every normalized actual-error component needs positive finite weight")
    mixture = config["mixture"]
    _exact(mixture, {"exploration_weight", "easy_retention_weight", "easy_quantile"}, "Sampling mixture")
    exploration = _float(mixture["exploration_weight"], "exploration_weight", 0, 1)
    easy = _float(mixture["easy_retention_weight"], "easy_retention_weight", 0, 1)
    quantile = _float(mixture["easy_quantile"], "easy_quantile", 0, 1)
    if exploration <= 0 or easy <= 0 or exploration + easy >= 1 or not 0 < quantile < 1:
        raise FeedbackError("Exploration, easy retention and curriculum must all retain nonzero mass")
    predictions = config["predictions"]
    _exact(predictions, {"mode", "maximum_age_epochs"}, "Prediction policy")
    if predictions["mode"] not in {"disabled", "optional"}:
        raise FeedbackError("Predictions must be explicitly disabled or optional")
    maximum_age = _int(predictions["maximum_age_epochs"], "maximum_age_epochs", 1, 250)
    if not isinstance(config["stages"], list) or len(config["stages"]) < 3:
        raise FeedbackError("Explicit easy, medium and hard stages are required")
    stages, previous_end, previous_low, previous_high = [], 0, 0., 0.
    for stage in config["stages"]:
        _exact(stage, {"name", "start_epoch", "end_epoch", "target_quantile"}, "Stage")
        start = _int(stage["start_epoch"], "start_epoch", 0, 249)
        end = _int(stage["end_epoch"], "end_epoch", 1, 250)
        band = stage["target_quantile"]
        if not isinstance(band, list) or len(band) != 2:
            raise FeedbackError("target_quantile requires [lower, upper]")
        low, high = [_float(value, "target_quantile", 0, 1) for value in band]
        if start != previous_end or end <= start or not low < high or low < previous_low or high < previous_high:
            raise FeedbackError("Stages must cover all epochs and progress monotonically through error quantiles")
        stages.append(dict(name=_name(stage["name"], "stage name"), start_epoch=start,
                           end_epoch=end, target_quantile=[low, high]))
        previous_end, previous_low, previous_high = end, low, high
    if previous_end != 250 or stages[0]["target_quantile"][0] != 0 or previous_high != 1 or previous_low <= 0:
        raise FeedbackError("Difficulty curriculum must span the full horizon from easy to hard")
    return dict(format=CONFIG_FORMAT, experiment_id=config["experiment_id"], num_epochs=250,
                candidate_count=128, cp_probability=.5,
                quality_gate=dict(score_floor=floor, max_score_drop=gap, minimum_choices=minimum),
                difficulty=dict(statistic=difficulty["statistic"], ema_alpha=alpha,
                                minimum_observations=observations, stale_after_epochs=stale,
                                measurement_definition=MEASUREMENT_DEFINITION),
                metrics=metrics, mixture=dict(exploration_weight=exploration,
                    easy_retention_weight=easy, easy_quantile=quantile),
                predictions=dict(mode=predictions["mode"], maximum_age_epochs=maximum_age),
                stages=stages, design_notes=config["design_notes"])


def config_sha256(config):
    return canonical_sha256(validate_feedback_config(config))


def feedback_config_sha256(config):
    """Explicit public name shared by publisher, launcher and trainer."""
    return config_sha256(config)


def stage_for_epoch(config, epoch):
    _int(epoch, "epoch", 0, 249)
    for index, stage in enumerate(config["stages"]):
        if stage["start_epoch"] <= epoch < stage["end_epoch"]:
            return index, stage
    raise FeedbackError("Epoch is outside the explicit feedback curriculum")


def quality_eligible_indices(scores, config):
    """Fixed compatibility filter only; never interprets rank as difficulty."""
    config = validate_feedback_config(config)
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (128,) or not np.all(np.isfinite(values)):
        raise FeedbackError("The full finite 128-candidate quality vector is required")
    gate = config["quality_gate"]
    eligible = values >= values.max() - gate["max_score_drop"]
    if gate["score_floor"] is not None:
        eligible &= values >= gate["score_floor"]
    indices = np.flatnonzero(eligible)
    if len(indices) < gate["minimum_choices"]:
        raise FeedbackError("Insufficient quality-eligible candidates; quality is not relaxed to manufacture difficulty")
    return indices


def _quantile_weights(values, lower, upper):
    """Fractional rank mass treats every member of a tied group identically."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    right = np.cumsum(counts, dtype=np.float64) / len(values)
    left = np.concatenate(([0.], right[:-1]))
    overlap = np.maximum(0., np.minimum(right, upper) - np.maximum(left, lower))
    weights = (overlap / counts)[inverse]
    # Sum the canonically sorted difficulty groups, not candidate-index order.
    # Reordering candidates must not change even normalization roundoff.
    mass = math.fsum(overlap.tolist())
    if not math.isfinite(mass) or mass <= 0:
        raise FeedbackError("No finite empirical mass for the requested difficulty quantile")
    return weights / mass


class FeedbackState:
    def __init__(self, config, entry_case_ids, candidate_count=128, *, identity=None):
        self.config = validate_feedback_config(config)
        if type(candidate_count) is not int or candidate_count != 128:
            raise FeedbackError("Feedback table must retain all 128 candidates")
        self.candidate_count = candidate_count
        if not isinstance(entry_case_ids, Mapping) or not entry_case_ids:
            raise FeedbackError("An explicit source-entry to training-case map is required")
        self.entry_case_ids = {_name(key, "entry ID"): _name(value, "case ID")
                               for key, value in entry_case_ids.items()}
        self.identity = {} if identity is None else copy.deepcopy(identity)
        if not isinstance(self.identity, dict):
            raise FeedbackError("External feedback identity must be a JSON mapping")
        self.config_sha256 = config_sha256(self.config)
        self.identity_sha256 = canonical_sha256(self._identity_payload())
        self.train_case_ids = self.identity.get("train_case_ids", sorted(set(self.entry_case_ids.values())))
        if (not isinstance(self.train_case_ids, list) or not self.train_case_ids
                or any(not isinstance(value, str) for value in self.train_case_ids)
                or len(set(self.train_case_ids)) != len(self.train_case_ids)
                or not set(self.entry_case_ids.values()).issubset(self.train_case_ids)):
            raise FeedbackError("Entry cases must belong to the complete verified training cohort")
        if set(self.train_case_ids) & set(self.identity.get("validation_case_ids", [])):
            raise FeedbackError("Feedback training and validation cohorts overlap")
        self._table = {entry: dict(count=np.zeros(128, dtype=np.int64),
                                  mean=np.full(128, np.nan), ema=np.full(128, np.nan),
                                  last_epoch=np.full(128, -1, dtype=np.int64))
                       for entry in self.entry_case_ids}
        self._active_snapshot = None
        self._frozen_only = False

    def _identity_payload(self):
        return dict(config_sha256=self.config_sha256, candidate_count=self.candidate_count,
                    entry_case_ids=self.entry_case_ids, external_identity=self.identity)

    def _check_identity(self):
        if config_sha256(self.config) != self.config_sha256 or canonical_sha256(self._identity_payload()) != self.identity_sha256:
            raise FeedbackError("Feedback configuration or experiment identity changed")

    def _encode_table(self):
        result = {}
        for entry, row in self._table.items():
            result[entry] = {"count": row["count"].tolist(),
                            "mean": [None if count == 0 else float(value) for count, value in zip(row["count"], row["mean"])],
                            "ema": [None if count == 0 else float(value) for count, value in zip(row["count"], row["ema"])],
                            "last_epoch": [None if count == 0 else int(value) for count, value in zip(row["count"], row["last_epoch"])]}
        return result

    def _decode_table(self, table, *, last_allowed_epoch):
        _exact(table, self.entry_case_ids, "Candidate table entries")
        decoded = {}
        for entry, row in table.items():
            _exact(row, {"count", "mean", "ema", "last_epoch"}, "Candidate statistics")
            if any(not isinstance(values, list) or len(values) != 128 for values in row.values()):
                raise FeedbackError("Every candidate statistic requires exactly 128 values")
            for index, count in enumerate(row["count"]):
                _int(count, "observation count", 0, int(np.iinfo(np.int64).max))
                values = [row[key][index] for key in ("mean", "ema", "last_epoch")]
                if count == 0:
                    if values != [None, None, None]:
                        raise FeedbackError("Unobserved candidates must have null difficulty and epoch, never fabricated zero")
                else:
                    _float(values[0], "mean error", 0, 1)
                    _float(values[1], "EMA error", 0, 1)
                    _int(values[2], "observation epoch", 0, last_allowed_epoch)
            decoded[entry] = dict(count=np.asarray(row["count"], dtype=np.int64),
                                 mean=np.asarray(row["mean"], dtype=np.float64),
                                 ema=np.asarray(row["ema"], dtype=np.float64),
                                 last_epoch=np.asarray([-1 if value is None else value for value in row["last_epoch"]], dtype=np.int64))
        return decoded

    def update_many(self, records, epoch):
        self._check_identity()
        self._require_snapshot(epoch)
        if self._frozen_only:
            raise FeedbackError("A worker snapshot cannot collect or modify observations")
        if not isinstance(records, (list, tuple)):
            raise FeedbackError("Observation records must be an explicit ordered sequence")
        grouped = {}
        required = {"entry_id", "case_id", "candidate_index", "error", "phase", "timing"}
        for record in records:
            if not isinstance(record, Mapping) or set(record) not in (required, required | {"epoch"}):
                raise FeedbackError("Malformed actual-error observation")
            if record["phase"] != "train" or record["timing"] != "pre_update":
                raise FeedbackError("Only actual pre-update training measurements may update difficulty")
            if "epoch" in record and _int(record["epoch"], "record epoch", 0, 249) != epoch:
                raise FeedbackError("Observation belongs to another epoch")
            entry = _name(record["entry_id"], "observation entry ID")
            if entry not in self.entry_case_ids or record["case_id"] != self.entry_case_ids[entry]:
                raise FeedbackError("Observation source entry/case is outside the verified training map")
            index = _int(record["candidate_index"], "candidate_index", 0, 127)
            error = _float(record["error"], "normalized actual error", 0, 1)
            grouped.setdefault((entry, index), []).append(error)
        # Validate the complete batch before mutating any table entry. EMA uses
        # caller-supplied training-batch order, not completion order of workers.
        alpha = self.config["difficulty"]["ema_alpha"]
        if any(int(self._table[entry]["count"][index]) + len(errors) > np.iinfo(np.int64).max
               for (entry, index), errors in grouped.items()):
            raise FeedbackError("Observation count overflows exact table representation")
        for (entry, index), errors in grouped.items():
            row = self._table[entry]
            old_count, count = int(row["count"][index]), len(errors)
            mean = math.fsum(errors) / count
            old_mean = row["mean"][index]
            row["mean"][index] = mean if old_count == 0 else (old_mean * old_count + mean * count) / (old_count + count)
            ema = errors[0] if old_count == 0 else float(row["ema"][index])
            for error in errors[1:] if old_count == 0 else errors:
                ema = (1 - alpha) * ema + alpha * error
            row["ema"][index], row["count"][index], row["last_epoch"][index] = ema, old_count + count, epoch
        return {"epoch": epoch, "observations": len(records), "candidates": len(grouped)}

    def _prediction_inputs(self, values, provenance, epoch):
        if values is None and provenance is None:
            return None, None
        if self.config["predictions"]["mode"] != "optional" or values is None or provenance is None:
            raise FeedbackError("Predictions need explicit enabled policy, values and independent provenance")
        if not isinstance(values, Mapping) or not values or not set(values).issubset(self.entry_case_ids):
            raise FeedbackError("Predicted entries must belong to the verified source map")
        _exact(provenance, {"format", "gnn_state_sha256", "training_observations_sha256", "train_case_ids",
                            "trained_through_epoch", "prediction_epoch", "measurement_definition", "nnunet_progress"}, "GNN prediction provenance")
        if provenance["format"] != PREDICTION_FORMAT or provenance["measurement_definition"] != MEASUREMENT_DEFINITION:
            raise FeedbackError("Predictions do not target the actual nnU-Net measurement definition")
        _digest(provenance["gnn_state_sha256"], "GNN state")
        _digest(provenance["training_observations_sha256"], "Actual training observations")
        cases = provenance["train_case_ids"]
        if not isinstance(cases, list) or len(cases) != len(set(cases)) or set(cases) != set(self.train_case_ids):
            raise FeedbackError("Predicted difficulties were fitted to another training cohort")
        trained = _int(provenance["trained_through_epoch"], "GNN trained-through epoch", 0, 249)
        if _int(provenance["prediction_epoch"], "prediction epoch", 0, 249) != epoch or trained >= epoch:
            raise FeedbackError("Predictions must precede the current epoch, never use same/future-epoch feedback")
        progress = provenance["nnunet_progress"]
        _exact(progress, {"completed_epoch", "optimizer_steps", "network_sha256"}, "nnU-Net prediction progress")
        if not trained <= _int(progress["completed_epoch"], "completed epoch", 0, 249) < epoch:
            raise FeedbackError("Prediction network progress does not precede this epoch")
        _int(progress["optimizer_steps"], "optimizer_steps", 1)
        _digest(progress["network_sha256"], "nnU-Net network")
        result = {}
        for entry, prediction in values.items():
            if not isinstance(prediction, (list, tuple, np.ndarray)) or len(prediction) != 128:
                raise FeedbackError("A predicted source requires all 128 finite candidate values")
            result[entry] = [_float(float(value) if isinstance(value, np.floating) else value,
                                   "predicted normalized difficulty", 0, 1) for value in prediction]
        return result, copy.deepcopy(provenance)

    def snapshot(self, epoch, *, predicted_difficulties=None, prediction_provenance=None):
        self._check_identity()
        stage_for_epoch(self.config, epoch)
        if self._active_snapshot is not None and self._active_snapshot["epoch"] == epoch:
            if predicted_difficulties is not None or prediction_provenance is not None:
                values, provenance = self._prediction_inputs(predicted_difficulties, prediction_provenance, epoch)
                if values != self._active_snapshot["predicted_difficulties"] or provenance != self._active_snapshot["prediction_provenance"]:
                    raise FeedbackError("An epoch-frozen prediction snapshot cannot be replaced")
            return copy.deepcopy(self._active_snapshot)
        expected = 0 if self._active_snapshot is None else self._active_snapshot["epoch"] + 1
        if self._frozen_only or epoch != expected:
            raise FeedbackError("Snapshots must advance one complete epoch at a time from epoch zero")
        values, provenance = self._prediction_inputs(predicted_difficulties, prediction_provenance, epoch)
        payload = dict(format=SNAPSHOT_FORMAT, config=copy.deepcopy(self.config), config_sha256=self.config_sha256,
                       entry_case_ids=copy.deepcopy(self.entry_case_ids), identity=copy.deepcopy(self.identity),
                       identity_sha256=self.identity_sha256, epoch=epoch, table=self._encode_table(),
                       predicted_difficulties=values, prediction_provenance=provenance)
        payload["snapshot_sha256"] = canonical_sha256(payload)
        self._active_snapshot = payload
        return copy.deepcopy(payload)

    def _require_snapshot(self, epoch):
        stage_for_epoch(self.config, epoch)
        if self._active_snapshot is None or self._active_snapshot["epoch"] != epoch:
            raise FeedbackError("Create the epoch-frozen feedback snapshot before sampling or updating")

    def _distribution(self, entry_id, quality_scores, epoch, basic_control):
        self._check_identity()
        self._require_snapshot(epoch)
        if type(basic_control) is not bool:
            raise FeedbackError("basic_control must be explicitly Boolean")
        if entry_id not in self.entry_case_ids:
            raise FeedbackError("Unknown source entry")
        scores = np.asarray(quality_scores, dtype=np.float64)
        if scores.shape != (128,) or not np.all(np.isfinite(scores)):
            raise FeedbackError("The full finite 128-candidate quality vector is required")
        if basic_control:
            return np.full(128, 1 / 128), ["basic_uniform"] * 128, [None] * 128, 1.
        eligible = np.zeros(128, dtype=bool)
        eligible[quality_eligible_indices(scores, self.config)] = True
        row = self._active_snapshot["table"][entry_id]
        spec = self.config["difficulty"]
        sources, difficulties = ["quality_excluded"] * 128, [None] * 128
        predicted = self._active_snapshot["predicted_difficulties"]
        provenance = self._active_snapshot["prediction_provenance"]
        prediction_fresh = provenance is not None and epoch - provenance["trained_through_epoch"] <= self.config["predictions"]["maximum_age_epochs"]
        for index in np.flatnonzero(eligible):
            count, last = row["count"][index], row["last_epoch"][index]
            fresh = count >= spec["minimum_observations"] and epoch - last <= spec["stale_after_epochs"]
            if fresh:
                sources[index], difficulties[index] = "measured", row[spec["statistic"]][index]
            elif prediction_fresh and entry_id in predicted:
                sources[index], difficulties[index] = "predicted", predicted[entry_id][index]
            else:
                sources[index] = ("exploration_stale_prediction" if predicted is not None and entry_id in predicted and not prediction_fresh else
                                  "exploration_unobserved" if count == 0 else
                                  "exploration_insufficient" if count < spec["minimum_observations"] else
                                  "exploration_stale")
        known = np.asarray([value is not None for value in difficulties], dtype=bool)
        probability = eligible.astype(np.float64) / np.count_nonzero(eligible)
        if not np.any(known):
            return probability, sources, difficulties, 1.
        mixture = self.config["mixture"]
        probability *= mixture["exploration_weight"]
        values = np.asarray([difficulties[index] for index in np.flatnonzero(known)])
        _, stage = stage_for_epoch(self.config, epoch)
        probability[known] += mixture["easy_retention_weight"] * _quantile_weights(values, 0, mixture["easy_quantile"])
        probability[known] += (1 - mixture["exploration_weight"] - mixture["easy_retention_weight"]) * _quantile_weights(values, *stage["target_quantile"])
        probability /= math.fsum(np.sort(probability).tolist())
        return probability, sources, difficulties, mixture["exploration_weight"]

    def probabilities(self, entry_id, quality_scores, epoch, basic_control=False):
        return self._distribution(entry_id, quality_scores, epoch, basic_control)[0]

    def select(self, entry_id, quality_scores, epoch, u, basic_control=False):
        draw = _float(u, "shared candidate draw", 0, 1)
        if draw == 1:
            raise FeedbackError("Shared candidate draw must be in [0,1)")
        probabilities = self.probabilities(entry_id, quality_scores, epoch, basic_control)
        support = np.flatnonzero(probabilities > 0)
        cumulative = np.cumsum(probabilities[support])
        cumulative[-1] = 1.
        if np.any(np.diff(np.concatenate(([0.], cumulative))) <= 0):
            raise FeedbackError("Sampling mass collapses finite-precision support; increase explicit retention/exploration weights")
        return int(support[np.searchsorted(cumulative, draw, side="right")])

    def selection_info(self, entry_id, candidate_index, quality_scores, epoch, basic_control=False):
        index = _int(candidate_index, "candidate_index", 0, 127)
        probabilities, sources, difficulties, exploration = self._distribution(entry_id, quality_scores, epoch, basic_control)
        if probabilities[index] <= 0:
            raise FeedbackError("Candidate is excluded by the fixed quality gate")
        stage, details = stage_for_epoch(self.config, epoch)
        return dict(candidate_index=index, probability=float(probabilities[index]),
                    estimate_source=sources[index], difficulty=difficulties[index],
                    observation_count=self._active_snapshot["table"][entry_id]["count"][index],
                    last_observed_epoch=self._active_snapshot["table"][entry_id]["last_epoch"][index],
                    stage=stage, stage_name=details["name"], effective_exploration_weight=exploration,
                    snapshot_sha256=self._active_snapshot["snapshot_sha256"])

    @classmethod
    def from_snapshot(cls, payload):
        _exact(payload, {"format", "config", "config_sha256", "entry_case_ids", "identity", "identity_sha256",
                         "epoch", "table", "predicted_difficulties", "prediction_provenance", "snapshot_sha256"}, "Frozen feedback snapshot")
        if payload["format"] != SNAPSHOT_FORMAT or payload["snapshot_sha256"] != canonical_sha256({key: value for key, value in payload.items() if key != "snapshot_sha256"}):
            raise FeedbackError("Frozen feedback snapshot format or checksum changed")
        state = cls(payload["config"], payload["entry_case_ids"], identity=payload["identity"])
        if payload["config_sha256"] != state.config_sha256 or payload["identity_sha256"] != state.identity_sha256:
            raise FeedbackError("Frozen feedback identity changed")
        epoch = _int(payload["epoch"], "snapshot epoch", 0, 249)
        state._table = state._decode_table(payload["table"], last_allowed_epoch=epoch - 1)
        state._prediction_inputs(payload["predicted_difficulties"], payload["prediction_provenance"], epoch)
        state._active_snapshot, state._frozen_only = copy.deepcopy(payload), True
        return state

    def state_dict(self):
        self._check_identity()
        if self._frozen_only:
            raise FeedbackError("Serialize worker samplers using their frozen snapshot, not mutable state")
        payload = dict(format=STATE_FORMAT, config=copy.deepcopy(self.config), config_sha256=self.config_sha256,
                       entry_case_ids=copy.deepcopy(self.entry_case_ids), identity=copy.deepcopy(self.identity),
                       identity_sha256=self.identity_sha256, table=self._encode_table(),
                       active_snapshot=copy.deepcopy(self._active_snapshot))
        payload["state_sha256"] = canonical_sha256(payload)
        return payload

    def load_state_dict(self, payload):
        self._check_identity()
        _exact(payload, {"format", "config", "config_sha256", "entry_case_ids", "identity", "identity_sha256",
                         "table", "active_snapshot", "state_sha256"}, "Feedback resume state")
        if payload["format"] != STATE_FORMAT or payload["state_sha256"] != canonical_sha256({key: value for key, value in payload.items() if key != "state_sha256"}):
            raise FeedbackError("Feedback resume format or checksum changed")
        for key in ("config", "config_sha256", "entry_case_ids", "identity", "identity_sha256"):
            if payload[key] != getattr(self, key):
                raise FeedbackError(f"Feedback resume experiment identity changed: {key}")
        snapshot = payload["active_snapshot"]
        active = None if snapshot is None else self.from_snapshot(snapshot)
        if active is not None and active.identity_sha256 != self.identity_sha256:
            raise FeedbackError("Resume snapshot belongs to another experiment")
        last_epoch = -1 if active is None else snapshot["epoch"]
        table = self._decode_table(payload["table"], last_allowed_epoch=last_epoch)
        if active is not None:
            for entry, row in table.items():
                previous = active._table[entry]
                if np.any(row["count"] < previous["count"]):
                    raise FeedbackError("Mutable observation counts precede the frozen snapshot")
                same = row["count"] == previous["count"]
                if (any(not np.array_equal(row[key][same], previous[key][same], equal_nan=True)
                        for key in ("mean", "ema", "last_epoch"))
                        or np.any(row["last_epoch"][~same] != last_epoch)):
                    raise FeedbackError("Mutable statistics contradict their frozen snapshot or update epoch")
        self._table, self._active_snapshot, self._frozen_only = table, copy.deepcopy(snapshot), False
