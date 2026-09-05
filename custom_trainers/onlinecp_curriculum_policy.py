"""Explicit, opt-in OnlineCP selection curriculum; no torch or medical I/O.

Scores are learned ranking scores, not calibrated medical probabilities.
All input candidates must already pass the unchanged anatomical validity gates.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np


CONFIG_FORMAT = "onlinecp_rank_curriculum_v1"
SCHEDULE_FORMAT = "onlinecp_curriculum_schedule_v1"
RESUME_FORMAT = "onlinecp_curriculum_resume_v1"
FNV_OFFSET = 0xCBF29CE484222325


class CurriculumError(ValueError):
    """A missing or incompatible explicit curriculum contract."""


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CurriculumError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise CurriculumError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise CurriculumError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CurriculumError(f"Contract is not finite JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def validate_curriculum_config(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"format", "experiment_id", "num_epochs", "candidate_count",
                "cp_probability", "score_floor", "max_score_drop", "stages"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise CurriculumError(f"Curriculum requires exactly these explicit fields: {sorted(required)}")
    if value["format"] != CONFIG_FORMAT:
        raise CurriculumError(f"Expected curriculum format {CONFIG_FORMAT}")
    name = value["experiment_id"]
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in name)):
        raise CurriculumError("experiment_id must be a safe, nonempty ASCII identifier")
    if _integer(value["num_epochs"], "num_epochs") != 250:
        raise CurriculumError("This new trainer preserves the complete 250-epoch horizon")
    if _integer(value["candidate_count"], "candidate_count") != 128:
        raise CurriculumError("This experiment preserves all 128 anatomical candidates per source")
    probability = _number(value["cp_probability"], "cp_probability", positive=True)
    if probability != 0.5:
        raise CurriculumError("This experiment preserves CP probability 0.5")
    floor = None if value["score_floor"] is None else _number(value["score_floor"], "score_floor")
    gap = _number(value["max_score_drop"], "max_score_drop", positive=True)
    raw_stages = value["stages"]
    if not isinstance(raw_stages, list) or len(raw_stages) < 2:
        raise CurriculumError("At least two explicit curriculum stages are required")
    stages = []
    previous_end, previous_rank, previous_minimum = 0, 0, 0
    keys = {"start_epoch", "end_epoch", "top_rank", "minimum_choices", "temperature"}
    for index, stage in enumerate(raw_stages):
        if not isinstance(stage, Mapping) or set(stage) != keys:
            raise CurriculumError(f"Stage {index} requires exactly {sorted(keys)}")
        start = _integer(stage["start_epoch"], "start_epoch")
        end = _integer(stage["end_epoch"], "end_epoch", 1)
        rank = _integer(stage["top_rank"], "top_rank", 1)
        minimum = _integer(stage["minimum_choices"], "minimum_choices", 1)
        temperature = (None if stage["temperature"] is None else
                       _number(stage["temperature"], "temperature", positive=True))
        if start != previous_end or end <= start or end > 250:
            raise CurriculumError("Stages must cover [0, 250) contiguously without gaps or overlap")
        if not previous_rank <= rank <= 128 or not previous_minimum <= minimum <= rank:
            raise CurriculumError("Stage rank support and minimum_choices must not shrink")
        stages.append(dict(start_epoch=start, end_epoch=end, top_rank=rank,
                           minimum_choices=minimum, temperature=temperature))
        previous_end, previous_rank, previous_minimum = end, rank, minimum
    if previous_end != 250 or stages[-1]["top_rank"] <= stages[0]["top_rank"]:
        raise CurriculumError("The full horizon must end with wider rank support than the first stage")
    if stages[-1]["minimum_choices"] < 2:
        raise CurriculumError("The final stage must require at least two score-eligible choices")
    return dict(format=CONFIG_FORMAT, experiment_id=name, num_epochs=250,
                candidate_count=128, cp_probability=probability, score_floor=floor,
                max_score_drop=gap, stages=stages)


def curriculum_config_sha256(config: Mapping[str, Any]) -> str:
    return canonical_sha256(validate_curriculum_config(config))


def stage_for_epoch(config: Mapping[str, Any], epoch: int) -> tuple[int, Mapping[str, Any]]:
    _integer(epoch, "epoch")
    for index, stage in enumerate(config["stages"]):
        if stage["start_epoch"] <= epoch < stage["end_epoch"]:
            return index, stage
    raise CurriculumError(f"Epoch {epoch} is outside the configured training horizon")


def _scores(scores: Any, config: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (config["candidate_count"],) or not np.all(np.isfinite(values)):
        raise CurriculumError("Expected the full, finite 128-candidate score vector")
    return values


def eligible_candidate_indices(scores: Any, config: Mapping[str, Any], epoch: int) -> np.ndarray:
    config = validate_curriculum_config(config)
    values = _scores(scores, config)
    _, stage = stage_for_epoch(config, epoch)
    order = np.argsort(-values, kind="stable")[:stage["top_rank"]]
    accepted = values[order] >= float(values.max()) - config["max_score_drop"]
    if config["score_floor"] is not None:
        accepted &= values[order] >= config["score_floor"]
    eligible = order[accepted]
    if len(eligible) < stage["minimum_choices"]:
        raise CurriculumError(
            f"Epoch {epoch}: {len(eligible)} score-eligible candidates, "
            f"but stage requires {stage['minimum_choices']}; do not relax the score gate silently"
        )
    return eligible


def _draw(probabilities: np.ndarray, u: float) -> int:
    draw = _number(u, "candidate draw")
    if not 0 <= draw < 1:
        raise CurriculumError("candidate draw must lie in [0, 1)")
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    cumulative[-1] = 1.0
    if np.any(np.diff(np.concatenate(([0.0], cumulative))) <= 0):
        raise CurriculumError("Temperature collapses finite-precision support; specify a larger temperature")
    return int(np.searchsorted(cumulative, draw, side="right"))


def select_curriculum_candidate(scores: Any, config: Mapping[str, Any], epoch: int,
                                u: float, *, basic_control: bool = False) -> int:
    config = validate_curriculum_config(config)
    values = _scores(scores, config)
    _, stage = stage_for_epoch(config, epoch)
    if basic_control:
        eligible = np.arange(values.size)
        probabilities = np.full(values.size, 1.0 / values.size)
    else:
        eligible = eligible_candidate_indices(values, config, epoch)
        if stage["temperature"] is None:
            probabilities = np.full(len(eligible), 1.0 / len(eligible))
        else:
            logits = (values[eligible] - values[eligible].max()) / stage["temperature"]
            weights = np.exp(logits)
            probabilities = weights / weights.sum()
    return int(eligible[_draw(probabilities, u)])


def schedule_token(*values: Any) -> int:
    payload = json.dumps([SCHEDULE_FORMAT, *values], separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def update_digest(value: int, tokens: Any) -> int:
    for token in np.asarray(tokens, dtype=np.uint64).reshape(-1):
        value = ((value ^ int(token)) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value
