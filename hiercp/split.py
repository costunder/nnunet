"""Deterministic case-level train/validation splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


SPLIT_FORMAT = "hiercp_case_split_v1"


def make_case_split(
    case_ids: Sequence[str],
    *,
    val_fraction: float,
    seed: int,
) -> dict[str, object]:
    unique = sorted(set(str(case_id) for case_id in case_ids))
    if not unique:
        raise ValueError("No case IDs supplied")
    rng = np.random.default_rng(int(seed))
    shuffled = list(unique)
    rng.shuffle(shuffled)
    if len(shuffled) <= 1 or val_fraction <= 0:
        validation: list[str] = []
        training = shuffled
    else:
        count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * val_fraction))))
        validation = sorted(shuffled[:count])
        training = sorted(shuffled[count:])
    return {
        "format": SPLIT_FORMAT,
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "train": training,
        "val": validation,
    }


def save_case_split(payload: dict[str, object], path: str | Path) -> None:
    if payload.get("format") != SPLIT_FORMAT:
        raise ValueError("Invalid split payload")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_case_split(path: str | Path) -> dict[str, object]:
    from hiercp.contracts import case_ids
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != SPLIT_FORMAT:
        raise ValueError(f"Unsupported split format: {payload.get('format')}")
    train = case_ids(payload.get("train"), "GNN training")
    val = case_ids(payload.get("val"), "GNN validation", allow_empty=True)
    if "outer_validation_excluded" in payload:
        excluded = case_ids(payload["outer_validation_excluded"], "outer validation")
        if (set(train) | set(val)) & set(excluded):
            raise ValueError("Outer validation/test patient occurs in GNN train/validation")
    if not train:
        raise ValueError("Split contains no training cases")
    if set(train) & set(val):
        raise ValueError("Training and validation case IDs overlap")
    payload["train"] = train
    payload["val"] = val
    return payload
