"""Selected historical BEST/arm binding regressions; no neural CPU smoke."""
import copy
from pathlib import Path

import unittest
import torch

from hiercp_v1x.historical_checkpoint import validate_checkpoint_metadata


SCOPE = "a1" * 32
EXPERIMENT = "b2" * 32


def fixture(arm="V1"):
    config = {"model": {"hidden_dim": 128, "heads": 4},
              "graph": {"geometry_contract": "physical"}, "ct_clip": [-200, 250]}
    state = {"v1x_bounded_scope_digest": torch.tensor(list(bytes.fromhex(SCOPE)), dtype=torch.uint8)}
    if arm in ("A", "B"):
        state[f"v1x_half_{arm.lower()}_digest"] = torch.tensor(list(bytes.fromhex(EXPERIMENT)), dtype=torch.uint8)
    common = {"architecture_version": "arch", "model_kwargs": config["model"],
              "graph_config": config["graph"], "geometry_contract": "physical",
              "ct_clip": (-200., 250.), "cache_publication": {"index": "stable"},
              "training_signature": {"seed": 42, "target_epochs": 40, "run_mode": "production"},
              "validation_policy": {"selection": "own8"}, "preflight_calibration": {"batch": 1},
              "state_dict": state, "training_complete": True, "target_epochs": 40,
              "best_epoch": 11, "best_selection": {"mrr": 1.},
              "gradient_connectivity": {"verified": True}}
    best = {**copy.deepcopy(common), "epoch": 11, "completed_epoch": 40}
    path = Path("checkpoint_best.pt").resolve()
    last = {**copy.deepcopy(common), "format": "hiercp_training_state_v1", "epoch": 40,
            "best_checkpoint": str(path)}
    kwargs = dict(config=config, checkpoint_path=path, scope_digest=SCOPE, arm=arm,
                  experiment_digest=EXPERIMENT if arm != "V1" else None)
    return best, last, kwargs


def selected_best_keeps_original_own_task_and_does_not_mutate(arm):
    best, last, kwargs = fixture(arm)
    before = copy.deepcopy(best["best_selection"])
    result = validate_checkpoint_metadata(best, last, **kwargs)
    assert result == {"selected_epoch": 11, "completed_epochs": 40,
                      "selection": before, "selection_task": "original8_candidate_curriculum",
                      "optimizer_imported": False, "last_used_as_best": False}
    result["selection"]["mrr"] = 0.
    assert best["best_selection"] == before


def rejects_unselected_or_incomplete_best(field, value):
    best, last, kwargs = fixture()
    best[field] = value
    with unittest.TestCase().assertRaisesRegex(ValueError, "selected by this complete"):
        validate_checkpoint_metadata(best, last, **kwargs)


def rejects_best_last_metadata_disagreement(field):
    best, last, kwargs = fixture()
    last[field] = None
    with unittest.TestCase().assertRaisesRegex(ValueError, field):
        validate_checkpoint_metadata(best, last, **kwargs)


def rejects_changed_selected_checkpoint_path():
    best, last, kwargs = fixture()
    last["best_checkpoint"] = str(Path("checkpoint_latest.pt").resolve())
    with unittest.TestCase().assertRaisesRegex(ValueError, "selected by this complete"):
        validate_checkpoint_metadata(best, last, **kwargs)


def rejects_marker_in_state_wrong_scope(arm):
    best, last, kwargs = fixture(arm)
    best["state_dict"]["v1x_bounded_scope_digest"][0] ^= 1
    with unittest.TestCase().assertRaisesRegex(ValueError, "bound identity"):
        validate_checkpoint_metadata(best, last, **kwargs)


def rejects_mixed_experimental_arm(arm, other):
    best, last, kwargs = fixture(arm)
    best["state_dict"][f"v1x_half_{other}_digest"] = torch.zeros(32, dtype=torch.uint8)
    with unittest.TestCase().assertRaisesRegex(ValueError, "another experimental arm"):
        validate_checkpoint_metadata(best, last, **kwargs)


def rejects_digest_with_wrong_type(dtype):
    best, last, kwargs = fixture()
    best["state_dict"]["v1x_bounded_scope_digest"] = best["state_dict"]["v1x_bounded_scope_digest"].to(dtype)
    with unittest.TestCase().assertRaisesRegex(ValueError, "bound identity"):
        validate_checkpoint_metadata(best, last, **kwargs)


def rejects_debug_training_signature_despite_complete_flag():
    best, last, kwargs = fixture()
    for payload in (best, last):
        payload["training_signature"]["run_mode"] = "debug"
    with unittest.TestCase().assertRaisesRegex(ValueError, "production contract"):
        validate_checkpoint_metadata(best, last, **kwargs)


def rejects_missing_gradient_connectivity():
    best, last, kwargs = fixture()
    best["gradient_connectivity"] = {"verified": False}
    with unittest.TestCase().assertRaisesRegex(ValueError, "production contract"):
        validate_checkpoint_metadata(best, last, **kwargs)


class HistoricalCheckpointBindingTests(unittest.TestCase):
    """Each generated check is an independent stdlib unittest case."""


def _check(function, args):
    def run(self):
        function(*args)
    return run


for _name, _function, _args in (
    [("selected_" + arm, selected_best_keeps_original_own_task_and_does_not_mutate, (arm,)) for arm in ("V1", "A", "B")]
    + [("incomplete_" + key, rejects_unselected_or_incomplete_best, (key, value)) for key, value in
       (("epoch", 40), ("completed_epoch", 39), ("training_complete", False), ("target_epochs", 1), ("best_epoch", 20))]
    + [("metadata_" + key, rejects_best_last_metadata_disagreement, (key,)) for key in
       ("model_kwargs", "graph_config", "geometry_contract", "ct_clip", "cache_publication", "training_signature", "validation_policy", "preflight_calibration")]
    + [("path", rejects_changed_selected_checkpoint_path, ())]
    + [("scope_" + arm, rejects_marker_in_state_wrong_scope, (arm,)) for arm in ("V1", "A", "B")]
    + [("mixed_" + arm, rejects_mixed_experimental_arm, (arm, other)) for arm, other in (("V1", "a"), ("A", "b"), ("B", "a"))]
    + [("marker_dtype_" + str(dtype).split(".")[-1], rejects_digest_with_wrong_type, (dtype,)) for dtype in (torch.float32, torch.int64)]
    + [("debug", rejects_debug_training_signature_despite_complete_flag, ()),
       ("gradient", rejects_missing_gradient_connectivity, ())]
):
    setattr(HistoricalCheckpointBindingTests, "test_" + _name, _check(_function, _args))


if __name__ == "__main__":
    unittest.main()
