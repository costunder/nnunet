"""DEBUG numeric feedback fixtures, not medical samples or trained predictions."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np

from custom_trainers.onlinecp_feedback_policy import (
    CONFIG_FORMAT, MEASUREMENT_DEFINITION, PREDICTION_FORMAT, FeedbackError,
    FeedbackState, canonical_sha256, config_sha256, quality_eligible_indices,
    stage_for_epoch, validate_feedback_config,
)

ROOT = Path(__file__).resolve().parents[1]
ENTRY = "entries/a.npz"


def debug_config():
    return json.loads((ROOT / "config" / "online_cp_feedback.json").read_text(encoding="utf-8"))


def make_state(config=None):
    return FeedbackState(debug_config() if config is None else config,
                         {ENTRY: "a", "entries/b.npz": "b"},
                         identity={"train_case_ids": ["a", "b", "no_cp_source"],
                                   "validation_case_ids": ["heldout"], "bank_sha256": "0" * 64})


def observation(index=0, error=.2, **changes):
    return {"entry_id": ENTRY, "case_id": "a", "candidate_index": index, "error": error,
            "phase": "train", "timing": "pre_update", **changes}


def advance(state, epoch, errors=None):
    snapshot = state.state_dict()["active_snapshot"]
    current = 0 if snapshot is None else snapshot["epoch"]
    if snapshot is None:
        state.snapshot(0)
    while current < epoch:
        if errors is not None and current == epoch - 1:
            state.update_many([observation(index, float(error)) for index, error in enumerate(errors)], current)
        state.snapshot(current + 1)
        current += 1


def provenance(epoch):
    return {"format": PREDICTION_FORMAT, "gnn_state_sha256": "1" * 64,
            "training_observations_sha256": "2" * 64,
            "train_case_ids": ["a", "b", "no_cp_source"], "trained_through_epoch": epoch - 1,
            "prediction_epoch": epoch, "measurement_definition": MEASUREMENT_DEFINITION,
            "nnunet_progress": {"completed_epoch": epoch - 1, "optimizer_steps": epoch * 250,
                                "network_sha256": "3" * 64}}


def rehash(payload, name):
    payload[name] = canonical_sha256({key: value for key, value in payload.items() if key != name})


class FeedbackPolicyDebugTests(unittest.TestCase):
    def test_explicit_full_configuration_and_hash(self):
        config = validate_feedback_config(debug_config())
        self.assertEqual(config["format"], CONFIG_FORMAT)
        self.assertEqual((config["num_epochs"], config["candidate_count"], config["cp_probability"]), (250, 128, .5))
        changed = copy.deepcopy(config)
        changed["metrics"]["adjacent_fp"] = 2
        self.assertNotEqual(config_sha256(config), config_sha256(changed))
        self.assertNotIn("top_rank", str(config))

    def test_no_hidden_reduction_or_missing_explicit_settings(self):
        for key, value in (("num_epochs", 2), ("candidate_count", 8), ("cp_probability", .2)):
            config = debug_config()
            config[key] = value
            with self.subTest(key=key), self.assertRaises(FeedbackError):
                validate_feedback_config(config)
        for key in debug_config():
            config = debug_config()
            del config[key]
            with self.subTest(missing=key), self.assertRaises(FeedbackError):
                validate_feedback_config(config)
        with self.assertRaises(FeedbackError):
            validate_feedback_config({**debug_config(), "fast_mode": True})

    def test_retention_and_exploration_cannot_be_disabled(self):
        for key in ("exploration_weight", "easy_retention_weight"):
            config = debug_config()
            config["mixture"][key] = 0
            with self.assertRaises(FeedbackError):
                validate_feedback_config(config)
        config = debug_config()
        config["stages"][-1]["target_quantile"] = [0., 1.]
        with self.assertRaises(FeedbackError):
            validate_feedback_config(config)

    def test_stage_boundaries_keep_entire_horizon(self):
        config = debug_config()
        for epoch, expected in ((0, 0), (49, 0), (50, 1), (100, 2), (150, 3), (200, 4), (249, 4)):
            self.assertEqual(stage_for_epoch(config, epoch)[0], expected)
        for epoch in (-1, 250, True, 1.5):
            with self.assertRaises(FeedbackError):
                stage_for_epoch(config, epoch)

    def test_quality_gate_is_fixed_and_not_an_error_label(self):
        scores = np.zeros(128)
        scores[100:] = -10
        self.assertEqual(quality_eligible_indices(scores, debug_config()).tolist(), list(range(100)))
        state = make_state()
        state.snapshot(0)
        probabilities = state.probabilities(ENTRY, scores, 0)
        np.testing.assert_allclose(probabilities[:100], .01)
        self.assertTrue(np.all(probabilities[100:] == 0))
        self.assertIsNone(state.selection_info(ENTRY, 0, scores, 0)["difficulty"])
        scores[:] = -10
        scores[0] = 0
        with self.assertRaisesRegex(FeedbackError, "quality"):
            state.probabilities(ENTRY, scores, 0)
        np.testing.assert_allclose(state.probabilities(ENTRY, scores, 0, basic_control=True), 1 / 128)

    def test_epoch_snapshot_freezes_updates_until_next_epoch(self):
        state = make_state()
        frozen = state.snapshot(0)
        state.update_many([observation(error=0.)], 0)
        self.assertEqual(state.snapshot(0), frozen)
        np.testing.assert_allclose(state.probabilities(ENTRY, np.zeros(128), 0), 1 / 128)
        self.assertIsNone(state.selection_info(ENTRY, 0, np.zeros(128), 0)["difficulty"])
        state.snapshot(1)
        info = state.selection_info(ENTRY, 0, np.zeros(128), 1)
        self.assertEqual((info["estimate_source"], info["difficulty"], info["observation_count"]), ("measured", 0., 1))
        self.assertEqual(state.selection_info(ENTRY, 1, np.zeros(128), 1)["estimate_source"], "exploration_unobserved")

    def test_actual_mean_ema_count_and_epoch(self):
        state = make_state()
        state.snapshot(0)
        state.update_many([observation(error=.2), observation(error=.8)], 0)
        row = state.state_dict()["table"][ENTRY]
        self.assertEqual((row["count"][0], row["last_epoch"][0]), (2, 0))
        self.assertAlmostEqual(row["mean"][0], .5)
        self.assertAlmostEqual(row["ema"][0], .35)
        self.assertEqual((row["mean"][1], row["ema"][1], row["last_epoch"][1]), (None, None, None))
        state.snapshot(1)
        state.update_many([observation(error=.4, epoch=1)], 1)
        row = state.state_dict()["table"][ENTRY]
        self.assertAlmostEqual(row["mean"][0], 1.4 / 3)
        self.assertAlmostEqual(row["ema"][0], .3625)

    def test_easy_medium_hard_uses_actual_error_with_easy_retention(self):
        distributions = []
        for epoch in (1, 100, 200):
            state = make_state()
            advance(state, epoch, np.linspace(0, 1, 128))
            distributions.append(state.probabilities(ENTRY, np.zeros(128), epoch))
        easy, medium, hard = distributions
        self.assertGreater(easy[0], easy[-1])
        self.assertGreater(medium[64], medium[-1])
        self.assertGreater(hard[-1], hard[0])
        self.assertGreater(hard[0], .2 / 128)
        self.assertTrue(np.all(hard >= .2 / 128 - 1e-15))

    def test_equal_difficulty_is_not_split_by_candidate_index(self):
        state = make_state()
        advance(state, 100, np.full(128, .7))
        np.testing.assert_allclose(state.probabilities(ENTRY, np.zeros(128), 100), 1 / 128)
        values = np.repeat([.1, .2, .3, .4], 32)
        first, second = make_state(), make_state()
        advance(first, 200, values)
        advance(second, 200, values[::-1])
        probability = first.probabilities(ENTRY, np.zeros(128), 200)
        np.testing.assert_array_equal(probability, second.probabilities(ENTRY, np.zeros(128), 200)[::-1])
        for offset in range(0, 128, 32):
            self.assertEqual(len(set(probability[offset:offset + 32])), 1)

    def test_stale_measurement_is_explicit_exploration_not_zero(self):
        state = make_state()
        state.snapshot(0)
        state.update_many([observation(error=.9)], 0)
        advance(state, 26)
        info = state.selection_info(ENTRY, 0, np.zeros(128), 26)
        self.assertEqual(info["estimate_source"], "exploration_stale")
        self.assertIsNone(info["difficulty"])
        self.assertEqual(info["observation_count"], 1)
        np.testing.assert_allclose(state.probabilities(ENTRY, np.zeros(128), 26), 1 / 128)

    def test_invalid_or_validation_batch_is_atomic_and_rejected(self):
        state = make_state()
        state.snapshot(0)
        before = state.state_dict()
        bad = [observation(phase="validation"), observation(timing="post_update"),
               observation(case_id="heldout"), observation(entry_id="missing"),
               observation(candidate_index=128), observation(candidate_index=True),
               observation(error=float("nan")), observation(error=float("inf")),
               observation(error=-.1), observation(error=1.1), observation(epoch=1)]
        for record in bad:
            with self.subTest(record=record), self.assertRaises(FeedbackError):
                state.update_many([observation(), record], 0)
            self.assertEqual(before, state.state_dict())

    def test_no_global_rng_draws_and_shared_uniform_is_deterministic(self):
        state = make_state()
        advance(state, 1, np.linspace(0, 1, 128))
        before = np.random.get_state()
        choices = [state.select(ENTRY, np.zeros(128), 1, u) for u in (.0, .1, .5, .999999)]
        self.assertEqual(choices, [state.select(ENTRY, np.zeros(128), 1, u) for u in (.0, .1, .5, .999999)])
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        for draw in (-.1, 1., float("nan")):
            with self.assertRaises(FeedbackError):
                state.select(ENTRY, np.zeros(128), 1, draw)

    def test_resume_roundtrip_and_frozen_worker_replay(self):
        state = make_state()
        state.snapshot(0)
        state.update_many([observation(error=.4)], 0)
        payload = json.loads(json.dumps(state.state_dict(), allow_nan=False))
        restored = make_state()
        restored.load_state_dict(payload)
        self.assertEqual(state.state_dict(), restored.state_dict())
        snapshot = state.snapshot(1)
        self.assertEqual(snapshot, restored.snapshot(1))
        worker = FeedbackState.from_snapshot(json.loads(json.dumps(snapshot)))
        np.testing.assert_array_equal(worker.probabilities(ENTRY, np.zeros(128), 1), state.probabilities(ENTRY, np.zeros(128), 1))
        with self.assertRaisesRegex(FeedbackError, "worker"):
            worker.update_many([observation()], 1)
        with self.assertRaises(FeedbackError):
            worker.snapshot(2)

    def test_resume_rejects_missing_tampered_or_cross_experiment_state(self):
        state = make_state()
        state.snapshot(0)
        with self.assertRaises(FeedbackError):
            state.load_state_dict({"table": {}})
        tampered = state.state_dict()
        tampered["table"][ENTRY]["mean"][0] = 0.
        with self.assertRaisesRegex(FeedbackError, "checksum"):
            state.load_state_dict(tampered)
        rehash(tampered, "state_sha256")
        with self.assertRaisesRegex(FeedbackError, "Unobserved"):
            state.load_state_dict(tampered)
        changed = debug_config()
        changed["difficulty"]["ema_alpha"] = .5
        with self.assertRaisesRegex(FeedbackError, "identity"):
            make_state(changed).load_state_dict(state.state_dict())

    def test_snapshot_rejects_same_epoch_observations_even_with_recomputed_checksum(self):
        state = make_state()
        payload = state.snapshot(0)
        row = payload["table"][ENTRY]
        row["count"][0], row["mean"][0], row["ema"][0], row["last_epoch"][0] = 1, .2, .2, 0
        rehash(payload, "snapshot_sha256")
        with self.assertRaises(FeedbackError):
            FeedbackState.from_snapshot(payload)

    def test_resume_same_counts_cannot_change_frozen_statistics(self):
        state = make_state()
        advance(state, 1, [.2])
        payload = state.state_dict()
        payload["table"][ENTRY]["mean"][0] = .4
        rehash(payload, "state_sha256")
        with self.assertRaisesRegex(FeedbackError, "contradict"):
            state.load_state_dict(payload)

    def test_snapshot_epoch_order_and_initialization_are_explicit(self):
        state = make_state()
        with self.assertRaises(FeedbackError):
            state.probabilities(ENTRY, np.zeros(128), 0)
        with self.assertRaises(FeedbackError):
            state.snapshot(1)
        state.snapshot(0)
        with self.assertRaises(FeedbackError):
            state.snapshot(2)

    def test_predictions_are_distinct_and_measured_error_has_priority(self):
        state = make_state()
        state.snapshot(0)
        state.update_many([observation(error=.8)], 0)
        state.snapshot(1, predicted_difficulties={ENTRY: [.2] * 128}, prediction_provenance=provenance(1))
        measured = state.selection_info(ENTRY, 0, np.zeros(128), 1)
        predicted = state.selection_info(ENTRY, 1, np.zeros(128), 1)
        self.assertEqual((measured["estimate_source"], measured["difficulty"]), ("measured", .8))
        self.assertEqual((predicted["estimate_source"], predicted["difficulty"]), ("predicted", .2))
        self.assertEqual(predicted["observation_count"], 0)
        self.assertIsNone(state.state_dict()["table"][ENTRY]["mean"][1])

    def test_missing_and_stale_predictions_do_not_become_labels(self):
        state = make_state()
        state.snapshot(0)
        state.snapshot(1)
        stale = provenance(2)
        stale["trained_through_epoch"] = 0
        state.snapshot(2, predicted_difficulties={ENTRY: [.9] * 128}, prediction_provenance=stale)
        info = state.selection_info(ENTRY, 0, np.zeros(128), 2)
        self.assertEqual(info["estimate_source"], "exploration_stale_prediction")
        self.assertIsNone(info["difficulty"])

    def test_prediction_provenance_rejects_future_validation_wrong_metric_and_missing_values(self):
        variants = [dict(trained_through_epoch=1), dict(prediction_epoch=2),
                    dict(train_case_ids=["heldout"]), dict(measurement_definition="quality_rank"),
                    dict(gnn_state_sha256="not-a-digest")]
        for changes in variants:
            state = make_state()
            state.snapshot(0)
            with self.subTest(changes=changes), self.assertRaises(FeedbackError):
                state.snapshot(1, predicted_difficulties={ENTRY: [.2] * 128},
                               prediction_provenance={**provenance(1), **changes})
        for values in ([.2] * 127, [None] * 128, [float("nan")] * 128):
            state = make_state()
            state.snapshot(0)
            with self.assertRaises(FeedbackError):
                state.snapshot(1, predicted_difficulties={ENTRY: values}, prediction_provenance=provenance(1))

    def test_epoch_prediction_snapshot_cannot_be_replaced(self):
        state = make_state()
        state.snapshot(0)
        state.snapshot(1, predicted_difficulties={ENTRY: [.2] * 128}, prediction_provenance=provenance(1))
        with self.assertRaisesRegex(FeedbackError, "cannot be replaced"):
            state.snapshot(1, predicted_difficulties={ENTRY: [.3] * 128}, prediction_provenance=provenance(1))


if __name__ == "__main__":
    unittest.main()
