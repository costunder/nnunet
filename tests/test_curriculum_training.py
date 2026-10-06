"""Synthetic UNIT gate/loss/orchestration checks; never CT or quality evidence.

Finite UNIT feature values and explicit fake scorer outputs verify routing and
metric definitions. No trained checkpoint or completed neural run is fabricated.
"""
from collections import Counter
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hiercp_v222.v1_execution import rng_state
from hiercp_v222.v1_local import support_for_recipient
from l0_regions.candidate_curriculum import Curriculum, FORMAT, METRIC_KEYS
from l0_regions.curriculum_training import evaluate_gate
from tools.run_local_cnn_experiment import run_experiment


def unit_config():
    return dict(format=FORMAT, group_size=16, required_pair_win=.9,
                required_hit_at_1=.9, minimum_mean_margin=0., sustained_epochs=2)


class PostPredictionRow(dict):
    """Fail if a query target is read before its fake UNIT score exists."""
    def __init__(self, value, index, predicted):
        super().__init__(value)
        self.index, self.predicted = index, predicted

    def __getitem__(self, key):
        if key == "target" and self.index not in self.predicted:
            raise AssertionError("Query GT was read before predicting the same query")
        return super().__getitem__(key)


class ForbiddenLocalEncoder(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("TRAIN gate must reuse fresh L0 memory; CNN must not run")


class FiniteUnitScorer(nn.Module):
    """Explicit fake UNIT scores; this is not a placement-model substitute."""
    def __init__(self, score_lookup, predicted):
        super().__init__()
        self.local = ForbiddenLocalEncoder()
        self.child = nn.Dropout(.1)
        self.score_lookup, self.predicted = score_lookup, predicted
        self.query_calls, self.support_calls = [], []
        self.fail = None
        self.output_override = None

    def prepare_support(self, embeddings, owners, classes):
        self.support_calls.append((embeddings.clone(), owners.clone(), classes.clone()))
        return {"UNIT_full_support": True}

    def predict_embeddings(self, query, state):
        if any(module.training for module in self.modules()):
            raise AssertionError("Gate scoring must run in eval mode")
        # Exercise all RNG streams the helper promises to preserve.
        torch.rand(1); np.random.random(); random.random()
        if self.fail is not None:
            raise RuntimeError(self.fail)
        self.asserted_query_shape = tuple(query.shape)
        ids = [int(value) for value in query[:, 0].tolist()]
        self.query_calls.append(ids)
        self.predicted.update(ids)
        values = torch.tensor([self.score_lookup[index] for index in ids], dtype=torch.float32)
        logits = torch.stack((torch.zeros_like(values), values), dim=1)
        if self.output_override is not None:
            logits = self.output_override(logits)
        return {"logits": logits}


def gate_fixture(*, ties=False, positives=(2, 1, 0, 1)):
    rows, scores = [], {}
    p_values = ((5., -1.), (-2.,), (), (1.,))
    for case_index, positive_count in enumerate(positives):
        case = "UNIT-case-" + str(case_index)
        group = "patient:" + case
        # The second patient's donor is the first patient. This checks the
        # original helper's donor-side exclusion as well as recipient exclusion.
        donor_case = "UNIT-case-0" if case_index == 1 else "UNIT-donor-" + str(case_index)
        for local_index in range(positive_count + 128):
            index = len(rows)
            target = int(local_index < positive_count)
            rows.append(dict(id=case + ":" + str(local_index), case_id=case,
                center=[local_index, case_index, 0], target=target,
                patient_group=group, donor_case_id=donor_case,
                donor_group="patient:" + donor_case, donor_component=1,
                bounds=dict(edges=local_index), UNIT_synthetic=True))
            scores[index] = 0. if ties or not target else p_values[case_index][local_index]
    curriculum = Curriculum(rows, unit_config())
    names = sorted({row["patient_group"] for row in rows})
    embeddings = torch.zeros((len(rows), 128), dtype=torch.float32)
    embeddings[:, 0] = torch.arange(len(rows))
    memory = dict(embeddings=embeddings, record_ids=[row["id"] for row in rows],
        patient_groups=names, donor_groups=[row["donor_group"] for row in rows],
        owners=torch.tensor([names.index(row["patient_group"]) for row in rows]),
        classes=torch.tensor([row["target"] for row in rows]))
    predicted = set()
    ds = SimpleNamespace(rows=[PostPredictionRow(row, index, predicted) for index, row in enumerate(rows)])
    net = FiniteUnitScorer(scores, predicted)
    net.train(); net.child.eval()  # Deliberately heterogeneous module modes.
    return SimpleNamespace(ds=ds, rows=rows, memory=memory, curriculum=curriculum,
                           net=net, predicted=predicted)


class CurriculumGateUnit(unittest.TestCase):
    def assert_rng_equal(self, actual, expected):
        self.assertTrue(torch.equal(actual["torch"], expected["torch"]))
        self.assertEqual(len(actual["cuda"]), len(expected["cuda"]))
        for left, right in zip(actual["cuda"], expected["cuda"]):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(actual["numpy"][0], expected["numpy"][0])
        np.testing.assert_array_equal(actual["numpy"][1], expected["numpy"][1])
        self.assertEqual(actual["numpy"][2:], expected["numpy"][2:])
        self.assertEqual(actual["python"], expected["python"])

    def test_gate_scores_exact_active_global_rows_including_zeroP_without_CNN_or_GT(self):
        fixture = gate_fixture()
        memory_before = fixture.memory["embeddings"].clone()
        report = evaluate_gate(fixture.net, fixture.ds, fixture.memory, fixture.curriculum)
        expected = fixture.curriculum.active_indices()
        actual = [index for query in fixture.net.query_calls for index in query]
        self.assertEqual(sorted(actual), expected)
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(len(actual), 4 + 4 * 16)
        self.assertTrue(any(index >= len(actual) for index in actual))
        self.assertEqual(fixture.predicted, set(expected))
        self.assertEqual(report["metrics"]["cases_without_P"], 1)
        zeroP = next(row for row in report["cases"] if row["case_id"] == "UNIT-case-2")
        self.assertFalse(zeroP["rank_evaluable"])
        self.assertEqual(zeroP["active_U"], 16)
        self.assertEqual(len(zeroP["query_record_ids"]), 16)
        self.assertFalse(report["query_GT_in_forward"])
        self.assertFalse(report["validation_used_for_gate"])
        self.assertFalse(report["L0_reencoded"])
        self.assertTrue(torch.equal(fixture.memory["embeddings"], memory_before))

    def test_original_exclusion_helper_receives_full_bank_with_deferred_U(self):
        fixture = gate_fixture()
        with patch("l0_regions.curriculum_training.support_for_recipient",
                   wraps=support_for_recipient) as excluded:
            evaluate_gate(fixture.net, fixture.ds, fixture.memory, fixture.curriculum)
        self.assertEqual(excluded.call_count, 4)
        self.assertTrue(all(call.args[0] is fixture.memory for call in excluded.call_args_list))
        self.assertEqual({call.args[1] for call in excluded.call_args_list}, set(fixture.memory["patient_groups"]))
        active = set(fixture.curriculum.active_indices())
        for case_index, (support_embeddings, _, _) in enumerate(fixture.net.support_calls):
            query_group = "patient:UNIT-case-" + str(case_index)
            expected = [index for index, row in enumerate(fixture.rows)
                        if query_group not in (row["patient_group"], row["donor_group"])]
            actual = [int(value) for value in support_embeddings[:, 0].tolist()]
            self.assertEqual(actual, expected)
            self.assertTrue(set(actual) - active)  # Deferred U still support.

    def test_metric_pair_micro_firstP_case_hit_and_bestP_minus_bestU_mean(self):
        fixture = gate_fixture()
        report = evaluate_gate(fixture.net, fixture.ds, fixture.memory, fixture.curriculum)
        metrics = report["metrics"]
        self.assertEqual(metrics["pairs"], 64)
        self.assertAlmostEqual(metrics["pair_win_rate"], .5)
        self.assertEqual(metrics["pair_tie_rate"], 0.)
        self.assertAlmostEqual(metrics["hit_at_1"], 2 / 3)
        self.assertNotAlmostEqual(metrics["hit_at_1"], 2 / 4)  # P-micro R@1 differs.
        self.assertAlmostEqual(metrics["mean_margin"], (5. - 2. + 1.) / 3)
        self.assertEqual(metrics["rank_evaluable_cases"], 3)
        self.assertEqual(metrics["active_query_records"], 68)
        self.assertIn("max(P score)", report["margin_definition"])

    def test_exact_ties_never_pass_training_gate_or_promote_next_stage(self):
        fixture = gate_fixture(ties=True)
        report = evaluate_gate(fixture.net, fixture.ds, fixture.memory, fixture.curriculum)
        metrics = report["metrics"]
        self.assertEqual(metrics["pair_win_rate"], 0.)
        self.assertEqual(metrics["pair_tie_rate"], 1.)
        self.assertEqual(metrics["hit_at_1"], 0.)
        self.assertEqual(metrics["mean_margin"], 0.)
        receipt = fixture.curriculum.observe({key: metrics[key] for key in METRIC_KEYS}, 1)
        self.assertFalse(receipt["gate_passed"])
        self.assertFalse(receipt["promoted"])
        self.assertEqual(receipt["streak"], 0)
        self.assertEqual(receipt["next_epoch_active_U"], 16)

    def test_explicit_70percent_pair_gate_can_progress_with_Hit1_log_only(self):
        fixture = gate_fixture()
        config = dict(unit_config(), required_pair_win=.7, required_hit_at_1=0.)
        fixture.curriculum = Curriculum(fixture.rows, config)
        # One case fails top1 while almost all of its P/U pairs still win.
        # The two other cases keep the declared mean best-P/best-U margin >0.
        for index, row in enumerate(fixture.rows):
            if row["case_id"] == "UNIT-case-0" and row["target"]:
                fixture.net.score_lookup[index] = 5.
            if row["case_id"] == "UNIT-case-1" and not row["target"]:
                fixture.net.score_lookup[index] = -3.
        hard_U = fixture.curriculum.unobserved_order["UNIT-case-1"][0]
        fixture.net.score_lookup[hard_U] = 0.
        report = evaluate_gate(fixture.net, fixture.ds, fixture.memory, fixture.curriculum)
        metrics = report["metrics"]
        self.assertGreater(metrics["pair_win_rate"], .7)
        self.assertAlmostEqual(metrics["hit_at_1"], 2 / 3)
        self.assertLess(metrics["hit_at_1"], .9)
        self.assertGreater(metrics["mean_margin"], 0.)
        for epoch in (1, 2):
            receipt = fixture.curriculum.observe({key: metrics[key] for key in METRIC_KEYS}, epoch)
            self.assertTrue(receipt["gate_passed"])
        self.assertTrue(receipt["promoted"])
        self.assertEqual(receipt["next_epoch_active_U"], 32)
        self.assertEqual(fixture.curriculum.config["required_hit_at_1"], 0.)

    def test_rng_and_every_heterogeneous_module_mode_restored_on_success_and_error(self):
        for failing in (False, True):
            with self.subTest(failing=failing):
                fixture = gate_fixture()
                modes = [(module, module.training) for module in fixture.net.modules()]
                before = rng_state()
                if failing:
                    fixture.net.fail = "explicit UNIT prediction failure"
                    with self.assertRaisesRegex(RuntimeError, "UNIT prediction failure"):
                        evaluate_gate(fixture.net, fixture.ds, fixture.memory, fixture.curriculum)
                else:
                    evaluate_gate(fixture.net, fixture.ds, fixture.memory, fixture.curriculum)
                self.assert_rng_equal(rng_state(), before)
                self.assertEqual([(module, module.training) for module, _ in modes], modes)

    def test_bad_full_bank_or_nonfinite_incomplete_scores_rejected_without_fake_metrics(self):
        for mode in ("order", "shape", "gradient", "nan", "logit_count"):
            with self.subTest(mode=mode):
                fixture = gate_fixture()
                if mode == "order":
                    fixture.memory["record_ids"] = fixture.memory["record_ids"][::-1]
                elif mode == "shape":
                    fixture.memory["embeddings"] = fixture.memory["embeddings"][:, :127]
                elif mode == "gradient":
                    fixture.memory["embeddings"].requires_grad_(True)
                elif mode == "nan":
                    fixture.net.output_override = lambda logits: torch.full_like(logits, float("nan"))
                else:
                    fixture.net.output_override = lambda logits: logits[:-1]
                with self.assertRaises(ValueError):
                    evaluate_gate(fixture.net, fixture.ds, fixture.memory, fixture.curriculum)

    def test_noP_population_reports_missing_gate_evidence_by_error(self):
        fixture = gate_fixture(positives=(0, 0, 0, 0))
        with self.assertRaisesRegex(ValueError, "No training P/U case"):
            evaluate_gate(fixture.net, fixture.ds, fixture.memory, fixture.curriculum)


class CurriculumLiveLossUnit(unittest.TestCase):
    def test_active_context_uses_live_rank_and_active_balanced_CE_denominators(self):
        from l0_regions.training import forward_loss
        fixture = gate_fixture()
        context = fixture.curriculum.context(32)
        ids = context.order[0]
        targets = torch.tensor([fixture.rows[index]["target"] for index in ids])
        features = torch.tensor([[.1 * index, .2 * index + .5] for index in range(len(ids))], requires_grad=True)
        class ExplicitUnitLossNet:
            def local(self, query):
                return query.UNIT_logits
            def prepare_support(self, *support, cluster_plan=None):
                return {"UNIT": True}
            def predict_embeddings(self, query, state):
                return dict(logits=query, alignment_loss=query.new_tensor(0.), alignment_loss_weight=0.)
        query = SimpleNamespace(UNIT_logits=features)
        with patch("l0_regions.training.legacy_forward_loss", side_effect=AssertionError("ActiveContext routed to legacy stale-reference loss")):
            loss, terms = forward_loss(ExplicitUnitLossNet(), query, (), None, targets, None,
                context, dict(ranking_weight=1., observation_auxiliary_weight=1.), indices=ids)
        scores = features[:, 1] - features[:, 0]
        differences = scores[targets == 1, None] - scores[None, targets == 0]
        expected_rank = F.softplus(-differences).sum() * context.steps / context.pairs
        weights = features.new_tensor([context.steps / (2 * context.counts[fixture.rows[index]["target"]]
                                                        * context.uses[index]) for index in ids])
        expected_ce = (F.cross_entropy(features, targets, reduction="none") * weights).sum()
        self.assertTrue(torch.allclose(terms["ranking_loss"], expected_rank))
        self.assertTrue(torch.allclose(terms["observation_auxiliary_loss"], expected_ce))
        self.assertTrue(torch.allclose(loss, expected_rank + expected_ce))
        self.assertEqual(int(terms["ranking_pairs"]), differences.numel())
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(bool((features.grad.abs().sum(dim=1) > 0).all()))


class CurriculumExperimentUnit(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / "work"
        parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="curriculum_training_UNIT_", dir=parent)
        self.root = Path(self.temporary.name)
        assert self.root.resolve().parent == parent.resolve()
        self.addCleanup(self.temporary.cleanup)
        cfg = self.root / "curriculum.json"
        cfg.write_text(json.dumps(unit_config()), encoding="utf8")
        self.args = SimpleNamespace(experiment=self.root / "new_m10_curriculum", cache=self.root / "original.json",
            config=self.root / "model.json", curriculum_config=cfg, margin_mm=10, debug=True,
            workers=4, cuda_gib=12, rss_gib=32, resident_gib=8, batch_candidates=[32],
            support_patients=16, device_cache_gib=0, debug_pause_step=None)
        self.request = dict(original_inventory_sha256="SYNTHETIC UNIT inventory marker",
            local_cnn={"margin_mm": 10}, candidate_curriculum=unit_config())
        self.calls = []

    def child(self, command):
        """Fake orchestration child, explicitly never a model checkpoint."""
        self.calls.append(command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        if "prepare" in command:
            (output / "index.json").write_text(json.dumps(dict(format="native_local_cnn_inventory_v1",
                original_inventory_sha256=self.request["original_inventory_sha256"],
                local_cnn=self.request["local_cnn"], debug=True)), encoding="utf8")
        else:
            (output / "checkpoint_latest.pt").write_bytes(b"SYNTHETIC UNIT orchestration marker; NOT MODEL WEIGHTS")

    def test_config_propagated_to_train_and_exact_own_resume_not_preparation(self):
        first = run_experiment(self.args, self.request, self.child)
        original = first.read_bytes()
        second = run_experiment(self.args, self.request, self.child)
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), original)
        training = [call for call in self.calls if "train" in call]
        self.assertEqual(len(training), 2)
        for command in training:
            self.assertEqual(command[command.index("--curriculum-config") + 1], str(self.args.curriculum_config.resolve()))
        self.assertEqual(training[1][training[1].index("--resume") + 1], str(first))
        preparation = [call for call in self.calls if "prepare" in call]
        self.assertEqual(len(preparation), 1)
        self.assertNotIn("--curriculum-config", preparation[0])

    def test_changed_curriculum_request_rejected_before_child_execution(self):
        first = run_experiment(self.args, self.request, self.child)
        calls = len(self.calls)
        altered = copy.deepcopy(self.request)
        altered["candidate_curriculum"]["required_pair_win"] = .95
        with self.assertRaisesRegex(ValueError, "settings/source/data differ"):
            run_experiment(self.args, altered, self.child)
        self.assertEqual(len(self.calls), calls)
        self.assertTrue(first.is_file())

    def test_curriculum_cannot_adopt_old_noncurriculum_checkpoint(self):
        inventory = self.args.experiment / "inventory"
        self.child(["prepare", "--output", str(inventory)])
        training = self.args.experiment / "training"
        training.mkdir()
        (training / "paused.json").write_text("{}", encoding="utf8")
        checkpoint = training / "checkpoint_latest.pt"
        checkpoint.write_bytes(b"SYNTHETIC OLD UNIT marker; NOT MODEL WEIGHTS")
        calls = len(self.calls)
        with self.assertRaisesRegex(ValueError, "separate experiment"):
            run_experiment(self.args, self.request, self.child)
        self.assertEqual(len(self.calls), calls)
        self.assertFalse((self.args.experiment / "experiment.json").exists())
        self.assertEqual(checkpoint.read_bytes(), b"SYNTHETIC OLD UNIT marker; NOT MODEL WEIGHTS")


if __name__ == "__main__":
    unittest.main()
