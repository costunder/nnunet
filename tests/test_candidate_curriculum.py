"""UNIT curriculum metadata/schedule evidence only; no CT/model/training output."""
from collections import Counter
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

from l0_regions.candidate_curriculum import ActiveContext, Curriculum, FORMAT, validate_config


def unit_config():
    return dict(format=FORMAT, group_size=16, required_pair_win=.9,
                required_hit_at_1=.9, minimum_mean_margin=0., sustained_epochs=2)


def unit_rows(*, U=128, positives=(3, 19, 0), shuffled=True):
    rows = []
    for case_index, count in enumerate(positives):
        case = "UNIT_case_" + str(case_index)
        for index in range(count+U):
            target = int(index < count)
            rows.append(dict(id=case+":"+str(index), case_id=case,
                target=target, center=[index, case_index, 0],
                patient_group="patient:"+case, donor_case_id="UNIT_donor_"+str(case_index),
                donor_group="patient:UNIT_donor_"+str(case_index), donor_component=1,
                bounds=dict(edges=index*10+case_index), UNIT_metadata_fixture=True))
    if shuffled:
        random.Random(815).shuffle(rows)
    return rows


def gate(curriculum, *, pair=.9, hit=.9, margin=.01):
    return dict(pair_win_rate=pair, hit_at_1=hit, mean_margin=margin,
                rank_evaluable_cases=curriculum.rank_evaluable_cases)


class CandidateCurriculumUnit(unittest.TestCase):
    def setUp(self):
        self.rows = unit_rows()
        self.cfg = unit_config()
        self.curriculum = Curriculum(self.rows, self.cfg)

    def test_config_is_explicit_finite_and_does_not_mutate_input(self):
        before = copy.deepcopy(self.cfg)
        self.assertEqual(validate_config(self.cfg), before)
        self.assertEqual(self.cfg, before)
        invalid = [dict(self.cfg, extra=True), {key: value for key, value in self.cfg.items() if key != "format"}]
        for key, value in (("format", "unknown"), ("group_size", 8), ("group_size", True),
                           ("required_pair_win", float("nan")), ("required_hit_at_1", 1.1),
                           ("required_pair_win", -.1), ("required_hit_at_1", True),
                           ("minimum_mean_margin", float("inf")), ("minimum_mean_margin", -.1),
                           ("sustained_epochs", 0), ("sustained_epochs", True)):
            invalid.append(dict(self.cfg, **{key: value}))
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)

    def test_production_requires_each_actual_128U_and_debug_never_proves_completion(self):
        with self.assertRaises(ValueError):
            Curriculum(unit_rows(U=7), self.cfg)
        debug = Curriculum(unit_rows(U=7), self.cfg, debug=True)
        self.assertTrue(debug.all_candidates_active)
        self.assertFalse(debug.curriculum_complete)
        self.assertTrue(debug.context(32).audit["debug"])
        for epoch in range(1, 17):
            debug.observe(gate(debug), epoch)
        self.assertEqual(debug.active_U, 128)
        self.assertTrue(debug.final_stage_gate_passed)
        self.assertFalse(debug.curriculum_complete)

    def test_allP_zeroP_and_original_global_indices_are_retained(self):
        current = self.curriculum
        active = current.active_indices()
        self.assertEqual(len(active), 22+3*16)
        self.assertEqual(active, sorted(active))
        self.assertTrue(any(index >= len(active) for index in active))
        self.assertEqual({i for i in active if self.rows[i]["target"]},
                         {i for i, row in enumerate(self.rows) if row["target"]})
        context = current.context(32)
        self.assertIsInstance(context, ActiveContext)
        self.assertEqual(list(context.rows), self.rows)
        self.assertEqual(context.counts, Counter({1: 22, 0: 48}))
        self.assertEqual(context.pairs, 22*16)
        self.assertEqual(context.audit["zero_positive_cases"], 1)
        self.assertEqual(context.audit["active_cases"], 3)
        self.assertEqual(context.audit["deferred_rows"], 3*112)
        self.assertTrue(any(not any(self.rows[i]["target"] for i in tile) for tile in context.order))

    def test_geometry_order_is_seeded_identity_based_not_edge_cost_or_input_ordinal(self):
        altered = copy.deepcopy(self.rows)
        for row in altered:
            row["bounds"]["edges"] = 10**7-row["bounds"]["edges"]
        same = Curriculum(altered, self.cfg)
        def selected_U_ids(curriculum):
            return {curriculum.rows[i]["id"] for i in curriculum.active_indices()
                    if curriculum.rows[i]["target"] == 0}
        self.assertEqual(selected_U_ids(same), selected_U_ids(self.curriculum))
        reordered = Curriculum(list(reversed(self.rows)), self.cfg)
        self.assertEqual(selected_U_ids(reordered), selected_U_ids(self.curriculum))
        different_seed = Curriculum(self.rows, self.cfg, seed=43)
        self.assertNotEqual(selected_U_ids(different_seed), selected_U_ids(self.curriculum))
        self.assertEqual(self.rows, unit_rows())

    def test_every_active_PU_pair_occurs_exactly_once_for_multiple_physical_batches(self):
        current = self.curriculum
        for batch in (2, 5, 16, 32, 64):
            schedule = list(current.groups(batch, epoch=3))
            observed_pairs = Counter()
            for tile in schedule:
                self.assertLessEqual(len(tile), batch)
                self.assertEqual(len(set(tile)), len(tile))
                self.assertEqual(len({self.rows[i]["case_id"] for i in tile}), 1)
                p = [i for i in tile if self.rows[i]["target"]]
                u = [i for i in tile if not self.rows[i]["target"]]
                observed_pairs.update((i, j) for i in p for j in u)
            active = current.active_indices()
            expected_pairs = {(i, j) for i in active for j in active
                              if self.rows[i]["target"] == 1 and self.rows[j]["target"] == 0
                              and self.rows[i]["case_id"] == self.rows[j]["case_id"]}
            with self.subTest(batch=batch):
                self.assertEqual(observed_pairs, Counter({pair: 1 for pair in expected_pairs}))
                self.assertEqual(set(i for tile in schedule for i in tile), set(active))

    def test_case_episodes_are_contiguous_and_shuffle_is_deterministic(self):
        first = list(self.curriculum.groups(5, 7))
        self.assertEqual(first, list(self.curriculum.groups(5, 7)))
        self.assertNotEqual(first, list(self.curriculum.groups(5, 8)))
        compressed = []
        for tile in first:
            case = self.rows[tile[0]]["case_id"]
            if not compressed or compressed[-1] != case:
                compressed.append(case)
        self.assertEqual(len(compressed), 3)
        self.assertEqual(len(set(compressed)), 3)
        for batch, epoch in ((1, 0), (True, 0), (32, -1), (32, True)):
            with self.assertRaises(ValueError):
                list(self.curriculum.groups(batch, epoch))

    def test_active_normalization_balances_both_classes_and_repeated_presentations(self):
        context = self.curriculum.context(5)
        mass = Counter()
        per_observation = Counter()
        for tile in context.order:
            for index in tile:
                cls = self.rows[index]["target"]
                coefficient = context.steps/(2*context.counts[cls]*context.uses[index])
                mass[cls] += coefficient/context.steps
                per_observation[index] += coefficient/context.steps
        self.assertAlmostEqual(mass[0], .5)
        self.assertAlmostEqual(mass[1], .5)
        for index, total in per_observation.items():
            self.assertAlmostEqual(total, 1/(2*context.counts[self.rows[index]["target"]]))
        self.assertGreater(max(context.uses.values()), 1)
        before = self.curriculum.context(32)
        self.curriculum.observe(gate(self.curriculum), 1)
        self.curriculum.observe(gate(self.curriculum), 2)
        after = self.curriculum.context(32)
        self.assertEqual(after.pairs, 2*before.pairs)
        self.assertEqual(after.counts[0], 2*before.counts[0])
        self.assertEqual(after.counts[1], before.counts[1])
        self.assertGreater(after.steps, before.steps)

    def test_gate_requires_all_conditions_sustained_and_promotes_only_next_epoch(self):
        current = self.curriculum
        original = set(current.active_indices())
        first = current.observe(gate(current), 1)
        self.assertFalse(first["promoted"])
        self.assertEqual(current.active_U, 16)
        second = current.observe(gate(current), 2)
        self.assertTrue(second["promoted"])
        self.assertEqual(second["completed_epoch_active_U"], 16)
        self.assertEqual(second["next_epoch_active_U"], 32)
        self.assertEqual(second["next_epoch"], 3)
        self.assertEqual(second["streak"], 0)
        self.assertLess(original, set(current.active_indices()))
        current.observe(gate(current), 3)
        failed = current.observe(gate(current, margin=0.), 4)
        self.assertFalse(failed["gate_passed"])
        self.assertEqual(current.streak, 0)
        current.observe(gate(current), 5)
        self.assertEqual(current.active_U, 32)
        current.observe(gate(current), 6)
        self.assertEqual(current.active_U, 48)

    def test_forty_epochs_do_not_force_stage_advance_or_fake_completion(self):
        current = self.curriculum
        for epoch in range(1, 41):
            receipt = current.observe(gate(current, pair=.89), epoch)
            self.assertFalse(receipt["promoted"])
        self.assertEqual(current.active_U, 16)
        self.assertFalse(current.all_candidates_active)
        self.assertFalse(current.curriculum_complete)
        self.assertEqual(len(current.history), 40)

    def test_full128_is_active_before_the_final_stage_gate_is_passed(self):
        current = self.curriculum
        for epoch in range(1, 15):
            current.observe(gate(current), epoch)
        self.assertEqual(current.active_U, 128)
        self.assertTrue(current.all_candidates_active)
        self.assertFalse(current.curriculum_complete)
        self.assertEqual(current.active_indices(), list(range(len(self.rows))))
        current.observe(gate(current), 15)
        current.observe(gate(current), 16)
        self.assertTrue(current.final_stage_gate_passed)
        self.assertTrue(current.curriculum_complete)
        current.observe(gate(current, pair=.1), 17)
        self.assertTrue(current.curriculum_complete)
        self.assertEqual(current.active_U, 128)

    def test_unknown_nonfinite_wrong_denominator_or_duplicate_epoch_rejected_without_state_change(self):
        current = self.curriculum
        invalid = [dict(gate(current), unexpected=True), dict(gate(current), rank_evaluable_cases=3),
                   dict(gate(current), mean_margin=float("nan")), dict(gate(current), hit_at_1=True),
                   dict(gate(current), pair_win_rate=1.01)]
        before = current.state_dict()
        for metrics in invalid:
            with self.subTest(metrics=metrics), self.assertRaises(ValueError):
                current.observe(metrics, 1)
            self.assertEqual(current.state_dict(), before)
        for epoch in (0, 2, True):
            with self.assertRaises(ValueError):
                current.observe(gate(current), epoch)
        current.observe(gate(current), 1)
        before = current.state_dict()
        with self.assertRaises(ValueError):
            current.observe(gate(current), 1)
        self.assertEqual(current.state_dict(), before)

    def test_zeroP_only_ranking_gate_has_no_fabricated_metrics_or_pair_division(self):
        current = Curriculum(unit_rows(positives=(0,)), self.cfg)
        with self.assertRaises(ValueError):
            current.context(32)
        metrics = dict(pair_win_rate=None, hit_at_1=None, mean_margin=None, rank_evaluable_cases=0)
        receipt = current.observe(metrics, 1)
        self.assertFalse(receipt["gate_passed"])
        self.assertFalse(receipt["promoted"])
        with self.assertRaises(ValueError):
            current.observe(dict(metrics, pair_win_rate=0.), 2)

    def test_resume_is_json_roundtrippable_and_exact_future_schedule_matches(self):
        current = self.curriculum
        current.observe(gate(current), 1)
        current.observe(gate(current), 2)
        current.observe(gate(current), 3)
        saved = json.loads(json.dumps(current.state_dict(), allow_nan=False))
        resumed = Curriculum(self.rows, self.cfg)
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.state_dict(), saved)
        self.assertEqual(list(resumed.groups(32, 3)), list(current.groups(32, 3)))
        self.assertEqual(resumed.observe(gate(resumed), 4), current.observe(gate(current), 4))

    def test_resume_rejects_tampered_stage_history_indices_config_seed_or_rows_atomically(self):
        current = self.curriculum
        current.observe(gate(current), 1)
        current.observe(gate(current), 2)
        saved = current.state_dict()
        for kind in ("stage", "streak", "indices", "history", "boolean", "config"):
            changed = copy.deepcopy(saved)
            if kind == "stage":
                changed["stage"] += 1
            elif kind == "streak":
                changed["streak"] += 1
            elif kind == "indices":
                changed["active_global_indices"].pop()
            elif kind == "history":
                changed["history"][0]["metrics"]["pair_win_rate"] = .1
            elif kind == "boolean":
                changed["final_stage_gate_passed"] = 0
            else:
                changed["config"]["required_hit_at_1"] = .8
            resumed = Curriculum(self.rows, self.cfg)
            before = resumed.state_dict()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                resumed.load_state_dict(changed)
            self.assertEqual(resumed.state_dict(), before)
        for resumed in (Curriculum(self.rows, self.cfg, seed=43),
                        Curriculum(list(reversed(self.rows)), self.cfg),
                        Curriculum(self.rows, self.cfg, debug=True)):
            with self.assertRaises(ValueError):
                resumed.load_state_dict(saved)

    def test_wrong_GT_duplicate_ids_and_changed_same_case_donor_are_refused(self):
        for kind in ("GT", "id", "donor", "center"):
            changed = copy.deepcopy(self.rows)
            if kind == "GT":
                changed[0]["target"] = True
            elif kind == "id":
                changed[0]["id"] = changed[1]["id"]
            elif kind == "donor":
                changed[0]["donor_component"] = 2
            else:
                changed[0]["center"] = [-1, 0, 0]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                Curriculum(changed, self.cfg)

    def test_helper_imports_no_torch_model_or_training_modules(self):
        code = "import sys; from l0_regions.candidate_curriculum import Curriculum; assert not any(x == 'torch' or x.startswith('torch.') for x in sys.modules); assert 'l0_regions.training' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[1],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
