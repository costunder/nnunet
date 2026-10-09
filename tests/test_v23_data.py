"""UNIT metadata fixtures only; no real CT, CUDA or quality evidence."""
from __future__ import annotations

import copy
from dataclasses import replace
import unittest

from hiercp_v1x.native30_data_contract import assignment
from hiercp_v1x.transition_evaluation import QUERY_FIELDS
from hiercp_v1x.v23_data import V23Population, cumulative_u_counts, ZERO_P_TRAIN_POLICY


def unit_inventory(*, production=False):
    counts = {"train_a": 2, "train_b": 3, "train_zero": 0, "val_a": 4, "val_zero": 0}
    train, val, excluded = ["train_a", "train_b", "train_zero"], ["val_a", "val_zero"], ["outer_unused"]
    if production:
        train, val, excluded = ([f"train_{i}" for i in range(84)], [f"val_{i}" for i in range(21)],
                                [f"outer_{i}" for i in range(26)])
        counts = {case: 2 for case in train + val}
    records, raw = [], []
    for case, count in counts.items():
        positives = [dict(component=i + 1, center=[i, 0, 0]) for i in range(count)]
        centers = [[i, 10, 0] for i in range(128)]
        raw.append(dict(case_id=case, positives=positives, comparison=dict(centers=centers),
                        image_sha256="a" * 64, label_sha256="b" * 64))
        for positive in positives:
            records.append(dict(id=f'{case}:P:{positive["component"]}', case_id=case,
                                patient_group="group:" + case, component=positive["component"],
                                center=positive["center"], target=1))
        for i, center in enumerate(centers):
            records.append(dict(id=f"{case}:U:{i}", case_id=case, patient_group="group:" + case,
                                component=None, center=center, target=0))
    meta = dict(format="native_local_cnn_inventory_v1", complete=True, debug=not production,
                learning_policy="same_donor_live_v1", config=dict(seed=42),
                split=dict(inner_train=train, inner_val=val, outer_train=train + val, outer_val=excluded),
                identities=dict(cases={case: dict(patient_group="group:" + case)
                                       for case in train + val + excluded}),
                donor_pool=[dict(case_id=case, component_id=1) for case in train[:2]],
                raw_records=raw, records=sorted(records, key=lambda row: row["id"]),
                UNIT_synthetic_metadata_only=True)
    meta["records"] = assignment(meta, 42)
    return meta


class V23DataUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = unit_inventory()

    def setUp(self):
        self.population = V23Population(self.meta, debug=True)

    def test_all_observed_components_active_at_every_cumulative_stage(self):
        previous = set()
        for count in cumulative_u_counts():
            plan = self.population.case("train_b", count)
            self.assertEqual(plan.partition, "inner_train")
            self.assertEqual(plan.observed_P, 3)
            self.assertEqual(len(plan.unobserved_indices), count)
            self.assertEqual({plan.rows[i]["component"] for i in plan.positive_indices}, {1, 2, 3})
            self.assertTrue(previous <= set(plan.record_ids))
            self.assertTrue(all(plan.rows[i]["target"] == 0 for i in plan.unobserved_indices))
            previous = set(plan.record_ids)
        self.assertEqual(previous, {row["id"] for row in self.meta["records"] if row["case_id"] == "train_b"})

    def test_bank_prefix_uses_raw_geometry_order_not_lexical_ID_order(self):
        plan = self.population.case("train_a", 7)
        self.assertEqual(set(plan.record_ids), {"train_a:P:1", "train_a:P:2"}
                         | {f"train_a:U:{i}" for i in range(7)})
        self.assertNotIn("train_a:U:10", plan.record_ids)
        self.assertEqual(plan.unobserved_bank_positions, tuple(range(7)))
        full = self.population.case("train_a")
        expected = [row["id"] for row in self.meta["records"] if row["case_id"] == "train_a"]
        self.assertEqual(list(full.record_ids), expected)
        self.assertNotEqual(full.unobserved_bank_positions, tuple(range(128)))

    def test_query_schema_removes_all_GT_and_retains_fixed_donor(self):
        donors = set()
        for count in cumulative_u_counts():
            plan = self.population.case("val_a", count)
            self.assertEqual(plan.partition, "inner_val")
            self.assertTrue(all(tuple(row) == QUERY_FIELDS for row in plan.query_rows))
            self.assertTrue(all("target" not in row and "component" not in row for row in plan.query_rows))
            donors.add((plan.donor_case_id, plan.donor_component, plan.donor_group))
            self.assertIn(plan.donor_case_id, self.meta["split"]["inner_train"])
            self.assertNotEqual(plan.patient_group, plan.donor_group)
        self.assertEqual(len(donors), 1)

    def test_zero_P_train_policy_explicit_validation_keeps_zero_P_case(self):
        manifest = self.population.manifest()
        self.assertEqual(manifest["zero_P_train_policy"], ZERO_P_TRAIN_POLICY)
        self.assertEqual(manifest["zero_P_train_cases"], ["train_zero"])
        self.assertEqual(self.population.partition_cases("inner_train", ranking_only=True), ("train_a", "train_b"))
        self.assertIn("train_zero", self.population.partition_cases("inner_train"))
        self.assertIn("val_zero", self.population.partition_cases("inner_val"))
        zero = self.population.case("val_zero", 7)
        self.assertEqual(zero.positive_indices, ())
        self.assertEqual(len(zero.unobserved_indices), 7)
        self.assertEqual(self.population.case("train_zero").observed_P, 0)
        self.assertTrue(manifest["validation_zero_P_cases_scored"])
        with self.assertRaisesRegex(ValueError, "validation retains every case"):
            self.population.partition_cases("inner_val", ranking_only=True)

    def test_provider_indices_match_exact_original_candidate_order(self):
        rows = [row for row in self.meta["records"] if row["case_id"] in self.meta["split"]["inner_train"]]
        rows.reverse()
        plan = self.population.case("train_a", 14)
        positions = self.population.indices_for(plan, rows)
        self.assertEqual([rows[i]["id"] for i in positions], list(plan.record_ids))
        changed = copy.deepcopy(rows)
        changed[positions[0]]["donor_component"] += 1
        with self.assertRaises(ValueError):
            self.population.indices_for(plan, changed)
        missing = copy.deepcopy(rows)
        del missing[positions[0]]
        with self.assertRaises(ValueError):
            self.population.indices_for(plan, missing)

    def test_copies_and_plan_cannot_mutate_signed_population(self):
        plan = self.population.case("train_a", 7)
        before = self.population.manifest()
        plan.rows[0]["target"] = 0
        plan.query_rows[0]["center"][0] = 999
        self.population.rows[0]["center"][0] = 999
        self.assertEqual(self.population.manifest(), before)
        self.assertEqual(plan.query_rows, self.population.case("train_a", 7).query_rows)
        bad = copy.deepcopy(plan._rows)
        bad[0]["target"] = 0
        with self.assertRaises(ValueError):
            self.population.indices_for(replace(plan, _rows=bad), self.population.rows)
        with self.assertRaises(TypeError):
            self.population.by_case["new"] = ()

    def test_changed_observed_donor_or_train_only_assignment_rejected(self):
        for donor in ("val_a", "train_zero"):
            changed = copy.deepcopy(self.meta)
            for row in changed["records"]:
                if row["case_id"] == "train_a":
                    row.update(donor_case_id=donor, donor_component=1, donor_group="group:" + donor)
            with self.subTest(donor=donor), self.assertRaises(ValueError):
                V23Population(changed, debug=True)
        changed = copy.deepcopy(self.meta)
        del changed["records"][next(i for i, row in enumerate(changed["records"]) if row["target"])]
        with self.assertRaises(ValueError):
            V23Population(changed, debug=True)

    def test_small_metadata_fixture_cannot_be_used_as_production(self):
        with self.assertRaises(ValueError):
            V23Population(self.meta)
        for value in (0, 129, True, 1.5):
            with self.subTest(active=value), self.assertRaises(ValueError):
                self.population.case("train_a", value)
        with self.assertRaises(ValueError):
            self.population.case("outer_unused", 7)

    def test_original_case_count_alone_does_not_satisfy_all_P_production_contract(self):
        with self.assertRaisesRegex(ValueError, "P527"):
            V23Population(unit_inventory(production=True))

    def test_stage_manifest_binds_all_P_and_actual_subset_forward(self):
        first = self.population.stage_manifest(7)
        second = self.population.stage_manifest(14)
        self.assertEqual(sum(row["observed_P"] for row in first["cases"]), 9)
        self.assertEqual(sum(row["observed_P"] for row in second["cases"]), 9)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["population_sha256"], second["population_sha256"])
        self.assertTrue(first["joint_upper_subset_forward_required"])
        self.assertFalse(first["full_score_slicing_for_stage_validation"])
        self.assertEqual(cumulative_u_counts()[-2:], (126, 128))
        for bad in (0, True, -1, 129):
            with self.subTest(initial=bad), self.assertRaises(ValueError):
                cumulative_u_counts(initial=bad)


if __name__ == "__main__":
    unittest.main()
