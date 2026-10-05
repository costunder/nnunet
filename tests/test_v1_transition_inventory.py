"""Pure metadata regression; no neural run, CT prediction or quality claim."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hiercp_v1x import transition_inventory as t


def explicit_controls():
    # These hashes identify unit-test metadata, not research measurements.
    return dict(seed=42, epochs=40, margin_mm=10,
                split_sha256=t.digest({"unit_test_split_metadata": True}),
                raw_cohort_sha256=t.digest({"unit_test_cohort_metadata": True}),
                outer_test_policy="untouched",
                initialization_policy={"declared": "fresh seed42; bind shared module initial bytes"},
                optimizer_policy={"baseline": "AdamW + cosine", "target": "AdamW constant LR"},
                physical_batch_policy={"unit": {"baseline": "source cache sample", "target": "observation row"},
                                       "candidate_work": "source-bound measured receipt required"},
                precision_policy={"baseline": "AMP plus scaler", "target": "FP32"},
                runtime_resource_policy={"selection": "explicit resource admission; no data reduction"})


def complete_groups():
    architecture = {"input", "L0", "upper"}
    inv = t.inventory()
    return {"half_A": [f["name"] for f in inv["factors"] if f["domain"] in architecture],
            "half_B": [f["name"] for f in inv["factors"] if f["domain"] not in architecture]}


def planned():
    return t.make_plan(complete_groups(), controls=explicit_controls())


class CompleteTransitionInventoryTests(unittest.TestCase):
    def test_all_input_task_objective_support_and_evaluation_axes_present(self):
        required = {"donor_condition", "physical_input_sampling", "recipient_tumor_erasure",
                    "role_shell_geometry", "GT_semantics", "training_sample_population",
                    "training_candidate_set", "comparison_corruption", "ranking_loss",
                    "observation_alignment_auxiliaries", "two_view_consistency",
                    "support_labels_population", "support_refresh", "support_episode_selection",
                    "update_schedule", "optimizer_LR_schedule", "precision_and_scaler",
                    "physical_batch_unit_and_work", "activation_storage",
                    "evaluation_candidate_universe", "checkpoint_selection"}
        self.assertTrue(required <= set(t.FACTOR_NAMES))
        self.assertEqual(len(t.FACTOR_NAMES), len(set(t.FACTOR_NAMES)))

    def test_inventory_is_copy_without_ready_or_quality_promotion(self):
        first = t.inventory()
        first["factors"][0]["target"] = "changed"
        second = t.inventory()
        self.assertNotEqual(first, second)
        self.assertFalse(second["runnable"])
        self.assertFalse(second["quality_verified"])
        self.assertFalse(second["training_started"])
        self.assertEqual(t.validate_inventory(second), second)

    def test_existing_ab_cannot_be_reported_as_full_transition(self):
        report = t.existing_coverage()
        self.assertFalse(report["all_transition_changes_covered"])
        self.assertFalse(report["native_endpoint_equivalence"])
        self.assertIn("local_encoder", report["fully_covered"])
        self.assertIn("local_readout", report["partially_covered"])
        self.assertIn("GT_semantics", report["not_covered"])
        self.assertIn("evaluation_candidate_universe", report["not_covered"])
        self.assertEqual(t.validate_coverage_claim(report), report)
        report["all_transition_changes_covered"] = True
        with self.assertRaises(ValueError):
            t.validate_coverage_claim(report)

    def test_modifying_old_bridge_coverage_is_rejected(self):
        value = t.inventory()
        value["factors"][0]["existing_half_A"] = "covered"
        with self.assertRaises(ValueError):
            t.validate_inventory(value)

    def test_complete_halves_cover_every_named_factor_once(self):
        groups = t.validate_partition(complete_groups())
        flat = groups["half_A"] + groups["half_B"]
        self.assertEqual(set(flat), set(t.FACTOR_NAMES))
        self.assertEqual(len(flat), len(t.FACTOR_NAMES))

    def test_missing_label_or_eval_axis_rejected(self):
        for axis in ("GT_semantics", "evaluation_candidate_universe"):
            groups = complete_groups()
            groups["half_B"].remove(axis)
            with self.assertRaises(ValueError):
                t.validate_partition(groups)

    def test_axis_shared_between_halves_rejected(self):
        groups = complete_groups()
        groups["half_A"].append(groups["half_B"][0])
        with self.assertRaises(ValueError):
            t.validate_partition(groups)

    def test_duplicate_inside_half_rejected(self):
        groups = complete_groups()
        groups["half_A"].append(groups["half_A"][0])
        with self.assertRaises(ValueError):
            t.validate_partition(groups)

    def test_unknown_half_and_unknown_axis_rejected(self):
        groups = complete_groups()
        groups["half_B"].append("pretend_all_remaining_changes")
        with self.assertRaises(ValueError):
            t.validate_partition(groups)
        groups = complete_groups()
        groups["combined"] = []
        with self.assertRaises(ValueError):
            t.validate_partition(groups)

    def test_plan_declares_changed_gt_and_nonmonotonic_interactions(self):
        plan = planned()
        self.assertFalse(plan["same_GT"])
        self.assertNotEqual(plan["baseline_task"], plan["target_task"])
        self.assertIn("not monotonic", plan["interaction_assumption"])
        self.assertEqual(plan["runnable"], {"half_A": False, "half_B": False})
        self.assertEqual(t.validate_plan(plan), plan)

    def test_changed_gt_cannot_be_claimed_identical(self):
        plan = planned()
        plan["same_GT"] = True
        # Rehashing the forged declaration does not restore correctness.
        plan.pop("contract_sha256")
        plan["contract_sha256"] = t.digest(plan)
        with self.assertRaises(ValueError):
            t.validate_plan(plan)

    def test_metadata_cannot_set_runnable_true(self):
        plan = planned()
        plan["runnable"]["half_A"] = True
        with self.assertRaises(ValueError):
            t.validate_plan(plan)

    def test_scope_seed_epochs_are_not_silently_reduced(self):
        for key, value in (("seed", 43), ("epochs", 1), ("margin_mm", 3)):
            controls = explicit_controls()
            controls[key] = value
            with self.assertRaises(ValueError):
                t.make_plan(complete_groups(), controls=controls)

    def test_full_control_fields_required(self):
        for key in t.CONTROL_FIELDS:
            controls = explicit_controls()
            del controls[key]
            with self.assertRaises(ValueError):
                t.validate_controls(controls)

    def test_physical_batch_numbers_without_work_unit_rejected(self):
        controls = explicit_controls()
        controls["physical_batch_policy"] = {"batch_size": 32}
        with self.assertRaises(ValueError):
            t.validate_controls(controls)

    def test_bad_cohort_hash_and_outer_test_consumption_rejected(self):
        controls = explicit_controls()
        controls["raw_cohort_sha256"] = "unknown"
        with self.assertRaises(ValueError):
            t.validate_controls(controls)
        controls = explicit_controls()
        controls["outer_test_policy"] = "use_for_best_selection"
        with self.assertRaises(ValueError):
            t.validate_controls(controls)

    def test_source_bindings_capture_actual_bytes_and_detect_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "unit_contract.py"
            source.write_bytes(b"metadata_contract = 1\n")
            with patch.object(t, "SOURCE_FILES", ("unit_contract.py",)):
                binding = t.source_bindings(root)
                self.assertEqual(binding["unit_contract.py"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
                self.assertEqual(t.validate_source_bindings(root, binding), binding)
                source.write_bytes(b"metadata_contract = 2\n")
                with self.assertRaises(ValueError):
                    t.validate_source_bindings(root, binding)

    def test_missing_actual_source_is_not_replaced_with_placeholder(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(t, "SOURCE_FILES", ("missing.py",)):
                with self.assertRaises(FileNotFoundError):
                    t.source_bindings(directory)

    def test_original_archive_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "versions" / "v1" / "pipeline_v1_source.zip"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"explicit corrupted byte-contract test fixture")
            with patch.object(t, "SOURCE_FILES", ("versions/v1/pipeline_v1_source.zip",)):
                with self.assertRaises(ValueError):
                    t.source_bindings(root)

    def test_source_sha_alone_does_not_register_smoke(self):
        with self.assertRaises(ValueError):
            t.register_cuda_smoke(planned(), "half_A", root=".",
                                  recipe={"source_bindings": {}}, receipt={}, artifact="missing.json")

    def test_semantics_do_not_relabel_observations_as_cp_suitability(self):
        semantics = t.inventory()["target_semantics"]
        self.assertFalse(semantics["P_is_donor_compatibility_GT"])
        self.assertFalse(semantics["U_is_CP_ineligible_GT"])

    def test_plan_digest_binds_control_policy(self):
        plan = planned()
        plan["controls"]["precision_policy"]["target"] = "unreported_half_precision"
        with self.assertRaises(ValueError):
            t.validate_plan(plan)

    def test_runtime_factors_are_included_in_exact_partition(self):
        groups = complete_groups()
        for axis in ("precision_and_scaler", "physical_batch_unit_and_work", "activation_storage"):
            self.assertIn(axis, groups["half_B"])
        self.assertEqual(len(t.validate_partition(groups)["half_A"]) + len(groups["half_B"]), len(t.FACTOR_NAMES))


if __name__ == "__main__":
    unittest.main()
