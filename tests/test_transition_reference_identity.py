"""Stdlib UNIT checks of exact crossed contracts and advancing reference evidence."""
from __future__ import annotations

import copy
import json
import unittest

from hiercp_v1x import transition_reference_identity as binding


def reference(epoch=2, cursor=3, phase="optimization"):
    steps = 10
    metadata = {name: dict(path="/native/" + name + ".json", bytes=100, sha256="a" * 64)
                for name in ("experiment", "inventory", "execution_contract", "learning_schedule")}
    return dict(format=binding.REFERENCE_FORMAT, admitted=True, debug=False,
        inventory_sha256="a" * 64, checkpoint="/native/attempts/0001/checkpoint_latest.pt",
        checkpoint_sha256="b" * 64, checkpoint_bytes=4096, weights_transferred=False,
        tensor_values_exported=False, neural_execution=False,
        validation_scope="recorded metadata and file identity only; no predictive quality claim",
        checkpoint_content_hash_recomputed=False, checkpoint_recorded_content_sha256="c" * 64,
        target_epochs=40, optimization_steps_per_epoch=steps,
        saved_cursor=dict(epoch=epoch, step=epoch * steps + cursor, phase=phase, next_batch=cursor, batch=32),
        optimizer_epochs_completed=epoch + int(phase in ("optimization", "refresh_memory", "validation") and cursor == steps),
        validated_epochs_completed=epoch, native_training_complete=phase == "complete",
        recorded_source_files=90, checks=dict(inventory_matches=True, saved_training_cursor_valid=True,
            **({"optimization_epoch_below_target": True} if phase == "optimization" else {})),
        required_fields_missing=[], failed_bindings=[], experiment_sha256="a" * 64, metadata_files=metadata)


def manifest(value=None):
    return dict(format="v17_crossed_training_identity_v1", arm="D", debug=False,
        native_experiment_binding=reference() if value is None else value,
        native_inventory_sha256="a" * 64, scope="d" * 64,
        settings=dict(workers=16, physical_batch_candidates=[32], cuda_gib=40, rss_gib=192),
        hardware=dict(name="RTX A6000", total_memory=48 * 2**30),
        sources={"original.py": "e" * 64}, original_curriculum_binding=dict(candidate_pool=128))


class ReferenceIdentityUnit(unittest.TestCase):
    def compare(self, old, new):
        return binding.compare_reference(old, new)

    def test_complete_valid_production_receipt_is_exact(self):
        result = self.compare(reference(), reference())
        self.assertTrue(result["admissible"])
        self.assertTrue(result["exact"])
        self.assertEqual(result["differences"], [])

    def test_current_checkpoint_bytes_and_cursor_can_advance(self):
        old, new = reference(), reference(epoch=7, cursor=5)
        new.update(checkpoint_sha256="f" * 64, checkpoint_recorded_content_sha256="0" * 64,
                   checkpoint_bytes=9000)
        result = self.compare(old, new)
        self.assertTrue(result["admissible"], result)
        self.assertFalse(result["exact"])
        self.assertIn("saved_cursor.step", {row["path"] for row in result["differences"]})

    def test_phase_specific_true_checks_can_change(self):
        old = reference()
        new = reference(epoch=3, cursor=10, phase="validation")
        new["checks"]["post_optimization_cursor_complete"] = True
        new["checks"]["post_optimization_epoch_below_target"] = True
        self.assertTrue(self.compare(old, new)["admissible"])

    def test_native_completion_only_changes_endpoint(self):
        old, new = reference(), reference(epoch=40, cursor=0, phase="complete")
        new["checks"].update(final_phase_completed_epochs=True, final_phase_query_cursor_reset=True)
        self.assertTrue(self.compare(old, new)["admissible"])

    def test_native_initial_phases_can_enter_optimization(self):
        validation = reference(epoch=0, cursor=0, phase="initial_validation")
        memory = reference(epoch=0, cursor=0, phase="initial_memory")
        optimization = reference(epoch=0, cursor=0, phase="optimization")
        self.assertTrue(self.compare(validation, optimization)["admissible"])
        self.assertTrue(self.compare(memory, optimization)["admissible"])
        self.assertFalse(self.compare(memory, validation)["admissible"])

    def test_all_fixed_metadata_files_and_paths_remain_exact(self):
        for name in ("experiment", "inventory", "execution_contract", "learning_schedule"):
            for field, value in (("path", "/another/file.json"), ("bytes", 101), ("sha256", "1" * 64)):
                with self.subTest(name=name, field=field):
                    old, new = reference(), reference()
                    new["metadata_files"][name][field] = value
                    self.assertFalse(self.compare(old, new)["admissible"])

    def test_experiment_hash_change_is_not_a_progress_exception(self):
        new = reference()
        new["experiment_sha256"] = "1" * 64
        new["metadata_files"]["experiment"]["sha256"] = "1" * 64
        self.assertFalse(self.compare(reference(), new)["admissible"])

    def test_inventory_and_native_fixed_recipe_remain_exact(self):
        for field, value in (("inventory_sha256", "1" * 64), ("target_epochs", 39),
                             ("optimization_steps_per_epoch", 11), ("recorded_source_files", 91),
                             ("checkpoint", "/native/attempts/0002/checkpoint_latest.pt")):
            with self.subTest(field=field):
                new = reference()
                new[field] = value
                self.assertFalse(self.compare(reference(), new)["admissible"])

    def test_unknown_reference_fields_need_exact_equality(self):
        old, new = reference(), reference()
        old["future_contract"] = new["future_contract"] = {"value": 5}
        self.assertTrue(self.compare(old, new)["admissible"])
        new["future_contract"] = {"value": 6}
        self.assertFalse(self.compare(old, new)["admissible"])
        self.assertFalse(self.compare(reference(), new)["admissible"])

    def test_invalid_admission_cannot_be_hidden_by_exact_equality(self):
        for field, value in (("admitted", False), ("debug", 0), ("weights_transferred", True),
                             ("neural_execution", True), ("tensor_values_exported", True),
                             ("checkpoint_content_hash_recomputed", True),
                             ("required_fields_missing", ["source"]), ("failed_bindings", ["GT"]),
                             ("checkpoint_bytes", True), ("checkpoint_sha256", "invalid")):
            with self.subTest(field=field):
                invalid = reference()
                invalid[field] = value
                self.assertFalse(self.compare(invalid, copy.deepcopy(invalid))["admissible"])

    def test_every_check_must_be_strictly_true_and_phase_unknown_keys_are_exact(self):
        for passed in (False, 1, None):
            new = reference()
            new["checks"]["post_optimization_cursor_complete"] = passed
            self.assertFalse(self.compare(reference(), new)["admissible"])
        new = reference()
        new["checks"]["unrecognized_phase_check"] = True
        self.assertFalse(self.compare(reference(), new)["admissible"])
        old = reference()
        del old["checks"]["inventory_matches"]
        self.assertFalse(self.compare(old, reference())["admissible"])

    def test_cursor_and_epoch_counts_are_internally_consistent(self):
        for key, value in (("step", 999), ("next_batch", 11), ("epoch", 41),
                           ("phase", "unknown"), ("phase", []), ("batch", 1), ("batch", True)):
            with self.subTest(key=key, value=value):
                new = reference()
                new["saved_cursor"][key] = value
                self.assertFalse(self.compare(reference(), new)["admissible"])
        for field, value in (("optimizer_epochs_completed", 1), ("validated_epochs_completed", 1),
                             ("native_training_complete", True)):
            new = reference()
            new[field] = value
            self.assertFalse(self.compare(reference(), new)["admissible"])

    def test_native_physical_batch_and_forward_progress_remain_fixed(self):
        new = reference(epoch=3, cursor=1)
        new["saved_cursor"]["batch"] = 64
        self.assertFalse(self.compare(reference(), new)["admissible"])
        self.assertFalse(self.compare(reference(epoch=4), reference(epoch=3))["admissible"])
        self.assertFalse(self.compare(reference(cursor=5), reference(cursor=2))["admissible"])
        old = reference(epoch=2, cursor=10, phase="validation")
        new = reference(epoch=2, cursor=10, phase="optimization")
        self.assertFalse(self.compare(old, new)["admissible"])
        self.assertTrue(self.compare(old, reference(epoch=3, cursor=0))["admissible"])

    def test_initialization_and_final_memory_phase_progress(self):
        old = reference(epoch=0, cursor=0, phase="initial_validation")
        new = reference(epoch=0, cursor=0, phase="optimization")
        for item in (old, new):
            item["checks"]["initial_phase_before_updates"] = True
        self.assertTrue(self.compare(old, new)["admissible"])
        self.assertTrue(self.compare(new, reference(epoch=0, cursor=0))["admissible"])
        old = reference(epoch=40, cursor=0, phase="final_memory")
        new = reference(epoch=40, cursor=0, phase="complete")
        for item in (old, new):
            item["checks"].update(final_phase_completed_epochs=True, final_phase_query_cursor_reset=True)
        self.assertTrue(self.compare(old, new)["admissible"])

    def test_debug_reference_is_exact_without_production_defaults(self):
        old = dict(debug=True, inventory_sha256="a" * 64, weights_transferred=False)
        self.assertTrue(self.compare(old, copy.deepcopy(old))["admissible"])
        new = {**old, "inventory_sha256": "b" * 64}
        self.assertFalse(self.compare(old, new)["admissible"])

    def test_manifest_reference_freezes_previous_endpoint_without_mutation(self):
        previous, current = manifest(), manifest(reference(epoch=4, cursor=2))
        before, after = copy.deepcopy(previous), copy.deepcopy(current)
        bound = binding.bind_manifest_reference(previous, current)
        self.assertEqual(bound, previous)
        self.assertIsNot(bound, previous)
        self.assertEqual(previous, before)
        self.assertEqual(current, after)
        bound["settings"]["workers"] = 1
        self.assertEqual(previous["settings"]["workers"], 16)

    def test_no_other_manifest_hardware_settings_sources_or_geometry_exception(self):
        for field, value in (("scope", "0" * 64), ("settings", {"workers": 2}),
                             ("hardware", {"name": "A100"}), ("sources", {"original.py": "0" * 64}),
                             ("original_curriculum_binding", {"candidate_pool": 8}), ("arm", "C")):
            with self.subTest(field=field):
                new = manifest()
                new[field] = value
                with self.assertRaisesRegex(ValueError, "Crossed manifest identity differs"):
                    binding.bind_manifest_reference(manifest(), new)

    def test_all_changed_fields_are_retained_in_machine_readable_error(self):
        old, new = manifest(), manifest()
        new["settings"]["workers"] = 8
        new["hardware"]["name"] = "A100"
        new["extra"] = "x" * 12000
        with self.assertRaises(ValueError) as caught:
            binding.bind_manifest_reference(old, new)
        detail = json.loads(str(caught.exception).split(": ", 1)[1])
        differences = detail["manifest_differences"]
        self.assertEqual({row["path"] for row in differences}, {"settings.workers", "hardware.name", "extra"})
        self.assertEqual(next(row for row in differences if row["path"] == "extra")["new"], "x" * 12000)

    def test_differences_distinguish_types_null_missing_and_list_items(self):
        rows = binding.structured_differences({"x": True, "y": None, "a": [1, 2]}, {"x": 1, "a": [1, 3]})
        self.assertEqual({row["path"] for row in rows}, {"x", "y", "a[1]"})
        y = next(row for row in rows if row["path"] == "y")
        self.assertTrue(y["old_present"])
        self.assertFalse(y["new_present"])


if __name__ == "__main__":
    unittest.main()
