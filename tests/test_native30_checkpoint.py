"""Legacy native30 metadata/source UNIT checks, never CPU neural evidence."""
from contextlib import contextmanager
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from hiercp_v1x import native30_checkpoint as native


def fixture():
    return dict(method="hiercp-full", framework="torch_geometric", training_complete=True,
                completed_epoch=40, target_epochs=40, epoch=30, best_epoch=30,
                model_kwargs=dict(hidden_dim=128, heads=4, local_layers=3, patient_layers=2,
                    prototype_layers=2, dropout=.1, dense_base_channels=12, dense_feature_dim=32,
                    dense_batch_size=4, channels_last_3d=True, checkpoint_local_blocks=True,
                    checkpoint_dense_encoder=True),
                graph_config=dict(adaptive_roi_margin_mm=30., context_outer_radius_mm=28.,
                    context_radius_mm=28., context_shells_mm=[4., 12., 28.], patch_size=48,
                    canonical_full_graph=True), ct_clip=(-200., 250.),
                state_dict={"metadata_UNIT_only": torch.tensor([1.])},
                training_signature={"seed": 42, "run_mode": "production"},
                prototype_training_cases=["liver_" + str(number) for number in range(105)],
                prototype_fingerprint="ab" * 32, best_selection={"mrr": 1., "acc": 1., "margin": 7.})


@contextmanager
def isolated_source():
    """A Python-file identity fixture only; no dummy model is run or saved."""
    saved_active, saved_path = native._ACTIVE, list(sys.path)
    previous = {name: module for name, module in sys.modules.items()
                if name == "hiercp" or name.startswith("hiercp.")}
    for name in previous:
        sys.modules.pop(name)
    native._ACTIVE = None
    try:
        with tempfile.TemporaryDirectory(prefix="native30_source_UNIT_") as directory:
            root = Path(directory)
            (root / "hiercp").mkdir()
            for name in (*native._SOURCE_REQUIRED, "model.py"):
                (root / "hiercp" / name).write_text("# Source identity UNIT fixture only\n", encoding="utf8")
            (root / "config").mkdir()
            (root / "config/train.json").write_text(json.dumps({"training": {"amp": True}}), encoding="utf8")
            yield root
    finally:
        for name in list(sys.modules):
            if name == "hiercp" or name.startswith("hiercp."):
                sys.modules.pop(name)
        sys.modules.update(previous)
        native._ACTIVE = saved_active
        sys.path[:] = saved_path
        importlib.invalidate_caches()


class Native30MetadataTests(unittest.TestCase):
    def test_real_native_scope_metadata_unknown_architecture_is_not_invalid(self):
        value = fixture()
        result = native.validate_native30_metadata(value, "/historical/full/model.pt")
        self.assertEqual(result["selected_epoch"], 30)
        self.assertEqual(result["completed_epochs"], 40)
        self.assertEqual(result["saved_architecture_version"], "UNKNOWN")
        self.assertEqual(result["source_provenance_at_training"], "UNKNOWN")
        self.assertEqual(len(result["prototype_training_cases"]), 105)
        self.assertFalse(result["training_started"])
        self.assertFalse(result["strict_loading_is_training_source_equivalence_proof"])
        result["selection"]["mrr"] = 0.
        self.assertEqual(value["best_selection"]["mrr"], 1.)

    def test_other_saved_best_epoch_is_kept(self):
        value = fixture()
        value["epoch"] = value["best_epoch"] = 24
        self.assertEqual(native.validate_native30_metadata(value, "model.pt")["selected_epoch"], 24)

    def test_legacy_missing_signature_and_fields_remain_unknown(self):
        for signature in (None, {}, {"format": "legacy"}, {"seed": None, "run_mode": None}):
            with self.subTest(signature=signature):
                value = fixture()
                value["training_signature"] = signature
                result = native.validate_native30_metadata(value, "model.pt")
                self.assertEqual(result["training_seed"], "UNKNOWN")
                self.assertEqual(result["training_seed_origin"], "UNKNOWN")
                self.assertEqual(result["training_seed_evidence"], {})
                self.assertEqual(result["training_run_mode"], "UNKNOWN")
                self.assertNotIn("seed", result)
        value.pop("training_signature")
        self.assertFalse(native.validate_native30_metadata(value, "model.pt")["training_signature_present"])

    def test_known_historical_seed_is_retained_without_becoming_evaluation_seed(self):
        for signature, top_seed, expected in ((None, 43, 43), ({"seed": 41}, None, 41),
                                              ({"seed": 42}, 42, 42)):
            value = fixture()
            value["training_signature"] = signature
            if top_seed is not None:
                value["seed"] = top_seed
            result = native.validate_native30_metadata(value, "model.pt")
            self.assertEqual(result["training_seed"], expected)
            self.assertTrue(result["training_seed_evidence"])

    def test_malformed_or_conflicting_historical_evidence_rejected(self):
        for patch, message in (({"training_signature": []}, "must be a mapping"),
                               ({"training_signature": {"seed": True}}, "must be an integer"),
                               ({"training_signature": {"seed": "42"}}, "must be an integer"),
                               ({"seed": 43}, "Conflicting saved"),
                               ({"training_signature": {"run_mode": "debug"}}, "non-production"),
                               ({"run_mode": "benchmark"}, "non-production"),
                               ({"training_signature": {"ablation_mode": "local_only"}}, "ablation")):
            with self.subTest(patch=patch):
                value = fixture()
                value.update(patch)
                with self.assertRaisesRegex(ValueError, message):
                    native.validate_native30_metadata(value, "model.pt")

    def test_missing_full_completion_or_different_selected_epoch_rejected(self):
        for field, bad in (("training_complete", False), ("completed_epoch", 39),
                           ("target_epochs", 1), ("best_epoch", 24), ("epoch", True),
                           ("method", "other"), ("framework", "other")):
            with self.subTest(field=field):
                value = fixture()
                value[field] = bad
                with self.assertRaisesRegex(ValueError, "complete40"):
                    native.validate_native30_metadata(value, "model.pt")

    def test_last_is_not_selected_best(self):
        for path in ("model.last.pt", "checkpoint_latest.pt", "model.pt.last.pt"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "LAST/latest"):
                native.validate_native30_metadata(fixture(), path)
        value = fixture()
        value["format"] = "hiercp_training_state_v1"
        with self.assertRaisesRegex(ValueError, "complete40"):
            native.validate_native30_metadata(value, "model.pt")

    def test_bounded30_is_not_original_native30(self):
        for field, bad in (("context_outer_radius_mm", 30.), ("adaptive_roi_margin_mm", 10.),
                           ("context_shells_mm", [4, 12, 30]), ("patch_size", 32)):
            value = fixture()
            value["graph_config"][field] = bad
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "ROI30/context28"):
                native.validate_native30_metadata(value, "model.pt")

    def test_empty_or_nonfinite_weights_rejected(self):
        for state in ({}, {"key": "no actual tensor"}, {"key": torch.tensor([float("nan")])}):
            value = fixture()
            value["state_dict"] = state
            with self.assertRaisesRegex(ValueError, "state|tensor"):
                native.validate_native30_metadata(value, "model.pt")

    def test_bounded_or_half_marker_cannot_enter_native_checkpoint(self):
        value = fixture()
        value["state_dict"]["v1x_bounded_scope_digest"] = torch.zeros(32, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "Bounded/half"):
            native.validate_native30_metadata(value, "model.pt")

    def test_actual_prototype_identity_and_clip_required(self):
        for field, bad in (("prototype_training_cases", []), ("prototype_training_cases", ["liver_1", "liver_1"]),
                           ("prototype_fingerprint", "missing"), ("ct_clip", [-100, 250])):
            value = fixture()
            value[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                native.validate_native30_metadata(value, "model.pt")

    def test_missing_saved_model_kwargs_not_filled_from_current_defaults(self):
        value = fixture()
        value["model_kwargs"].pop("dense_feature_dim")
        with self.assertRaisesRegex(ValueError, "Complete saved"):
            native.validate_native30_metadata(value, "model.pt")

    def test_actual_tensor_serialization_does_not_manufacture_trained_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="native30_tensor_UNIT_") as directory:
            path = Path(directory) / "UNIT_untrained_tensor_only.pt"
            value = {"format": "UNIT_untrained_tensor_only", "trained_weights": False,
                     "state_dict": {"UNIT_tensor": torch.tensor([1., 2.])}}
            torch.save(value, path)
            before = native.sha(path)
            loaded = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            self.assertTrue(torch.equal(loaded["state_dict"]["UNIT_tensor"], value["state_dict"]["UNIT_tensor"]))
            self.assertEqual(before, native.sha(path))
            with self.assertRaisesRegex(ValueError, "complete40"):
                native.validate_native30_metadata(loaded, "model.pt")

    def test_both_neural_entry_points_reject_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "actual CUDA"):
            native.load_native30("unused/model.pt", "unused/prototype.npz", "unused/source", lambda: None,
                                 device="cpu")
        with self.assertRaisesRegex(RuntimeError, "actual CUDA"):
            native.load_debug_native30("unused/source", {}, "unused/prototype.npz", lambda: None,
                                       device="cpu")


class Native30SourceIdentityTests(unittest.TestCase):
    def test_source_activation_records_actual_files_without_archive_or_scope_patch(self):
        with isolated_source() as root:
            proof = native.activate_native30_source(root)
            importlib.import_module("hiercp.model")
            after = native.verify_native30_source(proof)
            self.assertEqual(after["source"], str(root.resolve()))
            self.assertIn("hiercp.model", after["imported_source_modules"])
            self.assertFalse(after["archive_equivalence_asserted"])
            self.assertFalse(after["bounded_scope_installed"])
            self.assertTrue(after["source_preserved"])

    def test_source_edit_or_new_importable_file_rejected(self):
        for added in (False, True):
            with self.subTest(added=added), isolated_source() as root:
                proof = native.activate_native30_source(root)
                filename = "unexpected.py" if added else "local.py"
                (root / "hiercp" / filename).write_text("# changed UNIT source\n", encoding="utf8")
                with self.assertRaisesRegex(ValueError, "bytes changed"):
                    native.verify_native30_source(proof)

    def test_source_config_change_rejected(self):
        with isolated_source() as root:
            proof = native.activate_native30_source(root)
            (root / "config/train.json").write_text('{"training":{"amp":false}}', encoding="utf8")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                native.verify_native30_source(proof)

    def test_another_source_proof_and_already_imported_package_rejected(self):
        with isolated_source() as root:
            sys.path.insert(0, str(root))
            importlib.import_module("hiercp")
            with self.assertRaisesRegex(RuntimeError, "before every hiercp import"):
                native.activate_native30_source(root)
            sys.modules.pop("hiercp")
            proof = native.activate_native30_source(root)
            wrong = copy.deepcopy(proof)
            wrong["source"] = str(root.parent)
            with self.assertRaisesRegex(ValueError, "another source"):
                native.verify_native30_source(wrong)

    def test_imported_module_escape_not_accepted(self):
        with isolated_source() as root:
            native.activate_native30_source(root)
            import types
            escaped = types.ModuleType("hiercp.injected")
            escaped.__file__ = __file__
            sys.modules[escaped.__name__] = escaped
            with self.assertRaisesRegex(ValueError, "another source"):
                native.verify_native30_source()


class Native30EvaluationSettingsTests(unittest.TestCase):
    def source_config(self):
        # Explicit execution-setting UNIT fixture; no neural run or checkpoint.
        return {"seed": 42, "training": {"amp": True}, "cache": {"source_pad": 4}}

    def test_missing_signature_production_config_has_no_second_keyerror(self):
        for signature in (None, {}, {"seed": 43}, {"run_mode": "production"}):
            value = fixture()
            value["training_signature"] = signature
            before = copy.deepcopy(value)
            receipt = native.validate_native30_metadata(value, "model.pt")
            config, settings = native._evaluation_configuration(value, self.source_config())
            self.assertEqual(config["seed"], 42)
            self.assertEqual(settings["evaluation_seed"], 42)
            self.assertFalse(settings["evaluation_seed_is_training_provenance"])
            self.assertEqual(receipt["training_seed"], 43 if signature == {"seed": 43} else "UNKNOWN")
            self.assertEqual(settings["source_pad_at_training"], "UNKNOWN")
            self.assertEqual(value["model_kwargs"], before["model_kwargs"])
            self.assertEqual(value["graph_config"], before["graph_config"])
            self.assertTrue(torch.equal(value["state_dict"]["metadata_UNIT_only"], before["state_dict"]["metadata_UNIT_only"]))
            config["model"]["hidden_dim"] = -1
            config["graph"]["adaptive_roi_margin_mm"] = -1
            self.assertEqual(value["model_kwargs"], before["model_kwargs"])
            self.assertEqual(value["graph_config"], before["graph_config"])

    def test_signature_entirely_absent_uses_explicit_execution_settings(self):
        value = fixture()
        value.pop("training_signature")
        config, settings = native._evaluation_configuration(value, self.source_config())
        self.assertEqual(config["cache"]["source_pad"], 4)
        self.assertEqual(settings["source_pad_at_training"], "UNKNOWN")

    def test_explicit_saved_padding_retained_and_mismatch_rejected(self):
        value = fixture()
        value["training_signature"]["cache"] = {"source_pad": 4}
        _, settings = native._evaluation_configuration(value, self.source_config())
        self.assertEqual(settings["source_pad_at_training"], 4)
        for saved in ({"source_pad": 5}, {"source_pad": True}, {"source_pad": -1}, []):
            value["training_signature"]["cache"] = saved
            with self.subTest(saved=saved), self.assertRaisesRegex(ValueError, "padding|cache"):
                native._evaluation_configuration(value, self.source_config())

    def test_no_silent_evaluation_defaults(self):
        for field, bad in (("seed", None), ("seed", True), ("seed", 43),
                           ("training", {}), ("training", {"amp": 1}),
                           ("cache", {}), ("cache", {"source_pad": True})):
            source = self.source_config()
            source[field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                native._evaluation_configuration(fixture(), source)


if __name__ == "__main__":
    unittest.main()
