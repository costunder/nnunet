"""DEBUG-only publication/reuse regressions; no medical data or training.

All .pt/.pth/.nii.gz-labelled files are explicit temporary byte fixtures for
hash verification, never deserialized or represented as trained predictions.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools import downstream_level_ablation as d


class DownstreamReuseFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="debug_bank_reuse_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        paths = {name: self.root / name for name in (
            "project", "medical", "data", "full_work", "paired", "online", "gnn",
            "source_bank", "nnroot", "raw", "preprocessed", "results", "logs", "output",
        )}
        paths.update(train_config=self.root / "train.json", nnunet_config=self.root / "nnunet.json",
                     outer_splits=self.root / "outer_splits.json")
        self.run = d.RuntimeLayout(d.Layout(**paths), 0, 730)
        self.args = SimpleNamespace(local_chunk_size=8, score_tolerance=0.02,
                                    score_rtol=0.005, device="cpu", dry_run=False, overwrite_banks=False)
        self.code_hashes = {"DEBUG/no_real_model.py": "debug-fixture-code-identity"}
        self.code_patch = patch.object(d, "_scoring_code_hashes", return_value=self.code_hashes)
        self.code_patch.start()
        self.addCleanup(self.code_patch.stop)
        self.source_index = {"format": d.BANK_FORMAT, "outer_fold": 0, "dataset_id": 730,
                             "candidate_count": 2, "entries_by_case": {"debug_case": ["debug_case/c0.npz"]}}
        self.write(self.run.base.source_bank / "index.json", json.dumps(self.source_index))
        self.write(self.run.base.source_bank / "debug_case/c0.npz", "DEBUG source bytes, never loaded")
        self.write(self.run.base.gnn / "model.pt", "DEBUG full checkpoint bytes, never loaded")
        self.write(self.run.base.gnn / "prototype.pt", "DEBUG prototype bytes, never loaded")
        for mode in d.MODES:
            self.write(self.run.base.gnn / "ablation_independent" / mode / "model.pt", f"DEBUG {mode} checkpoint bytes")
        self.write(self.run.base.train_config, json.dumps({"seed": 42}))
        self.write(self.run.base.nnunet_config, json.dumps({"dataset": {"plans": "DebugPlans", "configuration": "3d_fullres"}}))
        self.write(self.run.base.outer_splits, json.dumps({"splits": [{"train": ["debug_train"], "val": ["debug_case"]}]}))
        self.write(self.run.base.data / "image/debug_case_0000.nii.gz", "DEBUG image bytes, never loaded")
        self.write(self.run.base.data / "labels/debug_case.nii.gz", "DEBUG label bytes, never loaded")
        self.write(self.run.base.online / "folds/fold_0/training_complete.json", "DEBUG receipt hash fixture; not a real baseline")

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def contracts(self):
        return {mode: d._bank_input_contract(self.run, mode, self.source_index, self.args) for mode in d.MODES}

    def snapshot(self):
        return {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}

    def build_debug_banks(self):
        contracts = self.contracts()
        self.assertFalse(d._preflight_derived_outputs(self.run, self.source_index, contracts, overwrite=False))
        for mode in d.MODES:
            root = self.run.bank_for(mode)
            root.mkdir(parents=True)
            d._publish_report_json(root / "build_started.json", {
                "build_id": "debug-build", "input_contract_sha256": d._contract_sha256(contracts[mode]),
                "version": d.DERIVED_BANK_VERSION, "complete": False,
            })
            d._publish_bank_npz(root / "debug_case/c0.npz", {"scores": np.asarray([0.2, 0.8])}, overwrite=False)
            d._write_derived_bank(
                self.run, mode=mode, source_index=self.source_index,
                checkpoint_path=self.run.base.gnn / "ablation_independent" / mode / "model.pt",
                rows=[{"case_id": "debug_case", "entry": "debug_case/c0.npz", "status": "debug_fixture_only"}],
                input_contract=contracts[mode], build_id="debug-build",
            )
        return contracts


class DerivedBankReuseDebugTests(DownstreamReuseFixture):
    def test_verified_whole_pair_reuse_does_not_rewrite_any_file(self):
        contracts = self.build_debug_banks()
        before = self.snapshot()
        self.assertTrue(d._preflight_derived_outputs(self.run, self.source_index, contracts, overwrite=False))
        self.assertEqual(self.snapshot(), before)

    def test_one_existing_bank_entry_blocks_before_any_write(self):
        self.write(self.run.bank_for(d.MODES[0]) / "debug_case/c0.npz", "DEBUG existing partial")
        before = self.snapshot()
        with self.assertRaisesRegex(d.DownstreamAblationError, "part of the derived-bank pair"):
            d._preflight_derived_outputs(self.run, self.source_index, self.contracts(), overwrite=False)
        self.assertEqual(self.snapshot(), before)

    def test_both_existing_npzs_without_contract_are_not_relabelled(self):
        for mode in d.MODES:
            self.write(self.run.bank_for(mode) / "debug_case/c0.npz", "DEBUG old unverified scores")
        before = self.snapshot()
        with self.assertRaises(d.DownstreamAblationError):
            d._preflight_derived_outputs(self.run, self.source_index, self.contracts(), overwrite=False)
        self.assertEqual(self.snapshot(), before)

    def test_changed_gnn_or_source_entry_prevents_reuse(self):
        self.build_debug_banks()
        paths = [self.run.base.gnn / "ablation_independent" / d.MODES[0] / "model.pt",
                 self.run.base.source_bank / "debug_case/c0.npz"]
        for path in paths:
            with self.subTest(path=path):
                before = path.read_bytes()
                path.write_bytes(before + b" CHANGED DEBUG INPUT")
                with self.assertRaisesRegex(d.DownstreamAblationError, "input changed"):
                    d._verified_derived_bank(self.run, d.MODES[0])
                path.write_bytes(before)

    def test_changed_scoring_implementation_prevents_reuse(self):
        self.build_debug_banks()
        with patch.object(d, "_scoring_code_hashes", return_value={"DEBUG/new.py": "changed"}):
            with self.assertRaisesRegex(d.DownstreamAblationError, "implementation changed"):
                d._verified_derived_bank(self.run, d.MODES[0])

    def test_bank_cannot_bind_to_a_different_current_gnn_location(self):
        self.build_debug_banks()
        alternate = d.RuntimeLayout(replace(self.run.base, gnn=self.root / "different_gnn"), 0, 730)
        with self.assertRaisesRegex(d.DownstreamAblationError, "different current input path"):
            d._verified_derived_bank(alternate, d.MODES[0])

    def test_completion_mode_and_count_tampering_are_rejected(self):
        self.build_debug_banks()
        path = self.run.bank_for(d.MODES[0]) / "complete.json"
        original = d._load_json(path)
        for key, value in (("ablation_mode", d.MODES[1]), ("source_entries", 999), ("candidate_count", 1)):
            with self.subTest(key=key):
                self.write(path, json.dumps({**original, key: value}))
                with self.assertRaises(d.DownstreamAblationError):
                    d._verified_derived_bank(self.run, d.MODES[0])
        self.write(path, json.dumps(original))

    def test_changed_npz_or_manifest_prevents_reuse(self):
        self.build_debug_banks()
        for name in ("debug_case/c0.npz", "manifest.csv"):
            with self.subTest(name=name):
                path = self.run.bank_for(d.MODES[0]) / name
                before = path.read_bytes()
                path.write_bytes(before + b" CHANGED DEBUG OUTPUT")
                with self.assertRaisesRegex(d.DownstreamAblationError, "output changed"):
                    d._verified_derived_bank(self.run, d.MODES[0])
                path.write_bytes(before)

    def test_missing_output_is_not_complete(self):
        self.build_debug_banks()
        (self.run.bank_for(d.MODES[0]) / "manifest.csv").unlink()
        with self.assertRaises(d.DownstreamAblationError):
            d._verified_derived_bank(self.run, d.MODES[0])

    def test_new_started_marker_invalidates_old_completion(self):
        contracts = self.build_debug_banks()
        root = self.run.bank_for(d.MODES[0])
        self.write(root / "build_started.json", json.dumps({
            "build_id": "debug-interrupted-overwrite", "input_contract_sha256": d._contract_sha256(contracts[d.MODES[0]]),
        }))
        with self.assertRaisesRegex(d.DownstreamAblationError, "Incomplete, stale"):
            d._verified_derived_bank(self.run, d.MODES[0])

    def test_unplanned_files_are_preserved_even_with_overwrite_flag(self):
        self.write(self.run.bank_for(d.MODES[0]) / "user_notes.txt", "DEBUG valuable notes")
        before = self.snapshot()
        with self.assertRaisesRegex(d.DownstreamAblationError, "Unplanned derived files"):
            d._preflight_derived_outputs(self.run, self.source_index, self.contracts(), overwrite=True)
        self.assertEqual(self.snapshot(), before)

    def test_atomic_no_clobber_catches_a_race_and_preserves_existing_bytes(self):
        path = self.run.bank_for(d.MODES[0]) / "debug_case/c0.npz"
        self.write(path, "DEBUG concurrent owner's bytes")
        before = path.read_bytes()
        with self.assertRaisesRegex(d.DownstreamAblationError, "write collision"):
            d._publish_bank_npz(path, {"scores": np.ones(2)}, overwrite=False)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_explicit_overwrite_changes_only_requested_derived_file(self):
        destination = self.run.bank_for(d.MODES[0]) / "debug_case/c0.npz"
        self.write(destination, "DEBUG old derived bytes")
        source = self.run.base.source_bank / "debug_case/c0.npz"
        source_before = source.read_bytes()
        d._publish_bank_npz(destination, {"scores": np.asarray([1.0, 2.0])}, overwrite=True)
        with np.load(destination, allow_pickle=False) as payload:
            np.testing.assert_array_equal(payload["scores"], [1.0, 2.0])
        self.assertEqual(source.read_bytes(), source_before)

    def test_path_traversal_and_duplicate_source_entries_are_rejected(self):
        for name in ("../outside.npz", "/absolute.npz", "a/../b.npz", "C:/outside.npz"):
            with self.subTest(name=name), self.assertRaises(d.DownstreamAblationError):
                d._bank_entry_names({"entries_by_case": {"debug_case": [name]}})
        with self.assertRaisesRegex(d.DownstreamAblationError, "Duplicate bank"):
            d._bank_entry_names({"entries_by_case": {"a": ["same.npz"], "b": ["same.npz"]}})

    def test_new_rescore_is_blocked_before_outputs_when_baseline_is_unverified(self):
        before = self.snapshot()
        with patch.object(d, "_reference_training_contract", side_effect=d.DownstreamAblationError("DEBUG missing baseline receipt")):
            with self.assertRaisesRegex(d.DownstreamAblationError, "missing baseline receipt"):
                d._rescore_banks(self.run, self.args)
        self.assertEqual(self.snapshot(), before)

    def test_non_exact_argmax_baseline_is_not_accepted_as_full_reference(self):
        from tools import online_cp_benchmark as online
        with patch.object(online, "_verified_training_completion", return_value=(
            {}, {"trainers": {"basic": d.BASIC_TRAINER, "hier": online.TRAINER_HIER}}
        )):
            with self.assertRaisesRegex(d.DownstreamAblationError, "different experiments"):
                d._reference_training_contract(self.run)


class TrainingReuseDebugTests(DownstreamReuseFixture):
    def training_fixture(self):
        result = self.root / "new_results/DebugTrainer/fold_0"
        identity = {"format": d.TRAINING_CONTRACT_FORMAT, "validation_ids": ["debug_case"],
                    "input_files": {"config": d._file_record(self.run.base.train_config)},
                    "mode": "no_patient", "expected_epochs": 250}
        return result, identity

    def complete_training_fixture(self):
        result, identity = self.training_fixture()
        self.write(result / "checkpoint_final.pth", "DEBUG checkpoint bytes; no training")
        self.write(result / "training_log_debug.txt", "DEBUG log fixture; no training")
        self.write(result / "validation/debug_case.nii.gz", "DEBUG prediction bytes; not medical data")
        self.write(result / "validation/summary.json", "{}")
        d._publish_report_json(result / d.TRAINING_STARTED_NAME, {
            "input_contract": identity, "input_contract_sha256": d._contract_sha256(identity),
        })
        d._publish_report_json(result / d.TRAINING_COMPLETE_NAME, {
            "format": d.TRAINING_CONTRACT_FORMAT, "complete": True,
            "input_contract_sha256": d._contract_sha256(identity),
            "started_sha256": d._sha256(result / d.TRAINING_STARTED_NAME),
            "outputs": d._training_outputs(result, identity["validation_ids"]),
            "debug_fixture": True, "training_executed": False,
        })
        return result, identity

    def test_new_training_requires_absent_output(self):
        result, identity = self.training_fixture()
        self.assertEqual(d._training_reuse_action(result, identity), "new")
        self.assertFalse(result.exists())

    def test_legacy_final_and_partial_results_do_not_trigger_automatic_resume(self):
        result, identity = self.training_fixture()
        for name in ("checkpoint_final.pth", "checkpoint_latest.pth", "checkpoint_best.pth"):
            with self.subTest(name=name):
                self.write(result / name, "DEBUG unverified legacy checkpoint")
                before = self.snapshot()
                with self.assertRaisesRegex(d.DownstreamAblationError, "No automatic"):
                    d._training_reuse_action(result, identity)
                self.assertEqual(self.snapshot(), before)

    def test_complete_training_reuse_rehashes_outputs_and_does_not_rewrite(self):
        result, identity = self.complete_training_fixture()
        before = self.snapshot()
        self.assertEqual(d._training_reuse_action(result, identity), "reuse")
        self.assertEqual(self.snapshot(), before)
        self.write(result / "validation/debug_case.nii.gz", "DEBUG changed bytes")
        with self.assertRaisesRegex(d.DownstreamAblationError, "outputs/logs changed"):
            d._training_reuse_action(result, identity)

    def test_changed_training_identity_and_extra_validation_case_are_rejected(self):
        result, identity = self.complete_training_fixture()
        changed = copy.deepcopy(identity)
        changed["expected_epochs"] = 249
        with self.assertRaisesRegex(d.DownstreamAblationError, "identity changed"):
            d._training_reuse_action(result, changed)
        self.write(result / "validation/extra_case.nii.gz", "DEBUG extra bytes")
        with self.assertRaisesRegex(d.DownstreamAblationError, "incomplete or has extras"):
            d._training_reuse_action(result, identity)

    def test_ablation_output_override_keeps_basic_and_full_paths_unchanged(self):
        alternate = self.root / "fresh_ablation_results"
        run = d.RuntimeLayout(self.run.base, 0, 730, ablation_results=alternate)
        for trainer in (d.BASIC_TRAINER, d.FULL_TRAINER):
            self.assertTrue(d._model_dir(run, trainer).is_relative_to(self.run.base.results))
        for trainer in d.MODE_TRAINERS.values():
            self.assertTrue(d._model_dir(run, trainer).is_relative_to(alternate))
        self.assertEqual(d._nn_env(run, self.run.bank_for(d.MODES[0]) / "index.json", "cpu")["nnUNet_results"], str(alternate))

    def test_ablation_output_cli_is_opt_in(self):
        parser = d.build_parser()
        self.assertIsNone(parser.parse_args(["evaluate"]).ablation_results_output)
        self.assertEqual(parser.parse_args(["evaluate", "--ablation-results-output", "debug_fresh"]).ablation_results_output,
                         "debug_fresh")


if __name__ == "__main__":
    unittest.main()
