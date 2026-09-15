"""CPU DEBUG boundary fixtures; no medical data or real training is claimed."""
from __future__ import annotations

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_feedback_experiment as runner
from tools import feedback_basic_reuse as reuse


class BasicReuseDebug(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cpbr_")
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        (self.project / "config").mkdir()
        config = runner._read_json(runner.PROJECT_ROOT / "config/nnunet.json")
        (self.project / "config/nnunet.json").write_text(json.dumps(config), encoding="utf-8")
        self.source = self.project / "work/old"
        self.source.mkdir(parents=True)
        self.plan = runner.build_plan(self.project, self.project / "Medical",
                                      experiment_name="new", reuse_basic_from="work/old")
        self.root = self.plan["run_root"]
        self.root.mkdir(parents=True)
        self.proof = {
            "format": "hiercp_basic_source_provenance_v1", "source_root": str(self.source),
            "checkpoint": {"path": str(self.source / "checkpoint_final.pth"), "sha256": "a" * 64},
            "bank_identity": {"original_bank": "v3"}, "runtime_inventory": {"native.py": "b" * 64},
            "training_journal_sha256": "c" * 64, "bank_binding": {"DEBUG": True},
            "training_contract": {"DEBUG": True}, "native_equivalence": {"DEBUG": True},
            "source_files": {"DEBUG": True},
        }

    def test_default_plan_retains_both_real_training_jobs(self):
        default = runner.build_plan(self.project, self.project / "Medical", experiment_name="default")
        explicit = runner.build_plan(self.project, self.project / "Medical", experiment_name="default",
                                     reuse_basic_from=None)
        self.assertEqual(default, explicit)
        self.assertNotIn("basic_source_root", default)
        self.assertIn("train_basic", [row["name"] for row in default["commands"]])

    def test_reuse_preflights_then_certifies_before_full_without_basic_job(self):
        names = [row["name"] for row in self.plan["commands"]]
        self.assertLess(names.index("basic_source_preflight"), names.index("split"))
        self.assertLess(names.index("feedback_contract"), names.index("basic_reuse"))
        self.assertLess(names.index("basic_reuse"), names.index("train_full"))
        self.assertNotIn("train_basic", names)
        self.assertNotIn("check_basic", names)
        for stage in ("split", "gnn-prepare", "gnn-train", "plan", "bank", "check_full", "train_full"):
            self.assertEqual(names.count(stage), 1)
        self.assertIn("no Basic retraining", self.plan["scope"])

    def test_preprocessing_and_basic_may_share_the_same_preserved_source(self):
        plan = runner.build_plan(self.project, self.project / "Medical", experiment_name="combined",
                                 reuse_basic_from="work/old", reuse_preprocessing_from="work/old")
        self.assertEqual(plan["basic_source_root"], plan["preprocessing_source_root"])

    def test_recovery_and_upgrade_are_not_silently_extended(self):
        for option in ("recover_from", "upgrade_bank_from"):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "NEW fresh"):
                runner.build_plan(self.project, self.project / "Medical", reuse_basic_from="work/old",
                                  **{option: "work/old"})

    def test_overlapping_and_outside_sources_refused(self):
        for source in ("work", "work/new", "work/new/child", "config"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                runner.build_plan(self.project, self.project / "Medical", experiment_name="new",
                                  reuse_basic_from=source)

    def test_source_publication_is_idempotent_and_preserves_attempt(self):
        with patch.object(reuse, "inspect_source", return_value=self.proof):
            reuse.admit_source(self.plan)
            path = self.root / "basic_reuse/source.json"
            before = path.read_bytes(), path.stat().st_mtime_ns
            reuse.admit_source(self.plan)
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
            self.assertEqual(reuse.verify_source(self.plan), self.proof)
        self.assertEqual(len(list((self.root / "basic_reuse/publication_attempts").glob("*.json"))), 1)
        self.assertEqual(list(self.source.iterdir()), [])

    def test_changed_source_never_overwrites_original_admission(self):
        with patch.object(reuse, "inspect_source", return_value=self.proof):
            reuse.admit_source(self.plan)
        path = self.root / "basic_reuse/source.json"
        before = path.read_bytes()
        changed = copy.deepcopy(self.proof)
        changed["checkpoint"]["sha256"] = "d" * 64
        with patch.object(reuse, "inspect_source", return_value=changed):
            with self.assertRaisesRegex(ValueError, "conditions changed"):
                reuse.verify_source(self.plan)
            with self.assertRaisesRegex(ValueError, "Different existing"):
                reuse.admit_source(self.plan)
        self.assertEqual(path.read_bytes(), before)

    def test_bad_checksum_is_not_republished(self):
        with patch.object(reuse, "inspect_source", return_value=self.proof):
            reuse.admit_source(self.plan)
            path = self.root / "basic_reuse/source.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["receipt_sha256"] = "0" * 64
            path.write_text(json.dumps(value), encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "checksum changed"):
                reuse.admit_source(self.plan)
            self.assertEqual(path.read_bytes(), before)

    def test_checkpoint_origin_never_becomes_target_bank(self):
        receipt = {"format": reuse.REUSE_FORMAT, "source": self.proof,
                   "target_bank_identity": {"new_bank": "v4"}}
        path = self.root / "basic_reuse/receipt.json"
        reuse._publish(self.root, path, receipt)
        with patch.object(reuse, "verify_reuse", return_value=receipt):
            origin = reuse.basic_origin(self.plan)
        self.assertEqual(origin["bank_identity"], self.proof["bank_identity"])
        self.assertEqual(origin["checkpoint"], self.proof["checkpoint"])
        self.assertEqual(origin["experiment_root"], str(self.source))
        self.assertEqual(origin["reuse_receipt"]["sha256"], runner._file_sha256(path))
        self.assertEqual(list(self.source.iterdir()), [])

    def test_changed_target_receipt_refused_without_replacement(self):
        first = {"format": reuse.REUSE_FORMAT, "DEBUG": "first"}
        with patch.object(reuse, "_certification", return_value=first):
            reuse.certify(self.plan)
            self.assertEqual(reuse.verify_reuse(self.plan), first)
        path = self.root / "basic_reuse/receipt.json"
        before = path.read_bytes()
        with patch.object(reuse, "_certification", return_value={**first, "DEBUG": "changed"}):
            with self.assertRaisesRegex(ValueError, "dependencies changed"):
                reuse.verify_reuse(self.plan)
            with self.assertRaisesRegex(ValueError, "Different existing"):
                reuse.certify(self.plan)
        self.assertEqual(path.read_bytes(), before)

    def test_unrecorded_worker_cannot_publish_receipts(self):
        (self.root / "launch_plan.json").write_text(json.dumps(self.plan, default=str), encoding="utf-8")
        row = {"name": "basic_source_preflight", "status": "running",
               "execution_backend": "injected_debug_runner"}
        journal = {"format": "feedback_fresh_execution_v1", "plan_sha256": runner._json_sha256(self.plan),
                   "stages": [row]}
        journal["journal_sha256"] = runner._json_sha256(journal)
        (self.root / "execution_journal.json").write_text(json.dumps(journal), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "recorded native"):
            reuse._worker_plan(self.root, "basic_source_preflight")
        self.assertFalse((self.root / "basic_reuse").exists())

    def test_cli_exposes_opt_in_without_launching_any_training(self):
        with patch.object(runner, "PROJECT_ROOT", self.project), \
             patch.object(runner, "locate_nnunet_root", return_value=self.project / "native"), \
             patch.object(reuse, "inspect_source", return_value=self.proof) as inspect, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            runner.main(["--medical-root", str(self.project / "Medical"), "--experiment-name", "new",
                         "--reuse-basic-from", "work/old", "--dry-run"])
        inspect.assert_called_once()
        self.assertIn("[basic_source_preflight]", output.getvalue())
        self.assertIn("[basic_reuse]", output.getvalue())
        self.assertIn("[train_full]", output.getvalue())
        self.assertNotIn("[train_basic]", output.getvalue())
        self.assertFalse((self.root / "basic_reuse").exists())

    def test_source_and_reuse_stages_require_actual_verifiers(self):
        folder = self.root / "basic_reuse"
        folder.mkdir()
        for stage, name, verifier in (("basic_source_preflight", "source.json", "verify_source"),
                                      ("basic_reuse", "receipt.json", "verify_reuse")):
            path = folder / name
            path.write_text("{}", encoding="utf-8")
            with self.subTest(stage=stage), patch.object(reuse, verifier, return_value={}) as check:
                evidence = runner._stage_evidence(self.plan, stage)
                check.assert_called_once_with(self.plan)
                self.assertEqual(evidence["files"], {"basic_reuse/" + name: runner._file_sha256(path)})
            with patch.object(reuse, verifier, side_effect=ValueError("DEBUG mismatch")), \
                 self.assertRaisesRegex(ValueError, "DEBUG mismatch"):
                runner._stage_evidence(self.plan, stage)

    def _native_fixture(self):
        from tools import online_cp_benchmark as online
        dataset = "Dataset760_LiverOnlineCP_OF0"
        raw = {"dataset_name": dataset, "train_ids": ["DEBUG_train"], "val_ids": ["DEBUG_val"],
               "source_cases": [{"case_id": "DEBUG_train", "image_sha256": "a" * 64,
                                  "label_sha256": "b" * 64}]}
        raw_path = Path(self.plan["env_updates"]["nnUNet_raw"]) / dataset / online.RAW_MARKER_NAME
        raw_path.parent.mkdir(parents=True)
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        # Native full-output verification is an explicit DEBUG boundary here.
        # These tests exercise the additional cross-experiment equality bridge.
        outputs = {"DEBUG_validated_full_output_manifest": "c" * 64}
        marker = {"input_contract": {"raw_marker_sha256": runner._file_sha256(raw_path),
                                      "raw_contract_sha256": online.value_sha256(raw)}, "outputs": outputs}
        path = self.root / "DEBUG_native_marker.json"
        path.write_text(json.dumps(marker), encoding="utf-8")
        record = {"path": str(path), "sha256": runner._file_sha256(path)}
        identity = {"dataset_name": dataset, "files": {"preprocess_marker": record}}
        source = copy.deepcopy(self.proof)
        source["native_equivalence"]["source_preprocessing"] = {
            "marker": {"path": str(self.source / "DEBUG_marker.json"), "sha256": "d" * 64},
            "raw_input_contract": marker["input_contract"], "native_outputs": copy.deepcopy(outputs),
            "raw_dataset_contract": copy.deepcopy(raw)}
        return source, identity, raw_path

    def test_equal_native_outputs_still_require_actual_target_validation(self):
        source, identity, _ = self._native_fixture()
        with patch.object(runner, "_verify_online_artifacts", return_value={}) as verify:
            first = reuse._verify_native_training_inputs(self.plan, source, identity)
            second = reuse._verify_native_training_inputs(self.plan, source, identity)
        self.assertEqual(first, second)
        self.assertEqual(verify.call_count, 2)
        with patch.object(runner, "_verify_online_artifacts", side_effect=ValueError("DEBUG native corrupt")), \
             self.assertRaisesRegex(ValueError, "DEBUG native corrupt"):
            reuse._verify_native_training_inputs(self.plan, source, identity)

    def test_cp_equality_cannot_hide_changed_native_training_data(self):
        source, identity, _ = self._native_fixture()
        original = source["native_equivalence"]["source_preprocessing"]
        original["native_outputs"]["DEBUG_validated_full_output_manifest"] = "e" * 64
        with patch.object(runner, "_verify_online_artifacts", return_value={}), \
             self.assertRaisesRegex(ValueError, "native data/seg/properties"):
            reuse._verify_native_training_inputs(self.plan, source, identity)

    def test_changed_original_full_ct_hash_is_not_only_a_cp_patch_difference(self):
        source, identity, _ = self._native_fixture()
        original = source["native_equivalence"]["source_preprocessing"]
        original["raw_dataset_contract"]["source_cases"][0]["image_sha256"] = "e" * 64
        with patch.object(runner, "_verify_online_artifacts", return_value={}), \
             self.assertRaisesRegex(ValueError, "source_cases"):
            reuse._verify_native_training_inputs(self.plan, source, identity)

    def test_raw_marker_tamper_is_rejected(self):
        source, identity, raw_path = self._native_fixture()
        raw_path.write_text("{}", encoding="utf-8")
        with patch.object(runner, "_verify_online_artifacts", return_value={}), \
             self.assertRaisesRegex(ValueError, "certified raw cohort"):
            reuse._verify_native_training_inputs(self.plan, source, identity)


if __name__ == "__main__":
    unittest.main()
