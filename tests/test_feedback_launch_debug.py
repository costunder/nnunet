"""DEBUG launch/provenance guards; never launches a trainer or downloads data."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from custom_trainers.onlinecp_curriculum_contract import verify_curriculum_bank_contract
from tools.train_online_feedback import training_command, TRAINERS


ROOT = Path(__file__).resolve().parents[1]


class FeedbackLaunchDebugTests(unittest.TestCase):
    def test_contract_filename_is_not_an_arbitrary_path(self):
        for filename in ("../elsewhere.json", "index.json", "/tmp/contract.json"):
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "filename"):
                verify_curriculum_bank_contract(ROOT / "debug_nonexistent_bank.json",
                    curriculum_sha256="0" * 64, expected_candidate_count=128,
                    dataset_name="Dataset900_Debug", nnunet_fold=0, contract_filename=filename)

    def test_basic_launch_is_separate_and_missing_resume_never_starts_fresh(self):
        with tempfile.TemporaryDirectory(prefix="debug_feedback_launch_") as tmp:
            root = Path(tmp)
            bank = root / "index.json"
            bank.write_text(json.dumps({"dataset_name": "Dataset900_Debug", "dataset_id": 900}))
            plans = root / "DebugPlans.json"
            plans.write_text(json.dumps({"configurations": {"3d_fullres": {}}}))
            identity = {"files": {"plans": {"path": str(plans)}}, "verified_configuration": "3d_fullres"}
            args = SimpleNamespace(bank=str(bank), feedback_config=str(ROOT / "config/online_cp_feedback.json"),
                arm="basic", configuration="3d_fullres", device="cpu", seed=42, resume=True,
                feedback_gnn_config=None, feedback_raw_root=None)
            with patch.dict(os.environ, {"nnUNet_results": str(root / "results")}), patch(
                    "tools.train_online_feedback.verify_curriculum_bank_contract", return_value=identity) as verify:
                with self.assertRaisesRegex(FileNotFoundError, "NOT launched"):
                    training_command(args)
                args.resume = False
                command, env, folder = training_command(args)
                self.assertIn(TRAINERS["basic"], command)
                self.assertNotIn("--c", command)
                self.assertIn("ONLINE_CP_FEEDBACK_CONFIG", env)
                self.assertEqual(verify.call_args.kwargs["contract_filename"], "feedback_contract.json")
                folder.mkdir(parents=True)
                (folder / "previous_log.txt").write_text("debug existing result")
                with self.assertRaisesRegex(FileExistsError, "preserved"):
                    training_command(args)
                (folder / "checkpoint_latest.pth").write_bytes(b"debug existence only; trainer verifies payload")
                args.resume = True
                self.assertIn("--c", training_command(args)[0])
                args.configuration = "2d"
                with self.assertRaisesRegex(ValueError, "configuration"):
                    training_command(args)

    def test_full_cannot_silently_omit_gnn(self):
        with tempfile.TemporaryDirectory(prefix="debug_feedback_no_gnn_") as tmp:
            root = Path(tmp)
            bank = root / "index.json"
            bank.write_text(json.dumps({"dataset_name": "Dataset901_Debug", "dataset_id": 901}))
            args = SimpleNamespace(bank=str(bank), feedback_config=str(ROOT / "config/online_cp_feedback.json"),
                arm="full", configuration="3d_fullres", device="cpu", seed=42, resume=False,
                feedback_gnn_config=None, feedback_raw_root=None)
            identity = {"files": {"plans": {"path": str(root / "DebugPlans.json")}},
                        "verified_configuration": "3d_fullres"}
            with patch.dict(os.environ, {"nnUNet_results": str(root / "results")}), patch(
                    "tools.train_online_feedback.verify_curriculum_bank_contract", return_value=identity):
                with self.assertRaisesRegex(ValueError, "Full feedback requires"):
                    training_command(args)


if __name__ == "__main__":
    unittest.main()
