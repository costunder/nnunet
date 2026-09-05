"""DEBUG launch guards; no subprocess or training is started."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.train_online_curriculum import training_command, TRAINERS


class CurriculumLaunchDebugTests(unittest.TestCase):
    def test_resume_missing_checkpoint_never_reaches_nnunet_fallback(self):
        with tempfile.TemporaryDirectory(prefix="debug_curriculum_launch_") as tmp:
            root = Path(tmp)
            bank = root / "index.json"
            bank.write_text(json.dumps({"dataset_name": "Dataset900_Debug", "dataset_id": 900}))
            plans = root / "DebugPlans.json"
            plans.write_text(json.dumps({"configurations": {"3d_fullres": {}}}))
            config = Path(__file__).resolve().parents[1] / "config" / "online_cp_curriculum.json"
            args = SimpleNamespace(bank=str(bank), curriculum_config=str(config), arm="full",
                                   configuration="3d_fullres", device="cpu", seed=42, resume=True)
            identity = {"files": {"plans": {"path": str(plans)}}, "verified_configuration": "3d_fullres"}
            with patch.dict(os.environ, {"nnUNet_results": str(root / "results")}), patch(
                "tools.train_online_curriculum.verify_curriculum_bank_contract", return_value=identity):
                with self.assertRaisesRegex(FileNotFoundError, "Training was NOT launched"):
                    training_command(args)
                args.resume = False
                command, env, folder = training_command(args)
                self.assertIn(TRAINERS["full"], command)
                self.assertNotIn("--c", command)
                folder.mkdir(parents=True)
                (folder / "training_log.txt").write_text("debug previous run")
                with self.assertRaisesRegex(FileExistsError, "preserved"):
                    training_command(args)
                (folder / "checkpoint_latest.pth").write_bytes(b"debug presence only, trainer validates payload")
                args.resume = True
                command, env, _ = training_command(args)
                self.assertIn("--c", command)
                self.assertEqual(env["ONLINE_CP_SEED"], "42")
                args.configuration = "2d"
                with self.assertRaisesRegex(ValueError, "differs"):
                    training_command(args)


if __name__ == "__main__":
    unittest.main()
