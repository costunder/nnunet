"""DEBUG launcher/provenance bridges; no training or medical data.

Completed-stage validation is explicitly replaced in the raw-marker tests;
the marker digest/cohort bridge itself and CLI plan construction are real.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from tools import run_feedback_experiment as launch

ROOT = Path(__file__).resolve().parents[1]


class FeedbackLaunchBridgeDebugTests(unittest.TestCase):
    def _main_failure(self, state, *, resume=False):
        """Real CLI/build_plan, one explicit execution-failure boundary only."""
        with tempfile.TemporaryDirectory(prefix="DEBUG_launch_failure_") as directory:
            project = Path(directory)
            shutil.copytree(ROOT / "config", project / "config")
            original = project / "work/original"
            original.mkdir(parents=True)
            (original / "checkpoint.pth").write_bytes(b"DEBUG preserved source; not a checkpoint")
            target = project / "work/new"
            if state != "absent":
                target.mkdir()
            if state in {"journal", "launch_only", "journal_directory"}:
                # Deliberately not a valid journal/plan: presence must not be
                # described as successful verification in an error handler.
                (target / "launch_plan.json").write_bytes(b"DEBUG unverified original plan")
            if state in {"journal", "journal_only"}:
                (target / "execution_journal.json").write_bytes(b"DEBUG unverified original journal")
            if state == "journal_directory":
                (target / "execution_journal.json").mkdir()
            paths_before = {path.relative_to(project).as_posix() for path in project.rglob("*")}
            files_before = {path.relative_to(project).as_posix(): path.read_bytes()
                            for path in project.rglob("*") if path.is_file()}
            argv = ["--medical-root", str(project / "medical"), "--experiment-name", "new",
                    "--reuse-preprocessing-from", "work/original", "--evaluate"]
            if resume:
                argv.append("--resume-experiment")
            failure = ValueError("DEBUG execution boundary failure; original exception must propagate")
            stdout, stderr = io.StringIO(), io.StringIO()
            with mock.patch.object(launch, "PROJECT_ROOT", project), \
                 mock.patch.object(launch, "execute_plan", side_effect=failure) as execute, \
                 mock.patch.object(launch.subprocess, "run", side_effect=AssertionError("DEBUG evaluation must not launch")) as child, \
                 contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaises(ValueError) as raised:
                    launch.main(argv)
            self.assertIs(raised.exception, failure)
            execute.assert_called_once()
            self.assertEqual(execute.call_args.kwargs, {"resume_preparation": False, "resume_experiment": resume})
            self.assertEqual(execute.call_args.args[0]["run_root"], target)
            child.assert_not_called()
            self.assertEqual(paths_before, {path.relative_to(project).as_posix() for path in project.rglob("*")})
            self.assertEqual(files_before, {path.relative_to(project).as_posix(): path.read_bytes()
                                           for path in project.rglob("*") if path.is_file()})
            self.assertEqual(target.exists(), state != "absent")
            self.assertNotIn("[TRAINING COMMANDS COMPLETED]", stdout.getvalue())
            self.assertIn("[FAILED]", stderr.getvalue())
            return stderr.getvalue()

    def test_main_prelaunch_failure_never_claims_a_journal_or_recommends_resume(self):
        message = self._main_failure("absent")
        self.assertIn("[PRE-LAUNCH]", message)
        self.assertIn("SAME arguments WITHOUT --resume-experiment", message)
        self.assertIn("no experiment directory or stage journal was created", message)
        self.assertNotIn("[JOURNAL PRESENT]", message)
        self.assertNotIn("[PARTIAL ROOT]", message)
        self.assertNotIn("[MISSING RESUME ROOT]", message)

    def test_main_missing_resume_root_never_recommends_a_fresh_restart(self):
        message = self._main_failure("absent", resume=True)
        self.assertIn("[MISSING RESUME ROOT]", message)
        self.assertIn("inspect", message)
        self.assertIn("No automatic fresh restart", message)
        self.assertNotIn("SAME arguments WITHOUT --resume-experiment", message)
        self.assertNotIn("[PRE-LAUNCH]", message)
        self.assertNotIn("[JOURNAL PRESENT]", message)
        self.assertNotIn("[PARTIAL ROOT]", message)

    def test_main_journal_presence_is_not_a_claim_of_valid_completion(self):
        for resume in (False, True):
            with self.subTest(resume=resume):
                message = self._main_failure("journal", resume=resume)
                self.assertIn("[JOURNAL PRESENT]", message)
                self.assertIn("--resume-experiment", message)
                self.assertIn("reverified", message)
                self.assertNotIn("has a durable stage journal", message)
                self.assertNotIn("[PRE-LAUNCH]", message)
                self.assertNotIn("[PARTIAL ROOT]", message)

    def test_main_partial_root_never_recommends_blind_fresh_or_resume(self):
        for state in ("empty", "launch_only", "journal_only", "journal_directory"):
            with self.subTest(state=state):
                message = self._main_failure(state)
                self.assertIn("[PARTIAL ROOT]", message)
                self.assertIn("Preserve", message)
                self.assertIn("inspect", message)
                self.assertIn("No automatic retry", message)
                self.assertNotIn("--resume-experiment", message)
                self.assertNotIn("[PRE-LAUNCH]", message)
                self.assertNotIn("[JOURNAL PRESENT]", message)

    def test_read_only_lock_never_creates_or_initializes_a_source_file(self):
        from tools.feedback_stage_execution import run_lock
        with tempfile.TemporaryDirectory(prefix="DEBUG_readonly_lock_") as directory:
            root = Path(directory)
            path = root / "feedback_execution.lock"
            with self.assertRaises(FileNotFoundError):
                with run_lock(root, create=False):
                    self.fail("An absent source lock was silently created")
            self.assertEqual(list(root.iterdir()), [])
            path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty"):
                with run_lock(root, create=False):
                    self.fail("An empty source lock was initialized")
            self.assertEqual(path.read_bytes(), b"")

    def test_read_only_existing_lock_checks_occupancy_and_preserves_bytes(self):
        from tools.feedback_stage_execution import run_lock
        with tempfile.TemporaryDirectory(prefix="DEBUG_existing_lock_") as directory:
            root = Path(directory)
            path = root / "feedback_execution.lock"
            before = b"DEBUG existing lock marker"
            path.write_bytes(before)
            with run_lock(root):
                with self.assertRaises(OSError):
                    with run_lock(root, create=False):
                        self.fail("An occupied source lock was accepted")
            # Windows byte-range locks prohibit reading the held byte through
            # another descriptor; byte preservation is checked after release.
            self.assertEqual(path.read_bytes(), before)
            with run_lock(root, create=False):
                self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes(), before)

    def test_default_cli_plan_still_builds_without_reuse(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            launch.main(["--medical-root", str(ROOT / "DEBUG_unused_medical"), "--dry-run"])
        self.assertIn("[gnn-train]", output.getvalue())
        self.assertIn("[train_basic]", output.getvalue())
        self.assertNotIn("preprocessing_source_root", launch.build_plan(ROOT, ROOT / "DEBUG_unused_medical"))

    def test_preprocessing_reuse_replaces_only_native_plan_command(self):
        common = dict(experiment_name="DEBUG_v4", python_executable="DEBUG_not_executed")
        before = launch.build_plan(ROOT, ROOT / "DEBUG_medical", **common)
        after = launch.build_plan(ROOT, ROOT / "DEBUG_medical", **common,
                                  reuse_preprocessing_from="work/feedback_medical_aug")
        self.assertEqual(before["minimum_free_bytes"], after["minimum_free_bytes"])
        self.assertEqual(len(before["commands"]), len(after["commands"]))
        for original, current in zip(before["commands"], after["commands"]):
            self.assertEqual(original["name"], current["name"])
            if current["name"] == "plan":
                self.assertEqual(current["argv"][1:3], ["-m", "tools.feedback_preprocessing_reuse"])
            else:
                self.assertEqual(original, current)
        self.assertEqual(after["preprocessing_source_root"], ROOT / "work/feedback_medical_aug")

    def test_reuse_mode_conflicts_and_overlapping_paths_rejected(self):
        for extra in ({"recover_from": "work/old"}, {"upgrade_bank_from": "work/old"}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                launch.build_plan(ROOT, ROOT, reuse_preprocessing_from="work/native", **extra)
        for source in ("work", "../outside", "work/DEBUG_v4", "work/DEBUG_v4/nested"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                launch.build_plan(ROOT, ROOT, experiment_name="DEBUG_v4", reuse_preprocessing_from=source)

    def test_training_proof_carries_only_bound_original_raw_input_contract(self):
        from tools import online_cp_benchmark as online
        with tempfile.TemporaryDirectory(prefix="DEBUG_feedback_proof_") as directory:
            project = Path(directory)
            root = project / "work/experiment"
            raw = root / "raw"
            raw.mkdir(parents=True)
            plan = {"project_root": str(project), "medical_root": str(project / "medical"),
                    "run_root": str(root), "train_config": str(project / "config/train.json"),
                    "package_destination": str(root / "runtime/nnunetv2"),
                    "outer_fold": 0, "dataset_id": 760}
            (root / "launch_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (root / "execution_journal.json").write_text("DEBUG journal boundary", encoding="utf-8")
            contract = {"dataset_name": "Dataset760_LiverOnlineCP_OF0", "val_ids": ["case_val"],
                        "source_cases": [{"case_id": "case_val", "image_sha256": "a" * 64,
                                          "label_sha256": "b" * 64}]}
            raw_path = raw / online.RAW_MARKER_NAME
            marker_path = root / "native_marker.json"
            for changed in (None, "marker", "raw", "cohort", "dataset"):
                original = copy.deepcopy(contract)
                if changed == "cohort":
                    original["val_ids"] = ["wrong_case"]
                if changed == "dataset":
                    original["dataset_name"] = "wrong_dataset"
                raw_path.write_text(json.dumps(original), encoding="utf-8")
                marker = {"input_contract": {"raw_marker_sha256": launch._file_sha256(raw_path),
                                              "raw_contract_sha256": online.value_sha256(original)}}
                marker_path.write_text(json.dumps(marker), encoding="utf-8")
                identity = {"dataset_name": contract["dataset_name"], "files": {
                    "preprocess_marker": {"path": str(marker_path), "sha256": launch._file_sha256(marker_path)}}}
                if changed == "marker":
                    marker_path.write_text("{}", encoding="utf-8")
                if changed == "raw":
                    raw_path.write_text("{}", encoding="utf-8")
                with self.subTest(changed=changed), mock.patch.object(launch, "PROJECT_ROOT", project), \
                     mock.patch.object(launch, "_bound_files"), \
                     mock.patch("tools.feedback_stage_execution.run_lock", return_value=contextlib.nullcontext()), \
                     mock.patch("tools.feedback_fresh_execution.load_journal", return_value={"complete": True, "runtime_inventory": {}}), \
                     mock.patch.object(launch, "_verify_online_artifacts", return_value=identity), \
                     mock.patch.object(launch, "_stage_evidence"), \
                     mock.patch.object(launch, "_training_folder", side_effect=lambda p, arm: root / arm), \
                     mock.patch.object(online, "make_layout", return_value=object()), \
                     mock.patch.object(online, "outer_split", return_value={"val": ["case_val"]}), \
                     mock.patch.object(online, "raw_dataset_dir", return_value=raw):
                    if changed is None:
                        proof = launch.load_completed_experiment(root)
                        self.assertEqual(proof["raw_input_contract"], contract)
                    else:
                        with self.assertRaises(ValueError):
                            launch.load_completed_experiment(root)


if __name__ == "__main__":
    unittest.main()
