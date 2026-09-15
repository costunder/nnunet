"""DEBUG process/journal contracts; no medical data, GPU, or model training.

Tiny native Python children test actual producer receipts. Fresh pipeline tests
use explicit runtime/artifact-verifier doubles, never clinical completion proof.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid
from unittest import mock

from tools import run_feedback_experiment as launch
from tools import feedback_stage_execution as stages
from tools import feedback_fresh_execution as fresh

ROOT = Path(__file__).resolve().parents[1]


class StageProducerDebugTests(unittest.TestCase):
    def test_native_success_records_exit_without_serializing_environment(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_stage_receipt_") as tmp:
            root = Path(tmp)
            plan = {"run_root": root, "project_root": ROOT, "python_executable": sys.executable}
            row = {"attempt_id": uuid.uuid4().hex, "argv": [sys.executable, "-c", "print('DEBUG child only')"]}
            env = {**os.environ, "PYTHONPATH": str(ROOT), "DEBUG_SECRET_DO_NOT_WRITE": "not-a-real-secret-value"}
            stages.run_owned_command(plan, row, runner=subprocess.run, env=env)
            self.assertEqual(stages.verified_child_result(plan, row)["returncode"], 0)
            for path in root.rglob("*.json"):
                self.assertNotIn("not-a-real-secret-value", path.read_text())
            modified = {**row, "argv": [sys.executable, "-c", "print('different')"]}
            with self.assertRaisesRegex(ValueError, "argv"):
                stages.verified_child_result(plan, modified)

    def test_native_failure_has_terminal_receipt_not_fake_success(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_stage_failure_") as tmp:
            plan = {"run_root": Path(tmp), "project_root": ROOT, "python_executable": sys.executable}
            row = {"attempt_id": uuid.uuid4().hex,
                   "argv": [sys.executable, "-c", "raise ValueError('DEBUG injected native failure')"]}
            with self.assertRaises(subprocess.CalledProcessError):
                stages.run_owned_command(plan, row, runner=subprocess.run,
                                         env={**os.environ, "PYTHONPATH": str(ROOT)})
            self.assertNotEqual(stages.verified_child_result(plan, row)["returncode"], 0)

    def test_missing_exit_receipt_never_infers_child_stopped_from_parent(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_stage_ambiguous_") as tmp:
            root = Path(tmp)
            row = {"attempt_id": uuid.uuid4().hex, "argv": ["DEBUG_NOT_EXECUTED"]}
            path = root / "execution_attempts" / row["attempt_id"]
            stages.publish_new_json(path / "attempt.json", {"attempt_id": row["attempt_id"], "argv": row["argv"]})
            with self.assertRaisesRegex(ValueError, "may still be running"):
                stages.verified_child_result({"run_root": root}, row)

    def test_no_clobber_keeps_existing_and_failed_staging(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_publish_") as tmp:
            target = Path(tmp) / "commit.json"
            stages.publish_new_json(target, {"value": 1})
            before = target.read_bytes()
            with self.assertRaises(FileExistsError):
                stages.publish_new_json(target, {"value": 2})
            self.assertEqual(target.read_bytes(), before)
            with self.assertRaises(ValueError):
                stages.publish_new_json(Path(tmp) / "invalid.json", {"bad": float("nan")})
            self.assertFalse((Path(tmp) / "invalid.json").exists())
            self.assertTrue(list(Path(tmp).glob("*.staging")))

    def test_os_lock_exclusion_and_release_without_file_deletion(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_lock_") as tmp:
            with stages.run_lock(tmp):
                with self.assertRaises(OSError):
                    with stages.run_lock(tmp):
                        self.fail("Second owner entered a held lock")
            path = Path(tmp) / "feedback_execution.lock"
            self.assertTrue(path.is_file())
            with stages.run_lock(tmp):
                self.assertTrue(path.is_file())


class FreshJournalDebugTests(unittest.TestCase):
    def fixture(self, root):
        project, medical = root / "checkout", root / "DEBUG_medical"
        (project / "config").mkdir(parents=True)
        for name in ("train.json", "nnunet.json", "online_cp_feedback.json", "online_cp_feedback_gnn.json"):
            (project / "config" / name).write_bytes((ROOT / "config" / name).read_bytes())
        for name in ("Data/image", "Data/labels"):
            (medical / name).mkdir(parents=True)
        source = root / "DEBUG_native_package"
        (source / "training/nnUNetTrainer").mkdir(parents=True)
        (source / "__init__.py").write_text("# DEBUG never imported\n")
        (source / "training/nnUNetTrainer/nnUNetTrainer.py").write_text("# DEBUG native copy only\n")
        plan = launch.build_plan(project, medical, python_executable=sys.executable)
        plan["minimum_free_bytes"] = 0  # Explicit DEBUG path-only fixture.
        return plan, source

    def evidence(self, plan, name):
        path = plan["run_root"] / "DEBUG_evidence" / (name + ".json")
        return {"format": "DEBUG_boundary_not_training", "files": launch._bound_files(plan["run_root"], [path])}

    def runner(self, plan, calls, fail=None):
        commands = {tuple(command["argv"]): command["name"] for command in plan["commands"]}
        def run(argv, **kwargs):
            if argv[1] == "-c":
                return subprocess.CompletedProcess(argv, 0)  # GPU boundary double.
            name = commands[tuple(argv)]
            calls.append(name)
            if name == fail:
                raise OSError("DEBUG stage failure before output publication")
            path = plan["run_root"] / "DEBUG_evidence" / (name + ".json")
            path.parent.mkdir(exist_ok=True)
            with path.open("x") as handle:
                json.dump({"DEBUG_stage": name}, handle)
            return subprocess.CompletedProcess(argv, 0)
        return run

    @contextlib.contextmanager
    def boundaries(self):
        with mock.patch.object(launch, "audit_sources"), \
             mock.patch.object(launch, "_runtime_inventory", return_value={"DEBUG": "runtime-boundary"}), \
             mock.patch.object(launch, "_stage_evidence", side_effect=self.evidence), \
             contextlib.redirect_stdout(io.StringIO()):
            yield

    def test_failed_fresh_stage_resumes_without_repeating_completed_work(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_fresh_resume_") as tmp, self.boundaries():
            plan, source = self.fixture(Path(tmp))
            first = []
            with self.assertRaisesRegex(OSError, "DEBUG stage"):
                launch.execute_plan(plan, package_root=source, runner=self.runner(plan, first, fail="bank"))
            before = {p.name: p.read_bytes() for p in (plan["run_root"] / "DEBUG_evidence").iterdir()}
            journal = fresh.load_journal(plan, allow_debug=True)
            self.assertFalse(journal["complete"])
            second = []
            launch.execute_plan(plan, package_root=source, runner=self.runner(plan, second), resume_experiment=True)
            self.assertEqual(second, ["bank", "feedback_contract", "check_full", "check_basic", "train_full", "train_basic"])
            self.assertTrue(fresh.load_journal(plan, allow_debug=True)["complete"])
            for name, value in before.items():
                self.assertEqual((plan["run_root"] / "DEBUG_evidence" / name).read_bytes(), value)
            with self.assertRaisesRegex(ValueError, "DEBUG"):
                fresh.load_journal(plan)

    def test_config_and_completed_artifact_changes_refused_before_more_work(self):
        for change in ("config", "artifact"):
            with self.subTest(change=change), tempfile.TemporaryDirectory(prefix="DEBUG_fresh_changed_") as tmp, self.boundaries():
                plan, source = self.fixture(Path(tmp))
                with self.assertRaises(OSError):
                    launch.execute_plan(plan, package_root=source, runner=self.runner(plan, [], fail="bank"))
                path = (plan["train_config"] if change == "config" else
                        plan["run_root"] / "DEBUG_evidence/split.json")
                path.write_text('{"DEBUG_changed": true}')
                later = []
                with self.assertRaises(ValueError):
                    launch.execute_plan(plan, package_root=source, runner=self.runner(plan, later), resume_experiment=True)
                self.assertEqual(later, [])

    def test_native_training_failure_without_checkpoint_has_no_fresh_fallback(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_fresh_no_checkpoint_") as tmp, self.boundaries():
            plan, source = self.fixture(Path(tmp))
            with self.assertRaises(OSError):
                launch.execute_plan(plan, package_root=source, runner=self.runner(plan, [], fail="train_full"))
            calls = []
            with self.assertRaisesRegex(ValueError, "no checkpoint"):
                launch.execute_plan(plan, package_root=source, runner=self.runner(plan, calls), resume_experiment=True)
            self.assertEqual(calls, [])

    def test_legacy_unjournaled_fresh_is_not_adopted(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_fresh_legacy_") as tmp, self.boundaries():
            plan, source = self.fixture(Path(tmp))
            plan["run_root"].mkdir(parents=True)
            (plan["run_root"] / "launch_plan.json").write_text(json.dumps(plan, default=str))
            with self.assertRaises((ValueError, FileNotFoundError)):
                launch.execute_plan(plan, package_root=source, runner=self.runner(plan, []), resume_experiment=True)


if __name__ == "__main__":
    unittest.main()
