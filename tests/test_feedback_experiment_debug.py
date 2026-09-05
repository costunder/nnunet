"""DEBUG launcher contracts only: no real data, training, GPU or remote calls.

Empty medical-directory fixtures only exercise path checks. The explicit runner
double records commands without executing them; package fixtures are never
imported. Nothing here verifies a completed experiment or medical performance.
"""
import builtins
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import run_feedback_experiment as launcher


ROOT = Path(__file__).resolve().parents[1]
STAGES = ["install_private_trainers", "environment", "split", "gnn-prepare",
          "gnn-train", "plan", "bank", "feedback_contract", "check_full",
          "check_basic", "train_full", "train_basic"]


def file_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


class FeedbackExperimentDebugTests(unittest.TestCase):
    def fixture(self, root):
        project = root / "checkout with spaces" / "HierCP"
        medical = root / "DEBUG medical directories only"
        (project / "config").mkdir(parents=True)
        (project / "config/nnunet.json").write_bytes((ROOT / "config/nnunet.json").read_bytes())
        for name in ("Data/image", "Data/labels"):
            (medical / name).mkdir(parents=True)
        source = root / "DEBUG installed package" / "nnunetv2"
        trainer = source / "training/nnUNetTrainer"
        trainer.mkdir(parents=True)
        (source / "__init__.py").write_text("# DEBUG copy fixture; never imported\n", encoding="utf-8")
        (trainer / "nnUNetTrainer.py").write_text("DEBUG untouched native trainer", encoding="utf-8")
        for name in launcher.MODULES:
            (trainer / name).write_text("DEBUG existing custom implementation " + name, encoding="utf-8")
        plan = launcher.build_plan(project, medical, python_executable="DEBUG_PYTHON_NOT_EXECUTED")
        # Separate DEBUG path-only execution: production still requires 80 GiB.
        plan["minimum_free_bytes"] = 0
        return project, medical, source, plan

    @staticmethod
    def option(argv, name):
        return argv[argv.index(name) + 1]

    def test_plan_preserves_all_stages_and_checked_in_full_training_contract(self):
        plan = launcher.build_plan(ROOT, ROOT / "DEBUG_unused_medical")
        self.assertEqual([command["name"] for command in plan["commands"]], STAGES)
        self.assertEqual(plan["minimum_free_bytes"], 80 * 1024**3)
        self.assertEqual(json.loads((ROOT / "config/train.json").read_text())["training"]["epochs"], 40)
        feedback = json.loads((ROOT / "config/online_cp_feedback.json").read_text())
        self.assertEqual((feedback["num_epochs"], feedback["candidate_count"]), (250, 128))
        commands = {command["name"]: command["argv"] for command in plan["commands"]}
        for stage in ("split", "gnn-prepare", "gnn-train"):
            self.assertEqual(commands[stage][1:4], ["-m", "tools.paired_benchmark", stage])
        for stage in ("plan", "bank"):
            self.assertEqual(commands[stage][1:4], ["-m", "tools.online_cp_benchmark", stage])
            self.assertEqual(self.option(commands[stage], "--candidate-count"), "128")
        self.assertEqual(commands["feedback_contract"][1:3], ["-m", "tools.online_cp_curriculum"])
        self.assertTrue(self.option(commands["feedback_contract"], "--curriculum-config").endswith("online_cp_feedback.json"))
        for name in ("check_full", "check_basic", "train_full", "train_basic"):
            argv = commands[name]
            self.assertEqual(argv[1:3], ["-m", "tools.train_online_feedback"])
            self.assertEqual("--dry-run" in argv, name.startswith("check_"))
            self.assertEqual("--feedback-gnn-config" in argv, name.endswith("full"))
        for command in plan["commands"]:
            self.assertTrue(all(isinstance(arg, str) for arg in command["argv"]))
            for forbidden in ("--epochs", "--num-epochs", "--batch-size", "--subset", "--resume", "--overwrite", "--c"):
                self.assertNotIn(forbidden, command["argv"])

    def test_nested_project_nondefault_fold_dataset_and_seed_paths_are_consistent(self):
        with tempfile.TemporaryDirectory(prefix="debug_feedback_plan_") as tmp:
            project, medical, _, _ = self.fixture(Path(tmp))
            plan = launcher.build_plan(project, medical, outer_fold=3, dataset_id=807, seed=17,
                                       python_executable="DEBUG_PYTHON")
            self.assertEqual(plan["project_root"], project.resolve())
            self.assertEqual(plan["medical_root"], medical.resolve())
            self.assertEqual(plan["run_root"], project / "work/feedback_experiment")
            self.assertEqual(plan["package_destination"], plan["run_root"] / "runtime/nnunetv2")
            commands = {command["name"]: command["argv"] for command in plan["commands"]}
            for stage in ("split", "gnn-prepare", "gnn-train", "plan", "bank", "feedback_contract"):
                self.assertEqual(self.option(commands[stage], "--outer-fold"), "3")
                self.assertEqual(self.option(commands[stage], "--project-root"), str(project))
            for stage in ("plan", "bank", "feedback_contract"):
                self.assertEqual(self.option(commands[stage], "--dataset-id"), "807")
                self.assertEqual(self.option(commands[stage], "--paired-root"), "feedback_experiment/paired")
                self.assertEqual(self.option(commands[stage], "--online-root"), "feedback_experiment/online")
            for stage in ("check_full", "check_basic", "train_full", "train_basic"):
                self.assertEqual(self.option(commands[stage], "--bank"),
                                 str(plan["run_root"] / "online/folds/fold_3/bank/index.json"))
                self.assertEqual(self.option(commands[stage], "--seed"), "17")
            self.assertEqual(self.option(commands["train_full"], "--feedback-raw-root"),
                             str(Path(plan["env_updates"]["nnUNet_raw"]) / "Dataset807_LiverOnlineCP_OF3"))
            for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
                self.assertEqual(Path(plan["env_updates"][name]), plan["run_root"] / "online/nnunetv2" / name)
            for kwargs in ({"outer_fold": -1}, {"outer_fold": 5}, {"outer_fold": True},
                           {"dataset_id": 0}, {"dataset_id": 1000}, {"dataset_id": True},
                           {"seed": -1}, {"seed": True}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    launcher.build_plan(project, medical, **kwargs)

    def test_copy_excludes_only_designated_trainers_and_runtime_directories_without_mutating_original(self):
        with tempfile.TemporaryDirectory(prefix="debug_feedback_copy_") as tmp:
            _, _, source, _ = self.fixture(Path(tmp))
            preserved_name = next(iter(launcher.MODULES))
            other = source / "other"; other.mkdir()
            (other / preserved_name).write_text("DEBUG same filename outside trainer directory", encoding="utf-8")
            excluded = ("__pycache__", ".git", "nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
            for parent in (source, other):
                for name in excluded:
                    folder = parent / name; folder.mkdir()
                    (folder / "DEBUG_payload").write_text("not copied", encoding="utf-8")
            before = file_bytes(source)
            destination = Path(tmp) / "private_runtime/nnunetv2"
            launcher.copy_nnunet_package(source, destination)
            self.assertEqual(file_bytes(source), before)
            self.assertEqual((destination / "training/nnUNetTrainer/nnUNetTrainer.py").read_bytes(),
                             (source / "training/nnUNetTrainer/nnUNetTrainer.py").read_bytes())
            self.assertEqual((destination / "other" / preserved_name).read_bytes(), (other / preserved_name).read_bytes())
            for name in launcher.MODULES:
                self.assertFalse((destination / "training/nnUNetTrainer" / name).exists())
            for parent in (destination, destination / "other"):
                for name in excluded:
                    self.assertFalse((parent / name).exists())

    def test_copy_refuses_existing_destination_overlap_and_invalid_package(self):
        with tempfile.TemporaryDirectory(prefix="debug_feedback_copy_guard_") as tmp:
            _, _, source, _ = self.fixture(Path(tmp))
            destination = Path(tmp) / "existing"; destination.mkdir()
            sentinel = destination / "DEBUG_previous_result"; sentinel.write_bytes(b"preserve")
            before = file_bytes(source)
            for target in (destination, source, source / "nested_copy", source.parent):
                with self.subTest(target=target), self.assertRaises((FileExistsError, ValueError)):
                    launcher.copy_nnunet_package(source, target)
            with self.assertRaises(ValueError):
                launcher.copy_nnunet_package(Path(tmp) / "missing_package", Path(tmp) / "never_created")
            self.assertEqual(sentinel.read_bytes(), b"preserve")
            self.assertEqual(file_bytes(source), before)
            self.assertFalse((source / "nested_copy").exists())

    def test_success_calls_gpu_preflight_then_all_checked_commands_with_private_env_only(self):
        with tempfile.TemporaryDirectory(prefix="debug_feedback_execute_") as tmp:
            project, _, source, plan = self.fixture(Path(tmp))
            calls = []
            def runner(argv, **kwargs):
                calls.append((list(argv), dict(kwargs)))
                return subprocess.CompletedProcess(argv, 0)
            before_source = file_bytes(source)
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "MIG-DEBUG-ALLOCATION",
                                              "PYTHONPATH": "DEBUG_original_pythonpath",
                                              "nnUNet_results": "DEBUG_original_results"}):
                before_env = dict(os.environ)
                with mock.patch.object(launcher, "locate_nnunet_root", side_effect=AssertionError("DEBUG explicit package only")), \
                     mock.patch.object(launcher, "audit_sources", wraps=launcher.audit_sources) as audit, \
                     contextlib.redirect_stdout(io.StringIO()):
                    launcher.execute_plan(plan, runner=runner, package_root=source)
                self.assertEqual(dict(os.environ), before_env)
            self.assertEqual(audit.call_count, 1)
            self.assertEqual(len(calls), 13)
            self.assertEqual(calls[0][0][:2], [plan["python_executable"], "-c"])
            self.assertIn("torch.cuda.device_count()", calls[0][0][2])
            self.assertIn("if n != 1:", calls[0][0][2])
            self.assertEqual([argv for argv, _ in calls[1:]], [item["argv"] for item in plan["commands"]])
            for _, kwargs in calls:
                self.assertIs(kwargs["check"], True)
                self.assertEqual(kwargs["cwd"], project)
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "MIG-DEBUG-ALLOCATION")
                self.assertEqual(kwargs["env"]["PYTHONPATH"], os.pathsep.join(
                    [str(plan["run_root"] / "runtime"), str(project), "DEBUG_original_pythonpath"]))
                for name, value in plan["env_updates"].items():
                    self.assertEqual(kwargs["env"][name], value)
            self.assertEqual(file_bytes(source), before_source)
            self.assertTrue(plan["package_destination"].is_dir())
            saved = json.loads((plan["run_root"] / "launch_plan.json").read_text())
            self.assertEqual(saved["commands"], plan["commands"])

    def test_existing_output_missing_data_and_insufficient_space_fail_before_runner(self):
        for problem in ("existing", "missing_data", "space"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory(prefix="debug_feedback_preflight_") as tmp:
                _, medical, source, plan = self.fixture(Path(tmp))
                if problem == "existing":
                    plan["run_root"].mkdir(parents=True)
                    (plan["run_root"] / "DEBUG_result").write_bytes(b"preserve")
                elif problem == "missing_data":
                    plan["medical_root"] = medical / "DEBUG_missing_input"
                else:
                    plan["minimum_free_bytes"] = 80 * 1024**3
                runner = mock.Mock()
                with mock.patch.object(launcher.shutil, "disk_usage", return_value=SimpleNamespace(free=0)), \
                     mock.patch.object(launcher, "audit_sources") as audit:
                    with self.assertRaises((FileExistsError, FileNotFoundError, RuntimeError)):
                        launcher.execute_plan(plan, runner=runner, package_root=source)
                self.assertEqual(runner.call_count, 0)
                self.assertEqual(audit.call_count, 0)
                if problem == "existing":
                    self.assertEqual((plan["run_root"] / "DEBUG_result").read_bytes(), b"preserve")
                else:
                    self.assertFalse(plan["run_root"].exists())

    def test_invalid_source_or_source_audit_failure_never_checks_gpu_or_creates_output(self):
        for problem in ("package", "source_audit"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory(prefix="debug_feedback_source_guard_") as tmp:
                _, _, source, plan = self.fixture(Path(tmp))
                runner = mock.Mock()
                with mock.patch.object(launcher, "audit_sources", side_effect=RuntimeError("DEBUG changed trainer source")):
                    with self.assertRaises((ValueError, RuntimeError)):
                        launcher.execute_plan(plan, runner=runner,
                            package_root=source if problem == "source_audit" else source / "missing")
                self.assertEqual(runner.call_count, 0)
                self.assertFalse(plan["run_root"].exists())

    def test_gpu_preflight_failure_does_not_copy_package_or_create_experiment(self):
        with tempfile.TemporaryDirectory(prefix="debug_feedback_gpu_guard_") as tmp:
            _, _, source, plan = self.fixture(Path(tmp))
            runner = mock.Mock(side_effect=subprocess.CalledProcessError(1, ["DEBUG GPU check"]))
            with mock.patch.object(launcher, "audit_sources", return_value={}), \
                 mock.patch.object(launcher, "copy_nnunet_package") as copier:
                with self.assertRaises(subprocess.CalledProcessError):
                    launcher.execute_plan(plan, runner=runner, package_root=source)
            self.assertEqual(runner.call_count, 1)
            self.assertIs(runner.call_args.kwargs["check"], True)
            self.assertEqual(copier.call_count, 0)
            self.assertFalse(plan["run_root"].exists())

    def test_every_command_failure_stops_later_stages_and_preserves_partial_output(self):
        for failed_stage in range(len(STAGES)):
            with self.subTest(stage=STAGES[failed_stage]), tempfile.TemporaryDirectory(prefix="debug_feedback_stop_") as tmp:
                _, _, source, plan = self.fixture(Path(tmp))
                calls = []
                def runner(argv, **kwargs):
                    calls.append(list(argv))
                    self.assertIs(kwargs["check"], True)
                    if len(calls) == failed_stage + 2:
                        raise subprocess.CalledProcessError(7, argv)
                    return subprocess.CompletedProcess(argv, 0)
                before = file_bytes(source)
                with mock.patch.object(launcher, "audit_sources", return_value={}), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(subprocess.CalledProcessError):
                        launcher.execute_plan(plan, runner=runner, package_root=source)
                self.assertEqual(len(calls), failed_stage + 2)
                self.assertEqual(calls[1:], [entry["argv"] for entry in plan["commands"][:failed_stage + 1]])
                self.assertTrue((plan["run_root"] / "launch_plan.json").is_file())
                self.assertTrue(plan["package_destination"].is_dir())
                self.assertEqual(file_bytes(source), before)

    def test_escaping_output_root_is_rejected_before_runner(self):
        with tempfile.TemporaryDirectory(prefix="debug_feedback_output_guard_") as tmp:
            _, _, source, plan = self.fixture(Path(tmp))
            plan["run_root"] = Path(tmp) / "outside_checkout"
            runner = mock.Mock()
            with self.assertRaisesRegex(ValueError, "escapes"):
                launcher.execute_plan(plan, runner=runner, package_root=source)
            self.assertEqual(runner.call_count, 0)
            self.assertFalse(plan["run_root"].exists())

    def test_dry_run_import_and_main_do_not_import_gpu_nnunet_or_run_children(self):
        with tempfile.TemporaryDirectory(prefix="debug_feedback_dryrun_") as tmp:
            project, medical, _, plan = self.fixture(Path(tmp))
            before = file_bytes(Path(tmp))
            actual_import = builtins.__import__
            def guarded_import(name, *args, **kwargs):
                if name.split(".")[0] in ("torch", "nnunetv2"):
                    raise AssertionError("Dry run imported a GPU/nnU-Net dependency: " + name)
                return actual_import(name, *args, **kwargs)
            with mock.patch("builtins.__import__", side_effect=guarded_import):
                spec = importlib.util.spec_from_file_location("debug_feedback_launcher_import", ROOT / "tools/run_feedback_experiment.py")
                detached = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(detached)
                detached.PROJECT_ROOT = project
                output = io.StringIO()
                with mock.patch.object(detached, "execute_plan") as execute, \
                     mock.patch.object(detached, "locate_nnunet_root") as locate, \
                     mock.patch.object(detached.subprocess, "run") as child, \
                     contextlib.redirect_stdout(output):
                    detached.main(["--medical-root", str(medical), "--dry-run"])
                self.assertEqual((execute.call_count, locate.call_count, child.call_count), (0, 0, 0))
            self.assertIn("[DRY RUN ONLY]", output.getvalue())
            for stage in STAGES:
                self.assertIn("[" + stage + "]", output.getvalue())
            self.assertFalse(plan["run_root"].exists())
            self.assertEqual(file_bytes(Path(tmp)), before)

    def test_main_reports_failure_and_propagates_error_without_further_execution(self):
        with tempfile.TemporaryDirectory(prefix="debug_feedback_main_failure_") as tmp:
            project, medical, _, _ = self.fixture(Path(tmp))
            error_output = io.StringIO()
            with mock.patch.object(launcher, "PROJECT_ROOT", project), \
                 mock.patch.object(launcher, "execute_plan", side_effect=RuntimeError("DEBUG failed stage")) as execute, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error_output):
                with self.assertRaisesRegex(RuntimeError, "DEBUG failed stage"):
                    launcher.main(["--medical-root", str(medical)])
            self.assertEqual(execute.call_count, 1)
            self.assertIn("[FAILED] Further stages were not launched", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
