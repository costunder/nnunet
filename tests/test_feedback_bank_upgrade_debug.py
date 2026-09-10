"""DEBUG orchestration/storage tests: synthetic files, no patients/GPU/training.

Heavy native GNN/preprocessing verification has explicit boundary doubles here;
the real journal hashes, file views, runtime inventory and stage driver run.
Separate native contract tests remain required for actual medical artifacts.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from tools import feedback_bank_upgrade as upgrade
from tools import run_feedback_experiment as launch
from tools import online_cp_benchmark as online

ROOT = Path(__file__).resolve().parents[1]
NEW_MODULE = b"# DEBUG current private trainer; never imported\n"


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else
                     json.dumps(value, default=str, indent=2).encode())


def inventory(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


class BankUpgradeDebugTests(unittest.TestCase):
    def fixture(self, temp):
        project, medical = temp / "p", temp / "m"
        for name in ("train.json", "nnunet.json", "online_cp_feedback.json", "online_cp_feedback_gnn.json"):
            put(project / "config" / name, (ROOT / "config" / name).read_bytes())
        for folder in ("Data/image", "Data/labels"):
            (medical / folder).mkdir(parents=True)
        source = project / "work/old"
        old = launch.build_plan(project, medical, experiment_name="old", recover_from="work/failed",
                                python_executable="DEBUG_PYTHON_NOT_EXECUTED")
        put(old["train_config"], (ROOT / "config/train.json").read_bytes())
        package = old["package_destination"]
        put(package / "__init__.py", b"# DEBUG native package; never imported\n")
        put(package / "training/nnUNetTrainer/nnUNetTrainer.py", b"# DEBUG native trainer\n")
        put(package / "training/nnUNetTrainer/debug_current.py", b"# DEBUG historical helper\n")
        put(source / "launch_plan.json", old)
        put(source / "recovery/complete.json", {"DEBUG": "native receipt boundary double"})
        put(source / "paired/DEBUG_gnn_evidence", b"DEBUG completed-GNN native boundary")
        name = "Dataset760_LiverOnlineCP_OF0"
        raw = Path(old["env_updates"]["nnUNet_raw"]) / name
        for folder in ("imagesTr", "labelsTr", "imagesTs"):
            (raw / folder).mkdir(parents=True)
        put(raw / "online_cp_dataset.json", {"materialization": "hardlink", "DEBUG": True})
        put(raw / "dataset.json", {"DEBUG": True})
        put(raw / "imagesTr/debug_0000.nii.gz", b"DEBUG not a medical image")
        put(raw / "labelsTr/debug.nii.gz", b"DEBUG not a medical label")
        pre = Path(old["env_updates"]["nnUNet_preprocessed"]) / name
        data = pre / "DEBUG_data"
        data.mkdir(parents=True)
        np.savez_compressed(data / "debug.npz", data=np.arange(8,dtype=np.float32).reshape(1,2,2,2),
                            seg=np.zeros((1,2,2,2),dtype=np.int16))
        put(data / "debug.pkl", b"DEBUG metadata, not deserialized")
        put(pre / "online_cp_preprocess_complete.json", {"DEBUG": "native completion double",
            "input_contract": {"train_ids": ["debug"]},
            "outputs": {"data_identifier": "DEBUG_data", "storage_format": "npz",
                        "cases": [{"case_id": "debug"}]}})
        put(pre / "dataset.json", {"DEBUG": True})
        put(pre / "splits_final.json", [{"train": ["debug"], "val": ["DEBUG_val"]}])
        identity_root = project / "work/failed"
        put(identity_root / "DEBUG_original_input", b"DEBUG immutable preparation-source evidence")
        source_identity = {"source_root": str(identity_root),
                           "files": launch._bound_files(identity_root, [identity_root / "DEBUG_original_input"])}
        journal = dict(format="feedback_preparation_execution_v1", plan_sha256=launch._json_sha256(old),
                       source_identity=source_identity,
                       runtime_inventory=launch._bound_files(package, [p for p in package.rglob("*") if p.is_file()]),
                       preparation_receipt={"DEBUG": "native receipt double"}, stages=[],
                       training_started=True, complete=False)
        for stage in ("copy_private_runtime", "install_private_trainers", "environment", "recover_preparation"):
            journal["stages"].append({"name": stage, "status": "completed"})
        evidence = {"gnn-train": {"format": "feedback_stage_completion_v1", "files": launch._bound_files(source, [source / "paired/DEBUG_gnn_evidence"])},
                    "plan": {"format": "feedback_stage_completion_v1", "files": launch._bound_files(source, [pre / "online_cp_preprocess_complete.json"])}}
        for stage in ("gnn-train", "plan"):
            journal["stages"].append({"name": stage, "status": "completed", "input_files": launch._resume_inputs(old),
                                       "completion_evidence": evidence[stage]})
        journal["stages"].append({"name": "bank", "status": "failed", "input_files": launch._resume_inputs(old),
                                  "error": "DEBUG old SourceMappingError; must be preserved"})
        launch._save_journal(source, journal)
        put(source / "online/folds/fold_0/bank/manifest.csv", b"status,reason\nerror,DEBUG historical donor mapping\n")
        plan = launch.build_plan(project, medical, upgrade_bank_from="work/old", experiment_name="new",
                                 python_executable="DEBUG_PYTHON_NOT_EXECUTED")
        plan["minimum_free_bytes"] = 0  # DEBUG path/storage fixture; production remains 80 GiB.
        return SimpleNamespace(project=project, medical=medical, source=source, old=old, plan=plan,
                               journal=journal, evidence=evidence, pre=pre, raw=raw)

    @contextlib.contextmanager
    def native_boundaries(self, fixture):
        def evidence(plan, name):
            if plan["run_root"] == fixture.source:
                return fixture.evidence[name]
            path = plan["run_root"] / "DEBUG_stage_evidence" / name
            return {"format": "feedback_stage_completion_v1", "files": launch._bound_files(plan["run_root"], [path])}
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(launch.MODULES, {"debug_current.py": hashlib.sha256(NEW_MODULE).hexdigest()}, clear=True))
            stack.enter_context(mock.patch.object(launch, "audit_sources"))
            stack.enter_context(mock.patch.object(launch, "_verify_preparation_receipt"))
            stack.enter_context(mock.patch.object(launch, "_stage_evidence", side_effect=evidence))
            stack.enter_context(mock.patch.object(launch, "_verify_online_artifacts", return_value={"DEBUG": "native verifier boundary"}))
            stack.enter_context(mock.patch.object(online, "_verified_gnn_causality", return_value={"DEBUG": "native GNN verifier boundary"}))
            stack.enter_context(mock.patch.object(online, "_verified_preprocess_contract", return_value=({"DEBUG": "native preprocessor verifier boundary"}, {}, [])))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield

    def runner(self, fixture, calls, fail=None):
        def run(argv, **kwargs):
            argv = list(argv)
            calls.append((argv, dict(kwargs)))
            if "-c" in argv:
                return subprocess.CompletedProcess(argv, 0)
            command = next(item for item in fixture.plan["commands"] if argv[:len(item["argv"])] == item["argv"])
            name = command["name"]
            if name == "install_private_trainers":
                put(fixture.plan["package_destination"] / "training/nnUNetTrainer/debug_current.py", NEW_MODULE)
            if name == fail:
                raise RuntimeError("DEBUG child failure at " + name)
            if name not in upgrade.SETUP:
                put(fixture.plan["run_root"] / "DEBUG_stage_evidence" / name, ("DEBUG returned " + name).encode())
            return subprocess.CompletedProcess(argv, 0)
        return run

    def test_new_plan_keeps_original_quality_paths_and_has_no_retraining_or_preprocessing(self):
        plan = launch.build_plan(ROOT, ROOT / "DEBUG_medical", upgrade_bank_from="work/DEBUG_old")
        self.assertEqual(plan["run_root"], ROOT / "work/feedback_rawcp")
        self.assertEqual([item["name"] for item in plan["commands"]],
                         ["install_private_trainers", "environment", "bank", "feedback_contract", "check_full", "check_basic", "train_full", "train_basic"])
        for command in plan["commands"]:
            if "--paired-root" in command["argv"]:
                self.assertEqual(command["argv"][command["argv"].index("--paired-root")+1], "DEBUG_old/paired")
            self.assertNotIn("--overwrite", command["argv"])
            self.assertNotIn("--resume", command["argv"])
        self.assertEqual(plan["minimum_free_bytes"], 80 * 1024**3)
        self.assertNotIn("upgrade_source_root", launch.build_plan(ROOT, ROOT / "DEBUG_medical"))

    def test_overlap_recovery_and_external_config_are_rejected(self):
        for kwargs in ({"experiment_name": "DEBUG_old"}, {"run_root": "work/DEBUG_old/nested"},
                       {"recover_from": "work/another"}, {"train_config": "config/train.json"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                launch.build_plan(ROOT, ROOT / "DEBUG_medical", upgrade_bank_from="work/DEBUG_old", **kwargs)

    def test_saved_historical_runtime_is_accepted_not_replaced_with_current_hashes(self):
        with tempfile.TemporaryDirectory(prefix="dbgu_") as folder:
            f = self.fixture(Path(folder))
            with self.native_boundaries(f):
                identity = upgrade.validate_source(f.plan)
                self.assertEqual(identity["runtime_inventory"], f.journal["runtime_inventory"])
                self.assertNotEqual(identity["runtime_inventory"]["training/nnUNetTrainer/debug_current.py"], launch.MODULES["debug_current.py"])
                put(f.old["package_destination"] / "training/nnUNetTrainer/debug_current.py", b"DEBUG unexpected mutation")
                with self.assertRaisesRegex(ValueError, "Historical source runtime"):
                    upgrade.validate_source(f.plan)

    def test_source_history_rejects_checksums_running_locks_and_entered_segmentation(self):
        for problem in ("checksum", "running", "lock", "training", "checkpoint", "config"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory(prefix="dbgu_") as folder:
                f = self.fixture(Path(folder))
                if problem == "checksum":
                    put(f.source / "execution_journal.json", {**launch._read_json(f.source / "execution_journal.json"), "journal_sha256": "wrong"})
                elif problem == "running":
                    f.journal["stages"][-1]["status"] = "running"; launch._save_journal(f.source, f.journal)
                elif problem == "lock":
                    put(f.source / "recovery_execution.lock", {"pid": "DEBUG_NOT_RUNNING"})
                elif problem == "training":
                    f.journal["stages"].append({"name": "train_full", "status": "failed"}); launch._save_journal(f.source, f.journal)
                elif problem == "checkpoint":
                    put(Path(f.old["env_updates"]["nnUNet_results"]) / "DEBUG/checkpoint_latest.pth", b"DEBUG not loaded")
                else:
                    config = launch._read_json(f.old["train_config"]); config["model"]["hidden_dim"] = 64
                    put(f.old["train_config"], config)
                with self.native_boundaries(f), self.assertRaises(ValueError):
                    upgrade.validate_source(f.plan)
                self.assertFalse(f.plan["run_root"].exists())

    def test_view_materialization_preserves_payloads_but_isolates_metadata_and_unpack(self):
        with tempfile.TemporaryDirectory(prefix="dbgu_") as folder:
            f = self.fixture(Path(folder))
            before = inventory(f.source)
            with self.native_boundaries(f):
                rows, resources = upgrade.view_plan(f.plan)
                self.assertGreater(resources["native_numpy_unpack_bytes"], 0)
                self.assertEqual(resources["bank_raw_baseline_payload_bytes"], 8 * 10)
                self.assertEqual(resources["bank_raw_baseline_case_inventory"][0]["shape"], [1, 2, 2, 2])
                self.assertEqual(resources["bank_raw_baseline_bound_cohort"], "all_outer_train_cases_from_verified_preprocess_contract")
                self.assertGreater(resources["private_runtime_copy_upper_bytes"], 0)
                self.assertEqual(resources["required_free_bytes"], resources["native_numpy_unpack_bytes"]
                    + resources["bank_raw_baseline_payload_bytes"] + resources["private_runtime_copy_upper_bytes"]
                    + sum(row["bytes"] for row in rows if row["mode"] == "copy"))
                f.plan["run_root"].mkdir(parents=True)
                receipt = upgrade.prepare_views(f.plan, rows, resources)
                upgrade._verify_views(f.plan, receipt)
                for row in rows:
                    target, source = Path(row["target"]), Path(row["source"])
                    self.assertEqual(target.read_bytes(), source.read_bytes())
                    self.assertEqual(target.samefile(source), row["mode"] == "hardlink")
                target_pre = Path(f.plan["env_updates"]["nnUNet_preprocessed"]) / f.pre.name
                put(target_pre / "DEBUG_data/debug.npy", b"DEBUG native unpack belongs to NEW root only")
                put(target_pre / "DEBUG_data/debug.pkl", b"DEBUG downstream metadata mutation")
                self.assertEqual(inventory(f.source), before)
                with self.assertRaises(ValueError):
                    upgrade._verify_views(f.plan, receipt)

    def test_storage_rejection_contains_measured_free_required_and_unpack_cost(self):
        with tempfile.TemporaryDirectory(prefix="dbgu_") as folder:
            f = self.fixture(Path(folder))
            with mock.patch.object(upgrade.shutil, "disk_usage", return_value=SimpleNamespace(total=1, free=0)), \
                 self.assertRaisesRegex(OSError, "native_numpy_unpack_bytes"):
                upgrade.view_plan(f.plan)
            self.assertFalse(f.plan["run_root"].exists())

    def test_failure_before_bank_publication_then_resume_preserves_source_and_history(self):
        with tempfile.TemporaryDirectory(prefix="dbgu_") as folder:
            f = self.fixture(Path(folder))
            before, calls = inventory(f.source), []
            with self.native_boundaries(f), mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "DEBUG_ASSIGNED_GPU"}):
                with self.assertRaisesRegex(RuntimeError, "DEBUG child failure at bank"):
                    launch.execute_plan(f.plan, runner=self.runner(f, calls, fail="bank"))
                original = (f.plan["run_root"] / "execution_journal.json").read_bytes()
                first_count = len(calls)
                launch.execute_plan(f.plan, runner=self.runner(f, calls), resume_experiment=True)
                new_calls = [argv for argv, _ in calls[first_count:]]
                self.assertFalse(any("install_onlinecp_custom_trainers.py" in " ".join(argv) for argv in new_calls))
                self.assertFalse(any("tools.paired_benchmark" in argv for argv in new_calls))
                self.assertFalse(any("plan" in argv for argv in new_calls))
                self.assertTrue(any(path.read_bytes() == original for path in (f.plan["run_root"] / "upgrade/journal_history").iterdir()))
                identity = upgrade.validate_source(f.plan)
                journal = upgrade.load_upgrade_journal(f.plan, identity)
                self.assertTrue(journal["complete"])
                self.assertEqual([row["status"] for row in journal["stages"] if row["name"] == "bank"], ["failed", "completed"])
                count = len(calls)
                launch.execute_plan(f.plan, runner=self.runner(f, calls), resume_experiment=True)
                self.assertEqual(len(calls) - count, 1)  # Only actual allocation GPU preflight remains.
                for _, kwargs in calls:
                    self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "DEBUG_ASSIGNED_GPU")
                    self.assertEqual(kwargs["env"]["nnUNet_results"], f.plan["env_updates"]["nnUNet_results"])
            self.assertEqual(inventory(f.source), before)

    def test_incomplete_private_setup_is_preserved_and_refused_on_resume(self):
        with tempfile.TemporaryDirectory(prefix="dbgu_") as folder:
            f = self.fixture(Path(folder))
            calls = []
            with self.native_boundaries(f):
                with self.assertRaises(RuntimeError):
                    launch.execute_plan(f.plan, runner=self.runner(f, calls, fail="environment"))
                before = inventory(f.plan["run_root"])
                with self.assertRaises(ValueError):
                    launch.execute_plan(f.plan, runner=self.runner(f, calls), resume_experiment=True)
                self.assertEqual(inventory(f.plan["run_root"]), before)

    def test_source_resource_defaults_can_differ_but_original_config_bytes_are_copied(self):
        with tempfile.TemporaryDirectory(prefix="dbgu_") as folder:
            f = self.fixture(Path(folder))
            original = f.old["train_config"].read_bytes()
            current = launch._read_json(f.project / "config/train.json")
            current["training"]["batch_size_candidates"] = [1, 2, 4]
            put(f.project / "config/train.json", current)
            with self.native_boundaries(f):
                identity = upgrade.validate_source(f.plan)
                self.assertEqual([row["path"] for row in identity["source_config_differences"]],
                                 ["training.batch_size_candidates"])
                rows, resources = upgrade.view_plan(f.plan)
                f.plan["run_root"].mkdir(parents=True)
                receipt = upgrade.prepare_views(f.plan, rows, resources)
                self.assertEqual(f.plan["train_config"].read_bytes(), original)
                self.assertEqual(receipt["source_config_differences"], identity["source_config_differences"])
                upgrade._verify_views(f.plan, receipt)
            self.assertEqual(f.old["train_config"].read_bytes(), original)

    def test_forged_incomplete_view_receipt_does_not_hide_missing_payloads(self):
        with tempfile.TemporaryDirectory(prefix="dbgu_") as folder:
            f = self.fixture(Path(folder))
            with self.native_boundaries(f):
                rows, resources = upgrade.view_plan(f.plan)
                f.plan["run_root"].mkdir(parents=True)
                receipt = upgrade.prepare_views(f.plan, rows, resources)
                receipt["files"] = receipt["files"][:-1]
                put(f.plan["run_root"] / "upgrade/views.json", receipt)
                with self.assertRaisesRegex(ValueError, "source manifest"):
                    upgrade._verify_views(f.plan, receipt)

    def test_new_segmentation_attempt_without_checkpoint_is_not_restarted(self):
        with tempfile.TemporaryDirectory(prefix="dbgu_") as folder:
            f = self.fixture(Path(folder))
            calls = []
            with self.native_boundaries(f):
                with self.assertRaisesRegex(RuntimeError, "DEBUG child failure at train_full"):
                    launch.execute_plan(f.plan, runner=self.runner(f, calls, fail="train_full"))
                count = len(calls)
                # Native checkpoint resolution is intentionally NOT mocked.
                with self.assertRaises((ValueError, FileNotFoundError, RuntimeError)):
                    launch.execute_plan(f.plan, runner=self.runner(f, calls), resume_experiment=True)
                full = next(row["argv"] for row in f.plan["commands"] if row["name"] == "train_full")
                self.assertFalse(any(argv[:len(full)] == full for argv, _ in calls[count:]))

    def test_dry_run_is_read_only_and_never_executes_children(self):
        with tempfile.TemporaryDirectory(prefix="dbgu_") as folder:
            f = self.fixture(Path(folder))
            before = inventory(f.project)
            with self.native_boundaries(f), mock.patch.object(upgrade.subprocess, "run", side_effect=AssertionError("DEBUG dry-run executed child")):
                upgrade.dry_run_upgrade(f.plan)
            self.assertEqual(inventory(f.project), before)
            self.assertFalse(f.plan["run_root"].exists())


class BankUpgradeResourceAndPathDebugTests(unittest.TestCase):
    """DEBUG metadata-only guards; no real medical data, GPU or training."""

    def test_historical_resource_changes_are_recorded_without_rewriting_inputs(self):
        import copy
        from tools.feedback_bank_upgrade import source_config_differences
        original = {"training": {"batch_size_candidates": [1, 2, 4], "num_workers": 2,
                    "gradient_accumulation_steps": 4, "target_effective_batch_size": 32},
                    "model": {"hidden_dim": 128}, "graph": {"adaptive_roi_max_voxels": 12000000}}
        current = copy.deepcopy(original)
        current["training"].update(batch_size_candidates="powers_of_two_to_cohort", num_workers="auto",
                                   gradient_accumulation_steps=1, target_effective_batch_size=None)
        current["graph"]["adaptive_roi_max_voxels"] = 8000000
        before = json.dumps([original, current], sort_keys=True)
        changes = source_config_differences(original, current)
        self.assertEqual(len(changes), 5)
        self.assertTrue(all(row["applied_to_historical_training"] is False for row in changes))
        self.assertEqual(changes[0]["classification"], "previously_verified_full_graph_allocation_ceiling")
        self.assertEqual(before, json.dumps([original, current], sort_keys=True))

    def test_structural_unknown_and_training_objective_changes_report_exact_path(self):
        import copy
        from tools.feedback_bank_upgrade import source_config_differences
        original = {"model": {"hidden_dim": 128}, "cache": {"source_pad": 2},
                    "training": {"epochs": 40, "lr": 0.0001}, "labels": {"tumor": 2},
                    "generation": {"num_candidates": 128},
                    "graph": {"context_radius_mm": 28.0, "adaptive_roi_max_voxels": 8000000}}
        mutations = [("model", "hidden_dim", 64), ("cache", "source_pad", 4),
                     ("training", "epochs", 39), ("training", "lr", 0.001),
                     ("labels", "tumor", 3), ("generation", "num_candidates", 64),
                     ("graph", "context_radius_mm", 27.0), ("training", "unknown_resource", 1)]
        for section, key, value in mutations:
            with self.subTest(path=f"{section}.{key}"):
                current = copy.deepcopy(original)
                current[section][key] = value
                with self.assertRaisesRegex(ValueError, section + r"\." + key):
                    source_config_differences(original, current)
        smaller = copy.deepcopy(original)
        smaller["graph"]["adaptive_roi_max_voxels"] = 4000000
        with self.assertRaisesRegex(ValueError, "allocation ceiling"):
            source_config_differences(smaller, original)

    def test_numpy_inventory_reads_header_without_array_loading(self):
        from tools.feedback_bank_upgrade import _preprocessed_data_shape
        with tempfile.TemporaryDirectory(prefix="DEBUG_bank_shape_") as directory:
            path = Path(directory) / "debug.npz"
            np.savez_compressed(path, data=np.zeros((1, 2, 3, 4), np.float32),
                                seg=np.zeros((1, 2, 3, 4), np.int16))
            with mock.patch.object(np, "load", side_effect=AssertionError("DEBUG full array loading forbidden")):
                self.assertEqual(_preprocessed_data_shape(path), [1, 2, 3, 4])

    def test_blosc_inventory_opens_read_only_metadata(self):
        from types import SimpleNamespace
        from tools.feedback_bank_upgrade import _preprocessed_data_shape
        with mock.patch("blosc2.open", return_value=SimpleNamespace(shape=(1, 2, 3, 4))) as opened:
            self.assertEqual(_preprocessed_data_shape(Path("DEBUG_data.b2nd")), [1, 2, 3, 4])
            opened.assert_called_once_with("DEBUG_data.b2nd", mode="r")

    def test_parent_directory_link_is_rejected_before_native_writes(self):
        import errno
        from tools.feedback_bank_upgrade import _owned_view_target
        with tempfile.TemporaryDirectory(prefix="DEBUG_bank_link_") as directory:
            base = Path(directory)
            source, target = base / "source", base / "new"
            source.mkdir()
            target.mkdir()
            link = target / "preprocessed"
            try:
                link.symlink_to(source, target_is_directory=True)
            except OSError as exc:
                if getattr(exc, "winerror", None) == 1314 or exc.errno in {errno.EPERM, errno.EACCES}:
                    self.skipTest("DEBUG host cannot create a directory symlink")
                raise
            with self.assertRaisesRegex(ValueError, "directory symlink"):
                _owned_view_target({"run_root": target}, link / "future_unpacked.npy")
            self.assertEqual(list(source.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
