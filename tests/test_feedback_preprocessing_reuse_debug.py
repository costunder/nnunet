"""DEBUG native file-contract/I/O tests; never real medical preprocessing.

Tiny generated NIfTI/NPZ fixtures exercise actual raw/preprocess hash validators,
hardlinks/copies, ownership, interruption and receipt publication. Split planning
and the launcher's stage-evidence dispatcher are explicit boundary fixtures; no
native planner, real dataset, GNN training or GPU execution is claimed here.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import tempfile
import threading
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

from hiercp.schema import GraphBuildConfig
from tools import feedback_preprocessing_reuse as reuse
from tools import feedback_fresh_execution as fresh
from tools import online_cp_benchmark as online
from tools import run_feedback_experiment as launch


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, default=str), encoding="utf-8")


class PreprocessingReuseDebugTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="DEBUG_preprocessing_reuse_")
        self.project = Path(self.temporary.name).resolve()
        self.medical = self.project / "medical"
        self.old_root, self.new_root = self.project / "work/old", self.project / "work/new"
        self.split = {"train": ["DEBUG_case_0"], "val": ["DEBUG_case_1"]}
        self.nn_config = {"dataset": {"plans": "DEBUGPlans", "configuration": "DEBUG", "planner": "DEBUGPlanner"}}
        write_json(self.project / "config/nnunet.json", self.nn_config)
        for name in ("online_cp_feedback.json", "online_cp_feedback_gnn.json"):
            write_json(self.project / "config" / name, {"DEBUG_metadata_only": True})
        legacy_graph = asdict(GraphBuildConfig())
        legacy_graph.pop("patient_graph_contract")
        original = {"method": "hiercp-full", "seed": 42, "labels": {"liver": 1, "tumor": 2},
                    "graph": legacy_graph, "training": {"batch_size": 8}}
        current = copy.deepcopy(original)
        current["graph"]["patient_graph_contract"] = "patient_source_content_population_v2"
        current["training"]["batch_size"] = "auto"
        self.old, self.plan = self._plan(self.old_root, original), self._plan(self.new_root, current)
        self.plan["preprocessing_source_root"] = self.old_root
        self.layouts = {self.old_root: self._layout(self.old), self.new_root: self._layout(self.plan)}
        self.patches = [mock.patch.object(reuse, "_layout", side_effect=lambda plan: self.layouts[Path(plan["run_root"])]),
                        mock.patch.object(online, "outer_split", side_effect=lambda *_: copy.deepcopy(self.split)),
                        mock.patch.object(launch, "_stage_evidence", side_effect=self._evidence)]
        for patch in self.patches:
            patch.start()
        self.addCleanup(lambda: [patch.stop() for patch in reversed(self.patches)])
        self.addCleanup(self.temporary.cleanup)
        self._native_source()
        self._journal(self.old, source=True)
        self._journal(self.plan, source=False)

    def _plan(self, root, train_config):
        root.mkdir(parents=True)
        path = root / "train_config.json"
        write_json(path, train_config)
        package = root / "runtime/nnunetv2"
        for relative in ("__init__.py", "training/nnUNetTrainer/nnUNetTrainer.py"):
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# DEBUG native-package identity fixture only\n", encoding="utf-8")
        return {"project_root": self.project, "medical_root": self.medical, "run_root": root,
                "train_config": path, "package_destination": package, "outer_fold": 0, "dataset_id": 901,
                "seed": 42, "python_executable": "DEBUG_NOT_EXECUTED", "minimum_free_bytes": 0,
                "env_updates": {name: str(root / "online/nnunetv2" / name)
                                for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")},
                "commands": [{"name": "split", "argv": ["DEBUG_boundary"]},
                             {"name": "plan", "argv": ["DEBUG_worker_boundary"]}]}

    def _layout(self, plan):
        root = plan["run_root"]
        paired = root / "paired"
        write_json(paired / "outer_splits.json", {"DEBUG_fixed_split": self.split})
        return online.Layout(project=self.project, medical=self.medical, data=self.medical / "Data",
            paired=paired, online=root / "online", source_work=self.medical / "work",
            train_config=plan["train_config"], nnunet_config=self.project / "config/nnunet.json",
            outer_splits=paired / "outer_splits.json", nnroot=root / "online/nnunetv2",
            raw=Path(plan["env_updates"]["nnUNet_raw"]), preprocessed=Path(plan["env_updates"]["nnUNet_preprocessed"]),
            results=Path(plan["env_updates"]["nnUNet_results"]), logs=root / "online/logs")

    def _native_source(self):
        layout = self.layouts[self.old_root]
        for index in range(2):
            name = f"DEBUG_case_{index}"
            for folder, suffix, array in (("image", "_0000.nii.gz", np.arange(64, dtype=np.float32).reshape(4, 4, 4) + index),
                                          ("labels", ".nii.gz", np.ones((4, 4, 4), dtype=np.int16))):
                path = layout.data / folder / (name + suffix)
                path.parent.mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(array, np.eye(4)), path)
        config = launch._read_json(layout.train_config)
        _, cases, dataset, raw_contract = online._raw_contract(layout, 0, config, 901, "hardlink")
        raw = online.raw_dataset_dir(layout, 901, 0)
        for directory in ("imagesTr", "labelsTr", "imagesTs"):
            (raw / directory).mkdir(parents=True)
        for case in cases:
            os.link(case.image, raw / "imagesTr" / (case.case_id + "_0000.nii.gz"))
            os.link(case.label, raw / "labelsTr" / (case.case_id + ".nii.gz"))
        write_json(raw / "dataset.json", dataset)
        write_json(raw / online.RAW_MARKER_NAME, raw_contract)
        pre = online.preprocessed_dataset_dir(layout, 901, 0)
        write_json(pre / "DEBUGPlans.json", {"configurations": {"DEBUG": {"data_identifier": "DEBUG_data"}}})
        write_json(pre / "dataset.json", dataset)
        write_json(pre / "dataset_fingerprint.json", {"DEBUG_synthetic_only": True})
        write_json(pre / "splits_final.json", [self.split])
        (pre / "DEBUG_data").mkdir()
        (pre / "gt_segmentations").mkdir()
        for case in cases:
            np.savez_compressed(pre / "DEBUG_data" / (case.case_id + ".npz"),
                data=np.zeros((1, 4, 4, 4), np.float32), seg=np.ones((1, 4, 4, 4), np.int16))
            with (pre / "DEBUG_data" / (case.case_id + ".pkl")).open("wb") as stream:
                pickle.dump({"DEBUG_fixture": True}, stream)
            os.link(case.label, pre / "gt_segmentations" / (case.case_id + ".nii.gz"))
        # Unbound old unpack cache and unrelated old GNN MUST NOT be copied.
        np.save(pre / "DEBUG_data/DEBUG_case_0.npy", np.full((1, 4, 4, 4), 999., np.float32))
        old_gnn = self.old_root / "paired/folds/fold_0/gnn/model.pt"
        old_gnn.parent.mkdir(parents=True)
        old_gnn.write_bytes(b"DEBUG obsolete learned artifact: never load or copy")
        inputs, split, cases = online._preprocess_input_contract(layout, 0, config, self.nn_config, 901)
        outputs = online._preprocess_output_record(layout, 0, self.nn_config, 901, cases, split)
        write_json(pre / online.PREPROCESS_MARKER_NAME, online._preprocess_marker_payload(inputs, outputs))

    def _evidence(self, plan, name):
        root = Path(plan["run_root"])
        layout = self.layouts[root]
        paths = ([layout.outer_splits] if name == "split" else
                 [online.raw_dataset_dir(layout, 901, 0) / online.RAW_MARKER_NAME,
                  online.preprocessed_dataset_dir(layout, 901, 0) / online.PREPROCESS_MARKER_NAME])
        return {"format": "feedback_stage_completion_v1", "files": launch._bound_files(root, paths)}

    def _journal(self, plan, *, source):
        root = plan["run_root"]
        inputs = launch._resume_inputs(plan)
        stages = [{"name": "split", "status": "completed", "input_files": inputs,
                   "completion_evidence": self._evidence(plan, "split")}]
        stages.append({"name": "plan", "status": "completed" if source else "running", "input_files": inputs,
                       **({"completion_evidence": self._evidence(plan, "plan")} if source else {})})
        journal = {"format": fresh.FORMAT, "plan_sha256": launch._json_sha256(plan), "source_identity": {},
                   "input_files": inputs, "runtime_inventory": fresh.native_inventory(plan["package_destination"]),
                   "stages": stages, "training_started": False, "complete": False}
        journal["journal_sha256"] = launch._json_sha256(journal)
        write_json(root / "launch_plan.json", plan)
        write_json(root / "execution_journal.json", journal)

    def test_native_complete_derivative_keeps_manifests_and_never_loads_old_gnn(self):
        before = {str(path): launch._file_sha256(path) for path in self.old_root.rglob("*") if path.is_file()}
        with mock.patch("hiercp.tensor.torch_load_compat", side_effect=AssertionError("must not load old weights")):
            receipt = reuse.execute_reuse(self.plan)
        self.assertEqual(receipt["source_output_manifest_sha256"], receipt["current_output_manifest_sha256"])
        self.assertFalse(receipt["learned_artifacts_reused"])
        self.assertFalse(any(Path(row["target"]).suffix in {".npy", ".pt"} for row in receipt["files"]))
        self.assertEqual(reuse.execute_reuse(self.plan), receipt)
        self.assertEqual(before, {str(path): launch._file_sha256(path) for path in self.old_root.rglob("*") if path.is_file()})

    def test_partial_materialization_retries_only_with_owned_manifest(self):
        original = reuse._materialize
        state, lock = {"calls": 0}, threading.Lock()
        def interrupt(plan, row):
            with lock:
                state["calls"] += 1
                number = state["calls"]
            if number == 2:
                raise RuntimeError("DEBUG interrupted native view")
            return original(plan, row)
        with mock.patch.object(reuse, "_materialize", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "DEBUG interrupted"):
                reuse.execute_reuse(self.plan)
        self.assertTrue((self.new_root / "preprocessing_reuse/attempt.json").is_file())
        self.assertFalse((self.new_root / "preprocessing_reuse/receipt.json").exists())
        reuse.execute_reuse(self.plan)
        reuse.verify_reuse(self.plan)

    def test_foreign_byte_identical_file_is_not_adopted(self):
        row = reuse.validate_reuse(self.plan, self.plan["package_destination"])["files"][0]
        target = Path(row["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(row["source"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "[Uu]nowned"):
            reuse.execute_reuse(self.plan)
        self.assertTrue(target.is_file())
        self.assertFalse((self.new_root / "preprocessing_reuse/attempt.json").exists())

    def test_marker_published_without_receipt_is_recoverable(self):
        publish = reuse._publish_json
        def interrupt(plan, path, value):
            if path.name == "receipt.json":
                raise RuntimeError("DEBUG receipt interruption")
            return publish(plan, path, value)
        with mock.patch.object(reuse, "_publish_json", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "DEBUG receipt interruption"):
                reuse.execute_reuse(self.plan)
        marker = online.preprocessed_dataset_dir(self.layouts[self.new_root], 901, 0) / online.PREPROCESS_MARKER_NAME
        self.assertTrue(marker.is_file())
        reuse.execute_reuse(self.plan)

    def test_native_raw_tamper_and_native_code_change_are_rejected(self):
        package = self.plan["package_destination"] / "__init__.py"
        original = package.read_bytes()
        package.write_bytes(b"# DEBUG changed native implementation")
        with self.assertRaisesRegex(ValueError, "Native nnUNet implementation changed"):
            reuse.validate_reuse(self.plan, self.plan["package_destination"])
        package.write_bytes(original)
        source = online.raw_dataset_dir(self.layouts[self.old_root], 901, 0) / "dataset.json"
        source.write_text("{}", encoding="utf-8")
        with self.assertRaises(online.OnlineBenchmarkError):
            reuse.validate_reuse(self.plan)

    def test_config_projection_is_explicit_and_preserves_native_labels(self):
        old, current = launch._read_json(self.old["train_config"]), launch._read_json(self.plan["train_config"])
        changes = reuse.configuration_differences(old, current)
        self.assertEqual({row["path"] for row in changes}, {"graph.patient_graph_contract", "training.batch_size"})
        for changed in ({**current, "mystery": {}}, {**current, "training": {"mystery": 1}},
                        {**current, "labels": {"liver": 1, "tumor": 3}}):
            with self.assertRaises(ValueError):
                reuse.configuration_differences(old, changed)

    def test_early_validation_without_new_split_and_late_split_mismatch(self):
        split = self.layouts[self.new_root].outer_splits
        content = split.read_bytes()
        split.unlink()
        self.assertEqual(reuse.validate_reuse(self.plan)["actual_new_split_check"], "required_at_plan_worker")
        split.write_bytes(content + b"\n")
        # Renew only the DEBUG new split boundary receipt; source stays unchanged.
        self._journal(self.plan, source=False)
        with self.assertRaisesRegex(ValueError, "NEW actual outer split"):
            reuse.execute_reuse(self.plan)

    def test_partial_copy_never_publishes_partial_final_file(self):
        copy_file = reuse.shutil.copyfileobj
        state, lock = {}, threading.Lock()
        def interrupted(incoming, outgoing, length):
            with lock:
                first = "source" not in state
                if first:
                    state["source"] = incoming.name
            if first:
                outgoing.write(b"DEBUG interrupted copy bytes")
                raise OSError("DEBUG copy interruption")
            return copy_file(incoming, outgoing, length)
        with mock.patch.object(reuse.shutil, "copyfileobj", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "DEBUG copy interruption"):
                reuse.execute_reuse(self.plan)
        attempt = launch._read_json(self.new_root / "preprocessing_reuse/attempt.json")
        failed = next(row for row in attempt["files"] if row["source"] == state["source"])
        self.assertFalse(Path(failed["target"]).exists())
        # A power interruption may leave staging bytes outside native dirs.
        # A retry must not adopt, overwrite or delete this unrelated temp file.
        orphan = self.new_root / "preprocessing_reuse/copy_staging/DEBUG_orphan.tmp"
        orphan.write_bytes(b"DEBUG orphan: preserve")
        reuse.execute_reuse(self.plan)
        self.assertEqual(orphan.read_bytes(), b"DEBUG orphan: preserve")
        reuse.verify_reuse(self.plan)

    def test_owned_partial_content_change_is_preserved_and_rejected(self):
        publish = reuse._publish_json
        def interrupt(plan, path, value):
            if path.name == "receipt.json":
                raise RuntimeError("DEBUG receipt interruption")
            return publish(plan, path, value)
        with mock.patch.object(reuse, "_publish_json", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "DEBUG receipt interruption"):
                reuse.execute_reuse(self.plan)
        old = online.raw_dataset_dir(self.layouts[self.old_root], 901, 0) / "dataset.json"
        target = online.raw_dataset_dir(self.layouts[self.new_root], 901, 0) / "dataset.json"
        original = old.read_bytes()
        target.write_bytes(b"DEBUG changed owned metadata")
        with self.assertRaisesRegex(ValueError, "content changed"):
            reuse.execute_reuse(self.plan)
        self.assertEqual(target.read_bytes(), b"DEBUG changed owned metadata")
        self.assertEqual(old.read_bytes(), original)

    def test_source_cp_only_registry_changes_are_recorded_not_checkpoint_approval(self):
        path = self.project / "config/online_cp_feedback.json"
        write_json(path, {"DEBUG_cp_only_change": True})
        result = reuse.validate_reuse(self.plan, self.plan["package_destination"])
        changes = result["source_identity"]["non_preprocessing_input_registry_differences"]
        self.assertEqual([row["path"] for row in changes], ["config/online_cp_feedback.json"])
        self.assertFalse(result["learned_artifacts_reused"])
        # NEW execution still needs its own exact, current four-file binding.
        with self.assertRaisesRegex(ValueError, "NEW fresh preprocessing worker"):
            reuse.execute_reuse(self.plan)
        self._journal(self.plan, source=False)
        reuse.execute_reuse(self.plan)
        current = launch._resume_inputs(self.old)
        with self.assertRaisesRegex(ValueError, "registry"):
            reuse._source_input_differences(self.old, {**current, "config/unknown.json": "0" * 64})
        for key in ("config/nnunet.json", self.old["train_config"].relative_to(self.project).as_posix()):
            with self.assertRaisesRegex(ValueError, "native preprocessing input changed"):
                reuse._source_input_differences(self.old, {**current, key: "0" * 64})

    def test_persistent_source_lock_probes_occupancy_without_rewriting(self):
        from tools.feedback_stage_execution import run_lock
        with run_lock(self.old_root):
            with self.assertRaises((RuntimeError, OSError)):
                reuse.validate_reuse(self.plan)
        lock = self.old_root / "feedback_execution.lock"
        original = lock.read_bytes()
        reuse.validate_reuse(self.plan)
        self.assertEqual(lock.read_bytes(), original)
        (self.old_root / "feedback_preparation_execution.lock").write_bytes(b"DEBUG legacy owner")
        with self.assertRaisesRegex(ValueError, "legacy"):
            reuse.validate_reuse(self.plan)

    def test_receipt_storage_arithmetic_is_checked_not_merely_present(self):
        receipt = reuse.execute_reuse(self.plan)
        receipt["storage"]["required_free_bytes"] += 1
        write_json(self.new_root / "preprocessing_reuse/receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "[Ss]torage"):
            reuse.verify_reuse(self.plan)

    def test_source_gt_symlink_requires_the_original_label_identity(self):
        gt = online.preprocessed_dataset_dir(self.layouts[self.old_root], 901, 0) / "gt_segmentations/DEBUG_case_0.nii.gz"
        original = self.medical / "Data/labels/DEBUG_case_0.nii.gz"
        gt.unlink()  # Only this test's new GT hardlink, not the original label.
        try:
            gt.symlink_to(original)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                self.skipTest("DEBUG symlink test requires Windows symlink privilege; no elevation")
            raise
        reuse.validate_reuse(self.plan)
        gt.unlink()
        gt.symlink_to(self.medical / "Data/labels/DEBUG_case_1.nii.gz")
        with self.assertRaisesRegex(ValueError, "ground.truth.*symlink|GT.*symlink"):
            reuse.validate_reuse(self.plan)

    def test_native_blosc2_payloads_use_complete_data_and_segmentation_manifest(self):
        try:
            import blosc2
        except ImportError:
            self.skipTest("DEBUG native B2ND codec dependency unavailable")
        if not hasattr(blosc2, "asarray"):
            self.skipTest("DEBUG installed blosc2 lacks the native B2ND array API")
        layout = self.layouts[self.old_root]
        pre = online.preprocessed_dataset_dir(layout, 901, 0)
        for name in ("DEBUG_case_0", "DEBUG_case_1"):
            (pre / "DEBUG_data" / (name + ".npz")).unlink()  # Owned tiny fixture only.
            for suffix, data in ((".b2nd", np.zeros((1, 4, 4, 4), np.float32)),
                                 ("_seg.b2nd", np.ones((1, 4, 4, 4), np.int16))):
                path = pre / "DEBUG_data" / (name + suffix)
                array = blosc2.asarray(data, urlpath=str(path), mode="w")
                np.testing.assert_array_equal(array[:], data)
                del array
        config = launch._read_json(layout.train_config)
        inputs, split, cases = online._preprocess_input_contract(layout, 0, config, self.nn_config, 901)
        outputs = online._preprocess_output_record(layout, 0, self.nn_config, 901, cases, split)
        self.assertEqual(outputs["storage_format"], "blosc2")
        write_json(pre / online.PREPROCESS_MARKER_NAME, online._preprocess_marker_payload(inputs, outputs))
        self._journal(self.old, source=True)
        receipt = reuse.execute_reuse(self.plan)
        self.assertEqual(receipt["storage"]["native_numpy_unpack_bytes_all_cases"], 0)
        self.assertEqual(sum(Path(row["target"]).suffix == ".b2nd" for row in receipt["files"]), 4)
        reuse.verify_reuse(self.plan)


if __name__ == "__main__":
    unittest.main()
