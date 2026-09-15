"""Synthetic CPU DEBUG of checkpoint/prediction/evaluation provenance.

The launcher proof and native predictor are explicit injected boundaries: no
nnU-Net training or medical data. NIfTI I/O, checkpoint tensors, prediction
receipts, matching/statistics and evaluation completion use actual code.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from custom_trainers.onlinecp_curriculum_contract import file_sha256
from tools import evaluate_feedback_experiment as producer
from tools import online_eval_v2 as evaluator
from tools.train_online_feedback import TRAINERS


class FeedbackEvaluationProducerDebugTests(unittest.TestCase):
    def setUp(self):
        # Keep the real, long native trainer basename without exceeding the
        # Windows torch zip writer's path budget in this DEBUG workspace.
        self.tmp = tempfile.TemporaryDirectory(prefix="DEBUG_eval_")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.project, self.medical = self.base / "project", self.base / "Medical"
        self.run = self.project / "work/DEBUG_experiment"
        self.output = self.run / "evaluation_feedback_v5"
        self.dataset = "Dataset900_LiverOnlineCP_OF0"
        self.cases = ["debug_case_1", "debug_case_2"]
        self.config = {"dataset": {"plans": "DebugPlans", "configuration": "3d_fullres"},
                       "preprocess": {"processes": 4}}
        self.write(self.project / "config/nnunet.json", self.config)
        self.write(self.run / "paired/outer_splits.json", {"splits": [{"train": ["debug_train"], "val": self.cases}]})
        self.plan = {"project_root": self.project, "medical_root": self.medical, "run_root": self.run,
            "outer_fold": 0, "dataset_id": 900, "package_destination": self.run / "runtime/nnunetv2",
            "env_updates": {name: str(self.run / "online/nnunetv2" / name) for name in
                            ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")}}
        self.raw = Path(self.plan["env_updates"]["nnUNet_raw"]) / self.dataset
        self.truth = np.ones((8, 8, 8), dtype=np.uint8)
        self.truth[2:4, 2:4, 2:4] = 2
        for case_id in self.cases:
            image = self.raw / "imagesTr" / (case_id + "_0000.nii.gz")
            original_image = self.medical / "Data/image" / image.name
            label = self.medical / "Data/labels" / (case_id + ".nii.gz")
            registered = self.raw / "labelsTr" / (case_id + ".nii.gz")
            for path in (image, original_image, label, registered):
                path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(np.full((8, 8, 8), 100., np.float32), np.eye(4)), image)
            shutil.copyfile(image, original_image)
            nib.save(nib.Nifti1Image(self.truth, np.eye(4)), label)
            shutil.copyfile(label, registered)
        bank_plans = self.run / "online/DEBUG_plans.json"
        bank_dataset = self.run / "online/DEBUG_dataset.json"
        self.write(bank_plans, {"debug_only": True, "configurations": {"3d_fullres": {}}})
        self.write(bank_dataset, {"channel_names": {"0": "CT"}, "labels": {"background": 0, "liver": 1, "tumor": 2},
                                  "file_ending": ".nii.gz"})
        self.bank = {"dataset_name": self.dataset, "files": {"plans": {"path": str(bank_plans)},
                                                            "dataset": {"path": str(bank_dataset)}}}
        self.checkpoints = {}
        for arm in producer.ARMS:
            folder = Path(self.plan["env_updates"]["nnUNet_results"]) / self.dataset / (TRAINERS[arm] + "__DebugPlans__3d_fullres")
            (folder / "fold_0").mkdir(parents=True)
            shutil.copyfile(bank_plans, folder / "plans.json")
            shutil.copyfile(bank_dataset, folder / "dataset.json")
            checkpoint = folder / "fold_0/checkpoint_final.pth"
            network = torch.nn.Linear(2, 1)
            torch.save({"current_epoch": 250, "trainer_name": TRAINERS[arm], "network_weights": network.state_dict(),
                "onlinecp_curriculum_resume": {"bank_identity": self.bank},
                "init_args": {"configuration": "3d_fullres"}, "inference_allowed_mirroring_axes": (0, 1, 2)}, checkpoint)
            self.checkpoints[arm] = checkpoint
        self.runtime_marker = self.plan["package_destination"] / "DEBUG_runtime.json"
        self.write(self.runtime_marker, {"debug_only": True, "injected_native_predictor": True})
        self.proof = {"plan": self.plan, "bank_identity": self.bank, "checkpoints": self.checkpoints,
            "validation_case_ids": self.cases, "runtime_inventory": {"DEBUG_runtime.json": file_sha256(self.runtime_marker)},
            "journal_sha256": "0" * 64}
        self.proof["raw_input_contract"] = {"source_cases": [
            {"case_id": case_id,
             "image_sha256": file_sha256(self.medical / "Data/image" / (case_id + "_0000.nii.gz")),
             "label_sha256": file_sha256(self.medical / "Data/labels" / (case_id + ".nii.gz"))}
            for case_id in self.cases]}
        self.calls = []
        self.fail_native = False
        self.wrong_weights = False
        self.change_checkpoint = False
        self.change_memory_weights = False
        self.change_runtime = False
        self.change_original_image = False
        self.missing_prediction = False
        self.wrong_settings = False
        self.args = producer.parser().parse_args(["--experiment-root", str(self.run), "--output-dir", str(self.output), "--device", "cpu"])

    @staticmethod
    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def factory(self, plan, device):
        owner = self
        class DebugPredictor:
            tile_step_size, use_gaussian, use_mirroring = .5, True, True
            def initialize_from_trained_model_folder(self, folder, use_folds, checkpoint_name):
                owner.assertEqual(use_folds, (0,))
                owner.assertEqual(checkpoint_name, "checkpoint_final.pth")
                folder = Path(folder)
                state = torch.load(folder / "fold_0" / checkpoint_name, map_location="cpu", weights_only=False)
                self.trainer_name = state["trainer_name"]
                self.arm = next(arm for arm in producer.ARMS if TRAINERS[arm] == self.trainer_name)
                self.list_of_parameters = [copy.deepcopy(state["network_weights"])]
                if owner.wrong_weights:
                    self.list_of_parameters[0]["weight"][0, 0] += 1
                self.network = torch.nn.Linear(2, 1)
                # DEBUG deferred-load boundary: the real native initializer may
                # store weights without installing them until prediction.
                self.allowed_mirroring_axes = state["inference_allowed_mirroring_axes"]
                self.plans_manager = SimpleNamespace(plans=json.loads((folder / "plans.json").read_text()))
                self.dataset_json = json.loads((folder / "dataset.json").read_text())
                if owner.wrong_settings:
                    self.tile_step_size = 1.

            def predict_from_files(self, images, output, **kwargs):
                for key, value in self.network.state_dict().items():
                    torch.testing.assert_close(value, self.list_of_parameters[0][key], rtol=0, atol=0)
                cases = [Path(row[0]).name.removesuffix("_0000.nii.gz") for row in images]
                owner.calls.append((self.arm, cases, kwargs))
                owner.assertFalse(kwargs["overwrite"])
                owner.assertEqual(kwargs["num_parts"], 1)
                owner.assertEqual(kwargs["part_id"], 0)
                owner.assertEqual(kwargs["num_processes_preprocessing"], 4)
                folder = Path(output)
                with (folder / "predict_from_raw_data_args.json").open("x") as handle:
                    json.dump({"debug_predictor_boundary": True, "cases": cases}, handle)
                for index, case_id in enumerate(cases):
                    if owner.missing_prediction and index == 1:
                        continue  # DEBUG malformed producer return; production must reject.
                    values = owner.truth.copy()
                    if self.arm == "basic":
                        values[2, 2:4, 2:4] = 1
                    target = folder / (case_id + ".nii.gz")
                    owner.assertFalse(target.exists())
                    nib.save(nib.Nifti1Image(values, np.eye(4)), target)
                    if owner.fail_native and index == 0:
                        owner.fail_native = False
                        raise RuntimeError("DEBUG native export interrupted; no producer case receipt")
                if owner.change_checkpoint:
                    with owner.checkpoints[self.arm].open("ab") as handle:
                        handle.write(b"DEBUG changed checkpoint during prediction")
                if owner.change_memory_weights:
                    with torch.no_grad():
                        self.network.weight.add_(1)
                if owner.change_runtime:
                    owner.runtime_marker.write_bytes(b"DEBUG runtime changed during prediction")
                if owner.change_original_image:
                    path = owner.medical / "Data/image" / (cases[0] + "_0000.nii.gz")
                    path.write_bytes(path.read_bytes() + b"DEBUG source changed during prediction")
        return DebugPredictor()

    def evaluate(self, args):
        # Explicit DEBUG statistics profile, never written to production defaults.
        args.bootstrap_iterations = 20
        args.permutation_iterations = 20
        return evaluator.evaluate(args)

    def execute(self, **overrides):
        options = {"load_experiment": lambda root: self.proof,
                   "predictor_factory": self.factory, "evaluate_fn": self.evaluate}
        options.update(overrides)
        return producer.execute(self.args, **options)

    def files(self):
        return {str(p.relative_to(self.output)): file_sha256(p) for p in self.output.rglob("*") if p.is_file()}

    def external_basic(self):
        """DEBUG injected launcher certificate, not a historical audit bypass."""
        old = self.base / "old"
        original_bank = copy.deepcopy(self.bank)
        original_bank["debug_original_training_identity"] = "unchanged_old_bank"
        model = old / "model"
        for key in ("plans", "dataset"):
            target = old / "bank" / (key + ".json")
            self.write(target, json.loads(Path(self.bank["files"][key]["path"]).read_text()))
            original_bank["files"][key] = producer._record(target)
            model.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, model / (key + ".json"))
        self.local_basic_checkpoint = self.checkpoints["basic"]
        state = torch.load(self.local_basic_checkpoint, map_location="cpu", weights_only=False)
        state["onlinecp_curriculum_resume"]["bank_identity"] = original_bank
        checkpoint = model / "fold_0/checkpoint_final.pth"
        checkpoint.parent.mkdir()
        torch.save(state, checkpoint)
        self.checkpoints["basic"] = checkpoint
        old_runtime = old / "runtime/nnunetv2/DEBUG_original_runtime.json"
        self.write(old_runtime, {"debug_only": True, "historical_training_runtime": True})
        journal = old / "execution_journal.json"
        self.write(journal, {"debug_only": True, "complete": True, "original_basic_completed": True})
        receipt = self.run / "basic_reuse/receipt.json"
        self.write(receipt, {"debug_only": True, "injected_verified_launcher_boundary": True,
                             "original_checkpoint": producer._record(checkpoint)})
        origin = {"format": "hiercp_verified_basic_origin_v1", "experiment_root": str(old),
                  "checkpoint": producer._record(checkpoint), "bank_identity": original_bank,
                  "runtime_inventory": {"DEBUG_original_runtime.json": file_sha256(old_runtime)},
                  "training_journal_sha256": file_sha256(journal), "reuse_receipt": producer._record(receipt)}
        self.proof["basic_origin"] = origin
        return origin

    def test_external_basic_retains_origin_and_generates_all_new_predictions(self):
        origin = self.external_basic()
        old = Path(origin["experiment_root"])
        legacy_prediction = old / "historical_predictions" / (self.cases[0] + ".nii.gz")
        legacy_prediction.parent.mkdir()
        nib.save(nib.Nifti1Image(np.zeros_like(self.truth), np.eye(4)), legacy_prediction)
        before = {str(p.relative_to(old)): file_sha256(p) for p in old.rglob("*") if p.is_file()}
        unused_local_basic = file_sha256(self.local_basic_checkpoint)
        self.execute()
        identity = json.loads((self.output / "identity.json").read_text())
        self.assertEqual(identity["basic_origin"], origin)
        self.assertEqual(identity["bank_identity"], self.bank)
        self.assertNotEqual(identity["basic_origin"]["bank_identity"], self.bank)
        self.assertEqual(identity["models"]["basic"]["checkpoint"], origin["checkpoint"])
        self.assertEqual(identity["models"]["full"]["checkpoint"]["path"], str(self.checkpoints["full"]))
        self.assertEqual([(arm, cases) for arm, cases, _ in self.calls], [(arm, self.cases) for arm in producer.ARMS])
        for path in self.output.glob("generations/*/*/case_receipts/*.json"):
            self.assertIsNone(json.loads(path.read_text())["imported_from"])
        self.assertEqual(before, {str(p.relative_to(old)): file_sha256(p) for p in old.rglob("*") if p.is_file()})
        self.assertEqual(unused_local_basic, file_sha256(self.local_basic_checkpoint))
        state = torch.load(self.checkpoints["basic"], map_location="cpu", weights_only=False)
        self.assertEqual(state["onlinecp_curriculum_resume"]["bank_identity"], origin["bank_identity"])
        files, count = self.files(), len(self.calls)
        self.args.resume = True
        self.execute()
        self.assertEqual(count, len(self.calls))
        self.assertEqual(files, self.files())

    def test_external_basic_proof_is_strict_and_never_silently_ignored(self):
        original = copy.deepcopy(self.external_basic())
        bad = [None, {}, {**original, "unexpected": True},
               {**original, "format": "legacy"}, {**original, "experiment_root": "relative"},
               {**original, "runtime_inventory": {}},
               {**original, "runtime_inventory": {"../escape.py": "0" * 64}},
               {**original, "runtime_inventory": {"module.py": True}},
               {**original, "training_journal_sha256": "BAD"},
               {**original, "training_journal_sha256": "0" * 64},
               {**original, "checkpoint": {**original["checkpoint"], "sha256": "0" * 64}},
               {**original, "reuse_receipt": {**original["reuse_receipt"], "sha256": "0" * 64}},
               {**original, "bank_identity": {}},
               {**original, "checkpoint": {"path": "relative", "sha256": "0" * 64}}]
        for index, value in enumerate(bad):
            with self.subTest(debug_fault=index):
                self.proof["basic_origin"] = value
                with self.assertRaises(ValueError):
                    self.execute()
                self.assertFalse(self.output.exists())
                self.assertEqual(self.calls, [])
        self.proof["basic_origin"] = original

    def test_external_origin_does_not_weaken_default_or_full_checkpoint_bank_guard(self):
        origin = self.external_basic()
        del self.proof["basic_origin"]
        with self.assertRaisesRegex(ValueError, "epoch/arm/bank identity mismatch: basic"):
            self.execute()
        self.proof["basic_origin"] = origin
        state = torch.load(self.checkpoints["full"], map_location="cpu", weights_only=False)
        state["onlinecp_curriculum_resume"]["bank_identity"] = origin["bank_identity"]
        torch.save(state, self.checkpoints["full"])
        with self.assertRaisesRegex(ValueError, "epoch/arm/bank identity mismatch: full"):
            self.execute()
        self.assertFalse(self.output.exists())

    def test_external_basic_checkpoint_path_and_receipt_ownership_are_bound(self):
        origin = self.external_basic()
        self.proof["checkpoints"]["basic"] = self.local_basic_checkpoint
        with self.assertRaisesRegex(ValueError, "checkpoint path"):
            self.execute()
        self.proof["checkpoints"]["basic"] = Path(origin["checkpoint"]["path"])
        outside = self.base / "unowned_receipt.json"
        shutil.copyfile(origin["reuse_receipt"]["path"], outside)
        origin["reuse_receipt"] = producer._record(outside)
        with self.assertRaisesRegex(ValueError, "new experiment root"):
            self.execute()
        self.assertFalse(self.output.exists())

    def test_external_basic_model_plans_must_match_original_and_current_bank(self):
        origin = self.external_basic()
        original_plans = Path(origin["bank_identity"]["files"]["plans"]["path"])
        self.write(original_plans, {"debug_only": True, "different_native_plan": True})
        origin["bank_identity"]["files"]["plans"] = producer._record(original_plans)
        checkpoint = Path(origin["checkpoint"]["path"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state["onlinecp_curriculum_resume"]["bank_identity"] = origin["bank_identity"]
        torch.save(state, checkpoint)
        origin["checkpoint"] = producer._record(checkpoint)
        self.write(Path(origin["reuse_receipt"]["path"]),
                   {"debug_only": True, "injected_verified_launcher_boundary": True,
                    "original_checkpoint": origin["checkpoint"]})
        origin["reuse_receipt"] = producer._record(origin["reuse_receipt"]["path"])
        with self.assertRaisesRegex(ValueError, "original bank"):
            self.execute()
        shutil.copyfile(original_plans, checkpoint.parent.parent / "plans.json")
        with self.assertRaisesRegex(ValueError, "verified bank preprocessing"):
            self.execute()
        self.assertFalse(self.output.exists())

    def test_external_basic_witness_rechecks_receipt_journal_and_checkpoint(self):
        origin = self.external_basic()
        targets = [Path(origin["reuse_receipt"]["path"]),
                   Path(origin["experiment_root"]) / "execution_journal.json",
                   Path(origin["checkpoint"]["path"])]
        for target in targets:
            with self.subTest(debug_target=target.name):
                before = target.read_bytes()
                target.write_bytes(before + b" DEBUG changed artifact")
                with self.assertRaisesRegex(ValueError, "changed"):
                    producer._basic_origin_witness({"basic_origin": origin})
                target.write_bytes(before)  # Restore only this temporary DEBUG fixture.

    def test_external_reuse_receipt_change_during_native_prediction_cannot_complete(self):
        origin = self.external_basic()
        receipt = Path(origin["reuse_receipt"]["path"])
        def changed_receipt_factory(plan, device):
            predictor = self.factory(plan, device)
            original_predict = predictor.predict_from_files
            def predict(*args, **kwargs):
                original_predict(*args, **kwargs)
                receipt.write_bytes(receipt.read_bytes() + b" DEBUG changed certificate")
            predictor.predict_from_files = predict
            return predictor
        with self.assertRaisesRegex(ValueError, "changed"):
            self.execute(predictor_factory=changed_receipt_factory)
        self.assertFalse(list(self.output.glob("generations/*/*/case_receipts/*.json")))
        self.assertFalse((self.output / "completion.json").exists())

    def test_external_origin_change_on_resume_preserves_existing_evaluation(self):
        origin = self.external_basic()
        self.execute()
        before, count = self.files(), len(self.calls)
        origin["runtime_inventory"] = {"DEBUG_changed_original_runtime.json": "a" * 64}
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, "Evaluation identity changed"):
            self.execute()
        self.assertEqual(before, self.files())
        self.assertEqual(count, len(self.calls))

    def test_external_basic_experiment_cannot_be_used_as_evaluation_output(self):
        origin = self.external_basic()
        self.args.output_dir = str(Path(origin["experiment_root"]) / "forbidden_evaluation")
        with self.assertRaisesRegex(ValueError, "original Basic experiment"):
            self.execute()
        self.assertFalse(Path(self.args.output_dir).exists())

    def test_full_cohort_loaded_checkpoint_to_actual_evaluator_completion(self):
        result = self.execute()
        self.assertNotIn("basic_origin", json.loads((self.output / "identity.json").read_text()))
        receipt = json.loads(result.read_text())
        self.assertTrue(receipt["complete"])
        self.assertEqual(receipt["checkpoint_linkage"], "producer_verified_actual_loaded_weights")
        self.assertEqual([(arm, cases) for arm, cases, _ in self.calls], [(arm, self.cases) for arm in producer.ARMS])
        self.assertEqual(set(receipt["producer_completions"]), set(producer.ARMS))
        summary = json.loads((Path(receipt["evaluation_completion"]["path"]).parent / "summary.json").read_text())
        self.assertEqual(summary["version"], "online_basic_hiercp_evaluation_v5")
        self.assertEqual(summary["validation_cases"], 2)
        self.assertEqual(summary["matching_objective"], evaluator.MATCHING_OBJECTIVE)
        contract = summary["evaluation_contract"]
        with self.assertRaisesRegex(ValueError, "cohort or completion identity"):
            producer._verify_eval_completion(Path(receipt["evaluation_completion"]["path"]).parent, self.cases,
                {"basic": contract["methods"]["basic"]["predictions"],
                 "full": contract["methods"]["hier"]["predictions"]},
                {case_id: "0" * 64 for case_id in self.cases})

    def test_complete_resume_verifies_and_does_not_reinfer_or_modify_files(self):
        self.execute()
        before, count = self.files(), len(self.calls)
        self.args.resume = True
        self.execute()
        self.assertEqual(len(self.calls), count)
        self.assertEqual(before, self.files())

    def test_uncommitted_native_partial_is_preserved_but_not_reused(self):
        self.fail_native = True
        with self.assertRaisesRegex(RuntimeError, "DEBUG native"):
            self.execute()
        before = self.files()
        self.assertFalse(list(self.output.glob("generations/*/*/case_receipts/*.json")))
        self.args.resume = True
        self.execute()
        self.assertEqual(self.calls[1][1], self.cases)
        self.assertTrue(before.items() <= self.files().items())

    def test_committed_first_half_imports_into_new_generation_after_receipt_failure(self):
        original = producer._commit_json
        fired = []
        def stop_after_first(path, payload):
            if path.parent.name == "case_receipts" and path.stem == self.cases[1] and not fired:
                fired.append(True)
                raise OSError("DEBUG case receipt interrupted")
            return original(path, payload)
        with patch.object(producer, "_commit_json", side_effect=stop_after_first):
            with self.assertRaisesRegex(OSError, "DEBUG case receipt"):
                self.execute()
        committed = list(self.output.glob("generations/*/full/case_receipts/*.json"))
        self.assertEqual(len(committed), 1)
        before = self.files()
        self.args.resume = True
        self.execute()
        self.assertEqual(self.calls[1][1], self.cases[1:])
        self.assertTrue(before.items() <= self.files().items())
        imported = [json.loads(p.read_text()) for p in self.output.glob("generations/*/full/case_receipts/*.json")]
        self.assertTrue(any(row["imported_from"] is not None for row in imported))

    def test_completed_full_arm_is_imported_after_basic_initialization_failure(self):
        original_factory = self.factory
        def fail_basic(plan, device):
            predictor = original_factory(plan, device)
            original_initialize = predictor.initialize_from_trained_model_folder
            def initialize(*args, **kwargs):
                original_initialize(*args, **kwargs)
                if predictor.arm == "basic":
                    raise RuntimeError("DEBUG Basic initializer interrupted")
            predictor.initialize_from_trained_model_folder = initialize
            return predictor
        with self.assertRaisesRegex(RuntimeError, "DEBUG Basic"):
            self.execute(predictor_factory=fail_basic)
        self.assertEqual(len(list(self.output.glob("generations/*/full/case_receipts/*.json"))), len(self.cases))
        before = self.files()
        self.args.resume = True
        self.execute()
        self.assertEqual([(arm, cases) for arm, cases, _ in self.calls],
                         [("full", self.cases), ("basic", self.cases)])
        self.assertTrue(before.items() <= self.files().items())

    def test_certified_raw_input_contract_is_required_and_not_recomputed_from_current_inputs(self):
        original = copy.deepcopy(self.proof["raw_input_contract"])
        for fault in ("missing", "hash", "duplicate"):
            with self.subTest(fault=fault):
                self.proof["raw_input_contract"] = copy.deepcopy(original)
                if fault == "missing":
                    del self.proof["raw_input_contract"]
                elif fault == "hash":
                    self.proof["raw_input_contract"]["source_cases"][0]["image_sha256"] = "0" * 64
                else:
                    self.proof["raw_input_contract"]["source_cases"].append(copy.deepcopy(original["source_cases"][0]))
                with self.assertRaisesRegex(ValueError, "raw input|certified raw source"):
                    self.execute()
                self.assertFalse(self.output.exists())
        self.proof["raw_input_contract"] = original

    def test_original_ground_truth_changed_during_evaluation_cannot_complete(self):
        def changed_after_evaluation(args):
            result = self.evaluate(args)
            path = self.medical / "Data/labels" / (self.cases[0] + ".nii.gz")
            changed = self.truth.copy()
            changed[0, 0, 0] = 2
            nib.save(nib.Nifti1Image(changed, np.eye(4)), path)
            return result
        with self.assertRaisesRegex(ValueError, "changed"):
            self.execute(evaluate_fn=changed_after_evaluation)
        self.assertFalse((self.output / "completion.json").exists())

    def test_loaded_weight_mismatch_rejected_before_prediction(self):
        self.wrong_weights = True
        with self.assertRaisesRegex(ValueError, "different checkpoint weights"):
            self.execute()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.output / "completion.json").exists())

    def test_checkpoint_changed_during_native_call_is_not_committed(self):
        self.change_checkpoint = True
        with self.assertRaisesRegex(ValueError, "changed"):
            self.execute()
        self.assertFalse(list(self.output.glob("generations/*/*/case_receipts/*.json")))

    def test_wrong_final_epoch_rejected_before_creating_output(self):
        state = torch.load(self.checkpoints["full"], map_location="cpu", weights_only=False)
        state["current_epoch"] = 249
        torch.save(state, self.checkpoints["full"])
        with self.assertRaisesRegex(ValueError, "epoch/arm/bank"):
            self.execute()
        self.assertFalse(self.output.exists())

    def test_existing_output_requires_explicit_resume_and_preserves_bytes(self):
        self.execute()
        before = self.files()
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual(before, self.files())

    def test_tampered_producer_completion_is_rejected(self):
        result = self.execute()
        completion = json.loads(result.read_text())
        target = Path(completion["producer_completions"]["full"]["path"])
        target.write_bytes(target.read_bytes() + b" ")
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, "changed"):
            self.execute()

    def test_missing_prediction_or_different_registered_ground_truth_is_not_zero_filled(self):
        target = self.raw / "labelsTr" / (self.cases[0] + ".nii.gz")
        with target.open("ab") as handle:
            handle.write(b"DEBUG different registered labels")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.execute()
        self.assertFalse(self.output.exists())

    def test_native_weights_changed_in_memory_are_not_certified(self):
        self.change_memory_weights = True
        with self.assertRaisesRegex(ValueError, "weights changed"):
            self.execute()
        self.assertFalse(list(self.output.glob("generations/*/*/case_receipts/*.json")))

    def test_runtime_mutation_during_prediction_is_not_certified(self):
        self.change_runtime = True
        with self.assertRaisesRegex(ValueError, "runtime.*changed"):
            self.execute()
        self.assertFalse(list(self.output.glob("generations/*/*/case_receipts/*.json")))

    def test_original_image_mutation_is_not_hidden_by_registered_copy(self):
        self.change_original_image = True
        with self.assertRaisesRegex(ValueError, "changed"):
            self.execute()
        self.assertFalse(list(self.output.glob("generations/*/*/case_receipts/*.json")))

    def test_native_missing_case_cannot_create_any_case_receipt(self):
        self.missing_prediction = True
        with self.assertRaisesRegex(evaluator.EvaluationError, "Prediction cohort mismatch"):
            self.execute()
        self.assertFalse(list(self.output.glob("generations/*/*/case_receipts/*.json")))

    def test_loaded_inference_settings_mismatch_stops_before_prediction(self):
        self.wrong_settings = True
        with self.assertRaisesRegex(ValueError, "inference settings"):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_existing_output_rejection_precedes_checkpoint_loading(self):
        self.output.mkdir()
        with patch.object(producer, "_checkpoint_metadata", side_effect=AssertionError("late guard")):
            with self.assertRaises(FileExistsError):
                self.execute()

    def test_registered_raw_leaf_link_preserves_source_identity_when_supported(self):
        source = self.medical / "Data/image" / (self.cases[0] + "_0000.nii.gz")
        leaf = self.base / "DEBUG_raw_link.nii.gz"
        try:
            leaf.symlink_to(source)
        except OSError as exc:
            self.skipTest(f"Platform does not permit a DEBUG symlink: {exc}")
        record = producer._input_record(leaf, source)
        self.assertEqual(record["path"], str(leaf.absolute()))
        self.assertEqual(record["source"]["path"], str(source.resolve()))
        self.assertEqual(producer._verify_record(record), leaf)
        original_alias = self.base / "DEBUG_original_input_alias.nii.gz"
        original_alias.symlink_to(source)
        self.assertEqual(producer._verify_record(producer._input_record(original_alias, original_alias)), original_alias)
        directory_alias = self.base / "DEBUG_input_directory"
        directory_alias.symlink_to(source.parent, target_is_directory=True)
        alias_file = directory_alias / source.name
        self.assertEqual(producer._verify_record(producer._input_record(alias_file, source)), alias_file)
        with self.assertRaisesRegex(ValueError, "nonregular"):
            producer._record(alias_file)  # Input-only aliases do not authorize output redirects.
        with self.assertRaisesRegex(ValueError, "nonregular"):
            producer._record(leaf)  # Output/checkpoint policy was not weakened.
        redirected = self.base / "DEBUG_redirected_output"
        redirected.mkdir()
        output_alias = self.base / "DEBUG_output_alias"
        output_alias.symlink_to(redirected, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "ancestor redirect"):
            producer._commit_json(output_alias / "forbidden.json", {"debug_only": True})
        self.assertEqual(list(redirected.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
