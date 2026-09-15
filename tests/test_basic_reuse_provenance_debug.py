"""DEBUG only: tiny checkpoint metadata and real source-byte/runtime proofs.

No segmentation model, GPU, clinical data, or original experiment is executed.
The bank verifier boundary is isolated in these tests; its full native/raw
payload validation is covered by the dedicated bank/preprocessing suites.
Native-attempt fixtures below are explicitly synthetic DEBUG records; they
exercise real receipt readers, not a claim that native training was executed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from custom_trainers.onlinecp_feedback_policy import FeedbackState, validate_feedback_config, feedback_config_sha256, stage_for_epoch
from hiercp.feedback import tensor_state_sha256
from tools import basic_reuse_provenance as p
from tools import run_feedback_experiment as launch


REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


def file_record(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class BasicReuseProvenanceDebugTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="br")
        self.root = Path(self.temp.name).resolve()
        self.source, self.target, self.project = self.root / "s", self.root / "t", self.root / "p"
        self.source.mkdir()
        self.target_native = self.root / "n"
        for package in (self.source / "runtime/nnunetv2", self.target_native):
            (package / "training/nnUNetTrainer").mkdir(parents=True)
            (package / "__init__.py").write_text("# DEBUG synthetic native package\n", encoding="utf-8")
            (package / "training/nnUNetTrainer/nnUNetTrainer.py").write_text(
                "class nnUNetTrainer:\n    def __init__(self):\n"
                "        self.num_iterations_per_epoch = 250\n"
                "        self.num_val_iterations_per_epoch = 50\n", encoding="utf-8")
        for name in p.CURRENT_CP:
            shutil.copyfile(REPO / "custom_trainers" / name,
                            self.source / "runtime/nnunetv2" / p.PREFIX / name)
        for name in ("online_cp_feedback.json", "nnunet.json", "online_cp_feedback_gnn.json", "train.json"):
            (self.project / "config").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / "config" / name, self.project / "config" / name)
        self.policy = validate_feedback_config(json.loads((self.project / "config/online_cp_feedback.json").read_text()))
        self.dataset = "Dataset760_LiverOnlineCP_OF0"
        self.bank = self.source / "online/folds/fold_0/bank/index.json"
        self.pre = self.source / "online/nnunetv2/nnUNet_preprocessed" / self.dataset
        self.raw = self.source / "online/nnunetv2/nnUNet_raw" / self.dataset
        self.results = self.source / "online/nnunetv2/nnUNet_results"
        self.plans = {"dataset_name": self.dataset,
                      "configurations": {"3d_fullres": {"batch_size": 2, "patch_size": [8, 8, 8]}}}
        self.dataset_json = {"channel_names": {"0": "CT"}, "labels": {"background": 0, "liver": 1, "tumor": 2}}
        plans_path, dataset_path = self.pre / "nnUNetResEncUNetMPlans.json", self.pre / "dataset.json"
        write_json(plans_path, self.plans)
        write_json(dataset_path, self.dataset_json)
        self.index = {"entries_by_case": {"DEBUG_train": ["entry.npz"]}, "candidate_count": 128,
                      "dataset_name": self.dataset, "paste_contract": "onlinecp_raw_target_paste_v1"}
        write_json(self.bank, self.index)
        write_json(self.bank.parent / "config.json", {"DEBUG": True})
        self.identity = {"dataset_name": self.dataset, "verified_configuration": "3d_fullres",
                         "train_case_ids": ["DEBUG_train"], "validation_case_ids": ["DEBUG_val"],
                         "files": {"plans": file_record(plans_path), "dataset": file_record(dataset_path),
                                   "nnunet_config": file_record(self.project / "config/nnunet.json")}}
        write_json(self.bank.parent / "feedback_contract.json", self.identity)
        raw = {"dataset_name": self.dataset, "train_ids": ["DEBUG_train"], "val_ids": ["DEBUG_val"],
               "source_cases": {"DEBUG": "bank boundary isolated"}}
        write_json(self.raw / "online_cp_dataset.json", raw)
        marker = {"input_contract": {"raw_marker_sha256": file_record(self.raw / "online_cp_dataset.json")["sha256"],
                                      "raw_contract_sha256": p._sha(raw)},
                  "outputs": {"DEBUG": "native bank boundary isolated"}}
        write_json(self.pre / "online_cp_preprocess_complete.json", marker)
        self.old = {"project_root": str(self.project), "medical_root": str(self.root / "Medical"),
                    "run_root": str(self.source), "package_destination": str(self.source / "runtime/nnunetv2"),
                    "outer_fold": 0, "dataset_id": 760, "seed": 42, "python_executable": "/DEBUG/python",
                    "train_config": str(self.project / "config/train.json"),
                    "env_updates": {"nnUNet_results": str(self.results), "nnUNet_preprocessed": str(self.pre.parent),
                                    "nnUNet_raw": str(self.raw.parent)}}
        argv = ["/DEBUG/python", "-m", "tools.train_online_feedback", "--bank", str(self.bank),
                "--feedback-config", str(self.project / "config/online_cp_feedback.json"), "--configuration", "3d_fullres",
                "--device", "cuda", "--seed", "42", "--arm", "basic"]
        self.old["commands"] = [{"name": name, "argv": argv if name == "train_basic" else ["DEBUG", name]}
                                for name in ("install_private_trainers", "environment", "bank", "feedback_contract", "check_basic", "train_basic")]
        self.target_plan = {**self.old, "run_root": self.target,
                            "package_destination": self.target / "runtime/nnunetv2"}
        self.checkpoint_path = self.results / self.dataset / (
            p.TRAINER + "__nnUNetResEncUNetMPlans__3d_fullres") / "fold_0/checkpoint_final.pth"
        self.checkpoint_path.parent.mkdir(parents=True)
        self.state = self._state()
        torch.save(self.state, self.checkpoint_path)
        inputs = launch._resume_inputs({**self.old, "project_root": self.project, "train_config": Path(self.old["train_config"])})
        origin = self.project / "work/original"
        write_json(origin / "evidence.json", {"DEBUG": "original preparation source"})
        identity = {"source_root": str(origin), "files": {"evidence.json": file_record(origin / "evidence.json")["sha256"]}}
        write_json(self.source / "recovery/prepared.json", {"DEBUG": "prepared artifact"})
        receipt = {"format": "hiercp_preparation_recovery_complete_v1", "train_config": self.old["train_config"],
                   "source_identity": identity, "original_results_preserved": True, "training_performed": False,
                   "files": {"recovery/prepared.json": file_record(self.source / "recovery/prepared.json")["sha256"]}}
        write_json(self.source / "recovery/complete.json", receipt)
        self.journal = {"format": "feedback_preparation_execution_v1", "plan_sha256": launch._json_sha256(self.old),
                        "source_identity": identity, "runtime_inventory": self._inventory(),
                        "preparation_receipt": receipt, "training_started": True, "complete": True,
                        "stages": []}
        for name in ("copy_private_runtime", "install_private_trainers", "environment", "recover_preparation",
                     "bank", "feedback_contract", "check_basic", "train_basic"):
            row = {"name": name, "status": "completed"}
            if name in {"bank", "feedback_contract", "check_basic", "train_basic"}:
                artifact = self.checkpoint_path if name == "train_basic" else self.bank.parent / "feedback_contract.json"
                row.update(input_files=inputs, completion_evidence={"format": "feedback_stage_completion_v1",
                    "files": {artifact.relative_to(self.source).as_posix(): file_record(artifact)["sha256"]}})
            self.journal["stages"].append(row)
        self._publish()

    def tearDown(self):
        self.temp.cleanup()

    def _inventory(self):
        package = self.source / "runtime/nnunetv2"
        return {path.relative_to(package).as_posix(): file_record(path)["sha256"]
                for path in package.rglob("*") if path.is_file()}

    def _state(self):
        code_names = {"curriculum_trainer": "nnUNetTrainer_OnlineCPCurriculum.py", "curriculum_policy": "onlinecp_curriculum_policy.py",
                      "bank_verifier": "onlinecp_curriculum_contract.py", "legacy_trainer": "nnUNetTrainer_OnlinePairedCP.py",
                      "feedback_trainer": "nnUNetTrainer_OnlineCPFeedback.py", "feedback_policy": "onlinecp_feedback_policy.py",
                      "feedback_metrics": "onlinecp_feedback_metrics.py", "raw_bank": "onlinecp_raw_bank.py",
                      "raw_resampling": "onlinecp_raw_resampling.py"}
        code = {key: p.CURRENT_CP[value] for key, value in code_names.items()}
        code.update(base_trainer=self._inventory()["training/nnUNetTrainer/nnUNetTrainer.py"], hiercp_sources=p._current_hiercp_sha())
        runtime = {"trainer": p.TRAINER, "online_seed": 42, "source_identity": code,
                   "plans_sha256": p._sha(self.plans), "dataset_json_sha256": p._sha(self.dataset_json),
                   "configuration": "3d_fullres", "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
                   "num_epochs": 250, "gradient_accumulation_steps": 1, "physical_batch_size": 2,
                   "train_iterations_per_epoch": 250, "validation_iterations_per_epoch": 50, "augmentation_workers": 8,
                   "device_type": "cuda", "cuda_device_count": 1, "feedback_gnn_config": None,
                   "feedback_measurement": self.policy["difficulty"]["measurement_definition"]}
        weights = {"DEBUG_weight": torch.tensor([1.0, 2.0])}
        progress = {"completed_epoch": 249, "optimizer_steps": 62500, "network_sha256": tensor_state_sha256(weights)}
        table = FeedbackState(self.policy, {"entry.npz": "DEBUG_train"}, candidate_count=128, identity=self.identity)
        extension = {"format": p.RESUME, "table": table.state_dict(), "gnn": None, "predictions": None,
                     "prediction_provenance": None, "prediction_bundle_sha256": p._sha({"predictions": None, "provenance": None}),
                     "last_observations": [], "optimizer_steps": 62500,
                     "last_epoch": {"epoch": 249, "observations": 0, "observation_sha256": p._sha([]),
                                    "gnn": None, "nnunet_progress": progress}}
        return {"network_weights": weights, "optimizer_state": {"state": {}, "param_groups": [{"params": [0]}]},
                "grad_scaler_state": {"scale": 1.0, "growth_factor": 2.0, "backoff_factor": 0.5,
                                      "growth_interval": 2000, "_growth_tracker": 0},
                "logging": {}, "_best_ema": 0.0, "current_epoch": 250,
                "init_args": {"plans": self.plans, "dataset_json": self.dataset_json, "configuration": "3d_fullres", "fold": 0},
                "trainer_name": p.TRAINER, "inference_allowed_mirroring_axes": [0, 1, 2],
                "onlinecp_curriculum_resume": {"format": p.RESUME, "next_epoch": 250, "bank_identity": self.identity,
                    "config": self.policy, "config_sha256": feedback_config_sha256(self.policy), "runtime_identity": runtime,
                    "last_epoch": {"epoch": 249, "stage": stage_for_epoch(self.policy, 249)[0], "applied": 100,
                                   "samples": 500, "event_digest": "1" * 16, "choice_digest": "2" * 16},
                    "extension": extension, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": [torch.ones(8, dtype=torch.uint8)], "lr_scheduler_state": {"epoch": 250}}}

    def _publish(self):
        write_json(self.source / "launch_plan.json", self.old)
        payload = {k: v for k, v in self.journal.items() if k != "journal_sha256"}
        write_json(self.source / "execution_journal.json", {**payload, "journal_sha256": launch._json_sha256(payload)})

    def _inspect(self):
        with patch.object(p, "_verify_bank", return_value=(self.identity, self.index)):
            return p.inspect_basic_source(self.source, self.target_plan, target_native_package=self.target_native)

    def _checkpoint_change(self, change):
        change(self.state)
        torch.save(self.state, self.checkpoint_path)
        self.journal["stages"][-1]["completion_evidence"]["files"] = {
            self.checkpoint_path.relative_to(self.source).as_posix(): file_record(self.checkpoint_path)["sha256"]}
        self._publish()

    def _native_receipts(self, row, *, spec_changes=None, child_pid=17, returncode=0):
        folder = self.source / "execution_attempts" / row["attempt_id"]
        spec = {"format": "feedback_child_attempt_v1", "attempt_id": row["attempt_id"],
                "argv": row["argv"], "cwd": str(self.project), "parent": row["owner"]}
        spec.update(spec_changes or {})
        child = {"host": "DEBUG", "pid": child_pid, "process_started": 1.5}
        for name, value in (
                ("attempt.json", spec), ("child_started.json", child),
                ("permit.json", {"attempt_sha256": p._sha(spec), "child_pid": child_pid}),
                ("child_complete.json", {"format": "feedback_child_complete_v1", "attempt_sha256": p._sha(spec),
                                         "child": child, "returncode": returncode})):
            write_json(folder / name, value)

    def _fresh_history(self):
        from tools import feedback_fresh_execution as fresh
        plan = {**self.old, "run_root": self.source, "project_root": self.project,
                "train_config": Path(self.old["train_config"])}
        inputs = launch._resume_inputs(plan)
        native = fresh.native_inventory(self.old["package_destination"])
        owner = {"host": "DEBUG", "pid": 13, "process_started": 1.0}
        self.journal = {"format": fresh.FORMAT, "plan_sha256": launch._json_sha256(self.old),
                        "source_identity": {"native_package": str(self.target_native), "native_files": native},
                        "input_files": inputs, "runtime_inventory": self._inventory(),
                        "training_started": True, "complete": True,
                        "stages": [{"name": "copy_private_runtime", "status": "completed", "owner": owner,
                                    "attempt_id": "0" * 32, "input_files": inputs,
                                    "completion_evidence": fresh._evidence(plan, "copy_private_runtime")}]}
        for number, command in enumerate(self.old["commands"], 1):
            name = command["name"]
            row = {"name": name, "status": "completed", "argv": list(command["argv"]), "owner": owner,
                   "attempt_id": f"{number:032x}", "input_files": inputs, "execution_backend": "native_receipted"}
            if name == "environment":
                # Use the actual fresh producer schema, not a guessed fixture.
                row["completion_evidence"] = fresh._evidence(plan, name)
            else:
                artifacts = ([Path(self.old["package_destination"]) / value for value in self._inventory()]
                             if name == "install_private_trainers" else
                             [self.checkpoint_path if name == "train_basic" else self.bank.parent / "feedback_contract.json"])
                row["completion_evidence"] = {"format": "feedback_stage_completion_v1",
                    "files": {value.relative_to(self.source).as_posix(): file_record(value)["sha256"] for value in artifacts}}
            self._native_receipts(row)
            self.journal["stages"].append(row)
        self._publish()

    def _read_history(self):
        return p._history(self.source, self.old, p._read(self.source / "execution_journal.json"))

    def test_debug_completed_original_control_is_read_only_and_path_independent(self):
        before = {str(path): path.read_bytes() for path in self.source.rglob("*") if path.is_file()}
        result = self._inspect()
        self.assertEqual(result["format"], p.FORMAT)
        self.assertEqual(result["checkpoint"], file_record(self.checkpoint_path))
        self.assertFalse(result["native_equivalence"]["bitwise_replay_claimed"])
        self.assertEqual(result["native_equivalence"]["source_preprocessing"]["native_outputs"],
                         {"DEBUG": "native bank boundary isolated"})
        shutil.copytree(self.target_native, self.target_plan["package_destination"])
        with patch.object(p, "_verify_bank", return_value=(self.identity, self.index)):
            later = p.inspect_basic_source(self.source, self.target_plan)
        self.assertEqual(result, later)
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.source.rglob("*") if path.is_file()})
        self.assertFalse((self.source / "feedback_execution.lock").exists())

    def test_debug_gnn_only_current_config_change_does_not_relabel_recorded_inputs(self):
        original = self._inspect()
        write_json(self.project / "config/online_cp_feedback_gnn.json", {"DEBUG": "new full-only GNN"})
        self.assertEqual(original, self._inspect())

    def test_debug_policy_or_seed_change_is_not_an_equivalent_basic(self):
        self.target_plan["seed"] = 43
        with self.assertRaisesRegex(ValueError, "seed"):
            self._inspect()
        self.target_plan["seed"] = 42
        self.journal["stages"][-1]["input_files"]["config/online_cp_feedback.json"] = "0" * 64
        self._publish()
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self._inspect()

    def test_debug_unknown_changed_runtime_is_rejected_even_with_resigned_inventory(self):
        with patch("hiercp.contracts.ARCHITECTURE_VERSION", "DEBUG_unreviewed_semantics"):
            with self.assertRaisesRegex(ValueError, "architecture contracts differ"):
                self._inspect()
        module = self.source / "runtime/nnunetv2" / p.PREFIX / "onlinecp_raw_resampling.py"
        module.write_bytes(module.read_bytes() + b"\n# DEBUG unreviewed mutation\n")
        self.journal["runtime_inventory"] = self._inventory()
        self._publish()
        with self.assertRaisesRegex(ValueError, "Unknown historical"):
            self._inspect()

    def test_debug_native_target_change_is_rejected(self):
        (self.target_native / "__init__.py").write_text("# DEBUG changed native\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "native nnU-Net"):
            self._inspect()

    def test_debug_original_d904_runtime_hashes_have_explicit_reviewed_transition(self):
        for name in p.HISTORICAL_CP:
            if p.HISTORICAL_CP[name] != p.CURRENT_CP[name]:
                data = subprocess.check_output(["git", "show", "d904eeb346da8592ea5f04029e8aaa475ec9126e:custom_trainers/" + name], cwd=REPO)
                (self.source / "runtime/nnunetv2" / p.PREFIX / name).write_bytes(data)
        inventory = self._inventory()
        actual, evidence = p._runtime_equivalence(self.source / "runtime/nnunetv2", inventory, self.target_native)
        self.assertEqual(actual, inventory)
        self.assertEqual(evidence["format"], "d904eeb_basic_training_equivalence_v1")
        self.assertEqual(evidence["source_architecture"], "hiercp_conditioned_readout_v3")
        self.journal["runtime_inventory"] = inventory
        code = self.state["onlinecp_curriculum_resume"]["runtime_identity"]["source_identity"]
        code.update(curriculum_trainer=p.HISTORICAL_CP["nnUNetTrainer_OnlineCPCurriculum.py"],
                    bank_verifier=p.HISTORICAL_CP["onlinecp_curriculum_contract.py"],
                    feedback_trainer=p.HISTORICAL_CP["nnUNetTrainer_OnlineCPFeedback.py"],
                    hiercp_sources=p.HISTORICAL_HIERCP_SHA)
        self._checkpoint_change(lambda state: None)
        proof = self._inspect()
        self.assertEqual(proof["native_equivalence"]["format"], "d904eeb_basic_training_equivalence_v1")
        self.assertFalse((self.source / "feedback_execution.lock").exists())

    def test_debug_active_stage_or_legacy_lock_cannot_be_assumed_stopped(self):
        self.journal["stages"][-1]["status"] = "running"
        self._publish()
        with self.assertRaisesRegex(ValueError, "active"):
            self._inspect()
        self.journal["stages"][-1]["status"] = "completed"
        self._publish()
        (self.source / "recovery_execution.lock").write_text("DEBUG", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "legacy execution lock"):
            self._inspect()

    def test_debug_original_final_checkpoint_hash_and_epoch_are_both_required(self):
        self.checkpoint_path.write_bytes(self.checkpoint_path.read_bytes() + b"DEBUG")
        with self.assertRaisesRegex(ValueError, "completed stage files changed"):
            self._inspect()
        self._checkpoint_change(lambda state: state.update(current_epoch=249))
        with self.assertRaisesRegex(ValueError, "epoch/policy/bank"):
            self._inspect()

    def test_debug_recorded_batch_versions_and_full_arm_are_rejected(self):
        original = copy.deepcopy(self.state)
        changes = ((lambda state: state["onlinecp_curriculum_resume"]["runtime_identity"].update(physical_batch_size=1), "physical batch"),
                   (lambda state: state["onlinecp_curriculum_resume"]["runtime_identity"].update(torch_version="DEBUG_OLD"), "torch/numpy"),
                   (lambda state: state["onlinecp_curriculum_resume"]["extension"].update(gnn={}), "Full-GNN"),
                   (lambda state: state["onlinecp_curriculum_resume"]["runtime_identity"].update(validation_iterations_per_epoch=1), "actual iterations"),
                   (lambda state: state["grad_scaler_state"].update(growth_factor=0.5), "AMP scaler"),
                   (lambda state: state["onlinecp_curriculum_resume"].update(cuda_rng=[]), "RNG evidence"))
        for change, message in changes:
            with self.subTest(message=message):
                self.state = copy.deepcopy(original)
                self._checkpoint_change(change)
                with self.assertRaisesRegex(ValueError, message):
                    self._inspect()

    def test_debug_network_and_event_lineage_are_not_just_epoch_counters(self):
        self._checkpoint_change(lambda state: state["network_weights"]["DEBUG_weight"].add_(1))
        with self.assertRaisesRegex(ValueError, "exact checkpoint weights"):
            self._inspect()

    def test_debug_no_incomplete_or_debug_stage_adoption(self):
        with self.assertRaisesRegex(ValueError, "Partial Basic"):
            p.inspect_basic_source(self.source, self.target_plan, require_complete=False)
        self.journal["stages"][-1]["execution_backend"] = "injected_debug_runner"
        self._publish()
        with self.assertRaisesRegex(ValueError, "DEBUG/unknown"):
            self._inspect()
        self.journal["stages"][-1].pop("execution_backend")
        self.journal["stages"][-1]["argv"] = ["DEBUG", "different_actual_command"]
        self._publish()
        with self.assertRaisesRegex(ValueError, "actual stage argv"):
            self._inspect()

    def test_debug_fresh_producer_evidence_and_native_receipts_are_read_only(self):
        self._fresh_history()
        before = {str(path): file_record(path)["sha256"] for path in self.source.rglob("*") if path.is_file()}
        original = copy.deepcopy(self.journal)
        basic, argv, receipts = self._read_history()
        self.assertEqual(basic["name"], "train_basic")
        self.assertEqual(argv, self.old["commands"][-1]["argv"])
        self.assertEqual(len(receipts), 4 * len(self.old["commands"]))
        self.assertEqual(self.journal, original)
        self.assertEqual(before, {str(path): file_record(path)["sha256"] for path in self.source.rglob("*") if path.is_file()})
        proof = p._history_receipt(self.source, self.old, self.journal, self.journal["source_identity"]["native_files"])
        self.assertEqual(proof["source_identity"], self.journal["source_identity"])

    def test_debug_fresh_copy_environment_inputs_and_backend_cannot_be_dropped(self):
        self._fresh_history()
        original = copy.deepcopy(self.journal)
        cases = (
            (lambda j: j["stages"][0]["completion_evidence"].update(native_files={"bad": "0" * 64}), "native copy"),
            (lambda j: j["stages"][2]["completion_evidence"].update(format="feedback_stage_completion_v1"), "environment"),
            (lambda j: j["stages"][-1].update(input_files={"bad": "0" * 64}), "fresh stage input"),
            (lambda j: j["stages"][-1].pop("execution_backend"), "fresh stage input"),
            (lambda j: j["stages"][-1].update(attempt_id=j["stages"][-2]["attempt_id"]), "reused"),
        )
        for change, message in cases:
            with self.subTest(message=message):
                self.journal = copy.deepcopy(original)
                change(self.journal)
                self._publish()
                with self.assertRaisesRegex(ValueError, message):
                    self._read_history()

    def test_debug_native_attempt_cwd_parent_format_and_planned_argv_are_bound(self):
        self._fresh_history()
        row = self.journal["stages"][-1]
        for change in ({"cwd": str(self.root)}, {"format": "DEBUG_wrong"},
                       {"parent": {"host": "DEBUG", "pid": 99, "process_started": 1.0}}):
            with self.subTest(change=change):
                # Re-sign all dependent receipts: their own consistency alone
                # must not excuse executing a different source/cwd/owner.
                self._native_receipts(row, spec_changes=change)
                with self.assertRaisesRegex(ValueError, "attempt identity/cwd/owner/argv"):
                    self._read_history()
        self._native_receipts(row)
        row["argv"] = [*row["argv"], "--DEBUG-unapproved"]
        self._native_receipts(row)
        self._publish()
        with self.assertRaisesRegex(ValueError, "actual stage argv"):
            self._read_history()

    def test_debug_native_resume_requires_failed_prior_attempt_and_complete_receipts(self):
        self._fresh_history()
        row = self.journal["stages"][-1]
        row["argv"].append("--resume")
        self._native_receipts(row)
        self._publish()
        with self.assertRaisesRegex(ValueError, "actual stage argv"):
            self._read_history()
        previous = copy.deepcopy(row)
        previous.update(status="failed", error="DEBUG original interrupted native training", attempt_id="f" * 32)
        previous.pop("completion_evidence")
        previous["argv"].pop()
        self._native_receipts(previous, returncode=1)
        self.journal["stages"].insert(-1, previous)
        self._publish()
        _, _, receipts = self._read_history()
        self.assertEqual(len(receipts), 4 * (len(self.old["commands"]) + 1))
        complete_path = self.source / "execution_attempts" / row["attempt_id"] / "child_complete.json"
        complete_path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or unsafe"):
            self._read_history()

    def test_debug_native_receipt_content_is_in_immutable_proof(self):
        self._fresh_history()
        _, _, first = self._read_history()
        self._native_receipts(self.journal["stages"][-1], child_pid=18)
        _, _, second = self._read_history()
        self.assertNotEqual(first, second)

    def test_debug_all_source_native_roots_must_remain_inside_original_experiment(self):
        original = copy.deepcopy(self.old)
        for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            with self.subTest(name=name):
                self.old = copy.deepcopy(original)
                self.old["env_updates"][name] = str(self.root / "outside" / name)
                self.journal["plan_sha256"] = launch._json_sha256(self.old)
                self._publish()
                with patch.object(p, "_verify_bank", side_effect=AssertionError("must reject before reading another bank")):
                    with self.assertRaisesRegex(ValueError, name + " root"):
                        p.inspect_basic_source(self.source, self.target_plan, target_native_package=self.target_native)

    def test_debug_provenance_file_extra_metadata_is_preserved_not_ignored_hashes(self):
        path = self.project / "config/nnunet.json"
        record = {**file_record(path), "bytes": path.stat().st_size, "role": "DEBUG original producer metadata"}
        original = copy.deepcopy(record)
        self.assertEqual(p._file_record(record, "nnunet_config"), file_record(path))
        self.assertEqual(record, original)
        for changed in ({**record, "sha256": "0" * 64}, {"sha256": record["sha256"]}, {**record, "path": None}):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    p._file_record(changed, "nnunet_config")

    def test_debug_conditional_native_iteration_literal_needs_explicit_review(self):
        path = self.target_native / "training/nnUNetTrainer/nnUNetTrainer.py"
        path.write_text("class nnUNetTrainer:\n    def __init__(self):\n"
                        "        if True:\n            self.num_iterations_per_epoch = 250\n"
                        "        self.num_val_iterations_per_epoch = 50\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "conditional"):
            p._native_iterations(self.target_native)


if __name__ == "__main__":
    unittest.main()
