"""DEBUG new-trainer plumbing: real torch operations, no medical training/data.

Repository helpers are temporarily aliased at their installed nnU-Net import
names. This tests the new trainer without overwriting the installed originals.
Small analytic tensors and explicit debug bank/table boundaries are not model
validation, checkpoints from training, or scientific performance results.
"""
from __future__ import annotations

import copy
import importlib.util
import inspect
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
    import torch
except ModuleNotFoundError as error:
    np = torch = None
    DEPENDENCY_ERROR = str(error)
else:
    DEPENDENCY_ERROR = ""


class DebugStrictTable:
    """Explicit test double for the separately tested policy state validator."""
    def __init__(self):
        self.revision = 0

    def state_dict(self):
        return {"identity": "DEBUG_dataset_A_only", "revision": self.revision}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("identity") != "DEBUG_dataset_A_only":
            raise ValueError("DEBUG table identity changed")
        self.revision = state["revision"]

    def snapshot(self, epoch, *, predicted_difficulties, prediction_provenance):
        """DEBUG boundary only; the real policy validator has its own suite."""
        if type(epoch) is not int or not 0 <= epoch < 250:
            raise ValueError("DEBUG next epoch is outside the unchanged training contract")
        if predicted_difficulties is not None or prediction_provenance is not None:
            raise ValueError("DEBUG Basic control fixture must have no predicted difficulty")
        return {"debug_next_epoch": epoch, "revision": self.revision}


@unittest.skipIf(torch is None, "DEBUG dependency unavailable: " + DEPENDENCY_ERROR)
class FeedbackTrainerIntegrationDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved_modules = {}
        cls.missing = object()
        root = Path(__file__).resolve().parents[1] / "custom_trainers"
        prefix = "nnunetv2.training.nnUNetTrainer."
        names = ["onlinecp_curriculum_policy", "onlinecp_curriculum_contract",
                 "onlinecp_feedback_metrics", "onlinecp_feedback_policy",
                 "nnUNetTrainer_OnlineCPCurriculum", "nnUNetTrainer_OnlineCPFeedback"]
        try:
            for name in names:
                alias = prefix + name
                cls.saved_modules[alias] = sys.modules.get(alias, cls.missing)
                spec = importlib.util.spec_from_file_location(alias, root / (name + ".py"))
                module = importlib.util.module_from_spec(spec)
                sys.modules[alias] = module
                spec.loader.exec_module(module)
            cls.module = sys.modules[prefix + names[-1]]
            cls.metrics = sys.modules[prefix + "onlinecp_feedback_metrics"]
        except ModuleNotFoundError as error:
            cls._restore_aliases()
            raise unittest.SkipTest("Actual integration dependency unavailable: " + str(error)) from error
        except Exception:
            cls._restore_aliases()
            raise  # Import/API failures are real failures, never an integration pass.

    @classmethod
    def _restore_aliases(cls):
        for alias, previous in reversed(tuple(cls.saved_modules.items())):
            if previous is cls.missing:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = previous

    @classmethod
    def tearDownClass(cls):
        cls._restore_aliases()

    def tensors(self):
        data = torch.linspace(-1, 1, 2 * 9 ** 3).reshape(2, 1, 9, 9, 9)
        labels = torch.ones((2, 1, 9, 9, 9), dtype=torch.int64)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[:, :, 3:6, 3:6, 3:6] = True
        labels[mask] = 2
        context = {"pasted_mask": mask, "valid_mask": torch.ones_like(mask),
                   "event_applied": torch.ones(2, dtype=torch.bool),
                   "mask_truncated": torch.zeros(2, dtype=torch.bool)}
        return data, labels, context

    def test_actual_single_forward_backward_and_preupdate_observation_preserve_loss(self):
        data, labels, context = self.tensors()
        network = torch.nn.Conv3d(1, 3, 1, bias=True)
        with torch.no_grad():
            network.weight.copy_(torch.tensor([0.2, -0.1, 0.3]).reshape(3, 1, 1, 1, 1))
            network.bias.copy_(torch.tensor([0.1, 0.2, -0.2]))
        reference = copy.deepcopy(network)
        calls = []
        handle = network.register_forward_hook(lambda *args: calls.append("forward"))

        class NormalClassLoss(torch.nn.Module):
            def forward(self, output, target):
                return torch.nn.functional.cross_entropy(output, target[:, 0])

        observer = self.module.FeedbackLossObserver(NormalClassLoss())
        observer.context = context
        output = network(data)
        expected_observation = self.metrics.compute_feedback_metrics(output, labels, **context)
        loss = observer(output, labels)
        expected_loss = NormalClassLoss()(reference(data), labels)
        torch.testing.assert_close(loss, expected_loss)
        loss.backward(); expected_loss.backward()
        for actual, expected in zip(network.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)
        before_update = observer.observation["foreground_ce"].clone()
        torch.testing.assert_close(before_update, expected_observation["foreground_ce"])
        optimizer = torch.optim.SGD(network.parameters(), lr=0.2)
        optimizer.step()
        self.assertEqual(calls, ["forward"])
        torch.testing.assert_close(observer.observation["foreground_ce"], before_update)
        self.assertFalse(observer.observation["foreground_ce"].requires_grad)
        handle.remove()

    def test_observer_keeps_every_deep_supervision_loss_but_measures_full_resolution(self):
        data, labels, context = self.tensors()
        full = torch.cat((data, data * 2, data * -1), dim=1).requires_grad_()
        coarse = torch.nn.functional.avg_pool3d(full, 3)
        coarse_labels = torch.nn.functional.interpolate(labels.float(), size=(3, 3, 3), mode="nearest-exact").long()

        class DebugDsLoss(torch.nn.Module):
            def forward(self, outputs, targets):
                return (torch.nn.functional.cross_entropy(outputs[0], targets[0][:, 0])
                        + 0.5 * torch.nn.functional.cross_entropy(outputs[1], targets[1][:, 0]))

        base = DebugDsLoss()
        observer = self.module.FeedbackLossObserver(base)
        observer.context = context
        result = observer([full, coarse], [labels, coarse_labels])
        torch.testing.assert_close(result, base([full, coarse], [labels, coarse_labels]))
        result.backward()
        self.assertGreater(float(full.grad.abs().sum()), 0)
        torch.testing.assert_close(observer.observation["foreground_voxels"], context["pasted_mask"].flatten(1).sum(1))

    def test_second_loss_call_in_same_event_is_rejected(self):
        data, labels, context = self.tensors()
        output = torch.cat((data, data, data), dim=1)
        observer = self.module.FeedbackLossObserver(mock.Mock(return_value=torch.tensor(1.0)))
        observer.context = context
        observer(output, labels)
        with self.assertRaisesRegex(self.module.CurriculumError, "twice"):
            observer(output, labels)
        self.assertEqual(observer.base_loss.call_count, 1)

    def test_loss_without_training_context_does_not_record_feedback(self):
        base = mock.Mock(return_value=torch.tensor(3.0))
        observer = self.module.FeedbackLossObserver(base)
        self.assertEqual(float(observer(torch.tensor(2.0), torch.tensor(0))), 3.0)
        self.assertIsNone(observer.observation)

    def test_public_constructor_matches_nnunet_reflective_init_contract(self):
        for trainer_name in ("nnUNetTrainer_250epochs_OnlineBasicCPFeedbackControl", "nnUNetTrainer_250epochs_OnlineHierCPFeedback"):
            signature = inspect.signature(getattr(self.module, trainer_name).__init__)
            self.assertEqual(list(signature.parameters), ["self", "plans", "configuration", "fold", "dataset_json", "device"])
            self.assertEqual(signature.parameters["device"].default, torch.device("cuda"))
            self.assertFalse(any(parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                                 for parameter in signature.parameters.values()))

    def sampler(self, draws, *, basic=False):
        loader = self.module.nnUNetDataLoaderOnlineCPFeedback.__new__(self.module.nnUNetDataLoaderOnlineCPFeedback)
        remaining, consumed, selected = list(draws), [], []
        def random_value():
            value = remaining.pop(0); consumed.append(value); return value
        entry = {"scores": np.linspace(0, 1, 128),
                 "candidate_centers": np.arange(128 * 3).reshape(128, 3)}
        loader.online_bank = SimpleNamespace(entry_names=lambda case: ["entry_a", "entry_b"],
            cp_probability=0.5, load_for_case=lambda case, index: entry,
            intensity_scale=(0.8, 1.2), intensity_shift_hu=(-10, 10), ct_mean=5, ct_std=2)
        loader._rng = lambda: SimpleNamespace(random=random_value)
        loader.online_epoch, loader.curriculum_sha256 = 4, "debug_configuration_sha"
        loader.basic_control = basic
        def choose(entry_id, scores, epoch, u, *, basic_control):
            selected.append((entry_id, epoch, u, basic_control))
            return 2 if basic_control else 7
        loader.feedback_state = SimpleNamespace(select=choose)
        loader._entry_ids, loader._candidate_indices, loader._choice_tokens = [], [], []
        return loader, consumed, selected

    def test_paired_sampler_uses_exact_five_draws_and_binds_actual_selected_candidate(self):
        draws = [0.1, 0.8, 0.3, 0.4, 0.5]
        full, full_draws, calls = self.sampler(draws)
        basic, basic_draws, _ = self.sampler(draws, basic=True)
        full_plan, full_event = full._sample_paste_plan("train_case")
        basic_plan, basic_event = basic._sample_paste_plan("train_case")
        self.assertEqual(full_draws, draws); self.assertEqual(basic_draws, draws)
        self.assertEqual(full_event, basic_event)
        self.assertEqual(calls, [("entry_b", 4, 0.3, False)])
        self.assertEqual(full._entry_ids, ["entry_b"])
        self.assertEqual(full._candidate_indices, [7])
        self.assertEqual(full_plan["center"], (21, 22, 23))
        for key in ("scale", "normalized_offset"):
            self.assertEqual(full_plan[key], basic_plan[key])
        self.assertNotEqual(full._choice_tokens, basic._choice_tokens)

    def test_no_event_still_consumes_five_draws_and_has_no_fake_candidate(self):
        draws = [0.8, 0.2, 0.3, 0.4, 0.5]
        loader, consumed, selected = self.sampler(draws)
        plan, _ = loader._sample_paste_plan("train_case")
        self.assertIsNone(plan)
        self.assertEqual(consumed, draws)
        self.assertEqual(selected, [])
        self.assertEqual(loader._entry_ids, [""])
        self.assertEqual(loader._candidate_indices, [-1])

    def paste_fixture(self):
        loader = self.module.nnUNetDataLoaderOnlineCPFeedback.__new__(self.module.nnUNetDataLoaderOnlineCPFeedback)
        loader.online_bank = SimpleNamespace(tumor_label=2)
        source_mask = np.zeros((3, 3, 3), dtype=bool)
        source_mask[1, 1, 1] = True; source_mask[1, 1, 2] = True
        entry = {"source_mask": source_mask, "source_data": np.full((1, 3, 3, 3), 7, dtype=np.float32),
                 "anchor_offset": np.array([1, 1, 1])}
        plan = {"entry": entry, "center": (4, 4, 4), "scale": 1.0, "normalized_offset": 0.0}
        return loader, np.zeros((1, 9, 9, 9), dtype=np.float32), np.full((1, 9, 9, 9), 2, dtype=np.int16), plan

    def test_exact_source_mask_not_label_difference_tracks_paste_over_existing_tumor(self):
        loader, data, segmentation, plan = self.paste_fixture()
        old_seg = segmentation.copy()
        loader._apply_paste_to_crop(data, segmentation, (0, 0, 0), plan, "train_case")
        np.testing.assert_array_equal(segmentation, old_seg)
        self.assertEqual(int(loader._crop_pasted_mask.sum()), 2)
        np.testing.assert_array_equal(data[0] != 0, loader._crop_pasted_mask)

    def test_paste_into_padding_rejected_before_image_or_label_mutation(self):
        loader, data, segmentation, plan = self.paste_fixture()
        segmentation[0, 4, 4, 4] = -1
        old_data, old_seg = data.copy(), segmentation.copy()
        with self.assertRaises((self.module.CurriculumError, ValueError)):
            loader._apply_paste_to_crop(data, segmentation, (0, 0, 0), plan, "train_case")
        np.testing.assert_array_equal(data, old_data)
        np.testing.assert_array_equal(segmentation, old_seg)

    def trainer_and_batch(self):
        trainer = self.module._nnUNetTrainer_250epochs_OnlineFeedback.__new__(self.module._nnUNetTrainer_250epochs_OnlineFeedback)
        data, labels, context = self.tensors()
        trainer._feedback_snapshot_sha256 = "debug_epoch_snapshot"
        trainer.curriculum_bank_identity = {"train_case_ids": ["train_a", "train_b"]}
        trainer._feedback_entry_cases = {"entry_a": "train_a", "entry_b": "train_b"}
        trainer.loss = self.module.FeedbackLossObserver(torch.nn.Identity())
        batch = {"data": data, "target": labels, "keys": ["train_a", "train_b"],
                 "online_cp_applied": np.ones(2, dtype=np.uint8),
                 "feedback_snapshot_sha256": trainer._feedback_snapshot_sha256,
                 "feedback_entry_ids": ["entry_a", "entry_b"], "feedback_candidate_indices": [7, 8]}
        for key, value in context.items():
            if key != "event_applied": batch["feedback_" + key] = value
        for key in ("raw_support_count", "removed_by_label_resampling", "removed_by_padding"):
            batch["feedback_" + key] = torch.zeros(2, dtype=torch.int64)
        return trainer, batch

    def assert_before_forward_rejection(self, trainer, batch):
        with mock.patch.object(self.module._nnUNetTrainer_250epochs_OnlineCurriculum, "train_step") as normal_step:
            with self.assertRaises(self.module.CurriculumError):
                trainer.train_step(batch)
            self.assertEqual(normal_step.call_count, 0, "Invalid feedback reached the normal forward/update path")

    def test_heldout_validation_case_rejected_before_normal_forward(self):
        trainer, batch = self.trainer_and_batch()
        batch["keys"][0] = "heldout_validation_case"
        self.assert_before_forward_rejection(trainer, batch)

    def test_cross_patient_entry_rejected_before_normal_forward(self):
        trainer, batch = self.trainer_and_batch()
        batch["feedback_entry_ids"][0] = "entry_b"
        self.assert_before_forward_rejection(trainer, batch)

    def test_candidate_and_no_event_metadata_must_agree_before_forward(self):
        for candidate, applied in ((128, 1), (-1, 1), (7, 0)):
            trainer, batch = self.trainer_and_batch()
            batch["feedback_candidate_indices"][0] = candidate
            batch["online_cp_applied"][0] = applied
            with self.subTest(candidate=candidate, applied=applied):
                self.assert_before_forward_rejection(trainer, batch)

    def checkpoint_fixture(self):
        trainer, _ = self.trainer_and_batch()
        trainer.feedback_state, trainer.feedback_gnn = DebugStrictTable(), None
        trainer.basic_control = True
        trainer.num_iterations_per_epoch = 250
        trainer.num_epochs = 250
        trainer._feedback_predictions = trainer._feedback_prediction_provenance = None
        trainer._feedback_optimizer_steps = 250
        trainer._feedback_records = []
        trainer._feedback_epoch_summary = {"epoch": 0, "observations": 0,
            "observation_sha256": self.module.canonical_sha256([]),
            "nnunet_progress": {"completed_epoch": 0, "optimizer_steps": 250,
                                "network_sha256": "a" * 64}}
        return trainer, trainer._checkpoint_extension()

    def checkpoint_wrapper_fixture(self, *, basic_control):
        """DEBUG wrapper boundary; parent deserialization/restore is mocked."""
        from hiercp.feedback import tensor_state_sha256
        trainer, extension = self.checkpoint_fixture()
        trainer.basic_control = basic_control
        weights = {"DEBUG_segmentation_weight": torch.tensor([0.25, -0.75])}
        current_progress = {"completed_epoch": 2, "optimizer_steps": 750,
                            "network_sha256": tensor_state_sha256(weights)}
        old_gnn_progress = {"completed_epoch": 1, "optimizer_steps": 500,
                            "network_sha256": "b" * 64}
        extension["last_epoch"].update(epoch=2, nnunet_progress=current_progress)
        extension["optimizer_steps"] = 750
        extension["gnn"] = (None if basic_control else
                            {"completed_epoch": 2, "nnunet_progress": old_gnn_progress})
        return trainer, {"network_weights": weights, "current_epoch": 3,
                         "onlinecp_curriculum_resume": {"extension": extension}}

    def test_checkpoint_load_uses_current_nnunet_progress_not_last_gnn_training_progress(self):
        for basic_control in (False, True):
            trainer, checkpoint = self.checkpoint_wrapper_fixture(basic_control=basic_control)
            result = object()
            with self.subTest(basic_control=basic_control), mock.patch.object(
                    self.module._nnUNetTrainer_250epochs_OnlineCurriculum, "load_checkpoint",
                    return_value=result) as parent_restore:
                self.assertIs(trainer.load_checkpoint(checkpoint), result)
                self.assertEqual(parent_restore.call_count, 1)
                self.assertIs(parent_restore.call_args.args[0], checkpoint)

    def test_checkpoint_changed_actual_nnunet_weights_rejected_before_parent_restore(self):
        for basic_control in (False, True):
            trainer, checkpoint = self.checkpoint_wrapper_fixture(basic_control=basic_control)
            checkpoint["network_weights"]["DEBUG_segmentation_weight"][0] += 1
            with self.subTest(basic_control=basic_control), mock.patch.object(
                    self.module._nnUNetTrainer_250epochs_OnlineCurriculum, "load_checkpoint") as parent_restore:
                with self.assertRaises(self.module.CurriculumError):
                    trainer.load_checkpoint(checkpoint)
                self.assertEqual(parent_restore.call_count, 0,
                                 "Changed segmentation weights reached parent checkpoint restore")

    def test_checkpoint_extension_rejects_missing_required_state(self):
        trainer, extension = self.checkpoint_fixture()
        for key in tuple(extension):
            incomplete = copy.deepcopy(extension); incomplete.pop(key)
            with self.subTest(key=key), self.assertRaises(self.module.CurriculumError):
                trainer._validate_checkpoint_extension(incomplete, 1)

    def test_checkpoint_table_identity_change_rejected_without_live_mutation(self):
        trainer, extension = self.checkpoint_fixture()
        extension["table"]["identity"] = "DEBUG_different_dataset"
        with self.assertRaisesRegex(ValueError, "identity changed"):
            trainer._validate_checkpoint_extension(extension, 1)
        self.assertEqual(trainer.feedback_state.revision, 0)

    def test_checkpoint_record_digest_or_optimizer_counter_mismatch_rejected(self):
        trainer, extension = self.checkpoint_fixture()
        changed_records = copy.deepcopy(extension); changed_records["last_observations"] = [{"debug": "unbound"}]
        changed_steps = copy.deepcopy(extension); changed_steps["optimizer_steps"] = 251
        for changed in (changed_records, changed_steps):
            with self.assertRaises(self.module.CurriculumError):
                trainer._validate_checkpoint_extension(changed, 1)

    def test_checkpoint_prediction_bundle_corruption_is_rejected_before_restore(self):
        trainer, extension = self.checkpoint_fixture()
        self.assertIn("prediction_bundle_sha256", extension)
        for name in ("predictions", "prediction_provenance"):
            changed = copy.deepcopy(extension)
            changed[name] = {"DEBUG_corrupted_field": True}
            with self.subTest(field=name), self.assertRaises(self.module.CurriculumError):
                trainer._validate_checkpoint_extension(changed, 1)
        self.assertEqual(trainer.feedback_state.revision, 0)

    def test_checkpoint_validation_uses_copy_then_explicit_restore_updates_state(self):
        trainer, extension = self.checkpoint_fixture()
        extension["table"]["revision"] = 1
        with mock.patch.object(DebugStrictTable, "snapshot", autospec=True,
                               side_effect=DebugStrictTable.snapshot) as snapshot:
            trainer._validate_checkpoint_extension(extension, 1)
            self.assertEqual(snapshot.call_count, 1)
            validated_table = snapshot.call_args.args[0]
            self.assertIsNot(validated_table, trainer.feedback_state)
            self.assertEqual(validated_table.revision, 1)
            self.assertEqual(snapshot.call_args.args[1:], (1,))
            self.assertEqual(snapshot.call_args.kwargs,
                             {"predicted_difficulties": None, "prediction_provenance": None})
            self.assertEqual(trainer.feedback_state.revision, 0)
            trainer._restore_checkpoint_extension(extension, 1)
            self.assertEqual(snapshot.call_count, 2)
        self.assertEqual(trainer.feedback_state.revision, 1)
        self.assertEqual(trainer._feedback_optimizer_steps, 250)


if __name__ == "__main__":
    unittest.main()
