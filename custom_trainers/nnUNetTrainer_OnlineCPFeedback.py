"""Train-only segmentation feedback; separate from the frozen rank-band experiment.

The ordinary nnU-Net forward/loss/backward is retained. A loss observer measures
the same pre-update predictions, without another segmentation forward. Workers
read an immutable epoch snapshot. Only completed training observations update
the difficulty GNN and the following epoch's sampler.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCP import (
    TRAINER_FORMAT, _stable_u64, _anchored_slices, nnUNetDataLoaderOnlineCP,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlineCPCurriculum import (
    _nnUNetTrainer_250epochs_OnlineCurriculum,
)
from nnunetv2.training.nnUNetTrainer.onlinecp_curriculum_policy import (
    CurriculumError, canonical_sha256, schedule_token,
)
from nnunetv2.training.nnUNetTrainer.onlinecp_feedback_policy import (
    FeedbackState, validate_feedback_config, feedback_config_sha256,
    stage_for_epoch,
)
from nnunetv2.training.nnUNetTrainer.onlinecp_feedback_metrics import (
    compute_feedback_metrics, transform_with_feedback,
)


class _FeedbackTransform:
    def __init__(self, transform, owner):
        self.transform, self.owner = transform, owner

    def __call__(self, *, image, segmentation):
        mask = self.owner._crop_pasted_mask
        if mask is None:
            mask = torch.zeros_like(segmentation[:1], dtype=torch.bool)
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool)[None]
        result = transform_with_feedback(
            self.transform, image=image, segmentation=segmentation, pasted_mask=mask,
        )
        # Difficulty is measured at highest resolution only. The ordinary
        # segmentation targets still contain every configured DS scale.
        fields = {"pasted_mask", "valid_mask", "mask_truncated", "raw_support_count",
                  "removed_by_label_resampling", "removed_by_padding"}
        if not fields.issubset(result):
            raise CurriculumError("Feedback transform is missing full-resolution attribution")
        self.owner._transformed_feedback.append({key: result[key] for key in fields})
        return {"image": result["image"], "segmentation": result["segmentation"]}


class nnUNetDataLoaderOnlineCPFeedback(nnUNetDataLoaderOnlineCP):
    def __init__(self, *args, curriculum_config, basic_control, feedback_state, **kwargs):
        self.curriculum_config = validate_feedback_config(curriculum_config)
        self.curriculum_sha256 = feedback_config_sha256(self.curriculum_config)
        self.basic_control = bool(basic_control)
        self.feedback_state = feedback_state
        self.snapshot_sha256 = None
        self._crop_pasted_mask = None
        super().__init__(*args, **kwargs)
        if self.patch_size_was_2d or self.online_bank.tumor_label != 2:
            raise CurriculumError("Feedback requires the verified 3-D, 0/1/2 liver label contract")
        if self.transforms is None:
            raise CurriculumError("Feedback requires the actual nnU-Net augmentation transform")
        self.transforms = _FeedbackTransform(self.transforms, self)

    def _sample_paste_plan(self, case_id):
        self._crop_pasted_mask = None
        names = self.online_bank.entry_names(case_id)
        rng = self._rng()
        apply_cp = float(rng.random()) < self.online_bank.cp_probability
        source_u, candidate_u, scale_u, shift_u = [float(rng.random()) for _ in range(4)]
        original = _stable_u64(TRAINER_FORMAT, self.online_epoch, str(case_id),
                               int(apply_cp and bool(names)), source_u.hex(),
                               candidate_u.hex(), scale_u.hex(), shift_u.hex())
        event = schedule_token("event", self.curriculum_sha256, int(original))
        plan = None
        entry_id, candidate_index = "", -1
        if apply_cp and names:
            entry_index = min(len(names) - 1, int(np.floor(source_u * len(names))))
            entry_id = names[entry_index]
            entry = self.online_bank.load_for_case(case_id, entry_index)
            candidate_index = self.feedback_state.select(
                entry_id, entry["scores"], int(self.online_epoch), candidate_u,
                basic_control=self.basic_control,
            )
            center = tuple(int(v) for v in entry["candidate_centers"][candidate_index])
            lo, hi = self.online_bank.intensity_scale
            shift_lo, shift_hi = self.online_bank.intensity_shift_hu
            scale = lo + scale_u * (hi - lo)
            shift = shift_lo + shift_u * (shift_hi - shift_lo)
            plan = dict(entry=entry, candidate_index=candidate_index, center=center,
                        scale=scale, normalized_offset=((scale - 1.0) * self.online_bank.ct_mean
                                                       + shift) / self.online_bank.ct_std)
        self._entry_ids.append(entry_id)
        self._candidate_indices.append(candidate_index)
        choice = None if plan is None else [candidate_index, list(plan["center"])]
        self._choice_tokens.append(schedule_token("choice", self.curriculum_sha256, event, choice))
        return plan, event

    def _apply_paste_to_crop(self, data_cropped, seg_cropped, bbox_lbs, plan, case_id):
        entry = plan["entry"]
        center = np.asarray(plan["center"]) - np.asarray(bbox_lbs)
        slices = _anchored_slices(center, entry["source_mask"].shape,
                                  entry["anchor_offset"], data_cropped.shape[1:])
        if slices is None:
            raise CurriculumError("Pasted support and actual CP crop disagree")
        source_mask = entry["source_mask"].astype(bool, copy=False)
        if np.any(seg_cropped[(0, *slices)][source_mask] < 0):
            raise CurriculumError("CP would overwrite padding outside the actual recipient image")
        super()._apply_paste_to_crop(data_cropped, seg_cropped, bbox_lbs, plan, case_id)
        self._crop_pasted_mask = np.zeros(data_cropped.shape[1:], dtype=bool)
        self._crop_pasted_mask[slices] = entry["source_mask"].astype(bool, copy=False)

    def generate_train_batch(self):
        if self.snapshot_sha256 is None:
            raise CurriculumError("Worker has no frozen feedback snapshot")
        self._entry_ids, self._candidate_indices, self._choice_tokens = [], [], []
        self._transformed_feedback = []
        batch = super().generate_train_batch()
        if len(self._transformed_feedback) != self.batch_size:
            raise CurriculumError("Augmentation bypassed feedback attribution")
        batch["online_cp_choice_token"] = np.asarray(self._choice_tokens, dtype=np.uint64)
        batch["feedback_entry_ids"] = self._entry_ids
        batch["feedback_candidate_indices"] = self._candidate_indices
        batch["feedback_snapshot_sha256"] = self.snapshot_sha256
        for key in self._transformed_feedback[0]:
            values = [item[key] for item in self._transformed_feedback]
            batch["feedback_" + key] = torch.stack([torch.as_tensor(v) for v in values])
        return batch


class FeedbackLossObserver(torch.nn.Module):
    """No change to the segmentation objective, optimizer, or model outputs."""
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss
        self.context = None
        self.observation = None

    def forward(self, output, target):
        if self.context is not None:
            if self.observation is not None:
                raise CurriculumError("The segmentation loss was evaluated twice for one feedback event")
            logits = output[0] if isinstance(output, (tuple, list)) else output
            labels = target[0] if isinstance(target, (tuple, list)) else target
            with torch.no_grad():
                context = {key: value.to(logits.device, non_blocking=True)
                           for key, value in self.context.items()}
                self.observation = compute_feedback_metrics(logits.detach(), labels, **context)
        return self.base_loss(output, target)


class _nnUNetTrainer_250epochs_OnlineFeedback(_nnUNetTrainer_250epochs_OnlineCurriculum):
    config_environment = "ONLINE_CP_FEEDBACK_CONFIG"
    config_format = "onlinecp_segmentation_feedback_v1"
    resume_format = "onlinecp_segmentation_feedback_resume_v1"
    validate_config = staticmethod(validate_feedback_config)
    config_hash = staticmethod(feedback_config_sha256)
    epoch_stage = staticmethod(stage_for_epoch)
    online_loader_class = nnUNetDataLoaderOnlineCPFeedback
    bank_contract_filename = "feedback_contract.json"

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        self.feedback_gnn = None
        self._feedback_gnn_config = None
        self._feedback_records = []
        self._feedback_epoch_summary = None
        self._feedback_predictions = None
        self._feedback_prediction_provenance = None
        self._feedback_snapshot_sha256 = None
        self._feedback_epoch_counts = {}
        self._feedback_optimizer_steps = 0
        self._feedback_step_hook = None
        super().__init__(plans, configuration, fold, dataset_json, device)
        if (self.is_cascaded or self.label_manager.has_regions
                or self.label_manager.has_ignore_label
                or len(self.configuration_manager.patch_size) != 3
                or self.label_manager.num_segmentation_heads != 3):
            raise CurriculumError("Feedback supports the explicit non-cascaded 3-D liver 0/1/2 label experiment")
        metadata = json.loads(Path(self.online_bank_path).read_text(encoding="utf-8"))
        entries = {name: case for case, names in metadata["entries_by_case"].items() for name in names}
        if len(entries) != sum(map(len, metadata["entries_by_case"].values())):
            raise CurriculumError("Bank entry belongs to more than one recipient")
        if not set(entries.values()).issubset(self.curriculum_bank_identity["train_case_ids"]):
            raise CurriculumError("Held-out recipient in feedback table")
        self._feedback_entry_cases = entries
        self.feedback_state = FeedbackState(self.curriculum_config, entries, candidate_count=128,
                                            identity=self.curriculum_bank_identity)
        if not self.basic_control:
            if self.curriculum_config["predictions"]["mode"] != "optional":
                raise CurriculumError("Full feedback must consume trained GNN predictions; disabling them requires a separately named ablation")
            path = os.environ.get("ONLINE_CP_FEEDBACK_GNN_CONFIG", "").strip()
            if not path:
                raise CurriculumError("Full feedback requires ONLINE_CP_FEEDBACK_GNN_CONFIG; no disconnected GNN fallback")
            self._feedback_gnn_config = json.loads(Path(path).read_text(encoding="utf-8"))
            raw_root = os.environ.get("ONLINE_CP_FEEDBACK_RAW_ROOT", "").strip()
            if not raw_root or not Path(raw_root).is_dir():
                raise CurriculumError("Set ONLINE_CP_FEEDBACK_RAW_ROOT to the original verified raw dataset directory")
            self._feedback_gnn_config["raw_data_root"] = str(Path(raw_root).resolve())
            self._feedback_gnn_config["graph_cache_dir"] = str(Path(self.output_folder) / "feedback_graph_cache")

    def _code_identity(self):
        result = super()._code_identity()
        for name, symbol in {"feedback_trainer": _nnUNetTrainer_250epochs_OnlineFeedback,
                             "feedback_policy": FeedbackState,
                             "feedback_metrics": compute_feedback_metrics}.items():
            path = inspect.getsourcefile(symbol)
            if path is None:
                raise CurriculumError(f"Cannot bind source: {name}")
            result[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        # The full GNN implementation is imported from the installed project,
        # not a partial copy of its ranking code bundled into a trainer.
        import hiercp
        root = Path(hiercp.__file__).resolve().parent
        result["hiercp_sources"] = canonical_sha256({
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))
        })
        return result

    def initialize(self):
        super().initialize()
        if not isinstance(self.loss, FeedbackLossObserver):
            self.loss = FeedbackLossObserver(self.loss)
        if self._feedback_step_hook is None:
            self._feedback_step_hook = self.optimizer.register_step_post_hook(self._count_optimizer_step)
        if not self.basic_control and self.feedback_gnn is None:
            from hiercp.feedback import FeedbackGNNRuntime
            self.feedback_gnn = FeedbackGNNRuntime.from_config(
                self._feedback_gnn_config,
                bank_root=Path(self.online_bank_path).parent,
                bank_contract=self.curriculum_bank_identity,
                bank_index=json.loads(Path(self.online_bank_path).read_text(encoding="utf-8")),
                device=self.device,
            )

    def _count_optimizer_step(self, optimizer, args, kwargs):
        self._feedback_optimizer_steps += 1

    def _nnunet_progress(self):
        from hiercp.feedback import tensor_state_sha256
        return {"completed_epoch": int(self.current_epoch),
                "optimizer_steps": self._feedback_optimizer_steps,
                "network_sha256": tensor_state_sha256(self._unwrapped_network().state_dict())}

    def _runtime_identity(self):
        return {**super()._runtime_identity(), "feedback_gnn_config": self._feedback_gnn_config,
                "feedback_measurement": self.curriculum_config["difficulty"]["measurement_definition"]}

    def _loader_policy_kwargs(self):
        if self._online_dummy_2d:
            raise CurriculumError("Feedback attribution currently requires native 3-D augmentation; dummy-2D is not silently changed")
        return {**super()._loader_policy_kwargs(), "feedback_state": self.feedback_state}

    def _make_train_augmenter(self, epoch):
        snapshot = self.feedback_state.snapshot(
            epoch, predicted_difficulties=self._feedback_predictions,
            prediction_provenance=self._feedback_prediction_provenance,
        )
        self._feedback_snapshot_sha256 = canonical_sha256(snapshot)
        self._online_train_loader.feedback_state = self.feedback_state
        self._online_train_loader.snapshot_sha256 = self._feedback_snapshot_sha256
        return super()._make_train_augmenter(epoch)

    def on_train_epoch_start(self):
        self._feedback_records = []
        self._feedback_epoch_counts = {}
        return super().on_train_epoch_start()

    def train_step(self, batch):
        if batch.pop("feedback_snapshot_sha256", None) != self._feedback_snapshot_sha256:
            raise CurriculumError("Prefetched batch belongs to a different feedback snapshot")
        entries = batch.pop("feedback_entry_ids")
        candidates = batch.pop("feedback_candidate_indices")
        cases = list(batch["keys"])
        raw_flags = torch.as_tensor(batch["online_cp_applied"])
        if (raw_flags.ndim != 1 or raw_flags.is_floating_point() or raw_flags.is_complex()
                or not bool(torch.all((raw_flags == 0) | (raw_flags == 1)))):
            raise CurriculumError("Feedback applied flags must be a one-dimensional binary integer array")
        flags = raw_flags.to(dtype=torch.bool)
        if not (len(entries) == len(candidates) == len(cases) == int(flags.numel())):
            raise CurriculumError("Feedback event metadata is not aligned with the physical batch")
        train_cases = set(self.curriculum_bank_identity["train_case_ids"])
        for entry, candidate, case, applied in zip(entries, candidates, cases, flags.tolist()):
            if str(case) not in train_cases:
                raise CurriculumError("Held-out case cannot enter feedback training")
            if isinstance(candidate, (bool, np.bool_)) or not isinstance(candidate, (int, np.integer)):
                raise CurriculumError("Feedback candidate index must be an integer, not a coerced value")
            if applied:
                if self._feedback_entry_cases.get(entry) != str(case) or not 0 <= candidate < 128:
                    raise CurriculumError("Applied feedback candidate/recipient does not match the verified bank")
            elif entry != "" or candidate != -1:
                raise CurriculumError("A non-CP event cannot claim a candidate or source entry")
        context = {
            "pasted_mask": batch.pop("feedback_pasted_mask"),
            "valid_mask": batch.pop("feedback_valid_mask"),
            "event_applied": flags,
            "mask_truncated": batch.pop("feedback_mask_truncated"),
        }
        attribution = {name: batch.pop("feedback_" + name) for name in
                       ("raw_support_count", "removed_by_label_resampling", "removed_by_padding")}
        if any(key.startswith("feedback_") for key in batch):
            raise CurriculumError("Unconsumed feedback metadata; transform/trainer contract differs")
        self.loss.context, self.loss.observation = context, None
        try:
            result = super().train_step(batch)
            observation = self.loss.observation
            if observation is None:
                raise CurriculumError("Actual nnU-Net loss did not provide pre-update predictions")
            # One packed transfer per physical batch, after the ordinary update.
            names = ("foreground_ce", "boundary_error", "adjacent_fp", "status")
            packed = torch.stack([observation[name].float() for name in names], dim=1).cpu().numpy()
        finally:
            self.loss.context, self.loss.observation = None, None
        weights = np.asarray([self.curriculum_config["metrics"][name] for name in names[:3]])
        errors = (packed[:, :3] * weights).sum(1) / weights.sum()
        for i, (entry, candidate, case) in enumerate(zip(entries, candidates, cases)):
            status = int(packed[i, 3])
            self._feedback_epoch_counts[str(status)] = self._feedback_epoch_counts.get(str(status), 0) + 1
            if status == 0:
                self._feedback_records.append({
                    "entry_id": entry, "case_id": str(case), "candidate_index": int(candidate),
                    "error": float(errors[i]), "epoch": int(self.current_epoch),
                    "phase": "train", "timing": "pre_update",
                })
        for key, value in attribution.items():
            self._feedback_epoch_counts[key] = self._feedback_epoch_counts.get(key, 0) + int(value.sum())
        return result

    def on_train_epoch_end(self, train_outputs):
        self.feedback_state.update_many(self._feedback_records, int(self.current_epoch))
        nnunet_progress = self._nnunet_progress()
        gnn_report = None
        if self.feedback_gnn is not None:
            # Last segmentation update has completed. These gradients are not
            # checkpoint state and the next nnU-Net step clears them anyway.
            self.optimizer.zero_grad(set_to_none=True)
            gnn_report = self.feedback_gnn.update(self._feedback_records, int(self.current_epoch),
                                                  nnunet_progress=nnunet_progress)
            if self.current_epoch + 1 < self.num_epochs:
                self._feedback_predictions, self._feedback_prediction_provenance = self.feedback_gnn.predict(
                    int(self.current_epoch) + 1)
            else:
                self._feedback_predictions, self._feedback_prediction_provenance = None, None
        self._feedback_epoch_summary = {
            "epoch": int(self.current_epoch), "snapshot_sha256": self._feedback_snapshot_sha256,
            "nnunet_progress": nnunet_progress,
            "measurement": self.curriculum_config["difficulty"]["measurement_definition"],
            "observations": len(self._feedback_records), "status_counts": self._feedback_epoch_counts,
            "observation_sha256": canonical_sha256(self._feedback_records), "gnn": gnn_report,
        }
        result = super().on_train_epoch_end(train_outputs)
        self.print_to_log_file("[OnlineCPFeedback] " + json.dumps(self._feedback_epoch_summary,
                                                                 sort_keys=True, allow_nan=False))
        return result

    def _checkpoint_extension(self):
        return {"format": self.resume_format, "table": self.feedback_state.state_dict(),
                "gnn": None if self.feedback_gnn is None else self.feedback_gnn.state_dict(),
                "predictions": self._feedback_predictions,
                "prediction_provenance": self._feedback_prediction_provenance,
                "prediction_bundle_sha256": canonical_sha256({
                    "predictions": self._feedback_predictions,
                    "provenance": self._feedback_prediction_provenance}),
                "last_epoch": self._feedback_epoch_summary,
                "optimizer_steps": self._feedback_optimizer_steps,
                "last_observations": self._feedback_records}

    def _validate_checkpoint_extension(self, extension, next_epoch):
        keys = {"format", "table", "gnn", "predictions", "prediction_provenance", "last_epoch", "last_observations", "optimizer_steps", "prediction_bundle_sha256"}
        if not isinstance(extension, dict) or set(extension) != keys or extension["format"] != self.resume_format:
            raise CurriculumError("Missing complete feedback table/GNN/epoch checkpoint state")
        summary = extension["last_epoch"]
        if (not isinstance(summary, dict) or summary.get("epoch") != next_epoch - 1
                or summary.get("observation_sha256") != canonical_sha256(extension["last_observations"])
                or summary.get("observations") != len(extension["last_observations"])):
            raise CurriculumError("Feedback records and completed epoch disagree")
        progress = summary.get("nnunet_progress")
        if (not isinstance(progress, dict) or progress.get("completed_epoch") != next_epoch - 1
                or progress.get("optimizer_steps") != extension["optimizer_steps"]
                or not isinstance(progress.get("network_sha256"), str)
                or len(progress["network_sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in progress["network_sha256"])):
            raise CurriculumError("Saved epoch does not identify the actual completed nnU-Net update")
        if self.basic_control != (extension["gnn"] is None):
            raise CurriculumError("Feedback GNN state does not match this training arm")
        if extension["gnn"] is not None and extension["gnn"].get("completed_epoch") != next_epoch - 1:
            raise CurriculumError("Feedback GNN has not consumed this completed epoch")
        if extension["prediction_bundle_sha256"] != canonical_sha256({
                "predictions": extension["predictions"], "provenance": extension["prediction_provenance"]}):
            raise CurriculumError("Saved difficulty prediction values/provenance changed")
        if (extension["predictions"] is None) != (extension["prediction_provenance"] is None):
            raise CurriculumError("Partial difficulty prediction/provenance state")
        if extension["prediction_provenance"] is not None:
            from hiercp.feedback import tensor_state_sha256
            provenance, gnn = extension["prediction_provenance"], extension["gnn"]
            if (gnn is None or provenance["gnn_state_sha256"] != tensor_state_sha256(gnn["model"])
                    or provenance["training_observations_sha256"] != canonical_sha256(gnn["observations"])
                    or provenance["nnunet_progress"] != gnn["nnunet_progress"]):
                raise CurriculumError("Predictions are not bound to the saved GNN model and actual training observations")
        if (type(extension["optimizer_steps"]) is not int
                or not 0 <= extension["optimizer_steps"] <= next_epoch * self.num_iterations_per_epoch):
            raise CurriculumError("Invalid successful nnU-Net optimizer step count")
        # Validate a temporary table before mutating the live network or sampler.
        import copy
        table = copy.deepcopy(self.feedback_state)
        table.load_state_dict(extension["table"])
        if next_epoch < self.num_epochs:
            table.snapshot(next_epoch, predicted_difficulties=extension["predictions"],
                           prediction_provenance=extension["prediction_provenance"])
        elif extension["predictions"] is not None:
            raise CurriculumError("A completed feedback run must not contain predictions for a nonexistent next epoch")

    def _restore_checkpoint_extension(self, extension, next_epoch):
        self._validate_checkpoint_extension(extension, next_epoch)
        self.feedback_state.load_state_dict(extension["table"])
        if self.feedback_gnn is not None:
            self.feedback_gnn.load_state_dict(extension["gnn"])
        self._feedback_predictions = extension["predictions"]
        self._feedback_prediction_provenance = extension["prediction_provenance"]
        self._feedback_epoch_summary = extension["last_epoch"]
        self._feedback_records = extension["last_observations"]
        self._feedback_optimizer_steps = extension["optimizer_steps"]

    def load_checkpoint(self, filename_or_checkpoint):
        # One trusted-file read; base restore receives the already decoded dict.
        checkpoint = (torch.load(str(filename_or_checkpoint), map_location="cpu", weights_only=False)
                      if isinstance(filename_or_checkpoint, (str, os.PathLike)) else filename_or_checkpoint)
        if not isinstance(checkpoint, dict):
            raise CurriculumError("Feedback checkpoint must contain a complete state dictionary")
        from hiercp.feedback import tensor_state_sha256
        summary = checkpoint.get("onlinecp_curriculum_resume", {}).get("extension", {}).get("last_epoch")
        progress = summary.get("nnunet_progress") if isinstance(summary, dict) else None
        # An epoch can train nnU-Net but yield no usable CP observations. The
        # GNN then retains its earlier *training* lineage. Bind the student to
        # the current completed epoch separately instead of falsifying that lineage.
        if (not isinstance(progress, dict) or progress.get("network_sha256") !=
                tensor_state_sha256(checkpoint.get("network_weights", {}))):
            raise CurriculumError("Feedback epoch was not paired with these nnU-Net checkpoint weights")
        return super().load_checkpoint(checkpoint)


class nnUNetTrainer_250epochs_OnlineBasicCPFeedbackControl(_nnUNetTrainer_250epochs_OnlineFeedback):
    basic_control = True
    online_policy = "basic"


class nnUNetTrainer_250epochs_OnlineHierCPFeedback(_nnUNetTrainer_250epochs_OnlineFeedback):
    """Quality-gated, segmentation-difficulty curriculum with a live difficulty GNN."""
