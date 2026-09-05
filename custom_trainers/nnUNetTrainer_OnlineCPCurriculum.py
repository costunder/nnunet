"""New opt-in curriculum experiment; the two original trainer modules are untouched.

This module is installed beside the original v2 trainer and its two new contract
helpers. No global monkeypatch changes a legacy trainer or loader.
"""
from __future__ import annotations

import json
import hashlib
import inspect
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCP import (
    TRAINER_FORMAT, OnlineCPBank, _nnUNetTrainer_250epochs_OnlineCP,
    _stable_seed, nnUNetDataLoaderOnlineCP,
)
from nnunetv2.training.nnUNetTrainer.onlinecp_curriculum_contract import (
    verify_curriculum_bank_contract,
)
from nnunetv2.training.nnUNetTrainer.onlinecp_curriculum_policy import (
    CONFIG_FORMAT, FNV_OFFSET, RESUME_FORMAT, SCHEDULE_FORMAT, CurriculumError,
    canonical_sha256, curriculum_config_sha256, schedule_token,
    select_curriculum_candidate, stage_for_epoch, update_digest,
    validate_curriculum_config,
)
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class nnUNetDataLoaderOnlineCPCurriculum(nnUNetDataLoaderOnlineCP):
    def __init__(self, *args: Any, curriculum_config: Mapping[str, Any],
                 basic_control: bool, **kwargs: Any) -> None:
        self.curriculum_config = validate_curriculum_config(curriculum_config)
        self.curriculum_sha256 = curriculum_config_sha256(self.curriculum_config)
        self.basic_control = bool(basic_control)
        self._choice_tokens: list[int] = []
        super().__init__(*args, **kwargs)
        if self.online_bank.candidate_count != self.curriculum_config["candidate_count"]:
            raise CurriculumError("Curriculum may not change the bank candidate count")
        if self.online_bank.cp_probability != self.curriculum_config["cp_probability"]:
            raise CurriculumError("Curriculum may not change the paired CP event probability")

    def _select_candidate(self, scores: np.ndarray, u: float) -> int:
        return select_curriculum_candidate(
            scores, self.curriculum_config, int(self.online_epoch), u,
            basic_control=self.basic_control,
        )

    def _sample_paste_plan(self, case_id: str):
        # The original method consumes exactly the original five RNG draws and
        # calls our selection hook. The event schedule is independent of choice.
        plan, original_event = super()._sample_paste_plan(case_id)
        event = schedule_token("event", self.curriculum_sha256, int(original_event))
        choice = (None if plan is None else
                  [int(plan["candidate_index"]), [int(v) for v in plan["center"]]])
        self._choice_tokens.append(schedule_token(
            "choice", self.curriculum_sha256, event, choice,
        ))
        return plan, event

    def generate_train_batch(self):
        self._choice_tokens = []
        batch = super().generate_train_batch()
        if len(self._choice_tokens) != len(batch["online_cp_applied"]):
            raise CurriculumError("Every sampled item must have an audited candidate choice")
        batch["online_cp_choice_token"] = np.asarray(self._choice_tokens, dtype=np.uint64)
        return batch


class _EpochValidationLoader(nnUNetDataLoader):
    """Validation RNG streams restart by epoch, not by process restart history."""
    def set_validation_epoch(self, epoch: int, seed: int) -> None:
        self.validation_epoch, self.validation_seed = int(epoch), int(seed)

    def set_thread_id(self, thread_id: int) -> None:
        super().set_thread_id(thread_id)
        seed = _stable_seed(SCHEDULE_FORMAT, self.validation_seed, "val",
                            self.validation_epoch, int(thread_id))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


class _nnUNetTrainer_250epochs_OnlineCurriculum(_nnUNetTrainer_250epochs_OnlineCP):
    basic_control = False
    expected_ablation_mode = "full"
    online_policy = "hier_argmax"  # accepted legacy loader ID; selection is overridden above
    config_environment = "ONLINE_CP_CURRICULUM_CONFIG"
    config_format = CONFIG_FORMAT
    resume_format = RESUME_FORMAT
    validate_config = staticmethod(validate_curriculum_config)
    config_hash = staticmethod(curriculum_config_sha256)
    epoch_stage = staticmethod(stage_for_epoch)
    online_loader_class = nnUNetDataLoaderOnlineCPCurriculum
    bank_contract_filename = "curriculum_contract.json"

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")) -> None:
        if type(fold) is not int or fold < 0:
            raise CurriculumError("An explicit integer nnU-Net fold is required; 'all' is unsafe")
        path = os.environ.get(self.config_environment, "").strip()
        if not path:
            raise CurriculumError(f"Set {self.config_environment} to an explicit new-experiment JSON")
        self.curriculum_config = self.validate_config(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )
        self.curriculum_sha256 = self.config_hash(self.curriculum_config)
        self.curriculum_dataset_name = str(plans.get("dataset_name", ""))
        self._curriculum_plans_sha256 = canonical_sha256(plans)
        self._curriculum_dataset_json_sha256 = canonical_sha256(dataset_json)
        self._curriculum_configuration = str(configuration)
        if not self.curriculum_dataset_name:
            raise CurriculumError("Plans must declare the exact nnU-Net dataset_name")
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.is_ddp:
            raise CurriculumError("Curriculum checkpoint replay currently requires the verified single-device path; DDP is not silently substituted")
        self._curriculum_code_identity = self._code_identity()
        self.curriculum_bank_identity = self._verify_bank()
        self._choice_digest = FNV_OFFSET
        self._last_epoch_record = None
        self._checkpoint_boundary = None
        self._resume_loaded = False
        self._existing_checkpoints = any(Path(self.output_folder).glob("checkpoint_*.pth"))
        self._curriculum_val_loader = None
        self._curriculum_val_augmenter = None
        self._curriculum_val_epoch = None
        self._curriculum_val_count = None
        self.print_to_log_file(
            "[OnlineCPCurriculumContract] " + json.dumps({
                "format": self.config_format, "schedule_format": SCHEDULE_FORMAT,
                "config": self.curriculum_config, "config_sha256": self.curriculum_sha256,
                "bank_identity_sha256": canonical_sha256(self.curriculum_bank_identity),
                "trainer": self.__class__.__name__, "basic_control": self.basic_control,
                "legacy_rng_draw_schedule_preserved": True,
            }, sort_keys=True), also_print_to_console=True,
        )

    def _verify_bank(self) -> dict:
        contract_options = ({} if self.bank_contract_filename == "curriculum_contract.json" else
                            {"contract_filename": self.bank_contract_filename})
        identity = verify_curriculum_bank_contract(
            self.online_bank_path, curriculum_sha256=self.curriculum_sha256,
            expected_candidate_count=self.curriculum_config["candidate_count"],
            dataset_name=self.curriculum_dataset_name, nnunet_fold=int(self.fold),
            **contract_options,
        )
        if not isinstance(identity, dict):
            raise CurriculumError("Bank verifier must return a complete stable identity")
        for name in ("train_case_ids", "validation_case_ids"):
            ids = identity.get(name)
            if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
                raise CurriculumError(f"Verified bank identity lacks exact {name}")
        if set(identity["train_case_ids"]) & set(identity["validation_case_ids"]):
            raise CurriculumError("Training and validation patients overlap")
        metadata = json.loads(Path(self.online_bank_path).read_text(encoding="utf-8"))
        if metadata.get("ablation_mode", "full") != self.expected_ablation_mode:
            raise CurriculumError("Bank scores come from the wrong model ablation")
        bank = OnlineCPBank(self.online_bank_path, cache_entries=1)
        if (bank.candidate_count != 128 or bank.cp_probability != 0.5):
            raise CurriculumError("The new selection policy must retain 128 candidates and CP probability 0.5")
        canonical_sha256(identity)
        return identity

    def _code_identity(self) -> dict:
        sources = {"curriculum_trainer": _nnUNetTrainer_250epochs_OnlineCurriculum,
                   "curriculum_policy": select_curriculum_candidate,
                   "bank_verifier": verify_curriculum_bank_contract,
                   "legacy_trainer": _nnUNetTrainer_250epochs_OnlineCP,
                   "base_trainer": nnUNetTrainer}
        result = {}
        for name, symbol in sources.items():
            source = inspect.getsourcefile(symbol)
            if source is None or not Path(source).is_file():
                raise CurriculumError(f"Cannot bind installed source for {name}")
            result[name] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        return result

    def _runtime_identity(self) -> dict:
        if self._code_identity() != self._curriculum_code_identity:
            raise CurriculumError("Installed source files changed during this trainer process")
        return {
            "trainer": self.__class__.__name__, "online_seed": int(self.online_seed),
            "source_identity": self._curriculum_code_identity,
            "plans_sha256": self._curriculum_plans_sha256,
            "dataset_json_sha256": self._curriculum_dataset_json_sha256,
            "configuration": self._curriculum_configuration,
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "num_epochs": int(self.num_epochs), "gradient_accumulation_steps": 1,
            "physical_batch_size": int(self.batch_size),
            "train_iterations_per_epoch": int(self.num_iterations_per_epoch),
            "validation_iterations_per_epoch": int(self.num_val_iterations_per_epoch),
            "augmentation_workers": int(get_allowed_n_proc_DA()),
            "device_type": self.device.type,
            "cuda_device_count": int(torch.cuda.device_count()) if self.device.type == "cuda" else 0,
        }

    def on_train_epoch_start(self):
        if self._existing_checkpoints and not self._resume_loaded:
            raise CurriculumError("Existing checkpoints require explicit verified resume; refusing a fresh restart over existing results")
        self.epoch_stage(self.curriculum_config, int(self.current_epoch))
        self._choice_digest = FNV_OFFSET
        return super().on_train_epoch_start()

    def train_step(self, batch: dict) -> dict:
        required = {"online_cp_choice_token", "online_cp_schedule_token", "online_cp_applied"}
        if not required.issubset(batch):
            raise CurriculumError("The curriculum loader/audit path was bypassed")
        choices = batch.pop("online_cp_choice_token")
        if torch.is_tensor(choices):
            choices = choices.detach().cpu().numpy()
        if np.asarray(choices).shape != np.asarray(batch["online_cp_applied"]).shape:
            raise CurriculumError("Choice and event audit shapes differ")
        result = super().train_step(batch)
        self._choice_digest = update_digest(self._choice_digest, choices)
        return result

    def on_train_epoch_end(self, train_outputs):
        # Deliberately do not emit the legacy [OnlineCP] schedule format.
        result = nnUNetTrainer.on_train_epoch_end(self, train_outputs)
        stage, _ = self.epoch_stage(self.curriculum_config, int(self.current_epoch))
        self._last_epoch_record = {
            "epoch": int(self.current_epoch), "stage": stage,
            "applied": int(self._online_cp_events), "samples": int(self._online_cp_samples),
            "event_digest": f"{self._online_schedule_hash:016x}",
            "choice_digest": f"{self._choice_digest:016x}",
        }
        self.print_to_log_file(
            "[OnlineCPCurriculum] " + json.dumps({
                **self._last_epoch_record, "config_sha256": self.curriculum_sha256,
                "schedule_format": SCHEDULE_FORMAT, "basic_control": self.basic_control,
            }, sort_keys=True), also_print_to_console=True,
        )
        return result

    def on_epoch_end(self):
        # Base nnU-Net saves latest/best here, after train AND validation.
        self._checkpoint_boundary = int(self.current_epoch) + 1
        return super().on_epoch_end()

    def _make_validation_augmenter(self, epoch: int):
        loader, count = self._curriculum_val_loader, self._curriculum_val_count
        loader.set_validation_epoch(epoch, self.online_seed)
        if count == 0:
            loader.set_thread_id(0)
            return SingleThreadedAugmenter(loader, None)
        return MultiThreadedAugmenter(
            loader, None, num_processes=count,
            num_cached_per_queue=max(2, max(3, count // 2) // count),
            seeds=[_stable_seed(SCHEDULE_FORMAT, self.online_seed, "val", epoch, i)
                   for i in range(count)], pin_memory=self.device.type == "cuda", wait_time=0.002,
        )

    def on_validation_epoch_start(self):
        if self._curriculum_val_epoch != int(self.current_epoch):
            if isinstance(self._curriculum_val_augmenter, MultiThreadedAugmenter):
                self._curriculum_val_augmenter._finish()
            self._curriculum_val_augmenter = self._make_validation_augmenter(int(self.current_epoch))
            self.dataloader_val = self._curriculum_val_augmenter
            _ = next(self.dataloader_val)
            self._curriculum_val_epoch = int(self.current_epoch)
        return super().on_validation_epoch_start()

    def get_dataloaders(self):
        identity = self._verify_bank()
        if identity != self.curriculum_bank_identity:
            raise CurriculumError("Verified bank/cohort changed before loader construction")
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch = self.configuration_manager.patch_size
        scales = self._get_deep_supervision_scales()
        rotation, dummy_2d, initial_patch, mirrors = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        self._online_dummy_2d = bool(dummy_2d)
        regions = self.label_manager.foreground_regions if self.label_manager.has_regions else None
        train_transforms = self.get_training_transforms(
            patch, rotation, scales, mirrors, dummy_2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=regions, ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=regions, ignore_label=self.label_manager.ignore_label,
        )
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        if (set(dataset_tr.keys()) != set(identity["train_case_ids"])
                or set(dataset_val.keys()) != set(identity["validation_case_ids"])):
            raise CurriculumError("Actual nnU-Net loader cohorts differ from the verified bank split")
        common = dict(oversample_foreground_percent=self.oversample_foreground_percent,
                      sampling_probabilities=None, pad_sides=None,
                      probabilistic_oversampling=self.probabilistic_oversampling)
        dl_tr = self.online_loader_class(
            dataset_tr, self.batch_size, initial_patch, patch, self.label_manager,
            transforms=train_transforms, bank_path=self.online_bank_path,
            policy=self.online_policy, online_seed=self.online_seed,
            **self._loader_policy_kwargs(), **common,
        )
        dl_val = _EpochValidationLoader(dataset_val, self.batch_size, patch, patch,
                                       self.label_manager, transforms=val_transforms, **common)
        count = int(get_allowed_n_proc_DA())
        self._online_train_loader, self._online_process_count = dl_tr, count
        self._online_train_augmenter = self._make_train_augmenter(int(self.current_epoch))
        self._online_active_epoch = int(self.current_epoch)
        self._curriculum_val_loader = dl_val
        self._curriculum_val_count = max(1, count // 2) if count > 0 else 0
        self._curriculum_val_augmenter = self._make_validation_augmenter(int(self.current_epoch))
        self._curriculum_val_epoch = int(self.current_epoch)
        _ = next(self._online_train_augmenter)
        _ = next(self._curriculum_val_augmenter)
        self.print_to_log_file("[OnlineCPCurriculumRuntime] " + json.dumps(
            self._runtime_identity(), sort_keys=True), also_print_to_console=True)
        return self._online_train_augmenter, self._curriculum_val_augmenter

    def _loader_policy_kwargs(self):
        return dict(curriculum_config=self.curriculum_config, basic_control=self.basic_control)

    def _checkpoint_extension(self):
        return {}

    def _validate_checkpoint_extension(self, extension, next_epoch):
        if extension:
            raise CurriculumError("This trainer cannot restore another policy's checkpoint extension")

    def _restore_checkpoint_extension(self, extension, next_epoch):
        self._validate_checkpoint_extension(extension, next_epoch)

    def _unwrapped_network(self):
        network = self.network.module if self.is_ddp else self.network
        return getattr(network, "_orig_mod", network)

    @classmethod
    def _validate_epoch_record(cls, record, config, next_epoch, expected_samples=None):
        required = {"epoch", "stage", "applied", "samples", "event_digest", "choice_digest"}
        if not isinstance(record, dict) or set(record) != required:
            raise CurriculumError("Checkpoint lacks a complete last-epoch candidate/event audit")
        if any(type(record[key]) is not int for key in ("epoch", "stage", "applied", "samples")):
            raise CurriculumError("Checkpoint last-epoch audit counters must be integers")
        stage, _ = cls.epoch_stage(config, next_epoch - 1)
        if (record["epoch"] != next_epoch - 1 or record["stage"] != stage
                or record["samples"] <= 0 or not 0 <= record["applied"] <= record["samples"]
                or (expected_samples is not None and record["samples"] != expected_samples)):
            raise CurriculumError("Checkpoint last-epoch stage or complete sample count disagrees")
        for name in ("event_digest", "choice_digest"):
            value = record[name]
            if (not isinstance(value, str) or len(value) != 16
                    or any(char not in "0123456789abcdef" for char in value)):
                raise CurriculumError(f"Checkpoint last-epoch {name} is not a complete uint64 digest")

    def save_checkpoint(self, filename: str) -> None:
        if self.disable_checkpointing:
            raise CurriculumError("This experiment requires recoverable curriculum checkpoints")
        next_epoch = int(self.current_epoch) + 1
        if (self._checkpoint_boundary != next_epoch or self._last_epoch_record is None
                or self._last_epoch_record["epoch"] != next_epoch - 1):
            raise CurriculumError("Only complete train+validation epoch boundaries may be checkpointed")
        self._validate_epoch_record(
            self._last_epoch_record, self.curriculum_config, next_epoch,
            int(self.num_iterations_per_epoch) * int(self.batch_size),
        )
        state = {
            "format": self.resume_format, "next_epoch": next_epoch,
            "config": self.curriculum_config, "config_sha256": self.curriculum_sha256,
            "bank_identity": self.curriculum_bank_identity,
            "runtime_identity": self._runtime_identity(), "last_epoch": self._last_epoch_record,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            "lr_scheduler_state": self.lr_scheduler.state_dict(),
            "extension": self._checkpoint_extension(),
        }
        checkpoint = {
            "network_weights": self._unwrapped_network().state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            "logging": self.logger.get_checkpoint(), "_best_ema": self._best_ema,
            "current_epoch": next_epoch, "init_args": self.my_init_kwargs,
            "trainer_name": self.__class__.__name__,
            "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
            "onlinecp_curriculum_resume": state,
        }
        target = Path(filename).absolute()
        if target.parent.resolve() != Path(self.output_folder).resolve():
            raise CurriculumError("Checkpoint destination must stay in this trainer's own fold output")
        descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                torch.save(checkpoint, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        # Only load trusted checkpoints produced by this project. Pickle-based
        # optimizer/RNG state is not an interchange format for untrusted files.
        checkpoint = (torch.load(str(filename_or_checkpoint), map_location="cpu", weights_only=False)
                      if isinstance(filename_or_checkpoint, (str, os.PathLike))
                      else filename_or_checkpoint)
        required = {"network_weights", "optimizer_state", "grad_scaler_state", "logging", "_best_ema",
                    "current_epoch", "init_args", "trainer_name", "inference_allowed_mirroring_axes",
                    "onlinecp_curriculum_resume"}
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            raise CurriculumError("Checkpoint lacks complete curriculum/network/optimizer resume state; no fresh-start fallback")
        state = checkpoint["onlinecp_curriculum_resume"]
        state_keys = {"format", "next_epoch", "config", "config_sha256", "bank_identity",
                      "runtime_identity", "last_epoch", "python_rng", "numpy_rng", "cpu_rng",
                      "cuda_rng", "lr_scheduler_state"}
        if not isinstance(state, dict) or not state_keys.issubset(state) or state["format"] != self.resume_format:
            raise CurriculumError("Malformed or legacy curriculum resume state")
        epoch = state["next_epoch"]
        if (type(epoch) is not int or not 1 <= epoch <= 250 or checkpoint["current_epoch"] != epoch
                or not isinstance(state["last_epoch"], dict)
                or state["last_epoch"].get("epoch") != epoch - 1):
            raise CurriculumError("Checkpoint epoch and completed epoch audit disagree")
        self._validate_epoch_record(state["last_epoch"], self.curriculum_config, epoch)
        self._validate_checkpoint_extension(state.get("extension", {}), epoch)
        if (checkpoint["trainer_name"] != self.__class__.__name__
                or state["config"] != self.curriculum_config
                or state["config_sha256"] != self.curriculum_sha256
                or state["bank_identity"] != self.curriculum_bank_identity
                or self._verify_bank() != self.curriculum_bank_identity):
            raise CurriculumError("Checkpoint trainer, curriculum or bank/cohort identity changed")
        if not self.was_initialized:
            self.initialize()
        if state["runtime_identity"] != self._runtime_identity():
            raise CurriculumError("Resume physical batch, iterations, workers, seed or device contract changed")
        self._validate_epoch_record(
            state["last_epoch"], self.curriculum_config, epoch,
            int(self.num_iterations_per_epoch) * int(self.batch_size),
        )
        cpu_rng = state["cpu_rng"]
        cuda_rng = state["cuda_rng"]
        if (not torch.is_tensor(cpu_rng) or cpu_rng.dtype != torch.uint8 or cpu_rng.ndim != 1
                or not isinstance(cuda_rng, list)
                or len(cuda_rng) != self._runtime_identity()["cuda_device_count"]
                or any(not torch.is_tensor(v) or v.dtype != torch.uint8 or v.ndim != 1 for v in cuda_rng)):
            raise CurriculumError("Missing or invalid CPU/CUDA RNG ByteTensor state")
        if (self.grad_scaler is None) != (checkpoint["grad_scaler_state"] is None):
            raise CurriculumError("Resume AMP scaler availability changed")
        self._unwrapped_network().load_state_dict(checkpoint["network_weights"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler_state"])
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]
        self.inference_allowed_mirroring_axes = checkpoint["inference_allowed_mirroring_axes"]
        self.my_init_kwargs = checkpoint["init_args"]
        self.current_epoch = epoch
        self._restore_checkpoint_extension(state.get("extension", {}), epoch)
        self._last_epoch_record = state["last_epoch"]
        self._checkpoint_boundary = epoch
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(cpu_rng.cpu())
        if cuda_rng:
            torch.cuda.set_rng_state_all([v.cpu() for v in cuda_rng])
        self._resume_loaded = True
        self.print_to_log_file(
            f"[OnlineCPCurriculumResume] restored epoch={epoch} config_sha256={self.curriculum_sha256} "
            "network/optimizer/scaler/scheduler/RNG restored; augmentation restarts by epoch; "
            "bitwise CUDA replay is not asserted", also_print_to_console=True,
        )


class nnUNetTrainer_250epochs_OnlineBasicCPCurriculumControl(_nnUNetTrainer_250epochs_OnlineCurriculum):
    basic_control = True
    online_policy = "basic"


class nnUNetTrainer_250epochs_OnlineHierCPCurriculum(_nnUNetTrainer_250epochs_OnlineCurriculum):
    """Stage-dependent rank/temperature sampling, never an implicit argmax fallback."""


class nnUNetTrainer_250epochs_OnlineHierCPNoPatientCurriculum(_nnUNetTrainer_250epochs_OnlineCurriculum):
    expected_ablation_mode = "no_patient"


class nnUNetTrainer_250epochs_OnlineHierCPNoPopulationCurriculum(_nnUNetTrainer_250epochs_OnlineCurriculum):
    expected_ablation_mode = "no_population"
