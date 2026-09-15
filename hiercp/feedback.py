"""Opt-in, train-only nnU-Net observation -> full-hierarchy difficulty learning.

Quality bank scores are immutable. Difficulty is expected segmentation error
under the actual CP/augmentation schedule, not a new anatomical quality score
or an assertion that raw graphs reproduce augmented nnU-Net patches exactly.
No targets are invented for unobserved candidates. All candidates remain in
the graph; BCE is evaluated only at genuinely observed candidate positions.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from custom_trainers.onlinecp_curriculum_contract import file_sha256, value_sha256
from hiercp.data import collate_samples
from hiercp.model import FeedbackDifficultyModel, HierarchicalPyGPlacementModel
from hiercp.feedback_resources import (EXECUTION_VERSION, allocation_guard, calibration_batches,
                                      content_digest, missing_optimizer_bytes, require_allocation,
                                      sample_inventory, snapshot_guard, tensor_bytes, validate_inventory)
from hiercp.feedback_storage import FeedbackGraphStore

FORMAT = "hiercp_feedback_gnn_state_v2"
MEASUREMENT = "onlinecp_surviving_lesion_feedback_v1"


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Stable named tensor bytes; never pickle/container timestamp dependent."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Feedback RNG CUDA topology differs from saved runtime")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def validate_feedback_gnn_config(config):
    value = copy.deepcopy(dict(config))
    if value.get("format") != "hiercp_feedback_gnn_config_v1" or value.get("enabled") is not True:
        raise ValueError("Feedback GNN requires an explicit enabled v1 configuration")
    if value.get("measurement_definition") != MEASUREMENT:
        raise ValueError("Unsupported observed segmentation-difficulty definition")
    if value.get("batch_size_candidates") != "powers_of_two_to_cohort":
        raise ValueError("Feedback batches must be measured through the available source cohort")
    for name in ("seed", "calibration_repeats", "prefetch_factor", "update_passes_per_epoch",
                 "prediction_every_epochs"):
        if type(value.get(name)) is not int or value[name] < (0 if name == "seed" else 1):
            raise ValueError(f"Invalid feedback configuration: {name}")
    if value["calibration_repeats"] < 3:
        raise ValueError("Feedback calibration needs at least three measured repetitions")
    for name in ("learning_rate", "weight_decay", "grad_clip", "max_vram_fraction"):
        if (not isinstance(value.get(name), (int, float)) or isinstance(value[name], bool)
                or not math.isfinite(value[name]) or value[name] <= 0):
            raise ValueError(f"Invalid feedback configuration: {name}")
    if not 0 < value["max_vram_fraction"] < 1 or type(value.get("amp")) is not bool:
        raise ValueError("Invalid feedback memory/precision contract")
    return value


def validate_observations(records, *, epoch, entry_cases, candidate_counts, train_cases):
    if type(epoch) is not int or epoch < 0:
        raise ValueError("Feedback epoch must be a nonnegative integer")
    grouped: dict[str, list[dict]] = {}
    normalized = []
    for record in records:
        entry = record.get("entry_id")
        case = record.get("case_id")
        index, error = record.get("candidate_index"), record.get("error")
        if record.get("phase") != "train" or record.get("timing") != "pre_update":
            raise ValueError("Feedback accepts only train/pre_update observations; validation is forbidden")
        if type(record.get("epoch", epoch)) is not int or record.get("epoch", epoch) != epoch:
            raise ValueError("Feedback observation epoch differs from the completed epoch")
        if entry not in entry_cases or case != entry_cases[entry] or case not in train_cases:
            raise ValueError("Feedback entry/source/recipient is not in the verified training cohort")
        if type(index) is not int or not 0 <= index < candidate_counts[entry]:
            raise ValueError("Feedback candidate index is outside its unchanged bank pool")
        if (not isinstance(error, (int, float)) or isinstance(error, bool)
                or not math.isfinite(error) or not 0 <= error <= 1):
            raise ValueError("Observed difficulty must be finite and normalized to [0,1]")
        row = {"entry_id": entry, "case_id": case, "candidate_index": index,
               "error": float(error), "phase": "train", "timing": "pre_update", "epoch": epoch}
        normalized.append(row)
        grouped.setdefault(entry, []).append(row)
    return normalized, grouped


def observed_difficulty_loss(logits, entry_ids, grouped):
    """Vectorized soft-target BCE; duplicate real exposures keep their weight."""
    positions, targets = [], []
    offset = 0
    for entry, values in zip(entry_ids, logits):  # metadata only; no per-sample GPU loss
        for row in grouped[entry]:
            positions.append(offset + row["candidate_index"])
            targets.append(row["error"])
        offset += values.shape[0]
    if not targets or len(logits) != len(entry_ids):
        raise ValueError("A difficulty update requires aligned, actually observed targets")
    flat = torch.cat(logits).float()
    index = torch.tensor(positions, dtype=torch.long, device=flat.device)
    target = torch.tensor(targets, dtype=torch.float32, device=flat.device)
    return nn.functional.binary_cross_entropy_with_logits(flat[index], target)


class BankGraphProvider:
    """Reconstruct exactly indexed raw candidates, then reuse immutable CPU cache.

    This supports the verified single-pool online bank only. Multi-pool argmax
    banks are rejected instead of flattening away the original graph context.
    Cold graphs are committed by measured patient-parallel preparation before
    loading batches. No voxel from held-out patients is loaded.
    """

    def __init__(self, *, config, bank_root, contract, index, checkpoint, prototype):
        from hiercp.schema import graph_config_from_dict
        from hiercp.contracts import PATIENT_GRAPH_CONTRACT
        from tools.online_cp_benchmark import RAW_MARKER_NAME

        self.root = Path(bank_root).resolve(strict=True)
        self.contract = copy.deepcopy(contract)
        self.index = copy.deepcopy(index)
        self.prototype = prototype
        if checkpoint["graph_config"].get("patient_graph_contract") != PATIENT_GRAPH_CONTRACT:
            raise ValueError("Feedback graph requires explicit current patient source-content semantics; legacy host graphs cannot be reused")
        self.graph_config = graph_config_from_dict(checkpoint["graph_config"])
        self.ct_clip = tuple(checkpoint["ct_clip"])
        self.train = _json(contract["files"]["train_config"]["path"])
        self.seed = int(self.train["seed"]) + int(contract["outer_fold"])
        self.raw = Path(config["raw_data_root"]).resolve(strict=True)
        marker_path = self.raw / RAW_MARKER_NAME
        if file_sha256(marker_path) != index.get("raw_marker_sha256"):
            raise ValueError("Feedback raw dataset marker does not match the verified bank")
        marker = _json(marker_path)
        if (set(marker["train_ids"]) != set(contract["train_case_ids"])
                or set(marker["val_ids"]) != set(contract["validation_case_ids"])):
            raise ValueError("Feedback raw patient split differs from the bank contract")
        raw_records = {row["case_id"]: row for row in marker["source_cases"]}
        self.raw_paths, self.source_stats = {}, {}
        for case in contract["train_case_ids"]:
            row = raw_records[case]
            paths = (self.raw / "imagesTr" / f"{case}_0000.nii.gz",
                     self.raw / "labelsTr" / f"{case}.nii.gz")
            for path, field in zip(paths, ("image_sha256", "label_sha256")):
                if file_sha256(path) != row[field]:
                    raise ValueError(f"Feedback train-only raw content mismatch: {path}")
                self.source_stats[str(path)] = self._stat(path)
            self.raw_paths[case] = paths
        self.entry_cases = {entry: case for case, entries in index["entries_by_case"].items()
                            for entry in entries}
        if (not self.entry_cases or set(self.entry_cases) != set(contract["entry_sha256"])
                or not set(self.entry_cases.values()) <= set(contract["train_case_ids"])):
            raise ValueError("Feedback bank entry/cohort inventory differs from its contract")
        self.counts = {entry: int(index["candidate_count"]) for entry in self.entry_cases}
        code = {name: file_sha256(Path(__file__).with_name(name)) for name in
                ("local.py", "hierarchy.py", "sample.py", "schema.py", "spatial.py",
                 "region.py", "common.py", "curriculum.py", "cache.py")}
        self.binding = {"format": "hiercp_feedback_graph_binding_v2",
                        "builder_contract": "exact_bank_centers_patient_shared_v1",
                        "bank_contract_sha256": value_sha256(contract), "code": code,
                        "prototype_fingerprint": prototype.fingerprint(),
                        "graph_config": checkpoint["graph_config"], "seed": self.seed,
                        "view_policy": "bank_scoring_fixed_inference_epoch0_seed0"}
        self.cache = Path(config["graph_cache_dir"]).resolve() / value_sha256(self.binding)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.store = FeedbackGraphStore(self.cache, minimum_free_bytes=config.get("graph_cache_minimum_free_bytes"))
        self.region_cache_dir = config.get("region_cache_dir")
        if self.region_cache_dir is not None and not Path(self.region_cache_dir).is_absolute():
            raise ValueError("Feedback region cache must have an explicit absolute source path")
        self.inventory, self.preparation_report = None, None
        self.require_committed = False

    @staticmethod
    def _stat(path):
        value = Path(path).stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def _assert_sources(self, case):
        for path in self.raw_paths[case]:
            if self._stat(path) != self.source_stats[str(path)]:
                raise ValueError(f"Feedback source changed since provenance verification: {path}")

    def _canonical(self, entry, prepared=None):
        case = self.entry_cases[entry]
        self._assert_sources(case)
        source = (self.root / entry).resolve(strict=True)
        if not source.is_relative_to(self.root) or file_sha256(source) != self.contract["entry_sha256"][entry]:
            raise ValueError("Feedback bank entry content/path changed")
        if self.require_committed and not self.store.committed(entry, self.binding):
            raise ValueError("Feedback DataLoader encountered an uncommitted graph; cold work must finish before calibration")
        def build():
            from hiercp.sample import materialize_sample_views
            sample = self._build(entry, source, case, prepared=prepared)
            return materialize_sample_views(sample, training=False, epoch=0, global_seed=0)
        return self.store.get_or_build(entry, self.binding, count=self.counts[entry], case_id=case, build=build)

    def _prepare_patient(self, case_id):
        from hiercp.feedback_patient import prepare_feedback_patient
        from hiercp.region import REGION_CACHE_SEED_SALT
        from tools.online_cp_benchmark import stable_seed
        self._assert_sources(case_id)
        return prepare_feedback_patient(
            case_id, self.raw_paths[case_id], liver_label=int(self.train["labels"]["liver"]),
            tumor_label=int(self.train["labels"]["tumor"]), config=self.graph_config,
            ct_clip=self.ct_clip, seed=stable_seed(self.seed, case_id, REGION_CACHE_SEED_SALT),
            cache_root=self.cache / "patient_shared", region_cache_dir=self.region_cache_dir)

    def ensure_graph_cache(self):
        """Commit and inventory the entire bank, sharing each cold patient once.

        Case concurrency is measured against the current allocation; it changes
        neither graph/cardinality nor the observed target distribution.
        """
        from hiercp.preparation_runtime import run_case_jobs
        import nibabel as nib
        started = time.perf_counter()
        pending = {}
        rows = {}
        for entry, case in sorted(self.entry_cases.items()):
            self._assert_sources(case)
            if self.store.committed(entry, self.binding):
                rows[entry] = self.store.read(entry, self.binding, count=self.counts[entry], case_id=case, with_sample=False)
            else:
                pending.setdefault(case, []).append(entry)
        cold = sum(len(entries) for entries in pending.values())
        def prepare(case):
            context = self._prepare_patient(case)
            result = {}
            for entry in pending[case]:
                sample = self._canonical(entry, prepared=context)
                del sample
                result[entry] = self.store.read(entry, self.binding, count=self.counts[entry], case_id=case, with_sample=False)
            return result
        report_path = self.cache / ("preparation." + uuid.uuid4().hex + ".json")
        tasks = sorted(pending, key=lambda case: (-int(np.prod(nib.load(str(self.raw_paths[case][0])).shape)), case))
        if tasks:
            run_case_jobs(tasks=tasks, function=prepare, commit=rows.update, workers="auto", report_path=report_path)
        validate_inventory(rows, list(self.entry_cases))
        self.inventory = dict(sorted(rows.items()))
        self.require_committed = True
        self.preparation_report = {"format": "hiercp_feedback_graph_preparation_v2",
                                   "cold_entries": cold, "warm_entries": len(rows) - cold,
                                   "total_entries": len(rows), "candidate_counts": sorted(set(self.counts.values())),
                                   "case_shared_preparation": True, "mmap_payloads": True,
                                   "seconds": time.perf_counter() - started,
                                   "case_measurement": str(report_path) if tasks else None,
                                   "inventory_sha256": value_sha256(self.inventory)}
        return self.inventory

    def _build(self, entry, entry_path, case_id, prepared=None):
        from hiercp.cache import build_inference_sample
        from hiercp.common import CandidateInfo, context_ring_mask, slices_for_center
        from tools.online_cp_benchmark import _source_from_component, stable_seed

        with np.load(entry_path, allow_pickle=False) as stored:
            centers = np.asarray(stored["candidate_raw_centers"]).copy()
            if (centers.shape != (self.counts[entry], 3) or centers.dtype.kind not in "iu"
                    or stored["scores"].shape != (self.counts[entry],)):
                raise ValueError("Feedback requires exact single-pool candidate order, without flattening")
            component = int(stored["source_component"].item())
            if component < 1:
                raise ValueError("Feedback source component must identify a real positive tumor component")
        prepared = self._prepare_patient(case_id) if prepared is None else prepared
        case, components, regions, distance = prepared.case, prepared.components, prepared.regions, prepared.distance
        tumor_label, liver_label = int(self.train["labels"]["tumor"]), int(self.train["labels"]["liver"])
        source = _source_from_component(case, components, component, int(self.train["generation"]["source_pad"]))
        ring = context_ring_mask(source.patch_mask, width=3)
        candidates = []
        for raw in centers:
            center = tuple(int(value) for value in raw)
            slices = slices_for_center(center, source.patch_mask.shape, case.shape)
            if slices is None:
                raise ValueError("A verified raw candidate is outside the source footprint domain")
            organ = regions.full_organ_mask[slices]
            values = case.image[slices][ring & organ]
            if values.size < 8:
                values = case.image[slices][organ & ~source.patch_mask]
            if values.size == 0:
                values = case.image[slices].reshape(-1)
            candidates.append(CandidateInfo(
                center, slices, float(np.sum(source.patch_mask & (case.label[slices] == liver_label)) / source.voxel_count),
                float(regions.organ_depth[center]), float(distance[center]), float(values.mean()), float(values.std()),
            ))
        sample, _ = build_inference_sample(
            case, source, candidates, self.prototype, graph_config=self.graph_config,
            liver_label=liver_label, tumor_label=tumor_label, ct_clip=self.ct_clip,
            seed=stable_seed(self.seed, case_id, component, "score"), regions=regions,
        )
        if not np.array_equal(sample["candidate_centers"].numpy(), centers):
            raise RuntimeError("Feedback graph reconstruction changed candidate index/center binding")
        self._assert_sources(case_id)
        return sample

    def get(self, entry):
        from hiercp.sample import materialize_sample_views
        return materialize_sample_views(self._canonical(entry), training=False, epoch=0, global_seed=0)


class _GraphDataset(Dataset):
    def __init__(self, provider, entries):
        self.provider, self.entries = provider, tuple(entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        sample = self.provider.get(entry)
        if (sample["case_id"] != self.provider.entry_cases[entry]
                or len(sample["local_graphs"]) != self.provider.counts[entry]):
            raise ValueError("Feedback cached graph/source/candidate identity differs from its bank entry")
        return entry, sample


def _collate_graphs(rows):
    entries, samples = zip(*rows)
    return list(entries), collate_samples(samples)


class FeedbackGNNRuntime:
    """Epoch-boundary adapter; caller owns the atomic nnU-Net checkpoint.

    The model/optimizer are parked on CPU outside update/predict. Dedicated
    Python/NumPy/Torch RNG states are checkpointed and isolated from nnU-Net.
    A real CUDA OOM is recorded during calibration only. An OOM in actual
    training/prediction is fatal, with no smaller graph/model/CPU fallback.
    """

    def __init__(self, *, model, provider, config, identity, device, num_workers, local_chunk_size):
        self.config = validate_feedback_gnn_config(config)
        self.model, self.provider = model.cpu(), provider
        self.identity, self.device = copy.deepcopy(identity), torch.device(device)
        self.identity["feedback_execution_version"] = EXECUTION_VERSION
        if type(num_workers) is not int or num_workers < 0 or type(local_chunk_size) is not int or local_chunk_size < 1:
            raise ValueError("Feedback needs measured worker and complete local chunking settings")
        self.num_workers, self.local_chunk_size = num_workers, local_chunk_size
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["learning_rate"],
                                           weight_decay=config["weight_decay"])
        self.rng = _capture_rng()
        self.records, self.connected_parameters = [], set()
        self.completed_epoch, self.trained_through_epoch, self.optimizer_steps = -1, -1, 0
        self.nnunet_progress, self.calibration, self.last_report = None, {}, {}
        self.incomplete_update = False
        self.inventory, self.inventory_sha256, self.preparation_report = None, None, None

    @classmethod
    def from_config(cls, config, *, bank_root, bank_contract, bank_index, device):
        from hiercp.contracts import require_current_checkpoint, validate_nested_cohorts
        from hiercp.prototype import PrototypeBank

        config = validate_feedback_gnn_config(_json(config) if isinstance(config, (str, Path)) else config)
        if type(bank_index.get("candidate_count")) is not int or bank_index["candidate_count"] != 128:
            raise ValueError("Production segmentation feedback requires each complete 128-candidate bank pool")
        if (torch.device(device).type == "cuda" and config["amp"]
                and not torch.cuda.is_bf16_supported()):
            raise ValueError("Feedback AMP requires supported bfloat16 hardware; no silent precision fallback")
        for path in ("raw_data_root", "graph_cache_dir"):
            if not config.get(path) or not Path(config[path]).is_absolute():
                raise ValueError(f"Feedback launcher must supply an explicit absolute {path}")
        for name in ("gnn_checkpoint", "prototype", "train_config", "index"):
            record = bank_contract["files"][name]
            if file_sha256(record["path"]) != record["sha256"]:
                raise ValueError(f"Feedback verified artifact changed: {name}")
        if _json(bank_contract["files"]["index"]["path"]) != bank_index:
            raise ValueError("Feedback bank index differs from the verified publication")
        checkpoint = torch.load(bank_contract["files"]["gnn_checkpoint"]["path"],
                                map_location="cpu", mmap=True, weights_only=False)
        snapshot_guard(3 * tensor_bytes(checkpoint.get("state_dict", {})),
                       phase="feedback_quality_and_difficulty_initialization")
        require_current_checkpoint(checkpoint)
        if (checkpoint.get("training_complete") is not True
                or checkpoint.get("gradient_connectivity", {}).get("verified") is not True):
            raise ValueError("Feedback initialization requires a completed connected quality model")
        validate_nested_cohorts(bank_contract["train_case_ids"], bank_contract["validation_case_ids"],
                                bank_contract["gnn_train_case_ids"], bank_contract["gnn_validation_case_ids"],
                                checkpoint["prototype_training_cases"])
        prototype = PrototypeBank.load(bank_contract["files"]["prototype"]["path"])
        if (prototype.fingerprint() != checkpoint["prototype_fingerprint"]
                or list(prototype.training_case_ids) != list(checkpoint["prototype_training_cases"])):
            raise ValueError("Feedback prototype/quality lineage differs")
        if config["num_workers"] != "inherit_measured_quality_training":
            raise ValueError("Feedback worker count must inherit the explicit quality-training measurement")
        workers = checkpoint["training_signature"]["num_workers"]
        calibration = checkpoint.get("preflight_calibration", {})
        accepted = []
        for row in calibration.get("worker_trials", []):
            speed = row.get("samples_per_second")
            if row.get("status") == "accepted":
                if (type(row.get("num_workers")) is not int or row["num_workers"] < 0
                        or not isinstance(speed, (int, float)) or isinstance(speed, bool)
                        or not math.isfinite(speed) or speed <= 0):
                    raise ValueError("Quality worker calibration has an invalid accepted measurement")
                accepted.append((float(speed), row["num_workers"]))
        if (type(workers) is not int or workers < 0
                or calibration.get("format") != "hiercp_preflight_calibration_v2"
                or calibration.get("selected_num_workers") != workers
                or not accepted or max(accepted)[1] != workers):
            raise ValueError("Quality checkpoint has no matching measured worker calibration; do not guess zero workers")
        if config["local_candidate_chunk_size"] != "inherit_quality_generation":
            raise ValueError("Feedback must retain the quality generation chunking contract")
        train = _json(bank_contract["files"]["train_config"]["path"])
        chunk = train["generation"]["local_candidate_chunk_size"]
        previous_rng = _capture_rng()
        try:
            random.seed(config["seed"])
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            quality = HierarchicalPyGPlacementModel(**checkpoint["model_kwargs"])
            quality.load_state_dict(checkpoint["state_dict"])
            provider = BankGraphProvider(config=config, bank_root=bank_root, contract=bank_contract,
                                         index=bank_index, checkpoint=checkpoint, prototype=prototype)
            identity = {"bank_contract_sha256": value_sha256(bank_contract),
                        "quality_checkpoint_sha256": bank_contract["files"]["gnn_checkpoint"]["sha256"],
                        "graph_binding": provider.binding,
                        "train_case_ids": list(bank_contract["train_case_ids"]),
                        "model_kwargs": checkpoint["model_kwargs"],
                        "config_sha256": value_sha256(config), "num_workers": workers,
                        "local_chunk_size": chunk}
            return cls(model=FeedbackDifficultyModel(quality), provider=provider, config=config,
                       identity=identity, device=device, num_workers=workers, local_chunk_size=chunk)
        finally:
            _restore_rng(previous_rng)

    @contextmanager
    def _isolated(self):
        caller = _capture_rng()
        _restore_rng(self.rng)
        try:
            yield
        finally:
            self.rng = _capture_rng()
            _restore_rng(caller)

    def _loader(self, entries, size, *, shuffle=False):
        options = {"num_workers": self.num_workers, "pin_memory": self.device.type == "cuda",
                   "persistent_workers": self.num_workers > 0}
        if self.num_workers:
            options.update(prefetch_factor=self.config["prefetch_factor"], multiprocessing_context="spawn")
        return DataLoader(_GraphDataset(self.provider, entries), batch_size=size, shuffle=shuffle,
                          drop_last=False, collate_fn=_collate_graphs, **options)

    def _activate(self):
        self.model.to(self.device)

    def _activate_optimizer(self):
        # Calibration has its own lr=0 optimizer. Keeping the actual optimizer
        # parked avoids measuring two simultaneous Adam moment allocations.
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(self.device)

    def _park(self):
        from hiercp.feedback_resources import tensors
        copy_bytes = sum(value.numel() * value.element_size()
                         for value in tensors((self.model.state_dict(), self.optimizer.state_dict()))
                         if value.device.type != "cpu")
        snapshot_guard(copy_bytes, phase="feedback_park_model_and_actual_optimizer")
        self.optimizer.zero_grad(set_to_none=True)
        self.model.cpu()
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cpu()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _forward_loss(self, batch, entries, grouped):
        with torch.autocast(self.device.type, enabled=self.config["amp"] and self.device.type == "cuda",
                            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.bfloat16):
            logits = self.model(batch, local_chunk_size=self.local_chunk_size)
            loss = observed_difficulty_loss(logits, entries, grouped)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite observed difficulty loss; no optimizer update applied")
        return loss

    def _backward(self, loss, optimizer, *, record):
        loss.backward()
        active = [(name, value) for name, value in self.model.named_parameters() if value.grad is not None]
        if not active or not bool(torch.stack([torch.isfinite(value.grad).all() for _, value in active]).all()):
            raise FloatingPointError("Missing or nonfinite feedback gradients; no optimizer update applied")
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        if not bool(torch.stack([torch.isfinite(value).all() for value in self.model.parameters()]).all()):
            raise FloatingPointError("Feedback optimizer produced nonfinite parameters; epoch cannot be checkpointed")
        if record:
            self.connected_parameters.update(name for name, _ in active)
            self.optimizer_steps += 1

    def _ensure_inventory(self):
        if self.inventory is None:
            rng = _capture_rng()
            try:
                if isinstance(self.provider, BankGraphProvider):
                    rows = self.provider.ensure_graph_cache()
                    self.preparation_report = copy.deepcopy(self.provider.preparation_report)
                else:
                    # Explicit DEBUG/custom providers still materialize every
                    # supplied source. Preparation must not alter stream RNG.
                    rows = {entry: sample_inventory(self.provider.get(entry), entry_id=entry,
                                                    candidate_count=self.provider.counts[entry])
                            for entry in sorted(self.provider.entry_cases)}
                    self.preparation_report = {"format": "explicit_provider_materialization",
                                               "total_entries": len(rows), "cohort_reduced": False}
                validate_inventory(rows, list(self.provider.entry_cases))
                self.inventory = rows
                self.inventory_sha256 = value_sha256(rows)
            finally:
                _restore_rng(rng)
        return self.inventory

    def _host_guard(self, entries, size):
        proof = allocation_guard(self._ensure_inventory(), entries, size, workers=self.num_workers,
                                 prefetch_factor=self.config["prefetch_factor"],
                                 pin_memory=self.device.type == "cuda")
        require_allocation(proof)
        return proof

    def _calibrate(self, entries, grouped=None):
        from hiercp.feedback_calibration import calibrate_feedback
        return calibrate_feedback(self, entries, grouped)

    def update(self, records, epoch, *, nnunet_progress):
        from hiercp.feedback_calibration import close_loader
        if self.incomplete_update:
            raise RuntimeError("Previous feedback update failed partway; restore a complete checkpoint before continuing")
        if epoch <= self.completed_epoch:
            raise ValueError("Feedback epoch was already consumed; duplicate updates are forbidden")
        rows, grouped = validate_observations(records, epoch=epoch, entry_cases=self.provider.entry_cases,
                                              candidate_counts=self.provider.counts,
                                              train_cases=set(self.identity["train_case_ids"]))
        if (nnunet_progress.get("completed_epoch") != epoch
                or type(nnunet_progress.get("optimizer_steps")) is not int
                or nnunet_progress["optimizer_steps"] < 0
                or not _is_sha256(nnunet_progress.get("network_sha256"))):
            raise ValueError("Feedback update requires actual nnU-Net epoch/optimizer/network lineage")
        if not rows:
            self.completed_epoch = epoch
            self.last_report = {"status": "no_observations", "epoch": epoch, "optimizer_steps": 0}
            return self.last_report
        entries = sorted(grouped)
        self.incomplete_update = True
        loader = None
        with self._isolated():
            try:
                self._ensure_inventory()
                self._activate()
                report = self._calibrate(entries, grouped)
                self.calibration["update"] = report
                self._activate_optimizer()
                batches = []
                self.model.train()
                self._host_guard(entries, report["physical_batch_size"])
                loader = self._loader(entries, report["physical_batch_size"], shuffle=True)
                for _ in range(self.config["update_passes_per_epoch"]):
                    for names, batch in loader:
                        self.optimizer.zero_grad(set_to_none=True)
                        self._backward(self._forward_loss(batch, names, grouped), self.optimizer, record=True)
                        batches.append(len(names))
                self.records.extend(rows)
                self.completed_epoch = self.trained_through_epoch = epoch
                self.nnunet_progress = copy.deepcopy(nnunet_progress)
                self.incomplete_update = False
                expected = {name for name, value in self.model.named_parameters() if value.requires_grad}
                self.last_report = {"status": "updated", "epoch": epoch, "observations": len(rows),
                                    "actual_physical_batches": batches, "optimizer_steps": self.optimizer_steps,
                                    "connected_parameter_names": sorted(self.connected_parameters),
                                    "not_yet_connected_parameter_names": sorted(expected - self.connected_parameters),
                                    "all_parameters_connected": expected <= self.connected_parameters}
                return copy.deepcopy(self.last_report)
            finally:
                if loader is not None:
                    close_loader(loader)
                self._park()

    def predict(self, epoch):
        from hiercp.feedback_calibration import close_loader
        if self.incomplete_update:
            raise RuntimeError("Cannot predict from an incompletely updated feedback model")
        if type(epoch) is not int or epoch <= self.completed_epoch:
            raise ValueError("Difficulty predictions must target a later nnU-Net epoch")
        # An observation-free completed epoch is a real stream watermark, not
        # a new GNN training event. Preserve the last training provenance and
        # do not republish that older model as fresh feedback for the next epoch.
        if (not self.records or self.completed_epoch != self.trained_through_epoch
                or epoch % self.config["prediction_every_epochs"]):
            return None, None
        entries = sorted(self.provider.entry_cases)
        predictions = {}
        loader = None
        with self._isolated():
            try:
                self._ensure_inventory()
                self._activate()
                report = self._calibrate(entries)
                self.calibration["predict"] = report
                self.model.eval()
                self._host_guard(entries, report["physical_batch_size"])
                loader = self._loader(entries, report["physical_batch_size"])
                with torch.inference_mode():
                    for names, batch in loader:
                        values = self.model.predict_logits(batch, local_chunk_size=self.local_chunk_size)
                        if len(values) != len(names):
                            raise RuntimeError("Feedback prediction lost source graph alignment")
                        for entry, logits in zip(names, values):
                            if not torch.isfinite(logits).all():
                                raise FloatingPointError("Nonfinite difficulty logits cannot be hidden by sigmoid saturation")
                            value = logits.float().sigmoid().cpu()
                            if value.shape != (self.provider.counts[entry],) or not torch.isfinite(value).all():
                                raise FloatingPointError("Incomplete/nonfinite feedback candidate pool prediction")
                            predictions[entry] = value.tolist()
                if set(predictions) != set(entries):
                    raise RuntimeError("Feedback prediction did not cover every bank source and candidate")
                provenance = {"format": "hiercp_feedback_prediction_v1", "prediction_epoch": epoch,
                              "gnn_state_sha256": tensor_state_sha256(self.model.state_dict()),
                              "training_observations_sha256": value_sha256(self.records),
                              "train_case_ids": list(self.identity["train_case_ids"]),
                              "trained_through_epoch": self.trained_through_epoch,
                              "measurement_definition": MEASUREMENT,
                              "nnunet_progress": copy.deepcopy(self.nnunet_progress)}
                return predictions, provenance
            finally:
                if loader is not None:
                    close_loader(loader)
                self._park()

    def state_dict(self):
        snapshot_guard(tensor_bytes((self.model.state_dict(), self.optimizer.state_dict())),
                       phase="feedback_native_checkpoint_state_copy")
        if self.incomplete_update:
            raise RuntimeError("Cannot checkpoint an incomplete feedback update as completed training")
        return {"format": FORMAT, "architecture_version": self.model.architecture_version,
                "identity": copy.deepcopy(self.identity), "config": copy.deepcopy(self.config),
                "model": copy.deepcopy(self.model.state_dict()), "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "rng": copy.deepcopy(self.rng), "observations": copy.deepcopy(self.records),
                "completed_epoch": self.completed_epoch, "trained_through_epoch": self.trained_through_epoch,
                "optimizer_steps": self.optimizer_steps, "nnunet_progress": copy.deepcopy(self.nnunet_progress),
                "calibration": copy.deepcopy(self.calibration), "connected_parameters": sorted(self.connected_parameters)}

    def validate_state_dict(self, state):
        """Pure prevalidation for the native trainer BEFORE its state is changed."""
        if (not isinstance(state, dict) or state.get("format") != FORMAT or state.get("architecture_version") != self.model.architecture_version
                or state.get("identity") != self.identity or state.get("config") != self.config):
            raise ValueError(f"Feedback resume identity/config/architecture differs; expected {FORMAT}/"
                             f"{self.model.architecture_version}. Old source-location semantics are not migrated.")
        required = {"format", "architecture_version", "identity", "config", "model", "optimizer", "rng",
                    "observations", "completed_epoch", "trained_through_epoch", "optimizer_steps",
                    "nnunet_progress", "calibration", "connected_parameters"}
        if set(state) != required:
            raise ValueError("Feedback resume state is incomplete or has an unknown schema")
        completed, trained = state.get("completed_epoch"), state.get("trained_through_epoch")
        if type(completed) is not int or type(trained) is not int or not -1 <= trained <= completed:
            raise ValueError("Feedback resume has invalid epoch provenance")
        history = state["observations"]
        if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
            raise ValueError("Feedback resume observations have an invalid schema")
        observed_epochs = []
        for row in history:
            validate_observations([row], epoch=row.get("epoch"), entry_cases=self.provider.entry_cases,
                                  candidate_counts=self.provider.counts, train_cases=set(self.identity["train_case_ids"]))
            observed_epochs.append(row["epoch"])
        if (observed_epochs != sorted(observed_epochs) or max(observed_epochs, default=-1) != trained
                or type(state.get("optimizer_steps")) is not int or state["optimizer_steps"] < bool(history)):
            raise ValueError("Feedback resume observation/trained-epoch/optimizer lineage differs")
        if history and (not isinstance(state["nnunet_progress"], dict)
                        or state["nnunet_progress"].get("completed_epoch") != trained
                        or type(state["nnunet_progress"].get("optimizer_steps")) is not int
                        or state["nnunet_progress"]["optimizer_steps"] < 0
                        or not _is_sha256(state["nnunet_progress"].get("network_sha256"))):
            raise ValueError("Feedback resume nnU-Net provenance differs from the last real GNN update")
        expected = {name for name, value in self.model.named_parameters() if value.requires_grad}
        if (not isinstance(state["connected_parameters"], list)
                or any(not isinstance(name, str) for name in state["connected_parameters"])
                or len(state["connected_parameters"]) != len(set(state["connected_parameters"]))
                or not set(state["connected_parameters"]) <= expected
                or not isinstance(state["calibration"], dict)):
            raise ValueError("Feedback resume gradient parameter names differ from the architecture")
        current = self.model.state_dict()
        if not isinstance(state["model"], dict) or set(state["model"]) != set(current):
            raise ValueError("Feedback resume model parameter/buffer names differ")
        revision = state["model"].get("hierarchy._architecture_revision")
        if not torch.is_tensor(revision) or not torch.equal(revision.cpu(), self.model.hierarchy._architecture_revision.cpu()):
            raise ValueError("Feedback resume hierarchy revision differs from the frozen quality lineage")
        for name, reference in current.items():
            value = state["model"][name]
            if (not torch.is_tensor(value) or value.shape != reference.shape or value.dtype != reference.dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError(f"Feedback resume model tensor differs: {name}")
        optimizer = state["optimizer"]
        if not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}:
            raise ValueError("Feedback resume optimizer schema differs")
        saved_groups, groups = optimizer["param_groups"], self.optimizer.param_groups
        if not isinstance(saved_groups, list) or len(saved_groups) != len(groups) or not isinstance(optimizer["state"], dict):
            raise ValueError("Feedback resume optimizer parameter groups differ")
        canonical_groups = self.optimizer.state_dict()["param_groups"]
        if any(not isinstance(saved, dict) or saved.get("params") != canonical["params"]
               for saved, canonical in zip(saved_groups, canonical_groups)):
            raise ValueError("Feedback resume optimizer canonical parameter-ID order differs")
        parameters = {}
        for saved, live in zip(saved_groups, groups):
            if (not isinstance(saved, dict) or set(saved) != set(live)
                    or any(saved[key] != live[key] for key in live if key != "params")
                    or len(saved["params"]) != len(live["params"])):
                raise ValueError("Feedback resume optimizer options/cardinality differ")
            for identity, parameter in zip(saved["params"], live["params"]):
                if type(identity) is not int or identity < 0 or identity in parameters:
                    raise ValueError("Feedback resume optimizer parameter identity differs")
                parameters[identity] = parameter
        if not set(optimizer["state"]) <= set(parameters):
            raise ValueError("Feedback resume optimizer has an unknown parameter")
        for identity, values in optimizer["state"].items():
            if not isinstance(values, dict) or set(values) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Feedback resume AdamW state is incomplete")
            step = values["step"]
            if (not torch.is_tensor(step) or step.numel() != 1 or not bool(torch.isfinite(step).all())
                    or not 0 <= step.item() <= state["optimizer_steps"] or step.item() != int(step.item())):
                raise ValueError("Feedback resume AdamW step differs")
            for name in ("exp_avg", "exp_avg_sq"):
                value, reference = values[name], parameters[identity]
                if (not torch.is_tensor(value) or value.shape != reference.shape or value.dtype != reference.dtype
                        or not bool(torch.isfinite(value).all())):
                    raise ValueError(f"Feedback resume AdamW moment differs: {identity}/{name}")
                if name == "exp_avg_sq" and bool((value < 0).any()):
                    raise ValueError("Feedback resume AdamW second moment is negative")
        rng = state["rng"]
        if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch", "cuda"}:
            raise ValueError("Feedback resume RNG schema differs")
        # Independent generators validate the state without advancing a caller.
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(rng["numpy"])
        if (not torch.is_tensor(rng["torch"]) or rng["torch"].dtype != torch.uint8
                or rng["torch"].shape != self.rng["torch"].shape):
            raise ValueError("Feedback resume CPU RNG differs")
        torch.Generator(device="cpu").set_state(rng["torch"].cpu())
        if (not isinstance(rng["cuda"], list) or len(rng["cuda"]) != len(self.rng["cuda"])
                or any(not torch.is_tensor(value) or value.dtype != torch.uint8 or value.shape != expected.shape
                       for value, expected in zip(rng["cuda"], self.rng["cuda"]))):
            raise ValueError("Feedback resume CUDA RNG topology/state differs")
        for index, value in enumerate(rng["cuda"]):
            torch.Generator(device=f"cuda:{index}").set_state(value.cpu())

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.incomplete_update = True
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng, self.records = copy.deepcopy(state["rng"]), copy.deepcopy(state["observations"])
        self.completed_epoch, self.trained_through_epoch = state["completed_epoch"], state["trained_through_epoch"]
        self.optimizer_steps, self.nnunet_progress = state["optimizer_steps"], copy.deepcopy(state["nnunet_progress"])
        self.calibration, self.connected_parameters = copy.deepcopy(state["calibration"]), set(state["connected_parameters"])
        self._park()
        self.incomplete_update = False
