"""Exact-argmax online Basic-CP versus HierCP trainers for nnU-Net v2.

Both trainers use the same original-only dataset, patient fold, source schedule,
proposal-pool schedule, CP-event schedule, appearance jitter and standard
nnU-Net augmentation RNG. Each source entry stores multiple independent pools
of hard-valid candidate positions and their fold-specific HierCP scores.

- OnlineBasicCPSharedPoolsV3: uniformly samples one candidate from the selected pool.
- OnlineHierCPArgmaxV3: deterministically chooses the exact GNN argmax in that pool.

There is no top-k random sampling in the HierCP arm. Copy-Paste is applied only
by the training loader; validation never sees CP.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


BANK_FORMAT = "hiercp_online_bank_argmax_v3"
TRAINER_FORMAT = "hiercp_online_trainer_argmax_v3"


class OnlineCPError(RuntimeError):
    """Online CP bank or runtime contract failure."""


def _stable_seed(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def _stable_u64(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _anchored_slices(
    center: Sequence[int],
    patch_shape: Sequence[int],
    anchor_offset: Sequence[int],
    volume_shape: Sequence[int],
) -> tuple[slice, ...] | None:
    result: list[slice] = []
    for coordinate, size, anchor, limit in zip(
        center, patch_shape, anchor_offset, volume_shape
    ):
        start = int(coordinate) - int(anchor)
        stop = start + int(size)
        if start < 0 or stop > int(limit):
            return None
        result.append(slice(start, stop))
    return tuple(result)


def _bbox_around_paste(
    loader: nnUNetDataLoader,
    data_shape: Sequence[int],
    center: Sequence[int],
    source_shape: Sequence[int],
    anchor_offset: Sequence[int],
) -> tuple[list[int], list[int]]:
    """Return a legal fixed-size crop that fully contains the pasted tumor patch."""
    shape = np.asarray(data_shape, dtype=np.int64)
    center_array = np.asarray(center, dtype=np.int64)
    source_shape_array = np.asarray(source_shape, dtype=np.int64)
    anchor = np.asarray(anchor_offset, dtype=np.int64)
    patch = np.asarray(loader.patch_size, dtype=np.int64)
    if np.any(source_shape_array > patch):
        raise OnlineCPError(
            f"Source patch {tuple(source_shape_array)} exceeds nnU-Net initial patch "
            f"{tuple(patch)}"
        )
    need_to_pad = np.asarray(loader.need_to_pad, dtype=np.int64).copy()
    for axis in range(len(shape)):
        if need_to_pad[axis] + shape[axis] < patch[axis]:
            need_to_pad[axis] = patch[axis] - shape[axis]
    legal_lower = -need_to_pad // 2
    legal_upper = shape + need_to_pad // 2 + need_to_pad % 2 - patch

    source_start = center_array - anchor
    source_stop = source_start + source_shape_array
    # To contain [source_start, source_stop), crop lower must satisfy
    # source_stop - patch <= lower <= source_start.
    containment_lower = source_stop - patch
    containment_upper = source_start
    feasible_lower = np.maximum(legal_lower, containment_lower)
    feasible_upper = np.minimum(legal_upper, containment_upper)
    if np.any(feasible_lower > feasible_upper):
        raise OnlineCPError(
            "Cannot place a legal nnU-Net crop around the online paste: "
            f"shape={tuple(shape)} center={tuple(center_array)} "
            f"source={tuple(source_shape_array)} anchor={tuple(anchor)} "
            f"patch={tuple(patch)}"
        )
    proposed = center_array - patch // 2
    lower = np.minimum(np.maximum(proposed, feasible_lower), feasible_upper)
    upper = lower + patch
    return lower.astype(int).tolist(), upper.astype(int).tolist()


class OnlineCPBank:
    """Lazy reader for one fold-specific preprocessed OnlineCP bank."""

    def __init__(self, index_path: str | os.PathLike[str], cache_entries: int = 64) -> None:
        self.index_path = Path(index_path).resolve()
        try:
            metadata = json.loads(self.index_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise OnlineCPError(f"ONLINE_CP_BANK does not exist: {self.index_path}") from exc
        except json.JSONDecodeError as exc:
            raise OnlineCPError(f"Invalid OnlineCP bank JSON: {self.index_path}: {exc}") from exc
        if not isinstance(metadata, dict) or metadata.get("format") != BANK_FORMAT:
            raise OnlineCPError(
                f"Unsupported OnlineCP bank format in {self.index_path}: {metadata.get('format')!r}"
            )
        self.metadata = metadata
        self.root = self.index_path.parent
        self.entries_by_case: dict[str, tuple[str, ...]] = {
            str(case_id): tuple(str(value) for value in values)
            for case_id, values in metadata.get("entries_by_case", {}).items()
        }
        if not self.entries_by_case:
            raise OnlineCPError(f"OnlineCP bank contains no case entries: {self.index_path}")
        self.candidate_count = int(metadata["candidate_count"])
        self.pools_per_source = int(metadata["pools_per_source"])
        self.tumor_label = int(metadata["tumor_label"])
        self.liver_label = int(metadata["liver_label"])
        self.cp_probability = float(metadata["cp_probability"])
        self.intensity_scale = tuple(float(v) for v in metadata["intensity_scale_range"])
        self.intensity_shift_hu = tuple(float(v) for v in metadata["intensity_shift_range_hu"])
        normalization = metadata.get("normalization", {})
        self.ct_mean = float(normalization.get("mean", 0.0))
        self.ct_std = float(normalization.get("std", 1.0))
        if not np.isfinite(self.ct_std) or self.ct_std <= 0:
            raise OnlineCPError(f"Invalid CT normalization std in bank: {self.ct_std}")
        if not (0.0 <= self.cp_probability <= 1.0):
            raise OnlineCPError(f"cp_probability must be in [0,1], got {self.cp_probability}")
        if self.candidate_count < 1 or self.pools_per_source < 1:
            raise OnlineCPError("candidate_count and pools_per_source must be positive")
        self._cache_limit = max(1, int(cache_entries))
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def entry_names(self, case_id: str) -> tuple[str, ...]:
        return self.entries_by_case.get(str(case_id), ())

    def _load(self, relative_path: str) -> dict[str, np.ndarray]:
        cached = self._cache.get(relative_path)
        if cached is not None:
            self._cache.move_to_end(relative_path)
            return cached
        path = (self.root / relative_path).resolve()
        if self.root not in path.parents:
            raise OnlineCPError(f"Bank entry escapes bank root: {relative_path}")
        try:
            with np.load(path, allow_pickle=False) as payload:
                entry = {key: np.asarray(payload[key]) for key in payload.files}
        except FileNotFoundError as exc:
            raise OnlineCPError(f"Missing OnlineCP bank entry: {path}") from exc
        required = {"source_data", "source_mask", "anchor_offset", "candidate_centers", "scores"}
        missing = sorted(required - set(entry))
        if missing:
            raise OnlineCPError(f"Bank entry {path} is missing {missing}")
        source_data = entry["source_data"]
        source_mask = entry["source_mask"]
        anchor_offset = entry["anchor_offset"]
        centers = entry["candidate_centers"]
        scores = entry["scores"]
        if source_data.ndim != 4 or source_mask.ndim != 3:
            raise OnlineCPError(f"Bad source tensors in {path}: {source_data.shape}, {source_mask.shape}")
        if tuple(source_data.shape[1:]) != tuple(source_mask.shape):
            raise OnlineCPError(f"Source data/mask shape mismatch in {path}")
        if anchor_offset.shape != (3,) or np.any(anchor_offset < 0) or np.any(
            anchor_offset >= np.asarray(source_mask.shape)
        ):
            raise OnlineCPError(
                f"Invalid anchor_offset in {path}: {anchor_offset.tolist()} "
                f"for mask={source_mask.shape}"
            )
        expected_centers = (self.pools_per_source, self.candidate_count, 3)
        expected_scores = (self.pools_per_source, self.candidate_count)
        if centers.shape != expected_centers or scores.shape != expected_scores:
            raise OnlineCPError(
                f"Bad multi-pool candidate tensors in {path}: "
                f"centers={centers.shape} scores={scores.shape} "
                f"expected={expected_centers}/{expected_scores}"
            )
        if not np.all(np.isfinite(scores)):
            raise OnlineCPError(f"Non-finite candidate scores in {path}")
        for pool_index in range(self.pools_per_source):
            if np.unique(centers[pool_index], axis=0).shape[0] != self.candidate_count:
                raise OnlineCPError(
                    f"Duplicate candidate centers in {path}, pool={pool_index}"
                )
        if not np.any(source_mask):
            raise OnlineCPError(f"Empty source mask in {path}")
        self._cache[relative_path] = entry
        self._cache.move_to_end(relative_path)
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return entry

    def load_for_case(self, case_id: str, entry_index: int) -> dict[str, np.ndarray]:
        names = self.entry_names(case_id)
        if not names:
            raise OnlineCPError(f"No OnlineCP source entries for case {case_id}")
        return self._load(names[int(entry_index) % len(names)])


def _select_candidate_index(policy: str, scores: np.ndarray, u: float) -> int:
    """Select within one shared proposal pool. HierCP is exact argmax."""
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise OnlineCPError(f"Invalid candidate score vector: {values.shape}")
    if policy == "basic":
        return min(values.size - 1, int(np.floor(float(u) * values.size)))
    if policy == "hier":
        # np.argmax is deterministic and uses the first index for an exact tie.
        return int(np.argmax(values))
    raise OnlineCPError(f"Unsupported OnlineCP policy: {policy}")


class nnUNetDataLoaderOnlineCP(nnUNetDataLoader):
    """nnU-Net loader that pastes one preprocessed small tumor online."""

    def __init__(
        self,
        *args: Any,
        bank_path: str,
        policy: str,
        online_seed: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if policy not in {"basic", "hier"}:
            raise OnlineCPError(f"Unsupported OnlineCP policy: {policy}")
        self.online_policy = str(policy)
        self.online_seed = int(online_seed)
        self.online_bank = OnlineCPBank(bank_path)
        self.online_epoch = 0
        self._cp_rng: np.random.Generator | None = None

    def set_epoch(self, epoch: int) -> None:
        self.online_epoch = int(epoch)
        self._cp_rng = None

    def set_thread_id(self, thread_id: int) -> None:
        super().set_thread_id(thread_id)
        worker_seed = _stable_seed(
            TRAINER_FORMAT, self.online_seed, "worker", self.online_epoch, int(thread_id)
        )
        # nnU-Net v2 transforms may consume NumPy, Python and torch RNGs inside
        # augmentation workers. Seed all three so Basic and Hier see the same
        # case/crop/standard-augmentation schedule.
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        self._cp_rng = np.random.default_rng(
            _stable_seed(
                TRAINER_FORMAT, self.online_seed, "cp", self.online_epoch, int(thread_id)
            )
        )

    def _rng(self) -> np.random.Generator:
        if self._cp_rng is None:
            worker_id = int(getattr(self, "thread_id", 0) or 0)
            self._cp_rng = np.random.default_rng(
                _stable_seed(
                    TRAINER_FORMAT, self.online_seed, "cp", self.online_epoch, worker_id
                )
            )
        return self._cp_rng

    def _select_candidate(self, scores: np.ndarray, u: float) -> int:
        return _select_candidate_index(self.online_policy, scores, u)

    def _sample_paste_plan(
        self, case_id: str
    ) -> tuple[dict[str, Any] | None, int]:
        entry_names = self.online_bank.entry_names(case_id)
        rng = self._rng()
        apply_cp = float(rng.random()) < self.online_bank.cp_probability
        # Consume the same fixed RNG schedule in both arms. The policy maps the
        # shared candidate draw to a different candidate set, but never consumes
        # a different number of random values.
        source_u = float(rng.random())
        pool_u = float(rng.random())
        # Consumed in both arms even though exact-argmax HierCP ignores it. This
        # keeps every subsequent RNG draw and standard augmentation identical.
        candidate_u = float(rng.random())
        scale_u = float(rng.random())
        shift_u = float(rng.random())
        schedule_token = _stable_u64(
            TRAINER_FORMAT,
            self.online_epoch,
            str(case_id),
            int(apply_cp and bool(entry_names)),
            source_u.hex(),
            pool_u.hex(),
            candidate_u.hex(),
            scale_u.hex(),
            shift_u.hex(),
        )
        if not apply_cp or not entry_names:
            return None, schedule_token

        entry_index = min(len(entry_names) - 1, int(np.floor(source_u * len(entry_names))))
        entry = self.online_bank.load_for_case(case_id, entry_index)
        centers_all = entry["candidate_centers"].astype(np.int64, copy=False)
        scores_all = entry["scores"].astype(np.float32, copy=False)
        pool_index = min(
            self.online_bank.pools_per_source - 1,
            int(np.floor(pool_u * self.online_bank.pools_per_source)),
        )
        centers = centers_all[pool_index]
        scores = scores_all[pool_index]
        candidate_index = self._select_candidate(scores, candidate_u)
        center = tuple(int(value) for value in centers[candidate_index])
        scale_low, scale_high = self.online_bank.intensity_scale
        shift_low, shift_high = self.online_bank.intensity_shift_hu
        scale = scale_low + scale_u * (scale_high - scale_low)
        shift_hu = shift_low + shift_u * (shift_high - shift_low)
        normalized_offset = (
            (scale - 1.0) * self.online_bank.ct_mean + shift_hu
        ) / self.online_bank.ct_std
        return {
            "entry": entry,
            "pool_index": int(pool_index),
            "candidate_index": int(candidate_index),
            "center": center,
            "scale": float(scale),
            "normalized_offset": float(normalized_offset),
        }, schedule_token

    def _apply_paste_to_crop(
        self,
        data_cropped: np.ndarray,
        seg_cropped: np.ndarray,
        bbox_lbs: Sequence[int],
        plan: Mapping[str, Any],
        case_id: str,
    ) -> None:
        entry = plan["entry"]
        source_mask = entry["source_mask"].astype(bool, copy=False)
        anchor_offset = entry["anchor_offset"].astype(np.int64, copy=False)
        full_center = np.asarray(plan["center"], dtype=np.int64)
        crop_center = full_center - np.asarray(bbox_lbs, dtype=np.int64)
        slices = _anchored_slices(
            crop_center, source_mask.shape, anchor_offset, data_cropped.shape[1:]
        )
        if slices is None:
            raise OnlineCPError(
                f"Source patch does not fit forced nnU-Net crop for {case_id}: "
                f"full_center={tuple(full_center)} crop_center={tuple(crop_center)} "
                f"source={source_mask.shape} crop={data_cropped.shape[1:]}"
            )
        source = entry["source_data"].astype(np.float32, copy=False)
        if source.shape[0] != data_cropped.shape[0]:
            raise OnlineCPError(
                f"Channel mismatch for {case_id}: "
                f"source={source.shape[0]} target={data_cropped.shape[0]}"
            )
        transformed = source * float(plan["scale"]) + float(plan["normalized_offset"])
        roi_data = data_cropped[(slice(None), *slices)]
        roi_seg = seg_cropped[(0, *slices)]
        for channel in range(data_cropped.shape[0]):
            roi_data[channel][source_mask] = transformed[channel][source_mask]
        roi_seg[source_mask] = self.online_bank.tumor_label

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = None
        seg_all = None
        cp_flags = np.zeros(self.batch_size, dtype=np.uint8)
        schedule_tokens = np.zeros(self.batch_size, dtype=np.uint64)
        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, case_id in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)
                    data, seg, seg_prev, properties = self._data.load_case(case_id)
                    paste_plan, schedule_token = self._sample_paste_plan(
                        str(case_id)
                    )
                    schedule_tokens[j] = np.uint64(schedule_token)
                    shape = data.shape[1:]
                    if paste_plan is None:
                        bbox_lbs, bbox_ubs = self.get_bbox(
                            shape, force_fg, properties["class_locations"]
                        )
                    else:
                        # Always expose the newly pasted lesion to the network.
                        # Otherwise a random crop could erase most online CP events.
                        paste_entry = paste_plan["entry"]
                        bbox_lbs, bbox_ubs = _bbox_around_paste(
                            self,
                            shape,
                            paste_plan["center"],
                            paste_entry["source_mask"].shape,
                            paste_entry["anchor_offset"],
                        )
                        cp_flags[j] = 1
                    bbox = [[lower, upper] for lower, upper in zip(bbox_lbs, bbox_ubs)]
                    # Crop directly from nnU-Net's mmap/blosc2 case. Only this
                    # patch is materialized; the complete 3-D case is never copied.
                    data_cropped_np = np.array(
                        crop_and_pad_nd(data, bbox, 0), dtype=np.float32, copy=True
                    )
                    seg_cropped_np = np.array(
                        crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16),
                        dtype=np.int16,
                        copy=True,
                    )
                    if paste_plan is not None:
                        self._apply_paste_to_crop(
                            data_cropped_np,
                            seg_cropped_np,
                            bbox_lbs,
                            paste_plan,
                            str(case_id),
                        )
                    data_cropped = torch.from_numpy(data_cropped_np).float()
                    seg_cropped = torch.from_numpy(seg_cropped_np).to(torch.int16)
                    if seg_prev is not None:
                        seg_prev_cropped = torch.from_numpy(
                            crop_and_pad_nd(seg_prev, bbox, -1, cast_cropped_to=np.int16)
                        ).to(torch.int16)
                        seg_cropped = torch.cat((seg_cropped, seg_prev_cropped[None]), dim=0)
                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        seg_cropped = seg_cropped[:, 0]
                    if self.transforms is not None:
                        transformed = self.transforms(
                            image=data_cropped, segmentation=seg_cropped
                        )
                        data_sample = transformed["image"]
                        seg_sample = transformed["segmentation"]
                    else:
                        data_sample = data_cropped
                        seg_sample = seg_cropped
                    if data_all is None:
                        data_all = torch.empty(
                            (self.batch_size, *data_sample.shape), dtype=torch.float32
                        )
                    data_all[j] = data_sample
                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [
                                torch.empty((self.batch_size, *item.shape), dtype=item.dtype)
                                for item in seg_sample
                            ]
                        for output_index, item in enumerate(seg_sample):
                            seg_all[output_index][j] = item
                    else:
                        if seg_all is None:
                            seg_all = torch.empty(
                                (self.batch_size, *seg_sample.shape), dtype=seg_sample.dtype
                            )
                        seg_all[j] = seg_sample
        return {
            "data": data_all,
            "target": seg_all,
            "keys": selected_keys,
            "online_cp_applied": cp_flags,
            "online_cp_schedule_token": schedule_tokens,
        }


class _nnUNetTrainer_250epochs_OnlineCP(nnUNetTrainer):
    online_policy = ""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Set the training horizon before initialize() creates PolyLRScheduler.
        # This is a real 250-epoch schedule, not a 1000-epoch run stopped early.
        self.num_epochs = 250
        bank = os.environ.get("ONLINE_CP_BANK", "").strip()
        if not bank:
            raise OnlineCPError(
                "ONLINE_CP_BANK must point to the fold-specific bank index.json"
            )
        self.online_bank_path = str(Path(bank).resolve())
        self.online_seed = int(os.environ.get("ONLINE_CP_SEED", "42"))
        self._online_cp_events = 0
        self._online_cp_samples = 0
        self._online_schedule_hash = 0xCBF29CE484222325
        self._online_train_loader = None
        self._online_train_augmenter = None
        self._online_active_epoch = None
        self._online_process_count = None

    def _make_train_augmenter(self, epoch: int):
        if self._online_train_loader is None or self._online_process_count is None:
            raise OnlineCPError("OnlineCP train loader is not initialized")
        self._online_train_loader.set_epoch(int(epoch))
        process_count = int(self._online_process_count)
        if process_count == 0:
            self._online_train_loader.set_thread_id(0)
            return SingleThreadedAugmenter(self._online_train_loader, None)
        train_seeds = [
            _stable_seed(
                TRAINER_FORMAT, self.online_seed, "train", int(epoch), index
            )
            for index in range(process_count)
        ]
        return MultiThreadedAugmenter(
            data_loader=self._online_train_loader,
            transform=None,
            num_processes=process_count,
            num_cached_per_queue=max(
                2, max(6, process_count // 2) // process_count
            ),
            seeds=train_seeds,
            pin_memory=self.device.type == "cuda",
            wait_time=0.002,
        )

    def on_train_epoch_start(self):
        # Epoch-specific worker seeds make the online CP and standard nnU-Net
        # augmentation schedules resume-safe. Restarting flushes prefetched
        # batches from the preceding epoch. One warm-up batch is discarded in
        # every epoch, including epoch 0, so uninterrupted and resumed runs use
        # the same schedule for a given epoch.
        if (
            self._online_train_loader is not None
            and self._online_train_augmenter is not None
            and self._online_active_epoch != int(self.current_epoch)
        ):
            if isinstance(self._online_train_augmenter, MultiThreadedAugmenter):
                self._online_train_augmenter._finish()
            self._online_train_augmenter = self._make_train_augmenter(
                int(self.current_epoch)
            )
            self.dataloader_train = self._online_train_augmenter
            _ = next(self._online_train_augmenter)
            self._online_active_epoch = int(self.current_epoch)
        self._online_cp_events = 0
        self._online_cp_samples = 0
        self._online_schedule_hash = 0xCBF29CE484222325
        return super().on_train_epoch_start()

    def train_step(self, batch: dict) -> dict:
        flags = batch.pop("online_cp_applied", None)
        tokens = batch.pop("online_cp_schedule_token", None)
        if flags is not None:
            if torch.is_tensor(flags):
                values = flags.detach().cpu().numpy().astype(np.int64, copy=False)
            else:
                values = np.asarray(flags, dtype=np.int64)
            values = values.reshape(-1)
            self._online_cp_events += int(values.sum())
            self._online_cp_samples += int(values.size)
        if tokens is not None:
            if torch.is_tensor(tokens):
                token_values = tokens.detach().cpu().numpy().astype(
                    np.uint64, copy=False
                )
            else:
                token_values = np.asarray(tokens, dtype=np.uint64)
            for token in token_values.reshape(-1):
                self._online_schedule_hash ^= int(token)
                self._online_schedule_hash = (
                    self._online_schedule_hash * 0x100000001B3
                ) & 0xFFFFFFFFFFFFFFFF
        return super().train_step(batch)

    def on_train_epoch_end(self, train_outputs: list[dict[str, object]]):
        result = super().on_train_epoch_end(train_outputs)
        rate = (
            self._online_cp_events / self._online_cp_samples
            if self._online_cp_samples > 0
            else 0.0
        )
        self.print_to_log_file(
            f"[OnlineCP] epoch={self.current_epoch} applied="
            f"{self._online_cp_events}/{self._online_cp_samples} rate={rate:.4f} "
            f"schedule={self._online_schedule_hash:016x}",
            also_print_to_console=True,
        )
        return result

    def initialize(self):
        # The two arms start from exactly the same network initialization.
        random.seed(self.online_seed)
        np.random.seed(self.online_seed)
        torch.manual_seed(self.online_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.online_seed)
        return super().initialize()

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = nnUNetDataLoaderOnlineCP(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            bank_path=self.online_bank_path,
            policy=self.online_policy,
            online_seed=self.online_seed,
        )
        dl_tr.set_epoch(int(self.current_epoch))
        dl_val = nnUNetDataLoader(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        process_count = get_allowed_n_proc_DA()
        val_count = max(1, process_count // 2) if process_count > 0 else 0
        val_seeds = [
            _stable_seed(TRAINER_FORMAT, self.online_seed, "val", index)
            for index in range(val_count)
        ]
        if process_count == 0:
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_val = MultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=val_count,
                num_cached_per_queue=max(
                    2, max(3, process_count // 4) // val_count
                ),
                seeds=val_seeds,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        self._online_train_loader = dl_tr
        self._online_process_count = int(process_count)
        self._online_train_augmenter = self._make_train_augmenter(
            int(self.current_epoch)
        )
        mt_gen_train = self._online_train_augmenter
        self._online_active_epoch = int(self.current_epoch)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        bank = OnlineCPBank(self.online_bank_path, cache_entries=1)
        self.print_to_log_file(
            "[OnlineCP] "
            f"policy={self.online_policy} bank={self.online_bank_path} "
            f"p={bank.cp_probability:.3f} pools={bank.pools_per_source} "
            f"candidates/pool={bank.candidate_count} "
            f"selection={'uniform-within-pool' if self.online_policy == 'basic' else 'exact-gnn-argmax'} "
            f"deterministic_workers={process_count}",
            also_print_to_console=True,
        )
        return mt_gen_train, mt_gen_val


class nnUNetTrainer_250epochs_OnlineBasicCPSharedPoolsV3(_nnUNetTrainer_250epochs_OnlineCP):
    online_policy = "basic"


class nnUNetTrainer_250epochs_OnlineHierCPArgmaxV3(_nnUNetTrainer_250epochs_OnlineCP):
    online_policy = "hier"


# Pure numpy smoke hooks used by the installer without constructing nnU-Net plans.
def _smoke_paste() -> dict[str, Any]:
    source = np.zeros((1, 5, 5, 5), dtype=np.float32)
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    source[0, mask] = 2.0
    target = np.zeros((1, 15, 15, 15), dtype=np.float32)
    seg = np.ones((1, 15, 15, 15), dtype=np.int16)
    center = (7, 7, 7)
    anchor = np.asarray(mask.shape, dtype=np.int64) // 2
    slices = _anchored_slices(center, mask.shape, anchor, target.shape[1:])
    assert slices is not None
    target[(0, *slices)][mask] = source[0][mask]
    seg[(0, *slices)][mask] = 2
    assert int(np.count_nonzero(seg == 2)) == int(mask.sum())
    assert float(target.max()) == 2.0
    return {"tumor_voxels": int(mask.sum()), "center": center}


def _smoke_policy() -> dict[str, int]:
    scores = np.asarray([0.2, 3.0, 1.0, 2.0], dtype=np.float32)
    basic = _select_candidate_index("basic", scores, 0.60)
    hier = _select_candidate_index("hier", scores, 0.60)
    assert basic == 2
    assert hier == 1
    return {"basic_index": basic, "hier_argmax_index": hier}
