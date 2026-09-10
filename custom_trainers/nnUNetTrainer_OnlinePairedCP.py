"""Deterministic online Basic-CP and HierCP trainers for nnU-Net v2.

The trainers consume a fold-specific, single-pool OnlineCP bank. Legacy banks
retain their preprocessed paste path. New feedback banks transport the raw-target
paste through the verified native preprocessing operator into a single crop,
then use ordinary nnU-Net augmentation. No per-event full-volume resampling or
source-position mask translation is used in that path. Validation never uses CP.

Available policies:
- nnUNetTrainer_250epochs_OnlineBasicCP: uniform random over all valid candidates.
- nnUNetTrainer_250epochs_OnlineHierCP: uniform random over the GNN top-k candidates
  (legacy ablation retained for reproducibility).
- nnUNetTrainer_250epochs_OnlineHierCPExactArgmax: deterministic GNN argmax.
- no-patient/no-population exact-argmax subclasses validate derived-bank metadata.

Basic and Hier arms deliberately consume the same fixed random-number schedule so
that CP event, source, appearance jitter, crop and standard augmentation remain
paired; only the candidate-selection policy differs.
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


BANK_FORMAT = "hiercp_online_bank_v2"
TRAINER_FORMAT = "hiercp_online_trainer_v2"


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
    """Lazy reader for one fold-specific, single-pool OnlineCP bank."""

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
                f"Unsupported OnlineCP bank format in {self.index_path}: "
                f"{metadata.get('format')!r}"
            )
        self.metadata = metadata
        self.root = self.index_path.parent
        self.paste_contract = metadata.get("paste_contract")
        if self.paste_contract not in {None, "onlinecp_raw_target_paste_v1"}:
            raise OnlineCPError(f"Unsupported paste contract: {self.paste_contract!r}")
        self._raw_store = None
        self.entries_by_case: dict[str, tuple[str, ...]] = {
            str(case_id): tuple(str(value) for value in values)
            for case_id, values in metadata.get("entries_by_case", {}).items()
        }
        if not self.entries_by_case:
            raise OnlineCPError(f"OnlineCP bank contains no case entries: {self.index_path}")
        self.source_slots_by_case: dict[str, tuple[str, ...]] | None = None
        self.no_placement_sources = 0
        self._validate_source_slots()
        self.candidate_count = int(metadata["candidate_count"])
        self.hier_top_k = int(metadata.get("hier_top_k", 8))
        self.tumor_label = int(metadata["tumor_label"])
        self.liver_label = int(metadata["liver_label"])
        self.cp_probability = float(metadata["cp_probability"])
        self.intensity_scale = tuple(float(v) for v in metadata["intensity_scale_range"])
        self.intensity_shift_hu = tuple(
            float(v) for v in metadata["intensity_shift_range_hu"]
        )
        normalization = metadata.get("normalization", {})
        self.ct_mean = float(normalization.get("mean", 0.0))
        self.ct_std = float(normalization.get("std", 1.0))
        if not np.isfinite(self.ct_std) or self.ct_std <= 0:
            raise OnlineCPError(f"Invalid CT normalization std in bank: {self.ct_std}")
        if not (0.0 <= self.cp_probability <= 1.0):
            raise OnlineCPError(
                f"cp_probability must be in [0,1], got {self.cp_probability}"
            )
        if self.candidate_count < 1:
            raise OnlineCPError("candidate_count must be positive")
        if self.hier_top_k < 1:
            raise OnlineCPError("hier_top_k must be positive")
        self._cache_limit = max(1, int(cache_entries))
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def entry_names(self, case_id: str) -> tuple[str, ...]:
        return self.entries_by_case.get(str(case_id), ())

    def _validate_source_slots(self) -> None:
        policy = self.metadata.get("no_placement_policy", "error")
        if policy not in {"error", "retain_original"}:
            raise OnlineCPError(f"Unknown no-placement policy: {policy!r}")
        raw_slots = self.metadata.get("source_slots_by_case")
        if policy == "error":
            if raw_slots is not None:
                raise OnlineCPError("Source slots require explicit retain_original policy")
            return
        inventory = self.metadata.get("eligible_sources_by_case")
        if (self.metadata.get("source_schedule_format") != "onlinecp_all_source_slots_v1"
                or not isinstance(raw_slots, dict) or not isinstance(inventory, dict)
                or set(raw_slots) != set(inventory)):
            raise OnlineCPError("Complete eligible-source slot inventory is required")
        slots_by_case = {}
        no_placement = 0
        for case_id, slots in raw_slots.items():
            components = inventory[case_id]
            if (not isinstance(components, list) or len(components) != len(set(components))
                    or any(type(value) is not int or value < 1 for value in components)
                    or not isinstance(slots, list) or len(slots) != len(components)):
                raise OnlineCPError(f"Malformed source slots for {case_id}")
            names = []
            for component, slot in zip(components, slots):
                if not isinstance(slot, dict) or slot.get("source_component") != component:
                    raise OnlineCPError(f"Source slot changes original component order for {case_id}")
                entry, status = slot.get("entry"), slot.get("status")
                if status == "no_placement" and entry == "":
                    no_placement += 1
                elif status != "ok" or not isinstance(entry, str) or not entry:
                    raise OnlineCPError(f"Invalid source slot for {case_id}/{component}")
                names.append(entry)
            usable = [name for name in names if name]
            if (len(usable) != len(set(usable))
                    or sorted(usable) != sorted(self.entry_names(case_id))):
                raise OnlineCPError(f"Usable NPZ entries differ from source slots for {case_id}")
            slots_by_case[case_id] = tuple(names)
        if (set(self.entries_by_case) - set(inventory)
                or self.metadata.get("eligible_source_slots") != sum(map(len, slots_by_case.values()))
                or self.metadata.get("no_placement_sources") != no_placement):
            raise OnlineCPError("Source-slot counts or patient inventory differ")
        self.source_slots_by_case = slots_by_case
        self.no_placement_sources = no_placement

    def _load(self, relative_path: str) -> dict[str, np.ndarray]:
        cached = self._cache.get(relative_path)
        if cached is not None:
            self._cache.move_to_end(relative_path)
            return cached
        path = (self.root / relative_path).resolve()
        if path != self.root and self.root not in path.parents:
            raise OnlineCPError(f"Bank entry escapes bank root: {relative_path}")
        try:
            with np.load(path, allow_pickle=False) as payload:
                entry = {key: np.asarray(payload[key]) for key in payload.files}
        except FileNotFoundError as exc:
            raise OnlineCPError(f"Missing OnlineCP bank entry: {path}") from exc
        if self.paste_contract == "onlinecp_raw_target_paste_v1":
            self._validate_raw_entry(entry, path)
            self._cache[relative_path] = entry
            self._cache.move_to_end(relative_path)
            while len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
            return entry
        if "paste_contract" in entry:
            raise OnlineCPError(f"Entry declares a paste contract absent from its bank index: {path}")
        required = {
            "source_data",
            "source_mask",
            "anchor_offset",
            "candidate_centers",
            "scores",
        }
        missing = sorted(required - set(entry))
        if missing:
            raise OnlineCPError(f"Bank entry {path} is missing {missing}")
        source_data = entry["source_data"]
        source_mask = entry["source_mask"]
        anchor_offset = entry["anchor_offset"]
        centers = entry["candidate_centers"]
        scores = entry["scores"]
        if source_data.ndim != 4 or source_mask.ndim != 3:
            raise OnlineCPError(
                f"Bad source tensors in {path}: {source_data.shape}, {source_mask.shape}"
            )
        if tuple(source_data.shape[1:]) != tuple(source_mask.shape):
            raise OnlineCPError(f"Source data/mask shape mismatch in {path}")
        if anchor_offset.shape != (3,) or np.any(anchor_offset < 0) or np.any(
            anchor_offset >= np.asarray(source_mask.shape)
        ):
            raise OnlineCPError(
                f"Invalid anchor_offset in {path}: {anchor_offset.tolist()} "
                f"for mask={source_mask.shape}"
            )
        expected_centers = (self.candidate_count, 3)
        expected_scores = (self.candidate_count,)
        if centers.shape != expected_centers or scores.shape != expected_scores:
            raise OnlineCPError(
                f"Bad candidate tensors in {path}: centers={centers.shape} "
                f"scores={scores.shape} expected={expected_centers}/{expected_scores}"
            )
        if not np.all(np.isfinite(scores)):
            raise OnlineCPError(f"Non-finite candidate scores in {path}")
        if np.unique(centers, axis=0).shape[0] != self.candidate_count:
            raise OnlineCPError(f"Duplicate candidate centers in {path}")
        if not np.any(source_mask):
            raise OnlineCPError(f"Empty source mask in {path}")
        self._cache[relative_path] = entry
        self._cache.move_to_end(relative_path)
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return entry

    def _validate_raw_entry(self, entry, path):
        required = {"paste_contract", "case_id", "candidate_centers", "candidate_raw_centers", "scores",
                    "source_component", "source_diameter_mm", "candidate_payloads",
                    "candidate_payload_sha256", "raw_case_reference", "raw_case_reference_sha256"}
        if not required.issubset(entry) or {"source_data", "source_mask", "anchor_offset"} & set(entry):
            raise OnlineCPError(f"Raw-target entry fields conflict with its contract: {path}")
        contract = np.asarray(entry["paste_contract"])
        if contract.size != 1 or contract.dtype.kind != "U" or str(contract.reshape(-1)[0]) != self.paste_contract:
            raise OnlineCPError(f"Raw-target entry/index paste contracts differ: {path}")
        for name in ("candidate_centers", "candidate_raw_centers"):
            centers = entry[name]
            if centers.shape != (self.candidate_count, 3) or centers.dtype.kind not in "iu":
                raise OnlineCPError(f"Invalid {name} in raw-target entry: {path}")
        # Native resampling may map distinct raw candidates to the same voxel.
        # Candidate identity/order is the raw-space pool, never deduplicated here.
        if np.unique(entry["candidate_raw_centers"], axis=0).shape[0] != self.candidate_count:
            raise OnlineCPError(f"Duplicate raw candidate centers in {path}")
        scores = entry["scores"]
        if scores.shape != (self.candidate_count,) or not np.all(np.isfinite(scores)):
            raise OnlineCPError(f"Invalid candidate scores in {path}")
        component = np.asarray(entry["source_component"])
        diameter = np.asarray(entry["source_diameter_mm"])
        if component.size != 1 or component.dtype.kind not in "iu" or int(component.reshape(-1)[0]) < 1:
            raise OnlineCPError(f"Invalid raw source component in {path}")
        if diameter.size != 1 or not np.all(np.isfinite(diameter)) or float(diameter.reshape(-1)[0]) <= 0:
            raise OnlineCPError(f"Invalid raw source diameter in {path}")
        for name, count in (("candidate_payloads", self.candidate_count),
                            ("candidate_payload_sha256", self.candidate_count),
                            ("raw_case_reference", 1), ("raw_case_reference_sha256", 1), ("case_id", 1)):
            values = entry[name]
            if values.shape != (count,) or values.dtype.kind != "U" or any(not str(value) for value in values):
                raise OnlineCPError(f"Invalid raw payload references in {path}: {name}")
            if name.endswith("sha256") and any(len(str(value)) != 64 or any(c not in "0123456789abcdef" for c in str(value)) for value in values):
                raise OnlineCPError(f"Invalid raw payload SHA-256 in {path}: {name}")

    def _get_raw_store(self):
        if self._raw_store is None:
            try:
                from nnunetv2.training.nnUNetTrainer.onlinecp_raw_bank import RawBankStore
            except ModuleNotFoundError as error:
                if error.name not in {"nnunetv2", "nnunetv2.training", "nnunetv2.training.nnUNetTrainer",
                                      "nnunetv2.training.nnUNetTrainer.onlinecp_raw_bank"}:
                    raise
                from custom_trainers.onlinecp_raw_bank import RawBankStore
            self._raw_store = RawBankStore(self.root)
        return self._raw_store

    def adopt_raw_verification(self, receipt):
        """Transfer only startup SHA/stat witnesses before worker spawning.

        The caller creates this in-memory receipt after a complete source audit.
        It is never loaded from an untrusted bank JSON or checkpoint. Runtime
        loads still re-stat every file and reject a changed verification witness.
        """
        if (self.paste_contract != "onlinecp_raw_target_paste_v1"
                or not isinstance(receipt, dict) or set(receipt) != {"root", "index_sha256", "witnesses"}
                or receipt["root"] != str(self.root.resolve())
                or receipt["index_sha256"] != hashlib.sha256(self.index_path.read_bytes()).hexdigest()
                or not isinstance(receipt["witnesses"], dict)
                or (not receipt["witnesses"] and any(self.entries_by_case.values()))):
            raise OnlineCPError("Raw verification receipt does not bind this exact bank root/index")
        store = self._get_raw_store()
        if getattr(store, "_cases", {}) or getattr(store, "_sources", {}):
            raise OnlineCPError("Raw verification must be transferred before runtime payload loading")
        # Copy the small metadata map, not case/source arrays. RawBankStore's
        # __getstate__ preserves witnesses while excluding mmap descriptors.
        store._witnesses = dict(receipt["witnesses"])

    @staticmethod
    def raw_apply_function():
        try:
            from nnunetv2.training.nnUNetTrainer.onlinecp_raw_resampling import apply_candidate
        except ModuleNotFoundError as error:
            if error.name not in {"nnunetv2", "nnunetv2.training", "nnunetv2.training.nnUNetTrainer",
                                  "nnunetv2.training.nnUNetTrainer.onlinecp_raw_resampling"}:
                raise
            from custom_trainers.onlinecp_raw_resampling import apply_candidate
        return apply_candidate

    def load_raw_candidate(self, entry, candidate_index):
        if self.paste_contract != "onlinecp_raw_target_paste_v1":
            raise OnlineCPError("Raw candidate loading requires its explicit bank contract")
        if (isinstance(candidate_index, (bool, np.bool_))
                or not isinstance(candidate_index, (int, np.integer))
                or not 0 <= int(candidate_index) < self.candidate_count):
            raise OnlineCPError("Raw candidate index is outside the original pool")
        store = self._get_raw_store()
        candidate = store.load_candidate(str(entry["candidate_payloads"][candidate_index]),
                                         str(entry["candidate_payload_sha256"][candidate_index]))
        case = store.load_case(str(entry["raw_case_reference"][0]),
                               str(entry["raw_case_reference_sha256"][0]))
        case_id = str(entry["case_id"][0])
        raw_center = np.asarray(candidate.get("raw_target_center", []))
        component = candidate.get("source_component")
        if (case["metadata"].get("case_id") != case_id or candidate.get("case_id") != case_id
                or type(component) is not int or component != int(entry["source_component"].reshape(-1)[0])
                or candidate.get("case_reference_sha256") != str(entry["raw_case_reference_sha256"][0])
                or raw_center.shape != (3,) or raw_center.dtype.kind not in "iu"
                or not np.array_equal(raw_center, entry["candidate_raw_centers"][candidate_index])):
            raise OnlineCPError("Selected raw candidate changes source, destination or case identity")
        return case, candidate

    def load_for_case(self, case_id: str, entry_index: int) -> dict[str, np.ndarray]:
        names = self.entry_names(case_id)
        if not names:
            raise OnlineCPError(f"No OnlineCP source entries for case {case_id}")
        return self._load(names[int(entry_index) % len(names)])


def _select_candidate_index(
    policy: str,
    scores: np.ndarray,
    u: float,
    *,
    hier_top_k: int = 8,
) -> int:
    """Map one shared candidate draw to the requested OnlineCP policy."""
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise OnlineCPError(f"Invalid candidate score vector: {values.shape}")
    count = int(values.size)
    clipped_u = min(np.nextafter(1.0, 0.0), max(0.0, float(u)))
    if policy == "basic":
        return min(count - 1, int(np.floor(clipped_u * count)))
    if policy == "hier_argmax":
        # Deterministic; np.argmax returns the first index for an exact tie.
        return int(np.argmax(values))
    if policy == "hier":
        top_k = min(count, max(1, int(hier_top_k)))
        # Stable sort gives deterministic tie handling. The selected set is the
        # top-k by score; the shared draw then samples uniformly within that set.
        top_indices = np.argsort(values, kind="stable")[-top_k:]
        local_index = min(top_k - 1, int(np.floor(clipped_u * top_k)))
        return int(top_indices[local_index])
    raise OnlineCPError(f"Unsupported OnlineCP policy: {policy}")


class nnUNetDataLoaderOnlineCP(nnUNetDataLoader):
    """nnU-Net loader with explicit legacy/native raw-target paste dispatch."""

    def __init__(
        self,
        *args: Any,
        bank_path: str,
        policy: str,
        online_seed: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if policy not in {"basic", "hier", "hier_argmax"}:
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
        return _select_candidate_index(
            self.online_policy, scores, u, hier_top_k=self.online_bank.hier_top_k
        )

    def _source_entry_names(self, case_id: str) -> tuple[str, ...]:
        slots = getattr(self.online_bank, "source_slots_by_case", None)
        if slots is None:
            return self.online_bank.entry_names(case_id)
        # Empty names are explicit, publisher-proven non-CP source slots, not
        # missing files. They remain in the uniform source-draw denominator.
        return slots.get(str(case_id), ())

    def _load_selected_source(self, case_id: str, source_index: int):
        if getattr(self.online_bank, "source_slots_by_case", None) is None:
            return self.online_bank.load_for_case(case_id, source_index)
        name = self._source_entry_names(case_id)[source_index]
        if not name:
            raise OnlineCPError("A no-placement slot may not be loaded as a fake CP entry")
        return self.online_bank._load(name)

    def _sample_paste_plan(
        self, case_id: str
    ) -> tuple[dict[str, Any] | None, int]:
        entry_names = self._source_entry_names(case_id)
        rng = self._rng()
        apply_cp = float(rng.random()) < self.online_bank.cp_probability
        # Fixed draw schedule shared by all policies. Exact-argmax consumes but
        # ignores candidate_u so subsequent jitter/augmentation draws stay paired.
        source_u = float(rng.random())
        candidate_u = float(rng.random())
        scale_u = float(rng.random())
        shift_u = float(rng.random())
        entry_index = (min(len(entry_names) - 1, int(np.floor(source_u * len(entry_names))))
                       if entry_names else -1)
        selected_entry = entry_names[entry_index] if entry_names else ""
        schedule_token = _stable_u64(
            TRAINER_FORMAT,
            self.online_epoch,
            str(case_id),
            int(apply_cp and bool(selected_entry)),
            source_u.hex(),
            candidate_u.hex(),
            scale_u.hex(),
            shift_u.hex(),
        )
        if not apply_cp or not selected_entry:
            return None, schedule_token
        entry = self._load_selected_source(case_id, entry_index)
        centers = entry["candidate_centers"].astype(np.int64, copy=False)
        scores = entry["scores"].astype(np.float32, copy=False)
        candidate_index = self._select_candidate(scores, candidate_u)
        center = tuple(int(value) for value in centers[candidate_index])
        scale_low, scale_high = self.online_bank.intensity_scale
        shift_low, shift_high = self.online_bank.intensity_shift_hu
        scale = scale_low + scale_u * (scale_high - scale_low)
        shift_hu = shift_low + shift_u * (shift_high - shift_low)
        return self._make_paste_plan(entry, candidate_index, scale, shift_hu, case_id), schedule_token

    def _make_paste_plan(self, entry, candidate_index, scale, shift_hu, case_id):
        center = tuple(int(value) for value in entry["candidate_centers"][candidate_index])
        plan = {
            "entry": entry,
            "candidate_index": int(candidate_index),
            "center": center,
            "scale": float(scale),
        }
        if getattr(self.online_bank, "paste_contract", None) == "onlinecp_raw_target_paste_v1":
            if str(entry["case_id"][0]) != str(case_id):
                raise OnlineCPError("Selected raw source entry belongs to another recipient case")
            raw_case, candidate = self.online_bank.load_raw_candidate(entry, candidate_index)
            bbox = np.asarray(candidate["output_bbox"])
            shape = np.asarray(raw_case["metadata"]["preprocessed_shape"])
            if (bbox.shape != (3, 2) or bbox.dtype.kind not in "iu" or shape.shape != (3,)
                    or shape.dtype.kind not in "iu" or np.any(shape <= 0)
                    or np.any(bbox[:, 0] < 0) or np.any(bbox[:, 1] > shape)
                    or np.any(bbox[:, 1] <= bbox[:, 0])):
                raise OnlineCPError(f"Invalid native candidate/case geometry for {case_id}")
            plan.update(paste_contract="onlinecp_raw_target_paste_v1", raw_case=raw_case,
                        raw_candidate=candidate, shift_hu=float(shift_hu),
                        crop_source_shape=tuple(int(v) for v in bbox[:, 1] - bbox[:, 0]),
                        crop_anchor_offset=tuple(int(v) for v in np.asarray(center) - bbox[:, 0]))
        else:
            plan["normalized_offset"] = float(((scale - 1.0) * self.online_bank.ct_mean + shift_hu) / self.online_bank.ct_std)
        return plan

    def _paste_crop_geometry(self, plan, shape, case_id):
        if plan.get("paste_contract") == "onlinecp_raw_target_paste_v1":
            if tuple(plan["raw_case"]["metadata"]["preprocessed_shape"]) != tuple(shape):
                raise OnlineCPError(f"Raw reference and actual nnU-Net case shape differ: {case_id}")
            return plan["crop_source_shape"], plan["crop_anchor_offset"]
        return plan["entry"]["source_mask"].shape, plan["entry"]["anchor_offset"]

    def _raw_candidate_crop_bbox(self, plan, shape, case_id):
        """Choose a legal fixed-size native crop without trimming the raw source.

        A small native bbox remains fully visible when possible. A larger lesion
        uses ordinary partial nnU-Net cropping: its complete raw paste still
        exists, and the engine evaluates exactly the requested native crop.
        Candidate coordinates, candidate pool, source mask and RNG stay intact.
        """
        self._paste_crop_geometry(plan, shape, case_id)
        shape = np.asarray(shape, dtype=np.int64)
        patch = np.asarray(self.patch_size, dtype=np.int64)
        need_to_pad = np.asarray(self.need_to_pad, dtype=np.int64)
        if (shape.shape != (3,) or patch.shape != (3,) or need_to_pad.shape != (3,)
                or np.any(shape <= 0) or np.any(patch <= 0)):
            raise OnlineCPError("Raw CP requires the unchanged three-dimensional nnU-Net patch geometry")
        need_to_pad = np.maximum(need_to_pad, patch - shape)
        legal_lower = -need_to_pad // 2
        legal_upper = shape + need_to_pad // 2 + need_to_pad % 2 - patch
        bbox = np.asarray(plan["raw_candidate"]["output_bbox"], dtype=np.int64)
        fits = bbox[:, 1] - bbox[:, 0] <= patch
        lower_bound = np.where(fits, np.maximum(legal_lower, bbox[:, 1] - patch), legal_lower)
        upper_bound = np.where(fits, np.minimum(legal_upper, bbox[:, 0]), legal_upper)
        if np.any(lower_bound > upper_bound):
            raise OnlineCPError(f"No legal native crop intersects the selected raw candidate: {case_id}")
        # An anchor may lie in inactive raw padding. Focus the crop on the real
        # output bbox while leaving the stored raw/native anchor unchanged.
        focus = np.clip(np.asarray(plan["center"], dtype=np.int64), bbox[:, 0], bbox[:, 1] - 1)
        lower = np.clip(focus - patch // 2, lower_bound, upper_bound)
        upper = lower + patch
        if np.any(upper <= bbox[:, 0]) or np.any(lower >= bbox[:, 1]):
            raise OnlineCPError(f"Legal native crop missed the selected raw candidate bbox: {case_id}")
        return lower.astype(int).tolist(), upper.astype(int).tolist()

    def _apply_raw_paste_to_crop(self, data_cropped, seg_cropped, bbox_lbs, plan, case_id):
        shape = np.asarray(plan["raw_case"]["metadata"]["preprocessed_shape"], dtype=np.int64)
        lower = np.asarray(bbox_lbs, dtype=np.int64)
        upper = lower + np.asarray(data_cropped.shape[1:])
        valid_lower, valid_upper = np.maximum(lower, 0), np.minimum(upper, shape)
        if np.any(valid_upper <= valid_lower) or data_cropped.shape[0] != 1 or seg_cropped.shape[0] != 1:
            raise OnlineCPError(f"Raw CP requires one-channel nonempty native crop: {case_id}")
        crop_bbox = [[int(lo), int(hi)] for lo, hi in zip(valid_lower, valid_upper)]
        slices = tuple(slice(int(lo - origin), int(hi - origin)) for lo, hi, origin in zip(valid_lower, valid_upper, lower))
        reference = plan["raw_case"].get("baseline_seg")
        if (not isinstance(reference, np.ndarray) or reference.shape != (1, *tuple(shape))
                or reference.dtype.kind not in "iu"):
            raise OnlineCPError("Raw reference is missing its complete native baseline segmentation")
        reference_slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(valid_lower, valid_upper))
        if not np.array_equal(seg_cropped[(0, *slices)], reference[(0, *reference_slices)]):
            raise OnlineCPError(f"Actual nnU-Net segmentation differs from the bound native baseline: {case_id}")
        result = self.online_bank.raw_apply_function()(
            plan["raw_case"], plan["raw_candidate"], crop_bbox,
            scale=float(plan["scale"]), shift_hu=float(plan["shift_hu"]),
        )
        if not isinstance(result, dict) or not {"data", "seg", "pasted_support", "audit"}.issubset(result):
            raise OnlineCPError("Raw CP engine did not return the complete crop and attribution")
        new_data, new_seg, support = result["data"], result["seg"], result["pasted_support"]
        expected = tuple(int(v) for v in valid_upper - valid_lower)
        if (not isinstance(new_data, np.ndarray) or new_data.shape != (1, *expected) or new_data.dtype != np.float32
                or not isinstance(new_seg, np.ndarray) or new_seg.shape != (1, *expected) or new_seg.dtype != np.int16
                or not isinstance(support, np.ndarray) or support.shape != expected or support.dtype != np.bool_
                or not np.all(np.isfinite(new_data)) or not np.all(np.isin(new_seg, [-1, 0, 1, 2]))
                or not isinstance(result["audit"], dict)):
            raise OnlineCPError("Raw CP engine returned invalid crop shapes, types, labels or audit")
        baseline = seg_cropped[(0, *slices)]
        if not np.array_equal(support, (new_seg[0] == 2) & (baseline != 2)):
            raise OnlineCPError("Raw CP attribution differs from the actual recipient/native label difference")
        raw_voxels = int(np.count_nonzero(plan["raw_candidate"]["source_mask"]))
        native_voxels = int(np.count_nonzero(plan["raw_candidate"]["pasted_support"]))
        crop_voxels = int(np.count_nonzero(support))
        if raw_voxels < 1 or crop_voxels > native_voxels:
            raise OnlineCPError("Raw CP source is empty or cropped support exceeds the complete native candidate")
        # Cubic CT changes extend beyond the label support: replace the complete
        # native valid crop, preserving the original external 0/-1 padding.
        data_cropped[(slice(None), *slices)] = new_data
        seg_cropped[(slice(None), *slices)] = new_seg
        self._last_raw_pasted_support = np.zeros(data_cropped.shape[1:], dtype=bool)
        self._last_raw_pasted_support[slices] = support
        self._last_raw_paste_audit = {"raw_source_voxels": raw_voxels,
                                      "native_support_voxels": native_voxels,
                                      "crop_support_voxels": crop_voxels,
                                      "native_zero_support": int(native_voxels == 0),
                                      "crop_zero_support": int(crop_voxels == 0)}

    def _apply_paste_to_crop(
        self,
        data_cropped: np.ndarray,
        seg_cropped: np.ndarray,
        bbox_lbs: Sequence[int],
        plan: Mapping[str, Any],
        case_id: str,
    ) -> None:
        if plan.get("paste_contract") == "onlinecp_raw_target_paste_v1":
            self._apply_raw_paste_to_crop(data_cropped, seg_cropped, bbox_lbs, plan, case_id)
            return
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
        native_audits = {name: np.zeros(self.batch_size, dtype=np.int64) for name in
                         ("raw_source_voxels", "native_support_voxels", "crop_support_voxels",
                          "native_zero_support", "crop_zero_support")}
        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, case_id in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)
                    data, seg, seg_prev, properties = self._data.load_case(case_id)
                    self._last_raw_paste_audit = None
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
                        if paste_plan.get("paste_contract") == "onlinecp_raw_target_paste_v1":
                            bbox_lbs, bbox_ubs = self._raw_candidate_crop_bbox(paste_plan, shape, str(case_id))
                        else:
                            # Preserve the historical translated-patch contract.
                            source_shape, anchor_offset = self._paste_crop_geometry(paste_plan, shape, str(case_id))
                            bbox_lbs, bbox_ubs = _bbox_around_paste(
                                self, shape, paste_plan["center"], source_shape, anchor_offset,
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
                        if self._last_raw_paste_audit is not None:
                            for name, values in native_audits.items():
                                values[j] = self._last_raw_paste_audit[name]
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
            **{"online_cp_" + name: values for name, values in native_audits.items()},
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
        self._online_paste_contract = OnlineCPBank(self.online_bank_path, cache_entries=1).paste_contract
        if (self._online_paste_contract == "onlinecp_raw_target_paste_v1"
                and getattr(self, "required_paste_contract", None) != self._online_paste_contract):
            raise OnlineCPError("Raw-target banks require the Full/Basic OnlineCPFeedback trainers; this legacy trainer requires its legacy bank")
        self.online_seed = int(os.environ.get("ONLINE_CP_SEED", "42"))
        self._online_cp_events = 0
        self._online_cp_samples = 0
        self._online_native_transport = dict(raw_events=0, raw_source_voxels=0,
                                            native_support_voxels=0, crop_support_voxels=0,
                                            native_zero_support_events=0, crop_zero_support_events=0)
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
        self._online_native_transport = dict(raw_events=0, raw_source_voxels=0,
                                            native_support_voxels=0, crop_support_voxels=0,
                                            native_zero_support_events=0, crop_zero_support_events=0)
        return super().on_train_epoch_start()

    def _consume_native_transport_audit(self, batch, flags):
        fields = ("raw_source_voxels", "native_support_voxels", "crop_support_voxels",
                  "native_zero_support", "crop_zero_support")
        records = [batch.pop("online_cp_" + field, None) for field in fields]
        contract = getattr(self, "_online_paste_contract", None)
        if contract is None:
            bank = getattr(getattr(self, "_online_train_loader", None), "online_bank", None)
            contract = getattr(bank, "paste_contract", None)
        if all(record is None for record in records):
            if contract == "onlinecp_raw_target_paste_v1":
                raise OnlineCPError("Raw-target training batch is missing native transport audit")
            return  # Legacy test/checkpoint paths retain their previous batch API.
        if any(record is None for record in records) or flags is None:
            raise OnlineCPError("Incomplete native transport/event audit")

        def as_array(value):
            return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

        applied = as_array(flags)
        arrays = [as_array(record) for record in records]
        if (applied.ndim != 1 or applied.dtype.kind not in "iu" or not np.all(np.isin(applied, [0, 1]))
                or any(array.shape != applied.shape or array.dtype.kind not in "iu" or np.any(array < 0)
                       for array in arrays)):
            raise OnlineCPError("Native transport counts require nonnegative integer vectors matching CP events")
        source, native, crop, zero, crop_zero = arrays
        active = source > 0
        if (np.any(active & (applied != 1)) or np.any((native > 0) & ~active) or np.any(crop > native)
                or not np.array_equal(zero, (active & (native == 0)).astype(np.int64))
                or not np.array_equal(crop_zero, (active & (crop == 0)).astype(np.int64))
                or (contract == "onlinecp_raw_target_paste_v1" and not np.array_equal(active, applied == 1))):
            raise OnlineCPError("Native transport/source counts contradict the preserved CP event schedule")
        totals = getattr(self, "_online_native_transport", None)
        if totals is None:
            totals = self._online_native_transport = dict(raw_events=0, raw_source_voxels=0,
                                                        native_support_voxels=0, crop_support_voxels=0,
                                                        native_zero_support_events=0, crop_zero_support_events=0)
        totals["raw_events"] += int(active.sum())
        totals["raw_source_voxels"] += int(source.sum())
        totals["native_support_voxels"] += int(native.sum())
        totals["crop_support_voxels"] += int(crop.sum())
        totals["native_zero_support_events"] += int(zero.sum())
        totals["crop_zero_support_events"] += int(crop_zero.sum())

    def _log_native_transport_audit(self):
        totals = getattr(self, "_online_native_transport", {})
        if getattr(self, "_online_paste_contract", None) != "onlinecp_raw_target_paste_v1" and not totals.get("raw_events", 0):
            return
        self.print_to_log_file("[OnlineCPNativeTransport] " + json.dumps({
            "format": "onlinecp_native_transport_audit_v1", "epoch": int(self.current_epoch),
            "measurement_stage": "before_standard_augmentation", **totals,
        }, sort_keys=True), also_print_to_console=True)

    def train_step(self, batch: dict) -> dict:
        flags = batch.pop("online_cp_applied", None)
        tokens = batch.pop("online_cp_schedule_token", None)
        self._consume_native_transport_audit(batch, flags)
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
        self._log_native_transport_audit()
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

    def _selection_name(self) -> str:
        if self.online_policy == "basic":
            return "uniform-all"
        if self.online_policy == "hier_argmax":
            return "exact-gnn-argmax"
        return "uniform-gnn-top-k"

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
            f"p={bank.cp_probability:.3f} candidates={bank.candidate_count} "
            f"hier_top_k={bank.hier_top_k} "
            f"no_placement_sources={bank.no_placement_sources} "
            f"selection={self._selection_name()} "
            f"deterministic_workers={process_count}",
            also_print_to_console=True,
        )
        return mt_gen_train, mt_gen_val


class nnUNetTrainer_250epochs_OnlineBasicCP(_nnUNetTrainer_250epochs_OnlineCP):
    """Uniform random selection over the complete hard-valid candidate pool."""

    online_policy = "basic"


class nnUNetTrainer_250epochs_OnlineHierCP(_nnUNetTrainer_250epochs_OnlineCP):
    """Legacy top-k-random HierCP policy retained for exact result reproduction."""

    online_policy = "hier"


class nnUNetTrainer_250epochs_OnlineHierCPExactArgmax(
    _nnUNetTrainer_250epochs_OnlineCP
):
    """Choose the highest-scoring candidate from the complete shared bank pool."""

    online_policy = "hier_argmax"


class _OnlineExactArgmaxAblationBankGuard:
    """Reject an exact-argmax score bank produced by the wrong GNN ablation."""

    expected_ablation_mode = ""

    def get_dataloaders(self):
        try:
            metadata = json.loads(
                Path(self.online_bank_path).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise OnlineCPError(
                f"Cannot read OnlineCP ablation bank metadata: "
                f"{self.online_bank_path}: {exc}"
            ) from exc
        actual = str(metadata.get("ablation_mode", ""))
        if actual != self.expected_ablation_mode:
            raise OnlineCPError(
                "OnlineCP ablation bank mismatch: "
                f"expected={self.expected_ablation_mode!r} actual={actual!r} "
                f"path={self.online_bank_path}"
            )
        candidate_policy = str(metadata.get("candidate_policy", ""))
        if candidate_policy != "exact_gnn_argmax":
            raise OnlineCPError(
                f"Expected exact_gnn_argmax bank, got {candidate_policy!r}: "
                f"{self.online_bank_path}"
            )
        return super().get_dataloaders()


class nnUNetTrainer_250epochs_OnlineHierCPNoPatientExactArgmax(
    _OnlineExactArgmaxAblationBankGuard,
    nnUNetTrainer_250epochs_OnlineHierCPExactArgmax,
):
    """Exact-argmax OnlineCP scored by M3 without Level 1 (Patient Graph)."""

    expected_ablation_mode = "no_patient"


class nnUNetTrainer_250epochs_OnlineHierCPNoPopulationExactArgmax(
    _OnlineExactArgmaxAblationBankGuard,
    nnUNetTrainer_250epochs_OnlineHierCPExactArgmax,
):
    """Exact-argmax OnlineCP scored by M3 without Level 2 (Population Graph)."""

    expected_ablation_mode = "no_population"


# Pure-NumPy smoke hooks used by installers without constructing nnU-Net plans.
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
    basic = _select_candidate_index("basic", scores, 0.60, hier_top_k=2)
    hier_topk = _select_candidate_index("hier", scores, 0.60, hier_top_k=2)
    hier_argmax = _select_candidate_index(
        "hier_argmax", scores, 0.60, hier_top_k=2
    )
    assert basic == 2
    assert hier_topk in {1, 3}
    assert hier_argmax == 1
    return {
        "basic_index": basic,
        "hier_topk_index": hier_topk,
        "hier_argmax_index": hier_argmax,
    }
