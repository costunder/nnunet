"""Seven independent full250 Basic CP interventions; frozen baseline is read-only.

Use BASIC_CP_EXPANDED_ARM and BASIC_CP_EXPANDED_PREFLIGHT alongside the
historical ONLINE_CP_BANK/ONLINE_CP_SEED. Source draws, CP gates, intensity
jitter, model, optimizer and spatial augmentation retain the historical code.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from . import nnUNetTrainer_BasicCPAblations as base

FORMAT = 'basic_cp_expanded_one_factor_full250_v1'
EXTENSION_FORMAT = 'basic_cp_expanded_candidates_preserved128_v1'
ARMS = ('multi5', 'multi7', 'multi10', 'candidates512', 'candidates1024', 'candidates2048', 'blend50')
HISTORICAL_SHA = base.HISTORICAL_SHA
BANK_SHA = base.BANK_SHA
OnlineCPError = base.OnlineCPError
require = base.require
sha = base.sha
historical = base.historical


def copies_per_event(arm):
    require(arm in ARMS, 'Explicit expanded Basic CP arm required')
    return int(arm[5:]) if arm.startswith('multi') else 1


def candidates_per_source(arm):
    require(arm in ARMS, 'Explicit expanded Basic CP arm required')
    return int(arm[10:]) if arm.startswith('candidates') else 128


def validate_metadata(metadata, arm):
    count = candidates_per_source(arm)
    require(metadata.get('candidate_count') == count and metadata.get('total_candidates') == 642 * count,
            'All642 source entries and exact requested candidate cardinality required')
    # Validate all fixed factors with the unmodified historical baseline guard.
    original_factors = dict(metadata, candidate_count=128, total_candidates=642 * 128)
    base.validate_metadata(original_factors, 'multi3')
    if count != 128:
        extension = metadata.get('ablation', {})
        for key, value in {'format': EXTENSION_FORMAT, 'uniform_candidate_count': count,
                           'scores_used': False, 'historical_score_count': 128,
                           'original_candidate_count': 128, 'original_index_sha256': BANK_SHA,
                           'additional_GNN_scores_generated': False,
                           'historical_graph_geometry_required': True}.items():
            require(extension.get(key) == value, f'Original128 preservation contract differs: {key}')


class ExpandedBank(historical.OnlineCPBank):
    def __init__(self, index_path, cache_entries=642, *, arm=None):
        self.arm = str(arm or os.environ.get('BASIC_CP_EXPANDED_ARM', ''))
        super().__init__(index_path, cache_entries=cache_entries)
        validate_metadata(self.metadata, self.arm)
        if self.candidate_count == 128:
            require(sha(self.index_path) == BANK_SHA, 'Actual historical bank index SHA differs')
        self._validated_entries = set()
        self._entry_sha256 = None
        self._patient_geometry = {}
        witness_path = os.environ.get('BASIC_CP_EXPANDED_PREFLIGHT', '')
        if witness_path and Path(witness_path).is_file():
            witness = json.loads(Path(witness_path).read_text(encoding='utf8'))
            require(witness.get('format') == FORMAT and witness.get('arm') == self.arm
                    and witness.get('index_sha256') == sha(self.index_path) and witness.get('complete') is True,
                    'Expanded preflight belongs to another bank/arm')
            self._entry_sha256 = witness.get('entries_sha256')
            require(isinstance(self._entry_sha256, dict) and len(self._entry_sha256) == 642,
                    'All642 actual source-entry SHA witnesses required')

    def _load(self, relative_path):
        if relative_path not in self._cache and self._entry_sha256 is not None:
            require(sha(self.root / relative_path) == self._entry_sha256.get(relative_path),
                    'Source NPZ bytes changed after complete expanded preflight')
        # This private baseline reader changes only the original score/count
        # guard; the real128 scores are preserved and never fabricated.
        entry = (base._load256(self, relative_path) if self.candidate_count != 128
                 else super()._load(relative_path))
        if relative_path not in self._validated_entries:
            centers, scores = entry['candidate_centers'], entry['scores']
            require(centers.dtype.kind in 'iu' and centers.shape == (self.candidate_count, 3)
                    and np.unique(centers, axis=0).shape[0] == self.candidate_count,
                    'All requested distinct integer preprocessed centers required')
            require(scores.shape == (128,) and np.isfinite(scores).all(), 'Original real128 finite scores required')
            require(np.isfinite(entry['source_data']).all(), 'Source image has nonfinite values')
            require(entry['anchor_offset'].dtype.kind in 'iu', 'Integer source anchor required')
            self._validated_entries.add(relative_path)
        return entry

    def case_entries(self, case_id):
        return tuple(self._load(name) for name in self.entry_names(case_id))

    def geometry_for_case(self, case_id):
        key = str(case_id)
        if key not in self._patient_geometry:
            self._patient_geometry[key] = CandidateGeometry(self.case_entries(key))
        return self._patient_geometry[key]


class CandidateGeometry:
    """All same-patient placements, cached bounds and exact overlap constraints.

    Static vectorization handles candidate/source geometry. Exact mask tests
    are needed only for intersecting bounding boxes and cached per placement.
    All candidates remain admitted to the search; there is no search cap.
    """
    def __init__(self, entries):
        require(bool(entries), 'Actual own-patient source entries required')
        self.entries = tuple(entries)
        self.entry_ids = np.concatenate([np.full(len(e['candidate_centers']), i, dtype=np.int64)
                                         for i, e in enumerate(self.entries)])
        self.centers = np.concatenate([e['candidate_centers'] for e in self.entries]).astype(np.int64, copy=False)
        self.starts = np.concatenate([e['candidate_centers'] - e['anchor_offset'] for e in self.entries])
        self.stops = np.concatenate([e['candidate_centers'] - e['anchor_offset']
                                    + np.asarray(e['source_mask'].shape) for e in self.entries])
        self._conflicts = {}
        self._entry_offsets = np.cumsum([0, *[len(e['candidate_centers']) for e in self.entries]])

    def primary_index(self, primary):
        for i, entry in enumerate(self.entries):
            if entry is primary['entry']:
                hits = np.flatnonzero(np.all(entry['candidate_centers'] == primary['center'], axis=1))
                require(hits.size == 1, 'Historical primary must have exactly one unchanged source/center identity')
                return int(self._entry_offsets[i] + hits[0])
        raise OnlineCPError('Primary source is outside this patient; donor substitution forbidden')

    def conflicts(self, index):
        if index not in self._conflicts:
            conflicts = np.all(self.centers == self.centers[index], axis=1)
            overlaps = np.flatnonzero(~conflicts & np.all(self.starts < self.stops[index], axis=1)
                                     & np.all(self.stops > self.starts[index], axis=1))
            chosen = self.entries[int(self.entry_ids[index])]
            for other in overlaps:
                if base.mask_overlap(chosen, self.centers[index],
                                     self.entries[int(self.entry_ids[other])], self.centers[other]):
                    conflicts[other] = True
            self._conflicts[index] = conflicts
        return self._conflicts[index]

    def randomized_indices(self, valid, rng):
        source_ids = np.unique(self.entry_ids[valid])
        if rng is not None:
            source_ids = rng.permutation(source_ids)
        for source_id in source_ids:
            indices = np.flatnonzero(valid & (self.entry_ids == source_id))
            if rng is not None:
                indices = rng.permutation(indices)
            yield from indices

    def choose(self, primary, lower, upper, count, rng=None, *, maximum_union_shape=(128, 128, 128)):
        require(type(count) is int and count in (5, 7, 10), 'Requested full five/seven/ten paste count required')
        lower, upper, maximum = np.asarray(lower), np.asarray(upper), np.asarray(maximum_union_shape)
        primary_index = self.primary_index(primary)
        require(np.all(self.starts[primary_index] >= lower) and np.all(self.stops[primary_index] <= upper),
                'Original primary source patch must fit the actual patient')
        require(np.all(self.stops[primary_index] - self.starts[primary_index] <= maximum),
                'Original full primary patch must fit network128')
        valid = np.all(self.starts >= lower, axis=1) & np.all(self.stops <= upper, axis=1)
        valid &= ~self.conflicts(primary_index)
        selected = [primary_index]
        order = np.asarray(list(self.randomized_indices(valid, rng)), dtype=np.int64)
        rank = np.full(len(valid), -1, dtype=np.int64)
        rank[order] = np.arange(len(order))

        def search(available, union_lower, union_upper):
            if len(selected) == count:
                return selected[1:].copy()
            available = available & np.all(np.maximum(self.stops, union_upper)
                                          - np.minimum(self.starts, union_lower) <= maximum, axis=1)
            if int(np.count_nonzero(available)) < count - len(selected):
                return None
            # All finite branches are explored until a complete solution is
            # found. Impossible geometry raises rather than reducing N.
            for index in order[available[order]]:
                next_lower = np.minimum(union_lower, self.starts[index])
                next_upper = np.maximum(union_upper, self.stops[index])
                selected.append(int(index))
                # A fixed randomized rank visits every unordered combination
                # once, avoiding factorially repeated source permutations.
                result = search(available & ~self.conflicts(int(index)) & (rank > rank[index]),
                                next_lower, next_upper)
                selected.pop()
                if result is not None:
                    return result
            return None

        result = search(valid, self.starts[primary_index], self.stops[primary_index])
        require(result is not None, f'Full {count}-paste geometry unavailable for unchanged primary; no partial paste fallback')
        return [{'entry': self.entries[int(self.entry_ids[index])],
                 'center': tuple(int(v) for v in self.centers[index])} for index in result]


def select_extra_pastes(entries, primary, lower, upper, rng=None, *, copies,
                        maximum_union_shape=(128, 128, 128), geometry=None):
    geometry = CandidateGeometry(entries) if geometry is None else geometry
    return geometry.choose(primary, lower, upper, copies, rng, maximum_union_shape=maximum_union_shape)


def joint_network_crop(shape, plans, patch=(128, 128, 128)):
    patch = np.asarray(patch, dtype=np.int64)
    require(tuple(patch) == (128, 128, 128) and len(plans) in (5, 7, 10), 'Requested fullN network128 crop required')
    union_lower, union_upper = base.paste_union(plans)
    require(np.all(union_upper - union_lower <= patch), 'All full source patches must fit network128 union')
    legal_lower, legal_upper = base.legal_crop_bounds(shape, patch, np.zeros(3, dtype=np.int64))
    feasible_lower = np.maximum(legal_lower, union_upper - patch)
    feasible_upper = np.minimum(legal_upper, union_lower)
    require(np.all(feasible_lower <= feasible_upper), 'No legal network128 crop contains fullN union')
    proposed, _ = base.canonical_crop(shape, plans[0], patch)
    lower = np.clip(proposed, feasible_lower, feasible_upper)
    return lower, lower + patch


def prepare_joint_plan(loader, shape, plan, case_id):
    if 'expanded_extra_plans' not in plan:
        rng = np.random.default_rng(historical._stable_seed(FORMAT, loader.arm, 'extra', loader.online_seed,
                                    loader.online_epoch, case_id, plan['expanded_schedule_token']))
        geometry = loader.online_bank.geometry_for_case(case_id)
        extras = geometry.choose(plan, np.zeros(3, dtype=np.int64), np.asarray(shape),
                                 copies_per_event(loader.arm), rng)
        for extra in extras:
            scale_low, scale_high = loader.online_bank.intensity_scale
            shift_low, shift_high = loader.online_bank.intensity_shift_hu
            scale = scale_low + float(rng.random()) * (scale_high - scale_low)
            shift_hu = shift_low + float(rng.random()) * (shift_high - shift_low)
            extra.update(scale=scale, normalized_offset=((scale - 1.) * loader.online_bank.ct_mean
                                                        + shift_hu) / loader.online_bank.ct_std)
        plans = [plan, *extras]
        lower, upper = joint_network_crop(shape, plans)
        plan.update(expanded_extra_plans=extras, expanded_joint128_lower=tuple(map(int, lower)),
                    expanded_joint128_upper=tuple(map(int, upper)))
    return [plan, *plan['expanded_extra_plans']]


def _remember_case_shape(loader, shape, center, source_shape, anchor):
    loader._expanded_case_shape = tuple(map(int, shape))
    if copies_per_event(loader.arm) > 1:
        plan = loader._pending_paste_plan
        require(plan is not None and tuple(center) == plan['center']
                and tuple(source_shape) == plan['entry']['source_mask'].shape
                and np.array_equal(anchor, plan['entry']['anchor_offset']),
                'Joint crop must retain this exact historical primary source/location')
        plans = prepare_joint_plan(loader, shape, plan, loader._pending_case_id)
        lower, upper, base_lower = base.initial_crop_around_joint(loader, shape, plans, plan['expanded_joint128_lower'])
        plan['expanded_joint128_lower'] = tuple(map(int, base_lower))
        plan['expanded_joint128_upper'] = tuple(map(int, base_lower + 128))
        return lower, upper
    return historical._bbox_around_paste(loader, shape, center, source_shape, anchor)


_generate = base.private_function(historical.nnUNetDataLoaderOnlineCP.generate_train_batch,
                                  _bbox_around_paste=_remember_case_shape)
_initialize_loader = base.private_function(historical.nnUNetDataLoaderOnlineCP.__init__, OnlineCPBank=ExpandedBank)


class ExpandedDataLoader(historical.nnUNetDataLoaderOnlineCP):
    def __init__(self, *args, bank_path, policy, online_seed, **kwargs):
        require(policy == 'basic' and online_seed == 42, 'Basic uniform policy and seed42 required')
        _initialize_loader(self, *args, bank_path=bank_path, policy=policy, online_seed=online_seed, **kwargs)
        self.arm = self.online_bank.arm

    def _select_candidate(self, scores, u):
        count = self.online_bank.candidate_count
        return min(count - 1, int(np.floor(float(u) * count)))

    def _sample_paste_plan(self, case_id):
        plan, token = super()._sample_paste_plan(case_id)
        if plan is not None:
            plan['expanded_schedule_token'] = int(token)
        self._pending_paste_plan, self._pending_case_id = plan, str(case_id)
        return plan, token

    def _apply_paste_to_crop(self, data_cropped, seg_cropped, bbox_lbs, plan, case_id):
        if self.arm.startswith('candidates'):
            return super()._apply_paste_to_crop(data_cropped, seg_cropped, bbox_lbs, plan, case_id)
        if self.arm == 'blend50':
            entry = plan['entry']
            mask = entry['source_mask'].astype(bool, copy=False)
            slices = historical._anchored_slices(np.asarray(plan['center']) - np.asarray(bbox_lbs),
                         mask.shape, entry['anchor_offset'], data_cropped.shape[1:])
            require(slices is not None and entry['source_data'].shape[0] == data_cropped.shape[0],
                    'Whole blend50 source and original channels must fit')
            require(not np.any(seg_cropped[(0, *slices)][mask] == self.online_bank.tumor_label),
                    'Blend50 paste overlaps existing tumor')
            donor = entry['source_data'].astype(np.float32, copy=False) * float(plan['scale']) + float(plan['normalized_offset'])
            roi = data_cropped[(slice(None), *slices)]
            roi[:, mask] = .5 * roi[:, mask] + .5 * donor[:, mask]
            seg_cropped[(0, *slices)][mask] = self.online_bank.tumor_label
            return
        plans = prepare_joint_plan(self, self._expanded_case_shape, plan, case_id)
        lower, upper = np.asarray(plan['expanded_joint128_lower']), np.asarray(plan['expanded_joint128_upper'])
        crop_lower = np.asarray(bbox_lbs)
        require(len(plans) == copies_per_event(self.arm) and np.all(lower >= crop_lower)
                and np.all(upper <= crop_lower + np.asarray(data_cropped.shape[1:])),
                'FullN joint network128 must be inside original augmentation crop')
        # Validate every source before writing any image or segmentation bytes.
        for i, item in enumerate(plans):
            entry = item['entry']
            slices = historical._anchored_slices(np.asarray(item['center']) - crop_lower,
                         entry['source_mask'].shape, entry['anchor_offset'], data_cropped.shape[1:])
            require(slices is not None and entry['source_data'].shape[0] == data_cropped.shape[0],
                    'All requested complete sources must fit original crop/channels')
            require(not np.any(seg_cropped[(0, *slices)][entry['source_mask'].astype(bool)]
                               == self.online_bank.tumor_label), 'A new paste overlaps existing tumor')
            for previous in plans[:i]:
                require(not base.mask_overlap(entry, item['center'], previous['entry'], previous['center']),
                        'Requested pasted masks must be pairwise nonoverlapping')
        for item in plans:
            super()._apply_paste_to_crop(data_cropped, seg_cropped, bbox_lbs, item, case_id)

    def generate_train_batch(self):
        batch = _generate(self)
        batch['online_cp_pasted_lesions'] = np.asarray(batch['online_cp_applied'], dtype=np.uint8) * copies_per_event(self.arm)
        return batch


_get_dataloaders = base.private_function(historical._nnUNetTrainer_250epochs_OnlineCP.get_dataloaders,
                                       nnUNetDataLoaderOnlineCP=ExpandedDataLoader, OnlineCPBank=ExpandedBank)


class _ExpandedBasicCP(historical.nnUNetTrainer_250epochs_OnlineBasicCP):
    expanded_arm = None

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: historical.torch.device = historical.torch.device('cuda')):
        require(os.environ.get('BASIC_CP_EXPANDED_ARM') == self.expanded_arm, 'Trainer and explicit expanded arm differ')
        super().__init__(plans, configuration, fold, dataset_json, device)
        require(self.online_seed == 42 and self.num_epochs == 250, 'Seed42 and full250 horizon required')
        witness_path = os.environ.get('BASIC_CP_EXPANDED_PREFLIGHT', '')
        require(bool(witness_path), 'Complete actual-bank expanded preflight receipt required')
        witness = json.loads(Path(witness_path).read_text(encoding='utf8'))
        require(witness.get('format') == FORMAT and witness.get('arm') == self.expanded_arm
                and witness.get('complete') is True and witness.get('index_sha256') == sha(self.online_bank_path)
                and witness.get('historical_trainer_sha256') == HISTORICAL_SHA,
                'Expanded preflight must bind this exact full bank/trainer/arm')
        require(witness.get('copies_per_event') == copies_per_event(self.expanded_arm)
                and witness.get('candidate_count') == candidates_per_source(self.expanded_arm)
                and witness.get('blend_border_preprocessed_voxels') == 0
                and witness.get('physical_batch_size') == 2 and witness.get('epochs') == 250,
                'Expanded preflight cannot change the requested one-factor intervention')
        if self.expanded_arm == 'blend50':
            require(witness.get('blend_mode') == 'constant' and witness.get('blend_alpha') == .5
                    and witness.get('label_blending') is False, 'Constant50 image-only hard-label preflight required')
        names = witness.get('entries_sha256', {})
        require(len(names) == 642, 'Every actual source entry must be SHA-bound')
        for name, checksum in names.items():
            require(sha(Path(self.online_bank_path).parent / name) == checksum, 'Source NPZ changed since full preflight')
        self._expanded_pasted_lesions = 0

    def get_dataloaders(self):
        return _get_dataloaders(self)

    def on_train_epoch_start(self):
        self._expanded_pasted_lesions = 0
        return super().on_train_epoch_start()

    def train_step(self, batch):
        counts = np.asarray(batch.pop('online_cp_pasted_lesions'), dtype=np.int64).reshape(-1)
        flags = np.asarray(batch['online_cp_applied'], dtype=np.int64).reshape(-1)
        require(np.array_equal(counts, flags * copies_per_event(self.expanded_arm)), 'Actual per-event paste count differs')
        self._expanded_pasted_lesions += int(counts.sum())
        return super().train_step(batch)

    def on_train_epoch_end(self, train_outputs):
        result = super().on_train_epoch_end(train_outputs)
        self.print_to_log_file(f'[BasicCPAblation] epoch={self.current_epoch} arm={self.expanded_arm} '
            f'pasted_lesions={self._expanded_pasted_lesions} events={self._online_cp_events} '
            f'blend_border_voxels=0 blend_mode={"constant" if self.expanded_arm == "blend50" else "hard"} '
            f'blend_alpha={0.5 if self.expanded_arm == "blend50" else 1.0}', also_print_to_console=True)
        return result


class nnUNetTrainer_250epochs_BasicCPMulti5(_ExpandedBasicCP):
    expanded_arm = 'multi5'


class nnUNetTrainer_250epochs_BasicCPMulti7(_ExpandedBasicCP):
    expanded_arm = 'multi7'


class nnUNetTrainer_250epochs_BasicCPMulti10(_ExpandedBasicCP):
    expanded_arm = 'multi10'


class nnUNetTrainer_250epochs_BasicCPCandidates512(_ExpandedBasicCP):
    expanded_arm = 'candidates512'


class nnUNetTrainer_250epochs_BasicCPCandidates1024(_ExpandedBasicCP):
    expanded_arm = 'candidates1024'


class nnUNetTrainer_250epochs_BasicCPCandidates2048(_ExpandedBasicCP):
    expanded_arm = 'candidates2048'


class nnUNetTrainer_250epochs_BasicCPBlend50(_ExpandedBasicCP):
    expanded_arm = 'blend50'


def _validate_patient(bank, dataset, case_id, network_patch_size):
    started = time.perf_counter()
    names = bank.entry_names(case_id)
    before = {name: sha(bank.root / name) for name in names}
    entries = bank.case_entries(case_id)
    _, seg, _, _ = dataset.load_case(case_id)
    segmentation = np.asarray(seg[:])[0]
    geometry = bank.geometry_for_case(case_id) if copies_per_event(bank.arm) > 1 else None
    valid_positions, multiple_positions = 0, 0
    for source_index, entry in enumerate(entries):
        mask = entry['source_mask'].astype(bool, copy=False)
        require(np.all(np.asarray(mask.shape) <= np.asarray(network_patch_size)), 'Source patch exceeds full network patch')
        for center in entry['candidate_centers']:
            slices = historical._anchored_slices(center, mask.shape, entry['anchor_offset'], segmentation.shape)
            require(slices is not None, f'Out-of-bounds CP candidate in {case_id}')
            roi = segmentation[slices]
            require(not np.any(roi[mask] == 2), f'CP candidate overlaps original tumor in {case_id}')
            require(np.mean(((roi == 1) | (roi == 2))[mask]) >= .85, f'CP candidate violates liver coverage in {case_id}')
            if geometry is not None:
                primary = {'entry': entry, 'center': tuple(map(int, center))}
                extras = geometry.choose(primary, np.zeros(3, dtype=np.int64), np.asarray(segmentation.shape),
                                         copies_per_event(bank.arm), maximum_union_shape=network_patch_size)
                joint_network_crop(segmentation.shape, [primary, *extras], network_patch_size)
                multiple_positions += 1
            valid_positions += 1
        print('[BasicCPAblation preflight source] ' + json.dumps({'case_id': case_id,
              'source_completed': source_index + 1, 'source_total': len(entries), 'arm': bank.arm,
              'validated_positions': valid_positions, 'multiple_paste_positions': multiple_positions,
              'seconds': time.perf_counter() - started}), flush=True)
    require(before == {name: sha(bank.root / name) for name in names}, 'Source files changed during full geometry preflight')
    return {'case_id': case_id, 'source_entries': len(entries), 'validated_positions': valid_positions,
            'multiple_paste_positions': multiple_positions, 'entries_sha256': before,
            'seconds': time.perf_counter() - started}


_PREFLIGHT_WORKER = None


def _initialize_preflight_worker(index_path, dataset, arm, network_patch_size):
    global _PREFLIGHT_WORKER
    require(not historical.torch.cuda.is_initialized(), 'CPU preflight cannot inherit initialized CUDA')
    require(getattr(dataset, '_v24_readonly_binding', None) is not None, 'Authoritative read-only dataset binding required')
    local = copy.copy(dataset)
    local.identifiers = list(dataset.identifiers)
    bank = ExpandedBank(index_path, arm=arm)
    base.validate_dataset_population(local, bank.entries_by_case)
    _PREFLIGHT_WORKER = (bank, local, tuple(network_patch_size))


def _preflight_patient_task(case_id):
    require(_PREFLIGHT_WORKER is not None, 'Actual fork worker initializer missing')
    bank, dataset, patch = _PREFLIGHT_WORKER
    try:
        result = _validate_patient(bank, dataset, case_id, patch)
        result['worker_pid'] = os.getpid()
        return result
    finally:
        bank._cache.clear()
        bank._validated_entries.clear()
        bank._patient_geometry.clear()


def preflight_bank(index_path, dataset, *, arm, network_patch_size=(128, 128, 128), workers=4):
    require(tuple(network_patch_size) == (128, 128, 128), 'Full network patch128 required')
    require(type(workers) is int and workers > 1, 'Measured parallel CPU preflight worker count required')
    bank = ExpandedBank(index_path, arm=arm)
    identifiers = base.validate_dataset_population(dataset, bank.entries_by_case)
    extension = None
    if candidates_per_source(arm) != 128:
        extension = json.loads((bank.root / 'complete.json').read_text(encoding='utf8'))
        require(extension.get('complete') is True and extension.get('index_sha256') == sha(index_path)
                and extension.get('original_index_sha256') == BANK_SHA
                and extension.get('all_original_arrays_and_first128_preserved') is True,
                'Completed original128 preservation proof required')
        require(set(extension.get('entries_sha256', {})) == set(
                name for values in bank.entries_by_case.values() for name in values), 'All642 extension entry SHA receipts required')
        for name, digest in extension['entries_sha256'].items():
            require(sha(bank.root / name) == digest, 'Published extension entry bytes changed')
    require('fork' in multiprocessing.get_all_start_methods() and not historical.torch.cuda.is_initialized(),
            'Full parallel preflight requires Linux fork with uninitialized CUDA')
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork'),
             initializer=_initialize_preflight_worker, initargs=(index_path, dataset, arm, tuple(network_patch_size))) as pool:
        cases = list(pool.map(_preflight_patient_task, sorted(bank.entries_by_case)))
    total = sum(row['validated_positions'] for row in cases)
    multiple = sum(row['multiple_paste_positions'] for row in cases)
    source_checksums = {name: checksum for row in cases for name, checksum in row['entries_sha256'].items()}
    require(len(source_checksums) == 642 and total == 642 * bank.candidate_count, 'Every642 source/candidate must pass')
    require(copies_per_event(arm) == 1 or multiple == total, 'Every primary choice requires exact fullN feasibility')
    return {'format': FORMAT, 'arm': arm, 'complete': True, 'index_sha256': sha(index_path),
            'historical_trainer_sha256': HISTORICAL_SHA, 'historical_bank_sha256': BANK_SHA,
            'entries_sha256': source_checksums, 'source_entries': 642, 'eligible_cases': 81,
            'full_training_cases': 105, 'actual_training_identifiers': list(identifiers), 'full_validation_cases': 26,
            'candidate_count': bank.candidate_count, 'validated_positions': total, 'multiple_paste_positions': multiple,
            'CP_probability': .5, 'copies_per_event': copies_per_event(arm), 'blend_border_preprocessed_voxels': 0,
            'blend_mode': 'constant' if arm == 'blend50' else 'hard', 'blend_alpha': .5 if arm == 'blend50' else 1.,
            'blend_recipient_weight': .5 if arm == 'blend50' else 0., 'label_blending': False,
            'source_selection': 'uniform own-patient sources; replacement allowed for additional copies',
            'extra_position_selection': 'randomized exhaustive feasible-source/candidate search; no caps or lower-count fallback',
            'multiple_crop_policy': 'minimum necessary shift from primary crop to contain all full source patches',
            'standard_spatial_augmentation_unchanged': True,
            'full_requested_patches_guaranteed_before_standard_spatial_augmentation': copies_per_event(arm) > 1,
            'full_masks_after_random_spatial_augmentation_guaranteed': False,
            'original_gate_primary_RNG_schedule_preserved': True, 'GNN_scores_used': False,
            'physical_batch_size': 2, 'seed': 42, 'epochs': 250, 'network_patch_size': [128] * 3,
            'workers': workers, 'case_receipts': cases, 'extension_complete': extension, 'training_executed': False,
            'summary': {'arm': arm, 'sources': 642, 'cases': 81, 'positions_validated': total,
                        'multiple_primary_positions': multiple, 'copies_per_event': copies_per_event(arm)}}
