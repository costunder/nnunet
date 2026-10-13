"""User-approved same-source128 rejection ONLY for impossible first positions.

The fullN intervention remains5/7/10. Original gate/source/position/intensity
draws are retained. An infeasible original first position alone is redrawn
uniformly among that SAME source's proven joint-feasible original128 centers.
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
from . import nnUNetTrainer_BasicCPExpanded as expanded

FORMAT = expanded.FORMAT
JOINT_POLICY_FORMAT = 'basic_cp_multin_same_source128_primary_rejection_v1'
ARMS = ('multi5', 'multi7', 'multi10')
require, sha, historical = expanded.require, expanded.sha, expanded.historical


def witness_path():
    ordinary = os.environ.get('BASIC_CP_EXPANDED_PREFLIGHT', '')
    optional = os.environ.get('BASIC_CP_MULTIN_JOINT_PREFLIGHT', ordinary)
    require(optional == ordinary, 'Joint and expanded preflight environment must bind the same fresh receipt')
    return ordinary


def validate_feasible_inventory(witness, bank):
    require(bank.arm in ARMS and witness.get('format') == FORMAT
            and witness.get('joint_policy_format') == JOINT_POLICY_FORMAT
            and witness.get('arm') == bank.arm and witness.get('complete') is True
            and witness.get('index_sha256') == sha(bank.index_path), 'Exact same-source128 joint preflight required')
    names = {name for group in bank.entries_by_case.values() for name in group}
    inventory = witness.get('feasible_primary_indices')
    require(isinstance(inventory, dict) and set(inventory) == names and len(names) == 642,
            'Every original642 source must have its exact feasible-primary inventory')
    for name, indices in inventory.items():
        require(isinstance(indices, list) and bool(indices) and len(indices) == len(set(indices))
                and indices == sorted(indices) and all(type(i) is int and 0 <= i < 128 for i in indices),
                f'At least one distinct original128 joint-feasible primary required for {name}')
    feasible = sum(map(len, inventory.values()))
    require(witness.get('validated_positions') == 642 * 128
            and witness.get('original_primary_positions_feasible') == feasible
            and witness.get('rejection_only_primary_positions') == 642 * 128 - feasible
            and witness.get('all_original128_CP_positions_validated') is True
            and witness.get('all642_sources_have_joint_feasible_primary') is True
            and witness.get('original_primary_selection_preserved') is False
            and witness.get('original_gate_and_primary_draw_RNG_schedule_preserved') is True,
            'Joint policy counters or unchanged original draw schedule differ')
    return inventory


class JointBank(expanded.ExpandedBank):
    def __init__(self, index_path, cache_entries=642, *, arm=None):
        super().__init__(index_path, cache_entries=cache_entries, arm=arm)
        require(self.arm in ARMS, 'Joint rejection applies only to multi5/7/10')
        self.feasible_primary_indices = None
        path = witness_path()
        if path and Path(path).is_file():
            witness = json.loads(Path(path).read_text(encoding='utf8'))
            self.feasible_primary_indices = validate_feasible_inventory(witness, self)

    def _load(self, relative_path):
        entry = super()._load(relative_path)
        entry['_joint_source_relative_path'] = str(relative_path)
        return entry


_initialize_loader = expanded.base.private_function(historical.nnUNetDataLoaderOnlineCP.__init__, OnlineCPBank=JointBank)


class JointDataLoader(expanded.ExpandedDataLoader):
    def __init__(self, *args, bank_path, policy, online_seed, **kwargs):
        require(policy == 'basic' and online_seed == 42, 'Basic uniform policy and seed42 required')
        _initialize_loader(self, *args, bank_path=bank_path, policy=policy, online_seed=online_seed, **kwargs)
        self.arm = self.online_bank.arm
        require(self.online_bank.feasible_primary_indices is not None, 'Actual full642 joint inventory must precede training')

    def _sample_paste_plan(self, case_id):
        plan, token = super()._sample_paste_plan(case_id)
        if plan is None:
            if hasattr(self, '_joint_batch_resampled'):
                self._joint_batch_resampled.append(False)
            return plan, token
        entry = plan['entry']; name = entry['_joint_source_relative_path']
        require(name in self.online_bank.entry_names(str(case_id)), 'Original source must remain this same patient')
        indices = np.flatnonzero(np.all(entry['candidate_centers'] == plan['center'], axis=1))
        require(indices.size == 1, 'Original primary must map to one real unchanged source candidate')
        original_index = int(indices[0])
        inventory = self.online_bank.feasible_primary_indices
        require(inventory is not None and name in inventory, 'Full actual source-specific joint inventory required')
        feasible = inventory[name]
        chosen = original_index
        if original_index not in feasible:
            rng = np.random.default_rng(historical._stable_seed(JOINT_POLICY_FORMAT, self.arm,
                self.online_seed, self.online_epoch, str(case_id), int(token), 'rejection-primary'))
            chosen = int(feasible[int(rng.integers(len(feasible)))])
            plan['center'] = tuple(map(int, entry['candidate_centers'][chosen]))
        plan.update(joint_primary_resampled=chosen != original_index,
                    joint_original_primary_index=original_index, joint_selected_primary_index=chosen,
                    joint_primary_rejection_policy=JOINT_POLICY_FORMAT)
        if hasattr(self, '_joint_batch_resampled'):
            self._joint_batch_resampled.append(plan['joint_primary_resampled'])
        # Expanded already retained this dict as the pending primary, so the
        # crop callback and paste execution see the SAME approved final center.
        return plan, token

    def generate_train_batch(self):
        # Capture per-event decisions during the exact historical batch path.
        self._joint_batch_resampled = []
        batch = super().generate_train_batch()
        flags = np.asarray(batch['online_cp_applied'], dtype=np.uint8)
        decisions = np.asarray(self._joint_batch_resampled, dtype=np.uint8)
        require(decisions.shape == flags.shape and np.all(decisions <= flags), 'Joint resampling telemetry must match actual CP events')
        batch['online_cp_primary_resampled'] = decisions
        return batch

    def _apply_paste_to_crop(self, data_cropped, seg_cropped, bbox_lbs, plan, case_id):
        return super()._apply_paste_to_crop(data_cropped, seg_cropped, bbox_lbs, plan, case_id)


_get_dataloaders = expanded.base.private_function(historical._nnUNetTrainer_250epochs_OnlineCP.get_dataloaders,
                                                 nnUNetDataLoaderOnlineCP=JointDataLoader, OnlineCPBank=JointBank)


class _JointBasicCP(expanded._ExpandedBasicCP):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: historical.torch.device = historical.torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        receipt = json.loads(Path(witness_path()).read_text(encoding='utf8'))
        bank = JointBank(self.online_bank_path, arm=self.expanded_arm)
        validate_feasible_inventory(receipt, bank)
        self._joint_primary_resampled = 0

    def get_dataloaders(self):
        return _get_dataloaders(self)

    def on_train_epoch_start(self):
        self._joint_primary_resampled = 0
        return super().on_train_epoch_start()

    def train_step(self, batch):
        decisions = np.asarray(batch.pop('online_cp_primary_resampled'), dtype=np.int64).reshape(-1)
        flags = np.asarray(batch['online_cp_applied'], dtype=np.int64).reshape(-1)
        require(decisions.shape == flags.shape and np.all((decisions == 0) | (decisions == 1))
                and np.all(decisions <= flags), 'Actual joint primary redraw counts differ from CP events')
        self._joint_primary_resampled += int(decisions.sum())
        return super().train_step(batch)

    def on_train_epoch_end(self, train_outputs):
        result = super().on_train_epoch_end(train_outputs)
        self.print_to_log_file(f'[BasicCPJointPrimary] epoch={self.current_epoch} arm={self.expanded_arm} '
            f'resampled_events={self._joint_primary_resampled} CP_events={self._online_cp_events} '
            f'policy={JOINT_POLICY_FORMAT} source_substitution=0 added_positions=0 reduced_paste_events=0',
            also_print_to_console=True)
        return result


class nnUNetTrainer_250epochs_BasicCPMulti5Joint(_JointBasicCP):
    expanded_arm = 'multi5'


class nnUNetTrainer_250epochs_BasicCPMulti7Joint(_JointBasicCP):
    expanded_arm = 'multi7'


class nnUNetTrainer_250epochs_BasicCPMulti10Joint(_JointBasicCP):
    expanded_arm = 'multi10'


def _validate_patient(bank, dataset, case_id, network_patch_size):
    started = time.perf_counter(); names = bank.entry_names(case_id)
    before = {name: sha(bank.root / name) for name in names}; entries = bank.case_entries(case_id)
    _, seg, _, _ = dataset.load_case(case_id); segmentation = np.asarray(seg[:])[0]
    geometry = bank.geometry_for_case(case_id); inventory = {}; valid_positions = 0
    for source_index, (name, entry) in enumerate(zip(names, entries)):
        mask = entry['source_mask'].astype(bool, copy=False); feasible_indices = []
        require(np.all(np.asarray(mask.shape) <= np.asarray(network_patch_size)), 'Source patch exceeds full network128')
        for candidate_index, center in enumerate(entry['candidate_centers']):
            slices = historical._anchored_slices(center, mask.shape, entry['anchor_offset'], segmentation.shape)
            require(slices is not None, f'Original128 CP candidate out of bounds: {case_id}/{name}/{candidate_index}')
            roi = segmentation[slices]
            require(not np.any(roi[mask] == 2), f'Original128 CP candidate overlaps tumor: {case_id}/{name}/{candidate_index}')
            require(np.mean(((roi == 1) | (roi == 2))[mask]) >= .85,
                    f'Original128 CP candidate violates liver coverage: {case_id}/{name}/{candidate_index}')
            primary = dict(entry=entry, center=tuple(map(int, center)))
            # The only caught failure is the explicit complete-search no-fullN
            # condition; malformed tensors and all other errors still fail.
            try:
                extras = geometry.choose(primary, np.zeros(3, dtype=np.int64), np.asarray(segmentation.shape),
                                         expanded.copies_per_event(bank.arm), maximum_union_shape=network_patch_size)
                expanded.joint_network_crop(segmentation.shape, [primary, *extras], network_patch_size)
            except expanded.OnlineCPError as error:
                expected = f'Full {expanded.copies_per_event(bank.arm)}-paste geometry unavailable for unchanged primary; no partial paste fallback'
                require(str(error) == expected, f'Unexpected joint geometry failure: {case_id}/{name}/{candidate_index}: {error}')
            else:
                feasible_indices.append(candidate_index)
            valid_positions += 1
        require(bool(feasible_indices), f'No full{expanded.copies_per_event(bank.arm)} joint primary exists in this source original128: {case_id}/{name}')
        inventory[name] = feasible_indices
        print('[BasicCPAblation joint preflight source] ' + json.dumps(dict(case_id=case_id, arm=bank.arm,
            source_completed=source_index + 1, source_total=len(entries), source_name=name,
            joint_feasible_primary_count=len(feasible_indices), rejected_primary_count=128 - len(feasible_indices),
            validated_positions=valid_positions, seconds=time.perf_counter() - started)), flush=True)
    require(before == {name: sha(bank.root / name) for name in names}, 'Original source bytes changed during joint preflight')
    return dict(case_id=case_id, source_entries=len(entries), validated_positions=valid_positions,
                multiple_paste_positions=sum(map(len, inventory.values())), feasible_primary_indices=inventory,
                entries_sha256=before, seconds=time.perf_counter() - started)


_WORKER = None


def _initialize_worker(index_path, dataset, arm, patch):
    global _WORKER
    require(not historical.torch.cuda.is_initialized(), 'Joint CPU preflight cannot inherit initialized CUDA')
    require(getattr(dataset, '_v24_readonly_binding', None) is not None, 'Authoritative read-only dataset binding required')
    local = copy.copy(dataset); local.identifiers = list(dataset.identifiers)
    bank = JointBank(index_path, arm=arm)
    expanded.base.validate_dataset_population(local, bank.entries_by_case)
    _WORKER = (bank, local, tuple(patch))


def _patient_task(case_id):
    require(_WORKER is not None, 'Joint CPU preflight initializer missing')
    bank, dataset, patch = _WORKER
    try:
        result = _validate_patient(bank, dataset, case_id, patch); result['worker_pid'] = os.getpid()
        return result
    finally:
        bank._cache.clear(); bank._validated_entries.clear(); bank._patient_geometry.clear()


def preflight_bank(index_path, dataset, *, arm, network_patch_size=(128, 128, 128), workers=4):
    require(arm in ARMS and tuple(network_patch_size) == (128, 128, 128), 'Full multi5/7/10 original128 preflight required')
    require(type(workers) is int and workers > 1, 'Measured parallel CPU preflight worker count required')
    bank = JointBank(index_path, arm=arm); identifiers = expanded.base.validate_dataset_population(dataset, bank.entries_by_case)
    require('fork' in multiprocessing.get_all_start_methods() and not historical.torch.cuda.is_initialized(),
            'Full parallel joint preflight requires Linux fork without initialized CUDA')
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork'),
                             initializer=_initialize_worker, initargs=(index_path, dataset, arm, tuple(network_patch_size))) as pool:
        cases = list(pool.map(_patient_task, sorted(bank.entries_by_case)))
    inventory = {name: indices for row in cases for name, indices in row['feasible_primary_indices'].items()}
    source_checksums = {name: checksum for row in cases for name, checksum in row['entries_sha256'].items()}
    total = sum(row['validated_positions'] for row in cases); feasible = sum(map(len, inventory.values()))
    require(total == 642 * 128 and len(source_checksums) == len(inventory) == 642,
            'Every original642 source and all128 original CP positions must be validated')
    result = dict(format=FORMAT, joint_policy_format=JOINT_POLICY_FORMAT, arm=arm, complete=True,
        index_sha256=sha(index_path), historical_trainer_sha256=expanded.HISTORICAL_SHA,
        historical_bank_sha256=expanded.BANK_SHA, entries_sha256=source_checksums,
        source_entries=642, eligible_cases=81, full_training_cases=105, full_validation_cases=26,
        actual_training_identifiers=list(identifiers), candidate_count=128, validated_positions=total,
        multiple_paste_positions=feasible, feasible_primary_indices=inventory,
        original_primary_positions_feasible=feasible, rejection_only_primary_positions=total - feasible,
        all_original128_CP_positions_validated=True, all642_sources_have_joint_feasible_primary=True,
        original_primary_selection_preserved=False, original_gate_and_primary_draw_RNG_schedule_preserved=True,
        original_gate_primary_RNG_schedule_preserved=False,
        same_source128_primary_rejection_policy='Keep feasible original primary; redraw only infeasible first position uniformly from same source original128 joint-feasible indices with independent token RNG',
        original_source_selection_preserved=True, intensity_jitter_schedule_preserved=True,
        source_substitution=False, additional_candidate_positions=False, reduced_paste_count=False, skipped_CP_events=False,
        CP_probability=.5, copies_per_event=expanded.copies_per_event(arm), blend_border_preprocessed_voxels=0,
        blend_mode='hard', blend_alpha=1., blend_recipient_weight=0., label_blending=False,
        source_selection='Original uniform same-patient source; unchanged source identity; replacement for additional copies',
        extra_position_selection='Randomized exhaustive same-patient feasible-source/candidate search; no caps',
        multiple_crop_policy='Minimum necessary primary crop shift to contain all full requested source patches',
        standard_spatial_augmentation_unchanged=True, full_requested_patches_guaranteed_before_standard_spatial_augmentation=True,
        full_masks_after_random_spatial_augmentation_guaranteed=False, GNN_scores_used=False,
        physical_batch_size=2, seed=42, epochs=250, network_patch_size=[128] * 3, workers=workers,
        case_receipts=cases, extension_complete=None, training_executed=False,
        summary=dict(arm=arm, sources=642, cases=81, positions_validated=total, multiple_primary_positions=feasible,
                     rejected_original_primary_positions=total - feasible, copies_per_event=expanded.copies_per_event(arm)))
    validate_feasible_inventory(result, bank)
    return result
