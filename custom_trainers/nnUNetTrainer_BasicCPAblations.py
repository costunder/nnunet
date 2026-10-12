"""One-factor Basic CP interventions on the actual frozen 642-source trainer.

The historical trainer owns network, loss, optimizer, 250-epoch horizon,
sampling, transforms and validation. This module changes only paste count,
uniform candidate cardinality, or image blending. It never edits that module.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from types import FunctionType, SimpleNamespace

import numpy as np
from scipy import ndimage as ndi
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer_OnlinePairedCP as historical

HISTORICAL_SHA = 'a3df56435fedff46fbc42a502596fdade2632d0211ea412630dba1555f32567d'
BANK_SHA = '9e5be3e3ab8f04fd23c166b3267bc4efbfb4b321ef974920dfc08fbf5dd275e6'
EXTENSION_FORMAT = 'basic_cp_ablation_candidates256_preserved128_v1'
FORMAT = 'basic_cp_one_factor_full250_v1'
ARMS = ('multi3', 'candidates256', 'blend')
OnlineCPError = historical.OnlineCPError


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise OnlineCPError(message)


require(sha(historical.__file__) == HISTORICAL_SHA,
        'Actual historical Basic trainer SHA differs; current raw-target trainer is not interchangeable')


def private_function(function, **bindings):
    namespace = dict(function.__globals__)
    namespace.update(bindings)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def validate_metadata(metadata, arm):
    require(arm in ARMS, 'Explicit Basic CP ablation arm required')
    expected = 256 if arm == 'candidates256' else 128
    require(metadata.get('format') == historical.BANK_FORMAT, 'Historical bank format differs')
    for key, value in {'source_entries': 642, 'eligible_cases': 81, 'candidate_count': expected,
                       'total_candidates': 642 * expected, 'cp_probability': .5,
                       'tumor_label': 2, 'liver_label': 1, 'maximum_diameter_mm': 20.,
                       'minimum_diameter_mm': 0., 'minimum_liver_coverage': .85,
                       'blend_border': 0, 'outer_fold': 0}.items():
        require(metadata.get(key) == value, f'Historical Basic bank contract differs: {key}')
    require(metadata.get('network_patch_size') == [128, 128, 128], 'Full patch128 contract differs')
    require(metadata.get('intensity_scale_range') == [.95, 1.05]
            and metadata.get('intensity_shift_range_hu') == [-5., 5.], 'Historical intensity schedule differs')
    entries = metadata.get('entries_by_case', {})
    require(len(entries) == 81 and sum(map(len, entries.values())) == 642,
            'All81 eligible patients and all642 source lesions required')
    names = [name for values in entries.values() for name in values]
    require(len(set(names)) == 642, 'Source entries are duplicated')
    if arm == 'candidates256':
        ablation = metadata.get('ablation', {})
        for key, value in {'format': EXTENSION_FORMAT, 'uniform_candidate_count': 256,
                           'scores_used': False, 'historical_score_count': 128,
                           'original_candidate_count': 128, 'original_index_sha256': BANK_SHA,
                           'additional_GNN_scores_generated': False,
                           'historical_graph_geometry_required': True}.items():
            require(ablation.get(key) == value, f'256 preservation contract differs: {key}')


def _extended_load_function():
    """Admit 256 centers plus real original128 scores; change only cardinality guard."""
    source = Path(historical.__file__).read_text(encoding='utf8')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OnlineCPBank')
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_load')
    changes = 0
    for node in ast.walk(function):
        if isinstance(node, ast.If) and ast.unparse(node.test) == (
                'centers.shape[0] != scores.shape[0] or centers.shape[0] != self.candidate_count'):
            node.test = ast.parse('scores.shape[0] != 128 or centers.shape[0] != self.candidate_count',
                                  mode='eval').body
            changes += 1
    require(changes == 1, 'Frozen historical load cardinality guard is not unique')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              function], type_ignores=[])
    namespace = dict(historical.__dict__)
    exec(compile(ast.fix_missing_locations(module), historical.__file__, 'exec'), namespace)
    return namespace['_load']


_load256 = _extended_load_function()


class AblationBank(historical.OnlineCPBank):
    def __init__(self, index_path, cache_entries=642, *, arm=None):
        self.arm = str(arm or os.environ.get('BASIC_CP_ABLATION', ''))
        super().__init__(index_path, cache_entries=cache_entries)
        validate_metadata(self.metadata, self.arm)
        if self.arm != 'candidates256':
            require(sha(self.index_path) == BANK_SHA, 'Actual historical bank index SHA differs')
        self._validated_entries = set()
        self._entry_sha256 = None
        witness_path = os.environ.get('BASIC_CP_ABLATION_PREFLIGHT', '')
        if witness_path and Path(witness_path).is_file():
            witness = json.loads(Path(witness_path).read_text(encoding='utf8'))
            require(witness.get('arm') == self.arm and witness.get('index_sha256') == sha(self.index_path)
                    and witness.get('complete') is True, 'Available preflight belongs to another bank/arm')
            self._entry_sha256 = witness.get('entries_sha256')
            require(isinstance(self._entry_sha256, dict) and len(self._entry_sha256) == 642,
                    'All642 actual source-entry SHA witnesses required')

    def _load(self, relative_path):
        if relative_path not in self._cache and self._entry_sha256 is not None:
            require(sha(self.root / relative_path) == self._entry_sha256.get(relative_path),
                    'Source NPZ bytes changed after complete preflight')
        entry = (_load256(self, relative_path) if self.arm == 'candidates256'
                 else super()._load(relative_path))
        if relative_path in self._validated_entries:
            return entry
        centers, scores = entry['candidate_centers'], entry['scores']
        require(centers.dtype.kind in 'iu' and np.unique(centers, axis=0).shape[0] == self.candidate_count,
                'Every candidate must have a distinct integer preprocessed center')
        require(scores.shape == (128,) and np.isfinite(scores).all(), 'Original real128 finite scores required')
        require(np.isfinite(entry['source_data']).all(), 'Source image has nonfinite values')
        require(entry['anchor_offset'].dtype.kind in 'iu', 'Integer source anchor required')
        if self.arm == 'blend':
            entry['_basic_ablation_alpha2'] = feather_alpha(entry['source_mask'])
        self._validated_entries.add(relative_path)
        return entry

    def case_entries(self, case_id):
        return tuple(self._load(name) for name in self.entry_names(case_id))


def mask_overlap(entry_a, center_a, entry_b, center_b):
    """Exact tumor-mask intersection; bounding boxes only eliminate impossible overlap."""
    start_a = np.asarray(center_a, dtype=np.int64) - entry_a['anchor_offset']
    start_b = np.asarray(center_b, dtype=np.int64) - entry_b['anchor_offset']
    lower = np.maximum(start_a, start_b)
    upper = np.minimum(start_a + entry_a['source_mask'].shape, start_b + entry_b['source_mask'].shape)
    if np.any(lower >= upper):
        return False
    sl_a = tuple(slice(int(a), int(b)) for a, b in zip(lower - start_a, upper - start_a))
    sl_b = tuple(slice(int(a), int(b)) for a, b in zip(lower - start_b, upper - start_b))
    return bool(np.any(entry_a['source_mask'][sl_a].astype(bool, copy=False)
                       & entry_b['source_mask'][sl_b].astype(bool, copy=False)))


def _available(entries, lower, upper, selected):
    pools = []
    occupied_geometry = []
    for item in selected:
        occupied_start = np.asarray(item['center']) - item['entry']['anchor_offset']
        occupied_geometry.append((item, np.asarray(item['center']), occupied_start,
                                  occupied_start + np.asarray(item['entry']['source_mask'].shape)))
    for entry in entries:
        centers = entry['candidate_centers']
        geometry = entry.get('_basic_ablation_candidate_bounds')
        if geometry is None:
            starts = centers - entry['anchor_offset']
            geometry = (starts, starts + np.asarray(entry['source_mask'].shape))
            entry['_basic_ablation_candidate_bounds'] = geometry
        starts, stops = geometry
        valid = np.all(starts >= lower, axis=1) & np.all(stops <= upper, axis=1)
        for occupied, occupied_center, occupied_start, occupied_stop in occupied_geometry:
            valid &= np.any(centers != occupied_center, axis=1)
            intersecting = np.flatnonzero(valid & np.all(starts < occupied_stop, axis=1)
                                         & np.all(stops > occupied_start, axis=1))
            for index in intersecting:
                if mask_overlap(entry, centers[index], occupied['entry'], occupied['center']):
                    valid[index] = False
        indices = np.flatnonzero(valid)
        if indices.size:
            pools.append((entry, indices))
    return pools


def select_extra_pastes(entries, primary, lower, upper, rng=None):
    """Find two full nonoverlapping copies; enumerate full pools, never lower paste count.

    Sources are sampled with replacement from this patient's existing lesions.
    Uniform random permutations select among feasible source/candidate pairs.
    A separate RNG leaves the historical gate and future visit schedule unchanged.
    """
    pools = _available(entries, lower, upper, [primary])
    order = np.arange(len(pools)) if rng is None else rng.permutation(len(pools))
    for pool_index in order:
        entry, indices = pools[int(pool_index)]
        candidate_order = indices if rng is None else rng.permutation(indices)
        for candidate_index in candidate_order:
            second = {'entry': entry, 'center': tuple(int(v) for v in entry['candidate_centers'][candidate_index])}
            remaining = _available(entries, lower, upper, [primary, second])
            if remaining:
                third_pool = 0 if rng is None else int(rng.integers(len(remaining)))
                third_entry, third_indices = remaining[third_pool]
                third_index = third_indices[0 if rng is None else int(rng.integers(third_indices.size))]
                third = {'entry': third_entry,
                         'center': tuple(int(v) for v in third_entry['candidate_centers'][third_index])}
                return [second, third]
    raise OnlineCPError('Full three-paste geometry unavailable for this primary source/location; no 1/2-paste fallback')


def canonical_crop(shape, primary, patch=(128, 128, 128)):
    loader = SimpleNamespace(patch_size=np.asarray(patch), need_to_pad=np.zeros(3, dtype=np.int64))
    lower, upper = historical._bbox_around_paste(loader, shape, primary['center'],
                 primary['entry']['source_mask'].shape, primary['entry']['anchor_offset'])
    return np.asarray(lower), np.asarray(upper)


def feather_alpha(mask, border=2):
    # Exact existing hiercp.common.feather_alpha formula; distance is in voxels.
    require(border == 2, 'This one-factor blend uses the declared two-voxel border')
    distance_inside = ndi.distance_transform_edt(mask.astype(bool))
    return np.clip(distance_inside / float(border), 0, 1).astype(np.float32)


def _remember_case_shape(loader, shape, center, source_shape, anchor):
    loader._ablation_case_shape = tuple(int(v) for v in shape)
    return historical._bbox_around_paste(loader, shape, center, source_shape, anchor)


_generate = private_function(historical.nnUNetDataLoaderOnlineCP.generate_train_batch,
                             _bbox_around_paste=_remember_case_shape)


class AblationDataLoader(historical.nnUNetDataLoaderOnlineCP):
    def __init__(self, *args, bank_path, policy, online_seed, **kwargs):
        require(policy == 'basic' and online_seed == 42, 'Basic uniform policy and seed42 required')
        super().__init__(*args, bank_path=bank_path, policy=policy, online_seed=online_seed, **kwargs)
        self.online_bank = AblationBank(bank_path)
        self.arm = self.online_bank.arm

    def _select_candidate(self, scores, u):
        count = self.online_bank.candidate_count
        return min(count - 1, int(np.floor(float(u) * count)))

    def _sample_paste_plan(self, case_id):
        plan, token = super()._sample_paste_plan(case_id)
        if plan is not None:
            plan['ablation_schedule_token'] = int(token)
        return plan, token

    def _apply_paste_to_crop(self, data_cropped, seg_cropped, bbox_lbs, plan, case_id):
        if self.arm == 'candidates256':
            return super()._apply_paste_to_crop(data_cropped, seg_cropped, bbox_lbs, plan, case_id)
        if self.arm == 'blend':
            entry = plan['entry']
            mask = entry['source_mask'].astype(bool, copy=False)
            slices = historical._anchored_slices(np.asarray(plan['center']) - np.asarray(bbox_lbs),
                       mask.shape, entry['anchor_offset'], data_cropped.shape[1:])
            require(slices is not None and entry['source_data'].shape[0] == data_cropped.shape[0],
                    'Full blend source must fit original crop/channels')
            alpha = entry.get('_basic_ablation_alpha2')
            if alpha is None:
                # In-memory derivation only; the original NPZ is never modified.
                alpha = feather_alpha(mask)
                entry['_basic_ablation_alpha2'] = alpha
            source = entry['source_data'].astype(np.float32, copy=False)
            transformed = source * float(plan['scale']) + float(plan['normalized_offset'])
            roi = data_cropped[(slice(None), *slices)]
            blended = (1. - alpha[None]) * roi + alpha[None] * transformed
            # Write mask voxels only, preserving all surrounding image bytes.
            roi[:, mask] = blended[:, mask]
            seg_cropped[(0, *slices)][mask] = self.online_bank.tumor_label
            return
        require(self.arm == 'multi3', 'Unknown ablation paste arm')
        lower, upper = canonical_crop(self._ablation_case_shape, plan)
        crop_lower = np.asarray(bbox_lbs)
        require(np.all(lower >= crop_lower)
                and np.all(upper <= crop_lower + np.asarray(data_cropped.shape[1:])),
                'Canonical network128 crop is not contained in the historical initial crop')
        rng = np.random.default_rng(historical._stable_seed(FORMAT, 'extra2', self.online_seed,
                                    self.online_epoch, case_id, plan['ablation_schedule_token']))
        extras = select_extra_pastes(self.online_bank.case_entries(case_id), plan, lower, upper, rng)
        plans = [plan]
        for extra in extras:
            scale_low, scale_high = self.online_bank.intensity_scale
            shift_low, shift_high = self.online_bank.intensity_shift_hu
            scale = scale_low + float(rng.random()) * (scale_high - scale_low)
            shift_hu = shift_low + float(rng.random()) * (shift_high - shift_low)
            extra.update(scale=scale, normalized_offset=((scale - 1.) * self.online_bank.ct_mean
                                                         + shift_hu) / self.online_bank.ct_std)
            plans.append(extra)
        # Validate all three before any array write: failure never leaves a partial augmentation.
        for item in plans:
            entry = item['entry']
            slices = historical._anchored_slices(np.asarray(item['center']) - crop_lower,
                       entry['source_mask'].shape, entry['anchor_offset'], data_cropped.shape[1:])
            require(slices is not None and entry['source_data'].shape[0] == data_cropped.shape[0],
                    'All three complete sources must fit the unmodified crop/channels')
            require(not np.any(seg_cropped[(0, *slices)][entry['source_mask'].astype(bool)]
                               == self.online_bank.tumor_label), 'A new paste overlaps existing tumor')
        for item in plans:
            super()._apply_paste_to_crop(data_cropped, seg_cropped, bbox_lbs, item, case_id)

    def generate_train_batch(self):
        batch = _generate(self)
        batch['online_cp_pasted_lesions'] = np.asarray(batch['online_cp_applied'], dtype=np.uint8) * (
            3 if self.arm == 'multi3' else 1)
        return batch


_get_dataloaders = private_function(historical._nnUNetTrainer_250epochs_OnlineCP.get_dataloaders,
                                  nnUNetDataLoaderOnlineCP=AblationDataLoader, OnlineCPBank=AblationBank)


class _BasicCPAblation(historical.nnUNetTrainer_250epochs_OnlineBasicCP):
    ablation_arm = None

    def __init__(self, *args, **kwargs):
        require(os.environ.get('BASIC_CP_ABLATION') == self.ablation_arm, 'Trainer and explicit arm differ')
        super().__init__(*args, **kwargs)
        require(self.online_seed == 42 and self.num_epochs == 250, 'Seed42 and full250 horizon required')
        witness_path = os.environ.get('BASIC_CP_ABLATION_PREFLIGHT', '')
        require(bool(witness_path), 'Complete actual-bank preflight receipt required')
        witness = json.loads(Path(witness_path).read_text(encoding='utf8'))
        require(witness.get('format') == FORMAT and witness.get('arm') == self.ablation_arm
                and witness.get('complete') is True and witness.get('index_sha256') == sha(self.online_bank_path)
                and witness.get('historical_trainer_sha256') == HISTORICAL_SHA,
                'Preflight does not bind this exact full bank/trainer/arm')
        source_files = witness.get('entries_sha256', {})
        require(len(source_files) == 642, 'Every actual source entry must be SHA-bound by preflight')
        for name, checksum in source_files.items():
            require(sha(Path(self.online_bank_path).parent / name) == checksum,
                    'Source NPZ changed since this exact full preflight')
        self._ablation_pasted_lesions = 0

    def get_dataloaders(self):
        return _get_dataloaders(self)

    def on_train_epoch_start(self):
        self._ablation_pasted_lesions = 0
        return super().on_train_epoch_start()

    def train_step(self, batch):
        counts = np.asarray(batch.pop('online_cp_pasted_lesions'), dtype=np.int64).reshape(-1)
        flags = np.asarray(batch['online_cp_applied'], dtype=np.int64).reshape(-1)
        expected = 3 if self.ablation_arm == 'multi3' else 1
        require(np.array_equal(counts, flags * expected), 'Actual per-event paste count differs')
        self._ablation_pasted_lesions += int(counts.sum())
        return super().train_step(batch)

    def on_train_epoch_end(self, train_outputs):
        result = super().on_train_epoch_end(train_outputs)
        self.print_to_log_file(f'[BasicCPAblation] epoch={self.current_epoch} arm={self.ablation_arm} '
                              f'pasted_lesions={self._ablation_pasted_lesions} '
                              f'events={self._online_cp_events} blend_border_voxels={2 if self.ablation_arm == "blend" else 0}',
                              also_print_to_console=True)
        return result


class nnUNetTrainer_250epochs_BasicCPMulti3(_BasicCPAblation):
    ablation_arm = 'multi3'


class nnUNetTrainer_250epochs_BasicCPCandidates256(_BasicCPAblation):
    ablation_arm = 'candidates256'


class nnUNetTrainer_250epochs_BasicCPBlend(_BasicCPAblation):
    ablation_arm = 'blend'


def _validate_patient(bank, dataset, case_id, network_patch_size):
    started = time.perf_counter()
    names = bank.entry_names(case_id)
    before = {name: sha(bank.root / name) for name in names}
    entries = bank.case_entries(case_id)
    _, seg, _, _ = dataset.load_case(case_id)
    segmentation = np.asarray(seg[:])[0]
    valid_positions, triple_positions = 0, 0
    for source_index, entry in enumerate(entries):
        mask = entry['source_mask'].astype(bool, copy=False)
        require(np.all(np.asarray(mask.shape) <= np.asarray(network_patch_size)),
                'Source patch exceeds the full network patch')
        for center in entry['candidate_centers']:
            slices = historical._anchored_slices(center, mask.shape, entry['anchor_offset'], segmentation.shape)
            require(slices is not None, f'Out-of-bounds CP candidate in {case_id}')
            roi = segmentation[slices]
            require(not np.any(roi[mask] == 2), f'CP candidate overlaps original tumor in {case_id}')
            require(np.mean(((roi == 1) | (roi == 2))[mask]) >= .85,
                    f'CP candidate violates liver coverage in {case_id}')
            primary = {'entry': entry, 'center': tuple(int(v) for v in center)}
            lower, upper = canonical_crop(segmentation.shape, primary, network_patch_size)
            if bank.arm == 'multi3':
                select_extra_pastes(entries, primary, lower, upper)
                triple_positions += 1
            valid_positions += 1
        print('[BasicCPAblation preflight source] ' + json.dumps({'case_id': case_id,
              'source_completed': source_index + 1, 'source_total': len(entries),
              'validated_positions': valid_positions, 'three_paste_positions': triple_positions,
              'seconds': time.perf_counter() - started}), flush=True)
    require(before == {name: sha(bank.root / name) for name in names}, 'Source files changed during geometry preflight')
    return {'case_id': case_id, 'source_entries': len(entries), 'validated_positions': valid_positions,
            'three_paste_positions': triple_positions, 'entries_sha256': before}


def preflight_bank(index_path, dataset, *, arm, network_patch_size=(128, 128, 128)):
    require(tuple(network_patch_size) == (128, 128, 128), 'Full network patch128 required')
    bank = AblationBank(index_path, arm=arm)
    require(len(dataset.keys()) == 105 and set(bank.entries_by_case).issubset(dataset.keys()),
            'Full105 inner-training dataset and all81 own-source cases required')
    extension = None
    if arm == 'candidates256':
        extension = json.loads((bank.root / 'complete.json').read_text(encoding='utf8'))
        require(extension.get('complete') is True and extension.get('index_sha256') == sha(index_path)
                and extension.get('original_index_sha256') == BANK_SHA
                and extension.get('all_original_arrays_and_first128_preserved') is True,
                'Completed exact original128 preservation proof required')
        require(set(extension.get('entries_sha256', {})) == set(
                name for values in bank.entries_by_case.values() for name in values), 'All642 extension entry SHA receipts required')
        for name, digest in extension['entries_sha256'].items():
            require(sha(bank.root / name) == digest, 'Published extension entry bytes changed')
    # Each task owns its bank/cache; no shared mutable LRU across threads. Full
    # independent patients run concurrently, using only shared read-only segmentation.
    workers = 4
    def patient(case_id):
        local = AblationBank(index_path, arm=arm)
        result = _validate_patient(local, dataset, case_id, network_patch_size)
        print('[BasicCPAblation preflight] ' + json.dumps(result), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        cases = list(pool.map(patient, sorted(bank.entries_by_case)))
    total = sum(row['validated_positions'] for row in cases)
    triples = sum(row['three_paste_positions'] for row in cases)
    source_checksums = {name: checksum for row in cases for name, checksum in row['entries_sha256'].items()}
    require(len(source_checksums) == 642, 'All642 source identities must be independently bound')
    require(total == 642 * bank.candidate_count, 'Every source/candidate must pass geometry preflight')
    require(arm != 'multi3' or triples == 642 * 128, 'All82176 primary choices require full triple feasibility')
    return {'format': FORMAT, 'arm': arm, 'complete': True, 'index_sha256': sha(index_path),
            'historical_trainer_sha256': HISTORICAL_SHA, 'historical_bank_sha256': BANK_SHA,
            'entries_sha256': source_checksums,
            'source_entries': 642, 'eligible_cases': 81, 'full_training_cases': 105,
            'full_validation_cases': 26, 'candidate_count': bank.candidate_count,
            'validated_positions': total, 'three_paste_positions': triples, 'CP_probability': .5,
            'copies_per_event': 3 if arm == 'multi3' else 1,
            'blend_border_preprocessed_voxels': 2 if arm == 'blend' else 0,
            'source_selection': 'uniform own-patient sources; replacement allowed for additional copies',
            'extra_position_selection': 'uniform feasible pools within original central128 crop, masks nonoverlapping',
            'original_gate_primary_RNG_schedule_preserved': True, 'GNN_scores_used': False,
            'physical_batch_size': 2, 'seed': 42, 'epochs': 250, 'network_patch_size': [128, 128, 128],
            'workers': workers, 'case_receipts': cases, 'extension_complete': extension,
            'training_executed': False, 'summary': {'arm': arm, 'sources': 642, 'cases': 81,
                'positions_validated': total, 'triple_primary_positions': triples,
                'copies_per_event': 3 if arm == 'multi3' else 1}}
