"""CPU UNIT fixtures for actual historical trainer binding and paste math.

The tiny synthetic masks below test geometry/RNG only. They are not clinical
data, full-bank preflight, native training, or evidence of segmentation quality.
"""
from __future__ import annotations
import ast
import copy
from collections import OrderedDict
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy import ndimage as ndi

ROOT = Path(__file__).absolute().parents[1]
HISTORICAL_NAME = 'nnunetv2.training.nnUNetTrainer.nnUNetTrainer_OnlinePairedCP'


class CPUUnitDataLoader:
    """Only original geometry fixtures are exercised; no train batch backend."""
    def set_thread_id(self, thread_id):
        self.thread_id = thread_id


class CPUUnitTrainer:
    pass


def source_text():
    explicit = os.environ.get('BASIC_CP_HISTORICAL_TEST_SOURCE')
    if explicit:
        return Path(explicit).read_text(encoding='utf8')
    receipt = ROOT.parents[1] / 'outputs/v24_gpu6_recovery_20261011/HISTORICAL_BASIC_CP_CONTRACT_READONLY_1791835410056009100.json'
    return json.loads(receipt.read_text(encoding='utf8'))['trainer_source']


def fixture_historical(path):
    source = source_text()
    Path(path).write_bytes(source.encode('utf8'))
    module = ModuleType(HISTORICAL_NAME)
    module.__file__ = str(path)
    module.__dict__.update(np=np, torch=torch, json=json, os=os, hashlib=hashlib, random=random,
        Path=Path, OrderedDict=OrderedDict, Any=Any, Mapping=Mapping, Sequence=Sequence,
        nnUNetDataLoader=CPUUnitDataLoader, nnUNetTrainer=CPUUnitTrainer)
    tree = ast.parse(source)
    kept = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == '__future__':
            kept.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            kept.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            kept.append(node)
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module


def fixture_actual_dataset(identifiers):
    """Compile the constructor read from the actual installed server class."""
    receipt_folder = ROOT.parents[1] / 'outputs/v24_gpu6_recovery_20261011'
    receipts = sorted(receipt_folder.glob('inspect_basic_actual_dataset_API_READONLY_20261013_*.log'))
    if not receipts:
        raise FileNotFoundError('Actual readonly dataset-interface receipt required for this CPU regression')
    actual = json.loads(receipts[-1].read_text(encoding='utf8'))
    if actual['sha256'] != '755175caad47cf1e719e7f0dd0771304eaad1f7f4497f993ed56a7b8f95a08f2':
        raise ValueError('Actual installed dataset class source differs')
    tree = ast.parse(actual['classes']['nnUNetBaseDataset'])
    original = next(n for n in tree.body[0].body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    constructor = copy.deepcopy(original)
    fixture = ast.ClassDef(name='ActualDatasetConstructorCPUFixture', bases=[], keywords=[],
        body=[constructor], decorator_list=[])
    namespace = {'List': list}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fixture], type_ignores=[])),
                 '<actual-server-dataset-constructor-CPU-unit>', 'exec'), namespace)
    return namespace['ActualDatasetConstructorCPUFixture']('CPU_UNIT_NO_DATA_READ', list(identifiers))


def actual_base_constructor_introspection():
    """Actual base constructor prefix, through its checkpoint-argument capture."""
    folder = ROOT.parents[1] / 'outputs/v24_gpu6_recovery_20261011'
    receipts = sorted(folder.glob('inspect_basic_actual_constructor_READONLY_20261013_*.log'))
    if not receipts:
        raise FileNotFoundError('Actual base constructor readonly receipt required')
    actual = json.loads(receipts[-1].read_text(encoding='utf8'))
    if actual['files']['training/nnUNetTrainer/nnUNetTrainer.py']['sha256'] != (
            '7096efb2040135eb60df3c8bf39cdbcc373e299671fcee365b3aff2f4da4dfbd'):
        raise ValueError('Actual installed base constructor source differs')
    function = ast.parse(actual['classes']['nnUNetTrainer']['constructor']).body[0]
    retained = []
    for statement in function.body:
        retained.append(statement)
        if isinstance(statement, ast.For) and 'inspect.signature(self.__init__)' in ast.unparse(statement):
            break
    else:
        raise ValueError('Actual constructor signature-introspection loop missing')
    function.body = retained
    namespace = {'torch': torch, 'dist': torch.distributed, 'inspect': inspect}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 '<actual-base-constructor-CPU-signature-regression>', 'exec'), namespace)
    return namespace['__init__']


def entry(centers, *, side=3, value=9.):
    return {'source_data': np.full((1, side, side, side), value, dtype=np.float32),
            'source_mask': np.ones((side, side, side), dtype=np.uint8),
            'anchor_offset': np.full(3, side // 2, dtype=np.int64),
            'candidate_centers': np.asarray(centers, dtype=np.int64),
            'scores': np.linspace(100, -100, 128, dtype=np.float32)}


class BasicCPAblationCPUUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / 'tmp').mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tmp')
        cls.historical = fixture_historical(Path(cls.temp.name) / 'historical_actual.py')
        packages = {}
        for name in ['nnunetv2', 'nnunetv2.training', 'nnunetv2.training.nnUNetTrainer']:
            package = ModuleType(name); package.__path__ = []
            packages[name] = package
        packages[HISTORICAL_NAME] = cls.historical
        cls.patch = patch.dict('sys.modules', packages); cls.patch.start()
        spec = importlib.util.spec_from_file_location('basic_cp_ablation_CPU_UNIT',
                   ROOT / 'custom_trainers/nnUNetTrainer_BasicCPAblations.py')
        cls.module = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop(); cls.temp.cleanup()

    def loader(self, arm, entries, count=128):
        loader = self.module.AblationDataLoader.__new__(self.module.AblationDataLoader)
        loader.arm = arm; loader.online_seed = 42; loader.online_epoch = 7
        loader.online_policy = 'basic'; loader._cp_rng = None; loader.thread_id = 0
        loader.online_bank = SimpleNamespace(candidate_count=count, cp_probability=.5,
            intensity_scale=(.95, 1.05), intensity_shift_hu=(-5., 5.), ct_mean=99.48007202148438,
            ct_std=37.13840103149414, tumor_label=2,
            entry_names=lambda case: tuple(range(len(entries))) if case == 'OWN_PATIENT' else (),
            load_for_case=lambda case, index: entries[index], case_entries=lambda case: entries)
        return loader

    def test_actual_historical_source_hash_and_private_globals_preserved(self):
        m = self.module
        self.assertEqual(m.sha(self.historical.__file__), m.HISTORICAL_SHA)
        self.assertEqual(m._generate.__code__, self.historical.nnUNetDataLoaderOnlineCP.generate_train_batch.__code__)
        self.assertEqual(m._get_dataloaders.__code__, self.historical._nnUNetTrainer_250epochs_OnlineCP.get_dataloaders.__code__)
        self.assertIs(m._get_dataloaders.__globals__['nnUNetDataLoader'], CPUUnitDataLoader)
        self.assertIs(self.historical.__dict__['nnUNetDataLoaderOnlineCP'], self.historical.nnUNetDataLoaderOnlineCP)
        self.assertIsNot(m._generate.__globals__, self.historical.__dict__)

    def test_actual_dataset_constructor_identifiers_API_full105_not_mapping_keys(self):
        identifiers = [f'CPU_UNIT_case_{number:03d}' for number in range(105)]
        dataset = fixture_actual_dataset(identifiers[::-1])
        self.assertFalse(hasattr(dataset, 'keys'))
        self.assertEqual(dataset.identifiers, identifiers)
        self.assertEqual(self.module.validate_dataset_population(dataset, identifiers[:81]), tuple(identifiers))
        for bad in (identifiers[:-1], identifiers + ['EXTRA_CASE'], identifiers[:-1] + [identifiers[0]]):
            with self.assertRaises(self.module.OnlineCPError):
                self.module.validate_dataset_population(fixture_actual_dataset(bad), identifiers[:81])
        with self.assertRaises(self.module.OnlineCPError):
            self.module.validate_dataset_population(dataset, ['FOREIGN_DONOR_PATIENT'])

    def test_preflight_consumes_actual_dataset_population_before_patient_tasks(self):
        identifiers = [f'CPU_UNIT_case_{number:03d}' for number in range(105)]
        dataset = fixture_actual_dataset(identifiers)
        populations = {key: ('UNIT_ENTRY',) for key in identifiers[:81]}
        # No NPZ/clinical metric is simulated: this test stops at the actual
        # population admission and deliberately fails the first geometry task.
        bank = SimpleNamespace(entries_by_case=populations)
        class CPUUnitForkPool:
            def __init__(self, **kwargs):
                self.configuration = kwargs
                if kwargs['max_workers'] != 4 or kwargs['initializer'] is not self_module._initialize_preflight_worker:
                    raise AssertionError('Full four-process initializer missing')
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def map(self, function, cases):
                return [function(next(iter(cases)))]  # deliberately fails at first UNIT geometry sentinel
        self_module = self.module
        with (patch.object(self.module, 'AblationBank', return_value=bank), patch.object(
                self.module, '_preflight_patient_task', side_effect=RuntimeError('CPU_UNIT_GEOMETRY_SENTINEL')),
                patch.object(self.module, 'ProcessPoolExecutor', CPUUnitForkPool), patch.object(
                self.module.multiprocessing, 'get_all_start_methods', return_value=['fork']), patch.object(
                self.module.multiprocessing, 'get_context', return_value=SimpleNamespace(CPU_UNIT_ONLY=True))):
            with self.assertRaisesRegex(RuntimeError, 'CPU_UNIT_GEOMETRY_SENTINEL'):
                self.module.preflight_bank('CPU_UNIT_NO_FILES', dataset, arm='blend')

    def test_actual_base_constructor_captures_exact_five_arguments_for_all_arms(self):
        actual_constructor = actual_base_constructor_introspection()
        names = ['plans', 'configuration', 'fold', 'dataset_json', 'device']
        original_signature = inspect.signature(self.historical._nnUNetTrainer_250epochs_OnlineCP.__init__)
        new_signature = inspect.signature(self.module._BasicCPAblation.__init__)
        self.assertEqual(list(original_signature.parameters), list(new_signature.parameters))
        self.assertEqual(original_signature.parameters['device'].default, new_signature.parameters['device'].default)
        bank_path = Path(self.temp.name) / 'CPU_UNIT_constructor_index.json'
        bank_path.write_text('{}', encoding='utf8')
        witness_path = Path(self.temp.name) / 'CPU_UNIT_constructor_preflight.json'
        for arm, cls in [('multi3', self.module.nnUNetTrainer_250epochs_BasicCPMulti3),
                         ('candidates256', self.module.nnUNetTrainer_250epochs_BasicCPCandidates256),
                         ('blend', self.module.nnUNetTrainer_250epochs_BasicCPBlend)]:
            witness_path.write_text(json.dumps({'format': self.module.FORMAT, 'arm': arm, 'complete': True,
                'index_sha256': 'CPU_UNIT_SHA', 'historical_trainer_sha256': self.module.HISTORICAL_SHA,
                'entries_sha256': {f'CPU_UNIT_NO_DATA_READ_{i}': 'CPU_UNIT_SHA' for i in range(642)}}), encoding='utf8')
            plans, dataset_json = {'CPU_UNIT_plans': True}, {'CPU_UNIT_dataset': True}
            with patch.object(CPUUnitTrainer, '__init__', actual_constructor), patch.dict(os.environ, {
                'ONLINE_CP_BANK': str(bank_path), 'ONLINE_CP_SEED': '42', 'BASIC_CP_ABLATION': arm,
                'BASIC_CP_ABLATION_PREFLIGHT': str(witness_path)}), patch.object(self.module, 'sha', return_value='CPU_UNIT_SHA'):
                trainer = cls(plans, '3d_fullres', 0, dataset_json, torch.device('cpu'))
            self.assertEqual(list(trainer.my_init_kwargs), names)
            self.assertIs(trainer.my_init_kwargs['plans'], plans)
            self.assertIs(trainer.my_init_kwargs['dataset_json'], dataset_json)
            self.assertEqual(trainer.my_init_kwargs['device'], torch.device('cpu'))
            self.assertEqual(trainer.num_epochs, 250)

    def test_actual_base_introspection_reproduces_the_rejected_varargs_constructor(self):
        actual_constructor = actual_base_constructor_introspection()
        class BrokenVarargsCPUUnit:
            def __init__(self, *args, **kwargs):
                actual_constructor(self, *args, **kwargs)
        with self.assertRaisesRegex(KeyError, 'args'):
            BrokenVarargsCPUUnit({}, '3d_fullres', 0, {}, torch.device('cpu'))

    def test_actual_metadata_full642_and_fixed_baseline_factors(self):
        receipt = ROOT.parents[1] / 'outputs/v24_gpu6_recovery_20261011/HISTORICAL_BASIC_CP_CONTRACT_READONLY_1791835410056009100.json'
        original = json.loads(receipt.read_text(encoding='utf8'))['bank']
        self.module.validate_metadata(original, 'multi3')
        self.module.validate_metadata(original, 'blend')
        for key, wrong in [('source_entries', 641), ('candidate_count', 64), ('cp_probability', 1.),
                           ('maximum_diameter_mm', 100.), ('network_patch_size', [64] * 3),
                           ('intensity_scale_range', [1., 1.])]:
            invalid = copy.deepcopy(original); invalid[key] = wrong
            with self.assertRaises(self.module.OnlineCPError):
                self.module.validate_metadata(invalid, 'multi3')

    def test256_metadata_cannot_claim_fabricated_scores_or_new_sources(self):
        receipt = ROOT.parents[1] / 'outputs/v24_gpu6_recovery_20261011/HISTORICAL_BASIC_CP_CONTRACT_READONLY_1791835410056009100.json'
        extended = copy.deepcopy(json.loads(receipt.read_text(encoding='utf8'))['bank'])
        extended.update(candidate_count=256, total_candidates=642 * 256, ablation={
            'format': self.module.EXTENSION_FORMAT, 'uniform_candidate_count': 256, 'scores_used': False,
            'historical_score_count': 128, 'original_candidate_count': 128,
            'original_index_sha256': self.module.BANK_SHA, 'additional_GNN_scores_generated': False,
            'historical_graph_geometry_required': True})
        self.module.validate_metadata(extended, 'candidates256')
        for key, wrong in [('scores_used', True), ('historical_score_count', 256),
                           ('additional_GNN_scores_generated', True), ('original_index_sha256', '0' * 64)]:
            invalid = copy.deepcopy(extended); invalid['ablation'][key] = wrong
            with self.assertRaises(self.module.OnlineCPError):
                self.module.validate_metadata(invalid, 'candidates256')

    def test_historical_primary_gate_RNG_token_source_and_intensity_unchanged(self):
        centers = np.stack([np.arange(128), np.full(128, 20), np.full(128, 20)], axis=1)
        own = entry(centers)
        new = self.loader('multi3', [own])
        old = self.loader('multi3', [own])
        old._select_candidate = lambda scores, u: self.historical.nnUNetDataLoaderOnlineCP._select_candidate(old, scores, u)
        for _ in range(100):
            expected, old_token = self.historical.nnUNetDataLoaderOnlineCP._sample_paste_plan(old, 'OWN_PATIENT')
            actual, new_token = new._sample_paste_plan('OWN_PATIENT')
            self.assertEqual(old_token, new_token)
            self.assertEqual(expected is None, actual is None)
            if expected:
                for key in ['entry', 'center', 'scale', 'normalized_offset']:
                    self.assertEqual(expected[key], actual[key])
        self.assertEqual(old._rng().random(), new._rng().random())

    def test_candidates256_uses_all_centers_uniformly_and_never_GNN_scores(self):
        own = entry([[i, 4, 4] for i in range(256)])
        loader = self.loader('candidates256', [own], count=256)
        self.assertEqual(loader._select_candidate(own['scores'], 0.), 0)
        self.assertEqual(loader._select_candidate(own['scores'], .5), 128)
        self.assertEqual(loader._select_candidate(own['scores'], 1. - 1e-8), 255)
        permuted_scores = own['scores'][::-1]
        self.assertEqual(loader._select_candidate(permuted_scores, .9), 230)

    def test_private256_reader_accepts_real128_scores_without_fabrication(self):
        folder = Path(self.temp.name) / 'reader256'; folder.mkdir(exist_ok=True)
        own = entry([[i, 4, 4] for i in range(256)])
        np.savez(folder / 'entry.npz', **own)
        bank = SimpleNamespace(_cache=OrderedDict(), _cache_limit=642, root=folder.resolve(), candidate_count=256)
        loaded = self.module._load256(bank, 'entry.npz')
        np.testing.assert_array_equal(loaded['scores'], own['scores'])
        self.assertEqual(loaded['scores'].shape, (128,))
        self.assertEqual(loaded['candidate_centers'].shape, (256, 3))
        with self.assertRaises(self.module.OnlineCPError):
            self.historical.OnlineCPBank._load(SimpleNamespace(_cache=OrderedDict(), _cache_limit=642,
                root=folder.resolve(), candidate_count=256), 'entry.npz')

    def test_exact_three_full_same_patient_pastes_with_primary_unchanged(self):
        own = entry([[12, 12, 12], [22, 12, 12], [32, 12, 12]])
        loader = self.loader('multi3', [own]); loader._ablation_case_shape = (160, 160, 160)
        image = np.full((1, 128, 128, 128), 1., dtype=np.float32)
        seg = np.ones((1, 128, 128, 128), dtype=np.int16)
        primary = {'entry': own, 'center': (12, 12, 12), 'scale': 1., 'normalized_offset': 0.,
                   'ablation_schedule_token': 98}
        loader._apply_paste_to_crop(image, seg, [0, 0, 0], primary, 'OWN_PATIENT')
        self.assertEqual(np.count_nonzero(seg == 2), 3 * 27)
        self.assertEqual(ndi.label(seg[0] == 2)[1], 3)
        np.testing.assert_array_equal(image[0, 11:14, 11:14, 11:14], own['source_data'][0])
        self.assertTrue(np.all(image[seg != 2] == 1.))

    def test_no_partial_paste_if_three_nonoverlapping_positions_do_not_exist(self):
        own = entry([[12, 12, 12], [13, 12, 12]])
        loader = self.loader('multi3', [own]); loader._ablation_case_shape = (160, 160, 160)
        image = np.ones((1, 128, 128, 128), dtype=np.float32)
        seg = np.ones_like(image, dtype=np.int16)
        primary = {'entry': own, 'center': (12, 12, 12), 'scale': 1., 'normalized_offset': 0.,
                   'ablation_schedule_token': 98}
        with self.assertRaisesRegex(self.module.OnlineCPError, 'no 1/2-paste fallback'):
            loader._apply_paste_to_crop(image, seg, [0, 0, 0], primary, 'OWN_PATIENT')
        self.assertTrue(np.all(image == 1.)); self.assertTrue(np.all(seg == 1))

    def test_exact_mask_overlap_not_merely_bbox_overlap(self):
        first = entry([[5, 5, 5]], side=3); second = entry([[5, 5, 5]], side=3)
        first['source_mask'][:] = 0; second['source_mask'][:] = 0
        first['source_mask'][0, 0, 0] = 1; second['source_mask'][2, 2, 2] = 1
        self.assertFalse(self.module.mask_overlap(first, (5, 5, 5), second, (5, 5, 5)))
        second['source_mask'][0, 0, 0] = 1
        self.assertTrue(self.module.mask_overlap(first, (5, 5, 5), second, (5, 5, 5)))

    def test_blend_exact_historical_feather_math_and_full_tumor_labels(self):
        own = entry([[5, 5, 5]], side=7, value=9.)
        # Pad the support to represent the actual stored source patch margin.
        own['source_mask'][:] = 0; own['source_mask'][1:6, 1:6, 1:6] = 1
        loader = self.loader('blend', [own])
        image = np.ones((1, 12, 12, 12), dtype=np.float32)
        seg = np.ones_like(image, dtype=np.int16)
        primary = {'entry': own, 'center': (5, 5, 5), 'scale': 1., 'normalized_offset': 0.}
        loader._apply_paste_to_crop(image, seg, [0, 0, 0], primary, 'OWN_PATIENT')
        alpha = np.clip(ndi.distance_transform_edt(own['source_mask'].astype(bool)) / 2., 0, 1).astype(np.float32)
        np.testing.assert_array_equal(self.module.feather_alpha(own['source_mask']), alpha)
        self.assertEqual(image[0, 3, 3, 3], 5.)  # innermost boundary alpha=.5
        self.assertEqual(image[0, 5, 5, 5], 9.)
        self.assertEqual(np.count_nonzero(seg == 2), 125)
        self.assertTrue(np.all(image[seg != 2] == 1.))

    def test_uniform_extra_sources_with_replacement_keeps_single_source_patient(self):
        own = entry([[10, 10, 10], [20, 10, 10], [30, 10, 10]])
        primary = {'entry': own, 'center': (10, 10, 10)}
        extra = self.module.select_extra_pastes([own], primary, np.zeros(3), np.full(3, 128),
                                               np.random.default_rng(42))
        self.assertEqual(len(extra), 2)
        self.assertTrue(all(plan['entry'] is own for plan in extra))
        self.assertEqual(len({primary['center'], *(plan['center'] for plan in extra)}), 3)

    def test_unchanged_gate_preserves_no_source_patient_without_donor_substitution(self):
        loader = self.loader('multi3', [entry([[10, 10, 10]])])
        for _ in range(30):
            plan, token = loader._sample_paste_plan('NO_OWN_SOURCE')
            self.assertIsNone(plan); self.assertIsInstance(token, int)

    def test_geometry_never_places_additional_copies_outside_original_network_crop(self):
        own = entry([[10, 10, 10], [20, 10, 10], [140, 10, 10]])
        primary = {'entry': own, 'center': (10, 10, 10)}
        with self.assertRaises(self.module.OnlineCPError):
            self.module.select_extra_pastes([own], primary, np.zeros(3), np.full(3, 128))

    def test_observed_liver28_geometry_moves_crop_without_changing_source_or_candidate_centers(self):
        # Shape/anchor/centers are actual read-only geometry summaries. The
        # single-voxel mask is explicitly synthetic, never a patient-array copy.
        own = {'source_data': np.full((1, 13, 34, 34), 9., dtype=np.float32),
               'source_mask': np.zeros((13, 34, 34), dtype=np.uint8),
               'anchor_offset': np.asarray([6, 16, 17], dtype=np.int64),
               'candidate_centers': np.asarray([[107, 407, 201], [111, 451, 252], [103, 464, 249]], dtype=np.int64),
               'scores': np.arange(128, dtype=np.float32)}
        own['source_mask'][6, 16, 17] = 1
        primary = {'entry': own, 'center': (107, 407, 201)}
        original_lower, original_upper = self.module.canonical_crop((129, 667, 667), primary)
        with self.assertRaises(self.module.OnlineCPError):
            self.module.select_extra_pastes([own], primary, original_lower, original_upper)
        extras = self.module.select_extra_pastes([own], primary, np.zeros(3), np.asarray([129, 667, 667]),
                                                maximum_union_shape=(128, 128, 128))
        plans = [primary, *extras]
        lower, upper = self.module.joint_network_crop((129, 667, 667), plans)
        np.testing.assert_array_equal(lower, [1, 354, 141])
        np.testing.assert_array_equal(upper - lower, [128] * 3)
        self.assertEqual(primary['center'], (107, 407, 201))
        self.assertTrue(all(plan['entry'] is own for plan in plans))
        self.assertEqual({plan['center'] for plan in plans}, set(map(tuple, own['candidate_centers'])))
        initial = SimpleNamespace(patch_size=np.asarray([218] * 3), final_patch_size=np.asarray([128] * 3),
                                  need_to_pad=np.asarray([90] * 3))
        augmented_lower, augmented_upper, base_lower = self.module.initial_crop_around_joint(initial,
                   (129, 667, 667), plans, lower)
        np.testing.assert_array_equal(np.asarray(augmented_upper) - augmented_lower, [218] * 3)
        np.testing.assert_array_equal(base_lower, lower)

    def test_joint_paste_three_inside_shifted128_and_no_partial_array_write(self):
        own = entry([[120, 130, 130], [180, 130, 130], [190, 130, 130]], side=3)
        loader = self.loader('multi3', [own]); loader._ablation_case_shape = (320, 320, 320)
        primary = {'entry': own, 'center': (120, 130, 130), 'scale': 1., 'normalized_offset': 0.,
                   'ablation_schedule_token': 98}
        plans = self.module.prepare_joint_plan(loader, loader._ablation_case_shape, primary, 'OWN_PATIENT')
        lower = np.asarray(primary['ablation_joint128_lower'])
        self.assertEqual(len(plans), 3)
        image = np.ones((1, 128, 128, 128), dtype=np.float32)
        seg = np.ones_like(image, dtype=np.int16)
        loader._apply_paste_to_crop(image, seg, lower, primary, 'OWN_PATIENT')
        self.assertEqual(np.count_nonzero(seg == 2), 81)
        self.assertEqual(ndi.label(seg[0] == 2)[1], 3)
        self.assertTrue(np.all(image[seg != 2] == 1.))
        impossible = entry([[10, 10, 10], [220, 10, 10], [230, 10, 10]])
        with self.assertRaisesRegex(self.module.OnlineCPError, 'no 1/2-paste fallback'):
            self.module.select_extra_pastes([impossible], {'entry': impossible, 'center': (10, 10, 10)},
                     np.zeros(3), np.full(3, 320), maximum_union_shape=(128, 128, 128))

    def test_joint_selection_preserves_every_primary_index_0_to127(self):
        # Full128 symbolic centers; count is never reduced to make union fit.
        centers = [[10 + i % 8 * 10, 10 + i // 8 * 6, 40] for i in range(128)]
        own = entry(centers)
        for index in range(128):
            primary = {'entry': own, 'center': tuple(own['candidate_centers'][index])}
            extras = self.module.select_extra_pastes([own], primary, np.zeros(3), np.full(3, 160),
                          np.random.default_rng(index), maximum_union_shape=(128, 128, 128))
            lower, upper = self.module.joint_network_crop((160, 160, 160), [primary, *extras])
            self.assertEqual(primary['center'], tuple(own['candidate_centers'][index]))
            np.testing.assert_array_equal(upper - lower, [128] * 3)

    def test_original_bbox_callback_and_apply_share_same_three_plans_and_initial_shape(self):
        own = entry([[120, 130, 130], [180, 130, 130], [190, 130, 130]])
        loader = self.loader('multi3', [own])
        loader.patch_size = np.asarray([218] * 3); loader.final_patch_size = np.asarray([128] * 3)
        loader.need_to_pad = np.asarray([90] * 3)
        primary = {'entry': own, 'center': (120, 130, 130), 'scale': 1., 'normalized_offset': 0.,
                   'ablation_schedule_token': 98}
        loader._pending_paste_plan = primary; loader._pending_case_id = 'OWN_PATIENT'
        original_rng_state = copy.deepcopy(loader._rng().bit_generator.state)
        lower, upper = self.module._remember_case_shape(loader, (320, 320, 320), primary['center'],
                            own['source_mask'].shape, own['anchor_offset'])
        np.testing.assert_array_equal(np.asarray(upper) - lower, [218] * 3)
        self.assertEqual(original_rng_state, loader._rng().bit_generator.state)
        before_plans = tuple((item['center'], item['scale'], item['normalized_offset'])
                             for item in primary['ablation_extra_plans'])
        image = np.ones((1, 218, 218, 218), dtype=np.float32)
        seg = np.ones_like(image, dtype=np.int16)
        loader._apply_paste_to_crop(image, seg, lower, primary, 'OWN_PATIENT')
        self.assertEqual(before_plans, tuple((item['center'], item['scale'], item['normalized_offset'])
                                           for item in primary['ablation_extra_plans']))
        self.assertEqual(np.count_nonzero(seg == 2), 81)
        offset = np.asarray(primary['ablation_joint128_lower']) - lower
        sl = tuple(slice(int(v), int(v + 128)) for v in offset)
        self.assertEqual(np.count_nonzero(seg[(0, *sl)] == 2), 81)
        self.assertEqual(original_rng_state, loader._rng().bit_generator.state)

    def test_fork_worker_rejects_initialized_CUDA_or_missing_readonly_instance_binding(self):
        dataset = fixture_actual_dataset([f'CPU_UNIT_case_{i:03d}' for i in range(105)])
        with self.assertRaisesRegex(self.module.OnlineCPError, 'read-only adapter binding'):
            self.module._initialize_preflight_worker('CPU_UNIT_NO_BANK', dataset, 'multi3', (128, 128, 128))
        dataset._v24_readonly_binding = {'CPU_UNIT_BINDING': True}
        with patch.object(self.module.historical.torch.cuda, 'is_initialized', return_value=True):
            with self.assertRaisesRegex(self.module.OnlineCPError, 'initialized CUDA'):
                self.module._initialize_preflight_worker('CPU_UNIT_NO_BANK', dataset, 'multi3', (128, 128, 128))

    def test_worker_copy_retains_authoritative_binding_and_releases_patient_cache(self):
        dataset = fixture_actual_dataset([f'CPU_UNIT_case_{i:03d}' for i in range(105)])
        binding = {'CPU_UNIT_readonly_binding': True}; dataset._v24_readonly_binding = binding
        bank = SimpleNamespace(entries_by_case={dataset.identifiers[0]: ['CPU_UNIT_ENTRY']},
                               _cache=OrderedDict([('CPU_UNIT_ENTRY', {'CPU_UNIT_ARRAY': True})]),
                               _validated_entries={'CPU_UNIT_ENTRY'})
        with patch.object(self.module, 'AblationBank', return_value=bank):
            self.module._initialize_preflight_worker('CPU_UNIT_INDEX', dataset, 'multi3', (128, 128, 128))
        worker_bank, worker_dataset, _ = self.module._PREFLIGHT_WORKER
        self.assertIsNot(worker_dataset, dataset)
        self.assertIs(worker_dataset._v24_readonly_binding, binding)
        self.assertEqual(worker_dataset.identifiers, dataset.identifiers)
        with patch.object(self.module, '_validate_patient', side_effect=RuntimeError('CPU_UNIT_EXPLICIT_FAILURE')):
            with self.assertRaisesRegex(RuntimeError, 'CPU_UNIT_EXPLICIT_FAILURE'):
                self.module._preflight_patient_task(dataset.identifiers[0])
        self.assertFalse(worker_bank._cache); self.assertFalse(worker_bank._validated_entries)
        self.module._PREFLIGHT_WORKER = None


if __name__ == '__main__':
    unittest.main()
