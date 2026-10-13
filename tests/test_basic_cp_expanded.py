"""CPU DEBUG geometry/RNG fixtures, never clinical training or quality evidence."""
from __future__ import annotations

from collections import OrderedDict
import copy
import importlib.util
import inspect
from itertools import combinations
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy import ndimage as ndi
from tests.test_basic_cp_ablation_trainer import (
    ROOT, HISTORICAL_NAME, CPUUnitTrainer, actual_base_constructor_introspection,
    fixture_historical, entry,
)


class ExpandedBasicCPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / 'tmp').mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tmp')
        cls.historical = fixture_historical(Path(cls.temp.name) / 'historical_actual.py')
        packages = {}
        for name in ['nnunetv2', 'nnunetv2.training', 'nnunetv2.training.nnUNetTrainer', 'expanded_cpu_unit']:
            package = ModuleType(name); package.__path__ = []
            packages[name] = package
        packages[HISTORICAL_NAME] = cls.historical
        cls.patch = patch.dict('sys.modules', packages); cls.patch.start()
        for name in ('nnUNetTrainer_BasicCPAblations', 'nnUNetTrainer_BasicCPExpanded'):
            spec = importlib.util.spec_from_file_location('expanded_cpu_unit.' + name,
                                                         ROOT / 'custom_trainers' / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            packages['expanded_cpu_unit'].__dict__[name] = module
            import sys
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        cls.base = packages['expanded_cpu_unit'].nnUNetTrainer_BasicCPAblations
        cls.module = packages['expanded_cpu_unit'].nnUNetTrainer_BasicCPExpanded

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop(); cls.temp.cleanup()

    def loader(self, arm, entries, count=128):
        loader = self.module.ExpandedDataLoader.__new__(self.module.ExpandedDataLoader)
        loader.arm = arm; loader.online_seed = 42; loader.online_epoch = 7
        loader.online_policy = 'basic'; loader._cp_rng = None; loader.thread_id = 0
        geometry = self.module.CandidateGeometry(entries)
        loader.online_bank = SimpleNamespace(candidate_count=count, cp_probability=.5,
            intensity_scale=(.95, 1.05), intensity_shift_hu=(-5., 5.), ct_mean=99.48007202148438,
            ct_std=37.13840103149414, tumor_label=2,
            entry_names=lambda case: tuple(range(len(entries))) if case == 'OWN_PATIENT' else (),
            load_for_case=lambda case, index: entries[index], case_entries=lambda case: entries,
            geometry_for_case=lambda case: geometry)
        return loader

    def grid_entry(self):
        return entry([[10 + i % 8 * 10, 10 + i // 8 * 6, 40] for i in range(128)])

    def test_seven_explicit_classes_keep_full250_constructor_signature(self):
        m = self.module
        names = [('multi5', 'Multi5'), ('multi7', 'Multi7'), ('multi10', 'Multi10'),
                 ('candidates512', 'Candidates512'), ('candidates1024', 'Candidates1024'),
                 ('candidates2048', 'Candidates2048'), ('blend50', 'Blend50')]
        actual_constructor = actual_base_constructor_introspection()
        bank = Path(self.temp.name) / 'CPU_UNIT_index.json'; bank.write_text('{}')
        witness = Path(self.temp.name) / 'CPU_UNIT_preflight.json'
        for arm, suffix in names:
            trainer_class = getattr(m, 'nnUNetTrainer_250epochs_BasicCP' + suffix)
            self.assertEqual(list(inspect.signature(trainer_class.__init__).parameters),
                             ['self', 'plans', 'configuration', 'fold', 'dataset_json', 'device'])
            receipt = dict(format=m.FORMAT, arm=arm, complete=True, index_sha256='CPU_UNIT_SHA',
                historical_trainer_sha256=m.HISTORICAL_SHA, entries_sha256={str(i): 'CPU_UNIT_SHA' for i in range(642)},
                copies_per_event=m.copies_per_event(arm), candidate_count=m.candidates_per_source(arm),
                blend_border_preprocessed_voxels=0, physical_batch_size=2, epochs=250,
                blend_mode='constant', blend_alpha=.5, label_blending=False)
            witness.write_text(json.dumps(receipt))
            with patch.object(CPUUnitTrainer, '__init__', actual_constructor), patch.dict(os.environ, {
                    'ONLINE_CP_BANK': str(bank), 'ONLINE_CP_SEED': '42', 'BASIC_CP_EXPANDED_ARM': arm,
                    'BASIC_CP_EXPANDED_PREFLIGHT': str(witness)}), patch.object(m, 'sha', return_value='CPU_UNIT_SHA'):
                trainer = trainer_class({}, '3d_fullres', 0, {}, torch.device('cpu'))
            self.assertEqual(trainer.num_epochs, 250)
            self.assertEqual(list(trainer.my_init_kwargs), ['plans', 'configuration', 'fold', 'dataset_json', 'device'])

    def test_metadata_all642_and_original_factors_remain_immutable(self):
        m = self.module
        receipt = ROOT.parents[1] / 'outputs/v24_gpu6_recovery_20261011/HISTORICAL_BASIC_CP_CONTRACT_READONLY_1791835410056009100.json'
        original = json.loads(receipt.read_text(encoding='utf8'))['bank']
        for arm in m.ARMS:
            data = copy.deepcopy(original); count = m.candidates_per_source(arm)
            data.update(candidate_count=count, total_candidates=642 * count)
            if count != 128:
                data['ablation'] = dict(format=m.EXTENSION_FORMAT, uniform_candidate_count=count,
                    scores_used=False, historical_score_count=128, original_candidate_count=128,
                    original_index_sha256=m.BANK_SHA, additional_GNN_scores_generated=False,
                    historical_graph_geometry_required=True)
            m.validate_metadata(data, arm)
            for key, wrong in [('source_entries', 641), ('cp_probability', 1.), ('network_patch_size', [64] * 3),
                               ('intensity_scale_range', [1., 1.]), ('candidate_count', count - 1)]:
                bad = copy.deepcopy(data); bad[key] = wrong
                with self.assertRaises(m.OnlineCPError): m.validate_metadata(bad, arm)
            if count != 128:
                for key, wrong in [('scores_used', True), ('historical_score_count', count),
                                   ('original_index_sha256', '0' * 64), ('additional_GNN_scores_generated', True)]:
                    bad = copy.deepcopy(data); bad['ablation'][key] = wrong
                    with self.assertRaises(m.OnlineCPError): m.validate_metadata(bad, arm)

    def test_every_original128_primary_has_exact_full5_7_10_without_source_substitution(self):
        m = self.module; own = self.grid_entry(); geometry = m.CandidateGeometry([own])
        for count in (5, 7, 10):
            for index, center in enumerate(own['candidate_centers']):
                primary = dict(entry=own, center=tuple(map(int, center)))
                extras = geometry.choose(primary, np.zeros(3), np.full(3, 160), count, np.random.default_rng(index))
                self.assertEqual(len(extras), count - 1)
                self.assertTrue(all(item['entry'] is own for item in extras))
                plans = [primary, *extras]
                lower, upper = m.joint_network_crop((160, 160, 160), plans)
                np.testing.assert_array_equal(upper - lower, [128] * 3)
                for i, item in enumerate(plans):
                    for previous in plans[:i]:
                        self.assertFalse(self.base.mask_overlap(item['entry'], item['center'], previous['entry'], previous['center']))
                self.assertEqual(primary['center'], tuple(center))

    def test_full_count_search_backtracks_and_rejects_impossible_without_caps(self):
        m = self.module
        # Bad first placements occupy the center of later dense groups; full
        # exhaustive search still reaches a feasible complete combination.
        own = entry([[20, 20, 20], [30, 20, 20], [29, 20, 20], [31, 20, 20],
                     [40, 20, 20], [50, 20, 20], [60, 20, 20], [70, 20, 20], [80, 20, 20], [90, 20, 20]])
        primary = dict(entry=own, center=(20, 20, 20))
        extras = m.select_extra_pastes([own], primary, np.zeros(3), np.full(3, 128), copies=7)
        self.assertEqual(len(extras), 6)
        with self.assertRaisesRegex(m.OnlineCPError, 'no partial paste fallback'):
            m.select_extra_pastes([own], primary, np.zeros(3), np.full(3, 128), copies=10)
        far = entry([[20, 20, 20], *[[i * 50, 20, 20] for i in range(2, 12)]])
        with self.assertRaises(m.OnlineCPError):
            m.select_extra_pastes([far], dict(entry=far, center=(20, 20, 20)), np.zeros(3), np.full(3, 600), copies=5)

    def test_exhaustive_search_matches_independent_bruteforce_and_backtracks_bad_first_source(self):
        m = self.module
        own = entry([[10, 60, 60], [30, 60, 60], [45, 60, 60], [65, 60, 60], [90, 60, 60]])
        blocker = entry([[50, 60, 60]], side=51)
        geometry = m.CandidateGeometry([blocker, own])
        primary = dict(entry=own, center=(10, 60, 60))
        result = geometry.choose(primary, np.zeros(3), np.full(3, 128), 5)
        self.assertEqual({item['center'] for item in result}, set(map(tuple, own['candidate_centers'][1:])))
        self.assertIn(0, geometry._conflicts, 'The impossible first source must actually be explored')
        for seed in range(20):
            positions = np.random.default_rng(seed).choice(np.arange(10, 35), 8, replace=False)
            own = entry([[int(v), 20, 20] for v in positions])
            geometry = m.CandidateGeometry([own]); primary = dict(entry=own, center=tuple(own['candidate_centers'][0]))
            feasible = False
            for extras in combinations(range(1, 8), 4):
                chosen = (0, *extras)
                if all(not self.base.mask_overlap(own, own['candidate_centers'][i], own, own['candidate_centers'][j])
                       for i, j in combinations(chosen, 2)):
                    feasible = True; break
            if feasible:
                self.assertEqual(len(geometry.choose(primary, np.zeros(3), np.full(3, 128), 5,
                                                     np.random.default_rng(seed))), 4)
            else:
                with self.assertRaises(m.OnlineCPError):
                    geometry.choose(primary, np.zeros(3), np.full(3, 128), 5, np.random.default_rng(seed))

    def test_historical_source_gate_intensity_and_future_rng_unchanged_for_all_arms(self):
        for arm in self.module.ARMS:
            count = self.module.candidates_per_source(arm)
            own = entry([[i, 20, 20] for i in range(count)])
            new = self.loader(arm, [own], count); old = self.loader(arm, [own], count)
            old._select_candidate = lambda scores, u: min(count - 1, int(np.floor(u * count)))
            for _ in range(100):
                expected, token = self.historical.nnUNetDataLoaderOnlineCP._sample_paste_plan(old, 'OWN_PATIENT')
                actual, new_token = new._sample_paste_plan('OWN_PATIENT')
                self.assertEqual(token, new_token); self.assertEqual(expected is None, actual is None)
                if expected:
                    for key in ['entry', 'center', 'scale', 'normalized_offset']:
                        self.assertEqual(expected[key], actual[key])
            self.assertEqual(old._rng().random(), new._rng().random())
            for _ in range(20):
                plan, token = new._sample_paste_plan('NO_OWN_SOURCE')
                self.assertIsNone(plan); self.assertIsInstance(token, int)

    def test_candidates512_1024_2048_uniform_all_and_real128_scores_unchanged(self):
        for count in (512, 1024, 2048):
            folder = Path(self.temp.name) / f'bank{count}'; folder.mkdir(exist_ok=True)
            own = entry([[i, 4, 4] for i in range(count)]); np.savez(folder / 'entry.npz', **own)
            bank = SimpleNamespace(_cache=OrderedDict(), _cache_limit=642, root=folder.resolve(), candidate_count=count)
            loaded = self.base._load256(bank, 'entry.npz')
            self.assertEqual(loaded['candidate_centers'].shape, (count, 3))
            np.testing.assert_array_equal(loaded['scores'], own['scores'])
            loader = self.loader(f'candidates{count}', [own], count)
            values = (np.arange(count) + .5) / count
            self.assertEqual([loader._select_candidate(own['scores'], u) for u in values], list(range(count)))
            self.assertEqual(loader._select_candidate(own['scores'][::-1], .9), int(.9 * count))

    def test_full5_7_10_masks_written_atomically_inside_shifted128_crop(self):
        m = self.module; own = self.grid_entry()
        for count in (5, 7, 10):
            loader = self.loader(f'multi{count}', [own]); loader._expanded_case_shape = (160, 160, 160)
            primary = dict(entry=own, center=(10, 10, 40), scale=1., normalized_offset=0., expanded_schedule_token=98)
            plans = m.prepare_joint_plan(loader, loader._expanded_case_shape, primary, 'OWN_PATIENT')
            lower = np.asarray(primary['expanded_joint128_lower'])
            image = np.ones((1, 128, 128, 128), dtype=np.float32); labels = np.ones_like(image, dtype=np.int16)
            loader._apply_paste_to_crop(image, labels, lower, primary, 'OWN_PATIENT')
            self.assertEqual(np.count_nonzero(labels == 2), count * 27)
            self.assertEqual(ndi.label(labels[0] == 2)[1], count)
            self.assertTrue(np.all(image[labels != 2] == 1.))
            self.assertEqual(len(plans), count)
            # Pre-write validation rejects any invalid extra source completely.
            bad = dict(primary); bad['expanded_extra_plans'] = [dict(item) for item in primary['expanded_extra_plans']]
            bad['expanded_extra_plans'][-1]['center'] = primary['center']
            image[:] = 1.; labels[:] = 1
            with self.assertRaises(m.OnlineCPError): loader._apply_paste_to_crop(image, labels, lower, bad, 'OWN_PATIENT')
            self.assertTrue(np.all(image == 1.)); self.assertTrue(np.all(labels == 1))

    def test_fixed50_entire_mask_after_jitter_and_hard_labels_no_feather(self):
        own = entry([[3, 3, 3]], side=5, value=100.)
        own['source_mask'][:] = 0; own['source_mask'][1:4, 1:4, 1:4] = 1
        own['_basic_ablation_alpha2'] = np.zeros((5, 5, 5))
        loader = self.loader('blend50', [own])
        image = np.full((1, 7, 7, 7), 10, dtype=np.float32); labels = np.ones_like(image, dtype=np.int16)
        plan = dict(entry=own, center=(3, 3, 3), scale=1.05, normalized_offset=5.)
        loader._apply_paste_to_crop(image, labels, [0, 0, 0], plan, 'OWN_PATIENT')
        expected = np.full_like(image, 10); expected[:, 2:5, 2:5, 2:5] = 60
        hard = np.ones_like(labels); hard[:, 2:5, 2:5, 2:5] = 2
        np.testing.assert_allclose(image, expected); np.testing.assert_array_equal(labels, hard)

    def test_callback_retains_original_augmentation_shape_and_dedicated_extra_rng(self):
        m = self.module; own = self.grid_entry()
        for count in (5, 7, 10):
            loader = self.loader(f'multi{count}', [own])
            loader.patch_size = np.asarray([218] * 3); loader.final_patch_size = np.asarray([128] * 3)
            loader.need_to_pad = np.asarray([90] * 3)
            primary = dict(entry=own, center=(10, 10, 40), scale=1., normalized_offset=0., expanded_schedule_token=98)
            loader._pending_paste_plan = primary; loader._pending_case_id = 'OWN_PATIENT'
            state = copy.deepcopy(loader._rng().bit_generator.state)
            lower, upper = m._remember_case_shape(loader, (160, 160, 160), primary['center'], own['source_mask'].shape, own['anchor_offset'])
            np.testing.assert_array_equal(np.asarray(upper) - lower, [218] * 3)
            self.assertEqual(state, loader._rng().bit_generator.state)
            self.assertEqual(len(primary['expanded_extra_plans']), count - 1)


if __name__ == '__main__': unittest.main()
