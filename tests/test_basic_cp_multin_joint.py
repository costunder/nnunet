"""CPU DEBUG: approved primary-only rejection, never clinical model results."""
from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tests.test_basic_cp_ablation_trainer import ROOT, CPUUnitTrainer, actual_base_constructor_introspection, entry
from tests import test_basic_cp_expanded as expanded_fixture


class MultiNJointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        expanded_fixture.ExpandedBasicCPTests.setUpClass()
        cls.expanded = expanded_fixture.ExpandedBasicCPTests.module
        cls.historical = expanded_fixture.ExpandedBasicCPTests.historical
        cls.folder = Path(expanded_fixture.ExpandedBasicCPTests.temp.name)
        spec = importlib.util.spec_from_file_location('expanded_cpu_unit.nnUNetTrainer_BasicCPMultiNJoint',
                                                     ROOT / 'custom_trainers/nnUNetTrainer_BasicCPMultiNJoint.py')
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        expanded_fixture.ExpandedBasicCPTests.tearDownClass()

    def loader(self, arm, own, feasible):
        m = self.module
        loader = m.JointDataLoader.__new__(m.JointDataLoader)
        loader.arm = arm; loader.online_seed = 42; loader.online_epoch = 7
        loader.online_policy = 'basic'; loader._cp_rng = None; loader.thread_id = 0
        own['_joint_source_relative_path'] = 'OWN_SOURCE.npz'
        geometry = self.expanded.CandidateGeometry([own])
        loader.online_bank = SimpleNamespace(candidate_count=128, cp_probability=.5,
            intensity_scale=(.95, 1.05), intensity_shift_hu=(-5., 5.), ct_mean=99.48007202148438,
            ct_std=37.13840103149414, tumor_label=2,
            entry_names=lambda case: ('OWN_SOURCE.npz',) if case == 'OWN_PATIENT' else (),
            load_for_case=lambda case, index: own, case_entries=lambda case: [own],
            geometry_for_case=lambda case: geometry, feasible_primary_indices={'OWN_SOURCE.npz': list(feasible)})
        return loader

    def grid_entry(self):
        return entry([[10 + i % 8 * 10, 10 + i // 8 * 6, 40] for i in range(128)])

    def test_only_infeasible_original_primary_redrawn_same_source_and_future_rng_unchanged(self):
        m = self.module
        for arm in m.ARMS:
            own = self.grid_entry(); feasible = [0, 1, 6, 127]
            new = self.loader(arm, own, feasible); old = self.loader(arm, own, feasible)
            new._joint_batch_resampled = []; observed_cp = []; observed_resampled = []
            for _ in range(400):
                original, token = self.historical.nnUNetDataLoaderOnlineCP._sample_paste_plan(old, 'OWN_PATIENT')
                actual, actual_token = new._sample_paste_plan('OWN_PATIENT')
                self.assertEqual(token, actual_token); self.assertEqual(original is None, actual is None)
                observed_cp.append(original is not None)
                if original is None:
                    observed_resampled.append(False); continue
                self.assertIs(actual['entry'], original['entry'])
                self.assertEqual(actual['scale'], original['scale'])
                self.assertEqual(actual['normalized_offset'], original['normalized_offset'])
                index = int(np.flatnonzero(np.all(own['candidate_centers'] == original['center'], axis=1))[0])
                self.assertEqual(actual['joint_original_primary_index'], index)
                if index in feasible:
                    self.assertEqual(actual['center'], original['center'])
                    self.assertFalse(actual['joint_primary_resampled'])
                else:
                    self.assertIn(actual['joint_selected_primary_index'], feasible)
                    self.assertTrue(actual['joint_primary_resampled'])
                observed_resampled.append(index not in feasible)
                self.assertEqual(actual['expanded_schedule_token'], token)
                self.assertIs(new._pending_paste_plan, actual)
            self.assertEqual(new._joint_batch_resampled, observed_resampled)
            self.assertTrue(all(not redraw or cp for redraw, cp in zip(observed_resampled, observed_cp)))
            self.assertEqual(new._rng().random(), old._rng().random())
            for _ in range(20):
                plan, token = new._sample_paste_plan('NO_OWN_SOURCE')
                self.assertIsNone(plan)

    def test_rejection_is_uniform_over_feasible_original_indices_and_deterministic(self):
        m = self.module; own = self.grid_entry(); feasible = [2, 99, 127]
        loader = self.loader('multi10', own, feasible); counts = {v: 0 for v in feasible}
        for token in range(1200):
            # Keep the real source/image/centers; only supply a deterministic
            # original schedule token to exercise the independent redraw RNG.
            original = dict(entry=own, center=tuple(own['candidate_centers'][0]), scale=1.04,
                            normalized_offset=-.3, expanded_schedule_token=token)
            with patch.object(self.expanded.ExpandedDataLoader, '_sample_paste_plan', return_value=(original, token)):
                plan, result_token = loader._sample_paste_plan('OWN_PATIENT')
            counts[plan['joint_selected_primary_index']] += 1
            self.assertEqual(plan['scale'], 1.04); self.assertEqual(plan['normalized_offset'], -.3)
            self.assertEqual(result_token, token)
        self.assertTrue(all(330 <= amount <= 470 for amount in counts.values()), counts)
        plans = []
        for _ in range(2):
            original = dict(entry=own, center=tuple(own['candidate_centers'][0]), scale=1., normalized_offset=0.)
            with patch.object(self.expanded.ExpandedDataLoader, '_sample_paste_plan', return_value=(original, 51)):
                plans.append(loader._sample_paste_plan('OWN_PATIENT')[0])
        self.assertEqual(plans[0]['center'], plans[1]['center'])

    def test_feasible_original_position_keeps_center_even_if_pool_order_would_map_it_elsewhere(self):
        m = self.module; own = self.grid_entry(); loader = self.loader('multi5', own, [0, 3, 90])
        original = dict(entry=own, center=tuple(own['candidate_centers'][3]), scale=.95, normalized_offset=.2)
        with patch.object(self.expanded.ExpandedDataLoader, '_sample_paste_plan', return_value=(original, 987)):
            plan, token = loader._sample_paste_plan('OWN_PATIENT')
        self.assertEqual(plan['center'], tuple(own['candidate_centers'][3]))
        self.assertEqual(plan['joint_selected_primary_index'], 3)
        self.assertFalse(plan['joint_primary_resampled'])

    def witness_and_bank(self, arm='multi5'):
        m = self.module; names = [f'DEBUG_source_{i}.npz' for i in range(642)]
        inventory = {name: [0] for name in names}
        bank = SimpleNamespace(arm=arm, index_path='DEBUG_INDEX_NO_READ', entries_by_case={'DEBUG_CASE': names})
        witness = dict(format=m.FORMAT, joint_policy_format=m.JOINT_POLICY_FORMAT, arm=arm, complete=True,
            index_sha256='DEBUG_SHA', feasible_primary_indices=inventory, validated_positions=642 * 128,
            original_primary_positions_feasible=642, rejection_only_primary_positions=642 * 127,
            all_original128_CP_positions_validated=True, all642_sources_have_joint_feasible_primary=True,
            original_primary_selection_preserved=False, original_gate_and_primary_draw_RNG_schedule_preserved=True,
            historical_trainer_sha256=self.expanded.HISTORICAL_SHA, entries_sha256={name: 'DEBUG_SHA' for name in names},
            copies_per_event=self.expanded.copies_per_event(arm), candidate_count=128,
            blend_border_preprocessed_voxels=0, physical_batch_size=2, epochs=250)
        return witness, bank

    def test_all642_sources_inventory_and_policy_counters_cannot_be_reduced_or_faked(self):
        m = self.module; witness, bank = self.witness_and_bank()
        with patch.object(m, 'sha', return_value='DEBUG_SHA'):
            self.assertEqual(m.validate_feasible_inventory(witness, bank), witness['feasible_primary_indices'])
            for bad in ([], [128], [-1], [True], [0, 0], [2, 1]):
                changed = copy.deepcopy(witness); changed['feasible_primary_indices']['DEBUG_source_0.npz'] = bad
                with self.assertRaises(self.expanded.OnlineCPError): m.validate_feasible_inventory(changed, bank)
            changed = copy.deepcopy(witness); del changed['feasible_primary_indices']['DEBUG_source_0.npz']
            with self.assertRaises(self.expanded.OnlineCPError): m.validate_feasible_inventory(changed, bank)
            for key, bad in [('validated_positions', 82175), ('original_primary_positions_feasible', 643),
                             ('rejection_only_primary_positions', 0), ('original_primary_selection_preserved', True),
                             ('original_gate_and_primary_draw_RNG_schedule_preserved', False)]:
                changed = dict(witness); changed[key] = bad
                with self.assertRaises(self.expanded.OnlineCPError): m.validate_feasible_inventory(changed, bank)

    def test_unique_joint_trainers_preserve_actual_full250_constructor_argument_capture(self):
        m = self.module; actual_constructor = actual_base_constructor_introspection()
        path = self.folder / 'joint_preflight_DEBUG.json'; index = self.folder / 'joint_index_DEBUG.json'
        index.write_text('{}')
        for arm, suffix in [('multi5', 'Multi5Joint'), ('multi7', 'Multi7Joint'), ('multi10', 'Multi10Joint')]:
            cls = getattr(m, 'nnUNetTrainer_250epochs_BasicCP' + suffix)
            self.assertEqual(list(inspect.signature(cls.__init__).parameters),
                             ['self', 'plans', 'configuration', 'fold', 'dataset_json', 'device'])
            self.assertFalse(hasattr(self.expanded, cls.__name__))
            witness, bank = self.witness_and_bank(arm); path.write_text(json.dumps(witness))
            with (patch.object(CPUUnitTrainer, '__init__', actual_constructor), patch.dict(os.environ, {
                    'ONLINE_CP_BANK': str(index), 'ONLINE_CP_SEED': '42', 'BASIC_CP_EXPANDED_ARM': arm,
                    'BASIC_CP_EXPANDED_PREFLIGHT': str(path), 'BASIC_CP_MULTIN_JOINT_PREFLIGHT': str(path)}),
                    patch.object(m, 'sha', return_value='DEBUG_SHA'), patch.object(self.expanded, 'sha', return_value='DEBUG_SHA'),
                    patch.object(m, 'JointBank', return_value=bank)):
                trainer = cls({}, '3d_fullres', 0, {}, torch.device('cpu'))
            self.assertEqual(trainer.num_epochs, 250)
            self.assertEqual(list(trainer.my_init_kwargs), ['plans', 'configuration', 'fold', 'dataset_json', 'device'])

    def test_preflight_records_impossible_positions_but_validates_all_original128(self):
        m = self.module; own = self.grid_entry(); own['candidate_centers'][-1] = [200, 200, 200]
        geometry = self.expanded.CandidateGeometry([own]); name = 'DEBUG_original_source.npz'
        bank = SimpleNamespace(arm='multi10', root=self.folder, entry_names=lambda case: (name,),
            case_entries=lambda case: (own,), geometry_for_case=lambda case: geometry)
        labels = np.ones((1, 220, 220, 220), dtype=np.uint8)
        dataset = SimpleNamespace(load_case=lambda case: (None, labels, None, {}))
        with patch.object(m, 'sha', return_value='DEBUG_UNCHANGED_SHA'):
            result = m._validate_patient(bank, dataset, 'DEBUG_CASE', (128, 128, 128))
        self.assertEqual(result['validated_positions'], 128)
        self.assertEqual(result['feasible_primary_indices'][name], list(range(127)))
        self.assertEqual(result['multiple_paste_positions'], 127)

    def test_preflight_never_removes_a_source_that_has_no_fullN_primary(self):
        m = self.module; own = entry([[30 + i % 8, 30 + i // 8, 30] for i in range(128)], side=20)
        geometry = self.expanded.CandidateGeometry([own]); name = 'DEBUG_unplaceable_source.npz'
        bank = SimpleNamespace(arm='multi5', root=self.folder, entry_names=lambda case: (name,),
            case_entries=lambda case: (own,), geometry_for_case=lambda case: geometry)
        labels = np.ones((1, 64, 64, 64), dtype=np.uint8)
        dataset = SimpleNamespace(load_case=lambda case: (None, labels, None, {}))
        with patch.object(m, 'sha', return_value='DEBUG_UNCHANGED_SHA'):
            with self.assertRaisesRegex(self.expanded.OnlineCPError, 'No full5 joint primary exists'):
                m._validate_patient(bank, dataset, 'DEBUG_CASE', (128, 128, 128))

    def test_training_fullN_geometry_and_paste_counts_use_approved_final_primary(self):
        own = self.grid_entry()
        for arm, count in [('multi5', 5), ('multi7', 7), ('multi10', 10)]:
            loader = self.loader(arm, own, [0]); loader._expanded_case_shape = (160, 160, 160)
            while True:
                plan, _ = loader._sample_paste_plan('OWN_PATIENT')
                if plan is not None: break
            self.assertEqual(plan['center'], tuple(own['candidate_centers'][0]))
            plans = self.expanded.prepare_joint_plan(loader, loader._expanded_case_shape, plan, 'OWN_PATIENT')
            lower = np.asarray(plan['expanded_joint128_lower'])
            image = np.ones((1, 128, 128, 128), dtype=np.float32); labels = np.ones_like(image, dtype=np.int16)
            loader._apply_paste_to_crop(image, labels, lower, plan, 'OWN_PATIENT')
            self.assertEqual(len(plans), count); self.assertEqual(np.count_nonzero(labels == 2), count * 27)
            self.assertTrue(np.all(image[labels != 2] == 1.))


if __name__ == '__main__': unittest.main()
