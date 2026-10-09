"""CPU UNIT tests only: frozen U admission, complete patient ordering and plans."""
import copy
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import HeteroData

from hiercp_v1x.v23_data import V23Population
from hiercp_v1x.v23_targets import (
    admit_targets, patient_epoch_order, target_selection_manifest,
    validate_admission, validate_target_selection,
)
from tests.test_v23_data import unit_inventory
from hiercp_v1x.historical_evaluation import sha
from hiercp_v1x.v23_geometry import V23UpperGeometryCache


class TargetAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.scores = [float(i) for i in range(128)]

    def test_top_adds_only_unseen_U_preserving_every_admitted_U(self):
        result = admit_targets('hard_score_top', list(range(7)), self.scores,
                               case_id='unit_train', epoch=8, model_sha256='a' * 64)
        self.assertEqual(result['indices'], list(range(7)) + list(range(121, 128)))
        self.assertEqual(result['receipt']['added_indices'], list(range(127, 120, -1)))
        self.assertFalse(result['receipt']['P_or_validation_labels_used'])
        self.assertEqual(validate_admission(result['receipt']), result['receipt'])
        changed = copy.deepcopy(result['receipt'])
        changed['indices'][0] = 100
        with self.assertRaises(ValueError):
            validate_admission(changed)

    def test_strata_seven_uses_three_high_two_middle_two_low(self):
        result = admit_targets('score_stratified_mix', list(range(7)), self.scores)
        receipt = result['receipt']
        self.assertEqual(receipt['stratum_quotas'], [3, 2, 2])
        self.assertEqual(receipt['added_indices'], [127, 126, 125, 86, 85, 46, 45])
        self.assertEqual([len(s) for s in receipt['score_strata']], [41, 40, 40])

    def test_ties_use_native_bank_position_and_no_random_redraw(self):
        scores = [0.] * 128
        first = admit_targets('hard_score_top', list(range(7)), scores)
        second = admit_targets('hard_score_top', list(range(7)), scores, seed=999)
        self.assertEqual(first['indices'], list(range(14)))
        self.assertEqual(first['indices'], second['indices'])
        self.assertEqual(admit_targets('score_stratified_mix', list(range(7)), scores)
                         ['receipt']['added_indices'], [7, 8, 9, 48, 49, 88, 89])

    def test_every_policy_reaches_full128_without_discarding_U(self):
        for policy in ('native_prefix', 'hard_score_top', 'score_stratified_mix'):
            active = list(range(7))
            while len(active) < 128:
                result = admit_targets(policy, active, self.scores)
                self.assertTrue(set(active) <= set(result['indices']))
                active = result['indices']
            self.assertEqual(active, list(range(128)))
            self.assertEqual(admit_targets(policy, active, self.scores)['receipt']['actual_additions'], 0)

    def test_tail_quota_and_capacity_preserve_all_remaining_U(self):
        for remaining in range(1, 9):
            active = list(range(128 - remaining))
            result = admit_targets('score_stratified_mix', active, self.scores)
            self.assertEqual(sum(result['receipt']['stratum_quotas']), min(7, remaining))
            self.assertEqual(len(result['indices']), min(128, len(active) + 7))
            if remaining <= 7:
                self.assertEqual(result['indices'], list(range(128)))

    def test_invalid_or_P_containing_scores_fail_explicitly(self):
        for bad in (self.scores[:127], self.scores + [1.], [math.nan] * 128,
                    [math.inf] * 128, [True] * 128):
            with self.assertRaises(ValueError):
                admit_targets('hard_score_top', list(range(7)), bad)
        for bad in ([0, 0], [True], [-1], [128], []):
            with self.assertRaises(ValueError):
                admit_targets('hard_score_top', bad, self.scores)


class FrozenPlanTests(unittest.TestCase):
    def setUp(self):
        self.population = V23Population(unit_inventory(), debug=True)
        self.cases = self.population.partition_cases('inner_train', ranking_only=True) + self.population.partition_cases('inner_val')

    def test_explicit_selection_retains_all_P_query_order_and_fixed_donor(self):
        selected = [127, 1, 2, 3, 4, 5, 6]
        plan = self.population.case('train_a', 7, active_u_indices=selected)
        prefix = self.population.case('train_a', 7)
        self.assertEqual(plan.observed_P, prefix.observed_P)
        self.assertEqual(plan.active_u_indices, (1, 2, 3, 4, 5, 6, 127))
        self.assertEqual(plan.donor_case_id, prefix.donor_case_id)
        self.assertEqual(set(plan.unobserved_bank_positions), set(selected))
        self.assertEqual(self.population.indices_for(plan, self.population.rows), plan.inventory_indices)
        self.assertTrue(all('target' not in row for row in plan.query_rows))
        selected[0] = 100
        self.assertEqual(plan.active_u_indices[-1], 127)

    def test_full128_and_explicit_prefix_match_legacy_manifest_exactly(self):
        for count in (7, 128):
            original = self.population.case('train_a', count)
            explicit = self.population.case('train_a', count, active_u_indices=list(reversed(range(count))))
            self.assertEqual(original.manifest(), explicit.manifest())
            self.assertEqual(original.query_rows, explicit.query_rows)

    def test_complete_stage_requires_all_val_including_zero_P(self):
        selected = {case: list(range(7)) for case in self.cases}
        manifest = target_selection_manifest(self.population, 7, selected)
        self.assertTrue(manifest['prefix_equivalent'])
        self.assertIn('val_zero', manifest['selections'])
        selected['train_a'] = [0, 1, 2, 3, 4, 5, 127]
        changed = target_selection_manifest(self.population, 7, selected)
        self.assertNotEqual(changed['sha256'], manifest['sha256'])
        self.assertFalse(changed['prefix_equivalent'])
        del selected['val_zero']
        with self.assertRaises(ValueError):
            validate_target_selection(self.population, 7, selected)


class PatientOrderTests(unittest.TestCase):
    def setUp(self):
        self.cases = ['c' + str(i) for i in range(8)]
        self.random = list(reversed(self.cases))
        self.rows = [dict(case_id=c, observed_P=1, per_P=dict(pair_loss=float(i), mrr=0.1))
                     for i, c in enumerate(self.cases)]

    def test_native_keeps_complete_seeded_permutation(self):
        self.assertEqual(patient_epoch_order(self.cases, None, 'native_prefix', self.random)['order'], self.random)

    def test_hard_sorts_previous_train_pair_loss_ties_by_random_order(self):
        result = patient_epoch_order(self.cases, self.rows, 'hard_score_top', self.cases)
        self.assertEqual(result['order'], self.random)
        self.rows[0]['per_P']['pair_loss'] = 7.
        order = patient_epoch_order(self.cases, self.rows, 'hard_score_top', self.cases)['order']
        self.assertEqual(order[:2], ['c0', 'c7'])
        self.assertFalse(result['receipt']['validation_used'])

    def test_stratified_order_mixes_terciles_and_keeps_each_patient_once(self):
        result = patient_epoch_order(self.cases, self.rows, 'score_stratified_mix', self.cases)
        self.assertEqual(result['order'], ['c7', 'c4', 'c1', 'c6', 'c3', 'c0', 'c5', 'c2'])
        self.assertEqual(set(result['order']), set(self.cases))
        self.assertEqual(len(result['order']), len(self.cases))
        mutated = copy.deepcopy(self.rows)
        for row in mutated:
            row['per_P']['mrr'] = 0.99
        self.assertEqual(patient_epoch_order(self.cases, mutated, 'score_stratified_mix', self.cases)['order'], result['order'])

    def test_missing_invalid_or_validation_rows_cannot_order_training(self):
        for rows in (self.rows[:-1], [self.rows[0]] * 8):
            with self.assertRaises(ValueError):
                patient_epoch_order(self.cases, rows, 'hard_score_top', self.random)
        for key, value in (('partition', 'inner_val'), ('observed_P', 0)):
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.assertRaises(ValueError):
                patient_epoch_order(self.cases, rows, 'score_stratified_mix', self.random)
        rows = copy.deepcopy(self.rows)
        rows[0]['per_P']['pair_loss'] = math.nan
        with self.assertRaises(ValueError):
            patient_epoch_order(self.cases, rows, 'hard_score_top', self.random)

    def test_full65_and_uneven_seven_patient_orders_have_exact_coverage(self):
        for count in (7, 65):
            cases = ['patient_' + str(i) for i in range(count)]
            permutation = list(reversed(cases))
            rows = [dict(case_id=case, observed_P=1, per_P=dict(pair_loss=float(i)))
                    for i, case in enumerate(cases)]
            for policy in ('native_prefix', 'hard_score_top', 'score_stratified_mix'):
                result = patient_epoch_order(cases, rows, policy, permutation)
                self.assertEqual(len(result['order']), count)
                self.assertEqual(set(result['order']), set(cases))
                self.assertEqual(result['receipt']['omissions'], 0)
                self.assertEqual(result['receipt']['duplicates'], 0)
                self.assertEqual(result['receipt']['training_P_U_labels_used'], policy != 'native_prefix')
                self.assertFalse(result['receipt']['validation_labels_or_metrics_used'])


def target_unit_upper(bundle, provider, rows, **kwargs):
    """Mechanical CPU UNIT tensor only, never a scientific geometry result."""
    graph, prototype = HeteroData(), HeteroData()
    graph['candidate'].raw_x = torch.tensor([r['center'] for r in rows], dtype=torch.float32)
    prototype['prototype'].raw_x = torch.tensor([[1.]])
    return graph, prototype, dict(candidate_count=len(rows), P_U_labels_in_forward=False,
                                 CPU_UNIT_fixture=True)


class FrozenGeometryTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / 'outputs/v23_CPU_UNIT/tmp'
        parent.mkdir(parents=True, exist_ok=True)
        self.tmp = TemporaryDirectory(dir=parent)
        self.root = Path(self.tmp.name)
        self.baseline, self.source = self.root / 'baseline', self.root / 'source'
        self.baseline.mkdir()
        self.source.mkdir()
        meta = unit_inventory()
        for raw in meta['raw_records']:
            case = raw['case_id']
            for name in ('image', 'label'):
                path = self.root / (case + '.' + name)
                path.write_bytes(b'CPU UNIT input only')
                raw[name], raw[name + '_sha256'] = str(path), sha(path)
            region = self.baseline / 'shared/regions' / case
            region.mkdir(parents=True)
            (region / 'metadata.json').write_text('{}')
        self.population = V23Population(meta, debug=True)
        self.cases = self.population.partition_cases('inner_train', ranking_only=True) + self.population.partition_cases('inner_val')
        self.bundle = SimpleNamespace(baseline=self.baseline, source=self.source,
            config=dict(graph=dict(num_regions=24), ct_clip=[-200., 250.]),
            prototype_bank=SimpleNamespace(fingerprint=lambda: 'CPU_UNIT_BANK',
                training_case_ids=list(self.population.partition_cases('inner_train'))))
        self.provider = SimpleNamespace(ds=SimpleNamespace(meta=self.population.meta))

    def tearDown(self):
        self.tmp.cleanup()

    def cache(self):
        with patch('hiercp_v1x.v23_geometry.adapted_builders',
                   return_value=(target_unit_upper, dict(CPU_UNIT_fixture=True))):
            return V23UpperGeometryCache(self.bundle, self.population, self.root / 'new',
                                         workers=2, rss_bytes=8 * 2**30)

    def selection(self, last=127):
        selected = {case: list(range(7)) for case in self.cases}
        selected['train_a'] = [0, 1, 2, 3, 4, 5, last]
        return selected

    def test_adaptive_same_count_gets_distinct_cache_preserving_prefix_bytes(self):
        cache = self.cache()
        cache.prepare(7)
        old = {str(path): sha(path) for path in (cache.output / 'active_U_7').rglob('*') if path.is_file()}
        first, second = self.selection(), self.selection(126)
        cache.prepare(7, target_selection=first)
        cache.prepare(7, target_selection=second)
        self.assertEqual(old, {path: sha(path) for path in old})
        self.assertEqual(len(list((cache.output / 'target_selections').iterdir())), 2)
        for selected, last in ((first, 127), (second, 126)):
            plan = self.population.case('train_a', 7, active_u_indices=selected['train_a'])
            graph, _, audit = cache.get(plan, self.provider)
            self.assertIn(float(last), graph['candidate'].raw_x[:, 0].tolist())
            self.assertEqual(graph['candidate'].raw_x.shape[0], plan.observed_P + 7)
            self.assertFalse(audit['P_as_negative'])
        self.assertNotEqual(cache._binding(self.population.case('train_a', 7,
            active_u_indices=first['train_a'])), cache._binding(self.population.case('train_a', 7)))

    def test_explicit_prefix_and_full_bank_use_original_cache_namespace(self):
        cache = self.cache()
        for count in (7, 128):
            original = cache.prepare(count)
            selected = {case: list(reversed(range(count))) for case in self.cases}
            self.assertEqual(cache.prepare(count, target_selection=selected), original)
            self.assertTrue((cache.output / ('active_U_' + str(count)) / 'index.json').is_file())
        self.assertFalse((cache.output / 'target_selections').exists())

    def test_readonly_rank_admits_exact_frozen_map_without_build(self):
        root, selected = self.cache(), self.selection()
        complete = root.prepare(7, target_selection=selected)
        rank = self.cache()
        with patch.object(rank, '_upper', side_effect=AssertionError('UNIT no rebuild allowed')):
            self.assertEqual(rank.admit(7, target_selection=selected), complete)
            plan = self.population.case('val_zero', 7, active_u_indices=selected['val_zero'])
            self.assertEqual(rank.get(plan, self.provider)[0]['candidate'].raw_x.shape[0], 7)
        with self.assertRaises(ValueError):
            rank.admit(7, target_selection=self.selection(125))
        with self.assertRaises(ValueError):
            rank.get(self.population.case('train_a', 7, active_u_indices=[0, 1, 2, 3, 4, 5, 124]), self.provider)


if __name__ == '__main__':
    unittest.main()
