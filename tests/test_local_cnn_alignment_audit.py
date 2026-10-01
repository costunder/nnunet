"""Schedule arithmetic fixtures; no medical/model accuracy claims."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from l0_regions.donor_learning import groups
from tools.local_cnn_alignment_audit import audit_alignment_schedule


def dataset(specification):
    rows = []
    for case, patient, positive, unobserved in specification:
        for target, count in ((1, positive), (0, unobserved)):
            for j in range(count):
                rows.append(dict(id=f'{case}-{target}-{j}', case_id=case,
                    patient_group=patient, donor_case_id=f'donor-{case}',
                    donor_component=1, donor_group=f'donor-group-{case}',
                    target=target, bounds=dict(edges=j)))
    return SimpleNamespace(rows=rows)


class Checks(unittest.TestCase):
    def test_variable_positive_counts_expose_alignment_weighting(self):
        ds = dataset([('a', 'p1', 1, 128), ('b', 'p2', 20, 128)])
        r = audit_alignment_schedule(ds, 32, 42, 22)
        a, b = r['patient_groups']
        self.assertEqual((a['tiles_per_patient_group'], b['tiles_per_patient_group']), (5, 13))
        self.assertAlmostEqual(a['current_alignment_step_mean_weight'], 5 / 18)
        self.assertAlmostEqual(b['current_alignment_step_mean_weight'], 13 / 18)
        self.assertAlmostEqual(a['current_to_equal_group_weight_ratio'], 5 / 9)
        self.assertAlmostEqual(b['current_to_equal_group_weight_ratio'], 13 / 9)
        self.assertFalse(r['production_loss_changed'])
        self.assertFalse(r['alignment_contract']['hypothetical_correction_applied'])
        self.assertFalse(r['tile_count_distribution']['all_groups_equal'])

    def test_equal_tile_counts_correction_is_one(self):
        ds = dataset([('a', 'p1', 3, 17), ('b', 'p2', 3, 17)])
        r = audit_alignment_schedule(ds, 8, 123, 4)
        self.assertTrue(r['tile_count_distribution']['all_groups_equal'])
        for g in r['patient_groups']:
            self.assertEqual(g['hypothetical_group_balanced_per_tile_multiplier'], 1)
            self.assertEqual(g['current_to_equal_group_weight_ratio'], 1)

    def test_multiple_cases_are_aggregated_by_patient_group(self):
        ds = dataset([('a', 'same', 1, 9), ('b', 'same', 6, 10), ('c', 'other', 2, 10)])
        r = audit_alignment_schedule(ds, 8, 42, 0)
        self.assertEqual(r['patient_group_count'], 2)
        same = next(g for g in r['patient_groups'] if g['patient_group'] == 'same')
        cases = [c for c in r['cases'] if c['patient_group'] == 'same']
        self.assertEqual(same['case_ids'], ['a', 'b'])
        self.assertEqual(same['tiles_per_patient_group'], sum(c['tiles'] for c in cases))
        self.assertEqual(same['positive_count'], 7)

    def test_exact_pair_coverage_and_balanced_ce_despite_repetitions(self):
        ds = dataset([('a', 'p1', 9, 13), ('b', 'p2', 1, 17)])
        r = audit_alignment_schedule(ds, 8, 42, 3)
        n = r['normalization']
        self.assertEqual(n['ranking_pairs_unique'], 9 * 13 + 17)
        self.assertEqual(n['ranking_pairs_scheduled'], n['ranking_pairs_unique'])
        self.assertEqual(n['ranking_pairs_repeated'], 0)
        self.assertEqual(n['ranking_pairs_missing'], 0)
        self.assertGreater(r['query_presentations'], r['unique_observations'])
        for target in ('0', '1'):
            self.assertAlmostEqual(n['observation_ce_step_mean_class_mass'][target], .5)
        self.assertTrue(n['observation_ce_global_balanced_mean_verified'])

    def test_single_class_cases_preserved_and_alignment_counted(self):
        ds = dataset([('a', 'p1', 1, 9), ('b', 'p2', 0, 11), ('c', 'p3', 10, 0)])
        r = audit_alignment_schedule(ds, 8, 42, 0)
        self.assertEqual(r['unique_observations'], len(ds.rows))
        self.assertEqual(r['case_count'], 3)
        for c in r['cases'][1:]:
            self.assertEqual(c['expected_ranking_pairs'], 0)
            self.assertEqual(c['tiles'], 2)
            self.assertGreater(c['current_alignment_step_mean_weight'], 0)

    def test_seed_and_epoch_change_order_not_normalization(self):
        ds = dataset([('a', 'p1', 9, 13), ('b', 'p2', 1, 17), ('c', 'p3', 5, 22)])
        a = audit_alignment_schedule(ds, 8, 42, 0)
        b = audit_alignment_schedule(ds, 8, 42, 22)
        self.assertNotEqual(list(groups(ds, 8, 42, 0)), list(groups(ds, 8, 42, 22)))
        for key in ('normalization', 'tile_count_distribution', 'actual_batch_size_counts'):
            self.assertEqual(a[key], b[key])
        for x, y in zip(a['patient_groups'], b['patient_groups']):
            self.assertEqual(x['tiles_per_patient_group'], y['tiles_per_patient_group'])

    def test_invalid_binding_not_ignored(self):
        ds = dataset([('a', 'p1', 1, 5)])
        ds.rows[-1]['donor_case_id'] = 'different-donor'
        with self.assertRaisesRegex(ValueError, 'different donors'):
            audit_alignment_schedule(ds, 4, 42, 0)

    def test_duplicated_pair_tile_is_rejected(self):
        ds = dataset([('a', 'p1', 2, 2)])
        # Maintain both tile count and per-observation presentation counts,
        # while duplicating one pair and omitting another.
        bad = [[0, 2], [0, 2], [1, 3], [1, 3]]
        with patch('tools.local_cnn_alignment_audit.groups', return_value=iter(bad)):
            with self.assertRaisesRegex(ValueError, 'multiplicity/coverage'):
                audit_alignment_schedule(ds, 2, 42, 0)

    def test_invalid_resources_or_epoch_are_explicit_errors(self):
        ds = dataset([('a', 'p1', 1, 2)])
        for batch, seed, epoch in ((1, 42, 0), (4, '42', 0), (4, 42, -1)):
            with self.assertRaises(ValueError):
                audit_alignment_schedule(ds, batch, seed, epoch)


if __name__ == '__main__':
    unittest.main()
