"""DEBUG contract/real original CPU admission tests; no clinical training claim."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from hiercp_v1x import v24_matched_basic_cp_scoring as matched


def payload_DEBUG():
    """Synthetic schema fixture only; production never fabricates CP inputs."""
    raw = np.stack((np.arange(128), np.ones(128), np.ones(128)), axis=1).astype(np.int32)
    return dict(source_data=np.ones((1, 3, 3, 3), np.float32),
        source_mask=np.ones((3, 3, 3), np.uint8), anchor_offset=np.ones(3, np.int16),
        candidate_centers=raw.copy(), candidate_raw_centers=raw,
        scores=np.arange(128, dtype=np.float32), source_component=np.array([3], np.int16),
        source_diameter_mm=np.array([10.], np.float32))


class PlanPopulation_DEBUG:
    def __init__(self, count=5):
        self.rows = [dict(id='native_P_' + str(i), case_id='case', center=[i, 3, 4],
            donor_case_id='original_independent_donor', donor_component=99,
            target=1, patient_group='original_group', utility=1000, recipient_GT=True)
            for i in range(count)]
    def case(self, case, active):
        if case != 'case' or active != 128:
            raise ValueError('Complete128 native plan required')
        return SimpleNamespace(partition='inner_train', query_rows=copy.deepcopy(self.rows),
            positive_indices=tuple(range(len(self.rows))))


class MatchedBasicScoringContractDEBUG(unittest.TestCase):
    def test_exact_original_payload_admitted(self):
        self.assertEqual(matched.validate_payload(payload_DEBUG()), 3)

    def test_original_payload_not_mutated(self):
        payload = payload_DEBUG()
        before = {key: value.copy() for key, value in payload.items()}
        matched.matched_plan('case', 'entries/case__component_003.npz', payload, PlanPopulation_DEBUG())
        for key in before:
            self.assertEqual(payload[key].dtype, before[key].dtype)
            np.testing.assert_array_equal(payload[key], before[key])

    def test_full_observed_P_and_original_CP_order_without_targets(self):
        payload = payload_DEBUG()
        population = PlanPopulation_DEBUG(count=37)
        plan = matched.matched_plan('case', 'entries/case__component_003.npz', payload, population)
        self.assertEqual(plan.observed_P, 37)
        self.assertEqual(len(plan.record_ids), 165)
        self.assertEqual([row['center'] for row in plan.query_rows[:37]], [row['center'] for row in population.rows])
        self.assertEqual([row['center'] for row in plan.query_rows[37:]], payload['candidate_raw_centers'].tolist())
        for row in plan.query_rows:
            self.assertEqual(set(row), {'id', 'case_id', 'center', 'donor_case_id', 'donor_component'})
            self.assertEqual(row['donor_case_id'], 'case')
            self.assertEqual(row['donor_component'], 3)

    def test_utility_annotation_and_old_donor_never_enter_new_query(self):
        population = PlanPopulation_DEBUG()
        first = matched.matched_plan('case', 'entries/source.npz', payload_DEBUG(), population)
        for row in population.rows:
            row.update(target=-999, utility=-888, recipient_GT=False,
                donor_case_id='unrelated_changed_donor', donor_component=123)
        second = matched.matched_plan('case', 'entries/source.npz', payload_DEBUG(), population)
        self.assertEqual(first, second)

    def test_zero_P_context_keeps_all128_real_CP_coordinates(self):
        plan = matched.matched_plan('case', 'entries/source.npz', payload_DEBUG(), PlanPopulation_DEBUG(0))
        self.assertEqual(plan.observed_P, 0)
        self.assertEqual(len(plan.record_ids), 128)

    def test_source_component_binding_comes_from_original_entry(self):
        payload = payload_DEBUG()
        payload['source_component'][0] = 7
        plan = matched.matched_plan('case', 'entries/source.npz', payload, PlanPopulation_DEBUG())
        self.assertEqual(plan.donor_component, 7)
        self.assertTrue(all(row['donor_component'] == 7 for row in plan.query_rows))

    def test_rejects_partial_reordered_schema_or_raw_target_bank(self):
        for mutation in ('partial', 'extra', 'missing', 'duplicates', 'negative', 'nonfinite', 'diameter', 'anchor', 'empty_mask'):
            payload = payload_DEBUG()
            if mutation == 'partial':
                payload['candidate_raw_centers'] = payload['candidate_raw_centers'][:127]
            elif mutation == 'extra':
                payload['raw_case_reference'] = np.array(['raw.json'])
            elif mutation == 'missing':
                del payload['source_component']
            elif mutation == 'duplicates':
                payload['candidate_raw_centers'][1] = payload['candidate_raw_centers'][0]
            elif mutation == 'negative':
                payload['candidate_raw_centers'][0, 0] = -1
            elif mutation == 'nonfinite':
                payload['source_data'][0, 0, 0, 0] = np.nan
            elif mutation == 'diameter':
                payload['source_diameter_mm'][0] = 21.
            elif mutation == 'anchor':
                payload['anchor_offset'][0] = 5
            else:
                payload['source_mask'].fill(0)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                matched.validate_payload(payload)

    def test_current_original_GTfree_admission_exact_three_comparisons(self):
        _, _, proof = matched.matched_builders([str(i) for i in range(105)])
        self.assertEqual(sum(proof['local']['admission_comparisons_changed'].values()), 1)
        self.assertEqual(sum(proof['upper']['admission_comparisons_changed'].values()), 2)
        self.assertFalse(proof['local']['neural_or_geometry_expressions_changed'])
        self.assertFalse(proof['upper']['neural_or_geometry_expressions_changed'])
        self.assertEqual(sum(proof['upper']['same_patient_recipient_other_lesion_input_removals'].values()), 1)

    def test_self_admission_refuses_independent_or_validation_donor(self):
        local, upper, _ = matched.matched_builders([str(i) for i in range(105)])
        for function in (local, upper):
            with self.assertRaisesRegex(ValueError, 'Only original outer-training'):
                function(SimpleNamespace(case_id='1'), SimpleNamespace(paths=SimpleNamespace(case_id='2')))
            with self.assertRaisesRegex(ValueError, 'Only original outer-training'):
                function(SimpleNamespace(case_id='heldout'), SimpleNamespace(paths=SimpleNamespace(case_id='heldout')))

    def test_changed_admission_expression_fails_without_bypassing(self):
        with self.assertRaisesRegex(ValueError, 'admission expressions changed'):
            matched._clone_admission(payload_DEBUG, {'recipient.case_id == donor_case.paths.case_id': 'False'})

    def test_incomplete_training_cohort_refused(self):
        with self.assertRaisesRegex(ValueError, 'full105'):
            matched.matched_builders([str(i) for i in range(104)])

    def test_complete_642_source_schedule_parallel_and_disjoint(self):
        # Unequal real-contract cardinalities; names are DEBUG identities only.
        entries = {f'case_{case:03d}': [f'entries/{case}_{i}.npz' for i in range(7 + (case < 75))]
            for case in range(81)}
        self.assertEqual(sum(map(len, entries.values())), 642)
        batches = matched.source_batches(entries, list(entries))
        flattened = [item for batch in batches for item in batch]
        self.assertEqual(set(flattened), {(case, name) for case, names in entries.items() for name in names})
        self.assertEqual(len(flattened), 642)
        self.assertTrue(all(2 <= len(batch) <= 4 for batch in batches))
        self.assertTrue(all(len(set(case for case, _ in batch)) == len(batch) for batch in batches))

    def test_heavily_unequal_sources_keep_every_entry(self):
        entries = {f'case_{i}': [f'entries/{i}_{j}.npz' for j in range(count)]
            for i, count in enumerate((37, 20, 15, 11, 6, 5, 4, 3))}
        batches = matched.source_batches(entries, list(entries))
        self.assertEqual(sum(map(len, batches)), sum(map(len, entries.values())))
        self.assertTrue(all(2 <= len(batch) <= 4 for batch in batches))

    def test_unmeasured_patient_capacity_and_impossible_parallelism_refused(self):
        with self.assertRaisesRegex(ValueError, 'measured physical'):
            matched.source_batches({'a': ['one']}, ['a'], physical=1)
        with self.assertRaisesRegex(ValueError, 'without duplication'):
            matched.source_batches({'a': ['one', 'two']}, ['a'])


class ActualOriginalSelfCPGeometryDEBUG(unittest.TestCase):
    """Real archived graph constructors on clearly synthetic CPU UNIT volumes."""
    @classmethod
    def setUpClass(cls):
        from tests.test_v24_gt_blind import RecipientGTBlindDebug
        RecipientGTBlindDebug.setUpClass()
        cls.fixture = RecipientGTBlindDebug
        local, upper, _ = matched.matched_builders(
            [cls.fixture.donor.paths.case_id] + ['UNIT_outer_' + str(i) for i in range(104)])
        cls.local, cls.upper = staticmethod(local), staticmethod(upper)
    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def test_other_same_patient_lesion_annotations_do_not_change_actual_L0_L1_L2_inputs(self):
        from hiercp_v1x.v24_inputs import prepare_donor, materialize_pair, tensor_digest
        fixture = self.fixture
        rows = [dict(id='UNIT_self_CP_' + str(i), case_id=fixture.donor.paths.case_id,
            donor_case_id=fixture.donor.paths.case_id, donor_component=fixture.source.component_id,
            center=list(center)) for i, center in enumerate(((13, 14, 13), (18, 17, 16), (10, 20, 18)))]
        references = None
        for other_lesion in (None, (18, 18, 18), (20, 7, 20)):
            donor = copy.deepcopy(fixture.donor)
            if other_lesion is not None:
                a, b, c = other_lesion
                donor.label[a:a+3, b:b+3, c:c+3] = 2
            prepared = prepare_donor(donor, fixture.source, config=fixture.config, seed=42, ct_clip=fixture.clip)
            graph, prototype, audit = self.upper(fixture.donor_context, donor, fixture.source,
                fixture.donor_regions, fixture.donor_regions, rows, fixture.bank,
                config=fixture.config, ct_clip=fixture.clip,
                training_case_ids=(fixture.donor.paths.case_id,))
            records = [self.local(fixture.donor_context, donor, fixture.source, prepared, row,
                config=fixture.config, seed=42, ct_clip=fixture.clip,
                scope_contract=fixture.scope['contract_sha256']) for row in rows]
            views = [materialize_pair(record, epoch=29) for record in records]
            actual = (tensor_digest((graph.to_dict(), prototype.to_dict())),
                tuple(record['tensor_sha256'] for record in records),
                tuple(tensor_digest((tuple(graph.to_dict() for graph in value[0]), value[1], value[2]))
                    for value in views))
            if references is None:
                references = actual
            self.assertEqual(actual, references)
            self.assertFalse(audit['recipient_GT_used_in_forward'])
            self.assertFalse(audit['same_patient_donor_other_lesion_occupancy_used'])
            self.assertNotIn('lesion', graph.node_types)

    def test_selected_original_source_cannot_be_replaced_or_erased(self):
        from hiercp_v1x.v24_inputs import prepare_donor
        fixture = self.fixture
        donor = copy.deepcopy(fixture.donor)
        donor.label[fixture.source.full_mask] = 1
        with self.assertRaisesRegex(ValueError, 'actual annotated full component'):
            prepare_donor(donor, fixture.source, config=fixture.config, seed=42, ct_clip=fixture.clip)

    def test_streamed_parallel_local_provider_preserves_exact_original_views_and_collation(self):
        from hiercp_v1x import v24_input_runtime
        from hiercp_v1x.v24_inputs import materialize_pair, collate, tensor_digest
        from unittest.mock import patch
        fixture = self.fixture
        rows = [dict(row, case_id=fixture.donor.paths.case_id, donor_case_id=fixture.donor.paths.case_id)
            for row in fixture.rows]
        records = [self.local(fixture.donor_context, fixture.donor, fixture.source,
            fixture.prepared, row, config=fixture.config, seed=42, ct_clip=fixture.clip,
            scope_contract=fixture.scope['contract_sha256']) for row in rows]
        lookup = {row['id']: record for row, record in zip(rows, records)}
        provider = matched.MatchedInputProvider([SimpleNamespace(query_rows=rows)],
            lambda row: lookup[row['id']], SimpleNamespace(check=lambda: None), workers=4)
        def signature(batch):
            return tensor_digest((batch.graph.to_dict(), batch.source_patches, batch.target_patches,
                batch.source_index, batch.graph_observation_index, batch.indices))
        try:
            expected = collate([(materialize_pair(record, epoch=29), index)
                for index, record in enumerate(records)])
            # CPU UNIT has no CUDA pinned allocator; production pin policy stays
            # true and is never changed by this separate test patch.
            with patch.object(v24_input_runtime, '_PIN_OUTPUTS', False):
                actual = provider.get([0, 1, 2], epoch=29)
            self.assertEqual(signature(actual), signature(expected))
            self.assertEqual(provider.profile_rows['observations'], 3)
            self.assertEqual(v24_input_runtime.profile()['source_live_chunks'], 0)
            with self.assertRaisesRegex(ValueError, 'fixed evaluation view29'):
                provider.get([0], epoch=1)
        finally:
            provider.close()


if __name__ == '__main__':
    unittest.main()
