"""UNIT fixtures only: metadata/ranking contracts, never CT quality evidence."""
import copy
import math
import unittest
import json
from pathlib import Path

from hiercp_v1x.transition_evaluation import (
    FORMAT, SCORING_FORMAT, PreparedL0Batch, evaluate_scores, run_scoring,
    validate_cohort, validate_native_assignments,
)


def unit_inventory(*, production=False):
    train = [f'train_{i}' for i in range(84 if production else 2)]
    val = [f'val_{i}' for i in range(21 if production else 3)]
    excluded = [f'outer_{i}' for i in range(26 if production else 1)]
    outer = train + val
    identity = {case: dict(patient_group='case:'+case) for case in outer + excluded}
    raw = []; rows = []
    for case in outer:
        # One zero-P held-out case stays in the evaluated population.
        count = 0 if case == val[-1] else 2
        positives = [dict(component=i+1, center=[i, 0, 0]) for i in range(count)]
        centers = [[i, 10, 0] for i in range(128)]
        raw.append(dict(case_id=case, positives=positives, comparison=dict(centers=centers),
                        image_sha256='a'*64, label_sha256='b'*64))
        observations = [(p['center'], 1, p['component']) for p in positives]
        observations += [(xyz, 0, None) for xyz in centers]
        for index, (center, target, component) in enumerate(observations):
            rows.append(dict(id=f'{case}:{index}', case_id=case, patient_group='case:'+case,
                             component=component, center=center, target=target))
    meta = dict(format='native_local_cnn_inventory_v1', complete=True, debug=not production,
                learning_policy='same_donor_live_v1', config=dict(seed=42),
                split=dict(inner_train=train, inner_val=val, outer_train=outer, outer_val=excluded),
                identities=dict(cases=identity), raw_records=raw,
                donor_pool=[dict(case_id=c, component_id=1) for c in train[:2]], records=rows,
                UNIT_synthetic_metadata_only=True)
    from l0_regions.donor_data import assignment
    meta['records'] = assignment(meta, 42)
    return meta


def scores(meta, positive=1., negative=0.):
    allowed = set(meta['split']['inner_val'])
    return {row['id']: positive if row['target'] else negative
            for row in meta['records'] if row['case_id'] in allowed}


def contract(ids, upper_invocations):
    return dict(format=SCORING_FORMAT, scored_record_ids=list(ids), l0_only_chunking=True,
                upper_chunking=False, upper_execution='single_joint_case',
                query_GT_in_forward=False, upper_invocations=upper_invocations)


class Whole128MetadataUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = unit_inventory()

    def test_full_scores_separate_case_hit_and_micro_recall(self):
        result = evaluate_scores(self.meta, scores(self.meta), debug=True)
        self.assertEqual(result['format'], FORMAT)
        self.assertEqual(result['metrics']['case_first_P_mrr'], 1.)
        self.assertEqual(result['metrics']['case_hit_at_1'], 1.)
        self.assertEqual(result['metrics']['observed_micro_recall_at_1'], .5)
        self.assertEqual(result['metrics']['observed_micro_recall_at_5'], 1.)
        self.assertEqual(result['metrics']['P_U_pair_win_rate'], 1.)
        self.assertAlmostEqual(result['metrics']['P_U_softplus_loss'], math.log1p(math.exp(-1)))
        self.assertEqual(result['denominators']['observed_P'], 4)
        self.assertEqual(result['denominators']['unobserved_U'], 384)
        self.assertEqual(result['denominators']['P_U_pairs'], 512)
        self.assertEqual(result['denominators']['zero_P_cases'], 1)
        self.assertEqual(result['denominators']['rank_evaluable_cases'], 2)
        self.assertFalse(result['quality_verified'])
        self.assertFalse(result['execution_contract_bound'])

    def test_ties_have_gt_independent_order_and_real_zero_std(self):
        result = evaluate_scores(self.meta, scores(self.meta, 0., 0.), debug=True)
        self.assertEqual(result['metrics']['score_std'], 0.)
        self.assertEqual(result['metrics']['P_U_pair_tie_rate'], 1.)
        self.assertEqual(result['metrics']['P_U_pair_win_rate'], 0.)
        self.assertAlmostEqual(result['metrics']['P_U_softplus_loss'], math.log(2))
        from tools.v22_candidate_order import candidate_order
        for case in result['cases']:
            expected = candidate_order([r['score'] for r in case['case_scores']],
                                       [r['candidate_key'] for r in case['case_scores']]).tolist()
            self.assertEqual(case['ordered_record_ids'], [case['case_scores'][i]['record_id'] for i in expected])

    def test_row_permutation_keeps_metrics(self):
        changed = copy.deepcopy(self.meta)
        changed['records'].reverse()
        self.assertEqual(evaluate_scores(self.meta, scores(self.meta), debug=True)['metrics'],
                         evaluate_scores(changed, scores(changed), debug=True)['metrics'])

    def test_production_defaults_all21_with_all128U(self):
        meta = unit_inventory(production=True)
        result = evaluate_scores(meta, scores(meta))
        self.assertEqual(result['denominators']['cases'], 21)
        self.assertEqual(result['denominators']['unobserved_U'], 2688)
        self.assertTrue(result['production_full_inner_val'])
        with self.assertRaises(ValueError):
            validate_cohort(meta, case_ids=meta['split']['inner_val'][:8])

    def test_small_fixture_needs_explicit_debug(self):
        with self.assertRaises(ValueError):
            evaluate_scores(self.meta, scores(self.meta))
        subset = self.meta['split']['inner_val'][:1]
        selected = {k:v for k,v in scores(self.meta).items() if k.split(':')[0] in subset}
        self.assertTrue(evaluate_scores(self.meta, selected, case_ids=subset, debug=True)['debug'])

    def test_missing_extra_and_nonfinite_scores_rejected(self):
        original = scores(self.meta)
        for kind in ('missing', 'extra', 'nan', 'inf', 'boolean'):
            changed = dict(original)
            identity = next(iter(changed))
            if kind == 'missing': del changed[identity]
            elif kind == 'extra': changed['unknown'] = 0.
            else: changed[identity] = {'nan':float('nan'), 'inf':float('inf'), 'boolean':True}[kind]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                evaluate_scores(self.meta, changed, debug=True)

    def test_missing_U_and_missing_P_and_extra_record_rejected(self):
        for target in (0, 1):
            changed = copy.deepcopy(self.meta)
            index = next(i for i,r in enumerate(changed['records']) if r['target'] == target)
            del changed['records'][index]
            with self.subTest(target=target), self.assertRaises(ValueError):
                validate_cohort(changed, debug=True)
        changed = copy.deepcopy(self.meta)
        changed['records'].append(dict(changed['records'][0], id='extra'))
        with self.assertRaises(ValueError): validate_cohort(changed, debug=True)

    def test_wrong_observed_component_and_center_rejected(self):
        for field, value in (('component', 123), ('center', [999,0,0]), ('target', True)):
            changed = copy.deepcopy(self.meta)
            changed['records'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_cohort(changed, debug=True)

    def test_wrong_U_center_and_raw_sha_rejected(self):
        changed = copy.deepcopy(self.meta)
        next(r for r in changed['records'] if not r['target'])['center'] = [999,10,0]
        with self.assertRaises(ValueError): validate_cohort(changed, debug=True)
        changed = copy.deepcopy(self.meta)
        changed['raw_records'][0]['label_sha256'] = None
        with self.assertRaises(ValueError): validate_cohort(changed, debug=True)

    def test_independent_donor_and_patient_leakage_rejected(self):
        changed = copy.deepcopy(self.meta)
        changed['records'][0]['donor_group'] = changed['records'][0]['patient_group']
        with self.assertRaises(ValueError): validate_cohort(changed, debug=True)
        changed = copy.deepcopy(self.meta)
        changed['identities']['cases']['val_0']['patient_group'] = 'case:train_0'
        with self.assertRaises(ValueError): validate_cohort(changed, debug=True)

    def test_actual_native_assignment_verified_not_just_any_training_donor(self):
        self.assertEqual(validate_native_assignments(self.meta)['records'], len(self.meta['records']))
        changed = copy.deepcopy(self.meta)
        for row in changed['records']:
            if row['case_id'] == 'val_0':
                other = 'train_1' if row['donor_case_id'] == 'train_0' else 'train_0'
                row.update(donor_case_id=other, donor_group='case:'+other)
        with self.assertRaises(ValueError): validate_cohort(changed, debug=True)
        with self.assertRaises(ValueError): validate_native_assignments(changed)

    def test_external_execution_contract_rejects_eight_upper_chunks(self):
        ids = [r['id'] for r in validate_cohort(self.meta, debug=True)['rows']]
        valid = contract(ids, 3)
        self.assertTrue(evaluate_scores(self.meta, scores(self.meta), debug=True, scoring_contract=valid)['execution_contract_bound'])
        for field, value in (('upper_chunking', True), ('l0_only_chunking', False),
                             ('upper_invocations', 8), ('query_GT_in_forward', True)):
            invalid = dict(valid, **{field:value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate_scores(self.meta, scores(self.meta), debug=True, scoring_contract=invalid)

    def test_new_crossed_ledger_contains_every29_factor_once_and_declares_coupling(self):
        from hiercp_v1x.transition_inventory import FACTOR_NAMES
        config = json.loads((Path(__file__).resolve().parents[1]/'config/v17_crossed_training.json').read_text())
        names = [row['name'] for row in config['factor_ledger']]
        self.assertEqual(len(names), 29)
        self.assertEqual(set(names), set(FACTOR_NAMES))
        grouped = [name for group in config['groups'].values() for name in group]
        self.assertEqual(len(grouped), 29)
        self.assertEqual(set(grouped), set(names))
        self.assertEqual([len(v) for v in config['groups'].values()], [3,10,16])
        self.assertFalse(config['coverage']['strict_independent_factor_crossing'])
        self.assertIn('comparison_corruption', config['coverage']['coupled_exceptions'])
        self.assertTrue(config['arms']['C']['native_CNN_checkpointing'])
        self.assertFalse(config['arms']['D']['checkpoint_dense_encoder'])
        self.assertFalse(config['arms']['D']['checkpoint_local_blocks'])


class Whole128ScoringUNIT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = unit_inventory()

    def test_l0_only_chunks_and_upper_once_per_complete_case(self):
        import torch
        seen = []; upper = []
        def provider(rows):
            self.assertTrue(all('target' not in row and 'component' not in row for row in rows))
            seen.extend(r['id'] for r in rows)
            return PreparedL0Batch(torch.zeros(len(rows), 128), tuple(r['id'] for r in rows), False)
        def score(features, rows):
            upper.append(len(rows))
            self.assertEqual(features.shape, (len(rows),128))
            return dict(scores=torch.zeros(len(rows)), contract=contract([r['id'] for r in rows], 1))
        report = run_scoring(self.meta, provider, lambda x:x, score, l0_batch_size=32, debug=True)
        self.assertEqual(upper, [130,130,128])
        self.assertEqual(len(seen), 388)
        self.assertTrue(report['scoring_callbacks_executed'])
        self.assertFalse(report['raw_CT_execution_verified'])
        self.assertFalse(report['callback_internal_execution_verified'])
        self.assertFalse(report['quality_verified'])

    def test_rejects_wrong_provider_binding_or_GT(self):
        import torch
        for wrong_ids, gt in ((True,False), (False,True)):
            def provider(rows):
                ids = ('wrong',)*len(rows) if wrong_ids else tuple(r['id'] for r in rows)
                return PreparedL0Batch(torch.zeros(len(rows),128), ids, gt)
            with self.subTest(wrong_ids=wrong_ids, gt=gt), self.assertRaises(ValueError):
                run_scoring(self.meta, provider, lambda x:x, lambda *_:None, l0_batch_size=32, debug=True)

    def test_rejects_chunked_upper_and_truncated_scores(self):
        import torch
        def provider(rows):
            return PreparedL0Batch(torch.zeros(len(rows),128), tuple(r['id'] for r in rows), False)
        for chunked in (True,False):
            def upper(features, rows):
                bound = contract([r['id'] for r in rows], 1)
                bound['upper_chunking'] = chunked
                return dict(scores=torch.zeros(len(rows) if chunked else len(rows)-1), contract=bound)
            with self.subTest(chunked=chunked), self.assertRaises(ValueError):
                run_scoring(self.meta, provider, lambda x:x, upper, l0_batch_size=32, debug=True)


if __name__ == '__main__':
    unittest.main()
