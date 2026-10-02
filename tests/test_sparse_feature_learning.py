"""Synthetic UNIT contracts only; never CT, learnability or CP evidence.

CPU checks exercise observation identity and complete physical32 schedules.
CUDA checks exercise memory tensors only and are a separate test class. No
training checkpoint, production profile, real-data subset or model is created.
"""
import copy
from collections import Counter
from types import SimpleNamespace
import unittest

import torch

from l0_regions.donor_data import assignment
from l0_regions.donor_learning import LiveContext, groups, validate_rows
from tools.local_cnn_interaction_runtime import support_binding
from l0_sparse_feature.learning import expand_rows, validate_train_only, build_memory


def synthetic_fixture():
    """Explicit fake UNIT IDs/coordinates with every original P and 128 U."""
    train = ['UNIT_A', 'UNIT_B', 'UNIT_C', 'UNIT_Z']
    val = ['UNIT_V']
    outer_val = ['UNIT_O']
    cases = train + val + outer_val
    positive_counts = dict(UNIT_A=21, UNIT_B=3, UNIT_C=2, UNIT_Z=0,
                           UNIT_V=5, UNIT_O=0)
    rows, raw = [], []
    for case_index, case in enumerate(cases):
        positives = [dict(component=i + 1, center=[case_index * 300 + i, 10, 20])
                     for i in range(positive_counts[case])]
        comparison = [[case_index * 300 + i, 30, 40] for i in range(128)]
        raw.append(dict(case_id=case, positives=positives,
                        comparison=dict(centers=comparison)))
        if case in outer_val:
            continue
        for positive in positives:
            rows.append(dict(id=f'{case}:P:{positive["component"]}', case_id=case,
                patient_group='case:' + case, component=positive['component'],
                center=copy.deepcopy(positive['center']), target=1))
        for i, center in enumerate(comparison):
            rows.append(dict(id=f'{case}:U:{i}', case_id=case,
                patient_group='case:' + case, component=None,
                center=copy.deepcopy(center), target=0))
    meta = dict(config=dict(seed=42), records=copy.deepcopy(rows), raw_records=raw,
        split=dict(inner_train=train, inner_val=val, outer_train=train + val,
                   outer_val=outer_val),
        donor_pool=[dict(case_id=case, component_id=1) for case in train[:-1]],
        identities=dict(format='hiercp_public_case_benchmark_v1',
            independence_scope='published_case_only', patient_independence_verified=False,
            annotation_scope='provided_masks_may_omit_lesions',
            cases={case:dict(patient_group='case:' + case,
                            identity_basis='published_case_id_only',
                            annotation_complete=None) for case in cases}))
    assigned = assignment(meta, 42)
    for i, row in enumerate(assigned):
        row['bounds'] = dict(edges=i)
    return meta, assigned


class CompleteScheduleCPUChecks(unittest.TestCase):
    def test_all_original_pairs_exactly_once_and_zero_positive_case_retained(self):
        meta, original = synthetic_fixture()
        rows = [row for row in original if row['case_id'] in meta['split']['inner_train']]
        dataset = SimpleNamespace(rows=rows)
        order = list(groups(dataset, 32, 42, 0))
        uses = Counter(i for tile in order for i in tile)
        pairs = Counter((p, u) for tile in order
                        for p in tile if rows[p]['target'] == 1
                        for u in tile if rows[u]['target'] == 0)
        expected = {(p, u) for p, positive in enumerate(rows) if positive['target'] == 1
                    for u, negative in enumerate(rows)
                    if negative['target'] == 0 and positive['case_id'] == negative['case_id']}
        self.assertEqual(set(uses), set(range(len(rows))))
        self.assertEqual(set(pairs), expected)
        self.assertEqual(set(pairs.values()), {1})
        self.assertEqual(max(map(len, order)), 32)
        self.assertTrue(all(0 < len(tile) <= 32 and len(tile) == len(set(tile)) for tile in order))
        self.assertTrue(all(len({rows[i]['case_id'] for i in tile}) == 1 for tile in order))
        zero_ids = {i for i, row in enumerate(rows) if row['case_id'] == 'UNIT_Z'}
        self.assertEqual(len(zero_ids), 128)
        self.assertTrue(zero_ids <= set(uses))
        context = LiveContext(dataset, 32)
        self.assertEqual(context.audit['physical_batch'], 32)
        self.assertEqual(context.audit['unique_observations'], len(rows))
        self.assertEqual(context.audit['ranking_pairs'], len(expected))
        self.assertEqual(context.audit['zero_positive_cases'], 1)
        self.assertEqual(context.uses, uses)

    def test_schedule_seed_changes_order_without_changing_comparisons(self):
        meta, original = synthetic_fixture()
        rows = [row for row in original if row['case_id'] in meta['split']['inner_train']]
        dataset = SimpleNamespace(rows=rows)
        first = list(groups(dataset, 32, 42, 0))
        repeated = list(groups(dataset, 32, 42, 0))
        following = list(groups(dataset, 32, 42, 1))
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, following)
        self.assertEqual(Counter(map(tuple, first)), Counter(map(tuple, following)))

    def test_inverse_multiplicity_preserves_balanced_observation_weight(self):
        meta, original = synthetic_fixture()
        rows = [row for row in original if row['case_id'] in meta['split']['inner_train']]
        context = LiveContext(SimpleNamespace(rows=rows), 32)
        accumulated = Counter()
        for tile in context.order:
            for index in tile:
                target = rows[index]['target']
                accumulated[index] += 1 / (2 * context.counts[target] * context.uses[index])
        for index, row in enumerate(rows):
            self.assertAlmostEqual(accumulated[index], 1 / (2 * context.counts[row['target']]))
        for target in (0, 1):
            self.assertAlmostEqual(sum(accumulated[i] for i, row in enumerate(rows)
                                       if row['target'] == target), .5)

    def test_duplicate_unknown_target_mixed_donor_and_self_patient_fail(self):
        _, original = synthetic_fixture()
        selected = [row for row in original if row['case_id'] == 'UNIT_A']
        for defect in ('duplicate', 'target', 'mixed_donor', 'self_patient'):
            with self.subTest(defect=defect):
                rows = copy.deepcopy(selected)
                if defect == 'duplicate':
                    rows.append(copy.deepcopy(rows[0]))
                elif defect == 'target':
                    rows[0]['target'] = 2
                elif defect == 'mixed_donor':
                    rows[0]['donor_component'] += 1
                else:
                    for row in rows:
                        row['donor_group'] = row['patient_group']
                with self.assertRaises(ValueError):
                    validate_rows(rows)


class ObservationExpansionCPUChecks(unittest.TestCase):
    def test_original_ids_targets_centers_donors_and_order_preserved(self):
        meta, original = synthetic_fixture()
        saved_meta, saved_rows = copy.deepcopy(meta), copy.deepcopy(original)
        cases = ['UNIT_V', 'UNIT_A', 'UNIT_Z']
        actual = expand_rows(meta, original, cases)
        expected = [row for row in original if row['case_id'] in cases]
        self.assertEqual(actual, expected)
        self.assertEqual(meta, saved_meta)
        self.assertEqual(original, saved_rows)
        for case, positives in [('UNIT_A', 21), ('UNIT_V', 5), ('UNIT_Z', 0)]:
            rows = [row for row in actual if row['case_id'] == case]
            self.assertEqual(sum(row['target'] == 1 for row in rows), positives)
            self.assertEqual(sum(row['target'] == 0 for row in rows), 128)
            self.assertEqual(len({(row['donor_case_id'], row['donor_component'], row['donor_group'])
                                  for row in rows}), 1)
        actual[0]['center'][0] += 1
        self.assertEqual(original, saved_rows)

    def test_missing_schedule_key_gets_original_global_index_only(self):
        meta, original = synthetic_fixture()
        stripped = [{key: value for key, value in row.items() if key != 'bounds'}
                    for row in original]
        actual = expand_rows(meta, stripped, ['UNIT_V'])
        expected = [(i, row) for i, row in enumerate(stripped) if row['case_id'] == 'UNIT_V']
        self.assertEqual(len(actual), 133)
        for got, (index, row) in zip(actual, expected):
            self.assertEqual(got['bounds'], {'edges': index})
            self.assertEqual({key: value for key, value in got.items() if key != 'bounds'}, row)
        self.assertTrue(all('bounds' not in row for row in stripped))

    def test_original_positive_and_comparison_coordinate_multisets_required(self):
        meta, original = synthetic_fixture()
        selected = [row for row in original if row['case_id'] == 'UNIT_A']
        defects = ('missing_P', 'missing_U', 'extra_U', 'duplicate_id',
                   'P_component', 'P_center', 'U_center', 'target')
        for defect in defects:
            with self.subTest(defect=defect):
                rows = copy.deepcopy(selected)
                p = next(i for i, row in enumerate(rows) if row['target'] == 1)
                u = next(i for i, row in enumerate(rows) if row['target'] == 0)
                if defect == 'missing_P':
                    rows.pop(p)
                elif defect == 'missing_U':
                    rows.pop(u)
                elif defect == 'extra_U':
                    rows.append(dict(rows[u], id='UNIT_A:extra'))
                elif defect == 'duplicate_id':
                    rows[1]['id'] = rows[0]['id']
                elif defect == 'P_component':
                    rows[p]['component'] = 999
                elif defect == 'P_center':
                    rows[p]['center'][0] += 1
                elif defect == 'U_center':
                    rows[u]['center'][0] += 1
                elif defect == 'target':
                    rows[p]['target'] = 0
                with self.assertRaises(ValueError):
                    expand_rows(meta, rows, ['UNIT_A'])

    def test_default_uses_existing_class_blind_deterministic_donor_assignment(self):
        meta, original = synthetic_fixture()
        changed = copy.deepcopy(original)
        for index, row in enumerate(changed):
            # This input is the old donor-independent observation list. Its
            # incidental donor fields must not relabel P/U or change anchors.
            row.update(donor_case_id='UNIT_OLD', donor_component=index,
                       donor_group='UNIT_OLD')
        before = copy.deepcopy(changed)
        actual = expand_rows(meta, changed, ['UNIT_A'])
        expected = [row for row in original if row['case_id'] == 'UNIT_A']
        self.assertEqual(actual, expected)
        self.assertEqual(changed, before)

    def test_unknown_missing_and_duplicate_requested_cases_fail(self):
        meta, original = synthetic_fixture()
        for cases in ([], ['UNIT_MISSING'], ['UNIT_A', 'UNIT_A'], ['UNIT_O']):
            with self.subTest(cases=cases):
                with self.assertRaises(ValueError):
                    expand_rows(meta, original, cases)

    def test_explicit_complete_donor_override_keeps_observation_semantics(self):
        meta, original = synthetic_fixture()
        cases = ['UNIT_A', 'UNIT_V']
        override = {case:dict(donor_case_id='UNIT_B', donor_component=1,
                             donor_group='case:UNIT_B') for case in cases}
        before = copy.deepcopy(original)
        actual = expand_rows(meta, original, cases, override)
        expected = [dict(row, **override[row['case_id']]) for row in original
                    if row['case_id'] in cases]
        self.assertEqual(actual, expected)
        self.assertEqual(original, before)
        for donor in ({'UNIT_A': override['UNIT_A']},
                      {case:dict(override[case], donor_case_id='UNIT_V',
                                 donor_group='case:UNIT_V') for case in cases},
                      {case:dict(override[case], donor_component=999) for case in cases},
                      {case:dict(override[case], donor_group='case:UNIT_C') for case in cases},
                      {case:dict(override[case], donor_case_id='UNIT_A',
                                 donor_group='case:UNIT_A') for case in cases}):
            with self.subTest(override=donor):
                with self.assertRaises(ValueError):
                    expand_rows(meta, original, cases, donor)

    def test_training_whitelist_and_explicit_identity_are_required(self):
        meta, original = synthetic_fixture()
        rows = expand_rows(meta, original, meta['split']['inner_train'])
        validate_train_only(rows, meta)
        held_out = expand_rows(meta, original, ['UNIT_V'])
        with self.assertRaises(ValueError):
            validate_train_only(rows + held_out, meta)
        for defect in ('recipient_identity', 'heldout_donor', 'unknown_component',
                       'mixed_donor', 'self_patient'):
            with self.subTest(defect=defect):
                changed = copy.deepcopy(rows)
                if defect == 'recipient_identity':
                    changed[0]['patient_group'] = 'case:UNIT_V'
                elif defect == 'heldout_donor':
                    for row in changed:
                        if row['case_id'] == changed[0]['case_id']:
                            row.update(donor_case_id='UNIT_V', donor_component=1,
                                       donor_group='case:UNIT_V')
                elif defect == 'unknown_component':
                    for row in changed:
                        if row['case_id'] == changed[0]['case_id']:
                            row['donor_component'] = 999
                elif defect == 'mixed_donor':
                    other = next(case for case in ('UNIT_B', 'UNIT_C')
                                 if case != changed[0]['donor_case_id'])
                    changed[0].update(donor_case_id=other, donor_component=1,
                                      donor_group='case:' + other)
                else:
                    for row in changed:
                        if row['case_id'] == changed[0]['case_id']:
                            row.update(donor_case_id='UNIT_A', donor_component=1,
                                       donor_group='case:UNIT_A')
                with self.assertRaises(ValueError):
                    validate_train_only(changed, meta)


def synthetic_memory_fixture(device):
    """Explicit UNIT metadata, with one entire recipient donated by query Q."""
    rows = []
    for group in ('UNIT_Q', 'UNIT_A', 'UNIT_B', 'UNIT_C'):
        donor = 'UNIT_Q' if group == 'UNIT_A' else 'UNIT_D'
        for target in (1, 0):
            rows.append(dict(id=f'{group}:{target}', case_id=group, patient_group=group,
                donor_case_id=donor, donor_component=1, donor_group=donor,
                component=1 if target else None, center=[target, 2, 3], target=target,
                bounds=dict(edges=len(rows))))
    embeddings = torch.arange(len(rows) * 128, dtype=torch.float32,
                              device=device).reshape(len(rows), 128) / 1000
    return rows, embeddings


def assert_memory_roundtrip(check, device):
    rows, embeddings = synthetic_memory_fixture(device)
    embeddings.requires_grad_(True)
    before = copy.deepcopy(rows)
    memory = build_memory(rows, embeddings)
    groups = sorted({row['patient_group'] for row in rows})
    check.assertEqual(memory['record_ids'], [row['id'] for row in rows])
    check.assertEqual(memory['patient_groups'], groups)
    check.assertEqual(memory['donor_groups'], [row['donor_group'] for row in rows])
    check.assertEqual(memory['source_rows'], rows)
    check.assertEqual(memory['embeddings'].shape, (len(rows), 128))
    check.assertEqual(memory['embeddings'].dtype, torch.float32)
    check.assertEqual(memory['embeddings'].device, embeddings.device)
    check.assertFalse(memory['embeddings'].requires_grad)
    check.assertEqual(memory['owners'].dtype, torch.long)
    check.assertEqual(memory['classes'].dtype, torch.long)
    check.assertEqual(memory['owners'].device, embeddings.device)
    check.assertEqual(memory['classes'].device, embeddings.device)
    check.assertEqual(memory['owners'].tolist(), [groups.index(row['patient_group']) for row in rows])
    check.assertEqual(memory['classes'].tolist(), [row['target'] for row in rows])
    torch.testing.assert_close(memory['embeddings'], embeddings.detach(), atol=0, rtol=0)
    actual, record_ids, query = support_binding(memory, rows, 'UNIT_Q')
    eligible = [i for i, row in enumerate(rows)
                if 'UNIT_Q' not in (row['patient_group'], row['donor_group'])]
    check.assertEqual(record_ids, [rows[i]['id'] for i in eligible])
    check.assertEqual(query, 'UNIT_Q')
    check.assertEqual(len(record_ids), 4)
    torch.testing.assert_close(actual[0], embeddings.detach()[eligible], atol=0, rtol=0)
    check.assertEqual(actual[1].tolist(), [0, 0, 1, 1])
    check.assertEqual(actual[2].tolist(), [1, 0, 1, 0])
    memory['source_rows'][0]['center'][0] += 10
    check.assertEqual(rows, before)


def assert_memory_failures(check, device):
    rows, embeddings = synthetic_memory_fixture(device)
    for invalid in (embeddings[:-1], embeddings[:, :-1], embeddings.double()):
        with check.subTest(shape=tuple(invalid.shape), dtype=invalid.dtype):
            with check.assertRaises(ValueError):
                build_memory(rows, invalid)
    for value in (float('nan'), float('inf')):
        with check.subTest(nonfinite=value):
            invalid = embeddings.clone()
            invalid[0, 0] = value
            with check.assertRaises(FloatingPointError):
                build_memory(rows, invalid)
    original = build_memory(rows, embeddings)
    for defect in ('order', 'zero_classes', 'owners', 'donor_groups', 'owner_dtype',
                   'class_shape', 'embedding_shape', 'nonfinite'):
        with check.subTest(defect=defect):
            memory = copy.deepcopy(original)
            if defect == 'order':
                memory['record_ids'].reverse()
            elif defect == 'zero_classes':
                memory['classes'].zero_()
            elif defect == 'owners':
                memory['owners'][0] = len(memory['patient_groups'])
            elif defect == 'donor_groups':
                memory['donor_groups'][0] = 'UNIT_WRONG'
            elif defect == 'owner_dtype':
                memory['owners'] = memory['owners'].float()
            elif defect == 'class_shape':
                memory['classes'] = memory['classes'][:, None]
            elif defect == 'embedding_shape':
                memory['embeddings'] = memory['embeddings'][:, :-1]
            else:
                memory['embeddings'][0, 0] = float('nan')
            with check.assertRaises(ValueError):
                support_binding(memory, rows, 'UNIT_Q')


class MemoryMetadataCPUChecks(unittest.TestCase):
    def test_exact_original_memory_metadata_and_query_exclusion(self):
        assert_memory_roundtrip(self, 'cpu')

    def test_shape_finite_metadata_order_and_ownership_failures(self):
        assert_memory_failures(self, 'cpu')


@unittest.skipUnless(torch.cuda.is_available(), 'Separate CUDA tensor UNIT class requires CUDA')
class MemoryMetadataCUDAChecks(unittest.TestCase):
    def test_cuda_memory_metadata_and_query_exclusion(self):
        assert_memory_roundtrip(self, 'cuda')

    def test_cuda_shape_finite_metadata_order_and_ownership_failures(self):
        assert_memory_failures(self, 'cuda')


if __name__ == '__main__':
    unittest.main(verbosity=2)
