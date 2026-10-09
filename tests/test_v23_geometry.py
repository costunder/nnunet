"""CPU UNIT mechanics only; these fixtures are not experimental results."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import HeteroData

from hiercp_v1x.historical_evaluation import sha
from hiercp_v1x.v23_geometry import (
    V23UpperGeometryCache, adapted_builders, recipient_admitted,
)


class Plan:
    def __init__(self, case, count, positive):
        self.case_id, self.active_u_count = case, count
        self.donor_case_id = 'train_p' if case != 'train_p' else 'train_zero'
        self.record_ids = tuple(str(i) for i in range(count + positive))
        self.query_rows = [dict(id=case + ':' + str(i), case_id=case,
            patient_group=case, center=[i, 0, 0], donor_case_id=self.donor_case_id,
            donor_component=1, donor_group=self.donor_case_id) for i in range(count + positive)]

    def manifest(self):
        return dict(case=self.case_id, U=self.active_u_count, ids=self.record_ids)


class Population:
    def __init__(self, root):
        raw = []
        for case in ('train_p', 'train_zero', 'val_p', 'val_zero'):
            image, label = root / (case + '.image'), root / (case + '.label')
            image.write_bytes(b'CPU UNIT image'); label.write_bytes(b'CPU UNIT label')
            raw.append(dict(case_id=case, image=str(image), label=str(label),
                            image_sha256=sha(image), label_sha256=sha(label)))
        self.meta = dict(raw_records=raw, split=dict(inner_train=['train_p', 'train_zero']))

    def partition_cases(self, partition, ranking_only=False):
        if partition == 'inner_train':
            return ('train_p',) if ranking_only else ('train_p', 'train_zero')
        return ('val_p', 'val_zero')

    def case(self, case, count=128):
        if type(count) is not int or not 1 <= count <= 128:
            raise ValueError('UNIT active U range')
        return Plan(case, count, int(case in ('train_p', 'val_p')))

    def manifest(self):
        return dict(sha256='CPU_UNIT_POPULATION', zero_P_train_cases=['train_zero'])


def mechanical_upper(bundle, provider, rows, **kwargs):
    graph = HeteroData()
    graph['candidate'].raw_x = torch.arange(len(rows), dtype=torch.float32)[:, None]
    prototype = HeteroData()
    prototype['prototype'].raw_x = torch.tensor([[1.]])
    audit = dict(candidate_count=len(rows), P_U_labels_in_forward=False,
                 annotation_blind=False, CPU_UNIT_fixture=True)
    return graph, prototype, audit


class AdaptationTest(unittest.TestCase):
    def test_exact_single_recipient_admission_change(self):
        upper, receipt = adapted_builders(('train',), ('val',))
        self.assertTrue(callable(upper))
        self.assertEqual(receipt['admission_replacements'], 1)
        self.assertEqual(receipt['upper_import_removals'], 1)
        self.assertTrue(receipt['original_feature_relation_prototype_equations_preserved'])
        self.assertFalse(receipt['original_helpers_modified'])
        self.assertTrue(receipt['recipient_annotation_exposed'])
        self.assertNotIn('historical_patient_graph', upper.__code__.co_names)

    def test_explicit_train_and_validation_admission(self):
        self.assertTrue(recipient_admitted('train', ('train',), ('train',), ('val',)))
        self.assertTrue(recipient_admitted('val', ('train',), ('train',), ('val',)))
        self.assertFalse(recipient_admitted('test', ('train',), ('train',), ('val',)))
        self.assertFalse(recipient_admitted('val', ('other',), ('train',), ('val',)))
        self.assertFalse(recipient_admitted('train', ('train',), ('train',), ('train',)))
        self.assertFalse(recipient_admitted('val', ('train', 'train'), ('train',), ('val',)))

    def test_changed_historical_admission_is_rejected(self):
        with patch('hiercp_v1x.v23_geometry.inspect.getsource', return_value='def changed():\n    return 1\n'):
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                adapted_builders(('train',), ('val',))


class CacheTest(unittest.TestCase):
    def setUp(self):
        # Keep generated UNIT fixtures inside this checkout on sandboxed Windows.
        parent = Path(__file__).resolve().parents[1] / 'outputs/v23_CPU_UNIT/tmp'
        parent.mkdir(parents=True, exist_ok=True)
        self.tmp = TemporaryDirectory(dir=parent)
        self.root = Path(self.tmp.name)
        self.assertTrue(self.root.resolve().is_relative_to(parent.resolve()))
        self.baseline = self.root / 'baseline'; self.baseline.mkdir()
        self.source = self.root / 'source'; self.source.mkdir()
        self.population = Population(self.root)
        for case in ('train_p', 'train_zero', 'val_p', 'val_zero'):
            path = self.baseline / 'shared/regions' / case
            path.mkdir(parents=True)
            (path / 'metadata.json').write_text('{}')
            (path / 'UNIT.tensor').write_bytes(b'original UNIT region fixture')
        self.bundle = SimpleNamespace(baseline=self.baseline, source=self.source,
            config=dict(graph=dict(num_regions=24), ct_clip=[-200., 250.]),
            prototype_bank=SimpleNamespace(fingerprint=lambda: 'CPU_UNIT_BANK',
                                           training_case_ids=['train_p', 'train_zero']))
        self.provider = SimpleNamespace(ds=SimpleNamespace(meta=self.population.meta))

    def tearDown(self):
        self.tmp.cleanup()

    def cache(self, **kwargs):
        with patch('hiercp_v1x.v23_geometry.adapted_builders',
                   return_value=(mechanical_upper, dict(CPU_UNIT_fixture=True))):
            return V23UpperGeometryCache(self.bundle, self.population, self.root / 'new',
                                         workers=2, rss_bytes=8 * 2**30, **kwargs)

    def test_complete_stage_binds_all_p_and_all_validation_cases(self):
        cache = self.cache()
        complete = cache.prepare(7)
        self.assertEqual([row['case_id'] for row in complete['cases']], ['train_p', 'val_p', 'val_zero'])
        self.assertEqual(complete['zero_P_train_cases_no_defined_ranking_loss'], ['train_zero'])
        for case, count in [('train_p', 8), ('val_p', 8), ('val_zero', 7)]:
            graph, _, audit = cache.get(self.population.case(case, 7), self.provider)
            self.assertEqual(graph['candidate'].raw_x.shape[0], count)
            self.assertFalse(audit['P_as_negative'])
            self.assertFalse(audit['P_U_labels_in_forward'])

    def test_private_payload_load_cannot_modify_another_call(self):
        cache = self.cache(); cache.prepare(7)
        plan = self.population.case('train_p', 7)
        first = cache(plan, self.provider)[0]
        first['candidate'].raw_x.fill_(999.)
        second = cache(plan, self.provider)[0]
        self.assertEqual(second['candidate'].raw_x[0].item(), 0.)

    def test_payload_tamper_and_unprepared_stage_are_rejected(self):
        cache = self.cache(); cache.prepare(7)
        with self.assertRaisesRegex(ValueError, 'prepared first'):
            cache(self.population.case('val_p', 14), self.provider)
        path = cache._payload_path(7, 'val_p')
        with path.open('ab') as output: output.write(b'UNIT tamper')
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            cache(self.population.case('val_p', 7), self.provider)

    def test_original_raw_and_region_mutation_rejected(self):
        cache = self.cache(); cache.prepare(7)
        image = Path(self.population.meta['raw_records'][0]['image'])
        image.write_bytes(b'UNIT changed original input')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            cache(self.population.case('train_p', 7), self.provider)

    def test_reopening_reuses_exact_stage_without_neural_state(self):
        cache = self.cache(); first = cache.prepare(7)
        again = self.cache(); second = again.prepare(7)
        self.assertEqual(first, second)
        self.assertTrue(again.receipt['geometry_only'])
        self.assertFalse(again.receipt['neural_features_cached'])
        self.assertTrue(again.finish()['final_full_input_SHA256_verified'])

    def test_stage_query_order_tamper_rejected(self):
        cache = self.cache(); cache.prepare(7)
        plan = self.population.case('val_p', 7)
        plan.query_rows.reverse()
        with self.assertRaisesRegex(ValueError, 'candidate order'):
            cache(plan, self.provider)

    def test_other_ddp_rank_readonly_admission(self):
        root = self.cache(); complete = root.prepare(7)
        rank = self.cache()
        self.assertEqual(rank.admit(7), complete)
        graph, _, _ = rank.get(self.population.case('val_zero', 7), self.provider)
        self.assertEqual(graph['candidate'].raw_x.shape[0], 7)
        self.assertTrue(rank.receipt['read_only_DDP_stage_admission'])

    def test_published_payload_without_receipt_resumes_without_replacement(self):
        root = self.cache(); root.prepare(7)
        payload = root._payload_path(7, 'val_p')
        before = sha(payload)
        # Remove only the exact newly generated UNIT receipt to model interruption.
        root._entry_path(7, 'val_p').unlink()
        again = self.cache(); again.prepare(7)
        self.assertEqual(sha(payload), before)
        self.assertTrue(again._entry_path(7, 'val_p').is_file())

    def test_new_rank_rejects_changed_original_raw_sha(self):
        root = self.cache(); root.prepare(7)
        image = Path(self.population.meta['raw_records'][0]['image'])
        image.write_bytes(b'UNIT changed original source')
        again = self.cache()
        with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
            again.admit(7)

    def test_requires_parallel_cpu_workers_and_preserved_namespace(self):
        with patch('hiercp_v1x.v23_geometry.adapted_builders', return_value=(mechanical_upper, {})):
            with self.assertRaisesRegex(ValueError, 'parallel'):
                V23UpperGeometryCache(self.bundle, self.population, self.root / 'new',
                                      workers=1, rss_bytes=8 * 2**30)
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                V23UpperGeometryCache(self.bundle, self.population, self.baseline / 'old',
                                      workers=2, rss_bytes=8 * 2**30)


if __name__ == '__main__':
    unittest.main()
