"""CPU UNIT mechanics only; these fixtures are not experimental results."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np
from scipy import ndimage as ndi
from torch_geometric.data import HeteroData

from hiercp_v1x.historical_evaluation import sha
from hiercp_v1x.contracts import canonical_hash
from hiercp_v1x.v23_geometry import (
    V23UpperGeometryCache, adapted_builders, center_admitted, recipient_admitted,
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
        self.assertEqual(receipt['native_center_admission_replacements'], 1)
        self.assertEqual(receipt['total_admission_replacements'], 2)
        self.assertFalse(receipt['native_coordinate_allowlist_bound'])
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

    def test_native_center_membership_is_target_free_and_does_not_move_center(self):
        mask = np.ones((5, 5, 5), bool); mask[2, 2, 2] = False
        regions = SimpleNamespace(full_organ_mask=mask)
        allowed = {'val': frozenset({(2, 2, 2), (1, 1, 1)})}
        self.assertTrue(center_admitted('val', (2, 2, 2), regions, allowed))
        self.assertFalse(center_admitted('val', (2, 2, 2), regions, None))
        self.assertFalse(center_admitted('val', (3, 3, 3), regions, allowed))
        self.assertFalse(center_admitted('other', (2, 2, 2), regions, allowed))
        self.assertFalse(center_admitted('val', (-1, 2, 2), regions, allowed))
        self.assertFalse(center_admitted('val', (5, 2, 2), regions, allowed))
        self.assertFalse(center_admitted('val', (2., 2., 2.), regions, allowed))

    @staticmethod
    def background_anchor_inputs():
        from tests.test_historical_patient_graph import unit_inputs
        inputs = unit_inputs(1)
        case, regions = inputs['recipient_case'], inputs['recipient_regions']
        center = (3, 4, 5)
        # The actual component is a hollow 3-cube. Its unchanged bbox midpoint
        # is background, while its complete surrounding lesion remains real.
        self_mask = case.label == 2
        self_mask[center] = False
        case.label[center] = 0
        regions.full_organ_mask = np.isin(case.label, (1, 2))
        regions.region_labels[center] = -1
        regions.organ_depth = ndi.distance_transform_edt(regions.full_organ_mask,
            sampling=case.spacing).astype(np.float32)
        inputs['specs'] = [replace(inputs['specs'][0], center=center,
            region_id=regions.region_at(center), border_distance_mm=0.)]
        return inputs, center

    def test_actual_hollow_bbox_anchor_graph_preserves_original_coordinate_region_and_equations(self):
        from hiercp import hierarchy
        from hiercp.common import normalized_position
        from hiercp_v1x.historical_patient_graph import build_external_hierarchy
        inputs, center = self.background_anchor_inputs()
        recipient = inputs['recipient_case'].paths.case_id
        with self.assertRaisesRegex(ValueError, 'actual recipient center and region'):
            build_external_hierarchy(**inputs)
        upper, receipt = adapted_builders(inputs['training_case_ids'], (recipient,),
            recorded_centers={recipient: [center]})
        build = upper.__globals__['build_external_hierarchy']
        graph, _, audit = build(**inputs)
        np.testing.assert_array_equal(graph['candidate'].pos.numpy()[0],
            normalized_position(center, inputs['recipient_case'].image.shape))
        self.assertEqual(graph['candidate'].region_index.item(), inputs['recipient_regions'].region_at(center))
        source_axis, anisotropy = hierarchy._principal_axis(inputs['donor_source'].full_mask,
                                                           inputs['donor_case'].spacing)
        case = inputs['recipient_case']; case.shape = case.image.shape
        expected = hierarchy._candidate_raw(case, inputs['recipient_source'], inputs['specs'][0],
            inputs['recipient_regions'], source_axis=source_axis, source_anisotropy=anisotropy,
            ct_clip=inputs['ct_clip'])
        np.testing.assert_array_equal(graph['candidate'].raw_x.numpy()[0], expected)
        self.assertFalse(audit['P_U_labels_in_forward'])
        self.assertTrue(receipt['native_coordinate_allowlist_bound'])
        self.assertTrue(receipt['coordinate_allowlist_contains_no_P_U_targets'])
        self.assertTrue(receipt['CT_coordinate_bounds_and_original_region_at_guard_preserved'])

    def test_background_anchor_requires_exact_native_coordinate_and_exact_original_region(self):
        inputs, center = self.background_anchor_inputs()
        recipient = inputs['recipient_case'].paths.case_id
        upper, _ = adapted_builders(inputs['training_case_ids'], (recipient,),
            recorded_centers={recipient: [(1, 1, 1)]})
        with self.assertRaisesRegex(ValueError, 'actual recipient center and region'):
            upper.__globals__['build_external_hierarchy'](**inputs)
        upper, _ = adapted_builders(inputs['training_case_ids'], (recipient,),
            recorded_centers={recipient: [center]})
        inputs['specs'] = [replace(inputs['specs'][0], region_id=(inputs['specs'][0].region_id + 1) % 2)]
        with self.assertRaisesRegex(ValueError, 'actual recipient center and region'):
            upper.__globals__['build_external_hierarchy'](**inputs)

    def test_normal_bound_candidates_graphs_equal_the_original_historical_equations(self):
        from tests.test_historical_patient_graph import unit_inputs
        from hiercp_v1x.historical_patient_graph import build_external_hierarchy
        inputs = unit_inputs(5)
        recipient = inputs['recipient_case'].paths.case_id
        upper, _ = adapted_builders(inputs['training_case_ids'], (recipient,),
            recorded_centers={recipient: [tuple(spec.center) for spec in inputs['specs']]})
        actual = upper.__globals__['build_external_hierarchy'](**inputs)
        original = build_external_hierarchy(**inputs)
        for left, right in zip(actual[:2], original[:2]):
            for store_name in (*left.node_types, *left.edge_types):
                self.assertEqual(set(left[store_name]), set(right[store_name]))
                for name, value in left[store_name].items():
                    if torch.is_tensor(value):
                        self.assertTrue(torch.equal(value, right[store_name][name]))
                    else:
                        self.assertEqual(value, right[store_name][name])
        self.assertEqual(actual[2], original[2])


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

    def test_memo_hit_avoids_deserialization_and_hashing_and_all_outputs_are_private(self):
        cache = self.cache(); cache.prepare(7)
        plan = self.population.case('train_p', 7)
        expected = cache.get(plan, self.provider)
        with patch('hiercp_v1x.v23_geometry.sha', side_effect=AssertionError('Repeated file SHA')), \
                patch('torch.load', side_effect=AssertionError('Repeated deserialization')):
            first = cache.get(plan, self.provider)
            first[0]['candidate'].raw_x.fill_(999.)
            first[1]['prototype'].raw_x.fill_(888.)
            first[2]['P_as_negative'] = True
            second = cache.get(plan, self.provider)
        self.assertTrue(torch.equal(second[0]['candidate'].raw_x, expected[0]['candidate'].raw_x))
        self.assertTrue(torch.equal(second[1]['prototype'].raw_x, expected[1]['prototype'].raw_x))
        self.assertEqual(second[2], expected[2])
        self.assertNotEqual(expected[0]['candidate'].raw_x.data_ptr(), second[0]['candidate'].raw_x.data_ptr())
        self.assertNotEqual(expected[1]['prototype'].raw_x.data_ptr(), second[1]['prototype'].raw_x.data_ptr())
        self.assertGreater(cache.receipt['CPU_resident_memo']['hits'], 0)

    def test_repeat_prepare_and_admit_reuse_complete_proof_without_payload_load(self):
        cache = self.cache(); complete = cache.prepare(7)
        with patch.object(cache, '_load', side_effect=AssertionError('Repeated stage load')), \
                patch('hiercp_v1x.v23_geometry.sha', side_effect=AssertionError('Repeated stage SHA')):
            self.assertEqual(cache.prepare(7), complete)
            self.assertEqual(cache.admit(7), complete)
        self.assertEqual(cache.receipt['CPU_resident_memo']['stage_admission_hits'], 2)

    def test_index_request_and_case_receipt_tamper_rejected_on_memo_hits(self):
        for relative in ('index.json', 'request.json', 'cases/' +
                         __import__('hashlib').sha256(b'train_p').hexdigest() + '.json'):
            with self.subTest(relative=relative):
                # Each independent new cache/test namespace gets its own proof.
                saved_output = self.root / ('guard_' + relative.replace('/', '_'))
                with patch('hiercp_v1x.v23_geometry.adapted_builders',
                           return_value=(mechanical_upper, dict(CPU_UNIT_fixture=True))):
                    cache = V23UpperGeometryCache(self.bundle, self.population, saved_output,
                                                  workers=2, rss_bytes=8 * 2**30)
                cache.prepare(7)
                path = cache._stage_path(7) / relative
                with path.open('ab') as output: output.write(b' ')
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    cache.get(self.population.case('train_p', 7), self.provider)
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    cache.admit(7)

    def test_atomic_payload_replacement_with_same_bytes_is_rejected(self):
        cache = self.cache(); cache.prepare(7)
        path = cache._payload_path(7, 'train_p')
        replacement = path.with_suffix('.CPU_UNIT_replace')
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)  # Exact newly generated UNIT file only.
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            cache.get(self.population.case('train_p', 7), self.provider)

    def test_source_mutation_and_new_source_file_are_rejected(self):
        source = self.source / 'CPU_UNIT_module.py'
        source.write_text('UNIT_VALUE = 1\n')
        cache = self.cache(); cache.prepare(7)
        source.write_text('UNIT_VALUE = 2\n')
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            cache.get(self.population.case('train_p', 7), self.provider)
        again = self.cache(); again.prepare(7)
        (self.source / 'CPU_UNIT_new.py').write_text('UNIT_VALUE = 3\n')
        with self.assertRaisesRegex(ValueError, 'source directory changed'):
            again.admit(7)

    def test_disabled_memo_and_admission_flags_preserve_tensors_and_reload(self):
        cache = self.cache(memoize_cpu_geometry=False, cache_stage_admission=False)
        complete = cache.prepare(7)
        plan = self.population.case('train_p', 7)
        expected = cache.get(plan, self.provider)
        original_load = torch.load
        with patch('torch.load', wraps=original_load) as load:
            result = cache.get(plan, self.provider)
            self.assertEqual(cache.admit(7), complete)
        self.assertEqual(load.call_count, 4)
        self.assertTrue(torch.equal(expected[0]['candidate'].raw_x, result[0]['candidate'].raw_x))
        self.assertEqual(expected[2], result[2])
        stats = cache.receipt['CPU_resident_memo']
        self.assertEqual(stats['entries'], 0)
        self.assertEqual(stats['resident_bytes'], 0)
        self.assertEqual(stats['stage_admission_hits'], 0)

    def test_resident_budget_evicts_or_streams_without_changing_whole_graph(self):
        cache = self.cache(resident_bytes=1); cache.prepare(128)
        graph, _, _ = cache.get(self.population.case('train_p', 128), self.provider)
        self.assertEqual(graph['candidate'].raw_x.shape[0], 129)
        self.assertEqual(cache.receipt['CPU_resident_memo']['resident_bytes'], 0)
        self.assertGreater(cache.receipt['CPU_resident_memo']['oversize_streams'], 0)
        answer = mechanical_upper(self.bundle, self.provider, self.population.case('train_p', 7).query_rows)
        budget = cache._resident_bytes((answer, {'UNIT': 1}, {}))
        cache.resident_budget = budget
        cache._remember('first', answer, {'UNIT': 1}, {})
        cache._remember('second', answer, {'UNIT': 1}, {})
        self.assertEqual(list(cache._memo), ['second'])
        self.assertLessEqual(cache.receipt['CPU_resident_memo']['resident_bytes'], budget)
        self.assertEqual(cache.receipt['CPU_resident_memo']['evictions'], 1)

    def test_actual_rss_limit_raises_instead_of_reducing_case(self):
        cache = self.cache(); cache.prepare(7)
        with patch('psutil.Process.memory_info', return_value=SimpleNamespace(rss=cache.rss_bytes + 1)):
            with self.assertRaisesRegex(MemoryError, 'no case/graph reduction'):
                cache.get(self.population.case('train_p', 7), self.provider)
        self.assertEqual(cache.receipt['CPU_resident_memo']['entries'], 0)

    def historical_fixture(self):
        # This mechanical mmap fixture exercises reuse/ownership only. It does
        # not claim the real 21-case sealed-cache constructor has been run.
        from hiercp_v1x.comparison_native_upper_cache import ROOT
        full_root = self.root / 'CPU_UNIT_historical'; full_root.mkdir()
        self.bundle.config['cache'] = dict(source_pad=4)
        self.bundle.scope = dict(CPU_UNIT_scope=True)
        self.bundle.receipt = dict(debug=False, baseline_proof=dict(CPU_UNIT=True))
        self.population.meta['config'] = dict(donor_max_diameter_mm=20.)
        self.provider.ds.meta = self.population.meta
        query = {case: self.population.case(case, 128).query_rows
                 for case in self.population.partition_cases('inner_val')}
        entries = {}
        for case, rows in query.items():
            graph, prototype, audit = mechanical_upper(self.bundle, self.provider, rows)
            relative = 'cases/' + __import__('hashlib').sha256(case.encode()).hexdigest() + '.pt'
            path = full_root / relative; path.parent.mkdir(exist_ok=True)
            torch.save(dict(graph=graph, prototype=prototype, audit=audit), path)
            entries[case] = dict(path=relative, payload_sha256=sha(path), case_id=case)
        inventory = full_root / 'inventory.json'; inventory.write_text('{}')
        bank = full_root / 'bank.json'; bank.write_text('{}')
        config = full_root / 'region_config.json'; config.write_text('{}')
        request = full_root / 'request.json'; request.write_text('{}')
        signature = dict(native_inventory_sha256=sha(inventory), bank_path=str(bank), bank_sha256=sha(bank),
            native_region_config_path=str(config), native_region_config_sha256=sha(config),
            execution_code_sha256={}, source_proof=dict(source=str(self.source), verified_files={},
                archive_sha256=sha(ROOT / 'versions/v1/pipeline_v1_source.zip')),
            raw_inputs={row['case_id']: row for row in self.population.meta['raw_records']})
        index = dict(signature=signature, request_sha256=sha(request), region_bindings={},
                     preserved_files={}, cases=list(entries.values()))
        index_path = full_root / 'index.json'; index_path.write_text(json.dumps(index))
        cohort = dict(by_case=query, case_ids=list(query), rows=sum(query.values(), []))
        full = SimpleNamespace(root=full_root, index_path=index_path, index_sha256=sha(index_path),
            inventory_path=inventory, index=index, signature=signature, cohort=cohort, entries=entries,
            _guard_inputs=lambda: None, calls=0)
        def original_load(bundle, provider, rows, **kwargs):
            full.calls += 1
            saved = torch.load(full_root / entries[rows[0]['case_id']]['path'],
                               map_location='cpu', weights_only=False, mmap=True)
            return saved['graph'], saved['prototype'], saved['audit']
        full.upper_graphs = original_load
        return full

    def test_historical_validation_memo_is_owned_private_and_source_guarded(self):
        full = self.historical_fixture()
        cache = self.cache(full_validation_cache=full); cache.prepare(128)
        self.assertEqual(full.calls, 2)
        plan = self.population.case('val_p', 128)
        first = cache.get(plan, self.provider)
        first[0]['candidate'].raw_x.fill_(999.)
        first[1]['prototype'].raw_x.fill_(888.)
        first[2]['CPU_UNIT_fixture'] = False
        with patch('torch.load', side_effect=AssertionError('Historical mmap reread')):
            second = cache.get(plan, self.provider)
            cache.admit(128)
        self.assertEqual(full.calls, 2)
        self.assertEqual(second[0]['candidate'].raw_x[0].item(), 0.)
        self.assertEqual(second[1]['prototype'].raw_x.item(), 1.)
        self.assertTrue(second[2]['CPU_UNIT_fixture'])
        # Mutating a source-proof file cannot be hidden by an owned snapshot.
        full.inventory_path.write_text('{"CPU_UNIT_mutation":true}')
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            cache.get(plan, self.provider)

    def test_historical_provider_and_metadata_change_rejected_on_memo_hit(self):
        full = self.historical_fixture()
        cache = self.cache(full_validation_cache=full); cache.prepare(128)
        self.provider.ds.meta['config']['donor_max_diameter_mm'] = 21.
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            cache.get(self.population.case('val_p', 128), self.provider)
        full.signature['CPU_UNIT_mutation'] = True
        with self.assertRaisesRegex(ValueError, 'metadata/source binding changed'):
            cache.admit(128)

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
