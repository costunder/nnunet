"""CPU DEBUG: original full upper equations; no native throughput claim."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests import test_v24_gt_blind as fixtures
from hiercp_v1x import v24_upper_reuse as reuse
from hiercp_v1x.v24_geometry import V24UpperGeometryCache, build_upper_graphs
from hiercp_v1x.v24_inputs import query_inputs, tensor_digest


class _Population:
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)

    def case(self, case, count=128, active_u_indices=None):
        if case != self.rows[0]['case_id'] or not 1 <= count <= 128:
            raise ValueError('CPU DEBUG actual case/count differs')
        positions = tuple(range(count)) if active_u_indices is None else tuple(active_u_indices)
        if len(positions) != count or len(set(positions)) != count:
            raise ValueError('CPU DEBUG complete unique U bank membership required')
        chosen = {f'U:{i}' for i in positions} | {'P:0', 'P:1'}
        rows = [copy.deepcopy(row) for row in self.rows if row['id'] in chosen]
        return SimpleNamespace(case_id=case, active_u_count=count, active_u_indices=positions,
            record_ids=tuple(row['id'] for row in rows), query_rows=rows)


class UpperReuseDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.RecipientGTBlindDebug.setUpClass()
        cls.f = fixtures.RecipientGTBlindDebug
        f = cls.f
        centers = [(x,y,z) for x in range(7,23,3) for y in range(7,23,3) for z in range(7,23,3)][:128]
        cls.rows = [dict(f.rows[0], id='P:0', center=[13,14,13]),
                    dict(f.rows[0], id='P:1', center=[17,15,12])]
        cls.rows += [dict(f.rows[1], id=f'U:{i}', center=list(center)) for i,center in enumerate(centers)]
        cls.rows.sort(key=lambda row: row['id'])
        cls.population = _Population(cls.rows)
        cls.expected = {count: cls.original(cls.population.case(f.recipient.case_id,count))
                        for count in (7,23,128)}

    @classmethod
    def tearDownClass(cls):
        fixtures.RecipientGTBlindDebug.tearDownClass()

    @classmethod
    def original(cls, plan):
        f = cls.f
        return build_upper_graphs(f.recipient,f.donor,f.source,f.regions,f.donor_regions,
            plan.query_rows,f.bank,config=f.config,ct_clip=f.clip,
            training_case_ids=(f.donor.paths.case_id,))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='v24-upper-reuse-CPU-UNIT-')
        self.addCleanup(self.directory.cleanup)
        self.state = dict(valid=True)
        self.inputs = self.make_inputs(Path(self.directory.name))
        self.full = self.population.case(self.f.recipient.case_id,128)
        self.inputs.geometry.get(self.full)
        self.adapter = reuse.bind(self.inputs)

    def make_inputs(self, root):
        f = self.f
        inputs = SimpleNamespace(population=self.population,runtime={'workers':4},
            rss_bytes=64*2**30,raw_resident_bytes=8*2**30,
            graph_config=f.config,bundle=SimpleNamespace(source=f.snapshot,prototype_bank=f.bank),
            _input_file_proofs={},_file_proofs={})
        def guard_source():
            if not self.state['valid']:
                raise ValueError('Actual CPU DEBUG source witness changed')
            for path,proof in inputs._file_proofs.items():
                if V24UpperGeometryCache._file_identity(Path(path)) != proof:
                    raise ValueError('Actual CPU DEBUG implementation witness changed')
        def binding(plan):
            return dict(recipient=f.recipient.binding(),
                donor=dict(case_id=f.donor.paths.case_id,component_id=f.source.component_id),
                prototype_bank_sha256=f.bank.fingerprint(),config_sha256=tensor_digest(f.config.to_dict()),
                source_sha256='actual_sealed_CPU_UNIT_hierarchy')
        def guard(plan, proof):
            guard_source()
            if proof != binding(plan):
                raise ValueError('Actual CPU DEBUG source/input binding changed')
        def builder(plan,provider):
            return self.original(plan)
        inputs.guard_source = guard_source
        inputs.geometry = V24UpperGeometryCache(self.population,root,builder,
            input_binding=binding,guard_inputs=guard,workers=4,resident_bytes=8*2**30,rss_bytes=64*2**30)
        return inputs

    @staticmethod
    def signature(value):
        return tensor_digest((value[0].to_dict(),value[1].to_dict())),value[2]

    def test_original_U7_U23_full128_tensor_dtype_order_and_audit_exact(self):
        for count in (7,23,128):
            plan=self.population.case(self.f.recipient.case_id,count)
            actual=self.adapter.derive(plan)
            self.assertEqual(self.signature(actual),self.signature(self.expected[count]))
        self.assertEqual(len(self.adapter.receipt()['derivations']),3)

    def test_exact_ID_mapping_keeps_all_P_and_inventory_order_not_first_N(self):
        plan=self.population.case(self.f.recipient.case_id,7)
        self.adapter.derive(plan)
        row=self.adapter.receipt()['derivations'][-1]
        self.assertEqual(row['record_ids'],list(plan.record_ids))
        self.assertNotEqual(row['full128_indices'],list(range(len(plan.record_ids))))
        self.assertTrue({'P:0','P:1'}.issubset(row['record_ids']))

    def test_candidate_KNN_recomputed_not_filtered_from_full128(self):
        from hiercp import hierarchy
        plan=self.population.case(self.f.recipient.case_id,7)
        with patch.object(hierarchy,'_knn',wraps=hierarchy._knn) as operation:
            graph,_,_=self.adapter.derive(plan)
        self.assertEqual(operation.call_count,1)
        self.assertEqual(operation.call_args.args[0].shape[0],len(plan.record_ids))
        edge=('candidate','spatial_neighbor','candidate')
        full=self.expected[128][0][edge].edge_index
        chosen=[self.full.record_ids.index(record) for record in plan.record_ids]
        mask=torch.tensor([int(a) in chosen and int(b) in chosen for a,b in full.t()])
        self.assertNotEqual(int(mask.sum()),graph[edge].edge_index.shape[1])
        self.assertTrue(torch.equal(graph[edge].edge_index,self.expected[7][0][edge].edge_index))

    def test_wrong_ID_duplicate_order_center_and_donor_rejected(self):
        for variant in ('unknown','duplicate','order','center','donor'):
            plan=copy.deepcopy(self.population.case(self.f.recipient.case_id,7))
            if variant=='unknown':plan.record_ids=('foreign',)+plan.record_ids[1:]
            if variant=='duplicate':plan.record_ids=(plan.record_ids[0],)*len(plan.record_ids)
            if variant=='order':plan.record_ids=tuple(reversed(plan.record_ids))
            if variant=='center':plan.query_rows[0]['center'][0]+=1
            if variant=='donor':plan.query_rows[0]['donor_component']+=1
            with self.subTest(variant=variant),self.assertRaises(ValueError):
                self.adapter.derive(plan)

    def test_missing_full_source_has_explicit_error_no_cold_builder(self):
        inputs=self.make_inputs(Path(self.directory.name)/'no_full')
        adapter=reuse.bind(inputs)
        with patch.object(inputs,'_case',create=True,side_effect=AssertionError('cold CT')):
            with self.assertRaisesRegex(FileNotFoundError,'no cold fallback'):
                adapter.derive(self.population.case(self.f.recipient.case_id,7))

    def test_no_CT_EDT_region_or_donor_rebuild_and_RNG_exact(self):
        before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
        with (patch('hiercp.common.organ_depth_mm',side_effect=AssertionError('EDT')),
              patch('hiercp_v1x.v24_factory.prepare_donor',side_effect=AssertionError('donor')),
              patch('hiercp_v1x.v24_geometry.build_recipient_regions',side_effect=AssertionError('region'))):
            self.adapter.derive(self.population.case(self.f.recipient.case_id,23))
        self.assertEqual(before[0],random.getstate())
        after=np.random.get_state()
        self.assertEqual(before[1][0],after[0]);self.assertTrue(np.array_equal(before[1][1],after[1]))
        self.assertEqual(before[1][2:],after[2:]);self.assertTrue(torch.equal(before[2],torch.get_rng_state()))

    def test_original_file_bytes_immutable_and_output_private(self):
        path=self.inputs.geometry._path(self.full)[0]
        before=path.read_bytes()
        plan=self.population.case(self.f.recipient.case_id,7)
        first=self.adapter.derive(plan)
        first[0]['candidate'].raw_x.add_(100);first[1]['prototype'].raw_x.add_(50)
        first[2]['recipient_binding']['spacing'][0]=100
        self.assertEqual(self.signature(self.adapter.derive(plan)),self.signature(self.expected[7]))
        self.assertEqual(path.read_bytes(),before)

    def test_source_guard_and_corrupted_tensor_fail_closed(self):
        plan=self.population.case(self.f.recipient.case_id,7)
        self.state['valid']=False
        with self.assertRaisesRegex(ValueError,'witness'):self.adapter.derive(plan)
        self.state['valid']=True
        path=self.inputs.geometry._path(self.full)[0]
        value=torch.load(path,map_location='cpu',weights_only=False)
        value['graph']['candidate'].raw_x[0,0]+=1
        torch.save(value,path)
        with self.assertRaisesRegex(ValueError,'tensor content changed'):self.adapter.derive(plan)

    def test_previously_admitted_full_file_replacement_rejected(self):
        plan=self.population.case(self.f.recipient.case_id,7)
        self.adapter.derive(plan)
        path=self.inputs.geometry._path(self.full)[0]
        with path.open('ab') as stream:stream.write(b'CPU DEBUG replacement')
        with self.assertRaisesRegex(ValueError,'replaced'):self.adapter.derive(plan)

    def test_changed_actual_prototype_bank_rejected_even_if_file_witness_unchanged(self):
        # Simulate a changed live descriptor behind an otherwise admitted file.
        # The full-source original prototype replay must reject its values.
        bank=copy.deepcopy(self.f.bank)
        bank.features[0,0]+=1
        self.inputs.bundle.prototype_bank=bank
        with self.assertRaisesRegex(ValueError,'prototype equations/bank'):
            self.adapter.derive(self.population.case(self.f.recipient.case_id,7))

    def test_full_source_query_audit_is_bound_to_actual_record_order(self):
        path=self.inputs.geometry._path(self.full)[0]
        value=torch.load(path,map_location='cpu',weights_only=False)
        value['audit']['query_inputs_sha256']='0'*64
        torch.save(value,path)
        with self.assertRaisesRegex(ValueError,'audit differs'):
            self.adapter.derive(self.population.case(self.f.recipient.case_id,7))

    def test_PU_supervision_changes_do_not_enter_derived_model_inputs(self):
        plan=self.population.case(self.f.recipient.case_id,7)
        for row in plan.query_rows:
            row['target']=1-int(row.get('target',0));row['component']=999
        self.assertEqual(self.signature(self.adapter.derive(plan)),self.signature(self.expected[7]))

    def test_original_cache_atomic_competing_derivations_exact_no_overwrite(self):
        root=Path(self.directory.name)
        other=self.make_inputs(root);second=reuse.bind(other)
        plan=self.population.case(self.f.recipient.case_id,23)
        barrier=threading.Barrier(2)
        def derive(adapter):
            result=adapter.derive(plan);barrier.wait(timeout=20);return result
        self.inputs.geometry.builder=lambda p,provider:derive(self.adapter)
        other.geometry.builder=lambda p,provider:derive(second)
        with ThreadPoolExecutor(max_workers=2) as pool:
            actual=list(pool.map(lambda cache:cache.get(plan),(self.inputs.geometry,other.geometry)))
        self.assertEqual(self.signature(actual[0]),self.signature(self.expected[23]))
        self.assertEqual(self.signature(actual[1]),self.signature(self.expected[23]))
        path=self.inputs.geometry._path(plan)[0];before=path.read_bytes()
        self.assertEqual(len(list(root.glob('*.pt'))),2)
        self.inputs.geometry.get(plan)
        self.assertEqual(path.read_bytes(),before)

    def test_atomic_different_payload_rejected_original_winner_preserved(self):
        plan=self.population.case(self.f.recipient.case_id,7)
        self.inputs.geometry.get(plan)
        cache=self.inputs.geometry;path,key,binding=cache._path(plan);before=path.read_bytes()
        value=torch.load(path,map_location='cpu',weights_only=False)
        value['graph']['candidate'].raw_x[0,0]+=1
        value['tensor_sha256']=tensor_digest((value['graph'].to_dict(),value['prototype'].to_dict()))
        with self.assertRaisesRegex(ValueError,'different tensors'):
            cache._publish(path,value,plan,key,binding)
        self.assertEqual(path.read_bytes(),before)

    def test_idempotent_CPU_binding_and_foreign_replacement_rejected(self):
        self.assertIs(reuse.bind(self.inputs),self.adapter)
        self.inputs.geometry.builder=lambda plan,provider:None
        with self.assertRaisesRegex(ValueError,'Foreign'):reuse.bind(self.inputs)
        with patch('torch.cuda.is_initialized',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'CPU-only'):
                reuse.bind(self.make_inputs(Path(self.directory.name)/'wrong_GPU'))


if __name__=='__main__':unittest.main()
