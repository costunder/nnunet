"""CPU UNIT: actual upper tensors with evictable raw arrays and immutable proofs."""
from collections import OrderedDict
import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from hiercp_v1x.contracts import canonical_hash
from hiercp_v1x.v24_factory import V24NativeInputs, _stat
from hiercp_v1x.v24_inputs import array_digest, query_inputs, tensor_digest
from hiercp_v22.contracts import sha

ROOT=Path(__file__).resolve().parents[1]


class ImmutableFactoryBindingsDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests import test_v24_gt_blind as fixture
        fixture.RecipientGTBlindDebug.setUpClass()
        cls.fixture=fixture.RecipientGTBlindDebug

    @classmethod
    def tearDownClass(cls): cls.fixture.tearDownClass()

    def fixture_inputs(self):
        f=self.fixture; root=ROOT/'outputs'/('v24_binding_proof_UNIT_'+uuid.uuid4().hex)
        root.mkdir(parents=True)
        inputs=object.__new__(V24NativeInputs)
        inputs.root=root; inputs._lock=threading.RLock()
        inputs._raw_cache=OrderedDict(); inputs._donors=OrderedDict(); inputs._regions=OrderedDict()
        inputs._raw_bytes=0; inputs.raw_resident_bytes=2**30
        inputs._bindings_path=root/'input_bindings.json'
        inputs._input_bindings={}; inputs._binding_receipt=None; inputs._binding_file_proof=None
        inputs._input_file_proofs={}; inputs._file_proofs={}; inputs.raw={}
        case=f.rows[0]['case_id']; donor=f.rows[0]['donor_case_id']
        # Tiny explicit UNIT source files; all neural upper tensors below use
        # the fixture's actual original full CT/organ/geometry equations.
        for identity in (case,donor):
            row={}
            for kind in ('image','label'):
                path=root/(identity+'_'+kind+'.UNIT')
                path.write_bytes((identity+kind).encode())
                row[kind]=str(path.resolve()); row[kind+'_sha256']=sha(path)
                inputs._file_proofs[str(path.resolve())]=_stat(path)
            inputs.raw[identity]=row
        inputs.source_sha256='1'*64; inputs.config_sha256='2'*64; inputs.bank_sha256='3'*64
        inputs.inventory=dict(records=copy.deepcopy(f.rows))
        plan=SimpleNamespace(case_id=case,query_rows=copy.deepcopy(f.rows),record_ids=tuple(row['id'] for row in f.rows))
        inputs.population=SimpleNamespace(partition_cases=lambda partition:[case] if partition=='inner_train' else [],
            case=lambda selected,count:plan)
        inputs._rss=lambda:None
        source=copy.deepcopy(f.source); source.v24_mask_sha256=array_digest(source.full_mask)
        inputs._case=lambda selected:dict(context=f.recipient,binding=f.recipient.binding())
        inputs._donor=lambda row:(f.donor,source,f.prepared)
        return inputs,plan

    def persist(self,inputs,plan):
        expected=inputs.input_binding(plan)
        checksum=inputs._publish_input_bindings()
        (inputs.root/'local').mkdir()
        (inputs.root/'local/index.json').write_text(json.dumps(dict(input_bindings_sha256=checksum)),encoding='utf8')
        return expected

    def test_real_upper_cache_hit_after_new_process_admission_never_loads_raw_or_EDT(self):
        from hiercp_v1x.v24_geometry import V24UpperGeometryCache
        inputs,plan=self.fixture_inputs(); expected=self.persist(inputs,plan)
        # A new runtime owns empty raw/donor/region LRUs and admits only proofs.
        inputs._input_bindings={}; inputs._binding_receipt=None; inputs._binding_file_proof=None
        inputs._admit_input_bindings()
        inputs._case=lambda selected: self.fail('Cached upper lookup reloaded raw CT/EDT')
        inputs._donor=lambda row: self.fail('Cached upper lookup rebuilt annotated donor')
        f=self.fixture
        builder=lambda selected,provider:(f.graph.clone(),f.prototype.clone(),copy.deepcopy(f.audit))
        import psutil
        cache=V24UpperGeometryCache(inputs.population,inputs.root/'upper',builder,
            input_binding=inputs.input_binding,guard_inputs=inputs.guard_inputs,
            workers=4,resident_bytes=2**29,rss_bytes=psutil.Process().memory_info().rss+2**30)
        with patch('hiercp.common.organ_depth_mm',side_effect=AssertionError('EDT on cache hit')):
            first=cache.get(plan); second=cache.get(plan)
        signature=lambda result:tensor_digest((result[0].to_dict(),result[1].to_dict()))
        self.assertEqual(signature(first),signature(second))
        self.assertEqual(inputs.input_binding(plan),expected)
        first[0]['candidate'].pos.add_(100)
        self.assertEqual(signature(second),signature(cache.get(plan)))
        self.assertEqual(inputs._raw_cache,{}); self.assertEqual(inputs._donors,{})

    def test_binding_built_once_outlives_raw_eviction_and_defensive_copy(self):
        inputs,plan=self.fixture_inputs()
        with patch.object(inputs,'_case',wraps=inputs._case) as raw, patch.object(inputs,'_donor',wraps=inputs._donor) as donor:
            first=inputs.input_binding(plan); first['recipient']['spacing'][0]=999
            inputs._raw_cache.clear(); inputs._donors.clear()
            second=inputs.input_binding(plan)
        self.assertEqual(raw.call_count,1); self.assertEqual(donor.call_count,1)
        self.assertNotEqual(first,second)

    def test_raw_file_replacement_and_binding_receipt_change_fail_closed(self):
        inputs,plan=self.fixture_inputs(); self.persist(inputs,plan)
        raw=Path(inputs.raw[plan.case_id]['image'])
        raw.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'): inputs.input_binding(plan)
        with self.assertRaisesRegex(ValueError,'changed or replaced'): inputs._admit_input_bindings()
        inputs,plan=self.fixture_inputs(); self.persist(inputs,plan)
        inputs._bindings_path.write_text(inputs._bindings_path.read_text()+' ',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'metadata changed'): inputs.input_binding(plan)

    def test_source_query_binding_tamper_and_unbound_canonical_index_are_rejected(self):
        inputs,plan=self.fixture_inputs(); self.persist(inputs,plan)
        inputs.source_sha256='4'*64
        with self.assertRaisesRegex(ValueError,'source/query'): inputs._admit_input_bindings()
        inputs,plan=self.fixture_inputs(); self.persist(inputs,plan)
        value=json.loads(inputs._bindings_path.read_text(encoding='utf8'))
        value['bindings'][plan.case_id]['donor']['component_id']+=1
        value['content_sha256']=canonical_hash({key:item for key,item in value.items() if key!='content_sha256'})
        inputs._bindings_path.write_text(json.dumps(value),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'binding differs'): inputs._admit_input_bindings()
        inputs,plan=self.fixture_inputs(); self.persist(inputs,plan)
        (inputs.root/'local/index.json').write_text('{}',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'not bound'): inputs._admit_input_bindings()

    def test_region_return_survives_shared_RSS_eviction(self):
        inputs,plan=self.fixture_inputs(); region=object()
        inputs._regions[plan.case_id]=region
        inputs._rss=lambda:inputs._regions.clear()
        self.assertIs(inputs._region(plan.case_id),region)
        self.assertEqual(inputs._regions,{})

    def test_all86_upper_preparation_discards_private_clones_in_each_worker(self):
        import weakref
        from hiercp_v1x.v24_geometry import V24UpperGeometryCache
        f=self.fixture; cache=object.__new__(V24UpperGeometryCache)
        cases=['UNIT_case_'+str(i) for i in range(86)]
        cache.population=SimpleNamespace(partition_cases=lambda partition,**kwargs:cases[:65] if partition=='inner_train' else cases[65:])
        cache.workers=4; cache._plan=lambda case,count,target:case
        refs=[]; completed=[]; peak=[0]; lock=threading.Lock()
        class PrivateResult:
            def __init__(self): self.graph=f.graph.clone(); self.prototype=f.prototype.clone()
        def get(case):
            result=PrivateResult()
            with lock:
                refs.append(weakref.ref(result)); completed.append(case)
                peak[0]=max(peak[0],sum(ref() is not None for ref in refs))
            return result
        cache.get=get
        receipt=cache.prepare(128)
        self.assertEqual(receipt['cases'],86)
        self.assertEqual(set(completed),set(cases)); self.assertEqual(len(completed),86)
        self.assertLessEqual(peak[0],4)
        self.assertTrue(all(ref() is None for ref in refs))


if __name__=='__main__':unittest.main()
