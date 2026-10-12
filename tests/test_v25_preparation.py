"""CPU UNIT concurrency contracts; synthetic loaders are not actual CT evidence."""
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from hiercp_v1x.v25_factory import V25NativeInputs
from hiercp_v1x.v25_preparation import AcceleratedNativeInputs
from hiercp_v22.contracts import sha


class PreparationConcurrencyUnit(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[1]/'tmp'/('UNIT_v25_preparation_'+uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup_owned_files)
        # Windows sandbox denies GetFinalPathNameByHandle on some otherwise
        # readable fixture files. Only these fresh UNIT paths use lexical
        # resolution; production admission and real file SHA/stat stay intact.
        original_resolve=Path.resolve
        def unit_resolve(path,strict=False):
            return original_resolve(path,strict=False if path.is_relative_to(self.root) else strict)
        resolver=patch.object(Path,'resolve',unit_resolve);resolver.start();self.addCleanup(resolver.stop)

    def cleanup_owned_files(self):
        workspace=Path(__file__).resolve().parents[1]
        if self.root.parent.resolve()!=(workspace/'tmp').resolve() or not self.root.name.startswith('UNIT_v25_preparation_'):
            raise ValueError('UNIT cleanup escaped its fresh owned workspace')
        for path in self.root.iterdir():
            if not path.resolve().is_relative_to(self.root.resolve()) or not path.is_file():
                raise ValueError('UNIT cleanup encountered an unexpected path')
            path.unlink()
        self.root.rmdir()

    def inputs(self, *, accelerated=False):
        cls=AcceleratedNativeInputs if accelerated else V25NativeInputs
        obj=object.__new__(cls)
        obj._lock=threading.RLock();obj._build_lock_guard=threading.Lock();obj._build_locks={}
        obj._raw_cache=OrderedDict();obj._donors=OrderedDict();obj._regions=OrderedDict();obj._raw_bytes=0
        obj._file_proofs={};obj._input_file_proofs={};obj.raw={}
        obj.raw_resident_bytes=8*2**30;obj.rss_bytes=64*2**30
        obj._input_bindings={};obj._binding_receipt=None;obj._binding_file_proof=None
        obj.source_sha256='1'*64;obj.config_sha256='2'*64;obj.bank_sha256='3'*64
        obj.training_cases={'gamma','delta'}
        obj.config={'seed':42,'cache':{'source_pad':2}}
        obj.inventory={'config':{'donor_max_diameter_mm':20}}
        obj.graph_config=object();obj.ct_clip=(-1024,1024);obj.runtime={'workers':4}
        for case in ('alpha','beta','gamma','delta'):
            row={}
            for kind in ('image','label'):
                path=self.root/(case+'_'+kind+'.UNIT')
                if not path.exists():path.write_bytes((case+kind).encode())
                row[kind]=str(path);row[kind+'_sha256']=sha(path)
            obj.raw[case]=row
        if accelerated:
            obj._static_regions=OrderedDict();obj._region_bytes=0;obj._region_admissions={}
            obj._upper_sources=OrderedDict();obj._source_bytes=0
            obj._region_profile={'evictions':0}
        return obj

    @staticmethod
    def loaded(case):
        value={'alpha':1.,'beta':2.,'gamma':3.,'delta':4.}[case]
        return SimpleNamespace(image=np.full((6,6,6),value,np.float32),
            label=np.ones((6,6,6),np.uint8),spacing=np.ones(3),image_affine=np.eye(4),
            paths=SimpleNamespace(case_id=case))

    @staticmethod
    def plans():
        return {case:SimpleNamespace(case_id=case,query_rows=[dict(id=case+':P',case_id=case,
            center=[2,2,2],donor_case_id='gamma',donor_component=1)]) for case in ('alpha','beta')}

    @staticmethod
    def sources(loaded,pad,maximum):
        source=SimpleNamespace(component_id=1,full_mask=np.ones(loaded.image.shape,bool))
        class Collection:
            entries=[(1,1.)]
            def __getitem__(self,index):return source,1.
        return Collection()

    def test_independent_case_SHA_decode_and_EDT_builds_overlap(self):
        inputs=self.inputs();barriers={name:threading.Barrier(2) for name in ('sha','load','depth')}
        counts=Counter();counter_lock=threading.Lock()
        def checksum(path):
            if Path(path).name.endswith('_image.UNIT'):barriers['sha'].wait(timeout=10)
            return sha(path)
        def load(paths):
            barriers['load'].wait(timeout=10)
            with counter_lock:counts[paths.case_id]+=1
            return self.loaded(paths.case_id)
        from hiercp.common import organ_depth_mm
        def depth(mask,spacing):
            barriers['depth'].wait(timeout=10)
            return organ_depth_mm(mask,spacing)
        with patch('hiercp_v1x.v25_factory.sha',side_effect=checksum), \
             patch('hiercp.common.load_case',side_effect=load), \
             patch('hiercp.common.organ_depth_mm',side_effect=depth):
            with ThreadPoolExecutor(max_workers=2) as pool:
                values=list(pool.map(inputs._case,('alpha','beta')))
        self.assertEqual(counts,{'alpha':1,'beta':1})
        self.assertEqual([v['context'].image[0,0,0] for v in values],[1.,2.])
        self.assertEqual(inputs._raw_bytes,sum(size for _,size in inputs._raw_cache.values()))
        for value in values:
            with self.assertRaises(ValueError):value['context'].image.flags.writeable=True

    def test_same_case_one_build_and_static_cache_hit_skips_all_reading(self):
        inputs=self.inputs();counts=Counter()
        def load(paths):
            counts[paths.case_id]+=1;time.sleep(.03)
            return self.loaded(paths.case_id)
        with patch('hiercp.common.load_case',side_effect=load), \
             patch('hiercp_v1x.v25_factory.sha',wraps=sha) as checksum:
            with ThreadPoolExecutor(max_workers=4) as pool:values=list(pool.map(inputs._case,['alpha']*4))
            self.assertEqual(checksum.call_count,2)
        self.assertEqual(counts,{'alpha':1});self.assertTrue(all(value is values[0] for value in values))
        with patch('hiercp.common.load_case',side_effect=AssertionError('cache read')), \
             patch('hiercp_v1x.v25_factory.sha',side_effect=AssertionError('cache SHA')):
            self.assertIs(inputs._case('alpha'),values[0])

    def test_original_CT_organ_full_EDT_and_binding_values_are_exact(self):
        from hiercp_v1x.v24_inputs import recipient_context
        inputs=self.inputs();loaded=self.loaded('alpha')
        expected=recipient_context('alpha',loaded.image,np.isin(loaded.label,(1,2)),loaded.spacing,loaded.image_affine)
        expected_depth=expected.organ_depth;expected_binding=expected.binding()
        with patch('hiercp.common.load_case',side_effect=lambda paths:self.loaded(paths.case_id)):
            actual=inputs._case('alpha')
        for name in ('image','organ_mask','spacing','image_affine'):
            self.assertTrue(np.array_equal(getattr(actual['context'],name),getattr(expected,name)))
            self.assertEqual(getattr(actual['context'],name).dtype,getattr(expected,name).dtype)
        self.assertTrue(np.array_equal(actual['organ_depth'],expected_depth))
        self.assertEqual(actual['binding'],expected_binding)
        self.assertTrue(np.array_equal(actual['donor_case'].label,loaded.label))

    def test_input_binding_callers_release_global_lock_and_same_binding_deduplicates(self):
        inputs=self.inputs();plans=self.plans();inputs.population=SimpleNamespace(case=lambda case,count:plans[case])
        barrier=threading.Barrier(2);counts=Counter();counter_lock=threading.Lock()
        def load(paths):
            if paths.case_id in ('alpha','beta'):barrier.wait(timeout=10)
            with counter_lock:counts[paths.case_id]+=1
            return self.loaded(paths.case_id)
        with patch('hiercp.common.load_case',side_effect=load), \
             patch('hiercp_v22.data.sources',side_effect=self.sources), \
             patch('hiercp_v1x.v25_factory.prepare_donor',return_value=object()) as prepared:
            with ThreadPoolExecutor(max_workers=2) as pool:
                bindings=list(pool.map(inputs.input_binding,(plans['alpha'],plans['beta'])))
            with ThreadPoolExecutor(max_workers=4) as pool:
                repeats=list(pool.map(inputs.input_binding,[plans['alpha']]*4))
        self.assertEqual(counts,{'alpha':1,'beta':1,'gamma':1});self.assertEqual(prepared.call_count,1)
        self.assertTrue(all(value==bindings[0] for value in repeats))
        repeats[0]['recipient']['spacing'][0]=999
        self.assertNotEqual(repeats[0],inputs.input_binding(plans['alpha']))

    def test_distinct_donor_decode_and_preparation_overlap(self):
        inputs=self.inputs();barriers={name:threading.Barrier(2) for name in ('load','prepare')}
        rows=[dict(case_id='alpha',donor_case_id=case,donor_component=1) for case in ('gamma','delta')]
        def load(paths):barriers['load'].wait(timeout=10);return self.loaded(paths.case_id)
        def prepare(*args,**kwargs):barriers['prepare'].wait(timeout=10);return object()
        with patch('hiercp.common.load_case',side_effect=load), \
             patch('hiercp_v22.data.sources',side_effect=self.sources), \
             patch('hiercp_v1x.v25_factory.prepare_donor',side_effect=prepare):
            with ThreadPoolExecutor(max_workers=2) as pool:values=list(pool.map(inputs._donor,rows))
        self.assertEqual({value[0].paths.case_id for value in values},{'gamma','delta'})
        self.assertEqual(len(inputs._donors),2)

    def test_LRU_eviction_accounting_and_live_private_workspace_preserved(self):
        inputs=self.inputs()
        with patch('hiercp.common.load_case',side_effect=lambda paths:self.loaded(paths.case_id)):
            alpha=inputs._case('alpha');size=inputs._raw_bytes
            inputs.raw_resident_bytes=size+size//2
            inputs._donors[('alpha',1)]=object();inputs._regions['alpha']=object()
            beta=inputs._case('beta')
        self.assertEqual(list(inputs._raw_cache),['beta']);self.assertEqual(inputs._raw_bytes,size)
        self.assertFalse(inputs._donors);self.assertFalse(inputs._regions)
        self.assertEqual(alpha['context'].image[0,0,0],1.);self.assertEqual(beta['context'].image[0,0,0],2.)
        expected=copy.deepcopy(alpha['binding'])
        with patch('hiercp.common.load_case',side_effect=lambda paths:self.loaded(paths.case_id)):
            self.assertEqual(inputs._case('alpha')['binding'],expected)
        self.assertEqual(inputs._raw_bytes,sum(size for _,size in inputs._raw_cache.values()))
        inputs.close();self.assertEqual(inputs._raw_bytes,0);self.assertFalse(inputs._raw_cache)

    def test_accelerated_shared_budget_and_close_counters_remain_exact(self):
        inputs=self.inputs(accelerated=True)
        with patch('hiercp.common.load_case',side_effect=lambda paths:self.loaded(paths.case_id)):
            inputs._case('alpha');size=inputs._raw_bytes
            inputs._static_regions['UNIT_region']=(object(),100);inputs._region_bytes=100
            inputs._upper_sources[('UNIT_source',1)]=(object(),50);inputs._source_bytes=50
            inputs.raw_resident_bytes=size+150
            inputs._case('beta')
        self.assertEqual(inputs._raw_bytes,sum(size for _,size in inputs._raw_cache.values()))
        self.assertLessEqual(inputs._raw_bytes+inputs._region_bytes+inputs._source_bytes,inputs.raw_resident_bytes)
        self.assertEqual(inputs._region_bytes,100);self.assertEqual(inputs._source_bytes,50)
        inputs.close();self.assertEqual((inputs._raw_bytes,inputs._region_bytes,inputs._source_bytes),(0,0,0))

    def test_concurrent_evictions_keep_exact_byte_ledger_and_every_case_result(self):
        inputs=self.inputs();barrier=threading.Barrier(4)
        # Every UNIT volume has 216 voxels: float32 CT, bool organ, uint8 label,
        # and the original float32 EDT. Residency is two cases; data stay full.
        inputs.raw_resident_bytes=2*216*(4+1+1+4)
        def load(paths):barrier.wait(timeout=10);return self.loaded(paths.case_id)
        with patch('hiercp.common.load_case',side_effect=load):
            with ThreadPoolExecutor(max_workers=4) as pool:
                values=list(pool.map(inputs._case,('alpha','beta','gamma','delta')))
        self.assertEqual([value['context'].image[0,0,0] for value in values],[1.,2.,3.,4.])
        self.assertEqual(len(inputs._raw_cache),2)
        self.assertEqual(inputs._raw_bytes,sum(size for _,size in inputs._raw_cache.values()))
        self.assertEqual(inputs._raw_bytes,inputs.raw_resident_bytes)

    def test_source_mutation_during_build_fails_before_admission(self):
        inputs=self.inputs()
        def load(paths):
            Path(inputs.raw[paths.case_id]['label']).write_bytes(b'UNIT changed while loading')
            return self.loaded(paths.case_id)
        with patch('hiercp.common.load_case',side_effect=load):
            with self.assertRaisesRegex(ValueError,'changed during complete case build'):inputs._case('alpha')
        self.assertFalse(inputs._raw_cache);self.assertEqual(inputs._raw_bytes,0);self.assertFalse(inputs._file_proofs)

    def test_own_admission_wrapper_uses_four_workers_and_keeps_every_original_plan(self):
        inputs=self.inputs(accelerated=True);cases=[f'UNIT_{index}' for index in range(12)]
        active=0;peak=0;completed=[];state_lock=threading.Lock();barrier=threading.Barrier(4)
        witness=self.root/'UNIT_complete_upper.pt';witness.write_bytes(b'UNIT')
        def get(plan,provider=None):
            nonlocal active,peak
            with state_lock:active+=1;peak=max(peak,active)
            barrier.wait(timeout=10)
            with state_lock:active-=1;completed.append(plan)
            return ('UNIT private graph tensors',)
        selection={case:[0,1] for case in cases}
        cache=SimpleNamespace(population=SimpleNamespace(partition_cases=lambda partition,**kwargs:
            cases[:8] if partition=='inner_train' else cases[8:]),get=get,
            prepare=lambda count,target_selection=None:None,
            _plan=lambda case,count,target:(case,count,target[case]),_path=lambda plan:(witness,'UNIT',{}))
        inputs.geometry=cache;inputs._install_prepare_progress()
        from hiercp_v1x import v24_geometry
        before=Path(v24_geometry.__file__).read_bytes()
        receipt=cache.admit(2,target_selection=selection)
        self.assertEqual(receipt,dict(format=v24_geometry.FORMAT,active_u=2,admitted_cases=12,recipient_GT_used_in_forward=False))
        self.assertEqual(peak,4);self.assertEqual(len(completed),12);self.assertEqual({plan[0] for plan in completed},set(cases))
        self.assertTrue(all(plan[1:]==(2,[0,1]) for plan in completed))
        self.assertEqual(before,Path(v24_geometry.__file__).read_bytes())


if __name__=='__main__':unittest.main()
