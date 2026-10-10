"""CPU DEBUG: actual sealed original upper equations, not native throughput."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import FrozenInstanceError
import gc
import io
from pathlib import Path
import random
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

from tests import test_v24_gt_blind as fixtures
from hiercp_v1x import v24_preparation_runtime as runtime


class PreparationRuntimeDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.RecipientGTBlindDebug.setUpClass()
        cls.fixture=fixtures.RecipientGTBlindDebug

    @classmethod
    def tearDownClass(cls):fixtures.RecipientGTBlindDebug.tearDownClass()

    def inputs(self):
        f=self.fixture;obj=object.__new__(runtime.AcceleratedNativeInputs)
        obj._lock=threading.RLock();obj._region_lock_guard=threading.Lock();obj._region_locks={}
        obj._static_regions=OrderedDict();obj._region_bytes=0;obj._region_admissions={}
        obj._upper_sources=OrderedDict();obj._source_bytes=0;obj._upper_runtime=None
        obj._raw_cache=OrderedDict();obj._raw_bytes=0;obj._regions=OrderedDict();obj._donors=OrderedDict()
        obj._region_profile=dict(builds=0,hits=0,original_equation_validations=0,evictions=0,
            source_only_donor_builds=0,source_only_donor_hits=0)
        obj._region_workspaces=threading.BoundedSemaphore(4)
        obj.runtime={'workers':4};obj.raw_resident_bytes=8*2**30;obj.rss_bytes=64*2**30
        obj._input_file_proofs={};obj._file_proofs={};obj._runtime_file_sha256={}
        obj.raw={case:{'image':'unused_DEBUG_path'} for case in (f.recipient.case_id,f.donor_context.case_id)}
        obj.config={'cache':{'source_pad':2}};obj.inventory={'config':{'donor_max_diameter_mm':20}}
        obj.graph_config=f.config;obj.ct_clip=f.clip;obj.training_cases={f.donor.paths.case_id}
        obj.bundle=SimpleNamespace(prototype_bank=f.bank)
        obj.guard_input_files=lambda *args:None
        obj.guard_source=lambda:None
        cases={f.recipient.case_id:dict(context=f.recipient,binding=f.recipient.binding()),
            f.donor_context.case_id:dict(context=f.donor_context,donor_case=f.donor,binding=f.donor_context.binding())}
        obj._case=lambda case:cases[case]
        return obj

    def test_full_original_upper_tensors_audit_and_RNG_exact(self):
        from hiercp_v1x.v24_geometry import build_recipient_regions,build_upper_graphs
        from hiercp_v1x.v24_inputs import tensor_digest
        from hiercp.common import stable_case_seed
        from hiercp.region import REGION_CACHE_SEED_SALT
        f=self.fixture;inputs=self.inputs()
        regions=[build_recipient_regions(context,config=f.config,
            seed=stable_case_seed(42,context.case_id,REGION_CACHE_SEED_SALT),ct_clip=f.clip)
            for context in (f.recipient,f.donor_context)]
        expected=build_upper_graphs(f.recipient,f.donor,f.source,*regions,f.rows,f.bank,
            config=f.config,ct_clip=f.clip,training_case_ids=inputs.training_cases)
        # Both depths came from the complete original organ_depth_mm equation.
        _=f.recipient.organ_depth;_=f.donor_context.organ_depth
        numpy_state=np.random.get_state();torch_state=torch.get_rng_state().clone();python_state=random.getstate()
        with (patch('hiercp.common.organ_depth_mm',side_effect=AssertionError('repeated full EDT')),
              patch('hiercp_v1x.v24_factory.prepare_donor',side_effect=AssertionError('unused local donor preparation'))):
            actual=inputs._upper(SimpleNamespace(case_id=f.recipient.case_id,query_rows=f.rows))
            warm=inputs._upper(SimpleNamespace(case_id=f.recipient.case_id,query_rows=f.rows))
        for result in (actual,warm):
            self.assertEqual(tensor_digest((expected[0].to_dict(),expected[1].to_dict())),
                tensor_digest((result[0].to_dict(),result[1].to_dict())))
            self.assertEqual(expected[2],result[2])
        self.assertIs(inputs._upper_runtime.__code__,build_upper_graphs.__code__)
        self.assertEqual(inputs._region_profile['builds'],2)
        self.assertEqual(inputs._region_profile['source_only_donor_builds'],1)
        self.assertEqual(inputs._region_profile['source_only_donor_hits'],1)
        self.assertEqual(python_state,random.getstate());self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        current=np.random.get_state();self.assertEqual(numpy_state[0],current[0])
        self.assertTrue(np.array_equal(numpy_state[1],current[1]));self.assertEqual(numpy_state[2:],current[2:])

    def test_same_case_inflight_deduplication(self):
        inputs=self.inputs();case=self.fixture.recipient.case_id
        with ThreadPoolExecutor(max_workers=4) as pool:regions=list(pool.map(inputs._region,[case]*4))
        self.assertTrue(all(region is regions[0] for region in regions))
        self.assertEqual(inputs._region_profile['builds'],1)
        self.assertEqual(inputs._region_profile['hits'],3)

    def test_distinct_cases_never_wait_under_global_LRU_lock(self):
        inputs=self.inputs();old_case=inputs._case;barrier=threading.Barrier(2)
        def case(identity):
            barrier.wait(timeout=15)
            return old_case(identity)
        inputs._case=case
        with ThreadPoolExecutor(max_workers=2) as pool:
            regions=list(pool.map(inputs._region,[self.fixture.recipient.case_id,self.fixture.donor_context.case_id]))
        self.assertEqual(len(regions),2);self.assertEqual(inputs._region_profile['builds'],2)

    def test_immutable_region_admission_rejects_unknown_and_changed_inputs(self):
        from hiercp_v1x.v24_inputs import recipient_context
        f=self.fixture;inputs=self.inputs();region=inputs._region(f.recipient.case_id)
        inputs._validate_admitted_regions(f.recipient,region)
        for name in runtime._ARRAY_NAMES:
            with self.assertRaises(ValueError):getattr(region,name).flags.writeable=True
        with self.assertRaises(FrozenInstanceError):region.input_sha256='0'*64
        unknown=runtime._AdmittedRegion(**{name:getattr(region,name) for name in runtime._ARRAY_NAMES},
            input_sha256=region.input_sha256)
        with self.assertRaisesRegex(ValueError,'Unknown'):inputs._validate_admitted_regions(f.recipient,unknown)
        other=recipient_context(f.recipient.case_id,f.CT,f.organ,f.spacing*2,f.recipient.image_affine)
        with self.assertRaisesRegex(ValueError,'different'):inputs._validate_admitted_regions(other,region)

    def test_shared_budget_raw_eviction_retains_static_and_admission_is_weak(self):
        inputs=self.inputs();f=self.fixture;region=inputs._region(f.recipient.case_id)
        size=inputs._region_bytes;inputs.raw_resident_bytes=size
        inputs._raw_cache['DEBUG']=({},size);inputs._raw_bytes=size
        with inputs._lock:inputs._rss_locked()
        self.assertFalse(inputs._raw_cache);self.assertEqual(inputs._region_bytes,size)
        inputs.raw_resident_bytes=0
        with inputs._lock:inputs._rss_locked()
        self.assertFalse(inputs._static_regions);self.assertEqual(inputs._region_bytes,0)
        # A caller's active region stays valid; no hidden strong reference survives.
        inputs._validate_admitted_regions(f.recipient,region)
        identity=id(region);reference=weakref.ref(region);del region;gc.collect()
        self.assertIsNone(reference());self.assertNotIn(identity,inputs._region_admissions)

    def test_actual_RSS_error_and_source_guard_are_not_suppressed(self):
        inputs=self.inputs();case=self.fixture.recipient.case_id;inputs._region(case)
        def reject(*args):raise ValueError('source changed')
        inputs.guard_input_files=reject
        with self.assertRaisesRegex(ValueError,'source changed'):inputs._region(case)
        process=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=inputs.rss_bytes+1))
        with patch.object(runtime.psutil,'Process',return_value=process):
            with self.assertRaisesRegex(MemoryError,'no subset'):inputs._rss_locked()

    def test_runtime_file_admission_preserves_science_hash_and_rejects_replacement(self):
        inputs=self.inputs();science=runtime._sha(Path(runtime.factory.__file__))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'runtime_DEBUG.py';path.write_text('original')
            inputs._admit_runtime_files((path,))
            self.assertIn(str(path.resolve()),inputs._input_file_proofs)
            self.assertIn(str(path.resolve()),inputs._file_proofs)
            inputs.guard_source=runtime._ORIGINAL.guard_source.__get__(inputs)
            inputs.guard_source();path.write_text('replaced runtime bytes')
            with self.assertRaisesRegex(ValueError,'changed'):inputs.guard_source()
            with self.assertRaisesRegex(ValueError,'changed'):inputs.preparation_runtime_receipt()
        self.assertEqual(science,runtime._sha(Path(runtime.factory.__file__)))

    def test_install_runtime_is_explicit_idempotent_and_preserves_five_science_files(self):
        original=runtime.factory.V24NativeInputs
        root=Path(runtime.factory.__file__).parent
        names=('v24_factory.py','v24_geometry.py','v24_inputs.py','v24_model.py','v24_provider.py')
        before={name:runtime._sha(root/name) for name in names}
        try:
            receipt=runtime.install_runtime();self.assertEqual(before,receipt['unchanged_scientific_files_sha256'])
            self.assertEqual(receipt,runtime.install_runtime())
            self.assertIs(runtime.factory.build_runtime.__globals__['V24NativeInputs'],runtime.AcceleratedNativeInputs)
            runtime.factory.V24NativeInputs=object
            with self.assertRaisesRegex(ValueError,'unrecognized'):runtime.install_runtime()
        finally:runtime.factory.V24NativeInputs=original
        self.assertEqual(before,{name:runtime._sha(root/name) for name in names})

    def test_progress_wraps_exact_original_prepare_and_restores_get(self):
        inputs=self.inputs();plans=[SimpleNamespace(case_id=str(i)) for i in range(4)];seen=[]
        cache=SimpleNamespace(population=SimpleNamespace(partition_cases=lambda partition,**kwargs:
            ['0','1'] if partition=='inner_train' else ['2','3']))
        def get(plan,provider=None):seen.append(plan.case_id);return ('exact tensor result',)
        cache.get=get
        expected={'format':'original_format','active_u':7,'cases':4,'recipient_GT_used_in_forward':False}
        def prepare(count,target_selection=None):
            self.assertEqual(count,7);self.assertEqual(target_selection,{'unchanged':'selection'})
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(cache.get,plans))
            return expected
        cache.prepare=prepare;inputs.geometry=cache;inputs._install_prepare_progress()
        output=io.StringIO()
        with contextlib.redirect_stdout(output):actual=cache.prepare(7,target_selection={'unchanged':'selection'})
        self.assertIs(actual,expected);self.assertIs(cache.get,get);self.assertEqual(sorted(seen),['0','1','2','3'])
        self.assertIn('completed=4/4 active_U=7',output.getvalue())


if __name__=='__main__':unittest.main()
