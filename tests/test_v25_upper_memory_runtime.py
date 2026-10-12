"""CPU UNIT lifetime/RSS tests, with no actual CT or CUDA performance claim."""
from collections import OrderedDict
import copy
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from hiercp_v1x import v25_upper_memory_runtime as runtime


class Payload:
    pass


class UpperMemoryRuntimeUnit(unittest.TestCase):
    def inputs(self):
        return SimpleNamespace(_raw_cache=OrderedDict(),_regions={},_donors={},
            _upper_sources=OrderedDict(),_static_regions=OrderedDict(),
            _raw_bytes=0,_source_bytes=0,_region_bytes=0,
            _region_profile={'evictions':0},_lock=threading.RLock(),
            raw_resident_bytes=8*2**30,rss_bytes=64*2**30)

    @staticmethod
    def process(rss):
        return SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=rss[0]))

    def test_no_coordinator_reclaims_real_allocator_before_strict_RSS_check(self):
        inputs=self.inputs();rss=[70*2**30];order=[]
        def collect():order.append('gc');return 3
        def trim():order.append('trim');rss[0]=24*2**30;return {'supported':True,'applied':True}
        with patch.object(runtime.psutil,'Process',side_effect=lambda:self.process(rss)), \
             patch.object(runtime.gc,'collect',side_effect=collect),patch.object(runtime,'_trim_heap',side_effect=trim):
            runtime._rss_locked(inputs)
        self.assertEqual(order,['gc','trim'])
        self.assertEqual(inputs._upper_memory_profile['last_reclamation']['after_RSS_bytes'],rss[0])
        self.assertEqual(inputs._upper_memory_profile['last_reclamation']['reclaimed_RSS_bytes'],46*2**30)
        self.assertAlmostEqual(inputs._upper_memory_profile['last_reclamation']['RSS_reclaimed_fraction'],46/70)
        self.assertEqual(inputs.rss_bytes,64*2**30)
        self.assertEqual(inputs.raw_resident_bytes,8*2**30)
        self.assertEqual(inputs._upper_memory_profile['strict_failures'],0)

    def test_raw_and_associated_source_payloads_dead_before_reclamation(self):
        inputs=self.inputs();raw=Payload();source=Payload()
        raw_ref,source_ref=weakref.ref(raw),weakref.ref(source)
        inputs._raw_cache['case']=(raw,6*2**30);inputs._raw_bytes=6*2**30
        inputs._upper_sources[('case',1)]=(source,4*2**30);inputs._source_bytes=4*2**30
        del raw,source
        rss=[20*2**30]
        def trim():
            self.assertIsNone(raw_ref());self.assertIsNone(source_ref())
            return {'supported':True,'applied':True}
        with patch.object(runtime.psutil,'Process',side_effect=lambda:self.process(rss)), \
             patch.object(runtime.gc,'collect',return_value=0),patch.object(runtime,'_trim_heap',side_effect=trim):
            runtime._rss_locked(inputs)
        self.assertEqual((inputs._raw_bytes,inputs._source_bytes),(0,0))

    def test_source_and_region_eviction_frames_do_not_keep_popped_payload(self):
        for kind in ('source','region'):
            with self.subTest(kind=kind):
                inputs=self.inputs();payload=Payload();ref=weakref.ref(payload)
                if kind=='source':inputs._upper_sources[('case',1)]=(payload,10*2**30);inputs._source_bytes=10*2**30
                else:inputs._static_regions['case']=(payload,10*2**30);inputs._region_bytes=10*2**30
                del payload
                result=runtime._evict_one(inputs)
                self.assertEqual(result,(kind,10*2**30));self.assertIsNone(ref())

    def test_active_immutable_workspace_survives_cache_ownership_eviction(self):
        inputs=self.inputs();live=Payload();live.values=(1,2,3);ref=weakref.ref(live)
        inputs._static_regions['case']=(live,10*2**30);inputs._region_bytes=10*2**30
        inputs._region_admissions={id(live):(ref,'unchanged proof')}
        before=copy.deepcopy(live.values);runtime._evict_one(inputs)
        self.assertIs(ref(),live);self.assertEqual(live.values,before)
        self.assertEqual(inputs._region_admissions[id(live)][1],'unchanged proof')
        self.assertEqual(inputs._region_profile['evictions'],1)

    def test_actual_live_overflow_still_fails_after_reclaim_and_all_evictions(self):
        inputs=self.inputs();inputs._raw_cache['case']=(Payload(),2*2**30);inputs._raw_bytes=2*2**30
        rss=[70*2**30]
        with patch.object(runtime.psutil,'Process',side_effect=lambda:self.process(rss)), \
             patch.object(runtime.gc,'collect',return_value=0),patch.object(runtime,'_trim_heap',return_value={'supported':False}), \
             self.assertRaisesRegex(MemoryError,'Actual complete v24 workspace exceeds process RSS'):
            runtime._rss_locked(inputs)
        self.assertFalse(inputs._raw_cache)
        self.assertEqual(inputs._upper_memory_profile['strict_failures'],1)
        self.assertEqual(inputs._upper_memory_profile['maximum_observed_RSS_bytes'],rss[0])

    def test_under_pressure_threshold_no_gc_and_complete_cache_hit_retained(self):
        inputs=self.inputs();payload=Payload();inputs._raw_cache['case']=(payload,2*2**30);inputs._raw_bytes=2*2**30
        rss=[20*2**30]
        with patch.object(runtime.psutil,'Process',side_effect=lambda:self.process(rss)), \
             patch.object(runtime.gc,'collect',side_effect=AssertionError('unnecessary GC')), \
             patch.object(runtime,'_trim_heap',side_effect=AssertionError('unnecessary trim')):
            runtime._rss_locked(inputs)
        self.assertIs(inputs._raw_cache['case'][0],payload)

    def test_training_provider_pressure_boundary_avoids_extra_40GiB_GC(self):
        inputs=self.inputs();calls=[];rss=[40*2**30]
        inputs.memory_coordinator=SimpleNamespace(
            providers=(SimpleNamespace(resident_bytes=32*2**30),),
            trim=lambda **kwargs:calls.append(kwargs))
        self.assertEqual(runtime._pressure_trigger(inputs,inputs.memory_coordinator),48*2**30)
        self.assertEqual(runtime._pressure_trigger(inputs,None),36*2**30)
        with patch.object(runtime.psutil,'Process',side_effect=lambda:self.process(rss)), \
             patch.object(runtime.gc,'collect',side_effect=AssertionError('unnecessary extra GC')), \
             patch.object(runtime,'_trim_heap',side_effect=AssertionError('unnecessary extra trim')):
            runtime._rss_locked(inputs)
        self.assertEqual(calls,[{'strict':False}])
        self.assertEqual(inputs._upper_memory_profile['allocator_releases'],0)
        self.assertEqual(inputs.rss_bytes,64*2**30)

    def test_runtime_admission_retains_frozen_files_and_adds_execution_sidecars(self):
        inputs=self.inputs();root=Path(runtime.frozen.__file__).parent
        old_files=(root/'v25_preparation.py',root.parent/'tools/run_v25.py')
        extra=root/'v25_independent_preparation.py';admitted=[]
        def original(instance,*args,**kwargs):
            instance._runtime_file_sha256={str(path):'DEBUG prior SHA' for path in old_files}
            instance._admit_runtime_files=lambda paths:admitted.extend(map(Path,paths))
        with patch.object(runtime,'_ORIGINAL_INIT',side_effect=original), \
             patch.object(runtime,'_EXECUTION_RUNTIME_FILES',(extra,)):
            runtime._init(inputs)
        self.assertEqual(admitted,[*old_files,runtime._SOURCE,extra])

    def test_sidecar_installer_is_idempotent_and_preserves_all_frozen_file_bytes(self):
        cls=runtime._CLASS;original=(cls.__init__,cls._rss_locked,cls.preparation_runtime_receipt)
        self.addCleanup(lambda:setattr(cls,'__init__',original[0]))
        self.addCleanup(lambda:setattr(cls,'_rss_locked',original[1]))
        self.addCleanup(lambda:setattr(cls,'preparation_runtime_receipt',original[2]))
        root=Path(runtime.frozen.__file__).parent
        files=[root/name for name in ('v25_preparation.py','v25_factory.py','v24_geometry.py','v24_inputs.py','v24_model.py','v24_provider.py')]
        before={str(p):p.read_bytes() for p in files}
        one=runtime.install_upper_memory_runtime();two=runtime.install_upper_memory_runtime()
        self.assertEqual(one,two)
        self.assertEqual(one['workers'],4)
        self.assertEqual(one['RSS_limit_bytes'],64*2**30)
        self.assertTrue(one['source_identity_unchanged'])
        self.assertEqual(before,{str(p):p.read_bytes() for p in files})


if __name__=='__main__':unittest.main()
