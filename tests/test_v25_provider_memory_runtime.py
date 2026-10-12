"""CPU DEBUG cache/allocator lifetime tests; no real CT/CUDA throughput claim."""
from collections import OrderedDict
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from hiercp_v1x import v25_provider_memory_runtime as runtime
from hiercp_v1x.v24_provider import V24InputCoordinator,V24InputProvider


class Payload:
    pass


class ProviderMemoryRuntimeUnit(unittest.TestCase):
    def setUp(self):
        self.coordinator=V24InputCoordinator(64*2**30)
        self.rss=[20*2**30]
        self.addCleanup(setattr,runtime._CLASS,'_trim_locked',runtime._ORIGINAL_TRIM)

    def provider(self):
        provider=object.__new__(V24InputProvider)
        provider._views=OrderedDict();provider._records=OrderedDict()
        provider._protected_views=set();provider._protected_ids=set()
        provider._stats={'view_evictions':0,'record_evictions':0}
        self.coordinator.providers.append(provider)
        return provider

    def process(self):
        return SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=self.rss[0]))

    def test_allocator_release_prevents_unnecessary_original_cache_evictions(self):
        provider=self.provider();payload=Payload();provider._records['record']=payload
        self.rss[0]=70*2**30;events=[]
        def collect():events.append('gc');return 5
        def trim():events.append('trim');self.rss[0]=24*2**30;return {'supported':True,'applied':True}
        with patch.object(runtime.psutil,'Process',side_effect=self.process), \
             patch.object(runtime.gc,'collect',side_effect=collect),patch.object(runtime,'_trim_heap',side_effect=trim):
            runtime._trim_locked(self.coordinator,strict=True)
        self.assertEqual(events,['gc','trim']);self.assertIs(provider._records['record'],payload)
        self.assertEqual(provider._stats['record_evictions'],0)
        self.assertEqual(self.coordinator.rss_bytes,64*2**30)

    def test_original_LRU_evictions_release_payload_before_second_allocator_check(self):
        provider=self.provider();payload=Payload();reference=weakref.ref(payload)
        provider._records['record']=payload;del payload
        self.rss[0]=70*2**30;checks=[]
        def trim():
            checks.append(reference() is None)
            if reference() is None:self.rss[0]=30*2**30
            return {'supported':True,'applied':True}
        with patch.object(runtime.psutil,'Process',side_effect=self.process), \
             patch.object(runtime.gc,'collect',return_value=0),patch.object(runtime,'_trim_heap',side_effect=trim):
            runtime._trim_locked(self.coordinator,strict=True)
        self.assertEqual(checks,[False,True]);self.assertIsNone(reference())
        self.assertEqual(provider._stats['record_evictions'],1)

    def test_inactive_graph_cycles_collected_after_original_ownership_eviction(self):
        provider=self.provider();payload=Payload();payload.cycle=payload;reference=weakref.ref(payload)
        provider._views['view']=payload;del payload
        self.rss[0]=70*2**30;collect=runtime.gc.collect
        def trim():
            if reference() is None:self.rss[0]=30*2**30
            return {'supported':True,'applied':True}
        with patch.object(runtime.psutil,'Process',side_effect=self.process), \
             patch.object(runtime.gc,'collect',side_effect=collect),patch.object(runtime,'_trim_heap',side_effect=trim):
            runtime._trim_locked(self.coordinator,strict=True)
        self.assertIsNone(reference());self.assertEqual(provider._stats['view_evictions'],1)

    def test_active_protected_inputs_survive_and_true_overflow_still_fails(self):
        provider=self.provider();payload=Payload();payload.values=(1,2,3)
        provider._records['protected']=payload;provider._protected_ids={'protected'}
        before=copy.deepcopy(payload.values);self.rss[0]=70*2**30
        with patch.object(runtime.psutil,'Process',side_effect=self.process), \
             patch.object(runtime.gc,'collect',return_value=0), \
             patch.object(runtime,'_trim_heap',return_value={'supported':False}), \
             self.assertRaisesRegex(MemoryError,'Full active GT-free input exceeds shared RSS'):
            runtime._trim_locked(self.coordinator,strict=True)
        self.assertIs(provider._records['protected'],payload);self.assertEqual(payload.values,before)
        self.assertEqual(self.coordinator._v25_provider_memory_profile['strict_failures'],1)

    def test_strict_false_reclaims_but_never_claims_hard_budget_admission(self):
        self.rss[0]=70*2**30
        with patch.object(runtime.psutil,'Process',side_effect=self.process), \
             patch.object(runtime.gc,'collect',return_value=0),patch.object(runtime,'_trim_heap',return_value={'supported':False}):
            runtime._trim_locked(self.coordinator,strict=False)
        self.assertEqual(self.coordinator.rss_bytes,64*2**30)
        self.assertEqual(self.coordinator._v25_provider_memory_profile['strict_failures'],0)

    def test_original_pressure_threshold_and_no_unnecessary_gc(self):
        for value in (20*2**30,63*2**30):
            self.rss[0]=value
            with patch.object(runtime.psutil,'Process',side_effect=self.process), \
                 patch.object(runtime.gc,'collect',side_effect=AssertionError('unnecessary collection')), \
                 patch.object(runtime,'_trim_heap',side_effect=AssertionError('unnecessary allocator hint')):
                runtime._trim_locked(self.coordinator,strict=True)

    def test_installer_idempotent_foreign_refused_and_frozen_bytes_unchanged(self):
        path=Path(runtime.frozen.__file__);before=path.read_bytes()
        first=runtime.install_runtime();self.assertEqual(first,runtime.install_runtime())
        self.assertIs(runtime._CLASS._trim_locked,runtime._trim_locked)
        self.assertEqual(path.read_bytes(),before)
        runtime._CLASS._trim_locked=lambda *args,**kwargs:None
        with self.assertRaisesRegex(ValueError,'Unknown existing'):runtime.install_runtime()

    def test_telemetry_reports_real_before_after_and_original_eviction_count(self):
        provider=self.provider();provider._records['record']=Payload();self.rss[0]=70*2**30
        count=[0]
        def trim():
            count[0]+=1
            if count[0]==2:self.rss[0]=30*2**30
            return {'supported':True,'applied':True}
        with patch.object(runtime.psutil,'Process',side_effect=self.process), \
             patch.object(runtime.gc,'collect',return_value=2),patch.object(runtime,'_trim_heap',side_effect=trim):
            runtime._trim_locked(self.coordinator,strict=True);value=runtime.receipt(self.coordinator)
        self.assertEqual(value['profile']['original_cache_evictions'],1)
        self.assertEqual(value['profile']['allocator_releases'],2)
        self.assertEqual(value['actual_RSS_bytes'],30*2**30)
        self.assertEqual(value['profile']['last_reclamation']['reclaimed_RSS_bytes'],40*2**30)
        self.assertTrue(value['contract']['model_optimizer_RNG_checkpoint_unchanged'])


if __name__=='__main__':unittest.main()
