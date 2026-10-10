"""CPU DEBUG: real original GT-free tensors plus measured-lifetime RSS controls."""
from collections import OrderedDict
import copy
import gc
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch

from hiercp_v1x import v24_memory_runtime as runtime
from hiercp_v1x.v24_provider import V24InputProvider, V24InputCoordinator
from tests import test_v24_provider as _provider_fixture


class MemoryRuntimeActualInputsDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _provider_fixture.V24CPUProviderDebug.setUpClass()
        cls.data = _provider_fixture.V24CPUProviderDebug

    @classmethod
    def tearDownClass(cls):
        _provider_fixture.V24CPUProviderDebug.tearDownClass()

    def provider(self, *, original=False, partition='inner_val'):
        import psutil
        rss = psutil.Process().memory_info().rss+2**30
        coordinator = runtime.MemorySafeCoordinator(rss)
        dataset = SimpleNamespace(rows=copy.deepcopy(self.data.fixture.rows),
            meta=dict(records=self.data.fixture.rows), partition=partition)
        provider = (V24InputProvider if original else runtime.MemorySafeInputProvider)(
            dataset, self.data.index, workers=4, resident_bytes=2**29,
            coordinator=coordinator)
        return provider

    def test_actual_cold_warm_and_evicted_inputs_are_bit_exact_and_private(self):
        original = self.provider(original=True); optimized = self.provider()
        try:
            rng = torch.get_rng_state().clone()
            expected = self.data.signature(original.get([0,1,2], epoch=29))
            first = optimized.get([0,1,2], epoch=29)
            self.assertEqual(expected, self.data.signature(first))
            first.graph['tumor_surface'].x.add_(10)
            first.target_patches.add_(10)
            self.assertEqual(expected, self.data.signature(optimized.get([0,1,2], epoch=29)))
            with optimized.coordinator.lock:
                while optimized.evict('view'): pass
                while optimized.evict('record'): pass
            self.assertEqual(expected, self.data.signature(optimized.get([0,1,2], epoch=29)))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(optimized._protected_ids, set())
            self.assertEqual(optimized._protected_views, set())
        finally:
            original.close(); optimized.close()

    def test_post_collate_frame_lifetime_reproduces_old_failure_and_new_admission(self):
        # Exact original CPU tensors. Only the RSS reader is a deterministic
        # DEBUG model of resident pages retained by the private clone frame.
        for original in (True, False):
            with self.subTest(original=original):
                provider = self.provider(original=original); refs = []
                collate = runtime._GET.__globals__['collate']
                def observed_collate(items):
                    refs.extend(weakref.ref(g) for ((graphs, _, _), _) in items for g in graphs)
                    return collate(items)
                provider.coordinator._rss_process = SimpleNamespace(memory_info=lambda:
                    SimpleNamespace(rss=provider.rss_bytes+1 if any(r() is not None for r in refs) else 0))
                try:
                    with patch.dict(runtime._GET.__globals__, collate=observed_collate), \
                            patch('hiercp_v1x.v24_provider.collate', observed_collate):
                        if original:
                            with self.assertRaisesRegex(MemoryError, 'Full active'):
                                provider.get([0,1,2], epoch=29)
                        else:
                            batch = provider.get([0,1,2], epoch=29)
                            self.assertEqual(batch.indices.tolist(), [0,1,2])
                            self.assertFalse(any(r() is not None for r in refs))
                    self.assertEqual(provider._protected_ids, set())
                    self.assertEqual(provider._protected_views, set())
                finally:
                    provider.close()

    def test_live_returned_batch_still_cannot_exceed_hard_RSS_limit(self):
        provider = self.provider(); refs = []
        collate = runtime._GET.__globals__['collate']
        def observed_collate(items):
            batch = collate(items); refs.append(weakref.ref(batch)); return batch
        provider.coordinator._rss_process = SimpleNamespace(memory_info=lambda:
            SimpleNamespace(rss=provider.rss_bytes+1 if any(r() is not None for r in refs) else 0))
        try:
            with patch.dict(runtime._GET.__globals__, collate=observed_collate):
                with self.assertRaisesRegex(MemoryError, 'Full active'):
                    provider.get([0,1,2], epoch=29)
            self.assertEqual(provider.coordinator._memory_stats['strict_failures'], 1)
        finally:
            provider.close()

    def test_training_epochs_and_complete_ids_survive_reclamation(self):
        original = self.provider(original=True, partition='inner_train')
        optimized = self.provider(partition='inner_train')
        try:
            for epoch in (1,2,29,40):
                self.assertEqual(self.data.signature(original.get([0,1,2], epoch=epoch)),
                    self.data.signature(optimized.get([0,1,2], epoch=epoch)))
            self.assertEqual(optimized.profile()['sampled_view_entries'], 0)
            with self.assertRaisesRegex(ValueError, 'unique'):
                optimized.get([0,0], epoch=1)
        finally:
            original.close(); optimized.close()

    def test_actual_protected_views_and_live_returned_inputs_survive_coalescing(self):
        provider=self.provider(); coordinator=provider.coordinator; releases=[]
        try:
            live=provider.get([0,1,2],epoch=29); expected=self.data.signature(live)
            original_keys=list(provider._views); protected=original_keys[-1]
            provider._protected_views={protected}; provider._protected_ids={protected[0]}
            resident=provider.resident_bytes
            trigger=coordinator.rss_bytes-(coordinator.rss_bytes-resident)//2
            coordinator._rss_process=SimpleNamespace(memory_info=lambda:SimpleNamespace(
                rss=trigger+2**20 if len(provider._views)>1 else trigger-2**26))
            coordinator._release_allocators=lambda:releases.append(len(provider._views))
            rng=torch.get_rng_state().clone()
            coordinator.trim()
            self.assertEqual(list(provider._views),[protected])
            self.assertEqual(provider._protected_views,{protected})
            self.assertEqual(provider._protected_ids,{protected[0]})
            self.assertEqual(releases,[3,1])
            self.assertEqual(coordinator._memory_stats['provider_coalesced_evictions'],2)
            self.assertEqual(expected,self.data.signature(live))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            provider._protected_views.clear(); provider._protected_ids.clear()
            self.assertEqual(expected,self.data.signature(provider.get([0,1,2],epoch=29)))
        finally:
            provider._protected_views.clear(); provider._protected_ids.clear(); provider.close()

    def test_immutable_record_guard_remains_on_warm_input(self):
        provider = self.provider()
        try:
            provider.get([0], epoch=29)
            entry = provider.records[self.data.fixture.rows[0]['id']]
            file = provider.root/entry['path']; stat = file.stat()
            import os
            os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns+1000000))
            with self.assertRaisesRegex(ValueError, 'replaced or changed'):
                provider.get([0], epoch=29)
        finally:
            if 'stat' in locals():
                os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            provider.close()


class MemoryRuntimePolicyDebug(unittest.TestCase):
    def coordinator(self, rss=1024):
        value = runtime.MemorySafeCoordinator(rss)
        value._rss_process = SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=0))
        return value

    def test_exactly_one_guard_call_changes_no_tensor_AST(self):
        contract = runtime.memory_runtime_contract()
        self.assertEqual(contract['provider_lifetime']['final_trim_expressions_changed'], 1)
        self.assertFalse(contract['provider_lifetime']['tensor_operations_changed'])
        self.assertEqual(set(contract['unchanged_scientific_files_sha256']), set(runtime._SCIENCE_NAMES))
        self.assertTrue(contract['original_initial_pressure_collection_preserved'])
        self.assertTrue(contract['original_eviction_category_provider_order_and_protection_preserved'])
        self.assertFalse(contract['fixed_eviction_batch_or_collection_cap'])
        self.assertFalse(contract['garbage_collection_disabled'])

    def test_inactive_allocator_release_preserves_live_tensor_and_RNG(self):
        coordinator = self.coordinator(); calls=[]
        live = torch.arange(30, dtype=torch.float64).reshape(3,10); expected=live.clone()
        rng = torch.get_rng_state().clone()
        coordinator._malloc_trim=lambda pad:calls.append(('malloc_trim',pad)) or 1
        coordinator._malloc_info=None
        with patch.object(torch.cuda, 'is_initialized', return_value=True), \
                patch.object(torch.cuda.memory, 'host_memory_stats', return_value={'active_bytes.current':240}, create=True), \
                patch.object(torch._C, '_host_emptyCache', lambda:calls.append(('host_empty',)), create=True):
            coordinator._release_allocators()
        self.assertEqual(calls, [('host_empty',), ('malloc_trim',0)])
        self.assertTrue(torch.equal(live, expected)); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(coordinator._memory_stats['host_cache_releases'],1)
        for name in ('allocator_release_wall_seconds','gc_wall_seconds',
                     'inactive_host_release_wall_seconds','malloc_trim_wall_seconds'):
            self.assertGreaterEqual(coordinator._memory_stats[name],0)

    def test_no_allocator_support_does_not_relax_RSS_guard(self):
        coordinator=self.coordinator(); coordinator._rss_process=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=1025))
        coordinator._malloc_trim=None; coordinator._malloc_info=None
        with patch.object(torch.cuda,'is_initialized',return_value=False):
            with self.assertRaisesRegex(MemoryError,'Full active'):
                coordinator.trim()
        self.assertEqual(coordinator.rss_bytes,1024)

    def test_external_cache_evicts_only_LRU_ownership_active_values_live(self):
        coordinator=self.coordinator(); live=torch.arange(6); expected=live.clone()
        geometry=SimpleNamespace(_lock=threading.RLock(),_memo=OrderedDict(
            key=(live,live,{},48)),_bytes=48,_proofs={'key':(1,2)})
        # SimpleNamespace cannot be weak-referenced; use actual weakref-able owners.
        class Owner: pass
        actual=Owner(); actual.__dict__.update(geometry.__dict__)
        inputs=Owner(); inputs._lock=threading.RLock(); inputs._raw_cache=OrderedDict()
        inputs._regions={}; inputs._donors={}; inputs._raw_bytes=0
        inputs._static_regions=OrderedDict(region=(live,24)); inputs._region_bytes=24
        coordinator.bind_inputs(actual,inputs)
        self.assertTrue(coordinator._evict_external())
        self.assertEqual(actual._bytes,0); self.assertEqual(inputs._region_bytes,0)
        self.assertTrue(torch.equal(live,expected))
        self.assertEqual(coordinator._memory_stats['released_cache_logical_bytes'],72)

    def test_external_lock_never_blocks_coordinator_and_no_lock_inversion(self):
        coordinator=self.coordinator()
        class Owner: pass
        geometry=Owner(); geometry._lock=threading.Lock()
        inputs=Owner(); inputs._lock=threading.Lock()
        coordinator.bind_inputs(geometry,inputs)
        geometry._lock.acquire(); inputs._lock.acquire()
        try:
            with coordinator.lock:
                self.assertFalse(coordinator._evict_external())
        finally:
            geometry._lock.release();inputs._lock.release()

    def test_pressure_trigger_uses_original_resident_workspace_headroom(self):
        coordinator=self.coordinator(1024); release=[]
        class Ledger:
            def bytes(self,group):return 0
        provider=SimpleNamespace(resident_bytes=512,_ledger=Ledger(),evict=lambda category:False)
        coordinator.providers=[provider]
        coordinator._rss_process=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=769))
        coordinator._release_allocators=lambda:release.append(True)
        coordinator.trim(strict=False)
        self.assertEqual(release,[True]); self.assertEqual(coordinator._memory_stats['pressure_events'],1)

    def test_install_is_idempotent_and_foreign_replacement_is_rejected(self):
        module=SimpleNamespace(V24InputProvider=V24InputProvider,V24InputCoordinator=V24InputCoordinator)
        first=runtime.install_memory_runtime(module)
        self.assertEqual(first,runtime.install_memory_runtime(module))
        module.V24InputProvider=object
        with self.assertRaisesRegex(ValueError,'Foreign'):runtime.install_memory_runtime(module)

    def test_runtime_source_stat_mutation_rejected(self):
        coordinator=self.coordinator()
        with patch.object(runtime._ORIGINAL_PROVIDER,'_stat',return_value=(0,0,0,0,0)):
            with self.assertRaisesRegex(ValueError,'runtime source changed'):coordinator.trim()

    def test_budget_and_forward_hooks_preserve_original_check_and_gradient(self):
        from hiercp_v1x.historical_evaluation import ResourceBudget
        from hiercp_v1x.transition_v1_data import _ResidentStorageLedger
        class Owner:
            def _rss(self): pass
            def guard_source(self): self.guard_calls += 1
        coordinator=self.coordinator(2**40)
        provider=object.__new__(runtime.MemorySafeInputProvider)
        provider.coordinator=coordinator; provider.ds=SimpleNamespace(partition='inner_val')
        provider.resident_bytes=2**39; provider._ledger=_ResidentStorageLedger()
        provider._protected_ids=set(); provider._protected_views=set()
        coordinator.providers=[provider]
        inputs=Owner(); inputs.memory_coordinator=coordinator; inputs.rss_bytes=coordinator.rss_bytes
        inputs._input_file_proofs={}; inputs._file_proofs={}; inputs.guard_calls=0
        inputs._lock=threading.RLock(); inputs._raw_cache=OrderedDict(); inputs._regions={}; inputs._donors={}
        geometry=Owner(); geometry.rss_bytes=coordinator.rss_bytes; geometry.memory_guard=inputs._rss
        geometry._lock=threading.RLock(); geometry._memo=OrderedDict(); geometry._proofs={}; geometry._bytes=0
        class Scorer(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.scale=torch.nn.Parameter(torch.tensor(2.))
                self.providers={'inner_val':provider}; self.geometry=geometry
                self.budget=ResourceBudget(2**30,coordinator.rss_bytes)
            def forward(self,x,*,epoch,training):
                return SimpleNamespace(values=x*self.scale,workload={})
        scorer=Scorer(); original=scorer.budget.check
        receipt=runtime.bind_memory_runtime(scorer)
        self.assertEqual(receipt['RSS_limit_bytes'],2**40)
        runtime.bind_memory_runtime(scorer)
        self.assertEqual(len(scorer._forward_pre_hooks),1)
        self.assertEqual(len(scorer._forward_hooks),1)
        before=coordinator._memory_stats['trim_calls']; rng=torch.get_rng_state().clone()
        output=scorer(torch.tensor([1.,2.]),epoch=29,training=False)
        output.values.sum().backward()
        self.assertTrue(torch.equal(output.values,torch.tensor([2.,4.])))
        self.assertEqual(scorer.scale.grad.item(),3.)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(coordinator._memory_stats['trim_calls']-before,2)
        self.assertEqual(output.workload['CPU_memory_runtime']['RSS_limit_bytes'],2**40)
        # The existing ResourceBudget remains responsible for CUDA errors too.
        with patch.object(torch.cuda,'memory_allocated',return_value=2**31):
            with self.assertRaisesRegex(MemoryError,'Historical evaluation CUDA'):
                scorer.budget.check()
        self.assertEqual(coordinator._bound_scorers[scorer][1],original)
        self.assertEqual(inputs.guard_calls,2)


class MemoryRuntimeCoalescingDebug(unittest.TestCase):
    """Deterministic DEBUG resident-page model; original protected/LRU evict."""
    def setup_policy(self, *, rss=900, reclaim_fraction=1.):
        coordinator=runtime.MemorySafeCoordinator(1024)
        state=dict(rss=rss,pending=0,events=[],releases=0)
        coordinator._rss_process=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=state['rss']))
        def release():
            state['events'].append('release'); state['releases']+=1
            state['rss']-=int(state['pending']*reclaim_fraction); state['pending']=0
        coordinator._release_allocators=release
        class Provider:
            resident_bytes=512
            def __init__(self,name,views,records=(),protected_views=(),protected_records=()):
                self.name=name; self._views=OrderedDict(views); self._records=OrderedDict(records)
                self._protected_views=set(protected_views); self._protected_ids=set(protected_records)
                self._stats=dict(view_evictions=0,record_evictions=0)
                self._ledger=SimpleNamespace(bytes=lambda group:sum(self._views.values())+sum(self._records.values()))
            def evict(self,category):
                cache=self._views if category=='view' else self._records
                protected=self._protected_views if category=='view' else self._protected_ids
                key=next((key for key in cache if key not in protected),None)
                before=self._ledger.bytes('all')
                result=V24InputProvider.evict(self,category)
                if result:
                    state['pending']+=before-self._ledger.bytes('all')
                    state['events'].append((self.name,category,key))
                return result
        return coordinator,state,Provider

    def test_coalesces_by_measured_deficit_and_preserves_provider_round_order(self):
        coordinator,state,Provider=self.setup_policy()
        first=Provider('A',[('live',32)]+[(str(i),32) for i in range(8)],protected_views=('live',))
        second=Provider('B',[(str(i),32) for i in range(8)],records=(('record',128),))
        coordinator.providers=[first,second]; coordinator.trim()
        self.assertEqual(state['releases'],2)
        self.assertEqual(state['events'],['release',('A','view','0'),('B','view','0'),
            ('A','view','1'),('B','view','1'),('A','view','2'),('B','view','2'),'release'])
        self.assertIn('live',first._views); self.assertIn('record',second._records)
        self.assertEqual(coordinator._memory_stats['provider_coalesced_logical_bytes'],192)
        self.assertLessEqual(state['rss'],752)

    def test_fresh_RSS_requires_more_batches_when_logical_freeing_overestimates_pages(self):
        coordinator,state,Provider=self.setup_policy(reclaim_fraction=.5)
        provider=Provider('A',[(str(i),16) for i in range(30)])
        coordinator.providers=[provider]; coordinator.trim()
        self.assertEqual(state['releases'],6)
        self.assertEqual(coordinator._memory_stats['provider_coalescing_batches'],5)
        self.assertEqual(coordinator._memory_stats['provider_coalesced_evictions'],19)
        self.assertEqual(len(provider._views),11)
        self.assertLessEqual(state['rss'],752)

    def test_zero_logical_alias_bytes_drain_inactive_entries_and_keep_hard_overflow(self):
        coordinator,state,Provider=self.setup_policy(rss=1025)
        provider=Provider('A',[('live',0),('0',0),('1',0),('2',0)],protected_views=('live',))
        coordinator.providers=[provider]
        with self.assertRaisesRegex(MemoryError,'Full active'):
            coordinator.trim()
        self.assertEqual(list(provider._views),['live'])
        self.assertEqual(state['releases'],2)
        self.assertEqual(coordinator._memory_stats['provider_coalesced_evictions'],3)
        self.assertEqual(coordinator._memory_stats['strict_failures'],1)

    def test_inactive_views_then_records_and_live_protected_records_survive(self):
        coordinator,state,Provider=self.setup_policy()
        provider=Provider('A',[('view',16)],records=(('live',64),('0',64),('1',64),('2',64)),protected_records=('live',))
        coordinator.providers=[provider]; coordinator.trim()
        self.assertEqual(state['events'],['release',('A','view','view'),'release',
            ('A','record','0'),('A','record','1'),('A','record','2'),'release'])
        self.assertEqual(list(provider._records),['live'])
        self.assertLessEqual(state['rss'],752)

    def test_coalesced_allocator_and_original_eviction_exceptions_propagate(self):
        for allocator in (True,False):
            with self.subTest(allocator=allocator):
                coordinator,state,Provider=self.setup_policy()
                provider=Provider('A',[(str(i),64) for i in range(5)])
                coordinator.providers=[provider]
                failure=RuntimeError('original allocation release failed' if allocator else 'original eviction failed')
                if allocator:
                    before=coordinator._release_allocators
                    def release():
                        if state['releases']:raise failure
                        before()
                    coordinator._release_allocators=release
                else:
                    def evict(category):raise failure
                    provider.evict=evict
                with self.assertRaises(RuntimeError) as caught:coordinator.trim()
                self.assertIs(caught.exception,failure)

    def test_external_zero_bytes_and_unavailable_locks_end_with_original_strict_error(self):
        for locked in (False,True):
            with self.subTest(locked=locked):
                coordinator,state,Provider=self.setup_policy(rss=1025)
                coordinator.providers=[Provider('empty',())]
                class Owner:pass
                geometry=Owner(); geometry._lock=threading.Lock()
                live=torch.arange(6); expected=live.clone(); rng=torch.get_rng_state().clone()
                geometry._memo=OrderedDict((str(i),(live,live,{},0)) for i in range(3))
                geometry._bytes=0; geometry._proofs={str(i):None for i in range(3)}
                inputs=Owner(); inputs._lock=threading.Lock(); inputs._raw_cache=OrderedDict()
                inputs._regions={}; inputs._donors={}; inputs._raw_bytes=0
                coordinator.bind_inputs(geometry,inputs)
                if locked:geometry._lock.acquire(); inputs._lock.acquire()
                try:
                    with self.assertRaisesRegex(MemoryError,'Full active'):coordinator.trim()
                finally:
                    if locked:geometry._lock.release();inputs._lock.release()
                self.assertEqual(len(geometry._memo),3 if locked else 0)
                self.assertEqual(state['releases'],1 if locked else 2)
                self.assertEqual(coordinator._memory_stats['strict_failures'],1)
                self.assertTrue(torch.equal(live,expected));self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_external_owned_cache_reclamation_rechecks_actual_RSS(self):
        coordinator,state,Provider=self.setup_policy()
        coordinator.providers=[Provider('empty',())]
        class Owner:pass
        geometry=Owner(); geometry._lock=threading.RLock()
        live=torch.arange(6); expected=live.clone()
        geometry._memo=OrderedDict((str(i),(live,live,{},64)) for i in range(6))
        geometry._bytes=384; geometry._proofs={str(i):None for i in range(6)}
        inputs=Owner(); inputs._lock=threading.RLock(); inputs._raw_cache=OrderedDict()
        inputs._regions={}; inputs._donors={}; inputs._raw_bytes=0
        coordinator.bind_inputs(geometry,inputs)
        def release():
            state['releases']+=1
            state['rss']=900-(384-geometry._bytes)//2
        coordinator._release_allocators=release
        coordinator.trim()
        self.assertEqual(state['releases'],4)
        self.assertEqual(coordinator._memory_stats['external_coalescing_batches'],3)
        self.assertEqual(len(geometry._memo),1)
        self.assertLessEqual(state['rss'],752)
        self.assertTrue(torch.equal(live,expected))


if __name__=='__main__':unittest.main()
