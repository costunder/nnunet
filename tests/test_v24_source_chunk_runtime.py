"""CPU DEBUG: exact donor-only chunk reuse, concurrency and failure lifetime.

The small original graph fixtures verify execution semantics only. They are
never described as native performance, training or whole-dataset validation.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import inspect
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from hiercp_v1x import v24_input_runtime as runtime, v24_memory_runtime as memory
from tests import test_v24_provider as fixture
from tests.test_v24_input_runtime import installed


class SourceSingleFlightDebug(unittest.TestCase):
    def assert_released(self,before):
        after=runtime.profile()
        for name in ('source_live_chunks','source_cache_live_entries','source_cache_live_key_bytes',
                     'source_cache_live_result_bytes','source_cache_live_tree_visible_array_bytes'):
            self.assertEqual(before[name],after[name],name)

    def test_four_workers_compute_once_and_every_hit_owns_private_exact_snapshot(self):
        before=runtime.profile();scope=runtime._ChunkSourceReuse([])
        ready=threading.Event();release=threading.Event();calls=[]
        def compute():
            calls.append(True);ready.set()
            if not release.wait(5):raise TimeoutError('DEBUG single-flight was not released')
            return np.arange(24,dtype=np.float32).reshape(4,6)
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(scope.resolve,'attribute',b'actual exact key',compute) for _ in range(4)]
                self.assertTrue(ready.wait(2));self.assertFalse(any(f.done() for f in futures))
                release.set();results=[f.result(timeout=3)[0] for f in futures]
            self.assertEqual(calls,[True])
            for result in results:self.assertTrue(np.array_equal(result,np.arange(24).reshape(4,6)))
            for result in results:result[:]=-123
            exact,hit=scope.resolve('attribute',b'actual exact key',compute)
            self.assertTrue(hit);self.assertTrue(np.array_equal(exact,np.arange(24).reshape(4,6)))
            self.assertEqual(runtime.profile()['source_attribute_misses']-before['source_attribute_misses'],1)
            self.assertEqual(runtime.profile()['source_attribute_hits']-before['source_attribute_hits'],4)
        finally:release.set();scope.close()
        self.assert_released(before)

    def test_original_failure_reaches_all_waiters_without_deadlock_or_retry(self):
        before=runtime.profile();scope=runtime._ChunkSourceReuse([]);calls=[]
        ready=threading.Event();release=threading.Event()
        def compute():
            calls.append(True);ready.set()
            if not release.wait(5):raise TimeoutError('DEBUG original failure was not released')
            raise ValueError('Original radius validation failed')
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(scope.resolve,'radius',b'fail exact key',compute) for _ in range(4)]
                self.assertTrue(ready.wait(2));release.set()
                for future in futures:
                    with self.assertRaisesRegex(ValueError,'Original radius validation failed'):
                        future.result(timeout=3)
            self.assertEqual(calls,[True])
            with self.assertRaisesRegex(ValueError,'Original radius validation failed'):
                scope.resolve('radius',b'fail exact key',compute)
        finally:release.set();scope.close()
        self.assert_released(before)

    def test_pending_release_recursive_call_and_post_close_are_explicit_failures(self):
        before=runtime.profile();scope=runtime._ChunkSourceReuse([])
        try:
            def recursive():return scope.resolve('attribute',b'recursive',lambda:np.zeros(1))[0]
            with self.assertRaisesRegex(RuntimeError,'Recursive identical'):
                scope.resolve('attribute',b'recursive',recursive)
            ready=threading.Event();release=threading.Event()
            def compute():
                ready.set()
                if not release.wait(5):raise TimeoutError('DEBUG unfinished computation')
                return np.arange(4)
            with ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(scope.resolve,'radius',b'pending',compute)
                self.assertTrue(ready.wait(2))
                with self.assertRaisesRegex(RuntimeError,'workers must finish'):scope.close()
                release.set();future.result(timeout=3)
        finally:scope.close()
        with self.assertRaisesRegex(RuntimeError,'already released'):
            scope.resolve('attribute',b'new',lambda:np.arange(2))
        self.assert_released(before)

    def test_unchanged_RSS_guard_error_is_propagated_and_snapshot_cache_released(self):
        before=runtime.profile();calls=[]
        def guard():calls.append(True);raise MemoryError('Actual unchanged RSS64GiB exceeded')
        scope=runtime._ChunkSourceReuse([],RSS_guard=guard)
        try:
            for _ in range(2):
                with self.assertRaisesRegex(MemoryError,'unchanged RSS64GiB'):
                    scope.resolve('attribute',b'guard',lambda:np.arange(30,dtype=np.float32))
            self.assertEqual(calls,[True])
        finally:scope.close()
        self.assert_released(before)


class SourceActualInputsDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixture.V24CPUProviderDebug.setUpClass();cls.data=fixture.V24CPUProviderDebug

    @classmethod
    def tearDownClass(cls):fixture.V24CPUProviderDebug.tearDownClass()

    def signature(self,payload):
        from hiercp_v1x.v24_inputs import tensor_digest
        graphs,source,target=payload
        return tensor_digest((tuple(graph.to_dict() for graph in graphs),source,target))

    def test_all_actual_views_metadata_rng_and_private_outputs_equal_without_epoch_cache(self):
        before=runtime.profile();records=self.data.fixture.records
        from hiercp_v1x.v24_inputs import tensor_digest
        proof=[tensor_digest(record) for record in records];rng=torch.get_rng_state().clone()
        for epoch in (0,1,2,29,40):
            expected=[self.signature(runtime._PAIR(record,epoch=epoch)) for record in records]
            scope=runtime._ChunkSourceReuse(records)
            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    actual=list(pool.map(lambda record:runtime._materialize_pair(record,epoch=epoch,source_reuse=scope),records))
                self.assertEqual(expected,[self.signature(value) for value in actual])
                actual[0][0][0]['tumor_surface'].x.add_(17)
                self.assertEqual(expected[1:], [self.signature(value) for value in actual[1:]])
            finally:scope.close()
        self.assertEqual(proof,[tensor_digest(record) for record in records])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertGreater(runtime.profile()['source_attribute_hits'],before['source_attribute_hits'])
        SourceSingleFlightDebug.assert_released(self,before)

    def test_complete_source_tables_admitted_but_subsets_and_target_positions_excluded(self):
        record=self.data.fixture.records[0];source=record['source_local']['nodes']['tumor_surface']
        feature=np.asarray(source['x'],dtype=np.float32);position=np.asarray(source['pos'],dtype=np.float32)
        original=runtime._sampler()['attributes'];signature=inspect.signature(original)
        edges=np.empty((2,0),dtype=np.int64);scope=runtime._ChunkSourceReuse([record])
        def bound(features,positions):
            result=signature.bind(features,feature,positions,position,edges);result.apply_defaults();return result
        try:
            self.assertTrue(scope.attributes_admitted(bound(feature,position)))
            self.assertFalse(scope.attributes_admitted(bound(feature[:1],position[:1])))
            changed=position.copy();changed[:,0]+=123
            self.assertFalse(scope.attributes_admitted(bound(feature,changed)))
            self.assertFalse(scope.attributes_admitted(bound(feature.astype(np.float64),position)))
        finally:scope.close()

    def test_original_ordered_provider_real_train_epochs_full_tensors_and_lifetimes(self):
        import psutil
        def provider():
            ds=SimpleNamespace(rows=copy.deepcopy(self.data.fixture.rows),
                meta=dict(records=self.data.fixture.rows),partition='inner_train')
            return memory.MemorySafeInputProvider(ds,self.data.index,workers=4,resident_bytes=2**29,
                coordinator=memory.MemorySafeCoordinator(psutil.Process().memory_info().rss+2**31))
        original=provider();optimized=provider();before=runtime.profile()
        try:
            for epoch in (1,2,29,40):
                expected=self.data.signature(original.get([2,0,1],epoch=epoch))
                with installed(pin=False):actual=optimized.get([2,0,1],epoch=epoch)
                self.assertEqual(expected,self.data.signature(actual));self.assertEqual(actual.indices.tolist(),[2,0,1])
                self.assertFalse(optimized._views);self.assertFalse(optimized._protected_ids)
                self.assertFalse(optimized._protected_views)
            self.assertEqual(runtime.profile()['source_chunks_created']-before['source_chunks_created'],4)
        finally:original.close();optimized.close()
        SourceSingleFlightDebug.assert_released(self,before)

    def test_parallel_load_path_original_identity_and_foreign_replacement_guard(self):
        with installed(pin=False):
            self.assertIs(memory.MemorySafeInputProvider._parallel,runtime._parallel)
            with patch.object(memory.MemorySafeInputProvider,'_parallel',lambda *args:None), \
                    patch.object(torch.cuda,'is_initialized',return_value=False):
                with self.assertRaisesRegex(ValueError,'Foreign parallel'):runtime.install_runtime(pin_final_outputs=False)


if __name__=='__main__':unittest.main()
