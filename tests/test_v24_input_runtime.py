"""CPU DEBUG exact-input tests; no native speed, CUDA or training claim."""
import copy
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hiercp_v1x import v24_input_runtime as runtime
from hiercp_v1x import v24_memory_runtime as memory
from tests import test_v24_provider as fixture


@contextmanager
def installed(*,pin=True):
    original={key:memory._GET.__globals__[key] for key in ('materialize_pair','collate')}
    before=runtime._PIN_OUTPUTS
    try:
        with patch.object(torch.cuda,'is_initialized',return_value=False):
            runtime.install_runtime(pin_final_outputs=pin)
        yield
    finally:
        memory._GET.__globals__.update(original);runtime._PIN_OUTPUTS=before


class OriginalInputRuntimeDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.V24CPUProviderDebug.setUpClass();cls.data=fixture.V24CPUProviderDebug

    @classmethod
    def tearDownClass(cls):fixture.V24CPUProviderDebug.tearDownClass()

    def signatures(self,payload):
        from hiercp_v1x.v24_inputs import tensor_digest
        graphs,source,target=payload
        return tensor_digest((tuple(graph.to_dict() for graph in graphs),source,target))

    def test_actual_original_pair_all_fields_epochs_RNG_and_record_bytes_exact(self):
        from hiercp_v1x.v24_inputs import tensor_digest
        rng=torch.get_rng_state().clone();before=runtime.profile()
        for record in self.data.fixture.records:
            before_record=tensor_digest(record)
            for epoch in (0,1,2,29,40):
                expected=runtime._PAIR(record,epoch=epoch)
                actual=runtime.materialize_pair(record,epoch=epoch)
                self.assertEqual(self.signatures(expected),self.signatures(actual))
                self.assertEqual(before_record,tensor_digest(record))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        after=runtime.profile();self.assertGreater(after['attribute_memo_hits'],before['attribute_memo_hits'])
        self.assertGreater(after['attribute_reused_output_bytes'],before['attribute_reused_output_bytes'])

    def test_second_view_is_private_and_no_memo_survives_epoch_calls(self):
        record=self.data.fixture.records[0]
        original=runtime.materialize_pair(record,epoch=29);expected=self.signatures(original)
        first,second=original[0];edge=next(kind for kind in first.edge_types if first[kind].edge_attr.numel())
        other=second[edge].edge_attr.clone();first[edge].edge_attr.add_(100)
        self.assertTrue(torch.equal(other,second[edge].edge_attr))
        self.assertNotEqual(first[edge].edge_attr.data_ptr(),second[edge].edge_attr.data_ptr())
        self.assertEqual(expected,self.signatures(runtime.materialize_pair(record,epoch=29)))

    def test_four_real_record_tasks_preserve_original_order_and_RNG(self):
        tasks=[(self.data.fixture.records[i%3],i) for i in range(4)]
        rng=torch.get_rng_state().clone()
        expected=[self.signatures(runtime._PAIR(record,epoch=epoch)) for record,epoch in tasks]
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual=list(pool.map(lambda item:self.signatures(runtime.materialize_pair(item[0],epoch=item[1])),tasks))
        self.assertEqual(expected,actual);self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_original_validation_errors_and_mutated_tensor_fail_identically(self):
        record=self.data.fixture.records[0]
        cases=[(record,-1),(record,41)]
        changed=copy.deepcopy(record);changed['target_patch'][0,0,0,0]+=1
        cases.extend(((changed,29),(dict(record,format='invalid'),29)))
        for actual,epoch in cases:
            errors=[]
            for function in (runtime._PAIR,runtime.materialize_pair):
                try:function(actual,epoch=epoch)
                except Exception as error:errors.append((type(error),str(error)))
            self.assertEqual(len(errors),2);self.assertEqual(errors[0],errors[1])

    def test_private_original_code_and_sampler_global_bindings_unchanged(self):
        admitted=runtime._sampler();original=admitted['build'];globals_before=dict(original.__globals__)
        chain=runtime._sampling_functions(original)
        self.assertGreater(len(chain),1)
        private=runtime._private_sampling_chain(original,runtime._attribute_reuse(admitted['attributes']))
        for left,right in zip(chain,runtime._sampling_functions(private)):
            self.assertIs(left.__code__,right.__code__)
            self.assertIsNot(left,right)
        runtime.materialize_pair(self.data.fixture.records[0],epoch=1)
        self.assertIs(admitted['module'].build_local_view,original)
        for key,value in globals_before.items():self.assertIs(original.__globals__[key],value)
        with patch.object(admitted['module'],'_edge_attributes',lambda *args,**kwargs:None):
            with self.assertRaisesRegex(ValueError,'sampling implementation changed'):
                runtime.materialize_pair(self.data.fixture.records[0],epoch=1)

    def test_private_collator_full_PyG_offsets_metadata_debatch_and_patches_exact(self):
        items=[(runtime._PAIR(record,epoch=29),i) for i,record in enumerate(self.data.fixture.records)]
        # This allocator is explicitly CPU DEBUG. It exercises the final-out
        # path without pretending pageable tensors are actual pinned memory.
        allocations=[]
        def allocate(shape,dtype):
            value=torch.empty(shape,dtype=dtype);allocations.append(value);return value
        expected=runtime._COLLATE(items);rng=torch.get_rng_state().clone()
        with patch.object(runtime,'_allocate',side_effect=allocate):actual=runtime._pinned_collator()(items)
        self.assertEqual(self.data.signature(expected),self.data.signature(actual))
        self.assertEqual(actual.graph.num_graphs,6)
        for left,right in zip(expected.graph.to_data_list(),actual.graph.to_data_list()):
            from hiercp_v1x.v24_inputs import tensor_digest
            self.assertEqual(tensor_digest(left.to_dict()),tensor_digest(right.to_dict()))
        self.assertTrue(any(actual.target_patches is value for value in allocations))
        self.assertTrue(any(actual.source_patches is value for value in allocations))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        expected_copy=self.data.signature(expected)
        actual.graph['tumor_surface'].x.add_(100);actual.target_patches.add_(100)
        self.assertEqual(expected_copy,self.data.signature(expected))

    def test_original_collate_malformed_inputs_raise_identical_errors(self):
        valid=runtime._PAIR(self.data.fixture.records[0],epoch=1)
        bad=copy.deepcopy(valid);bad[0][1].transition_epoch+=1
        cases=[[],[(valid,-1)],[(bad,0)],[(valid,'0')]]
        for items in cases:
            errors=[]
            with patch.object(runtime,'_allocate',side_effect=lambda shape,dtype:torch.empty(shape,dtype=dtype)):
                for function in (runtime._COLLATE,runtime._pinned_collator()):
                    try:function(items)
                    except Exception as error:errors.append((type(error),str(error)))
            self.assertEqual(len(errors),2);self.assertEqual(errors[0],errors[1])

    def test_actual_provider_get_cold_warm_private_inputs_and_protection_unchanged(self):
        import psutil
        ds=SimpleNamespace(rows=copy.deepcopy(self.data.fixture.rows),meta=dict(records=self.data.fixture.rows),partition='inner_val')
        def make():return memory.MemorySafeInputProvider(ds,self.data.index,workers=4,resident_bytes=2**29,
            coordinator=memory.MemorySafeCoordinator(psutil.Process().memory_info().rss+2**31))
        original=make();optimized=make()
        try:
            expected=self.data.signature(original.get([0,1,2],epoch=29))
            with installed(),patch.object(runtime,'_allocate',side_effect=lambda shape,dtype:torch.empty(shape,dtype=dtype)):
                result=optimized.get([0,1,2],epoch=29)
                self.assertEqual(expected,self.data.signature(result))
                result.graph['tumor_surface'].x.add_(100);result.target_patches.add_(100)
                self.assertEqual(expected,self.data.signature(optimized.get([0,1,2],epoch=29)))
            self.assertEqual(optimized._protected_ids,set());self.assertEqual(optimized._protected_views,set())
        finally:original.close();optimized.close()


class AttributeAndOutputExecutionDebug(unittest.TestCase):
    def test_empty_multidimensional_relation_array_uses_shape_and_exact_zero_bytes(self):
        calls=[]
        def original(edge_index,values):
            calls.append(True);return np.empty((edge_index.shape[1],10),dtype=np.float32)
        memo=runtime._attribute_reuse(original)
        edges=np.empty((2,0),dtype=np.int64);values=np.empty((0,3,2),dtype=np.float32)
        first=memo(edges,values);second=memo(edges.copy(),values.copy())
        self.assertEqual(first.shape,(0,10));self.assertEqual(len(calls),1)
        self.assertIsNot(first,second)
        memo(edges,np.empty((0,2,3),dtype=np.float32));self.assertEqual(len(calls),2)

    def test_foreign_PyG_function_or_original_source_change_fails_before_cached_collation(self):
        import importlib
        pyg=importlib.import_module('torch_geometric.data.collate')
        with patch.object(pyg,'_collate',lambda *args,**kwargs:None):
            with self.assertRaisesRegex(ValueError,'Foreign original'):
                runtime._pinned_collator()
        key=next(iter(runtime._ORIGINAL_SOURCE_STATS))
        with patch.dict(runtime._ORIGINAL_SOURCE_STATS,{key:None}):
            with self.assertRaisesRegex(ValueError,'source changed'):
                runtime.runtime_contract()

    def test_memo_uses_all_input_bytes_overrides_and_independent_output(self):
        calls=[]
        def original(a,b,*,override=None,chunk_edges=65536):
            calls.append(True)
            return (a+b+(0 if override is None else override)).astype(np.float32)
        attributes=runtime._attribute_reuse(original)
        a=np.arange(20,dtype=np.float32).reshape(5,4);b=a.copy()
        first=attributes(a,b);second=attributes(a.copy(),b.copy())
        self.assertEqual(len(calls),1);self.assertTrue(np.array_equal(first,second))
        second[:]=999;self.assertTrue(np.array_equal(attributes(a,b),first))
        attributes(a,b,override=np.ones_like(a));attributes(a,b,chunk_edges=16)
        changed=a.copy();changed[0,0]+=1;attributes(changed,b)
        self.assertEqual(len(calls),4)

    def test_direct_out_cat_stack_values_dtypes_order_and_original_error_types(self):
        proxy=runtime._PinnedTorch()
        with patch.object(runtime,'_allocate',side_effect=lambda shape,dtype:torch.empty(shape,dtype=dtype)):
            for dtype in (torch.float16,torch.float32,torch.float64,torch.int32,torch.int64,torch.bool):
                a=torch.arange(12).reshape(3,4).to(dtype);b=a.flip(1)
                for dim in (0,1,-1):
                    self.assertTrue(torch.equal(torch.cat([a,b],dim=dim),proxy.cat([a,b],dim=dim)))
                    self.assertTrue(torch.equal(torch.stack([a,b],dim=dim),proxy.stack([a,b],dim=dim)))
                    self.assertEqual(proxy.cat([a,b],dim=dim).dtype,dtype)
            for function,original,args in ((proxy.cat,torch.cat,([],)),
                    (proxy.cat,torch.cat,([torch.tensor(1)],)),
                    (proxy.stack,torch.stack,([torch.zeros(2),torch.zeros(3)],))):
                errors=[]
                for call in (function,original):
                    try:call(*args)
                    except Exception as error:errors.append((type(error),str(error)))
                self.assertEqual(errors[0],errors[1])

    def test_preexisting_out_and_grad_inputs_keep_original_semantics(self):
        proxy=runtime._PinnedTorch();a=torch.arange(4,dtype=torch.float32,requires_grad=True)
        result=proxy.cat([a,a]);result.sum().backward();self.assertTrue(torch.equal(a.grad,torch.full_like(a,2)))
        out=torch.empty(8);self.assertIs(proxy.cat([a.detach(),a.detach()],out=out),out)

    def test_nondefault_layout_keeps_original_output_strides_and_values(self):
        proxy=runtime._PinnedTorch()
        inputs=[torch.arange(120).reshape(2,3,4,5).to(memory_format=torch.channels_last),
            torch.arange(24).reshape(4,6).transpose(0,1),
            torch.arange(360).reshape(2,3,3,4,5).to(memory_format=torch.channels_last_3d)]
        with patch.object(runtime,'_allocate',side_effect=AssertionError('Original layout path required')):
            for value in inputs:
                for original,optimized in ((torch.cat,proxy.cat),(torch.stack,proxy.stack)):
                    expected=original([value,value]);actual=optimized([value,value])
                    self.assertTrue(torch.equal(expected,actual));self.assertEqual(expected.stride(),actual.stride())

    def test_installation_preserves_helpers_and_rejects_foreign_or_liveCUDA(self):
        original_pair=memory._GET.__globals__['materialize_pair'];original_collate=memory._GET.__globals__['collate']
        with installed(pin=False):
            self.assertIs(memory._GET.__globals__['materialize_pair'],runtime.materialize_pair)
            self.assertIs(memory._GET.__globals__['collate'],runtime.collate)
            self.assertIs(runtime.inputs.materialize_pair,runtime._PAIR)
            self.assertIs(runtime.inputs.collate,runtime._COLLATE)
            self.assertFalse(runtime.runtime_contract()['pinned_final_outputs'])
            with patch.object(torch.cuda,'is_initialized',return_value=True):
                with self.assertRaisesRegex(RuntimeError,'pre-CUDA'):runtime.install_runtime()
        self.assertIs(memory._GET.__globals__['materialize_pair'],original_pair)
        self.assertIs(memory._GET.__globals__['collate'],original_collate)
        with patch.dict(memory._GET.__globals__,materialize_pair=lambda *args:None), \
                patch.object(torch.cuda,'is_initialized',return_value=False):
            with self.assertRaisesRegex(ValueError,'Foreign'):runtime.install_runtime()


if __name__=='__main__':unittest.main()
