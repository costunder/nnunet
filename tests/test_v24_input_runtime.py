"""CPU DEBUG exact-input tests; no native speed, CUDA or training claim."""
import copy
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import threading
import weakref
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
    parallel=memory.MemorySafeInputProvider._parallel
    own_parallel='_parallel' in vars(memory.MemorySafeInputProvider)
    try:
        with patch.object(torch.cuda,'is_initialized',return_value=False):
            runtime.install_runtime(pin_final_outputs=pin)
        yield
    finally:
        memory._GET.__globals__.update(original);runtime._PIN_OUTPUTS=before
        if own_parallel:memory.MemorySafeInputProvider._parallel=parallel
        else:del memory.MemorySafeInputProvider._parallel


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
        for name in ('radius_live_pairs','radius_live_result_bytes','radius_live_key_bytes','radius_live_tree_visible_array_bytes'):
            self.assertEqual(after[name],before[name])

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

    def test_private_context_and_radius_code_RNG_paths_and_source_guards(self):
        admitted=runtime._sampler();select=admitted['select_context'];neighbor=admitted['radius']
        original_globals=dict(select.__globals__);radius_globals=dict(neighbor.__globals__)
        memo=runtime._PairRadius(neighbor,runtime._SCIPY_TREE)
        try:
            namespace=dict(select.__globals__);namespace['_radius_neighbor_ids']=memo
            private=runtime._clone(select,namespace)
            self.assertIs(private.__code__,select.__code__);self.assertIs(memo.original.__code__,neighbor.__code__)
            # Actual original context algorithm takes seeded balanced samples
            # then two complete hop unions. Same RNG state must survive both.
            config=SimpleNamespace(sample_context_nodes=3,sample_interface_radius_mm=.1,
                sample_hops=2,sample_hop_radius_mm=1.1,context_radial_bins=3,
                context_azimuth_bins=4,context_elevation_bins=3)
            position=np.column_stack((np.arange(12),np.zeros(12),np.zeros(12))).astype(np.float32)
            node={'pos_mm':torch.from_numpy(position),'x':torch.zeros((12,16))}
            anchors=position[[0]]
            for seed in (1,42,123):
                left=np.random.default_rng(seed);right=np.random.default_rng(seed)
                expected=select(node,anchors,config,left);actual=private(node,anchors,config,right)
                self.assertTrue(np.array_equal(expected,actual));self.assertEqual(left.bit_generator.state,right.bit_generator.state)
            for key,value in original_globals.items():self.assertIs(select.__globals__[key],value)
            for key,value in radius_globals.items():self.assertIs(neighbor.__globals__[key],value)
        finally:memo.close()
        for name in ('_select_context','_radius_neighbor_ids','cKDTree'):
            with patch.object(admitted['module'],name,lambda *args,**kwargs:None):
                with self.assertRaisesRegex(ValueError,'sampling implementation changed'):
                    runtime._sampler()

    def test_actual_pair_radius_table_released_on_original_validation_error(self):
        before=runtime.profile()
        with self.assertRaises(ValueError):runtime.materialize_pair(self.data.fixture.records[0],epoch=41)
        after=runtime.profile();self.assertEqual(after['radius_pairs_closed'],before['radius_pairs_closed']+1)
        self.assertEqual(after['radius_live_pairs'],before['radius_live_pairs'])

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


class RadiusExecutionDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixture._gt_fixture.RecipientGTBlindDebug.setUpClass()

    @classmethod
    def tearDownClass(cls):fixture._gt_fixture.RecipientGTBlindDebug.tearDownClass()

    def original_radius(self):
        # Same original admitted archived routine; never approximate neighbors.
        return runtime._sampler()['radius']

    def test_radius_keys_all_bytes_radius_dtype_shape_and_independent_outputs(self):
        original=self.original_radius();memo=runtime._PairRadius(original,runtime._SCIPY_TREE)
        full=np.array([[0,0,0],[1,0,0],[2,0,0],[2,1,0]],dtype=np.float32)
        query=full[[0,2]];before=runtime.profile()
        try:
            expected=original(full,query,1.0);first=memo(full,query,1.0);second=memo(full.copy(),query.copy(),1.0)
            self.assertTrue(np.array_equal(first,expected));self.assertTrue(np.array_equal(second,expected))
            self.assertIsNot(first,second);first[:]=-1;second[:]=-2
            self.assertTrue(np.array_equal(memo(full,query,1.0),expected))
            cases=[(full,query,.5),(full,query[[0]],1.0),(full.astype(np.float64),query,1.0),
                (full.copy(),query.copy(),1.0)]
            cases[-1][0][1,0]=1.01
            for points,queries,radius in cases:
                self.assertTrue(np.array_equal(original(points,queries,radius),memo(points,queries,radius)))
            after=runtime.profile();self.assertEqual(after['radius_memo_hits']-before['radius_memo_hits'],2)
            self.assertGreater(after['radius_tree_hits'],before['radius_tree_hits'])
            self.assertGreater(after['radius_live_key_bytes'],before['radius_live_key_bytes'])
            self.assertGreater(after['radius_live_tree_visible_array_bytes'],before['radius_live_tree_visible_array_bytes'])
        finally:memo.close()
        after=runtime.profile()
        for key in ('radius_live_pairs','radius_live_key_bytes','radius_live_result_bytes','radius_live_tree_visible_array_bytes'):
            self.assertEqual(after[key],before[key]-(1 if key=='radius_live_pairs' else 0))
        with self.assertRaisesRegex(RuntimeError,'already released'):memo(full,query,1.0)

    def test_tree_reuse_keeps_exact_kwargs_owned_data_and_releases_handles(self):
        original=self.original_radius();trees=[];calls=[]
        class ObservedTree:
            # cKDTree itself has no weakref slot. This CPU-only observation
            # wrapper delegates every actual original query unchanged.
            def __init__(self,value):self.value=value
            def __getattr__(self,name):return getattr(self.value,name)
        def tree(points,*args,**kwargs):
            calls.append((points.copy(),args,kwargs.copy()));result=runtime._SCIPY_TREE(points,*args,**kwargs)
            observed=ObservedTree(result);trees.append(weakref.ref(observed));return observed
        memo=runtime._PairRadius(original,tree);full=np.arange(30,dtype=np.float64).reshape(10,3)
        saved=full.copy()
        try:
            for query in (saved[[0]],saved[[3]],saved[[9]]):
                self.assertTrue(np.array_equal(original(saved,query,4.),memo(full,query,4.)))
            self.assertEqual(len(calls),1);self.assertEqual(calls[0][1],())
            self.assertEqual(calls[0][2],{'compact_nodes':True,'balanced_tree':True})
            full[:]=999
            self.assertTrue(np.array_equal(original(saved,saved[[1]],4.),memo(saved,saved[[1]],4.)))
        finally:memo.close()
        self.assertTrue(all(reference() is None for reference in trees))

    def test_empty_neighbors_and_original_invalid_inputs_keep_results_and_exceptions(self):
        original=self.original_radius();memo=runtime._PairRadius(original,runtime._SCIPY_TREE)
        full=np.array([[0,0,0],[1,1,1]],dtype=np.float32)
        cases=[(np.empty((0,3),np.float32),full,1.),(full,np.empty((0,3),np.float32),1.),
            (np.zeros((2,2),np.float32),full,1.),(np.array([[np.nan,0,0]],np.float32),full,1.),
            (full,np.zeros((2,2),np.float32),1.),(full,full,'invalid')]
        try:
            for args in cases:
                values=[];errors=[]
                for function in (original,memo):
                    try:values.append(function(*args))
                    except Exception as error:errors.append((type(error),str(error)))
                if errors:self.assertEqual(len(errors),2);self.assertEqual(errors[0],errors[1])
                else:self.assertTrue(np.array_equal(values[0],values[1]))
        finally:memo.close()

    def test_original_argument_binding_errors_and_nonfinite_radius_are_not_hidden(self):
        original=self.original_radius();memo=runtime._PairRadius(original,runtime._SCIPY_TREE)
        full=np.array([[0,0,0],[1,1,1]],dtype=np.float32)
        cases=[((full,),{}),((full,full,1.),{'radius_mm':1.}),
            ((full,full,1.),{'unknown':True}),((full,full,float('nan')),{}),
            ((full,full,float('inf')), {})]
        try:
            for args,kwargs in cases:
                errors=[];values=[]
                for function in (original,memo):
                    try:values.append(function(*args,**kwargs))
                    except Exception as error:errors.append((type(error),str(error)))
                if errors:self.assertEqual(len(errors),2);self.assertEqual(errors[0],errors[1])
                else:self.assertTrue(np.array_equal(values[0],values[1]))
        finally:memo.close()

    def test_scipy_identity_and_binary_stat_guard(self):
        import scipy.spatial
        with patch.object(scipy.spatial,'cKDTree',lambda *args,**kwargs:None):
            with self.assertRaisesRegex(ValueError,'SciPy'):runtime.runtime_contract()
        with patch.object(runtime,'_SCIPY_TREE_STAT',None):
            with self.assertRaisesRegex(ValueError,'SciPy'):runtime.runtime_contract()


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
