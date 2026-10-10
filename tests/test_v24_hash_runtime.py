"""CPU UNIT parity of every input byte; these are not training/performance claims."""
from __future__ import annotations

import copy
import importlib
import random
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch

from hiercp_v1x import v24_hash_runtime as runtime


@contextmanager
def restored_references():
    modules=[importlib.import_module(name) for name in runtime.REFERENCE_MODULES]
    memory=importlib.import_module('hiercp_v1x.v24_memory_runtime')
    refs=[(vars(module),'tensor_digest',vars(module)['tensor_digest'])
        for module in modules if 'tensor_digest' in vars(module)]
    refs.append((memory._GET.__globals__,'tensor_digest',memory._GET.__globals__['tensor_digest']))
    try:yield modules,memory
    finally:
        for namespace,key,value in refs:namespace[key]=value


class ExactHashRuntimeDebug(unittest.TestCase):
    def assert_digest_parity(self,value):
        self.assertEqual(runtime._ORIGINAL_DIGEST(value),runtime.tensor_digest(value))

    def test_tensor_dtypes_empty_scalar_and_shapes(self):
        for dtype in (torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,
                      torch.int64,torch.float16,torch.bfloat16,torch.float32,
                      torch.float64,torch.complex64,torch.complex128):
            for shape in ((),(0,),(2,0,3),(2,3,4)):
                count=int(np.prod(shape)) if shape else 1
                tensor=torch.arange(count,dtype=torch.int64).to(dtype).reshape(shape)
                with self.subTest(dtype=dtype,shape=shape):self.assert_digest_parity(tensor)

    def test_strides_storage_offset_expansion_and_aliases(self):
        original=torch.arange(120,dtype=torch.float64).reshape(5,6,4)
        values=(original.transpose(0,2),original[1:4,::2,1:],original[:,0:1,:].expand(5,9,4),
                original.flatten().as_strided((8,8),(1,2),3),original[2,3,1])
        for value in values:self.assert_digest_parity(value)
        self.assert_digest_parity({'alias1':original,'alias2':original[:,::2],'list':[values[0],values[0]]})

    def test_nan_payloads_signed_zero_and_inf_are_raw_bytes(self):
        bits=np.array([0x7fc00001,0x7fc12345,0xffc00001,0,0x80000000,0x7f800000,0xff800000],dtype=np.uint32)
        value=torch.from_numpy(bits.view(np.float32))
        self.assert_digest_parity(value)
        first=runtime.tensor_digest(value)
        bits[0]=0x7fc00002
        self.assertNotEqual(first,runtime.tensor_digest(value));self.assert_digest_parity(value)
        self.assertNotEqual(runtime.tensor_digest(torch.tensor([0.])),runtime.tensor_digest(torch.tensor([-0.])))

    def test_conjugate_and_negative_view_logical_bytes(self):
        complex_value=torch.tensor([1+2j,3-4j],dtype=torch.complex128)
        self.assertTrue(complex_value.conj().is_conj())
        self.assert_digest_parity(complex_value.conj())
        negative=torch._neg_view(torch.arange(8,dtype=torch.float64))
        self.assertTrue(negative.is_neg());self.assert_digest_parity(negative)
        self.assert_digest_parity(torch._neg_view(complex_value.conj()))

    def test_nested_numpy_and_mixed_keys_original_order(self):
        numpy=np.arange(60,dtype='>i4').reshape(5,12)[:,::3]
        value={3:('row',numpy,np.zeros((2,0),dtype=np.float64)),None:[True,False,None],
               '3':{'tensor':torch.tensor([1.,-0.]),'strings':np.array(['a','한글'])},
               (1,'k'):[np.arange(8,dtype=np.float16)[::-1],{'nested':5.5}]}
        self.assert_digest_parity(value)
        self.assertEqual(runtime.tensor_digest(value),runtime.tensor_digest(dict(reversed(list(value.items())))))
        self.assertNotEqual(runtime.tensor_digest([1,2]),runtime.tensor_digest((1,2)))

    def test_numpy_scalar_readonly_structured_object_endian_and_datetime(self):
        object_value=np.array([object(),None,'marker'],dtype=object)
        mixed=np.array([(object(),'2020-01-01')],dtype=[('object','O'),('date','datetime64[D]')])
        values=(np.array(-0.,dtype=np.float64),np.zeros((3,0,2),dtype='>f8'),
            np.array(['2020-01-01','2021-07-08'],dtype='datetime64[D]'),
            np.array([1,3],dtype='timedelta64[ms]'),object_value,mixed,
            np.array([(1,2.),(7,-0.)],dtype=[('int','>i4'),('float','<f8')]),
            np.array(['a','한글']),np.array([b'alpha',b'beta']))
        for array in values:
            array.flags.writeable=False
            with self.subTest(dtype=array.dtype,shape=array.shape):self.assert_digest_parity(array)
        first=runtime.tensor_digest(object_value);object_value.flags.writeable=True
        object_value[0]=object();self.assertNotEqual(first,runtime.tensor_digest(object_value))
        self.assertIs(runtime.inputs.array_digest,runtime._ORIGINAL_ARRAY_DIGEST)

    def test_mutation_detected_without_version_or_hash_memo(self):
        tensor=torch.arange(24,dtype=torch.float32).reshape(4,6)
        first=runtime.tensor_digest(tensor);tensor[0,0]=99
        self.assertNotEqual(first,runtime.tensor_digest(tensor));self.assert_digest_parity(tensor)
        alias=tensor.numpy();version=tensor._version;second=runtime.tensor_digest(tensor)
        alias[1,1]=-13
        self.assertEqual(version,tensor._version)
        self.assertNotEqual(second,runtime.tensor_digest(tensor));self.assert_digest_parity(tensor)
        array=np.arange(20,dtype=np.int16);expected=runtime.tensor_digest(array);array[4]=7
        self.assertNotEqual(expected,runtime.tensor_digest(array));self.assert_digest_parity(array)

    def test_hash_has_no_data_gradient_or_rng_side_effect(self):
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        python_state=random.getstate();numpy_state=copy.deepcopy(np.random.get_state());torch_state=torch.get_rng_state().clone()
        tensor=torch.randn(7,9,requires_grad=True);torch_state=torch.get_rng_state().clone()
        contents=tensor.detach().clone();version=tensor._version;pointer=tensor.data_ptr()
        self.assert_digest_parity({'tensor':tensor,'view':tensor.T})
        self.assertTrue(torch.equal(contents,tensor));self.assertEqual(version,tensor._version)
        self.assertEqual(pointer,tensor.data_ptr());self.assertIsNone(tensor.grad)
        self.assertEqual(python_state,random.getstate())
        actual=np.random.get_state();self.assertEqual(numpy_state[0],actual[0])
        np.testing.assert_array_equal(numpy_state[1],actual[1]);self.assertEqual(numpy_state[2:],actual[2:])
        self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))

    def test_original_error_semantics_preserved(self):
        for value in (object(),{'x':float('nan')},float('inf'),{object():1}):
            with self.subTest(value=type(value).__name__):
                try:runtime._ORIGINAL_DIGEST(value)
                except Exception as original_error:
                    with self.assertRaises(type(original_error)):runtime.tensor_digest(value)
                else:self.fail('Expected unsupported original input')

    def test_actual_source_code_and_no_copy_contract(self):
        contract=runtime.runtime_contract()
        self.assertTrue(contract['original_function_code_matches_unchanged_source'])
        self.assertTrue(contract['every_actual_tensor_value_hashed_each_call'])
        self.assertFalse(contract['hash_result_memoization'])
        # The eliminated original flat allocation is guarded mechanically.
        with patch.object(torch,'empty',side_effect=AssertionError('redundant tensor allocation')):
            runtime.tensor_digest(torch.arange(64,dtype=torch.float32))

    def test_known_modules_and_private_memory_reference_install_idempotently(self):
        with restored_references() as (modules,memory):
            first=runtime.install();second=runtime.install()
            for module in modules:
                if 'tensor_digest' in vars(module):self.assertIs(module.tensor_digest,runtime.tensor_digest)
            self.assertIs(memory._GET.__globals__['tensor_digest'],runtime.tensor_digest)
            self.assertEqual(first,second)
            private=first['installed_private_global_references'][0]
            self.assertTrue(private['private_get_code_reconstructed_from_original'])
            self.assertFalse(private['private_get_AST_proof']['tensor_operations_changed'])

    def test_foreign_module_reference_refused_before_any_partial_install(self):
        with restored_references() as (modules,memory):
            upper=sys.modules['hiercp_v1x.v24_upper_reuse'];upper.tensor_digest=lambda value:'foreign'
            before=runtime.inputs.tensor_digest
            with self.assertRaisesRegex(ValueError,'Foreign imported'):runtime.install()
            self.assertIs(runtime.inputs.tensor_digest,before)
            self.assertIs(memory._GET.__globals__['tensor_digest'],runtime._ORIGINAL_DIGEST)

    def test_private_memory_foreign_reference_refused_before_install(self):
        with restored_references() as (modules,memory):
            memory._GET.__globals__['tensor_digest']=lambda value:'foreign'
            before=runtime.inputs.tensor_digest
            with self.assertRaisesRegex(ValueError,'Foreign private'):runtime.install()
            self.assertIs(runtime.inputs.tensor_digest,before)

    def test_source_identity_change_is_explicit_failure(self):
        original=runtime._stat;target=str(runtime._SOURCE_PATHS[1])
        def changed(path):
            value=original(path)
            return value[:-1]+(value[-1]+1,) if str(path)==target else value
        with patch.object(runtime,'_stat',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'source changed'):runtime.tensor_digest(torch.arange(3))
            with self.assertRaisesRegex(ValueError,'source changed'):runtime.install()

    def test_initialized_cuda_process_cannot_be_hot_swapped(self):
        before=runtime.inputs.tensor_digest
        with patch.object(torch.cuda,'is_initialized',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'precede CUDA'):runtime.install()
        self.assertIs(runtime.inputs.tensor_digest,before)


class ActualOriginalCPUInputHashDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_v24_provider import V24CPUProviderDebug
        V24CPUProviderDebug.setUpClass();cls.fixture=V24CPUProviderDebug

    @classmethod
    def tearDownClass(cls):cls.fixture.tearDownClass()

    def test_actual_canonical_upper_and_two_view_batches_match_every_byte(self):
        fixture=self.fixture.fixture
        payloads=[{key:record[key] for key in ('source_patch','target_patch','source_local','target_local')}
            for record in fixture.records]
        expected=[runtime._ORIGINAL_DIGEST(value) for value in payloads]
        self.assertEqual(expected,[runtime.tensor_digest(value) for value in payloads])
        self.assertEqual(expected,[record['tensor_sha256'] for record in fixture.records])
        self.assertEqual(runtime._ORIGINAL_DIGEST((fixture.graph.to_dict(),fixture.prototype.to_dict())),
                         runtime.tensor_digest((fixture.graph.to_dict(),fixture.prototype.to_dict())))
        helper=self.fixture(methodName='runTest');provider=helper.provider('inner_train')
        try:
            original=provider.get([0,1,2],epoch=29)
            baseline=runtime._ORIGINAL_DIGEST((original.graph.to_dict(),original.source_patches,
                original.target_patches,original.source_index,original.graph_observation_index,original.indices))
            rng=torch.get_rng_state().clone()
            with restored_references():
                runtime.install();adapted=provider.get([0,1,2],epoch=29)
                actual=runtime.tensor_digest((adapted.graph.to_dict(),adapted.source_patches,
                    adapted.target_patches,adapted.source_index,adapted.graph_observation_index,adapted.indices))
                self.assertEqual(baseline,actual)
                self.assertEqual(runtime._ORIGINAL_DIGEST(adapted.graph.to_dict()),runtime.tensor_digest(adapted.graph.to_dict()))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(provider.profile()['sampled_pairs_materialized'],6)
        finally:provider.close()


if __name__=='__main__':unittest.main()
