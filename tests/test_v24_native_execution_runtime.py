"""DEBUG: event-cache protocol and actual lossless native crop parity.

Protocol tests use explicitly marked decoder fixtures. Codec/native pipeline
tests require the real nnUNet and Blosc2 installation and never substitute a
decoder, raw resampling engine, loss or prediction when those are unavailable.
"""
import ast
import gc
import hashlib
import importlib.util
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

from hiercp_v1x import v24_native_execution_runtime as runtime
from hiercp_v1x import v24_native_crop_runtime as crop
from hiercp_v1x import v24_lossless_raw_storage as storage


class NativeCropReuseProtocolDebug(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='v24_crop_reuse_protocol_DEBUG_', dir=crop.ROOT)
        self.addCleanup(self.directory.cleanup); self.root = Path(self.directory.name)
        self.array = np.arange(2*7*8*9, dtype=np.int16).reshape(2,7,8,9)[:1].copy()
        path = self.root/'labels.npy'
        np.save(path, self.array, allow_pickle=False)
        self.store = storage.LosslessRawBankStore(self.root); self.addCleanup(self.store.close)
        self.spec = dict(path='labels.npy', sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         shape=list(self.array.shape), dtype=self.array.dtype.str, storage=storage.FORMAT)
        self.proxy = storage._CropArray(self.store, self.spec)
        self.store._cases[('protocol_DEBUG', self.spec['sha256'])] = ({}, {'baseline_seg':self.proxy})
        self.store._check(self.spec['path'], self.spec['sha256'])
        self.index = (0, slice(1,6), slice(2,7), slice(1,8))
        self.decodes = []
        def decoder_fixture(proxy, index):
            # Isolated ownership/key tests only. Real codec tests below do not
            # use this fixture, and no training output is produced here.
            proxy._store()._check(proxy._spec['path'], proxy._spec['sha256'])
            self.decodes.append((threading.get_ident(), index))
            result = self.array[index].copy(); result.flags.writeable = False
            return result
        self.base_patch = patch.object(runtime, '_BASE_GETITEM', decoder_fixture)
        self.base_patch.start(); self.addCleanup(self.base_patch.stop)
        self.get_patch = patch.object(storage._CropArray, '__getitem__', runtime._cached_getitem)
        self.get_patch.start(); self.addCleanup(self.get_patch.stop)

    def plan(self):return dict(raw_case=dict(baseline_seg=self.proxy))

    def scope(self, function, loader=None):
        loader = loader or SimpleNamespace()
        result = runtime.wrap_paste_method(function)(loader, None, None, None, self.plan(), 'protocol_DEBUG')
        return loader, result

    def test_exact_pair_deduplicates_decode_and_returns_independent_readonly_arrays(self):
        references = []
        def apply(loader, data, seg, bbox, plan, case):
            first = self.proxy[self.index]
            second = self.proxy[self.index]
            third = self.proxy[self.index]
            state = runtime._ACTIVE.get(); cached = next(iter(state['cache'].values()))
            references.append(weakref.ref(cached))
            self.assertFalse(np.shares_memory(first, second)); self.assertFalse(np.shares_memory(second, third))
            for array in (first,second,third):
                self.assertFalse(array.flags.writeable)
                self.assertEqual(array.tobytes(), self.array[self.index].tobytes())
            # A caller making its independent result writable cannot mutate
            # the cached snapshot or any other returned array.
            first.flags.writeable = True; first[...] = -2
            self.assertEqual(second.tobytes(), self.array[self.index].tobytes())
            self.assertEqual(third.tobytes(), self.array[self.index].tobytes())
            return second
        loader, result = self.scope(apply)
        self.assertEqual(len(self.decodes),1); self.assertEqual(loader._v24_native_crop_reuse_last['crop_hits'],2)
        self.assertEqual(loader._v24_native_crop_reuse_last['live_crop_bytes'],0)
        self.assertIsNone(runtime._ACTIVE.get()); gc.collect(); self.assertIsNone(references[0]())
        self.assertEqual(result.tobytes(), self.array[self.index].tobytes())

    def test_different_crop_and_other_proxy_never_hit_one_another(self):
        other = storage._CropArray(self.store, self.spec)
        def apply(loader, data, seg, bbox, plan, case):
            self.proxy[self.index]; self.proxy[(0,slice(0,4),slice(1,6),slice(1,8))]
            other[self.index]; self.proxy[self.index]
        loader, _ = self.scope(apply)
        self.assertEqual(len(self.decodes),3)
        self.assertEqual(loader._v24_native_crop_reuse_last['original_decodes'],2)
        self.assertEqual(loader._v24_native_crop_reuse_last['crop_hits'],1)
        self.proxy[self.index]
        self.assertEqual(len(self.decodes),4)

    def test_native_singleton_channel_int_and_full_slice_share_crop_with_original_rank(self):
        full = (slice(None), *self.index[1:])
        bounded = (slice(0,1,1), *self.index[1:])
        for indices in ((self.index, full, bounded), (full, self.index, bounded)):
            self.decodes.clear()
            def apply(loader,data,seg,bbox,plan,case):
                outputs = [self.proxy[index] for index in indices]
                for index, actual in zip(indices, outputs):
                    expected = self.array[index]
                    self.assertEqual(expected.shape,actual.shape)
                    self.assertEqual(expected.tobytes(),actual.tobytes())
                    self.assertFalse(actual.flags.writeable)
                for left in range(len(outputs)):
                    for right in range(left+1,len(outputs)):
                        self.assertFalse(np.shares_memory(outputs[left],outputs[right]))
                return outputs
            loader,_ = self.scope(apply)
            self.assertEqual(len(self.decodes),1)
            self.assertEqual(loader._v24_native_crop_reuse_last['crop_hits'],2)
        # Larger channel axes and other integer selectors retain distinct
        # keys and their original indexing semantics.
        self.assertNotEqual(runtime._crop_cache_key(self.index,(2,7,8,9))[0],
                            runtime._crop_cache_key(full,(2,7,8,9))[0])
        self.assertNotEqual(runtime._crop_cache_key((-1,*self.index[1:]),self.array.shape)[0],
                            runtime._crop_cache_key(full,self.array.shape)[0])

    def test_every_hit_rechecks_payload_and_changed_bytes_fail_with_scope_released(self):
        def apply(loader, data, seg, bbox, plan, case):
            self.proxy[self.index]
            # A size change guarantees a changed original stat witness even
            # on filesystems whose time tick coalesces very fast writes.
            with (self.root/'labels.npy').open('ab') as stream:
                stream.write(b'\x01')
            self.proxy[self.index]
        loader = SimpleNamespace()
        with self.assertRaisesRegex(ValueError,'changed after verification'):
            self.scope(apply, loader)
        self.assertIsNone(runtime._ACTIVE.get()); self.assertEqual(loader._v24_native_crop_reuse_last['failed_events'],1)
        self.assertEqual(loader._v24_native_crop_reuse_last['live_crop_bytes'],0)

    def test_exception_and_nested_scope_fail_explicitly_without_retaining_references(self):
        references = []
        def fail(loader, data, seg, bbox, plan, case):
            self.proxy[self.index]; references.append(weakref.ref(next(iter(runtime._ACTIVE.get()['cache'].values()))))
            raise RuntimeError('actual CP event DEBUG failure')
        loader=SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError,'CP event DEBUG failure'):self.scope(fail,loader)
        self.assertIsNone(runtime._ACTIVE.get()); gc.collect(); self.assertIsNone(references[0]())
        self.assertEqual(loader._v24_native_crop_reuse_last['closed_event_scopes'],1)
        def nested(loader,data,seg,bbox,plan,case):
            runtime.wrap_paste_method(fail)(loader,data,seg,bbox,plan,case)
        with self.assertRaisesRegex(ValueError,'Nested'):self.scope(nested)
        self.assertIsNone(runtime._ACTIVE.get())

    def test_thread_contexts_are_isolated_and_RNG_is_unchanged(self):
        barrier=threading.Barrier(2); results=[]; failures=[]
        py_state=random.getstate(); np_state=np.random.get_state(); torch_state=torch.get_rng_state().clone()
        def apply(loader,data,seg,bbox,plan,case):
            self.proxy[self.index]; barrier.wait(timeout=5); self.proxy[self.index]
        def worker():
            try:results.append(self.scope(apply)[0]._v24_native_crop_reuse_last)
            except BaseException as error:failures.append(error)
        threads=[threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:thread.start()
        for thread in threads:thread.join(6)
        self.assertFalse(failures); self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(self.decodes),2); self.assertEqual([item['crop_hits'] for item in results],[1,1])
        self.assertEqual(py_state,random.getstate()); self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        after=np.random.get_state(); self.assertEqual(np_state[0],after[0]);self.assertTrue(np.array_equal(np_state[1],after[1]));self.assertEqual(np_state[2:],after[2:])

    def test_changed_source_and_closed_proxy_fail_and_exact_index_is_conservative(self):
        source=self.root/'runtime_DEBUG.py';source.write_text('a=1\n')
        def apply(loader,data,seg,bbox,plan,case):return self.proxy[self.index]
        wrapped=runtime.wrap_paste_method(apply,source_proofs=[(source,runtime._stat(source))])
        source.write_text('a=123\n')
        with self.assertRaisesRegex(ValueError,'source changed'):wrapped(SimpleNamespace(),None,None,None,self.plan(),'DEBUG')
        self.assertIsNone(runtime._ACTIVE.get())
        for value in ((Ellipsis,), (0,slice(None,None,2)), (None,), (np.array([1,2]),), (0,)*5, (True,), (np.bool_(True),)):
            self.assertIsNone(runtime._basic_index(value,4))
        self.assertEqual(runtime._basic_index((np.int64(0),slice(np.int64(1),6)),4),
                         runtime._basic_index((0,slice(1,6)),4))

    def test_closed_proxy_and_changed_descriptor_after_first_read_fail_without_cache_leak(self):
        def closed(loader,data,seg,bbox,plan,case):
            self.proxy[self.index];self.proxy.close();self.proxy[self.index]
        loader=SimpleNamespace()
        with self.assertRaisesRegex(ValueError,'closed'):self.scope(closed,loader)
        self.assertEqual(loader._v24_native_crop_reuse_last['live_crop_bytes'],0);self.assertIsNone(runtime._ACTIVE.get())
        self.proxy._closed=False
        def changed(loader,data,seg,bbox,plan,case):
            self.proxy[self.index];self.proxy._spec['path']='another.npy';self.proxy[self.index]
        with self.assertRaisesRegex(ValueError,'descriptor changed'):self.scope(changed,loader)
        self.assertEqual(loader._v24_native_crop_reuse_last['live_crop_bytes'],0);self.assertIsNone(runtime._ACTIVE.get())

    def test_installer_is_cold_only_idempotent_and_rejects_foreign_override(self):
        def method(loader,data,seg,bbox,plan,case):return self.proxy[self.index]
        class Loader:pass
        frozen=SimpleNamespace(FrozenV23Loader=Loader)
        def install_crop():
            Loader._v24_crop_protocol_contract=crop.crop_protocol_contract()
            setattr(Loader,crop.METHOD,method);Loader._v24_crop_protocol_method=method
        with (patch.object(runtime.importlib,'import_module',return_value=frozen),
              patch.object(crop,'install_crop_protocol',side_effect=install_crop) as install,
              patch.object(torch.cuda,'is_initialized',return_value=False)):
            receipt=runtime.install_runtime();replacement=Loader.__dict__[crop.METHOD]
            self.assertEqual(receipt,runtime.install_runtime());self.assertIs(Loader.__dict__[crop.METHOD],replacement)
            self.assertEqual(install.call_count,1)
            setattr(Loader,crop.METHOD,lambda *args:None)
            with self.assertRaisesRegex(ValueError,'identity changed'):runtime.install_runtime()
        with patch.object(torch.cuda,'is_initialized',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'hot-swap'):runtime.install_runtime()

    def test_CPU_numerical_hash_covers_scalar_dtype_value_order_and_layout(self):
        from tools.probe_v24_native_execution_cuda_debug import _hash_cpu_tensors
        tensor=torch.arange(24,dtype=torch.float32).reshape(4,6).T
        state={'scalar':torch.tensor(1.),'array':tensor,'nested':{'ints':np.arange(5,dtype=np.int16)}}
        expected=_hash_cpu_tensors(state)
        self.assertEqual(expected,_hash_cpu_tensors(dict(reversed(list(state.items())))))
        self.assertEqual(expected,_hash_cpu_tensors(dict(state,array=tensor.contiguous())))
        self.assertNotEqual(expected,_hash_cpu_tensors(dict(state,scalar=torch.tensor(2.))))
        self.assertNotEqual(expected,_hash_cpu_tensors(dict(state,array=tensor.double())))

    def test_DEBUG_gate_optimizer_membership_and_loaded_checkpoint_binding_reject_corruption(self):
        from tools.probe_v24_native_execution_cuda_debug import (_hash_cpu_tensors,
            _loaded_checkpoint_binding,_require_unique_optimizer_membership)
        model=torch.nn.Linear(3,2);optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
        parameters=dict(model.named_parameters());_require_unique_optimizer_membership(parameters,optimizer)
        duplicate=SimpleNamespace(param_groups=[{'params':list(parameters.values())+list(parameters.values())[:1]}])
        with self.assertRaisesRegex(ValueError,'exactly once'):_require_unique_optimizer_membership(parameters,duplicate)
        class TrainerCheckpointProtocolDebug:pass
        trainer=TrainerCheckpointProtocolDebug();trainer.current_epoch=150
        initial={name:value.detach().clone() for name,value in model.state_dict().items()}
        saved=dict(network_weights=initial,optimizer_state=optimizer.state_dict(),grad_scaler_state={'scale':131072.},
                   current_epoch=150,trainer_name=type(trainer).__name__)
        actual_hash=_hash_cpu_tensors(optimizer.state_dict())
        proof=_loaded_checkpoint_binding(trainer,saved,initial,actual_hash,{'scale':131072.})
        self.assertTrue(proof['all_actual_loaded_states_match_source'])
        wrong_weights={name:value+1 for name,value in initial.items()}
        with self.assertRaisesRegex(ValueError,'model differs'):_loaded_checkpoint_binding(trainer,saved,wrong_weights,actual_hash,{'scale':131072.})
        with self.assertRaisesRegex(ValueError,'optimizer differs'):_loaded_checkpoint_binding(trainer,saved,initial,'0'*64,{'scale':131072.})
        with self.assertRaisesRegex(ValueError,'AMP scaler differs'):_loaded_checkpoint_binding(trainer,saved,initial,actual_hash,{'scale':65536.})


class NativeCropReuseActualLosslessDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec('nnunetv2') is None or importlib.util.find_spec('blosc2') is None:
            raise unittest.SkipTest('Actual nnUNet/Blosc2 required; no codec or pipeline substitute')
        from tests import test_raw_cp_resampling_debug as fixtures
        from custom_trainers.onlinecp_raw_resampling import prepare_candidate, apply_candidate
        cls.fixtures=fixtures;cls.prepare_candidate=staticmethod(prepare_candidate);cls.apply_candidate=staticmethod(apply_candidate)
        source=(crop.ROOT/crop.PARENT_FILE).read_text(encoding='utf8');tree=ast.parse(source)
        error=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='OnlineCPError'][0]
        method=crop._method_tree(source).body[0];namespace=dict(np=np,__name__='actual_native_crop_reuse_CPU_DEBUG')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[error,method],type_ignores=[])),str(crop.ROOT/crop.PARENT_FILE),'exec'),namespace)
        cls.original=staticmethod(namespace[crop.METHOD]);cls.adapted=staticmethod(crop._adapt_method(cls.original,storage))

    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(prefix='v24_crop_reuse_actual_DEBUG_',dir=crop.ROOT)
        self.addCleanup(self.directory.cleanup);self.root=Path(self.directory.name)
        self.case=self.fixtures.RawCPResamplingDebugTests().fixture()[-1]
        source=np.arange(125,dtype=np.float32).reshape(5,5,5)*12-650
        mask=np.zeros((5,5,5),bool);mask[1:4,1:4,1:4]=True
        self.candidate=self.prepare_candidate(self.case,source,mask,np.array([2,2,2]),(10,10,10))
        checksum=storage.save_case(self.root,'case.json',self.case)
        self.store=storage.LosslessRawBankStore(self.root);self.addCleanup(self.store.close)
        self.bound=self.store.load_case('case.json',checksum)
        self.wrapped=runtime.wrap_paste_method(self.adapted)

    def run_case(self, method, box, scale, shift):
        shape=np.asarray(self.case['metadata']['preprocessed_shape']);lower=np.asarray([pair[0] for pair in box]);upper=np.asarray([pair[1] for pair in box])
        lo=np.maximum(lower,0);hi=np.minimum(upper,shape);dimensions=upper-lower
        data=np.zeros((1,*dimensions),np.float32);seg=np.full((1,*dimensions),-1,np.int16)
        original=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));target=tuple(slice(int(a-origin),int(b-origin)) for a,b,origin in zip(lo,hi,lower))
        seg[(slice(None),*target)]=self.case['baseline_seg'][(slice(None),*original)]
        loader=SimpleNamespace(online_bank=SimpleNamespace(raw_apply_function=lambda:self.apply_candidate))
        plan=dict(raw_case=self.bound,raw_candidate=self.candidate,scale=scale,shift_hu=shift,paste_contract='onlinecp_raw_target_paste_v1')
        method(loader,data,seg,lower.tolist(),plan,'actual_DEBUG')
        return loader,data,seg

    def test_actual_native_engine_crop_support_audit_and_RNG_exact_all_jitters(self):
        shape=self.case['metadata']['preprocessed_shape'];boxes=([[0,n] for n in shape],[[-2,9],[2,12],[3,14]])
        py=random.getstate();numpy=np.random.get_state();torch_rng=torch.get_rng_state().clone()
        for box in boxes:
            for scale,shift in ((1.,0.),(1.3,350.),(.4,-550.)):
                with self.subTest(box=box,scale=scale,shift=shift):
                    expected,data,seg=self.run_case(self.adapted,box,scale,shift)
                    with patch.object(storage._CropArray,'__getitem__',runtime._cached_getitem):
                        actual,new_data,new_seg=self.run_case(self.wrapped,box,scale,shift)
                    self.assertEqual(data.tobytes(),new_data.tobytes());self.assertEqual(seg.tobytes(),new_seg.tobytes())
                    self.assertEqual(expected._last_raw_pasted_support.tobytes(),actual._last_raw_pasted_support.tobytes())
                    self.assertEqual(expected._last_raw_paste_audit,actual._last_raw_paste_audit)
                    self.assertEqual(actual._v24_native_crop_reuse_last['original_decodes'],1)
                    self.assertEqual(actual._v24_native_crop_reuse_last['crop_hits'],1)
                    self.assertEqual(actual._v24_native_crop_reuse_last['live_crop_bytes'],0)
        self.assertEqual(py,random.getstate());self.assertTrue(torch.equal(torch_rng,torch.get_rng_state()))
        after=np.random.get_state();self.assertEqual(numpy[0],after[0]);self.assertTrue(np.array_equal(numpy[1],after[1]));self.assertEqual(numpy[2:],after[2:])

    def test_actual_outputs_loss_all_gradients_and_optimizer_update_exact_CPU_DEBUG(self):
        # CPU DEBUG downstream connectivity test, explicitly separate from
        # the full 102M-parameter native CUDA admission probe.
        shape=self.case['metadata']['preprocessed_shape'];box=[[0,n] for n in shape]
        _,expected_data,expected_seg=self.run_case(self.adapted,box,1.3,350.)
        with patch.object(storage._CropArray,'__getitem__',runtime._cached_getitem):
            _,actual_data,actual_seg=self.run_case(self.wrapped,box,1.3,350.)
        model=torch.nn.Conv3d(1,3,3,padding=1);other=torch.nn.Conv3d(1,3,3,padding=1);other.load_state_dict(model.state_dict())
        optimizers=[torch.optim.SGD(module.parameters(),lr=.01,momentum=.9) for module in (model,other)]
        results=[]
        for module,optimizer,data,seg in zip((model,other),optimizers,(expected_data,actual_data),(expected_seg,actual_seg)):
            output=module(torch.from_numpy(data[None]));loss=torch.nn.functional.cross_entropy(output,torch.from_numpy(seg[None,0]).long())
            loss.backward();gradients={name:parameter.grad.detach().clone() for name,parameter in module.named_parameters()};optimizer.step()
            results.append((output.detach(),loss.detach(),gradients,{name:parameter.detach().clone() for name,parameter in module.named_parameters()}))
        for left,right in zip(results[0][:2],results[1][:2]):self.assertTrue(torch.equal(left,right))
        for position in (2,3):
            self.assertEqual(set(results[0][position]),set(results[1][position]))
            for name in results[0][position]:self.assertTrue(torch.equal(results[0][position][name],results[1][position][name]))


if __name__=='__main__':unittest.main()
