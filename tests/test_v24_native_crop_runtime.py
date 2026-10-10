"""CPU DEBUG: exact parent native crop method and real lossless codec parity."""
import ast
import copy
from pathlib import Path
import random
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hiercp_v1x import v24_native_crop_runtime as runtime
from hiercp_v1x import v24_lossless_raw_storage as storage
from custom_trainers.onlinecp_raw_resampling import apply_candidate,prepare_candidate
from tests import test_raw_cp_resampling_debug as native_fixtures


class NativeCropProtocolDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Compile only the unchanged actual method; constructor/framework setup
        # is outside this CPU operator test. The resampling engine is real.
        source=(runtime.ROOT/runtime.PARENT_FILE).read_text(encoding='utf8')
        original_tree=ast.parse(source)
        error=[node for node in original_tree.body if isinstance(node,ast.ClassDef) and node.name=='OnlineCPError'][0]
        method=runtime._method_tree(source).body[0]
        namespace={'np':np,'__name__':'original_native_parent_crop_CPU_DEBUG'}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[error,method],type_ignores=[])),
            str(runtime.ROOT/runtime.PARENT_FILE),'exec'),namespace)
        cls.original=staticmethod(namespace[runtime.METHOD])
        cls.error=namespace['OnlineCPError']

    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(prefix='v24_native_crop_DEBUG_',dir=runtime.ROOT)
        self.addCleanup(self.directory.cleanup);self.root=Path(self.directory.name)
        fixture=native_fixtures.RawCPResamplingDebugTests().fixture();self.case=fixture[-1]
        source=np.arange(125,dtype=np.float32).reshape(5,5,5)*12-650
        mask=np.zeros((5,5,5),bool);mask[1:4,1:4,1:4]=True
        self.candidate=prepare_candidate(self.case,source,mask,np.array([2,2,2]),(10,10,10))
        new_sha=storage.save_case(self.root,'compressed/case.json',self.case)
        old_sha=storage.original.save_case(self.root,'original/case.json',self.case)
        self.new_store=storage.LosslessRawBankStore(self.root)
        self.old_store=storage.original.RawBankStore(self.root)
        self.addCleanup(self.new_store.close);self.addCleanup(self.old_store.close)
        self.new=self.new_store.load_case('compressed/case.json',new_sha)
        self.old=self.old_store.load_case('original/case.json',old_sha)
        self.adapted=runtime._adapt_method(self.original,storage)

    def plan(self,case,scale=1.,shift=0.):
        return dict(raw_case=case,raw_candidate=self.candidate,scale=scale,shift_hu=shift,
            paste_contract='onlinecp_raw_target_paste_v1')

    def loader(self):return SimpleNamespace(online_bank=SimpleNamespace(raw_apply_function=lambda:apply_candidate))

    def crops(self,box):
        shape=np.asarray(self.case['metadata']['preprocessed_shape']);lower=np.asarray([pair[0] for pair in box])
        upper=np.asarray([pair[1] for pair in box]);valid_lower=np.maximum(lower,0);valid_upper=np.minimum(upper,shape)
        dimensions=upper-lower;data=np.zeros((1,*dimensions),np.float32);seg=np.full((1,*dimensions),-1,np.int16)
        original=tuple(slice(int(a),int(b)) for a,b in zip(valid_lower,valid_upper))
        target=tuple(slice(int(a-origin),int(b-origin)) for a,b,origin in zip(valid_lower,valid_upper,lower))
        seg[(slice(None),*target)]=self.case['baseline_seg'][(slice(None),*original)]
        return data,seg,lower.tolist()

    def test_actual_parent_NPY_vs_lossless_all_six_jitter_crop_outputs_and_audit_exact(self):
        shape=self.case['metadata']['preprocessed_shape']
        boxes=([[0,n] for n in shape],[[-2,9],[2,12],[3,14]])
        numpy_state=np.random.get_state();python_state=random.getstate();torch_state=torch.get_rng_state().clone()
        for box in boxes:
            for scale,shift in ((1.,0.),(1.3,350.),(.4,-550.)):
                with self.subTest(box=box,scale=scale,shift=shift):
                    old_loader=self.loader();new_loader=self.loader()
                    expected_data,expected_seg,lower=self.crops(box)
                    actual_data,actual_seg=expected_data.copy(),expected_seg.copy()
                    self.original(old_loader,expected_data,expected_seg,lower,self.plan(self.old,scale,shift),'DEBUG')
                    # Any accidental whole-volume conversion is a hard error.
                    with patch.object(storage._CropArray,'__array__',side_effect=AssertionError('whole-volume conversion')):
                        self.adapted(new_loader,actual_data,actual_seg,lower,self.plan(self.new,scale,shift),'DEBUG')
                    self.assertEqual(expected_data.tobytes(),actual_data.tobytes())
                    self.assertEqual(expected_seg.tobytes(),actual_seg.tobytes())
                    self.assertEqual(old_loader._last_raw_pasted_support.tobytes(),new_loader._last_raw_pasted_support.tobytes())
                    self.assertEqual(old_loader._last_raw_paste_audit,new_loader._last_raw_paste_audit)
                    self.assertGreater(new_loader._last_raw_paste_audit['raw_source_voxels'],0)
        self.assertEqual(python_state,random.getstate());self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        now=np.random.get_state();self.assertEqual(numpy_state[0],now[0]);self.assertTrue(np.array_equal(numpy_state[1],now[1]))
        self.assertEqual(numpy_state[2:],now[2:])

    def test_original_rejection_is_real_and_only_admitted_exact_proxy_is_extended(self):
        shape=self.case['metadata']['preprocessed_shape'];data,seg,lower=self.crops([[0,n] for n in shape])
        with self.assertRaisesRegex(self.error,'baseline segmentation'):
            self.original(self.loader(),data,seg,lower,self.plan(self.new),'DEBUG')
        self.assertTrue(runtime._admitted_reference(self.new['baseline_seg'],storage))
        self.assertFalse(runtime._admitted_reference(self.new['baseline_unclipped'],storage))
        spec=self.new['baseline_seg']._spec
        unadmitted=storage._CropArray(self.new_store,spec)
        self.assertFalse(runtime._admitted_reference(unadmitted,storage))
        class UnexpectedProxy(storage._CropArray):pass
        self.assertFalse(runtime._admitted_reference(UnexpectedProxy(self.new_store,spec),storage))
        self.assertFalse(runtime._admitted_reference(SimpleNamespace(shape=tuple([1,*shape]),dtype=np.dtype('int16')),storage))

    def test_wrong_actual_segmentation_shape_dtype_and_source_mutation_still_fail(self):
        shape=self.case['metadata']['preprocessed_shape'];box=[[0,n] for n in shape]
        data,seg,lower=self.crops(box);seg[0,0,0,0]=0
        with self.assertRaisesRegex(self.error,'segmentation differs'):
            self.adapted(self.loader(),data,seg,lower,self.plan(self.new),'DEBUG')
        for replacement in (self.new['baseline_unclipped'],np.zeros((1,2,3,4),np.int16),np.zeros((1,*shape),np.float32)):
            altered=dict(self.new,baseline_seg=replacement);data,seg,lower=self.crops(box)
            with self.assertRaisesRegex(self.error,'baseline segmentation'):
                self.adapted(self.loader(),data,seg,lower,self.plan(altered),'DEBUG')
        proxy=self.new['baseline_seg'];path=self.root/proxy._spec['path']
        with path.open('r+b') as stream:
            stream.seek(-1,2);byte=stream.read(1);stream.seek(-1,2);stream.write(bytes([byte[0]^1]))
        with self.assertRaisesRegex(ValueError,'changed after verification'):
            runtime._admitted_reference(proxy,storage)

    def test_only_one_predicate_changes_and_all_parent_equations_stay_exact(self):
        before=runtime._method_tree((runtime.ROOT/runtime.PARENT_FILE).read_text(encoding='utf8'))
        after=runtime._replace_predicate(before)
        count=0
        class Restore(ast.NodeTransformer):
            def visit_Call(self,node):
                nonlocal count
                if ast.dump(node)==ast.dump(runtime._REPLACEMENT):
                    count+=1;return copy.deepcopy(runtime._PREDICATE)
                return self.generic_visit(node)
        self.assertEqual(ast.dump(before),ast.dump(Restore().visit(after)));self.assertEqual(count,1)
        missing=copy.deepcopy(before)
        method=missing.body[0]
        method.body=[ast.Pass()]
        with self.assertRaisesRegex(ValueError,'Exactly one'):runtime._replace_predicate(missing)
        with self.assertRaisesRegex(ValueError,'source changed'):
            rejected=runtime._adapt_method(self.original,storage,source_guard=lambda:(_ for _ in ()).throw(ValueError('source changed')))
            data,seg,lower=self.crops([[0,n] for n in self.case['metadata']['preprocessed_shape']])
            rejected(self.loader(),data,seg,lower,self.plan(self.old),'DEBUG')

    def test_pure_contract_is_stable_and_does_not_install_or_import_native_framework(self):
        before=runtime.crop_protocol_contract()
        with patch.object(runtime.importlib,'import_module',side_effect=AssertionError('native import')):
            after=runtime.crop_protocol_contract()
        self.assertEqual(before,after);self.assertEqual(after['replaced_predicates'],1)
        self.assertFalse(after['parent_method_changed']);self.assertFalse(after['private_package_files_changed'])
        self.assertFalse(after['full_volume_conversion_enabled'])

    def test_install_overrides_only_frozen_loader_is_idempotent_and_rejects_foreign_override(self):
        # Framework-only metadata fixture: the installed method is still the
        # exact real parent algorithm exercised above, not a substitute paste.
        class ParentLoader:pass
        setattr(ParentLoader,runtime.METHOD,self.original)
        ParentLoader.__module__='native_parent_protocol_CPU_DEBUG'
        class FrozenLoader(ParentLoader):pass
        frozen=SimpleNamespace(__file__=runtime.ROOT/runtime.FROZEN_FILE,
            nnUNetDataLoaderOnlineCP=ParentLoader,FrozenV23Loader=FrozenLoader)
        parent=SimpleNamespace(__file__=runtime.ROOT/runtime.PARENT_FILE)
        modules={runtime.FROZEN_MODULE:frozen,ParentLoader.__module__:parent,
            'hiercp_v1x.v24_lossless_raw_storage':storage}
        original=ParentLoader.__dict__[runtime.METHOD]
        with patch.object(runtime.importlib,'import_module',side_effect=lambda name:modules[name]):
            contract=runtime.install_crop_protocol();self.assertEqual(contract,runtime.crop_protocol_contract())
            replacement=FrozenLoader.__dict__[runtime.METHOD]
            self.assertIs(ParentLoader.__dict__[runtime.METHOD],original)
            self.assertEqual(contract,runtime.install_crop_protocol())
            self.assertIs(FrozenLoader.__dict__[runtime.METHOD],replacement)
            FrozenLoader._apply_raw_paste_to_crop=lambda *args:None
            with self.assertRaisesRegex(ValueError,'override differs'):runtime.install_crop_protocol()
            del FrozenLoader._v24_crop_protocol_contract
            with self.assertRaisesRegex(ValueError,'unrecognized'):runtime.install_crop_protocol()

    def test_training_entry_preserves_exact_native_CLI_argument_tail(self):
        from nnunetv2.run import run_training as cli
        from hiercp_v1x import v24_native_best_eval_runtime as best_runtime
        tail=['730','3d_fullres','0','-tr','nnUNetTrainer_250epochs_FrozenV23CP','-p','nnUNetResEncUNetMPlans','--val_best']
        captured=[]
        def native_entry():captured.append(sys.argv[1:]);return 'original native CLI return'
        with (patch.object(runtime,'install_crop_protocol',return_value=runtime.crop_protocol_contract()) as install,
              patch.object(best_runtime,'install_best_evaluation_guard') as best_install,
              patch.object(sys,'argv',['-c',*tail]),patch.object(cli,'run_training_entry',side_effect=native_entry)):
            self.assertEqual(runtime.run_training_entry(),'original native CLI return')
        install.assert_called_once_with();best_install.assert_called_once_with();self.assertEqual(captured,[tail])


if __name__=='__main__':unittest.main()
