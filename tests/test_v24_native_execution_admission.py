"""DEBUG admission guards with explicitly small isolated numerical fixtures.

These exercise byte/schema/source failure handling. They do not represent the
full native model, real patients, CUDA admission or final experiment results.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools import run_v24_native_execution_admission as gate


class NativeActualTensorAdmissionDebug(unittest.TestCase):
    def fixture(self):
        return dict(inputs=dict(data=torch.arange(24,dtype=torch.float32).reshape(2,1,2,2,3),
                                metadata=['explicit_DEBUG_fixture',42],flags=np.array([True,False])),
                    outputs=[torch.tensor([1.,-2.])],loss=torch.tensor(0.5),
                    gradients={'debug.weight':torch.ones((2,3))},
                    initial_model={'debug.weight':torch.ones((2,3))},
                    final_model={'debug.weight':torch.ones((2,3))},
                    rng_before=dict(torch=torch.tensor([1,2,3],dtype=torch.uint8),
                                    numpy=('MT19937',np.array([1,2],dtype=np.uint32),2,0,0.)),
                    rng_after=dict(python=(3,(1,2,3),None)))

    def gradient_fixture(self):
        numerical=self.fixture()
        numerical['initial_model']['debug.inactive']=torch.zeros(1)
        report=dict(trainable_parameter_names=['debug.weight','debug.inactive'],
            trainable_parameter_schema={'debug.weight':dict(shape=[2,3],dtype='torch.float32',numel=6),
                                        'debug.inactive':dict(shape=[1],dtype='torch.float32',numel=1)},
            optimizer_parameter_names=[['debug.weight','debug.inactive']],
            inactive_head_proof=dict(official_inactive_parameter_names=['debug.inactive']),
            missing_parameter_names=['debug.inactive'],gradient_tensors=1,trainable_parameter_count=7)
        return report,numerical

    def declared_contract_fixture(self):
        # Metadata-only guard fixture, never passed to full admission and
        # never represented as a real full native tensor/model result.
        return dict(format=gate.PROBE_FORMAT,debug=True,mode='baseline',physical_batch=2,epochs=250,
            actual_input_shape=[2,1,128,128,128],full105_train=True,ordinary26_validation=True,
            original_production_augmentation_workers=4,original_validation_workers=2,
            trainable_parameter_count=102350575,optimizer_trainable_parameters_exactly_once=True,
            optimizer_state_unchanged=True,AMP_scaler_state_unchanged=True,optimizer_updates=0,
            production_training_performed=False,CUDA_deterministic_DEBUG_only=True,
            production_numerics_changed=False,explicit_DEBUG_seed=42,original_ONLINE_CP_SEED=42,
            exact_future_original_production_RNG_recovery_claimed=False,batch_position=0,
            CP_flags=[True,False],input_keys=['DEBUG_A','DEBUG_B'],source_epoch=126,
            initial_model_sha256='a'*64,initial_optimizer_sha256='b'*64,final_optimizer_sha256='b'*64,
            initial_scaler={'scale':65536.},final_scaler={'scale':65536.},
            actual_loaded_checkpoint_binding=dict(all_actual_loaded_states_match_source=True,
                network_weights_sha256='a'*64,optimizer_state_sha256='b'*64,grad_scaler_state={'scale':65536.},
                current_epoch=126,trainer_name='nnUNetTrainer_250epochs_FrozenV23CP'))

    def test_actual_tensor_numpy_and_RNG_bytes_match_independent_fixture_copies(self):
        left=self.fixture();right=copy.deepcopy(left)
        result=gate._equal_leaves(left,right)
        self.assertGreater(result['torch_tensors'],5);self.assertEqual(result['numpy_arrays'],2)
        self.assertGreater(result['tensor_bytes'],24*4)

    def test_real_Module_state_dict_source_OrderedDict_matches_owned_plain_clones_and_tamper_fails(self):
        module=torch.nn.Linear(2,3)
        original=module.state_dict()
        clone={name:value.detach().clone() for name,value in original.items()}
        self.assertIsNot(type(original),type(clone))
        gate._equal_leaves(dict(original),clone,'actual_source_checkpoint_weights')
        clone['weight'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'tensor bytes differ'):
            gate._equal_leaves(dict(original),clone,'actual_source_checkpoint_weights')
        # Paired probe artifacts retain their declared exact dict schema.
        with self.assertRaisesRegex(ValueError,'Dictionary schema differs'):
            gate._equal_leaves(original,{name:value.detach().clone() for name,value in original.items()})

    def test_changed_input_loss_output_gradient_model_and_RNG_bytes_fail(self):
        for field in ('inputs','outputs','loss','gradients','initial_model','final_model','rng_before','rng_after'):
            with self.subTest(field=field):
                left=self.fixture();right=copy.deepcopy(left)
                if field=='inputs':right[field]['data'].reshape(-1)[0]+=1
                elif field=='outputs':right[field][0][0]+=1
                elif field=='loss':right[field]+=1
                elif field in ('gradients','initial_model','final_model'):right[field]['debug.weight'][0,0]+=1
                elif field=='rng_before':right[field]['numpy'][1][0]+=1
                else:right[field]['python']=(3,(1,2,4),None)
                with self.assertRaisesRegex(ValueError,'bytes differ|leaf differs'):gate._equal_leaves(left,right)

    def test_dtype_shape_scalar_and_signed_zero_are_byte_exact(self):
        for first,second in ((torch.tensor(1.),torch.tensor(1.,dtype=torch.float64)),
                             (torch.ones((2,3)),torch.ones((3,2))),
                             (torch.tensor(0.),torch.tensor(-0.)),
                             (np.array([1],dtype=np.int16),np.array([1],dtype=np.int32)),
                             ([1,2],(1,2))):
            with self.subTest(first=str(first)):
                with self.assertRaises(ValueError):gate._equal_leaves(first,second)
        gate._equal_leaves(torch.arange(6).reshape(2,3).t(),torch.arange(6).reshape(2,3).t().contiguous())

    def test_nonfinite_and_object_array_leaves_fail(self):
        for value in (torch.tensor(float('nan')),torch.tensor(float('inf')),
                      np.array([np.inf]),np.array(['DEBUG'],dtype=object)):
            with self.subTest(value=str(value)):
                with self.assertRaisesRegex(ValueError,'Nonfinite|object data'):gate._equal_leaves(value,copy.deepcopy(value))

    def test_RNG_Python_and_NumPy_scalar_signed_zero_is_byte_exact_and_nonfinite_fails(self):
        for first,second in (({'cached_gaussian':0.},{'cached_gaussian':-0.}),
                             (np.float64(0.),np.float64(-0.)),(np.float32(0.),np.float32(-0.))):
            with self.assertRaisesRegex(ValueError,'scalar bytes differ'):gate._equal_leaves(first,second,'actual_RNG')
        for value in (float('inf'),float('nan'),np.float64(np.inf),np.float32(np.nan)):
            with self.assertRaisesRegex(ValueError,'Nonfinite'):gate._equal_leaves(value,copy.deepcopy(value))

    def test_missing_or_extra_scientific_source_entry_cannot_be_admitted(self):
        names=['explicit_DEBUG_source_a.py','explicit_DEBUG_source_b.py']
        original=dict(source_files_sha256={name:'a'*64 for name in names})
        gate._scientific_source_contract(original,names)
        for changed in (dict(source_files_sha256={names[0]:'a'*64}),
                        dict(source_files_sha256={**original['source_files_sha256'],'extra.py':'b'*64})):
            with self.assertRaisesRegex(ValueError,'complete original scientific'):gate._scientific_source_contract(changed,names)

    def test_missing_unknown_or_duplicate_gradient_and_optimizer_membership_fail(self):
        report,numerical=self.gradient_fixture();gate._gradient_contract(report,numerical)
        cases=[]
        bad=copy.deepcopy(numerical);bad['gradients'].clear();cases.append((report,bad))
        bad=copy.deepcopy(numerical);bad['gradients']['other']=torch.ones(1);cases.append((report,bad))
        bad=copy.deepcopy(report);bad['optimizer_parameter_names'][0].append('debug.weight');cases.append((bad,numerical))
        bad=copy.deepcopy(report);bad['inactive_head_proof']['official_inactive_parameter_names']=[];cases.append((bad,numerical))
        bad=copy.deepcopy(report);bad['missing_parameter_names']=[];cases.append((bad,numerical))
        for candidate,values in cases:
            with self.assertRaises(ValueError):gate._gradient_contract(candidate,values)

    def test_actual_gradient_schema_and_finite_values_are_checked(self):
        report,numerical=self.gradient_fixture()
        for value in (torch.ones(6),torch.ones((2,3),dtype=torch.float64),torch.full((2,3),float('inf'))):
            changed=copy.deepcopy(numerical);changed['gradients']['debug.weight']=value
            with self.assertRaisesRegex(ValueError,'gradient shape/dtype/finite'):gate._gradient_contract(report,changed)

    def test_SHA_stat_guard_rejects_stale_tampered_and_nonregular_artifacts(self):
        with tempfile.TemporaryDirectory(prefix='v24_actual_admission_DEBUG_',dir=gate.ROOT) as folder:
            path=Path(folder)/'DEBUG_only.bin';path.write_bytes(b'DEBUG fixture actual bytes')
            proof=gate._guard(path);gate._recheck(proof)
            with self.assertRaisesRegex(ValueError,'SHA/stat'):gate._guard(path,'0'*64)
            with path.open('ab') as stream:stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError,'changed'):gate._recheck(proof)
            with self.assertRaisesRegex(ValueError,'regular'):gate._guard(Path(folder))

    def test_reports_cannot_claim_full_native_contract_with_small_fixture_or_missing_proofs(self):
        for report in ({},dict(format=gate.PROBE_FORMAT,debug=True,mode='baseline',
                              physical_batch=1,epochs=1,trainable_parameter_count=7)):
            with self.assertRaisesRegex(ValueError,'full native'):gate._report_contract(report,'baseline')

    def test_declared_contract_rejects_scale_seed_checkpoint_and_scaler_corruption(self):
        original=self.declared_contract_fixture();gate._report_contract(original,'baseline')
        for name,value in (('physical_batch',1),('epochs',1),('full105_train',False),('explicit_DEBUG_seed',0),
                           ('optimizer_updates',1),('source_epoch',True),('final_optimizer_sha256','c'*64),
                           ('final_scaler',{'scale':32768.}),('CP_flags',[False,False])):
            changed=copy.deepcopy(original);changed[name]=value
            with self.subTest(field=name),self.assertRaises(ValueError):gate._report_contract(changed,'baseline')
        for name,value in (('network_weights_sha256','c'*64),('optimizer_state_sha256','c'*64),
                           ('current_epoch',125),('grad_scaler_state',{'scale':32768.}),
                           ('all_actual_loaded_states_match_source',False)):
            changed=copy.deepcopy(original);changed['actual_loaded_checkpoint_binding'][name]=value
            with self.subTest(binding=name),self.assertRaises(ValueError):gate._report_contract(changed,'baseline')

    def test_foreign_owner_file_is_rejected_with_explicit_UID_DEBUG_fixture(self):
        with tempfile.TemporaryDirectory(prefix='v24_actual_admission_DEBUG_',dir=gate.ROOT) as folder:
            path=Path(folder)/'DEBUG_uid.bin';path.write_bytes(b'DEBUG own file')
            with patch.object(gate.os,'getuid',return_value=path.lstat().st_uid+1,create=True):
                with self.assertRaisesRegex(ValueError,'Owned regular'):gate._guard(path)

    def test_small_or_metadata_mismatched_input_is_not_admitted_as_full_native(self):
        values=self.fixture();values['inputs'].update(keys=['DEBUG_A','DEBUG_B'],online_cp_applied=np.array([True,False]))
        report=dict(input_keys=['DEBUG_A','DEBUG_B'],CP_flags=[True,False])
        with self.assertRaisesRegex(ValueError,'full native input'):gate._actual_batch_contract(report,values)

    def test_existing_output_and_identical_report_are_rejected_before_loading(self):
        with tempfile.TemporaryDirectory(prefix='v24_actual_admission_DEBUG_',dir=gate.ROOT) as folder:
            path=Path(folder)/'report.json';path.write_text(json.dumps(dict(debug=True)),encoding='utf8')
            with self.assertRaises(FileExistsError):gate.admit(path,path,path)
            with self.assertRaisesRegex(ValueError,'Distinct'):gate.admit(path,path,Path(folder)/'fresh.json')


if __name__=='__main__':unittest.main()
