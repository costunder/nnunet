"""Mechanical UNIT contracts only; no claimed patient/model training."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io
import numpy as np
from hiercp_v1x import v24_nnunet_cp as cp


def split_fixture():
    train=[f'UNIT_train_{i}' for i in range(84)]
    val=[f'UNIT_inner_{i}' for i in range(21)]
    return dict(inner_train=train,inner_val=val,outer_train=train+val,
                outer_val=[f'UNIT_outer_{i}' for i in range(26)])


def pin_fixture():
    rows=[]
    for gpu,score in ((1,.3),(5,.4),(6,.2)):
        rows.append(dict(gpu=gpu,best_path=f'/UNIT_gpu{gpu}/training/checkpoint_best.pt',
            best_file_sha256='a'*64,best_content_sha256='b'*64,best_model_sha256='c'*64,identity_sha256='d'*64,
            best_record=dict(epoch=15,updates=178,selection_key=[score,.2,-.4],
                candidate_universe='all native P + fixed128U',selected_by='full_validation_only')))
    return dict(format=cp.PIN_FORMAT,all_three_paused=True,recipient_annotation_exposed=True,
        selection_metric='fixed full128 per-P patient-macro MRR, top1, negativepairloss',
        candidates=rows,selected=copy.deepcopy(rows[1]))


class NativeContracts(unittest.TestCase):
    def test_complete_nested_split(self):
        split=split_fixture()
        result=cp.validate_split(split)
        self.assertEqual(len(result['outer_train']),105)
        result['inner_train'][0]='different'
        self.assertNotEqual(result,split)

    def test_held_out_recipient_leakage_rejected(self):
        split=split_fixture();split['outer_val'][0]=split['outer_train'][0]
        with self.assertRaisesRegex(ValueError,'leakage'):cp.validate_split(split)

    def test_incomplete_dataset_rejected(self):
        split=split_fixture();split['inner_train'].pop()
        with self.assertRaises(ValueError):cp.validate_split(split)

    def test_best_pin_recomputes_comparison_winner(self):
        pin=pin_fixture();self.assertEqual(cp.validate_pin(pin)['selected']['gpu'],5)
        pin['selected']=pin['candidates'][0]
        with self.assertRaisesRegex(ValueError,'actual best'):cp.validate_pin(pin)

    def test_stage_validation_and_unpaused_pin_rejected(self):
        for key,value in (('all_three_paused',False),('recipient_annotation_exposed',False)):
            pin=pin_fixture();pin[key]=value
            with self.assertRaises(ValueError):cp.validate_pin(pin)
        pin=pin_fixture();pin['candidates'][0]['best_record']['selected_by']='stage_validation'
        with self.assertRaises(ValueError):cp.validate_pin(pin)

    def test_nonfinite_selection_key_rejected(self):
        pin=pin_fixture();pin['candidates'][0]['best_record']['selection_key'][0]=float('nan')
        with self.assertRaises(ValueError):cp.validate_pin(pin)

    def test_joint_P_retained_U_bank_mapping(self):
        bank=[f'UNIT_U_{i}' for i in range(128)]
        rows=[dict(id='UNIT_P',center=[1,1,1])]+[dict(id=name,center=[i+2,3,4]) for i,name in enumerate(bank)]
        plan=SimpleNamespace(record_ids=('UNIT_P',*bank),rows=rows,bank_record_ids=tuple(reversed(bank)),
            unobserved_indices=tuple(range(1,129)),unobserved_bank_positions=tuple(reversed(range(128))))
        values,centers=cp.bank_u_values(np.arange(129),plan)
        self.assertEqual(values.tolist(),list(range(128,0,-1)))
        self.assertEqual(centers[0].tolist(),[129,3,4])
        self.assertNotIn(0,values)

    def test_nonfinite_or_incomplete_CP_scores_rejected(self):
        plan=SimpleNamespace(record_ids=tuple(range(129)))
        with self.assertRaises(ValueError):cp.bank_u_values(np.zeros(8),plan)
        values=np.zeros(129);values[0]=float('nan')
        with self.assertRaises(ValueError):cp.bank_u_values(values,plan)

    def test_historical_baseline_split_and_patch_exact(self):
        with tempfile.TemporaryDirectory(prefix='v24_UNIT_',dir=Path.cwd()) as temp:
            pre=Path(temp)/'Dataset730_LiverOnlineCP_OF0';pre.mkdir();(pre/'UNIT_arrays').mkdir()
            split=split_fixture()
            documents={'splits_final.json':[dict(train=split['outer_train'],val=split['outer_val'])],
                'dataset.json':{},'dataset_fingerprint.json':{},cp.PLANS+'.json':
                {'configurations':{'3d_fullres':{'patch_size':[128]*3,'batch_size':2,
                    'architecture':{'network_class_name':'UNIT.ResidualEncoderUNet'},'data_identifier':'UNIT_arrays'}}}}
            for name,value in documents.items():(pre/name).write_text(json.dumps(value),encoding='utf8')
            actual=cp.validate_baseline(pre,split)
            self.assertEqual(actual['epochs'],250);self.assertEqual(actual['cp_probability'],.5)
            self.assertIn('not retrospectively certified',actual['normalization_fitting_scope'])
            plans=documents[cp.PLANS+'.json'];plans['configurations']['3d_fullres']['patch_size']=[64]*3
            (pre/(cp.PLANS+'.json')).write_text(json.dumps(plans),encoding='utf8')
            with self.assertRaises(ValueError):cp.validate_baseline(pre,split)


class ActualPastePredicates(unittest.TestCase):
    def fixture(self):
        case=SimpleNamespace(image=np.zeros((20,20,20),np.float32),label=np.ones((20,20,20),np.int16),spacing=np.ones(3))
        source=SimpleNamespace(patch_mask=np.ones((3,3,3),bool),voxel_count=27)
        centers=np.array([[x,y,z] for x in range(3,11) for y in range(3,7) for z in range(3,7)],np.int64)
        settings=dict(occupied_clearance_vox=0,min_center_separation_mm=0.,min_center_separation_vox=0.,min_liver_coverage=1.)
        return case,source,centers,settings

    def test_all128_full_footprints_are_checked(self):
        case,source,centers,settings=self.fixture()
        result=cp.verify_cp_centers(case,source,centers,settings)
        self.assertEqual(result['source_occupied_voxels'],27)
        self.assertTrue(result['all128_full_footprints_valid'])

    def test_late_candidate_tumor_overlap_not_silently_ignored(self):
        case,source,centers,settings=self.fixture()
        case.label[tuple(centers[-1])]=2
        with self.assertRaisesRegex(ValueError,'existing_tumor'):cp.verify_cp_centers(case,source,centers,settings)

    def test_complete_bounds_and_liver_coverage(self):
        case,source,centers,settings=self.fixture();centers[-1]=[0,0,0]
        with self.assertRaisesRegex(ValueError,'outside_CT'):cp.verify_cp_centers(case,source,centers,settings)
        case,source,centers,settings=self.fixture();case.label[tuple(centers[-1])]=0
        with self.assertRaisesRegex(ValueError,'liver_coverage'):cp.verify_cp_centers(case,source,centers,settings)

    def test_center_clearance_is_enforced(self):
        case,source,centers,settings=self.fixture();case.label[14,6,6]=2
        settings['min_center_separation_mm']=5.
        with self.assertRaisesRegex(ValueError,'center_separation'):cp.verify_cp_centers(case,source,centers,settings)


class CLITests(unittest.TestCase):
    def test_action_specific_required_inputs_and_single_gpu(self):
        from tools.run_v24_nnunet_cp import parse
        self.assertEqual(parse(['train','--native','UNIT.json']).gpu,1)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):parse(['train','--native','UNIT.json','--gpu','5'])
            with self.assertRaises(SystemExit):parse(['prepare-bank','--pin','UNIT.json'])

    def test_CP_geometry_includes_zero_P_train_cases(self):
        from hiercp_v1x.v23_geometry import V23UpperGeometryCache
        split=split_fixture()
        population=SimpleNamespace(case=lambda case,count:SimpleNamespace(case_id=case,active_u_count=count))
        with patch.object(V23UpperGeometryCache,'__init__',return_value=None):
            geometry=cp.make_geometry(None,population,'UNIT',dict(workers=12,rss_gib_per_rank=192,
                geometry_resident_gib_per_rank=128,memoize_cpu_geometry=True,cache_stage_admission=True),None)
        geometry.train_cases=tuple(split['inner_train']);geometry.validation_cases=tuple(split['inner_val'])
        geometry.population=population;geometry._plan_stages={}
        cases,plans,count,selection=geometry._plans_for_stage(128,None)
        self.assertEqual(len(cases),105);self.assertEqual(len(plans),105)
        self.assertEqual(count,128);self.assertIsNone(selection)
        with self.assertRaises(ValueError):geometry._plans_for_stage(7,None)


class NativeCloneAdmission(unittest.TestCase):
    def trial(self,batch=2):
        measurement=dict(input_shape=[batch,1,128,128,128],loss=.6,gradient_finite=True,
            present_gradient_tensors=100,changed_parameter_tensors=100,encoder_and_decoder_updated=True,
            project_resource_snapshot=dict(actual_process_and_children_RSS_bytes=8*2**30))
        return dict(format='v24_actual_native_CP_full128_clone_calibration_trial_v1',debug=True,
            accepted=True,bank_sha256='a'*64,patch_size=[128]*3,cp_probability=.5,
            original_model_architecture_unchanged=True,production_updates=0,baseline_physical_batch=2,
            physical_batch=batch,project_resource_contract=cp.resource_contract(),
            measurements=[copy.deepcopy(measurement) for _ in range(3)],warmup=measurement,
            clone_optimizer_updates=4,native_CP_transport=dict(raw_events=2,native_support_voxels=20),
            full105_training_loader=True,ordinary26_validation_loader=True)

    def test_fixed_baseline_actual_native_updates_required(self):
        self.assertTrue(cp.validate_native_trial(self.trial(),baseline_batch=2,bank_sha256='a'*64)['accepted'])
        value=self.trial();value['measurements'].pop()
        with self.assertRaises(ValueError):cp.validate_native_trial(value,baseline_batch=2,bank_sha256='a'*64)

    def test_larger_CUDA_OOM_is_explicit_rejection_and_baseline_is_preserved(self):
        value=self.trial(4);value.update(accepted=False,rejection='actual_torch_CUDA_OutOfMemoryError')
        self.assertFalse(cp.validate_native_trial(value,baseline_batch=2,bank_sha256='a'*64)['accepted'])
        value['physical_batch']=2
        with self.assertRaises(ValueError):cp.validate_native_trial(value,baseline_batch=2,bank_sha256='a'*64)

    def test_other_errors_cannot_be_disguised_as_OOM(self):
        value=self.trial(4);value.update(accepted=False,rejection='ValueError')
        with self.assertRaises(ValueError):cp.validate_native_trial(value,baseline_batch=2,bank_sha256='a'*64)

    def test_nonfinite_disconnected_smaller_input_and_overbudget_rejected(self):
        mutations=(lambda x:x['measurements'][0].update(loss=float('nan')),
            lambda x:x['measurements'][0].update(encoder_and_decoder_updated=False),
            lambda x:x['measurements'][0].update(input_shape=[2,1,64,64,64]),
            lambda x:x['measurements'][0]['project_resource_snapshot'].update(actual_process_and_children_RSS_bytes=49*2**30))
        for mutate in mutations:
            value=self.trial();mutate(value)
            with self.assertRaises(ValueError):cp.validate_native_trial(value,baseline_batch=2,bank_sha256='a'*64)

    def test_missing_actual_CP_transport_rejected(self):
        value=self.trial();value['native_CP_transport']['native_support_voxels']=0
        with self.assertRaises(ValueError):cp.validate_native_trial(value,baseline_batch=2,bank_sha256='a'*64)

    def test_project_budget_is_explicit_not_host_available(self):
        budget=cp.resource_contract()
        self.assertEqual(budget['logical_CPU_cores'],4)
        self.assertEqual(budget['process_and_children_RSS_bytes'],48*2**30)
        self.assertEqual(budget['data_augmentation_workers'],4)
        self.assertTrue(budget['host_available_resources_are_not_project_allocation'])


if __name__=='__main__':unittest.main()
