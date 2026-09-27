"""Negative tests for DEBUG evidence coverage, not medical performance tests."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tools.v22_integrated_smoke_debug import (
    IntegratedSmoke, UPDATE_STAGES, validate_augmentation_coverage,
)


def fixture():
    batches=[];rows=[]
    for stage in UPDATE_STAGES:
        identity=dict(stage=stage,role='update',epoch=int(stage!='initial'),batch=1)
        batches.append(dict(**identity,cp_flags=[1,0]))
        row=dict(**identity,sample=0,augmented_CT_changed_voxels=25,
                 augmented_target_added_tumor_voxels=4)
        rows.extend([dict(row,role='warmup',batch=0),row])
    return batches,rows


class CoverageTests(unittest.TestCase):
    def setUp(self):self.batches,self.rows=fixture()

    def check(self):
        return validate_augmentation_coverage(self.batches,self.rows,UPDATE_STAGES)

    def test_normal_exact_coverage(self):
        result=self.check()
        self.assertEqual(result['expected'],result['observed'])
        self.assertEqual(len(result['observed']),3)
        self.assertEqual(result['warmup_records_excluded'],3)

    def test_empty_observations_rejected(self):
        self.rows=[]
        with self.assertRaisesRegex(AssertionError,'coverage mismatch'):self.check()

    def test_warmup_only_rejected(self):
        self.rows=[r for r in self.rows if r['role']=='warmup']
        with self.assertRaisesRegex(AssertionError,'coverage mismatch'):self.check()

    def test_each_missing_update_rejected(self):
        for stage in UPDATE_STAGES:
            with self.subTest(stage=stage):
                rows=[r for r in self.rows if not (r['stage']==stage and r['role']=='update')]
                with self.assertRaisesRegex(AssertionError,'coverage mismatch'):
                    validate_augmentation_coverage(self.batches,rows,UPDATE_STAGES)

    def test_duplicate_update_rejected(self):
        self.rows.append(copy.deepcopy(self.rows[1]))
        with self.assertRaisesRegex(AssertionError,'Duplicate CP'):self.check()

    def test_wrong_sample_rejected(self):
        self.rows[1]['sample']=1  # no-CP sample cannot stand in for sample 0
        with self.assertRaisesRegex(AssertionError,'coverage mismatch'):self.check()

    def test_wrong_batch_rejected(self):
        self.rows[1]['batch']=0
        with self.assertRaisesRegex(AssertionError,'coverage mismatch'):self.check()

    def test_wrong_epoch_rejected(self):
        self.rows[1]['epoch']=99
        with self.assertRaisesRegex(AssertionError,'coverage mismatch'):self.check()

    def test_continuous_cannot_replace_resumed(self):
        self.rows[-1]['stage']='continuous'
        with self.assertRaisesRegex(AssertionError,'Duplicate CP'):self.check()

    def test_zero_or_negative_or_nan_differences_rejected(self):
        for field in ('augmented_CT_changed_voxels','augmented_target_added_tumor_voxels'):
            for value in (0,-1,float('nan')):
                with self.subTest(field=field,value=value):
                    rows=copy.deepcopy(self.rows);rows[1][field]=value
                    with self.assertRaisesRegex(AssertionError,'CP effect absent'):
                        validate_augmentation_coverage(self.batches,rows,UPDATE_STAGES)

    def test_missing_expected_stage_rejected(self):
        self.batches.pop();self.rows=self.rows[:-2]
        with self.assertRaisesRegex(AssertionError,'Missing or unexpected'):self.check()

    def test_empty_expected_rejected(self):
        self.batches=[];self.rows=[]
        with self.assertRaisesRegex(AssertionError,'Missing or unexpected'):self.check()

    def test_duplicate_expected_batch_rejected(self):
        self.batches.append(copy.deepcopy(self.batches[0]))
        with self.assertRaisesRegex(AssertionError,'Duplicate or non-update'):self.check()

    def test_actual_cp_flags_define_all_expected_samples(self):
        self.batches[0]['cp_flags']=[1,1]
        with self.assertRaisesRegex(AssertionError,'coverage mismatch'):self.check()
        self.rows.append(dict(self.rows[1],sample=1))
        self.assertEqual(len(self.check()['expected']),4)

    def test_no_cp_update_is_not_coverage(self):
        self.batches[0]['cp_flags']=[0,0]
        with self.assertRaisesRegex(AssertionError,'actual CP samples'):self.check()

    def test_record_order_does_not_matter(self):
        self.rows.reverse()
        self.assertTrue(self.check()['exact'])

    def test_guard_blocks_real_update_call_when_observer_missing(self):
        smoke=IntegratedSmoke.__new__(IntegratedSmoke)
        smoke.trainer=SimpleNamespace(train_step=Mock())
        smoke.completed_stages=[];smoke.update_batches=[];smoke.augmentation_checks=[]
        batch=dict(_DEBUG_augmentation_identity={k:v for k,v in self.batches[0].items() if k!='cp_flags'},
                   online_cp_applied=[1,0])
        with self.assertRaisesRegex(AssertionError,'coverage mismatch'):
            smoke.checked_train_step(batch,'initial')
        smoke.trainer.train_step.assert_not_called()
        self.assertEqual(smoke.completed_stages,[])

    def test_guard_tracks_three_actual_calls_and_rejects_fourth(self):
        smoke=IntegratedSmoke.__new__(IntegratedSmoke)
        smoke.trainer=SimpleNamespace(train_step=Mock(return_value={'loss':1}))
        smoke.completed_stages=[];smoke.update_batches=[];smoke.augmentation_checks=[]
        for i,expected in enumerate(self.batches):
            smoke.augmentation_checks.extend(self.rows[2*i:2*i+2])
            batch=dict(_DEBUG_augmentation_identity={k:v for k,v in expected.items() if k!='cp_flags'},
                       online_cp_applied=expected['cp_flags'])
            self.assertEqual(smoke.checked_train_step(batch,expected['stage']),{'loss':1})
        self.assertEqual(smoke.completed_stages,list(UPDATE_STAGES))
        self.assertEqual(smoke.trainer.train_step.call_count,3)
        with self.assertRaisesRegex(AssertionError,'stage/order mismatch'):
            smoke.checked_train_step(batch,'resumed')
        self.assertEqual(smoke.trainer.train_step.call_count,3)


if __name__=='__main__':unittest.main()
