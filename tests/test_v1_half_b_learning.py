"""CPU metadata tests for HALF B's real-support and comparison admission rules.

No neural execution, CT fabrication or model predictions are generated here.
The query contract/curve is existing recorded v1.4 evidence; support metadata
below exercises the explicit DEBUG contract without claiming a completed run.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from hiercp_v1x.contracts import canonical_hash
from tools.prepare_v1_half_b_support_debug import FORMAT, check_extension_case
from tools.verify_v1_half_b_learning import matched_comparisons, validate_support_manifest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'validation/v14_matched_learning_20261004'


class HalfBLearningContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = json.loads((EVIDENCE/'native_fixture/fixture_manifest.json').read_text(encoding='utf8'))
        cls.baseline = json.loads((EVIDENCE/'matched10mm/report.json').read_text(encoding='utf8'))

    def setUp(self):
        self.raw = dict(complete=True,debug=False,split=dict(inner_train=['liver_5','liver_6','liver_1'],
            inner_val=['liver_31']),raw_records=[dict(case_id='liver_1')])
        self.extension = dict(format=FORMAT,status='COMPLETE',debug=True,actual_CT=True,actual_CUDA=False,
            full_training=False,full_evaluation=False,quality_verified=False,production_ready=False,
            checkpoint_written=False,GPU_context_initialized=False,prototype_bank_unchanged=True,
            original_inputs_preserved=True,source_preserved=True,
            additional_support_only_cases=['liver_1'],support_train_cases=['liver_5','liver_6','liver_1'],
            query_train_cases=deepcopy(self.original['train_cases']),validation_cases=deepcopy(self.original['validation_cases']),
            raw_training_split=deepcopy(self.raw['split']['inner_train']),
            prototype_training_cases=deepcopy(self.original['prototype_training_cases']),
            original_fixture_sha256=self.original['fixture_sha256'],
            original_supervision_sha256=self.baseline['supervision_sha256'],
            configuration=deepcopy(self.original['config']),original_configuration_sha256=canonical_hash(self.original['config']),
            bounded_scope=deepcopy(self.baseline['bounded_scope']),candidate_count=8,candidate_pool_size=128)
        self.sign()

    def sign(self):
        self.extension['identity_sha256']=canonical_hash({k:v for k,v in self.extension.items() if k!='identity_sha256'})

    def validate(self):
        return validate_support_manifest(self.extension,self.original,self.baseline,self.raw)

    def test_support_adds_third_training_case_without_changing_queries(self):
        before = deepcopy(self.extension)
        self.assertEqual(self.validate(),['liver_5','liver_6','liver_1'])
        self.assertEqual(self.extension,before)
        self.assertEqual(self.extension['query_train_cases'],['liver_5','liver_6'])

    def test_two_support_tasks_or_duplicate_patient_rejected(self):
        for cases in ([],['liver_5'],['liver_1','liver_1']):
            with self.subTest(cases=cases):
                self.extension['additional_support_only_cases']=cases
                self.extension['support_train_cases']=['liver_5','liver_6',*cases]
                self.sign()
                with self.assertRaises(ValueError): self.validate()

    def test_validation_patient_cannot_enter_support_even_when_resigned(self):
        self.extension['additional_support_only_cases']=['liver_31']
        self.extension['support_train_cases']=['liver_5','liver_6','liver_31']
        self.raw['split']['inner_train'].append('liver_31')
        self.extension['raw_training_split']=self.raw['split']['inner_train']
        self.sign()
        with self.assertRaises(ValueError): self.validate()

    def test_original_queries_bank_full_candidates_scope_cannot_change(self):
        before=deepcopy(self.extension)
        changes=(('query_train_cases',['liver_5','liver_1']),('validation_cases',['liver_1']),
            ('prototype_training_cases',['liver_5','liver_6','liver_1']),('candidate_count',7),
            ('candidate_pool_size',64),('original_fixture_sha256','0'*64),
            ('original_supervision_sha256','0'*64),('bounded_scope',{}))
        for field,value in changes:
            with self.subTest(field=field):
                self.extension=deepcopy(before); self.extension[field]=value; self.sign()
                with self.assertRaises(ValueError): self.validate()
        self.extension=deepcopy(before)
        self.extension['configuration']['training']['epochs']=8
        self.extension['original_configuration_sha256']=canonical_hash(self.extension['configuration'])
        self.sign()
        with self.assertRaises(ValueError): self.validate()

    def test_unverified_or_quality_claiming_support_rejected(self):
        before=deepcopy(self.extension)
        for field,value in (('identity_sha256','0'*64),('actual_CT',False),('debug',False),
            ('actual_CUDA',True),('prototype_bank_unchanged',False),('source_preserved',False),
            ('full_training',True),('quality_verified',True),('checkpoint_written',True),
            ('GPU_context_initialized',True)):
            with self.subTest(field=field):
                self.extension=deepcopy(before); self.extension[field]=value
                if field!='identity_sha256': self.sign()
                with self.assertRaises(ValueError): self.validate()

    def test_extra_case_must_be_real_unique_signed_inner_training_case(self):
        self.assertEqual(check_extension_case('liver_1',['liver_5','liver_6'],['liver_31'],self.raw),dict(case_id='liver_1'))
        for case in ('liver_5','liver_6','liver_31','liver_unknown'):
            with self.subTest(case=case),self.assertRaises(ValueError):
                check_extension_case(case,['liver_5','liver_6'],['liver_31'],self.raw)
        self.raw['raw_records'].append(dict(case_id='liver_1'))
        with self.assertRaises(ValueError): check_extension_case('liver_1',['liver_5','liver_6'],['liver_31'],self.raw)

    def test_matching_successful_updates_only_and_support_confounds_are_explicit(self):
        old=self.baseline['curve']; short=[deepcopy(r) for r in old if r['step']<=8]
        before=deepcopy(short); comparisons=matched_comparisons(short,old)
        self.assertEqual([r['step'] for r in comparisons],[0,2,4,8])
        self.assertEqual(short,before)
        for row in comparisons:
            self.assertIn('half_b',row); self.assertNotIn('half_a',row)
            self.assertTrue(row['support_cohort_differs_from_baseline'])
            self.assertTrue(row['score_scale_differs_from_baseline'])
            self.assertFalse(row['loss_comparison_available'])


if __name__=='__main__': unittest.main()
