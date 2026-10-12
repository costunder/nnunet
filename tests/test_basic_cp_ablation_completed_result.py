"""CPU DEBUG artifact-contract tests, no clinical training or metric execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from tools import basic_cp_ablation_completed_result as result


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT.parents[1]/'outputs/v24_gpu6_recovery_20261011/GPU4_GPU0_RESULT_SCHEMA1_20261013_1791834438897140200/evaluation'


def saved_fixture():
    return dict(current_epoch=250,logging=dict(train_losses=[-.7]*250,val_losses=[-.6]*250,
        dice_per_class_or_region=[[.95,.7]]*250,epoch_start_timestamps=list(range(250)),
        epoch_end_timestamps=[float(i)+.5 for i in range(250)]))


def log_fixture(arm='multi3'):
    return '\n'.join(f'[OnlineCP] epoch={i} applied=200/500\n[BasicCPAblation] epoch={i} arm={arm} pasted_lesions={600 if arm=="multi3" else 200} events=200 blend_border_voxels={3 if arm=="blend" else 0}' for i in range(250))


class CompletedResultTests(unittest.TestCase):
    def setUp(self):
        folder = ROOT/'tmp'
        folder.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix='DEBUG_Basic_result_',dir=folder)
        self.root = Path(self.directory.name)
        self.native_root = self.root/'native'
        self.fold = self.native_root/'nnUNet_results/Dataset730_DEBUG/DEBUG_Trainer/fold_0'
        prediction = self.fold/'validation'
        prediction.mkdir(parents=True)
        old = json.loads((HISTORICAL/'summary.json').read_text(encoding='utf8'))
        self.cases = old['evaluation_contract']['cohort']
        self.training = ['DEBUG_train_'+str(i) for i in range(105)]
        for case in self.cases:
            (prediction/(case+'.nii.gz')).write_bytes(('CPU DEBUG contract only '+case).encode())
        # Raw native NaN is preserved, and normalized only in report evidence.
        (prediction/'summary.json').write_text(json.dumps({'mean':{'1':{'Dice':.95},'2':{'Dice':float('nan')}}}),encoding='utf8')
        torch.save(saved_fixture(),self.fold/'checkpoint_final.pth')
        (self.fold/'checkpoint_best.pth').write_bytes(b'CPU DEBUG BEST provenance fixture')
        self.best = dict(complete=True,original_checkpoint_load_called=True,original_full_validation_called=True,
            final_checkpoint_fallback=False,checkpoint_name='checkpoint_best.pth',best_current_epoch=77,
            checkpoint_sha256=result.sha(self.fold/'checkpoint_best.pth'),summary_sha256=result.sha(prediction/'summary.json'),
            validation_cases=sorted(self.cases),predictions_sha256={case:result.sha(prediction/(case+'.nii.gz')) for case in self.cases})
        result.publish(self.fold/'best_validation_complete.json',self.best)
        self.proof = dict(epochs=250,checkpoint='checkpoint_best.pth',evaluation_checkpoint_is_final=False,
            best_sha256=self.best['checkpoint_sha256'],validation_cases=sorted(self.cases),fold=str(self.fold),
            final_training_witness_sha256=result.sha(self.fold/'checkpoint_final.pth'))
        result.publish(self.root/'training_and_BEST26_complete.json',self.proof)
        model = dict(physical_batch=2,effective_batch=2,gradient_accumulation=1,epochs=250,
            patch_size=[128]*3,debug=False,subset=False,pretrained_segmentation_used=False,
            actual_train_cases=self.training,actual_validation_cases=self.cases,
            model_class='CPU DEBUG contract only',parameters=123,trainable_parameters=123,workers=4)
        result.publish(self.root/'actual_model.json',model)
        for name in ['preflight.json','calibration.json','native.json']:
            result.publish(self.root/name,{'DEBUG':True})
        (self.root/'train.log').write_text(log_fixture(),encoding='utf8')
        pre = self.native_root/'nnUNet_preprocessed/Dataset730_DEBUG'
        pre.mkdir(parents=True)
        result.publish(pre/'dataset.json',{'labels':{'background':0,'liver':1,'tumor':2}})
        result.publish(self.root/'bank.json',dict(source_entries=642,eligible_cases=81,cp_probability=.5,
            candidate_count=128,intensity_scale_range=[.95,1.05],intensity_shift_range_hu=[-5.,5.]))
        self.native = dict(root=str(self.root),native_root=str(self.native_root),private_pre=str(pre),
            arm='multi3',gpu=1,bank=str(self.root/'bank.json'),bank_sha256=result.sha(self.root/'bank.json'),
            split={'outer_train':self.training,'outer_val':self.cases},archive_root=str(self.native_root/'completed_native_archive'),
            archive_storage_kind='tmpfs',persistent_archive=False,archive_reserve_bytes=2*2**30,
            persistent_required_free_bytes=10*2**30,archive_required_free_bytes=40*2**30)
        evaluation = self.root/'evaluation'
        evaluation.mkdir()
        for path in HISTORICAL.iterdir():
            if path.is_file(): shutil.copyfile(path,evaluation/path.name)
        own = copy.deepcopy(old)
        own['hier_validation'] = str(prediction)
        own['basic_validation'] = str(self.root/'unavailable_original_Basic')
        for item in own['evaluation_contract']['file_inventory']:
            if item['role']=='hier':
                item['path'] = str(prediction/(item['case_id']+'.nii.gz'))
                item['sha256'] = self.best['predictions_sha256'][item['case_id']]
        (evaluation/'summary.json').write_text(json.dumps(own),encoding='utf8')
        certificate = result.read(evaluation/'completion.json')
        certificate['outputs']['summary.json'] = certificate['summary_sha256'] = result.sha(evaluation/'summary.json')
        (evaluation/'completion.json').write_text(json.dumps(certificate),encoding='utf8')
        self.guard = patch('hiercp_v1x.v24_native_best_eval_runtime.validate_best_validation_receipt',return_value=self.best)
        self.guard.start()

    def tearDown(self):
        self.guard.stop()
        self.directory.cleanup()

    def test_fixed_schema_complete_own_best_and_basic_unverified(self):
        reports = result.completed_report(self.native,self.proof)
        document = result.read(reports['json']['path'])
        self.assertEqual({k:len(document['tables'][k]) for k in ['whole_lesion_detection','size_stratified_detection','basic_cp_comparison']},
            {'whole_lesion_detection':12,'size_stratified_detection':36,'basic_cp_comparison':44})
        self.assertEqual(len(document['historical_reference_comparison']),44)
        self.assertTrue(all(row['experiment_minus_basic'] is None for row in document['tables']['basic_cp_comparison']))
        self.assertIn('unverified',document['basic_metadata']['checkpoint_origin_status'])
        evidence = result.read(reports['evidence']['path'])
        train = evidence['experiments'][0]['training']
        self.assertEqual(train['cp_applied'],50000)
        self.assertEqual(train['cp_attempted'],125000)
        self.assertEqual(train['pasted_lesions'],150000)
        self.assertEqual(train['training_seconds'],125.)
        self.assertIsNone(evidence['experiments'][0]['native']['summary']['mean']['2']['Dice'])
        self.assertIn('NaN',(self.fold/'validation/summary.json').read_text())
        with self.assertRaises(FileExistsError):result.completed_report(self.native,self.proof)

    def test_completion_partial_final_or_wrong_cohort_rejected(self):
        for field,value in [('epochs',249),('checkpoint','checkpoint_final.pth'),('evaluation_checkpoint_is_final',True)]:
            proof = dict(self.proof,**{field:value})
            with self.assertRaises(ValueError):result.completed_report(self.native,proof)
        native = copy.deepcopy(self.native)
        native['split']['outer_val'] = native['split']['outer_val'][:-1]
        with self.assertRaisesRegex(ValueError,'full105train/26'):
            result.completed_report(native,self.proof)

    def test_evaluator_inputs_must_match_own_best_not_another_run(self):
        summary = result.read(self.root/'evaluation/summary.json')
        summary['hier_validation'] = str(self.root/'other_own_BEST')
        (self.root/'evaluation/summary.json').write_text(json.dumps(summary),encoding='utf8')
        certificate = result.read(self.root/'evaluation/completion.json')
        certificate['outputs']['summary.json'] = certificate['summary_sha256'] = result.sha(self.root/'evaluation/summary.json')
        (self.root/'evaluation/completion.json').write_text(json.dumps(certificate),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'namespace differs'):
            result.completed_report(self.native,self.proof)

    def test_training_log_no_partial_or_conflicting_counts(self):
        path = self.root/'DEBUG_counts.log'
        path.write_text(log_fixture(),encoding='utf8')
        result.training_record(saved_fixture(),self.best,path)
        path.write_text(log_fixture()+'\n[OnlineCP] epoch=0 applied=201/500',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            result.training_record(saved_fixture(),self.best,path)
        path.write_text('[OnlineCP] epoch=0 applied=200/500',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'all250'):
            result.training_record(saved_fixture(),self.best,path)
        saved = saved_fixture()
        saved['current_epoch'] = 249
        with self.assertRaisesRegex(ValueError,'witness must be250'):
            result.training_record(saved,self.best,path)

    def test_volatile_archive_all_checkpoints_preds_reports_and_original_unchanged(self):
        reports = result.completed_report(self.native,self.proof)
        free = types.SimpleNamespace(free=100*2**30)
        originals = {p:result.stat(p) for p in self.fold.rglob('*') if p.is_file()}
        with patch.object(result.shutil,'disk_usage',return_value=free):
            archived = result.archive_completed(self.native,self.proof,reports)
        receipt = result.read(archived['path'])
        self.assertTrue(receipt['complete'])
        self.assertFalse(receipt['persistent_archive'])
        self.assertFalse(receipt['durable_persistence_claimed'])
        names = {row['relative'] for row in receipt['entries']}
        self.assertIn('fold_0/checkpoint_best.pth',names)
        self.assertIn('fold_0/checkpoint_final.pth',names)
        self.assertEqual(sum(name.endswith('.nii.gz') for name in names),26)
        self.assertEqual(originals,{p:result.stat(p) for p in originals})
        for row in receipt['entries']:
            self.assertEqual(result.sha(row['path']),row['sha256'])
            self.assertNotEqual(result.stat(row['path'])[:2],result.stat(row['source'])[:2])
        with self.assertRaisesRegex(ValueError,'Fresh own archive'):
            result.archive_completed(self.native,self.proof,reports)

    def test_archive_budget_and_shared_floor_fail_before_copy(self):
        reports = result.completed_report(self.native,self.proof)
        native = dict(self.native,archive_reserve_bytes=1)
        with self.assertRaisesRegex(ValueError,'no files dropped'):
            result.archive_completed(native,self.proof,reports)
        self.assertFalse(Path(self.native['archive_root']).exists())
        with patch.object(result.shutil,'disk_usage',return_value=types.SimpleNamespace(free=40*2**30)):
            with self.assertRaisesRegex(ValueError,'required persistent reserve'):
                result.archive_completed(self.native,self.proof,reports)
        self.assertFalse(Path(self.native['archive_root']).exists())

    def test_archive_outside_own_namespace_and_false_persistent_claim_rejected(self):
        reports = result.completed_report(self.native,self.proof)
        with self.assertRaisesRegex(ValueError,'own native namespace'):
            result.archive_completed(dict(self.native,archive_root=str(self.root/'outside')),self.proof,reports)
        with self.assertRaisesRegex(ValueError,'durable persistence'):
            result.archive_completed(dict(self.native,persistent_archive=True),self.proof,reports)


if __name__=='__main__':
    unittest.main()
