"""Isolated unit checks only; generated tiny arrays are not CT/CUDA evidence."""
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('tumorcp_profile_runner_unit', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class LiverContractTests(unittest.TestCase):
    def setUp(self):
        base = HERE / 'test_artifacts'
        base.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='liver-unit-', dir=base)).resolve()
        self.assertIn(base.resolve(), self.root.parents)
        split = json.loads((HERE.parents[1] / 'config/split_cp80_fold0.json').read_text())
        self.profile = dict(format='tumorcp_paper_dataset_profile_v1', dataset='MSD Task03_Liver',
            task='Task029_Liver', raw_layout='msd', allow_empty_intra=True,
            all_ids=sorted(split['outer_train']+split['outer_val']),
            train_ids=split['outer_train'], heldout_ids=split['outer_val'],
            split_provenance='UNIT original project split; no medical data',
            source_split_file=str(self.root/'source-split.json'),
            source_split_sha256='7a993854c570b2cabe98dffc5df97550d74da3a688c3c3b75942ce4cbd15bc44')
        self.source = Path(self.profile['source_split_file'])
        self.source.write_text(json.dumps([dict(train=split['outer_train'], val=split['outer_val'])]))
        self.path = self.root/'profile-input.json'
        self.path.write_text(json.dumps(self.profile))
        self.saved = (runner.DATASET_PROFILE, runner.CASE_IDS, runner.TASK, runner.RAW_LAYOUT)
        self.addCleanup(self.restore)

    def restore(self):
        runner.DATASET_PROFILE, runner.CASE_IDS, runner.TASK, runner.RAW_LAYOUT = self.saved

    def configure(self):
        original = runner.fingerprint
        def known_fixture(path):
            return self.profile['source_split_sha256'] if Path(path) == self.source else original(path)
        with patch.object(runner, 'fingerprint', side_effect=known_fixture):
            runner.configure_profile(self.path, self.root)

    def test_missing_and_duplicate_patients_rejected(self):
        broken = dict(self.profile, train_ids=self.profile['train_ids'][:-1])
        self.path.write_text(json.dumps(broken))
        with self.assertRaises(ValueError): runner.configure_profile(self.path, self.root)
        broken = dict(self.profile, heldout_ids=[self.profile['train_ids'][0]]+self.profile['heldout_ids'][1:])
        self.path.write_text(json.dumps(broken))
        with self.assertRaises(ValueError): runner.configure_profile(self.path, self.root)

    def test_changed_source_split_rejected(self):
        with self.assertRaises(ValueError): runner.configure_profile(self.path, self.root)

    def test_liver_exact_split_and_environment(self):
        self.configure()
        folder = self.root/'preprocessed/Task029_Liver'
        folder.mkdir(parents=True)
        path, training, heldout = runner._write_split(self.root)
        self.assertEqual(set(training), set(self.profile['train_ids']))
        self.assertEqual(set(heldout), set(self.profile['heldout_ids']))
        with path.open('rb') as f: self.assertEqual(len(pickle.load(f)), 1)
        self.assertEqual(runner.cohort_counts(), (131,105,26))
        env = runner.runtime_env(self.root,4)
        self.assertEqual(env['TUMORCP_PAPER_DATASET_JSON'],str(self.root/'dataset_profile.json'))
        self.assertEqual(runner.raw_paths(self.root,'liver_9'),
                         (self.root/'imagesTr/liver_9.nii.gz', self.root/'labelsTr/liver_9.nii.gz'))

    def test_final26_empty_and_positive_scopes(self):
        import nibabel as nib
        import numpy as np
        self.configure()
        predictions=runner._training_folder(self.root)/'validation_final_26_raw'
        predictions.mkdir(parents=True)
        labels=self.root/'raw/labelsTr'
        labels.mkdir(parents=True)
        cases=self.profile['heldout_ids']
        for i,case in enumerate(cases):
            truth=np.zeros((2,2,2),np.uint8)
            pred=np.zeros_like(truth)
            if i<24: truth[0,0,0]=2
            if i<23: pred[0,0,0]=2
            nib.save(nib.Nifti1Image(truth,np.eye(4)),labels/(case+'.nii.gz'))
            nib.save(nib.Nifti1Image(pred,np.eye(4)),predictions/(case+'.nii.gz'))
        report=runner.evaluate_report(self.root,self.root/'raw',cases)
        self.assertAlmostEqual(report['metrics']['tumor']['mean_Dice'],25/26)
        self.assertAlmostEqual(report['mean_tumor_Dice_GT_positive'],23/24)
        self.assertEqual(report['metrics']['tumor']['empty_both_cases'],2)
        self.assertIsNone(report['published_target_tumor_Dice'])
        (predictions/(cases[-1]+'.nii.gz')).rename(predictions/'missing-unit-patient.nii.gz')
        with self.assertRaises(ValueError): runner.evaluate_report(self.root,self.root/'raw',cases)


if __name__=='__main__': unittest.main(verbosity=2)
