"""CPU DEBUG resume fixtures; no CUDA, clinical training or model-scale claim."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import psutil
import torch

from hiercp_v1x import basic_cp_ablation_resume as resume


class NativeResumeTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[1]/'tmp';parent.mkdir(exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='native_resume_debug_',dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve()
        self.oldroot=self.root/'old';self.newroot=self.root/'new'
        self.oldroot.mkdir();self.newroot.mkdir()
        self.uid=self.root.stat().st_uid
        self.uid_patch=patch.object(resume,'UID',self.uid);self.uid_patch.start();self.addCleanup(self.uid_patch.stop)
        self.namespace_patch=patch.object(resume,'SOURCE_NAMESPACE',self.root)
        self.namespace_patch.start();self.addCleanup(self.namespace_patch.stop)
        self.actor={'pid':987654,'create_time':123.,'UID':self.uid,'command':['python','CPU_DEBUG_controller']}
        self.process_patch=patch.object(resume.psutil,'Process',side_effect=psutil.NoSuchProcess(self.actor['pid']))
        self.process_patch.start();self.addCleanup(self.process_patch.stop)
        self.baseline={'dataset_name':'Dataset730_CPUDEBUG','plans_name':'nnUNetResEncUNetMPlans'}
        self.split={'outer_train':[f'CPU_DEBUG_train{i}' for i in range(105)],
                    'outer_val':[f'CPU_DEBUG_val{i}' for i in range(26)]}
        common={'gpu':6,'arm':'blend','trainer':resume.TRAINER,'bank':'/CPU_DEBUG/original/index.json',
            'bank_sha256':resume.BANK_SHA,'baseline':self.baseline,'split':self.split,
            'historical_trainer_sha256':resume.HISTORICAL_SHA,'readonly_admission':'/CPU_DEBUG/frozen_admission.json',
            'readonly_admission_sha256':'1'*64,'shared_preprocessed_cases':131,
            'source_preprocessed_arrays_written':False,'full_training_cases':105,'full_validation_cases':26}
        self.old=dict(common,root=str(self.oldroot),native_root=str(self.oldroot/'native'),
                      private_runtime=str(self.oldroot/'native/runtime'))
        self.native=dict(common,root=str(self.newroot),native_root=str(self.newroot/'native'),
                         private_runtime=str(self.newroot/'native/runtime'))
        for native in (self.old,self.native):
            path=Path(native['private_runtime'])/'nnunetv2/training/nnUNetTrainer'
            path.mkdir(parents=True)
            (path/'nnUNetTrainer_OnlinePairedCP.py').write_bytes(b'CPU_DEBUG historical trainer source')
            (path/'nnUNetTrainer_BasicCPAblations.py').write_bytes(b'CPU_DEBUG same blend source')
        historical=resume.sha(Path(self.old['private_runtime'])/'nnunetv2/training/nnUNetTrainer/nnUNetTrainer_OnlinePairedCP.py')
        self.historical_patch=patch.object(resume,'HISTORICAL_SHA',historical)
        self.historical_patch.start();self.addCleanup(self.historical_patch.stop)
        self.old['historical_trainer_sha256']=self.native['historical_trainer_sha256']=historical
        self.write_json(self.oldroot/'native.json',self.old)
        request={'uid':self.uid,'gpu':6,'arm':'blend','controller_identity':self.actor}
        self.write_json(self.oldroot/'request.json',request)
        model={'trainer':resume.TRAINER,'epochs':250,'physical_batch':2,'effective_batch':2,
            'gradient_accumulation':1,'patch_size':[128]*3,'workers':4,'debug':False,'subset':False,
            'pretrained_segmentation_used':False,'actual_train_cases':self.split['outer_train'],
            'actual_validation_cases':self.split['outer_val']}
        self.write_json(self.oldroot/'actual_model.json',model)
        self.oldfold=resume.fold(self.old);self.oldfold.mkdir(parents=True)
        self.latest=self.oldfold/'checkpoint_latest.pth';self.best=self.oldfold/'checkpoint_best.pth'
        torch.save(self.saved(200),self.latest);torch.save(self.saved(193),self.best)
        self.log=b''.join((f'2026-10-13 00:00:00: Epoch {i}\n'
            f'[OnlineCP] epoch={i} applied=210/500\nEpoch time: 80 s\n').encode() for i in range(217))
        (self.oldroot/'train.log').write_bytes(self.log)
        self.native['completed_native_resume']={'root':str(self.oldroot),
            'request_sha256':resume.sha(self.oldroot/'request.json'),'controller_identity':self.actor,
            'checkpoint_sha256':resume.sha(self.latest),'best_sha256':resume.sha(self.best),'completed_epochs':200}

    def saved(self,epoch):
        return {'current_epoch':epoch,'trainer_name':resume.TRAINER,'network_weights':{'CPU_DEBUG.weight':torch.ones(2,2)},
            'optimizer_state':{'state':{0:{'momentum_buffer':torch.ones(2,2)}},'param_groups':[{'params':[0],'lr':.001}]},
            'grad_scaler_state':{'scale':65536.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000,'_growth_tracker':5},
            '_best_ema':.9,'init_args':{'CPU_DEBUG':True},
            'logging':{'train_losses':[-.7]*epoch,'val_losses':[-.6]*epoch,'dice_per_class_or_region':[[.95,.7]]*epoch,
                'epoch_start_timestamps':[float(i) for i in range(epoch)],
                'epoch_end_timestamps':[float(i)+.5 for i in range(epoch)]}}

    @staticmethod
    def write_json(path,value):path.write_text(json.dumps(value),encoding='utf8')

    def test_verified200_resume_keeps_optimizer_best_prefix_and_old_results(self):
        before={str(path):resume.snapshot(path) for path in self.oldroot.rglob('*') if path.is_file()}
        receipt=resume.prepare_resume(self.native)
        self.assertEqual(receipt['completed_epochs'],200);self.assertEqual(receipt['native_goal_epochs'],250)
        self.assertEqual(receipt['remaining_epochs'],50);self.assertEqual(receipt['rollback_completed_epochs'],15)
        self.assertEqual(receipt['BEST_current_epoch_raw'],193)
        for source in (self.latest,self.best):
            copied=resume.fold(self.native)/source.name
            self.assertEqual(copied.read_bytes(),source.read_bytes())
            self.assertNotEqual(resume.snapshot(copied)[:2],resume.snapshot(source)[:2])
        prefix=(self.newroot/'resumed_training_prefix.log').read_bytes()
        self.assertTrue(self.log.startswith(prefix));self.assertNotIn(b'Epoch 200\n',prefix)
        self.assertEqual(receipt['training_prefix_sha256'],resume.sha(self.newroot/'resumed_training_prefix.log'))
        self.assertFalse(receipt['full_training_complete'])
        self.assertEqual(before,{str(path):resume.snapshot(path) for path in self.oldroot.rglob('*') if path.is_file()})

    def test_partial_raw216_and_incomplete_logger_are_rejected(self):
        for epoch,missing in [(216,None),(200,'epoch_end_timestamps'),(200,'train_losses')]:
            saved=self.saved(epoch)
            if missing:saved['logging'][missing].pop()
            torch.save(saved,self.latest)
            self.native['completed_native_resume']['checkpoint_sha256']=resume.sha(self.latest)
            with self.assertRaises(ValueError):resume.prepare_resume(self.native)
        self.assertFalse((self.newroot/'resume.json').exists())

    def test_missing_optimizer_or_scaler_is_not_silently_reinitialized(self):
        for field in ['optimizer_state','grad_scaler_state']:
            saved=self.saved(200);saved[field]={};torch.save(saved,self.latest)
            self.native['completed_native_resume']['checkpoint_sha256']=resume.sha(self.latest)
            with self.assertRaisesRegex(ValueError,'optimizer|GradScaler'):resume.prepare_resume(self.native)

    def test_exact_source_sha_and_scientific_trainer_must_match(self):
        self.native['completed_native_resume']['checkpoint_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA/stat changed'):resume.prepare_resume(self.native)
        self.native['completed_native_resume']['checkpoint_sha256']=resume.sha(self.latest)
        new=Path(self.native['private_runtime'])/'nnunetv2/training/nnUNetTrainer/nnUNetTrainer_BasicCPAblations.py'
        new.write_bytes(b'CPU_DEBUG unauthorized different scientific paste')
        with self.assertRaisesRegex(ValueError,'scientific trainer bytes'):resume.prepare_resume(self.native)

    def test_live_original_controller_refuses_resume_without_sending_signals(self):
        process=unittest.mock.Mock();process.create_time.return_value=self.actor['create_time']
        process.uids.return_value=type('uids',(),{'real':self.uid})()
        with patch.object(resume.psutil,'Process',return_value=process):
            with self.assertRaisesRegex(RuntimeError,'still present'):resume.prepare_resume(self.native)
        process.terminate.assert_not_called();process.kill.assert_not_called()

    def test_wrong_cohort_best_outside_prefix_and_duplicate_cp_epoch_fail(self):
        wrong=copy.deepcopy(self.native);wrong['split']['outer_train'].pop()
        with self.assertRaisesRegex(ValueError,'scientific identity'):resume.prepare_resume(wrong)
        torch.save(self.saved(201),self.best);self.native['completed_native_resume']['best_sha256']=resume.sha(self.best)
        with self.assertRaisesRegex(ValueError,'BEST epoch'):resume.prepare_resume(self.native)
        torch.save(self.saved(193),self.best);self.native['completed_native_resume']['best_sha256']=resume.sha(self.best)
        (self.oldroot/'train.log').write_bytes(self.log.replace(b'[OnlineCP] epoch=0 applied=210/500\n',
            b'[OnlineCP] epoch=0 applied=210/500\n[OnlineCP] epoch=0 applied=210/500\n'))
        with self.assertRaisesRegex(ValueError,'Duplicate/invalid'):resume.prepare_resume(self.native)

    def test_existing_fold_is_never_overwritten(self):
        path=resume.fold(self.native);path.mkdir(parents=True)
        sentinel=path/'user_result.txt';sentinel.write_text('CPU_DEBUG existing result',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'Fresh untouched resume fold'):resume.prepare_resume(self.native)
        self.assertEqual(sentinel.read_text(),'CPU_DEBUG existing result')


if __name__=='__main__':unittest.main()
