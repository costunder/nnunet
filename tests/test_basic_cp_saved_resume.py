"""CPU contract tests for later whole-checkpoint continuation; no training."""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_single_parent_resources as resource
from hiercp_v1x import basic_cp_verified_continuation as continuation
from tools import prepare_basic_cp_single_parent as prepare
from tools import run_basic_cp_single_parent as runner


class SavedResumeTests(unittest.TestCase):
    def binding(self,arm='multi5',epoch=136):
        result={name:{'path':'/saved/'+arm+'/'+name,'sha256':'a'*64}
                for name in ('effective_bank','preflight','LATEST','BEST','log_prefix','calibration')}
        if epoch is not None:result['completed_epochs']=epoch
        return result

    def row(self,arm='multi5',epoch=136):
        binding=self.binding(arm,epoch)
        row=copy.deepcopy(binding)
        row.update(completed_epochs=epoch,
            actual_resume_checkpoint_proof={'current_epoch':epoch,'logging_epochs':epoch},
            actual_original_BEST_proof={'current_epoch':100,'logging_epochs':100},
            raw_original_log=copy.deepcopy(binding['log_prefix']),
            log_prefix={'path':'/saved/completed-prefix.log','sha256':'b'*64},
            completed_log_prefix_proof={'epochs':epoch,'first_epoch':0,'last_epoch':epoch-1,
                'CP_attempted':epoch*500,'original_raw_log':copy.deepcopy(binding['log_prefix'])})
        return binding,row

    def test_original_binding_without_explicit_epoch_remains_backward_compatible(self):
        self.assertEqual(runner.bound_completed_epochs('multi5',self.binding(epoch=None)),130)

    def test_later_explicit_whole_checkpoint_is_accepted(self):
        binding,row=self.row()
        self.assertEqual(runner.validate_bound_resume('multi5',row,binding),136)
        self.assertEqual(resource.EPOCHS['multi5'],130)

    def test_backward_invalid_or_finished_binding_epoch_is_rejected(self):
        for epoch in (129,0,-1,True,136.0,'136',None,250,251):
            binding=self.binding();binding['completed_epochs']=epoch
            with self.subTest(epoch=epoch),self.assertRaises(ValueError):
                runner.bound_completed_epochs('multi5',binding)

    def test_bound_epoch_checkpoint_and_logger_proofs_must_all_match(self):
        for field in ('completed_epochs','checkpoint','logging'):
            binding,row=self.row()
            if field=='completed_epochs':row['completed_epochs']=130
            elif field=='checkpoint':row['actual_resume_checkpoint_proof']['current_epoch']=135
            else:row['actual_resume_checkpoint_proof']['logging_epochs']=135
            with self.subTest(field=field),self.assertRaises(ValueError):
                runner.validate_bound_resume('multi5',row,binding)

    def test_latest_best_and_scientific_references_cannot_drift_from_bound_inputs(self):
        for field in ('LATEST','BEST','effective_bank','preflight','calibration'):
            binding,row=self.row();row[field]['sha256']='c'*64
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'artifact differs'):
                runner.validate_bound_resume('multi5',row,binding)

    def test_best_future_or_incomplete_proof_is_rejected(self):
        for epoch,logging in ((137,137),(0,0),(100,99)):
            binding,row=self.row();row['actual_original_BEST_proof']={'current_epoch':epoch,'logging_epochs':logging}
            with self.subTest(epoch=epoch,logging=logging),self.assertRaisesRegex(ValueError,'BEST proof'):
                runner.validate_bound_resume('multi5',row,binding)

    def test_completed_log_epoch_counter_and_original_reference_must_match(self):
        for change in ('epoch','last','counter','raw','proof_raw'):
            binding,row=self.row()
            if change=='epoch':row['completed_log_prefix_proof']['epochs']=130
            elif change=='last':row['completed_log_prefix_proof']['last_epoch']=134
            elif change=='counter':row['completed_log_prefix_proof']['CP_attempted']=135*500
            elif change=='raw':row['raw_original_log']['sha256']='c'*64
            else:row['completed_log_prefix_proof']['original_raw_log']['sha256']='c'*64
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'log prefix differs'):
                runner.validate_bound_resume('multi5',row,binding)

    def test_latest_actual_saved_epoch_mismatch_fails_before_native_state_admission(self):
        with patch.object(continuation,'_state')as native:
            for actual in (135,137,True,None):
                with self.subTest(actual=actual),self.assertRaisesRegex(ValueError,'latest completed epoch'):
                    prepare.checkpoint_resume_state({'current_epoch':actual},epoch=136,trainer='native',latest=True)
            native.assert_not_called()

    def test_native_full_state_validation_is_preserved_for_latest_and_best(self):
        for actual,latest in ((136,True),(100,False)):
            saved={'current_epoch':actual}
            with self.subTest(latest=latest),patch.object(continuation,'_state',return_value={'current_epoch':actual})as native:
                proof=prepare.checkpoint_resume_state(saved,epoch=136,trainer='unchanged-full-native',latest=latest)
                native.assert_called_once_with(saved,actual,'unchanged-full-native')
                self.assertEqual(proof['current_epoch'],actual)

    def test_log_prefix_uses_new_whole_epoch_and_keeps_unfinished_raw_log_unchanged(self):
        def block(epoch):return 'Epoch '+str(epoch)+'\n[OnlineCP] epoch='+str(epoch)+' applied=200/500\nEpoch time: 80 s\n'
        completed=''.join(block(epoch)for epoch in range(136)).encode()
        raw=completed+b'Epoch 136\n[OnlineCP] epoch=136 applied=10/500\npartial-error\n'
        workspace=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='basic_cp_saved_resume_CPU_DEBUG_',dir=str(workspace))as folder, \
                patch.object(resource,'exact',side_effect=lambda path,**kwargs:Path(path)):
            root=Path(folder).resolve();self.assertTrue(root.is_relative_to(workspace))
            source=root/'raw.log';source.write_bytes(raw)
            reference={'path':str(source),'sha256':hashlib.sha256(raw).hexdigest()}
            prefix,proof=prepare.completed_log_prefix(reference,136,root/'complete.log')
            self.assertEqual(Path(prefix['path']).read_bytes(),completed)
            self.assertEqual(proof['epochs'],136)
            self.assertEqual(source.read_bytes(),raw)
            with self.assertRaisesRegex(ValueError,'completion evidence'):
                prepare.completed_log_prefix(reference,137,root/'incomplete.log')

    def test_cohort_reads_owner_checked_SHA_pinned_bindings_before_admitting_new_epoch(self):
        bindings={};experiments={}
        for arm in resource.ARMS:
            epoch=resource.EPOCHS[arm]+1
            binding,row=self.row(arm,epoch)
            # The test BEST epoch must be within each arm's original/later save.
            row['actual_original_BEST_proof']={'current_epoch':1,'logging_epochs':1}
            row.update(physical_batch=2,epochs=250,workers=4,output_root='/dev/shm/continuation/'+arm,
                       evaluator_root='/shared/code',source_native={'path':'/saved/'+arm+'/native','sha256':'a'*64})
            bindings[arm]=binding;experiments[arm]=row
        reference={'path':'/saved/bindings.json','sha256':'d'*64}
        cohort={'format':runner.FORMAT,'experiments':experiments,'retained_complete_arms':['blend50'],
                'debug':False,'canonical_code':{'path':'/shared/code'},'root':'/dev/shm/continuation','input_bindings':reference}
        with patch.object(runner,'Path',PurePosixPath),patch.object(runner,'ROOT',PurePosixPath('/shared/code')), \
             patch.object(runner,'pin',return_value='/saved/bindings.json')as pin,patch.object(runner,'read',return_value=bindings):
            self.assertIs(runner.validate_cohort(cohort),cohort)
            pin.assert_called_once_with(reference,owner=True)
            bindings['multi5']['completed_epochs']+=1
            with self.assertRaisesRegex(ValueError,'pinned input binding'):
                runner.validate_cohort(cohort)


if __name__=='__main__':unittest.main()
