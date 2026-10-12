"""CPU UNIT admission fixtures only; no CT/GPU/training/result is produced.

Synthetic65/21 reports exercise the original deterministic whole-population
validator. Tiny CPU tensors test exact provenance refusal, not model efficacy.
"""
from __future__ import annotations
import ast
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from hiercp_v1x import v25_authorized_early_downstream as adapter
from hiercp_v1x import v25_cp as original_cp,v25_native_cp as original_native,v25_downstream as original_downstream
from hiercp_v1x.u_bridge_training import digest
from hiercp_v1x.v25_training import FORMAT
from tests.test_v25_training_resume import completed_state,fresh_state,after_initial,pending_epoch,train_batches
from tests.test_v25_targets import TRAIN_CASES,VAL_CASES


def save_json(path,value):
    Path(path).write_text(json.dumps(value,allow_nan=False),encoding='utf8')


def payload(owner,state,model):
    value=dict(format=FORMAT,identity_sha256=owner['identity_sha256'],model=model,
        optimizer={'CPU_UNIT_ONLY':True},scheduler={'CPU_UNIT_ONLY':True},scaler={'CPU_UNIT_ONLY':True},
        rank_rng={'CPU_UNIT_ONLY':True},shuffle_generator=torch.tensor([4],dtype=torch.uint8),state=state)
    value['content_sha256']=digest(value)
    return value


class AuthorizedBAdmissionCPUUnit(unittest.TestCase):
    def setUp(self):
        scratch=adapter.ROOT/'tmp';scratch.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=scratch);self.root=Path(self.temp.name)
        self.training=self.root/'training';self.training.mkdir()
        self.authpath=self.root/'authorization.json'
        self.auth=dict(format=adapter.AUTHORIZATION_FORMAT,arm='B',physical_GPU=5,
            explicit_user_authorization=True,user_instruction='CPU UNIT ONLY authorization fixture',
            stop_after_current_epoch=True,use_own_full128_BEST=True,
            original_planned_epochs=40,original_planned_optimizer_updates=680,target_completed_epochs=3,
            CP_source_entries=642,CP_candidates_per_source=128,native_epochs=250,native_evaluation_cases=26,
            native_evaluation_checkpoint='checkpoint_best.pth',context_unchanged=True,model_and_loss_unchanged=True,
            release_actors=[dict(pid=123,create_time=1.,UID=41833,command=['CPU_UNIT_ONLY'])])
        save_json(self.authpath,self.auth)
        self.state=completed_state(3,16,16);self.state['status']='PAUSED'
        self.binding=dict(CPU_UNIT_ONLY=True,debug=False,epochs=40,physical_patient_batch=4,workers=4,
            train_cases=TRAIN_CASES,val_cases=VAL_CASES,identity=dict(config={'arm':'B'},request_sha256='b'*64),
            config=dict(v24_runtime=dict(curriculum=self.state['curriculum']['policy'])),
            model_contract=dict(context=dict(contract_sha256='c'*64)))
        self.owner=dict(binding=self.binding,identity_sha256=digest(self.binding))
        save_json(self.training/'training_identity.json',self.owner)
        self.beststate=pending_epoch(after_initial(fresh_state(16,16)))
        self.model={'CPU_UNIT_ONLY_weight':torch.tensor([1.,2.])}
        self.publish()
        self.mock=patch.object(adapter,'released',return_value=True);self.mock.start()
        adapter._VERIFIED_PINS.clear()

    def tearDown(self):
        self.mock.stop();self.temp.cleanup();adapter._VERIFIED_PINS.clear()

    def publish(self):
        torch.save(payload(self.owner,self.state,self.model),self.training/'checkpoint_latest.pt')
        torch.save(payload(self.owner,self.beststate,self.model),self.training/'checkpoint_best.pt')
        save_json(self.training/'initial_full_validation_epoch_000.json',self.state['initial_validation'])
        for curve in self.state['history']:
            for role in ('train_probe','stage_validation','full_validation'):
                save_json(self.training/f'{role}_epoch_{curve["epoch"]:03d}.json',curve[role])

    def pin(self):
        return adapter.inspect_partial_best(self.training,authorization_path=self.authpath)

    def validate(self,pin):
        return adapter.validate_partial_provenance(pin,model_sha256=digest(self.model),context_contract_sha256='c'*64)

    def test_truthful_partial_epoch_and_original_128_best_without_checkpoint_write(self):
        before={p.name:adapter.witness(p)for p in self.training.iterdir()}
        pin=self.pin();self.validate(pin)
        self.assertEqual(pin['completed_epochs'],3);self.assertEqual(pin['optimizer_updates'],51)
        self.assertFalse(pin['full_training']);self.assertEqual(pin['best_record']['epoch'],1)
        self.assertEqual(pin['native_epochs'],250);self.assertEqual(pin['native_evaluation_cases'],26)
        self.assertEqual(before,{p.name:adapter.witness(p)for p in self.training.iterdir()})
        with self.assertRaises(ValueError):
            original_cp.validate_best_provenance(pin,model_sha256=digest(self.model),context_contract_sha256='c'*64)

    def test_truthful_next_epoch_cursor_not_arbitrarily_capped(self):
        train_batches(self.state,7);self.publish()
        pin=self.pin();self.validate(pin)
        self.assertEqual(pin['optimizer_updates'],58)
        self.assertEqual(pin['next_epoch_cursor']['train_position'],28)
        self.assertEqual(pin['completed_epochs'],3)

    def test_actual_integer_recalled_P_keys_preserve_winning_report_after_JSON_serialization(self):
        # Actual reports keep threshold keys 1/5/10 as ints in torch checkpoint
        # dictionaries and serialize them to strings in the JSON publication.
        winning=self.beststate['pending_epoch_completion']['full_validation']
        historical=self.state['history'][0]['full_validation']
        for report in (winning,historical):
            for row in report['cases']:
                row['recalled_observed_P']={1:1,5:2,10:3}
        self.publish()
        saved=json.loads((self.training/'full_validation_epoch_001.json').read_text())
        self.assertNotEqual(winning,saved)
        self.assertEqual(digest(winning),digest(saved))
        self.validate(self.pin())
        # Value changes still fail the binding even though BEST selection scalar
        # metadata remains unchanged and checkpoint content hashes are valid.
        winning['cases'][0]['recalled_observed_P'][1]=4
        self.publish()
        with self.assertRaisesRegex(ValueError,'BEST model full128 winning report'):
            self.pin()

    def test_checkpoint_report_identity_and_full_patient_changes_are_refused(self):
        for mutate in ('patient','counter','best','identity','report'):
            with self.subTest(mutate=mutate):
                original=copy.deepcopy(self.state)
                if mutate=='patient':self.state['history'][1]['full_validation']['cases'].pop()
                elif mutate=='counter':self.state['updates']+=1;self.state['attempts']+=1
                elif mutate=='best':self.beststate['best']['epoch']=2
                elif mutate=='identity':self.owner['identity_sha256']='a'*64
                self.publish()
                if mutate=='report':
                    path=self.training/'full_validation_epoch_001.json';data=json.loads(path.read_text())
                    data['metrics']['per_P_patient_mrr']=.99;save_json(path,data)
                with self.assertRaises(ValueError):self.pin()
                self.state=original;self.owner['identity_sha256']=digest(self.binding)
                self.beststate=pending_epoch(after_initial(fresh_state(16,16)))
                self.publish()

    def test_pin_tampering_and_source_replacement_cannot_pass_cached_admission(self):
        pin=self.pin()
        for field,value in (('optimizer_updates',52),('full_training',True),('completed_epochs',40),
                ('model_sha256','d'*64),('native_epochs',2),('original_U_history',[])):
            bad=copy.deepcopy(pin);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.validate(bad)
        changed=torch.load(self.training/'checkpoint_best.pt',weights_only=False)
        changed['model']['CPU_UNIT_ONLY_weight'][0]=100.
        torch.save(changed,self.training/'checkpoint_best.pt')
        with self.assertRaises(ValueError):self.validate(pin)

    def test_no_pending_epoch_or_unfinished_validation_is_admitted(self):
        for phase in ('full_validation','epoch_completion'):
            self.state['phase']=phase;self.publish()
            with self.assertRaises(ValueError):self.pin()

    def test_wrong_or_modified_authorization_and_live_predecessor_are_refused(self):
        for field,value in (('arm','A'),('physical_GPU',1),('CP_source_entries',641),
                ('native_epochs',10),('native_evaluation_checkpoint','checkpoint_final.pth')):
            altered=dict(self.auth,**{field:value});save_json(self.authpath,altered)
            with self.assertRaises(ValueError):self.pin()
        save_json(self.authpath,self.auth)
        with patch.object(adapter,'released',return_value=False),self.assertRaises(RuntimeError):self.pin()


class PrivateNamespacesCPUUnit(unittest.TestCase):
    def test_actual_private_CP_entry_uses_B_rebound_lease_instead_of_PBS_only_lease(self):
        from hiercp_v1x import v25_B_early_downstream_resources as resources,v25_server_resources as pbs
        private=adapter.private_modules()['CP']
        self.assertEqual(private.authorized_B_adaptation_proof['admission_and_child_import_counts']['B_CP_resource_lease'],1)
        net=torch.nn.Linear(1,1);net.eval()  # CPU UNIT guard path, deliberately no CUDA/model execution.
        with patch.object(resources,'BEarlyDownstreamAllocationLease')as constructor,\
                patch.object(pbs,'AllocationLease',side_effect=AssertionError('PBS-only lease must not be reached'))as old,\
                patch('sys.argv',['CPU_UNIT_ENTRY','--allocation-request','CPU_UNIT_B_DIRECT.json']):
            with self.assertRaisesRegex(RuntimeError,'actual complete v2.5 STU model'):
                private.score_v25_cp_bank(net=net,inputs=None,population=None,source_bank=None,
                    baseline_preprocessed=None,physical_candidate_batch=32,budget=SimpleNamespace(check=lambda:None),
                    amp=True,best_provenance={},output=Path('CPU_UNIT_NO_OUTPUT'))
            constructor.assert_called_once_with('CPU_UNIT_B_DIRECT.json')
            self.assertEqual(constructor.return_value.verify.call_count,1)
            old.assert_not_called()

    def test_original_globals_and_full_completion_validator_remain_unchanged(self):
        before=(original_cp.validate_best_provenance,original_native.cp,original_downstream.cp,original_downstream.native)
        private=adapter.private_modules()
        self.assertIs(private['CP'].validate_best_provenance,adapter.validate_partial_provenance)
        self.assertIs(private['NATIVE'].cp,private['CP'])
        self.assertIs(private['DOWNSTREAM'].native,private['NATIVE'])
        self.assertEqual(before,(original_cp.validate_best_provenance,original_native.cp,original_downstream.cp,original_downstream.native))
        for name in ('score_v25_cp_bank','validate_payload','matched_plan','source_batches'):
            original=getattr(original_cp,name);cloned=getattr(private['CP'],name)
            if hasattr(original,'__code__'):self.assertEqual(original.__code__.co_code,cloned.__code__.co_code)

    def test_native_child_entries_and_generated_runtime_use_separate_admission(self):
        private=adapter.private_modules();native=private['NATIVE'];parts=native.private_modules()
        self.assertEqual(native.train.__code__.co_consts.count(
            'from hiercp_v1x.v25_authorized_early_downstream import run_training_entry; run_training_entry()'),1)
        self.assertIs(parts['NATIVE_RUNTIME'].validate_overlay,private['CP'].validate_overlay)
        self.assertEqual(native.authorized_B_adaptation_proof['admission_and_child_import_counts']['dynamic_native_import'],5)

    def test_original_score_cli_only_routes_admission_output_and_resource_import(self):
        from tools import run_v25
        private=adapter._private_clone(run_v25,score_output=True)
        self.assertEqual(private.authorized_B_adaptation_proof['admission_and_child_import_counts'],
            {'B_resource_lease':1,'absolute_v25_cp':1,'fresh_cp_bank':1,'fresh_cp_scores.json':1})
        self.assertEqual(private.configuration.__code__.co_code,run_v25.configuration.__code__.co_code)
        self.assertEqual(private.request.__code__.co_code,run_v25.request.__code__.co_code)

    def test_evidence_records_real_partial_counts_and_keeps_native250_full(self):
        pin=dict(completed_epochs=13,optimizer_updates=222,source_completion_kind='USER_AUTHORIZED_EPOCH_BOUNDARY_EARLY_STOP',
            authorization={'path':'CPU_UNIT_ONLY'},original_U_history=[{'CPU_UNIT_ONLY':True}])
        value=dict(experiments=[dict(settings=dict(gnn=dict(epochs=40,optimizer_updates=680)))],
            verification=dict(full_training=True),limitations=[])
        overlay=dict(best_provenance=pin,model_sha256='a'*64,context_contract={'contract_sha256':'b'*64})
        with patch.object(adapter,'read',return_value=overlay),patch.object(adapter,'validate_partial_provenance'):
            result=adapter._truthful_collect(lambda *_:copy.deepcopy(value),{'score_overlay':'CPU_UNIT_ONLY'},None,None)
        self.assertEqual(result['experiments'][0]['settings']['gnn']['epochs'],13)
        self.assertEqual(result['experiments'][0]['settings']['gnn']['optimizer_updates'],222)
        self.assertFalse(result['verification']['full_GNN_training'])
        self.assertEqual(result['verification']['full_training']['completed_epochs'],250)
        from tools.render_nnunet_result_schema_v1 import render_report
        result['experiments'][0]['experiment']='CPU_UNIT B display only'
        report,text=render_report(result)
        self.assertEqual(report['verification']['full_training']['scope'],'native segmentation')
        self.assertIn('native segmentation',text)
        self.assertIn('original40/680 was not completed',text)
        self.assertEqual(len(report['tables']['whole_lesion_detection']),12)
        self.assertEqual(len(report['tables']['size_stratified_detection']),36)
        self.assertEqual(len(report['tables']['basic_cp_comparison']),44)


if __name__=='__main__':unittest.main()
