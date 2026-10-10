"""CPU UNIT checkpoint/ownership fixtures; no real GNN or nnUNet training.

Tiny serialized tensors exercise the actual six-state/BEST admission path.
They never replace production CUDA/full-population evidence.
"""
import copy
import contextlib
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from hiercp_v1x import v24_nnunet_cp as cp
from hiercp_v1x import v24_native_calibration_runtime as native_runtime
from hiercp_v1x.u_bridge_training import digest
from hiercp_v1x.contracts import canonical_hash
from tools import run_v24_all_p as cli
from tools.run_v24_nnunet_cp import parse


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')


def save_checkpoint(path,value):
    value=copy.deepcopy(value)
    value['content_sha256']=digest({key:item for key,item in value.items() if key!='content_sha256'})
    torch.save(value,path)


class CurrentOwnArmAdmission(unittest.TestCase):
    def setUp(self):
        (cp.ROOT/'outputs').mkdir(exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='v24_chain_CPU_UNIT_',dir=cp.ROOT/'outputs')
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.output=self.root/'UNIT_complete_GNN';self.output.mkdir()
        self.inventory=self.root/'UNIT_inventory.json';write_json(self.inventory,{'CPU_UNIT_ONLY':True})
        self.cache=self.root/'UNIT_input_cache';self.cache.mkdir()
        self.code=self.root/'UNIT_scientific_code';self.code.mkdir()
        for name in cli.FILES:
            target=self.code/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(cp.ROOT/name,target)
        self.gpu=5;self.stu=None
        self.make_fixture()

    def make_fixture(self):
        gpu=self.gpu
        config_path=cp.ROOT/({4:'config/v24_gpu4_STUNetS_all_U_GT_blind.json',
            5:'config/v24_gpu5_GT_blind.json',6:'config/v24_gpu6_STUNetS_GT_blind.json'}[gpu])
        for name in cli.execution_source_files(gpu):
            target=self.code/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(cp.ROOT/name,target)
        config=json.loads(config_path.read_text(encoding='utf8'))
        args=SimpleNamespace(config=config_path,native_experiment=self.root/'UNIT_native',inventory=self.inventory,
            input_cache=self.cache,gpu=gpu,stunet_checkpoint=self.stu)
        self.request=cli.request(args,config)
        contract=dict(recipient_GT_used_in_forward=False,trained_v23_weights_loaded=False,
            parameters=7,trainable_parameters=7,encoder='original_CNN_GAT' if gpu==5 else 'official_pretrained_STU_Net_S')
        calibration=dict(request_sha256=self.request['request_sha256'],physical_GPU=gpu,
            model_contract=contract,initial_state_sha256='a'*64,selected_physical_patient_batch=4,
            selected_physical_candidate_batch=32 if gpu==5 else 64,CPU_UNIT_only=True)
        bound_config=copy.deepcopy(config);bound_config['v24_runtime']['batch_calibration']=calibration
        binding=dict(identity=self.request,config=bound_config,train_cases=['UNIT_train_'+str(i) for i in range(65)],
            val_cases=['UNIT_val_'+str(i) for i in range(21)],physical_patient_batch=4,
            physical_candidate_batch=32 if gpu==5 else 64,epochs=40,debug=False,
            model_contract=contract,initial_state_sha256='a'*64,population={'CPU_UNIT_only':True})
        self.owner=dict(binding=binding,identity_sha256=digest(binding))
        reports=[dict(phase='full_validation',active_u=128,epoch=epoch,debug=False,
            recipient_GT_used_in_forward=False,all_P_scored=True,evaluation_view_epoch=29,
            metrics=dict(per_P_patient_mrr=.6 if epoch==17 else .3+epoch/1000,
                per_P_patient_top1=.2,patient_balanced_pair_loss=.4)) for epoch in range(1,41)]
        self.winner=dict(epoch=17,updates=289,selection_key=[.6,.2,-.4],selected_by='fixed_full128_validation_only')
        history=[dict(epoch=epoch,full_validation=reports[epoch-1]) for epoch in range(1,41)]
        if gpu==4:
            for row in history:
                row.update(train_probe=dict(active_u=128),stage_validation=dict(active_u=128),
                    curriculum_transition=dict(epoch=row['epoch'],previous_active_u=128,next_active_u=128,
                        expanded=False,reason='full_bank_replay'))
        state=dict(epoch=41,phase='complete',status='COMPLETE',updates=680,attempts=680,history=history,
            best=self.winner,connected=['UNIT_parameter_'+str(i) for i in range(981 if gpu==5 else 537)],
            curriculum=dict(policy=config['v24_runtime']['curriculum'],active_u=128),
            train_order=None,train_rows=[],train_position=0,evaluation_rows=[],evaluation_position=0)
        self.latest=dict(format='v24_GT_blind_all_P_patient_balanced_training_v1',
            identity_sha256=self.owner['identity_sha256'],model={'UNIT.weight':torch.full((7,),2.)},
            optimizer={'state':{0:dict(step=torch.tensor(680.),exp_avg=torch.ones(7),exp_avg_sq=torch.ones(7))}},
            scheduler={'last_epoch':40},scaler={'scale':8.},rank_rng=[{'UNIT_CPU_rng':torch.arange(4)}],
            shuffle_generator=torch.arange(4,dtype=torch.uint8),state=state)
        self.best=copy.deepcopy(self.latest)
        self.best['model']['UNIT.weight']=torch.ones(7)
        self.best['state'].update(epoch=17,phase='epoch_completion',status='RUNNING',updates=289,attempts=289,
            history=history[:16],pending_epoch_completion={'full_validation':reports[16]})
        self.best['optimizer']['state'][0]['step']=torch.tensor(289.)
        self.best['scheduler']['last_epoch']=16
        self.invocation=dict(status='COMPLETE',full_training=True,debug=False,actual_CUDA=True,
            completed_epochs=40,optimizer_updates=680,connected_parameter_tensors=981 if gpu==5 else 537,
            expected_parameter_tensors=981 if gpu==5 else 537,best=self.winner,quality_verified=False)
        write_json(self.output/'request.json',self.request)
        write_json(self.output/'calibration.json',calibration)
        write_json(self.output/'training/training_identity.json',self.owner)
        self.persist()

    def persist(self):
        save_checkpoint(self.output/'training/checkpoint_latest.pt',self.latest)
        save_checkpoint(self.output/'training/checkpoint_best.pt',self.best)
        (self.output/'training/invocations.jsonl').write_text(json.dumps(self.invocation)+'\n',encoding='utf8')

    def inspect(self):
        return cp._inspect_completed_current(self.output,self.code,self.inventory,gpu=self.gpu,
            input_cache=self.cache,stunet_checkpoint=self.stu)

    def test_actual_own_best_not_last_and_all_six_numerical_states_bound(self):
        pin=self.inspect()
        self.assertEqual(pin['selected']['best_model_sha256'],digest(self.best['model']))
        self.assertNotEqual(pin['selected']['best_model_sha256'],digest(self.latest['model']))
        self.assertEqual(set(pin['BEST_checkpoint_proof']['numerical_state_sha256']),
            {'model','optimizer','scheduler','scaler','rank_rng','shuffle_generator'})
        self.assertFalse(pin['recipient_annotation_exposed'])
        self.assertEqual(pin['selection_scope'],'this completed arm only; no cross-arm winner')
        self.assertEqual(cp.pin_parameters(pin),7)

    def test_gpu6_stunet_is_independent_and_asset_sha_is_exact(self):
        self.gpu=6;self.stu=self.root/'UNIT_official_asset.model';self.stu.write_bytes(b'CPU UNIT provenance only')
        self.make_fixture();pin=self.inspect()
        self.assertEqual(pin['physical_GPU'],6)
        self.assertEqual(pin['model_contract']['encoder'],'official_pretrained_STU_Net_S')
        self.stu.write_bytes(b'changed asset')
        with self.assertRaisesRegex(ValueError,'STU checkpoint'):self.inspect()

    def test_fresh_gpu4_actual_all128_best_has_its_own_stu_and_execution_manifest(self):
        self.gpu=4;self.stu=self.root/'UNIT_official_asset.model';self.stu.write_bytes(b'CPU UNIT provenance only')
        self.make_fixture();pin=self.inspect()
        self.assertFalse((self.output/'training/training_continuation.json').exists())
        self.assertEqual(pin['physical_GPU'],4)
        self.assertEqual(pin['selected']['best_model_sha256'],digest(self.best['model']))
        self.assertEqual(pin['training_U_policy'],dict(all_U_from_epoch_one=True,initial_u=128,total_u=128,total_epochs=40))
        self.assertEqual(set(pin['source_files_sha256']),set(cli.execution_source_files(4)))
        self.assertEqual(pin['stunet_checkpoint'],str(self.stu.resolve()))
        self.assertNotIn('training_U_policy',self.request)
        changed=copy.deepcopy(pin);changed['training_U_policy']['initial_u']=7
        changed['pin_sha256']=canonical_hash({k:v for k,v in changed.items() if k!='pin_sha256'})
        with self.assertRaisesRegex(ValueError,'full128U from epoch one'):cp.validate_current_pin(changed)

    def test_gpu4_every_training_epoch_must_replay_all128_not_only_validation(self):
        self.gpu=4;self.stu=self.root/'UNIT_official_asset.model';self.stu.write_bytes(b'CPU UNIT provenance only')
        self.make_fixture()
        for key,value in (('train_probe',dict(active_u=7)),('stage_validation',dict(active_u=112)),
                ('curriculum_transition',dict(epoch=1,previous_active_u=7,next_active_u=128,expanded=True,reason='coverage_deadline'))):
            original=copy.deepcopy(self.latest);self.latest['state']['history'][0][key]=value;self.persist()
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'actual GPU4 training epoch'):self.inspect()
            self.latest=original

    def test_gpu4_asset_and_optimized_execution_sources_remain_exact(self):
        self.gpu=4;self.stu=self.root/'UNIT_official_asset.model';self.stu.write_bytes(b'CPU UNIT provenance only')
        self.make_fixture()
        self.stu.write_bytes(b'changed officialasset fixture')
        with self.assertRaisesRegex(ValueError,'STU checkpoint'):self.inspect()
        self.make_fixture()
        (self.code/'tools/run_v24_gpu4_all_u.py').write_text('# changed CPU UNIT execution source')
        with self.assertRaisesRegex(ValueError,'immutable completed v24 request'):self.inspect()

    def test_gpu4_native_cli_requires_exact_current_arm_and_official_asset(self):
        common=['--source-output','UNIT_output','--source-code','UNIT_code','--inventory','UNIT_inventory',
            '--input-cache','UNIT_cache','--output','UNIT_pin','--gpu','4']
        self.assertEqual(parse(['pin-current-gnn',*common,'--stunet-checkpoint','UNIT_official_asset']).gpu,4)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse(['pin-current-gnn',*common])
        native=self.root/'native_GPU4.json';write_json(native,dict(format=cp.CURRENT_FORMAT,physical_GPU=4))
        for action in ('calibrate-native','train'):
            self.assertEqual(parse([action,'--native',str(native),'--gpu','4']).gpu,4)
        native_runtime._require_explicit_native_gpu(native,4)
        write_json(native,dict(format=cp.CURRENT_FORMAT,physical_GPU=6))
        with self.assertRaisesRegex(ValueError,'GPU1'):native_runtime._require_explicit_native_gpu(native,4)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse(['train','--native',str(native),'--gpu','4'])
        write_json(native,dict(format=cp.FORMAT,physical_GPU=4))
        with self.assertRaisesRegex(ValueError,'GPU1'):native_runtime._require_explicit_native_gpu(native,4)

    def test_live_paused_debug_or_incomplete_latest_cannot_trigger_native(self):
        for change in ({'status':'RUNNING'},{'status':'PAUSED'},{'phase':'training'},{'updates':679}):
            original=copy.deepcopy(self.latest);self.latest['state'].update(change);self.persist()
            with self.assertRaises(ValueError):self.inspect()
            self.latest=original
        self.persist()
        for key,value in (('debug',True),('full_training',False),('completed_epochs',39),('optimizer_updates',679)):
            original=copy.deepcopy(self.invocation);self.invocation[key]=value;self.persist()
            with self.assertRaisesRegex(ValueError,'full-training invocation'):self.inspect()
            self.invocation=original

    def test_stage_metric_last_weights_and_incorrect_winner_refused(self):
        for mutation in ('stage','metric','last'):
            original=copy.deepcopy(self.best)
            if mutation=='stage':self.best['state']['pending_epoch_completion']['full_validation']['active_u']=7
            elif mutation=='metric':self.best['state']['pending_epoch_completion']['full_validation']['metrics']['per_P_patient_mrr']=.7
            else:self.best['state']['epoch']=40
            self.persist()
            with self.assertRaises(ValueError):self.inspect()
            self.best=original

    def test_nonfinite_model_and_optimizer_tensors_fail_explicitly(self):
        for key in ('model','optimizer'):
            original=copy.deepcopy(self.best)
            tensor=self.best['model']['UNIT.weight'] if key=='model' else self.best['optimizer']['state'][0]['exp_avg']
            tensor[0]=float('nan');self.persist()
            with self.assertRaises(FloatingPointError):self.inspect()
            self.best=original

    def test_pin_rechecks_source_and_checkpoint_and_never_overwrites(self):
        destination=self.root/'pin.json'
        cp.pin_completed_current(source_output=self.output,source_code=self.code,inventory_path=self.inventory,
            input_cache=self.cache,output=destination,gpu=5)
        pin,_=cp.admit_pin(destination,self.inventory);self.assertEqual(pin,self.inspect())
        with self.assertRaises(FileExistsError):
            cp.pin_completed_current(source_output=self.output,source_code=self.code,inventory_path=self.inventory,
                input_cache=self.cache,output=destination,gpu=5)
        self.best['model']['UNIT.weight'][0]=3.;self.persist()
        with self.assertRaisesRegex(ValueError,'proof changed'):cp.admit_pin(destination,self.inventory)

    def test_signed_pin_gpu_or_model_contract_tampering_is_rejected(self):
        pin=self.inspect()
        for key,value in (('physical_GPU',6),('completed_epochs',39),('recipient_annotation_exposed',True)):
            changed=copy.deepcopy(pin);changed[key]=value
            changed['pin_sha256']=canonical_hash({k:v for k,v in changed.items() if k!='pin_sha256'})
            with self.assertRaises(ValueError):cp.validate_current_pin(changed)

    def test_exact_gpu_binding_retains_historical_gpu1_guard(self):
        pin=self.inspect();cp._admit_arm_gpu(pin,5)
        with self.assertRaises(ValueError):cp._admit_arm_gpu(pin,6)
        with self.assertRaises(ValueError):cp._admit_arm_gpu({'format':cp.PIN_FORMAT},5)

    def test_cli_explicit_current_actions_and_native_binding(self):
        args=parse(['pin-current-gnn','--source-output','UNIT_output','--source-code','UNIT_code',
            '--inventory','UNIT_inventory','--input-cache','UNIT_cache','--output','UNIT_pin','--gpu','5'])
        self.assertEqual(args.gpu,5)
        native=self.root/'native.json';write_json(native,dict(format=cp.CURRENT_FORMAT,physical_GPU=5))
        self.assertEqual(parse(['train','--native',str(native),'--gpu','5']).gpu,5)
        native_runtime._require_explicit_native_gpu(native,5)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            parse(['train','--native',str(native),'--gpu','6'])
        with self.assertRaisesRegex(ValueError,'GPU1'):
            native_runtime._require_explicit_native_gpu(native,6)
        write_json(native,dict(format=cp.FORMAT))
        with self.assertRaisesRegex(ValueError,'GPU1'):
            native_runtime._require_explicit_native_gpu(native,5)


if __name__=='__main__':unittest.main()
