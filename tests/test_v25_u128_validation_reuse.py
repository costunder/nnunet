"""CPU DEBUG synthetic model/population, real backward and checkpoint resume.

Only model admission is mocked. Production control-flow/report flags remain
exact for the unchanged resume validator; these artifacts are not real CT,
native training, CUDA validation or quality evidence.
"""
from contextlib import ExitStack,redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from hiercp_v1x import v25_training as training,v25_u128_validation_reuse as reuse
from hiercp_v1x.u_bridge_training import digest
from tests.test_v25_targets import policy


class U128ReuseCPUDebug(unittest.TestCase):
    def setUp(self):
        self.original=training.run_training
        self.addCleanup(setattr,training,'run_training',self.original)
        self.addCleanup(training.__dict__.pop,reuse._GATE_NAME,None)

    def exercise(self,initial=128,pause_phase=None,pause_position=None,install_first=True,legacy_difference=False):
        torch.manual_seed(42);net=torch.nn.Linear(2,1);initial_state=copy.deepcopy(net.state_dict())
        train=[f'CPU_DEBUG_train_{i:03d}'for i in range(65)];val=[f'CPU_DEBUG_val_{i:03d}'for i in range(21)]
        positives={case:8+int(i>=58)for i,case in enumerate(train)}
        positives.update({case:8 if i<15 else 15 if i==15 else 0 for i,case in enumerate(val)})
        current={};calls=[];prepared=[];save_hooks=[]
        class Population:
            def partition_cases(self,partition,**kwargs):return train if partition=='inner_train'else val
            def case(self,case,active_u_count=128,**kwargs):
                count=positives[case]+active_u_count
                rows=[dict(id=case+':'+str(i),case_id=case,donor_case_id='CPU_DEBUG_independent_donor',
                    donor_component=1,center=[i,0,0],target=int(i<positives[case]))for i in range(count)]
                return SimpleNamespace(case_id=case,rows=rows,record_ids=tuple(r['id']for r in rows),
                    positive_indices=tuple(range(positives[case])),unobserved_indices=tuple(range(positives[case],count)),
                    partition='inner_train'if case in train else'inner_val')
            def manifest(self):return dict(CPU_DEBUG_SYNTHETIC=True,train65=train,val21=val,observed_P=662)
        class Scorer:
            def __init__(self):
                self.net=net;self.amp=False;self.physical_candidate_batch=32
                self.geometry=SimpleNamespace(prepare=lambda active:prepared.append((current.get('phase'),active)))
            def __call__(self,plans,*,epoch,training):
                calls.append(dict(phase=current.get('phase'),epoch=epoch,training=training,
                    cases=[p.case_id for p in plans],U=len(plans[0].unobserved_indices),grad=torch.is_grad_enabled()))
                scores=[]
                for plan in plans:
                    features=torch.arange(len(plan.record_ids)*2,dtype=torch.float32).reshape(-1,2)/1000
                    value=self.net(features).flatten()
                    if legacy_difference and current.get('phase')=='full_validation':
                        value=value.clone();value[0]+=0.001
                    scores.append(value)
                total=sum(len(value)for value in scores)
                return SimpleNamespace(scores=tuple(scores),consistency=self.net.bias.square().expand(total),
                    workload=dict(patients=len(plans),upper_invocations=1,CPU_DEBUG_SYNTHETIC=True))
        scorer=Scorer();population=Population()
        gradient=dict(finite=True,missing=[],trainable_parameter_tensors=2,gradient_present=2)
        trial=dict(physical_patient_batch=4,physical_candidate_batch=32,accepted=True,all_P_and_U128=True,
            measurements=[dict(seconds=.1,gradient=copy.deepcopy(gradient))for _ in range(3)],
            optimizer_updates_on_clone=3,peak_cuda_bytes=100,workload=dict(patients=4,upper_invocations=1))
        contract=dict(parameters=3,trainable_parameters=3,recipient_GT_used_in_forward=False,
            trained_v23_weights_loaded=False,CPU_DEBUG_SYNTHETIC_MODEL=True)
        calibration=dict(format=training.FORMAT,initial_state_sha256=digest(net.state_dict()),world_size=1,
            selected_physical_patient_batch=4,selected_physical_candidate_batch=32,model_contract=contract,
            recipient_GT_used_in_forward=False,debug=False,original_model_and_RNG_preserved=True,
            measured_full_P_U128_backward=True,trials=[trial])
        config=dict(v24_runtime=dict(curriculum=policy(initial,None if initial==128 else initial),
            batch_calibration=calibration,calibration_repeats=3),training=dict(lr=1e-4,weight_decay=.01,
            fused_optimizer=False,amp=False,grad_clip=1.,consistency_weight=.1,gradient_accumulation_steps=1))
        with tempfile.TemporaryDirectory(prefix='CPU_DEBUG_SYNTHETIC_U128_')as temporary:
            root=Path(temporary);marker=root/'STOP_AFTER_BATCH';paused=[];prefix=None;pause_requested=False
            def save(path,payload):
                nonlocal pause_requested
                save_hooks.append(Path(path).name);torch.save(payload,path)
                if Path(path).name!='checkpoint_latest.pt':return
                current.clear();current.update(payload['state'])
                if (not pause_requested and pause_phase and current['phase']==pause_phase
                        and current['evaluation_position']==pause_position):
                    marker.write_text('CPU DEBUG owned pause');pause_requested=True
                if current['phase']=='training'and len(current['history'])==1:marker.write_text('CPU DEBUG complete epoch pause')
            if install_first:reuse.install_runtime(training)
            with ExitStack()as stack:
                # This hook is intentionally installed AFTER the AST clone.
                stack.enter_context(patch.object(training,'atomic_save',side_effect=save))
                stack.enter_context(patch.object(training,'validate_model',side_effect=lambda model,*a,**k:dict(CPU_DEBUG_SYNTHETIC=model)))
                stack.enter_context(patch.object(training,'_groups',side_effect=lambda model:dict(CPU_DEBUG=model)))
                for name in ('reset_peak_memory_stats','synchronize'):stack.enter_context(patch('torch.cuda.'+name))
                stack.enter_context(patch('torch.cuda.max_memory_allocated',return_value=0))
                stack.enter_context(redirect_stdout(io.StringIO()))
                def run():return training.run_training(net,scorer,population,config,output=root,
                    identity=dict(CPU_DEBUG_SYNTHETIC=True),budget=SimpleNamespace(cuda_bytes=1000,check=lambda:None),
                    model_contract=contract,physical_patient_batch=4,workers=4,debug=False)
                first=run()
                if pause_phase:
                    saved=torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
                    paused.append(copy.deepcopy(saved['state']));prefix=copy.deepcopy(saved['state']['evaluation_rows'])
                    marker.unlink();net.load_state_dict(initial_state);config['v24_runtime']['resume_checkpoint']=str(root/'checkpoint_latest.pt')
                    if not install_first:reuse.install_runtime(training)
                    second=run()
                else:second=first
                latest=torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
                best=torch.load(root/'checkpoint_best.pt',map_location='cpu',weights_only=False)
            timing=[json.loads(line)for line in (root/'validation_reuse.jsonl').read_text().splitlines()]if(root/'validation_reuse.jsonl').exists()else[]
            return dict(receipt=second,calls=calls,prepared=prepared,latest=latest,best=best,
                timing=timing,paused=paused,prefix=prefix,save_hooks=save_hooks,config=config)

    def assert_complete_epoch(self,value):
        state=value['latest']['state'];receipt=value['receipt']
        self.assertEqual((receipt['completed_epochs'],receipt['optimizer_updates'],receipt['status']),(1,17,'PAUSED'))
        self.assertFalse(receipt['actual_CUDA']);self.assertFalse(receipt['full_training'])
        self.assertEqual(value['config']['v24_runtime']['curriculum']['total_epochs'],40)
        self.assertEqual(state['best']['selected_by'],'fixed_full128_validation_only')
        training.validate_curriculum_progress(state,policy=state['curriculum']['policy'],
            train_cases=value['latest']['state']['curriculum']['training_case_ids'],
            val_cases=value['latest']['state']['curriculum']['validation_case_ids'],physical_patient_batch=4)
        self.assertEqual(digest({k:v for k,v in value['latest'].items()if k!='content_sha256'}),value['latest']['content_sha256'])
        self.assertGreater(len(value['save_hooks']),17)
        self.assertEqual({int(row['step'])for row in value['latest']['optimizer']['state'].values()},{17})

    def test_U128_one_real_stage_pass_skips_full_geometry_input_and_forward_then_original_BEST(self):
        value=self.exercise();self.assert_complete_epoch(value)
        calls=value['calls'];stage=[c for c in calls if c['phase']=='stage_validation'];full=[c for c in calls if c['phase']=='full_validation']
        self.assertEqual(sum(len(c['cases'])for c in stage),21);self.assertEqual(full,[])
        self.assertNotIn(('full_validation',128),value['prepared'])
        curve=value['latest']['state']['history'][0]
        expected=copy.deepcopy(curve['stage_validation']);expected['phase']='full_validation'
        self.assertEqual(digest(expected),digest(curve['full_validation']))
        self.assertEqual(value['timing'][-1]['new_forward_calls'],0)
        self.assertEqual(value['timing'][-1]['actual_source_P'],135)
        self.assertEqual(len([c for c in calls if c['training']]),17)
        self.assertTrue(all(c['epoch']==29 and not c['grad']for c in calls if not c['training']))

    def test_U8_retains_both_real_21_patient_passes(self):
        value=self.exercise(initial=8);self.assert_complete_epoch(value)
        for phase,U in [('stage_validation',8),('full_validation',128)]:
            rows=[c for c in value['calls']if c['phase']==phase]
            self.assertEqual(sum(len(c['cases'])for c in rows),21);self.assertTrue(all(c['U']==U for c in rows))
        self.assertEqual(value['timing'],[])

    def test_mid_stage_checkpoint_finishes_remaining_actual_cases_then_reuses_full(self):
        value=self.exercise(pause_phase='stage_validation',pause_position=8);self.assert_complete_epoch(value)
        self.assertEqual(value['paused'][0]['evaluation_position'],8)
        self.assertEqual(sum(len(c['cases'])for c in value['calls']if c['phase']=='stage_validation'),21)
        self.assertFalse(any(c['phase']=='full_validation'for c in value['calls']))

    def test_legacy_mid_full_differing_scores_are_retained_and_only_remaining_actual_batches_run(self):
        value=self.exercise(pause_phase='full_validation',pause_position=8,install_first=False,legacy_difference=True)
        self.assert_complete_epoch(value)
        curve=value['latest']['state']['history'][0]
        self.assertEqual(digest(curve['full_validation']['cases'][:8]),digest(value['prefix']))
        self.assertNotEqual(digest(curve['stage_validation']['cases'][:8]),digest(value['prefix']))
        self.assertEqual(sum(len(c['cases'])for c in value['calls']if c['phase']=='full_validation'),21)
        self.assertTrue(value['timing'][-1]['legacy_partial_full_preserved'])
        self.assertFalse(value['timing'][-1]['reused'])

    def test_legacy_full_position0_resume_reuses_completed_actual_stage(self):
        value=self.exercise(pause_phase='full_validation',pause_position=0,install_first=False);self.assert_complete_epoch(value)
        self.assertFalse(any(c['phase']=='full_validation'for c in value['calls']))
        self.assertTrue(value['timing'][-1]['resumed_completed_stage'])

    def test_source_and_original_callable_admission_and_idempotency(self):
        with self.assertRaisesRegex(ValueError,'frozen'):reuse.install_runtime(training,expected_source_sha256='0'*64)
        receipt=reuse.install_runtime(training);self.assertEqual(receipt,reuse.install_runtime(training))
        self.assertIs(training.run_training.__globals__,training.__dict__)
        self.assertEqual(reuse._sha(training.__file__),receipt['frozen_training_source_sha256'])
        self.assertFalse(receipt['checkpoint_identity_schema_changed'])


if __name__=='__main__':unittest.main()
