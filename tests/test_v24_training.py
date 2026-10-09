"""CPU DEBUG policy/state/optimizer tests; not native training results."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hiercp_v1x.v24_targets import new_curriculum,finish_curriculum_epoch,validate_policy
from hiercp_v1x.v24_training import optimizer_groups,_validate_progress,V24Scorer,patient_batches,validate_calibration,validate_saved_best,FORMAT
from hiercp_v1x.v23_training import patient_balanced_objective,best_selection_key,parallel_patient_batches
from tools.run_v24_all_p import validate_config,parse

ROOT=Path(__file__).resolve().parents[1]
POLICY=json.loads((ROOT/'config/v24_gpu5_GT_blind.json').read_text())['v24_runtime']['curriculum']


def report(epoch,active,phase,mrr=.6,top1=.4):
    return dict(epoch=epoch,active_u=active,phase=phase,evaluation_view_epoch=29,all_P_scored=True,
        cases=[dict(case_id='UNIT_A'),dict(case_id='UNIT_B')],
        metrics=dict(per_P_patient_mrr=mrr,per_P_patient_top1=top1,patient_balanced_pair_loss=.2))


def step(state,*,val=(.6,.4),train=(.7,.5)):
    epoch=state['last_completed_epoch']+1; active=state['active_u']
    return finish_curriculum_epoch(state,report(epoch,active,'train_probe',*train),
        report(epoch,active,'stage_validation',*val),epoch=epoch,
        actual_train_cases=['UNIT_A','UNIT_B'],expected_train_cases=['UNIT_A','UNIT_B'],updates_in_epoch=1)


class V24CurriculumDebug(unittest.TestCase):
    def test_full_bank_by33_with8_full_epochs_even_never_mastered(self):
        state=new_curriculum(POLICY); before=[]
        for epoch in range(1,41):
            before.append(state['active_u']); state,receipt=step(state)
            if epoch%4==0 and epoch<=32:
                self.assertEqual(receipt['reason'],'coverage_deadline')
                self.assertTrue(receipt['coverage_is_not_mastery'])
                self.assertFalse(receipt['validation_gate_passed'])
        self.assertEqual(before,[u for u in (7,23,39,55,71,87,103,119) for _ in range(4)]+[128]*8)
        self.assertEqual(state['last_completed_epoch'],40)
        self.assertEqual(len(state['history']),40)

    def test_validation_requires_two_consecutive_completed_epochs(self):
        state=new_curriculum(POLICY);state,one=step(state,val=(.8,.7))
        self.assertFalse(one['expanded'])
        state,two=step(state,val=(.8,.7))
        self.assertTrue(two['expanded']);self.assertEqual(two['reason'],'validation_mastery')
        self.assertEqual(state['active_u'],23)

    def test_failed_validation_resets_consecutive_passes(self):
        state=new_curriculum(POLICY)
        state,_=step(state,val=(.9,.8));state,_=step(state,val=(.79,.8))
        state,row=step(state,val=(.9,.8))
        self.assertFalse(row['expanded']);self.assertEqual(row['validation_pass_streak'],1)

    def test_overfit_is_mastery_and_gap_and_plateau_not_one_condition(self):
        state=new_curriculum(POLICY)
        for _ in range(2):state,row=step(state,train=(.95,.90))
        self.assertFalse(row['expanded'])
        state,row=step(state,train=(.95,.90))
        self.assertEqual(row['reason'],'train_mastery_overfit_plateau')
        self.assertTrue(row['train_mastery']);self.assertEqual(row['plateau_epochs'],2)

    def test_low_train_top1_cannot_claim_overfit_mastery(self):
        state=new_curriculum(POLICY)
        for _ in range(3):state,row=step(state,train=(.95,.84))
        self.assertFalse(row['expanded']);self.assertFalse(row['train_mastery'])

    def test_gap_must_meet_threshold(self):
        state=new_curriculum(POLICY)
        for _ in range(3):state,row=step(state,val=(.85,.6),train=(.90,.90))
        self.assertFalse(row['expanded']);self.assertLess(row['train_minus_stage_val_mrr'],.1)

    def test_plateau_reference_resets_on_meaningful_progress(self):
        state=new_curriculum(POLICY)
        state,_=step(state);state,_=step(state)
        state,row=step(state,val=(.61,.4),train=(.95,.9))
        self.assertFalse(row['expanded']);self.assertEqual(row['plateau_epochs'],0)

    def test_every_admission_resets_stage_reference_and_patience(self):
        state=new_curriculum(POLICY)
        for _ in range(3):state,_=step(state,train=(.95,.9))
        self.assertEqual(state['active_u'],23);self.assertIsNone(state['plateau_reference'])
        self.assertEqual(state['plateau_epochs'],0)
        state,row=step(state,train=(.95,.9))
        self.assertEqual(row['plateau_epochs'],0);self.assertFalse(row['expanded'])

    def test_infeasible_original_plus7_max4_has_no_silent_extension(self):
        policy=copy.deepcopy(POLICY);policy['increment_u']=7
        with self.assertRaisesRegex(ValueError,'no silent extension'):validate_policy(policy)

    def test_duplicate_probe_and_incomplete_epoch_rejected(self):
        state=new_curriculum(POLICY);probe=report(1,7,'train_probe');probe['cases'].append(dict(case_id='UNIT_A'))
        with self.assertRaisesRegex(ValueError,'every ranking'):
            finish_curriculum_epoch(state,probe,report(1,7,'stage_validation'),epoch=1,
                actual_train_cases=['UNIT_A','UNIT_B'],expected_train_cases=['UNIT_A','UNIT_B'],updates_in_epoch=1)
        with self.assertRaisesRegex(ValueError,'Complete nonduplicated'):
            finish_curriculum_epoch(state,report(1,7,'train_probe'),report(1,7,'stage_validation'),epoch=1,
                actual_train_cases=['UNIT_A'],expected_train_cases=['UNIT_A','UNIT_B'],updates_in_epoch=1)

    def test_probe_must_share_fixed_view_stage_and_epoch(self):
        state=new_curriculum(POLICY);probe=report(1,7,'train_probe');probe['evaluation_view_epoch']=1
        with self.assertRaisesRegex(ValueError,'Same-stage'):
            finish_curriculum_epoch(state,probe,report(1,7,'stage_validation'),epoch=1,
                actual_train_cases=['UNIT_A','UNIT_B'],expected_train_cases=['UNIT_A','UNIT_B'],updates_in_epoch=1)

    def test_full_best_separate_from_gate(self):
        state=new_curriculum(POLICY)
        state,one=step(state,val=(.8,.7));state,two=step(state,val=(.8,.7))
        self.assertTrue(two['expanded']);self.assertFalse(two['full_validation_used_for_curriculum'])
        weak=report(1,128,'full_validation',.3,.1);strong=report(2,128,'full_validation',.4,.2)
        self.assertGreater(best_selection_key(strong),best_selection_key(weak))


class V24TrainingContractDebug(unittest.TestCase):
    def test_every_parameter_in_optimizer_and_real_update_all(self):
        torch.manual_seed(42);net=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.SiLU(),torch.nn.Linear(5,1))
        optimizer=torch.optim.AdamW(optimizer_groups(net,dict(lr=1e-4,weight_decay=.01)))
        initial=[p.detach().clone() for p in net.parameters()]
        score=net(torch.tensor([[1.,2.,3.],[2.,-1.,.5],[-3.,1.,2.]])).flatten()
        loss,_=patient_balanced_objective((score,),((0,1),),torch.zeros(3))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))
        optimizer.step();self.assertTrue(all(not torch.equal(a,b) for a,b in zip(initial,net.parameters())))

    def test_custom_groups_cannot_duplicate_or_omit_parameters(self):
        net=torch.nn.Linear(2,1);p=list(net.parameters())
        net.v24_optimizer_parameter_groups=lambda lr,wd:[dict(params=[p[0],p[0]],lr=lr,weight_decay=wd)]
        with self.assertRaisesRegex(ValueError,'exactly once'):optimizer_groups(net,dict(lr=1e-4,weight_decay=.01))

    def test_gt_provider_rejected_before_any_neural_call(self):
        net=torch.nn.Linear(2,1);provider=SimpleNamespace(recipient_GT_used_in_forward=True)
        with self.assertRaisesRegex(ValueError,'before forward'):
            V24Scorer(net,{'inner_train':provider,'inner_val':provider},lambda *a:None,
                physical_candidate_batch=32,checkpoint_local_chunks=True,amp=False,budget=None)

    def test_legacy_geometry_rejected_before_original_scorer_forward(self):
        net=torch.nn.Linear(2,1);provider=SimpleNamespace(recipient_GT_used_in_forward=False,
            ds=SimpleNamespace(rows=[dict(id='UNIT_record')]))
        scorer=V24Scorer(net,{'inner_train':provider,'inner_val':provider},lambda *a:None,
            physical_candidate_batch=32,checkpoint_local_chunks=True,amp=False,budget=None)
        with patch('hiercp_v1x.v24_training.V23Scorer.forward',side_effect=AssertionError('neural call')):
            with self.assertRaisesRegex(ValueError,'before forward'):
                scorer([SimpleNamespace(partition='inner_train')],epoch=1,training=True)

    def test_resume_cursor_must_represent_complete_patient_batch(self):
        train=[str(i) for i in range(65)];val=['v'+str(i) for i in range(21)]
        state=dict(phase='train_probe',epoch=2,history=[{}],train_order=train,train_position=65,
            train_rows=[dict(case_id=case) for case in train],evaluation_position=4,
            evaluation_rows=[dict(case_id=case) for case in train[:4]],updates=17,attempts=17)
        _validate_progress(state,train,val,4,40,SimpleNamespace(last_epoch=1))
        state['evaluation_position']=3;state['evaluation_rows']=state['evaluation_rows'][:3]
        with self.assertRaisesRegex(ValueError,'complete physical batch'):
            _validate_progress(state,train,val,4,40,SimpleNamespace(last_epoch=1))

    def test_complete_patient_batching_no_drop_or_duplicate(self):
        cases=list(range(65));batches=patient_batches(cases,4)
        self.assertEqual([case for batch in batches for case in batch],cases)
        self.assertTrue(all(len(batch)>=2 for batch in batches))

    def test_configs_differ_only_encoder_experiment_gpu(self):
        five=json.loads((ROOT/'config/v24_gpu5_GT_blind.json').read_text())
        six=json.loads((ROOT/'config/v24_gpu6_STUNetS_GT_blind.json').read_text())
        validate_config(five,5);validate_config(six,6,Path('explicit_official_asset'))
        for value in (five,six):
            value.pop('encoder');value.pop('experiment');value['v24_runtime'].pop('physical_GPU')
        self.assertEqual(five,six)

    def test_STU_random_substitute_and_wrong_gpu_refused(self):
        six=json.loads((ROOT/'config/v24_gpu6_STUNetS_GT_blind.json').read_text())
        with self.assertRaisesRegex(ValueError,'supplied explicitly'):validate_config(six,6)
        with self.assertRaisesRegex(ValueError,'singleGPU'):validate_config(six,5,Path('asset'))

    def test_cli_distinct_shared_inputs_independent_results(self):
        args=parse(['--mode','prepare','--native-experiment','native','--inventory','inventory.json',
            '--input-cache','new_GT_free_shared','--output','GPU5','--gpu','5'])
        self.assertEqual(args.gpu,5);self.assertNotEqual(args.input_cache,args.output)
        self.assertIsNone(args.resume)

    def test_factory_activates_sealed_archive_before_original_import(self):
        import ast,inspect
        from hiercp_v1x.v24_factory import V24NativeInputs
        import textwrap
        tree=ast.parse(textwrap.dedent(inspect.getsource(V24NativeInputs.__init__)))
        activated=[node.lineno for node in ast.walk(tree) if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Name) and node.func.id=='_load_geometry_inputs']
        imports=[node.lineno for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)
            and node.module and (node.module=='hiercp' or node.module.startswith('hiercp.'))]
        self.assertEqual(len(activated),1);self.assertTrue(all(line>activated[0] for line in imports))

    def test_selected_calibration_requires_all_real_gradients_and_repeats(self):
        from hiercp_v1x.u_bridge_training import digest
        net=torch.nn.Linear(2,1);scorer=SimpleNamespace(physical_candidate_batch=32)
        gradient=dict(finite=True,missing=[],trainable_parameter_tensors=2,gradient_present=2)
        trial=dict(physical_patient_batch=4,physical_candidate_batch=32,accepted=True,all_P_and_U128=True,
            measurements=[dict(seconds=.1,gradient=copy.deepcopy(gradient)) for _ in range(3)],
            optimizer_updates_on_clone=3,peak_cuda_bytes=100,workload=dict(patients=4,upper_invocations=1))
        report=dict(format=FORMAT,initial_state_sha256=digest(net.state_dict()),world_size=1,
            selected_physical_patient_batch=4,selected_physical_candidate_batch=32,model_contract={},
            recipient_GT_used_in_forward=False,debug=True,original_model_and_RNG_preserved=True,
            measured_full_P_U128_backward=True,trials=[trial])
        validate_calibration(report,net,scorer,{},4,repeats=3,budget=SimpleNamespace(cuda_bytes=1000),debug=True)
        trial['measurements'][1]['gradient']['missing']=['weight']
        with self.assertRaisesRegex(ValueError,'finite loss gradient'):
            validate_calibration(report,net,scorer,{},4,repeats=3,budget=SimpleNamespace(cuda_bytes=1000),debug=True)

    def test_resume_preserves_actual_best_or_rejects_missing_and_corrupt(self):
        import uuid
        from hiercp_v1x.u_bridge_training import digest,atomic_save
        root=ROOT/'outputs'/('v24_BEST_CPU_UNIT_'+uuid.uuid4().hex);root.mkdir()
        full=report(3,128,'full_validation',.4,.2)
        best=dict(epoch=3,updates=7,selection_key=list(best_selection_key(full)))
        state=dict(best=best)
        with self.assertRaisesRegex(ValueError,'missing'):validate_saved_best(root,state,'UNIT_identity')
        payload=dict(format=FORMAT,identity_sha256='UNIT_identity',model=dict(weight=torch.ones(1)),
            state=dict(epoch=3,updates=7,best=best,pending_epoch_completion=dict(full_validation=full)))
        payload['content_sha256']=digest(payload);atomic_save(root/'checkpoint_best.pt',payload)
        validate_saved_best(root,state,'UNIT_identity')
        payload['model']['weight'].add_(1);atomic_save(root/'checkpoint_best.pt',payload)
        with self.assertRaisesRegex(ValueError,'content/identity'):validate_saved_best(root,state,'UNIT_identity')

    def test_geometry_RSS_callback_owns_raw_lock_under_parallel_workers(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from hiercp_v1x.v24_factory import V24NativeInputs
        raw=object.__new__(V24NativeInputs);raw._lock=threading.RLock()
        raw._rss_locked=lambda:raw._lock._is_owned()
        with ThreadPoolExecutor(max_workers=4)as pool:
            self.assertEqual(list(pool.map(lambda _:raw._rss(),range(16))),[True]*16)


if __name__=='__main__':unittest.main()
