"""DEBUG CPU admission/dispatch tests; no actual model, GPU or remote jobs."""
import ast
import copy
import inspect
import io
import json
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import v24_early_stop_downstream as early
from hiercp_v1x import v24_scored_native_chain as chain
from tests import test_v24_scored_native_chain as native_fixtures
from tools import run_v24_early_stop_downstream as cli


class EarlyDownstreamCPU_DEBUG(unittest.TestCase):
    def setUp(self):
        self.fixture = native_fixtures.ScoredNativeCPUUnit(methodName='test_three_original_native_contracts_and_exact_external_score_paths')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.base = self.fixture.base
        self.native = self.fixture.request(5)
        self.native_path = self.fixture.write(Path(self.native['chain_root'])/'request.json',self.native)
        self.authorization = self.fixture.write(self.base/'authorization.json',dict(DEBUG_fixture=True,target_epoch=27))
        self.controller = self.base/'early_controller'; self.controller.mkdir()
        self.scoring = self.base/'early_scoring'; self.scoring.mkdir()
        self.actors = [dict(pid=101+i,create_time=1000.+i,command=['python','owned-'+str(i)],UID=41833) for i in range(3)]
        self.request = early.build_request(self.native_path,controller_root=self.controller,scoring_root=self.scoring,
            code=self.fixture.code,commit='e'*40,code_files={name:'a'*64 for name in set(chain.FILES)|set(early.FILES)},
            target_epoch=27,authorization_path=self.authorization,release_actors=self.actors,protected_files=[])

    def test_truthful_early_contract_reuses_unchanged_full_native_storage(self):
        self.assertEqual(self.request['resource_contract'],self.native)
        self.assertFalse(self.request['full_GNN_training'])
        self.assertEqual((self.request['target_epoch'],self.request['original_planned_epochs']),(27,40))
        self.assertEqual((self.request['source_entries'],self.request['CP_positions']),(642,82176))
        self.assertEqual((self.request['native_epochs'],self.request['native_physical_batch'],self.request['full_native_evaluation_cases']),(250,2,26))
        self.assertEqual(tuple(row['action'] for row in self.request['stages']),early.ACTIONS)
        self.assertEqual(self.request['pin_path'],str(self.controller/'own_early_stop_best_pin.json'))
        self.assertEqual(chain.read(self.native_path),self.native)

    def test_false_full_completion_or_reduced_downstream_contract_is_rejected(self):
        for key,value in dict(full_GNN_training=True,target_epoch=40,source_entries=641,CP_positions=82175,
                native_epochs=249,native_physical_batch=1,full_native_evaluation_cases=25,debug=True,context_depth1=False).items():
            with self.subTest(key=key):
                changed=copy.deepcopy(self.request);changed[key]=value
                with self.assertRaises(ValueError):early.validate_request(changed)
        changed=copy.deepcopy(self.request);changed['resource_contract']['epochs']=249
        with self.assertRaises(ValueError):early.validate_request(changed)

    def test_overlap_pid_identity_and_added_resume_are_rejected(self):
        changed=copy.deepcopy(self.request);changed['scoring_root']=self.native['scoring_root']
        with self.assertRaisesRegex(ValueError,'overlaps'):early.validate_request(changed)
        changed=copy.deepcopy(self.request);changed['release_actors'][0]['UID']=0
        with self.assertRaisesRegex(ValueError,'PID/birth'):early.validate_request(changed)
        changed=copy.deepcopy(self.request);changed['stages'][-1]['command'].append('--resume')
        with self.assertRaisesRegex(ValueError,'commands'):early.validate_request(changed)

    def test_private_admission_does_not_modify_original_functions_or_full40_gate(self):
        from hiercp_v1x import v24_matched_basic_cp_scoring as scoring
        from hiercp_v1x import v24_matched_basic_cp_runtime as runtime
        from hiercp_v1x import v24_matched_basic_cp_pipeline as pipeline
        from hiercp_v1x import v24_nnunet_cp as original
        original_functions=(scoring._score_matched_basic_bank,runtime.validate_overlay,pipeline.train,original.validate_current_pin)
        original_hashes=[chain.sha(module.__file__) for module in (scoring,runtime,pipeline,original)]
        modules=early.private_modules()
        self.assertEqual(original.CURRENT_PIN_FORMAT,'v24_completed_single_arm_GT_blind_best_v1')
        self.assertEqual(early.CURRENT_PIN_FORMAT,'v24_user_authorized_epoch_boundary_early_stop_best_v1')
        self.assertEqual(original_functions,(scoring._score_matched_basic_bank,runtime.validate_overlay,pipeline.train,original.validate_current_pin))
        self.assertEqual(original_hashes,[chain.sha(module.__file__) for module in (scoring,runtime,pipeline,original)])
        self.assertIsNot(modules['EARLY_RUNTIME']._BINDINGS,runtime._BINDINGS)
        self.assertEqual(modules['EARLY_PIPELINE'].FILES,pipeline.FILES)
        self.assertEqual(modules['proofs']['pipeline']['child_entry_redirects']['tools/run_v24_matched_basic_cp.py'],1)
        for proof in modules['proofs'].values():
            self.assertFalse(proof['model_geometry_loss_optimizer_and_loader_expressions_changed'])
            self.assertFalse(proof['original_module_globals_changed'])
        body=inspect.getsource(scoring._score_matched_basic_bank)
        self.assertIn('_inspect_completed_current(',body)
        original_admission=inspect.getsource(original.validate_current_pin)
        self.assertIn("pin.get('completed_epochs')!=40",original_admission)
        self.assertIn("pin.get('optimizer_updates')!=680",original_admission)

    def test_scorer_final_inspection_uses_exact_early_authorization_and_epoch(self):
        from hiercp_v1x import v24_early_stop_gnn as pin_api
        expected=dict(DEBUG_fixture=True,full_training=False,completed_epochs=27)
        with (patch.object(early,'environment_request',return_value=(self.request,self.controller/'request.json')),
                patch.object(pin_api,'inspect_early_stop',return_value=expected) as inspect_pin):
            result=early._inspect_completed_current(self.request['source_output'],self.request['source_code'],self.native['inventory'],
                gpu=5,input_cache=self.native['input_cache'],stunet_checkpoint=None)
        self.assertEqual(result,expected)
        self.assertEqual(inspect_pin.call_args.kwargs['target_epoch'],27)
        self.assertEqual(inspect_pin.call_args.kwargs['authorization_path'],str(self.authorization))
        with self.assertRaisesRegex(ValueError,'STU'):
            with patch.object(early,'environment_request',return_value=(self.request,None)):
                early._inspect_completed_current('source','code','inventory',gpu=6,input_cache='cache')

    def test_live_original_actor_prevents_all_downstream_readiness(self):
        actor=object()
        with patch.object(chain,'process_snapshot',side_effect=[None,actor]):
            self.assertFalse(early.released(self.request))
        controller=early.Controller.__new__(early.Controller)
        controller.request=self.request;controller.cache=Mock();controller.update=Mock()
        with (patch.object(early,'released',return_value=False),patch.object(chain,'storage_ready') as storage,
                patch.object(chain,'resources_ready') as resources):
            self.assertFalse(controller.ready('score'))
        storage.assert_not_called();resources.assert_not_called()
        self.assertEqual(controller.update.call_args.kwargs['status'],'WAITING_FOR_EARLY_STOP_RELEASE')

    def test_pin_boundary_binds_authorization_and_all_frozen_source_files(self):
        artifact=self.base/'paused_source.json';artifact.write_text('{"PAUSED":true}',encoding='utf8')
        cache=chain.FileProofCache();proof=cache.capture(artifact)
        self.fixture.write(self.request['pin_path'],{'DEBUG_fixture':'metadata boundary only'})
        pin=dict(completed_epochs=27,source_output=self.request['source_output'],source_code=self.request['source_code'],
            inventory=self.native['inventory'],input_cache=self.native['input_cache'],
            early_stop_authorization=self.request['authorization'],early_stop_source_file_proofs={str(artifact):proof})
        with patch.object(early,'validate_current_pin',return_value=pin):
            self.assertEqual(early.verify_pin_binding(self.request,cache),pin)
            artifact.write_text('{"PAUSED":false}',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'identity changed'):early.verify_pin_binding(self.request,cache)
        pin['early_stop_authorization']=dict(self.request['authorization'],sha256='0'*64)
        with patch.object(early,'validate_current_pin',return_value=pin):
            with self.assertRaisesRegex(ValueError,'authorization differs'):early.verify_pin_binding(self.request,cache)

    def test_new_readiness_uses_original_storage_and_never_original_full40_source_gate(self):
        controller=early.Controller.__new__(early.Controller)
        controller.request=self.request;controller.cache=Mock();controller.update=Mock()
        with (patch.object(early,'released',return_value=True),patch.object(chain,'source_complete',side_effect=AssertionError('full40 gate must stay unused')),
                patch.object(chain,'storage_ready',return_value={'actual':'storage'}) as storage,
                patch.object(chain,'resources_ready',return_value={'actual':'GPU'}) as resources):
            self.assertTrue(controller.ready('score'))
        storage.assert_called_once_with(self.native,controller.cache)
        resources.assert_called_once_with(self.native,'prepare-native')

    def test_stage_failure_stops_before_any_native_stage_and_keeps_honest_flags(self):
        controller=early.Controller.__new__(early.Controller)
        controller.request=self.request;controller.state={'full_GNN_training':False,'full_native_training_completed':False}
        controller.update=Mock();controller.ready=Mock(return_value=True)
        controller.execute=Mock(side_effect=RuntimeError('actual score failure'))
        with self.assertRaisesRegex(RuntimeError,'actual score failure'):controller.run()
        self.assertEqual(controller.execute.call_count,1)
        self.assertEqual(controller.update.call_args.kwargs['status'],'FAILED')
        self.assertFalse(controller.state['full_GNN_training'])
        self.assertFalse(controller.state['full_native_training_completed'])

    def test_exclusive_lease_and_global_source_claim_block_replay_or_second_namespace(self):
        actor=types.SimpleNamespace(pid=900,create_time=lambda:1000.,cmdline=lambda:['python','owned-watch'],
            cpu_affinity=lambda:[42,43,44,45],uids=lambda:types.SimpleNamespace(real=41833))
        claim=early.claim_path(self.request);claim.parent.mkdir()
        request_path=self.fixture.write(self.controller/'request.json',self.request)
        with patch.object(early,'verify_deployment'),patch('psutil.Process',return_value=actor):
            controller=early.Controller(request_path)
            self.assertFalse(controller.state['full_GNN_training'])
            with self.assertRaisesRegex(FileExistsError,'recovery'):early.Controller(request_path)
            second=copy.deepcopy(self.request)
            second_root=self.base/'second_controller';second_root.mkdir()
            second['controller_root']=str(second_root);second['pin_path']=str(second_root/'own_early_stop_best_pin.json')
            for row in second['stages']:row['command'][-1]=str(second_root/'request.json')
            second_path=self.fixture.write(second_root/'request.json',second)
            with self.assertRaises(FileExistsError):early.Controller(second_path)
        self.assertEqual(chain.read(self.native_path),self.native)

    def test_cli_keeps_production_contract_sealed_and_debug_clone_separate(self):
        self.assertEqual(cli.parse(['watch','--request','/new/request.json']).action,'watch')
        self.assertEqual(cli.parse(['clone-debug-worker','--native','/new/native.json','--physical-batch','4','--output','/new/debug']).physical_batch,4)
        with patch('sys.stderr',new=io.StringIO()):
            with self.assertRaises(SystemExit):cli.parse(['train','--request','/new/request.json','--physical-batch','2'])


if __name__=='__main__':
    unittest.main()
