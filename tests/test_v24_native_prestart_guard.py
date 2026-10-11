"""CPU DEBUG adoption metadata/OS fixtures; no medical/CUDA execution."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from hiercp_v1x import v24_native_prestart_guard as guard
from tests import test_v24_scored_native_chain as native_fixtures


class NativePrestartGuardCpuDebugTests(unittest.TestCase):
    def setUp(self):
        self.fixture=native_fixtures.ScoredNativeCPUUnit(methodName='test_three_original_native_contracts_and_exact_external_score_paths')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.base=self.fixture.base;self.manifest={name:'a'*64 for name in
            set(guard.FILES)|set(guard.chain.FILES)|set(guard.pipeline.FILES)|set(guard.early.FILES)}
        self.code=self.base/'fixed_code_DEBUG';self.code.mkdir()
        self.processes={}
        self.ps=patch('psutil.Process',side_effect=self.process);self.ps.start();self.addCleanup(self.ps.stop)

    def process(self,pid=None):
        import psutil
        if pid is None:return self.own_actor
        row=self.processes.get(pid)
        if row is None:raise psutil.NoSuchProcess(pid)
        return SimpleNamespace(pid=pid,create_time=lambda:row['create_time'],cmdline=lambda:row['command'],
            uids=lambda:SimpleNamespace(real=row['UID']),status=lambda:row.get('status','sleeping'))

    def setup_arm(self,gpu):
        self.gpu=gpu;resource=self.fixture.request(gpu);self.resource=resource
        native_path=self.fixture.write(Path(resource['chain_root'])/'request.json',resource)
        claim=Path(resource['native_claim_path']);claim.parent.mkdir(exist_ok=True)
        self.fixture.write(claim,{'CPU_DEBUG_original_native_claim':True})
        if gpu==5:
            controller=self.base/'old_early_controller_DEBUG';controller.mkdir()
            scoring=self.base/'live_early_scoring_DEBUG';scoring.mkdir()
            auth=self.fixture.write(self.base/'authorization_DEBUG.json',{'CPU_DEBUG':True})
            release=[dict(pid=801,create_time=8001.,command=['DEBUG original GNN'],UID=41833)]
            old=guard.early.build_request(native_path,controller_root=controller,scoring_root=scoring,
                code=self.fixture.code,commit='b'*40,code_files=self.manifest,target_epoch=27,
                authorization_path=auth,release_actors=release,protected_files=[])
            pin={'CPU_DEBUG_metadata':True,'physical_GPU':5,'completed_epochs':27,'full_training':False}
            self.fixture.write(old['pin_path'],pin)
            self.fixture.write(controller/'stage_pin_early_stopped.json',{'CPU_DEBUG_original_pin_receipt':True})
            early_claim=Path(old['claim_path']);early_claim.parent.mkdir(exist_ok=True)
            self.fixture.write(early_claim,{'CPU_DEBUG_original_early_claim':True})
            old_path=self.fixture.write(controller/'request.json',old)
            score_command=old['stages'][1]['command']
        else:
            old=resource;old_path=native_path;controller=old_path.parent
            score_command=[resource['python'],'-B','-u',resource['scoring_worker'],'--request',str(Path(resource['scoring_root'])/'request.json')]
            self.fixture.write(Path(resource['scoring_root'])/'registration.json',dict(pid=201,create_time=2001.,command=score_command,UID=41833))
        self.old=old;self.old_path=old_path
        command=[resource['python'],'-B','-u',str(Path(old['code'])/('tools/watch_v24_scored_bank_to_nnunet.py' if gpu==4
            else 'tools/run_v24_early_stop_downstream.py')),*([] if gpu==4 else ['watch']),'--request',str(old_path)]
        self.parent=dict(pid=101,create_time=1001.,command=command,UID=41833)
        self.scorer=dict(pid=201,create_time=2001.,command=score_command,UID=41833)
        self.processes.update({101:self.parent,201:self.scorer})
        registration=dict(self.parent,request_sha256=guard.chain.sha(old_path))
        self.fixture.write(controller/'registration.json',registration)
        self.fixture.write(controller/'lease.json',registration)
        state=dict(GPU_arm=gpu,request_sha256=guard.chain.sha(old_path),supervisor_pid=101,supervisor_create_time=1001.,
            UID=41833,full_native_training_completed=False,full_native_evaluation_completed=False,
            child_pid=201 if gpu==5 else None,child_create_time=2001. if gpu==5 else None,
            stage='score' if gpu==5 else None,status='RUNNING' if gpu==5 else 'WAITING_FOR_FULL642_SCORING',
            stages_completed=['pin_early_stopped'] if gpu==5 else [],command=score_command,resource_violation=None)
        self.fixture.write(controller/'status.json',state)
        self.inspection=dict(GPU=gpu,request=old,registration=registration,native_exists=False)
        self.request=guard.build_request(old_path,self.base/'recovery_DEBUG',self.code,'c'*40,self.manifest,self.inspection)
        return self.request

    def controller(self,gpu):
        self.setup_arm(gpu)
        with patch.object(guard.early,'validate_current_pin',side_effect=copy.deepcopy):
            path=guard.seal_request(self.request)
        command=[self.resource['python'],'-B','-u',str(self.code/guard.FILES[1]),'watch','--request',str(path)]
        self.own_actor=SimpleNamespace(pid=901,create_time=lambda:9001.,cmdline=lambda:command,
            uids=lambda:SimpleNamespace(real=41833),cpu_affinity=lambda:self.resource['CPU_affinity'])
        with (patch.object(guard,'verify_code'),patch.object(guard,'verify_previous_code'),
                patch.object(guard.chain,'verify_code'),patch.object(guard.early,'verify_pin_binding')):
            return guard.Controller(path)

    def test_both_declarations_retain_scale_and_exact_fixed_stage_tails(self):
        for gpu in (4,5):
            with self.subTest(gpu=gpu):
                # Separate fixture per arm keeps each fresh namespace explicit.
                if gpu==5:
                    self.fixture.doCleanups();self.setUp()
                request=self.setup_arm(gpu)
                self.assertEqual(request['resource_contract'],self.resource)
                self.assertEqual(request['predecessor_actor'],self.parent)
                self.assertEqual(request['scoring_actor'],self.scorer)
                self.assertFalse(request['rescore']);self.assertFalse(request['resume'])
                self.assertEqual([row['action'] for row in request['stages']],list(guard.chain.ACTIONS))
                for row in request['stages']:
                    self.assertTrue(Path(row['command'][3]).is_relative_to(self.code))
                    self.assertNotIn('--resume',row['command'])
                if gpu==4:
                    for first,second in zip(self.old['stages'],request['stages']):
                        self.assertEqual(first['command'][4:],second['command'][4:])
                else:
                    self.assertEqual(request['early_request_path'],str(Path(request['recovery_root'])/'early/request.json'))
                    self.assertEqual(request['early_declaration']['target_epoch'],27)
                    self.assertFalse(request['early_declaration']['full_GNN_training'])
                    self.assertEqual(request['early_declaration']['scoring_root'],self.old['scoring_root'])

    def test_seal_copies_only_early_pin_metadata_and_never_original_artifacts(self):
        self.setup_arm(5)
        before={item['path']:Path(item['path']).read_bytes() for item in self.request['protected_files']}
        with patch.object(guard.early,'validate_current_pin',side_effect=copy.deepcopy):
            path=guard.seal_request(self.request)
        early=guard.chain.read(self.request['early_request_path'])
        self.assertEqual(early,self.request['early_declaration'])
        self.assertEqual(Path(early['pin_path']).read_bytes(),Path(self.old['pin_path']).read_bytes())
        self.assertEqual(guard.chain.read(path),self.request)
        self.assertEqual(before,{name:Path(name).read_bytes() for name in before})
        with self.assertRaises(FileExistsError):guard.seal_request(self.request)

    def test_initial_wait_has_original_source_exact_CP_and_no_native_launch(self):
        controller=self.controller(5)
        controller.update()
        with patch.object(guard.subprocess,'Popen') as launch:
            self.assertFalse(controller.adopt_scoring())
        launch.assert_not_called()
        self.assertEqual(controller.state['status'],'WAITING_FOR_EXISTING_SCORING')
        self.assertEqual(controller.state['source_output'],self.resource['source_output'])
        self.assertEqual(controller.state['child_pid'],201)
        self.assertFalse(controller.state['full_native_training_completed'])
        self.assertFalse((Path(self.resource['chain_root'])/'native').exists())

    def test_parent_alive_alone_blocks_adoption_and_PID_reuse_is_not_release(self):
        controller=self.controller(4);self.processes.pop(201)
        self.assertFalse(controller.adopt_scoring())
        self.processes[101]=dict(self.parent,create_time=777.)
        with self.assertRaisesRegex(ValueError,'PID/birth'):controller.adopt_scoring()

    def test_gpu4_full_score_proof_is_required_before_original_arm_append(self):
        controller=self.controller(4);self.processes.clear()
        complete=dict(format=guard.chain.SCORING_FORMAT,complete_sources=642,complete_positions=82176,rescore=False)
        with (patch.object(guard.chain,'preserve_predecessors'),patch.object(guard.chain,'source_complete',return_value={'completed_epochs':40}),
                patch.object(guard.chain,'scoring_complete',return_value=complete)):
            self.assertTrue(controller.adopt_scoring())
        self.assertEqual(guard.chain.read(controller.arm/'admitted_scoring.json'),complete)
        self.assertEqual(controller.state['stages_completed'],['adopted_existing_scoring'])
        self.assertFalse(controller.state['full_native_training_completed'])

    def prepare_early_complete(self,controller,*,done=642):
        self.processes.clear();bank=Path(self.old['scoring_root'])/'bank';bank.mkdir()
        overlay=self.fixture.write(bank/'score_overlay.json',{'CPU_DEBUG_overlay_metadata':True})
        receipt=dict(format=guard.early.FORMAT,status='COMPLETE',action='score',request_sha256=self.request['original_request']['sha256'],
            full_GNN_training=False,completed_GNN_epochs=27,actual_CUDA=True,completed_sources=done,
            completed_CP_positions=done*128,output=str(overlay),output_sha256=guard.chain.sha(overlay))
        self.fixture.write(self.old_path.parent/'stage_score.json',receipt)
        admitted=dict(format=guard.early.FORMAT,full_GNN_training=False,completed_GNN_epochs=27,
            pin=guard.witness(self.old['pin_path']),overlay=guard.witness(overlay))
        self.fixture.write(controller.arm/'admitted_scoring.json',admitted)
        return overlay

    def test_gpu5_whole_genuine_CUDA_score_adoption_keeps_truthful_early_contract(self):
        controller=self.controller(5);self.prepare_early_complete(controller)
        pin={'CPU_DEBUG_own_early_pin':True}
        private={'EARLY_PIPELINE':SimpleNamespace(validate_profiled_overlay=Mock(return_value={'overlay':{'pin':pin}}))}
        with patch.object(guard.early,'verify_pin_binding',return_value=pin),patch.object(guard.early,'private_modules',return_value=private):
            self.assertTrue(controller.adopt_scoring())
        self.assertEqual(controller.scoring['completed_sources'],642)
        self.assertEqual(controller.scoring['completed_CP_positions'],82176)
        self.assertFalse(controller.scoring['full_GNN_training'])
        self.assertEqual(controller.scoring['completed_GNN_epochs'],27)
        self.assertEqual(controller.state['adopted_GNN_epochs'],27)

    def test_gpu5_partial_score_cannot_be_adopted_or_launch_native(self):
        controller=self.controller(5);self.prepare_early_complete(controller,done=641)
        with patch.object(guard.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'CUDA642/82176'):controller.adopt_scoring()
        launch.assert_not_called()

    def test_mutated_original_witness_and_duplicate_global_lease_are_refused(self):
        controller=self.controller(4)
        original=Path(self.request['original_request']['path']);original.write_text('{}',encoding='utf8')
        with self.assertRaises(ValueError):controller.adopt_scoring()
        with patch.object(guard,'verify_code'),patch.object(guard,'verify_previous_code'),patch.object(guard.chain,'verify_code'):
            with self.assertRaises(ValueError):guard.Controller(controller.path)

    def test_atomic_publish_cleans_temp_on_serialization_and_link_failure(self):
        root=self.base/'atomic_DEBUG';root.mkdir();target=root/'result.json'
        with self.assertRaises(ValueError):guard.publish(target,{'bad':float('nan')})
        self.assertEqual(list(root.iterdir()),[])
        with patch.object(guard.os,'link',side_effect=OSError('DEBUG link failure')):
            with self.assertRaisesRegex(OSError,'link failure'):guard.publish(target,{'valid':True})
        self.assertEqual(list(root.iterdir()),[])
        guard.publish(target,{'original':True})
        with self.assertRaises(FileExistsError):guard.publish(target,{'original':False})
        self.assertEqual(guard.chain.read(target),{'original':True})
        self.assertEqual([path.name for path in root.iterdir()],['result.json'])

    def test_atomic_status_replace_failure_preserves_previous_bytes_and_cleans_temp(self):
        target=self.base/'own_status_DEBUG.json';guard.publish(target,{'status':'BEFORE'})
        before=target.read_bytes()
        with patch.object(guard.os,'replace',side_effect=OSError('DEBUG replace failure')):
            with self.assertRaisesRegex(OSError,'replace failure'):guard.publish(target,{'status':'AFTER'},replace=True)
        self.assertEqual(target.read_bytes(),before)
        self.assertFalse(list(self.base.glob('.own_status_DEBUG.json.*.tmp')))

    def test_duplicate_global_adoption_lease_is_exclusive_and_never_replaces_original(self):
        controller=self.controller(4);lease=controller.arm/'native_prestart_guard.lease.json'
        before=lease.read_bytes()
        with patch.object(guard,'verify_code'),patch.object(guard,'verify_previous_code'),patch.object(guard.chain,'verify_code'):
            with self.assertRaises(FileExistsError):guard.Controller(controller.path)
        self.assertEqual(lease.read_bytes(),before)

    def test_zombie_scorer_is_released_but_still_requires_whole_score_proof(self):
        import psutil
        controller=self.controller(4);self.processes.pop(101);self.processes[201]['status']=psutil.STATUS_ZOMBIE
        self.assertFalse(guard.actor_alive(self.scorer))
        with (patch.object(guard.chain,'preserve_predecessors'),patch.object(guard.chain,'source_complete',return_value={'completed_epochs':40}),
                patch.object(guard.chain,'scoring_complete',return_value=None)):
            with self.assertRaisesRegex(RuntimeError,'whole642'):controller.adopt_scoring()

    def test_stage_execution_uses_fixed_code_exact_environment_and_original_proof_paths(self):
        controller=self.controller(4);self.processes.clear()
        controller.storage={'sha256':'d'*64};controller.ready=Mock()
        controller.state['stages_completed']=['adopted_existing_scoring']
        stage=self.request['stages'][0]
        child=SimpleNamespace(pid=301,poll=lambda:0,wait=lambda:0)
        self.processes[301]=dict(pid=301,create_time=3001.,command=stage['command'],UID=41833)
        proof={'CPU_DEBUG_original_native_proof':True}
        with (patch.object(guard.subprocess,'Popen',return_value=child) as launched,
                patch.object(guard.chain,'native_stage_proof',return_value=proof) as admitted):
            self.assertEqual(controller.execute(stage),proof)
        self.assertEqual(launched.call_args.args[0],stage['command'])
        env=launched.call_args.kwargs['env']
        self.assertEqual(env['V24_CONTEXT_DEPTH1_CHAIN_REQUEST'],str(self.old_path))
        self.assertEqual(env['V24_CONTEXT_DEPTH1_CHAIN_REQUEST_SHA256'],self.request['original_request']['sha256'])
        self.assertEqual(env['V24_READONLY_STORAGE_ADMISSION_SHA256'],'d'*64)
        self.assertEqual(env['V24_ARM_RSS_GIB'],'48')
        admitted.assert_called_once_with(self.resource,'prepare-native')

    def test_gpu5_stage_uses_separate_fixed_early_request_and_truthful_new_receipt(self):
        controller=self.controller(5);self.processes.clear()
        controller.storage={'sha256':'d'*64};controller.ready=Mock()
        controller.state['stages_completed']=['adopted_existing_scoring']
        stage=self.request['stages'][0];early_path=Path(self.request['early_request_path'])
        self.fixture.write(early_path.parent/'stage_prepare_native.json',dict(status='COMPLETE',action='prepare-native',
            request_sha256=guard.chain.sha(early_path),full_GNN_training=False))
        child=SimpleNamespace(pid=301,poll=lambda:0,wait=lambda:0)
        self.processes[301]=dict(pid=301,create_time=3001.,command=stage['command'],UID=41833)
        proof={'CPU_DEBUG_truthful_early_native_proof':True}
        with (patch.object(guard.subprocess,'Popen',return_value=child) as launched,
                patch.object(guard.early,'native_proof',return_value=proof) as admitted):
            self.assertEqual(controller.execute(stage),proof)
        self.assertEqual(launched.call_args.args[0],stage['command'])
        env=launched.call_args.kwargs['env']
        self.assertEqual(env[guard.early.ENV_REQUEST],str(early_path))
        self.assertEqual(env[guard.early.ENV_SHA],guard.chain.sha(early_path))
        admitted.assert_called_once_with(self.request['early_declaration'],'prepare-native')

    def test_guard_never_sends_process_signals(self):
        tree=ast.parse(Path(guard.__file__).read_text(encoding='utf8'))
        calls=[ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)]
        self.assertFalse(any(name.endswith(('.kill','.terminate','.send_signal')) for name in calls))


if __name__=='__main__':unittest.main()
