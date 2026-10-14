"""CPU-only DEBUG: real exit75, exact outside actors and natural retry cleanup."""
import copy
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import basic_cp_ablation_resources as common
from hiercp_v1x.basic_cp_gpu_admission import GPUAdmissionUnavailable, GPU_GUARD_FILE, GPU_WAIT_FORMAT
from tools import run_basic_cp_expanded as runner
from tests import test_basic_cp_ablation_resources as fixture
VALIDATE_REQUEST=common.validate_request


def outside():
    return dict(pid=901,create_time=123.0,UID=66006,command=['python','DEBUG_outside'],
        status='sleeping',GPU_UUID='GPU-exact',physical_GPU=0,fresh_tree_PIDs=[100,501],
        admitted=False,processes_signaled=False,observed_at=1000)


class TypedGPUResourceTests(unittest.TestCase):
    start=fixture.BasicLeaseTests.start
    process=fixture.BasicLeaseTests.process
    query=fixture.BasicLeaseTests.query

    def setUp(self):
        fixture.BasicLeaseTests.setUp(self)
        protocol=self.root/GPU_GUARD_FILE;protocol.write_text('DEBUG exact queue protocol')
        self.value['code_sha256'][GPU_GUARD_FILE]=common.sha(protocol)
        self.value['GPU_admission_retry_version']=1
        self.lease=common.BasicLease(self.value)

    def test_actual_foreign_actor_queues_even_when_initial_free_vram_dropped(self):
        self.actors[777]=fixture.Actor(777,['python','DEBUG_foreign'],birth=703,uid=66006)
        self.apps='777, GPU-exact';self.free_gpu=12000
        with self.assertRaises(GPUAdmissionUnavailable)as caught:self.lease.verify(idle=True)
        observed=caught.exception.observation
        self.assertEqual((observed['pid'],observed['create_time'],observed['UID']),(777,703,66006))
        self.assertEqual(observed['command'],['python','DEBUG_foreign'])
        self.assertFalse(observed['admitted']);self.assertFalse(observed['processes_signaled'])
        self.assertEqual(observed['occupied_GPU_compute_actors'][0]['pid'],777)

    def test_resource_identity_and_own_idle_faults_are_not_retryable_GPU_errors(self):
        self.apps='200, GPU-exact'
        with self.assertRaises(RuntimeError)as caught:self.lease.verify(idle=True)
        self.assertNotIsInstance(caught.exception,GPUAdmissionUnavailable)
        self.apps='';self.worker.uid=999
        with self.assertRaises(RuntimeError)as caught:self.lease.verify(force=True)
        self.assertNotIsInstance(caught.exception,GPUAdmissionUnavailable)
        self.worker.uid=41833;self.stage.rss=97*2**30
        with self.assertRaises(MemoryError):self.lease.verify(force=True)

    def test_live_outside_identity_change_cannot_be_reclassified_as_wait(self):
        actor=fixture.Actor(777,['python','DEBUG_foreign'],birth=703,uid=66006);self.actors[777]=actor
        with patch.object(actor,'create_time',side_effect=[703,704]),self.assertRaisesRegex(RuntimeError,'PID/birth/UID'):
            self.lease._live_outside_cuda_actor(777,'GPU-exact',{})

    def test_explicit_retry_marker_requires_integer_version_and_protocol_SHA(self):
        for marker in(True,1.0,2):
            value=fixture.request();value['GPU_admission_retry_version']=marker
            with patch.object(common,'Path',PurePosixPath),self.assertRaisesRegex(ValueError,'typed protocol SHA'):
                VALIDATE_REQUEST(value)
        # setUp mocks validation for actual OS fixtures; exercise the original function.
        value=fixture.request();value['GPU_admission_retry_version']=1
        with patch.object(common,'Path',PurePosixPath),self.assertRaises(ValueError):VALIDATE_REQUEST(value)
        value['code_sha256'][GPU_GUARD_FILE]='d'*64
        with patch.object(common,'Path',PurePosixPath):self.assertIs(VALIDATE_REQUEST(value),value)


class NativeGPUQueueTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.native=dict(root=str(self.root),bank='/DEBUG/bank/index.json',bank_sha256='a'*64,
            GPU_admission_retry_version=1,controller_identity=dict(pid=100,create_time=10,UID=41833,command=['python','controller']),
            gpu=0,gpu_uuid='GPU-exact',arm='multi5',copies_per_CP_event=5,candidates_per_source=128,
            epochs=250,physical_batch=2,cpu_affinity=[0,1,2,3,4,5,6,7])

    def test_race_before_child_launch_requeues_same_scientific_stage(self):
        events=[];before=copy.deepcopy(self.native)
        def actor(*args):
            events.append('actor')
            if events.count('actor')==1:raise GPUAdmissionUnavailable(outside())
        with(patch.object(runner,'_wait_for_GPU',side_effect=lambda v:events.append('wait')),
             patch.object(runner,'actor',side_effect=actor)as launch,patch.object(runner,'status'),
             patch.object(runner,'environment',return_value={})):
            runner.child(self.native,'train')
        self.assertEqual(events,['wait','actor','wait','actor']);self.assertEqual(self.native,before)
        self.assertEqual(launch.call_args_list[0].args[1],launch.call_args_list[1].args[1])

    def test_other_child_errors_remain_fatal_without_duplicate_actor(self):
        with(patch.object(runner,'_wait_for_GPU'),patch.object(runner,'status'),patch.object(runner,'environment',return_value={}),
             patch.object(runner,'actor',side_effect=RuntimeError('actual native error'))as launch,
             self.assertRaisesRegex(RuntimeError,'actual native error')):
            runner.child(self.native,'train')
        launch.assert_called_once()

    def run_actor_fixture(self,returncode,*,force_error=None):
        """Actual actor loop, with CPU-only deterministic process observations."""
        events=[];calls=[0]
        child=SimpleNamespace(pid=501,returncode=None)
        def poll():
            calls[0]+=1
            if calls[0]>=3:child.returncode=returncode
            return child.returncode
        child.poll=poll
        process=SimpleNamespace(pid=501,create_time=lambda:11)
        lease=Mock()
        if force_error is not None:lease.verify.side_effect=force_error
        cleanup=dict(actor=dict(pid=501),returncode=returncode,naturally_released={},processes_signaled=False)
        failure=GPUAdmissionUnavailable(outside())
        def cleaned(*args):events.append('natural_cleanup');return cleanup
        with(patch.object(runner,'resource_guard',side_effect=[None,failure]),
             patch.object(runner,'_lease_for',return_value=lease),patch.object(runner,'_capture_native_tree'),
             patch.object(runner.subprocess,'Popen',return_value=child)as launch,
             patch.object(common.psutil,'Process',return_value=process),patch.object(runner.os,'getuid',return_value=41833,create=True),
             patch.object(runner.time,'sleep'),patch.object(runner,'status')as state,
             patch.object(runner,'_wait_native_release',side_effect=cleaned),
             patch.object(runner,'_validate_native_GPU_wait',return_value=failure)):
            try:runner.actor(self.native,['python','DEBUG_native'],'train',{})
            except RuntimeError as error:return events,state,lease,launch,error
        return events,state,lease,launch,None

    def test_transient_foreign_occupancy_readmits_the_existing_child_without_duplicate(self):
        events,state,lease,launch,error=self.run_actor_fixture(0)
        self.assertIsNone(error);launch.assert_called_once()
        lease.verify.assert_called_once_with(force=True,idle=False)
        self.assertEqual([call.args[1:3]for call in state.call_args_list],
                         [('waiting_GPU','WAITING'),('train','RUNNING')])
        self.assertEqual(events,['natural_cleanup'])

    def test_parent_typed_race_retries_only_after_child75_exact_natural_cleanup(self):
        events,state,lease,launch,error=self.run_actor_fixture(75,force_error=GPUAdmissionUnavailable(outside()))
        self.assertIsInstance(error,GPUAdmissionUnavailable)
        self.assertEqual(events,['natural_cleanup']);launch.assert_called_once()

    def test_parent_typed_race_does_not_hide_a_later_ordinary_child_failure(self):
        events,state,lease,launch,error=self.run_actor_fixture(1)
        self.assertIsInstance(error,RuntimeError);self.assertNotIsInstance(error,GPUAdmissionUnavailable)
        self.assertIn('train failed (1)',str(error));self.assertEqual(events,['natural_cleanup'])

    def test_prelaunch_typed_race_creates_no_child_at_all(self):
        with(patch.object(runner,'resource_guard',side_effect=GPUAdmissionUnavailable(outside())),
             patch.object(runner.subprocess,'Popen')as launch,self.assertRaises(GPUAdmissionUnavailable)):
            runner.actor(self.native,['python','DEBUG_native'],'train',{})
        launch.assert_not_called()

    def test_debug_retry_preserves_partial_updates_and_uses_successful_fresh_attempt(self):
        output=self.root/'B2_DEBUG';output.mkdir();partial=output/'update_0.json';partial.write_text('original DEBUG partial')
        paths=[]
        def actor(native,command,label,env):
            path=Path(command[command.index('--output')+1]);paths.append(path)
            if len(paths)==1:raise GPUAdmissionUnavailable(outside())
            self.assertNotEqual(path,output)
            for i in range(4):(path/('update_'+str(i)+'.json')).write_text(json.dumps({'debug':True,'iteration':i}))
            (path/'trial.json').write_text(json.dumps(dict(debug=True,accepted=True,measurements=list(range(4)))))
        with(patch.object(runner,'_wait_for_GPU'),patch.object(runner,'status'),patch.object(runner,'environment',return_value={}),
             patch.object(runner,'actor',side_effect=actor)):
            runner.child(self.native,'debug',batch=2,output=output)
        self.assertEqual(partial.read_text(),'original DEBUG partial')
        trial=json.loads((output/'trial.json').read_text())
        self.assertEqual(trial['measurements'],[0,1,2,3])
        self.assertEqual(trial['GPU_admission_retry_DEBUG']['actual_completed_attempt_output'],str(paths[1]))
        self.assertEqual(trial['GPU_admission_retry_DEBUG']['actual_trial_sha256'],runner.sha(paths[1]/'trial.json'))
        self.assertEqual(len(list(paths[1].glob('update_*.json'))),4)

    def test_exact_natural_release_wait_captures_new_workers_and_never_signals(self):
        actors={501:fixture.Actor(501,['python','native'],birth=11),601:fixture.Actor(601,['python','worker'],birth=12)}
        tracked={pid:fixture.identity(actor)for pid,actor in actors.items()}
        child=SimpleNamespace(returncode=None,poll=lambda:child.returncode)
        lease=Mock();lease.observe_waiting.return_value=dict(GPU_available=False)
        counter=[0]
        def process(pid):
            if pid not in actors or actors[pid].state=='gone':raise common.psutil.NoSuchProcess(pid)
            return actors[pid]
        def capture(native,known):
            if counter[0]==1:
                actors[602]=fixture.Actor(602,['python','late_worker'],birth=13)
                known[602]=fixture.identity(actors[602])
        def sleep(seconds):
            counter[0]+=1
            if counter[0]==1:child.returncode=75;actors[501].state='gone'
            if counter[0]==2:actors[601].state='gone';actors[602].state='gone'
        with(patch.object(runner,'_lease_for',return_value=lease),patch.object(runner,'_capture_native_tree',side_effect=capture),
             patch.object(common.psutil,'Process',side_effect=process),patch.object(runner.time,'sleep',side_effect=sleep),
             patch.object(runner,'status')):
            receipt=runner._wait_native_release(self.native,child,dict(pid=501),tracked,GPUAdmissionUnavailable(outside()))
        self.assertEqual(counter[0],2);self.assertEqual(set(receipt['naturally_released']),{501,601,602})
        self.assertFalse(receipt['processes_signaled']);self.assertEqual(lease.observe_waiting.call_count,3)

    def test_cleanup_reused_PID_or_UID_cannot_be_treated_as_release(self):
        actor=fixture.Actor(501,['python','native'],birth=99)
        child=SimpleNamespace(returncode=75,poll=lambda:75)
        expected=dict(pid=501,create_time=11,UID=41833,command=['python','native'])
        with(patch.object(runner,'_lease_for',return_value=Mock()),patch.object(runner,'_capture_native_tree'),
             patch.object(common.psutil,'Process',return_value=actor),self.assertRaisesRegex(RuntimeError,'reused or changed')):
            runner._wait_native_release(self.native,child,{}, {501:expected},GPUAdmissionUnavailable(outside()))

    def test_capture_reuses_coherent_admitted_identity_without_racing_argv_exit(self):
        controller=fixture.Actor(100,['python','controller'],birth=10)
        stage=fixture.Actor(501,['python','native'],birth=11)
        lease=Mock();lease._managed={501:fixture.identity(stage)};lease._tree.return_value={100:controller,501:stage}
        with(patch.object(runner,'_lease_for',return_value=lease),patch.object(common,'_exact',return_value=controller),
             patch.object(stage,'cmdline',side_effect=common.psutil.NoSuchProcess(501))as argv):
            tracked={};runner._capture_native_tree(self.native,tracked)
        argv.assert_not_called();self.assertEqual(tracked[501],fixture.identity(stage))

    def test_newly_disappeared_descendant_retains_earlier_exact_cleanup_record(self):
        controller=fixture.Actor(100,['python','controller'],birth=10)
        stage=fixture.Actor(501,['python','native'],birth=11);expected=fixture.identity(stage)
        lease=Mock();lease._managed={};lease._tree.return_value={100:controller,501:stage}
        with(patch.object(runner,'_lease_for',return_value=lease),patch.object(common,'_exact',return_value=controller),
             patch.object(stage,'create_time',side_effect=common.psutil.NoSuchProcess(501))):
            tracked={501:copy.deepcopy(expected)};runner._capture_native_tree(self.native,tracked)
        self.assertEqual(tracked[501],expected)

    def test_exit75_receipt_requires_exact_actor_bank_stage_and_outside_identity(self):
        identity=dict(pid=501,birth=11,UID=41833,command=['python','native'])
        receipt=dict(format=GPU_WAIT_FORMAT,stage='train',label='train',actor_identity=dict(pid=501,create_time=11,UID=41833,command=['python','native']),
            bank_sha256=self.native['bank_sha256'],exit_code=75,native_stage_finally_completed=True,
            processes_signaled=False,observation=GPUAdmissionUnavailable(outside()).observation)
        with patch.object(runner,'read',return_value=receipt):
            self.assertIsInstance(runner._validate_native_GPU_wait(self.native,'train',identity),GPUAdmissionUnavailable)
            for key,value in(('bank_sha256','b'*64),('exit_code',1),('native_stage_finally_completed',False),('stage','preflight')):
                changed=dict(receipt,**{key:value})
                with patch.object(runner,'read',return_value=changed),self.assertRaises(RuntimeError):
                    runner._validate_native_GPU_wait(self.native,'train',identity)

    def test_entire_train_lifecycle_finishes_real_active_loaders_before_typed_publication(self):
        events=[]
        def core(native,active):
            active.append(SimpleNamespace(dataloader_train=SimpleNamespace(_finish=lambda:events.append('train_finish')),
                                          dataloader_val=SimpleNamespace(_finish=lambda:events.append('val_finish'))))
            raise GPUAdmissionUnavailable(outside())
        with patch.object(runner,'_train_native',side_effect=core),self.assertRaises(GPUAdmissionUnavailable):
            runner.train(self.native)
        self.assertEqual(events,['train_finish','val_finish'])

    def test_real_CPU_native_process_returns75_after_finally_and_preserves_unknown_errors(self):
        native_path=self.root/'native.json';value=dict(self.native)
        # DEBUG only: inherit the real CPU mask; the mocked stage never initializes CUDA.
        import psutil
        value['cpu_affinity']=psutil.Process().cpu_affinity();native_path.write_text(json.dumps(value))
        for typed in(True,False):
            script="""
import json,sys,psutil
from unittest.mock import patch
from tools import run_basic_cp_expanded as runner
from hiercp_v1x import basic_cp_ablation_resources as common
from hiercp_v1x.basic_cp_gpu_admission import GPUAdmissionUnavailable
sys.argv=['DEBUG_native_stage','--native',sys.argv[1],'--stage','preflight']
typed=BOOL
def stage(value):
    try:
        if typed:raise GPUAdmissionUnavailable(OBS)
        raise RuntimeError('DEBUG actual unknown native failure')
    finally:
        runner.publish(runner.Path(value['root'])/('finally_'+str(typed)+'.json'),{'cleanup_finished':True})
def identity(p):return {'pid':p.pid,'create_time':p.create_time(),'UID':41833,'command':p.cmdline()}
with patch.object(runner,'preflight',stage),patch.object(common,'_actor',identity):
    result=runner.main()
    if result is not None:raise SystemExit(result)
""".replace('BOOL',repr(typed)).replace('OBS',repr(outside()))
            result=subprocess.run([sys.executable,'-B','-c',script,str(native_path)],cwd=runner.ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,75 if typed else 1,result.stderr)
            self.assertTrue(json.loads((self.root/('finally_'+str(typed)+'.json')).read_text())['cleanup_finished'])
            state=json.loads((self.root/'status.json').read_text())
            self.assertEqual(state['status'],'WAITING'if typed else'FAILED')
            if typed:
                receipt=json.loads((self.root/'preflight_GPU_admission_wait.json').read_text())
                self.assertEqual(receipt['exit_code'],75);self.assertTrue(receipt['native_stage_finally_completed'])
            else:self.assertIn('actual unknown native failure',state['error'])

    def test_shared_recovery_selector_retains_full642_2048_and_exact_snapshot_arguments(self):
        from hiercp_v1x import basic_cp_large_bank
        banks=self.root/'banks';target=banks/'2048';target.mkdir(parents=True)
        native=dict(self.native,candidates_per_source=2048,arm='candidates2048',shared_banks=str(banks),
            shared_candidate_geometry_cache=True,shared_bank_workers=24,shared_bank_RSS_limit_GiB=384,
            candidate_binding='/DEBUG/exact_binding.json',candidate_binding_sha256='b'*64,
            previous_partial_banks='/DEBUG/old_partial',previous_producer_request='/DEBUG/old_request.json',
            previous_producer_request_sha256='c'*64,interrupted_candidate_recovery_revision=1,
            interrupted_candidate_recovery_snapshot='/DEBUG/exact_snapshot.json',
            interrupted_candidate_recovery_snapshot_sha256='d'*64,
            code_sha256={'hiercp_v1x/basic_cp_large_bank_recovery.py':'e'*64})
        commands=[]
        def actor(value,command,*args):
            commands.append(command)
            (target/'index.json').write_text(json.dumps(dict(source_entries=642,candidate_count=2048)))
            (target/'complete.json').write_text(json.dumps(dict(complete=True,index_sha256=runner.sha(target/'index.json'))))
            for filename in('preservation_manifest.json','preparation_started.json'):(target/filename).write_text('{}')
        with(patch.object(runner,'_lease_for',return_value=Mock()),patch.object(runner,'status'),patch.object(runner,'actor',side_effect=actor),
             patch.object(basic_cp_large_bank,'verify_prepared_bank')):
            result=runner.prepare_bank(native)
        command=commands[0]
        self.assertEqual(command[command.index('-m')+1],'hiercp_v1x.basic_cp_large_bank_recovery')
        self.assertEqual(command[command.index('--recovery-snapshot')+1],native['interrupted_candidate_recovery_snapshot'])
        self.assertEqual(command[command.index('--recovery-snapshot-sha256')+1],'d'*64)
        self.assertEqual(command[command.index('--workers')+1],'24')
        self.assertEqual(result['source_bank'],self.native['bank']);self.assertEqual(result['source_bank_sha256'],'a'*64)

    def test_recovery_selector_rejects_noninteger_marker_or_missing_new_SHA(self):
        banks=self.root/'unpublished_banks';banks.mkdir()
        for marker in(True,1.0,2,1):
            native=dict(self.native,candidates_per_source=2048,shared_banks=str(banks),shared_candidate_geometry_cache=True,
                        interrupted_candidate_recovery_revision=marker,code_sha256={})
            with(patch.object(runner,'_lease_for',return_value=Mock()),patch.object(runner,'actor')as launch,
                 self.assertRaisesRegex(ValueError,'recovery revision1')):
                runner.prepare_bank(native)
            launch.assert_not_called()


if __name__=='__main__':unittest.main()
