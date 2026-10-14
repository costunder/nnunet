"""CPU-only DEBUG: exact same-actor RAM pauses with unchanged scientific caps."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import basic_cp_RAM_admission_pause as pause
from hiercp_v1x import basic_cp_group_resources as original
from hiercp_v1x import basic_cp_seven_resources as resource
from hiercp_v1x.basic_cp_gpu_admission import GPUAdmissionUnavailable
from tools import run_basic_cp_expanded as runner
from tests import test_basic_cp_resources_v5 as native_fixture
from tests import test_basic_cp_shared_bank_resources_v6 as shared_fixture


def shortage():
    return original.RAMAdmissionUnavailable(host_total_bytes=1008*2**30,available_bytes=780*2**30,
        group_resident_bytes=7*2**30,required_bytes=800*2**30)


def admit_pause(self, cls):
    source=self.root/pause.GUARD_FILE;source.write_text('DEBUG exact immutable pause guard')
    for gpu,value in self.requests.items():
        value['RAM_admission_pause_revision']=1
        value['code_sha256'][pause.GUARD_FILE]=resource.sha(source)
        path=Path(value['root'])/'request.json';path.write_text(json.dumps(value))
        self.document['registration'][str(gpu)]['request_sha256']=resource.sha(path)
    self.registry_path.write_text(json.dumps(self.document))
    self.lease=cls(self.value)


class NativeActualPauseTests(unittest.TestCase):
    start=native_fixture.GroupV5Tests.start
    process=native_fixture.GroupV5Tests.process
    query=native_fixture.GroupV5Tests.query
    def setUp(self):
        native_fixture.GroupV5Tests.setUp(self);admit_pause(self,pause.BasicLease)

    def low(self):self.host.available=780*2**30

    def test_actual_shortage_pauses_same_native_actor_then_force_readmits800_without_scale_change(self):
        self.low();identity=copy.deepcopy(self.lease.self_identity)
        def recovered(seconds):self.assertEqual(seconds,5);self.host.available=900*2**30
        with patch.object(pause.time,'sleep',side_effect=recovered)as sleeping:
            result=self.lease.verify(force=True)
        sleeping.assert_called_once();self.assertEqual(self.lease.self_identity,identity)
        self.assertEqual(result['aggregate_native_and_cache_capacity_bytes'],800*2**30)
        self.assertEqual(result['own_process_tree_RSS_limit_bytes'],96*2**30)
        self.assertEqual(result['candidate_subpool_RSS_limit_bytes'],384*2**30)
        self.assertEqual(result['group_process_tree_RSS_limit_bytes'],768*2**30)
        evidence=json.loads(self.lease._pause_file().read_bytes())
        self.assertEqual(evidence['status'],'READMITTED');self.assertEqual(evidence['actor_identity'],identity)
        self.assertEqual(evidence['RAM_admission']['observed_available_plus_group_RSS_bytes'],
                         evidence['RAM_admission']['host_available_RAM_bytes']+evidence['RAM_admission']['group_process_tree_RSS_bytes'])
        self.assertFalse(evidence['resource_caps_changed']);self.assertFalse(evidence['processes_signaled'])

    def test_actual_native96_hardcap_is_fatal_and_never_pauses(self):
        self.low();self.stage.rss=97*2**30
        with patch.object(pause.time,'sleep')as sleeping,self.assertRaises(MemoryError):self.lease.verify(force=True)
        sleeping.assert_not_called()

    def test_actual_candidate384_hardcap_is_fatal_and_never_pauses(self):
        self.low();self.actors[305].rss=385*2**30
        with patch.object(pause.time,'sleep')as sleeping,self.assertRaises(MemoryError):self.lease.verify(force=True)
        sleeping.assert_not_called()

    def test_actual_cgroup799_is_fatal_even_with_temporary_shortage(self):
        self.low();self.admission['memory_limits'][0]['limit_bytes']=799*2**30
        with patch.object(pause.time,'sleep')as sleeping,self.assertRaises(MemoryError):self.lease.verify(force=True)
        sleeping.assert_not_called()

    def test_actual_total_host_less_than800_is_not_a_pause(self):
        self.host.total=799*2**30;self.low()
        with patch.object(pause.time,'sleep')as sleeping,self.assertRaises(MemoryError):self.lease.verify(force=True)
        sleeping.assert_not_called()

    def test_exact_expiry_still_fails_during_pause_instead_of_extending_lifetime(self):
        self.low()
        def expired(seconds):self.wall=4900;self.monotonic=4400
        with patch.object(pause.time,'sleep',side_effect=expired)as sleeping,self.assertRaisesRegex(RuntimeError,'180seconds'):
            self.lease.verify(force=True)
        sleeping.assert_called_once();self.assertEqual(self.lease.lifetime['expires_at_unix'],5000)

    def test_code_or_source_SHA_metadata_change_during_pause_remains_fatal(self):
        self.low()
        def changed(seconds):self.bank.write_text('DEBUG mutated source')
        with patch.object(pause.time,'sleep',side_effect=changed),self.assertRaisesRegex(ValueError,'metadata changed'):
            self.lease.verify(force=True)

    def test_exact_UID_change_during_pause_remains_fatal(self):
        self.low()
        def changed(seconds):self.worker.uid=999
        with patch.object(pause.time,'sleep',side_effect=changed),self.assertRaisesRegex(RuntimeError,'foreign'):
            self.lease.verify(force=True)

    def test_GPU_UUID_is_checked_during_RAM_shortage_before_any_sleep(self):
        self.low();self.gpu_total=49140
        with patch.object(pause.time,'sleep')as sleeping,self.assertRaisesRegex(RuntimeError,'UUID/MIG'):
            self.lease.verify(force=True)
        sleeping.assert_not_called()

    def test_storage_floor_is_checked_during_RAM_shortage(self):
        self.low();self.storage.free=9*2**30
        with patch.object(pause.time,'sleep')as sleeping,self.assertRaises(OSError):self.lease.verify(force=True)
        sleeping.assert_not_called()

    def test_failed_child_status_during_pause_is_not_restored_to_RUNNING(self):
        self.low();status=Path(self.value['root'])/'status.json'
        status.write_text(json.dumps(dict(status='FAILED',error='DEBUG actual child failure')))
        with patch.object(pause.time,'sleep')as sleeping,self.assertRaisesRegex(RuntimeError,'actual child failure'):
            self.lease.verify(force=True)
        sleeping.assert_not_called()

    def test_pause_requires_explicit_integer_marker_and_new_guard_SHA(self):
        for marker in(None,True,1.0,0,2):
            value=copy.deepcopy(self.value)
            if marker is None:value.pop('RAM_admission_pause_revision')
            else:value['RAM_admission_pause_revision']=marker
            with self.assertRaisesRegex(ValueError,'revision1'):pause.BasicLease(value)
        value=copy.deepcopy(self.value);value['code_sha256'].pop(pause.GUARD_FILE)
        with self.assertRaisesRegex(ValueError,'SHA'):pause.BasicLease(value)

    def test_no_implicit_mixed_peer_upgrade(self):
        self.requests[1].pop('RAM_admission_pause_revision')
        path=Path(self.requests[1]['root'])/'request.json';path.write_text(json.dumps(self.requests[1]))
        self.document['registration']['1']['request_sha256']=resource.sha(path);self.registry_path.write_text(json.dumps(self.document))
        with self.assertRaisesRegex(ValueError,'revision1'):pause.BasicLease(self.value)

    def test_actual_CPU_wait_retains_foreign_GPU_queue_during_transient_RAM_shortage(self):
        self.low();self.apps='901, GPU-exact'
        with patch.object(pause.time,'sleep',side_effect=lambda seconds:setattr(self.host,'available',900*2**30))as sleeping:
            result=self.lease.observe_waiting()
        sleeping.assert_called_once();self.assertFalse(result['GPU_available'])
        self.assertEqual(result['occupied_GPU_compute_actors'][0]['pid'],901)
        self.assertFalse(result['occupied_GPU_compute_actors'][0]['processes_signaled'])
        self.assertTrue(result['RAM_pause_readmitted_same_actor'])


class SharedActualPauseTests(unittest.TestCase):
    start=shared_fixture.SharedV6Tests.start
    process=shared_fixture.SharedV6Tests.process
    query=shared_fixture.SharedV6Tests.query
    def setUp(self):
        shared_fixture.SharedV6Tests.setUp(self);admit_pause(self,pause.SharedBankLease)

    def test_shared_CPU_worker_same_actor_quick_boundary_blocks_then_preserves64CPU384(self):
        self.lease.verify(force=True);identity=copy.deepcopy(self.lease.self_identity)
        self.host.available=780*2**30;self.monotonic+=5
        def recovered(seconds):self.host.available=900*2**30
        with patch.object(pause.time,'sleep',side_effect=recovered)as sleeping:
            result=self.lease.quick_verify()
        sleeping.assert_called_once();self.assertEqual(self.lease.self_identity,identity)
        self.assertEqual(result['own_CPU_affinity'],list(pause.shared.SHARED_CPU_AFFINITY))
        self.assertEqual(result['own_process_tree_RSS_limit_bytes'],384*2**30)
        self.assertEqual(result['native_RSS_limit_bytes_preserved'],96*2**30)
        self.assertEqual(result['aggregate_native_and_cache_capacity_bytes'],800*2**30)

    def test_shared384_hard_cap_is_not_caught_as_transient_RAM(self):
        self.worker.rss=385*2**30;self.host.available=780*2**30
        with patch.object(pause.time,'sleep')as sleeping,self.assertRaises(MemoryError):self.lease.verify(force=True)
        sleeping.assert_not_called()

    def test_source_CPU_worker_does_not_overwrite_controller_status(self):
        path=Path(self.value['root'])/'status.json'
        state=dict(stage='candidate_bank',status='RUNNING',completed_epochs=0)
        path.write_text(json.dumps(state));self.host.available=780*2**30
        with patch.object(pause.time,'sleep',side_effect=lambda seconds:setattr(self.host,'available',900*2**30)):
            self.lease.verify(force=True)
        self.assertEqual(json.loads(path.read_bytes()),state)
        self.assertEqual(json.loads(self.lease._pause_file().read_bytes())['status'],'READMITTED')


class ControllerStatusTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.lease=object.__new__(pause.BasicLease)
        identity=dict(pid=123,create_time=100.0,UID=41833,command=['python','DEBUG_controller'])
        self.lease.request=dict(root=str(self.root),gpu=1,arm='multi7',controller_identity=copy.deepcopy(identity))
        self.lease.self_identity=identity;self.lease._RAM_pause_state=None
        self.state=self.root/'status.json'

    def test_controller_pause_and_resume_keep_same_actor_original_phase_and_epoch(self):
        state=dict(stage='train',status='RUNNING',completed_epochs=52)
        self.state.write_text(json.dumps(state));identity=copy.deepcopy(self.lease.self_identity)
        self.lease._publish_pause(shortage(),dict(same_actor_preserved=True))
        paused=json.loads(self.state.read_bytes())
        self.assertEqual((paused['stage'],paused['status'],paused['completed_epochs']),('waiting_RAM','WAITING',52))
        paused['completed_epochs']=53;self.state.write_text(json.dumps(paused))
        self.lease._publish_pause(shortage(),dict(same_actor_preserved=True),resumed=True)
        actual=json.loads(self.state.read_bytes())
        self.assertEqual((actual['stage'],actual['status'],actual['completed_epochs']),('train','RUNNING',53))
        self.assertEqual(self.lease.self_identity,identity)

    def test_GPU_queue_restore_stays_WAITING_without_false_native_start(self):
        self.state.write_text(json.dumps(dict(stage='waiting_GPU',status='WAITING',reason='DEBUG actor occupied')))
        self.lease._publish_pause(shortage(),{},resumed=False);self.lease._publish_pause(shortage(),{},resumed=True)
        state=json.loads(self.state.read_bytes())
        self.assertEqual((state['stage'],state['status']),('waiting_GPU','WAITING'))
        self.assertEqual(state['reason'],'DEBUG actor occupied')

    def test_child_failure_at_readmission_cannot_be_overwritten_with_RUNNING(self):
        self.state.write_text(json.dumps(dict(stage='train',status='RUNNING',completed_epochs=52)))
        self.lease._publish_pause(shortage(),{})
        failure=dict(stage='train',status='FAILED',error='DEBUG actual child failure at readmission')
        self.state.write_text(json.dumps(failure))
        with self.assertRaisesRegex(RuntimeError,'failure at readmission'):
            self.lease._publish_pause(shortage(),{},resumed=True)
        self.assertEqual(json.loads(self.state.read_bytes()),failure)

    def test_no_start_before_same_actor_full_readmission_and_no_catch_of_GPU_errors(self):
        events=[]
        class Base:
            def verify(self,*,force=False,idle=False):
                events.append(('verify',force,idle))
                if len([e for e in events if e[0]=='verify'])==1:raise shortage()
                return {'actual_resource_contract':'unchanged'}
        class Probe(pause._RAMPause,Base):
            def _pause_boundary(self,*,idle):events.append(('boundary',idle));return {}
            def _publish_pause(self,*args,**kwargs):events.append(('publish',kwargs.get('resumed',False)))
            def _pause_file(self):return Path('/DEBUG/evidence')
        lease=object.__new__(Probe);lease._RAM_pause_state=None
        with patch.object(pause.time,'sleep',side_effect=lambda seconds:events.append(('sleep',seconds))):
            lease.verify();events.append(('optimizer_after_guard',))
        self.assertEqual(events[-1],('optimizer_after_guard',))
        self.assertEqual(events[0],('verify',False,False));self.assertIn(('verify',True,False),events)
        self.assertLess(events.index(('sleep',5)),events.index(('optimizer_after_guard',)))
        with (patch.object(Base,'verify',side_effect=GPUAdmissionUnavailable({'pid':901})),
              patch.object(pause.time,'sleep')as sleeping,self.assertRaises(GPUAdmissionUnavailable)):
            lease.verify()
        sleeping.assert_not_called()

    def test_factory_uses_pause_only_for_explicit_upgrade(self):
        self.assertIs(runner._lease_class(dict(RAM_admission_pause_revision=1)),pause.BasicLease)
        self.assertIs(runner._lease_class(dict(group_observation_revision=5)),pause.group.BasicLease)


if __name__=='__main__':unittest.main()
