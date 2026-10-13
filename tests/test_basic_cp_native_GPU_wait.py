"""CPU-only DEBUG: exact cached admission before native stage creation."""
import copy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from tools import run_basic_cp_expanded as runner
from tests import test_basic_cp_resources_v5 as fixture


class NativeStageWaitTests(unittest.TestCase):
    def setUp(self):
        self.cache = patch.dict(runner.LEASES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.native = dict(root='/DEBUG/native_stage', bank='/DEBUG/actual512/index.json',
            bank_sha256='a'*64, source_bank='/DEBUG/original128/index.json',
            source_bank_sha256='b'*64, group_observation_revision=5,
            shared_resource_guard_version=3, candidates_per_source=512,
            copies_per_CP_event=1, epochs=250, physical_batch_size=2)

    def test_native_stages_wait_before_actor_and_restore_existing_display_phase(self):
        events=[]
        lease=Mock()
        def wait(*,on_wait):
            events.append('wait')
            on_wait(dict(GPU_available=False,occupied_GPU_compute_actors=[dict(pid=901)]))
            events.append('released')
        lease.wait_until_available.side_effect=wait
        factory=Mock(return_value=lease)
        def actor(native,command,label,env):
            self.assertEqual(events[-1],'released')
            events.append(label)
        with (patch.object(runner,'_lease_class',return_value=factory),
              patch.object(runner,'status')as status,patch.object(runner,'environment',return_value={}),
              patch.object(runner,'actor',side_effect=actor)as launch):
            for stage in('preflight','debug','train'):
                runner.child(self.native,stage)
        factory.assert_called_once_with(self.native)
        self.assertEqual(launch.call_count,3)
        self.assertEqual(lease.wait_until_available.call_count,3)
        self.assertEqual([call.args[1:3]for call in status.call_args_list],
            [('waiting_GPU','WAITING'),('preflight','RUNNING'),
             ('waiting_GPU','WAITING'),('CUDA_DEBUG','RUNNING'),
             ('waiting_GPU','WAITING'),('train','RUNNING')])

    def test_actual_derivative_bank_and_unchanged_scientific_mapping_are_admitted(self):
        before=copy.deepcopy(self.native)
        lease=Mock();factory=Mock(return_value=lease)
        with (patch.object(runner,'_lease_class',return_value=factory),patch.object(runner,'status'),
              patch.object(runner,'environment',return_value={}),patch.object(runner,'actor')as launch):
            runner.child(self.native,'train')
        factory.assert_called_once_with(self.native)
        self.assertEqual(self.native,before)
        self.assertEqual(self.native['candidates_per_source'],512)
        self.assertEqual(self.native['epochs'],250)
        launch.assert_called_once()

    def test_initial_wait_bank_preparation_and_stage_wait_reuse_one_exact_lease(self):
        native=dict(self.native,candidates_per_source=128)
        lease=Mock();factory=Mock(return_value=lease)
        with (patch.object(runner,'_lease_class',return_value=factory),patch.object(runner,'status'),
              patch.object(runner,'environment',return_value={}),patch.object(runner,'actor')):
            runner._wait_for_GPU(native)
            self.assertIs(runner.prepare_bank(native),native)
            runner.child(native,'train')
            runner.resource_guard(native)
        factory.assert_called_once_with(native)
        self.assertEqual(lease.wait_until_available.call_count,2)
        lease.verify.assert_called_once_with(force=False,idle=False)

    def test_wait_identity_failure_stops_before_any_child_or_environment(self):
        lease=Mock();lease.wait_until_available.side_effect=RuntimeError('Exact PID/birth/UID changed')
        with (patch.object(runner,'_lease_class',return_value=Mock(return_value=lease)),
              patch.object(runner,'actor')as launch,patch.object(runner,'environment')as env,
              self.assertRaisesRegex(RuntimeError,'PID/birth/UID')):
            runner.child(self.native,'train')
        launch.assert_not_called();env.assert_not_called()

    def test_source_and_derivative_get_separate_exact_bank_admissions_without_duplicate_controller(self):
        original=dict(self.native,bank=self.native['source_bank'],bank_sha256=self.native['source_bank_sha256'])
        factory=Mock(side_effect=[Mock(),Mock()])
        with patch.object(runner,'_lease_class',return_value=factory):
            original_lease=runner._lease_for(original)
            derivative_lease=runner._lease_for(self.native)
            self.assertIs(runner._lease_for(self.native),derivative_lease)
        self.assertIsNot(original_lease,derivative_lease)
        self.assertEqual(factory.call_count,2)
        self.assertEqual(original['root'],self.native['root'])


class ActualGroupWaitingTests(unittest.TestCase):
    start=fixture.GroupV5Tests.start
    process=fixture.GroupV5Tests.process
    query=fixture.GroupV5Tests.query

    def setUp(self):
        fixture.GroupV5Tests.setUp(self)

    def test_foreign_GPU_actor_queues_then_releases_without_registering_or_signaling_it(self):
        self.apps='999, GPU-exact'
        callbacks=[]
        def released(observation):
            callbacks.append(observation)
            self.apps=''
        result=self.lease.wait_until_available(on_wait=released)
        self.assertEqual(len(callbacks),1)
        self.assertFalse(callbacks[0]['GPU_available'])
        self.assertEqual(callbacks[0]['occupied_GPU_compute_actors'][0]['pid'],999)
        self.assertFalse(callbacks[0]['occupied_GPU_compute_actors'][0]['processes_signaled'])
        self.assertEqual(set(self.lease.registration),{str(g)for g in range(7)})
        self.assertEqual(result['own_GPU_compute_PIDs'],[])

    def test_actual_wait_rejects_changed_controller_identity_without_launch(self):
        self.apps='999, GPU-exact'
        def replaced(observation):self.controller.birth+=1
        with self.assertRaisesRegex(RuntimeError,'PID/birth/UID'):
            self.lease.wait_until_available(on_wait=replaced)

    def test_actual_wait_rejects_changed_original_bank_metadata(self):
        self.apps='999, GPU-exact'
        def changed(observation):self.bank.write_text('changed DEBUG source metadata')
        with self.assertRaisesRegex(ValueError,'metadata changed'):
            self.lease.wait_until_available(on_wait=changed)


if __name__=='__main__':unittest.main()
