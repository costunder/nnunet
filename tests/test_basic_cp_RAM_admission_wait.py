"""CPU-only DEBUG: exact temporary RAM admission waits without shrinking caps."""
import unittest
from unittest.mock import Mock,patch

from hiercp_v1x import basic_cp_group_resources as original
from tests import test_basic_cp_resources_v5 as fixture
from tools import run_basic_cp_expanded as runner


class ActualRAMAdmissionTests(unittest.TestCase):
    start=fixture.GroupV5Tests.start
    process=fixture.GroupV5Tests.process
    query=fixture.GroupV5Tests.query

    def setUp(self):fixture.GroupV5Tests.setUp(self)

    def test_only_available_plus_actual_group_rss_shortage_is_typed_with_exact_bytes(self):
        self.host.available=700*2**30
        with self.assertRaises(original.RAMAdmissionUnavailable)as caught:
            self.lease.verify(force=True)
        values=caught.exception.observation
        self.assertEqual(values['host_available_RAM_bytes'],self.host.available)
        self.assertEqual(values['host_RAM_bytes'],self.host.total)
        self.assertEqual(values['required_group_and_cache_capacity_bytes'],800*2**30)
        self.assertEqual(values['observed_available_plus_group_RSS_bytes'],
                         self.host.available+values['group_process_tree_RSS_bytes'])
        self.assertEqual(values['shortfall_bytes'],800*2**30-values['observed_available_plus_group_RSS_bytes'])

    def test_total_host_capacity_and_cgroup_limits_remain_fatal_not_admission_wait(self):
        self.host.total=799*2**30
        with self.assertRaises(MemoryError)as caught:self.lease.verify(force=True)
        self.assertNotIsInstance(caught.exception,original.RAMAdmissionUnavailable)
        self.host.total=1007*2**30;self.admission['memory_limits'][0]['limit_bytes']=799*2**30
        self.lease=fixture.group.BasicLease(self.value)
        with self.assertRaises(MemoryError)as caught:self.lease.verify(force=True)
        self.assertNotIsInstance(caught.exception,original.RAMAdmissionUnavailable)

    def test_native_and_candidate_rss_caps_are_still_fatal_not_admission_wait(self):
        self.actors[301].rss=97*2**30
        with self.assertRaises(MemoryError)as caught:self.lease.verify(force=True)
        self.assertNotIsInstance(caught.exception,original.RAMAdmissionUnavailable)
        self.actors[301].rss=2**20;self.actors[305].rss=385*2**30
        with self.assertRaises(MemoryError)as caught:self.lease.verify(force=True)
        self.assertNotIsInstance(caught.exception,original.RAMAdmissionUnavailable)


class RunnerRAMWaitTests(unittest.TestCase):
    def setUp(self):
        self.native=dict(root='/DEBUG/RAM_wait',bank='/DEBUG/source/index.json',bank_sha256='a'*64)
        self.cache=patch.dict(runner.LEASES,{},clear=True);self.cache.start();self.addCleanup(self.cache.stop)
        self.shortage=original.RAMAdmissionUnavailable(host_total_bytes=1007*2**30,
            available_bytes=793*2**30,group_resident_bytes=2**20,required_bytes=800*2**30)

    def test_initial_wait_rechecks_same_lease_and_publishes_exact_existing_waiting_schema(self):
        lease=Mock();lease.wait_until_available.side_effect=[self.shortage,dict(admitted=True)]
        factory=Mock(return_value=lease)
        with (patch.object(runner,'_lease_class',return_value=factory),patch.object(runner,'status')as status,
              patch.object(runner.time,'sleep')as sleep):
            result=runner._wait_for_GPU(self.native)
        factory.assert_called_once_with(self.native)
        self.assertEqual(lease.wait_until_available.call_count,2)
        self.assertEqual(result,dict(admitted=True))
        sleep.assert_called_once_with(5)
        self.assertEqual(status.call_args.args[1:3],('waiting_RAM','WAITING'))
        self.assertEqual(status.call_args.kwargs['RAM_admission'],self.shortage.observation)

    def test_other_memory_or_identity_failure_is_not_caught_or_retried(self):
        for failure in(MemoryError('candidate384 cap'),RuntimeError('UID changed')):
            runner.LEASES.clear();lease=Mock();lease.wait_until_available.side_effect=failure
            with (patch.object(runner,'_lease_class',return_value=Mock(return_value=lease)),
                  patch.object(runner.time,'sleep')as sleep,patch.object(runner,'status')as status,
                  self.assertRaises(type(failure))as caught):
                runner._wait_for_GPU(self.native)
            self.assertIs(caught.exception,failure);sleep.assert_not_called();status.assert_not_called()

    def test_mid_training_resource_guard_propagates_even_typed_shortage(self):
        lease=Mock();lease.verify.side_effect=self.shortage
        with (patch.object(runner,'_lease_class',return_value=Mock(return_value=lease)),
              patch.object(runner.time,'sleep')as sleep,self.assertRaises(original.RAMAdmissionUnavailable)):
            runner.resource_guard(self.native)
        sleep.assert_not_called()


if __name__=='__main__':unittest.main()
