"""CPU DEBUG: finished source claim may close before controller restores CPU8."""
import copy
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_512_resources as resource
from tests import test_basic_cp_512_resources as fixture
from tests.test_basic_cp_seven_resources import Actor

GIB=resource.GIB
PROBE=['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits']


class SourceCompletionTransitionTests(unittest.TestCase):
    start=fixture.NativeGuardTests.start
    process=fixture.NativeGuardTests.process
    query=fixture.NativeGuardTests.query
    ident=staticmethod(fixture.FullGuardTests.ident)
    inventory=fixture.FullGuardTests.inventory
    setUp=fixture.NativeGuardTests.setUp

    def finished_source(self,*,terminal=False):
        self.source_control=self.actors[self.requests[5]['controller_identity']['pid']]
        self.source_control.cpus=list(resource.SHARED_CPU_AFFINITY)
        self.source=self.actors[305];identity=self.ident(self.source)
        self.state['claims'].pop('5')
        self.state.update(source_actor_identity=identity,source_captured=[identity])
        self.source_control.descendants=[]
        if terminal:self.source.state='zombie';self.source.command=[]
        else:self.actors.pop(305)
        self.lease.source_complete.return_value=True
        self.start(patch.object(self.lease,'_phase_observations',return_value={
            'source_complete':not terminal,'source_phase_receipt_publication_mocked_for_CPU_DEBUG':True}))

    def probe(self,command=PROBE,*,rss=2**20):
        probe=Actor(9999,list(command));probe.cpus=list(resource.SHARED_CPU_AFFINITY);probe.rss=rss
        self.actors[probe.pid]=probe;self.source_control.descendants.append(probe)
        return probe

    def test_finished_bound_source_controller64_is_auxiliary_while_peer_native_continues(self):
        self.finished_source();value=self.lease.verify(force=True)
        self.assertEqual(value['source_process_tree_RSS_bytes'],0)
        self.assertGreater(value['native_RSS_by_GPU_bytes']['0'],0)
        detail=value['group_controller_observations']['5']
        self.assertTrue(detail['completed_source_controller_auxiliary_only'])
        self.assertFalse(detail['source_work_or_CUDA_permission_granted'])
        self.assertIn(self.source_control.pid,value['auxiliary_process_PIDs'])
        self.assertEqual(self.source_control.cpus,resource.SHARED_CPU_AFFINITY)

    def test_exact_readonly_probe_after_claim_close_is_auxiliary_and_RSS_is_counted(self):
        self.finished_source();probe=self.probe(rss=2*GIB)
        value=self.lease.verify(force=True)
        self.assertIn(probe.pid,value['auxiliary_process_PIDs'])
        self.assertGreaterEqual(value['auxiliary_RSS_bytes'],2*GIB)
        self.assertEqual(value['source_process_tree_RSS_bytes'],0)

    def test_unknown_live_model_child_remains_fatal(self):
        self.finished_source();self.probe(['python','--stage','train'])
        with self.assertRaisesRegex(RuntimeError,'Unexpected live child'):
            self.lease.verify(force=True)

    def test_no_previously_bound_actor_or_substituted_capture_cannot_borrow_cleanup(self):
        self.finished_source()
        for actor in (None,dict(self.state['source_actor_identity'],pid=12345)):
            self.state['source_actor_identity']=actor
            with self.subTest(actor=actor),self.assertRaisesRegex(RuntimeError,'provenance|original bound'):
                self.lease.verify(force=True)

    def test_full642_scientific_bank_completion_is_required(self):
        self.finished_source();self.lease.source_complete.return_value=False
        with self.assertRaisesRegex(RuntimeError,'full642'):self.lease.verify(force=True)

    def test_claim_removed_while_old_producer_is_live_still_fails(self):
        self.finished_source();self.actors[305]=self.source
        self.source_control.descendants=[self.source]
        with self.assertRaisesRegex(RuntimeError,'Live source'):
            self.lease.verify(force=True)

    def test_terminal_captured_actor_does_not_unlock_postsource_store_or_change_ledger(self):
        self.finished_source(terminal=True);before=copy.deepcopy(self.state)
        value=self.lease.verify(force=True)
        self.assertEqual(value['owned_tmpfs_limit_bytes'],144*GIB)
        self.assertEqual(value['group_controller_observations']['5']['captured_source_actual_states'],['terminal'])
        self.assertEqual(self.state,before)

    def test_aux16_cap_and_no_ownCUDA_survive_completed_source_transition(self):
        self.finished_source();probe=self.probe(rss=17*GIB)
        with self.assertRaisesRegex(MemoryError,'16GiB'):self.lease.verify(force=True)
        probe.rss=2**20;self.apps=f'{self.source_control.pid}, GPU-exact'
        with self.assertRaisesRegex(RuntimeError,'forbidden CUDA'):
            self.lease.verify(force=True)

    def test_probe_wrong_CPU_affinity_still_fails_original_group_guard(self):
        self.finished_source();self.probe().cpus=list(range(8))
        with self.assertRaisesRegex(RuntimeError,'affinity'):
            self.lease.verify(force=True)


if __name__=='__main__':unittest.main()
