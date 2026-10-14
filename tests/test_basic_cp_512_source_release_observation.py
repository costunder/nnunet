"""CPU DEBUG exact source64 survivor accounting after parent restores native8."""
import copy
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_512_resources as resource
from tests import test_basic_cp_512_resources as fixture
from tests.test_basic_cp_seven_resources import Actor

GIB=resource.GIB


class SourceReleaseTests(unittest.TestCase):
    start=fixture.FullGuardTests.start
    process=fixture.FullGuardTests.process
    query=fixture.FullGuardTests.query
    ident=staticmethod(fixture.FullGuardTests.ident)
    inventory=fixture.FullGuardTests.inventory
    setUp=fixture.FullGuardTests.setUp

    def parent_view(self):
        self.controller.cpus=list(self.value['cpu_affinity'])
        self.state['claims']['source']['token']='source'
        original_process=self.process
        self.start(patch.object(resource.psutil,'Process',side_effect=lambda pid=None:
            self.controller if pid is None else original_process(pid)))
        parent=resource.BasicLease(self.value)
        return parent

    def test_native8_parent_with_exact_live_source64_uses_source320_and_true_RSS(self):
        parent=self.parent_view();self.worker.rss=294*GIB
        value=parent.reservation_observer(self.state,None)
        self.assertEqual(value['own_CPU_affinity'],self.value['cpu_affinity'])
        self.assertEqual(value['source_process_tree_RSS_bytes'],self.stage.rss+self.worker.rss)
        self.assertEqual(value['effective_shared_bank_RSS_limit_bytes'],320*GIB)
        self.assertEqual(value['effective_RAM_cap_bytes'],512*GIB)
        self.assertTrue(value['source_release_observation']['closing_source_controller_native8'])
        self.assertEqual(self.controller.cpus,self.value['cpu_affinity'])
        self.assertEqual(self.worker.cpus,resource.SHARED_CPU_AFFINITY)
        self.assertIsNone(parent._claim_override)

    def test_retained_source320_hard_cap_and_ownCUDA_stay_fatal(self):
        parent=self.parent_view();self.worker.rss=321*GIB
        with self.assertRaisesRegex(MemoryError,'320GiB'):parent.reservation_observer(self.state,None)
        self.worker.rss=GIB;self.apps=f'{self.worker.pid}, GPU-exact'
        with self.assertRaisesRegex(RuntimeError,'forbidden CUDA'):
            parent.reservation_observer(self.state,None)

    def test_changed_source_UID_argv_and_CPU_cannot_borrow_teardown_permission(self):
        parent=self.parent_view()
        for key,value,reason in [('uid',999,'UID'),('command',['foreign','replacement'],'argv'),
                ('cpus',[0],'64CPU')]:
            original=getattr(self.worker,key);setattr(self.worker,key,value)
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,reason):
                parent.reservation_observer(self.state,None)
            setattr(self.worker,key,original)

    def test_unstarted_source_claim_keeps_native8_before_Popen(self):
        parent=self.parent_view();self.controller.descendants=[]
        claim=self.state['claims']['source'];claim['child_identity']=None;claim['captured']=[]
        value=parent.reservation_observer(self.state,None)
        self.assertEqual(value['source_process_tree_RSS_bytes'],0)
        self.assertNotIn('source_release_observation',value)

    def test_shared_execution_view_after_source_release_returns_exact_native8(self):
        self.parent_view();self.controller.descendants=[];self.state['claims']={}
        value=self.lease.reservation_observer(self.state,None)
        self.assertEqual(value['own_CPU_affinity'],self.value['cpu_affinity'])
        self.assertEqual(value['source_process_tree_RSS_bytes'],0)

    def test_unknown_direct_controller_child_never_borrows_source_permission(self):
        parent=self.parent_view()
        unknown=Actor(9999,['unknown','model']);unknown.cpus=list(resource.SHARED_CPU_AFFINITY)
        self.actors[unknown.pid]=unknown;self.controller.descendants.append(unknown)
        with self.assertRaisesRegex(RuntimeError,'Unexpected live actor'):
            parent.reservation_observer(self.state,None)


class NativePeerReleaseTests(unittest.TestCase):
    start=fixture.NativeGuardTests.start
    process=fixture.NativeGuardTests.process
    query=fixture.NativeGuardTests.query
    ident=staticmethod(fixture.FullGuardTests.ident)
    inventory=fixture.FullGuardTests.inventory
    setUp=fixture.NativeGuardTests.setUp

    def test_native_peer_guard_keeps_true_source_RSS_after_peer_controller_CPU8_restore(self):
        control=self.actors[self.requests[5]['controller_identity']['pid']]
        control.cpus=list(self.requests[5]['cpu_affinity'])
        source=self.actors[305];source.cpus=list(resource.SHARED_CPU_AFFINITY);source.rss=294*GIB
        self.state['claims']['5']['token']='source5'
        value=self.lease.verify(force=True)
        self.assertEqual(value['source_process_tree_RSS_bytes'],294*GIB)
        self.assertEqual(value['native_RSS_by_GPU_bytes']['0'],self.stage.rss+self.worker.rss)
        self.assertTrue(value['group_controller_observations']['5']['closing_source_controller_native8'])
        self.assertEqual(value['effective_RAM_cap_bytes'],512*GIB)

    def test_terminal_source_controller_does_not_drop_retained_orphan_source_RSS(self):
        control=self.actors[self.requests[5]['controller_identity']['pid']]
        control.state='zombie';control.command=[]
        source=self.actors[305];source.cpus=list(resource.SHARED_CPU_AFFINITY);source.rss=294*GIB
        self.state['claims']['5']['token']='source5'
        value=self.lease.verify(force=True)
        self.assertEqual(value['source_process_tree_RSS_bytes'],294*GIB)
        self.assertIsNone(value['group_controller_observations']['5']['actual_CPU_affinity'])


if __name__=='__main__':unittest.main()
