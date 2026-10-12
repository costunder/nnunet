"""CPU unit evidence for terminal stage races; no server or CUDA execution."""
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import v25_direct_resources as direct
from hiercp_v1x import v25_terminal_descendant_resources as runtime

GIB = 2**30


def identity(pid, *, uid=41833):
    return dict(pid=pid, create_time=float(pid),
        command=['python', 'DEBUG_stage_%d.py' % pid], UID=uid)


def process(value, affinity):
    result = Mock()
    result.pid = value['pid']
    result.status.return_value = direct.psutil.STATUS_RUNNING
    result.create_time.return_value = value['create_time']
    result.cmdline.return_value = value['command']
    result.uids.return_value = Mock(real=value['UID'])
    result.cpu_affinity.return_value = affinity
    result.children.return_value = []
    result.memory_info.return_value = Mock(rss=GIB)
    return result


class TerminalDescendantTests(unittest.TestCase):
    def setUp(self):
        self.original_methods = {name: getattr(direct.DirectAllocationLease, name)
            for name in ('_tree', '_resources', '_gpu', '_host_lifetime', 'verify')}
        self.original_source = Path(direct.__file__).read_bytes()
        self.addCleanup(setattr, direct.DirectAllocationLease, '_tree', self.original_methods['_tree'])
        # Isolate installation from another test module's interpreter runtime.
        direct.DirectAllocationLease._tree = runtime._ORIGINAL_TREE
        self.lease = direct.DirectAllocationLease.__new__(direct.DirectAllocationLease)
        self.controller_identity = identity(100)
        self.own_identity = identity(101)
        self.sibling_identity = identity(102)
        self.controller = process(self.controller_identity, list(range(24)))
        self.own = process(self.own_identity, list(range(6)))
        self.sibling = process(self.sibling_identity, list(range(6,12)))
        self.controller.children.return_value = [self.own, self.sibling]
        self.processes = {100:self.controller,101:self.own,102:self.sibling}
        self.lease.request = dict(controller_identity=self.controller_identity,
            aggregate_CPU_affinity=list(range(24)), CPU_affinity=list(range(6)),
            GPU_UUID='GPU-DEBUG', gpu=1, node='ece-a6gpu6',
            resource_guard_sha256=hashlib.sha256(self.original_source).hexdigest(),
            cgroup_admission=dict(controller_cgroup='0::/DEBUG_direct\n',
                memory_limits=[dict(path='/sys/fs/cgroup/memory.max',limit_bytes=None)]),
            host_lifetime=dict(allocation_request_sha256='a'*64))
        self.lease.self_identity = self.own_identity
        self.lease.managed = {}
        self.process_patch = patch.object(direct.psutil, 'Process',
            side_effect=lambda pid=None:self.processes[101 if pid is None else pid])
        self.process_patch.start(); self.addCleanup(self.process_patch.stop)

    def install(self):
        return runtime.install_runtime()

    def zombie(self, actor=None):
        actor = self.sibling if actor is None else actor
        actor.status.return_value = direct.psutil.STATUS_ZOMBIE
        actor.cmdline.side_effect = direct.psutil.ZombieProcess(actor.pid)

    def test_original_reproduces_normal_completed_sibling_failure(self):
        self.zombie()
        with self.assertRaisesRegex(RuntimeError, 'Direct managed actor is a zombie'):
            self.lease._tree()

    def test_verified_own_zombie_is_inactive_and_live_actors_remain(self):
        self.install(); self.zombie()
        controller, own, tree = self.lease._tree()
        self.assertIs(controller,self.controller); self.assertIs(own,self.own)
        self.assertEqual(set(tree),{100,101})
        self.assertEqual(self.lease._terminal_descendants[102],
            dict(pid=102,create_time=102.,UID=41833))
        self.sibling.cmdline.assert_not_called()
        self.lease._tree()  # The same unreaped child remains safely inactive.

    def test_running_to_zombie_during_cmdline_is_verified(self):
        self.install()
        self.sibling.status.side_effect = [direct.psutil.STATUS_RUNNING,
            direct.psutil.STATUS_ZOMBIE,direct.psutil.STATUS_ZOMBIE]
        self.sibling.cmdline.side_effect = direct.psutil.ZombieProcess(102)
        self.assertEqual(set(self.lease._tree()[2]),{100,101})

    def test_linux_empty_cmdline_terminal_transition_is_verified(self):
        self.install()
        self.sibling.status.side_effect = [direct.psutil.STATUS_RUNNING,
            direct.psutil.STATUS_ZOMBIE,direct.psutil.STATUS_ZOMBIE,direct.psutil.STATUS_ZOMBIE]
        self.sibling.cmdline.return_value = []
        self.assertEqual(set(self.lease._tree()[2]),{100,101})

    def test_child_disappearing_during_snapshot_is_not_admitted(self):
        self.install()
        self.sibling.status.side_effect = direct.psutil.NoSuchProcess(102)
        self.assertEqual(set(self.lease._tree()[2]),{100,101})
        self.assertNotIn(102,self.lease.managed)

    def test_current_actor_and_controller_zombies_still_fail(self):
        self.install()
        for actor in (self.own,self.controller):
            with self.subTest(pid=actor.pid):
                actor.status.return_value=direct.psutil.STATUS_ZOMBIE
                with self.assertRaisesRegex(RuntimeError,'Direct managed actor is a zombie'):
                    self.lease._tree()
                actor.status.return_value=direct.psutil.STATUS_RUNNING

    def test_foreign_uid_active_or_terminal_still_fails(self):
        self.install()
        self.sibling.uids.return_value.real=1
        for terminal in (False,True):
            with self.subTest(terminal=terminal):
                self.sibling.status.return_value=(direct.psutil.STATUS_ZOMBIE if terminal
                    else direct.psutil.STATUS_RUNNING)
                with self.assertRaisesRegex(RuntimeError,'Foreign-UID'):
                    self.lease._tree()

    def test_managed_pid_birth_or_argv_change_still_fails(self):
        self.install();self.lease._tree()
        self.sibling.create_time.return_value=999.
        with self.assertRaisesRegex(RuntimeError,'PID/birth/argv'):
            self.lease._tree()
        self.zombie()
        with self.assertRaisesRegex(RuntimeError,'PID/birth/UID'):
            self.lease._tree()
        self.sibling.create_time.return_value=102.
        self.sibling.status.return_value=direct.psutil.STATUS_RUNNING
        self.sibling.cmdline.side_effect=None
        self.sibling.cmdline.return_value=['python','different.py']
        with self.assertRaisesRegex(RuntimeError,'PID/birth/argv'):
            self.lease._tree()

    def test_terminal_pid_reuse_and_resurrection_still_fail(self):
        self.install();self.zombie();self.lease._tree()
        self.sibling.create_time.return_value=999.
        with self.assertRaisesRegex(RuntimeError,'Terminal direct child PID/birth'):
            self.lease._tree()
        self.sibling.create_time.return_value=102.
        self.sibling.status.return_value=direct.psutil.STATUS_RUNNING
        self.sibling.cmdline.side_effect=None
        with self.assertRaisesRegex(RuntimeError,'Previously terminal'):
            self.lease._tree()

    def test_unverifiable_terminal_uid_or_fake_zombie_error_is_not_ignored(self):
        self.install();self.zombie()
        self.sibling.uids.side_effect=direct.psutil.ZombieProcess(102)
        with self.assertRaisesRegex(RuntimeError,'UID/birth could not be verified'):
            self.lease._tree()
        self.sibling.uids.side_effect=None
        self.sibling.status.return_value=direct.psutil.STATUS_RUNNING
        with self.assertRaisesRegex(RuntimeError,'must have actual zombie status'):
            self.lease._tree()

    def test_access_denied_and_invalid_live_identity_are_not_hidden(self):
        self.install()
        self.sibling.status.side_effect=direct.psutil.AccessDenied(102)
        with self.assertRaises(direct.psutil.AccessDenied):self.lease._tree()
        self.sibling.status.side_effect=None
        self.sibling.cmdline.return_value=[]
        with self.assertRaisesRegex(RuntimeError,'Exact live descendant'):
            self.lease._tree()

    def test_terminal_birth_must_be_finite_positive(self):
        self.install();self.zombie()
        for birth in (0.,-1.,float('nan'),True):
            with self.subTest(birth=birth):
                self.sibling.create_time.return_value=birth
                with self.assertRaisesRegex(RuntimeError,'PID and birth'):
                    self.lease._tree()

    def test_controller_identity_cpu_and_current_ancestry_cpu_unchanged(self):
        self.install();self.zombie()
        self.controller.cmdline.return_value=['python','changed.py']
        with self.assertRaisesRegex(RuntimeError,'identity changed'):self.lease._tree()
        self.controller.cmdline.return_value=self.controller_identity['command']
        self.controller.cpu_affinity.return_value=[0]
        with self.assertRaisesRegex(RuntimeError,'aggregate24'):self.lease._tree()
        self.controller.cpu_affinity.return_value=list(range(24))
        self.own.cpu_affinity.return_value=[0]
        with self.assertRaisesRegex(RuntimeError,'six-CPU'):self.lease._tree()
        self.own.cpu_affinity.return_value=list(range(6))
        self.controller.children.return_value=[self.sibling]
        with self.assertRaisesRegex(RuntimeError,'actual descendant'):self.lease._tree()

    def test_zombie_pid_is_not_a_cuda_allowlist_exception(self):
        self.install();self.zombie()
        tree=self.lease._tree()[2]
        def query(command):
            return ('102, GPU-DEBUG' if '--query-compute-apps=pid,gpu_uuid' in command
                else '1, GPU-DEBUG, 49140, 49000, Disabled')
        with patch.object(direct,'_query',side_effect=query):
            with self.assertRaisesRegex(RuntimeError,'Foreign or unrelated.*102'):
                self.lease._gpu(tree,initial=True)

    def test_resource_memory_and_cgroup_guards_still_enforced(self):
        self.install();self.zombie()
        controller,own,tree=self.lease._tree()
        memory=Mock(total=1024*GIB,available=800*GIB)
        cgroup=copy.deepcopy(self.lease.request['cgroup_admission'])
        with patch.object(direct,'capture_cgroup_admission',side_effect=lambda pid:copy.deepcopy(cgroup)),\
                patch.object(direct.psutil,'virtual_memory',return_value=memory):
            self.lease._resources(controller,own,tree)
            self.assertEqual(self.lease.actual_resources['aggregate_process_and_children_RSS_bytes'],2*GIB)
            self.own.memory_info.return_value.rss=385*GIB
            with self.assertRaisesRegex(MemoryError,'Aggregate exact'):
                self.lease._resources(controller,own,tree)
            self.own.memory_info.return_value.rss=GIB
            memory.available=GIB
            with self.assertRaisesRegex(MemoryError,'available host RAM'):
                self.lease._resources(controller,own,tree)
            memory.available=800*GIB;cgroup['controller_cgroup']='changed'
            with self.assertRaisesRegex(RuntimeError,'cgroup.*changed'):
                self.lease._resources(controller,own,tree)

    def test_host_lifetime_guard_still_uses_exact_original_job(self):
        self.install()
        self.lease.lifetime=dict(job_id='123.ECE-util1',expires_at_unix=10000.)
        job=dict(job_state='R',Job_Owner='aicompetition06@ece-a6gpu6',
            Resource_List=dict(ngpus=1,walltime='01:00:00'),exec_host='ece-a6gpu6/32',stime=6400)
        with patch.object(direct,'_query',side_effect=lambda command:json.dumps(dict(Jobs={'123.ECE-util1':job}))):
            self.lease._host_lifetime()
            job['stime']=6401.1
            with self.assertRaisesRegex(RuntimeError,'start/walltime'):
                self.lease._host_lifetime()

    def test_installer_idempotent_frozen_bytes_unchanged_other_checks_same(self):
        first=self.install();second=self.install()
        self.assertEqual(first,second)
        self.assertEqual(first['original_resource_guard_sha256'],hashlib.sha256(self.original_source).hexdigest())
        self.assertEqual(Path(direct.__file__).read_bytes(),self.original_source)
        for name in ('_resources','_gpu','_host_lifetime','verify'):
            self.assertIs(getattr(direct.DirectAllocationLease,name),self.original_methods[name])
        direct.DirectAllocationLease._tree=lambda lease:None
        with self.assertRaisesRegex(ValueError,'Unknown existing'):self.install()

    def test_changed_execution_source_is_rejected(self):
        self.install()
        with patch.object(runtime,'_sha',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'execution source changed'):self.lease._tree()
            with self.assertRaisesRegex(ValueError,'execution source changed'):self.install()


if __name__=='__main__':unittest.main()
