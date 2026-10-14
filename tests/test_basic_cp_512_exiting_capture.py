"""CPU DEBUG: real resolver/ledger capture of naturally exiting source probes."""
import copy
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_512_claims as claims
from hiercp_v1x import basic_cp_512_resources as resources
from tests.test_basic_cp_ablation_resources import Actor, identity


class Probe(Actor):
    def __init__(self, pid, *, finish=False):
        super().__init__(pid, [], birth=1200+pid, cpus=resources.SHARED_CPU_AFFINITY)
        self.finish = finish
        self.command_reads = 0

    def cmdline(self):
        self.command_reads += 1
        if self.finish and self.command_reads == 2:
            self.state = 'zombie'
        return self.command


class ExitingCaptureTests(unittest.TestCase):
    def setUp(self):
        self.source = Actor(200, ['DEBUG_full642_source'], birth=1200,
            cpus=resources.SHARED_CPU_AFFINITY)
        self.workers = [Actor(300+i, ['DEBUG_worker', str(i)], birth=1300+i,
            cpus=resources.SHARED_CPU_AFFINITY) for i in range(24)]
        self.tracker = Actor(400, ['DEBUG_resource_tracker'], birth=1400,
            cpus=resources.SHARED_CPU_AFFINITY)
        self.probe = Probe(500)
        self.source.descendants = self.workers+[self.tracker, self.probe]
        self.processes = {p.pid:p for p in [self.source, *self.source.descendants]}
        self.lease = resources.common.BasicLease.__new__(resources.common.BasicLease)
        self.lease._managed = {}
        self.lease._fork_pending = {}
        self.lease._terminal_observations = {}
        self.resolver = resources.ProcessResolver(self.lease)
        self.ledger = claims.Ledger.__new__(claims.Ledger)
        self.ledger.resolver = self.resolver
        self.claim = dict(command=self.source.command, child_identity=identity(self.source), captured=[])
        self.flags = {self.probe.pid:4}
        self.patches = [patch.object(resources.psutil, 'Process', side_effect=self.process),
            patch.object(resources.common, '_task_flags', side_effect=lambda p:self.flags.get(p.pid,0)),
            patch.object(resources.common.time, 'sleep')]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)

    def process(self, pid):
        if pid not in self.processes:
            raise resources.psutil.NoSuchProcess(pid)
        return self.processes[pid]

    def test_full24_workers_tracker_retained_and_unseen_probe_proves_terminal(self):
        self.probe.finish = True
        self.ledger._capture(self.claim)
        self.assertEqual({a['pid'] for a in self.claim['captured']},
            {self.source.pid, self.tracker.pid, *(p.pid for p in self.workers)})
        self.assertEqual(len(self.claim['captured']),26)
        evidence=self.lease._terminal_observations[self.probe.pid]
        self.assertEqual(evidence['status'],'zombie')
        self.assertEqual(evidence['UID'],41833)
        self.assertIn('completed kernel exit',evidence['reason'])
        self.assertNotIn(self.probe.pid,self.lease._managed)

    def test_known_probe_keeps_original_argv_while_kernel_exit_is_live(self):
        original = dict(identity(self.probe),command=['nvidia-smi',
            '--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'])
        self.lease._managed[self.probe.pid] = copy.deepcopy(original)
        self.ledger._capture(self.claim)
        stored=next(a for a in self.claim['captured'] if a['pid']==self.probe.pid)
        self.assertEqual(stored,original)
        self.assertEqual(self.ledger._actual(stored),'live')
        self.probe.state='zombie'
        self.assertEqual(self.ledger._actual(stored),'terminal')

    def test_uncaptured_pf_exiting_still_live_is_fatal_not_silently_skipped(self):
        with self.assertRaisesRegex(RuntimeError,'still live after bounded'):
            self.ledger._capture(self.claim)
        self.assertEqual(self.claim['captured'],[])
        self.assertEqual(self.probe.state,'sleeping')

    def test_unknown_live_empty_argv_without_kernel_exit_remains_fatal(self):
        self.flags[self.probe.pid]=0
        with self.assertRaisesRegex(RuntimeError,'unknown empty argv'):
            self.ledger._capture(self.claim)
        self.assertEqual(self.claim['captured'],[])

    def test_foreign_uid_or_reused_known_birth_cannot_be_excluded_as_probe_exit(self):
        self.probe.finish=True;self.probe.uid=999
        with self.assertRaisesRegex(RuntimeError,'foreign'):
            self.ledger._capture(self.claim)
        self.probe.uid=41833
        self.lease._managed[self.probe.pid]=dict(identity(self.probe),
            create_time=self.probe.birth-1,command=['nvidia-smi'])
        with self.assertRaisesRegex(RuntimeError,'PID/birth/UID'):
            self.ledger._capture(self.claim)

    def test_identity_contract_never_accepts_dummy_or_empty_argv(self):
        with self.assertRaisesRegex(ValueError,'nonempty argv'):
            claims.identity(identity(self.probe))
        self.probe.command=['changed-live-command'];self.flags[self.probe.pid]=0
        self.lease._managed[self.probe.pid]=dict(identity(self.probe),command=['nvidia-smi'])
        with self.assertRaisesRegex(RuntimeError,'argv changed'):
            self.ledger._capture(self.claim)


if __name__=='__main__':unittest.main()
