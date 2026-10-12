"""CPU UNIT evidence for truthful direct admission, never CUDA execution."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import v25_direct_resources as direct

GIB = 2**30


def proof_document():
    return dict(format=direct.DIRECT_FORMAT, arm='A', gpu=1, GPU_UUID='GPU-DEBUG',
        CPU_affinity=list(range(6)), UID=41833, owner='aicompetition06', node='ece-a6gpu6',
        PBS_owned_reservation=False, explicit_direct_execution_authorized=True,
        controller_identity=dict(pid=100, create_time=100., command=['python', 'DEBUG_controller.py'], UID=41833),
        group_root='/var/tmp/DEBUG_group',
        host_lifetime=dict(allocation_request='/var/tmp/DEBUG_owned6.json',
            allocation_request_sha256='a'*64, expires_at_unix=10000.),
        resource_guard_sha256='b'*64, aggregate_CPU_affinity=list(range(24)),
        aggregate_native_and_cache_RSS_bytes=416*GIB,
        aggregate_GNN_and_cache_RSS_bytes=288*GIB, canonical_cache_reserve_bytes=32*GIB,
        cgroup_admission=dict(controller_cgroup='0::/DEBUG_direct\n',
            memory_limits=[dict(path='/sys/fs/cgroup/memory.max', limit_bytes=None)]))


def process(pid, identity, affinity, rss=GIB):
    actor = Mock()
    actor.pid = pid
    actor.status.return_value = 'running'
    actor.create_time.return_value = identity['create_time']
    actor.cmdline.return_value = identity['command']
    actor.uids.return_value = Mock(real=identity['UID'])
    actor.cpu_affinity.return_value = affinity
    actor.memory_info.return_value = Mock(rss=rss)
    actor.children.return_value = []
    return actor


class DeclarationTests(unittest.TestCase):
    def test_closed_full_four_arm_host_budget_and_honest_direct_mode(self):
        value = proof_document()
        self.assertIs(direct.validate_direct_document(value), value)
        self.assertFalse(value['PBS_owned_reservation'])
        self.assertEqual(value['aggregate_native_and_cache_RSS_bytes'], (4*96+32)*GIB)
        self.assertEqual(value['aggregate_GNN_and_cache_RSS_bytes'], (4*64+32)*GIB)

    def test_changed_scale_authorization_gpu_uid_arm_or_cpu_rejected(self):
        changes = [('PBS_owned_reservation', True), ('explicit_direct_execution_authorized', False),
            ('gpu', 0), ('gpu', True), ('arm', 'E'), ('UID', 1), ('CPU_affinity', list(range(4))),
            ('aggregate_CPU_affinity', list(range(6))), ('aggregate_native_and_cache_RSS_bytes', 128*GIB),
            ('aggregate_GNN_and_cache_RSS_bytes', 64*GIB), ('canonical_cache_reserve_bytes', 1*GIB),
            ('hidden_subset', True)]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                document = proof_document(); document[key] = value
                with self.assertRaises(ValueError):
                    direct.validate_direct_document(document)

    def test_restricted_actual_cgroup_is_not_relabelled_as_host_1tib(self):
        document = proof_document()
        document['cgroup_admission']['memory_limits'][0]['limit_bytes'] = 128*GIB
        with self.assertRaisesRegex(ValueError, 'actual cgroup memory limit'):
            direct.validate_direct_document(document)


class LiveGuardUnitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='DEBUG_direct_')
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.group = root/'group'; self.group.mkdir(); (self.group/'A').mkdir()
        self.lifetime_path = root/'owned6.json'
        self.lifetime = dict(format=direct.OWNED_FORMAT, UID=41833, owner='aicompetition06',
            node='ece-a6gpu6', gpu=6, GPU_UUID='GPU-OWNED6', job_id='129664.ECE-util1', expires_at_unix=10000.)
        self.lifetime_path.write_text(json.dumps(self.lifetime))
        self.document = proof_document()
        self.document['group_root'] = str(self.group)
        self.document['host_lifetime']['allocation_request'] = str(self.lifetime_path)
        self.document['host_lifetime']['allocation_request_sha256'] = hashlib.sha256(self.lifetime_path.read_bytes()).hexdigest()
        self.document['resource_guard_sha256'] = hashlib.sha256(Path(direct.__file__).read_bytes()).hexdigest()
        self.path = root/'direct.json'; self.write_document()
        self.self_identity = dict(pid=101, create_time=101., command=['python', 'DEBUG_stage.py'], UID=41833)
        self.controller = process(100, self.document['controller_identity'], list(range(24)))
        self.actor = process(101, self.self_identity, list(range(6)))
        self.controller.children.return_value = [self.actor]
        self.processes = {100:self.controller, 101:self.actor}
        self.apps = ''
        self.gpu = '1, GPU-DEBUG, 49140, 49000, Disabled'
        self.job = dict(job_state='R', Job_Owner='aicompetition06@ece-a6gpu6', exec_host='ece-a6gpu6/32',
            Resource_List=dict(ngpus=1, walltime='01:00:00'), stime=6400)
        self.memory = Mock(total=1024*GIB, available=800*GIB)
        self.cgroup = copy.deepcopy(self.document['cgroup_admission'])
        for context in (
            patch.object(direct, '_owned_regular', side_effect=lambda path, **kwargs: Path(path)),
            patch.object(direct.os, 'getuid', create=True, return_value=41833),
            patch.object(direct.socket, 'gethostname', return_value='ece-a6gpu6'),
            patch.object(direct.psutil, 'Process', side_effect=lambda pid=None: self.processes[101 if pid is None else pid]),
            patch.object(direct.psutil, 'virtual_memory', side_effect=lambda:self.memory),
            patch.object(direct, 'capture_cgroup_admission', side_effect=lambda pid:copy.deepcopy(self.cgroup)),
            patch.object(direct, '_query', side_effect=self.query),
            patch.object(direct.time, 'time', return_value=9000.),
            patch.object(direct.time, 'monotonic', return_value=500.),
            patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'1'})):
            context.start(); self.addCleanup(context.stop)

    def write_document(self):
        self.path.write_text(json.dumps(self.document))

    def query(self, command):
        if command[0] == '/opt/pbs/bin/qstat':
            return json.dumps(dict(Jobs={'129664.ECE-util1':self.job}))
        if '--query-compute-apps=pid,gpu_uuid' in command:
            return self.apps
        return self.gpu

    def test_real_admission_calls_uuid_apps_lifetime_and_aggregate_budget(self):
        lease = direct.DirectAllocationLease(self.path)
        self.assertEqual(lease.request['arm'], 'A')
        self.assertEqual(lease.actual_resources['aggregate_process_and_children_RSS_bytes'], 2*GIB)
        self.assertFalse(lease.actual_resources['PBS_GPU_owned_reservation'])
        self.assertEqual(lease.actual_resources['PBS_job_role'], 'host_lifetime_only')

    def test_foreign_or_unrelated_own_cuda_pid_refused(self):
        self.apps = '999, GPU-DEBUG'
        with self.assertRaisesRegex(RuntimeError, 'Foreign or unrelated CUDA process.*999'):
            direct.DirectAllocationLease(self.path)

    def test_exact_own_descendant_cuda_pid_is_accepted(self):
        self.apps = '101, GPU-DEBUG'
        direct.DirectAllocationLease(self.path)

    def test_same_uid_without_controller_ancestry_is_rejected(self):
        self.controller.children.return_value = []
        with self.assertRaisesRegex(RuntimeError, 'actual descendant'):
            direct.DirectAllocationLease(self.path)

    def test_controller_pid_reuse_or_changed_argv_refused(self):
        self.controller.cmdline.return_value = ['python', 'unrelated.py']
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            direct.DirectAllocationLease(self.path)

    def test_original_full_gpu_free_memory_is_required_at_stage_start(self):
        self.gpu = '1, GPU-DEBUG, 49140, 1024, Disabled'
        with self.assertRaisesRegex(RuntimeError, 'initial40GiB'):
            direct.DirectAllocationLease(self.path)

    def test_different_uuid_and_mig_are_rejected(self):
        for row in ('1, GPU-DIFFERENT, 49140, 49000, Disabled', '1, GPU-DEBUG, 49140, 49000, Enabled'):
            with self.subTest(row=row):
                self.gpu = row
                with self.assertRaisesRegex(RuntimeError, 'physical GPU'):
                    direct.DirectAllocationLease(self.path)

    def test_expired_host_lifetime_stops_cooperatively_before_external_calls(self):
        with patch.object(direct.time, 'time', return_value=9900.), patch.object(direct, '_query') as query:
            with self.assertRaisesRegex(RuntimeError, 'preserve exact checkpoint'):
                direct.DirectAllocationLease(self.path)
            query.assert_not_called()

    def test_changed_owned_job_stime_is_rejected(self):
        self.job['stime'] += 30
        with self.assertRaisesRegex(RuntimeError, 'start/walltime'):
            direct.DirectAllocationLease(self.path)

    def test_actual_pbs_local_text_start_keeps_same_sealed_expiration(self):
        self.lifetime['expires_at_unix'] = 1791868590.
        self.lifetime_path.write_text(json.dumps(self.lifetime))
        self.document['host_lifetime']['expires_at_unix'] = 1791868590.
        self.document['host_lifetime']['allocation_request_sha256'] = hashlib.sha256(self.lifetime_path.read_bytes()).hexdigest()
        self.write_document()
        self.job['stime'] = 'Mon Oct 12 14:16:30 2026'
        self.job['Resource_List']['walltime'] = '24:00:00'
        with patch.object(direct.time, 'mktime', return_value=1791782190.) as local:
            lease = direct.DirectAllocationLease(self.path)
            self.assertEqual(lease.lifetime['expires_at_unix'], 1791868590.)
            self.assertEqual(local.call_args.args[0].tm_year, 2026)
            self.assertEqual(local.call_args.args[0].tm_hour, 14)

    def test_unparseable_pbs_text_is_reported_and_never_accepted(self):
        self.job['stime'] = 'not a valid PBS date'
        with self.assertRaisesRegex(RuntimeError, 'Actual PBS start time.*server-local PBS date text'):
            direct.DirectAllocationLease(self.path)

    def test_numeric_pbs_seconds_and_numeric_text_keep_same_expiration(self):
        for value in (6400, 6400., '6400'):
            with self.subTest(stime=value):
                self.job['stime'] = value
                with patch.object(direct.time, 'mktime') as local:
                    lease = direct.DirectAllocationLease(self.path)
                    self.assertEqual(lease.lifetime['expires_at_unix'], 10000.)
                    local.assert_not_called()

    def test_changed_cpu_and_cgroup_are_not_ignored(self):
        self.actor.cpu_affinity.return_value = [0]
        with self.assertRaisesRegex(RuntimeError, 'six-CPU'):
            direct.DirectAllocationLease(self.path)
        self.actor.cpu_affinity.return_value = list(range(6))
        self.cgroup['controller_cgroup'] = '0::/changed\n'
        with self.assertRaisesRegex(RuntimeError, 'cgroup.*changed'):
            direct.DirectAllocationLease(self.path)

    def test_low_actual_host_memory_or_excess_tree_rss_rejected(self):
        self.memory.available = 1*GIB
        with self.assertRaisesRegex(MemoryError, 'available host RAM'):
            direct.DirectAllocationLease(self.path)
        self.memory.available = 800*GIB
        self.actor.memory_info.return_value.rss = 385*GIB
        with self.assertRaisesRegex(MemoryError, 'Aggregate exact'):
            direct.DirectAllocationLease(self.path)

    def test_immutable_proof_bytes_change_rejected(self):
        lease = direct.DirectAllocationLease(self.path)
        self.document['arm'] = 'B'; self.write_document()
        with self.assertRaisesRegex(ValueError, 'Immutable direct request'):
            lease.verify()

    def test_cached_select_cuda_does_not_query_scheduler_per_training_step(self):
        lease = direct.DirectAllocationLease(self.path)
        with patch.object(direct, '_query') as query:
            lease.select_cuda(); lease.select_cuda(); lease.verify()
            query.assert_not_called()
        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-DEBUG')

    def test_later_foreign_gpu_conflict_is_rejected_at_force_boundary(self):
        lease = direct.DirectAllocationLease(self.path)
        self.apps = '999, GPU-DEBUG'
        with self.assertRaisesRegex(RuntimeError, 'Foreign or unrelated'):
            lease.verify(force=True)

    def test_scorer_guard_executes_original_encode_with_original_training_flag(self):
        lease = direct.DirectAllocationLease(self.path)
        scorer = Mock(); scorer._encode = Mock(return_value='ACTUAL_FORWARD')
        original = scorer._encode
        restore = lease.install_scorer_guard(scorer)
        self.assertEqual(scorer._encode('REAL_CPU_BATCH', training=True), 'ACTUAL_FORWARD')
        original.assert_called_once_with('REAL_CPU_BATCH', training=True)
        restore(); self.assertIs(scorer._encode, original)

    def test_generic_resource_entry_rejects_arm_cpu_gpu_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'physical GPU differs'):
            direct.validate_resource_proof(self.path, gpu=4)
        with self.assertRaisesRegex(ValueError, 'CPU set differs'):
            direct.validate_resource_proof(self.path, gpu=1, cpu_affinity=[0])

    def test_generic_owned_mode_delegates_frozen_original_class(self):
        with patch.object(direct, 'OwnedAllocationLease') as original:
            actor = original.return_value; actor.request = self.lifetime
            guard = direct.validate_resource_proof(self.lifetime_path, gpu=6)
            self.assertIs(guard, actor)
            original.assert_called_once_with(self.lifetime_path)
            actor.verify.assert_called_once_with(force=True)


if __name__ == '__main__':
    unittest.main()
