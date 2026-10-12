"""Actual lease invariants, mocked OS observations; no server/CUDA execution."""
import copy
import json
from pathlib import Path, PurePosixPath
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_ablation_resources as resource


class Actor:
    def __init__(self, pid, command, *, birth=700, uid=41833, cpus=None, rss=2**20):
        self.pid = pid
        self.command = command
        self.birth = birth
        self.uid = uid
        self.cpus = list(range(16, 22)) if cpus is None else cpus
        self.rss = rss
        self.state = 'sleeping'
        self.descendants = []
        self.fork_no_exec = False
        self.parent_actor = None

    def status(self): return self.state
    def create_time(self): return self.birth
    def uids(self): return SimpleNamespace(real=self.uid)
    def cmdline(self): return self.command
    def cpu_affinity(self): return self.cpus
    def memory_info(self): return SimpleNamespace(rss=self.rss)
    def children(self, recursive=False): return self.descendants
    def parent(self): return self.parent_actor
    def is_running(self): return self.state != 'gone'


def identity(actor):
    return dict(pid=actor.pid, create_time=actor.birth, UID=actor.uid, command=actor.command)


def request():
    return dict(format=resource.FORMAT, uid=41833, root='/var/tmp/own_basic/arm',
        native_root='/dev/shm/own_basic/arm', gpu=1, gpu_uuid='GPU-exact',
        cpu_affinity=list(range(16, 22)), GPU_memory_total_MiB=46068,
        controller_identity=dict(pid=100, create_time=700, UID=41833, command=['python', 'controller']),
        code_sha256={'hiercp_v1x/basic_cp_ablation_resources.py': 'a'*64},
        bank='/home/aicompetition06/bank/index.json', bank_sha256='b'*64,
        host_lifetime={'path': '/var/tmp/v25/allocation_request.json', 'sha256': 'c'*64,
            'expires_at_unix': 5000}, runtime_reserve_bytes=10*2**30, RSS_limit_bytes=96*2**30)


class BasicRequestTests(unittest.TestCase):
    def validate(self, value):
        with patch.object(resource, 'Path', PurePosixPath):
            return resource.validate_request(value)

    def test_exact_supported_each_gpu_cpu_mapping(self):
        for gpu, cpus in resource.CPUS.items():
            value=request(); value.update(gpu=gpu, cpu_affinity=cpus)
            self.assertIs(self.validate(value), value)

    def test_native_checkpoint_namespace_cannot_use_small_persistent_disk(self):
        value=request(); value['native_root']='/var/tmp/native'
        with self.assertRaisesRegex(ValueError, 'tmpfs'):
            self.validate(value)

    def test_256_derivative_requires_original_bank_sha(self):
        value=request(); value['source_bank']='/home/aicompetition06/original/index.json'
        with self.assertRaisesRegex(ValueError, 'together'):
            self.validate(value)
        value['source_bank_sha256']='d'*64
        self.assertIs(self.validate(value), value)

    def test_cannot_silently_shrink_ram_storage_cpu_or_gpu(self):
        for key, replacement in [('RSS_limit_bytes', 64*2**30),
                ('runtime_reserve_bytes', 2**30), ('cpu_affinity', [16]),
                ('GPU_memory_total_MiB', 24000), ('gpu', 5), ('uid', 0)]:
            value=request(); value[key]=replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(value)

    def test_private_guard_must_be_admitted_and_code_cannot_escape_root(self):
        for paths in [{'tools/runner.py': 'a'*64},
                {'hiercp_v1x/basic_cp_ablation_resources.py': 'a'*64, '../else.py': 'b'*64}]:
            value=request(); value['code_sha256']=paths
            with self.assertRaisesRegex(ValueError, 'SHA'):
                self.validate(value)

    def test_original_cache_admission_requires_a_matching_path_sha_pair(self):
        value=request();value['readonly_admission']='/exact/cache_admission.json'
        with self.assertRaisesRegex(ValueError,'pinned together'):
            self.validate(value)
        value['readonly_admission_sha256']='e'*64
        self.assertIs(self.validate(value),value)


class BasicForkFlagTests(unittest.TestCase):
    def test_linux_field9_handles_spaces_and_parentheses_in_the_task_name(self):
        actor = Actor(201, ['python'])
        for flags, expected in [(64, True), (4194304, False)]:
            row = f'201 (python (spawn) worker) S 100 100 100 0 -1 {flags} 0 0 0'
            with patch.object(resource.Path, 'read_text', return_value=row):
                self.assertIs(resource._fork_no_exec(actor), expected)

    def test_malformed_or_reused_pid_stat_cannot_supply_an_exec_fallback(self):
        actor = Actor(201, ['python'])
        for row in ['202 (python) S 100 100 100 0 -1 64',
                    '201 (python) S 100', '201 (python) S 100 100 100 0 -1 invalid']:
            with patch.object(resource.Path, 'read_text', return_value=row), self.assertRaisesRegex(
                    RuntimeError, 'fork/exec observation'):
                resource._fork_no_exec(actor)

    def test_naturally_released_worker_missing_proc_stat_becomes_no_such_process(self):
        actor = Actor(201, ['python'])
        actor.state = 'gone'
        with patch.object(resource.Path, 'read_text', side_effect=FileNotFoundError), self.assertRaises(
                resource.psutil.NoSuchProcess):
            resource._fork_no_exec(actor)

    def test_live_worker_missing_proc_stat_remains_an_explicit_failure(self):
        actor = Actor(201, ['python'])
        with patch.object(resource.Path, 'read_text', side_effect=FileNotFoundError), self.assertRaisesRegex(
                RuntimeError, 'Live Basic descendant'):
            resource._fork_no_exec(actor)


class BasicLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        (self.root/'hiercp_v1x').mkdir()
        code=self.root/'hiercp_v1x/basic_cp_ablation_resources.py'; code.write_text('real admitted code\n')
        self.bank=self.root/'bank.json'; self.bank.write_text('{"source_count":642}\n')
        self.lifetime=self.root/'allocation.json'
        self.lifetime.write_text(json.dumps(dict(format='v25_owned_GPU_allocation_v1', UID=41833,
            owner='aicompetition06', node='ece-a6gpu6', gpu=6, job_id='129664.ECE-util1', expires_at_unix=5000)))
        self.controller=Actor(100, ['python', 'controller'])
        self.stage=Actor(200, ['python', 'native'])
        self.worker=Actor(201, ['python', 'worker'])
        self.controller.descendants=[self.stage, self.worker]
        self.actors={x.pid:x for x in (self.controller, self.stage, self.worker)}
        self.wall=1000
        self.monotonic=500
        self.admission=dict(controller_cgroup='11:memory:/own\n',
            memory_limits=[dict(path='/sys/fs/cgroup/memory/own/memory.limit_in_bytes', limit_bytes=1024*2**30)])
        self.query_calls=[]
        self.apps=''
        self.free_gpu=45622
        self.gpu_total=46068
        self.job=dict(Job_Owner='aicompetition06@eceutil1', job_state='R',
            exec_host='ece-a6gpu6/0*6', stime=1000, Resource_List=dict(walltime='01:06:40', ngpus=1))
        self.host=SimpleNamespace(total=1024*2**30, available=900*2**30)
        self.storage=SimpleNamespace(free=20*2**30)
        value=request(); value.update(root=str(self.root), native_root=str(self.root), bank=str(self.bank),
            bank_sha256=resource.sha(self.bank), code_sha256={'hiercp_v1x/basic_cp_ablation_resources.py':resource.sha(code)},
            host_lifetime=dict(path=str(self.lifetime), sha256=resource.sha(self.lifetime), expires_at_unix=5000))
        self.value=value
        self.start(patch.object(resource, 'ROOT', self.root))
        self.start(patch.object(resource, 'validate_request', side_effect=lambda x:x))
        self.start(patch.object(resource, '_owned', side_effect=lambda p, **kwargs:Path(p)))
        self.start(patch.object(resource.psutil, 'Process', side_effect=self.process))
        self.start(patch.object(resource, '_fork_no_exec', side_effect=lambda p:p.fork_no_exec))
        self.start(patch.object(resource.os, 'getuid', return_value=41833, create=True))
        self.start(patch.object(resource.socket, 'gethostname', return_value='ece-a6gpu6'))
        self.start(patch.object(resource.time, 'time', side_effect=lambda:self.wall))
        self.start(patch.object(resource.time, 'monotonic', side_effect=lambda:self.monotonic))
        self.start(patch.object(resource.time, 'sleep', return_value=None))
        self.start(patch.object(resource, 'capture_cgroup_admission', side_effect=lambda pid:copy.deepcopy(self.admission)))
        self.start(patch.object(resource, '_query', side_effect=self.query))
        self.start(patch.object(resource.psutil, 'virtual_memory', side_effect=lambda:self.host))
        self.start(patch.object(resource.shutil, 'disk_usage', side_effect=lambda p:self.storage))
        self.start(patch.dict(resource.os.environ, {'CUDA_VISIBLE_DEVICES':'1'}))
        self.lease=resource.BasicLease(value)

    def start(self, patcher):
        value=patcher.start(); self.addCleanup(patcher.stop); return value

    def process(self, pid=None):
        selected=self.stage.pid if pid is None else pid
        if selected not in self.actors:
            raise resource.psutil.NoSuchProcess(selected)
        return self.actors[selected]

    def query(self, argv):
        self.query_calls.append(argv)
        if argv[0]=='/opt/pbs/bin/qstat':
            return json.dumps({'Jobs':{'129664.ECE-util1':self.job}})
        if '--query-compute-apps=pid,gpu_uuid' in argv:
            return self.apps
        return '1, GPU-exact, NVIDIA RTX A6000, '+str(self.gpu_total)+', '+str(self.free_gpu)+', [N/A]'

    def test_spawned_stage_and_worker_are_accepted_actual_descendants(self):
        result=self.lease.verify(idle=True)
        self.assertEqual(result['own_process_tree_RSS_bytes'], 3*2**20)
        self.assertFalse(result['PBS_GPU_owned_reservation'])
        self.assertEqual(result['host_lifetime_job_id'], '129664.ECE-util1')

    def test_live_cuda_can_only_belong_to_own_tree_after_initial_idle(self):
        self.lease.verify(idle=True)
        self.apps='200, GPU-exact\n999, GPU-GPU5'
        self.free_gpu=20000
        result=self.lease.verify(force=True)
        self.assertEqual(result['own_GPU_compute_PIDs'], [200])
        self.apps+='\n777, GPU-exact'
        with self.assertRaisesRegex(RuntimeError, 'non-admitted'):
            self.lease.verify(force=True)

    def test_idle_admission_rejects_even_own_gpu_actor(self):
        self.apps='200, GPU-exact'
        with self.assertRaisesRegex(RuntimeError, 'occupied'):
            self.lease.verify(idle=True)

    def test_fresh_child_lease_accepts_own_existing_cuda_unless_idle_requested(self):
        self.apps='200, GPU-exact'
        self.free_gpu=20000
        result=self.lease.verify()
        self.assertEqual(result['own_GPU_compute_PIDs'], [200])
        with self.assertRaisesRegex(RuntimeError, 'initial free'):
            self.lease.verify(idle=True)

    def test_empty_gpu_or_mig_identity_change_is_not_an_alternate_device(self):
        self.gpu_total=49140
        with self.assertRaisesRegex(RuntimeError, 'UUID/MIG'):
            self.lease.verify(force=True)

    def test_exact_stage_must_be_in_controller_tree(self):
        self.controller.descendants=[self.worker]
        with self.assertRaisesRegex(RuntimeError, 'outside'):
            self.lease.verify(force=True)

    def test_changed_controller_pid_birth_or_argv_is_rejected(self):
        self.controller.birth+=1
        with self.assertRaisesRegex(RuntimeError, 'PID/birth'):
            self.lease.verify(force=True)

    def test_changed_descendant_uid_or_cpu_is_rejected(self):
        self.worker.uid=0
        with self.assertRaisesRegex(RuntimeError, 'foreign'):
            self.lease.verify(force=True)
        self.worker.uid=41833; self.worker.cpus=[16]
        with self.assertRaisesRegex(RuntimeError, 'six-CPU'):
            self.lease.verify(force=True)

    def test_genuinely_zombie_worker_is_removed_without_fake_live_identity(self):
        self.lease.verify(force=True)
        self.worker.state=resource.psutil.STATUS_ZOMBIE; self.worker.command=[]
        result=self.lease.verify(force=True)
        self.assertEqual(result['own_process_tree_RSS_bytes'], 2*2**20)

    def test_unknown_empty_live_argv_remains_failure(self):
        self.worker.command=[]
        with self.assertRaisesRegex(RuntimeError, 'empty argv'):
            self.lease.verify(force=True)

    def provisional_fork(self):
        self.worker.parent_actor = self.stage
        self.worker.command = list(self.stage.command)
        self.worker.fork_no_exec = True
        self.lease.verify(force=True)
        self.assertIn(self.worker.pid, self.lease._fork_pending)

    def test_first_exec_replaces_only_the_exact_provisional_parent_argv(self):
        self.provisional_fork()
        self.worker.command = ['python', '-c', 'from multiprocessing.spawn import spawn_main; spawn_main()']
        self.worker.fork_no_exec = False
        self.lease.verify(force=True)
        self.assertNotIn(self.worker.pid, self.lease._fork_pending)
        self.assertEqual(self.lease._managed[self.worker.pid]['command'], self.worker.command)
        self.worker.command = ['python', 'unrequested_second_exec']
        with self.assertRaisesRegex(RuntimeError, 'live argv changed'):
            self.lease.verify(force=True)

    def test_fork_only_dataloader_worker_keeps_its_exact_inherited_identity(self):
        self.provisional_fork()
        for _ in range(4):
            result = self.lease.verify(force=True)
            self.assertEqual(result['own_process_tree_RSS_bytes'], 3*2**20)
        self.worker.command = ['python', 'unexpected_inherited_argv']
        with self.assertRaisesRegex(RuntimeError, 'inherited parent argv'):
            self.lease.verify(force=True)

    def test_first_exec_cannot_mask_reused_pid_foreign_uid_or_parent_change(self):
        self.provisional_fork()
        self.worker.fork_no_exec = False
        self.worker.command = ['python', 'spawned_worker']
        original_birth = self.worker.birth
        self.worker.birth += 1
        with self.assertRaisesRegex(RuntimeError, 'PID/birth/UID'):
            self.lease.verify(force=True)
        self.worker.birth = original_birth
        self.worker.uid = 0
        with self.assertRaisesRegex(RuntimeError, 'foreign'):
            self.lease.verify(force=True)
        self.worker.uid = 41833
        self.worker.parent_actor = self.controller
        with self.assertRaisesRegex(RuntimeError, 'parent lineage'):
            self.lease.verify(force=True)

    def test_fork_flag_is_not_a_generic_permission_to_change_a_finalized_command(self):
        self.lease.verify(force=True)
        self.worker.parent_actor = self.stage
        self.worker.command = list(self.stage.command)
        self.worker.fork_no_exec = True
        with self.assertRaisesRegex(RuntimeError, 'regressed'):
            self.lease.verify(force=True)

    def test_first_exec_race_is_reobserved_before_binding_any_argv(self):
        self.worker.parent_actor = self.stage
        self.worker.command = list(self.stage.command)
        self.worker.fork_no_exec = True
        readings = iter([True, False, False, False, False])
        def observe(actor):
            if actor is self.worker:
                result = next(readings)
                if not result:
                    self.worker.fork_no_exec = False
                    self.worker.command = ['python', 'stable_exec_worker']
                return result
            return False
        with patch.object(resource, '_fork_no_exec', side_effect=observe):
            self.lease.verify(force=True)
        self.assertEqual(self.lease._managed[self.worker.pid]['command'], ['python', 'stable_exec_worker'])

    def test_late_first_exec_during_final_consistency_check_is_bounded_and_reobserved(self):
        self.worker.parent_actor = self.stage
        self.worker.command = list(self.stage.command)
        self.worker.fork_no_exec = True
        readings = iter([True, True, False, False, False, False])
        def observe(actor):
            if actor is self.worker:
                result = next(readings)
                if not result:
                    self.worker.fork_no_exec = False
                    self.worker.command = ['python', 'late_stable_exec_worker']
                return result
            return False
        with patch.object(resource, '_fork_no_exec', side_effect=observe):
            self.lease.verify(force=True)
        self.assertEqual(self.lease._managed[self.worker.pid]['command'], ['python', 'late_stable_exec_worker'])

    def test_pbs_nonrunning_foreign_owner_othernode_or_extension_is_rejected(self):
        originals=copy.deepcopy(self.job)
        for key, value in [('job_state','Q'), ('Job_Owner','other@node'),
                ('exec_host','othernode/0*6'), ('stime',1001.5)]:
            self.job=copy.deepcopy(originals); self.job[key]=value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.lease.verify(force=True)

    def test_monotonic_expiry_blocks_wall_clock_rollback(self):
        self.wall=500; self.monotonic=4350
        with self.assertRaisesRegex(RuntimeError, '180seconds'):
            self.lease.verify()

    def test_resource_and_sha_polls_are_cached_but_metadata_each_batch(self):
        with patch.object(resource, 'sha', wraps=resource.sha) as hashing:
            self.lease.verify(force=True)
            queries=len(self.query_calls)
            for _ in range(20): self.lease.verify()
            self.assertEqual(len(self.query_calls), queries)
            self.assertEqual(hashing.call_count, 0)
            self.monotonic+=5
            self.lease.verify()
            self.assertGreater(len(self.query_calls), queries)
            self.bank.write_text('{"source_count":641}\n')
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                self.lease.verify()

    def test_memory_and_storage_are_checked_between_full_resource_polls(self):
        self.lease.verify(force=True)
        self.worker.rss=97*2**30
        with self.assertRaisesRegex(MemoryError, '96GiB'):
            self.lease.verify()
        self.worker.rss=2**20; self.storage.free=9*2**30
        with self.assertRaisesRegex(OSError, '10GiB'):
            self.lease.verify()

    def test_small_actual_cgroup_and_insufficient_host_ram_are_not_hidden(self):
        self.admission['memory_limits'][0]['limit_bytes']=128*2**30
        with self.assertRaisesRegex(MemoryError, 'four96GiB'):
            self.lease.verify(force=True)
        self.admission['memory_limits'][0]['limit_bytes']=1024*2**30
        self.lease._controller_cgroup=None
        self.host.available=100*2**30
        with self.assertRaisesRegex(MemoryError, 'aggregate416GiB'):
            self.lease.verify(force=True)

    def test_original_bank_has_no_hash_fallback_on_missing_file(self):
        self.bank.rename(self.root/'removed-bank.json')
        with self.assertRaises(FileNotFoundError):
            self.lease.verify()

    def test_original_request_and_original_lifetime_cannot_change(self):
        self.lease.request['gpu']=4
        with self.assertRaisesRegex(ValueError, 'in memory'):
            self.lease.verify()
        self.lease.request['gpu']=1
        self.lifetime.write_text(self.lifetime.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'metadata changed'):
            self.lease.verify()

    def frozen_admission_request(self):
        original=self.root/'old_code'
        source=original/'hiercp_v1x/v24_readonly_native_storage.py'
        source.parent.mkdir(parents=True)
        source.write_text('exact original adapter\n')
        scientific={}
        for relative in resource.FROZEN_SCIENCE_FILES:
            p=original/relative;p.parent.mkdir(exist_ok=True,parents=True)
            p.write_text('exact original source '+relative+'\n')
            scientific[relative]=resource.sha(p)
        admission=self.root/'original_admission.json'
        admission.write_text(json.dumps(dict(format='v24_native_immutable_full_cache_storage_v1',
            model_data_scale_preserved=True,debug=False,source_arrays_written=False,
            adapter_source=dict(path=str(source),sha256=resource.sha(source)),
            source_bank=dict(scientific_source_files_sha256=scientific))))
        request=dict(self.value,readonly_admission=str(admission),readonly_admission_sha256=resource.sha(admission))
        return request,admission,source,original,scientific

    def test_all_old_admitted_scientific_sources_remain_pinned_after_fresh_code_differs(self):
        request,admission,adapter,original,scientific=self.frozen_admission_request()
        lease=resource.BasicLease(request)
        lease.verify(force=True)
        source=original/'tools/online_raw_bank_preparation.py'
        source.write_text(source.read_text()+'# original source changed\n')
        with self.assertRaisesRegex(ValueError,'metadata changed'):
            lease.verify()

    def test_frozen_adapter_wrong_digest_or_missing_scientific_source_is_rejected(self):
        request,admission,adapter,original,scientific=self.frozen_admission_request()
        adapter.write_text('unknown changed adapter\n')
        with self.assertRaisesRegex(ValueError,'admitted Basic source'):
            resource.BasicLease(request)
        adapter.write_text('exact original adapter\n')
        source=original/'tools/online_raw_bank_preparation.py'
        source.rename(source.with_suffix('.missing'))
        with self.assertRaises(FileNotFoundError):
            resource.BasicLease(request)

    def test_historical_cache_admission_sha_is_verified_before_adapter_sources(self):
        request,admission,adapter,original,scientific=self.frozen_admission_request()
        admission.write_text(admission.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'admitted Basic source'):
            resource.BasicLease(request)


if __name__=='__main__': unittest.main()
