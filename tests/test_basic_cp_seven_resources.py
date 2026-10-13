"""Seven-arm resource checks with actual-interface mocked OS observations."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_ablation_resources as old
from hiercp_v1x import basic_cp_seven_resources as resource


def request(gpu=0):
    return dict(format=resource.FORMAT, uid=41833, root='/var/tmp/own_seven/arm',
        native_root='/dev/shm/own_seven/arm', gpu=gpu, gpu_uuid='GPU-exact',
        cpu_affinity=list(resource.CPUS[gpu]), GPU_memory_total_MiB=46068,
        controller_identity=dict(pid=100, create_time=700, UID=41833, command=['python', 'controller']),
        code_sha256={resource.ORIGINAL_GUARD_FILE:'a'*64, resource.SEVEN_GUARD_FILE:'b'*64},
        bank='/home/aicompetition06/bank/index.json', bank_sha256='c'*64,
        host_lifetime=dict(path='/var/tmp/owned/allocation.json', sha256='d'*64, expires_at_unix=5000),
        runtime_reserve_bytes=10*2**30, RSS_limit_bytes=96*2**30)


class SevenRequestTests(unittest.TestCase):
    def validate(self, value):
        with patch.object(resource, 'Path', PurePosixPath):
            return resource.validate_request(value)

    def test_seven_gpu_disjoint_eight_core_map_full_capacity(self):
        self.assertEqual(set(resource.CPUS), set(range(7)))
        self.assertEqual(sum(resource.CPUS.values(), []), list(range(56)))
        self.assertEqual(resource.AGGREGATE_CAPACITY, (7*96+32)*2**30)
        for gpu in range(7):
            value=request(gpu)
            self.assertIs(self.validate(value), value)

    def test_original_guard_constants_and_function_globals_are_unchanged(self):
        self.assertEqual(set(old.CPUS), {1,4,6})
        self.assertEqual(old.AGGREGATE_CAPACITY, 416*2**30)
        self.assertIs(old.BasicLease.verify.__globals__, vars(old))
        self.assertIs(resource.BasicLease.verify.__globals__, vars(resource))
        self.assertIs(resource.BasicLease._host_lifetime.__globals__, vars(resource))
        self.assertIs(resource._pbs_lock.__wrapped__.__globals__, vars(resource))
        self.assertIs(old._pbs_lock.__wrapped__.__globals__, vars(old))

    def test_both_immutable_guards_are_mandatory(self):
        for filename in (resource.ORIGINAL_GUARD_FILE, resource.SEVEN_GUARD_FILE):
            value=request();value['code_sha256'].pop(filename)
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError,'SHA'):
                self.validate(value)

    def test_cannot_shrink_ram_cpus_gpu_storage_or_change_uid(self):
        for key, replacement in [('RSS_limit_bytes',64*2**30),('runtime_reserve_bytes',2**30),
                ('cpu_affinity',[0]),('GPU_memory_total_MiB',24000),('gpu',7),('uid',0)]:
            value=request();value[key]=replacement
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.validate(value)

    def test_old_three_arm_cpu_map_cannot_be_accepted_in_new_run(self):
        value=request(1);value['cpu_affinity']=old.CPUS[1]
        with self.assertRaises(ValueError):self.validate(value)

    def test_native_tmpfs_and_original_bank_source_sha_still_required(self):
        value=request();value['native_root']='/var/tmp/native'
        with self.assertRaisesRegex(ValueError,'tmpfs'):self.validate(value)
        value=request();value['source_bank']='/home/own/source.json'
        with self.assertRaisesRegex(ValueError,'together'):self.validate(value)


class Actor:
    def __init__(self, pid, command, *, birth=700, uid=41833):
        self.pid=pid;self.command=command;self.birth=birth;self.uid=uid
        self.cpus=list(range(8));self.rss=2**20;self.state='sleeping';self.descendants=[]
    def status(self):return self.state
    def create_time(self):return self.birth
    def uids(self):return SimpleNamespace(real=self.uid)
    def cmdline(self):return self.command
    def cpu_affinity(self):return self.cpus
    def memory_info(self):return SimpleNamespace(rss=self.rss)
    def children(self,recursive=False):return self.descendants
    def is_running(self):return self.state!='gone'


class SevenLeaseTests(unittest.TestCase):
    def start(self, patcher):
        value=patcher.start();self.addCleanup(patcher.stop);return value

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup);self.root=Path(self.temporary.name)
        (self.root/'hiercp_v1x').mkdir()
        self.code={}
        for relative in (resource.ORIGINAL_GUARD_FILE,resource.SEVEN_GUARD_FILE):
            path=self.root/relative;path.write_text('exact admitted '+relative+'\n')
            self.code[relative]=resource.sha(path)
        self.bank=self.root/'bank.json';self.bank.write_text('{"source_count":642}\n')
        self.lifetime=self.root/'allocation.json'
        self.lifetime.write_text(json.dumps(dict(format='v25_owned_GPU_allocation_v1',UID=41833,
            owner='aicompetition06',node='ece-a6gpu6',gpu=6,job_id='129776.ECE-util1',expires_at_unix=5000)))
        self.controller=Actor(100,['python','controller'])
        self.stage=Actor(200,['python','native'])
        self.worker=Actor(201,['python','shared_candidate_CPU_worker'])
        self.controller.descendants=[self.stage,self.worker]
        self.actors={a.pid:a for a in (self.controller,self.stage,self.worker)}
        self.wall=1000;self.monotonic=500;self.apps='';self.free_gpu=45622;self.gpu_total=46068
        self.query_calls=[];self.timestamp={'timestamp':''}
        self.job=dict(Job_Owner='aicompetition06@eceutil1',job_state='R',exec_host='ece-a6gpu6/0*6',
            stime=1000,Resource_List=dict(walltime='01:06:40',ngpus=1))
        self.admission=dict(controller_cgroup='0::/ssh.service\n',memory_limits=[dict(path='/sys/fs/cgroup/memory.max',limit_bytes=None)])
        self.host=SimpleNamespace(total=1007*2**30,available=937*2**30)
        self.storage=SimpleNamespace(free=449*2**30)
        self.value=request();self.value.update(root=str(self.root),native_root=str(self.root),
            bank=str(self.bank),bank_sha256=resource.sha(self.bank),code_sha256=self.code,
            host_lifetime=dict(path=str(self.lifetime),sha256=resource.sha(self.lifetime),expires_at_unix=5000))
        self.start(patch.object(resource,'ROOT',self.root))
        self.start(patch.object(resource,'validate_request',side_effect=lambda x:x))
        self.start(patch.object(resource,'_owned',side_effect=lambda p,**kwargs:Path(p)))
        self.start(patch.object(resource.psutil,'Process',side_effect=self.process))
        self.start(patch.object(resource,'_fork_no_exec',return_value=False))
        self.start(patch.object(resource,'_kernel_exiting',return_value=False))
        self.start(patch.object(resource.os,'getuid',return_value=41833,create=True))
        self.start(patch.object(resource.socket,'gethostname',return_value='ece-a6gpu6'))
        self.start(patch.object(resource.time,'time',side_effect=lambda:self.wall))
        self.start(patch.object(resource.time,'monotonic',side_effect=lambda:self.monotonic))
        self.sleep=self.start(patch.object(resource.time,'sleep',return_value=None))
        self.start(patch.object(resource,'capture_cgroup_admission',side_effect=lambda pid:copy.deepcopy(self.admission)))
        self.start(patch.object(resource,'_query',side_effect=self.query))
        self.start(patch.object(resource,'_pbs_query',side_effect=self.query))
        self.start(patch.object(resource,'_pbs_private',side_effect=lambda p,**kwargs:Path(p)))
        @contextmanager
        def lock(folder):yield True
        self.start(patch.object(resource,'_pbs_lock',side_effect=lock))
        self.start(patch.object(resource.os,'O_NOFOLLOW',0,create=True))
        self.start(patch.object(resource.psutil,'virtual_memory',side_effect=lambda:self.host))
        self.start(patch.object(resource.shutil,'disk_usage',side_effect=lambda p:self.storage))
        self.start(patch.dict(resource.os.environ,{'CUDA_VISIBLE_DEVICES':'0'}))
        self.lease=resource.BasicLease(self.value)

    def process(self,pid=None):
        selected=200 if pid is None else pid
        if selected not in self.actors:raise resource.psutil.NoSuchProcess(selected)
        return self.actors[selected]

    def query(self,argv,**kwargs):
        self.query_calls.append(argv)
        if argv[0]=='/opt/pbs/bin/qstat':
            return json.dumps(dict(self.timestamp,Jobs={'129776.ECE-util1':self.job}))
        if '--query-compute-apps=pid,gpu_uuid' in argv:return self.apps
        return str(self.value['gpu'])+', GPU-exact, NVIDIA RTX A6000, '+str(self.gpu_total)+', '+str(self.free_gpu)+', [N/A]'

    def test_all_seven_devices_execute_full_checks_with_own_eight_cpus(self):
        for gpu in range(7):
            self.value.update(gpu=gpu,cpu_affinity=list(resource.CPUS[gpu]))
            for actor in self.actors.values():actor.cpus=list(resource.CPUS[gpu])
            with patch.dict(resource.os.environ,{'CUDA_VISIBLE_DEVICES':str(gpu)}):
                lease=resource.BasicLease(self.value);result=lease.verify(force=True,idle=True)
            self.assertEqual(result['aggregate_native_and_cache_capacity_bytes'],704*2**30)
            self.assertEqual(result['own_CPU_affinity'],resource.CPUS[gpu])
            self.assertFalse(result['PBS_GPU_owned_reservation'])

    def test_shared_cpu_bank_worker_is_exact_own_descendant_and_counted(self):
        result=self.lease.verify(idle=True)
        self.assertEqual(result['own_process_tree_RSS_bytes'],3*2**20)
        self.assertIn(self.worker.pid,self.lease._cached_tree)

    def test_seven_capacity_and_actual_cgroup_are_mandatory(self):
        self.admission['memory_limits'][0]['limit_bytes']=703*2**30
        with self.assertRaisesRegex(MemoryError,'seven96GiB'):self.lease.verify()
        self.admission['memory_limits'][0]['limit_bytes']=None;self.lease._controller_cgroup=None
        self.host.available=700*2**30
        with self.assertRaisesRegex(MemoryError,'704GiB'):self.lease.verify()

    def test_waiting_observation_reports_foreign_gpu_row_without_admitting_or_signaling(self):
        self.apps='777, GPU-exact\n999, GPU-other'
        result=self.lease.observe_waiting()
        self.assertFalse(result['GPU_available'])
        self.assertEqual(result['occupied_GPU_compute_actors'],[
            dict(pid=777,GPU_UUID='GPU-exact',admitted=False,processes_signaled=False)])
        self.assertFalse(result['processes_signaled'])
        self.assertIsNone(self.lease.last_verified)

    def test_waiting_requires_40gib_and_exact_gpu_uuid_total(self):
        self.free_gpu=39000
        self.assertFalse(self.lease.observe_waiting()['GPU_available'])
        self.gpu_total=49140
        with self.assertRaisesRegex(RuntimeError,'UUID/MIG'):self.lease.observe_waiting()

    def test_wait_until_available_polls_busy_then_full_idle_admission(self):
        self.apps='777, GPU-exact';waiting=[]
        def released(observation):waiting.append(observation);self.apps=''
        result=self.lease.wait_until_available(on_wait=released)
        self.assertEqual(len(waiting),1);self.sleep.assert_called_once_with(5)
        self.assertEqual(result['own_GPU_compute_PIDs'],[])
        self.assertIsNotNone(self.lease.last_verified)

    def test_invalid_wait_poll_is_not_zero_spin_or_unbounded_value(self):
        for interval in (0,-1,float('inf'),True,'5'):
            with self.subTest(interval=interval),self.assertRaises(ValueError):
                self.lease.wait_until_available(poll_seconds=interval)

    def test_waiting_checks_actor_birth_affinity_and_source_metadata(self):
        self.controller.birth+=1
        with self.assertRaisesRegex(RuntimeError,'PID/birth'):self.lease.observe_waiting()
        self.controller.birth-=1;self.worker.cpus=[0]
        with self.assertRaisesRegex(RuntimeError,'allocation'):self.lease.observe_waiting()
        self.worker.cpus=list(range(8));self.bank.write_text('changed\n')
        with self.assertRaisesRegex(ValueError,'metadata'):self.lease.observe_waiting()

    def test_live_foreign_cuda_actor_cannot_pass_actual_training_admission(self):
        self.apps='777, GPU-exact';self.actors[777]=Actor(777,['other_user','trainer'],uid=999)
        with self.assertRaisesRegex(RuntimeError,'non-admitted CUDA'):self.lease.verify()

    def test_foreign_worker_and_reused_pid_cannot_join_cpu_producer_tree(self):
        self.worker.uid=999
        with self.assertRaisesRegex(RuntimeError,'foreign'):self.lease.verify()
        self.worker.uid=41833;self.lease.verify();self.worker.birth+=1
        with self.assertRaisesRegex(RuntimeError,'PID/birth/UID'):self.lease.verify(force=True)

    def test_ram_storage_and_rss_preserve_original_full_contract(self):
        self.lease.verify();self.worker.rss=97*2**30
        with self.assertRaisesRegex(MemoryError,'96GiB'):self.lease.verify()
        self.worker.rss=2**20;self.storage.free=9*2**30
        with self.assertRaisesRegex(OSError,'10GiB'):self.lease.verify()

    def test_original_host_lifetime_pbs_blank_and_missing_clocks_are_inherited(self):
        for clock in ({'timestamp':''},{}):
            self.timestamp=clock
            (self.lease._pbs_shared_root/'observation.json').unlink(missing_ok=True)
            self.lease._pbs_verified_at=None
            result=self.lease.verify(force=True)
            self.assertEqual(result['host_lifetime_PBS_observation_age_seconds'],0)
            record=json.loads((self.lease._pbs_shared_root/'observation.json').read_bytes())
            self.assertEqual(record['server_timestamp'],clock.get('timestamp'))
            self.assertEqual(record['observed_at_unix'],1000)

    def test_finished_or_foreign_pbs_jobs_block_even_while_gpu_busy(self):
        self.apps='777, GPU-exact'
        for field, replacement in [('job_state','F'),('Job_Owner','other@host')]:
            original=self.job[field];self.job[field]=replacement
            (self.lease._pbs_shared_root/'observation.json').unlink(missing_ok=True)
            self.lease._pbs_verified_at=None
            with self.subTest(field=field),self.assertRaisesRegex(RuntimeError,'no longer proves'):
                self.lease.observe_waiting()
            self.job[field]=original

    def test_expiry_wait_never_extends_lifetime(self):
        self.wall=4821;self.monotonic=4321;self.apps='777, GPU-exact'
        with self.assertRaisesRegex(RuntimeError,'180seconds'):self.lease.observe_waiting()


if __name__=='__main__':unittest.main()
