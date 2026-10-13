"""CPU-only DEBUG: immutable group registry and deduplicated RAM admission."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_group_resources as group
from hiercp_v1x import basic_cp_seven_resources as resource
from tests import test_basic_cp_seven_resources as fixture


class GroupLeaseTests(unittest.TestCase):
    start=fixture.SevenLeaseTests.start
    process=fixture.SevenLeaseTests.process
    query=fixture.SevenLeaseTests.query

    def setUp(self):
        fixture.SevenLeaseTests.setUp(self)
        source=self.root/group.GUARD_FILE;source.write_text('exact fresh group guard\n')
        self.value['code_sha256'][group.GUARD_FILE]=resource.sha(source)
        self.registry_path=self.root/'shared_registry.json'
        self.value.update(arm='multi5',shared_resource_registry=str(self.registry_path))
        self.requests={};registration={}
        for gpu in range(7):
            folder=self.root if gpu==0 else self.root/('GPU'+str(gpu))
            folder.mkdir(exist_ok=True)
            controller=self.controller if gpu==0 else fixture.Actor(100+gpu,['python','controller_GPU'+str(gpu)])
            controller.cpus=list(resource.CPUS[gpu]);self.actors[controller.pid]=controller
            if gpu!=0:
                stage=fixture.Actor(300+gpu,['python','stage_GPU'+str(gpu)]);stage.cpus=list(resource.CPUS[gpu])
                controller.descendants=[stage];self.actors[stage.pid]=stage
            value=copy.deepcopy(self.value)
            value.update(root=str(folder),native_root=str(folder),gpu=gpu,arm=group.ARMS[gpu],
                cpu_affinity=list(resource.CPUS[gpu]),
                controller_identity=dict(pid=controller.pid,create_time=controller.birth,UID=controller.uid,command=controller.command))
            if gpu==5:
                value.update(shared_bank_CPU_affinity=list(group.SHARED_CPU_AFFINITY),shared_bank_RSS_limit_GiB=288)
                controller.cpus=list(group.SHARED_CPU_AFFINITY)
                self.actors[305].cpus=list(group.SHARED_CPU_AFFINITY)
            path=folder/'request.json';path.write_text(json.dumps(value))
            self.requests[gpu]=value
            registration[str(gpu)]=dict(root=str(folder),arm=value['arm'],
                controller_identity=value['controller_identity'],request_sha256=resource.sha(path))
        self.document=dict(format=group.FORMAT,registration=registration)
        self.registry_path.write_text(json.dumps(self.document))
        self.value=self.requests[0]
        self.lease=group.BasicLease(self.value)

    def rewrite(self):self.registry_path.write_text(json.dumps(self.document))

    def test_available_plus_all_group_rss_admits_without_counting_only_own(self):
        for gpu in(0,1,2,6):
            self.actors[100+gpu].rss=1*2**30
            self.actors[200 if gpu==0 else 300+gpu].rss=39*2**30
        self.worker.rss=0
        self.actors[105].rss=4*2**30;self.actors[305].rss=146*2**30
        self.host.available=500*2**30
        result=self.lease.verify(force=True)
        self.assertLess(result['host_available_RAM_bytes']+result['own_process_tree_RSS_bytes'],704*2**30)
        self.assertGreaterEqual(result['host_available_RAM_bytes']+result['group_process_tree_RSS_bytes'],704*2**30)
        self.assertEqual(result['own_process_tree_RSS_limit_bytes'],96*2**30)
        self.assertEqual(result['candidate_subpool_RSS_limit_bytes'],288*2**30)
        self.assertFalse(result['processes_signaled'])

    def test_same_pid_is_counted_once_in_group_and_candidate_pool(self):
        shared=self.stage;shared.rss=32*2**30
        def tree(gpu):
            return {shared.pid:shared},dict(status='LIVE',shared_CPU_bank_stage=False)
        with patch.object(self.lease,'_group_tree',side_effect=tree):
            result=self.lease.verify(force=True)
        self.assertEqual(result['group_process_tree_RSS_bytes'],32*2**30)
        self.assertEqual(result['candidate_subpool_RSS_bytes'],32*2**30)
        self.assertEqual(result['group_RSS_deduplicated_PIDs'],[shared.pid])

    def test_registry_publication_wait_is_bounded_and_does_not_extend_pbs_lifetime(self):
        content=self.registry_path.read_text();self.registry_path.unlink()
        def published(seconds):self.registry_path.write_text(content)
        with patch.object(group.time,'sleep',side_effect=published)as sleeper:
            lease=group.BasicLease(self.value)
        self.assertEqual(lease.lifetime['expires_at_unix'],5000)
        sleeper.assert_called_once_with(.1)

    def test_own_native96gib_limit_is_not_increased_by_group_fix(self):
        self.stage.rss=97*2**30
        with self.assertRaisesRegex(MemoryError,'96GiB'):self.lease.verify(force=True)

    def test_other_native96gib_limits_are_not_hidden(self):
        self.actors[301].rss=97*2**30
        with self.assertRaisesRegex(MemoryError,'GPU1.*96GiB'):self.lease.verify(force=True)
        self.actors[301].rss=2**20;self.actors[303].rss=97*2**30
        with self.assertRaisesRegex(MemoryError,'GPU3.*96GiB'):self.lease.verify(force=True)

    def test_candidate_subpool288gib_includes_bank_and_two_native_candidates(self):
        self.actors[303].rss=80*2**30;self.actors[304].rss=80*2**30
        self.actors[305].rss=140*2**30
        with self.assertRaisesRegex(MemoryError,'shared288GiB'):self.lease.verify(force=True)

    def test_foreign_uid_pid_birth_or_live_controller_argv_is_never_admitted(self):
        self.actors[101].birth+=1
        with self.assertRaisesRegex(RuntimeError,'PID/birth/UID'):self.lease.verify(force=True)
        self.actors[101].birth-=1;self.actors[301].uid=999
        with self.assertRaisesRegex(RuntimeError,'foreign'):self.lease.verify(force=True)
        self.actors[301].uid=41833;self.actors[101].command=['python','replacement']
        with self.assertRaisesRegex(RuntimeError,'PID/birth/UID/argv'):self.lease.verify(force=True)

    def test_live_other_arm_wrong_cpu_affinity_is_not_a_resource_fallback(self):
        self.actors[304].cpus=[24]
        with self.assertRaisesRegex(RuntimeError,'CPU affinity'):self.lease.verify(force=True)

    def test_gpu5_shared64cpu_is_admitted_only_with_explicit_budget_and_cpu_only(self):
        result=self.lease.verify(force=True)
        self.assertTrue(result['group_controller_observations']['5']['shared_CPU_bank_stage'])
        self.apps='305, GPU-other'
        with self.assertRaisesRegex(RuntimeError,'forbidden CUDA'):self.lease.verify(force=True)

    def test_gpu5_after_bank_returns_to_native8cpu96gib(self):
        self.actors[105].cpus=list(resource.CPUS[5]);self.actors[305].cpus=list(resource.CPUS[5])
        result=self.lease.verify(force=True)
        self.assertFalse(result['group_controller_observations']['5']['shared_CPU_bank_stage'])
        self.actors[305].rss=97*2**30
        with self.assertRaisesRegex(MemoryError,'GPU5.*96GiB'):self.lease.verify(force=True)

    def test_terminal_controller_does_not_silently_drop_still_live_exact_children(self):
        self.lease.verify(force=True);self.actors[301].rss=12*2**30
        self.actors.pop(101)
        result=self.lease.verify(force=True)
        self.assertEqual(result['group_controller_observations']['1']['status'],'NoSuchProcess')
        self.assertIn(301,result['group_RSS_deduplicated_PIDs'])
        self.assertEqual(result['group_RSS_by_GPU_bytes']['1'],12*2**30)
        self.actors.pop(301)
        result=self.lease.verify(force=True)
        self.assertEqual(result['group_RSS_by_GPU_bytes']['1'],0)

    def test_controller_natural_exit_between_identity_and_children_read_is_reobserved(self):
        self.lease.verify(force=True);controller=self.actors[101]
        def ended(recursive=False):
            self.actors.pop(101,None)
            raise resource.psutil.NoSuchProcess(101)
        with patch.object(controller,'children',side_effect=ended):
            result=self.lease.verify(force=True)
        self.assertEqual(result['group_controller_observations']['1']['status'],'NoSuchProcess')
        self.assertIn(301,result['group_RSS_deduplicated_PIDs'])

    def test_terminal_controller_pid_reuse_is_rejected(self):
        self.lease.verify(force=True)
        self.actors[101]=fixture.Actor(101,['python','other'],birth=999)
        with self.assertRaisesRegex(RuntimeError,'reused'):self.lease.verify(force=True)

    def test_registry_and_every_request_file_metadata_are_immutable(self):
        self.lease.verify(force=True)
        path=Path(self.requests[4]['root'])/'request.json';path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'metadata changed'):self.lease.verify(force=True)

    def test_modified_registry_bytes_and_in_memory_rows_fail(self):
        self.lease.registration['1']['arm']='changed'
        with self.assertRaisesRegex(ValueError,'in memory'):self.lease.verify(force=True)
        self.lease.registration['1']['arm']='multi7'
        self.registry_path.write_text(self.registry_path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'metadata changed'):self.lease.verify(force=True)

    def test_each_request_sha_and_all_seven_rows_are_required_at_initialization(self):
        self.document['registration']['3']['request_sha256']='f'*64;self.rewrite()
        with self.assertRaisesRegex(ValueError,'admitted Basic source'):group.BasicLease(self.value)
        self.document['registration'].pop('3');self.rewrite()
        with self.assertRaisesRegex(ValueError,'All seven'):group.BasicLease(self.value)

    def test_missing_group_guard_or_duplicate_controller_registration_rejected(self):
        value=copy.deepcopy(self.value);value['code_sha256'].pop(group.GUARD_FILE)
        with self.assertRaisesRegex(ValueError,'group guard'):group.BasicLease(value)
        row=self.requests[1];row['controller_identity']=copy.deepcopy(self.value['controller_identity'])
        path=Path(row['root'])/'request.json';path.write_text(json.dumps(row))
        self.document['registration']['1'].update(controller_identity=row['controller_identity'],request_sha256=resource.sha(path))
        self.rewrite()
        with self.assertRaisesRegex(ValueError,'duplicate registry'):group.BasicLease(self.value)

    def test_actual_cgroup_host_ram_and_storage_floor_remain_full704gib(self):
        self.admission['memory_limits'][0]['limit_bytes']=703*2**30
        with self.assertRaisesRegex(MemoryError,'704GiB'):self.lease.verify(force=True)
        self.admission['memory_limits'][0]['limit_bytes']=None;self.lease._controller_cgroup=None
        self.host.available=100*2**30
        with self.assertRaisesRegex(MemoryError,'deduplicated'):self.lease.verify(force=True)
        self.host.available=937*2**30;self.storage.free=9*2**30
        with self.assertRaisesRegex(OSError,'10GiB'):self.lease.verify(force=True)

    def test_same_owned_host_lifetime_required_for_every_registered_controller(self):
        row=self.requests[4];row['host_lifetime']['expires_at_unix']+=1
        path=Path(row['root'])/'request.json';path.write_text(json.dumps(row))
        self.document['registration']['4']['request_sha256']=resource.sha(path);self.rewrite()
        with self.assertRaisesRegex(ValueError,'same immutable'):group.BasicLease(self.value)

    def test_pbs_blank_timestamp_and_real_expiry_are_preserved(self):
        result=self.lease.verify(force=True)
        self.assertEqual(result['host_lifetime_PBS_observation_age_seconds'],0)
        self.wall=4821;self.monotonic=4321
        with self.assertRaisesRegex(RuntimeError,'180seconds'):self.lease.verify(force=True)


if __name__=='__main__':unittest.main()
