"""CPU-only DEBUG: measured384/800 admission and unchanged native scale."""
import copy
import json
from pathlib import Path,PurePosixPath
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_shared_bank_resources as shared_old
from hiercp_v1x import basic_cp_shared_bank_resources_v3 as shared
from hiercp_v1x import basic_cp_group_resources as group_old
from hiercp_v1x import basic_cp_group_resources_v2 as group_v2
from hiercp_v1x import basic_cp_group_resources_v3 as group
from hiercp_v1x import basic_cp_seven_resources as seven
from tests import test_basic_cp_shared_bank_resources as shared_fixture
from tests import test_basic_cp_group_resources as group_fixture


class V3RequestTests(unittest.TestCase):
    def request(self):
        value=shared_fixture.request();value.update(shared_resource_guard_version=3,shared_bank_RSS_limit_GiB=384)
        value['code_sha256'][shared.SHARED_GUARD_FILE]='a'*64
        return value
    def validate(self,value):
        with patch.object(seven,'Path',PurePosixPath),patch.object(shared,'Path',PurePosixPath):
            return shared.validate_shared_request(value)
    def test_explicit384_800_preserves_native96_8cpu_full_guard_dependency(self):
        value=self.request();before=copy.deepcopy(value);self.assertIs(self.validate(value),value)
        self.assertEqual(value,before);self.assertEqual(value['RSS_limit_bytes'],96*2**30)
        self.assertEqual(len(value['cpu_affinity']),8);self.assertEqual(len(shared.SHARED_CPU_AFFINITY),64)
        self.assertEqual(shared.RSS_LIMIT,384*2**30);self.assertEqual(shared.AGGREGATE_CAPACITY,800*2**30)
        self.assertEqual(group.GROUP_RSS_LIMIT,768*2**30)
    def test_v2_marker_or288_budget_cannot_silently_enter_v3(self):
        for key,replacement in [('shared_resource_guard_version',2),('shared_resource_guard_version',True),
                ('shared_bank_RSS_limit_GiB',288),('shared_bank_RSS_limit_GiB',384.)]:
            value=self.request();value[key]=replacement
            with self.subTest(key=key,replacement=replacement),self.assertRaises(ValueError):self.validate(value)
    def test_original_shared_and_new_shared_sha_required(self):
        for filename in(shared.SHARED_GUARD_FILE,shared.ORIGINAL_SHARED_GUARD_FILE):
            value=self.request();value['code_sha256'].pop(filename)
            with self.assertRaises(ValueError):self.validate(value)
    def test_runtime384_view_strictly_keeps_original_native_request_fields(self):
        value=self.request();value.update(cpu_affinity=list(shared.SHARED_CPU_AFFINITY),RSS_limit_bytes=384*2**30,
            native_CPU_affinity_preserved=list(range(40,48)),native_RSS_limit_bytes_preserved=96*2**30)
        with patch.object(shared,'Path',PurePosixPath),patch.object(seven,'Path',PurePosixPath):
            self.assertIs(shared.validate_request(value),value)
            value['native_RSS_limit_bytes_preserved']=384*2**30
            with self.assertRaisesRegex(ValueError,'native eight-CPU/96GiB'):shared.validate_request(value)
    def test_original_live_resource_constants_unchanged(self):
        self.assertEqual(shared_old.RSS_LIMIT,288*2**30)
        self.assertEqual(shared_old.AGGREGATE_CAPACITY,704*2**30)
        self.assertEqual(group_old.CANDIDATE_RSS_LIMIT,288*2**30)
        self.assertEqual(group_old.AGGREGATE_CAPACITY,704*2**30)
        self.assertEqual(seven.RSS_LIMIT,96*2**30)


class SharedV3Tests(unittest.TestCase):
    start=shared_fixture.SharedLeaseTests.start
    process=shared_fixture.SharedLeaseTests.process
    query=shared_fixture.SharedLeaseTests.query
    def setUp(self):
        shared_fixture.SharedLeaseTests.setUp(self)
        self.value.update(shared_resource_guard_version=3,shared_bank_RSS_limit_GiB=384)
        path=self.root/shared.SHARED_GUARD_FILE;path.write_text('exact v3 shared guard fixture\n')
        self.value['code_sha256'][shared.SHARED_GUARD_FILE]=shared.sha(path)
        self.request_path.write_text(json.dumps(self.value))
        for name in('ROOT','validate_request','_owned','_fork_no_exec','_kernel_exiting','capture_cgroup_admission','_query','_pbs_query','_pbs_private','_pbs_lock'):
            self.start(patch.object(shared,name,getattr(shared_old,name)))
        self.lease=shared.SharedBankLease(self.value)
    def test_measured320gib_passes384_and_native_request_stays96(self):
        self.worker.rss=320*2**30
        result=self.lease.verify(force=True)
        self.assertEqual(result['own_process_tree_RSS_limit_bytes'],384*2**30)
        self.assertEqual(result['aggregate_native_and_cache_capacity_bytes'],800*2**30)
        self.assertEqual(result['native_RSS_limit_bytes_preserved'],96*2**30)
        self.assertEqual(self.value['RSS_limit_bytes'],96*2**30)
    def test385gib_source_scope_is_still_a_real_failure(self):
        self.worker.rss=385*2**30
        with self.assertRaisesRegex(MemoryError,'384GiB'):self.lease.verify(force=True)
    def test_full_cgroup800_requirement_no704_fallback(self):
        self.admission['memory_limits'][0]['limit_bytes']=799*2**30
        with self.assertRaisesRegex(MemoryError,'shared384GiB'):self.lease.verify(force=True)
    def test_quick_retains_stat_identity_only_and_periodic_full(self):
        self.lease.verify(force=True)
        with patch.object(self.lease,'_inputs',wraps=self.lease._inputs)as inputs:
            for _ in range(10):self.lease.quick_verify()
            self.assertEqual(inputs.call_count,0)
            self.monotonic+=5;self.lease.quick_verify();self.assertGreater(inputs.call_count,0)
    def test_final_kernel_exiting_emptyargv_keeps_actual_source_worker_rss(self):
        self.worker.fork_no_exec=True;self.worker.parent=lambda:self.stage
        with patch.object(shared,'_fork_no_exec',side_effect=lambda p:getattr(p,'fork_no_exec',False)),patch.object(
                seven,'_kernel_exiting',return_value=True),patch.object(self.worker,'cmdline',side_effect=[list(self.stage.command),[]]):
            result=self.lease._descendant(self.worker)
        self.assertIs(result,self.worker)
        self.assertTrue(self.lease._terminal_observations[201]['RSS_still_accounted'])
    def test_no_exit_flag_is_still_not_an_admitted_live_unknown_argv(self):
        self.worker.fork_no_exec=True;self.worker.parent=lambda:self.stage
        with patch.object(shared,'_fork_no_exec',side_effect=lambda p:getattr(p,'fork_no_exec',False)),patch.object(
                seven,'_kernel_exiting',return_value=False),patch.object(self.worker,'cmdline',side_effect=[list(self.stage.command),[]]),self.assertRaisesRegex(
                RuntimeError,'changed during live observation'):
            self.lease._descendant(self.worker)


class GroupV3Tests(unittest.TestCase):
    start=group_fixture.GroupLeaseTests.start
    process=group_fixture.GroupLeaseTests.process
    query=group_fixture.GroupLeaseTests.query
    def setUp(self):
        group_fixture.GroupLeaseTests.setUp(self)
        checks={}
        for filename in(group_v2.GUARD_FILE,group.GUARD_FILE):
            path=self.root/filename;path.write_text('exact v3 group dependency fixture\n');checks[filename]=seven.sha(path)
        for gpu,value in self.requests.items():
            value.update(shared_resource_guard_version=3)
            value['code_sha256'].update(checks)
            if gpu==5:value['shared_bank_RSS_limit_GiB']=384
            path=Path(value['root'])/'request.json';path.write_text(json.dumps(value))
            self.document['registration'][str(gpu)]['request_sha256']=seven.sha(path)
        self.registry_path.write_text(json.dumps(self.document));self.value=self.requests[0]
        self.lease=group.BasicLease(self.value)
        self.start(patch.dict(group.BasicLease._descendant.__globals__,{
            '_fork_no_exec':lambda p:getattr(p,'fork_no_exec',False),
            '_kernel_exiting':lambda p:getattr(p,'kernel_exiting',False)}))
    def test_candidate320_gib_and_group800_admit_without_native_budget_increase(self):
        self.actors[305].rss=320*2**30
        result=self.lease.verify(force=True)
        self.assertEqual(result['candidate_subpool_RSS_limit_bytes'],384*2**30)
        self.assertEqual(result['group_process_tree_RSS_limit_bytes'],768*2**30)
        self.assertEqual(result['aggregate_native_and_cache_capacity_bytes'],800*2**30)
        self.assertEqual(result['own_process_tree_RSS_limit_bytes'],96*2**30)
    def test_candidate384_budget_still_includes_native_gpu3_and4(self):
        self.actors[305].rss=340*2**30;self.actors[303].rss=30*2**30;self.actors[304].rss=30*2**30
        with self.assertRaisesRegex(MemoryError,'shared384GiB'):self.lease.verify(force=True)
    def test_native97gib_not_borrowed_from_shared_budget(self):
        self.actors[303].rss=97*2**30
        with self.assertRaisesRegex(MemoryError,'96GiB'):self.lease.verify(force=True)
    def test_actual799gib_cgroup_is_rejected_not_reinterpreted_as800(self):
        self.admission['memory_limits'][0]['limit_bytes']=799*2**30
        with self.assertRaisesRegex(MemoryError,'800GiB'):self.lease.verify(force=True)
    def test_group_marker2_cannot_enter_new800gib_lease(self):
        value=copy.deepcopy(self.value);value['shared_resource_guard_version']=2
        with self.assertRaisesRegex(ValueError,'v3'):group.BasicLease(value)


if __name__=='__main__':unittest.main()
