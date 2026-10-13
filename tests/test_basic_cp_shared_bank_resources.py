"""CPU-only DEBUG resource proof; no server process or CUDA is launched."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_shared_bank_resources as resource
from hiercp_v1x import basic_cp_seven_resources as seven
from hiercp_v1x import basic_cp_ablation_resources as old
from tests import test_basic_cp_seven_resources as fixture


def request():
    value=fixture.request(5)
    value.update(shared_bank_CPU_affinity=list(resource.SHARED_CPU_AFFINITY),shared_bank_RSS_limit_GiB=288)
    value['code_sha256'][resource.SHARED_GUARD_FILE]='f'*64
    return value


class SharedRequestTests(unittest.TestCase):
    def validate(self,value):
        with patch.object(seven,'Path',PurePosixPath),patch.object(resource,'Path',PurePosixPath):
            return resource.validate_shared_request(value)

    def test_explicit288gib64cpu_keeps_native8cpu96gib_and704gib_total(self):
        value=request();before=copy.deepcopy(value)
        self.assertIs(self.validate(value),value)
        self.assertEqual(value,before)
        self.assertEqual(value['cpu_affinity'],list(range(40,48)))
        self.assertEqual(value['RSS_limit_bytes'],96*2**30)
        self.assertEqual(len(resource.SHARED_CPU_AFFINITY),64)
        self.assertEqual(resource.RSS_LIMIT+4*seven.RSS_LIMIT+32*2**30,704*2**30)

    def test_implicit_budget_other_gpu_or_wrong_shared_cpu_rejected(self):
        for key,replacement in [('shared_bank_CPU_affinity',[40]),('shared_bank_RSS_limit_GiB',96),
                ('shared_bank_RSS_limit_GiB',288.),('shared_bank_RSS_limit_GiB',True),('gpu',4)]:
            value=request();value[key]=replacement
            with self.subTest(key=key,replacement=replacement),self.assertRaises(ValueError):self.validate(value)

    def test_new_and_previous_guard_shas_are_all_required(self):
        for filename in (resource.SHARED_GUARD_FILE,seven.SEVEN_GUARD_FILE,seven.ORIGINAL_GUARD_FILE):
            value=request();value['code_sha256'].pop(filename)
            with self.subTest(filename=filename),self.assertRaises(ValueError):self.validate(value)

    def test_original_native_guard_and_globals_are_never_rebound(self):
        self.assertEqual(old.RSS_LIMIT,96*2**30)
        self.assertEqual(seven.RSS_LIMIT,96*2**30)
        self.assertEqual(seven.CPUS[5],list(range(40,48)))
        self.assertIs(old.BasicLease._host_lifetime.__globals__,vars(old))
        self.assertIs(seven.BasicLease._host_lifetime.__globals__,vars(seven))
        self.assertIs(resource.SharedBankLease._host_lifetime.__globals__,vars(resource))
        self.assertIs(resource._pbs_lock.__wrapped__.__globals__,vars(resource))


class SharedLeaseTests(unittest.TestCase):
    start=fixture.SevenLeaseTests.start
    process=fixture.SevenLeaseTests.process
    query=fixture.SevenLeaseTests.query

    def setUp(self):
        fixture.SevenLeaseTests.setUp(self)
        self.value.update(gpu=5,cpu_affinity=list(range(40,48)),
            shared_bank_CPU_affinity=list(resource.SHARED_CPU_AFFINITY),shared_bank_RSS_limit_GiB=288)
        path=self.root/resource.SHARED_GUARD_FILE;path.write_text('actual shared guard CPU fixture\n')
        self.value['code_sha256'][resource.SHARED_GUARD_FILE]=resource.sha(path)
        self.request_path=self.root/'request.json';self.request_path.write_text(json.dumps(self.value))
        self.start(patch.object(resource,'ROOT',self.root))
        self.start(patch.object(resource,'validate_request',side_effect=lambda x:x))
        self.start(patch.object(resource,'_owned',side_effect=lambda p,**kwargs:Path(p)))
        self.start(patch.object(resource,'_fork_no_exec',return_value=False))
        self.start(patch.object(resource,'_kernel_exiting',return_value=False))
        self.start(patch.object(resource,'capture_cgroup_admission',side_effect=lambda pid:copy.deepcopy(self.admission)))
        self.start(patch.object(resource,'_query',side_effect=self.query))
        self.start(patch.object(resource,'_pbs_query',side_effect=self.query))
        self.start(patch.object(resource,'_pbs_private',side_effect=lambda p,**kwargs:Path(p)))
        @contextmanager
        def lock(folder):yield True
        self.start(patch.object(resource,'_pbs_lock',side_effect=lock))
        self.start(patch.object(resource.psutil,'cpu_count',return_value=96))
        self.start(patch.dict(resource.os.environ,{'CUDA_VISIBLE_DEVICES':'5'}))
        # Construction is legal before expansion, while request stays native8.
        self.lease=resource.SharedBankLease(self.value)
        for actor in self.actors.values():actor.cpus=list(resource.SHARED_CPU_AFFINITY)

    def test_constructor_before_cpu_expansion_and_full288gib_admission(self):
        original=copy.deepcopy(self.value)
        result=self.lease.verify(force=True)
        self.assertEqual(self.value,original)
        self.assertEqual(self.lease.native_request['cpu_affinity'],list(range(40,48)))
        self.assertEqual(self.lease.request['cpu_affinity'],resource.SHARED_CPU_AFFINITY)
        self.assertEqual(result['own_process_tree_RSS_limit_bytes'],288*2**30)
        self.assertEqual(result['aggregate_native_and_cache_capacity_bytes'],704*2**30)
        self.assertFalse(result['PBS_CPU_owned_reservation'])
        self.assertTrue(result['shared_bank_CPU_only_CUDA_verified'])

    def test_original_request_file_bytes_must_match_the_admitted_mapping(self):
        changed=copy.deepcopy(self.value);changed['unrecorded_request_metadata']='changed'
        with self.assertRaisesRegex(ValueError,'request.json bytes'):
            resource.SharedBankLease(changed)

    def test_quick_does_not_scan_immutable_file_inventory_or_worker_rss(self):
        self.lease.verify(force=True)
        with patch.object(self.lease,'_inputs',wraps=self.lease._inputs)as inputs,patch.object(
                self.lease,'_rss',wraps=self.lease._rss)as rss,patch.object(resource,'_snapshot',wraps=resource._snapshot)as snapshots:
            for _ in range(100):self.lease.quick_verify()
            self.assertEqual(inputs.call_count,0)
            self.assertEqual(rss.call_count,0)
            self.assertEqual(snapshots.call_count,100)
            self.assertTrue(all(call.args[0]==self.request_path for call in snapshots.call_args_list))

    def test_full_metadata_and_rss_checks_happen_at_five_seconds(self):
        self.lease.verify(force=True)
        self.worker.rss=289*2**30
        self.monotonic+=4.99
        self.lease.quick_verify()
        self.monotonic+=.01
        with self.assertRaisesRegex(MemoryError,'288GiB'):self.lease.quick_verify()

    def test_force_boundary_detects_source_changed_between_periodic_observations(self):
        self.lease.verify(force=True)
        self.bank.write_text('source metadata changed\n')
        self.lease.quick_verify()
        with self.assertRaisesRegex(ValueError,'metadata changed'):self.lease.verify(force=True)

    def test_request_metadata_and_execution_identity_change_fail_immediately(self):
        self.lease.verify(force=True)
        self.request_path.write_text(self.request_path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'request.json metadata'):self.lease.quick_verify()

    def test_in_memory_controller_request_cannot_authorize_replacement_pid(self):
        self.lease.verify(force=True)
        self.lease.request['controller_identity']['pid']=777
        with self.assertRaisesRegex(ValueError,'identity/request contract'):self.lease.quick_verify()

    def test_own_or_controller_birth_uid_argv_or_affinity_change_is_not_cached(self):
        self.lease.verify(force=True)
        self.controller.birth+=1
        with self.assertRaisesRegex(RuntimeError,'PID/birth'):self.lease.quick_verify()
        self.controller.birth-=1;self.stage.cpus=list(range(40,48))
        with self.assertRaisesRegex(RuntimeError,'64CPU'):self.lease.quick_verify()

    def test_worker_must_be_an_exact64cpu_own_descendant(self):
        self.worker.cpus=list(range(40,48))
        with self.assertRaisesRegex(RuntimeError,'64CPU'):self.lease.verify(force=True)
        self.worker.cpus=list(resource.SHARED_CPU_AFFINITY);self.worker.uid=999
        with self.assertRaisesRegex(RuntimeError,'foreign'):self.lease.verify(force=True)

    def test_actual_capacity_cgroup_storage_or_cpu_inventory_cannot_be_assumed(self):
        self.admission['memory_limits'][0]['limit_bytes']=703*2**30
        with self.assertRaisesRegex(MemoryError,'shared288GiB'):self.lease.verify(force=True)
        self.admission['memory_limits'][0]['limit_bytes']=None;self.lease._controller_cgroup=None
        self.storage.free=9*2**30
        with self.assertRaisesRegex(OSError,'10GiB'):self.lease.verify(force=True)
        self.storage.free=449*2**30
        with patch.object(resource.psutil,'cpu_count',return_value=64):
            with self.assertRaisesRegex(RuntimeError,'CPU inventory'):self.lease.verify(force=True)

    def test_own_cuda_on_different_gpu_is_forbidden_not_ignored(self):
        self.apps='201, GPU-other'
        with self.assertRaisesRegex(RuntimeError,'forbidden CUDA'):self.lease.verify(force=True)

    def test_foreign_training_on_other_gpu_is_observed_without_eviction(self):
        self.apps='777, GPU-other'
        result=self.lease.verify(force=True)
        self.assertTrue(result['shared_bank_CPU_only_CUDA_verified'])
        self.assertFalse(result['processes_signaled'])

    def test_own_cuda_initialization_is_forbidden_during_quick_check(self):
        self.lease.verify(force=True)
        with patch.dict(resource.sys.modules,{'torch':SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True))}):
            with self.assertRaisesRegex(RuntimeError,'CPU-only'):self.lease.quick_verify()

    def test_optional_blank_pbs_timestamp_and_true_deadline_are_inherited(self):
        result=self.lease.verify(force=True)
        self.assertEqual(result['host_lifetime_job_id'],'129776.ECE-util1')
        self.assertEqual(result['host_lifetime_PBS_observation_age_seconds'],0)
        self.wall=4821;self.monotonic=4321
        with self.assertRaisesRegex(RuntimeError,'180seconds'):self.lease.quick_verify()

    def test_finished_pbs_job_and_wallclock_rollback_do_not_extend_cpu_stage(self):
        self.lease.verify(force=True)
        self.monotonic+=61;self.job['job_state']='F'
        with self.assertRaisesRegex(RuntimeError,'no longer proves'):self.lease.verify(force=True)
        self.wall=500;self.monotonic=4350
        with self.assertRaisesRegex(RuntimeError,'180seconds'):self.lease.quick_verify()


if __name__=='__main__':unittest.main()
