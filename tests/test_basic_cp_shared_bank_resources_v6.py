"""CPU-only DEBUG: corrected shared available+all-group-resident RAM formula."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_shared_bank_resources_v6 as shared
from hiercp_v1x import basic_cp_seven_resources as resource
from tests import test_basic_cp_resources_v5 as fixture
from tests import test_basic_cp_seven_resources as actors


class SharedV6Tests(unittest.TestCase):
    start = fixture.SharedV5Tests.start
    process = fixture.SharedV5Tests.process
    query = fixture.SharedV5Tests.query

    def setUp(self):
        fixture.SharedV5Tests.setUp(self)
        self.registry_path = self.root / 'exact_seven_registry.json'
        for filename in (shared.SHARED_GUARD_FILE, shared.group.GUARD_FILE,
                         shared.group.resource_v3.GUARD_FILE,
                         shared.group.resource_v3.original.GUARD_FILE):
            source = self.root / filename
            source.write_text('exact future shared/group guard dependency\n')
            self.value['code_sha256'][filename] = resource.sha(source)
        self.value.update(arm='candidates2048',shared_resource_registry=str(self.registry_path),
                          shared_resource_observation_revision=6)
        self.requests = {}
        registration = {}
        for gpu in range(7):
            value = copy.deepcopy(self.value)
            if gpu == 5:
                controller = self.controller
                folder = self.root
            else:
                folder = self.root / ('GPU'+str(gpu));folder.mkdir()
                controller = actors.Actor(500+gpu,['python','controller_GPU'+str(gpu)])
                controller.cpus = list(resource.CPUS[gpu])
                stage = actors.Actor(600+gpu,['python','stage_GPU'+str(gpu)])
                stage.cpus = list(resource.CPUS[gpu])
                controller.descendants = [stage]
                self.actors[controller.pid] = controller;self.actors[stage.pid] = stage
            value.update(root=str(folder),native_root=str(folder),gpu=gpu,
                arm=shared.group.prior.original.ARMS[gpu],cpu_affinity=list(resource.CPUS[gpu]),
                controller_identity=dict(pid=controller.pid,create_time=controller.birth,
                    UID=controller.uid,command=controller.command))
            path = folder/'request.json';path.write_text(json.dumps(value))
            self.requests[gpu] = value
            registration[str(gpu)] = dict(root=str(folder),arm=value['arm'],
                controller_identity=value['controller_identity'],request_sha256=resource.sha(path))
        self.value = self.requests[5]
        self.document = dict(format=shared.group.FORMAT,registration=registration)
        self.registry_path.write_text(json.dumps(self.document))
        self.lease = shared.SharedBankLease(self.value)
        self.start(patch.dict(shared._CPUStageGroupLease._descendant.__globals__,{
            '_fork_no_exec':lambda p:getattr(p,'fork_no_exec',False),
            '_kernel_exiting':lambda p:getattr(p,'kernel_exiting',False)}))

    def test_available_below800_plus_actual_native_rss_admits_shared_without_double_counting(self):
        self.host.available = 760 * 2**30
        for gpu in (0,1,2,3,6):self.actors[600+gpu].rss = 11 * 2**30
        result = self.lease.verify(force=True)
        self.assertLess(result['host_available_RAM_bytes'] + result['own_process_tree_RSS_bytes'],800*2**30)
        self.assertGreaterEqual(result['host_available_RAM_bytes'] + result['group_process_tree_RSS_bytes'],800*2**30)
        self.assertEqual(result['own_process_tree_RSS_limit_bytes'],384*2**30)
        self.assertEqual(result['candidate_subpool_RSS_limit_bytes'],384*2**30)
        self.assertEqual(result['group_process_tree_RSS_limit_bytes'],768*2**30)
        self.assertEqual(result['aggregate_native_and_cache_capacity_bytes'],800*2**30)
        self.assertEqual(self.value['RSS_limit_bytes'],96*2**30)
        self.assertEqual(result['own_CPU_affinity'],shared.SHARED_CPU_AFFINITY)

    def test_source_peak_adds_back_exact_tree_once_and_keeps_other_native96_limits(self):
        self.host.available = 550 * 2**30
        self.worker.rss = 275 * 2**30
        result = self.lease.verify(force=True)
        self.assertGreaterEqual(result['group_process_tree_RSS_bytes'],275*2**30)
        self.assertEqual(len(result['group_RSS_deduplicated_PIDs']),len(set(result['group_RSS_deduplicated_PIDs'])))
        self.actors[600].rss = 97 * 2**30
        with self.assertRaisesRegex(MemoryError,'GPU0.*96GiB'):
            self.lease.verify(force=True)

    def test_candidate384_budget_includes_existing_native_gpu3_and4(self):
        self.worker.rss = 360 * 2**30
        self.actors[603].rss = 20 * 2**30
        self.actors[604].rss = 20 * 2**30
        with self.assertRaisesRegex(MemoryError,'shared384GiB'):
            self.lease.verify(force=True)

    def test_actual_capacity_or_cgroup_cannot_be_faked(self):
        self.host.available = 700 * 2**30
        with self.assertRaisesRegex(MemoryError,'available RAM'):
            self.lease.verify(force=True)
        self.host.available = 937 * 2**30
        self.admission['memory_limits'][0]['limit_bytes'] = 799 * 2**30
        self.lease = shared.SharedBankLease(self.value)
        with self.assertRaisesRegex(MemoryError,'800GiB'):
            self.lease.verify(force=True)

    def test_pinned_peer_request_metadata_and_identity_remain_exact(self):
        peer = Path(self.requests[0]['root'])/'request.json'
        peer.write_text(peer.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'metadata changed'):
            self.lease.verify(force=True)

    def test_foreign_peer_uid_or_reused_controller_cannot_supply_fake_resident_memory(self):
        self.actors[600].uid = 999
        with self.assertRaisesRegex(RuntimeError,'foreign'):
            self.lease.verify(force=True)

    def test_quick_does_not_repeat_seven_row_source_inventory_scan(self):
        self.lease.verify(force=True)
        with patch.object(self.lease._group_view,'_inputs',wraps=self.lease._group_view._inputs)as inputs:
            for _ in range(10):self.lease.quick_verify()
            self.assertEqual(inputs.call_count,0)
            self.monotonic += 5
            self.lease.quick_verify()
            self.assertGreater(inputs.call_count,0)

    def test_missing_new_sha_group_sha_or_implicit_revision_is_rejected(self):
        for replacement in (5,True,6.0):
            value = copy.deepcopy(self.value);value['shared_resource_observation_revision'] = replacement
            with self.assertRaisesRegex(ValueError,'observation6'):
                shared.SharedBankLease(value)
        for filename in (shared.SHARED_GUARD_FILE,shared.group.GUARD_FILE):
            value = copy.deepcopy(self.value);value['code_sha256'].pop(filename)
            with self.assertRaisesRegex(ValueError,'SHAs'):
                shared.SharedBankLease(value)

    def test_peer_without_explicit_new_source_admission_cannot_hide_old_group_formula(self):
        peer = self.requests[0]
        peer.pop('shared_resource_observation_revision')
        path = Path(peer['root'])/'request.json';path.write_text(json.dumps(peer))
        self.document['registration']['0']['request_sha256'] = resource.sha(path)
        self.registry_path.write_text(json.dumps(self.document))
        with self.assertRaisesRegex(ValueError,'All seven current peers'):
            shared.SharedBankLease(self.value)


if __name__=='__main__':unittest.main()
