"""CPU DEBUG: Shared6 retains only actual GroupV2-admitted immutable peer ledgers."""
import copy
import json
from pathlib import Path
import unittest

from hiercp_v1x import basic_cp_shared_bank_resources_v6 as shared
from hiercp_v1x import basic_cp_seven_resources as resource
from tests import test_basic_cp_shared_bank_resources_v6 as fixture


class RetainedRegistryTests(unittest.TestCase):
    start=fixture.SharedV6Tests.start
    process=fixture.SharedV6Tests.process
    query=fixture.SharedV6Tests.query

    def setUp(self):
        fixture.SharedV6Tests.setUp(self)
        self.prior_path=self.registry_path
        self.prior_sha=resource.sha(self.prior_path)
        self.registry_path=self.root/'fresh_source_registry.json'
        self.value=copy.deepcopy(self.value)
        self.value['shared_resource_registry']=str(self.registry_path)
        path=Path(self.value['root'])/'request.json';path.write_text(json.dumps(self.value))
        self.document=copy.deepcopy(self.document)
        self.document['registration']['5']['request_sha256']=resource.sha(path)
        self.document['allowed_previous_resource_registries']={str(gpu):dict(path=str(self.prior_path),sha256=self.prior_sha)
            for gpu in (0,1,2,3,4,6)}
        self.rewrite()

    def rewrite(self):
        self.registry_path.write_text(json.dumps(self.document))

    def new_lease(self):
        return shared.SharedBankLease(self.value)

    def test_verified_six_retained_peers_accept_current_fresh_source_and_full_budgets(self):
        lease=self.new_lease()
        result=lease.verify(force=True)
        self.assertEqual(result['aggregate_native_and_cache_capacity_bytes'],800*2**30)
        self.assertEqual(result['candidate_subpool_RSS_limit_bytes'],384*2**30)
        self.assertEqual(result['group_process_tree_RSS_limit_bytes'],768*2**30)
        self.assertEqual(result['native_RSS_limit_bytes_preserved'],96*2**30)
        self.assertEqual(result['own_CPU_affinity'],shared.SHARED_CPU_AFFINITY)
        for gpu in (0,1,2,3,4,6):
            self.assertEqual(lease._group_view.registered_requests[gpu]['shared_resource_registry'],str(self.prior_path))
            self.assertTrue(lease._peer_registry_admitted(gpu,lease._group_view.registered_requests[gpu]))
        self.assertTrue(lease._peer_registry_admitted(5,lease._group_view.registered_requests[5]))
        self.assertFalse(result['processes_signaled'])

    def test_missing_descriptor_is_not_implicit_prior_permission(self):
        self.document['allowed_previous_resource_registries'].pop('0');self.rewrite()
        with self.assertRaisesRegex(ValueError,'explicitly pinned prior ledger'):self.new_lease()

    def test_descriptor_wrong_path_rejected_before_shared_admission(self):
        self.document['allowed_previous_resource_registries']['0']['path']=str(self.registry_path);self.rewrite()
        with self.assertRaisesRegex(ValueError,'Distinct absolute immutable prior registry'):self.new_lease()

    def test_selected_controller_row_must_equal_old_pinned_ledger(self):
        self.document['registration']['0']['controller_identity']['create_time']+=1;self.rewrite()
        with self.assertRaisesRegex(ValueError,'exact native arm/GPU/UID|selected controller/request row differs'):self.new_lease()

    def test_own_fresh_controller_cannot_receive_previous_ledger_permission(self):
        self.document['allowed_previous_resource_registries']['5']=dict(path=str(self.prior_path),sha256=self.prior_sha);self.rewrite()
        with self.assertRaisesRegex(ValueError,'other-controller'):self.new_lease()

    def test_prior_ledger_bytes_remain_sha_pinned_initially_and_at_runtime(self):
        lease=self.new_lease()
        self.prior_path.write_text(self.prior_path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'metadata changed'):lease.verify(force=True)
        with self.assertRaisesRegex(ValueError,'admitted Basic source'):self.new_lease()

    def test_prior_peer_request_bytes_are_pinned_separately(self):
        peer=Path(self.requests[0]['root'])/'request.json';peer.write_text(peer.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'admitted Basic source'):self.new_lease()

    def test_descriptor_without_constructor_verified_cache_is_rejected(self):
        lease=self.new_lease();view=lease._group_view
        del view._previous_registry_cache['0']
        self.assertFalse(lease._peer_registry_admitted(0,view.registered_requests[0]))

    def test_cached_selected_row_mismatch_is_rejected(self):
        lease=self.new_lease();view=lease._group_view
        view._previous_registry_cache['0']['registration']['0']['request_sha256']='f'*64
        self.assertFalse(lease._peer_registry_admitted(0,view.registered_requests[0]))

    def test_unverified_descriptor_path_not_in_admitted_files_is_rejected(self):
        lease=self.new_lease();view=lease._group_view
        del view._files[str(self.prior_path)]
        self.assertFalse(lease._peer_registry_admitted(0,view.registered_requests[0]))

    def test_mutated_manifest_permissions_are_rejected(self):
        lease=self.new_lease();view=lease._group_view
        view._previous_registry_manifest['0']['sha256']='f'*64
        self.assertFalse(lease._peer_registry_admitted(0,view.registered_requests[0]))

    def test_unused_permission_is_not_accepted_for_own_current_row(self):
        lease=self.new_lease();view=lease._group_view
        view._previous_registry_manifest['5']=dict(path=str(self.prior_path),sha256=self.prior_sha)
        self.assertFalse(lease._peer_registry_admitted(5,view.registered_requests[5]))

    def test_short_sha_alias_keeps_exact_prior_ledger_support(self):
        descriptor=self.document['allowed_previous_resource_registries']['0']
        descriptor['sha']=descriptor.pop('sha256');self.rewrite()
        lease=self.new_lease()
        self.assertTrue(lease._peer_registry_admitted(0,lease._group_view.registered_requests[0]))

    def test_native96_and_candidate384_hard_caps_remain_fatal(self):
        lease=self.new_lease();self.actors[600].rss=97*2**30
        with self.assertRaisesRegex(MemoryError,'GPU0.*96GiB'):lease.verify(force=True)
        self.actors[600].rss=1*2**30;self.worker.rss=385*2**30
        with self.assertRaises(MemoryError):lease.verify(force=True)


if __name__=='__main__':unittest.main()
