"""CPU DEBUG: a fresh exact registry may use the one approved GPU permutation."""
import copy
import json
from pathlib import Path
import unittest

from hiercp_v1x import basic_cp_gpu_placement as placement
from hiercp_v1x import basic_cp_group_resources as group
from hiercp_v1x import basic_cp_group_resources_v5 as current
from hiercp_v1x import basic_cp_seven_resources as resource
from hiercp_v1x import basic_cp_512_resources as RAM512
from tests import test_basic_cp_group_resources as fixture


class PlacementTests(unittest.TestCase):
    start=fixture.GroupLeaseTests.start
    process=fixture.GroupLeaseTests.process
    query=fixture.GroupLeaseTests.query
    setUp=fixture.GroupLeaseTests.setUp

    def configure(self):
        source=self.root/placement.FILE
        source.write_text('exact CPU DEBUG placement guard\n')
        self.receipt=self.root/'physical_GPU_placement.json'
        self.receipt.write_text(json.dumps(dict(format=placement.FORMAT,operation=placement.OPERATION,
            physical_GPU_arms={str(g):a for g,a in placement.APPROVED.items()})))
        reference=dict(path=str(self.receipt),sha256=resource.sha(self.receipt))
        for gpu,peer in self.requests.items():
            folder=self.root/('GPU'+str(gpu));folder.mkdir(exist_ok=True)
            peer.update(root=str(folder),arm=placement.APPROVED[gpu])
            peer[placement.FIELD]=copy.deepcopy(reference)
            peer['code_sha256'][placement.FILE]=resource.sha(source)
            self.document['registration'][str(gpu)].update(root=str(folder),arm=peer['arm'])
        self.write_requests()

    def write_requests(self):
        for gpu,peer in self.requests.items():
            path=Path(peer['root'])/'request.json';path.write_text(json.dumps(peer))
            self.document['registration'][str(gpu)]['request_sha256']=resource.sha(path)
        self.rewrite()

    rewrite=fixture.GroupLeaseTests.rewrite

    def test_no_receipt_preserves_original_mapping_and_guard(self):
        self.assertEqual(placement.admit_placement(self.value,lambda *args:self.fail('unneeded admission')),group.ARMS)
        self.assertEqual(group.BasicLease(self.value).registered_requests[6]['arm'],'blend50')

    def test_approved_fresh_registry_preserves_exact_physical_CPU_and_controller_identities(self):
        self.configure();lease=group.BasicLease(self.value)
        for gpu in range(7):
            self.assertEqual(lease.registered_requests[gpu]['arm'],placement.APPROVED[gpu])
            self.assertEqual(lease.registered_requests[gpu]['gpu'],gpu)
            self.assertEqual(lease.registered_requests[gpu]['cpu_affinity'],resource.CPUS[gpu])
        self.assertEqual(lease.registered_requests[6]['controller_identity'],self.requests[6]['controller_identity'])
        self.assertIn(str(self.receipt),lease._files)

    def test_completed_blend50_proof_follows_its_arm_and_does_not_claim_multi7_finished(self):
        self.assertEqual(RAM512._RAM512._priority_root(self.lease),Path(self.requests[6]['root']))
        self.configure();lease=group.BasicLease(self.value)
        self.assertEqual(RAM512._RAM512._priority_root(lease),Path(self.requests[1]['root']))
        self.assertNotEqual(RAM512._RAM512._priority_root(lease),Path(self.requests[6]['root']))
        lease.registered_requests[6]['arm']='blend50'
        with self.assertRaisesRegex(ValueError,'One exact registered blend50'):
            RAM512._RAM512._priority_root(lease)

    def test_current_v5_AST_cloned_constructor_keeps_the_placement_admission(self):
        self.configure()
        for name in ('basic_cp_group_resources_v2.py','basic_cp_group_resources_v3.py',
                     'basic_cp_group_resources_v4.py','basic_cp_group_resources_v5.py'):
            relative='hiercp_v1x/'+name;path=self.root/relative;path.write_text('exact CPU DEBUG guard\n')
            for peer in self.requests.values():peer['code_sha256'][relative]=resource.sha(path)
        for peer in self.requests.values():peer.update(shared_resource_guard_version=3,group_observation_revision=5)
        self.write_requests();lease=current.BasicLease(self.value)
        self.assertEqual(lease.registered_requests[6]['arm'],'multi7')
        self.assertEqual(lease.registered_requests[1]['arm'],'blend50')

    def test_unapproved_permutation_or_duplicate_arm_is_rejected(self):
        self.configure();original=self.receipt.read_text()
        for modify in (lambda arms:arms.update({'0':'multi7','6':'multi5'}),
                       lambda arms:arms.update({'6':'multi5'})):
            value=json.loads(original);modify(value['physical_GPU_arms'])
            self.receipt.write_text(json.dumps(value))
            self.value[placement.FIELD]['sha256']=resource.sha(self.receipt)
            with self.assertRaisesRegex(ValueError,'approved whole seven-arm'):
                group.BasicLease(self.value)

    def test_mapping_code_pin_missing_or_receipt_outside_shared_group_is_rejected(self):
        self.configure();self.value['code_sha256'].pop(placement.FILE)
        with self.assertRaisesRegex(ValueError,'receipt and code SHA'):group.BasicLease(self.value)
        self.value['code_sha256'][placement.FILE]=resource.sha(self.root/placement.FILE)
        self.value[placement.FIELD]['path']=str(self.root.parent/'unrelated.json')
        with self.assertRaisesRegex(ValueError,'receipt and code SHA'):group.BasicLease(self.value)

    def test_any_peer_with_old_or_different_receipt_or_code_pin_is_rejected(self):
        self.configure();original=copy.deepcopy(self.requests[4])
        mutations=(lambda peer:peer.pop(placement.FIELD),
                   lambda peer:peer[placement.FIELD].update(sha256='f'*64),
                   lambda peer:peer['code_sha256'].update({placement.FILE:'f'*64}))
        for mutate in mutations:
            self.requests[4]=copy.deepcopy(original);mutate(self.requests[4]);self.write_requests()
            with self.assertRaisesRegex(ValueError,'identical immutable|same exact mapping'):
                group.BasicLease(self.value)

    def test_modified_receipt_is_rejected_by_the_existing_immutable_input_guard(self):
        self.configure();lease=group.BasicLease(self.value)
        self.receipt.write_text(self.receipt.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'metadata changed'):lease._inputs()

    def test_changed_row_arm_or_gpu_or_CPU_cannot_borrow_placement_permission(self):
        self.configure();original=copy.deepcopy(self.requests[6])
        mutations=({'arm':'multi5'},{'gpu':1},{'cpu_affinity':list(resource.CPUS[1])})
        for mutation in mutations:
            self.requests[6]=dict(copy.deepcopy(original),**mutation);self.write_requests()
            with self.assertRaises(ValueError):group.BasicLease(self.value)


if __name__=='__main__':unittest.main()
