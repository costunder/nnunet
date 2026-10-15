"""CPU DEBUG: completed candidate provenance survives the approved GPU0/5 swap."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from hiercp_v1x import basic_cp_512_resources as resource
from tools import persist_basic_cp_seven as durable


class CompletedSourcePlacementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bootstrap = self.root/'bootstrap.json'
        self.bootstrap.write_text(json.dumps(dict(source_initially_pending=False)))
        self.requests = {g: dict(gpu=g, arm='other'+str(g)) for g in range(7)}
        self.requests[0].update(arm='candidates2048', shared_banks=str(self.root/'banks'),
            shared_bank_workers=24, shared_bank_CPU_affinity=list(resource.SHARED_CPU_AFFINITY),
            shared_bank_RSS_limit_GiB=384, candidate_binding='actual_frozen_binding',
            candidate_binding_sha256='a'*64)
        self.requests[5].update(arm='multi5')
        self.view = SimpleNamespace(registered_requests=self.requests, registry_sha256='b'*64)

    def full_bank_metadata(self):
        banks = Path(self.requests[0]['shared_banks'])
        for count in (512, 1024, 2048):
            prefix = banks/str(count)
            prefix.mkdir(parents=True)
            index = prefix/'index.json'
            index.write_text(json.dumps(dict(source_entries=642, candidate_count=count)))
            (prefix/'complete.json').write_text(json.dumps(dict(complete=True,
                index_sha256=resource.common.sha(index))))
        value = dict(complete=True, full_sources=642, prefixes={str(c): {} for c in (512, 1024, 2048)},
            CUDA_initialized=False)
        (banks/'complete.json').write_text(json.dumps(value))
        return banks, value

    def test_source_selector_tracks_scientific_arm_and_keeps_original_default(self):
        self.assertIs(resource._source_request(self.view), self.requests[0])
        self.requests[0], self.requests[5] = self.requests[5], self.requests[0]
        self.assertIs(resource._source_request(self.view), self.requests[5])

    def test_duplicate_or_missing_source_arm_has_no_slot5_fallback(self):
        self.requests[5]['arm'] = 'candidates2048'
        with self.assertRaisesRegex(ValueError, 'One exact registered'):
            resource._source_request(self.view)
        self.requests[5]['arm'] = self.requests[0]['arm'] = 'multi5'
        with self.assertRaisesRegex(ValueError, 'One exact registered'):
            resource._source_request(self.view)

    def test_completed_source_validation_uses_moved_source_full_prefixes(self):
        banks, value = self.full_bank_metadata()
        admitted = []
        self.view._admit = lambda *args: admitted.append(args)
        def complete(path, validator):
            self.assertEqual(path, banks/'complete.json')
            validator(value)
            return True
        self.view._completion = complete
        self.assertTrue(resource._RAM512.source_complete(self.view))
        self.assertEqual(len(admitted), 6)
        self.assertNotIn('shared_banks', self.requests[5])

    def test_moved_source_does_not_bypass_full642_prefix_completion(self):
        banks, value = self.full_bank_metadata()
        (banks/'1024/index.json').write_text(json.dumps(dict(source_entries=641, candidate_count=1024)))
        self.view._admit = mock.Mock()
        self.view._completion = lambda _, validator: validator(value)
        with self.assertRaisesRegex(ValueError, 'complete642'):
            resource._RAM512.source_complete(self.view)

    def common_view(self):
        for request in self.requests.values():
            request.update(shared_resource_registry=str(self.root/'registry.json'),
                owned_tmpfs_root=str(self.root), RAM512_claim_ledger=str(self.root/'claims.json'),
                RAM512_source_pending_receipt=dict(path=str(self.bootstrap), sha256=resource.common.sha(self.bootstrap)),
                code_sha256={name: 'c'*64 for name in (resource.GUARD_FILE, resource.CLAIMS_FILE,
                    resource.group.GUARD_FILE, resource.shared.SHARED_GUARD_FILE, resource.pause.GUARD_FILE)})
        self.view.registry_path = self.root/'registry.json'
        self.view.request = self.requests[5]
        self.view._admit = mock.Mock()
        return self.view

    def test_full_common_guard_accepts_only_completed_source_after_slot_move(self):
        view = self.common_view()
        with mock.patch.object(resource, 'validate_request'):
            resource._RAM512._validate_common(view, view)
        self.view._admit.assert_called_once_with(str(self.bootstrap), resource.common.sha(self.bootstrap))
        self.bootstrap.write_text(json.dumps(dict(source_initially_pending=True)))
        with mock.patch.object(resource, 'validate_request'), self.assertRaisesRegex(ValueError, 'fully completed source'):
            resource._RAM512._validate_common(view, view)

    def test_moved_source_keeps_original24worker64CPU384declaration(self):
        view = self.common_view()
        self.requests[0]['shared_bank_workers'] = 4
        with mock.patch.object(resource, 'validate_request'), self.assertRaisesRegex(ValueError, 'full24-worker'):
            resource._RAM512._validate_common(view, view)

    def test_source_release_receipt_names_moved_actual_source_bank(self):
        banks, _ = self.full_bank_metadata()
        self.view.source_complete = lambda: True
        self.view.priority_complete = lambda: False
        self.view._phase_receipt = mock.Mock(return_value=dict(path='CPU_DEBUG_release', sha256='c'*64))
        resolver = mock.Mock()
        resolver.inspect.return_value = dict(state='gone')
        resolver.has_cuda.return_value = False
        state = dict(source_captured=[{'CPU_DEBUG_actor': True}], source_actor_identity={'CPU_DEBUG_actor': True}, claims={})
        with mock.patch.object(resource, 'ProcessResolver', return_value=resolver):
            result = resource._RAM512._phase_observations(self.view, state)
        self.assertTrue(result['source_complete'])
        receipt = self.view._phase_receipt.call_args.args[1]
        self.assertEqual(receipt['scientific_completion']['path'], str(banks/'complete.json'))

    def test_durable_snapshot_selects_moved_actual_source_and_its_binding(self):
        requests = {str(g): q for g, q in self.requests.items()}
        selected = durable.shared_source_request(requests)
        self.assertIs(selected, requests['0'])
        self.assertEqual(selected['candidate_binding_sha256'], 'a'*64)
        self.assertNotIn('shared_banks', requests['5'])
        requests['5']['arm'] = 'candidates2048'
        with self.assertRaisesRegex(ValueError, 'One exact registered'):
            durable.shared_source_request(requests)


if __name__ == '__main__':
    unittest.main()
