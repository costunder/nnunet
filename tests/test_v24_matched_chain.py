"""CPU DEBUG metadata tests; no model, production data or training is run."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import watch_v24_gnn_to_nnunet as chain
from hiercp_v1x import v24_readonly_native_storage, v24_readonly_static_operator_storage


def _fixture_class():
    path = Path(__file__).with_name('test_v24_gnn_to_nnunet_chain.py')
    spec = importlib.util.spec_from_file_location('_v24_matched_DEBUG_chain_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CompletedArmFixture


class MatchedChainTest(unittest.TestCase):
    def setUp(self):
        self.f = _fixture_class()(); self.f.setUp()
        self.module = ModuleType('hiercp_v1x.v24_matched_basic_cp_pipeline')
        self.module.verify_admission = Mock()
        self.module.check_aggregate_disk = Mock(return_value={'required_free_bytes': 17 * 2**30})
        self.module.stage_proof = Mock(return_value={'DEBUG_only': True})

    def tearDown(self):
        self.f.tearDown()

    def admitted(self, *, process=None):
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, {self.module.__name__: self.module}))
        stack.enter_context(patch('hiercp_v1x.v24_readonly_native_storage.verify_admission'))
        stack.enter_context(self.f.static_mock())
        stack.enter_context(patch.object(chain.subprocess, 'check_output', side_effect=self.f.evaluation_git_head))
        stack.enter_context(patch.object(chain, 'process_witness', return_value=process))
        return stack

    def make_matched(self, gpu=4):
        f = self.f; f.make_best_evaluation_handoff(gpu)
        previous = copy.deepcopy(f.request); previous_root = f.target; previous_root.mkdir()
        f.write_json(previous_root / 'request.json', previous)
        previous_checksum = chain.sha(previous_root / 'request.json')
        f.evaluation_request_checksum = previous_checksum; f.write_evaluation_lease()
        claim_checksum = chain.sha(previous['source_claim_path'])
        command = [previous['python'], '-B', '-u', str(Path(previous['code']) / 'tools/watch_v24_gnn_to_nnunet.py'),
                   '--request', str(previous_root / 'request.json')]
        f.write_json(previous_root / 'registration.json', dict(GPU_arm=gpu, root=str(previous_root),
            pid=888, create_time=125., command=command, request_sha256=previous_checksum,
            CPU_affinity=previous['CPU_affinity'], CUDA_visible_devices_while_waiting=''))
        f.write_json(previous_root / 'launch.lease.json', dict(request_sha256=previous_checksum, supervisor_pid=888))
        f.write_json(previous_root / 'status.json', dict(status='WAITING_FOR_FULL_GNN', GPU_arm=gpu,
            request_sha256=previous_checksum, supervisor_pid=888, supervisor_create_time=125., stage=None,
            child_pid=None, child_create_time=None, gnn_completion=None, stages_completed=[], stage_proofs=[],
            signals_sent=False, source_job=previous['source_job'], actual_CPU_affinity=previous['CPU_affinity'],
            source_claim_path=previous['source_claim_path'], source_claim_sha256=claim_checksum))
        (previous_root / 'watcher.log').write_text('DEBUG CPU-only closed BEST waiter; no training\n')
        self.previous_root, self.previous = previous_root, previous
        self.preserved = {str(path): path.read_bytes() for path in
            (*previous_root.iterdir(), Path(previous['source_claim_path']),
             Path(previous['evaluation_handoff']['proof']), Path(previous['evaluation_handoff']['lease']))}
        self.preserved.update(f.previous_evaluation_bytes)
        code = f.root / ('DEBUG_matched_runtime_gpu' + str(gpu)); code.mkdir()
        names = (*previous['code_files'], chain.MATCHED_BASIC_ENTRY, chain.MATCHED_BASIC_PIPELINE, chain.MATCHED_BASIC_SCORING)
        for name in names:
            path = code / name; path.parent.mkdir(parents=True, exist_ok=True)
            old = Path(previous['code']) / name
            path.write_bytes(old.read_bytes() if old.exists() else b'# DEBUG metadata witness only\n')
        f.evaluation_heads[previous['code']] = previous['commit']
        f.evaluation_heads[str(code)] = 'e' * 40
        self.target = f.root / ('DEBUG_matched_native_gpu' + str(gpu))
        self.basic = f.root / 'DEBUG_historical_Basic642'; self.basic.mkdir()
        f.write_json(self.basic / 'index.json', {'DEBUG_only': True})
        self.storage = f.root / 'DEBUG_matched_storage.json'
        self.storage_document = dict(format='v24_matched_Basic642_storage_admission_v1', profile=chain.MATCHED_BASIC_PROFILE,
            original_readonly_admission=dict(path=previous['storage_admission'],
                sha256=previous['storage_admission_sha256'], stat=previous['storage_admission_stat']),
            historical_basic_bank=dict(path=str(self.basic), index_sha256=chain.sha(self.basic / 'index.json'),
                index_stat=chain._file_stat(self.basic / 'index.json'), manifest_sha256='a' * 64,
                manifestpath=str(f.root / 'DEBUG_manifest.json')),
            inventory=dict(path=str(f.inventory), sha256=chain.sha(f.inventory)),
            baseline=dict(preprocessed=str(f.preprocessed)), arm_roots={'gpu' + str(gpu): str(self.target)},
            aggregate_budget=dict(required_free_bytes=17 * 2**30, global_peak_checkpoint_slots=7))
        f.write_json(self.storage, self.storage_document)
        self.request = copy.deepcopy(previous)
        self.request.update(chain_root=str(self.target), code=str(code), commit='e' * 40,
            code_files={name: chain.sha(code / name) for name in names}, storage_profile=chain.MATCHED_BASIC_PROFILE,
            storage_extension=None, storage_admission=str(self.storage), storage_admission_sha256=chain.sha(self.storage),
            storage_admission_stat=chain._file_stat(self.storage), min_free_disk_GiB=17,
            historical_basic_bank=str(self.basic), historical_basic_bank_index_sha256=chain.sha(self.basic / 'index.json'))
        for stage in self.request['stages']:
            stage['command'] = [arg.replace(str(previous_root), str(self.target)) for arg in stage['command']]
            stage['command'][3] = str(code / chain.MATCHED_BASIC_ENTRY)
            index = stage['command'].index('--storage-admission') + 1; stage['command'][index] = str(self.storage)
            if stage['action'] == 'prepare-current-bank':
                stage['command'] += ['--historical-basic-bank', str(self.basic)]
            if stage['action'] == 'prepare-native':
                index = stage['command'].index('--bank') + 1
                stage['command'][index] = str(self.target / 'bank/score_overlay.json')
        claim_path = Path(previous['source_claim_path'])
        self.proof_path = claim_path.with_name(claim_path.stem + '.cp_parity_handoff_proof_DEBUG.json')
        proof = dict(format=chain.CP_PARITY_HANDOFF_PROOF, previous_chain_root=str(previous_root),
            chain_root=str(self.target), source_output=previous['source_output'], source_claim_sha256=claim_checksum,
            previous_files_sha256={name: chain.sha(previous_root / name) for name in
                ('request.json', 'status.json', 'launch.lease.json', 'registration.json')},
            previous_supervisor_pid=888, previous_supervisor_create_time=125., previous_supervisor_command=command,
            old_waiter_closed=True, old_waiter_children=[], old_waiter_cuda_visible_devices='',
            original_GNN_processes_signaled=False, previous_results_written=False, evaluation_checkpoint='checkpoint_best.pth',
            full_training_preserved=True, matched_historical_Basic_sources=642,
            matched_historical_Basic_candidates_per_source=128, source_payloads_written=False)
        f.write_json(self.proof_path, proof)
        self.request['cp_parity_handoff'] = dict(proof=str(self.proof_path), proof_sha256=chain.sha(self.proof_path),
            lease=str(claim_path.with_name(claim_path.stem + '.cp_parity_handoff_lease.json')))
        self.request_checksum = 'f' * 64
        handoff = self.request['cp_parity_handoff']
        f.write_json(handoff['lease'], dict(format=chain.CP_PARITY_HANDOFF_LEASE, GPU_arm=gpu,
            source_output=previous['source_output'], source_claim_path=previous['source_claim_path'],
            source_claim_sha256=claim_checksum, previous_chain_root=str(previous_root), chain_root=str(self.target),
            request_sha256=self.request_checksum, proof=handoff['proof'], proof_sha256=handoff['proof_sha256']))

    def resign_proof(self, changes):
        proof = chain.read(self.proof_path); proof.update(changes); self.f.write_json(self.proof_path, proof)
        self.request['cp_parity_handoff']['proof_sha256'] = chain.sha(self.proof_path)

    def resign_storage(self, changes):
        self.storage_document.update(changes); self.f.write_json(self.storage, self.storage_document)
        self.request.update(storage_admission_sha256=chain.sha(self.storage), storage_admission_stat=chain._file_stat(self.storage))

    def test_gpu4_full_lineage_is_preserved_with_new_matched_commands(self):
        self.make_matched()
        with self.admitted():
            chain.validate_request(self.request)
            self.assertEqual(chain.verify_source_claim(self.request, self.request_checksum), chain.sha(self.request['source_claim_path']))
        for path, before in self.preserved.items(): self.assertEqual(Path(path).read_bytes(), before)
        self.assertEqual(self.request['evaluation_handoff'], self.previous['evaluation_handoff'])
        self.module.verify_admission.assert_called_with(Path(self.request['storage_admission']), full_hash=False)

    def test_gpu5_and_gpu6_keep_both_prior_exclusive_leases(self):
        for gpu in (5, 6):
            with self.subTest(gpu=gpu):
                self.f.tearDown(); self.f = _fixture_class()(); self.f.setUp(); self.make_matched(gpu)
                with self.admitted():
                    chain.validate_request(self.request)
                    chain.verify_source_claim(self.request, self.request_checksum)
                self.assertEqual(self.request['storage_handoff'], self.previous['storage_handoff'])
                for path, before in self.preserved.items(): self.assertEqual(Path(path).read_bytes(), before)

    def test_all_original_source_and_unrecognized_contract_fields_are_immutable(self):
        self.make_matched()
        for field, value in (('source_commit', 'f' * 40), ('source_output', str(self.basic)),
                             ('scheduler', {'changed': True}), ('CPU_affinity', [1, 2, 3, 4]),
                             ('evaluation_handoff', {}), ('training_contract_extra', 'unreviewed')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                chain._validate_cp_parity_handoff(dict(self.request, **{field: value}))

    def test_best_full_scale_no_write_contract_is_required(self):
        self.make_matched(); original = chain.read(self.proof_path)
        for changes in ({'evaluation_checkpoint': 'checkpoint_final.pth'}, {'full_training_preserved': False},
                        {'matched_historical_Basic_sources': 105}, {'matched_historical_Basic_candidates_per_source': 8},
                        {'source_payloads_written': True}, {'original_GNN_processes_signaled': True},
                        {'old_waiter_cuda_visible_devices': '4'}, {'old_waiter_children': [123]},
                        {'previous_results_written': True}):
            self.resign_proof(dict(original, **changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'CPU-only full642'):
                chain._validate_cp_parity_handoff(self.request)

    def test_exact_old_process_identity_and_closed_waiter_are_required(self):
        self.make_matched(); original = chain.read(self.proof_path)
        with self.admitted():
            for changes in ({'previous_supervisor_pid': 999}, {'previous_supervisor_create_time': 126.},
                            {'previous_supervisor_command': ['other']}, {'previous_supervisor_create_time': float('nan')}):
                self.resign_proof(dict(original, **changes))
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'registered CPU-only'):
                    chain._validate_cp_parity_handoff(self.request)
        self.resign_proof(original)
        with self.admitted(process={'status': 'running'}), self.assertRaisesRegex(RuntimeError, 'remains live'):
            chain._validate_cp_parity_handoff(self.request)

    def test_prior_native_artifacts_or_started_stage_refuse_parity_switch(self):
        self.make_matched()
        with self.admitted():
            (self.previous_root / 'native').mkdir()
            with self.assertRaisesRegex(ValueError, 'native stage/output'):
                chain._validate_cp_parity_handoff(self.request)
            (self.previous_root / 'native').rmdir()
            path = self.previous_root / 'status.json'; status = chain.read(path); status['stage'] = 'pin_current_gnn'
            self.f.write_json(path, status)
            proof = chain.read(self.proof_path); proof['previous_files_sha256']['status.json'] = chain.sha(path)
            self.resign_proof(proof)
            with self.assertRaisesRegex(ValueError, 'registered CPU-only'):
                chain._validate_cp_parity_handoff(self.request)

    def test_changed_previous_code_file_or_artifact_is_rejected(self):
        self.make_matched(); path = self.previous_root / 'registration.json'; path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            chain._validate_cp_parity_handoff(self.request)

    def test_previous_best_and_storage_lease_tampering_is_rejected_recursively(self):
        self.make_matched(5)
        for key in ('evaluation_handoff', 'storage_handoff'):
            path = Path(self.previous[key]['lease']); before = path.read_bytes(); path.write_text('{}')
            with self.subTest(key=key), self.admitted(), self.assertRaisesRegex(ValueError, 'handoff lease'):
                chain._validate_cp_parity_handoff(self.request)
            path.write_bytes(before)

    def test_new_parity_lease_cannot_be_replayed_for_other_request(self):
        self.make_matched()
        with self.admitted(), self.assertRaisesRegex(ValueError, 'another request/root'):
            chain.verify_source_claim(self.request, 'e' * 64)

    def test_historical_bank_and_admission_sha_stat_checked_at_each_poll(self):
        self.make_matched(); path = self.basic / 'index.json'; path.write_text('{"changed":true}')
        with self.admitted(), self.assertRaisesRegex(ValueError, 'immutable Basic642'):
            chain._readonly_storage(self.request)

    def test_missing_profile_handoff_helper_or_exact_historical_bank_flag_rejected(self):
        self.make_matched()
        with self.admitted():
            for field in ('cp_parity_handoff', 'historical_basic_bank_index_sha256'):
                request = copy.deepcopy(self.request); request.pop(field)
                with self.subTest(field=field), self.assertRaises(ValueError): chain.validate_request(request)
            request = copy.deepcopy(self.request); request['code_files'].pop(chain.MATCHED_BASIC_SCORING)
            with self.assertRaisesRegex(ValueError, 'SHA pinned'): chain.validate_request(request)
            request = copy.deepcopy(self.request)
            index = request['stages'][1]['command'].index('--historical-basic-bank')
            request['stages'][1]['command'][index + 1] = str(self.f.cache)
            with self.assertRaisesRegex(ValueError, 'stage changed'): chain.validate_request(request)

    def test_admitted_global_checkpoint_budget_cannot_be_weakened(self):
        self.make_matched()
        with self.admitted():
            with self.assertRaisesRegex(ValueError, 'global7 budget'):
                chain.validate_request(dict(self.request, min_free_disk_GiB=16))
            self.resign_storage(dict(aggregate_budget=dict(required_free_bytes=17 * 2**30, global_peak_checkpoint_slots=9)))
            with self.assertRaisesRegex(ValueError, 'global7 budget'): chain.validate_request(self.request)

    def test_matched_source_execution_is_not_retargeted_to_new_native_checkout(self):
        self.make_matched(5)
        with self.admitted(), self.assertRaisesRegex(ValueError, 'exact scientific execution'):
            chain._readonly_storage(dict(self.request, source_execution_code=self.request['code']))

    def test_stageproof_delegates_to_matched_provenance_validator_only(self):
        with self.admitted():
            request = dict(storage_profile=chain.MATCHED_BASIC_PROFILE)
            self.assertEqual(chain.stage_proof(request, 'train'), {'DEBUG_only': True})
        self.module.stage_proof.assert_called_once_with(request, 'train')

    def test_resources_keep_gpu_occupancy_and_matched_global_budget_guards(self):
        self.make_matched(5)
        process = SimpleNamespace(cpu_affinity=lambda: self.request['CPU_affinity'])
        rows = ['5, GPU-5, 49140, 49140, Disabled', '']
        with self.admitted(), patch('psutil.Process', return_value=process), \
                patch.object(chain, '_run', side_effect=rows), \
                patch.object(chain.shutil, 'disk_usage', return_value=SimpleNamespace(free=30 * 2**30)):
            result = chain.resources(self.request)
        self.assertEqual(result['minimum_free_disk_GiB'], 17)
        self.module.check_aggregate_disk.assert_called_once_with(self.storage_document, self.request['chain_root'], preparation=True)
        rows = ['5, GPU-5, 49140, 49140, Disabled', '123, GPU-5']
        with self.admitted(), patch('psutil.Process', return_value=process), patch.object(chain, '_run', side_effect=rows), \
                self.assertRaisesRegex(chain.AssignedGPUUnavailable, 'compute application'):
            chain.resources(self.request)


if __name__ == '__main__':
    unittest.main()
