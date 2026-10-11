"""CPU DEBUG protocol fixtures only; no CUDA, model or real training is run."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import watch_v24_gnn_to_nnunet as chain
from hiercp_v1x import v24_matched_cli_recovery as repair


def _matched_fixture():
    path = Path(__file__).with_name('test_v24_matched_chain.py')
    spec = importlib.util.spec_from_file_location('_v24_cli_DEBUG_matched_fixture', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    fixture = module.MatchedChainTest(); fixture.setUp()
    return fixture


class MatchedCLIRepairTest(unittest.TestCase):
    def setUp(self):
        self.m = _matched_fixture()

    def tearDown(self):
        self.m.tearDown()

    def make(self, gpu=4, *, failed=False):
        m = self.m; m.make_matched(gpu); f = m.f
        self.previous = copy.deepcopy(m.request); self.old = m.target; self.old.mkdir()
        f.write_json(self.old / 'request.json', self.previous)
        checksum = chain.sha(self.old / 'request.json')
        lease_path = self.previous['cp_parity_handoff']['lease']
        lease = chain.read(lease_path); lease['request_sha256'] = checksum; f.write_json(lease_path, lease)
        claim_checksum = chain.sha(self.previous['source_claim_path'])
        command = [self.previous['python'], '-B', '-u',
                   str(Path(self.previous['code']) / 'tools/watch_v24_gnn_to_nnunet.py'),
                   '--request', str(self.old / 'request.json')]
        f.write_json(self.old / 'registration.json', dict(GPU_arm=gpu, root=str(self.old), pid=9090,
            create_time=150., command=command, request_sha256=checksum,
            CPU_affinity=self.previous['CPU_affinity'], CUDA_visible_devices_while_waiting=''))
        f.write_json(self.old / 'launch.lease.json', dict(request_sha256=checksum, supervisor_pid=9090))
        status = dict(status='WAITING_FOR_FULL_GNN', GPU_arm=gpu, request_sha256=checksum,
            supervisor_pid=9090, supervisor_create_time=150., stage=None, child_pid=None,
            child_create_time=None, gnn_completion=None, stages_completed=[], stage_proofs=[], signals_sent=False,
            source_job=self.previous['source_job'], actual_CPU_affinity=self.previous['CPU_affinity'],
            source_claim_path=self.previous['source_claim_path'], source_claim_sha256=claim_checksum)
        if failed:
            with m.admitted(): completion = chain.inspect_source(self.previous)
            self.assertIsNotNone(completion)
            status.update(status='FAILED', stage='pin_current_gnn', error=repair.CONTROLLER_ERROR,
                command=self.previous['stages'][0]['command'], child_pid=9091, child_create_time=151.,
                stage_log=str(self.old / 'pin_current_gnn.log'), gnn_completion=completion)
            (self.old / 'pin_current_gnn.log').write_text(
                'Traceback (most recent call last):\n  File "' + str(Path(self.previous['code']) /
                'hiercp_v1x/v24_nnunet_cp.py') + '", line 1\n' + repair.PIN_ERROR + '\n', encoding='utf8')
        f.write_json(self.old / 'status.json', status)
        (self.old / 'watcher.log').write_text('DEBUG metadata CPU controller; no real training\n', encoding='utf8')
        code = f.root / 'DEBUG_CLI_repair_runtime'; code.mkdir()
        names = (*self.previous['code_files'], repair.HELPER)
        for name in names:
            path = code / name; path.parent.mkdir(parents=True, exist_ok=True)
            old = Path(self.previous['code']) / name
            path.write_bytes(old.read_bytes() if old.exists() else b'# DEBUG repair source SHA witness\n')
        f.evaluation_heads[str(code)] = '9' * 40
        self.target = f.root / 'DEBUG_CLI_repair_chain'
        self.storage = f.root / 'DEBUG_CLI_repair_storage.json'
        storage = copy.deepcopy(m.storage_document)
        storage['arm_roots']['gpu' + str(gpu)] = str(self.target)
        f.write_json(self.storage, storage)
        self.request = copy.deepcopy(self.previous)
        self.request.update(chain_root=str(self.target), code=str(code), commit='9' * 40,
            code_files={name: chain.sha(code / name) for name in names}, storage_admission=str(self.storage),
            storage_admission_sha256=chain.sha(self.storage), storage_admission_stat=chain._file_stat(self.storage))
        for stage in self.request['stages']:
            stage['command'] = [arg.replace(str(self.old), str(self.target)) for arg in stage['command']]
            stage['command'][3] = str(code / chain.MATCHED_BASIC_ENTRY)
            index = stage['command'].index('--storage-admission') + 1; stage['command'][index] = str(self.storage)
        claim = Path(self.request['source_claim_path'])
        self.proof_path = claim.with_name(claim.stem + '.matched_cli_handoff_proof_DEBUG.json')
        self.proof = dict(format=repair.FORMAT, mode='failed_first_pin' if failed else 'closed_waiter',
            previous_chain_root=str(self.old), chain_root=str(self.target), source_output=self.request['source_output'],
            source_claim_sha256=claim_checksum,
            previous_files_sha256={name: chain.sha(self.old / name) for name in
                repair.COMMON_FILES | ({'pin_current_gnn.log'} if failed else set())},
            previous_supervisor_pid=9090, previous_supervisor_create_time=150., previous_supervisor_command=command,
            old_waiter_closed=True, old_waiter_children=[], old_waiter_cuda_visible_devices='',
            original_GNN_processes_signaled=False, previous_results_written=False,
            evaluation_checkpoint='checkpoint_best.pth', full_training_preserved=True,
            matched_historical_Basic_sources=642, matched_historical_Basic_candidates_per_source=128,
            source_payloads_written=False)
        if failed:
            self.proof.update(previous_child_pid=9091, previous_child_create_time=151.,
                              previous_child_command=self.previous['stages'][0]['command'])
        f.write_json(self.proof_path, self.proof)
        self.request['matched_cli_handoff'] = dict(proof=str(self.proof_path), proof_sha256=chain.sha(self.proof_path),
            lease=str(claim.with_name(claim.stem + '.matched_cli_handoff_lease.json')))
        self.checksum = '8' * 64
        handoff = self.request['matched_cli_handoff']
        f.write_json(handoff['lease'], dict(format=repair.LEASE, GPU_arm=gpu,
            source_output=self.request['source_output'], source_claim_path=str(claim), source_claim_sha256=claim_checksum,
            previous_chain_root=str(self.old), chain_root=str(self.target), request_sha256=self.checksum,
            proof=handoff['proof'], proof_sha256=handoff['proof_sha256']))
        self.preserved = {str(path): path.read_bytes() for path in
            (*self.old.iterdir(), claim, Path(self.previous['cp_parity_handoff']['lease']))}
        self.preserved.update(m.preserved)

    def resign(self):
        self.m.f.write_json(self.proof_path, self.proof)
        self.request['matched_cli_handoff']['proof_sha256'] = chain.sha(self.proof_path)

    def mutate_status(self, changes):
        path = self.old / 'status.json'; status = chain.read(path); status.update(changes)
        self.m.f.write_json(path, status); self.proof['previous_files_sha256']['status.json'] = chain.sha(path)
        self.resign()

    def test_closed_cpu_waiter_keeps_every_previous_lease_and_source_byte(self):
        self.make()
        with self.m.admitted():
            chain.validate_request(self.request)
            self.assertEqual(chain.verify_source_claim(self.request, self.checksum), self.proof['source_claim_sha256'])
        for path, original in self.preserved.items(): self.assertEqual(Path(path).read_bytes(), original)
        for key in ('cp_parity_handoff', 'evaluation_handoff', 'source_job', 'scheduler'):
            self.assertEqual(self.request[key], self.previous[key])

    def test_gpu5_closed_waiter_preserves_storage_evaluation_and_parity_leases(self):
        self.make(5)
        with self.m.admitted():
            chain.validate_request(self.request); chain.verify_source_claim(self.request, self.checksum)
        self.assertEqual(self.request['storage_handoff'], self.previous['storage_handoff'])

    def test_gpu6_only_exact_failed_first_pin_and_completed_full_gnn_is_admitted(self):
        self.make(6, failed=True)
        with self.m.admitted():
            chain.validate_request(self.request); chain.verify_source_claim(self.request, self.checksum)
        for path, original in self.preserved.items(): self.assertEqual(Path(path).read_bytes(), original)

    def test_unknown_new_fields_or_scientific_scheduler_changes_are_rejected(self):
        self.make(6, failed=True)
        for key, value in (('unknown_fallback', True), ('source_commit', '7' * 40), ('scheduler', None),
                           ('source_execution_code', self.request['code']), ('cp_parity_handoff', {})):
            with self.subTest(key=key), self.m.admitted(), self.assertRaisesRegex(ValueError, 'science/assets/PBS'):
                repair.validate(dict(self.request, **{key: value}))

    def test_original_claim_and_all_previous_handoff_leases_are_verified_recursively(self):
        self.make(6, failed=True)
        for path in (Path(self.request['source_claim_path']), *(Path(self.previous[key]['lease']) for key in
                     ('storage_handoff', 'evaluation_handoff', 'cp_parity_handoff'))):
            original = path.read_bytes(); path.write_text('{}', encoding='utf8')
            with self.subTest(path=str(path)), self.m.admitted(), self.assertRaises(ValueError): repair.validate(self.request)
            path.write_bytes(original)

    def test_exact_old_controller_and_child_must_be_stopped(self):
        self.make(6, failed=True)
        for target in (9090, 9091):
            def witness(pid, birth, command=None): return dict(status='running') if pid == target else None
            with self.subTest(target=target), self.m.admitted(), patch.object(chain, 'process_witness', side_effect=witness), \
                    self.assertRaisesRegex(RuntimeError, 'remains live'): repair.validate(self.request)

    def test_every_old_artifact_sha_including_both_logs_is_required(self):
        self.make(6, failed=True)
        for name in repair.COMMON_FILES | {'pin_current_gnn.log'}:
            path = self.old / name; before = path.read_bytes(); path.write_bytes(before + b' ')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'artifact changed'): repair.validate(self.request)
            path.write_bytes(before)

    def test_first_pin_request_and_every_old_code_source_are_immutable(self):
        self.make()
        path = Path(self.previous['code']) / chain.MATCHED_BASIC_ENTRY
        path.write_text('# DEBUG mutated actual CLI source\n', encoding='utf8')
        with self.m.admitted(), self.assertRaisesRegex(ValueError, 'Actual source changed'): repair.validate(self.request)

    def test_no_pin_bank_native_or_later_stage_artifact_can_be_replayed(self):
        self.make(6, failed=True)
        for name in ('pin.json', 'bank', 'native', 'prepare_current_bank.log'):
            path = self.old / name; path.write_text('DEBUG prior artifact', encoding='utf8')
            with self.subTest(name=name), self.m.admitted(), self.assertRaisesRegex(ValueError, 'artifact; no replay'):
                repair.validate(self.request)
            path.unlink()

    def test_later_failed_stage_or_prior_stage_proof_cannot_be_recovered(self):
        self.make(6, failed=True)
        original = chain.read(self.old / 'status.json')
        for changes in ({'stage': 'prepare_current_bank'}, {'stages_completed': ['pin_current_gnn']},
                        {'stage_proofs': [{'accepted': True}]}, {'error': 'RuntimeError: CUDA out of memory'},
                        {'command': self.previous['stages'][1]['command']}):
            self.m.f.write_json(self.old / 'status.json', original); self.mutate_status(changes)
            with self.subTest(changes=changes), self.m.admitted(), self.assertRaises(ValueError): repair.validate(self.request)

    def test_complete_full40_receipt_must_equal_old_failure_snapshot(self):
        self.make(6, failed=True); self.mutate_status({'gnn_completion': {'completed_epochs': 1}})
        with self.m.admitted(), self.assertRaisesRegex(ValueError, 'full40/680'): repair.validate(self.request)

    def test_failed_traceback_cannot_be_other_error_or_checkout(self):
        self.make(6, failed=True); path = self.old / 'pin_current_gnn.log'
        for text in ('Traceback (most recent call last):\n' + repair.PIN_ERROR,
                     'Traceback (most recent call last):\nRuntimeError: CUDA out of memory'):
            path.write_text(text, encoding='utf8'); self.proof['previous_files_sha256'][path.name] = chain.sha(path); self.resign()
            with self.subTest(text=text), self.m.admitted(), self.assertRaisesRegex(ValueError, 'CLI traceback'):
                repair.validate(self.request)

    def test_new_exclusive_lease_cannot_be_replayed_or_retarged(self):
        self.make()
        with self.m.admitted(), self.assertRaisesRegex(ValueError, 'another request/root'):
            chain.verify_source_claim(self.request, '7' * 64)
        request = copy.deepcopy(self.request); request['matched_cli_handoff']['lease'] += '.another'
        with self.assertRaisesRegex(ValueError, 'globally derived exclusive lease'): repair.validate(request)

    def test_helper_must_be_sha_pinned_and_unknown_proof_fields_rejected(self):
        self.make(); request = copy.deepcopy(self.request); request['code_files'].pop(repair.HELPER)
        with self.assertRaisesRegex(ValueError, 'SHA-pinned'): repair.validate(request)
        self.proof['unknown_error_bypass'] = True; self.resign()
        with self.assertRaisesRegex(ValueError, 'Exact stopped'): repair.validate(self.request)

    def test_original_controller_cannot_already_have_cli_repair_lineage(self):
        self.make(); path = self.old / 'request.json'; previous = chain.read(path)
        previous['matched_cli_handoff'] = {}; self.m.f.write_json(path, previous)
        self.proof['previous_files_sha256']['request.json'] = chain.sha(path); self.resign()
        with self.assertRaisesRegex(ValueError, 'nested repair/replay'): repair.validate(self.request)


if __name__ == '__main__':
    unittest.main()
