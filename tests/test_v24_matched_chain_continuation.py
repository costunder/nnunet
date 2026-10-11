"""CPU UNIT protocol receipts only; no actual allocation or GPU launch claim."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import watch_v24_gnn_to_nnunet as chain
from hiercp_v1x import v24_matched_chain_continuation as continuation


class MatchedContextContinuationCPUUnit(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name('test_v24_matched_cli_recovery.py')
        spec = importlib.util.spec_from_file_location('_depth1_chain_CPU_fixture', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.fixture = module.MatchedCLIRepairTest(); self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def make(self):
        fixture = self.fixture; fixture.make(6, failed=True)
        self.request = copy.deepcopy(fixture.request)
        self.request.pop('matched_cli_handoff')
        self.request['context_depth1'] = True
        self.request['stages'][1]['command'].append('--context-depth1')
        self.request['scheduler'] = dict(self.request['scheduler'], job_id='129528.ECE-util1',
            expires_at_unix=self.request['scheduler']['expires_at_unix'] + 86400)
        for name in (continuation.HELPER, continuation.CONTEXT_HELPER):
            path = Path(self.request['code']) / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# CPU UNIT source witness only\n', encoding='utf8')
            self.request['code_files'][name] = chain.sha(path)
        claim = Path(self.request['source_claim_path'])
        self.binding_path = claim.with_name(claim.stem + '.allocation_binding_CPU_UNIT.json')
        self.binding = dict(format=continuation.ALLOCATION_FORMAT, scheduler=self.request['scheduler'],
            GPU_arm=6, GPU_UUID=self.request['GPU_UUID'], CPU_affinity=self.request['CPU_affinity'],
            assigned_CPU_ids=[32, 33, 34, 35, 36, 37], ncpus=6, ngpus=1, RAM_GiB=128,
            owner='aicompetition06', node='ece-a6gpu6', physical_GPU_indices=[6], PBS_job_id='129528.ECE-util1',
            allocation_process=dict(pid=8181, create_time=100., command=['CPU_UNIT_own_PBS_witness'], UID=41833))
        fixture.m.f.write_json(self.binding_path, self.binding)
        self.proof = copy.deepcopy(fixture.proof)
        self.proof['format'] = continuation.FORMAT
        self.proof['allocation_binding'] = dict(path=str(self.binding_path), sha256=chain.sha(self.binding_path))
        self.proof_path = claim.with_name(claim.stem + '.context_allocation_continuation_proof_CPU_UNIT.json')
        fixture.m.f.write_json(self.proof_path, self.proof)
        self.request['context_allocation_continuation'] = dict(proof=str(self.proof_path),
            proof_sha256=chain.sha(self.proof_path), lease=str(continuation.lease_path(claim)))
        self.checksum = '7' * 64
        fixture.m.f.write_json(self.request['context_allocation_continuation']['lease'],
            continuation.lease_document(self.request, self.checksum, self.proof))

    def admitted(self, *, live_old=None, allocation_live=True, environment=None, affinity=None):
        stack = ExitStack(); stack.enter_context(self.fixture.m.admitted())
        stack.enter_context(patch.object(continuation, 'GPU6_UUID', self.request['GPU_UUID']))
        def witness(pid, birth, command=None):
            if pid == live_old:return dict(status='running')
            return dict(status='running') if pid == 8181 and allocation_live else None
        stack.enter_context(patch.object(chain, 'process_witness', side_effect=witness))
        process = SimpleNamespace(uids=lambda:SimpleNamespace(real=41833),
            cpu_affinity=lambda: affinity if affinity is not None else self.binding['assigned_CPU_ids'],
            environ=lambda: environment if environment is not None else dict(PBS_JOBID='129528.ECE-util1',
                CUDA_VISIBLE_DEVICES=self.request['GPU_UUID']))
        stack.enter_context(patch('psutil.Process', return_value=process))
        return stack

    def resign(self):
        self.fixture.m.f.write_json(self.proof_path, self.proof)
        self.request['context_allocation_continuation']['proof_sha256'] = chain.sha(self.proof_path)

    def test_exact_completed_source_and_every_old_lineage_remain_immutable(self):
        self.make()
        with self.admitted():
            chain.validate_request(self.request)
            self.assertEqual(chain.verify_source_claim(self.request, self.checksum), self.proof['source_claim_sha256'])
        for path, original in self.fixture.preserved.items():
            self.assertEqual(Path(path).read_bytes(), original)
        for key in ('source_job', 'source_code', 'source_commit', 'source_files', 'source_output',
                    'storage_handoff', 'evaluation_handoff', 'cp_parity_handoff'):
            self.assertEqual(self.request[key], self.fixture.previous[key])

    def test_source_science_and_unknown_fields_cannot_change(self):
        self.make()
        for key, value in (('unknown_fallback', True), ('source_commit', '6' * 40),
                           ('source_output', self.request['chain_root']), ('source_job', {})):
            with self.subTest(key=key), self.admitted(), self.assertRaises(ValueError):
                continuation.validate(dict(self.request, **{key:value}))

    def test_allocation_cannot_expand_to_gpu4_or5_or_other_pbs_job(self):
        self.make()
        for changes in ({'GPU_arm':4}, {'GPU_arm':5}, {'scheduler':dict(self.request['scheduler'], job_id='129999.ECE-util1')},
                        {'scheduler':dict(self.request['scheduler'], minimum_remaining_seconds=100)},
                        {'scheduler':dict(self.request['scheduler'], native_minimum_remaining_seconds=100)}):
            with self.subTest(changes=changes), self.admitted(), self.assertRaises(ValueError):
                continuation.validate_scheduler(dict(self.request, **changes))

    def test_old_exact_supervisor_and_child_must_be_dead(self):
        self.make()
        for pid in (9090, 9091):
            with self.subTest(pid=pid), self.admitted(live_old=pid), self.assertRaisesRegex(RuntimeError, 'remains live'):
                continuation.validate(self.request)

    def test_new_allocation_process_and_physical_binding_must_remain_live(self):
        self.make()
        for options in ({'allocation_live':False}, {'affinity':[32, 33]},
                        {'environment':dict(PBS_JOBID='wrong', CUDA_VISIBLE_DEVICES=self.request['GPU_UUID'])},
                        {'environment':dict(PBS_JOBID='129528.ECE-util1', CUDA_VISIBLE_DEVICES='other_GPU')}):
            with self.subTest(options=options), self.admitted(**options), self.assertRaises(ValueError):
                continuation.validate_scheduler(self.request)

    def test_each_old_file_and_recursive_prior_claim_lease_is_sha_guarded(self):
        self.make()
        for path in (self.fixture.old/'status.json', Path(self.request['source_claim_path']),
                     *(Path(self.request[key]['lease']) for key in ('storage_handoff','evaluation_handoff','cp_parity_handoff'))):
            original = path.read_bytes(); path.write_text('{}', encoding='utf8')
            with self.subTest(path=str(path)), self.admitted(), self.assertRaises(ValueError):
                continuation.validate(self.request)
            path.write_bytes(original)

    def test_no_old_pin_scores_native_or_result_may_be_replayed(self):
        self.make()
        for name in ('pin.json','bank','native','prepare_current_bank.log'):
            path=self.fixture.old/name;path.write_text('CPU UNIT fake prior artifact',encoding='utf8')
            with self.subTest(name=name), self.admitted(), self.assertRaisesRegex(ValueError,'artifact'):
                continuation.validate(self.request)
            path.unlink()

    def test_profile_flag_is_exactly_one_matched_prepare_bank_only(self):
        self.make()
        with self.admitted():chain.validate_request(self.request)
        for stage_index in (0, 2, 3, 4):
            request=copy.deepcopy(self.request);request['stages'][stage_index]['command'].append('--context-depth1')
            with self.subTest(stage=stage_index), self.admitted(), self.assertRaisesRegex(ValueError,'only to matched bank'):
                chain.validate_request(request)
        request=copy.deepcopy(self.request);request['stages'][1]['command'].pop()
        with self.admitted(),self.assertRaisesRegex(ValueError,'stage changed'):chain.validate_request(request)

    def test_globally_derived_new_lease_cannot_be_replayed(self):
        self.make()
        with self.admitted(),self.assertRaisesRegex(ValueError,'another request/root'):
            chain.verify_source_claim(self.request,'8'*64)
        request=copy.deepcopy(self.request);request['context_allocation_continuation']['lease']+='another'
        with self.admitted(),self.assertRaisesRegex(ValueError,'globally derived'):continuation.validate(request)


if __name__=='__main__':unittest.main()
