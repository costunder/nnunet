"""CPU UNIT boundary fixtures only; no Torch, CUDA, GNN or nnUNet learning.

Mocks test admission failures and routing into the existing full scorer. They
are never production checkpoint/progress evidence.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import run_v24_completed_matched_scoring as worker


class CompletedScoringCPUUnit(unittest.TestCase):
    def setUp(self):
        self.actual_code = Path(worker.__file__).resolve().parents[1]
        self.temporary = tempfile.TemporaryDirectory(prefix='completed_scoring_CPU_UNIT_',
            dir=self.actual_code / 'outputs')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / 'new_scoring'
        self.root.mkdir()
        self.code = self.base / 'sealed_code_CPU_UNIT'
        self.request_path = self.root / 'request.json'
        if os.name == 'nt':
            # Windows sandbox forbids hardlinks. This explicit CPU UNIT stand-in
            # tests exclusive complete-byte publication only, not Linux link
            # atomicity. The real os.link is exercised unchanged on Linux.
            def exclusive_copy(source, target):
                with Path(target).open('xb') as stream:
                    stream.write(Path(source).read_bytes())
            link = patch.object(worker.os, 'link', side_effect=exclusive_copy)
            link.start()
            self.addCleanup(link.stop)

    def request(self, gpu=4):
        source = self.base / 'old_GNN'
        bank = self.base / 'historical_Basic'
        preserved = {str(source / name): 'a' * 64 for name in
            ('request.json', 'calibration.json', 'training/training_identity.json',
             'training/invocations.jsonl', 'training/checkpoint_latest.pt', 'training/checkpoint_best.pt')}
        preserved[str(bank / 'index.json')] = 'b' * 64
        return dict(format=worker.FORMAT, gpu=gpu, debug=False, full642_scoring=True,
            source_entries=642, CP_positions=82176, physical_patient_batch=4,
            physical_candidate_chunk=64 if gpu == 4 else 32, CPU_affinity=worker.AFFINITIES[gpu],
            source_output=str(source), source_code=str(self.base / 'old_science'),
            inventory_path=str(self.base / 'inventory.json'), input_cache=str(self.base / 'GT_free_cache'),
            baseline_preprocessed=str(self.base / 'baseline'), source_bank=str(bank),
            code=str(self.code), commit='c' * 40,
            code_files={name: 'd' * 64 for name in worker.REQUIRED_CODE_FILES},
            worker_sha256='e' * 64, preserved_source_hashes=preserved,
            stunet_checkpoint=str(self.base / 'official_STU.pt') if gpu == 4 else None,
            pin_path=str(self.root / 'own_best_pin.json'), output=str(self.root / 'bank'),
            direct_guard_module=str(self.root / 'direct_GPU45_access_guard.py'),
            direct_guard_sha256=worker.GUARD_SHA256,
            direct_access_proof=str(self.root / 'direct_GPU_access_proof.json'),
            direct_access_binding=str(self.root / 'direct_access_binding.json'),
            native_pending_reason='CPU UNIT: actual native storage not admitted')

    def validate(self, request):
        return worker.validate_request(request, self.request_path,
            self.code / 'tools/run_v24_completed_matched_scoring.py')

    def test_gpu4_and_gpu5_preserve_full_original_contract(self):
        for gpu in (4, 5):
            with self.subTest(gpu=gpu):
                self.assertEqual(self.validate(self.request(gpu)), self.root)

    def test_scale_or_assignment_changes_rejected(self):
        changes = dict(gpu=6, debug=True, source_entries=641, CP_positions=82175,
            physical_patient_batch=1, physical_candidate_chunk=32, CPU_affinity=[4, 5, 6, 7])
        for key, value in changes.items():
            with self.subTest(key=key):
                request = self.request()
                request[key] = value
                with self.assertRaises(ValueError):
                    self.validate(request)

    def test_cnn_does_not_load_stu_weights(self):
        request = self.request(5)
        request['stunet_checkpoint'] = str(self.base / 'other.pt')
        with self.assertRaisesRegex(ValueError, 'CNN/GAT'):
            self.validate(request)

    def test_original_results_and_old_pin_are_never_destinations(self):
        request = self.request()
        request['output'] = request['source_output']
        with self.assertRaises(ValueError):
            self.validate(request)
        request = self.request()
        request['source_output'] = str(self.root / 'old_GNN')
        with self.assertRaisesRegex(ValueError, 'separate'):
            self.validate(request)

    def test_omitted_scientific_code_or_source_checkpoint_rejected(self):
        for field, key in (('code_files', 'hiercp_v1x/v24_context_depth1.py'),
                ('preserved_source_hashes', str(self.base / 'old_GNN/training/checkpoint_best.pt'))):
            with self.subTest(field=field):
                request = self.request()
                del request[field][key]
                with self.assertRaises(ValueError):
                    self.validate(request)

    def test_changed_guard_or_manifest_traversal_rejected(self):
        request = self.request()
        request['direct_guard_sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            self.validate(request)
        request = self.request()
        request['code_files']['../other.py'] = 'f' * 64
        with self.assertRaises(ValueError):
            self.validate(request)

    def binding(self, request):
        self.request_path.write_text(json.dumps(request), encoding='utf8')
        guard_path = Path(request['direct_guard_module'])
        guard_path.write_text('CPU UNIT guard bytes, never executed', encoding='utf8')
        request['direct_guard_sha256'] = worker.sha(guard_path)
        proof_path = Path(request['direct_access_proof'])
        proof = dict(format=worker.GUARD_FORMAT, GPU_arm=4, CPU_affinity=worker.AFFINITIES[4], UID=41833,
            request=dict(path=str(self.request_path), sha256=worker.sha(self.request_path)),
            guard_sha256=request['direct_guard_sha256'])
        proof_path.write_text(json.dumps(proof), encoding='utf8')
        witness = lambda path: dict(path=str(path), sha256=worker.sha(path), stat=worker.identity(path))
        binding = dict(format=worker.BINDING_FORMAT, GPU_arm=4, UID=41833,
            CPU_affinity=worker.AFFINITIES[4], request=witness(self.request_path),
            proof=witness(proof_path), guard=witness(guard_path))
        Path(request['direct_access_binding']).write_text(json.dumps(binding), encoding='utf8')
        return proof, binding

    def test_direct_proof_must_bind_new_scoring_request_not_old_GNN(self):
        request = self.request()
        proof, binding = self.binding(request)
        with patch.object(worker, 'owned', side_effect=lambda path, **kwargs: Path(path)):
            self.assertEqual(len(worker.verify_binding(request, self.request_path)), 4)
            proof['request']['path'] = request['source_output'] + '/request.json'
            proof_path = Path(request['direct_access_proof'])
            proof_path.write_text(json.dumps(proof), encoding='utf8')
            binding['proof'].update(sha256=worker.sha(proof_path), stat=worker.identity(proof_path))
            Path(request['direct_access_binding']).write_text(json.dumps(binding), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'never the old GNN'):
                worker.verify_binding(request, self.request_path)

    def test_mutated_source_is_detected_at_safe_callback(self):
        path = self.base / 'protected.pt'
        path.write_bytes(b'CPU UNIT only')
        with patch.object(worker, 'owned', side_effect=lambda path, **kwargs: Path(path)):
            record = worker.verify_file(path, worker.sha(path))
            path.write_bytes(b'changed CPU UNIT only')
            with self.assertRaisesRegex(ValueError, 'changed'):
                worker.verify_unchanged([record])

    def test_exclusive_publish_keeps_old_result(self):
        path = self.root / 'complete.json'
        worker.publish(path, dict(original=True))
        with self.assertRaises(FileExistsError):
            worker.publish(path, dict(original=False))
        self.assertEqual(worker.read(path), dict(original=True))
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ['complete.json'])

    def execute_fixture(self, *, pin_error=None, incomplete=False):
        request = self.request()
        self.request_path.write_text(json.dumps(request), encoding='utf8')
        calls = []
        def pin(**kwargs):
            calls.append(('pin', kwargs))
            if pin_error:
                raise pin_error
            worker.publish(Path(kwargs['output']), dict(format='v24_completed_single_arm_GT_blind_best_v1',
                physical_GPU=4, completed_epochs=40, optimizer_updates=680, full_training=True,
                debug=False, source_output=request['source_output'], source_code=request['source_code'],
                selected=dict(gpu=4, best_record={'epoch': 12})))
        def score(**kwargs):
            calls.append(('score', kwargs))
            kwargs['deadline_guard']()
            path = self.root / 'bank/score_overlay.json'
            path.parent.mkdir()
            worker.publish(path, dict(format='v24_own_BEST_scores_on_unchanged_Basic_81case642source128positions_v1',
                complete=True, debug=False, complete_cases=81, complete_sources=641 if incomplete else 642,
                complete_positions=82176, pin=worker.read(request['pin_path']), full_P_context=True,
                recipient_GT_used_in_forward=False, production_optimizer_updates=0, model_and_RNG_unchanged=True,
                original_non_score_payloads_preserved=True, original_source_schedule_and_paste_contract_preserved=True,
                inference_profile=dict(format='v24_matched_context_liver_depth_gt1mm_inference_v1')))
            return path
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(worker, '__file__', str(self.code / 'tools/run_v24_completed_matched_scoring.py')))
            stack.enter_context(patch.object(worker, 'owned', side_effect=lambda path, **kwargs: Path(path)))
            stack.enter_context(patch.object(worker, 'verify_inputs', return_value=[]))
            stack.enter_context(patch.object(worker, 'load_guard', return_value=lambda: dict(CPU_UNIT=True)))
            stack.enter_context(patch.object(worker, 'load_scientific_api', return_value=(pin, score)))
            stack.enter_context(patch.object(worker, 'cuda_metrics', return_value=dict(CPU_UNIT=True)))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            if pin_error or incomplete:
                with self.assertRaises((ValueError, RuntimeError)):
                    worker.run(self.request_path)
            else:
                worker.run(self.request_path)
        return request, calls

    def test_pin_failure_never_starts_CUDA_scorer_or_running_receipt(self):
        _, calls = self.execute_fixture(pin_error=ValueError('Incomplete actual epoch39 source'))
        self.assertEqual([name for name, _ in calls], ['pin'])
        self.assertFalse((self.root / 'started.json').exists())
        self.assertFalse((self.root / 'complete.json').exists())
        self.assertEqual(worker.read(self.root / 'failed.json')['status'], 'FULL642_SCORING_FAILED')

    def test_completed_pin_routes_original_full_scorer_without_native_execution(self):
        request, calls = self.execute_fixture()
        self.assertEqual([name for name, _ in calls], ['pin', 'score'])
        score_args = calls[1][1]
        self.assertTrue(score_args['context_depth1'])
        self.assertEqual(score_args['output'], request['output'])
        self.assertEqual(score_args['source_bank'], request['source_bank'])
        completed = worker.read(self.root / 'complete.json')
        self.assertEqual(completed['completed_sources'], 642)
        self.assertEqual(completed['completed_CP_positions'], 82176)
        self.assertFalse(completed['full_native_training'])
        self.assertFalse(completed['full_native_evaluation'])
        self.assertEqual(completed['production_optimizer_updates'], 0)

    def test_partial_source_overlay_is_failure_never_complete(self):
        self.execute_fixture(incomplete=True)
        self.assertFalse((self.root / 'complete.json').exists())
        self.assertEqual(worker.read(self.root / 'failed.json')['status'], 'FULL642_SCORING_FAILED')

    def test_standalone_help_has_no_scientific_or_CUDA_import(self):
        completed = subprocess.run([sys.executable, '-B', str(Path(worker.__file__)), '--help'],
            cwd=self.base, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('--request', completed.stdout)
        completed = subprocess.run([sys.executable, '-B', '-c',
            "import sys; sys.path.insert(0, sys.argv[1]); from tools import run_v24_completed_matched_scoring; "
            "assert 'torch' not in sys.modules and 'numpy' not in sys.modules", str(self.actual_code)],
            cwd=self.base, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
