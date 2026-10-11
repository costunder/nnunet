"""DEBUG read-only CPU metadata integration, never actual training/scoring."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import psutil

from tools import watch_v24_direct_gnn_training as direct
from tools import watch_v24_gpu_pipeline as pipeline
from tests.test_watch_v24_matched_cp_scoring import event, process_fixture, scoring_fixture, write_json


RENDERER = Path(__file__).resolve().parents[1] / 'tools/watch_v24_training.py'


def direct_fixture(root, gpu=4, completed=True):
    root.mkdir()
    output = root / 'gnn'
    (output / 'training').mkdir(parents=True)
    source_code, code = str(root.parent / 'UNIT_original_science'), str(root.parent / 'UNIT_execution_code')
    latest = dict(raw_sha256='a' * 64, epoch=39, phase='full_validation', updates=663,
        history_epochs=list(range(1, 40)), active_u=128)
    source = dict(source_output=str(root.parent / 'UNIT_interrupted_original'), source_code=source_code,
        GPU_arm=gpu, latest=latest)
    cursor = dict(source=source, copied_files={}, created_at=1.)
    write_json(output / 'training_continuation.json', cursor)
    receipt = dict(format='v24_interrupted_exact_saved_GNN_continuation_v1', status='PREPARED_EXACT_SAVED_STATE',
        destination=str(output), execution_code=code, source=source, six_numerical_states_unchanged=True,
        scientific_request_unchanged=True, original_files_modified=False, production_optimizer_updates_performed=0,
        viewer_cursor=dict(path=str(output / 'training_continuation.json'),
            sha256=hashlib.sha256((output / 'training_continuation.json').read_bytes()).hexdigest()))
    write_json(output / 'interrupted_gnn_continuation.json', receipt)
    request = dict(format='v24_existing_direct_GPU_exact_continuation_request_v1', GPU_arm=gpu,
        output=str(output), original_epochs=40, physical_patient_batch=4, physical_candidate_batch=64 if gpu == 4 else 32,
        full_original_population=True, debug=False, PBS_owned_reservation=False, six_saved_states_unchanged=True,
        BEST_unchanged=True, code=code, source_output=source['source_output'], source_code=source_code,
        saved_checkpoint='a' * 64, checkpoint_epoch=39, checkpoint_phase='full_validation', checkpoint_updates=663,
        original_args=['--UNIT_scientific_fixture', 'DEBUG'])
    write_json(root / 'direct_training_request.json', request)
    command = ['UNIT_python', '-B', '-u', str(Path(code) / 'tools/run_v24_interrupted_gnn_continuation.py'),
        '--action', 'train', '--source-output', source['source_output'], '--source-code', source_code,
        '--interruption-proof', str(root / 'original_interruption_witness.json'),
        '--allocation-guard-module', str(root.parent / 'direct_GPU45_access_guard.py'),
        '--allocation-guard-sha256', 'b' * 64, '--allocation-proof', str(root / 'direct_GPU_access_proof.json'),
        '--allocation-proof-sha256', 'c' * 64, *request['original_args'], '--output', str(output)]
    registration = dict(format='v24_existing_direct_GPU_exact_continuation_registration_v1', GPU_arm=gpu,
        root=str(root), output=str(output), request_sha256=hashlib.sha256((root / 'direct_training_request.json').read_bytes()).hexdigest(),
        PBS_owned_reservation=False, six_saved_states_unchanged=True, old_results_written=False, signals_sent=False,
        source_checkpoint_sha256='a' * 64, resumed_epoch=39, resumed_phase='full_validation', resumed_updates=663,
        pid=70001, create_time=100., python='UNIT_python', command=command, guard_sha256='b' * 64, proof_sha256='c' * 64)
    write_json(root / 'direct_training_registration.json', registration)
    if completed:
        (output / 'training/invocations.jsonl').write_text(json.dumps(dict(status='COMPLETE', full_training=True,
            debug=False, completed_epochs=40, optimizer_updates=680, actual_CUDA=True)) + '\n', encoding='utf8')
    return request, registration


class CurrentPipelineCPU_DEBUG(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='GPU_pipeline_DEBUG_',
            dir=Path(__file__).resolve().parents[1] / 'outputs')
        self.addCleanup(self.folder.cleanup)
        self.base = Path(self.folder.name)
        self.arm, self.score = self.base / 'gpu4', self.base / 'score'
        self.direct_request, self.direct_reg = direct_fixture(self.arm)
        self.score_request, self.score_reg = scoring_fixture(self.score, 4, self.arm / 'gnn', self.direct_request['source_code'])
        self.renderer = direct.load_renderer(RENDERER, direct.sha(RENDERER))
        self.viewer = pipeline.PipelineViewer(self.renderer, self.arm, self.score)
        self.gnn_alive, self.score_alive = False, True
        self.native_reg, self.native_alive, self.native_child = None, False, False

    def process(self, pid=None):
        if pid == self.direct_reg['pid'] and self.gnn_alive:
            return process_fixture(self.direct_reg)
        if pid == self.score_reg['pid'] and self.score_alive:
            return process_fixture(self.score_reg)
        if self.native_reg and pid == self.native_reg['pid'] and self.native_alive:
            return process_fixture(self.native_reg)
        if pid == 70004 and self.native_child:
            state = pipeline.read(self.base / 'native/status.json')
            return process_fixture(dict(create_time=400., command=state['command']))
        raise psutil.NoSuchProcess(pid)

    def pin(self, **changes):
        value = dict(format='v24_completed_single_arm_GT_blind_best_v1', physical_GPU=4,
            source_output=str(self.arm / 'gnn'), source_code=self.direct_request['source_code'],
            full_training=True, completed_epochs=40, optimizer_updates=680)
        value.update(changes)
        write_json(self.score / 'own_best_pin.json', value)

    def complete_scoring(self):
        self.pin()
        (self.score / 'console.log').write_text(json.dumps(event(4, 642)) + '\n', encoding='utf8')
        write_json(self.score / 'complete.json', dict(status='FULL642_SCORING_COMPLETE_NATIVE_STORAGE_PENDING',
            completed_sources=642, completed_CP_positions=82176, full_native_training=False,
            full_native_evaluation=False, native_pending_reason='UNIT native capacity unavailable'))

    def native(self, status='WAITING_FOR_NATIVE_STORAGE', *, child=False, new=False):
        root = self.base / 'native'
        root.mkdir()
        command = ['UNIT_python', '-B', '-u', 'UNIT_native_worker', '--native', str(root / 'native/native.json'), '--gpu', '4']
        request = dict(chain_root=str(root), GPU_arm=4, source_output=str(self.arm / 'gnn'),
            source_code=self.direct_request['source_code'], stages=[dict(name='train', action='train', command=command)])
        if new:
            request.update(format=pipeline.NATIVE_REQUEST,
                storage_admission=str(root.parent / 'storage_admission.json'),
                storage_binding=str(root.parent / 'storage_binding.json'),
                arm_roots=dict(gpu4=str(root), gpu5=str(root.parent / 'native5'), gpu6=str(root.parent / 'native6')))
        write_json(root / 'request.json', request)
        digest = hashlib.sha256((root / 'request.json').read_bytes()).hexdigest()
        self.native_reg = dict(pid=70003, create_time=300., root=str(root), request_sha256=digest,
            command=['UNIT_python', '-B', '-u', 'watch_v24_gnn_to_nnunet.py', '--request', str(root / 'request.json')])
        if new:
            self.native_reg['format'] = pipeline.NATIVE_REGISTRATION
        write_json(root / 'registration.json', self.native_reg)
        state = dict(format='v24_completed_gnn_to_native_chain_status_v1', GPU_arm=4,
            request_sha256=digest, source_output=str(self.arm / 'gnn'), supervisor_pid=70003,
            supervisor_create_time=300., status=status, stage='train' if child else None,
            waiting_reason='UNIT measured native reserve insufficient', child_pid=70004 if child else None,
            child_create_time=400. if child else None, command=command if child else None,
            stage_log=str(root / 'train.log') if child else None, stages_completed=[])
        if new:
            state['format'] = pipeline.NATIVE_STATUS
        write_json(root / 'status.json', state)
        if child:
            (root / 'train.log').write_text('Epoch 12\ntrain_loss -0.1\nval_loss -0.2\nPseudo dice [0.4]\n', encoding='utf8')
        self.viewer.native_root = root
        self.native_alive, self.native_child = True, child

    def native_completion(self):
        root = self.base / 'native'
        request = pipeline.read(root / 'request.json')
        admission_path = Path(request['storage_admission'])
        write_json(admission_path, dict(format='v24_matched_Basic642_storage_admission_v1',
            arm_roots=request['arm_roots']))
        admission_sha = pipeline.checksum(admission_path)
        storage = dict(path=str(admission_path), sha256=admission_sha, stat=pipeline._file_stat(admission_path))
        write_json(Path(request['storage_binding']), dict(format='v24_scored_native_storage_ready_v1',
            admission=storage, arm_roots=request['arm_roots'], model_data_scale_preserved=True))
        native_root = root / 'native'
        native_root.mkdir()
        bank = self.score / 'bank'
        bank.mkdir()
        overlay = bank / 'score_overlay.json'
        write_json(overlay, dict(pin=pipeline.read(self.score / 'own_best_pin.json')))
        bank_sha = pipeline.checksum(overlay)
        fold = native_root / 'nnUNet_results/UNIT_dataset/UNIT_trainer/fold_0'
        fold.mkdir(parents=True)
        cases = ['UNIT_val_' + str(i).zfill(3) for i in range(26)]
        native = dict(format='v24_matched_Basic642_native_v1', root=str(native_root), physical_GPU=4,
            bank=str(overlay), bank_sha256=bank_sha, epochs=250, train_cases=105, validation_cases=26, debug=False,
            physical_batch=2, patch_size=[128] * 3, cp_probability=.5, source_entries=642, candidates_per_source=128,
            storage_admission=str(admission_path), storage_admission_sha256=admission_sha,
            split=dict(outer_val=cases), checkpoint_coordination=dict(folds=dict(gpu4=str(fold))))
        write_json(native_root / 'native.json', native)
        receipt_path = fold / 'best_validation_complete.json'
        receipt = dict(format='v24_native_best_loaded_full26_validation_v1', checkpoint_name='checkpoint_best.pth',
            checkpoint_path=str(fold / 'checkpoint_best.pth'), checkpoint_sha256='b' * 64, native_fold=str(fold),
            bank_sha256=bank_sha, validation_cases=cases, complete=True, native_validation_returned_successfully=True,
            original_checkpoint_load_called=True, original_full_validation_called=True, final_checkpoint_fallback=False,
            GT_passed_to_predictor=False, CP_at_inference=False,
            predictions_sha256={case:'c' * 64 for case in cases}, summary_sha256='d' * 64)
        write_json(receipt_path, receipt)
        complete_path = native_root / 'training_complete_UNIT_DEBUG.json'
        complete = dict(format=native['format'], epochs=250, GPU=4, checkpoint=str(fold / 'checkpoint_final.pth'),
            checkpoint_sha256='a' * 64, checkpoint_role='full250_training_completion_witness', bank_sha256=bank_sha,
            evaluation_checkpoint_name='checkpoint_best.pth', evaluation_checkpoint_path=receipt['checkpoint_path'],
            evaluation_checkpoint_sha256=receipt['checkpoint_sha256'], best_validation_receipt=str(receipt_path),
            best_validation_receipt_sha256=pipeline.checksum(receipt_path))
        write_json(complete_path, complete)
        stage_path = root / 'matched_storage_train.json'
        write_json(stage_path, dict(format='v24_matched_basic_native_stage_v1', status='COMPLETE', action='train',
            storage_admission_sha256=admission_sha, model_data_scale_preserved=True, cached_inputs_written=False))
        state = pipeline.read(root / 'status.json')
        state.update(full_native_training_completed=True, full_native_evaluation_completed=True, storage=storage,
            completion_proof=dict(action='train', physical_GPU=4, path=str(complete_path), sha256=pipeline.checksum(complete_path),
                matched_stage_receipt=str(stage_path), matched_stage_receipt_sha256=pipeline.checksum(stage_path)))
        write_json(root / 'status.json', state)
        return complete_path, receipt_path

    def observed(self):
        with patch.object(pipeline.psutil, 'Process', side_effect=self.process):
            return self.viewer.snapshot()

    def test_real_direct_completion_switches_to_CPU_pin_then_full642_scoring(self):
        self.gnn_alive = True
        self.assertEqual(self.observed()['active_stage'], 'gnn')
        self.gnn_alive = False
        pending = self.observed()
        self.assertEqual(pending['status'], 'CPU_PIN_PENDING')
        self.assertEqual(pending['active_stage'], 'scoring')
        self.assertEqual(pending['gnn']['optimizer_updates'], 680)
        self.pin()
        (self.score / 'console.log').write_text(json.dumps(event(4, 20)) + '\n', encoding='utf8')
        observed = self.observed()
        self.assertEqual(observed['scoring']['CP_done'], 2560)
        self.assertTrue(observed['scoring']['own_BEST_pin_verified'])
        self.assertFalse(observed['full_native_training'])
        self.assertIn('storage budget unavailable', pipeline.display(observed, self.renderer))

    def test_dead_GNN_without_full_receipt_cannot_advance(self):
        path = self.arm / 'gnn/training/invocations.jsonl'
        path.write_text(json.dumps(dict(status='PAUSED', debug=False)) + '\n', encoding='utf8')
        observed = self.observed()
        self.assertEqual(observed['active_stage'], 'gnn')
        self.assertEqual(observed['status'], 'PAUSED')
        self.assertIsNone(observed['scoring'])

    def test_dead_scoring_PID_without_receipt_is_not_CPU_pin_pending(self):
        self.score_alive = False
        observed = self.observed()
        self.assertEqual(observed['status'], 'EXITED_WITHOUT_COMPLETION')
        self.assertFalse(observed['scoring']['alive'])

    def test_completed_scoring_and_absent_native_remain_explicitly_pending(self):
        self.complete_scoring()
        self.score_alive = False
        self.viewer.native_root = self.base / 'future_native'
        observed = self.observed()
        self.assertEqual(observed['scoring']['source_done'], 642)
        self.assertEqual(observed['native']['status'], 'WAITING_NATIVE_METADATA')
        self.assertFalse(observed['full_native_training'])
        self.assertIn('actual request', observed['native']['native_pending_reason'].replace('실제', 'actual'))

    def test_cross_GPU_or_original_interrupted_source_pin_cannot_connect(self):
        for changes in (dict(physical_GPU=6), dict(source_output=self.direct_request['source_output'])):
            self.pin(**changes)
            observed = self.observed()
            self.assertEqual(observed['status'], 'READ_ERROR')
            self.assertIn('BEST source/GPU differs', observed['read_problem']['detail'])

    def test_source_request_points_to_original_interrupted_path_rejected(self):
        self.score_request['source_output'] = self.direct_request['source_output']
        write_json(self.score / 'request.json', self.score_request)
        self.score_reg['request_sha256'] = hashlib.sha256((self.score / 'request.json').read_bytes()).hexdigest()
        write_json(self.score / 'registration.json', self.score_reg)
        observed = self.observed()
        self.assertEqual(observed['status'], 'READ_ERROR')
        self.assertIn('current GNN', observed['read_problem']['detail'])

    def test_partial_pin_preserves_last_display_persistent_error_and_recovery(self):
        self.pin()
        (self.score / 'console.log').write_text(json.dumps(event(4, 4)) + '\n', encoding='utf8')
        self.assertTrue(self.observed()['snapshot_fresh'])
        path = self.score / 'own_best_pin.json'
        path.write_text('{', encoding='utf8')
        with patch.object(pipeline.time, 'monotonic', return_value=100.):
            pending = self.observed()
        self.assertEqual(pending['status'], 'READ_PENDING')
        self.assertFalse(pending['snapshot_fresh'])
        self.assertEqual(pending['scoring']['CP_done'], 512)
        with patch.object(pipeline.time, 'monotonic', return_value=111.):
            error = self.observed()
        self.assertEqual(error['status'], 'READ_ERROR')
        self.assertIn(str(path), pipeline.display(error, self.renderer))
        self.pin()
        self.assertTrue(self.observed()['snapshot_fresh'])

    def test_native_storage_wait_and_dead_controller_are_actual_states(self):
        self.complete_scoring()
        self.native()
        observed = self.observed()
        self.assertEqual(observed['active_stage'], 'native')
        self.assertEqual(observed['status'], 'WAITING_FOR_NATIVE_STORAGE')
        self.assertFalse(observed['native']['native_training_started'])
        self.assertIn('measured native reserve', pipeline.display(observed, self.renderer))
        self.native_alive = False
        self.assertEqual(self.observed()['status'], 'EXITED_WITHOUT_COMPLETION')

    def test_native_only_reports_training_for_exact_live_declared_child(self):
        self.complete_scoring()
        self.native(status='RUNNING', child=True)
        observed = self.observed()
        self.assertTrue(observed['native']['native_training_started'])
        self.assertEqual(observed['native']['epoch'], 13)
        shown = pipeline.display(observed, self.renderer)
        self.assertIn('실행 중', shown)
        self.assertNotIn('후속 nnU-Net: 아직 시작 전', shown)
        self.native_child = False
        self.assertFalse(self.observed()['native']['native_training_started'])

    def test_native_cross_source_and_unproven_COMPLETE_never_claim_full_training(self):
        self.complete_scoring()
        self.native(status='COMPLETE')
        state = pipeline.read(self.base / 'native/status.json')
        state['full_native_training_completed'] = True
        write_json(self.base / 'native/status.json', state)
        observed = self.observed()
        self.assertFalse(observed['full_native_training'])
        self.assertTrue(observed['native']['native_training_completion_reported'])
        state['source_output'] = self.direct_request['source_output']
        write_json(self.base / 'native/status.json', state)
        self.assertEqual(self.observed()['status'], 'READ_ERROR')

    def test_native_preregistration_is_legal_during_GNN_and_CP_wait(self):
        self.native(status='WAITING_FOR_FULL_GNN', new=True)
        self.gnn_alive = True
        observed = self.observed()
        self.assertEqual(observed['active_stage'], 'gnn')
        self.assertEqual(observed['native']['status'], 'WAITING_FOR_FULL_GNN')
        self.gnn_alive = False
        self.pin()
        (self.score / 'console.log').write_text(json.dumps(event(4, 20)) + '\n', encoding='utf8')
        state = pipeline.read(self.base / 'native/status.json')
        state['status'] = 'WAITING_FOR_FULL642_SCORING'
        write_json(self.base / 'native/status.json', state)
        observed = self.observed()
        self.assertEqual(observed['active_stage'], 'scoring')
        self.assertEqual(observed['scoring']['source_done'], 20)
        self.assertEqual(observed['native']['status'], 'WAITING_FOR_FULL642_SCORING')
        self.assertFalse(observed['native']['native_training_started'])

    def test_native_running_before_full642_is_rejected(self):
        self.pin()
        (self.score / 'console.log').write_text(json.dumps(event(4, 20)) + '\n', encoding='utf8')
        self.native(status='RUNNING', child=True, new=True)
        observed = self.observed()
        self.assertEqual(observed['status'], 'READ_ERROR')
        self.assertIn('before confirmed full642', observed['read_problem']['detail'])

    def test_metadata_publication_wait_keeps_current_CP_active(self):
        self.pin()
        (self.score / 'console.log').write_text(json.dumps(event(4, 20)) + '\n', encoding='utf8')
        self.native(status='WAITING_METADATA_PUBLICATION', new=True)
        observed = self.observed()
        self.assertEqual(observed['active_stage'], 'scoring')
        self.assertEqual(observed['native']['status'], 'WAITING_METADATA_PUBLICATION')
        self.assertFalse(observed['native']['native_training_started'])

    def test_late_storage_binding_requires_exact_admission_SHA_stat_and_arm(self):
        self.complete_scoring()
        self.native(status='COMPLETE', new=True)
        self.native_completion()
        self.assertTrue(self.observed()['full_native_training'])
        request = pipeline.read(self.base / 'native/request.json')
        path = Path(request['storage_binding'])
        original = path.read_bytes()
        binding = pipeline.read(path)
        for changes in (dict(sha256='e' * 64), dict(stat=[0] * 5), dict(path=str(self.base / 'other_admission.json'))):
            binding['admission'].update(changes)
            write_json(path, binding)
            self.assertEqual(self.observed()['status'], 'READ_ERROR')
            path.write_bytes(original)
            binding = pipeline.read(path)

    def test_new_native_completion_requires_real_full250_and_BEST26_metadata(self):
        self.complete_scoring()
        self.native(status='COMPLETE', new=True)
        self.assertEqual(self.observed()['status'], 'READ_ERROR')
        self.native_completion()
        self.native_alive = False
        observed = self.observed()
        self.assertTrue(observed['full_native_training'])
        self.assertTrue(observed['full_native_evaluation'])
        self.assertTrue(observed['native']['completion_metadata_verified'])
        self.assertIn('bytes not reread', observed['native']['completion_verification_scope'])

    def test_new_native_completion_refuses_FINAL_evaluation_or_partial26(self):
        self.complete_scoring()
        self.native(status='COMPLETE', new=True)
        complete_path, receipt_path = self.native_completion()
        receipt = pipeline.read(receipt_path)
        receipt['checkpoint_name'] = 'checkpoint_final.pth'
        write_json(receipt_path, receipt)
        complete = pipeline.read(complete_path)
        complete['best_validation_receipt_sha256'] = pipeline.checksum(receipt_path)
        write_json(complete_path, complete)
        state = pipeline.read(self.base / 'native/status.json')
        state['completion_proof']['sha256'] = pipeline.checksum(complete_path)
        write_json(self.base / 'native/status.json', state)
        self.assertEqual(self.observed()['status'], 'READ_ERROR')
        receipt['checkpoint_name'] = 'checkpoint_best.pth'
        receipt['validation_cases'] = receipt['validation_cases'][:-1]
        write_json(receipt_path, receipt)
        complete['best_validation_receipt_sha256'] = pipeline.checksum(receipt_path)
        write_json(complete_path, complete)
        state['completion_proof']['sha256'] = pipeline.checksum(complete_path)
        write_json(self.base / 'native/status.json', state)
        self.assertEqual(self.observed()['status'], 'READ_ERROR')

    def test_malformed_native_split_is_visible_READ_ERROR_without_viewer_crash(self):
        self.complete_scoring()
        self.native(status='COMPLETE', new=True)
        self.native_completion()
        path = self.base / 'native/native/native.json'
        record = pipeline.read(path)
        record['split']['outer_val'][0] = {'bad': 'shape'}
        write_json(path, record)
        observed = self.observed()
        self.assertEqual(observed['status'], 'READ_ERROR')
        self.assertIn('BEST/full26', observed['read_problem']['detail'])
        self.assertFalse(observed['full_native_training'])

    def test_CLI_json_is_one_snapshot_and_CtrlC_only_returns(self):
        args = ['--arm-root', str(self.arm), '--renderer', str(RENDERER), '--renderer-sha', direct.sha(RENDERER),
            '--scoring-root', str(self.score), '--no-clear']
        self.pin()
        with patch.object(pipeline.psutil, 'Process', side_effect=self.process), patch('sys.stdout', new=io.StringIO()) as out:
            pipeline.main(args + ['--json'])
        self.assertEqual(json.loads(out.getvalue())['GPU'], 4)
        with patch.object(pipeline.PipelineViewer, 'snapshot', side_effect=KeyboardInterrupt), patch('sys.stdout', new=io.StringIO()) as out:
            pipeline.main(args)
        self.assertIn('뷰어만 종료', out.getvalue())


class CompletedGPU6PipelineCPU_DEBUG(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='GPU6_pipeline_DEBUG_', dir=RENDERER.parents[1] / 'outputs')
        self.addCleanup(self.folder.cleanup)
        self.base = Path(self.folder.name)
        self.output, self.score = self.base / 'completed_gnn', self.base / 'score'
        (self.output / 'training').mkdir(parents=True)
        self.invocation = dict(status='COMPLETE', full_training=True, debug=False, completed_epochs=40,
            optimizer_updates=680, actual_CUDA=True, connected_parameter_tensors=537, expected_parameter_tensors=537)
        (self.output / 'training/invocations.jsonl').write_text(json.dumps(self.invocation) + '\n', encoding='utf8')
        self.request, self.registration = scoring_fixture(self.score, 6)
        # The actual old GPU6 scoring request binds its source only through pin_path.
        del self.request['source_output'], self.request['source_code']
        write_json(self.score / 'request.json', self.request)
        self.registration['request_sha256'] = pipeline.checksum(self.score / 'request.json')
        write_json(self.score / 'registration.json', self.registration)
        self.pin = dict(format='v24_completed_single_arm_GT_blind_best_v1', physical_GPU=6, source_output=str(self.output),
            source_code=str(self.base / 'UNIT_scientific_code'), full_training=True, debug=False,
            completed_epochs=40, optimizer_updates=680,
            source_artifacts_sha256={'training/invocations.jsonl':pipeline.checksum(self.output / 'training/invocations.jsonl')})
        write_json(self.score / 'own_best_pin.json', self.pin)
        (self.score / 'console.log').write_text(json.dumps(event(6, 300)) + '\n', encoding='utf8')
        self.renderer = direct.load_renderer(RENDERER, direct.sha(RENDERER))
        self.viewer = pipeline.PipelineViewer(self.renderer, None, self.score, gnn_root=self.output, gpu=6)

    def observed(self):
        with patch.object(pipeline.psutil, 'Process', return_value=process_fixture(self.registration)):
            return self.viewer.snapshot()

    def test_actual_GPU6_completed_invocation_and_own_pin_connect_to_scoring(self):
        observed = self.observed()
        self.assertEqual(observed['GPU'], 6)
        self.assertEqual(observed['gnn']['optimizer_updates'], 680)
        self.assertEqual(observed['scoring']['source_done'], 300)
        self.assertFalse(observed['full_native_training'])

    def test_GPU6_source_hash_or_incomplete_invocation_cannot_claim_complete(self):
        self.invocation['completed_epochs'] = 39
        (self.output / 'training/invocations.jsonl').write_text(json.dumps(self.invocation) + '\n', encoding='utf8')
        self.assertEqual(self.observed()['status'], 'READ_ERROR')
        self.invocation['completed_epochs'] = 40
        (self.output / 'training/invocations.jsonl').write_text(json.dumps(self.invocation) + '\n ', encoding='utf8')
        self.assertEqual(self.observed()['status'], 'READ_ERROR')

    def test_GPU6_pin_cannot_select_another_arm_or_source(self):
        self.pin['physical_GPU'] = 4
        write_json(self.score / 'own_best_pin.json', self.pin)
        self.assertEqual(self.observed()['status'], 'READ_ERROR')
        self.pin['physical_GPU'] = 6
        self.pin['source_output'] = str(self.base / 'different_gnn')
        write_json(self.score / 'own_best_pin.json', self.pin)
        self.assertEqual(self.observed()['status'], 'READ_ERROR')

    def test_GPU6_cli_requires_explicit_arm_and_json_returns_actual_GPU(self):
        args = ['--gnn-root', str(self.output), '--renderer', str(RENDERER), '--renderer-sha', direct.sha(RENDERER),
            '--scoring-root', str(self.score), '--json']
        with patch.object(pipeline.psutil, 'Process', return_value=process_fixture(self.registration)), patch('sys.stdout', new=io.StringIO()) as out:
            pipeline.main(args + ['--gpu', '6'])
        self.assertEqual(json.loads(out.getvalue())['GPU'], 6)
        with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            pipeline.main(args)


if __name__ == '__main__':
    unittest.main()
