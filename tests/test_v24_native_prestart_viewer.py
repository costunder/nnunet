"""CPU DEBUG real JSON/receipt and mocked OS actors for prestart observation.

No production training/scoring, model imports, remote connections or signals.
The Windows fixture bypasses only Unix path ownership, not metadata validators.
"""
import ast
import copy
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import watch_v24_native_prestart_guard as viewer
from tools import watch_v24_training as renderer


class NativePrestartViewerCPU_DEBUG(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='native_prestart_viewer_CPU_DEBUG_', dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).absolute()
        self.path = self.root / 'adoption/request.json'
        self.old_path = self.root / 'original/request.json'
        self.code = self.root / 'code_C_DEBUG'
        self.source = self.root / 'source/gnn'
        self.resource = dict(format=viewer.pipeline.NATIVE_REQUEST, GPU_arm=4,
            chain_root=str(self.root / 'group/gpu4'), source_output=str(self.source),
            source_code=str(self.root / 'original_GNN_code_DEBUG'), scoring_root=str(self.root / 'scoring'),
            epochs=250, physical_batch=2, train_cases=105, validation_cases=26,
            source_entries=642, candidates_per_source=128, evaluation_checkpoint='checkpoint_best.pth')
        self.original = self.resource
        self.request = dict(format=viewer.REQUEST, GPU_arm=4, recovery_root=str(self.path.parent),
            python='/DEBUG/nnunet/bin/python3.10', code=str(self.code),
            resource_contract=copy.deepcopy(self.resource), stages=[])
        self.command = [self.request['python'], '-B', '-u',
            str(self.code / 'tools/run_v24_native_prestart_guard.py'), 'watch', '--request', str(self.path)]
        self.score_command = ['DEBUG_python', '-B', '-u', 'DEBUG_original_CP_worker', '--request', '/DEBUG/scoring/request.json']
        self.native_command = ['DEBUG_python', '-B', '-u', 'DEBUG_fixed_C_native_worker', 'train', '--native', '/DEBUG/native.json', '--gpu', '4']
        self.request['scoring_actor'] = dict(pid=123, create_time=2., command=self.score_command, UID=41833)
        self.request['stages'] = [dict(name='prepare_native', action='prepare-native', command=['DEBUG_prepare']),
            dict(name='calibrate_native', action='calibrate-native', command=['DEBUG_calibrate']),
            dict(name='train', action='train', command=self.native_command)]
        self.registration = dict(pid=999, create_time=1., UID=41833, command=self.command)
        self.lease = copy.deepcopy(self.registration)
        self.state = dict(format=viewer.STATUS, GPU_arm=4, source_output=str(self.source),
            supervisor_pid=999, supervisor_create_time=1., status='RUNNING', stage='score',
            child_pid=123, child_create_time=2., command=self.score_command,
            stage_log=str(self.path.parent / 'score.log'), child_process_tree_RSS_bytes=5 * 2**30)
        self.base = dict(GPU=4, status='EXITED_WITHOUT_COMPLETION', active_stage='scoring', snapshot_fresh=True,
            gnn=dict(job_root=str(self.source), full_training_completed=True, epoch=40, epochs=40,
                completed_epochs=40, optimizer_updates=680),
            scoring=dict(status='EXITED_WITHOUT_COMPLETION', alive=True, source_done=611, source_total=642,
                CP_done=611 * 128, CP_total=82176),
            native=dict(status='EXITED_WITHOUT_COMPLETION', native_training_started=False),
            native_pending_reason='DEBUG original controller exited',
            full_native_training=False, full_native_evaluation=False,
            read_problem=dict(detail='DEBUG preserved old dead-parent status'))
        self.os_processes = {999:dict(birth=1., command=self.command, UID=41833),
            123:dict(birth=2., command=self.score_command, UID=41833)}
        self.log = Mock(return_value=[])
        self.renderer = SimpleNamespace(log_tail=self.log, parse_nnunet_lines=renderer.parse_nnunet_lines)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(viewer.pipeline, 'input_path', side_effect=Path))
        self.stack.enter_context(patch.object(viewer.direct, 'owned', side_effect=lambda path:Path(path).stat()))
        self.stack.enter_context(patch.object(viewer.scoring.os, 'getuid', return_value=41833, create=True))
        self.stack.enter_context(patch.object(viewer.scoring.psutil, 'Process', side_effect=self.process))
        self.publish_all()

    @staticmethod
    def write(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding='utf8')

    def publish_all(self):
        self.write(self.old_path, self.original)
        self.request['resource_contract'] = copy.deepcopy(self.resource)
        self.request['original_request'] = dict(path=str(self.old_path), sha256=viewer.direct.sha(self.old_path),
            stat=viewer.pipeline._file_stat(self.old_path))
        self.write(self.path, self.request)
        digest = viewer.direct.sha(self.path)
        for record in (self.registration, self.lease, self.state):
            record['request_sha256'] = digest
        self.write(self.path.parent / 'registration.json', self.registration)
        self.write(self.path.parent / 'lease.json', self.lease)
        self.write(self.path.parent / 'status.json', self.state)

    def process(self, pid):
        row = self.os_processes.get(pid)
        if row is None:
            raise viewer.scoring.psutil.NoSuchProcess(pid)
        return SimpleNamespace(create_time=lambda:row['birth'], cmdline=lambda:row['command'],
            uids=lambda:SimpleNamespace(real=row['UID']), status=lambda:row.get('status', 'sleeping'),
            memory_info=lambda:SimpleNamespace(rss=2 * 2**30))

    def snapshot(self):
        return viewer.guard_snapshot(copy.deepcopy(self.base), self.renderer, self.path)

    def gpu5_early(self):
        self.resource['GPU_arm'] = 5
        self.resource['chain_root'] = str(self.root / 'group/gpu5')
        self.original = dict(format=viewer.early_view.REQUEST,
            target_epoch=27, full_GNN_training=False, resource_contract=self.resource)
        self.request['GPU_arm'] = self.base['GPU'] = self.state['GPU_arm'] = 5
        self.base['gnn'].update(epoch=27, completed_epochs=27, optimizer_updates=459, full_training_completed=False)
        self.publish_all()

    def train(self):
        self.base['scoring'].update(status='SCORING_COMPLETE_NATIVE_PENDING', source_done=642, CP_done=82176, alive=False)
        self.state.update(stage='train', command=self.native_command, stage_log=str(self.path.parent / 'train.log'))
        self.os_processes[123]['command'] = self.native_command
        self.publish_all()

    def completion(self):
        self.train()
        self.state.update(status='COMPLETE', full_native_training_completed=True, full_native_evaluation_completed=True)
        group = Path(self.resource['chain_root']).parent
        arms = {f'gpu{gpu}':str(group / f'gpu{gpu}') for gpu in (4, 5, 6)}
        self.resource.update(arm_roots=arms, storage_admission=str(group / 'storage_admission.json'),
            storage_binding=str(group / 'storage_binding.json'))
        self.write(self.resource['storage_admission'], dict(format='v24_matched_Basic642_storage_admission_v1', arm_roots=arms))
        admission = Path(self.resource['storage_admission'])
        self.state['storage'] = dict(path=str(admission), stat=viewer.pipeline._file_stat(admission), sha256=viewer.direct.sha(admission))
        self.write(self.resource['storage_binding'], dict(format='v24_scored_native_storage_ready_v1',
            arm_roots=arms, model_data_scale_preserved=True, admission=self.state['storage']))
        arm = Path(self.resource['chain_root'])
        native_root = arm / 'native'
        fold = native_root / 'nnUNet_results/DEBUG_dataset/DEBUG_trainer/fold_0'
        bank = self.root / 'scoring/bank/score_overlay.json'
        self.write(bank, dict(pin=dict(source_output=str(self.source), source_code=self.resource['source_code'],
            physical_GPU=self.request['GPU_arm'], full_training=self.request['GPU_arm'] == 4,
            completed_epochs=40 if self.request['GPU_arm'] == 4 else 27)))
        bank_sha = viewer.direct.sha(bank)
        cases = [f'DEBUG_val_{i:03}' for i in range(26)]
        native = dict(format='v24_matched_Basic642_native_v1', root=str(native_root), physical_GPU=self.request['GPU_arm'],
            bank=str(bank), bank_sha256=bank_sha, epochs=250, train_cases=105, validation_cases=26,
            debug=False, physical_batch=2, patch_size=[128] * 3, cp_probability=.5,
            source_entries=642, candidates_per_source=128, storage_admission=str(admission),
            storage_admission_sha256=self.state['storage']['sha256'], split=dict(outer_val=cases),
            checkpoint_coordination=dict(folds={f'gpu{self.request["GPU_arm"]}':str(fold)}))
        self.write(native_root / 'native.json', native)
        receipt = dict(format='v24_native_best_loaded_full26_validation_v1', checkpoint_name='checkpoint_best.pth',
            checkpoint_path=str(fold / 'checkpoint_best.pth'), checkpoint_sha256='b' * 64,
            native_fold=str(fold), bank_sha256=bank_sha, validation_cases=cases, complete=True,
            native_validation_returned_successfully=True, original_checkpoint_load_called=True,
            original_full_validation_called=True, final_checkpoint_fallback=False, GT_passed_to_predictor=False,
            CP_at_inference=False, predictions_sha256={case:'c' * 64 for case in cases}, summary_sha256='d' * 64)
        self.best_path = fold / 'best_validation_complete.json'
        self.write(self.best_path, receipt)
        self.complete_path = native_root / 'training_complete_CPU_DEBUG.json'
        self.complete_record = dict(format=native['format'], epochs=250, GPU=self.request['GPU_arm'],
            checkpoint=str(fold / 'checkpoint_final.pth'), checkpoint_sha256='a' * 64,
            checkpoint_role='full250_training_completion_witness', bank_sha256=bank_sha,
            evaluation_checkpoint_name='checkpoint_best.pth', evaluation_checkpoint_path=receipt['checkpoint_path'],
            evaluation_checkpoint_sha256=receipt['checkpoint_sha256'], best_validation_receipt=str(self.best_path),
            best_validation_receipt_sha256=viewer.direct.sha(self.best_path))
        self.write(self.complete_path, self.complete_record)
        stage_path = arm / 'matched_storage_train.json'
        self.write(stage_path, dict(format='v24_matched_basic_native_stage_v1', status='COMPLETE', action='train',
            storage_admission_sha256=self.state['storage']['sha256'], model_data_scale_preserved=True, cached_inputs_written=False))
        self.state['completion_proof'] = dict(action='train', physical_GPU=self.request['GPU_arm'], path=str(self.complete_path),
            sha256=viewer.direct.sha(self.complete_path), matched_stage_receipt=str(stage_path), matched_stage_receipt_sha256=viewer.direct.sha(stage_path))
        self.publish_all()

    def test_own_GPU4_full40_scoring_uses_actual_child_and_overrides_old_dead_parent(self):
        result = self.snapshot()
        self.assertEqual(result['status'], 'SCORING_PROCESS_ALIVE')
        self.assertEqual(result['active_stage'], 'scoring')
        self.assertEqual((result['scoring']['source_done'], result['scoring']['CP_done']), (611, 78208))
        self.assertTrue(result['native']['controller_alive'])
        self.assertTrue(result['native']['child_alive'])
        self.assertNotIn('read_problem', result)
        self.assertEqual(result['gnn'], self.base['gnn'])
        self.assertFalse(result['full_native_training'])
        self.assertFalse(result['native']['native_training_started'])
        self.log.assert_not_called()

    def test_own_GPU5_preserves_actual_early27_without_full40_projection(self):
        self.gpu5_early()
        result = self.snapshot()
        self.assertEqual(result['GPU'], 5)
        self.assertEqual(result['gnn']['completed_epochs'], 27)
        self.assertEqual(result['gnn']['optimizer_updates'], 459)
        self.assertFalse(result['gnn']['full_training_completed'])
        self.assertEqual(result['status'], 'SCORING_PROCESS_ALIVE')
        self.base['gnn']['full_training_completed'] = True
        with self.assertRaisesRegex(ValueError, 'early27'):
            self.snapshot()

    def test_native_without_first_epoch_log_reports_unknown_not_zero_or_completion(self):
        self.train()
        result = self.snapshot()
        self.assertEqual(result['active_stage'], 'native')
        self.assertTrue(result['native']['native_training_started'])
        self.assertEqual(result['native']['epochs'], 250)
        self.assertEqual(result['native']['RSS_GiB'], 5)
        self.assertEqual(result['native']['RAM_limit_GiB'], 48)
        self.assertNotIn('epoch', result['native'])
        self.assertNotIn('completed_epochs', result['native'])
        self.assertFalse(result['full_native_training'])
        self.assertFalse(result['full_native_evaluation'])
        self.log.assert_called_once_with(self.path.parent / 'train.log', 65536)

    def test_actual_nnunet_log_metrics_retain_native_epoch_and_pseudo_dice(self):
        self.train()
        self.log.return_value = ['Epoch 7', 'train_loss -0.7644', 'val_loss -0.7228',
            'Pseudo dice [np.float32(0.9603), np.float32(0.7794)]', 'Epoch time: 79.97 s']
        native = self.snapshot()['native']
        self.assertEqual((native['epoch'], native['completed_epochs']), (8, 8))
        self.assertEqual(native['native_metrics']['pseudo_dice'], [.9603, .7794])
        self.assertFalse(native['full_native_evaluation'])

    def test_adopted_score_PID_birth_and_argv_must_equal_sealed_actor(self):
        for key, value in [('child_pid', 124), ('child_create_time', 9.), ('command', ['DEBUG foreign worker'])]:
            with self.subTest(key=key):
                original = self.state[key]
                self.state[key] = value
                self.publish_all()
                try:
                    with self.assertRaisesRegex(ValueError, 'adopted|child command'):
                        self.snapshot()
                finally:
                    self.state[key] = original

    def test_real_process_snapshot_rejects_reused_birth_foreign_argv_and_UID(self):
        for pid in (999, 123):
            for key, value in [('birth', 9.), ('command', ['DEBUG reused PID']), ('UID', 0)]:
                with self.subTest(pid=pid, key=key):
                    original = self.os_processes[pid][key]
                    self.os_processes[pid][key] = value
                    try:
                        with self.assertRaisesRegex(ValueError, 'PID/birth/full argv/UID'):
                            self.snapshot()
                    finally:
                        self.os_processes[pid][key] = original

    def test_dead_controller_child_and_zombie_cannot_claim_native_training(self):
        self.train()
        actor = self.os_processes.pop(999)
        result = self.snapshot()
        self.assertEqual(result['status'], 'CONTROLLER_EXITED_CHILD_ALIVE')
        self.assertFalse(result['native']['native_training_started'])
        self.os_processes[999] = actor
        self.os_processes[123]['status'] = viewer.scoring.psutil.STATUS_ZOMBIE
        result = self.snapshot()
        self.assertFalse(result['native']['child_alive'])
        self.assertFalse(result['native']['native_training_started'])
        self.os_processes.pop(999)
        self.assertEqual(self.snapshot()['status'], 'EXITED_WITHOUT_COMPLETION')
        self.log.assert_not_called()

    def test_recorded_waiting_reason_and_failure_are_actual_controller_state(self):
        self.train()
        self.state.update(status='WAITING_FOR_ASSIGNED_GPU', stage=None, child_pid=None,
            child_create_time=None, command=None, waiting_reason='DEBUG assigned GPU6 CP is live')
        self.publish_all()
        result = self.snapshot()
        self.assertEqual(result['status'], 'WAITING_FOR_ASSIGNED_GPU')
        self.assertEqual(result['native_pending_reason'], 'DEBUG assigned GPU6 CP is live')
        self.assertFalse(result['native']['native_training_started'])
        self.state.update(status='FAILED', error='DEBUG actual admission SHA differs')
        self.publish_all()
        result = self.snapshot()
        self.assertEqual(result['status'], 'FAILED')
        self.assertIn('admission SHA', result['read_problem']['detail'])
        self.assertNotIn('GPU6 CP is live', result['native_pending_reason'])

    def test_full250_BEST26_completion_uses_real_read_only_metadata_verifier(self):
        for gpu in (4, 5):
            with self.subTest(gpu=gpu):
                if gpu == 5:
                    self.gpu5_early()
                self.completion()
                result = self.snapshot()
                self.assertEqual(result['status'], 'COMPLETE')
                self.assertEqual(result['active_stage'], 'native')
                self.assertTrue(result['full_native_training'])
                self.assertTrue(result['full_native_evaluation'])
                self.assertTrue(result['native']['completion_metadata_verified'])

    def test_missing_or_foreign_BEST_evaluation_cannot_be_completed_from_flags(self):
        self.completion()
        receipt = viewer.pipeline.read(self.best_path)
        for key, value in [('checkpoint_name', 'checkpoint_final.pth'), ('validation_cases', ['DEBUG_val_000']),
                ('original_checkpoint_load_called', False), ('final_checkpoint_fallback', True),
                ('CP_at_inference', True)]:
            with self.subTest(key=key):
                changed = dict(receipt, **{key:value})
                self.write(self.best_path, changed)
                self.complete_record['best_validation_receipt_sha256'] = viewer.direct.sha(self.best_path)
                self.write(self.complete_path, self.complete_record)
                self.state['completion_proof']['sha256'] = viewer.direct.sha(self.complete_path)
                self.publish_all()
                with self.assertRaisesRegex(ValueError, 'BEST/full26'):
                    self.snapshot()

    def test_missing_full250_or_full26_flags_rejected_before_metadata_completion(self):
        self.completion()
        self.state['full_native_evaluation_completed'] = False
        self.publish_all()
        with self.assertRaisesRegex(ValueError, 'complete250/BEST26'):
            self.snapshot()
        self.state['full_native_evaluation_completed'] = True
        self.complete_record['epochs'] = 249
        self.write(self.complete_path, self.complete_record)
        self.state['completion_proof']['sha256'] = viewer.direct.sha(self.complete_path)
        self.publish_all()
        with self.assertRaisesRegex(ValueError, 'full250'):
            self.snapshot()

    def test_stale_source_not_adopted_and_read_error_preserves_last_confirmed_snapshot(self):
        stale = dict(self.base, snapshot_fresh=False)
        self.assertIs(viewer.guard_snapshot(stale, self.renderer, self.path), stale)
        instance = viewer.Viewer.__new__(viewer.Viewer)
        instance.renderer, instance.path, instance.last_valid = self.renderer, self.path, None
        instance.base = SimpleNamespace(snapshot=lambda:copy.deepcopy(self.base))
        first = instance.snapshot()
        self.assertTrue(first['snapshot_fresh'])
        self.lease['request_sha256'] = 'c' * 64
        self.write(self.path.parent / 'lease.json', self.lease)
        result = instance.snapshot()
        self.assertEqual(result['status'], 'READ_ERROR')
        self.assertFalse(result['snapshot_fresh'])
        self.assertTrue(result['last_valid_snapshot_preserved'])
        self.assertEqual(result['last_confirmed_status'], first['status'])
        self.assertIn('controller/lease', result['read_problem']['detail'])
        self.assertEqual(result['gnn'], first['gnn'])

    def test_unknown_original_declaration_or_unregistered_stage_rejected(self):
        self.original['format'] = 'DEBUG_unknown_declaration'
        self.publish_all()
        with self.assertRaisesRegex(ValueError, 'declaration|binding'):
            self.snapshot()
        self.original['format'] = viewer.pipeline.NATIVE_REQUEST
        self.state.update(stage='DEBUG_unknown_stage', child_pid=None)
        self.publish_all()
        with self.assertRaisesRegex(ValueError, 'stage'):
            self.snapshot()

    def test_viewer_has_no_model_imports_mutation_or_process_signal_calls(self):
        tree = ast.parse(Path(viewer.__file__).read_text(encoding='utf8'))
        modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        self.assertFalse(any(name and name.startswith(('torch', 'hiercp_v1x', 'nnunet')) for name in modules))
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(name.endswith(('.write_text', '.write_bytes', '.unlink', '.rename', '.replace',
            '.kill', '.terminate', '.send_signal', '.load_state_dict')) for name in calls))


if __name__ == '__main__':
    unittest.main()
