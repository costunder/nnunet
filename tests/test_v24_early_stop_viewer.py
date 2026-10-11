"""DEBUG CPU metadata tests for the additive early-stop display adapter."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import watch_v24_early_stop_pipeline as viewer


class EarlyViewerDEBUG(unittest.TestCase):
    def setUp(self):
        self.real_read = viewer.pipeline.read
        self.path = Path('/DEBUG/early/controller/request.json')
        self.source = Path('/DEBUG/original/gpu5/gnn')
        self.guardian = Path('/DEBUG/early/pause_guardian')
        self.code = '/DEBUG/code'
        self.digest = 'a' * 64
        self.auth = dict(path='/DEBUG/early/authorization.json', sha256=self.digest, stat=[1, 2, 3, 4, 5])
        command = ['/DEBUG/nnunet/bin/python', '-B', '-u', str(Path(self.code) / 'tools/run_v24_early_stop_downstream.py'),
            'watch', '--request', str(self.path)]
        self.request = dict(format=viewer.REQUEST, GPU_arm=5, full_GNN_training=False, debug=False, target_epoch=27,
            original_planned_epochs=40, source_output=str(self.source), source_code='/DEBUG/scientific',
            source_entries=642, CP_positions=82176, native_epochs=250, native_physical_batch=2,
            full_native_evaluation_cases=26, controller_root=str(self.path.parent), code=self.code,
            resource_contract=dict(GPU_arm=5, source_output=str(self.source), python=command[0], chain_root='/DEBUG/native'),
            authorization=self.auth, pin_path=str(self.path.parent/'own_early_stop_best_pin.json'),
            stages=[dict(name=name, command=['DEBUG-child', name]) for name in ('pin_early_stopped', 'score', 'train')])
        self.state = dict(format=viewer.STATUS, GPU_arm=5, request_sha256=self.digest, full_GNN_training=False,
            authorized_stop_epoch=27, source_output=str(self.source), supervisor_pid=999, supervisor_create_time=1.,
            status='WAITING_FOR_EARLY_STOP_RELEASE', stage=None, child_pid=None, stages_completed=[],
            waiting_reason='Actual epoch27 full128 validation pending')
        self.gnn = dict(GPU=5, status='RUNNING', epoch=27, epochs=40, completed_epochs=26, active_U=128,
            stage='full_validation', full_training_completed=False, snapshot_fresh=True)
        self.records = {str(self.path):self.request, str(Path(self.auth['path'])):dict(user_authorized=True, target_epoch=27,
            source_output=str(self.source)), str(self.path.parent/'status.json'):self.state,
            str(self.path.parent/'lease.json'):dict(pid=999, create_time=1., command=command, UID=41833, request_sha256=self.digest)}
        self.instance = viewer.Viewer.__new__(viewer.Viewer)
        self.instance.path = self.path; self.instance.guardian = self.guardian
        self.instance.source = SimpleNamespace(output=self.source)
        self.instance.gnn = SimpleNamespace(snapshot=lambda _:copy.deepcopy(self.gnn))
        self.instance.renderer = SimpleNamespace(parse_nnunet_lines=Mock(return_value=None), log_tail=Mock(return_value=''))
        self.instance.last_valid = None
        self.stack = []
        for item in (patch.object(viewer.pipeline, 'read', side_effect=lambda p, **_:self.records.get(str(p))),
                patch.object(viewer, 'proof', side_effect=lambda p:Path(p['path'])),
                patch.object(viewer.direct, 'sha', return_value=self.digest),
                patch.object(viewer, 'process', side_effect=lambda pid, *_:dict(alive=pid is not None, process_status='sleeping'))):
            item.start(); self.addCleanup(item.stop)

    def stop_receipt(self):
        self.gnn['status'] = 'PAUSED'
        self.records[str(self.guardian/'receipt.json')] = dict(status='VERIFIED_PAUSED_AT_EPOCH_BOUNDARY', verified=True,
            completed_epochs=27, optimizer_updates=459, source_output=str(self.source), authorization=self.auth)

    def score_controller(self):
        """A fresh DEBUG console path; all scientific metadata remains mocked."""
        test_root = Path(__file__).resolve().parent
        temporary = tempfile.TemporaryDirectory(prefix='DEBUG_early_stop_viewer_', dir=test_root)
        self.assertTrue(Path(temporary.name).resolve().is_relative_to(test_root))
        self.addCleanup(temporary.cleanup)
        old = self.path
        lease = self.records.pop(str(old.parent / 'lease.json'))
        self.records.pop(str(old)); self.records.pop(str(old.parent / 'status.json'))
        self.path = Path(temporary.name).absolute() / 'request.json'
        self.instance.path = self.path
        self.request['controller_root'] = str(self.path.parent)
        self.request['pin_path'] = str(self.path.parent / 'own_early_stop_best_pin.json')
        lease['command'][-1] = str(self.path)
        self.records.update({str(self.path):self.request, str(self.path.parent/'status.json'):self.state,
            str(self.path.parent/'lease.json'):lease})
        self.stop_receipt()
        self.state.update(stage='score', status='RUNNING', child_pid=123, child_create_time=2.,
            command=['DEBUG-child', 'score'])
        pin = dict(format=viewer.PIN, physical_GPU=5, full_training=False, completed_epochs=27,
            optimizer_updates=459, source_output=str(self.source), source_code=self.request['source_code'])
        self.records[self.request['pin_path']] = pin
        self.records[str(self.path.parent/'stage_pin_early_stopped.json')] = dict(
            output=self.request['pin_path'], output_sha256=self.digest)
        return self.path.parent / 'score.log'

    @staticmethod
    def score_event(done):
        return dict(phase=viewer.scoring.PHASE, GPU=5, total_sources=642,
            completed_sources=done, complete_CP_positions=done*128, debug=False,
            production_optimizer_updates=0, seconds=3.5)

    def test_current_epoch_validation_wait_is_not_full_training(self):
        result = self.instance.snapshot()
        self.assertTrue(result['snapshot_fresh'])
        self.assertEqual(result['active_stage'], 'gnn')
        self.assertEqual(result['gnn']['completed_epochs'], 26)
        self.assertFalse(result['gnn']['full_training_completed'])
        self.assertFalse(result['native']['native_training_started'])

    def test_verified_pause_displays_actual27_of40_without_complete_claim(self):
        self.stop_receipt()
        result = self.instance.snapshot()
        self.assertEqual(result['status'], 'EARLY_STOPPED')
        self.assertEqual((result['gnn']['epoch'], result['gnn']['completed_epochs'], result['gnn']['epochs']), (27, 27, 40))
        self.assertFalse(result['gnn']['full_training_completed'])
        self.assertEqual(len(viewer.pipeline.display(result, experiment_version='v2.4').splitlines()), 9)

    def test_receipt_cannot_turn_live_training_into_early_stopped(self):
        self.stop_receipt(); self.gnn['status'] = 'RUNNING'
        result = self.instance.snapshot()
        self.assertEqual(result['status'], 'READ_ERROR')
        self.assertFalse(result['snapshot_fresh'])

    def test_wrong_full_gnn_flag_is_visible_and_retains_stale_snapshot(self):
        first = self.instance.snapshot()
        self.request['full_GNN_training'] = True
        result = self.instance.snapshot()
        self.assertTrue(result['last_valid_snapshot_preserved'])
        self.assertFalse(result['snapshot_fresh'])
        self.assertEqual(result['gnn'], first['gnn'])

    def test_native_tree_rss_does_not_conflict_with_child_rss(self):
        self.stop_receipt()
        self.state.update(stage='train', status='RUNNING', child_pid=123, child_create_time=2., command=['DEBUG-child', 'train'],
            stage_log=str(self.path.parent/'train.log'), child_process_tree_RSS_bytes=5 * 2**30)
        with patch.object(viewer, 'process', return_value=dict(alive=True, process_status='sleeping', RSS_GiB=2.)):
            result = self.instance.snapshot()
        self.assertEqual(result['active_stage'], 'native')
        self.assertEqual(result['native']['RSS_GiB'], 5.)
        self.assertTrue(result['native']['native_training_started'])
        self.assertFalse(result['native']['full_native_training'])

    def test_complete_without_actual_native_evaluation_flags_is_rejected(self):
        self.stop_receipt(); self.state['status'] = 'COMPLETE'
        with patch.object(viewer.pipeline, 'native_completion') as complete:
            result = self.instance.snapshot()
        self.assertEqual(result['status'], 'READ_ERROR')
        complete.assert_not_called()

    def test_cp_progress_uses_last_complete_event_and_verified_own_pin(self):
        console = self.score_controller()
        console.write_bytes(json.dumps(self.score_event(16)).encode() + b'\n'
            + json.dumps(self.score_event(24)).encode() + b'\n'
            + json.dumps(self.score_event(32)).encode())
        result = self.instance.snapshot()
        self.assertTrue(result['snapshot_fresh'])
        self.assertEqual(result['active_stage'], 'scoring')
        self.assertEqual(result['status'], 'SCORING_PROCESS_ALIVE')
        self.assertEqual((result['scoring']['source_done'], result['scoring']['CP_done']), (24, 3072))
        self.assertEqual((result['scoring']['source_total'], result['scoring']['CP_total']), (642, 82176))
        self.assertTrue(result['scoring']['own_BEST_pin_verified'])
        self.assertTrue(result['scoring']['partial_console_line_pending'])
        self.assertFalse(result['full_native_training'])
        self.assertFalse(result['full_native_evaluation'])
        self.assertEqual(len(viewer.pipeline.display(result, experiment_version='v2.4').splitlines()), 8)

    def test_cp_rejects_wrong_pin_sha_or_wrong_early_stop_update_count(self):
        console = self.score_controller()
        console.write_text(json.dumps(self.score_event(8)) + '\n', encoding='utf8')
        for changed in ('pin_completion_sha', 'pin_optimizer_updates'):
            with self.subTest(changed=changed):
                if changed == 'pin_completion_sha':
                    self.records[str(self.path.parent/'stage_pin_early_stopped.json')]['output_sha256'] = 'b'*64
                else:
                    self.records[str(self.path.parent/'stage_pin_early_stopped.json')]['output_sha256'] = self.digest
                    self.records[self.request['pin_path']]['optimizer_updates'] = 460
                result = self.instance.snapshot()
                self.assertEqual(result['status'], 'READ_ERROR')
                self.assertFalse(result['snapshot_fresh'])
                self.assertFalse(result.get('full_native_training'))
                self.assertFalse(result.get('full_native_evaluation'))
                self.assertIn('pin', result['read_problem']['detail'].lower())

    def test_failed_score_keeps_measured_cp_count_without_completion_flags(self):
        console = self.score_controller()
        console.write_text(json.dumps(self.score_event(8)) + '\n', encoding='utf8')
        self.state.update(status='FAILED', error='DEBUG actual scoring failure: finite score check',
            child_pid=None, child_create_time=None)
        result = self.instance.snapshot()
        self.assertTrue(result['snapshot_fresh'])
        self.assertEqual(result['status'], 'FAILED')
        self.assertEqual(result['scoring']['CP_done'], 1024)
        self.assertFalse(result['native']['native_training_started'])
        self.assertFalse(result['full_native_training'])
        self.assertFalse(result['full_native_evaluation'])
        self.assertIn('finite score check', result['read_problem']['detail'])

    def test_corrupt_controller_json_retains_cp_counts_as_explicitly_stale(self):
        console = self.score_controller()
        console.write_text(json.dumps(self.score_event(8)) + '\n', encoding='utf8')
        first = self.instance.snapshot()
        status = self.path.parent / 'status.json'
        status.write_text('{"status":', encoding='utf8')
        def read(path, **options):
            return self.real_read(path, **options) if Path(path) == status else self.records.get(str(path))
        with patch.object(viewer.pipeline, 'read', side_effect=read):
            result = self.instance.snapshot()
        self.assertEqual(result['status'], 'READ_PENDING')
        self.assertFalse(result['snapshot_fresh'])
        self.assertTrue(result['last_valid_snapshot_preserved'])
        self.assertEqual(result['scoring'], first['scoring'])
        self.assertEqual(result['last_confirmed_status'], 'SCORING_PROCESS_ALIVE')
        self.assertIn(str(status), result['read_problem']['detail'])
        self.assertIn('새 기록이 아님', viewer.pipeline.display(result, experiment_version='v2.4'))

    def test_missing_required_controller_status_is_pending_without_invented_cp_progress(self):
        self.score_controller()
        status = self.path.parent / 'status.json'
        def read(path, **options):
            return self.real_read(path, **options) if Path(path) == status else self.records.get(str(path))
        with patch.object(viewer.pipeline, 'read', side_effect=read):
            result = self.instance.snapshot()
        self.assertEqual(result['status'], 'READ_PENDING')
        self.assertFalse(result['snapshot_fresh'])
        self.assertFalse(result['last_valid_snapshot_preserved'])
        self.assertFalse(result['scoring'])
        self.assertIn(str(status), result['read_problem']['detail'])

    def test_viewer_cannot_load_or_rewrite_frozen_scientific_sources(self):
        import ast
        tree = ast.parse(Path(viewer.__file__).read_text(encoding='utf8'))
        modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        self.assertFalse(any(name and name.startswith(('torch', 'hiercp_v1x', 'nnunet')) for name in modules))
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(name.endswith(('.write_text', '.write_bytes', '.unlink', '.rename', '.replace',
            '.kill', '.terminate', '.send_signal', '.load_state_dict')) for name in calls))


if __name__ == '__main__':
    unittest.main()
