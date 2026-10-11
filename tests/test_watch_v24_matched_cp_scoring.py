"""DEBUG CPU metadata fixtures only; no model, CUDA or clinical score claim."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import psutil

from tools import watch_v24_matched_cp_scoring as viewer


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


def scoring_fixture(root, gpu=6, source_output=None, source_code='UNIT_original_code'):
    root.mkdir()
    request = dict(format='v24_full642_depth1_current_BEST_scoring_request_v1',
        gpu=gpu, source_entries=642, CP_positions=82176, debug=False, full642_scoring=True,
        source_output=str(source_output) if source_output else 'UNIT_source', source_code=source_code,
        pin_path=str(root / 'own_best_pin.json'), output=str(root / 'bank'),
        native_pending_reason='UNIT: original whole-native storage budget unavailable')
    write_json(root / 'request.json', request)
    registration = dict(format='v24_full642_depth1_current_BEST_scoring_registration_v1',
        sources=642, CP_positions=82176, root=str(root), pid=70002, create_time=200.,
        request_sha256=hashlib.sha256((root / 'request.json').read_bytes()).hexdigest(),
        command=['UNIT_python', '-B', '-u', str(root / 'worker.py'), '--request', str(root / 'request.json')])
    write_json(root / 'registration.json', registration)
    write_json(root / 'started.json', dict(status='FULL642_SCORING_RUNNING', debug=False,
        full_native_training=False, full_native_evaluation=False))
    return request, registration


def event(gpu=6, done=4):
    return dict(phase=viewer.PHASE, GPU=gpu, total_sources=642, completed_sources=done,
        complete_CP_positions=done * 128, debug=False, production_optimizer_updates=0, seconds=12.5)


def process_fixture(registration, *, birth=None, command=None, state='running'):
    return SimpleNamespace(create_time=lambda: registration['create_time'] if birth is None else birth,
        cmdline=lambda: registration['command'] if command is None else command,
        uids=lambda: SimpleNamespace(real=os.getuid() if hasattr(os, 'getuid') else 0),
        status=lambda: state, memory_info=lambda: SimpleNamespace(rss=2**30), is_running=lambda: True,
        children=lambda recursive=True: [])


class Full642ScoringViewerCPU_DEBUG(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='scoring_viewer_DEBUG_',
            dir=Path(__file__).resolve().parents[1] / 'outputs')
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name) / 'score'

    def test_each_sealed_arm_displays_its_actual_GPU_and_all_candidates(self):
        for gpu in (4, 5, 6):
            with self.subTest(gpu=gpu):
                root = self.root.parent / str(gpu)
                _, registration = scoring_fixture(root, gpu)
                (root / 'console.log').write_text(json.dumps(event(gpu)) + '\n', encoding='utf8')
                with patch.object(viewer.psutil, 'Process', return_value=process_fixture(registration)):
                    observed = viewer.snapshot(root)
                self.assertEqual(observed['GPU'], gpu)
                self.assertEqual((observed['source_done'], observed['CP_done']), (4, 512))
                self.assertEqual(observed['status'], 'SCORING_PROCESS_ALIVE')
                self.assertIn('GPU' + str(gpu) + ' GNN BEST', viewer.display(observed))
                self.assertFalse(observed['full_native_training'])

    def test_cross_GPU_console_event_is_rejected(self):
        _, registration = scoring_fixture(self.root, 4)
        (self.root / 'console.log').write_text(json.dumps(event(6)) + '\n', encoding='utf8')
        with patch.object(viewer.psutil, 'Process', return_value=process_fixture(registration)):
            with self.assertRaisesRegex(ValueError, 'GPU event differs'):
                viewer.snapshot(self.root)

    def test_invalid_or_boolean_GPU_rejected_before_process_inspection(self):
        for gpu in (1, True, '4'):
            root = self.root.parent / ('invalid_' + str(gpu))
            scoring_fixture(root, gpu)
            with self.subTest(gpu=gpu), patch.object(viewer.psutil, 'Process') as process:
                with self.assertRaisesRegex(ValueError, 'sealed'):
                    viewer.snapshot(root)
                process.assert_not_called()

    def test_dead_or_zombie_PID_cannot_be_running(self):
        _, registration = scoring_fixture(self.root)
        with patch.object(viewer.psutil, 'Process', side_effect=psutil.NoSuchProcess(registration['pid'])):
            self.assertEqual(viewer.snapshot(self.root)['status'], 'EXITED_WITHOUT_COMPLETION')
        with patch.object(viewer.psutil, 'Process', return_value=process_fixture(registration, state=psutil.STATUS_ZOMBIE)):
            self.assertEqual(viewer.snapshot(self.root)['status'], 'EXITED_WITHOUT_COMPLETION')

    def test_PID_reuse_or_foreign_command_is_visible(self):
        _, registration = scoring_fixture(self.root)
        for altered in (dict(birth=201.), dict(command=['foreign'])):
            with self.subTest(altered=altered), patch.object(viewer.psutil, 'Process',
                    return_value=process_fixture(registration, **altered)):
                self.assertEqual(viewer.ScoringViewer().snapshot(self.root)['status'], 'READ_ERROR')

    def test_complete_receipt_still_reports_native_storage_pending(self):
        _, registration = scoring_fixture(self.root, 4)
        write_json(self.root / 'complete.json', dict(status='FULL642_SCORING_COMPLETE_NATIVE_STORAGE_PENDING',
            completed_sources=642, completed_CP_positions=82176, full_native_training=False,
            full_native_evaluation=False, native_pending_reason='UNIT no native capacity'))
        with patch.object(viewer.psutil, 'Process', side_effect=psutil.NoSuchProcess(registration['pid'])):
            observed = viewer.snapshot(self.root)
        self.assertEqual((observed['source_done'], observed['CP_done']), (642, 82176))
        self.assertIn('UNIT no native capacity', viewer.display(observed))
        self.assertFalse(observed['full_native_training'])

    def test_partial_final_console_line_retains_only_completed_events(self):
        self.root.mkdir()
        path = self.root / 'console.log'
        path.write_bytes((json.dumps(event(4)) + '\n{"phase":').encode())
        rows, pending = viewer.complete_events(path, 4)
        self.assertEqual(len(rows), 1)
        self.assertTrue(pending)
        path.write_bytes(path.read_bytes() + b'bad}\n')
        with self.assertRaisesRegex(ValueError, 'completed console JSON'):
            viewer.complete_events(path, 4)

    def test_partial_metadata_preserves_stale_then_persistent_error_and_recovers(self):
        _, registration = scoring_fixture(self.root, 4)
        (self.root / 'console.log').write_text(json.dumps(event(4)) + '\n', encoding='utf8')
        monitor = viewer.ScoringViewer()
        with patch.object(viewer.psutil, 'Process', return_value=process_fixture(registration)):
            self.assertTrue(monitor.snapshot(self.root)['snapshot_fresh'])
            path = self.root / 'complete.json'
            path.write_text('{', encoding='utf8')
            with patch.object(viewer.time, 'monotonic', return_value=100.):
                pending = monitor.snapshot(self.root)
            self.assertEqual(pending['status'], 'READ_PENDING')
            self.assertEqual(pending['source_done'], 4)
            self.assertFalse(pending['snapshot_fresh'])
            with patch.object(viewer.time, 'monotonic', return_value=111.):
                self.assertEqual(monitor.snapshot(self.root)['status'], 'READ_ERROR')
            write_json(path, dict(status='FULL642_SCORING_COMPLETE_NATIVE_STORAGE_PENDING', completed_sources=642,
                completed_CP_positions=82176, full_native_training=False, full_native_evaluation=False))
            self.assertEqual(monitor.snapshot(self.root)['source_done'], 642)

    def test_malformed_shape_is_immediate_error_with_exact_file(self):
        scoring_fixture(self.root)
        write_json(self.root / 'registration.json', [])
        observed = viewer.ScoringViewer().snapshot(self.root)
        self.assertEqual(observed['status'], 'READ_ERROR')
        self.assertIn(str(self.root / 'registration.json'), observed['detail'])

    def test_CtrlC_only_returns_from_viewer_no_signal(self):
        scoring_fixture(self.root)
        with patch.object(viewer.ScoringViewer, 'snapshot', side_effect=KeyboardInterrupt), patch('sys.stdout', new=io.StringIO()) as out:
            viewer.main(['--root', str(self.root), '--no-clear'])
        self.assertIn('뷰어만 종료', out.getvalue())


if __name__ == '__main__':
    unittest.main()
