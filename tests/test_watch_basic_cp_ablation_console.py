"""Read-only viewer fixtures; no model, GPU, training or server execution."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import training_monitor_ui as ui
from tools import watch_basic_cp_ablation as watch


class TTY(io.StringIO):
    def isatty(self): return True


class BasicNativeConsoleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.log = self.root/'train.log'
        self.previous = (
            '2026-10-13 15:00:00.000001: \n'
            '2026-10-13 15:00:00.000002: Epoch 5\n'
            '2026-10-13 15:00:00.000003: Current learning rate: 0.00982\n'
            '2026-10-13 15:01:30.000001: [OnlineCP] epoch=5 applied=193/500 rate=0.3860 schedule=abc\n'
            '2026-10-13 15:01:30.000002: [BasicCPAblation] epoch=5 arm=multi3 pasted_lesions=579 events=193\n'
            '2026-10-13 15:01:35.000001: train_loss -0.2\n'
            '2026-10-13 15:01:35.000002: val_loss -0.1\n'
            '2026-10-13 15:01:35.000003: Pseudo dice [np.float32(0.9), np.float32(0.4)]\n'
            '2026-10-13 15:01:35.000004: Epoch time: 95.0 s\n')
        self.current = (
            '2026-10-13 15:01:37.000001: \n'
            '2026-10-13 15:01:37.000002: Epoch 6\n'
            '2026-10-13 15:01:37.000003: Current learning rate: 0.00978\n')
        self.log.write_bytes(('older startup metadata\n'+self.previous+self.current).encode())
        self.value = dict(stage='nnunet_training', stage_log=str(self.log),
            GPU='1 / Basic CP multi3', physical_GPU=1, arm='multi3', status='RUNNING',
            epoch=7, completed_epochs=6, epochs=250, native_metrics={}, snapshot_fresh=True)

    def main(self, *args, stream=None):
        stream = stream or io.StringIO()
        with patch.object(watch, 'snapshot', return_value=self.value), patch.object(watch.sys, 'stdout', stream):
            watch.main(['--root', str(self.root), *args])
        return stream.getvalue()

    def test_once_preserves_last_completed_and_current_epoch_console_exactly(self):
        before = self.log.read_bytes()
        shown = self.main('--once')
        self.assertEqual(shown, self.previous+self.current)
        self.assertEqual(self.log.read_bytes(), before)
        self.assertNotIn('GPU1 / Basic CP', shown)
        self.assertNotIn('RAM', shown)
        self.assertNotIn('UI 2', shown)

    def test_native_tty_uses_original_console_not_legacy_tqdm_dashboard(self):
        shown = self.main('--once', stream=TTY())
        self.assertEqual(shown, self.previous+self.current)
        self.assertNotIn('\x1b[', shown)

    def test_following_appends_new_bytes_once_without_repeating_epoch_blocks(self):
        stream = io.StringIO(); console = watch.NativeConsole(stream)
        console.render(self.log); console.render(self.log)
        appended = '2026-10-13 15:03:11.000001: train_loss -0.35\n'
        with self.log.open('ab') as output: output.write(appended.encode())
        console.render(self.log); console.render(self.log)
        self.assertEqual(stream.getvalue(), self.previous+self.current+appended)

    def test_binary_output_preserves_actual_console_bytes(self):
        output = io.BytesIO()
        class BinaryStream:
            buffer = output
            def flush(self): pass
        watch.NativeConsole(BinaryStream()).render(self.log)
        self.assertEqual(output.getvalue(), (self.previous+self.current).encode())

    def test_partial_utf8_is_completed_on_next_poll_without_replacement(self):
        self.log.write_bytes(b'Epoch 0\n'+ '종'.encode()[:2])
        stream = io.StringIO(); console = watch.NativeConsole(stream)
        console.render(self.log)
        self.assertEqual(stream.getvalue(), 'Epoch 0\n')
        with self.log.open('ab') as output: output.write('종'.encode()[2:]+b'\n')
        console.render(self.log)
        self.assertEqual(stream.getvalue(), 'Epoch 0\n종\n')

    def test_missing_console_waits_without_fabricating_dashboard_or_progress(self):
        self.log.unlink()
        self.assertEqual(self.main('--once'), '')

    def test_changed_log_identity_starts_from_actual_new_console(self):
        stream = io.StringIO(); console = watch.NativeConsole(stream)
        console.render(self.log)
        another = self.root/'other_train.log'; another.write_bytes(b'Epoch 200\nCurrent learning rate: 0.002\n')
        console.render(another)
        self.assertTrue(stream.getvalue().endswith('Epoch 200\nCurrent learning rate: 0.002\n'))

    def test_failed_native_console_keeps_actual_trace_and_completed_epoch(self):
        error = 'Traceback (most recent call last):\nRuntimeError: exact native failure\n'
        with self.log.open('ab') as output: output.write(error.encode())
        self.value['status'] = 'FAILED'
        shown = self.main('--once')
        self.assertEqual(shown, self.previous+self.current+error)

    def test_json_metadata_remains_the_existing_snapshot_schema(self):
        shown = self.main('--json')
        self.assertEqual(json.loads(shown), self.value)

    def test_preparation_keeps_original_cp_eight_row_ui2(self):
        self.value.update(stage='scoring', stage_label='CP 위치 후보 256개 준비',
            sources_done=211, candidates_done=211*256, candidates_total=642*256,
            cp_input='Basic uniform / 자기 환자 병변 / GNN 미사용', RSS_GiB=18,
            RSS_source='현재 프로세스 트리', process_state='sleeping',
            controller_status='RUNNING', controller_stage='candidate_bank', error=None)
        shown = self.main('--once')
        self.assertEqual(tuple(line.partition(': ')[0] for line in shown.splitlines()), ui.CP_ROW_LABELS)
        self.assertIn('UI 2', shown)
        self.assertIn('211/642', shown)
        self.assertNotIn('Epoch 6', shown)

    def test_running_native_tty_appends_after_initial_block_and_closes_only_viewer(self):
        stream = TTY()
        appended = '2026-10-13 15:03:11.000001: train_loss -0.35\n'
        def append_after_first_poll(*args):
            if not getattr(append_after_first_poll, 'done', False):
                with self.log.open('ab') as output: output.write(appended.encode())
                append_after_first_poll.done = True
        with patch.object(watch, 'snapshot', side_effect=[self.value, self.value, KeyboardInterrupt()]), \
                patch.object(watch.sys, 'stdout', stream), patch.object(watch.time, 'sleep', side_effect=append_after_first_poll):
            watch.main(['--root', str(self.root)])
        self.assertTrue(stream.getvalue().startswith(self.previous+self.current+appended))
        self.assertEqual(stream.getvalue().count('Epoch 5'), 1)
        self.assertIn('진행 뷰어만 종료했습니다.', stream.getvalue())


if __name__ == '__main__':
    unittest.main()
