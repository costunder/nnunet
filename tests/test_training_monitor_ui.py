"""DEBUG CPU fixtures for the fixed monitor UI; no files, jobs or models read."""
from copy import deepcopy
import io
import math
import unittest
from unittest.mock import patch

from tools import training_monitor_ui as ui


def measured_model():
    return {
        'gpu': 5, 'experiment_version': 'v2.4',
        'state': {'status': 'RUNNING', 'stage': 'gnn', 'fresh': True},
        'gnn': {'epoch': 5, 'epochs': 40, 'completed_epochs': 4, 'active_u': 9,
            'optimizer_updates': 76, 'loss': .125, 'loss_update': 76,
            'full_completed': False, 'phase_label': 'train',
            'progress': {'done': 8, 'total': 21, 'unit': 'batch'},
            'stage_validation': {'epoch': 4, 'active_u': 7, 'mrr': .4, 'top1': .3},
            'full_validation': {'epoch': 4, 'active_u': 128, 'mrr': .2, 'top1': .1}},
        'cp': {'status': 'WAITING_FOR_FULL_GNN', 'source_done': None,
            'source_total': 642, 'positions_done': None, 'positions_total': 82176,
            'last_batch_seconds': None, 'log_age_seconds': 3.5,
            'input_basis': '입력 계획: GPU5 GNN BEST'},
        'native': {'status': 'WAITING_FOR_FULL642_SCORING', 'stage': None,
            'epoch': None, 'epochs': 250, 'completed_epochs': None, 'started': False,
            'full_training': False, 'full_evaluation': False,
            'train_loss': None, 'val_loss': None, 'pseudo_dice': None},
        'resources': {'rss_gib': 42.25, 'ram_limit_gib': 64,
            'source': 'current_process_tree', 'process_status': 'sleeping'},
        'waiting_reason': 'GPU5 GNN BEST 확정 대기', 'warning': None, 'error': None,
    }


class TTYBuffer(io.StringIO):
    def isatty(self):
        return True


class FixedMonitorUI_CPU_DEBUG(unittest.TestCase):
    def test_row_labels_and_order_are_identical_across_versions_and_stages(self):
        for version in ('v1 basic', 'v2.3', 'v2.4', 'future'):
            for stage in ('gnn', 'scoring', 'native', 'READ_ERROR'):
                with self.subTest(version=version, stage=stage):
                    model = measured_model()
                    model['experiment_version'] = version
                    model['state']['stage'] = stage
                    lines = ui.render_lines(model)
                    self.assertEqual(len(lines), 16)
                    self.assertEqual(tuple(line.split(':', 1)[0] for line in lines), ui.ROW_LABELS)
                    self.assertEqual(len(ui.ROW_KEYS), 16)
                    self.assertIn('GPU5 | 버전 ' + version + ' | UI 1', lines[0])
                    self.assertEqual(ui.render(model).splitlines(), lines)

    def test_measured_phase_counts_and_validation_context_are_preserved(self):
        lines = ui.render_lines(measured_model())
        self.assertIn('epoch 5/40 | 완료 4 | U 9 | update 76', lines[2])
        self.assertIn('train | 8/21 batch (38.1%)', lines[3])
        self.assertIn('0.125000 | 측정 update 76', lines[4])
        self.assertIn('최근 완료 e4 U7 MRR 0.4000 top1 0.3000', lines[5])
        self.assertIn('최근 완료 e4 U128 MRR 0.2000 top1 0.1000', lines[6])
        self.assertIn('console 경과 3.5 s', lines[13])

    def test_missing_measurements_are_not_fabricated_as_zero_or_complete(self):
        lines = ui.render_lines({'gpu': 4})
        self.assertEqual(len(lines), 16)
        self.assertIn('epoch -/- | 완료 - | U - | update -', lines[2])
        self.assertIn('-/-', lines[3])
        self.assertIn('MRR - top1 -', lines[5])
        self.assertIn('train - | val -', lines[10])
        self.assertIn('RSS - GiB / 한도 - GiB', lines[12])
        self.assertNotIn('0.0000', '\n'.join(lines))
        self.assertNotIn('100.0%', '\n'.join(lines))
        self.assertNotIn('전체 완료 확인', '\n'.join(lines))

    def test_zero_is_displayed_only_when_measured_with_a_known_total(self):
        model = measured_model()
        model['cp'].update(source_done=0, positions_done=0, last_batch_seconds=0)
        lines = ui.render_lines(model)
        self.assertIn('0/642 (0.0%)', lines[7])
        self.assertIn('0/82176 (0.0%)', lines[8])
        self.assertIn('최근 batch 0.00 s', lines[13])

    def test_corrupt_or_unknown_progress_never_gets_a_percentage(self):
        cases = ((None, 21), (8, None), (0, 0), (True, 21), (.5, 21),
            (8, 21.5), (-1, 21), (8, -1), (math.nan, 21), (8, math.inf), ('8', 21))
        for done, total in cases:
            with self.subTest(done=done, total=total):
                model = measured_model()
                model['gnn']['progress'].update(done=done, total=total)
                self.assertNotIn('%', ui.render_lines(model)[3])
        model['gnn']['progress'].update(done=22, total=21)
        line = ui.render_lines(model)[3]
        self.assertIn('계수 확인 필요', line)
        self.assertNotIn('%', line)

    def test_nonfinite_metrics_are_visible_missing_values_not_fake_zero(self):
        model = measured_model()
        model['gnn']['loss'] = math.nan
        model['gnn']['stage_validation'].update(mrr=math.inf, top1=-math.inf)
        model['native']['pseudo_dice'] = [.7, math.nan]
        lines = ui.render_lines(model)
        self.assertIn('미기록(비유한 값)', lines[4])
        self.assertIn('MRR 미기록(비유한 값) top1 미기록(비유한 값)', lines[5])
        self.assertIn('[0.7000, 미기록(비유한 값)]', lines[11])

    def test_stale_snapshot_is_explicit_and_keeps_last_confirmed_metrics(self):
        model = measured_model()
        model['state'].update(status='READ_PENDING', fresh=False, last_confirmed_status='RUNNING')
        model['warning'] = 'report 기록 중'
        model['resources']['source'] = 'last_confirmed_history'
        lines = ui.render_lines(model)
        self.assertIn('새 기록이 아님 | READ_PENDING', lines[1])
        self.assertIn('마지막 확인 RUNNING', lines[1])
        self.assertIn('0.125000', lines[4])
        self.assertIn('출처 저장된 학습 이력', lines[12])
        self.assertIn('새 기록이 아님 | report 기록 중', lines[15])
        fresh = ui.render_lines(measured_model())[1]
        self.assertIn('읽기 성공', fresh)
        self.assertNotIn('최신', fresh)

    def test_native_epoch_and_status_do_not_prove_full_training_or_evaluation(self):
        model = measured_model()
        model['native'].update(status='COMPLETE', stage='train', epoch=250,
            completed_epochs=250, started=True, train_loss=-.3, val_loss=-.2, pseudo_dice=[.8, .7])
        lines = ui.render_lines(model)
        self.assertIn('학습/BEST 평가 작업 실행 중', lines[9])
        self.assertIn('평가 전체 완료 미확인', lines[9])
        self.assertNotIn('전체 완료 확인', lines[9])
        self.assertIn('train -0.300000 | val -0.200000 | 관측 epoch 250', lines[10])
        self.assertIn('[0.8000, 0.7000] | 관측 epoch 250', lines[11])
        model['native'].update(started=False, full_training=True, full_evaluation=True)
        complete = ui.render_lines(model)[9]
        self.assertIn('전체 완료 확인 | 평가 전체 완료 확인', complete)

    def test_inactive_native_after_failure_is_not_claimed_to_have_never_started(self):
        for status in ('FAILED', 'EXITED_WITHOUT_COMPLETION', 'WAITING_FOR_ASSIGNED_GPU'):
            model = measured_model()
            model['native'].update(status=status, started=False, epoch=17)
            line = ui.render_lines(model)[9]
            self.assertIn('현재 실행 미확인', line)
            self.assertNotIn('시작 전', line)

    def test_controller_status_and_stage_remain_visible_without_waiting_reason(self):
        model = measured_model()
        model['waiting_reason'] = None
        model['native'].update(status='RUNNING', stage='calibrate_native')
        line = ui.render_lines(model)[14]
        self.assertIn('상태 RUNNING | 단계 calibrate_native | 사유 -', line)
        self.assertIn('입력 기준 입력 계획: GPU5 GNN BEST', ui.render_lines(model)[8])

    def test_plain_errors_are_complete_single_lines_and_terminal_commands_removed(self):
        model = measured_model()
        path = '/real/source/training/' + 'long_' * 40 + 'full_validation_epoch_004.json'
        cause = 'JSONDecodeError: Expecting value line 1 column 1'
        model['error'] = '\x1b[31m' + path + '\x1b[0m\n' + cause + '\r\x00'
        model['warning'] = '\x1b]0;rename-terminal\x07warn\tmessage'
        lines = ui.render_lines(model)
        self.assertEqual(len(ui.render(model).splitlines()), 16)
        self.assertIn(path + ' ' + cause, lines[15])
        self.assertIn('warn message', lines[15])
        self.assertNotIn('\x1b', '\n'.join(lines))
        self.assertNotIn('\x00', '\n'.join(lines))
        terminal_lines = ui.render_lines(model, width=80)
        self.assertTrue(terminal_lines[15].endswith('…'))
        self.assertTrue(all(ui.display_width(line) <= 80 for line in terminal_lines))
        self.assertIn(cause, lines[15])

    def test_unicode_truncation_preserves_whole_codepoints_and_cell_budget(self):
        text = '한글 e\u0301界😀 오류'
        self.assertEqual(ui.display_width('한글'), 4)
        self.assertEqual(ui.display_width('e\u0301'), 1)
        self.assertEqual(ui.display_width('😀'), 2)
        for width in range(1, 20):
            truncated = ui.truncate(text, width)
            self.assertLessEqual(ui.display_width(truncated), width)
            self.assertEqual(truncated.encode('utf-8').decode('utf-8'), truncated)
        self.assertEqual(ui.truncate('e\u0301界X', 2), 'e\u0301…')

    def test_render_is_pure_and_rejects_invalid_normalized_shapes_clearly(self):
        model = measured_model()
        before = deepcopy(model)
        ui.render(model)
        self.assertEqual(model, before)
        for malformed in ([], {'gnn': []}, {'gnn': {'progress': []}},
                {'gnn': {'stage_validation': []}}):
            with self.subTest(malformed=malformed), self.assertRaisesRegex(ValueError, 'must be an object'):
                ui.render_lines(malformed)
        for width in (0, -1, True, 2.5):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, 'positive integer'):
                ui.render_lines(model, width=width)

    def test_plain_output_schema_is_independent_of_tqdm_environment(self):
        for toggle in ('0', '1'):
            with patch.dict('os.environ', {'TQDM_DISABLE': toggle}):
                stream = io.StringIO()
                view = ui.TerminalView(stream)
                lines = view.render(measured_model())
                view.close()
                self.assertEqual(stream.getvalue().splitlines(), lines)
                self.assertEqual(len(lines), 16)
                self.assertNotIn('\x1b', stream.getvalue())
                self.assertFalse(stream.closed)

    def test_tty_redraw_updates_only_owned_rows_and_restores_cursor_once(self):
        stream = TTYBuffer()
        view = ui.TerminalView(stream, size_provider=lambda: (100, 24))
        first = view.render(measured_model())
        marker = len(stream.getvalue())
        model = measured_model()
        model['state']['status'] = 'READ_PENDING'
        second = view.render(model)
        redraw = stream.getvalue()[marker:]
        self.assertEqual(len(first), 16)
        self.assertEqual(len(second), 16)
        self.assertTrue(redraw.startswith('\x1b[16A\r'))
        self.assertEqual(redraw.count('\x1b[2K'), 16)
        self.assertNotIn('\x1b[2J', stream.getvalue())
        self.assertNotIn('\x1b[H', stream.getvalue())
        self.assertTrue(all(ui.display_width(line) <= 99 for line in second))
        view.close()
        view.close()
        self.assertEqual(stream.getvalue().count('\x1b[?25l'), 1)
        self.assertEqual(stream.getvalue().count('\x1b[?25h'), 1)
        self.assertFalse(stream.closed)
        with self.assertRaisesRegex(RuntimeError, 'already closed'):
            view.render(model)

    def test_small_terminal_keeps_full_state_without_cursor_animation(self):
        for dimensions in ((60, 24), (120, 16), None, (True, 24)):
            with self.subTest(dimensions=dimensions):
                stream = TTYBuffer()
                view = ui.TerminalView(stream, size_provider=lambda: dimensions)
                model = measured_model()
                model['error'] = '/real/very_long/report.json: JSONDecodeError actual cause'
                view.render(model)
                view.close()
                output = stream.getvalue()
                self.assertEqual(len(output.splitlines()), 16)
                self.assertNotIn('\x1b', output)
                self.assertIn('JSONDecodeError actual cause', output)
                self.assertIn('WAITING_FOR_FULL642_SCORING', output)

    def test_resize_starts_a_new_region_instead_of_moving_over_reflowed_history(self):
        dimensions = [(100, 24)]
        stream = TTYBuffer()
        view = ui.TerminalView(stream, size_provider=lambda: dimensions[0])
        view.render(measured_model())
        marker = len(stream.getvalue())
        dimensions[0] = (120, 24)
        view.render(measured_model())
        self.assertNotIn('\x1b[16A', stream.getvalue()[marker:])
        marker = len(stream.getvalue())
        view.render(measured_model())
        self.assertTrue(stream.getvalue()[marker:].startswith('\x1b[16A'))
        marker = len(stream.getvalue())
        dimensions[0] = (40, 10)
        view.render(measured_model())
        tail = stream.getvalue()[marker:]
        self.assertTrue(tail.startswith('\x1b[?25h'))
        self.assertNotIn('\x1b[16A', tail)
        self.assertIn('전체 완료 미확인', tail)
        view.close()

    def test_ctrl_c_context_restores_only_viewer_cursor_and_propagates(self):
        stream = TTYBuffer()
        with self.assertRaises(KeyboardInterrupt):
            with ui.TerminalView(stream, size_provider=lambda: (100, 24)) as view:
                view.render(measured_model())
                raise KeyboardInterrupt()
        self.assertTrue(stream.getvalue().endswith('\x1b[?25h'))
        self.assertFalse(stream.closed)


if __name__ == '__main__':
    unittest.main()
