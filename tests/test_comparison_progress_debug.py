"""CPU UNIT scheduling/telemetry checks; no training or clinical evaluation."""
from contextlib import closing
import io
import json
from pathlib import Path
import random
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from hiercp_v1x.comparison_progress import PhaseProgress, RunningPatientMetrics
from hiercp_v1x.comparison_runtime import _prefetch, closing as close_staged
from hiercp_v1x.u_bridge_training import aggregate_rows, capture_rng, digest, score_row


class Batch:
    def __init__(self, indices):
        self.indices = tuple(indices)
        self.pinned = False

    def pin_memory(self):
        self.pinned = True
        return self


class Provider:
    global_rng_free = True

    def __init__(self):
        self.calls = []

    def batch(self, indices, arm, epoch, training, full=False):
        self.calls.append((tuple(indices), arm, epoch, training, full, threading.current_thread().name))
        return Batch(indices)


class PrefetchUNIT(unittest.TestCase):
    def test_validation_exact_order_flags_remainder_and_no_global_rng(self):
        provider = Provider()
        batches = [[4, 8], [1, 7], [3]]
        with patch('torch.cuda.is_available', return_value=False):
            before = digest(capture_rng())
            results = list(_prefetch(provider, batches, 'native_fixed', 0,
                                    training=False, full=True, pin_memory=False, with_timing=True))
            self.assertEqual(digest(capture_rng()), before)
        self.assertEqual([batch.indices for batch, _ in results], list(map(tuple, batches)))
        self.assertFalse(any(batch.pinned for batch, _ in results))
        for (_, arm, epoch, training, full, name), (_, timing) in zip(provider.calls, results):
            self.assertEqual((arm, epoch, training, full), ('native_fixed', 0, False, True))
            self.assertTrue(name.startswith('comparison-cpu'))
            self.assertGreaterEqual(timing['loader_seconds'], 0.)
            self.assertGreaterEqual(timing['loader_wait_seconds'], 0.)

    def test_training_default_still_pins_same_batches(self):
        provider = Provider()
        results = list(_prefetch(provider, [[1, 2], [3]], 'selected', 7))
        self.assertEqual([batch.indices for batch in results], [(1, 2), (3,)])
        self.assertTrue(all(batch.pinned for batch in results))
        self.assertTrue(all(call[2:5] == (7, True, False) for call in provider.calls))

    def test_exactly_one_following_batch_overlaps_consumption(self):
        second_started = threading.Event()
        release_second = threading.Event()

        class BlockingProvider(Provider):
            def batch(self, indices, *args, **kwargs):
                result = super().batch(indices, *args, **kwargs)
                if indices == [2]:
                    second_started.set()
                    if not release_second.wait(3):
                        raise TimeoutError('UNIT test did not release staged second batch')
                return result

        provider = BlockingProvider()
        with closing(_prefetch(provider, [[1], [2], [3]], 'native', 1)) as staged:
            try:
                first = next(staged)
                self.assertEqual(first.indices, (1,))
                self.assertTrue(second_started.wait(3))
                self.assertEqual([call[0] for call in provider.calls], [(1,), (2,)])
                self.assertEqual(len({call[-1] for call in provider.calls}), 1)
            finally:
                release_second.set()
            self.assertEqual(next(staged).indices, (2,))
            self.assertEqual(next(staged).indices, (3,))
            with self.assertRaises(StopIteration):
                next(staged)

    def test_provider_exception_propagates_at_same_cursor(self):
        class FailedProvider(Provider):
            def batch(self, indices, *args, **kwargs):
                if indices == [3]:
                    raise ValueError('original source graph failure')
                return super().batch(indices, *args, **kwargs)

        consumed = []
        with self.assertRaisesRegex(ValueError, 'original source graph failure'):
            for batch in _prefetch(FailedProvider(), [[1, 2], [3]], 'native', 1):
                consumed.extend(batch.indices)
        self.assertEqual(consumed, [1, 2])

    def test_uncertified_provider_rejected_before_loading_and_empty_safe(self):
        provider = Provider()
        provider.global_rng_free = False
        with self.assertRaisesRegex(ValueError, 'global-RNG-independent'):
            list(_prefetch(provider, [[1]], 'native', 1))
        self.assertEqual(provider.calls, [])
        self.assertEqual(list(_prefetch(Provider(), [], 'native', 1)), [])

    def test_pause_drains_provider_failure_and_preserves_consumer_error(self):
        class FailedNext(Provider):
            def batch(self, indices, *args, **kwargs):
                if indices == [2]:
                    raise ValueError('queued provider failure')
                return super().batch(indices, *args, **kwargs)

        with self.assertRaisesRegex(ValueError, 'queued provider failure'):
            with close_staged(_prefetch(FailedNext(), [[1], [2]], 'native', 1)) as staged:
                self.assertEqual(next(staged).indices, (1,))
        with self.assertRaisesRegex(RuntimeError, 'consumer failure') as raised:
            with close_staged(_prefetch(FailedNext(), [[1], [2]], 'native', 1)) as staged:
                next(staged)
                raise RuntimeError('consumer failure')
        self.assertIsInstance(raised.exception.__cause__, ValueError)


class MetricsUNIT(unittest.TestCase):
    def rows(self):
        values = []
        for index, case, scores in ((0, 'A', [2., 0.]), (1, 'A', [0., 2.]),
                                    (2, 'B', [2., 0.]), (3, 'A', [0., 0.])):
            example = dict(index=index, id=f'{case}:{index}', case_id=case,
                           source_component=1, positive_center=[0, 0, 0], native_centers=[[1, 2, 3]])
            values.append(score_row(torch.tensor(scores), example, ('P', 'U:0'), expected_candidates=2))
        return values

    def test_partial_and_resumed_patient_macro_matches_original_aggregation(self):
        rows = self.rows()
        running = RunningPatientMetrics(rows[:2])
        for boundary in (2, 3, 4):
            if boundary > 2:
                running.add(rows[boundary - 1:boundary])
            expected = aggregate_rows(rows[:boundary])['metrics']
            for key, value in running.metrics().items():
                self.assertAlmostEqual(value, expected[key], places=15)
        self.assertEqual(running.metrics(), RunningPatientMetrics(rows).metrics())
        self.assertEqual(RunningPatientMetrics().metrics(), {})

    def test_duplicate_source_and_nonfinite_are_errors(self):
        row = self.rows()[0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            RunningPatientMetrics([row, row])
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            RunningPatientMetrics([dict(row, mrr=float('nan'))])


class ProgressUNIT(unittest.TestCase):
    def test_one_time_gpu_setup_is_not_multiplied_into_remaining_eta(self):
        # No files or progress thread: exercise only the live ETA formula.
        progress = object.__new__(PhaseProgress)
        progress.initial = 0
        progress.started = 100.
        progress.data = dict(completed_sources=2, total_sources=10,
                             timings=dict(execution_calibration_seconds=120.))
        self.assertEqual(progress._eta(230.), 40.)
        self.assertEqual(progress.data['timings']['execution_calibration_seconds'], 120.)
        progress.data['completed_sources'] = 0
        self.assertIsNone(progress._eta(230.))

    def test_visible_training_count_means_all_eight_candidates(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            stream = io.StringIO()
            with PhaseProgress(root=directory, arm='native', phase='train7', epoch=1,
                               epochs=40, total=151, initial=0, physical_batch=1,
                               stream=stream) as progress:
                self.assertIn('train8', stream.getvalue())
                self.assertNotIn('train7', stream.getvalue())
                progress.finish('PAUSED')

    def progress(self, root, stream, **options):
        return PhaseProgress(root=root, arm='native_listwise', phase='validation129',
                             epoch=3, epochs=40, total=9, initial=2, physical_batch=2,
                             stream=stream, **options)

    def test_start_visible_before_load_and_complete_atomic_receipt(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            stream = io.StringIO()
            with self.progress(directory, stream, interval=60.) as progress:
                self.assertIn('2/9 22% load', stream.getvalue())
                first = json.loads((Path(directory) / 'progress.json').read_text())
                self.assertEqual(first['stage'], 'load')
                self.assertEqual(first['completed_sources'], 2)
                for position in (4, 6, 8, 9):
                    progress.update(stage='forward', completed=position,
                                    metrics={'mrr': .5, 'top1': .25, 'pair_win': .75})
                    progress.update(stage='save', timings={'loader_wait_seconds': 1.5})
                progress.finish()
            final = json.loads((Path(directory) / 'progress.json').read_text())
            self.assertEqual(final['status'], 'COMPLETE')
            self.assertEqual(final['completed_sources'], 9)
            self.assertEqual(final['timings']['loader_wait_seconds'], 6.)
            self.assertEqual(len(stream.getvalue().splitlines()), 2)
            self.assertNotIn('\r', stream.getvalue())
            self.assertEqual(list(Path(directory).iterdir()), [Path(directory) / 'progress.json'])

    def test_blocked_stage_emits_bounded_heartbeats_without_rng_use(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            stream = io.StringIO()
            before = random.getstate()
            with self.progress(directory, stream, interval=.05, heartbeat_interval=.01) as progress:
                threading.Event().wait(.14)
                interim = json.loads((Path(directory) / 'progress.json').read_text())
                self.assertEqual(interim['stage'], 'load')
                self.assertGreater(interim['stage_elapsed_seconds'], .04)
                progress.finish('PAUSED')
            self.assertEqual(random.getstate(), before)
            self.assertGreaterEqual(len(stream.getvalue().splitlines()), 3)
            self.assertLessEqual(len(stream.getvalue().splitlines()), 5)

    def test_narrow_tty_does_not_wrap_and_writer_thread_closes(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            stream = Terminal()
            import os
            with patch('hiercp_v1x.comparison_progress.shutil.get_terminal_size', return_value=os.terminal_size((61, 24))):
                with self.progress(directory, stream) as progress:
                    progress.update(metrics={'mrr': .7, 'top1': .2, 'pair_win': .8})
                    progress.finish('PAUSED')
            for line in stream.getvalue().replace('\n', '\r').split('\r'):
                if line.startswith('native_listwise |'):
                    continue  # Static legend may use normal newline wrapping.
                self.assertLessEqual(len(line), 60)
            self.assertFalse(progress.thread.is_alive())

    def test_tty_keeps_all_three_scores_at_80_columns(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        import os
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            with patch('hiercp_v1x.comparison_progress.shutil.get_terminal_size', return_value=os.terminal_size((80, 24))):
                with self.progress(directory, Terminal()) as progress:
                    progress.update(stage='forward', completed=4, metrics={'mrr': .5, 'top1': .25, 'pair_win': .75}, loss=.421)
                    line = progress._line(progress.stage_started)
                    self.assertIn('M*=', line)
                    self.assertIn('T1=', line)
                    self.assertIn('PW=', line)
                    self.assertIn('L=', line)
                    self.assertLessEqual(len(line), 79)
                    progress.finish('PAUSED')

    def test_progress_cleanup_keeps_model_failure_primary(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            with self.assertRaisesRegex(ValueError, 'model forward failed') as caught:
                with self.progress(directory, io.StringIO()) as progress:
                    progress.failure = OSError('progress write failed')
                    raise ValueError('model forward failed')
            self.assertIsInstance(caught.exception.__cause__, RuntimeError)
            self.assertIsInstance(caught.exception.__cause__.__cause__, OSError)

    def test_very_narrow_display_retains_counts_and_score_values(self):
        import os
        class Terminal(io.StringIO):
            def isatty(self): return True
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            with patch('hiercp_v1x.comparison_progress.shutil.get_terminal_size', return_value=os.terminal_size((40, 24))):
                with self.progress(directory, Terminal()) as progress:
                    progress.update(stage='forward', metrics={'mrr': .123, 'top1': .456, 'pair_win': .789})
                    line = progress._line(progress.stage_started)
                    for value in ('.123', '.456', '.789', '%', '/'):
                        self.assertIn(value, line)
                    self.assertLessEqual(len(line), 39)
                    progress.finish('PAUSED')

    def test_failure_receipt_and_cursor_cannot_regress(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            with self.assertRaisesRegex(ValueError, 'cannot regress'):
                with self.progress(directory, io.StringIO()) as progress:
                    progress.update(completed=1)
            final = json.loads((Path(directory) / 'progress.json').read_text())
            self.assertEqual(final['status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
