"""UNIT console verbosity and exact JSONL retention; no cache/model changes."""
import builtins
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from tools import resume_comparison_cached as runner
from tools.resume_comparison_cached import _CacheEventLogger, _field_read_console


class Clock:
    def __init__(self):
        self.seconds = 0.
    def __call__(self):
        return self.seconds


class CacheLoggingTests(unittest.TestCase):
    def directory(self):
        return TemporaryDirectory(prefix="UNIT_cache_logging_", dir=Path(__file__).resolve().parents[1])

    @staticmethod
    def records(path):
        return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]

    def test_each_cache_event_is_preserved_exactly_without_per_entry_console_spam(self):
        with self.directory() as directory:
            path = Path(directory) / "events.jsonl"
            console = []
            logger = _CacheEventLogger(path, clock=Clock(), console=console.append)
            rows = [dict(format="UNIT", kind="canonical_local", status="reused", key="a" * 64,
                         source="UNIT/source.pt", destination="UNIT/target.pt", bytes=2**30,
                         artifact_sha256="b" * 64, wall_seconds=.1),
                    dict(format="UNIT", kind="whole_case_fields", status="cache_miss", case_id="UNIT_case"),
                    dict(format="UNIT", kind="upper_static", status="binding_miss", helper_kind="lesions")]
            for row in rows:
                logger(row)
            self.assertFalse(console)
            self.assertEqual(self.records(path), rows)
            logger.finish()
            self.assertEqual(len(console), 1)
            self.assertIn("reused=1 miss=1 binding_miss=1", console[0])
            self.assertIn("reused_GiB=1.00", console[0])
            self.assertNotIn("artifact_sha256", console[0])
            self.assertNotIn("a" * 64, console[0])
            self.assertEqual(self.records(path), rows)

    def test_aggregate_cadence_is_once_per_minute_and_end_is_idempotent(self):
        with self.directory() as directory:
            path = Path(directory) / "events.jsonl"
            console, clock = [], Clock()
            logger = _CacheEventLogger(path, clock=clock, console=console.append)
            row = dict(kind="canonical_local", status="cache_miss")
            logger(row)
            clock.seconds = 59.9
            logger(row)
            self.assertFalse(console)
            clock.seconds = 60.
            logger(row)
            self.assertEqual(len(console), 1)
            self.assertIn("preparation", console[0])
            clock.seconds = 119.9
            logger(row)
            self.assertEqual(len(console), 1)
            logger.finish()
            logger.finish()
            self.assertEqual(len(console), 2)
            self.assertIn("end", console[-1])
            self.assertEqual(len(self.records(path)), 4)

    def test_memory_pressure_and_budget_failure_remain_visible_but_compact(self):
        with self.directory() as directory:
            path = Path(directory) / "events.jsonl"
            console = []
            logger = _CacheEventLogger(path, clock=Clock(), console=console.append)
            pressure = dict(stage="host_cache_pressure", rss_before_bytes=180 * 2**30,
                rss_after_bytes=120 * 2**30, rss_limit_bytes=192 * 2**30,
                measured_rss_within_hard_budget=True,
                cache_providers=[dict(UNIT_diagnostic="x" * 2000)], reclamation=dict(UNIT_more="y" * 2000))
            failure = dict(stage="hard_budget_failed", resource="RSS", actual_bytes=200 * 2**30,
                           limit_bytes=192 * 2**30, cuda_bytes=10 * 2**30)
            logger(pressure)
            logger(failure)
            self.assertEqual(len(console), 2)
            self.assertIn("RSS_GiB=180.00->120.00", console[0])
            self.assertIn("Resource budget failed", console[1])
            self.assertIn("actual_GiB=200.00", console[1])
            self.assertTrue(all(len(line) < 500 for line in console))
            self.assertEqual(self.records(path), [pressure, failure])

    def test_explicit_errors_are_visible_and_full_error_stays_in_jsonl(self):
        with self.directory() as directory:
            path = Path(directory) / "events.jsonl"
            console = []
            logger = _CacheEventLogger(path, clock=Clock(), console=console.append)
            row = dict(kind="canonical_local", status="failed", error="UNIT full traceback " + "detail " * 500)
            logger(row)
            self.assertEqual(len(console), 1)
            self.assertIn("Cache execution error", console[0])
            self.assertNotIn("UNIT full traceback", console[0])
            self.assertEqual(self.records(path), [row])

    def test_concurrent_events_remain_complete_atomic_jsonl_records(self):
        with self.directory() as directory:
            path = Path(directory) / "events.jsonl"
            console = []
            logger = _CacheEventLogger(path, clock=Clock(), console=console.append)
            rows = [dict(kind="canonical_local", status="reused", bytes=index + 1,
                         UNIT_id=index, nested=dict(identity="x" * 300)) for index in range(80)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(logger, rows))
            self.assertFalse(console)
            actual = self.records(path)
            self.assertEqual(sorted(actual, key=lambda row: row["UNIT_id"]), rows)
            logger.finish()
            self.assertEqual(len(console), 1)
            self.assertIn("events=80 reused=80", console[0])

    def test_jsonl_write_or_serialization_failure_is_never_hidden(self):
        with self.directory() as directory:
            path = Path(directory) / "missing" / "events.jsonl"
            logger = _CacheEventLogger(path, clock=Clock(), console=lambda line: None)
            with self.assertRaises(FileNotFoundError):
                logger(dict(status="reused", UNIT_id=1))
            self.assertEqual(logger._events, 0)
            valid = Path(directory) / "events.jsonl"
            logger = _CacheEventLogger(valid, clock=Clock(), console=lambda line: None)
            with self.assertRaises(ValueError):
                logger(dict(UNIT_invalid=float("nan")))
            self.assertEqual(logger._events, 0)

    def test_empty_log_does_not_generate_an_unhelpful_final_line(self):
        with self.directory() as directory:
            console = []
            logger = _CacheEventLogger(Path(directory) / "events.jsonl", clock=Clock(), console=console.append)
            logger.finish()
            self.assertFalse(console)

    def test_final_console_failure_does_not_replace_active_training_exception(self):
        with self.directory() as directory:
            path = Path(directory) / "events.jsonl"
            def broken_console(line):
                raise BrokenPipeError("UNIT closed console")
            logger = _CacheEventLogger(path, clock=Clock(), console=broken_console)
            logger(dict(kind="canonical_local", status="cache_miss"))
            original = RuntimeError("UNIT original training failure")
            with self.assertRaises(RuntimeError) as caught:
                try:
                    raise original
                finally:
                    logger.finish()
            self.assertIs(caught.exception, original)
            if hasattr(original, "__notes__"):
                self.assertIn("Final cache-summary console output failed", original.__notes__[0])
            else:
                self.assertTrue(any("Final cache-summary console output failed" in str(arg) for arg in original.args))
            self.assertEqual(self.records(path), [dict(kind="canonical_local", status="cache_miss")])

    def test_final_console_failure_without_original_error_is_not_hidden(self):
        with self.directory() as directory:
            def broken_console(line):
                raise BrokenPipeError("UNIT closed console")
            logger = _CacheEventLogger(Path(directory) / "events.jsonl", clock=Clock(), console=broken_console)
            logger(dict(status="cache_miss"))
            with self.assertRaises(BrokenPipeError):
                logger.finish()


class FieldReadConsoleTests(unittest.TestCase):
    LINE = ('v1.8 whole-case fields case=UNIT_case status=reopened '
            'array_bytes=1024 disk_bytes=2048 wall_seconds=1.250')

    def directory(self):
        return TemporaryDirectory(prefix="UNIT_field_console_", dir=Path(__file__).resolve().parents[1])

    def test_original_single_line_reports_keep_complete_evidence_and_aggregate_seconds(self):
        from hiercp_v1x import u_bridge_data
        with self.directory() as directory:
            path = Path(directory) / "events.jsonl"
            console = []
            logger = _CacheEventLogger(path, clock=Clock(), console=console.append)
            old_builtin = builtins.print
            had_global = 'print' in vars(u_bridge_data)
            old_global = vars(u_bridge_data).get('print')
            with _field_read_console(logger):
                for status in ('built', 'reopened', 'resident_mapping'):
                    u_bridge_data.print(self.LINE.replace('status=reopened', 'status=' + status), flush=True)
                self.assertIs(builtins.print, old_builtin)
            self.assertEqual('print' in vars(u_bridge_data), had_global)
            if had_global:
                self.assertIs(vars(u_bridge_data)['print'], old_global)
            rows = CacheLoggingTests.records(path)
            self.assertEqual([row['status'] for row in rows], ['built', 'reopened', 'resident_mapping'])
            self.assertTrue(all(row['stage'] == 'whole_case_fields_read' for row in rows))
            self.assertTrue(all(row['array_bytes'] == 1024 and row['disk_bytes'] == 2048 for row in rows))
            self.assertEqual(rows[1]['source_line'], self.LINE)
            self.assertFalse(console)
            logger.finish()
            self.assertIn('field_reads=3 field_seconds=3.75', console[0])

    def test_untrusted_status_or_format_is_recorded_without_invented_parsed_fields(self):
        from hiercp_v1x import u_bridge_data
        rows = []
        malformed = self.LINE.replace('status=reopened', 'status=UNIT_unknown')
        with _field_read_console(rows.append):
            u_bridge_data.print(malformed, flush=True)
        self.assertEqual(rows, [dict(format='preserved_v18_field_read_console_v1',
            stage='whole_case_fields_read', source_line=malformed)])

    def test_other_arguments_and_print_parameters_are_forwarded_unchanged(self):
        from hiercp_v1x import u_bridge_data
        rows, forwarded = [], Mock(return_value='UNIT_original_print_return')
        destination = StringIO()
        with patch.dict(vars(u_bridge_data), {'print': forwarded}):
            with _field_read_console(rows.append):
                calls = [(('UNIT normal progress',), {'flush': True}),
                         ((self.LINE,), {'flush': True, 'file': destination}),
                         ((self.LINE,), {'flush': True, 'end': '~'}),
                         ((self.LINE, 'UNIT second argument'), {'flush': True, 'sep': '|'}),
                         ((self.LINE + '\nUNIT second line',), {'flush': True}),
                         ((self.LINE,), {})]
                for args, kwargs in calls:
                    self.assertEqual(u_bridge_data.print(*args, **kwargs), 'UNIT_original_print_return')
                    forwarded.assert_called_with(*args, **kwargs)
                self.assertFalse(rows)
            self.assertIs(vars(u_bridge_data)['print'], forwarded)

    def test_other_module_and_builtin_output_are_not_redirected(self):
        from hiercp_v1x import u_bridge_data
        rows, output = [], StringIO()
        with _field_read_console(rows.append):
            builtins.print(self.LINE, flush=True, file=output)
            self.assertEqual(output.getvalue(), self.LINE + '\n')
            self.assertFalse(rows)
            u_bridge_data.print(self.LINE, flush=True)
        self.assertEqual(len(rows), 1)

    def test_logging_failure_propagates_and_restores_absent_module_print(self):
        from hiercp_v1x import u_bridge_data
        had_global = 'print' in vars(u_bridge_data)
        old_global = vars(u_bridge_data).get('print')
        def failing_event(row):
            raise OSError('UNIT JSONL write failure')
        with self.assertRaisesRegex(OSError, 'UNIT JSONL write failure'):
            with _field_read_console(failing_event):
                u_bridge_data.print(self.LINE, flush=True)
        self.assertEqual('print' in vars(u_bridge_data), had_global)
        if had_global:
            self.assertIs(vars(u_bridge_data)['print'], old_global)

    def test_provider_report_capture_is_connected_to_both_runner_routes(self):
        from hiercp_v1x import u_bridge_data
        from tests.test_comparison_cached_resume import unit_fixture, write_json
        from tools import run_v18_u_bridge, run_v18_independent
        for independent in (False, True):
            with self.subTest(independent=independent), self.directory() as directory:
                base = Path(directory)
                root, repository, _, manifest, supplied = unit_fixture(base)
                supplied.cache_sources[0].mkdir()
                if independent:
                    write_json(root / 'continuation.json', dict(request=dict(
                        source_root=str(base / 'UNIT_old_source'), arm='selected',
                        contract_sha256=manifest['sha256'], data_root=str(root / 'data'))))
                controller = run_v18_independent if independent else run_v18_u_bridge
                had_global = 'print' in vars(u_bridge_data)
                def no_model_controller(args):
                    u_bridge_data.print(self.LINE, flush=True)
                    return dict(status='UNIT_no_model_execution')
                before = (root / 'experiment.json').read_bytes()
                console = StringIO()
                with patch.object(runner, 'ROOT', repository), patch('tools.local_cnn_device.select'), \
                        patch.object(controller, 'run', side_effect=no_model_controller), redirect_stdout(console):
                    result = runner.run(supplied)
                self.assertEqual(result, dict(status='UNIT_no_model_execution'))
                rows = CacheLoggingTests.records(root / 'preparation_reuse.jsonl')
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['source_line'], self.LINE)
                self.assertIn('field_reads=1', console.getvalue())
                self.assertNotIn(self.LINE, console.getvalue())
                self.assertEqual('print' in vars(u_bridge_data), had_global)
                self.assertEqual((root / 'experiment.json').read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
