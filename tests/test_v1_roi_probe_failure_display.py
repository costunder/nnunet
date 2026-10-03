"""Failure-display UNIT tests; no CT, neural work, cap changes or training."""
import copy
import unittest

from hiercp_v1x.roi_budget_probe import format_probe_result, summarize_probe_failure


def failed(**fields):
    row = dict(case_id='liver_129', sample_index=1, status='FAILED',
               peak_rss_bytes=2*2**30, wall_seconds=6.29, returncode=1,
               error='', stdout='')
    row.update(fields)
    return row


class RoiProbeFailureDisplayUnits(unittest.TestCase):
    def test_actual_qualified_exception_is_shown(self):
        row = failed(error='Traceback (most recent call last):\n  File "probe.py", line 9\n'
                           '    worker()\nhiercp.spatial.AdaptiveRoiBudgetError: exact guard detail\n')
        summary = summarize_probe_failure(row)
        self.assertEqual(summary['error_class'], 'hiercp.spatial.AdaptiveRoiBudgetError')
        self.assertEqual(summary['message'], 'exact guard detail')
        self.assertIn('AdaptiveRoiBudgetError: exact guard detail', format_probe_result(row))

    def test_final_chained_exception_is_selected(self):
        row = failed(error='Traceback (most recent call last):\nValueError: old cause\n\n'
            'The above exception was the direct cause of the following exception:\n\n'
            'Traceback (most recent call last):\n  File "probe.py", line 9\n'
            'RuntimeError: propagated failure\n')
        self.assertEqual(summarize_probe_failure(row)['error_class'], 'RuntimeError')
        self.assertEqual(summarize_probe_failure(row)['message'], 'propagated failure')

    def test_multiline_message_is_one_terminal_line(self):
        row = failed(error='Traceback (most recent call last):\nValueError: geometry mismatch\n'
                           'requested shape differs\nactual full mask unchanged\n')
        text = format_probe_result(row)
        self.assertNotIn('\n', text)
        self.assertIn('geometry mismatch requested shape differs actual full mask unchanged', text)

    def test_multiline_colon_detail_is_not_an_exception_header(self):
        row = failed(error='Traceback (most recent call last):\n  File "probe.py", line 9\n'
                     '    worker()\nValueError: geometry mismatch\n'
                     'reason: actual shape differs\nTip: measure before changing the guard\n')
        summary = summarize_probe_failure(row)
        self.assertEqual(summary['error_class'], 'ValueError')
        self.assertEqual(summary['message'],
                         'geometry mismatch reason: actual shape differs Tip: measure before changing the guard')

    def test_raw_outputs_are_preserved_when_summary_is_truncated(self):
        row = failed(error='Traceback (most recent call last):\nValueError: '+('x'*2000)+'\n',
                     stdout='full child stdout\n')
        original = copy.deepcopy(row)
        summary = summarize_probe_failure(row, message_limit=24)
        self.assertTrue(summary['message_truncated'])
        self.assertIn('[full output in report]', summary['message'])
        self.assertEqual(row, original)
        self.assertLess(len(format_probe_result(row)), 700)

    def test_unclassified_stderr_is_not_called_an_exception(self):
        summary = summarize_probe_failure(failed(error='Warning: preceding message\nprocess crashed\n'))
        self.assertIsNone(summary['error_class'])
        self.assertEqual(summary['source'], 'stderr_unclassified')
        self.assertEqual(summary['message'], 'process crashed')

    def test_stdout_only_is_labeled_without_guessed_cause(self):
        summary = summarize_probe_failure(failed(stdout='progress\nchild output without stderr\n'))
        self.assertIsNone(summary['error_class'])
        self.assertEqual(summary['source'], 'stdout_unclassified')
        self.assertEqual(summary['message'], 'child output without stderr')

    def test_missing_answer_and_unknown_exit_are_distinct(self):
        missing = summarize_probe_failure(failed(returncode=0, answer_missing=True))
        self.assertEqual(missing['source'], 'child_exit_without_answer')
        self.assertIn('returned 0', missing['message'])
        unknown = summarize_probe_failure(failed(returncode=-11))
        self.assertEqual(unknown['source'], 'child_exit_without_output')
        self.assertIn('cause is unknown', unknown['message'])
        self.assertIsNone(unknown['error_class'])

    def test_resource_stop_is_not_a_child_exception(self):
        for status in ('RSS_BUDGET', 'TIME_BUDGET'):
            summary = summarize_probe_failure(failed(status=status, returncode=-15))
            self.assertEqual(summary['source'], 'diagnostic_resource_budget')
            self.assertIsNone(summary['error_class'])
            self.assertIn(status, summary['message'])

    def test_message_less_exception_and_pass_output(self):
        summary = summarize_probe_failure(failed(error='Traceback (most recent call last):\nAssertionError\n'))
        self.assertEqual(summary['error_class'], 'AssertionError')
        self.assertEqual(summary['message'], '')
        row = failed(status='PASS')
        self.assertEqual(format_probe_result(row),
                         'CPU ROI DEBUG | liver_129[1] | PASS | RSS=2.000 GiB | wall=6.29s')
        with self.assertRaises(ValueError):
            summarize_probe_failure(row)

    def test_invalid_display_limit_is_rejected(self):
        for limit in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                summarize_probe_failure(failed(), message_limit=limit)

    def test_message_less_exception_precedes_detail_lines(self):
        row = failed(error='Traceback (most recent call last):\nAssertionError\nTip: later detail\n')
        summary = summarize_probe_failure(row)
        self.assertEqual(summary['error_class'], 'AssertionError')
        self.assertEqual(summary['message'], 'Tip: later detail')


if __name__ == '__main__':
    unittest.main()
