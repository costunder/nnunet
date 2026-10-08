"""CPU UNIT failure evidence checks; no data, model, or training execution."""
from __future__ import annotations

import errno
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from hiercp_v1x.comparison_runtime import (
    _StagingCleanupError, _failure_details, _record_training_failure, closing,
)


def geometry_failure(label):
    try:
        raise ValueError('UNIT empty geometry ' + label)
    except ValueError as cause:
        try:
            raise RuntimeError('UNIT canonical geometry wrapper ' + label) from cause
        except RuntimeError as error:
            return error


class FailureTracebackTests(unittest.TestCase):
    def setUp(self):
        self.state = dict(epoch=8, phase='training', position=32, updates=1089,
                          validation_position=0)

    def test_failure_record_keeps_cause_frames_and_cursor_without_locals(self):
        error = geometry_failure('case46')
        rows = []
        _record_training_failure(Path('UNIT-output'), self.state, error,
                                 lambda path, row: rows.append((path, row)))
        path, row = rows[0]
        self.assertEqual(path, Path('UNIT-output/failures.jsonl'))
        self.assertEqual({key: row[key] for key in self.state}, self.state)
        self.assertEqual(row['error'], 'RuntimeError: UNIT canonical geometry wrapper case46')
        self.assertIn('geometry_failure', row['traceback'])
        self.assertIn('ValueError: UNIT empty geometry case46', row['traceback'])
        self.assertIn('direct cause', row['traceback'])
        self.assertGreater(row['observed_at'], 0)
        self.assertEqual(row['exception_graph'][0]['cause'], 1)
        self.assertNotIn('locals', row['exception_graph'][0])
        json.dumps(row, allow_nan=False)

    def test_suppressed_context_is_explicitly_retained(self):
        try:
            raise OSError(errno.EIO, 'UNIT underlying read error')
        except OSError:
            try:
                raise RuntimeError('UNIT hidden context') from None
            except RuntimeError as error:
                details = _failure_details(error)
        self.assertIsNone(details['exception_graph'][0]['cause'])
        self.assertTrue(details['exception_graph'][0]['suppress_context'])
        self.assertEqual(details['exception_graph'][0]['context'], 1)
        self.assertIn('UNIT underlying read error', details['traceback'])

    def test_all_queued_worker_causes_survive_aggregate(self):
        first, second = geometry_failure('first'), geometry_failure('second')
        error = _StagingCleanupError([first, second])
        details = _failure_details(error)
        self.assertEqual(len(details['exception_graph'][0]['staging_failures']), 2)
        for label in ('first', 'second'):
            self.assertIn('ValueError: UNIT empty geometry ' + label, details['traceback'])
        json.dumps(details, allow_nan=False)

    def test_consumer_original_cause_survives_worker_drain_chaining(self):
        primary = geometry_failure('consumer')
        original_cause = primary.__cause__
        queued = geometry_failure('queued')

        class Staged:
            def close(self):
                raise queued

        try:
            with closing(Staged()):
                raise primary
        except RuntimeError as error:
            self.assertIs(error, primary)
            self.assertIs(error.__cause__, queued)
            details = _failure_details(error)
        self.assertIs(primary.comparison_retained_causes[0], original_cause)
        for label in ('consumer', 'queued'):
            self.assertIn('ValueError: UNIT empty geometry ' + label, details['traceback'])
        self.assertTrue(details['exception_graph'][0]['retained_causes'])
        self.assertEqual(details['exception_graph'][0]['secondary_errors'][0]['operation'],
                         'CPU staging drain')

    def test_exception_cycles_are_references_not_recursive_serialization(self):
        first, second = RuntimeError('UNIT cycle first'), ValueError('UNIT cycle second')
        first.__cause__ = second
        second.__context__ = first
        first.comparison_secondary_errors = [('already reported', second)]
        details = _failure_details(first)
        self.assertEqual(len(details['exception_graph']), 2)
        self.assertEqual(details['exception_graph'][1]['context'], 0)
        self.assertEqual(details['exception_graph'][0]['secondary_errors'][0]['exception'], 1)
        json.dumps(details, allow_nan=False)

    def test_failure_audit_quota_does_not_replace_training_error_or_cause(self):
        primary = geometry_failure('case46')
        cause = primary.__cause__
        quota = OSError(getattr(errno, 'EDQUOT', 122), 'UNIT quota during failure append')
        console = io.StringIO()

        def append(path, row):
            raise quota

        with patch('sys.stderr', console):
            try:
                raise primary
            except RuntimeError as error:
                _record_training_failure(Path('UNIT-output'), self.state, error, append)
                self.assertIs(error, primary)
                self.assertIs(error.__cause__, cause)
        self.assertIn('training failure audit', console.getvalue())
        self.assertIs(primary.comparison_secondary_errors[-1][1], quota)
        self.assertEqual(self.state['updates'], 1089)


if __name__ == '__main__':
    unittest.main()
