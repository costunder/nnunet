"""Read-only terminal CLI contracts; no neural or server execution."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.export_v1_v22_transition_receipt import main


class TransitionTerminalCLIUnitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='terminal_cli_UNIT_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'metadata'
        self.args = [value for name in ('native-run', 'baseline', 'half-a', 'half-b')
                     for value in ('--' + name, str(self.root / name))]
        self.args += ['--output', str(self.output)]

    def test_default_is_no_archive_and_prints_summary_without_dumping_record_inventory(self):
        report = {'scope': 'UNIT metadata fixture', 'training_started': False}
        summary = {'scope': 'UNIT terminal rendering', 'training_started': False}
        stream = io.StringIO()
        with patch('hiercp_v1x.native_transition_receipt.export_receipt', return_value=(report, None)) as export, \
                patch('hiercp_v1x.transition_terminal_report.build_terminal_summary', return_value=summary) as build, \
                patch('hiercp_v1x.transition_terminal_report.render_terminal_summary', return_value='UNIT TERMINAL SUMMARY'), \
                contextlib.redirect_stdout(stream):
            main(self.args)
        self.assertIs(export.call_args.kwargs['create_archive'], False)
        build.assert_called_once_with(report, self.output)
        self.assertIn('UNIT TERMINAL SUMMARY', stream.getvalue())
        self.assertNotIn('OPTIONAL LOCAL ARCHIVE', stream.getvalue())
        path = self.root / 'metadata.terminal_summary.json'
        self.assertEqual(json.loads(path.read_text(encoding='utf8')), summary)
        self.assertEqual(summary['full_summary_path'], str(path.resolve()))
        self.assertFalse(self.output.exists())  # Mock collector never wrote this receipt.

    def test_archive_is_only_requested_by_explicit_flag(self):
        bundle = self.output / 'transition_receipt.zip'
        with patch('hiercp_v1x.native_transition_receipt.export_receipt', return_value=({}, bundle)) as export, \
                patch('hiercp_v1x.transition_terminal_report.build_terminal_summary', return_value={}), \
                patch('hiercp_v1x.transition_terminal_report.render_terminal_summary', return_value='UNIT'), \
                contextlib.redirect_stdout(io.StringIO()):
            main(self.args + ['--archive'])
        self.assertIs(export.call_args.kwargs['create_archive'], True)

    def test_previous_summary_is_preserved_before_collection(self):
        path = self.root / 'metadata.terminal_summary.json'
        path.write_text('preserved UNIT evidence', encoding='utf8')
        with patch('hiercp_v1x.native_transition_receipt.export_receipt') as export:
            with self.assertRaises(FileExistsError):
                main(self.args)
        export.assert_not_called()
        self.assertEqual(path.read_text(encoding='utf8'), 'preserved UNIT evidence')


if __name__ == '__main__':
    unittest.main()
