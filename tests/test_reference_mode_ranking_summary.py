"""Saved-measurement formatting checks; no CT, torch, model or optimizer run."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from tools.local_cnn_causal_summary import format_causal_summary
from tools.summarize_local_cnn_reference import format_causal, main, read_summary


REPORT = Path(__file__).resolve().parents[1] / 'validation/reference_causal_20261001/actual_CT_DEBUG_report.json'


class SavedModeRankingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = REPORT.read_bytes()
        cls.report = json.loads(cls.raw)

    def test_saved_gpu_measurements_are_shown_from_exact_ranking_paths(self):
        before = copy.deepcopy(self.report)
        summary = format_causal(self.report, mode_ranking=True)
        count = 0
        for branch in self.report['comparison']['branches']:
            for update in branch['updates']:
                control = update.get('causal', {}).get('mode_control', {})
                for mode in control.get('modes', []):
                    route = mode['per_loss_query_CNN_gradients']['ranking']
                    expected = (' weighted-rank-loss='+format(route['weighted_loss'], '.12g')
                        +' rank-query-grad='+format(route['query_embedding_gradient_norm'], '.12g')
                        +' rank-CNN-grad='+format(route['CNN_parameter_gradient_norm'], '.12g'))
                    line = next(line for line in summary.splitlines()
                                if line.startswith('    BN/dropout '+mode['mode']+' |'))
                    self.assertIn(expected, line)
                    self.assertIn(' align-query-grad=', line)
                    count += 1
        self.assertEqual(count, 4)
        self.assertEqual(self.report, before)
        self.assertEqual(REPORT.read_bytes(), self.raw)

    def test_default_bytes_stay_unchanged_even_when_routes_are_present(self):
        baseline = format_causal(self.report)
        self.assertEqual(baseline, format_causal(self.report, mode_ranking=False))
        opt_in = format_causal(self.report, mode_ranking=True)
        restored = '\n'.join(line.split(' weighted-rank-loss=', 1)[0]
                             for line in opt_in.splitlines())+'\n'
        self.assertEqual(restored, baseline)
        self.assertNotIn('rank-query-grad=', baseline)

    def test_unit_output_fixture_preserves_zero_versus_missing_without_fallback(self):
        # Explicit UNIT formatting fixture, never represented as a GPU measurement.
        modes = [dict(mode='UNIT_zero', per_loss_query_CNN_gradients=dict(
            ranking=dict(weighted_loss=0., query_embedding_gradient_norm=0.,
                         CNN_parameter_gradient_norm=0.))),
            dict(mode='UNIT_partial', per_loss_query_CNN_gradients=dict(
                ranking=dict(weighted_loss=.25, query_embedding_gradient_norm=None))),
            dict(mode='UNIT_missing', weighted_loss=999., terms=dict(ranking_loss=999.))]
        comparison = dict(causal_probe=True, branches=[dict(branch='UNIT', updates=[dict(
            causal=dict(mode_control=dict(modes=modes)))])])
        before = copy.deepcopy(comparison)
        summary = format_causal_summary(comparison, mode_ranking=True)
        self.assertIn('weighted-rank-loss=0 rank-query-grad=0 rank-CNN-grad=0', summary)
        self.assertIn('weighted-rank-loss=0.25 rank-query-grad=UNAVAILABLE rank-CNN-grad=UNAVAILABLE', summary)
        self.assertIn('weighted-rank-loss=UNAVAILABLE rank-query-grad=UNAVAILABLE rank-CNN-grad=UNAVAILABLE', summary)
        self.assertNotIn('weighted-rank-loss=999', summary)
        self.assertEqual(comparison, before)

    def test_reader_opt_in_uses_no_model_dependency_and_rejects_wrong_mode(self):
        expected = format_causal(self.report, mode_ranking=True)
        with patch.dict(sys.modules, {'torch': None, 'tools.local_cnn_reference_mode_probe': None}), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            main([str(REPORT), '--causal', '--mode-ranking'])
        self.assertEqual(output.getvalue(), expected)
        self.assertEqual(read_summary(REPORT, causal=True, mode_ranking=True), expected)
        with self.assertRaisesRegex(ValueError, 'require the causal summary'):
            read_summary(REPORT, mode_ranking=True)
        with self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
            main([str(REPORT), '--mode-ranking'])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(REPORT.read_bytes(), self.raw)


if __name__ == '__main__':
    unittest.main()
