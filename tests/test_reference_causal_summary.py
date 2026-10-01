"""Saved-JSON/output checks only; no torch import, CT read or model execution."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.local_cnn_causal_summary import format_causal_summary
from tools.summarize_local_cnn_reference import format_causal, main, read_summary


ROOT = Path(__file__).resolve().parents[1]
ACTUAL_REPORT = ROOT / 'validation/reference_causal_20261001/actual_CT_DEBUG_report.json'


def missing_fields_fixture():
    """Older-schema output fixture, never an actual CT/model measurement."""
    return dict(causal_probe=True, branches=[dict(branch='old', fitted_case_ids=['case'],
        fitted_case_timeline=[dict(after_updates=0, evaluation=dict(train=dict(cases=[dict(
            case_id='case', records=199, all_case_candidates_retained=True,
            metrics=dict(ranking_mrr=1.), score=dict(pair_win_rate=.4))])))],
        updates=[dict(step=1, case_id='case', causal=dict(objective_direction=dict(
            delta_cosines=dict(CNN=dict(rank_only_vs_full=dict(cosine=.3))))))])])


class CausalSummaryOutputChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.formatter = staticmethod(format_causal_summary)
        cls.report_bytes = ACTUAL_REPORT.read_bytes()
        cls.actual = json.loads(cls.report_bytes)

    def test_actual_report_names_first_positive_and_all_positive_recall(self):
        summary = self.formatter(self.actual['comparison'])
        self.assertIn('First-positive MRR is reciprocal first-positive rank', summary)
        self.assertIn('recall@k covers all eligible positives', summary)
        self.assertIn('N=133 eligibleP(computed)=5 unknownU(computed)=128', summary)
        self.assertIn('recall@1=0 hits(computed)=0/5 max(computed)=0.2', summary)
        self.assertIn('recall@5=0.2 hits(computed)=1/5 max(computed)=1', summary)
        self.assertIn('recall@10=0.4 hits(computed)=2/5 max(computed)=1', summary)
        self.assertIn('first-positive MRR=0.0714285714286', summary)

    def test_actual_report_preserves_score_precision_and_separates_cosines(self):
        summary = self.formatter(self.actual['comparison'])
        self.assertIn('score std/min/max=0.0143250320107/-0.017829194665/0.0572658628225', summary)
        self.assertIn('mean-pair-loss=0.688906371593', summary)
        self.assertIn('exact-tie=0 strict-pair-win=0.651562511921 half-tie-pair-win(computed)=0.651562511921', summary)
        self.assertIn('cos delta rank-full=0.87208243074 | cos ranking-gradient/full-descent=0.491816683566', summary)
        for branch in self.actual['comparison']['branches']:
            for step in branch['updates']:
                delta = step['causal']['objective_direction']['delta_cosines']
                for module in ('CNN', 'readout_fusion', 'L1', 'L2'):
                    expected = ('cos delta rank-full='+format(delta[module]['rank_only_vs_full']['cosine'], '.12g')
                        +' | cos ranking-gradient/full-descent='
                        +format(delta[module]['ranking_gradient_vs_full_descent']['cosine'], '.12g'))
                    self.assertIn(expected, summary)

    def test_missing_fields_are_unavailable_without_inferred_counts_or_zero_ties(self):
        fixture = missing_fields_fixture()
        summary = self.formatter(fixture)
        self.assertIn('N=199 eligibleP(computed)=UNAVAILABLE unknownU(computed)=UNAVAILABLE', summary)
        self.assertIn('recall@1=UNAVAILABLE hits(computed)=UNAVAILABLE/UNAVAILABLE max(computed)=UNAVAILABLE', summary)
        self.assertIn('score std/min/max=UNAVAILABLE/UNAVAILABLE/UNAVAILABLE', summary)
        self.assertIn('exact-tie=UNAVAILABLE strict-pair-win=0.4 half-tie-pair-win(computed)=UNAVAILABLE', summary)
        self.assertIn('cos delta rank-full=0.3 | cos ranking-gradient/full-descent=UNAVAILABLE', summary)
        self.assertIn('target Fisher raw/normalized | UNAVAILABLE', summary)
        self.assertNotIn('UNDEFINED', summary)

    def test_computed_half_tie_and_recall_ceiling_require_saved_inputs(self):
        fixture = missing_fields_fixture()
        case = fixture['branches'][0]['fitted_case_timeline'][0]['evaluation']['train']['cases'][0]
        case.update(records=4, observed_ranks=[1, 3])
        case['metrics'].update(ranking_recall_at_1=.5, ranking_recall_at_5=1., ranking_recall_at_10=1.)
        case['score'].update(pair_win_rate=0., exact_tie_rate=1., score_std=0., score_min=0., score_max=0.)
        summary = self.formatter(fixture)
        self.assertIn('N=4 eligibleP(computed)=2 unknownU(computed)=2', summary)
        self.assertIn('recall@1=0.5 hits(computed)=1/2 max(computed)=0.5', summary)
        self.assertIn('recall@10=1 hits(computed)=2/2 max(computed)=1', summary)
        self.assertIn('exact-tie=1 strict-pair-win=0 half-tie-pair-win(computed)=0.5', summary)
        case['all_case_candidates_retained'] = False
        self.assertIn('eligibleP(computed)=UNAVAILABLE unknownU(computed)=UNAVAILABLE', self.formatter(fixture))

    def test_formatter_preserves_actual_json_and_older_record(self):
        actual_before = copy.deepcopy(self.actual)
        fixture = missing_fields_fixture()
        fixture_before = copy.deepcopy(fixture)
        source_hash = hashlib.sha256(self.report_bytes).hexdigest()
        self.formatter(self.actual['comparison'])
        self.formatter(fixture)
        self.assertEqual(self.actual, actual_before)
        self.assertEqual(fixture, fixture_before)
        self.assertEqual(hashlib.sha256(ACTUAL_REPORT.read_bytes()).hexdigest(), source_hash)

    def test_formatter_retains_explicit_causal_admission(self):
        with self.assertRaises(ValueError):
            self.formatter(dict(branches=[]))

    def test_reader_default_and_signals_keep_pre_causal_output_bytes(self):
        expected = {
            ('reference_l1_20261001', False): 'cfb56f3c738062c8ddac4ad81253c64c6a1016ff836c14857c75f40838eb0637',
            ('reference_l1_20261001', True): '8c0441726f116efdf2daabba1e66eb0c75440196f846f6fef2348e7d825e6c2f',
            ('reference_transfer_20261001', False): '8d5fd52af4d85ef73e58a480130ecc55af137101c70203370d5b1744ab6ebac3',
            ('reference_transfer_20261001', True): 'c5535b8f7b6aaf4bafea7c725af743248f5c19ccd590357dc4542059d50a58a0',
        }
        for (directory, signals), digest in expected.items():
            path = ROOT / 'validation' / directory / 'native_ct_report.json'
            argv = [str(path)] + (['--signals'] if signals else [])
            output = io.StringIO()
            # Any eager causal formatter access fails these existing modes.
            with patch.dict(sys.modules, {'tools.local_cnn_causal_summary': None}), contextlib.redirect_stdout(output):
                main(argv)
            self.assertEqual(hashlib.sha256(output.getvalue().encode('utf8')).hexdigest(), digest)

    def test_causal_cli_reads_actual_json_and_preserves_exclusive_output(self):
        expected = self.formatter(self.actual['comparison']) + '\n'
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
                'torch': None, 'tools.local_cnn_reference_causal': None}):
            path = Path(directory) / 'new-summary.txt'
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main([str(ACTUAL_REPORT), '--causal', '--output', str(path)])
            self.assertEqual(output.getvalue(), expected)
            self.assertEqual(path.read_bytes(), expected.encode('utf8'))
            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                main([str(ACTUAL_REPORT), '--causal', '--output', str(path)])
            self.assertEqual(path.read_bytes(), expected.encode('utf8'))
        self.assertEqual(ACTUAL_REPORT.read_bytes(), self.report_bytes)

    def test_causal_reader_rejects_unmeasured_comparison_and_conflicting_modes(self):
        for report in (dict(comparison=dict(diagnostic_only=True)),
                       dict(comparison=dict(diagnostic_only=True, causal_probe=False)),
                       dict(comparison=dict(causal_probe=True))):
            with self.assertRaisesRegex(ValueError, 'causal_probe=True'):
                format_causal(report)
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            read_summary(ACTUAL_REPORT, signals=True, causal=True)
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
            main([str(ACTUAL_REPORT), '--causal', '--signals'])
        self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
