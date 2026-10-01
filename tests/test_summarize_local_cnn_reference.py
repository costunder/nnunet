"""CPU-only checks of saved JSON formatting; no CT/model execution."""
import contextlib
import copy
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from tools.summarize_local_cnn_reference import format_signals, format_summary, main, read_summary


ROOT = Path(__file__).resolve().parents[1]
FIXED_REPORT = ROOT / 'validation/reference_transfer_20261001/native_ct_report.json'
OLD_REPORT = ROOT / 'validation/reference_l1_20261001/native_ct_report.json'


def fixture():
    """Formatting fixture; never represented as an actual CT measurement."""
    return {
        'format': 'local_cnn_reference_comparison_debug_v1',
        'snapshot': {'epoch': 30, 'step': 15990, 'phase': 'optimization'},
        'comparison': {'diagnostic_only': True, 'fixed_weight': True,
                       'physical_batch': 32, 'configured_physical_batch': 32,
                       'native_tile_binding': {'saved_next_optimization_tile': False,
                                               'actual_batch': 32,
                                               'original_physical_batch': 32},
                       'branches': [{'branch': 'legacy', 'modes': [
                           {'mode': 'eval_fresh', 'L0': {'normalized_centered_energy': 8.213e-8},
                            'layers': [{'layer': 1, 'stages': {'output': {'normalized_centered_energy': 3.767e-9}},
                                        'normalized_centered_energy_output_over_input': 0.04586874}],
                            'score': {'status': 'NOT_EVALUABLE', 'reason': 'physical tile has no observed positive',
                                      'observed': 0, 'candidates': 32}}],
                                     'per_loss_query_gradient': {'losses': {
                                         'alignment': {'query_embedding_dependency': False,
                                                       'query_embedding_gradient_norm': None}}}}]}}


class SavedReferenceSummaryTest(unittest.TestCase):
    def test_real_debug_report_covers_every_branch_mode_split_case_and_rank(self):
        report = json.loads(FIXED_REPORT.read_text(encoding='utf-8'))
        summary = format_summary(report)
        for branch in report['comparison']['branches']:
            self.assertIn('BRANCH ' + branch['branch'], summary)
            for mode in branch['modes']:
                self.assertIn('mode=' + mode['mode'], summary)
            for split in ('train', 'validation'):
                group = branch['initial_full_case_evaluation'][split]
                self.assertIn(f'{split} aggregate | MRR={group["metrics"]["ranking_mrr"]}', summary)
                for case in group['cases']:
                    self.assertIn('case=' + case['case_id'], summary)
                    self.assertIn('observed_ranks=' + json.dumps(case['observed_ranks']), summary)
                    for threshold in (1, 5, 10):
                        self.assertIn(f'R@{threshold}={case["metrics"][f"ranking_recall_at_{threshold}"]}', summary)
        self.assertEqual(summary.count('initial_full_case_evaluation (separate from physical tile)'), 4)
        self.assertEqual(summary.count('train aggregate'), 4)
        self.assertEqual(summary.count('validation aggregate'), 4)
        self.assertEqual(summary.count('observed_ranks='), 8)

    def test_real_debug_exact_stored_energies_metrics_and_alignment(self):
        summary = read_summary(FIXED_REPORT)
        self.assertIn('snapshot | epoch=1 | step=4 | phase=complete', summary)
        self.assertIn('L0=0.0003025695914402604', summary)
        self.assertIn('L1_1=0.00030093928216956556', summary)
        self.assertIn('rank_loss=0.6763873100280762', summary)
        self.assertIn('alignment:dep=False,norm=none (no query dependency)', summary)
        self.assertIn('tile saved_next=False', summary)

    def test_old_update_report_preserves_before_after_every_case(self):
        report = json.loads(OLD_REPORT.read_text(encoding='utf-8'))
        summary = format_summary(report)
        self.assertEqual(summary.count('BRANCH '), 2)
        self.assertEqual(summary.count('  before | scope='), 2)
        self.assertEqual(summary.count('  after | scope='), 2)
        for branch in report['comparison']['branches']:
            self.assertIn('stored_updates=' + str(len(branch['updates'])), summary)
            for period in ('before', 'after'):
                for split in ('train', 'validation'):
                    group = branch[period][split]
                    self.assertIn(f'{split} aggregate | MRR={group["metrics"]["ranking_mrr"]}', summary)
                    for case in group['cases']:
                        self.assertIn('observed_ranks=' + json.dumps(case['observed_ranks']), summary)

    def test_missing_and_null_updates_are_unavailable_explicit_empty_is_zero(self):
        report = fixture()
        report['comparison'].pop('fixed_weight')
        report['comparison']['branches'] = [
            {'branch': 'missing'}, {'branch': 'null', 'updates': None},
            {'branch': 'empty', 'updates': []}]
        summary = format_summary(report)
        self.assertEqual(summary.count('stored_updates=unavailable'), 2)
        self.assertEqual(summary.count('stored_updates=0'), 1)

    def test_only_actual_batch_does_not_invent_configuration(self):
        report = fixture()
        comparison = report['comparison']
        comparison.pop('configured_physical_batch')
        comparison.pop('native_tile_binding')
        summary = format_summary(report)
        self.assertIn('actual_batch=32 | configured_batch=unavailable', summary)

    def test_update_schema_physical_batch_does_not_invent_actual_batch(self):
        report = fixture()
        comparison = report['comparison']
        comparison.pop('fixed_weight')
        comparison.pop('native_tile_binding')
        comparison['branches'] = [{'branch': 'legacy', 'updates': []}]
        summary = format_summary(report)
        self.assertIn('actual_batch=unavailable | configured_batch=32', summary)

    def test_gradient_outer_not_run_status_and_reason_are_preserved(self):
        report = fixture()
        report['comparison']['branches'][0]['per_loss_query_gradient'] = {
            'status': 'NOT_RUN', 'reason': 'original explicit resource limit'}
        summary = format_summary(report)
        self.assertIn('query gradients | NOT_RUN | original explicit resource limit', summary)
        self.assertNotIn('ranking:dep=', summary)

    def test_gradient_individual_not_run_status_and_reason_are_preserved(self):
        report = fixture()
        losses = report['comparison']['branches'][0]['per_loss_query_gradient']['losses']
        losses['ranking'] = {'status': 'NOT_RUN', 'reason': 'no P/U pairs'}
        summary = format_summary(report)
        self.assertIn('ranking:NOT_RUN | no P/U pairs', summary)
        self.assertIn('alignment:dep=False,norm=none (no query dependency)', summary)

    def test_case_recalls_are_printed_inline_from_saved_metrics(self):
        report = fixture()
        report['comparison']['branches'][0]['initial_full_case_evaluation'] = {
            'validation': {'cases': [{'case_id': 'retained_case', 'records': 133,
                                     'metrics': {'ranking_recall_at_1': 0.2,
                                                 'ranking_recall_at_5': 0.4,
                                                 'ranking_recall_at_10': 0.6},
                                     'observed_ranks': [1, 5, 10, 30, 40]}]}}
        summary = format_summary(report)
        case_line = next(line for line in summary.splitlines() if 'case=retained_case' in line)
        self.assertIn('R@1=0.2 | R@5=0.4 | R@10=0.6', case_line)
        self.assertIn('observed_ranks=[1, 5, 10, 30, 40]', case_line)

    def test_unavailable_is_not_synthesized_as_zero(self):
        summary = format_summary(fixture())
        self.assertIn('initial_full_case_evaluation (separate from physical tile): unavailable', summary)
        self.assertIn('pair-win=unavailable | observed=0 | candidates=32', summary)
        self.assertIn('ranking:dep=unavailable,norm=unavailable', summary)
        self.assertNotIn('pair-win=0', summary)

    def test_not_evaluable_tile_remains_distinct_from_case_ranking(self):
        report = fixture()
        report['comparison']['branches'][0]['initial_full_case_evaluation'] = {
            'scope': 'all selected case candidates',
            'train': {'metrics': {'ranking_mrr': 0.3}, 'pair_win_rate': 0.6,
                      'cases': [{'case_id': 'different_full_case', 'records': 131,
                                 'observed_ranks': [1, 8, 20], 'score': {'pair_win_rate': 0.6}}]}}
        summary = format_summary(report)
        self.assertIn('NOT_EVALUABLE | physical tile has no observed positive', summary)
        self.assertIn('train aggregate | MRR=0.3 | pair-win=0.6', summary)
        self.assertIn('case=different_full_case | records=131', summary)

    def test_not_run_and_reasons_propagate(self):
        report = fixture()
        branch = report['comparison']['branches'][0]
        branch['initial_full_case_evaluation'] = {'train': {'status': 'NOT_RUN', 'reason': 'explicit resource limit'},
                                                 'validation': {'status': 'NOT_RUN'}}
        branch['modes'].append({'mode': 'eval_after_one_same_tile_bn_pass', 'status': 'NOT_RUN', 'reason': 'not performed'})
        summary = format_summary(report)
        self.assertIn('train: NOT_RUN | explicit resource limit', summary)
        self.assertIn('validation: NOT_RUN', summary)
        self.assertIn('mode=eval_after_one_same_tile_bn_pass | NOT_RUN | not performed', summary)

    def test_every_rank_and_case_is_printed_without_cap(self):
        report = fixture()
        ranks = list(range(1, 258))
        cases = [{'case_id': f'case_{i}', 'records': 385, 'observed_ranks': ranks} for i in range(45)]
        report['comparison']['branches'][0]['initial_full_case_evaluation'] = {'train': {'cases': cases}}
        summary = format_summary(report)
        self.assertEqual(summary.count('    case=case_'), 45)
        self.assertEqual(summary.count('observed_ranks=' + json.dumps(ranks)), 45)
        self.assertIn('case=case_44', summary)
        self.assertNotIn('...', summary)

    def test_reject_nonobject_and_incorrect_scope(self):
        for value in ([], None, 'report'):
            with self.assertRaises(ValueError):
                format_summary(value)
        report = fixture()
        report['comparison']['diagnostic_only'] = False
        with self.assertRaisesRegex(ValueError, 'diagnostic-only'):
            format_summary(report)
        report = fixture()
        report['format'] = 'production_checkpoint'
        with self.assertRaisesRegex(ValueError, 'format'):
            format_summary(report)

    def test_reject_nan_inf_even_in_unformatted_metadata(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            report = fixture()
            report['unused_metadata'] = {'value': value}
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                format_summary(report)

    def test_numeric_bool_is_not_accepted_as_measurement(self):
        report = fixture()
        report['snapshot']['epoch'] = True
        with self.assertRaisesRegex(ValueError, 'finite number'):
            format_summary(report)
        report = fixture()
        report['comparison']['native_tile_binding']['saved_next_optimization_tile'] = 1
        with self.assertRaisesRegex(ValueError, 'boolean'):
            format_summary(report)

    def test_formatter_does_not_mutate_dictionary(self):
        report = fixture()
        before = copy.deepcopy(report)
        format_summary(report)
        self.assertEqual(report, before)

    def test_cli_prints_all_once_and_new_output_matches_preserving_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'report.json'
            target = Path(directory) / 'new.txt'
            raw = json.dumps(fixture(), indent=4).encode('utf-8')
            source.write_bytes(raw)
            expected = read_summary(source)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main([str(source), '--output', str(target)])
            self.assertEqual(stdout.getvalue(), expected)
            self.assertEqual(target.read_text(encoding='utf-8'), expected)
            self.assertEqual(source.read_bytes(), raw)

    def test_existing_output_collision_preserves_both_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'report.json'
            target = Path(directory) / 'existing.txt'
            source.write_text(json.dumps(fixture()), encoding='utf-8')
            target.write_bytes(b'existing user result')
            raw = source.read_bytes()
            with self.assertRaises(FileExistsError):
                main([str(source), '--output', str(target)])
            self.assertEqual(target.read_bytes(), b'existing user result')
            self.assertEqual(source.read_bytes(), raw)

    def test_source_as_output_collision_cannot_overwrite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'report.json'
            source.write_text(json.dumps(fixture()), encoding='utf-8')
            raw = source.read_bytes()
            with self.assertRaises(FileExistsError):
                main([str(source), '--output', str(source)])
            self.assertEqual(source.read_bytes(), raw)

    def test_module_loads_and_formats_without_torch_or_model_access(self):
        class Forbidden(types.ModuleType):
            def __getattr__(self, name):
                raise AssertionError('Model dependency access is forbidden')
        forbidden = {name: Forbidden(name) for name in ('torch', 'torch_geometric', 'hiercp_v222', 'l0_regions')}
        with patch.dict(sys.modules, forbidden):
            module = runpy.run_path(str(ROOT / 'tools/summarize_local_cnn_reference.py'), run_name='saved_only_test')
            summary = module['format_summary'](fixture())
            signals = module['format_signals'](fixture())
        self.assertIn('No model execution', summary)
        self.assertIn('No model execution, score reranking or metric recomputation', signals)

    def test_signal_mode_preserves_every_real_branch_split_case_score_and_trace(self):
        report = json.loads(FIXED_REPORT.read_text(encoding='utf-8'))
        signals = format_signals(report)
        for branch in report['comparison']['branches']:
            self.assertIn('BRANCH ' + branch['branch'], signals)
            self.assertIn('ranking_pairs=' + str(branch['per_loss_query_gradient']['ranking_pairs']), signals)
            for split in ('train', 'validation'):
                self.assertIn('initial_full_case_evaluation ' + split, signals)
                for case in branch['initial_full_case_evaluation'][split]['cases']:
                    self.assertIn('case=' + case['case_id'], signals)
                    for key in ('score_std', 'score_min', 'score_max', 'mean_positive_minus_unobserved', 'exact_tie_rate', 'comparisons'):
                        self.assertIn(key + '=' + str(case['score'][key]), signals)
                    for stage in case['trace']['stages']:
                        self.assertIn(stage['stage'] + '(raw=' + str(stage['centered_energy'])
                                      + ',normalized=' + str(stage['normalized_centered_energy'])
                                      + ',mean_norm=' + str(stage['mean_norm']) + ')', signals)
        self.assertEqual(signals.count('    case='), 8)
        self.assertEqual(signals.count('      trace '), 8)

    def test_signal_mode_uses_stored_prototype_statistics_without_invention(self):
        report = json.loads(FIXED_REPORT.read_text(encoding='utf-8'))
        signals = format_signals(report)
        case_trace = report['comparison']['branches'][0]['initial_full_case_evaluation']['train']['cases'][0]['trace']
        prototype = '/'.join(str(case_trace['prototype_cross_class_cosine_' + key]) for key in ('min', 'max', 'mean'))
        self.assertIn('prototype_cosine min/max/mean=' + prototype, signals)
        self.assertIn('prototype_cosine min/max/mean=unavailable/unavailable/unavailable', signals)

    def test_signal_mode_prints_every_old_update_and_all_stored_terms(self):
        report = json.loads(OLD_REPORT.read_text(encoding='utf-8'))
        signals = format_signals(report)
        total = 0
        for branch in report['comparison']['branches']:
            for update in branch['updates']:
                total += 1
                expected = 'update step=' + str(update['step']) + ' | ' + ' | '.join(
                    key + '=' + str(update['terms'][key])
                    for key in ('ranking_pairs', 'ranking_loss', 'observation_auxiliary_loss', 'alignment_loss'))
                self.assertIn(expected, signals)
        self.assertEqual(signals.count('  update step='), total)
        self.assertEqual(signals.count('    case='), 8)

    def test_signal_mode_missing_and_not_run_are_preserved(self):
        report = fixture()
        branch = report['comparison']['branches'][0]
        branch['per_loss_query_gradient'] = {'status': 'NOT_RUN', 'reason': 'explicit memory limit'}
        branch['initial_full_case_evaluation'] = {
            'train': {'status': 'NOT_RUN', 'reason': 'not evaluated'},
            'validation': {'cases': [{'case_id': 'missing', 'trace': {'stages': None}},
                                     {'case_id': 'blocked', 'score': {'status': 'NOT_RUN', 'reason': 'saved failure'},
                                      'trace': {'status': 'NOT_RUN', 'reason': 'not measured'}}]}}
        signals = format_signals(report)
        self.assertIn('query gradients | NOT_RUN | explicit memory limit', signals)
        self.assertIn('train: NOT_RUN | not evaluated', signals)
        self.assertIn('score_std=unavailable', signals)
        self.assertIn('trace unavailable | prototype_cosine min/max/mean=unavailable/unavailable/unavailable', signals)
        self.assertIn('score_status=NOT_RUN | saved failure', signals)
        self.assertIn('trace=NOT_RUN | not measured', signals)
        self.assertNotIn('score_std=0', signals)

    def test_signal_mode_loss_not_run_propagates_without_fabricating_count(self):
        report = fixture()
        losses = report['comparison']['branches'][0]['per_loss_query_gradient']['losses']
        losses['ranking'] = {'status': 'NOT_RUN', 'reason': 'no positive pairs'}
        signals = format_signals(report)
        self.assertIn('ranking_pairs=unavailable', signals)
        self.assertIn('ranking:NOT_RUN | no positive pairs', signals)

    def test_signal_mode_has_no_update_cap_and_distinguishes_missing_updates(self):
        report = fixture()
        report['comparison'].pop('fixed_weight')
        updates = [{'step': i, 'terms': {'ranking_pairs': 10, 'ranking_loss': i / 100,
                                        'observation_auxiliary_loss': 0.5, 'alignment_loss': 0.01}}
                   for i in range(75)]
        report['comparison']['branches'] = [{'branch': 'all_steps', 'updates': updates},
                                           {'branch': 'missing'}, {'branch': 'null', 'updates': None},
                                           {'branch': 'empty', 'updates': []}]
        signals = format_signals(report)
        self.assertEqual(signals.count('  update step='), 75)
        self.assertIn('update step=74', signals)
        self.assertEqual(signals.count('updates: unavailable'), 2)
        self.assertEqual(signals.count('updates: [] (stored empty list)'), 1)

    def test_signal_cli_prints_once_preserves_json_and_uses_new_output_only(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'report.json'
            target = Path(directory) / 'signals.txt'
            source.write_text(json.dumps(fixture(), indent=2), encoding='utf-8')
            before = source.read_bytes()
            expected = read_summary(source, signals=True)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main([str(source), '--signals', '--output', str(target)])
            self.assertEqual(stdout.getvalue(), expected)
            self.assertEqual(target.read_text(encoding='utf-8'), expected)
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                main([str(source), '--signals', '--output', str(target)])
            self.assertEqual(target.read_text(encoding='utf-8'), expected)

    def test_signal_mode_rejects_wrong_scope_and_numeric_bool(self):
        report = fixture()
        report['comparison']['diagnostic_only'] = False
        with self.assertRaisesRegex(ValueError, 'diagnostic-only'):
            format_signals(report)
        report = fixture()
        report['comparison']['branches'][0]['initial_full_case_evaluation'] = {
            'train': {'cases': [{'case_id': 'invalid', 'score': {'score_std': True}}]}}
        with self.assertRaisesRegex(ValueError, 'finite number'):
            format_signals(report)


if __name__ == '__main__':
    unittest.main()
