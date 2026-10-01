"""Saved-JSON rendering checks only; fixtures are not CT/model measurements."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from tools.summarize_local_cnn_reference import format_signals, format_summary


ROOT = Path(__file__).resolve().parents[1]
FORMATTERS = (format_summary, format_signals)


def fixture():
    tile = dict(schedule_index=17, case_id='formatting_case', physical_batch=32,
                observed=4, unobserved=28, ranking_pairs=112)
    return dict(format='local_cnn_reference_comparison_debug_v1',
        comparison=dict(diagnostic_only=True, steps_per_branch=1,
            update_selection=dict(selection_policy='rankable_full_batch_prefix',
                full_schedule_tiles=544, full_cohort_observations=12800,
                full_schedule_audit=dict(physical_batch=32),
                selected_schedule_indices=[17], selected_tiles=[tile],
                normalization=dict(steps=544, pairs=64000,
                    scope='unchanged full-cohort LiveContext; never selected-prefix normalization')),
            branches=[dict(branch='formatting_branch', updates=[dict(step=1, **tile,
                terms=dict(ranking_pairs=112, ranking_loss=0.00052,
                    observation_auxiliary_loss=0.3, alignment_loss=0.2),
                ranking_parameter_gradient=dict(status='MEASURED', weighted_loss=0.00052,
                    module_gradient_norms=dict(CNN=1.2e-7, readout_fusion=2.3e-6,
                                               L1=3.4e-5, L2=4.5e-4)))] )]))


class ReferenceRankableSummaryTest(unittest.TestCase):
    def test_old_saved_fixture_outputs_are_byte_identical(self):
        baselines = {
            'validation/reference_transfer_20261001/native_ct_report.json': (
                '8d5fd52af4d85ef73e58a480130ecc55af137101c70203370d5b1744ab6ebac3',
                'c5535b8f7b6aaf4bafea7c725af743248f5c19ccd590357dc4542059d50a58a0'),
            'validation/reference_l1_20261001/native_ct_report.json': (
                'cfb56f3c738062c8ddac4ad81253c64c6a1016ff836c14857c75f40838eb0637',
                '8c0441726f116efdf2daabba1e66eb0c75440196f846f6fef2348e7d825e6c2f'),
        }
        for relative, expected_hashes in baselines.items():
            report = json.loads((ROOT / relative).read_text(encoding='utf-8'))
            for formatter, expected in zip(FORMATTERS, expected_hashes):
                with self.subTest(report=relative, formatter=formatter.__name__):
                    output = formatter(report).encode('utf-8')
                    self.assertEqual(hashlib.sha256(output).hexdigest(), expected)

    def test_both_formats_expose_original_selection_and_ranking_gradients(self):
        report = fixture()
        before = copy.deepcopy(report)
        for formatter in FORMATTERS:
            with self.subTest(formatter=formatter.__name__):
                text = formatter(report)
                self.assertIn('DEBUG update selection | policy=rankable_full_batch_prefix', text)
                self.assertIn('original_schedule_tiles=544 | full_cohort_observations=12800', text)
                self.assertIn('configured_batch=32 | selected_schedule_indices=[17]', text)
                self.assertIn('original normalization | steps=544 | ranking_pairs=64000', text)
                self.assertIn('selected tile | schedule_index=17 | case=formatting_case | P=4 | U=28 | actual_batch=32 | ranking_pairs=112', text)
                self.assertIn('update step=1 | schedule_index=17 | P=4 | U=28 | actual_batch=32 | ranking_pairs=112', text)
                self.assertIn('ranking-only parameter gradients | status=MEASURED | weighted_loss=0.00052', text)
                self.assertIn('CNN=1.2e-07 | readout_fusion=2.3e-06 | L1=3.4e-05 | L2=0.00045', text)
        self.assertEqual(report, before)

    def test_not_run_and_missing_saved_norms_remain_explicit(self):
        report = fixture()
        gradient = report['comparison']['branches'][0]['updates'][0]['ranking_parameter_gradient']
        gradient.clear()
        gradient.update(status='NOT_RUN', reason='historical complete-prefix control')
        for formatter in FORMATTERS:
            with self.subTest(formatter=formatter.__name__):
                text = formatter(report)
                self.assertIn('status=NOT_RUN | historical complete-prefix control', text)
                self.assertIn('weighted_loss=unavailable', text)
                self.assertIn('CNN=unavailable | readout_fusion=unavailable | L1=unavailable | L2=unavailable', text)
                self.assertNotIn('CNN=0', text)

    def test_new_numeric_fields_and_selection_lists_are_validated(self):
        for formatter in FORMATTERS:
            report = fixture()
            gradient = report['comparison']['branches'][0]['updates'][0]['ranking_parameter_gradient']
            gradient['module_gradient_norms']['CNN'] = True
            with self.subTest(formatter=formatter.__name__, field='gradient'):
                with self.assertRaisesRegex(ValueError, 'finite number'):
                    formatter(report)
            report = fixture()
            report['comparison']['update_selection']['selected_schedule_indices'] = '17'
            with self.subTest(formatter=formatter.__name__, field='positions'):
                with self.assertRaisesRegex(ValueError, 'expected list'):
                    formatter(report)


if __name__ == '__main__':
    unittest.main()
