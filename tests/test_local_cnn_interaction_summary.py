"""Saved interaction summaries: no model computation or performance claims."""
import copy
import json
import subprocess
import sys
import unittest

from tests.test_local_cnn_diagnosis_summary import fixture
from tools.summarize_local_cnn_diagnosis import format_summary


def interaction_fixture():
    report = fixture()
    head = dict(candidate_weight_variance_mean=1.508377e-20,
                mean_pairwise_cosine=.9999998, mean_pairwise_jensen_shannon=3.7783e-17)
    branches = []
    for name, scale, win in (('legacy_additive', None, .5553977),
                             ('diagnostic_l1_additive_plus_dot_v1', 0., .5553977),
                             ('diagnostic_l1_additive_plus_dot_v1', 1., .5319602)):
        branches.append(dict(branch=name, interaction_scale=scale,
            support_query_equation_matched=True,
            score=dict(score_std=1.820107e-5, pair_win_rate=win, mean_pairwise_loss=.6931452),
            layers=[dict(layer=2, attention=dict(per_head=[dict(head, head=i) for i in range(4)]))]))
    report['cases'][0]['l1_interaction_candidate'] = dict(
        scale_zero_production_parity_passed=True, original_weights_unchanged=True, branches=branches)

    def evaluation(mrr, win, loss, nested):
        metrics = dict(ranking_mrr=mrr, pair_win_rate=win, ranking_pairwise_loss=loss)
        return dict(metrics=metrics) if nested else metrics
    update_branches = []
    for name, scale in (('baseline', 0.), ('candidate', 1.)):
        update_branches.append(dict(branch=name, scale=scale,
            before=dict(train=evaluation(.1666667, .5693044, .6927898, False),
                        validation=evaluation(.1310241, .4383371, .6933103, True)),
            after=dict(train=evaluation(.1979167, .5884577, .6908739, False),
                       validation=evaluation(.03635204, .3618862, .6935976, True)),
            updates=[dict(module_gradient_norms=dict(CNN=.003)),
                     dict(module_gradient_norms=dict(CNN=.006))],
            module_parameter_delta_norms=dict(CNN=.01234567), prefix_seconds=2.2834567))
    report['l1_interaction_updates'] = dict(
        cloned_optimizer_updates_per_branch=2, physical_batch=32,
        prefix_unique_observations=63, full_cohort_observations=11279,
        optimizer=dict(history='fresh reset; no saved optimizer moments'),
        exact_resume=False, next_saved_update=False, branches=update_branches,
        production_optimizer_updates=0, checkpoints_written=0,
        original_weights_types_methods_modes_preserved=True, caller_rng_restored=True)
    return report


class Checks(unittest.TestCase):
    def test_fixed_weight_branches_all_scales_heads_and_small_values_visible(self):
        report = interaction_fixture()
        text = format_summary(report)
        for value in ('L1 INTERACTION fixed-weight', 'beta=legacy', 'beta=0', 'beta=1',
                      '1.820107e-05/0.5553977/0.6931452', '1.820107e-05/0.5319602/0.6931452',
                      'head-var=1.508377e-20/1.508377e-20/1.508377e-20/1.508377e-20',
                      '3.7783e-17', 'min(head-mean cos)=0.9999998'):
            self.assertIn(value, text)
        self.assertEqual(text.count('Interaction diagnostic_l1_additive_plus_dot_v1 beta='), 2)

    def test_full_objective_prefix_eval_layouts_and_cnn_learning_visible(self):
        text = format_summary(interaction_fixture())
        for value in ('cloned fresh-AdamW full-objective prefix', 'steps/branch=2 batch=32',
                      'prefix-observations=63/11279', 'no saved optimizer moments',
                      'exact-resume=false next-saved-update=false',
                      'Update baseline beta=0', 'Update candidate beta=1',
                      'CNN grad min/max=0.003/0.006', 'CNN parameter-delta=0.01234567',
                      '0.1666667/0.5693044/0.6927898 -> 0.1979167/0.5884577/0.6908739',
                      '0.1310241/0.4383371/0.6933103 -> 0.03635204/0.3618862/0.6935976',
                      'Production updates=0 checkpoints=0 original-preserved=true caller-RNG=true',
                      'not production continuation or final CP performance'):
            self.assertIn(value, text)
        self.assertNotIn('performance improved', text)

    def test_missing_optional_fields_leave_legacy_summary_unchanged(self):
        report = fixture()
        text = format_summary(report)
        self.assertNotIn('INTERACTION', text)
        self.assertIn('SHADOW loss rank/CE/align/full=', text)
        self.assertIn('L1 scale=1 parity=true', text)

    def test_actual_prefix_callback_has_split_root_win_and_nested_ranking_metrics(self):
        report = interaction_fixture()
        # Actual CT DEBUG callback layout, reduced to the saved values this
        # formatter reads. No execution or external CT/checkpoint is required.
        actual_shape = dict(metrics=dict(ranking_mrr=1.0,
            ranking_pairwise_loss=.6763873100280762), pair_win_rate=1.0)
        report['l1_interaction_updates']['branches'][0]['before']['train'] = actual_shape
        text = format_summary(report)
        self.assertIn('baseline train MRR/win/loss before -> after: 1/1/0.6763873 ->', text)
        self.assertNotIn('baseline train MRR/win/loss before -> after: 1/unavailable/', text)

    def test_not_run_reason_preserved_without_invented_branch_metrics(self):
        report = fixture()
        report['cases'][0]['l1_interaction_candidate'] = dict(status='NOT_RUN', reason='Explicit query binding missing')
        report['l1_interaction_updates'] = dict(status='NOT_RUN', reason='No explicit prefix requested')
        text = format_summary(report)
        self.assertIn('L1 INTERACTION NOT_RUN: Explicit query binding missing', text)
        self.assertIn('L1 INTERACTION UPDATES NOT_RUN: No explicit prefix requested', text)
        self.assertNotIn('Update candidate beta=', text)
        report['l1_interaction_updates'] = dict(branches=[dict(branch='candidate',
            before=dict(status='NOT_RUN', reason='No matched evaluation provider'), after={})])
        text = format_summary(report)
        self.assertIn('NOT_RUN: No matched evaluation provider -> unavailable/unavailable/unavailable', text)

    def test_missing_gradient_is_not_reduced_as_valid_subset(self):
        report = interaction_fixture()
        report['l1_interaction_updates']['branches'][0]['updates'][1]['module_gradient_norms'].clear()
        text = format_summary(report)
        self.assertIn('Update baseline beta=0 CNN grad min/max=unavailable', text)
        self.assertIn('Update candidate beta=1 CNN grad min/max=0.003/0.006', text)

    def test_nonfinite_null_missing_and_preservation_false_stay_distinct(self):
        report = interaction_fixture()
        branch = report['l1_interaction_updates']['branches'][0]
        branch['updates'][0]['module_gradient_norms']['CNN'] = float('nan')
        branch['module_parameter_delta_norms']['CNN'] = None
        report['l1_interaction_updates']['original_weights_types_methods_modes_preserved'] = False
        report['cases'][0]['l1_interaction_candidate']['branches'][0]['score'].clear()
        text = format_summary(report)
        self.assertIn('CNN grad min/max=nonfinite CNN parameter-delta=undefined', text)
        self.assertIn('original-preserved=false', text)
        self.assertIn('beta=legacy std/win/loss=unavailable/unavailable/unavailable', text)

    def test_report_unchanged_and_four_case_output_wraps_without_pager(self):
        report = interaction_fixture()
        report['cases'] = [copy.deepcopy(report['cases'][0]) for _ in range(4)]
        original = copy.deepcopy(report)
        text = format_summary(report)
        self.assertEqual(report, original)
        self.assertLessEqual(max(map(len, text.splitlines())), 140)
        self.assertLessEqual(len(text.splitlines()), 100)
        self.assertEqual(text.count('L1 INTERACTION fixed-weight'), 4)

    def test_candidate_json_summary_has_no_model_dependency(self):
        code = '''import json,sys
class RejectTorch:
    def find_spec(self,name,*args):
        if name=='torch' or name.startswith('torch.'):
            raise AssertionError('Saved summary must not import torch')
sys.meta_path.insert(0,RejectTorch())
from tools.summarize_local_cnn_diagnosis import format_summary
print(format_summary(json.loads(sys.stdin.read())))
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-B', '-c', code],
            input=json.dumps(interaction_fixture()), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Update candidate beta=1', result.stdout)


if __name__ == '__main__':
    unittest.main()
