"""Saved JSON formatter checks only; no torch import or model execution."""
import copy
import unittest

from tools.local_cnn_causal_summary import format_causal_summary


def fixture():
    return dict(causal_probe=True, branches=[dict(branch='UNIT', updates=[dict(
        step=1, schedule_index=4, case_id='UNIT', observed=16, unobserved=16, causal={})])])


class FiniteSummaryChecks(unittest.TestCase):
    def test_absent_opt_in_keeps_existing_summary_bytes(self):
        data=fixture()
        original=format_causal_summary(data)
        data['finite_shadow_score']=False
        self.assertEqual(original,format_causal_summary(data))
        self.assertNotIn('finite shadow',original)

    def test_all_arms_precision_ties_and_common_versus_ranking_change(self):
        data=fixture()
        arms={}
        for name,margin in [('no_change',-3.15075610047e-8),('rank_only',5.2e-7),('full',-1.5e-7)]:
            arms[name]=dict(score=dict(mean_positive_minus_unobserved=margin,
                pair_win_rate=.4,exact_tie_rate=.04,score_std=7.99309759714e-7,
                score_min=-.571722745895,score_max=-.571717262268,mean_pairwise_loss=.693147182465),
                synchronized_scoring_seconds=.123456789012,
                score_change_from_no_change=dict(mean=.2,positive_minus_unobserved_mean_change=1e-8))
        data['branches'][0]['updates'][0]['causal']['finite_shadow_scores']=dict(arms=arms)
        before=copy.deepcopy(data)
        text=format_causal_summary(data)
        self.assertEqual(data,before)
        for name in arms:self.assertIn('      '+name+' | margin=',text)
        self.assertIn('half-tie-pair-win(computed)=0.42',text)
        self.assertIn('score std/min/max=7.99309759714e-07/-0.571722745895/-0.571717262268',text)
        self.assertIn('mean-pair-loss=0.693147182465',text)
        self.assertIn('common score change=0.2 P-minus-U change=1e-08',text)

    def test_missing_measured_fields_are_unavailable(self):
        data=fixture()
        data['branches'][0]['updates'][0]['causal']['finite_shadow_scores']=dict(arms={'no_change':{}})
        text=format_causal_summary(data)
        self.assertIn('rank_only | margin=UNAVAILABLE',text)
        self.assertIn('common score change=UNAVAILABLE P-minus-U change=UNAVAILABLE',text)
        self.assertIn('half-tie-pair-win(computed)=UNAVAILABLE',text)


if __name__=='__main__':unittest.main()
