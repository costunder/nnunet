"""Metric reporting must retain degradation and a consistent validation cohort."""
import csv
import tempfile
import unittest
from pathlib import Path
from l0_regions.ranking_history import measurement,write_history,progress_line


class Checks(unittest.TestCase):
    def metrics(self):
        return dict(ranking_pairs=4,ranking_evaluable_cases=2,ranking_cases_without_observed=0,
            ranking_pairwise_loss=.6,ranking_mrr=.75,ranking_recall_at_1=.5,
            ranking_recall_at_5=1.,ranking_recall_at_10=1.)

    def test_degradation_is_reported_and_resumed_history_preserved(self):
        initial=self.metrics();later=dict(initial,ranking_mrr=.5,ranking_pairwise_loss=.9)
        rows=[measurement(0,0,initial,initial),measurement(1,533,later,initial)]
        self.assertEqual(rows[1]['mrr_change_from_initial'],-.25)
        self.assertIn('MRR=-0.2500',progress_line(rows[1]))
        with tempfile.TemporaryDirectory() as root:
            write_history(root,rows[:1]);write_history(root,rows)
            with (Path(root)/'validation_history.csv').open() as f:
                saved=list(csv.DictReader(f))
            self.assertEqual([r['epoch'] for r in saved],['0','1'])
            self.assertEqual(float(saved[-1]['mrr_change_from_initial']),-.25)

    def test_nonfinite_or_different_cohort_rejected(self):
        initial=self.metrics()
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            measurement(1,533,dict(initial,ranking_mrr=float('nan')),initial)
        with self.assertRaisesRegex(ValueError,'population'):
            measurement(1,533,dict(initial,ranking_pairs=3),initial)


if __name__=='__main__':unittest.main()
