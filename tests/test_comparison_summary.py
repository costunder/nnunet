"""UNIT corruption and bounded-display checks; no medical/model execution."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hiercp_v1x.comparison_experiment import FORMAT, digest
from tools.summarize_v19_comparison import invocation_paths, metric_text, summarize


class SummaryTests(unittest.TestCase):
    def test_native_prefix_never_selects_fixed_or_listwise_receipts(self):
        with TemporaryDirectory(prefix='UNIT_v19_arm_receipt_', dir=Path.cwd()) as directory:
            root = Path(directory)
            (root / 'invocations').mkdir()
            for arm in ('selected', 'native', 'native_fixed', 'native_listwise'):
                (root / 'invocations' / (arm + '_' + 'a' * 32 + '.json')).touch()
            for arm in ('selected', 'native', 'native_fixed', 'native_listwise'):
                self.assertEqual([path.name for path in invocation_paths(root, arm)],
                                 [arm + '_' + 'a' * 32 + '.json'])
            with self.assertRaisesRegex(ValueError, 'Unknown declared'):
                invocation_paths(root, 'unknown')

    def test_missing_results_are_pending_without_invented_metrics(self):
        with TemporaryDirectory(prefix='UNIT_v19_summary_', dir=Path.cwd()) as directory:
            root = Path(directory)
            content = dict(format=FORMAT, debug=True, epochs=2,
                           samples=[dict(partition='train'), dict(partition='val')])
            (root/'experiment.json').write_text(json.dumps(dict(content, sha256=digest(content))), encoding='utf8')
            output = StringIO()
            with redirect_stdout(output):
                summarize(root)
            self.assertEqual(output.getvalue().count('no completed/paused invocation'), 4)
            self.assertNotIn('mrr=', output.getvalue())

    def test_changed_experiment_content_is_rejected(self):
        with TemporaryDirectory(prefix='UNIT_v19_summary_', dir=Path.cwd()) as directory:
            root = Path(directory)
            content = dict(format=FORMAT, debug=True, epochs=2, samples=[dict(partition='val')])
            value = dict(content, sha256=digest(content))
            value['epochs'] = 40
            (root/'experiment.json').write_text(json.dumps(value), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'byte-bound'):
                summarize(root)

    def test_partial_candidate_or_nonfinite_scores_cannot_be_full129_metrics(self):
        report = dict(source_problems=1, full_candidate_evaluation=True,
                      joint_upper_once_per_source_problem=True, rows=[dict(candidate_count=128)],
                      metrics=dict(mrr=.5, top1=0., pair_win=.5, pair_loss=.7))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            metric_text(report, 1)
        report['rows'][0]['candidate_count'] = 129
        report['metrics']['mrr'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            metric_text(report, 1)


if __name__ == '__main__':
    unittest.main()
