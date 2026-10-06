"""Production/DEBUG separation at the real native30 evaluator entry point."""
import contextlib
import io
import unittest
from pathlib import Path
from tools.evaluate_native_v1_full128 import parse


class EntryTests(unittest.TestCase):
    def args(self):
        return ['--gpu', '6', '--original-source', '/original', '--inventory', '/inventory.json',
                '--prototype', '/prototype.pt', '--output', '/new', '--workers', '16',
                '--physical-batch-candidates', '8', '16', '32', '--cuda-gib', '40',
                '--rss-gib', '192', '--resident-gib', '128']

    def reject(self, args):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse(args)

    def test_production_explicit_best_and_whole_cohort(self):
        a = parse(self.args() + ['--checkpoint', '/full/model.pt'])
        self.assertEqual(a.gpu, 6)
        self.assertIsNone(a.debug_case_ids)
        self.assertFalse(a.debug_fresh_model)

    def test_production_requires_checkpoint(self):
        self.reject(self.args())

    def test_no_production_subset(self):
        self.reject(self.args() + ['--checkpoint', '/full/model.pt', '--debug-case-ids', 'liver_31'])

    def test_debug_cannot_read_production_weights(self):
        self.reject(self.args() + ['--checkpoint', '/full/model.pt', '--debug-fresh-model',
                                  '--debug-case-ids', 'liver_31'])

    def test_debug_requires_explicit_cases(self):
        self.reject(self.args() + ['--debug-fresh-model'])

    def test_batch_candidates_no_duplicate_or_automatic_small_default(self):
        args = self.args(); index = args.index('8'); args[index:index+3] = ['32', '32']
        self.reject(args + ['--checkpoint', '/full/model.pt'])

    def test_complete_read_only_reuse_and_all_observed_are_explicit(self):
        a = parse(self.args() + ['--checkpoint', '/full/model.pt',
            '--reuse-geometry', '/failed/geometry', '--lesion-policy', 'all_observed'])
        self.assertEqual(a.reuse_geometry, Path('/failed/geometry'))
        self.assertEqual(a.lesion_policy, 'all_observed')
        original = parse(self.args() + ['--checkpoint', '/full/model.pt'])
        self.assertIsNone(original.reuse_geometry)
        self.assertEqual(original.lesion_policy, 'saved_guard')

    def test_unknown_lesion_policy_rejected(self):
        self.reject(self.args() + ['--checkpoint', '/full/model.pt', '--lesion-policy', 'top12'])


if __name__ == '__main__':
    unittest.main()
