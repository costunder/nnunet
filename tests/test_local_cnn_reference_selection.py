"""Diagnostic selection is explicit, stable and independent of scores."""
import unittest
from tools.diagnose_local_cnn_reference import select_cases


class ReferenceCaseSelection(unittest.TestCase):
    def test_keeps_all_case_candidates_for_fixed_eligible_spread(self):
        rows = [dict(case_id=name, target=t) for name in ('a', 'b', 'c', 'd') for t in (0, 1)]
        rows += [dict(case_id='no_positive', target=0), dict(case_id='no_unobserved', target=1)]
        self.assertEqual(select_cases(rows, 2), ['a', 'c'])
        self.assertEqual(select_cases(list(reversed(rows)), 2), ['a', 'c'])
        self.assertEqual(select_cases(rows, 4), ['a', 'b', 'c', 'd'])
        self.assertEqual(len(rows), 10)

    def test_rejects_unavailable_counts_instead_of_shrinking_or_substituting(self):
        rows = [dict(case_id='a', target=0), dict(case_id='a', target=1)]
        for count in (0, -1, 2, True, 1.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                select_cases(rows, count)
        with self.assertRaises(ValueError):
            select_cases([dict(case_id='a', target=0)], 1)


if __name__ == '__main__':
    unittest.main()
