"""UNIT v1.9 exposure and four-arm admission; no CT or neural execution."""
import copy
from contextlib import redirect_stderr
from io import StringIO
import unittest

from hiercp_v1x.comparison_data import ComparisonData
from hiercp_v1x.comparison_experiment import ARMS, joint_calibration
from hiercp_v1x.u_bridge_data import UBridgeData
from tools.run_v19_comparison import parse


def measured_reports():
    return {arm: dict(arm=arm, original_model_and_RNG_preserved=True, reports=[
        dict(physical_batch=n, accepted=True, samples_per_second=float(n),
             actual_sample_indices=list(range(n))) for n in (1, 2, 4)]) for arm in ARMS}


class ComparisonDataTests(unittest.TestCase):
    def setUp(self):
        self.data = object.__new__(ComparisonData)
        self.data._examples = [dict(index=0)]

    def test_fixed_keys_and_rotating_keys_preserve_original_positive(self):
        fixed = ('P', *(f'U:{i}' for i in range(7)))
        seen = set()
        for epoch in range(1, 41):
            self.assertEqual(self.data.candidate_keys(0, 'native_fixed', epoch), fixed)
            rotating = self.data.candidate_keys(0, 'native', epoch)
            self.assertEqual(rotating, self.data.candidate_keys(0, 'native_listwise', epoch))
            self.assertEqual(len(set(rotating)), 8)
            self.assertEqual(rotating[0], 'P')
            seen.update(rotating[1:])
        self.assertEqual(seen, {f'U:{i}' for i in range(128)})

    def test_every_arm_evaluates_the_identical_complete_candidate_universe(self):
        full = ('P', *(f'U:{i}' for i in range(128)))
        for arm in ARMS:
            for epoch in (1, 19, 29, 40):
                self.assertEqual(self.data.candidate_keys(0, arm, epoch, full=True), full)

    def test_actual_batch_and_view_generation_code_is_not_overridden(self):
        self.assertIs(ComparisonData.batch, UBridgeData.batch)
        self.assertIs(ComparisonData.sample, UBridgeData.sample)
        self.assertEqual(self.data.candidate_keys(0, 'selected', 40),
                         ('P', *(f'S:{i}' for i in range(7))))

    def test_bad_arm_or_source_cannot_fallback(self):
        with self.assertRaisesRegex(ValueError, 'Unknown declared'):
            self.data.candidate_keys(0, 'invented', 1)
        with self.assertRaises(IndexError):
            self.data.candidate_keys(1, 'native_fixed', 1)
        with self.assertRaisesRegex(ValueError, 'One-based'):
            self.data.candidate_keys(0, 'native_fixed', 0)


class CalibrationTests(unittest.TestCase):
    def test_shared_batch_uses_measured_slowest_arm_in_all_four(self):
        reports = measured_reports()
        reports['native_listwise']['reports'][2].update(accepted=False)
        reports['native_fixed']['reports'][0]['samples_per_second'] = 1000.
        reports['native']['reports'][0]['samples_per_second'] = .2
        preserved = copy.deepcopy(reports)
        result = joint_calibration(reports, [1, 2, 4])
        self.assertEqual(result['physical_batch'], 2)
        self.assertEqual(result['accepted_common'], [1, 2])
        self.assertEqual(reports, preserved)

    def test_missing_measurement_or_fake_physical_count_is_rejected(self):
        reports = measured_reports()
        del reports['native_fixed']
        with self.assertRaisesRegex(ValueError, 'All four'):
            joint_calibration(reports, [1, 2, 4])
        reports = measured_reports()
        reports['native_fixed']['reports'][1]['actual_sample_indices'] = [0]
        with self.assertRaisesRegex(ValueError, 'real source count'):
            joint_calibration(reports, [1, 2, 4])
        reports = measured_reports()
        reports['native_listwise']['reports'].pop()
        with self.assertRaisesRegex(ValueError, 'every explicit'):
            joint_calibration(reports, [1, 2, 4])

    def test_no_accepted_batch_has_no_automatic_small_fallback(self):
        reports = measured_reports()
        for row in reports['native_fixed']['reports']:
            row['accepted'] = False
        with self.assertRaisesRegex(RuntimeError, 'No explicit'):
            joint_calibration(reports, [1, 2, 4])

    def test_same_physical_sources_and_order_are_required_across_arms(self):
        reports = measured_reports()
        reports['native_fixed']['reports'][1]['actual_sample_indices'] = [1, 0]
        with self.assertRaisesRegex(ValueError, 'source order differs'):
            joint_calibration(reports, [1, 2, 4])
        reports = measured_reports()
        reports['native_listwise']['reports'][1]['actual_sample_indices'] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'real source count'):
            joint_calibration(reports, [1, 2, 4])

    def test_nonfinite_or_duplicate_receipts_are_rejected(self):
        reports = measured_reports()
        reports['native']['reports'][0]['samples_per_second'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite measured'):
            joint_calibration(reports, [1, 2, 4])
        reports = measured_reports()
        reports['native']['reports'].append(reports['native']['reports'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            joint_calibration(reports, [1, 2, 4])


class EntryTests(unittest.TestCase):
    def test_explicit_new_arms_and_separate_full_or_debug_inputs(self):
        base = ['--gpu', '3', '--experiment', 'UNIT_new_v19', '--baseline', 'UNIT_preserved',
                '--inventory', 'UNIT_inventory.json', '--workers', '4', '--batch-candidates', '2',
                '--cuda-gib', '12', '--rss-gib', '32', '--resident-gib', '8',
                '--validation-local-chunk', '8']
        for arm in (*ARMS, 'all'):
            self.assertEqual(parse(base + ['--arm', arm]).arm, arm)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse(base + ['--arm', 'both'])
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse(base + ['--arm', 'all', '--debug-epochs', '1'])


if __name__ == '__main__':
    unittest.main()
