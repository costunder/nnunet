"""CPU-only tests for the PBS bootstrap's duration and six-arm admission."""
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import start_basic_cp_single_parent_pbs as start


class PBSBootstrapAdmissionTests(unittest.TestCase):
    def test_hours_beyond_twenty_four_and_invalid_elapsed(self):
        self.assertEqual(start.duration('72:00:00'), 259200)
        for value in ('72:60:00', '72:00:60', '-1:00:00', '72:00', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                start.duration(value)

    def fixture(self):
        return dict(format=start.INPUT_FORMAT, debug=False, retained_complete_arms=['blend50'],
            canonical_code=str(start.ROOT), source_snapshot={'path': '/input/snapshot', 'sha256': 'a' * 64},
            bindings={'path': '/input/bindings', 'sha256': 'b' * 64},
            VRAM_measurement_evidence={'path': '/input/debug', 'sha256': 'c' * 64},
            output_root='/dev/shm/aicompetition06_basic_cp/experiments')

    def test_exact_six_unfinished_arms_are_admitted(self):
        value = self.fixture()
        with patch.object(start, 'pin', side_effect=lambda ref: Path(ref['path'])), \
                patch.object(start, 'read', return_value={arm: {} for arm in start.ARMS}):
            self.assertIs(start.admit_inputs(value), value)

    def test_completed_blend50_extra_missing_arms_or_debug_cannot_launch(self):
        for failure in ('blend', 'debug', 'extra', 'missing', 'code'):
            value = self.fixture()
            arms = {arm: {} for arm in start.ARMS}
            if failure == 'blend': value['retained_complete_arms'] = []
            if failure == 'debug': value['debug'] = True
            if failure == 'extra': arms['blend50'] = {}
            if failure == 'missing': arms.pop(start.ARMS[0])
            if failure == 'code': value['canonical_code'] = '/another/code/copy'
            with self.subTest(failure=failure), \
                    patch.object(start, 'pin', side_effect=lambda ref: Path(ref['path'])), \
                    patch.object(start, 'read', return_value=arms), self.assertRaises(ValueError):
                start.admit_inputs(value)


if __name__ == '__main__':
    unittest.main()
