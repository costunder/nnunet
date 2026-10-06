"""DEBUG: preserve the current definition and fail before patch generation."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy import ndimage as ndi
from hiercp.common import stable_case_seed
from hiercp_v222.data import select_comparison_centers, preflight_comparison_centers


class PreflightDebug(unittest.TestCase):
    def test_preserved_rule_and_seed_exactly(self):
        label = np.ones((12,13,14), np.int16); label[2:4,2:4,2:4] = 2
        case = SimpleNamespace(paths=SimpleNamespace(case_id='DEBUG'), label=label, spacing=np.array([1.,1.2,1.5]))
        cfg = dict(seed=42, comparison_centers_per_patient=128)
        contract = dict(blind_radius_mm=2.)
        eligible = np.argwhere((label == 1) & (ndi.distance_transform_edt(label != 2, sampling=case.spacing) > 2+np.linalg.norm(case.spacing)/2))
        rng = np.random.default_rng(stable_case_seed(42, 'DEBUG', 'v222-comparison'))
        expected = eligible[rng.choice(len(eligible), 128, replace=False)]
        np.testing.assert_array_equal(select_comparison_centers(case, contract, cfg), expected)

    def test_zero_candidates_is_explicit_failure(self):
        label = np.ones((6,6,6), np.int16); label[3,3,3] = 2
        case = SimpleNamespace(paths=SimpleNamespace(case_id='DEBUG'), label=label, spacing=np.ones(3))
        with self.assertRaisesRegex(ValueError, 'only 0'):
            select_comparison_centers(case, dict(blind_radius_mm=28.), dict(seed=42, comparison_centers_per_patient=128))

    def test_preflight_failure_never_generates_patch_or_complete_receipt(self):
        label = np.ones((6,6,6), np.int16); label[3,3,3] = 2
        paths = dict(A=SimpleNamespace(case_id='A', image_path=Path('DEBUG_CT'), label_path=Path('DEBUG_GT')))
        case = SimpleNamespace(paths=paths['A'], label=label, spacing=np.ones(3))
        inventory = dict(A=dict(image_sha256='DEBUG', label_sha256='DEBUG'))
        with tempfile.TemporaryDirectory() as d, patch('hiercp_v222.data.load_case', return_value=case), patch('hiercp_v222.data.sha', return_value='DEBUG'), patch('hiercp_v222.data.volume_memory_bound', return_value=1024**2), patch('hiercp_v222.data.context_patch') as build:
            with self.assertRaisesRegex(ValueError, 'only 0'):
                preflight_comparison_centers(paths, inventory, dict(outer_train=['A']),
                    dict(blind_radius_mm=28.), dict(seed=42, comparison_centers_per_patient=128, preparation_workers='auto'), d)
            build.assert_not_called()
            self.assertFalse((Path(d)/'comparison_preflight.json').exists())
            self.assertFalse((Path(d)/'patches').exists())


if __name__ == '__main__': unittest.main()
