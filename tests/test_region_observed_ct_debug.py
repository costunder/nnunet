"""DEBUG label-policy tests using native NumPy region construction, not medical data."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from hiercp.region import (REGION_CACHE_FORMAT, REGION_CACHE_SEED_SALT,
                          REGION_DESCRIPTOR_POLICY, _context_only_image,
                          _observed_context_image, build_patient_regions)
from hiercp.schema import GraphBuildConfig


def _case(image, radius):
    labels = np.ones(image.shape, dtype=np.int16)
    center = np.asarray(image.shape) // 2
    bounds = tuple(slice(int(n - radius), int(n + radius + 1)) for n in center)
    labels[bounds] = 2
    return SimpleNamespace(image=image.copy(), label=labels, shape=image.shape,
                           spacing=(1., 1., 1.), paths=SimpleNamespace(case_id="debug_observed_ct"))


class RegionObservedCTDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        payload = json.loads((Path(__file__).resolve().parents[1] / "config/train.json").read_text())
        cls.config = GraphBuildConfig(**payload["graph"])
        assert cls.config.num_regions == 24
        xyz = np.indices((17, 17, 17), dtype=np.float32)
        cls.image = 20 + 3 * xyz[0] + 2 * xyz[1] + xyz[2]

    def build(self, case):
        return build_patient_regions(case, config=self.config, liver_label=1, tumor_label=2,
                                     ct_clip=(-200., 300.), rng=np.random.default_rng(42))

    def test_fixed_image_and_organ_union_are_invariant_to_tumor_annotation(self):
        small, large = _case(self.image, 1), _case(self.image, 4)
        np.testing.assert_array_equal(small.label > 0, large.label > 0)
        self.assertNotEqual(int((small.label == 2).sum()), int((large.label == 2).sum()))
        left, right = self.build(small), self.build(large)
        np.testing.assert_array_equal(left.region_labels, right.region_labels)
        np.testing.assert_array_equal(left.region_features, right.region_features)
        self.assertEqual(left.region_features.shape, (24, 16))
        np.testing.assert_array_equal(small.image, self.image)
        np.testing.assert_array_equal(large.image, self.image)

    def test_observed_image_information_is_not_erased_or_constant(self):
        original, changed = _case(self.image, 2), _case(self.image + 30, 2)
        left, right = self.build(original), self.build(changed)
        np.testing.assert_array_equal(left.region_labels, right.region_labels)
        self.assertTrue(np.any(np.abs(left.region_features[:, 6] - right.region_features[:, 6]) > 1e-4))
        self.assertTrue(np.any(left.region_features[:, 7] > 0))
        np.testing.assert_array_equal(_observed_context_image(original), self.image)
        np.testing.assert_array_equal(_context_only_image(original, liver_label=1, tumor_label=2), self.image)

    def test_version_changes_without_geometry_seed_change(self):
        self.assertEqual(REGION_CACHE_FORMAT, "hiercp_patient_regions_v3")
        self.assertEqual(REGION_DESCRIPTOR_POLICY, "observed_ct_organ_union_v2")
        self.assertEqual(REGION_CACHE_SEED_SALT, "patient_regions_v2")
        bad = _case(self.image, 2)
        bad.image[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            _observed_context_image(bad)


if __name__ == "__main__":
    unittest.main()
