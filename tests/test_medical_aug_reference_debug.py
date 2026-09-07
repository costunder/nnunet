"""DEBUG small arrays: reference CP predicates, not training or medical metrics."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy import ndimage as ndi

from hiercp.common import build_candidate_pool, choose_source_tumor, paste_source


class MedicalAugReferenceDebugTests(unittest.TestCase):
    def fixture(self):
        shape = (35, 11, 11)
        image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        label = np.ones(shape, dtype=np.uint8)
        label[17, 5, 5] = 2
        case = SimpleNamespace(image=image, label=label, shape=shape, spacing=(3.0, 0.7, 0.7))
        source, _, _ = choose_source_tumor(image, label, tumor_label=2,
                                          rng=np.random.default_rng(42), selection="random", pad=2)
        return case, source

    def pool(self, case, source, **overrides):
        kwargs = dict(placement_mask=case.label == 1, full_organ_mask=case.label > 0,
                      occupied_mask=case.label == 2, organ_distance=np.ones(case.shape),
                      rng=np.random.default_rng(42), num_candidates=int(np.prod(case.shape)),
                      max_draws=0, force_exhaustive=True, min_liver_coverage=0.85,
                      occupied_clearance_vox=2, min_center_separation_mm=0.0,
                      min_center_separation_vox=12.0)
        kwargs.update(overrides)
        return build_candidate_pool(case, source, **kwargs)

    def test_shared_default_matches_reference_geometry_without_model_reduction(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / "config/train.json").read_text())
        for section in ("cache", "generation"):
            self.assertEqual(cfg[section]["source_pad"], 2)
            self.assertEqual(cfg[section]["occupied_clearance_vox"], 2)
            self.assertEqual(cfg[section]["min_liver_coverage"], 0.85)
            self.assertEqual(cfg[section]["min_center_separation_mm"], 0.0)
            self.assertEqual(cfg[section]["min_center_separation_vox"], 12.0)
            self.assertEqual(cfg[section]["no_placement_policy"], "retain_original")
        self.assertEqual(cfg["cache"]["total_candidates"], 8)
        self.assertEqual(cfg["cache"]["candidate_pool_size"], 128)

    def test_native_voxel_reference_predicate_and_physical_gnn_features(self):
        case, source = self.fixture()
        self.assertEqual(source.patch_mask.shape, (5, 5, 5))
        candidates, physical_distance = self.pool(case, source)
        occupied = case.label == 2
        forbidden = ndi.binary_dilation(occupied, structure=ndi.generate_binary_structure(3, 1), iterations=2)
        voxel_distance = ndi.distance_transform_edt(~occupied)
        expected = set()
        # Original mask/coverage/distance predicates; exhaustive extension also
        # visits the last fully in-bounds patch (legacy random proposals do not).
        for center in np.ndindex(case.shape):
            if any(v < 2 or v + 3 > n for v, n in zip(center, case.shape)):
                continue
            slc = tuple(slice(v - 2, v + 3) for v in center)
            if (case.label[center] == 1 and voxel_distance[center] >= 12 and
                    not np.any(forbidden[slc] & source.patch_mask) and
                    np.mean((case.label[slc] == 1)[source.patch_mask]) >= 0.85):
                expected.add(center)
        self.assertTrue(expected)
        self.assertEqual({candidate.center for candidate in candidates}, expected)
        for candidate in candidates:
            self.assertAlmostEqual(candidate.occupied_distance_mm, physical_distance[candidate.center])
        physical_candidates, _ = self.pool(case, source, min_center_separation_vox=0.0,
                                           min_center_separation_mm=12.0)
        self.assertGreater(len(physical_candidates), len(candidates))

    def test_random_and_exhaustive_share_voxel_predicate(self):
        case, source = self.fixture()
        candidates, _ = self.pool(case, source, num_candidates=128, max_draws=50000,
                                  force_exhaustive=False)
        distance = ndi.distance_transform_edt(case.label != 2)
        self.assertEqual(len(candidates), 128)
        self.assertTrue(all(distance[c.center] >= 12.0 for c in candidates))

    def test_hard_paste_preserves_every_non_tumor_voxel_and_source(self):
        case, source = self.fixture()
        candidates, _ = self.pool(case, source, num_candidates=1)
        candidate = candidates[0]
        out_image, out_label = case.image.copy(), case.label.copy()
        occupied = case.label == 2
        scale, shift = paste_source(out_image, out_label, occupied, source, candidate,
                                    tumor_label=2, rng=np.random.default_rng(3),
                                    intensity_scale_range=(0.95, 1.05),
                                    intensity_shift_range=(-5.0, 5.0), blend_border=0)
        changed = np.zeros(case.shape, dtype=bool)
        changed[candidate.slices] = source.patch_mask
        np.testing.assert_array_equal(out_image[~changed], case.image[~changed])
        np.testing.assert_array_equal(out_label[~changed], case.label[~changed])
        self.assertTrue(np.all(out_label[changed] == 2))
        np.testing.assert_allclose(out_image[changed], (source.patch_image * scale + shift)[source.patch_mask])
        self.assertEqual(case.label[17, 5, 5], 2)

    def test_conflicting_distance_units_are_rejected(self):
        case, source = self.fixture()
        with self.assertRaisesRegex(ValueError, "one center separation unit"):
            self.pool(case, source, min_center_separation_mm=12.0)
