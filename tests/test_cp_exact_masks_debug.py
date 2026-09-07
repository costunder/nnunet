"""DEBUG exact mask arithmetic; synthetic fixtures are not medical results."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
from scipy import ndimage as ndi

from hiercp import common


class ExactPlacementMaskDebugTests(unittest.TestCase):
    def test_binary_integer_context_masks_match_boolean_masks(self):
        mask = np.zeros((7, 9, 11), dtype=bool)
        mask[2:5, 3:6, 4:7] = True
        for width in (0, 1, 3):
            expected = common.context_ring_mask(mask, width)
            for dtype in (np.uint8, np.int16):
                np.testing.assert_array_equal(
                    common.context_ring_mask(mask.astype(dtype), width), expected)
        # Compare binary-mask handling for the unrestricted background ring.
        image = np.arange(mask.size, dtype=np.float32).reshape(mask.shape)
        organ = np.ones(mask.shape, dtype=bool)
        self.assertEqual(
            common.context_stats_for_local_mask(image, organ, mask, ring_width=0),
            common.context_stats_for_local_mask(image, organ, mask.astype(np.uint8), ring_width=0))

    def test_sparse_context_fallback_matches_boolean_and_integer_masks(self):
        mask = np.ones((2, 2, 2), dtype=bool)
        mask[0, 0, 0] = False
        organ = np.ones(mask.shape, dtype=bool)
        image = np.arange(mask.size, dtype=np.float32).reshape(mask.shape)
        image[0, 0, 0] = 123.5
        # One context voxel forces the <8 fallback. Integer advanced indexing
        # would incorrectly gather image planes instead of this single voxel.
        ring = common.context_ring_mask(mask, width=0) & organ
        self.assertEqual(int(np.count_nonzero(ring)), 1)
        for dtype in (bool, np.uint8, np.int16):
            with self.subTest(dtype=dtype):
                self.assertEqual(
                    common.context_stats_for_local_mask(
                        image, organ, mask.astype(dtype), ring_width=0),
                    (123.5, 0.0))

    def test_physical_distance_is_explicit_not_legacy_uint8_edt(self):
        mask = np.zeros((9, 9, 9), dtype=np.uint8)
        mask[4, 4, 4] = 1
        np.testing.assert_array_equal(np.unique(~mask), [254, 255])
        self.assertNotEqual(float(ndi.distance_transform_edt(~mask)[4, 4, 4]), 0.)
        actual = common.distance_to_mask_mm(mask, (0.5, 1., 3.))
        self.assertEqual(float(actual[4, 4, 4]), 0.)
        self.assertEqual(float(actual[5, 4, 4]), 0.5)
        self.assertEqual(float(actual[4, 4, 5]), 3.)

    def test_exhaustive_predicates_match_independent_full_slice_reference(self):
        rng = np.random.default_rng(228)
        shape = (12, 13, 14)
        for patch_shape in ((3, 4, 5), (4, 3, 6), (2, 2, 2)):
            footprint = rng.random(patch_shape) > .6
            footprint[0, 0, 0] = True
            count = int(footprint.sum())
            source = SimpleNamespace(patch_mask=footprint, voxel_count=count)
            for clearance in (0, 1):
                placement = rng.random(shape) > .15
                occupied = rng.random(shape) < .003
                forbidden = (ndi.binary_dilation(occupied, iterations=clearance)
                             if clearance else occupied)
                distances = common.distance_to_mask_mm(occupied, (.7, 1.2, 2.1))
                expected = []
                expected_rejections = dict(center_separation=0, forbidden_overlap=0,
                                           liver_coverage=0)
                for center in np.ndindex(shape):
                    slc = common.slices_for_center(center, patch_shape, shape)
                    if slc is None or not placement[center]:
                        continue
                    if distances[center] < 1.5:
                        expected_rejections['center_separation'] += 1
                    elif np.any(footprint & forbidden[slc]):
                        expected_rejections['forbidden_overlap'] += 1
                    else:
                        coverage = float(np.count_nonzero(footprint & placement[slc]) / count)
                        if coverage < .85:
                            expected_rejections['liver_coverage'] += 1
                        else:
                            expected.append((center, coverage, float(distances[center])))
                for memory in (1024, 64 * 1024 * 1024):
                    actual, diagnostics = [], {}
                    done = common._extend_candidates_exhaustively(
                        SimpleNamespace(shape=shape), source,
                        np.asfortranarray(placement), np.asfortranarray(forbidden), distances,
                        min_liver_coverage=.85, min_center_separation_mm=1.5,
                        tested_flat=set(), excluded=set(), target=np.prod(shape) + 1,
                        accepted=actual,
                        append_candidate=lambda c, cov, d: actual.append((c, cov, d)),
                        diagnostics=diagnostics, working_memory_bytes=memory)
                    self.assertTrue(done)
                    self.assertEqual(actual, expected)
                    for name, value in expected_rejections.items():
                        self.assertEqual(diagnostics['exhaustive_rejections'][name], value)
                    self.assertLessEqual(diagnostics['exhaustive_max_matrix_elements'] * 16, memory)

    def test_proven_overlap_retires_work_without_skipping_legal_centers(self):
        flat = np.arange(32, dtype=np.int64) * 5
        offsets = np.arange(8192, dtype=np.int64)
        forbidden = np.ones(9000, dtype=bool)
        placement = np.ones_like(forbidden)
        diagnostic = dict(exhaustive_naive_mask_elements=0,
                          exhaustive_forbidden_mask_elements=0,
                          exhaustive_coverage_mask_elements=0,
                          exhaustive_retired_overlap_centers=0,
                          exhaustive_max_matrix_elements=0)
        blocked, coverage = common._exact_footprint_counts(
            flat, offsets, forbidden, placement,
            matrix_elements=4096, diagnostics=diagnostic)
        self.assertTrue(blocked.all())
        self.assertTrue((coverage == 0).all())  # Not evaluated for rejected centers.
        self.assertEqual(diagnostic['exhaustive_retired_overlap_centers'], len(flat))
        self.assertEqual(diagnostic['exhaustive_coverage_mask_elements'], 0)
        self.assertLess(diagnostic['exhaustive_forbidden_mask_elements'],
                        diagnostic['exhaustive_naive_mask_elements'] // 10)


if __name__ == '__main__':
    unittest.main()
