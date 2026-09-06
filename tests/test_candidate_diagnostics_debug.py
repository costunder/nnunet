"""DEBUG synthetic candidate search/diagnostics, not medical training or evaluation.

The frozen legacy reference proves successful-cache proposal/RNG compatibility.
Small masks test exhaustive-domain coverage and tiled numerical equivalence;
they are not production resource benchmarks or reduced production defaults.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from scipy import ndimage as ndi

from hiercp import common, curriculum


def fixture(size=13, tumor_width=3):
    shape = (size,) * 3
    image = np.arange(size ** 3, dtype=np.float32).reshape(shape) / 13.
    label = np.ones(shape, dtype=np.int16)
    middle = size // 2
    half = tumor_width // 2
    label[(slice(middle - half, middle + half + 1),) * 3] = 2
    case = SimpleNamespace(image=image, label=label, shape=shape,
                           spacing=np.ones(3), paths=SimpleNamespace(case_id="debug_candidate"))
    source, _, _ = common.choose_source_tumor(
        image, label, tumor_label=2, rng=np.random.default_rng(9), selection="largest", pad=1)
    options = dict(placement_mask=label == 1, full_organ_mask=label > 0,
                   occupied_mask=label == 2, organ_distance=np.full(shape, 8., np.float32),
                   num_candidates=128, max_draws=4000, min_liver_coverage=.85,
                   occupied_clearance_vox=0, min_center_separation_mm=0.)
    return case, source, options


def legacy_centers(mask, patch_shape, rng, requested, max_draws):
    """Frozen original proposal/RNG operations, including historical bounds."""
    if requested <= 0 or not np.any(mask):
        return
    patch, shape = np.asarray(patch_shape, np.int64), np.asarray(mask.shape, np.int64)
    half = patch // 2
    objects = ndi.find_objects(mask.astype(np.uint8, copy=False))
    box = objects[0]
    low = np.maximum(half, np.asarray([item.start for item in box], np.int64))
    high = np.minimum(shape - (patch - half), np.asarray([item.stop for item in box], np.int64))
    if np.any(high <= low):
        return
    seen, yielded, draws = set(), 0, 0
    while yielded < requested and draws < max_draws:
        batch = min(2048, max_draws - draws)
        random_values = rng.random((batch, 3), dtype=np.float64)
        coords = np.floor(low + random_values * (high - low)).astype(np.int64)
        draws += batch
        for row in coords:
            center = tuple(int(value) for value in row)
            if center in seen:
                continue
            seen.add(center)
            if not mask[center]:
                continue
            yielded += 1
            yield center
            if yielded >= requested:
                break


def legacy_pool(case, source, rng, **options):
    """Frozen original anatomy tests, metadata arithmetic and selection order."""
    placement = options["placement_mask"]
    occupied = options["occupied_mask"]
    if options["occupied_clearance_vox"] > 0:
        forbidden = ndi.binary_dilation(occupied, structure=ndi.generate_binary_structure(3, 1),
                                       iterations=options["occupied_clearance_vox"])
    else:
        forbidden = occupied.astype(bool, copy=False)
    distance = common.distance_to_mask_mm(occupied, case.spacing)
    target = options["num_candidates"]
    raw = max(target * options.get("candidate_oversample_factor", 20), target)
    ring = common.context_ring_mask(source.patch_mask, width=3)
    accepted = []
    for center in legacy_centers(placement, source.patch_mask.shape, rng, raw, options["max_draws"]):
        slc = common.slices_for_center(center, source.patch_mask.shape, case.shape)
        if slc is None or np.any(source.patch_mask & forbidden[slc]):
            continue
        coverage = float(np.sum(source.patch_mask & placement[slc]) / max(1, source.voxel_count))
        if coverage < options["min_liver_coverage"]:
            continue
        separation = float(distance[center])
        if separation < options["min_center_separation_mm"]:
            continue
        image, organ = case.image[slc], options["full_organ_mask"][slc]
        values = image[ring & organ]
        if values.size < 8:
            values = image[organ & ~source.patch_mask]
        if values.size == 0:
            values = image.reshape(-1)
        accepted.append(common.CandidateInfo(
            center=center, slices=slc, liver_coverage=coverage,
            border_distance_mm=float(options["organ_distance"][center]),
            occupied_distance_mm=separation, context_mean_hu=float(np.mean(values)),
            context_std_hu=float(np.std(values))))
        if len(accepted) >= target:
            break
    return accepted, distance


class CandidateDiagnosticsDebugTests(unittest.TestCase):
    def test_debug_success_matches_frozen_legacy_output_and_rng_exactly(self):
        case, source, options = fixture()
        options["num_candidates"] = 8
        old_rng, new_rng = np.random.default_rng(4), np.random.default_rng(4)
        expected, old_distance = legacy_pool(case, source, old_rng, **options)
        diagnostics = {}
        actual, distance = common.build_candidate_pool(case, source, rng=new_rng,
                                                       diagnostics=diagnostics, **options)
        self.assertEqual(len(expected), 8)
        self.assertEqual(actual, expected)
        np.testing.assert_array_equal(distance, old_distance)
        self.assertEqual(new_rng.bit_generator.state, old_rng.bit_generator.state)
        self.assertFalse(diagnostics["exhaustive_used"])

    def test_debug_existing_short_pool_with_seven_negatives_is_unchanged(self):
        case, source, options = fixture(size=7, tumor_width=1)
        old_rng, new_rng = np.random.default_rng(2), np.random.default_rng(2)
        expected, _ = legacy_pool(case, source, old_rng, **options)
        diagnostics = {}
        actual, _ = common.build_candidate_pool(case, source, rng=new_rng,
            required_candidates=7, diagnostics=diagnostics, **options)
        self.assertGreaterEqual(len(expected), 7)
        self.assertLess(len(expected), 128)
        self.assertEqual(actual, expected)
        self.assertEqual(new_rng.bit_generator.state, old_rng.bit_generator.state)
        self.assertFalse(diagnostics["exhaustive_used"])

    def test_debug_raw_oversample_shortfall_extends_same_source_to_full_pool(self):
        case, source, options = fixture()
        options["candidate_oversample_factor"] = 1
        before = source.patch_mask.copy()
        old_rng, new_rng = np.random.default_rng(7), np.random.default_rng(7)
        expected, _ = legacy_pool(case, source, old_rng, **options)
        diagnostics = {}
        actual, _ = common.build_candidate_pool(case, source, rng=new_rng,
            diagnostics=diagnostics, **options)
        self.assertLess(len(expected), 128)
        self.assertEqual(actual[:len(expected)], expected)
        self.assertEqual(len(actual), 128)
        self.assertEqual(len({candidate.center for candidate in actual}), 128)
        self.assertTrue(diagnostics["exhaustive_used"])
        self.assertFalse(diagnostics["fullsearch_exhausted"])
        self.assertEqual(new_rng.bit_generator.state, old_rng.bit_generator.state)
        np.testing.assert_array_equal(source.patch_mask, before)
        for candidate in actual:
            self.assertFalse(np.any(source.patch_mask & options["occupied_mask"][candidate.slices]))
            self.assertGreaterEqual(candidate.liver_coverage, options["min_liver_coverage"])

    def test_debug_exhaustion_covers_every_legal_center_including_upper_boundary(self):
        case, source, options = fixture(size=7, tumor_width=1)
        diagnostics = {}
        actual, _ = common.build_candidate_pool(case, source, rng=np.random.default_rng(2),
            diagnostics=diagnostics, **options)
        expected = set()
        for center in np.ndindex(case.shape):
            slc = common.slices_for_center(center, source.patch_mask.shape, case.shape)
            if slc is not None and options["placement_mask"][center]:
                expected.add(center)
        self.assertEqual({candidate.center for candidate in actual}, expected)
        self.assertEqual(len(actual), 124)
        self.assertTrue(diagnostics["fullsearch_exhausted"])
        self.assertFalse(diagnostics["required_candidates_met"])
        self.assertIn((5, 5, 5), expected)

    def test_debug_tile_size_changes_neither_output_nor_rng(self):
        case, source, options = fixture()
        options.update(candidate_oversample_factor=1)
        outputs, states = [], []
        for scratch in (1024, 4096, 64 * 1024 * 1024):
            rng, diagnostics = np.random.default_rng(7), {}
            pool, _ = common.build_candidate_pool(case, source, rng=rng, diagnostics=diagnostics,
                exhaustive_working_memory_bytes=scratch, **options)
            outputs.append(pool)
            states.append(copy.deepcopy(rng.bit_generator.state))
            self.assertLessEqual(diagnostics["exhaustive_max_matrix_elements"] * 16, scratch)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[0], states[2])

    def test_debug_geometry_retry_excludes_only_explicit_centers(self):
        case, source, options = fixture()
        options.update(candidate_oversample_factor=1)
        original, _ = common.build_candidate_pool(case, source, rng=np.random.default_rng(7),
                                                  required_candidates=7, **options)
        excluded = {candidate.center for candidate in original[:3]}
        diagnostics = {}
        retried, _ = common.build_candidate_pool(case, source, rng=np.random.default_rng(7),
            required_candidates=7, force_exhaustive=True, excluded_centers=excluded,
            diagnostics=diagnostics, **options)
        self.assertEqual(len(retried), 128)
        self.assertFalse(excluded & {candidate.center for candidate in retried})
        self.assertTrue(diagnostics["exhaustive_used"])
        self.assertEqual(diagnostics["excluded_center_count"], 3)

    def test_debug_sampler_reports_generated_and_examined_coordinates_separately(self):
        mask = np.ones((7, 7, 7), dtype=bool)
        diagnostics = {}
        points = list(common.sample_candidate_centers(mask, (3, 3, 3),
            rng=np.random.default_rng(3), requested=999, max_draws=33, diagnostics=diagnostics))
        self.assertEqual(diagnostics["random_draws"], 33)
        self.assertEqual(diagnostics["random_centers_yielded"], len(points))
        self.assertEqual(diagnostics["random_coordinates_examined"],
            diagnostics["random_duplicate_centers"] + diagnostics["random_center_mask_rejections"]
            + diagnostics["random_centers_yielded"])

    def test_debug_no_placement_mask_reports_exhaustion_without_fake_candidates(self):
        case, source, options = fixture(size=7, tumor_width=1)
        options["placement_mask"] = np.zeros(case.shape, bool)
        diagnostics = {}
        pool, _ = common.build_candidate_pool(case, source, rng=np.random.default_rng(4),
                                              diagnostics=diagnostics, **options)
        self.assertEqual(pool, [])
        self.assertEqual(diagnostics["random_draws"], 0)
        self.assertTrue(diagnostics["fullsearch_exhausted"])

    def test_debug_error_carries_immutable_json_safe_evidence(self):
        source = {"counts": {"accepted": 3}}
        error = common.CandidatePreparationError("insufficient_valid_candidate_pool", source)
        source["counts"]["accepted"] = 9
        self.assertEqual(error.diagnostics["counts"]["accepted"], 3)
        self.assertIn("insufficient_valid_candidate_pool", str(error))
        json.dumps(error.diagnostics, allow_nan=False)
        with self.assertRaises(ValueError):
            common.CandidatePreparationError("invalid", {"value": float("nan")})

    def test_debug_curriculum_insufficient_pool_raises_structured_reason(self):
        with self.assertRaises(common.CandidatePreparationError) as raised:
            curriculum.build_training_specs(None, None, [], None, None,
                total_candidates=8, easy_fraction=.34, inter_fraction=.33, intra_fraction=.33,
                tumor_label=2, config=None, rng=np.random.default_rng(1))
        self.assertEqual(raised.exception.reason, "insufficient_valid_candidate_pool")
        self.assertEqual(raised.exception.diagnostics["required_negative_candidates"], 7)

    def test_debug_curriculum_duplicate_centers_have_distinct_failure_reason(self):
        case, source, options = fixture(size=7, tumor_width=1)
        candidates, _ = common.build_candidate_pool(case, source, rng=np.random.default_rng(2),
                                                   required_candidates=7, **options)
        regions = SimpleNamespace(region_at=lambda center: 0, region_features=np.ones((1, 2)),
                                  organ_depth=np.ones(case.shape))
        bank = SimpleNamespace(assign=lambda *args, **kwargs: (np.zeros((1, 1), dtype=int), np.ones((1, 1))))
        config = SimpleNamespace(prototype_top_m=1, prototype_temperature=.5)
        with mock.patch.object(curriculum, "_source_context", return_value=(1., 1.)):
            with self.assertRaises(common.CandidatePreparationError) as raised:
                curriculum.build_training_specs(case, source, [candidates[0]] * 7, regions, bank,
                    total_candidates=8, easy_fraction=.34, inter_fraction=.33, intra_fraction=.33,
                    tumor_label=2, config=config, rng=np.random.default_rng(1))
        self.assertEqual(raised.exception.reason, "insufficient_distinct_curriculum_candidates")
        self.assertEqual(raised.exception.diagnostics["distinct_centers"], 1)


if __name__ == "__main__":
    unittest.main()
