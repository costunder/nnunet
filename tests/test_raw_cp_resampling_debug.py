"""DEBUG only: tiny synthetic native nnU-Net raw-paste equivalence fixtures."""
from __future__ import annotations

import copy
import time
import unittest

import numpy as np
from skimage.transform import resize

from custom_trainers.onlinecp_raw_resampling import (
    RawResamplingError, apply_candidate, baseline_output,
    estimate_runtime_bytes, prepare_candidate, prepare_case,
    _axis_operator,
)
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import CTNormalization
from nnunetv2.preprocessing.resampling.default_resampling import (
    compute_new_shape, resample_data_or_seg_to_shape,
)


def debug_plan(*, spacing=(1.0, 0.7578125, 0.7578125), transpose=(0, 1, 2), separate=None):
    intensity = {"mean": 100.0, "std": 75.0, "percentile_00_5": -500.0,
                 "percentile_99_5": 600.0}
    return {
        "image_reader_writer": "NibabelIO", "transpose_forward": list(transpose),
        "transpose_backward": list(np.argsort(transpose)),
        "foreground_intensity_properties_per_channel": {"0": intensity},
        "configurations": {"3d_fullres": {
            # Preprocessing-only DEBUG fixture; no network is constructed.
            "architecture": {},
            "preprocessor_name": "DefaultPreprocessor",
            "normalization_schemes": ["CTNormalization"], "use_mask_for_norm": [False],
            "spacing": list(spacing),
            "resampling_fn_data": "resample_data_or_seg_to_shape",
            "resampling_fn_seg": "resample_data_or_seg_to_shape",
            "resampling_fn_data_kwargs": {"is_seg": False, "order": 3,
                                          "order_z": 0, "force_separate_z": separate},
            "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1,
                                         "order_z": 0, "force_separate_z": separate},
        }},
    }


def debug_native(raw_ct, raw_seg, raw_spacing, plans):
    transpose = plans["transpose_forward"]
    data = raw_ct.transpose(2, 1, 0).transpose(transpose)[None].copy()
    seg = raw_seg.transpose(2, 1, 0).transpose(transpose)[None].copy().astype(np.int16)
    before = list(data.shape[1:])
    data, seg, bbox = crop_to_nonzero(data, seg)
    properties = {"spacing": list(raw_spacing[::-1]), "shape_before_cropping": before,
                  "bbox_used_for_cropping": bbox,
                  "shape_after_cropping_and_before_resampling": list(data.shape[1:])}
    config = plans["configurations"]["3d_fullres"]
    spacing = np.asarray(raw_spacing[::-1])[transpose]
    new_shape = compute_new_shape(data.shape[1:], spacing, config["spacing"])
    normalizer = CTNormalization(use_mask_for_norm=False,
        intensityproperties=plans["foreground_intensity_properties_per_channel"]["0"])
    normalized = normalizer.run(data[0], seg[0])[None]
    image_result = resample_data_or_seg_to_shape(normalized, new_shape, spacing,
        config["spacing"], **config["resampling_fn_data_kwargs"])
    label_result = resample_data_or_seg_to_shape(seg, new_shape, spacing,
        config["spacing"], **config["resampling_fn_seg_kwargs"])
    return image_result, label_result, properties


class RawCPResamplingDebugTests(unittest.TestCase):
    debug_max_abs_error = 0.0
    debug_comparisons = 0

    @classmethod
    def tearDownClass(cls):
        print(f"[DEBUG native parity] comparisons={cls.debug_comparisons} "
              f"CT_max_abs_error={cls.debug_max_abs_error:.12g}; label/support require exact equality")

    def fixture(self, *, shape=(19, 21, 23), raw_spacing=(0.705078125, 0.705078125, 0.7),
                plans=None, cropped=False):
        rng = np.random.default_rng(192)
        ct = rng.uniform(-430, 490, size=shape).astype(np.float32)
        seg = np.ones(shape, dtype=np.int16)
        if cropped:
            ct[:2] = 0; ct[-2:] = 0
            ct[:, :2] = 0; ct[:, -2:] = 0
            ct[:, :, :2] = 0; ct[:, :, -2:] = 0
            seg[ct == 0] = 0
        plans = plans or debug_plan()
        expected_data, expected_seg, properties = debug_native(ct, seg, raw_spacing, plans)
        case = prepare_case(ct, seg, properties, plans, configuration_name="3d_fullres",
                            raw_spacing_xyz=raw_spacing)
        np.testing.assert_allclose(baseline_output(case), expected_data, atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(case["baseline_seg"], expected_seg)
        return ct, seg, raw_spacing, plans, case

    def compare_paste(self, fixture, *, center=(10, 10, 10), scale=1.0, shift=0.0,
                      mask=None, source=None, crop=None):
        ct, seg, spacing, plans, case = fixture
        if source is None:
            source = np.arange(125, dtype=np.float32).reshape(5, 5, 5) * 12 - 650
        if mask is None:
            mask = np.zeros(source.shape, dtype=bool)
            mask[1:4, 1:4, 1:4] = True
        anchor = np.array(source.shape) // 2
        candidate = prepare_candidate(case, source, mask, anchor, center)
        pasted_ct, pasted_seg = ct.copy(), seg.copy()
        origin = np.array(center) - anchor
        region = tuple(slice(int(a), int(a+b)) for a, b in zip(origin, source.shape))
        pasted_ct[region][mask] = (source * float(scale) + float(shift))[mask]
        pasted_seg[region][mask] = 2
        expected_data, expected_seg, _ = debug_native(pasted_ct, pasted_seg, spacing, plans)
        if crop is None:
            crop = [[0, int(n)] for n in expected_seg.shape[1:]]
        slices = tuple(slice(a, b) for a, b in crop)
        persisted = {key: value for key, value in case.items() if key != "preparation"}
        actual = apply_candidate(persisted, candidate, crop, scale=scale, shift_hu=shift)
        error = float(np.max(np.abs(actual["data"].astype(np.float64)
                                   - expected_data[(slice(None), *slices)].astype(np.float64))))
        type(self).debug_max_abs_error = max(type(self).debug_max_abs_error, error)
        type(self).debug_comparisons += 1
        np.testing.assert_allclose(actual["data"], expected_data[(slice(None), *slices)],
                                   atol=3e-6, rtol=3e-6)
        np.testing.assert_array_equal(actual["seg"], expected_seg[(slice(None), *slices)])
        expected_support = (expected_seg[0] == 2) & (case["baseline_seg"][0] != 2)
        np.testing.assert_array_equal(actual["pasted_support"], expected_support[slices])
        self.assertTrue(all(b > a for a, b in candidate["output_bbox"]))
        return candidate, actual, expected_support

    def test_debug_native_ct_label_and_jitter_equivalence(self):
        fixture = self.fixture()
        for scale, shift in ((1.0, 0.0), (1.3, 350.0), (0.4, -550.0)):
            with self.subTest(scale=scale, shift=shift):
                self.compare_paste(fixture, scale=scale, shift=shift)

    def test_debug_native_transpose_crop_and_training_crop(self):
        fixture = self.fixture(plans=debug_plan(transpose=(2, 0, 1)), cropped=True)
        self.compare_paste(fixture, center=(9, 11, 12), scale=1.1, shift=150.0,
                           crop=[[2, 9], [1, 12], [2, 10]])

    def test_debug_single_voxel_disappearance_depends_on_target(self):
        fixture = self.fixture(shape=(31, 31, 31))
        mask = np.zeros((5, 5, 5), dtype=bool); mask[2, 2, 2] = True
        at_source, _, support_source = self.compare_paste(fixture, center=(15, 15, 15), mask=mask)
        at_target, _, support_target = self.compare_paste(fixture, center=(15, 15, 3), mask=mask)
        self.assertEqual(int(support_source.sum()), 0)
        self.assertEqual(int(support_target.sum()), 1)
        self.assertEqual(at_source["audit"]["raw_pasted_voxels"], 1)
        self.assertEqual(at_target["audit"]["raw_pasted_voxels"], 1)

    def test_debug_separate_z_native_all_axes(self):
        for axis in range(3):
            reader_spacing = np.array([0.8, 0.9, 0.7]); reader_spacing[axis] = 4.0
            target_spacing = reader_spacing.copy(); target_spacing[axis] = 2.7
            target_spacing[(axis+1) % 3] *= 1.2
            with self.subTest(axis=axis):
                fixture = self.fixture(raw_spacing=tuple(reader_spacing[::-1]),
                    plans=debug_plan(spacing=target_spacing, separate=True))
                self.compare_paste(fixture, scale=1.4, shift=-230.0)

    def test_debug_cubic_global_tail_reaches_beyond_label_support(self):
        fixture = self.fixture(shape=(25, 25, 25))
        candidate, actual, _ = self.compare_paste(fixture, center=(12, 12, 12),
                                                scale=1.3, shift=500.0)
        delta = actual["data"][0] - baseline_output(fixture[4])[0]
        outside = np.ones(delta.shape, dtype=bool)
        outside[tuple(slice(a, b) for a, b in candidate["output_bbox"])] = False
        self.assertGreater(np.count_nonzero(delta[outside]), 0)

    def test_debug_full_recipient_label_stencil_includes_other_tumors(self):
        ct, seg, spacing, plans, _ = self.fixture()
        seg[7:9, 9:12, 9:12] = 2
        _, _, properties = debug_native(ct, seg, spacing, plans)
        case = prepare_case(ct, seg, properties, plans, configuration_name="3d_fullres",
                            raw_spacing_xyz=spacing)
        mask = np.zeros((5, 5, 5), dtype=bool); mask[2:4, 1:4, 1:4] = True
        self.compare_paste((ct, seg, spacing, plans, case), mask=mask)

    def test_debug_unsupported_kernel_raises_without_fallback(self):
        ct, seg, spacing, plans, _ = self.fixture()
        _, _, properties = debug_native(ct, seg, spacing, plans)
        bad = copy.deepcopy(plans)
        bad["configurations"]["3d_fullres"]["resampling_fn_data_kwargs"]["order"] = 1
        with self.assertRaises(RawResamplingError):
            prepare_case(ct, seg, properties, bad, configuration_name="3d_fullres",
                         raw_spacing_xyz=spacing)

    def test_debug_original_85_percent_liver_coverage_is_preserved(self):
        ct, seg, spacing, plans, _ = self.fixture()
        for xyz in ((9, 9, 9), (9, 9, 10), (9, 9, 11), (9, 10, 9)):
            seg[xyz] = 0
        _, _, properties = debug_native(ct, seg, spacing, plans)
        case = prepare_case(ct, seg, properties, plans, configuration_name="3d_fullres",
                            raw_spacing_xyz=spacing)
        candidate, _, _ = self.compare_paste((ct, seg, spacing, plans, case))
        self.assertAlmostEqual(candidate["audit"]["raw_liver_coverage"], 23 / 27)

    def test_debug_inactive_padding_outside_crop_does_not_drop_candidate(self):
        fixture = self.fixture(cropped=True)
        mask = np.zeros((5, 5, 5), dtype=bool); mask[2, 2, 2] = True
        candidate, _, _ = self.compare_paste(fixture, center=(3, 10, 10), mask=mask)
        self.assertLess(int(np.min(candidate["target_input_origin"])), 0)
        self.assertEqual(candidate["source_mask"].shape, (5, 5, 5))

    def test_debug_real_crop_boundary_is_explicitly_unproved_not_skipped(self):
        fixture = self.fixture(cropped=True)
        mask = np.zeros((5, 5, 5), dtype=bool); mask[2, 2, 2] = True
        with self.assertRaisesRegex(RawResamplingError, "does not prove raw CP impossible"):
            prepare_candidate(fixture[4], np.ones((5, 5, 5), np.float32), mask,
                              [2, 2, 2], [2, 10, 10])

    def test_debug_replacing_all_global_extrema_uses_actual_complement(self):
        ct, seg, spacing, plans, _ = self.fixture()
        ct.fill(150.0); ct[9, 9, 9] = -300.0; ct[10, 10, 10] = 450.0
        _, _, properties = debug_native(ct, seg, spacing, plans)
        case = prepare_case(ct, seg, properties, plans, configuration_name="3d_fullres",
                            raw_spacing_xyz=spacing)
        candidate, _, _ = self.compare_paste((ct, seg, spacing, plans, case),
                                             source=np.full((5, 5, 5), 100., np.float32))
        self.assertEqual(candidate["audit"]["extrema_complement_scans"], 1)

    def test_debug_exact_half_threshold_label_tie(self):
        fixture = self.fixture(shape=(20, 20, 20),
            plans=debug_plan(spacing=(1.4, 1.41015625, 1.41015625)))
        mask = np.zeros((5, 5, 5), dtype=bool); mask[1:4, 1:4, 2] = True
        _, _, support = self.compare_paste(fixture, mask=mask)
        self.assertEqual(int(support.sum()), 1)

    def test_debug_vectorized_axis_basis_against_native_1d_oracle(self):
        for order in (1, 3):
            for length, target in ((31, 22), (32, 17), (63, 85)):
                start = time.perf_counter()
                actual = _axis_operator(length, target, order=order)
                vector_seconds = time.perf_counter() - start
                start = time.perf_counter()
                expected = np.column_stack([
                    resize(np.eye(length, dtype=np.float64)[:, column], (target,), order=order,
                           mode="edge", clip=False, anti_aliasing=False, preserve_range=True)
                    for column in range(length)])
                oracle_seconds = time.perf_counter() - start
                if order == 1:
                    np.testing.assert_array_equal(actual, expected)
                else:
                    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-15)
                print(f"[DEBUG axis basis] order={order} {length}->{target} "
                      f"max_abs={np.max(np.abs(actual-expected)):.12g} "
                      f"vector_seconds={vector_seconds:.6g} oracle_seconds={oracle_seconds:.6g}")

    def test_debug_inputs_unchanged_and_estimates_explicit(self):
        fixture = self.fixture()
        before_ct, before_seg = fixture[0].copy(), fixture[1].copy()
        self.compare_paste(fixture)
        np.testing.assert_array_equal(before_ct, fixture[0])
        np.testing.assert_array_equal(before_seg, fixture[1])
        estimate = estimate_runtime_bytes((31, 31, 31), (22, 29, 29), (5, 5, 5), (10, 10, 10))
        self.assertTrue(estimate)


if __name__ == "__main__":
    unittest.main()
