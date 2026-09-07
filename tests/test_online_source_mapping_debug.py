"""DEBUG synthetic arrays with the installed, real nnU-Net segmentation resampler.

These test source identity and spatial mapping, not medical model performance.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest

import numpy as np
from scipy import ndimage as ndi
from nnunetv2.preprocessing.resampling.default_resampling import (
    compute_new_shape, resample_data_or_seg_to_shape,
)

from tools.online_cp_benchmark import (
    SOURCE_MAPPING_FORMAT, OnlineBenchmarkError, SourceMappingError,
    _preprocessed_source, _source_from_component, _require_source_mapping_policy,
)


class OnlineSourceMappingDebugTests(unittest.TestCase):
    def fixture(self, mask=None, *, spacing=(1, 1, 1), target=None, transpose=(0, 1, 2),
                crop=None, order=1, global_raw_mask=None):
        if mask is None:
            mask = np.zeros((9, 10, 11), dtype=bool)
            mask[2:5, 3:6, 4:7] = True
        spacing = np.asarray(spacing, dtype=np.float32)
        raw_data = np.arange(mask.size, dtype=np.float32).reshape(mask.shape) / np.float32(997)
        source = _source_from_component(SimpleNamespace(image=raw_data), mask.astype(np.int16), 1, 2)
        kwargs = {"is_seg": True, "order": order, "order_z": 0, "force_separate_z": None}
        transposed = mask.transpose(2, 1, 0).transpose(transpose)
        before_shape = transposed.shape
        if crop is None:
            crop = [[0, int(value)] for value in before_shape]
        slices = tuple(slice(*bounds) for bounds in crop)
        original_spacing = spacing[::-1][list(transpose)]
        target = original_spacing.copy() if target is None else np.asarray(target, dtype=float)
        cropped_shape = transposed[slices].shape
        pre_shape = compute_new_shape(cropped_shape, original_spacing, target)
        labels = mask if global_raw_mask is None else global_raw_mask
        cropped_labels = labels.transpose(2, 1, 0).transpose(transpose)[slices].astype(np.int16)[None] * 2
        pre_seg = resample_data_or_seg_to_shape(cropped_labels, pre_shape, original_spacing, target, **kwargs)
        pre_data = (np.arange(int(np.prod(pre_shape)), dtype=np.float32).reshape((1, *pre_shape))
                    / np.float32(997))
        plans = {"transpose_forward": list(transpose), "image_reader_writer": "NibabelIO",
                 "configurations": {"3d_fullres": {"preprocessor_name": "DefaultPreprocessor",
                     # The real ConfigurationManager needs the modern schema;
                     # no architecture/model is instantiated in these tests.
                     "architecture": {},
                     "spacing": target.tolist(), "resampling_fn_seg": "resample_data_or_seg_to_shape",
                     "resampling_fn_seg_kwargs": kwargs}}}
        properties = {"spacing": spacing[::-1].tolist(), "shape_before_cropping": list(before_shape),
                      "shape_after_cropping_and_before_resampling": list(cropped_shape),
                      "bbox_used_for_cropping": crop}
        return dict(data=pre_data, seg=pre_seg, properties=properties, plans=plans, source=source,
                    tumor_label=2, configuration_name="3d_fullres", raw_spacing=spacing,
                    raw_spatial_unit="mm")

    def run_mapping(self, args):
        report = {}
        result = _preprocessed_source(**args, diagnostics=report)
        json.dumps(report, allow_nan=False)
        return result, report

    def assert_failure(self, args, reason):
        with self.assertRaises(SourceMappingError) as raised:
            self.run_mapping(args)
        self.assertEqual(raised.exception.reason, reason)
        json.dumps(raised.exception.diagnostics, allow_nan=False)

    def test_identity_grid_ct_mask_same_slice_and_no_float16_roundtrip(self):
        args = self.fixture()
        data_before, seg_before = args["data"].copy(), args["seg"].copy()
        (data, mask, anchor), report = self.run_mapping(args)
        slices = tuple(slice(lo, hi) for lo, hi in zip(report["patch_lower"], report["patch_upper"]))
        np.testing.assert_array_equal(data, args["data"][(slice(None), *slices)])
        np.testing.assert_array_equal(mask, args["seg"][(0, *slices)] == 2)
        self.assertEqual(data.dtype, np.dtype(np.float32))
        self.assertTrue(np.any(data.astype(np.float16).astype(np.float32) != data))
        np.testing.assert_array_equal(args["data"], data_before)
        np.testing.assert_array_equal(args["seg"], seg_before)
        self.assertFalse(report["other_lesions_substituted"])
        self.assertTrue(np.all(anchor >= 0))

    def test_real_resampler_anisotropy_transpose_crop_and_target_shape(self):
        args = self.fixture(spacing=(0.75, 0.8, 4.0), target=(1.2, 0.7, 2.0),
                            transpose=(1, 2, 0), crop=[[1, 9], [1, 8], [1, 10]])
        (_, mask, _), report = self.run_mapping(args)
        self.assertGreater(int(mask.sum()), 0)
        self.assertEqual(report["original_spacing"], args["raw_spacing"][[1, 0, 2]].astype(float).tolist())
        self.assertEqual(report["resampling_fn_seg"], "resample_data_or_seg_to_shape")
        self.assertEqual(report["resampling_fn_seg_kwargs"]["force_separate_z"], None)
        self.assertEqual(report["preprocessed_shape"], list(args["seg"].shape[1:]))

    def test_nonconvex_source_never_substitutes_lesion_at_its_centroid(self):
        mask = np.zeros((9, 9, 9), dtype=bool)
        mask[2:7, 2:7, 4] = True
        mask[3:6, 3:6, 4] = False
        all_tumors = mask.copy()
        all_tumors[4, 4, 4] = True
        args = self.fixture(mask, global_raw_mask=all_tumors)
        (_, actual, _), report = self.run_mapping(args)
        self.assertEqual(int(actual.sum()), int(mask.sum()))
        slices = tuple(slice(lo, hi) for lo, hi in zip(report["patch_lower"], report["patch_upper"]))
        np.testing.assert_array_equal(actual, mask.transpose(2, 1, 0)[slices])

    def test_global_component_merger_does_not_copy_other_source(self):
        args = self.fixture()
        original_positive = int(np.count_nonzero(args["seg"] == 2))
        args["seg"][0, 4:7, 3:6, 5:8] = 2
        self.assertGreater(int(np.count_nonzero(args["seg"] == 2)), original_positive)
        (_, actual, _), report = self.run_mapping(args)
        self.assertEqual(int(actual.sum()), original_positive)
        self.assertFalse(report["other_lesions_substituted"])

    def test_transformed_source_can_have_multiple_components_without_relabeling(self):
        mask = np.zeros((9, 9, 9), dtype=bool)
        mask[0:3, 0:3, 0:3] = True
        mask[6:9, 0:3, 0:3] = True
        mask[3:6, 0, 0] = True
        self.assertEqual(ndi.label(mask)[1], 1)
        # The real configured resampler removes this thin connector while both
        # lobes survive. Neither surviving fragment may be dropped/replaced.
        args = self.fixture(mask, target=(3, 3, 3), order=0)
        (_, actual, _), _ = self.run_mapping(args)
        self.assertEqual(ndi.label(actual)[1], 2)
        self.assertEqual(int(actual.sum()), int(np.count_nonzero(args["seg"] == 2)))

    def test_disappeared_raw_source_does_not_select_surviving_neighbor(self):
        mask = np.zeros((9, 9, 9), dtype=bool)
        mask[0, 0, 0] = True
        global_mask = mask.copy()
        global_mask[4:8, 4:8, 4:8] = True
        args = self.fixture(mask, target=(3, 3, 3), order=0, global_raw_mask=global_mask)
        self.assertTrue(np.any(args["seg"] == 2))
        self.assert_failure(args, "selected_source_disappeared_after_resampling")

    def test_mapped_source_outside_actual_tumor_is_explicit_conflict(self):
        args = self.fixture()
        index = tuple(np.argwhere(args["seg"] == 2)[0])
        args["seg"][index] = 1
        self.assert_failure(args, "mapped_source_conflicts_with_preprocessed_tumor")

    def test_wrong_reader_and_unknown_spatial_units_fail_without_guessing(self):
        args = self.fixture()
        args["plans"]["image_reader_writer"] = "NibabelIOWithReorient"
        self.assert_failure(args, "unsupported_raw_axis_contract")
        args = self.fixture()
        args["raw_spatial_unit"] = "unknown"
        self.assert_failure(args, "unverified_raw_spacing_unit")

    def test_source_and_reader_spacing_mismatch_fails(self):
        args = self.fixture()
        args["properties"]["spacing"][0] = 2
        self.assert_failure(args, "raw_reader_spacing_mismatch")

    def test_bounded_header_spacing_roundoff_does_not_require_float32_equality(self):
        args = self.fixture(spacing=(0.64453125, 0.64453125, 0.70000005))
        spacing = np.asarray(args["properties"]["spacing"], dtype=np.float32)
        args["properties"]["spacing"] = np.nextafter(spacing, np.float32(np.inf)).astype(float).tolist()
        (_, mask, _), report = self.run_mapping(args)
        self.assertGreater(report["reader_spacing_max_corner_mm"], 0)
        self.assertLessEqual(report["reader_spacing_max_corner_mm"], report["spacing_extent_limits"]["mm"])
        self.assertEqual(int(mask.sum()), int(args["source"].voxel_count))

    def test_spacing_extent_limits_check_both_mm_and_voxels(self):
        args = self.fixture()
        args["properties"]["spacing"][0] += 2e-5
        self.assert_failure(args, "raw_reader_spacing_mismatch")
        args = self.fixture(spacing=(0.001, 0.001, 0.001))
        args["properties"]["spacing"][0] += 2e-8
        with self.assertRaises(SourceMappingError) as raised:
            self.run_mapping(args)
        self.assertEqual(raised.exception.reason, "raw_reader_spacing_mismatch")
        self.assertLess(raised.exception.diagnostics["reader_spacing_max_corner_mm"], 1e-4)
        self.assertGreater(raised.exception.diagnostics["reader_spacing_max_corner_raw_voxels"], 1e-4)

    def test_invalid_anchor_is_rejected_not_clipped_to_a_different_voxel(self):
        args = self.fixture()
        args["source"] = SimpleNamespace(**{**vars(args["source"]),
                                           "anchor_center": np.array([-100, -100, -100])})
        self.assert_failure(args, "mapped_anchor_outside_source_patch")

    def test_bank_index_and_config_both_require_the_exact_mapping_policy(self):
        valid = {"source_mapping_format": SOURCE_MAPPING_FORMAT}
        _require_source_mapping_policy(valid, valid)
        for stale in ({}, {"source_mapping_format": "legacy"}):
            for index, config in ((stale, valid), (valid, stale)):
                with self.assertRaisesRegex(OnlineBenchmarkError, "source_mapping_format"):
                    _require_source_mapping_policy(index, config)

    def test_invalid_target_spacing_and_resampled_shape_fail(self):
        args = self.fixture()
        args["plans"]["configurations"]["3d_fullres"]["spacing"] = [0, 1, 1]
        self.assert_failure(args, "invalid_target_spacing")
        args = self.fixture()
        args["plans"]["configurations"]["3d_fullres"]["spacing"] = [2, 1, 1]
        self.assert_failure(args, "resampled_shape_mismatch")

    def test_missing_source_crop_support_not_silently_discarded(self):
        args = self.fixture(crop=[[5, 11], [0, 10], [0, 9]])
        self.assert_failure(args, "selected_source_lost_during_cropping")

    def test_invalid_transpose_and_shapes_rejected(self):
        args = self.fixture()
        args["plans"]["transpose_forward"] = [0, 0, 2]
        self.assert_failure(args, "invalid_transpose_forward")
        args = self.fixture()
        args["properties"]["shape_before_cropping"][0] -= 1
        self.assert_failure(args, "raw_shape_before_cropping_mismatch")
        args = self.fixture()
        args["data"] = args["data"][:, :-1]
        self.assert_failure(args, "preprocessed_grid_mismatch")

    def test_custom_preprocessor_is_not_silently_assumed_default(self):
        args = self.fixture()
        args["plans"]["configurations"]["3d_fullres"]["preprocessor_name"] = "DifferentPreprocessor"
        self.assert_failure(args, "unsupported_preprocessor_contract")

    def test_configuration_inheritance_uses_resolved_exact_resampler(self):
        args = self.fixture()
        config = copy.deepcopy(args["plans"]["configurations"]["3d_fullres"])
        args["plans"]["configurations"] = {"base": config, "3d_fullres": {"inherits_from": "base"}}
        (_, mask, _), _ = self.run_mapping(args)
        self.assertEqual(int(mask.sum()), int(args["source"].voxel_count))


if __name__ == "__main__":
    unittest.main()
