"""DEBUG header-only grid fixtures, never full-volume data or medical training."""
from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

import nibabel as nib
import numpy as np

from hiercp.nifti_geometry import validate_donor_geometry, _float32_ulp_distance


def header_fixture(affine=None, *, shape=(512, 512, 630), unit="mm", nifti2=False,
                   sform_code=1, qform=None):
    header = nib.Nifti2Header() if nifti2 else nib.Nifti1Header()
    header.set_data_shape(shape)
    header.set_xyzt_units(unit)
    affine = np.eye(4) if affine is None else np.asarray(affine, dtype=np.float64)
    header.set_qform(affine if qform is None else qform, code=2)
    header.set_sform(affine, code=sform_code)
    return SimpleNamespace(shape=shape, header=header, affine=header.get_best_affine())


def user_headers(**kwargs):
    image = np.eye(4)
    label = np.eye(4)
    image[:3] = np.asarray([[0.64453125, 0, 0, -163.35547],
                           [0, 0.64453125, 0, -295.35547],
                           [0, 0, 0.70000005, 54.399963]], dtype=np.float32)
    label[:3] = np.asarray([[0.6445313, 0, 0, -163.3555],
                           [0, 0.6445313, 0, -295.3555],
                           [0, 0, 0.7, 54.399963]], dtype=np.float32)
    return header_fixture(image, **kwargs), header_fixture(label, **kwargs)


class DonorGeometryDebugTests(unittest.TestCase):
    def check(self, image, label):
        return validate_donor_geometry(image, label, case_id="DEBUG_liver_85_headers")

    def reject(self, image, label, reason="affine mismatch"):
        with self.assertRaisesRegex(ValueError, reason):
            self.check(image, label)

    def test_reported_user_header_roundoff_full_extent_without_full_volume(self):
        image, label = user_headers()
        originals = [nii.header.binaryblock for nii in (image, label)]
        report = self.check(image, label)
        self.assertEqual(report["accepted_as"], "roundoff")
        self.assertFalse(report["legacy_strict_pass"])
        self.assertAlmostEqual(report["max_corner_displacement_mm"], 5.721992347138872e-5, places=14)
        self.assertAlmostEqual(report["max_corner_displacement_image_voxels"], 8.582337313998776e-5, places=14)
        self.assertEqual(report["max_affine_float32_ulps"], 2)
        self.assertEqual(report["max_pixdim_float32_ulps"], 1)
        self.assertEqual(report["image_forms"]["selected_form"], "sform")
        self.assertFalse(report["data_resampled"])
        self.assertEqual(originals, [nii.header.binaryblock for nii in (image, label)])
        json.dumps(report, allow_nan=False)

    def test_exact_grid_known_and_unknown_units_remains_strict(self):
        for unit in ("mm", "meter", "micron", "unknown"):
            with self.subTest(unit=unit):
                report = self.check(header_fixture(unit=unit), header_fixture(unit=unit))
                self.assertEqual(report["accepted_as"], "strict")
                self.assertEqual(report["max_corner_displacement_image_voxels"], 0)
                self.assertEqual(report["max_corner_displacement_mm"], None if unit == "unknown" else 0)
                json.dumps(report, allow_nan=False)

    def test_legacy_strict_small_shift_with_unknown_units_remains_valid(self):
        shifted = np.eye(4)
        shifted[0, 3] = 5e-6
        report = self.check(header_fixture(unit="unknown"), header_fixture(shifted, unit="unknown"))
        self.assertEqual(report["accepted_as"], "strict")
        self.assertIsNone(report["max_corner_displacement_mm"])

    def test_roundoff_selected_sform_code_two_is_also_supported(self):
        report = self.check(*user_headers(sform_code=2))
        self.assertEqual(report["accepted_as"], "roundoff")

    def test_selected_sform_not_alternate_qform_determines_grid(self):
        image, label = user_headers()
        alternate = label.header.get_qform()
        alternate[:3, 3] = [-163.355, -295.355, 54.4]
        label.header.set_qform(alternate, code=2)
        label.affine = label.header.get_best_affine()
        report = self.check(image, label)
        self.assertEqual(report["accepted_as"], "roundoff")
        self.assertEqual(report["label_forms"]["selected_form"], "sform")
        self.assertNotEqual(report["label_forms"]["qform_affine"], report["label_selected_affine"])

    def test_one_mm_and_one_voxel_shifts_rejected(self):
        image, label = user_headers()
        for shift in (1.0, float(image.affine[0, 0])):
            with self.subTest(shift=shift):
                label.affine = image.affine.copy()
                label.affine[0, 3] += shift
                self.reject(image, label, "max_cell_corner")

    def test_small_coefficient_spacing_error_accumulates_over_extent(self):
        image = header_fixture()
        changed = np.eye(4)
        changed[0, 0] += 5e-6
        label = header_fixture(changed)
        self.assertTrue(np.allclose(image.affine, label.affine, rtol=0, atol=1e-5))
        self.reject(image, label, "max_cell_corner")

    def test_small_rotation_accumulates_over_extent(self):
        angle = 5e-6
        changed = np.eye(4)
        changed[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        image, label = header_fixture(), header_fixture(changed)
        self.assertTrue(np.allclose(image.affine, label.affine, rtol=0, atol=1e-5))
        self.reject(image, label, "max_cell_corner")

    def test_axis_flip_rejected(self):
        changed = np.eye(4)
        changed[0, 0] = -1
        self.reject(header_fixture(), header_fixture(changed))

    def test_huge_origin_float32_ulp_does_not_override_physical_bound(self):
        a, b = np.eye(4), np.eye(4)
        a[0, 3] = np.float32(1e6)
        b[0, 3] = np.nextafter(np.float32(1e6), np.float32(np.inf))
        self.assertEqual(_float32_ulp_distance(a, b).max(), 1)
        self.reject(header_fixture(a), header_fixture(b), "max_cell_corner")

    def test_voxel_bound_prevents_small_mm_shift_in_very_fine_grid(self):
        a = np.diag([1e-3, 1e-3, 1e-3, 1])
        b = a.copy()
        b[0, 3] = 5e-6
        self.reject(header_fixture(a), header_fixture(b), "max_cell_corner")

    def test_both_grid_inverse_bounds_are_reported(self):
        report = self.check(*user_headers())
        self.assertNotEqual(report["max_corner_displacement_image_voxels"],
                            report["max_corner_displacement_label_voxels"])

    def test_nonfinite_singular_and_invalid_homogeneous_affines_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                image, label = header_fixture(), header_fixture()
                label.affine[0, 3] = value
                self.reject(image, label, "nonfinite")
        image, label = header_fixture(), header_fixture()
        label.affine[0, 0] = 0
        self.reject(image, label, "singular")
        label.affine = np.eye(4)
        label.affine[3, 0] = 1e-8
        self.reject(image, label, "homogeneous")

    def test_invalid_pixdim_rejected(self):
        for value in (0, -1, np.nan, np.inf):
            image, label = header_fixture(), header_fixture()
            label.header["pixdim"][1] = value
            self.reject(image, label, "pixdim")

    def test_unit_mismatch_rejected_even_for_exact_coefficients(self):
        self.reject(header_fixture(unit="mm"), header_fixture(unit="meter"), "units differ")
        self.reject(header_fixture(unit="unknown"), header_fixture(unit="mm"), "units differ")

    def test_known_meter_units_are_converted_for_strict_physical_bound(self):
        changed = np.eye(4)
        changed[0, 3] = 5e-6
        self.reject(header_fixture(unit="meter"), header_fixture(changed, unit="meter"), "max_cell_corner")

    def test_unknown_units_do_not_gain_roundoff_path(self):
        self.reject(*user_headers(unit="unknown"), reason="not eligible")

    def test_nifti2_does_not_gain_float32_roundoff_path(self):
        self.reject(*user_headers(nifti2=True), reason="not eligible")

    def test_nonselected_sform_cannot_enable_roundoff_path(self):
        self.reject(*user_headers(sform_code=0), reason="not eligible")

    def test_different_selected_sform_codes_reject_additional_path(self):
        image, label = user_headers()
        label.header["sform_code"] = 2
        self.reject(image, label, "not eligible")

    def test_more_than_two_affine_ulps_rejected_within_corner_tolerance(self):
        a = np.eye(4)
        a[0, 3] = np.float32(-163.35547)
        b = a.copy()
        value = np.float32(a[0, 3])
        for _ in range(3):
            value = np.nextafter(value, np.float32(-np.inf))
        b[0, 3] = value
        self.reject(header_fixture(a), header_fixture(b), "not eligible")

    def test_more_than_two_pixdim_ulps_rejects_additional_path(self):
        image, label = user_headers()
        value = label.header["pixdim"][1]
        for _ in range(3):
            value = np.nextafter(value, np.float32(np.inf))
        label.header["pixdim"][1] = value
        self.reject(image, label, "not eligible")

    def test_singleton_fourth_dimension_matches_three_dimensional_grid(self):
        image, label = user_headers(shape=(512, 512, 630, 1))
        label.shape = label.shape[:3]
        label.header.set_data_shape(label.shape)
        report = self.check(image, label)
        self.assertEqual(report["spatial_shape"], [512, 512, 630])

    def test_non_singleton_fourth_dimension_and_shape_mismatch_rejected(self):
        self.reject(header_fixture(), header_fixture(shape=(512, 512, 630, 2)), "spatial dimensions")
        self.reject(header_fixture(), header_fixture(shape=(511, 512, 630)), "spatial dimensions")

    def test_float32_steps_across_zero_are_not_origin_relative(self):
        tiny = np.nextafter(np.float32(0), np.float32(1))
        self.assertEqual(int(_float32_ulp_distance([-tiny], [tiny])[0]), 2)
        self.assertEqual(int(_float32_ulp_distance([-0.0], [0.0])[0]), 0)


if __name__ == "__main__":
    unittest.main()
