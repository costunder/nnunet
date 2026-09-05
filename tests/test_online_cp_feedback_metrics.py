"""DEBUG analytic CPU fixtures only; no medical data or final training claim."""
from __future__ import annotations

import math
import unittest
from unittest import mock

try:
    import numpy as np
    import torch
    from custom_trainers import onlinecp_feedback_metrics as feedback
except ModuleNotFoundError as error:
    np = torch = feedback = None
    DEPENDENCY_ERROR = str(error)
else:
    DEPENDENCY_ERROR = ""


@unittest.skipIf(torch is None, "DEBUG dependency unavailable: " + DEPENDENCY_ERROR)
class BatchedActualFeedbackDebugTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([0.5, 0.25, 0.25]).log()[None, :, None, None, None].expand(2, 3, 11, 11, 11).clone()
        self.labels = torch.ones((2, 1, 11, 11, 11), dtype=torch.int16)
        self.mask = torch.zeros_like(self.labels, dtype=torch.bool)
        self.mask[0, :, 4:7, 4:7, 4:7] = True
        self.mask[1, :, 3:8, 3:8, 3:8] = True
        self.labels[self.mask] = 2
        self.valid = torch.ones_like(self.mask)
        self.applied = torch.ones(2, dtype=torch.bool)

    def compute(self, **overrides):
        values = dict(logits=self.logits, labels=self.labels, pasted_mask=self.mask,
                      valid_mask=self.valid, event_applied=self.applied)
        values.update(overrides)
        return feedback.compute_feedback_metrics(**values)

    def test_size_normalized_analytic_components_for_two_lesion_sizes(self):
        out = self.compute()
        self.assertEqual(out["foreground_voxels"].tolist(), [27, 125])
        self.assertEqual(out["boundary_voxels"].tolist(), [26, 98])
        self.assertEqual(out["adjacent_voxels"].tolist(), [98, 218])
        self.assertTrue(out["available"].all())
        for key, expected in (("foreground_ce_raw", math.log(4)), ("foreground_ce", 0.75),
                              ("foreground_error", 0.75), ("boundary_error", 0.75), ("adjacent_fp", 0.25)):
            torch.testing.assert_close(out[key], torch.full((2,), expected))

    def test_bad_prediction_on_unrelated_tumor_does_not_become_lesion_feedback(self):
        original = self.compute()
        self.labels[:, :, 1, 1, 1] = 2
        self.logits[:, :, 1, 1, 1] = torch.tensor([30.0, -30.0, -30.0])
        changed = self.compute()
        for key in ("foreground_ce", "boundary_error", "adjacent_fp"):
            torch.testing.assert_close(original[key], changed[key])

    def test_adjacent_existing_tumor_is_not_a_false_positive_target(self):
        original = self.compute()
        self.labels[0, 0, 3, 4, 4] = 2
        self.logits[0, :, 3, 4, 4] = torch.tensor([-30.0, -30.0, 30.0])
        result = self.compute()
        self.assertEqual(int(result["adjacent_voxels"][0]), int(original["adjacent_voxels"][0]) - 1)
        torch.testing.assert_close(result["adjacent_fp"], original["adjacent_fp"])

    def test_padding_is_ignored_instead_of_becoming_background_false_positive(self):
        original = self.compute()
        self.valid[:, :, :2] = False
        self.labels[:, :, :2] = -1
        self.logits[:, 2, :2] = 50
        changed = self.compute()
        for key in ("foreground_ce", "boundary_error", "adjacent_fp"):
            torch.testing.assert_close(original[key], changed[key])

    def test_no_event_and_erased_event_have_distinct_unavailable_nan_status(self):
        self.mask.zero_()
        self.applied[0] = False
        out = self.compute()
        self.assertEqual(out["status"].tolist(), [feedback.STATUS_NO_EVENT, feedback.STATUS_EMPTY_AFTER_AUGMENTATION])
        self.assertFalse(out["available"].any())
        for key in ("foreground_ce", "foreground_ce_raw", "boundary_error", "adjacent_fp"):
            self.assertTrue(torch.isnan(out[key]).all())

    def test_boundary_contact_and_reported_truncation_are_explicitly_unavailable(self):
        self.mask[0, :, :5, 4:7, 4:7] = True
        self.labels[self.mask] = 2
        out = self.compute(mask_truncated=torch.tensor([False, True]))
        self.assertEqual(out["status"].tolist(), [feedback.STATUS_BOUNDARY_CONTACT, feedback.STATUS_REPORTED_TRUNCATED])
        self.assertTrue(torch.isnan(out["foreground_ce"]).all())

    def test_empty_adjacent_region_is_not_zero_false_positive(self):
        self.labels.fill_(2)
        out = self.compute()
        self.assertEqual(out["status"].tolist(), [feedback.STATUS_EMPTY_ADJACENT] * 2)
        self.assertTrue(torch.isnan(out["adjacent_fp"]).all())
        self.assertTrue((out["foreground_voxels"] > 0).all())

    def test_nonfinite_and_inconsistent_inputs_fail_without_fallback(self):
        cases = []
        bad = self.logits.clone(); bad[0, 0, 0, 0, 0] = float("nan")
        cases.append({"logits": bad})
        bad_mask = self.mask.float(); bad_mask[0, 0, 0, 0, 0] = 0.5
        cases.append({"pasted_mask": bad_mask})
        bad_label = self.labels.clone(); bad_label[0, 0, 4, 4, 4] = 1
        cases.append({"labels": bad_label})
        bad_valid = self.valid.clone(); bad_valid[0, 0, 4, 4, 4] = False
        cases.append({"valid_mask": bad_valid})
        cases.extend([{"event_applied": torch.zeros(2, dtype=torch.bool)},
                      {"boundary_width": 0}, {"adjacent_width": True},
                      {"logits": self.logits[:, :2]}, {"labels": self.labels[:, 0]}])
        for changes in cases:
            with self.subTest(keys=list(changes)), self.assertRaises(feedback.FeedbackMetricError):
                self.compute(**changes)

    def test_batched_result_equals_independent_debug_reference(self):
        batch = self.compute()
        for index in range(2):  # DEBUG oracle only; production metric has no per-item loop.
            one = feedback.compute_feedback_metrics(
                self.logits[index:index + 1], self.labels[index:index + 1], self.mask[index:index + 1],
                valid_mask=self.valid[index:index + 1], event_applied=self.applied[index:index + 1])
            for key in batch:
                torch.testing.assert_close(batch[key][index:index + 1], one[key])

    def test_observations_detach_without_changing_training_logits_or_inputs(self):
        self.logits.requires_grad_()
        snapshots = [value.clone() for value in (self.logits, self.labels, self.mask, self.valid)]
        result = self.compute()
        self.assertTrue(all(not value.requires_grad for value in result.values()))
        self.assertIsNone(self.logits.grad)
        for value, before in zip((self.logits, self.labels, self.mask, self.valid), snapshots):
            torch.testing.assert_close(value, before)
        # The actual forward's unchanged logits can still contribute to the normal loss.
        torch.nn.functional.cross_entropy(self.logits, self.labels[:, 0].long()).backward()
        self.assertIsNotNone(self.logits.grad)

    def test_geometric_ce_normalization_is_not_clipped_or_arithmetic_error(self):
        self.logits[0, :, 4, 4, 4] = torch.tensor([20.0, 0.0, -20.0])
        out = self.compute()
        expected = -torch.expm1(-out["foreground_ce_raw"])
        torch.testing.assert_close(out["foreground_ce"], expected)
        self.assertGreater(float(out["foreground_ce"][0]), float(out["foreground_error"][0]))
        self.assertTrue(((out["foreground_ce"] >= 0) & (out["foreground_ce"] <= 1)).all())

    def test_single_voxel_spatial_dimension_is_unavailable_not_a_pooling_crash(self):
        out = feedback.compute_feedback_metrics(
            torch.zeros((1, 3, 1, 1, 1)), torch.full((1, 1, 1, 1, 1), 2),
            torch.ones((1, 1, 1, 1, 1), dtype=torch.bool),
            valid_mask=torch.ones((1, 1, 1, 1, 1), dtype=torch.bool), event_applied=torch.tensor([True]))
        self.assertEqual(int(out["status"][0]), feedback.STATUS_BOUNDARY_CONTACT)
        self.assertTrue(torch.isnan(out["foreground_ce"]).all())

    def test_finite_extreme_logits_cannot_publish_nonfinite_measurement(self):
        self.logits[:, 0] = torch.finfo(torch.float32).max
        self.logits[:, 2] = -torch.finfo(torch.float32).max
        with self.assertRaisesRegex(feedback.FeedbackMetricError, "nonfinite tumor log probability"):
            self.compute()

    def test_saturated_probability_components_retain_analytic_unit_interval(self):
        self.logits.fill_(-100)
        self.logits[0, 0] = 100
        self.logits[1, 2] = 100
        result = self.compute()
        self.assertTrue(result["available"].all())
        for name in ("foreground_ce", "foreground_error", "boundary_error", "adjacent_fp"):
            self.assertTrue(((result[name] >= 0) & (result[name] <= 1)).all())


@unittest.skipIf(torch is None, "DEBUG dependency unavailable: " + DEPENDENCY_ERROR)
class SharedNnUNetTransformDebugTests(unittest.TestCase):
    def setUp(self):
        try:
            from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
            from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTransform
            from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
            from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
            from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
        except ModuleNotFoundError as error:
            self.skipTest("Actual nnU-Net augmentation dependency unavailable: " + str(error))
        self.Compose, self.Remove, self.DS = ComposeTransforms, RemoveLabelTransform, DownsampleSegForDSTransform
        self.Spatial, self.Mirror = SpatialTransform, MirrorTransform
        self.image = torch.arange(11 ** 3).reshape(1, 11, 11, 11).float() / 1000
        self.seg = torch.ones_like(self.image, dtype=torch.int16)
        self.mask = torch.zeros_like(self.seg, dtype=torch.bool)
        self.mask[:, 4:7, 4:7, 4:7] = True
        self.seg[self.mask] = 2

    def wrap(self, transform, **kwargs):
        return feedback.transform_with_feedback(transform, self.image, self.seg, self.mask, **kwargs)

    def test_all_ds_targets_are_stripped_and_masks_follow_same_nearest_exact(self):
        transform = self.Compose([self.Remove(-1, 0), self.DS([1, 0.5, 0.25])])
        self.seg[:, :2] = -1
        before = self.seg.clone()
        result = self.wrap(transform)
        self.assertEqual(len(result["segmentation"]), 3)
        for target, mask, valid in zip(result["segmentation"], result["pasted_mask_ds"], result["valid_mask_ds"]):
            self.assertEqual(target.shape[0], 1)
            self.assertEqual(tuple(target.shape), tuple(mask.shape))
            self.assertTrue((~mask | ((target == 2) & valid)).all())
        torch.testing.assert_close(self.seg, before)
        self.assertEqual(int(result["input_pasted_voxels"]), 27)
        self.assertFalse(result["valid_mask"][:, :2].any())

    def test_mirror_flips_segmentation_pasted_support_and_regression_validity_together(self):
        mirror = self.Mirror((0, 1, 2))
        self.seg[:, :2] = -1
        self.mask.zero_(); self.mask[:, 2:4, 4:6, 6:8] = True; self.seg[self.mask] = 2
        transform = self.Compose([mirror, self.Remove(-1, 0)])
        with mock.patch.object(mirror, "get_parameters", return_value={"axes": [0, 2]}):
            result = self.wrap(transform)
        torch.testing.assert_close(result["pasted_mask"], torch.flip(self.mask, (1, 3)))
        torch.testing.assert_close(result["valid_mask"], torch.flip(self.seg != -1, (1, 3)))
        torch.testing.assert_close(result["image"], torch.flip(self.image, (1, 3)))

    def test_all_valid_input_does_not_make_spatial_padding_valid(self):
        spatial = self.Spatial((15, 15, 15), patch_center_dist_from_border=0,
                               random_crop=False, p_elastic_deform=0, p_rotation=0, p_scaling=0)
        result = self.wrap(self.Compose([spatial, self.Remove(-1, 0)]))
        self.assertEqual(int(result["valid_mask"].sum()), 11 ** 3)
        self.assertFalse(result["valid_mask"][:, 0].any())
        self.assertEqual(int(result["pasted_mask"].sum()), 27)

    def test_affine_all_valid_input_has_invalid_out_of_grid_support(self):
        spatial = self.Spatial((15, 15, 15), patch_center_dist_from_border=0,
                               random_crop=False, p_elastic_deform=0, p_rotation=1,
                               rotation=(0.31, 0.31), p_scaling=1, scaling=(1.2, 1.2),
                               bg_style_seg_sampling=False)
        result = self.wrap(self.Compose([spatial, self.Remove(-1, 0)]))
        self.assertFalse(bool(result["valid_mask"][0, 0, 0, 0]))
        self.assertTrue(bool(result["valid_mask"][0, 7, 7, 7]))
        self.assertLess(int(result["valid_mask"].sum()), 15 ** 3)

    def test_rotation_and_scaling_preserve_authoritative_targets_and_audit_intersection(self):
        # A real three-class junction creates distinct per-channel argmax ties;
        # no synthetic transform output is substituted for this regression.
        generator = torch.Generator().manual_seed(197)
        self.seg = torch.randint(0, 3, self.seg.shape, generator=generator, dtype=torch.int16)
        self.mask = self.seg == 2
        spatial = self.Spatial((9, 9, 9), patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                               p_rotation=1, rotation=(0.37, 0.37), p_scaling=1, scaling=(0.9, 0.9),
                               bg_style_seg_sampling=False)
        transform = self.Compose([spatial, self.Remove(-1, 0), self.DS([1, 0.5])])
        np.random.seed(91); torch.manual_seed(91)
        baseline = transform(image=self.image.clone(), segmentation=self.seg.clone())
        np.random.seed(91); torch.manual_seed(91)
        result = self.wrap(transform)
        torch.testing.assert_close(result["image"], baseline["image"], rtol=0, atol=0)
        for actual, expected in zip(result["segmentation"], baseline["segmentation"]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Binary support and semantic 3-class interpolation can have different
        # extents; never inflate the attributed mask to all target tumor voxels.
        self.assertNotEqual(int((result["segmentation"][0] == 2).sum()), int(result["pasted_mask"].sum()))
        self.assertEqual(int(result["raw_support_count"]), int(result["pasted_mask"].sum()
                         + result["removed_by_label_resampling"] + result["removed_by_padding"]))

    def test_complete_cropping_away_is_reported_as_empty_not_identity_fallback(self):
        self.mask.zero_(); self.mask[:, :2, :2, :2] = True; self.seg[self.mask] = 2
        spatial = self.Spatial((5, 5, 5), patch_center_dist_from_border=0,
                               random_crop=False, p_elastic_deform=0, p_rotation=0, p_scaling=0)
        result = self.wrap(self.Compose([spatial, self.Remove(-1, 0)]))
        self.assertEqual(int(result["input_pasted_voxels"]), 8)
        self.assertEqual(int(result["pasted_mask"].sum()), 0)
        out = feedback.compute_feedback_metrics(torch.zeros((1, 3, 5, 5, 5)), result["segmentation"][None],
            result["pasted_mask"][None], valid_mask=result["valid_mask"][None], event_applied=torch.tensor([True]))
        self.assertEqual(int(out["status"][0]), feedback.STATUS_EMPTY_AFTER_AUGMENTATION)

    def test_unsupported_configuration_and_pretransform_mismatch_are_errors(self):
        identity = self.Compose([])
        for options in ({"is_cascaded": True}, {"regions": []}, {"do_dummy_2d_data_aug": True}, {"ignore_label": 255}):
            with self.subTest(options=options), self.assertRaises(feedback.FeedbackMetricError):
                self.wrap(identity, **options)
        self.seg[self.mask] = 1
        with self.assertRaisesRegex(feedback.FeedbackMetricError, "input pasted mask disagrees"):
            self.wrap(identity)

    def test_actual_nnunet_training_pipeline_keeps_rng_image_and_all_ds_labels(self):
        try:
            from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        except ModuleNotFoundError as error:
            self.skipTest("Actual nnU-Net dependency unavailable: " + str(error))
        transform = nnUNetTrainer.get_training_transforms(
            (9, 9, 9), (-0.3, 0.3), [[1, 1, 1], [0.5, 0.5, 0.5]], (0, 1, 2), False,
            use_mask_for_norm=[True], is_cascaded=False, foreground_labels=[1, 2], regions=None, ignore_label=None)
        self.seg[:, :1] = -1
        np.random.seed(824); torch.manual_seed(824)
        baseline = transform(image=self.image.clone(), segmentation=self.seg.clone())
        baseline_numpy_state, baseline_torch_state = np.random.get_state(), torch.get_rng_state()
        np.random.seed(824); torch.manual_seed(824)
        result = self.wrap(transform)
        torch.testing.assert_close(result["image"], baseline["image"], rtol=0, atol=0)
        for actual, expected in zip(result["segmentation"], baseline["segmentation"]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        np.testing.assert_equal(np.random.get_state(), baseline_numpy_state)
        torch.testing.assert_close(torch.get_rng_state(), baseline_torch_state, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
