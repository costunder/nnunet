"""DEBUG preprocessing/storage integration; no patient or model performance claims."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from custom_trainers.onlinecp_raw_bank import RawBankStore
from tools.online_raw_bank_preparation import prepare_raw_case, prepare_source_candidates


def fixture():
    from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
    image = np.full((17, 17, 17), -40, dtype=np.float32)
    labels = np.ones(image.shape, dtype=np.int16)
    labels[13, 13, 13], image[13, 13, 13] = 2, 400
    spacing = [.7, .7, .7]
    target = [1., 1., 1.]
    shape = [12, 12, 12]
    data_kwargs = dict(is_seg=False, order=3, order_z=0, force_separate_z=None)
    seg_kwargs = dict(is_seg=True, order=1, order_z=0, force_separate_z=None)
    plans = {"image_reader_writer": "NibabelIO", "transpose_forward": [0, 1, 2],
             "foreground_intensity_properties_per_channel": {
                 "0": dict(mean=0., std=200., percentile_00_5=-1000., percentile_99_5=1000.)},
             "configurations": {"debug": {"architecture": {},  # No network is constructed by this test.
                 "spacing": target, "preprocessor_name": "DefaultPreprocessor",
                 "normalization_schemes": ["CTNormalization"], "use_mask_for_norm": [False],
                 "resampling_fn_data": "resample_data_or_seg_to_shape", "resampling_fn_data_kwargs": data_kwargs,
                 "resampling_fn_seg": "resample_data_or_seg_to_shape", "resampling_fn_seg_kwargs": seg_kwargs}}}
    properties = {"spacing": spacing, "shape_before_cropping": [17, 17, 17],
                  "shape_after_cropping_and_before_resampling": [17, 17, 17],
                  "bbox_used_for_cropping": [[0, 17], [0, 17], [0, 17]]}
    native_data = resample_data_or_seg_to_shape(image.transpose(2, 1, 0)[None] / 200,
                                              shape, spacing, target, **data_kwargs)
    native_seg = resample_data_or_seg_to_shape(labels.transpose(2, 1, 0)[None],
                                             shape, spacing, target, **seg_kwargs)
    return image, labels, properties, plans, native_data, native_seg


class OnlineRawBankPreparationDebug(unittest.TestCase):
    def test_complete_native_baseline_and_every_debug_candidate_preserved(self):
        image, labels, properties, plans, data, seg = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case, relative, digest = prepare_raw_case(
                root, "debug", image, labels, properties, plans, data, seg,
                configuration_name="debug", raw_spacing=[.7] * 3, raw_spatial_unit="mm",
                minimum_free_bytes=0)
            source = image[11:16, 11:16, 11:16]
            mask = labels[11:16, 11:16, 11:16] == 2
            centers = np.asarray([[3, 3, 3], [3, 3, 4], [3, 4, 3], [4, 3, 3]])
            seen = []

            def debug_ordered_map(function, tasks):
                # Explicit DEBUG mapper: test each real supplied task, not the production resource pilot.
                seen.extend(index for index, _ in tasks)
                return [function(task) for task in tasks]

            paths, hashes, audit = prepare_source_candidates(
                root, "debug", 1, case, digest, source, mask, [2, 2, 2], centers, [12] * 3,
                candidate_map=debug_ordered_map)
            self.assertEqual(seen, list(range(4)))
            self.assertEqual(len(paths), 4)
            self.assertEqual(len(audit["native_support_voxels"]), 4)
            self.assertTrue(audit["all_selected_raw_candidates_retained"])
            self.assertFalse(audit["source_origin_resampling_used"])
            store = RawBankStore(root)
            try:
                stored = store.load_case(relative, digest)
                np.testing.assert_array_equal(stored["baseline_seg"], seg)
                for index, (path, sha) in enumerate(zip(paths, hashes)):
                    candidate = store.load_candidate(str(path), str(sha))
                    self.assertEqual(candidate["raw_target_center"], centers[index].tolist())
                    self.assertEqual(candidate["case_reference_sha256"], digest)
                    self.assertEqual(int(candidate["source_mask"].sum()), 1)
            finally:
                store.close()

    def test_native_baseline_mismatch_is_not_certified(self):
        image, labels, properties, plans, data, seg = fixture()
        wrong = seg.copy()
        wrong[0, 0, 0, 0] = 0
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "full-label baseline mismatch"):
                prepare_raw_case(directory, "debug", image, labels, properties, plans, data, wrong,
                                 configuration_name="debug", raw_spacing=[.7] * 3, raw_spatial_unit="mm",
                                 minimum_free_bytes=0)
            self.assertFalse((Path(directory) / "raw_cases/debug.json").exists())

    def test_disk_reserve_checked_before_native_preparation(self):
        image, labels, properties, plans, data, seg = fixture()
        with tempfile.TemporaryDirectory() as directory, \
                patch("tools.online_raw_bank_preparation.shutil.disk_usage") as usage, \
                patch("tools.online_raw_bank_preparation.prepare_case") as prepare:
            usage.return_value.free = 10 * int(np.prod(data.shape[1:])) + 9
            with self.assertRaisesRegex(RuntimeError, "reserved_free=10"):
                prepare_raw_case(directory, "debug", image, labels, properties, plans, data, seg,
                                 configuration_name="debug", raw_spacing=[.7] * 3, raw_spatial_unit="mm",
                                 minimum_free_bytes=10)
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
