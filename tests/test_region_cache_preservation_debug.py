"""DEBUG native region-cache publication/retry checks on synthetic volumes only.

The complete configured 24 regions are retained. Files live exclusively in each
test's temporary directory; no medical case, model, or training result is used.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from hiercp import region
from hiercp.common import load_case
from hiercp.schema import GraphBuildConfig
from tests.test_region_observed_ct_debug import _case


def _snapshot(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


class RegionCachePreservationDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        payload = json.loads((Path(__file__).resolve().parents[1] / "config/train.json").read_text(encoding="utf-8"))
        cls.config = GraphBuildConfig(**payload["graph"])
        assert cls.config.num_regions == 24
        xyz = np.indices((17, 17, 17), dtype=np.float32)
        cls.image = 20 + 3 * xyz[0] + 2 * xyz[1] + xyz[2]
        cls.regions = region.build_patient_regions(_case(cls.image, 2), config=cls.config,
            liver_label=1, tumor_label=2, ct_clip=(-200., 300.), rng=np.random.default_rng(42))
        cls.metadata = {"format": region.REGION_CACHE_FORMAT,
                        "descriptor_policy": region.REGION_DESCRIPTOR_POLICY,
                        "case_id": "debug_publication", "shape": list(cls.image.shape)}

    def save(self, target, **kwargs):
        region.save_patient_regions(self.regions, target, metadata=dict(self.metadata), **kwargs)

    def test_existing_legacy_and_current_results_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            legacy = parent / "legacy_v2"
            legacy.mkdir()
            (legacy / "metadata.json").write_text('{"format":"hiercp_patient_regions_v2"}', encoding="utf-8")
            (legacy / "historical_result.bin").write_bytes(b"DEBUG preserved historical artifact")
            before = _snapshot(legacy)
            for overwrite in (False, True):
                with self.assertRaisesRegex(FileExistsError, "preserved"):
                    self.save(legacy, overwrite=overwrite)
                self.assertEqual(before, _snapshot(legacy))
            current = parent / "current_v3"
            self.save(current)
            before = _snapshot(current)
            loaded, _ = region.load_patient_regions(current)
            np.testing.assert_array_equal(loaded.region_features, self.regions.region_features)
            with self.assertRaisesRegex(FileExistsError, "preserved"):
                self.save(current, overwrite=True)
            self.assertEqual(before, _snapshot(current))

    def test_interrupted_staging_does_not_block_same_final_path_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "retry"
            with patch.object(region.np, "savez_compressed", side_effect=OSError("DEBUG injected staging write failure")):
                with self.assertRaisesRegex(OSError, "DEBUG injected"):
                    self.save(target)
            self.assertFalse(target.exists())
            failed_staging = list(parent.glob(".retry.tmp.*"))
            self.assertEqual(len(failed_staging), 1)
            before = _snapshot(failed_staging[0])
            self.save(target)
            loaded, _ = region.load_patient_regions(target)
            np.testing.assert_array_equal(loaded.region_features, self.regions.region_features)
            self.assertEqual(before, _snapshot(failed_staging[0]))

    def test_nonempty_rename_collision_preserves_foreign_results_and_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "collision"
            native_rename = Path.rename

            def collide(staging, destination):
                if Path(destination) == target:
                    target.mkdir()
                    (target / "foreign_result.bin").write_bytes(b"DEBUG concurrent owner")
                return native_rename(staging, destination)

            with patch.object(Path, "rename", new=collide):
                with self.assertRaises(OSError):
                    self.save(target)
            self.assertEqual((target / "foreign_result.bin").read_bytes(), b"DEBUG concurrent owner")
            completed = list(parent.glob(".collision.tmp.*"))
            self.assertEqual(len(completed), 1)
            loaded, _ = region.load_patient_regions(completed[0])
            np.testing.assert_array_equal(loaded.region_features, self.regions.region_features)

    def test_directory_symlink_guard_precedes_cache_reader(self):
        # Boundary injection avoids requiring Windows symlink privileges. The
        # native cache reader must never follow a case path marked as a symlink.
        with tempfile.TemporaryDirectory() as directory:
            case = _case(self.image, 2)
            target = Path(directory) / case.paths.case_id
            target.mkdir()
            native_is_symlink = Path.is_symlink

            def marked_symlink(path):
                return path == target or native_is_symlink(path)

            with patch.object(region, "_region_cache_metadata", return_value={}), \
                    patch.object(Path, "is_symlink", new=marked_symlink), \
                    patch.object(region, "load_patient_regions", side_effect=AssertionError("must not follow alias")):
                with self.assertRaisesRegex(FileExistsError, "symlink is preserved"):
                    region.load_or_build_patient_regions(case, cache_dir=directory,
                        liver_label=1, tumor_label=2, config=self.config,
                        ct_clip=(-200., 300.), seed=42)
            self.assertTrue(target.is_dir())

    def test_native_source_verified_cache_reuse_ignores_destructive_overwrite_request(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = _case(self.image, 2)
            image_path, label_path = parent / "debug_0000.nii.gz", parent / "debug.nii.gz"
            image_nii = nib.Nifti1Image(source.image, np.eye(4))
            label_nii = nib.Nifti1Image(source.label, np.eye(4))
            image_nii.header.set_xyzt_units("mm")
            label_nii.header.set_xyzt_units("mm")
            nib.save(image_nii, image_path)
            nib.save(label_nii, label_path)
            case = load_case(SimpleNamespace(case_id="debug_cache", image_path=image_path, label_path=label_path))
            options = dict(cache_dir=parent / "cache", liver_label=1, tumor_label=2,
                           config=self.config, ct_clip=(-200., 300.), seed=42)
            initial = region.load_or_build_patient_regions(case, **options)
            target = parent / "cache" / "debug_cache"
            before = _snapshot(target)
            with patch.object(region, "build_patient_regions", side_effect=AssertionError("Verified current cache must not be rebuilt")):
                for overwrite in (False, True):
                    reused = region.load_or_build_patient_regions(case, overwrite=overwrite, **options)
                    np.testing.assert_array_equal(initial.region_features, reused.region_features)
            self.assertEqual(before, _snapshot(target))


if __name__ == "__main__":
    unittest.main()
