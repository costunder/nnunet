"""UNIT exact-array persistence tests; no original CT or neural execution."""
from contextlib import contextmanager
import copy
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import numpy as np

from hiercp_v1x import u_bridge_fields as fields


def close_mappings():
    for arrays, _ in fields._OPENED.values():
        for array in arrays:
            array._mmap.close()
    fields._OPENED.clear()
    gc.collect()


@contextmanager
def unit_directory():
    with TemporaryDirectory(prefix="UNIT_u_bridge_fields_",
                            dir=Path(__file__).resolve().parents[1]) as directory:
        try:
            yield Path(directory)
        finally:
            close_mappings()


def binding():
    return dict(image_sha256="a" * 64, label_sha256="b" * 64, common_sha256="c" * 64,
                shape=(3, 4, 5), spacing=(.7, .7, 2.), source_identity="UNIT_original_CT")


class FieldTests(unittest.TestCase):
    def test_exact_bytes_dtype_shape_units_and_readonly_mmaps(self):
        with unit_directory() as root:
            depth = np.arange(60, dtype=np.float32).reshape(3, 4, 5) / 8
            occupied = np.asfortranarray(np.arange(60, dtype=np.float64).reshape(3, 4, 5) / 3)
            mapped_depth, mapped_occupied, receipt = fields.cached_fields(
                root, "UNIT_case", binding(), lambda: depth, lambda: occupied, lambda: None)
            for actual, expected in ((mapped_depth, depth), (mapped_occupied, occupied)):
                self.assertIsInstance(actual, np.memmap)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.tobytes(order="A"), expected.tobytes(order="A"))
                self.assertFalse(actual.flags.writeable)
                with self.assertRaises(ValueError):
                    actual.flat[0] = 99
            self.assertTrue(mapped_occupied.flags.f_contiguous)
            self.assertEqual(receipt["array_bytes"], depth.nbytes + occupied.nbytes)
            self.assertGreater(receipt["disk_bytes"], receipt["array_bytes"])
            self.assertEqual(receipt["status"], "built")
            self.assertGreaterEqual(receipt["wall_seconds"], 0.)
            self.assertTrue(receipt["whole_case"])
            if fields.sys.platform == "win32":
                self.assertFalse(receipt["page_release_hint"]["supported"])
                self.assertFalse(receipt["page_release_hint"]["applied"])
                self.assertEqual(receipt["page_release_hint"]["advised_bytes"], 0)
            metadata = json.loads((root / "UNIT_case/metadata.json").read_text())
            self.assertEqual(metadata["fields"]["depth"]["units"], "mm")
            self.assertEqual(metadata["fields"]["occupied"]["sha256"], fields._sha(root / "UNIT_case/occupied.npy"))

    def test_factories_called_once_and_zero_times_after_cold_reopen(self):
        with unit_directory() as root:
            count = dict(depth=0, occupied=0, budget=0)
            def factory(name):
                count[name] += 1
                return np.full((3, 4, 5), count[name], dtype=np.float32)
            def budget():
                count["budget"] += 1
            args = (root, "UNIT_case", binding(), lambda: factory("depth"), lambda: factory("occupied"), budget)
            first = fields.cached_fields(*args)
            second = fields.cached_fields(*args)
            self.assertIs(first[0], second[0])
            self.assertEqual(second[2]["status"], "resident_mapping")
            close_mappings()
            third = fields.cached_fields(*args)
            self.assertEqual(third[2]["status"], "reopened")
            self.assertEqual((count["depth"], count["occupied"]), (1, 1))
            self.assertGreater(count["budget"], 3)
            np.testing.assert_array_equal(third[0], np.ones((3, 4, 5), dtype=np.float32))

    def test_Linux_page_release_hint_only_when_available_and_mapping_readonly(self):
        advise = Mock()
        array = SimpleNamespace(_mmap=SimpleNamespace(madvise=advise), nbytes=4096,
                                flags=SimpleNamespace(writeable=False))
        with patch.object(fields.sys, "platform", "linux"), \
                patch.object(fields.mmap, "MADV_DONTNEED", 4, create=True):
            result = fields._release_readonly_pages(array)
            advise.assert_called_once_with(4)
            self.assertTrue(result["supported"])
            self.assertTrue(result["applied"])
            self.assertEqual(result["advised_bytes"], 4096)
            array.flags.writeable = True
            with self.assertRaisesRegex(ValueError, "read-only"):
                fields._release_readonly_pages(array)
        advise.reset_mock()
        array.flags.writeable = False
        with patch.object(fields.sys, "platform", "win32"):
            self.assertFalse(fields._release_readonly_pages(array)["supported"])
            advise.assert_not_called()
        with patch.object(fields.sys, "platform", "linux"), \
                patch.object(fields.mmap, "MADV_DONTNEED", None, create=True):
            self.assertFalse(fields._release_readonly_pages(array)["supported"])
            advise.assert_not_called()

    def test_page_release_applied_only_after_cold_validation_and_values_preserved(self):
        with unit_directory() as root:
            depth = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
            occupied = np.arange(60, dtype=np.float64).reshape(3, 4, 5) / 2
            def hint(array):
                self.assertFalse(array.flags.writeable)
                return dict(supported=True, applied=True, advised_bytes=array.nbytes,
                            reason="UNIT mocked OS hint", method="madvise(MADV_DONTNEED)")
            args = (root, "UNIT_case", binding(), lambda: depth, lambda: occupied, lambda: None)
            with patch.object(fields, "_release_readonly_pages", side_effect=hint) as release:
                first = fields.cached_fields(*args)
                self.assertEqual(release.call_count, 2)
                self.assertEqual(first[2]["page_release_hint"]["advised_bytes"], depth.nbytes + occupied.nbytes)
                second = fields.cached_fields(*args)
                self.assertEqual(release.call_count, 2)
                self.assertFalse(second[2]["page_release_hint"]["applied"])
                self.assertEqual(second[2]["page_release_hint"]["advised_bytes"], 0)
                self.assertTrue(all(not h["applied"] for h in second[2]["page_release_hint"]["fields"].values()))
                np.testing.assert_array_equal(second[0], depth)
                np.testing.assert_array_equal(second[1], occupied)
                close_mappings()
                third = fields.cached_fields(*args)
                self.assertEqual(release.call_count, 4)
                self.assertEqual(third[2]["status"], "reopened")
                np.testing.assert_array_equal(third[0], depth)
                np.testing.assert_array_equal(third[1], occupied)

    def test_OS_hint_rejection_visible_without_mutating_array(self):
        array = SimpleNamespace(_mmap=SimpleNamespace(madvise=Mock(side_effect=OSError("UNIT hint rejected"))),
                                nbytes=4096, flags=SimpleNamespace(writeable=False))
        with patch.object(fields.sys, "platform", "linux"), \
                patch.object(fields.mmap, "MADV_DONTNEED", 4, create=True):
            result = fields._release_readonly_pages(array)
        self.assertTrue(result["supported"])
        self.assertFalse(result["applied"])
        self.assertEqual(result["advised_bytes"], 0)
        self.assertIn("UNIT hint rejected", result["reason"])

    def test_changed_raw_source_or_spacing_binding_fails_without_recompute(self):
        with unit_directory() as root:
            array = np.ones((3, 4, 5), np.float32)
            fields.cached_fields(root, "UNIT_case", binding(), lambda: array, lambda: array, lambda: None)
            close_mappings()
            for key, value in (("image_sha256", "d" * 64), ("common_sha256", "e" * 64),
                               ("spacing", [1., 1., 2.]), ("shape", [4, 4, 5])):
                changed = dict(binding(), **{key: value})
                with self.subTest(key=key), patch.object(fields, "_check_array") as check:
                    with self.assertRaisesRegex(ValueError, "binding/metadata identity differs"):
                        fields.cached_fields(root, "UNIT_case", changed,
                            lambda: self.fail("Changed binding recomputed depth"),
                            lambda: self.fail("Changed binding recomputed occupied"), lambda: None)
                    check.assert_not_called()

    def test_saved_array_corruption_and_metadata_hash_corruption_fail(self):
        for mode in ("array", "metadata", "header"):
            with self.subTest(mode=mode), unit_directory() as root:
                array = np.ones((3, 4, 5), np.float32)
                fields.cached_fields(root, "UNIT_case", binding(), lambda: array, lambda: array, lambda: None)
                close_mappings()
                path = root / "UNIT_case/depth.npy"
                metadata_path = root / "UNIT_case/metadata.json"
                metadata = json.loads(metadata_path.read_text())
                if mode == "array":
                    with path.open("r+b") as stream:
                        stream.seek(-1, 2)
                        stream.write(b"X")
                    expected = "SHA256 differs"
                elif mode == "metadata":
                    metadata["fields"]["depth"]["dtype"] = "<f8"
                    metadata_path.write_text(json.dumps(metadata))
                    expected = "metadata identity differs"
                else:
                    # Self-consistent file hash with incompatible NumPy header
                    # must still fail the independent shape/dtype contract.
                    np.save(path, np.ones((3, 4, 5), dtype=np.float64))
                    metadata["fields"]["depth"].update(file_bytes=path.stat().st_size, sha256=fields._sha(path))
                    metadata.pop("metadata_sha256")
                    metadata["metadata_sha256"] = fields._digest(metadata)
                    metadata_path.write_text(json.dumps(metadata))
                    expected = "NumPy header differs"
                before = path.read_bytes(), metadata_path.read_bytes()
                with self.assertRaisesRegex(ValueError, expected):
                    fields.cached_fields(root, "UNIT_case", binding(),
                        lambda: self.fail("Corrupt cache rebuilt"), lambda: self.fail("Corrupt cache rebuilt"), lambda: None)
                self.assertEqual((path.read_bytes(), metadata_path.read_bytes()), before)

    def test_nonfinite_wrong_shape_and_nonfloating_fields_never_publish(self):
        for bad in (np.full((3, 4, 5), np.nan, np.float32), np.full((3, 4, 5), np.inf, np.float32),
                    np.full((3, 4, 5), -1., np.float32), np.ones((3, 4, 4), np.float32),
                    np.ones((3, 4, 5), np.int32)):
            with self.subTest(dtype=bad.dtype, shape=bad.shape), unit_directory() as root:
                with self.assertRaises(ValueError):
                    fields.cached_fields(root, "UNIT_case", binding(), lambda: bad,
                        lambda: self.fail("Second factory ran after invalid first field"), lambda: None)
                self.assertFalse((root / "UNIT_case").exists())
                self.assertEqual(len(list(root.glob(".UNIT_case.attempt.*"))), 1)

    def test_interrupted_attempt_preserved_and_fresh_attempt_can_complete(self):
        with unit_directory() as root:
            array = np.ones((3, 4, 5), np.float32)
            def interrupted():
                raise RuntimeError("UNIT interrupted second EDT")
            with self.assertRaisesRegex(RuntimeError, "interrupted second EDT"):
                fields.cached_fields(root, "UNIT_case", binding(), lambda: array, interrupted, lambda: None)
            attempts = list(root.glob(".UNIT_case.attempt.*"))
            self.assertEqual(len(attempts), 1)
            before = (attempts[0] / "depth.npy").read_bytes()
            self.assertFalse((root / "UNIT_case").exists())
            fields.cached_fields(root, "UNIT_case", binding(), lambda: array, lambda: array, lambda: None)
            self.assertTrue((root / "UNIT_case/metadata.json").is_file())
            self.assertEqual((attempts[0] / "depth.npy").read_bytes(), before)

    def test_incomplete_final_directory_is_not_rebuilt_or_overwritten(self):
        with unit_directory() as root:
            target = root / "UNIT_case"
            target.mkdir()
            original = target / "depth.npy"
            original.write_bytes(b"UNIT interrupted existing final file")
            with self.assertRaisesRegex(ValueError, "Incomplete whole-case field publication"):
                fields.cached_fields(root, "UNIT_case", binding(), lambda: self.fail("Incomplete cache replaced"),
                                     lambda: self.fail("Incomplete cache replaced"), lambda: None)
            self.assertEqual(original.read_bytes(), b"UNIT interrupted existing final file")

    def test_disk_budget_checked_before_factory_without_input_fallback(self):
        with unit_directory() as root, patch.object(fields.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(OSError, "no field cropped, quantized, or replaced"):
                fields.cached_fields(root, "UNIT_case", binding(), lambda: self.fail("EDT ran with no storage"),
                                     lambda: self.fail("EDT ran with no storage"), lambda: None)
            self.assertEqual(list(root.iterdir()), [])

    def test_binding_and_case_identity_required_before_any_factory(self):
        with unit_directory() as root:
            for bad in (dict(binding(), shape=[3, 0, 5]), dict(binding(), spacing=[1., float("nan"), 1.]),
                        {k: v for k, v in binding().items() if k != "common_sha256"}):
                with self.assertRaises(ValueError):
                    fields.cached_fields(root, "UNIT_case", bad, lambda: self.fail("Invalid binding ran EDT"),
                                         lambda: self.fail("Invalid binding ran EDT"), lambda: None)
            with self.assertRaises(ValueError):
                fields.cached_fields(root, "../outside", binding(), lambda: None, lambda: None, lambda: None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
