"""DEBUG storage inventory checks; no preprocessing or medical evaluation."""
import tempfile
import unittest
from pathlib import Path

from tools.online_cp_benchmark import _preprocessed_case_files, OnlineBenchmarkError


class PreprocessedStorageDebugTests(unittest.TestCase):
    def test_numpy_and_blosc2_exact_complete_inventories(self):
        for ending in ("npz", "b2nd"):
            with self.subTest(storage=ending), tempfile.TemporaryDirectory(prefix="debug_storage_") as tmp:
                root = Path(tmp)
                for name in ("a", "b"):
                    for suffix in ([f".{ending}", ".pkl"] + (["_seg.b2nd"] if ending == "b2nd" else [])):
                        (root / (name + suffix)).write_bytes(b"debug storage inventory only")
                storage, data, seg, props = _preprocessed_case_files(root, ["a", "b"])
                self.assertEqual(storage, "blosc2" if ending == "b2nd" else "npz")
                self.assertEqual(set(data), {"a", "b"})
                self.assertEqual(set(props), {"a", "b"})
                self.assertEqual(set(seg), {"a", "b"} if ending == "b2nd" else set())

    def test_missing_segmentation_and_mixed_formats_fail(self):
        with tempfile.TemporaryDirectory(prefix="debug_storage_") as tmp:
            root = Path(tmp)
            (root / "a.pkl").write_bytes(b"debug")
            (root / "a.b2nd").write_bytes(b"debug")
            with self.assertRaisesRegex(OnlineBenchmarkError, "incomplete"):
                _preprocessed_case_files(root, ["a"])
            (root / "a_seg.b2nd").write_bytes(b"debug")
            (root / "a.npz").write_bytes(b"debug")
            with self.assertRaisesRegex(OnlineBenchmarkError, "Mixed"):
                _preprocessed_case_files(root, ["a"])


if __name__ == "__main__":
    unittest.main()
