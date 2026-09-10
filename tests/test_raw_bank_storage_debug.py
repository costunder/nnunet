"""DEBUG-only serialization fixtures, not patient/model performance tests."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from custom_trainers.onlinecp_raw_bank import RawBankStore, save_case, save_candidate


class RawBankStorageDebug(unittest.TestCase):
    def test_case_mmap_and_candidate_roundtrip_without_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {"metadata": {"preprocessed_shape": [3, 4, 5]},
                    "baseline_unclipped": np.arange(60, dtype=np.float64).reshape(1, 3, 4, 5),
                    "data_operators": [np.eye(3), np.eye(4), np.eye(5)],
                    "preparation": {"raw_ct": np.ones((1, 30, 40, 50))}}
            candidate = {"output_bbox": [[0, 1], [1, 3], [2, 4]],
                         "pasted_support": np.zeros((1, 2, 2), dtype=bool),
                         "source_mask": np.ones((2, 3, 4), dtype=bool)}
            case_sha = save_case(root, "cases/a.json", case)
            candidate_sha = save_candidate(root, "candidates/a/0.json", candidate)
            store = RawBankStore(root)
            loaded = store.load_case("cases/a.json", case_sha)
            self.assertNotIn("preparation", loaded)
            self.assertIsInstance(loaded["baseline_unclipped"], np.memmap)
            self.assertFalse(loaded["baseline_unclipped"].flags.writeable)
            np.testing.assert_array_equal(loaded["baseline_unclipped"], case["baseline_unclipped"])
            self.assertIs(loaded["baseline_unclipped"], store.load_case("cases/a.json", case_sha)["baseline_unclipped"])
            result = store.load_candidate("candidates/a/0.json", candidate_sha)
            np.testing.assert_array_equal(result["source_mask"], candidate["source_mask"])
            self.assertFalse(result["pasted_support"].any())
            self.assertEqual(save_case(root, "cases/a.json", case), case_sha)
            self.assertEqual(save_candidate(root, "candidates/a/0.json", candidate), candidate_sha)
            store.close()

    def test_immutable_publication_does_not_replace_existing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = save_candidate(root, "c/0.json", {"x": np.array([1])})
            with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                save_candidate(root, "c/0.json", {"x": np.array([2])})
            self.assertEqual(RawBankStore(root).load_candidate("c/0.json", first)["x"].item(), 1)

    def test_array_tamper_rejected_before_and_after_cached_load(self):
        for cached in (False, True):
            with self.subTest(cached=cached), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                digest = save_case(root, "case.json", {"x": np.ones((3, 4))})
                store = RawBankStore(root)
                if cached:
                    store.load_case("case.json", digest)
                manifest = json.loads((root / "case.json").read_text())
                array = root / manifest["arrays"]["array_0000"]["path"]
                # Changing the header-free final data byte does not resize a live mmap.
                with array.open("r+b") as handle:
                    handle.seek(-1, 2)
                    value = handle.read(1)
                    handle.seek(-1, 2)
                    handle.write(bytes([value[0] ^ 1]))
                with self.assertRaisesRegex(ValueError, "changed|hash mismatch"):
                    store.load_case("case.json", digest)
                store.close()  # Release Windows mmap before fixture cleanup.

    def test_pickle_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Object arrays"):
                save_candidate(directory, "c.json", {"x": np.array([{}], dtype=object)})
            for relative in ("../c.json", "/c.json", "a/../c.json", "C:/c.json", "a\\c.json"):
                with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "Unsafe"):
                    save_case(directory, relative, {"x": np.ones(1)})


if __name__ == "__main__":
    unittest.main()
