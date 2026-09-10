"""DEBUG: storage deduplication preserves every candidate and donor byte."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from custom_trainers.onlinecp_raw_bank import RawBankStore, save_candidate


class RawBankSharedSourcesDebug(unittest.TestCase):
    def _candidate(self, index):
        return {"source_ct": np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4),
                "source_mask": np.ones((2, 3, 4), dtype=bool),
                "recipient_ct": np.full((1, 2, 3, 4), index, dtype=np.float32),
                "candidate_index": index}

    def test_all_128_candidates_share_only_identical_source_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [(f"c/{i}.json", save_candidate(root, f"c/{i}.json", self._candidate(i)))
                       for i in range(128)]
            self.assertEqual(len(list((root / "raw_sources").glob("*.npy"))), 2)
            self.assertEqual(len(list((root / "c").glob("*.npz"))), 128)
            store = RawBankStore(root)
            try:
                source = None
                for i, (relative, digest) in enumerate(entries):
                    value = store.load_candidate(relative, digest)
                    np.testing.assert_array_equal(value["source_ct"], self._candidate(i)["source_ct"])
                    self.assertTrue(np.all(value["recipient_ct"] == i))
                    self.assertEqual(value["candidate_index"], i)
                    self.assertIsInstance(value["source_ct"], np.memmap)
                    self.assertFalse(value["source_ct"].flags.writeable)
                    if source is not None:
                        self.assertIs(value["source_ct"], source)
                    source = value["source_ct"]
                    manifest = json.loads((root / relative).read_text())
                    shared = {k for k, spec in manifest["arrays"].items() if "path" in spec}
                    self.assertEqual(len(shared), 2)
                    with np.load(root / manifest["archive"]["path"], allow_pickle=False) as archive:
                        self.assertTrue(shared.isdisjoint(archive.files))
                self.assertFalse(store.__getstate__()["_sources"])
                self.assertFalse(store.__getstate__()["_cases"])
            finally:
                store.close()

    def test_nonidentical_source_arrays_are_not_substituted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = self._candidate(0), self._candidate(1)
            second["source_ct"][0, 0, 0, 0] = 99
            a = save_candidate(root, "a.json", first)
            b = save_candidate(root, "b.json", second)
            self.assertEqual(len(list((root / "raw_sources").glob("*.npy"))), 3)
            store = RawBankStore(root)
            try:
                self.assertEqual(store.load_candidate("a.json", a)["source_ct"][0, 0, 0, 0], 0)
                self.assertEqual(store.load_candidate("b.json", b)["source_ct"][0, 0, 0, 0], 99)
            finally:
                store.close()

    def test_modified_shared_source_is_rejected_before_cached_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = save_candidate(root, "c.json", self._candidate(0))
            store = RawBankStore(root)
            try:
                store.load_candidate("c.json", digest)
                manifest = json.loads((root / "c.json").read_text())
                key = manifest["tree"]["source_ct"]["$array"]
                path = root / manifest["arrays"][key]["path"]
                with path.open("r+b") as handle:
                    handle.seek(-1, 2)
                    old = handle.read(1)
                    handle.seek(-1, 2)
                    handle.write(bytes([old[0] ^ 1]))
                with self.assertRaisesRegex(ValueError, "changed"):
                    store.load_candidate("c.json", digest)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
