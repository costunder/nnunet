"""CPU UNIT storage integrity regressions; no CT, neural or quality claims."""
from __future__ import annotations

from tests.artifacts import unit_artifact_root
from concurrent.futures import ThreadPoolExecutor
import copy
import gc
import gzip
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import weakref

import numpy as np
import torch

from hiercp_v22 import storage as original
from hiercp_v1x import transition_preparation_storage as optimized


ROOT = Path(__file__).resolve().parents[1]


def canonical(offset=0):
    return dict(format="canonical-full-v22", geometry_contract="UNIT",
        nodes={"UNIT-context": dict(grid=torch.tensor([[0, 0, 0], [1, 1, 1]]),
                                    x=torch.arange(8, dtype=torch.float32).reshape(2, 4) + offset)},
        edges={("UNIT-context", "UNIT-relation", "UNIT-context"): torch.tensor([[0, 1], [1, 0]])},
        counts={"UNIT-context": 2}, footprint_voxels=3, metadata={"units": "UNIT", "values": [1, .5, None]})


def record(source=None, ordinal=0):
    if source is None:
        source = dict(source_local=canonical(), source_patch=torch.arange(24, dtype=torch.float16).reshape(2, 3, 4))
    return dict(**source, target_local=canonical(ordinal + 1),
                target_patch=torch.ones(2, 3, 4, dtype=torch.float16) * (ordinal + 1),
                observation_id="UNIT:" + str(ordinal), ordinal=ordinal)


def assert_nested_equal(test, expected, actual):
    test.assertEqual(type(actual), type(expected))
    if torch.is_tensor(expected):
        test.assertEqual(actual.dtype, expected.dtype)
        test.assertEqual(tuple(actual.shape), tuple(expected.shape))
        test.assertTrue(torch.equal(expected, actual))
    elif isinstance(expected, np.ndarray):
        test.assertEqual(actual.dtype, expected.dtype)
        test.assertTrue(np.array_equal(expected, actual))
    elif isinstance(expected, dict):
        test.assertEqual(list(actual), list(expected))
        for key in expected:
            assert_nested_equal(test, expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        test.assertEqual(len(actual), len(expected))
        for left, right in zip(expected, actual):
            assert_nested_equal(test, left, right)
    else:
        test.assertEqual(expected, actual)


class CheckedDonorStorageUNIT(unittest.TestCase):
    def setUp(self):
        self.root = unit_artifact_root() / ("transition_storage_UNIT_" + uuid.uuid4().hex)
        self.root.mkdir()
        self.writer = optimized.GraphWriter(self.root, minimum_free_bytes=0)
        self.record = record()

    def test_exact_format_reference_digest_bounds_and_full_roundtrip(self):
        expected_raw = original.encode({key: self.record[key] for key in ("source_local", "source_patch")})
        stored = self.writer.write("graphs/UNIT.pt.gz", self.record, "UNIT-source")
        self.assertEqual(set(stored), {"path", "sha256", "bounds", "shared_source"})
        self.assertEqual(stored["shared_source"]["content_sha256"], hashlib.sha256(expected_raw).hexdigest())
        source_file = self.root / stored["shared_source"]["path"]
        self.assertEqual(gzip.decompress(source_file.read_bytes()), expected_raw)
        self.assertEqual(original.sha(source_file), stored["shared_source"]["sha256"])
        self.assertEqual(original.sha(self.root / stored["path"]), stored["sha256"])
        a, b = original._local_bound(self.record["source_local"]), original._local_bound(self.record["target_local"])
        self.assertEqual(stored["bounds"], dict(nodes=a[0] + b[0], edges=a[1] + b[1],
            bytes=a[2] + b[2] + 4 * (self.record["source_patch"].numel() + self.record["target_patch"].numel())))
        loaded = original.load_record(self.root, stored["path"])
        # The historical loader appends restored shared fields at the end.
        self.assertEqual(set(loaded), set(self.record))
        for key in self.record:
            assert_nested_equal(self, self.record[key], loaded[key])

    def test_same_immutable_donor_serializes_once_each_observation_still_serializes(self):
        source = {key: self.record[key] for key in ("source_local", "source_patch")}
        calls = []
        encode = original.encode
        def counted(value):
            calls.append("source" if set(value) == {"source_local", "source_patch"} else "observation")
            return encode(value)
        with patch.object(original, "encode", side_effect=counted):
            results = [self.writer.write(f"graphs/{index}.pt.gz", record(source, index), "UNIT-source")
                       for index in range(5)]
        self.assertEqual(calls.count("source"), 1)
        self.assertEqual(calls.count("observation"), 5)
        self.assertEqual(self.writer.source_encode_calls, 1)
        self.assertEqual(self.writer.source_memo_hits, 4)
        self.assertTrue(all(row["shared_source"] == results[0]["shared_source"] for row in results))
        for index, row in enumerate(results):
            loaded = original.load_record(self.root, row["path"])
            self.assertEqual(loaded["ordinal"], index)
            self.assertTrue(torch.equal(loaded["source_patch"], source["source_patch"]))

    def _refuse_change(self, change):
        before = self.writer.write("graphs/before.pt.gz", self.record, "UNIT-source")
        source_bytes = (self.root / before["shared_source"]["path"]).read_bytes()
        change(self.record)
        with self.assertRaisesRegex(ValueError, "Source tensors changed"):
            self.writer.write("graphs/after.pt.gz", self.record, "UNIT-source")
        self.assertFalse((self.root / "graphs/after.pt.gz").exists())
        self.assertEqual((self.root / before["shared_source"]["path"]).read_bytes(), source_bytes)
        self.assertEqual(self.writer.source_encode_calls, 2)
        self.assertEqual(self.writer.source_memo_hits, 0)

    def test_source_tensor_inplace_mutation_is_refused(self):
        self._refuse_change(lambda value: value["source_patch"].add_(1))

    def test_nested_graph_edge_mutation_is_refused(self):
        self._refuse_change(lambda value: next(iter(value["source_local"]["edges"].values())).fill_(0))

    def test_nested_metadata_mutation_is_refused(self):
        self._refuse_change(lambda value: value["source_local"]["metadata"]["values"].append(19))

    def test_replaced_changed_tensor_is_refused(self):
        self._refuse_change(lambda value: value.__setitem__("source_patch", torch.zeros_like(value["source_patch"])))

    def test_equal_replacement_tensor_reencodes_and_checks_bytes(self):
        self.writer.write("graphs/before.pt.gz", self.record, "UNIT-source")
        previous = self.record["source_patch"]
        self.record["source_patch"] = previous.clone()
        # Historical byte equality decides acceptance, never object equality.
        raw = original.encode({key: self.record[key] for key in ("source_local", "source_patch")})
        expected = hashlib.sha256(raw).hexdigest()
        if expected != self.writer.sources["UNIT-source"]["content_sha256"]:
            with self.assertRaisesRegex(ValueError, "Source tensors changed"):
                self.writer.write("graphs/after.pt.gz", self.record, "UNIT-source")
        else:
            self.writer.write("graphs/after.pt.gz", self.record, "UNIT-source")
        self.assertEqual(self.writer.source_encode_calls, 2)
        self.assertEqual(self.writer.source_memo_hits, 0)

    def test_tensor_stride_offset_storage_changes_reencode_and_refuse(self):
        self._refuse_change(lambda value: value["source_patch"].transpose_(0, 1))

    def test_mutable_numpy_metadata_is_hashed_each_check_and_change_refused(self):
        self.record["source_local"]["metadata"]["numpy"] = np.arange(6, dtype=np.float32)
        self._refuse_change(lambda value: value["source_local"]["metadata"]["numpy"].fill(10))

    def test_weak_references_do_not_keep_donor_tensors_resident(self):
        tensor = self.record["source_patch"]
        reference = weakref.ref(tensor)
        self.writer.write("graphs/before.pt.gz", self.record, "UNIT-source")
        del tensor
        del self.record
        gc.collect()
        self.assertIsNone(reference())

    def test_memo_eviction_only_repeats_serialization_never_omits_data(self):
        writer = optimized.GraphWriter(self.root, minimum_free_bytes=0, source_memo_entries=1)
        first, second = record(), record(ordinal=2)
        writer.write("graphs/a.pt.gz", first, "source-A")
        writer.write("graphs/b.pt.gz", second, "source-B")
        writer.write("graphs/a2.pt.gz", first, "source-A")
        self.assertEqual(writer.source_encode_calls, 3)
        self.assertEqual(len(writer._source_memos), 1)
        self.assertEqual(len(writer.sources), 2)
        self.assertEqual(len(list((self.root / "graphs").glob("*.pt.gz"))), 3)
        self.assertTrue(torch.equal(original.load_record(self.root, "graphs/a2.pt.gz")["source_patch"], first["source_patch"]))

    def test_independent_payloads_parallel_share_one_source_publication(self):
        source = {key: self.record[key] for key in ("source_local", "source_patch")}
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(lambda index: self.writer.write(f"graphs/{index}.pt.gz", record(source, index), "source"), range(8)))
        self.assertEqual(self.writer.source_encode_calls, 1)
        self.assertEqual(self.writer.source_memo_hits, 7)
        self.assertEqual(len(list((self.root / "shared_sources").glob("*.pt.gz"))), 1)
        for index, row in enumerate(rows):
            self.assertEqual(original.load_record(self.root, row["path"])["ordinal"], index)

    def test_disk_reserve_and_existing_output_still_refuse_no_overwrite(self):
        with patch.object(optimized.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(OSError, "reserve"):
                self.writer.write("graphs/none.pt.gz", self.record, "source")
        self.assertFalse((self.root / "graphs").exists())
        stored = self.writer.write("graphs/exists.pt.gz", self.record, "source")
        before = (self.root / stored["path"]).read_bytes()
        with self.assertRaises(FileExistsError):
            self.writer.write("graphs/exists.pt.gz", self.record, "source")
        self.assertEqual((self.root / stored["path"]).read_bytes(), before)

    def test_original_graphwriter_and_optimized_writer_have_exact_reference_and_payload_bytes(self):
        legacy_root = self.root / "legacy"
        legacy_root.mkdir()
        legacy = original.GraphWriter(legacy_root, minimum_free_bytes=0)
        expected = legacy.write("graphs/same.pt.gz", self.record, "source")
        actual = self.writer.write("graphs/same.pt.gz", self.record, "source")
        self.assertEqual(actual, expected)
        self.assertEqual((self.root / actual["path"]).read_bytes(), (legacy_root / expected["path"]).read_bytes())

    def test_concurrent_standard_tensor_mutation_during_serialization_refused(self):
        encode = original.encode
        def mutate(value):
            raw = encode(value)
            if set(value) == {"source_local", "source_patch"}:
                value["source_patch"].add_(1)
            return raw
        with patch.object(original, "encode", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "during serialization"):
                self.writer.write("graphs/no.pt.gz", self.record, "source")
        self.assertFalse((self.root / "graphs/no.pt.gz").exists())
        self.assertFalse((self.root / "shared_sources").exists())


if __name__ == "__main__":
    unittest.main()
