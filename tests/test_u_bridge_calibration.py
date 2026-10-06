"""UNIT metadata-only calibration ordering; no CT, CUDA or neural execution."""
from copy import deepcopy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import torch

from hiercp_v1x.u_bridge_calibration import CalibrationProvider, SOURCE_COPIES
from hiercp_v1x.u_bridge_data import _resident_size


def sample(case, index=0, *, nodes=2, edges=3, split="train"):
    relation = ("source_context", "near", "source_context")
    local = dict(format="canonical-full-v22", footprint_voxels=5,
        nodes={"source_context": {"x": torch.zeros(nodes, 16)}},
        edges={relation: torch.zeros((2, edges), dtype=torch.int64)},
        counts={"source_context": nodes}, edge_counts={relation: edges})
    return dict(case_id=case, sample_index=index, source_component=1, split=split,
        candidate_centers=torch.tensor([(1, 2, 3)] + [(10+i, 2, 3) for i in range(7)]),
        source_local=local, target_locals=[deepcopy(local) for _ in range(8)])


class UnitBase:
    global_rng_free = True

    def __init__(self, samples, hashes=None):
        hashes = hashes or [None] * len(samples)
        self.rows = [dict(index=i, id=f"{s['case_id']}:{s['sample_index']}",
            case_id=s["case_id"], sample_index=s["sample_index"], source_component=s["source_component"],
            positive_center=tuple(s["candidate_centers"][0].tolist()),
            original_sample_sha256=h, partition=s["split"]) for i, (s,h) in enumerate(zip(samples, hashes))]
        self.batch = Mock(return_value=object())
        self.candidate_keys = Mock(return_value=("P", "U:0"))

    def examples(self, partition):
        name = "val" if partition in ("val", "validation", "inner_val") else "train"
        return deepcopy([row for row in self.rows if row["partition"] == name])


class CalibrationProxyUNIT(unittest.TestCase):
    def test_large_sources_rank_first_without_changing_base_order(self):
        samples = [sample("small"), sample("large", nodes=11, edges=30), sample("val", split="val")]
        base = UnitBase(samples)
        prior = deepcopy(base.rows)
        wrapped = CalibrationProvider(base, samples)
        self.assertEqual([r["index"] for r in wrapped.examples("train")], [1, 0])
        self.assertEqual([r["index"] for r in base.examples("train")], [0, 1])
        self.assertEqual(base.rows, prior)
        self.assertEqual(wrapped.examples("val"), base.examples("val"))
        self.assertEqual(wrapped.examples("inner_train"), wrapped.examples("train"))
        self.assertTrue(wrapped.global_rng_free)

    def test_proxy_receipts_use_sixteen_sources_and_remain_detached(self):
        original = sample("one")
        wrapped = CalibrationProvider(UnitBase([original]), [original])
        row = wrapped.stress_receipts[0]
        self.assertEqual(SOURCE_COPIES, 16)
        self.assertEqual(row["original_source_storage_bytes"], _resident_size(original["source_local"]))
        self.assertEqual(row["stress_ordering_proxy_bytes"],
                         16 * _resident_size(original["source_local"]) + _resident_size(original))
        self.assertEqual(row["source_nodes"], 2)
        self.assertEqual(row["source_edges"], 3)
        self.assertFalse(row["future_cohort_worst_case_verified"])
        row["source_node_type_counts"]["source_context"] = 999
        row["positive_center"][0] = 999
        self.assertEqual(wrapped.stress_receipts[0]["source_node_type_counts"]["source_context"], 2)
        self.assertEqual(wrapped.stress_receipts[0]["positive_center"], [1, 2, 3])
        detached = wrapped.examples("train"); detached[0]["index"] = 999
        self.assertEqual(wrapped.examples("train")[0]["index"], 0)

    def test_batch_and_candidate_keys_forward_exact_arguments_and_return_identity(self):
        original = sample("one")
        base = UnitBase([original]); wrapped = CalibrationProvider(base, [original])
        ids = [0]
        self.assertIs(wrapped.batch(ids, "native", 19, True, full=False), base.batch.return_value)
        base.batch.assert_called_once_with(ids, "native", 19, True, full=False)
        self.assertIs(wrapped.candidate_keys(0, "native", 19, full=True), base.candidate_keys.return_value)
        base.candidate_keys.assert_called_once_with(0, "native", 19, full=True)

    def test_signed_mmap_metadata_reuses_verified_sha_receipt_without_rehash(self):
        original = sample("signed")
        # The loader is intentionally mocked: this UNIT checks regular-file
        # receipt binding and mmap arguments, not PyTorch deserialization.
        path = Path(__file__).resolve()
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        base = UnitBase([original], [source_sha])
        with patch("hiercp_v1x.u_bridge_calibration.torch.load", return_value=original) as load:
            wrapped = CalibrationProvider(base, [dict(path=str(path), sha256=source_sha)])
        load.assert_called_once_with(path, map_location="cpu", weights_only=False, mmap=True)
        self.assertEqual(wrapped.stress_receipts[0]["original_sample_sha256"], source_sha)
        self.assertEqual(wrapped.stress_receipts[0]["original_sample_path"], str(path))
        with self.assertRaises(ValueError):
            CalibrationProvider(base, [dict(path=str(path), sha256="0"*64)])

    def test_missing_duplicate_or_changed_source_identity_is_rejected(self):
        first, second = sample("first"), sample("second")
        base = UnitBase([first, second])
        for inputs in ([first], [first, first], [first, dict(second, source_component=2)],
                       [first, dict(second, candidate_centers=torch.zeros((8,3), dtype=torch.int64))]):
            with self.subTest(inputs=len(inputs)), self.assertRaises(ValueError):
                CalibrationProvider(base, inputs)

    def test_canonical_count_mismatch_and_noncanonical_input_are_rejected(self):
        original = sample("one")
        base = UnitBase([original])
        broken = deepcopy(original); broken["source_local"]["counts"]["source_context"] = 4
        with self.assertRaises(ValueError): CalibrationProvider(base, [broken])
        broken = deepcopy(original); broken["source_local"]["edge_counts"] = {}
        with self.assertRaises(ValueError): CalibrationProvider(base, [broken])
        broken = deepcopy(original); broken["source_local"]["format"] = "UNIT_fake"
        with self.assertRaises(ValueError): CalibrationProvider(base, [broken])
        with self.assertRaises(ValueError): CalibrationProvider(base, [Path("unsigned.pt")])


if __name__ == "__main__":
    unittest.main()
