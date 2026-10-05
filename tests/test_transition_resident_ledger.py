"""Mechanical UNIT storage accounting; no CT, learned output or quality claim.

Synthetic array owners here exercise alias/refcount/eviction behavior only.
The production observation population, graph rules and preparation format are
not modified by these tests.
"""
from collections import OrderedDict
from types import SimpleNamespace
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hiercp_v1x import transition_v1_data as data


def provider(limit=10_000):
    dataset = object.__new__(data.NativeObservationDataset)
    actual = data.OriginalInputProvider(dataset, workers=2, resident_bytes=limit,
                                       rss_bytes=max(1_000_000, limit * 2))
    actual._guard = lambda: None  # UNIT owners have no clinical inventory.
    return actual


def interval_oracle(ranges):
    by_domain = {}
    for domain, first, last in ranges:
        by_domain.setdefault(domain, []).append((first, last))
    count = 0
    for intervals in by_domain.values():
        end = None
        for first, last in sorted(intervals):
            count += last - first if end is None or first >= end else max(0, last - end)
            end = last if end is None else max(end, last)
    return count


class IntervalUnionUNIT(unittest.TestCase):
    def test_partial_overlap_duplicate_and_removal_are_exact(self):
        union = data._StorageIntervalUnion()
        left, right = ("cpu", 100, 200), ("cpu", 150, 250)
        union.add(left); union.add(right); union.add(left)
        self.assertEqual(union.total_bytes, 150)
        union.remove(left)
        self.assertEqual(union.total_bytes, 150)
        union.remove(left)
        self.assertEqual(union.total_bytes, 100)
        union.remove(right)
        self.assertEqual(union.total_bytes, 0)
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            union.remove(right)

    def test_nested_bridge_gaps_and_distinct_device_domains(self):
        union = data._StorageIntervalUnion()
        actual = [("cpu", 0, 10), ("cpu", 30, 40), ("cpu", 5, 35),
                  ("cpu", 12, 22), ("cuda:0", 0, 100)]
        for value in actual:
            union.add(value)
        self.assertEqual(union.total_bytes, 140)
        union.remove(actual[2])
        self.assertEqual(union.total_bytes, 130)
        for value in (actual[0], actual[3], actual[4], actual[1]):
            union.remove(value)
        self.assertEqual(union.total_bytes, 0)

    def test_randomized_add_remove_matches_independent_union_oracle(self):
        random_state = random.Random(294814)
        union, owners = data._StorageIntervalUnion(), []
        for _ in range(2500):
            if owners and random_state.random() < .42:
                ordinal = random_state.randrange(len(owners))
                union.remove(owners.pop(ordinal))
            else:
                first = random_state.randrange(400)
                value = (random_state.choice(("cpu", "cuda:0")),
                         first, first + random_state.randrange(1, 70))
                owners.append(value); union.add(value)
            self.assertEqual(union.total_bytes, interval_oracle(owners))
        for value in owners:
            union.remove(value)
        self.assertEqual(union.total_bytes, 0)

    def test_many_nonoverlapping_insertions_use_cached_aggregate(self):
        union = data._StorageIntervalUnion()
        for index in range(4000):
            union.add(("cpu", index * 16, index * 16 + 8))
        with patch.object(data, "_interval_rows", side_effect=AssertionError("global scan")):
            self.assertEqual(union.total_bytes, 32_000)
            # Identical shared backing stores update only their refcount.
            union.add(("cpu", 64, 72)); union.remove(("cpu", 64, 72))
            self.assertEqual(union.total_bytes, 32_000)


class OwnerStorageUNIT(unittest.TestCase):
    def test_numpy_views_negative_stride_and_torch_alias_count_backing_once(self):
        owner = np.arange(100, dtype=np.float32)
        view, reversed_view = owner[20:40], owner[::-1]
        tensor = torch.from_numpy(owner[10:80])
        self.assertEqual(data._bytes((view, reversed_view, tensor)), owner.nbytes)
        self.assertEqual(data._bytes(view), owner.nbytes)
        ledger = data._ResidentStorageLedger()
        ledger.put("raw", owner, "raw")
        ledger.put("record", dict(view=view, tensor=tensor), "record")
        self.assertEqual(ledger.bytes("all"), owner.nbytes)
        ledger.remove("raw")
        self.assertEqual(ledger.bytes("all"), owner.nbytes)
        ledger.remove("record")
        self.assertEqual(ledger.bytes("all"), 0)

    def test_torch_views_numpy_conversion_keep_full_storage_alive(self):
        owner = torch.arange(40, dtype=torch.float32)
        narrow = owner[5:8]
        numpy_view = narrow.numpy()
        self.assertEqual(data._bytes((narrow, numpy_view)), owner.untyped_storage().nbytes())
        ledger = data._ResidentStorageLedger()
        ledger.put("donor", (narrow,), "donor")
        ledger.put("record", dict(image=numpy_view), "record")
        self.assertEqual(ledger.bytes("donor"), 160)
        self.assertEqual(ledger.bytes("record"), 160)
        self.assertEqual(ledger.bytes("canonical"), 160)
        ledger.remove("donor")
        self.assertEqual(ledger.bytes("canonical"), 160)

    def test_prepared_donor_source_masks_share_raw_and_record_storage(self):
        raw = np.arange(80, dtype=np.uint8)
        source = SimpleNamespace(full_mask=raw, patch_mask=raw[10:30], patch_image=raw[30:])
        original = SimpleNamespace(source_footprint=raw[:40], source_patch=torch.from_numpy(raw[10:50]),
                                   canonical_nodes={"UNIT": torch.arange(8)}, canonical_edges={})
        prepared = object.__new__(data.local.PreparedDonor)
        # The production dataclass may be frozen; these UNIT fields are exact
        # storage-owner fixtures, never neural graph records.
        object.__setattr__(prepared, "original", original)
        object.__setattr__(prepared, "source_patch", torch.from_numpy(raw[10:50]))
        ledger = data._ResidentStorageLedger()
        ledger.put("raw", dict(image=raw), "raw")
        ledger.put("source", (source, prepared), "donor")
        ledger.put("record", dict(source_local=original.canonical_nodes,
                                  source_patch=prepared.source_patch), "record")
        self.assertEqual(ledger.bytes("all"), 80 + 8 * 8)
        self.assertEqual(ledger.bytes("canonical"), 80 + 8 * 8)
        ledger.remove("source")
        self.assertEqual(ledger.bytes("all"), 80 + 8 * 8)
        ledger.remove("record")
        self.assertEqual(ledger.bytes("all"), 80)

    def test_accounted_cache_replacement_pop_lru_and_clear(self):
        ledger = data._ResidentStorageLedger()
        shared, separate = torch.ones(12), torch.ones(9)
        cache = data._AccountedCache(ledger, "record", lambda value: value[0])
        cache["first"] = dict(feature=shared), 48
        cache["second"] = dict(feature=shared), 48
        self.assertEqual(ledger.bytes("all"), 48)
        cache.move_to_end("first")
        self.assertEqual(list(cache), ["second", "first"])
        cache["first"] = dict(feature=separate), 36
        self.assertEqual(ledger.bytes("all"), 84)
        key, _ = cache.popitem(last=False)
        self.assertEqual(key, "second")
        self.assertEqual(ledger.bytes("all"), 36)
        self.assertEqual(cache.pop("missing", None), None)
        cache.clear()
        self.assertEqual(ledger.bytes("all"), 0)
        self.assertEqual(ledger._objects, {})


class ProviderResidencyUNIT(unittest.TestCase):
    def test_raw_donor_record_overlap_is_not_double_counted_or_evicted(self):
        actual = provider(limit=500)
        shared = np.arange(100, dtype=np.float32)
        record = dict(feature=torch.from_numpy(shared[10:80]))
        actual._raw = SimpleNamespace(cache=OrderedDict(UNIT=dict(image=shared)))
        actual._donors[("UNIT", 1)] = shared, record, 400
        actual._records["UNIT_record"] = record, 280
        actual._make_room(active_cases=["UNIT"], active_records=[record])
        self.assertEqual(len(actual._records), 1)
        memory = actual._memory_measurement([record])
        self.assertEqual(memory["raw_resident_bytes"], 400)
        self.assertEqual(memory["canonical_resident_bytes"], 400)
        self.assertEqual(memory["total_resident_bytes"], 400)
        self.assertEqual(len(actual._raw.cache), 1)

    def test_active_records_stay_admitted_after_cache_eviction(self):
        actual = provider(limit=100)
        active, inactive = dict(feature=torch.zeros(20)), dict(feature=torch.zeros(20))
        actual._records["active"] = active, 80
        actual._records["inactive"] = inactive, 80
        actual._make_room(active_records=[active])
        self.assertEqual(len(actual._records), 0)
        memory = actual._memory_measurement([active])
        self.assertEqual(memory["canonical_record_resident_bytes"], 80)
        self.assertEqual(memory["total_resident_bytes"], 80)
        self.assertEqual(actual._resident.bytes("all"), 0)
        self.assertEqual(len(active["feature"]), 20)

    def test_oversized_complete_active_record_raises_without_truncating(self):
        actual = provider(limit=40)
        complete = dict(feature=torch.zeros(20))
        actual._records["whole"] = complete, 80
        with self.assertRaisesRegex(MemoryError, "Complete active"):
            actual._make_room(active_records=[complete])
        self.assertEqual(len(complete["feature"]), 20)
        self.assertEqual(actual._resident.bytes("all"), 0)

    def test_reporting_and_room_checks_do_not_walk_accumulated_records(self):
        actual = provider(limit=1_000_000)
        actual._raw = SimpleNamespace(cache={"UNIT": dict(image=np.zeros(7))})
        actual._account_raw_cache()
        for index in range(200):
            record = dict(feature=torch.ones(4))
            actual._records[str(index)] = record, 16
        active = actual._records["199"][0]
        expected = 200 * 16 + 7 * 8
        with patch.object(data, "_storage_footprint", side_effect=AssertionError("repeat graph walk")),\
                patch.object(actual._records, "values", side_effect=AssertionError("all cached records")),\
                patch.object(actual._donors, "values", side_effect=AssertionError("all cached donors")),\
                patch.object(actual._raw.cache, "values", side_effect=AssertionError("all raw volumes")):
            for _ in range(50):
                self.assertEqual(actual._memory_measurement([active])["total_resident_bytes"], expected)
                actual._make_room(active_records=[active])
        self.assertEqual(len(actual._records), 200)


if __name__ == "__main__":
    unittest.main()
