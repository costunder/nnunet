"""Explicit UNIT concurrency/storage fixtures, never CT or accuracy evidence.

Seven fabricated metadata rows are used only to check that complete work is
parallel, receipts stay signed/ordered, and failures remain visible.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch
import uuid
import weakref

from hiercp_v1x import transition_v1_data as data
from tests.test_transition_v1_preparation import UnitProvider, unit_dataset, unit_payload


ROOT = Path(__file__).resolve().parents[1]


class ThreadedStorageUNIT:
    """Only signed synthetic storage bytes, no fake canonical model output."""
    instances = []
    force_reverse_finish = False

    def __init__(self, root, minimum_free_bytes):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.source_encode_calls = self.source_memo_hits = 0
        self.finished = []
        self.odd_finished = {first: threading.Event() for first in (0, 2, 4)}
        type(self).instances.append(self)

    def write(self, relative, record, source_key):
        ordinal = record["UNIT_ordinal"]
        source = self.root / "shared_sources" / "UNIT_shared_donor_metadata.pt"
        with self.lock:
            if source.exists():
                self.source_memo_hits += 1
            else:
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(b"UNIT_source_metadata_only")
                self.source_encode_calls += 1
        if self.force_reverse_finish and ordinal in self.odd_finished:
            if not self.odd_finished[ordinal].wait(timeout=5):
                raise AssertionError("Post-build storage was serialized")
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("UNIT_storage_only_" + str(ordinal)).encode())
        with self.lock:
            self.finished.append(ordinal)
        if self.force_reverse_finish and ordinal - 1 in self.odd_finished:
            self.odd_finished[ordinal - 1].set()
        return dict(path=relative, sha256=data._sha(path),
                    bounds=dict(nodes=1000 + ordinal, edges=2000 + ordinal, bytes=4096 + 128 * ordinal),
                    shared_source=dict(path=str(source.relative_to(self.root)).replace("\\", "/"),
                                       sha256=data._sha(source), content_sha256="b" * 64))


class GuardedProviderUNIT(UnitProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thread_state = threading.local()
        self.guard_states = []
        self.guard_lock = threading.Lock()

    def _guard(self):
        super()._guard()
        views = getattr(self.thread_state, "views", None)
        if views is not None:
            live = tuple(reference() is not None for reference in views)
            with self.guard_lock:
                self.guard_states.append((self.thread_state.ordinal, live))


def make_fixture(name):
    root = ROOT / "work" / ("transition_D_parallel_UNIT_" + name + "_" + uuid.uuid4().hex)
    root.mkdir()
    dataset = unit_dataset(root)
    provider = GuardedProviderUNIT(dataset, workers=2, resident_bytes=1024**3, rss_bytes=2 * 1024**3)
    return root, dataset, provider, root / "canonical"


class ParallelPreparationUNIT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.dataset, cls.provider, cls.output = make_fixture("complete")
        cls.barrier = threading.Barrier(2)
        cls.materialize_threads = set()
        cls.thread_lock = threading.Lock()
        cls.published_ordinals = []
        publish = data._publish_new_json

        def materialize(record, *, epoch):
            ordinal = record["UNIT_ordinal"]
            if ordinal < 6:
                cls.barrier.wait(timeout=5)  # A serial materialize implementation fails.
            payload = unit_payload(record, epoch=epoch)
            cls.provider.thread_state.ordinal = ordinal
            cls.provider.thread_state.views = tuple(weakref.ref(graph) for graph in payload[0])
            with cls.thread_lock:
                cls.materialize_threads.add(threading.get_ident())
            return payload

        def ordered_publish(path, value):
            if Path(path).parent.name == "completed":
                cls.published_ordinals.append(int(Path(path).stem))
            return publish(path, value)

        ThreadedStorageUNIT.force_reverse_finish = True
        with patch("hiercp_v1x.transition_preparation_storage.GraphWriter", ThreadedStorageUNIT),\
                patch.object(data.local, "materialize_pair", side_effect=materialize),\
                patch.object(data, "_publish_new_json", side_effect=ordered_publish):
            cls.path = cls.provider.preflight(cls.output, minimum_free_bytes=1)
        cls.writer = ThreadedStorageUNIT.instances[-1]
        cls.index = json.loads(cls.path.read_text(encoding="utf8"))
        cls.events = [json.loads(line) for name in cls.index["preparation_metric_files"]
                      for line in (cls.output / name).read_text(encoding="utf8").splitlines()]
        ThreadedStorageUNIT.force_reverse_finish = False

    def test_two_view_materialization_and_storage_are_actually_concurrent(self):
        self.assertGreaterEqual(len(self.materialize_threads), 2)
        for first in (0, 2, 4):
            self.assertLess(self.writer.finished.index(first + 1), self.writer.finished.index(first))
        self.assertEqual(sorted(self.writer.finished), list(range(7)))

    def test_signed_receipts_and_index_keep_original_order_despite_reverse_completion(self):
        self.assertEqual(self.published_ordinals, list(range(7)))
        self.assertEqual([row["id"] for row in self.index["records"]],
                         [row["id"] for row in self.dataset.rows])
        completed = [event for event in self.events if event["event"] == "observation_completed"]
        self.assertEqual([event["observation_id"] for event in completed],
                         [row["id"] for row in self.dataset.rows])
        self.assertEqual(self.index["prepared_assignment_sha256"], data.assignment_digest(self.dataset.rows))
        self.assertEqual(self.index["views_per_observation"], 2)
        self.assertEqual(self.index["skipped_observations"], 0)
        self.assertFalse(self.index["hidden_subset"])

    def test_shared_source_disk_bytes_are_counted_once_in_parallel_stage(self):
        completed = [event for event in self.events if event["event"] == "observation_completed"]
        self.assertEqual(self.writer.source_encode_calls, 1)
        self.assertEqual(self.writer.source_memo_hits, 6)
        actual_source_size = len(b"UNIT_source_metadata_only")
        self.assertEqual(sum(event["newly_written_shared_source_bytes"] for event in completed), actual_source_size)
        self.assertEqual(sum(event["newly_written_shared_source_bytes"] > 0 for event in completed), 1)
        disk = sum(path.stat().st_size for path in (self.output / "segments").rglob("*") if path.is_file())
        self.assertEqual(self.index["disk_cumulative_canonical_bytes"], disk)
        self.assertEqual(self.index["disk_written_canonical_bytes"], disk)

    def test_worker_rss_guard_runs_while_both_sampled_views_are_alive(self):
        for ordinal in range(7):
            states = [live for seen, live in self.provider.guard_states if seen == ordinal]
            self.assertIn((True, True), states, (ordinal, states))
            self.assertIn((False, False), states, (ordinal, states))
        post_build = [event for event in self.events if event["event"] == "chunk_post_build_completed"]
        self.assertEqual(len(post_build), 4)
        for event in post_build:
            self.assertGreaterEqual(event["post_build_wall_seconds"], 0)
            self.assertGreater(event["total_resident_bytes"], 0)

    def test_worker_failure_preserves_completed_receipt_and_refuses_complete_index(self):
        root, dataset, provider, output = make_fixture("failure")
        barrier = threading.Barrier(2)

        def fail_one(record, *, epoch):
            ordinal = record["UNIT_ordinal"]
            barrier.wait(timeout=5)
            if ordinal == 1:
                raise RuntimeError("Explicit UNIT worker failure; no successful sample substitution")
            return unit_payload(record, epoch=epoch)

        with patch("hiercp_v1x.transition_preparation_storage.GraphWriter", ThreadedStorageUNIT),\
                patch.object(data.local, "materialize_pair", side_effect=fail_one):
            with self.assertRaisesRegex(RuntimeError, "Explicit UNIT worker failure"):
                provider.preflight(output, minimum_free_bytes=1)
        self.assertFalse((output / "index.json").exists())
        completed = sorted((output / "completed").glob("*.json"))
        self.assertEqual([path.stem for path in completed], ["000000"])
        preserved_hash = data._sha(completed[0])
        first_row = json.loads(completed[0].read_text(encoding="utf8"))
        self.assertEqual(first_row["id"], dataset.rows[0]["id"])
        data._verify_stored_files(output, first_row)
        data._validate_measurement(first_row)
        self.assertTrue((output / "prepare_request.json").exists())
        # Resume checks the preserved receipt, writes missing rows to a fresh
        # owned segment, and never treats the failed observation as completed.
        with patch("hiercp_v1x.transition_preparation_storage.GraphWriter", ThreadedStorageUNIT),\
                patch.object(data.local, "materialize_pair", side_effect=unit_payload):
            result = provider.preflight(output, minimum_free_bytes=1)
        index = json.loads(result.read_text(encoding="utf8"))
        self.assertEqual(index["resumed_exact_completed_observations"], 1)
        self.assertEqual(index["prepared_observations"], 7)
        self.assertEqual(index["skipped_observations"], 0)
        self.assertEqual(data._sha(completed[0]), preserved_hash)


if __name__ == "__main__":
    unittest.main()
