"""DEBUG bank host-budget/spool plumbing, not clinical or GPU validation.

The model/collate below are explicitly lightweight DEBUG instrumentation.
Canonical tables, Torch serialization/mmap, host accounting, and report
validators are real. Production batch candidates and 128 score positions are
retained; no production configuration is changed by these tests.
"""
from __future__ import annotations

import copy
import gc
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

from hiercp.schema import GraphBuildConfig
from test_full_edge_view_debug import _canonical_fixture
from tools import online_scoring as scoring


DEBUG_CANDIDATE_COUNT = 128


class DebugWeakCanonical(dict):
    """Weak-referenceable mapping preserved by real Torch pickle round trips."""


def debug_config():
    return {
        "scoring_batch_size": "auto",
        "scoring_batch_size_candidates": [1, 2, 4, 8, 16],
        "scoring_batch_calibration_repeats": 3,
        "scoring_batch_max_vram_fraction": 0.85,
        "local_candidate_chunk_size": 4,
        "amp": False,
        "pin_memory": False,
    }


def debug_sample(index):
    source, target = _canonical_fixture(37, 8)
    return DebugWeakCanonical(
        case_id=f"DEBUG_SCORING_{index}",
        source_local=source,
        target_locals=[target] * DEBUG_CANDIDATE_COUNT,
        graph_config=GraphBuildConfig().to_dict(),
        source_component_id=1,
        sample_index=0,
        debug_payload=torch.tensor([index], dtype=torch.float32),
    )


def debug_allocation(available):
    """Explicit resource-simulation fixture, not measured machine telemetry."""
    return {
        "available_memory_bytes": available,
        "host_total_memory_bytes": 10**12,
        "rss_bytes": 1_000_000,
        "cpu_capacity": 8,
        "cpu_affinity_cores": 8,
        "cpu_allocation_cores": 8.0,
        "process_cpu_seconds": 1.0,
        "cgroup": {"locations_resolved": True, "values": {"DEBUG_memory_limit": available}},
    }


class DebugScoringModel:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def score_inference_chunked(self, batch, *, local_chunk_size):
        self.calls.append((batch.sample_count, local_chunk_size))
        if self.fail:
            raise RuntimeError("DEBUG scorer forward failure")
        return [torch.arange(DEBUG_CANDIDATE_COUNT, dtype=torch.float32)
                + 1000 * int(case_id.rsplit("_", 1)[1]) for case_id in batch.case_ids]


def debug_collate(samples):
    # Reproduce the destructive *mapping* operations of materialization. The
    # scorer must supply independent dictionaries without cloning all tensors.
    for sample in samples:
        if "target_locals" in sample:
            sample["local_graphs"] = sample.pop("target_locals")
            sample.pop("source_local")
    return SimpleNamespace(
        case_ids=tuple(sample["case_id"] for sample in samples),
        counts=tuple(len(sample["local_graphs"]) for sample in samples),
        sample_count=len(samples),
    )


class BankScoringResourcesDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)  # Explicit DEBUG CPU profile only.

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_scorer(self, *, spool_directory=None, progress=None, fail=False):
        scorer = scoring.PendingBankScorer(
            DebugScoringModel(fail=fail), torch.device("cpu"), debug_config(),
            DEBUG_CANDIDATE_COUNT, spool_directory=spool_directory, progress=progress,
        )
        scorer.collate = debug_collate
        return scorer

    def assert_nested_equal(self, actual, expected):
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected):
                self.assert_nested_equal(value, reference)
        else:
            self.assertEqual(actual, expected)

    def assert_score_rows(self, values, indices):
        self.assertEqual(len(values), len(indices))
        for value, index in zip(values, indices):
            np.testing.assert_array_equal(
                value, np.arange(DEBUG_CANDIDATE_COUNT, dtype=np.float32) + 1000 * index,
            )

    def test_infer_preserves_canonical_keys_tensors_and_all_rng_states(self):
        samples = [debug_sample(3), debug_sample(7)]
        saved = copy.deepcopy(samples)
        with self.make_scorer() as scorer:
            rng_before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
            first = scorer._infer(samples)
            second = scorer._infer(samples)
            rng_after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        self.assert_nested_equal(samples, saved)
        self.assert_nested_equal(rng_after, rng_before)
        self.assert_score_rows(first, [3, 7])
        self.assert_score_rows(second, [3, 7])
        self.assertTrue(all("local_graphs" not in sample for sample in samples))

    def test_spool_submit_does_not_retain_the_raw_canonical(self):
        with tempfile.TemporaryDirectory(prefix="debug_bank_spool_parent_") as root:
            sibling = Path(root) / "existing_sibling.txt"
            sibling.touch()
            published = []
            with self.make_scorer(spool_directory=root) as scorer:
                scorer.selected_batch_size = 2  # DEBUG execution-state fixture.
                sample = debug_sample(4)
                raw_reference = weakref.ref(sample)
                tensor_reference = weakref.ref(sample["debug_payload"])
                scorer.submit([sample], lambda values: published.extend(values))
                del sample
                gc.collect()
                self.assertIsNone(raw_reference())
                self.assertIsNone(tensor_reference())
                self.assertTrue(list(Path(root).rglob("*.pt")))
                scorer.flush()
                self.assertEqual(scorer.selected_batch_size, 2)
            self.assert_score_rows(published, [4])
            self.assertEqual(list(Path(root).iterdir()), [sibling])

    def test_real_mmap_spool_loads_never_retain_more_than_the_physical_batch(self):
        with tempfile.TemporaryDirectory(prefix="debug_bank_live_loads_") as root:
            published = []
            live_tensors = []
            peak_live = 0
            load_calls = 0
            real_load = scoring.torch_load_compat

            def observed_load(*args, **kwargs):
                nonlocal peak_live, load_calls
                loaded = real_load(*args, **kwargs)
                load_calls += 1
                self.assertTrue(kwargs.get("mmap"))
                live_tensors.append(weakref.ref(loaded["debug_payload"]))
                peak_live = max(peak_live, sum(reference() is not None for reference in live_tensors))
                return loaded

            with self.make_scorer(spool_directory=root) as scorer:
                scorer.selected_batch_size = 2
                samples = [debug_sample(index) for index in range(5)]
                scorer.submit(samples, lambda values: published.extend(values))
                del samples
                with patch.object(scoring, "torch_load_compat", side_effect=observed_load):
                    scorer.flush()
                self.assertEqual(scorer.actual_batches, [2, 2, 1])
                self.assertEqual(scorer.selected_batch_size, 2)
                self.assertEqual([count for count, _ in scorer.model.calls], [2, 2, 1])
            gc.collect()
            self.assertEqual(load_calls, 5)
            self.assertLessEqual(peak_live, 2)
            self.assertTrue(all(reference() is None for reference in live_tensors))
            self.assert_score_rows(published, range(5))
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_host_budget_arithmetic_uses_fresh_allocation_not_host_total(self):
        rows = [dict(case_id="DEBUG_A", canonical_bytes=100, input_bytes_upper_bound=1000),
                dict(case_id="DEBUG_B", canonical_bytes=200, input_bytes_upper_bound=2000)]
        states = [debug_allocation(20_000), debug_allocation(100)]
        with patch.object(scoring, "snapshot", side_effect=states) as capture:
            roomy = scoring._bank_host_budget(rows, pin_memory=True, load_canonical=True)
            limited = scoring._bank_host_budget(rows, pin_memory=True, load_canonical=True)
        self.assertEqual(capture.call_count, 2)
        for result, state in zip((roomy, limited), states):
            self.assertEqual(result["format"], "hiercp_bank_scoring_host_v1")
            self.assertEqual(result["source_count"], 2)
            self.assertEqual(result["input_bytes_upper_bound"], 3000)
            self.assertEqual(result["canonical_load_bytes"], 300)
            self.assertEqual(result["input_copies_accounted"], 4)
            self.assertEqual(result["estimated_input_bytes"], 12300)
            self.assertEqual(result["permitted_input_bytes"], int(state["available_memory_bytes"] * 0.8))
            self.assertEqual(result["available_allocation_bytes"], state["available_memory_bytes"])
            self.assertEqual(result["allocation"], state)
            self.assertFalse(result["graph_or_data_reduction"])
        self.assertTrue(roomy["accepted"])
        self.assertFalse(limited["accepted"])
        with patch.object(scoring, "snapshot", return_value=debug_allocation(20_000)):
            resident = scoring._bank_host_budget(rows, pin_memory=False, load_canonical=False)
        self.assertEqual(resident["canonical_load_bytes"], 0)
        self.assertEqual(resident["estimated_input_bytes"], 9000)

    def test_all_host_rejected_candidates_are_reported_without_inference(self):
        with self.make_scorer() as scorer:
            scorer.submit([debug_sample(index) for index in range(16)], lambda _: None)
            with patch.object(scoring, "snapshot", return_value=debug_allocation(0)):
                with self.assertRaises(RuntimeError):
                    scorer.flush()
            self.assertEqual([trial["batch_size"] for trial in scorer.trials], [1, 2, 4, 8, 16])
            self.assertTrue(all(trial["status"] == "host_input_budget_exceeded" for trial in scorer.trials))
            self.assertTrue(all(trial["host_budgets"] for trial in scorer.trials))
            self.assertEqual(scorer.model.calls, [])
            self.assertIsNone(scorer.selected_batch_size)

    def test_selected_physical_batch_cannot_silently_shrink_after_host_pressure(self):
        with self.make_scorer() as scorer:
            scorer.selected_batch_size = 4
            scorer.submit([debug_sample(index) for index in range(4)], lambda _: None)
            with patch.object(scoring, "snapshot", return_value=debug_allocation(0)):
                with self.assertRaises(RuntimeError):
                    scorer.flush()
            self.assertEqual(scorer.selected_batch_size, 4)
            self.assertEqual(scorer.model.calls, [])
            self.assertEqual(scorer.scored_samples, 0)

    def test_flush_ready_uses_measured_selection_not_maximum_trial_candidate(self):
        with self.make_scorer() as scorer:
            scorer.selected_batch_size = 2
            scorer.submit([debug_sample(0)], lambda _: None)
            with patch.object(scorer, "flush") as flush:
                scorer.flush_ready()
                flush.assert_not_called()
                scorer.submit([debug_sample(1)], lambda _: None)
                scorer.flush_ready()
                flush.assert_called_once_with()

    def test_ordered_group_publication_is_preserved_across_physical_batches(self):
        published = []
        with tempfile.TemporaryDirectory(prefix="debug_bank_groups_") as root:
            with self.make_scorer(spool_directory=root) as scorer:
                scorer.selected_batch_size = 2
                scorer.submit([debug_sample(5), debug_sample(1), debug_sample(8)],
                              lambda values: published.append(("first", values)))
                scorer.submit([debug_sample(2)], lambda values: published.append(("second", values)))
                scorer.submit([debug_sample(7), debug_sample(4)],
                              lambda values: published.append(("third", values)))
                scorer.flush()
                self.assertEqual(scorer.actual_batches, [2, 2, 2])
            self.assertEqual([name for name, _ in published], ["first", "second", "third"])
            for (_, rows), indices in zip(published, ([5, 1, 8], [2], [7, 4])):
                self.assert_score_rows(rows, indices)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_context_cleans_owned_spool_on_forward_failure_without_publishing(self):
        with tempfile.TemporaryDirectory(prefix="debug_bank_failure_") as root:
            sibling = Path(root) / "existing_sibling.txt"
            sibling.touch()
            published = []
            with self.assertRaisesRegex(RuntimeError, "DEBUG scorer forward failure"):
                with self.make_scorer(spool_directory=root, fail=True) as scorer:
                    scorer.selected_batch_size = 2
                    scorer.submit([debug_sample(2), debug_sample(5)], lambda values: published.extend(values))
                    scorer.flush()
            self.assertEqual(published, [])
            self.assertEqual(list(Path(root).iterdir()), [sibling])

    def test_real_timed_calibration_covers_every_candidate_and_validates_report_arithmetic(self):
        events = []
        published = []
        with self.make_scorer(progress=lambda **fields: events.append(fields)) as scorer:
            scorer.submit([debug_sample(index) for index in range(16)], lambda values: published.extend(values))
            scorer.flush()
            report = scorer.report()
        self.assert_score_rows(published, range(16))
        self.assertEqual([trial["batch_size"] for trial in report["calibration_trials"]], [1, 2, 4, 8, 16])
        self.assertEqual(report["candidate_count"], DEBUG_CANDIDATE_COUNT)
        self.assertEqual(report["calibration_sample_count"], 16)
        self.assertEqual(report["scored_samples"], 16)
        self.assertEqual(report["scored_candidates"], 16 * DEBUG_CANDIDATE_COUNT)
        self.assertEqual(sum(report["actual_batches"]), 16)
        self.assertGreaterEqual(len(events), 5)
        self.assertTrue(all(isinstance(event["stage"], str) and isinstance(event["event"], str) for event in events))
        safe = [trial for trial in report["calibration_trials"] if trial["status"] == "safe"]
        self.assertTrue(safe)
        for trial in safe:
            self.assertEqual(len(trial["seconds"]), 3)
            self.assertEqual(len(trial["host_measurements"]), 3)
            self.assertTrue(all(value["status"] == "complete" for value in trial["host_measurements"]))
            self.assertAlmostEqual(trial["samples_per_second"], 16 / float(np.median(trial["seconds"])))
        scoring.validate_scoring_report(report, debug_config())
        wrong_count = copy.deepcopy(report)
        wrong_count["scored_candidates"] += 1
        with self.assertRaises(ValueError):
            scoring.validate_scoring_report(wrong_count, debug_config())
        wrong_rate = copy.deepcopy(report)
        next(trial for trial in wrong_rate["calibration_trials"] if trial["status"] == "safe")["samples_per_second"] *= 1.25
        with self.assertRaises(ValueError):
            scoring.validate_scoring_report(wrong_rate, debug_config())
        failed_measurement = copy.deepcopy(report)
        next(trial for trial in failed_measurement["calibration_trials"] if trial["status"] == "safe")["host_measurements"][0]["status"] = "failed"
        with self.assertRaises(ValueError):
            scoring.validate_scoring_report(failed_measurement, debug_config())


if __name__ == "__main__":
    unittest.main()
