"""DEBUG storage faults and resource accounting; no clinical/GPU training."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from test_feedback_gnn_debug import DebugGraphProvider, debug_runtime, debug_records, debug_progress
from hiercp.feedback_resources import (allocation_guard, calibration_batches, content_digest,
                                      missing_optimizer_bytes, optimizer_probe_gap, sample_inventory, validate_inventory,
                                      worst_batch_bytes)
from hiercp.feedback_storage import FeedbackGraphStore


class FeedbackStorageResourcesDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, root):
        provider = DebugGraphProvider((2,))
        entry = next(iter(provider.entry_cases))
        sample = provider.get(entry)
        store = FeedbackGraphStore(root, minimum_free_bytes=0)
        return store, entry, sample, {"DEBUG": "complete_two_candidate_fixture"}

    def publish(self, store, entry, sample, binding):
        return store.get_or_build(entry, binding, count=2, case_id=sample["case_id"], build=lambda: sample)

    def test_debug_transaction_preserves_exact_tensors_and_warm_witness(self):
        with tempfile.TemporaryDirectory(prefix="feedback_storage_DEBUG_") as temporary:
            store, entry, sample, binding = self.fixture(temporary)
            empty = torch.empty((0, 5), dtype=torch.long).as_strided((0, 5), (0, 0))
            singleton = torch.tensor([7], dtype=torch.long).as_strided((1,), (0,))
            self.assertEqual(content_digest(empty), content_digest(torch.empty((0, 5), dtype=torch.long)))
            self.assertEqual(content_digest(singleton), content_digest(torch.tensor([7], dtype=torch.long)))
            sample["DEBUG_empty_zero_stride"] = empty
            sample["DEBUG_singleton_zero_stride"] = singleton
            actual = self.publish(store, entry, sample, binding)
            self.assertEqual(content_digest(actual), content_digest(sample))
            with mock.patch("hiercp.feedback_storage._sha", side_effect=AssertionError("warm content rehashed")):
                warm = self.publish(store, entry, sample, binding)
                inventory = store.read(entry, binding, count=2, case_id=sample["case_id"], with_sample=False)
            self.assertEqual(content_digest(warm), content_digest(sample))
            self.assertEqual(inventory, sample_inventory(sample, entry_id=entry, candidate_count=2))
            self.assertEqual(len(list(Path(temporary).glob("*.json"))), 1)

    def test_debug_failed_commit_preserves_generation_and_retry_does_not_overwrite_it(self):
        with tempfile.TemporaryDirectory(prefix="feedback_storage_DEBUG_") as temporary:
            store, entry, sample, binding = self.fixture(temporary)
            with mock.patch("hiercp.feedback_storage.os.link", side_effect=OSError("DEBUG publication interruption")):
                with self.assertRaisesRegex(OSError, "publication interruption"):
                    self.publish(store, entry, sample, binding)
            self.assertFalse(store.committed(entry, binding))
            original = {p: p.read_bytes() for p in Path(temporary).rglob("*") if p.is_file()}
            actual = self.publish(store, entry, sample, binding)
            self.assertEqual(content_digest(actual), content_digest(sample))
            for path, value in original.items():
                self.assertEqual(path.read_bytes(), value)
            self.assertEqual(len(list((Path(temporary) / "generations").iterdir())), 2)

    def test_debug_corrupt_committed_payload_is_fatal_not_rebuilt(self):
        with tempfile.TemporaryDirectory(prefix="feedback_storage_DEBUG_") as temporary:
            store, entry, sample, binding = self.fixture(temporary)
            self.publish(store, entry, sample, binding)
            manifest = json.loads(next(Path(temporary).glob("*.json")).read_text())
            payload = Path(temporary) / manifest["generation"]
            with payload.open("ab") as handle:
                handle.write(b"DEBUG_corruption")
            with self.assertRaisesRegex(ValueError, "changed immutable"):
                self.publish(store, entry, sample, binding)

    def test_debug_disk_admission_honors_explicit_reserve_before_payload_write(self):
        with tempfile.TemporaryDirectory(prefix="feedback_storage_DEBUG_") as temporary:
            store, entry, sample, binding = self.fixture(temporary)
            store.minimum_free_bytes = 80 * 1024**3
            with mock.patch("hiercp.feedback_storage.shutil.disk_usage", return_value=SimpleNamespace(free=80 * 1024**3)):
                with self.assertRaisesRegex(RuntimeError, "disk preflight"):
                    self.publish(store, entry, sample, binding)
            self.assertEqual(list(Path(temporary).rglob("payload.pt")), [])
            self.assertFalse(store.committed(entry, binding))

    def test_debug_entire_inventory_largest_mixed_and_repeated_batch_accounting(self):
        provider = DebugGraphProvider((2, 3, 4))
        rows = {entry: sample_inventory(provider.get(entry), entry_id=entry, candidate_count=count)
                for entry, count in provider.counts.items()}
        entries = list(rows)
        validate_inventory(rows, entries)
        rows[entries[-1]]["input_bytes_upper_bound"] = 10**9
        patterns = calibration_batches(rows, entries, 2)
        self.assertTrue(all(pattern[0] == entries[-1] for pattern in patterns))
        self.assertEqual(len(patterns), 2)
        sizes = sorted([row["input_bytes_upper_bound"] for row in rows.values()], reverse=True)
        self.assertEqual(worst_batch_bytes(rows, entries, 5), sum(sizes) + sum(sizes[:2]))
        with self.assertRaisesRegex(ValueError, "every distinct"):
            validate_inventory({entries[0]: rows[entries[0]]}, entries)
        state = {"available_memory_bytes": 100 * 1024**3, "cpu_capacity": 8}
        with mock.patch("hiercp.training_resources.snapshot", return_value=state):
            single = allocation_guard(rows, entries, 2, workers=0, prefetch_factor=2, pin_memory=False)
            parallel = allocation_guard(rows, entries, 2, workers=4, prefetch_factor=3, pin_memory=True)
        self.assertGreater(parallel["required_bytes"], single["required_bytes"])
        self.assertEqual(parallel["allocation"], state)

    def test_debug_historical_adam_moments_missing_from_probe_are_not_ignored(self):
        first, second = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(5))
        actual = torch.optim.AdamW([first, second])
        probe = torch.optim.AdamW([first, second], lr=0.)
        (first.sum() + second.sum()).backward()
        actual.step()
        actual.zero_grad(set_to_none=True)
        first.sum().backward()
        probe.step()
        self.assertEqual(missing_optimizer_bytes(actual, probe), 5 * 4 * 2)

    def test_debug_pure_state_prevalidation_rejects_late_moment_before_any_mutation(self):
        runtime = debug_runtime()
        with mock.patch.object(runtime, "_calibrate", return_value={"physical_batch_size": 2}):
            runtime.update(debug_records(runtime.provider), 0, nnunet_progress=debug_progress(0))
        saved = runtime.state_dict()
        before = content_digest(saved)
        rng = torch.get_rng_state().clone()
        runtime.validate_state_dict(saved)
        broken = copy.deepcopy(saved)
        last = list(broken["optimizer"]["state"])[-1]
        broken["optimizer"]["state"][last]["exp_avg"] = torch.tensor([float("nan")])
        with self.assertRaisesRegex(ValueError, "moment"):
            runtime.load_state_dict(broken)
        self.assertEqual(content_digest(runtime.state_dict()), before)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_debug_pure_prevalidation_rejects_model_rng_and_old_semantics_without_mutation(self):
        from hiercp.feedback import _capture_rng
        runtime = debug_runtime()
        saved = runtime.state_dict()
        before, rng = content_digest(saved), content_digest(_capture_rng())
        bad_model = copy.deepcopy(saved)
        key = next(name for name, value in bad_model["model"].items() if value.is_floating_point())
        bad_model["model"][key].fill_(float("nan"))
        bad_rng = copy.deepcopy(saved)
        bad_rng["rng"] = {}
        old_semantics = copy.deepcopy(saved)
        old_semantics["architecture_version"] = "hiercp_observed_difficulty_v1"
        reordered = copy.deepcopy(saved)
        identities = reordered["optimizer"]["param_groups"][0]["params"]
        identities[0], identities[1] = identities[1], identities[0]
        for label, broken in (("model", bad_model), ("rng", bad_rng), ("semantics", old_semantics),
                              ("optimizer_parameter_order", reordered)):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    runtime.validate_state_dict(broken)
                self.assertEqual(content_digest(runtime.state_dict()), before)
                self.assertEqual(content_digest(_capture_rng()), rng)

    def test_debug_cold_inventory_preparation_does_not_advance_stream_rng(self):
        import random
        import numpy as np
        from hiercp.feedback import _capture_rng
        runtime = debug_runtime()
        original = runtime.provider.get
        def measured_get(entry):
            random.random()
            np.random.random()
            torch.rand(3)
            return original(entry)
        before = content_digest(_capture_rng())
        with mock.patch.object(runtime.provider, "get", side_effect=measured_get):
            runtime._ensure_inventory()
        self.assertEqual(content_digest(_capture_rng()), before)
        self.assertEqual(set(runtime.inventory), set(runtime.provider.entry_cases))

    def test_debug_probe_gap_includes_future_moments_and_unprobed_gradient_workspace(self):
        first = torch.nn.Parameter(torch.ones(3))
        second = torch.nn.Parameter(torch.ones(5))
        third = torch.nn.Parameter(torch.ones(7))
        actual = torch.optim.AdamW([first, second, third])
        probe = torch.optim.AdamW([first, second, third], lr=0.)
        (first.sum() + second.sum()).backward()
        actual.step()
        actual.zero_grad(set_to_none=True)
        first.sum().backward()
        probe.step()
        before = content_digest(actual.state_dict())
        gap = optimizer_probe_gap(actual, probe, [first, second, third])
        self.assertEqual(gap["historical_moment_bytes"], 5 * 4 * 2)
        self.assertEqual(gap["potential_new_moment_bytes"], 7 * 4 * 2)
        self.assertEqual(gap["unprobed_gradient_and_step_workspace_bytes"], (5 + 7) * 4 * 3)
        self.assertEqual(gap["additional_bytes"], (5 + 7) * 4 * 5)
        self.assertEqual(content_digest(actual.state_dict()), before)

    def test_debug_probe_exception_restores_model_mode_rng_and_actual_optimizer(self):
        runtime = debug_runtime()
        runtime._ensure_inventory()
        runtime.model.eval()
        before = content_digest(runtime.state_dict())
        rng = torch.get_rng_state().clone()
        grouped = {entry: [row for row in debug_records(runtime.provider) if row["entry_id"] == entry]
                   for entry in runtime.provider.entry_cases}
        with mock.patch.object(runtime, "_forward_loss", side_effect=RuntimeError("DEBUG probe failure")):
            with self.assertRaisesRegex(RuntimeError, "probe failure"):
                runtime._calibrate(sorted(grouped), grouped)
        self.assertFalse(runtime.model.training)
        self.assertEqual(content_digest(runtime.state_dict()), before)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_debug_real_calibration_preserves_initialized_optimizer_and_entire_pool(self):
        import contextlib
        import io
        from hiercp.feedback import _capture_rng
        runtime = debug_runtime()
        records = debug_records(runtime.provider)
        with mock.patch.object(runtime, "_calibrate", return_value={"physical_batch_size": 2}):
            runtime.update(records, 0, nnunet_progress=debug_progress(0))
        runtime.model.eval()
        before, rng = content_digest(runtime.state_dict()), content_digest(_capture_rng())
        grouped = {entry: [row for row in records if row["entry_id"] == entry]
                   for entry in runtime.provider.entry_cases}
        with contextlib.redirect_stdout(io.StringIO()):
            report = runtime._calibrate(sorted(grouped), grouped)
        self.assertEqual(report["all_bank_entries"], len(runtime.provider.entry_cases))
        self.assertEqual(report["candidate_counts"], sorted(set(runtime.provider.counts.values())))
        self.assertTrue(all(row["candidate_pools_unchanged"] for row in report["trials"]))
        self.assertTrue(any(row["status"] == "safe" for row in report["trials"]))
        self.assertEqual(content_digest(runtime.state_dict()), before)
        self.assertEqual(content_digest(_capture_rng()), rng)
        self.assertFalse(runtime.model.training)

    def test_debug_new_region_payloads_are_fsynced_before_pointer_publication(self):
        import os
        from hiercp.feedback_patient import _new_regions
        # Boundary-only DEBUG native writer double; real native region/content
        # parity is exercised separately by test_feedback_graph_binding_debug.
        with tempfile.TemporaryDirectory(prefix="feedback_region_commit_DEBUG_") as temporary:
            root = Path(temporary)
            expected = {"DEBUG": "owned_native_generation_publication"}
            case = SimpleNamespace(paths=SimpleNamespace(case_id="debug_patient"))
            regions, payloads, synced = object(), {}, set()
            original_fsync, original_link = os.fsync, os.link
            def write_native(case, **kwargs):
                destination = Path(kwargs["cache_dir"]) / case.paths.case_id
                destination.mkdir()
                payloads[destination / "regions.npz"] = b"DEBUG native payload bytes"
                payloads[destination / "metadata.json"] = b'{"DEBUG": "metadata"}'
                for path, body in payloads.items():
                    path.write_bytes(body)
                return regions
            def fsync(descriptor):
                stat = os.fstat(descriptor)
                synced.add((stat.st_dev, stat.st_ino))
                return original_fsync(descriptor)
            def publish(source, destination):
                self.assertEqual(len(payloads), 2)
                for path, body in payloads.items():
                    stat = path.stat()
                    self.assertIn((stat.st_dev, stat.st_ino), synced)
                    self.assertEqual(path.read_bytes(), body)
                return original_link(source, destination)
            with mock.patch("hiercp.feedback_patient.load_or_build_patient_regions", side_effect=write_native), \
                 mock.patch("hiercp.feedback_patient.load_patient_regions", return_value=(regions, expected)), \
                 mock.patch("hiercp.feedback_patient.region_alias_compatible", return_value=True), \
                 mock.patch("hiercp.feedback_patient.os.fsync", side_effect=fsync), \
                 mock.patch("hiercp.feedback_patient.os.link", side_effect=publish):
                self.assertIs(_new_regions(case, expected, liver_label=1, tumor_label=2,
                                            config=None, ct_clip=(-200., 250.), seed=42, root=root), regions)
            self.assertEqual(len(list(root.glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
