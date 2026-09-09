"""DEBUG resource-control tests, not medical/GPU performance validation."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref

import torch

from tools import causality
from tools import causality_resources as resources


class DebugMeasurement:
    def __enter__(self):
        self.report = {"status": "complete"}
        return self

    def __exit__(self, *args):
        return False


class DebugBatch:
    def __init__(self, size):
        self.counts = tuple(range(size))


def budget(*args, accepted=True, phase="full", **kwargs):
    return {"phase": phase, "accepted": accepted,
            "estimated_input_bytes": 70 if accepted else 90,
            "available_allocation_bytes": 100, "reserved_headroom_fraction": 0.2}


class CausalityPreflightDebugTests(unittest.TestCase):
    @staticmethod
    def identity_inputs():
        # DEBUG shape-accounting metadata only; no cache/data file is loaded.
        selected = [Path(f"DEBUG_identity_{index}.pt").resolve() for index in range(3)]
        inventory = {
            "format": resources.INPUT_ACCOUNTING_VERSION,
            "audit_format": resources.AUDIT_INPUT_ACCOUNTING_VERSION,
            "selected_files": [str(path) for path in selected],
            "sample_count": len(selected),
            "rows": [{"cache_file": path.name, "input_bytes_upper_bound": size}
                     for path, size in zip(selected, (10, 30, 70))],
        }
        artifact = {"format": "DEBUG_artifact_contract", "audit": {"split": "val", "seed": 42}}
        measurement = {
            "format": "hiercp_causality_measurement_plan_v2_host_bounded",
            "batch_candidates": [1, 2, 4, 8], "worker_candidates": [0, 2],
            "maximum_vram_fraction": 0.9, "loader_measurement_batches": 2,
            "prefetch_factor": 1, "pin_memory": True,
            "training_preflight_sha256": "a" * 64,
            "training_candidate_statuses": {
                "batch": [{"batch_size": size, "status": "accepted"} for size in (1, 2, 4, 8)],
                "worker": [{"num_workers": size, "status": "accepted"} for size in (0, 2)],
            },
            "candidate_policy": "remeasure_inference_only_after_fresh_host_budget; no cohort-oversize batches",
            "calibration_order": "descending_canonical_input_upper_bound",
        }
        return artifact, measurement, inventory

    def test_shared_identity_matches_76bc3ee_v2_golden_dict_and_hash_without_mutation(self):
        artifact, measurement, inventory = self.identity_inputs()
        kwargs = {"device": torch.device("cuda:0"), "repeats": 3, "inventory": inventory}
        before = copy.deepcopy((artifact, measurement, kwargs))

        def old_canonical_sha(value):
            return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()

        # Literal schema from 76bc3ee _execute_audit, not the new builder or its
        # format constant. Bounds are independently calculated from 10/30/70.
        golden = {
            "format": "hiercp_causality_preflight_identity_v2_host_bounded",
            "artifact_contract_sha256": old_canonical_sha(artifact),
            "measurement_plan": copy.deepcopy(measurement),
            "device": "cuda:0",
            "repeats": 3,
            "input_inventory_sha256": old_canonical_sha(inventory),
            "input_inventory_sample_count": 3,
            "batch_input_upper_bounds": {"1": 70, "2": 100, "4": 180, "8": 320},
        }
        actual = causality._build_preflight_identity(artifact, measurement, **kwargs)
        self.assertEqual(actual, golden)
        self.assertEqual(causality._value_sha256(actual), old_canonical_sha(golden))
        self.assertEqual((artifact, measurement, kwargs), before)

    def test_shared_identity_rejects_invalid_repeats_device_and_incomplete_inventory(self):
        artifact, measurement, inventory = self.identity_inputs()
        for repeats in (0, 2, True, 3.0, None):
            with self.subTest(repeats=repeats), self.assertRaises(ValueError):
                causality._build_preflight_identity(artifact, measurement, device="cpu",
                                                   repeats=repeats, inventory=inventory)
        for device in (None, 123, {}, "not-a-device", "cuda:-1"):
            with self.subTest(device=device), self.assertRaises((ValueError, RuntimeError)):
                causality._build_preflight_identity(artifact, measurement, device=device,
                                                   repeats=3, inventory=inventory)
        for mutate in (
            lambda value: value.pop("rows"),
            lambda value: value.pop("audit_format"),
            lambda value: value.update(sample_count=4),
            lambda value: value.update(selected_files=[]),
            lambda value: value["rows"].pop(),
        ):
            incomplete = copy.deepcopy(inventory)
            mutate(incomplete)
            before = copy.deepcopy(incomplete)
            with self.assertRaises(ValueError):
                causality._build_preflight_identity(artifact, measurement, device="cpu",
                                                   repeats=3, inventory=incomplete)
            self.assertEqual(incomplete, before)

    def test_cpu_verifier_preserves_recorded_cuda_identity_without_cuda_initialization(self):
        artifact, measurement, inventory = self.identity_inputs()
        before = copy.deepcopy((artifact, measurement, inventory))
        with mock.patch.object(torch.cuda, "is_available", return_value=False), \
             mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("DEBUG must not initialize CUDA")) as initialize:
            self.assertFalse(torch.cuda.is_available())
            recorded = causality._build_preflight_identity(
                artifact, measurement, device="cuda:0", repeats=3, inventory=inventory)
            local_cpu = causality._build_preflight_identity(
                artifact, measurement, device=torch.device("cpu"), repeats=3, inventory=inventory)
        initialize.assert_not_called()
        self.assertEqual(recorded["device"], "cuda:0")
        self.assertEqual(local_cpu["device"], "cpu")
        self.assertNotEqual(causality._value_sha256(recorded), causality._value_sha256(local_cpu))
        self.assertEqual({key: value for key, value in recorded.items() if key != "device"},
                         {key: value for key, value in local_cpu.items() if key != "device"})
        self.assertEqual((artifact, measurement, inventory), before)

    def test_host_rejected_and_oversize_batches_never_load(self):
        selected = [Path(f"DEBUG_{i}.pt") for i in range(4)]
        events, fetched, previous = [], [], []

        class Dataset:
            def __len__(self):
                return len(selected)

            def __getitem__(self, index):
                # Previous collated inputs must die before another load starts.
                self_test.assertTrue(all(ref() is None for ref in previous))
                fetched.append(index)
                return index

        self_test = self

        def collate(samples):
            batch = DebugBatch(len(samples))
            previous.append(weakref.ref(batch))
            return batch

        def host_budget(inventory, files, size, *args, **kwargs):
            return budget(accepted=size == 1, phase=kwargs.get("phase", "full"))

        def workload(model, batch, *, device, seed, progress):
            progress(event="condition_start", condition="DEBUG")

        with mock.patch.object(causality, "HierarchicalCacheDataset", return_value=Dataset()), \
             mock.patch.object(causality, "collate_samples", side_effect=collate), \
             mock.patch.object(causality, "audit_host_budget", side_effect=host_budget), \
             mock.patch.object(causality, "Measurement", DebugMeasurement), \
             mock.patch.object(causality, "_run_preflight_workload", new=workload):
            winner, rows = causality._measure_batch_candidates(
                object(), selected, [1, 2, 8], repeats=3,
                maximum_vram_fraction=0.9, device=torch.device("cpu"), seed=42,
                inventory={}, progress=lambda **fields: events.append(fields),
            )
        self.assertEqual(winner, 1)
        self.assertEqual(fetched, [0, 1, 2])
        self.assertEqual([row["status"] for row in rows],
                         ["accepted", "rejected_host_input_budget", "not_measured_cohort"])
        self.assertEqual([row["completed_samples"] for row in rows], [3, 0, 0])
        self.assertTrue(all(ref() is None for ref in previous))
        self.assertEqual(len([e for e in events if e["event"] == "batch_trial_end"]), 3)
        self.assertEqual(len([e for e in events if e["event"] == "host_budget_check"]), 3)

    def test_rejected_workers_do_not_start_loader_and_consumers_are_released(self):
        previous, calls, events = [], [], []
        self_test = self

        class Loader:
            def __iter__(self):
                for _ in range(3):
                    self_test.assertTrue(all(ref() is None for ref in previous))
                    batch = DebugBatch(2)
                    previous.append(weakref.ref(batch))
                    yield batch
                    del batch

        def load(dataset, **kwargs):
            calls.append(kwargs["num_workers"])
            return Loader()

        def host_budget(inventory, selected, size, workers, *args, **kwargs):
            return budget(accepted=workers == 0)

        with mock.patch.object(causality, "HierarchicalCacheDataset"), \
             mock.patch.object(causality, "DataLoader", side_effect=load), \
             mock.patch.object(causality, "audit_host_budget", side_effect=host_budget), \
             mock.patch.object(causality, "Measurement", DebugMeasurement):
            winner, rows = causality._measure_worker_candidates(
                [Path("DEBUG_a.pt"), Path("DEBUG_b.pt")], [0, 8], batch_size=2,
                measurement_batches=3, pin_memory=False, prefetch_factor=1, seed=42,
                inventory={}, progress=lambda **fields: events.append(fields),
            )
        self.assertEqual(winner, 0)
        self.assertEqual(calls, [0])
        self.assertEqual(rows[0]["samples"], 6)
        self.assertEqual(rows[1]["status"], "rejected_host_input_budget")
        self.assertTrue(all(ref() is None for ref in previous))
        self.assertEqual(len([e for e in events if e["event"] == "worker_trial_end"]), 2)

    def test_live_allocation_guard_does_not_silently_change_settings(self):
        events = []
        with mock.patch.object(causality, "audit_host_budget", return_value=budget(accepted=False)) as guard:
            with self.assertRaisesRegex(RuntimeError, "No graph/data/model reduction"):
                causality._require_host_budget({}, [Path("DEBUG.pt")], 4, 8, 2, True,
                    progress=lambda **fields: events.append(fields), phase="resident_transform")
        self.assertEqual(guard.call_args.args[2:], (4, 8, 2, True))
        self.assertEqual(guard.call_args.kwargs["phase"], "resident_transform")
        self.assertEqual(events[0]["event"], "host_budget_check")

    def record(self):
        selected = [Path(f"DEBUG_{i}.pt").resolve() for i in range(4)]
        inventory = {"format": resources.INPUT_ACCOUNTING_VERSION,
                     "audit_format": resources.AUDIT_INPUT_ACCOUNTING_VERSION,
                     "selected_files": [str(path) for path in selected], "sample_count": 4,
                     "rows": [{"cache_file": path.name, "input_bytes_upper_bound": 10}
                              for path in selected]}

        def rich_budget(size, workers, *, accepted=True):
            with mock.patch("hiercp.training_resources.snapshot", return_value={
                "available_memory_bytes": 1_000_000_000 if accepted else 1,
            }):
                return resources.audit_host_budget(inventory, selected, size, workers, 1, False)

        identity = {"repeats": 3, "input_inventory_sample_count": 4, "device": "cpu",
                    "batch_input_upper_bounds": {"1": 10, "2": 20, "8": 80},
                    "measurement_plan": {"batch_candidates": [1, 2, 8],
                        "worker_candidates": [0, 4], "loader_measurement_batches": 2,
                        "maximum_vram_fraction": 0.9, "prefetch_factor": 1, "pin_memory": False}}
        accepted = {"batch_size": 1, "status": "accepted", "repeats": 3,
                    "cohort_size": 4, "completed_samples": 3, "elapsed_seconds": 1.0,
                    "samples_per_second": 3.0, "peak_vram_fraction": None,
                    "host_budget": rich_budget(1, 0), "host_measurement": {"status": "complete"}}
        rejected = {**accepted, "batch_size": 2, "status": "rejected_host_input_budget",
                    "completed_samples": 0, "host_budget": rich_budget(2, 0, accepted=False),
                    "host_measurement": None, "samples_per_second": None}
        oversize = {**rejected, "batch_size": 8, "status": "not_measured_cohort", "host_budget": None}
        worker = {"num_workers": 0, "status": "accepted", "host_budget": rich_budget(1, 0),
                  "host_measurement": {"status": "complete"}, "measurement_batches": 2,
                  "samples": 2, "elapsed_seconds": 1.0, "samples_per_second": 2.0}
        worker_rejected = {**worker, "num_workers": 4, "status": "rejected_host_input_budget",
                           "samples": 0, "measurement_batches": 0,
                           "host_budget": rich_budget(1, 4, accepted=False), "host_measurement": None}
        record = {"format": causality.PREFLIGHT_FORMAT, "identity": identity,
                  "identity_sha256": causality._value_sha256(identity),
                  "resource_fingerprint": {"DEBUG": True}, "selected_batch_size": 1,
                  "selected_num_workers": 0, "batch_trials": [accepted, rejected, oversize],
                  "worker_trials": [worker, worker_rejected]}
        return record, identity

    def validate(self, record, identity):
        return causality._validate_preflight_record(record, identity=identity,
                                                     resource_fingerprint={"DEBUG": True})

    def test_reader_accepts_only_evidenced_rejections_and_measured_winners(self):
        record, identity = self.record()
        self.assertEqual(self.validate(record, identity), (1, 0))
        for path, key, value in [
            ("batch_trials", "status", "ignored_failure"),
            ("batch_trials", "completed_samples", 2),
            ("batch_trials", "samples_per_second", float("nan")),
            ("batch_trials", "peak_vram_fraction", 0.99),
            ("worker_trials", "measurement_batches", 1),
        ]:
            with self.subTest(path=path, key=key):
                broken = copy.deepcopy(record)
                broken[path][0][key] = value
                with self.assertRaises(ValueError):
                    self.validate(broken, identity)

    def test_reader_rejects_false_host_budget_claim_and_legacy_receipt(self):
        record, identity = self.record()
        for mutate in (
            lambda r: r["batch_trials"][1]["host_budget"].update(accepted=True),
            lambda r: r["worker_trials"][1].update(samples=2),
            lambda r: r["batch_trials"][2].update(batch_size=2),
            lambda r: r.update(format="hiercp_causality_preflight_v1"),
        ):
            broken = copy.deepcopy(record)
            mutate(broken)
            with self.assertRaises(ValueError):
                self.validate(broken, identity)

    def test_verified_preflight_reuse_does_not_remeasure_or_rewrite(self):
        record, identity = self.record()
        with tempfile.TemporaryDirectory(prefix="DEBUG_causality_preflight_") as folder:
            path = Path(folder) / "preflight.json"
            causality._atomic_json(path, record)
            original = path.read_bytes()
            with mock.patch.object(causality, "_measure_batch_candidates") as batch_probe, \
                 mock.patch.object(causality, "_measure_worker_candidates") as worker_probe:
                actual, batch, workers = causality._resolve_preflight(
                    path=path, identity=identity, resource_fingerprint={"DEBUG": True},
                    model=None, selected=[], device=torch.device("cpu"), seed=42,
                    repeats=3, overwrite=False, inventory={}, progress=lambda **fields: None,
                )
            batch_probe.assert_not_called()
            worker_probe.assert_not_called()
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual((actual, batch, workers), (record, 1, 0))


if __name__ == "__main__":
    unittest.main()
