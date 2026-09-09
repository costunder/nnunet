"""DEBUG-only resource fixtures: no patient data, GPU or production training."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tools.causality_resources import (
    AUDIT_INPUT_ACCOUNTING_VERSION,
    TRANSFORM_WORKSPACE_INPUT_COPIES,
    audit_allocation_fingerprint,
    audit_batch_input_bytes,
    audit_host_budget,
    build_audit_inventory,
    create_progress_log,
    validate_audit_host_budget,
)
from hiercp.training_resources import INPUT_ACCOUNTING_VERSION


def debug_snapshot(available=10000):
    return {
        "rss_bytes": 250,
        "available_memory_bytes": available,
        "host_total_memory_bytes": 10_000_000,
        "cpu_affinity_cores": 8,
        "cpu_allocation_cores": 4.0,
        "cgroup": {
            "locations_resolved": True,
            "values": {
                "v2:/debug/job": {"memory.max": "12000", "memory.current": "2000", "cpu.max": "400000 100000"},
                "v2:/debug": {"memory.max": "20000", "memory.current": "5000", "cpu.max": "max 100000"},
            },
            "missing_files": [],
            "limitation": "DEBUG synthetic cgroup evidence",
        },
    }


def debug_inventory(sizes=(100, 200, 300)):
    paths = [Path(f"debug_{index}.pt").resolve() for index in range(len(sizes))]
    return paths, {
        "format": INPUT_ACCOUNTING_VERSION,
        "audit_format": AUDIT_INPUT_ACCOUNTING_VERSION,
        "selected_files": [str(path) for path in paths],
        "sample_count": len(paths),
        "rows": [{"cache_file": path.name, "input_bytes_upper_bound": size}
                 for path, size in zip(paths, sizes)],
    }


class CausalityResourcesDebugTests(unittest.TestCase):
    def setUp(self):
        # DEBUG arithmetic fixtures use byte-sized inputs, not an actual audit.
        workspace = patch("tools.causality_resources.local_edge_jaccard_workspace_bytes", return_value=128)
        workspace.start()
        self.addCleanup(workspace.stop)

    def test_complete_inventory_preserves_cache_bytes(self):
        def local(nodes, edges):
            return {"format": "canonical-full-v22", "nodes": {
                "debug": {"x": torch.zeros(nodes, 3)}},
                "edges": {("debug", "debug", "debug"): torch.zeros(2, edges, dtype=torch.int32)}}
        with tempfile.TemporaryDirectory(prefix="debug_causality_inventory_") as root:
            paths = [Path(root) / f"case_{index}.pt" for index in range(3)]
            for index, path in enumerate(paths):
                torch.save({"source_local": local(2 + index, 5 + index),
                            "target_locals": [local(2, 4), local(4, 5)],
                            "source_patch": torch.zeros(1, 4, 4, 4, dtype=torch.float16)}, path)
            before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            report = build_audit_inventory(paths)
            self.assertEqual(report["sample_count"], len(paths))
            self.assertEqual(report["selected_files"], [str(path.resolve()) for path in paths])
            self.assertEqual(len(report["rows"]), len(paths))
            self.assertEqual(report["rows"][0]["candidate_count"], 2)
            self.assertFalse(report["cache_modified"])
            self.assertFalse(report["graph_or_data_reduction"])
            self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])

    def test_repeated_batches_bound_every_cyclic_start_and_order(self):
        paths, inventory = debug_inventory()
        for batch_size in range(1, 11):
            bound = audit_batch_input_bytes(inventory, paths, batch_size)
            for order in itertools.permutations((100, 200, 300)):
                for start in range(len(order)):
                    actual = sum(order[(start + index) % len(order)] for index in range(batch_size))
                    self.assertGreaterEqual(bound, actual)
        self.assertEqual(audit_batch_input_bytes(inventory, paths, 8), 1700)

    def test_exact_full_cohort_required(self):
        paths, inventory = debug_inventory()
        with self.assertRaisesRegex(ValueError, "exact complete"):
            audit_batch_input_bytes(inventory, paths[:-1], 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_audit_inventory(paths + [paths[0]])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            build_audit_inventory([])
        inventory["rows"].pop()
        with self.assertRaisesRegex(ValueError, "rows"):
            audit_batch_input_bytes(inventory, paths, 1)

    def test_reject_invalid_resource_candidates(self):
        paths, inventory = debug_inventory()
        for field, value in (("batch_size", 0), ("batch_size", True), ("workers", -1),
                             ("prefetch_factor", 0), ("pin_memory", 1), ("phase", "unknown")):
            args = {"batch_size": 1, "workers": 0, "prefetch_factor": 2, "pin_memory": False}
            args[field] = value
            with self.assertRaises(ValueError):
                audit_host_budget(inventory, paths, **args)

    def test_worker_prefetch_and_transform_copies_use_allocation_headroom(self):
        paths, inventory = debug_inventory((100,))
        with patch("hiercp.training_resources.snapshot", return_value=debug_snapshot(2000)):
            serial = audit_host_budget(inventory, paths, 1, 0, 2, False)
            parallel = audit_host_budget(inventory, paths, 1, 4, 2, True)
        self.assertEqual(serial["input_copies_accounted"], 3 + TRANSFORM_WORKSPACE_INPUT_COPIES)
        self.assertEqual(parallel["estimated_input_bytes"], 100 * (18 + TRANSFORM_WORKSPACE_INPUT_COPIES) + 128)
        self.assertEqual(parallel["overlap_workspace_bytes"], 128)
        self.assertTrue(serial["accepted"])
        self.assertFalse(parallel["accepted"])
        self.assertEqual(parallel["available_allocation_bytes"], 2000)
        self.assertEqual(parallel["allocation"]["host_total_memory_bytes"], 10_000_000)
        self.assertFalse(parallel["graph_or_data_reduction"])

    def test_resident_guard_does_not_charge_allocated_queues_again(self):
        paths, inventory = debug_inventory((100,))
        with patch("hiercp.training_resources.snapshot", return_value=debug_snapshot(1000)):
            full = audit_host_budget(inventory, paths, 1, 4, 2, True)
            resident = audit_host_budget(inventory, paths, 1, 4, 2, True, phase="resident_transform")
        self.assertFalse(full["accepted"])
        self.assertTrue(resident["accepted"])
        self.assertEqual(resident["loader_input_copies_accounted"], 0)
        self.assertEqual(resident["estimated_input_bytes"], 100 * TRANSFORM_WORKSPACE_INPUT_COPIES + 128)

    def test_real_numeric_overlap_allowance_is_not_scaled_down_with_small_inputs(self):
        from hiercp.causality_overlap import local_edge_jaccard_workspace_bytes
        actual_workspace = local_edge_jaccard_workspace_bytes()
        self.assertEqual(actual_workspace, 512 * 262144 + 1024 * 1024)
        paths, inventory = debug_inventory((100,))
        with patch("hiercp.training_resources.snapshot", return_value=debug_snapshot(1_000_000_000)), \
                patch("tools.causality_resources.local_edge_jaccard_workspace_bytes", return_value=actual_workspace):
            budget = audit_host_budget(inventory, paths, 1, 0, 2, False, phase="resident_transform")
        self.assertEqual(budget["estimated_input_bytes"], actual_workspace + 100 * TRANSFORM_WORKSPACE_INPUT_COPIES)

    def test_every_guard_takes_a_fresh_allocation_snapshot(self):
        paths, inventory = debug_inventory((100,))
        with patch("hiercp.training_resources.snapshot", side_effect=[debug_snapshot(10000), debug_snapshot(100)]) as observe:
            self.assertTrue(audit_host_budget(inventory, paths, 1, 0, 2, False)["accepted"])
            self.assertFalse(audit_host_budget(inventory, paths, 1, 0, 2, False)["accepted"])
        self.assertEqual(observe.call_count, 2)

    def test_saved_budget_validation_does_not_sample_current_resources(self):
        paths, inventory = debug_inventory((100,))
        for phase in ("full", "resident_transform"):
            with patch("hiercp.training_resources.snapshot", return_value=debug_snapshot()):
                saved = audit_host_budget(inventory, paths, 1, 4, 2, True, phase=phase)
            with patch("tools.causality_resources.snapshot", side_effect=AssertionError("must not resample")), \
                    patch("hiercp.training_resources.snapshot", side_effect=AssertionError("must not resample")):
                validate_audit_host_budget(saved, batch_size=1, workers=4, prefetch_factor=2,
                                          pin_memory=True, batch_input_bytes_upper_bound=100, phase=phase)

    def test_saved_budget_rejects_forged_arithmetic_candidate_and_snapshot(self):
        paths, inventory = debug_inventory((100,))
        with patch("hiercp.training_resources.snapshot", return_value=debug_snapshot()):
            saved = audit_host_budget(inventory, paths, 1, 0, 2, False)
        invalid = {"estimated_input_bytes": 0, "num_workers": 4, "batch_size": 2,
                   "overlap_workspace_bytes": 0, "permitted_input_bytes": 0,
                   "input_copies_accounted": 1, "pin_memory": 0,
                   "accepted": 1, "graph_or_data_reduction": True,
                   "batch_input_bytes_upper_bound": 99, "selected_cohort_samples": 0}
        for name, value in invalid.items():
            with self.subTest(field=name):
                forged = copy.deepcopy(saved)
                forged[name] = value
                with self.assertRaises(ValueError):
                    validate_audit_host_budget(forged, batch_size=1, workers=0, prefetch_factor=2,
                                              pin_memory=False, batch_input_bytes_upper_bound=100)
        forged = copy.deepcopy(saved)
        forged["allocation"]["available_memory_bytes"] -= 1
        with self.assertRaisesRegex(ValueError, "snapshot"):
            validate_audit_host_budget(forged, batch_size=1, workers=0, prefetch_factor=2,
                                      pin_memory=False, batch_input_bytes_upper_bound=100)

    def test_fingerprint_keeps_all_ancestor_limits_but_not_current_usage(self):
        original = debug_snapshot()
        changed_use = copy.deepcopy(original)
        changed_use["available_memory_bytes"] = 3
        changed_use["rss_bytes"] = 9000
        changed_use["cgroup"]["values"]["v2:/debug/job"]["memory.current"] = "11999"
        changed_limit = copy.deepcopy(changed_use)
        changed_limit["cgroup"]["values"]["v2:/debug"]["memory.max"] = "15000"
        with patch("tools.causality_resources.snapshot", side_effect=[original, changed_use, changed_limit]):
            first, second, third = (audit_allocation_fingerprint() for _ in range(3))
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(set(first["cgroup_limits"]), {"v2:/debug/job", "v2:/debug"})

    def test_progress_is_unique_flushed_append_only_and_preserves_output(self):
        with tempfile.TemporaryDirectory(prefix="debug_causality_progress_") as root:
            output = Path(root) / "audit.json"
            output.write_text("DEBUG existing report", encoding="utf-8")
            with patch("tools.causality_resources.snapshot", return_value=debug_snapshot()), \
                    patch("tools.causality_resources.os.fsync", wraps=__import__("os").fsync) as sync:
                with create_progress_log(output) as progress:
                    progress(stage="batch", event="start", batch_size=2)
                    prefix = progress.path.read_bytes()
                    progress(stage="condition", condition="node_order", event="done")
                    self.assertTrue(progress.path.read_bytes().startswith(prefix))
                    rows = [json.loads(line) for line in progress.path.read_text(encoding="utf-8").splitlines()]
                    self.assertEqual([row["sequence"] for row in rows], [0, 1])
                    self.assertEqual(rows[1]["resource_snapshot"]["available_memory_bytes"], 10000)
                    self.assertGreater(rows[0]["pid"], 0)
                    self.assertEqual(sync.call_count, 2)
                with create_progress_log(output) as second:
                    second(stage="resume", event="start")
                    self.assertNotEqual(progress.path, second.path)
            self.assertEqual(output.read_text(encoding="utf-8"), "DEBUG existing report")
            with self.assertRaisesRegex(ValueError, "closed"):
                progress(stage="bad")

    def test_progress_creation_never_overwrites_a_collision(self):
        with tempfile.TemporaryDirectory(prefix="debug_causality_collision_") as root:
            output = Path(root) / "audit.json"
            with patch("tools.causality_resources.uuid.uuid4") as make_uuid:
                make_uuid.return_value.hex = "debug_fixed"
                with create_progress_log(output) as first:
                    with patch("tools.causality_resources.snapshot", return_value=debug_snapshot()):
                        first(stage="existing")
                    before = first.path.read_bytes()
                    with self.assertRaises(FileExistsError):
                        create_progress_log(output)
                    self.assertEqual(first.path.read_bytes(), before)

    def test_progress_rejects_nonfinite_and_reserved_fields_without_partial_record(self):
        with tempfile.TemporaryDirectory(prefix="debug_causality_json_") as root:
            with create_progress_log(Path(root) / "audit.json") as progress, \
                    patch("tools.causality_resources.snapshot", return_value=debug_snapshot()):
                with self.assertRaises(ValueError):
                    progress(stage="invalid", value=float("nan"))
                with self.assertRaisesRegex(ValueError, "reserved"):
                    progress(pid=0)
                self.assertEqual(progress.path.read_bytes(), b"")
                progress(stage="valid")
                self.assertEqual(json.loads(progress.path.read_text(encoding="utf-8"))["sequence"], 0)


if __name__ == "__main__":
    unittest.main()
