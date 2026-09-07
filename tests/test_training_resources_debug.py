"""DEBUG input-accounting fixtures; no patient data or production training."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from hiercp.training_resources import cache_input_inventory, host_input_budget, worst_input_bytes


class TrainingResourceDebugTests(unittest.TestCase):
    def test_inventory_counts_all_candidates_two_views_without_rewriting_cache(self):
        def local(nodes, edges):
            return {"format": "canonical-full-v22", "nodes": {
                "debug": {"x": torch.zeros(nodes, 3)}},
                "edges": {("debug", "debug", "debug"): torch.zeros(2, edges, dtype=torch.int32)}}
        sample = {"source_local": local(3, 7),
                  "target_locals": [local(2, 4), local(4, 5)],
                  "source_patch": torch.zeros(1, 4, 4, 4, dtype=torch.float16)}
        with tempfile.TemporaryDirectory(prefix="debug_input_inventory_") as root:
            paths = [Path(root) / "a.pt", Path(root) / "b.pt"]
            for path in paths:
                torch.save(sample, path)
            before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            report = cache_input_inventory(paths)
            self.assertEqual(report["sample_count"], 2)
            self.assertEqual(report["rows"][0]["canonical_nodes_two_views"], 24)
            self.assertEqual(report["rows"][0]["canonical_edges_two_views"], 46)
            self.assertEqual(worst_input_bytes(report, paths, 2),
                             2 * report["rows"][0]["input_bytes_upper_bound"])
            self.assertGreaterEqual(report["rows"][0]["input_bytes_upper_bound"], 46 * 56)
            self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])
            with self.assertRaisesRegex(ValueError, "cover"):
                worst_input_bytes(report, paths + [Path(root) / "missing.pt"], 1)

    def test_worker_prefetch_uses_allocation_memory_not_host_total(self):
        with patch("hiercp.training_resources.snapshot", return_value={
            "available_memory_bytes": 1000, "host_total_memory_bytes": 100000}):
            serial = host_input_budget(100, workers=0, prefetch_factor=1, pin_memory=False)
            parallel = host_input_budget(100, workers=4, prefetch_factor=2, pin_memory=True)
        self.assertTrue(serial["accepted"])
        self.assertFalse(parallel["accepted"])
        self.assertEqual(parallel["estimated_input_bytes"], 1800)
        self.assertEqual(parallel["available_allocation_bytes"], 1000)


if __name__ == "__main__":
    unittest.main()
