"""Metadata and metric tests; these do not claim neural learning or GPU proof."""
from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import json
from pathlib import Path
import unittest
from zipfile import ZipFile

from hiercp_v1x.scope_learning_loop import (
    execution_contract, independent_transfer, score_metrics, run_learning,
)


ROOT = Path(__file__).resolve().parents[1]


def original_config():
    with ZipFile(ROOT / "versions/v1/pipeline_v1_source.zip") as archive:
        return json.loads(archive.read("config/train.json"))


class ExecutionContractUnits(unittest.TestCase):
    def test_complete_native_epochs_and_curriculum_kept(self):
        config = original_config()
        before = copy.deepcopy(config)
        result = execution_contract(config, physical_batch=4, workers=8, epochs=40, smoke=False)
        self.assertEqual(config, before)
        self.assertEqual(result["scheduler_t_max"], 40)
        self.assertEqual(result["fixed_validation_epoch"], 29)
        self.assertEqual(result["physical_candidate_graph_batch"], 32)
        self.assertEqual(result["effective_sample_batch"], 4)
        self.assertFalse(result["production_ready"])
        self.assertFalse(result["quality_verified"])
        self.assertFalse(result["checkpoint_written"])
        self.assertFalse(result["full_training"])

    def test_smoke_does_not_compress_native_lr_schedule(self):
        for epochs in (1, 2):
            result = execution_contract(original_config(), physical_batch=2, workers=2,
                                        epochs=epochs, smoke=True)
            self.assertEqual(result["scheduler_t_max"], 40)
            self.assertEqual(result["epochs"], epochs)
            self.assertTrue(result["smoke"])

    def test_arbitrary_shortening_rejected(self):
        for epochs, smoke in ((1, False), (39, False), (41, False), (3, True), (0, True)):
            with self.subTest(epochs=epochs, smoke=smoke), self.assertRaises(ValueError):
                execution_contract(original_config(), physical_batch=2, workers=2,
                                   epochs=epochs, smoke=smoke)

    def test_native_contract_mutation_rejected(self):
        changes = (("seed", None, 43), ("cache", "total_candidates", 7),
                   ("cache", "candidate_pool_size", 64),
                   ("training", "epochs", 2),
                   ("training", "fixed_validation_epoch", 1),
                   ("training", "gradient_accumulation_steps", 2),
                   ("training", "target_effective_batch_size", 8))
        for section, key, value in changes:
            config = original_config()
            if key is None:
                config[section] = value
            else:
                config[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                execution_contract(config, physical_batch=2, workers=2, epochs=40, smoke=False)

    def test_explicit_parallel_batch_and_workers_required(self):
        for batch, workers in ((1, 2), (2, 1), (True, 2), (2, False)):
            with self.subTest(batch=batch, workers=workers), self.assertRaises(ValueError):
                execution_contract(original_config(), physical_batch=batch, workers=workers,
                                   epochs=40, smoke=False)


class RankingMetricUnits(unittest.TestCase):
    def test_full8_ranking_scores(self):
        result = score_metrics([[8, 7, 6, 5, 4, 3, 2, 1], [7, 8, 6, 5, 4, 3, 2, 1]])
        self.assertEqual(result["positive_ranks"], [1, 2])
        self.assertEqual(result["top1"], .5)
        self.assertEqual(result["MRR"], .75)
        self.assertEqual(result["margin"], 0)
        self.assertAlmostEqual(result["pair_win"], 13 / 14)

    def test_conservative_ties_match_original_native_metrics(self):
        result = score_metrics([[1] * 8])
        self.assertEqual(result["top1"], 0)
        self.assertEqual(result["MRR"], 1 / 8)
        self.assertEqual(result["pair_win"], 0)
        self.assertEqual(result["margin"], 0)

    def test_missing_and_nonfinite_scores_rejected(self):
        for values in ([], [[1] * 7], [[1] * 9], [[float("nan")] + [0] * 7],
                       [[0] * 7 + [float("inf")]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                score_metrics(values)

    def test_sample_weighting_is_not_batch_mean_weighting(self):
        result = score_metrics([[8, 7, 6, 5, 4, 3, 2, 1]] * 3 + [[0, 7, 6, 5, 4, 3, 2, 1]])
        self.assertEqual(result["top1"], .75)
        self.assertAlmostEqual(result["MRR"], (3 + 1 / 8) / 4)


class CpuCacheTransferUnits(unittest.TestCase):
    def test_pyg_cpu_stores_not_mutated_by_independent_transfer(self):
        # CPU/meta-only storage correctness; no fabricated medical/GPU input.
        import torch
        from torch_geometric.data import Batch, HeteroData
        from hiercp.data import HierarchicalBatch
        graph = HeteroData()
        graph["node"].x = torch.ones(2, 3)
        graph["node", "edge", "node"].edge_index = torch.tensor([[0], [1]])
        batched = Batch.from_data_list([graph, graph])
        batch = HierarchicalBatch(source_patches=torch.ones(2, 5, 2, 2, 2),
            target_patches=torch.ones(16, 5, 2, 2, 2),
            local_batch=batched, local_batch_view2=copy.copy(batched),
            patient_batch=copy.copy(batched), prototype_batch=copy.copy(batched),
            difficulties=torch.arange(16), counts=(8, 8), case_ids=("unit_a", "unit_b"))
        transferred = independent_transfer(batch, "meta", non_blocking=False)
        self.assertEqual(transferred.source_patches.device.type, "meta")
        self.assertEqual(batch.source_patches.device.type, "cpu")
        for field in ("local_batch", "local_batch_view2", "patient_batch", "prototype_batch"):
            self.assertEqual(getattr(transferred, field)["node"].x.device.type, "meta")
            self.assertEqual(getattr(batch, field)["node"].x.device.type, "cpu")
            self.assertEqual(getattr(batch, field)["node", "edge", "node"].edge_index.device.type, "cpu")

    def test_no_checkpoint_writer_or_node_loop(self):
        parsed = ast.parse(inspect.getsource(run_learning))
        forbidden = {"save_checkpoint", "save_checkpoint_atomic", "save", "load_state_dict"}
        called = {node.func.attr for node in ast.walk(parsed)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(called & forbidden)
        forward_calls = [node for node in ast.walk(parsed) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == "net"]
        self.assertEqual(len(forward_calls), 2)  # batched training and batched eval only


if __name__ == "__main__":
    unittest.main()
