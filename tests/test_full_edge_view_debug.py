"""DEBUG-only full-edge regressions; no medical data or training claims.

The 1,499,368-edge analytic cached-topology fixture reproduces the failing
relation size. It tests execution of every supplied edge, not medical radius
graph validity. Production graph/model configuration and caches are untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import unittest

import numpy as np
import psutil
import torch
from torch_geometric.data import Batch

from hiercp.sample import (
    EDGE_MATERIALIZATION_CHUNK_EDGES,
    FULL_INDUCED_EDGE_CONTRACT,
    LEVEL0_GEOMETRY_CONTRACT,
    _edge_attributes,
    _induced_edge_index,
    build_local_view,
    materialize_sample_views,
)
from hiercp.schema import GraphBuildConfig, LOCAL_EDGE_TYPES, LOCAL_NODE_TYPES


LARGE_EDGE_COUNT = 1_499_368
LARGE_RELATION = ("tumor_surface", "surface_neighbor", "tumor_surface")


def _digest(value: torch.Tensor) -> str:
    return hashlib.sha256(memoryview(value.detach().numpy()).cast("B")).hexdigest()


def _reference_attributes(source, destination, source_position, destination_position,
                          edge, *, position_override=None, normal_override=None):
    """Independent row-wise mathematical oracle for the original 10-D schema."""
    src, dst = edge
    pos = source_position if position_override is None else position_override
    normal = source[:, 9:12] if normal_override is None else normal_override
    delta = destination_position[dst] - pos[src]
    result = np.empty((edge.shape[1], 10), dtype=np.float32)
    result[:, :3] = delta
    result[:, 3] = np.sqrt(np.sum(delta * delta, axis=1))
    for column, feature in ((4, 0), (5, 7), (6, 8)):
        result[:, column] = np.abs(destination[dst, feature] - source[src, feature])
    result[:, 7] = np.sum(normal[src] * destination[dst, 9:12], axis=1)
    result[:, 8] = source[src, 13]
    result[:, 9] = destination[dst, 13]
    return result


def _node(count: int, offset: float) -> dict[str, torch.Tensor]:
    index = torch.arange(count, dtype=torch.float32)
    pos = torch.stack((index.remainder(13), index.remainder(17), index.remainder(19)), 1) / 100
    features = torch.stack([((index + feature) % (feature + 3)) / (feature + 3)
                            for feature in range(16)], 1) + offset
    return {"x": features, "pos": pos, "pos_mm": pos * 10,
            "grid": torch.stack((index, index * 0, index * 0), 1).to(torch.int64)}


def _canonical_fixture(edge_count: int, surface_nodes: int):
    """Create real compact tensors, never mocked edge counts or saved caches."""
    source_types = {"tumor_surface", "tumor_interior", "source_context", "source_liver_surface"}
    nodes = {name: _node(surface_nodes if name == "tumor_surface" else 2, offset / 100)
             for offset, name in enumerate(LOCAL_NODE_TYPES)}
    # Unique directed pairs, without self-loops, in a deterministic cache order.
    flat = np.arange(edge_count, dtype=np.int32)
    src, remainder = flat // (surface_nodes - 1), flat % (surface_nodes - 1)
    dst = remainder + (remainder >= src)
    large = torch.from_numpy(np.stack((src, dst)).astype(np.int32, copy=False))
    edges = {relation: (large if relation == LARGE_RELATION else
                       torch.tensor([[0], [1]], dtype=torch.int32))
             for relation in LOCAL_EDGE_TYPES}
    source = {"format": "canonical-full-v22", "geometry_contract": LEVEL0_GEOMETRY_CONTRACT,
              "nodes": {key: value for key, value in nodes.items() if key in source_types},
              "edges": {key: value for key, value in edges.items()
                        if key[0] in source_types and key[2] in source_types}}
    target = {"format": "canonical-full-v22", "geometry_contract": LEVEL0_GEOMETRY_CONTRACT,
              "nodes": {key: value for key, value in nodes.items() if key not in source_types},
              "edges": {key: value for key, value in edges.items() if key not in source["edges"]},
              "transform": torch.eye(3)}
    return source, target


class EdgeMaterializationDebugTests(unittest.TestCase):
    def test_chunked_induced_endpoints_match_reordered_subset_reference(self):
        canonical = np.array([[0, 1, 2, 3, 2, 0, 3, 1], [3, 2, 1, 0, 3, 2, 1, 0]], dtype=np.int32)
        original = canonical.copy()
        selected_source, selected_destination = np.array([3, 0, 2]), np.array([1, 3, 0])
        smap = {int(node): local for local, node in enumerate(selected_source)}
        dmap = {int(node): local for local, node in enumerate(selected_destination)}
        expected = np.array([(smap[int(src)], dmap[int(dst)]) for src, dst in canonical.T
                             if int(src) in smap and int(dst) in dmap], dtype=np.int64).T
        for chunk in (1, 3, 7, 64):
            with self.subTest(chunk=chunk):
                actual = _induced_edge_index(canonical, selected_source, selected_destination,
                                             source_count=4, destination_count=4, chunk_edges=chunk)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(canonical, original)
        complete = _induced_edge_index(canonical, np.arange(4), np.arange(4),
                                      source_count=4, destination_count=4, chunk_edges=3)
        np.testing.assert_array_equal(complete, canonical)
        self.assertEqual(complete.dtype, np.dtype("int64"))
        self.assertFalse(np.shares_memory(complete, canonical))

    def test_malformed_edges_and_duplicate_nodes_still_fail(self):
        for edge in (np.array([[0], [-1]]), np.array([[4], [0]]), np.array([[0.0], [1.0]])):
            with self.subTest(edge=edge.tolist()), self.assertRaises(ValueError):
                _induced_edge_index(edge, np.arange(4), np.arange(4), source_count=4, destination_count=4)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _induced_edge_index(np.array([[0], [1]]), np.array([0, 0]), np.arange(4),
                                source_count=4, destination_count=4)
        for invalid_chunk in (0, -1, True, 1.5):
            with self.subTest(chunk=invalid_chunk), self.assertRaises(ValueError):
                _induced_edge_index(np.array([[0], [1]]), np.arange(4), np.arange(4),
                                    source_count=4, destination_count=4, chunk_edges=invalid_chunk)

    def test_chunked_all_ten_attributes_and_transform_overrides_are_exact(self):
        rng = np.random.default_rng(73)
        source = rng.normal(size=(17, 16)).astype(np.float32)
        destination = rng.normal(size=(13, 16)).astype(np.float32)
        source_position = rng.normal(size=(17, 3)).astype(np.float32)
        destination_position = rng.normal(size=(13, 3)).astype(np.float32)
        edge = np.stack((rng.integers(17, size=113), rng.integers(13, size=113)))
        saved = [value.copy() for value in (source, destination, source_position, destination_position, edge)]
        for overridden in (False, True):
            pos = source_position[:, [1, 2, 0]] if overridden else None
            normal = source[:, [10, 11, 9]] if overridden else None
            expected = _reference_attributes(source, destination, source_position, destination_position,
                                             edge, position_override=pos, normal_override=normal)
            for chunk in (1, 7, 32, 128):
                actual = _edge_attributes(source, destination, source_position, destination_position, edge,
                                          source_position_override=pos, source_normal_override=normal,
                                          chunk_edges=chunk)
                np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip((source, destination, source_position, destination_position, edge), saved):
            np.testing.assert_array_equal(actual, expected)

    def test_materializing_two_views_does_not_mutate_saved_graph_configuration(self):
        source, target = _canonical_fixture(37, 8)
        config = GraphBuildConfig().to_dict()
        config["sample_relation_edge_limit"] = 500000  # historical cache value, not a new threshold
        saved = copy.deepcopy(config)
        sample = {"source_local": source, "target_locals": [target], "graph_config": config,
                  "case_id": "DEBUG_ANALYTIC", "source_component_id": 1, "sample_index": 0}
        materialize_sample_views(sample, training=True, epoch=3, global_seed=42)
        self.assertEqual(sample["graph_config"], saved)
        self.assertEqual(len(sample["local_graphs"]), 1)
        self.assertEqual(len(sample["local_graphs_view2"]), 1)
        self.assertEqual(sample["local_graphs"][0].edge_execution_contract, FULL_INDUCED_EDGE_CONTRACT)
        self.assertEqual(source["edges"][LARGE_RELATION].dtype, torch.int32)


class FullSizeEdgeViewDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        available = int(psutil.virtual_memory().available)
        # This is a host-allocation safety check, never an edge-count fallback.
        # The test fails explicitly if it cannot run all requested edges.
        if available < 1_500_000_000:
            raise RuntimeError(f"DEBUG full 1,499,368-edge test needs 1.5 GB free RAM; available={available}")
        cls.started = time.perf_counter()
        cls.rss_before = psutil.Process().memory_info().rss
        cls.source, cls.target = _canonical_fixture(LARGE_EDGE_COUNT, 1226)
        cls.config = GraphBuildConfig(sample_relation_edge_limit=500000)
        cls.saved_config = copy.deepcopy(cls.config.to_dict())
        cls.canonical_sha = _digest(cls.source["edges"][LARGE_RELATION])
        cls.graph = build_local_view(cls.source, cls.target, cls.config, seed=42)
        print("[DEBUG full-edge view resources] " + json.dumps({
            "cpu_logical": os.cpu_count(), "cuda_available": torch.cuda.is_available(),
            "ram_available_before": available, "edges": LARGE_EDGE_COUNT,
            "materialization_seconds": time.perf_counter() - cls.started,
            "rss_bytes": psutil.Process().memory_info().rss,
            "medical_data_used": False, "production_configuration_modified": False,
        }), flush=True)

    @classmethod
    def tearDownClass(cls):
        del cls.graph, cls.source, cls.target

    def test_all_1499368_endpoints_attributes_and_legacy_diagnostics_preserved(self):
        graph, canonical = self.graph, self.source["edges"][LARGE_RELATION]
        store = graph[LARGE_RELATION]
        self.assertEqual(tuple(store.edge_index.shape), (2, LARGE_EDGE_COUNT))
        self.assertEqual(tuple(store.edge_attr.shape), (LARGE_EDGE_COUNT, 10))
        np.testing.assert_array_equal(store.edge_index.numpy(), canonical.numpy())
        node = graph["tumor_surface"]
        x, pos = node.x.numpy(), node.pos.numpy()
        for start in range(0, LARGE_EDGE_COUNT, EDGE_MATERIALIZATION_CHUNK_EDGES):
            stop = min(LARGE_EDGE_COUNT, start + EDGE_MATERIALIZATION_CHUNK_EDGES)
            expected = _reference_attributes(x, x, pos, pos, store.edge_index[:, start:stop].numpy())
            np.testing.assert_array_equal(store.edge_attr[start:stop].numpy(), expected)
        self.assertEqual(graph.edge_execution_contract, FULL_INDUCED_EDGE_CONTRACT)
        self.assertEqual(graph.legacy_sample_relation_edge_limit.tolist(), [500000])
        self.assertEqual(graph.relation_exceeds_legacy_limit.sum().item(), 1)
        self.assertTrue(graph.relation_exceeds_legacy_limit[0, LOCAL_EDGE_TYPES.index(LARGE_RELATION)])
        self.assertEqual(graph.relation_edges_dropped_by_limit.sum().item(), 0)
        self.assertEqual(graph.relation_edge_counts.tolist(), graph.canonical_edge_counts.tolist())
        self.assertEqual(graph.sampled_counts.tolist(), graph.canonical_counts.tolist())
        self.assertEqual(self.config.to_dict(), self.saved_config)
        self.assertEqual(_digest(canonical), self.canonical_sha)
        self.assertEqual(canonical.dtype, torch.int32)

    def test_disjoint_two_graph_batch_keeps_all_edges_and_all_edge_gradients(self):
        batch = Batch.from_data_list([self.graph, self.graph])
        store = batch[LARGE_RELATION]
        node_count = int(self.graph["tumor_surface"].num_nodes)
        self.assertEqual(tuple(store.edge_index.shape), (2, 2 * LARGE_EDGE_COUNT))
        torch.testing.assert_close(store.edge_index[:, :LARGE_EDGE_COUNT], self.graph[LARGE_RELATION].edge_index)
        torch.testing.assert_close(store.edge_index[:, LARGE_EDGE_COUNT:] - node_count,
                                   self.graph[LARGE_RELATION].edge_index)
        self.assertEqual(batch.legacy_sample_relation_edge_limit.tolist(), [500000, 500000])
        self.assertEqual(tuple(batch.relation_edge_counts.shape), (2, len(LOCAL_EDGE_TYPES)))
        attributes = store.edge_attr.detach().requires_grad_(True)
        source_value = batch["tumor_surface"].x[:, 0].detach().clone().requires_grad_(True)
        gain = torch.nn.Parameter(torch.tensor(0.75))
        optimizer = torch.optim.SGD([gain], lr=0.01)
        messages = gain * (source_value[store.edge_index[0]] + attributes.sum(1))
        aggregated = torch.zeros_like(source_value).index_add(0, store.edge_index[1], messages)
        loss = aggregated.sum() / int(store.edge_index.shape[1])
        loss.backward()
        self.assertEqual(tuple(attributes.grad.shape), (2 * LARGE_EDGE_COUNT, 10))
        expected_gradient = 0.75 / (2 * LARGE_EDGE_COUNT)
        self.assertAlmostEqual(float(attributes.grad.min()) / expected_gradient, 1.0, places=6)
        self.assertAlmostEqual(float(attributes.grad.max()) / expected_gradient, 1.0, places=6)
        self.assertTrue(torch.isfinite(source_value.grad).all())
        self.assertTrue(torch.isfinite(gain.grad))
        self.assertGreater(float(gain.grad.abs()), 0)
        before = gain.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(gain.detach(), before))
        self.assertEqual(_digest(self.source["edges"][LARGE_RELATION]), self.canonical_sha)

    def test_full_relation_actual_attention_forward_backward_cpu_debug(self):
        from hiercp.model import CompatibilityGatedGATv2Conv

        # Narrow dimensions are an explicit analytic operator DEBUG fixture;
        # no production hidden/head/depth/batch configuration is changed.
        torch.manual_seed(43)
        conv = CompatibilityGatedGATv2Conv((4, 4), 2, heads=2, edge_dim=10,
                                         add_self_loops=False, dropout=0.0)
        node_count = int(self.graph["tumor_surface"].num_nodes)
        source = torch.randn(node_count, 4, requires_grad=True)
        destination = torch.randn(node_count, 4, requires_grad=True)
        edge = self.graph[LARGE_RELATION].edge_index
        attributes = self.graph[LARGE_RELATION].edge_attr.detach().requires_grad_(True)
        optimizer = torch.optim.SGD(conv.parameters(), lr=0.001)
        started = time.perf_counter()
        result = conv((source, destination), edge, attributes)
        forward_seconds = time.perf_counter() - started
        self.assertEqual(tuple(result.shape), (node_count, 4))
        self.assertTrue(torch.isfinite(result).all())
        result.square().mean().backward()
        self.assertEqual(tuple(attributes.grad.shape), (LARGE_EDGE_COUNT, 10))
        self.assertTrue(torch.isfinite(attributes.grad).all())
        self.assertTrue(torch.all(attributes.grad.abs().sum(1) > 0))
        for name, parameter in conv.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertTrue(torch.isfinite(destination.grad).all())
        before = {name: parameter.detach().clone() for name, parameter in conv.named_parameters()}
        optimizer.step()
        self.assertTrue(any(not torch.equal(parameter, before[name]) for name, parameter in conv.named_parameters()))
        print("[DEBUG full-edge actual attention] " + json.dumps({
            "device": "cpu", "edges": LARGE_EDGE_COUNT, "input_channels": 4,
            "heads": 2, "out_channels_per_head": 2, "forward_seconds": forward_seconds,
            "forward_backward_seconds": time.perf_counter() - started,
            "rss_bytes": psutil.Process().memory_info().rss,
            "peak_working_set_bytes": getattr(psutil.Process().memory_info(), "peak_wset", None),
            "chunk_edges": conv._edge_chunk_size(torch.float32),
            "medical_data_used": False, "production_model_validation": False,
        }), flush=True)


if __name__ == "__main__":
    unittest.main()
