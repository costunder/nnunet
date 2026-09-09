"""DEBUG synthetic exact-overlap regressions, not real-data/GPU validation."""
from __future__ import annotations

import copy
import errno
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import HeteroData

from hiercp.causality_overlap import (
    local_edge_jaccard, local_edge_jaccard_disk_bytes, local_edge_jaccard_workspace_bytes,
)
from hiercp.schema import LOCAL_EDGE_TYPES, LOCAL_NODE_TYPES


def debug_graph(ids, relations, *, dtype=torch.long):
    graph = HeteroData()
    for node_type in LOCAL_NODE_TYPES:
        graph[node_type].full_id = torch.tensor(ids, dtype=dtype)
    for index, edges in relations.items():
        graph[LOCAL_EDGE_TYPES[index]].edge_index = torch.tensor(edges, dtype=torch.long).reshape(2, -1)
    return graph


def debug_legacy_edges(graph):
    result = set()
    for relation, kind in enumerate(LOCAL_EDGE_TYPES):
        if kind in graph.edge_types:
            edges = graph[kind].edge_index
            result.update((relation, int(source), int(destination)) for source, destination in zip(
                graph[kind[0]].full_id[edges[0]].tolist(),
                graph[kind[2]].full_id[edges[1]].tolist()))
    return result


class CausalityOverlapDebugTests(unittest.TestCase):
    def assert_overlap(self, first, second, *, chunk_edges=2):
        expected_first, expected_second = debug_legacy_edges(first), debug_legacy_edges(second)
        union = expected_first | expected_second
        expected = len(expected_first & expected_second) / len(union) if union else 1.0
        before = [copy.deepcopy(graph.to_dict()) for graph in (first, second)]
        stats = {}
        actual = local_edge_jaccard(first, second, chunk_edges=chunk_edges, diagnostics=stats)
        self.assertEqual(actual, expected)
        self.assertEqual(stats["first_unique_edges"], len(expected_first))
        self.assertEqual(stats["second_unique_edges"], len(expected_second))
        self.assertEqual(stats["union_edges"], len(union))
        self.assertLessEqual(stats.get("peak_key_chunk_rows", 0), chunk_edges)
        self.assertLessEqual(stats.get("peak_read_rows", 0), chunk_edges)
        self.assertLessEqual(stats.get("peak_paired_input_rows", 0), 2 * chunk_edges)
        self.assertLessEqual(stats.get("peak_merge_output_rows", 0), 2 * chunk_edges)
        for graph, original in zip((first, second), before):
            for store, fields in original.items():
                for name, value in fields.items():
                    torch.testing.assert_close(graph[store][name], value, rtol=0, atol=0)
        return stats

    def test_empty_and_missing_relations(self):
        empty = debug_graph([], {})
        present_empty = debug_graph([], {0: [[], []], 11: [[], []]})
        populated = debug_graph([1, 2], {0: [[0], [1]]})
        self.assert_overlap(empty, present_empty)
        self.assert_overlap(empty, populated)
        self.assert_overlap(populated, empty)

    def test_relation_identity_direction_duplicates_and_int64_extremes(self):
        ids = [-(2**63), -1, 0, 2**63 - 1, 2**53 + 1, 2**53 + 2]
        first = debug_graph(ids, {0: [[0, 1, 1, 2, 3, 4, 5], [3, 2, 2, 1, 0, 5, 4]],
                                  11: [[0, 0, 4], [3, 3, 5]], 15: [[], []]})
        second = debug_graph(ids, {0: [[0, 1, 5, 4, 0], [3, 2, 4, 5, 3]],
                                   1: [[0], [3]], 11: [[3, 4], [0, 5]]})
        stats = self.assert_overlap(first, second, chunk_edges=1)
        self.assertGreater(stats["merge_levels"], 1)
        self.assertGreater(stats["initial_runs"], 2)
        self.assert_overlap(first, second, chunk_edges=64)

    def test_duplicate_full_ids_and_edges_across_runs(self):
        first = debug_graph([5, 5, -2], {2: [[0, 1, 0, 2, 1, 2], [2, 2, 2, 1, 2, 0]]})
        second = debug_graph([-2, 5], {2: [[0, 1, 1], [1, 0, 0]]})
        self.assert_overlap(first, second)

    def test_disjoint_and_identical_multichunk_sets(self):
        first = debug_graph(range(9), {0: [list(range(9)) * 3, list(reversed(range(9))) * 3]})
        second = debug_graph(range(9, 18), {0: [list(range(9)), list(reversed(range(9)))]})
        self.assert_overlap(first, first, chunk_edges=3)
        self.assert_overlap(first, second, chunk_edges=3)

    def test_deterministic_randomized_legacy_parity(self):
        generator = np.random.default_rng(918)
        for size in (1, 3, 17, 31):
            for chunk in (1, 4, 64):
                graphs = [debug_graph([-9, 2, 10, 10, 12], {
                    relation: generator.integers(0, 5, size=(2, size)).tolist()
                    for relation in (0, 2, 11)}) for _ in range(2)]
                self.assert_overlap(*graphs, chunk_edges=chunk)

    def test_signed_tensor_dtypes_and_noncontiguous_views(self):
        first = debug_graph([-300, 0, 300], {0: [[0, 1, 2, 0], [1, 2, 0, 2]]}, dtype=torch.int16)
        first[LOCAL_EDGE_TYPES[0]].edge_index = first[LOCAL_EDGE_TYPES[0]].edge_index[:, ::2]
        second = debug_graph([-300, 0, 300], {0: [[0, 2], [1, 0]]}, dtype=torch.int32)
        self.assert_overlap(first, second, chunk_edges=1)

    def test_own_temporary_files_cleaned_on_success_and_failure(self):
        graph = debug_graph([1, 2], {0: [[0, 1, 0], [1, 0, 1]]})
        with tempfile.TemporaryDirectory(prefix="debug_overlap_parent_") as root:
            sentinel = Path(root) / "existing-user-file.txt"
            sentinel.touch()
            local_edge_jaccard(graph, graph, chunk_edges=1, temporary_directory=root)
            self.assertEqual(list(Path(root).iterdir()), [sentinel])
            with patch("hiercp.causality_overlap._merge_runs", side_effect=OSError("DEBUG disk failure")):
                with self.assertRaisesRegex(OSError, "DEBUG disk failure"):
                    local_edge_jaccard(graph, graph, chunk_edges=1, temporary_directory=root)
            self.assertEqual(list(Path(root).iterdir()), [sentinel])

    def test_invalid_workspace_or_ids_fail_explicitly(self):
        graph = debug_graph([1, 2], {0: [[0], [1]]})
        for chunk in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "workspace"):
                local_edge_jaccard(graph, graph, chunk_edges=chunk)
        malformed = copy.deepcopy(graph)
        malformed[LOCAL_NODE_TYPES[0]].full_id = torch.tensor([1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "integral full_id"):
            local_edge_jaccard(malformed, graph)
        malformed = copy.deepcopy(graph)
        malformed[LOCAL_EDGE_TYPES[0]].edge_index[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "bounds"):
            local_edge_jaccard(malformed, graph)

    def test_fixed_workspace_bound_and_full_edge_disk_requirement(self):
        graph = debug_graph([1, 2], {0: [[0, 1, 0], [1, 0, 1]], 11: [[0], [1]]})
        self.assertEqual(local_edge_jaccard_workspace_bytes(), 129 * 1024 * 1024)
        self.assertEqual(local_edge_jaccard_workspace_bytes(2), 1024 + 1024 * 1024)
        # Eight edges total, three runs per graph at C=2; duplicates still
        # count toward input/storage preflight rather than being guessed away.
        self.assertEqual(local_edge_jaccard_disk_bytes(graph, graph, chunk_edges=2),
                         72 * 8 + 16384 * (6 + 64))

    def test_disk_preflight_refuses_before_writing_runs_and_cleans_own_directory(self):
        graph = debug_graph([1, 2], {0: [[0], [1]]})
        with tempfile.TemporaryDirectory(prefix="debug_overlap_space_") as root:
            with patch("hiercp.causality_overlap.shutil.disk_usage", return_value=SimpleNamespace(free=0)), \
                    patch("hiercp.causality_overlap._sorted_unique_run") as create:
                with self.assertRaisesRegex(OSError, "Choose scratch storage") as failure:
                    local_edge_jaccard(graph, graph, temporary_directory=root)
                self.assertEqual(failure.exception.errno, errno.ENOSPC)
                create.assert_not_called()
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_disk_exhaustion_after_preflight_is_contextualized_and_cleaned(self):
        graph = debug_graph([1, 2], {0: [[0, 1, 0], [1, 0, 1]]})
        with tempfile.TemporaryDirectory(prefix="debug_overlap_full_") as root:
            with patch("hiercp.causality_overlap._merge_runs", side_effect=OSError(errno.ENOSPC, "DEBUG disk full")):
                with self.assertRaisesRegex(OSError, "No metric was substituted") as failure:
                    local_edge_jaccard(graph, graph, chunk_edges=1, temporary_directory=root)
                self.assertEqual(failure.exception.errno, errno.ENOSPC)
            self.assertEqual(list(Path(root).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
