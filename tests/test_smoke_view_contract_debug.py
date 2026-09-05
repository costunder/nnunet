"""DEBUG-only regression for the narrowly justified identical-view exception."""

import importlib.util
from types import SimpleNamespace
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "scipy"))
if AVAILABLE:
    import numpy as np
    from tools.smoke import _identical_context_is_complete


@unittest.skipUnless(AVAILABLE, "DEBUG smoke-contract tests require torch/scipy")
class IdenticalViewContractDebugTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.column_stack([np.arange(6), np.zeros(6), np.zeros(6)])
        self.config = SimpleNamespace(sample_context_nodes=2, sample_interface_radius_mm=.01,
                                      sample_hops=1, sample_hop_radius_mm=1.01)

    def test_debug_accepts_exact_mandatory_closure_with_no_free_seed_slots(self):
        self.assertTrue(_identical_context_is_complete(
            np.array([0, 1, 2]), np.array([0, 1, 2]), self.positions,
            self.positions[:2], self.config,
        ))

    def test_debug_rejects_truncated_or_extra_nodes_even_with_exhausted_budget(self):
        for ids in (np.array([0, 1]), np.array([0, 1, 2, 3]), np.array([0, 1, 1])):
            self.assertFalse(_identical_context_is_complete(
                ids, ids, self.positions, self.positions[:2], self.config,
            ))

    def test_debug_rejects_identical_partial_views_when_random_choice_remains(self):
        self.assertFalse(_identical_context_is_complete(
            np.array([0, 1, 2]), np.array([0, 1, 2]), self.positions,
            self.positions[:1], self.config,
        ))

    def test_debug_both_views_must_preserve_complete_node_set(self):
        self.assertTrue(_identical_context_is_complete(
            np.arange(6), np.arange(6), self.positions, self.positions[:1], self.config,
        ))
        self.assertFalse(_identical_context_is_complete(
            np.arange(6), np.arange(5), self.positions, self.positions[:1], self.config,
        ))


if __name__ == "__main__":
    unittest.main()
