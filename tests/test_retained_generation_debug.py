"""DEBUG generation evidence tests on a tiny synthetic volume; no training."""
import copy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from hiercp.common import build_candidate_pool, choose_source_tumor
from hiercp.pipeline import _is_retained_generation_row


class RetainedGenerationDebugTests(unittest.TestCase):
    def setUp(self):
        shape = (10, 10, 10)
        label = np.ones(shape, dtype=np.int16)
        label[1:9, 1:9, 1:9] = 2
        image = np.zeros(shape, dtype=np.float32)
        case = SimpleNamespace(image=image, label=label, shape=shape,
            spacing=np.asarray([0.7, 0.7, 2.5], dtype=np.float32),
            paths=SimpleNamespace(case_id="DEBUG_zero_placement"))
        source, _, _ = choose_source_tumor(image, label, tumor_label=2,
            rng=np.random.default_rng(3), selection="random", pad=2)
        self.generation = {"no_placement_policy": "retain_original", "num_candidates": 128,
            "max_draws": 4000, "min_liver_coverage": .85, "occupied_clearance_vox": 2,
            "min_center_separation_mm": 0.0, "min_center_separation_vox": 12.0}
        pool = {}
        # The selected source patch fills the tiny DEBUG volume: every possible
        # placement overlaps the original. Use the actual complete search, not
        # fabricated diagnostics or a mocked zero-candidate result.
        candidates, _ = build_candidate_pool(case, source, placement_mask=label == 1,
            full_organ_mask=label > 0, occupied_mask=label == 2,
            organ_distance=np.ones(shape, dtype=np.float32),
            rng=np.random.default_rng(4), num_candidates=128,
            required_candidates=128, diagnostics=pool,
            **{key: self.generation[key] for key in ("max_draws", "min_liver_coverage",
                "occupied_clearance_vox", "min_center_separation_mm", "min_center_separation_vox")})
        self.assertEqual(candidates, [])
        self.assertTrue(pool["fullsearch_exhausted"])
        self.pool = pool
        # Persisted CSV row fields are strings, while JSON diagnostic counters
        # remain numbers/booleans exactly as emitted by the candidate search.
        self.row = {"status": "retained_original", "requested": "1", "pasted": "0",
            "source_component": str(source.component_id), "candidate_search_json": json.dumps(pool)}

    def changed_pool(self, **changes):
        return {**self.row, "candidate_search_json": json.dumps({**self.pool, **changes})}

    def test_actual_zero_placement_evidence_is_accepted(self):
        self.assertTrue(_is_retained_generation_row(self.row, self.generation))

    def test_strict_and_missing_legacy_policy_do_not_accept_retained_rows(self):
        for policy in ("error", None):
            generation = copy.deepcopy(self.generation)
            if policy is None:
                generation.pop("no_placement_policy")
            else:
                generation["no_placement_policy"] = policy
            self.assertFalse(_is_retained_generation_row(self.row, generation))

    def test_nonzero_and_incomplete_or_excluded_searches_are_rejected(self):
        for changes in ({"accepted": 1}, {"accepted": 6}, {"accepted": 128},
                        {"fullsearch_exhausted": False}, {"exhaustive_used": False},
                        {"excluded_center_count": 1}, {"source_voxels": 0},
                        {"source_component": self.pool["source_component"] + 1},
                        {"format": "DEBUG_wrong_format"}, {"search_version": "DEBUG_stale"}):
            with self.subTest(changes=changes):
                self.assertFalse(_is_retained_generation_row(self.changed_pool(**changes), self.generation))
        for key in ("accepted", "source_component", "source_voxels", "excluded_center_count",
                    "required_candidates", "target_candidates", "max_draws", "occupied_clearance_vox"):
            with self.subTest(boolean_counter=key):
                self.assertFalse(_is_retained_generation_row(self.changed_pool(**{key: False}), self.generation))

    def test_each_changed_cp_search_condition_is_rejected(self):
        for key, value in (("max_draws", 4001), ("min_liver_coverage", .8),
                           ("occupied_clearance_vox", 1), ("min_center_separation_mm", 1.0),
                           ("min_center_separation_vox", 11.0), ("target_candidates", 127),
                           ("required_candidates", 7)):
            with self.subTest(key=key):
                self.assertFalse(_is_retained_generation_row(self.changed_pool(**{key: value}), self.generation))

    def test_changed_row_counts_status_and_broken_json_are_rejected(self):
        for changes in ({"status": "no_candidate"}, {"requested": "0"}, {"pasted": "1"},
                        {"source_component": "999"}, {"candidate_search_json": "not JSON"},
                        {"requested": True}, {"pasted": False}, {"source_component": True}):
            with self.subTest(changes=changes):
                self.assertFalse(_is_retained_generation_row({**self.row, **changes}, self.generation))


if __name__ == "__main__":
    unittest.main()
