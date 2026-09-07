"""DEBUG pool-search orchestration fixtures, not medical/model validation.

The real candidate search covers the complete synthetic volume. Graph and
curriculum doubles isolate retry control flow; no production defaults change.
"""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hiercp import cache, common
from hiercp.spatial import EmptyCanonicalNodeError
from tests.test_candidate_diagnostics_debug import fixture


class CandidatePoolReuseDebugTests(unittest.TestCase):
    def setUp(self):
        self.case, self.source, self.pool_options = fixture(size=7, tumor_width=1)
        self.regions = SimpleNamespace(full_organ_mask=self.pool_options["full_organ_mask"],
                                       organ_depth=self.pool_options["organ_distance"])
        self.options = dict(sample_index=0, split_name="debug", graph_config=SimpleNamespace(),
            liver_label=1, tumor_label=2, source_selection="largest", source_pad=1,
            total_candidates=8, candidate_pool_size=128, easy_fraction=.34,
            inter_fraction=.33, intra_fraction=.33, max_draws=4000,
            min_liver_coverage=.85, occupied_clearance_vox=0,
            min_center_separation_mm=0., ct_clip=(-200., 200.), seed=42)

    def run_sample(self):
        return cache.build_training_sample(self.case, SimpleNamespace(), self.regions, **self.options)

    def test_debug_initial_exhausted_short_pool_is_not_searched_again(self):
        self.options["min_center_separation_mm"] = 10.
        with patch.object(cache, "choose_source_tumor", return_value=(self.source, None, None)), \
             patch.object(cache, "prepare_local_source"), \
             patch.object(cache, "build_candidate_pool", wraps=common.build_candidate_pool) as pool, \
             patch.object(cache, "build_training_specs") as specs, \
             patch.object(cache, "build_local_graph") as graph:
            with self.assertRaises(common.CandidatePreparationError) as raised:
                self.run_sample()
        self.assertEqual(pool.call_count, 1)
        self.assertEqual(raised.exception.reason, "insufficient_valid_candidate_pool")
        diagnostics = raised.exception.diagnostics
        self.assertTrue(diagnostics["pool"]["fullsearch_exhausted"])
        self.assertEqual(diagnostics["pool"]["required_candidates"], 7)
        self.assertEqual(diagnostics["pool"]["target_candidates"], 128)
        self.assertEqual(diagnostics["pool_search_reuses"], 1)
        specs.assert_not_called()
        graph.assert_not_called()

    def test_debug_completed_exhaustive_pool_reused_for_curriculum_failures(self):
        self.options["max_draws"] = 0
        error = common.CandidatePreparationError("debug_curriculum_failure", {})
        with patch.object(cache, "choose_source_tumor", return_value=(self.source, None, None)), \
             patch.object(cache, "prepare_local_source"), \
             patch.object(cache, "build_candidate_pool", wraps=common.build_candidate_pool) as pool, \
             patch.object(cache, "build_training_specs", side_effect=error) as specs:
            with self.assertRaises(common.CandidatePreparationError) as raised:
                self.run_sample()
        self.assertEqual(pool.call_count, 1)
        self.assertEqual(specs.call_count, 2)
        self.assertEqual(raised.exception.reason, "curriculum_or_negative_geometry_search_exhausted")
        self.assertEqual(raised.exception.diagnostics["pool_search_reuses"], 2)
        self.assertGreaterEqual(raised.exception.diagnostics["pool"]["accepted"], 7)

    def test_debug_short_legacy_pool_still_gets_one_exhaustive_extension(self):
        requests = []

        def search(case, source, *, rng, **kwargs):
            requests.append({"source": source, "rng": copy.deepcopy(rng.bit_generator.state),
                             "force_exhaustive": kwargs.get("force_exhaustive", False),
                             "excluded": frozenset(kwargs.get("excluded_centers", ()))})
            return common.build_candidate_pool(case, source, rng=rng, **kwargs)

        error = common.CandidatePreparationError("debug_curriculum_failure", {})
        with patch.object(cache, "choose_source_tumor", return_value=(self.source, None, None)), \
             patch.object(cache, "prepare_local_source"), \
             patch.object(cache, "build_candidate_pool", side_effect=search), \
             patch.object(cache, "build_training_specs", side_effect=error) as specs:
            with self.assertRaises(common.CandidatePreparationError) as raised:
                self.run_sample()
        self.assertEqual(len(requests), 2)
        self.assertFalse(requests[0]["force_exhaustive"])
        self.assertTrue(requests[1]["force_exhaustive"])
        self.assertEqual(requests[0]["rng"], requests[1]["rng"])
        self.assertEqual(requests[0]["excluded"], requests[1]["excluded"])
        self.assertTrue(all(row["source"] is self.source for row in requests))
        self.assertEqual(specs.call_count, 2)
        self.assertEqual(raised.exception.diagnostics["pool_search_reuses"], 1)

    def test_debug_new_geometry_exclusion_requires_new_search(self):
        candidates, _ = common.build_candidate_pool(self.case, self.source,
            rng=np.random.default_rng(5), required_candidates=7, **self.pool_options)
        # Seven candidates are an explicit orchestration double, not a subset
        # setting or clinical output. Their graph failure changes the request.
        initial = candidates[:7]
        searches = []

        def search(case, source, *, diagnostics, **kwargs):
            excluded = frozenset(kwargs.get("excluded_centers", ()))
            searches.append(excluded)
            result = [candidate for candidate in initial if candidate.center not in excluded]
            diagnostics.clear()
            diagnostics.update(fullsearch_exhausted=True, target_candidates_met=False,
                               accepted=len(result), excluded_center_count=len(excluded))
            return result, np.zeros(case.shape)

        def specs(case, source, viable, *args, **kwargs):
            centers = [source.anchor_center] + [candidate.center for candidate in viable]
            return [SimpleNamespace(center=center, difficulty=index != 0, corruption=0)
                    for index, center in enumerate(centers)]

        with patch.object(cache, "choose_source_tumor", return_value=(self.source, None, None)), \
             patch.object(cache, "prepare_local_source"), \
             patch.object(cache, "build_candidate_pool", side_effect=search), \
             patch.object(cache, "build_training_specs", side_effect=specs), \
             patch.object(cache, "build_local_graph", side_effect=[object(), EmptyCanonicalNodeError("debug_node", (0, 3))]):
            with self.assertRaises(common.CandidatePreparationError) as raised:
                self.run_sample()
        self.assertEqual(searches, [frozenset(), frozenset({initial[0].center})])
        self.assertEqual(raised.exception.reason, "insufficient_valid_candidate_pool")
        self.assertEqual(raised.exception.diagnostics["pool_search_reuses"], 0)
        self.assertEqual(raised.exception.diagnostics["rejected_centers"], [list(initial[0].center)])


if __name__ == "__main__":
    unittest.main()
