"""DEBUG zero-placement accounting only; structural fixtures, no model training."""
import copy
import json
import unittest
from unittest.mock import patch

import test_cache_recovery_debug as fixture_module
from hiercp import cache
from hiercp.common import CandidatePreparationError, CANDIDATE_SEARCH_VERSION
from hiercp.data import summarize_cache_usage, list_cache_files, split_files_from_cache


class NoPlacementCacheDebugTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.CacheMigrationDebugTests(
            "test_migration_preserves_sources_and_only_relabels_verified_artifact_metadata")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.request = {**self.fixture.kwargs, "no_placement_policy": "retain_original"}
        self.calls = []

    def error(self, case_id, sample_index, *, accepted=0, **changes):
        config = self.request
        pool = {"search_version": CANDIDATE_SEARCH_VERSION, "source_component": 1,
            "source_anchor": [1, 2, 3], "source_patch_shape": [2, 2, 2], "source_voxels": 8,
            "target_candidates": config["candidate_pool_size"],
            "required_candidates": config["total_candidates"] - 1,
            "max_draws": config["max_draws"], "min_liver_coverage": config["min_liver_coverage"],
            "occupied_clearance_vox": config["occupied_clearance_vox"],
            "min_center_separation_mm": config["min_center_separation_mm"],
            "min_center_separation_vox": config.get("min_center_separation_vox", 0.0),
            "fullsearch_exhausted": True, "exhaustive_used": True, "accepted": accepted,
            "required_candidates_met": False, "excluded_center_count": 0}
        details = {"candidate_search_version": CANDIDATE_SEARCH_VERSION,
            "case_id": case_id, "sample_index": sample_index, "source_component": 1,
            "pool": pool, "rejected_centers": [], "geometry_rejections": [],
            "curriculum_failure": None}
        details.update(changes)
        return CandidatePreparationError("insufficient_valid_candidate_pool", details)

    @staticmethod
    def jobs(*, tasks, function, commit, **kwargs):
        # All real DEBUG NIfTI cases are visited. Resource scheduling is tested
        # independently; this fixture isolates publication and retry behavior.
        for task in tasks:
            commit(function(task))

    def prepare(self, blocked=None, *, accepted=0):
        blocked = {("donor", 0)} if blocked is None else blocked

        def sample(case, bank, regions, *, sample_index, **kwargs):
            key = (case.paths.case_id, sample_index)
            self.calls.append(key)
            if key in blocked:
                raise self.error(*key, accepted=accepted)
            return self.fixture.sample(*key)

        with patch("hiercp.cache.build_training_sample", side_effect=sample), \
             patch("hiercp.preparation_runtime.run_case_jobs", side_effect=self.jobs):
            return cache.prepare_hierarchical_cache(**self.request)

    def test_all_attempts_preserved_without_materializing_a_fake_graph(self):
        self.prepare()
        root = self.request["cache_dir"]
        index = cache.validate_cache_publication(root)
        self.assertEqual(set(self.calls), {("donor", 0), ("donor", 1), ("pending", 0), ("pending", 1)})
        self.assertEqual(index["expected_entries"], 4)
        self.assertEqual(len(index["entries"]), 3)
        self.assertEqual(len(index["no_placement_entries"]), 1)
        self.assertFalse((root / "donor__000.pt").exists())
        report = summarize_cache_usage(root)
        self.assertEqual(report["source_patient_case_ids"], ["absent", "donor", "pending"])
        self.assertEqual(report["materialized_sample_ratio"], .75)
        self.assertEqual(report["resolved_sample_ratio"], 1.0)
        self.assertEqual(report["no_placement_sample_count"], 1)
        self.assertFalse(report["subset_active"])
        train, val = split_files_from_cache(list_cache_files(root))
        self.assertEqual((len(train), len(val)), (1, 2))
        for entry in index["entries"]:
            self.assertEqual(len(cache._torch_load_cpu(root / entry["path"])["target_locals"]), 8)
        config = json.loads((root / "config.json").read_text())
        self.assertEqual(config["candidate_pool_size"], 128)
        self.assertEqual(config["total_candidates"], 8)
        self.calls.clear()
        self.prepare()
        self.assertEqual(self.calls, [])

    def test_legacy_error_policy_still_refuses_zero_placement(self):
        self.request.pop("no_placement_policy")
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.prepare()
        self.assertFalse((self.request["cache_dir"] / "complete.json").exists())

    def test_nonzero_pool_shortage_is_not_accepted(self):
        for accepted in range(1, 7):
            with self.subTest(accepted=accepted):
                self.request["cache_dir"] = self.fixture.root / f"DEBUG_nonzero_{accepted}"
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    self.prepare(accepted=accepted)
                self.assertFalse((self.request["cache_dir"] / "complete.json").exists())

    def test_only_exact_unexcluded_full_search_zero_proofs_are_recognized(self):
        kwargs = dict(case_id="donor", sample_index=0, required_candidates=7)
        self.assertIsNotNone(cache.no_placement_evidence(self.error("donor", 0), **kwargs))
        mutations = (
            lambda d: d["pool"].update(fullsearch_exhausted=False),
            lambda d: d["pool"].update(exhaustive_used=False),
            lambda d: d["pool"].update(excluded_center_count=1),
            lambda d: d["pool"].update(source_component=2),
            lambda d: d["pool"].update(required_candidates=3),
            lambda d: d["pool"].update(accepted=False),
            lambda d: d.update(case_id="different"),
            lambda d: d.update(sample_index=1),
            lambda d: d.update(geometry_rejections=[{"reason": "missing_context"}]),
            lambda d: d.update(curriculum_failure={"reason": "quality"}),
        )
        for mutate in mutations:
            error = self.error("donor", 0)
            mutate(error.diagnostics)
            self.assertIsNone(cache.no_placement_evidence(error, **kwargs))
        for error in (OSError("DEBUG I/O"), MemoryError("DEBUG OOM"),
                      CandidatePreparationError("positive_canonical_geometry_unavailable", {})):
            self.assertIsNone(cache.no_placement_evidence(error, **kwargs))

    def test_no_placement_index_tampering_is_rejected_even_after_rehashing_sidecar(self):
        self.prepare()
        root = self.request["cache_dir"]
        original = json.loads((root / "index.json").read_text())
        for change in ("remove", "duplicate", "change_source"):
            with self.subTest(change=change):
                altered = copy.deepcopy(original)
                if change == "remove":
                    altered["no_placement_entries"] = []
                elif change == "duplicate":
                    altered["no_placement_entries"] *= 2
                else:
                    altered["no_placement_entries"][0]["source_label_sha256"] = "a" * 64
                cache._atomic_json_save(altered, root / "index.json")
                complete = json.loads((root / "complete.json").read_text())
                complete["index_sha256"] = cache._sha256_file(root / "index.json")
                cache._atomic_json_save(complete, root / "complete.json")
                with self.assertRaisesRegex(ValueError, "no-placement"):
                    cache.validate_cache_publication(root)

    def test_all_zero_outcomes_do_not_create_training_files(self):
        self.prepare({(case, index) for case in ("donor", "pending") for index in range(2)})
        root = self.request["cache_dir"]
        self.assertEqual(cache.validate_cache_publication(root)["entries"], [])
        self.assertEqual(summarize_cache_usage(root)["materialized_sample_ratio"], 0.0)
        with self.assertRaisesRegex(RuntimeError, "empty"):
            list_cache_files(root)


if __name__ == "__main__":
    unittest.main()
