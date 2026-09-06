"""DEBUG accounting fixtures only; no source files, training or metrics."""
import unittest
from unittest import mock

from hiercp.data import summarize_cache_usage


class DonorUsageDebugTests(unittest.TestCase):
    def values(self):
        return ({"train_case_ids": ["DEBUG_A", "DEBUG_NEG"], "val_case_ids": ["DEBUG_VAL"],
                 "selected_case_ids": ["DEBUG_A", "DEBUG_NEG", "DEBUG_VAL"], "samples_per_case": 2,
                 "donor_eligibility": {"eligible_case_ids": ["DEBUG_A", "DEBUG_VAL"],
                                       "ineligible_case_ids": ["DEBUG_NEG"]},
                 "donor_contract_sha256": "a" * 64},
                {"entries": [{"case_id": case, "path": f"{case}__{i:03d}.pt"}
                             for case in ("DEBUG_A", "DEBUG_VAL") for i in range(2)]})

    def invoke(self, config, index, **kwargs):
        with mock.patch("hiercp.data.load_cache_config", return_value=config), \
             mock.patch("hiercp.data.load_cache_index", return_value=index):
            return summarize_cache_usage("DEBUG_NOT_READ", **kwargs)

    def test_full_patient_population_and_donor_denominator_are_distinct(self):
        config, index = self.values()
        report = self.invoke(config, index)
        self.assertEqual(report["requested_case_count"], 3)
        self.assertEqual(report["configured_selected_case_count"], 3)
        self.assertEqual(report["eligible_source_case_count"], 2)
        self.assertEqual(report["ineligible_source_case_count"], 1)
        self.assertEqual(report["actually_used_case_count"], 2)
        self.assertEqual(report["materialized_sample_ratio"], 1.0)
        self.assertEqual(report["expected_selected_sample_count"], 4)
        self.assertFalse(report["subset_active"])

    def test_missing_eligible_graph_is_not_counted_complete(self):
        config, index = self.values()
        index["entries"].pop()
        self.assertEqual(self.invoke(config, index)["materialized_sample_ratio"], 0.75)

    def test_legacy_contract_keeps_all_case_requirement(self):
        config, index = self.values()
        del config["donor_eligibility"]
        del config["donor_contract_sha256"]
        report = self.invoke(config, index)
        self.assertEqual(report["expected_selected_sample_count"], 6)
        self.assertEqual(report["ineligible_source_case_count"], 0)
        self.assertFalse(report["eligibility_is_not_patient_subset"])

    def test_invalid_partition_fails(self):
        config, index = self.values()
        config["donor_eligibility"]["eligible_case_ids"].append("DEBUG_NEG")
        with self.assertRaises(ValueError):
            self.invoke(config, index)


if __name__ == "__main__":
    unittest.main()
