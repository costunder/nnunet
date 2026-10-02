"""CPU metadata/integrity UNIT checks; no model execution or accuracy claims."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from hiercp_v1x.contracts import (
    ContractError, EVALUATION_IDENTITY_FIELDS, STAGES, TARGET_CONTRACT,
    V1_ARCHIVE_SHA256, canonical_hash, compare_reports, config_diff,
    make_run_contract, make_stage_config, validate_resume, validate_transition,
    verify_archive, resolve_execution_config, validate_stage_config,
)


ROOT = Path(__file__).resolve().parents[1]


class ProgressiveV1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive = verify_archive(ROOT)
        cls.base = cls.archive["base_config"]

    def test_exact_preserved_archive_and_baseline_config(self):
        self.assertEqual(self.archive["verified_files"], 202)
        self.assertEqual(self.archive["archive_sha256"], V1_ARCHIVE_SHA256)
        baseline = make_stage_config(self.base, "v1.0")
        self.assertEqual(baseline, self.base)
        self.assertIsNot(baseline, self.base)
        self.assertEqual(baseline["seed"], 42)
        self.assertEqual(baseline["training"]["epochs"], 40)
        self.assertEqual(baseline["cache"]["candidate_pool_size"], 128)
        self.assertEqual(baseline["cache"]["total_candidates"], 8)

    def test_tampered_archive_rejected_before_extraction(self):
        original_read = Path.read_bytes
        def altered(path):
            raw = original_read(path)
            return raw + b"tamper" if path.name == "pipeline_v1_source.zip" else raw
        with patch.object(Path, "read_bytes", altered):
            with self.assertRaisesRegex(ContractError, "archive identity"):
                verify_archive(ROOT)

    def test_tampered_manifest_rejected(self):
        original_read = Path.read_bytes
        def altered(path):
            raw = original_read(path)
            return raw + b" " if path.name == "manifest.json" else raw
        with patch.object(Path, "read_bytes", altered):
            with self.assertRaisesRegex(ContractError, "manifest"):
                verify_archive(ROOT)

    def test_each_transition_changes_one_declared_factor(self):
        for old, new in zip(STAGES, list(STAGES)[1:]):
            before = make_stage_config(self.base, old)
            after = make_stage_config(self.base, new)
            result = validate_transition(old, before, new, after)
            self.assertEqual(result["changed_factor"], STAGES[new].changed_factor)
            expected_keys = {"model.checkpoint_dense_encoder"}
            self.assertEqual(set(result["config_diff"]), expected_keys if new == "v1.1" else set())
            self.assertTrue(after["model"]["checkpoint_local_blocks"])
            self.assertFalse(after["model"]["checkpoint_dense_encoder"])

    def test_loss_graph_batch_new_keys_and_target_changes_rejected(self):
        for key, value in (("pairwise_weight", 0.5), ("epochs", 2), ("batch_size", 4)):
            base = copy.deepcopy(self.base)
            base["training"][key] = value
            with self.assertRaises(ContractError):
                make_stage_config(base, "v1.1")
        for mutate in (
            lambda c: c["graph"].update(sample_context_nodes=48),
            lambda c: c["cache"].update(candidate_pool_size=16),
            lambda c: c.update(target_contract="observed_P_U"),
            lambda c: c["model"].update(hidden_dim=64),
            lambda c: c["model"].update(new_silent_scale=1),
        ):
            base = copy.deepcopy(self.base)
            mutate(base)
            with self.assertRaises(ContractError):
                make_stage_config(base, "v1.2")

    def test_nonadjacent_transition_and_undeclared_change_rejected(self):
        before = make_stage_config(self.base, "v1.0")
        after = make_stage_config(self.base, "v1.2")
        with self.assertRaisesRegex(ContractError, "compared"):
            validate_transition("v1.0", before, "v1.2", after)
        before = make_stage_config(self.base, "v1.1")
        after["training"]["val_fraction"] = 0.3
        with self.assertRaisesRegex(ContractError, "undeclared"):
            validate_transition("v1.1", before, "v1.2", after)

    def test_bool_integer_type_change_is_not_equal(self):
        self.assertIn("value", config_diff({"value": True}, {"value": 1}))

    def test_measured_execution_lock_is_same_for_every_stage(self):
        calibration = dict(selected_batch_size=32, selected_num_workers=8,
                           baseline_calibration_sha256="d" * 64)
        previous = None
        for stage in STAGES:
            config = resolve_execution_config(make_stage_config(self.base, stage), calibration)
            validate_stage_config(config, stage, execution_lock=calibration)
            self.assertEqual(config["training"]["batch_size"], 32)
            self.assertEqual(config["training"]["num_workers"], 8)
            if previous is not None:
                validate_transition(previous[0], previous[1], stage, config, execution_lock=calibration)
            previous = (stage, config)
        different = copy.deepcopy(config)
        different["training"]["batch_size"] = 16
        with self.assertRaisesRegex(ContractError, "undeclared"):
            validate_stage_config(different, "v1.3", execution_lock=calibration)

    def test_invalid_calibration_and_batch_mismatch_rejected(self):
        config = make_stage_config(self.base, "v1.0")
        for calibration in ({}, dict(selected_batch_size=0, selected_num_workers=8),
                            dict(selected_batch_size=32, selected_num_workers=-1),
                            dict(selected_batch_size=True, selected_num_workers=8)):
            with self.assertRaises(ContractError):
                resolve_execution_config(config, calibration)
        calibration = dict(selected_batch_size=32, selected_num_workers=8)
        resolved = resolve_execution_config(config, calibration)
        with self.assertRaisesRegex(ContractError, "Physical batch"):
            make_run_contract("v1.0", resolved, cache_identity={"sha": "a"},
                              source_identity={"sha": "b"}, evaluation_identity={"sha": "c"},
                              physical_batch_size=16, debug=True, execution_lock=calibration)

    def contract(self, stage="v1.0"):
        return make_run_contract(stage, make_stage_config(self.base, stage),
                                 cache_identity={"sha256": "a" * 64},
                                 source_identity={"sha256": "b" * 64},
                                 evaluation_identity={"split_sha256": "c" * 64},
                                 physical_batch_size=32, debug=True)

    def test_same_stage_resume_only(self):
        original = self.contract()
        validate_resume(original, copy.deepcopy(original))
        changed = self.contract("v1.1")
        with self.assertRaisesRegex(ContractError, "resume contract mismatch"):
            validate_resume(changed, original)

    def test_resume_tampering_and_batch_change_rejected(self):
        original = self.contract()
        broken = copy.deepcopy(original)
        broken["physical_batch_size"] = 16
        with self.assertRaisesRegex(ContractError, "hash mismatch"):
            validate_resume(original, broken)
        raw = {k: v for k, v in broken.items() if k != "contract_sha256"}
        broken["contract_sha256"] = canonical_hash(raw)
        with self.assertRaisesRegex(ContractError, "resume contract mismatch"):
            validate_resume(original, broken)

    def report(self, stage, mrr=0.7):
        evaluation = {k: "same" for k in EVALUATION_IDENTITY_FIELDS}
        evaluation.update(target_contract=TARGET_CONTRACT,
                          source_archive_sha256=V1_ARCHIVE_SHA256,
                          denominator={"patients": 26, "ranking_samples": 36})
        return dict(stage=stage, complete=True, debug=True, evaluation=evaluation,
                    metrics={"mrr": mrr, "top1": 0.5, "rank_loss": 0.6})

    def test_comparison_requires_original_and_predecessor(self):
        baseline = self.report("v1.0")
        candidate = self.report("v1.2", 0.8)
        with self.assertRaisesRegex(ContractError, "predecessor"):
            compare_reports(baseline, candidate)
        result = compare_reports(baseline, candidate, self.report("v1.1", 0.75))
        self.assertAlmostEqual(result["metric_deltas"]["baseline"]["mrr"], 0.1)
        self.assertAlmostEqual(result["metric_deltas"]["predecessor"]["mrr"], 0.05)
        self.assertFalse(result["automatic_promotion"])

    def test_changed_denominator_mask_gt_or_split_cannot_be_compared(self):
        baseline = self.report("v1.0")
        for key in EVALUATION_IDENTITY_FIELDS:
            altered = self.report("v1.1")
            altered["evaluation"][key] = {"patients": 1} if key == "denominator" else "different"
            with self.assertRaises(ContractError):
                compare_reports(baseline, altered)

    def test_missing_nonfinite_metric_and_incomplete_evaluation_rejected(self):
        baseline = self.report("v1.0")
        for value in (float("nan"), float("inf"), True):
            altered = self.report("v1.1")
            altered["metrics"]["mrr"] = value
            with self.assertRaises(ContractError):
                compare_reports(baseline, altered)
        altered = self.report("v1.1")
        altered["complete"] = False
        with self.assertRaises(ContractError):
            compare_reports(baseline, altered)
        altered = self.report("v1.1")
        del altered["metrics"]["rank_loss"]
        with self.assertRaisesRegex(ContractError, "metric set"):
            compare_reports(baseline, altered)


if __name__ == "__main__":
    unittest.main()
