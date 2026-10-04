"""CPU metadata contract checks using the recorded actual-CT/CUDA reference.

These tests do not construct medical inputs, generate neural predictions, or
execute either the native graph or the smaller graph on a GPU.
"""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from tools.verify_v14_learning_replay import verify_reference


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "validation" / "v14_matched_learning_20261004"
REFERENCE = EVIDENCE / "native_reference"
FIXTURE = EVIDENCE / "native_fixture"


class LearningReplayReferenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = {
            "report": REFERENCE / "report.json",
            "contract": REFERENCE / "execution_contract.json",
            "fixture_manifest": FIXTURE / "fixture_manifest.json",
        }
        cls.recorded = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in cls.paths.items()
        }

    def setUp(self):
        self.inputs = deepcopy(self.recorded)

    def verify(self):
        return verify_reference(self.inputs["report"], self.inputs["contract"],
                                self.inputs["fixture_manifest"])

    def reject(self, input_name, field_path, value):
        self.inputs = deepcopy(self.recorded)
        node = self.inputs[input_name]
        for key in field_path[:-1]:
            node = node[key]
        node[field_path[-1]] = value
        with self.assertRaises(ValueError):
            self.verify()

    def test_recorded_reference_accepted_without_mutating_metadata(self):
        before_inputs = deepcopy(self.inputs)
        before_bytes = {name: path.read_bytes() for name, path in self.paths.items()}
        # The execution contract was saved before the run. Its requested flags
        # bind the planned run; the completed report supplies actual evidence.
        self.assertFalse(self.inputs["contract"]["actual_CT_verified"])
        self.assertFalse(self.inputs["contract"]["actual_CUDA_verified"])
        protocol = self.verify()
        self.assertEqual(protocol["reference_updates"], 2)
        self.assertEqual(protocol["physical_sample_batch"], 2)
        self.assertEqual(protocol["train_cases"], ["liver_5", "liver_6"])
        self.assertEqual(protocol["validation_cases"], ["liver_31"])
        self.assertEqual(protocol["initial_neural_sha256"],
                         self.inputs["report"]["shared_initial_state_sha256"])
        self.assertEqual(self.inputs, before_inputs)
        self.assertEqual(before_bytes, {name: path.read_bytes() for name, path in self.paths.items()})

    def test_actual_ct_and_cuda_evidence_required(self):
        for name, key in (("report", "actual_CT"), ("report", "actual_CUDA"),
                          ("contract", "actual_CT_requested"), ("contract", "actual_CUDA_requested")):
            with self.subTest(input=name, field=key):
                self.reject(name, [key], False)

    def test_debug_evidence_cannot_be_claimed_as_production(self):
        for name in ("report", "contract", "fixture_manifest"):
            for key in ("debug", "full_training", "production_ready"):
                with self.subTest(input=name, field=key):
                    self.reject(name, [key], key != "debug")
        for name in ("report", "contract"):
            with self.subTest(input=name, field="full_evaluation"):
                self.reject(name, ["full_evaluation"], True)
        for key in ("training_quality_verified", "graph_size_quality_verified"):
            with self.subTest(input="report", field=key):
                self.reject("report", [key], True)

    def test_wrong_fixture_cannot_reuse_native_results(self):
        for name in ("report", "fixture_manifest"):
            with self.subTest(input=name):
                self.reject(name, ["fixture_sha256"], "0" * 64)

    def test_matching_invalid_fixture_hash_rejected(self):
        self.inputs["report"]["fixture_sha256"] = "x" * 64
        self.inputs["fixture_manifest"]["fixture_sha256"] = "x" * 64
        with self.assertRaises(ValueError):
            self.verify()

    def test_architecture_and_original_graph_must_match(self):
        changes = (
            ("report", ["original_model", "hidden_dim"], 64),
            ("contract", ["original_model", "local_layers"], 2),
            ("fixture_manifest", ["config", "model", "heads"], 2),
            ("report", ["original_graph", "adaptive_roi_margin_mm"], 10.0),
            ("contract", ["original_graph", "context_outer_radius_mm"], 10.0),
            ("fixture_manifest", ["config", "graph", "patch_size"], 32),
        )
        for name, field, value in changes:
            with self.subTest(input=name, field=field):
                self.reject(name, field, value)

    def test_native_loss_and_runtime_must_match(self):
        changes = (
            ("contract", ["original_loss", "lr"], 0.001),
            ("contract", ["original_loss", "consistency_weight"], 0.0),
            ("contract", ["original_runtime", "allow_tf32"], True),
            ("report", ["runtime", "matmul_tf32"], True),
            ("report", ["runtime", "cudnn_deterministic"], False),
        )
        for name, field, value in changes:
            with self.subTest(input=name, field=field):
                self.reject(name, field, value)

    def test_cohort_and_batch_contract_must_match(self):
        changes = (
            ("contract", ["train_cases"], ["liver_5"]),
            ("contract", ["held_out_cases"], ["liver_6"]),
            ("fixture_manifest", ["validation_cases"], ["liver_5"]),
            ("report", ["physical_sample_batch"], 1),
            ("report", ["physical_graph_batch"], 8),
            ("contract", ["effective_sample_batch"], 4),
            ("contract", ["gradient_accumulation_steps"], 2),
            ("contract", ["physical_candidate_graph_batch"], 8),
            ("report", ["candidates_per_sample"], 4),
            ("contract", ["candidates_per_sample"], 4),
            ("report", ["candidate_pool"], 64),
            ("contract", ["candidate_pool_size"], 64),
            ("contract", ["train_cases"], ["liver_6", "liver_5"]),
        )
        for name, field, value in changes:
            with self.subTest(input=name, field=field):
                self.reject(name, field, value)

    def test_missing_or_invalid_native_learning_evidence_rejected(self):
        changes = (
            (["branches", "original_v1", "updates"], []),
            (["branches", "original_v1", "updates", 0, "loss"], float("nan")),
            (["branches", "original_v1", "updates", 0, "gradient_groups", "L0"], 0.0),
            (["branches", "original_v1", "updates", 0, "gradient_groups", "L1"], float("inf")),
            (["branches", "original_v1", "trainable_parameters"], 0),
        )
        for field, value in changes:
            with self.subTest(field=field):
                self.reject("report", field, value)

    def test_missing_native_core_gradient_rejected(self):
        del self.inputs["report"]["branches"]["original_v1"]["updates"][0]["gradient_groups"]["CNN"]
        with self.assertRaises(ValueError):
            self.verify()

    def test_validation_overlap_rejected_even_with_matching_cohort_fields(self):
        self.inputs["fixture_manifest"]["validation_cases"] = ["liver_5"]
        self.inputs["contract"]["held_out_cases"] = ["liver_5"]
        with self.assertRaises(ValueError):
            self.verify()

    def test_nonfinite_ranking_metrics_or_scores_rejected(self):
        changes = (
            (["branches", "original_v1", "final_train", "MRR"], float("nan")),
            (["branches", "original_v1", "initial_held_out", "scores", 0, 0], float("inf")),
            (["branches", "original_v1", "final_held_out", "positive_minus_best_other", 0], float("nan")),
        )
        for field, value in changes:
            with self.subTest(field=field):
                self.reject("report", field, value)

    def test_invalid_neural_identity_hash_rejected(self):
        for value in ("", "x" * 64):
            with self.subTest(value=value):
                self.reject("report", ["shared_initial_state_sha256"], value)


if __name__ == "__main__":
    unittest.main()
