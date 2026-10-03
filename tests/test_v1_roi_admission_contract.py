"""CPU metadata UNIT checks only; no CT preparation, GPU run or quality claim."""
import copy
import unittest

from hiercp_v1x.contracts import (
    ContractError, STAGES, canonical_hash, config_diff,
    make_preparation_admission, make_run_contract, make_stage_config,
    resolve_execution_config, validate_preparation_admission, validate_resume,
    validate_stage_config, validate_transition, verify_archive,
)


class V1RoiAdmissionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proof = verify_archive()
        cls.base = cls.proof["base_config"]
        # Explicit UNIT-fixture budget and artifact identity, not measured resources.
        cls.admission = make_preparation_admission(12_000_000, "a" * 64)

    def config(self, stage="v1.0", admission=None):
        return make_stage_config(self.base, stage,
                                 preparation_admission=self.admission if admission is None else admission)

    def binding(self, config=None, admission=None):
        selected = self.admission if admission is None else admission
        return make_run_contract("v1.0", self.config() if config is None else config,
                                 preparation_admission=selected,
                                 cache_identity={"sha256": "b" * 64},
                                 source_identity={"sha256": "c" * 64},
                                 evaluation_identity={"split_sha256": "d" * 64},
                                 physical_batch_size=32, debug=True)

    def test_omitted_admission_preserves_exact_default_and_run_schema(self):
        self.assertEqual(self.base["graph"]["adaptive_roi_max_voxels"], 8_000_000)
        original = make_stage_config(self.base, "v1.0")
        self.assertEqual(original, self.base)
        self.assertEqual(make_stage_config(self.base, "v1.0", preparation_admission=None), original)
        run = make_run_contract("v1.0", original,
                                cache_identity={"sha": "b"}, source_identity={"sha": "c"},
                                evaluation_identity={"sha": "d"}, physical_batch_size=32, debug=True)
        self.assertNotIn("preparation_admission", run)

    def test_only_declared_guard_changes_in_every_stage(self):
        for stage in STAGES:
            original = make_stage_config(self.base, stage)
            admitted = self.config(stage)
            self.assertEqual(set(config_diff(original, admitted)), {"graph.adaptive_roi_max_voxels"})
            self.assertEqual(admitted["graph"]["adaptive_roi_max_voxels"], 12_000_000)
            validate_stage_config(admitted, stage, preparation_admission=self.admission)
        self.assertEqual(self.base["graph"]["adaptive_roi_max_voxels"], 8_000_000)

    def test_budget_must_be_explicit_integer_increase(self):
        for invalid in (None, True, False, 0, -1, 8_000_000, 7_999_999,
                        12_000_000.0, "12000000", float("inf"), float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ContractError):
                make_preparation_admission(invalid, "a" * 64)

    def test_probe_identity_must_be_complete_lowercase_sha256(self):
        for invalid in (None, "", "a" * 63, "a" * 65, "A" * 64,
                        "z" * 64, b"a" * 64, 123):
            with self.subTest(invalid=invalid), self.assertRaises(ContractError):
                make_preparation_admission(12_000_000, invalid)

    def test_profile_is_self_hashed_and_does_not_claim_quality(self):
        profile = validate_preparation_admission(self.admission)
        raw = {key: value for key, value in profile.items() if key != "contract_sha256"}
        self.assertEqual(profile["contract_sha256"], canonical_hash(raw))
        self.assertEqual(profile["config_path"], "graph.adaptive_roi_max_voxels")
        self.assertEqual(profile["original_max_voxels"], 8_000_000)
        self.assertIs(profile["quality_verified"], False)
        self.assertIsNot(profile, self.admission)

    def test_profile_tampering_is_rejected_even_with_recomputed_hash(self):
        edits = (("format", "arbitrary"), ("config_path", "graph.context_radius_mm"),
                 ("original_max_voxels", 7_000_000), ("scope", "training approved"),
                 ("quality_verified", True), ("unexpected_fallback", "skip"))
        for field, value in edits:
            for repair_hash in (False, True):
                altered = copy.deepcopy(self.admission)
                altered[field] = value
                if repair_hash:
                    altered["contract_sha256"] = canonical_hash(
                        {key: item for key, item in altered.items() if key != "contract_sha256"})
                with self.subTest(field=field, repair_hash=repair_hash), self.assertRaises(ContractError):
                    validate_preparation_admission(altered)
        for key in self.admission:
            altered = copy.deepcopy(self.admission)
            del altered[key]
            with self.subTest(missing=key), self.assertRaises(ContractError):
                validate_preparation_admission(altered)

    def test_non_mapping_admission_is_not_a_default(self):
        for invalid in (None, [], "research", 12_000_000):
            with self.subTest(invalid=invalid), self.assertRaises(ContractError):
                validate_preparation_admission(invalid)

    def test_changed_guard_without_bound_profile_is_rejected(self):
        config = self.config()
        with self.assertRaisesRegex(ContractError, "undeclared"):
            validate_stage_config(config, "v1.0")
        with self.assertRaisesRegex(ContractError, "undeclared"):
            make_run_contract("v1.0", config, cache_identity={"sha": "b"},
                              source_identity={"sha": "c"}, evaluation_identity={"sha": "d"},
                              physical_batch_size=32, debug=True)

    def test_altered_archive_base_cannot_be_adopted(self):
        for field in ("adaptive_roi_max_voxels", "adaptive_roi_margin_mm", "sample_context_nodes"):
            altered = copy.deepcopy(self.base)
            altered["graph"][field] += 1
            with self.subTest(field=field), self.assertRaisesRegex(ContractError, "exact archived"):
                make_stage_config(altered, "v1.0", preparation_admission=self.admission)

    def test_resource_profile_cannot_authorize_other_model_or_data_changes(self):
        for group, key in (("graph", "adaptive_roi_margin_mm"), ("graph", "context_radius_mm"),
                           ("graph", "sample_context_nodes"), ("graph", "patch_size"),
                           ("model", "hidden_dim"), ("training", "epochs"),
                           ("training", "pairwise_weight"), ("cache", "candidate_pool_size")):
            config = self.config()
            config[group][key] += 1
            with self.subTest(group=group, key=key), self.assertRaisesRegex(ContractError, "undeclared"):
                validate_stage_config(config, "v1.0", preparation_admission=self.admission)

    def test_execution_lock_and_stage_transitions_share_one_admission(self):
        lock = {"selected_batch_size": 32, "selected_num_workers": 8}
        previous = None
        for stage in STAGES:
            config = resolve_execution_config(self.config(stage), lock)
            validate_stage_config(config, stage, execution_lock=lock,
                                  preparation_admission=self.admission)
            if previous:
                result = validate_transition(previous[0], previous[1], stage, config,
                                             execution_lock=lock, preparation_admission=self.admission)
                self.assertNotIn("graph.adaptive_roi_max_voxels", result["config_diff"])
            previous = stage, config

    def test_different_guard_cannot_enter_same_transition(self):
        different = make_preparation_admission(13_000_000, "a" * 64)
        with self.assertRaisesRegex(ContractError, "undeclared"):
            validate_transition("v1.0", self.config(), "v1.1", self.config("v1.1", different),
                                preparation_admission=self.admission)

    def test_run_and_resume_bind_exact_admission_and_probe_identity(self):
        binding = self.binding()
        self.assertEqual(binding["preparation_admission"], self.admission)
        self.assertIsNot(binding["preparation_admission"], self.admission)
        validate_resume(binding, copy.deepcopy(binding))
        other_probe = make_preparation_admission(12_000_000, "e" * 64)
        changed = self.binding(self.config(admission=other_probe), other_probe)
        self.assertEqual(changed["config"], binding["config"])
        with self.assertRaisesRegex(ContractError, "resume contract mismatch"):
            validate_resume(binding, changed)
        other_budget = make_preparation_admission(13_000_000, "a" * 64)
        with self.assertRaisesRegex(ContractError, "resume contract mismatch"):
            validate_resume(binding, self.binding(self.config(admission=other_budget), other_budget))

    def test_original_source_archive_stays_byte_exact(self):
        after = verify_archive()
        self.assertEqual(after["archive_sha256"], self.proof["archive_sha256"])
        self.assertEqual(after["manifest_sha256"], self.proof["manifest_sha256"])
        self.assertEqual(after["file_hashes"], self.proof["file_hashes"])
        self.assertEqual(after["verified_files"], 202)


if __name__ == "__main__":
    unittest.main()
