"""DEBUG metadata fixtures only. No real checkpoint is loaded or created."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp.contracts import ARCHITECTURE_VERSION, GEOMETRY_CONTRACT
from hiercp.schema import UPPER_FEATURE_POLICY, graph_config_from_dict
from tools import ablation


def debug_payload(mode="full"):
    config = json.loads((Path(__file__).resolve().parents[1] / "config" / "train.json").read_text(encoding="utf-8"))
    training = config["training"]
    curriculum_keys = ("easy_epochs", "inter_epochs", "intra_epochs", "model_mine_start_epoch",
                       "semi_hard_low_percentile", "semi_hard_high_percentile",
                       "cross_entropy_weight", "pairwise_weight", "ordinal_weight", "mined_weight")
    signature = {
        "format": "hiercp_training_signature_v1", "run_mode": "production" if mode == "full" else "ablation",
        "target_epochs": training["epochs"], "seed": config["seed"],
        "batch_setting": training["batch_size"], "batch_size": 4,
        "worker_setting": training["num_workers"], "num_workers": 4,
        "gradient_accumulation_setting": training["gradient_accumulation_steps"],
        "gradient_accumulation_steps": 1, "target_effective_batch_size": training["target_effective_batch_size"],
        "resolved_effective_batch_size": 4, "calibration_resource_fingerprint": "DEBUG-resource",
        "consistency_weight": training["consistency_weight"],
        "optimizer": {"name": "AdamW", "lr": training["lr"], "weight_decay": training["weight_decay"], "fused": True},
        "scheduler": {"name": "CosineAnnealingLR", "t_max": training["epochs"]},
        "amp": True, "grad_clip": training["grad_clip"], "deterministic": True, "allow_tf32": False,
        "curriculum": {key: training[key] for key in curriculum_keys},
        "train_cache_files": ["DEBUG_train.pt"], "val_cache_files": ["DEBUG_val.pt"],
    }
    kwargs = dict(config["model"])
    if mode != "full":
        signature["ablation_mode"] = mode
        kwargs["ablation_mode"] = mode
    return {
        "method": "DEBUG-metadata-only", "framework": "torch_geometric",
        "architecture_version": ARCHITECTURE_VERSION, "geometry_contract": GEOMETRY_CONTRACT,
        "upper_feature_policy": UPPER_FEATURE_POLICY, "model_kwargs": kwargs,
        "graph_config": graph_config_from_dict(config["graph"]).to_dict(), "ct_clip": tuple(config["ct_clip"]),
        "prototype_training_cases": ["DEBUG_train"], "prototype_fingerprint": "a" * 64,
        "validation_policy": {"format": "hiercp_fixed_validation_v1", "epoch": training["fixed_validation_epoch"]},
        "training_signature": signature, "training_complete": True,
        "target_epochs": training["epochs"], "completed_epoch": training["epochs"],
        "epoch": 12, "best_epoch": 12,
        "cache_publication": {f"{name}_sha256": "b" * 64 for name in ("config", "index", "complete")},
        "runtime": {"amp": True, "workers": 4, "fused_adamw": True},
        "gradient_connectivity": {"verified": True, "missing_parameters": [],
                                  "connected_parameters": ["DEBUG-weight"], "connected_parameter_count": 1,
                                  "expected_parameter_count": 1},
        "selection": {"mrr": .7, "acc": .6, "margin": .2, "ranking": .8, "consistency": .1},
    }


def record(payload, mode):
    with patch.object(ablation, "_load_checkpoint", return_value=payload):
        return ablation._checkpoint_record(Path("DEBUG-NO-FILESYSTEM-ACCESS"), mode)


class AblationComparabilityDebugTests(unittest.TestCase):
    def pair(self, changed=None):
        full = debug_payload()
        variant = debug_payload("no_population") if changed is None else changed
        return {"full": record(full, "full"), "no_population": record(variant, "no_population")}

    def test_debug_complete_matched_roles_allow_real_difference(self):
        records = self.pair()
        records["no_population"]["mrr"] = .65
        self.assertEqual(ablation._pairwise_comparability(records, "no_population")["status"], "comparable")
        self.assertAlmostEqual(ablation._difference(records, "mrr", "no_population"), .05)
        self.assertEqual(records["full"]["epoch"], debug_payload()["target_epochs"])

    def test_debug_partial_checkpoint_never_produces_contribution(self):
        payload = debug_payload("no_population")
        payload["training_complete"] = False
        payload["completed_epoch"] -= 1
        records = self.pair(payload)
        self.assertIsNone(ablation._difference(records, "mrr", "no_population"))
        self.assertIn("training_not_complete", records["no_population"]["comparison_issues"])

    def test_debug_changed_seed_epochs_batch_cache_graph_model_are_incomparable(self):
        changes = (
            ("training_signature", "seed", 100),
            ("training_signature", "target_epochs", 41),
            ("training_signature", "batch_size", 8),
            ("training_signature", "resolved_effective_batch_size", 8),
            ("training_signature", "val_cache_files", ["OTHER_debug_val.pt"]),
            ("cache_publication", "index_sha256", "c" * 64),
            ("graph_config", "sample_hops", 3),
            ("model_kwargs", "hidden_dim", 256),
            ("validation_policy", "epoch", 30),
        )
        for group, key, value in changes:
            with self.subTest(group=group, key=key):
                payload = debug_payload("no_population")
                payload[group][key] = value
                records = self.pair(payload)
                comparison = ablation._pairwise_comparability(records, "no_population")
                self.assertEqual(comparison["status"], "incomparable")
                self.assertTrue(comparison["reasons"])
                self.assertIsNone(ablation._difference(records, "mrr", "no_population"))

    def test_debug_legacy_or_fingerprint_change_is_not_silently_accepted(self):
        for key in ("architecture_version", "geometry_contract", "cache_publication",
                    "prototype_fingerprint", "training_signature", "gradient_connectivity"):
            with self.subTest(key=key):
                payload = debug_payload("no_population")
                payload.pop(key)
                records = self.pair(payload)
                self.assertIsNone(ablation._difference(records, "mrr", "no_population"))
                self.assertTrue(records["no_population"]["comparison_issues"])
        payload = debug_payload("no_population")
        payload["prototype_fingerprint"] = "d" * 64
        self.assertIsNone(ablation._difference(self.pair(payload), "mrr", "no_population"))

    def test_debug_training_complete_truthy_string_is_not_completion(self):
        payload = debug_payload("no_population")
        payload["training_complete"] = "false"
        self.assertIsNone(ablation._difference(self.pair(payload), "mrr", "no_population"))

    def test_debug_request_guard_detects_changed_config_without_any_training(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "config" / "train.json").read_text(encoding="utf-8"))
        reference = record(debug_payload(), "full")
        args = SimpleNamespace(seed=None, epochs=None, batch_size=None, num_workers=None)
        self.assertEqual(ablation._reference_request_issues(reference, config, args), [])
        changed = copy.deepcopy(config)
        changed["training"]["epochs"] += 1
        self.assertIn("training_signature.target_epochs", ablation._reference_request_issues(reference, changed, args))
        args.seed = 123
        self.assertIn("training_signature.seed", ablation._reference_request_issues(reference, config, args))

    def test_debug_full_reuse_requires_current_cache_and_prototype_hashes(self):
        reference = record(debug_payload(), "full")
        cache = {
            "prototype_artifact_sha256": "e" * 64,
            "prototype_fingerprint": reference["comparison_contract"]["prototype_fingerprint"],
            "graph_config": reference["comparison_contract"]["graph_config"],
            "train_case_ids": ["DEBUG_train"], "val_case_ids": ["DEBUG_val"],
        }
        split = {"train": ["DEBUG_train"], "val": ["DEBUG_val"]}

        def read_debug(path, **_kwargs):
            return json.dumps(split if path.name == "split.json" else cache)

        def matching_hash(path):
            return "e" * 64 if path.name == "prototype.pt" else "b" * 64

        with patch.object(ablation, "_checkpoint_record", return_value=reference), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", autospec=True, side_effect=read_debug), \
             patch.object(ablation, "_sha256_file", side_effect=matching_hash):
            self.assertEqual(ablation._require_shared_assets(Path("DEBUG-no-real-assets")), reference)
            with patch.object(ablation, "_sha256_file", return_value="f" * 64):
                with self.assertRaisesRegex(ValueError, "cache publication"):
                    ablation._require_shared_assets(Path("DEBUG-no-real-assets"))
            cache["prototype_artifact_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "prototype artifact"):
                ablation._require_shared_assets(Path("DEBUG-no-real-assets"))

    def test_debug_summary_will_not_overwrite_existing_reports_by_default(self):
        records = [record(debug_payload(mode), mode) for mode in ablation.ABLATION_ORDER]
        args = SimpleNamespace(project=str(ablation.PROJECT_ROOT), work=None,
                               output="DEBUG-report-no-write", overwrite=False)
        with patch.object(ablation, "_complete_records", return_value=records), \
             patch.object(Path, "mkdir"), patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "open", side_effect=AssertionError("No report should be opened")):
            with self.assertRaises(FileExistsError):
                ablation.command_summarize(args)


if __name__ == "__main__":
    unittest.main()
