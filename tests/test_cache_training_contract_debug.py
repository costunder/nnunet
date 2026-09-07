"""DEBUG metadata fixtures only: no medical data, training or final metrics."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hiercp.cache import CACHE_FORMAT
from hiercp import pipeline
from hiercp.schema import graph_config_from_dict


class CacheTrainingContractDebugTests(unittest.TestCase):
    def setUp(self):
        # Checked-in final configuration is read only; all mutations are isolated
        # in-memory DEBUG fixtures and never change production settings.
        config_path = Path(__file__).resolve().parents[1] / "config" / "train.json"
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.graph = graph_config_from_dict(self.config["graph"]).to_dict()
        cache = self.config["cache"]
        self.metadata = {
            "format": CACHE_FORMAT,
            "prototype_fingerprint": "DEBUG_PROTOTYPE",
            "graph_config": self.graph,
            "ct_clip": self.config["ct_clip"],
            "labels": self.config["labels"],
            **{key: cache[key] for key in (
                "source_selection", "source_pad", "samples_per_case",
                "total_candidates", "candidate_pool_size", "max_draws",
                "occupied_clearance_vox", "min_liver_coverage",
                "min_center_separation_mm",
                "min_center_separation_vox", "no_placement_policy",
            )},
            "difficulty_fractions": {
                "easy": cache["easy_fraction"],
                "inter": cache["inter_fraction"],
                "intra_corrupted": cache["intra_fraction"],
            },
            "seed": 1042,
        }

    def validate(self, *, config=None, metadata=None, index_fingerprint="DEBUG_PROTOTYPE"):
        config = self.config if config is None else config
        metadata = self.metadata if metadata is None else metadata
        with mock.patch("hiercp.data.load_cache_config", return_value=metadata), \
             mock.patch("hiercp.data.load_cache_index", return_value={
                 "prototype_fingerprint": index_fingerprint,
             }):
            pipeline._validate_cache_metadata(
                "DEBUG_NOT_READ", prototype_fingerprint="DEBUG_PROTOTYPE",
                graph_config=graph_config_from_dict(config["graph"]).to_dict(),
                ct_clip=tuple(config["ct_clip"]),
                cache_config=config["cache"], labels=config["labels"],
            )

    def test_same_flat_contract_is_accepted(self):
        self.validate()

    def test_legacy_missing_optional_fields_only_match_legacy_policy(self):
        config, metadata = deepcopy(self.config), deepcopy(self.metadata)
        for key in ("min_center_separation_vox", "no_placement_policy"):
            del config["cache"][key]
            del metadata[key]
        self.validate(config=config, metadata=metadata)
        with self.assertRaisesRegex(ValueError, "no_placement_policy"):
            self.validate(metadata=metadata)

    def test_every_changed_cp_generation_field_is_rejected(self):
        for key in self.config["cache"]:
            with self.subTest(key=key):
                changed = deepcopy(self.config)
                value = changed["cache"][key]
                changed["cache"][key] = "largest" if isinstance(value, str) else value + 1
                metadata_key = "difficulty_fractions" if key.endswith("_fraction") else key
                with self.assertRaisesRegex(ValueError, metadata_key):
                    self.validate(config=changed)

    def test_missing_cp_contract_fields_are_not_assumed_to_match(self):
        for key in ("source_selection", "source_pad", "samples_per_case",
                    "total_candidates", "candidate_pool_size", "max_draws",
                    "occupied_clearance_vox", "min_liver_coverage",
                    "min_center_separation_mm", "difficulty_fractions", "labels"):
            with self.subTest(key=key):
                metadata = deepcopy(self.metadata)
                del metadata[key]
                with self.assertRaisesRegex(ValueError, key):
                    self.validate(metadata=metadata)

    def test_changed_liver_or_tumor_label_is_rejected(self):
        for key in ("liver", "tumor"):
            with self.subTest(key=key):
                config = deepcopy(self.config)
                config["labels"][key] += 1
                with self.assertRaisesRegex(ValueError, "labels"):
                    self.validate(config=config)

    def test_prepare_numeric_normalization_is_preserved(self):
        config = deepcopy(self.config)
        for key, value in config["cache"].items():
            if not isinstance(value, str):
                config["cache"][key] = str(value)
        config["labels"] = {key: str(value) for key, value in config["labels"].items()}
        self.validate(config=config)

    def test_unrelated_training_runtime_generation_and_model_seed_are_not_hashed(self):
        config = deepcopy(self.config)
        config["seed"] = 17
        config["training"]["lr"] *= 2
        config["training"]["epochs"] += 1
        config["runtime"]["prepare_workers"] = 4
        config["model"]["dropout"] = 0.2
        config["generation"]["source_pad"] += 1
        self.validate(config=config)

    def test_existing_graph_prototype_ct_clip_and_index_guards_remain(self):
        for key in ("format", "prototype_fingerprint", "graph_config", "ct_clip"):
            with self.subTest(key=key):
                metadata = deepcopy(self.metadata)
                metadata[key] = "DEBUG_WRONG"
                with self.assertRaisesRegex(ValueError, key):
                    self.validate(metadata=metadata)
        with self.assertRaisesRegex(ValueError, "Cache index prototype fingerprint"):
            self.validate(index_fingerprint="DEBUG_WRONG")

    def test_error_identifies_cached_and_requested_condition(self):
        config = deepcopy(self.config)
        config["cache"]["source_pad"] = 4
        with self.assertRaisesRegex(ValueError, r"source_pad \(cached=2, requested=4\)"):
            self.validate(config=config)

    def test_production_train_checks_actual_request_before_overwrite_or_training(self):
        config = deepcopy(self.config)
        config["cache"]["source_pad"] += 1
        args = argparse.Namespace(
            run_mode="production", config="DEBUG_NOT_READ", seed=None,
            epochs=None, batch_size=None, num_workers=None,
            checkpoint="DEBUG_NOT_WRITTEN.pt", overwrite=True,
            cache_dir="DEBUG_NOT_READ", prototype_bank="DEBUG_NOT_READ",
        )
        bank = mock.Mock()
        bank.fingerprint.return_value = "DEBUG_PROTOTYPE"
        with mock.patch.object(pipeline, "_load_json", return_value=config), \
             mock.patch.object(pipeline.PrototypeBank, "load", return_value=bank), \
             mock.patch("hiercp.data.list_cache_files", return_value=[Path("DEBUG.pt")]), \
             mock.patch("hiercp.data.load_cache_config", return_value=self.metadata), \
             mock.patch("hiercp.data.load_cache_index") as index, \
             mock.patch("hiercp.tensor.set_seed") as training_start, \
             mock.patch.object(Path, "unlink") as delete:
            with self.assertRaisesRegex(ValueError, "source_pad"):
                pipeline.run_train(args)
            index.assert_not_called()
            training_start.assert_not_called()
            delete.assert_not_called()

    def test_rejected_production_cohort_preserves_existing_checkpoint_bytes(self):
        cases = (
            ({"subset_active": True, "materialized_sample_ratio": 1.0, "resolved_sample_ratio": 1.0},
             [Path("DEBUG_VAL.pt")], "complete configured cache cohort"),
            ({"subset_active": False, "materialized_sample_ratio": 0.5, "resolved_sample_ratio": 0.5},
             [Path("DEBUG_VAL.pt")], "complete configured cache cohort"),
            ({"subset_active": False, "materialized_sample_ratio": 1.0, "resolved_sample_ratio": 1.0},
             [], "non-empty fixed validation split"),
        )
        for usage, validation, message in cases:
            with self.subTest(usage=usage, validation=bool(validation)), \
                 tempfile.TemporaryDirectory(prefix="hiercp_contract_debug_") as folder:
                checkpoint = Path(folder) / "DEBUG.pt"
                resume = Path(folder) / "DEBUG.last.pt"
                checkpoint.write_bytes(b"DEBUG existing checkpoint: preserve exactly")
                resume.write_bytes(b"DEBUG existing training state: preserve exactly")
                originals = {path: path.read_bytes() for path in (checkpoint, resume)}
                args = argparse.Namespace(
                    run_mode="production", config="DEBUG_NOT_READ", seed=None,
                    epochs=None, batch_size=None, num_workers=None,
                    checkpoint=str(checkpoint), overwrite=True,
                    cache_dir="DEBUG_NOT_READ", prototype_bank="DEBUG_NOT_READ",
                )
                bank = mock.Mock()
                bank.fingerprint.return_value = "DEBUG_PROTOTYPE"
                with mock.patch.object(pipeline, "_load_json", return_value=self.config), \
                     mock.patch.object(pipeline.PrototypeBank, "load", return_value=bank), \
                     mock.patch("hiercp.data.list_cache_files", return_value=[Path("DEBUG.pt")]), \
                     mock.patch("hiercp.data.load_cache_config", return_value=self.metadata), \
                     mock.patch("hiercp.data.load_cache_index", return_value={
                         "prototype_fingerprint": "DEBUG_PROTOTYPE",
                     }), \
                     mock.patch("hiercp.data.split_files_from_cache", return_value=(
                         [Path("DEBUG.pt")], validation,
                     )), \
                     mock.patch("hiercp.data.summarize_cache_usage", return_value=usage), \
                     mock.patch("hiercp.tensor.set_seed") as training_start:
                    with self.assertRaisesRegex(RuntimeError, message):
                        pipeline.run_train(args)
                    training_start.assert_not_called()
                for path, expected in originals.items():
                    self.assertEqual(path.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
