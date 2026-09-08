"""DEBUG metadata-contract tests, not medical-data training or GPU validation.

Use real checkpoint/JSON serialization and immutable artifact hashes. Only the
prototype deserializer is replaced; its file existence and SHA checks are real.
The tiny Linear supplies real parameter names, not a claimed trained GNN.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from hiercp.tensor import torch_load_compat
from tools import causality


class CausalityCheckpointContractDebugTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="DEBUG_causality_contract_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.serial = 0

    def fixture(self, mutate=None):
        self.serial += 1
        root = self.root / str(self.serial)
        root.mkdir()
        cache_dir = root / "graphs"
        cache_dir.mkdir()
        prototype_path = root / "prototype.pt"
        prototype_path.write_bytes(b"DEBUG prototype deserializer stand-in; no trained data")
        prototype_sha = hashlib.sha256(prototype_path.read_bytes()).hexdigest()
        model = torch.nn.Linear(2, 2)
        parameter_names = sorted(name for name, _ in model.named_parameters())
        train_ids = ["DEBUG_train_patient"]
        fingerprint = "DEBUG_prototype_fingerprint"
        graph_config = {"debug_metadata_fixture": True, "layers": [3, 2, 2]}
        checkpoint = {
            "debug_metadata_fixture": True,
            "method": "hiercp-full",
            "framework": "torch_geometric",
            "training_complete": True,
            "target_epochs": 40,
            "completed_epoch": 40,
            "training_signature": {
                "format": "hiercp_training_signature_v1",
                "run_mode": "benchmark",
                "ablation_mode": "full",
                "target_epochs": 40,
                "train_cache_files": ["DEBUG_train.pt"],
                "val_cache_files": ["DEBUG_val.pt"],
            },
            "graph_config": graph_config.copy(),
            "ct_clip": (-200.0, 250.0),
            "prototype_fingerprint": fingerprint,
            "prototype_training_cases": train_ids.copy(),
            "gradient_connectivity": {
                "format": "hiercp_gradient_connectivity_v1",
                "verified": True,
                "expected_parameter_count": len(parameter_names),
                "connected_parameter_count": len(parameter_names),
                "connected_parameters": parameter_names,
                "missing_parameters": [],
            },
        }
        config = {
            "debug_metadata_fixture": True,
            "run_mode": "benchmark",
            "graph_config": graph_config.copy(),
            "ct_clip": [-200.0, 250.0],
            "prototype_bank": str(prototype_path),
            "prototype_artifact_sha256": prototype_sha,
            "prototype_fingerprint": fingerprint,
            "train_case_ids": train_ids.copy(),
        }
        index = {"entries": [
            {"path": "DEBUG_val.pt", "split": "val"},
            {"path": "DEBUG_train.pt", "split": "train"},
        ]}
        if mutate is not None:
            mutate(checkpoint, config, index)
        checkpoint_path = root / "DEBUG_metadata_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        for name, payload in (("config.json", config), ("index.json", index)):
            (cache_dir / name).write_text(json.dumps(payload), encoding="utf-8")
        arguments = {
            "checkpoint": torch_load_compat(checkpoint_path, map_location="cpu"),
            "checkpoint_path": checkpoint_path,
            "prototype_path": prototype_path,
            "cache_dir": cache_dir,
            "cache_config": json.loads((cache_dir / "config.json").read_text(encoding="utf-8")),
            "cache_index": json.loads((cache_dir / "index.json").read_text(encoding="utf-8")),
            "model": model,
            "run_mode": "benchmark",
        }
        bank = SimpleNamespace(training_case_ids=train_ids, fingerprint=lambda: fingerprint)
        files = [checkpoint_path, prototype_path, cache_dir / "config.json", cache_dir / "index.json"]
        return arguments, bank, files

    def validate(self, fixture):
        arguments, bank, files = fixture
        metadata = [arguments[key] for key in ("checkpoint", "cache_config", "cache_index")]
        original_metadata = pickle.dumps(metadata)
        original_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        try:
            with mock.patch.object(causality.PrototypeBank, "load", return_value=bank):
                return causality._validate_checkpoint_contract(**arguments)
        finally:
            self.assertEqual(pickle.dumps(metadata), original_metadata, "Validator mutated input metadata")
            self.assertEqual(
                {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
                original_hashes,
                "Validator rewrote an input artifact",
            )

    def test_real_torch_tuple_and_json_list_roundtrip_passes_without_rewriting(self):
        fixture = self.fixture()
        arguments = fixture[0]
        self.assertIsInstance(arguments["checkpoint"]["ct_clip"], tuple)
        self.assertIsInstance(arguments["cache_config"]["ct_clip"], list)
        self.assertIs(self.validate(fixture), arguments["checkpoint"]["training_signature"])

    def test_list_tuple_and_numeric_representations_are_equivalent(self):
        for checkpoint_clip in ((-200.0, 250.0), [-200, 250], (-200, 250.0)):
            for cache_clip in ([-200.0, 250.0], (-200, 250), [-200, 250.0]):
                with self.subTest(checkpoint=checkpoint_clip, cache=cache_clip):
                    fixture = self.fixture(lambda checkpoint, config, index: (
                        checkpoint.update(ct_clip=checkpoint_clip), config.update(ct_clip=cache_clip)
                    ))
                    # JSON has no tuple type; also exercise the inverse in-memory
                    # container direction explicitly after its real disk roundtrip.
                    if isinstance(cache_clip, tuple):
                        fixture[0]["cache_config"]["ct_clip"] = tuple(fixture[0]["cache_config"]["ct_clip"])
                    self.validate(fixture)

    def test_each_bound_change_including_one_float_step_is_rejected(self):
        for artifact in ("checkpoint", "cache_config"):
            for bound, value in ((0, -199.0), (1, 251.0),
                                 (0, math.nextafter(-200.0, math.inf)),
                                 (1, math.nextafter(250.0, math.inf))):
                with self.subTest(artifact=artifact, bound=bound, value=value):
                    fixture = self.fixture()
                    self.validate(fixture)
                    changed = list(fixture[0][artifact]["ct_clip"])
                    changed[bound] = value
                    fixture[0][artifact]["ct_clip"] = changed
                    with self.assertRaisesRegex(ValueError, "ct_clip"):
                        self.validate(fixture)

    def test_invalid_bounds_fail_on_either_or_both_artifacts(self):
        missing = object()
        malformed = [missing, None, True, 12, "[-200, 250]", [], [-200],
                     [-200, 0, 250], {"low": -200, "high": 250},
                     [False, 250], [-200, True], ["-200", 250], [-200, "250"],
                     [float("nan"), 250], [-200, float("nan")],
                     [float("-inf"), 250], [-200, float("inf")],
                     [250, -200], [250, 250], [[-200], [250]]]
        for value in malformed:
            for target in ("checkpoint", "cache", "both"):
                with self.subTest(value=repr(value), target=target):
                    def mutate(checkpoint, config, index):
                        selected = [checkpoint] if target == "checkpoint" else [config]
                        if target == "both":
                            selected = [checkpoint, config]
                        for artifact in selected:
                            if value is missing:
                                artifact.pop("ct_clip")
                            else:
                                artifact["ct_clip"] = value
                    with self.assertRaisesRegex(ValueError, "ct_clip"):
                        self.validate(self.fixture(mutate))

    def test_training_completion_and_signature_guards_remain_strict(self):
        fixture = self.fixture()
        self.validate(fixture)
        changes = [
            ("method", "other", "hiercp-full"),
            ("framework", "other", "torch_geometric"),
            ("training_complete", False, "completed full training"),
            ("completed_epoch", 39, "completed_epoch"),
            ("target_epochs", 0, "target_epochs"),
            ("training_signature", None, "training signature"),
        ]
        for key, value, error in changes:
            with self.subTest(field=key):
                current = self.fixture(lambda checkpoint, config, index: checkpoint.update({key: value}))
                with self.assertRaisesRegex(ValueError, error):
                    self.validate(current)
        for field, value in (("run_mode", "production"), ("ablation_mode", "no_patient"),
                             ("target_epochs", 39), ("format", "invalid")):
            with self.subTest(signature_field=field):
                current = self.fixture(lambda checkpoint, config, index:
                                       checkpoint["training_signature"].update({field: value}))
                with self.assertRaisesRegex(ValueError, "training signature"):
                    self.validate(current)

    def test_cache_cohort_and_graph_guards_remain_strict(self):
        self.validate(self.fixture())
        for cohort, error in (("train_cache_files", "training cache cohort"),
                              ("val_cache_files", "validation cache cohort")):
            with self.subTest(cohort=cohort):
                current = self.fixture(lambda checkpoint, config, index:
                                       checkpoint["training_signature"].update({cohort: []}))
                with self.assertRaisesRegex(ValueError, error):
                    self.validate(current)
        for field, value, error in (("graph_config", {"changed": True}, "graph_config"),
                                    ("run_mode", "production", "run_mode")):
            with self.subTest(cache_field=field):
                current = self.fixture(lambda checkpoint, config, index: config.update({field: value}))
                with self.assertRaisesRegex(ValueError, error):
                    self.validate(current)

    def test_prototype_path_sha_and_fingerprint_guards_remain_strict(self):
        self.validate(self.fixture())
        mutations = [
            (lambda checkpoint, config, index: config.update(prototype_bank=str(self.root / "other.pt")),
             "Requested prototype path"),
            (lambda checkpoint, config, index: config.update(prototype_artifact_sha256="0" * 64),
             "SHA-256"),
            (lambda checkpoint, config, index: config.update(prototype_fingerprint="changed"),
             "Prototype-bank fingerprint"),
            (lambda checkpoint, config, index: checkpoint.update(prototype_fingerprint="changed"),
             "Checkpoint prototype fingerprint"),
        ]
        for mutate, error in mutations:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    self.validate(self.fixture(mutate))

    def test_prototype_training_cohort_guards_remain_strict(self):
        self.validate(self.fixture())
        for field, value, error in (("train_case_ids", [], "exact training case cohort"),
                                    ("train_case_ids", ["DEBUG_different"], "Prototype-bank training cohort")):
            with self.subTest(value=value):
                current = self.fixture(lambda checkpoint, config, index: config.update({field: value}))
                with self.assertRaisesRegex(ValueError, error):
                    self.validate(current)
        current = self.fixture(lambda checkpoint, config, index:
                               checkpoint.update(prototype_training_cases=["DEBUG_different"]))
        with self.assertRaisesRegex(ValueError, "Checkpoint prototype training cohort"):
            self.validate(current)

    def test_actual_named_parameter_gradient_guards_remain_strict(self):
        self.validate(self.fixture())
        for field, value in (("format", "invalid"), ("verified", False),
                             ("expected_parameter_count", 1), ("connected_parameter_count", 1),
                             ("connected_parameters", ["weight"]), ("missing_parameters", ["bias"])):
            with self.subTest(field=field):
                current = self.fixture(lambda checkpoint, config, index:
                                       checkpoint["gradient_connectivity"].update({field: value}))
                with self.assertRaisesRegex(ValueError, "gradient connectivity"):
                    self.validate(current)
        current = self.fixture()
        current[0]["model"].requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "no trainable parameters"):
            self.validate(current)


if __name__ == "__main__":
    unittest.main()
