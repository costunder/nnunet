"""DEBUG NumPy regressions, not medical/model training or performance evidence.

The reproduced terminal-membership case retains production K=16, the 30-update
budget, 16 descriptor dimensions, and all 24 regions of every synthetic case.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hiercp.prototype import PrototypeBank, audit_legacy_bank, audit_prototype_bank, build_prototype_bank
from hiercp.region import numpy_kmeans


def _config():
    payload = json.loads((Path(__file__).resolve().parents[1] / "config/train.json").read_text(encoding="utf-8"))
    config = SimpleNamespace(**payload["graph"])
    assert (config.num_prototypes, config.prototype_lloyd_iters, config.prototype_k) == (16, 30, 4)
    return config


def _reproduction():
    values = np.random.default_rng(20260912).uniform(0, 1, (64 * 24, 16)).astype(np.float32)
    values[:, :3] = values[:, :3] * 2 - 1
    values[:, 15] = 1
    groups = [(f"debug_case_{i:03d}", values[i * 24:(i + 1) * 24]) for i in range(64)]
    return values, build_prototype_bank(groups, config=_config(), rng=np.random.default_rng(1051))


class _UnsupportedAuditValue:
    """A deliberately non-tensor pickle global for the DEBUG safe-reader test."""


class PrototypeFitV2Debug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values, cls.bank = _reproduction()

    def test_actual_default30_terminal_assignment_and_statistics(self):
        bank, values = self.bank, self.values
        distances = bank.standardized_distances(values)
        standardized = (values - bank.descriptor_mean) / bank.descriptor_std
        labels = ((standardized[:, None] - bank.standardized_centers[None]) ** 2).sum(-1).argmin(axis=1)
        np.testing.assert_array_equal(bank.fit_labels, labels)
        np.testing.assert_array_equal(bank.fit_counts, np.bincount(labels, minlength=16))
        np.testing.assert_allclose(bank.features[:, 16], bank.fit_counts / len(values), atol=1e-7)
        for cluster in range(16):
            self.assertAlmostEqual(float(bank.cluster_mean_distance[cluster]),
                                   float(distances[labels == cluster, cluster].mean()), places=6)
        self.assertEqual(bank.fit_provenance["max_iterations"], 30)
        self.assertEqual(bank.fit_provenance["iterations_used"], 30)
        self.assertEqual(bank.fit_provenance["final_reassigned_count"], 1)
        self.assertFalse(bank.fit_provenance["membership_stable"])
        # A final E-step fixes membership without adding M-steps or claiming
        # that a budget-limited result is a stationary Lloyd solution.
        standardized = (values - bank.descriptor_mean) / bank.descriptor_std
        gap = max(np.linalg.norm(standardized[labels == i].mean(0) - bank.standardized_centers[i])
                  for i in range(16))
        self.assertGreater(float(gap), 0)
        bank.validate(verify_fit=True)
        self.assertEqual(bank.edge_index.shape, (2, 16 * 4))
        self.assertEqual(bank.features.shape, (16, 18))

    def test_bank_readers_never_fall_back_to_unsafe_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "debug_unsupported_global.pt"
            torch.save({"format": "hiercp_prototype_bank_v1", "extra": _UnsupportedAuditValue()}, path)
            original = path.read_bytes()
            for reader in (PrototypeBank.load, audit_prototype_bank, audit_legacy_bank):
                with self.subTest(reader=reader.__name__):
                    with self.assertRaises(pickle.UnpicklingError):
                        reader(path)
                    with patch("hiercp.prototype.torch.load", side_effect=TypeError("DEBUG old API")) as loader:
                        with self.assertRaisesRegex(RuntimeError, "unsafe pickle fallback is not permitted"):
                            reader(path)
                        loader.assert_called_once_with(path, map_location="cpu", weights_only=True)
            self.assertEqual(original, path.read_bytes())

    def test_finite_and_early_stopping_contract(self):
        values = np.asarray([[0., 0.], [0., 0.], [8., 8.], [8., 8.]], dtype=np.float32)
        centers, labels, diagnostics = numpy_kmeans(values, 2, rng=np.random.default_rng(3),
                                                    iterations=30, return_diagnostics=True)
        np.testing.assert_array_equal(labels, ((values[:, None] - centers[None]) ** 2).sum(-1).argmin(1))
        self.assertEqual(diagnostics["stopping_reason"], "center_allclose")
        values[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            numpy_kmeans(values, 2, rng=np.random.default_rng(3), iterations=30)

    def test_fit_evidence_tamper_is_not_auto_approved(self):
        for field in ("fit_labels", "fit_counts", "descriptor_std", "features"):
            with self.subTest(field=field):
                bank = copy.deepcopy(self.bank)
                if field == "fit_labels":
                    bank.fit_labels[0] = (bank.fit_labels[0] + 1) % 16
                elif field == "fit_counts":
                    bank.fit_counts[0] += 1
                elif field == "descriptor_std":
                    bank.descriptor_std[0] *= 2
                else:
                    bank.features[0, 17] += .1
                with self.assertRaises(ValueError):
                    bank.validate(verify_fit=True)

    def test_current_roundtrip_is_idempotent_but_different_bank_cannot_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "debug_prototypes_v2.pt"
            self.bank.save(path)
            original = path.read_bytes()
            loaded = PrototypeBank.load(path)
            self.assertEqual(self.bank.fingerprint(), loaded.fingerprint())
            audit = audit_prototype_bank(path)
            self.assertTrue(audit["membership_verified"])
            self.assertFalse(audit["can_use_for_current_training"])
            self.assertEqual(audit["fit_descriptor_count"], 64 * 24)
            self.assertEqual(audit["knn_degree"], 4)
            self.bank.save(path, overwrite=True)
            self.assertEqual(original, path.read_bytes())
            # An independently repeated fit has different Python objects and
            # would produce a new Torch archive, but exactly the same semantic
            # payload. A partial factory retry must retain the original bytes.
            _, refitted = _reproduction()
            self.assertIsNot(refitted, self.bank)
            self.assertEqual(refitted.fingerprint(), self.bank.fingerprint())
            refitted.save(path, overwrite=False)
            self.assertEqual(original, path.read_bytes())
            changed = copy.deepcopy(self.bank)
            changed.fit_provenance["numpy_version"] += "-different-provenance"
            with self.assertRaises((FileExistsError, ValueError)):
                changed.save(path, overwrite=True)
            self.assertEqual(original, path.read_bytes())

    def test_historical_v1_is_audit_only_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "debug_historical_v1.pt"
            fields = ("features", "standardized_centers", "descriptor_mean", "descriptor_std",
                      "edge_index", "training_case_ids")
            payload = {name: torch.from_numpy(getattr(self.bank, name)) for name in fields if name != "training_case_ids"}
            payload["training_case_ids"] = list(self.bank.training_case_ids)
            payload["format"] = "hiercp_prototype_bank_v1"
            torch.save(payload, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            report = audit_legacy_bank(path)
            self.assertFalse(report["can_use_for_current_training"])
            self.assertFalse(audit_prototype_bank(path)["membership_verified"])
            self.assertIn("final_membership_support_and_dispersion", report["not_verifiable_without_fit_descriptors"])
            with self.assertRaisesRegex(ValueError, "legacy|v1|format"):
                PrototypeBank.load(path)
            with self.assertRaises((FileExistsError, ValueError)):
                self.bank.save(path, overwrite=True)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_historical_knn_corruption_is_not_waived_by_legacy_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "debug_legacy_corrupt.pt"
            fields = ("features", "standardized_centers", "descriptor_mean", "descriptor_std", "edge_index")
            payload = {name: torch.from_numpy(getattr(self.bank, name).copy()) for name in fields}
            payload.update(format="hiercp_prototype_bank_v1", training_case_ids=list(self.bank.training_case_ids))
            payload["edge_index"][0, 0] = payload["edge_index"][1, 0]
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "KNN"):
                audit_prototype_bank(path)

    def test_relative_top2_weight_is_not_absolute_distance(self):
        patterns = np.zeros((16, 16), np.float32)
        patterns[:, 0] = np.linspace(-.9, .9, 16)
        patterns[:, 3:6] = .5
        patterns[:, 6] = np.linspace(.2, .8, 16)
        patterns[:, 7] = .05
        patterns[:, 8] = 1 / 24
        patterns[:, 9:15] = .25
        patterns[:, 15] = 1
        population = np.tile(patterns, (3, 1))
        bank = build_prototype_bank([("debug_a", population[:24]), ("debug_b", population[24:])],
                                    config=_config(), rng=np.random.default_rng(1051))
        near = (patterns[7] + patterns[8]) / 2
        far = near.copy()
        far[7] = .45
        queries = np.stack([near, far])
        ids, weights = bank.assign(queries, top_k=2, temperature=.5)
        np.testing.assert_array_equal(ids[0], ids[1])
        np.testing.assert_allclose(weights[0], weights[1], atol=1e-7)
        distances = bank.standardized_distances(queries)[np.arange(2)[:, None], ids]
        self.assertTrue(np.all(distances[1] > distances[0]))
        excess = distances - bank.cluster_mean_distance[ids]
        self.assertTrue(np.isfinite(excess).all())
        self.assertTrue(np.all(excess[1] > excess[0]))


if __name__ == "__main__":
    unittest.main()
