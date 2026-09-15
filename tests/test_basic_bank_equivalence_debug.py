"""DEBUG synthetic native raw-bank equivalence, never historical training proof.

Every usable source has the production 128 candidates. No model, medical case,
checkpoint, GPU training, or experiment-scale reduction is involved.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from custom_trainers.onlinecp_raw_bank import (ENTRY_STORAGE, PASTE_CONTRACT, RawBankStore,
                                               save_candidate, save_case)
from custom_trainers.onlinecp_raw_resampling import apply_candidate
from tests.test_online_raw_bank_preparation_debug import fixture
from tools import basic_bank_equivalence as comparison
from tools import online_cp_benchmark as online
from tools.online_raw_bank_preparation import prepare_raw_case, prepare_source_candidates


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _publish(root, config, index, rows, architecture):
    _write(root / "config.json", config)
    _write(root / "index.json", index)
    _write(root / "feedback_contract.json", {"debug_original_architecture": architecture})
    fields = sorted({key for row in rows for key in row})
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write(root / "complete.json", {"format": online.BANK_FORMAT,
        "index_sha256": comparison._sha(root / "index.json"),
        "config_sha256": comparison._sha(root / "config.json"),
        "manifest_sha256": comparison._sha(root / "manifest.csv"),
        "eligible_inventory_sha256": online.value_sha256(index["eligible_sources_by_case"]),
        "eligible_cases": index["eligible_cases"], "source_entries": index["source_entries"],
        "candidate_count": 128})
    return {"format": comparison.PROOF_FORMAT,
        "index_sha256": comparison._sha(root / "index.json"),
        "config_sha256": comparison._sha(root / "config.json"),
        "feedback_contract_sha256": comparison._sha(root / "feedback_contract.json"),
        "original_identity_sha256": _digest("DEBUG independently-authenticated original " + architecture)}


def _bank(root, *, architecture, scores=None, permute=False, change_candidate=False,
          change_baseline=False, repack=False, cp_probability=.5):
    image, labels, properties, plans, data, seg = fixture()
    root.mkdir(parents=True, exist_ok=True)
    case, case_path, case_sha = prepare_raw_case(root, "debug", image, labels, properties,
        plans, data, seg, configuration_name="debug", raw_spacing=[.7] * 3,
        raw_spatial_unit="mm", minimum_free_bytes=0)  # Explicit tiny DEBUG I/O reserve.
    if change_baseline:
        case["baseline_unclipped"] = case["baseline_unclipped"].copy()
        case["baseline_unclipped"].flat[0] += .125
    # Real producer storage, but deliberately different publication addresses.
    case_path = "published_" + architecture + "/case.json"
    case_sha = save_case(root, case_path, case)
    centers = np.asarray(list(np.ndindex((6, 6, 6)))[:128], dtype=np.int64) + 3
    source = image[11:16, 11:16, 11:16]
    mask = labels[11:16, 11:16, 11:16] == 2
    paths, hashes, _ = prepare_source_candidates(root, "debug", 1, case, case_sha, source,
        mask, [2, 2, 2], centers, [12] * 3,
        candidate_map=lambda function, tasks: [function(task) for task in tasks])
    paths, hashes = [str(value) for value in paths], [str(value) for value in hashes]
    if change_candidate:
        store = RawBankStore(root)
        try:
            candidate = dict(store.load_candidate(paths[-1], hashes[-1]))
            candidate["recipient_ct"] = candidate["recipient_ct"].copy()
            candidate["recipient_ct"].flat[0] += 1
            paths[-1] = "changed/candidate127.json"
            hashes[-1] = save_candidate(root, paths[-1], candidate)
        finally:
            store.close()
    if repack:
        manifest_path = root / paths[0]
        manifest = json.loads(manifest_path.read_text())
        archive = root / manifest["archive"]["path"]
        with zipfile.ZipFile(archive) as original:
            members = [(name, original.read(name)) for name in original.namelist()]
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
            for name, payload in members:
                member = zipfile.ZipInfo(name, date_time=(2026, 9, 12, 12, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                target.writestr(member, payload)
        manifest["archive"]["sha256"] = comparison._sha(archive)
        _write(manifest_path, manifest)
        hashes[0] = comparison._sha(manifest_path)
    scores = np.asarray(scores if scores is not None else np.linspace(-2, 3, 128), dtype=np.float32)
    if permute:
        order = np.arange(128)[::-1]
        centers, scores = centers[order], scores[order]
        paths, hashes = [paths[i] for i in order], [hashes[i] for i in order]
    native_centers = np.rint((centers[:, ::-1] + .5) * np.asarray([12] * 3) / 17 - .5).astype(np.int64)
    entry_name = f"entries/{architecture}_source1.npz"
    (root / "entries").mkdir()
    np.savez_compressed(root / entry_name, paste_contract=np.asarray([PASTE_CONTRACT]),
        case_id=np.asarray(["debug"]), source_component=np.asarray([1], dtype=np.int64),
        source_diameter_mm=np.asarray([2.]), candidate_raw_centers=centers,
        candidate_centers=native_centers, scores=scores, candidate_payloads=np.asarray(paths),
        candidate_payload_sha256=np.asarray(hashes), raw_case_reference=np.asarray([case_path]),
        raw_case_reference_sha256=np.asarray([case_sha]))
    rows = [{"case_id": "debug", "source_component": 1, "status": "ok", "entry": entry_name,
             "candidate_count": 128, "entry_sha256": comparison._sha(root / entry_name),
             "candidate_pool_sha256": online.candidate_pool_hash(centers, native_centers, scores)},
            {"case_id": "debug_empty", "source_component": -1, "status": "no_eligible_source",
             "entry": "", "candidate_count": 0}]
    config = {"format": online.BANK_FORMAT, "version": online.VERSION,
        "paste_contract": PASTE_CONTRACT, "entry_storage": ENTRY_STORAGE,
        "candidate_count": 128, "hier_top_k": 8, "no_placement_policy": "error",
        "tumor_label": 2, "liver_label": 1, "cp_probability": cp_probability,
        "intensity_scale_range": [.8, 1.2], "intensity_shift_range_hu": [-50., 50.],
        "normalization": {"mean": 0., "std": 200.}, "network_patch_size": [12] * 3,
        "raw_resampling_sha256": comparison._sha(Path(online.__file__).parents[1] /
                                                    "custom_trainers/onlinecp_raw_resampling.py"),
        "checkpoint_sha256": _digest(architecture), "prototype_sha256": _digest(architecture + " proto"),
        "train_config_sha256": _digest(architecture + " config"),
        "preprocess_marker_sha256": _digest(architecture + " preprocess marker"),
        "preprocess_contract_sha256": _digest(architecture + " preprocess contract")}
    inventory = {"debug": [1], "debug_empty": []}
    index = {**config, "entries_by_case": {"debug": [entry_name]},
        "eligible_sources_by_case": inventory, "no_eligible_cases": ["debug_empty"],
        "eligible_cases": 1, "source_entries": 1, "total_candidates": 128,
        "manifest": str((root / "manifest.csv").resolve()),
        "scoring_execution": {"debug_gnn_provenance": architecture}}
    proof = _publish(root, config, index, rows, architecture)
    return {"root": root, "index": root / "index.json", "proof": proof,
            "config": config, "metadata": index, "rows": rows, "architecture": architecture,
            "case": (case_path, case_sha), "candidates": list(zip(paths, hashes))}


def _republish(bank):
    bank["proof"] = _publish(bank["root"], bank["config"], bank["metadata"], bank["rows"], bank["architecture"])


class BasicBankEquivalenceDebug(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        comparison._CACHE.clear()
        comparison._NPY_PROOFS.clear()

    def tearDown(self):
        comparison._CACHE.clear()
        comparison._NPY_PROOFS.clear()
        self.directory.cleanup()

    def pair(self, **kwargs):
        return (_bank(self.root / "old", architecture="debug_v3"),
                _bank(self.root / "new", architecture="debug_v4", **kwargs))

    def compare(self, old, new):
        return comparison.compare_basic_banks(old["index"], new["index"],
            source_proof=old["proof"], target_proof=new["proof"])

    def test_real_native_128_storage_gnn_scores_paths_zip_and_warm_receipt(self):
        old, new = self.pair(scores=np.linspace(90, -50, 128), repack=True)
        receipt = self.compare(old, new)
        self.assertIs(receipt["equal"], True)
        self.assertFalse(receipt["checkpoint_reuse_authorized"])
        self.assertEqual(len(receipt["ordered_cases"]), 2)
        self.assertEqual(len(receipt["ordered_cases"][0]["source_slots"][0]["candidate_sha256"]), 128)
        self.assertEqual(receipt["ordered_cases"][1]["source_slots"], [])
        self.assertIn("full_native_preprocessing_images_segmentations_properties_splits_and_raw_case_hashes_equal_before_full",
                      receipt["required_external_checks"])
        with patch.object(RawBankStore, "_load", side_effect=AssertionError("warm payload decoded again")):
            self.assertEqual(receipt, self.compare(old, new))
        # Compare real native paste data/seg/support for fixed draws/jitter.
        left, right = RawBankStore(old["root"]), RawBankStore(new["root"])
        try:
            a, b = left.load_case(*old["case"]), right.load_case(*new["case"])
            for index, scale, shift in ((0, .8, -50.), (63, 1., 0.), (127, 1.2, 50.)):
                ca = left.load_candidate(*old["candidates"][index])
                cb = right.load_candidate(*new["candidates"][index])
                crop = [[0, int(n)] for n in a["metadata"]["preprocessed_shape"]]
                pa = apply_candidate(a, ca, crop, scale=scale, shift_hu=shift)
                pb = apply_candidate(b, cb, crop, scale=scale, shift_hu=shift)
                for field in ("data", "seg", "pasted_support"):
                    np.testing.assert_array_equal(pa[field], pb[field])
        finally:
            left.close()
            right.close()

    def test_candidate_order_is_not_distribution_only_equivalence(self):
        old, new = self.pair(permute=True)
        with self.assertRaisesRegex(ValueError, "semantic mismatch"):
            self.compare(old, new)

    def test_last_candidate_and_background_are_not_skipped(self):
        for parameter in ("change_candidate", "change_baseline"):
            with self.subTest(parameter=parameter), tempfile.TemporaryDirectory() as directory:
                old = _bank(Path(directory) / "old", architecture="debug_v3")
                new = _bank(Path(directory) / "new", architecture="debug_v4", **{parameter: True})
                with self.assertRaisesRegex(ValueError, "semantic mismatch"):
                    self.compare(old, new)

    def test_cp_conditions_and_unknown_config_are_not_ignored(self):
        old, new = self.pair(cp_probability=.6)
        with self.assertRaisesRegex(ValueError, "config.cp_probability"):
            self.compare(old, new)
        new["config"]["cp_probability"] = new["metadata"]["cp_probability"] = .5
        new["config"]["unknown_consumer_option"] = new["metadata"]["unknown_consumer_option"] = True
        _republish(new)
        with self.assertRaisesRegex(ValueError, "keys differ"):
            self.compare(old, new)

    def test_nonfinite_scores_still_fail(self):
        scores = np.arange(128, dtype=np.float32)
        scores[-1] = np.nan
        old, new = self.pair(scores=scores)
        with self.assertRaisesRegex((ValueError, online.OnlineBenchmarkError),
                                    "score|finite|Invalid raw-target candidate arrays"):
            self.compare(old, new)

    def test_original_proof_and_complete_binding_not_waived(self):
        old, new = self.pair()
        new["proof"]["config_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.compare(old, new)
        _republish(new)
        complete = json.loads((new["root"] / "complete.json").read_text())
        complete["manifest_sha256"] = "0" * 64
        _write(new["root"] / "complete.json", complete)
        with self.assertRaisesRegex(ValueError, "complete"):
            self.compare(old, new)

    def test_extra_entry_and_warm_payload_tamper_fail(self):
        old, new = self.pair()
        self.compare(old, new)
        extra = new["root"] / "entries/foreign.npz"
        extra.write_bytes(b"DEBUG unrelated artifact")
        with self.assertRaisesRegex(ValueError, "entr"):
            self.compare(old, new)
        extra.unlink()  # Own temporary DEBUG artifact only.
        path = next((new["root"] / "raw_sources").glob("*.npy"))
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            value = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([value[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.compare(old, new)

    def test_inventory_and_duplicate_json_reject(self):
        old, new = self.pair()
        new["metadata"]["eligible_sources_by_case"]["omitted_patient"] = []
        _republish(new)
        with self.assertRaisesRegex(ValueError, "inventory|manifest|source"):
            self.compare(old, new)
        content = (new["root"] / "config.json").read_text()
        content = content.replace("{", '{"cp_probability": 0.5,', 1)
        (new["root"] / "config.json").write_text(content, encoding="utf-8")
        new["proof"]["config_sha256"] = comparison._sha(new["root"] / "config.json")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            self.compare(old, new)


if __name__ == "__main__":
    unittest.main()
