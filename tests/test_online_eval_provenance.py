"""Debug-only byte fixtures; no medical data or model execution is involved."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.online_eval_provenance import (
    EvaluationProvenanceError, build_evaluation_contract, contract_comparability,
    full_method_identity, prepare_new_output, verify_evaluation_contract, _hash_file,
)


class EvaluationProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="debug_eval_provenance_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cohort = ["case_2", "case_10"]
        self.files = {}
        for role in ("ground_truth", "basic", "hier"):
            directory = self.root / role
            directory.mkdir()
            self.files[role] = {}
            for case_id in self.cohort:
                path = directory / f"{case_id}.bytes"
                path.write_bytes(f"debug fixture {role}/{case_id}".encode())
                self.files[role][case_id] = path

    def build(self, **overrides):
        arguments = {
            "cohort": self.cohort, "ground_truth_files": self.files["ground_truth"],
            "prediction_files": {role: self.files[role] for role in ("basic", "hier")},
            "trainers": {"basic": "BasicTrainer", "hier": "FullTrainer"},
            "evaluation_definition": {"connectivity": 26, "thresholds": [0.1, 0.25]},
        }
        arguments.update(overrides)
        return build_evaluation_contract(**arguments)

    @staticmethod
    def resign(contract):
        payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        contract["contract_sha256"] = hashlib.sha256(encoded).hexdigest()
        return contract

    def test_valid_roundtrip_and_source_mutation(self):
        contract = json.loads(json.dumps(self.build()))
        verify_evaluation_contract(contract)
        path = self.files["hier"]["case_2"]
        content = path.read_bytes()
        path.write_bytes(b"X" + content[1:])
        with self.assertRaisesRegex(EvaluationProvenanceError, "bytes changed"):
            verify_evaluation_contract(contract)

    @staticmethod
    def hash_metadata(path):
        values = path.stat()
        return {
            field: getattr(values, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
        }

    def test_hash_accepts_stable_cross_api_ctime_difference(self):
        path = self.files["ground_truth"]["case_2"]
        path_metadata = self.hash_metadata(path)
        fd_metadata = {**path_metadata, "st_ctime_ns": path_metadata["st_ctime_ns"] + 100}
        with patch.object(Path, "stat", return_value=SimpleNamespace(**path_metadata)):
            with patch("tools.online_eval_provenance.os.fstat", return_value=SimpleNamespace(**fd_metadata)):
                actual = _hash_file(path)
        self.assertEqual(actual, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_hash_rejects_changes_within_either_metadata_api(self):
        path = self.files["ground_truth"]["case_2"]
        metadata = self.hash_metadata(path)
        stable = SimpleNamespace(**metadata)
        changed = SimpleNamespace(**{**metadata, "st_ctime_ns": metadata["st_ctime_ns"] + 100})
        for changed_api in ("path", "fd"):
            with self.subTest(changed_api=changed_api):
                path_values = [stable, changed if changed_api == "path" else stable]
                fd_values = [stable, changed if changed_api == "fd" else stable]
                with patch.object(Path, "stat", side_effect=path_values):
                    with patch("tools.online_eval_provenance.os.fstat", side_effect=fd_values):
                        with self.assertRaisesRegex(EvaluationProvenanceError, "changed while hashing"):
                            _hash_file(path)

    def test_hash_rejects_cross_api_file_identity_or_size_mismatch(self):
        path = self.files["ground_truth"]["case_2"]
        metadata = self.hash_metadata(path)
        for field in ("st_dev", "st_ino", "st_size"):
            with self.subTest(field=field):
                other = SimpleNamespace(**{**metadata, field: metadata[field] + 1})
                with patch.object(Path, "stat", return_value=SimpleNamespace(**metadata)):
                    with patch("tools.online_eval_provenance.os.fstat", return_value=other):
                        with self.assertRaisesRegex(EvaluationProvenanceError, "changed while hashing"):
                            _hash_file(path)

    def test_checksum_tampering_raises_even_with_missing_reference(self):
        contract = self.build()
        contract["evaluation_definition"]["connectivity"] = 6
        with self.assertRaisesRegex(EvaluationProvenanceError, "checksum"):
            contract_comparability(None, contract)

    def test_changed_trainer_prediction_cohort_and_definition(self):
        old = self.build()
        changed_trainer = self.build(trainers={"basic": "OtherTrainer", "hier": "FullTrainer"})
        self.assertEqual(contract_comparability(old, changed_trainer)["status"], "not_comparable")
        changed_order = self.build(cohort=list(reversed(self.cohort)))
        self.assertIn("cohort changed", contract_comparability(old, changed_order)["reasons"])
        changed_definition = self.build(evaluation_definition={"connectivity": 6})
        self.assertEqual(contract_comparability(old, changed_definition)["status"], "not_comparable")
        self.files["hier"]["case_2"].write_bytes(b"new debug prediction")
        changed_prediction = self.build()
        self.assertIn("methods.hier.predictions changed", contract_comparability(old, changed_prediction)["reasons"])

    def test_historical_comparison_does_not_read_old_paths(self):
        old = self.build()
        current = copy.deepcopy(old)
        for entry in old["file_inventory"]:
            Path(entry["path"]).unlink()
        self.assertEqual(contract_comparability(old, current)["status"], "matched_inputs")
        with self.assertRaises(EvaluationProvenanceError):
            verify_evaluation_contract(current)

    def test_full_identity_excludes_comparator_and_locations(self):
        old = self.build()
        self.files["basic"]["case_2"].write_bytes(b"changed comparator only")
        current = self.build(trainers={"basic": "AblationTrainer", "hier": "FullTrainer"})
        self.assertEqual(full_method_identity(old, "hier"), full_method_identity(current, "hier"))
        moved = copy.deepcopy(old)
        moved["methods"]["hier"]["validation_dir"] = "/unavailable/hier"
        for entry in moved["file_inventory"]:
            if entry["role"] == "hier":
                entry["path"] = f"/unavailable/hier/{entry['case_id']}.bytes"
        self.resign(moved)
        self.assertEqual(full_method_identity(old, "hier"), full_method_identity(moved, "hier"))
        self.assertEqual(contract_comparability(old, moved)["status"], "matched_inputs")

    def test_missing_duplicate_extra_and_inconsistent_inventory_rejected(self):
        contract = self.build()
        variants = []
        missing = copy.deepcopy(contract)
        del missing["file_inventory"]
        variants.append(missing)
        duplicate = copy.deepcopy(contract)
        duplicate["file_inventory"][1] = copy.deepcopy(duplicate["file_inventory"][0])
        variants.append(duplicate)
        extra = copy.deepcopy(contract)
        extra["file_inventory"].append(copy.deepcopy(extra["file_inventory"][0]))
        variants.append(extra)
        disagreement = copy.deepcopy(contract)
        disagreement["file_inventory"][0]["sha256"] = "a" * 64
        variants.append(disagreement)
        for variant in variants:
            with self.subTest(variant=variant):
                with self.assertRaises(EvaluationProvenanceError):
                    contract_comparability(contract, self.resign(variant))

    def test_safe_exact_cohort_and_maps(self):
        for cohort in (["case_2", "case_2"], ["../case_2"], ["a/b"], ["a\\b"], []):
            with self.subTest(cohort=cohort):
                with self.assertRaises(EvaluationProvenanceError):
                    self.build(cohort=cohort)
        with self.assertRaises(EvaluationProvenanceError):
            self.build(ground_truth_files={"case_2": self.files["ground_truth"]["case_2"]})

    def test_legacy_result_is_unverified(self):
        contract = self.build()
        self.assertEqual(contract_comparability(None, contract)["status"], "unverified_provenance")
        self.assertEqual(contract_comparability({"format": "legacy"}, contract)["status"], "unverified_provenance")

    def test_foreign_host_paths_compare_without_becoming_relative_reads(self):
        contract = self.build()
        foreign = copy.deepcopy(contract)
        foreign["file_inventory"][0]["path"] = (
            "/foreign/case_2.bytes" if os.name == "nt" else "C:/foreign/case_2.bytes"
        )
        self.resign(foreign)
        self.assertEqual(contract_comparability(contract, foreign)["status"], "matched_inputs")
        with patch("tools.online_eval_provenance._hash_file", side_effect=AssertionError("Unexpected file read")):
            with self.assertRaisesRegex(EvaluationProvenanceError, "on this host"):
                verify_evaluation_contract(foreign)

    def test_output_no_clobber_including_empty_directory_and_file(self):
        output = self.root / "new_run"
        self.assertEqual(prepare_new_output(output), output)
        with self.assertRaises(EvaluationProvenanceError):
            prepare_new_output(output)
        marker = output / "existing.bytes"
        marker.write_bytes(b"keep")
        with self.assertRaises(EvaluationProvenanceError):
            prepare_new_output(output)
        with self.assertRaises(EvaluationProvenanceError):
            prepare_new_output(marker)
        self.assertEqual(marker.read_bytes(), b"keep")

    def test_symlink_keeps_original_path_when_platform_allows(self):
        original = self.files["ground_truth"]["case_2"]
        link = original.with_name("case_2_link.bytes")
        try:
            link.symlink_to(original)
        except OSError as exc:
            self.skipTest(f"Platform does not permit a debug fixture symlink: {exc}")
        self.files["ground_truth"]["case_2"] = link
        contract = self.build()
        entry = next(row for row in contract["file_inventory"] if row["role"] == "ground_truth" and row["case_id"] == "case_2")
        self.assertEqual(entry["path"], str(link))
        verify_evaluation_contract(contract)
        original.write_bytes(b"symlink target changed")
        with self.assertRaisesRegex(EvaluationProvenanceError, "bytes changed"):
            verify_evaluation_contract(contract)


if __name__ == "__main__":
    unittest.main()
