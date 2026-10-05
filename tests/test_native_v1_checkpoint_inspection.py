"""UNIT fake checkpoint bytes/metadata only; no trained or medical evidence."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.inspect_native_v1_checkpoint import (
    extract_metadata, inspect_candidate, inspect_known, known_candidates, main, sha, write_new_report,
)


def metadata_fixture():
    return dict(architecture_version="UNIT_architecture", method="UNIT_legacy_hiercp", framework="UNIT_PyG",
                state_dict={"UNIT_weight": object()},
                graph_config=dict(adaptive_roi_margin_mm=30., context_outer_radius_mm=28.,
                                  context_shells_mm=[4., 12., 28.]),
                model_kwargs=dict(hidden_dim=128, local_layers=3), ct_clip=[-200., 250.],
                training_signature=dict(seed=42, train_cache_files=["liver_31_MUST_NOT_INFER.pt"]),
                epoch=30, completed_epoch=40, target_epochs=40, training_complete=True,
                prototype_training_cases=["train_1", "train_2"])


class NativeMetadataUnit(unittest.TestCase):
    def test_legacy_absent_tag_retains_saved_method_and_kwargs_without_compatibility_claim(self):
        payload = metadata_fixture()
        del payload["architecture_version"]
        result = extract_metadata(payload)
        architecture = result["architecture_identity"]
        self.assertEqual(architecture["status"], "ARCHITECTURE_TAG_NOT_RECORDED")
        self.assertEqual(architecture["saved_method"], payload["method"])
        self.assertEqual(architecture["model_kwargs"], payload["model_kwargs"])
        self.assertTrue(architecture["model_kwargs_present"])
        self.assertFalse(architecture["strict_model_load_executed"])
        self.assertFalse(architecture["model_operator_compatibility_verified"])
        self.assertTrue(result["state_dict_present"])
        payload["state_dict"] = {}
        self.assertFalse(extract_metadata(payload)["state_dict_present"])

    def test_original_native30_distinguished_from_bounded30(self):
        original = metadata_fixture()
        result = extract_metadata(original, heldout_case_ids=["heldout_1"])
        self.assertEqual(result["scope"]["classification"], "ORIGINAL_NATIVE_ROI30_CONTEXT28")
        bounded = copy.deepcopy(original)
        bounded["graph_config"]["context_outer_radius_mm"] = 30.
        self.assertEqual(extract_metadata(bounded)["scope"]["classification"],
                         "BOUNDED30_CONTEXT30_NOT_ORIGINAL_NATIVE")
        del original["graph_config"]
        self.assertEqual(extract_metadata(original)["scope"]["classification"], "UNKNOWN")

    def test_filenames_and_prototype_ids_do_not_prove_training_patients(self):
        payload = metadata_fixture()
        result = extract_metadata(payload, heldout_case_ids=["liver_31"])
        self.assertEqual(result["training_case_status"], "UNKNOWN")
        self.assertIsNone(result["training_case_ids"])
        self.assertIsNone(result["training_heldout_overlap"])
        self.assertEqual(result["heldout_independence"], "UNKNOWN")
        self.assertFalse(result["prototype_cases_are_optimizer_training_proof"])
        self.assertTrue(result["state_dict_present"])
        self.assertNotIn("state_dict", result["saved_fields"])

    def test_actual_explicit_training_overlap_and_separate_prototype_overlap(self):
        payload = metadata_fixture()
        payload["training_case_ids"] = ["train_1", "heldout_1"]
        payload["prototype_training_cases"] = ["train_1", "heldout_2"]
        result = extract_metadata(payload, heldout_case_ids=["heldout_1", "heldout_2"])
        self.assertEqual(result["training_case_status"], "EXPLICIT_SAVED_IDS")
        self.assertEqual(result["training_heldout_overlap"], ["heldout_1"])
        self.assertEqual(result["prototype_heldout_overlap"], ["heldout_2"])
        self.assertEqual(result["heldout_independence"], "OVERLAP_DETECTED")

    def test_preflight_literal_train_ids_are_extracted_without_guessing(self):
        payload = metadata_fixture()
        payload["preflight_calibration"] = dict(train_case_ids=["train_1", "train_2"])
        result = extract_metadata(payload, heldout_case_ids=["heldout_1"])
        self.assertEqual(result["training_case_ids"], ["train_1", "train_2"])
        self.assertEqual(result["heldout_independence"], "NO_OVERLAP_IN_EXPLICIT_SAVED_IDS")

    def test_conflicting_or_invalid_training_fields_are_not_hidden(self):
        payload = metadata_fixture()
        payload.update(training_case_ids=["train_1"], train_case_ids=["train_2"])
        result = extract_metadata(payload, heldout_case_ids=["heldout_1"])
        self.assertEqual(result["training_case_status"], "CONFLICT")
        self.assertEqual(result["heldout_independence"], "CONFLICTING_TRAINING_EVIDENCE")
        payload["train_case_ids"] = ["train_2", "train_2"]
        with self.assertRaises(ValueError):
            extract_metadata(payload)


class KnownCheckpointUnit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="native_v1_inspection_UNIT_")
        self.root = Path(self.temp.name)
        self.specs = known_candidates(self.root)

    def tearDown(self):
        resolved = self.root.resolve()
        if resolved.parent != Path(tempfile.gettempdir()).resolve() or not resolved.name.startswith("native_v1_inspection_UNIT_"):
            raise RuntimeError("Refusing cleanup outside this test's explicit UNIT workspace")
        self.temp.cleanup()

    def create_checkpoint(self, spec):
        spec["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
        spec["checkpoint"].write_bytes(b"UNIT_checkpoint_bytes_not_a_trained_model")

    def test_default_two_historical_candidates_are_examined_without_selection(self):
        seen = []
        for spec in self.specs:
            self.create_checkpoint(spec)
        def loader(path):
            seen.append(path)
            return metadata_fixture()
        result = inspect_known(self.root, load_checkpoint=loader)
        self.assertEqual(seen, [spec["checkpoint"] for spec in self.specs])
        self.assertEqual(result["candidate_count"], 2)
        self.assertFalse(result["include_recent_native"])
        self.assertEqual(result["checkpoint_files_present"], 2)
        self.assertEqual(result["checkpoint_metadata_read"], 2)
        self.assertEqual(result["checkpoint_state_dicts_present"], 2)
        self.assertEqual(result["inspection_errors"], 0)
        self.assertTrue(all(row["file_preserved"] for row in result["candidates"]))
        self.assertFalse(result["checkpoint_selected"])
        self.assertFalse(result["training_started"])
        self.assertFalse(result["neural_forward_executed"])
        self.assertFalse(result["GPU_used"])
        self.assertTrue(all(row["metadata"]["training_case_status"] == "UNKNOWN" for row in result["candidates"]))
        self.assertTrue(all(row["adjacent_JSON"]["preflight"]["status"] == "MISSING" for row in result["candidates"]))

    def test_recent_native_is_a_separate_explicit_opt_in(self):
        self.assertEqual([spec["label"] for spec in self.specs], ["historical_full", "historical_paired_fold0"])
        expanded = known_candidates(self.root, include_recent_native=True)
        self.assertEqual([spec["label"] for spec in expanded],
                         ["historical_full", "historical_paired_fold0", "native_suite_20261003"])
        for spec in expanded:
            self.create_checkpoint(spec)
        result = inspect_known(self.root, include_recent_native=True, load_checkpoint=lambda _: metadata_fixture())
        self.assertTrue(result["include_recent_native"])
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(result["checkpoint_metadata_read"], 3)

    def test_missing_and_load_failure_are_per_file_explicit_no_fallback(self):
        self.create_checkpoint(self.specs[0])
        seen = []
        def loader(path):
            seen.append(path)
            raise RuntimeError("UNIT mmap unsupported")
        result = inspect_known(self.root, load_checkpoint=loader)
        self.assertEqual(seen, [self.specs[0]["checkpoint"]])
        self.assertEqual([row["status"] for row in result["candidates"]], ["ERROR", "MISSING"])
        self.assertEqual(result["inspection_errors"], 2)
        self.assertEqual(result["checkpoint_files_present"], 1)
        self.assertEqual(result["checkpoint_metadata_read"], 0)
        self.assertTrue(result["candidates"][0]["file_preserved"])
        self.assertIn("mmap unsupported", result["candidates"][0]["errors"][0]["message"])

    def test_checkpoint_tamper_detected_even_if_loader_returns_metadata(self):
        spec = self.specs[0]
        self.create_checkpoint(spec)
        def loader(path):
            path.write_bytes(b"UNIT_tamper_by_test_only")
            return metadata_fixture()
        result = inspect_candidate(spec, load_checkpoint=loader)
        self.assertEqual(result["status"], "ERROR")
        self.assertFalse(result["file_preserved"])
        self.assertNotEqual(result["sha256_before"], result["sha256_after"])
        self.assertEqual(result["errors"][-1]["phase"], "verify_checkpoint_preservation")

    def test_sha_bound_index_uses_literal_case_ids_and_reports_overlap(self):
        spec = self.specs[0]
        self.create_checkpoint(spec)
        index = spec["sidecars"]["cache_index"]
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(dict(entries=[dict(path="liver_31_MUST_NOT_INFER.pt",
            case_id="actual_heldout", split="train")])), encoding="utf8")
        payload = metadata_fixture()
        payload["cache_publication"] = dict(index_sha256=sha(index))
        result = inspect_candidate(spec, heldout_case_ids=["actual_heldout"], load_checkpoint=lambda _: payload)
        self.assertEqual(result["status"], "READ_CPU_METADATA")
        self.assertEqual(result["metadata"]["training_case_ids"], ["actual_heldout"])
        self.assertEqual(result["metadata"]["heldout_independence"], "OVERLAP_DETECTED")
        self.assertFalse(result["metadata"]["training_case_evidence"][0]["filename_patient_inference"])
        payload["cache_publication"]["index_sha256"] = "0"*64
        mismatch = inspect_candidate(spec, load_checkpoint=lambda _: payload)
        self.assertEqual(mismatch["status"], "ERROR")
        self.assertIn("index SHA256", mismatch["errors"][0]["message"])

    def test_reference_failure_does_not_skip_independent_checkpoint_inspection(self):
        self.create_checkpoint(self.specs[0])
        with patch("tools.inspect_native_v1_checkpoint.inspect_reference_summary", side_effect=ValueError("UNIT invalid reference")):
            result = inspect_known(self.root, reference_summary=self.root/"summary.json", load_checkpoint=lambda _: metadata_fixture())
        self.assertEqual(result["reference"]["status"], "ERROR")
        self.assertEqual(result["candidates"][0]["status"], "READ_CPU_METADATA")
        self.assertEqual(result["candidates"][0]["metadata"]["heldout_independence"], "UNKNOWN")

    def test_current_index_mapping_never_promotes_unknown_training_independence(self):
        spec = self.specs[0]
        self.create_checkpoint(spec)
        index = spec["sidecars"]["cache_index"]
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(dict(entries=[dict(path="liver_31_MUST_NOT_INFER.pt",
            case_id="actual_heldout", split="train")])), encoding="utf8")
        payload = metadata_fixture()
        result = inspect_candidate(spec, heldout_case_ids=["actual_heldout"], load_checkpoint=lambda _: payload)
        metadata = result["metadata"]
        mapping = metadata["reconstructed_current_training_mapping"]
        self.assertEqual(result["status"], "READ_CPU_METADATA")
        self.assertEqual(mapping["status"], "RECONSTRUCTED_CURRENT_MAPPING_NO_TRAINING_TIME_HASH_BINDING")
        self.assertEqual(mapping["case_ids"], ["actual_heldout"])
        self.assertEqual(mapping["current_mapping_reference_overlap"], ["actual_heldout"])
        self.assertFalse(mapping["training_time_hash_binding"])
        self.assertFalse(mapping["historical_optimizer_training_verified"])
        self.assertFalse(mapping["used_for_heldout_independence"])
        self.assertFalse(mapping["filename_patient_inference"])
        self.assertEqual(metadata["training_case_evidence"], [])
        self.assertIsNone(metadata["training_case_ids"])
        self.assertIsNone(metadata["training_heldout_overlap"])
        self.assertEqual(metadata["heldout_independence"], "UNKNOWN")
        unrelated = inspect_candidate(spec, heldout_case_ids=["other_heldout"], load_checkpoint=lambda _: payload)
        self.assertEqual(unrelated["metadata"]["reconstructed_current_training_mapping"]["current_mapping_reference_overlap"], [])
        self.assertEqual(unrelated["metadata"]["heldout_independence"], "UNKNOWN")
        payload["training_case_ids"] = ["saved_train"]
        explicit = inspect_candidate(spec, heldout_case_ids=["actual_heldout"], load_checkpoint=lambda _: payload)
        self.assertEqual(explicit["metadata"]["training_case_ids"], ["saved_train"])
        self.assertEqual(explicit["metadata"]["heldout_independence"], "NO_OVERLAP_IN_EXPLICIT_SAVED_IDS")

    def test_invalid_current_mapping_is_an_explicit_auxiliary_error_after_metadata_read(self):
        spec = self.specs[0]
        self.create_checkpoint(spec)
        index = spec["sidecars"]["cache_index"]
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(dict(entries=[dict(path="liver_31_MUST_NOT_INFER.pt",
            case_id="actual_case", split="val")])), encoding="utf8")
        result = inspect_candidate(spec, load_checkpoint=lambda _: metadata_fixture())
        mapping = result["metadata"]["reconstructed_current_training_mapping"]
        self.assertEqual(result["status"], "READ_CPU_METADATA")
        self.assertTrue(result["checkpoint_metadata_read"])
        self.assertTrue(result["file_preserved"])
        self.assertEqual(mapping["status"], "ERROR")
        self.assertEqual(mapping["errors"][0]["phase"], "reconstruct_current_cache_mapping")
        self.assertEqual(result["metadata"]["heldout_independence"], "UNKNOWN")

    def test_terminal_distinguishes_existing_weights_from_optional_legacy_metadata(self):
        for spec in self.specs:
            self.create_checkpoint(spec)
        payload = metadata_fixture()
        del payload["architecture_version"]
        output = self.root / "UNIT_cli_inspection.json"
        printed = io.StringIO()
        with patch("sys.argv", ["inspect_native_v1_checkpoint", "--medical-root", str(self.root),
                                "--output", str(output)]), \
                patch("tools.inspect_native_v1_checkpoint._cpu_load", return_value=payload), redirect_stdout(printed):
            main()
        text = printed.getvalue()
        self.assertIn("Checkpoint files: 2/2 present | metadata: 2/2 read | saved state_dict weights: 2/2", text)
        self.assertIn("ARCHITECTURE_TAG_NOT_RECORDED", text)
        self.assertIn("saved method=UNIT_legacy_hiercp", text)
        self.assertIn('"hidden_dim": 128', text)
        self.assertIn("Optional adjacent JSON absent:", text)
        self.assertIn("does not mean checkpoint weights are missing", text)
        self.assertNotIn("native_suite_20261003", text)

    def test_exclusive_new_report_never_overwrites_checkpoint_or_results(self):
        for spec in self.specs:
            self.create_checkpoint(spec)
        report = inspect_known(self.root, load_checkpoint=lambda _: metadata_fixture())
        output = self.root / "new_inspection.json"
        write_new_report(output, report)
        self.assertEqual(json.loads(output.read_text(encoding="utf8"))["candidate_count"], 2)
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            write_new_report(output, report)
        self.assertEqual(output.read_bytes(), before)
        with self.assertRaises(ValueError):
            write_new_report(self.specs[0]["checkpoint"], report)


if __name__ == "__main__":
    unittest.main()
