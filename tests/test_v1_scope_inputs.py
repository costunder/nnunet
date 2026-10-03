"""Metadata/path UNIT fixtures only; no synthetic neural or medical claim."""
from __future__ import annotations

import copy
import csv
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

from hiercp_v1x import scope_inputs


def temporary_unit_directory():
    root = Path(__file__).resolve().parents[1] / "work/scope_UNIT_metadata"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="scope_UNIT_", dir=root)


def row(case, name="train", index="0", *, status="ok", candidates="8"):
    return dict(case_id=case, split=name, sample_index=index,
                status=status, candidates=candidates)


class ScopeRecordSelectionUnits(unittest.TestCase):
    def setUp(self):
        self.split = dict(train=["UNIT_A", "UNIT_B", "UNIT_C"], val=["UNIT_V", "UNIT_W"], outer=["UNIT_OUTER"])

    def test_exact_explicit_physical_batch_distinct_cases_plus_one_held_out(self):
        rows = [row("UNIT_C"), row("UNIT_A", index="1"), row("UNIT_A"),
                row("UNIT_B"), row("UNIT_V", "val"), row("UNIT_W", "val")]
        before = copy.deepcopy(rows)
        selected = scope_inputs.select_records(rows, self.split, 2)
        self.assertEqual([(x["case_id"], x["sample_index"]) for x in selected],
                         [("UNIT_A", "0"), ("UNIT_B", "0"), ("UNIT_V", "0")])
        self.assertEqual(rows, before)
        self.assertEqual(sum(x["split"] == "train" for x in selected), 2)
        self.assertEqual(sum(x["split"] == "val" for x in selected), 1)

    def test_larger_explicit_batch_is_not_silently_capped_at_two(self):
        selected = scope_inputs.select_records(
            [row("UNIT_A"), row("UNIT_B"), row("UNIT_C"), row("UNIT_V", "val")], self.split, 3)
        self.assertEqual(len(selected), 4)

    def test_failed_records_cannot_fill_requested_batch(self):
        rows = [row("UNIT_A"), row("UNIT_B", status="failed"), row("UNIT_V", "val")]
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            scope_inputs.select_records(rows, self.split, 2)

    def test_duplicate_case_is_not_a_second_independent_patient(self):
        rows = [row("UNIT_A"), row("UNIT_A", index="1"), row("UNIT_V", "val")]
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            scope_inputs.select_records(rows, self.split, 2)

    def test_out_of_split_success_is_rejected_even_after_quota_is_filled(self):
        rows = [row("UNIT_A"), row("UNIT_B"), row("UNIT_V", "val"), row("ZZ_OUTSIDE", "val")]
        with self.assertRaisesRegex(ValueError, "outside"):
            scope_inputs.select_records(rows, self.split, 2)

    def test_eight_original_candidates_required(self):
        for count in ("7", "1", "128", None, 8):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "eight-candidate"):
                scope_inputs.select_records([row("UNIT_A", candidates=count), row("UNIT_B"),
                                             row("UNIT_V", "val")], self.split, 2)

    def test_batch_is_explicit_integer_parallel_debug_size(self):
        for value in (True, 1, 0, -1, 2.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scope_inputs.select_records([], self.split, value)


class ScopeInputPackagingUnits(unittest.TestCase):
    """Temporary .pt names contain plain UNIT bytes and are never torch-loaded."""

    def fixture(self, base, *, member="hiercp/__init__.py"):
        repo, experiment, medical = base / "repo", base / "native", base / "medical"
        cache = experiment / "shared/cache"
        cache.mkdir(parents=True)
        (experiment / "manifest.json").write_text('{"UNIT_METADATA_ONLY": true}')
        config = {"config_fingerprint": "UNIT_config_identity"}
        (cache / "config.json").write_text(json.dumps(config))
        records = []
        for i, (case, split) in enumerate((("UNIT_A", "train"), ("UNIT_B", "train"), ("UNIT_V", "val"))):
            sample = cache / f"UNIT_sample_{i}.pt"
            sample.write_bytes(f"UNIT_METADATA_ONLY_{case}".encode())
            r = dict(row(case, split), path=sample.name, artifact_sha256=scope_inputs.sha(sample),
                     file_size=str(sample.stat().st_size), config_fingerprint=config["config_fingerprint"])
            for kind, folder, suffix in (("image", "image", "_0000.nii.gz"), ("label", "labels", ".nii.gz")):
                raw = medical / "Data" / folder / (case + suffix)
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw.write_bytes(f"UNIT_METADATA_NOT_NIFTI_{case}_{kind}".encode())
                r[f"source_{kind}_sha256"] = scope_inputs.sha(raw)
            records.append(r)
        archive = repo / "versions/v1/pipeline_v1_source.zip"
        archive.parent.mkdir(parents=True)
        with ZipFile(archive, "w") as z:
            z.writestr(member, "# UNIT archive member; never executed\n")
        manifest = {"sampling_contract": {"mode": "native"},
                    "split": dict(train=["UNIT_A", "UNIT_B"], val=["UNIT_V"], outer=["UNIT_O"]),
                    "medical_root": str(medical)}
        self.write_rows(cache, records)
        return repo, experiment, cache, manifest, records, archive

    @staticmethod
    def write_rows(cache, rows):
        with (cache / "manifest.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def signatures(root):
        return {str(p.relative_to(root)): scope_inputs.sha(p) for p in root.rglob("*") if p.is_file()}

    def run_fixture(self, setup, output):
        repo, experiment, cache, manifest, records, archive = setup
        with ExitStack() as stack:
            stack.enter_context(patch.object(scope_inputs, "ROOT", repo))
            load = stack.enter_context(patch("hiercp_v1x.experiment.load_suite", return_value=manifest))
            stack.enter_context(patch("hiercp_v1x.experiment.preparation_root", return_value=cache.parent))
            stack.enter_context(patch("hiercp_v1x.contracts.verify_archive", return_value={"archive_sha256": scope_inputs.sha(archive)}))
            result = scope_inputs.prepare_debug_inputs(experiment, output, 2)
            load.assert_called_once_with(experiment.resolve())
            return result

    def test_read_only_debug_packaging_bindings_and_no_completion_promotion(self):
        with temporary_unit_directory() as name:
            base = Path(name)
            setup = self.fixture(base)
            before = self.signatures(base / "native")
            source, samples = self.run_fixture(setup, base / "DEBUG_OUTPUT")
            result = json.loads((samples / "fixture_manifest.json").read_text())
            self.assertEqual(self.signatures(base / "native"), before)
            self.assertEqual(len(result["files"]), 3)
            self.assertEqual(len(result["source_records"]), 3)
            for key in ("source_preparation_complete_claimed", "full_training", "quality_verified", "production_ready"):
                self.assertIs(result[key], False)
            self.assertIs(result["debug"], True)
            self.assertTrue((source / "hiercp/__init__.py").is_file())
            for item in result["files"]:
                self.assertEqual(scope_inputs.sha(samples / item["name"]), item["sha256"])
            self.assertFalse(any(base.rglob("checkpoint*")))

    def test_existing_output_and_overlapping_source_are_preserved(self):
        with temporary_unit_directory() as name:
            base = Path(name)
            setup = self.fixture(base)
            existing = base / "EXISTING_DEBUG"
            existing.mkdir()
            (existing / "keep.txt").write_text("UNIT keep")
            with self.assertRaises(FileExistsError):
                self.run_fixture(setup, existing)
            self.assertEqual((existing / "keep.txt").read_text(), "UNIT keep")
            with self.assertRaisesRegex(ValueError, "disjoint"):
                self.run_fixture(setup, setup[1] / "nested_debug")

    def test_non_native_owner_cannot_be_relabelled_as_scope_baseline(self):
        with temporary_unit_directory() as name:
            base = Path(name)
            setup = self.fixture(base)
            setup[3]["sampling_contract"]["mode"] = "strict_nested"
            with self.assertRaisesRegex(ValueError, "native preparation owner"):
                self.run_fixture(setup, base / "DEBUG_OUTPUT")

    def test_artifact_hash_size_config_and_raw_provenance_reject_corruption(self):
        for kind in ("hash", "size", "config", "image", "label"):
            with self.subTest(kind=kind), temporary_unit_directory() as name:
                base = Path(name)
                setup = self.fixture(base)
                r = setup[4][0]
                if kind == "hash": r["artifact_sha256"] = "0" * 64
                elif kind == "size": r["file_size"] = "999999"
                elif kind == "config": r["config_fingerprint"] = "OTHER"
                else: r[f"source_{kind}_sha256"] = "0" * 64
                self.write_rows(setup[2], setup[4])
                with self.assertRaises(ValueError):
                    self.run_fixture(setup, base / "DEBUG_OUTPUT")
                self.assertFalse((base / "DEBUG_OUTPUT").exists())

    def test_cache_filename_paths_cannot_escape_or_use_windows_drive(self):
        for unsafe in ("../escape.pt", "/escape.pt", "sub/sample.pt", "sub\\sample.pt", "C:escape.pt", ".."):
            with self.subTest(unsafe=unsafe), temporary_unit_directory() as name:
                base = Path(name)
                setup = self.fixture(base)
                setup[4][0]["path"] = unsafe
                self.write_rows(setup[2], setup[4])
                with self.assertRaisesRegex(ValueError, "contained|unsafe"):
                    self.run_fixture(setup, base / "DEBUG_OUTPUT")

    def test_archive_members_cannot_escape_or_use_drive_backslash(self):
        for unsafe in ("../escape.py", "/escape.py", "C:/escape.py"):
            with self.subTest(unsafe=unsafe), temporary_unit_directory() as name:
                base = Path(name)
                setup = self.fixture(base, member=unsafe)
                before = self.signatures(base / "native")
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    self.run_fixture(setup, base / "DEBUG_OUTPUT")
                self.assertEqual(self.signatures(base / "native"), before)
                self.assertFalse((base / "escape.py").exists())

    def test_archive_guard_rejects_raw_backslash_member_before_extraction(self):
        # Windows ZipInfo normalizes backslashes during construction/read, so
        # exercise the guard's raw incoming metadata independently of ZipInfo.
        with temporary_unit_directory() as name:
            base = Path(name)
            setup = self.fixture(base)
            archive = MagicMock()
            archive.__enter__.return_value.infolist.return_value = [
                SimpleNamespace(filename="hiercp\\escape.py")]
            with patch.object(scope_inputs, "ZipFile", return_value=archive), \
                 self.assertRaisesRegex(ValueError, "unsafe"):
                self.run_fixture(setup, base / "DEBUG_OUTPUT")
            archive.__enter__.return_value.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
