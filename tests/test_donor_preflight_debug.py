"""Debug-only synthetic header preflight tests; no clinical data or training."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import nibabel as nib
import numpy as np

from hiercp.donor_preflight import audit_donor_headers


class HeaderOnlyProxy:
    def __init__(self, shape):
        self.shape = shape

    def __array__(self, *args, **kwargs):
        raise AssertionError("Header audit must not read voxel arrays")


class DonorPreflightDebugTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="debug-donor-headers-")
        self.root = Path(self.temporary.name)
        self.report = self.root / "header_audit.json"

    def tearDown(self):
        self.temporary.cleanup()

    def case(self, case_id, *, shift=0.0):
        paths = SimpleNamespace(case_id=case_id, image_path=self.root / f"{case_id}_image.nii",
                                label_path=self.root / f"{case_id}_label.nii")
        for name in ("image", "label"):
            affine = np.eye(4)
            if name == "label":
                affine[0, 3] = shift
            image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.uint8), affine)
            image.header.set_xyzt_units("mm")
            nib.save(image, getattr(paths, name + "_path"))
        return paths

    def audit(self, cases, **kwargs):
        with redirect_stdout(io.StringIO()):
            return audit_donor_headers(case_paths=cases,
                                       selected_case_ids=kwargs.pop("selected_case_ids", [item.case_id for item in cases]),
                                       workers=kwargs.pop("workers", 2), report_path=self.report, **kwargs)

    def reports(self):
        return [json.loads(path.read_text(encoding="utf-8")) for path in self.root.glob("header_audit.*.json")
                if ".resources." not in path.name]

    def test_all_pass_without_data_array_access(self):
        cases = [self.case("first"), self.case("second"), self.case("third")]
        original_load, loaded = nib.load, []

        def load(path, **kwargs):
            image = original_load(path, **kwargs)
            loaded.append(str(path))
            image._dataobj = HeaderOnlyProxy(image.shape)
            return image

        with patch("hiercp.donor_preflight.nib.load", side_effect=load):
            result = self.audit(cases, workers="auto")
        self.assertEqual(list(result), [item.case_id for item in cases])
        self.assertEqual(len(loaded), 6)
        self.assertEqual(len(set(loaded)), 6)
        report = self.reports()[0]
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["passed_cases"], 3)
        self.assertFalse(report["voxel_arrays_read"])
        self.assertFalse(report["full_file_hashes_read"])
        self.assertEqual(report["cases"][0]["headers"]["image"]["shape"], [4, 5, 6])
        self.assertEqual(report["cases"][0]["headers"]["image"]["xyzt_units"][0], "mm")

    def test_all_bad_cases_reported_after_full_cohort(self):
        cases = [self.case("bad_first", shift=0.25), self.case("good"), self.case("bad_last", shift=-0.5)]
        with self.assertRaisesRegex(ValueError, "bad_first:.*bad_last:.*report="):
            self.audit(cases, workers="auto")
        report = self.reports()[0]
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["audited_cases"], 3)
        self.assertEqual(report["failed_case_ids"], ["bad_first", "bad_last"])
        self.assertEqual(report["passed_cases"], 1)
        self.assertIsNone(report["cases"][0]["geometry_audit"])
        self.assertTrue(report["cases"][0]["errors"])

    def test_missing_and_corrupt_headers_do_not_stop_other_cases(self):
        cases = [self.case("missing"), self.case("corrupt"), self.case("good")]
        cases[0].image_path.unlink()
        cases[1].label_path.write_bytes(b"synthetic corrupt debug header")
        with self.assertRaisesRegex(ValueError, "2/3 cases"):
            self.audit(cases)
        report = self.reports()[0]
        self.assertEqual(report["failed_case_ids"], ["missing", "corrupt"])
        self.assertIn("label", report["cases"][0]["headers"])
        self.assertIn("image", report["cases"][1]["headers"])
        self.assertEqual(report["cases"][2]["status"], "passed")

    def test_each_attempt_keeps_older_report_unchanged(self):
        cases = [self.case("one")]
        self.report.write_text("existing user file", encoding="utf-8")
        self.audit(cases)
        original_reports = {path: path.read_bytes() for path in self.root.glob("header_audit.*.json")}
        self.audit(cases)
        self.assertEqual(len(self.reports()), 2)
        self.assertEqual(self.report.read_text(encoding="utf-8"), "existing user file")
        for path, content in original_reports.items():
            self.assertEqual(path.read_bytes(), content)

    def test_truncated_compressed_header_is_aggregated_not_early_abort(self):
        cases = [self.case("truncated"), self.case("good")]
        original_load = nib.load
        for failure in (EOFError("Compressed file ended before the end-of-stream marker was reached"),
                        zlib.error("Error -3 while decompressing data: invalid block type")):
            with self.subTest(decoder_error=type(failure).__name__):
                def load(path, **kwargs):
                    if Path(path) == cases[0].image_path:
                        raise failure
                    return original_load(path, **kwargs)

                previous = {path for path in self.root.glob("header_audit.*.json")}
                with patch("hiercp.donor_preflight.nib.load", side_effect=load):
                    with self.assertRaisesRegex(ValueError, f"truncated: image header: {type(failure).__name__}"):
                        self.audit(cases)
                new_reports = [path for path in self.root.glob("header_audit.*.json")
                               if path not in previous and ".resources." not in path.name]
                self.assertEqual(len(new_reports), 1)
                report = json.loads(new_reports[0].read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["audited_cases"], 2)
                self.assertEqual(report["failed_case_ids"], ["truncated"])
                self.assertEqual(report["cases"][1]["status"], "passed")

    def test_duplicate_selected_cohort_rejected_before_loading(self):
        case = self.case("one")
        with patch("hiercp.donor_preflight.nib.load") as load:
            with self.assertRaisesRegex(ValueError, "unique"):
                self.audit([case], selected_case_ids=["one", "one"])
            load.assert_not_called()

    def test_duplicate_paths_or_mismatched_selected_cohort_rejected(self):
        case = self.case("one")
        for paths, selected in (([case, case], ["one"]), ([case], ["two"]), ([], [])):
            with self.subTest(selected=selected, count=len(paths)):
                with patch("hiercp.donor_preflight.nib.load") as load:
                    with self.assertRaises(ValueError):
                        self.audit(paths, selected_case_ids=selected)
                    load.assert_not_called()

    def test_file_mutation_during_pair_read_is_explicit_failure(self):
        case = self.case("changed")
        original_load = nib.load

        def load(path, **kwargs):
            image = original_load(path, **kwargs)
            if Path(path) == case.label_path:
                state = case.image_path.stat()
                os.utime(case.image_path, ns=(state.st_atime_ns, state.st_mtime_ns + 1_000_000))
            return image

        with patch("hiercp.donor_preflight.nib.load", side_effect=load):
            with self.assertRaisesRegex(ValueError, "file changed during header preflight"):
                self.audit([case])
        row = self.reports()[0]["cases"][0]
        self.assertEqual(row["status"], "failed")
        self.assertNotEqual(row["files"]["image"]["before"], row["files"]["image"]["after"])

    def test_unexpected_error_propagates_with_failed_resource_report(self):
        case = self.case("unexpected")
        with patch("hiercp.donor_preflight.nifti_geometry.validate_donor_geometry",
                   side_effect=RuntimeError("synthetic unexpected debug error")):
            with self.assertRaisesRegex(RuntimeError, "unexpected debug error"):
                self.audit([case])
        self.assertEqual(self.reports(), [])
        resources = list(self.root.glob("header_audit.resources.*.json"))
        self.assertEqual(len(resources), 1)
        self.assertEqual(json.loads(resources[0].read_text(encoding="utf-8"))["status"], "failed")


if __name__ == "__main__":
    unittest.main()
