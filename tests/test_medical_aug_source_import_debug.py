"""DEBUG source provenance/privacy checks, no medical data or notebook execution."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

from tools.export_gpt_handoff import source_record


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference" / "medical_data_aug"


class MedicalAugSourceImportDebugTests(unittest.TestCase):
    def test_all_restored_sources_match_recorded_digest(self):
        manifest = json.loads((REFERENCE / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["format"], "medical_data_aug_source_import_v1")
        self.assertEqual({row["zip_member"] for row in manifest["imported"]}, {
            "3d_copy_paste_tumor.py", "README.md", "view_nifti_liver.ipynb"})
        for row in manifest["imported"]:
            path = REFERENCE / row["repository_file"]
            self.assertEqual(path.parent, REFERENCE)
            record, _ = source_record(ROOT, path.relative_to(ROOT).as_posix())
            self.assertEqual(record["export_sha256"], row["repository_lf_sha256"])

    def test_all_original_notebook_cells_preserved_without_patient_outputs(self):
        manifest = json.loads((REFERENCE / "provenance.json").read_text(encoding="utf-8"))
        path = REFERENCE / "view_nifti_liver.ipynb"
        _, body = source_record(ROOT, path.relative_to(ROOT).as_posix())
        notebook = json.loads(body)
        self.assertEqual(manifest["notebook_guard_cells"], 2)
        self.assertEqual(manifest["notebook_original_cell_count"], 24)
        self.assertEqual(len(notebook["cells"]), 26)
        guard = ast.parse("".join(notebook["cells"][1]["source"]))
        self.assertIsInstance(guard.body[0], ast.Raise)
        original = notebook["cells"][2:]
        self.assertEqual([hashlib.sha256("".join(c["source"]).encode()).hexdigest()
                          for c in original], manifest["notebook_original_cell_sources_sha256"])
        for cell in original:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), feature_version=(3, 10))

    def test_only_reviewed_sources_are_in_reference_directory(self):
        self.assertEqual({p.name for p in REFERENCE.iterdir()}, {
            "README.md", "README.original.md", "provenance.json",
            "3d_copy_paste_tumor.py.txt", "view_nifti_liver.ipynb"})
        self.assertNotIn("drive.google.com", (REFERENCE / "README.original.md").read_text())
        self.assertIn("/Medical Data Aug.zip", (ROOT / ".gitignore").read_text())


if __name__ == "__main__":
    unittest.main()
