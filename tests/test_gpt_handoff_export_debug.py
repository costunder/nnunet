"""DEBUG: documentation completeness/hash checks; no medical/model execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import export_gpt_handoff as subject


class HandoffExportDebug(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hiercp_export_debug_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "gpt_handoff.md").write_bytes(b"Current evidence\r\n")
        (self.root / "main.py").write_bytes(b"# ===== FILE: not-a-real-file =====\r\nvalue = 1\r\n")
        (self.root / "new.py").write_bytes(b"new_value = 2\n")
        (self.root / "code.txt").write_bytes(b"OLD STALE EXPORT")
        (self.root / ".env").write_bytes(b"PRIVATE=not-exported")
        self.tracked = ["gpt_handoff.md", "main.py", "removed.py", "code.txt", ".env"]

    def fake_git(self, root, *args):
        self.assertEqual(root, self.root)
        if args == ("ls-files", "-z"):
            return b"\0".join(value.encode() for value in self.tracked) + b"\0"
        self.assertEqual(args, ("rev-parse", "HEAD"))
        return b"1" * 40 + b"\n"

    def export(self, additions):
        with patch.object(subject, "git", self.fake_git):
            return subject.export(self.root, self.root / "code.txt", additions)

    def test_complete_selected_sources_and_current_working_tree_label(self):
        result = self.export(["new.py"])
        raw = (self.root / "code.txt").read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        text = raw.decode("utf-8")
        manifest = json.loads(text.split("## Source manifest\n\n", 1)[1].split("\n\n===== FILE:", 1)[0])
        self.assertEqual(manifest["file_count"], 3)
        self.assertEqual({row["path"] for row in manifest["files"]}, {"gpt_handoff.md", "main.py", "new.py"})
        self.assertIn("not a claim", manifest["snapshot"])
        self.assertNotIn("OLD STALE EXPORT", text)
        self.assertNotIn("PRIVATE=", text)
        self.assertNotIn("\r\n", text)
        self.assertEqual(result["sha256"], hashlib.sha256(raw).hexdigest())
        for row in manifest["files"]:
            source = (self.root / row["path"]).read_bytes()
            body = source.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
            self.assertEqual(row["source_sha256"], hashlib.sha256(source).hexdigest())
            self.assertEqual(row["export_sha256"], hashlib.sha256(body).hexdigest())
            self.assertEqual(row["export_bytes"], len(body))

    def test_unselected_new_source_not_silently_exported(self):
        self.assertEqual(self.export([])["file_count"], 2)
        self.assertNotIn(b"new_value", (self.root / "code.txt").read_bytes())

    def test_missing_selected_source_preserves_old_export(self):
        with self.assertRaises(ValueError):
            self.export(["missing.py"])
        self.assertEqual((self.root / "code.txt").read_bytes(), b"OLD STALE EXPORT")

    def test_no_arbitrary_output_overwrite(self):
        with self.assertRaises(ValueError):
            subject.export(self.root, self.root / "main.py", [])
        self.assertTrue((self.root / "main.py").read_bytes().endswith(b"value = 1\r\n"))

    def test_outside_and_binary_sources_refused(self):
        with self.assertRaises(ValueError):
            subject.source_record(self.root, "../outside.py")
        (self.root / "binary.bin").write_bytes(b"\x00\x01")
        with self.assertRaises(ValueError):
            subject.source_record(self.root, "binary.bin")

    def test_source_only_notebook_export_and_rendered_data_rejection(self):
        notebook = {"nbformat": 4, "nbformat_minor": 4, "metadata": {}, "cells": [
            {"cell_type": "code", "metadata": {}, "source": ["value = 1\n"],
             "execution_count": None, "outputs": []}]}
        target = self.root / "reference.ipynb"
        target.write_text(json.dumps(notebook), encoding="utf-8")
        self.assertEqual(self.export([target.name])["file_count"], 3)
        clean_export = (self.root / "code.txt").read_bytes()
        variants = []
        for key, value in (("outputs", [{"data": {"image/png": "DEBUG_NOT_REAL_DATA"}}]),
                           ("execution_count", 1), ("metadata", {"widgets": "DEBUG"}),
                           ("attachments", {"image.png": "DEBUG"})):
            changed = copy.deepcopy(notebook)
            changed["cells"][0][key] = value
            variants.append(changed)
        changed = copy.deepcopy(notebook)
        changed["metadata"] = {"widgets": "DEBUG"}
        variants.append(changed)
        for changed in variants:
            with self.subTest(changed=changed):
                target.write_text(json.dumps(changed), encoding="utf-8")
                source_before = target.read_bytes()
                with self.assertRaises(ValueError):
                    self.export([target.name])
                self.assertEqual((self.root / "code.txt").read_bytes(), clean_export)
                self.assertEqual(target.read_bytes(), source_before)


if __name__ == "__main__":
    unittest.main()
