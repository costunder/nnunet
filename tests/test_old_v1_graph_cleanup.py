"""UNIT tests operate only on newly created workspace temporary files."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.retire_old_v1_graphs import APPROVED, apply, plan


class OldGraphCleanupTests(unittest.TestCase):
    def fixture(self):
        temp = TemporaryDirectory(prefix="UNIT_old_graph_cleanup_", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        for relative, count in APPROVED:
            directory = root / relative
            directory.mkdir(parents=True)
            entries = []
            for i in range(count):
                name = f"liver_{i}__000.pt"
                (directory / name).write_bytes(b"UNIT generated disposable cache")
                entries.append(dict(case_id=f"liver_{i}", sample_index=0, path=name,
                                    file_size=(directory / name).stat().st_size))
            (directory / "index.json").write_text(json.dumps(dict(entries=entries)), encoding="utf8")
            (directory / "model.pt").write_bytes(b"UNIT protected model")
        protected = root / "experiments" / "v19_native_fixed_m10_seed42" / "data"
        protected.mkdir(parents=True)
        (protected / "active.pt").write_bytes(b"UNIT protected active cache")
        return root

    def test_only_609_indexed_samples_removed_and_rerun_safe(self):
        root = self.fixture()
        planned = plan(root)
        self.assertTrue((root / APPROVED[0][0] / "liver_0__000.pt").is_file())
        result = apply(planned, lambda line: None)
        self.assertEqual(result["removed"], 609)
        for relative, _ in APPROVED:
            directory = root / relative
            self.assertTrue((directory / "index.json").is_file())
            self.assertEqual((directory / "model.pt").read_bytes(), b"UNIT protected model")
        self.assertTrue((root / "experiments/v19_native_fixed_m10_seed42/data/active.pt").is_file())
        repeat = apply(plan(root), lambda line: None)
        self.assertEqual((repeat["removed"], repeat["already_absent"]), (0, 609))

    def test_shared_hardlink_refused_before_any_deletion(self):
        root = self.fixture()
        sample = root / APPROVED[-1][0] / "liver_0__000.pt"
        os.link(sample, root / "UNIT_link.pt")
        with self.assertRaisesRegex(ValueError, "Shared hardlink"):
            plan(root)
        self.assertTrue((root / APPROVED[0][0] / "liver_0__000.pt").is_file())

    def test_changed_index_refused_before_any_deletion(self):
        root = self.fixture()
        planned = plan(root)
        index = root / APPROVED[-1][0] / "index.json"
        index.write_text(index.read_text() + " ", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "Index changed"):
            apply(planned)
        self.assertTrue((root / APPROVED[0][0] / "liver_0__000.pt").is_file())

    def test_changed_sample_refused_before_any_deletion(self):
        root = self.fixture()
        planned = plan(root)
        (root / APPROVED[-1][0] / "liver_0__000.pt").write_bytes(b"UNIT changed")
        with self.assertRaisesRegex(ValueError, "Sample changed"):
            apply(planned)
        self.assertTrue((root / APPROVED[0][0] / "liver_0__000.pt").is_file())

    def test_unsafe_filename_and_duplicate_refused(self):
        root = self.fixture()
        index = root / APPROVED[0][0] / "index.json"
        data = json.loads(index.read_text())
        data["entries"][0]["path"] = "../../model.pt"
        index.write_text(json.dumps(data), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "Unexpected or duplicate"):
            plan(root)
        data["entries"][0] = data["entries"][1]
        index.write_text(json.dumps(data), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "Unexpected or duplicate"):
            plan(root)

    def test_wrong_inventory_count_and_changed_plan_refused(self):
        root = self.fixture()
        planned = plan(root)
        planned["groups"][0]["directory"] = str(root / "experiments")
        with self.assertRaisesRegex(ValueError, "approved three-directory"):
            apply(planned)
        index = root / APPROVED[0][0] / "index.json"
        data = json.loads(index.read_text())
        data["entries"].pop()
        index.write_text(json.dumps(data), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "inventory count"):
            plan(root)

    def test_protected_model_cannot_replace_an_indexed_sample_in_plan(self):
        root = self.fixture()
        planned = plan(root)
        planned["groups"][0]["samples"][0]["name"] = "model.pt"
        with self.assertRaisesRegex(ValueError, "Planned sample names"):
            apply(planned)
        self.assertTrue((root / APPROVED[0][0] / "model.pt").is_file())


if __name__ == "__main__":
    unittest.main()
