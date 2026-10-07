"""Directory organization checks only; no model or data processing."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.artifacts import unit_artifact_root


class ArtifactLocationUnits(unittest.TestCase):
    def test_import_does_not_create_directories(self):
        source = Path(__file__).with_name("artifacts.py")
        spec = importlib.util.spec_from_file_location("UNIT_artifacts_import", source)
        module = importlib.util.module_from_spec(spec)
        with patch.object(Path, "mkdir", side_effect=AssertionError("import wrote a directory")):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.unit_artifact_root))

    def test_location_is_created_lazily_and_reused(self):
        expected = Path(__file__).resolve().parents[1] / "work" / "cache" / "tests"
        actual = unit_artifact_root()
        self.assertEqual(actual, expected)
        self.assertTrue(actual.is_dir())
        self.assertEqual(unit_artifact_root(), actual)


if __name__ == "__main__":
    unittest.main()
