"""One lazy workspace location for generated test fixtures."""
from pathlib import Path


def unit_artifact_root() -> Path:
    """Create the test cache on demand without touching historical outputs."""
    root = Path(__file__).resolve().parents[1] / "work" / "cache" / "tests"
    root.mkdir(parents=True, exist_ok=True)
    return root
