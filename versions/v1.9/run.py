"""v1.9 four comparison controls; preserve explicit runner arguments."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    runpy.run_path(str(ROOT / "tools/run_v19_comparison.py"), run_name="__main__")
