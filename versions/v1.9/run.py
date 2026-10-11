"""v1.9 four comparison controls; preserve explicit runner arguments."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    runpy.run_path(str(ROOT / 'versions/run.py'))['launch']('v1.9', sys.argv[1:])
