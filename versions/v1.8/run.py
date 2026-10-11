"""v1.8 matched comparison positions; preserve all explicit runner arguments."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'versions/run.py'))['launch']('v1.8', sys.argv[1:])
