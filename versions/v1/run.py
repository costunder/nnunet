"""보존 v1 실행: unchanged original arguments/runtime."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'versions/run.py'), run_name='version_entry')['launch']('v1', sys.argv[1:])
