"""좌표 탐색 그래프: unchanged original arguments/runtime."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'versions/run.py'), run_name='version_entry')['launch']('v2.2/explore', sys.argv[1:])
