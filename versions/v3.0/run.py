"""Short entry point for the explicit v3.0 arm and physical GPU CLI."""
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT/'versions/run.py'))['launch']('v3.0',sys.argv[1:])
