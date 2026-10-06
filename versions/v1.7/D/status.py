"""Display saved D training metrics; never starts or alters training."""
from pathlib import Path
import runpy
ROOT = Path(__file__).resolve().parents[3]
if __name__ == '__main__':
    runpy.run_path(str(ROOT/'tools/watch_v17_d_learning.py'),run_name='__main__')
