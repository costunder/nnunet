"""Cold native CLI entry with exact per-event segmentation crop reuse.

This is the original nnUNet CLI argument tail, including its explicit --c
continuation option. The caller must assign the original GPU1/four CPUs and
a fresh result namespace containing the admitted checkpoint copies.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True


def main():
    from hiercp_v1x.v24_native_execution_runtime import run_training_entry
    return run_training_entry()


if __name__=='__main__':main()
