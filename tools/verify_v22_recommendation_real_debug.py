"""Current geometry/final-artifact DEBUG test; replaces the old hard-coded cache test."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache',type=Path,help='Rebuilt review_repair DEBUG cache index')
    parser.add_argument('checkpoint',type=Path,help='Completed typed DEBUG checkpoint.pt')
    parser.add_argument('output',type=Path,help='New output directory')
    args=parser.parse_args()
    from tools.verify_v22_review_selection_debug import main as verify
    return verify(args.cache,args.checkpoint,args.output)


if __name__=='__main__':main()
