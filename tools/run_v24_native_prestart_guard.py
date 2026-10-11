"""Run an explicitly sealed existing-CP/fixed-native adoption supervisor."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('watch',))
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args(argv)
    from hiercp_v1x.v24_native_prestart_guard import watch
    return watch(args.request)


if __name__=='__main__':main()
