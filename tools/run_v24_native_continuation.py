"""Continue a complete owned CP/native bank with explicit constructor compatibility.

This entry point never rebuilds or replaces the saved 105-case CP bank,
private native package, plans, preprocessing, or previous trial outputs.
"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('calibrate','train'),required=True)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--gpu',type=int,choices=(1,),required=True)
    args=parser.parse_args(argv)
    from hiercp_v1x import v24_native_calibration_runtime as runtime
    if args.mode=='calibrate':
        print(runtime.calibrate_native(args.native,gpu=args.gpu),flush=True)
    else:
        print(runtime.train_native(args.native,gpu=args.gpu),flush=True)

if __name__=='__main__': main()
