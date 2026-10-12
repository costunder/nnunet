"""Sealed GPU4-source native experiment executed on physical GPU0."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('watch','prepare-native','calibrate-native','train','train-entry','clone-debug-worker'))
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--native',type=Path)
    parser.add_argument('--physical-batch',type=int,choices=(2,4))
    parser.add_argument('--output',type=Path)
    values = list(sys.argv[1:] if argv is None else argv); tail = []
    if '--' in values:
        boundary=values.index('--'); tail=values[boundary+1:]; values=values[:boundary]
    args = parser.parse_args(values)
    if args.action != 'train-entry' and tail:
        parser.error('Only train-entry accepts unchanged native CLI arguments')
    if args.action != 'clone-debug-worker' and any(value is not None for value in (args.native,args.physical_batch,args.output)):
        parser.error('DEBUG clone parameters apply only to clone-debug-worker')
    from hiercp_v1x import v24_gpu0_native_execution as execution
    if args.native is not None and str(args.native) != execution.read(args.request)['native_path']:
        raise ValueError('Clone native metadata path differs from sealed request')
    return execution.dispatch(args.request,args.request_sha256,args.action,native_argv=tail,
        physical_batch=args.physical_batch,output=args.output)


if __name__ == '__main__':
    main()
