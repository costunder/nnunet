"""Fresh explicit early-stop GPU5 CP/native adapter; no original-source edits."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode = True


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('watch','pin-early-stopped','score','prepare-native','calibrate-native','train','clone-debug-worker'))
    parser.add_argument('--request',type=Path)
    parser.add_argument('--native',type=Path)
    parser.add_argument('--physical-batch',type=int,choices=(2,4))
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    if args.action == 'clone-debug-worker':
        if args.native is None or args.physical_batch is None or args.output is None or args.request is not None:
            parser.error('clone-debug-worker requires only --native/--physical-batch/--output; sealed request is inherited')
    elif args.request is None or any(value is not None for value in (args.native,args.physical_batch,args.output)):
        parser.error('Each production action requires only --request; all science/paths are sealed there')
    return args


def main(argv=None):
    args = parse(argv)
    from hiercp_v1x import v24_early_stop_downstream as adapter
    if args.action == 'watch':
        return adapter.run(args.request)
    if args.action == 'clone-debug-worker':
        return adapter.calibration_worker(args.native,args.physical_batch,args.output)
    return adapter.stage(args.request,args.action)


if __name__ == '__main__':
    main()
