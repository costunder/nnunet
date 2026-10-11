"""Explicit GPU6 missing-resource-name startup retry; never implicit resume."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('watch', 'entry'))
    parser.add_argument('--request', type=Path, required=True)
    values = list(sys.argv[1:] if argv is None else argv)
    native = []
    if '--' in values:
        boundary = values.index('--'); native = values[boundary+1:]; values = values[:boundary]
    args = parser.parse_args(values)
    if args.action == 'watch' and native:
        parser.error('watch has no native CLI arguments')
    from hiercp_v1x import v24_native_startup_retry as retry
    return retry.entry(args.request, native) if args.action == 'entry' else retry.watch(args.request)


if __name__ == '__main__':
    main()
