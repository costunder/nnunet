"""Resume an owned v24 checkpoint with an explicit RSS reclamation overlay.

The original training CLI, scientific identity and checkpoint bytes stay intact.
Preparation copies the admitted artifacts into a fresh result namespace.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--action', choices=('prepare', 'train'), required=True)
    parser.add_argument('--source-output', type=Path, required=True)
    parser.add_argument('--source-code', type=Path, required=True)
    options, original_argv = parser.parse_known_args(argv)
    from tools.run_v24_all_p import parse
    args = parse(original_argv)
    if args.mode != 'train':
        raise ValueError('Continuation preserves the original train execution mode')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if options.action == 'prepare' else str(args.gpu)
    from hiercp_v1x.v24_training_continuation import main as continue_training
    return continue_training(args, source_output=options.source_output,
        source_code=options.source_code, action=options.action)


if __name__ == '__main__':
    main()
