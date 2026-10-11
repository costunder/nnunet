"""Resume original GPU4/5 GNN state in a new append-only execution namespace."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--action', choices=('prepare', 'train'), required=True)
    parser.add_argument('--source-output', type=Path, required=True)
    parser.add_argument('--source-code', type=Path, required=True)
    parser.add_argument('--interruption-proof', type=Path, required=True)
    parser.add_argument('--allocation-guard-module', type=Path)
    parser.add_argument('--allocation-guard-sha256')
    parser.add_argument('--allocation-proof', type=Path)
    parser.add_argument('--allocation-proof-sha256')
    options, remaining = parser.parse_known_args(argv)
    # Select visibility before importing any sealed scientific dependency. A
    # dependency may query CUDA availability while checkpoints are CPU loaded;
    # changing visibility after that query cannot repair the driver's cache.
    early_parser = argparse.ArgumentParser(add_help=False)
    early_parser.add_argument('--gpu', type=int, choices=(4, 5), required=True)
    early_parser.add_argument('--cpu-affinity', type=lambda text: [int(value) for value in text.split(',')], required=True)
    early, _ = early_parser.parse_known_args(remaining)
    if (len(early.cpu_affinity) != 4 or len(set(early.cpu_affinity)) != 4
            or any(value < 0 for value in early.cpu_affinity)):
        raise ValueError('Four explicit assigned CPU IDs required before source activation')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if options.action == 'prepare' else str(early.gpu)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('_v24_interrupted_execution',
        ROOT / 'hiercp_v1x/v24_interrupted_gnn_continuation.py')
    if spec is None or spec.loader is None:
        raise ValueError('Actual append-only continuation helper required')
    helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    helper._context()
    guard_fields = (options.allocation_guard_module, options.allocation_guard_sha256,
                    options.allocation_proof, options.allocation_proof_sha256)
    if options.action == 'prepare':
        if any(value is not None for value in guard_fields):
            raise ValueError('CPU preparation does not admit or borrow a GPU allocation')
        guard = binding = None
    else:
        if any(value is None for value in guard_fields):
            raise ValueError('Actual owned scheduler allocation module/proof/SHA required before GPU train')
        # The owned launcher guard performs its first live check before any
        # scientific imports or checkpoint reads. Visibility alone initializes
        # neither a model nor CUDA and makes no reservation ownership claim.
        guard, binding = helper.load_allocation_guard(*guard_fields, early.gpu, early.cpu_affinity)
    cli, continuation = helper.load_science(options.source_code)
    args = cli.parse(remaining)
    if args.gpu != early.gpu or args.cpu_affinity != early.cpu_affinity:
        raise ValueError('Original source parser and early GPU/CPU identity differ')
    keywords = dict(source_output=options.source_output, source_code=options.source_code,
        interruption_proof=options.interruption_proof, cli=cli, continuation=continuation)
    if options.action == 'prepare':
        result = helper.prepare(args, **keywords)
        print(json.dumps(result, allow_nan=False), flush=True)
        return result
    return helper.train(args, allocation_guard=guard, allocation_binding=binding, **keywords)


if __name__ == '__main__':
    main()
