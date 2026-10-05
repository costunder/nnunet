"""Export exact historical metadata/source for complete v1 to v2.2 diagnosis.

No GPU selection is needed: this command executes no neural model or optimizer.
Only trusted own-run checkpoint metadata is read; CT/masks/weights are not copied.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native-run', 'baseline', 'half-a', 'half-b', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-root', type=Path, action='append', default=[],
                        help='Historical checkout with source matching native checkpoint. Repeat if needed; current checkout is also checked.')
    args = parser.parse_args()
    from hiercp_v1x.native_transition_receipt import export_receipt
    print('READ-ONLY METADATA EXPORT | no neural forward, GPU allocation, or training', flush=True)
    receipt, bundle = export_receipt(args.native_run, args.baseline, args.half_a, args.half_b, args.output,
                                     source_roots=args.source_root)
    print(json.dumps(dict(receipt=str(args.output.resolve() / 'receipt.json'), zip=str(bundle),
                          exact_native_recipe_bound=receipt['exact_native_recipe_bound'],
                          required_fields_missing=receipt['required_fields_missing'],
                          native_records=receipt['inventory'].get('records'),
                          training_started=False), indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
