"""Print the server's bound v1/v2.2 settings and preserve full metadata locally.

No GPU selection is needed: this command executes no neural model or optimizer.
Only trusted own-run checkpoint metadata is read; CT/masks/weights are not copied.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native-run', 'baseline', 'half-a', 'half-b', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-root', type=Path, action='append', default=[],
                        help='Historical checkout with source matching native checkpoint. Repeat if needed; current checkout is also checked.')
    parser.add_argument('--archive', action='store_true',
                        help='Optional local archive; the default creates no ZIP and prints a terminal summary.')
    args = parser.parse_args(argv)
    from hiercp_v1x.native_transition_receipt import export_receipt
    from hiercp_v1x.transition_terminal_report import build_terminal_summary, render_terminal_summary
    summary_path = args.output.with_name(args.output.name + '.terminal_summary.json').resolve()
    if summary_path.exists():
        raise FileExistsError(f'Previous terminal summary preserved: {summary_path}')
    print('READ-ONLY METADATA EXPORT | no neural forward, GPU allocation, or training', flush=True)
    receipt, bundle = export_receipt(args.native_run, args.baseline, args.half_a, args.half_b, args.output,
                                     source_roots=args.source_root, create_archive=args.archive)
    summary = build_terminal_summary(receipt, args.output)
    summary['full_summary_path'] = str(summary_path)
    # Keep this convenience report outside the immutable receipt manifest.
    with summary_path.open('x', encoding='utf8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(render_terminal_summary(summary), flush=True)
    print(f'SERVER SUMMARY JSON: {summary_path}', flush=True)
    print(f'FULL SERVER METADATA: {args.output.resolve() / "receipt.json"}', flush=True)
    if bundle is not None:
        print(f'OPTIONAL LOCAL ARCHIVE: {bundle}', flush=True)


if __name__ == '__main__':
    main()
