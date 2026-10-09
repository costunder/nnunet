"""Prepare every exact native validation upper graph once, with parallel CPU workers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--arm',choices=('selected','native','native_fixed','native_listwise'),required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--region-cache-source',type=Path,required=True)
    parser.add_argument('--workers',type=int,required=True)
    parser.add_argument('--rss-gib',type=float,required=True)
    args=parser.parse_args()
    if args.workers<2 or args.rss_gib<=0:
        raise ValueError('Parallel CPU workers and explicit positive measured RAM budget required')
    from hiercp_v1x.comparison_native_upper_cache import prepare_cache
    result=prepare_cache(args.experiment,args.arm,args.inventory,args.output,
        region_cache_source=args.region_cache_source,workers=args.workers,rss_gib=args.rss_gib)
    print('ALL21 EXACT ORIGINAL UPPER GRAPHS PREPARED | '+json.dumps(result),flush=True)


if __name__=='__main__':
    main()
