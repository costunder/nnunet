"""Run the actual full own v2.5 native250/BEST26/metric/schema1 chain."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))


def execution_cpu_affinity(value):
    """Set the declared allowed CPU set before any live native lease exists."""
    import psutil
    requested=[int(item) for item in value.split(',')]
    if len(requested)<4 or len(set(requested))!=len(requested) or any(cpu<0 for cpu in requested):
        raise ValueError('Native execution requires distinct nonnegative CPUs and at least four workers')
    requested=sorted(requested)
    process=psutil.Process()
    if not set(requested)<=set(process.cpu_affinity()):
        raise ValueError('Native execution CPU affinity exceeds actual allowed CPUs')
    process.cpu_affinity(requested)
    if process.cpu_affinity()!=requested:
        raise RuntimeError('Native execution CPU affinity was not applied exactly')
    return requested


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    clone=arguments and arguments[0]=='clone-debug-worker'
    if clone:arguments=arguments[1:]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    if clone:
        parser.add_argument('--native',required=True,type=Path)
        parser.add_argument('--physical-batch',required=True,type=int,choices=(2,4))
    else:
        parser.add_argument('--arm',required=True,choices=('A','B','C','D'))
        parser.add_argument('--gpu',required=True,type=int)
        parser.add_argument('--score-overlay',required=True,type=Path)
        parser.add_argument('--storage-admission',required=True,type=Path)
        parser.add_argument('--project',required=True,type=Path)
        parser.add_argument('--basic-validation',required=True,type=Path)
        parser.add_argument('--cpu-affinity',required=True)
        parser.add_argument('--allocation-request',required=True,type=Path)
        parser.add_argument('--evaluator-root',required=True,type=Path)
    args=parser.parse_args(arguments)
    if clone:
        from hiercp_v1x.v25_native_cp import calibration_worker
        calibration_worker(args.native,args.physical_batch,args.output)
        return 0
    cpu_affinity=execution_cpu_affinity(args.cpu_affinity)
    from hiercp_v1x.v25_downstream import run_downstream
    proof=run_downstream(arm=args.arm,gpu=args.gpu,output=args.output,score_overlay=args.score_overlay,
        storage_admission=args.storage_admission,project=args.project,basic_validation=args.basic_validation,
        cpu_affinity=cpu_affinity,allocation_request=args.allocation_request,
        evaluator_root=args.evaluator_root)
    print(json.dumps(proof,ensure_ascii=False,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
