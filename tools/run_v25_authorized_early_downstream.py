"""Full native250/BEST26 child entry for explicitly authorized partial B GNN."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))


def main(argv=None):
    arguments=list(sys.argv[1:]if argv is None else argv)
    clone=bool(arguments and arguments[0]=='clone-debug-worker')
    if clone:arguments.pop(0)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    if clone:
        parser.add_argument('--native',required=True,type=Path)
        parser.add_argument('--physical-batch',required=True,type=int,choices=(2,4))
    else:
        parser.add_argument('--arm',required=True,choices=('B',))
        parser.add_argument('--gpu',required=True,type=int,choices=(5,))
        for name in ('score-overlay','storage-admission','project','basic-validation','allocation-request','evaluator-root'):
            parser.add_argument('--'+name,required=True,type=Path)
        parser.add_argument('--cpu-affinity',required=True)
    args=parser.parse_args(arguments)
    from hiercp_v1x import v25_authorized_early_downstream as adapter
    adapter.install_from_environment()
    if clone:
        adapter.calibration_worker(args.native,args.physical_batch,args.output)
        return 0
    from tools.run_v25_downstream import execution_cpu_affinity
    cpus=execution_cpu_affinity(args.cpu_affinity)
    from hiercp_v1x import v25_B_early_downstream_resources as resources
    seal=resources.load_seal()
    if (cpus!=seal['slot']['CPU_affinity'] or str(args.output)!=str(Path(seal['group_root'])/'B')
            or str(args.allocation_request)!=seal['slot']['direct_proof']['path']):
        raise ValueError('Exact original B native reserved output/GPU5/CPU/direct proof required')
    proof=adapter.run_downstream(authorization_path=adapter.os.environ[adapter.ENV_AUTHORIZATION],
        arm=args.arm,gpu=args.gpu,output=args.output,score_overlay=args.score_overlay,
        storage_admission=args.storage_admission,project=args.project,basic_validation=args.basic_validation,
        cpu_affinity=cpus,allocation_request=args.allocation_request,evaluator_root=args.evaluator_root)
    print(json.dumps(proof,ensure_ascii=False,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
