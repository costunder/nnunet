"""Entry point for the user-requested v1-style L0 application.

Includes complete paired-cohort GNN preparation and training. Native online
augmentation/nnU-Net remains a separately identified integration step.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('check')
    verify=sub.add_parser('verify')
    verify.add_argument('--output',required=True)
    verify.add_argument('--check-cp128',action='store_true')
    for name in ('prepare','train','fit'):
        command=sub.add_parser(name)
        command.add_argument('--index' if name!='train' else '--cache',required=True)
        command.add_argument('--output',required=True)
        if name!='train':command.add_argument('--reuse',help='Verified immutable partial paired cache to reuse in a new run')
        if name=='train':
            command.add_argument('--resume',help='Exact mid-epoch checkpoint; writes a new run directory')
            command.add_argument('--calibration',help='Prior identical model/cohort measured execution report directory')
            command.add_argument('--release-unused',action=argparse.BooleanOptionalAction,default=True,
                help='Release unused CUDA allocator cache between full batches (measured default)')
    args=p.parse_args()
    if args.command in ('prepare','train','fit'):
        from hiercp_v222.v1_cache import prepare
        from hiercp_v222.v1_training import train
        import traceback
        output=Path(args.output).resolve()
        try:
            if args.command=='prepare':result=prepare(args.index,output,reuse=args.reuse)
            elif args.command=='train':result=train(args.cache,output,resume=args.resume,
                calibration=args.calibration,release_unused=args.release_unused)
            else:
                output.mkdir(parents=True,exist_ok=False)
                cache=prepare(args.index,output/'cache',reuse=args.reuse)
                result=train(cache,output/'training')
            print(result,flush=True)
        except Exception:
            if output.is_dir():
                with (output/'failure.txt').open('x',encoding='utf-8') as f:f.write(traceback.format_exc())
            raise
        return
    if args.command=='verify':
        from tools.verify_v222_v1_l0 import main as verify_main
        sys.argv=[sys.argv[0],'--output',args.output]+(['--check-cp128'] if args.check_cp128 else [])
        verify_main()
        return
    import torch
    from hiercp_v222.v1_local import model
    from hiercp_v222.contracts import read_json, verify_v1
    root=Path(__file__).resolve().parent
    cfg=read_json(root/'config/prompt_graph_v222_v1_l0.json')
    base=read_json(root/cfg['base_config'])
    with torch.device('meta'):
        net=model(cfg,base)
    print(json.dumps(dict(config=cfg,parameters=sum(p.numel() for p in net.parameters()),
        L0_parameters=sum(p.numel() for p in net.local.parameters()),
        L1_L2_parameters=sum(p.numel() for n,p in net.named_parameters() if not n.startswith('local.')),
        v1_revision=verify_v1(),full_training=False,native_cp_bank_integrated=False),indent=2))


if __name__=='__main__':
    main()
