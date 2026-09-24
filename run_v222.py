"""Independent v2.22 raw-CT spatial GNN and online nnU-Net pipeline."""
import argparse
import json
from pathlib import Path

def main():
    from hiercp_v222.contracts import load_config, verify_v1
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(Path(__file__).parent/'config/prompt_graph_v222.json'))
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    s = sub.add_parser('prepare')
    for name in ('medical-root', 'split', 'identities', 'output'): s.add_argument('--'+name, required=True)
    s = sub.add_parser('train'); s.add_argument('--cache', required=True); s.add_argument('--output', required=True)
    s = sub.add_parser('reuse-native'); s.add_argument('--native', required=True); s.add_argument('--output', required=True)
    s = sub.add_parser('nnunet-plan')
    for name in ('medical-root', 'split', 'identities', 'output'): s.add_argument('--'+name, required=True)
    s.add_argument('--dataset-id', type=int, required=True)
    s = sub.add_parser('bank')
    for name in ('native', 'checkpoint', 'output'): s.add_argument('--'+name, required=True)
    s = sub.add_parser('nnunet-train'); s.add_argument('--native', required=True); s.add_argument('--bank', required=True)
    s.add_argument('--seed',type=int,default=42)
    s.add_argument('--workers',type=int,required=True,help='Measured worker count; use the same count as Basic CP')
    s = sub.add_parser('predict'); s.add_argument('--native', required=True); s.add_argument('--output', required=True)
    s = sub.add_parser('evaluate')
    for name in ('native', 'predictions', 'output'): s.add_argument('--'+name, required=True)
    args = p.parse_args(); cfg, base = load_config(args.config); revision = verify_v1()
    if args.command == 'check':
        import torch
        from hiercp.preparation_runtime import snapshot
        print(json.dumps(dict(config=cfg, base=base, resource_snapshot=snapshot(), v1_revision=revision,
                             cuda=torch.cuda.is_available(), gpu_count=torch.cuda.device_count(),
                             gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                             training_executed=False), indent=2)); return
    if args.command == 'prepare':
        from hiercp_v222.data import prepare
        result = prepare(args.medical_root, args.split, args.identities, args.output, cfg, base)
    elif args.command == 'train':
        from hiercp_v222.training import train
        result = train(args.cache, args.output, cfg, base)
    elif args.command == 'reuse-native':
        from hiercp_v222.migration import reuse_native
        result = reuse_native(args.native, args.output)
    elif args.command == 'nnunet-plan':
        from hiercp_v222.nnunet import prepare
        result = prepare(args.medical_root, args.split, args.output, args.dataset_id, cfg, args.identities)
    elif args.command == 'bank':
        from hiercp_v222.bank import build
        result = build(args.native, args.checkpoint, args.output)
    elif args.command == 'nnunet-train':
        from hiercp_v222.nnunet import train
        result = train(args.native, args.bank,seed=args.seed,workers=args.workers)
    elif args.command == 'predict':
        from hiercp_v222.nnunet import predict
        result = predict(args.native, args.output)
    else:
        from hiercp_v222.evaluation import evaluate
        result = evaluate(args.native, args.predictions, args.output)
    print(result)

if __name__ == '__main__':
    main()
