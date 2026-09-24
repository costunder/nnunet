"""Independent v2.2 CT-only CNN L0/L1 CLI. Existing run.py and the v1 feedback launcher are unchanged."""
from pathlib import Path
import argparse
import json

def main():
    from hiercp_v22.contracts import load_config,verify_v1,read_json,write_new,validate_split
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).parent/'config/prompt_graph_v22.json'))
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('check',help='Read-only source/config/dependency/hardware report')
    p=sub.add_parser('import-split'); p.add_argument('--outer-splits',required=True); p.add_argument('--inner-split',required=True)
    p.add_argument('--outer-fold',type=int,required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--medical-root',required=True); p.add_argument('--split',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('train'); p.add_argument('--cache',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('convert-cache',help='Rejected for r3: raw CT rebuild is required')
    p.add_argument('--cache',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('reuse-native',help='Reuse verified v2.1 preprocessing with independent results')
    p.add_argument('--native',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('nnunet-plan'); p.add_argument('--medical-root',required=True); p.add_argument('--split',required=True)
    p.add_argument('--output',required=True); p.add_argument('--dataset-id',type=int,required=True)
    p=sub.add_parser('bank'); p.add_argument('--native',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('nnunet-train'); p.add_argument('--native',required=True); p.add_argument('--bank',required=True)
    p=sub.add_parser('predict'); p.add_argument('--native',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('evaluate'); p.add_argument('--native',required=True); p.add_argument('--predictions',required=True); p.add_argument('--output',required=True)
    args=parser.parse_args(); cfg,base=load_config(args.config); revision=verify_v1()
    if args.command in ('train','bank','nnunet-train','predict','evaluate'):
        from hiercp_v22.contracts import require_training_objective
        require_training_objective()
    if args.command=='check':
        import importlib.util,os,shutil
        report={'v1_revision':revision,'config':cfg,'dependencies':{n:importlib.util.find_spec(n) is not None for n in ('torch','torch_geometric','scipy','nibabel','nnunetv2')},
                'cpu_count':os.cpu_count(),'disk_free_bytes':shutil.disk_usage('.').free,'training_executed':False}
        if report['dependencies']['torch']:
            import torch
            report.update(torch=torch.__version__,cuda_available=torch.cuda.is_available(),cuda_devices=torch.cuda.device_count())
            report['gpu_names']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(json.dumps(report,ensure_ascii=False,indent=2)); return
    if args.command=='import-split':
        outer=read_json(args.outer_splits); inner=read_json(args.inner_split)
        rows=outer['splits'] if isinstance(outer,dict) else outer; chosen=rows[args.outer_fold]
        split={'outer_fold':args.outer_fold,'outer_train':chosen['train'],'outer_val':chosen['val'],
               'inner_train':inner['train'],'inner_val':inner['val']}
        validate_split(split); write_new(args.output,split); print(args.output); return
    if args.command=='convert-cache':
        from hiercp_v22.migration import convert_cache
        result=convert_cache(args.cache,args.output,cfg,base)
    elif args.command=='reuse-native':
        from hiercp_v22.migration import reuse_native
        result=reuse_native(args.native,args.output)
    elif args.command=='prepare':
        from hiercp_v22.data import prepare
        result=prepare(args.medical_root,args.split,args.output,cfg,base)
    elif args.command=='train':
        from hiercp_v22.training import train
        result=train(args.cache,args.output,cfg,base)
    elif args.command=='nnunet-plan':
        from hiercp_v22.nnunet import prepare
        result=prepare(args.medical_root,args.split,args.output,args.dataset_id,cfg)
    elif args.command=='bank':
        from hiercp_v22.bank import build
        result=build(args.native,args.checkpoint,args.output)
    elif args.command=='nnunet-train':
        from hiercp_v22.nnunet import train
        result=train(args.native,args.bank)
    elif args.command=='predict':
        from hiercp_v22.nnunet import predict
        result=predict(args.native,args.output)
    else:
        from hiercp_v22.evaluation import evaluate
        result=evaluate(args.native,args.predictions,args.output)
    print(result)

if __name__=='__main__':
    main()
