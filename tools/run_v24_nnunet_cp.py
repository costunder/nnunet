"""Explicit GPU1 frozen-best-v23 → real native nnUNet CP stages."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True


def parse(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify','prepare-bank','materialize-bank','prepare-native','calibrate-native','train','predict','evaluate'))
    parser.add_argument('--pin',type=Path)
    parser.add_argument('--inventory',type=Path)
    parser.add_argument('--baseline-preprocessed',type=Path)
    parser.add_argument('--scores',type=Path)
    parser.add_argument('--score-source-code',type=Path)
    parser.add_argument('--bank',type=Path)
    parser.add_argument('--native',type=Path)
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--basic-predictions',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if args.gpu!=1:parser.error('This downstream arm is explicitly assigned GPU1')
    required={'verify':('pin','inventory','baseline_preprocessed'),
        'prepare-bank':('pin','inventory','baseline_preprocessed','output'),
        'materialize-bank':('pin','inventory','baseline_preprocessed','scores','output'),
        'prepare-native':('bank','output'),'calibrate-native':('native',),'train':('native',),
        'predict':('native','inventory','output'),'evaluate':('native','inventory','predictions','output')}
    for name in required[args.action]:
        if getattr(args,name) is None:parser.error('--'+name.replace('_','-')+' is required')
    if args.resume and args.action!='train':parser.error('Explicit native resume applies only to train')
    return args


def main(args):
    from hiercp_v1x import v24_nnunet_cp as pipeline
    if args.action=='verify':
        pin,request=pipeline.admit_pin(args.pin,args.inventory)
        baseline=pipeline.validate_baseline(args.baseline_preprocessed,pipeline.read(args.inventory)['split'])
        print(dict(actual_best=pin['selected'],baseline=baseline,actual_native_GNN=request['inventory_sha256']))
    elif args.action=='prepare-bank':
        print(pipeline.prepare_bank(pin_path=args.pin,inventory_path=args.inventory,
            baseline_preprocessed=args.baseline_preprocessed,output=args.output,gpu=args.gpu))
    elif args.action=='materialize-bank':
        print(pipeline.materialize_bank(pin_path=args.pin,inventory_path=args.inventory,
            baseline_preprocessed=args.baseline_preprocessed,scores=args.scores,output=args.output,
            score_source_code=args.score_source_code,gpu=args.gpu))
    elif args.action=='prepare-native':print(pipeline.prepare_nnunet(args.bank,args.output))
    elif args.action=='calibrate-native':print(pipeline.calibrate_native(args.native,gpu=args.gpu))
    elif args.action=='train':print(pipeline.train(args.native,gpu=args.gpu,resume=args.resume))
    elif args.action=='predict':print(pipeline.predict(args.native,inventory_path=args.inventory,output=args.output,gpu=args.gpu))
    else:
        print(pipeline.evaluate(args.native,inventory_path=args.inventory,predictions=args.predictions,
            output=args.output,basic_predictions=args.basic_predictions))


if __name__=='__main__':main(parse())
