"""Explicit GPU1 frozen-best-v23 → real native nnUNet CP stages."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True


def parse(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify','prepare-bank','materialize-bank','prepare-native','calibrate-native','train','predict','evaluate',
        'pin-current-gnn','prepare-current-bank'))
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
    parser.add_argument('--source-output',type=Path)
    parser.add_argument('--source-code',type=Path)
    parser.add_argument('--input-cache',type=Path)
    parser.add_argument('--stunet-checkpoint',type=Path)
    parser.add_argument('--gpu',type=int,choices=(1,5,6),default=1)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if args.gpu!=1 and args.action not in ('pin-current-gnn','prepare-current-bank','prepare-native','calibrate-native','train'):
        parser.error('Historical v23 actions remain explicitly assigned GPU1')
    if args.action in ('pin-current-gnn','prepare-current-bank'):
        if args.gpu not in (5,6):parser.error('Completed current v24 actions require the matching physical GPU5/6')
        if (args.gpu==6)!=(args.stunet_checkpoint is not None):
            parser.error('Only GPU6 requires its exact original --stunet-checkpoint')
    required={'verify':('pin','inventory','baseline_preprocessed'),
        'prepare-bank':('pin','inventory','baseline_preprocessed','output'),
        'materialize-bank':('pin','inventory','baseline_preprocessed','scores','output'),
        'prepare-native':('bank','output'),'calibrate-native':('native',),'train':('native',),
        'predict':('native','inventory','output'),'evaluate':('native','inventory','predictions','output'),
        'pin-current-gnn':('source_output','source_code','inventory','input_cache','output'),
        'prepare-current-bank':('pin','inventory','baseline_preprocessed','input_cache','output')}
    for name in required[args.action]:
        if getattr(args,name) is None:parser.error('--'+name.replace('_','-')+' is required')
    if args.gpu in (5,6) and args.action in ('calibrate-native','train'):
        import json
        from hiercp_v1x.v24_nnunet_cp import CURRENT_FORMAT
        path=args.native
        native=json.loads(path.read_text(encoding='utf8')) if path.is_file() and not path.is_symlink() else {}
        if native.get('format')!=CURRENT_FORMAT or native.get('physical_GPU')!=args.gpu:
            parser.error('GPU5/6 native actions require their explicit completed current-arm native.json; historical arm remains GPU1')
    if args.resume and args.action!='train':parser.error('Explicit native resume applies only to train')
    if args.resume and args.gpu!=1:parser.error('Current automatic downstream native stage starts fresh; explicit recovery uses its own receipt')
    return args


def main(args):
    from hiercp_v1x import v24_nnunet_cp as pipeline
    if args.action=='pin-current-gnn':
        print(pipeline.pin_completed_current(source_output=args.source_output,source_code=args.source_code,
            inventory_path=args.inventory,input_cache=args.input_cache,output=args.output,gpu=args.gpu,
            stunet_checkpoint=args.stunet_checkpoint))
    elif args.action=='prepare-current-bank':
        print(pipeline.prepare_current_bank(pin_path=args.pin,inventory_path=args.inventory,
            baseline_preprocessed=args.baseline_preprocessed,input_cache=args.input_cache,output=args.output,
            gpu=args.gpu,stunet_checkpoint=args.stunet_checkpoint))
    elif args.action=='verify':
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
    elif args.action=='prepare-native':
        pipeline._admit_arm_gpu(pipeline.validate_bank(pipeline.read(args.bank))['pin'],args.gpu)
        print(pipeline.prepare_nnunet(args.bank,args.output))
    elif args.action=='calibrate-native':
        if args.gpu==1:print(pipeline.calibrate_native(args.native,gpu=args.gpu))
        else:
            from hiercp_v1x.v24_native_calibration_runtime import calibrate_native
            print(calibrate_native(args.native,gpu=args.gpu))
    elif args.action=='train':
        if args.gpu==1:print(pipeline.train(args.native,gpu=args.gpu,resume=args.resume))
        else:
            from hiercp_v1x.v24_native_calibration_runtime import train_native
            print(train_native(args.native,gpu=args.gpu))
    elif args.action=='predict':print(pipeline.predict(args.native,inventory_path=args.inventory,output=args.output,gpu=args.gpu))
    else:
        print(pipeline.evaluate(args.native,inventory_path=args.inventory,predictions=args.predictions,
            output=args.output,basic_predictions=args.basic_predictions))


if __name__=='__main__':main(parse())
