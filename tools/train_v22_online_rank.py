"""Foreground entry point for the separate observed-rank native trainer.

Uses the installed nnU-Net without copying/replacing its package. --check-only
validates paths/catalog only; it does not prove G3/G4 capacity or efficacy.
"""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True,help='Results root; existing folds require explicit --resume')
    parser.add_argument('--workers',type=int,required=True,help='Measured native augmentation worker count')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Resume a verified completed epoch; no fresh fallback')
    parser.add_argument('--checkpoint',type=Path,help='Explicit checkpoint inside this fold; requires --resume')
    args=parser.parse_args()
    if args.workers<2:raise ValueError('Production requires measured parallel workers, not a DEBUG serial setting')
    from tools.v22_online_rank_bank import validate_catalog
    from hiercp_v222.contracts import read_json,validate_native
    meta=validate_catalog(read_json(args.bank));native=validate_native(read_json(meta['native_preparation']))
    name='nnUNetTrainer_250epochs_OnlineRankV22'
    plans=Path(native['plans']);results=args.results.resolve()
    folder=results/native['dataset_name']/f'{name}__{plans.stem}__3d_fullres'/'fold_0'
    if args.checkpoint and not args.resume:raise ValueError('--checkpoint requires --resume')
    if folder.exists() and not args.resume:raise FileExistsError(f'Existing segmentation results preserved: {folder}; use --resume')
    os.environ.update(ONLINE_CP_BANK=str(args.bank.resolve()),ONLINE_CP_SEED='42',COMPARISON_SEED='42',
        nnUNet_raw=str(Path(native['raw']).parent),nnUNet_preprocessed=str(Path(native['preprocessed']).parent),
        nnUNet_results=str(results),nnUNet_n_proc_DA=str(args.workers))
    if not (Path(native['preprocessed'])/'dataset.json').is_file():raise FileNotFoundError('Native dataset.json missing')
    import torch
    from tools.v22_online_runtime import admit_cuda_workspace
    admit_cuda_workspace()
    from tools.v22_online_checkpoint import choose_checkpoint,validate_checkpoint,run_contract,file_hash
    from tools.v22_online_rank_adapter import RankedBank
    selected=None;checkpoint_hash=None;cursor=0
    if args.resume:
        selected=choose_checkpoint(folder,args.checkpoint);checkpoint_hash=file_hash(selected)
        value=torch.load(selected,map_location='cpu',weights_only=False)
        validate_checkpoint(value,run_contract(args.bank,meta,native,args.workers,read_json(plans)))
        cursor=value['current_epoch'];del value
    print(json.dumps(dict(trainer=name,output=str(folder),epochs=250,cp_probability=.8,seed=42,
        workers=args.workers,physical_batch=read_json(plans)['configurations']['3d_fullres']['batch_size'],
        mode='resume' if args.resume else 'fresh',checkpoint=str(selected) if selected else None,
        actual_paste_engine=RankedBank.paste_engine_identity(),
        checkpoint_sha256=checkpoint_hash,completed_epochs=cursor,resume_scope='epoch boundary; interrupted epoch repeats',
        check_only=args.check_only,scope='native execution entry; G3/G4 full-scale acceptance remains required')),flush=True)
    if args.check_only:return
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Verified single allocated CUDA device required')
    from custom_trainers.nnUNetTrainer_OnlineRankV22 import nnUNetTrainer_250epochs_OnlineRankV22
    import nnunetv2.run.run_training as runner
    original=runner.recursive_find_trainer_class_by_name
    original_load=runner.maybe_load_checkpoint
    def resolve(requested):
        return nnUNetTrainer_250epochs_OnlineRankV22 if requested==name else original(requested)
    runner.recursive_find_trainer_class_by_name=resolve
    def load(trainer,continue_training,validation_only,pretrained_weights_file=None):
        if args.resume:
            if not continue_training or validation_only or pretrained_weights_file is not None:raise ValueError('Resume mode changed')
            if file_hash(selected)!=checkpoint_hash:raise ValueError('Selected checkpoint changed after admission')
            trainer.load_checkpoint(selected)
        else:
            if continue_training:raise ValueError('Unexpected continuation')
            original_load(trainer,continue_training,validation_only,pretrained_weights_file)
    runner.maybe_load_checkpoint=load
    try:
        runner.run_training(native['dataset_name'],'3d_fullres',0,trainer_class_name=name,
            plans_identifier=plans.stem,continue_training=args.resume,device=torch.device('cuda'))
    finally:
        runner.recursive_find_trainer_class_by_name=original
        runner.maybe_load_checkpoint=original_load


if __name__=='__main__':main()
