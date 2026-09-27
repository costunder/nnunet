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
    parser.add_argument('--results',type=Path,required=True,help='New results root; existing fold results are refused')
    parser.add_argument('--workers',type=int,required=True,help='Measured native augmentation worker count')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.workers<2:raise ValueError('Production requires measured parallel workers, not a DEBUG serial setting')
    from tools.v22_online_rank_bank import validate_catalog
    from hiercp_v222.contracts import read_json,validate_native
    meta=validate_catalog(read_json(args.bank));native=validate_native(read_json(meta['native_preparation']))
    name='nnUNetTrainer_250epochs_OnlineRankV22'
    plans=Path(native['plans']);results=args.results.resolve()
    folder=results/native['dataset_name']/f'{name}__{plans.stem}__3d_fullres'/'fold_0'
    if folder.exists():raise FileExistsError(f'Existing segmentation results preserved: {folder}')
    os.environ.update(ONLINE_CP_BANK=str(args.bank.resolve()),ONLINE_CP_SEED='42',COMPARISON_SEED='42',
        nnUNet_raw=str(Path(native['raw']).parent),nnUNet_preprocessed=str(Path(native['preprocessed']).parent),
        nnUNet_results=str(results),nnUNet_n_proc_DA=str(args.workers))
    if not (Path(native['preprocessed'])/'dataset.json').is_file():raise FileNotFoundError('Native dataset.json missing')
    print(json.dumps(dict(trainer=name,output=str(folder),epochs=250,cp_probability=.8,seed=42,
        workers=args.workers,physical_batch=read_json(plans)['configurations']['3d_fullres']['batch_size'],
        check_only=args.check_only,scope='native execution entry; G3/G4 full-scale acceptance remains required')),flush=True)
    if args.check_only:return
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Verified single allocated CUDA device required')
    from custom_trainers.nnUNetTrainer_OnlineRankV22 import nnUNetTrainer_250epochs_OnlineRankV22
    import nnunetv2.run.run_training as runner
    original=runner.recursive_find_trainer_class_by_name
    def resolve(requested):
        return nnUNetTrainer_250epochs_OnlineRankV22 if requested==name else original(requested)
    runner.recursive_find_trainer_class_by_name=resolve
    try:
        runner.run_training(native['dataset_name'],'3d_fullres',0,trainer_class_name=name,
            plans_identifier=plans.stem,device=torch.device('cuda'))
    finally:runner.recursive_find_trainer_class_by_name=original


if __name__=='__main__':main()
