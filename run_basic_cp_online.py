"""Original Basic CP: prepare static inputs, then isolated online nnU-Net training."""
import argparse,os,shutil,subprocess,sys,importlib.util
from pathlib import Path
from hiercp_v2.contracts import write_new,safe_new_root,sha,verify_v1,validate_native,read_json

def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare');prep.add_argument('--native',required=True);prep.add_argument('--output',required=True)
    train=commands.add_parser('train');train.add_argument('--manifest',required=True);train.add_argument('--output',required=True)
    train.add_argument('--workers',type=int,required=True,help='Measured count on this allocated host; no guessed default')
    train.add_argument('--seed',type=int,default=42,help='Shared segmentation comparison seed (v2.22 uses the same option)')
    train.add_argument('--cp-probability',type=float,choices=(0.8,1.0),default=0.8,
        help='User-approved comparison defaults to 0.8; 1.0 selects the preserved original trainer')
    args=parser.parse_args();verify_v1()
    if args.command=='prepare':
        from basic_cp_online.prepare import prepare
        print(prepare(args.native,args.output));return
    if args.workers<1:raise ValueError('Supply a positive, measured worker count')
    if not 0<=args.seed<2**32:raise ValueError('Seed must be a uint32')
    from basic_cp_online.runtime import validate_manifest
    meta=validate_manifest(args.manifest)
    if meta.get('debug'):raise ValueError('DEBUG manifest is not production data')
    native=validate_native(read_json(meta['native_path']));root=safe_new_root(args.output)
    source=Path(importlib.util.find_spec('nnunetv2').origin).parent
    shutil.copytree(source,root/'runtime/nnunetv2',ignore=shutil.ignore_patterns('__pycache__'))
    target=root/'runtime/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_OriginalBasicCPOnline.py'
    shutil.copy2(Path(__file__).parent/'basic_cp_online/nnUNetTrainer_OriginalBasicCPOnline.py',target)
    shutil.copy2(Path(__file__).parent/'basic_cp_online/nnUNetTrainer_BasicCP80Online.py',target.with_name('nnUNetTrainer_BasicCP80Online.py'))
    env=os.environ.copy();env.update(ORIGINAL_BASIC_CP_MANIFEST=str(Path(args.manifest).resolve()),
        nnUNet_raw=str(Path(native['root'])/'nnUNet_raw'),nnUNet_preprocessed=str(Path(native['root'])/'nnUNet_preprocessed'),
        nnUNet_results=str(root/'nnUNet_results'),nnUNet_n_proc_DA=str(args.workers),
        COMPARISON_SEED=str(args.seed),ONLINE_CP_SEED=str(args.seed),
        PYTHONPATH=os.pathsep.join([str(root/'runtime'),str(Path(__file__).resolve().parent)]),PYTHONUNBUFFERED='1')
    trainer='nnUNetTrainer_250epochs_BasicCP80Online' if args.cp_probability==0.8 else 'nnUNetTrainer_250epochs_OriginalBasicCPOnline'
    command=[sys.executable,'-m','nnunetv2.run.run_training',str(native['dataset_id']),'3d_fullres','0',
             '-tr',trainer,'-p',Path(native['plans']).stem]
    from comparison_randomness import FORMAT as RNG_FORMAT
    write_new(root/'started.json',dict(command=command,manifest_sha256=sha(args.manifest),epochs=250,workers=args.workers,
        seed=args.seed,cp_probability=args.cp_probability,trainer=trainer,
        randomness_contract=RNG_FORMAT,physical_batch=read_json(native['plans'])['configurations']['3d_fullres']['batch_size']))
    with (root/'train.log').open('x',encoding='utf-8') as output:result=subprocess.run(command,env=env,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Original Basic CP training failed; see {root / "train.log"}')
    write_new(root/'complete.json',dict(epochs=250,full_medical_training=True,cp_probability=args.cp_probability,
        trainer=trainer,seed=args.seed,log_sha256=sha(root/'train.log')))

if __name__=='__main__':main()
