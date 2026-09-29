"""Foreground fixed-region GNN prepare/train and short local resume smoke.

Full training rejects DEBUG weights/caches and all admission failures. This entry
does not launch nnU-Net or alter Basic CP. SIGINT saves at the next batch boundary.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from l0_regions.training_data import Budget,prepare_cache,write_new
from l0_regions.training import train,hash_state

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','train','smoke'))
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--partition-checkpoint',type=Path)
    p.add_argument('--prepare-batch',type=int)
    p.add_argument('--reg-scale1',type=float);p.add_argument('--reg-scale2',type=float)
    p.add_argument('--view-epoch',type=int)
    p.add_argument('--workers',type=int,required=True)
    p.add_argument('--cuda-gib',type=float,required=True);p.add_argument('--rss-gib',type=float,required=True)
    p.add_argument('--resident-gib',type=float)
    p.add_argument('--batch-candidates',type=int,nargs='+')
    p.add_argument('--resume',type=Path)
    p.add_argument('--debug',action='store_true')
    p.add_argument('--debug-pause-step',type=int)
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if a.workers<1:raise ValueError('Positive explicit workers required')
    if a.mode=='smoke' and not a.debug:raise ValueError('Smoke requires explicit --debug')
    if a.mode in ('prepare','smoke') and any(x is None for x in (a.partition_checkpoint,a.prepare_batch,a.reg_scale1,a.reg_scale2,a.view_epoch)):
        raise ValueError('Explicit CNN snapshot, preparation batch, both reg values and fixed view required')
    if a.mode in ('train','smoke') and (not a.batch_candidates or not a.resident_gib or a.resident_gib<=0):
        raise ValueError('Explicit batch candidates and resident tensor budget required')
    if a.mode!='train' and (a.resume or a.debug_pause_step is not None):raise ValueError('Resume/pause only on train')
    budget=Budget(int(a.cuda_gib*2**30),int(a.rss_gib*2**30))
    total=torch.cuda.get_device_properties(0).total_memory
    if budget.cuda_bytes>=total:raise ValueError('CUDA allocator budget must leave device headroom')
    previous=torch.cuda.get_per_process_memory_fraction();torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/total)
    torch.set_num_threads(a.workers)
    try:
        if a.mode=='prepare':
            result=prepare_cache(a.cache,a.partition_checkpoint,a.output,batch=a.prepare_batch,workers=a.workers,
                reg1=a.reg_scale1,reg2=a.reg_scale2,view_epoch=a.view_epoch,budget=budget,debug=a.debug)
        elif a.mode=='train':
            result=train(a.cache,a.output,workers=a.workers,resident_bytes=int(a.resident_gib*2**30),
                candidates=a.batch_candidates,budget=budget,debug=a.debug,resume=a.resume,debug_pause_step=a.debug_pause_step)
            if (a.output/'training_complete.json').exists():
                from l0_regions.final import export
                export(result,a.cache,a.output/'checkpoint.pt')
        else:
            a.output.mkdir(parents=True,exist_ok=False)
            common=['--workers',str(a.workers),'--cuda-gib',str(a.cuda_gib),'--rss-gib',str(a.rss_gib),'--debug']
            def run(mode,*args):
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),mode,*map(str,args),*common],check=True,cwd=ROOT)
            run('prepare','--cache',a.cache,'--output',a.output/'cache','--partition-checkpoint',a.partition_checkpoint,
                '--prepare-batch',a.prepare_batch,'--reg-scale1',a.reg_scale1,'--reg-scale2',a.reg_scale2,'--view-epoch',a.view_epoch)
            def training(name,*extra):
                run('train','--cache',a.output/'cache/index.json','--output',a.output/name,
                    '--resident-gib',a.resident_gib,'--batch-candidates',*a.batch_candidates,*extra)
            training('uninterrupted')
            training('paused','--debug-pause-step',1)
            training('resumed','--resume',a.output/'paused/checkpoint_latest.pt')
            left=torch.load(a.output/'uninterrupted/checkpoint_latest.pt',map_location='cpu',weights_only=False)
            right=torch.load(a.output/'resumed/checkpoint_latest.pt',map_location='cpu',weights_only=False)
            comparisons={k:hash_state(left[k])==hash_state(right[k]) for k in ('model','optimizer','state','rng')}
            report=dict(debug=True,full_training=False,production_ready=False,comparisons=comparisons,
                status='PASS' if all(comparisons.values()) else 'FAIL',
                epochs=left['identity']['epochs'],steps=left['state']['step'],
                scope='Same training entry: full supplied DEBUG train/val, calibration, refresh, validation, best selection, final memory, fresh-process resume')
            write_new(a.output/'smoke_result.json',report)
            if not all(comparisons.values()):raise RuntimeError('Fresh-process resume differs; do not launch full training')
            result=a.output/'smoke_result.json'
        print('RESULT:',result,flush=True)
    except Exception as exc:
        if a.output.is_dir() and not (a.output/'failed.json').exists():
            write_new(a.output/'failed.json',dict(error_type=type(exc).__name__,error=str(exc),
                mode=a.mode,debug=a.debug,production_ready=False,existing_outputs_preserved=True))
        raise
    finally:torch.cuda.set_per_process_memory_fraction(previous)

if __name__=='__main__':main()
