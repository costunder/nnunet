"""Foreground fixed-region GNN prepare/train and short local resume smoke.

Full training rejects DEBUG weights/caches. Strict is the default; explicit
research-report records uncalibrated profile violations without promoting them. This entry
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
from l0_regions.profile_policy import POLICIES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','train','smoke'))
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--partition-checkpoint',type=Path)
    p.add_argument('--prepare-batch',type=int)
    p.add_argument('--reg-scale1',type=float)
    p.add_argument('--view-epoch',type=int)
    p.add_argument('--workers',type=int,required=True)
    p.add_argument('--cuda-gib',type=float,required=True);p.add_argument('--rss-gib',type=float,required=True)
    p.add_argument('--resident-gib',type=float)
    p.add_argument('--batch-candidates',type=int,nargs='+')
    p.add_argument('--resume',type=Path)
    p.add_argument('--support-patients',type=int,help='Explicit patient minibatch count; retain every eligible observation of each selected patient')
    p.add_argument('--execution-pipeline',choices=('synchronous','overlapped'),default='synchronous')
    p.add_argument('--device-cache-gib',type=float,help='Explicit verified input GPU cache ceiling; included in --cuda-gib, not an extra allowance')
    p.add_argument('--sage-workspace-mib',type=int,help='Explicit SAGE edge workspace; does not trim rows or graph edges')
    p.add_argument('--resume-support-minibatch',action='store_true',help='Explicit new learning policy from a pinned full-support checkpoint; preserve model/Adam/query cursor/batch')
    p.add_argument('--activation-storage',choices=('checkpointed','retained'),default='checkpointed',
        help='retained: keep CNN/L0/L1/L2 activations rather than recomputing; respect explicit CUDA budget')
    p.add_argument('--resume-execution-upgrade',action='store_true',
        help='Explicit reviewed execution-only continuation; keep saved model/Adam/RNG/support/cursor/batch')
    p.add_argument('--resume-cuda-budget-change',action='store_true',
        help='Explicit GPU migration: change only CUDA allocator budget; preserve saved batch and training state')
    p.add_argument('--reuse-prepared',type=Path,help='Validate and reuse completed preparation batches in a new output directory')
    p.add_argument('--debug',action='store_true')
    p.add_argument('--debug-pause-step',type=int)
    p.add_argument('--profile-policy',choices=POLICIES,default='strict',
        help='research-report: train with recorded uncalibrated profile violations; integrity and resource checks remain mandatory')
    a=p.parse_args()
    if a.execution_pipeline=='overlapped' and (a.device_cache_gib is None or a.device_cache_gib<0):raise ValueError('Explicit nonnegative --device-cache-gib required')
    if a.execution_pipeline=='synchronous' and a.device_cache_gib is not None:raise ValueError('Device cache requires overlapped pipeline')
    if a.execution_pipeline=='overlapped' and (a.sage_workspace_mib is None or a.sage_workspace_mib<=0):raise ValueError('Explicit positive --sage-workspace-mib required')
    if a.execution_pipeline=='synchronous' and a.sage_workspace_mib is not None:raise ValueError('Workspace option requires overlapped pipeline')
    if a.mode!='train' and a.execution_pipeline!='synchronous':raise ValueError('Pipeline options apply to train only')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if a.workers<1:raise ValueError('Positive explicit workers required')
    if a.mode=='smoke' and not a.debug:raise ValueError('Smoke requires explicit --debug')
    if a.mode in ('prepare','smoke'):
        if any(x is None for x in (a.prepare_batch,a.reg_scale1,a.view_epoch)):
            raise ValueError('Explicit preparation batch, scale1 reg and fixed view required')
        if a.reuse_prepared is not None and a.partition_checkpoint is not None:
            raise ValueError('Reuse loads the verified frozen CNN; omit --partition-checkpoint')
        if a.reuse_prepared is None and a.partition_checkpoint is None:
            raise ValueError('New preparation requires an explicit partition checkpoint')
    if a.mode in ('train','smoke') and (not a.batch_candidates or not a.resident_gib or a.resident_gib<=0):
        raise ValueError('Explicit batch candidates and resident tensor budget required')
    if a.mode!='train' and (a.support_patients is not None or a.resume_support_minibatch):raise ValueError('Support learning policy options apply to train only')
    if a.mode!='train' and (a.resume or a.debug_pause_step is not None):raise ValueError('Resume/pause only on train')
    if a.mode!='train' and (a.resume_execution_upgrade or a.resume_cuda_budget_change or a.activation_storage!='checkpointed'):
        raise ValueError('Explicit retained activation/upgrade options apply to train mode only')
    if a.reuse_prepared is not None and a.mode!='prepare':raise ValueError('Prepared-batch reuse only on prepare')
    if a.resume_cuda_budget_change and not (a.resume and a.resume_execution_upgrade):
        raise ValueError('CUDA budget migration requires --resume and --resume-execution-upgrade')
    from tools.v22_region_preflight import check
    check()  # Before any output directory, checkpoint read or graph materialization.
    budget=Budget(int(a.cuda_gib*2**30),int(a.rss_gib*2**30))
    total=torch.cuda.get_device_properties(0).total_memory
    if budget.cuda_bytes>=total:raise ValueError('CUDA allocator budget must leave device headroom')
    previous=torch.cuda.get_per_process_memory_fraction();torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/total)
    torch.set_num_threads(a.workers)
    try:
        if a.mode=='prepare':
            result=prepare_cache(a.cache,a.partition_checkpoint,a.output,batch=a.prepare_batch,workers=a.workers,
                reg1=a.reg_scale1,view_epoch=a.view_epoch,budget=budget,debug=a.debug,profile_policy=a.profile_policy,reuse_prepared=a.reuse_prepared)
        elif a.mode=='train':
            result=train(a.cache,a.output,workers=a.workers,resident_bytes=int(a.resident_gib*2**30),
                candidates=a.batch_candidates,budget=budget,debug=a.debug,resume=a.resume,debug_pause_step=a.debug_pause_step,profile_policy=a.profile_policy,
                activation_storage=a.activation_storage,resume_execution_upgrade=a.resume_execution_upgrade,
                resume_cuda_budget_change=a.resume_cuda_budget_change,support_patients=a.support_patients,
                resume_support_minibatch=a.resume_support_minibatch,execution_pipeline=a.execution_pipeline,
                device_cache_bytes=int((a.device_cache_gib or 0)*2**30),sage_workspace_bytes=(a.sage_workspace_mib or 64)*2**20)
            if (a.output/'training_complete.json').exists():
                from l0_regions.final import export
                export(result,a.cache,a.output/'checkpoint.pt')
        else:
            a.output.mkdir(parents=True,exist_ok=False)
            common=['--workers',str(a.workers),'--cuda-gib',str(a.cuda_gib),'--rss-gib',str(a.rss_gib),'--debug','--profile-policy',a.profile_policy]
            def run(mode,*args):
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),mode,*map(str,args),*common],check=True,cwd=ROOT)
            run('prepare','--cache',a.cache,'--output',a.output/'cache','--partition-checkpoint',a.partition_checkpoint,
                '--prepare-batch',a.prepare_batch,'--reg-scale1',a.reg_scale1,'--view-epoch',a.view_epoch)
            def training(name,*extra):
                run('train','--cache',a.output/'cache/index.json','--output',a.output/name,
                    '--resident-gib',a.resident_gib,'--batch-candidates',*a.batch_candidates,*extra)
            training('uninterrupted')
            training('paused','--debug-pause-step',1)
            training('resumed','--resume',a.output/'paused/checkpoint_latest.pt')
            left=torch.load(a.output/'uninterrupted/checkpoint_latest.pt',map_location='cpu',weights_only=False)
            right=torch.load(a.output/'resumed/checkpoint_latest.pt',map_location='cpu',weights_only=False)
            comparisons={k:hash_state(left[k])==hash_state(right[k]) for k in ('model','optimizer','state','rng')}
            report=dict(debug=True,full_training=False,production_ready=False,profile_policy=a.profile_policy,comparisons=comparisons,
                status='PASS' if all(comparisons.values()) else 'FAIL',
                epochs=left['identity']['epochs'],steps=left['state']['step'],
                scope='Same training entry: full supplied DEBUG train/val, calibration, refresh, validation, best selection, final memory, fresh-process resume')
            write_new(a.output/'smoke_result.json',report)
            if not all(comparisons.values()):raise RuntimeError('Fresh-process resume differs; do not launch full training')
            result=a.output/'smoke_result.json'
        print('RESULT:',result,flush=True)
    except Exception as exc:
        report=getattr(exc,'admission_report',None)
        if report is not None:
            from l0_regions.admission_report import print_rejection
            print_rejection(report)
            if a.output.is_dir() and not (a.output/'admission_rejection.json').exists():
                write_new(a.output/'admission_rejection.json',report)
                print('ADMISSION REPORT:',a.output/'admission_rejection.json',flush=True)
        if a.output.is_dir() and not (a.output/'failed.json').exists():
            write_new(a.output/'failed.json',dict(error_type=type(exc).__name__,error=str(exc),
                mode=a.mode,debug=a.debug,production_ready=False,existing_outputs_preserved=True,
                admission_report=report))
        raise
    finally:torch.cuda.set_per_process_memory_fraction(previous)

if __name__=='__main__':main()
