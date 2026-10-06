"""v2.2 native local CNN L0; foreground, explicit FOV and resource limits."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from l0_local_cnn.data import prepare,Dataset
from l0_local_cnn.model import validate_config
from l0_regions.training_data import Budget
from l0_regions.training import train

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','train','run'],help='run: create metadata inventory then train; never prepares a graph')
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--margin-mm',type=float,required=True,help='Physical margin on EACH face outside donor tumor bbox; not full cube width')
    p.add_argument('--config',type=Path,default=ROOT/'config/v22_local_cnn.json')
    p.add_argument('--workers',type=int)
    p.add_argument('--cuda-gib',type=float);p.add_argument('--rss-gib',type=float);p.add_argument('--resident-gib',type=float)
    p.add_argument('--batch-candidates',nargs='+',type=int)
    p.add_argument('--support-patients',type=int)
    p.add_argument('--device-cache-gib',type=float,default=0)
    p.add_argument('--resume',type=Path)
    p.add_argument('--curriculum-config',type=Path,help='Explicit cumulative U16 TRAIN schedule; full128 validation/support remain fixed')
    p.add_argument('--debug',action='store_true')
    p.add_argument('--debug-pause-step',type=int)
    a=p.parse_args();cfg=json.loads(a.config.read_text());cfg['margin_mm']=a.margin_mm;validate_config(cfg)
    curriculum=None
    if a.curriculum_config is not None:
        from l0_regions.candidate_curriculum import validate_config as validate_curriculum
        curriculum=validate_curriculum(json.loads(a.curriculum_config.read_text(encoding='utf8')))
    if a.mode=='run':
        if a.resume:raise ValueError('Use train with the existing inventory for exact resume')
        root=a.output
        if root.exists():raise FileExistsError('Choose a new experiment output directory')
        a.cache=prepare(a.cache,cfg,root/'inventory',debug=a.debug)
        a.output=root/'training';a.mode='train'
    if a.mode=='prepare':
        if a.resume or a.debug_pause_step is not None:raise ValueError('Resume applies to training only')
        result=prepare(a.cache,cfg,a.output,debug=a.debug)
    else:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        if a.workers is None or a.workers<2 or not a.batch_candidates or any(v is None or v<=0 for v in (a.cuda_gib,a.rss_gib,a.resident_gib)):
            raise ValueError('Explicit parallel workers, resource limits and batch candidates required')
        if a.support_patients is None:raise ValueError('Explicit existing support patient count required')
        ds=Dataset(a.cache,'inner_train',a.debug)
        if ds.meta['local_cnn']!=cfg:raise ValueError('CLI FOV/config differs from inventory: create a separate preparation for this FOV')
        if a.resident_gib>=a.rss_gib or not 0<=a.device_cache_gib<a.cuda_gib:raise ValueError('Cache budgets must leave workspace headroom')
        budget=Budget(int(a.cuda_gib*2**30),int(a.rss_gib*2**30));total=torch.cuda.get_device_properties(0).total_memory
        if budget.cuda_bytes>=total:raise ValueError('CUDA budget must leave device headroom')
        previous=torch.cuda.get_per_process_memory_fraction();torch.cuda.set_per_process_memory_fraction(budget.cuda_bytes/total)
        try:
            result=train(a.cache,a.output,workers=a.workers,resident_bytes=int(a.resident_gib*2**30),candidates=a.batch_candidates,
                budget=budget,debug=a.debug,resume=a.resume,debug_pause_step=a.debug_pause_step,activation_storage='retained',
                execution_pipeline='overlapped',device_cache_bytes=int(a.device_cache_gib*2**30),support_patients=a.support_patients,
                learning_policy='same_donor_live_v1',local_cnn=True,curriculum_config=curriculum)
            if (a.output/'training_complete.json').exists():
                from l0_local_cnn.artifact import export
                result=export(result,a.cache,a.output/'checkpoint.pt')
        finally:torch.cuda.set_per_process_memory_fraction(previous)
    print('RESULT:',result,flush=True)
if __name__=='__main__':main()
