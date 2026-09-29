"""Short actual-CT DEBUG integration and fresh-process resume, never full training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from l0_regions.training import hash_state,load_checkpoint
from l0_regions.training_data import source_identity,sha,write_new,RegionDataset
from l0_regions.support_episodes import contract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','legacy-checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    ds=RegionDataset(a.cache,'inner_train',True,'research-report')
    old=torch.load(a.legacy_checkpoint,map_location='cpu',weights_only=False)
    if not old['identity']['debug']:raise ValueError('Only a DEBUG legacy checkpoint is allowed')
    identity=old['identity'];limits=identity['resource_limits'];a.output.mkdir(parents=True,exist_ok=False)
    common=['--debug','--profile-policy','research-report','--cache',str(a.cache),
        '--workers',str(identity['workers']),'--cuda-gib',str(limits['cuda_bytes']/2**30),
        '--rss-gib',str(limits['rss_bytes']/2**30),'--resident-gib',str(identity['resident_budget_bytes']/2**30),
        '--batch-candidates',*map(str,identity['candidates']),'--activation-storage',identity['activation_storage'],
        '--support-patients','2']
    def run(name,*extra):
        subprocess.run([sys.executable,'-B','-u','tools/run_fixed_regions.py','train',*common,
            '--output',str(a.output/name),*map(str,extra)],cwd=ROOT,check=True)
    source_hash=sha(a.legacy_checkpoint)
    current=dict(identity,source=source_identity(),support_training=contract(2))
    transitioned=load_checkpoint(a.legacy_checkpoint,current,allow_support_migration=True)
    preserved={k:hash_state(old[k])==hash_state(transitioned[k]) for k in ('model','optimizer','rng')}
    for k in ('step','epoch','next_batch','batch','memory','phase'):
        preserved[k]=hash_state(old['state'][k])==hash_state(transitioned['state'][k])
    if not all(preserved.values()):raise AssertionError('Migration changed the saved state')
    if transitioned['state']['plan'] is not None or transitioned['state']['last_group'] is not None:
        raise AssertionError('Old cluster plan was retained')
    run('uninterrupted');run('paused','--debug-pause-step','1')
    run('resumed','--resume',a.output/'paused/checkpoint_latest.pt')
    run('migrated','--resume',a.legacy_checkpoint,'--resume-support-minibatch',
        '--debug-pause-step',old['state']['step']+1)
    def read(name):return torch.load(a.output/name/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    left,right,migrated=read('uninterrupted'),read('resumed'),read('migrated')
    parity={k:hash_state(left[k])==hash_state(right[k]) for k in ('model','optimizer','state','rng')}
    changed={label:sum(torch.is_tensor(v) and k.startswith(prefix) and not torch.equal(v,old['model'][k])
                       for k,v in migrated['model'].items())
             for label,prefix in [('CNN','local.core.dense_encoder.'),('L0','local.'),('L1','l1.'),('L2','l2.'),('L2_updates','l2_updates.')]}
    if not all(parity.values()) or not all(changed.values()):raise AssertionError('Resume/update verification failed')
    if sha(a.legacy_checkpoint)!=source_hash:raise AssertionError('Legacy checkpoint changed')
    write_new(a.output/'report.json',dict(debug=True,full_training=False,production_ready=False,
        runtime_source=source_identity(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        train_observations=len(ds),validation_observations=len(RegionDataset(a.cache,'inner_val',True,'research-report')),
        physical_batch=left['state']['batch'],steps=left['state']['step'],
        migration_preserved=preserved,resume_parity=parity,changed_parameter_tensors=changed,
        original_checkpoint_preserved=True,cluster_plan_invalidated=True,
        full_debug_memory_refresh_validation_best_selection_and_export=True,
        scope='Actual CT DEBUG only; all supplied debug rows, separate processes, no production training'))
    print('REPORT:',a.output/'report.json')

if __name__=='__main__':main()
