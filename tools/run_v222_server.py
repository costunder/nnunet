"""Run the documented paired GNN stages with the current Python/MIG environment.

No environment installation, GPU reassignment, data transfer or system changes.
This starts paired GNN training only; native nnU-Net integration is not complete.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def stages(medical,output,profile_batch,allocator_gb):
    """Delegate to existing verified entry points without changing model config."""
    run=lambda name,*args:[sys.executable,'-u',str(ROOT/name),*map(str,args)]
    return [
        ('resources',run('tools/v1_server.py','resources','--output',output/'resources.json')),
        ('model_check',run('run_v222_v1_l0.py','check')),
        ('observations',run('tools/v1_server.py','observations','--medical-root',medical,'--output',output/'observations')),
        ('graph_DEBUG',run('run_v222_v1_l0.py','verify','--output',output/'graph_DEBUG')),
        ('paired_cache',run('run_v222_v1_l0.py','prepare','--index',output/'observations/index.json','--output',output/'paired_cache')),
        ('profile_DEBUG',run('tools/profile_v1_execution.py',output/'paired_cache/index.json',output/'profile_DEBUG',
                            'release_unused','--batch-size',profile_batch,'--allocator-gb',allocator_gb)),
        ('gnn_training',run('run_v222_v1_l0.py','train','--cache',output/'paired_cache/index.json',
                           '--output',output/'training','--release-unused')),
    ]


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False)


def execute(plan,output,env,runner=subprocess.run):
    for number,(name,command) in enumerate(plan):
        started=time.time()
        receipt=dict(stage=name,command=command,started_at=started)
        write_new(output/f'{number:02d}_{name}.started.json',receipt)
        print(json.dumps(dict(event='stage_started',**receipt)),flush=True)
        try:
            result=runner(command,cwd=ROOT,env=env,check=False)
        except OSError as error:
            write_new(output/f'{number:02d}_{name}.failed.json',dict(**receipt,error=str(error)))
            raise
        if result.returncode:
            write_new(output/f'{number:02d}_{name}.failed.json',dict(**receipt,returncode=result.returncode,
                      next_stage_started=False,existing_artifacts_preserved=True))
            raise RuntimeError(f'{name} failed (code {result.returncode}); see the log. Later stages were not started.')
        write_new(output/f'{number:02d}_{name}.complete.json',dict(**receipt,seconds=time.time()-started))
        print(json.dumps(dict(event='stage_complete',stage=name,seconds=time.time()-started)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--medical-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile-batch',type=int,default=32,help='DEBUG only; production batch remains auto')
    p.add_argument('--profile-allocator-gb',type=float,default=9.0,help='DEBUG PyTorch allocator cap only')
    a=p.parse_args()
    if a.profile_batch<1 or a.profile_allocator_gb<=0:raise ValueError('Positive DEBUG profile settings required')
    medical=a.medical_root.resolve();output=a.output.resolve()
    for folder in ('image','labels'):
        if not (medical/'Data'/folder).is_dir():raise FileNotFoundError(medical/'Data'/folder)
    # No duplicate run or replacement of an earlier output, even after failure.
    output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,HIERCP_TEST_OBSERVATION_INDEX=str(output/'observations/index.json'),
             HIERCP_TEST_FIXTURE=str(output/'graph_DEBUG/actual_graphs_DEBUG.pt'))
    plan=stages(medical,output,a.profile_batch,a.profile_allocator_gb)
    write_new(output/'requested.json',dict(python=sys.executable,medical_root=str(medical),
        cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),conda_prefix=env.get('CONDA_PREFIX'),
        production_model_config_changed=False,production_batch='auto',gnn_epochs=40,
        profile_batch=a.profile_batch,profile_allocator_gb=a.profile_allocator_gb,
        scope='paired GNN only; native online bank and nnU-Net not integrated',
        stages=[dict(name=n,command=c) for n,c in plan]))
    execute(plan,output,env)
    marker=output/'training/training_complete.json'
    if not marker.is_file():
        if (output/'training/paused.json').is_file():
            write_new(output/'pipeline_paused.json',dict(gnn_training_complete=False,checkpoint=str(output/'training/checkpoint_latest.pt')))
            print('GNN paused with checkpoint; not complete.',flush=True)
            return
        raise RuntimeError('Training command returned without completion or pause evidence')
    write_new(output/'pipeline_complete.json',dict(gnn_training_complete=True,nnunet_training_complete=False))


if __name__=='__main__':main()
