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
import psutil

ROOT=Path(__file__).resolve().parents[1]


def launch_worker(command,output):
    """Own a detached worker; the foreground is only an attachable viewer."""
    output.mkdir(parents=True,exist_ok=False)
    options = (dict(creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
               if os.name=='nt' else dict(start_new_session=True))
    with (output/'console.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                               stderr=subprocess.STDOUT,**options)
    write_new(output/'worker.json',dict(pid=child.pid,created_at=psutil.Process(child.pid).create_time(),
                                      command=command,viewer_can_disconnect=True))
    return child


def stages(medical,output,profile_batch,allocator_gb,edge_workspace_mib=64):
    """Delegate to existing verified entry points without changing model config."""
    run=lambda name,*args:[sys.executable,'-u',str(ROOT/name),*map(str,args)]
    def gpu(name,*args):
        if edge_workspace_mib==64:return run(name,*args)
        return run('tools/v222_gpu_workspace.py','--workspace-mib',edge_workspace_mib,name,*args)
    return [
        ('resources',run('tools/v1_server.py','resources','--output',output/'resources.json')),
        ('model_check',run('run_v222_v1_l0.py','check')),
        ('observations',run('tools/v1_server.py','observations','--medical-root',medical,'--output',output/'observations')),
        ('graph_DEBUG',gpu('run_v222_v1_l0.py','verify','--output',output/'graph_DEBUG')),
        ('paired_cache',run('run_v222_v1_l0.py','prepare','--index',output/'observations/index.json','--output',output/'paired_cache')),
        ('profile_DEBUG',gpu('tools/profile_v1_execution.py',output/'paired_cache/index.json',output/'profile_DEBUG',
                            'release_unused','--batch-size',profile_batch,'--allocator-gb',allocator_gb)),
        ('gnn_training',gpu('run_v222_v1_l0.py','train','--cache',output/'paired_cache/index.json',
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
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--edge-workspace-mib',type=int,choices=(64,256,512),default=64,
                   help='Execution scratch budget only; 256 measured locally. Preserve old runs at 64.')
    a=p.parse_args()
    if a.profile_batch<1 or a.profile_allocator_gb<=0:raise ValueError('Positive DEBUG profile settings required')
    medical=a.medical_root.resolve();output=a.output.resolve()
    for folder in ('image','labels'):
        if not (medical/'Data'/folder).is_dir():raise FileNotFoundError(medical/'Data'/folder)
    if not a.worker:
        if not sys.stderr.isatty():
            raise RuntimeError('Run directly in a terminal: logs/background execution are automatic; omit nohup and redirection')
        # Import/check tqdm before starting any background work.
        sys.path.insert(0,str(ROOT))
        from tools.watch_v222_server import watch
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--medical-root',str(medical),
                 '--output',str(output),'--profile-batch',str(a.profile_batch),
                 '--profile-allocator-gb',str(a.profile_allocator_gb),
                 '--edge-workspace-mib',str(a.edge_workspace_mib),'--worker']
        launch_worker(command,output)
        return watch(output)
    if (output/'requested.json').exists():
        raise FileExistsError('Existing run cannot be overwritten')
    env=dict(os.environ,HIERCP_TEST_OBSERVATION_INDEX=str(output/'observations/index.json'),
             HIERCP_TEST_FIXTURE=str(output/'graph_DEBUG/actual_graphs_DEBUG.pt'))
    plan=stages(medical,output,a.profile_batch,a.profile_allocator_gb,a.edge_workspace_mib)
    write_new(output/'requested.json',dict(python=sys.executable,medical_root=str(medical),
        cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),conda_prefix=env.get('CONDA_PREFIX'),
        production_model_config_changed=False,production_batch='auto',gnn_epochs=40,
        profile_batch=a.profile_batch,profile_allocator_gb=a.profile_allocator_gb,
        edge_workspace_mib=a.edge_workspace_mib,
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


if __name__=='__main__':
    raise SystemExit(main())
