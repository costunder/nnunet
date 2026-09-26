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
sys.path.insert(0,str(ROOT))
from tools.v222_resume_guard import assert_source_runs_idle


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


def stages(medical,output,profile_batch,allocator_gb,edge_workspace_mib=64,*,optimized=False,cache=None,resume=None,migrate_workspace=None,process_loader=False,feature_coordinates=None,training_objective=None):
    """Delegate to existing verified entry points without changing model config."""
    run=lambda name,*args:[sys.executable,'-u',str(ROOT/name),*map(str,args)]
    if process_loader and not optimized:
        raise ValueError('Process producer requires optimized execution')
    if feature_coordinates is not None and not process_loader:
        raise ValueError('Feature coordinate contract requires --runtime process')
    if training_objective is not None and not process_loader:
        raise ValueError('Ranking objective requires --runtime process')
    if process_loader and not resume and feature_coordinates=='legacy' and training_objective!='observation_ce':
        raise ValueError('New ranking runs require stride4; legacy coordinates require explicit observation_ce')
    if migrate_workspace is not None and (not optimized or not resume or migrate_workspace!=256):
        raise ValueError('Workspace migration requires optimized resume at 256MiB')
    def gpu(name,*args):
        if process_loader:
            return run('tools/v222_gpu_workspace.py','--workspace-mib',edge_workspace_mib,
                       '--feature-coordinates',feature_coordinates or 'stride4',name,*args)
        if edge_workspace_mib==64:return run(name,*args)
        return run('tools/v222_gpu_workspace.py','--workspace-mib',edge_workspace_mib,name,*args)
    if optimized:
        plan=[('resources',run('tools/v1_server.py','resources','--output',output/'resources.json')),
              ('model_check',run('run_v222_v1_l0.py','check'))]
        if resume and cache is None:raise ValueError('--resume requires the original --cache index')
        if not resume:
            if cache is None:
                plan.append(('observations',run('tools/v1_server.py','observations','--medical-root',medical,'--output',output/'observations')))
            plan.append(('graph_DEBUG',gpu('run_v222_v1_l0.py','verify','--output',output/'graph_DEBUG')))
            if cache is None:
                plan.append(('paired_cache',run('tools/v222_prepare_optimized.py','--index',output/'observations/index.json','--output',output/'paired_cache')))
            index=cache if cache is not None else output/'paired_cache/index.json'
            review_args=['--feature-coordinates',feature_coordinates or 'stride4'] if process_loader else []
            if process_loader and training_objective!='observation_ce':
                plan.append(('profile_DEBUG',run('tools/verify_v22_ranking_real_debug.py',
                    '--cache',index,'--output',output/'profile_DEBUG','--workspace-mib',edge_workspace_mib,
                    *review_args,'--batch-size',profile_batch,'--allocator-gb',allocator_gb)))
            else:
                plan.append(('profile_DEBUG',run('tools/v222_gpu_workspace.py','--workspace-mib',edge_workspace_mib,
                    '--optimized-runtime',*review_args,'tools/profile_v1_execution.py',index,output/'profile_DEBUG','default',
                    '--batch-size',profile_batch,'--allocator-gb',allocator_gb)))
        index=cache if cache is not None else output/'paired_cache/index.json'
        arguments=['--cache',index,'--output',output/'training']
        if resume:arguments+=['--resume',resume] # Inherit saved numerical/allocator policy.
        else:arguments+=['--workspace-mib',edge_workspace_mib]
        if migrate_workspace is not None:arguments+=['--migrate-workspace-mib',migrate_workspace]
        if feature_coordinates is not None:arguments+=['--feature-coordinates',feature_coordinates]
        if training_objective is not None:arguments+=['--training-objective',training_objective]
        entry='tools/run_v222_process_runtime.py' if process_loader else 'tools/run_v222_optimized.py'
        plan.append(('gnn_training',run(entry,*arguments)))
        return plan
    if cache or resume:raise ValueError('Cache reuse/resume is supported by --runtime optimized')
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
            if name=='gnn_training':
                def argument(flag):
                    return command[command.index(flag)+1] if flag in command else None
                assert_source_runs_idle(argument('--cache'),argument('--resume'),output=argument('--output'))
            result=runner(command,cwd=ROOT,env=env,check=False)
        except (OSError,RuntimeError,psutil.Error) as error:
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
    p.add_argument('--runtime',choices=('optimized','process','legacy'),default='process')
    p.add_argument('--training-objective',choices=('observed_rank_v1','observation_ce'),
                   help='New process runs use observed ranking; resume retains saved objective')
    p.add_argument('--feature-coordinates',choices=('legacy','stride4'),
                   help='Process runtime: stride4 for new training, saved coordinate policy for resume')
    p.add_argument('--cache',type=Path,help='Existing complete paired index; skips raw/paired preparation')
    p.add_argument('--resume',type=Path,help='Rolling checkpoint with the same cache; resumes directly')
    p.add_argument('--migrate-workspace-mib',type=int,choices=(256,),help='Resume state with explicitly changed 256MiB execution workspace')
    p.add_argument('--edge-workspace-mib',type=int,choices=(64,256,512),
                   help='Execution scratch budget only; 256 measured locally. Preserve old runs at 64.')
    a=p.parse_args()
    if a.feature_coordinates is not None and a.runtime!='process':raise ValueError('--feature-coordinates requires --runtime process')
    if a.training_objective is not None and a.runtime!='process':raise ValueError('--training-objective requires --runtime process')
    if a.runtime=='process' and not a.resume and a.feature_coordinates=='legacy' and a.training_objective!='observation_ce':
        raise ValueError('New ranking runs require stride4; legacy coordinates require explicit observation_ce')
    if a.migrate_workspace_mib is not None and (not a.resume or a.runtime=='legacy'):
        raise ValueError('--migrate-workspace-mib requires optimized --resume')
    if a.resume and not a.cache:raise ValueError('--resume requires --cache')
    if a.resume and a.edge_workspace_mib is not None:raise ValueError('Resume inherits workspace; omit --edge-workspace-mib')
    if a.edge_workspace_mib is None:a.edge_workspace_mib=256 if a.runtime!='legacy' else 64
    if a.cache:a.cache=a.cache.resolve()
    if a.resume:a.resume=a.resume.resolve()
    for file in (a.cache,a.resume):
        if file is not None and not file.is_file():raise FileNotFoundError(file)
    assert_source_runs_idle(a.cache,a.resume,output=a.output.resolve()/'training')
    if a.profile_batch<1 or a.profile_allocator_gb<=0:raise ValueError('Positive DEBUG profile settings required')
    medical=a.medical_root.resolve();output=a.output.resolve()
    if a.cache is None:
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
                 '--runtime',a.runtime,'--worker']
        if not a.resume:command+=['--edge-workspace-mib',str(a.edge_workspace_mib)]
        if a.cache:command+=['--cache',str(a.cache)]
        if a.resume:command+=['--resume',str(a.resume)]
        if a.migrate_workspace_mib is not None:command+=['--migrate-workspace-mib',str(a.migrate_workspace_mib)]
        if a.feature_coordinates is not None:command+=['--feature-coordinates',a.feature_coordinates]
        if a.training_objective is not None:command+=['--training-objective',a.training_objective]
        launch_worker(command,output)
        return watch(output)
    if (output/'requested.json').exists():
        raise FileExistsError('Existing run cannot be overwritten')
    env=dict(os.environ,HIERCP_TEST_OBSERVATION_INDEX=str(a.cache or output/'observations/index.json'),
             HIERCP_TEST_FIXTURE=str(output/'graph_DEBUG/actual_graphs_DEBUG.pt'))
    plan=stages(medical,output,a.profile_batch,a.profile_allocator_gb,a.edge_workspace_mib,
                optimized=a.runtime!='legacy',cache=a.cache,resume=a.resume,migrate_workspace=a.migrate_workspace_mib,
                process_loader=a.runtime=='process',feature_coordinates=a.feature_coordinates,training_objective=a.training_objective)
    write_new(output/'requested.json',dict(python=sys.executable,medical_root=str(medical),
        cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),conda_prefix=env.get('CONDA_PREFIX'),
        production_model_config_changed=False,production_batch='auto',gnn_epochs=40,
        profile_batch=a.profile_batch,profile_allocator_gb=a.profile_allocator_gb,
        runtime=a.runtime,cache=str(a.cache) if a.cache else None,resume=str(a.resume) if a.resume else None,
        requested_feature_coordinates=a.feature_coordinates,
        requested_training_objective=a.training_objective or ('saved' if a.resume else 'observed_rank_v1' if a.runtime=='process' else 'observation_ce'),
        edge_workspace_mib='inherited from checkpoint' if a.resume else a.edge_workspace_mib,
        explicit_workspace_migration_mib=a.migrate_workspace_mib,
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
